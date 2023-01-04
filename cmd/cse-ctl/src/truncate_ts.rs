// Copyright 2022 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    collections::HashMap, path::PathBuf, str::FromStr, sync::mpsc::SyncSender, time::Duration,
};

use clap::Args;
use http::{Request, Uri};
use hyper::Body;
use kvengine::{EngineStats, ShardTruncateTsStats};
use kvproto::metapb::Store;
use pd_client::PdClient;
use security::SecurityConfig;
use slog_global::{error, info};
use tikv_client::transaction::{Client as TiKVClient, ResolveLocksOptions};
use tikv_util::time::Instant;
use tokio::runtime::Runtime;

use crate::common::{create_pd_client, get_all_stores_except_tiflash, send_request_to_store};

const DEFAULT_TRUNCATE_TS_TIMEOUT: u64 = 5 * 60; // 5 min
const MAX_WAIT_TRUNCATE_TS_CNT: usize = 10;
const TRUNCATE_TS_QUERY_INTERVAL: Duration = Duration::from_secs(10);

#[derive(Args)]
pub struct TruncateTsArgs {
    /// The path of the config file.
    #[clap(long, default_value = "")]
    pub config: PathBuf,
    /// The truncate ts
    #[clap(long)]
    pub truncate_ts: u64,
    /// The timeout in seconds
    #[clap(long, default_value_t = DEFAULT_TRUNCATE_TS_TIMEOUT)]
    pub timeout: u64,
    /// PD endpoints, use `,` to separate multiple PDs
    #[clap(long, default_value_t = String::new())]
    pub pd: String,
    /// Path of file that contains list of trusted SSL CAs
    #[clap(long, default_value = "")]
    pub cacert: PathBuf,
    /// Path of file that contains X509 certificate in PEM format
    #[clap(long, default_value = "")]
    pub cert: PathBuf,
    /// Path of file that contains X509 key in PEM format
    #[clap(long, default_value = "")]
    pub key: PathBuf,
}

pub(crate) fn execute_truncate_ts(args: TruncateTsArgs) {
    let timeout = Duration::from_secs(args.timeout);
    let config = get_truncate_ts_config_from_args(&args);
    let pd_client = create_pd_client(&config.security, &config.pd);
    let truncate_ts = args.truncate_ts;
    let cluster_id = pd_client.get_cluster_id().unwrap();
    info!("Cluster {} truncate ts {}.", cluster_id, truncate_ts,);

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(8)
        .enable_all()
        .build()
        .unwrap();

    if let Err(e) = runtime.block_on(resolve_async_commit_locks(&config)) {
        error!("resolve_async_commit_locks error: {:?}", e);
        return;
    }

    let start = Instant::now();
    while Instant::now().duration_since(start) < timeout {
        let stores = get_all_stores_except_tiflash(&pd_client).unwrap();
        let mut remain_stores = HashMap::new();
        for store in stores {
            remain_stores.insert(store.id, store);
        }
        match request_truncate_ts_on_all_stores(cluster_id, &remain_stores, truncate_ts, &runtime) {
            Ok(shard_cnt) => {
                if shard_cnt == 0 {
                    info!(
                        "Current max ts is already smaller than truncate ts {:?}",
                        truncate_ts
                    );
                    return;
                }
                info!(
                    "Send requests to all stores to truncate ts {:?} succeed, {:?} shards is executing truncate ts",
                    truncate_ts, shard_cnt
                );
            }
            Err(e) => {
                error!("Request to truncate ts failed {:?}", e);
                return;
            }
        }

        if let Err(e) = wait_truncate_ts_finish(
            &mut remain_stores,
            truncate_ts,
            &runtime,
            TRUNCATE_TS_QUERY_INTERVAL,
        ) {
            error!("Fail to wait truncate ts finish {:?}", e);
        }

        if remain_stores.is_empty() {
            info!("All stores truncate ts to {:?} succeed", truncate_ts);
            return;
        }
    }
    error!("Check truncate ts timeout.");
}

// send truncate ts request and return the count of shard which execute truncate ts.
fn request_truncate_ts_on_all_stores(
    cluster_id: u64,
    stores: &HashMap<u64, Store>,
    truncate_ts: u64,
    runtime: &Runtime,
) -> Result<usize, String> {
    let mut shard_cnt = 0;
    let store_cnt = stores.len();
    let (tx, rx) = std::sync::mpsc::sync_channel(store_cnt);
    for store in stores.values() {
        runtime.spawn(request_truncate_ts_store(
            cluster_id,
            store.clone(),
            truncate_ts,
            tx.clone(),
        ));
    }
    let mut errs = vec![];
    for _ in 0..store_cnt {
        match rx.recv().unwrap() {
            Ok((store_id, resp)) => {
                info!(
                    "Request store {:?} to truncate ts succeed, {:?}",
                    store_id, resp
                );
                shard_cnt += resp.len();
            }
            Err(err) => {
                errs.push(err);
            }
        }
    }
    if !errs.is_empty() {
        return Err(errs.join(";"));
    }
    Ok(shard_cnt)
}

async fn request_truncate_ts_store(
    cluster_id: u64,
    store: Store,
    truncate_ts: u64,
    tx: SyncSender<Result<(u64, Vec<ShardTruncateTsStats>), String>>,
) {
    let uri = Uri::from_str(&format!("http://{}/truncate-ts", &store.status_address)).unwrap();
    let store_id = store.get_id();
    let mut body_map = HashMap::new();
    body_map.insert("cluster_id".to_string(), cluster_id.to_string());
    body_map.insert("truncate_ts".to_string(), truncate_ts.to_string());
    let json_string = serde_json::to_string(&body_map).unwrap();
    let req = Request::post(uri).body(Body::from(json_string)).unwrap();
    match send_request_to_store(req, store).await {
        Ok(resp) => {
            let resp: Vec<ShardTruncateTsStats> = serde_json::from_slice(&resp.to_vec()).unwrap();
            tx.send(Ok((store_id, resp))).unwrap()
        }
        Err(e) => tx
            .send(Err(format!("Store {:?} failed, {:?}", store_id, e)))
            .unwrap(),
    }
}

// query max ts on all stores and remove the finished store in stores.
fn wait_truncate_ts_finish(
    stores: &mut HashMap<u64, Store>,
    truncate_ts: u64,
    runtime: &Runtime,
    interval: Duration,
) -> Result<(), String> {
    for _ in 0..MAX_WAIT_TRUNCATE_TS_CNT {
        // wait a while for truncate finish.
        std::thread::sleep(interval);

        let cnt = stores.len();
        let (tx, rx) = std::sync::mpsc::sync_channel(cnt);
        for (_, store) in stores.clone() {
            runtime.spawn(query_max_ts_store(store, tx.clone()));
        }
        let mut errs = vec![];
        for _ in 0..cnt {
            match rx.recv().unwrap() {
                Ok((store_id, max_ts)) => {
                    if max_ts <= truncate_ts {
                        stores.remove(&store_id);
                        info!(
                            "Store {:?} truncate to ts {:?} succeed, cur max ts {:?}",
                            store_id, truncate_ts, max_ts
                        )
                    } else {
                        info!(
                            "Store {:?}'s max ts {:?} is still larger than truncate ts {:?}",
                            store_id, max_ts, truncate_ts
                        )
                    }
                }
                Err(err) => {
                    error!("Query store max ts failed, {:?}", err);
                    errs.push(err);
                }
            }
        }
        if !errs.is_empty() {
            return Err(errs.join(";"));
        }
        if stores.is_empty() {
            return Ok(());
        }
    }
    Ok(())
}

async fn query_max_ts_store(store: Store, tx: SyncSender<Result<(u64, u64), String>>) {
    let uri = Uri::from_str(&format!("http://{}/kvengine", &store.status_address)).unwrap();
    let store_id = store.get_id();
    let client = hyper::Client::new();
    match client.get(uri).await {
        Ok(resp) => {
            let body = hyper::body::to_bytes(resp.into_body()).await.unwrap();
            let engine_stats: EngineStats = serde_json::from_slice(&body.to_vec()).unwrap();
            tx.send(Ok((store_id, engine_stats.max_ts))).unwrap()
        }
        Err(e) => tx
            .send(Err(format!("Store {:?} failed, {:?}", store_id, e)))
            .unwrap(),
    }
}

async fn resolve_async_commit_locks(config: &TruncateTsConfig) -> tikv_client::Result<()> {
    let tikv_client_config = if config.security.ca_path.is_empty() {
        tikv_client::Config::default()
    } else {
        tikv_client::Config::default().with_security(
            config.security.ca_path.clone(),
            config.security.cert_path.clone(),
            config.security.key_path.clone(),
        )
    };
    let tikv_client = TiKVClient::new_with_config(
        config.pd.endpoints.clone(),
        tikv_client_config,
        Some(slog_global::get_global().new(slog::o!())),
    )
    .await?;

    let safepoint = tikv_client.current_timestamp().await?;
    let options = ResolveLocksOptions {
        async_commit_only: true,
        ..Default::default()
    };
    let result = tikv_client.cleanup_locks(&safepoint, options).await?;
    info!(
        "resolve_async_commit_locks succeed, meet locks: {}",
        result.meet_locks
    );
    Ok(())
}

#[derive(Clone, Serialize, Deserialize, PartialEq, Debug, Default)]
#[serde(default)]
#[serde(rename_all = "kebab-case")]
pub struct TruncateTsConfig {
    pub pd: pd_client::Config,
    pub security: SecurityConfig,
}

fn get_truncate_ts_config_from_args(args: &TruncateTsArgs) -> TruncateTsConfig {
    let mut config = TruncateTsConfig::default();
    if args.config.exists() {
        let data = std::fs::read(args.config.clone()).expect("failed to read config file");
        config = toml::from_slice(&data).unwrap();
    }
    // override from args
    if !args.pd.is_empty() {
        config.pd.endpoints = args.pd.split(",").map(|x| x.to_owned()).collect();
    }
    if args.cacert.exists() {
        config.security.ca_path = args.cacert.to_str().unwrap().to_owned();
    }
    if args.cert.exists() {
        config.security.cert_path = args.cert.to_str().unwrap().to_owned();
    }
    if args.key.exists() {
        config.security.key_path = args.key.to_str().unwrap().to_owned();
    }
    config
}
