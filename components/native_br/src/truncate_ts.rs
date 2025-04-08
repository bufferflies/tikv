// Copyright 2023 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    collections::HashMap,
    sync::{mpsc::SyncSender, Arc},
    time::Duration,
};

use api_version::ApiV2;
use http::Request;
use hyper::Body;
use kvengine::{EngineStats, ShardStats, ShardTruncateTsStats};
use kvproto::metapb::Store;
use pd_client::PdClient;
use security::{SecurityConfig, SecurityManager};
use slog_global::{error, info};
use tikv_client::transaction::{Client as TiKVClient, ResolveLocksOptions};
use tikv_util::time::Instant;
use tokio::runtime::Runtime;

use crate::{
    common::{get_all_stores_except_tiflash, send_request_to_store_with_retry},
    error::Error,
};

const MAX_WAIT_TRUNCATE_TS_CNT: usize = 10;
const TRUNCATE_TS_QUERY_INTERVAL: Duration = Duration::from_secs(10);
const TRUNCATE_TS_STORE_TIMEOUT: Duration = Duration::from_secs(30);

pub type Result<T> = std::result::Result<T, Error>;

pub fn truncate_ts_with_cfg(
    config: TruncateTsConfig,
    pd_client: Arc<dyn PdClient>,
    truncate_ts: u64,
    timeout: Duration,
    keyspace_id: Option<u32>,
) -> Result<()> {
    let cluster_id = pd_client.get_cluster_id()?;
    info!("Cluster {} begin truncate ts {}.", cluster_id, truncate_ts);

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(8)
        .enable_all()
        .build()
        .unwrap();

    let range = keyspace_id.map(|id| ApiV2::get_keyspace_range_by_id(id));
    if !config.skip_resolve_lock {
        if let Err(e) = runtime.block_on(resolve_async_commit_locks(&config, range.clone())) {
            error!("resolve_async_commit_locks error: {:?}", e);
            return Err(e);
        }
    }

    let security_mgr = pd_client.get_security_mgr();
    let start = Instant::now();
    while Instant::now().duration_since(start) < timeout {
        let stores = get_all_stores_except_tiflash(pd_client.as_ref())?;
        let mut remain_stores = HashMap::new();
        for store in stores {
            remain_stores.insert(store.id, store);
        }
        match request_truncate_ts_on_all_stores(
            cluster_id,
            &remain_stores,
            truncate_ts,
            range.clone(),
            &runtime,
            security_mgr.clone(),
        ) {
            Ok(shard_cnt) => {
                if shard_cnt == 0 {
                    info!(
                        "Current max ts is already smaller than truncate ts {:?}",
                        truncate_ts
                    );
                    return Ok(());
                }
                info!(
                    "Send requests to all stores to truncate ts {:?} succeed, {:?} shards is executing truncate ts",
                    truncate_ts, shard_cnt
                );
            }
            Err(e) => {
                error!("Request to truncate ts failed {:?}", e);
                return Err(e);
            }
        }

        if let Err(e) = wait_truncate_ts_finish(
            &mut remain_stores,
            truncate_ts,
            keyspace_id,
            &runtime,
            TRUNCATE_TS_QUERY_INTERVAL,
            security_mgr.clone(),
        ) {
            error!("Fail to wait truncate ts finish {:?}", e);
        }

        if remain_stores.is_empty() {
            info!("All stores truncate ts to {:?} succeed", truncate_ts);
            return Ok(());
        }
    }
    Err(Error::Timeout(
        "Wait truncate ts".to_string(),
        timeout.as_secs(),
    ))
}

// send truncate ts request and return the count of shard which execute truncate
// ts.
fn request_truncate_ts_on_all_stores(
    cluster_id: u64,
    stores: &HashMap<u64, Store>,
    truncate_ts: u64,
    range: Option<(Vec<u8>, Vec<u8>)>,
    runtime: &Runtime,
    security_mgr: Arc<SecurityManager>,
) -> Result<usize> {
    let mut shard_cnt = 0;
    let store_cnt = stores.len();
    let (tx, rx) = std::sync::mpsc::sync_channel(store_cnt);
    for store in stores.values() {
        runtime.spawn(request_truncate_ts_store(
            cluster_id,
            store.clone(),
            truncate_ts,
            range.clone(),
            tx.clone(),
            security_mgr.clone(),
        ));
    }
    let mut last_err = None;
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
                error!("Truncate ts on store failed, {:?}", err);
                last_err = Some(err);
            }
        }
    }
    if let Some(e) = last_err {
        return Err(e);
    }
    Ok(shard_cnt)
}

async fn request_truncate_ts_store(
    cluster_id: u64,
    store: Store,
    truncate_ts: u64,
    range: Option<(Vec<u8>, Vec<u8>)>,
    tx: SyncSender<Result<(u64, Vec<ShardTruncateTsStats>)>>,
    security_mgr: Arc<SecurityManager>,
) {
    let uri = security_mgr
        .build_uri(format!("{}/truncate-ts", &store.status_address))
        .unwrap();
    let store_id = store.get_id();
    let config = cloud_server::TruncateTsConfig {
        cluster_id,
        ts: truncate_ts,
        range,
    };
    let json_string = serde_json::to_string(&config).unwrap();
    let req = || {
        Request::post(uri.clone())
            .body(Body::from(json_string.clone()))
            .unwrap()
    };
    match send_request_to_store_with_retry(
        req,
        &store,
        security_mgr.as_ref(),
        TRUNCATE_TS_STORE_TIMEOUT,
    )
    .await
    {
        Ok(resp) => {
            let resp: Vec<ShardTruncateTsStats> = serde_json::from_slice(&resp).unwrap();
            tx.send(Ok((store_id, resp))).unwrap()
        }
        Err(e) => tx
            .send(Err(Error::ServerError(format!(
                "Store {:?} failed, {:?}",
                store_id, e
            ))))
            .unwrap(),
    }
}

// query max ts on all stores and remove the finished store in stores.
fn wait_truncate_ts_finish(
    stores: &mut HashMap<u64, Store>,
    truncate_ts: u64,
    keyspace_id: Option<u32>,
    runtime: &Runtime,
    interval: Duration,
    security_mgr: Arc<SecurityManager>,
) -> Result<()> {
    for _ in 0..MAX_WAIT_TRUNCATE_TS_CNT {
        // wait a while for truncate finish.
        std::thread::sleep(interval);

        let cnt = stores.len();
        let (tx, rx) = std::sync::mpsc::sync_channel(cnt);
        for (_, store) in stores.clone() {
            runtime.spawn(query_max_ts_store(
                store,
                keyspace_id,
                tx.clone(),
                security_mgr.clone(),
            ));
        }
        let mut last_err = None;
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
                    last_err = Some(err);
                }
            }
        }
        if let Some(e) = last_err {
            return Err(e);
        }
        if stores.is_empty() {
            return Ok(());
        }
    }
    Ok(())
}

async fn query_max_ts_store(
    store: Store,
    keyspace_id: Option<u32>,
    tx: SyncSender<Result<(u64, u64)>>,
    security_mgr: Arc<SecurityManager>,
) {
    let uri = match keyspace_id {
        Some(id) => security_mgr
            .build_uri(format!(
                "{}/kvengine/keyspace/{}",
                &store.status_address, id
            ))
            .unwrap(),
        None => security_mgr
            .build_uri(format!("{}/kvengine", &store.status_address))
            .unwrap(),
    };
    let store_id = store.get_id();
    let client = security_mgr.http_client(hyper::Client::builder()).unwrap();
    match client.get(uri).await {
        Ok(resp) => {
            let body = hyper::body::to_bytes(resp.into_body()).await.unwrap();
            match keyspace_id {
                Some(_) => {
                    let shard_stats: Vec<ShardStats> = serde_json::from_slice(&body).unwrap();
                    let max_ts = shard_stats.iter().map(|s| s.max_ts).max().unwrap_or(0);
                    tx.send(Ok((store_id, max_ts))).unwrap();
                }
                None => {
                    let engine_stats: EngineStats = serde_json::from_slice(&body).unwrap();
                    tx.send(Ok((store_id, engine_stats.max_ts))).unwrap();
                }
            }
        }
        Err(e) => tx
            .send(Err(Error::ServerError(format!(
                "Store {:?} failed, {:?}",
                store_id, e
            ))))
            .unwrap(),
    }
}

async fn resolve_async_commit_locks(
    config: &TruncateTsConfig,
    range: Option<(Vec<u8>, Vec<u8>)>,
) -> Result<()> {
    let tikv_client_config = if config.security.ca_path.is_empty() {
        tikv_client::Config::default()
    } else {
        tikv_client::Config::default().with_security(
            config.security.ca_path.clone(),
            config.security.cert_path.clone(),
            config.security.key_path.clone(),
        )
    };
    let tikv_client =
        TiKVClient::new_with_config(config.pd.endpoints.clone(), tikv_client_config).await?;

    let safepoint = tikv_client.current_timestamp().await?;
    let options = ResolveLocksOptions {
        async_commit_only: true,
        ..Default::default()
    };
    let result = tikv_client
        .cleanup_locks(range.unwrap_or_default(), &safepoint, options)
        .await?;
    info!(
        "resolve_async_commit_locks succeed, resolved locks: {}",
        result.resolved_locks
    );
    Ok(())
}

#[derive(Clone, Serialize, Deserialize, PartialEq, Debug, Default)]
#[serde(default)]
#[serde(rename_all = "kebab-case")]
pub struct TruncateTsConfig {
    pub pd: pd_client::Config,
    pub security: SecurityConfig,
    pub skip_resolve_lock: bool,
}
