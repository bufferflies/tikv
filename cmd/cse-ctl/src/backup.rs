// Copyright 2022 TiKV Project Authors. Licensed under Apache-2.0.

use std::{collections::HashMap, path::PathBuf, str::FromStr, sync::mpsc::SyncSender};

use bytes::Bytes;
use clap::Args;
use futures::executor::block_on;
use http::{Request, Uri};
use hyper::Body;
use kvengine::dfs::{DFSConfig, S3FS};
use kvproto::metapb::Store;
use pd_client::PdClient;
use protobuf::Message;
use rfenginepb::{ClusterBackupMeta, StoreBackupMeta};
use security::SecurityConfig;
use slog_global::{error, info};

use crate::common::{create_pd_client, send_request_to_store};

#[derive(Args)]
pub struct BackupArgs {
    /// The path of the config file.
    #[clap(long)]
    pub config: PathBuf,
    /// The name of the backup file, if empty, a system generated name will be used.
    #[clap(long)]
    pub name: String,
}

pub(crate) fn execute_backup(args: BackupArgs) {
    let result = std::fs::read(args.config);
    if result.is_err() {
        error!("failed to read config file {:?}", result.unwrap_err());
        return;
    }
    let data = result.unwrap();
    let config: BackupConfig = toml::from_slice(&data).unwrap();
    let pd_client = create_pd_client(&config.security, &config.pd);
    let stores = pd_client.get_all_stores(true).unwrap();
    let backup_ts = block_on(pd_client.get_tso()).unwrap().into_inner();
    let cluster_id = pd_client.get_cluster_id().unwrap();
    let mut cluster_backup_meta = ClusterBackupMeta::new();
    cluster_backup_meta.set_backup_ts(backup_ts);
    cluster_backup_meta.set_cluster_id(cluster_id);
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(8)
        .enable_all()
        .build()
        .unwrap();
    let num_stores = stores.len();
    let (tx, rx) = std::sync::mpsc::sync_channel(num_stores);
    for store in stores {
        let tx = tx.clone();
        let uri =
            Uri::from_str(&format!("http://{}/rfengine/backup", &store.status_address)).unwrap();
        let mut body_map = HashMap::new();
        body_map.insert("cluster_id".to_string(), cluster_id.to_string());
        let json_string = serde_json::to_string(&body_map).unwrap();
        let req = Request::post(uri).body(Body::from(json_string)).unwrap();
        runtime.spawn(request_backup_store(req, store, tx));
    }
    let mut errs = vec![];
    for _ in 0..num_stores {
        match rx.recv().unwrap() {
            Ok(store_backup_meta) => {
                cluster_backup_meta.mut_stores().push(store_backup_meta);
            }
            Err(err) => {
                errs.push(err);
            }
        }
    }
    if errs.len() > 0 {
        error!("backup errors {:?}", errs);
        if errs.len() > config.tolerate_err {
            return;
        }
    }
    let alloc_id = pd_client.alloc_id().unwrap();
    let safe_ts = runtime.block_on(pd_client.get_gc_safe_point()).unwrap();
    if safe_ts > backup_ts {
        error!(
            "safe ts {} is greater than backup ts {}",
            safe_ts, backup_ts
        );
        return;
    }
    cluster_backup_meta.set_alloc_id(alloc_id);
    cluster_backup_meta.set_safe_ts(safe_ts);
    info!(
        "cluster backup cluster_id:{}, backup_ts:{}, alloc_id:{}, safe_ts:{}, num_stores:{}",
        cluster_backup_meta.cluster_id,
        cluster_backup_meta.backup_ts,
        cluster_backup_meta.alloc_id,
        cluster_backup_meta.safe_ts,
        cluster_backup_meta.get_stores().len(),
    );
    let dfs_conf = config.dfs.clone();
    let s3fs = S3FS::new(
        dfs_conf.prefix,
        dfs_conf.s3_endpoint,
        dfs_conf.s3_key_id,
        dfs_conf.s3_secret_key,
        dfs_conf.s3_region,
        dfs_conf.s3_bucket,
    );
    let backup_key = if args.name.is_empty() {
        format!("{}/backup/{}.meta", config.dfs.prefix, backup_ts)
    } else {
        format!("{}/backup/{}", config.dfs.prefix, &args.name)
    };
    let backup_data = Bytes::from(cluster_backup_meta.write_to_bytes().unwrap());
    runtime
        .block_on(s3fs.put_object(backup_key.clone(), backup_data, backup_key.clone()))
        .unwrap();
    info!("finished build backup file {}", backup_key);
}

pub(crate) async fn request_backup_store(
    req: Request<Body>,
    store: Store,
    tx: SyncSender<Result<StoreBackupMeta, String>>,
) {
    match send_request_to_store(req, store).await {
        Ok(resp) => {
            let mut store_backup_meta = StoreBackupMeta::default();
            store_backup_meta.merge_from_bytes(&resp).unwrap();
            tx.send(Ok(store_backup_meta)).unwrap()
        }
        Err(e) => tx.send(Err(e)).unwrap(),
    }
}

#[derive(Clone, Serialize, Deserialize, PartialEq, Debug, Default)]
#[serde(default)]
#[serde(rename_all = "kebab-case")]
pub struct BackupConfig {
    pub pd: pd_client::Config,
    pub security: SecurityConfig,
    pub dfs: DFSConfig,
    pub tolerate_err: usize,
}
