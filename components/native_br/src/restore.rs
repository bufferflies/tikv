// Copyright 2023 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    path::{Path, PathBuf},
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use cloud_server::TikvServer;
use etcd_client::{Compare, CompareOp, Txn, TxnOp};
use kvengine::dfs::{DFSConfig, Dfs, S3Fs};
use pd_client::PdClient;
use protobuf::Message;
use rfenginepb::ClusterBackupMeta;
use security::{GetSecurityManager, SecurityConfig};
use slog_global::info;
use tikv::config::TikvConfig;
use tikv_util::config::ensure_dir_exist;

use crate::{
    backup::backup_file_full_path,
    common::{generate_etcd_connect_opt, replay_wal_logs},
    error::{Error, Result},
};

const PD_ROOT_PATH: &str = "/pd";
const PD_CLUSTER_ID_PATH: &str = "/pd/cluster_id";
const MAX_TXN_OPTS: usize = 128; // Default configuration in etcd server.

struct MockPdClient {}

impl PdClient for MockPdClient {}

impl GetSecurityManager for MockPdClient {}

pub fn restore_tikv(config: &RestoreConfig, name: String, store_id: u64, path: &str) {
    let dfs_conf = config.dfs.clone();
    let s3fs = S3Fs::new(
        dfs_conf.prefix,
        dfs_conf.s3_endpoint,
        dfs_conf.s3_key_id,
        dfs_conf.s3_secret_key,
        dfs_conf.s3_region,
        dfs_conf.s3_bucket,
    );
    let cluster_backup = get_cluster_backup_meta(&s3fs, name);
    let is_lightweight = cluster_backup.get_is_lightweight();
    if store_id > 0 {
        if is_lightweight {
            let truncate_ts = cluster_backup.backup_ts;
            info!(
                "start replay wal for store {} with truncate_ts {}",
                store_id, truncate_ts
            );
            let tikv_conf = generate_store_config(path);
            setup_raft_engine(store_id, &cluster_backup, &tikv_conf, Arc::new(s3fs)).unwrap();
        } else {
            rfengine::restore(
                Arc::new(s3fs),
                &cluster_backup,
                store_id,
                &PathBuf::from(path),
                None,
            );
        }
    }
}

fn generate_store_config(path: &str) -> TikvConfig {
    ensure_dir_exist(path).unwrap();

    let mut config = TikvConfig::default();
    config.raft_store.raftdb_path = path.to_string();
    config.raft_engine.enable = false;
    config.rfengine.lightweight_backup = false;
    config
}

fn setup_raft_engine(
    store_id: u64,
    cluster_backup: &ClusterBackupMeta,
    conf: &TikvConfig,
    dfs: Arc<S3Fs>,
) -> Result<()> {
    let snap_epoch = rfengine::lightweight_restore(
        dfs.clone(),
        &dfs.get_prefix(),
        cluster_backup,
        store_id,
        Path::new(&conf.raft_store.raftdb_path),
        None,
    )
    .map_err(|x| Error::RfEngine(x))?;

    let rf_engine = TikvServer::init_raft_engine(conf)?;

    // `snap_epoch` is the latest snapshot manifest epoch. If no snapshot found,
    // the `snap_epoch` is 0. Replay wal logs from `snap_epoch` + 1 to backup point.
    replay_wal_logs(
        Arc::new(MockPdClient {}),
        dfs,
        store_id,
        cluster_backup,
        &rf_engine,
        snap_epoch,
        true,
    )?;
    Ok(())
}

pub fn restore_pd(config: RestoreConfig, name: String) {
    let dfs_conf = config.dfs.clone();
    let s3fs = S3Fs::new(
        dfs_conf.prefix,
        dfs_conf.s3_endpoint,
        dfs_conf.s3_key_id,
        dfs_conf.s3_secret_key,
        dfs_conf.s3_region,
        dfs_conf.s3_bucket,
    );
    let cluster_backup = get_cluster_backup_meta(&s3fs, name);
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(restore_pd_keyspace_meta(&config, &cluster_backup));
}

pub fn get_cluster_backup_meta(s3fs: &S3Fs, name: String) -> ClusterBackupMeta {
    let backup_key = backup_file_full_path(s3fs.get_prefix(), name.clone(), None);
    let runtime = s3fs.get_runtime();
    let data = runtime
        .block_on(s3fs.get_object(backup_key, name, engine_traits::GetObjectOptions::default()))
        .unwrap();
    let mut cluster_backup = ClusterBackupMeta::new();
    cluster_backup.merge_from_bytes(&data).unwrap();
    info!(
        "get_cluster_backup_meta: cluster_id {}, alloc_id {}, backup_ts {}, safe_ts {}, store cnt {}",
        cluster_backup.cluster_id,
        cluster_backup.alloc_id,
        cluster_backup.backup_ts,
        cluster_backup.safe_ts,
        cluster_backup.stores.len()
    );
    cluster_backup
}

// Mainly ref `recoverFromNewPDCluster` in `pd-recover`.
async fn restore_pd_keyspace_meta(config: &RestoreConfig, meta: &ClusterBackupMeta) {
    let option = generate_etcd_connect_opt(&config.security).unwrap();
    let mut etcd_client = etcd_client::Client::connect(&config.pd.endpoints, Some(option))
        .await
        .unwrap();
    let mut txn_opts = Vec::with_capacity(4);
    let root_path = format!("{}/{}", PD_ROOT_PATH, meta.cluster_id);
    // recover cluster_id
    txn_opts.push(TxnOp::put(
        PD_CLUSTER_ID_PATH.as_bytes().to_vec(),
        meta.cluster_id.to_be_bytes().to_vec(),
        None,
    ));
    // recover alloc id
    let alloc_id_path = format!("{}/{}", root_path, "alloc_id");
    txn_opts.push(TxnOp::put(
        alloc_id_path.as_bytes().to_vec(),
        meta.alloc_id.to_be_bytes().to_vec(),
        None,
    ));
    // recover meta of cluster
    let cluster_raft_path = format!("{}/{}", root_path, "raft");
    let cluster_meta = kvproto::metapb::Cluster {
        id: meta.cluster_id,
        ..Default::default()
    };
    txn_opts.push(TxnOp::put(
        cluster_raft_path.as_bytes().to_vec(),
        cluster_meta.write_to_bytes().unwrap(),
        None,
    ));
    // set raft bootstrap time
    let raft_bootstrap_time_path = format!(
        "{}/{}/{}",
        cluster_raft_path, "status", "raft_bootstrap_time"
    );
    let cur_nano = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64;
    txn_opts.push(TxnOp::put(
        raft_bootstrap_time_path.as_bytes().to_vec(),
        cur_nano.to_be_bytes().to_vec(),
        None,
    ));
    let resp = etcd_client
        .txn(
            Txn::new()
                .when(
                    &[Compare::create_revision(
                        cluster_raft_path,
                        CompareOp::Equal,
                        0,
                    )][..],
                )
                .and_then(txn_opts),
        )
        .await
        .unwrap();
    if !resp.succeeded() {
        panic!(
            "Failed to restore pd keyspace meta, please JUST start new pd-server(s) without tikv nodes."
        );
    }
    // recover key space meta
    let meta_cnt = meta.keyspace_meta.len();
    let mut txn_opts = Vec::with_capacity(std::cmp::min(meta_cnt, MAX_TXN_OPTS));
    let mut idx = 0;
    for (key, value) in &meta.keyspace_meta {
        txn_opts.push(TxnOp::put(key.to_owned(), value.to_owned(), None));
        idx += 1;
        if txn_opts.len() == MAX_TXN_OPTS || idx == meta_cnt {
            // There is no batch put interface now, use txn to do batch.
            let batch_cnt = txn_opts.len();
            let resp = etcd_client
                .txn(Txn::new().and_then(txn_opts))
                .await
                .unwrap();
            if !resp.succeeded() {
                panic!(
                    "Fail to restore pd meta, cur idx {} batch {} total {}",
                    idx - batch_cnt,
                    batch_cnt,
                    meta_cnt
                );
            }
            txn_opts = Vec::with_capacity(std::cmp::min(meta_cnt - idx, MAX_TXN_OPTS));
        }
    }
    info!(
        "Restore PD {} meta data(revision: {}) of cluster {} successfully",
        meta_cnt, meta.meta_revision, meta.cluster_id
    );
}

#[derive(Clone, Serialize, Deserialize, PartialEq, Debug, Default)]
#[serde(default)]
#[serde(rename_all = "kebab-case")]
pub struct RestoreConfig {
    pub pd: pd_client::Config,
    pub security: SecurityConfig,
    pub dfs: DFSConfig,
    pub skip_resolve_lock: bool,
}
