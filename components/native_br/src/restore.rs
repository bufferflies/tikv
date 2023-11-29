// Copyright 2023 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    path::{Path, PathBuf},
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use cloud_server::TikvServer;
use etcd_client::{Compare, CompareOp, Txn, TxnOp};
use kvengine::dfs::{DFSConfig, Dfs, S3Fs};
use kvproto::raft_serverpb::StoreIdent;
use pd_client::PdClient;
use protobuf::Message;
use rfengine::{load_store_ident, region_state_key, RfEngine, KV_ENGINE_META_KEY, STORE_IDENT_KEY};
use rfenginepb::{ClusterBackupMeta, StoreBackupMeta};
use rfstore::store::load_region_state;
use security::{GetSecurityManager, SecurityConfig};
use tikv::config::TikvConfig;
use tikv_util::{
    config::{ensure_dir_exist, ReadableSize},
    debug, info, warn,
};

use crate::{
    backup::backup_file_full_path,
    common::{
        check_store_id_exists, generate_etcd_connect_opt, get_latest_backup_meta, replay_wal_logs,
    },
    error::{Error, Result},
};

const PD_ROOT_PATH: &str = "/pd";
const PD_CLUSTER_ID_PATH: &str = "/pd/cluster_id";
const MAX_TXN_OPTS: usize = 128; // Default configuration in etcd server.

struct MockPdClient {}

impl PdClient for MockPdClient {}

impl GetSecurityManager for MockPdClient {}

// Generate new store id by the ordered index of the store.
//
// Even we use the alloc_id in latest cluster backup meta as the base, it still
// has conflict possibility if new store added to the cluster after the backup.
// If conflict happens, specify the `delta` to avoid conflict. `alloc_id_base =
// alloc_id + delta`.
fn get_new_store_id(alloc_id_base: u64, stores: &[StoreBackupMeta], store_id: u64) -> u64 {
    let mut store_ids = stores
        .iter()
        .map(|store| store.store_id)
        .collect::<Vec<_>>();
    store_ids.sort_unstable();
    alloc_id_base + store_ids.into_iter().position(|id| id == store_id).unwrap() as u64 + 1
}

pub fn restore_tikv(
    config: &RestoreConfig,
    name: String,
    store_id: u64,
    new_store_id_delta: u64,
    path: &str,
) {
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
    let s3fs_clone = s3fs.clone();
    // Get the alloc_id in the cluster latest backup meta.
    let latest_cluster_backup = s3fs_clone
        .get_runtime()
        .block_on(get_latest_backup_meta(
            &s3fs_clone,
            cluster_backup.cluster_id,
        ))
        .unwrap_or_else(|_| cluster_backup.clone());
    let alloc_id = latest_cluster_backup.alloc_id + new_store_id_delta;

    // Pre-check all new store ids to avoid conflicts.
    cluster_backup.get_stores().iter().for_each(|store| {
        let new_store_id = get_new_store_id(alloc_id, cluster_backup.get_stores(), store.store_id);
        info!(
            "check new store id {} for store id {}",
            new_store_id, store.store_id
        );
        match check_store_id_exists(&s3fs_clone, new_store_id) {
            Ok(false) => {}
            Ok(true) => {
                panic!("new store id {} already exists", new_store_id);
            }
            Err(e) => {
                panic!("check store id failed: {}", e);
            }
        }
    });
    let is_lightweight = cluster_backup.get_is_lightweight();
    if store_id > 0 {
        let tikv_conf = generate_store_config(path, config.wal_target_size);
        if is_lightweight {
            let truncate_ts = cluster_backup.backup_ts;
            info!(
                "start replay wal for store {} with truncate_ts {}",
                store_id, truncate_ts
            );
        }
        setup_raft_engine(
            store_id,
            alloc_id,
            &cluster_backup,
            is_lightweight,
            &tikv_conf,
            path,
            Arc::new(s3fs),
        )
        .unwrap();

        if new_store_id_delta > 0 {
            info!(
                "!!!ATTENTION!!! new_store_id_delta {}, please also specify this in restore pd.",
                new_store_id_delta
            );
        }
    }
}

fn put_store_ident(rf: &RfEngine, ident: &StoreIdent) {
    let val = ident.write_to_bytes().unwrap();
    let mut wb = rfengine::WriteBatch::new();
    wb.set_state(0, 0, STORE_IDENT_KEY, val.as_slice());
    rf.write(wb).unwrap();
}

// Update all peers' store id in local region state to `new_store_id_base +
// old_store_id`. We should guarantee to use the same `new_store_id_base` for
// all full restores in other nodes.
fn update_local_region_state_store_id(
    rf: &RfEngine,
    cluster_backup: &ClusterBackupMeta,
    alloc_id: u64,
) {
    let mut wb = rfengine::WriteBatch::new();
    let region_to_peers = rf.get_region_peer_map();
    for (region_id, peer_id) in region_to_peers {
        debug!("region: {} peer: {}", region_id, peer_id);
        if let Some(val) = rf.get_state(peer_id, KV_ENGINE_META_KEY) {
            let mut cs = kvenginepb::ChangeSet::default();
            cs.merge_from_bytes(&val).unwrap();
            let mut region_local_state = load_region_state(rf, peer_id, cs.shard_ver).unwrap();
            let peers = region_local_state.mut_region().mut_peers();

            // Remove learner peer if exists to remove TiFlash replica.
            if let Some(pos) = peers
                .iter()
                .position(|peer| peer.get_role() == kvproto::metapb::PeerRole::Learner)
            {
                peers.remove(pos);
            }

            for peer in peers.iter_mut() {
                let new_store_id =
                    get_new_store_id(alloc_id, cluster_backup.get_stores(), peer.get_store_id());
                debug!(
                    "update peer {} store id from {} to {}",
                    peer.get_id(),
                    peer.get_store_id(),
                    new_store_id
                );
                peer.set_store_id(new_store_id);
            }

            let region_state_key = region_state_key(cs.shard_ver);
            let region_state_val = region_local_state.write_to_bytes().unwrap();
            wb.set_state(
                peer_id,
                cs.shard_id,
                region_state_key.as_ref(),
                region_state_val.as_slice(),
            );
        } else {
            warn!(
                "region: {} peer: {} has no raft local state",
                region_id, peer_id
            );
        }
    }
    if !wb.is_empty() {
        rf.write(wb).unwrap();
    }
}

fn setup_raft_engine_new_store_id(
    rf: &RfEngine,
    cluster_backup: &ClusterBackupMeta,
    store_id: u64,
    alloc_id: u64,
) {
    // Update all peers' store id in local region state.
    update_local_region_state_store_id(rf, cluster_backup, alloc_id);

    // Update store id in raft engine.
    if let Some(mut store_ident) = load_store_ident(rf) {
        let stores = cluster_backup.get_stores();
        let new_store_id = get_new_store_id(alloc_id, stores, store_id);
        info!(
            "replace store id {} with new store id {}",
            store_ident.get_store_id(),
            new_store_id
        );
        store_ident.set_store_id(new_store_id);
        put_store_ident(rf, &store_ident);
    };
}

fn generate_store_config(path: &str, wal_target_size: ReadableSize) -> TikvConfig {
    ensure_dir_exist(path).unwrap();

    let mut config = TikvConfig::default();
    config.raft_store.raftdb_path = path.to_string();
    config.raft_engine.enable = false;
    config.rfengine.lightweight_backup = false;
    config.rfengine.target_file_size = wal_target_size;
    config
}

fn setup_raft_engine(
    store_id: u64,
    alloc_id: u64,
    cluster_backup: &ClusterBackupMeta,
    lightweight: bool,
    conf: &TikvConfig,
    path: &str,
    dfs: Arc<S3Fs>,
) -> Result<()> {
    let snap_epoch = if lightweight {
        Some(
            rfengine::lightweight_restore(
                dfs.clone(),
                &dfs.get_prefix(),
                cluster_backup,
                store_id,
                Path::new(&conf.raft_store.raftdb_path),
                None,
            )
            .map_err(|x| Error::RfEngine(x))?,
        )
    } else {
        rfengine::restore(
            dfs.clone(),
            cluster_backup,
            store_id,
            &PathBuf::from(path),
            None,
        );
        None
    };

    let rf_engine = TikvServer::init_raft_engine(conf)?;
    rf_engine.set_engine_id(store_id);

    if lightweight {
        // `snap_epoch` is the latest snapshot manifest epoch. If no snapshot found,
        // the `snap_epoch` is 0. Replay wal logs from `snap_epoch` + 1 to backup point.
        replay_wal_logs(
            Arc::new(MockPdClient {}),
            dfs,
            store_id,
            cluster_backup,
            &rf_engine,
            snap_epoch.unwrap(),
            true,
        )?;
    }
    setup_raft_engine_new_store_id(&rf_engine, cluster_backup, store_id, alloc_id);

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
    // Get the alloc_id in the cluster latest backup meta.
    let latest_cluster_backup = s3fs
        .get_runtime()
        .block_on(get_latest_backup_meta(&s3fs, cluster_backup.cluster_id))
        .unwrap_or_else(|_| cluster_backup.clone());
    let new_alloc_id = latest_cluster_backup.alloc_id
        + cluster_backup.get_stores().len() as u64
        + config.new_store_id_delta
        + 1;
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(restore_pd_keyspace_meta(
        &config,
        &cluster_backup,
        new_alloc_id,
    ));
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
async fn restore_pd_keyspace_meta(
    config: &RestoreConfig,
    meta: &ClusterBackupMeta,
    new_alloc_id: u64,
) {
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
        new_alloc_id.to_be_bytes().to_vec(),
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
    pub wal_target_size: ReadableSize,
    pub new_store_id_delta: u64,
}
