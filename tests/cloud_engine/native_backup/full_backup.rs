// Copyright 2023 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    iter::FromIterator,
    path::{Path, PathBuf},
};

use cse_ctl::{backup, restore};
use kvengine::dfs::DFSConfig;
use rand::Rng;
use test_cloud_server::{client, ServerCluster};
use tikv::config::TiKvConfig;
use tikv_util::{info, warn};

use crate::alloc_node_id;

const NODES_SIZE: usize = 3;
const DATA_SIZE: usize = 2000;

// TODO: make the test independent to S3/Minio.
// TODO: test on more conditions (e.g. split and merge).

fn start_cluster_and_full_backup(
    backup_name: String,
    dfs_config: DFSConfig,
) -> (rfenginepb::ClusterBackupMeta, client::RefStore) {
    let nodes = Vec::from_iter((0..NODES_SIZE).into_iter().map(|_| alloc_node_id()));
    let mut cluster = ServerCluster::new(nodes, |_, conf: &mut TiKvConfig| {
        conf.dfs = dfs_config.clone();
    });
    cluster.wait_region_replicated(&[], 3);
    let mut client = cluster.new_client();

    // split to 3 regions
    client.split(&i_to_key(DATA_SIZE / 3));
    client.split(&i_to_key(DATA_SIZE * 2 / 3));
    cluster.wait_pd_region_count(3);

    // import data
    client.put_kv(0..DATA_SIZE, i_to_key, i_to_val);
    client.verify_data_with_ref_store();

    // execute backup
    let backup_config = backup::BackupConfig {
        dfs: dfs_config,
        ..Default::default()
    };
    let backup_meta = backup::backup_cluster(
        backup_config,
        false,
        backup_name,
        cluster.get_pd_client().as_ref(),
        None,
    )
    .expect("backup::backup_cluster");
    info!("backup_cluster result: {:?}", backup_meta);

    cluster.stop();

    let ref_store = client.take_ref_store();
    assert_eq!(ref_store.len(), DATA_SIZE);

    (backup_meta, ref_store)
}

fn restore_cluster(
    backup_name: String,
    base_path: &Path,
    dfs_config: DFSConfig,
    backup_meta: rfenginepb::ClusterBackupMeta,
    ref_store: client::RefStore,
) {
    let get_storage_path =
        |node_id: u16| -> PathBuf { base_path.join(format!("restore_{node_id}")) };

    let mut nodes = Vec::with_capacity(backup_meta.get_stores().len());
    for store in backup_meta.get_stores() {
        let node_id = alloc_node_id();
        nodes.push(node_id);

        let raft_db_path = get_storage_path(node_id).join("raft");
        let restore_config = restore::RestoreConfig {
            dfs: dfs_config.clone(),
            ..Default::default()
        };
        restore::restore_tikv(
            &restore_config,
            backup_name.to_string(),
            store.get_store_id(),
            raft_db_path.to_str().unwrap(),
        );
    }

    let mut cluster = ServerCluster::new(nodes, |node_id, conf: &mut TiKvConfig| {
        conf.storage.data_dir = get_storage_path(node_id).to_str().unwrap().to_string();
        conf.dfs = dfs_config.clone();
    });
    cluster.wait_region_replicated(&[], 3);

    let mut client = cluster.new_client();
    client.ingest_ref_store(ref_store);
    client.verify_data_with_ref_store();

    cluster.stop();
}

#[test]
fn test_native_full_backup() {
    test_util::init_log_for_test();

    let mut dfs_config = DFSConfig::default();
    dfs_config.override_from_env();
    if dfs_config.s3_endpoint.is_empty() {
        warn!("Environment variable DFS_S3_ENDPOINT is not set. Test case ignored.");
        return;
    }

    let backup_name = format!("backup_{}", rand::thread_rng().gen::<u16>());

    let (backup_meta, ref_store) =
        start_cluster_and_full_backup(backup_name.clone(), dfs_config.clone());

    let base_dir = tempfile::Builder::new()
        .prefix("test_restore_cluster")
        .tempdir()
        .unwrap();
    restore_cluster(
        backup_name,
        base_dir.path(),
        dfs_config,
        backup_meta,
        ref_store,
    );
}

fn i_to_key(i: usize) -> Vec<u8> {
    format!("key_{:08}", i).into_bytes()
}

fn i_to_val(i: usize) -> Vec<u8> {
    format!("val_{:08}", i).into_bytes().repeat(100)
}
