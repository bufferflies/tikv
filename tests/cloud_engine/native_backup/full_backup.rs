// Copyright 2023 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    iter::FromIterator,
    path::{Path, PathBuf},
};

use kvengine::dfs::DFSConfig;
use native_br::{backup, restore};
use rand::Rng;
use test_cloud_server::{client, oss::ObjectStorageService, ServerCluster};
use tikv::config::TikvConfig;
use tikv_util::{config::ReadableSize, info};

use crate::alloc_node_id;

const NODES_SIZE: usize = 3;
const DATA_SIZE: usize = 2000;

const WAL_TARGET_SIZE: ReadableSize = ReadableSize::mb(1);

// TODO: make the test independent to S3/Minio.
// TODO: test on more conditions (e.g. split and merge).

fn start_cluster_and_backup(
    backup_name: String,
    dfs_config: DFSConfig,
    lightweight: bool,
) -> (rfenginepb::ClusterBackupMeta, client::RefStore) {
    let nodes = Vec::from_iter((0..NODES_SIZE).into_iter().map(|_| alloc_node_id()));
    let mut cluster = ServerCluster::new(nodes, |_, conf: &mut TikvConfig| {
        conf.dfs = dfs_config.clone();
        conf.rfengine.lightweight_backup = lightweight;
        conf.rfengine.target_file_size = WAL_TARGET_SIZE;
        conf.rfengine.wal_chunk_target_file_size = ReadableSize::kb(128);
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
    // `skip_keyspace_meta: true` as there is no ETCD in CI.
    // TODO: support testing for keyspace meta backup.
    let backup_config = backup::BackupConfig {
        dfs: dfs_config,
        skip_keyspace_meta: true,
        ..Default::default()
    };
    let backup_ts = client.get_ts().into_inner();
    let backup_type = if lightweight {
        backup::BackupType::Lightweight
    } else {
        backup::BackupType::Full
    };
    let (_, backup_meta) = backup::backup_cluster_with_ts(
        backup_config,
        backup_type,
        backup_name,
        cluster.get_pd_client().as_ref(),
        backup_ts,
        None,
    )
    .expect("backup::backup_cluster");
    info!("backup_cluster result: {:?}", backup_meta);

    cluster.stop();

    let ref_store = client.dump_ref_store();
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
            wal_target_size: WAL_TARGET_SIZE,
            ..Default::default()
        };
        restore::restore_tikv(
            &restore_config,
            backup_name.to_string(),
            store.get_store_id(),
            100,
            raft_db_path.to_str().unwrap(),
        );
    }

    let mut cluster = ServerCluster::new(nodes, |node_id, conf: &mut TikvConfig| {
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

    let base_dir = tempfile::Builder::new()
        .prefix("test_restore_cluster")
        .tempdir()
        .unwrap();

    let oss_dir = base_dir.path().join("oss");
    let mut oss = ObjectStorageService::new(oss_dir);
    oss.start_server();

    let dfs_config = DFSConfig {
        prefix: "test_full_backup".to_string(),
        s3_endpoint: format!("http://127.0.0.1:{}", oss.port()),
        s3_key_id: "admin".to_string(),
        s3_secret_key: "admin".to_string(),
        s3_bucket: "test_full_backup".to_string(),
        s3_region: "local".to_string(),
        zstd_compression_level: "3".to_string(),
        ..Default::default()
    };

    let backup_name = format!("backup_{}", rand::thread_rng().gen::<u16>());

    let (backup_meta, ref_store) =
        start_cluster_and_backup(backup_name.clone(), dfs_config.clone(), false);

    let backup_dir = base_dir.path().join("backup");
    restore_cluster(
        backup_name,
        backup_dir.as_path(),
        dfs_config,
        backup_meta,
        ref_store,
    );

    // Don't graceful shutdown oss (`oss.shutdown()`), as some S3FS threads are
    // still alive and holding connections.
}

#[test]
fn test_native_lightweight_backup() {
    test_util::init_log_for_test();

    let base_dir = tempfile::Builder::new()
        .prefix("test_restore_cluster")
        .tempdir()
        .unwrap();

    let oss_dir = base_dir.path().join("oss");
    let mut oss = ObjectStorageService::new(oss_dir);
    oss.start_server();

    let dfs_config = DFSConfig {
        prefix: "test_lightweight_backup".to_string(),
        s3_endpoint: format!("http://127.0.0.1:{}", oss.port()),
        s3_key_id: "admin".to_string(),
        s3_secret_key: "admin".to_string(),
        s3_bucket: "test_lightweight_backup".to_string(),
        s3_region: "local".to_string(),
        zstd_compression_level: "3".to_string(),
        ..Default::default()
    };

    let backup_name = format!("backup_{}", rand::thread_rng().gen::<u16>());

    let (backup_meta, ref_store) =
        start_cluster_and_backup(backup_name.clone(), dfs_config.clone(), true);

    let backup_dir = base_dir.path().join("backup");
    restore_cluster(
        backup_name,
        backup_dir.as_path(),
        dfs_config,
        backup_meta,
        ref_store,
    );

    // Don't graceful shutdown oss (`oss.shutdown()`), as some S3FS threads are
    // still alive and holding connections.
}

fn i_to_key(i: usize) -> Vec<u8> {
    format!("xkey_{:08}", i).into_bytes()
}

fn i_to_val(i: usize) -> Vec<u8> {
    format!("val_{:08}", i).into_bytes().repeat(100)
}
