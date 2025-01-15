// Copyright 2024 TiKV Project Authors. Licensed under Apache-2.0.

use std::sync::Arc;

use futures::executor::block_on;
use kvengine::{dfs::S3Fs, WRITE_CF};
use merged_engine::{MergedEngine, MergedEngineConfig, MergedEngineContext};
use native_br::{backup, backup::BackupType, common::send_request_to_store};
use pd_client::PdClient;
use security::GetSecurityManager;
use test_cloud_server::{must_wait, oss::prepare_dfs, ServerCluster};
use test_pd_client::PdClientExt;
use tikv::config::TikvConfig;
use tikv_util::{
    codec::bytes::encode_bytes,
    config::{ReadableDuration, ReadableSize},
};

use crate::alloc_node_id_vec;

#[test]
fn test_merged_engine() {
    test_util::init_log_for_test();
    let (base_dir, _oss, dfs_conf) = prepare_dfs("test_merged_engine");
    let node_ids = alloc_node_id_vec(3);
    let cluster = ServerCluster::new(node_ids.clone(), |_, conf: &mut TikvConfig| {
        conf.dfs = dfs_conf.clone();
        conf.rfengine.lightweight_backup = true;
        conf.rfengine.target_file_size = ReadableSize::mb(16);
        conf.enable_inner_key_offset = true;
    });
    cluster.wait_region_replicated(&[], 3);
    let pd_client = cluster.get_pd_client();
    let split_key = encode_bytes(&crate::i_to_key(700));
    block_on(pd_client.split_regions(vec![split_key])).unwrap();
    let mut client = cluster.new_client();
    for i in 0..10 {
        client.put_kv(i * 100..i * 100 + 50, crate::i_to_key, crate::i_to_val);
    }
    let backup_config = backup::BackupConfig {
        dfs: dfs_conf.clone(),
        tolerate_err: 1,
        skip_keyspace_meta: true,
        ..Default::default()
    };
    let (_backup_key, backup_meta) = backup::backup_cluster(
        backup_config.clone(),
        BackupType::Lightweight,
        "merged_engine".to_string(),
        pd_client.as_ref(),
        None,
    )
    .unwrap();
    let s3fs = Arc::new(S3Fs::new_from_config(dfs_conf));
    let ctx = MergedEngineContext {
        pd: pd_client.clone(),
        fs: s3fs,
        local_dir: base_dir.path().join("merged_engine"),
        master_key: cluster.get_kvengine(node_ids[0]).get_master_key(),
        config: MergedEngineConfig {
            block_cache_size: ReadableSize::mb(1),
            timeout_fetch_wal: ReadableDuration::secs(10),
            merged_store_id: 1024,
            mem_table_size: cluster
                .get_node_config(node_ids[0])
                .rocksdb
                .writecf
                .write_buffer_size,
        },
        security_config: Arc::new(cluster.get_node_config(node_ids[0]).security.clone()),
    };
    let mut merged_engine = MergedEngine::new(ctx.clone(), backup_meta.clone());
    let merged_kv = merged_engine.get_kv();
    let all_shards = merged_kv.get_all_shard_id_vers();
    assert_eq!(all_shards.len(), 2);
    for i in 0..10 {
        let region_id = client.get_region_id(&crate::i_to_key(i * 100));
        let shard = merged_kv.get_shard(region_id).unwrap();
        let snap_access = shard.new_snap_access();
        for j in 0..50 {
            let key = crate::i_to_key(i * 100 + j);
            let val = crate::i_to_val(i * 100 + j);
            let item = snap_access.get(WRITE_CF, &key, u64::MAX);
            assert_eq!(item.get_value(), val.as_slice());
        }
    }
    for i in 0..10 {
        client.put_kv(i * 100 + 50..i * 100 + 60, crate::i_to_key, crate::i_to_val);
        if i == 5 {
            let encoded_key_500 = encode_bytes(&crate::i_to_key(500));
            block_on(pd_client.split_regions(vec![encoded_key_500])).unwrap();
        }
        if i == 8 {
            let encoded_key_600 = encode_bytes(&crate::i_to_key(600));
            let encoded_key_700 = encode_bytes(&crate::i_to_key(700));
            let merge_source = pd_client.get_region(&encoded_key_600).unwrap();
            let merge_target = pd_client.get_region(&encoded_key_700).unwrap();
            pd_client.merge_region(merge_source.get_id(), merge_target.get_id());
            must_wait(
                || {
                    let region = pd_client.get_region(&encoded_key_600).unwrap();
                    region.get_id() == merge_target.get_id()
                },
                10,
                || "merge region".to_string(),
            );
        }
    }
    let security_mgr = pd_client.get_security_mgr();
    let dfs = cluster.get_dfs().unwrap();
    for store_meta in backup_meta.get_stores() {
        let store_id = store_meta.get_store_id();
        let epoch = store_meta.get_epoch();
        let start_off = store_meta.get_offset();
        let store = pd_client.get_store(store_id).unwrap();
        let uri = security_mgr
            .build_uri(format!(
                "{}/rfengine/wal_chunk?epoch_id={}&start_off={}&end_off=0",
                &store.status_address, epoch, start_off
            ))
            .unwrap();
        let req = http::Request::get(uri.clone())
            .body(hyper::Body::empty())
            .unwrap();
        let runtime = dfs.get_runtime();
        let data = runtime
            .block_on(send_request_to_store(req, &store, &security_mgr))
            .unwrap();
        merged_engine
            .update_wal(store_id, epoch, start_off, data)
            .unwrap();
    }
    merged_engine.sync_merged().unwrap();
    for i in 0..10 {
        let region_id = client.get_region_id(&crate::i_to_key(i * 100));
        let shard = merged_kv.get_shard(region_id).unwrap();
        let snap_access = shard.new_snap_access();
        for j in 50..60 {
            let key = crate::i_to_key(i * 100 + j);
            let val = crate::i_to_val(i * 100 + j);
            let item = snap_access.get(WRITE_CF, &key, u64::MAX);
            assert_eq!(item.get_value(), val.as_slice());
        }
    }
    let merged_raft = merged_engine.get_raft();
    let region_peers = merged_raft.get_region_peer_map();
    for (region_id, _) in region_peers {
        if let Some(progress) = merged_engine.get_region_progress(region_id) {
            let truncated_index = merged_raft.get_truncated_index(region_id).unwrap();
            assert_eq!(progress.truncated_index, truncated_index);
        }
    }
}
