// Copyright 2023 TiKV Project Authors. Licensed under Apache-2.0.

use std::sync::Arc;

use bytes::Bytes;
use kvengine::{
    dfs::Dfs,
    table::txn_file::{TxnChunkBuilder, OP_PUT},
};
use kvproto::kvrpcpb::{
    BatchRollbackRequest, CheckTxnStatusRequest, CommitRequest, PrewriteRequest,
    ResolveLockRequest, TxnHeartBeatRequest,
};
use test_cloud_server::{
    client::{ClusterClient, RequestOptions},
    ServerCluster,
};
use tikv_util::time::Instant;

use crate::{alloc_node_id_vec, i_to_key, i_to_val};

#[allow(clippy::redundant_clone)]
#[test]
fn test_txn_file() {
    test_util::init_log_for_test();
    let mut cluster = ServerCluster::new(alloc_node_id_vec(3), |_, _| {});
    cluster.wait_region_replicated(&[], 3);
    let mut client = cluster.new_client();
    let dfs = cluster.get_dfs().unwrap();
    let region_id = client.get_region_id(&[]);
    let ctx = client.new_rpc_ctx(region_id).unwrap();
    let kv_client = client.get_kv_client(ctx.get_peer().get_store_id());

    // test rollback lock.
    let start_ts = client.get_ts().into_inner();
    let chunk_ids = build_txn_files(&dfs, start_ts, 0, 300);
    let primary_lock = i_to_key(0);

    let mut req = PrewriteRequest::new();
    req.set_context(ctx.clone());
    req.set_txn_file_chunks(chunk_ids.clone());
    req.set_primary_lock(primary_lock.clone());
    req.set_lock_ttl(6000);
    req.set_start_version(start_ts);
    req.set_min_commit_ts(start_ts + 1);
    kv_client.kv_prewrite(&req).unwrap();

    let mut req = CheckTxnStatusRequest::new();
    req.set_context(ctx.clone());
    req.set_primary_key(primary_lock.clone());
    req.set_lock_ts(start_ts);
    req.set_caller_start_ts(start_ts + 5);
    req.set_is_txn_file(true);
    let resp = kv_client.kv_check_txn_status(&req).unwrap();
    let lock_info = resp.get_lock_info();
    // verify that min commit ts is pushed.
    assert_eq!(
        lock_info.min_commit_ts,
        start_ts + 6,
        "lock info {:?}",
        lock_info
    );

    let mut req = TxnHeartBeatRequest::new();
    req.set_context(ctx.clone());
    req.set_start_version(start_ts);
    req.set_primary_lock(primary_lock.clone());
    req.set_advise_lock_ttl(10000);
    req.set_is_txn_file(true);
    let resp = kv_client.kv_txn_heart_beat(&req).unwrap();
    // verify that lock ttl is pushed.
    assert_eq!(resp.lock_ttl, 10000);

    let mut req = BatchRollbackRequest::new();
    req.set_context(ctx.clone());
    req.set_is_txn_file(true);
    req.set_start_version(start_ts);
    kv_client.kv_batch_rollback(&req).unwrap();

    // test rollback lock.
    let start_ts = client.get_ts().into_inner();
    let chunk_ids = build_txn_files(&dfs, start_ts, 0, 300);
    let primary_lock = i_to_key(0);

    let mut req = PrewriteRequest::new();
    req.set_context(ctx.clone());
    req.set_txn_file_chunks(chunk_ids.clone());
    req.set_primary_lock(primary_lock.clone());
    req.set_lock_ttl(6000);
    req.set_start_version(start_ts);
    req.set_min_commit_ts(start_ts + 1);
    kv_client.kv_prewrite(&req).unwrap();

    let mut req = BatchRollbackRequest::new();
    req.set_context(ctx.clone());
    req.set_is_txn_file(true);
    req.set_start_version(start_ts);
    kv_client.kv_batch_rollback(&req).unwrap();

    // test resolve lock.
    let start_ts = client.get_ts().into_inner();
    let chunk_ids = build_txn_files(&dfs, start_ts, 0, 300);
    let primary_lock = i_to_key(0);

    let mut req = PrewriteRequest::new();
    req.set_context(ctx.clone());
    req.set_txn_file_chunks(chunk_ids.clone());
    req.set_primary_lock(primary_lock.clone());
    req.set_lock_ttl(6000);
    req.set_start_version(start_ts);
    req.set_min_commit_ts(start_ts + 1);
    kv_client.kv_prewrite(&req).unwrap();

    let mut req = ResolveLockRequest::new();
    req.set_context(ctx.clone());
    req.set_start_version(start_ts);
    req.set_is_txn_file(true);
    kv_client.kv_resolve_lock(&req).unwrap();

    // test success 2pc.
    let start_ts = client.get_ts().into_inner();
    let chunk_ids = build_txn_files(&dfs, start_ts, 0, 300);
    let primary_lock = i_to_key(0);

    let mut req = PrewriteRequest::new();
    req.set_context(ctx.clone());
    req.set_txn_file_chunks(chunk_ids.clone());
    req.set_primary_lock(primary_lock.clone());
    req.set_lock_ttl(6000);
    req.set_start_version(start_ts);
    req.set_min_commit_ts(start_ts + 1);
    kv_client.kv_prewrite(&req).unwrap();

    let mut req = CommitRequest::new();
    req.set_context(ctx.clone());
    req.set_start_version(start_ts);
    req.set_is_txn_file(true);
    let commit_ts = client.get_ts().into_inner();
    req.set_commit_version(commit_ts);
    kv_client.kv_commit(&req).unwrap();

    verify_range(&mut client, 0, 300);
    cluster.stop();
}

fn build_txn_files(dfs: &Arc<dyn Dfs>, start_ts: u64, start: usize, end: usize) -> Vec<u64> {
    let mut chunk_ids = vec![];
    let mut txn_chunk_builder = TxnChunkBuilder::new(10);
    let mut chunk_id = start_ts + 1;
    for i in start..end {
        let key = i_to_key(i);
        let val = i_to_val(i);
        txn_chunk_builder.add_entry(&key, OP_PUT, &val);
        if (i + 1) % 100 == 0 {
            let mut data_buf = vec![];
            txn_chunk_builder.finish(&mut data_buf);
            txn_chunk_builder = TxnChunkBuilder::new(10);
            dfs.get_runtime()
                .block_on(dfs.create_txn_chunk(chunk_id, Bytes::from(data_buf)))
                .unwrap();
            chunk_ids.push(chunk_id);
            chunk_id += 1;
        }
    }
    chunk_ids
}

fn verify_range(client: &mut ClusterClient, start: usize, end: usize) {
    let put_time = Instant::now();
    for i in start..end {
        let key = i_to_key(i);
        let val = i_to_val(i);
        let opt = RequestOptions::default();
        client
            .verify_key_value(&key, Some(&val), put_time, &opt)
            .unwrap();
    }
}
