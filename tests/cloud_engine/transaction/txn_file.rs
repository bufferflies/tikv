// Copyright 2023 TiKV Project Authors. Licensed under Apache-2.0.

use std::{sync::Arc, time::Duration};

use anyhow::{self, bail};
use bytes::Bytes;
use futures::future::join_all;
use kvengine::{
    dfs::Dfs,
    table::txn_file::{TxnChunkBuilder, OP_PUT},
};
use kvproto::kvrpcpb::{
    BatchRollbackRequest, CheckTxnStatusRequest, CommitRequest, PrewriteRequest,
    ResolveLockRequest, TxnHeartBeatRequest,
};
use security::SecurityConfig;
use test_cloud_server::{
    client::{
        ClusterClient, ClusterClientOptions, CommitAction, MutateOptions, RequestOptions,
        TxnWriteMethod,
    },
    oss::prepare_dfs,
    ServerCluster,
};
use test_pd_client::PdWrapper;
use tikv_util::{info, time::Instant};

use crate::{alloc_node_id_vec, generate_keyspace_key, i_to_key, i_to_val, i_to_val_opt};

const NODES_COUNT: usize = 3;
const KEYSPACE_ID: u32 = 10;

#[test]
fn test_txn_file_commands() {
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
    req.set_txn_file_chunks(chunk_ids);
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
    req.set_primary_lock(primary_lock);
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
    req.set_txn_file_chunks(chunk_ids);
    req.set_primary_lock(primary_lock);
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
    req.set_txn_file_chunks(chunk_ids);
    req.set_primary_lock(primary_lock);
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
    req.set_txn_file_chunks(chunk_ids);
    req.set_primary_lock(primary_lock);
    req.set_lock_ttl(6000);
    req.set_start_version(start_ts);
    req.set_min_commit_ts(start_ts + 1);
    kv_client.kv_prewrite(&req).unwrap();

    let mut req = CommitRequest::new();
    req.set_context(ctx);
    req.set_start_version(start_ts);
    req.set_is_txn_file(true);
    let commit_ts = client.get_ts().into_inner();
    req.set_commit_version(commit_ts);
    kv_client.kv_commit(&req).unwrap();

    verify_range(&mut client, 0, 300);
    cluster.stop();
}

#[ignore]
#[test]
fn test_txn_file_basic() {
    test_util::init_log_for_test();

    let cases = vec![
        // size_factor, write_method, enable_inner_key_off
        (10, TxnWriteMethod::FileBased, true),
        // TODO: (10, TxnWriteMethod::FileBased, false),
        // Regression tests:
        (1, TxnWriteMethod::FileBased, true),
        // TODO: (1, TxnWriteMethod::FileBased, false),
        (1, TxnWriteMethod::Normal, true),
    ];

    let rt = tokio::runtime::Builder::new_multi_thread()
        .thread_name("test-txn-file-basic")
        .worker_threads(cases.len())
        .enable_all()
        .build()
        .unwrap();
    let mut handles = Vec::with_capacity(cases.len());
    for (size_factor, txn_write_method, enable_inner_key_off) in cases {
        handles.push(rt.spawn_blocking(move || {
            test_txn_file_basic_impl(size_factor, txn_write_method, enable_inner_key_off)
        }));
    }
    rt.block_on(join_all(handles));
}

fn test_txn_file_basic_impl(
    size_factor: usize,
    write_method: TxnWriteMethod,
    enable_inner_key_off: bool,
) {
    let (_temp_dir, mut oss, dfs_config) = prepare_dfs("test");

    let node_ids = alloc_node_id_vec(NODES_COUNT);
    let pd_wrapper = PdWrapper::new_test(1, &SecurityConfig::default(), None);
    let cluster_id = pd_wrapper.client().get_cluster_id().unwrap();
    info!("test_txn_file_basic";
        "size_factor" => size_factor,
        "enable_inner_key_off" => enable_inner_key_off,
        "write_method" => ?write_method,
        "cluster_id" => cluster_id);
    let mut cluster = ServerCluster::new_opt(
        node_ids,
        |_, conf| {
            conf.dfs = dfs_config.clone();
            conf.enable_inner_key_offset = enable_inner_key_off;
        },
        pd_wrapper,
    );
    cluster.start_tikv_workers(1, 2, false);
    cluster.wait_region_replicated(&[], 3);

    let gen_key = generate_keyspace_key(KEYSPACE_ID);

    let mut verify_data = {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let mut client = cluster.new_client();
        let mut txn_client = rt.block_on(cluster.new_txn_client());

        move |range: Option<(&[u8], &[u8])>, expected: usize| -> anyhow::Result<()> {
            let guard = client.ref_store();
            let ref_store = guard.lock().unwrap();
            let cnt = rt
                .block_on(txn_client.verify_data_by_scan(&ref_store, range))
                .unwrap();
            if cnt != expected {
                bail!("txn client verify data failed, expect {expected}, got {cnt}");
            }
            let cnt = client
                .verify_data_with_given_ref_store(&ref_store, range, &RequestOptions::default())
                .unwrap();
            if cnt != expected {
                bail!("client verify data failed, expect {expected}, got {cnt}");
            }
            Ok(())
        }
    };

    let rt = tokio::runtime::Runtime::new().unwrap();
    let _enter = rt.enter();

    let mut client = cluster.new_client_opt(ClusterClientOptions {
        txn_file_max_chunk_size: Some(1024),
        ..Default::default()
    });

    client.split_keyspace(KEYSPACE_ID);

    {
        // To work around that data not existed in ref store will not be checked.
        let guard = client.ref_store();
        let mut ref_store = guard.lock().unwrap();
        for k in 0..20 * size_factor {
            ref_store.del_kv(gen_key(k));
        }
    }

    {
        client
            .try_put_kv(
                0..10 * size_factor,
                &gen_key,
                i_to_val_opt("value0_", 3),
                MutateOptions {
                    commit_action: CommitAction::AsyncCommitSecondaryKeys(Duration::ZERO),
                    write_method,
                },
            )
            .unwrap();
        verify_data(None, 10 * size_factor).unwrap();
    }

    {
        // Verify locks of committed txn.
        client
            .try_put_kv(
                size_factor..11 * size_factor,
                &gen_key,
                i_to_val_opt("value1_", 3),
                MutateOptions {
                    commit_action: CommitAction::AsyncCommitSecondaryKeys(Duration::MAX),
                    write_method,
                },
            )
            .unwrap();
        // Verify on non-primary txn file when there are more than one.
        verify_data(
            Some((&gen_key(6 * size_factor), &gen_key(11 * size_factor))),
            5 * size_factor,
        )
        .unwrap();
    }

    {
        // Verify rollback & resolve locks.
        client
            .try_put_kv(
                2 * size_factor..12 * size_factor,
                &gen_key,
                i_to_val_opt("value2_", 3),
                MutateOptions {
                    commit_action: CommitAction::NoCommit,
                    write_method,
                },
            )
            .unwrap();
        let start = Instant::now_coarse();
        // Verify on non-primary txn file when there are more than one.
        verify_data(
            Some((&gen_key(7 * size_factor), &gen_key(12 * size_factor))),
            4 * size_factor,
        )
        .unwrap();
        // Wait until locks expired.
        // Note that txn client will not wait lock timeout during scan because of min
        // commit ts pushed.
        let elapsed = start.saturating_elapsed();
        assert!(elapsed > Duration::from_secs(2), "elapsed {:?}", elapsed);
    }

    verify_data(None, 11 * size_factor).unwrap();
    cluster.stop();
    oss.shutdown();
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
