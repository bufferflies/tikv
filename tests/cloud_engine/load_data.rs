// Copyright 2023 TiKV Project Authors. Licensed under Apache-2.0.

use std::{fs, mem, sync::Arc};

use api_version::ApiV2;
use bytes::{BufMut, BytesMut};
use futures::executor::block_on;
use kvengine::{dfs::DFSConfig, table::sstable::ZSTD_COMPRESSION};
use load_data::task::{
    LoadDataContext, LoadTaskMsg, LoadTaskScheduler, LoadTaskWorker, TaskContext,
};
use pd_client::PdClient;
use test_cloud_server::{
    client::{RefStore, RequestOptions},
    oss::ObjectStorageService,
    try_wait, ServerCluster,
};
use tikv::config::TikvConfig;

use crate::alloc_node_id_vec;

const KEYSPACE_ID: u32 = 123;
const DATA_COUNT: usize = 10000;
const DATA_BATCH_SIZE: usize = 10;
const COMPRESSION_TYPE: u8 = ZSTD_COMPRESSION;

#[test]
fn test_load_data() {
    test_util::init_log_for_test();

    impl_test_load_data(false);
    impl_test_load_data(true);
}

fn impl_test_load_data(enable_inner_key_off: bool) {
    let base_dir = tempfile::Builder::new()
        .prefix("test_load_data")
        .tempdir()
        .unwrap();

    let oss_dir = base_dir.path().join("oss");
    let mut oss = ObjectStorageService::new(oss_dir);
    oss.start_server();
    let dfs_conf = DFSConfig {
        prefix: "load_data".to_string(),
        s3_endpoint: format!("http://127.0.0.1:{}", oss.port()),
        s3_key_id: "admin".to_string(),
        s3_secret_key: "admin".to_string(),
        s3_bucket: "load_data".to_string(),
        s3_region: "local".to_string(),
        zstd_compression_level: "3".to_string(),
        ..Default::default()
    };

    let runtime = Arc::new(
        tokio::runtime::Builder::new_multi_thread()
            .worker_threads(4)
            .thread_name("load_data_worker")
            .enable_all()
            .build()
            .unwrap(),
    );

    let mut cluster = ServerCluster::new(alloc_node_id_vec(3), |_, conf: &mut TikvConfig| {
        conf.dfs = dfs_conf.clone();
        conf.enable_inner_key_offset = enable_inner_key_off;
    });
    cluster.wait_region_replicated(&[], 3);
    let pd_client = cluster.get_pd_client();
    let mut client = cluster.new_client();
    client.split_keyspace(KEYSPACE_ID);

    // Init task.
    let dfs = Arc::new(kvengine::dfs::S3Fs::new(
        dfs_conf.prefix,
        dfs_conf.s3_endpoint,
        dfs_conf.s3_key_id,
        dfs_conf.s3_secret_key,
        dfs_conf.s3_region,
        dfs_conf.s3_bucket,
    ));
    let load_data_dir = base_dir.path().join("load_data");
    fs::create_dir_all(&load_data_dir).unwrap();
    let start_ts = block_on(pd_client.get_tso()).unwrap().into_inner();
    let commit_ts = block_on(pd_client.get_tso()).unwrap().into_inner();
    let load_data_ctx = LoadDataContext {
        dir: load_data_dir,
        dfs,
        pd: pd_client,
        runtime,
        max_in_mem_size: 1024, // 1KB
    };
    let scheduler = init_task(load_data_ctx, start_ts, commit_ts);

    // Put chunks.
    let (chunk_ids, ref_store) = put_chunks(&scheduler);

    // Build.
    build(&scheduler, chunk_ids);

    // Verify data consistency.
    let verified_count = client
        .verify_data_with_given_ref_store(&ref_store, None, &RequestOptions::default())
        .expect("verify_data_with_given_ref_store");
    assert_eq!(verified_count, DATA_COUNT);

    cluster.stop();
}

fn init_task(ctx: LoadDataContext, start_ts: u64, commit_ts: u64) -> LoadTaskScheduler {
    let task_ctx = TaskContext {
        start_ts,
        commit_ts,
        inner_key_off: None,
        key_prefix: vec![],
    };

    let mut worker = LoadTaskWorker::new(ctx, task_ctx);
    let scheduler = worker.get_scheduler();
    std::thread::spawn(move || {
        worker.run();
    });

    assert!(
        !scheduler.is_canceled(),
        "task canceled: {}",
        scheduler.error_msg()
    );
    scheduler
}

fn put_chunks(scheduler: &LoadTaskScheduler) -> (Vec<u64>, RefStore) {
    let mut chunk_ids =
        Vec::with_capacity((DATA_COUNT as f64 / DATA_BATCH_SIZE as f64).ceil() as usize);
    let mut ref_store = RefStore::default();
    let keyspace_prefix = ApiV2::get_txn_keyspace_prefix(KEYSPACE_ID);

    let capacity = (2 + i_to_key_with_prefix(&keyspace_prefix, 0).len() + 4 + i_to_val(0).len())
        * DATA_BATCH_SIZE;
    for i in (0..DATA_COUNT).step_by(DATA_BATCH_SIZE) {
        let mut buf = BytesMut::with_capacity(capacity);

        for j in 0..DATA_BATCH_SIZE {
            let key = i_to_key_with_prefix(&keyspace_prefix, i + j);
            let val = i_to_val(i + j);

            buf.put_u16_le(key.len() as u16);
            buf.put_slice(&key);
            buf.put_u32_le(val.len() as u32);
            buf.put_slice(&val);

            ref_store.put_kv(key, val);
        }

        let chunk_id = i as u64;
        scheduler
            .sender
            .send(LoadTaskMsg::AddChunk {
                chunk_id,
                chunk_data: buf.freeze(),
            })
            .unwrap();

        chunk_ids.push(chunk_id)
    }

    let mut unhandled_chunk_ids = chunk_ids.clone();
    try_wait(
        || {
            assert!(
                !scheduler.is_canceled(),
                "task canceled: {}",
                scheduler.error_msg()
            );
            let mut res = block_on(scheduler.query_unhandled_chunks(unhandled_chunk_ids.clone()));
            unhandled_chunk_ids = mem::take(&mut res);
            unhandled_chunk_ids.is_empty()
        },
        10,
    );

    (chunk_ids, ref_store)
}

fn build(scheduler: &LoadTaskScheduler, chunk_ids: Vec<u64>) {
    scheduler
        .sender
        .send(LoadTaskMsg::Build {
            chunk_ids,
            compression_type: COMPRESSION_TYPE,
        })
        .unwrap();

    try_wait(
        || {
            assert!(
                !scheduler.is_canceled(),
                "task canceled: {}",
                scheduler.error_msg()
            );
            scheduler.is_finished()
        },
        10,
    );
}

fn i_to_val(i: usize) -> Vec<u8> {
    format!("val_{:08}", i).into_bytes()
}

fn i_to_key_with_prefix(prefix: &[u8], i: usize) -> Vec<u8> {
    let mut key = prefix.to_vec();
    key.extend_from_slice(&format!("key_{:06}", i).into_bytes());
    key
}
