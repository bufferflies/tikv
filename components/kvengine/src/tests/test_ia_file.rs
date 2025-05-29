// Copyright 2025 TiKV Project Authors. Licensed under Apache-2.0.

use std::{rc::Rc, sync::Arc, time::Duration};

use futures::executor::block_on;
use test_util::init_log_for_test;
use tikv_util::time::Instant;

use crate::{
    dfs,
    dfs::FileType,
    ia::{ia_file::IaFile, manager::IaManager, util::IaManagerOptionsBuilder},
    table::sstable::{BlockCache, SsTable},
    tests::{new_table, new_test_engine_opt},
    util::test_util::KeyBuilder,
    Iterator,
};

#[rstest::rstest]
#[case::concurrency(2)]
#[case::concurrency(8)]
fn test_sync_read(#[case] concurrency: usize) {
    init_log_for_test();

    let tempdir = tempfile::tempdir().unwrap();
    let local_path = tempdir.path();
    let file_id = 1;

    let (engine, _) = new_test_engine_opt(true, 1024, "");
    let mut saved_vals: Vec<Rc<Vec<u8>>> = Vec::new();
    let t = new_table(&engine, file_id, 0, 100, 1000, false, &mut saved_vals);

    let ia_rt = tokio::runtime::Builder::new_multi_thread()
        .thread_name("ia-mgr")
        .enable_all()
        .worker_threads(2)
        .build()
        .unwrap();
    let sync_read_concurrency = (concurrency + 1) / 2;
    let options = IaManagerOptionsBuilder::default()
        .segment_size(4096)
        .sync_read(false, sync_read_concurrency, Duration::from_secs(3))
        .build()
        .unwrap();
    let mgr = IaManager::new(options, engine.fs.clone(), None, ia_rt.into()).unwrap();

    let dfs_opts = dfs::Options::default().with_shard(1, 1);
    block_on(IaFile::prepare_table_meta(
        file_id,
        FileType::Sst,
        t.meta_offset() as u64,
        local_path,
        &dfs_opts,
        &mgr,
    ))
    .unwrap();
    let ia_file = IaFile::open_in_path(file_id, FileType::Sst, local_path, mgr.clone()).unwrap();
    let t_async = SsTable::new(Arc::new(ia_file), BlockCache::None, None).unwrap();

    // Test in tokio async context.
    // Simulate unified pool.
    let unified_pool = tokio::runtime::Builder::new_multi_thread()
        .thread_name("sync-read-unified-pool")
        .enable_all()
        .worker_threads(concurrency)
        .build()
        .unwrap();
    let start_time = Instant::now_coarse();
    let mut js = tokio::task::JoinSet::new();
    for i in 0..concurrency {
        let kb = engine.key_builder.clone();
        let tt = if i & 1 == 0 {
            t_async.clone()
        } else {
            t.clone()
        };
        let task = unified_pool.spawn(async move {
            verify_table_by_scan(&kb, &tt, 0, 100, 1000);
        });
        js.spawn_on(task, unified_pool.handle());
    }
    block_on(async move {
        while let Some(r) = js.join_next().await {
            r.unwrap().unwrap();
        }
    });
    println!(
        "test_sync_read: concurrency {}, take {:?}",
        concurrency,
        start_time.saturating_elapsed()
    );

    // Test in sync context.
    verify_table_by_scan(&engine.key_builder, &t_async, 0, 100, 1000);
}

#[track_caller]
fn verify_table_by_scan(kb: &KeyBuilder, t: &SsTable, begin: usize, end: usize, ver: u64) {
    let mut i = begin;
    let mut it = t.new_iterator(false, false);
    it.rewind();
    while it.valid() {
        let key = it.key();
        assert_eq!(kb.i_to_inner_key(i).as_ref(), key);
        let val = it.value();
        assert_eq!(val.version, ver);
        assert_eq!(val.get_value(), key.repeat(2));

        it.next();
        i += 1;
    }
    assert_eq!(i, end);
}
