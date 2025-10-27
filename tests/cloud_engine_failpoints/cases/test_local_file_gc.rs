// Copyright 2025 TiKV Project Authors. Licensed under Apache-2.0.

use std::{thread, time::Duration};

use security::SecurityConfig;
use test_cloud_server::{
    alloc_node_id,
    client::{ClusterClientOptions, CommitAction, MutateOptions, TxnWriteMethod},
    oss::prepare_dfs,
    ServerClusterBuilder, TikvWorkerOptions,
};
use test_pd_client::PdWrapper;
use test_util::*;
use tikv_util::config::{ReadableDuration, ReadableSize};

use crate::cases::{i_to_key, i_to_keyspace_key, i_to_val};

// To reproduce https://github.com/tidbcloud/cloud-storage-engine/issues/3281, where file gc
// mistakenly deletes files that have already been loaded locally but haven't
// been applied to the kvengine yet.
#[test]
fn test_no_mistakenly_delete_pending_flushed_files() {
    init_log_for_test();

    let node_id = alloc_node_id();
    let cluster = test_cloud_server::ServerCluster::new(vec![node_id], |_, cfg| {
        // Set a short GC timeout and tick interval to trigger the GC procedure faster
        cfg.raft_store.local_file_gc_timeout = ReadableDuration::millis(300);
        cfg.raft_store.local_file_gc_tick_interval = ReadableDuration::millis(300);
        // Set a small write buffer size to trigger flush faster
        cfg.rocksdb.writecf.write_buffer_size = ReadableSize::kb(1);
        // Set a short file TTL to ensure file reads happen.
        cfg.kvengine.file_ttl = 1;
    });

    // Enable failpoint to pause after load local files before apply.
    fail::cfg("after_load_local_files", "pause").unwrap();

    // Write some data to trigger flush
    let mut client = cluster.new_client();
    client.put_kv(0..100, i_to_key, i_to_val);

    // Wait for GC to run
    thread::sleep(Duration::from_secs(1));

    // Remove failpoint and wait for apply and ttl cache expire
    fail::remove("after_load_local_files");
    thread::sleep(Duration::from_secs(2));

    // This will trigger a local file read in contains_in_older_table.
    client.del_kv(0..10, i_to_key);
}

// This test verifies that local file GC takes a snapshot of the filesystem at
// the beginning of its process, and any new files created after the snapshot
// (e.g., by post-split region) will be ignored and not participate in the GC
// process. This ensures that GC only considers files that existed at the time
// the snapshot was taken, preventing accidental deletion of newly created
// files.
#[test]
fn test_no_mistakenly_delete_files_when_region_split() {
    init_log_for_test();

    let node_id = alloc_node_id();
    let cluster = test_cloud_server::ServerCluster::new(vec![node_id], |_, cfg| {
        // Set a short GC timeout and tick interval to trigger the GC procedure faster
        cfg.raft_store.local_file_gc_timeout = ReadableDuration::millis(300);
        cfg.raft_store.local_file_gc_tick_interval = ReadableDuration::millis(300);
        // Set a small write buffer size to trigger flush faster
        cfg.rocksdb.writecf.write_buffer_size = ReadableSize::kb(1);
        // Set a short file TTL to ensure file reads happen.
        cfg.kvengine.file_ttl = 1;
    });

    // Enable failpoint to pause after load local files before apply.
    fail::cfg("after_load_local_files", "pause").unwrap();

    // Write some data to trigger flush
    let mut client = cluster.new_client();
    client.put_kv(0..100, i_to_key, i_to_val);

    // Wait for GC to run
    fail::cfg("after_gc_collect_kvengine_files", "pause").unwrap();
    thread::sleep(Duration::from_secs(1));

    // Split a new shard and write some data to the new shard to trigger flush on
    // the new shard.
    let split_key = b"xkey50";
    client.split(split_key);
    cluster.wait_pd_region_count(2);
    client.put_kv(0..100, i_to_key, i_to_val);

    // Wait for files to exceeds the local_file_gc_timeout.
    thread::sleep(Duration::from_secs(1));

    // Remove failpoint to resume GC.
    fail::remove("after_gc_collect_kvengine_files");
    // Wait for GC to finish.
    thread::sleep(Duration::from_secs(1));

    // Verify that SST files are not deleted.
    let config = cluster.get_node_config(node_id);
    let db_dir = std::path::Path::new(&config.storage.data_dir).join("db");
    let sst_count: usize = std::fs::read_dir(&db_dir)
        .unwrap_or_else(|_| panic!("failed to read db directory: {:?}", db_dir))
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map_or(false, |ext| ext == "sst"))
        .count();
    assert_ne!(sst_count, 0, "the number of SST files should not be 0");
}

// This test verifies that txn files are properly protected from GC during their
// lifecycle. The test:
// 1. Pauses txn file apply using failpoint after files are downloaded but
//    before apply
// 2. Lets GC run while files are in this intermediate state
// 3. Verifies that GC correctly protects these files and does not delete them
// This ensures the GC protection mechanism works for files that are downloaded
// but not yet applied.
#[test]
fn test_no_mistakenly_delete_txn_files() {
    init_log_for_test();

    const KEYSPACE_ID: u32 = 10;
    let (_temp_dir, mut oss, dfs_config) = prepare_dfs("test");
    let pd_wrapper = PdWrapper::new_test(1, &SecurityConfig::default(), None);

    let node_id = alloc_node_id();
    let mut cluster = ServerClusterBuilder::new(vec![node_id], |_, cfg| {
        cfg.dfs = dfs_config.clone();
        cfg.raft_store.enable_inner_key_offset = true;
        // Set a short GC timeout and tick interval to trigger the GC procedure faster
        cfg.raft_store.local_file_gc_timeout = ReadableDuration::secs(1);
        cfg.raft_store.local_file_gc_tick_interval = ReadableDuration::secs(1);
        // Set a short file TTL to ensure file reads happen.
        cfg.kvengine.file_ttl = 1;
    })
    .pd(pd_wrapper)
    .build();
    cluster.start_tikv_workers(vec![alloc_node_id()], TikvWorkerOptions::default());
    cluster.wait_region_replicated(&[], 1);

    // Enable failpoint to pause after load local files before apply.
    fail::cfg("after_load_local_txn_files", "pause").unwrap();
    // Leader preloads txn files to local during Scheduler module processing.
    // During preprocess, we should skip txn file exist-checks, to ensure that our
    // tests actually cover the logic where the local files are loaded but not
    // yet applied.
    fail::cfg("skip_check_txn_file_exists_when_preprocess", "return").unwrap();

    let mut client = cluster.new_client_opt(ClusterClientOptions {
        txn_file_max_chunk_size: Some(1024),
        ..Default::default()
    });
    client.split_keyspace(KEYSPACE_ID);

    // Write data using txn file method to generate txn files. We use a separate
    // thread instead of the main thread, because pausing
    // `after_load_local_txn_files` will block the `try_put_kv` procedure and
    // thus block the main thread.
    let write_handle = thread::spawn(move || {
        let gen_key = i_to_keyspace_key(KEYSPACE_ID);
        client
            .try_put_kv(
                0..100,
                &gen_key,
                |i: usize| format!("value_{:03}", i).into_bytes().repeat(3),
                MutateOptions {
                    commit_action: CommitAction::AsyncCommitSecondaryKeys(Duration::ZERO),
                    write_method: TxnWriteMethod::FileBased,
                    ..Default::default()
                },
            )
            .unwrap();
    });

    // Wait for GC to run (while txn file apply is paused)
    thread::sleep(Duration::from_secs(1));

    // Check txn directory: <temp_dir>/cluster_<id>/<node_id>/db/txn/ Assert txn
    // files should NOT be empty (should be protected from GC)
    let config = cluster.get_node_config(node_id);
    let txn_dir = std::path::Path::new(&config.storage.data_dir).join("db/txn");
    let txn_files: Vec<_> = std::fs::read_dir(&txn_dir)
        .unwrap_or_else(|_| panic!("failed to read txn directory: {:?}", txn_dir))
        .filter_map(|entry| entry.ok())
        .filter(|entry| entry.path().extension().map_or(false, |ext| ext == "txn"))
        .collect();
    assert!(
        !txn_files.is_empty(),
        "test failed: all txn files were deleted by GC!"
    );
    fail::remove("after_load_local_txn_files");
    write_handle.join().unwrap();
    oss.shutdown();
}

// This test verifies that local file GC takes a snapshot of the filesystem at
// the beginning of its process, and any new files created after the snapshot
// (e.g., by post-split region) will be ignored and not participate in the GC
// process. This ensures that GC only considers files that existed at the time
// the snapshot was taken, preventing accidental deletion of newly created
// files.
#[test]
fn test_no_mistakenly_delete_txn_files_when_region_split() {
    init_log_for_test();

    const KEYSPACE_ID: u32 = 10;
    let (_temp_dir, mut oss, dfs_config) = prepare_dfs("test");
    let pd_wrapper = PdWrapper::new_test(1, &SecurityConfig::default(), None);

    let node_id = alloc_node_id();
    let mut cluster = ServerClusterBuilder::new(vec![node_id], |_, cfg| {
        cfg.dfs = dfs_config.clone();
        cfg.raft_store.enable_inner_key_offset = true;
        // Set a short GC timeout and tick interval to trigger the GC procedure faster
        cfg.raft_store.local_file_gc_timeout = ReadableDuration::millis(500);
        cfg.raft_store.local_file_gc_tick_interval = ReadableDuration::millis(500);
        // Set a short file TTL to ensure file reads happen.
        cfg.kvengine.file_ttl = 1;
    })
    .pd(pd_wrapper)
    .build();
    cluster.start_tikv_workers(vec![alloc_node_id()], TikvWorkerOptions::default());
    cluster.wait_region_replicated(&[], 1);

    // Enable failpoint to pause after file download but before apply
    fail::cfg("after_load_local_txn_files", "pause").unwrap();
    fail::cfg("after_gc_collect_kvengine_files", "pause").unwrap();
    // See the comment in test_no_mistakenly_delete_txn_files for the purpose of
    // this line.
    fail::cfg("skip_check_txn_file_exists_when_preprocess", "return").unwrap();

    // Create client with txn file support and write data using txn file method
    let mut client = cluster.new_client_opt(ClusterClientOptions {
        txn_file_max_chunk_size: Some(1024),
        ..Default::default()
    });
    let config = cluster.get_node_config(node_id);
    client.split_keyspace(KEYSPACE_ID);

    // Write data using txn file method to generate txn files then split the
    // region to generate another newer txn files.
    //
    // See comments in test_no_mistakenly_delete_txn_files for the purpose of
    // using a separate thread.
    let write_handle = thread::spawn(move || {
        let gen_key = i_to_keyspace_key(KEYSPACE_ID);
        client
            .try_put_kv(
                0..100,
                &gen_key,
                |i: usize| format!("value_{:03}", i).into_bytes().repeat(3),
                MutateOptions {
                    commit_action: CommitAction::AsyncCommitSecondaryKeys(Duration::ZERO),
                    write_method: TxnWriteMethod::FileBased,
                    ..Default::default()
                },
            )
            .unwrap();
        // Wait for txn files to be collected by local gc procedure.
        thread::sleep(Duration::from_secs(1));

        let split_key = b"xkey50";
        client.split(split_key);
        thread::sleep(Duration::from_millis(300));
        // Create new txn file by the post-split region.
        client
            .try_put_kv(
                0..100,
                &gen_key,
                |i: usize| format!("value_{:03}", i).into_bytes().repeat(3),
                MutateOptions {
                    commit_action: CommitAction::AsyncCommitSecondaryKeys(Duration::ZERO),
                    write_method: TxnWriteMethod::FileBased,
                    ..Default::default()
                },
            )
            .unwrap();
    });

    // Wait for txn files to be collected by local gc procedure.
    thread::sleep(Duration::from_secs(1));
    fail::remove("after_load_local_txn_files");
    // Wait for region split + put finish.
    thread::sleep(Duration::from_secs(1));

    // Continue GC.
    fail::remove("after_gc_collect_kvengine_files");
    // Wait for GC to finish
    thread::sleep(Duration::from_secs(1));

    // Check txn directory: <temp_dir>/cluster_<id>/<node_id>/db/txn/
    // Assert txn files should NOT be empty (should be protected from GC)
    let txn_dir = std::path::Path::new(&config.storage.data_dir).join("db/txn");
    let txn_files: Vec<_> = std::fs::read_dir(&txn_dir)
        .unwrap_or_else(|_| panic!("failed to read txn directory: {:?}", txn_dir))
        .filter_map(|entry| entry.ok())
        .filter(|entry| entry.path().extension().map_or(false, |ext| ext == "txn"))
        .collect();
    assert!(
        !txn_files.is_empty(),
        "test failed: all txn files were deleted by GC!"
    );
    write_handle.join().unwrap();
    oss.shutdown();
}

// This test verifies that files are protected during node restart when shard
// meta is successfully persisted. The test simulates a scenario where:
// 1. Files are preprocessed and shard meta is successfully persisted to disk
// 2. During TiKVServer::setup, the persisted changeset is applied to kvengine
// 3. When GC worker starts later in TiKVServer::run, these files are already
//    tracked in kvengine's metadata
// 4. GC correctly protects these files from deletion since they are properly
//    tracked
// This ensures that files are safe from GC when they are properly recorded in
// kvengine's metadata through successful shard meta persistence and apply.
#[test]
fn test_no_mistakenly_delete_files_when_node_restart() {
    init_log_for_test();

    let node_id = alloc_node_id();
    let mut cluster = test_cloud_server::ServerCluster::new(vec![node_id], |_, cfg| {
        // Set a short GC timeout and tick interval to trigger the GC procedure faster
        cfg.raft_store.local_file_gc_timeout = ReadableDuration::millis(300);
        cfg.raft_store.local_file_gc_tick_interval = ReadableDuration::millis(300);
        // Set a small write buffer size to trigger flush faster
        cfg.rocksdb.writecf.write_buffer_size = ReadableSize::kb(1);
        // Set a short file TTL to ensure file reads happen.
        cfg.kvengine.file_ttl = 1;
    });

    // Enable failpoint to pause after load local files before apply.
    fail::cfg("after_load_local_files", "pause").unwrap();

    // Write some data to trigger flush
    let mut client = cluster.new_client();
    client.put_kv(0..100, i_to_key, i_to_val);
    // Wait for ShardMeta is persisted to the disk, so that recover_change_set()
    // will apply the change set to kvengine during TikvServer::setup().
    thread::sleep(Duration::from_secs(1));

    // Stop the node.
    cluster.stop_node(node_id);
    // Enable skip tracking files in preprocess change set.
    fail::cfg("before_track_pending_files", "return").unwrap();

    cluster.start_node(node_id, |_, cfg| {
        // Set a short GC timeout and tick interval to trigger the GC procedure faster
        cfg.raft_store.local_file_gc_timeout = ReadableDuration::millis(300);
        cfg.raft_store.local_file_gc_tick_interval = ReadableDuration::millis(300);
        // Set a small write buffer size to trigger flush faster
        cfg.rocksdb.writecf.write_buffer_size = ReadableSize::kb(1);
        // Set a short file TTL to ensure file reads happen.
        cfg.kvengine.file_ttl = 1;
    });

    // Wait for GC to run
    thread::sleep(Duration::from_secs(1));

    let config = cluster.get_node_config(node_id);
    let db_dir = std::path::Path::new(&config.storage.data_dir).join("db");
    let sst_count: usize = std::fs::read_dir(&db_dir)
        .unwrap_or_else(|_| panic!("failed to read db directory: {:?}", db_dir))
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map_or(false, |ext| ext == "sst"))
        .count();
    assert_ne!(sst_count, 0, "the number of SST files should not be 0");
}

// This test verifies the handling of files during node restart when shard meta
// persistence is unpersisted. The test simulates a scenario where:
// 1. Files are preprocessed but shard meta (preprocess index) is not persisted
//    to disk yet.
// 2. Node restarts in this intermediate state
// 3. GC runs and might mistakenly delete these files
// 4. Preprocess is re-executed and files are re-downloaded
// This ensures that even if GC deletes files that weren't properly tracked due
// to incomplete shard meta persistence, the system can recover by
// re-downloading the files during preprocess.
#[test]
fn test_no_mistakenly_delete_files_when_node_restart_and_failed_to_persist_shard_meta() {
    init_log_for_test();

    const KEYSPACE_ID: u32 = 10;
    let (_temp_dir, mut oss, dfs_config) = prepare_dfs("test");
    let pd_wrapper = PdWrapper::new_test(1, &SecurityConfig::default(), None);

    let node_id = alloc_node_id();
    let mut cluster = ServerClusterBuilder::new(vec![node_id], |_, cfg| {
        cfg.dfs = dfs_config.clone();
        cfg.raft_store.enable_inner_key_offset = true;
        // Set a small write buffer size to trigger flush faster
        cfg.rocksdb.writecf.write_buffer_size = ReadableSize::kb(1);
    })
    .pd(pd_wrapper)
    .build();
    cluster.start_tikv_workers(vec![alloc_node_id()], TikvWorkerOptions::default());
    cluster.wait_region_replicated(&[], 1);

    // Enable failpoint to pause after load local files before apply.
    fail::cfg("after_load_local_files", "pause").unwrap();
    fail::cfg("skip_apply_flush_shard_meta", "return").unwrap();

    // Write some data to trigger flush (creates SST files)
    let mut client = cluster.new_client();
    client.put_kv(0..100, i_to_key, i_to_val);

    // Create client with txn file support and write data using txn file method
    // (creates Txn files)
    let mut txn_client = cluster.new_client_opt(ClusterClientOptions {
        txn_file_max_chunk_size: Some(1024),
        ..Default::default()
    });
    txn_client.split_keyspace(KEYSPACE_ID);

    // Write data using txn file method to generate txn files
    let write_handle = thread::spawn(move || {
        let gen_key = i_to_keyspace_key(KEYSPACE_ID);
        txn_client
            .try_put_kv(
                0..50,
                &gen_key,
                |i: usize| format!("txn_value_{:03}", i).into_bytes().repeat(3),
                MutateOptions {
                    commit_action: CommitAction::AsyncCommitSecondaryKeys(Duration::ZERO),
                    write_method: TxnWriteMethod::FileBased,
                    ..Default::default()
                },
            )
            .unwrap();
    });

    // Wait for ShardMeta is persisted to the disk, so that recover_change_set()
    // will apply the change set to kvengine during TikvServer::setup().
    thread::sleep(Duration::from_secs(1));

    // Wait for txn file write to complete
    write_handle.join().unwrap();

    // Stop the node.
    cluster.stop_node(node_id);

    let config = cluster.get_node_config(node_id);
    let data_dir = std::path::Path::new(&config.storage.data_dir);
    let db_dir = data_dir.join("db");
    let txn_dir = db_dir.join("txn");

    // Manually delete all SST files from data_dir to simulate GC deletion
    // Note: Simulating GC deletion during preprocessing is not simple to implement,
    // so we use manual file deletion to achieve similar effects.
    let sst_files: Vec<_> = std::fs::read_dir(&db_dir)
        .unwrap_or_else(|_| panic!("failed to read db directory: {:?}", db_dir))
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map_or(false, |ext| ext == "sst"))
        .map(|e| e.path())
        .collect();
    for sst_file in sst_files {
        if let Err(e) = std::fs::remove_file(&sst_file) {
            panic!("failed to delete SST file {:?}: {}", sst_file, e);
        }
    }

    // Manually delete all Txn files from txn directory to simulate GC deletion
    let txn_files: Vec<_> = std::fs::read_dir(&txn_dir)
        .unwrap_or_else(|_| panic!("failed to read txn directory: {:?}", txn_dir))
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map_or(false, |ext| ext == "txn"))
        .map(|e| e.path())
        .collect();
    for txn_file in txn_files {
        if let Err(e) = std::fs::remove_file(&txn_file) {
            panic!("failed to delete Txn file {:?}: {}", txn_file, e);
        }
    }

    cluster.start_node(node_id, |_, cfg| {
        // Set a short GC timeout and tick interval to trigger the GC procedure faster
        cfg.raft_store.local_file_gc_timeout = ReadableDuration::millis(300);
        cfg.raft_store.local_file_gc_tick_interval = ReadableDuration::millis(300);
        // Set a small write buffer size to trigger flush faster
        cfg.rocksdb.writecf.write_buffer_size = ReadableSize::kb(1);
        // Set a short file TTL to ensure file reads happen.
        cfg.kvengine.file_ttl = 1;
    });
    // Wait for preprocess load files
    thread::sleep(Duration::from_secs(1));

    // Verify SST files are re-downloaded
    let sst_count: usize = std::fs::read_dir(&db_dir)
        .unwrap_or_else(|_| panic!("failed to read db directory: {:?}", db_dir))
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map_or(false, |ext| ext == "sst"))
        .count();
    assert_ne!(sst_count, 0, "the number of SST files should not be 0");

    // Verify Txn files are re-downloaded
    let txn_count: usize = std::fs::read_dir(&txn_dir)
        .unwrap_or_else(|_| panic!("failed to read txn directory: {:?}", txn_dir))
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map_or(false, |ext| ext == "txn"))
        .count();
    assert_ne!(txn_count, 0, "the number of Txn files should not be 0");

    oss.shutdown();
}

// This test verifies that even if a previous change set apply is aborted
// abnormally (e.g., abort by shard version not match), leaving tracked pending
// files behind, subsequent normal writes or change set applies will clean up
// these leftover tracked files during their own apply process, thus preventing
// memory leaks in unapplied_pending_files.
#[test]
fn test_no_pending_files_leak_when_abort_change_set_apply() {
    init_log_for_test();

    let node_id = alloc_node_id();
    let cluster = test_cloud_server::ServerCluster::new(vec![node_id], |_, cfg| {
        // Set a small write buffer size to trigger flush faster
        cfg.rocksdb.writecf.write_buffer_size = ReadableSize::kb(1);
    });

    // This test generates at least 2 changeset log, and 1 write log. We skip the
    // first two cleanups to ensure that at least one changeset log is aborted
    // early.
    fail::cfg("before_cleanup_unapplied_pending_files", "2*return").unwrap();

    let mut client = cluster.new_client();
    let shard_id = client.get_region_id(&i_to_key(0));
    // Will generate several flush change sets.
    client.put_kv(0..100, i_to_key, i_to_val);
    // Wait change set apply finish.
    thread::sleep(Duration::from_secs(1));

    let engine = cluster.get_kvengine(node_id);
    let shard = engine.get_shard(shard_id).unwrap();
    let pending_files = shard.get_unapplied_pending_files();
    assert!(
        pending_files.is_empty(),
        "pending files should be empty after cleanup"
    );
}

// This test verifies that even if a previous txn file apply is aborted
// abnormally, leaving tracked pending files behind, subsequent normal writes or
// txn file applies will clean up these leftover tracked files during their own
// apply process, thus preventing memory leaks in unapplied_pending_files.
#[test]
fn test_no_pending_files_leak_when_abort_txn_file_apply() {
    init_log_for_test();

    const KEYSPACE_ID: u32 = 10;
    let (_temp_dir, mut oss, dfs_config) = prepare_dfs("test");
    let pd_wrapper = PdWrapper::new_test(1, &SecurityConfig::default(), None);

    let node_id = alloc_node_id();
    let mut cluster = ServerClusterBuilder::new(vec![node_id], |_, cfg| {
        cfg.dfs = dfs_config.clone();
        cfg.raft_store.enable_inner_key_offset = true;
    })
    .pd(pd_wrapper)
    .build();
    cluster.start_tikv_workers(vec![alloc_node_id()], TikvWorkerOptions::default());
    cluster.wait_region_replicated(&[], 1);

    // Configure failpoint to abort the first cleanup by returning early.
    // This simulates a scenario where the first txn file apply is aborted,
    // leaving tracked pending files. When the subsequent txn file is applied,
    // it should clean up all pending files including those left by the first abort.
    fail::cfg("before_cleanup_unapplied_pending_files", "1*return").unwrap();

    let mut client = cluster.new_client_opt(ClusterClientOptions {
        txn_file_max_chunk_size: Some(1024),
        ..Default::default()
    });
    client.split_keyspace(KEYSPACE_ID);

    let shard_id = client.get_region_id(&i_to_key(0));
    let gen_key = i_to_keyspace_key(KEYSPACE_ID);
    // Will generate two txn file logs because the size exceeds the
    // `txn_file_max_chunk_size`.
    client
        .try_put_kv(
            0..100,
            &gen_key,
            |i: usize| format!("value_{:03}", i).into_bytes().repeat(3),
            MutateOptions {
                commit_action: CommitAction::AsyncCommitSecondaryKeys(Duration::ZERO),
                write_method: TxnWriteMethod::FileBased,
                ..Default::default()
            },
        )
        .unwrap();

    let engine = cluster.get_kvengine(node_id);
    let shard = engine.get_shard(shard_id).unwrap();
    let pending_files = shard.get_unapplied_pending_files();
    assert!(
        pending_files.is_empty(),
        "pending files should be empty after cleanup"
    );
    fail::remove("before_cleanup_unapplied_pending_files");
    oss.shutdown();
}
