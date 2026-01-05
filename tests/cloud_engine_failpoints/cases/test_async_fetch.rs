// Copyright 2025 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    collections::HashMap,
    sync::{mpsc, Mutex},
    thread::sleep,
    time::Duration,
};

use pd_client::PdClient;
use rfstore::store::{state::RaftTruncatedState, RAFT_INIT_LOG_INDEX};
use test_rfstore::{must_get_equal, new_node_cluster};
use tikv_util::{
    config::*,
    store::{find_peer, new_peer},
};

// Test the case that cache is compacted when one node down to check
// if it goes well after the node is back online which triggers async fetch.
// This test mirrors the failpoints/test_node_async_fetch test.
#[test]
fn test_node_async_fetch() {
    let mut cluster = new_node_cluster(0, 3);
    cluster.cfg.raft_store.raft_log_gc_count_limit = Some(100000);
    cluster.cfg.raft_store.raft_log_gc_threshold = 200;
    cluster.cfg.raft_store.raft_log_gc_size_limit = Some(ReadableSize::mb(20));
    cluster.cfg.rfengine.target_file_size = ReadableSize::kb(1);
    cluster.cfg.rfengine.rlog_soft_memory_limit = ReadableSize::kb(4);

    cluster.run();

    // Initial data
    cluster.must_put(b"k1", b"v1");
    let region = cluster.pd_client.get_region(b"k1").unwrap();

    // Stop one node to cause log lag
    cluster.stop_node(1);

    // Generate more entries while node is down - equivalent to the loop in original
    // test
    for i in 2..60usize {
        let k = format!("k{}", i);
        let v = format!("v{}", i);
        cluster.must_put(k.as_bytes(), v.as_bytes());
    }

    // Set up callback to detect async fetch completion
    let (sender, receiver) = mpsc::channel();
    let sync_sender = Mutex::new(sender);
    fail::cfg_callback("on_async_fetch_return", move || {
        let sender = sync_sender.lock().unwrap();
        let _ = sender.send(true);
    })
    .unwrap();

    // Restart the stopped node
    cluster.run_node(1).unwrap();

    // Wait for node to rejoin
    std::thread::sleep(Duration::from_millis(100));

    // Wait for async fetch to complete
    assert_eq!(
        receiver.recv_timeout(Duration::from_millis(2000)).unwrap(),
        true
    );

    // logs should be replicated to node 1 successfully.
    let engine = cluster.get_engine(1);
    // Verify data written while node 1 was down
    for i in 2..60usize {
        let k = format!("k{}", i);
        let v = format!("v{}", i);
        must_get_equal(&engine, region.get_id(), k.as_bytes(), v.as_bytes());
    }

    fail::remove("on_async_fetch_return");
}

#[test]
fn test_persist_delay_block_log_compaction() {
    let mut cluster = new_node_cluster(0, 3);

    cluster.cfg.raft_store.cmd_batch_concurrent_ready_max_count = 0;
    cluster.cfg.raft_store.store_io_pool_size = 1;
    cluster.cfg.raft_store.max_apply_unpersisted_log_limit = 10000;

    cluster.cfg.raft_store.raft_log_gc_count_limit = Some(100000);
    cluster.cfg.raft_store.raft_log_gc_threshold = 50;
    cluster.cfg.raft_store.raft_log_gc_size_limit = Some(ReadableSize::mb(20));
    cluster.cfg.raft_store.raft_log_gc_tick_interval = ReadableDuration::millis(50);
    cluster.cfg.raft_store.raft_log_reserve_max_ticks = 2;
    cluster.cfg.raft_store.raft_entry_cache_life_time = ReadableDuration::millis(100);

    cluster.pd_client.disable_default_operator();

    let r1 = cluster.run_conf_change();
    cluster.pd_client.must_add_peer(r1, new_peer(2, 2));
    cluster.pd_client.must_add_peer(r1, new_peer(3, 3));

    cluster.must_put(b"k1", b"v1");

    let region = cluster.pd_client.get_region(b"k1").unwrap();
    let peer_1 = find_peer(&region, 1).cloned().unwrap();
    cluster.must_transfer_leader(region.get_id(), peer_1.clone());

    let raft_before_save_on_store_1_fp = "rfstore_before_save_on_store_1";

    for i in 0..100 {
        let k = format!("k{}", i).into_bytes();
        let v = "v1".as_bytes().to_owned();
        cluster.must_put(&k, &v);
    }
    // Wait log gc.
    sleep(Duration::from_millis(200));

    // Wait for the truncate state to be prepared.
    test_util::eventually(Duration::from_millis(200), Duration::from_secs(2), || {
        let mut initialized = 0;
        for id in cluster.engines.keys() {
            let peer = find_peer(&region, *id).unwrap();
            if cluster
                .truncated_state(peer.get_id(), peer.store_id)
                .is_some()
            {
                initialized += 1;
            }
        }
        initialized == cluster.engines.len()
    });

    let mut before_states: HashMap<u64, RaftTruncatedState> = HashMap::default();
    for (&id, engines) in &cluster.engines {
        must_get_equal(&engines.kv, region.id, b"k1", b"v1");
        let peer = find_peer(&region, id).unwrap();
        let state = cluster
            .truncated_state(peer.get_id(), peer.store_id)
            .unwrap();
        // Should trigger compact.
        assert!(state.get_index() > RAFT_INIT_LOG_INDEX);
        before_states.insert(id, state);
    }

    // Skip persisting to simulate raft log persist lag but not block node restart.
    fail::cfg(raft_before_save_on_store_1_fp, "pause").unwrap();

    for i in 0..100 {
        let k = format!("k{}", i).into_bytes();
        let v = "v2".as_bytes().to_owned();
        cluster.must_put(&k, &v);
    }
    for i in 0..100 {
        let k = format!("k{}", i).into_bytes();
        must_get_equal(&cluster.engines[&1].kv, region.id, &k, "v2".as_bytes());
    }

    // Wait log gc.
    sleep(Duration::from_millis(100));
    // Log perisist is block, should not trigger log gc for peer 1, but the others
    // should keep going.
    let mut others_truncated_index = 0_u64;
    let mut paused_truncated_index = 0_u64;
    for id in cluster.engines.keys() {
        let peer = find_peer(&region, *id).unwrap();
        let after_state = cluster
            .truncated_state(peer.get_id(), peer.store_id)
            .unwrap();
        let before_state = &before_states[id];
        if *id == 1 {
            assert!(after_state.get_index() >= before_state.get_index() + 10);
            paused_truncated_index = after_state.get_index();
        } else {
            assert!(after_state.get_index() >= before_state.get_index() + 100);
            others_truncated_index = after_state.get_index();
        }
    }
    assert!(others_truncated_index >= paused_truncated_index + 50);

    fail::remove(raft_before_save_on_store_1_fp);

    // Wait log persist and trigger gc.
    sleep(Duration::from_millis(200));

    // Log perisist is block, should not trigger log gc.
    for id in cluster.engines.keys() {
        let peer = find_peer(&region, *id).unwrap();
        let after_state = cluster
            .truncated_state(peer.get_id(), peer.store_id)
            .unwrap();
        let before_state = &before_states[id];
        assert!(after_state.get_index() > before_state.get_index() + 100);
    }
}
