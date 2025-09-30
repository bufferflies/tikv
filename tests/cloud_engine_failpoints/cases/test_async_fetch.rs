// Copyright 2025 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    sync::{mpsc, Mutex},
    time::Duration,
};

use pd_client::PdClient;
use test_rfstore::{must_get_equal, new_node_cluster};
use tikv_util::config::*;

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
