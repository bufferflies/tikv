// Copyright 2025 TiKV Project Authors. Licensed under Apache-2.0.

use std::sync::Arc;

use pd_client::PdClient;
use test_rfstore::*;
use tikv_util::store::find_peer;

#[test]
fn test_rfstore_async_io_commit_without_leader_persist() {
    let mut cluster = new_node_cluster(2, 3);
    cluster.cfg.raft_store.max_apply_unpersisted_log_limit = 0;
    let pd_client = Arc::clone(&cluster.pd_client);
    pd_client.disable_default_operator();

    cluster.run();

    let region = pd_client.get_region(b"k1").unwrap();
    let peer_1 = find_peer(&region, 1).cloned().unwrap();

    cluster.must_put(b"k1", b"v1");
    cluster.must_transfer_leader(region.get_id(), peer_1);

    let raft_before_save_on_store_1_fp = "rfstore_before_save_on_store_1";
    fail::cfg(raft_before_save_on_store_1_fp, "pause").unwrap();

    for i in 2..10 {
        let _ = cluster
            .async_put(format!("k{}", i).as_bytes(), b"v1")
            .unwrap();
    }

    // Although leader can not persist entries, these entries can be committed
    must_get_equal(&cluster.get_engine(2), region.id, b"k9", b"v1");
    must_get_equal(&cluster.get_engine(3), region.id, b"k9", b"v1");
    // For now, entries must be applied after persisting
    must_get_none(&cluster.get_engine(1), region.id, b"k9");

    fail::remove(raft_before_save_on_store_1_fp);
    must_get_equal(&cluster.get_engine(3), region.id, b"k9", b"v1");
}

#[test]
fn test_rfstore_async_io_apply_without_leader_persist() {
    let mut cluster = new_node_cluster(0, 3);
    cluster.cfg.raft_store.cmd_batch_concurrent_ready_max_count = 0;
    cluster.cfg.raft_store.store_io_pool_size = 1;
    cluster.cfg.raft_store.max_apply_unpersisted_log_limit = 10000;
    let pd_client = Arc::clone(&cluster.pd_client);
    pd_client.disable_default_operator();

    cluster.run();

    let region = pd_client.get_region(b"k1").unwrap();
    let peer_1 = find_peer(&region, 1).cloned().unwrap();

    cluster.must_transfer_leader(region.get_id(), peer_1);
    cluster.must_put(b"k1", b"v1");

    let raft_before_save_on_store_1_fp = "rfstore_before_save_on_store_1";
    // Skip persisting to simulate raft log persist lag but not block node restart.
    fail::cfg(raft_before_save_on_store_1_fp, "return").unwrap();

    for i in 2..10 {
        let _ = cluster
            .async_put(format!("k{}", i).as_bytes(), b"v1")
            .unwrap();
    }

    // All node can apply these entries.
    for i in 1..=3 {
        must_get_equal(&cluster.get_engine(i), region.id, b"k9", b"v1");
    }

    cluster.stop_node(1);
    fail::remove(raft_before_save_on_store_1_fp);

    // Node 1 can recover successfully.
    cluster.run_node(1).unwrap();

    cluster.must_put(b"k1", b"v2");
    std::thread::sleep(std::time::Duration::from_millis(100));
    for i in 1..=3 {
        must_get_equal(&cluster.get_engine(i), region.id, b"k1", b"v2");
    }
}
