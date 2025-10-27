// Copyright 2025 TiKV Project Authors. Licensed under Apache-2.0.

use std::{sync::Arc, time::Duration};

use crossbeam::channel;
use kvproto::raft_serverpb::RaftMessage;
use raft::eraftpb::MessageType;
use test_raftstore::{new_peer, sleep_ms};
use test_rfstore::*;
use tikv_util::config::ReadableDuration;

#[test]
fn test_on_check_busy_on_apply_peers() {
    let mut cluster = new_node_cluster(0, 3);

    cluster.cfg.raft_store.raft_base_tick_interval = ReadableDuration::millis(5);
    cluster.cfg.raft_store.raft_store_max_leader_lease = ReadableDuration::millis(40);
    cluster.cfg.raft_store.leader_transfer_max_log_lag = 10;
    cluster.cfg.raft_store.check_long_uncommitted_interval = ReadableDuration::millis(10); // short check interval for recovery
    cluster.cfg.raft_store.pd_heartbeat_tick_interval = ReadableDuration::millis(50);

    let pd_client = Arc::clone(&cluster.pd_client);
    // Disable default max peer count check.
    pd_client.disable_default_operator();

    let r1 = cluster.run_conf_change();
    pd_client.must_add_peer(r1, new_peer(2, 1002));
    pd_client.must_add_peer(r1, new_peer(3, 1003));

    cluster.must_put(b"k1", b"v1");
    must_get_equal(&cluster.get_engine(2), r1, b"k1", b"v1");
    must_get_equal(&cluster.get_engine(3), r1, b"k1", b"v1");

    // Check the start status for peer 1003.
    cluster.must_send_store_heartbeat(3);
    sleep_ms(100);
    let stats = cluster.pd_client.get_store_stats(3).unwrap();
    assert!(!stats.is_busy);

    // Pause peer 1003 on applying logs to make it pending.
    let before_apply_stat = cluster.apply_state(r1, 3);
    cluster.stop_node(3);
    for i in 0..=cluster.cfg.raft_store.leader_transfer_max_log_lag {
        let bytes = format!("k{:03}", i).into_bytes();
        cluster.must_put(&bytes, &bytes);
    }
    cluster.must_put(b"k2", b"v2");
    must_get_equal(&cluster.get_engine(1), r1, b"k2", b"v2");
    must_get_equal(&cluster.get_engine(2), r1, b"k2", b"v2");

    // Restart peer 1003 and make it busy for applying pending logs.
    fail::cfg("on_handle_apply_1003", "pause").unwrap();
    // Case 1: check the leader committed index comes from MsgAppend and
    // MsgReadIndexResp is valid.
    let (read_tx, read_rx) = channel::unbounded::<RaftMessage>();
    let (append_tx, append_rx) = channel::unbounded::<RaftMessage>();
    cluster.add_send_filter_on_node(
        1,
        Box::new(
            RegionPacketFilter::new(r1, 1)
                .direction(Direction::Send)
                .msg_type(MessageType::MsgReadIndexResp)
                .set_msg_callback(Arc::new(move |msg: &RaftMessage| {
                    read_tx.send(msg.clone()).unwrap();
                })),
        ),
    );
    cluster.add_send_filter_on_node(
        1,
        Box::new(
            RegionPacketFilter::new(r1, 1)
                .direction(Direction::Send)
                .msg_type(MessageType::MsgAppend)
                .set_msg_callback(Arc::new(move |msg: &RaftMessage| {
                    append_tx.send(msg.clone()).unwrap();
                })),
        ),
    );
    let leader_apply_state = cluster.apply_state(r1, 1);
    cluster.run_node(3).unwrap();
    let append_msg = append_rx.recv_timeout(Duration::from_secs(2)).unwrap();
    assert_eq!(
        append_msg.get_message().get_commit(),
        leader_apply_state.applied_index
    );
    let read_msg = read_rx.recv_timeout(Duration::from_secs(2)).unwrap();
    assert_eq!(
        read_msg.get_message().get_index(),
        leader_apply_state.applied_index
    );
    cluster.clear_send_filter_on_node(1);

    // Case 2: completed regions < target count.
    let after_apply_stat = cluster.apply_state(r1, 3);
    assert!(after_apply_stat.applied_index == before_apply_stat.applied_index);
    sleep_ms(100);
    cluster.must_send_store_heartbeat(3);
    sleep_ms(100);
    let stats = cluster.pd_client.get_store_stats(3).unwrap();
    assert!(stats.is_busy);
    sleep_ms(100);

    // Case 3: completed_apply_peers_count > completed_target_count but
    //        there exists busy peers.
    fail::cfg("on_mock_store_completed_target_count", "return").unwrap();
    cluster.must_send_store_heartbeat(3);
    sleep_ms(100);
    let stats = cluster.pd_client.get_store_stats(3).unwrap();
    assert!(stats.is_busy);
    fail::remove("on_mock_store_completed_target_count");
    // After peer 1003 is recovered, store also should not be marked with busy.
    fail::remove("on_handle_apply_1003");
    sleep_ms(100);
    must_get_equal(&cluster.get_engine(3), r1, b"k2", b"v2");
    sleep_ms(100);
    let after_apply_stat = cluster.apply_state(r1, 3);
    assert!(after_apply_stat.applied_index > before_apply_stat.applied_index);
    cluster.must_send_store_heartbeat(3);
    sleep_ms(100);
    let stats = cluster.pd_client.get_store_stats(3).unwrap();
    assert!(!stats.is_busy);
}
