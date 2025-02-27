// Copyright 2025 TiKV Project Authors. Licensed under Apache-2.0.

use std::{iter::*, sync::*, thread, time::*};

use kvproto::{
    raft_cmdpb::CmdType,
    raft_serverpb::{PeerState, RaftMessage},
};
use pd_client::PdClient;
use raft::eraftpb::{ConfChangeType, MessageType};
use test_raftstore::{
    config_ignore_merge_target_integrity, configure_for_lease_read, configure_for_merge, find_peer,
    new_admin_request, new_get_cmd, new_prepare_merge, new_request, peer_on_store, sleep_ms,
};
use test_rfstore::{
    must_get_equal, must_get_none, new_node_cluster, shard_must_not_exist, CloneFilterFactory,
    Direction, IsolationFilterFactory, RegionPacketFilter, Simulator,
};
use tikv_util::{
    config::*,
    debug,
    store::{new_learner_peer, new_peer},
    HandyRwLock,
};
use txn_types::Key;

#[test]
fn test_node_merge_with_slow_learner() {
    test_util::init_log_for_test();
    let mut cluster = new_node_cluster(1, 2);
    configure_for_merge(&mut cluster.cfg);
    cluster.cfg.raft_store.raft_log_gc_size_limit = Some(ReadableSize(256));
    cluster.pd_client.disable_default_operator();

    // Create a cluster with peer 1 as leader and peer 2 as learner.
    let r1 = cluster.run_conf_change();
    let pd_client = Arc::clone(&cluster.pd_client);
    pd_client.must_add_peer(r1, new_learner_peer(2, 2));

    // Split the region.
    let pd_client = Arc::clone(&cluster.pd_client);
    let region = pd_client.get_region(b"k1").unwrap();
    let split_k2 = Key::from_raw(b"k2");
    cluster.must_split(&region, split_k2.as_encoded());
    let left = pd_client.get_region(b"k1").unwrap();
    let right = pd_client.get_region(b"k3").unwrap();
    assert_eq!(region.get_id(), right.get_id());
    assert_eq!(left.get_end_key(), right.get_start_key());
    assert_eq!(right.get_start_key(), split_k2.as_encoded());

    // Make sure the leader has received the learner's last index.
    cluster.must_put(b"k1", b"v1");
    cluster.must_put(b"k3", b"v3");
    must_get_equal(&cluster.get_engine(2), left.id, b"k1", b"v1");
    must_get_equal(&cluster.get_engine(2), right.id, b"k3", b"v3");

    cluster.add_send_filter(IsolationFilterFactory::new(2));
    (0..20).for_each(|i| cluster.must_put(b"k1", format!("v{}", i).as_bytes()));

    // Merge 2 regions under isolation should fail.
    let merge = new_prepare_merge(right.clone());
    let req = new_admin_request(left.get_id(), left.get_region_epoch(), merge);
    let resp = cluster
        .call_command_on_leader(req, Duration::from_secs(3))
        .unwrap();
    assert!(
        resp.get_header()
            .get_error()
            .get_message()
            .contains("log gap")
    );

    cluster.clear_send_filters();
    cluster.must_put(b"k11", b"v100");
    must_get_equal(&cluster.get_engine(1), left.id, b"k11", b"v100");
    must_get_equal(&cluster.get_engine(2), left.id, b"k11", b"v100");

    pd_client.must_merge(left.get_id(), right.get_id());

    // Test slow learner will be cleaned up when merge can't be continued.
    let region = pd_client.get_region(b"k1").unwrap();
    let split_k5 = Key::from_raw(b"k5");
    cluster.must_split(&region, split_k5.as_encoded());
    cluster.must_put(b"k4", b"v4");
    cluster.must_put(b"k6", b"v6");

    let left = pd_client.get_region(b"k1").unwrap();
    let right = pd_client.get_region(b"k6").unwrap();
    must_get_equal(&cluster.get_engine(2), left.id, b"k4", b"v4");
    must_get_equal(&cluster.get_engine(2), right.id, b"k6", b"v6");
    cluster.add_send_filter(IsolationFilterFactory::new(2));
    pd_client.must_merge(left.get_id(), right.get_id());

    let right_peer_1 = find_peer(&right, 1).unwrap();
    let state1 = cluster.truncated_state(right_peer_1.id, 1).unwrap();
    (0..50).for_each(|i| cluster.must_put(b"k2", format!("v{}", i).as_bytes()));

    // Wait to trigger compact raft log
    cluster.wait_log_truncated(right.get_id(), 1, state1.get_index() + 1);
    cluster.clear_send_filters();
    cluster.must_put(b"k7", b"v7");
    must_get_equal(&cluster.get_engine(2), right.id, b"k7", b"v7");
}

#[test]
fn test_node_merge_prerequisites_check() {
    let mut cluster = new_node_cluster(1, 3);
    configure_for_merge(&mut cluster.cfg);
    let pd_client = Arc::clone(&cluster.pd_client);

    cluster.run();

    cluster.must_put(b"k1", b"v1");
    cluster.must_put(b"k3", b"v3");

    let region = pd_client.get_region(b"k1").unwrap();
    let split_k2 = Key::from_raw(b"k2");
    cluster.must_split(&region, split_k2.as_encoded());
    let left = pd_client.get_region(b"k1").unwrap();
    let right = pd_client.get_region(b"k3").unwrap();
    let left_on_store1 = find_peer(&left, 1).unwrap().to_owned();
    cluster.must_transfer_leader(left.get_id(), left_on_store1);
    let right_on_store1 = find_peer(&right, 1).unwrap().to_owned();
    cluster.must_transfer_leader(right.get_id(), right_on_store1);

    // first MsgAppend will append log, second MsgAppend will set commit index,
    // So only allowing first MsgAppend to make source peer have uncommitted
    // entries.
    cluster.add_send_filter(CloneFilterFactory(
        RegionPacketFilter::new(left.get_id(), 3)
            .direction(Direction::Recv)
            .msg_type(MessageType::MsgAppend)
            .allow(1),
    ));
    // make the source peer's commit index can't be updated by MsgHeartbeat.
    cluster.add_send_filter(CloneFilterFactory(
        RegionPacketFilter::new(left.get_id(), 3)
            .msg_type(MessageType::MsgHeartbeat)
            .direction(Direction::Recv),
    ));
    let split_k11 = Key::from_raw(b"k11");
    cluster.must_split(&left, split_k11.as_encoded());
    let res = cluster.try_merge(left.get_id(), right.get_id());
    // log gap (min_committed, last_index] contains admin entries.
    assert!(res.get_header().has_error(), "{:?}", res);
    cluster.clear_send_filters();
    cluster.must_put(b"k22", b"v22");
    must_get_equal(&cluster.get_engine(3), right.id, b"k22", b"v22");

    cluster.add_send_filter(CloneFilterFactory(RegionPacketFilter::new(
        right.get_id(),
        3,
    )));
    // It doesn't matter if the index and term is correct.
    cluster.must_put(b"k23", b"v23");
    // v2 doesn't respond error.
    // assert!(res.get_header().has_error(), "{:?}", res);
    let res = cluster.try_merge(right.get_id(), left.get_id());
    // log gap (min_matched, last_index] contains admin entries.
    assert!(res.get_header().has_error(), "{:?}", res);
    cluster.clear_send_filters();
    cluster.must_put(b"k24", b"v24");
    must_get_equal(&cluster.get_engine(3), right.id, b"k24", b"v24");
}

/// Test if stale peer will be handled properly after merge.
#[test]
fn test_node_check_merged_message() {
    let mut cluster = new_node_cluster(1, 4);
    configure_for_merge(&mut cluster.cfg);
    config_ignore_merge_target_integrity(&mut cluster.cfg, &cluster.pd_client);
    let pd_client = Arc::clone(&cluster.pd_client);
    pd_client.disable_default_operator();

    cluster.run_conf_change();

    cluster.must_put(b"k1", b"v1");
    cluster.must_put(b"k3", b"v3");

    // test if stale peer before conf removal is destroyed automatically
    let mut region = pd_client.get_region(b"k1").unwrap();
    pd_client.must_add_peer(region.get_id(), new_peer(2, 2));
    pd_client.must_add_peer(region.get_id(), new_peer(3, 3));

    let split_k2 = Key::from_raw(b"k2");
    cluster.must_split(&region, split_k2.as_encoded());
    let mut left = pd_client.get_region(b"k1").unwrap();
    let mut right = pd_client.get_region(b"k3").unwrap();
    pd_client.must_add_peer(left.get_id(), new_peer(4, 4));
    must_get_equal(&cluster.get_engine(4), left.id, b"k1", b"v1");
    cluster.add_send_filter(IsolationFilterFactory::new(4));
    pd_client.must_remove_peer(left.get_id(), new_peer(4, 4));
    pd_client.must_merge(left.get_id(), right.get_id());
    cluster.clear_send_filters();
    shard_must_not_exist(&cluster.get_engine(4), left.id);

    println!("k1");

    // test gc work under complicated situation.
    cluster.must_put(b"k5", b"v5");
    region = pd_client.get_region(b"k2").unwrap();
    cluster.must_split(&region, split_k2.as_encoded());
    region = pd_client.get_region(b"k4").unwrap();

    let split_k4 = Key::from_raw(b"k4");
    cluster.must_split(&region, split_k4.as_encoded());
    left = pd_client.get_region(b"k1").unwrap();
    let middle = pd_client.get_region(b"k3").unwrap();
    let middle_on_store1 = find_peer(&middle, 1).unwrap().to_owned();
    cluster.must_transfer_leader(middle.get_id(), middle_on_store1);
    right = pd_client.get_region(b"k5").unwrap();
    let left_on_store3 = find_peer(&left, 3).unwrap().to_owned();
    pd_client.must_remove_peer(left.get_id(), left_on_store3);
    shard_must_not_exist(&cluster.get_engine(3), left.id);
    cluster.add_send_filter(IsolationFilterFactory::new(3));
    left = pd_client.get_region(b"k1").unwrap();
    pd_client.must_add_peer(left.get_id(), new_peer(3, 5));
    left = pd_client.get_region(b"k1").unwrap();
    pd_client.must_merge(middle.get_id(), left.get_id());
    pd_client.must_merge(right.get_id(), left.get_id());
    cluster.must_delete(b"k3");
    cluster.must_delete(b"k5");
    cluster.must_put(b"k6", b"v6");
    cluster.clear_send_filters();
    let engine3 = cluster.get_engine(3);
    must_get_equal(&engine3, left.id, b"k1", b"v1");
    must_get_equal(&engine3, left.id, b"k6", b"v6");
    must_get_none(&engine3, left.id, b"k3");
    must_get_none(&engine3, left.id, b"v5");
}

/// Test if an uninitialized stale peer will be handled properly after merge.
#[test]
// FIXME: after cherry-pick https://github.com/tikv/tikv/pull/15934
#[ignore]
fn test_node_gc_uninitialized_peer_after_merge() {
    test_util::init_log_for_test();
    let mut cluster = new_node_cluster(1, 4);
    configure_for_merge(&mut cluster.cfg);
    config_ignore_merge_target_integrity(&mut cluster.cfg, &cluster.pd_client);
    cluster.cfg.raft_store.raft_election_timeout_ticks = 5;
    cluster.cfg.raft_store.raft_store_max_leader_lease = ReadableDuration::millis(40);
    cluster.cfg.raft_store.max_leader_missing_duration = ReadableDuration::millis(150);
    cluster.cfg.raft_store.abnormal_leader_missing_duration = ReadableDuration::millis(100);
    cluster.cfg.raft_store.peer_stale_state_check_interval = ReadableDuration::millis(100);

    let pd_client = Arc::clone(&cluster.pd_client);
    pd_client.disable_default_operator();

    cluster.run_conf_change();

    cluster.must_put(b"k1", b"v1");
    cluster.must_put(b"k3", b"v3");

    // test if an uninitialized stale peer before conf removal is destroyed
    // automatically
    let region = pd_client.get_region(b"k1").unwrap();
    pd_client.must_add_peer(region.get_id(), new_peer(2, 2));
    pd_client.must_add_peer(region.get_id(), new_peer(3, 3));

    let split_k2 = Key::from_raw(b"k2");
    cluster.must_split(&region, split_k2.as_encoded());
    let left = pd_client.get_region(b"k1").unwrap();
    let right = pd_client.get_region(b"k3").unwrap();

    // Block snapshot messages, so that new peers will never be initialized.
    cluster.add_send_filter(CloneFilterFactory(
        RegionPacketFilter::new(left.get_id(), 4)
            .msg_type(MessageType::MsgSnapshot)
            .direction(Direction::Recv),
    ));
    // Add peer (4,4), remove peer (4,4) and then merge regions.
    // Peer (4,4) will be an an uninitialized stale peer.
    pd_client.must_add_peer(left.get_id(), new_peer(4, 4));
    cluster.must_region_exist(left.get_id(), 4);
    cluster.add_send_filter(IsolationFilterFactory::new(4));
    pd_client.must_remove_peer(left.get_id(), new_peer(4, 4));
    pd_client.must_merge(left.get_id(), right.get_id());
    cluster.clear_send_filters();

    // Wait for the peer (4,4) to be destroyed.
    sleep_ms(
        4 * cluster
            .cfg
            .raft_store
            .max_leader_missing_duration
            .as_millis(),
    );
    cluster.must_region_not_exist(left.get_id(), 4);
}

// Test if a merge handled properly when there is a unfinished slow split before
// merge.
#[test]
fn test_node_merge_slow_split() {
    fn imp(is_right_derive: bool) {
        let mut cluster = new_node_cluster(1, 3);
        configure_for_merge(&mut cluster.cfg);
        config_ignore_merge_target_integrity(&mut cluster.cfg, &cluster.pd_client);
        let pd_client = Arc::clone(&cluster.pd_client);
        pd_client.disable_default_operator();
        cluster.cfg.raft_store.right_derive_when_split = is_right_derive;

        cluster.run();

        cluster.must_put(b"k1", b"v1");
        cluster.must_put(b"k3", b"v3");

        let region = pd_client.get_region(b"k1").unwrap();
        let split_k2 = Key::from_raw(b"k2");
        cluster.must_split(&region, split_k2.as_encoded());
        let left = pd_client.get_region(b"k1").unwrap();
        let right = pd_client.get_region(b"k3").unwrap();

        let target_leader = right
            .get_peers()
            .iter()
            .find(|p| p.get_store_id() == 1)
            .unwrap()
            .clone();
        cluster.must_transfer_leader(right.get_id(), target_leader);
        let target_leader = left
            .get_peers()
            .iter()
            .find(|p| p.get_store_id() == 2)
            .unwrap()
            .clone();
        cluster.must_transfer_leader(left.get_id(), target_leader);
        must_get_equal(&cluster.get_engine(1), right.id, b"k3", b"v3");

        // So cluster becomes:
        //  left region: 1         2(leader) I 3
        // right region: 1(leader) 2         I 3
        // I means isolation.(here just means 3 can not receive append log)
        cluster.add_send_filter(CloneFilterFactory(
            RegionPacketFilter::new(left.get_id(), 3)
                .direction(Direction::Recv)
                .msg_type(MessageType::MsgAppend),
        ));
        cluster.add_send_filter(CloneFilterFactory(
            RegionPacketFilter::new(right.get_id(), 3)
                .direction(Direction::Recv)
                .msg_type(MessageType::MsgAppend),
        ));
        let split_k3 = Key::from_raw(b"k3");
        cluster.must_split(&right, split_k3.as_encoded());

        // left region and right region on store 3 fall behind
        // so after split, the new generated region is not on store 3 now
        let right1 = pd_client.get_region(b"k22").unwrap();
        let right2 = pd_client.get_region(b"k4").unwrap();
        assert_ne!(right1.get_id(), right2.get_id());
        pd_client.must_merge(left.get_id(), right1.get_id());
        // after merge, the left region still exists on store 3

        cluster.must_put(b"k0", b"v0");
        cluster.clear_send_filters();
        must_get_equal(&cluster.get_engine(3), right1.id, b"k0", b"v0");
    }
    imp(true);
    imp(false);
}

/// Test various cases that a store is isolated during merge.
#[test]
fn test_node_merge_dist_isolation() {
    let mut cluster = new_node_cluster(1, 3);
    configure_for_merge(&mut cluster.cfg);
    config_ignore_merge_target_integrity(&mut cluster.cfg, &cluster.pd_client);
    let pd_client = Arc::clone(&cluster.pd_client);
    pd_client.disable_default_operator();

    cluster.run();

    cluster.must_put(b"k1", b"v1");
    cluster.must_put(b"k3", b"v3");

    let region = pd_client.get_region(b"k1").unwrap();
    let split_k2 = Key::from_raw(b"k2");
    cluster.must_split(&region, split_k2.as_encoded());
    let left = pd_client.get_region(b"k1").unwrap();
    let right = pd_client.get_region(b"k3").unwrap();

    cluster.must_transfer_leader(right.get_id(), new_peer(1, 1));
    let target_leader = left
        .get_peers()
        .iter()
        .find(|p| p.get_store_id() == 3)
        .unwrap()
        .clone();
    cluster.must_transfer_leader(left.get_id(), target_leader);
    must_get_equal(&cluster.get_engine(1), right.id, b"k3", b"v3");

    // So cluster becomes:
    //  left region: 1         I 2 3(leader)
    // right region: 1(leader) I 2 3
    // I means isolation.
    cluster.add_send_filter(IsolationFilterFactory::new(1));
    pd_client.must_merge(left.get_id(), right.get_id());
    cluster.must_put(b"k4", b"v4");
    cluster.clear_send_filters();
    must_get_equal(&cluster.get_engine(1), right.id, b"k4", b"v4");

    let region = pd_client.get_region(b"k1").unwrap();
    cluster.must_split(&region, split_k2.as_encoded());
    let left = pd_client.get_region(b"k1").unwrap();
    let right = pd_client.get_region(b"k3").unwrap();

    cluster.must_put(b"k11", b"v11");
    pd_client.must_remove_peer(right.get_id(), new_peer(3, 3));
    cluster.must_put(b"k33", b"v33");

    cluster.add_send_filter(CloneFilterFactory(
        RegionPacketFilter::new(right.get_id(), 3).direction(Direction::Recv),
    ));
    pd_client.must_add_peer(right.get_id(), new_peer(3, 4));
    let right = pd_client.get_region(b"k3").unwrap();
    // So cluster becomes:
    //  left region: 1         2   3(leader)
    // right region: 1(leader) 2  [3]
    // [x] means a replica exists logically but is not created on the store x yet.
    let res = cluster.try_merge(region.get_id(), right.get_id());
    // Leader can't find replica 3 of right region, so it fails.
    assert!(res.get_header().has_error(), "{:?}", res);

    let target_leader = left
        .get_peers()
        .iter()
        .find(|p| p.get_store_id() == 2)
        .unwrap()
        .clone();
    cluster.must_transfer_leader(left.get_id(), target_leader);
    pd_client.must_merge(left.get_id(), right.get_id());
    cluster.must_put(b"k4", b"v4");

    cluster.clear_send_filters();
    must_get_equal(&cluster.get_engine(3), right.id, b"k4", b"v4");
}

/// Similar to `test_node_merge_dist_isolation`, but make the isolated store
/// way behind others so others have to send it a snapshot.
#[test]
fn test_node_merge_brain_split() {
    test_util::init_log_for_test();
    let mut cluster = new_node_cluster(1, 3);
    configure_for_merge(&mut cluster.cfg);
    config_ignore_merge_target_integrity(&mut cluster.cfg, &cluster.pd_client);
    cluster.cfg.raft_store.raft_log_gc_threshold = 12;
    cluster.cfg.raft_store.raft_log_gc_count_limit = Some(12);

    cluster.run();
    cluster.must_put(b"k1", b"v1");
    cluster.must_put(b"k3", b"v3");

    let pd_client = Arc::clone(&cluster.pd_client);
    pd_client.disable_default_operator();
    let region = pd_client.get_region(b"k1").unwrap();

    let split_k2 = Key::from_raw(b"k2");
    cluster.must_split(&region, split_k2.as_encoded());
    let left = pd_client.get_region(b"k1").unwrap();
    let right = pd_client.get_region(b"k3").unwrap();

    // The split regions' leaders could be at store 3, so transfer them to peer 1.
    let left_peer_1 = find_peer(&left, 1).cloned().unwrap();
    cluster.must_transfer_leader(left.get_id(), left_peer_1);
    let right_peer_1 = find_peer(&right, 1).cloned().unwrap();
    cluster.must_transfer_leader(right.get_id(), right_peer_1);

    cluster.must_put(b"k11", b"v11");
    cluster.must_put(b"k21", b"v21");
    // Make sure peers on store 3 have replicated latest update, which means
    // they have already reported their progresses to leader.
    must_get_equal(&cluster.get_engine(3), left.id, b"k11", b"v11");
    must_get_equal(&cluster.get_engine(3), right.id, b"k21", b"v21");

    cluster.add_send_filter(IsolationFilterFactory::new(3));
    // So cluster becomes:
    //  left region: 1(leader) 2 I 3
    // right region: 1(leader) 2 I 3
    // I means isolation.
    pd_client.must_merge(left.get_id(), right.get_id());

    for i in 0..100 {
        cluster.must_put(format!("k4{}", i).as_bytes(), b"v4");
    }
    must_get_equal(&cluster.get_engine(2), right.id, b"k40", b"v4");
    must_get_equal(&cluster.get_engine(1), right.id, b"k40", b"v4");

    cluster.clear_send_filters();

    // Wait until store 3 get data after merging
    must_get_equal(&cluster.get_engine(3), right.id, b"k40", b"v4");
    let right_peer_3 = find_peer(&right, 3).cloned().unwrap();
    cluster.must_transfer_leader(right.get_id(), right_peer_3);
    cluster.must_put(b"k40", b"v5");

    // Make sure the two regions are already merged on store 3.
    let left_peer_3 = peer_on_store(&left, 3);
    let state = cluster.region_local_state(left_peer_3.id, 3);
    assert!(state.is_none() || state.unwrap().get_state() == PeerState::Tombstone);
    must_get_equal(&cluster.get_engine(3), right.id, b"k40", b"v5");
    for i in 1..100 {
        must_get_equal(
            &cluster.get_engine(3),
            right.id,
            format!("k4{}", i).as_bytes(),
            b"v4",
        );
    }

    let region = pd_client.get_region(b"k1").unwrap();
    cluster.must_split(&region, split_k2.as_encoded());
    let region = pd_client.get_region(b"k22").unwrap();
    let split_k3 = Key::from_raw(b"k3");
    cluster.must_split(&region, split_k3.as_encoded());
    let middle = pd_client.get_region(b"k22").unwrap();
    let peer_on_store1 = find_peer(&middle, 1).unwrap().to_owned();
    cluster.must_transfer_leader(middle.get_id(), peer_on_store1);
    cluster.must_put(b"k22", b"v22");
    cluster.must_put(b"k33", b"v33");
    must_get_equal(&cluster.get_engine(3), region.id, b"k33", b"v33");
    let left = pd_client.get_region(b"k1").unwrap();
    let peer_on_left = find_peer(&left, 3).unwrap().to_owned();
    pd_client.must_remove_peer(left.get_id(), peer_on_left);
    let right = pd_client.get_region(b"k33").unwrap();
    let peer_on_right = find_peer(&right, 3).unwrap().to_owned();
    pd_client.must_remove_peer(right.get_id(), peer_on_right);
    shard_must_not_exist(&cluster.get_engine(3), left.id);
    must_get_equal(&cluster.get_engine(3), middle.id, b"k22", b"v22");
    shard_must_not_exist(&cluster.get_engine(3), right.id);
    cluster.add_send_filter(IsolationFilterFactory::new(3));
    pd_client.must_add_peer(left.get_id(), new_peer(3, 11));
    pd_client.must_merge(middle.get_id(), left.get_id());
    pd_client.must_remove_peer(left.get_id(), new_peer(3, 11));
    pd_client.must_merge(right.get_id(), left.get_id());
    pd_client.must_add_peer(left.get_id(), new_peer(3, 12));
    let region = pd_client.get_region(b"k1").unwrap();
    // So cluster becomes
    // store   3: k2 [middle] k3
    // store 1/2: [  new_left ] k4 [left]
    let split_k4 = Key::from_raw(b"k4");
    cluster.must_split(&region, split_k4.as_encoded());
    cluster.must_put(b"k12", b"v12");
    cluster.clear_send_filters();
    let left = pd_client.get_region(b"k1").unwrap();
    must_get_equal(&cluster.get_engine(3), left.id, b"k12", b"v12");
}

#[test]
fn test_node_merge_update_region() {
    let mut cluster = new_node_cluster(1, 3);
    configure_for_merge(&mut cluster.cfg);
    // Election timeout and max leader lease is 1s.
    configure_for_lease_read(&mut cluster.cfg, Some(100), Some(10));

    cluster.run();

    cluster.must_put(b"k1", b"v1");
    cluster.must_put(b"k3", b"v3");

    let pd_client = Arc::clone(&cluster.pd_client);
    let region = pd_client.get_region(b"k1").unwrap();
    let split_k2 = Key::from_raw(b"k2");
    cluster.must_split(&region, split_k2.as_encoded());
    let left = pd_client.get_region(b"k1").unwrap();
    let right = pd_client.get_region(b"k3").unwrap();

    // Make sure the leader is in lease.
    cluster.must_put(b"k1", b"v2");

    // "k3" is not in the range of left.
    let get = new_request(
        left.get_id(),
        left.get_region_epoch().clone(),
        vec![new_get_cmd(b"k3")],
        false,
    );
    debug!("requesting key not in range {:?}", get);
    let resp = cluster
        .call_command_on_leader(get, Duration::from_secs(5))
        .unwrap();
    assert!(resp.get_header().has_error(), "{:?}", resp);
    assert!(
        resp.get_header().get_error().has_key_not_in_region(),
        "{:?}",
        resp
    );

    // Merge right to left.
    pd_client.must_merge(right.get_id(), left.get_id());

    let origin_leader = cluster.leader_of_region(left.get_id()).unwrap();
    let new_leader = left
        .get_peers()
        .iter()
        .find(|&p| p.get_id() != origin_leader.get_id())
        .cloned()
        .unwrap();

    // Make sure merge is done in the new_leader.
    // There is only one region in the cluster, "k0" must belongs to it.
    cluster.must_put(b"k0", b"v0");
    must_get_equal(
        &cluster.get_engine(new_leader.get_store_id()),
        left.id,
        b"k0",
        b"v0",
    );

    // Transfer leadership to the new_leader.
    cluster.must_transfer_leader(left.get_id(), new_leader);

    // Make sure the leader is in lease.
    cluster.must_put(b"k0", b"v1");

    let new_region = pd_client.get_region(b"k2").unwrap();
    let get = new_request(
        new_region.get_id(),
        new_region.get_region_epoch().clone(),
        vec![new_get_cmd(b"k3")],
        false,
    );
    debug!("requesting {:?}", get);
    let resp = cluster
        .call_command_on_leader(get, Duration::from_secs(5))
        .unwrap();
    assert!(!resp.get_header().has_error(), "{:?}", resp);
    assert_eq!(resp.get_responses().len(), 1);
    assert_eq!(resp.get_responses()[0].get_cmd_type(), CmdType::Get);
    assert_eq!(resp.get_responses()[0].get_get().get_value(), b"v3");
}

#[test]
fn test_merge_with_slow_promote() {
    let mut cluster = new_node_cluster(1, 3);
    configure_for_merge(&mut cluster.cfg);
    let pd_client = Arc::clone(&cluster.pd_client);
    pd_client.disable_default_operator();

    let r1 = cluster.run_conf_change();
    pd_client.must_add_peer(r1, new_peer(2, 2));

    let region = pd_client.get_region(b"k1").unwrap();
    let split_k2 = Key::from_raw(b"k2");
    cluster.must_split(&region, split_k2.as_encoded());

    let left = pd_client.get_region(b"k1").unwrap();
    let right = pd_client.get_region(b"k3").unwrap();

    pd_client.must_add_peer(left.get_id(), new_peer(3, left.get_id() + 3));
    pd_client.must_add_peer(right.get_id(), new_learner_peer(3, right.get_id() + 3));

    cluster.must_put(b"k1", b"v1");
    cluster.must_put(b"k3", b"v3");
    must_get_equal(&cluster.get_engine(3), left.id, b"k1", b"v1");
    must_get_equal(&cluster.get_engine(3), right.id, b"k3", b"v3");

    let delay_filter =
        Box::new(RegionPacketFilter::new(right.get_id(), 3).direction(Direction::Recv));
    cluster.sim.wl().add_send_filter(3, delay_filter);

    pd_client.must_add_peer(right.get_id(), new_peer(3, right.get_id() + 3));
    pd_client.must_merge(right.get_id(), left.get_id());
    cluster.sim.wl().clear_send_filters(3);
    cluster.must_transfer_leader(left.get_id(), new_peer(3, left.get_id() + 3));
}

/// Test whether a isolated store recover properly if there is no target peer
/// on this store before isolated.
/// - A (-∞, k2), B [k2, +∞) on store 1,2,4
/// - store 4 is isolated
/// - B merge to A (target peer A is not created on store 4. It‘s just exist
/// logically)
/// - A split => C (-∞, k3), A [k3, +∞)
/// - Then network recovery
// No v2, it requires all peers to be available to check trim status.
#[test]
fn test_merge_isolated_store_with_no_target_peer() {
    let mut cluster = new_node_cluster(1, 4);
    configure_for_merge(&mut cluster.cfg);
    config_ignore_merge_target_integrity(&mut cluster.cfg, &cluster.pd_client);
    cluster.cfg.raft_store.right_derive_when_split = true;
    let pd_client = Arc::clone(&cluster.pd_client);
    pd_client.disable_default_operator();

    let r1 = cluster.run_conf_change();
    pd_client.must_add_peer(r1, new_peer(2, 2));
    pd_client.must_add_peer(r1, new_peer(3, 3));

    for i in 0..10 {
        cluster.must_put(format!("k{}", i).as_bytes(), b"v1");
    }

    let region = pd_client.get_region(b"k1").unwrap();
    // (-∞, k2), [k2, +∞)
    let split_k2 = Key::from_raw(b"k2");
    cluster.must_split(&region, split_k2.as_encoded());

    let left = pd_client.get_region(b"k1").unwrap();
    let right = pd_client.get_region(b"k3").unwrap();

    let left_on_store1 = find_peer(&left, 1).unwrap().to_owned();
    cluster.must_transfer_leader(left.get_id(), left_on_store1);
    let right_on_store1 = find_peer(&right, 1).unwrap().to_owned();
    cluster.must_transfer_leader(right.get_id(), right_on_store1);

    pd_client.must_add_peer(right.get_id(), new_peer(4, 4));
    let right_on_store3 = find_peer(&right, 3).unwrap().to_owned();
    pd_client.must_remove_peer(right.get_id(), right_on_store3);

    // Ensure snapshot is sent and applied.
    must_get_equal(&cluster.get_engine(4), right.id, b"k4", b"v1");
    cluster.must_put(b"k22", b"v22");
    // Ensure leader has updated its progress.
    must_get_equal(&cluster.get_engine(4), right.id, b"k22", b"v22");

    cluster.add_send_filter(IsolationFilterFactory::new(4));

    pd_client.must_add_peer(left.get_id(), new_peer(4, 5));
    let left_on_store3 = find_peer(&left, 3).unwrap().to_owned();
    pd_client.must_remove_peer(left.get_id(), left_on_store3);

    pd_client.must_merge(right.get_id(), left.get_id());

    let new_left = pd_client.get_region(b"k1").unwrap();
    // (-∞, k3), [k3, +∞)
    let split_k3 = Key::from_raw(b"k3");
    cluster.must_split(&new_left, split_k3.as_encoded());
    // Now new_left region range is [k3, +∞)
    cluster.must_put(b"k345", b"v345");
    cluster.clear_send_filters();

    must_get_equal(&cluster.get_engine(4), new_left.id, b"k345", b"v345");
}

/// Test whether a isolated peer can recover when two other regions merge to its
/// region.
#[test]
fn test_merge_cascade_merge_isolated() {
    let mut cluster = new_node_cluster(1, 3);
    configure_for_merge(&mut cluster.cfg);
    let pd_client = Arc::clone(&cluster.pd_client);
    pd_client.disable_default_operator();

    cluster.run();

    let mut region = pd_client.get_region(b"k1").unwrap();
    let split_k2 = Key::from_raw(b"k2");
    cluster.must_split(&region, split_k2.as_encoded());
    region = pd_client.get_region(b"k3").unwrap();
    let split_k3 = Key::from_raw(b"k3");
    cluster.must_split(&region, split_k3.as_encoded());

    cluster.must_put(b"k11", b"v11");
    cluster.must_put(b"k22", b"v22");
    cluster.must_put(b"k33", b"v33");

    let r1 = pd_client.get_region(b"k1").unwrap();
    let r2 = pd_client.get_region(b"k22").unwrap();
    let r3 = pd_client.get_region(b"k33").unwrap();

    must_get_equal(&cluster.get_engine(3), r1.id, b"k11", b"v11");
    must_get_equal(&cluster.get_engine(3), r2.id, b"k22", b"v22");
    must_get_equal(&cluster.get_engine(3), r3.id, b"k33", b"v33");

    let r1_on_store1 = find_peer(&r1, 1).unwrap().to_owned();
    cluster.must_transfer_leader(r1.get_id(), r1_on_store1);
    let r2_on_store2 = find_peer(&r2, 2).unwrap().to_owned();
    cluster.must_transfer_leader(r2.get_id(), r2_on_store2);
    let r3_on_store1 = find_peer(&r3, 1).unwrap().to_owned();
    cluster.must_transfer_leader(r3.get_id(), r3_on_store1);

    // Wait will all followers respond their progress.
    thread::sleep(Duration::from_millis(100));

    cluster.add_send_filter(IsolationFilterFactory::new(3));

    // r1, r3 both merge to r2
    pd_client.must_merge(r1.get_id(), r2.get_id());
    pd_client.must_merge(r3.get_id(), r2.get_id());

    cluster.must_put(b"k4", b"v4");

    cluster.clear_send_filters();

    must_get_equal(&cluster.get_engine(3), r2.id, b"k4", b"v4");
}

// Test if a learner can be destroyed properly when it's isolated and removed by
// conf change before its region merge to another region
#[test]
fn test_merge_isolated_not_in_merge_learner() {
    let mut cluster = new_node_cluster(1, 3);
    configure_for_merge(&mut cluster.cfg);
    let pd_client = Arc::clone(&cluster.pd_client);
    pd_client.disable_default_operator();

    cluster.run_conf_change();

    let region = pd_client.get_region(b"k1").unwrap();
    let split_k2 = Key::from_raw(b"k2");
    cluster.must_split(&region, split_k2.as_encoded());

    let left = pd_client.get_region(b"k1").unwrap();
    let right = pd_client.get_region(b"k3").unwrap();
    let left_on_store1 = find_peer(&left, 1).unwrap().to_owned();
    let right_on_store1 = find_peer(&right, 1).unwrap().to_owned();

    pd_client.must_add_peer(left.get_id(), new_learner_peer(2, 2));
    // Ensure this learner exists
    cluster.must_put(b"k1", b"v1");
    must_get_equal(&cluster.get_engine(2), left.id, b"k1", b"v1");

    cluster.stop_node(2);

    pd_client.must_remove_peer(left.get_id(), new_learner_peer(2, 2));

    pd_client.must_add_peer(left.get_id(), new_peer(3, 3));
    pd_client.must_remove_peer(left.get_id(), left_on_store1);

    pd_client.must_add_peer(right.get_id(), new_peer(3, 4));
    pd_client.must_remove_peer(right.get_id(), right_on_store1);

    pd_client.must_merge(left.get_id(), right.get_id());
    // Add a new learner on store 2 to trigger peer 2 send check-stale-peer msg to
    // other peers
    pd_client.must_add_peer(right.get_id(), new_learner_peer(2, 5));

    cluster.must_put(b"k123", b"v123");

    cluster.run_node(2).unwrap();
    // We can see if the old peer 2 is destroyed
    must_get_equal(&cluster.get_engine(2), right.id, b"k123", b"v123");
}

// Test if a learner can be destroyed properly when it's isolated and removed by
// conf change before another region merge to its region
#[test]
fn test_merge_isolated_stale_learner() {
    let mut cluster = new_node_cluster(1, 3);
    configure_for_merge(&mut cluster.cfg);
    cluster.cfg.raft_store.right_derive_when_split = true;
    // Do not rely on pd to remove stale peer
    cluster.cfg.raft_store.max_leader_missing_duration = ReadableDuration::hours(2);
    cluster.cfg.raft_store.abnormal_leader_missing_duration = ReadableDuration::minutes(10);
    cluster.cfg.raft_store.peer_stale_state_check_interval = ReadableDuration::minutes(5);
    let pd_client = Arc::clone(&cluster.pd_client);
    pd_client.disable_default_operator();

    cluster.run_conf_change();

    let mut region = pd_client.get_region(b"k1").unwrap();
    let split_k2 = Key::from_raw(b"k2");
    cluster.must_split(&region, split_k2.as_encoded());

    let left = pd_client.get_region(b"k1").unwrap();
    let right = pd_client.get_region(b"k3").unwrap();

    pd_client.must_add_peer(left.get_id(), new_learner_peer(2, 2));
    // Ensure this learner exists
    cluster.must_put(b"k1", b"v1");
    must_get_equal(&cluster.get_engine(2), left.id, b"k1", b"v1");

    cluster.stop_node(2);

    pd_client.must_remove_peer(left.get_id(), new_learner_peer(2, 2));

    pd_client.must_merge(right.get_id(), left.get_id());

    region = pd_client.get_region(b"k1").unwrap();
    cluster.must_split(&region, split_k2.as_encoded());

    let new_left = pd_client.get_region(b"k1").unwrap();
    assert_ne!(left.get_id(), new_left.get_id());
    // Add a new learner on store 2 to trigger peer 2 send check-stale-peer msg to
    // other peers
    pd_client.must_add_peer(new_left.get_id(), new_learner_peer(2, 5));
    cluster.must_put(b"k123", b"v123");

    cluster.run_node(2).unwrap();
    // We can see if the old peer 2 is destroyed
    must_get_equal(&cluster.get_engine(2), new_left.id, b"k123", b"v123");
}

/// Test if a learner can be destroyed properly in such conditions as follows
/// 1. A peer is isolated
/// 2. Be the last removed peer in its peer list
/// 3. Then its region merges to another region.
/// 4. Isolation disappears
#[test]
fn test_merge_isolated_not_in_merge_learner_2() {
    let mut cluster = new_node_cluster(1, 3);
    configure_for_merge(&mut cluster.cfg);
    cluster.cfg.raft_store.raft_election_timeout_ticks = 5;
    cluster.cfg.raft_store.raft_store_max_leader_lease = ReadableDuration::millis(40);
    cluster.cfg.raft_store.max_leader_missing_duration = ReadableDuration::millis(150);
    cluster.cfg.raft_store.abnormal_leader_missing_duration = ReadableDuration::millis(100);
    cluster.cfg.raft_store.peer_stale_state_check_interval = ReadableDuration::millis(100);

    let pd_client = Arc::clone(&cluster.pd_client);
    pd_client.disable_default_operator();

    cluster.run_conf_change();

    let region = pd_client.get_region(b"k1").unwrap();
    let split_k2 = Key::from_raw(b"k2");
    cluster.must_split(&region, split_k2.as_encoded());

    let left = pd_client.get_region(b"k1").unwrap();
    let right = pd_client.get_region(b"k3").unwrap();
    let left_on_store1 = find_peer(&left, 1).unwrap().to_owned();
    let right_on_store1 = find_peer(&right, 1).unwrap().to_owned();

    pd_client.must_add_peer(left.get_id(), new_learner_peer(2, 2));
    // Ensure this learner exists
    cluster.must_put(b"k1", b"v1");
    must_get_equal(&cluster.get_engine(2), left.id, b"k1", b"v1");

    cluster.stop_node(2);

    pd_client.must_add_peer(left.get_id(), new_peer(3, 3));
    pd_client.must_remove_peer(left.get_id(), left_on_store1);

    pd_client.must_add_peer(right.get_id(), new_peer(3, 4));
    pd_client.must_remove_peer(right.get_id(), right_on_store1);
    // The peer list of peer 2 is (1001, 1), (2, 2)
    pd_client.must_remove_peer(left.get_id(), new_learner_peer(2, 2));

    pd_client.must_merge(left.get_id(), right.get_id());

    cluster.run_node(2).unwrap();
    // When the abnormal leader missing duration has passed, the check-stale-peer
    // msg will be sent to peer 1001. After that, a new peer list will be
    // returned (2, 2) (3, 3). Then peer 2 sends the check-stale-peer msg to
    // peer 3 and it will get a tombstone response. Finally peer 2 will be
    // destroyed.
    shard_must_not_exist(&cluster.get_engine(2), left.id);
}

/// Test if a peer can be removed if its target peer has been removed and
/// doesn't apply the CommitMerge log.
#[test]
fn test_merge_remove_target_peer_isolated() {
    let mut cluster = new_node_cluster(1, 4);
    configure_for_merge(&mut cluster.cfg);
    let pd_client = Arc::clone(&cluster.pd_client);
    pd_client.disable_default_operator();

    cluster.run_conf_change();

    let mut region = pd_client.get_region(b"k1").unwrap();
    pd_client.must_add_peer(region.get_id(), new_peer(2, 2));
    pd_client.must_add_peer(region.get_id(), new_peer(3, 3));

    let split_k2 = Key::from_raw(b"k2");
    cluster.must_split(&region, split_k2.as_encoded());
    region = pd_client.get_region(b"k3").unwrap();
    let split_k4 = Key::from_raw(b"k4");
    cluster.must_split(&region, split_k4.as_encoded());

    let r1 = pd_client.get_region(b"k1").unwrap();
    let r2 = pd_client.get_region(b"k3").unwrap();
    let r3 = pd_client.get_region(b"k5").unwrap();

    let r1_on_store1 = find_peer(&r1, 1).unwrap().to_owned();
    cluster.must_transfer_leader(r1.get_id(), r1_on_store1);
    let r2_on_store2 = find_peer(&r2, 2).unwrap().to_owned();
    cluster.must_transfer_leader(r2.get_id(), r2_on_store2);

    for i in [1, 3, 5] {
        cluster.must_put(format!("k{}", i).as_bytes(), b"v1");
    }

    must_get_equal(&cluster.get_engine(3), r1.id, b"k1", b"v1");
    must_get_equal(&cluster.get_engine(3), r2.id, b"k3", b"v1");
    must_get_equal(&cluster.get_engine(3), r3.id, b"k5", b"v1");

    cluster.add_send_filter(IsolationFilterFactory::new(3));
    // Make region r2's epoch > r2 peer on store 3.
    // r2 peer on store 3 will be removed whose epoch is staler than the epoch when
    // r1 merge to r2.
    pd_client.must_add_peer(r2.get_id(), new_peer(4, 4));
    pd_client.must_remove_peer(r2.get_id(), new_peer(4, 4));

    let r2_on_store3 = find_peer(&r2, 3).unwrap().to_owned();
    let r3_on_store3 = find_peer(&r3, 3).unwrap().to_owned();

    pd_client.must_merge(r1.get_id(), r2.get_id());

    pd_client.must_remove_peer(r2.get_id(), r2_on_store3);
    pd_client.must_remove_peer(r3.get_id(), r3_on_store3);

    pd_client.must_merge(r2.get_id(), r3.get_id());

    cluster.clear_send_filters();

    for i in [r1.id, r2.id, r3.id] {
        shard_must_not_exist(&cluster.get_engine(3), i);
    }
}

/// If a follower is demoted by a snapshot, its meta will be changed. The case
/// is to ensure asserts in code can tolerate the change.
#[test]
fn test_merge_snapshot_demote() {
    test_util::init_log_for_test();
    let mut cluster = new_node_cluster(1, 4);
    configure_for_merge(&mut cluster.cfg);
    cluster.cfg.raft_store.raft_log_gc_tick_interval = ReadableDuration::millis(20);
    cluster.cfg.raft_store.raft_log_gc_size_limit = Some(ReadableSize(1));
    // trigger flush on write to avoid blocking merge due to "over bound data"
    cluster.cfg.rocksdb.writecf.write_buffer_size = ReadableSize(1);
    let pd_client = Arc::clone(&cluster.pd_client);
    pd_client.disable_default_operator();

    cluster.run_conf_change();

    let region = pd_client.get_region(b"k1").unwrap();
    pd_client.must_add_peer(region.get_id(), new_peer(2, 2));
    pd_client.must_add_peer(region.get_id(), new_peer(3, 3));

    let split_k2 = Key::from_raw(b"k2");
    cluster.must_split(&region, split_k2.as_encoded());

    let r1 = pd_client.get_region(b"k1").unwrap();
    let r2 = pd_client.get_region(b"k3").unwrap();

    let r2_on_store1 = find_peer(&r2, 1).unwrap().to_owned();
    cluster.must_transfer_leader(r2.get_id(), r2_on_store1);

    // So r2 on store 3 will lag behind.
    cluster.add_send_filter(CloneFilterFactory(
        RegionPacketFilter::new(r2.get_id(), 3)
            .direction(Direction::Recv)
            .msg_type(MessageType::MsgAppend),
    ));

    let peer1_id = find_peer(&r2, 1).unwrap().id;
    let last_index = cluster.raft_state(peer1_id, 1).unwrap().get_last_index();
    for i in 1..4 {
        cluster.must_put(format!("k{}", i).as_bytes(), b"v1");
    }

    pd_client.must_merge(r1.get_id(), r2.get_id());
    cluster.wait_log_truncated(r2.get_id(), 1, last_index + 1);

    // Now demote r2 on store 3 to learner, so its meta will be changed.
    let r2_on_store3 = find_peer(&r2, 3).unwrap().to_owned();
    pd_client.must_joint_confchange(
        r2.get_id(),
        vec![
            (ConfChangeType::AddLearnerNode, new_learner_peer(4, 4)),
            (
                ConfChangeType::AddLearnerNode,
                new_learner_peer(3, r2_on_store3.get_id()),
            ),
        ],
    );

    cluster.clear_send_filters();
    // Now snapshot should be generated and merge on store 3 should be aborted.
    cluster.must_put(b"k4", b"v4");
    must_get_equal(&cluster.get_engine(3), r2.id, b"k4", b"v4");
}

/// Check if merge is cleaned up if the merge target is destroyed several times
/// before it's ever scheduled.
#[test]
fn test_node_merge_long_isolated() {
    test_util::init_log_for_test();
    let mut cluster = new_node_cluster(1, 3);
    configure_for_merge(&mut cluster.cfg);
    config_ignore_merge_target_integrity(&mut cluster.cfg, &cluster.pd_client);
    cluster.cfg.raft_store.max_leader_missing_duration = ReadableDuration::millis(150);
    let pd_client = Arc::clone(&cluster.pd_client);
    pd_client.disable_default_operator();

    cluster.run();

    cluster.must_put(b"k1", b"v1");
    cluster.must_put(b"k3", b"v3");

    let region = pd_client.get_region(b"k1").unwrap();
    let split_k2 = Key::from_raw(b"k2");
    cluster.must_split(&region, split_k2.as_encoded());
    let left = pd_client.get_region(b"k1").unwrap();
    let right = pd_client.get_region(b"k3").unwrap();

    cluster.must_transfer_leader(right.get_id(), new_peer(3, 3));
    let left_leader = peer_on_store(&left, 3);
    cluster.must_transfer_leader(left.get_id(), left_leader);
    must_get_equal(&cluster.get_engine(1), right.id, b"k3", b"v3");

    // So cluster becomes:
    //  left region: 1 I 2 3(leader)
    // right region: 1 I 2 3(leader)
    // I means isolation.
    cluster.add_send_filter(IsolationFilterFactory::new(1));
    pd_client.must_merge(left.get_id(), right.get_id());
    pd_client.must_remove_peer(right.get_id(), peer_on_store(&right, 1));
    // Split to make sure the range of new peer won't overlap with source.
    let right = pd_client.get_region(b"k1").unwrap();
    cluster.must_split(&right, split_k2.as_encoded());
    cluster.must_put(b"k4", b"v4");
    // Ensure the node is removed, so it will not catch up any logs but just destroy
    // itself.
    must_get_equal(&cluster.get_engine(3), right.id, b"k4", b"v4");
    must_get_equal(&cluster.get_engine(2), right.id, b"k4", b"v4");

    let filter = RegionPacketFilter::new(left.get_id(), 1);
    cluster.clear_send_filters();
    // Ensure source region will not take any actions.
    cluster.add_send_filter(CloneFilterFactory(filter));
    shard_must_not_exist(&cluster.get_engine(1), right.id);
    must_get_equal(&cluster.get_engine(1), left.id, b"k1", b"v1");

    // So new peer will not apply snapshot.
    let filter = RegionPacketFilter::new(right.get_id(), 1).msg_type(MessageType::MsgSnapshot);
    cluster.add_send_filter(CloneFilterFactory(filter));
    pd_client.must_add_peer(right.get_id(), new_peer(1, 1010));
    cluster.must_put(b"k5", b"v5");
    must_get_equal(&cluster.get_engine(2), right.id, b"k5", b"v5");
    shard_must_not_exist(&cluster.get_engine(1), right.id);

    // Now peer(1, 1010) should probably created in memory but not persisted.
    pd_client.must_remove_peer(right.get_id(), new_peer(1, 1010));
    cluster.clear_send_filters();
    cluster.wait_tombstone(new_peer(1, 1010), true);
}

#[test]
fn test_stale_message_after_merge() {
    let mut cluster = new_node_cluster(1, 3);
    configure_for_merge(&mut cluster.cfg);
    cluster.run();
    let pd_client = Arc::clone(&cluster.pd_client);
    pd_client.disable_default_operator();

    cluster.must_transfer_leader(1, new_peer(1, 1));

    cluster.must_put(b"k1", b"v1");
    cluster.must_put(b"k3", b"v3");

    let region = cluster.get_region(b"k1");
    let split_k2 = Key::from_raw(b"k2");
    cluster.must_split(&region, split_k2.as_encoded());
    let left = cluster.get_region(b"k1");
    let right = cluster.get_region(b"k3");

    pd_client.must_remove_peer(left.get_id(), find_peer(&left, 3).unwrap().to_owned());
    pd_client.must_add_peer(left.get_id(), new_peer(3, 1004));
    pd_client.must_merge(left.get_id(), right.get_id());

    // Such stale message can be sent due to network error, consider the following
    // example:
    // - Store 1 and Store 3 can't reach each other, so peer 1003
    // start election and send `RequestVote` message to peer 1001, and fail
    // due to network error, but this message is keep backoff-retry to send out
    // - Peer 1002 become the new leader and remove peer 1003 and add peer 1004 on
    // store 3, then the region is merged into other region, the merge can
    // success because peer 1002 can reach both peer 1001 and peer 1004
    // - Network recover, so peer 1003's `RequestVote` message is sent to peer 1001
    // after it is merged
    //
    // the backoff-retry of a stale message is hard to simulated in test, so here
    // just send this stale message directly
    let mut raft_msg = RaftMessage::default();
    raft_msg.set_region_id(left.get_id());
    raft_msg.set_from_peer(find_peer(&left, 3).unwrap().to_owned());
    raft_msg.set_to_peer(find_peer(&left, 1).unwrap().to_owned());
    raft_msg.set_region_epoch(left.get_region_epoch().to_owned());
    cluster.send_raft_msg(raft_msg).unwrap();

    cluster.must_put(b"k4", b"v4");
    must_get_equal(&cluster.get_engine(3), right.id, b"k4", b"v4");
}

/// Check whether merge should be prevented if follower may not have enough
/// logs.
#[test]
fn test_prepare_merge_with_reset_matched() {
    let mut cluster = new_node_cluster(1, 3);
    configure_for_merge(&mut cluster.cfg);
    let pd_client = Arc::clone(&cluster.pd_client);
    pd_client.disable_default_operator();
    let r = cluster.run_conf_change();
    pd_client.must_add_peer(r, new_peer(2, 2));
    cluster.add_send_filter(IsolationFilterFactory::new(3));
    pd_client.add_peer(r, new_peer(3, 3));

    cluster.must_put(b"k1", b"v1");
    cluster.must_put(b"k3", b"v3");

    let region = cluster.get_region(b"k1");
    let split_k2 = Key::from_raw(b"k2");
    cluster.must_split(&region, split_k2.as_encoded());
    let left = cluster.get_region(b"k1");
    let right = cluster.get_region(b"k3");
    thread::sleep(Duration::from_millis(10));
    // So leader will replicate next command but can't know whether follower (2, 2)
    // also commits the command. Supposing the index is i0.
    cluster.add_send_filter(CloneFilterFactory(
        RegionPacketFilter::new(left.get_id(), 2)
            .direction(Direction::Recv)
            .msg_type(MessageType::MsgAppendResponse)
            .allow(1),
    ));
    cluster.must_put(b"k11", b"v11");
    cluster.clear_send_filters();
    cluster.add_send_filter(IsolationFilterFactory::new(2));
    // So peer (3, 3) only have logs after i0.
    must_get_equal(&cluster.get_engine(3), left.id, b"k11", b"v11");
    // Clear match information.
    let left_on_store3 = find_peer(&left, 3).unwrap().to_owned();
    cluster.must_transfer_leader(left.get_id(), left_on_store3);
    let left_on_store1 = find_peer(&left, 1).unwrap().to_owned();
    cluster.must_transfer_leader(left.get_id(), left_on_store1);
    let res = cluster.try_merge(left.get_id(), right.get_id());
    // Now leader still knows peer(2, 2) has committed i0 - 1, so the min_match will
    // become i0 - 1. But i0 - 1 is not a safe index as peer(3, 3) starts from i0 +
    // 1.
    assert!(res.get_header().has_error(), "{:?}", res);
    cluster.clear_send_filters();
    // Now leader should replicate more logs and figure out a safe index.
    pd_client.must_merge(left.get_id(), right.get_id());
}

/// Check if prepare merge min index is chosen correctly even if all match
/// indexes are correct.
#[test]
fn test_prepare_merge_with_5_nodes_snapshot() {
    let mut cluster = new_node_cluster(1, 5);
    configure_for_merge(&mut cluster.cfg);
    let pd_client = Arc::clone(&cluster.pd_client);
    pd_client.disable_default_operator();
    cluster.run();
    cluster.must_put(b"k1", b"v1");
    cluster.must_put(b"k3", b"v3");

    let region = cluster.get_region(b"k1");
    let split_k2 = Key::from_raw(b"k2");
    cluster.must_split(&region, split_k2.as_encoded());
    let left = cluster.get_region(b"k1");
    let right = cluster.get_region(b"k3");

    let peer_on_store1 = find_peer(&left, 1).unwrap().clone();
    cluster.must_transfer_leader(left.get_id(), peer_on_store1);
    must_get_equal(&cluster.get_engine(5), left.id, b"k1", b"v1");
    let peer_on_store5 = find_peer(&left, 5).unwrap().clone();
    pd_client.must_remove_peer(left.get_id(), peer_on_store5);
    shard_must_not_exist(&cluster.get_engine(5), left.id);
    cluster.add_send_filter(IsolationFilterFactory::new(5));
    pd_client.add_peer(left.get_id(), new_peer(5, 16));

    // Make sure there will be no admin entries after min_matched.
    for (k, v) in [(b"k11", b"v11"), (b"k12", b"v12")] {
        cluster.must_put(k, v);
        must_get_equal(&cluster.get_engine(4), left.id, k, v);
    }
    cluster.add_send_filter(IsolationFilterFactory::new(4));
    // So index of peer 4 becomes min_matched.
    cluster.must_put(b"k13", b"v13");
    must_get_equal(&cluster.get_engine(1), left.id, b"k13", b"v13");

    // Only remove send filter on store 5.
    cluster.clear_send_filters();
    cluster.add_send_filter(IsolationFilterFactory::new(4));
    must_get_equal(&cluster.get_engine(5), left.id, b"k13", b"v13");
    let res = cluster.try_merge(left.get_id(), right.get_id());
    // min_matched from peer 4 is beyond the first index of peer 5, it should not be
    // chosen for prepare merge.
    assert!(res.get_header().has_error(), "{:?}", res);
    cluster.clear_send_filters();
    // Now leader should replicate more logs and figure out a safe index.
    pd_client.must_merge(left.get_id(), right.get_id());
}
