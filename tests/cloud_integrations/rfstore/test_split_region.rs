// Copyright 2025 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    sync::{mpsc::channel, Arc},
    thread,
    time::Duration,
};

use kvproto::{raft_cmdpb::*, raft_serverpb::RaftMessage};
use pd_client::PdClient;
use raft::eraftpb::MessageType;
use rfstore::{
    store::{Callback, WriteResponse},
    Result,
};
use test_raftstore::{
    configure_for_lease_read, find_peer, new_admin_request, new_get_cmd, new_peer, new_request,
    peer_on_store, sleep_ms,
};
use test_rfstore::*;
use tikv_util::{config::*, debug};
use txn_types::Key;

#[test]
fn test_server_base_split_region() {
    let count = 5;
    let mut cluster = new_node_cluster(1, count);
    cluster.run();

    let pd_client = Arc::clone(&cluster.pd_client);

    let tbls = vec![
        (b"k22", b"k11", b"k33"),
        (b"k11", b"k00", b"k11"),
        (b"k33", b"k22", b"k33"),
    ];

    for (split_key, left_key, right_key) in tbls {
        let split_key = Key::from_raw(split_key).into_encoded();
        let left_key = Key::from_raw(left_key).into_encoded();
        let right_key = Key::from_raw(right_key).into_encoded();
        cluster.must_put(&left_key, b"v1");
        cluster.must_put(&right_key, b"v3");

        // Left and right key must be in same region before split.
        let region = pd_client.get_region(&left_key).unwrap();
        let region2 = pd_client.get_region(&right_key).unwrap();
        assert_eq!(region.get_id(), region2.get_id());

        // Split with split_key, so left_key must in left, and right_key in right.
        cluster.must_split(&region, &split_key);

        let left = pd_client.get_region(&left_key).unwrap();
        let right = pd_client.get_region(&right_key).unwrap();

        assert_eq!(region.get_id(), right.get_id());
        assert_eq!(region.get_start_key(), left.get_start_key());
        assert_eq!(left.get_end_key(), right.get_start_key());
        assert_eq!(region.get_end_key(), right.get_end_key());

        cluster.must_put(&left_key, b"vv1");
        assert_eq!(cluster.get(&left_key).unwrap(), b"vv1".to_vec());

        cluster.must_put(&right_key, b"vv3");
        assert_eq!(cluster.get(&right_key).unwrap(), b"vv3".to_vec());

        let epoch = left.get_region_epoch().clone();
        let get = new_request(left.get_id(), epoch, vec![new_get_cmd(&right_key)], false);
        debug!("requesting {:?}", get);
        let resp = cluster
            .call_command_on_leader(get, Duration::from_secs(5))
            .unwrap();
        assert!(resp.get_header().has_error(), "{:?}", resp);
        assert!(
            resp.get_header().get_error().has_key_not_in_region(),
            "{:?}",
            resp
        );
    }
}

#[test]
fn test_server_split_region_twice() {
    let count = 5;
    let mut cluster = new_node_cluster(1, count);
    cluster.run();
    let pd_client = Arc::clone(&cluster.pd_client);

    let (left_key, right_key) = (b"k11", b"k33");
    let split_key = Key::from_raw(b"k22").into_encoded();
    cluster.must_put(left_key, b"v1");
    cluster.must_put(right_key, b"v3");

    // Left and right key must be in same region before split.
    let region = pd_client.get_region(left_key).unwrap();
    let region2 = pd_client.get_region(right_key).unwrap();
    assert_eq!(region.get_id(), region2.get_id());

    let key = split_key.clone();
    let (tx, rx) = channel();
    let c = Box::new(move |write_resp: WriteResponse| {
        let mut resp = write_resp.response;
        let admin_resp = resp.mut_admin_response();
        let split_resp = admin_resp.mut_splits();
        let mut regions: Vec<_> = split_resp.take_regions().into();
        let mut d = regions.drain(..);
        let (left, right) = (d.next().unwrap(), d.next().unwrap());
        assert_eq!(left.get_end_key(), key.as_slice());
        assert_eq!(region2.get_start_key(), left.get_start_key());
        assert_eq!(left.get_end_key(), right.get_start_key());
        assert_eq!(region2.get_end_key(), right.get_end_key());
        tx.send(right).unwrap();
    });
    cluster.split_region(&region, &split_key, Callback::write(c));
    let region3 = rx.recv_timeout(Duration::from_secs(5)).unwrap();

    cluster.must_put(&split_key, b"v2");

    let (tx1, rx1) = channel();
    let c = Box::new(move |write_resp: WriteResponse| {
        assert!(write_resp.response.has_header());
        assert!(write_resp.response.get_header().has_error());
        assert!(!write_resp.response.has_admin_response());
        tx1.send(()).unwrap();
    });
    cluster.split_region(&region3, &split_key, Callback::write(c));
    rx1.recv_timeout(Duration::from_secs(5)).unwrap();
}

#[test]
fn test_auto_split_region() {
    test_util::init_log_for_test();
    let count = 3;
    let mut cluster = new_node_cluster(0, count);
    cluster.cfg.raft_store.split_region_check_tick_interval = ReadableDuration::millis(100);
    cluster.cfg.coprocessor.region_split_keys = Some(8);
    cluster.cfg.coprocessor.region_max_keys = Some(12);

    cluster.run();

    let pd_client = Arc::clone(&cluster.pd_client);
    pd_client.disable_default_operator();

    let region = pd_client.get_region(b"").unwrap();

    for i in 0..8 {
        let key = format!("k{:02}", i);
        cluster.must_put(key.as_bytes(), b"v0");
    }

    // it should be finished in millis if split.
    thread::sleep(Duration::from_millis(300));

    let target = pd_client.get_region(b"k07").unwrap();

    assert_eq!(region, target);

    for i in 8..13 {
        let key = format!("k{:02}", i);
        cluster.must_put(key.as_bytes(), b"v0");
    }

    let left = pd_client.get_region(b"").unwrap();
    let right = pd_client.get_region(b"k12").unwrap();
    if left == right {
        cluster.wait_region_split(&region);
    }

    let left = pd_client.get_region(b"").unwrap();
    let right = pd_client.get_region(b"k12").unwrap();

    println!("region: {:?}, left: {:?}, right: {:?}", region, left, right);

    assert_ne!(left, right);
    assert_eq!(region.get_start_key(), left.get_start_key());
    assert_eq!(right.get_start_key(), left.get_end_key());
    assert_eq!(region.get_end_key(), right.get_end_key());
    assert_eq!(pd_client.get_region(b"k12").unwrap(), right);
    assert_eq!(pd_client.get_region(left.get_end_key()).unwrap(), right);

    let epoch = left.get_region_epoch().clone();
    let get = new_request(left.get_id(), epoch, vec![new_get_cmd(b"k12")], false);
    let resp = cluster
        .call_command_on_leader(get, Duration::from_secs(5))
        .unwrap();
    assert!(resp.get_header().has_error());
    assert!(resp.get_header().get_error().has_key_not_in_region());
}

// A filter that disable commitment by heartbeat.
#[derive(Clone)]
struct EraseHeartbeatCommit;

impl Filter for EraseHeartbeatCommit {
    fn before(&self, msgs: &mut Vec<RaftMessage>) -> rfstore::Result<()> {
        for msg in msgs {
            if msg.get_message().get_msg_type() == MessageType::MsgHeartbeat {
                msg.mut_message().set_commit(0);
            }
        }
        Ok(())
    }
}

fn get_key(engine: &kvengine::Engine, region_id: u64, key: &[u8]) -> Option<Vec<u8>> {
    if let Some(snapshot) = engine.get_snap_access(region_id) {
        let item = snapshot.get(kvengine::WRITE_CF, key, 0);
        let res = item.get_value();
        if !res.is_empty() {
            return Some(res.into());
        }
    }
    None
}

macro_rules! check_cluster {
    ($cluster:expr, $k:expr, $v:expr, $all_committed:expr) => {
        let region = $cluster.pd_client.get_region($k).unwrap();
        let mut tried_cnt = 0;
        let leader = loop {
            match $cluster.leader_of_region(region.get_id()) {
                None => {
                    tried_cnt += 1;
                    if tried_cnt >= 3 {
                        panic!("leader should be elected");
                    }
                    continue;
                }
                Some(l) => break l,
            }
        };
        let mut missing_count = 0;
        for i in 1..=region.get_peers().len() as u64 {
            let engine = $cluster.get_engine(i);
            if $all_committed || i == leader.get_store_id() {
                must_get_equal(&engine, region.id, $k, $v);
            } else {
                // Note that a follower can still commit the log by an empty MsgAppend
                // when bcast commit is disabled. A heartbeat response comes to leader
                // before MsgAppendResponse will trigger MsgAppend.
                match get_key(&engine, region.id, &keys::data_key($k)) {
                    Some(res) => assert_eq!($v, &res[..]),
                    None => missing_count += 1,
                }
            }
        }
        assert!($all_committed || missing_count > 0);
    };
}

/// TiKV enables lazy broadcast commit optimization, which can delay split
/// on follower node. So election of new region will delay. We need to make
/// sure broadcast commit is disabled when split.
#[test]
fn test_delay_split_region() {
    let mut cluster = new_node_cluster(0, 3);
    cluster.cfg.raft_store.raft_log_gc_count_limit = Some(500);
    cluster.cfg.raft_store.merge_max_log_gap = 100;
    cluster.cfg.raft_store.raft_log_gc_threshold = 500;

    // To stable the test, we use a large hearbeat timeout 200ms(100ms * 2).
    // And to elect leader quickly, set election timeout to 1s(100ms * 10).
    configure_for_lease_read(&mut cluster.cfg, Some(100), Some(10));

    // We use three nodes for this test.
    cluster.run();

    let pd_client = Arc::clone(&cluster.pd_client);

    let region = pd_client.get_region(b"").unwrap();

    cluster.must_put(b"k1", b"v1");
    cluster.must_put(b"k3", b"v3");

    // Although skip bcast is enabled, but heartbeat will commit the log in period.
    check_cluster!(cluster, b"k1", b"v1", true);
    check_cluster!(cluster, b"k3", b"v3", true);
    cluster.must_transfer_leader(region.get_id(), new_peer(1, 1));

    cluster.add_send_filter(CloneFilterFactory(EraseHeartbeatCommit));

    cluster.must_put(b"k4", b"v4");
    sleep_ms(100);
    // skip bcast is enabled by default, so all followers should not commit
    // the log.
    check_cluster!(cluster, b"k4", b"v4", false);

    cluster.must_transfer_leader(region.get_id(), new_peer(3, 3));
    // New leader should flush old committed entries eagerly.
    check_cluster!(cluster, b"k4", b"v4", true);
    cluster.must_put(b"k5", b"v5");
    // New committed entries should be broadcast lazily.
    check_cluster!(cluster, b"k5", b"v5", false);
    cluster.add_send_filter(CloneFilterFactory(EraseHeartbeatCommit));

    let k2 = Key::from_raw(b"k2");
    // Split should be bcast eagerly, otherwise following must_put will fail
    // as no leader is available.
    cluster.must_split(&region, k2.as_encoded());
    cluster.must_put(b"k6", b"v6");

    sleep_ms(100);
    // After split, skip bcast is enabled again, so all followers should not
    // commit the log.
    check_cluster!(cluster, b"k6", b"v6", false);
}

#[test]
fn test_node_split_overlap_snapshot() {
    let mut cluster = new_node_cluster(0, 3);
    // We use three nodes([1, 2, 3]) for this test.
    cluster.run();

    // guarantee node 1 is leader
    cluster.must_transfer_leader(1, new_peer(1, 1));
    cluster.must_put(b"k0", b"v0");
    assert_eq!(cluster.leader_of_region(1), Some(new_peer(1, 1)));

    let pd_client = Arc::clone(&cluster.pd_client);

    // isolate node 3 for region 1.
    cluster.add_send_filter(CloneFilterFactory(RegionPacketFilter::new(1, 3)));
    cluster.must_put(b"k1", b"v1");

    let region = pd_client.get_region(b"").unwrap();

    // split (-inf, +inf) -> (-inf, k2), [k2, +inf]
    let split_key = Key::from_raw(b"k2");
    cluster.must_split(&region, split_key.as_encoded());

    cluster.must_put(b"k3", b"v3");

    // node 1 and node 2 must have k2, but node 3 must not.
    for i in 1..3 {
        let engine = cluster.get_engine(i);
        must_get_equal(&engine, region.id, b"k3", b"v3");
    }

    let engine3 = cluster.get_engine(3);
    must_get_none(&engine3, region.id, b"k3");

    cluster.clear_send_filters();
    cluster.must_put(b"k4", b"v4");

    // node 3 must have k4.
    must_get_equal(&engine3, region.id, b"k4", b"v4");
}

#[test]
fn test_apply_new_version_snapshot() {
    let mut cluster = new_node_cluster(0, 3);
    // truncate the log quickly so that we can force sending snapshot.
    cluster.cfg.raft_store.raft_log_gc_tick_interval = ReadableDuration::millis(20);
    cluster.cfg.raft_store.raft_log_gc_count_limit = Some(5);
    cluster.cfg.raft_store.raft_log_gc_threshold = 5;

    // We use three nodes([1, 2, 3]) for this test.
    cluster.run();

    // guarantee node 1 is leader
    cluster.must_transfer_leader(1, new_peer(1, 1));
    cluster.must_put(b"k0", b"v0");
    assert_eq!(cluster.leader_of_region(1), Some(new_peer(1, 1)));

    let pd_client = Arc::clone(&cluster.pd_client);

    // isolate node 3 for region 1.
    cluster.add_send_filter(CloneFilterFactory(RegionPacketFilter::new(1, 3)));
    cluster.must_put(b"k1", b"v1");

    let region = pd_client.get_region(b"").unwrap();

    // split (-inf, +inf) -> (-inf, k2), [k2, +inf]
    let split_key = Key::from_raw(b"k2");
    cluster.must_split(&region, split_key.as_encoded());
    cluster.must_put(b"k3", b"v3");

    // node 1 and node 2 must have k2, but node 3 must not.
    for i in 1..3 {
        let engine = cluster.get_engine(i);
        must_get_equal(&engine, region.id, b"k3", b"v3");
    }

    let engine3 = cluster.get_engine(3);
    must_get_none(&engine3, region.id, b"k2");

    // transfer leader to ease the preasure of store 1.
    cluster.must_transfer_leader(1, new_peer(2, 2));

    for _ in 0..100 {
        // write many logs to force log GC for region 1 and region 2.
        cluster.must_put(b"k1", b"v1");
        cluster.must_put(b"k3", b"v3");
    }

    cluster.clear_send_filters();

    sleep_ms(3000);
    // node 3 must have k1, k2.
    let left = pd_client.get_region(b"k1").unwrap();
    must_get_equal(&engine3, left.id, b"k1", b"v1");
    must_get_equal(&engine3, region.id, b"k3", b"v3");
}

#[test]
fn test_server_split_with_stale_peer() {
    let mut cluster = new_node_cluster(0, 3);
    // disable raft log gc.
    cluster.cfg.raft_store.raft_log_gc_tick_interval = ReadableDuration::secs(60);
    cluster.cfg.raft_store.peer_stale_state_check_interval = ReadableDuration::millis(500);

    let pd_client = Arc::clone(&cluster.pd_client);
    // Disable default max peer count check.
    pd_client.disable_default_operator();

    let r1 = cluster.run_conf_change();

    // add peer (2,2) to region 1.
    pd_client.must_add_peer(r1, new_peer(2, 2));

    // add peer (3,3) to region 1.
    pd_client.must_add_peer(r1, new_peer(3, 3));

    cluster.must_put(b"k0", b"v0");

    let region = pd_client.get_region(b"").unwrap();
    // check node 3 has k0.
    let engine3 = cluster.get_engine(3);
    must_get_equal(&engine3, region.id, b"k0", b"v0");

    // guarantee node 1 is leader.
    cluster.must_transfer_leader(r1, new_peer(1, 1));

    // isolate node 3 for region 1.
    // only filter MsgAppend to avoid election when recover.
    cluster.add_send_filter(CloneFilterFactory(
        RegionPacketFilter::new(1, 3).msg_type(MessageType::MsgAppend),
    ));

    // split (-inf, +inf) -> (-inf, k2), [k2, +inf]
    let split_key = Key::from_raw(b"k2");
    cluster.must_split(&region, split_key.as_encoded());
    cluster.must_put(b"k3", b"v2");

    let region2 = pd_client.get_region(b"k3").unwrap();

    // remove peer3 in region 2.
    let peer3 = peer_on_store(&region2, 3);
    pd_client.must_remove_peer(region2.get_id(), peer3.clone());

    // clear isolation so node 3 can split region 1.
    // now node 3 has a stale peer for region 2, but
    // it will be removed soon.
    cluster.clear_send_filters();
    cluster.must_put(b"k1", b"v1");

    let left = pd_client.get_region(b"k1").unwrap();
    // check node 3 has k1
    must_get_equal(&engine3, left.id, b"k1", b"v1");

    // split [k2, +inf) -> [k2, k3), [k3, +inf]
    let split_k3 = Key::from_raw(b"k3");
    cluster.must_split(&region2, split_k3.as_encoded());
    let region3 = pd_client.get_region(b"k4").unwrap();
    // region 3 can't contain node 3.
    assert_eq!(region3.get_peers().len(), 2);
    assert!(find_peer(&region3, 3).is_none());

    let new_peer_id = pd_client.alloc_id().unwrap();
    // add peer (3, new_peer_id) to region 3
    pd_client.must_add_peer(region3.get_id(), new_peer(3, new_peer_id));

    cluster.must_put(b"k4", b"v4");
    // node 3 must have k3.
    must_get_equal(&engine3, region3.id, b"k4", b"v4");
}

// Test steps
// set max region size/split size 2000 and put data till 1000
// set max region size/split size < 1000 and reboot
// verify the region is splitted.
#[test]
fn test_node_split_region_after_reboot_with_config_change() {
    let count = 1;
    let mut cluster = new_node_cluster(0, count);
    cluster.cfg.raft_store.split_region_check_tick_interval = ReadableDuration::millis(50);
    cluster.cfg.raft_store.raft_log_gc_tick_interval = ReadableDuration::secs(20);
    cluster.cfg.coprocessor.enable_region_bucket = true;
    cluster.cfg.coprocessor.region_max_keys = Some(10);
    cluster.cfg.coprocessor.region_split_keys = Some(8);

    cluster.run();

    let pd_client = Arc::clone(&cluster.pd_client);

    for i in 0..5 {
        let key = format!("k{}", i);
        cluster.must_put(key.as_bytes(), b"v");
    }

    // there should be 1 region
    sleep_ms(200);
    assert_eq!(pd_client.get_split_count(), 0);

    cluster.stop_node(1);
    // change the config to make the region splitable
    cluster.cfg.coprocessor.region_max_keys = Some(4);
    cluster.cfg.coprocessor.region_split_keys = Some(3);
    cluster.run_node(1).unwrap();

    let mut try_cnt = 0;
    loop {
        sleep_ms(20);
        if pd_client.get_split_count() > 0 {
            break;
        }
        try_cnt += 1;
        if try_cnt == 200 {
            panic!("expect get_split_count > 0 after 4s");
        }
    }
}

// NOTE: CSE does not support configuring `right_derive_when_split`.
#[test]
fn test_split_epoch_not_match() {
    let mut cluster = new_node_cluster(0, 3);
    cluster.run();
    let pd_client = Arc::clone(&cluster.pd_client);
    let old = pd_client.get_region(b"k1").unwrap();
    // Construct a get command using old region meta.
    let get_old = new_request(
        old.get_id(),
        old.get_region_epoch().clone(),
        vec![new_get_cmd(b"k1")],
        false,
    );
    let split_k2 = Key::from_raw(b"k2");
    cluster.must_split(&old, split_k2.as_encoded());
    let r = pd_client.get_region(b"k3").unwrap();
    let get_middle = new_request(
        r.get_id(),
        r.get_region_epoch().clone(),
        vec![new_get_cmd(b"k3")],
        false,
    );
    let split_k3 = Key::from_raw(b"k3");
    cluster.must_split(&r, split_k3.as_encoded());
    let r = pd_client.get_region(b"k4").unwrap();
    let split_k4 = Key::from_raw(b"k4");
    cluster.must_split(&r, split_k4.as_encoded());
    let regions: Vec<_> = [b"k00", b"k22", b"k33", b"k44"]
        .iter()
        .map(|&k| pd_client.get_region(k).unwrap())
        .collect();

    let new = regions[3].clone();
    // Newer epoch also triggers the EpochNotMatch error.
    let mut latest_epoch = new.get_region_epoch().clone();
    let latest_version = latest_epoch.get_version() + 1;
    latest_epoch.set_version(latest_version);
    let get_new = new_request(new.get_id(), latest_epoch, vec![new_get_cmd(b"k1")], false);

    let cases = vec![
        // All regions should be returned as request uses an oldest epoch.
        (get_old, vec![regions[3].clone()]),
        // Only new split regions should be returned.
        (get_middle, vec![regions[3].clone()]),
        // Epoch is too new that TiKV can't offer any useful hint.
        (get_new, vec![regions[3].clone()]),
    ];
    for (idx, (get, exp)) in cases.into_iter().enumerate() {
        let resp = cluster
            .call_command_on_leader(get.clone(), Duration::from_secs(5))
            .unwrap();
        assert!(resp.get_header().has_error(), "{:?}", get);
        assert!(
            resp.get_header().get_error().has_epoch_not_match(),
            "{:?}",
            get
        );
        assert_eq!(
            resp.get_header()
                .get_error()
                .get_epoch_not_match()
                .get_current_regions(),
            &*exp,
            "case: {}, {:?}",
            idx,
            get
        );
    }
}

#[test]
fn test_node_quick_election_after_split() {
    let mut cluster = new_node_cluster(0, 3);

    // For the peer which is the leader of the region before split, it should
    // campaigns immediately. and then this peer may take the leadership
    // earlier. `test_quick_election_after_split` is a helper function for testing
    // this feature.
    // Calculate the reserved time before a new campaign after split.
    let reserved_time =
        Duration::from_millis(cluster.cfg.raft_store.raft_base_tick_interval.as_millis() * 2);

    cluster.run();
    cluster.must_put(b"k1", b"v1");
    cluster.must_put(b"k3", b"v3");
    let region = cluster.get_region(b"k1");
    let old_leader = cluster.leader_of_region(region.get_id()).unwrap();

    let split_key = Key::from_raw(b"k2");
    cluster.must_split(&region, split_key.as_encoded());

    // Wait for the peer of new region to start campaign.
    thread::sleep(reserved_time);

    // The campaign should always succeeds in the ideal test environment.
    let new_region = cluster.get_region(b"k3");
    // Ensure the new leader is established for the newly split region, and it
    // shares the same store with the leader of old region.
    let new_leader = cluster.query_leader(
        old_leader.get_store_id(),
        new_region.get_id(),
        Duration::from_secs(5),
    );
    assert!(new_leader.is_some());
}

#[test]
fn test_node_split_update_region_right_derive() {
    let mut cluster = new_node_cluster(0, 3);
    // Election timeout and max leader lease is 1s.
    configure_for_lease_read(&mut cluster.cfg.tikv, Some(100), Some(10));

    cluster.run();

    cluster.must_put(b"k1", b"v1");
    cluster.must_put(b"k3", b"v3");

    let pd_client = Arc::clone(&cluster.pd_client);
    let region = pd_client.get_region(b"k1").unwrap();
    let split_key = Key::from_raw(b"k2");
    cluster.must_split(&region, split_key.as_encoded());
    let right = pd_client.get_region(b"k3").unwrap();

    let origin_leader = cluster.leader_of_region(right.get_id()).unwrap();
    let new_leader = right
        .get_peers()
        .iter()
        .find(|&p| p.get_id() != origin_leader.get_id())
        .cloned()
        .unwrap();

    // Make sure split is done in the new_leader.
    // "k4" belongs to the right.
    cluster.must_put(b"k4", b"v4");
    must_get_equal(
        &cluster.get_engine(new_leader.get_store_id()),
        right.id,
        b"k4",
        b"v4",
    );

    // Transfer leadership to another peer.
    cluster.must_transfer_leader(right.get_id(), new_leader);

    // Make sure the new_leader is in lease.
    cluster.must_put(b"k4", b"v5");

    // "k1" is not in the range of right.
    let get = new_request(
        right.get_id(),
        right.get_region_epoch().clone(),
        vec![new_get_cmd(b"k1")],
        false,
    );
    debug!("requesting {:?}", get);
    let resp = cluster
        .call_command_on_leader(get, Duration::from_secs(5))
        .unwrap();
    assert!(resp.get_header().has_error(), "{:?}", resp);
    assert!(
        resp.get_header().get_error().has_key_not_in_region(),
        "{:?}",
        resp
    );
}

#[test]
fn test_split_with_epoch_not_match() {
    let mut cluster = new_node_cluster(0, 3);
    let pd_client = Arc::clone(&cluster.pd_client);
    pd_client.disable_default_operator();

    cluster.run();

    cluster.must_transfer_leader(1, new_peer(1, 1));

    // Remove a peer to make conf version become 2.
    pd_client.must_remove_peer(1, new_peer(2, 2));
    let region = cluster.get_region(b"");

    let mut admin_req = AdminRequest::default();
    admin_req.set_cmd_type(AdminCmdType::BatchSplit);

    let mut batch_split_req = BatchSplitRequest::default();
    batch_split_req.mut_requests().push(SplitRequest::default());
    batch_split_req.mut_requests()[0].set_split_key(Key::from_raw(b"s").into_encoded());
    batch_split_req.mut_requests()[0].set_new_region_id(1000);
    batch_split_req.mut_requests()[0].set_new_peer_ids(vec![1001, 1002]);
    batch_split_req.mut_requests()[0].set_right_derive(true);
    admin_req.set_splits(batch_split_req);

    let mut epoch = region.get_region_epoch().clone();
    epoch.conf_ver -= 1;
    let req = new_admin_request(1, &epoch, admin_req);
    let resp = cluster
        .call_command_on_leader(req, Duration::from_secs(3))
        .unwrap();
    assert!(resp.get_header().get_error().has_epoch_not_match());
}

#[test]
fn test_catch_up_peers_after_split() {
    let mut cluster = new_node_cluster(0, 3);
    let pd_client = Arc::clone(&cluster.pd_client);
    pd_client.disable_default_operator();

    cluster.run();

    let left_key = b"k1";
    let right_key = b"k3";
    let split_key = Key::from_raw(b"k2").into_encoded();
    cluster.must_put(left_key, b"v1");
    cluster.must_put(right_key, b"v3");

    // Left and right key must be in same region before split.
    let region = pd_client.get_region(left_key).unwrap();
    let region2 = pd_client.get_region(right_key).unwrap();
    assert_eq!(region.get_id(), region2.get_id());

    // Split with split_key, so left_key must in left, and right_key in right.
    cluster.must_split(&region, &split_key);

    // Get new split region by right_key because default right_derive is false.
    let right_region = pd_client.get_region(right_key).unwrap();

    let pending_peers = pd_client.get_pending_peers();

    // Ensure new split region has no pending peers.
    for p in right_region.get_peers() {
        assert!(!pending_peers.contains_key(&p.id))
    }
}

// A filter that disable read index by heartbeat.
#[derive(Clone)]
struct EraseHeartbeatContext;

impl Filter for EraseHeartbeatContext {
    fn before(&self, msgs: &mut Vec<RaftMessage>) -> Result<()> {
        for msg in msgs {
            if msg.get_message().get_msg_type() == MessageType::MsgHeartbeat {
                msg.mut_message().clear_context();
            }
        }
        Ok(())
    }
}

#[test]
fn test_clear_uncampaigned_regions_after_split() {
    let mut cluster = new_node_cluster(0, 3);
    cluster.cfg.raft_store.raft_base_tick_interval = ReadableDuration::millis(50);
    cluster.cfg.raft_store.raft_election_timeout_ticks = 10;

    let pd_client = cluster.pd_client.clone();
    pd_client.disable_default_operator();

    cluster.run();
    cluster.must_put(b"k1", b"v1");
    cluster.must_put(b"k2", b"v2");
    cluster.must_put(b"k3", b"v3");
    // Transfer leader to peer 3.
    let region = pd_client.get_region(b"k2").unwrap();
    cluster.must_transfer_leader(region.get_id(), new_peer(3, 3));

    // New split regions will be recorded into uncampaigned region list of
    // followers (in peer 1 and peer 2).
    let split_k2 = Key::from_raw(b"k2");
    cluster.split_region(
        &region,
        split_k2.as_encoded(),
        Callback::write(Box::new(move |_write_resp: WriteResponse| {})),
    );
    // Wait the old lease of the leader timeout and followers clear its
    // uncampaigned region list.
    thread::sleep(
        cluster.cfg.raft_store.raft_base_tick_interval.0
            * cluster.cfg.raft_store.raft_election_timeout_ticks as u32
            * 3,
    );
    // The leader of the parent region should still be peer 3 as no
    // other peers can become leader.
    cluster.reset_leader_of_region(region.get_id());
    assert_eq!(
        cluster.leader_of_region(region.get_id()).unwrap(),
        new_peer(3, 3)
    );
}
