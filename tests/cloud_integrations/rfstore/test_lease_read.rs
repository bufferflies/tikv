// Copyright 2025 TiKV Project Authors. Licensed under Apache-2.0.

//! A module contains test cases for lease read on Raft leader.

use std::{
    sync::{atomic::*, mpsc, Arc},
    thread,
    time::Duration,
};

use kvproto::{metapb, raft_serverpb::RaftMessage};
use more_asserts::assert_le;
use pd_client::PdClient;
use raft::eraftpb::{ConfChangeType, MessageType};
use test_raftstore::{
    configure_for_lease_read, new_admin_request, new_change_peer_request, new_get_cmd, new_peer,
    new_read_index_cmd, new_request, peer_on_store, sleep_ms,
};
use test_rfstore::*;
use tikv_util::{config::*, time::Instant, HandyRwLock};

// A helper function for testing the lease reads and lease renewing.
// The leader keeps a record of its leader lease, and uses the system's
// monotonic raw clocktime to check whether its lease has expired.
// If the leader lease has not expired, when the leader receives a read request
//   - with `read_quorum == false`, the leader will serve it by reading local
// data. This way of handling request is called "lease read".
//   - with `read_quorum == true`, the leader will serve it by doing index read
// (see raft's doc). This way of handling request is called "index read".
// If the leader lease has expired, leader will serve both kinds of requests by
// index read, and propose an no-op entry to raft quorum to renew the lease.
// No matter what status the leader lease is, a write request is always served
// by writing a Raft log to the Raft quorum. It is called "consistent write".
// All writes are consistent writes. Every time the leader performs a consistent
// read/write, it will try to renew its lease.
fn test_renew_lease(count: usize) {
    let mut cluster = new_node_cluster(0, count);
    // Avoid triggering the log compaction in this test case.
    cluster.cfg.raft_store.raft_log_gc_threshold = 100;
    // Increase the Raft tick interval to make this test case running reliably.
    // Use large election timeout to make leadership stable.
    configure_for_lease_read(&mut cluster.cfg, Some(50), Some(10_000));
    // Override max leader lease to 2 seconds.
    let max_lease = Duration::from_secs(2);
    cluster.cfg.raft_store.raft_store_max_leader_lease = ReadableDuration(max_lease);
    cluster.cfg.raft_store.check_leader_lease_interval = ReadableDuration::hours(10);
    cluster.cfg.raft_store.renew_leader_lease_advance_duration = ReadableDuration::secs(0);

    let node_id = 1u64;
    let store_id = 1u64;
    let peer = new_peer(store_id, node_id);
    cluster.pd_client.disable_default_operator();
    let region_id = cluster.run_conf_change();

    let key = b"k";
    cluster.must_put(key, b"v0");
    for id in 2..=cluster.engines.len() as u64 {
        cluster.pd_client.must_add_peer(region_id, new_peer(id, id));
        must_get_equal(&cluster.get_engine(id), region_id, key, b"v0");
    }

    // Write the initial value for a key.
    let key = b"k";
    cluster.must_put(key, b"v1");
    // Force `peer` to become leader.
    let region = cluster.get_region(key);
    let region_id = region.get_id();
    cluster.must_transfer_leader(region_id, peer.clone());
    let state = cluster.raft_state(peer.id, store_id).unwrap();
    let last_index = state.get_last_index();

    let detector = LeaseReadFilter::default();
    cluster.add_send_filter(CloneFilterFactory(detector.clone()));

    // Issue a read request and check the value on response.
    must_read_on_peer(&mut cluster, peer.clone(), &region, key, b"v1");
    assert_eq!(detector.ctx.rl().len(), 0);

    let mut expect_lease_read = 0;

    if cluster.engines.len() > 1 {
        // Wait for the leader lease to expire.
        thread::sleep(max_lease);

        // Issue a read request and check the value on response.
        must_read_on_peer(&mut cluster, peer.clone(), &region, key, b"v1");

        // Check if the leader does a index read and renewed its lease.
        assert_eq!(cluster.leader_of_region(region_id), Some(peer.clone()));
        expect_lease_read += 1;
        assert_eq!(detector.ctx.rl().len(), expect_lease_read);
    }

    // Wait for the leader lease to expire.
    thread::sleep(max_lease);

    // Issue a write request.
    cluster.must_put(key, b"v2");

    // Check if the leader has renewed its lease so that it can do lease read.
    assert_eq!(cluster.leader_of_region(region_id), Some(peer.clone()));
    let state = cluster.raft_state(peer.id, store_id).unwrap();
    assert_eq!(state.get_last_index(), last_index + 1);

    // Issue a read request and check the value on response.
    must_read_on_peer(&mut cluster, peer, &region, key, b"v2");

    // Check if the leader does a local read.
    assert_eq!(detector.ctx.rl().len(), expect_lease_read);
}

#[test]
fn test_one_node_renew_lease() {
    test_renew_lease(1);
}

#[test]
fn test_node_renew_lease() {
    test_renew_lease(3);
}

// Test lease reads when the lease has expired.
// If the leader lease has expired, there may be new leader elected and
// the old leader will fail to renew its lease.
#[test]
fn test_node_lease_expired() {
    let count = 3;
    let mut cluster = new_node_cluster(0, count);
    let pd_client = Arc::clone(&cluster.pd_client);
    // Disable default max peer number check.
    pd_client.disable_default_operator();

    // Avoid triggering the log compaction in this test case.
    cluster.cfg.raft_store.raft_log_gc_threshold = 100;
    // Increase the Raft tick interval to make this test case running reliably.
    let election_timeout = configure_for_lease_read(&mut cluster.cfg, Some(50), None);

    let node_id = 3u64;
    let store_id = 3u64;
    let peer = new_peer(store_id, node_id);
    cluster.run();

    // Write the initial value for a key.
    let key = b"k";
    cluster.must_put(key, b"v1");
    // Force `peer` to become leader.
    let region = cluster.get_region(key);
    let region_id = region.get_id();
    cluster.must_transfer_leader(region_id, peer.clone());

    // Isolate the leader `peer` from other peers.
    cluster.add_send_filter(IsolationFilterFactory::new(store_id));

    // Wait for the leader lease to expire and a new leader is elected.
    thread::sleep(election_timeout * 2);

    // Issue a read request and check the value on response.
    must_error_read_on_peer(&mut cluster, peer, &region, key, Duration::from_secs(1));
}

// Test leader holds unsafe lease during the leader transfer procedure.
// When leader transfer procedure aborts later, the leader would use and update
// the lease as usual.
#[test]
fn test_node_lease_unsafe_during_leader_transfers() {
    let count = 3;
    let mut cluster = new_node_cluster(0, count);
    // Avoid triggering the log compaction in this test case.
    cluster.cfg.raft_store.raft_log_gc_threshold = 100;
    // Increase the Raft tick interval to make this test case running reliably.
    let election_timeout = configure_for_lease_read(&mut cluster.cfg, Some(500), Some(5));
    cluster.cfg.raft_store.check_leader_lease_interval = ReadableDuration::hours(10);
    cluster.cfg.raft_store.renew_leader_lease_advance_duration = ReadableDuration::secs(0);

    let store_id = 1u64;
    let peer = new_peer(store_id, 1);
    let peer3_store_id = 3u64;
    let peer3 = new_peer(peer3_store_id, 3);

    cluster.pd_client.disable_default_operator();
    let r1 = cluster.run_conf_change();
    cluster.must_put(b"k0", b"v0");
    cluster.pd_client.must_add_peer(r1, new_peer(2, 2));
    must_get_equal(&cluster.get_engine(2), r1, b"k0", b"v0");
    cluster.pd_client.must_add_peer(r1, new_peer(3, 3));
    must_get_equal(&cluster.get_engine(3), r1, b"k0", b"v0");

    let detector = LeaseReadFilter::default();
    cluster.add_send_filter(CloneFilterFactory(detector.clone()));

    // write the initial value for a key.
    let key = b"k";
    cluster.must_put(key, b"v1");
    // Force `peer1` to became leader.
    let region = cluster.get_region(key);
    let region_id = region.get_id();
    cluster.must_transfer_leader(region_id, peer.clone());

    // Issue a read request and check the value on response.
    must_read_on_peer(&mut cluster, peer.clone(), &region, key, b"v1");

    let state = cluster.raft_state(peer.id, store_id).unwrap();
    let last_index = state.get_last_index();

    // Check if the leader does a local read.
    must_read_on_peer(&mut cluster, peer.clone(), &region, key, b"v1");
    let state = cluster.raft_state(peer.id, store_id).unwrap();
    assert_eq!(state.get_last_index(), last_index);
    assert_eq!(detector.ctx.rl().len(), 0);

    // Ensure peer 3 is ready to transfer leader.
    must_get_equal(&cluster.get_engine(3), region_id, key, b"v1");

    // Drop MsgTimeoutNow to `peer3` so that the leader transfer procedure would
    // abort later.
    cluster.add_send_filter(CloneFilterFactory(
        RegionPacketFilter::new(region_id, peer3_store_id)
            .msg_type(MessageType::MsgTimeoutNow)
            .direction(Direction::Recv),
    ));

    // Issue a transfer leader request to transfer leader from `peer` to `peer3`.
    cluster.transfer_leader(region_id, peer3);

    // Delay a while to ensure transfer leader procedure is triggered inside raft
    // module.
    thread::sleep(election_timeout / 2);

    // Issue a read request and it will fall back to read index.
    must_read_on_peer(&mut cluster, peer.clone(), &region, key, b"v1");
    assert_eq!(detector.ctx.rl().len(), 1);

    // And read index should not update lease.
    must_read_on_peer(&mut cluster, peer.clone(), &region, key, b"v1");
    assert_eq!(detector.ctx.rl().len(), 2);

    // Make sure the leader transfer procedure timeouts.
    thread::sleep(election_timeout * 2);

    // Then the leader transfer procedure aborts, now the leader could do lease read
    // or consistent read/write and renew/reuse the lease as usual.

    // Issue a read request and check the value on response.
    must_read_on_peer(&mut cluster, peer.clone(), &region, key, b"v1");
    assert_eq!(detector.ctx.rl().len(), 3);

    // Check if the leader also propose an entry to renew its lease.
    cluster.wait_last_index(region_id, store_id, last_index + 1, Duration::from_secs(5));

    // Wait some time for the proposal to be applied.
    let now = Instant::now();
    loop {
        thread::sleep(Duration::from_millis(100));
        if now.saturating_elapsed() > election_timeout * 2 {
            panic!("store {} must apply to {}", store_id, last_index + 1);
        }
        // NOTE: CSE will not update its applied_index(last sequence) for a noop entry,
        // so we check the last_preprocessed_index instead.
        let raft_state = cluster.raft_state(region_id, store_id).unwrap();
        if raft_state.get_last_preprocessed_index() > last_index {
            break;
        }
    }

    // Check if the leader does a local read.
    must_read_on_peer(&mut cluster, peer.clone(), &region, key, b"v1");
    let state = cluster.raft_state(peer.id, store_id).unwrap();
    assert_eq!(state.get_last_index(), last_index + 1);
    assert_eq!(detector.ctx.rl().len(), 3);
}

/// test whether the read index callback will be handled when a region is
/// destroyed. If it's not handled properly, it will cause dead lock in
/// transaction scheduler.
#[test]
fn test_node_callback_when_destroyed() {
    let count = 3;
    let mut cluster = new_node_cluster(0, count);
    // Increase the election tick to make this test case running reliably.
    configure_for_lease_read(&mut cluster.cfg, None, Some(50));
    cluster.run();
    cluster.must_put(b"k1", b"v1");
    let leader = cluster.leader_of_region(1).unwrap();
    let cc = new_change_peer_request(ConfChangeType::RemoveNode, leader.clone());
    let epoch = cluster.get_region_epoch(1);
    let req = new_admin_request(1, &epoch, cc);
    // so the leader can't commit the conf change yet.
    let block = Arc::new(AtomicBool::new(true));
    cluster.add_send_filter(CloneFilterFactory(
        RegionPacketFilter::new(1, leader.get_store_id())
            .msg_type(MessageType::MsgAppendResponse)
            .direction(Direction::Recv)
            .when(Arc::clone(&block)),
    ));
    let mut filter = LeaseReadFilter::default();
    filter.take = true;
    // so the leader can't perform read index.
    cluster.add_send_filter(CloneFilterFactory(filter.clone()));
    // it always timeout, no need to wait.
    let _ = cluster.call_command_on_leader(req, Duration::from_millis(500));

    // To make sure `get` is handled before destroy leader, we must issue
    // `get` then unblock append responses.
    let leader_node_id = leader.get_store_id();
    let get = new_get_cmd(b"k1");
    let mut req = new_request(1, epoch, vec![get], true);
    req.mut_header().set_peer(leader);
    let (cb, mut rx) = make_cb(&req);
    cluster
        .sim
        .rl()
        .async_command_on_node(leader_node_id, req, cb)
        .unwrap();
    // Unblock append responses after we issue the req.
    block.store(false, Ordering::SeqCst);
    let resp = rx.recv_timeout(Duration::from_secs(3)).unwrap();

    assert!(
        !filter.ctx.rl().is_empty(),
        "read index should be performed"
    );
    assert!(
        resp.get_header().get_error().has_region_not_found(),
        "{:?}",
        resp
    );
}

#[test]
fn test_local_read_cache() {
    let mut cluster = new_node_cluster(0, 3);
    configure_for_lease_read(&mut cluster.cfg, Some(50), None);
    cluster.pd_client.disable_default_operator();
    cluster.run();
    let pd_client = Arc::clone(&cluster.pd_client);

    cluster.must_put(b"k1", b"v1");
    let r1 = cluster.get_region(b"k1");

    must_get_equal(&cluster.get_engine(1), r1.id, b"k1", b"v1");
    must_get_equal(&cluster.get_engine(2), r1.id, b"k1", b"v1");
    must_get_equal(&cluster.get_engine(3), r1.id, b"k1", b"v1");

    let leader = cluster.leader_of_region(r1.get_id()).unwrap();
    let new_leader = new_peer((leader.get_id() + 1) % 3 + 1, (leader.get_id() + 1) % 3 + 1);
    cluster.must_transfer_leader(r1.get_id(), new_leader);

    // Add the peer back and make sure it catches up latest logs.
    pd_client.must_remove_peer(r1.get_id(), leader.clone());
    let replace_peer = new_peer(leader.get_store_id(), 10000);
    pd_client.must_add_peer(r1.get_id(), replace_peer.clone());
    cluster.must_put(b"k2", b"v2");
    must_get_equal(
        &cluster.get_engine(leader.get_store_id()),
        r1.id,
        b"k2",
        b"v2",
    );

    cluster.must_transfer_leader(r1.get_id(), replace_peer);
    cluster.must_put(b"k3", b"v3");
    must_get_equal(
        &cluster.get_engine(leader.get_store_id()),
        r1.id,
        b"k3",
        b"v3",
    );
}

/// Test latency changes when a leader becomes follower right after it receives
/// read_index heartbeat response.
#[test]
fn test_not_leader_read_lease() {
    let mut cluster = new_node_cluster(0, 3);
    // Avoid triggering the log compaction in this test case.
    cluster.cfg.raft_store.raft_log_gc_threshold = 100;
    // Increase the Raft tick interval to make this test case running reliably.
    configure_for_lease_read(&mut cluster.cfg, Some(50), None);
    let heartbeat_interval = cluster.cfg.raft_store.raft_heartbeat_interval();

    cluster.run();

    cluster.must_put(b"k1", b"v1");
    cluster.must_transfer_leader(1, new_peer(1, 1));
    cluster.must_put(b"k2", b"v2");
    must_get_equal(&cluster.get_engine(3), 1, b"k2", b"v2");

    // Add a filter to delay heartbeat response until transfer leader begins.
    cluster.sim.wl().add_recv_filter(
        1,
        Box::new(LeadingFilter::new(
            MessageType::MsgHeartbeatResponse,
            MessageType::MsgRequestVote,
        )),
    );
    cluster.add_send_filter(CloneFilterFactory(
        RegionPacketFilter::new(1, 2)
            .direction(Direction::Recv)
            .msg_type(MessageType::MsgAppend),
    ));

    let mut region = cluster.get_region(b"k1");
    let region_id = region.get_id();
    let mut req = new_request(
        region_id,
        region.take_region_epoch(),
        vec![new_get_cmd(b"k2")],
        true,
    );
    req.mut_header().set_peer(new_peer(1, 1));
    let (cb, mut rx) = make_cb(&req);
    cluster.sim.rl().async_command_on_node(1, req, cb).unwrap();

    cluster.must_transfer_leader(region_id, new_peer(3, 3));
    // Even the leader steps down, it should respond to read index in time.
    rx.recv_timeout(heartbeat_interval).unwrap();
}

/// Test whether read index is greater than applied index.
/// 1. Add heartbeat msg filter.
/// 2. Propose a read index request.
/// 3. Put a key and get the latest applied index.
/// 4. Propose another read index request.
/// 5. Remove the filter and check whether the latter read index is greater than
/// applied index.
///
/// In previous implementation, these two read index request will be batched and
/// will get the same read index which breaks the correctness because the latter
/// one is proposed after the applied index has increased and replied to client.
#[test]
fn test_read_index_after_write() {
    let mut cluster = new_node_cluster(0, 3);
    configure_for_lease_read(&mut cluster.cfg, Some(50), Some(10));
    let heartbeat_interval = cluster.cfg.raft_store.raft_heartbeat_interval();
    let pd_client = Arc::clone(&cluster.pd_client);
    pd_client.disable_default_operator();

    cluster.run();

    cluster.must_put(b"k1", b"v1");
    let region = pd_client.get_region(b"k1").unwrap();
    let region_on_store1 = peer_on_store(&region, 1).to_owned();
    cluster.must_transfer_leader(region.get_id(), region_on_store1.clone());

    cluster.add_send_filter(IsolationFilterFactory::new(3));
    // Add heartbeat msg filter to prevent the leader to reply the read index
    // response.
    let filter = Box::new(
        RegionPacketFilter::new(region.get_id(), 2)
            .direction(Direction::Recv)
            .msg_type(MessageType::MsgHeartbeat),
    );
    cluster.sim.wl().add_recv_filter(2, filter);

    let mut req = new_request(
        region.get_id(),
        region.get_region_epoch().clone(),
        vec![new_read_index_cmd()],
        true,
    );
    req.mut_header()
        .set_peer(new_peer(1, region_on_store1.get_id()));
    // Don't care about the first one's read index
    let (cb, _) = make_cb(&req);
    cluster.sim.rl().async_command_on_node(1, req, cb).unwrap();

    cluster.must_put(b"k2", b"v2");
    let applied_index = cluster.apply_state(region.get_id(), 1).get_applied_index();

    let mut req = new_request(
        region.get_id(),
        region.get_region_epoch().clone(),
        vec![new_read_index_cmd()],
        true,
    );
    req.mut_header()
        .set_peer(new_peer(1, region_on_store1.get_id()));
    let (cb, mut rx) = make_cb(&req);
    cluster.sim.rl().async_command_on_node(1, req, cb).unwrap();

    cluster.sim.wl().clear_recv_filters(2);

    let response = rx.recv_timeout(heartbeat_interval).unwrap();
    assert!(
        response.get_responses()[0]
            .get_read_index()
            .get_read_index()
            >= applied_index
    );
}

#[test]
#[ignore = "rfstore does not support check_leader_lease_tick"]
fn test_infinite_lease() {
    let mut cluster = new_node_cluster(0, 3);
    // Avoid triggering the log compaction in this test case.
    cluster.cfg.raft_store.raft_log_gc_threshold = 100;
    // Increase the Raft tick interval to make this test case running reliably.
    // Use large election timeout to make leadership stable.
    configure_for_lease_read(&mut cluster.cfg, Some(50), Some(10_000));
    // Override max leader lease to 2 seconds.
    let max_lease = Duration::from_secs(2);
    cluster.cfg.raft_store.raft_store_max_leader_lease = ReadableDuration(max_lease);

    let peer = new_peer(1, 1);
    cluster.pd_client.disable_default_operator();
    let region_id = cluster.run_conf_change();

    let key = b"k";
    cluster.must_put(key, b"v0");
    for id in 2..=cluster.engines.len() as u64 {
        cluster.pd_client.must_add_peer(region_id, new_peer(id, id));
        must_get_equal(&cluster.get_engine(id), region_id, key, b"v0");
    }

    // Force `peer` to become leader.
    let region = cluster.get_region(key);
    let region_id = region.get_id();
    cluster.must_transfer_leader(region_id, peer.clone());

    let detector = LeaseReadFilter::default();
    cluster.add_send_filter(CloneFilterFactory(detector.clone()));

    // Issue a read request and check the value on response.
    must_read_on_peer(&mut cluster, peer.clone(), &region, key, b"v0");
    assert_eq!(detector.ctx.rl().len(), 0);

    // Wait for the leader lease to expire.
    thread::sleep(max_lease);

    // Check if renew-lease-tick proposed a read index and renewed the leader lease.
    assert_eq!(cluster.leader_of_region(region_id), Some(peer.clone()));
    assert_eq!(detector.ctx.rl().len(), 1);
    // Issue a read request to verify the lease.
    must_read_on_peer(&mut cluster, peer.clone(), &region, key, b"v0");
    assert_eq!(cluster.leader_of_region(region_id), Some(peer));
    assert_eq!(detector.ctx.rl().len(), 1);

    // renew-lease-tick shouldn't propose any request if the leader lease is not
    // expired.
    for _ in 0..4 {
        cluster.must_put(key, b"v0");
        thread::sleep(max_lease / 4);
    }
    assert_eq!(detector.ctx.rl().len(), 1);
}

// LocalReader will try to renew lease in advance, so the region that has
// continuous reads should not go to hibernate.
#[test]
fn test_node_local_read_renew_lease() {
    let mut cluster = new_node_cluster(0, 3);
    cluster.cfg.raft_store.raft_store_max_leader_lease = ReadableDuration::millis(500);
    let (base_tick_ms, election_ticks) = (50, 10);
    configure_for_lease_read(&mut cluster.cfg, Some(50), Some(10));
    cluster.pd_client.disable_default_operator();
    let region_id = cluster.run_conf_change();

    let key = b"k";
    cluster.must_put(key, b"v0");
    for id in 2..=3 {
        cluster.pd_client.must_add_peer(region_id, new_peer(id, id));
        must_get_equal(&cluster.get_engine(id), region_id, key, b"v0");
    }

    // Write the initial value for a key.
    let key = b"k";
    cluster.must_put(key, b"v1");
    // Force `peer` to become leader.
    let region = cluster.get_region(key);
    let region_id = region.get_id();
    let peer = new_peer(1, 1);
    cluster.must_transfer_leader(region_id, peer.clone());

    let detector = LeaseReadFilter::default();
    cluster.add_send_filter(CloneFilterFactory(detector.clone()));

    // election_timeout_ticks * base_tick_interval * 3
    let hibernate_wait = election_ticks * Duration::from_millis(base_tick_ms) * 3;
    let request_wait = Duration::from_millis(base_tick_ms);
    let max_renew_lease_time = 3;
    let round = hibernate_wait.as_millis() / request_wait.as_millis();
    for i in 0..round {
        // Issue a read request and check the value on response.
        must_read_on_peer(&mut cluster, peer.clone(), &region, key, b"v1");
        // Plus 1 to prevent case failure when test machine is too slow.
        assert_le!(detector.ctx.rl().len(), max_renew_lease_time + 1, "{}", i);
        thread::sleep(request_wait);
    }
}

#[test]
#[ignore = "need to cherry pick https://github.com/tikv/tikv/pull/15085"]
fn test_node_lease_restart_during_isolation() {
    let mut cluster = new_node_cluster(0, 3);
    let election_timeout = configure_for_lease_read(&mut cluster.cfg, Some(500), Some(3));
    cluster.run();
    sleep_ms(election_timeout.as_millis() as _);
    let mut region;
    let start = Instant::now_coarse();
    let key = b"k";
    loop {
        region = cluster.get_region(key);
        if region.get_peers().len() == 3
            && region
                .get_peers()
                .iter()
                .all(|p| p.get_role() == metapb::PeerRole::Voter)
        {
            break;
        }
        if start.saturating_elapsed() > Duration::from_secs(5) {
            panic!("timeout");
        }
    }

    let region_id = region.get_id();
    let peer1 = peer_on_store(&region, 1);
    let peer2 = peer_on_store(&region, 2);

    cluster.must_put(key, b"v0");
    cluster.must_transfer_leader(region_id, peer1.clone());
    must_read_on_peer(&mut cluster, peer1.clone(), &region, key, b"v0");
    cluster.must_transfer_leader(region_id, peer2.clone());
    must_read_on_peer(&mut cluster, peer2.clone(), &region, key, b"v0");

    cluster.add_send_filter(IsolationFilterFactory::new(2));

    // Restart node 3.
    cluster.stop_node(3);
    cluster.run_node(3).unwrap();

    // Let peer1 start election.
    let mut timeout = RaftMessage::default();
    timeout.mut_message().set_to(peer1.get_id());
    timeout
        .mut_message()
        .set_msg_type(MessageType::MsgTimeoutNow);
    timeout
        .mut_message()
        .set_msg_type(MessageType::MsgTimeoutNow);
    timeout.set_region_id(region.get_id());
    timeout.set_from_peer(peer2.clone());
    timeout.set_to_peer(peer1.clone());
    timeout.set_region_epoch(region.get_region_epoch().clone());
    cluster.send_raft_msg(timeout).unwrap();

    let (tx, rx) = mpsc::channel();
    let append_resp_notifier = Box::new(MessageTypeNotifier::new(
        MessageType::MsgAppendResponse,
        tx,
        Arc::from(AtomicBool::new(true)),
    ));
    cluster.sim.wl().add_send_filter(3, append_resp_notifier);
    let timeout = Duration::from_secs(5);
    rx.recv_timeout(timeout).unwrap();

    let mut put = new_write_request(
        region.get_id(),
        region.get_region_epoch().clone(),
        new_put_cmd(key, b"v1"),
    );
    put.mut_header().set_peer(peer1.clone());
    let resp = cluster.call_command(put, timeout).unwrap();
    assert!(!resp.get_header().has_error(), "{:?}", resp);
    must_read_on_peer(&mut cluster, peer1.clone(), &region, key, b"v1");
    must_error_read_on_peer(&mut cluster, peer2.clone(), &region, key, timeout);
}
