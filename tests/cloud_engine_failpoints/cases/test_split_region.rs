// Copyright 2025 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc, Arc, Mutex,
    },
    time::Duration,
};

use kvproto::raft_serverpb::RaftMessage;
use pd_client::PdClient;
use raft::eraftpb::MessageType;
use raftstore::store::util::is_vote_msg;
use rfstore::{
    store::{Callback, WriteResponse},
    Result,
};
use test_raftstore::{configure_for_merge, find_peer, new_peer, sleep_ms};
use test_rfstore::{
    check_messages, must_get_equal, new_node_cluster, shard_must_not_exist, CloneFilterFactory,
    Direction, Filter, IsolationFilterFactory, MessageTypeNotifier, RegionPacketFilter, Simulator,
};
use tikv_util::{
    config::{ReadableDuration, ReadableSize},
    HandyRwLock,
};
use txn_types::Key;

// Filter prevote message and record the range.
struct PrevoteRangeFilter {
    filter: RegionPacketFilter,
    before: Option<mpsc::Sender<(Vec<u8>, Vec<u8>)>>,
    after: Option<mpsc::Sender<()>>,
}

impl Filter for PrevoteRangeFilter {
    fn before(&self, msgs: &mut Vec<RaftMessage>) -> Result<()> {
        self.filter.before(msgs)?;
        if let Some(msg) = msgs.iter().filter(|m| is_vote_msg(m.get_message())).last() {
            let start_key = msg.get_start_key().to_owned();
            let end_key = msg.get_end_key().to_owned();
            if let Some(before) = self.before.as_ref() {
                let _ = before.send((start_key, end_key));
            }
        }
        Ok(())
    }
    fn after(&self, _: Result<()>) -> Result<()> {
        if let Some(after) = self.after.as_ref() {
            let _ = after.send(());
        }
        Ok(())
    }
}

#[test]
fn test_rfstore_follower_slow_split() {
    test_util::init_log_for_test();
    let mut cluster = new_node_cluster(1, 3);
    let pd_client = Arc::clone(&cluster.pd_client);
    pd_client.disable_default_operator();
    cluster.run();
    let region = cluster.get_region(b"");

    // Only need peer 1 and 3. Stop node 2 to avoid extra vote messages.
    cluster.must_transfer_leader(1, new_peer(1, 1));
    pd_client.must_remove_peer(1, new_peer(2, 2));
    cluster.stop_node(2);

    // Use a channel to retrieve start_key and end_key in pre-vote messages.
    let (range_tx, range_rx) = mpsc::channel();
    let prevote_filter = PrevoteRangeFilter {
        // Only send 1 pre-vote message to peer 3 so if peer 3 drops it,
        // it needs to start a new election.
        filter: RegionPacketFilter::new(1000, 1) // new region id is 1000
            .msg_type(MessageType::MsgRequestPreVote)
            .direction(Direction::Send)
            .allow(1),
        before: Some(range_tx),
        after: None,
    };
    cluster
        .sim
        .wl()
        .add_send_filter(1, Box::new(prevote_filter));

    // Ensure pre-vote response is really sended.
    let (tx, rx) = mpsc::channel();
    let prevote_resp_notifier = Box::new(MessageTypeNotifier::new(
        MessageType::MsgRequestPreVoteResponse,
        tx,
        Arc::from(AtomicBool::new(true)),
    ));
    cluster.sim.wl().add_send_filter(3, prevote_resp_notifier);

    // After split, pre-vote message should be sent to peer 2.
    fail::cfg("apply_before_split_1_3", "pause").unwrap();
    let split_key = Key::from_raw(b"k2");
    cluster.must_split(&region, split_key.as_encoded());
    let range = range_rx.recv_timeout(Duration::from_millis(100)).unwrap();
    assert_eq!(range.0, b"");
    assert_eq!(&range.1, split_key.as_encoded());

    // After the follower split success, it will response to the pending vote.
    fail::cfg("apply_before_split_1_3", "off").unwrap();
    rx.recv_timeout(Duration::from_millis(100)).unwrap();
}

// Test if a peer is created from splitting when another initialized peer with
// the same region id has already existed. In previous implementation, it can be
// created and panic will happen because there are two initialized peer with the
// same region id.
#[test]
fn test_split_not_to_split_existing_region() {
    let mut cluster = new_node_cluster(1, 4);
    configure_for_merge(&mut cluster.cfg);
    cluster.cfg.raft_store.right_derive_when_split = true;
    cluster.cfg.raft_store.apply_batch_system.max_batch_size = Some(1);
    cluster.cfg.raft_store.apply_batch_system.pool_size = 2;
    let pd_client = Arc::clone(&cluster.pd_client);
    pd_client.disable_default_operator();

    let r1 = cluster.run_conf_change();
    pd_client.must_add_peer(r1, new_peer(2, 2));
    pd_client.must_add_peer(r1, new_peer(3, 3));

    let mut region_a = pd_client.get_region(b"k1").unwrap();
    // [-∞, k2), [k2, +∞)
    //    b         a
    let split_key_k2 = Key::from_raw(b"k2");
    cluster.must_split(&region_a, split_key_k2.as_encoded());

    cluster.put(b"k0", b"v0").unwrap();

    let region_b = pd_client.get_region(b"k0").unwrap();
    let peer_b_1 = find_peer(&region_b, 1).cloned().unwrap();

    must_get_equal(&cluster.get_engine(3), region_b.id, b"k0", b"v0");

    cluster.must_transfer_leader(region_b.get_id(), peer_b_1);

    let peer_b_3 = find_peer(&region_b, 3).cloned().unwrap();
    assert_eq!(peer_b_3.get_id(), 1003);
    let on_handle_apply_1003_fp = "on_handle_apply_1003";
    fail::cfg(on_handle_apply_1003_fp, "pause").unwrap();
    // [-∞, k1), [k1, k2), [k2, +∞)
    //    c         b          a
    let split_key_k1 = Key::from_raw(b"k1");
    cluster.must_split(&region_b, split_key_k1.as_encoded());

    pd_client.must_remove_peer(region_b.get_id(), peer_b_3);
    pd_client.must_add_peer(region_b.get_id(), new_peer(4, 4));

    let mut region_c = pd_client.get_region(b"k0").unwrap();
    let peer_c_3 = find_peer(&region_c, 3).cloned().unwrap();
    pd_client.must_remove_peer(region_c.get_id(), peer_c_3);
    pd_client.must_add_peer(region_c.get_id(), new_peer(4, 5));
    // [-∞, k2), [k2, +∞)
    //     c        a
    pd_client.must_merge(region_b.get_id(), region_c.get_id());

    region_a = pd_client.get_region(b"k3").unwrap();
    let peer_a_3 = find_peer(&region_a, 3).cloned().unwrap();
    pd_client.must_remove_peer(region_a.get_id(), peer_a_3);
    pd_client.must_add_peer(region_a.get_id(), new_peer(4, 6));
    // [-∞, +∞)
    //    c
    pd_client.must_merge(region_a.get_id(), region_c.get_id());

    region_c = pd_client.get_region(b"k1").unwrap();
    // [-∞, k2), [k2, +∞)
    //     d        c
    cluster.must_split(&region_c, split_key_k2.as_encoded());

    let peer_c_4 = find_peer(&region_c, 4).cloned().unwrap();
    pd_client.must_remove_peer(region_c.get_id(), peer_c_4);
    pd_client.must_add_peer(region_c.get_id(), new_peer(3, 7));

    cluster.put(b"k3", b"v3").unwrap();

    // Remove the failpoint before verifying data on store 3.
    // The failpoint on peer 1003's apply can randomly block peer 7's apply
    // if they are assigned to the same apply worker thread (pool_size=2),
    // preventing peer 7 from completing snapshot application and shard creation.
    fail::remove(on_handle_apply_1003_fp);

    must_get_equal(&cluster.get_engine(3), region_c.id, b"k3", b"v3");

    let r = pd_client.get_region(b"k0").unwrap();
    // If peer_c_3 is created, this should fail.
    assert!(cluster.get_engine(3).get_snap_access(r.id).is_none());
}

// Test if a peer is created from splitting when another initialized peer with
// the same region id existed before and has been destroyed now.
#[test]
fn test_split_not_to_split_existing_tombstone_region() {
    let mut cluster = new_node_cluster(1, 3);
    configure_for_merge(&mut cluster.cfg);
    cluster.cfg.raft_store.right_derive_when_split = true;
    cluster.cfg.raft_store.store_batch_system.max_batch_size = Some(1);
    cluster.cfg.raft_store.store_batch_system.pool_size = 2;
    cluster.cfg.raft_store.apply_batch_system.max_batch_size = Some(1);
    cluster.cfg.raft_store.apply_batch_system.pool_size = 2;
    let pd_client = Arc::clone(&cluster.pd_client);
    pd_client.disable_default_operator();

    fail::cfg("on_raft_log_gc_tick", "return()").unwrap();
    let r1 = cluster.run_conf_change();
    assert_eq!(r1, 1);

    pd_client.must_add_peer(r1, new_peer(3, 3));
    pd_client.must_add_peer(r1, new_peer(2, 2));

    cluster.must_put(b"k1", b"v1");
    cluster.must_put(b"k2", b"v2");

    let region = pd_client.get_region(b"k1").unwrap();
    let split_key = Key::from_raw(b"k2");
    cluster.must_split(&region, split_key.as_encoded());
    cluster.must_put(b"k22", b"v22");

    let left = pd_client.get_region(b"k1").unwrap();
    must_get_equal(&cluster.get_engine(2), left.id, b"k1", b"v1");

    let left_peer_2 = find_peer(&left, 2).cloned().unwrap();
    pd_client.must_remove_peer(left.get_id(), left_peer_2);
    shard_must_not_exist(&cluster.get_engine(2), left.id);

    let on_handle_apply_2_fp = "on_handle_apply_2";
    fail::cfg("on_handle_apply_2", "pause").unwrap();

    // Wait for the logs
    sleep_ms(100);

    // If left_peer_2 can be created, dropping all msg to make it exist.
    cluster.add_send_filter(IsolationFilterFactory::new(2));
    // Also don't send check stale msg to PD
    let peer_check_stale_state_fp = "peer_check_stale_state";
    fail::cfg(peer_check_stale_state_fp, "return()").unwrap();

    fail::remove(on_handle_apply_2_fp);

    let right = pd_client.get_region(b"k22").unwrap();
    // If value of `k22` is equal to `v22`, the previous split log must be applied.
    must_get_equal(&cluster.get_engine(2), right.id, b"k22", b"v22");

    // If left_peer_2 is created, `must_get_none` will fail.
    shard_must_not_exist(&cluster.get_engine(2), left.id);

    cluster.clear_send_filters();

    pd_client.must_add_peer(left.get_id(), new_peer(2, 4));

    must_get_equal(&cluster.get_engine(2), left.id, b"k1", b"v1");
}

/// A filter that collects all snapshots.
///
/// It's different from the one in simulate_transport in three aspects:
/// 1. It will not flush the collected snapshots.
/// 2. It will not report error when collecting snapshots.
/// 3. It callers can access the collected snapshots.
pub struct CollectSnapshotFilter {
    pending_msg: Arc<Mutex<HashMap<u64, RaftMessage>>>,
    pending_count_sender: mpsc::Sender<usize>,
}

impl CollectSnapshotFilter {
    pub fn new(sender: mpsc::Sender<usize>) -> CollectSnapshotFilter {
        CollectSnapshotFilter {
            pending_msg: Arc::default(),
            pending_count_sender: sender,
        }
    }
}

impl Filter for CollectSnapshotFilter {
    fn before(&self, msgs: &mut Vec<RaftMessage>) -> Result<()> {
        let mut to_send = vec![];
        let mut pending_msg = self.pending_msg.lock().unwrap();
        for msg in msgs.drain(..) {
            let (is_pending, from_peer_id) = {
                if msg.get_message().get_msg_type() == MessageType::MsgSnapshot {
                    let from_peer_id = msg.get_from_peer().get_id();
                    if pending_msg.contains_key(&from_peer_id) {
                        // Drop this snapshot message directly since it's from a seen peer
                        continue;
                    } else {
                        // Pile the snapshot from unseen peer
                        (true, from_peer_id)
                    }
                } else {
                    (false, 0)
                }
            };
            if is_pending {
                pending_msg.insert(from_peer_id, msg);
                self.pending_count_sender.send(pending_msg.len()).unwrap();
            } else {
                to_send.push(msg);
            }
        }
        msgs.extend(to_send);
        check_messages(msgs)?;
        Ok(())
    }
}

/// If the uninitialized peer and split peer are fetched into one batch, and the
/// first one doesn't generate ready, the second one does, ready should not be
/// mapped to the first one.
#[test]
fn test_split_duplicated_batch() {
    let mut cluster = new_node_cluster(1, 3);
    cluster.cfg.raft_store.raft_log_gc_size_limit = Some(ReadableSize::mb(20));
    // Disable raft log gc in this test case.
    cluster.cfg.raft_store.raft_log_gc_tick_interval = ReadableDuration::secs(60);
    // Use one thread to make it more possible to be fetched into one batch.
    cluster.cfg.raft_store.store_batch_system.pool_size = 1;

    let pd_client = Arc::clone(&cluster.pd_client);
    // Disable default max peer count check.
    pd_client.disable_default_operator();

    let r1 = cluster.run_conf_change();
    cluster.must_put(b"k1", b"v1");
    pd_client.must_add_peer(r1, new_peer(2, 2));
    // Force peer 2 to be followers all the way.
    cluster.add_send_filter(CloneFilterFactory(
        RegionPacketFilter::new(r1, 2)
            .msg_type(MessageType::MsgRequestVote)
            .direction(Direction::Send),
    ));
    cluster.must_transfer_leader(r1, new_peer(1, 1));
    cluster.must_put(b"k3", b"v3");

    // Pile up snapshots of overlapped region ranges
    let (tx, rx) = mpsc::channel();
    let filter = CollectSnapshotFilter::new(tx);
    let pending_msgs = filter.pending_msg.clone();
    cluster.sim.wl().add_recv_filter(3, Box::new(filter));
    pd_client.must_add_peer(r1, new_peer(3, 3));
    let region = cluster.get_region(b"k1");
    // Ensure the snapshot of range ("", "") is sent and piled in filter.
    if let Err(e) = rx.recv_timeout(Duration::from_secs(1)) {
        panic!("the snapshot is not sent before split, e: {:?}", e);
    }
    // Split the region range and then there should be another snapshot for the
    // split ranges.
    let split_k2 = Key::from_raw(b"k2");
    cluster.must_split(&region, split_k2.as_encoded());
    // Ensure second is also sent and piled in filter.
    if let Err(e) = rx.recv_timeout(Duration::from_secs(1)) {
        panic!("the snapshot is not sent before split, e: {:?}", e);
    }

    let (tx1, rx1) = mpsc::sync_channel(0);
    fail::cfg_callback("on_split", move || {
        // First is for notification, second is waiting for configuration.
        let _ = tx1.send(());
        let _ = tx1.send(());
    })
    .unwrap();

    let r2 = cluster.get_region(b"k0");
    let filter_r2 = Arc::new(AtomicBool::new(true));
    // So uninitialized peer will not generate ready for response.
    cluster.add_send_filter(CloneFilterFactory(
        RegionPacketFilter::new(r2.get_id(), 3)
            .when(filter_r2.clone())
            .direction(Direction::Recv),
    ));
    // So peer can catch up logs and execute split
    cluster.add_send_filter(CloneFilterFactory(
        RegionPacketFilter::new(r1, 3)
            .msg_type(MessageType::MsgSnapshot)
            .direction(Direction::Recv),
    ));
    cluster.sim.wl().clear_recv_filters(3);
    // Start applying snapshot in source peer.
    for (peer_id, msg) in pending_msgs.lock().unwrap().iter() {
        if *peer_id < 1000 {
            cluster.sim.wl().send_raft_msg(msg.clone()).unwrap();
        }
    }

    let (tx2, rx2) = mpsc::sync_channel(0);
    // r1 has split.
    rx1.recv_timeout(Duration::from_secs(3)).unwrap();
    // Notify uninitialized peer to be ready be fetched at next try.
    for (peer_id, msg) in pending_msgs.lock().unwrap().iter() {
        if *peer_id >= 1000 {
            cluster.sim.wl().send_raft_msg(msg.clone()).unwrap();
        }
    }
    let tx2 = Mutex::new(tx2);
    fail::cfg_callback("after_split", move || {
        // First is for notification, second is waiting for configuration.
        let _ = tx2.lock().unwrap().send(());
        let _ = tx2.lock().unwrap().send(());
    })
    .unwrap();
    // Resume on_split hook.
    rx1.recv_timeout(Duration::from_secs(3)).unwrap();
    // Pause at the end of on_split.
    rx2.recv_timeout(Duration::from_secs(3)).unwrap();
    // New peer is generated, no need to filter any more.
    filter_r2.store(false, Ordering::SeqCst);
    // Force generating new messages so split peer will be notified and ready to
    // be fetched at next try.
    cluster.must_put(b"k11", b"v11");
    // Exit on_split hook.
    rx2.recv_timeout(Duration::from_secs(3)).unwrap();
    must_get_equal(&cluster.get_engine(3), r2.id, b"k11", b"v11");
}

/// This test case test if a split failed for some reason,
/// it can continue run split check and eventually the split will finish
#[test]
fn test_split_by_split_check_on_size() {
    let mut cluster = new_node_cluster(1, 1);
    cluster.cfg.raft_store.right_derive_when_split = true;
    cluster.cfg.raft_store.split_region_check_tick_interval = ReadableDuration::millis(50);
    cluster.cfg.raft_store.pd_heartbeat_tick_interval = ReadableDuration::millis(100);
    cluster.cfg.raft_store.region_split_check_diff = Some(ReadableSize(10));
    cluster.cfg.coprocessor.region_max_keys = Some(12);
    cluster.cfg.coprocessor.region_split_keys = Some(8);
    let pd_client = cluster.pd_client.clone();
    pd_client.disable_default_operator();
    let _r = cluster.run_conf_change();

    // make first split fail
    // 1*return means it would run "return" action once
    fail::cfg("fail_pre_propose_split", "1*return").unwrap();

    // Insert region_max_keys into the cluster.
    // It should trigger the split
    for i in 0..7 {
        let key = format!("k{:02}", i);
        cluster.must_put(key.as_bytes(), b"v0");
    }
    let region = pd_client.get_region(b"k06").unwrap();
    for i in 7..13 {
        let key = format!("k{:02}", i);
        cluster.must_put(key.as_bytes(), b"v0");
    }
    // waiting the split,
    cluster.wait_region_split(&region);
}

/// Test that if the original leader of the parent region is tranfered to
/// another peer, the new leader of the parent region will notify the new split
/// region to campaign.
#[test]
fn test_region_split_after_parent_leader_transfer() {
    let mut cluster = new_node_cluster(1, 3);
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

    // Setting: only peers on store 2 can become leader.
    for id in [1, 3] {
        cluster.add_send_filter(CloneFilterFactory(
            RegionPacketFilter::new(region.get_id(), id)
                .msg_type(MessageType::MsgRequestPreVote)
                .direction(Direction::Send),
        ));
        cluster.add_send_filter(CloneFilterFactory(
            RegionPacketFilter::new(1000, id)
                .msg_type(MessageType::MsgRequestPreVote)
                .direction(Direction::Send),
        ));
    }

    // Split region to peer 1 & 2, not allow peer 3 (leader) to split.
    let no_split_on_store_3 = "on_split";
    fail::cfg(no_split_on_store_3, "pause").unwrap();
    let split_k2 = Key::from_raw(b"k2");
    cluster.split_region(
        &region,
        split_k2.as_encoded(),
        Callback::write(Box::new(move |_write_resp: WriteResponse| {})),
    );
    // Wait the old lease of the leader timeout and peer 2 gets votes
    // to become the new leader.
    std::thread::sleep(
        cluster.cfg.raft_store.raft_base_tick_interval.0
            * cluster.cfg.raft_store.raft_election_timeout_ticks as u32
            * 2,
    );
    // As the split is paused, the leader of the parent region should
    // be peer 2, not peer 3. And peer 2 will notify the new split region
    //  `campaign` to become leader.
    cluster.reset_leader_of_region(region.get_id());
    assert_eq!(
        cluster.leader_of_region(region.get_id()).unwrap(),
        new_peer(2, 2)
    );
    // The leader of the new split region should be peer 1002.
    let new_region = pd_client.get_region(b"k1").unwrap();
    assert_eq!(
        cluster.leader_of_region(new_region.get_id()).unwrap(),
        new_peer(2, 1002)
    );
    fail::remove(no_split_on_store_3);
}

/// Test that the leader of the new split region will not be changed after
/// the leader of the parent region is transferred.
#[test]
fn test_region_split_after_new_leader_elected() {
    let mut cluster = new_node_cluster(1, 3);
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

    // Setting: only peers on store 2 can become leader.
    for id in [1, 3] {
        cluster.add_send_filter(CloneFilterFactory(
            RegionPacketFilter::new(region.get_id(), id)
                .msg_type(MessageType::MsgRequestPreVote)
                .direction(Direction::Send),
        ));
    }

    // Split region to peer 1 & 2, not allow peer 3 (leader) to split.
    let skip_clear_uncampaign = "on_skip_check_uncampaigned_regions";
    fail::cfg(skip_clear_uncampaign, "return").unwrap();
    let no_split_on_store_3 = "on_split";
    fail::cfg(no_split_on_store_3, "pause").unwrap();
    let split_k2 = Key::from_raw(b"k2");
    cluster.split_region(
        &region,
        split_k2.as_encoded(),
        Callback::write(Box::new(move |_write_resp: WriteResponse| {})),
    );
    // Wait the leader of the new split region has been elected.
    std::thread::sleep(
        cluster.cfg.raft_store.raft_base_tick_interval.0
            * cluster.cfg.raft_store.raft_election_timeout_ticks as u32
            * 2,
    );
    cluster.reset_leader_of_region(region.get_id());
    assert_eq!(
        cluster.leader_of_region(region.get_id()).unwrap(),
        new_peer(2, 2)
    );
    // The leader of the new split region should be elected.
    let new_region = pd_client.get_region(b"k1").unwrap();
    let new_region_leader = cluster.leader_of_region(new_region.get_id()).unwrap();
    // The new leader will notify the new split region  `campaign` to become
    // leader, but the leader of the new split region is already elected.
    fail::remove(no_split_on_store_3);
    // The leader of the new split region should not changed.
    cluster.reset_leader_of_region(new_region.get_id());
    assert_eq!(
        cluster.leader_of_region(new_region.get_id()).unwrap(),
        new_region_leader
    );
    fail::remove(skip_clear_uncampaign);
}
