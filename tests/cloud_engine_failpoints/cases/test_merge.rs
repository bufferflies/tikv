// Copyright 2023 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    sync::{
        atomic::{AtomicUsize, Ordering},
        mpsc::channel,
        Arc,
    },
    thread::JoinHandle,
    time::Duration,
};

use kvproto::raft_serverpb::PeerState;
use pd_client::PdClient;
use rfstore::store::Callback;
use test_cloud_server::{client::ClusterClient, ServerCluster};
use test_raftstore::{configure_for_merge, new_peer, peer_on_store, sleep_ms};
use test_rfstore::{
    load_region_local_state, must_get_equal, must_get_none, new_node_cluster, new_put_cmd,
    new_write_request, IsolationFilterFactory,
};
use tikv_util::{config::ReadableSize, info, store::find_peer, time::Instant};
use txn_types::Key;

use crate::cases::{alloc_node_id_vec, i_to_key, i_to_val};

// Test for the scenario that prepare merge & rollback merge have write conflict
// on pending merge state.
// See https://github.com/tidbcloud/cloud-storage-engine/issues/664
#[test]
fn test_merge_pending_state_conflict() {
    const TEST_DURATION: Duration = Duration::from_secs(10);
    const APPLY_DELAY: &str = "sleep(100)";

    let schedule_merge_error_fp = "on_schedule_merge_error";
    let on_follower_exec_rollback_merge_fp = "on_follower_exec_rollback_merge";

    test_util::init_log_for_test();
    let mut cluster = ServerCluster::new(alloc_node_id_vec(3), |_, _| {});
    cluster.wait_region_replicated(&[], 3);
    cluster.get_pd_client().disable_default_operator();

    let mut client = cluster.new_client();
    let prev_keyspace = i_to_key(0);
    client.split(&prev_keyspace);

    let split_key = i_to_key(5);
    client.split(&split_key);
    cluster.wait_pd_region_count(3);

    client.put_kv(0..10, i_to_key, i_to_val);

    // `schedule_merge` returns error to make merges rollback.
    fail::cfg(schedule_merge_error_fp, "return").unwrap();

    let merge_thread = spawn_merge(
        cluster.new_client(),
        i_to_key(0),
        i_to_key(10),
        TEST_DURATION,
    );

    // Delay the apply of rollback merge.
    // The delay is applied to followers only, as this issue would not happen on
    // leader (when there is a pending merge, leader will reject other merge
    // proposals).
    fail::cfg(on_follower_exec_rollback_merge_fp, APPLY_DELAY).unwrap();

    merge_thread.join().unwrap();

    fail::remove(schedule_merge_error_fp);
    fail::remove(on_follower_exec_rollback_merge_fp);

    // `try_merge` may be rollback by target region changed.
    for _ in 0..10 {
        client.try_merge(&i_to_key(0), &i_to_key(10));
        if cluster.get_pd_client_ext().get_regions_number() == 2 {
            break;
        }
        std::thread::sleep(Duration::from_secs(1));
    }

    client.verify_data_with_ref_store();
    cluster.stop();
}

fn spawn_merge(
    mut client: ClusterClient,
    mut source_key: Vec<u8>,
    mut target_key: Vec<u8>,
    duration: Duration,
) -> JoinHandle<()> {
    let merge_counter = AtomicUsize::new(0);
    std::thread::spawn(move || {
        let start_time = Instant::now();
        while start_time.saturating_elapsed() < duration {
            let req_sent = client.try_merge(&source_key, &target_key);
            assert!(req_sent);
            merge_counter.fetch_add(1, Ordering::SeqCst);

            // Swap keys to propose more merges, and make issue reproduction more easily.
            std::mem::swap(&mut source_key, &mut target_key);
        }
        info!(
            "merge thread exit, merge counter {}",
            merge_counter.load(Ordering::Relaxed)
        );
    })
}

#[test]
fn test_rfstore_node_merge_rollback() {
    let mut cluster = new_node_cluster(1, 3);
    configure_for_merge(&mut cluster.cfg);
    let pd_client = Arc::clone(&cluster.pd_client);
    pd_client.disable_default_operator();

    cluster.run_conf_change();

    let region = pd_client.get_region(b"k1").unwrap();
    let split_key = Key::from_raw(b"k2");
    cluster.must_split(&region, split_key.as_encoded());
    let left = pd_client.get_region(b"k1").unwrap();
    let right = pd_client.get_region(b"k3").unwrap();

    pd_client.must_add_peer(left.get_id(), new_peer(2, 2));
    pd_client.must_add_peer(right.get_id(), new_peer(2, 4));

    cluster.must_put(b"k1", b"v1");
    cluster.must_put(b"k3", b"v3");

    let region = pd_client.get_region(b"k1").unwrap();
    let target_region = pd_client.get_region(b"k3").unwrap();

    let schedule_merge_fp = "on_schedule_merge";
    fail::cfg(schedule_merge_fp, "return()").unwrap();

    let (tx, rx) = channel();
    fail::cfg_callback("on_prepare_merge_result", move || {
        tx.send(()).unwrap();
    })
    .unwrap();

    cluster.merge_region(region.get_id(), target_region.get_id(), Callback::None);
    // PrepareMerge is applied.
    rx.recv_timeout(Duration::from_secs(5)).unwrap();

    // Add a peer to trigger rollback.
    pd_client.must_add_peer(right.get_id(), new_peer(3, 5));
    cluster.must_put(b"k4", b"v4");
    must_get_equal(&cluster.get_engine(3), right.get_id(), b"k4", b"v4");

    let mut region = pd_client.get_region(b"k1").unwrap();
    // After split and prepare_merge, version becomes 1 + 2 = 3;
    assert_eq!(region.get_region_epoch().get_version(), 3);
    // After ConfChange and prepare_merge, conf version becomes 1 + 2 = 3;
    assert_eq!(region.get_region_epoch().get_conf_ver(), 3);
    fail::remove(schedule_merge_fp);
    // Wait till rollback.
    cluster.must_put(b"k11", b"v11");

    // After rollback, version becomes 3 + 1 = 4;
    region.mut_region_epoch().set_version(4);
    for i in 1..3 {
        must_get_equal(&cluster.get_engine(i), region.get_id(), b"k11", b"v11");
        let peer = peer_on_store(&region, i);
        let state = cluster.region_local_state(peer.id, i).unwrap();
        assert_eq!(state.get_state(), PeerState::Normal);
        assert_eq!(*state.get_region(), region);
    }

    pd_client.must_remove_peer(right.get_id(), new_peer(3, 5));
    fail::cfg(schedule_merge_fp, "return()").unwrap();

    let target_region = pd_client.get_region(b"k3").unwrap();
    cluster.merge_region(region.get_id(), target_region.get_id(), Callback::None);
    // PrepareMerge is applied.
    rx.recv().unwrap();

    let mut region = pd_client.get_region(b"k1").unwrap();

    // Split to trigger rollback.
    let split_key = Key::from_raw(b"k3");
    cluster.must_split(&right, split_key.as_encoded());
    fail::remove(schedule_merge_fp);
    // Wait till rollback.
    cluster.must_put(b"k12", b"v12");

    // After premerge and rollback, conf_ver becomes 3 + 1 = 4, version becomes 4 +
    // 2 = 6;
    region.mut_region_epoch().set_conf_ver(4);
    region.mut_region_epoch().set_version(6);
    for i in 1..3 {
        must_get_equal(&cluster.get_engine(i), region.get_id(), b"k12", b"v12");
        let peer = peer_on_store(&region, i);
        let state = cluster.region_local_state(peer.id, i).unwrap();
        assert_eq!(state.get_state(), PeerState::Normal);
        assert_eq!(*state.get_region(), region);
    }
}

/// Test if merge is still working when restart a cluster during merge.
#[test]
fn test_rfstore_node_merge_restart() {
    let mut cluster = new_node_cluster(1, 3);
    configure_for_merge(&mut cluster.cfg);
    let pd_client = Arc::clone(&cluster.pd_client);
    pd_client.disable_default_operator();
    cluster.run();

    cluster.must_put(b"k1", b"v1");
    cluster.must_put(b"k3", b"v3");

    let region = pd_client.get_region(b"k1").unwrap();
    let split_key = Key::from_raw(b"k2");
    cluster.must_split(&region, split_key.as_encoded());
    let left = pd_client.get_region(b"k1").unwrap();
    let right = pd_client.get_region(b"k3").unwrap();

    cluster.must_put(b"k1", b"v11");
    cluster.must_put(b"k3", b"v31");

    let schedule_merge_fp = "on_schedule_merge";
    fail::cfg(schedule_merge_fp, "return()").unwrap();

    test_util::eventually(
        Duration::from_millis(100),
        Duration::from_millis(3000),
        || {
            let resp = cluster.try_merge(left.id, right.id);
            !resp.get_header().has_error()
        },
    );
    // cluster.must_try_merge(left.id, right.id);
    let leader = cluster.leader_of_region(left.get_id()).unwrap();

    cluster.shutdown();
    let rf_engine = cluster.get_raft_engine(leader.get_store_id());
    let state = load_region_local_state(&rf_engine, leader.id).unwrap();
    assert_eq!(state.get_state(), PeerState::Merging, "{:?}", state);
    let right_peer = find_peer(&right, leader.store_id).unwrap().id;
    let state = load_region_local_state(&rf_engine, right_peer).unwrap();
    assert_eq!(state.get_state(), PeerState::Normal, "{:?}", state);
    fail::remove(schedule_merge_fp);
    cluster.start().unwrap();

    // Wait till merge is finished.
    pd_client.check_merged_timeout(left.get_id(), Duration::from_secs(5));

    cluster.must_put(b"k4", b"v4");

    for i in 1..4 {
        must_get_equal(&cluster.get_engine(i), right.id, b"k4", b"v4");
        let rf_engine = cluster.get_raft_engine(i);
        let peer_id = find_peer(&left, i).unwrap().id;
        let state = load_region_local_state(&rf_engine, peer_id).unwrap();
        assert_eq!(state.get_state(), PeerState::Tombstone, "{:?}", state);
        let peer_id = find_peer(&right, i).unwrap().id;
        let state = load_region_local_state(&rf_engine, peer_id).unwrap();
        assert_eq!(state.get_state(), PeerState::Normal, "{:?}", state);
        assert!(state.get_region().get_start_key().is_empty());
        assert!(state.get_region().get_end_key().is_empty());
    }

    // Now test if cluster works fine when it crash after merge is applied
    // but before notifying raftstore thread.
    let region = pd_client.get_region(b"k1").unwrap();
    let peer_on_store1 = find_peer(&region, 1).unwrap().to_owned();
    cluster.must_transfer_leader(region.get_id(), peer_on_store1);
    cluster.must_split(&region, split_key.as_encoded());
    let left = pd_client.get_region(b"k1").unwrap();
    let right = pd_client.get_region(b"k3").unwrap();
    let peer_on_store1 = find_peer(&left, 1).unwrap().to_owned();
    cluster.must_transfer_leader(left.get_id(), peer_on_store1);
    cluster.must_put(b"k11", b"v11");
    must_get_equal(&cluster.get_engine(3), left.id, b"k11", b"v11");
    let skip_destroy_fp = "destroy_peer";
    fail::cfg(skip_destroy_fp, "return()").unwrap();
    cluster.add_send_filter(IsolationFilterFactory::new(3));
    pd_client.must_merge(left.get_id(), right.get_id());
    let peer = find_peer(&right, 3).unwrap().to_owned();
    pd_client.must_remove_peer(right.get_id(), peer);
    cluster.shutdown();
    fail::remove(skip_destroy_fp);
    cluster.clear_send_filters();
    cluster.start().unwrap();
    std::thread::sleep(Duration::from_secs(3));
    test_util::eventually(
        Duration::from_millis(100),
        Duration::from_millis(1000),
        || cluster.get_engine(3).get_snap_access(right.id).is_none(),
    );
}

/// Test if merging state will be removed after accepting a snapshot.
#[test]
fn test_rfstore_node_merge_recover_snapshot() {
    let mut cluster = new_node_cluster(1, 3);
    configure_for_merge(&mut cluster.cfg);
    cluster.cfg.raft_store.raft_log_gc_size_limit = Some(ReadableSize::kb(1));
    let pd_client = Arc::clone(&cluster.pd_client);
    pd_client.disable_default_operator();

    // Start the cluster and evict the region leader from peer 3.
    cluster.run();
    cluster.must_transfer_leader(1, new_peer(1, 1));

    let region = pd_client.get_region(b"k1").unwrap();
    let split_key = Key::from_raw(b"k2");
    cluster.must_split(&region, split_key.as_encoded());
    let left = pd_client.get_region(b"k1").unwrap();

    cluster.must_put(b"k1", b"v1");
    cluster.must_put(b"k3", b"v3");

    let region = pd_client.get_region(b"k3").unwrap();
    let target_region = pd_client.get_region(b"k1").unwrap();

    let schedule_merge_fp = "on_schedule_merge";
    fail::cfg(schedule_merge_fp, "return()").unwrap();

    cluster.must_try_merge(region.get_id(), target_region.get_id());

    // Remove a peer to trigger rollback.
    pd_client.must_remove_peer(left.get_id(), left.get_peers()[0].to_owned());
    must_get_none(&cluster.get_engine(3), region.id, b"k4");

    let step_store_3_region_1 = "step_message_3_1";
    fail::cfg(step_store_3_region_1, "return()").unwrap();
    fail::remove(schedule_merge_fp);

    for i in 0..100 {
        cluster.must_put(format!("k4{}", i).as_bytes(), b"v4");
    }
    fail::remove(step_store_3_region_1);
    must_get_equal(&cluster.get_engine(3), region.id, b"k40", b"v4");
    cluster.must_transfer_leader(1, new_peer(3, 3));
    cluster.must_put(b"k40", b"v5");
}

// Test if the rollback merge proposal is proposed before the majority of peers
// want to rollback
#[test]
fn test_rfstore_node_multiple_rollback_merge() {
    let mut cluster = new_node_cluster(1, 3);
    configure_for_merge(&mut cluster.cfg);
    cluster.cfg.raft_store.right_derive_when_split = true;
    let pd_client = Arc::clone(&cluster.pd_client);
    pd_client.disable_default_operator();

    cluster.run();

    for i in 0..10 {
        cluster.must_put(format!("k{}", i).as_bytes(), b"v");
    }

    let region = pd_client.get_region(b"k1").unwrap();
    let split_key = Key::from_raw(b"k2");
    cluster.must_split(&region, split_key.as_encoded());

    let left = pd_client.get_region(b"k1").unwrap();
    let right = pd_client.get_region(b"k3").unwrap();

    let left_peer_1 = find_peer(&left, 1).unwrap().to_owned();
    cluster.must_transfer_leader(left.get_id(), left_peer_1.clone());
    assert_eq!(left_peer_1.get_id(), 1001);

    let on_schedule_merge_fp = "on_schedule_merge";
    let on_check_merge_not_1001_fp = "on_check_merge_not_1001";

    let mut right_peer_1_id = find_peer(&right, 1).unwrap().get_id();

    for i in 0..3 {
        fail::cfg(on_schedule_merge_fp, "return()").unwrap();
        cluster.must_try_merge(left.get_id(), right.get_id());
        // Change the epoch of target region and the merge will fail
        pd_client.must_remove_peer(right.get_id(), new_peer(1, right_peer_1_id));
        right_peer_1_id += 100;
        pd_client.must_add_peer(right.get_id(), new_peer(1, right_peer_1_id));
        // Only the source leader is running `on_check_merge`
        fail::cfg(on_check_merge_not_1001_fp, "return()").unwrap();
        fail::remove(on_schedule_merge_fp);
        // In previous implementation, rollback merge proposal can be proposed by leader
        // itself So wait for the leader propose rollback merge if possible
        sleep_ms(100);
        // Check if the source region is still in merging mode.
        let mut l_r = pd_client.get_region(b"k1").unwrap();
        let req = new_write_request(
            l_r.get_id(),
            l_r.take_region_epoch(),
            new_put_cmd(format!("k1{}", i).as_bytes(), b"vv"),
        );
        let resp = cluster
            .call_command_on_leader(req, Duration::from_millis(100))
            .unwrap();
        if !resp
            .get_header()
            .get_error()
            .get_message()
            .contains("merging mode")
        {
            panic!("resp {:?} does not contain merging mode error", resp);
        }

        fail::remove(on_check_merge_not_1001_fp);
        // Write data for waiting the merge to rollback easily
        cluster.must_put(format!("k1{}", i).as_bytes(), b"vv");
        // Make sure source region is not merged to target region
        assert_eq!(pd_client.get_region(b"k1").unwrap().get_id(), left.get_id());
    }
}
