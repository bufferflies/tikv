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
use test_raftstore::{adjust_config_for_merge, new_peer};
use test_rfstore::{must_get_equal, new_node_cluster};
use tikv_util::{info, time::Instant};
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
    test_util::init_log_for_test();

    let mut cluster = new_node_cluster(1, 3);
    adjust_config_for_merge(&mut cluster.cfg);
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
        let state = cluster.region_local_state(region.get_id(), i);
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
        let state = cluster.region_local_state(region.get_id(), i);
        assert_eq!(state.get_state(), PeerState::Normal);
        assert_eq!(*state.get_region(), region);
    }
}
