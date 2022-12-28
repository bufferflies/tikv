// Copyright 2022 TiKV Project Authors. Licensed under Apache-2.0.

use std::{thread, time::Duration};

use futures::executor::block_on;
use pd_client::PdClient;
use raftstore::store::util::{find_peer, new_learner_peer};
use test_cloud_server::{try_wait, ServerCluster};
use tikv::config::TiKvConfig;
use tikv_util::config::ReadableDuration;

use crate::alloc_node_id;

#[test]
fn test_region_merge() {
    test_util::init_log_for_test();
    let node_ids = vec![alloc_node_id(), alloc_node_id(), alloc_node_id()];
    let mut cluster = ServerCluster::new(node_ids.clone(), |_, _| {});
    cluster.wait_region_replicated(&[], 3);
    let mut client = cluster.new_client();
    let split_key = i_to_key(5);
    client.split(&split_key);
    cluster.wait_pd_region_count(2);
    client.put_kv(0..10, i_to_key, i_to_val);
    client.merge(&i_to_key(0), &i_to_key(10));
    cluster.wait_pd_region_count(1);
    client.verify_data_with_ref_store();
    for &node_id in &node_ids {
        cluster.stop_node(node_id);
    }
    for &node_id in &node_ids {
        cluster.start_node(node_id, |_, _| {});
    }
    client.verify_data_with_ref_store();
    cluster.stop();
}

/// Test if a peer can be destroyed properly in such conditions as follows
/// 1. A peer is isolated
/// 2. Then its region merges to another region.
/// 3. Isolation disappears
#[test]
fn test_region_merge_isolated_peer() {
    test_util::init_log_for_test();
    let node_ids = vec![
        alloc_node_id(),
        alloc_node_id(),
        alloc_node_id(),
        alloc_node_id(),
    ];
    let update_conf_fn = |_, conf: &mut TiKvConfig| {
        conf.raft_store.peer_stale_state_check_interval = ReadableDuration::secs(1);
        conf.raft_store.abnormal_leader_missing_duration = ReadableDuration::secs(3);
        conf.raft_store.max_leader_missing_duration = ReadableDuration::secs(5);
    };
    let mut cluster = ServerCluster::new(node_ids.clone(), update_conf_fn);
    let pd_client = cluster.get_pd_client();
    let mut client = cluster.new_client();
    cluster.wait_region_replicated(&[], 3);
    pd_client.disable_default_operator();

    let split_key = i_to_key(5);
    client.split(&split_key);
    cluster.wait_pd_region_count(2);
    client.put_kv(0..10, i_to_key, i_to_val);

    let left_id = client.get_region_id(&i_to_key(0));
    let left = block_on(pd_client.get_region_by_id(left_id))
        .unwrap()
        .unwrap();

    let isolated_node_id = node_ids
        .iter()
        .find(|node_id| {
            let store_id = cluster.get_store_id(**node_id);
            find_peer(&left, store_id).is_none()
        })
        .unwrap()
        .to_owned();
    let isolated_store_id = cluster.get_store_id(isolated_node_id);
    let learner_peer = new_learner_peer(isolated_store_id, 2);

    pd_client.must_add_peer(left_id, learner_peer.clone());
    // Ensure this learner exists.
    assert!(try_wait(
        || {
            cluster
                .get_kvengine(isolated_node_id)
                .get_shard(left_id)
                .is_some()
        },
        10
    ));

    cluster.stop_node(isolated_node_id);
    pd_client.must_remove_peer(left_id, learner_peer);
    thread::sleep(Duration::from_millis(500)); // Wait for node to stop completely.

    client.merge(&i_to_key(0), &i_to_key(10));
    cluster.wait_pd_region_count(1);
    client.verify_data_with_ref_store();

    cluster.start_node(isolated_node_id, update_conf_fn);
    assert!(try_wait(
        || {
            cluster
                .get_kvengine(isolated_node_id)
                .get_shard(left_id)
                .is_none()
        },
        10
    ));

    cluster.get_data_stats().check_data().unwrap();
    client.verify_data_with_ref_store();
    cluster.stop();
}

fn i_to_key(i: usize) -> Vec<u8> {
    format!("key_{:03}", i).into_bytes()
}

fn i_to_val(i: usize) -> Vec<u8> {
    format!("val_{:03}", i).into_bytes().repeat(3)
}
