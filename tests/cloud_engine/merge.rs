// Copyright 2022 TiKV Project Authors. Licensed under Apache-2.0.

use std::{thread, time::Duration};

use futures::executor::block_on;
use pd_client::PdClient;
use test_cloud_server::{try_wait, ServerCluster};
use tikv::config::TikvConfig;
use tikv_util::{
    config::ReadableDuration,
    store::{find_peer, new_learner_peer},
};

use crate::{
    alloc_node_id, generate_keyspace_key, get_keyspace_prefix, i_to_key, i_to_val,
    is_region_belongs_to_keyspace,
};

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
    let update_conf_fn = |_, conf: &mut TikvConfig| {
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

#[test]
fn test_region_split_merge_without_inner_key_offset() {
    region_split_merge_inner_key_offset(false);
}

#[test]
fn test_region_split_merge_with_inner_key_offset() {
    region_split_merge_inner_key_offset(true);
}

fn region_split_merge_inner_key_offset(enabled: bool) {
    test_util::init_log_for_test();

    let expected_inner_key_off = if enabled { 4 } else { 0 };

    let node_ids = vec![alloc_node_id(), alloc_node_id(), alloc_node_id()];
    let mut cluster = ServerCluster::new(node_ids.clone(), |_, conf: &mut TikvConfig| {
        conf.enable_inner_key_offset = enabled;
    });
    cluster.wait_region_replicated(&[], 3);

    let mut client = cluster.new_client();
    let pd_client = cluster.get_pd_client();
    // 1. Generate 1 keyspace region.
    // 2. Split it into 10 regions.
    // 3. Check all regions inner_key_off.
    // 4. Merge 10 inner regions into 1 region.
    // 5. Check the keyspace region's inner_key_off.
    let (ks100, ks101) = (get_keyspace_prefix(100), get_keyspace_prefix(101));
    let generate_ks100_keys = generate_keyspace_key(100);
    let split_keys = vec![ks100.clone(), ks101.clone()];
    for sk in split_keys {
        client.split(sk.as_slice());
    }
    cluster.wait_pd_region_count(3);

    // split keyspace inner regions
    for i in 0..10 {
        let split_key = generate_ks100_keys(i);
        client.split(split_key.as_slice());
    }

    cluster.wait_pd_region_count(13);

    // check inner_key_off
    let mut regions = pd_client.get_all_regions();
    regions.sort_by(|a, b| a.get_start_key().cmp(b.get_start_key()));

    for region in &regions {
        if is_region_belongs_to_keyspace(region, 100) {
            for node_id in &node_ids {
                let kv_engine = cluster.get_kvengine(*node_id);
                let region_id = region.get_id();
                let shard = kv_engine.get_shard(region_id).unwrap();
                assert_eq!(shard.range.inner_key_off, expected_inner_key_off);
            }
        }
    }

    // merge inner regions
    for region in &regions {
        if is_region_belongs_to_keyspace(region, 100) {
            let start_key = region.get_start_key();
            let end_key = region.get_end_key();
            if end_key.starts_with(ks101.as_slice()) {
                continue;
            }

            client
                .try_merge_adjacent_region(
                    start_key,
                    Some(ks100.as_slice()),
                    Duration::from_secs(3),
                )
                .unwrap();
        }
    }

    cluster.wait_pd_region_count(3);

    // check inner_key_off
    let regions = pd_client.get_all_regions();
    for region in &regions {
        if is_region_belongs_to_keyspace(region, 100) {
            for node_id in &node_ids {
                let region_id = region.get_id();
                let kv_engine = cluster.get_kvengine(*node_id);
                let shard = kv_engine.get_shard(region_id).unwrap();
                assert_eq!(shard.range.inner_key_off, expected_inner_key_off);
            }
        }
    }

    for &node_id in &node_ids {
        cluster.stop_node(node_id);
    }
    for &node_id in &node_ids {
        cluster.start_node(node_id, |_, _| {});
    }
    client.verify_data_with_ref_store();
    cluster.stop();
}
