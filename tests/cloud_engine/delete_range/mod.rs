// Copyright 2022 TiKV Project Authors. Licensed under Apache-2.0.

use std::{thread, time::Duration};

use kvproto::kvrpcpb::UnsafeDestroyRangeRequest;
use pd_client::PdClient;
use test_cloud_server::{client::ClusterClient, ServerCluster};
use tidb_query_common::util::convert_to_prefix_next;

use crate::alloc_node_id;

#[test]
fn test_delete_range() {
    test_util::init_log_for_test();
    let node_id = alloc_node_id();
    let mut cluster = ServerCluster::new(vec![node_id], |_, _| {});
    let mut client = cluster.new_client();
    // insert "key_100".."key_500"
    client.put_kv(100..500, i_to_key, i_to_val);
    let store_id = cluster.get_stores()[0];
    destroy_range(&mut client, store_id, "key_20".as_bytes());
    let snap = cluster.get_snap(node_id, "key_".as_bytes());
    assert!(!snap.has_data_in_prefix("key_200".as_bytes()));
    assert!(!snap.has_data_in_prefix("key_209".as_bytes()));

    assert!(snap.has_data_in_prefix("key_24".as_bytes()));

    destroy_range(&mut client, store_id, "key_22".as_bytes());
    let snap = cluster.get_snap(node_id, "key_".as_bytes());
    assert!(!snap.has_data_in_prefix("key_20".as_bytes()));
    assert!(!snap.has_data_in_prefix("key_22".as_bytes()));

    assert!(snap.has_data_in_prefix("key_21".as_bytes()));

    destroy_range(&mut client, store_id, "key_2".as_bytes());
    let snap = cluster.get_snap(node_id, "key_".as_bytes());
    assert!(!snap.has_data_in_prefix("key_20".as_bytes()));
    assert!(!snap.has_data_in_prefix("key_21".as_bytes()));
    assert!(!snap.has_data_in_prefix("key_22".as_bytes()));
    assert!(!snap.has_data_in_prefix("key_23".as_bytes()));

    assert!(snap.has_data_in_prefix("key_3".as_bytes()));
    cluster.stop();
}

#[test]
fn test_delete_range_recover() {
    test_util::init_log_for_test();
    let node_ids = &[alloc_node_id(), alloc_node_id(), alloc_node_id()];
    let mut cluster = ServerCluster::new(node_ids.to_vec(), |_, _| {});
    let region = cluster.get_pd_client().get_region_info(&[]).unwrap();
    let leader_store_id = region.leader.unwrap().store_id;
    let &stop_node_id = node_ids
        .iter()
        .find(|&&node_id| cluster.get_store_id(node_id) != leader_store_id)
        .unwrap();
    cluster.set_dfs_delay(Duration::from_millis(10));
    for _ in 0..3 {
        let mut client = cluster.new_client();
        for i in 0..10 {
            client.put_kv(i * 100..i * 100 + 10, i_to_key, i_to_val);
        }
        destroy_range(&mut client, leader_store_id, "key_".as_bytes());
        client.put_kv(0..10, i_to_key, i_to_val);
        cluster.stop_node(stop_node_id);
        cluster.start_node(stop_node_id, |_, _| {});
        thread::sleep(Duration::from_secs(1));
    }
    cluster.stop();
}

fn new_destroy_range_req(prefix: &[u8]) -> UnsafeDestroyRangeRequest {
    let mut req = UnsafeDestroyRangeRequest::default();
    let mut end_key = prefix.to_vec();
    convert_to_prefix_next(&mut end_key);
    req.set_start_key(prefix.to_vec());
    req.set_end_key(end_key);
    req
}

fn destroy_range(client: &mut ClusterClient, store_id: u64, prefix: &[u8]) {
    let req = new_destroy_range_req(prefix);
    let kv_client = client.get_kv_client(store_id);
    let resp = kv_client.unsafe_destroy_range(&req).unwrap();
    assert!(resp.get_error().is_empty(), "{:?}", resp.get_error());
    assert!(!resp.has_region_error());
}

fn i_to_key(i: usize) -> Vec<u8> {
    format!("key_{:03}", i).into_bytes()
}

fn i_to_val(i: usize) -> Vec<u8> {
    format!("val_{:03}", i).into_bytes()
}
