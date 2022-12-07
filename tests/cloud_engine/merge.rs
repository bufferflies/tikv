// Copyright 2022 TiKV Project Authors. Licensed under Apache-2.0.

use test_cloud_server::ServerCluster;

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

fn i_to_key(i: usize) -> Vec<u8> {
    format!("key_{:03}", i).into_bytes()
}

fn i_to_val(i: usize) -> Vec<u8> {
    format!("val_{:03}", i).into_bytes().repeat(3)
}
