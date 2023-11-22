// Copyright 2023 TiKV Project Authors. Licensed under Apache-2.0.

//! Test for test_cloud_server & test_pd_client themselves.

use std::{sync::atomic::AtomicU16, time::Duration};

use futures::executor::block_on;
use pd_client::PdClient;
use tikv_client::TimestampExt;
use tikv_util::{codec::bytes::encode_bytes, info};
use tokio::runtime::Runtime;

use crate::{client::CommitAction, ServerCluster};

#[test]
fn it_works() {
    test_util::init_log_for_test();
    let node_ids = alloc_node_id_vec(3);
    let mut cluster = ServerCluster::new(node_ids.clone(), |_, _| {});
    let stores = cluster.get_stores();
    assert_eq!(stores.len(), 3);
    let mut client = cluster.new_client();
    client.put_kv(0..100, i_to_key, i_to_val);
    client.put_kv(100..200, i_to_key, i_to_val);
    client.put_kv(200..300, i_to_key, i_to_val);
    let split_keys = vec![i_to_key(50), i_to_key(150), i_to_key(200)];
    for split_key in &split_keys {
        client.split(split_key);
    }
    cluster.wait_pd_region_count(4);
    cluster.get_pd_client().disable_default_operator();
    cluster.remove_node_peers(node_ids[0]);
    cluster.stop_node(node_ids[0]);
    std::thread::sleep(Duration::from_millis(100));
    cluster.start_node(node_ids[0], |_, _| {});
    cluster.get_pd_client().enable_default_operator();
    info!("enable replica operator");
    cluster.wait_region_replicated(&[], 3);
    for split_key in &split_keys {
        cluster.wait_region_replicated(split_key, 3);
    }
    client.verify_data_with_ref_store();

    client
        .try_put_kv(
            50..250,
            i_to_key,
            prefixed_i_to_val("async".to_string()),
            CommitAction::AsyncCommitSecondaryKeys(Duration::from_millis(100)),
        )
        .unwrap();
    client.verify_data_with_ref_store();

    client
        .try_put_kv(
            150..300,
            i_to_key,
            prefixed_i_to_val("no".to_string()),
            CommitAction::NoCommit,
        )
        .unwrap();
    client.verify_data_with_ref_store();

    client
        .try_del_kv(
            100..200,
            i_to_key,
            CommitAction::AsyncCommitSecondaryKeys(Duration::MAX),
        )
        .unwrap();
    client.verify_data_with_ref_store();

    cluster.stop();
}

#[test]
fn test_split_regions() {
    test_util::init_log_for_test();
    let mut cluster = ServerCluster::new(alloc_node_id_vec(3), |_, _| {});
    cluster.wait_region_replicated(&[], 3);
    let pd_client = cluster.get_pd_client();

    let keys0: Vec<Vec<u8>> = [10, 20, 30]
        .into_iter()
        .map(|i| encode_bytes(&i_to_key(i)))
        .collect();
    {
        let new_regions = block_on(pd_client.split_regions(keys0.clone())).unwrap();
        cluster.wait_pd_region_count(keys0.len() + 1);
        let mut region_keys = new_regions
            .into_iter()
            .map(|region_id| {
                block_on(pd_client.get_region_by_id(region_id))
                    .unwrap()
                    .unwrap()
                    .start_key
            })
            .collect::<Vec<_>>();
        region_keys.sort();
        assert_eq!(region_keys, keys0);

        let mut all_regions = block_on(pd_client.scan_regions(vec![], vec![], 100))
            .unwrap()
            .into_iter()
            .map(|mut r| r.take_region().take_start_key())
            .collect::<Vec<_>>();
        all_regions.sort();
        assert_eq!(all_regions[1..], keys0[..]); // Skip first region with empty start key.
    }

    let mut keys1: Vec<Vec<u8>> = [5, 10, 15, 25, 30, 40, 50]
        .into_iter()
        .map(|i| encode_bytes(&i_to_key(i)))
        .collect();
    {
        let new_regions = block_on(pd_client.split_regions(keys1.clone())).unwrap();
        cluster.wait_pd_region_count(keys1.len() + 1 + 1); // The `1` is 20 of keys0.
        let mut region_keys = new_regions
            .into_iter()
            .map(|region_id| {
                block_on(pd_client.get_region_by_id(region_id))
                    .unwrap()
                    .unwrap()
                    .start_key
            })
            .collect::<Vec<_>>();
        region_keys.sort();
        keys1.drain_filter(|k| keys0.contains(k));
        assert_eq!(region_keys, keys1);

        let mut all_regions = block_on(pd_client.scan_regions(vec![], vec![], 100))
            .unwrap()
            .into_iter()
            .map(|mut r| r.take_region().take_start_key())
            .collect::<Vec<_>>();
        let expected = {
            let mut expected = keys0.clone();
            expected.append(&mut keys1);
            expected.sort();
            expected.dedup();
            expected
        };
        all_regions.sort();
        assert_eq!(all_regions[1..], expected); // Skip first region with empty start key.
    }

    cluster.stop();
}

// Test for update region cache after split.
// https://github.com/tidbcloud/cloud-storage-engine/pull/933.
#[test]
fn test_client_split_region() {
    test_util::init_log_for_test();
    let mut cluster = ServerCluster::new(alloc_node_id_vec(3), |_, _| {});
    cluster.wait_region_replicated(&[], 3);
    let mut client = cluster.new_client();

    let split_key = b"xkey";
    client.split(split_key);
    cluster.wait_pd_region_count(2);

    assert_ne!(client.get_region_id(b""), client.get_region_id(split_key));
    {
        let region = client.get_region_by_key(b"");
        assert_eq!(region.raw_start(), b"");
        assert_eq!(region.raw_end(), split_key);
    }
    {
        let region = client.get_region_by_key(split_key);
        assert_eq!(region.raw_start(), split_key);
        assert_eq!(region.raw_end(), kvengine::GLOBAL_SHARD_END_KEY);
    }

    cluster.stop();
}

#[test]
fn test_txn_client() {
    test_util::init_log_for_test();
    let node_ids = alloc_node_id_vec(3);
    let mut cluster = ServerCluster::new(node_ids, |_, _| {});

    Runtime::new().unwrap().block_on(async {
        cluster.start_pd_server(1);
        let mut txn_client = cluster.new_txn_client().await;

        // Check TSO.
        {
            let pd_client = cluster.get_pd_client();
            let mut last_tso = 0;
            for i in 0..=10 {
                let tso = if i % 2 == 0 {
                    pd_client.get_tso().await.unwrap().into_inner()
                } else {
                    txn_client.current_timestamp().await.unwrap().version()
                };
                assert!(tso > last_tso);
                last_tso = tso;
            }
        }

        // Prepare data.
        let mut client = cluster.new_client();
        {
            client.put_kv(0..100, i_to_key, i_to_val);
            client.put_kv(100..200, i_to_key, i_to_val);
            client.put_kv(200..300, i_to_key, i_to_val);
            let split_keys = vec![i_to_key(50), i_to_key(150), i_to_key(200)];
            for split_key in &split_keys {
                client.split(split_key);
            }
            cluster.wait_pd_region_count(4);
            client.verify_data_with_ref_store();
        }
        let ref_store = client.dump_ref_store();

        // Verify by scan.
        assert_eq!(
            txn_client
                .verify_data_by_scan(&ref_store, None)
                .await
                .unwrap(),
            300
        );
        assert_eq!(
            txn_client
                .verify_data_by_scan(&ref_store, Some((&i_to_key(100), &i_to_key(250))))
                .await
                .unwrap(),
            150
        );
    });

    cluster.stop();
}

static NODE_ALLOCATOR: AtomicU16 = AtomicU16::new(1);

pub(crate) fn alloc_node_id() -> u16 {
    let node_id = NODE_ALLOCATOR.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    info!("allocated node_id {}", node_id);
    node_id
}

pub(crate) fn alloc_node_id_vec(count: usize) -> Vec<u16> {
    let mut nodes = vec![];
    nodes.resize_with(count, || alloc_node_id());
    nodes
}

fn i_to_key(i: usize) -> Vec<u8> {
    format!("xkey{:08}", i).into_bytes()
}

fn i_to_val(i: usize) -> Vec<u8> {
    // `repeat` must > 0. CSE treat empty value as not found.
    format!("val{:04}", i).repeat(i % 32 + 1).into_bytes()
}

fn prefixed_i_to_val(prefix: String) -> impl Fn(usize) -> Vec<u8> {
    move |i| {
        format!("{}:{:04}", prefix, i)
            .repeat(i % 32 + 1)
            .into_bytes()
    }
}
