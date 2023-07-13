// Copyright 2023 TiKV Project Authors. Licensed under Apache-2.0.

use std::{str::FromStr, time::Duration};

use api_version::ApiV2;
use http::Uri;
use hyper::{Body, Request};
use kvengine::{CF_LEVELS, WRITE_CF};
use kvproto::{
    metapb::Store,
    pdpb::CheckPolicy,
    raft_cmdpb::{RaftCmdRequest, RaftRequestHeader},
};
use pd_client::PdClient;
use rand::Rng;
use rfstore::store::CustomBuilder;
use test_cloud_server::ServerCluster;
use tikv_util::time::Instant;

use crate::alloc_node_id;

async fn request_major_compact_on_store(store: &Store, query: &str) {
    let uri = Uri::from_str(&format!(
        "http://{}/major-compact?{}",
        &store.status_address, query
    ))
    .unwrap();
    let req = Request::post(uri).body(Body::empty()).unwrap();
    let client = hyper::Client::new();
    let resp: http::Response<Body> = client.request(req).await.unwrap();
    assert!(
        resp.status().is_success(),
        "{:?}",
        hyper::body::to_bytes(resp.into_body()).await.unwrap()
    );
    hyper::body::to_bytes(resp.into_body()).await.unwrap();
}

#[test]
fn test_major_compaction() {
    test_util::init_log_for_test();
    let mut nodes = vec![];
    let node_cnt = 3;
    for _ in 0..node_cnt {
        nodes.push(alloc_node_id());
    }
    let mut cluster = ServerCluster::new(nodes.clone(), |_, conf| {
        conf.kvengine.compaction_request_version = 3;
    });
    cluster.wait_region_replicated(&[], 3);
    let mut client = cluster.new_client();
    let pd_client = cluster.get_pd_client();
    pd_client.disable_default_operator();

    // Prepare 2 keyspaces, each has 2 regions, each region has 500 keys.
    let region0 = pd_client.get_all_regions().first().unwrap().clone();
    let keys = vec![
        ApiV2::get_txn_keyspace_prefix(1),
        i_to_key_with_keyspace(1)(500),
        ApiV2::get_txn_keyspace_prefix(2),
        i_to_key_with_keyspace(2)(500),
        ApiV2::get_txn_keyspace_prefix(3),
    ];
    let split_keys = keys
        .into_iter()
        .map(|k| txn_types::Key::from_raw(&k).into_encoded())
        .collect::<Vec<_>>();
    pd_client.must_split_region(region0, CheckPolicy::Usekey, split_keys.clone());
    cluster.wait_pd_region_min_count(split_keys.len() + 1);
    client.put_kv(0..1000, i_to_key_with_keyspace(1), i_to_random_value);
    client.put_kv(0..1000, i_to_key_with_keyspace(2), i_to_random_value);
    let ks1_r1 = pd_client.get_region(&split_keys[0]).unwrap();
    let ks1_r2 = pd_client.get_region(&split_keys[1]).unwrap();
    let ks2_r1 = pd_client.get_region(&split_keys[2]).unwrap();
    let ks2_r2 = pd_client.get_region(&split_keys[3]).unwrap();

    let wait_for_memtable_flush =
        |cluster: &ServerCluster, node_ids: &[u16], region_id: u64, timeout: Duration| {
            let mut curr_shard_stats;
            let mut flushed;
            let start = Instant::now_coarse();
            loop {
                curr_shard_stats = node_ids
                    .iter()
                    .map(|id| cluster.get_kvengine(*id).get_shard_stat(region_id))
                    .collect::<Vec<_>>();
                flushed = curr_shard_stats
                    .iter()
                    .all(|curr| curr.mem_table_size == 0 && curr.mem_table_count == 1);
                if flushed {
                    break;
                }
                if start.saturating_elapsed() >= timeout {
                    break;
                }
                std::thread::sleep(Duration::from_millis(1000));
            }
            flushed
        };

    // Trigger switching and flushing memtable.
    let flush_memtable = |cluster: &ServerCluster, node_ids: &[u16], region_id: u64| {
        let mut client = cluster.new_client();
        let ctx = client.new_rpc_ctx(region_id).unwrap();
        let mut req = RaftCmdRequest::default();
        let mut header = RaftRequestHeader::default();
        header.set_region_id(ctx.get_region_id());
        header.set_peer(ctx.get_peer().clone());
        header.set_region_epoch(ctx.get_region_epoch().clone());
        header.set_term(6);
        req.set_header(header);
        let mut custom_builder = CustomBuilder::new();
        custom_builder.set_switch_mem_table(1);
        req.set_custom_request(custom_builder.build());
        cluster.send_raft_command(req);

        let flushed = wait_for_memtable_flush(cluster, node_ids, region_id, Duration::from_secs(6));
        assert!(flushed);
    };

    flush_memtable(&cluster, &nodes, ks1_r1.get_id());
    flush_memtable(&cluster, &nodes, ks1_r2.get_id());
    flush_memtable(&cluster, &nodes, ks2_r1.get_id());
    flush_memtable(&cluster, &nodes, ks2_r2.get_id());

    let stores: Vec<Store> = pd_client
        .get_all_stores(true)
        .unwrap()
        .into_iter()
        .filter(|s| {
            !s.get_labels().iter().any(|l| {
                l.key.to_lowercase() == "engine" && l.value.to_lowercase().starts_with("tiflash")
            })
        })
        .collect();
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(1)
        .enable_all()
        .build()
        .unwrap();

    // Trigger major compaction on one region.
    for store in &stores {
        let query = format!(
            "major_compact=true&keyspace_id=1&region_id={}",
            ks1_r1.get_id()
        );
        runtime.block_on(request_major_compact_on_store(store, &query));
    }

    let wait_for_major_compaction =
        |cluster: &ServerCluster, node_ids: &[u16], region_id: u64, timeout: Duration| {
            let mut curr_shard_stats;
            let mut major_compacted;
            let start = Instant::now_coarse();
            loop {
                curr_shard_stats = node_ids
                    .iter()
                    .map(|id| cluster.get_kvengine(*id).get_shard_stat(region_id))
                    .collect::<Vec<_>>();
                major_compacted = curr_shard_stats.iter().all(|curr| {
                    let bottom_most_level = curr.cfs[WRITE_CF].levels.last().unwrap();
                    curr.mem_table_size == 0
                        && curr.mem_table_count == 1
                        && curr.l0_table_count == 0
                        && bottom_most_level.level == CF_LEVELS[WRITE_CF]
                        && bottom_most_level.num_tables != 0
                });
                if major_compacted {
                    break;
                }
                if start.saturating_elapsed() >= timeout {
                    break;
                }
                std::thread::sleep(Duration::from_millis(200));
            }
            major_compacted
        };
    let major_compacted = |cluster: &ServerCluster, node_ids: &[u16], region_id: u64| {
        let curr_shard_stats = node_ids
            .iter()
            .map(|id| cluster.get_kvengine(*id).get_shard_stat(region_id))
            .collect::<Vec<_>>();
        curr_shard_stats.iter().any(|curr| {
            let bottom_most_level = curr.cfs[WRITE_CF].levels.last().unwrap();
            curr.mem_table_size == 0
                && curr.mem_table_count == 1
                && curr.l0_table_count == 0
                && bottom_most_level.level == CF_LEVELS[WRITE_CF]
                && bottom_most_level.num_tables != 0
        })
    };
    assert!(wait_for_major_compaction(
        &cluster,
        &nodes,
        ks1_r1.get_id(),
        Duration::from_secs(6)
    ));
    assert!(!major_compacted(&cluster, &nodes, ks1_r2.get_id()));
    assert!(!major_compacted(&cluster, &nodes, ks2_r1.get_id()));
    assert!(!major_compacted(&cluster, &nodes, ks2_r2.get_id()));

    // Trigger major compaction on one keyspace.
    for store in &stores {
        let query = "major_compact=true&keyspace_id=2";
        runtime.block_on(request_major_compact_on_store(store, query));
    }
    assert!(wait_for_major_compaction(
        &cluster,
        &nodes,
        ks2_r1.get_id(),
        Duration::from_secs(6)
    ));
    assert!(wait_for_major_compaction(
        &cluster,
        &nodes,
        ks2_r2.get_id(),
        Duration::from_secs(6)
    ));
    assert!(!major_compacted(&cluster, &nodes, ks1_r2.get_id()));

    cluster.stop();
}

fn i_to_key_with_keyspace(keyspace_id: u32) -> impl Fn(usize) -> Vec<u8> {
    move |i: usize| -> Vec<u8> {
        let mut key = ApiV2::get_txn_keyspace_prefix(keyspace_id);
        key.extend(format!("xkey{:08}", i).into_bytes());
        key
    }
}

fn i_to_random_value(_i: usize) -> Vec<u8> {
    let random_number: usize = rand::thread_rng().gen();
    format!("value_{:20}", random_number).into_bytes()
}
