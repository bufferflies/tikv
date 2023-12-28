// Copyright 2022 TiKV Project Authors. Licensed under Apache-2.0.

use futures::executor::block_on;
use slog_global::info;
use test_cloud_server::ServerCluster;
use tikv_util::config::ReadableSize;

use crate::cases::{alloc_node_id_vec, i_to_key, random_value_1kb};

#[test]
fn test_trim_over_bound() {
    let cases = vec![
        (100, false), // no over bound.
        (150, true),
        (175, false),
        (200, true), // no over bound.
    ];

    for (split_key_idx, do_leader_transfer) in cases {
        test_trim_over_bound_impl(split_key_idx, do_leader_transfer);
    }
}

fn test_trim_over_bound_impl(split_key_idx: usize, do_leader_transfer: bool) {
    let fp = "before_engine_trigger_compact";
    test_util::init_log_for_test();
    let node_ids = alloc_node_id_vec(3);
    let mut cluster = ServerCluster::new(node_ids.clone(), |_, conf| {
        // Set small memtable size to make data reach SSTables.
        conf.rocksdb.writecf.write_buffer_size = ReadableSize::kb(1);
    });
    cluster.wait_region_replicated(&[], 3);
    let mut client = cluster.new_client();
    let prev_keyspace = i_to_key(0);
    client.split(&prev_keyspace);

    client.put_kv(100..200, i_to_key, random_value_1kb);

    // Disable compaction to make shards over bound after split.
    fail::cfg(fp, "return").unwrap();
    let split_key = i_to_key(split_key_idx);
    client.split(&split_key);
    cluster.wait_pd_region_count(3);

    // Verify shard bound.
    {
        let stats = cluster.get_kvengine(node_ids[0]).get_all_shard_stats();
        info!("engine0 stats: {:?}", stats);

        assert_eq!(stats.len(), 3);
        if split_key_idx > 100 && split_key_idx < 200 {
            for i in 0..3 {
                if stats[i].start.is_empty() {
                    // skip the first shard.
                    continue;
                }
                assert!(stats[i].has_over_bound_data);
            }
        }
    }

    // Transfer leader of region0 to another store.
    if do_leader_transfer {
        let region_id0 = client.get_region_id(&i_to_key(0));
        let (region0, leader0) = block_on(client.pd_client.get_region_leader_by_id(region_id0))
            .unwrap()
            .unwrap();

        let region_id300 = client.get_region_id(&i_to_key(300));
        let (_, leader300) = block_on(client.pd_client.get_region_leader_by_id(region_id300))
            .unwrap()
            .unwrap();

        if leader0.store_id == leader300.store_id {
            let target = region0
                .get_peers()
                .iter()
                .find(|x| x.store_id != leader300.store_id)
                .unwrap();
            client
                .pd_client
                .transfer_leader(region0.id, target.clone(), vec![]);
            client
                .pd_client
                .region_leader_must_be(region0.id, target.clone());
        }
    }

    // Enable compaction for trim_over_bound.
    fail::cfg(fp, "off").unwrap();
    client.merge(&i_to_key(0), &i_to_key(300));
    cluster.wait_pd_region_count(2);
    client.verify_data_with_ref_store();

    fail::remove(fp);
    cluster.stop();
}
