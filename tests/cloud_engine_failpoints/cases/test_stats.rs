// Copyright 2023 TiKV Project Authors. Licensed under Apache-2.0.

use test_cloud_server::{try_wait, ServerCluster};
use tikv_util::{config::ReadableSize, info};

use crate::cases::{alloc_node_id_vec, i_to_key, random_value_1kb};

// Only 1 node to always get stats from leader for test stability.
const NODES_COUNT: usize = 1;

// TODO: improve coverage for different number of tables & blocks.
#[test]
fn test_estimated_size() {
    test_util::init_log_for_test();
    let node_ids = alloc_node_id_vec(NODES_COUNT);
    let mut cluster = ServerCluster::new(node_ids.clone(), |_, conf| {
        // Small memtable size to flush to sst easily.
        conf.rocksdb.writecf.write_buffer_size = ReadableSize::kb(1);
        // Appropriate block size to make estimated size based on blocks more accurate.
        conf.rocksdb.writecf.block_size = ReadableSize::kb(1);
        // Large file size to make less tables to verify estimated size based on blocks.
        conf.rocksdb.writecf.target_file_size_base = ReadableSize::mb(1);
        // Large `region_split_size` to avoid split by TiKV itself.
        conf.coprocessor.region_split_size = ReadableSize::kb(1024);
        // Any value less than `region_split_size`.
        conf.coprocessor.region_bucket_size = ReadableSize::kb(16);
    });
    cluster.wait_region_replicated(&[], NODES_COUNT);
    let mut client = cluster.new_client();

    // Put [0, 300) x 1kb data.
    (0..300).step_by(100).for_each(|i| {
        client.put_kv(i..i + 100, i_to_key, random_value_1kb);
    });

    // The estimated size is implemented for L1+ only by now.
    // So always trigger L0 compaction.
    // TODO: also test for L0.
    let fp_compact_l0 = "refresh_compaction_priority_for_l0";
    fail::cfg(fp_compact_l0, "return").unwrap();
    let ok = try_wait(
        // Wait for mem-table flush & L0 compaction finished.
        || {
            let stats = cluster.get_kvengine(node_ids[0]).get_all_shard_stats();
            stats[0].mem_table_size == 0 && stats[0].l0_table_count == 0
        },
        10,
    );
    let stats = cluster.get_kvengine(node_ids[0]).get_all_shard_stats();
    info!("engine0 stats (before split): {:?}", stats[0]);
    assert!(ok, "{:?}", stats[0]);

    // We verify the size of WRITE_CF only.
    // Size of LOCK_CF is not stable as it depends on the timing of compaction.
    let get_shard_write_cf_size = |shard_id: u64| -> u64 {
        let kvengine = cluster.get_kvengine(node_ids[0]);
        let shard = kvengine.get_shard(shard_id).unwrap();
        let stats = shard.get_stats();
        assert_eq!(shard.get_estimated_size(), stats.total_size, "{:?}", stats);

        stats.cfs[0].levels.iter().map(|l| l.data_size).sum::<u64>()
    };

    let expect_total_size = get_shard_write_cf_size(client.get_region_id(&i_to_key(100)));

    // Disable compaction then split, to make over-bound or overlap sstables.
    let fp_disable_compact = "before_engine_trigger_compact";
    fail::cfg(fp_disable_compact, "return").unwrap();
    let region_indexes = [0, 300 / 4, 300 / 2, 300];
    for idx in region_indexes.iter().skip(1) {
        client.split(&i_to_key(*idx));
    }
    cluster.wait_pd_region_count(region_indexes.len());

    let region_right = client.get_region_id(&i_to_key(*region_indexes.last().unwrap()));
    // Verify shard bound.
    {
        let stats = cluster.get_kvengine(node_ids[0]).get_all_shard_stats();
        info!("engine0 stats: {:?}", stats);
        assert_eq!(stats.len(), region_indexes.len());
        for stat in stats {
            // The rightmost shard is not over bound.
            assert_eq!(
                stat.has_over_bound_data,
                stat.id != region_right,
                "{:?}",
                stat
            );
        }
    }

    assert_eq!(get_shard_write_cf_size(region_right), 0);
    let sizes = region_indexes
        .iter()
        .map(|idx| {
            let region_id = client.get_region_id(&i_to_key(*idx));
            get_shard_write_cf_size(region_id)
        })
        .collect::<Vec<_>>();
    let total_size: u64 = sizes.clone().into_iter().sum();
    assert!(
        (total_size as f64 - expect_total_size as f64).abs() < 0.05 * expect_total_size as f64,
        "total_size {}, expect_total_size {}, sizes {:?}",
        total_size,
        expect_total_size,
        sizes,
    );

    client.verify_data_with_ref_store();
    fail::remove(fp_compact_l0);
    fail::remove(fp_disable_compact);
    cluster.stop();
}
