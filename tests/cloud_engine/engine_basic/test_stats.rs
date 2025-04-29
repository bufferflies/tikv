// Copyright 2022 TiKV Project Authors. Licensed under Apache-2.0.

use std::cmp;

use kvengine::WRITE_CF;
use rand::Rng;
use test_cloud_server::ServerCluster;
use tikv_util::config::ReadableSize;

use super::*;
use crate::alloc_node_id;

// Returns (estimated_kv_size, max_ts).
// Note: Results would change over time due to compaction.
fn get_stats_by_shard_interface(engine: &kvengine::Engine) -> (u64, u64) {
    engine
        .get_all_shard_id_vers()
        .into_iter()
        .map(|id_ver| engine.get_shard(id_ver.id))
        .fold((0u64, 0u64), |acc, x| {
            let shard = x.unwrap();
            (
                acc.0 + shard.get_estimated_kv_size(),
                cmp::max(acc.1, shard.get_max_ts()),
            )
        })
}

#[test]
fn test_shard_stats() {
    const TEST_COUNT: i32 = 100;
    const RANDOM_VALUE_SIZE: usize = 1024;
    let key_size: usize = i_to_key(0).len();

    test_util::init_log_for_test();
    let node_id = alloc_node_id();
    let mut cluster = ServerCluster::new(vec![node_id], |_, conf| {
        // Set small memtable size to make data reach SSTables.
        conf.rocksdb.writecf.write_buffer_size = ReadableSize::kb(1);
    });
    let mut client = cluster.new_client();
    let engine = cluster.get_kvengine(node_id);
    let mut rng = rand::thread_rng();

    let random_val = |_: usize| -> Vec<u8> {
        let mut bytes = [0u8; RANDOM_VALUE_SIZE];
        rand::thread_rng().fill(&mut bytes);
        bytes.to_vec()
    };

    let mut expect_kv_size = 0u64;
    for i in 0..=TEST_COUNT {
        // Do NOT put kv on i==0 to test for empty engine.
        if i > 0 {
            let idx = rng.gen_range(0..10);
            let range = idx * 100..(idx + 1) * 100;
            if !client.ref_store_contains_key(&i_to_key(range.start)) {
                expect_kv_size += 100 * (key_size + RANDOM_VALUE_SIZE) as u64;
            }
            client.put_kv(range, i_to_key, random_val);
        }

        // Split region to test for multiple shards.
        if i == 50 {
            client.split(&i_to_key(700));
            client.split(&i_to_key(900));
            // Wait for split to complete.
            assert!(
                try_wait(|| engine.get_all_shard_id_vers().len() == 3, 10),
                "not split as expected"
            );
        }

        let all_shard_stats = engine.get_all_shard_stats();
        let (kv_size, mem_table_size, kv_size_lower_l0, max_ts) =
            all_shard_stats
                .iter()
                .fold((0u64, 0u64, 0u64, 0u64), |acc, x| {
                    let kv_size_lower_l0: u64 = x.cfs[WRITE_CF]
                        .levels
                        .iter()
                        .map(|lv_stats| lv_stats.kv_size)
                        .sum();
                    (
                        acc.0 + x.kv_size,
                        acc.1 + x.mem_table_size,
                        acc.2 + kv_size_lower_l0,
                        cmp::max(acc.3, x.max_ts),
                    )
                });

        // Multiple latest versions of a key in different SSTs are not deduplicated.
        // `mem_table_size` dose NOT calculate into `kv_size`.
        assert!(
            kv_size + mem_table_size >= expect_kv_size,
            "kv_size wrong, kv_size:{}, expect_kv_size:{}, mem_table_size:{}, shards:{:?}",
            kv_size,
            expect_kv_size,
            mem_table_size,
            all_shard_stats,
        );
        assert_eq!(max_ts, client.max_ts());

        if i == TEST_COUNT {
            // Verify we cover levels lower than l0.
            // The logics for l0 are different with other levels.
            assert!(
                kv_size_lower_l0 > 0,
                ">=l1 not covered, shards:{:?}",
                all_shard_stats
            );
        }

        // Verify `Engine::get_engine_stats` interface.
        let engine_stats = engine.get_engine_stats(all_shard_stats);
        assert_eq!(
            engine_stats.kv_size, kv_size,
            "engine.kv_size wrong, engine:{:?}",
            engine_stats
        );
        assert_eq!(engine_stats.max_ts, client.max_ts());

        // Verify `Shard.get_estimated_kv_size()` & `Shard.get_max_ts` interface.
        let expected_max_ts = client.max_ts();
        let ok = try_wait(
            || {
                let kv_size: u64 = engine
                    .get_all_shard_stats()
                    .into_iter()
                    .map(|x| x.kv_size)
                    .sum();
                let (estimated_kv_size, max_ts) = get_stats_by_shard_interface(&engine);
                estimated_kv_size == kv_size && max_ts == expected_max_ts
            },
            10,
        );
        assert!(
            ok,
            "shard stats interface wrong, shard_interface:{:?}, shards:{:?}",
            get_stats_by_shard_interface(&engine),
            engine.get_all_shard_stats()
        );
    }

    cluster.stop();
}
