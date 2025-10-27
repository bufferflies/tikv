// Copyright 2025 TiKV Project Authors. Licensed under Apache-2.0.

use std::time::{Duration, Instant};

use api_version::ApiV2;
use futures::executor::block_on;
use more_asserts::assert_le;
use pd_client::PdClient;
use test_cloud_server::{
    alloc_node_id_vec, keyspace::CreateKeyspaceOptions, util::request_major_compaction,
    ServerCluster, ServerClusterBuilder,
};
use txn_types::{Key, NULL_KEYSPACE_ID};

#[test]
fn test_gc_safe_point_passing_to_compaction() {
    test_util::init_log_for_test();
    let node_ids = alloc_node_id_vec(3);
    fail::cfg(
        "rfstore_config_from_old_force_short_update_gc_safe_point_interval",
        "return",
    )
    .unwrap();
    let cluster = ServerClusterBuilder::new(node_ids.clone(), |_, conf| {
        conf.raft_store.enable_inner_key_offset = true;
    })
    .build();
    cluster.wait_region_replicated(&[], 3);
    let pd_client = cluster.get_pd_client();
    pd_client.disable_default_operator();

    fail::remove("rfstore_config_from_old_force_short_update_gc_safe_point_interval");

    let keyspace_ids = &[
        block_on(cluster.create_keyspace(&Default::default(), Duration::from_secs(10))),
        block_on(cluster.create_keyspace(
            &CreateKeyspaceOptions {
                table_count: 1,
                ..Default::default()
            },
            Duration::from_secs(10),
        )),
        block_on(cluster.create_keyspace(
            &CreateKeyspaceOptions {
                table_count: 2,
                ..Default::default()
            },
            Duration::from_secs(10),
        )),
    ];

    // Simulate some modification to GC states.
    // Note that GC safe point can't be advanced before advancing txn safe point, as
    // there's a constraint that GC safe point never exceeds txn safe point.
    // Pass different GC safe point and txn safe point so that we can determine if
    // they are correctly set.
    for (keyspace_id, txn_safe_point, gc_safe_point) in [
        (NULL_KEYSPACE_ID, 10002u64, 10001u64),
        (keyspace_ids[0], 10004, 10003),
        (keyspace_ids[1], 10006, 10005),
        (keyspace_ids[2], 10008, 10007),
    ] {
        block_on(pd_client.advance_txn_safe_point(keyspace_id, txn_safe_point.into())).unwrap();
        block_on(pd_client.advance_gc_safe_point(keyspace_id, gc_safe_point.into())).unwrap();
    }

    // Check the current GC states
    let cluster_gc_states = block_on(pd_client.get_all_keyspaces_gc_states()).unwrap();
    assert_eq!(cluster_gc_states.keyspace_gc_states.len(), 4);

    for (keyspace_id, txn_safe_point, gc_safe_point) in [
        (NULL_KEYSPACE_ID, 10002u64, 10001u64),
        (keyspace_ids[0], 10004, 10003),
        (keyspace_ids[1], 10006, 10005),
        (keyspace_ids[2], 10008, 10007),
    ] {
        let gc_state = cluster_gc_states
            .keyspace_gc_states
            .get(&keyspace_id)
            .unwrap();
        assert_eq!(gc_state.txn_safe_point, txn_safe_point.into());
        assert_eq!(gc_state.gc_safe_point, gc_safe_point.into());
    }

    // Wait for store tick to update the GC safe poitns.
    std::thread::sleep(
        cluster
            .get_node_config(node_ids[0])
            .raft_store
            .raft_base_tick_interval
            .0
            * 2,
    );

    fail::cfg("trigger_major_compaction_no_skip_on_empty", "return").unwrap();

    let runtime = tokio::runtime::Runtime::new().unwrap();
    request_major_compaction(&runtime, &pd_client, NULL_KEYSPACE_ID);
    for &keyspace_id in keyspace_ids {
        request_major_compaction(&runtime, &pd_client, keyspace_id);
    }

    // TODO: The `major-compact` HTTP API, which is used in
    //   `request_major_compaction`, doesn't handle the null keyspace properly.
    //   Therefore the check on the null keyspace cannot pass as expected. we
    //   may enable this check when the API is fixed, or there's found to be some
    //   other ways to trigger the compaction.
    // let null_ks_region = pd_client.get_region(b"t").unwrap().id;
    // wait_max_used_gc_safe_point(
    //     &cluster,
    //     &node_ids,
    //     null_ks_region,
    //     10001,
    //     Duration::from_secs(5),
    // );
    let ks1_region = pd_client
        .get_region(Key::from_raw(&ApiV2::get_txn_keyspace_prefix(keyspace_ids[0])).as_encoded())
        .unwrap()
        .id;
    wait_max_used_gc_safe_point(
        &cluster,
        &node_ids,
        ks1_region,
        10003,
        Duration::from_secs(5),
    );
    let ks2_region = pd_client
        .get_region(Key::from_raw(&ApiV2::get_txn_keyspace_prefix(keyspace_ids[1])).as_encoded())
        .unwrap()
        .id;
    wait_max_used_gc_safe_point(
        &cluster,
        &node_ids,
        ks2_region,
        10005,
        Duration::from_secs(5),
    );
    let ks3_region = pd_client
        .get_region(Key::from_raw(&ApiV2::get_txn_keyspace_prefix(keyspace_ids[2])).as_encoded())
        .unwrap()
        .id;
    wait_max_used_gc_safe_point(
        &cluster,
        &node_ids,
        ks3_region,
        10007,
        Duration::from_secs(5),
    );

    fail::remove("trigger_major_compaction_no_skip_on_empty");
}

fn wait_max_used_gc_safe_point(
    cluster: &ServerCluster,
    node_ids: &[u16],
    region_id: u64,
    expected_gc_safe_point: u64,
    timeout: Duration,
) {
    let start = Instant::now();
    while start.elapsed() < timeout {
        let satisfied = node_ids.iter().any(|&node_id| {
            let max_used_gc_safe_point = cluster
                .get_kvengine(node_id)
                .get_max_used_gc_safe_point(region_id);
            assert_le!(max_used_gc_safe_point, expected_gc_safe_point.into());
            max_used_gc_safe_point == expected_gc_safe_point.into()
        });
        if satisfied {
            return;
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    panic!(
        "wait_max_used_gc_safe_point timeout, node_ids: {:?}, region_id: {}",
        node_ids, region_id
    );
}
