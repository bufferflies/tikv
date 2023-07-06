// Copyright 2023 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    sync::{atomic::Ordering, Arc, RwLock},
    time::Duration,
};

use api_version::ApiV2;
use kvengine::dfs::DFSConfig;
use kvproto::pdpb::CheckPolicy;
use native_br::{backup, backup_worker};
use rand::Rng;
use test_cloud_server::{try_wait_result, ServerCluster};
use tikv_util::{
    config::{ReadableDuration, ReadableSize},
    info,
    time::Instant,
};
use txn_types::Key;

use crate::{
    alloc_node_id_vec, generate_keyspace_key, prepare_dfs, random_node_restart,
    spawn_create_keyspace, spawn_keyspace_write, spawn_merge, spawn_move, spawn_transfer,
    test_native_br::{check_br, spawn_incremental_backup, spawn_restore_keyspace},
    TikvConfig, BACKUP_COUNTER, CONCURRENCY, KEYSPACE_COUNTER, MERGE_COUNTER, MOVE_COUNTER,
    NODE_RESTART_COUNTER, RESTORE_COUNTER, TIMEOUT, TRANSFER_COUNTER, WRITE_COUNTER,
};

const INITIAL_KEYSPACE_COUNT: usize = 10;
const NODES_COUNT: usize = 4;
const RESTORE_CONCURRENCY: usize = 2;
// `INSTANT_BACKUP_INTERVAL` is more than 1 second as the incremental backup
// file name has a precision of 1 second.
const INSTANT_BACKUP_INTERVAL: Duration = Duration::from_millis(1050);

const REGION_BUCKET_SIZE: ReadableSize = ReadableSize::kb(64);

#[test]
fn test_random_all() {
    test_util::init_log_for_test();

    // Prepare.
    let (_temp_dir, _oss, dfs_config) = prepare_dfs("random_br_");
    let mut cluster = prepare_cluster(&dfs_config, NODES_COUNT, INITIAL_KEYSPACE_COUNT);
    let pd_client = cluster.get_pd_client();
    let keyspace_manager = cluster.keyspace_manager().clone();
    let backup_worker = {
        let backup_config = backup::BackupConfig {
            dfs: dfs_config.clone(),
            tolerate_err: 0, // TODO: enable tolerate_err = 1.
            skip_keyspace_meta: true,
            ..Default::default()
        };
        Arc::new(backup_worker::BackupWorker::new(
            backup_config,
            pd_client.clone(),
            INSTANT_BACKUP_INTERVAL,
            100,
        ))
    };

    // Start workloads & schedulers.
    let mut handles = vec![
        spawn_merge(cluster.new_scheduler(), true),
        spawn_transfer(cluster.new_scheduler()),
        spawn_move(cluster.new_scheduler(), Arc::new(RwLock::new(()))),
        // TODO: enable GC worker after verification error of restore is addressed.
        // spawn_gc_worker(cluster.get_pd_client(), TIMEOUT),
        spawn_create_keyspace(cluster.get_pd_client(), keyspace_manager.clone(), TIMEOUT),
        spawn_incremental_backup(
            cluster.new_client(),
            keyspace_manager.clone(),
            backup_worker,
            Duration::from_secs(10),
            TIMEOUT,
        ),
    ];
    for _ in 0..RESTORE_CONCURRENCY {
        handles.push(spawn_restore_keyspace(
            cluster.get_pd_client(),
            cluster.new_keyspace_client(),
            dfs_config.clone(),
            keyspace_manager.clone(),
            TIMEOUT,
        ));
    }
    for idx in 0..CONCURRENCY {
        handles.push(spawn_keyspace_write(
            idx,
            cluster.new_keyspace_client(),
            TIMEOUT,
        ));
    }

    // Main loop.
    let start_time = Instant::now();
    while start_time.saturating_elapsed() < TIMEOUT {
        // Restart nodes.
        random_node_restart(&mut cluster);
    }

    // Finish.
    info!("test finished, stopping all workers");
    for handle in handles {
        handle.join().unwrap();
    }

    // Verify.
    info!("verify cluster");
    let verified_records_count = verify_cluster(&mut cluster);

    // Stop cluster.
    info!("stopping cluster");
    cluster.stop();

    // Statistics.
    let total_write_count = WRITE_COUNTER.load(Ordering::SeqCst);
    let total_keyspace_count = KEYSPACE_COUNTER.load(Ordering::SeqCst);
    let total_merge_count = MERGE_COUNTER.load(Ordering::SeqCst);
    let total_move_count = MOVE_COUNTER.load(Ordering::SeqCst);
    let total_transfer_count = TRANSFER_COUNTER.load(Ordering::SeqCst);
    let total_node_restart = NODE_RESTART_COUNTER.load(Ordering::SeqCst);
    let total_backup_count = BACKUP_COUNTER.load(Ordering::SeqCst);
    let total_restore_count = RESTORE_COUNTER.load(Ordering::SeqCst);
    let region_number = pd_client.get_regions_number();
    info!(
        "TEST SUCCEED: write {}, keyspace {}, region {}, merge {}, move {}, transfer {}, node restart {}, backup {}, restore {}, verified_records {}",
        total_write_count,
        total_keyspace_count,
        region_number,
        total_merge_count,
        total_move_count,
        total_transfer_count,
        total_node_restart,
        total_backup_count,
        total_restore_count,
        verified_records_count,
    );
}

fn prepare_cluster(
    dfs_config: &DFSConfig,
    nodes_count: usize,
    initial_keyspace_count: usize,
) -> ServerCluster {
    let mut rng = rand::thread_rng();
    let nodes = alloc_node_id_vec(nodes_count);
    let dfs_config = Arc::new(dfs_config.clone());
    let update_conf_fn = move |_, conf: &mut TikvConfig| {
        conf.dfs = (*dfs_config).clone();
        conf.coprocessor.region_split_size = ReadableSize::kb(192);
        conf.coprocessor.region_bucket_size = REGION_BUCKET_SIZE;
        conf.raft_store.peer_stale_state_check_interval = ReadableDuration::secs(1);
        conf.raft_store.abnormal_leader_missing_duration = ReadableDuration::secs(3);
        conf.raft_store.max_leader_missing_duration = ReadableDuration::secs(5);
        conf.rocksdb.writecf.target_file_size_base = ReadableSize::kb(16);
        conf.rfengine.target_file_size = ReadableSize::mb(1);
        conf.rfengine.batch_compression_threshold =
            ReadableSize::kb(rand::thread_rng().gen_range(0..2));
        // TODO: test for both enable and disable inner_key_offset
        conf.enable_inner_key_offset = true;
    };
    let cluster = ServerCluster::new(nodes, update_conf_fn);
    cluster.wait_region_replicated(&[], 3);
    let pd_client = cluster.get_pd_client();
    pd_client.disable_default_operator();
    let region0 = pd_client.get_all_regions().first().unwrap().clone();

    // Split keyspaces.
    let mut keys = vec![];
    let mut keyspaces: Vec<u32> = vec![];
    for keyspace_id in 0..initial_keyspace_count {
        let keyspace_id = keyspace_id as u32;
        keyspaces.push(keyspace_id);
        keys.push(ApiV2::get_txn_keyspace_prefix(keyspace_id));
        let i_to_key = generate_keyspace_key(keyspace_id);
        for i in 0..rng.gen_range(0..10) {
            keys.push(i_to_key(i * 100));
        }
    }
    keys.push(ApiV2::get_txn_keyspace_prefix(
        initial_keyspace_count as u32,
    ));
    let encoded_keys = keys
        .iter()
        .map(|k| Key::from_raw(k).into_encoded())
        .collect();
    pd_client.must_split_region(region0, CheckPolicy::Usekey, encoded_keys);
    cluster.wait_pd_region_min_count(keys.len() + 1);
    cluster
        .keyspace_manager()
        .create_keyspaces(&keyspaces, 0, Some(&mut rng));
    KEYSPACE_COUNTER.store(initial_keyspace_count, Ordering::Relaxed);

    // Scatter regions.
    let move_scheduler = cluster.new_scheduler();
    for _ in 0..(keys.len() + 1) {
        move_scheduler.move_random_region();
    }

    // TODO: restart cluster with inner key offset enabled

    cluster
}

fn verify_cluster(cluster: &mut ServerCluster) -> usize /* records count in ref store */ {
    // Check statistics.
    let (res, data_stats) = try_wait_result(
        || {
            let stats = cluster.get_data_stats();
            (stats.check_data(), stats)
        },
        20,
    );
    assert!(
        res.is_ok(),
        "check_data failed: {:?}, stats: {:?}",
        res,
        data_stats
    );
    data_stats
        .check_buckets(&cluster.get_pd_client(), REGION_BUCKET_SIZE.0)
        .unwrap();

    // Verify data.
    let mut handles = vec![];
    for keyspace_id in cluster.keyspace_manager().ref_stores().all_keyspace_ids() {
        let mut client = cluster.new_keyspace_client();
        handles.push(std::thread::spawn(move || {
            (
                keyspace_id,
                client.verify_keyspace_with_ref_store(keyspace_id),
            )
        }));
    }
    let mut records_cnt = 0;
    for handle in handles {
        let (keyspace_id, res) = handle.join().unwrap();
        records_cnt += res.unwrap_or_else(|err| {
            panic!(
                "{} verify_keyspace_with_ref_store failed: {:?}",
                keyspace_id, err
            );
        });
    }
    assert!(records_cnt > 1000, "too few records in ref store");

    // Check BR.
    check_br();

    records_cnt
}
