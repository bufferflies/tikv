// Copyright 2023 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    sync::{atomic::Ordering, Arc, RwLock},
    time::Duration,
};

use api_version::ApiV2;
use cloud_encryption::KeyspaceEncryptionConfig;
use futures::executor::block_on;
use kvengine::dfs::DFSConfig;
use kvproto::pdpb::CheckPolicy;
use load_data::task::LoadDataConfig;
use native_br::{backup, backup_worker, restore::RestoreConfig};
use pd_client::PdClient;
use rand::Rng;
use security::SecurityConfig;
use test_cloud_server::{oss::prepare_dfs, try_wait_result, ServerCluster};
use tikv_util::{
    config::{ReadableDuration, ReadableSize},
    info,
    time::Instant,
    warn,
};
use txn_types::Key;

use crate::{test_drop_table::*, test_load_data::*, test_native_br::*, TikvConfig, *};

const INITIAL_KEYSPACE_COUNT: usize = 10;
const INITIAL_TABLE_COUNT: usize = 3;
const NODES_COUNT: usize = 4;
const RESTORE_CONCURRENCY: usize = 2;
const LOAD_DATA_CONCURRENCY: usize = 2;
// `INSTANT_BACKUP_INTERVAL` is more than 1 second as the incremental backup
// file name has a precision of 1 second.
const INSTANT_BACKUP_INTERVAL: Duration = Duration::from_millis(1050);

const REGION_BUCKET_SIZE: ReadableSize = ReadableSize::kb(64);

#[test]
fn test_random_all() {
    test_util::init_log_for_test();
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .worker_threads(4)
        .thread_name("random-workload")
        .build()
        .unwrap();
    let _guard = runtime.enter();

    // Prepare.
    let (_temp_dir, _oss, dfs_config) = prepare_dfs("oss_");
    let security_conf = new_security_config();
    let mut cluster = prepare_cluster(
        &dfs_config,
        &security_conf,
        NODES_COUNT,
        INITIAL_KEYSPACE_COUNT,
    );
    let pd_client = cluster.get_pd_client();
    let keyspace_manager = cluster.keyspace_manager().clone();

    let backup_config = backup::BackupConfig {
        dfs: dfs_config.clone(),
        tolerate_err: 0, // TODO: enable tolerate_err = 1.
        skip_keyspace_meta: true,
        ..Default::default()
    };
    let backup_worker = {
        Arc::new(backup_worker::BackupWorker::new(
            backup_config,
            pd_client.clone(),
            INSTANT_BACKUP_INTERVAL,
            100,
        ))
    };
    let load_data_config = {
        let tikv_config = cluster.get_node_config(cluster.get_nodes()[0]);
        let region_size = tikv_config.coprocessor.region_split_size.0 as usize;
        LoadDataConfig {
            block_size: tikv_config.rocksdb.writecf.block_size.0 as usize,
            sst_file_size: tikv_config.rocksdb.writecf.target_file_size_base.0 as usize,
            region_size,
            coarse_split_size: region_size * 4,
            enable_check_point: false,
        }
    };

    // Set the first service safe point for backup before starting workloads.
    let ts = block_on(pd_client.get_tso()).unwrap();
    backup::update_service_safe_point(pd_client.as_ref(), ts.into_inner()).unwrap();

    // Start workloads & schedulers.
    let mut handles = vec![
        spawn_merge(cluster.new_scheduler(), true),
        spawn_transfer(cluster.new_scheduler()),
        spawn_move(cluster.new_scheduler(), Arc::new(RwLock::new(()))),
        spawn_create_keyspace(
            cluster.get_pd_client(),
            keyspace_manager.clone(),
            INITIAL_TABLE_COUNT,
            TIMEOUT,
        ),
        spawn_major_compact(cluster.get_pd_client(), keyspace_manager.clone(), TIMEOUT),
    ];

    let restore_config = RestoreConfig {
        dfs: dfs_config.clone(),
        security: security_conf.clone(),
        timeout_wait_flush: ReadableDuration::secs(30),
        timeout_restore_snapshot: ReadableDuration::secs(30),
        max_retry: 20,
        ..Default::default()
    };
    for _ in 0..RESTORE_CONCURRENCY {
        handles.push(spawn_restore_keyspace(
            cluster.get_pd_client(),
            runtime.block_on(cluster.new_keyspace_client()),
            restore_config.clone(),
            keyspace_manager.clone(),
            TIMEOUT,
        ));
    }
    for _ in 0..LOAD_DATA_CONCURRENCY {
        handles.push(spawn_load_data(
            cluster.get_pd_client(),
            runtime.block_on(cluster.new_keyspace_client()),
            dfs_config.clone(),
            security_conf.clone(),
            load_data_config.clone(),
            keyspace_manager.clone(),
            Duration::from_secs(15),
            TIMEOUT,
        ));
    }

    let mut async_handles = vec![
        spawn_backup(
            cluster.new_client(),
            keyspace_manager.clone(),
            backup_worker,
            Duration::from_secs(5),
            TIMEOUT,
        ),
        spawn_gc_worker(
            runtime.block_on(cluster.new_txn_client()),
            pd_client.clone(),
            keyspace_manager.clone(),
            TIMEOUT,
        ),
    ];
    for _ in 0..DROP_TABLE_CONCURRENCY {
        async_handles.push(spawn_drop_table(
            runtime.block_on(cluster.new_keyspace_client()),
            Duration::from_secs(10),
            TIMEOUT,
        ));
    }
    for idx in 0..CONCURRENCY {
        async_handles.push(spawn_keyspace_write(
            idx,
            runtime.block_on(cluster.new_keyspace_client()),
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
    runtime.block_on(async move {
        for handle in async_handles {
            handle.await.unwrap();
        }
    });

    info!("run all pending destroy range");
    let txn_client = runtime.block_on(cluster.new_txn_client());
    let all_keyspaces = keyspace_manager.get_all_keyspaces();
    for keyspace_id in all_keyspaces {
        runtime
            .block_on(pick_and_run_pending_destroy_range(
                keyspace_id,
                u64::MAX,
                &keyspace_manager,
                &txn_client,
            ))
            .unwrap();
    }
    // Read requests can still get data before destroy range is done by compaction.
    // So must wait for destroy range to finish before verifying.
    info!("wait destroy range");
    cluster.wait_for_destroy_range(Duration::from_secs(30));

    // Verify.
    info!("verify cluster");
    let verified_records_count = runtime.block_on(verify_cluster(&mut cluster));

    // Stop cluster.
    info!("stopping cluster");
    cluster.stop();

    // Statistics.
    let total_write_count = WRITE_COUNTER.load(Ordering::SeqCst);
    let total_keyspace_count = KEYSPACE_COUNTER.load(Ordering::SeqCst);
    let total_table_count = TABLE_COUNTER.load(Ordering::SeqCst);
    let total_drop_table_count = DROP_TABLE_COUNTER.load(Ordering::SeqCst);
    let total_merge_count = MERGE_COUNTER.load(Ordering::SeqCst);
    let total_move_count = MOVE_COUNTER.load(Ordering::SeqCst);
    let total_transfer_count = TRANSFER_COUNTER.load(Ordering::SeqCst);
    let total_node_restart = NODE_RESTART_COUNTER.load(Ordering::SeqCst);
    let total_backup_count = BACKUP_COUNTER.load(Ordering::SeqCst);
    let total_restore_count = RESTORE_COUNTER.load(Ordering::SeqCst);
    let total_load_data_count = LOAD_DATA_COUNTER.load(Ordering::SeqCst);
    let total_manual_major_compact = MANUAL_MAJOR_COMPACT_COUNTER.load(Ordering::SeqCst);
    let total_gc_resolved_locks = GC_ADVANCE_SAFE_POINT_COUNTER.load(Ordering::SeqCst);
    let region_number = pd_client.get_regions_number();
    info!(
        "TEST SUCCEED: write {}, keyspace {}, table {}, drop table {}, region {}, merge {}, move {}, transfer {}, node restart {}, backup {}, restore {}, load_data {}, manual_major_compact {}, verified_records {}, gc {}",
        total_write_count,
        total_keyspace_count,
        total_table_count,
        total_drop_table_count,
        region_number,
        total_merge_count,
        total_move_count,
        total_transfer_count,
        total_node_restart,
        total_backup_count,
        total_restore_count,
        total_load_data_count,
        total_manual_major_compact,
        verified_records_count,
        total_gc_resolved_locks,
    );
}

fn prepare_cluster(
    dfs_config: &DFSConfig,
    security_conf: &SecurityConfig,
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
        conf.rocksdb.writecf.block_size = ReadableSize::kb(4);
        conf.rocksdb.writecf.target_file_size_base = ReadableSize::kb(16);
        conf.rfengine.target_file_size = ReadableSize::mb(8);
        conf.rfengine.batch_compression_threshold =
            ReadableSize::kb(rand::thread_rng().gen_range(0..2));
        conf.rfengine.lightweight_backup = true;
        conf.rfengine.wal_chunk_target_file_size = ReadableSize::kb(512);
        // TODO: test for both enable and disable inner_key_offset
        conf.enable_inner_key_offset = true;
        conf.security = security_conf.clone();
        conf.kvengine.compaction_tombs_count = 100;
        conf.kvengine.max_del_range_delay = ReadableDuration(Duration::from_secs(3));
    };
    let mut cluster = ServerCluster::new(nodes, update_conf_fn);
    cluster.start_pd_server(1);
    cluster.wait_region_replicated(&[], 3);
    let pd_client = cluster.get_pd_client();
    pd_client.disable_default_operator();
    let region0 = pd_client.get_all_regions().first().unwrap().clone();

    // Split keyspaces.
    let mut keys = vec![];
    let mut keyspaces: Vec<u32> = vec![];
    let mut data_keys = vec![];
    for _ in 0..initial_keyspace_count {
        // New keyspace must allocated by keyspace manager to avoid conflicts.
        let keyspace_id = cluster.keyspace_manager().new_keyspace_id(1);
        keyspaces.push(keyspace_id);
        keys.push(ApiV2::get_txn_keyspace_prefix(keyspace_id));
        let i_to_key = generate_keyspace_key(keyspace_id);
        for i in 0..rng.gen_range(0..10) {
            data_keys.push(Key::from_raw(&i_to_key(i * 100)).into_encoded());
        }
        let cfg = KeyspaceEncryptionConfig { enabled: true };
        match pd_client.set_keyspace_encryption(keyspace_id, cfg) {
            Ok(_) => {}
            Err(err) if pd_client::grpc_error_is_unimplemented(&err) => {
                info!("set_keyspace_encryption is not supported, skip");
            }
            Err(err) => {
                panic!("set_keyspace_encryption failed: {:?}", err)
            }
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
    cluster.keyspace_manager().create_keyspaces(
        &keyspaces,
        DEFAULT_INNER_KEY_OFFSET,
        INITIAL_TABLE_COUNT,
        Some(&mut rng),
    );
    KEYSPACE_COUNTER.store(initial_keyspace_count, Ordering::Relaxed);
    TABLE_COUNTER.store(
        initial_keyspace_count * INITIAL_TABLE_COUNT,
        Ordering::Relaxed,
    );
    let res = block_on(pd_client.split_regions(data_keys));
    if let Err(err) = res {
        warn!("split regions failed: {:?}", err);
    }

    // Scatter regions.
    let move_scheduler = cluster.new_scheduler();
    for _ in 0..(keys.len() + 1) {
        move_scheduler.move_random_region();
    }

    // TODO: restart cluster with inner key offset enabled

    cluster
}

async fn verify_cluster(cluster: &mut ServerCluster) -> usize /* records count in ref store */ {
    // Verify data.
    let mut handles = vec![];
    for keyspace_id in cluster.keyspace_manager().ref_stores().all_keyspace_ids() {
        let mut client = cluster.new_keyspace_client().await;
        handles.push(tokio::spawn(async move {
            (keyspace_id, client.verify_keyspace(keyspace_id).await)
        }));
    }
    let mut records_cnt = 0;
    for handle in handles {
        let (keyspace_id, res) = handle.await.unwrap();
        records_cnt += res.unwrap_or_else(|err| {
            panic!(
                "{} verify_keyspace_with_ref_store failed: {:?}",
                keyspace_id, err
            );
        });
    }
    assert!(records_cnt > 1000, "too few records in ref store");

    // Check statistics.
    // Check after verify data, to ensure that PD heartbeat have updated region
    // stats.
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
    cluster.wait_region_version_match();
    data_stats
        .check_buckets(&cluster.get_pd_client(), REGION_BUCKET_SIZE.0)
        .unwrap();

    check_br();
    check_load_data();
    check_gc();
    check_drop_table();

    records_cnt
}
