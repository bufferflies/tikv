// Copyright 2023 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    sync::{atomic::Ordering, Arc, RwLock},
    thread::{sleep, JoinHandle},
    time::Duration,
};

use api_version::ApiV2;
use futures::executor::block_on;
use kvengine::dfs::{DFSConfig, S3Fs};
use native_br::{
    backup, backup_worker,
    error::Error,
    restore_keyspace,
    restore_keyspace::{ReportRestoreStepTrait, RestoreStep},
};
use pd_client::PdClient;
use rand::{prelude::SliceRandom, Rng};
use security::SecurityConfig;
use test_cloud_server::{
    client::ClusterClient,
    keyspace::{ClusterKeyspaceClient, KeyspaceManager},
    try_wait, try_wait_result, ServerCluster,
};
use tikv_util::{
    config::{ReadableDuration, ReadableSize},
    info,
    time::Instant,
    warn,
};
use tokio::runtime::Runtime;

use crate::{
    alloc_node_id_vec, new_security_config, prepare_dfs, spawn_gc_worker,
    spawn_keyspace_write_deprecated, spawn_merge, spawn_move, spawn_transfer, TikvConfig,
    BACKUP_COUNTER, CONCURRENCY, MERGE_COUNTER, MOVE_COUNTER, NODE_RESTART_COUNTER,
    RESTORE_COUNTER, TIMEOUT, TRANSFER_COUNTER, WRITE_COUNTER,
};

const KEYSPACE_COUNT: usize = 10;
// `INSTANT_BACKUP_INTERVAL` is more than 1 second as the incremental backup
// file name has a precision of 1 second.
const INSTANT_BACKUP_INTERVAL: Duration = Duration::from_millis(1050);

// TODO: remove `test_random_br` after `test_random_all` implements "data
// branching".
#[test]
fn test_random_br() {
    test_random_br_helper(false, false);
    test_random_br_helper(true, false);
}

#[test]
fn test_random_data_branching() {
    test_random_br_helper(true, true);
}

fn test_random_br_helper(enable_inner_key_offset: bool, restore_to_new: bool) {
    test_util::init_log_for_test();

    let (_temp_dir, _oss, dfs_config) = prepare_dfs("random_br_");
    let security_conf = new_security_config();

    // Prepare cluster.
    let nodes = alloc_node_id_vec(5);
    let update_conf_fn = |_, conf: &mut TikvConfig| {
        conf.dfs = dfs_config.clone();
        conf.coprocessor.region_split_size = ReadableSize::kb(192);
        conf.coprocessor.region_bucket_size = ReadableSize::kb(64);
        conf.raft_store.peer_stale_state_check_interval = ReadableDuration::secs(1);
        conf.raft_store.abnormal_leader_missing_duration = ReadableDuration::secs(3);
        conf.raft_store.max_leader_missing_duration = ReadableDuration::secs(5);
        conf.rocksdb.writecf.target_file_size_base = ReadableSize::kb(16);
        conf.rfengine.target_file_size = ReadableSize::mb(1);
        conf.rfengine.batch_compression_threshold =
            ReadableSize::kb(rand::thread_rng().gen_range(0..2));
        conf.enable_inner_key_offset = enable_inner_key_offset;
        conf.security = security_conf.clone();
    };
    let mut cluster = ServerCluster::new(nodes.clone(), update_conf_fn);
    cluster.wait_region_replicated(&[], 3);
    let pd_client = cluster.get_pd_client();
    pd_client.disable_default_operator();
    let mut client = cluster.new_client();
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

    // Split keyspaces.
    for keyspace_id in 0..=KEYSPACE_COUNT {
        client.split(&ApiV2::get_txn_keyspace_prefix(keyspace_id as u32));
    }

    cluster.wait_pd_region_min_count(KEYSPACE_COUNT + 2);

    let mut rng = rand::thread_rng();
    let delta = rng.gen_range(5000..100000);

    // Split target keyspace region.
    if restore_to_new {
        for keyspace_id in delta..=delta + KEYSPACE_COUNT {
            client.split(&ApiV2::get_txn_keyspace_prefix(keyspace_id as u32));
        }
        cluster.wait_pd_region_min_count(2 * KEYSPACE_COUNT + 2);
    }

    let move_scheduler = cluster.new_scheduler();
    for _ in 0..20 {
        move_scheduler.move_random_region();
    }

    // Start workloads & schedulers.
    let mut handles = vec![
        spawn_transfer(cluster.new_scheduler()),
        spawn_move(cluster.new_scheduler(), Arc::new(RwLock::new(()))),
        spawn_merge(cluster.new_scheduler(), true),
        spawn_gc_worker(cluster.get_pd_client(), TIMEOUT),
        spawn_incremental_backup(
            cluster.new_client(),
            cluster.keyspace_manager().clone(),
            backup_worker,
            Duration::from_secs(3),
            TIMEOUT,
        ),
    ];
    for idx in 0..CONCURRENCY {
        handles.push(spawn_keyspace_write_deprecated(
            idx,
            cluster.new_client(),
            KEYSPACE_COUNT,
            TIMEOUT,
        ));
    }

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(8)
        .enable_all()
        .thread_name("restore-keyspace")
        .build()
        .unwrap();
    let reporter = Arc::new(DummyStepReporter::default());

    let start_time = Instant::now();
    while start_time.saturating_elapsed() < TIMEOUT {
        // TODO: restart node in another thread.
        // TODO: backup & restore when node is down.
        if rng.gen_ratio(1, 5) {
            let node_id = *nodes.choose(&mut rng).unwrap();
            info!(
                "stop node {}, store_id {}",
                node_id,
                cluster.get_store_id(node_id)
            );
            cluster.stop_node(node_id);
            info!(
                "finish stop node {}, current nodes {:?}",
                node_id,
                cluster.get_nodes()
            );

            let sleep_sec = rng.gen_range(1..5);
            sleep(Duration::from_secs(sleep_sec));

            info!("start node {}", node_id);
            cluster.start_node(node_id, update_conf_fn);
            info!(
                "finish start node {}, store_id {}, current nodes {:?}",
                node_id,
                cluster.get_store_id(node_id),
                cluster.get_nodes()
            );
            NODE_RESTART_COUNTER.fetch_add(1, Ordering::SeqCst);

            // TODO: Wait for leader election before another restart
            // let ok = try_wait(||
            // cluster.get_data_stats().check_leader().is_ok(), 20);
            // if !ok {
            //     cluster.get_data_stats().check_leader().unwrap();
            // }
        }

        let (backup_name, backup_ts) = match cluster
            .keyspace_manager()
            .ref_stores()
            .get_random_backup(&mut rng)
        {
            Some(backup) => backup,
            None => {
                sleep(Duration::from_millis(100));
                continue;
            }
        };

        let keyspace = rng.gen_range(0..KEYSPACE_COUNT) as u32;
        let target_keyspace = if restore_to_new {
            keyspace + delta as u32
        } else {
            keyspace
        };
        do_restore_keyspace(
            pd_client.clone(),
            &runtime,
            dfs_config.clone(),
            security_conf.clone(),
            keyspace,
            target_keyspace,
            &backup_name,
            Some(backup_ts),
            reporter.clone(),
        )
        .unwrap();
        info!(
            "restore keyspace {}->{} from backup {} success",
            keyspace, target_keyspace, backup_name,
        );
        RESTORE_COUNTER.fetch_add(1, Ordering::SeqCst);
    }

    info!("test finished, stopping all workers");
    for handle in handles {
        handle.join().unwrap();
    }
    let ok = try_wait(
        || {
            let data_stats = cluster.get_data_stats();
            data_stats.check_data().is_ok()
        },
        20,
    );
    if !ok {
        cluster.get_data_stats().check_data().unwrap();
    }

    // TODO: verify data
    // let mut client = cluster.new_client();
    // client.verify_data_with_ref_store();
    check_br();

    cluster.stop();
    let total_write_count = WRITE_COUNTER.load(Ordering::SeqCst);
    let total_merge_count = MERGE_COUNTER.load(Ordering::SeqCst);
    let total_move_count = MOVE_COUNTER.load(Ordering::SeqCst);
    let total_transfer_count = TRANSFER_COUNTER.load(Ordering::SeqCst);
    let total_node_restart = NODE_RESTART_COUNTER.load(Ordering::SeqCst);
    let total_backup_count = BACKUP_COUNTER.load(Ordering::SeqCst);
    let total_restore_count = RESTORE_COUNTER.load(Ordering::SeqCst);
    let region_number = pd_client.get_regions_number();
    info!(
        "TEST SUCCEED: write {}, region {}, merge {}, move {}, transfer {}, node restart {}, backup {}, restore {}",
        total_write_count,
        region_number,
        total_merge_count,
        total_move_count,
        total_transfer_count,
        total_node_restart,
        total_backup_count,
        total_restore_count
    );
}

pub(crate) fn do_restore_keyspace(
    pd_client: Arc<dyn PdClient>,
    runtime: &Runtime,
    dfs_config: DFSConfig,
    security_config: SecurityConfig,
    keyspace: u32,
    target_keyspace: u32,
    backup_name: &str,
    truncate_ts: Option<u64>,
    reporter: Arc<dyn ReportRestoreStepTrait>,
) -> native_br::Result<restore_keyspace::RestoredKeyspace> {
    let s3fs = Arc::new(S3Fs::new(
        dfs_config.prefix,
        dfs_config.s3_endpoint,
        dfs_config.s3_key_id,
        dfs_config.s3_secret_key,
        dfs_config.s3_region,
        dfs_config.s3_bucket,
    ));
    restore_keyspace::restore_keyspace(
        keyspace,
        target_keyspace,
        backup_name,
        None,
        s3fs,
        security_config,
        pd_client,
        runtime,
        truncate_ts,
        reporter,
    )
}

pub(crate) fn spawn_incremental_backup(
    client: ClusterClient,
    keyspace_manager: KeyspaceManager,
    backup_worker: Arc<backup_worker::BackupWorker>,
    interval: Duration,
    timeout: Duration,
) -> JoinHandle<()> {
    std::thread::spawn(move || {
        let start_time = Instant::now();
        let mut last_backup_time = start_time;
        while start_time.saturating_elapsed() < timeout {
            let guard = keyspace_manager.lock_for_backup();
            let backup_ts = client.get_ts().into_inner();
            let ref_stores = keyspace_manager.ref_stores().dump();
            drop(guard);

            let backup_file = match block_on(backup_worker.instant_backup()) {
                Ok(backup_file) => backup_file,
                Err(err) if is_backup_error_retryable(&err) => {
                    warn!("backup failed, retry: {:?}", err);
                    continue;
                }
                Err(err) => {
                    panic!("backup failed: {:?}", err);
                }
            };
            keyspace_manager.ref_stores().add_backup(
                backup_file.name().to_string(),
                backup_ts,
                ref_stores,
            );

            info!("backup done: {:?}", backup_file);
            BACKUP_COUNTER.fetch_add(1, Ordering::SeqCst);

            let backup_elapsed = last_backup_time.saturating_elapsed();
            sleep(interval.saturating_sub(backup_elapsed));
            last_backup_time = Instant::now();
        }
        info!("incremental backup thread exit");
    })
}

fn is_backup_error_retryable(err: &Error) -> bool {
    match err {
        Error::TopoChanged(_) | Error::MetaNotFound(_) | Error::HttpError(_) => true,
        Error::SharedError(err) => is_backup_error_retryable(err.inner()),
        _ => false,
    }
}

pub(crate) fn check_br() {
    let total_backup_count = BACKUP_COUNTER.load(Ordering::SeqCst);
    let total_restore_count = RESTORE_COUNTER.load(Ordering::SeqCst);

    assert!(
        // It's possible that backup thread is difficult to acquire the write lock.
        total_backup_count > 0,
        "backup count too small: {}",
        total_backup_count
    );
    assert!(
        total_restore_count > 0,
        "restore count too small: {}",
        total_restore_count
    );
}

pub(crate) fn spawn_restore_keyspace(
    pd_client: Arc<dyn PdClient>,
    mut client: ClusterKeyspaceClient,
    dfs_config: DFSConfig,
    security_config: SecurityConfig,
    keyspace_manager: KeyspaceManager,
    timeout: Duration,
) -> JoinHandle<()> {
    std::thread::spawn(move || {
        let mut rng = rand::thread_rng();
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(4)
            .enable_all()
            .thread_name("restore-keyspace")
            .build()
            .unwrap();
        let reporter = Arc::new(DummyStepReporter::default());

        let start_time = Instant::now();
        sleep(Duration::from_secs(3));
        while start_time.saturating_elapsed() < timeout {
            let (backup_name, backup_ts) =
                match keyspace_manager.ref_stores().get_random_backup(&mut rng) {
                    Some(backup) => backup,
                    None => {
                        sleep(Duration::from_millis(500));
                        continue;
                    }
                };

            // Restore pick a random keyspace uniformly, to generate the scenario that some
            // big keyspaces are never restored.
            let keyspace = keyspace_manager.get_uniform_random_keyspace(&mut rng);
            // TODO: test data branching.
            let target_keyspace = keyspace;

            // TODO: test without blocking write workloads.
            {
                let lock = keyspace_manager.get_keyspace_lock(keyspace);
                let _guard = lock.lock_for_restore_with_verify();

                client
                    .verify_keyspace_with_ref_store(target_keyspace)
                    .unwrap_or_else(|err| {
                        panic!(
                            "{}->{} verify_keyspace_with_ref_store (before restore): {:?}",
                            keyspace, target_keyspace, err
                        )
                    });

                // We always perform PiTR here. Snapshot restore will block write workload
                // during the whole backup process, which is not efficient.
                // And actually there are only trivial differences between PiTR and snapshot
                // restore.
                match do_restore_keyspace(
                    pd_client.clone(),
                    &runtime,
                    dfs_config.clone(),
                    security_config.clone(),
                    keyspace,
                    target_keyspace,
                    &backup_name,
                    Some(backup_ts),
                    reporter.clone(),
                ) {
                    Ok(_) => {}
                    Err(Error::BackupEmptyForKeyspace(_)) => {
                        // Empty backup will happen on newly created keyspace. Retry.
                        warn!("{}->{} backup is empty, retry", keyspace, target_keyspace);
                        continue;
                    }
                    Err(err) => panic!(
                        "{}->{} restore failed: {:?}",
                        keyspace, target_keyspace, err
                    ),
                }
                keyspace_manager.ref_stores().restore_keyspace(
                    keyspace,
                    &backup_name,
                    target_keyspace,
                );

                // To find data corruption early, and generate read workload as well.
                // The retry should not be necessary.
                // TODO: Remove the retry after verification issue is addressed.
                let (verify_res, _) = try_wait_result(
                    || {
                        let verify_res = client.verify_keyspace_with_ref_store(target_keyspace);
                        if verify_res.is_err() {
                            warn!(
                                "{}->{} verify_keyspace_with_ref_store failed (after restore): {:?}",
                                keyspace, target_keyspace, verify_res
                            );
                        }
                        (verify_res.map(|_| ()), ())
                    },
                    10,
                );
                assert!(
                    verify_res.is_ok(),
                    "{}->{} verify_keyspace_with_ref_store (after restore): {:?}",
                    keyspace,
                    target_keyspace,
                    verify_res
                );
            }
            info!(
                "restore keyspace {}->{} from backup {} success",
                keyspace, target_keyspace, backup_name,
            );
            RESTORE_COUNTER.fetch_add(1, Ordering::Relaxed);

            sleep(Duration::from_secs(rng.gen_range(0..10)));
        }
        info!("restore keyspace thread exit");
    })
}

#[derive(Default)]
struct DummyStepReporter {}

impl ReportRestoreStepTrait for DummyStepReporter {
    fn report_step(&self, _step: RestoreStep) {}
}
