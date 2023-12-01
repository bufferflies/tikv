// Copyright 2023 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    sync::{atomic::Ordering, Arc},
    thread::{sleep, JoinHandle},
    time::Duration,
};

use kvengine::dfs::{DFSConfig, S3Fs};
use native_br::{
    backup_worker,
    error::Error,
    restore::get_cluster_backup_meta,
    restore_keyspace,
    restore_keyspace::{ReportRestoreStepTrait, RestoreStep},
};
use pd_client::PdClient;
use rand::Rng;
use security::SecurityConfig;
use test_cloud_server::{
    client::ClusterClient,
    keyspace::{ClusterKeyspaceClient, KeyspaceManager},
    try_wait_result,
};
use tikv_util::{info, time::Instant, warn};
use tokio::runtime::Runtime;

use crate::{BACKUP_COUNTER, RESTORE_COUNTER};

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

pub(crate) fn spawn_backup(
    client: ClusterClient,
    keyspace_manager: KeyspaceManager,
    backup_worker: Arc<backup_worker::BackupWorker>,
    interval: Duration,
    timeout: Duration,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let start_time = Instant::now();
        let mut last_backup_time = start_time;
        while start_time.saturating_elapsed() < timeout {
            // Note:
            // 1. Backup for RefStore of single keyspace to avoid global locking.
            // 2. The backup of TiKV is still for the whole cluster.
            // 3. As the keyspace of backup determines the keyspace to restore, we pick the
            //    random keyspace uniformly, to generate the scenario that some big
            //    keyspaces are never restored.
            let keyspace_id = {
                let mut rng = rand::thread_rng();
                keyspace_manager.get_uniform_random_keyspace(&mut rng)
            };

            let lock = keyspace_manager.get_keyspace_lock(keyspace_id);
            let guard = lock.mutex_lock().await;
            let backup_ts = client.get_ts().into_inner();
            let mut keyspace_backup = keyspace_manager.backup_keyspace(keyspace_id, backup_ts);
            // Downgrade to shared lock, to enable write workloads on the keyspace, but
            // block restore workload.
            // As in scene of restoration, `truncate_ts` cannot truncate the extra restored
            // data after `backup_ts`.
            // See https://github.com/tidbcloud/cloud-storage-engine/issues/1094.
            let shared_guard = guard.downgrade();

            let do_lightweight_backup = keyspace_id % 2 == 0;

            if do_lightweight_backup {
                info!("spawn lightweight backup");
            } else {
                info!("spawn incremental backup");
            }
            let backup_file = match backup_worker.instant_backup(do_lightweight_backup).await {
                Ok(backup_file) => backup_file,
                Err(err) if is_backup_error_retryable(&err) => {
                    warn!("backup failed, retry: {:?}", err);
                    continue;
                }
                Err(err) => {
                    panic!("backup failed: {:?}", err);
                }
            };

            keyspace_backup.backup_name = Some(backup_file.name().to_string());
            keyspace_manager.add_backup(keyspace_backup);
            drop(shared_guard);

            info!(
                "instant backup success, keyspace {}, lightweight {}, file {:?}",
                keyspace_id, do_lightweight_backup, backup_file
            );

            BACKUP_COUNTER.fetch_add(1, Ordering::SeqCst);

            let backup_elapsed = last_backup_time.saturating_elapsed();
            tokio::time::sleep(interval.saturating_sub(backup_elapsed)).await;
            last_backup_time = Instant::now();
        }
        info!("backup thread exit");
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
        // The total_backup_count is unstable and lower than expected.
        // TODO: investigate the reason.
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
            let backup = match keyspace_manager.get_random_backup(&mut rng) {
                Some(backup) => backup,
                None => {
                    sleep(Duration::from_millis(500));
                    continue;
                }
            };

            let source_keyspace = backup.keyspace_id;
            // TODO: test data branching.
            let target_keyspace = source_keyspace;
            let backup_name = backup.backup_name().to_string();
            let tag = format!("{}->{}[{}]", source_keyspace, target_keyspace, backup_name);

            // TODO: test without blocking write workloads.
            {
                let lock = keyspace_manager.get_keyspace_lock(target_keyspace);
                let _guard = runtime.block_on(lock.mutex_lock());

                // The process of destroying ranges is not determined. So skip verifying the
                // destroying ranges.
                runtime
                    .block_on(client.verify_keyspace_and_skip_destroyed_ranges(target_keyspace))
                    .unwrap_or_else(|err| {
                        panic!(
                            "{} verify_keyspace_with_ref_store (before restore): {:?}",
                            tag, err
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
                    source_keyspace,
                    target_keyspace,
                    &backup_name,
                    Some(backup.backup_ts),
                    reporter.clone(),
                ) {
                    Ok(_) => {}
                    Err(Error::BackupEmptyForKeyspace(_)) => {
                        // Empty backup will happen on newly created keyspace. Retry.
                        warn!("{} backup is empty, retry", tag);
                        continue;
                    }
                    Err(err) => panic!("{} restore failed: {:?}", tag, err),
                }
                keyspace_manager.restore_keyspace(&tag, backup, target_keyspace);

                // To find data corruption early, and generate read workload as well.
                // The retry should not be necessary.
                // TODO: Remove the retry after verification issue is addressed.
                let (verify_res, _) = try_wait_result(
                    || {
                        let verify_res = runtime.block_on(
                            client.verify_keyspace_and_skip_destroyed_ranges(target_keyspace),
                        );
                        if verify_res.is_err() {
                            warn!(
                                "{} verify_keyspace_with_ref_store failed (after restore): {:?}",
                                tag, verify_res
                            );
                        }
                        (verify_res.map(|_| ()), ())
                    },
                    10,
                );
                if verify_res.is_err() {
                    let s3fs = S3Fs::new(
                        dfs_config.prefix,
                        dfs_config.s3_endpoint,
                        dfs_config.s3_key_id,
                        dfs_config.s3_secret_key,
                        dfs_config.s3_region,
                        dfs_config.s3_bucket,
                    );
                    let backup_meta = get_cluster_backup_meta(&s3fs, backup_name);
                    info!("{} backup_meta: {:?}", tag, backup_meta);
                    panic!(
                        "{} verify_keyspace_with_ref_store (after restore): {:?}",
                        tag, verify_res
                    );
                }
            }
            info!("{} restore keyspace success", tag);
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
