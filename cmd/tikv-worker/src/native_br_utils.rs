// Copyright 2023 TiKV Project Authors. Licensed under Apache-2.0.

use std::{fmt, mem, sync::Arc, time::Duration};

use native_br::{
    backup,
    backup::{BackupConfig, IncrementalBackupFile},
};
use pd_client::PdClient;
use rfenginepb::ClusterBackupMeta;
use tikv_util::{
    debug, error, info,
    worker::{Builder as WorkerBuilder, Runnable, RunnableWithTimer, Scheduler, Worker},
};

use crate::{
    error::{Error, SharedError},
    native_br::{Result, SharedResult, MAX_RESTORE_CONCURRENCY},
};

type InstantBackupCallback = Box<dyn FnOnce(SharedResult<Arc<IncrementalBackupFile>>) + Send>;

enum BackupTask {
    InstantBackup { cb: InstantBackupCallback },
}

impl fmt::Display for BackupTask {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BackupTask::InstantBackup { .. } => write!(f, "instant backup"),
        }
    }
}

pub(crate) struct BackupWorker {
    worker: Worker,
    scheduler: Scheduler<BackupTask>,
}

impl BackupWorker {
    pub fn new(
        config: BackupConfig,
        pd_client: Arc<dyn PdClient>,
        backup_interval: Duration,
    ) -> Self {
        let worker = WorkerBuilder::new("backup-worker").create();
        let scheduler = worker.start_with_timer(
            "backup-worker",
            BackupRunner::new(config, pd_client, backup_interval),
        );
        Self { worker, scheduler }
    }

    pub fn stop(&self) {
        self.scheduler.stop();
        self.worker.stop();
    }

    pub async fn instant_backup(&self) -> SharedResult<Arc<IncrementalBackupFile>> {
        let (cb, fut) = tikv_util::future::paired_future_callback();
        self.scheduler
            .schedule(BackupTask::InstantBackup { cb })
            .unwrap();
        fut.await.unwrap()
    }
}

struct BackupRunner {
    config: BackupConfig,
    pd_client: Arc<dyn PdClient>,
    backup_interval: Duration,
    backup_request_queue: Vec<InstantBackupCallback>,
    last_backup_meta: Option<ClusterBackupMeta>,
}

impl BackupRunner {
    fn new(config: BackupConfig, pd_client: Arc<dyn PdClient>, backup_interval: Duration) -> Self {
        Self {
            config,
            pd_client,
            backup_interval,
            backup_request_queue: Vec::new(),
            last_backup_meta: None,
        }
    }

    fn handle_backup_requests(&mut self) {
        let backup_requests = mem::take(&mut self.backup_request_queue);
        if backup_requests.is_empty() {
            return;
        }

        let res = match self.do_instant_backup() {
            Ok(f) => {
                info!("Backup succeeded: {:?}", f);
                Ok(Arc::new(f))
            }
            Err(err) => {
                error!("Backup failed: {:?}", err);
                Err(SharedError::from(err))
            }
        };
        for cb in backup_requests {
            cb(res.clone());
        }
    }

    fn do_instant_backup(&mut self) -> Result<IncrementalBackupFile> {
        let pd_client = self.pd_client.clone();
        let last_backup_meta = self.last_backup_meta.take();

        let exec_backup = |incremental: bool,
                           last_backup_meta: Option<ClusterBackupMeta>|
         -> native_br::error::Result<(String, ClusterBackupMeta)> {
            backup::backup_cluster(
                self.config.clone(),
                incremental,
                "".to_string(),
                pd_client.as_ref(),
                last_backup_meta,
            )
        };

        let (backup_path, backup_meta) = match exec_backup(true, last_backup_meta) {
            Ok(res) => Ok(res),
            Err(e) => {
                if backup::need_full_backup(&e) {
                    info!("Backup failed with {:?}, try full backup", e);
                    exec_backup(false, None).map_err(|e| Error::NativeBackupRestoreError(e))
                } else {
                    Err(Error::NativeBackupRestoreError(e))
                }
            }
        }?;
        debug!(
            "Backup succeeded: {:?}, meta: {:?}",
            backup_path, backup_meta
        );
        self.last_backup_meta = Some(backup_meta);
        Ok(IncrementalBackupFile::try_from_full_path(&backup_path).unwrap())
    }
}

impl Runnable for BackupRunner {
    type Task = BackupTask;

    fn run(&mut self, task: BackupTask) {
        match task {
            BackupTask::InstantBackup { cb } => {
                if self.backup_request_queue.len() >= MAX_RESTORE_CONCURRENCY {
                    error!("Too many backup requests, drop this one");
                    cb(Err(SharedError::from(Error::ReachConcurrencyLimit(
                        MAX_RESTORE_CONCURRENCY,
                    ))));
                    return;
                }
                self.backup_request_queue.push(cb);
            }
        }
    }
}

impl RunnableWithTimer for BackupRunner {
    fn on_timeout(&mut self) {
        self.handle_backup_requests();
    }

    fn get_interval(&self) -> Duration {
        self.backup_interval
    }
}
