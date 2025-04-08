// Copyright 2023 TiKV Project Authors. Licensed under Apache-2.0.

use std::{fmt, mem, sync::Arc, time::Duration};

use pd_client::PdClient;
use rfenginepb::ClusterBackupMeta;
use tikv_util::{
    config::ReadableDuration,
    debug, error, info,
    retry::try_wait_result_async,
    warn,
    worker::{Builder as WorkerBuilder, Runnable, RunnableWithTimer, Scheduler, Worker},
};

use crate::{
    backup,
    backup::{BackupConfig, BackupType, IncrementalBackupFile, Result, SharedResult},
    error::{Error, SharedError},
};

pub const DEFAULT_TIMEOUT_INSTANT_BACKUP: ReadableDuration = ReadableDuration::secs(60);

type InstantBackupCallback = Box<dyn FnOnce(SharedResult<Arc<IncrementalBackupFile>>) + Send>;

enum BackupTask {
    InstantBackup { cb: InstantBackupCallback },
    LightweightBackup { cb: InstantBackupCallback },
}

impl fmt::Display for BackupTask {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BackupTask::InstantBackup { .. } => write!(f, "instant backup"),
            BackupTask::LightweightBackup { .. } => write!(f, "lightweight backup"),
        }
    }
}

pub struct BackupWorker {
    worker: Worker,
    scheduler: Scheduler<BackupTask>,
}

impl BackupWorker {
    pub fn new(
        config: BackupConfig,
        pd_client: Arc<dyn PdClient>,
        backup_interval: Duration,
        max_concurrency: usize,
    ) -> Self {
        let worker = WorkerBuilder::new("backup-worker").create();
        let scheduler = worker.start_with_timer(
            "backup-worker",
            BackupRunner::new(config, pd_client, backup_interval, max_concurrency),
        );
        Self { worker, scheduler }
    }

    pub fn stop(&self) {
        self.scheduler.stop();
        self.worker.stop();
    }

    pub async fn instant_backup(&self, lightweight: bool) -> Result<Arc<IncrementalBackupFile>> {
        let (cb, fut) = tikv_util::future::paired_future_callback();
        let task = if lightweight {
            BackupTask::LightweightBackup { cb }
        } else {
            BackupTask::InstantBackup { cb }
        };
        self.scheduler.schedule(task).unwrap();
        fut.await.unwrap().map_err(|e| {
            warn!("instant backup failed"; "error" => ?e);
            e.into()
        })
    }

    pub async fn instant_backup_with_retry(
        &self,
        lightweight: bool,
        timeout: Duration,
    ) -> Result<Arc<IncrementalBackupFile>> {
        // TODO: retry only when the error is retryable
        try_wait_result_async(
            || Box::pin(self.instant_backup(lightweight)),
            timeout,
            || Duration::from_millis(500),
        )
        .await
    }
}

struct BackupRunner {
    config: BackupConfig,
    pd_client: Arc<dyn PdClient>,
    backup_interval: Duration,
    max_concurrency: usize,
    backup_request_queue: Vec<InstantBackupCallback>,
    last_backup_meta: Option<ClusterBackupMeta>,
}

impl BackupRunner {
    fn new(
        config: BackupConfig,
        pd_client: Arc<dyn PdClient>,
        backup_interval: Duration,
        max_concurrency: usize,
    ) -> Self {
        Self {
            config,
            pd_client,
            backup_interval,
            max_concurrency,
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

        let exec_backup = |backup_type: BackupType,
                           last_backup_meta: Option<ClusterBackupMeta>|
         -> Result<(String, ClusterBackupMeta)> {
            backup::backup_cluster(
                self.config.clone(),
                backup_type,
                "".to_string(),
                pd_client.as_ref(),
                last_backup_meta,
            )
        };

        let (backup_path, backup_meta) =
            match exec_backup(BackupType::Incremental, last_backup_meta) {
                Ok(res) => Ok(res),
                Err(e) if backup::need_full_backup(&e) => {
                    info!("Backup failed with {:?}, try full backup", e);
                    exec_backup(BackupType::Full, None)
                }
                Err(e) => Err(e),
            }?;
        debug!(
            "Backup succeeded: {:?}, meta: {:?}",
            backup_path, backup_meta
        );
        self.last_backup_meta = Some(backup_meta);
        Ok(IncrementalBackupFile::try_from_full_path(&backup_path).unwrap())
    }

    fn do_lightweight_backup(&mut self) -> Result<Arc<IncrementalBackupFile>> {
        let pd_client = self.pd_client.clone();
        let (backup_path, _) = backup::backup_cluster(
            self.config.clone(),
            BackupType::Lightweight,
            "".to_string(),
            pd_client.as_ref(),
            None,
        )?;
        Ok(Arc::new(
            IncrementalBackupFile::try_from_full_path(&backup_path).unwrap(),
        ))
    }
}

impl Runnable for BackupRunner {
    type Task = BackupTask;

    fn run(&mut self, task: BackupTask) {
        match task {
            BackupTask::InstantBackup { cb } => {
                if self.backup_request_queue.len() >= self.max_concurrency {
                    error!("Too many backup requests, drop this one");
                    cb(Err(SharedError::from(Error::ReachConcurrencyLimit(
                        self.max_concurrency,
                    ))));
                    return;
                }
                self.backup_request_queue.push(cb);
            }
            BackupTask::LightweightBackup { cb } => {
                let res = match self.do_lightweight_backup() {
                    Ok(backup_path) => Ok(backup_path),
                    Err(err) => Err(SharedError::from(err)),
                };
                cb(res);
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
