// Copyright 2023 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    fmt,
    sync::Arc,
    thread,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use pd_client::PdClient;
use rfenginepb::ClusterBackupMeta;
use tikv_util::{
    config::ReadableDuration,
    error, info,
    retry::try_wait_result_async,
    time::Instant,
    warn,
    worker::{Builder as WorkerBuilder, Runnable, RunnableWithTimer, Scheduler, Worker},
};

use crate::{
    backup,
    backup::{BackupConfig, BackupType, IncrementalBackupFile, Result, SharedResult},
    error::SharedError,
    metrics::{NATIVE_BR_BACKUP_ERROR, NATIVE_BR_BACKUP_SUCCESS},
};

pub const DEFAULT_TIMEOUT_INSTANT_BACKUP: ReadableDuration = ReadableDuration::secs(60);
const MIN_BACKUP_INTERVAL: Duration = Duration::from_millis(1050);

type InstantBackupCallback = Box<dyn FnOnce(SharedResult<Arc<IncrementalBackupFile>>) + Send>;

enum BackupTask {
    LightweightBackup { cb: InstantBackupCallback },
}

impl fmt::Display for BackupTask {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
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
    ) -> Self {
        let worker = WorkerBuilder::new("backup-worker").create();
        let runner = BackupRunner::new(config, pd_client, backup_interval);
        let name = "backup-worker";
        let scheduler = if runner.periodic_backup_enabled() {
            worker.start_with_timer(name, runner)
        } else {
            worker.start(name, runner)
        };
        Self { worker, scheduler }
    }

    pub fn stop(&self) {
        self.scheduler.stop();
        self.worker.stop();
    }

    pub async fn instant_backup(&self) -> Result<Arc<IncrementalBackupFile>> {
        let (cb, fut) = tikv_util::future::paired_future_callback();
        self.scheduler
            .schedule(BackupTask::LightweightBackup { cb })
            .unwrap();
        fut.await.unwrap().map_err(|e| {
            warn!("instant backup failed"; "error" => ?e);
            e.into()
        })
    }

    pub async fn instant_backup_with_retry(
        &self,
        timeout: Duration,
    ) -> Result<Arc<IncrementalBackupFile>> {
        // TODO: retry only when the error is retryable
        try_wait_result_async(
            || Box::pin(self.instant_backup()),
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
    last_backup_meta: Option<ClusterBackupMeta>,
    last_backup_time: Instant,
}

impl BackupRunner {
    fn new(config: BackupConfig, pd_client: Arc<dyn PdClient>, backup_interval: Duration) -> Self {
        Self {
            config,
            pd_client,
            backup_interval,
            last_backup_meta: None,
            last_backup_time: Instant::now() - MIN_BACKUP_INTERVAL,
        }
    }

    fn handle_periodic_backup(&mut self) {
        debug_assert!(self.periodic_backup_enabled());
        if self.last_backup_time.saturating_elapsed() < MIN_BACKUP_INTERVAL {
            return;
        }

        match self.do_lightweight_backup_inner() {
            Ok(backup_path) => {
                info!("periodic backup succeeded"; "backup" => ?backup_path);
            }
            Err(err) => {
                error!("periodic backup failed: {:?}", err);
            }
        }
    }

    fn do_lightweight_backup(&mut self) -> Result<Arc<IncrementalBackupFile>> {
        if self.last_backup_time.saturating_elapsed() < MIN_BACKUP_INTERVAL {
            // The backup is named according to the seconds of TSO physical time.
            // So sleep for more than 1 second to avoid the name conflict.
            thread::sleep(MIN_BACKUP_INTERVAL);
        }

        self.do_lightweight_backup_inner()
    }

    fn do_lightweight_backup_inner(&mut self) -> Result<Arc<IncrementalBackupFile>> {
        let (backup_path, backup_meta) = backup::backup_cluster(
            self.config.clone(),
            BackupType::Lightweight,
            "".to_string(),
            self.pd_client.as_ref(),
            self.last_backup_meta.take(),
        )
        .map_err(|err| {
            NATIVE_BR_BACKUP_ERROR.inc();
            err
        })?;
        info!("backup succeeded"; "path" => ?backup_path, "meta" => %backup_meta);
        self.last_backup_time = Instant::now();
        if self.periodic_backup_enabled() {
            self.last_backup_meta = Some(backup_meta);
        }
        NATIVE_BR_BACKUP_SUCCESS.inc();
        Ok(Arc::new(
            IncrementalBackupFile::try_from_full_path(&backup_path).unwrap(),
        ))
    }

    fn periodic_backup_enabled(&self) -> bool {
        !self.backup_interval.is_zero()
    }
}

impl Runnable for BackupRunner {
    type Task = BackupTask;

    fn run(&mut self, task: BackupTask) {
        match task {
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
        self.handle_periodic_backup();
    }

    fn get_interval(&self) -> Duration {
        let mut interval = self.backup_interval.as_secs();
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default();
        interval -= now.as_secs() % interval;
        Duration::from_secs(interval)
    }
}
