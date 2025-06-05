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
    worker::{LazyWorker, Runnable, RunnableWithTimer, Scheduler},
};

use crate::{
    backup,
    backup::{
        update_service_safe_point, BackupConfig, BackupType, IncrementalBackupFile, Result,
        SharedResult,
    },
    error::SharedError,
    metrics::{NATIVE_BR_BACKUP_ERROR, NATIVE_BR_BACKUP_SUCCESS},
};

pub const DEFAULT_TIMEOUT_INSTANT_BACKUP: ReadableDuration = ReadableDuration::secs(60);
const MIN_BACKUP_INTERVAL: Duration = Duration::from_millis(1050);

type InstantBackupCallback = Box<dyn FnOnce(SharedResult<Arc<IncrementalBackupFile>>) + Send>;

enum BackupTask {
    LightweightBackup {
        cb: InstantBackupCallback,
    },
    BackupResult {
        res: Result<(String, ClusterBackupMeta)>,
        cb: Option<InstantBackupCallback>,
    },
}

impl fmt::Display for BackupTask {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BackupTask::LightweightBackup { .. } => write!(f, "lightweight backup"),
            BackupTask::BackupResult { .. } => write!(f, "backup result"),
        }
    }
}

pub struct BackupWorker {
    worker: LazyWorker<BackupTask>,
    scheduler: Scheduler<BackupTask>,
}

impl BackupWorker {
    pub fn new(
        config: BackupConfig,
        pd_client: Arc<dyn PdClient>,
        backup_interval: Duration,
    ) -> Self {
        let mut worker = LazyWorker::new("backup-worker");
        let scheduler = worker.scheduler();
        let runner = BackupRunner::new(config, pd_client, backup_interval, scheduler.clone());
        let ok = if runner.periodic_backup_enabled() {
            worker.start_with_timer(runner)
        } else {
            worker.start(runner)
        };
        assert!(ok);
        Self { worker, scheduler }
    }

    pub fn stop(&mut self) {
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
    last_backup_time: Instant,

    last_backup_ts: u64,
    last_backup_meta: Option<ClusterBackupMeta>,

    scheduler: Scheduler<BackupTask>,
}

impl BackupRunner {
    fn new(
        config: BackupConfig,
        pd_client: Arc<dyn PdClient>,
        backup_interval: Duration,
        scheduler: Scheduler<BackupTask>,
    ) -> Self {
        Self {
            config,
            pd_client,
            backup_interval,
            last_backup_time: Instant::now() - MIN_BACKUP_INTERVAL,
            last_backup_ts: 0,
            last_backup_meta: None,
            scheduler,
        }
    }

    fn handle_periodic_backup(&mut self) {
        debug_assert!(self.periodic_backup_enabled());
        if self.last_backup_time.saturating_elapsed() < MIN_BACKUP_INTERVAL {
            return;
        }

        self.do_lightweight_backup_inner(None);
    }

    fn do_lightweight_backup(&mut self, cb: InstantBackupCallback) {
        if self.last_backup_time.saturating_elapsed() < MIN_BACKUP_INTERVAL {
            // The backup is named according to the seconds of TSO physical time.
            // So sleep for more than 1 second to avoid the name conflict.
            thread::sleep(MIN_BACKUP_INTERVAL);
        }

        self.do_lightweight_backup_inner(Some(cb));
    }

    fn prepare_backup(&self) -> Result<u64 /* backup_ts */> {
        backup::get_backup_ts(self.pd_client.as_ref()).map_err(Into::into)
    }

    fn do_lightweight_backup_inner(&mut self, cb: Option<InstantBackupCallback>) {
        let backup_ts = match self.prepare_backup() {
            Ok(backup_ts) => backup_ts,
            Err(err) => {
                error!("backup worker: prepare backup failed"; "err" => ?err);
                NATIVE_BR_BACKUP_ERROR.inc();
                if let Some(cb) = cb {
                    cb(Err(SharedError::from(err)));
                }
                return;
            }
        };

        self.last_backup_time = Instant::now();

        let config = self.config.clone();
        let pd_client = self.pd_client.clone();
        let last_backup_meta = self.last_backup_meta.clone();
        let scheduler = self.scheduler.clone();
        thread::spawn(move || {
            info!("backup worker: start backup"; "backup_ts" => backup_ts, "delay" => ?config.backup_delay);
            thread::sleep(config.backup_delay.0);
            let res = backup::backup_cluster_with_ts(
                config,
                BackupType::Lightweight,
                "".to_string(),
                pd_client.as_ref(),
                backup_ts,
                last_backup_meta,
            );
            scheduler.schedule(BackupTask::BackupResult { res, cb })
        });
    }

    fn handle_backup_result(
        &mut self,
        res: Result<(String, ClusterBackupMeta)>,
        cb: Option<InstantBackupCallback>,
    ) {
        match res {
            Ok((backup_path, backup_meta)) => {
                info!("backup succeeded"; "path" => ?backup_path, "meta" => %backup_meta);
                if self.last_backup_ts < backup_meta.backup_ts {
                    self.last_backup_ts = backup_meta.backup_ts;
                    if self.periodic_backup_enabled() {
                        self.last_backup_meta = Some(backup_meta);

                        if let Err(err) =
                            update_service_safe_point(self.pd_client.as_ref(), self.last_backup_ts)
                        {
                            error!("backup worker: update safepoint failed"; "err" => ?err);
                            NATIVE_BR_BACKUP_ERROR.inc();
                        }
                    }
                }

                NATIVE_BR_BACKUP_SUCCESS.inc();
                if let Some(cb) = cb {
                    cb(Ok(Arc::new(
                        IncrementalBackupFile::try_from_full_path(&backup_path).unwrap(),
                    )))
                }
            }
            Err(err) => {
                NATIVE_BR_BACKUP_ERROR.inc();
                if let Some(cb) = cb {
                    cb(Err(SharedError::from(err)));
                }
            }
        }
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
                self.do_lightweight_backup(cb);
            }
            BackupTask::BackupResult { res, cb } => {
                self.handle_backup_result(res, cb);
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
