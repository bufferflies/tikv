// Copyright 2024 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    collections::{HashMap, VecDeque},
    fmt, fs,
    io::{Read, Seek, SeekFrom},
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering},
        Arc,
    },
    thread::JoinHandle,
};

use arc_swap::ArcSwap;
use bytes::Bytes;
use file_system::IoRateLimiter;
use rfenginepb::StoreBackupMeta;
use slog_global::{error, info};
use tikv_util::{
    mpsc::{Receiver, Sender},
    sys::thread::StdThreadBuildWrapper,
    time::Instant,
    DFS_WORKER_THREAD_NAME,
};

use crate::{
    compact_worker::{backup_callback, wal_file_name, CompactTask, CompactWorker, WorkerHandle},
    dfs_worker::{Healthy, LightweightBackupConfig, ObjectStorageTask, ObjectStorageWorker},
    log_batch::RaftLogBlock,
    manifest::{Manifest, PeerFile},
    metrics::ENGINE_COMPACT_CACHE_WAL_SKIPPED_COUNTER,
    write_batch::WriteBatch,
    writer::WalWriter,
    BackupTask, Error,
};

// the maximum number of WriteBatch cached in memory for compact.
// We limit this number when background compact is slow to avoid
// consuming too much memory.
// TODO: maybe we should better control the memory usage based on
// the process memory usage.
const MAX_PENDING_COMPACT_WB_NUM: usize = 4;

pub(crate) struct ObjectStorageWorkerHandle {
    task_sender: Sender<ObjectStorageTask>,
    handle: JoinHandle<()>,
}

pub(crate) enum ServiceTask {
    Dump {
        epoch_id: u32,
        start_off: u64,
        end_off: u64,
        callback: Box<dyn FnOnce(crate::Result<(Bytes, bool)>) + Send>,
    },
    Rotate {
        epoch_id: u32,
        // Whether to cache new WriteBatch in memory for compaction.
        cache_wb_for_compact: bool,
    },
    Write {
        wb: crate::write_batch::WriteBatch,
    },
    Backup(BackupTask),
    Truncates(Vec<Vec<RaftLogBlock>>),
    Upload,
    Close {
        force: bool,
    },
}

impl fmt::Debug for ServiceTask {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ServiceTask::Dump { .. } => write!(f, "ServiceTask::Dump"),
            ServiceTask::Rotate { .. } => write!(f, "ServiceTask::Rotate"),
            ServiceTask::Write { .. } => write!(f, "ServiceTask::Write"),
            ServiceTask::Backup(_) => write!(f, "ServiceTask::Backup"),
            ServiceTask::Truncates(_) => write!(f, "ServiceTask::Truncates"),
            ServiceTask::Upload => write!(f, "ServiceTask::Upload"),
            ServiceTask::Close { force } => write!(f, "ServiceTask::Close({})", force),
        }
    }
}

const ASYNC_WRITER_SYNC_SIZE: u64 = 256 * 1024;

/// Service worker maintains the async WAL files, provide read service for HTTP
/// API, compaction and backup tasks.
pub(crate) struct ServiceWorker {
    engine_id: Arc<AtomicU64>,
    async_wal_writer: Option<WalWriter>,
    compact_wb: Option<WriteBatch>,
    // number of pending WriteBatch cached for compaction.
    pending_compact_wb_count: Arc<AtomicUsize>,
    bytes_since_last_sync: u64,
    rx: Receiver<ServiceTask>,
    compact_worker_handle: WorkerHandle,
    dfs_worker_handle: Option<ObjectStorageWorkerHandle>,
    dfs_worker_healthy: Healthy,
}

impl ServiceWorker {
    pub(crate) fn new(
        dir: PathBuf,
        epoch_id: u32,
        async_wal_writer: Option<WalWriter>,
        rx: Receiver<ServiceTask>,
        manifest: Manifest,
        compacted_epoch: Arc<AtomicU32>,
        lightweight_backup_config: Option<LightweightBackupConfig>,
        healthy: Healthy,
        compact_rate_limiter: Arc<IoRateLimiter>,
        compact_wal_sync_concurrency: usize,
        peer_rlog_files: Arc<ArcSwap<HashMap<u64, VecDeque<PeerFile>>>>,
    ) -> Self {
        let s3fs = lightweight_backup_config.as_ref().map(|cfg| {
            let s3fs = kvengine::dfs::S3Fs::new_from_config(cfg.dfs_config.clone());
            Arc::new(s3fs)
        });
        let engine_id = manifest.engine_id.clone();
        let pending_compact_wb_count = Arc::new(AtomicUsize::default());
        let (compact_worker_tx, compact_rx) = tikv_util::mpsc::unbounded();
        let mut compact_worker = CompactWorker::new(
            dir.clone(),
            compact_rx,
            manifest,
            compacted_epoch.clone(),
            lightweight_backup_config.as_ref(),
            s3fs.clone(),
            healthy.clone(),
            compact_rate_limiter,
            compact_wal_sync_concurrency,
            pending_compact_wb_count.clone(),
            peer_rlog_files,
        );
        let handle = std::thread::Builder::new()
            .name("compact-wal-worker".to_string())
            .spawn_wrapper(move || compact_worker.run())
            .unwrap();
        let compact_worker_handle = WorkerHandle {
            task_sender: compact_worker_tx.clone(),
            handle: Some(handle),
        };

        let dfs_worker_handle = if let Some(cfg) = lightweight_backup_config {
            let (dfs_worker_tx, dfs_worker_rx) = tikv_util::mpsc::unbounded();
            let mut dfs_worker = ObjectStorageWorker::new(
                cfg,
                epoch_id,
                engine_id.clone(),
                healthy.clone(),
                dfs_worker_rx,
                compact_worker_tx,
            );
            let handle = std::thread::Builder::new()
                .name(DFS_WORKER_THREAD_NAME.to_string())
                .spawn_wrapper(move || dfs_worker.run())
                .unwrap();
            Some(ObjectStorageWorkerHandle {
                task_sender: dfs_worker_tx,
                handle,
            })
        } else {
            None
        };
        ServiceWorker {
            engine_id,
            async_wal_writer,
            compact_wb: None,
            pending_compact_wb_count,
            bytes_since_last_sync: 0,
            dfs_worker_handle,
            rx,
            compact_worker_handle,
            dfs_worker_healthy: healthy,
        }
    }

    pub(crate) fn run(&mut self) {
        while let Ok(task) = self.rx.recv() {
            match task {
                ServiceTask::Write { wb } => {
                    self.handle_write(wb);
                }
                ServiceTask::Dump {
                    epoch_id,
                    start_off,
                    end_off,
                    callback,
                } => {
                    self.handle_dump(epoch_id, start_off, end_off, callback);
                }
                ServiceTask::Backup(task) => {
                    self.handle_backup(task);
                }
                ServiceTask::Rotate {
                    epoch_id,
                    cache_wb_for_compact,
                } => {
                    self.handle_rotate(epoch_id, cache_wb_for_compact);
                }
                ServiceTask::Truncates(truncates) => drop(truncates),
                ServiceTask::Upload => {
                    self.handle_flush();
                }
                ServiceTask::Close { force } => {
                    self.handle_close(force);
                    break;
                }
            }
        }
    }

    fn is_lightweight_enabled(&self) -> bool {
        self.dfs_worker_handle.is_some()
    }

    fn get_engine_id(&self) -> u64 {
        self.engine_id.load(Ordering::SeqCst)
    }

    fn handle_write(&mut self, wb: crate::write_batch::WriteBatch) {
        if let Some(wal_writer) = &mut self.async_wal_writer {
            let old_file_off = wal_writer.file_off;
            let (epoch_id, file_off, rotated) = wal_writer.write_batch(&wb).unwrap();
            if rotated {
                // When rotated, the old file is already synced.
                self.bytes_since_last_sync = 0;
            } else {
                self.bytes_since_last_sync += file_off - old_file_off;
            }
            if self.bytes_since_last_sync > ASYNC_WRITER_SYNC_SIZE {
                wal_writer.file().sync_data().unwrap();
                self.bytes_since_last_sync = 0;
            }
            if self.is_lightweight_enabled() {
                // Send write task to object storage worker.
                let task = crate::dfs_worker::ObjectStorageTask::Sync { epoch_id, file_off };
                self.dfs_worker_handle
                    .as_ref()
                    .unwrap()
                    .task_sender
                    .send(task)
                    .unwrap();
            }
        }
        if let Some(compact_wb) = &mut self.compact_wb {
            compact_wb.merge_write_batch(wb);
        }
    }

    fn handle_dump(
        &mut self,
        epoch_id: u32,
        start_off: u64,
        end_off: u64,
        callback: Box<dyn FnOnce(crate::Result<(Bytes, bool)>) + Send>,
    ) {
        if let Some(writer) = self.async_wal_writer.as_ref() {
            let mut partial_content = false;
            let store_id = self.get_engine_id();
            // Check the WAL chunk meta is valid.
            if epoch_id > writer.epoch_id || (end_off > 0 && start_off >= end_off) {
                let msg = format!(
                    "{}: invalid dump wal chunk epoch {} start_off {} end_off {:?} writer epoch {}, file_off {}",
                    store_id, epoch_id, start_off, end_off, writer.epoch_id, writer.file_off
                );
                error!("{}", msg);
                callback(Err(crate::Error::Other(msg)));
                return;
            } else if epoch_id + 3 < writer.epoch_id {
                // The epoch_id is too old and has been overwritten.
                info!(
                    "{}: handle dump: epoch is overwritten: epoch {} writer epoch {}",
                    store_id, epoch_id, writer.epoch_id
                );
                callback(Err(crate::Error::WalEpochOverwritten { epoch_id }));
                return;
            }
            // Dump the WAL chunk from offset start_off to end_off.
            info!(
                "{}: dump latest wal epoch {} start_off {} end_off {} writer epoch {}",
                store_id, epoch_id, start_off, end_off, writer.epoch_id,
            );
            let effective_end_off = if end_off == 0 {
                if writer.epoch_id == epoch_id {
                    writer.file_off
                } else {
                    partial_content = true;
                    let wal_file_name = wal_file_name(&writer.dir, epoch_id);
                    let file_meta = fs::metadata(wal_file_name).unwrap();
                    file_meta.len()
                }
            } else {
                end_off
            };
            match dump_wal_chunk(&writer.dir, epoch_id, start_off, effective_end_off) {
                Ok(chunk) => callback(Ok((chunk, partial_content))),
                Err(err) => {
                    let msg = format!(
                        "{}: dump wal chunk epoch {} start_off {} end_off {} failed {:?}",
                        self.get_engine_id(),
                        epoch_id,
                        start_off,
                        effective_end_off,
                        err
                    );
                    error!("{}", msg);
                    callback(Err(crate::Error::Other(msg)));
                }
            }
        }
    }

    fn handle_flush(&mut self) {
        if self.is_lightweight_enabled() {
            // Send flush task to object storage worker.
            self.dfs_worker_handle
                .as_ref()
                .unwrap()
                .task_sender
                .send(ObjectStorageTask::Flush)
                .unwrap();
        }
    }

    fn handle_rotate(&mut self, epoch_id: u32, cache_wb: bool) {
        if let Some(writer) = self.async_wal_writer.as_mut() {
            debug_assert_eq!(writer.epoch_id, epoch_id);
            let file_off = writer.file_off;
            writer.rotate().unwrap();
            if let Some(dfs_worker_handle) = &self.dfs_worker_handle {
                // Send rotate task to object storage worker.
                dfs_worker_handle
                    .task_sender
                    .send(ObjectStorageTask::Rotate { epoch_id, file_off })
                    .unwrap();
            }
        }
        self.compact_worker_handle
            .task_sender
            .send(CompactTask::Compact {
                epoch_id,
                compact_wb: self.compact_wb.take(),
            })
            .unwrap();
        let can_cache =
            self.pending_compact_wb_count.load(Ordering::SeqCst) < MAX_PENDING_COMPACT_WB_NUM;
        // init async compact write batch after the rotate.
        if cache_wb {
            if can_cache {
                self.compact_wb = Some(WriteBatch::default());
                self.pending_compact_wb_count.fetch_add(1, Ordering::SeqCst);
            } else {
                ENGINE_COMPACT_CACHE_WAL_SKIPPED_COUNTER.inc();
            }
        }
    }

    fn handle_close(&mut self, force: bool) {
        // Close and join object storage thread.
        if let Some(ObjectStorageWorkerHandle {
            task_sender,
            handle,
        }) = self.dfs_worker_handle.take()
        {
            // If force close, we skip flushing wal chunk and close task thread.
            if !force {
                task_sender.send(ObjectStorageTask::Flush).unwrap();
            }
            task_sender.send(ObjectStorageTask::Close).unwrap();
            handle.join().unwrap();
        }
        self.compact_worker_handle
            .task_sender
            .send(CompactTask::Close)
            .unwrap();
        let join_handle = self.compact_worker_handle.handle.take().unwrap();
        join_handle.join().unwrap();
    }

    fn handle_backup(&mut self, mut task: BackupTask) {
        if let Some(async_writer) = self.async_wal_writer.as_ref() {
            task.file_off = async_writer.file_off;
        }
        if task.config.lightweight {
            self.lightweight_backup(task);
        } else {
            self.compact_worker_handle
                .task_sender
                .send(CompactTask::HeavyBackup(task))
                .unwrap();
        }
    }

    fn lightweight_backup(&mut self, task: BackupTask) {
        if self.async_wal_writer.is_none() {
            return backup_callback(
                task,
                Err("async wal writer is not set".to_string()),
                "light_fail",
                Instant::now(),
            );
        }
        let async_writer = self.async_wal_writer.as_ref().unwrap();

        if !self.dfs_worker_healthy.is_healthy(async_writer.epoch_id) {
            return backup_callback(
                task,
                Err("dfs worker unhealthy".to_string()),
                "light_fail",
                Instant::now(),
            );
        }

        let mut backup_meta = StoreBackupMeta::default();
        backup_meta.set_store_id(self.engine_id.load(Ordering::SeqCst));
        backup_meta.set_epoch(async_writer.epoch_id);
        backup_meta.set_offset(task.file_off);

        backup_callback(task, Ok(backup_meta), "light_success", Instant::now());
    }
}

fn dump_wal_chunk(dir: &Path, epoch_id: u32, start_off: u64, end_off: u64) -> crate::Result<Bytes> {
    // `epoch_id` already checked in the caller.
    let mut file = fs::File::open(wal_file_name(dir, epoch_id))?;
    let file_len = file.metadata()?.len();
    if end_off > file_len {
        return Err(Error::Eof);
    }

    // `start_off` < `end_off` already checked in the caller.
    file.seek(SeekFrom::Start(start_off))?;
    let dump_len = (end_off - start_off) as usize;
    let mut buf = vec![0; dump_len];
    file.read_exact(&mut buf)?;
    Ok(Bytes::from(buf))
}
