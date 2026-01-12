// Copyright 2023 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    convert::TryInto,
    fs,
    io::{Read, Seek, SeekFrom},
    os::unix::fs::FileExt,
    path::PathBuf,
    sync::{
        atomic::{AtomicI64, AtomicU32, AtomicU64, Ordering},
        Arc,
    },
    time::Instant,
};

use bytes::{Buf, BufMut, Bytes, BytesMut};
use engine_traits::ObjectStorage;
use kvengine::dfs::{DFSConfig, Dfs, S3Fs};
use slog_global::*;
use tikv_util::{
    config::ReadableDuration,
    errors::{Context as _, IoError},
    mpsc::{Receiver, Sender},
};
use tokio::task::JoinHandle;

use crate::{
    compact_worker::CompactTask,
    compress_lz4, decompress_lz4, get_integral_wal_chunks, last_wal_chunk_file_key,
    manifest::Manifest,
    metrics::{self, RFENGINE_DFS_WORKER_HEALTHY_GAUGE},
    wal_chunk_file_key, wal_chunk_file_prefix, wal_file_name,
    writer::EPOCH_ROTATE_LEN,
    Error, NotifyOnDrop, Result, WalChunkMeta,
};

/// DFS request statistics meter.
/// Used to observe the uploaded bytes and request count.
#[derive(Default)]
pub struct DfsMeter {
    uploaded_bytes: AtomicU64,
    request_count: AtomicU64,
}

/// DFS request statistics measure.
/// Captured from `DfsMeter`.
/// Used to acknowledge the uploaded bytes and request count.
pub struct DfsMeasure {
    from: Arc<DfsMeter>,
    pub uploaded_bytes: u64,
    pub request_count: u64,
}

impl Drop for DfsMeasure {
    fn drop(&mut self) {
        if !self.acknowledged() {
            warn!("recollecting a dfs measure failed to upload.";
                "uploaded_bytes" => self.uploaded_bytes,
                "request_count" => self.request_count,
            );
            self.from.recollect(self);
        }
    }
}

impl DfsMeasure {
    /// Acknowledge the measure, preventing it from being recollected.
    pub fn acknowledge(mut self) {
        use crate::metrics::*;

        RFENGINE_DFS_UPLOAD_BYTES_ACKED.inc_by(self.uploaded_bytes);
        RFENGINE_DFS_REQUESTS_ACKED.inc_by(self.request_count);
        self.uploaded_bytes = 0;
        self.request_count = 0;
    }

    fn acknowledged(&self) -> bool {
        self.uploaded_bytes == 0 && self.request_count == 0
    }
}

impl DfsMeter {
    /// Observe a DFS request with uploaded bytes.
    pub fn observe_request(&self, uploaded_bytes: u64) {
        use crate::metrics::*;

        RFENGINE_DFS_UPLOAD_BYTES.inc_by(uploaded_bytes);
        RFENGINE_DFS_REQUESTS.inc();
        self.uploaded_bytes
            .fetch_add(uploaded_bytes, Ordering::SeqCst);
        self.request_count.fetch_add(1, Ordering::SeqCst);
    }

    /// Capture the current uploaded bytes and request count.
    ///
    /// You should acknowledge the measure after uploading it to PD
    /// successfully.
    #[must_use = "If not acknowledged this measure will be recollected to the meter."]
    pub fn measure(self: &Arc<Self>) -> DfsMeasure {
        DfsMeasure {
            from: Arc::clone(self),
            uploaded_bytes: self.uploaded_bytes.swap(0, Ordering::SeqCst),
            request_count: self.request_count.swap(0, Ordering::SeqCst),
        }
    }

    fn recollect(&self, acknowledged: &DfsMeasure) {
        self.uploaded_bytes
            .fetch_add(acknowledged.uploaded_bytes, Ordering::SeqCst);
        self.request_count
            .fetch_add(acknowledged.request_count, Ordering::SeqCst);
    }
}

#[derive(Debug)]
pub(crate) struct LightweightBackupConfig {
    pub(crate) dir: PathBuf,
    pub(crate) wal_chunk_target_file_size: usize,
    pub(crate) compression_type: CompressionType,
    // for upgrade compatiblity, set rlog_compression_type to false first
    // so we need another compress configration.
    pub(crate) rlog_compression_type: CompressionType,
    pub(crate) dfs_config: DFSConfig,

    pub(crate) rlog_cache_capacity: usize,
    pub(crate) rlog_cache_size_threshold: usize,
    pub(crate) memory_limit: usize,

    pub(crate) flush_chunk_interval: Option<ReadableDuration>,
}

impl LightweightBackupConfig {
    pub(crate) fn new(
        dir: PathBuf,
        wal_chunk_target_file_size: usize,
        compression_type: CompressionType,
        rlog_compression_type: CompressionType,
        dfs_config: DFSConfig,
        rlog_cache_capacity: usize,
        rlog_cache_size_threshold: usize,
        memory_limit: usize,
        flush_chunk_interval: Option<ReadableDuration>,
    ) -> Self {
        Self {
            dir,
            wal_chunk_target_file_size,
            compression_type,
            rlog_compression_type,
            dfs_config,
            rlog_cache_capacity,
            rlog_cache_size_threshold,
            memory_limit,
            flush_chunk_interval,
        }
    }
}

struct BackgroundUpload {
    file_key: String,
    epoch_id: u32,
    join_handle: JoinHandle<()>,
}

pub(crate) struct ObjectStorageWorker {
    config: LightweightBackupConfig,
    engine_id: Arc<AtomicU64>,
    task_rx: Receiver<ObjectStorageTask>,
    compact_worker_tx: Sender<CompactTask>,
    service_worker_epoch: Arc<AtomicU32>,
    buf: Vec<u8>,
    async_wal_file: Option<fs::File>,
    epoch_id: u32,
    start_off: u64, // The start offset of the current chunk.
    sync_off: u64,  // The offset of the syncing of current wal.
    s3fs: Arc<S3Fs>,
    healthy: Healthy,
    memory_limiter: MemoryLimiter,
    last_chunk_upload_time: Instant,

    statistic: Arc<DfsMeter>,
    background_uploads: Vec<BackgroundUpload>,
}

impl ObjectStorageWorker {
    fn reset(&mut self, epoch_id: u32) {
        self.epoch_id = epoch_id;
        self.buf.clear();
        self.async_wal_file = None;
        self.start_off = 0;
        self.sync_off = 0;
    }

    pub(crate) fn new(
        config: LightweightBackupConfig,
        epoch_id: u32,
        engine_id: Arc<AtomicU64>,
        dfs_worker_healthy: Healthy,
        task_rx: Receiver<ObjectStorageTask>,
        compact_worker_tx: Sender<CompactTask>,
        service_worker_epoch: Arc<AtomicU32>,
        statistic: Arc<DfsMeter>,
    ) -> Self {
        info!("dfs worker config: {:?}", config);
        let dfs_config = config.dfs_config.clone();
        let wal_chunk_target_file_size = config.wal_chunk_target_file_size;
        let s3fs = Arc::new(S3Fs::new_from_config(dfs_config.clone()));
        let memory_limiter = MemoryLimiter::new(config.memory_limit);
        Self {
            config,
            epoch_id,
            engine_id,
            task_rx,
            compact_worker_tx,
            service_worker_epoch,
            buf: Vec::with_capacity(wal_chunk_target_file_size),
            async_wal_file: None,
            start_off: 0,
            sync_off: 0,
            s3fs,
            healthy: dfs_worker_healthy,
            memory_limiter,
            statistic,
            last_chunk_upload_time: Instant::now(),
            background_uploads: vec![],
        }
    }

    // `init` will rebuild the last wal chunk persistence states. If no wal chunk
    // found in the epoch range from `epoch_id - 3` to `epoch_id`, trigger an
    // instant rfengine snapshot.
    pub(crate) fn init(&mut self) -> Result<bool> {
        self.healthy.set_healthy();
        let mut need_snapshot = false;
        // Wait for node bootstrapped.
        info!("dfs worker wait for store bootstrapped.");
        let store_id = self.wait_for_bootstrapped();
        debug_assert!(store_id > 0);
        info!("{}: dfs worker start init.", store_id);
        let mut rebuild_epoch = self.epoch_id;
        let store_id = self.get_engine_id();
        let mut last_chunk = None;
        loop {
            let scan_prefix = wal_chunk_file_prefix(store_id, rebuild_epoch);
            info!(
                "{}: rebuild last wal chunk list chunks with prefix {}",
                store_id, scan_prefix
            );

            // Chunks in an epoch should be listed in one iterate.
            let (chunks, has_more) = self.s3fs.list_objects("", Some(&scan_prefix), None)?;
            debug_assert_eq!(has_more, None);
            if !chunks.is_empty() {
                let chunk_metas: Vec<WalChunkMeta> = chunks
                    .into_iter()
                    .filter_map(|x| {
                        x.key
                            .try_into()
                            .map_err(|e| warn!("skip invalid WAL chunk: {:?}", e))
                            .ok()
                    })
                    .collect();
                if let Ok((mut integral_chunks, ..)) = get_integral_wal_chunks(&chunk_metas) {
                    last_chunk = integral_chunks.pop();
                    if last_chunk.is_some() {
                        info!("{}: found integral wal chunks", store_id;
                            "integral_chunks" => ?integral_chunks, "last_chunk" => ?last_chunk);
                        break;
                    }
                }
            }

            if rebuild_epoch <= 1 || rebuild_epoch <= self.near_overwritten_epoch() {
                need_snapshot = true;
                break;
            }
            rebuild_epoch -= 1;
        }
        match last_chunk {
            None => {
                // Rebuild from the earliest epoch.
                self.epoch_id = rebuild_epoch;
                self.sync_off = 0;
                self.start_off = 0;
                info!(
                    "{}: no wal chunk found, rebuild from epoch {}",
                    store_id, rebuild_epoch
                );
            }
            Some(chunk) => {
                self.epoch_id = chunk.epoch;
                self.start_off = chunk.end_off;
                self.sync_off = chunk.end_off;
                info!(
                    "{}: found the last wal chunk {} rebuild from epoch {} offset {}",
                    store_id, chunk.key, chunk.epoch, chunk.end_off
                );
            }
        }

        Ok(need_snapshot)
    }

    fn wait_for_bootstrapped(&self) -> u64 {
        let mut engine_id = self.get_engine_id();
        while engine_id == 0 {
            std::thread::sleep(std::time::Duration::from_millis(100));
            engine_id = self.get_engine_id();
        }
        engine_id
    }

    pub(crate) fn run(&mut self) {
        match self.init() {
            Ok(need_snapshot) => {
                if need_snapshot {
                    // Send task to compact worker to trigger a snapshot.
                    self.compact_worker_tx.send(CompactTask::Snapshot).unwrap();
                }
            }
            Err(err) => {
                // Disable lightweight backup if init failed.
                error!("dfs worker init failed, set unhealthy"; "err" => ?err);
                self.healthy.set_unhealthy(self.epoch_id, "init");
            }
        }
        while let Ok(task) = self.task_rx.recv() {
            if let ObjectStorageTask::Close = task {
                info!("ObjectStorageWorker close");
                return;
            }

            // If dfs worker is unhealthy, skip handle some tasks and downgrade to disable
            // lightweight backup.
            // Try to recover when receive snapshot task.
            if !self
                .healthy
                .is_healthy(task.epoch_id().unwrap_or(self.epoch_id))
            {
                if let ObjectStorageTask::Rotate { epoch_id, .. } = task {
                    // Reset to the new epoch. Otherwise, `handle_sync` will sync from previous
                    // unhealthy epoch.
                    self.reset(epoch_id + 1);
                }
                continue;
            }
            match task {
                ObjectStorageTask::Sync { epoch_id, file_off } => {
                    if let Err(err) = self.handle_sync(epoch_id, file_off) {
                        error!("dfs worker handle_sync failed, set unhealthy"; "err" => ?err);
                        self.healthy.set_unhealthy(epoch_id, "handle sync")
                    }
                }
                ObjectStorageTask::Rotate { epoch_id, file_off } => {
                    if self.need_sync(epoch_id, file_off) {
                        info!("{} dfs worker need sync before rotate", self.get_engine_id();
                            "current_epoch" => self.epoch_id, "current_sync_off" => self.sync_off,
                            "epoch_id" => epoch_id, "file_off" => file_off,
                        );
                        if let Err(err) = self.handle_sync(epoch_id, file_off) {
                            error!("dfs worker handle_sync failed, set unhealthy"; "err" => ?err);
                            self.healthy.set_unhealthy(epoch_id, "handle sync");
                            return;
                        }
                    }
                    if let Err(err) = self.handle_rotate(epoch_id) {
                        error!("dfs worker handle_rotate failed, set unhealthy"; "err" => ?err);
                        self.healthy.set_unhealthy(epoch_id, "handle rotate");
                    }
                }
                ObjectStorageTask::Flush(notify) => {
                    // Send flush task before close in normal case. If close without flush, we can
                    // construct the case for wal chunk recovery in random test.
                    if !self.buf.is_empty() && self.next_chunk(false).is_err() {
                        self.healthy.set_unhealthy(self.epoch_id, "handle flush");
                    }
                    self.wait_uploads();
                    drop(notify);
                }
                ObjectStorageTask::Close => unreachable!(),
            }
        }
    }

    // Write wal chunk from `start_off` to end of current epoch in a single write.
    fn rebuild_wal_chunk(&mut self) -> Result<()> {
        let store_id = self.get_engine_id();
        let mut fd = fs::File::open(wal_file_name(self.config.dir.as_path(), self.epoch_id))?;
        if self.start_off > 0 {
            fd.seek(SeekFrom::Start(self.start_off))?;
        }
        let sync_len = fd
            .read_to_end(&mut self.buf)
            .ctx("rebuild_wal_chunk_read_wal")?;
        self.check_overwritten_epoch("rebuild_wal_chunk")?;

        self.sync_off = self.start_off + sync_len as u64;
        info!("{}: rebuild_wal_chunk", store_id; "epoch" => self.epoch_id,
            "start_off" => self.start_off, "sync_off" => self.sync_off);
        let file_key =
            last_wal_chunk_file_key(store_id, self.epoch_id, self.start_off, self.sync_off);
        let chunk = self.take_chunk_data()?;

        let fs = self.s3fs.clone();
        let healthy = self.healthy.clone();
        let acquired = self.memory_limiter.acquire(chunk.len())?;
        let epoch_id = self.epoch_id;
        let stat = Arc::clone(&self.statistic);
        self.s3fs.get_runtime().spawn_blocking(move || {
            let length = chunk.len();
            if let Err(err) = fs.put_objects(vec![(file_key, Bytes::from(chunk))]) {
                error!("{} put wal chunk failed", store_id; "err" => ?err);
                healthy.set_unhealthy(epoch_id, "put wal chunk");
            }
            stat.observe_request(length as u64);
            drop(acquired);
        });

        Ok(())
    }

    fn need_sync(&self, epoch_id: u32, file_off: u64) -> bool {
        self.epoch_id < epoch_id || (self.epoch_id == epoch_id && self.sync_off < file_off)
    }

    fn handle_sync(&mut self, epoch_id: u32, file_off: u64) -> Result<()> {
        let store_id = self.get_engine_id();

        if (epoch_id, file_off) < (self.epoch_id, self.sync_off) {
            // Happens when in-place restore TiKV store.
            // It's not safe to overwrite the existed remote WAL chunks. Return error and
            // set unhealthy, then wait for next snapshot to become healthy.
            error!("{}: handle_sync for early data", store_id;
                "epoch_id" => epoch_id, "file_off" => file_off,
                "self.epoch_id" => self.epoch_id, "start_off" => self.start_off, "sync_off" => self.sync_off,
            );
            debug_assert!(false);
            return Err(Error::Other("handle_sync for early data".to_string()));
        }

        if epoch_id != self.epoch_id {
            // If epoch is overwritten, the WAL chunks in DFS must be incomplete.
            // Return error to make unhealthy.
            self.check_overwritten_epoch("handle_sync")?;

            // Need to rebuild all previous epoch.
            for rebuild_epoch in self.epoch_id..epoch_id {
                self.rebuild_wal_chunk()?;
                self.reset(rebuild_epoch + 1);
            }
        }
        assert_eq!(epoch_id, self.epoch_id);
        assert!(file_off >= self.sync_off);
        if file_off == self.sync_off {
            // Should not happen, but not a fatal error. Allow for safety.
            warn!("{}: handle_sync: no new data", store_id;
                "epoch_id" => epoch_id, "file_off" => file_off,
                "self.epoch_id" => self.epoch_id, "start_off" => self.start_off, "sync_off" => self.sync_off,
            );
            debug_assert!(false);
        }

        let sync_len = file_off - self.sync_off;

        if self.should_chunk(sync_len as usize) {
            if let Some(async_wal_file) = &self.async_wal_file {
                // sync data before write to S3, to avoid S3 file ahead of async local file
                // after restart.
                async_wal_file.sync_data()?;
            }
            info!(
                "{}: handle_sync put wal epoch {} start_off {} sync_off {}",
                store_id, epoch_id, self.start_off, self.sync_off
            );
            self.next_chunk(false)?;
        }

        // Sync WAL of `epoch_id` from `self.sync_off` to file_off
        let async_wal_file = match self.async_wal_file {
            Some(ref mut fd) => fd,
            None => {
                let filename = wal_file_name(self.config.dir.as_path(), epoch_id);
                let fd = fs::File::open(filename)?;
                self.async_wal_file = Some(fd);
                self.async_wal_file.as_mut().unwrap()
            }
        };

        let buf_start = self.buf.len();
        let buf_end = buf_start + sync_len as usize;
        debug!(
            "{}: handle_sync epoch {} from {} to {} buf_start {} buf_end {}",
            store_id, epoch_id, self.sync_off, file_off, buf_start, buf_end
        );
        self.buf.resize(buf_end, 0);
        async_wal_file
            .read_exact_at(&mut self.buf[buf_start..buf_end], self.sync_off)
            .ctx("handle_sync_read_async_wal")?;
        self.check_overwritten_epoch("handle_sync")?;

        // Update the sync offset.
        self.sync_off = file_off;
        Ok(())
    }

    // The epoch <= `overwritten_epoch` is overwritten and should not read.
    #[inline]
    fn overwritten_epoch(&self) -> u32 {
        self.service_worker_epoch
            .load(Ordering::SeqCst)
            .saturating_sub(EPOCH_ROTATE_LEN)
    }

    // `service_worker_epoch - 3` (`EPOCH_ROTATE_LEN - 1` == 3) is the cut-off value
    // and would be overwritten soon.
    // So `service_worker_epoch - 2` is used.
    #[inline]
    fn near_overwritten_epoch(&self) -> u32 {
        self.service_worker_epoch
            .load(Ordering::SeqCst)
            .saturating_sub(EPOCH_ROTATE_LEN - 2)
    }

    fn check_overwritten_epoch(&self, ctx: &str) -> Result<()> {
        let overwritten_epoch = self.overwritten_epoch();
        if self.epoch_id <= overwritten_epoch {
            error!("{}: {}: epoch is overwritten", self.get_engine_id(), ctx;
                    "dfs_worker.epoch" => self.epoch_id, "overwritten_epoch" => overwritten_epoch);
            Err(Error::Other(format!("{ctx}: epoch is overwritten")))
        } else {
            Ok(())
        }
    }

    fn handle_rotate(&mut self, epoch_id: u32) -> Result<()> {
        debug!("{}: handle_rotate epoch {}", self.get_engine_id(), epoch_id);
        // Call next_chunk even self.buf is empty. This can cover the case the last
        // chunk flushed during stop with no `.last` suffix.
        let res = self.next_chunk(true);
        self.reset(epoch_id + 1);
        res
    }

    pub(crate) fn should_chunk(&mut self, to_read: usize) -> bool {
        if !self.buf.is_empty()
            && self.buf.len() + ChunkHeader::len() + to_read
                > self.config.wal_chunk_target_file_size
        {
            return true;
        }
        if let Some(max_dur) = self.config.flush_chunk_interval
            && !self.buf.is_empty()
            && self.last_chunk_upload_time.elapsed() > max_dur.0
        {
            return true;
        }
        false
    }

    #[cfg(test)]
    pub(crate) fn set_buf(&mut self, buf: Vec<u8>) {
        self.buf = buf;
        self.sync_off = self.start_off + self.buf.len() as u64;
    }

    pub(crate) fn take_chunk_data(&mut self) -> Result<Vec<u8>> {
        debug_assert_eq!(self.buf.len() as u64, self.sync_off - self.start_off);

        let chunk_header = ChunkHeader::new(self.compression_type());
        let buf_len = self.buf.len();
        let mut chunk = Vec::with_capacity(ChunkHeader::len() + buf_len);
        chunk_header.encode_to(&mut chunk);
        // Get slice of the buffer but keep the memory.
        let res = if self.need_compression() {
            compress_lz4(&self.buf, &mut chunk).map(|_| chunk).map_err(|err| {
                error!("{} take chunk data: compress_lz4 failed", self.get_engine_id(); "err" => ?err);
                Error::Io(IoError::new(err, "compress_chunk".to_string()))
            })
        } else {
            chunk.extend_from_slice(&self.buf);
            Ok(chunk)
        };

        // Always clear the buf. Otherwise, when error occurs, chunks of different epoch
        // will be combined.
        self.buf.clear();
        self.start_off = self.sync_off;
        res
    }

    fn wait_an_upload(&self, upload: BackgroundUpload) {
        match self.s3fs.get_runtime().block_on(upload.join_handle) {
            Ok(()) => info!("noticed an upload has finished."; "tag" => upload.file_key),
            Err(err) => {
                let msg = format!(
                    "wait_a_upload: background task exits abnormally: {}",
                    upload.file_key
                );
                error!("wait_a_upload failed"; "err" => ?err, "tag" => upload.file_key);
                self.healthy.set_unhealthy(upload.epoch_id, &msg);
            }
        }
    }

    /// Remove finished background uploads.
    /// Won't block on unfinished uploads.
    fn gc_finished_uploads(&mut self) {
        let finished = self
            .background_uploads
            .extract_if(|upload| upload.join_handle.is_finished())
            .collect::<Vec<_>>();

        finished.into_iter().for_each(|v| self.wait_an_upload(v));
    }

    /// Wait all background uploads to finish.
    fn wait_uploads(&mut self) {
        std::mem::take(&mut self.background_uploads)
            .into_iter()
            .for_each(|upload| self.wait_an_upload(upload));
    }

    fn next_chunk(&mut self, rotate: bool) -> Result<()> {
        self.last_chunk_upload_time = Instant::now();

        let store_id = self.get_engine_id();
        let file_key = if rotate {
            last_wal_chunk_file_key(store_id, self.epoch_id, self.start_off, self.sync_off)
        } else {
            wal_chunk_file_key(store_id, self.epoch_id, self.start_off, self.sync_off)
        };
        let buf_len = self.buf.len();
        let chunk = self.take_chunk_data()?;
        info!(
            "{}: put wal chunk {} len {} compress len {}",
            store_id,
            file_key,
            ChunkHeader::len() + buf_len,
            chunk.len()
        );
        let fs = self.s3fs.clone();
        let healthy = self.healthy.clone();
        let acquired = self.memory_limiter.acquire(chunk.len())?;
        let epoch_id = self.epoch_id;
        let stat = Arc::clone(&self.statistic);
        metrics::RFENGINE_DFS_RUNNING_UPLOADS.inc();
        let handle = {
            let file_key = file_key.clone();
            self.s3fs.get_runtime().spawn_blocking(move || {
                let length = chunk.len();
                if let Err(err) = fs.put_objects(vec![(file_key, Bytes::from(chunk))]) {
                    error!("{} put wal chunk failed", store_id, ; "err" => ?err);
                    healthy.set_unhealthy(epoch_id, "put wal chunk");
                }
                stat.observe_request(length as u64);
                metrics::RFENGINE_DFS_RUNNING_UPLOADS.dec();
                drop(acquired);
            })
        };

        let bg_upload = BackgroundUpload {
            file_key,
            epoch_id,
            join_handle: handle,
        };
        self.background_uploads.push(bg_upload);
        self.gc_finished_uploads();
        Ok(())
    }

    fn need_compression(&self) -> bool {
        self.config.compression_type != CompressionType::NoCompression
    }

    fn compression_type(&self) -> CompressionType {
        self.config.compression_type
    }

    fn get_engine_id(&self) -> u64 {
        self.engine_id.load(Ordering::Acquire)
    }
}

// More data will be appended to epoch_wal after call this, so return BytesMut.
pub fn assemble_wal_chunks(chunks: Vec<Bytes>) -> Result<BytesMut> {
    let mut epoch_wal = BytesMut::new();
    for chunk in chunks.into_iter() {
        epoch_wal.put(decompress_wal_chunk(chunk)?);
    }
    Ok(epoch_wal)
}

pub fn decompress_wal_chunk(chunk: Bytes) -> Result<Bytes> {
    // Read chunk header.
    let header = ChunkHeader::decode(chunk.slice(0..ChunkHeader::len()).chunk())?;
    let decompressed_data = match header.compression_type {
        CompressionType::Lz4Compression => {
            Bytes::from(decompress_lz4(chunk.slice(ChunkHeader::len()..).chunk())?)
        }
        CompressionType::NoCompression => chunk.slice(ChunkHeader::len()..),
    };
    Ok(decompressed_data)
}

#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(u32)]
pub enum CompressionType {
    NoCompression,
    Lz4Compression,
    // Add more compression types here.
}

impl CompressionType {
    pub fn from(v: u32) -> CompressionType {
        match v {
            0 => CompressionType::NoCompression,
            1 => CompressionType::Lz4Compression,
            _ => panic!("unknown compression type"),
        }
    }

    pub fn to(&self) -> u32 {
        match self {
            CompressionType::NoCompression => 0,
            CompressionType::Lz4Compression => 1,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(u32)]
enum ChunkVersion {
    V1 = 1,
}

impl From<u32> for ChunkVersion {
    fn from(v: u32) -> ChunkVersion {
        match v {
            1 => ChunkVersion::V1,
            _ => panic!("unknown chunk version"),
        }
    }
}

#[derive(Debug, PartialEq)]
pub struct ChunkHeader {
    version: ChunkVersion,
    pub compression_type: CompressionType,
}

impl ChunkHeader {
    pub fn new(compression_type: CompressionType) -> Self {
        Self {
            version: ChunkVersion::V1,
            compression_type,
        }
    }

    pub const fn len() -> usize {
        8
    }

    pub fn encode_to(&self, buf: &mut Vec<u8>) {
        buf.put_u32_le(self.version as u32);
        buf.put_u32_le(self.compression_type as u32);
    }

    pub fn decode(mut buf: &[u8]) -> Result<Self> {
        if buf.len() < Self::len() {
            return Err(Error::Corruption {
                msg: format!("chunk header mismatch: len {}", buf.len()),
                epoch_id: 0,
                offset: 0,
                data: buf.to_vec(),
            });
        }

        let version = ChunkVersion::from(buf.get_u32_le());
        if version != ChunkVersion::V1 {
            return Err(Error::Corruption {
                msg: format!("chunk version mismatch: version {:?}", version),
                epoch_id: 0,
                offset: 0,
                data: buf.to_vec(),
            });
        }
        let compression_type = CompressionType::from(buf.get_u32_le());
        Ok(Self {
            version,
            compression_type,
        })
    }
}

pub(crate) enum ObjectStorageTask {
    Sync { epoch_id: u32, file_off: u64 }, // Sync the `epoch_id` wal file to `file_off`.
    Rotate { epoch_id: u32, file_off: u64 }, // Rotate to next epoch.
    Flush(NotifyOnDrop),                   // Trigger flush the last chunk, mainly for test.
    Close,
}

impl ObjectStorageTask {
    fn epoch_id(&self) -> Option<u32> {
        match self {
            Self::Sync { epoch_id, .. } | Self::Rotate { epoch_id, .. } => Some(*epoch_id),
            Self::Flush(_) | Self::Close => None,
        }
    }
}

#[derive(Clone)]
pub(crate) struct Healthy(
    Arc<AtomicU32>, // The epoch id since which DFS worker is healthy.
);

impl Default for Healthy {
    fn default() -> Self {
        Self(Arc::new(AtomicU32::new(0)))
    }
}

impl Healthy {
    pub(crate) fn set_healthy(&self) {
        RFENGINE_DFS_WORKER_HEALTHY_GAUGE.set(1);
        self.0.store(0, Ordering::Release);
    }

    pub(crate) fn set_unhealthy(&self, current_epoch: u32, ctx: &str) {
        RFENGINE_DFS_WORKER_HEALTHY_GAUGE.set(0);
        let next_snapshot_epoch = Manifest::next_snapshot_epoch(current_epoch);
        self.0.fetch_max(next_snapshot_epoch, Ordering::Release);
        warn!("dfs worker unhealthy"; "ctx" => ctx,
            "current_epoch" => current_epoch, "next_snapshot" => next_snapshot_epoch);
    }

    pub(crate) fn is_healthy(&self, current_epoch: u32) -> bool {
        current_epoch >= self.0.load(Ordering::Acquire)
    }

    pub(crate) fn check_healthy(&self, current_epoch: u32) -> bool {
        let ok = self.is_healthy(current_epoch);
        if ok {
            info!("dfs worker become healthy"; "epoch" => current_epoch);
            RFENGINE_DFS_WORKER_HEALTHY_GAUGE.set(1);
        }
        ok
    }
}

#[derive(Clone)]
struct MemoryLimiter {
    available: Arc<AtomicI64>, // Use i64 to avoid overflow.
}

impl MemoryLimiter {
    fn new(cap: usize) -> Self {
        Self {
            available: Arc::new(AtomicI64::new(cap as i64)),
        }
    }

    // Mutable ref to ensure that it's used in single threading-context. As the
    // get-and-set is not atomic.
    fn acquire(&mut self, request: usize) -> Result<MemoryLimiterGuard> {
        let available = self.available.load(Ordering::Acquire);
        if available >= request as i64 {
            self.available.fetch_sub(request as i64, Ordering::AcqRel);
            Ok(MemoryLimiterGuard {
                limiter: self.clone(),
                request,
            })
        } else {
            Err(Error::MemoryLimitExceed { request, available })
        }
    }

    fn release(&self, request: usize) {
        self.available.fetch_add(request as i64, Ordering::AcqRel);
    }
}

struct MemoryLimiterGuard {
    limiter: MemoryLimiter,
    request: usize,
}

impl Drop for MemoryLimiterGuard {
    fn drop(&mut self) {
        self.limiter.release(self.request);
    }
}

#[cfg(test)]
mod tests {
    use bytes::Bytes;
    use rand::Rng;

    use super::*;

    #[test]
    fn test_chunk_header() {
        let header = ChunkHeader::new(CompressionType::Lz4Compression);
        let mut buf = Vec::with_capacity(ChunkHeader::len());
        header.encode_to(&mut buf);
        let header2 = ChunkHeader::decode(&buf).unwrap();
        assert_eq!(header, header2);
    }

    #[test]
    fn test_wal_chunk_integrity() {
        let (_, rx) = tikv_util::mpsc::unbounded();
        let (tx, _) = tikv_util::mpsc::unbounded();
        let dfs_config = kvengine::dfs::DFSConfig::default();
        let service_worker_epoch = Arc::new(AtomicU32::new(0));
        let mut worker = ObjectStorageWorker::new(
            LightweightBackupConfig::new(
                std::env::temp_dir(),
                1024 * 1024,
                CompressionType::Lz4Compression,
                CompressionType::Lz4Compression,
                dfs_config,
                1024 * 1024,
                4096,
                1 << 20,
                None,
            ),
            1,
            Arc::new(AtomicU64::new(1)),
            Healthy::default(),
            rx,
            tx,
            service_worker_epoch,
            Arc::new(DfsMeter::default()),
        );

        let mut origin_data = vec![];
        let mut chunks_data = vec![];

        for _ in 0..10 {
            let buf = generate_random_bytes(1024 * 128);
            // Save buf to data first.
            origin_data.extend_from_slice(&buf);
            worker.set_buf(buf);
            let chunk = worker.take_chunk_data().unwrap();
            // Save chunk data to chunks_data.
            chunks_data.push(Bytes::from(chunk));
        }
        // Append empty chunk to chunks_data should not affect the result.
        worker.set_buf(vec![]);
        let chunk = worker.take_chunk_data().unwrap();
        chunks_data.push(Bytes::from(chunk));

        // Assemble chunk data and verify with origin data.
        let assembled_data = assemble_wal_chunks(chunks_data).unwrap();
        let assembled_data = assembled_data.to_vec();
        assert_eq!(origin_data, assembled_data);
    }

    #[test]
    fn test_overwritten_epoch() {
        let (_, rx) = tikv_util::mpsc::unbounded();
        let (tx, _) = tikv_util::mpsc::unbounded();
        let dfs_config = kvengine::dfs::DFSConfig::default();
        let service_worker_epoch = Arc::new(AtomicU32::new(0));
        let worker = ObjectStorageWorker::new(
            LightweightBackupConfig::new(
                std::env::temp_dir(),
                1024 * 1024,
                CompressionType::Lz4Compression,
                CompressionType::Lz4Compression,
                dfs_config,
                1024 * 1024,
                4096,
                1 << 20,
                None,
            ),
            1,
            Arc::new(AtomicU64::new(1)),
            Healthy::default(),
            rx,
            tx,
            service_worker_epoch.clone(),
            Arc::new(DfsMeter::default()),
        );

        let cases = vec![
            (1, 0, 0),
            (2, 0, 0),
            (3, 0, 1),
            (4, 0, 2),
            (5, 1, 3),
            (6, 2, 4),
            (7, 3, 5),
            (u32::MAX, 0xffff_fffb, 0xffff_fffd),
        ];

        for (service_epoch_id, overwritten_epoch, near_overwritten_epoch) in cases {
            service_worker_epoch.store(service_epoch_id, Ordering::SeqCst);
            assert_eq!(worker.overwritten_epoch(), overwritten_epoch);
            assert_eq!(worker.near_overwritten_epoch(), near_overwritten_epoch);
        }
    }

    fn generate_random_bytes(size: usize) -> Vec<u8> {
        let mut rng = rand::thread_rng();
        let mut random_bytes = Vec::with_capacity(size);

        for _ in 0..size {
            random_bytes.push(rng.gen::<u8>());
        }

        random_bytes
    }

    use std::time::Duration;

    use rstest::rstest;

    #[rstest]
    #[case::no_interval(
        None,
        100,
        100,
        1024 * 1024,
        Some(100),
        false,
        "Should not chunk without flush_chunk_interval set"
    )]
    #[case::with_interval_before_elapsed(
        Some(Duration::from_millis(50)),
        100,
        100,
        1024 * 1024,
        None,
        false,
        "Should not chunk immediately after buffer set"
    )]
    #[case::with_interval_after_elapsed(
        Some(Duration::from_millis(50)),
        100,
        100,
        1024 * 1024,
        Some(60),
        true,
        "Should chunk after flush_chunk_interval elapsed"
    )]
    fn test_flush_chunk_interval(
        #[case] flush_interval: Option<Duration>,
        #[case] buffer_size: usize,
        #[case] to_read: usize,
        #[case] target_file_size: usize,
        #[case] sleep_ms: Option<u64>,
        #[case] expected_should_chunk: bool,
        #[case] assertion_msg: &str,
    ) {
        let (_, rx) = tikv_util::mpsc::unbounded();
        let (tx, _) = tikv_util::mpsc::unbounded();
        let dfs_config = kvengine::dfs::DFSConfig::default();
        let service_worker_epoch = Arc::new(AtomicU32::new(0));

        let mut worker = ObjectStorageWorker::new(
            LightweightBackupConfig::new(
                std::env::temp_dir(),
                target_file_size,
                CompressionType::Lz4Compression,
                CompressionType::Lz4Compression,
                dfs_config,
                1024 * 1024,
                4096,
                1 << 20,
                flush_interval.map(ReadableDuration),
            ),
            1,
            Arc::new(AtomicU64::new(1)),
            Healthy::default(),
            rx,
            tx,
            service_worker_epoch,
            Arc::new(DfsMeter::default()),
        );

        // Set buffer
        worker.set_buf(vec![0u8; buffer_size]);

        // Sleep if specified
        if let Some(ms) = sleep_ms {
            std::thread::sleep(Duration::from_millis(ms));
        }

        // Assert the result
        assert_eq!(
            worker.should_chunk(to_read),
            expected_should_chunk,
            "{}",
            assertion_msg
        );
    }

    #[test]
    fn test_meter_basic() {
        // Test basic observe and measure functionality
        let meter = Arc::new(DfsMeter::default());

        // Initially should be empty
        let measure = meter.measure();
        assert_eq!(measure.uploaded_bytes, 0);
        assert_eq!(measure.request_count, 0);
        measure.acknowledge();

        // Observe some requests
        meter.observe_request(100);
        meter.observe_request(200);
        meter.observe_request(300);

        // Measure should capture and reset
        let measure = meter.measure();
        assert_eq!(measure.uploaded_bytes, 600);
        assert_eq!(measure.request_count, 3);

        // After measure, meter should be reset
        let measure2 = meter.measure();
        assert_eq!(measure2.uploaded_bytes, 0);
        assert_eq!(measure2.request_count, 0);
        measure2.acknowledge();

        // Test RAII: measure without acknowledge should recollect
        meter.observe_request(1000);
        let measure3 = meter.measure();
        assert_eq!(measure3.uploaded_bytes, 1000);
        assert_eq!(measure3.request_count, 1);
        // Drop measure3 without acknowledging - it should recollect
        drop(measure3);

        // The data should be back in the meter
        let measure4 = meter.measure();
        assert_eq!(measure4.uploaded_bytes, 1000);
        assert_eq!(measure4.request_count, 1);
        measure4.acknowledge();

        // Test that acknowledged measure won't recollect
        meter.observe_request(500);
        let measure5 = meter.measure();
        assert_eq!(measure5.uploaded_bytes, 500);
        assert_eq!(measure5.request_count, 1);
        measure5.acknowledge(); // This should clear the measure
        // Now meter should be empty
        let measure6 = meter.measure();
        assert_eq!(measure6.uploaded_bytes, 0);
        assert_eq!(measure6.request_count, 0);
        measure6.acknowledge();

        // Finally acknowledge the first measure (delayed acknowledge scenario)
        measure.acknowledge();

        // Test edge case: zero bytes
        meter.observe_request(0);
        let measure7 = meter.measure();
        assert_eq!(measure7.uploaded_bytes, 0);
        assert_eq!(measure7.request_count, 1);
        measure7.acknowledge();

        // Test large values
        meter.observe_request(u64::MAX / 2);
        meter.observe_request(u64::MAX / 2);
        let measure8 = meter.measure();
        assert_eq!(measure8.uploaded_bytes, u64::MAX - 1);
        assert_eq!(measure8.request_count, 2);
        measure8.acknowledge();
    }

    #[test]
    fn test_meter_concurrent_randomized() {
        use std::{sync::Barrier, thread, time::Duration};

        const NUM_OBSERVER_THREADS: usize = 8;
        const NUM_MEASURER_THREADS: usize = 4;
        const OPERATIONS_PER_THREAD: usize = 1000;

        let meter = Arc::new(DfsMeter::default());
        let barrier = Arc::new(Barrier::new(NUM_OBSERVER_THREADS + NUM_MEASURER_THREADS));

        // Track total bytes and requests for validation
        let total_bytes_sent = Arc::new(AtomicU64::new(0));
        let total_requests_sent = Arc::new(AtomicU64::new(0));
        let total_bytes_acked = Arc::new(AtomicU64::new(0));
        let total_requests_acked = Arc::new(AtomicU64::new(0));

        let mut handles = vec![]; // Spawn observer threads that continuously call observe_request
        for _thread_id in 0..NUM_OBSERVER_THREADS {
            let meter = Arc::clone(&meter);
            let barrier = Arc::clone(&barrier);
            let total_bytes = Arc::clone(&total_bytes_sent);
            let total_requests = Arc::clone(&total_requests_sent);

            let handle = thread::spawn(move || {
                let mut rng = rand::thread_rng();
                barrier.wait(); // Synchronize start for maximum concurrency

                for _ in 0..OPERATIONS_PER_THREAD {
                    // Random bytes between 1 and 10000
                    let bytes = rng.gen_range(1..10000);
                    meter.observe_request(bytes);
                    total_bytes.fetch_add(bytes, Ordering::SeqCst);
                    total_requests.fetch_add(1, Ordering::SeqCst);

                    // Random tiny sleep to simulate real-world timing
                    if rng.gen_bool(0.1) {
                        thread::sleep(Duration::from_micros(rng.gen_range(1..10)));
                    }
                }
            });
            handles.push(handle);
        }

        // Spawn measurer threads that continuously measure and randomly acknowledge
        for _ in 0..NUM_MEASURER_THREADS {
            let meter = Arc::clone(&meter);
            let barrier = Arc::clone(&barrier);
            let total_bytes_acked = Arc::clone(&total_bytes_acked);
            let total_requests_acked = Arc::clone(&total_requests_acked);

            let handle = thread::spawn(move || {
                let mut rng = rand::thread_rng();
                barrier.wait(); // Synchronize start for maximum concurrency

                let mut deferred_measures = vec![];
                for _ in 0..OPERATIONS_PER_THREAD {
                    let measure = meter.measure();

                    // Randomly decide: acknowledge immediately, defer, or drop (simulate failure)
                    let choice = rng.gen_range(0..100);
                    if choice < 60 {
                        // 60% chance: acknowledge immediately (success case)
                        total_bytes_acked.fetch_add(measure.uploaded_bytes, Ordering::SeqCst);
                        total_requests_acked.fetch_add(measure.request_count, Ordering::SeqCst);
                        measure.acknowledge();
                    } else if choice < 85 {
                        // 25% chance: defer acknowledgment (will ack later)
                        deferred_measures.push(measure);
                    } else {
                        // 15% chance: drop without acknowledging (simulate upload failure)
                        // This will trigger recollect via Drop
                        drop(measure); // Explicitly drop to trigger recollect
                    }

                    // Randomly acknowledge some deferred measures
                    if rng.gen_bool(0.4) && !deferred_measures.is_empty() {
                        let idx = rng.gen_range(0..deferred_measures.len());
                        let measure = deferred_measures.swap_remove(idx);
                        total_bytes_acked.fetch_add(measure.uploaded_bytes, Ordering::SeqCst);
                        total_requests_acked.fetch_add(measure.request_count, Ordering::SeqCst);
                        measure.acknowledge();
                    }

                    // Randomly drop some deferred measures (simulate batch failure)
                    if rng.gen_bool(0.1) && !deferred_measures.is_empty() {
                        let idx = rng.gen_range(0..deferred_measures.len());
                        let measure = deferred_measures.swap_remove(idx);
                        drop(measure); // Trigger recollect
                    }

                    // Small random delay
                    if rng.gen_bool(0.1) {
                        thread::sleep(Duration::from_micros(rng.gen_range(1..10)));
                    }
                }

                // At the end, acknowledge all remaining deferred measures
                for measure in deferred_measures {
                    total_bytes_acked.fetch_add(measure.uploaded_bytes, Ordering::SeqCst);
                    total_requests_acked.fetch_add(measure.request_count, Ordering::SeqCst);
                    measure.acknowledge();
                }
            });
            handles.push(handle);
        }

        // Wait for all threads to complete
        for handle in handles {
            handle.join().unwrap();
        }

        // Important: After recollection, the data goes back to the meter
        // We need to measure one more time to capture recollected data
        let final_measure = meter.measure();
        total_bytes_acked.fetch_add(final_measure.uploaded_bytes, Ordering::SeqCst);
        total_requests_acked.fetch_add(final_measure.request_count, Ordering::SeqCst);
        final_measure.acknowledge();

        // Validate: all observed data should be acknowledged (including recollected)
        let sent_bytes = total_bytes_sent.load(Ordering::SeqCst);
        let sent_requests = total_requests_sent.load(Ordering::SeqCst);
        let acked_bytes = total_bytes_acked.load(Ordering::SeqCst);
        let acked_requests = total_requests_acked.load(Ordering::SeqCst);

        assert_eq!(
            sent_bytes, acked_bytes,
            "Bytes mismatch: sent {} but acknowledged {}",
            sent_bytes, acked_bytes
        );
        assert_eq!(
            sent_requests, acked_requests,
            "Requests mismatch: sent {} but acknowledged {}",
            sent_requests, acked_requests
        );

        // Verify meter is empty after everything
        let final_check = meter.measure();
        assert_eq!(final_check.uploaded_bytes, 0);
        assert_eq!(final_check.request_count, 0);
        final_check.acknowledge();

        println!(
            "Concurrent test passed: {} bytes in {} requests",
            sent_bytes, sent_requests
        );
    }
}
