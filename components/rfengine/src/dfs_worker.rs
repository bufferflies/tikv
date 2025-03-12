// Copyright 2023 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    fs,
    io::{Read, Seek, SeekFrom},
    path::PathBuf,
    sync::{
        atomic::{AtomicI64, AtomicU32, AtomicU64, Ordering},
        Arc,
    },
};

use bytes::{Buf, BufMut, Bytes, BytesMut};
use engine_traits::ObjectStorage;
use kvengine::dfs::{DFSConfig, Dfs, S3Fs};
use slog_global::*;
use tikv_util::mpsc::{Receiver, Sender};

use crate::{
    compact_worker::CompactTask, compress_lz4, decompress_lz4, get_integral_wal_chunks,
    last_wal_chunk_file_key, manifest::Manifest, metrics::RFENGINE_DFS_WORKER_HEALTHY_GAUGE,
    parse_wal_chunk_key, wal_chunk_file_key, wal_chunk_file_prefix, wal_file_name, Error, Result,
};

#[derive(Debug)]
pub(crate) struct LightweightBackupConfig {
    pub(crate) dir: PathBuf,
    pub(crate) wal_chunk_target_file_size: usize,
    pub(crate) compression_type: CompressionType,
    pub(crate) dfs_config: DFSConfig,

    pub(crate) rlog_cache_capacity: usize,
    pub(crate) rlog_cache_size_threshold: usize,
    pub(crate) memory_limit: usize,
}

impl LightweightBackupConfig {
    pub(crate) fn new(
        dir: PathBuf,
        wal_chunk_target_file_size: usize,
        compression_type: CompressionType,
        dfs_config: DFSConfig,
        rlog_cache_capacity: usize,
        rlog_cache_size_threshold: usize,
        memory_limit: usize,
    ) -> Self {
        Self {
            dir,
            wal_chunk_target_file_size,
            compression_type,
            dfs_config,
            rlog_cache_capacity,
            rlog_cache_size_threshold,
            memory_limit,
        }
    }
}

pub(crate) struct ObjectStorageWorker {
    config: LightweightBackupConfig,
    engine_id: Arc<AtomicU64>,
    task_rx: Receiver<ObjectStorageTask>,
    compact_worker_tx: Sender<CompactTask>,
    buf: Vec<u8>,
    async_wal_file: Option<fs::File>,
    epoch_id: u32,
    chunk_id: u32,
    start_off: u64, // The start offset of the current chunk.
    sync_off: u64,  // The offset of the syncing of current wal.
    s3fs: Arc<S3Fs>,
    healthy: Healthy,
    memory_limiter: MemoryLimiter,
}

impl ObjectStorageWorker {
    pub(crate) fn new(
        config: LightweightBackupConfig,
        epoch_id: u32,
        engine_id: Arc<AtomicU64>,
        dfs_worker_healthy: Healthy,
        task_rx: Receiver<ObjectStorageTask>,
        compact_worker_tx: Sender<CompactTask>,
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
            buf: Vec::with_capacity(wal_chunk_target_file_size),
            async_wal_file: None,
            chunk_id: 1,
            start_off: 0,
            sync_off: 0,
            s3fs,
            healthy: dfs_worker_healthy,
            memory_limiter,
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
        self.wait_for_bootstrapped();
        info!("dfs worker start init.");
        let mut rebuild_epoch = self.epoch_id;
        let store_id = self.get_engine_id();
        let mut last_chunk_key = None;
        loop {
            let scan_prefix = wal_chunk_file_prefix(store_id, rebuild_epoch);
            info!(
                "rebuild last wal chunk list chunks with prefix {}",
                scan_prefix
            );

            // Chunks in an epoch should be listed in one iterate.
            let (chunks, has_more) = self.s3fs.list_objects("", Some(&scan_prefix), None)?;
            debug_assert_eq!(has_more, None);
            if !chunks.is_empty() {
                let chunk_keys = chunks.into_iter().map(|x| x.key).collect::<Vec<_>>();
                if let Ok((mut integral_chunks, ..)) = get_integral_wal_chunks(&chunk_keys) {
                    last_chunk_key = integral_chunks.pop();
                    if last_chunk_key.is_some() {
                        info!("found integral wal chunks";
                            "integral_chunks" => ?integral_chunks, "last_chunk" => ?last_chunk_key);
                        break;
                    }
                }
            }

            if rebuild_epoch <= 1 || self.epoch_id >= rebuild_epoch + 3 {
                need_snapshot = true;
                break;
            }
            rebuild_epoch -= 1;
        }
        match last_chunk_key {
            None => {
                // Rebuild from the earliest epoch.
                self.epoch_id = rebuild_epoch;
                self.sync_off = 0;
                self.start_off = 0;
                info!("no wal chunk found, rebuild from epoch {}", rebuild_epoch);
            }
            Some(key) => match parse_wal_chunk_key(Some(&key)) {
                Some((epoch_id, _, file_off, _)) => {
                    self.epoch_id = epoch_id;
                    self.start_off = file_off;
                    self.sync_off = file_off;
                    info!(
                        "found the last wal chunk {} rebuild from epoch {} offset {}",
                        key, epoch_id, file_off
                    );
                }
                None => return Err(Error::Other(format!("parse wal chunk key {} failed", key))),
            },
        }

        Ok(need_snapshot)
    }

    fn wait_for_bootstrapped(&self) {
        while self.get_engine_id() == 0 {
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
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
                ObjectStorageTask::Flush => {
                    // Send flush task before close in normal case. If close without flush, we can
                    // construct the case for wal chunk recovery in random test.
                    if !self.buf.is_empty() && self.next_chunk(false).is_err() {
                        self.healthy.set_unhealthy(self.epoch_id, "handle flush");
                    }
                }
                ObjectStorageTask::Close => unreachable!(),
            }
        }
    }

    // Write wal chunk from `start_off` to `file_off` in a single write.
    // rebuild all epoch async if `file_off` is 0
    fn rebuild_wal_chunk(&mut self, mut file_off: u64) -> Result<()> {
        let store_id = self.get_engine_id();
        let mut fd = fs::File::open(wal_file_name(self.config.dir.as_path(), self.epoch_id))?;
        fd.seek(SeekFrom::Start(self.start_off))?;
        if file_off == 0 {
            file_off = fd.metadata()?.len();
        }

        let sync_len = file_off - self.start_off;
        let buf_start = self.buf.len();
        let buf_end = buf_start + sync_len as usize;
        debug!(
            "{}: rebuild_wal_chunk epoch {} from {} to {} buf_start {} buf_end {} sync_len {}",
            store_id, self.epoch_id, self.start_off, file_off, buf_start, buf_end, sync_len
        );
        self.buf.resize(buf_end, 0);
        let bytes_read = fd.read(&mut self.buf[buf_start..buf_end])?;
        debug_assert_eq!(bytes_read, sync_len as usize);

        let file_key = last_wal_chunk_file_key(store_id, self.epoch_id, self.start_off, file_off);
        let chunk = self.take_chunk_data()?;

        let fs = self.s3fs.clone();
        let healthy = self.healthy.clone();
        let acquired = self.memory_limiter.acquire(chunk.len())?;
        let epoch_id = self.epoch_id;
        self.s3fs.get_runtime().spawn_blocking(move || {
            if let Err(err) = fs.put_objects(vec![(file_key, Bytes::from(chunk))]) {
                error!("{} put wal chunk failed", store_id; "err" => ?err);
                healthy.set_unhealthy(epoch_id, "put wal chunk");
            }
            drop(acquired);
        });

        // Reset the offset and buf after rebuild one epoch.
        self.start_off = 0;
        self.sync_off = 0;
        self.buf.clear();

        Ok(())
    }

    fn need_sync(&self, epoch_id: u32, file_off: u64) -> bool {
        self.epoch_id < epoch_id || (self.epoch_id == epoch_id && self.sync_off < file_off)
    }

    fn handle_sync(&mut self, epoch_id: u32, file_off: u64) -> Result<()> {
        if epoch_id != self.epoch_id {
            // Need to rebuild all previous epoch.
            for rebuild_epoch in self.epoch_id..epoch_id {
                self.rebuild_wal_chunk(0)?;
                self.epoch_id = rebuild_epoch + 1;
            }
        }
        debug_assert_eq!(epoch_id, self.epoch_id);

        let sync_len = file_off - self.sync_off;
        let store_id = self.get_engine_id();

        if self.should_chunk(sync_len as usize) {
            if let Some(async_wal_file) = &self.async_wal_file {
                // sync data before write to S3, to avoid S3 file ahead of async local file
                // after restart.
                async_wal_file.sync_data()?;
            }
            info!(
                "{}: handle_sync put wal epoch {} chunk {}",
                store_id, epoch_id, self.chunk_id
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

        async_wal_file.seek(SeekFrom::Start(self.sync_off))?;

        let buf_start = self.buf.len();
        let buf_end = buf_start + sync_len as usize;
        debug!(
            "{}: handle_sync epoch {} from {} to {} buf_start {} buf_end {}",
            store_id, epoch_id, self.sync_off, file_off, buf_start, buf_end
        );
        self.buf.resize(buf_end, 0);
        let bytes_read = async_wal_file.read(&mut self.buf[buf_start..buf_end])?;
        debug_assert_eq!(bytes_read, sync_len as usize);

        // Update the sync offset.
        self.sync_off = file_off;
        Ok(())
    }

    fn handle_rotate(&mut self, epoch_id: u32) -> Result<()> {
        debug!("{}: handle_rotate epoch {}", self.get_engine_id(), epoch_id);
        // Call next_chunk even self.buf is empty. This can cover the case the last
        // chunk flushed during stop with no `.last` suffix.
        self.next_chunk(true)?;
        self.async_wal_file = None;
        self.epoch_id = epoch_id + 1;
        self.start_off = 0;
        self.sync_off = 0;
        self.chunk_id = 1;

        Ok(())
    }

    pub(crate) fn should_chunk(&mut self, to_read: usize) -> bool {
        if !self.buf.is_empty()
            && self.buf.len() + ChunkHeader::len() + to_read
                > self.config.wal_chunk_target_file_size
        {
            return true;
        }
        false
    }

    // For test only.
    #[allow(dead_code)]
    pub(crate) fn set_buf(&mut self, buf: Vec<u8>) {
        self.buf = buf;
    }

    pub(crate) fn take_chunk_data(&mut self) -> Result<Vec<u8>> {
        let chunk_header = ChunkHeader::new(self.compression_type());
        let buf_len = self.buf.len();
        let mut chunk = Vec::with_capacity(ChunkHeader::len() + buf_len);
        chunk_header.encode_to(&mut chunk);
        // Get slice of the buffer but keep the memory.
        if self.need_compression() {
            let _ = compress_lz4(&self.buf, &mut chunk).map_err(|err| {
                error!("{} take chunk data: compress_lz4 failed", self.get_engine_id(); "err" => ?err);
                err
            })?;
        } else {
            chunk.extend_from_slice(self.buf.as_slice());
        };
        Ok(chunk)
    }

    fn next_chunk(&mut self, rotate: bool) -> Result<()> {
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
        self.s3fs.get_runtime().spawn_blocking(move || {
            if let Err(err) = fs.put_objects(vec![(file_key, Bytes::from(chunk))]) {
                error!("{} put wal chunk failed", store_id, ; "err" => ?err);
                healthy.set_unhealthy(epoch_id, "put wal chunk");
            }
            drop(acquired);
        });

        self.start_off = self.sync_off;
        self.chunk_id += 1;
        self.buf.clear();

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
        // Read chunk header.
        let header = ChunkHeader::decode(chunk.slice(0..ChunkHeader::len()).chunk())?;
        let decompressed_data = match header.compression_type {
            CompressionType::Lz4Compression => Bytes::from(
                decompress_lz4(chunk.slice(ChunkHeader::len()..).chunk()).map_err(Error::from)?,
            ),
            CompressionType::NoCompression => chunk.slice(ChunkHeader::len()..),
        };
        epoch_wal.put(decompressed_data);
    }
    Ok(epoch_wal)
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

    pub(crate) fn encode_to(&self, buf: &mut Vec<u8>) {
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
    Flush,                                 // Trigger flush the last chunk, mainly for test.
    Close,
}

impl ObjectStorageTask {
    fn epoch_id(&self) -> Option<u32> {
        match self {
            Self::Sync { epoch_id, .. } | Self::Rotate { epoch_id, .. } => Some(*epoch_id),
            Self::Flush | Self::Close => None,
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
        let mut worker = ObjectStorageWorker::new(
            LightweightBackupConfig::new(
                std::env::temp_dir(),
                1024 * 1024,
                CompressionType::Lz4Compression,
                dfs_config,
                1024 * 1024,
                4096,
                1 << 20,
            ),
            1,
            Arc::new(AtomicU64::new(1)),
            Healthy::default(),
            rx,
            tx,
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

    fn generate_random_bytes(size: usize) -> Vec<u8> {
        let mut rng = rand::thread_rng();
        let mut random_bytes = Vec::with_capacity(size);

        for _ in 0..size {
            random_bytes.push(rng.gen::<u8>());
        }

        random_bytes
    }
}
