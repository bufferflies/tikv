// Copyright 2021 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    cmp::min,
    fs,
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicU32, Ordering},
        Arc,
    },
    thread,
};

use bytes::{Buf, BufMut, Bytes};
use engine_traits::ObjectStorage;
use file_system::{DirectWriter, IORateLimitMode, IOType};
use rfenginepb::{StoreBackupMeta, WalChunk};
use slog_global::*;
use tikv_util::{mpsc::Receiver, time::Instant};

use crate::{log_batch::RaftLogBlock, manifest::Manifest, write_batch::PeerBatch, *};

const MAX_WAL_CHUNK_SIZE: u64 = 8 * 1024 * 1024;

pub(crate) struct Worker {
    dir: PathBuf,
    manifest: Manifest,
    writer: DirectWriter,
    task_rx: Receiver<Task>,
    buf: Vec<u8>,
    compacted_epoch: Arc<AtomicU32>,
}

impl Worker {
    pub(crate) fn new(
        dir: PathBuf,
        task_rx: Receiver<Task>,
        manifest: Manifest,
        compacted_epoch: Arc<AtomicU32>,
    ) -> Self {
        let rate_limiter = Arc::new(file_system::IORateLimiter::new(
            IORateLimitMode::WriteOnly,
            true,
            false,
        ));
        rate_limiter.set_io_rate_limit(128 * 1024 * 1024);
        let writer = DirectWriter::new(rate_limiter, IOType::Compaction);
        Self {
            dir,
            manifest,
            writer,
            task_rx,
            buf: vec![],
            compacted_epoch,
        }
    }

    pub(crate) fn run(&mut self) {
        while let Ok(task) = self.task_rx.recv() {
            match task {
                Task::Rotate { epoch_id } => {
                    if let Err(err) = self.compact(epoch_id) {
                        let engine_id = self.manifest.get_engine_id();
                        error!(
                            "{}: failed to compact epoch {} {:?}",
                            engine_id, epoch_id, err
                        );
                    }
                }
                Task::Truncates(truncates) => drop(truncates),
                Task::Close => return,
                Task::Backup(backup_task) => {
                    if backup_task.config.incremental {
                        self.incremental_backup(backup_task);
                    } else {
                        self.full_backup(backup_task);
                    }
                }
            }
        }
    }

    fn compact(&mut self, epoch_id: u32) -> Result<()> {
        let timer = Instant::now_coarse();
        let mut batch = WriteBatch::default();
        let mut it = WALIterator::new(self.dir.clone(), epoch_id);
        it.iterate(|region_batch| {
            batch.merge_peer(region_batch);
        })?;
        let mut change_set = rfenginepb::ChangeSet::default();
        change_set.set_epoch_id(epoch_id);
        let mut generated_files = 0;
        for (_, mut peer_batch) in batch.peers {
            let mut peer_meta_pb = rfenginepb::PeerMeta::default();
            peer_meta_pb.set_peer_id(peer_batch.peer_id);
            peer_meta_pb.set_region_id(peer_batch.meta.region_id);
            peer_meta_pb.set_truncated_index(peer_batch.truncated_idx);
            for (k, v) in &peer_batch.meta.states {
                let mut state_pb = rfenginepb::PeerState::new();
                state_pb.set_key(k.to_vec());
                state_pb.set_value(v.to_vec());
                peer_meta_pb.mut_states().push(state_pb);
            }
            peer_batch.truncate(peer_batch.truncated_idx);
            if !peer_batch.raft_logs.is_empty() {
                let file = self.write_raft_log_file(peer_batch)?;
                peer_meta_pb.mut_files().push(file);
                generated_files += 1;
            }
            change_set.mut_peers().push(peer_meta_pb);
        }
        let _ = file_system::sync_dir(self.dir.as_path());
        let engine_id = self.manifest.get_engine_id();
        info!(
            "{}: epoch {} compact wal file generated {} files",
            engine_id, epoch_id, generated_files,
        );
        self.manifest.handle_compaction(change_set)?;
        ENGINE_COMPACT_WAL_DURATION_HISTOGRAM.observe(timer.saturating_elapsed_secs());
        self.compacted_epoch.store(epoch_id, Ordering::SeqCst);
        Ok(())
    }

    fn write_raft_log_file(&mut self, peer_batch: PeerBatch) -> Result<rfenginepb::RaftLogFile> {
        let first = peer_batch.raft_logs.front().unwrap().index;
        let last = peer_batch.raft_logs.back().unwrap().index;
        let filename = raft_log_file_name(self.dir.as_path(), peer_batch.peer_id, first, last);
        self.buf.truncate(0);
        let header = RlogHeader::new(peer_batch.raft_logs.len() as u32);
        header.encode_to(&mut self.buf);
        // Write index first to make the file addressable.
        let mut log_end_off = 0;
        for rlog in &peer_batch.raft_logs {
            log_end_off += rlog.encoded_len() as u32 + 4 /* checksum */;
            self.buf.put_u32_le(log_end_off);
        }
        for rlog in &peer_batch.raft_logs {
            let origin_len = self.buf.len();
            rlog.encode_to(&mut self.buf);
            let checksum = crc32c::crc32c(&self.buf[origin_len..]);
            self.buf.put_u32_le(checksum);
        }
        self.writer.write_to_file(&self.buf, &filename)?;
        let mut file = rfenginepb::RaftLogFile::default();
        file.first_index = first;
        file.last_index = last;
        Ok(file)
    }

    fn full_backup(&mut self, task: BackupTask) {
        let engine_id = self.manifest.get_engine_id();
        info!("{}: start backup task", engine_id);
        let wal_epoch = self.manifest.epoch_id + 1;
        let mut objects = vec![];
        let mut backup_meta = StoreBackupMeta::default();
        backup_meta.set_store_id(engine_id);
        match self.backup_wal(&mut backup_meta, wal_epoch, 0, task.file_off) {
            Ok(mut objs) => objects.append(&mut objs),
            Err(e) => {
                (task.callback)(Err(format!("backup wal failed {:?}", e)));
                return;
            }
        }
        let manifest = self.manifest.to_change_set(true); // Exclude tombstone peers.
        for peer in manifest.get_peers() {
            for file in peer.get_files() {
                let file_name =
                    raft_log_file_name(&self.dir, peer.peer_id, file.first_index, file.last_index);
                match fs::read(&file_name) {
                    Ok(rlog_data) => {
                        let rlog_key = raft_log_file_key(
                            backup_meta.get_store_id(),
                            peer.get_peer_id(),
                            file.first_index,
                            file.last_index,
                        );
                        objects.push((rlog_key, Bytes::from(rlog_data)));
                    }
                    Err(err) => {
                        (task.callback)(Err(format!("read {:?} failed {:?}", &file_name, err)));
                        return;
                    }
                }
            }
        }
        backup_meta.set_manifest(manifest);
        let total_size: usize = objects.iter().map(|(_, data)| data.len()).sum();
        info!(
            "backup read file count: {}, size: {}",
            objects.len(),
            total_size
        );
        // Starts a background task in case the object storage is slow and blocking WAL compaction.
        thread::spawn(move || {
            if let Err(err) = task.object_storage.put_objects(objects) {
                (task.callback)(Err(err));
                return;
            }
            (task.callback)(Ok(backup_meta));
        });
    }

    fn backup_wal(
        &mut self,
        backup_meta: &mut StoreBackupMeta,
        wal_epoch: u32,
        start_off: u64,
        end_off: u64,
    ) -> Result<Vec<(String, Bytes)>> {
        let wal_file_name = wal_file_name(&self.dir, wal_epoch);
        let mut wal_file = File::open(wal_file_name)?;
        let mut chunks = vec![];
        let mut total_size = 0;
        let backup_size = end_off - start_off;
        while total_size < backup_size {
            let chunk_size = min(MAX_WAL_CHUNK_SIZE, backup_size - total_size);
            let chunk = vec![0u8; chunk_size as usize];
            chunks.push(chunk);
            total_size += chunk_size;
        }
        let mut objects = vec![];
        let mut offset = start_off;
        if offset > 0 {
            wal_file.seek(SeekFrom::Start(offset))?;
        }
        for mut chunk in chunks.drain(..) {
            wal_file.read_exact(chunk.as_mut_slice())?;
            let mut wal_chunk = WalChunk::default();
            wal_chunk.set_epoch(wal_epoch);
            wal_chunk.set_start_off(offset);
            wal_chunk.set_end_off(offset + chunk.len() as u64);
            let wal_key = wal_file_key(
                backup_meta.get_store_id(),
                wal_chunk.get_epoch(),
                wal_chunk.get_start_off(),
                wal_chunk.get_end_off(),
            );
            backup_meta.mut_wal_chunks().push(wal_chunk);
            offset += chunk.len() as u64;
            objects.push((wal_key, Bytes::from(chunk)));
        }
        Ok(objects)
    }

    fn incremental_backup(&mut self, mut task: BackupTask) {
        let engine_id = self.manifest.get_engine_id();
        let wal_epoch = self.manifest.epoch_id + 1;
        // If epoch is not matched, fallback to full backup.
        if wal_epoch != task.config.wal_epoch {
            warn!(
                "Fallback to full backup as wal epoch changed, cur: {}, input:{}",
                wal_epoch, task.config.wal_epoch
            );
            task.config.incremental = false;
            task.config.start_offset = 0;
            return self.full_backup(task);
        }
        if task.file_off < task.config.start_offset {
            return (task.callback)(Err(format!(
                "WAL offset invalid, current {}, given start {}",
                task.file_off, task.config.start_offset
            )));
        }
        info!(
            "Engine {} start incremental backup task, epoch {}",
            engine_id, wal_epoch
        );
        let mut backup_meta = StoreBackupMeta::default();
        backup_meta.set_store_id(engine_id);
        let mut objects = vec![];
        match self.backup_wal(
            &mut backup_meta,
            wal_epoch,
            task.config.start_offset,
            task.file_off,
        ) {
            Ok(mut objs) => objects.append(&mut objs),
            Err(e) => {
                return (task.callback)(Err(format!("Backup WAL failed {:?}", e)));
            }
        }
        let total_size: usize = objects.iter().map(|(_, data)| data.len()).sum();
        info!(
            "incremental backup read file count: {}, size: {}",
            objects.len(),
            total_size
        );
        // Starts a background task in case the object storage is slow and blocking WAL compaction.
        thread::spawn(move || {
            if let Err(err) = task.object_storage.put_objects(objects) {
                return (task.callback)(Err(err));
            }
            (task.callback)(Ok(backup_meta));
        });
    }
}

pub(crate) fn raft_log_file_name(dir: &Path, peer_id: u64, first: u64, last: u64) -> PathBuf {
    dir.join(format!(
        "{:016x}_{:016x}_{:016x}.rlog",
        peer_id, first, last,
    ))
}

pub(crate) fn raft_log_file_key(store_id: u64, peer_id: u64, first: u64, last: u64) -> String {
    format!(
        "{:016x}/p{:016x}/{:016x}_{:016x}.rlog",
        store_id, peer_id, first, last
    )
}

pub(crate) fn wal_file_key(store_id: u64, epoch_id: u32, start_off: u64, end_off: u64) -> String {
    format!(
        "{:016x}/e{:08x}/{:016x}_{:016x}.wal",
        store_id, epoch_id, start_off, end_off
    )
}

/// Magic Number of rlog files. It's picked by running
///    echo rfengine.rlog | sha1sum
/// and taking the leading 64 bits.
const RLOG_MAGIC_NUMBER: u64 = 0x50ed2c1e89d6aa91;

#[derive(Clone, Copy)]
#[repr(u64)]
enum RlogVersion {
    V1 = 1,
}

pub(crate) struct RlogHeader {
    version: RlogVersion,
    pub(crate) count: u32,
}

impl RlogHeader {
    pub(crate) fn new(count: u32) -> Self {
        Self {
            version: RlogVersion::V1,
            count,
        }
    }

    pub(crate) const fn len() -> usize {
        20 // magic number + version + count
    }

    fn encode_to(&self, buf: &mut Vec<u8>) {
        buf.put_u64_le(RLOG_MAGIC_NUMBER);
        buf.put_u64_le(self.version as u64);
        buf.put_u32_le(self.count);
    }

    pub(crate) fn decode(mut buf: &[u8]) -> Result<Self> {
        if buf.len() < Self::len() {
            return Err(Error::Corruption("rlog header mismatch".to_owned()));
        }
        let magic_number = buf.get_u64_le();
        if magic_number != RLOG_MAGIC_NUMBER {
            return Err(Error::Corruption("rlog magic number mismatch".to_owned()));
        }
        let version = buf.get_u64_le();
        if version != RlogVersion::V1 as u64 {
            return Err(Error::Corruption("rlog version mismatch".to_owned()));
        }
        let count = buf.get_u32_le();
        Ok(Self {
            version: RlogVersion::V1,
            count,
        })
    }
}

pub(crate) fn wal_file_name(dir: &Path, epoch_id: u32) -> PathBuf {
    let idx = epoch_to_idx(epoch_id);
    dir.join(format!("{}.wal", idx))
}

pub(crate) enum Task {
    Rotate { epoch_id: u32 },
    Truncates(Vec<Vec<RaftLogBlock>>),
    Close,
    Backup(BackupTask),
}

#[derive(Default, Debug, Serialize, Deserialize)]
#[serde(default)]
#[serde(rename_all = "kebab-case")]
// If incremental is true, backup the same `epoch` WAL from the `start_offset`
pub struct BackupConfig {
    pub cluster_id: u64,
    pub store_id: u64,
    pub incremental: bool,
    pub wal_epoch: u32,
    pub start_offset: u64,
}
pub struct BackupTask {
    pub object_storage: Box<dyn ObjectStorage>,
    pub callback: Box<dyn FnOnce(std::result::Result<StoreBackupMeta, String>) + Send>,
    pub(crate) file_off: u64,
    pub(crate) config: BackupConfig,
}

impl BackupTask {
    pub fn new(
        object_storage: Box<dyn ObjectStorage>,
        callback: Box<dyn FnOnce(std::result::Result<StoreBackupMeta, String>) + Send>,
        config: BackupConfig,
    ) -> Self {
        Self {
            file_off: 0,
            object_storage,
            callback,
            config,
        }
    }
}
