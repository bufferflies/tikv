// Copyright 2021 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicU32, Ordering},
        mpsc::Receiver,
        Arc,
    },
};

use bytes::{Buf, BufMut};
use file_system::{DirectWriter, IORateLimitMode, IOType};
use slog_global::*;
use tikv_util::time::Instant;

use crate::{log_batch::RaftLogBlock, manifest::Manifest, write_batch::PeerBatch, *};

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
}

pub(crate) fn raft_log_file_name(dir: &Path, peer_id: u64, first: u64, last: u64) -> PathBuf {
    dir.join(format!(
        "{:016x}_{:016x}_{:016x}.rlog",
        peer_id, first, last,
    ))
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
}
