// Copyright 2021 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    cmp::min,
    collections::HashMap,
    fmt::{Display, Formatter},
    fs,
    io::{Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, AtomicU32, Ordering},
        Arc,
    },
    thread,
    thread::JoinHandle,
};

use api_version::{api_v2, ApiV2};
use bytes::{Buf, BufMut, Bytes, BytesMut};
use engine_traits::ObjectStorage;
use kvproto::raft_serverpb::RegionLocalState;
use protobuf::Message;
use rfenginepb::{
    KeySpaceBackupMeta, RaftLogBackupFile, RaftLogFile, StoreBackupMeta, StoreRaftLogBackupMeta,
    WalChunk,
};
use slog_global::*;
use tikv_util::{
    mpsc::{Receiver, Sender},
    sys::thread::StdThreadBuildWrapper,
    time::Instant,
    DFS_WORKER_THREAD_NAME,
};

use crate::{
    log_batch::RaftLogBlock,
    manifest::Manifest,
    metrics::{RFENGINE_BACKUP_COUNTER, RFENGINE_BACKUP_DURATION_HISTOGRAM},
    write_batch::PeerBatch,
    *,
};

const MAX_WAL_CHUNK_SIZE: u64 = 128 * 1024 * 1024;
pub(crate) struct Worker {
    dir: PathBuf,
    manifest: Manifest,
    task_rx: Receiver<Task>,
    dfs_worker_handle: Option<ObjectStorageWorkerHandle>,
    buf: Vec<u8>,
    compacted_epoch: Arc<AtomicU32>,
    async_wal_writer: Option<WalWriter>,
    dfs_worker_healthy: Arc<AtomicBool>,
}

impl Worker {
    pub(crate) fn new(
        dir: PathBuf,
        task_rx: Receiver<Task>,
        callback: Sender<Task>,
        manifest: Manifest,
        compacted_epoch: Arc<AtomicU32>,
        async_wal_writer: Option<WalWriter>,
        object_storage_config: Option<ObjectStorageConfig>,
        dfs_worker_healthy: Arc<AtomicBool>,
    ) -> Self {
        // Create new thread for object storage worker if lightweight backup enabled.
        let dfs_worker_handle = if let Some(config) = object_storage_config {
            assert!(
                async_wal_writer.is_some(),
                "async wal writer must be enabled"
            );

            let (tx, rx) = tikv_util::mpsc::unbounded();
            let epoch_id = manifest.epoch_id + 1;
            let mut object_storage_worker = ObjectStorageWorker::new(
                config,
                epoch_id,
                manifest.engine_id.clone(),
                dfs_worker_healthy.clone(),
                rx,
                callback,
            );
            let handle = thread::Builder::new()
                .name(DFS_WORKER_THREAD_NAME.to_string())
                .spawn_wrapper(move || object_storage_worker.run())
                .unwrap();
            Some(ObjectStorageWorkerHandle {
                task_sender: tx,
                handle,
            })
        } else {
            None
        };
        Self {
            dir,
            manifest,
            task_rx,
            dfs_worker_handle,
            buf: vec![],
            compacted_epoch,
            async_wal_writer,
            dfs_worker_healthy,
        }
    }

    pub(crate) fn is_lightweight_enabled(&self) -> bool {
        self.dfs_worker_handle.is_some()
    }

    fn is_dfs_worker_healthy(&self) -> bool {
        self.dfs_worker_healthy.load(Ordering::Acquire)
    }

    pub(crate) fn run(&mut self) {
        while let Ok(task) = self.task_rx.recv() {
            match task {
                Task::Rotate { epoch_id } => self.handle_rotate(epoch_id),
                Task::Truncates(truncates) => drop(truncates),
                Task::Close { force } => {
                    self.handle_close(force);
                    return;
                }
                Task::Backup(task) => self.handle_backup(task),
                Task::Dump {
                    epoch_id,
                    start_off,
                    end_off,
                    callback,
                } => self.handle_dump(epoch_id, start_off, end_off, callback),
                Task::Write { wb } => self.handle_write(&wb),
                Task::Snapshot => {
                    info!("{}: init trigger snapshot", self.manifest.get_engine_id());
                    self.handle_snapshot()
                }
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
    }

    fn handle_backup(&mut self, mut task: BackupTask) {
        if let Some(async_writer) = self.async_wal_writer.as_ref() {
            task.file_off = async_writer.file_off;
        }
        if task.config.lightweight {
            self.lightweight_backup(task);
        } else if task.config.incremental {
            self.incremental_backup(task);
        } else {
            self.full_backup(task);
        }
    }

    fn handle_dump(
        &mut self,
        epoch_id: u32,
        start_off: u64,
        end_off: u64,
        callback: Box<dyn FnOnce(Result<Bytes>) + Send>,
    ) {
        if let Some(writer) = self.async_wal_writer.as_ref() {
            // Check the WAL chunk meta is valid.
            if epoch_id > writer.epoch_id || epoch_id + 3 < writer.epoch_id || start_off >= end_off
            {
                let msg = format!(
                    "{}: invalid dump wal chunk epoch {} start_off {} end_off {} writer epoch {}, file_off {}",
                    self.manifest.get_engine_id(),
                    epoch_id,
                    start_off,
                    end_off,
                    writer.epoch_id,
                    writer.file_off
                );
                error!("{}", msg);
                callback(Err(Error::Other(msg)));
                return;
            }
            // Dump the WAL chunk from offset start_off to end_off.
            info!(
                "{}: dump latest wal epoch {} start_off {} end_off {}",
                self.manifest.get_engine_id(),
                epoch_id,
                start_off,
                end_off,
            );
            match dump_wal_chunk(&self.dir, epoch_id, start_off, end_off) {
                Ok(chunk) => callback(Ok(chunk)),
                Err(err) => {
                    let msg = format!(
                        "{}: dump wal chunk epoch {} start_off {} end_off {} failed {:?}",
                        self.manifest.get_engine_id(),
                        epoch_id,
                        start_off,
                        end_off,
                        err
                    );
                    error!("{}", msg);
                    callback(Err(Error::Other(msg)));
                }
            }
        }
    }

    fn handle_rotate(&mut self, epoch_id: u32) {
        if let Some(async_writer) = self.async_wal_writer.as_mut() {
            assert_eq!(async_writer.epoch_id, epoch_id);
            let file_off = async_writer.file_off;
            async_writer.rotate().unwrap();

            if self.is_lightweight_enabled() {
                // Send rotate task to object storage worker.
                self.dfs_worker_handle
                    .as_ref()
                    .unwrap()
                    .task_sender
                    .send(ObjectStorageTask::Rotate { epoch_id, file_off })
                    .unwrap();
            }
        }

        if let Err(err) = self.compact(epoch_id) {
            let engine_id = self.manifest.get_engine_id();
            error!(
                "{}: failed to compact epoch {} {:?}",
                engine_id, epoch_id, err
            );
        }

        // NOTE: Snapshot frequency is same as the manifest rewritten frequency. It is
        // not configurable.
        let should_snapshot = self.manifest.should_snapshot();
        if should_snapshot && self.is_lightweight_enabled() {
            self.handle_snapshot();
        }
    }

    fn handle_snapshot(&mut self) {
        info!("{}: handle snapshot", self.manifest.get_engine_id());
        match self.snapshot_backup() {
            Ok(snapshot_objects) => {
                self.dfs_worker_handle
                    .as_ref()
                    .unwrap()
                    .task_sender
                    .send(ObjectStorageTask::Snapshot { snapshot_objects })
                    .unwrap();
            }
            Err(err) => {
                let engine_id = self.manifest.get_engine_id();
                let epoch_id = self.manifest.epoch_id;
                error!(
                    "{}: failed to snapshot epoch {} {:?}",
                    engine_id, epoch_id, err
                );
            }
        }
    }

    fn handle_write(&mut self, wb: &WriteBatch) {
        if let Some(wal_writer) = &mut self.async_wal_writer {
            wal_writer.write_batch(wb).unwrap();
            let file_off = wal_writer.file_off;
            let epoch_id = wal_writer.epoch_id;
            if self.is_lightweight_enabled() {
                // Send write task to object storage worker.
                let task = ObjectStorageTask::Sync { epoch_id, file_off };
                self.dfs_worker_handle
                    .as_ref()
                    .unwrap()
                    .task_sender
                    .send(task)
                    .unwrap();
            }
        }
    }

    fn compact(&mut self, epoch_id: u32) -> Result<()> {
        let timer = Instant::now_coarse();
        let mut batch = WriteBatch::default();
        let mut it = WalIterator::new(self.dir.clone(), epoch_id);
        it.iterate_batch(|data, _| {
            WalIterator::iterate_peer_batch(data, |region_batch| {
                batch.merge_peer(region_batch);
            });
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
        let duration = timer.saturating_elapsed();
        info!(
            "{}: epoch {} compact wal file generated {} files takes {:?}",
            engine_id, epoch_id, generated_files, duration,
        );
        self.manifest.handle_compaction(change_set)?;
        ENGINE_COMPACT_WAL_DURATION_HISTOGRAM.observe(duration.as_secs_f64());
        self.compacted_epoch.store(epoch_id, Ordering::SeqCst);
        Ok(())
    }

    // rfenginepb::RaftLogFile format:
    // RlogHeader + [endoffset] + [RaftLogOp + checksum]
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
        let mut file = fs::File::create(filename)?;
        file.write_all(&self.buf)?;
        file.sync_data()?;
        let mut file = rfenginepb::RaftLogFile::default();
        file.first_index = first;
        file.last_index = last;
        Ok(file)
    }

    fn backup_callback(
        task: BackupTask,
        ret: std::result::Result<StoreBackupMeta, String>,
        label: &str,
        ob_start_time: Instant,
    ) {
        RFENGINE_BACKUP_COUNTER.with_label_values(&[label]).inc();
        if ret.is_ok() {
            RFENGINE_BACKUP_DURATION_HISTOGRAM
                .with_label_values(&[label])
                .observe(ob_start_time.saturating_elapsed_secs());
        }
        (task.callback)(ret);
    }

    fn snapshot_backup(&mut self) -> Result<Vec<(String, Bytes)>> {
        let engine_id = self.manifest.get_engine_id();
        info!("{}: start snapshot task", engine_id);
        let mut objects = vec![];
        let mut backup_meta = StoreBackupMeta::default();
        backup_meta.set_store_id(engine_id);

        let manifest = self.manifest.to_change_set(true); // Exclude tombstone peers.
        match self.backup_raft_log_files(&manifest, &mut backup_meta, true) {
            Ok(obj) => objects.push(obj),
            Err(e) => {
                return Err(Error::Other(format!("backup raft log failed {:?}", e)));
            }
        }
        // manifest epoch id already increased by 1.
        let epoch_id = manifest.get_epoch_id();
        backup_meta.set_manifest(manifest);
        let total_size: usize = objects.iter().map(|(_, data)| data.len()).sum();
        info!(
            "snapshot backup write file count: {}, size: {}, raft meta offset {}",
            objects.len(),
            total_size,
            backup_meta.raft_meta_start_off,
        );

        // Also need snapshot backup_meta.
        let meta_key = snapshot_store_meta_key(engine_id, epoch_id);
        let meta_data = backup_meta.write_to_bytes().unwrap();
        objects.push((meta_key, Bytes::from(meta_data)));

        Ok(objects)
    }

    fn full_backup(&mut self, task: BackupTask) {
        let engine_id = self.manifest.get_engine_id();
        info!("{}: start backup task", engine_id);
        let ob_start_time = Instant::now();
        let wal_epoch = self.manifest.epoch_id + 1;
        let mut objects = vec![];
        let mut backup_meta = StoreBackupMeta::default();
        backup_meta.set_store_id(engine_id);
        match self.backup_wal(&mut backup_meta, wal_epoch, 0, task.file_off) {
            Ok(mut objs) => objects.append(&mut objs),
            Err(e) => {
                return Self::backup_callback(
                    task,
                    Err(format!("backup wal failed {:?}", e)),
                    "full_fail",
                    ob_start_time,
                );
            }
        }
        let manifest = self.manifest.to_change_set(true); // Exclude tombstone peers.
        match self.backup_raft_log_files(&manifest, &mut backup_meta, false) {
            Ok(obj) => objects.push(obj),
            Err(e) => {
                return Self::backup_callback(
                    task,
                    Err(format!("backup raft log failed {:?}", e)),
                    "full_fail",
                    ob_start_time,
                );
            }
        }
        backup_meta.set_manifest(manifest);
        let total_size: usize = objects.iter().map(|(_, data)| data.len()).sum();
        info!(
            "backup write file count: {}, size: {}, raft meta offset {}",
            objects.len(),
            total_size,
            backup_meta.raft_meta_start_off,
        );
        // Starts a background task in case the object storage is slow and blocking WAL
        // compaction.
        thread::spawn(move || {
            if let Err(err) = task.object_storage.put_objects(objects) {
                return Self::backup_callback(task, Err(err), "full_fail", ob_start_time);
            }
            Self::backup_callback(task, Ok(backup_meta), "full_success", ob_start_time);
        });
    }

    fn get_keyspace_id_from_peer(peer_meta: &rfenginepb::PeerMeta) -> u32 {
        let keyspace_id = match peer_meta
            .get_states()
            .iter()
            .rev() // the states are got from BTreeMap iter, so the last one is latest.
            .find(|s| s.get_key().starts_with(REGION_META_KEY_PREFIX))
        {
            Some(state) => {
                let mut local_state = RegionLocalState::default();
                local_state.merge_from_bytes(state.get_value()).unwrap();
                utils::get_region_keyspace_id(local_state.get_region())
            }
            None => {
                warn!("Get unknown keyspace id peer {:?}", peer_meta);
                debug_assert!(peer_meta.peer_id == 0 && peer_meta.region_id == 0);
                api_v2::UNKNOWN_KEYSPACE_ID
            }
        };
        ApiV2::get_u32_keyspace_id(keyspace_id)
    }

    // Aggregate all raft logs into one file by keyspace id.
    fn backup_raft_log_files(
        &mut self,
        manifest: &rfenginepb::ChangeSet,
        store_meta: &mut StoreBackupMeta,
        is_snapshot: bool,
    ) -> Result<(String, Bytes)> {
        let store_id = self.manifest.get_engine_id();
        // keyspace_id -> Vec<(peer_id, RaftLogFile)>
        let mut keyspace_map: HashMap<u32, Vec<(u64, &RaftLogFile)>> = HashMap::default();
        let mut raft_log_size = 0;
        for peer in manifest.get_peers() {
            let peer_id = peer.get_peer_id();
            let keyspace_id = Self::get_keyspace_id_from_peer(peer);
            let peer_files = peer.get_files();
            let mut files = Vec::with_capacity(peer_files.len());
            for f in peer_files {
                files.push((peer_id, f));
                let file_name = raft_log_file_name(&self.dir, peer_id, f.first_index, f.last_index);
                raft_log_size += fs::metadata(file_name)?.len();
            }
            keyspace_map
                .entry(keyspace_id)
                .and_modify(|k| k.append(&mut files))
                .or_insert_with(|| files);
        }
        let object_key = if is_snapshot {
            snapshot_rlog_key(store_id, manifest.get_epoch_id())
        } else {
            store_raft_log_file_key(store_id, manifest.get_epoch_id())
        };
        // Reserve 10MB for `rlog_meta`, which should be enough in most scenarios.
        let mut object = BytesMut::with_capacity(raft_log_size as usize + 10 * 1024 * 1024);
        let mut rlog_meta = StoreRaftLogBackupMeta::default();
        rlog_meta.mut_header().version = 1;
        // Aggregate the raft log data with keyspace id.
        for (keyspace_id, files) in keyspace_map {
            let mut keyspace_meta = KeySpaceBackupMeta::default();
            keyspace_meta.keyspace_id = keyspace_id;
            for (peer_id, file) in files {
                let file_name =
                    raft_log_file_name(&self.dir, peer_id, file.first_index, file.last_index);
                let data = fs::read(file_name)?;
                let mut backup_file = RaftLogBackupFile::default();
                backup_file.peer_id = peer_id;
                backup_file.start_off = object.len() as u64;
                object.put_slice(&data);
                backup_file.end_off = object.len() as u64;
                backup_file.first_index = file.first_index;
                backup_file.last_index = file.last_index;
                keyspace_meta.mut_files().push(backup_file);
            }
            rlog_meta.mut_raft_logs().insert(keyspace_id, keyspace_meta);
        }
        store_meta.raft_meta_start_off = object.len() as u64;
        let meta = rlog_meta.write_to_bytes().unwrap();
        object.put_slice(&meta);
        Ok((object_key, object.freeze()))
    }

    fn backup_wal(
        &mut self,
        backup_meta: &mut StoreBackupMeta,
        wal_epoch: u32,
        start_off: u64,
        end_off: u64,
    ) -> Result<Vec<(String, Bytes)>> {
        let wal_file_name = wal_file_name(&self.dir, wal_epoch);
        let mut wal_file = fs::File::open(wal_file_name)?;
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
        let ob_start_time = Instant::now();
        if task.file_off < task.config.start_offset {
            let msg = format!(
                "WAL offset invalid, current {}, given start {}",
                task.file_off, task.config.start_offset
            );
            return Self::backup_callback(task, Err(msg), "incr_fail", ob_start_time);
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
                return Self::backup_callback(
                    task,
                    Err(format!("Backup WAL failed {:?}", e)),
                    "incr_fail",
                    ob_start_time,
                );
            }
        }
        let total_size: usize = objects.iter().map(|(_, data)| data.len()).sum();
        info!(
            "incremental backup read file count: {}, size: {}",
            objects.len(),
            total_size
        );
        // Starts a background task in case the object storage is slow and blocking WAL
        // compaction.
        thread::spawn(move || {
            if let Err(err) = task.object_storage.put_objects(objects) {
                return Self::backup_callback(task, Err(err), "incr_fail", ob_start_time);
            }
            Self::backup_callback(task, Ok(backup_meta), "incr_success", ob_start_time);
        });
    }

    fn lightweight_backup(&mut self, task: BackupTask) {
        if !self.is_dfs_worker_healthy() {
            return Self::backup_callback(
                task,
                Err("dfs worker unhealthy".to_string()),
                "light_fail",
                Instant::now(),
            );
        }

        let engine_id = self.manifest.get_engine_id();
        let wal_epoch = self.manifest.epoch_id + 1;
        let file_off = task.file_off;

        let mut backup_meta = StoreBackupMeta::default();
        backup_meta.set_store_id(engine_id);
        backup_meta.set_epoch(wal_epoch);
        backup_meta.set_offset(file_off);

        Self::backup_callback(task, Ok(backup_meta), "light_success", Instant::now());
    }
}

pub(crate) struct ObjectStorageWorkerHandle {
    task_sender: Sender<ObjectStorageTask>,
    handle: JoinHandle<()>,
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
            return Err(Error::Corruption {
                msg: format!("rlog header mismatch: len {}", buf.len()),
                epoch_id: 0,
                offset: 0,
                data: buf.to_vec(),
            });
        }
        let magic_number = buf.get_u64_le();
        if magic_number != RLOG_MAGIC_NUMBER {
            return Err(Error::Corruption {
                msg: format!(
                    "rlog magic number mismatch: magic_number {:x}",
                    magic_number
                ),
                epoch_id: 0,
                offset: 0,
                data: buf.to_vec(),
            });
        }
        let version = buf.get_u64_le();
        if version != RlogVersion::V1 as u64 {
            return Err(Error::Corruption {
                msg: format!("rlog version mismatch: version {}", version),
                epoch_id: 0,
                offset: 0,
                data: buf.to_vec(),
            });
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

fn dump_wal_chunk(dir: &Path, epoch_id: u32, start_off: u64, end_off: u64) -> Result<Bytes> {
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

pub(crate) enum Task {
    Rotate {
        epoch_id: u32,
    },
    Truncates(Vec<Vec<RaftLogBlock>>),
    Close {
        force: bool,
    },
    Backup(BackupTask),
    Dump {
        epoch_id: u32,
        start_off: u64,
        end_off: u64,
        callback: Box<dyn FnOnce(Result<Bytes>) + Send>,
    },
    Write {
        wb: WriteBatch,
    },
    Snapshot, // Used for dfs_worker callback to trigger snapshot.
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
    pub lightweight: bool,
}

impl Display for BackupConfig {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "cluster_id {}, store_id {}, incremental {}, lightweight {} wal_epoch {}, start_offset {} ",
            self.cluster_id,
            self.store_id,
            self.incremental,
            self.lightweight,
            self.wal_epoch,
            self.start_offset,
        )
    }
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

    pub fn is_lightweight(&self) -> bool {
        self.config.lightweight
    }
}

#[cfg(test)]
mod tests {
    use std::{
        collections::HashMap,
        fs, iter,
        sync::atomic::{AtomicBool, AtomicU32, AtomicU64},
    };

    use bytes::Buf;
    use kvproto::raft_serverpb::RegionLocalState;
    use protobuf::Message;
    use rand::{distributions::Alphanumeric, Rng};
    use rfenginepb::{ChangeSet, PeerState, StoreBackupMeta, StoreRaftLogBackupMeta};
    use tikv_util::defer;

    use crate::{
        log_batch::{RaftLogOp, RaftLogs},
        manifest::{persist_change_set, Manifest},
        raft_log_file_name, region_state_key, store_raft_log_file_key,
        tests::{get_txn_endkey_prefix, get_txn_startkey_prefix, init_logger},
        write_batch::PeerBatch,
        RfEngine, RfEngineConfig, WalWriter, Worker, WriterType,
    };

    fn generate_random_str() -> Vec<u8> {
        let mut rng = rand::thread_rng();
        let len = rng.gen::<usize>() % 1024 + 1;
        iter::repeat(())
            .map(|()| rng.sample(Alphanumeric))
            .take(len)
            .collect()
    }

    fn write_keyspace_state(peer_meta: &mut rfenginepb::PeerMeta, keyspace: u32) {
        let mut state = PeerState::default();
        // mock a fake keyspace epoch version.
        let region_epoch = 10;
        state.key = region_state_key(region_epoch - 1).to_vec();
        let mut local_stat = RegionLocalState::default();
        local_stat.mut_region().start_key = get_txn_startkey_prefix(keyspace - 1).to_vec();
        local_stat.mut_region().end_key = get_txn_endkey_prefix(keyspace - 1).to_vec();
        state.value = local_stat.write_to_bytes().unwrap();
        peer_meta.mut_states().push(state.clone());

        state.key = region_state_key(region_epoch).to_vec();
        local_stat.mut_region().start_key = get_txn_startkey_prefix(keyspace).to_vec();
        local_stat.mut_region().end_key = get_txn_endkey_prefix(keyspace).to_vec();
        state.value = local_stat.write_to_bytes().unwrap();
        peer_meta.mut_states().push(state);
    }

    #[test]
    fn test_backup_raft_log_files() {
        init_logger();
        let tmp_dir = tempfile::tempdir().unwrap();
        let tmp_path = tmp_dir.path();
        defer!(fs::remove_dir_all(tmp_path).unwrap());
        let (tx, rx) = tikv_util::mpsc::unbounded();
        let engine_id = 999;
        let manifest = Manifest::open(tmp_path, AtomicU64::new(engine_id).into()).unwrap();
        let mut worker = Worker::new(
            tmp_path.to_path_buf(),
            rx,
            tx,
            manifest,
            AtomicU32::new(0).into(),
            None,
            None,
            std::sync::Arc::new(AtomicBool::new(true)),
        );
        let epoch = 990;
        let mut cs = ChangeSet::new();
        cs.epoch_id = epoch;
        let mut peer_rlog_files = HashMap::new();
        let mut total_file_cnt = 0;
        for i in 100..203 {
            // peers
            let peer_id = i;
            let mut meta_pb = rfenginepb::PeerMeta::new();
            meta_pb.set_peer_id(peer_id);
            meta_pb.set_region_id(peer_id * 2);
            meta_pb.set_truncated_index(i * 10);
            let keyspace = (peer_id / 5) as u32;
            write_keyspace_state(&mut meta_pb, keyspace);
            let mut files = vec![];
            for j in 0..10 {
                // files
                let mut raft_log_file = rfenginepb::RaftLogFile::default();
                let first_index = j * 100 + 1;
                let last_index = (j + 1) * 100;
                raft_log_file.set_first_index(first_index);
                raft_log_file.set_last_index(last_index);
                meta_pb.mut_files().push(raft_log_file);
                let file_name = raft_log_file_name(tmp_path, peer_id, first_index, last_index);
                let content = generate_random_str();
                fs::write(file_name, &content).unwrap();
                files.push(content);
                total_file_cnt += 1;
            }
            peer_rlog_files.insert(peer_id, files);
            cs.mut_peers().push(meta_pb);
        }
        let mut store_meta = StoreBackupMeta::default();
        let (key, object) = worker
            .backup_raft_log_files(&cs, &mut store_meta, false)
            .unwrap();
        let raft_file_key = store_raft_log_file_key(engine_id, epoch);
        assert_eq!(raft_file_key, key);
        let mut raft_meta = StoreRaftLogBackupMeta::default();
        let size = object.len();
        let meta_bytes = &object.chunk()[store_meta.raft_meta_start_off as usize..size];
        raft_meta.merge_from_bytes(meta_bytes).unwrap();
        assert_eq!(raft_meta.get_header().version, 1);
        let mut ret_file_cnt = 0;
        for (keyspace_id, keyspace_data) in raft_meta.take_raft_logs() {
            assert_eq!(keyspace_id, keyspace_data.get_keyspace_id());
            for file in keyspace_data.get_files() {
                let peer_id = file.peer_id;
                assert_eq!(keyspace_id as u64, peer_id / 5);
                let data = &object.chunk()[file.start_off as usize..file.end_off as usize];
                let rlog_files = peer_rlog_files[&peer_id].clone();
                let idx = ((file.first_index - 1) / 100) as usize;
                let file_data = rlog_files[idx].clone();
                assert_eq!(
                    file_data.len(),
                    data.len(),
                    "peer id {}, index {}-{}",
                    peer_id,
                    file.first_index,
                    file.last_index
                );
                assert_eq!(
                    &file_data, data,
                    "peer id {}, index {}-{}",
                    peer_id, file.first_index, file.last_index
                );
                ret_file_cnt += 1;
            }
        }
        assert_eq!(ret_file_cnt, total_file_cnt);
    }

    fn generate_rlog_files(
        worker: &mut Worker,
        raft_logs: &mut RaftLogs,
        peer_id: u64,
        region_id: u64,
        first_index: u64,
        last_index: u64,
    ) -> rfenginepb::RaftLogFile {
        let mut peer_batch = PeerBatch::new(peer_id, region_id);
        for index in first_index..=last_index {
            let op = RaftLogOp {
                index,
                term: index as u32,
                e_type: 1,
                context: 2,
                data: generate_random_str().into(),
            };
            peer_batch.append_raft_log(op.clone());
            raft_logs.append(op);
        }
        worker.write_raft_log_file(peer_batch).unwrap()
    }

    #[test]
    fn test_backup_and_load_raft_log_files() {
        init_logger();
        let tmp_dir = tempfile::tempdir().unwrap();
        let tmp_path = tmp_dir.path();
        defer!(fs::remove_dir_all(tmp_path).unwrap());
        let (tx, rx) = tikv_util::mpsc::unbounded();
        let engine_id = 1999;
        let manifest = Manifest::open(tmp_path, AtomicU64::new(engine_id).into()).unwrap();
        let mut worker = Worker::new(
            tmp_path.to_path_buf(),
            rx,
            tx,
            manifest,
            AtomicU32::new(0).into(),
            None,
            None,
            std::sync::Arc::new(AtomicBool::new(true)),
        );
        let mut cs = ChangeSet::new();
        let mut peer_data_map = HashMap::new();
        let peers_range = 100..150;
        let mut total_file_cnt = 0;
        for i in peers_range.clone() {
            let peer_id = i;
            let region_id = peer_id * 2;
            let mut meta_pb = rfenginepb::PeerMeta::new();
            meta_pb.set_peer_id(peer_id);
            meta_pb.set_region_id(region_id);
            meta_pb.set_truncated_index(0);
            let keyspace_id = i as u32 / 10;
            write_keyspace_state(&mut meta_pb, keyspace_id);
            let mut peer_raft_log = RaftLogs::default();
            for j in 0..10 {
                let first_index = j * 100 + 1;
                let last_index = (j + 1) * 100;
                let raft_log_file = generate_rlog_files(
                    &mut worker,
                    &mut peer_raft_log,
                    peer_id,
                    region_id,
                    first_index,
                    last_index,
                );
                meta_pb.mut_files().push(raft_log_file);
                total_file_cnt += 1;
            }
            cs.mut_peers().push(meta_pb);
            cs.epoch_id += 1;
            peer_data_map.insert(peer_id, peer_raft_log);
        }
        // 1. backup
        let mut store_meta = StoreBackupMeta::default();
        let (key, object) = worker
            .backup_raft_log_files(&cs, &mut store_meta, false)
            .unwrap();
        let raft_file_key = store_raft_log_file_key(engine_id, cs.epoch_id);
        assert_eq!(raft_file_key, key);

        let mut raft_meta = StoreRaftLogBackupMeta::default();
        let size = object.len();
        let meta_bytes = &object.chunk()[store_meta.raft_meta_start_off as usize..size];
        raft_meta.merge_from_bytes(meta_bytes).unwrap();
        assert_eq!(raft_meta.get_header().version, 1);

        // 2. restore to new dir
        let tmp_dir2 = tempfile::tempdir().unwrap();
        let tmp_path2 = tmp_dir2.path();
        defer!(fs::remove_dir_all(tmp_path2).unwrap());
        // restore raft log files by keyspace
        let mut restore_file_cnt = 0;
        for (keyspace_id, keyspace_data) in raft_meta.take_raft_logs() {
            assert_eq!(keyspace_id, keyspace_data.get_keyspace_id());
            for file in keyspace_data.get_files() {
                let peer_id = file.peer_id;
                assert_eq!(peer_id / 10, keyspace_id as u64);
                let data = &object.chunk()[file.start_off as usize..file.end_off as usize];
                let file_name =
                    raft_log_file_name(tmp_path2, peer_id, file.first_index, file.last_index);
                fs::write(file_name, data).unwrap();
                restore_file_cnt += 1;
            }
        }
        assert_eq!(total_file_cnt, restore_file_cnt);
        // restore manifest file
        let manifest = Manifest::open(tmp_path2, AtomicU64::new(engine_id).into()).unwrap();
        persist_change_set(&manifest.file, 0, &cs).unwrap();

        // 3. open RfEngine with restored dir.
        let wal_size = 4 * 1024 * 1024;
        let cfg = RfEngineConfig::new(wal_size);
        // Hack wal file to let RfEngine::open pass.
        let mut wal_writer = WalWriter::new(
            tmp_path2,
            wal_size,
            1024,
            AtomicU32::new(cs.epoch_id + 1).into(),
            WriterType::Sync,
        );
        wal_writer.open_file(cs.epoch_id + 1, 0).unwrap();
        // checksum inner should succeeds.
        let engine = RfEngine::open(tmp_path2, &cfg, None, None).unwrap();
        // 4. Check peer data, restored data should be same with previous one.
        for peer_id in peers_range {
            let cache_peer_data = peer_data_map.get(&peer_id).unwrap();
            for index in 1..=1000 {
                let entry1 = cache_peer_data.get(index);
                let entry2 = engine.get_raft_entry(peer_id, index);
                assert_eq!(entry1, entry2);
            }
        }
    }

    #[test]
    fn test_get_keyspace_id_from_peer() {
        init_logger();
        let mut peer_meta = rfenginepb::PeerMeta::default();
        assert_eq!(Worker::get_keyspace_id_from_peer(&peer_meta), 16777215);
        write_keyspace_state(&mut peer_meta, 100);
        write_keyspace_state(&mut peer_meta, 200);
        assert_eq!(Worker::get_keyspace_id_from_peer(&peer_meta), 200);
    }
}
