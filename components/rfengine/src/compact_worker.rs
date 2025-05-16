// Copyright 2021 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    borrow::Cow,
    cmp::min,
    collections::HashMap,
    fmt,
    fmt::{Display, Formatter},
    fs,
    fs::File,
    io::{Read, Seek, SeekFrom, Write},
    mem,
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicU32, Ordering},
        Arc,
    },
    thread,
    thread::JoinHandle,
    time::Duration,
};

use bytes::{Buf, BufMut, Bytes, BytesMut};
use engine_traits::ObjectStorage;
use kvengine::dfs::S3Fs;
use protobuf::Message;
use quick_cache::unsync::Cache as QuickCache;
use rfenginepb::{
    KeySpaceBackupMeta, RaftLogBackupFile, RaftLogFile, StoreBackupMeta, StoreRaftLogBackupMeta,
    WalChunk,
};
use slog_global::*;
use tikv_util::{
    backoff,
    mpsc::{Receiver, SendError, Sender},
    sys::thread::StdThreadBuildWrapper,
    time::Instant,
    DFS_WORKER_THREAD_NAME,
};

use crate::{
    manifest::Manifest,
    metrics::{RFENGINE_BACKUP_COUNTER, RFENGINE_BACKUP_DURATION_HISTOGRAM},
    write_batch::PeerBatch,
    *,
};

const MAX_WAL_CHUNK_SIZE: u64 = 128 * 1024 * 1024;

// Maximum size of snapshot. This size is determined by maximum object size of
// S3, which is 5 GiB.
const MAX_SNAPSHOT_SIZE: u64 = 4500 * 1024 * 1024; // 4.5 GiB

const COMPACT_RETRY_TIMES: usize = 30;

pub(crate) struct WorkerHandle {
    pub(crate) task_sender: Sender<CompactTask>,
    pub(crate) handle: Option<JoinHandle<()>>,
}

impl WorkerHandle {
    pub(crate) fn try_send(&self, task: CompactTask) {
        if let Err(SendError(t)) = self.task_sender.send(task) {
            warn!("failed to send task {:?}", t);
        }
    }
}

pub(crate) struct CompactWorker {
    dir: PathBuf,
    manifest: Manifest,
    task_rx: Receiver<CompactTask>,
    snap_task_handle: Option<
        JoinHandle<
            std::result::Result<
                u32,           // epoch_id
                (u32, String), // epoch_id, err
            >,
        >,
    >,
    buf: Vec<u8>,
    compacted_epoch: Arc<AtomicU32>,
    s3fs: Option<Arc<S3Fs>>,
    last_snap_epoch_id: u32,

    // Used to cache small rlogs to reduce disk IO when taking snapshot.
    rlog_cache: RlogCache,
    rlog_compression_type: CompressionType,
    healthy: Healthy,

    sync_concurrency: usize,
    files_to_sync: Vec<File>,
}

impl CompactWorker {
    pub(crate) fn new(
        dir: PathBuf,
        task_rx: Receiver<CompactTask>,
        manifest: Manifest,
        compacted_epoch: Arc<AtomicU32>,
        lightweight_backup_cfg: Option<&LightweightBackupConfig>,
        s3fs: Option<Arc<S3Fs>>,
        healthy: Healthy,
        sync_concurrency: usize,
    ) -> Self {
        // Create new thread for object storage worker if lightweight backup enabled.
        let (rlog_cache, compress_type) = if let Some(config) = lightweight_backup_cfg {
            (
                RlogCache::new(config.rlog_cache_capacity, config.rlog_cache_size_threshold),
                config.rlog_compression_type,
            )
        } else {
            (RlogCache::none(), CompressionType::NoCompression)
        };

        Self {
            dir,
            manifest,
            task_rx,
            snap_task_handle: None,
            buf: vec![],
            compacted_epoch,
            s3fs,
            last_snap_epoch_id: 0,
            rlog_cache,
            rlog_compression_type: compress_type,
            healthy,
            sync_concurrency,
            files_to_sync: vec![],
        }
    }

    pub(crate) fn is_lightweight_enabled(&self) -> bool {
        self.s3fs.is_some()
    }

    pub(crate) fn run(&mut self) {
        while let Ok(task) = self.task_rx.recv() {
            match task {
                CompactTask::Compact { epoch_id } => self.handle_compact(epoch_id),
                CompactTask::HeavyBackup(task) => self.handle_heavy_backup(task),
                CompactTask::Snapshot => {
                    self.handle_snapshot();
                }
                CompactTask::Close { force } => {
                    self.handle_close(force);
                    return;
                }
            }
        }
    }

    fn handle_close(&mut self, force: bool) {
        // Close and join snapshot task thread.
        if let Some(snap_handle) = self.snap_task_handle.take() {
            if !force {
                let _ = snap_handle.join().unwrap();
            }
        }
    }

    fn handle_heavy_backup(&mut self, task: BackupTask) {
        if task.config.incremental {
            self.incremental_backup(task);
        } else {
            self.full_backup(task);
        }
    }

    fn handle_compact_with_backoff(&mut self, epoch_id: u32) {
        let mut back_off = backoff::ExponentialBackoff::new(
            Duration::from_secs(5),
            Duration::from_secs(60),
            COMPACT_RETRY_TIMES,
        );
        let engine_id = self.manifest.get_engine_id();
        loop {
            let next_delay = back_off.next_delay();
            if next_delay.is_err() {
                panic!("compact epoch {} retry times exceeded", epoch_id);
            }
            match self.compact(epoch_id) {
                Ok(_) => break,
                Err(err) => {
                    error!(
                        "{}: failed to compact epoch {} retry_times: {}, {:?}",
                        engine_id,
                        epoch_id,
                        back_off.current_attempts(),
                        err
                    );
                    thread::sleep(next_delay.unwrap());
                }
            }
        }
    }

    fn handle_compact(&mut self, epoch_id: u32) {
        info!("handle compact {}", epoch_id);
        if let Err(err) = self.compact(epoch_id) {
            let engine_id = self.manifest.get_engine_id();
            error!(
                "{}: failed to compact epoch {} {:?}",
                engine_id, epoch_id, err
            );
            self.handle_compact_with_backoff(epoch_id);
            info!("handle compact {} success after retry", epoch_id);
        }

        if self.manifest.should_snapshot() {
            self.handle_snapshot();
        }
    }

    fn handle_snapshot(&mut self) {
        if !self.is_lightweight_enabled() {
            return;
        }
        self.snapshot_backup();
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
        let mut cached_files = 0;
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
                let (file, is_cached) = self.write_raft_log_file(peer_batch)?;
                peer_meta_pb.mut_files().push(file);
                generated_files += 1;
                cached_files += is_cached as usize;
            }
            change_set.mut_peers().push(peer_meta_pb);
        }
        self.sync_files();
        let _ = file_system::sync_dir(self.dir.as_path());

        self.manifest.handle_compaction(change_set)?;

        let engine_id = self.manifest.get_engine_id();
        let duration = timer.saturating_elapsed();
        let pending_tasks = self.task_rx.len();
        let cache_size = self.rlog_cache.cache_size();
        info!(
            "{}: compact wal", engine_id;
            "size" => it.offset,
            "generated_files" => generated_files,
            "cached_files" => cached_files,
            "cache_size" => cache_size,
            "takes" => ?duration,
            "pending_tasks" => pending_tasks,
        );
        ENGINE_COMPACT_WAL_DURATION_HISTOGRAM.observe(duration.as_secs_f64());
        self.compacted_epoch.store(epoch_id, Ordering::SeqCst);
        Ok(())
    }

    // rfenginepb::RaftLogFile format:
    // RlogHeader + [endoffset] + [RaftLogOp + checksum]
    fn write_raft_log_file(
        &mut self,
        peer_batch: PeerBatch,
    ) -> Result<(rfenginepb::RaftLogFile, bool /* is_cached */)> {
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
        // delay the files_to_sync to sync files in parallel.
        self.files_to_sync.push(file);
        let mut file = rfenginepb::RaftLogFile::default();
        file.first_index = first;
        file.last_index = last;

        let is_cached = self
            .rlog_cache
            .add_rlog(peer_batch.peer_id, &file, &self.buf);
        Ok((file, is_cached))
    }

    fn snapshot_backup(&mut self) {
        self.try_join_snap_task();
        let engine_id = self.manifest.get_engine_id();
        if self.has_unfinished_snap_task() {
            self.healthy
                .set_unhealthy(self.manifest.epoch_id, "snapshot task unfinished");
            warn!("{}: snapshot task unfinished", engine_id);
        }
        if !self.healthy.check_healthy(self.manifest.epoch_id) {
            warn!("{}: skip unhealthy snapshot backup", engine_id);
            return;
        }
        let timer = Instant::now_coarse();
        info!("{}: start snapshot task", engine_id);
        let mut backup_meta = StoreBackupMeta::default();
        backup_meta.set_store_id(engine_id);

        let manifest = self.manifest.to_change_set(true); // Exclude tombstone peers.
        let rlog_obj_res = self.backup_raft_log_files(&manifest, &mut backup_meta, true);
        if rlog_obj_res.is_err() {
            warn!(
                "{}: create snapshot rlog file failed {:?}",
                engine_id,
                rlog_obj_res.unwrap_err()
            );
            self.healthy
                .set_unhealthy(self.manifest.epoch_id, "create snapshot rlog");
            return;
        }
        let rlog_obj = rlog_obj_res.unwrap();
        // manifest epoch id already increased by 1.
        let epoch_id = manifest.get_epoch_id();
        backup_meta.set_manifest(manifest);
        let duration = timer.saturating_elapsed();
        info!(
            "snapshot backup write file size: {}, raft meta offset {}, takes {:?}",
            rlog_obj.1.len(),
            backup_meta.raft_meta_start_off,
            duration,
        );
        ENGINE_TAKE_SNAPSHOT_DURATION_HISTOGRAM.observe(duration.as_secs_f64());

        // Also need snapshot backup_meta.
        let meta_key = snapshot_store_meta_key(engine_id, epoch_id);
        let meta_data = backup_meta.write_to_bytes().unwrap();
        let meta_obj = (meta_key, Bytes::from(meta_data));

        // `rlog_obj` should be written to DFS at the end, as we scan for latest
        // snapshot by the rlog object.
        // See https://github.com/tidbcloud/cloud-storage-engine/issues/1840.
        let s3fs = self.s3fs.as_ref().unwrap().clone();
        // We use the same DFS_WORKER_THREAD_NAME thread name to make panic mark file
        // work.
        let snap_task_handle = thread::Builder::new()
            .name(DFS_WORKER_THREAD_NAME.into())
            .spawn_wrapper(move || {
                for obj in [meta_obj, rlog_obj] {
                    if let Err(err) = s3fs.put_objects(vec![obj]) {
                        error!("{} put snapshot object failed", engine_id; "err" => ?err, "epoch" => epoch_id);
                        return Err((epoch_id, err));
                    }
                }
                Ok(epoch_id)
            })
            .unwrap();
        self.snap_task_handle = Some(snap_task_handle);
    }

    fn try_join_snap_task(&mut self) {
        if self
            .snap_task_handle
            .as_ref()
            .map(|h| h.is_finished())
            .unwrap_or_default()
        {
            let handle = self.snap_task_handle.take().unwrap();
            match handle.join().unwrap() {
                Ok(epoch_id) => {
                    self.last_snap_epoch_id = epoch_id;
                }
                Err((epoch_id, err)) => {
                    self.healthy.set_unhealthy(epoch_id, "join snapshot task");
                    let engine_id = self.manifest.get_engine_id();
                    error!("{} joined snapshot task failed {:?}", engine_id, err);
                }
            }
        }
    }

    fn has_unfinished_snap_task(&self) -> bool {
        self.snap_task_handle
            .as_ref()
            .map(|h| !h.is_finished())
            .unwrap_or_default()
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
                return backup_callback(
                    task,
                    Err(Error::Backup(format!("backup wal failed {:?}", e))),
                    "full_fail",
                    ob_start_time,
                );
            }
        }
        let manifest = self.manifest.to_change_set(true); // Exclude tombstone peers.
        match self.backup_raft_log_files(&manifest, &mut backup_meta, false) {
            Ok(obj) => objects.push(obj),
            Err(e) => {
                return backup_callback(
                    task,
                    Err(Error::Backup(format!("backup raft log failed {:?}", e))),
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
                return backup_callback(task, Err(Error::Backup(err)), "full_fail", ob_start_time);
            }
            backup_callback(task, Ok(backup_meta), "full_success", ob_start_time);
        });
    }

    fn get_keyspace_id_from_peer(store_id: u64, peer_meta: &rfenginepb::PeerMeta) -> u32 {
        match utils::get_keyspace_id_from_peer(peer_meta) {
            Some(keyspace_id) => keyspace_id,
            None => {
                if peer_meta.peer_id != 0 || peer_meta.region_id != 0 {
                    warn!(
                        "{}: get unknown keyspace id peer: {:?}",
                        store_id, peer_meta
                    );
                    debug_assert!(
                        false,
                        "{}: unknown keyspace id peer: {:?}",
                        store_id, peer_meta
                    );
                }
                0
            }
        }
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
            let keyspace_id = Self::get_keyspace_id_from_peer(store_id, peer);
            let peer_files = peer.get_files();
            let mut files = Vec::with_capacity(peer_files.len());
            for f in peer_files {
                files.push((peer_id, f));

                let rlog_size = if let Some(rlog_size) = self.rlog_cache.get_rlog_size(peer_id, f) {
                    rlog_size
                } else {
                    let file_name =
                        raft_log_file_name(&self.dir, peer_id, f.first_index, f.last_index);
                    fs::metadata(file_name)?.len() as usize
                };
                raft_log_size += rlog_size;

                if raft_log_size as u64 > MAX_SNAPSHOT_SIZE {
                    // The size is limited by maximum size of S3 objects (5 GiB).
                    // The alternative solution is to split the snapshot to multiple chunks. But as
                    // we (will) have flow control on WAL size, exceeds the limitation should be
                    // rare.
                    return Err(Error::SnapshotOversize(raft_log_size as u64));
                }
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
        let mut object = BytesMut::with_capacity(raft_log_size + 10 * 1024 * 1024);
        let mut rlog_meta = StoreRaftLogBackupMeta::default();
        rlog_meta.mut_header().version = 2;
        rlog_meta
            .mut_header()
            .set_compression_type(self.rlog_compression_type.to());

        // Aggregate the raft log data with keyspace id.
        let (mut total_file_cnt, mut cached_file_cnt, mut cached_file_size) = (0, 0, 0);
        for (keyspace_id, files) in keyspace_map {
            let mut keyspace_meta = KeySpaceBackupMeta::default();
            keyspace_meta.keyspace_id = keyspace_id;
            for (peer_id, file) in files {
                let data = if let Some(data) = self.rlog_cache.get_rlog_data(peer_id, file) {
                    cached_file_cnt += 1;
                    cached_file_size += data.len();
                    Cow::from(data)
                } else {
                    let file_name =
                        raft_log_file_name(&self.dir, peer_id, file.first_index, file.last_index);
                    let data = fs::read(file_name)?;
                    self.rlog_cache.add_rlog(peer_id, file, &data);
                    Cow::from(data)
                };
                let compressed_data = match self.rlog_compression_type {
                    CompressionType::Lz4Compression => {
                        let mut chunk = Vec::with_capacity(data.len());
                        compress_lz4(&data, &mut chunk)?;
                        Cow::Owned(chunk)
                    }
                    CompressionType::NoCompression => Cow::Borrowed(data.as_ref()),
                };
                let mut backup_file = RaftLogBackupFile::default();
                backup_file.peer_id = peer_id;
                backup_file.start_off = object.len() as u64;
                object.put_slice(compressed_data.as_ref());
                backup_file.end_off = object.len() as u64;
                backup_file.first_index = file.first_index;
                backup_file.last_index = file.last_index;
                keyspace_meta.mut_files().push(backup_file);

                total_file_cnt += 1;
            }
            rlog_meta.mut_raft_logs().insert(keyspace_id, keyspace_meta);
        }

        info!("{}: backup raft log files", store_id;
            "total_file" => total_file_cnt, "total_size" => object.len(),
            "cached_file" => cached_file_cnt, "cached_size" => cached_file_size);

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
            return backup_callback(task, Err(Error::Backup(msg)), "incr_fail", ob_start_time);
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
                return backup_callback(
                    task,
                    Err(Error::Backup(format!("Backup WAL failed {:?}", e))),
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
                return backup_callback(task, Err(Error::Backup(err)), "incr_fail", ob_start_time);
            }
            backup_callback(task, Ok(backup_meta), "incr_success", ob_start_time);
        });
    }

    fn sync_files(&mut self) {
        if self.files_to_sync.is_empty() {
            return;
        }
        let group_size = (self.files_to_sync.len() / self.sync_concurrency) + 1;
        let mut group_files = vec![];
        let mut join_handles = vec![];
        for file in self.files_to_sync.drain(..) {
            group_files.push(file);
            if group_files.len() == group_size {
                let files = mem::take(&mut group_files);
                let handle = thread::spawn(|| {
                    for file in files {
                        file.sync_data().unwrap()
                    }
                });
                join_handles.push(handle);
            }
        }
        if !group_files.is_empty() {
            for file in group_files {
                file.sync_data().unwrap()
            }
        }
        for handle in join_handles {
            handle.join().unwrap();
        }
    }
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

#[derive(Debug)]
pub(crate) enum CompactTask {
    Compact { epoch_id: u32 },
    HeavyBackup(BackupTask),
    Close { force: bool },
    Snapshot,
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
    pub callback: Box<dyn FnOnce(Result<StoreBackupMeta>) + Send>,
    pub(crate) file_off: u64,
    pub(crate) config: BackupConfig,
}

impl fmt::Debug for BackupTask {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BackupTask")
            .field("file_off", &self.file_off)
            .field("config", &self.config)
            .finish()
    }
}

impl BackupTask {
    pub fn new(
        object_storage: Box<dyn ObjectStorage>,
        callback: Box<dyn FnOnce(Result<StoreBackupMeta>) + Send>,
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

pub(crate) fn backup_callback(
    task: BackupTask,
    ret: Result<StoreBackupMeta>,
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

struct RlogCache {
    size_threshold: usize,
    inner: Option<QuickCache<RlogCacheKey, RlogCacheValue, RlogWeighter>>,
}

#[derive(Clone)]
struct RlogWeighter;

impl quick_cache::Weighter<RlogCacheKey, RlogCacheValue> for RlogWeighter {
    fn weight(&self, _: &RlogCacheKey, val: &RlogCacheValue) -> u64 {
        val.data.len() as u64
    }
}

#[derive(PartialEq, Eq, Hash)]
struct RlogCacheKey {
    peer_id: u64,
    first_index: u64,
}

struct RlogCacheValue {
    last_index: u64, // For verification.
    data: Bytes,
}

impl RlogCache {
    fn new(capacity: usize, size_threshold: usize) -> Self {
        // Consider `size_threshold / 2` as average rlog size.
        let inner =
            QuickCache::with_weighter(capacity * 2 / size_threshold, capacity as u64, RlogWeighter);
        Self {
            size_threshold,
            inner: Some(inner),
        }
    }

    fn none() -> Self {
        Self {
            size_threshold: 0,
            inner: None,
        }
    }

    fn cache_size(&self) -> u64 {
        self.inner.as_ref().map(|x| x.weight()).unwrap_or_default()
    }

    fn add_rlog(&mut self, peer_id: u64, rlog: &RaftLogFile, rlog_data: &[u8]) -> bool /* is_cached */
    {
        if let Some(inner) = self.inner.as_mut() {
            let k = RlogCacheKey {
                peer_id,
                first_index: rlog.first_index,
            };

            if rlog_data.len() > self.size_threshold {
                // The rlog cache must always be overwritten, as the content may be changed.
                // See https://github.com/tidbcloud/cloud-storage-engine/issues/2154.
                inner.remove(&k);
                return false;
            }

            let v = RlogCacheValue {
                last_index: rlog.last_index,
                data: Bytes::copy_from_slice(rlog_data),
            };
            inner.insert(k, v);
            true
        } else {
            false
        }
    }

    #[inline]
    fn get_rlog_size(&self, peer_id: u64, rlog: &RaftLogFile) -> Option<usize> {
        self.get_rlog_data(peer_id, rlog).map(|x| x.len())
    }

    fn get_rlog_data(&self, peer_id: u64, rlog: &RaftLogFile) -> Option<&[u8]> {
        let inner = self.inner.as_ref()?;
        let k = RlogCacheKey {
            peer_id,
            first_index: rlog.first_index,
        };
        let v = inner.get(&k)?;
        if v.last_index == rlog.last_index {
            Some(&v.data)
        } else {
            debug_assert!(
                false,
                "rlog cache: last_index not match, peer_id {}, rlog {:?}, cache.last_index {}",
                peer_id, rlog, v.last_index
            );
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{
        collections::HashMap,
        fs, iter,
        sync::atomic::{AtomicU32, AtomicU64},
    };

    use bytes::Buf;
    use kvproto::raft_serverpb::RegionLocalState;
    use protobuf::Message;
    use rand::{distributions::Alphanumeric, Rng};
    use rfenginepb::{ChangeSet, PeerState, StoreBackupMeta, StoreRaftLogBackupMeta};
    use tikv_util::defer;

    use super::*;
    use crate::{
        dfs_worker,
        log_batch::{RaftLogOp, RaftLogs},
        manifest::{persist_change_set, Manifest},
        raft_log_file_name, region_state_key, store_raft_log_file_key,
        test_util::{get_txn_endkey_prefix, get_txn_startkey_prefix, init_logger},
        write_batch::PeerBatch,
        CompactWorker, RfEngine, RfEngineConfig, WalWriter, WriterType,
    };

    const RANDOM_STR_MAX_LEN: usize = 1024;

    fn generate_random_str() -> Vec<u8> {
        let mut rng = rand::thread_rng();
        let len = rng.gen::<usize>() % RANDOM_STR_MAX_LEN + 1;
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

    #[rstest::rstest]
    #[case(false, false)]
    #[case::with_cache(true, false)]
    #[case::with_cache_and_compress(true, true)]
    fn test_backup_raft_log_files(#[case] with_cache: bool, #[case] with_compress: bool) {
        init_logger();
        let tmp_dir = tempfile::tempdir().unwrap();
        let tmp_path = tmp_dir.path();
        defer!(fs::remove_dir_all(tmp_path).unwrap());
        let (_, rx) = tikv_util::mpsc::unbounded();
        let engine_id = 999;
        let manifest = Manifest::open(tmp_path, AtomicU64::new(engine_id).into()).unwrap();
        let mut worker = CompactWorker::new(
            tmp_path.to_path_buf(),
            rx,
            manifest,
            AtomicU32::new(0).into(),
            None,
            None,
            dfs_worker::Healthy::default(),
            1,
        );
        worker.rlog_cache = if with_cache {
            RlogCache::new(RANDOM_STR_MAX_LEN * 80, RANDOM_STR_MAX_LEN / 2)
        } else {
            RlogCache::none()
        };
        worker.rlog_compression_type = if with_compress {
            CompressionType::Lz4Compression
        } else {
            CompressionType::NoCompression
        };

        let epoch = 990;
        let mut cs = ChangeSet::new();
        cs.epoch_id = epoch;
        let mut peer_rlog_files = HashMap::new();
        let mut total_file_cnt = 0;
        let mut cached_file_cnt = 0;
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
                let file_name = raft_log_file_name(tmp_path, peer_id, first_index, last_index);
                let content = generate_random_str();
                fs::write(file_name, &content).unwrap();
                let is_cached = worker
                    .rlog_cache
                    .add_rlog(peer_id, &raft_log_file, &content);
                cached_file_cnt += is_cached as usize;
                meta_pb.mut_files().push(raft_log_file);
                files.push(content);
                total_file_cnt += 1;
            }
            peer_rlog_files.insert(peer_id, files);
            cs.mut_peers().push(meta_pb);
        }
        assert!(!with_cache || cached_file_cnt > 0);
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
        assert_eq!(raft_meta.get_header().version, 2);
        let header = raft_meta.get_header();
        if with_compress {
            assert_eq!(header.compression_type, 1);
        } else {
            assert_eq!(header.compression_type, 0);
        };

        let mut ret_file_cnt = 0;
        for (keyspace_id, keyspace_data) in raft_meta.take_raft_logs() {
            assert_eq!(keyspace_id, keyspace_data.get_keyspace_id());
            for file in keyspace_data.get_files() {
                let peer_id = file.peer_id;
                assert_eq!(keyspace_id as u64, peer_id / 5);
                let data = if with_compress {
                    decompress_lz4(&object.chunk()[file.start_off as usize..file.end_off as usize])
                        .unwrap()
                } else {
                    object.chunk()[file.start_off as usize..file.end_off as usize].to_vec()
                };
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
                    &file_data, &data,
                    "peer id {}, index {}-{}",
                    peer_id, file.first_index, file.last_index
                );
                ret_file_cnt += 1;
            }
        }
        assert_eq!(ret_file_cnt, total_file_cnt);
    }

    fn generate_rlog_files(
        worker: &mut CompactWorker,
        raft_logs: &mut RaftLogs,
        peer_id: u64,
        region_id: u64,
        first_index: u64,
        last_index: u64,
    ) -> (rfenginepb::RaftLogFile, bool /* is_cached */) {
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

    #[rstest::rstest]
    #[case(false, false)]
    #[case::with_cache(true, false)]
    #[case::with_cache_and_compress(true, true)]
    fn test_backup_and_load_raft_log_files(#[case] with_cache: bool, #[case] with_compress: bool) {
        init_logger();
        let tmp_dir = tempfile::tempdir().unwrap();
        let tmp_path = tmp_dir.path();
        defer!(fs::remove_dir_all(tmp_path).unwrap());
        let (_, rx) = tikv_util::mpsc::unbounded();
        let engine_id = 1999;
        let manifest = Manifest::open(tmp_path, AtomicU64::new(engine_id).into()).unwrap();
        let mut worker = CompactWorker::new(
            tmp_path.to_path_buf(),
            rx,
            manifest,
            AtomicU32::new(0).into(),
            None,
            None,
            dfs_worker::Healthy::default(),
            1,
        );
        worker.rlog_cache = if with_cache {
            RlogCache::new(RANDOM_STR_MAX_LEN * 100 * 5, RANDOM_STR_MAX_LEN * 100 / 2)
        } else {
            RlogCache::none()
        };
        worker.rlog_compression_type = if with_compress {
            CompressionType::Lz4Compression
        } else {
            CompressionType::NoCompression
        };

        let mut cs = ChangeSet::new();
        let mut peer_data_map = HashMap::new();
        let peers_range = 100..150;
        let mut total_file_cnt = 0;
        let mut cached_file_cnt = 0;
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
                let (raft_log_file, is_cached) = generate_rlog_files(
                    &mut worker,
                    &mut peer_raft_log,
                    peer_id,
                    region_id,
                    first_index,
                    last_index,
                );
                meta_pb.mut_files().push(raft_log_file);
                total_file_cnt += 1;
                cached_file_cnt += is_cached as usize;
            }
            cs.mut_peers().push(meta_pb);
            cs.epoch_id += 1;
            peer_data_map.insert(peer_id, peer_raft_log);
        }
        assert!(!with_cache || cached_file_cnt > 0);

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
        assert_eq!(raft_meta.get_header().version, 2);
        let header = raft_meta.get_header();
        if with_compress {
            assert_eq!(header.compression_type, 1);
        } else {
            assert_eq!(header.compression_type, 0);
        };
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
                let data = if with_compress {
                    decompress_lz4(&object.chunk()[file.start_off as usize..file.end_off as usize])
                        .unwrap()
                } else {
                    object.chunk()[file.start_off as usize..file.end_off as usize].to_vec()
                };
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
            cfg.write_throttle_duration.0,
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
        assert_eq!(CompactWorker::get_keyspace_id_from_peer(1, &peer_meta), 0);
        write_keyspace_state(&mut peer_meta, 100);
        write_keyspace_state(&mut peer_meta, 200);
        assert_eq!(CompactWorker::get_keyspace_id_from_peer(1, &peer_meta), 200);
    }
}
