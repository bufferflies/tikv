// Copyright 2021 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    collections::{BTreeMap, HashMap, HashSet},
    fmt::{Display, Formatter},
    fs,
    fs::{create_dir_all, File, OpenOptions},
    ops::{Deref, DerefMut},
    os::unix::fs::{FileExt, MetadataExt},
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering},
        Arc, Mutex, RwLock,
    },
    thread::{self, JoinHandle},
};

use bytes::{Buf, Bytes};
use dashmap::mapref::one::Ref;
use engine_traits::{GetObjectOptions, ObjectStorage};
use file_system::open_direct_file;
use kvengine::dfs::DFSConfig;
use kvproto::raft_serverpb::{self, StoreIdent};
use protobuf::Message;
use raft_proto::{eraftpb, eraftpb::Entry};
use rfenginepb::{ClusterBackupMeta, StoreBackupMeta, StoreRaftLogBackupMeta};
use tikv_util::{
    error, info, mpsc::Sender, panic_mark_dfs_worker_file_exists, time::Instant, warn,
};

use crate::{
    config::Config,
    log_batch::{RaftLogBlock, RaftLogs},
    manifest::{manifest_path, persist_change_set, Manifest},
    metrics::*,
    write_batch::{PeerBatch, WriteBatch},
    *,
};

pub const TRUNCATE_ALL_INDEX: u64 = u64::MAX;
pub const MAX_EPOCH_BACKWARD: u32 = 100;

/// `RfEngine` is a persistent storage engine for multi-raft logs.
/// It stores part of raft logs and states(key/value pair) in memory and
/// persists them to disk.
///
/// A typical directory structure is:
///   .
///   ├── {epoch}.wal
///   ├── {epoch}.states
///   ├── {epoch}_{region_id}_{first_log_index}_{last_log_index}.rlog
///   ├── {epoch}_{region_id}_{first_log_index}_{last_log_index}.rlog
///   ├── ...
///   └── recycle
///       └── {epoch}.wal
///       └── {epoch}.wal
///       └── ...
///
/// # Memory Layout
///
/// `RfEngine` contains all raft group states and non-truncated logs in memory,
/// so that it can get raft logs quickly.
///
/// # WAL
///
/// `RfEngine` writes all raft groups' logs and states to a WAL file
/// sequentially. When the WAL file size exceeds the threshold, it triggers
/// rotation and switching to a new WAL file. The name of a WAL file is
/// `{epoch}.wal`. Epoch increases when rotating.
///
/// ## Rotation
///
/// Rotation splits the data of a WAL file to several files:
///   - `{epoch}.states`: Contains **all** raft groups states, not just states
///     in the corresponding WAL file.
///   The old states file will be removed after rewriting.
///
///   - `{epoch}_{region_id}_{first_log_index}_{last_log_index}.rlog`: Contains
///     logs in
///   [first_log_index, last_log_index) of a single raft group.
///
/// After splitting, the WAL file is moved to the `recycle` directory for later
/// use. `RfEngine` recycles old WAL files for better I/O performance. To
/// distinguish between old data and new data, the data format of WAL contains
/// epoch, i.e., valid data's epoch equals the epoch in the WAL file name.
///
/// # Garbage Collection
///
/// Raft logs that has been applied and persisted to FSM can be truncated. All
/// in-memory logs and `rlog` files before the truncated index will be removed.
#[derive(Clone)]
pub struct RfEngine {
    core: Arc<RfEngineCore>,
}

impl Deref for RfEngine {
    type Target = RfEngineCore;

    fn deref(&self) -> &Self::Target {
        &self.core
    }
}

impl RfEngine {
    pub fn open(
        dir: &Path,
        cfg: &Config,
        data_dir: Option<&Path>, // for check panic mark file exists
        dfs_conf: Option<DFSConfig>,
    ) -> Result<Self> {
        let core = RfEngineCore::open(dir, cfg, data_dir, dfs_conf)?;
        Ok(Self {
            core: Arc::new(core),
        })
    }
}

pub struct RfEngineCore {
    pub dir: PathBuf,

    pub wal_sync_dir: Option<PathBuf>,

    pub(crate) writer: Mutex<WalWriter>,

    pub(crate) peers: dashmap::DashMap<u64, RwLock<PeerData>>,

    pub(crate) dependants: dashmap::DashMap<u64, RwLock<HashSet<u64>>>,

    pub(crate) task_sender: Sender<Task>,

    pub(crate) worker_handle: Mutex<WorkerHandle>,

    pub(crate) engine_id: Arc<AtomicU64>,

    pub(crate) lightweight: bool,

    _lock: fslock::LockFile, // hold lock to avoid release
}

pub(crate) struct WorkerHandle {
    task_sender: Sender<Task>,
    handle: Option<JoinHandle<()>>,
}

impl RfEngineCore {
    fn open(
        dir: &Path,
        cfg: &Config,
        data_dir: Option<&Path>,
        dfs_conf: Option<DFSConfig>,
    ) -> Result<Self> {
        let wal_size = cfg.target_file_size.0 as usize;
        let compression_threshold = cfg.batch_compression_threshold.0 as usize;
        let wal_sync_dir = (!cfg.wal_sync_dir.is_empty()).then(|| PathBuf::from(&cfg.wal_sync_dir));
        init_wal_files(dir, wal_sync_dir.as_ref())?;

        // Lock the rfengine directory to prevent concurrent opening.
        let lock_path = dir.join("LOCK");
        let mut lock = fslock::LockFile::open(&lock_path)?;
        if !lock.try_lock()? {
            panic!("rfengine lock failed, maybe already used by another process");
        }

        let engine_id = Arc::new(AtomicU64::new(0));
        let manifest = Manifest::open(dir, engine_id.clone())?;
        let (tx, rx) = tikv_util::mpsc::unbounded();
        let compacted_epoch = Arc::new(AtomicU32::new(manifest.epoch_id));
        let wal_dir = wal_sync_dir.as_deref().unwrap_or(dir);
        let writer_type = if cfg.cli_mode {
            WriterType::CliMode
        } else {
            WriterType::Sync
        };
        let writer = WalWriter::new(
            wal_dir,
            wal_size,
            compression_threshold,
            compacted_epoch.clone(),
            writer_type,
        );

        let dfs_worker_healthy = Arc::new(AtomicBool::new(true));
        let mut en = Self {
            dir: dir.to_owned(),
            wal_sync_dir,
            peers: Default::default(),
            dependants: Default::default(),
            writer: Mutex::new(writer),
            task_sender: tx.clone(),
            worker_handle: Mutex::new(WorkerHandle {
                task_sender: tx.clone(),
                handle: None,
            }),
            engine_id,
            lightweight: cfg.lightweight_backup,
            _lock: lock,
        };
        let async_offset = en.load(&manifest)?;
        {
            let async_wal_writer = if en.is_async_wal_enabled() {
                let mut async_wal_writer = WalWriter::new(
                    dir,
                    wal_size,
                    compression_threshold,
                    compacted_epoch.clone(),
                    WriterType::Async,
                );
                async_wal_writer.open_file(manifest.epoch_id + 1, async_offset)?;
                Some(async_wal_writer)
            } else {
                None
            };

            let object_storage_config: Option<ObjectStorageConfig> = if cfg.lightweight_backup {
                if data_dir.is_some() && panic_mark_dfs_worker_file_exists(data_dir.unwrap()) {
                    // If panic_mark_dfs_worker_file exists, skip init dfs worker thread and mark
                    // dfs worker unhealthy.
                    dfs_worker_healthy.store(false, Ordering::Release);
                    error!(
                        "lightweight backup is enabled, but panic_mark_dfs_worker_file exists, skip init dfs worker thread"
                    );
                    None
                } else {
                    Some(ObjectStorageConfig::new(
                        dir.to_owned(),
                        cfg.wal_chunk_target_file_size.0 as usize,
                        CompressionType::Lz4Compression,
                        dfs_conf.unwrap(),
                    ))
                }
            } else {
                None
            };

            let mut worker = Worker::new(
                dir.to_owned(),
                rx,
                tx,
                manifest,
                compacted_epoch,
                async_wal_writer,
                object_storage_config,
                dfs_worker_healthy,
            );
            let join_handle = thread::spawn(move || worker.run());
            en.worker_handle.lock().unwrap().handle = Some(join_handle);
        }

        Ok(en)
    }

    pub(crate) fn wal_dir(&self) -> &Path {
        self.wal_sync_dir.as_ref().unwrap_or(&self.dir)
    }

    pub(crate) fn get_or_init_peer_data(
        &self,
        peer_id: u64,
        region_id: u64,
    ) -> Ref<'_, u64, RwLock<PeerData>> {
        self.peers
            .entry(peer_id)
            .or_insert_with(|| RwLock::new(PeerData::new(peer_id, region_id)))
            .downgrade()
    }

    /// Applies and persists the write batch.
    pub fn write(&self, mut wb: WriteBatch) -> Result<usize> {
        self.apply(&mut wb);
        self.persist(wb)
    }

    /// Applies the write batch to memory without persisting it to WAL.
    pub fn apply(&self, wb: &mut WriteBatch) {
        let timer = Instant::now_coarse();
        let mut truncated_logs = vec![];
        for (&peer_id, batch_data) in &wb.peers {
            let region_id = batch_data.meta.region_id;
            let peer_data = self.get_or_init_peer_data(peer_id, region_id);
            let mut peer_data = peer_data.write().unwrap();
            let truncated = peer_data.apply(batch_data);
            drop(peer_data);
            if !truncated.is_empty() {
                truncated_logs.push(truncated);
            }
        }
        if !truncated_logs.is_empty() {
            self.task_sender
                .send(Task::Truncates(truncated_logs))
                .unwrap();
        }
        ENGINE_APPLY_DURATION_HISTOGRAM.observe(timer.saturating_elapsed_secs());
    }

    /// Persists the write batch to WAL. It can be used in another thread to
    /// implement async I/O, i.e., call `apply` in the main thread and call
    /// `persist` in the I/O thread.
    pub fn persist(&self, wb: WriteBatch) -> Result<usize> {
        let timer = Instant::now_coarse();
        let mut writer = self.writer.lock().unwrap();
        let epoch_id = writer.epoch_id;
        let (size, rotated) = writer.write_batch(&wb)?;
        if rotated {
            self.task_sender.send(Task::Rotate { epoch_id }).unwrap();
        }
        if self.is_async_wal_enabled() {
            self.task_sender.send(Task::Write { wb }).unwrap();
        }
        ENGINE_PERSIST_DURATION_HISTOGRAM.observe(timer.saturating_elapsed_secs());
        Ok(size)
    }

    pub fn is_empty(&self) -> bool {
        self.peers.is_empty()
    }

    pub fn get_term(&self, peer_id: u64, index: u64) -> Option<u64> {
        self.peers
            .get(&peer_id)
            .and_then(|data| data.read().unwrap().term(index))
    }

    pub fn get_last_index(&self, peer_id: u64) -> Option<u64> {
        self.peers
            .get(&peer_id)
            .map(|data| data.read().unwrap().raft_logs.last_index())
            .and_then(|index| if index != 0 { Some(index) } else { None })
    }

    pub fn get_state(&self, peer_id: u64, key: &[u8]) -> Option<Bytes> {
        self.peers.get(&peer_id).and_then(|data| {
            data.read().unwrap().get_state(key).and_then(|val| {
                // TODO: seems it's impossible.
                if !val.is_empty() {
                    Some(val.clone())
                } else {
                    None
                }
            })
        })
    }

    /// Get the value of the last state key with the `prefix`. `prefix` must be
    /// non-empty.
    pub fn get_last_state_with_prefix(&self, peer_id: u64, prefix: &[u8]) -> Option<Bytes> {
        debug_assert!(!prefix.is_empty());
        let peer_data = self.peers.get(&peer_id)?;
        let peer_data = peer_data.read().unwrap();

        let mut end_prefix = prefix.to_vec();
        end_prefix[prefix.len() - 1] += 1;
        let range = Bytes::copy_from_slice(prefix)..Bytes::from(end_prefix);
        peer_data
            .meta
            .states
            .range(range)
            .rev()
            .next()
            .map(|(_, v)| v.clone())
    }

    /// Iterates states of the region in order or in desc order if `desc` is
    /// true until `f` returns error.
    pub fn iterate_peer_states<F>(&self, peer_id: u64, desc: bool, mut f: F)
    where
        F: FnMut(&[u8], &[u8]),
    {
        let peer_data = self.peers.get(&peer_id);
        let peer_data = match &peer_data {
            Some(data) => data.read().unwrap(),
            None => return,
        };

        let states = &peer_data.meta.states;
        if desc {
            for (k, v) in states.iter().rev() {
                f(k.chunk(), v.chunk());
            }
        } else {
            for (k, v) in states.iter() {
                f(k.chunk(), v.chunk());
            }
        }
    }

    /// Iterates stats of all regions in order or in desc order if `desc` is
    /// true and breaks one regions iteration if `f` returns false.
    pub fn iterate_all_states<F>(&self, desc: bool, mut f: F)
    where
        F: FnMut(u64, u64, &[u8], &[u8]) -> bool,
    {
        self.peers.iter().for_each(|data| {
            let data = data.read().unwrap();
            if data.truncated_idx == TRUNCATE_ALL_INDEX {
                return;
            }
            let peer_id = data.peer_id;
            let region_id = data.region_id;
            if desc {
                data.states
                    .iter()
                    .rev()
                    .take_while(|(k, v)| f(peer_id, region_id, k, v))
                    .count();
            } else {
                data.states
                    .iter()
                    .take_while(|(k, v)| f(peer_id, region_id, k, v))
                    .count();
            }
        });
    }

    pub fn stop_worker(&self, force: bool) {
        let mut handle_ref = self.worker_handle.lock().unwrap();
        if let Some(h) = handle_ref.handle.take() {
            handle_ref.task_sender.send(Task::Close { force }).unwrap();
            h.join().unwrap();
        }
    }

    /// After split and before the new region is initially flushed, the old
    /// region's raft log can not be truncated, otherwise, it would not be
    /// able to recover the new region. So we can call `add_dependent` after
    /// split to protect the raft log. After the new region is initially
    /// flushed or re-ingested or destroyed, call `remove_dependent` to
    /// resume truncating the raft log.
    pub fn add_dependent(&self, region_id: u64, dependent_id: u64) {
        let hs_ref = self.dependants.entry(region_id).or_default();
        let mut hs = hs_ref.write().unwrap();
        hs.insert(dependent_id);
        let len = hs.len();
        drop(hs);
        drop(hs_ref);
        let tag = PeerTag::new(self.get_engine_id(), region_id);
        info!(
            "{} add dependent {}, dependents_len {}",
            tag, dependent_id, len
        );
    }

    pub fn remove_dependent(&self, region_id: u64, dependent_id: u64) -> usize {
        self.dependants
            .get(&region_id)
            .map(|hs| {
                let len = {
                    let mut hs = hs.write().unwrap();
                    hs.remove(&dependent_id);
                    hs.len()
                };
                let tag = PeerTag::new(self.get_engine_id(), region_id);
                info!(
                    "{} remove dependent {}, dependents_len {}",
                    tag, dependent_id, len
                );
                len
            })
            .unwrap_or_default()
    }

    pub fn with_dependents(&self, region_id: u64, f: impl FnOnce(&HashSet<u64>)) {
        if let Some(hs) = self.dependants.get(&region_id) {
            f(&hs.read().unwrap());
        }
    }

    pub fn has_dependents(&self, region_id: u64) -> bool {
        self.dependants
            .get(&region_id)
            .map_or(false, |hs| !hs.read().unwrap().is_empty())
    }

    /// Dumps the state of the engine.
    pub fn get_engine_stats(&self) -> EngineStats {
        let mut total_mem_size = 0;
        let mut total_mem_entries = 0;
        let mut peers_stats = self
            .peers
            .iter()
            .map(|data| {
                let peer_stats = data.read().unwrap().get_stats();
                total_mem_size += peer_stats.size;
                total_mem_entries += peer_stats.num_logs;
                peer_stats
            })
            .collect::<Vec<PeerStats>>();
        peers_stats.sort_by(|a, b| (b.size).cmp(&a.size));
        peers_stats.truncate(10);

        let mut disk_size = 0;
        let mut num_files = 0;
        if let Ok(read_dir) = self.dir.read_dir() {
            for e in read_dir.flatten() {
                if let Ok(m) = e.metadata() {
                    num_files += 1;
                    disk_size += m.size();
                }
            }
        }
        EngineStats {
            total_mem_size,
            total_mem_entries,
            disk_size,
            num_files,
            top_10_size_peers: peers_stats,
        }
    }

    /// Dumps the state of the region.
    pub fn get_peer_stats(&self, peer_id: u64) -> PeerStats {
        self.peers
            .get(&peer_id)
            .map(|data| data.read().unwrap().get_stats())
            .unwrap_or_default()
    }

    /// Returns the index that truncating to the given index can limit the
    /// memory usage to size.
    pub fn index_to_truncate_to_size(&self, peer_id: u64, size: usize) -> u64 {
        self.peers
            .get(&peer_id)
            .map(|data| {
                data.read()
                    .unwrap()
                    .raft_logs
                    .index_to_truncate_to_size(size)
            })
            .unwrap_or_default()
    }

    pub fn set_engine_id(&self, engine_id: u64) {
        self.engine_id.store(engine_id, Ordering::Release)
    }

    pub fn get_engine_id(&self) -> u64 {
        self.engine_id.load(Ordering::Acquire)
    }

    pub fn get_region_peer_map(&self) -> HashMap<u64, u64> {
        let mut region_to_peer = HashMap::with_capacity(self.peers.len());
        let mut id_pairs = Vec::with_capacity(self.peers.len());
        for peer_ref in self.peers.iter() {
            let peer_data = peer_ref.read().unwrap();
            let is_truncated = peer_data.truncated_idx == TRUNCATE_ALL_INDEX;
            id_pairs.push((peer_data.peer_id, peer_data.region_id, is_truncated));
        }
        // ensure the newer peer_id appear after the older peer_id, so it can replace
        // older.
        id_pairs.sort_by(|(peer_a, ..), (peer_b, ..)| peer_a.cmp(peer_b));
        for (peer_id, region_id, truncated) in id_pairs {
            if truncated {
                // The newer peer is already destroyed, the old peer is invalid too.
                region_to_peer.remove(&region_id);
            } else {
                // new peer_id replaces the older peer_id.
                region_to_peer.insert(region_id, peer_id);
            }
        }
        region_to_peer
    }

    pub fn get_raft_entry(&self, peer_id: u64, index: u64) -> Option<Entry> {
        self.peers
            .get(&peer_id)
            .and_then(|data| data.read().unwrap().get(index))
    }

    pub fn fetch_raft_entries_to(
        &self,
        peer_id: u64,
        low: u64,
        high: u64,
        max_size: Option<usize>, // size limit of fetched entries
        buf: &mut Vec<Entry>,
    ) -> engine_traits::Result<usize> /* entry count */ {
        if high <= low {
            return Ok(0);
        }
        let old_len = buf.len();
        let peer_data = self
            .peers
            .get(&peer_id)
            .ok_or(engine_traits::Error::EntriesCompacted)?;
        let peer_data = peer_data.read().unwrap();
        if low <= peer_data.meta.truncated_idx {
            return Err(engine_traits::Error::EntriesCompacted);
        }

        let timer = Instant::now_coarse();
        let mut total_size = 0;
        for i in low..high {
            let entry = peer_data
                .get(i)
                .ok_or(engine_traits::Error::EntriesUnavailable)?;
            total_size += entry.compute_size() as usize;
            buf.push(entry);
            if max_size.map_or(false, |s| total_size >= s) {
                // At least return one entry regardless of size limit.
                break;
            }
        }
        ENGINE_FETCH_ENTRIES_DURATION_HISTOGRAM.observe(timer.saturating_elapsed_secs());
        Ok(buf.len() - old_len)
    }

    pub fn dump_wal_chunk(
        &self,
        epoch_id: u32,
        start_off: u64,
        end_off: u64,
        callback: Box<dyn FnOnce(Result<Bytes>) + Send>,
    ) {
        self.task_sender
            .send(Task::Dump {
                epoch_id,
                start_off,
                end_off,
                callback,
            })
            .unwrap();
    }

    pub fn backup(&self, mut task: BackupTask) {
        if !self.is_async_wal_enabled() {
            // Note: when async wal is enabled, `file_off` is acquired from
            // `async_wal_writer` in worker.
            let writer = self.writer.lock().unwrap();
            task.file_off = writer.file_off;
        }
        self.task_sender.send(Task::Backup(task)).unwrap();
    }

    pub(crate) fn is_async_wal_enabled(&self) -> bool {
        self.wal_sync_dir.is_some()
    }

    pub fn is_lightweight_backup_enabled(&self) -> bool {
        self.lightweight
    }

    pub fn load_region_state(
        &self,
        peer_id: u64,
        version: u64,
    ) -> Option<raft_serverpb::RegionLocalState> {
        let region_state_key = region_state_key(version);
        let region_state_val = self.get_state(peer_id, &region_state_key)?;
        let mut region_state = raft_serverpb::RegionLocalState::new();
        region_state.merge_from_bytes(&region_state_val).unwrap();
        Some(region_state)
    }
}

fn restore_all_raft_logs(
    object_storage: &Arc<dyn ObjectStorage>,
    store_meta: &StoreBackupMeta,
    dir: &Path,
    snapshot_rlog: Option<String>,
) {
    let store_id = store_meta.store_id;
    let raft_file_key = snapshot_rlog
        .unwrap_or_else(|| store_raft_log_file_key(store_id, store_meta.get_manifest().epoch_id));
    let raft_file = object_storage
        .get_objects(vec![(raft_file_key, GetObjectOptions::default())])
        .unwrap();
    let (_, rlog_data) = raft_file.first().unwrap();
    let mut raft_meta = StoreRaftLogBackupMeta::default();
    let size = rlog_data.len();
    debug_assert!(size as u64 > store_meta.raft_meta_start_off);
    raft_meta
        .merge_from_bytes(&rlog_data.chunk()[store_meta.raft_meta_start_off as usize..size])
        .unwrap();
    for (_, keyspace_meta) in raft_meta.raft_logs {
        for file in keyspace_meta.get_files() {
            let path = raft_log_file_name(dir, file.peer_id, file.first_index, file.last_index);
            fs::write(
                path,
                &rlog_data.chunk()[file.start_off as usize..file.end_off as usize],
            )
            .unwrap();
        }
    }
}

fn restore_keyspace_raft_logs(
    object_storage: &Arc<dyn ObjectStorage>,
    store_meta: &StoreBackupMeta,
    dir: &Path,
    keyspace_id: u32,
    snapshot_rlog: Option<String>,
) {
    let store_id = store_meta.store_id;
    let raft_file_key = snapshot_rlog
        .unwrap_or_else(|| store_raft_log_file_key(store_id, store_meta.get_manifest().epoch_id));
    let option = GetObjectOptions {
        start_off: store_meta.raft_meta_start_off,
        end_off: None,
    };
    let raft_meta_data = object_storage
        .get_objects(vec![(raft_file_key.clone(), option)])
        .unwrap();
    let (_, rlog_meta_data) = raft_meta_data.first().unwrap();
    let mut raft_meta = StoreRaftLogBackupMeta::default();
    raft_meta.merge_from_bytes(rlog_meta_data.chunk()).unwrap();
    let keyspace_meta = raft_meta.raft_logs.get(&keyspace_id);
    if keyspace_meta.is_none() {
        info!("There is no raft log files for keyspace {}", keyspace_id);
        return;
    }
    let raft_files = keyspace_meta.unwrap().get_files();
    info!(
        "Restore {} raft files for keyspace {}",
        raft_files.len(),
        keyspace_id
    );
    if raft_files.is_empty() {
        return;
    }
    let option = GetObjectOptions {
        start_off: raft_files.first().unwrap().start_off,
        end_off: Some(raft_files.last().unwrap().end_off),
    };
    let keyspace_raft_data = object_storage
        .get_objects(vec![(raft_file_key, option)])
        .unwrap();
    let (_, keyspace_raft_data) = keyspace_raft_data.first().unwrap();
    let mut cur_offset = 0;
    for file in raft_files {
        let path = raft_log_file_name(dir, file.peer_id, file.first_index, file.last_index);
        let data_len = (file.end_off - file.start_off) as usize;
        fs::write(
            path,
            &keyspace_raft_data.chunk()[cur_offset..cur_offset + data_len],
        )
        .unwrap();
        cur_offset += data_len;
    }
}

pub fn find_latest_snapshot(
    object_storage: Arc<dyn ObjectStorage>,
    prefix: &str,
    store_meta: &StoreBackupMeta,
) -> Result<Option<String>> {
    let epoch_id = store_meta.get_epoch();
    let start_epoch = if epoch_id > MAX_EPOCH_BACKWARD {
        epoch_id - MAX_EPOCH_BACKWARD
    } else {
        1
    };
    let store_id = store_meta.get_store_id();

    // Find the latest snapshot smaller than cluster_backup epoch.
    let snapshot = match object_storage.list_objects(
        &snapshot_rlog_key_suffix(start_epoch - 1),
        Some(&snapshot_rlog_key_prefix(store_id)),
        Some(MAX_EPOCH_BACKWARD),
    ) {
        Ok((objects, _)) => {
            if objects.is_empty() {
                None
            } else {
                objects.into_iter().rev().find(|obj| {
                    let obj_epoch = parse_epoch_from_snapshot_key(Some(obj.key.deref())).unwrap();
                    obj_epoch < epoch_id
                })
            }
        }
        Err(err) => {
            error!("failed to list snapshot full backup files: {}", err);
            None
        }
    };
    Ok(snapshot.map(|snap| {
        snap.key
            .strip_prefix(&format!("{}/", prefix))
            .unwrap()
            .to_owned()
    }))
}

// For lightweight restore, find a latest snapshot full backup before
// `cluster_backup.backup_ts` and replay all WAL chunk files from snapshot epoch
// to epoch of the backup.
pub fn lightweight_restore(
    object_storage: Arc<dyn ObjectStorage>,
    prefix: &str,
    cluster_backup: &ClusterBackupMeta,
    store_id: u64,
    dir: &Path,
    keyspace: Option<u32>,
) -> Result<u32> {
    let store_meta = cluster_backup
        .get_stores()
        .iter()
        .find(|x| x.store_id == store_id)
        .ok_or(Error::Other(format!("store {} not found", store_id)))?;
    let epoch_id = store_meta.get_epoch();
    init_wal_files(dir, None).unwrap();

    info!("try to find snapshot before backup epoch {}", epoch_id);
    let snap_key = find_latest_snapshot(object_storage.clone(), prefix, store_meta)?;

    // Fetch store backup meta from object storage.
    let snap_store_meta = match parse_epoch_from_snapshot_key(snap_key.as_deref()) {
        Some(snap_epoch) => {
            info!(
                "fetch last snapshot {} snap epoch {} before backup epoch {}",
                snap_key.as_deref().unwrap(),
                snap_epoch,
                epoch_id
            );
            let store_meta_key = snapshot_store_meta_key(store_id, snap_epoch);
            let store_backup_meta = object_storage
                .get_objects(vec![(store_meta_key, GetObjectOptions::default())])
                .map(|data| {
                    let mut meta = StoreBackupMeta::default();
                    meta.merge_from_bytes(data[0].1.chunk()).unwrap();
                    meta
                })?;

            assert_eq!(snap_epoch, store_backup_meta.get_manifest().epoch_id);
            store_backup_meta
        }
        None => {
            // If no snapshot available and the epoch_id > MAX_EPOCH_BACKWARD, it means
            // data incomplete.
            if epoch_id > MAX_EPOCH_BACKWARD {
                let msg = format!(
                    "no snapshot available from epoch {}",
                    epoch_id - MAX_EPOCH_BACKWARD
                );
                error!("{}", msg);
                return Err(Error::Other(msg));
            }
            info!("no snapshot available from epoch 1, create empty manifest file");
            StoreBackupMeta::default()
        }
    };

    // If no snapshot found, no need to restore raft log files.
    if snap_key.is_some() {
        match keyspace {
            Some(keyspace_id) => {
                info!(
                    "lightweight restore keyspace {} raft log files from snapshot {:?}",
                    keyspace_id, snap_key
                );
                restore_keyspace_raft_logs(
                    &object_storage,
                    &snap_store_meta,
                    dir,
                    keyspace_id,
                    snap_key,
                );
            }
            None => {
                info!(
                    "lightweight restore all raft log files from snapshot {:?}",
                    snap_key
                );
                restore_all_raft_logs(&object_storage, &snap_store_meta, dir, snap_key);
            }
        };
    }

    info!("manifest file path: {:?}", manifest_path(dir));
    let manifest_file = OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(manifest_path(dir))
        .unwrap();

    persist_change_set(&manifest_file, 0, snap_store_meta.get_manifest()).unwrap();

    Ok(snap_store_meta.get_manifest().epoch_id)
}

// If keyspace is none, restore all keyspaces, else, only restore given one.
pub fn restore(
    object_storage: Arc<dyn ObjectStorage>,
    cluster_backup: &ClusterBackupMeta,
    store_id: u64,
    dir: &Path,
    keyspace: Option<u32>,
) {
    let store_meta = cluster_backup
        .get_stores()
        .iter()
        .find(|x| x.store_id == store_id)
        .expect("store not found");
    init_wal_files(dir, None).unwrap();
    let wal_chunks = store_meta.get_wal_chunks();
    if !wal_chunks.is_empty() {
        let keys: Vec<(String, GetObjectOptions)> = wal_chunks
            .iter()
            .map(|chunk| {
                (
                    wal_file_key(store_id, chunk.epoch, chunk.start_off, chunk.end_off),
                    GetObjectOptions::default(),
                )
            })
            .collect();
        let mut objects = object_storage.get_objects(keys).unwrap();
        objects.sort_by(|(a, _), (b, _)| a.cmp(b));
        let wal_path = wal_file_name(dir, store_meta.get_manifest().epoch_id + 1);
        let file = OpenOptions::new().write(true).open(wal_path).unwrap();
        for (i, (_, data)) in objects.into_iter().enumerate() {
            file.write_at(&data, store_meta.get_wal_chunks()[i].start_off)
                .unwrap();
        }
        let end_off = wal_chunks.last().unwrap().end_off;
        let eof = vec![0u8; 4096];
        file.write_at(&eof, end_off).unwrap();
        file.sync_data().unwrap();
    }
    match keyspace {
        Some(keyspace_id) => {
            restore_keyspace_raft_logs(&object_storage, store_meta, dir, keyspace_id, None)
        }
        None => restore_all_raft_logs(&object_storage, store_meta, dir, None),
    }
    let manifest_file = OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(manifest_path(dir))
        .unwrap();
    if store_meta.has_manifest() {
        persist_change_set(&manifest_file, 0, store_meta.get_manifest()).unwrap();
    }
}

pub(crate) fn init_wal_files(dir: &Path, wal_sync_dir: Option<&PathBuf>) -> Result<()> {
    if !dir.exists() {
        create_dir_all(dir)?;
    }
    open_wal_files(dir)?;
    if wal_sync_dir.is_none() {
        return Ok(());
    }
    let wal_sync_dir = wal_sync_dir.unwrap();
    if !wal_sync_dir.exists() {
        create_dir_all(wal_sync_dir)?;
    }
    // upgrade_mark file is used to make the upgrade procedure idempotent.
    // In case the upgrade process is interrupted, we can resume it later.
    let upgrade_mark_file = upgrade_mark_file_path(dir);
    if !upgrade_mark_file.exists() {
        if all_wal_files_exists(wal_sync_dir.as_path()) {
            return Ok(()); // already upgraded.
        }
        File::create(upgrade_mark_file.as_path())?;
    }
    copy_wal_files(dir, wal_sync_dir.as_path())?;
    fs::remove_file(upgrade_mark_file.as_path())?;
    file_system::sync_dir(dir)?;
    Ok(())
}

fn upgrade_mark_file_path(dir: &Path) -> PathBuf {
    dir.join("upgrade_mark")
}

fn open_wal_files(dir: &Path) -> Result<()> {
    // create 4 wal files and always reuse them, so we never need to sync dir on
    // writer thread.
    for i in 0..4 {
        let file_path = wal_file_path(dir, i);
        let _ = open_direct_file(&file_path, true)?;
    }
    file_system::sync_dir(dir)?;
    Ok(())
}

fn wal_file_path(dir: &Path, idx: usize) -> PathBuf {
    dir.join(format!("{}.wal", idx))
}

fn all_wal_files_exists(dir: &Path) -> bool {
    for i in 0..4 {
        if !wal_file_path(dir, i).exists() {
            return false;
        }
    }
    true
}

fn copy_wal_files(dir: &Path, wal_sync_dir: &Path) -> Result<()> {
    for i in 0..4 {
        let src_file_path = wal_file_path(dir, i);
        let dst_file_path = wal_file_path(wal_sync_dir, i);
        fs::copy(src_file_path, dst_file_path)?;
    }
    file_system::sync_dir(wal_sync_dir)?;
    Ok(())
}

pub fn load_store_ident(rf: &RfEngine) -> Option<StoreIdent> {
    let val = rf.get_state(0, STORE_IDENT_KEY);
    val.as_ref()?;
    let mut ident = StoreIdent::new();
    ident.merge_from_bytes(val.unwrap().chunk()).unwrap();
    Some(ident)
}

#[derive(Debug, Clone, Default)]
pub(crate) struct PeerMeta {
    pub(crate) region_id: u64,
    pub(crate) truncated_idx: u64,
    pub(crate) states: BTreeMap<Bytes, Bytes>,
}

impl PeerMeta {
    pub(crate) fn new(region_id: u64) -> Self {
        Self {
            region_id,
            ..Default::default()
        }
    }

    pub(crate) fn merge(&mut self, other: &PeerMeta, keep_empty: bool) {
        assert_eq!(self.region_id, other.region_id);
        if self.truncated_idx < other.truncated_idx {
            self.truncated_idx = other.truncated_idx;
        }
        for (key, val) in &other.states {
            if keep_empty || !val.is_empty() {
                self.states.insert(key.clone(), val.clone());
            } else {
                self.states.remove(key);
            }
        }
    }
}

/// `PeerData` contains region data and state in memory.
#[derive(Clone, Default)]
pub(crate) struct PeerData {
    pub(crate) peer_id: u64,
    pub(crate) meta: PeerMeta,
    pub(crate) raft_logs: RaftLogs,
}

impl Deref for PeerData {
    type Target = PeerMeta;

    fn deref(&self) -> &Self::Target {
        &self.meta
    }
}

impl DerefMut for PeerData {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.meta
    }
}

impl PeerData {
    pub(crate) fn new(peer_id: u64, region_id: u64) -> Self {
        Self {
            peer_id,
            meta: PeerMeta::new(region_id),
            ..Default::default()
        }
    }

    pub(crate) fn get(&self, index: u64) -> Option<eraftpb::Entry> {
        self.raft_logs.get(index)
    }

    pub(crate) fn term(&self, index: u64) -> Option<u64> {
        self.raft_logs.get(index).map(|e| e.term)
    }

    pub(crate) fn get_state(&self, key: &[u8]) -> Option<&Bytes> {
        self.states.get(key)
    }

    pub(crate) fn apply(&mut self, batch: &PeerBatch) -> Vec<RaftLogBlock> {
        debug_assert_eq!(self.peer_id, batch.peer_id);
        let mut truncated_blocks = vec![];
        for op in &batch.raft_logs {
            let truncated = self.raft_logs.append(op.clone());
            if !truncated.is_empty() {
                truncated_blocks.extend(truncated);
            }
        }
        let truncated_index = batch.truncated_idx;
        if self.truncated_idx < truncated_index {
            self.truncated_idx = truncated_index;
            truncated_blocks.extend(self.raft_logs.truncate(truncated_index));
        }
        if self.truncated_idx == TRUNCATE_ALL_INDEX
            && truncated_index > 0
            && truncated_index != TRUNCATE_ALL_INDEX
        {
            warn!(
                "region: {} peer:{} restore truncate all index to index {}",
                self.region_id, self.peer_id, truncated_index,
            );
            self.truncated_idx = truncated_index;
        }
        for (key, val) in &batch.states {
            if val.is_empty() {
                self.states.remove(key.chunk());
            } else {
                self.states.insert(key.clone(), val.clone());
            }
        }
        truncated_blocks
    }

    pub(crate) fn get_stats(&self) -> PeerStats {
        let size = self.raft_logs.size();
        let first_idx = self.raft_logs.first_index();
        let last_idx = self.raft_logs.last_index();
        let num_logs = if last_idx != 0 {
            (last_idx - first_idx + 1) as usize
        } else {
            0
        };
        PeerStats {
            peer_id: self.peer_id,
            region_id: self.meta.region_id,
            size,
            num_logs,
            num_states: self.meta.states.len(),
            first_idx,
            last_idx,
            truncated_idx: self.meta.truncated_idx,
        }
    }
}

#[derive(Default, Serialize, Deserialize, Debug)]
#[serde(default)]
#[serde(rename_all = "kebab-case")]
pub struct EngineStats {
    pub total_mem_size: usize,
    pub total_mem_entries: usize,
    pub num_files: usize,
    pub disk_size: u64,
    pub top_10_size_peers: Vec<PeerStats>,
}

#[derive(Default, Serialize, Deserialize, Debug, PartialEq)]
#[serde(default)]
#[serde(rename_all = "kebab-case")]
pub struct PeerStats {
    pub peer_id: u64,
    pub region_id: u64,
    pub size: usize,
    pub num_logs: usize,
    pub num_states: usize,
    pub first_idx: u64,
    pub last_idx: u64,
    pub truncated_idx: u64,
}

pub struct PeerTag {
    pub engine_id: u64,
    pub region_id: u64,
}

impl PeerTag {
    pub fn new(engine_id: u64, region_id: u64) -> Self {
        Self {
            engine_id,
            region_id,
        }
    }
}

impl Display for PeerTag {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{}", self.engine_id, self.region_id)
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, fs::OpenOptions, io::BufReader, os::unix::prelude::FileExt};

    use bytes::{BufMut, BytesMut};
    use engine_traits::Error as TraitError;
    use eraftpb::{Entry, EntryType};
    use protobuf::Message;

    use super::*;
    use crate::{log_batch::RaftLogOp, tests::init_logger};

    #[test]
    fn test_rfengine() {
        init_logger();
        let tmp_dir = tempfile::tempdir().unwrap();
        let cfg = Config::new(128 * 1024_usize);
        let engine = RfEngine::open(tmp_dir.path(), &cfg, None, None).unwrap();
        let mut wb = WriteBatch::new();
        for peer_id in 1..=10_u64 {
            let (key, val) = make_state_kv(2, 1);
            let region_id = peer_id + 1;
            wb.set_state(peer_id, region_id, key.chunk(), val.chunk());
        }
        engine.write(wb).unwrap();

        let mut truncated_regions = vec![];
        for idx in 1..=1050_u64 {
            let mut wb = WriteBatch::new();
            for peer_id in 1..=10_u64 {
                if peer_id == 1 && (idx > 100 && idx < 900) {
                    continue;
                }
                let region_id = peer_id + 1;
                wb.append_raft_log(peer_id, region_id, &make_log_data(idx, 128));
                let (key, val) = make_state_kv(1, idx);
                wb.set_state(peer_id, region_id, key.chunk(), val.chunk());
                if idx % 100 == 0 && peer_id != 1 {
                    truncated_regions.push((peer_id, region_id, idx - 100));
                    wb.truncate_raft_log(peer_id, region_id, idx - 100);
                }
            }
            engine.write(wb).unwrap();
        }
        assert_eq!(engine.peers.len(), 10);
        let wal_cnt = engine
            .dir
            .read_dir()
            .unwrap()
            .filter(|p| {
                p.as_ref()
                    .unwrap()
                    .path()
                    .extension()
                    .map_or(false, |e| e == "wal")
            })
            .count();
        assert_eq!(wal_cnt, 4);

        let mut old_entries_map = HashMap::new();
        for peer_ref in engine.peers.iter() {
            let peer_data = peer_ref.read().unwrap();
            assert_eq!(*peer_ref.key(), peer_data.peer_id);
            old_entries_map.insert(peer_data.peer_id, peer_data.clone());
        }
        assert_eq!(old_entries_map.len(), 10);
        engine.stop_worker(true);

        for _ in 0..2 {
            let engine = RfEngine::open(tmp_dir.path(), &cfg, None, None).unwrap();
            let mut wb = WriteBatch::new();
            for &(peer_id, region_id, truncated_idx) in truncated_regions.iter() {
                wb.truncate_raft_log(peer_id, region_id, truncated_idx);
            }
            engine.apply(&mut wb);
            engine.iterate_all_states(false, |peer_id, _, key, _| {
                let old_region_data = old_entries_map.get(&peer_id).unwrap();
                assert!(old_region_data.get_state(key).is_some());
                true
            });
            assert_eq!(engine.peers.len(), 10);
            for new_data_ref in engine.peers.iter() {
                let new_data = new_data_ref.read().unwrap();
                let old_data = old_entries_map.get(new_data_ref.key()).unwrap();
                assert_eq!(
                    old_data.raft_logs.first_index(),
                    new_data.raft_logs.first_index()
                );
                assert_eq!(
                    old_data.raft_logs.last_index(),
                    new_data.raft_logs.last_index()
                );
                for i in new_data.raft_logs.first_index()..=new_data.raft_logs.last_index() {
                    let entry = new_data.raft_logs.get(i).unwrap();
                    assert_eq!(
                        old_data.get(entry.index).unwrap().data.chunk(),
                        entry.data.chunk()
                    );
                }
            }
        }
    }

    fn make_log_data(index: u64, size: usize) -> eraftpb::Entry {
        let mut entry = eraftpb::Entry::new();
        entry.set_entry_type(eraftpb::EntryType::EntryConfChange);
        entry.set_index(index);
        entry.set_term(1);

        let mut data = BytesMut::with_capacity(size);
        data.resize(size, 0);
        entry.set_data(data.freeze());
        entry
    }

    fn make_state_kv(key_byte: u8, idx: u64) -> (BytesMut, BytesMut) {
        let mut key = BytesMut::new();
        key.put_u8(key_byte);
        let mut val = BytesMut::new();
        val.put_u64_le(idx);
        (key, val)
    }

    fn new_raft_entry(tp: EntryType, term: u64, index: u64, data: &[u8], context: u8) -> Entry {
        let mut entry = Entry::new();
        entry.set_entry_type(tp);
        entry.set_term(term);
        entry.set_index(index);
        entry.set_data(data.to_vec().into());
        if context > 0 {
            entry.set_context(vec![context].into());
        }
        entry
    }

    #[test]
    fn test_region_data() {
        init_logger();
        let mut region_data = PeerData::new(1, 2);

        let mut region_batch = PeerBatch::new(1, 2);
        for i in 1..=5 {
            region_batch.append_raft_log(RaftLogOp::new(&new_raft_entry(
                EntryType::EntryNormal,
                i,
                i,
                b"data",
                0,
            )));
        }
        assert!(region_data.apply(&region_batch).is_empty());
        for i in 1..=5 {
            assert_eq!(
                region_data.get(i).unwrap(),
                region_batch.raft_logs[(i - 1) as usize].to_entry()
            );
            assert_eq!(region_data.term(i).unwrap(), i);
        }
        assert!(region_data.get(6).is_none());
        let region_stats = region_data.get_stats();
        assert_eq!(
            region_stats,
            PeerStats {
                peer_id: 1,
                region_id: 2,
                size: 20,
                num_logs: 5,
                num_states: 0,
                first_idx: 1,
                last_idx: 5,
                truncated_idx: 0,
            }
        );

        region_batch = PeerBatch::new(1, 2);
        region_batch.truncate(5);
        region_batch.set_state(b"k1", b"v1");
        region_batch.set_state(b"k2", b"v2");
        let truncated = region_data.apply(&region_batch);
        assert_eq!(truncated.len(), 1);
        assert_eq!(truncated[0].first_index(), 1);
        assert_eq!(truncated[0].last_index(), 5);
        for i in 1..=5 {
            assert!(region_data.get(i).is_none());
        }
        assert_eq!(region_data.get_state(b"k1"), Some(&b"v1".to_vec().into()));
        assert_eq!(region_data.get_state(b"k2"), Some(&b"v2".to_vec().into()));
        let region_stats = region_data.get_stats();
        assert_eq!(
            region_stats,
            PeerStats {
                peer_id: 1,
                region_id: 2,
                size: 0,
                num_logs: 0,
                num_states: 2,
                first_idx: 0,
                last_idx: 0,
                truncated_idx: 5,
            }
        );

        region_batch = PeerBatch::new(1, 2);
        region_batch.truncate(5);
        region_batch.set_state(b"k1", b"");
        assert!(region_data.apply(&region_batch).is_empty());
        assert!(region_data.get_state(b"k1").is_none());
        assert_eq!(region_data.get_state(b"k2"), Some(&b"v2".to_vec().into()));

        region_batch = PeerBatch::new(1, 2);
        region_batch.truncate(100);
        assert!(region_data.apply(&region_batch).is_empty());
        assert_eq!(region_data.truncated_idx, 100);
    }

    #[test]
    fn test_rfengine_basic() {
        init_logger();
        const STATE_PREFIX: u8 = b'p';

        let dir = tempfile::tempdir().unwrap();
        let cfg = Config::new(128 * 1024);
        let engine = RfEngine::open(dir.path(), &cfg, None, None).unwrap();

        // Write 10 logs and states to 2 region.
        let mut data_map = HashMap::new();
        let mut wb = WriteBatch::new();
        for peer_id in 1..=2 {
            let region_id = peer_id + 1;
            for i in 1..=10 {
                let entry = new_raft_entry(EntryType::EntryNormal, peer_id, i, b"data", 0);
                let (state_key, state_val) = (&[STATE_PREFIX, i as u8], &[i as u8]);
                wb.append_raft_log(peer_id, region_id, &entry);
                wb.set_state(peer_id, region_id, state_key, state_val);

                let (entries, states) = data_map
                    .entry(peer_id)
                    .or_insert_with(|| (vec![], BTreeMap::new()));
                entries.push(entry);
                states.insert(state_key.to_vec(), state_val.to_vec());
            }
        }
        engine.write(wb).unwrap();

        assert_eq!(engine.get_term(1, 1), Some(1));
        assert_eq!(engine.get_term(1, 11), None);
        assert_eq!(engine.get_term(3, 1), None);
        assert_eq!(engine.get_last_index(1), Some(10));
        assert_eq!(engine.get_last_index(3), None);

        // Test `get_entry` and `get_state`.
        for (&peer_id, (entries, states)) in &data_map {
            entries.iter().for_each(|entry| {
                assert_eq!(entry, &engine.get_raft_entry(peer_id, entry.index).unwrap(),);
            });
            states
                .iter()
                .for_each(|(key, val)| assert_eq!(val, &engine.get_state(peer_id, key).unwrap()));
        }
        assert!(engine.get_raft_entry(1, 11).is_none());
        assert!(engine.get_raft_entry(3, 1).is_none());
        assert!(engine.get_state(1, b"k").is_none());

        // Test `fetch_entries_to`.
        let mut buf = vec![];
        for peer_id in 1..=2 {
            for low in 1..=10 {
                for high in low + 1..=11 {
                    assert_eq!(
                        engine
                            .fetch_raft_entries_to(peer_id, low, high, None, &mut buf)
                            .unwrap(),
                        (high - low) as usize
                    );
                    assert_eq!(
                        data_map.get(&peer_id).unwrap().0[(low - 1) as usize..(high - 1) as usize],
                        buf
                    );
                    buf.clear();
                }
            }
        }
        // Test `fetch_entries_to` should push logs to the buf.
        let peer1_entries = &data_map.get(&1).unwrap().0;
        for i in 1..=10 {
            assert_eq!(
                engine
                    .fetch_raft_entries_to(1, i, i + 1, None, &mut buf)
                    .unwrap(),
                1
            );
            assert_eq!(buf, peer1_entries[..i as usize]);
        }
        assert!(matches!(
            engine.fetch_raft_entries_to(1, 11, 12, None, &mut buf),
            Err(TraitError::EntriesUnavailable),
        ));
        // Test `fetch_entries_to` limits size.
        let mut max_size = 0;
        for (i, entry) in peer1_entries.iter().enumerate() {
            buf.clear();
            max_size += entry.compute_size();
            assert_eq!(
                engine
                    .fetch_raft_entries_to(1, 1, 11, Some(max_size as usize), &mut buf)
                    .unwrap(),
                i + 1
            );
            assert_eq!(buf, peer1_entries[..=i]);
        }

        // Test fetch empty logs.
        buf.clear();
        assert_eq!(
            engine
                .fetch_raft_entries_to(1, 1, 1, None, &mut buf)
                .unwrap(),
            0
        );
        assert!(buf.is_empty());

        // Test `get_last_state_with_prefix`
        assert_eq!(
            engine
                .get_last_state_with_prefix(1, &[STATE_PREFIX])
                .unwrap(),
            [10_u8].as_slice()
        );
        assert!(
            engine
                .get_last_state_with_prefix(1, &[STATE_PREFIX + 1])
                .is_none()
        );

        // Test `iterate_region_states`
        for desc in [false, true] {
            let mut expect_index = if desc { 10 } else { 1 };
            engine.iterate_peer_states(1, desc, |k, v| {
                assert_eq!(k, &[STATE_PREFIX, expect_index]);
                assert_eq!(v, &[expect_index]);
                if desc {
                    expect_index -= 1;
                } else {
                    expect_index += 1;
                }
            });
            assert_eq!(expect_index, if desc { 0 } else { 11 });
        }

        // Test `iterate_all_states`
        for desc in [false, true] {
            let mut count = 0;
            engine.iterate_all_states(desc, |id, _, k, v| {
                assert_eq!(v, data_map.get(&id).unwrap().1.get(k).unwrap());
                count += 1;
                true
            });
            assert_eq!(count, 20);
        }
        // Test `iterate_all_states` breaks.
        let mut count = 0;
        engine.iterate_all_states(false, |_, _, _, _| {
            count += 1;
            false
        });
        assert_eq!(count, 2);

        // Test `add_dependent` and `remove_dependent`.
        engine.add_dependent(1, 1);
        assert!(
            engine
                .dependants
                .get(&1)
                .unwrap()
                .read()
                .unwrap()
                .contains(&1)
        );
        engine.remove_dependent(1, 1);
        assert!(
            !engine
                .dependants
                .get(&1)
                .unwrap()
                .read()
                .unwrap()
                .contains(&1)
        );
    }

    #[test]
    fn test_rfengine_wal() {
        init_logger();
        let tmp_dir = tempfile::tempdir().unwrap();
        let wal_size = 128 * 1024_usize;
        let dir_path = tmp_dir.path();
        let cfg = Config::new(wal_size);
        let engine = RfEngine::open(dir_path, &cfg, None, None).unwrap();
        let mut wb = WriteBatch::new();
        for peer_id in 1..=10_u64 {
            let (key, val) = make_state_kv(2, 1);
            let region_id = peer_id + 1;
            wb.set_state(peer_id, region_id, key.chunk(), val.chunk());
        }
        engine.write(wb).unwrap();
        for idx in 1..=1050_u64 {
            let mut wb = WriteBatch::new();
            for peer_id in 1..=10_u64 {
                let region_id = peer_id + 1;
                wb.append_raft_log(peer_id, region_id, &make_log_data(idx, 128));
                let (key, val) = make_state_kv(1, idx);
                wb.set_state(peer_id, region_id, key.chunk(), val.chunk());
            }
            engine.write(wb).unwrap();
        }
        assert_eq!(engine.peers.len(), 10);
        engine.stop_worker(true);
        for _ in 0..2 {
            let engine = RfEngine::open(dir_path, &cfg, None, None).unwrap();
            assert_eq!(engine.peers.len(), 10);
            engine.stop_worker(true);
        }
        let (compacted_epoch, current_epoch) = {
            let writer = engine.writer.lock().unwrap();
            (
                writer.compacted_epoch.load(Ordering::SeqCst),
                writer.epoch_id,
            )
        };

        for ep in compacted_epoch + 1..=current_epoch {
            let filename = wal_file_name(dir_path, ep);
            let mut it = WalIterator::new(dir_path.to_owned(), ep);
            let fd = File::open(filename.clone()).unwrap();
            let mut buf_reader: Box<dyn std::io::Read> = Box::new(BufReader::new(fd));
            it.check_wal_header(&mut buf_reader).unwrap();
            let mut offsets = vec![it.offset];
            loop {
                match it.read_batch(&mut buf_reader) {
                    Err(err) => {
                        if let Error::Eof = err {
                            break;
                        }
                        panic!("{:?}", err);
                    }
                    Ok(_data) => offsets.push(it.offset),
                }
            }
            offsets.pop().unwrap();
            for (idx, offset) in offsets.iter().enumerate() {
                if idx == 0 || idx == offsets.len() / 2 || idx == offsets.len() - 1 {
                    for pos in &vec![0, 4, 8, 12] {
                        let fd = OpenOptions::new()
                            .read(true)
                            .write(true)
                            .open(filename.as_path())
                            .unwrap();
                        let mut buf = [0u8; 4096];
                        fd.read_exact_at(&mut buf, *offset).unwrap();
                        if buf[*pos] == 255 {
                            continue;
                        }
                        buf[*pos] += 1;
                        fd.write_all_at(buf.as_ref(), *offset).unwrap();
                        fd.sync_data().unwrap();
                        let open_engine = RfEngine::open(dir_path, &cfg, None, None);
                        // RfEngine can auto recover from corruption for the last epoch wal
                        // corruption.
                        assert!(if ep == current_epoch {
                            open_engine.is_ok()
                        } else {
                            open_engine.is_err()
                        });
                        buf[*pos] -= 1;
                        fd.write_all_at(buf.as_ref(), *offset).unwrap();
                        fd.sync_data().unwrap();
                    }
                }
            }
        }
    }

    #[test]
    fn test_truncate_all_logs() {
        init_logger();
        let tmp_dir = tempfile::tempdir().unwrap();
        let wal_size = 4096 * 10;
        let cfg = Config::new(wal_size);
        let engine = RfEngine::open(tmp_dir.path(), &cfg, None, None).unwrap();
        for i in 1..=50 {
            let mut wb = WriteBatch::new();
            wb.append_raft_log(1, 2, &make_log_data(i, 128));
            engine.write(wb).unwrap();
        }
        let mut wb = WriteBatch::new();
        wb.truncate_raft_log(1, 2, TRUNCATE_ALL_INDEX);
        engine.write(wb).unwrap();
        // Trigger WAL rotation twice to compact older WALs.
        let mut wb = WriteBatch::new();
        wb.append_raft_log(2, 3, &make_log_data(1, wal_size));
        wb.append_raft_log(2, 3, &make_log_data(2, wal_size));
        engine.write(wb).unwrap();
        // Waiting for compacting WAL.
        for _ in 0..10 {
            let wal_cnt = engine
                .dir
                .read_dir()
                .unwrap()
                .filter(|p| {
                    p.as_ref()
                        .unwrap()
                        .path()
                        .extension()
                        .map_or(false, |e| e == "wal")
                })
                .count();
            if wal_cnt <= 2 {
                break;
            }
            thread::sleep(std::time::Duration::from_secs(1));
        }
        // Check no file of region 1 left.
        assert_eq!(
            engine
                .dir
                .read_dir()
                .unwrap()
                .filter(|p| {
                    let path = p.as_ref().unwrap().path().to_str().unwrap().to_owned();
                    let parts: Vec<_> = path.split('_').collect();
                    parts.len() == 4 && parts[1] == format!("{:016x}", 1)
                })
                .count(),
            0
        );
    }

    #[test]
    fn test_init_wal_files() {
        init_logger();
        let tmp_dir = tempfile::tempdir().unwrap();
        init_wal_files(tmp_dir.path(), None).unwrap();
        let check_file_exists = |path: &Path| {
            for idx in 0..4 {
                assert!(wal_file_path(path, idx).exists());
            }
        };
        check_file_exists(tmp_dir.path());

        let file_contents: Vec<String> = (0..4).map(|i| format!("wal {}", i)).collect();
        let write_files = |path: &Path| {
            for idx in 0..4 {
                let wal_file_path = wal_file_path(path, idx);
                fs::write(wal_file_path.as_path(), file_contents[idx].as_bytes()).unwrap();
            }
        };
        write_files(tmp_dir.path());
        File::create(manifest_path(tmp_dir.path())).unwrap();

        // upgrade to use wal_sync_dir
        let wal_sync_dir = tmp_dir.path().join("wal_sync_dir");
        init_wal_files(tmp_dir.path(), Some(&wal_sync_dir)).unwrap();
        assert!(!upgrade_mark_file_path(tmp_dir.path()).exists());
        let check_files = || {
            for idx in 0..4 {
                let async_wal_file_path = wal_file_path(tmp_dir.path(), idx);
                assert!(async_wal_file_path.exists());
                let sync_wal_file_path = wal_file_path(&wal_sync_dir, idx);
                assert!(sync_wal_file_path.exists());
                let data = fs::read_to_string(sync_wal_file_path.as_path()).unwrap();
                assert_eq!(data, file_contents[idx]);
            }
        };
        check_files();

        // simulate upgrade interrupted.
        write_files(tmp_dir.path());
        fs::remove_file(wal_file_path(wal_sync_dir.as_path(), 3)).unwrap();
        File::create(upgrade_mark_file_path(tmp_dir.path())).unwrap();

        // init_wal_files again should recover from the interrupted upgrade.
        init_wal_files(tmp_dir.path(), Some(&wal_sync_dir)).unwrap();
        check_files();
    }
}
