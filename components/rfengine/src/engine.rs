// Copyright 2021 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    borrow::Cow,
    collections::{BTreeMap, HashMap, HashSet, VecDeque},
    fmt::{Display, Formatter},
    fs,
    fs::{create_dir_all, File, OpenOptions},
    ops::{Deref, DerefMut},
    os::unix::fs::{FileExt, MetadataExt},
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicU32, AtomicU64, Ordering},
        Arc, Mutex, RwLock,
    },
    thread::JoinHandle,
};

use arc_swap::ArcSwap;
use bytes::{Buf, Bytes};
use dashmap::mapref::one::Ref;
use engine_traits::{GetObjectOptions, ObjectStorage};
use file_system::{open_direct_file, IoRateLimitMode, IoRateLimiter};
use futures::{Future, FutureExt};
use kvengine::dfs::DFSConfig;
use kvproto::raft_serverpb::{self, StoreIdent};
use protobuf::Message;
use raft_proto::{eraftpb, eraftpb::Entry};
use rfenginepb::{ClusterBackupMeta, KeySpaceBackupMeta, StoreBackupMeta, StoreRaftLogBackupMeta};
use tikv_util::{
    defer, error, errors::Context as _, info, mpsc::Sender, panic_mark_dfs_worker_file_exists,
    spawn_anonymous_thread_with, sys::thread::StdThreadBuildWrapper, time::Instant, warn,
};

use crate::{
    config::Config,
    log_batch::{RaftLogBlock, RaftLogs},
    manifest::{
        generate_rlog_read_plan, manifest_path, persist_change_set, rlog_by_entry_index, Manifest,
        PeerFile,
    },
    metrics::*,
    service_worker::{ServiceTask, ServiceWorker},
    write_batch::{PeerBatch, WriteBatch},
    *,
};

pub const TRUNCATE_ALL_INDEX: u64 = u64::MAX;
pub const MAX_EPOCH_BACKWARD: u32 = 100;
const RAFT_INIT_LOG_INDEX: u64 = 5;
const MIN_RLOG_FILE_SIZE: u64 = 16 * 1024 * 1024; // 16MB

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

impl RfEngine {
    #[inline]
    pub fn db_dir(&self) -> Option<&Path> {
        Some(&self.core.dir)
    }

    #[inline]
    pub fn wal_dir(&self) -> Option<&Path> {
        Some(self.core.wal_dir())
    }
}

pub struct RfEngineCore {
    /// The primary directory holding engine data (e.g., WAL files).
    pub dir: PathBuf,

    /// An optional directory for synchronous WAL writes. When set, WAL files
    /// are synchronously written to `wal_sync_dir` and asynchronously written
    /// to `dir`. `wal_sync_dir` is write-only, while `dir` handles reads as
    /// well (e.g. compaction and backup). This setup allows `wal_sync_dir` to
    /// use dedicated IOPS for writes, reducing tail latency.
    pub wal_sync_dir: Option<PathBuf>,

    /// The sync WAL writer.
    pub(crate) writer: Mutex<WalWriterExt>,

    /// A concurrent map for storing all peers' in-memory state.
    pub(crate) peers: dashmap::DashMap<u64, RwLock<PeerData>>,

    /// For each `region_id`, holds a set of peer IDs that depend on this
    /// region’s logs, preventing their early truncation.
    pub(crate) dependants: dashmap::DashMap<u64, RwLock<HashSet<u64>>>,

    /// A channel sender used to schedule background tasks (e.g., WAL rotation,
    /// truncation).
    pub(crate) task_sender: Sender<ServiceTask>,

    /// A handle for the background worker thread.
    pub(crate) service_worker_handle: Mutex<Option<JoinHandle<()>>>,

    /// A unique id for this engine instance.
    pub(crate) engine_id: Arc<AtomicU64>,

    /// Whether lightweight backup mode is enabled.
    pub(crate) lightweight: bool,

    /// Tracks the latest epoch ID for WAL. Initialized during engine load and
    /// updated after wal rotation of the sync writer.
    pub(crate) current_epoch_id: Arc<AtomicU32>,

    /// Tracks the epoch ID that has been compacted.
    pub(crate) compacted_epoch: Arc<AtomicU32>,

    /// Per-peer Rlog files, shared between RfEngineCore and CompactWorker.
    /// Updated by the CompactWorker after compaction; read by RfEngineCore to
    /// decide which raft entries to offload to disk. Uses `ArcSwap` for
    /// lock-free reads to avoid impacting RfEngineCore's write path.
    pub(crate) peer_rlog_files: Arc<ArcSwap<HashMap<u64, VecDeque<PeerFile>>>>,

    /// The number of WAL files whose raft entries can be kept in memory.
    /// Calculated from the rlog memory limit and the WAL file size.
    pub(crate) in_mem_rlog_epoch_count: u32,

    /// DFS statistics collected by DFS worker.
    pub(crate) dfs_statistics: Arc<DfsMeter>,

    pub(crate) dfs_worker_healthy: Healthy,

    _lock: fslock::LockFile, // hold lock to avoid release
}

impl RfEngineCore {
    fn open(
        dir: &Path,
        cfg: &Config,
        data_dir: Option<&Path>,
        dfs_conf: Option<DFSConfig>,
    ) -> Result<Self> {
        let wal_size = cfg.target_file_size.0;
        if wal_size == 0 {
            return Err(Error::Other(
                "invalid config: wal_size must be non-zero".to_owned(),
            ));
        }

        if cfg.rlog_file_size.0 < MIN_RLOG_FILE_SIZE || cfg.rlog_file_size.0 > u32::MAX as u64 {
            return Err(Error::Other(
                "invalid config: rlog_file_size must be between [16MB, 4GB]".to_owned(),
            ));
        }
        let rlog_file_size = cfg.rlog_file_size.0 as u32;

        let in_mem_rlog_epoch_count = cfg.rlog_soft_memory_limit.0.div_ceil(wal_size);
        info!(
            "rfengine in_mem_rlog_epoch_count: {}",
            in_mem_rlog_epoch_count
        );
        let wal_sync_dir = (!cfg.wal_sync_dir.is_empty()).then(|| PathBuf::from(&cfg.wal_sync_dir));
        let wal_secondary_dir =
            (!cfg.wal_secondary_dir.is_empty()).then(|| PathBuf::from(&cfg.wal_secondary_dir));
        init_wal_files(dir, wal_sync_dir.as_ref(), wal_secondary_dir.as_ref())?;

        // Lock the rfengine directory to prevent concurrent opening.
        let lock_path = dir.join("LOCK");
        let mut lock = fslock::LockFile::open(&lock_path)?;
        if !lock.try_lock()? {
            panic!("rfengine lock failed, maybe already used by another process");
        }

        let engine_id = Arc::new(AtomicU64::new(0));
        let manifest = Manifest::open(dir, engine_id.clone())?;
        let (service_tx, service_rx) = tikv_util::mpsc::unbounded();
        let compacted_epoch = Arc::new(AtomicU32::new(manifest.epoch_id));
        let wal_dir = wal_sync_dir.as_deref().unwrap_or(dir);
        let writer_type = if cfg.cli_mode {
            WriterType::CliMode
        } else {
            WriterType::Sync
        };
        let compression_threshold = cfg.batch_compression_threshold.0 as usize;
        let writer = WalWriter::new(
            wal_dir,
            wal_size as usize,
            compression_threshold,
            compacted_epoch.clone(),
            writer_type,
            cfg.write_throttle_duration.0,
        );
        let writer_ext = if cfg.wal_secondary_dir.is_empty() {
            WalWriterExt::SingleWriter(writer)
        } else {
            let wal_secondary_dir = PathBuf::from(&cfg.wal_secondary_dir);
            let secondary_writer = WalWriter::new(
                &wal_secondary_dir,
                wal_size as usize,
                compression_threshold,
                compacted_epoch.clone(),
                writer_type,
                cfg.write_throttle_duration.0,
            );
            WalWriterExt::DoubleWriter(DoubleWriter::new(
                writer,
                secondary_writer,
                manifest.epoch_id,
            )?)
        };
        let dfs_worker_healthy = dfs_worker::Healthy::default();
        let peer_rlog_files = Arc::new(ArcSwap::from_pointee(manifest.peer_rlog_files()));
        let mut en = Self {
            dir: dir.to_owned(),
            wal_sync_dir,
            peers: Default::default(),
            dependants: Default::default(),
            writer: Mutex::new(writer_ext),
            task_sender: service_tx,
            service_worker_handle: Mutex::new(None),
            engine_id,
            lightweight: cfg.lightweight_backup,
            current_epoch_id: Arc::new(AtomicU32::new(0)),
            compacted_epoch: compacted_epoch.clone(),
            peer_rlog_files: peer_rlog_files.clone(),
            in_mem_rlog_epoch_count: in_mem_rlog_epoch_count as u32,
            _lock: lock,
            dfs_statistics: Arc::default(),
            dfs_worker_healthy: dfs_worker_healthy.clone(),
        };
        let async_offset = en.load(&manifest)?;
        if cfg.cli_mode {
            return Ok(en);
        }
        {
            let async_wal_writer = if en.is_async_wal_enabled() {
                let mut async_wal_writer = WalWriter::new(
                    dir,
                    wal_size as usize,
                    compression_threshold,
                    compacted_epoch.clone(),
                    WriterType::Async,
                    cfg.write_throttle_duration.0,
                );
                async_wal_writer.open_file(manifest.epoch_id + 1, async_offset)?;
                Some(async_wal_writer)
            } else {
                None
            };

            let lightweight_backup_config: Option<LightweightBackupConfig> = if cfg
                .lightweight_backup
            {
                if data_dir.is_some() && panic_mark_dfs_worker_file_exists(data_dir.unwrap()) {
                    // If panic_mark_dfs_worker_file exists, skip init dfs worker thread and mark
                    // dfs worker unhealthy.
                    dfs_worker_healthy.set_unhealthy(u32::MAX, "open");
                    error!(
                        "lightweight backup is enabled, but panic_mark_dfs_worker_file exists, skip init dfs worker thread"
                    );
                    None
                } else {
                    Some(LightweightBackupConfig::new(
                        dir.to_owned(),
                        cfg.wal_chunk_target_file_size.0 as usize,
                        CompressionType::Lz4Compression,
                        CompressionType::Lz4Compression,
                        dfs_conf.unwrap(),
                        cfg.rlog_cache_capacity.0 as usize,
                        cfg.rlog_cache_size_threshold.0 as usize,
                        cfg.dfs_worker_memory_limit.as_memory_size() as usize,
                        cfg.max_wal_chunk_gap_duration,
                    ))
                }
            } else {
                None
            };
            let compact_rate_limiter =
                Arc::new(IoRateLimiter::new(IoRateLimitMode::WriteOnly, true, false));
            compact_rate_limiter.set_io_rate_limit(cfg.compact_bytes_per_sec.0 as usize);
            let epoch_id = en.current_epoch_id.load(Ordering::SeqCst);
            let mut service_worker = ServiceWorker::new(
                dir.to_owned(),
                epoch_id,
                async_wal_writer,
                service_rx,
                manifest,
                compacted_epoch.clone(),
                lightweight_backup_config,
                dfs_worker_healthy,
                compact_rate_limiter,
                cfg.compact_wal_sync_concurrency,
                rlog_file_size,
                peer_rlog_files,
            );
            if let Some(dfs_statistic) = service_worker.dfs_statistic() {
                en.dfs_statistics = dfs_statistic
            }
            let join_handle = spawn_anonymous_thread_with!(move || service_worker.run());
            let mut guard = en.service_worker_handle.lock().unwrap();
            *guard = Some(join_handle);
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
        let peer_rlog_files = self.peer_rlog_files.load();
        let offload_epoch = calc_offload_epoch(
            self.in_mem_rlog_epoch_count,
            &self.current_epoch_id,
            &self.compacted_epoch,
        );
        for (&peer_id, batch_data) in &wb.peers {
            let region_id = batch_data.meta.region_id;
            tikv_util::set_current_region_thread_local(region_id);
            let peer_data = self.get_or_init_peer_data(peer_id, region_id);
            let mut peer_data = peer_data.write().unwrap();
            let mut truncated = peer_data.apply(batch_data);
            if let Some(offload_idx) = peer_rlog_files
                .get(&peer_id)
                .and_then(|r| find_offload_index(r, &peer_data.raft_logs, offload_epoch))
            {
                truncated.extend(peer_data.raft_logs.truncate(offload_idx));
            }
            drop(peer_data);
            if !truncated.is_empty() {
                truncated_logs.push(truncated);
            }
        }
        if !truncated_logs.is_empty() {
            if let Err(e) = self
                .task_sender
                .send(ServiceTask::Truncates(truncated_logs))
            {
                warn!("send service Truncates task failed {:?}", e);
            }
        }
        ENGINE_APPLY_DURATION_HISTOGRAM.observe(timer.saturating_elapsed_secs());
    }

    /// Persists the write batch to WAL. It can be used in another thread to
    /// implement async I/O, i.e., call `apply` in the main thread and call
    /// `persist` in the I/O thread.
    /// When the epoch is rotated, the return size would be 4096 which is not
    /// accurate but ok to be used as metrics.
    pub fn persist(&self, wb: WriteBatch) -> Result<usize> {
        let timer = Instant::now_coarse();
        let wb = Arc::new(wb.into_vector());
        let mut writer = self.writer.lock().unwrap();
        let old_file_off = writer.get_file_off();
        let (epoch_id, file_off, rotated) = writer.write_batch(wb.clone())?;
        if rotated {
            self.current_epoch_id.store(epoch_id, Ordering::SeqCst);
            if let Err(e) = self.task_sender.send(ServiceTask::Rotate {
                epoch_id: epoch_id - 1, // the epoch of the old raft log
                cache_wb_for_compact: true,
            }) {
                warn!("send service rotate task failed: {:?}", e);
            }
        }
        if let Err(e) = self.task_sender.send(ServiceTask::Write { wb }) {
            warn!("send service write task failed: {:?}", e);
        }
        ENGINE_PERSIST_DURATION_HISTOGRAM.observe(timer.saturating_elapsed_secs());
        if rotated {
            return Ok(file_off as usize);
        }
        Ok(file_off.saturating_sub(old_file_off) as usize)
    }

    pub fn is_empty(&self) -> bool {
        self.peers.is_empty()
    }

    fn get_entry_from_rlog(
        &self,
        peer_id: u64,
        target_index: u64,
        peer_file: PeerFile,
    ) -> Result<Entry> {
        fetch_entries_from_rlog(
            &self.dir,
            peer_id,
            peer_file.first_index,
            peer_file.last_index,
            target_index,
            target_index,
        )?
        .into_iter()
        .next()
        .map(|op| op.to_entry())
        .ok_or_else(|| {
            Error::Other(format!(
                "peer {peer_id} entry {target_index} not found in {:?}",
                peer_file
            ))
        })
    }

    pub fn get_term(&self, peer_id: u64, index: u64) -> Option<u64> {
        let peer_data_ref = self.peers.get(&peer_id)?;
        let data = peer_data_ref.read().unwrap();
        if let Some(term_in_mem) = data.term(index) {
            return Some(term_in_mem);
        }
        // Check that the index is in the expected range.
        if index <= data.truncated_idx
            || (data.raft_logs.last_index() != 0 && index > data.raft_logs.last_index())
        {
            return None;
        }
        // Get the term from the rlog files.
        let file = self
            .peer_rlog_files
            .load()
            .get(&peer_id)
            .and_then(|files| rlog_by_entry_index(files, index))?;
        info!(
            "Fetching term for peer_id {} index: {} from offloaded rlog file {:?}",
            peer_id, index, file
        );
        self.get_entry_from_rlog(peer_id, index, file)
            .ok()
            .map(|entry| entry.term)
    }

    pub fn get_truncated_index(&self, peer_id: u64) -> Option<u64> {
        let peer_data_ref = self.peers.get(&peer_id)?;
        let data = peer_data_ref.read().unwrap();
        Some(data.truncated_idx)
    }

    pub fn get_last_index(&self, peer_id: u64) -> Option<u64> {
        let peer_data_ref = self.peers.get(&peer_id)?;
        let data = peer_data_ref.read().unwrap();

        let last_index_in_mem = data.raft_logs.last_index();
        if last_index_in_mem != 0 {
            return Some(last_index_in_mem);
        }
        // Get the last index from offloaded entries.
        self.peer_rlog_files
            .load()
            .get(&peer_id)
            .and_then(|rlog_files| rlog_files.back())
            .map(|file| file.last_index)
            .filter(|&last_index| last_index > data.truncated_idx)
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
            .next_back()
            .map(|(_, v)| v.clone())
    }

    /// Iterates states of the region in order or in desc order if `desc` is
    /// true until `f` returns error. The ietrator will stop if the function
    /// returns false.
    pub fn iterate_peer_states<F>(&self, peer_id: u64, desc: bool, mut f: F)
    where
        F: FnMut(&[u8], &[u8]) -> bool,
    {
        let peer_data = self.peers.get(&peer_id);
        let peer_data = match &peer_data {
            Some(data) => data.read().unwrap(),
            None => return,
        };

        let states = &peer_data.meta.states;
        if desc {
            for (k, v) in states.iter().rev() {
                if !f(k.chunk(), v.chunk()) {
                    break;
                }
            }
        } else {
            for (k, v) in states.iter() {
                if !f(k.chunk(), v.chunk()) {
                    break;
                }
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
        let mut handle = self.service_worker_handle.lock().unwrap();
        if let Some(h) = handle.take() {
            self.task_sender.send(ServiceTask::Close { force }).unwrap();
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
        let newly_inserted = hs.insert(dependent_id);
        if !newly_inserted {
            return;
        }
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

    pub fn pending_compaction_wals(&self) -> u8 {
        (self.current_epoch_id.load(Ordering::SeqCst)
            - 1
            - self.compacted_epoch.load(Ordering::SeqCst)) as u8
    }

    /// Dumps the state of the engine.
    pub fn get_engine_stats(&self) -> EngineStats {
        let mut total_mem_size = 0;
        let mut total_mem_entries = 0;
        let mut total_offloaded_entries = 0;
        let mut total_num_logs = 0;
        let mut peers_stats = self
            .peers
            .iter()
            .map(|data| {
                let peer_stats = data.read().unwrap().get_stats();
                total_mem_size += peer_stats.size;
                total_mem_entries += peer_stats.num_logs;
                total_offloaded_entries += peer_stats.num_logs_offloaded;
                // peer_stats.num_logs is the log number in memory, so we use
                // first_index and last_index to calculate the real log count.
                let truncated_index = self.get_truncated_index(*data.key());
                let last_index = self.get_last_index(*data.key());
                let num_logs = match (truncated_index, last_index) {
                    // trancated_index is inited with 0, but the min raft truncated index should be
                    // `RAFT_INIT_LOG_INDEX`,
                    (Some(t), Some(l)) => l - t.max(RAFT_INIT_LOG_INDEX),
                    // last_index is 0 means there is no raft logs,
                    _ => 0,
                };
                total_num_logs += num_logs;
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
        let pending_compaction_wals = self.pending_compaction_wals();
        ENGINE_PENDING_COMPACTION_WALS_GAUGE.set(pending_compaction_wals as i64);
        // Flush metrics.
        ENGINE_RESOUCE_USAGE.memory.set(total_mem_size as i64);
        ENGINE_RESOUCE_USAGE.disk.set(disk_size as i64); // TODO: move it to CSE.store_size
        ENGINE_ENTRIES_COUNT.memory.set(total_mem_entries as i64);
        ENGINE_ENTRIES_COUNT
            .offloaded
            .set(total_offloaded_entries as i64);
        ENGINE_TOTAL_WALS_GAUGE.set(num_files as i64);

        EngineStats {
            total_mem_size,
            total_mem_entries,
            disk_size,
            num_files,
            pending_compaction_wals,
            top_10_size_peers: peers_stats,
            total_num_logs,
        }
    }

    pub fn get_dfs_stats(&self) -> EngineDfsStats {
        let dfs = self.dfs_statistics.measure();
        RFENGINE_DFS_UPLOAD_BYTES.inc_by(dfs.uploaded_bytes);
        RFENGINE_DFS_REQUESTS.inc_by(dfs.request_count);
        EngineDfsStats {
            requests: dfs.request_count,
            uploaded_bytes: dfs.uploaded_bytes,
        }
    }

    pub fn dfs_stats(&self) -> &Arc<DfsMeter> {
        &self.dfs_statistics
    }

    /// Dumps the state of the region.
    pub fn get_peer_stats(&self, peer_id: u64) -> PeerStats {
        self.peers
            .get(&peer_id)
            .map(|data| data.read().unwrap().get_stats())
            .unwrap_or_default()
    }

    /// Returns the log index up to which logs should be truncated for the given
    /// peer so that its total log usage (including in-memory and offloaded
    /// entries) is reduced below the specified size limit.
    pub fn index_to_truncate_to_size(&self, peer_id: u64, size_limit: usize) -> u64 {
        self.peers
            .get(&peer_id)
            .map(|data| data.read().unwrap().index_to_truncate_to_size(size_limit))
            .unwrap_or_default()
    }

    pub fn set_engine_id(&self, engine_id: u64) {
        self.engine_id.store(engine_id, Ordering::Release)
    }

    pub fn get_engine_id(&self) -> u64 {
        self.engine_id.load(Ordering::Acquire)
    }

    pub fn get_region_peer_map(&self) -> HashMap<u64 /* region_id */, u64 /* peer_id */> {
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
        let peer_data_ref = self.peers.get(&peer_id)?;
        let data = peer_data_ref.read().unwrap();
        if let Some(entry) = data.get(index) {
            return Some(entry);
        }
        // Check that the index is in the expected range.
        if index <= data.truncated_idx
            || (data.raft_logs.last_index() != 0 && index > data.raft_logs.last_index())
        {
            return None;
        }
        // Get the entry from the rlog files.
        let file = self
            .peer_rlog_files
            .load()
            .get(&peer_id)
            .and_then(|files| rlog_by_entry_index(files, index))?;
        self.get_entry_from_rlog(peer_id, index, file).ok()
    }

    pub fn fetch_raft_entries_to(
        &self,
        peer_id: u64,
        low: u64,
        high: u64,
        max_size: Option<usize>,
        buf: &mut Vec<Entry>,
        context: Option<raft::GetEntriesContext>,
    ) -> crate::Result<usize> {
        if high <= low {
            return Ok(0);
        }
        let old_len = buf.len();
        let peer_data = self
            .peers
            .get(&peer_id)
            .ok_or(crate::Error::EntriesCompacted)?;
        let peer_data = peer_data.read().unwrap();
        if low <= peer_data.meta.truncated_idx {
            return Err(crate::Error::EntriesCompacted);
        }

        let timer = Instant::now_coarse();
        defer! {
            ENGINE_FETCH_ENTRIES_DURATION_HISTOGRAM.observe(timer.saturating_elapsed_secs())
        };
        let mut total_size = 0;
        // Some Raft logs may have been offloaded to disk. To fetch all entries
        // in the given range, we first check whether any part of the range
        // falls below the first in-memory index. If so, we need to read from
        // the offloaded log files (rlogs).
        //
        // Step 1: Load offloaded entries if necessary. `first_idx_in_mem == 0`
        // is a special case where all raft entries have been offloaded.
        let first_idx_in_mem = peer_data.raft_logs.first_index();
        let need_read_from_disk = first_idx_in_mem == 0 || low < first_idx_in_mem;

        if need_read_from_disk {
            info!(
                "peer {}: fetch_raft_entries_to entries [{},{}), first_idx_in_mem: {}",
                peer_id, low, high, first_idx_in_mem
            );

            // Calculate the range that needs to be read from disk
            let (disk_fetch_low, disk_fetch_high) = if first_idx_in_mem == 0 {
                (low, high)
            } else {
                (low, first_idx_in_mem.min(high))
            };

            // Generate read plan for the disk fetch
            let read_plan = generate_rlog_read_plan(
                self.peer_rlog_files
                    .load()
                    .get(&peer_id)
                    .ok_or(crate::Error::EntriesUnavailable)?,
                disk_fetch_low,
                disk_fetch_high,
            )?;

            // Check if we should handle this asynchronously
            if let Some(ctx) = &context {
                if ctx.can_async() {
                    // For async requests, return detailed fetch info so the upper layer (rfstore)
                    // can handle the async scheduling with full context information.
                    let async_fetch_info = crate::AsyncFetchInfo {
                        peer_id,
                        low,
                        high,
                        max_size,
                        region_id: peer_data.region_id,
                    };

                    info!(
                        "peer {}: async read requested - {}",
                        peer_id, async_fetch_info
                    );

                    return Err(crate::Error::AsyncFetch(async_fetch_info));
                }
            }

            // Synchronous disk read path
            let mut expected_idx = disk_fetch_low;
            for (range, peer_file) in read_plan {
                info!(
                    "peer {}: loading offloaded entries [{},{}] from file={:?}",
                    peer_id, range.start, range.end, peer_file
                );
                let entries = fetch_entries_from_rlog(
                    &self.dir,
                    peer_id,
                    peer_file.first_index,
                    peer_file.last_index,
                    range.start,
                    range.end,
                )
                .map_err(|_| crate::Error::EntriesUnavailable)?;
                for entry in entries {
                    let entry = entry.to_entry();
                    debug_assert!(
                        entry.index == expected_idx,
                        "peer {}, entry.index:{}, expected_idx:{}",
                        peer_id,
                        entry.index,
                        expected_idx
                    );
                    expected_idx += 1;
                    total_size += entry.compute_size() as usize;
                    buf.push(entry);
                    if max_size.map_or(false, |s| total_size >= s) {
                        // At least return one entry regardless of size limit.
                        return Ok(buf.len() - old_len);
                    }
                }
            }
            // Assert that the last entry index == disk_fetch_high - 1
            if let Some(last_entry) = buf.last() {
                debug_assert_eq!(
                    last_entry.index,
                    disk_fetch_high - 1,
                    "peer {}: last_entry.index = {}, expected {}",
                    peer_id,
                    last_entry.index,
                    disk_fetch_high - 1
                );
            }
        }

        // Step 2: Fetch in-memory entries to results.
        let need_read_from_mem = first_idx_in_mem != 0 && first_idx_in_mem < high;
        if need_read_from_mem {
            let start = low.max(first_idx_in_mem);
            for i in start..high {
                let entry = peer_data.get(i).ok_or(crate::Error::EntriesUnavailable)?;
                total_size += entry.compute_size() as usize;
                buf.push(entry);
                if max_size.map_or(false, |s| total_size >= s) {
                    // At least return one entry regardless of size limit.
                    break;
                }
            }
        }
        Ok(buf.len() - old_len)
    }

    // Upload latest wal chunk to object storage
    pub fn upload_wal_chunk(&self) -> impl Future<Output = ()> {
        let (tx, rx) = tokio::sync::oneshot::channel();
        self.task_sender.send(ServiceTask::Upload(tx)).unwrap();
        rx.map(|_| ())
    }

    pub fn dfs_worker_is_healthy(&self) -> bool {
        self.dfs_worker_healthy
            .is_healthy(self.current_epoch_id.load(Ordering::SeqCst))
    }

    pub fn dump_wal_chunk(
        &self,
        epoch_id: u32,
        start_off: u64,
        end_off: u64,
        callback: Box<dyn FnOnce(Result<(Bytes, bool /* partial content */)>) + Send>,
    ) {
        self.task_sender
            .send(ServiceTask::Dump {
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
            task.file_off = writer.get_file_off();
        }
        self.task_sender.send(ServiceTask::Backup(task)).unwrap();
    }

    pub fn get_epoch_offset(&self) -> (u32, u64) {
        let writer = self.writer.lock().unwrap();
        (writer.get_epoch_id(), writer.get_file_off())
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

/// Calculates the epoch up to which raft logs can be offloaded.
///
/// Keeps `in_mem_rlog_epoch_count` most recent epochs in memory. Also,
/// offloaded epoch must not exceed `compacted_epoch`.
///
/// Returns:
/// - 0 if offloading is disabled (`in_mem_rlog_epoch_count == 0`)
/// - otherwise, `min(current_epoch - N, compacted_epoch)`
pub(crate) fn calc_offload_epoch(
    in_mem_rlog_epoch_count: u32,
    current_epoch: &AtomicU32,
    compacted_epoch: &AtomicU32,
) -> u32 {
    if in_mem_rlog_epoch_count == 0 {
        return 0;
    }

    let current_epoch = current_epoch.load(Ordering::SeqCst);
    let compacted_epoch = compacted_epoch.load(Ordering::SeqCst);
    current_epoch
        .saturating_sub(in_mem_rlog_epoch_count)
        .min(compacted_epoch)
}

/// Finds the largest index eligible for offloading, given a target
/// `offload_epoch`.
///
/// Scans rlog files from newest to oldest and returns the last index of the
/// first file that satisfies:
/// 1. Has a valid epoch and `epoch_id <= offload_epoch`.
/// 2. Its last term matches the term of the corresponding in-memory entry. If
/// the terms don't match, the in-memory entry is newer and must not be
/// offloaded.
///
/// Returns:
/// - Some(index) if offloading can proceed up to that index
/// - None if nothing is eligible
pub(crate) fn find_offload_index(
    rlog_files: &VecDeque<PeerFile>,
    raft_logs: &RaftLogs,
    offload_epoch: u32,
) -> Option<u64> {
    if offload_epoch == 0 || raft_logs.first_index() == 0 {
        return None;
    }

    for f in rlog_files.iter().rev() {
        if f.epoch_id == 0 || f.epoch_id > offload_epoch {
            continue;
        }

        match raft_logs.get(f.last_index) {
            Some(entry) if entry.term == f.last_term as u64 => return Some(f.last_index),
            Some(_) => continue, // term mismatch - skip
            None => {
                if f.last_index > raft_logs.last_index() {
                    // In-memory entries with lower index have overwritten this
                    // rlog file. Skip it.
                    continue;
                } else {
                    // Entries in this rlog file are not in memory (i.e. already
                    // offloaded). No need to check older files.
                    return None;
                }
            }
        }
    }
    None
}

fn restore_all_raft_logs(
    object_storage: &Arc<dyn ObjectStorage>,
    store_meta: &StoreBackupMeta,
    dir: &Path,
    snapshot_rlog: Option<String>,
) -> Result<()> {
    let store_id = store_meta.store_id;
    let raft_file_key = snapshot_rlog
        .unwrap_or_else(|| store_raft_log_file_key(store_id, store_meta.get_manifest().epoch_id));
    let raft_file = object_storage
        .get_objects(vec![(raft_file_key, GetObjectOptions::default())])
        .unwrap();
    let (_, rlog_data) = raft_file.first().unwrap();
    restore_all_raft_logs_with_snap_rlog_file(None, store_meta, dir, rlog_data)
}

fn decompress_snap_rlog_file(compression_type: u32, content: &[u8]) -> Result<Cow<'_, [u8]>> {
    match CompressionType::from(compression_type) {
        CompressionType::Lz4Compression => {
            let decompressed = decompress_lz4(content).map_err(Error::from)?;
            Ok(Cow::Owned(decompressed))
        }
        CompressionType::NoCompression => Ok(Cow::Borrowed(content)),
    }
}

fn restore_all_raft_logs_with_snap_rlog_file(
    keyspace_id: Option<u32>,
    store_meta: &StoreBackupMeta,
    dir: &Path,
    rlog_data: &Bytes,
) -> Result<()> {
    let mut raft_meta = StoreRaftLogBackupMeta::default();
    let size = rlog_data.len();
    debug_assert!(size as u64 > store_meta.raft_meta_start_off);

    raft_meta
        .merge_from_bytes(&rlog_data.chunk()[store_meta.raft_meta_start_off as usize..size])
        .unwrap();

    let compression_type = raft_meta.get_header().get_compression_type();
    let handle_keyspace = |keyspace_meta: &KeySpaceBackupMeta| -> Result<()> {
        for file in keyspace_meta.get_files() {
            let path = raft_log_file_name(dir, file.peer_id, file.first_index, file.last_index);
            let content = decompress_snap_rlog_file(
                compression_type,
                &rlog_data.chunk()[file.start_off as usize..file.end_off as usize],
            )
            .unwrap();
            fs::write(&path, &content).with_ctx(|| format!("write rlog {}", path.display()))?;
        }
        Ok(())
    };

    if let Some(keyspace_id) = keyspace_id {
        if let Some(keyspace_meta) = raft_meta.raft_logs.get(&keyspace_id) {
            handle_keyspace(keyspace_meta)?;
        }
    } else {
        for (_, keyspace_meta) in raft_meta.raft_logs {
            handle_keyspace(&keyspace_meta)?;
        }
    }
    Ok(())
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
        start_off: Some(store_meta.raft_meta_start_off),
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
        start_off: Some(raft_files.first().unwrap().start_off),
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
    store_id: u64,
    keyspace_id: Option<u32>,
    dir: &Path,
    snap_epoch: u32,
    snap_meta: Bytes,
    snap_rlog: Bytes,
) -> Result<u32> {
    let start_time = Instant::now_coarse();

    init_wal_files(dir, None, None)?;

    let mut snap_store_meta = StoreBackupMeta::default();
    if snap_epoch > 0 {
        snap_store_meta.merge_from_bytes(snap_meta.chunk()).unwrap();
        assert_eq!(snap_epoch, snap_store_meta.get_manifest().epoch_id);
        restore_all_raft_logs_with_snap_rlog_file(keyspace_id, &snap_store_meta, dir, &snap_rlog)?;
    }
    let dur_restore_rlogs = start_time.saturating_elapsed();

    info!("{} manifest file path: {:?}", store_id, manifest_path(dir); "keyspace" => ?keyspace_id);
    let manifest_file = OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(manifest_path(dir))
        .ctx("open manifest")?;
    if let Some(keyspace_id) = keyspace_id {
        let before = snap_store_meta.get_manifest().peers.len();
        filter_manifest_peers_for_keyspace(snap_store_meta.mut_manifest(), keyspace_id);
        let after = snap_store_meta.get_manifest().peers.len();
        info!("{} filter manifest peers: {} -> {}", store_id, before, after; "keyspace" => keyspace_id);
    }
    persist_change_set(&manifest_file, 0, snap_store_meta.get_manifest())
        .ctx("persist change set")?;
    let dur_persist_manifest = start_time.saturating_elapsed() - dur_restore_rlogs;

    info!("{} restore rfengine", store_id;
        "restore_rlogs" => ?dur_restore_rlogs, "persist_manifest" => ?dur_persist_manifest);
    Ok(snap_store_meta.get_manifest().epoch_id)
}

fn filter_manifest_peers_for_keyspace(cs: &mut rfenginepb::ChangeSet, keyspace_id: u32) {
    let peers = cs.take_peers();
    peers
        .into_iter()
        .filter(|peer| {
            peer.region_id == 0
                || peer.peer_id == 0
                || get_keyspace_id_from_peer(peer).is_some_and(|x| x == keyspace_id)
        })
        .for_each(|peer| cs.mut_peers().push(peer));
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
    init_wal_files(dir, None, None).unwrap();
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
            file.write_all_at(&data, store_meta.get_wal_chunks()[i].start_off)
                .unwrap();
        }
        let end_off = wal_chunks.last().unwrap().end_off;
        let eof = vec![0u8; 4096];
        file.write_all_at(&eof, end_off).unwrap();
        file.sync_data().unwrap();
    }
    match keyspace {
        Some(keyspace_id) => {
            restore_keyspace_raft_logs(&object_storage, store_meta, dir, keyspace_id, None)
        }
        None => restore_all_raft_logs(&object_storage, store_meta, dir, None).unwrap(),
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

pub(crate) fn init_wal_files(
    dir: &Path,
    wal_sync_dir: Option<&PathBuf>,
    wal_secondary_dir: Option<&PathBuf>,
) -> Result<()> {
    if !dir.exists() {
        create_dir_all(dir)?;
    }
    open_wal_files(dir)?;
    if let Some(wal_secondary_dir) = wal_secondary_dir {
        if !wal_secondary_dir.exists() {
            create_dir_all(wal_secondary_dir)?;
        }
    }
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

pub fn save_store_ident(rf: &RfEngine, store_ident: &StoreIdent) {
    let val = store_ident.write_to_bytes().unwrap();
    let mut wb = WriteBatch::new();
    wb.set_state(0, 0, STORE_IDENT_KEY, &val);
    rf.write(wb).unwrap();
}

#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct PeerMeta {
    pub(crate) region_id: u64,
    pub(crate) truncated_idx: u64,
    pub(crate) states: BTreeMap<Bytes, Bytes>,
    pub(crate) states_encoded_len: usize,
}

const ENTRY_BASE_LEN: usize = 2 /* key len */ + 4 /* value len */;

impl PeerMeta {
    pub(crate) fn new(region_id: u64) -> Self {
        Self {
            region_id,
            ..Default::default()
        }
    }

    pub fn set_state(&mut self, key: &[u8], val: &[u8]) {
        self.states_encoded_len += key.len() + val.len() + ENTRY_BASE_LEN;
        let old = self
            .states
            .insert(Bytes::copy_from_slice(key), Bytes::copy_from_slice(val));
        if let Some(old) = old {
            self.states_encoded_len -= key.len() + old.len() + ENTRY_BASE_LEN;
        }
    }

    pub fn remove_state(&mut self, key: &[u8]) {
        if let Some(old) = self.states.remove(key) {
            self.states_encoded_len -= key.len() + old.len() + ENTRY_BASE_LEN;
        }
    }

    pub fn get_state(&self, key: &[u8]) -> Option<&[u8]> {
        self.states.get(key).map(|v| v.chunk())
    }

    pub fn get_latest_state(&self, key_prefix: &[u8]) -> Option<&[u8]> {
        self.states
            .iter()
            .rev()
            .find(|(k, _)| k.starts_with(key_prefix))
            .map(|(_, v)| v.chunk())
    }

    pub(crate) fn merge(&mut self, other: &PeerMeta, keep_empty: bool) {
        assert_eq!(self.region_id, other.region_id);
        if self.truncated_idx < other.truncated_idx {
            self.truncated_idx = other.truncated_idx;
        }
        for (key, val) in &other.states {
            let old = if keep_empty || !val.is_empty() {
                self.states_encoded_len += key.len() + val.len() + ENTRY_BASE_LEN;
                self.states.insert(key.clone(), val.clone())
            } else {
                self.states.remove(key)
            };
            if let Some(old) = old {
                self.states_encoded_len -= key.len() + old.len() + ENTRY_BASE_LEN;
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
        self.meta.merge(&batch.meta, false);
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
        let num_logs_offloaded = self.get_num_logs_offloaded() as usize;
        PeerStats {
            peer_id: self.peer_id,
            region_id: self.meta.region_id,
            size,
            num_logs,
            num_logs_offloaded,
            num_states: self.meta.states.len(),
            first_idx,
            last_idx,
            truncated_idx: self.meta.truncated_idx,
        }
    }

    // Returns the number of raft log entries that have been offloaded.
    //
    // Caveat: if all raft logs have been offloaded (which can only happen if
    // TiKV restarts and the peer has no new writes), the returned value will be
    // 0, which is inaccurate.
    fn get_num_logs_offloaded(&self) -> u64 {
        self.raft_logs.first_index().saturating_sub(
            self.meta
                .truncated_idx
                .max(RAFT_INIT_LOG_INDEX)
                .saturating_add(1),
        )
    }

    /// Returns the index such that truncating logs up to this index will reduce
    /// its total raft log usage (including in-memory and offloaded entries) to
    /// below the specified size limit.
    fn index_to_truncate_to_size(&self, size_limit: usize) -> u64 {
        let in_mem_bytes = self.raft_logs.size();
        // If in‑memory logs alone exceed the limit, no need to check offloaded
        // entries.
        if in_mem_bytes >= size_limit {
            return self.raft_logs.index_to_truncate_to_size(size_limit);
        }

        // Check if we need to truncate the offloaded raft logs. We estimate the
        // offloaded raft log size based on the average raft entry size in
        // memory.
        let in_mem_cnt = self.raft_logs.len();
        let offloaded_cnt = self.get_num_logs_offloaded();
        if in_mem_cnt == 0 || offloaded_cnt == 0 || in_mem_bytes == 0 {
            return 0;
        }
        let avg_entry_size = in_mem_bytes.div_ceil(in_mem_cnt);
        let offloaded_cnt_allowed = (size_limit - in_mem_bytes) / avg_entry_size;
        let to_truncate = offloaded_cnt.saturating_sub(offloaded_cnt_allowed as u64);
        if to_truncate == 0 {
            return 0;
        }

        info!(
            "peer {}: truncating offloaded entries, \
            in_mem_bytes {}, in_mem_cnt {}, offloaded_cnt {}, to_truncate {}",
            self.peer_id, in_mem_bytes, in_mem_cnt, offloaded_cnt, to_truncate
        );
        self.meta.truncated_idx.max(RAFT_INIT_LOG_INDEX) + to_truncate
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
    pub pending_compaction_wals: u8,
    pub top_10_size_peers: Vec<PeerStats>,
    pub total_num_logs: u64,
}

#[derive(Default, Serialize, Deserialize, Debug)]
#[serde(default)]
#[serde(rename_all = "kebab-case")]
pub struct EngineDfsStats {
    pub uploaded_bytes: u64,
    pub requests: u64,
}

#[derive(Default, Serialize, Deserialize, Debug, PartialEq)]
#[serde(default)]
#[serde(rename_all = "kebab-case")]
pub struct PeerStats {
    pub peer_id: u64,
    pub region_id: u64,
    pub size: usize,
    pub num_logs: usize,
    pub num_logs_offloaded: usize,
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
    use std::{collections::HashMap, fs::OpenOptions, os::unix::prelude::FileExt, time::Duration};

    use ::test_util::eventually;
    use eraftpb::EntryType;
    use protobuf::Message;
    use tikv_util::config::ReadableSize;

    use super::*;
    use crate::{
        log_batch::{RaftLogOp, RAFT_LOG_BLOCK_CAP},
        test_util::{init_logger, make_log_data, make_region_state, make_state_kv, new_raft_entry},
    };

    #[test]
    fn test_rfengine() {
        init_logger();
        let tmp_dir = tempfile::tempdir().unwrap();
        let cfg = Config::new(128 * 1024_usize);
        let engine = RfEngine::open(tmp_dir.path(), &cfg, None, None).unwrap();
        assert!(engine.is_empty());

        let init_stats = engine.get_engine_stats();
        assert_eq!(init_stats.total_mem_size, 0);
        assert_eq!(init_stats.total_mem_entries, 0);
        assert!(init_stats.num_files > 0);
        assert!(init_stats.disk_size > 0);
        assert_eq!(init_stats.pending_compaction_wals, 0);
        assert_eq!(init_stats.top_10_size_peers.len(), 0);

        let mut wb = WriteBatch::new();
        for peer_id in 1..=10_u64 {
            let (key, val) = make_state_kv(2, 1);
            let region_id = peer_id + 1;
            wb.set_state(peer_id, region_id, key.chunk(), val.chunk());
        }
        engine.write(wb).unwrap();
        assert!(!engine.is_empty());

        let mut truncated_regions = vec![];
        let mut truncated_idx = 0;
        for idx in 1..=1050_u64 {
            let mut wb = WriteBatch::new();
            for peer_id in 1..=10_u64 {
                let region_id = peer_id + 1;
                if peer_id == 1 {
                    if idx > 100 && idx < 900 {
                        continue;
                    } else if idx == 900 {
                        wb.truncate_raft_log(peer_id, region_id, 899);
                    }
                }
                wb.append_raft_log(peer_id, region_id, &make_log_data(idx, 128));
                let (key, val) = make_state_kv(1, idx);
                wb.set_state(peer_id, region_id, key.chunk(), val.chunk());
                if idx % 100 == 0 && peer_id != 1 {
                    truncated_idx = idx - 100;
                    truncated_regions.push((peer_id, region_id, truncated_idx));
                    wb.truncate_raft_log(peer_id, region_id, truncated_idx);
                }
            }
            engine.write(wb).unwrap();
        }
        assert_eq!(engine.peers.len(), 10);
        for peer_id in 2..=10_u64 {
            assert_eq!(engine.get_truncated_index(peer_id), Some(truncated_idx));
            assert_eq!(
                engine.index_to_truncate_to_size(peer_id, 0),
                engine.get_last_index(peer_id).unwrap()
            );
            let peer_stats = engine.get_peer_stats(peer_id);
            assert_eq!(peer_stats.peer_id, peer_id);
            assert_eq!(peer_stats.num_logs as u64, 1050 - truncated_idx);
            assert_eq!(peer_stats.truncated_idx, truncated_idx);
        }

        let stats = engine.get_engine_stats();
        assert!(stats.total_mem_size > 0);
        assert_eq!(
            stats.total_mem_entries as u64,
            (1050-900+1) /* peer 1 */ + (1050-truncated_idx)*9 // peer 2 to 10
        );
        assert!(stats.pending_compaction_wals > 0);
        assert_eq!(stats.top_10_size_peers.len(), 10);

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
                    engine.get_truncated_index(*new_data_ref.key()).unwrap() + 1,
                    "peer: {}",
                    *new_data_ref.key(),
                );
                assert_eq!(
                    old_data.raft_logs.last_index(),
                    engine.get_last_index(*new_data_ref.key()).unwrap(),
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
                num_logs_offloaded: 0,
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
                num_logs_offloaded: 0,
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
                            .fetch_raft_entries_to(peer_id, low, high, None, &mut buf, None)
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
                    .fetch_raft_entries_to(1, i, i + 1, None, &mut buf, None)
                    .unwrap(),
                1
            );
            assert_eq!(buf, peer1_entries[..i as usize]);
        }
        assert!(matches!(
            engine.fetch_raft_entries_to(1, 11, 12, None, &mut buf, None),
            Err(Error::EntriesUnavailable),
        ));
        // Test `fetch_entries_to` limits size.
        let mut max_size = 0;
        for (i, entry) in peer1_entries.iter().enumerate() {
            buf.clear();
            max_size += entry.compute_size();
            assert_eq!(
                engine
                    .fetch_raft_entries_to(1, 1, 11, Some(max_size as usize), &mut buf, None)
                    .unwrap(),
                i + 1
            );
            assert_eq!(buf, peer1_entries[..=i]);
        }

        // Test fetch empty logs.
        buf.clear();
        assert_eq!(
            engine
                .fetch_raft_entries_to(1, 1, 1, None, &mut buf, None)
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
                true
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
        engine.add_dependent(1, 2);
        assert!(
            engine
                .dependants
                .get(&1)
                .unwrap()
                .read()
                .unwrap()
                .contains(&1)
        );
        engine.with_dependents(1, |_dep| {
            assert_eq!(_dep.len(), 2);
        });
        engine.remove_dependent(1, 2);
        assert!(engine.has_dependents(1));
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
        assert!(!engine.has_dependents(1));
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
        let compacted_epoch = engine.compacted_epoch.load(Ordering::Relaxed);
        let current_epoch = {
            let writer = engine.writer.lock().unwrap();
            writer.get_epoch_id()
        };
        {
            let mut it = WalIterator::new(dir_path, current_epoch + 1).unwrap();
            let Error::Corruption {
                msg: _,
                epoch_id: _,
                offset,
                data: _,
            } = it.check_wal_header().unwrap_err()
            else {
                panic!("expected corruption error");
            };
            // header epoch mismatch error offset should be 0
            assert_eq!(offset, 0);
        }
        for ep in compacted_epoch + 1..=current_epoch {
            let filename = wal_file_name(dir_path, ep);
            let mut it = WalIterator::new(dir_path, ep).unwrap();
            it.check_wal_header().unwrap();
            let mut offsets = vec![it.offset];
            loop {
                match it.read_batch() {
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
                    for pos in &[0, 4, 8, 12] {
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
        {
            let mut wb = WriteBatch::new();
            let (key, val) = make_region_state(10, 42);
            wb.set_state(1, 2, &key, &val);
            engine.write(wb).unwrap();
        }
        for i in 1..=50 {
            let mut wb = WriteBatch::new();
            wb.append_raft_log(1, 2, &make_log_data(i, 128));
            engine.write(wb).unwrap();
        }

        // Truncate all index.
        let mut wb = WriteBatch::new();
        wb.truncate_raft_log(1, 2, TRUNCATE_ALL_INDEX);
        engine.write(wb).unwrap();

        // Write more batch to trigger WAL compaction.
        {
            let mut wb = WriteBatch::new();
            let (key, val) = make_region_state(11, 43);
            wb.set_state(2, 3, &key, &val);
            engine.write(wb).unwrap();
        }
        for i in 1..=10 {
            let mut wb = WriteBatch::new();
            wb.append_raft_log(2, 3, &make_log_data(i, wal_size));
            engine.write(wb).unwrap();
        }

        // Check no file of peer 1 left.
        wait_for_rlogs_truncated(&engine, 1, 10);
    }

    #[test]
    fn test_init_wal_files() {
        init_logger();
        let tmp_dir = tempfile::tempdir().unwrap();
        init_wal_files(tmp_dir.path(), None, None).unwrap();
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
        init_wal_files(tmp_dir.path(), Some(&wal_sync_dir), None).unwrap();
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
        init_wal_files(tmp_dir.path(), Some(&wal_sync_dir), None).unwrap();
        check_files();

        // wal_secondary_dir is created no matter if wal_sync_dir is provided.
        let wal_secondary_dir = tmp_dir.path().join("wal_secondary_dir");
        init_wal_files(
            tmp_dir.path(),
            Some(&wal_sync_dir),
            Some(&wal_secondary_dir),
        )
        .unwrap();
        assert!(wal_secondary_dir.exists());
        fs::remove_dir(wal_secondary_dir.as_path()).unwrap();
        init_wal_files(tmp_dir.path(), None, Some(&wal_secondary_dir)).unwrap();
        assert!(wal_secondary_dir.exists());
    }

    fn wait_for_rlogs_truncated(en: &RfEngine, peer_id: u64, seconds: usize) {
        let mut ok = false;
        let peer_id_str = format!("{:016x}", peer_id);

        let start_time = Instant::now_coarse();
        let timeout = Duration::from_secs(seconds as u64);
        while start_time.saturating_elapsed() < timeout {
            let read_dir = en.dir.read_dir().unwrap();
            let found = read_dir.into_iter().any(|entry| {
                let filename = entry.unwrap().file_name();
                let filename = filename.to_string_lossy();
                let parts: Vec<_> = filename.as_ref().split('_').collect();
                parts.len() == 3 && parts[0] == peer_id_str
            });
            if !found {
                ok = true;
                break;
            }
            std::thread::sleep(Duration::from_secs(1));
        }

        assert!(ok);
    }

    #[test]
    fn test_get_region_peer_map() {
        let dir = tempfile::tempdir().unwrap();
        let cfg = Config::new(128 * 1024);
        let engine = RfEngine::open(dir.path(), &cfg, None, None).unwrap();

        // ---------------------
        // Region 10: two peers (peer 1 -> older, peer 2 -> newer)
        // ---------------------
        // 1) Write something for peer 1, region 10:
        let mut wb = WriteBatch::new();
        wb.append_raft_log(
            1,
            10,
            &new_raft_entry(EntryType::EntryNormal, 1, 1, b"data", 0),
        );
        engine.write(wb).unwrap();

        // 2) Write something for peer 2, region 10 (the "newer" peer).
        let mut wb = WriteBatch::new();
        wb.append_raft_log(
            2,
            10,
            &new_raft_entry(EntryType::EntryNormal, 2, 1, b"data", 0),
        );
        engine.write(wb).unwrap();

        // At this point, region 10 should map to peer_id = 2 (newer).
        let map = engine.get_region_peer_map();
        assert_eq!(map.get(&10), Some(&2), "peer 2 should override peer 1");

        // 3) Truncate the newer peer to TRUNCATE_ALL_INDEX, which means that region
        // is destroyed.
        let mut wb = WriteBatch::new();
        wb.truncate_raft_log(2, 10, TRUNCATE_ALL_INDEX);
        engine.write(wb).unwrap();

        // Because the newest peer is truncated, region 10 should be removed entirely.
        let map = engine.get_region_peer_map();
        assert!(!map.contains_key(&10), "region 10 should be removed");

        // ---------------------
        // Region 20: multiple peers, confirm the highest ID remains.
        // ---------------------
        // Add peer 5 (older) and peer 7 (newer) to the same region 20.
        for pid in [5u64, 7u64] {
            let mut wb = WriteBatch::new();
            wb.append_raft_log(
                pid,
                20,
                &new_raft_entry(EntryType::EntryNormal, pid, 1, b"data", 0),
            );
            engine.write(wb).unwrap();
        }
        // get_region_peer_map should pick peer 7 for region 20.
        let map = engine.get_region_peer_map();
        assert_eq!(map.get(&20), Some(&7), "peer 7 should override peer 5");
    }

    #[test]
    fn test_load_store_region_states() {
        init_logger();
        let tmp_dir = tempfile::tempdir().unwrap();
        let cfg = Config::new(128 * 1024_usize);
        let engine = RfEngine::open(tmp_dir.path(), &cfg, None, None).unwrap();

        let cluster_id = 1;
        let store_id = 2;
        let engine_id = 3;

        engine.set_engine_id(engine_id);
        assert_eq!(engine.get_engine_id(), engine_id);

        let mut ident = StoreIdent::default();
        ident.set_cluster_id(cluster_id);
        ident.set_store_id(store_id);
        let bin = ident.write_to_bytes().unwrap();
        let mut wb = WriteBatch::new();
        wb.set_state(0, 0, STORE_IDENT_KEY, bin.as_slice());
        engine.write(wb).unwrap();

        let loaded_ident = load_store_ident(&engine).unwrap();
        assert_eq!(loaded_ident.cluster_id, cluster_id);
        assert_eq!(loaded_ident.store_id, store_id);

        let peer_id = 5;
        let region_id = 6;
        let region_epoch = 11;
        let mut wb = WriteBatch::new();
        let (key, val) = make_region_state(region_epoch, 12 /* keyspace_id */);
        wb.set_state(peer_id, region_id, &key, &val);
        engine.write(wb).unwrap();

        let state = engine.load_region_state(peer_id, region_epoch);
        assert!(state.is_some(), "the region state should not be None");
    }

    #[test]
    fn test_calc_offload_epoch() {
        let current_epoch = Arc::new(AtomicU32::new(10)); // unchanged
        let compacted_epoch = Arc::new(AtomicU32::new(0));
        for (count, compacted, expected) in [
            (0, 10, 0),
            (0, 9, 0),
            (5, 4, 4),
            (5, 5, 5),
            (5, 6, 5),
            (10, 5, 0),
            (10, 10, 0),
            (20, 10, 0),
        ] {
            compacted_epoch.store(compacted, Ordering::SeqCst);
            let offload_epoch = calc_offload_epoch(count, &current_epoch, &compacted_epoch);
            assert_eq!(
                offload_epoch, expected,
                "current_epoch=10, compacted={}, in_mem_rlog_epoch_count={}, expected offload_epoch={}, got {}",
                compacted, count, expected, offload_epoch,
            );
        }
    }

    #[test]
    fn test_find_offload_index() {
        let mut logs = RaftLogs::default();
        let index_and_terms = vec![
            (20, 5),
            (21, 5),
            (22, 5),
            (23, 7),
            (24, 7),
            (25, 7),
            (26, 9),
            (27, 9),
            (28, 9),
        ];
        for (index, term) in index_and_terms {
            let mut e = Entry::default();
            e.index = index;
            e.term = term;
            logs.append(RaftLogOp::new(&e));
        }

        let files = VecDeque::from(vec![
            PeerFile::new(1, 15, 19, 4), // offloaded
            PeerFile::new(2, 20, 21, 5), // term matched
            PeerFile::new(0, 22, 22, 5), // rlog epoch == 0 (skipped)
            PeerFile::new(4, 23, 25, 6), // term mismatch
            PeerFile::new(5, 23, 24, 7), // term matched
            PeerFile::new(0, 25, 25, 7), // rlog epoch == 0 (skipped)
            PeerFile::new(7, 26, 30, 8), // term mismatch
            PeerFile::new(8, 26, 27, 9), // term matched
        ]);

        assert_eq!(find_offload_index(&files, &logs, 0), None);
        assert_eq!(find_offload_index(&files, &logs, 1), None);
        assert_eq!(find_offload_index(&files, &logs, 2), Some(21));
        assert_eq!(find_offload_index(&files, &logs, 3), Some(21));
        assert_eq!(find_offload_index(&files, &logs, 4), Some(21));
        assert_eq!(find_offload_index(&files, &logs, 5), Some(24));
        assert_eq!(find_offload_index(&files, &logs, 6), Some(24));
        assert_eq!(find_offload_index(&files, &logs, 7), Some(24));
        assert_eq!(find_offload_index(&files, &logs, 8), Some(27));

        // Edge cases: empty raft log or empty rlog files
        for offload_epoch in 0..=10 {
            assert_eq!(
                find_offload_index(&VecDeque::new(), &logs, offload_epoch),
                None
            );
            assert_eq!(
                find_offload_index(&files, &RaftLogs::default(), offload_epoch),
                None
            );
        }
    }

    #[test]
    fn test_rfengine_fetch_offloaded() {
        init_logger();
        const STATE_PREFIX: u8 = b'p';

        let dir = tempfile::tempdir().unwrap();
        let mut cfg = Config::new(8 * 1024); // WAL size
        cfg.rlog_soft_memory_limit = ReadableSize(8 * 1024); // Offload immediately after compaction.

        let engine = RfEngine::open(dir.path(), &cfg, None, None).unwrap();
        // Record the entries and states inserted into rfengine. Will be used
        // for comparison later.
        let mut data_map = HashMap::new();
        // Track the next index that should be inserted for each peer.
        let mut next_index = [1u64; 2];
        for peer_id in [1, 2, 1, 2] {
            let mut wb = WriteBatch::new();
            for i in 1..=1000 {
                let entry_index = next_index[peer_id - 1];
                next_index[peer_id - 1] += 1;

                let peer_id = peer_id as u64;
                let region_id = peer_id + 1000;
                let entry =
                    new_raft_entry(EntryType::EntryNormal, peer_id, entry_index, b"data", 0);
                let (state_key, state_val) = (&[STATE_PREFIX, i as u8], &[i as u8]);
                wb.append_raft_log(peer_id, region_id, &entry);
                wb.set_state(peer_id, region_id, state_key, state_val);

                let (entries, states) = data_map
                    .entry(peer_id)
                    .or_insert_with(|| (vec![], BTreeMap::new()));
                entries.push(entry);
                states.insert(state_key.to_vec(), state_val.to_vec());
            }
            engine.write(wb).unwrap();
            // Wait for the compaction to finish.
            eventually(Duration::from_millis(100), Duration::from_secs(1), || {
                engine.current_epoch_id.load(Ordering::SeqCst)
                    == engine.compacted_epoch.load(Ordering::SeqCst) + 1
            });
        }
        assert_eq!(engine.get_last_index(1), Some(2000));
        assert_eq!(engine.get_last_index(2), Some(2000));

        // Overwrite one entry at index 500 on peer 2.
        let mut wb = WriteBatch::new();
        wb.append_raft_log(
            2,
            1002,
            &new_raft_entry(EntryType::EntryNormal, 10_u64, 500, b"data", 0),
        );
        engine.write(wb).unwrap();
        assert_eq!(engine.get_last_index(2), Some(500));

        let peer_meta_1 = extract_peer_meta(&engine, 1, |raft_logs| {
            // Entries [1,1000] are offloaded, [1001,2000] are eligible for
            // offloading but still in memory.
            assert_eq!(raft_logs.first_index(), 1001);
        });
        let peer_meta_2 = extract_peer_meta(&engine, 2, |_| {});
        check_engine_data(&engine, &data_map);

        // Restart rfengine.
        engine.stop_worker(true);
        let engine = RfEngine::open(dir.path(), &cfg, None, None).unwrap();

        // Check rfengine again after the restart.
        check_engine_data(&engine, &data_map);
        assert_eq!(
            peer_meta_1,
            extract_peer_meta(&engine, 1, |raft_logs| {
                // all entries are offloaded after the restart.
                assert_eq!(raft_logs.first_index(), 0);
            })
        );
        assert_eq!(peer_meta_2, extract_peer_meta(&engine, 2, |_| {}));
    }

    // Extracts and returns the peer meta after applying a custom check on its
    // raft logs.
    fn extract_peer_meta<F>(engine: &RfEngine, peer_id: u64, raft_logs_check: F) -> PeerMeta
    where
        F: FnOnce(&RaftLogs),
    {
        let peer_ref = engine.peers.get(&peer_id).unwrap();
        let peer = peer_ref.read().unwrap();
        raft_logs_check(&peer.raft_logs);
        peer.meta.clone()
    }

    // Tests raft log fetching across in-memory and offloaded entries.
    fn check_engine_data(
        engine: &RfEngine,
        data_map: &HashMap<u64, (Vec<Entry>, BTreeMap<Vec<u8>, Vec<u8>>)>,
    ) {
        assert_eq!(engine.get_term(1, 100), Some(1));
        assert_eq!(engine.get_term(1, 1000), Some(1)); // offloaded entry
        assert_eq!(engine.get_term(1, 1001), Some(1));
        assert!(engine.get_raft_entry(1, 1000).is_some());
        assert!(engine.get_raft_entry(1, 2000).is_some());

        // Test `fetch_entries_to`.
        let mut buf = vec![];

        for (low, high) in [(100, 200), (900, 1100), (1200, 1300), (1, 2001)] {
            assert_eq!(
                engine
                    .fetch_raft_entries_to(1, low, high, None, &mut buf, None)
                    .unwrap(),
                (high - low) as usize
            );
            assert_eq!(
                data_map.get(&1).unwrap().0[(low - 1) as usize..(high - 1) as usize],
                buf
            );
            buf.clear();
        }

        // Test fetch unavailable entries.
        assert!(matches!(
            engine.fetch_raft_entries_to(1, 2001, 2002, None, &mut buf, None),
            Err(Error::EntriesUnavailable),
        ));

        // Test fetch empty logs.
        buf.clear();
        assert_eq!(
            engine
                .fetch_raft_entries_to(1, 1, 1, None, &mut buf, None)
                .unwrap(),
            0
        );
        assert!(buf.is_empty());

        // Test `fetch_entries_to` should push logs to the buf.
        let peer1_entries = &data_map.get(&1).unwrap().0;
        for i in 1..=10 {
            assert_eq!(
                engine
                    .fetch_raft_entries_to(1, i, i + 1, None, &mut buf, None)
                    .unwrap(),
                1
            );
            assert_eq!(buf, peer1_entries[..i as usize]);
        }

        // Test `fetch_entries_to` should respect size limit.
        let mut max_size = 0;
        for (i, entry) in peer1_entries.iter().enumerate() {
            if i == 10 {
                break;
            }
            buf.clear();
            max_size += entry.compute_size();
            assert_eq!(
                engine
                    .fetch_raft_entries_to(1, 1, 11, Some(max_size as usize), &mut buf, None)
                    .unwrap(),
                i + 1
            );
            assert_eq!(buf, peer1_entries[..=i]);
        }

        // Peer 2: entries [1, 499] are offloaded, entry at index 500 is
        // in-memory.
        assert_eq!(engine.get_term(2, 100), Some(2));
        assert_eq!(engine.get_term(2, 499), Some(2));
        assert_eq!(engine.get_term(2, 500), Some(10));
        assert_eq!(engine.get_term(2, 501), None);
        assert!(matches!(
            engine.fetch_raft_entries_to(2, 501, 502, None, &mut buf, None),
            Err(Error::EntriesUnavailable),
        ));

        buf.clear();
        assert_eq!(
            engine
                .fetch_raft_entries_to(2, 1, 501, None, &mut buf, None)
                .unwrap(),
            500
        );

        let last_idx = buf.len() - 1;
        let second_last_idx = last_idx - 1;
        assert_eq!(buf[last_idx].term, 10);
        assert_eq!(buf[second_last_idx].term, 2);
    }

    #[test]
    fn test_peer_index_to_truncate_to_size() {
        let mut peer = PeerData::default();
        peer.peer_id = 1;

        // Insert 4 blocks of logs into the peer.
        let data = b"data";
        let total_cnt = RAFT_LOG_BLOCK_CAP * 4;
        let total_size = total_cnt * data.len();
        for i in 1..=total_cnt {
            let log = RaftLogOp::new(&new_raft_entry(
                EntryType::EntryNormal,
                1,
                i as u64,
                data,
                0,
            ));
            peer.raft_logs.append(log);
        }

        fn check_truncate_idx(peer: &PeerData, total_cnt: usize, total_size: usize) {
            let q = total_cnt as u64 / 4;
            assert_eq!(peer.index_to_truncate_to_size(total_size + 1), 0);
            assert_eq!(peer.index_to_truncate_to_size(total_size / 4 * 3 + 1), q);
            assert_eq!(peer.index_to_truncate_to_size(total_size / 2 + 1), q * 2);
            assert_eq!(peer.index_to_truncate_to_size(total_size / 4 + 1), q * 3);
        }

        for i in 0..=3 {
            // Simulate progressive offloading by truncating more in-memory logs.
            peer.raft_logs
                .truncate(i as u64 * RAFT_LOG_BLOCK_CAP as u64);
            check_truncate_idx(&peer, total_cnt, total_size);
        }
    }
}
