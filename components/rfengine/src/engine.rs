// Copyright 2021 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    collections::{BTreeMap, HashMap, HashSet},
    fmt::{Display, Formatter},
    fs,
    fs::{create_dir_all, OpenOptions},
    ops::{Deref, DerefMut},
    os::unix::fs::{FileExt, MetadataExt},
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicU32, AtomicU64, Ordering},
        Arc, Mutex, RwLock,
    },
    thread::{self, JoinHandle},
};

use bytes::{Buf, Bytes};
use dashmap::mapref::one::Ref;
use engine_traits::ObjectStorage;
use file_system::open_direct_file;
use protobuf::Message;
use raft_proto::{eraftpb, eraftpb::Entry};
use rfenginepb::ClusterBackupMeta;
use tikv_util::{info, mpsc::Sender, time::Instant, warn};

use crate::{
    log_batch::{RaftLogBlock, RaftLogs},
    manifest::{manifest_path, persist_change_set, Manifest},
    metrics::*,
    write_batch::{PeerBatch, WriteBatch},
    *,
};

pub const TRUNCATE_ALL_INDEX: u64 = u64::MAX;

/// `RfEngine` is a persistent storage engine for multi-raft logs.
/// It stores part of raft logs and states(key/value pair) in memory and persists them to disk.
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
/// `RfEngine` contains all raft group states and non-truncated logs in memory, so that it
/// can get raft logs quickly.
///
/// # WAL
///
/// `RfEngine` writes all raft groups' logs and states to a WAL file sequentially.
/// When the WAL file size exceeds the threshold, it triggers rotation and switching to a new WAL file.
/// The name of a WAL file is `{epoch}.wal`. Epoch increases when rotating.
///
/// ## Rotation
///
/// Rotation splits the data of a WAL file to several files:
///   - `{epoch}.states`: Contains **all** raft groups states, not just states in the corresponding WAL file.
///   The old states file will be removed after rewriting.
///
///   - `{epoch}_{region_id}_{first_log_index}_{last_log_index}.rlog`: Contains logs in
///   [first_log_index, last_log_index) of a single raft group.
///
/// After splitting, the WAL file is moved to the `recycle` directory for later use.
/// `RfEngine` recycles old WAL files for better I/O performance. To distinguish between
/// old data and new data, the data format of WAL contains epoch, i.e., valid data's epoch equals the
/// epoch in the WAL file name.
///
/// # Garbage Collection
///
/// Raft logs that has been applied and persisted to FSM can be truncated. All in-memory logs and `rlog` files before the
/// truncated index will be removed.
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
    pub fn open(dir: &Path, wal_size: usize, compression_threshold: usize) -> Result<Self> {
        let core = RfEngineCore::open(dir, wal_size, compression_threshold)?;
        Ok(Self {
            core: Arc::new(core),
        })
    }
}

pub struct RfEngineCore {
    pub dir: PathBuf,

    pub(crate) writer: Mutex<WalWriter>,

    pub(crate) peers: dashmap::DashMap<u64, RwLock<PeerData>>,

    pub(crate) dependants: dashmap::DashMap<u64, RwLock<HashSet<u64>>>,

    pub(crate) task_sender: Sender<Task>,

    pub(crate) worker_handle: Mutex<WorkerHandle>,

    pub(crate) engine_id: Arc<AtomicU64>,
}

pub(crate) struct WorkerHandle {
    task_sender: Sender<Task>,
    handle: Option<JoinHandle<()>>,
}

impl RfEngineCore {
    fn open(dir: &Path, wal_size: usize, compression_threshold: usize) -> Result<Self> {
        maybe_create_wal_files(dir)?;
        let engine_id = Arc::new(AtomicU64::new(0));
        let manifest = Manifest::open(dir, engine_id.clone())?;
        let (tx, rx) = tikv_util::mpsc::unbounded();
        let compacted_epoch = Arc::new(AtomicU32::new(manifest.epoch_id));
        let writer = WalWriter::new(
            dir,
            wal_size,
            compression_threshold,
            compacted_epoch.clone(),
        );
        let mut en = Self {
            dir: dir.to_owned(),
            peers: Default::default(),
            dependants: Default::default(),
            writer: Mutex::new(writer),
            task_sender: tx.clone(),
            worker_handle: Mutex::new(WorkerHandle {
                task_sender: tx,
                handle: None,
            }),
            engine_id,
        };
        en.load(&manifest)?;
        {
            let mut worker = Worker::new(dir.to_owned(), rx, manifest, compacted_epoch);
            let join_handle = thread::spawn(move || worker.run());
            en.worker_handle.lock().unwrap().handle = Some(join_handle);
        }

        Ok(en)
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

    /// Persists the write batch to WAL. It can be used in another thread to implement async I/O,
    /// i.e., call `apply` in the main thread and call `persist` in the I/O thread.
    pub fn persist(&self, wb: WriteBatch) -> Result<usize> {
        let timer = Instant::now_coarse();
        let mut writer = self.writer.lock().unwrap();
        let epoch_id = writer.epoch_id;
        for data in wb.peers.values() {
            writer.append_region_data(data);
        }
        let (size, rotated) = writer.flush()?;
        if rotated {
            self.task_sender.send(Task::Rotate { epoch_id }).unwrap();
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

    /// Get the value of the last state key with the `prefix`. `prefix` must be non-empty.
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

    /// Iterates states of the region in order or in desc order if `desc` is true until `f` returns
    /// error.
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

    /// Iterates stats of all regions in order or in desc order if `desc` is true and breaks one
    /// regions iteration if `f` returns false.
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

    pub fn stop_worker(&self) {
        let mut handle_ref = self.worker_handle.lock().unwrap();
        if let Some(h) = handle_ref.handle.take() {
            handle_ref.task_sender.send(Task::Close).unwrap();
            h.join().unwrap();
        }
    }

    /// After split and before the new region is initially flushed, the old region's raft log
    /// can not be truncated, otherwise, it would not be able to recover the new region.
    /// So we can call `add_dependent` after split to protect the raft log.
    /// After the new region is initially flushed or re-ingested or destroyed, call
    /// `remove_dependent` to resume truncating the raft log.
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

    /// Returns the index that truncating to the given index can limit the memory usage to size.
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
        // ensure the newer peer_id appear after the older peer_id, so it can replace older.
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

    pub fn backup(&self, mut task: BackupTask) {
        let writer = self.writer.lock().unwrap();
        task.file_off = writer.file_off;
        self.task_sender.send(Task::Backup(task)).unwrap();
    }
}

pub fn restore(
    object_storage: Box<dyn ObjectStorage>,
    cluster_backup: &ClusterBackupMeta,
    store_id: u64,
    dir: &Path,
) {
    let store_meta = cluster_backup
        .get_stores()
        .iter()
        .find(|x| x.store_id == store_id)
        .expect("store not found");
    maybe_create_wal_files(dir).unwrap();
    let wal_chunks = store_meta.get_wal_chunks();
    if !wal_chunks.is_empty() {
        let keys: Vec<String> = wal_chunks
            .iter()
            .map(|chunk| wal_file_key(store_id, chunk.epoch, chunk.start_off, chunk.end_off))
            .collect();
        let mut objects = object_storage.get_objects(keys).unwrap();
        objects.sort_by(|(a, _), (b, _)| a.cmp(b));
        let wal_path = wal_file_name(dir, store_meta.get_manifest().epoch_id + 1);
        let file = OpenOptions::new().write(true).open(&wal_path).unwrap();
        let mut i = 0;
        for (_, data) in objects {
            file.write_at(&data, store_meta.get_wal_chunks()[i].start_off)
                .unwrap();
            i += 1;
        }
        let end_off = wal_chunks.last().unwrap().end_off;
        let eof = vec![0u8; 4096];
        file.write_at(&eof, end_off).unwrap();
        file.sync_data().unwrap();
    }
    let mut key_path_map = HashMap::new();
    for peer in store_meta.get_manifest().get_peers() {
        for file in peer.get_files() {
            let rlog_key = raft_log_file_key(
                store_id,
                peer.get_peer_id(),
                file.get_first_index(),
                file.get_last_index(),
            );
            let path = raft_log_file_name(dir, peer.peer_id, file.first_index, file.last_index);
            key_path_map.insert(rlog_key, path);
        }
    }
    let keys: Vec<String> = key_path_map.keys().cloned().collect();
    let objects = object_storage.get_objects(keys).unwrap();
    for (key, data) in objects {
        let path = key_path_map.get(&key).unwrap();
        fs::write(path, &data).unwrap();
    }
    let manifest_file = OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(&manifest_path(dir))
        .unwrap();
    if store_meta.has_manifest() {
        persist_change_set(&manifest_file, 0, store_meta.get_manifest()).unwrap();
    }
}

pub(crate) fn maybe_create_wal_files(dir: &Path) -> Result<()> {
    if !dir.exists() {
        create_dir_all(dir)?;
    }
    // create 4 wal files and always reuse them, so we never need to sync dir on writer thread.
    for i in 0..4 {
        let file_path = dir.join(format!("{}.wal", i));
        let _ = open_direct_file(&file_path, true)?;
    }
    file_system::sync_dir(dir)?;
    Ok(())
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
        if self.truncated_idx == TRUNCATE_ALL_INDEX && truncated_index > 0 {
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
    use std::{
        collections::HashMap, fs, fs::OpenOptions, io::BufReader, os::unix::prelude::FileExt,
    };

    use bytes::{BufMut, BytesMut};
    use engine_traits::Error as TraitError;
    use eraftpb::{Entry, EntryType};
    use protobuf::Message;
    use slog::o;

    use super::*;
    use crate::log_batch::RaftLogOp;

    #[test]
    fn test_rfengine() {
        init_logger();
        let tmp_dir = tempfile::tempdir().unwrap();
        let wal_size = 128 * 1024_usize;
        let compression_threshold = 8 * 1024_usize;
        let engine = RfEngine::open(tmp_dir.path(), wal_size, compression_threshold).unwrap();
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
        engine.stop_worker();

        for _ in 0..2 {
            let engine = RfEngine::open(tmp_dir.path(), wal_size, compression_threshold).unwrap();
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

    fn init_logger() {
        use slog::Drain;
        let decorator = slog_term::PlainDecorator::new(std::io::stdout());
        let drain = slog_term::CompactFormat::new(decorator).build();
        let drain = std::sync::Mutex::new(drain).fuse();
        let logger = slog::Logger::root(drain, o!());
        slog_global::set_global(logger);
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
        const STATE_PREFIX: u8 = b'p';

        let dir = tempfile::tempdir().unwrap();
        let engine = RfEngine::open(dir.path(), 128 * 1024, 8 * 1024).unwrap();

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
        assert!(engine.write(wb).is_ok());

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
        let tmp_dir = tempfile::tempdir().unwrap();
        let wal_size = 128 * 1024_usize;
        let compression_threshold = 8 * 1024_usize;
        let dir_path = tmp_dir.path();
        let engine = RfEngine::open(dir_path, wal_size, compression_threshold).unwrap();
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
        engine.stop_worker();
        for _ in 0..2 {
            let engine = RfEngine::open(dir_path, wal_size, compression_threshold).unwrap();
            assert_eq!(engine.peers.len(), 10);
            engine.stop_worker();
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
            let mut it = WALIterator::new(dir_path.to_owned(), ep);
            let fd = fs::File::open(filename.clone()).unwrap();
            let mut buf_reader = BufReader::new(fd);
            let wal_header = it.check_wal_header(&mut buf_reader).unwrap();
            let mut offsets = vec![it.offset];
            loop {
                match it.read_batch(&mut buf_reader, &wal_header) {
                    Err(err) => {
                        if let Error::EOF = err {
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
                        buf[*pos] += 1;
                        fd.write_all_at(buf.as_ref(), *offset).unwrap();
                        fd.sync_data().unwrap();
                        assert!(RfEngine::open(dir_path, wal_size, compression_threshold).is_err());
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
        let compression_threshold = 8 * 1024_usize;
        let engine = RfEngine::open(tmp_dir.path(), wal_size, compression_threshold).unwrap();
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
}
