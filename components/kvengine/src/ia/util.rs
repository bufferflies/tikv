// Copyright 2024 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    collections::HashMap,
    fs,
    io::Write,
    os::unix::fs::FileExt,
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicU64, Ordering::Relaxed},
        Arc,
    },
    time::Duration,
};

use async_trait::async_trait;
use bytes::{Buf, Bytes};
use dashmap::DashMap;
use quick_cache::sync::Cache;
use tikv_util::config::{AbsoluteOrPercentSize, ReadableDuration};

use crate::{
    ia::{
        manager::{IaManagerOptions, QueueOptions},
        types::SegmentHandle,
    },
    table::{file::LocalFile, Error, Result},
    IoContext,
};

pub const TEMPORARY_FILE_SUFFIX: &str = "tmp";

/// LocalStore is used to provide uniform access to both local disk and memory.
#[async_trait]
pub trait LocalStore: Send + Sync {
    /// Return the path of store. Will be `None` when it's in memory.
    fn path(&self) -> Option<&Path>;

    fn init(&self) -> Result<()>;

    /// Return the existed keys & suffixes in the store from last startup.
    fn scan(&self) -> Result<HashMap<String /* suffix */, Vec<String> /* keys */>>;

    fn save(&self, file_id: u64, key: &str, data: Bytes) -> Result<()>;

    /// Read exactly `buf.len()` bytes into `buf` starting at `offset`.
    fn read_at(&self, file_id: u64, key: &str, buf: &mut [u8], offset: u64) -> Result<Option<()>>;

    fn read(&self, file_id: u64, key: &str, start_off: u64, end_off: u64) -> Result<Option<Bytes>>;

    fn remove(&self, file_id: u64, key: &str) -> Result<Option<()>>;

    /// Get a handle to hold the underlying data from being removed.
    fn handle(&self, file_id: u64, key: &str) -> Result<Option<SegmentHandle>>;
}

pub fn new_local_store(path: Option<PathBuf>, fd_cache_capacity: usize) -> Arc<dyn LocalStore> {
    if let Some(path) = path {
        Arc::new(LocalFileStore::new(path, fd_cache_capacity)) as _
    } else {
        Arc::new(LocalMemoryStore::default()) as _
    }
}

macro_rules! try_fs {
    ($e:expr) => {{
        match $e {
            Ok(f) => Ok(f),
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
                return Ok(None);
            }
            Err(err) => Err(err),
        }
    }};
}

macro_rules! try_open {
    ($path:expr) => {{ try_fs!(std::fs::File::open($path)) }};
}

#[macro_export]
macro_rules! try_some {
    ($expr:expr) => {{
        match $expr {
            Some(v) => v,
            None => return Ok(None),
        }
    }};
}

pub struct LocalFileStore {
    dir: PathBuf,
    fd_cache: Arc<Cache<String, Arc<std::fs::File>>>,
}

impl LocalFileStore {
    pub fn new(dir: PathBuf, fd_cache_capacity: usize) -> Self {
        let fd_cache = Arc::new(Cache::new(fd_cache_capacity));
        Self { dir, fd_cache }
    }

    fn open_file(&self, file_id: u64, key: &str) -> Result<Option<Arc<std::fs::File>>> {
        let f = if let Some(f) = self.fd_cache.get(key) {
            debug!("FileDataStore.open_file cache hit"; "file_id" => file_id, "key" => key);
            f.clone()
        } else {
            let path = self.dir.join(key);
            debug!("FileDataStore.open_file cache miss"; "file_id" => file_id, "key" => key, "path" => ?path);
            let f = try_open!(path).table_ctx(file_id, format!("open.{key}"))?;
            let arc_f = Arc::new(f);
            self.fd_cache.insert(key.to_owned(), arc_f.clone());
            arc_f
        };
        Ok(Some(f))
    }
}

#[async_trait]
impl LocalStore for LocalFileStore {
    fn path(&self) -> Option<&Path> {
        Some(&self.dir)
    }

    fn init(&self) -> Result<()> {
        fs::create_dir_all(&self.dir).table_ctx(0, format!("create_dir.{:?}", self.dir))
    }

    fn scan(&self) -> Result<HashMap<String /* suffix */, Vec<String> /* keys */>> {
        let entries = fs::read_dir(&self.dir).table_ctx(0, format!("read_dir.{:?}", self.dir))?;
        let mut map = HashMap::new();
        for entry in entries {
            let entry = entry.table_ctx(0, "entry")?;
            let path = entry.path();
            if path.is_file() {
                let key = path.file_name().unwrap().to_string_lossy().into_owned();
                let suffix = path
                    .extension()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .into_owned();
                map.entry(suffix).or_insert_with(Vec::new).push(key);
            }
        }
        Ok(map)
    }

    fn save(&self, file_id: u64, key: &str, data: Bytes) -> Result<()> {
        lazy_static::lazy_static! {
            static ref TMP_ID: AtomicU64 = AtomicU64::new(0);
        }

        let tmp_filename = format!(
            "{}.{}.{}",
            key,
            TMP_ID.fetch_add(1, Relaxed),
            TEMPORARY_FILE_SUFFIX
        );
        let tmp_path = self.dir.join(tmp_filename);
        let mut f = fs::File::create(&tmp_path).table_ctx(file_id, format!("create_tmp.{key}"))?;

        f.write_all(&data)
            .table_ctx(file_id, format!("write_tmp.{key}"))?;
        f.sync_data()
            .table_ctx(file_id, format!("sync_tmp.{key}"))?;

        let path = self.dir.join(key);
        fs::rename(&tmp_path, &path).table_ctx(file_id, format!("rename.{key}"))?;

        debug!("FileDataStore.save"; "file_id" => file_id, "key" => key, "path" => ?path);
        Ok(())
    }

    fn read_at(&self, file_id: u64, key: &str, buf: &mut [u8], offset: u64) -> Result<Option<()>> {
        let f = try_some!(self.open_file(file_id, key)?);
        f.read_exact_at(buf, offset)
            .table_ctx(file_id, format!("read_exact.{key}"))?;
        Ok(Some(()))
    }

    fn read(&self, file_id: u64, key: &str, start_off: u64, end_off: u64) -> Result<Option<Bytes>> {
        let mut buf = vec![0; (end_off - start_off) as usize];
        try_some!(self.read_at(file_id, key, &mut buf, start_off)?);
        Ok(Some(Bytes::from(buf)))
    }

    fn remove(&self, file_id: u64, key: &str) -> Result<Option<()>> {
        let path = self.dir.join(key);
        self.fd_cache.remove(key);
        try_fs!(std::fs::remove_file(path)).table_ctx(file_id, format!("remove.{key}"))?;
        Ok(Some(()))
    }

    fn handle(&self, file_id: u64, key: &str) -> Result<Option<SegmentHandle>> {
        let f = try_some!(self.open_file(file_id, key)?);
        let path = self.dir.join(key);
        let local_file = LocalFile::from_file(file_id, path, f)?;
        Ok(Some(local_file.into()))
    }
}

#[derive(Default)]
pub(crate) struct LocalMemoryStore {
    m: DashMap<String, Bytes>,
}

impl LocalMemoryStore {
    fn get(&self, key: &str) -> Option<Bytes> {
        self.m.get(key).map(|x| x.value().clone())
    }

    fn get_with_check(&self, key: &str, end_off: u64) -> Result<Option<Bytes>> {
        let data = try_some!(self.get(key));
        if end_off > data.len() as u64 {
            return Err(Error::Io(format!(
                "read out of range, key: {}, end_off: {}, data length: {}",
                key,
                end_off,
                data.len()
            )));
        }
        Ok(Some(data))
    }
}

#[async_trait]
impl LocalStore for LocalMemoryStore {
    fn path(&self) -> Option<&Path> {
        None
    }

    fn init(&self) -> Result<()> {
        Ok(())
    }

    fn scan(&self) -> Result<HashMap<String /* suffix */, Vec<String> /* keys */>> {
        Ok(HashMap::new())
    }

    fn save(&self, _file_id: u64, key: &str, data: Bytes) -> Result<()> {
        self.m.insert(key.to_owned(), data);
        Ok(())
    }

    fn read_at(&self, _file_id: u64, key: &str, buf: &mut [u8], offset: u64) -> Result<Option<()>> {
        let end_off = offset + buf.len() as u64;
        let data = try_some!(self.get_with_check(key, end_off)?);
        buf.copy_from_slice(&data[offset as usize..end_off as usize]);
        Ok(Some(()))
    }

    fn read(
        &self,
        _file_id: u64,
        key: &str,
        start_off: u64,
        end_off: u64,
    ) -> Result<Option<Bytes>> {
        let data = try_some!(self.get_with_check(key, end_off)?);
        Ok(Some(Bytes::copy_from_slice(
            data.slice(start_off as usize..end_off as usize).chunk(),
        )))
    }

    fn remove(&self, _file_id: u64, key: &str) -> Result<Option<()>> {
        Ok(self.m.remove(key).map(|_| ()))
    }

    fn handle(&self, file_id: u64, key: &str) -> Result<Option<SegmentHandle>> {
        let data = try_some!(self.get(key));
        Ok(Some(SegmentHandle::from_bytes(file_id, data)))
    }
}

const IA_SEGMENT_SIZE_DEF: i64 = 1 << 20; // 1MiB
const IA_FREQ_UPDATE_INTERVAL_DEF: Duration = Duration::from_secs(60);
const IA_DFS_CONCURRENCY_DEF: usize = 64;
const IA_DFS_KEYSPACE_CONCURRENCY_DEF: usize = 20;
const IA_FD_CACHE_CAPACITY_DEF: usize = 102400; // 100k
const IA_TABLE_META_MTIME_INTERVAL_DEF: Duration = Duration::from_secs(3600); // 1 hour

const MAIN_QUEUE_CAPACITY_FACTOR: i64 = 10; // Main queue is 10x larger than small queue.

pub enum IaCapacity {
    Manual {
        small_queue: QueueOptions,
        main_queue: QueueOptions,
    },
    MemoryCap(AbsoluteOrPercentSize),
    /// Small queue in memory & main queue in disk.
    MemoryAndDiskCap(
        AbsoluteOrPercentSize, // mem_cap
        PathBuf,
        AbsoluteOrPercentSize, // disk_cap
    ),
}

impl Default for IaCapacity {
    fn default() -> Self {
        IaCapacity::MemoryCap(AbsoluteOrPercentSize::Percent(20.0))
    }
}

impl IaCapacity {
    fn build_options(self, options: &mut IaManagerOptions) -> Result<()> {
        match self {
            IaCapacity::Manual {
                small_queue,
                main_queue,
            } => {
                options.small_queue = small_queue;
                options.main_queue = main_queue;
            }
            IaCapacity::MemoryCap(cap) => {
                let mem_cap = cap.as_memory_size() as i64;
                options.small_queue.path = None;
                options.small_queue.cap = mem_cap / MAIN_QUEUE_CAPACITY_FACTOR;
                options.main_queue.path = None;
                options.main_queue.cap = mem_cap - options.small_queue.cap;
            }
            IaCapacity::MemoryAndDiskCap(mem_cap, local_dir, disk_cap) => {
                let mem_cap = mem_cap.as_memory_size();
                options.small_queue.path = None;
                options.small_queue.cap = mem_cap as i64;

                let disk_cap = disk_cap
                    .as_disk_size(&local_dir)
                    .map_err(|err| Error::Io(format!("get disk capacity failed: {err:?}")))?;
                options.main_queue.path = Some(local_dir);
                options.main_queue.cap = disk_cap as i64;
            }
        }
        Ok(())
    }

    #[cfg(any(test, feature = "testexport"))]
    pub fn set_parent_dir(&mut self, parent: PathBuf) {
        match self {
            IaCapacity::Manual {
                small_queue,
                main_queue,
            } => {
                if let Some(ref mut path) = small_queue.path {
                    *path = parent.join(&path);
                }
                if let Some(ref mut path) = main_queue.path {
                    *path = parent.join(&path);
                }
            }
            IaCapacity::MemoryAndDiskCap(_, dir, _) => {
                *dir = parent.join(&dir);
            }
            IaCapacity::MemoryCap(..) => {}
        }
    }
}

#[derive(Default)]
pub struct IaManagerOptionsBuilder {
    capacity: Option<IaCapacity>,
    segment_size: Option<i64>,
    freq_update_interval: Option<Duration>,
    dfs_concurrency: Option<usize>,
    dfs_keyspace_concurrency: Option<usize>,
    fd_cache_capacity: Option<usize>,
    table_meta_mtime_interval: Option<Duration>,
}

impl IaManagerOptionsBuilder {
    pub fn capacity(mut self, capacity: IaCapacity) -> Self {
        self.capacity = Some(capacity);
        self
    }

    pub fn segment_size(mut self, size: i64) -> Self {
        self.segment_size = Some(size);
        self
    }

    pub fn freq_update_interval(mut self, interval: Duration) -> Self {
        self.freq_update_interval = Some(interval);
        self
    }

    pub fn dfs_concurrency(mut self, concurrency: usize, keyspace_concurrency: usize) -> Self {
        self.dfs_concurrency = Some(concurrency);
        self.dfs_keyspace_concurrency = Some(keyspace_concurrency);
        self
    }

    pub fn fd_cache_capacity(mut self, capacity: usize) -> Self {
        self.fd_cache_capacity = Some(capacity);
        self
    }

    pub fn table_meta_mtime_interval(mut self, interval: Duration) -> Self {
        self.table_meta_mtime_interval = Some(interval);
        self
    }

    pub fn build(mut self) -> Result<IaManagerOptions> {
        let mut options = IaManagerOptions::default();

        let cap = self.capacity.take().unwrap_or_default();
        cap.build_options(&mut options)?;

        options.segment_size = self.segment_size.unwrap_or(IA_SEGMENT_SIZE_DEF);
        options.freq_update_interval = self
            .freq_update_interval
            .unwrap_or(IA_FREQ_UPDATE_INTERVAL_DEF);
        options.dfs_concurrency = self.dfs_concurrency.unwrap_or(IA_DFS_CONCURRENCY_DEF);
        options.dfs_keyspace_concurrency = self
            .dfs_keyspace_concurrency
            .unwrap_or(IA_DFS_KEYSPACE_CONCURRENCY_DEF);
        options.fd_cache_capacity = self.fd_cache_capacity.unwrap_or(IA_FD_CACHE_CAPACITY_DEF);
        options.table_meta_mtime_interval = self
            .table_meta_mtime_interval
            .unwrap_or(IA_TABLE_META_MTIME_INTERVAL_DEF);

        Ok(options)
    }
}

/// The config of IA (Infrequent Access) which use memory for small queue and
/// disk for main queue.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(default)]
#[serde(rename_all = "kebab-case")]
pub struct IaConfig {
    pub mem_cap: AbsoluteOrPercentSize,
    pub disk_cap: AbsoluteOrPercentSize,
    pub segment_size: i64,
    pub freq_update_interval: ReadableDuration,
    pub dfs_concurrency: usize,
    pub dfs_keyspace_concurrency: usize,
    pub fd_cache_capacity: usize,
    pub table_meta_mtime_interval: ReadableDuration,
}

impl Default for IaConfig {
    fn default() -> Self {
        Self {
            mem_cap: Default::default(),
            disk_cap: Default::default(),
            segment_size: IA_SEGMENT_SIZE_DEF,
            freq_update_interval: ReadableDuration(IA_FREQ_UPDATE_INTERVAL_DEF),
            dfs_concurrency: IA_DFS_CONCURRENCY_DEF,
            dfs_keyspace_concurrency: IA_DFS_KEYSPACE_CONCURRENCY_DEF,
            fd_cache_capacity: IA_FD_CACHE_CAPACITY_DEF,
            table_meta_mtime_interval: ReadableDuration(IA_TABLE_META_MTIME_INTERVAL_DEF),
        }
    }
}

impl IaConfig {
    pub fn to_manager_options(&self, data_path: PathBuf) -> Result<IaManagerOptions> {
        let cap = IaCapacity::MemoryAndDiskCap(self.mem_cap, data_path, self.disk_cap);
        IaManagerOptionsBuilder::default()
            .capacity(cap)
            .segment_size(self.segment_size)
            .freq_update_interval(self.freq_update_interval.0)
            .dfs_concurrency(self.dfs_concurrency, self.dfs_keyspace_concurrency)
            .fd_cache_capacity(self.fd_cache_capacity)
            .table_meta_mtime_interval(self.table_meta_mtime_interval.0)
            .build()
    }
}

#[cfg(any(test, feature = "testexport"))]
pub mod test_util {
    #[cfg(feature = "debug-trace-ia-segments")]
    use crate::ia::debug::SegmentAction;
    use crate::ia::{
        queue::{QueueItem, QueueItemPos},
        types::{FileSegmentData, FileSegmentIdent},
    };

    #[cfg(feature = "debug-trace-ia-segments")]
    fn dump_segment_action_history(ident: &FileSegmentIdent) -> Vec<SegmentAction> {
        crate::ia::debug::dump_segment_action_history(ident)
    }

    #[cfg(not(feature = "debug-trace-ia-segments"))]
    fn dump_segment_action_history(_ident: &FileSegmentIdent) -> &'static str {
        "(disabled)"
    }

    const SEGMENT_POS_MISMATCH_THRESHOLD: usize = 3;

    pub fn verify_local_segments(
        segments: &[(FileSegmentIdent, FileSegmentData, Option<QueueItem>)],
        small_cap: i64,
        main_cap: i64,
        expected_total_size: Option<u64>,
    ) {
        let mut total_size = 0;
        let mut small_cached_size = 0;
        let mut main_cached_size = 0;
        let mut mismatch_segments = vec![];
        for (ident, segment, queue_item) in segments {
            let history = dump_segment_action_history(ident);
            debug!("verify_local_segments"; "ident" => ?ident, "segment" => ?segment,
                "queue_item" => ?queue_item, "history" => ?history);

            total_size += ident.size();
            match &segment {
                FileSegmentData::InMem(_) => {
                    small_cached_size += ident.size();
                }
                FileSegmentData::InStore => {
                    main_cached_size += ident.size();
                }
            }

            match queue_item {
                None => {
                    panic!(
                        "pos mismatch, not in queue: ident: {:?}, segment: {:?}, history: {:?}",
                        ident, segment, history
                    );
                }
                Some(queue_item) => match (&queue_item.pos, segment) {
                    (QueueItemPos::Small, FileSegmentData::InMem(_))
                    | (QueueItemPos::Main, FileSegmentData::InStore) => {}
                    (QueueItemPos::Small, FileSegmentData::InStore) => {
                        // Tolerate this case.
                        // Happen when the movement to main store is delayed to execute after
                        // another evict and insert to memory for the same segment.
                        // See https://github.com/tidbcloud/cloud-storage-engine/issues/1993#issuecomment-2567583092.
                        warn!("pos mismatch, queue: {:?}, store: {:?}", queue_item.pos, segment;
                                "ident" =>?ident, "segment" => ?segment, "queue_item" => ?queue_item, "history" => ?history);
                        mismatch_segments.push(ident.clone());
                    }
                    (QueueItemPos::Main, FileSegmentData::InMem(_)) => {
                        panic!(
                            "pos mismatch, ident: {:?}, queue_item: {:?}, segment: {:?}, history: {:?}",
                            ident, queue_item, segment, history
                        );
                    }
                },
            }
        }

        assert!(
            mismatch_segments.len() <= SEGMENT_POS_MISMATCH_THRESHOLD,
            "pos mismatch: {:?}, segments: {:?}, threshold: {}",
            mismatch_segments,
            segments,
            SEGMENT_POS_MISMATCH_THRESHOLD
        );

        if let Some(expected_total_size) = expected_total_size {
            assert!(total_size <= expected_total_size);
        }

        info!("cached size";
            "small_cached_size" => small_cached_size,
            "small_cap" => small_cap,
            "main_cached_size" => main_cached_size,
            "main_cap" => main_cap,
            "total_size" => total_size,
        );
        assert!(
            small_cached_size as i64 <= small_cap,
            "total_size: {}, small_cached_size: {}, small_cap: {}, segments: {:?}",
            total_size,
            small_cached_size,
            small_cap,
            segments
        );
        let mismatch_size = mismatch_segments
            .iter()
            .map(|ident| ident.size())
            .sum::<u64>();
        assert!(
            main_cached_size as i64 <= main_cap + mismatch_size as i64,
            "total_size: {}, main_cached_size: {}, main_cap: {}, segments: {:?}",
            total_size,
            main_cached_size,
            main_cap,
            segments
        );
    }
}

#[cfg(test)]
mod tests {
    use tempfile::TempDir;

    use super::*;

    #[test]
    fn test_ia_capacity() {
        let mut options = IaManagerOptions::default();

        let ia_cap = IaCapacity::MemoryCap(AbsoluteOrPercentSize::Percent(20.0));
        ia_cap.build_options(&mut options).unwrap();
        println!("options for MemoryCap(20%): {:?}", options);

        let temp_dir = TempDir::new().unwrap();
        let ia_cap = IaCapacity::MemoryAndDiskCap(
            AbsoluteOrPercentSize::Percent(10.0),
            temp_dir.path().to_path_buf(),
            AbsoluteOrPercentSize::Percent(10.0),
        );
        ia_cap.build_options(&mut options).unwrap();
        println!("options for MemoryAndDiskCap(10%+10%): {:?}", options);
    }
}
