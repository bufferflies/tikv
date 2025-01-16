// Copyright 2024 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    ops,
    ops::Deref,
    path::PathBuf,
    sync::{
        atomic::{AtomicU64, Ordering::Relaxed},
        Arc,
    },
    time::Duration,
};

use bytes::Bytes;
use dashmap::DashMap;
use tikv_util::{deadline::Deadline, time::Instant};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use crate::{
    dfs,
    dfs::{Dfs, FileType},
    ia::{
        queue::S3FifoHandle,
        types::{
            FileSegmentData, FileSegmentIdent, GuardMap, LocalSegmentMap, SegmentHandle,
            FILE_SEGMENT_DATA_IN_MEMORY,
        },
        util::{new_local_store, LocalStore},
    },
    table::{Error, Result},
    try_some,
    util::WorkerPool,
};

/// `buf` can be empty, which means to request the specified range of data but
/// do not actually read it. Used to prefetch segments from remote.
pub(crate) struct ReadAt<'a> {
    buf: &'a mut [u8],
    offset: u64,

    /// Get handle to keep underlying data available, even the segment is
    /// evicted.
    need_segment_handle: bool,
    segment_handle: Option<SegmentHandle>,
}

impl<'a> ReadAt<'a> {
    pub(crate) fn new(buf: &'a mut [u8], offset: u64, need_segment_handle: bool) -> Self {
        Self {
            buf,
            offset,
            need_segment_handle,
            segment_handle: None,
        }
    }

    pub(crate) fn start_off(&self) -> u64 {
        self.offset
    }

    pub(crate) fn end_off(&self) -> u64 {
        self.offset + self.buf.len() as u64
    }

    fn read_from_segment_bytes(&mut self, ident: &FileSegmentIdent, seg_data: &Bytes) {
        let (start_off, end_off) = (self.start_off(), self.end_off());
        debug_assert!(
            start_off <= end_off && ident.start_off <= start_off && end_off <= ident.end_off
        );
        if self.need_segment_handle {
            let handle = SegmentHandle::from_bytes(ident.file_id, seg_data.clone());
            self.set_handle(handle);
        }
        if !self.buf.is_empty() {
            let seg_slice = seg_data.slice(
                (start_off - ident.start_off) as usize..(end_off - ident.start_off) as usize,
            );
            self.buf.copy_from_slice(&seg_slice);
        }
    }

    fn set_handle(&mut self, handle: SegmentHandle) {
        let prev = self.segment_handle.replace(handle);
        debug_assert!(prev.is_none());
    }
}

/// Note that the capacity is not strictly limited for performance. So some
/// additional buffer (maybe 10%) should be reserved.
#[derive(Default, Clone, Debug)]
pub struct QueueOptions {
    /// It means in memory when `path` is `None`.
    pub path: Option<PathBuf>,
    pub cap: i64,
}

#[derive(Default, Clone, Debug)]
pub struct IaManagerOptions {
    pub small_queue: QueueOptions,
    pub main_queue: QueueOptions,
    pub segment_size: i64,

    /// The minimum interval to update "freq" counter in queue.
    ///
    /// Used to handle the scene that a single request touch multiple slice of a
    /// segment and increase the freq unexpectedly.
    pub freq_update_interval: Duration,

    pub dfs_concurrency: usize,
    pub dfs_keyspace_concurrency: usize,

    // The capacity of the file descriptor cache.
    pub fd_cache_capacity: usize,
}

impl IaManagerOptions {
    pub fn total_capacity(&self) -> i64 {
        self.small_queue.cap + self.main_queue.cap
    }
}

#[derive(Clone)]
pub struct IaManager {
    core: Arc<IaManagerCore>,
}

impl ops::Deref for IaManager {
    type Target = IaManagerCore;

    fn deref(&self) -> &Self::Target {
        &self.core
    }
}

impl IaManager {
    pub async fn new(
        opts: IaManagerOptions,
        fs: Arc<dyn dfs::Dfs>,
        runtime: WorkerPool,
    ) -> Result<Self> {
        assert!(
            opts.small_queue.path.is_none(),
            "small queue must be in memory"
        );

        info!("create IA manager"; "opts" => ?opts);
        let main_store = new_local_store(opts.main_queue.path, opts.fd_cache_capacity);
        let segments = Arc::new(LocalSegmentMap::default());
        let segment_data_ctx = SegmentDataContext {
            segments: segments.clone(),
            main_store: main_store.clone(),
        };

        let fifo = S3FifoHandle::new(
            opts.small_queue.cap,
            opts.main_queue.cap,
            opts.segment_size,
            opts.freq_update_interval,
            runtime.handle(),
            segment_data_ctx,
        );

        let dfs_concurrency = Arc::new(Semaphore::new(opts.dfs_concurrency));

        let core = Arc::new(IaManagerCore {
            segment_size: opts.segment_size,
            fs,
            runtime,
            main_store,
            loading_segments: Default::default(),
            segments,
            fifo,
            cache_hit_counter: Default::default(),
            cache_miss_counter: Default::default(),
            dfs_concurrency,
            dfs_keyspace_concurrency_limit: opts.dfs_keyspace_concurrency,
            dfs_keyspace_concurrency: Default::default(),
        });

        let mgr = Self { core };
        mgr.init().await?;
        Ok(mgr)
    }
}

pub struct IaManagerCore {
    segment_size: i64,
    fs: Arc<dyn dfs::Dfs>,
    runtime: WorkerPool,
    main_store: Arc<dyn LocalStore>,

    loading_segments: GuardMap<FileSegmentIdent, ()>,
    segments: Arc<LocalSegmentMap>,

    fifo: S3FifoHandle,
    cache_hit_counter: AtomicU64,
    cache_miss_counter: AtomicU64,

    dfs_concurrency: Arc<Semaphore>,
    dfs_keyspace_concurrency_limit: usize,
    dfs_keyspace_concurrency: Arc<DashMap<u32 /* keyspace_id */, Arc<Semaphore>>>,
}

impl IaManagerCore {
    pub fn enter_runtime(&self) -> tokio::runtime::EnterGuard<'_> {
        self.runtime.enter()
    }

    pub fn get_dfs(&self) -> &dyn Dfs {
        self.fs.deref()
    }

    pub(crate) fn segment_size(&self) -> i64 {
        self.segment_size
    }

    async fn init(&self) -> Result<()> {
        self.main_store.init().await?;
        let mut entries = self.main_store.scan().await?;
        if let Some(segments) = entries.remove("seg") {
            self.init_segments(segments).await?;
        }

        Ok(())
    }

    // TODO: take snapshot for queue & restore from it.
    async fn init_segments(&self, keys: Vec<String>) -> Result<()> {
        for k in keys {
            if let Some(ident) = FileSegmentIdent::parse_local_filename(&k) {
                self.segments
                    .set_segment_data(ident.clone(), FileSegmentData::InStore);
                self.fifo.read(ident, true)?;
            }
        }
        Ok(())
    }

    /// Offset in `read_at` are absolute offsets of the file.
    pub(crate) async fn read_segment(
        &self,
        ident: FileSegmentIdent,
        ftype: FileType,
        keyspace_id: Option<u32>,
        deadline: Option<Deadline>,
        read_at: &mut ReadAt<'_>,
    ) -> Result<()> {
        let (start_off, end_off) = (read_at.start_off(), read_at.end_off());
        if start_off > end_off || start_off < ident.start_off || ident.end_off < end_off {
            debug_assert!(
                false,
                "invalid range, segment {:?}, range: {}-{}",
                ident, start_off, end_off
            );
            return Err(Error::IaMgr(format!(
                "invalid range, ident: {:?}, range: {}-{}",
                ident, start_off, end_off
            )));
        }

        let start_time = Instant::now_coarse();
        debug!("read segment"; "ident" => %ident, "start_off" => start_off, "end_off" => end_off);
        if let Some(()) = self.read_segment_from_cache(&ident, read_at)? {
            debug!("read segment finished (cache hit)";
                "ident" => %ident,
                "elapsed" => ?start_time.saturating_elapsed());
            return Ok(());
        }

        let data = {
            let _loading_guard = self.loading_segments.get_locked(ident.clone()).await;

            // Check cache again. Another thread may have filled the cache.
            if let Some(()) = self.read_segment_from_cache(&ident, read_at)? {
                debug!("read segment finished (cache hit)";
                    "ident" => %ident,
                    "elapsed" => ?start_time.saturating_elapsed());

                // Not necessary to remove ident from `loading_segments`, as there must be
                // another concurrent request read the segment from remote.
                return Ok(());
            }

            let data = self
                .read_segment_from_remote(&ident, ftype, keyspace_id, deadline)
                .await?;
            self.segments
                .set_segment_data(ident.clone(), FileSegmentData::InMem(data.clone()));

            self.loading_segments.remove(&ident);

            data
        };

        self.fifo.read(ident.clone(), false)?;

        read_at.read_from_segment_bytes(&ident, &data);
        self.cache_miss_counter.fetch_add(1, Relaxed);
        debug!("read segment finished (cache missed)";
            "ident" => %ident,
            "elapsed" => ?start_time.saturating_elapsed());
        Ok(())
    }

    async fn read_segment_from_remote(
        &self,
        ident: &FileSegmentIdent,
        ftype: FileType,
        keyspace_id: Option<u32>,
        deadline: Option<Deadline>,
    ) -> Result<Bytes> {
        let _permit = self.acquire_concurrency_permit(keyspace_id).await;
        if deadline.is_some_and(|d| d.check().is_err()) {
            return Err(Error::DeadlineExceeded(format!(
                "acquire concurrency permit timeout: {ident}"
            )));
        }

        let opts = dfs::Options::default()
            .with_type(ftype)
            .with_start_off(ident.start_off)
            .with_end_off(ident.end_off);
        let _enter = self.fs.get_runtime().enter();
        Ok(self.fs.read_file(ident.file_id, opts).await?)
    }

    fn read_segment_from_cache(
        &self,
        ident: &FileSegmentIdent,
        read_at: &mut ReadAt<'_>,
    ) -> Result<Option<()>> {
        let segment_data = try_some!(self.segments.get_segment(ident));
        let res = match &segment_data {
            FileSegmentData::InMem(data) => {
                read_at.read_from_segment_bytes(ident, data);
                Some(())
            }
            FileSegmentData::InStore => self.read_segment_from_local_store(ident, read_at)?,
        };

        if res.is_some() {
            self.fifo.read(ident.clone(), false)?;
            self.cache_hit_counter.fetch_add(1, Relaxed);
        }
        Ok(res)
    }

    // Note: when `read_at.buf` is empty, the existence of segment is not checked.
    fn read_segment_from_local_store(
        &self,
        ident: &FileSegmentIdent,
        read_at: &mut ReadAt<'_>,
    ) -> Result<Option<()>> {
        debug!("read segment from local"; "ident" => %ident);
        let local_filename = ident.local_filename();
        if read_at.need_segment_handle {
            match self.main_store.handle(ident.file_id, &local_filename)? {
                Some(handle) => read_at.set_handle(handle),
                None => return Ok(None),
            }
        }
        if read_at.buf.is_empty() {
            return Ok(Some(()));
        }
        self.main_store.read_at(
            ident.file_id,
            &local_filename,
            read_at.buf,
            read_at.offset - ident.start_off,
        )
    }

    // Note: It's possible that the segment is of `SegmentData::InStore` but
    // actually not existed. In such condition this method will still return
    // true.
    pub fn is_segment_cached(&self, ident: &FileSegmentIdent) -> bool {
        self.segments.get_segment(ident).is_some()
    }

    pub async fn prefetch_segment(
        &self,
        ident: FileSegmentIdent,
        ftype: FileType,
        keyspace_id: u32,
        deadline: Deadline,
    ) -> Result<()> {
        let mut buf: [u8; 0] = [];
        let mut read_at = ReadAt::new(buf.as_mut_slice(), ident.start_off, false);
        self.read_segment(
            ident,
            ftype,
            Some(keyspace_id),
            Some(deadline),
            &mut read_at,
        )
        .await
    }

    pub async fn get_segment_handle(
        &self,
        ident: FileSegmentIdent,
        ftype: FileType,
    ) -> Result<SegmentHandle> {
        let mut buf: [u8; 0] = [];
        let mut read_at = ReadAt::new(buf.as_mut_slice(), ident.start_off, true);
        self.read_segment(ident.clone(), ftype, None, None, &mut read_at)
            .await?;
        let handle = read_at.segment_handle.unwrap();
        info!("get segment handle"; "ident" => ?ident, "ftype" => ?ftype, "handle" => ?handle);
        Ok(handle)
    }

    pub async fn acquire_concurrency_permit(
        &self,
        keyspace_id: Option<u32>,
    ) -> (Option<OwnedSemaphorePermit>, OwnedSemaphorePermit) {
        let keyspace_permit = match keyspace_id {
            Some(keyspace_id) => {
                let sem = self
                    .dfs_keyspace_concurrency
                    .entry(keyspace_id)
                    .or_insert_with(|| {
                        Arc::new(Semaphore::new(self.dfs_keyspace_concurrency_limit))
                    })
                    .clone();
                Some(sem.acquire_owned().await.unwrap())
            }
            None => None,
        };
        let global_permit = self.dfs_concurrency.clone().acquire_owned().await.unwrap();
        (keyspace_permit, global_permit)
    }

    pub fn cache_hit_rate(&self) -> f64 {
        let hit = self.cache_hit_counter.load(Relaxed);
        let miss = self.cache_miss_counter.load(Relaxed);
        if hit + miss > 0 {
            hit as f64 / (hit + miss) as f64
        } else {
            0.0
        }
    }
}

#[cfg(any(test, feature = "testexport"))]
impl IaManager {
    // Flush all tasks and wait for completion.
    pub async fn flush_tasks(&self, timeout: Duration) -> Result<()> {
        let start_time = Instant::now_coarse();
        while start_time.saturating_elapsed() < timeout {
            let pending_tasks = self.fifo.flush_tasks().await?;
            if pending_tasks == 0 {
                return Ok(());
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        Err(Error::IaMgr("wait for pending tasks timeout".to_string()))
    }

    pub async fn get_local_segments(
        &self,
    ) -> Vec<(
        FileSegmentIdent,
        crate::ia::types::FileSegmentData,
        Option<crate::ia::queue::QueueItem>,
    )> {
        let local_segments = self
            .segments
            .iter()
            .map(|r| (r.key().clone(), r.value().clone()))
            .collect::<Vec<_>>();
        let mut segments = Vec::with_capacity(local_segments.len());
        // TODO: find segments in queue but not in local store.
        for (ident, segment) in local_segments {
            let queue_item = self.fifo.get_item(ident.clone()).await.unwrap();
            segments.push((ident, segment, queue_item));
        }
        segments
    }
}

#[derive(Clone)]
pub(crate) struct SegmentDataContext {
    segments: Arc<LocalSegmentMap>,
    main_store: Arc<dyn LocalStore>,
}

impl SegmentDataContext {
    #[inline]
    pub(crate) fn set_segment_data_from_mem_to_store(
        &self,
        ident: FileSegmentIdent,
    ) -> std::result::Result<(), Option<FileSegmentData>> {
        self.compare_and_set_segment_data(
            ident,
            &FILE_SEGMENT_DATA_IN_MEMORY,
            Some(FileSegmentData::InStore),
        )
    }

    fn is_pos_match(m: &FileSegmentData, n: &FileSegmentData) -> bool {
        matches!(
            (m, n),
            (FileSegmentData::InMem(_), FileSegmentData::InMem(_))
                | (FileSegmentData::InStore, FileSegmentData::InStore)
        )
    }

    fn compare_and_set_segment_data(
        &self,
        ident: FileSegmentIdent,
        expected: &FileSegmentData,
        segment_data: Option<FileSegmentData>,
    ) -> std::result::Result<(), Option<FileSegmentData>> {
        self.segments
            .compare_and_set(ident, expected, segment_data, Self::is_pos_match)
    }

    #[inline]
    pub(crate) fn get_segment_data(&self, ident: &FileSegmentIdent) -> Option<FileSegmentData> {
        self.segments.get_segment(ident)
    }

    #[inline]
    pub(crate) fn remove_segment_data(&self, ident: &FileSegmentIdent) -> Option<FileSegmentData> {
        self.segments.remove(ident)
    }

    #[inline]
    pub(crate) fn remove_from_main_store(&self, ident: &FileSegmentIdent) -> Result<Option<()>> {
        self.main_store
            .remove(ident.file_id, &ident.local_filename())
    }

    #[inline]
    pub(crate) async fn save_to_main_store(
        &self,
        ident: &FileSegmentIdent,
        bytes: Bytes,
    ) -> Result<()> {
        self.main_store
            .save(ident.file_id, &ident.local_filename(), bytes)
            .await
    }

    #[cfg(test)]
    pub(crate) fn new_for_test() -> Self {
        Self {
            main_store: Arc::new(crate::ia::util::LocalMemoryStore::default()),
            segments: Arc::new(LocalSegmentMap::default()),
        }
    }
}
