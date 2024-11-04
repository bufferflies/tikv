// Copyright 2024 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    cmp, ops,
    path::PathBuf,
    sync::{
        atomic::{AtomicU64, Ordering::Relaxed},
        Arc,
    },
    time::Duration,
};

use bytes::Bytes;
use dashmap::mapref::entry::Entry;
use engine_traits::GetObjectOptions;
use futures::future::try_join_all;
use tikv_util::time::Instant;
use tokio::sync::mpsc;

use crate::{
    dfs::{Dfs, FileType, S3Fs},
    ia::{
        ia_file::IaFile,
        queue::S3FifoHandle,
        types::{
            FileSegmentData, FileSegmentIdent, FooterInfo, GuardMap, LocalSegmentMap,
            FILE_SEGMENT_DATA_IN_MEMORY,
        },
        util::{new_local_store, split_to_segments, LocalStore},
    },
    table::{Error, Result},
    try_some,
};

pub const SEGMENTS_SUB_DIR: &str = "segment";
pub const FOOTERS_SUB_DIR: &str = "footer";

pub(crate) struct ReadAt<'a> {
    buf: &'a mut [u8],
    offset: u64,
}

impl<'a> ReadAt<'a> {
    pub(crate) fn new(buf: &'a mut [u8], offset: u64) -> Self {
        Self { buf, offset }
    }

    fn start_off(&self) -> u64 {
        self.offset
    }

    fn end_off(&self) -> u64 {
        self.offset + self.buf.len() as u64
    }

    fn read_from_segment_bytes(&mut self, ident: &FileSegmentIdent, seg_data: &Bytes) {
        let (start_off, end_off) = (self.start_off(), self.end_off());
        debug_assert!(
            start_off < end_off && ident.start_off <= start_off && end_off <= ident.end_off
        );
        let seg_slice = seg_data
            .slice((start_off - ident.start_off) as usize..(end_off - ident.start_off) as usize);
        self.buf.copy_from_slice(&seg_slice);
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
        s3fs: S3Fs,
        runtime: tokio::runtime::Handle,
    ) -> Result<Self> {
        assert!(
            opts.small_queue.path.is_none(),
            "small queue must be in memory"
        );

        let main_store = new_local_store(
            opts.main_queue
                .path
                .as_ref()
                .map(|x| x.join(SEGMENTS_SUB_DIR)),
        );
        let segments = Arc::new(LocalSegmentMap::default());
        let segment_data_ctx = SegmentDataContext {
            segments: segments.clone(),
            main_store: main_store.clone(),
        };

        let footer_store = new_local_store(
            opts.main_queue
                .path
                .as_ref()
                .map(|x| x.join(FOOTERS_SUB_DIR)),
        );

        let fifo = S3FifoHandle::new(
            opts.small_queue.cap,
            opts.main_queue.cap,
            opts.segment_size,
            opts.freq_update_interval,
            runtime.clone(),
            segment_data_ctx,
        );

        let core = Arc::new(IaManagerCore {
            segment_size: opts.segment_size,
            s3fs,
            runtime,
            main_store,
            footer_store,
            loading_segments: Default::default(),
            segments,
            fifo,
            cache_hit_counter: Default::default(),
            cache_miss_counter: Default::default(),
        });

        let mgr = Self { core };
        mgr.init().await?;
        Ok(mgr)
    }

    pub async fn open_file(&self, file_id: u64, ftype: FileType) -> Result<IaFile> {
        let (footer, total_size) = self.read_footer(file_id, ftype).await?;
        Ok(IaFile {
            id: file_id,
            size: total_size,
            ftype,
            footer,
            mgr: self.clone(),
        })
    }

    pub(crate) async fn read_range(
        &self,
        file_id: u64,
        ftype: FileType,
        total_size: u64,
        read_at: ReadAt<'_>,
    ) -> Result<()> {
        let (start_off, end_off) = (read_at.start_off(), read_at.end_off());
        let mut segments = split_to_segments(
            file_id,
            start_off,
            end_off,
            total_size,
            self.segment_size as u64,
        );
        debug!("read_range"; "start_off" => start_off, "end_off" => end_off, "segments" => ?segments);
        if segments.len() == 1 {
            self.read_segment(segments.pop().unwrap(), ftype, read_at)
                .await
        } else {
            self.read_from_multi_segments(ftype, start_off, end_off, segments, read_at)
                .await?;
            Ok(())
        }
    }

    async fn read_from_multi_segments(
        &self,
        ftype: FileType,
        start_off: u64,
        end_off: u64,
        segments: Vec<FileSegmentIdent>,
        read_at: ReadAt<'_>,
    ) -> Result<()> {
        let segments_len = segments.len();
        let mut handles = Vec::with_capacity(segments_len);

        debug!("read_from_multi_segments"; "start_off" => start_off, "end_off" => end_off, "segments" => ?segments);
        for (idx, ident) in segments.into_iter().enumerate() {
            let seg_start_off = if idx == 0 { start_off } else { ident.start_off };
            let seg_end_off = if idx == segments_len - 1 {
                end_off
            } else {
                ident.end_off
            };

            // We do not limit concurrency here as the segment size should be larger
            // than block size of file, so there are at most 2 segments.
            let mgr = self.clone();
            handles.push(self.runtime.spawn(async move {
                let mut buf = vec![0; (seg_end_off - seg_start_off) as usize];
                let read_at = ReadAt::new(buf.as_mut_slice(), seg_start_off);
                mgr.read_segment(ident, ftype, read_at).await.map(|()| buf)
            }));
        }

        let mut buf = read_at.buf;
        for seg_slice in try_join_all(handles)
            .await
            .map_err(|e| Error::IaMgr(format!("read from multi segments: failed: {e:?}")))?
        {
            let seg_slice = seg_slice?;
            let (left, right) = buf.split_at_mut(seg_slice.len());
            left.copy_from_slice(&seg_slice);
            buf = right;
        }
        debug_assert!(buf.is_empty());
        Ok(())
    }

    pub async fn prepare_footers(
        &self,
        files: &[(u64 /* file_id */, FileType)],
        concurrency: usize,
    ) -> Result<()> {
        let (tx, mut rx) = mpsc::channel(cmp::min(concurrency, files.len()));
        let mut errs: Vec<Error> = vec![];
        let mut msg_count: usize = 0;
        for &(file_id, ftype) in files {
            if self
                .footer_store
                .exists(file_id, &FooterInfo::local_filename(file_id))
                .await
            {
                continue;
            }

            let mgr = self.clone();
            let task = async move {
                let (footer, total_size) = mgr.read_footer_from_remote(file_id, ftype).await?;
                let footer_info = FooterInfo {
                    file_id,
                    ftype,
                    file_total_size: total_size,
                };
                footer_info
                    .save_to_local(mgr.footer_store.as_ref(), &footer)
                    .await?;
                Ok(())
            };
            let tx = tx.clone();
            self.runtime.spawn(async move {
                let res = task.await;
                if let Err(err) = tx.send(res).await {
                    warn!("prepare footers: send error"; "file_id" => file_id, "err" => ?err);
                }
            });
            msg_count += 1;

            if msg_count >= concurrency {
                let res = rx.recv().await.unwrap();
                if let Err(err) = res {
                    errs.push(err);
                }
                msg_count -= 1;
            }
        }

        for _ in 0..msg_count {
            let res = rx.recv().await.unwrap();
            if let Err(err) = res {
                errs.push(err);
            }
        }

        if errs.is_empty() {
            Ok(())
        } else {
            Err(Error::IaMgr(format!("prepare footers failed: {:?}", errs)))
        }
    }
}

pub struct IaManagerCore {
    segment_size: i64,
    s3fs: S3Fs,
    runtime: tokio::runtime::Handle,
    main_store: Arc<dyn LocalStore>,
    footer_store: Arc<dyn LocalStore>,

    loading_segments: GuardMap<FileSegmentIdent, ()>,
    segments: Arc<LocalSegmentMap>,

    fifo: S3FifoHandle,
    cache_hit_counter: AtomicU64,
    cache_miss_counter: AtomicU64,
}

impl IaManagerCore {
    async fn init(&self) -> Result<()> {
        self.footer_store.init().await?;

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
    async fn read_segment(
        &self,
        ident: FileSegmentIdent,
        ftype: FileType,
        mut read_at: ReadAt<'_>,
    ) -> Result<()> {
        let (start_off, end_off) = (read_at.start_off(), read_at.end_off());
        if start_off >= end_off || start_off < ident.start_off || ident.end_off < end_off {
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
        if let Some(()) = self.read_segment_from_cache(&ident, &mut read_at).await? {
            debug!("read segment finished (cache hit)";
                "ident" => %ident,
                "elapsed" => ?start_time.saturating_elapsed());
            return Ok(());
        }

        let data = {
            let _loading_guard = self.loading_segments.get_locked(ident.clone()).await;

            // Check cache again. Another thread may have filled the cache.
            if let Some(()) = self.read_segment_from_cache(&ident, &mut read_at).await? {
                debug!("read segment finished (cache hit)";
                    "ident" => %ident,
                    "elapsed" => ?start_time.saturating_elapsed());

                // Not necessary to remove ident from `loading_segments`, as there must be
                // another concurrent request read the segment from remote.
                return Ok(());
            }

            let data = self.read_segment_from_remote(&ident, ftype).await?;
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
    ) -> Result<Bytes> {
        // TODO: rate limit

        let opts = GetObjectOptions {
            start_off: Some(ident.start_off),
            end_off: Some(ident.end_off),
        };
        let file_key = self.s3fs.file_key(ident.file_id, ftype);
        let filename = format!("{ident}.seg");
        let _enter = self.s3fs.get_runtime().enter();
        Ok(self.s3fs.get_object(file_key, filename, opts).await?)
    }

    async fn read_segment_from_cache(
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
            FileSegmentData::InStore => self.read_segment_from_local_store(ident, read_at).await?,
        };

        if res.is_some() {
            self.fifo.read(ident.clone(), false)?;
            self.cache_hit_counter.fetch_add(1, Relaxed);
        }
        Ok(res)
    }

    async fn read_segment_from_local_store(
        &self,
        ident: &FileSegmentIdent,
        read_at: &mut ReadAt<'_>,
    ) -> Result<Option<()>> {
        debug!("read segment from local"; "ident" => %ident);
        self.main_store
            .read_at(
                ident.file_id,
                &ident.local_filename(),
                read_at.buf,
                read_at.offset - ident.start_off,
            )
            .await
    }

    async fn read_footer(
        &self,
        file_id: u64,
        ftype: FileType,
    ) -> Result<(Bytes, u64 /* total_size */)> {
        match FooterInfo::read_from_local(self.footer_store.as_ref(), file_id).await {
            Ok((footer_info, footer)) => Ok((footer, footer_info.file_total_size)),
            Err(err) => {
                error!("read footer: failed"; "file_id" => file_id, "ftype" => ?ftype, "err" => ?err);
                let _ = self.drop_footer_from_local(file_id).await;
                Err(err)
            }
        }
    }

    /// Use when meet error on opening/reading local footer.
    async fn drop_footer_from_local(&self, file_id: u64) -> Result<()> {
        match FooterInfo::drop_from_local(file_id, self.footer_store.as_ref()).await {
            Ok(Some(())) => Ok(()),
            Ok(None) => {
                warn!("drop footer: failed, not existed";
                    "file_id" => file_id);
                Ok(())
            }
            Err(err) => {
                warn!("drop footer: failed";
                    "file_id" => file_id,
                    "err" => ?err);
                Err(err)
            }
        }
    }

    async fn read_footer_from_remote(
        &self,
        file_id: u64,
        ftype: FileType,
    ) -> Result<(Bytes, u64 /* total_size */)> {
        let opts = GetObjectOptions {
            start_off: None,
            end_off: Some(ftype.footer_size() as u64),
        };
        let file_key = self.s3fs.file_key(file_id, ftype);
        let filename = format!("{}.{}.footer", file_id, ftype.suffix());
        let (footer, total_size) = self
            .s3fs
            .get_object_ext(file_key, filename, opts, true)
            .await?;

        let total_size = total_size.unwrap();
        if footer.len() != ftype.footer_size() {
            return Err(Error::IaMgr(format!(
                "invalid footer size, expect {}, got {}, file_id {}, footer {:?}, total_size {}",
                ftype.footer_size(),
                footer.len(),
                file_id,
                footer,
                total_size,
            )));
        }

        Ok((footer, total_size))
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
        match self.segments.entry(ident) {
            Entry::Occupied(mut entry) => {
                let prev = entry.get();
                if Self::is_pos_match(prev, expected) {
                    if let Some(segment_data) = segment_data {
                        entry.insert(segment_data);
                    } else {
                        entry.remove();
                    }
                    Ok(())
                } else {
                    Err(Some(prev.clone()))
                }
            }
            Entry::Vacant(_) => Err(None),
        }
    }

    #[inline]
    pub(crate) fn get_segment_data(&self, ident: &FileSegmentIdent) -> Option<FileSegmentData> {
        self.segments.get_segment(ident)
    }

    #[inline]
    pub(crate) fn remove_segment_data(&self, ident: &FileSegmentIdent) -> Option<FileSegmentData> {
        self.segments.remove(ident).map(|x| x.1)
    }

    #[inline]
    pub(crate) async fn remove_from_main_store(
        &self,
        ident: &FileSegmentIdent,
    ) -> Result<Option<()>> {
        self.main_store
            .remove(ident.file_id, &ident.local_filename())
            .await
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
}
