// Copyright 2024 TiKV Project Authors. Licensed under Apache-2.0.

// TODO: remote this
#![allow(dead_code)]

use std::{ops, path::PathBuf, sync::Arc, time::Duration};

use bytes::Bytes;
use dashmap::mapref::entry::Entry;
use engine_traits::GetObjectOptions;

use crate::{
    dfs::{FileType, S3Fs},
    ia::{
        queue::S3FifoHandle,
        types::{
            FileSegmentData, FileSegmentIdent, FooterInfo, GuardMap, LocalSegmentMap,
            FILE_SEGMENT_DATA_IN_MEMORY,
        },
        util::{LocalFileStore, LocalMemoryStore, LocalStore},
    },
    table::{Error, Result},
};

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

        let main_store: Arc<dyn LocalStore> = if let Some(local_dir) = opts.main_queue.path.as_ref()
        {
            Arc::new(LocalFileStore::new(local_dir.to_path_buf())) as _
        } else {
            Arc::new(LocalMemoryStore::default()) as _
        };
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
            runtime.clone(),
            segment_data_ctx,
        );

        let core = Arc::new(IaManagerCore {
            segment_size: opts.segment_size,
            s3fs,
            runtime,
            main_store,
            segments,
            footers: Default::default(),
            fifo,
        });

        let mgr = Self { core };
        mgr.init().await?;
        Ok(mgr)
    }
}

pub struct IaManagerCore {
    segment_size: i64,
    s3fs: S3Fs,
    runtime: tokio::runtime::Handle,
    main_store: Arc<dyn LocalStore>,

    segments: Arc<LocalSegmentMap>,
    footers: GuardMap<u64 /* file_id */, Option<FooterInfo>>,

    fifo: S3FifoHandle,
}

impl IaManagerCore {
    async fn init(&self) -> Result<()> {
        let mut entries = self.main_store.init().await?;
        if let Some(footers) = entries.remove("footer") {
            self.init_footers(footers).await?;
        }
        Ok(())
    }

    async fn init_footers(&self, keys: Vec<String>) -> Result<()> {
        for k in keys {
            if let Some(file_id) = FooterInfo::parse_local_filename(&k) {
                let (footer_info, _) =
                    FooterInfo::read_from_local(self.main_store.as_ref(), file_id).await?;
                self.footers.get_locked(file_id).await.replace(footer_info);
            }
        }
        Ok(())
    }

    async fn read_footer(
        &self,
        file_id: u64,
        ftype: FileType,
    ) -> Result<(Bytes, u64 /* total_size */)> {
        let mut guard = self.footers.get_locked(file_id).await;
        if let Some(footer_info) = guard.clone() {
            match FooterInfo::read_from_local(self.main_store.as_ref(), footer_info.file_id).await {
                Ok((local_info, footer)) => {
                    debug_assert_eq!(local_info, footer_info);
                    Ok((footer, footer_info.file_total_size))
                }
                Err(err) => {
                    error!("read footer from local failed"; "file_id" => file_id, "err" => ?err);
                    let _ = self.drop_footer_from_local(file_id).await;
                    Err(err)
                }
            }
        } else {
            let (footer, total_size) = self.read_footer_from_remote(file_id, ftype).await?;
            if footer.len() != ftype.footer_size() {
                return Err(Error::Other(format!(
                    "invalid footer size, expect {}, got {}, file_id {}, footer {:?}",
                    ftype.footer_size(),
                    footer.len(),
                    file_id,
                    footer,
                )));
            }
            let footer_info = FooterInfo {
                file_id,
                ftype,
                file_total_size: total_size,
            };
            if let Err(err) = footer_info
                .save_to_local(self.main_store.as_ref(), &footer)
                .await
            {
                error!("save footer to local failed"; "file_id" => file_id, "err" => ?err);
                debug_assert!(false);
            } else {
                *guard = Some(footer_info);
            }
            Ok((footer, total_size))
        }
    }

    /// Use when meet error on opening/reading local footer.
    async fn drop_footer_from_local(&self, file_id: u64) -> Result<()> {
        let mut guard = self.footers.get_locked(file_id).await;
        if guard.is_some() {
            let res = FooterInfo::drop_from_local(file_id, self.main_store.as_ref()).await;
            *guard = None;

            match res {
                Ok(Some(())) => Ok(()),
                Ok(None) => {
                    warn!("drop footer from local failed: not existed";
                        "file_id" => file_id);
                    Ok(())
                }
                Err(err) => {
                    warn!("drop footer from local failed";
                        "file_id" => file_id,
                        "err" => ?err);
                    Err(err)
                }
            }
        } else {
            Ok(())
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
        Ok((footer, total_size.unwrap()))
    }

    #[cfg(any(test, feature = "testexport"))]
    pub fn get_footers_file_id(&self) -> Vec<u64> {
        self.footers.iter().map(|r| *r.key()).collect()
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
