// Copyright 2024 TiKV Project Authors. Licensed under Apache-2.0.

// TODO: remote this
#![allow(dead_code)]

use std::{cmp, ops, path::PathBuf, sync::Arc, time::Duration};

use bytes::Bytes;
use dashmap::mapref::entry::Entry;
use engine_traits::GetObjectOptions;
use tokio::sync::mpsc;

use crate::{
    dfs::{FileType, S3Fs},
    ia::{
        queue::S3FifoHandle,
        types::{
            FileSegmentData, FileSegmentIdent, FooterInfo, LocalSegmentMap,
            FILE_SEGMENT_DATA_IN_MEMORY,
        },
        util::{new_local_store, LocalStore},
    },
    table::{Error, Result},
};

pub const SEGMENTS_SUB_DIR: &str = "segment";
pub const FOOTERS_SUB_DIR: &str = "footer";

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
            segments,
            fifo,
        });

        let mgr = Self { core };
        mgr.init().await?;
        Ok(mgr)
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

    segments: Arc<LocalSegmentMap>,

    fifo: S3FifoHandle,
}

impl IaManagerCore {
    async fn init(&self) -> Result<()> {
        self.footer_store.init().await?;

        self.main_store.init().await?;
        // TODO: self.main_store.scan() && self.init_segments

        Ok(())
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
