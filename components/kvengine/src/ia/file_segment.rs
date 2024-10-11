// Copyright 2024 TiKV Project Authors. Licensed under Apache-2.0.

// TODO: remote this
#![allow(dead_code)]

use std::{fmt, ops, path::PathBuf, sync::Arc};

use bytes::Bytes;
use engine_traits::GetObjectOptions;

use crate::{
    dfs::{FileType, S3Fs},
    ia::{
        queue::FifoItem,
        types::{FooterInfo, GuardMap},
        util::{LocalFileStore, LocalMemoryStore, LocalStore},
    },
    table::{Error, Result},
};

pub(crate) struct EvictTask {
    pub(crate) items: Vec<FifoItem>,
    pub(crate) cb: Option<Box<dyn FnOnce(usize /* evicted_cnt */) + Send>>,
}

impl fmt::Debug for EvictTask {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EvictTask")
            .field("items", &self.items)
            .finish()
    }
}

#[derive(Clone)]
pub struct FileSegmentManager {
    core: Arc<FileSegmentManagerCore>,
}

impl ops::Deref for FileSegmentManager {
    type Target = FileSegmentManagerCore;

    fn deref(&self) -> &Self::Target {
        &self.core
    }
}

impl FileSegmentManager {
    // TODO: specify local store type for small & main queue separately.
    pub async fn new(local_dir: Option<PathBuf>, s3fs: S3Fs) -> Result<Self> {
        let store = if let Some(local_dir) = local_dir {
            Arc::new(LocalFileStore::new(local_dir)) as Arc<dyn LocalStore>
        } else {
            Arc::new(LocalMemoryStore::default()) as Arc<dyn LocalStore>
        };

        let core = Arc::new(FileSegmentManagerCore {
            s3fs,
            store,
            footers: Default::default(),
        });

        let mgr = Self { core };
        mgr.init().await?;

        Ok(mgr)
    }
}

pub struct FileSegmentManagerCore {
    s3fs: S3Fs,
    store: Arc<dyn LocalStore>,
    footers: GuardMap<u64 /* file_id */, Option<FooterInfo>>,
}

impl FileSegmentManagerCore {
    async fn init(&self) -> Result<()> {
        let mut entries = self.store.init().await?;
        if let Some(footers) = entries.remove("footer") {
            self.init_footers(footers).await?;
        }
        Ok(())
    }

    async fn init_footers(&self, keys: Vec<String>) -> Result<()> {
        for k in keys {
            if let Some(file_id) = FooterInfo::parse_local_filename(&k) {
                let (footer_info, _) =
                    FooterInfo::read_from_local(self.store.clone(), file_id).await?;
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
        if let Some(footer_info) = guard.as_ref() {
            match FooterInfo::read_from_local(self.store.clone(), footer_info.file_id).await {
                Ok((local_info, footer)) => {
                    debug_assert_eq!(local_info, *footer_info);
                    Ok((footer, footer_info.file_total_size))
                }
                Err(err) => {
                    error!("read footer from local failed"; "file_id" => file_id, "err" => ?err);
                    *guard = None;
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
            if let Err(err) = footer_info.save_to_local(self.store.clone(), &footer).await {
                error!("save footer to local failed"; "file_id" => file_id, "err" => ?err);
                debug_assert!(false);
            } else {
                *guard = Some(footer_info);
            }
            Ok((footer, total_size))
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
