// Copyright 2024 TiKV Project Authors. Licensed under Apache-2.0.

// TODO: remote this
#![allow(dead_code)]

use std::{fmt, hash::Hash, ops, sync::Arc};

use bytes::{Buf, BufMut, Bytes};
use dashmap::{mapref::entry::Entry, DashMap};
use tikv_util::codec::number::{U64_SIZE, U8_SIZE};
use tokio::sync::{Mutex, OwnedMutexGuard};

use crate::{
    dfs::FileType,
    ia::{queue::FifoItemPos, util::LocalStore},
    table::{Error, Result},
};

/// The identifier of a file segment.
#[repr(C)]
#[derive(Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct FileSegmentIdent {
    pub file_id: u64,
    pub start_off: u64,
    pub end_off: u64,
}

impl FileSegmentIdent {
    pub(crate) fn fingerprint(&self) -> u32 {
        let ptr = unsafe {
            std::slice::from_raw_parts(self as *const _ as *const u8, std::mem::size_of::<Self>())
        };
        farmhash::fingerprint32(ptr)
    }

    pub fn size(&self) -> u64 {
        self.end_off - self.start_off
    }

    pub(crate) fn local_filename(&self) -> String {
        format!("{}-{}-{}.seg", self.file_id, self.start_off, self.end_off)
    }

    pub(crate) fn parse_local_filename(filename: &str) -> Option<Self> {
        let parts: Option<Vec<u64>> = filename
            .strip_suffix(".seg")?
            .split('-')
            .map(|x| x.parse::<u64>().ok())
            .collect();
        let parts = parts?;
        (parts.len() == 3).then(|| {
            let (file_id, start_off, end_off) = (parts[0], parts[1], parts[2]);
            Self {
                file_id,
                start_off,
                end_off,
            }
        })
    }
}

impl fmt::Display for FileSegmentIdent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}-{}-{}", self.file_id, self.start_off, self.end_off)
    }
}

impl fmt::Debug for FileSegmentIdent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self)
    }
}

/// The status of a file segment to indicate whether it's cached (in local
/// disk/memory).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FileSegmentStatus {
    pub cached: bool,
}

pub(crate) type FileSegmentGuard = OwnedMutexGuard<FileSegmentStatus>;

/// The information of a file segment in the queue.
#[derive(Debug, Default)]
pub(crate) struct FileSegmentQueueInfo {
    pub(crate) freq: u8,
    pub(crate) access_time: u64,
    pub(crate) pos: FifoItemPos,
}

pub(crate) type FileSegmentQueueGuard = OwnedMutexGuard<FileSegmentQueueInfo>;

#[derive(Clone, Default)]
pub(crate) struct FileSegmentInfo {
    // Used to block concurrent operations on the same local file/memory.
    status: Arc<Mutex<FileSegmentStatus>>,

    queue_info: Arc<Mutex<FileSegmentQueueInfo>>,
}

impl fmt::Debug for FileSegmentInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut de = f.debug_struct("FileSegmentInfo");
        if let Ok(queue_info) = self.queue_info.try_lock() {
            de.field("queue_info", &queue_info);
        } else {
            de.field("queue_info", &"<locked>");
        }
        de.finish()
    }
}

impl FileSegmentInfo {
    pub(crate) async fn lock(&self) -> FileSegmentGuard {
        self.status.clone().lock_owned().await
    }

    pub(crate) async fn lock_queue_info(&self) -> FileSegmentQueueGuard {
        self.queue_info.clone().lock_owned().await
    }
}

#[derive(Default)]
pub(crate) struct GuardMap<K: Eq + Hash, V> {
    core: DashMap<K, Arc<Mutex<V>>>,
}

impl<K, V> ops::Deref for GuardMap<K, V>
where
    K: Eq + Hash,
{
    type Target = DashMap<K, Arc<Mutex<V>>>;

    fn deref(&self) -> &Self::Target {
        &self.core
    }
}

impl<K, V> GuardMap<K, V>
where
    K: Eq + Hash,
    V: Default,
{
    pub(crate) async fn get_locked(&self, key: K) -> OwnedMutexGuard<V> {
        match self.core.entry(key) {
            Entry::Occupied(entry) => {
                let value = entry.get().clone();
                drop(entry);
                value.lock_owned().await
            }
            Entry::Vacant(entry) => {
                let value = Arc::new(Mutex::new(Default::default()));
                let guard = value.clone().try_lock_owned().unwrap();
                entry.insert(value);
                guard
            }
        }
    }
}

/// Version tag used to marshall footer info. Reserved for future use.
const FOOTER_INFO_VER: u8 = 0;
/// It's a hint for the footer length which should be enough for most cases.
const FOOTER_LEN_HINT: u64 = 64;

#[derive(PartialEq, Debug, Clone)]
pub(crate) struct FooterInfo {
    pub(crate) file_id: u64,
    pub(crate) ftype: FileType,
    pub(crate) file_total_size: u64,
}

impl FooterInfo {
    pub(crate) async fn read_from_local(
        store: &dyn LocalStore,
        file_id: u64,
    ) -> Result<(Self, Bytes)> {
        let mut data = Vec::with_capacity(FOOTER_LEN_HINT as usize);
        let res = store
            .read_all(file_id, &Self::local_filename(file_id), &mut data)
            .await?;
        if res.is_none() {
            return Err(Error::IaMgr(format!("footer not found: {file_id}")));
        }
        let data = Bytes::from(data);

        // ver + ftype + total_size
        let footer = data.slice(U8_SIZE + U8_SIZE + U64_SIZE..);

        let mut header = data.as_ref();
        let ver = header.get_u8();
        if ver != FOOTER_INFO_VER {
            return Err(Error::IaMgr(format!(
                "invalid footer version, expect {}, got {}, file_id {}, data {:?}",
                FOOTER_INFO_VER, ver, file_id, data
            )));
        }
        let Some(ftype) = FileType::from_u8(header.get_u8()) else {
            return Err(Error::IaMgr(format!(
                "invalid file type, file_id {}, data {:?}",
                file_id, data
            )));
        };
        let file_total_size = header.get_u64_le();
        let footer_info = Self {
            file_id,
            ftype,
            file_total_size,
        };

        Ok((footer_info, footer))
    }

    pub(crate) async fn save_to_local(&self, store: &dyn LocalStore, footer: &[u8]) -> Result<()> {
        // ver + ftype + total_size + footer
        let mut data = Vec::with_capacity(U8_SIZE + U8_SIZE + U64_SIZE + footer.len());
        data.put_u8(FOOTER_INFO_VER);
        data.put_u8(self.ftype as u8);
        data.put_u64_le(self.file_total_size);
        data.put(footer);

        store
            .save(
                self.file_id,
                &Self::local_filename(self.file_id),
                Bytes::from(data),
            )
            .await
    }

    pub(crate) async fn drop_from_local(
        file_id: u64,
        store: &dyn LocalStore,
    ) -> Result<Option<()>> {
        store.remove(file_id, &Self::local_filename(file_id)).await
    }

    fn local_filename(file_id: u64) -> String {
        format!("{}.footer", file_id)
    }

    pub(crate) fn parse_local_filename(filename: &str) -> Option<u64> {
        filename
            .strip_suffix(".footer")
            .and_then(|s| s.parse().ok())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ia::util::LocalMemoryStore;

    #[test]
    fn test_file_segment_ident() {
        let ident = FileSegmentIdent {
            file_id: 1,
            start_off: 10,
            end_off: 20,
        };
        let fingerprint = ident.fingerprint();
        assert_eq!(ident.size(), 10);
        assert_eq!(fingerprint, 17319032);
        assert_eq!(ident.local_filename(), "1-10-20.seg");
        assert_eq!(
            FileSegmentIdent::parse_local_filename(&ident.local_filename()).as_ref(),
            Some(&ident)
        );

        assert_eq!(
            FileSegmentIdent {
                file_id: 1,
                start_off: 10,
                end_off: 20
            }
            .fingerprint(),
            fingerprint
        );
        assert_ne!(
            FileSegmentIdent {
                file_id: 1,
                start_off: 11,
                end_off: 20
            }
            .fingerprint(),
            fingerprint
        );
    }

    #[tokio::test]
    async fn test_footer_info() {
        assert_eq!(FooterInfo::local_filename(1), "1.footer");
        assert_eq!(FooterInfo::parse_local_filename("1.footer"), Some(1));

        let store = Arc::new(LocalMemoryStore::default());
        let footer_info = FooterInfo {
            file_id: 1,
            ftype: FileType::Sst,
            file_total_size: 100,
        };
        let footer = b"footer";
        footer_info
            .save_to_local(store.as_ref(), footer)
            .await
            .unwrap();

        let (footer_info1, footer1) = FooterInfo::read_from_local(store.as_ref(), 1)
            .await
            .unwrap();
        assert_eq!(footer_info, footer_info1);
        assert_eq!(footer, footer1.as_ref());
    }
}
