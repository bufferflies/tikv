// Copyright 2024 TiKV Project Authors. Licensed under Apache-2.0.

use std::{fmt, hash::Hash, ops, sync::Arc};

use bytes::Bytes;
use dashmap::{mapref::entry::Entry, DashMap};
use tokio::sync::{Mutex, OwnedMutexGuard};

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

    pub fn local_filename(&self) -> String {
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

#[derive(Clone)]
pub enum FileSegmentData {
    InMem(Bytes),
    InStore,
}

impl fmt::Debug for FileSegmentData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InMem(_) => write!(f, "InMem"),
            Self::InStore => write!(f, "InStore"),
        }
    }
}

lazy_static::lazy_static! {
    pub static ref FILE_SEGMENT_DATA_IN_MEMORY: FileSegmentData = FileSegmentData::InMem(Bytes::new());
}

#[derive(Default)]
pub(crate) struct LocalSegmentMap {
    core: DashMap<FileSegmentIdent, FileSegmentData>,
}

impl ops::Deref for LocalSegmentMap {
    type Target = DashMap<FileSegmentIdent, FileSegmentData>;

    fn deref(&self) -> &Self::Target {
        &self.core
    }
}

impl LocalSegmentMap {
    #[inline]
    pub(crate) fn get_segment(&self, ident: &FileSegmentIdent) -> Option<FileSegmentData> {
        self.core.get(ident).map(|x| x.value().clone())
    }

    #[inline]
    pub(crate) fn set_segment_data(
        &self,
        ident: FileSegmentIdent,
        segment_data: FileSegmentData,
    ) -> Option<FileSegmentData> {
        self.core.insert(ident, segment_data)
    }
}

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

impl<K, V> Default for GuardMap<K, V>
where
    K: Eq + Hash,
{
    fn default() -> Self {
        Self {
            core: DashMap::default(),
        }
    }
}

impl<K, V> GuardMap<K, V>
where
    K: Eq + Hash + Clone + fmt::Debug,
    V: Default,
{
    pub(crate) async fn get_locked(&self, key: K) -> OwnedMutexGuard<V> {
        match self.core.entry(key) {
            Entry::Occupied(entry) => {
                let mutex = entry.get().clone();
                drop(entry);
                mutex.lock_owned().await
            }
            Entry::Vacant(entry) => {
                let mutex = Arc::new(Mutex::new(Default::default()));
                let guard = mutex.clone().try_lock_owned().unwrap();
                entry.insert(mutex);
                guard
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
