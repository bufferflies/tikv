// Copyright 2024 TiKV Project Authors. Licensed under Apache-2.0.

// TODO: remote this
#![allow(dead_code)]

use std::{fmt, hash::Hash, sync::Arc};

use tokio::sync::{Mutex, OwnedMutexGuard};

use crate::ia::queue::FifoItemPos;

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
