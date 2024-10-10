// Copyright 2024 TiKV Project Authors. Licensed under Apache-2.0.

//! An implementation of S3-FIFO algorithm.
//!
//! See https://s3fifo.com/ for the paper and relevant materials.
//!
//! Some detail:
//!
//! - Access to different files are lock-free by using the lock-free queue ring
//!   buffer `crossbeam_queue::SegQueue` as underlying data structure.
//!
//! - Access to the same file are serialized by two locks:
//!
//!   - Cache status (whether it's in local disk/memory) is maintained by
//!     `FileSegmentInfo.status` to avoid duplicated download from remote and
//!     save to local. This lock is heavy as the duration of download will be
//!     long.
//!
//!   - Queue status (whether it's in the queue and in which queue) is
//!     maintained by `FileSegmentInfo.queue_info` to keep consistency of the
//!     actual position of the item.
//!
//! - The reason for using two locks is that in process of enqueue, we need to
//!   evict some items because the queue is full. The evict process need to
//!   change queue info of other items, if using only one lock, it is likely to
//!   deadlock as the lock for cache status is heavy as mentioned above.
//!
//! - In enqueue process, we must release the locks of the enqueue item before
//!   change the queue info of the evicted items. Otherwise, the deadlock would
//!   happen.
//!
//! - The eviction process is async for performance reason, but the channel is
//!   bounded to limit the excessive capacity usage.
//!
//! - About ghost queue:
//!
//!   - The ghost queue keep the access frequency after a item has been evicted
//!     from small queue, and determine whether the item is *WARM* and should
//!     enqueue to main on next access.
//!
//!   - Currently a simplified implementation is used (using an array with
//!     length the same as the main queue according to the paper, but timestamp
//!     is not considered), to be cheap for reading & writing. But it may not be
//!     efficiency enough. Subsequently, we need to evaluate and optimize by
//!     simulating the load of real environment.

// TODO: remove this
#![allow(dead_code)]

use std::{
    assert_matches::debug_assert_matches,
    cmp, fmt,
    sync::atomic::{AtomicI64, Ordering::Relaxed},
};

use crossbeam_queue::SegQueue;
use tokio::sync::{mpsc, Mutex};

use crate::{
    ia::{
        file_segment::EvictTask,
        types::{FileSegmentIdent, FileSegmentInfo, FileSegmentQueueGuard},
    },
    table::{Error, Result},
};

/// Lifecycle of a segment:
/// Null -> Small -> Main -> ToEvict -> Null
#[repr(u8)]
#[derive(Debug, PartialEq, Default)]
pub(crate) enum FifoItemPos {
    #[default]
    Null, // The item is newly created of has been dropped.
    Small,
    Main,
    ToEvict,
}

#[derive(Clone)]
pub(crate) struct FifoItem {
    pub(crate) ident: FileSegmentIdent,
    pub(crate) segment: FileSegmentInfo,
}

impl fmt::Debug for FifoItem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "FifoItem {{ ident: {}, segment: {:?} }}",
            self.ident, self.segment
        )
    }
}

impl FifoItem {
    fn size(&self) -> u64 {
        self.ident.size()
    }
}

type OnEnqueueCallback = Box<dyn FnOnce() + Send>;

pub(crate) struct S3Fifo {
    main_queue: Queue,
    small_queue: Queue,
    ghost_queue: Vec<Mutex<Option<FileSegmentIdent>>>,
    pub(crate) evict_tx: mpsc::Sender<EvictTask>,
}

impl S3Fifo {
    /// `queue_len` is the length of Fifo queue. Currently it is used to
    /// calculate length of ghost queue. Normally it can be `capacity /
    /// average item size`.
    pub(crate) fn new(capacity: i64, queue_len: usize, evict_tx: mpsc::Sender<EvictTask>) -> Self {
        let small_cap = capacity / 10;
        let small_len = queue_len / 10;
        let mut ghost_queue = vec![];
        ghost_queue.resize_with(queue_len - small_len, || Mutex::new(None));
        Self {
            main_queue: Queue::new(capacity - small_cap),
            small_queue: Queue::new(small_cap),
            ghost_queue,
            evict_tx,
        }
    }

    /// `on_enqueue` will not be invoked if the item is already in the queue.
    pub(crate) async fn read(
        &self,
        item: FifoItem,
        on_enqueue: Option<OnEnqueueCallback>,
    ) -> Result<()> {
        let ident = item.ident.clone();
        debug!("fifo.read"; "ident" => %ident);
        if item.ident.size() > self.small_queue.capacity() as u64 {
            return Err(Error::Other(format!(
                "too large item, ident: {:?}, size: {}",
                item.ident,
                item.ident.size()
            )));
        }
        let mut queue_info = item.segment.lock_queue_info().await;
        if matches!(queue_info.pos, FifoItemPos::Main | FifoItemPos::Small) {
            queue_info.freq = cmp::min(queue_info.freq + 1, 3);
        } else {
            queue_info.freq = 0;
            if self.is_in_ghost(&item.ident).await {
                self.insert_main(item, queue_info, on_enqueue).await;
            } else {
                self.insert_small(item, queue_info, on_enqueue).await;
            }
        }
        Ok(())
    }

    async fn insert_small(
        &self,
        item: FifoItem,
        mut queue_info: FileSegmentQueueGuard,
        on_enqueue: Option<OnEnqueueCallback>,
    ) {
        let ident = item.ident.clone();
        queue_info.pos = FifoItemPos::Small;
        self.small_queue.push(item);
        drop(queue_info);
        if let Some(cb) = on_enqueue {
            cb();
        }

        self.evict_small(ident).await;
    }

    async fn evict_small(&self, ident: FileSegmentIdent) {
        debug!("fifo.evict_small"; "for" => %ident);
        let mut evicted_items = vec![];

        while self.small_queue.is_oversize() {
            let Some(tail) = self.small_queue.pop() else {
                break;
            };

            let mut queue_info = tail.segment.lock_queue_info().await;
            debug_assert_matches!(queue_info.pos, FifoItemPos::Small);
            debug!("fifo.evict_tail (small)";
                "tail" => %tail.ident,
                "queue_info" => ?queue_info,
                "for" => %ident);
            if queue_info.freq > 1 {
                self.insert_main(tail, queue_info, None).await;
            } else {
                self.insert_ghost(tail.ident.clone()).await;
                queue_info.pos = FifoItemPos::ToEvict;
                evicted_items.push(tail);
            };
        }

        if !evicted_items.is_empty() {
            debug!("fifo.evict_small send evict items"; "for" => %ident, "items" => ?evicted_items);
            self.send_evict_items(evicted_items).await;
        }
    }

    async fn insert_main(
        &self,
        item: FifoItem,
        mut queue_info: FileSegmentQueueGuard,
        on_enqueue: Option<OnEnqueueCallback>,
    ) {
        debug!("fifo.insert_main"; "ident" => %item.ident);
        let ident = item.ident.clone();

        queue_info.pos = FifoItemPos::Main;
        self.main_queue.push(item);
        drop(queue_info);
        if let Some(cb) = on_enqueue {
            cb();
        }

        self.evict_main(ident).await;
    }

    async fn evict_main(&self, ident: FileSegmentIdent) {
        debug!("fifo.evict_main"; "for" => %ident);
        let mut evicted_items = vec![];

        while self.main_queue.is_oversize() {
            let Some(tail) = self.main_queue.pop() else {
                break;
            };
            let mut queue_info = tail.segment.lock_queue_info().await;
            debug_assert_matches!(queue_info.pos, FifoItemPos::Main);
            debug!("fifo.evict_tail (main)";
                "tail" => %tail.ident,
                "queue_info" => ?queue_info,
                "for" => %ident);
            if queue_info.freq > 0 {
                queue_info.freq -= 1;
                self.main_queue.push(tail);
            } else {
                queue_info.pos = FifoItemPos::ToEvict;
                evicted_items.push(tail);
            };
        }

        if !evicted_items.is_empty() {
            debug!("fifo.evict_main send evicted items"; "for" => %ident, "items" => ?evicted_items);
            self.send_evict_items(evicted_items).await;
        }
    }

    async fn is_in_ghost(&self, ident: &FileSegmentIdent) -> bool {
        let fingerprint = ident.fingerprint();
        self.ghost_queue[fingerprint as usize % self.ghost_queue.len()]
            .lock()
            .await
            .as_ref()
            .map_or(false, |x| x == ident)
    }

    async fn insert_ghost(&self, ident: FileSegmentIdent) {
        let fingerprint = ident.fingerprint();
        let mut guard = self.ghost_queue[fingerprint as usize % self.ghost_queue.len()]
            .lock()
            .await;
        *guard = Some(ident);
    }

    async fn send_evict_items(&self, items: Vec<FifoItem>) {
        if let Err(err) = self.evict_tx.send(EvictTask { items, cb: None }).await {
            warn!("failed to send evict items to channel"; "err" => ?err);
        }
    }
}

struct Queue {
    queue: SegQueue<FifoItem>,
    cap: i64,
    total_size: AtomicI64,
}

impl Queue {
    fn new(cap: i64) -> Self {
        Self {
            queue: SegQueue::new(),
            cap,
            total_size: AtomicI64::new(0),
        }
    }

    fn push(&self, item: FifoItem) {
        self.total_size.fetch_add(item.size() as i64, Relaxed);
        self.queue.push(item);
    }

    fn pop(&self) -> Option<FifoItem> {
        let tail = self.queue.pop();
        if let Some(item) = tail.as_ref() {
            self.total_size.fetch_sub(item.size() as i64, Relaxed);
        }
        tail
    }

    fn is_oversize(&self) -> bool {
        self.total_size.load(Relaxed) > self.cap
    }

    fn capacity(&self) -> i64 {
        self.cap
    }
}
