// Copyright 2021 TiKV Project Authors. Licensed under Apache-2.0.

use std::{cmp::Ordering::*, fmt, iter::Iterator as _, mem};

use crate::{next, next_async, next_version, next_version_async, table::*};

#[derive(Debug)]
pub struct MergeIterator<'a> {
    smaller: Box<MergeIteratorChild<'a>>,
    bigger: Box<MergeIteratorChild<'a>>,
    reverse: bool,
    same_key: bool,
}

pub(crate) struct MergeIteratorChild<'a> {
    valid: bool,
    iter: Box<dyn Iterator + 'a>,
    ver: u64,
}

impl fmt::Debug for MergeIteratorChild<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut de = f.debug_struct("MergeIteratorChild");
        de.field("valid", &self.valid).field("ver", &self.ver);
        if self.valid {
            de.field("key", &self.iter.key());
        }
        de.finish()
    }
}

impl<'a> MergeIteratorChild<'a> {
    pub(crate) fn new(iter: Box<dyn Iterator + 'a>) -> Self {
        MergeIteratorChild {
            valid: false,
            iter,
            ver: 0,
        }
    }

    fn reset(&mut self) {
        self.valid = self.iter.valid();
        if self.valid {
            self.ver = self.iter.value().version;
        }
    }
}

impl Iterator for MergeIterator<'_> {
    fn next(&mut self) {
        self.smaller.iter.next();
        self.smaller.reset();
        if self.same_key && self.bigger.valid {
            self.bigger.iter.next();
            self.bigger.reset();
        }
        self.fix()
    }

    fn next_version(&mut self) -> bool {
        let curr_version = self.smaller.ver;
        if self.smaller.iter.next_version() {
            self.smaller.reset();
            if self.same_key
                && self.bigger.valid
                && self.bigger.ver < curr_version
                && self.bigger.ver > self.smaller.ver
            {
                self.swap();
            }
            return true;
        }
        if !self.same_key {
            return false;
        }
        if !self.bigger.valid {
            return false;
        }
        if self.smaller.ver < self.bigger.ver {
            return false;
        }
        if self.smaller.ver == self.bigger.ver {
            debug_assert!(false, "ver is equal: {:?}", self);
            // have duplicated key in the two iterators.
            if self.bigger.iter.next_version() {
                self.bigger.reset();
                self.swap();
                return true;
            }
            return false;
        }
        self.swap();
        true
    }

    fn rewind(&mut self) {
        self.smaller.iter.rewind();
        self.smaller.reset();
        self.bigger.iter.rewind();
        self.bigger.reset();
        self.fix();
    }

    fn seek(&mut self, key: InnerKey<'_>) {
        self.smaller.iter.seek(key);
        self.smaller.reset();
        self.bigger.iter.seek(key);
        self.bigger.reset();
        self.fix();
    }

    fn key(&self) -> InnerKey<'_> {
        self.smaller.iter.key()
    }

    fn value(&self) -> Value {
        self.smaller.iter.value()
    }

    fn valid(&self) -> bool {
        self.smaller.valid
    }
}

impl<'a> MergeIterator<'a> {
    pub(crate) fn new(
        first: Box<MergeIteratorChild<'a>>,
        second: Box<MergeIteratorChild<'a>>,
        reverse: bool,
    ) -> Self {
        Self {
            smaller: first,
            bigger: second,
            reverse,
            same_key: false,
        }
    }

    fn fix(&mut self) {
        if !self.bigger.valid {
            return;
        }
        if self.smaller.valid {
            match self.smaller.iter.key().cmp(&self.bigger.iter.key()) {
                Equal => {
                    self.same_key = true;
                    debug_assert!(
                        self.smaller.ver != self.bigger.ver,
                        "ver is equal: {:?}",
                        self
                    );
                    if self.smaller.ver < self.bigger.ver {
                        self.swap();
                    }
                }
                Less => {
                    self.same_key = false;
                    if self.reverse {
                        self.swap();
                    }
                }
                Greater => {
                    self.same_key = false;
                    if !self.reverse {
                        self.swap();
                    }
                }
            }
            return;
        }
        self.swap();
    }

    fn swap(&mut self) {
        mem::swap(&mut self.smaller, &mut self.bigger)
    }
}

struct HeapNode<'a> {
    valid: bool,
    iter: Box<dyn Iterator + 'a>,
    ver: Option<u64>,
    is_next_sync: bool,
}

impl fmt::Debug for HeapNode<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut de = f.debug_struct("HeapNode");
        de.field("valid", &self.valid)
            .field("ver", &self.ver)
            .field("is_next_sync", &self.is_next_sync);
        if self.valid {
            de.field("key", &self.iter.key());
        }
        de.finish()
    }
}

impl<'a> HeapNode<'a> {
    fn new(iter: Box<dyn Iterator + 'a>) -> Self {
        HeapNode {
            valid: false,
            iter,
            ver: None,
            is_next_sync: false,
        }
    }

    fn reset(&mut self) {
        self.valid = self.iter.valid();
        if self.valid {
            self.ver = Some(self.iter.value().version);
            self.is_next_sync = self.iter.is_next_sync();
        }
    }

    #[maybe_async::both]
    #[inline]
    async fn next(&mut self) {
        self.iter.next().await;
        self.reset();
    }

    // To make `HeapNode` be able to use with `next!`.
    #[inline]
    fn is_next_sync(&self) -> bool {
        self.is_next_sync
    }

    #[maybe_async::both]
    #[inline]
    async fn next_version(&mut self) -> bool {
        let ok = self.iter.next_version().await;
        self.ver = ok.then(|| self.iter.value().version);
        ok
    }

    #[inline]
    fn is_next_version_sync(&self) -> bool {
        self.iter.is_next_version_sync()
    }
}

#[derive(Debug)]
pub struct AsyncMergeIterator<'a> {
    heap: Vec<HeapNode<'a>>,
    len: usize,
    reverse: bool,
    is_always_sync: bool,
}

impl<'a> AsyncMergeIterator<'a> {
    /// `is_always_sync`: Set `true` when all `iters` are ALWAYS sync.
    pub fn new(iters: Vec<Box<dyn Iterator + 'a>>, reverse: bool, is_always_sync: bool) -> Self {
        Self {
            heap: iters.into_iter().map(HeapNode::new).collect(),
            len: 0,
            reverse,
            is_always_sync,
        }
    }

    fn down(&mut self, i0: usize) -> bool {
        let n = self.len;
        let mut i = i0;
        loop {
            let left = 2 * i + 1;
            if left >= n {
                break;
            }
            let right = left + 1;
            let j = if right < n && self.less(right, left) {
                right
            } else {
                left
            };
            if !self.less(j, i) {
                break;
            }
            self.heap.swap(i, j);
            i = j;
        }
        i > i0
    }

    fn less(&self, m: usize, n: usize) -> bool {
        debug_assert!(m < self.len && n < self.len);
        let m = &self.heap[m];
        let n = &self.heap[n];
        debug_assert!(m.valid && n.valid);

        let mut cmp = m.iter.key().cmp(&n.iter.key());
        if cmp.is_eq() {
            return match (m.ver, n.ver) {
                (Some(m_ver), Some(n_ver)) => {
                    debug_assert!(m_ver != n_ver, "ver is equal: {:?}", self);
                    m_ver > n_ver
                }
                (None, None) => false, // To make `down()` break loop earlier.
                (Some(_), None) => true,
                (None, Some(_)) => false,
            };
        }
        if self.reverse {
            cmp = cmp.reverse()
        }
        cmp.is_lt()
    }

    // Used after all heap nodes are reset.
    fn init_heap(&mut self) {
        self.len = self.heap.len();
        let mut i = 0;
        while i < self.len {
            if self.heap[i].valid {
                i += 1;
            } else {
                self.remove(i, false);
            }
        }
        for i in (0..self.len / 2).rev() {
            self.down(i);
        }
    }

    fn remove(&mut self, i: usize, fix_heap: bool) {
        debug_assert!(i < self.len);
        self.len -= 1;
        if i < self.len {
            self.heap.swap(i, self.len);
            if fix_heap {
                self.down(i);
            }
        }
    }
}

#[maybe_async::async_trait]
impl Iterator for AsyncMergeIterator<'_> {
    #[maybe_async]
    async fn next(&mut self) {
        if self.len == 0 {
            return;
        }

        // SAFETY: extend the lifetime. The following loop will not change
        // `self.heap[0]` when using `last_key`.
        let last_key =
            unsafe { std::mem::transmute::<InnerKey<'_>, InnerKey<'static>>(self.key()) };
        let len = self.len;
        for i in (0..len).rev() {
            let key = self.heap[i].iter.key();
            debug_assert!(
                self.heap[i].valid
                    && (!self.reverse && key >= last_key || self.reverse && key <= last_key)
            );
            if i == 0 || key == last_key {
                next!(self.heap[i]).await;
                if self.heap[i].valid {
                    self.down(i);
                } else {
                    self.remove(i, true);
                }
            }
        }
    }

    fn is_next_sync(&self) -> bool {
        if self.is_always_sync || self.len == 0 {
            return true;
        }

        if !self.heap[0].is_next_sync() {
            return false;
        }
        let key = self.key();
        self.heap[1..self.len]
            .iter()
            .all(|n| n.is_next_sync() || n.iter.key() != key)
    }

    #[maybe_async]
    async fn next_version(&mut self) -> bool {
        if self.len == 0 || self.heap[0].ver.is_none() {
            return false;
        }

        next_version!(self.heap[0]).await;
        self.down(0);
        self.heap[0].ver.is_some()
    }

    fn is_next_version_sync(&self) -> bool {
        self.is_always_sync || self.len == 0 || self.heap[0].is_next_version_sync()
    }

    #[maybe_async]
    async fn rewind(&mut self) {
        for n in self.heap.iter_mut() {
            n.iter.rewind().await;
            n.reset()
        }
        self.init_heap();
    }

    #[maybe_async]
    async fn seek(&mut self, key: InnerKey<'_>) {
        for n in self.heap.iter_mut() {
            n.iter.seek(key).await;
            n.reset();
        }
        self.init_heap();
    }

    fn key(&self) -> InnerKey<'_> {
        self.heap[0].iter.key()
    }

    fn value(&self) -> Value {
        self.heap[0].iter.value()
    }

    fn valid(&self) -> bool {
        self.len > 0
    }
}
