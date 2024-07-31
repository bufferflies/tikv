// Copyright 2021 TiKV Project Authors. Licensed under Apache-2.0.

use std::{cmp::Ordering::*, fmt, mem};

use crate::table::*;

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
