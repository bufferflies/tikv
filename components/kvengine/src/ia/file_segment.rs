// Copyright 2024 TiKV Project Authors. Licensed under Apache-2.0.

// TODO: remote this
#![allow(dead_code)]

use std::fmt;

use crate::ia::queue::FifoItem;

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
