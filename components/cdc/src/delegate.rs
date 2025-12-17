// Copyright 2020 TiKV Project Authors. Licensed under Apache-2.0.

use std::sync::atomic::{AtomicUsize, Ordering};

static DOWNSTREAM_ID_ALLOC: AtomicUsize = AtomicUsize::new(0);

/// A unique identifier of a Downstream.
#[derive(Clone, Copy, Debug, PartialEq, Hash)]
pub struct DownstreamId(usize);

impl DownstreamId {
    pub fn new() -> DownstreamId {
        DownstreamId(DOWNSTREAM_ID_ALLOC.fetch_add(1, Ordering::SeqCst))
    }
}

impl Default for DownstreamId {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DownstreamState {
    /// It's just created and rejects change events and resolved timestamps.
    Uninitialized,
    /// It has got a snapshot for incremental scan, and change events will be
    /// accepted. However it still rejects resolved timestamps.
    Initializing,
    /// Incremental scan is finished so that resolved timestamps are acceptable
    /// now.
    Normal,
    Stopped,
}

impl Default for DownstreamState {
    fn default() -> Self {
        Self::Uninitialized
    }
}

impl DownstreamState {
    pub fn ready_for_change_events(&self) -> bool {
        match *self {
            DownstreamState::Uninitialized | DownstreamState::Stopped => false,
            DownstreamState::Initializing | DownstreamState::Normal => true,
        }
    }

    pub fn ready_for_advancing_ts(&self) -> bool {
        match *self {
            DownstreamState::Normal => true,

            DownstreamState::Uninitialized
            | DownstreamState::Stopped
            | DownstreamState::Initializing => false,
        }
    }
}
