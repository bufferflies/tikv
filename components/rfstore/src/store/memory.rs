// Copyright 2025 TiKV Project Authors. Licensed under Apache-2.0.

use std::sync::Arc;

use lazy_static::lazy_static;
use tikv_alloc::{
    mem_trace,
    trace::{Id, MemoryTrace},
};

// Copy from `components/raftstore/fsm/apply.rs`.
// todo: we need to add some memory trace metrics.
lazy_static! {
    pub static ref MEMTRACE_ROOT: Arc<MemoryTrace> = mem_trace!(
        raftstore, [ applys ]
    );

    /// Memory usage for apply fsms.
    pub static ref MEMTRACE_APPLYS: Arc<MemoryTrace> =
        MEMTRACE_ROOT.sub_trace(Id::Name("applys"));
}
