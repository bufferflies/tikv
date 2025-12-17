// Copyright 2016 TiKV Project Authors. Licensed under Apache-2.0.

// #[PerformanceCriticalPath]
use std::{marker::PhantomData, u64};

use engine_traits::{KvEngine, RaftEngine};
use getset::{Getters, MutGetters};

/// Statistics about raft peer.
#[derive(Default, Clone)]
pub struct PeerStat {
    pub written_bytes: u64,
    pub written_keys: u64,
}

#[derive(Getters, MutGetters)]
pub struct Peer<EK, ER>
where
    EK: KvEngine,
    ER: RaftEngine,
{
    _phantom: PhantomData<(EK, ER)>,
}
