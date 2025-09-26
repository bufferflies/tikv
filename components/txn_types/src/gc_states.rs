// Copyright 2025 TiKV Project Authors. Licensed under Apache-2.0.

use std::time::Duration;

use collections::HashMap;
use kvproto::pdpb;
use tikv_util::time::Instant;

use crate::TimeStamp;

#[derive(Clone, Debug)]
pub struct GcBarrier {
    pub barrier_id: String,
    pub barrier_ts: TimeStamp,
    pub ttl: Option<Duration>,
    #[allow(dead_code)]
    fetched_start_time: Instant,
}

impl GcBarrier {
    pub fn new(
        barrier_id: String,
        barrier_ts: TimeStamp,
        ttl: Option<Duration>,
        fetched_start_time: Instant,
    ) -> Self {
        Self {
            barrier_id,
            barrier_ts,
            ttl,
            fetched_start_time,
        }
    }

    pub fn from_pb(pb: pdpb::GcBarrierInfo, fetch_time: Instant) -> Self {
        let ttl = match pb.get_ttl_seconds() {
            i64::MAX => None,
            t if t <= 0 => Some(Duration::from_secs(0)),
            t => Some(Duration::from_secs(t as u64)),
        };
        Self {
            barrier_id: pb.get_barrier_id().to_owned(),
            barrier_ts: pb.get_barrier_ts().into(),
            ttl,
            fetched_start_time: fetch_time,
        }
    }
}

fn keyspace_id_from_keyspace_scope_pb(pb: Option<&pdpb::KeyspaceScope>) -> u32 {
    match pb {
        Some(k) => k.get_keyspace_id(),
        None => NULL_KEYSPACE_ID,
    }
}

pub const NULL_KEYSPACE_ID: u32 = 0xffffffff;
pub const DEFAULT_KEYSPACE_ID: u32 = 0;

#[derive(Clone, Debug)]
pub struct GcState {
    pub keyspace_id: u32,
    pub is_keyspace_level_gc: bool,
    pub gc_safe_point: TimeStamp,
    pub txn_safe_point: TimeStamp,
    pub gc_barriers: Vec<GcBarrier>,
    #[allow(dead_code)]
    fetched_start_time: Instant,
}

impl GcState {
    pub fn default(keyspace_id: u32) -> Self {
        Self {
            keyspace_id,
            is_keyspace_level_gc: keyspace_id != NULL_KEYSPACE_ID,
            gc_safe_point: TimeStamp::zero(),
            txn_safe_point: TimeStamp::zero(),
            gc_barriers: vec![],
            fetched_start_time: Instant::now_coarse(),
        }
    }

    pub fn new(
        keyspace_id: u32,
        is_keyspace_level_gc: bool,
        gc_safe_point: TimeStamp,
        txn_safe_point: TimeStamp,
        gc_barriers: Vec<GcBarrier>,
        fetched_start_time: Instant,
    ) -> Self {
        Self {
            keyspace_id,
            is_keyspace_level_gc,
            gc_safe_point,
            txn_safe_point,
            gc_barriers,
            fetched_start_time,
        }
    }

    pub fn from_pb(pb: pdpb::GcState, fetch_time: Instant) -> Self {
        Self {
            keyspace_id: keyspace_id_from_keyspace_scope_pb(pb.keyspace_scope.as_ref()),
            is_keyspace_level_gc: pb.is_keyspace_level_gc,
            gc_safe_point: pb.gc_safe_point.into(),
            txn_safe_point: pb.txn_safe_point.into(),
            gc_barriers: pb
                .gc_barriers
                .into_iter()
                .map(|b| GcBarrier::from_pb(b, fetch_time))
                .collect(),
            fetched_start_time: fetch_time,
        }
    }
}

#[derive(Clone, Debug)]
pub struct ClusterGcStates {
    pub keyspace_gc_states: HashMap<u32, GcState>,
    #[allow(dead_code)]
    fetched_start_time: Instant,
}

impl ClusterGcStates {
    pub fn new(keyspace_gc_states: HashMap<u32, GcState>, fetched_start_time: Instant) -> Self {
        Self {
            keyspace_gc_states,
            fetched_start_time,
        }
    }

    pub fn from_pb(pb: pdpb::GetAllKeyspacesGcStatesResponse, fetch_time: Instant) -> Self {
        Self {
            keyspace_gc_states: pb
                .gc_states
                .into_iter()
                .map(|s| {
                    let gc_state = GcState::from_pb(s, fetch_time);
                    (gc_state.keyspace_id, gc_state)
                })
                .collect(),
            fetched_start_time: fetch_time,
        }
    }
}
