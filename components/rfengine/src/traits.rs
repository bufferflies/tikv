// Copyright 2021 TiKV Project Authors. Licensed under Apache-2.0.

use engine_traits::{
    Error, PerfContextExt, PerfContextKind, PerfLevel, RaftEngine, RaftEngineReadOnly,
    RaftLogBatch, Result,
};
use kvproto::{
    metapb::Region,
    raft_serverpb::{RaftApplyState, RaftLocalState, RegionLocalState, StoreIdent},
};
use raft::eraftpb::Entry;

use crate::{RfEngine, WriteBatch};

impl RaftEngineReadOnly for RfEngine {
    fn is_empty(&self) -> Result<bool> {
        panic!()
    }

    fn get_store_ident(&self) -> Result<Option<StoreIdent>> {
        panic!()
    }

    fn get_prepare_bootstrap_region(&self) -> Result<Option<Region>> {
        panic!()
    }

    fn get_raft_state(&self, _raft_group_id: u64) -> Result<Option<RaftLocalState>> {
        panic!()
    }

    fn get_region_state(&self, _raft_group_id: u64) -> Result<Option<RegionLocalState>> {
        panic!()
    }

    fn get_apply_state(&self, _raft_group_id: u64) -> Result<Option<RaftApplyState>> {
        panic!()
    }

    fn get_recover_state(&self) -> Result<Option<kvproto::raft_serverpb::StoreRecoverState>> {
        panic!()
    }

    fn get_entry(&self, _raft_group_id: u64, _index: u64) -> Result<Option<Entry>> {
        panic!()
    }

    fn fetch_entries_to(
        &self,
        _region_id: u64,
        _low: u64,
        _high: u64,
        _max_size: Option<usize>, // size limit of fetched entries
        _buf: &mut Vec<Entry>,
    ) -> Result<usize> /* entry count */ {
        panic!()
    }

    fn get_all_entries_to(&self, _region_id: u64, _buf: &mut Vec<Entry>) -> Result<()> {
        unreachable!("todo")
    }
}

impl PerfContextExt for RfEngine {
    type PerfContext = PerfContext;

    fn get_perf_context(_level: PerfLevel, _kind: PerfContextKind) -> Self::PerfContext {
        panic!()
    }
}

pub struct PerfContext {}

impl engine_traits::PerfContext for PerfContext {
    fn start_observe(&mut self) {
        panic!()
    }

    fn report_metrics(&mut self, _trackers: &[tracker::TrackerToken]) {
        panic!()
    }
}

impl RaftEngine for RfEngine {
    type LogBatch = WriteBatch;

    fn log_batch(&self, _capacity: usize) -> Self::LogBatch {
        panic!()
    }

    fn sync(&self) -> Result<()> {
        panic!()
    }

    fn consume(&self, _batch: &mut Self::LogBatch, _sync_log: bool) -> Result<usize> {
        panic!()
    }

    fn consume_and_shrink(
        &self,
        _batch: &mut Self::LogBatch,
        _sync_log: bool,
        _max_capacity: usize,
        _shrink_to: usize,
    ) -> Result<usize> {
        panic!()
    }

    fn clean(
        &self,
        _raft_group_id: u64,
        _first_index: u64,
        _state: &RaftLocalState,
        _batch: &mut Self::LogBatch,
    ) -> Result<()> {
        panic!()
    }

    fn append(&self, _raft_group_id: u64, _entries: Vec<Entry>) -> Result<usize> {
        panic!()
    }

    fn put_store_ident(&self, _ident: &StoreIdent) -> Result<()> {
        panic!()
    }

    fn put_raft_state(&self, _raft_group_id: u64, _state: &RaftLocalState) -> Result<()> {
        panic!()
    }

    fn gc(&self, _raft_group_id: u64, mut _from: u64, _to: u64) -> Result<usize> {
        panic!()
    }

    fn reset_statistics(&self) {
        panic!()
    }

    fn dump_stats(&self) -> Result<String> {
        panic!()
    }

    fn get_engine_size(&self) -> Result<u64> {
        panic!()
    }

    fn get_engine_path(&self) -> &str {
        panic!()
    }

    fn for_each_raft_group<E, F>(&self, _f: &mut F) -> std::result::Result<(), E>
    where
        F: FnMut(u64) -> std::result::Result<(), E>,
        E: From<Error>,
    {
        panic!()
    }

    fn put_recover_state(&self, _state: &kvproto::raft_serverpb::StoreRecoverState) -> Result<()> {
        panic!()
    }
}

impl RaftLogBatch for WriteBatch {
    fn append(&mut self, _raft_group_id: u64, _entries: Vec<Entry>) -> Result<()> {
        panic!()
    }

    fn cut_logs(&mut self, _raft_group_id: u64, _from: u64, _to: u64) {
        panic!()
    }

    fn put_store_ident(&mut self, _ident: &StoreIdent) -> Result<()> {
        panic!()
    }

    fn put_prepare_bootstrap_region(&mut self, _region: &Region) -> Result<()> {
        panic!()
    }

    fn remove_prepare_bootstrap_region(&mut self) -> Result<()> {
        panic!()
    }

    fn put_raft_state(&mut self, _raft_group_id: u64, _state: &RaftLocalState) -> Result<()> {
        panic!()
    }

    fn put_region_state(&mut self, _raft_group_id: u64, _state: &RegionLocalState) -> Result<()> {
        panic!()
    }

    fn put_apply_state(&mut self, _raft_group_id: u64, _state: &RaftApplyState) -> Result<()> {
        panic!()
    }

    fn persist_size(&self) -> usize {
        panic!()
    }

    fn is_empty(&self) -> bool {
        panic!()
    }

    fn merge(&mut self, _: Self) -> Result<()> {
        panic!()
    }
}
