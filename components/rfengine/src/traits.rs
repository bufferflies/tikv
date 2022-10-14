// Copyright 2021 TiKV Project Authors. Licensed under Apache-2.0.

use engine_traits::{RaftEngine, RaftEngineReadOnly, RaftLogBatch, Result};
use kvproto::raft_serverpb::RaftLocalState;
use raft::eraftpb::Entry;

use crate::{metrics::*, RfEngine, WriteBatch};

impl RaftEngineReadOnly for RfEngine {
    fn get_raft_state(&self, _raft_group_id: u64) -> Result<Option<RaftLocalState>> {
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

    fn put_raft_state(&self, _raft_group_id: u64, _state: &RaftLocalState) -> Result<()> {
        panic!()
    }

    fn gc(&self, _raft_group_id: u64, mut _from: u64, _to: u64) -> Result<usize> {
        panic!()
    }

    fn purge_expired_files(&self) -> Result<Vec<u64>> {
        panic!()
    }

    fn flush_metrics(&self, instance: &str) {
        flush_engine_properties(&self, instance);
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
}

impl RaftLogBatch for WriteBatch {
    fn append(&mut self, _raft_group_id: u64, _entries: Vec<Entry>) -> Result<()> {
        panic!()
    }

    fn cut_logs(&mut self, _raft_group_id: u64, _from: u64, _to: u64) {
        panic!()
    }

    fn put_raft_state(&mut self, _raft_group_id: u64, _state: &RaftLocalState) -> Result<()> {
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
