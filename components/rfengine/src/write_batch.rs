// Copyright 2022 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    collections::{hash_map::Entry, HashMap, VecDeque},
    ops::{Deref, DerefMut},
};

use bytes::{Buf, BufMut, Bytes};
use raft_proto::eraftpb;

use crate::{log_batch::RaftLogOp, PeerMeta};

/// `WriteBatch` contains multiple regions' `RegionBatch`.
#[derive(Default)]
pub struct WriteBatch {
    pub(crate) peers: HashMap<u64, PeerBatch>,
}

impl WriteBatch {
    pub fn new() -> Self {
        Self {
            peers: Default::default(),
        }
    }

    pub(crate) fn get_peer(&mut self, peer_id: u64, region_id: u64) -> &mut PeerBatch {
        self.peers
            .entry(peer_id)
            .or_insert_with(|| PeerBatch::new(peer_id, region_id))
    }

    pub fn append_raft_log(&mut self, peer_id: u64, region_id: u64, entry: &eraftpb::Entry) {
        let op = RaftLogOp::new(entry);
        self.get_peer(peer_id, region_id).append_raft_log(op);
    }

    pub fn truncate_raft_log(&mut self, peer_id: u64, region_id: u64, index: u64) {
        self.get_peer(peer_id, region_id).truncate(index);
    }

    pub fn set_state(&mut self, peer_id: u64, region_id: u64, key: &[u8], val: &[u8]) {
        self.get_peer(peer_id, region_id).set_state(key, val);
    }

    pub fn get_state(&mut self, peer_id: u64, region_id: u64, key: &[u8]) -> Option<&[u8]> {
        self.get_peer(peer_id, region_id).get_state(key)
    }

    pub fn clear_peer(&mut self, peer_id: u64) {
        self.peers.remove(&peer_id);
    }

    pub fn reset(&mut self) {
        self.peers.clear()
    }

    pub fn is_empty(&self) -> bool {
        self.peers.is_empty()
    }

    pub(crate) fn merge_peer(&mut self, peer_batch: PeerBatch) {
        match self.peers.entry(peer_batch.peer_id) {
            Entry::Occupied(old_peer_batch) => {
                old_peer_batch.into_mut().merge(peer_batch);
            }
            Entry::Vacant(e) => {
                e.insert(peer_batch);
            }
        }
    }

    pub fn merge_write_batch(&mut self, other: WriteBatch) {
        for (_, peer_batch) in other.peers {
            self.merge_peer(peer_batch);
        }
    }

    pub fn iterate_peer_states(&self, peer_id: u64, mut f: impl FnMut(&[u8], &[u8])) {
        if let Some(peer_batch) = self.peers.get(&peer_id) {
            for (k, v) in &peer_batch.states {
                f(k.chunk(), v.chunk());
            }
        }
    }

    pub fn get_region_peer_map(&self) -> HashMap<u64, u64> {
        self.peers.iter().map(|(k, v)| (v.region_id, *k)).collect()
    }
}

/// `RegionBatch` is a batch of modifications in one region.
pub(crate) struct PeerBatch {
    pub(crate) peer_id: u64,
    pub(crate) meta: PeerMeta,
    pub(crate) raft_logs: VecDeque<RaftLogOp>,
}

impl Deref for PeerBatch {
    type Target = PeerMeta;
    fn deref(&self) -> &Self::Target {
        &self.meta
    }
}

impl DerefMut for PeerBatch {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.meta
    }
}

impl PeerBatch {
    pub(crate) fn new(peer_id: u64, region_id: u64) -> Self {
        Self {
            peer_id,
            meta: PeerMeta::new(region_id),
            raft_logs: Default::default(),
        }
    }

    pub(crate) fn truncate(&mut self, idx: u64) {
        if idx < self.truncated_idx {
            return;
        }
        while let Some(true) = self.raft_logs.front().map(|l| l.index <= idx) {
            self.raft_logs.pop_front();
        }
        self.truncated_idx = idx;
    }

    pub fn set_state(&mut self, key: &[u8], val: &[u8]) {
        self.states
            .insert(Bytes::copy_from_slice(key), Bytes::copy_from_slice(val));
    }

    pub fn get_state(&self, key: &[u8]) -> Option<&[u8]> {
        self.states.get(key).map(|v| v.chunk())
    }

    pub fn append_raft_log(&mut self, op: RaftLogOp) {
        debug_assert!(
            self.truncated_idx < op.index,
            "{} {}",
            self.truncated_idx,
            op.index
        );
        while let Some(true) = self.raft_logs.back().map(|l| l.index + 1 != op.index) {
            self.raft_logs.pop_back();
        }
        debug_assert!(
            self.raft_logs.is_empty()
                || self
                    .raft_logs
                    .back()
                    .map(|l| l.index + 1 == op.index)
                    .unwrap()
        );
        self.raft_logs.push_back(op);
    }

    pub fn merge(&mut self, other: PeerBatch) {
        debug_assert_eq!(self.peer_id, other.peer_id);
        self.meta.merge(&other.meta, true);
        let truncated_idx = other.truncated_idx;
        for op in other.raft_logs {
            self.append_raft_log(op);
        }
        self.truncate(truncated_idx);
    }

    pub(crate) fn encoded_len(&self) -> usize {
        let mut len = 8 /* peer_id */ + 8 /* region_id */ + 8 /* truncated_idx */ + 8 /* start_index */ + 8 /* end_index */ + 4 /* states_len */;
        for (key, val) in &self.states {
            len += 2 /* key_len */ + key.len() + 4 /* val_len */ + val.len();
        }
        len += self.raft_logs.len() * 4 /* log_end_offset */;
        self.raft_logs
            .iter()
            .fold(len, |acc, l| acc + l.encoded_len())
    }

    ///  +-----------+-------------+-----------------+---------------+--------------+--------------+-----------------+------------+-------------------+--------------+-----+------------------+-----+--------------+-----+
    ///  |peer_id(8B)|region_id(8B)|truncated_idx(8B)|first_index(8B)|last_index(8B)|states_len(4B)|state_key_len(2B)|state_key(n)|state_value_len(4B)|state_value(n)| ... |log_end_offset(4B)| ... |raft_log_op(n)| ... |
    ///  +-----------+-------------+-----------------+---------------+--------------+--------------+-----------------+------------+-------------------+--------------+-----+------------------+-----+--------------+-----+
    pub(crate) fn encode_to(&self, buf: &mut impl BufMut) {
        buf.put_u64_le(self.peer_id);
        buf.put_u64_le(self.region_id);
        buf.put_u64_le(self.truncated_idx);
        let first = self.raft_logs.front().map_or(0, |x| x.index);
        let end = self.raft_logs.back().map_or(0, |x| x.index + 1);
        buf.put_u64_le(first);
        buf.put_u64_le(end);
        buf.put_u32_le(self.states.len() as u32);
        for (key, val) in &self.meta.states {
            buf.put_u16_le(key.len() as u16);
            buf.put_slice(key.chunk());
            buf.put_u32_le(val.len() as u32);
            buf.put_slice(val.chunk());
        }
        let mut end_off = 0;
        for op in &self.raft_logs {
            end_off += op.encoded_len() as u32;
            buf.put_u32_le(end_off);
        }
        for op in &self.raft_logs {
            op.encode_to(buf);
        }
    }

    pub(crate) fn decode(mut buf: &[u8]) -> Self {
        let mut batch = PeerBatch::new(0, 0);
        batch.peer_id = buf.get_u64_le();
        batch.region_id = buf.get_u64_le();
        batch.truncated_idx = buf.get_u64_le();
        let first = buf.get_u64_le();
        let end = buf.get_u64_le();
        let states_len = buf.get_u32_le();
        for _ in 0..states_len {
            let key_len = buf.get_u16_le() as usize;
            let key = Bytes::copy_from_slice(&buf[..key_len]);
            buf = &buf[key_len..];
            let val_len = buf.get_u32_le() as usize;
            let val = Bytes::copy_from_slice(&buf[..val_len]);
            buf = &buf[val_len..];
            batch.states.insert(key, val);
        }
        let num_logs = (end - first) as usize;
        let log_index_len = num_logs * 4;
        let mut log_index_buf = &buf[..log_index_len];
        buf = &buf[log_index_len..];
        let mut start_index = 0;
        for _ in 0..num_logs {
            let end_index = log_index_buf.get_u32_le() as usize;
            let log_op = RaftLogOp::decode(&buf[start_index..end_index]);
            start_index = end_index;
            batch.raft_logs.push_back(log_op);
        }
        batch
    }
}

#[cfg(test)]
mod tests {
    use eraftpb::{Entry, EntryType};

    use super::*;

    fn new_raft_entry(tp: EntryType, term: u64, index: u64, data: &[u8], context: u8) -> Entry {
        let mut entry = Entry::new();
        entry.set_entry_type(tp);
        entry.set_term(term);
        entry.set_index(index);
        entry.set_data(data.to_vec().into());
        entry.set_context(vec![context].into());
        entry
    }

    #[test]
    fn test_region_batch() {
        let mut region_batch = PeerBatch::new(1, 2);

        let mut logs = vec![];
        for i in 1..=10 {
            let log = RaftLogOp::new(&new_raft_entry(EntryType::EntryNormal, 1, i, b"data", 0));
            logs.push(log.clone());
            region_batch.append_raft_log(log);
        }
        assert_eq!(region_batch.raft_logs, logs);
        region_batch.set_state(b"k1", b"v1");
        region_batch.set_state(b"k2", b"v2");
        region_batch.truncate(5);
        assert_eq!(region_batch.truncated_idx, 5);
        assert_eq!(region_batch.raft_logs, &logs[5..]);

        let mut buf = vec![];
        region_batch.encode_to(&mut buf);
        assert_eq!(buf.len(), region_batch.encoded_len());
        let decoded = PeerBatch::decode(&buf);
        assert_eq!(decoded.region_id, region_batch.region_id);
        assert_eq!(decoded.states, region_batch.states);
        assert_eq!(decoded.raft_logs, region_batch.raft_logs);

        // append conflicted log
        let log = RaftLogOp::new(&new_raft_entry(EntryType::EntryNormal, 1, 6, b"data", 0));
        region_batch.append_raft_log(log.clone());
        assert_eq!(region_batch.raft_logs.len(), 1);
        assert_eq!(region_batch.raft_logs[0], log);

        let mut region_batch = PeerBatch::new(1, 2);
        for log in &logs[..5] {
            region_batch.append_raft_log(log.clone());
        }
        region_batch.set_state(b"k0", b"v0");
        region_batch.merge(decoded);
        assert_eq!(region_batch.raft_logs, logs[5..].to_vec());
        assert_eq!(region_batch.states.len(), 3);
    }
}
