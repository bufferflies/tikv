// Copyright 2021 TiKV Project Authors. Licensed under Apache-2.0.

use std::cell::RefCell;

use byteorder::{ByteOrder, LittleEndian};
use bytes::{Buf, BufMut, Bytes, BytesMut};
use cloud_encryption::EncryptionKey;
use collections::HashSet;
use kvengine::{ShardMeta, ENCRYPTION_KEY};
use kvproto::{
    raft_serverpb::{MergeState, PeerState, RaftMessage},
    *,
};
use protobuf::Message;
use raft::{GetEntriesContext, Ready, StorageError};
use raft_proto::{
    eraftpb,
    eraftpb::{ConfState, HardState},
};
use raft_serverpb::RegionLocalState;
use raftstore::store::{util, util::conf_state_from_region};
use rfengine::{
    self, raft_state_key, region_state_key, KV_ENGINE_META_KEY, RAFT_TRUNCATED_STATE_KEY,
    REGION_META_KEY_PREFIX,
};
use tikv_util::{box_err, debug, info};

use crate::{
    errors::*,
    store::{
        Engines, PeerTag, RaftApplyState, RaftContext, RaftState, RaftTruncatedState, RegionIdVer,
        TERM_KEY,
    },
};

// When we create a region peer, we should initialize its log term/index > 0,
// so that we can force the follower peer to sync the snapshot first.
pub const RAFT_INIT_LOG_TERM: u64 = 5;
pub const RAFT_INIT_LOG_INDEX: u64 = 5;

/// The initial region epoch version.
pub const INIT_EPOCH_VER: u64 = 1;
/// The initial region epoch conf_version.
pub const INIT_EPOCH_CONF_VER: u64 = 1;

pub const JOB_STATUS_PENDING: usize = 0;
pub const JOB_STATUS_RUNNING: usize = 1;
pub const JOB_STATUS_CANCELLING: usize = 2;
pub const JOB_STATUS_CANCELLED: usize = 3;
pub const JOB_STATUS_FINISHED: usize = 4;
pub const JOB_STATUS_FAILED: usize = 5;

/// Possible status returned by `check_applying_snap`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CheckApplyingSnapStatus {
    /// A snapshot is just applied.
    Success,
    /// A snapshot is being applied.
    Applying,
    /// No snapshot is being applied at all or the snapshot is canceled
    Idle,
}

#[derive(Debug, PartialEq)]
pub enum SnapState {
    Relax,
    Applying,
    ApplyAborted,
}

pub struct RestoreSnapResult {
    // prev_region is the region before snapshot applied.
    pub prev_region: metapb::Region,
    pub region: metapb::Region,
    pub destroyed_regions: Vec<metapb::Region>,
    // `snap_last_index` is the last index of snapshot.
    pub snap_last_index: u64,
}

pub(crate) struct PeerStorage {
    pub(crate) engines: Engines,

    pub(crate) peer_id: u64,
    pub(crate) store_id: u64,
    pub(crate) region: metapb::Region,
    // The preprocessed_region applies all the committed conf change, we should use it instead of
    // the current region to do preprocessed_split and preprocessed_conf_change because the current
    // region may be outdated due to slow apply.
    pub(crate) preprocessed_region: Option<metapb::Region>,
    pub(crate) raft_state: RaftState,
    apply_state: RaftApplyState,
    truncated_state: RaftTruncatedState,
    last_term: u64,

    pub(crate) snap_state: SnapState,

    pub(crate) shard_meta: Option<kvengine::ShardMeta>,
    pub(crate) restored_snapshot: Option<(kvenginepb::ChangeSet, u64)>,
    pub(crate) on_apply_snapshot_msgs: Vec<RaftMessage>,
    /// !initial_flushed peer can't generate snapshot until initial_flush change
    /// set is committed. If majority followers are created by raft msg
    /// which always request snapshot from leader, the initial_flush change
    /// set can't be committed due to snapshot not ready. It's a deadlock!
    ///
    /// Peer created by split doesn't request snapshot, so we can wait for those
    /// peer replacing peer created by raft message. We record these
    /// requesting snapshot peers to set their progress to probe manually
    /// due to raft-rs's implementation.
    pub(crate) snapshot_not_ready_peers: RefCell<HashSet<u64>>,
}

impl raft::Storage for PeerStorage {
    fn initial_state(&self) -> raft::Result<raft::RaftState> {
        let hard_state = self.raft_state.get_hard_state();
        if hard_state == HardState::default() {
            assert!(
                !self.is_initialized(),
                "peer {} for region {:?} is initialized but local state {:?} has empty hard \
                 state",
                self.peer_id,
                self.region,
                self.raft_state
            );

            return Ok(raft::RaftState::new(hard_state, ConfState::default()));
        }
        Ok(raft::RaftState::new(
            hard_state,
            util::conf_state_from_region(self.region()),
        ))
    }

    fn entries(
        &self,
        low: u64,
        high: u64,
        max_size: impl Into<Option<u64>>,
        _: GetEntriesContext,
    ) -> raft::Result<Vec<eraftpb::Entry>> {
        self.check_range(low, high)?;
        let mut ents = Vec::with_capacity((high - low) as usize);
        if low == high {
            return Ok(ents);
        }
        let peer_id = self.peer_id;
        self.engines.raft.fetch_raft_entries_to(
            peer_id,
            low,
            high,
            max_size.into().map(|x| x as usize),
            &mut ents,
        )?;
        Ok(ents)
    }

    fn term(&self, idx: u64) -> raft::Result<u64> {
        if self.shard_meta.is_none() {
            return Ok(0);
        }
        if idx == self.last_index() {
            return Ok(self.last_term());
        }
        if idx == self.truncated_index() {
            return Ok(self.truncated_term());
        }
        self.check_range(idx, idx + 1)?;
        if self.truncated_term() == self.last_term {
            return Ok(self.last_term);
        }
        let peer_id = self.peer_id;
        Ok(self
            .engines
            .raft
            .get_term(peer_id, idx)
            .unwrap_or_else(|| {
                panic!(
                    "failed to get term: region {}, idx {}, first_index {}, last_index {}, applied_index {}, truncated_index {}, region_stats: {:?}",
                    self.tag(), idx, self.first_index(), self.last_index(), self.applied_index(), self.truncated_index(), self.engines.raft.get_peer_stats(peer_id)
                );
            }))
    }

    fn first_index(&self) -> raft::Result<u64> {
        Ok(self.first_index())
    }

    fn last_index(&self) -> raft::Result<u64> {
        Ok(self.last_index())
    }

    fn snapshot(&self, request_index: u64, to: u64) -> raft::Result<eraftpb::Snapshot> {
        if !self.initial_flushed() {
            info!("shard has not flushed for generating snapshot"; "region" => self.tag(), "to" => to);
            self.snapshot_not_ready_peers.borrow_mut().insert(to);
            return Err(raft::Error::Store(
                StorageError::SnapshotTemporarilyUnavailable,
            ));
        }
        if !self.region_match_preprocessed() {
            // If the region is staler than preprocessed region, the preprocessed region may
            // not contain the to peer, cause the to peer panic. Or the region
            // epoch version may not equal to the shard meta version, cause
            // inconsistency.
            return Err(raft::Error::Store(
                StorageError::SnapshotTemporarilyUnavailable,
            ));
        }
        let snap_index = self.snapshot_index();
        if snap_index < request_index {
            info!("requesting index is too high"; "region" => self.tag(),
                "request_index" => request_index, "snap_index" => snap_index);
            return Err(raft::Error::Store(StorageError::Unavailable));
        }
        let snap_term = self.snapshot_term();

        let mut snap = eraftpb::Snapshot::default();
        let change_set = self.shard_meta.as_ref().unwrap().to_change_set();
        let snap_data = encode_snap_data(self.region(), &change_set);
        snap.set_data(snap_data);
        let mut snap_meta = eraftpb::SnapshotMetadata::default();
        snap_meta.set_index(snap_index);
        snap_meta.set_term(snap_term);
        let conf_state = conf_state_from_region(self.region());
        snap_meta.set_conf_state(conf_state);
        snap.set_metadata(snap_meta);
        self.snapshot_not_ready_peers.borrow_mut().remove(&to);
        Ok(snap)
    }
}

impl PeerStorage {
    pub(crate) fn new(
        engines: Engines,
        region: metapb::Region,
        peer_id: u64,
        store_id: u64,
    ) -> Result<PeerStorage> {
        let raft_state = init_raft_state(&engines.raft, peer_id, &region)?;
        let apply_state = init_apply_state(&engines.kv, &region);
        let truncated_state = init_truncated_state(&engines.raft, peer_id, &region);
        let mut shard_meta: Option<ShardMeta> = None;
        if apply_state.applied_index > 0 {
            let meta = load_engine_meta(&engines.raft, store_id, peer_id).unwrap();
            if let Some(parent) = &meta.parent {
                engines.raft.add_dependent(parent.id, meta.id);
            }
            shard_meta = Some(meta);
        }
        let last_term = init_last_term(&engines, peer_id, &region, raft_state)?;
        Ok(PeerStorage {
            engines,
            peer_id,
            store_id,
            region,
            preprocessed_region: None,
            raft_state,
            apply_state,
            truncated_state,
            last_term,
            snap_state: SnapState::Relax,
            shard_meta,
            restored_snapshot: None,
            on_apply_snapshot_msgs: vec![],
            snapshot_not_ready_peers: RefCell::default(),
        })
    }

    pub(crate) fn tag(&self) -> PeerTag {
        PeerTag::new(self.store_id, RegionIdVer::from_region(&self.region))
    }

    pub(crate) fn truncate_raft_log(
        &mut self,
        wb: &mut rfengine::WriteBatch,
        index: u64,
        term: u64,
    ) {
        self.truncated_state.truncated_index = index;
        self.truncated_state.truncated_index_term = term;
        wb.truncate_raft_log(self.peer_id, self.get_region_id(), index);
        wb.set_state(
            self.peer_id,
            self.get_region_id(),
            RAFT_TRUNCATED_STATE_KEY,
            &self.truncated_state.marshal(),
        );
    }

    pub(crate) fn clear_meta(&mut self, rwb: &mut rfengine::WriteBatch, truncate_logs: bool) {
        let peer_id = self.peer_id;
        let region_id = self.region().id;
        self.engines
            .raft
            .iterate_peer_states(peer_id, false, |k, _| {
                rwb.set_state(peer_id, region_id, k, &[]);
            });
        if truncate_logs {
            self.truncate_raft_log(rwb, rfengine::TRUNCATE_ALL_INDEX, self.raft_state.term);
        }
    }

    pub(crate) fn get_region_id(&self) -> u64 {
        self.region.get_id()
    }

    pub(crate) fn is_initialized(&self) -> bool {
        self.shard_meta.is_some()
    }

    pub(crate) fn initial_flushed(&self) -> bool {
        self.shard_meta
            .as_ref()
            .map_or(false, |m| m.parent.is_none())
    }

    #[inline]
    pub fn first_index(&self) -> u64 {
        self.truncated_index() + 1
    }

    #[inline]
    pub fn last_index(&self) -> u64 {
        self.raft_state.last_index
    }

    #[inline]
    pub fn last_term(&self) -> u64 {
        self.last_term
    }

    #[inline]
    pub fn set_applied_state(&mut self, apply_state: RaftApplyState) {
        self.apply_state = apply_state;
    }

    #[inline]
    pub fn apply_state(&self) -> RaftApplyState {
        self.apply_state
    }

    #[inline]
    pub fn applied_index(&self) -> u64 {
        self.apply_state.applied_index
    }

    #[inline]
    pub fn applied_index_term(&self) -> u64 {
        self.apply_state.applied_index_term
    }

    #[inline]
    pub fn commit_index(&self) -> u64 {
        self.raft_state.commit
    }

    #[inline]
    pub fn truncated_index(&self) -> u64 {
        self.truncated_state.truncated_index
    }

    #[inline]
    pub fn truncated_term(&self) -> u64 {
        self.truncated_state.truncated_index_term
    }

    pub fn snapshot_term(&self) -> u64 {
        self.shard_meta.as_ref().map_or(0, |m| {
            debug!("get property term key");
            m.get_property(TERM_KEY).unwrap().get_u64_le()
        })
    }

    pub fn snapshot_index(&self) -> u64 {
        self.shard_meta.as_ref().unwrap().data_sequence
    }

    pub fn region(&self) -> &metapb::Region {
        &self.region
    }

    pub fn set_region(&mut self, region: metapb::Region) {
        self.region = region;
    }

    #[inline]
    pub fn is_applying_snapshot(&self) -> bool {
        self.snap_state == SnapState::Applying
    }

    pub fn check_range(&self, low: u64, high: u64) -> raft::Result<()> {
        if low > high {
            return Err(raft::Error::Store(raft::StorageError::Other(box_err!(
                "low: {} is greater that high: {}",
                low,
                high
            ))));
        } else if low <= self.truncated_index() {
            return Err(raft::Error::Store(StorageError::Compacted));
        } else if high > self.last_index() + 1 {
            return Err(raft::Error::Store(raft::StorageError::Other(box_err!(
                "entries' high {} is out of bound lastindex {}",
                high,
                self.last_index()
            ))));
        }
        Ok(())
    }

    pub fn flush_cache_metrics(&mut self) {
        // TODO(x)
    }

    pub fn handle_raft_ready(
        &mut self,
        ctx: &mut RaftContext,
        ready: &mut raft::Ready,
        last_preprocessed_index: u64,
    ) -> Option<RestoreSnapResult> {
        let mut res = None;
        let prev_raft_state = self.raft_state;
        if !ready.snapshot().is_empty() {
            let prev_region = self.region().clone();
            self.restore_snapshot(ready, ctx).unwrap();
            let region = self.region.clone();
            res = Some(RestoreSnapResult {
                prev_region,
                region,
                destroyed_regions: vec![],
                snap_last_index: ready.snapshot().get_metadata().get_index(),
            })
        }
        if !ready.entries().is_empty() {
            self.append(ready.take_entries(), &mut ctx.raft_wb);
        }

        // Not initialized means the peer is created from raft message
        // and has not applied snapshot yet, so skip persistent hard state.
        if self.is_initialized() {
            if let Some(hs) = ready.hs() {
                self.raft_state.set_hard_state(hs);
            }
            // When there is snapshot, `self.restore_snapshot` will set
            // `last_preprocessed_index` to the last index of snapshot.
            if ready.snapshot().is_empty() {
                self.raft_state.last_preprocessed_index = last_preprocessed_index;
            }
        }
        if prev_raft_state != self.raft_state || !ready.snapshot().is_empty() {
            self.write_raft_state(ctx);
        }
        res
    }

    pub fn write_raft_state(&mut self, ctx: &mut RaftContext) {
        debug!("{} write raft state {:?}", self.tag(), self.raft_state);
        write_raft_state(
            &mut ctx.raft_wb,
            self.peer_id,
            self.shard_meta.as_ref().unwrap(),
            &self.raft_state,
        );
    }

    fn restore_snapshot(&mut self, ready: &Ready, ctx: &mut RaftContext) -> Result<()> {
        info!(
            "{} peer storage restore snapshot, ready_number {}",
            self.tag(),
            ready.number()
        );
        let snap = ready.snapshot();
        let (region, change_set) = decode_snap_data(snap.get_data())?;
        if region.get_id() != self.get_region_id() {
            return Err(box_err!(
                "mismatch region id {} != {}",
                region.get_id(),
                self.get_region_id()
            ));
        }
        // If the region is created by raft message, parent_id is always None, but it's
        // possible a split request is applied lately and added it to the
        // dependent, so we avoid adding dependent when splitting regions by
        // checking peer existence to handle such a case.
        if let Some(parent_id) = self.parent_id() {
            ctx.remove_dependent(parent_id, self.get_region_id());
        }
        if self.is_initialized() {
            // we can only delete the old data when the peer is initialized.
            self.clear_meta(&mut ctx.raft_wb, false);
        }
        write_peer_state(
            &mut ctx.raft_wb,
            self.peer_id,
            &region,
            PeerState::Normal,
            None,
        );
        let last_index = snap.get_metadata().get_index();
        let last_term = snap.get_metadata().get_term();
        self.raft_state.last_index = last_index;
        self.raft_state.commit = last_index;
        self.raft_state.term = last_term;
        self.raft_state.last_preprocessed_index = last_index;
        self.last_term = last_term;
        self.apply_state.applied_index = last_index;
        self.apply_state.applied_index_term = last_term;
        let shard_meta = ShardMeta::new(self.store_id, &change_set);
        write_engine_meta(&mut ctx.raft_wb, self.peer_id, &shard_meta);
        self.truncate_raft_log(&mut ctx.raft_wb, last_index, last_term);
        self.shard_meta = Some(shard_meta);
        self.region = region;
        self.snap_state = SnapState::Applying;
        self.restored_snapshot = Some((change_set, ready.number()));
        Ok(())
    }

    fn append(&mut self, entries: Vec<eraftpb::Entry>, raft_wb: &mut rfengine::WriteBatch) {
        if entries.is_empty() {
            return;
        }
        for e in &entries {
            raft_wb.append_raft_log(self.peer_id, self.get_region_id(), e);
        }
        let last_entry = entries.last().unwrap();
        self.raft_state.last_index = last_entry.get_index();
        self.last_term = last_entry.get_term();
    }

    pub(crate) fn parent_id(&self) -> Option<u64> {
        self.shard_meta
            .as_ref()
            .and_then(|m| m.parent.as_ref())
            .map(|p| p.id)
    }

    /// The last index of raft logs that have been applied and persisted to the
    /// state machine.
    pub(crate) fn data_persisted_log_index(&self) -> Option<u64> {
        self.shard_meta.as_ref().map(|meta| meta.data_sequence)
    }

    pub(crate) fn get_preprocessed_region(&self) -> &metapb::Region {
        self.preprocessed_region
            .as_ref()
            .unwrap_or_else(|| self.region())
    }

    pub(crate) fn region_match_preprocessed(&self) -> bool {
        self.preprocessed_region
            .as_ref()
            .map(|p| p.get_region_epoch() == self.region.get_region_epoch())
            .unwrap_or(true)
    }

    pub(crate) fn get_encryption_key(&self) -> Option<EncryptionKey> {
        let meta = self.shard_meta.as_ref()?;
        let exported_key = meta.get_property(ENCRYPTION_KEY)?;
        let mgr = self.engines.kv.get_master_key();
        Some(mgr.decrypt_encryption_key(exported_key.chunk()).unwrap())
    }
}

fn init_raft_state(
    raft_engine: &rfengine::RfEngine,
    peer_id: u64,
    region: &metapb::Region,
) -> Result<RaftState> {
    let mut rs = RaftState::default();
    if region.peers.is_empty() {
        return Ok(rs);
    }
    let rs_key = raft_state_key(region.get_region_epoch().get_version());
    let rs_val = raft_engine.get_state(peer_id, rs_key.chunk());
    if let Some(val) = rs_val {
        rs.unmarshal(&val);
    } else {
        // new split region.
        rs.last_index = RAFT_INIT_LOG_INDEX;
        rs.term = RAFT_INIT_LOG_TERM;
        rs.commit = RAFT_INIT_LOG_INDEX;
        rs.last_preprocessed_index = RAFT_INIT_LOG_INDEX;
        let mut wb = rfengine::WriteBatch::new();
        wb.set_state(peer_id, region.id, rs_key.chunk(), rs.marshal().chunk());
        raft_engine.write(wb)?;
    }
    Ok(rs)
}

fn init_apply_state(kv_engine: &kvengine::Engine, region: &metapb::Region) -> RaftApplyState {
    if region.get_peers().is_empty() {
        return RaftApplyState::new(0, 0);
    }
    if let Some(shard) = kv_engine.get_shard(region.get_id()) {
        let mut term_bin = shard.get_property(TERM_KEY).unwrap();
        let applied_index = shard.get_write_sequence();
        return RaftApplyState::new(applied_index, term_bin.get_u64_le());
    }
    RaftApplyState::new(RAFT_INIT_LOG_INDEX, RAFT_INIT_LOG_TERM)
}

fn init_truncated_state(
    raft_engine: &rfengine::RfEngine,
    peer_id: u64,
    region: &metapb::Region,
) -> RaftTruncatedState {
    match load_raft_truncated_state(raft_engine, peer_id) {
        Some(ts) if ts.truncated_index != rfengine::TRUNCATE_ALL_INDEX => ts,
        _ => {
            let mut ts = RaftTruncatedState::default();
            if !region.get_peers().is_empty() {
                ts.truncated_index = RAFT_INIT_LOG_INDEX;
                ts.truncated_index_term = RAFT_INIT_LOG_TERM;
            }
            ts
        }
    }
}

fn init_last_term(
    engines: &Engines,
    peer_id: u64,
    region: &metapb::Region,
    raft_state: RaftState,
) -> Result<u64> {
    let last_index = raft_state.last_index;
    if last_index == 0 {
        return Ok(0);
    } else if last_index == RAFT_INIT_LOG_INDEX {
        return Ok(RAFT_INIT_LOG_TERM);
    } else {
        assert!(last_index > RAFT_INIT_LOG_INDEX)
    }
    let term = engines.raft.get_term(peer_id, last_index);
    if let Some(term) = term {
        return Ok(term);
    }
    if let Some(shard) = engines.kv.get_shard(region.get_id()) {
        let term = shard.get_property(TERM_KEY).unwrap().get_u64_le();
        return Ok(term);
    }
    return Err(box_err!(
        "region {} at index {} doesn't exists, may lost data",
        region.get_id(),
        last_index
    ));
}

// When we bootstrap the region we must call this to initialize region local
// state first.
pub fn write_initial_raft_state(
    raft_wb: &mut rfengine::WriteBatch,
    peer_id: u64,
    region_id: u64,
    region_version: u64,
) {
    let raft_state = RaftState {
        last_index: RAFT_INIT_LOG_INDEX,
        vote: 0,
        term: RAFT_INIT_LOG_TERM,
        commit: RAFT_INIT_LOG_INDEX,
        last_preprocessed_index: RAFT_INIT_LOG_INDEX,
    };
    raft_wb.set_state(
        peer_id,
        region_id,
        &raft_state_key(region_version),
        &raft_state.marshal(),
    );
}

pub fn write_peer_state(
    raft_wb: &mut rfengine::WriteBatch,
    peer_id: u64,
    region: &metapb::Region,
    peer_state: PeerState,
    merge_state: Option<MergeState>,
) {
    let store_id = region
        .get_peers()
        .iter()
        .find(|p| p.id == peer_id)
        .map(|p| p.store_id)
        .unwrap_or(0);
    let tag = PeerTag::new(store_id, RegionIdVer::from_region(region));
    info!("{} write peer state", tag);
    let mut region_state = RegionLocalState::default();
    region_state.set_state(peer_state);
    region_state.set_region(region.clone());
    if let Some(merge_state) = merge_state {
        region_state.set_merge_state(merge_state);
    }
    let state_bin = region_state.write_to_bytes().unwrap();
    let epoch = region.get_region_epoch();
    let key = region_state_key(epoch.get_version());
    raft_wb.set_state(peer_id, region.get_id(), &key, &state_bin);
}

pub fn write_engine_meta(raft_wb: &mut rfengine::WriteBatch, peer_id: u64, meta: &ShardMeta) {
    info!("{} write engine meta, sequence: {}", meta.tag(), meta.seq);
    raft_wb.set_state(peer_id, meta.id, KV_ENGINE_META_KEY, &meta.marshal());
}

pub(crate) fn write_raft_state(
    raft_wb: &mut rfengine::WriteBatch,
    peer_id: u64,
    meta: &ShardMeta,
    raft_state: &RaftState,
) {
    // The meta's version is the latest region version, use it to persist raft
    // state.
    let key = raft_state_key(meta.ver);
    raft_wb.set_state(peer_id, meta.id, &key, &raft_state.marshal());
}

pub fn load_engine_meta(
    raft: &rfengine::RfEngine,
    store_id: u64,
    peer_id: u64,
) -> Option<ShardMeta> {
    let shard_meta_bin = raft.get_state(peer_id, KV_ENGINE_META_KEY)?;
    let mut change_set = kvenginepb::ChangeSet::default();
    change_set.merge_from_bytes(&shard_meta_bin).unwrap();
    Some(ShardMeta::new(store_id, &change_set))
}

pub fn encode_snap_data(region: &metapb::Region, change_set: &kvenginepb::ChangeSet) -> Bytes {
    let size1 = region.compute_size() as usize;
    let size2 = change_set.compute_size() as usize;
    let mut buf = BytesMut::with_capacity(4 + size1 + 4 + size2);
    buf.put_u32_le(size1 as u32);
    buf.extend_from_slice(&region.write_to_bytes().unwrap());
    buf.put_u32_le(size2 as u32);
    buf.extend_from_slice(&change_set.write_to_bytes().unwrap());
    buf.freeze()
}

pub fn decode_snap_data(data: &[u8]) -> Result<(metapb::Region, kvenginepb::ChangeSet)> {
    let mut offset = 0;
    let size1 = LittleEndian::read_u32(data) as usize;
    offset += 4;
    let mut region = metapb::Region::default();
    region.merge_from_bytes(&data[offset..(offset + size1)])?;
    offset += size1;
    let size2 = LittleEndian::read_u32(&data[offset..]) as usize;
    offset += 4;
    let mut change_set = kvenginepb::ChangeSet::default();
    change_set.merge_from_bytes(&data[offset..(offset + size2)])?;
    Ok((region, change_set))
}

pub fn load_last_peer_state(raft: &rfengine::RfEngine, peer_id: u64) -> Option<RegionLocalState> {
    raft.get_last_state_with_prefix(peer_id, REGION_META_KEY_PREFIX)
        .map(|v| {
            let mut state = RegionLocalState::default();
            state.merge_from_bytes(&v).unwrap();
            state
        })
}

pub fn load_raft_truncated_state(
    raft: &rfengine::RfEngine,
    peer_id: u64,
) -> Option<RaftTruncatedState> {
    raft.get_state(peer_id, RAFT_TRUNCATED_STATE_KEY).map(|v| {
        let mut state = RaftTruncatedState::default();
        state.unmarshal(&v);
        state
    })
}

pub fn load_raft_engine_meta(
    raft: &rfengine::RfEngine,
    peer_id: u64,
) -> Option<kvenginepb::ChangeSet> {
    raft.get_state(peer_id, rfengine::KV_ENGINE_META_KEY)
        .map(|engine_meta_val| {
            let mut cs = kvenginepb::ChangeSet::new();
            cs.merge_from_bytes(&engine_meta_val).unwrap();
            cs
        })
}

pub fn load_peer_raft_state(
    raft: &rfengine::RfEngine,
    peer_id: u64,
    region_version: u64,
) -> Option<RaftState> {
    let raft_state_key = rfengine::raft_state_key(region_version);
    let raft_state_val = raft.get_state(peer_id, &raft_state_key)?;
    let mut raft_state = RaftState::default();
    raft_state.unmarshal(raft_state_val.as_ref());
    Some(raft_state)
}

pub fn collect_prefix_regions(
    raft: &rfengine::RfEngine,
    prefix: &[u8],
) -> Option<Vec<(u64, u64, u64)>> {
    let region_peers = raft.get_region_peer_map();
    let mut prefix_peers = vec![];
    for (region_id, peer_id) in region_peers {
        if region_id == 0 {
            continue;
        }
        let engine_meta = match load_raft_engine_meta(raft, peer_id) {
            Some(cs) => cs,
            None => continue,
        };
        let snap = engine_meta.get_snapshot();
        let start = snap.get_outer_start();
        if start.starts_with(prefix) {
            prefix_peers.push((peer_id, region_id, engine_meta.shard_ver));
        }
    }
    Some(prefix_peers)
}

pub fn load_region_state(
    rf: &rfengine::RfEngine,
    peer_id: u64,
    ver: u64,
) -> Option<RegionLocalState> {
    let region_state_key = region_state_key(ver);
    let region_state_val = rf.get_state(peer_id, region_state_key.chunk())?;
    let mut region_local_state = RegionLocalState::new();
    region_local_state
        .merge_from_bytes(&region_state_val)
        .unwrap();
    Some(region_local_state)
}
