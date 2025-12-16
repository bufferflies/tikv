// Copyright 2016 TiKV Project Authors. Licensed under Apache-2.0.

// #[PerformanceCriticalPath]
use std::{
    cell::RefCell,
    error,
    ops::{Deref, DerefMut},
    sync::{
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
        mpsc::{self, Receiver, TryRecvError},
        Arc,
    },
    u64,
};

use engine_traits::{Engines, KvEngine, Mutable, Peekable, RaftEngine, RaftLogBatch, CF_RAFT};
use fail::fail_point;
use into_other::into_other;
use keys::{self, enc_end_key, enc_start_key};
use kvproto::{
    metapb::{self, Region},
    raft_serverpb::{
        MergeState, PeerState, RaftApplyState, RaftLocalState, RaftSnapshotData, RegionLocalState,
    },
};
use protobuf::Message;
use raft::{
    self,
    eraftpb::{self, ConfState, Entry, HardState, Snapshot},
    Error as RaftError, GetEntriesContext, RaftState, Ready, Storage, StorageError,
};
use tikv_util::{
    box_err, box_try, debug, defer, error, info, store::find_peer_by_id, time::Instant, warn,
    worker::Scheduler,
};

use super::{metrics::*, worker::RegionTask, SnapEntry, SnapKey, SnapManager};
use crate::{
    store::{
        async_io::{read::ReadTask, write::WriteTask},
        entry_storage::EntryStorage,
        fsm::GenSnapTask,
        peer::PersistSnapshotResult,
        util,
    },
    Error, Result,
};

// When we create a region peer, we should initialize its log term/index > 0,
// so that we can force the follower peer to sync the snapshot first.
pub const RAFT_INIT_LOG_TERM: u64 = 5;
pub const RAFT_INIT_LOG_INDEX: u64 = 5;
const MAX_SNAP_TRY_CNT: usize = 5;

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

#[derive(Debug)]
pub enum SnapState {
    Relax,
    Generating {
        canceled: Arc<AtomicBool>,
        index: Arc<AtomicU64>,
        receiver: Receiver<Snapshot>,
    },
    Applying(Arc<AtomicUsize>),
    ApplyAborted,
}

impl PartialEq for SnapState {
    fn eq(&self, other: &SnapState) -> bool {
        match (self, other) {
            (&SnapState::Relax, &SnapState::Relax)
            | (&SnapState::ApplyAborted, &SnapState::ApplyAborted)
            | (&SnapState::Generating { .. }, &SnapState::Generating { .. }) => true,
            (SnapState::Applying(b1), SnapState::Applying(b2)) => {
                b1.load(Ordering::Relaxed) == b2.load(Ordering::Relaxed)
            }
            _ => false,
        }
    }
}

pub fn storage_error<E>(error: E) -> raft::Error
where
    E: Into<Box<dyn error::Error + Send + Sync>>,
{
    raft::Error::Store(StorageError::Other(error.into()))
}

impl From<Error> for RaftError {
    fn from(err: Error) -> RaftError {
        storage_error(err)
    }
}

#[derive(PartialEq, Debug)]
pub struct HandleSnapshotResult {
    pub msgs: Vec<eraftpb::Message>,
    pub snap_region: metapb::Region,
    /// The regions whose range are overlapped with this region
    pub destroy_regions: Vec<Region>,
    /// The first index before applying the snapshot.
    pub last_first_index: u64,
    pub for_witness: bool,
}

#[derive(PartialEq, Debug)]
pub enum HandleReadyResult {
    SendIoTask,
    Snapshot(Box<HandleSnapshotResult>), // use boxing to reduce total size of the enum
    NoIoTask,
}

fn init_raft_state<EK: KvEngine, ER: RaftEngine>(
    engines: &Engines<EK, ER>,
    region: &Region,
) -> Result<RaftLocalState> {
    if let Some(state) = engines.raft.get_raft_state(region.get_id())? {
        return Ok(state);
    }

    let mut raft_state = RaftLocalState::default();
    if util::is_region_initialized(region) {
        // new split region
        raft_state.last_index = RAFT_INIT_LOG_INDEX;
        raft_state.mut_hard_state().set_term(RAFT_INIT_LOG_TERM);
        raft_state.mut_hard_state().set_commit(RAFT_INIT_LOG_INDEX);
        engines.raft.put_raft_state(region.get_id(), &raft_state)?;
    }
    Ok(raft_state)
}

fn init_apply_state<EK: KvEngine, ER: RaftEngine>(
    engines: &Engines<EK, ER>,
    region: &Region,
) -> Result<RaftApplyState> {
    Ok(
        match engines
            .kv
            .get_msg_cf(CF_RAFT, &keys::apply_state_key(region.get_id()))?
        {
            Some(s) => s,
            None => {
                let mut apply_state = RaftApplyState::default();
                if util::is_region_initialized(region) {
                    apply_state.set_applied_index(RAFT_INIT_LOG_INDEX);
                    let state = apply_state.mut_truncated_state();
                    state.set_index(RAFT_INIT_LOG_INDEX);
                    state.set_term(RAFT_INIT_LOG_TERM);
                }
                apply_state
            }
        },
    )
}

pub struct PeerStorage<EK, ER>
where
    EK: KvEngine,
{
    pub engines: Engines<EK, ER>,

    peer_id: u64,
    peer: Option<metapb::Peer>, // when uninitialized the peer info is unknown.
    region: metapb::Region,

    snap_state: RefCell<SnapState>,
    gen_snap_task: RefCell<Option<GenSnapTask>>,
    region_scheduler: Scheduler<RegionTask<EK::Snapshot>>,
    snap_tried_cnt: RefCell<usize>,

    entry_storage: EntryStorage<EK, ER>,

    pub tag: String,
}

impl<EK: KvEngine, ER: RaftEngine> Deref for PeerStorage<EK, ER> {
    type Target = EntryStorage<EK, ER>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.entry_storage
    }
}

impl<EK: KvEngine, ER: RaftEngine> DerefMut for PeerStorage<EK, ER> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.entry_storage
    }
}

impl<EK, ER> Storage for PeerStorage<EK, ER>
where
    EK: KvEngine,
    ER: RaftEngine,
{
    fn initial_state(&self) -> raft::Result<RaftState> {
        self.initial_state()
    }

    fn entries(
        &self,
        low: u64,
        high: u64,
        max_size: impl Into<Option<u64>>,
        context: GetEntriesContext,
    ) -> raft::Result<Vec<Entry>> {
        let max_size = max_size.into();
        self.entry_storage
            .entries(low, high, max_size.unwrap_or(u64::MAX), context)
    }

    fn term(&self, idx: u64) -> raft::Result<u64> {
        self.entry_storage.term(idx)
    }

    fn first_index(&self) -> raft::Result<u64> {
        Ok(self.entry_storage.first_index())
    }

    fn last_index(&self) -> raft::Result<u64> {
        Ok(self.entry_storage.last_index())
    }

    fn snapshot(&self, request_index: u64, to: u64) -> raft::Result<Snapshot> {
        self.snapshot(request_index, to)
    }
}

impl<EK, ER> PeerStorage<EK, ER>
where
    EK: KvEngine,
    ER: RaftEngine,
{
    pub fn new(
        engines: Engines<EK, ER>,
        region: &metapb::Region,
        region_scheduler: Scheduler<RegionTask<EK::Snapshot>>,
        raftlog_fetch_scheduler: Scheduler<ReadTask<EK>>,
        peer_id: u64,
        tag: String,
    ) -> Result<PeerStorage<EK, ER>> {
        debug!(
            "creating storage on specified path";
            "region_id" => region.get_id(),
            "peer_id" => peer_id,
            "path" => ?engines.kv.path(),
        );
        let raft_state = init_raft_state(&engines, region)?;
        let apply_state = init_apply_state(&engines, region)?;

        let entry_storage = EntryStorage::new(
            peer_id,
            engines.raft.clone(),
            raft_state,
            apply_state,
            region,
            raftlog_fetch_scheduler,
        )?;

        Ok(PeerStorage {
            engines,
            peer_id,
            peer: find_peer_by_id(region, peer_id).cloned(),
            region: region.clone(),
            snap_state: RefCell::new(SnapState::Relax),
            gen_snap_task: RefCell::new(None),
            region_scheduler,
            snap_tried_cnt: RefCell::new(0),
            tag,
            entry_storage,
        })
    }

    pub fn is_initialized(&self) -> bool {
        util::is_region_initialized(self.region())
    }

    pub fn initial_state(&self) -> raft::Result<RaftState> {
        let hard_state = self.raft_state().get_hard_state().clone();
        if hard_state == HardState::default() {
            assert!(
                !self.is_initialized(),
                "peer for region {:?} is initialized but local state {:?} has empty hard \
                 state",
                self.region,
                self.raft_state()
            );

            return Ok(RaftState::new(hard_state, ConfState::default()));
        }
        Ok(RaftState::new(
            hard_state,
            util::conf_state_from_region(self.region()),
        ))
    }

    #[inline]
    pub fn region(&self) -> &metapb::Region {
        &self.region
    }

    #[inline]
    pub fn set_region(&mut self, region: metapb::Region) {
        self.peer = find_peer_by_id(&region, self.peer_id).cloned();
        self.region = region;
    }

    #[inline]
    pub fn raw_snapshot(&self) -> EK::Snapshot {
        self.engines.kv.snapshot()
    }

    #[inline]
    pub fn save_snapshot_raft_state_to(
        &self,
        snapshot_index: u64,
        kv_wb: &mut impl Mutable,
    ) -> Result<()> {
        let mut snapshot_raft_state = self.raft_state().clone();
        snapshot_raft_state
            .mut_hard_state()
            .set_commit(snapshot_index);
        snapshot_raft_state.set_last_index(snapshot_index);

        kv_wb.put_msg_cf(
            CF_RAFT,
            &keys::snapshot_raft_state_key(self.region.get_id()),
            &snapshot_raft_state,
        )?;
        Ok(())
    }

    #[inline]
    pub fn save_apply_state_to(&self, kv_wb: &mut impl Mutable) -> Result<()> {
        kv_wb.put_msg_cf(
            CF_RAFT,
            &keys::apply_state_key(self.region.get_id()),
            self.apply_state(),
        )?;
        Ok(())
    }

    fn validate_snap(&self, snap: &Snapshot, request_index: u64) -> bool {
        let idx = snap.get_metadata().get_index();
        if idx < self.truncated_index() || idx < request_index {
            // stale snapshot, should generate again.
            info!(
                "snapshot is stale, generate again";
                "region_id" => self.region.get_id(),
                "peer_id" => self.peer_id,
                "snap_index" => idx,
                "truncated_index" => self.truncated_index(),
                "request_index" => request_index,
            );
            STORE_SNAPSHOT_VALIDATION_FAILURE_COUNTER.stale.inc();
            return false;
        }

        let mut snap_data = RaftSnapshotData::default();
        if let Err(e) = snap_data.merge_from_bytes(snap.get_data()) {
            error!(
                "failed to decode snapshot, it may be corrupted";
                "region_id" => self.region.get_id(),
                "peer_id" => self.peer_id,
                "err" => ?e,
            );
            STORE_SNAPSHOT_VALIDATION_FAILURE_COUNTER.decode.inc();
            return false;
        }
        let snap_epoch = snap_data.get_region().get_region_epoch();
        let latest_epoch = self.region().get_region_epoch();
        if snap_epoch.get_conf_ver() < latest_epoch.get_conf_ver() {
            info!(
                "snapshot epoch is stale";
                "region_id" => self.region.get_id(),
                "peer_id" => self.peer_id,
                "snap_epoch" => ?snap_epoch,
                "latest_epoch" => ?latest_epoch,
            );
            STORE_SNAPSHOT_VALIDATION_FAILURE_COUNTER.epoch.inc();
            return false;
        }

        true
    }

    /// Gets a snapshot. Returns `SnapshotTemporarilyUnavailable` if there is no
    /// available snapshot.
    pub fn snapshot(&self, request_index: u64, to: u64) -> raft::Result<Snapshot> {
        if self.peer.as_ref().unwrap().is_witness {
            // witness could be the leader for a while, do not generate snapshot now
            return Err(raft::Error::Store(
                raft::StorageError::SnapshotTemporarilyUnavailable,
            ));
        }

        if find_peer_by_id(&self.region, to).map_or(false, |p| p.is_witness) {
            // generate an empty snapshot for witness directly
            return Ok(util::new_empty_snapshot(
                self.region.clone(),
                self.applied_index(),
                self.applied_term(),
                true, // for witness
            ));
        }

        let mut snap_state = self.snap_state.borrow_mut();
        let mut tried_cnt = self.snap_tried_cnt.borrow_mut();

        let mut tried = false;
        let mut last_canceled = false;
        if let SnapState::Generating {
            canceled, receiver, ..
        } = &*snap_state
        {
            tried = true;
            last_canceled = canceled.load(Ordering::SeqCst);
            match receiver.try_recv() {
                Err(TryRecvError::Empty) => {
                    return Err(raft::Error::Store(
                        raft::StorageError::SnapshotTemporarilyUnavailable,
                    ));
                }
                Ok(s) if !last_canceled => {
                    *snap_state = SnapState::Relax;
                    *tried_cnt = 0;
                    if self.validate_snap(&s, request_index) {
                        return Ok(s);
                    }
                }
                Err(TryRecvError::Disconnected) | Ok(_) => {
                    *snap_state = SnapState::Relax;
                    warn!(
                        "failed to try generating snapshot";
                        "region_id" => self.region.get_id(),
                        "peer_id" => self.peer_id,
                        "times" => *tried_cnt,
                        "request_peer" => to,
                    );
                }
            }
        }

        if SnapState::Relax != *snap_state {
            panic!("{} unexpected state: {:?}", self.tag, *snap_state);
        }

        if *tried_cnt >= MAX_SNAP_TRY_CNT {
            let cnt = *tried_cnt;
            *tried_cnt = 0;
            return Err(raft::Error::Store(box_err!(
                "failed to get snapshot after {} times",
                cnt
            )));
        }
        if !tried || !last_canceled {
            *tried_cnt += 1;
        }

        info!(
            "requesting snapshot";
            "region_id" => self.region.get_id(),
            "peer_id" => self.peer_id,
            "request_index" => request_index,
            "request_peer" => to,
        );

        let (sender, receiver) = mpsc::sync_channel(1);
        let canceled = Arc::new(AtomicBool::new(false));
        let index = Arc::new(AtomicU64::new(0));
        *snap_state = SnapState::Generating {
            canceled: canceled.clone(),
            index: index.clone(),
            receiver,
        };

        let store_id = self
            .region()
            .get_peers()
            .iter()
            .find(|p| p.id == to)
            .map(|p| p.store_id)
            .unwrap_or(0);
        let task = GenSnapTask::new(self.region.get_id(), index, canceled, sender, store_id);

        let mut gen_snap_task = self.gen_snap_task.borrow_mut();
        assert!(gen_snap_task.is_none());
        *gen_snap_task = Some(task);
        Err(raft::Error::Store(
            raft::StorageError::SnapshotTemporarilyUnavailable,
        ))
    }

    pub fn has_gen_snap_task(&self) -> bool {
        self.gen_snap_task.borrow().is_some()
    }

    pub fn mut_gen_snap_task(&mut self) -> &mut Option<GenSnapTask> {
        self.gen_snap_task.get_mut()
    }

    pub fn take_gen_snap_task(&mut self) -> Option<GenSnapTask> {
        self.gen_snap_task.get_mut().take()
    }

    pub fn on_compact_raftlog(&mut self, idx: u64) {
        self.entry_storage.compact_entry_cache(idx);
        self.cancel_generating_snap(Some(idx));
    }

    // Apply the peer with given snapshot.
    pub fn apply_snapshot(
        &mut self,
        snap: &Snapshot,
        task: &mut WriteTask<EK, ER>,
        destroy_regions: &[metapb::Region],
    ) -> Result<(metapb::Region, bool)> {
        info!(
            "begin to apply snapshot";
            "region_id" => self.region.get_id(),
            "peer_id" => self.peer_id,
        );

        let mut snap_data = RaftSnapshotData::default();
        snap_data.merge_from_bytes(snap.get_data())?;

        let for_witness = snap_data.get_meta().get_for_witness();

        let region_id = self.get_region_id();
        let region = snap_data.take_region();
        if region.get_id() != region_id {
            return Err(box_err!(
                "mismatch region id {} != {}",
                region_id,
                region.get_id()
            ));
        }

        if task.raft_wb.is_none() {
            task.raft_wb = Some(self.engines.raft.log_batch(64));
        }
        let raft_wb = task.raft_wb.as_mut().unwrap();
        let kv_wb = task.extra_write.ensure_v1(|| self.engines.kv.write_batch());

        if self.is_initialized() {
            // we can only delete the old data when the peer is initialized.
            let first_index = self.entry_storage.first_index();
            // It's possible that logs between `last_compacted_idx` and `first_index` are
            // being deleted in raftlog_gc worker. But it's OK as:
            // - If the peer accepts a new snapshot, it must start with an index larger than
            //   this `first_index`;
            // - If the peer accepts new entries after this snapshot or new snapshot, it
            //   must start with the new applied index, which is larger than `first_index`.
            // So new logs won't be deleted by on going raftlog_gc task accidentally.
            // It's possible that there will be some logs between `last_compacted_idx` and
            // `first_index` are not deleted. So a cleanup task for the range should be
            // triggered after applying the snapshot.
            self.clear_meta(first_index, kv_wb, raft_wb)?;
        }
        // Write its source peers' `RegionLocalState` together with itself for atomicity
        for r in destroy_regions {
            write_peer_state(kv_wb, r, PeerState::Tombstone, None)?;
        }

        // Witness snapshot is applied atomically as no async applying operation to
        // region worker, so no need to set the peer state to `Applying`
        let state = if for_witness {
            PeerState::Normal
        } else {
            PeerState::Applying
        };
        write_peer_state(kv_wb, &region, state, None)?;

        let snap_index = snap.get_metadata().get_index();
        let snap_term = snap.get_metadata().get_term();

        self.raft_state_mut().set_last_index(snap_index);
        self.set_last_term(snap_term);
        self.apply_state_mut().set_applied_index(snap_index);
        self.set_applied_term(snap_term);

        // The snapshot only contains log which index > applied index, so
        // here the truncate state's (index, term) is in snapshot metadata.
        self.apply_state_mut()
            .mut_truncated_state()
            .set_index(snap_index);
        self.apply_state_mut()
            .mut_truncated_state()
            .set_term(snap_term);

        // `region` will be updated after persisting.
        // Although there is an interval that other metadata are updated while `region`
        // is not after handing snapshot from ready, at the time of writing, it's no
        // problem for now.
        // The reason why the update of `region` is delayed is that we expect `region`
        // stays consistent with the one in `StoreMeta::regions` which should be updated
        // after persisting due to atomic snapshot and peer create process. So if we can
        // fix these issues in future(maybe not?), the `region` and `StoreMeta::regions`
        // can updated here immediately.

        info!(
            "apply snapshot with state ok";
            "region_id" => self.region.get_id(),
            "peer_id" => self.peer_id,
            "region" => ?region,
            "state" => ?self.apply_state(),
        );

        Ok((region, for_witness))
    }

    /// Delete all meta belong to the region. Results are stored in `wb`.
    pub fn clear_meta(
        &mut self,
        first_index: u64,
        kv_wb: &mut EK::WriteBatch,
        raft_wb: &mut ER::LogBatch,
    ) -> Result<()> {
        let region_id = self.get_region_id();
        clear_meta(
            &self.engines,
            kv_wb,
            raft_wb,
            region_id,
            first_index,
            self.raft_state(),
        )?;
        self.entry_storage.clear();
        Ok(())
    }

    /// Delete all data belong to the region.
    /// If return Err, data may get partial deleted.
    pub fn clear_data(&self) -> Result<()> {
        let (start_key, end_key) = (enc_start_key(self.region()), enc_end_key(self.region()));
        let region_id = self.get_region_id();
        box_try!(
            self.region_scheduler
                .schedule(RegionTask::destroy(region_id, start_key, end_key))
        );
        Ok(())
    }

    /// Delete all data that is not covered by `new_region`.
    fn clear_extra_data(
        &self,
        old_region: &metapb::Region,
        new_region: &metapb::Region,
    ) -> Result<()> {
        let (old_start_key, old_end_key) = (enc_start_key(old_region), enc_end_key(old_region));
        let (new_start_key, new_end_key) = (enc_start_key(new_region), enc_end_key(new_region));
        if old_start_key < new_start_key {
            box_try!(self.region_scheduler.schedule(RegionTask::destroy(
                old_region.get_id(),
                old_start_key,
                new_start_key
            )));
        }
        if new_end_key < old_end_key {
            box_try!(self.region_scheduler.schedule(RegionTask::destroy(
                old_region.get_id(),
                new_end_key,
                old_end_key
            )));
        }
        Ok(())
    }

    /// Delete all extra split data from the `start_key` to `end_key`.
    pub fn clear_extra_split_data(&self, start_key: Vec<u8>, end_key: Vec<u8>) -> Result<()> {
        box_try!(self.region_scheduler.schedule(RegionTask::destroy(
            self.get_region_id(),
            start_key,
            end_key
        )));
        Ok(())
    }

    pub fn raft_engine(&self) -> &ER {
        self.entry_storage.raft_engine()
    }

    /// Check whether the storage has finished applying snapshot.
    #[inline]
    pub fn is_applying_snapshot(&self) -> bool {
        matches!(*self.snap_state.borrow(), SnapState::Applying(_))
    }

    #[inline]
    pub fn is_generating_snapshot(&self) -> bool {
        fail_point!("is_generating_snapshot", |_| { true });
        matches!(*self.snap_state.borrow(), SnapState::Generating { .. })
    }

    /// Check if the storage is applying a snapshot.
    #[inline]
    pub fn check_applying_snap(&mut self) -> CheckApplyingSnapStatus {
        let mut res = CheckApplyingSnapStatus::Idle;
        let new_state = match *self.snap_state.borrow() {
            SnapState::Applying(ref status) => {
                let s = status.load(Ordering::Relaxed);
                if s == JOB_STATUS_FINISHED {
                    res = CheckApplyingSnapStatus::Success;
                    SnapState::Relax
                } else if s == JOB_STATUS_CANCELLED {
                    SnapState::ApplyAborted
                } else if s == JOB_STATUS_FAILED {
                    // TODO: cleanup region and treat it as tombstone.
                    panic!("{} applying snapshot failed", self.tag,);
                } else {
                    return CheckApplyingSnapStatus::Applying;
                }
            }
            _ => return res,
        };
        *self.snap_state.borrow_mut() = new_state;
        res
    }

    /// Cancel applying snapshot, return true if the job can be considered not
    /// be run again.
    pub fn cancel_applying_snap(&mut self) -> bool {
        let is_canceled = match *self.snap_state.borrow() {
            SnapState::Applying(ref status) => {
                if status
                    .compare_exchange(
                        JOB_STATUS_PENDING,
                        JOB_STATUS_CANCELLING,
                        Ordering::SeqCst,
                        Ordering::SeqCst,
                    )
                    .is_ok()
                {
                    true
                } else if status
                    .compare_exchange(
                        JOB_STATUS_RUNNING,
                        JOB_STATUS_CANCELLING,
                        Ordering::SeqCst,
                        Ordering::SeqCst,
                    )
                    .is_ok()
                {
                    return false;
                } else {
                    false
                }
            }
            _ => return false,
        };
        if is_canceled {
            *self.snap_state.borrow_mut() = SnapState::ApplyAborted;
            return true;
        }
        // now status can only be JOB_STATUS_CANCELLING, JOB_STATUS_CANCELLED,
        // JOB_STATUS_FAILED and JOB_STATUS_FINISHED.
        self.check_applying_snap() != CheckApplyingSnapStatus::Applying
    }

    /// Cancel generating snapshot.
    pub fn cancel_generating_snap(&mut self, compact_to: Option<u64>) {
        let snap_state = self.snap_state.borrow();
        if let SnapState::Generating {
            ref canceled,
            ref index,
            ..
        } = *snap_state
        {
            if !canceled.load(Ordering::SeqCst) {
                if let Some(idx) = compact_to {
                    let snap_index = index.load(Ordering::SeqCst);
                    if snap_index == 0 || idx <= snap_index + 1 {
                        return;
                    }
                }
                canceled.store(true, Ordering::SeqCst);
            }
        }
    }

    #[inline]
    pub fn set_snap_state(&mut self, state: SnapState) {
        *self.snap_state.borrow_mut() = state
    }

    #[inline]
    pub fn is_snap_state(&self, state: SnapState) -> bool {
        *self.snap_state.borrow() == state
    }

    pub fn get_region_id(&self) -> u64 {
        self.region().get_id()
    }

    pub fn schedule_applying_snapshot(&mut self) {
        let status = Arc::new(AtomicUsize::new(JOB_STATUS_PENDING));
        self.set_snap_state(SnapState::Applying(Arc::clone(&status)));
        let task = RegionTask::Apply {
            region_id: self.get_region_id(),
            status,
            peer_id: self.peer_id,
        };

        // Don't schedule the snapshot to region worker.
        fail_point!("skip_schedule_applying_snapshot", |_| {});

        // TODO: gracefully remove region instead.
        if let Err(e) = self.region_scheduler.schedule(task) {
            info!(
                "failed to to schedule apply job, are we shutting down?";
                "region_id" => self.region.get_id(),
                "peer_id" => self.peer_id,
                "err" => ?e,
            );
        }
    }

    /// Handle raft ready then generate `HandleReadyResult` and `WriteTask`.
    ///
    /// It's caller's duty to write `WriteTask` explicitly to disk.
    pub fn handle_raft_ready(
        &mut self,
        ready: &mut Ready,
        destroy_regions: Vec<metapb::Region>,
    ) -> Result<(HandleReadyResult, WriteTask<EK, ER>)> {
        let region_id = self.get_region_id();
        let prev_raft_state = self.raft_state().clone();

        let mut write_task = WriteTask::new(region_id, self.peer_id, ready.number());

        let mut res = if ready.snapshot().is_empty() {
            HandleReadyResult::SendIoTask
        } else {
            fail_point!("raft_before_apply_snap");
            let last_first_index = self.first_index().unwrap();
            let (snap_region, for_witness) =
                self.apply_snapshot(ready.snapshot(), &mut write_task, &destroy_regions)?;

            let res = HandleReadyResult::Snapshot(Box::new(HandleSnapshotResult {
                msgs: ready.take_persisted_messages(),
                snap_region,
                destroy_regions,
                last_first_index,
                for_witness,
            }));
            fail_point!("raft_after_apply_snap");
            res
        };

        if !ready.entries().is_empty() {
            self.append(ready.take_entries(), &mut write_task);
        }

        // Last index is 0 means the peer is created from raft message
        // and has not applied snapshot yet, so skip persistent hard state.
        if self.raft_state().get_last_index() > 0 {
            if let Some(hs) = ready.hs() {
                self.raft_state_mut().set_hard_state(hs.clone());
            }
        }

        // Save raft state if it has changed or there is a snapshot.
        if prev_raft_state != *self.raft_state() || !ready.snapshot().is_empty() {
            write_task.raft_state = Some(self.raft_state().clone());
        }

        if !ready.snapshot().is_empty() {
            // In case of restart happens when we just write region state to Applying,
            // but not write raft_local_state to raft db in time.
            // We write raft state to kv db, with last index set to snap index,
            // in case of recv raft log after snapshot.
            self.save_snapshot_raft_state_to(
                ready.snapshot().get_metadata().get_index(),
                write_task.extra_write.v1_mut().unwrap(),
            )?;
            self.save_apply_state_to(write_task.extra_write.v1_mut().unwrap())?;
        }

        if !write_task.has_data() {
            res = HandleReadyResult::NoIoTask;
        }

        Ok((res, write_task))
    }

    pub fn persist_snapshot(&mut self, res: &PersistSnapshotResult) {
        // cleanup data before scheduling apply task
        if self.is_initialized() {
            if let Err(e) = self.clear_extra_data(self.region(), &res.region) {
                // No need panic here, when applying snapshot, the deletion will be tried
                // again. But if the region range changes, like [a, c) -> [a, b) and [b, c),
                // [b, c) will be kept in rocksdb until a covered snapshot is applied or
                // store is restarted.
                error!(?e;
                    "failed to cleanup data, may leave some dirty data";
                    "region_id" => self.get_region_id(),
                    "peer_id" => self.peer_id,
                );
            }
        }

        // Note that the correctness depends on the fact that these source regions MUST
        // NOT serve read request otherwise a corrupt data may be returned.
        // For now, it is ensured by
        // - After `PrepareMerge` log is committed, the source region leader's lease
        //   will be suspected immediately which makes the local reader not serve read
        //   request.
        // - No read request can be responded in peer fsm during merging. These
        //   conditions are used to prevent reading **stale** data in the past. At
        //   present, they are also used to prevent reading **corrupt** data.
        for r in &res.destroy_regions {
            if let Err(e) = self.clear_extra_data(r, &res.region) {
                error!(?e;
                    "failed to cleanup data, may leave some dirty data";
                    "region_id" => r.get_id(),
                );
            }
        }

        if !res.for_witness {
            self.schedule_applying_snapshot();
        } else {
            // Bypass apply snapshot process for witness as the snapshot is empty, so mark
            // status as finished directly here
            let status = Arc::new(AtomicUsize::new(JOB_STATUS_FINISHED));
            self.set_snap_state(SnapState::Applying(Arc::clone(&status)));
        }

        // The `region` is updated after persisting in order to stay consistent with the
        // one in `StoreMeta::regions` (will be updated soon).
        // See comments in `apply_snapshot` for more details.
        self.set_region(res.region.clone());
    }
}

/// Delete all meta belong to the region. Results are stored in `wb`.
pub fn clear_meta<EK, ER>(
    engines: &Engines<EK, ER>,
    kv_wb: &mut EK::WriteBatch,
    raft_wb: &mut ER::LogBatch,
    region_id: u64,
    first_index: u64,
    raft_state: &RaftLocalState,
) -> Result<()>
where
    EK: KvEngine,
    ER: RaftEngine,
{
    let t = Instant::now();
    box_try!(kv_wb.delete_cf(CF_RAFT, &keys::region_state_key(region_id)));
    box_try!(kv_wb.delete_cf(CF_RAFT, &keys::apply_state_key(region_id)));
    box_try!(
        engines
            .raft
            .clean(region_id, first_index, raft_state, raft_wb)
    );

    info!(
        "finish clear peer meta";
        "region_id" => region_id,
        "meta_key" => 1,
        "apply_key" => 1,
        "raft_key" => 1,
        "takes" => ?t.saturating_elapsed(),
    );
    Ok(())
}

pub fn do_snapshot<E>(
    mgr: SnapManager,
    engine: &E,
    kv_snap: E::Snapshot,
    region_id: u64,
    last_applied_term: u64,
    last_applied_state: RaftApplyState,
    for_balance: bool,
    allow_multi_files_snapshot: bool,
) -> raft::Result<Snapshot>
where
    E: KvEngine,
{
    debug!(
        "begin to generate a snapshot";
        "region_id" => region_id,
    );

    let apply_state: RaftApplyState = kv_snap
        .get_msg_cf(CF_RAFT, &keys::apply_state_key(region_id))
        .map_err(into_other::<_, raft::Error>)
        .and_then(|v| {
            v.ok_or_else(|| {
                storage_error(format!("could not load raft state of region {}", region_id))
            })
        })?;
    assert_eq!(apply_state, last_applied_state);

    let key = SnapKey::new(
        region_id,
        last_applied_term,
        apply_state.get_applied_index(),
    );
    mgr.register(key.clone(), SnapEntry::Generating);
    defer!(mgr.deregister(&key, &SnapEntry::Generating));

    let region_state: RegionLocalState = kv_snap
        .get_msg_cf(CF_RAFT, &keys::region_state_key(key.region_id))
        .map_err(into_other::<_, raft::Error>)
        .and_then(|v| {
            v.ok_or_else(|| {
                storage_error(format!("region {} could not find region info", region_id))
            })
        })?;
    if region_state.get_state() != PeerState::Normal {
        return Err(storage_error(format!(
            "snap job for {} seems stale, skip.",
            region_id
        )));
    }

    let mut snapshot = Snapshot::default();
    // Set snapshot metadata.
    snapshot.mut_metadata().set_index(key.idx);
    snapshot.mut_metadata().set_term(key.term);
    snapshot
        .mut_metadata()
        .set_conf_state(util::conf_state_from_region(region_state.get_region()));
    // Set snapshot data.
    let mut s = mgr.get_snapshot_for_building(&key)?;
    let snap_data = s.build(
        engine,
        &kv_snap,
        region_state.get_region(),
        allow_multi_files_snapshot,
        for_balance,
    )?;
    snapshot.set_data(snap_data.write_to_bytes()?.into());

    Ok(snapshot)
}

// When we bootstrap the region we must call this to initialize region local
// state first.
pub fn write_initial_raft_state<W: RaftLogBatch>(raft_wb: &mut W, region_id: u64) -> Result<()> {
    let mut raft_state = RaftLocalState {
        last_index: RAFT_INIT_LOG_INDEX,
        ..Default::default()
    };
    raft_state.mut_hard_state().set_term(RAFT_INIT_LOG_TERM);
    raft_state.mut_hard_state().set_commit(RAFT_INIT_LOG_INDEX);
    raft_wb.put_raft_state(region_id, &raft_state)?;
    Ok(())
}

// When we bootstrap the region or handling split new region, we must
// call this to initialize region apply state first.
pub fn write_initial_apply_state<T: Mutable>(kv_wb: &mut T, region_id: u64) -> Result<()> {
    let mut apply_state = RaftApplyState::default();
    apply_state.set_applied_index(RAFT_INIT_LOG_INDEX);
    apply_state
        .mut_truncated_state()
        .set_index(RAFT_INIT_LOG_INDEX);
    apply_state
        .mut_truncated_state()
        .set_term(RAFT_INIT_LOG_TERM);

    kv_wb.put_msg_cf(CF_RAFT, &keys::apply_state_key(region_id), &apply_state)?;
    Ok(())
}

pub fn write_peer_state<T: Mutable>(
    kv_wb: &mut T,
    region: &metapb::Region,
    state: PeerState,
    merge_state: Option<MergeState>,
) -> Result<()> {
    let region_id = region.get_id();
    let mut region_state = RegionLocalState::default();
    region_state.set_state(state);
    region_state.set_region(region.clone());
    if let Some(state) = merge_state {
        region_state.set_merge_state(state);
    }

    debug!(
        "writing merge state";
        "region_id" => region_id,
        "state" => ?region_state,
    );
    kv_wb.put_msg_cf(CF_RAFT, &keys::region_state_key(region_id), &region_state)?;
    Ok(())
}
