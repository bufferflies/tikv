// Copyright 2016 TiKV Project Authors. Licensed under Apache-2.0.

// #[PerformanceCriticalPath]
use std::{
    cell::RefCell,
    collections::VecDeque,
    fmt, mem,
    sync::{
        atomic::{AtomicUsize, Ordering},
        mpsc::SyncSender,
        Arc, Mutex,
    },
    time::Instant,
    u64, usize,
};

use bitflags::bitflags;
use bytes::Bytes;
use collections::{HashMap, HashSet};
use crossbeam::atomic::AtomicCell;
use engine_traits::{KvEngine, RaftEngine};
use fail::fail_point;
use getset::{Getters, MutGetters};
use kvproto::{
    kvrpcpb::{ExtraOp as TxnExtraOp, LockInfo},
    metapb::{self, PeerRole},
    pdpb,
    raft_cmdpb::{
        self, AdminCmdType, CmdType, CommitMergeRequest, RaftCmdRequest, TransferLeaderRequest,
    },
    raft_serverpb::{ExtraMessage, ExtraMessageType, MergeState, RaftMessage},
    replication_modepb::{DrAutoSyncState, ReplicationMode},
};
use pd_client::{BucketStat, INVALID_ID};
use raft::{self, eraftpb, ProgressState, RawNode, StateRole, INVALID_INDEX};
use tikv_util::{
    box_err, debug, error, info, sys::disk::DiskUsage, time::Instant as TiInstant, warn,
};
use time::{Duration as TimeDuration, Timespec};
use txn_types::WriteBatchFlags;
use uuid::Uuid;

use super::{
    peer_storage::PeerStorage,
    read_queue::{ReadIndexQueue, ReadIndexRequest},
    transport::Transport,
    util::{Lease, LeaseState},
    LocalReadContext,
};
use crate::{
    coprocessor::{CoprocessorHost, RegionChangeEvent, RegionChangeReason},
    router::RaftStoreRouter,
    store::{
        fsm::{
            apply::{self, CatchUpLogs},
            store::{PollContext, RaftRouter},
            Proposal,
        },
        msg::{PeerMsg, SignificantMsg, StoreMsg},
        util::RegionReadProgress,
        worker::{RaftlogGcTask, ReadDelegate, ReadExecutor, ReadProgress},
        Callback, Config, GlobalReplicationState, ReadIndexContext, TxnExt,
    },
    Error, Result,
};

/// The returned states of the peer after checking whether it is stale
#[derive(Debug, PartialEq)]
pub enum StaleState {}

#[derive(Debug)]
pub struct ProposalQueue<C> {
    queue: VecDeque<Proposal<C>>,
}

bitflags! {
    // TODO: maybe declare it as protobuf struct is better.
    /// A bitmap contains some useful flags when dealing with `eraftpb::Entry`.
    pub struct ProposalContext: u8 {
        const SYNC_LOG       = 0b0000_0001;
        const SPLIT          = 0b0000_0010;
        const PREPARE_MERGE  = 0b0000_0100;
        const COMMIT_MERGE   = 0b0000_1000;
    }
}

impl ProposalContext {
    /// Converts itself to a vector.
    pub fn to_vec(self) -> Vec<u8> {
        if self.is_empty() {
            return vec![];
        }
        let ctx = self.bits();
        vec![ctx]
    }

    /// Initializes a `ProposalContext` from a byte slice.
    pub fn from_bytes(ctx: &[u8]) -> ProposalContext {
        if ctx.is_empty() {
            ProposalContext::empty()
        } else if ctx.len() == 1 {
            ProposalContext::from_bits_truncate(ctx[0])
        } else {
            panic!("invalid ProposalContext {:?}", ctx);
        }
    }
}

/// `ConsistencyState` is used for consistency check.
pub struct ConsistencyState {
    pub last_check_time: Instant,
    // (computed_result_or_to_be_verified, index, hash)
    pub index: u64,
    pub context: Vec<u8>,
    pub hash: Vec<u8>,
}

/// Statistics about raft peer.
#[derive(Default, Clone)]
pub struct PeerStat {
    pub written_bytes: u64,
    pub written_keys: u64,
}

#[derive(Default, Debug, Clone, Copy)]
pub struct CheckTickResult {
    leader: bool,
    up_to_date: bool,
    reason: &'static str,
}

#[derive(PartialEq, Debug)]
pub struct ApplySnapshotContext {
    /// The number of ready which has a snapshot.
    pub ready_number: u64,
    /// Whether this snapshot is scheduled.
    pub scheduled: bool,
    /// The message should be sent after snapshot is applied.
    pub msgs: Vec<eraftpb::Message>,
    pub persist_res: Option<PersistSnapshotResult>,
}

#[derive(PartialEq, Debug)]
pub struct PersistSnapshotResult {
    /// prev_region is the region before snapshot applied.
    pub prev_region: metapb::Region,
    pub region: metapb::Region,
    pub destroy_regions: Vec<metapb::Region>,
    pub for_witness: bool,
}

#[derive(Debug)]
pub struct UnpersistedReady {
    /// Number of ready.
    pub number: u64,
    /// Max number of following ready whose data to be persisted is empty.
    pub max_empty_number: u64,
    pub raft_msgs: Vec<Vec<eraftpb::Message>>,
}

#[derive(Debug)]
/// ForceLeader process would be:
/// - If it's hibernated, enter wait ticks state, and wake up the peer
/// - Enter pre force leader state, become candidate and send request vote to
///   all peers
/// - Wait for the responses of the request vote, no reject should be received.
/// - Enter force leader state, become leader without leader lease
/// - Execute recovery plan(some remove-peer commands)
/// - After the plan steps are all applied, exit force leader state
pub enum ForceLeaderState {
    WaitTicks {
        syncer: UnsafeRecoveryForceLeaderSyncer,
        failed_stores: HashSet<u64>,
        ticks: usize,
    },
    PreForceLeader {
        syncer: UnsafeRecoveryForceLeaderSyncer,
        failed_stores: HashSet<u64>,
    },
    ForceLeader {
        time: TiInstant,
        failed_stores: HashSet<u64>,
    },
}

// Following shared states are used while reporting to PD for unsafe recovery
// and shared among all the regions per their life cycle.
// The work flow is like:
// 1. report phase
//   - start_unsafe_recovery_report
//      - broadcast wait-apply commands
//      - wait for all the peers' apply indices meet their targets
//      - broadcast fill out report commands
//      - wait for all the peers fill out the reports for themselves
//      - send a store report (through store heartbeat)
// 2. force leader phase
//   - dispatch force leader commands
//     - wait for all the peers that received the command become force leader
//     - start_unsafe_recovery_report
// 3. plan execution phase
//   - dispatch recovery plans
//     - wait for all the creates, deletes and demotes to finish, for the
//       demotes, procedures are:
//       - exit joint state if it is already in joint state
//       - demote failed voters, and promote self to be a voter if it is a
//         learner
//       - exit joint state
//     - start_unsafe_recovery_report
//
// Intends to use RAII to sync unsafe recovery procedures between peers, in
// addition to that, it uses a closure to avoid having a raft router as a member
// variable, which is statically dispatched, thus needs to propagate the
// generics everywhere.
pub struct InvokeClosureOnDrop(Box<dyn Fn() + Send + Sync>);

impl fmt::Debug for InvokeClosureOnDrop {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "InvokeClosureOnDrop")
    }
}

impl Drop for InvokeClosureOnDrop {
    fn drop(&mut self) {
        self.0();
    }
}

pub fn start_unsafe_recovery_report<EK: KvEngine, ER: RaftEngine>(
    router: &RaftRouter<EK, ER>,
    report_id: u64,
    exit_force_leader: bool,
) {
    let wait_apply =
        UnsafeRecoveryWaitApplySyncer::new(report_id, router.clone(), exit_force_leader);
    router.broadcast_normal(|| {
        PeerMsg::SignificantMsg(SignificantMsg::UnsafeRecoveryWaitApply(wait_apply.clone()))
    });
}

// Propose a read index request to the raft group, return the request id and
// whether this request had dropped silently
// #[RaftstoreCommon], copied from Peer::propose_read_index
pub fn propose_read_index<T: raft::Storage>(
    raft_group: &mut RawNode<T>,
    request: Option<&raft_cmdpb::ReadIndexRequest>,
    locked: Option<&LockInfo>,
) -> (Uuid, bool) {
    let last_pending_read_count = raft_group.raft.pending_read_count();
    let last_ready_read_count = raft_group.raft.ready_read_count();

    let id = Uuid::new_v4();
    raft_group.read_index(ReadIndexContext::fields_to_bytes(id, request, locked));

    let pending_read_count = raft_group.raft.pending_read_count();
    let ready_read_count = raft_group.raft.ready_read_count();
    (
        id,
        pending_read_count == last_pending_read_count && ready_read_count == last_ready_read_count,
    )
}

pub fn should_renew_lease(
    is_leader: bool,
    is_splitting: bool,
    is_merging: bool,
    has_force_leader: bool,
) -> bool {
    // A splitting leader should not renew its lease.
    // Because we split regions asynchronous, the leader may read stale results
    // if splitting runs slow on the leader.
    // A merging leader should not renew its lease.
    // Because we merge regions asynchronous, the leader may read stale results
    // if commit merge runs slow on sibling peers.
    // when it enters force leader mode, should not renew lease.
    is_leader && !is_splitting && !is_merging && !has_force_leader
}

// check if the request can be amended to the last pending read?
// return true if it can.
pub fn can_amend_read<C>(
    last_pending_read: Option<&ReadIndexRequest<C>>,
    req: &RaftCmdRequest,
    lease_state: LeaseState,
    max_lease: TimeDuration,
    now: Timespec,
) -> bool {
    match lease_state {
        // Here, combining the new read request with the previous one even if the lease expired
        // is ok because in this case, the previous read index must be sent out with a valid
        // lease instead of a suspect lease. So there must no pending transfer-leader
        // proposals before or after the previous read index, and the lease can be renewed
        // when get heartbeat responses.
        LeaseState::Valid | LeaseState::Expired => {
            if let Some(read) = last_pending_read {
                let is_read_index_request = req
                    .get_requests()
                    .first()
                    .map(|req| req.has_read_index())
                    .unwrap_or_default();
                // A read index request or a read with addition request always needs the
                // response of checking memory lock for async
                // commit, so we cannot apply the optimization here
                if !is_read_index_request
                    && read.addition_request.is_none()
                    && read.propose_time + max_lease > now
                {
                    return true;
                }
            }
        }
        // If the current lease is suspect, new read requests can't be appended into
        // `pending_reads` because if the leader is transferred, the latest read could
        // be dirty.
        _ => {}
    }
    false
}

#[derive(Clone, Debug)]
pub struct UnsafeRecoveryForceLeaderSyncer(Arc<InvokeClosureOnDrop>);

impl UnsafeRecoveryForceLeaderSyncer {
    pub fn new(report_id: u64, router: RaftRouter<impl KvEngine, impl RaftEngine>) -> Self {
        let thread_safe_router = Mutex::new(router);
        let inner = InvokeClosureOnDrop(Box::new(move || {
            info!("Unsafe recovery, force leader finished.");
            let router_ptr = thread_safe_router.lock().unwrap();
            start_unsafe_recovery_report(&*router_ptr, report_id, false);
        }));
        UnsafeRecoveryForceLeaderSyncer(Arc::new(inner))
    }
}

#[derive(Clone, Debug)]
pub struct UnsafeRecoveryExecutePlanSyncer {
    _closure: Arc<InvokeClosureOnDrop>,
    abort: Arc<Mutex<bool>>,
}

impl UnsafeRecoveryExecutePlanSyncer {
    pub fn new(report_id: u64, router: RaftRouter<impl KvEngine, impl RaftEngine>) -> Self {
        let thread_safe_router = Mutex::new(router);
        let abort = Arc::new(Mutex::new(false));
        let abort_clone = abort.clone();
        let closure = InvokeClosureOnDrop(Box::new(move || {
            info!("Unsafe recovery, plan execution finished");
            if *abort_clone.lock().unwrap() {
                warn!("Unsafe recovery, plan execution aborted");
                return;
            }
            let router_ptr = thread_safe_router.lock().unwrap();
            start_unsafe_recovery_report(&*router_ptr, report_id, true);
        }));
        UnsafeRecoveryExecutePlanSyncer {
            _closure: Arc::new(closure),
            abort,
        }
    }

    pub fn abort(&self) {
        *self.abort.lock().unwrap() = true;
    }
}
// Syncer only send to leader in 2nd BR restore
#[derive(Clone, Debug)]
pub struct SnapshotRecoveryWaitApplySyncer {
    _closure: Arc<InvokeClosureOnDrop>,
    abort: Arc<Mutex<bool>>,
}

impl SnapshotRecoveryWaitApplySyncer {
    pub fn new(region_id: u64, sender: SyncSender<u64>) -> Self {
        let thread_safe_router = Mutex::new(sender);
        let abort = Arc::new(Mutex::new(false));
        let abort_clone = abort.clone();
        let closure = InvokeClosureOnDrop(Box::new(move || {
            info!("region {} wait apply finished", region_id);
            if *abort_clone.lock().unwrap() {
                warn!("wait apply aborted");
                return;
            }
            let router_ptr = thread_safe_router.lock().unwrap();

            _ = router_ptr.send(region_id).map_err(|_| {
                warn!("reply waitapply states failure.");
            });
        }));
        SnapshotRecoveryWaitApplySyncer {
            _closure: Arc::new(closure),
            abort,
        }
    }

    pub fn abort(&self) {
        *self.abort.lock().unwrap() = true;
    }
}

#[derive(Clone, Debug)]
pub struct UnsafeRecoveryWaitApplySyncer {
    _closure: Arc<InvokeClosureOnDrop>,
    abort: Arc<Mutex<bool>>,
}

impl UnsafeRecoveryWaitApplySyncer {
    pub fn new(
        report_id: u64,
        router: RaftRouter<impl KvEngine, impl RaftEngine>,
        exit_force_leader: bool,
    ) -> Self {
        let thread_safe_router = Mutex::new(router);
        let abort = Arc::new(Mutex::new(false));
        let abort_clone = abort.clone();
        let closure = InvokeClosureOnDrop(Box::new(move || {
            info!("Unsafe recovery, wait apply finished");
            if *abort_clone.lock().unwrap() {
                warn!("Unsafe recovery, wait apply aborted");
                return;
            }
            let router_ptr = thread_safe_router.lock().unwrap();
            if exit_force_leader {
                (*router_ptr).broadcast_normal(|| {
                    PeerMsg::SignificantMsg(SignificantMsg::ExitForceLeaderState)
                });
            }
            let fill_out_report =
                UnsafeRecoveryFillOutReportSyncer::new(report_id, (*router_ptr).clone());
            (*router_ptr).broadcast_normal(|| {
                PeerMsg::SignificantMsg(SignificantMsg::UnsafeRecoveryFillOutReport(
                    fill_out_report.clone(),
                ))
            });
        }));
        UnsafeRecoveryWaitApplySyncer {
            _closure: Arc::new(closure),
            abort,
        }
    }

    pub fn abort(&self) {
        *self.abort.lock().unwrap() = true;
    }
}

#[derive(Clone, Debug)]
pub struct UnsafeRecoveryFillOutReportSyncer {
    _closure: Arc<InvokeClosureOnDrop>,
    reports: Arc<Mutex<Vec<pdpb::PeerReport>>>,
}

impl UnsafeRecoveryFillOutReportSyncer {
    pub fn new(report_id: u64, router: RaftRouter<impl KvEngine, impl RaftEngine>) -> Self {
        let thread_safe_router = Mutex::new(router);
        let reports = Arc::new(Mutex::new(vec![]));
        let reports_clone = reports.clone();
        let closure = InvokeClosureOnDrop(Box::new(move || {
            info!("Unsafe recovery, peer reports collected");
            let mut store_report = pdpb::StoreReport::default();
            {
                let mut reports_ptr = reports_clone.lock().unwrap();
                store_report.set_peer_reports(mem::take(&mut *reports_ptr).into());
            }
            store_report.set_step(report_id);
            let router_ptr = thread_safe_router.lock().unwrap();
            if let Err(e) = (*router_ptr).send_control(StoreMsg::UnsafeRecoveryReport(store_report))
            {
                error!("Unsafe recovery, fail to schedule reporting"; "err" => ?e);
            }
        }));
        UnsafeRecoveryFillOutReportSyncer {
            _closure: Arc::new(closure),
            reports,
        }
    }

    pub fn report_for_self(&self, report: pdpb::PeerReport) {
        let mut reports_ptr = self.reports.lock().unwrap();
        (*reports_ptr).push(report);
    }
}

pub enum SnapshotRecoveryState {
    // This state is set by the leader peer fsm. Once set, it sync and check leader commit index
    // and force forward to last index once follower appended and then it also is checked
    // every time this peer applies a the last index, if the last index is met, this state is
    // reset / droppeds. The syncer is droped and send the response to the invoker, triggers
    // the next step of recovery process.
    WaitLogApplyToLast {
        target_index: u64,
        syncer: SnapshotRecoveryWaitApplySyncer,
    },
}

pub enum UnsafeRecoveryState {
    // Stores the state that is necessary for the wait apply stage of unsafe recovery process.
    // This state is set by the peer fsm. Once set, it is checked every time this peer applies a
    // new entry or a snapshot, if the target index is met, this state is reset / droppeds. The
    // syncer holds a reference counted inner object that is shared among all the peers, whose
    // destructor triggers the next step of unsafe recovery report process.
    WaitApply {
        target_index: u64,
        syncer: UnsafeRecoveryWaitApplySyncer,
    },
    DemoteFailedVoters {
        syncer: UnsafeRecoveryExecutePlanSyncer,
        failed_voters: Vec<metapb::Peer>,
        target_index: u64,
        // Failed regions may be stuck in joint state, if that is the case, we need to ask the
        // region to exit joint state before proposing the demotion.
        demote_after_exit: bool,
    },
    Destroy(UnsafeRecoveryExecutePlanSyncer),
}

#[derive(Getters, MutGetters)]
pub struct Peer<EK, ER>
where
    EK: KvEngine,
    ER: RaftEngine,
{
    /// The ID of the Region which this Peer belongs to.
    region_id: u64,
    // TODO: remove it once panic!() support slog fields.
    /// Peer_tag, "[region <region_id>] <peer_id>"
    pub tag: String,
    /// The Peer meta information.
    pub peer: metapb::Peer,

    /// The Raft state machine of this Peer.
    pub raft_group: RawNode<PeerStorage<EK, ER>>,
    /// The online configurable Raft configurations
    raft_max_inflight_msgs: usize,
    /// The cache of meta information for Region's other Peers.
    peer_cache: RefCell<HashMap<u64, metapb::Peer>>,
    /// Record the last instant of each peer's heartbeat response.
    pub peer_heartbeats: HashMap<u64, Instant>,
    /// Record the waiting data status of each follower or learner peer.
    pub wait_data_peers: Vec<u64>,

    proposals: ProposalQueue<Callback<EK::Snapshot>>,
    #[getset(get = "pub", get_mut = "pub")]
    leader_lease: Lease,
    pending_reads: ReadIndexQueue<Callback<EK::Snapshot>>,

    /// If it fails to send messages to leader.
    pub leader_unreachable: bool,
    /// Indicates whether the peer should be woken up.
    pub should_wake_up: bool,
    /// Whether this peer is destroyed asynchronously.
    /// If it's true,
    /// - when merging, its data in storeMeta will be removed early by the
    ///   target peer.
    /// - all read requests must be rejected.
    pub pending_remove: bool,
    /// Currently it's used to indicate whether the witness -> non-witess
    /// convertion operation is complete. The meaning of completion is that
    /// this peer must contain the applied data, then PD can consider that
    /// the conversion operation is complete, and can continue to schedule
    /// other operators to prevent the existence of multiple witnesses in
    /// the same time period.
    pub wait_data: bool,

    /// Force leader state is only used in online recovery when the majority of
    /// peers are missing. In this state, it forces one peer to become leader
    /// out of accordance with Raft election rule, and forbids any
    /// read/write proposals. With that, we can further propose remove
    /// failed-nodes conf-change, to make the Raft group forms majority and
    /// works normally later on.
    ///
    /// For details, see the comment of `ForceLeaderState`.
    pub force_leader: Option<ForceLeaderState>,

    /// Record the instants of peers being added into the configuration.
    /// Remove them after they are not pending any more.
    pub peers_start_pending_time: Vec<(u64, Instant)>,
    /// A inaccurate cache about which peer is marked as down.
    down_peer_ids: Vec<u64>,

    /// An inaccurate difference in region size since last reset.
    /// It is used to decide whether split check is needed.
    pub size_diff_hint: u64,
    /// The count of deleted keys since last reset.
    delete_keys_hint: u64,
    /// An inaccurate difference in region size after compaction.
    /// It is used to trigger check split to update approximate size and keys
    /// after space reclamation of deleted entries.
    pub compaction_declined_bytes: u64,
    /// Approximate size of the region.
    pub approximate_size: Option<u64>,
    /// Approximate keys of the region.
    pub approximate_keys: Option<u64>,
    /// Whether this region has scheduled a split check task. If we just
    /// splitted  the region or ingested one file which may be overlapped
    /// with the existed data, reset the flag so that the region can be
    /// splitted again.
    pub may_skip_split_check: bool,

    /// The state for consistency check.
    pub consistency_state: ConsistencyState,

    /// The counter records pending snapshot requests.
    pub pending_request_snapshot_count: Arc<AtomicUsize>,
    /// The index of last scheduled committed raft log.
    pub last_applying_idx: u64,
    /// The index of last compacted raft log. It is used for the next compact
    /// log task.
    pub last_compacted_idx: u64,
    /// The index of the latest committed split command.
    last_committed_split_idx: u64,
    /// Approximate size of logs that is applied but not compacted yet.
    pub raft_log_size_hint: u64,

    /// The write fence index.
    /// If there are pessimistic locks, PrepareMerge can be proposed after
    /// applying to this index. When a pending PrepareMerge exists, no more
    /// write commands should be proposed. This avoids proposing pessimistic
    /// locks that are already deleted before PrepareMerge.
    pub prepare_merge_fence: u64,
    pub pending_prepare_merge: Option<RaftCmdRequest>,

    /// The index of the latest committed prepare merge command.
    last_committed_prepare_merge_idx: u64,
    /// The merge related state. It indicates this Peer is in merging.
    pub pending_merge_state: Option<MergeState>,
    /// The rollback merge proposal can be proposed only when the number
    /// of peers is greater than the majority of all peers.
    /// There are more details in the annotation above
    /// `test_node_merge_write_data_to_source_region_after_merging`
    /// The peers who want to rollback merge.
    pub want_rollback_merge_peers: HashSet<u64>,
    /// Source region is catching up logs for merge.
    pub catch_up_logs: Option<CatchUpLogs>,

    /// Write Statistics for PD to schedule hot spot.
    pub peer_stat: PeerStat,

    /// Time of the last attempt to wake up inactive leader.
    pub bcast_wake_up_time: Option<TiInstant>,
    /// Current replication mode version.
    pub replication_mode_version: u64,
    /// The required replication state at current version.
    pub dr_auto_sync_state: DrAutoSyncState,
    /// A flag that caches sync state. It's set to true when required
    /// replication state is reached for current region.
    pub replication_sync: bool,

    /// The known newest conf version and its corresponding peer list
    /// Send to these peers to check whether itself is stale.
    pub check_stale_conf_ver: u64,
    pub check_stale_peers: Vec<metapb::Peer>,
    /// Whether this peer is created by replication and is the first
    /// one of this region on local store.
    pub local_first_replicate: bool,

    pub txn_extra_op: Arc<AtomicCell<TxnExtraOp>>,

    /// Transaction extensions related to this peer.
    pub txn_ext: Arc<TxnExt>,

    // disk full peer set.
    pub disk_full_peers: DiskFullPeers,

    // show whether an already disk full TiKV appears in the potential majority set.
    pub dangerous_majority_set: bool,

    // region merge logic need to be broadcast to all followers when disk full happens.
    pub has_region_merge_proposal: bool,

    pub region_merge_proposal_index: u64,

    pub read_progress: Arc<RegionReadProgress>,

    pub memtrace_raft_entries: usize,
    /// Used for async write io.
    unpersisted_readies: VecDeque<UnpersistedReady>,
    /// The message count in `unpersisted_readies` for memory caculation.
    unpersisted_message_count: usize,
    /// The context of applying snapshot.
    apply_snap_ctx: Option<ApplySnapshotContext>,
    /// region buckets.
    pub region_buckets: Option<BucketStat>,
    pub last_region_buckets: Option<BucketStat>,
    /// lead_transferee if this peer(leader) is in a leadership transferring.
    pub lead_transferee: u64,
    pub unsafe_recovery_state: Option<UnsafeRecoveryState>,
    // Used as the memory state for Flashback to reject RW/Schedule before proposing.
    pub is_in_flashback: bool,
    pub snapshot_recovery_state: Option<SnapshotRecoveryState>,
}

impl<EK, ER> Peer<EK, ER>
where
    EK: KvEngine,
    ER: RaftEngine,
{
    /// Sets commit group to the peer.
    pub fn init_replication_mode(&mut self, state: &mut GlobalReplicationState) {
        debug!("init commit group"; "state" => ?state, "region_id" => self.region_id, "peer_id" => self.peer.id);
        if self.is_initialized() {
            let version = state.status().get_dr_auto_sync().state_id;
            let gb = state.calculate_commit_group(version, self.get_store().region().get_peers());
            self.raft_group.raft.assign_commit_groups(gb);
        }
        self.replication_sync = false;
        if state.status().get_mode() == ReplicationMode::Majority {
            self.raft_group.raft.enable_group_commit(false);
            self.replication_mode_version = 0;
            self.dr_auto_sync_state = DrAutoSyncState::Async;
            return;
        }
        self.replication_mode_version = state.status().get_dr_auto_sync().state_id;
        let enable = state.status().get_dr_auto_sync().get_state() != DrAutoSyncState::Async;
        self.raft_group.raft.enable_group_commit(enable);
        self.dr_auto_sync_state = state.status().get_dr_auto_sync().get_state();
    }

    #[inline]
    pub fn get_index_term(&self, idx: u64) -> u64 {
        match self.raft_group.raft.raft_log.term(idx) {
            Ok(t) => t,
            Err(e) => panic!("{} fail to load term for {}: {:?}", self.tag, idx, e),
        }
    }

    pub fn maybe_append_merge_entries(&mut self, merge: &CommitMergeRequest) -> Option<u64> {
        let mut entries = merge.get_entries();
        if entries.is_empty() {
            // Though the entries is empty, it is possible that one source peer has caught
            // up the logs but commit index is not updated. If other source peers are
            // already destroyed, so the raft group will not make any progress, namely the
            // source peer can not get the latest commit index anymore.
            // Here update the commit index to let source apply rest uncommitted entries.
            return if merge.get_commit() > self.raft_group.raft.raft_log.committed {
                self.raft_group.raft.raft_log.commit_to(merge.get_commit());
                Some(merge.get_commit())
            } else {
                None
            };
        }
        let first = entries.first().unwrap();
        // make sure message should be with index not smaller than committed
        let mut log_idx = first.get_index() - 1;
        debug!(
            "append merge entries";
            "log_index" => log_idx,
            "merge_commit" => merge.get_commit(),
            "commit_index" => self.raft_group.raft.raft_log.committed,
        );
        if log_idx < self.raft_group.raft.raft_log.committed {
            // There are maybe some logs not included in CommitMergeRequest's entries, like
            // CompactLog, so the commit index may exceed the last index of the entires from
            // CommitMergeRequest. If that, no need to append
            if self.raft_group.raft.raft_log.committed - log_idx >= entries.len() as u64 {
                return None;
            }
            entries = &entries[(self.raft_group.raft.raft_log.committed - log_idx) as usize..];
            log_idx = self.raft_group.raft.raft_log.committed;
        }
        let log_term = self.get_index_term(log_idx);

        let last_log = entries.last().unwrap();
        if last_log.term > self.term() {
            // Hack: In normal flow, when leader sends the entries, it will use a term
            // that's not less than the last log term. And follower will update its states
            // correctly. For merge, we append the log without raft, so we have to take care
            // of term explicitly to get correct metadata.
            info!(
                "become follower for new logs";
                "new_log_term" => last_log.term,
                "new_log_index" => last_log.index,
                "term" => self.term(),
                "region_id" => self.region_id,
                "peer_id" => self.peer.get_id(),
            );
            self.raft_group
                .raft
                .become_follower(last_log.term, INVALID_ID);
        }

        self.raft_group
            .raft
            .raft_log
            .maybe_append(log_idx, log_term, merge.get_commit(), entries)
            .map(|(_, last_index)| last_index)
    }

    #[inline]
    pub fn is_initialized(&self) -> bool {
        self.get_store().is_initialized()
    }

    #[inline]
    pub fn region(&self) -> &metapb::Region {
        self.get_store().region()
    }

    /// Check whether the peer can be hibernated.
    ///
    /// This should be used with `check_after_tick` to get a correct conclusion.
    pub fn check_before_tick(&self, cfg: &Config) -> CheckTickResult {
        let mut res = CheckTickResult::default();
        if !self.is_leader() {
            return res;
        }
        res.leader = true;
        if self.raft_group.raft.election_elapsed + 1 < cfg.raft_election_timeout_ticks {
            return res;
        }
        let status = self.raft_group.status();
        let last_index = self.raft_group.raft.raft_log.last_index();
        for (id, pr) in status.progress.unwrap().iter() {
            // Even a recent inactive node is also considered. If we put leader into sleep,
            // followers or learners may not sync its logs for a long time and become
            // unavailable. We choose availability instead of performance in this case.
            if *id == self.peer.get_id() {
                continue;
            }
            if pr.matched != last_index {
                res.reason = "replication";
                return res;
            }
        }
        if self.raft_group.raft.pending_read_count() > 0 {
            res.reason = "pending read";
            return res;
        }
        if self.raft_group.raft.lead_transferee.is_some() {
            res.reason = "transfer leader";
            return res;
        }
        if self.force_leader.is_some() {
            res.reason = "force leader";
            return res;
        }
        // Unapplied entries can change the configuration of the group.
        if self.get_store().applied_index() < last_index {
            res.reason = "unapplied";
            return res;
        }
        if self.replication_mode_need_catch_up() {
            res.reason = "replication mode";
            return res;
        }
        res.up_to_date = true;
        res
    }

    #[inline]
    pub fn has_valid_leader(&self) -> bool {
        if self.raft_group.raft.leader_id == raft::INVALID_ID {
            return false;
        }
        for p in self.region().get_peers() {
            if p.get_id() == self.raft_group.raft.leader_id && p.get_role() != PeerRole::Learner {
                return true;
            }
        }
        false
    }

    /// Pings if followers are still connected.
    ///
    /// Leader needs to know exact progress of followers, and
    /// followers just need to know whether leader is still alive.
    pub fn ping(&mut self) {
        if self.is_leader() {
            self.raft_group.ping();
        }
    }

    pub fn has_uncommitted_log(&self) -> bool {
        self.raft_group.raft.raft_log.committed < self.raft_group.raft.raft_log.last_index()
    }

    /// Set the region of a peer.
    ///
    /// This will update the region of the peer, caller must ensure the region
    /// has been preserved in a durable device.
    pub fn set_region(
        &mut self,
        host: &CoprocessorHost<impl KvEngine>,
        reader: &mut ReadDelegate,
        region: metapb::Region,
        reason: RegionChangeReason,
    ) {
        if self.region().get_region_epoch().get_version() < region.get_region_epoch().get_version()
        {
            // Epoch version changed, disable read on the local reader for this region.
            self.leader_lease.expire_remote_lease();
        }
        self.mut_store().set_region(region.clone());
        let progress = ReadProgress::region(region);
        // Always update read delegate's region to avoid stale region info after a
        // follower becoming a leader.
        self.maybe_update_read_progress(reader, progress);

        // Update leader info
        self.read_progress
            .update_leader_info(self.leader_id(), self.term(), self.region());

        {
            let mut pessimistic_locks = self.txn_ext.pessimistic_locks.write();
            pessimistic_locks.term = self.term();
            pessimistic_locks.version = self.region().get_region_epoch().get_version();
        }

        if !self.pending_remove {
            host.on_region_changed(
                self.region(),
                RegionChangeEvent::Update(reason),
                self.get_role(),
            );
        }
    }

    #[inline]
    pub fn peer_id(&self) -> u64 {
        self.peer.get_id()
    }

    #[inline]
    pub fn leader_id(&self) -> u64 {
        self.raft_group.raft.leader_id
    }

    #[inline]
    pub fn is_leader(&self) -> bool {
        self.raft_group.raft.state == StateRole::Leader
    }

    #[inline]
    pub fn is_witness(&self) -> bool {
        self.peer.is_witness
    }

    #[inline]
    pub fn get_role(&self) -> StateRole {
        self.raft_group.raft.state
    }

    #[inline]
    pub fn get_store(&self) -> &PeerStorage<EK, ER> {
        self.raft_group.store()
    }

    #[inline]
    pub fn mut_store(&mut self) -> &mut PeerStorage<EK, ER> {
        self.raft_group.mut_store()
    }

    /// Whether the snapshot is handling.
    /// See the comments of `check_snap_status` for more details.
    #[inline]
    pub fn is_handling_snapshot(&self) -> bool {
        self.apply_snap_ctx.is_some() || self.get_store().is_applying_snapshot()
    }

    /// Returns `true` if the raft group has replicated a snapshot but not
    /// committed it yet.
    #[inline]
    pub fn has_pending_snapshot(&self) -> bool {
        self.get_pending_snapshot().is_some()
    }

    #[inline]
    pub fn get_pending_snapshot(&self) -> Option<&eraftpb::Snapshot> {
        self.raft_group.snap()
    }

    #[inline]
    pub fn in_joint_state(&self) -> bool {
        self.region().get_peers().iter().any(|p| {
            p.get_role() == PeerRole::IncomingVoter || p.get_role() == PeerRole::DemotingVoter
        })
    }

    #[inline]
    pub fn ready_to_handle_pending_snap(&self) -> bool {
        // If apply worker is still working, written apply state may be overwritten
        // by apply worker. So we have to wait here.
        // Please note that commit_index can't be used here. When applying a snapshot,
        // a stale heartbeat can make the leader think follower has already applied
        // the snapshot, and send remaining log entries, which may increase
        // commit_index.
        // TODO: add more test
        self.last_applying_idx == self.get_store().applied_index()
            // Requesting snapshots also triggers apply workers to write
            // apply states even if there is no pending committed entry.
            // TODO: Instead of sharing the counter, we should apply snapshots
            //       in apply workers.
            && self.pending_request_snapshot_count.load(Ordering::SeqCst) == 0
    }

    #[inline]
    pub fn is_splitting(&self) -> bool {
        self.last_committed_split_idx > self.get_store().applied_index()
    }

    #[inline]
    pub fn is_merging(&self) -> bool {
        self.last_committed_prepare_merge_idx > self.get_store().applied_index()
            || self.pending_merge_state.is_some()
    }

    /// Checks if leader needs to keep sending logs for follower.
    ///
    /// In DrAutoSync mode, if leader goes to sleep before the region is sync,
    /// PD may wait longer time to reach sync state.
    pub fn replication_mode_need_catch_up(&self) -> bool {
        self.replication_mode_version > 0
            && self.dr_auto_sync_state != DrAutoSyncState::Async
            && !self.replication_sync
    }

    pub fn schedule_raftlog_gc<T: Transport>(
        &mut self,
        ctx: &mut PollContext<EK, ER, T>,
        to: u64,
    ) -> bool {
        let task = RaftlogGcTask::gc(self.region_id, self.last_compacted_idx, to);
        debug!(
            "scheduling raft log gc task";
            "region_id" => self.region_id,
            "peer_id" => self.peer_id(),
            "task" => %task,
        );
        if let Err(e) = ctx.raftlog_gc_scheduler.schedule(task) {
            error!(
                "failed to schedule raft log gc task";
                "region_id" => self.region_id,
                "peer_id" => self.peer_id(),
                "err" => %e,
            );
            false
        } else {
            true
        }
    }

    pub fn unpersisted_ready_len(&self) -> usize {
        self.unpersisted_readies.len()
    }

    pub fn has_unpersisted_ready(&self) -> bool {
        !self.unpersisted_readies.is_empty()
    }

    pub fn post_split(&mut self) {
        // Reset delete_keys_hint and size_diff_hint.
        self.delete_keys_hint = 0;
        self.size_diff_hint = 0;
        self.reset_region_buckets();
    }

    pub fn reset_region_buckets(&mut self) {
        if self.region_buckets.is_some() {
            self.last_region_buckets = self.region_buckets.take();
            self.region_buckets = None;
        }
    }

    fn maybe_update_read_progress(&self, reader: &mut ReadDelegate, progress: ReadProgress) {
        if self.pending_remove {
            return;
        }
        debug!(
            "update read progress";
            "region_id" => self.region_id,
            "peer_id" => self.peer.get_id(),
            "progress" => ?progress,
        );
        reader.update(progress);
    }

    pub fn maybe_campaign(&mut self, parent_is_leader: bool) -> bool {
        if self.region().get_peers().len() <= 1 {
            // The peer campaigned when it was created, no need to do it again.
            return false;
        }

        if !parent_is_leader {
            return false;
        }

        // If last peer is the leader of the region before split, it's intuitional for
        // it to become the leader of new split region.
        let _ = self.raft_group.campaign();
        true
    }

    pub fn transfer_leader(&mut self, peer: &metapb::Peer) {
        info!(
            "transfer leader";
            "region_id" => self.region_id,
            "peer_id" => self.peer.get_id(),
            "peer" => ?peer,
        );

        self.raft_group.transfer_leader(peer.get_id());
        self.should_wake_up = true;
    }

    pub fn ready_to_transfer_leader<T>(
        &self,
        ctx: &mut PollContext<EK, ER, T>,
        mut index: u64,
        peer: &metapb::Peer,
    ) -> Option<&'static str> {
        let peer_id = peer.get_id();
        let status = self.raft_group.status();
        let progress = status.progress.unwrap();

        if !progress.conf().voters().contains(peer_id) {
            return Some("non voter");
        }

        for (id, pr) in progress.iter() {
            if pr.state == ProgressState::Snapshot {
                return Some("pending snapshot");
            }
            if *id == peer_id && index == 0 {
                // index will be zero if it's sent from an instance without
                // pre-transfer-leader feature. Set it to matched to make it
                // possible to transfer leader to an older version. It may be
                // useful during rolling restart.
                index = pr.matched;
            }
        }

        if self.raft_group.raft.has_pending_conf()
            || self.raft_group.raft.pending_conf_index > index
        {
            return Some("pending conf change");
        }

        let last_index = self.get_store().last_index();
        if last_index >= index + ctx.cfg.leader_transfer_max_log_lag {
            return Some("log gap");
        }
        None
    }

    pub fn pre_read_index(&self) -> Result<()> {
        fail_point!(
            "before_propose_readindex",
            |s| if s.map_or(true, |s| s.parse().unwrap_or(true)) {
                Ok(())
            } else {
                Err(box_err!(
                    "{} can not read due to injected failure",
                    self.tag
                ))
            }
        );

        // See more in ready_to_handle_read().
        if self.is_splitting() {
            return Err(Error::ReadIndexNotReady {
                reason: "can not read index due to split",
                region_id: self.region_id,
            });
        }
        if self.is_merging() {
            return Err(Error::ReadIndexNotReady {
                reason: "can not read index due to merge",
                region_id: self.region_id,
            });
        }
        Ok(())
    }

    pub fn has_unresolved_reads(&self) -> bool {
        self.pending_reads.has_unresolved()
    }

    /// `ReadIndex` requests could be lost in network, so on followers commands
    /// could queue in `pending_reads` forever. Sending a new `ReadIndex`
    /// periodically can resolve this.
    pub fn retry_pending_reads(&mut self, cfg: &Config) {
        if self.is_leader()
            || !self.pending_reads.check_needs_retry(cfg)
            || self.pre_read_index().is_err()
        {
            return;
        }

        let read = self.pending_reads.back_mut().unwrap();
        debug_assert!(read.read_index.is_none());
        self.raft_group
            .read_index(ReadIndexContext::fields_to_bytes(
                read.id,
                read.addition_request.as_deref(),
                None,
            ));
        debug!(
            "request to get a read index";
            "request_id" => ?read.id,
            "region_id" => self.region_id,
            "peer_id" => self.peer.get_id(),
        );
    }

    pub fn push_pending_read(
        &mut self,
        read: ReadIndexRequest<Callback<EK::Snapshot>>,
        is_leader: bool,
    ) {
        self.pending_reads.push_back(read, is_leader);
    }

    // Propose a read index request to the raft group, return the request id and
    // whether this request had dropped silently
    pub fn propose_read_index(
        &mut self,
        request: Option<&raft_cmdpb::ReadIndexRequest>,
        locked: Option<&LockInfo>,
    ) -> (Uuid, bool) {
        propose_read_index(&mut self.raft_group, request, locked)
    }

    /// Returns (minimal matched, minimal committed_index)
    ///
    /// For now, it is only used in merge.
    pub fn get_min_progress(&self) -> Result<(u64, u64)> {
        let (mut min_m, mut min_c) = (None, None);
        if let Some(progress) = self.raft_group.status().progress {
            for (id, pr) in progress.iter() {
                // Reject merge if there is any pending request snapshot,
                // because a target region may merge a source region which is in
                // an invalid state.
                if pr.state == ProgressState::Snapshot
                    || pr.pending_request_snapshot != INVALID_INDEX
                {
                    return Err(box_err!(
                        "there is a pending snapshot peer {} [{:?}], skip merge",
                        id,
                        pr
                    ));
                }
                if min_m.unwrap_or(u64::MAX) > pr.matched {
                    min_m = Some(pr.matched);
                }
                if min_c.unwrap_or(u64::MAX) > pr.committed_index {
                    min_c = Some(pr.committed_index);
                }
            }
        }
        let (mut min_m, min_c) = (min_m.unwrap_or(0), min_c.unwrap_or(0));
        if min_m < min_c {
            warn!(
                "min_matched < min_committed, raft progress is inaccurate";
                "region_id" => self.region_id,
                "peer_id" => self.peer.get_id(),
                "min_matched" => min_m,
                "min_committed" => min_c,
            );
            // Reset `min_matched` to `min_committed`, since the raft log at `min_committed`
            // is known to be committed in all peers, all of the peers should also have
            // replicated it
            min_m = min_c;
        }
        Ok((min_m, min_c))
    }

    pub fn maybe_reject_transfer_leader_msg<T>(
        &mut self,
        ctx: &mut PollContext<EK, ER, T>,
        msg: &eraftpb::Message,
        peer_disk_usage: DiskUsage,
    ) -> bool {
        if self.is_witness() {
            // shouldn't transfer leader to witness peer
            return true;
        }

        let pending_snapshot = self.is_handling_snapshot() || self.has_pending_snapshot();
        if pending_snapshot
            || msg.get_from() != self.leader_id()
            // Transfer leader to node with disk full will lead to write availablity downback.
            // But if the current leader is disk full, and send such request, we should allow it,
            // because it may be a read leader balance request.
            || (!matches!(ctx.self_disk_usage, DiskUsage::Normal) &&
            matches!(peer_disk_usage, DiskUsage::Normal))
        {
            info!(
                "reject transferring leader";
                "region_id" => self.region_id,
                "peer_id" => self.peer.get_id(),
                "from" => msg.get_from(),
                "pending_snapshot" => pending_snapshot,
                "disk_usage" => ?ctx.self_disk_usage,
            );
            return true;
        }
        false
    }

    /// Before ack the transfer leader message sent by the leader.
    /// Currently, it only warms up the entry cache in this stage.
    ///
    /// This return whether the msg should be acked. When cache is warmed up
    /// or the warmup operation is timeout, it is true.
    pub fn pre_ack_transfer_leader_msg<T>(
        &mut self,
        ctx: &mut PollContext<EK, ER, T>,
        msg: &eraftpb::Message,
    ) -> bool {
        if !ctx.cfg.warmup_entry_cache_enabled() {
            return true;
        }

        // The start index of warmup range. It is leader's entry_cache_first_index,
        // which in general is equal to the lowest matched index.
        let mut low = msg.get_index();
        let last_index = self.get_store().last_index();
        let mut should_ack_now = false;

        // Need not to warm up when the index is 0.
        // There are two cases where index can be 0:
        // 1. During rolling upgrade, old instances may not support warmup.
        // 2. The leader's entry cache is empty.
        if low == 0 || low > last_index {
            // There is little possibility that the warmup_range_start
            // is larger than the last index. Check the test case
            // `test_when_warmup_range_start_is_larger_than_last_index`
            // for details.
            should_ack_now = true;
        } else {
            if low < self.last_compacted_idx {
                low = self.last_compacted_idx
            };
            // Check if the entry cache is already warmed up.
            if let Some(first_index) = self.get_store().entry_cache_first_index() {
                if low >= first_index {
                    fail_point!("entry_cache_already_warmed_up");
                    should_ack_now = true;
                }
            }
        }

        if should_ack_now {
            return true;
        }

        // Check if the warmup operation is timeout if warmup is already started.
        if let Some(state) = self.mut_store().entry_cache_warmup_state_mut() {
            // If it is timeout, this peer should ack the message so that
            // the leadership transfer process can continue.
            state.check_task_timeout(ctx.cfg.max_entry_cache_warmup_duration.0)
        } else {
            self.mut_store().async_warm_up_entry_cache(low).is_none()
        }
    }

    pub fn ack_transfer_leader_msg(
        &mut self,
        reply_cmd: bool, // whether it is a reply to a TransferLeader command
    ) {
        let mut msg = eraftpb::Message::new();
        msg.set_from(self.peer_id());
        msg.set_to(self.leader_id());
        msg.set_msg_type(eraftpb::MessageType::MsgTransferLeader);
        msg.set_index(self.get_store().applied_index());
        msg.set_log_term(self.term());
        if reply_cmd {
            msg.set_context(Bytes::from_static(TRANSFER_LEADER_COMMAND_REPLY_CTX));
        }
        self.raft_group.raft.msgs.push(msg);
    }

    pub fn voters(&self) -> raft::util::Union<'_> {
        self.raft_group.raft.prs().conf().voters().ids()
    }

    pub fn term(&self) -> u64 {
        self.raft_group.raft.term
    }

    pub fn stop(&mut self) {
        self.mut_store().cancel_applying_snap();
        self.pending_reads.clear_all(None);
    }

    pub fn maybe_add_want_rollback_merge_peer(&mut self, peer_id: u64, extra_msg: &ExtraMessage) {
        if !self.is_leader() {
            return;
        }
        if let Some(ref state) = self.pending_merge_state {
            if state.get_commit() == extra_msg.get_index() {
                self.add_want_rollback_merge_peer(peer_id);
            }
        }
    }

    pub fn add_want_rollback_merge_peer(&mut self, peer_id: u64) {
        assert!(self.pending_merge_state.is_some());
        self.want_rollback_merge_peers.insert(peer_id);
    }

    pub fn clear_disk_full_peers<T>(&mut self, ctx: &PollContext<EK, ER, T>) {
        let disk_full_peers = mem::take(&mut self.disk_full_peers);
        let raft = &mut self.raft_group.raft;
        for peer in disk_full_peers.peers.into_keys() {
            raft.adjust_max_inflight_msgs(peer, ctx.cfg.raft_max_inflight_msgs);
        }
    }
}

#[derive(Default, Debug)]
pub struct DiskFullPeers {
    majority: bool,
    // Indicates whether a peer can help to establish a quorum.
    peers: HashMap<u64, (DiskUsage, bool)>,
}

impl DiskFullPeers {
    pub fn is_empty(&self) -> bool {
        self.peers.is_empty()
    }
    pub fn majority(&self) -> bool {
        self.majority
    }
    pub fn has(&self, peer_id: u64) -> bool {
        !self.peers.is_empty() && self.peers.contains_key(&peer_id)
    }
    pub fn get(&self, peer_id: u64) -> Option<DiskUsage> {
        self.peers.get(&peer_id).map(|x| x.0)
    }
}

impl<EK, ER> Peer<EK, ER>
where
    EK: KvEngine,
    ER: RaftEngine,
{
    pub fn insert_peer_cache(&mut self, peer: metapb::Peer) {
        self.peer_cache.borrow_mut().insert(peer.get_id(), peer);
    }

    pub fn remove_peer_from_cache(&mut self, peer_id: u64) {
        self.peer_cache.borrow_mut().remove(&peer_id);
    }

    pub fn get_peer_from_cache(&self, peer_id: u64) -> Option<metapb::Peer> {
        if peer_id == 0 {
            return None;
        }
        fail_point!("stale_peer_cache_2", peer_id == 2, |_| None);
        if let Some(peer) = self.peer_cache.borrow().get(&peer_id) {
            return Some(peer.clone());
        }

        // Try to find in region, if found, set in cache.
        for peer in self.region().get_peers() {
            if peer.get_id() == peer_id {
                self.peer_cache.borrow_mut().insert(peer_id, peer.clone());
                return Some(peer.clone());
            }
        }

        None
    }

    fn prepare_raft_message(&self) -> RaftMessage {
        let mut send_msg = RaftMessage::default();
        send_msg.set_region_id(self.region_id);
        // set current epoch
        send_msg.set_region_epoch(self.region().get_region_epoch().clone());
        send_msg.set_from_peer(self.peer.clone());
        send_msg
    }

    pub fn send_extra_message<T: Transport>(
        &self,
        msg: ExtraMessage,
        trans: &mut T,
        to: &metapb::Peer,
    ) {
        let mut send_msg = self.prepare_raft_message();
        let ty = msg.get_type();
        debug!("send extra msg";
            "region_id" => self.region_id,
            "peer_id" => self.peer.get_id(),
            "msg_type" => ?ty,
            "to" => to.get_id()
        );
        send_msg.set_extra_msg(msg);
        send_msg.set_to_peer(to.clone());
        if let Err(e) = trans.send(send_msg) {
            error!(?e;
                "failed to send extra message";
                "type" => ?ty,
                "region_id" => self.region_id,
                "peer_id" => self.peer.get_id(),
                "target" => ?to,
            );
        }
    }

    pub fn bcast_wake_up_message<T: Transport>(&self, ctx: &mut PollContext<EK, ER, T>) {
        for peer in self.region().get_peers() {
            if peer.get_id() == self.peer_id() {
                continue;
            }
            self.send_wake_up_message(ctx, peer);
        }
    }

    pub fn send_wake_up_message<T: Transport>(
        &self,
        ctx: &mut PollContext<EK, ER, T>,
        peer: &metapb::Peer,
    ) {
        let mut msg = ExtraMessage::default();
        msg.set_type(ExtraMessageType::MsgRegionWakeUp);
        self.send_extra_message(msg, &mut ctx.trans, peer);
    }

    pub fn bcast_check_stale_peer_message<T: Transport>(
        &mut self,
        ctx: &mut PollContext<EK, ER, T>,
    ) {
        if self.check_stale_conf_ver < self.region().get_region_epoch().get_conf_ver() {
            self.check_stale_conf_ver = self.region().get_region_epoch().get_conf_ver();
            self.check_stale_peers = self.region().get_peers().to_vec();
        }
        for peer in &self.check_stale_peers {
            if peer.get_id() == self.peer_id() {
                continue;
            }
            let mut extra_msg = ExtraMessage::default();
            extra_msg.set_type(ExtraMessageType::MsgCheckStalePeer);
            self.send_extra_message(extra_msg, &mut ctx.trans, peer);
        }
    }

    pub fn on_check_stale_peer_response(
        &mut self,
        check_conf_ver: u64,
        check_peers: Vec<metapb::Peer>,
    ) {
        if self.check_stale_conf_ver < check_conf_ver {
            self.check_stale_conf_ver = check_conf_ver;
            self.check_stale_peers = check_peers;
        }
    }

    pub fn send_want_rollback_merge<T: Transport>(
        &self,
        premerge_commit: u64,
        ctx: &mut PollContext<EK, ER, T>,
    ) {
        let to_peer = match self.get_peer_from_cache(self.leader_id()) {
            Some(p) => p,
            None => {
                warn!(
                    "failed to look up recipient peer";
                    "region_id" => self.region_id,
                    "peer_id" => self.peer.get_id(),
                    "to_peer" => self.leader_id(),
                );
                return;
            }
        };
        let mut extra_msg = ExtraMessage::default();
        extra_msg.set_type(ExtraMessageType::MsgWantRollbackMerge);
        extra_msg.set_index(premerge_commit);
        self.send_extra_message(extra_msg, &mut ctx.trans, &to_peer);
    }

    pub fn adjust_cfg_if_changed<T>(&mut self, ctx: &PollContext<EK, ER, T>) {
        let raft_max_inflight_msgs = ctx.cfg.raft_max_inflight_msgs;
        if self.is_leader() && (raft_max_inflight_msgs != self.raft_max_inflight_msgs) {
            let peers: Vec<_> = self.region().get_peers().into();
            for p in peers {
                if p != self.peer {
                    self.raft_group
                        .raft
                        .adjust_max_inflight_msgs(p.get_id(), raft_max_inflight_msgs);
                }
            }
            self.raft_max_inflight_msgs = raft_max_inflight_msgs;
        }
        self.raft_group.raft.r.max_msg_size = ctx.cfg.raft_max_size_per_msg.0;
    }

    /// Update states of the peer which can be changed in the previous raft
    /// tick.
    pub fn post_raft_group_tick(&mut self) {
        self.lead_transferee = self.raft_group.raft.lead_transferee.unwrap_or_default();
    }
}

/// `RequestPolicy` decides how we handle a request.
#[derive(Clone, PartialEq, Debug)]
pub enum RequestPolicy {
    // Handle the read request directly without dispatch.
    ReadLocal,
    StaleRead,
    // Handle the read request via raft's SafeReadIndex mechanism.
    ReadIndex,
    ProposeNormal,
    ProposeTransferLeader,
    ProposeConfChange,
}

/// `RequestInspector` makes `RequestPolicy` for requests.
pub trait RequestInspector {
    /// Has the current term been applied?
    fn has_applied_to_current_term(&mut self) -> bool;
    /// Inspects its lease.
    fn inspect_lease(&mut self) -> LeaseState;

    /// Inspect a request, return a policy that tells us how to
    /// handle the request.
    fn inspect(&mut self, req: &RaftCmdRequest) -> Result<RequestPolicy> {
        if req.has_admin_request() {
            if apply::is_conf_change_cmd(req) {
                return Ok(RequestPolicy::ProposeConfChange);
            }
            if get_transfer_leader_cmd(req).is_some()
                && !WriteBatchFlags::from_bits_truncate(req.get_header().get_flags())
                    .contains(WriteBatchFlags::TRANSFER_LEADER_PROPOSAL)
            {
                return Ok(RequestPolicy::ProposeTransferLeader);
            }
            return Ok(RequestPolicy::ProposeNormal);
        }

        let mut has_read = false;
        let mut has_write = false;
        for r in req.get_requests() {
            match r.get_cmd_type() {
                CmdType::Get | CmdType::Snap | CmdType::ReadIndex => has_read = true,
                CmdType::Delete | CmdType::Put | CmdType::DeleteRange | CmdType::IngestSst => {
                    has_write = true
                }
                CmdType::Prewrite | CmdType::Invalid => {
                    return Err(box_err!(
                        "invalid cmd type {:?}, message maybe corrupted",
                        r.get_cmd_type()
                    ));
                }
            }

            if has_read && has_write {
                return Err(box_err!("read and write can't be mixed in one batch"));
            }
        }

        if has_write {
            return Ok(RequestPolicy::ProposeNormal);
        }

        fail_point!("perform_read_index", |_| Ok(RequestPolicy::ReadIndex));

        fail_point!("perform_read_local", |_| Ok(RequestPolicy::ReadLocal));

        let flags = WriteBatchFlags::from_bits_check(req.get_header().get_flags());
        if flags.contains(WriteBatchFlags::STALE_READ) {
            return Ok(RequestPolicy::StaleRead);
        }

        if req.get_header().get_read_quorum() {
            return Ok(RequestPolicy::ReadIndex);
        }

        // If applied index's term differs from current raft's term, leader
        // transfer must happened, if read locally, we may read old value.
        if !self.has_applied_to_current_term() {
            return Ok(RequestPolicy::ReadIndex);
        }

        // Local read should be performed, if and only if leader is in lease.
        // None for now.
        match self.inspect_lease() {
            LeaseState::Valid => Ok(RequestPolicy::ReadLocal),
            LeaseState::Expired | LeaseState::Suspect => {
                // Perform a consistent read to Raft quorum and try to renew the leader lease.
                Ok(RequestPolicy::ReadIndex)
            }
        }
    }
}

impl<EK, ER> RequestInspector for Peer<EK, ER>
where
    EK: KvEngine,
    ER: RaftEngine,
{
    fn has_applied_to_current_term(&mut self) -> bool {
        self.get_store().applied_term() == self.term()
    }

    fn inspect_lease(&mut self) -> LeaseState {
        if !self.raft_group.raft.in_lease() {
            return LeaseState::Suspect;
        }
        // None means now.
        let state = self.leader_lease.inspect(None);
        if LeaseState::Expired == state {
            debug!(
                "leader lease is expired";
                "region_id" => self.region_id,
                "peer_id" => self.peer.get_id(),
                "lease" => ?self.leader_lease,
            );
            // The lease is expired, call `expire` explicitly.
            self.leader_lease.expire();
        }
        state
    }
}

impl<EK, ER, T> ReadExecutor for PollContext<EK, ER, T>
where
    EK: KvEngine,
    ER: RaftEngine,
{
    type Tablet = EK;

    fn get_tablet(&mut self) -> &EK {
        &self.engines.kv
    }

    fn get_snapshot(&mut self, _: &Option<LocalReadContext<'_, EK>>) -> Arc<EK::Snapshot> {
        Arc::new(self.engines.kv.snapshot())
    }
}

fn get_transfer_leader_cmd(msg: &RaftCmdRequest) -> Option<&TransferLeaderRequest> {
    if !msg.has_admin_request() {
        return None;
    }
    let req = msg.get_admin_request();
    if !req.has_transfer_leader() {
        return None;
    }

    Some(req.get_transfer_leader())
}

pub fn get_sync_log_from_request(msg: &RaftCmdRequest) -> bool {
    if msg.has_admin_request() {
        let req = msg.get_admin_request();
        return matches!(
            req.get_cmd_type(),
            AdminCmdType::ChangePeer
                | AdminCmdType::ChangePeerV2
                | AdminCmdType::Split
                | AdminCmdType::BatchSplit
                | AdminCmdType::PrepareMerge
                | AdminCmdType::CommitMerge
                | AdminCmdType::RollbackMerge
                | AdminCmdType::PrepareFlashback
                | AdminCmdType::FinishFlashback
        );
    }

    msg.get_header().get_sync_log()
}

/// We enable follower lazy commit to get a better performance.
/// But it may not be appropriate for some requests. This function

// The Raft message context for a MsgTransferLeader if it is a reply of a
// TransferLeader command.
pub const TRANSFER_LEADER_COMMAND_REPLY_CTX: &[u8] = &[1];

mod memtrace {
    use std::mem;

    use tikv_util::memory::HeapSize;

    use super::*;

    impl<EK, ER> Peer<EK, ER>
    where
        EK: KvEngine,
        ER: RaftEngine,
    {
        pub fn proposal_size(&self) -> usize {
            let mut heap_size = self.pending_reads.heap_size();
            for prop in &self.proposals.queue {
                heap_size += prop.heap_size();
            }
            heap_size
        }

        pub fn rest_size(&self) -> usize {
            // 2 words for every item in `peer_heartbeats`.
            16 * self.peer_heartbeats.capacity()
            // 2 words for every item in `peers_start_pending_time`.
            + 16 * self.peers_start_pending_time.capacity()
            // 1 word for every item in `down_peer_ids`
            + 8 * self.down_peer_ids.capacity()
            + mem::size_of::<metapb::Peer>() * self.check_stale_peers.capacity()
            // 1 word for every item in `want_rollback_merge_peers`
            + 8 * self.want_rollback_merge_peers.capacity()
            // Ignore more heap content in `raft::eraftpb::Message`.
            + (self.unpersisted_message_count
                + self.apply_snap_ctx.as_ref().map_or(0, |ctx| ctx.msgs.len()))
                * mem::size_of::<eraftpb::Message>()
            + mem::size_of_val(self.pending_request_snapshot_count.as_ref())
        }
    }
}
