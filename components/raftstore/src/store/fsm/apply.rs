// Copyright 2017 TiKV Project Authors. Licensed under Apache-2.0.

// #[PerformanceCriticalPath]
use std::{
    borrow::Cow,
    cmp,
    cmp::Ord,
    collections::VecDeque,
    fmt::{self, Debug, Formatter},
    io::BufRead,
    mem,
    ops::{Deref, DerefMut},
    sync::{
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
        mpsc::SyncSender,
        Arc,
    },
    usize,
};

use batch_system::{BasicMailbox, BatchRouter, BatchSystem, Fsm, Priority};
use collections::HashMap;
use crossbeam::channel::TryRecvError;
use engine_traits::{
    util::SequenceNumber, KvEngine, RaftEngine, RaftEngineReadOnly, Snapshot, SstMetaInfo,
};
use kvproto::{
    metapb::{Region, RegionEpoch},
    raft_cmdpb::{
        AdminCmdType, ChangePeerRequest, CommitMergeRequest, RaftCmdRequest, RaftCmdResponse,
    },
    raft_serverpb::{MergeState, RaftApplyState, RaftTruncatedState},
};
use pd_client::{BucketMeta, BucketStat};
use protobuf::{wire_format::WireType, CodedInputStream};
use raft::eraftpb::{ConfChangeType, ConfChangeV2, Entry, EntryType, Snapshot as RaftSnapshot};
use smallvec::SmallVec;
use tikv_alloc::trace::TraceEvent;
use tikv_util::{
    box_err, box_try, debug, error, info,
    memory::HeapSize,
    mpsc::{loose_bounded, LooseBoundedSender, Receiver},
    safe_panic,
    time::Instant,
    worker::Scheduler,
    MustConsumeVec,
};
use time::Timespec;

use crate::{
    bytes_capacity,
    coprocessor::{Cmd, CmdBatch, CmdObserveInfo, ObserveHandle, ObserveLevel},
    store::{
        cmd_resp,
        entry_storage::CachedEntries,
        local_metrics::{RaftMetrics, TimeTracker},
        memory::*,
        metrics::*,
        msg::{Callback, ErrorCallback, PeerMsg},
        peer::Peer,
        util::LatencyInspector,
        RegionTask, WriteCallback,
    },
    Error, Result,
};

// These consts are shared in both v1 and v2.
pub const DEFAULT_APPLY_WB_SIZE: usize = 4 * 1024;
pub const APPLY_WB_SHRINK_SIZE: usize = 1024 * 1024;
pub const SHRINK_PENDING_CMD_QUEUE_CAP: usize = 64;
pub const MAX_APPLY_BATCH_SIZE: usize = 64 * 1024 * 1024;

#[allow(dead_code)]
pub struct PendingCmd<C> {
    pub index: u64,
    pub term: u64,
    pub cb: Option<C>,
}

#[allow(dead_code)]
impl<C> PendingCmd<C> {
    fn new(index: u64, term: u64, cb: C) -> PendingCmd<C> {
        PendingCmd {
            index,
            term,
            cb: Some(cb),
        }
    }
}

impl<C> Drop for PendingCmd<C> {
    fn drop(&mut self) {
        if self.cb.is_some() {
            safe_panic!(
                "callback of pending command at [index: {}, term: {}] is leak",
                self.index,
                self.term
            );
        }
    }
}

impl<C> Debug for PendingCmd<C> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "PendingCmd [index: {}, term: {}, has_cb: {}]",
            self.index,
            self.term,
            self.cb.is_some()
        )
    }
}

impl<C> HeapSize for PendingCmd<C> {}

/// Commands waiting to be committed and applied.
#[derive(Debug)]
#[allow(dead_code)]
pub struct PendingCmdQueue<C> {
    normals: VecDeque<PendingCmd<C>>,
    conf_change: Option<PendingCmd<C>>,
}

#[allow(dead_code)]
impl<C> PendingCmdQueue<C> {
    fn new() -> PendingCmdQueue<C> {
        PendingCmdQueue {
            normals: VecDeque::new(),
            conf_change: None,
        }
    }

    fn pop_normal(&mut self, index: u64, term: u64) -> Option<PendingCmd<C>> {
        self.normals.pop_front().and_then(|cmd| {
            if self.normals.capacity() > SHRINK_PENDING_CMD_QUEUE_CAP
                && self.normals.len() < SHRINK_PENDING_CMD_QUEUE_CAP
            {
                self.normals.shrink_to_fit();
            }
            if (cmd.term, cmd.index) > (term, index) {
                self.normals.push_front(cmd);
                return None;
            }
            Some(cmd)
        })
    }

    fn append_normal(&mut self, cmd: PendingCmd<C>) {
        self.normals.push_back(cmd);
    }

    fn take_conf_change(&mut self) -> Option<PendingCmd<C>> {
        // conf change will not be affected when changing between follower and leader,
        // so there is no need to check term.
        self.conf_change.take()
    }

    // TODO: seems we don't need to separate conf change from normal entries.
    fn set_conf_change(&mut self, cmd: PendingCmd<C>) {
        self.conf_change = Some(cmd);
    }
}

#[derive(Default, Debug)]
pub struct ChangePeer {
    pub index: u64,
    // The proposed ConfChangeV2 or (legacy) ConfChange
    // ConfChange (if it is) will convert to ConfChangeV2
    pub conf_change: ConfChangeV2,
    // The change peer requests come along with ConfChangeV2
    // or (legacy) ConfChange, for ConfChange, it only contains
    // one element
    pub changes: Vec<ChangePeerRequest>,
    pub region: Region,
}

#[allow(dead_code)]
pub struct Range {
    pub cf: String,
    pub start_key: Vec<u8>,
    pub end_key: Vec<u8>,
}

impl Debug for Range {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{{ cf: {:?}, start_key: {:?}, end_key: {:?} }}",
            self.cf,
            log_wrappers::Value::key(&self.start_key),
            log_wrappers::Value::key(&self.end_key)
        )
    }
}

#[allow(dead_code)]
impl Range {
    fn new(cf: String, start_key: Vec<u8>, end_key: Vec<u8>) -> Range {
        Range {
            cf,
            start_key,
            end_key,
        }
    }
}

#[derive(Debug)]
pub enum ExecResult<S> {
    ChangePeer(ChangePeer),
    CompactLog {
        state: RaftTruncatedState,
        first_index: u64,
    },
    SplitRegion {
        regions: Vec<Region>,
        derived: Region,
        new_split_regions: HashMap<u64, NewSplitPeer>,
    },
    PrepareMerge {
        region: Region,
        state: MergeState,
    },
    CommitMerge {
        index: u64,
        region: Region,
        source: Region,
    },
    RollbackMerge {
        region: Region,
        commit: u64,
    },
    ComputeHash {
        region: Region,
        index: u64,
        context: Vec<u8>,
        snap: S,
    },
    VerifyHash {
        index: u64,
        context: Vec<u8>,
        hash: Vec<u8>,
    },
    DeleteRange {
        ranges: Vec<Range>,
    },
    IngestSst {
        ssts: Vec<SstMetaInfo>,
    },
    TransferLeader {
        term: u64,
    },
    SetFlashbackState {
        region: Region,
    },
}

/// The possible returned value when applying logs.
#[derive(Debug)]
pub enum ApplyResult<S> {
    None,
    Yield,
    /// Additional result that needs to be sent back to raftstore.
    Res(ExecResult<S>),
    /// It is unable to apply the `CommitMerge` until the source peer
    /// has applied to the required position and sets the atomic boolean
    /// to true.
    WaitMergeSource(Arc<AtomicU64>),
}

// The applied command and their callback
#[allow(dead_code)]
struct ApplyCallbackBatch<S>
where
    S: Snapshot,
{
    cmd_batch: Vec<CmdBatch>,
    // The max observe level of current `Vec<CmdBatch>`
    batch_max_level: ObserveLevel,
    cb_batch: MustConsumeVec<(Callback<S>, RaftCmdResponse)>,
}

#[allow(dead_code)]
impl<S: Snapshot> ApplyCallbackBatch<S> {
    fn new() -> ApplyCallbackBatch<S> {
        ApplyCallbackBatch {
            cmd_batch: vec![],
            batch_max_level: ObserveLevel::None,
            cb_batch: MustConsumeVec::new("callback of apply callback batch"),
        }
    }

    fn push_batch(&mut self, observe_info: &CmdObserveInfo, region_id: u64) {
        let cb = CmdBatch::new(observe_info, region_id);
        self.batch_max_level = cmp::max(self.batch_max_level, cb.level);
        self.cmd_batch.push(cb);
    }

    fn push_cb(&mut self, cb: Callback<S>, resp: RaftCmdResponse) {
        self.cb_batch.push((cb, resp));
    }

    fn push(
        &mut self,
        cb: Option<Callback<S>>,
        cmd: Cmd,
        observe_info: &CmdObserveInfo,
        region_id: u64,
    ) {
        if let Some(cb) = cb {
            self.cb_batch.push((cb, cmd.response.clone()));
        }
        self.cmd_batch
            .last_mut()
            .unwrap()
            .push(observe_info, region_id, cmd);
    }
}

pub trait Notifier<EK: KvEngine>: Send {
    fn notify(&self, apply_res: Vec<ApplyRes<EK::Snapshot>>);
    fn notify_one(&self, region_id: u64, msg: PeerMsg<EK>);
    fn clone_box(&self) -> Box<dyn Notifier<EK>>;
}

/// Calls the callback of `cmd` when the Region is removed.
#[allow(dead_code)]
fn notify_region_removed(region_id: u64, peer_id: u64, mut cmd: PendingCmd<impl ErrorCallback>) {
    debug!(
        "region is removed, notify commands";
        "region_id" => region_id,
        "peer_id" => peer_id,
        "index" => cmd.index,
        "term" => cmd.term
    );
    notify_req_region_removed(region_id, cmd.cb.take().unwrap());
}

pub fn notify_req_region_removed(region_id: u64, cb: impl ErrorCallback) {
    let region_not_found = Error::RegionNotFound(region_id);
    let resp = cmd_resp::new_error(region_not_found);
    cb.report_error(resp);
}

/// Calls the callback of `cmd` when it can not be processed further.
#[allow(dead_code)]
fn notify_stale_command(
    region_id: u64,
    peer_id: u64,
    term: u64,
    mut cmd: PendingCmd<impl ErrorCallback>,
) {
    info!(
        "command is stale, skip";
        "region_id" => region_id,
        "peer_id" => peer_id,
        "index" => cmd.index,
        "term" => cmd.term
    );
    notify_stale_req(term, cmd.cb.take().unwrap());
}

pub fn notify_stale_req(term: u64, cb: impl ErrorCallback) {
    let resp = cmd_resp::err_resp(Error::StaleCommand, term);
    cb.report_error(resp);
}

pub fn notify_stale_req_with_msg(term: u64, msg: String, cb: impl ErrorCallback) {
    let mut resp = cmd_resp::err_resp(Error::StaleCommand, term);
    resp.mut_header().mut_error().set_message(msg);
    cb.report_error(resp);
}

/// Checks if a write is needed to be issued before handling the command.
#[allow(dead_code)]
fn should_write_to_engine(cmd: &RaftCmdRequest) -> bool {
    if cmd.has_admin_request() {
        match cmd.get_admin_request().get_cmd_type() {
            // ComputeHash require an up to date snapshot.
            AdminCmdType::ComputeHash |
            // Merge needs to get the latest apply index.
            AdminCmdType::CommitMerge |
            AdminCmdType::RollbackMerge => return true,
            _ => {}
        }
    }

    // Some commands may modify keys covered by the current write batch, so we
    // must write the current write batch to the engine first.
    for req in cmd.get_requests() {
        if req.has_delete_range() {
            return true;
        }
        if req.has_ingest_sst() {
            return true;
        }
    }

    false
}

/// Checks if a write has high-latency operation.
#[allow(dead_code)]
fn has_high_latency_operation(cmd: &RaftCmdRequest) -> bool {
    for req in cmd.get_requests() {
        if req.has_delete_range() {
            return true;
        }
        if req.has_ingest_sst() {
            return true;
        }
    }
    false
}

/// Checks if a write is needed to be issued after handling the command.
#[allow(dead_code)]
fn should_sync_log(cmd: &RaftCmdRequest) -> bool {
    if cmd.has_admin_request() {
        if cmd.get_admin_request().get_cmd_type() == AdminCmdType::CompactLog {
            // We do not need to sync WAL before compact log, because this request will send
            // a msg to raft_gc_log thread to delete the entries before this
            // index instead of deleting them in apply thread directly.
            return false;
        }
        return true;
    }

    for req in cmd.get_requests() {
        // After ingest sst, sst files are deleted quickly. As a result,
        // ingest sst command can not be handled again and must be synced.
        // See more in Cleanup worker.
        if req.has_ingest_sst() {
            return true;
        }
    }

    false
}

#[allow(dead_code)]
fn can_witness_skip(entry: &Entry) -> bool {
    // need to handle ConfChange entry type
    if entry.get_entry_type() != EntryType::EntryNormal {
        return false;
    }

    // HACK: check admin request field in serialized data from `RaftCmdRequest`
    // without deserializing all. It's done by checking the existence of the
    // field number of `admin_request`.
    // See the encoding in `write_to_with_cached_sizes()` of `RaftCmdRequest` in
    // `raft_cmdpb.rs` for reference.
    let mut is = CodedInputStream::from_bytes(entry.get_data());
    if is.eof().unwrap() {
        return true;
    }
    let (mut field_number, wire_type) = is.read_tag_unpack().unwrap();
    // Header field is of number 1
    if field_number == 1 {
        if wire_type != WireType::WireTypeLengthDelimited {
            panic!("unexpected wire type");
        }
        let len = is.read_raw_varint32().unwrap();
        // skip parsing the content of `Header`
        is.consume(len as usize);
        // read next field number
        (field_number, _) = is.read_tag_unpack().unwrap();
    }

    // `Requests` field is of number 2 and `AdminRequest` field is of number 3.
    // - If the next field is 2, there must be no admin request as in one
    //   `RaftCmdRequest`, either requests or admin_request is filled.
    // - If the next field is 3, it's exactly an admin request.
    // - If the next field is others, neither requests nor admin_request is filled,
    //   so there is no admin request.
    field_number != 3
}

/// A struct that stores the state related to Merge.
///
/// When executing a `CommitMerge`, the source peer may have not applied
/// to the required index, so the target peer has to abort current execution
/// and wait for it asynchronously.
///
/// When rolling the stack, all states required to recover are stored in
/// this struct.
/// TODO: check whether generator/coroutine is a good choice in this case.
struct WaitSourceMergeState {
    /// A flag that indicates whether the source peer has applied to the
    /// required index. If the source peer is ready, this flag should be set
    /// to the region id of source peer.
    logs_up_to_date: Arc<AtomicU64>,
}

struct YieldState<EK>
where
    EK: KvEngine,
{
    /// All of the entries that need to continue to be applied after
    /// the source peer has applied its logs.
    pending_entries: Vec<Entry>,
    /// All of messages that need to continue to be handled after
    /// the source peer has applied its logs and pending entries
    /// are all handled.
    pending_msgs: Vec<Msg<EK>>,

    /// Cache heap size for itself.
    #[allow(dead_code)]
    heap_size: Option<usize>,
}

impl<EK> Debug for YieldState<EK>
where
    EK: KvEngine,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("YieldState")
            .field("pending_entries", &self.pending_entries.len())
            .field("pending_msgs", &self.pending_msgs.len())
            .finish()
    }
}

impl Debug for WaitSourceMergeState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WaitSourceMergeState")
            .field("logs_up_to_date", &self.logs_up_to_date)
            .finish()
    }
}

#[derive(Debug, Clone)]
pub struct NewSplitPeer {
    pub peer_id: u64,
    // `None` => success,
    // `Some(s)` => fail due to `s`.
    pub result: Option<String>,
}

#[allow(dead_code)]
mod confchange_cmd_metric {
    use super::*;

    pub fn inc_all(cct: ConfChangeType) {
        let metrics = match cct {
            ConfChangeType::AddNode => &PEER_ADMIN_CMD_COUNTER.add_peer,
            ConfChangeType::RemoveNode => &PEER_ADMIN_CMD_COUNTER.remove_peer,
            ConfChangeType::AddLearnerNode => &PEER_ADMIN_CMD_COUNTER.add_learner,
        };
        metrics.all.inc();
    }

    pub fn inc_success(cct: ConfChangeType) {
        let metrics = match cct {
            ConfChangeType::AddNode => &PEER_ADMIN_CMD_COUNTER.add_peer,
            ConfChangeType::RemoveNode => &PEER_ADMIN_CMD_COUNTER.remove_peer,
            ConfChangeType::AddLearnerNode => &PEER_ADMIN_CMD_COUNTER.add_learner,
        };
        metrics.success.inc();
    }
}

pub fn is_conf_change_cmd(msg: &RaftCmdRequest) -> bool {
    if !msg.has_admin_request() {
        return false;
    }
    let req = msg.get_admin_request();
    req.has_change_peer() || req.has_change_peer_v2()
}

/// Updates the `state` with given `compact_index` and `compact_term`.
///
/// Remember the Raft log is not deleted here.
pub fn compact_raft_log(
    tag: &str,
    state: &mut RaftApplyState,
    compact_index: u64,
    compact_term: u64,
) -> Result<()> {
    debug!("{} compact log entries to prior to {}", tag, compact_index);

    if compact_index <= state.get_truncated_state().get_index() {
        return Err(box_err!("try to truncate compacted entries"));
    } else if compact_index > state.get_applied_index() {
        return Err(box_err!(
            "compact index {} > applied index {}",
            compact_index,
            state.get_applied_index()
        ));
    }

    // we don't actually delete the logs now, we add an async task to do it.

    state.mut_truncated_state().set_index(compact_index);
    state.mut_truncated_state().set_term(compact_term);

    Ok(())
}

pub struct Apply<C> {
    pub peer_id: u64,
    pub region_id: u64,
    pub term: u64,
    pub commit_index: u64,
    pub commit_term: u64,
    pub entries: SmallVec<[CachedEntries; 1]>,
    pub entries_size: usize,
    pub cbs: Vec<Proposal<C>>,
    pub bucket_meta: Option<Arc<BucketMeta>>,
}

impl<C: WriteCallback> Apply<C> {
    pub fn on_schedule(&mut self, metrics: &RaftMetrics) {
        let now = std::time::Instant::now();
        for cb in &mut self.cbs {
            if let Some(trackers) = cb.cb.write_trackers_mut() {
                for tracker in trackers {
                    tracker.observe(now, &metrics.store_time, |t| {
                        t.metrics.write_instant = Some(now);
                        &mut t.metrics.store_time_nanos
                    });
                    if let TimeTracker::Instant(t) = tracker {
                        *t = now;
                    }
                }
            }
        }
    }

    #[allow(dead_code)]
    fn try_batch(&mut self, other: &mut Apply<C>) -> bool {
        assert_eq!(self.region_id, other.region_id);
        assert_eq!(self.peer_id, other.peer_id);
        if self.entries_size + other.entries_size <= MAX_APPLY_BATCH_SIZE {
            if other.bucket_meta.is_some() {
                self.bucket_meta = other.bucket_meta.take();
            }

            assert!(other.term >= self.term);
            self.term = other.term;

            assert!(other.commit_index >= self.commit_index);
            self.commit_index = other.commit_index;
            assert!(other.commit_term >= self.commit_term);
            self.commit_term = other.commit_term;

            self.entries.append(&mut other.entries);
            self.entries_size += other.entries_size;

            self.cbs.append(&mut other.cbs);
            true
        } else {
            false
        }
    }
}

pub struct Registration {
    pub id: u64,
    pub term: u64,
    pub apply_state: RaftApplyState,
    pub applied_term: u64,
    pub region: Region,
    pub pending_request_snapshot_count: Arc<AtomicUsize>,
    pub is_merging: bool,
    #[allow(dead_code)]
    raft_engine: Box<dyn RaftEngineReadOnly>,
}

impl Registration {
    pub fn new<EK: KvEngine, ER: RaftEngine>(peer: &Peer<EK, ER>) -> Registration {
        Registration {
            id: peer.peer_id(),
            term: peer.term(),
            apply_state: peer.get_store().apply_state().clone(),
            applied_term: peer.get_store().applied_term(),
            region: peer.region().clone(),
            pending_request_snapshot_count: peer.pending_request_snapshot_count.clone(),
            is_merging: peer.pending_merge_state.is_some(),
            raft_engine: Box::new(peer.get_store().engines.raft.clone()),
        }
    }
}

#[derive(Debug)]
pub struct Proposal<C> {
    pub is_conf_change: bool,
    pub index: u64,
    pub term: u64,
    pub cb: C,
    /// `propose_time` is set to the last time when a peer starts to renew
    /// lease.
    pub propose_time: Option<Timespec>,
    pub must_pass_epoch_check: bool,
}

impl<C> Proposal<C> {
    pub fn new(index: u64, term: u64, cb: C) -> Self {
        Self {
            index,
            term,
            cb,
            propose_time: None,
            must_pass_epoch_check: false,
            is_conf_change: false,
        }
    }
}

impl<C> HeapSize for Proposal<C> {}

pub struct Destroy {
    region_id: u64,
    #[allow(dead_code)]
    merge_from_snapshot: bool,
}

/// A message that asks the delegate to apply to the given logs and then reply
/// to target mailbox.
#[derive(Default, Debug)]
pub struct CatchUpLogs {
    /// The target region to be notified when given logs are applied.
    pub target_region_id: u64,
    /// Merge request that contains logs to be applied.
    pub merge: CommitMergeRequest,
    /// A flag indicate that all source region's logs are applied.
    ///
    /// This is still necessary although we have a mailbox field already.
    /// Mailbox is used to notify target region, and trigger a round of polling.
    /// But due to the FIFO natural of channel, we need a flag to check if it's
    /// ready when polling.
    pub logs_up_to_date: Arc<AtomicU64>,
}

pub struct GenSnapTask {
    pub(crate) region_id: u64,
    // Fill it after the RocksDB snapshot is taken.
    pub index: Arc<AtomicU64>,
    // Fetch it to cancel the task if necessary.
    pub canceled: Arc<AtomicBool>,

    snap_notifier: SyncSender<RaftSnapshot>,
    // indicates whether the snapshot is triggered due to load balance
    for_balance: bool,
    // the store id the snapshot will be sent to
    to_store_id: u64,
}

impl GenSnapTask {
    pub fn new(
        region_id: u64,
        index: Arc<AtomicU64>,
        canceled: Arc<AtomicBool>,
        snap_notifier: SyncSender<RaftSnapshot>,
        to_store_id: u64,
    ) -> GenSnapTask {
        GenSnapTask {
            region_id,
            index,
            canceled,
            snap_notifier,
            for_balance: false,
            to_store_id,
        }
    }

    pub fn set_for_balance(&mut self) {
        self.for_balance = true;
    }

    pub fn generate_and_schedule_snapshot<EK>(
        self,
        kv_snap: EK::Snapshot,
        last_applied_term: u64,
        last_applied_state: RaftApplyState,
        region_sched: &Scheduler<RegionTask<EK::Snapshot>>,
    ) -> Result<()>
    where
        EK: KvEngine,
    {
        self.index
            .store(last_applied_state.applied_index, Ordering::SeqCst);
        let snapshot = RegionTask::Gen {
            region_id: self.region_id,
            notifier: self.snap_notifier,
            for_balance: self.for_balance,
            last_applied_term,
            last_applied_state,
            canceled: self.canceled,
            // This snapshot may be held for a long time, which may cause too many
            // open files in rocksdb.
            kv_snap,
            to_store_id: self.to_store_id,
        };
        box_try!(region_sched.schedule(snapshot));
        Ok(())
    }
}

impl Debug for GenSnapTask {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GenSnapTask")
            .field("region_id", &self.region_id)
            .finish()
    }
}

#[derive(Debug)]
enum ObserverType {
    Cdc(ObserveHandle),
    Rts(ObserveHandle),
    Pitr(ObserveHandle),
}

impl ObserverType {
    #[allow(dead_code)]
    fn handle(&self) -> &ObserveHandle {
        match self {
            ObserverType::Cdc(h) => h,
            ObserverType::Rts(h) => h,
            ObserverType::Pitr(h) => h,
        }
    }
}

#[derive(Debug)]
pub struct ChangeObserver {
    #[allow(dead_code)]
    ty: ObserverType,
    region_id: u64,
}

impl ChangeObserver {
    pub fn from_cdc(region_id: u64, id: ObserveHandle) -> Self {
        Self {
            ty: ObserverType::Cdc(id),
            region_id,
        }
    }

    pub fn from_rts(region_id: u64, id: ObserveHandle) -> Self {
        Self {
            ty: ObserverType::Rts(id),
            region_id,
        }
    }

    pub fn from_pitr(region_id: u64, id: ObserveHandle) -> Self {
        Self {
            ty: ObserverType::Pitr(id),
            region_id,
        }
    }
}

pub enum Msg<EK>
where
    EK: KvEngine,
{
    Apply {
        start: Instant,
        apply: Apply<Callback<EK::Snapshot>>,
    },
    Registration(Registration),
    LogsUpToDate(CatchUpLogs),
    Noop,
    Destroy(Destroy),
    Snapshot(GenSnapTask),
    Change {
        cmd: ChangeObserver,
        region_epoch: RegionEpoch,
        cb: Callback<EK::Snapshot>,
    },
    #[cfg(any(test, feature = "testexport"))]
    #[allow(clippy::type_complexity)]
    Validate(u64, Box<dyn FnOnce(*const u8) + Send>),
}

impl<EK> Msg<EK>
where
    EK: KvEngine,
{
    pub fn apply(apply: Apply<Callback<EK::Snapshot>>) -> Msg<EK> {
        Msg::Apply {
            start: Instant::now(),
            apply,
        }
    }

    pub fn register<ER: RaftEngine>(peer: &Peer<EK, ER>) -> Msg<EK> {
        Msg::Registration(Registration::new(peer))
    }

    pub fn destroy(region_id: u64, merge_from_snapshot: bool) -> Msg<EK> {
        Msg::Destroy(Destroy {
            region_id,
            merge_from_snapshot,
        })
    }
}

impl<EK> Debug for Msg<EK>
where
    EK: KvEngine,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Msg::Apply { apply, .. } => write!(f, "[region {}] async apply", apply.region_id),
            Msg::Registration(ref r) => {
                write!(f, "[region {}] Reg {:?}", r.region.get_id(), r.apply_state)
            }
            Msg::LogsUpToDate(_) => write!(f, "logs are updated"),
            Msg::Noop => write!(f, "noop"),
            Msg::Destroy(ref d) => write!(f, "[region {}] destroy", d.region_id),
            Msg::Snapshot(GenSnapTask { region_id, .. }) => {
                write!(f, "[region {}] requests a snapshot", region_id)
            }
            Msg::Change {
                cmd: ChangeObserver { region_id, .. },
                ..
            } => write!(f, "[region {}] change cmd", region_id),
            #[cfg(any(test, feature = "testexport"))]
            Msg::Validate(region_id, _) => write!(f, "[region {}] validate", region_id),
        }
    }
}

#[derive(Default, Clone, Debug, PartialEq)]
pub struct ApplyMetrics {
    /// an inaccurate difference in region size since last reset.
    pub size_diff_hint: i64,
    /// delete keys' count since last reset.
    pub delete_keys_hint: u64,

    pub written_bytes: u64,
    pub written_keys: u64,
    pub lock_cf_written_bytes: u64,
}

#[derive(Debug)]
pub struct ApplyRes<S>
where
    S: Snapshot,
{
    pub region_id: u64,
    pub apply_state: RaftApplyState,
    pub applied_term: u64,
    pub exec_res: VecDeque<ExecResult<S>>,
    pub metrics: ApplyMetrics,
    pub bucket_stat: Option<Box<BucketStat>>,
    pub write_seqno: Vec<SequenceNumber>,
}

#[derive(Debug)]
pub enum TaskRes<S>
where
    S: Snapshot,
{
    Apply(ApplyRes<S>),
    Destroy {
        // ID of region that has been destroyed.
        region_id: u64,
        // ID of peer that has been destroyed.
        peer_id: u64,
        // Whether destroy request is from its target region's snapshot
        merge_from_snapshot: bool,
    },
}

pub struct ApplyFsm<EK>
where
    EK: KvEngine,
{
    #[allow(dead_code)]
    receiver: Receiver<Msg<EK>>,
    mailbox: Option<BasicMailbox<ApplyFsm<EK>>>,
}

impl<EK> Fsm for ApplyFsm<EK>
where
    EK: KvEngine,
{
    type Message = Msg<EK>;

    #[inline]
    fn is_stopped(&self) -> bool {
        true
    }

    #[inline]
    fn set_mailbox(&mut self, mailbox: Cow<'_, BasicMailbox<Self>>)
    where
        Self: Sized,
    {
        self.mailbox = Some(mailbox.into_owned());
    }

    #[inline]
    fn take_mailbox(&mut self) -> Option<BasicMailbox<Self>>
    where
        Self: Sized,
    {
        self.mailbox.take()
    }

    #[inline]
    fn get_priority(&self) -> Priority {
        unimplemented!()
    }
}

pub enum ControlMsg {
    LatencyInspect {
        send_time: Instant,
        inspector: LatencyInspector,
    },
}

pub struct ControlFsm {
    receiver: Receiver<ControlMsg>,
    stopped: bool,
}

impl ControlFsm {
    pub fn new() -> (LooseBoundedSender<ControlMsg>, Box<ControlFsm>) {
        let (tx, rx) = loose_bounded(std::usize::MAX);
        let fsm = Box::new(ControlFsm {
            stopped: false,
            receiver: rx,
        });
        (tx, fsm)
    }

    pub fn handle_messages(&mut self, pending_latency_inspect: &mut Vec<LatencyInspector>) {
        // Usually there will be only 1 control message.
        loop {
            match self.receiver.try_recv() {
                Ok(ControlMsg::LatencyInspect {
                    send_time,
                    mut inspector,
                }) => {
                    inspector.record_apply_wait(send_time.saturating_elapsed());
                    pending_latency_inspect.push(inspector);
                }
                Err(TryRecvError::Empty) => {
                    return;
                }
                Err(TryRecvError::Disconnected) => {
                    self.stopped = true;
                    return;
                }
            }
        }
    }
}

impl Fsm for ControlFsm {
    type Message = ControlMsg;

    #[inline]
    fn is_stopped(&self) -> bool {
        self.stopped
    }
}

#[derive(Clone)]
pub struct ApplyRouter<EK>
where
    EK: KvEngine,
{
    pub router: BatchRouter<ApplyFsm<EK>, ControlFsm>,
}

impl<EK> Deref for ApplyRouter<EK>
where
    EK: KvEngine,
{
    type Target = BatchRouter<ApplyFsm<EK>, ControlFsm>;

    fn deref(&self) -> &BatchRouter<ApplyFsm<EK>, ControlFsm> {
        &self.router
    }
}

impl<EK> DerefMut for ApplyRouter<EK>
where
    EK: KvEngine,
{
    fn deref_mut(&mut self) -> &mut BatchRouter<ApplyFsm<EK>, ControlFsm> {
        &mut self.router
    }
}

impl<EK> ApplyRouter<EK>
where
    EK: KvEngine,
{
    pub fn schedule_task(&self, _region_id: u64, _msg: Msg<EK>) {
        unimplemented!()
    }

    pub fn register(&self, region_id: u64, mailbox: BasicMailbox<ApplyFsm<EK>>) {
        self.router.register(region_id, mailbox);
        self.update_trace();
    }

    pub fn register_all(&self, mailboxes: Vec<(u64, BasicMailbox<ApplyFsm<EK>>)>) {
        self.router.register_all(mailboxes);
        self.update_trace();
    }

    pub fn close(&self, region_id: u64) {
        self.router.close(region_id);
        self.update_trace();
    }

    fn update_trace(&self) {
        let router_trace = self.router.trace();
        MEMTRACE_APPLY_ROUTER_ALIVE.trace(TraceEvent::Reset(router_trace.alive));
        MEMTRACE_APPLY_ROUTER_LEAK.trace(TraceEvent::Reset(router_trace.leak));
    }
}

pub struct ApplyBatchSystem<EK: KvEngine> {
    system: BatchSystem<ApplyFsm<EK>, ControlFsm>,
}

impl<EK: KvEngine> Deref for ApplyBatchSystem<EK> {
    type Target = BatchSystem<ApplyFsm<EK>, ControlFsm>;

    fn deref(&self) -> &BatchSystem<ApplyFsm<EK>, ControlFsm> {
        &self.system
    }
}

impl<EK: KvEngine> DerefMut for ApplyBatchSystem<EK> {
    fn deref_mut(&mut self) -> &mut BatchSystem<ApplyFsm<EK>, ControlFsm> {
        &mut self.system
    }
}

impl<EK: KvEngine> ApplyBatchSystem<EK> {
    pub fn schedule_all<'a, ER: RaftEngine>(&self, _peers: impl Iterator<Item = &'a Peer<EK, ER>>) {
        unimplemented!()
    }
}

mod memtrace {
    use memory_trace_macros::MemoryTraceHelper;

    use super::*;

    #[derive(MemoryTraceHelper, Default, Debug)]
    pub struct ApplyMemoryTrace {
        pub pending_cmds: usize,
        pub merge_yield: usize,
    }

    impl<C> HeapSize for PendingCmdQueue<C> {
        fn heap_size(&self) -> usize {
            // Some fields of `PendingCmd` are on stack, but ignore them because they are
            // just some small boxed closures.
            self.normals.capacity() * mem::size_of::<PendingCmd<C>>()
        }
    }

    impl<EK> HeapSize for YieldState<EK>
    where
        EK: KvEngine,
    {
        fn heap_size(&self) -> usize {
            let mut size = self.pending_entries.capacity() * mem::size_of::<Entry>();
            for e in &self.pending_entries {
                size += bytes_capacity(&e.data) + bytes_capacity(&e.context);
            }

            size += self.pending_msgs.capacity() * mem::size_of::<Msg<EK>>();
            for msg in &self.pending_msgs {
                size += msg.heap_size();
            }

            size
        }
    }

    impl<EK> HeapSize for Msg<EK>
    where
        EK: KvEngine,
    {
        /// Only consider large fields in `Msg`.
        fn heap_size(&self) -> usize {
            match self {
                Msg::LogsUpToDate(l) => l.heap_size(),
                // For entries in `Msg::Apply`, heap size is already updated when fetching them
                // from `raft::Storage`. So use `0` here.
                Msg::Apply { .. } => 0,
                Msg::Registration(_)
                | Msg::Snapshot(_)
                | Msg::Destroy(_)
                | Msg::Noop
                | Msg::Change { .. } => 0,
                #[cfg(any(test, feature = "testexport"))]
                Msg::Validate(..) => 0,
            }
        }
    }

    impl HeapSize for CatchUpLogs {
        fn heap_size(&self) -> usize {
            let mut size: usize = 0;
            for e in &self.merge.entries {
                size += bytes_capacity(&e.data) + bytes_capacity(&e.context);
            }
            size
        }
    }
}
