// Copyright 2016 TiKV Project Authors. Licensed under Apache-2.0.

// #[PerformanceCriticalPath]
use std::{
    cell::Cell,
    collections::{
        BTreeMap,
        Bound::{Excluded, Unbounded},
    },
    marker::PhantomData,
    mem,
    ops::Deref,
    sync::{Arc, Mutex},
    time::Duration,
    u64,
};

use batch_system::{BasicMailbox, BatchRouter, Fsm};
use collections::{HashMap, HashMapEntry, HashSet};
use crossbeam::channel::TrySendError;
use engine_traits::{Engines, KvEngine, RaftEngine};
use fail::fail_point;
use futures::{compat::Future01CompatExt, FutureExt};
use keys::{self};
use kvproto::{
    metapb::{self, Region},
    raft_serverpb::RaftMessage,
};
use pd_client::{Feature, FeatureGate};
use sst_importer::SstImporter;
use tikv_alloc::trace::TraceEvent;
use tikv_util::{
    defer, error, future::poll_future_notify, sys::disk::DiskUsage, time::Instant as TiInstant,
    timer::SteadyTimer, warn, worker::Scheduler, Either, RingQueue,
};
use time::{self, Timespec};

use crate::{
    bytes_capacity,
    coprocessor::{CoprocessorHost, RegionChangeReason},
    store::{
        async_io::{read::ReadTask, write::Worker as WriteWorker, write_router::WriteSenders},
        config::Config,
        fsm::{metrics::*, peer::PeerFsm, ApplyNotifier, ApplyRes, ApplyRouter, ApplyTaskRes},
        local_metrics::RaftMetrics,
        memory::*,
        util,
        util::RegionReadProgressRegistry,
        worker::{ConsistencyCheckTask, RaftlogGcTask, ReadDelegate, RegionTask, SplitCheckTask},
        GlobalReplicationState, InspectedRaftMessage, PeerMsg, PeerTick, RaftCommand,
        SignificantMsg, SnapManager, StoreMsg,
    },
};

pub const PENDING_MSG_CAP: usize = 100;
const ENTRY_CACHE_EVICT_TICK_DURATION: Duration = Duration::from_secs(1);
pub const MULTI_FILES_SNAPSHOT_FEATURE: Feature = Feature::require(6, 1, 0); // it only makes sense for large region

pub struct StoreInfo<EK, ER> {
    pub kv_engine: EK,
    pub raft_engine: ER,
    pub capacity: u64,
}

pub struct StoreMeta {
    pub store_id: Option<u64>,
    /// region_end_key -> region_id
    pub region_ranges: BTreeMap<Vec<u8>, u64>,
    /// region_id -> region
    pub regions: HashMap<u64, Region>,
    /// region_id -> reader
    pub readers: HashMap<u64, ReadDelegate>,
    /// `MsgRequestPreVote`, `MsgRequestVote` or `MsgAppend` messages from newly
    /// split Regions shouldn't be dropped if there is no such Region in this
    /// store now. So the messages are recorded temporarily and will be handled
    /// later.
    pub pending_msgs: RingQueue<RaftMessage>,
    /// The regions with pending snapshots.
    pub pending_snapshot_regions: Vec<Region>,
    /// A marker used to indicate the peer of a Region has received a merge
    /// target message and waits to be destroyed. target_region_id ->
    /// (source_region_id -> merge_target_region)
    pub pending_merge_targets: HashMap<u64, HashMap<u64, metapb::Region>>,
    /// An inverse mapping of `pending_merge_targets` used to let source peer
    /// help target peer to clean up related entry. source_region_id ->
    /// target_region_id
    pub targets_map: HashMap<u64, u64>,
    /// `atomic_snap_regions` and `destroyed_region_for_snap` are used for
    /// making destroy overlapped regions and apply snapshot atomically.
    /// region_id -> wait_destroy_regions_map(source_region_id -> is_ready)
    /// A target peer must wait for all source peer to ready before applying
    /// snapshot.
    pub atomic_snap_regions: HashMap<u64, HashMap<u64, bool>>,
    /// source_region_id -> need_atomic
    /// Used for reminding the source peer to switch to ready in
    /// `atomic_snap_regions`.
    pub destroyed_region_for_snap: HashMap<u64, bool>,
    /// region_id -> `RegionReadProgress`
    pub region_read_progress: RegionReadProgressRegistry,
    /// record sst_file_name -> (sst_smallest_key, sst_largest_key)
    pub damaged_ranges: HashMap<String, (Vec<u8>, Vec<u8>)>,
}

impl StoreMeta {
    pub fn new(vote_capacity: usize) -> StoreMeta {
        StoreMeta {
            store_id: None,
            region_ranges: BTreeMap::default(),
            regions: HashMap::default(),
            readers: HashMap::default(),
            pending_msgs: RingQueue::with_capacity(vote_capacity),
            pending_snapshot_regions: Vec::default(),
            pending_merge_targets: HashMap::default(),
            targets_map: HashMap::default(),
            atomic_snap_regions: HashMap::default(),
            destroyed_region_for_snap: HashMap::default(),
            region_read_progress: RegionReadProgressRegistry::new(),
            damaged_ranges: HashMap::default(),
        }
    }

    #[inline]
    pub fn set_region<EK: KvEngine, ER: RaftEngine>(
        &mut self,
        host: &CoprocessorHost<EK>,
        region: Region,
        peer: &mut crate::store::Peer<EK, ER>,
        reason: RegionChangeReason,
    ) {
        let prev = self.regions.insert(region.get_id(), region.clone());
        if prev.map_or(true, |r| r.get_id() != region.get_id()) {
            // TODO: may not be a good idea to panic when holding a lock.
            panic!("{} region corrupted", peer.tag);
        }
        let reader = self.readers.get_mut(&region.get_id()).unwrap();
        peer.set_region(host, reader, region, reason);
    }

    /// Update damaged ranges and return true if overlap exists.
    ///
    /// Condition:
    /// end_key > file.smallestkey
    /// start_key <= file.largestkey
    pub fn update_overlap_damaged_ranges(&mut self, fname: &str, start: &[u8], end: &[u8]) -> bool {
        // `region_ranges` is promised to have no overlap so just check the first
        // region.
        if let Some((_, id)) = self
            .region_ranges
            .range((Excluded(start.to_owned()), Unbounded::<Vec<u8>>))
            .next()
        {
            let region = &self.regions[id];
            if keys::enc_start_key(region).as_slice() <= end {
                if let HashMapEntry::Vacant(v) = self.damaged_ranges.entry(fname.to_owned()) {
                    v.insert((start.to_owned(), end.to_owned()));
                }
                return true;
            }
        }

        // It's OK to remove the range here before deleting real file.
        let _ = self.damaged_ranges.remove(fname);
        false
    }

    /// Get all region ids overlapping damaged ranges.
    pub fn get_all_damaged_region_ids(&self) -> HashSet<u64> {
        let mut ids = HashSet::default();
        for (_fname, (start, end)) in self.damaged_ranges.iter() {
            for (_, id) in self
                .region_ranges
                .range((Excluded(start.clone()), Unbounded::<Vec<u8>>))
            {
                let region = &self.regions[id];
                if &keys::enc_start_key(region) <= end {
                    ids.insert(*id);
                } else {
                    // `region_ranges` is promised to have no overlap.
                    break;
                }
            }
        }
        if !ids.is_empty() {
            warn!(
                "detected damaged regions overlapping damaged file ranges";
                "id" => ?&ids,
            );
        }
        ids
    }
}

pub struct RaftRouter<EK, ER>
where
    EK: KvEngine,
    ER: RaftEngine,
{
    pub router: BatchRouter<PeerFsm<EK, ER>, StoreFsm<EK>>,
}

impl<EK, ER> Clone for RaftRouter<EK, ER>
where
    EK: KvEngine,
    ER: RaftEngine,
{
    fn clone(&self) -> Self {
        RaftRouter {
            router: self.router.clone(),
        }
    }
}

impl<EK, ER> Deref for RaftRouter<EK, ER>
where
    EK: KvEngine,
    ER: RaftEngine,
{
    type Target = BatchRouter<PeerFsm<EK, ER>, StoreFsm<EK>>;

    fn deref(&self) -> &BatchRouter<PeerFsm<EK, ER>, StoreFsm<EK>> {
        &self.router
    }
}

impl<EK, ER> ApplyNotifier<EK> for RaftRouter<EK, ER>
where
    EK: KvEngine,
    ER: RaftEngine,
{
    fn notify(&self, apply_res: Vec<ApplyRes<EK::Snapshot>>) {
        for r in apply_res {
            let region_id = r.region_id;
            if let Err(e) = self.router.force_send(
                region_id,
                PeerMsg::ApplyRes {
                    res: ApplyTaskRes::Apply(r),
                },
            ) {
                error!("failed to send apply result"; "region_id" => region_id, "err" => ?e);
            }
        }
    }
    fn notify_one(&self, region_id: u64, msg: PeerMsg<EK>) {
        if let Err(e) = self.router.force_send(region_id, msg) {
            error!("failed to notify apply msg"; "region_id" => region_id, "err" => ?e);
        }
    }

    fn clone_box(&self) -> Box<dyn ApplyNotifier<EK>> {
        Box::new(self.clone())
    }
}

impl<EK, ER> RaftRouter<EK, ER>
where
    EK: KvEngine,
    ER: RaftEngine,
{
    pub fn send_raft_message(
        &self,
        msg: RaftMessage,
    ) -> std::result::Result<(), TrySendError<RaftMessage>> {
        fail_point!("send_raft_message_full", |_| Err(TrySendError::Full(
            RaftMessage::default()
        )));

        let id = msg.get_region_id();

        let mut heap_size = 0;
        for e in msg.get_message().get_entries() {
            heap_size += bytes_capacity(&e.data) + bytes_capacity(&e.context);
        }
        let peer_msg = PeerMsg::RaftMessage(InspectedRaftMessage { heap_size, msg });
        let event = TraceEvent::Add(heap_size);
        let send_failed = Cell::new(true);

        MEMTRACE_RAFT_MESSAGES.trace(event);
        defer!(if send_failed.get() {
            MEMTRACE_RAFT_MESSAGES.trace(TraceEvent::Sub(heap_size));
        });

        let store_msg = match self.try_send(id, peer_msg) {
            Either::Left(Ok(())) => {
                fail_point!("memtrace_raft_messages_overflow_check_send");
                send_failed.set(false);
                return Ok(());
            }
            Either::Left(Err(TrySendError::Full(PeerMsg::RaftMessage(im)))) => {
                return Err(TrySendError::Full(im.msg));
            }
            Either::Left(Err(TrySendError::Disconnected(PeerMsg::RaftMessage(im)))) => {
                return Err(TrySendError::Disconnected(im.msg));
            }
            Either::Right(PeerMsg::RaftMessage(im)) => StoreMsg::RaftMessage(im),
            _ => unreachable!(),
        };
        match self.send_control(store_msg) {
            Ok(()) => {
                send_failed.set(false);
                Ok(())
            }
            Err(TrySendError::Full(StoreMsg::RaftMessage(im))) => Err(TrySendError::Full(im.msg)),
            Err(TrySendError::Disconnected(StoreMsg::RaftMessage(im))) => {
                Err(TrySendError::Disconnected(im.msg))
            }
            _ => unreachable!(),
        }
    }

    #[inline]
    pub fn send_raft_command(
        &self,
        cmd: RaftCommand<EK::Snapshot>,
    ) -> std::result::Result<(), TrySendError<RaftCommand<EK::Snapshot>>> {
        let region_id = cmd.request.get_header().get_region_id();
        match self.send(region_id, PeerMsg::RaftCommand(cmd)) {
            Ok(()) => Ok(()),
            Err(TrySendError::Full(PeerMsg::RaftCommand(cmd))) => Err(TrySendError::Full(cmd)),
            Err(TrySendError::Disconnected(PeerMsg::RaftCommand(cmd))) => {
                Err(TrySendError::Disconnected(cmd))
            }
            _ => unreachable!(),
        }
    }

    /// Broadcasts resolved result to all regions.
    pub fn report_resolved(&self, store_id: u64, group_id: u64) {
        self.broadcast_normal(|| {
            PeerMsg::SignificantMsg(SignificantMsg::StoreResolved { store_id, group_id })
        })
    }

    pub fn register(&self, region_id: u64, mailbox: BasicMailbox<PeerFsm<EK, ER>>) {
        self.router.register(region_id, mailbox);
        self.update_trace();
    }

    pub fn register_all(&self, mailboxes: Vec<(u64, BasicMailbox<PeerFsm<EK, ER>>)>) {
        self.router.register_all(mailboxes);
        self.update_trace();
    }

    pub fn close(&self, region_id: u64) {
        self.router.close(region_id);
        self.update_trace();
    }

    pub fn clear_cache(&self) {
        self.router.clear_cache();
    }

    fn update_trace(&self) {
        let router_trace = self.router.trace();
        MEMTRACE_RAFT_ROUTER_ALIVE.trace(TraceEvent::Reset(router_trace.alive));
        MEMTRACE_RAFT_ROUTER_LEAK.trace(TraceEvent::Reset(router_trace.leak));
    }
}

#[derive(Default)]
pub struct PeerTickBatch {
    pub ticks: Vec<Box<dyn FnOnce() + Send>>,
    pub wait_duration: Duration,
}

impl PeerTickBatch {
    #[inline]
    pub fn schedule(&mut self, timer: &SteadyTimer) {
        if self.ticks.is_empty() {
            return;
        }
        let peer_ticks = mem::take(&mut self.ticks);
        let f = timer.delay(self.wait_duration).compat().map(move |_| {
            for tick in peer_ticks {
                tick();
            }
        });
        poll_future_notify(f);
    }
}

impl Clone for PeerTickBatch {
    fn clone(&self) -> PeerTickBatch {
        PeerTickBatch {
            ticks: vec![],
            wait_duration: self.wait_duration,
        }
    }
}

pub struct PollContext<EK, ER, T>
where
    EK: KvEngine,
    ER: RaftEngine,
{
    pub cfg: Config,
    pub store: metapb::Store,
    pub consistency_check_scheduler: Scheduler<ConsistencyCheckTask<EK::Snapshot>>,
    pub split_check_scheduler: Scheduler<SplitCheckTask>,
    pub raftlog_gc_scheduler: Scheduler<RaftlogGcTask>,
    pub raftlog_fetch_scheduler: Scheduler<ReadTask<EK>>,
    pub region_scheduler: Scheduler<RegionTask<EK::Snapshot>>,
    pub apply_router: ApplyRouter<EK>,
    pub router: RaftRouter<EK, ER>,
    pub importer: Arc<SstImporter>,
    pub store_meta: Arc<Mutex<StoreMeta>>,
    pub feature_gate: FeatureGate,
    /// region_id -> (peer_id, is_splitting)
    /// Used for handling race between splitting and creating new peer.
    /// An uninitialized peer can be replaced to the one from splitting iff they
    /// are exactly the same peer.
    ///
    /// WARNING:
    /// To avoid deadlock, if you want to use `store_meta` and
    /// `pending_create_peers` together, the lock sequence MUST BE:
    /// 1. lock the store_meta.
    /// 2. lock the pending_create_peers.
    pub pending_create_peers: Arc<Mutex<HashMap<u64, (u64, bool)>>>,
    pub raft_metrics: RaftMetrics,
    pub snap_mgr: SnapManager,
    pub coprocessor_host: CoprocessorHost<EK>,
    pub timer: SteadyTimer,
    pub trans: T,
    /// WARNING:
    /// To avoid deadlock, if you want to use `store_meta` and
    /// `global_replication_state` together, the lock sequence MUST BE:
    /// 1. lock the store_meta.
    /// 2. lock the global_replication_state.
    pub global_replication_state: Arc<Mutex<GlobalReplicationState>>,
    pub global_stat: GlobalStoreStat,
    pub store_stat: LocalStoreStat,
    pub engines: Engines<EK, ER>,
    pub pending_count: usize,
    pub ready_count: usize,
    pub has_ready: bool,
    pub current_time: Option<Timespec>,
    pub raft_perf_context: ER::PerfContext,
    pub kv_perf_context: EK::PerfContext,
    pub tick_batch: Vec<PeerTickBatch>,
    pub node_start_time: Option<TiInstant>,
    /// Disk usage for the store itself.
    pub self_disk_usage: DiskUsage,

    // TODO: how to remove offlined stores?
    /// Disk usage for other stores. The store itself is not included.
    /// Only contains items which is not `DiskUsage::Normal`.
    pub store_disk_usages: HashMap<u64, DiskUsage>,
    pub write_senders: WriteSenders<EK, ER>,
    pub sync_write_worker: Option<WriteWorker<EK, ER, RaftRouter<EK, ER>, T>>,
    pub pending_latency_inspect: Vec<util::LatencyInspector>,
}

impl<EK, ER, T> PollContext<EK, ER, T>
where
    EK: KvEngine,
    ER: RaftEngine,
{
    #[inline]
    pub fn store_id(&self) -> u64 {
        self.store.get_id()
    }

    pub fn update_ticks_timeout(&mut self) {
        self.tick_batch[PeerTick::Raft as usize].wait_duration = self.cfg.raft_base_tick_interval.0;
        self.tick_batch[PeerTick::RaftLogGc as usize].wait_duration =
            self.cfg.raft_log_gc_tick_interval.0;
        self.tick_batch[PeerTick::EntryCacheEvict as usize].wait_duration =
            ENTRY_CACHE_EVICT_TICK_DURATION;
        self.tick_batch[PeerTick::PdHeartbeat as usize].wait_duration =
            self.cfg.pd_heartbeat_tick_interval.0;
        self.tick_batch[PeerTick::SplitRegionCheck as usize].wait_duration =
            self.cfg.split_region_check_tick_interval.0;
        self.tick_batch[PeerTick::CheckPeerStaleState as usize].wait_duration =
            self.cfg.peer_stale_state_check_interval.0;
        self.tick_batch[PeerTick::CheckMerge as usize].wait_duration =
            self.cfg.merge_check_tick_interval.0;
        self.tick_batch[PeerTick::CheckLeaderLease as usize].wait_duration =
            self.cfg.check_leader_lease_interval.0;
        self.tick_batch[PeerTick::ReactivateMemoryLock as usize].wait_duration =
            self.cfg.reactive_memory_lock_tick_interval.0;
        self.tick_batch[PeerTick::ReportBuckets as usize].wait_duration =
            self.cfg.report_region_buckets_tick_interval.0;
        self.tick_batch[PeerTick::CheckLongUncommitted as usize].wait_duration =
            self.cfg.check_long_uncommitted_interval.0;
        self.tick_batch[PeerTick::CheckPeersAvailability as usize].wait_duration =
            self.cfg.check_peers_availability_interval.0;
    }
}

struct Store {
    stopped: bool,
}

pub struct StoreFsm<EK>
where
    EK: KvEngine,
{
    store: Store,
    _phantom: PhantomData<EK>,
}

impl<EK> Fsm for StoreFsm<EK>
where
    EK: KvEngine,
{
    type Message = StoreMsg<EK>;

    #[inline]
    fn is_stopped(&self) -> bool {
        self.store.stopped
    }
}

pub struct RaftPoller<EK: KvEngine + 'static, ER: RaftEngine + 'static, T: 'static> {
    phantom: PhantomData<(EK, ER, T)>,
}

#[derive(Debug, PartialEq)]
enum CheckMsgStatus {}
