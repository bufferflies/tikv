// Copyright 2021 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    cmp,
    cmp::Ordering as CmpOrdering,
    fmt::{self, Display, Formatter},
    mem,
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc,
        mpsc::{Sender, SyncSender},
        Arc,
    },
    thread::JoinHandle,
    time::{Duration, Instant},
};

use api_version::{api_v2::is_whole_keyspace_range, ApiV2};
use cloud_encryption::{KeyspaceEncryptionConfig, MasterKeyConfig};
use collections::HashMap;
use concurrency_manager::ConcurrencyManager;
use fail::fail_point;
use futures::{compat::Future01CompatExt, FutureExt};
use health_controller::{
    metrics::{
        flush_store_inspect_disk_duration_metrics, flush_store_inspect_network_duration_metrics,
        flush_store_inspect_slow_score_metrics,
    },
    reporters::{Config as ReporterConfig, RfStoreReporter},
    HealthController, InspectDuration, InspectFactor, LatencyInspector,
};
use kvengine::{Shard, GLOBAL_SHARD_END_KEY};
use kvproto::{
    metapb,
    metapb::Region,
    pdpb,
    pdpb::{DfsStatItem, Peers, SyncRegionResponse},
    raft_cmdpb::{
        AdminCmdType, AdminRequest, ChangePeerRequest, ChangePeerV2Request, RaftCmdRequest,
        SplitRequest,
    },
    raft_serverpb::RaftMessage,
    replication_modepb::RegionReplicationStatus,
};
use pd_client::{
    keyspace::to_keyspace_name, merge_bucket_stats, metrics::*, BucketStat, PdClient, RegionStat,
};
use prometheus::local::LocalHistogram;
use protobuf::Message;
use raft::{eraftpb::ConfChangeType, StateRole};
use raftstore::store::{util, util::ConfChangeKind, ReadStats, TxnExt, WriteStats};
use resource_metering::{Collector, RawRecords};
use schema::schema::StorageClass;
use tikv_util::{
    debug, defer, error, info,
    metrics::ThreadInfoStatistics,
    store::{find_peer, QueryStats},
    sys::{disk, thread::StdThreadBuildWrapper, SysQuota},
    thd_name,
    time::{Instant as TiInstant, UnixSecs},
    timer::GLOBAL_TIMER_HANDLE,
    topn::TopN,
    warn,
    worker::{Runnable, Scheduler},
    Either, GLOBAL_SERVER_READINESS,
};
use trace_event::types::TraceContext;
use txn_types::{Key, NULL_KEYSPACE_ID};
use yatp::Remote;

use crate::{
    store::{
        encode_split_flag_encryption_metas, raw_end_key, raw_start_key, util::KeysInfoFormatter,
        Callback, CasualMessage, Config as RfStoreConfig, CpuUtilCollector, PeerMsg, PeerTag,
        RegionIdVer, RegionMap, StoreInfo, StoreMsg,
    },
    RaftRouter, RaftStoreRouter,
};

type RecordPairVec = Vec<pdpb::RecordPair>;

#[derive(Clone)]
pub struct FlowStatsReporter {
    scheduler: Scheduler<PdTask>,
}

impl FlowStatsReporter {
    pub fn new(scheduler: Scheduler<PdTask>) -> Self {
        Self { scheduler }
    }
}

impl raftstore::store::FlowStatsReporter for FlowStatsReporter {
    fn report_read_stats(&self, read_stats: ReadStats) {
        if let Err(e) = self.scheduler.schedule(PdTask::ReadStats { read_stats }) {
            error!("Failed to send read flow statistics"; "err" => ?e);
        }
    }

    fn report_write_stats(&self, write_stats: WriteStats) {
        if let Err(e) = self.scheduler.schedule(PdTask::WriteStats { write_stats }) {
            error!("Failed to send write flow statistics"; "err" => ?e);
        }
    }
}

pub struct HeartbeatTask {
    pub term: u64,
    pub region: metapb::Region,
    pub peer: metapb::Peer,
    pub down_peers: Vec<pdpb::PeerStats>,
    pub pending_peers: Vec<metapb::Peer>,
    pub written_bytes: u64,
    pub written_keys: u64,
    pub approximate_size: u64,
    pub approximate_keys: u64,
    pub approximate_kv_size: u64,
    pub replication_status: Option<RegionReplicationStatus>,
    pub bucket_stat: Option<BucketStat>,
}

/// Uses an asynchronous thread to tell PD something.
pub enum PdTask {
    AskBatchSplit {
        region: metapb::Region,
        split_keys: Vec<Vec<u8>>,
        peer: metapb::Peer,
        // If true, right Region derives origin region_id.
        right_derive: bool,
        callback: Callback,
    },
    Heartbeat(HeartbeatTask),
    StoreHeartbeat {
        stats: pdpb::StoreStats,
        store_info: StoreInfo,
        send_detailed_report: bool,
    },
    ReportBatchSplit {
        regions: Vec<metapb::Region>,
    },
    ValidatePeer {
        region: metapb::Region,
        peer: metapb::Peer,
    },
    ReadStats {
        read_stats: ReadStats,
    },
    WriteStats {
        write_stats: WriteStats,
    },
    RegionCpuRecords(Arc<RawRecords>),
    DestroyPeer {
        region_id: u64,
        keyspace_id: Option<u32>,
    },
    StoreInfos {
        cpu_usages: RecordPairVec,
        read_io_rates: RecordPairVec,
        write_io_rates: RecordPairVec,
    },
    UpdateMaxTimestamp {
        region_id: u64,
        initial_status: u64,
        txn_ext: Arc<TxnExt>,
    },
    UpdateGcSafePoint,
    SyncRegion {
        start: Vec<u8>,
        end: Vec<u8>,
        limit: usize,
        reverse: bool,
        callback: Box<dyn FnOnce(SyncRegionResponse) + Send>,
    },
    SyncRegionById {
        region_id: u64,
        callback: Box<dyn FnOnce(SyncRegionResponse) + Send>,
    },
    RoleChanged {
        region_id: u64,
        keyspace_id: Option<u32>,
        role: StateRole,
    },
    UpdateRaftCpuUtil,
    InspectLatency {
        factor: InspectFactor,
    },
    UpdateSlowScore {
        id: u64,
        factor: InspectFactor,
        duration: InspectDuration,
    },
}

impl Display for PdTask {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            PdTask::AskBatchSplit {
                ref region,
                ref split_keys,
                ..
            } => write!(
                f,
                "ask split region {} with {}",
                region.get_id(),
                KeysInfoFormatter(split_keys.iter())
            ),
            PdTask::Heartbeat(ref hb_task) => write!(
                f,
                "heartbeat for region {:?}, leader {}, replication status {:?}",
                hb_task.region,
                hb_task.peer.get_id(),
                hb_task.replication_status
            ),
            PdTask::StoreHeartbeat { ref stats, .. } => {
                write!(f, "store heartbeat stats: {:?}", stats)
            }
            PdTask::ReportBatchSplit { ref regions } => write!(f, "report split {:?}", regions),
            PdTask::ValidatePeer {
                ref region,
                ref peer,
            } => write!(f, "validate peer {:?} with region {:?}", peer, region),
            PdTask::ReadStats { ref read_stats } => {
                write!(f, "get the read statistics {:?}", read_stats)
            }
            PdTask::WriteStats { ref write_stats } => {
                write!(f, "get the write statistics {:?}", write_stats)
            }
            PdTask::RegionCpuRecords(ref cpu_records) => {
                write!(f, "get region cpu records: {:?}", cpu_records)
            }
            PdTask::DestroyPeer {
                ref region_id,
                keyspace_id,
            } => {
                write!(
                    f,
                    "destroy peer of region {} keyspace_id {:?}",
                    region_id, keyspace_id
                )
            }
            PdTask::StoreInfos {
                ref cpu_usages,
                ref read_io_rates,
                ref write_io_rates,
            } => write!(
                f,
                "get store's information: cpu_usages {:?}, read_io_rates {:?}, write_io_rates {:?}",
                cpu_usages, read_io_rates, write_io_rates,
            ),
            PdTask::UpdateMaxTimestamp { region_id, .. } => write!(
                f,
                "update the max timestamp for region {} in the concurrency manager",
                region_id
            ),
            PdTask::UpdateGcSafePoint => write!(f, "update GC safe point"),
            PdTask::SyncRegion { start, end, .. } => {
                write!(
                    f,
                    "sync region, range: [{}, {})",
                    log_wrappers::Value(start),
                    log_wrappers::Value(end)
                )
            }
            PdTask::SyncRegionById { region_id, .. } => {
                write!(f, "sync region by id: {}", region_id)
            }
            PdTask::RoleChanged {
                region_id,
                keyspace_id,
                role,
            } => {
                write!(
                    f,
                    "region {} keyspace {} change role to {:?}",
                    region_id,
                    keyspace_id.unwrap_or_default(),
                    role
                )
            }
            PdTask::UpdateRaftCpuUtil => write!(f, "update raft cpu utilization"),
            PdTask::InspectLatency { factor } => {
                write!(f, "inspect raftstore latency: {:?}", factor)
            }
            PdTask::UpdateSlowScore {
                id,
                factor,
                ref duration,
            } => {
                write!(
                    f,
                    "compute slow score: id {}, factor: {:?}, duration {:?}",
                    id, factor, duration
                )
            }
        }
    }
}

pub const NUM_COLLECT_STORE_INFOS_PER_HEARTBEAT: u32 = 2;
/// The upper bound of buffered stats messages.
/// It prevents unexpected memory buildup when AutoSplitController
/// runs slowly.
const STATS_CHANNEL_CAPACITY_LIMIT: usize = 128;

const DEFAULT_LOAD_BASE_SPLIT_CHECK_INTERVAL: Duration = Duration::from_secs(1);
const DEFAULT_COLLECT_TICK_INTERVAL: Duration = Duration::from_secs(1);

fn default_collect_tick_interval() -> Duration {
    fail_point!("mock_collect_tick_interval", |_| {
        Duration::from_millis(1)
    });
    DEFAULT_COLLECT_TICK_INTERVAL
}

/// Max limitation of delayed store_heartbeat.
const STORE_HEARTBEAT_DELAY_LIMIT: u64 = 5 * 60;

/// Determines the minimal interval for latency inspection ticks based on raft
/// and kvdb inspection intervals.
///
/// This function handles different scenarios for latency inspection:
/// 1. Both intervals are zero: Inspection is disabled, returns a large interval
///    (1 hour)
/// 2. Only raft interval is zero: Uses kvdb interval (raft inspection disabled)
/// 3. Only kvdb interval is zero: Uses raft interval (kvdb inspection disabled)
/// 4. Both intervals non-zero: Uses the smaller of the two intervals
///
/// # Arguments
///
/// * `inspect_raft_latency_interval` - Interval for raft latency inspection
/// * `inspect_kvdb_latency_interval` - Interval for kvdb latency inspection
///
/// # Returns
///
/// The minimal interval that should be used for latency inspection ticks
fn get_minimal_inspect_tick_interval(
    inspect_raft_latency_interval: Duration,
    inspect_kvdb_latency_interval: Duration,
) -> Duration {
    match (
        inspect_raft_latency_interval.is_zero(),
        inspect_kvdb_latency_interval.is_zero(),
    ) {
        (true, true) => {
            // Both inspections are disabled - return a large interval to avoid misleading
            // tick checks
            Duration::from_secs(3600)
        }
        (true, false) => {
            // raft inspection disabled - use kvdb interval
            inspect_kvdb_latency_interval
        }
        (false, true) => {
            // kvdb inspection disabled - use raft interval
            inspect_raft_latency_interval
        }
        (false, false) => {
            // Both inspections enabled - use the smaller interval
            std::cmp::min(inspect_raft_latency_interval, inspect_kvdb_latency_interval)
        }
    }
}

#[inline]
fn convert_record_pairs(m: HashMap<String, u64>) -> RecordPairVec {
    m.into_iter()
        .map(|(k, v)| {
            let mut pair = pdpb::RecordPair::default();
            pair.set_key(k);
            pair.set_value(v);
            pair
        })
        .collect()
}

#[derive(Clone)]
pub struct WrappedScheduler(Scheduler<PdTask>);

impl Collector for WrappedScheduler {
    fn collect(&self, records: Arc<RawRecords>) {
        self.0.schedule(PdTask::RegionCpuRecords(records)).ok();
    }
}

pub trait StoreStatsReporter: Send + Clone + Sync + 'static + Collector {
    fn report_store_infos(
        &self,
        cpu_usages: RecordPairVec,
        read_io_rates: RecordPairVec,
        write_io_rates: RecordPairVec,
    );
    fn update_latency_stats(&self, timer_tick: u64, factor: InspectFactor);
}

impl StoreStatsReporter for WrappedScheduler {
    fn report_store_infos(
        &self,
        cpu_usages: RecordPairVec,
        read_io_rates: RecordPairVec,
        write_io_rates: RecordPairVec,
    ) {
        let task = PdTask::StoreInfos {
            cpu_usages,
            read_io_rates,
            write_io_rates,
        };
        if let Err(e) = self.0.schedule(task) {
            error!(
                "failed to send store infos to pd worker";
                "err" => ?e,
            );
        }
    }

    fn update_latency_stats(&self, timer_tick: u64, factor: InspectFactor) {
        debug!("update latency statistics for rfstore";
                "tick" => timer_tick);
        let task = PdTask::InspectLatency { factor };
        if let Err(e) = self.0.schedule(task) {
            warn!(
                "failed to send inspect rfstore latency task to pd worker";
                "err" => ?e,
            );
        }
    }
}

pub struct StatsMonitor<T>
where
    T: StoreStatsReporter,
{
    reporter: T,
    handle: Option<JoinHandle<()>>,
    timer: Option<Sender<bool>>,
    read_stats_sender: Option<SyncSender<ReadStats>>,
    cpu_stats_sender: Option<SyncSender<Arc<RawRecords>>>,
    collect_store_infos_interval: Duration,
    load_base_split_check_interval: Duration, // Unimplemented!()
    collect_tick_interval: Duration,
    inspect_raft_latency_interval: Duration, // for raft mount path
    inspect_kvdb_latency_interval: Duration, // for kvdb mount path
    inspect_network_interval: Duration,
}

impl<T> StatsMonitor<T>
where
    T: StoreStatsReporter,
{
    pub fn new(
        interval: Duration,
        inspect_raft_latency_interval: Duration,
        inspect_kvdb_latency_interval: Duration,
        inspect_network_interval: Duration,
        reporter: T,
    ) -> Self {
        StatsMonitor {
            reporter,
            handle: None,
            timer: None,
            read_stats_sender: None,
            cpu_stats_sender: None,
            collect_store_infos_interval: interval,
            load_base_split_check_interval: cmp::min(
                DEFAULT_LOAD_BASE_SPLIT_CHECK_INTERVAL,
                interval,
            ),
            // Use the smallest inspect latency as the minimal limitation for collecting tick.
            collect_tick_interval: cmp::min(
                get_minimal_inspect_tick_interval(
                    inspect_raft_latency_interval,
                    inspect_kvdb_latency_interval,
                ),
                interval.min(default_collect_tick_interval()),
            ),
            inspect_raft_latency_interval,
            inspect_kvdb_latency_interval,
            inspect_network_interval,
        }
    }

    // Collecting thread information and obtaining qps informations.
    // They run together in the same thread by taking module at different intervals.
    pub fn start(&mut self) -> Result<(), std::io::Error> {
        if self.collect_tick_interval
            < cmp::min(
                get_minimal_inspect_tick_interval(
                    self.inspect_raft_latency_interval,
                    self.inspect_kvdb_latency_interval,
                ),
                default_collect_tick_interval(),
            )
        {
            info!(
                "interval is too small, skip stats monitoring. If we are running tests, it is normal, otherwise a check is needed."
            );
            return Ok(());
        }
        let mut timer_cnt = 0; // to run functions with different intervals in a loop
        let tick_interval = self.collect_tick_interval;
        let collect_store_infos_interval = self
            .collect_store_infos_interval
            .div_duration_f64(tick_interval) as u64;
        let _load_base_split_check_interval = self
            .load_base_split_check_interval
            .div_duration_f64(tick_interval) as u64;
        let update_raftdisk_latency_stats_interval =
            self.inspect_raft_latency_interval
                .div_duration_f64(tick_interval) as u64;
        let update_kvdisk_latency_stats_interval =
            self.inspect_kvdb_latency_interval
                .div_duration_f64(tick_interval) as u64;
        let update_network_latency_stats_interval =
            self.inspect_network_interval
                .div_duration_f64(tick_interval) as u64;

        let (timer_tx, timer_rx) = mpsc::channel();
        self.timer = Some(timer_tx);

        let (read_stats_sender, _read_stats_receiver) =
            mpsc::sync_channel(STATS_CHANNEL_CAPACITY_LIMIT);
        self.read_stats_sender = Some(read_stats_sender);

        let (cpu_stats_sender, _cpu_stats_receiver) =
            mpsc::sync_channel(STATS_CHANNEL_CAPACITY_LIMIT);
        self.cpu_stats_sender = Some(cpu_stats_sender);

        let reporter = self.reporter.clone();
        let props = tikv_util::thread_group::current_properties();

        fn is_enable_tick(timer_cnt: u64, interval: u64) -> bool {
            interval != 0 && timer_cnt % interval == 0
        }
        let h = std::thread::Builder::new()
            .name(thd_name!("stats-monitor"))
            .spawn_wrapper(move || {
                tikv_util::thread_group::set_properties(props);

                // Create different `ThreadInfoStatistics` for different purposes to
                // make sure the record won't be disturbed.
                let mut collect_store_infos_thread_stats = ThreadInfoStatistics::new();
                while let Err(mpsc::RecvTimeoutError::Timeout) =
                    timer_rx.recv_timeout(tick_interval)
                {
                    if is_enable_tick(timer_cnt, collect_store_infos_interval) {
                        StatsMonitor::collect_store_infos(
                            &mut collect_store_infos_thread_stats,
                            &reporter,
                        );
                    }
                    if is_enable_tick(timer_cnt, update_raftdisk_latency_stats_interval) {
                        reporter.update_latency_stats(timer_cnt, InspectFactor::RaftDisk);
                    }
                    if is_enable_tick(timer_cnt, update_kvdisk_latency_stats_interval) {
                        reporter.update_latency_stats(timer_cnt, InspectFactor::KvDisk);
                    }
                    if is_enable_tick(timer_cnt, update_network_latency_stats_interval) {
                        reporter.update_latency_stats(timer_cnt, InspectFactor::Network);
                    }
                    timer_cnt += 1;
                }
            })?;

        self.handle = Some(h);
        Ok(())
    }

    pub fn collect_store_infos(thread_stats: &mut ThreadInfoStatistics, reporter: &T) {
        thread_stats.record();
        let cpu_usages = convert_record_pairs(thread_stats.get_cpu_usages());
        let read_io_rates = convert_record_pairs(thread_stats.get_read_io_rates());
        let write_io_rates = convert_record_pairs(thread_stats.get_write_io_rates());

        reporter.report_store_infos(cpu_usages, read_io_rates, write_io_rates);
    }

    pub fn stop(&mut self) {
        if let Some(h) = self.handle.take() {
            drop(self.timer.take());
            drop(self.read_stats_sender.take());
            drop(self.cpu_stats_sender.take());
            if let Err(e) = h.join() {
                error!("join stats collector failed"; "err" => ?e);
            }
        }
    }

    #[inline]
    pub fn maybe_send_read_stats(&self, read_stats: ReadStats) {
        if let Some(sender) = &self.read_stats_sender {
            if sender.try_send(read_stats).is_err() {
                debug!("send read_stats failed, are we shutting down or channel is full?")
            }
        }
    }

    #[inline]
    pub fn maybe_send_cpu_stats(&self, cpu_stats: &Arc<RawRecords>) {
        if let Some(sender) = &self.cpu_stats_sender {
            if sender.try_send(cpu_stats.clone()).is_err() {
                debug!("send region cpu info failed, are we shutting down or channel is full?")
            }
        }
    }
}

#[derive(Default, Clone)]
struct PeerCmpReadStat {
    pub region_id: u64,
    pub report_stat: u64,
}

impl Ord for PeerCmpReadStat {
    fn cmp(&self, other: &Self) -> CmpOrdering {
        self.report_stat.cmp(&other.report_stat)
    }
}

impl Eq for PeerCmpReadStat {}

impl PartialEq for PeerCmpReadStat {
    fn eq(&self, other: &Self) -> bool {
        self.report_stat == other.report_stat
    }
}

impl PartialOrd for PeerCmpReadStat {
    fn partial_cmp(&self, other: &Self) -> Option<CmpOrdering> {
        Some(self.report_stat.cmp(&other.report_stat))
    }
}

pub struct StoreStat {
    pub engine_total_bytes_read: u64,
    pub engine_total_keys_read: u64,
    pub engine_total_query_num: QueryStats,
    pub engine_last_total_bytes_read: u64,
    pub engine_last_total_keys_read: u64,
    pub engine_last_query_num: QueryStats,
    pub engine_last_capacity_size: u64,
    pub engine_last_used_size: u64,
    pub engine_last_available_size: u64,
    pub last_report_ts: UnixSecs,

    pub region_bytes_read: LocalHistogram,
    pub region_keys_read: LocalHistogram,
    pub region_bytes_written: LocalHistogram,
    pub region_keys_written: LocalHistogram,

    pub store_cpu_usages: RecordPairVec,
    pub store_read_io_rates: RecordPairVec,
    pub store_write_io_rates: RecordPairVec,

    store_cpu_quota: f64, // quota of cpu usage
    store_cpu_busy_thd: f64,
}

impl Default for StoreStat {
    fn default() -> StoreStat {
        StoreStat {
            region_bytes_read: REGION_READ_BYTES_HISTOGRAM.local(),
            region_keys_read: REGION_READ_KEYS_HISTOGRAM.local(),
            region_bytes_written: REGION_WRITTEN_BYTES_HISTOGRAM.local(),
            region_keys_written: REGION_WRITTEN_KEYS_HISTOGRAM.local(),

            engine_last_capacity_size: 0,
            engine_last_used_size: 0,
            engine_last_available_size: 0,
            last_report_ts: UnixSecs::zero(),
            engine_total_bytes_read: 0,
            engine_total_keys_read: 0,
            engine_last_total_bytes_read: 0,
            engine_last_total_keys_read: 0,
            engine_total_query_num: QueryStats::default(),
            engine_last_query_num: QueryStats::default(),

            store_cpu_usages: RecordPairVec::default(),
            store_read_io_rates: RecordPairVec::default(),
            store_write_io_rates: RecordPairVec::default(),

            store_cpu_quota: 0.0_f64,
            store_cpu_busy_thd: 0.8_f64,
        }
    }
}

impl StoreStat {
    fn set_cpu_quota(&mut self, cpu_cores: f64, busy_thd: f64) {
        self.store_cpu_quota = cpu_cores * 100.0;
        self.store_cpu_busy_thd = busy_thd;
    }

    fn maybe_busy(&self) -> bool {
        if self.store_cpu_quota < 1.0 || self.store_cpu_busy_thd > 1.0 {
            return false;
        }

        let mut cpu_usage = 0_u64;
        for record in self.store_cpu_usages.iter() {
            cpu_usage += record.get_value();
        }

        (cpu_usage as f64 / self.store_cpu_quota) >= self.store_cpu_busy_thd
    }
}

#[derive(Default)]
pub struct PeerStat {
    pub read_bytes: u64,
    pub read_keys: u64,
    pub query_stats: QueryStats,
    // last_region_report_attributes records the state of the last region heartbeat
    pub last_region_report_read_bytes: u64,
    pub last_region_report_read_keys: u64,
    pub last_region_report_query_stats: QueryStats,
    pub last_region_report_written_bytes: u64,
    pub last_region_report_written_keys: u64,
    pub last_region_report_ts: UnixSecs,
    // last_store_report_attributes records the state of the last store heartbeat
    pub last_store_report_read_bytes: u64,
    pub last_store_report_read_keys: u64,
    pub last_store_report_query_stats: QueryStats,
    pub approximate_keys: u64,
    pub approximate_size: u64,
    pub approximate_kv_size: u64,
    pub role: StateRole,

    pub down_peers: Vec<pdpb::PeerStats>,
    pub pending_peers: Vec<metapb::Peer>,
}

#[derive(Default)]
pub struct ReportBucket {
    current_stat: BucketStat,
    last_report_stat: Option<BucketStat>,
    last_report_ts: UnixSecs,
}

impl ReportBucket {
    #[allow(unused)]
    fn new(current_stat: BucketStat) -> Self {
        Self {
            current_stat,
            ..Default::default()
        }
    }

    fn new_report(&mut self, report_ts: UnixSecs) -> BucketStat {
        self.last_report_ts = report_ts;
        match self.last_report_stat.replace(self.current_stat.clone()) {
            Some(last) => {
                let mut delta = BucketStat::new(
                    self.current_stat.meta.clone(),
                    pd_client::new_bucket_stats(&self.current_stat.meta),
                );
                if last.meta.version != self.current_stat.meta.version {
                    // Do not update delta if the bucket version is changed for simplicity.
                    return delta;
                }
                for i in 0..delta.meta.keys.len() - 1 {
                    delta.stats.write_bytes[i] =
                        self.current_stat.stats.write_bytes[i] - last.stats.write_bytes[i];
                    delta.stats.write_keys[i] =
                        self.current_stat.stats.write_keys[i] - last.stats.write_keys[i];
                    delta.stats.write_qps[i] =
                        self.current_stat.stats.write_qps[i] - last.stats.write_qps[i];

                    delta.stats.read_bytes[i] =
                        self.current_stat.stats.read_bytes[i] - last.stats.read_bytes[i];
                    delta.stats.read_keys[i] =
                        self.current_stat.stats.read_keys[i] - last.stats.read_keys[i];
                    delta.stats.read_qps[i] =
                        self.current_stat.stats.read_qps[i] - last.stats.read_qps[i];
                }
                delta
            }
            None => self.current_stat.clone(),
        }
    }
}

pub struct PdRunner {
    store_id: u64,
    cluster_id: u64,
    pd_client: Arc<dyn PdClient>,
    router: RaftRouter,
    region_peers: HashMap<u64, PeerStat>,
    region_map: RegionMap,
    region_buckets: HashMap<u64, ReportBucket>,
    store_stat: StoreStat,
    store_map: HashMap<u64, bool>,
    store_heartbeat_interval: Duration,
    is_hb_receiver_scheduled: bool,
    // Records the boot time.
    start_ts: UnixSecs,

    // use for Runner inner handle function to send Task to itself
    // actually it is the sender connected to Runner's Worker which
    // calls Runner's run() on Task received.
    scheduler: Scheduler<PdTask>,

    // region_id -> total_cpu_time_ms (since last region heartbeat)
    region_cpu_records: HashMap<u64, u32>,

    concurrency_manager: ConcurrencyManager,
    remote: Remote<yatp::task::future::TaskCell>,
    kv: kvengine::Engine,
    raft_cpu_collector: CpuUtilCollector,

    update_gc_safe_point_in_progress: Arc<AtomicBool>,

    stats_monitor: StatsMonitor<WrappedScheduler>,
    health_reporter: RfStoreReporter,
    _health_controller: HealthController,
}

const HOTSPOT_KEY_RATE_THRESHOLD: u64 = 128;
const HOTSPOT_QUERY_RATE_THRESHOLD: u64 = 128;
const HOTSPOT_BYTE_RATE_THRESHOLD: u64 = 8 * 1024;
const HOTSPOT_REPORT_CAPACITY: usize = 1000;

// TODO: support dyamic configure threshold in future
fn hotspot_key_report_threshold() -> u64 {
    #[cfg(feature = "failpoints")]
    fail_point!("mock_hotspot_threshold", |_| { 0 });

    HOTSPOT_KEY_RATE_THRESHOLD * 10
}

fn hotspot_byte_report_threshold() -> u64 {
    #[cfg(feature = "failpoints")]
    fail_point!("mock_hotspot_threshold", |_| { 0 });

    HOTSPOT_BYTE_RATE_THRESHOLD * 10
}

fn hotspot_query_num_report_threshold() -> u64 {
    #[cfg(feature = "failpoints")]
    fail_point!("mock_hotspot_threshold", |_| { 0 });

    HOTSPOT_QUERY_RATE_THRESHOLD * 10
}

impl PdRunner {
    #[allow(unused)]
    const INTERVAL_DIVISOR: u32 = 2;

    pub fn new(
        cfg: &RfStoreConfig,
        store_id: u64,
        pd_client: Arc<dyn PdClient>,
        router: RaftRouter,
        scheduler: Scheduler<PdTask>,
        concurrency_manager: ConcurrencyManager,
        remote: Remote<yatp::task::future::TaskCell>,
        kv: kvengine::Engine,
        raft_cpu_collector: CpuUtilCollector,
        health_controller: HealthController,
    ) -> PdRunner {
        let cluster_id = pd_client.get_cluster_id().unwrap();
        // Initialize the `StoreStat` with the sepecified inpsecting threshold for CPU
        // usage, which is used to lower down the bias when updating `SlowScore`.
        let mut store_stat = StoreStat::default();
        store_stat.set_cpu_quota(SysQuota::cpu_cores_quota(), cfg.inspect_cpu_util_thd);

        let store_heartbeat_interval = cfg.pd_store_heartbeat_tick_interval.0;
        let interval = store_heartbeat_interval / NUM_COLLECT_STORE_INFOS_PER_HEARTBEAT;
        let mut stats_monitor = StatsMonitor::new(
            interval,
            cfg.inspect_interval.0,
            cfg.inspect_kvdb_interval.0,
            cfg.inspect_network_interval.0,
            WrappedScheduler(scheduler.clone()),
        );
        if let Err(e) = stats_monitor.start() {
            error!("failed to start stats collector, error = {:?}", e);
        }

        let health_reporter_config = ReporterConfig {
            inspect_raft_interval: cfg.inspect_interval.0,
            inspect_kvdb_interval: cfg.inspect_kvdb_interval.0,
            inspect_network_interval: cfg.inspect_network_interval.0,
        };

        let health_reporter = RfStoreReporter::new(&health_controller, health_reporter_config);
        PdRunner {
            store_id,
            cluster_id,
            pd_client,
            router,
            store_heartbeat_interval,
            is_hb_receiver_scheduled: false,
            region_peers: HashMap::default(),
            region_map: Default::default(),
            region_buckets: HashMap::default(),
            store_stat, // Use the initialized one.
            store_map: HashMap::default(),
            start_ts: UnixSecs::now(),
            scheduler,
            region_cpu_records: HashMap::default(),
            concurrency_manager,
            remote,
            kv,
            raft_cpu_collector,

            update_gc_safe_point_in_progress: Arc::new(AtomicBool::new(false)),
            stats_monitor,
            health_reporter,
            _health_controller: health_controller,
        }
    }

    fn peer_tag(&self, region: &Region) -> PeerTag {
        let id_ver = RegionIdVer::from_region(region);
        PeerTag::new(self.store_id, id_ver)
    }

    // Note: The parameter doesn't contain `self` because this function may
    // be called in an asynchronous context.
    fn handle_ask_batch_split(
        tag: PeerTag,
        router: RaftRouter,
        _scheduler: Scheduler<PdTask>,
        pd_client: Arc<dyn PdClient>,
        mut region: metapb::Region,
        split_keys: Vec<Vec<u8>>,
        peer: metapb::Peer,
        right_derive: bool,
        callback: Callback,
        task: String,
        remote: Remote<yatp::task::future::TaskCell>,
        dfs: Arc<dyn kvengine::dfs::Dfs>,
    ) {
        if split_keys.is_empty() {
            info!("empty split key, skip ask batch split";
                "region" => tag);
            return;
        }
        let mut encryption_cfgs = vec![];
        for i in 0..=split_keys.len() {
            let raw_start = if i == 0 {
                raw_start_key(&region)
            } else {
                Key::from_encoded_slice(&split_keys[i - 1])
                    .into_raw()
                    .unwrap_or_default()
            };
            let raw_end = if i == split_keys.len() {
                raw_end_key(&region)
            } else {
                Key::from_encoded_slice(&split_keys[i])
                    .into_raw()
                    .unwrap_or_default()
            };
            if is_whole_keyspace_range(&raw_start, &raw_end) {
                let keyspace_id = ApiV2::get_u32_keyspace_id_by_key(&raw_start).unwrap_or_default();
                match pd_client.get_keyspace_encryption(keyspace_id) {
                    Err(e) if pd_client::grpc_error_is_unimplemented(&e) => {
                        warn!(
                            "get_keyspace_encryption is unimplemented, skip encryption";
                            "region" => tag,
                            "err" => ?e,
                        );
                    }
                    Err(e) => {
                        warn!(
                            "get keyspace encryption config failed";
                            "region" => tag,
                            "err" => ?e,
                        );
                        return;
                    }
                    Ok(cfg) => {
                        if cfg.enabled {
                            info!(
                                "CMEK config found during keyspace split";
                                "keyspace_id" => keyspace_id,
                                "config" => format!("{:?}", cfg),
                            );
                            let vendor = cfg.vendor.clone().unwrap_or_default();
                            let cmek_id = cfg.cmek_id.clone().unwrap_or_default();
                            if vendor.is_empty() || cmek_id.is_empty() {
                                warn!(
                                    "CMEK config invalid, skip split";
                                    "keyspace_id" => keyspace_id,
                                );
                                return;
                            }
                            encryption_cfgs.push((keyspace_id, cfg));
                        }
                    }
                }
            }
        }

        let resp = pd_client.ask_batch_split(region.clone(), split_keys.len());
        let f = async move {
            let ids = match resp.await {
                Ok(mut resp) => resp.take_ids().into(),
                Err(e) => {
                    warn!(
                        "ask batch split failed";
                        "region" => tag,
                        "err" => ?e,
                    );
                    return;
                }
            };

            let encryption_metas =
                match prepare_and_persist_encryption_metas(&pd_client, dfs, &encryption_cfgs).await
                {
                    Ok(metas) => metas,
                    Err(e) => {
                        warn!("failed to prepare/persist CMEK encryption metas"; "err" => ?e);
                        return;
                    }
                };

            info!(
                "try to batch split region";
                "region" => tag,
                "new_region_ids" => ?ids,
                "region" => ?region,
                "task" => task,
            );
            let admin_req = new_batch_split_region_request(split_keys, ids, right_derive);
            let region_id = region.get_id();
            let epoch = region.take_region_epoch();
            let mut req = new_admin_command(region_id, epoch, peer, admin_req);
            if !encryption_metas.is_empty() {
                let header = req.mut_header();
                header.set_flag_data(encode_split_flag_encryption_metas(encryption_metas));
            }
            router.send_command(req, TraceContext::default(), callback);
        };
        remote.spawn(f);
    }

    fn handle_heartbeat(
        &mut self,
        term: u64,
        region: metapb::Region,
        peer: metapb::Peer,
        mut region_stat: RegionStat,
        replication_status: Option<RegionReplicationStatus>,
    ) {
        self.store_stat
            .region_bytes_written
            .observe(region_stat.written_bytes as f64);
        self.store_stat
            .region_keys_written
            .observe(region_stat.written_keys as f64);
        self.store_stat
            .region_bytes_read
            .observe(region_stat.read_bytes as f64);
        self.store_stat
            .region_keys_read
            .observe(region_stat.read_keys as f64);

        for p in region.get_peers() {
            let store_id = p.get_store_id();
            if let std::collections::hash_map::Entry::Vacant(entry) = self.store_map.entry(store_id)
            {
                let resp = self.pd_client.get_store(store_id);
                match resp {
                    Ok(store) => {
                        let is_tiflash = store.get_labels().iter().any(|label| {
                            label.get_key() == "engine" && label.get_value() == "tiflash"
                        });
                        entry.insert(is_tiflash);
                    }
                    Err(e) => {
                        error!("failed to get store from pd"; "store_id" => store_id, "err" => ?e);
                    }
                }
            }
        }

        let has_tiflash_replicas = region.get_peers().iter().any(|p| {
            let store_id = p.get_store_id();
            self.store_map
                .get(&store_id)
                .map_or(false, |is_tiflash| *is_tiflash)
        });

        // If a region has any TiFlash replicas, it is typically created according to a
        // placement rule that ensures the region's key range contains only row data,
        // not index data. Therefore, we can set `approximate_columnar_kv_size` equal to
        // `approximate_kv_size` to accurately record the region’s uncompressed columnar
        // KV size. Check whether `region_stat.approximate_columnar_kv_size` is zero to
        // avoid any unexpected overwrites.
        if has_tiflash_replicas && region_stat.approximate_columnar_kv_size == 0 {
            region_stat.approximate_columnar_kv_size = region_stat.approximate_kv_size;
        };

        let shard = self.kv.get_shard(region.get_id());
        PdRunner::set_storage_size_metric(
            region.id,
            rfengine::get_region_keyspace_id_u32(&region),
            Some(region_stat.approximate_kv_size),
            has_tiflash_replicas,
            shard,
        );

        let changed = self
            .region_map
            .get(region.id)
            .map(|old| old.get_region_epoch() != region.get_region_epoch())
            .unwrap_or(true);
        if changed {
            self.region_map.put(region.clone())
        }

        STORE_ENGINE_FLOW_VEC
            .with_label_values(&["kv", "bytes_read"])
            .inc_by(region_stat.read_bytes);
        STORE_ENGINE_FLOW_VEC
            .with_label_values(&["kv", "keys_read"])
            .inc_by(region_stat.read_keys);
        STORE_ENGINE_FLOW_VEC
            .with_label_values(&["kv", "bytes_written"])
            .inc_by(region_stat.written_bytes);
        STORE_ENGINE_FLOW_VEC
            .with_label_values(&["kv", "keys_written"])
            .inc_by(region_stat.written_keys);
        STORE_ENGINE_FLOW_VEC
            .with_label_values(&["kv", "wal_file_bytes"])
            .inc_by(region_stat.written_bytes);

        let resp = self.pd_client.region_heartbeat(
            term,
            region.clone(),
            peer,
            region_stat,
            replication_status,
        );
        let f = async move {
            if let Err(e) = resp.await {
                debug!(
                    "failed to send heartbeat";
                    "region_id" => region.get_id(),
                    "err" => ?e
                );
            }
        };
        self.remote.spawn(f);
    }

    pub fn set_storage_size_metric(
        region_id: u64,
        keyspace_id: Option<u32>,
        kv_size: Option<u64>,
        has_tiflash_replicas: bool,
        shard: Option<Arc<Shard>>,
    ) {
        let region_id_string = region_id.to_string();
        let region_id_str = region_id_string.as_str();
        match keyspace_id {
            None => {}
            Some(keyspace_id_u32) => {
                let Some(keyspace_name) = to_keyspace_name(keyspace_id_u32) else {
                    return;
                };
                let keyspace_name_str = keyspace_name.as_str();
                match kv_size {
                    None => {
                        let _ = STORE_SIZE_GAUGE_VEC.remove_label_values(&[
                            "used",
                            region_id_str,
                            keyspace_name_str,
                            "standard",
                        ]);
                        let _ = STORE_SIZE_GAUGE_VEC.remove_label_values(&[
                            "tiflash_used",
                            region_id_str,
                            keyspace_name_str,
                            "standard",
                        ]);
                        let _ = STORE_SIZE_GAUGE_VEC.remove_label_values(&[
                            "used",
                            region_id_str,
                            keyspace_name_str,
                            "ia",
                        ]);
                        let _ = STORE_SIZE_GAUGE_VEC.remove_label_values(&[
                            "tiflash_used",
                            region_id_str,
                            keyspace_name_str,
                            "ia",
                        ]);
                    }
                    Some(size) => {
                        debug!(
                            "update STORE_SIZE_GAUGE_VEC";
                            "region_id_str" => region_id_str,
                            "keyspace_name" => keyspace_name_str,
                            "kv_size"=>size,
                            "has_tiflash_replicas" => has_tiflash_replicas,
                        );
                        // Get storage class from shard if present
                        let storage_class = if let Some(shard) = &shard {
                            match shard.get_storage_class() {
                                StorageClass::Ia => "ia",
                                _ => "standard",
                            }
                        } else {
                            "standard"
                        };
                        STORE_SIZE_GAUGE_VEC
                            .with_label_values(&[
                                if has_tiflash_replicas {
                                    "tiflash_used"
                                } else {
                                    "used"
                                },
                                region_id_str,
                                keyspace_name_str,
                                storage_class,
                            ])
                            .set(size as i64);
                    }
                }
            }
        }
    }

    fn handle_store_heartbeat(
        &mut self,
        mut stats: pdpb::StoreStats,
        store_info: Option<StoreInfo>,
        _send_detailed_report: bool,
    ) {
        let mut report_peers = HashMap::default();
        for (region_id, region_peer) in &mut self.region_peers {
            let read_bytes = region_peer.read_bytes - region_peer.last_store_report_read_bytes;
            let read_keys = region_peer.read_keys - region_peer.last_store_report_read_keys;
            let query_stats = region_peer
                .query_stats
                .sub_query_stats(&region_peer.last_store_report_query_stats);
            region_peer.last_store_report_read_bytes = region_peer.read_bytes;
            region_peer.last_store_report_read_keys = region_peer.read_keys;
            region_peer
                .last_store_report_query_stats
                .fill_query_stats(&region_peer.query_stats);
            if read_bytes < hotspot_byte_report_threshold()
                && read_keys < hotspot_key_report_threshold()
                && query_stats.get_read_query_num() < hotspot_query_num_report_threshold()
            {
                continue;
            }
            let mut read_stat = pdpb::PeerStat::default();
            read_stat.set_region_id(*region_id);
            read_stat.set_read_keys(read_keys);
            read_stat.set_read_bytes(read_bytes);
            read_stat.set_query_stats(query_stats.0);
            report_peers.insert(*region_id, read_stat);
        }

        stats = collect_report_read_peer_stats(HOTSPOT_REPORT_CAPACITY, report_peers, stats);

        let (capacity, used_size, available) = if store_info.is_some() {
            match collect_engine_size(store_info.as_ref()) {
                Some((capacity, used_size, available)) => (capacity, used_size, available),
                None => return,
            }
        } else {
            // Use last recorded statistics to report.
            (
                self.store_stat.engine_last_capacity_size,
                self.store_stat.engine_last_used_size,
                self.store_stat.engine_last_available_size,
            )
        };
        if available == 0 {
            warn!("no available space");
        }
        stats.set_capacity(capacity);
        stats.set_used_size(used_size);
        stats.set_available(available);
        // Update last reported infos on engine_size.
        self.store_stat.engine_last_capacity_size = capacity;
        self.store_stat.engine_last_used_size = used_size;
        self.store_stat.engine_last_available_size = available;

        stats.set_bytes_read(
            self.store_stat.engine_total_bytes_read - self.store_stat.engine_last_total_bytes_read,
        );
        stats.set_keys_read(
            self.store_stat.engine_total_keys_read - self.store_stat.engine_last_total_keys_read,
        );

        self.store_stat
            .engine_total_query_num
            .add_query_stats(stats.get_query_stats()); // add write query stat
        let res = self
            .store_stat
            .engine_total_query_num
            .sub_query_stats(&self.store_stat.engine_last_query_num);
        stats.set_query_stats(res.0);

        stats.set_cpu_usages(self.store_stat.store_cpu_usages.clone().into());
        stats.set_read_io_rates(self.store_stat.store_read_io_rates.clone().into());
        stats.set_write_io_rates(self.store_stat.store_write_io_rates.clone().into());

        let mut interval = pdpb::TimeInterval::default();
        interval.set_start_timestamp(self.store_stat.last_report_ts.into_inner());
        stats.set_interval(interval);
        self.store_stat.engine_last_total_bytes_read = self.store_stat.engine_total_bytes_read;
        self.store_stat.engine_last_total_keys_read = self.store_stat.engine_total_keys_read;
        self.store_stat
            .engine_last_query_num
            .fill_query_stats(&self.store_stat.engine_total_query_num);
        self.store_stat.last_report_ts = UnixSecs::now();
        self.store_stat.region_bytes_written.flush();
        self.store_stat.region_keys_written.flush();
        self.store_stat.region_bytes_read.flush();
        self.store_stat.region_keys_read.flush();

        let mut dfs_measure = None;
        if let Some(store_info) = store_info {
            // Update the timestap for reporting heratbeat.
            // If `store_info` is None, the given Task::StoreHeartbeat should be a fake
            // heartbeat to PD, we won't update the last_report_ts to avoid incorrectly
            // marking current TiKV node in normal state.
            self.store_stat.last_report_ts = UnixSecs::now();

            let dfs_meter = store_info.rf_engine.dfs_stats().clone();
            let measure = dfs_meter.measure();
            let mut rf_dfs_stat = DfsStatItem::default();
            let scope = rf_dfs_stat.mut_scope();
            scope.set_component("rfengine".to_owned());
            scope.set_is_global(true);
            rf_dfs_stat.set_write_requests(measure.request_count);
            rf_dfs_stat.set_written_bytes(measure.uploaded_bytes);
            stats.mut_dfs().push(rf_dfs_stat);
            dfs_measure = Some(measure);
        }

        // Set slow score for this node.
        stats.set_slow_score(self.health_reporter.get_disk_slow_score() as u64);
        // Filter out network slow scores equal to 1 to reduce message volume
        let network_scores = self
            .health_reporter
            .get_network_slow_score()
            .into_iter()
            .filter(|(_, score)| *score != 1)
            .collect();
        stats.set_network_slow_scores(network_scores);

        debug!("Sending store heartbeat."; "stats" => ?stats);
        let optional_report = None;
        let resp = self.pd_client.store_heartbeat(stats, optional_report, None);
        let f = async move {
            match resp.await {
                Ok(_resp) => {
                    if let Some(dfs_measure) = dfs_measure {
                        dfs_measure.acknowledge();
                    }

                    // TODO(x): UpdateReplicationMode
                    // TODO(x): support recovery plan.
                    if GLOBAL_SERVER_READINESS
                        .connected_to_pd
                        .compare_exchange(false, true, Ordering::Relaxed, Ordering::Relaxed)
                        .is_ok()
                    {
                        // Log when the server readiness condition changes.
                        info!("ServerReadiness: connected to PD");
                    }
                }
                Err(e) => {
                    error!("store heartbeat failed"; "err" => ?e);
                }
            }
        };
        self.remote.spawn(f);
    }

    /// Force to send a special heartbeat to pd when current store is hung on
    /// some special circumstances, i.e. disk busy, handler busy and others.
    fn handle_fake_store_heartbeat(&mut self) {
        let mut stats = pdpb::StoreStats::default();
        stats.set_store_id(self.store_id);
        stats.set_region_count(self.region_peers.len() as u32);
        stats.set_start_time(self.start_ts.into_inner() as u32);

        // This calling means that the current node cannot report heartbeat in normaly
        // scheduler. That is, the current node must in `busy` state.
        stats.set_is_busy(true);

        // We do not need to report store_info, so we just set `None` here.
        self.handle_store_heartbeat(stats, None, false);
        warn!("scheduling store_heartbeat timeout, force report store slow score to pd.";
            "store_id" => self.store_id,
        );
    }

    fn is_store_heartbeat_delayed(&self) -> bool {
        let now = UnixSecs::now();
        let interval_second = now.into_inner() - self.store_stat.last_report_ts.into_inner();
        // Only if the `last_report_ts`, that is, the last timestamp of
        // store_heartbeat, exceeds the interval of store heartbaet but less than
        // the given limitation, will it trigger a report of fake heartbeat to
        // make the statistics of slowness percepted by PD timely.
        (interval_second > self.store_heartbeat_interval.as_secs())
            && (interval_second <= STORE_HEARTBEAT_DELAY_LIMIT)
    }

    fn handle_report_batch_split(&self, regions: Vec<metapb::Region>) {
        let resp = self.pd_client.report_batch_split(regions);
        let f = async move {
            if let Err(e) = resp.await {
                warn!("report split failed"; "err" => ?e);
            }
        };
        self.remote.spawn(f);
    }

    fn handle_validate_peer(&self, local_region: metapb::Region, peer: metapb::Peer) {
        let router = self.router.clone();
        let resp = self.pd_client.get_region_by_id(local_region.get_id());
        let tag = self.peer_tag(&local_region);
        let f = async move {
            match resp.await {
                Ok(Some(pd_region)) => {
                    if util::is_epoch_stale(
                        pd_region.get_region_epoch(),
                        local_region.get_region_epoch(),
                    ) {
                        // The local Region epoch is fresher than Region epoch in PD
                        // This means the Region info in PD is not updated to the latest even
                        // after `max_leader_missing_duration`. Something is wrong in the system.
                        // Just add a log here for this situation.
                        info!(
                            "local region epoch is greater the \
                             region epoch in PD ignore validate peer";
                            "region" => tag,
                            "peer_id" => peer.get_id(),
                            "local_region_epoch" => ?local_region.get_region_epoch(),
                            "pd_region_epoch" => ?pd_region.get_region_epoch()
                        );
                        PD_VALIDATE_PEER_COUNTER_VEC
                            .with_label_values(&["region epoch error"])
                            .inc();
                        return;
                    }

                    if pd_region
                        .get_peers()
                        .iter()
                        .all(|p| p.get_id() != peer.get_id())
                    {
                        // Peer is not a member of this Region anymore. Probably it's removed out.
                        // Send it a raft massage to destroy it since it's obsolete.
                        info!(
                            "peer is not a valid member of region, to be \
                             destroyed soon";
                            "region" => tag,
                            "peer_id" => peer.get_id(),
                            "pd_region" => ?pd_region
                        );
                        PD_VALIDATE_PEER_COUNTER_VEC
                            .with_label_values(&["peer stale"])
                            .inc();
                        send_destroy_peer_message(&router, local_region, peer, pd_region);
                    } else {
                        info!(
                            "peer is still a valid member of region";
                            "region" => tag,
                            "peer_id" => peer.get_id(),
                            "pd_region" => ?pd_region
                        );
                        PD_VALIDATE_PEER_COUNTER_VEC
                            .with_label_values(&["peer valid"])
                            .inc();
                    }
                }
                Ok(None) => {
                    // splitted Region has not yet reported to PD.
                    // TODO: handle merge
                }
                Err(e) => {
                    error!("{} get region failed", tag; "err" => ?e);
                }
            }
        };
        self.remote.spawn(f);
    }

    fn schedule_heartbeat_receiver(&mut self) {
        let router = self.router.clone();
        let store_id = self.store_id;

        let fut = self.pd_client
            .handle_region_heartbeat_response(store_id, Box::new(move |mut resp: pdpb::RegionHeartbeatResponse| {
                let region_id = resp.get_region_id();
                let epoch = resp.take_region_epoch();
                let peer = resp.take_target_peer();
                let tag = PeerTag::new(store_id, RegionIdVer::new(region_id, epoch.version));

                if resp.has_change_peer() {
                    PD_HEARTBEAT_COUNTER_VEC
                        .with_label_values(&["change peer"])
                        .inc();

                    let mut change_peer = resp.take_change_peer();
                    info!(
                        "try to change peer";
                        "region" => tag,
                        "change_type" => ?change_peer.get_change_type(),
                        "peer" => ?change_peer.get_peer()
                    );
                    let req = new_change_peer_request(
                        change_peer.get_change_type(),
                        change_peer.take_peer(),
                    );
                    send_admin_request(&router, region_id, epoch, peer, req, Callback::None);
                } else if resp.has_change_peer_v2() {
                    PD_HEARTBEAT_COUNTER_VEC
                        .with_label_values(&["change peer"])
                        .inc();

                    let mut change_peer_v2 = resp.take_change_peer_v2();
                    info!(
                        "try to change peer";
                        "region" => tag,
                        "changes" => ?change_peer_v2.get_changes(),
                        "kind" => ?ConfChangeKind::confchange_kind(change_peer_v2.get_changes().len()),
                    );
                    let req = new_change_peer_v2_request(change_peer_v2.take_changes().into());
                    send_admin_request(&router, region_id, epoch, peer, req, Callback::None);
                } else if resp.has_transfer_leader() {
                    PD_HEARTBEAT_COUNTER_VEC
                        .with_label_values(&["transfer leader"])
                        .inc();

                    let mut transfer_leader = resp.take_transfer_leader();
                    info!(
                        "try to transfer leader";
                        "region" => tag,
                        "from_peer" => ?peer,
                        "to_peer" => ?transfer_leader.get_peer()
                    );
                    let req = new_transfer_leader_request(transfer_leader.take_peer());
                    send_admin_request(&router, region_id, epoch, peer, req, Callback::None);
                } else if resp.has_split_region() {
                    PD_HEARTBEAT_COUNTER_VEC
                        .with_label_values(&["split region"])
                        .inc();

                    let mut split_region = resp.take_split_region();
                    info!("try to split"; "region" => tag, "region_epoch" => ?epoch);
                    let msg = if split_region.get_policy() == pdpb::CheckPolicy::Usekey {
                        CasualMessage::SplitRegion {
                            region_epoch: epoch,
                            split_keys: split_region.take_keys().into(),
                            callback: Callback::None,
                            source: "pd".into(),
                        }
                    } else {
                        CasualMessage::HalfSplitRegion {
                            region_epoch: epoch,
                            policy: split_region.get_policy(),
                            source: "pd",
                        }
                    };
                    router.send(region_id, PeerMsg::CasualMessage(msg));
                } else if resp.has_merge() {
                    PD_HEARTBEAT_COUNTER_VEC.with_label_values(&["merge"]).inc();

                    let merge = resp.take_merge();
                    info!("try to merge"; "region" => tag, "merge" => ?merge);
                    let request = new_merge_request(merge);
                    let req = new_admin_command(region_id, epoch, peer, request);
                    router.send_store(StoreMsg::PrepareMerge {region_id, req, callback: Callback::None });
                } else {
                    PD_HEARTBEAT_COUNTER_VEC.with_label_values(&["noop"]).inc();
                }
            }));
        let f = async move {
            match fut.await {
                Ok(_) => {
                    info!(
                        "region heartbeat response handler exit";
                        "store_id" => store_id,
                    );
                }
                Err(e) => panic!("unexpected error: {:?}", e),
            }
        };
        self.remote.spawn(f);
        self.is_hb_receiver_scheduled = true;
    }

    fn handle_read_stats(&mut self, mut read_stats: ReadStats) {
        for (region_id, region_info) in read_stats.region_infos.iter_mut() {
            let peer_stat = self.region_peers.entry(*region_id).or_default();
            peer_stat.read_bytes += region_info.flow.read_bytes as u64;
            peer_stat.read_keys += region_info.flow.read_keys as u64;
            self.store_stat.engine_total_bytes_read += region_info.flow.read_bytes as u64;
            self.store_stat.engine_total_keys_read += region_info.flow.read_keys as u64;
            peer_stat
                .query_stats
                .add_query_stats(&region_info.query_stats.0);
            self.store_stat
                .engine_total_query_num
                .add_query_stats(&region_info.query_stats.0);
        }
        for (_, region_buckets) in mem::take(&mut read_stats.region_buckets) {
            self.merge_buckets(region_buckets);
        }
        if !read_stats.region_infos.is_empty() {
            self.stats_monitor.maybe_send_read_stats(read_stats);
        }
    }

    fn handle_write_stats(&mut self, mut write_stats: WriteStats) {
        for (region_id, region_info) in write_stats.region_infos.iter_mut() {
            let peer_stat = self.region_peers.entry(*region_id).or_default();
            peer_stat.query_stats.add_query_stats(&region_info.0);
            self.store_stat
                .engine_total_query_num
                .add_query_stats(&region_info.0);
        }
    }

    // Notice: CPU records here we collect are all from the outside RPC workloads,
    // CPU consumption from internal TiKV are not included. Also, since the write
    // path CPU consumption is not large but the logging is complex, the current
    // CPU time for the write path only takes into account the lock checking,
    // which is the read load portion of the write path.
    // TODO: more accurate CPU consumption of a specified region.
    fn handle_region_cpu_records(&mut self, records: Arc<RawRecords>) {
        // Send Region CPU info to AutoSplitController inside the stats_monitor.
        self.stats_monitor.maybe_send_cpu_stats(&records);
        calculate_region_cpu_records(self.store_id, records, &mut self.region_cpu_records);
    }

    fn handle_destroy_peer(&mut self, region_id: u64, keyspace_id: Option<u32>) {
        self.region_map.remove(region_id);
        self.region_buckets.remove(&region_id);
        match self.region_peers.remove(&region_id) {
            None => {}
            Some(_) => {
                let tag = PeerTag::new(self.store_id, RegionIdVer::new(region_id, 0));
                info!("remove peer statistic record in pd"; "region" => tag)
            }
        }
        Self::set_storage_size_metric(region_id, keyspace_id, None, false, None);
    }

    fn handle_store_infos(
        &mut self,
        cpu_usages: RecordPairVec,
        read_io_rates: RecordPairVec,
        write_io_rates: RecordPairVec,
    ) {
        self.store_stat.store_cpu_usages = cpu_usages;
        self.store_stat.store_read_io_rates = read_io_rates;
        self.store_stat.store_write_io_rates = write_io_rates;
    }

    fn handle_update_max_timestamp(
        &mut self,
        region_id: u64,
        initial_status: u64,
        txn_ext: Arc<TxnExt>,
    ) {
        let pd_client = self.pd_client.clone();
        let concurrency_manager = self.concurrency_manager.clone();
        let tag = PeerTag::new(self.store_id, RegionIdVer::new(region_id, 0));
        let f = async move {
            let mut success = false;
            while txn_ext.max_ts_sync_status.load(Ordering::SeqCst) == initial_status {
                match pd_client.get_tso().await {
                    Ok(ts) => {
                        if let Err(e) =
                            concurrency_manager.update_max_ts(ts, "pd_runner_update_max_ts")
                        {
                            error!("failed to update max timestamp for region {}: {:?}", tag, e);
                        }
                        // Set the least significant bit to 1 to mark it as synced.
                        success = txn_ext
                            .max_ts_sync_status
                            .compare_exchange(
                                initial_status,
                                initial_status | 1,
                                Ordering::SeqCst,
                                Ordering::SeqCst,
                            )
                            .is_ok();
                        break;
                    }
                    Err(e) => {
                        warn!("failed to update max timestamp for region {}: {:?}", tag, e);
                        let _ = GLOBAL_TIMER_HANDLE
                            .delay(Instant::now() + Duration::from_secs(1))
                            .compat()
                            .await;
                    }
                }
            }
            if success {
                info!("succeed to update max timestamp"; "region" => tag);
            } else {
                info!(
                    "updating max timestamp is stale";
                    "region" => tag,
                    "initial_status" => initial_status,
                );
            }
        };

        #[cfg(feature = "failpoints")]
        let delay = (|| {
            fail_point!("delay_update_max_ts", |_| true);
            false
        })();
        #[cfg(not(feature = "failpoints"))]
        let delay = false;

        if delay {
            info!("[failpoint] delay update max ts for 1s"; "region" => tag);
            let deadline = Instant::now() + Duration::from_secs(1);
            self.remote
                .spawn(GLOBAL_TIMER_HANDLE.delay(deadline).compat().then(|_| f));
        } else {
            self.remote.spawn(f);
        }
    }

    fn handle_update_gc_safe_point(&mut self) {
        // Avoid multiple concurrent updating process when PD responses slowly.
        if self
            .update_gc_safe_point_in_progress
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_err()
        {
            return;
        }

        let pd_client = self.pd_client.clone();
        let kv = self.kv.clone();
        let update_gc_safe_point_in_progress_flag = self.update_gc_safe_point_in_progress.clone();
        let f = async move {
            defer!(update_gc_safe_point_in_progress_flag.store(false, Ordering::SeqCst));

            match pd_client.get_all_keyspaces_gc_states().await {
                Ok(cluster_gc_states) => {
                    // Update metrics.
                    // Skip updating if the GC safe point is 0, so that if some special keyspaces
                    // (null keyspace or default keyspace) exists but never used, it won't
                    // show up as a zero in the metrics.
                    for (&keyspace_id, gc_state) in &cluster_gc_states.keyspace_gc_states {
                        if keyspace_id == NULL_KEYSPACE_ID {
                            if !gc_state.gc_safe_point.is_zero() {
                                raftstore::store::metrics::AUTO_GC_SAFE_POINT_GAUGE
                                    .set(gc_state.gc_safe_point.into_inner() as i64);
                            }
                        } else if !gc_state.gc_safe_point.is_zero() {
                            let keyspace_name = match to_keyspace_name(keyspace_id) {
                                Some(name) => Either::Left(name),
                                None => Either::Right(format!("<unknown_{}>", keyspace_id)),
                            };
                            let keyspace_name_ref = match keyspace_name {
                                Either::Left(ref name) => name.as_str(),
                                Either::Right(ref name) => name.as_str(),
                            };

                            raftstore::store::metrics::KEYSPACE_GC_SAFE_POINTS_GAUGE_VEC
                                .with_label_values(&[&keyspace_name_ref])
                                .set(gc_state.gc_safe_point.into_inner() as i64);
                        }
                    }

                    info!(
                        "updating gc_safe_point (keyspace, gc_safe_point)";
                        "keyspace_gc_safe_points" => ?cluster_gc_states
                            .keyspace_gc_states
                            .iter()
                            // Do not print null keyspace as it's actually never used in next gen.
                            .filter(|(id, _)| **id != NULL_KEYSPACE_ID)
                            .map(|(k, v)| (*k, v.gc_safe_point))
                            .collect::<Vec<_>>(),
                    );
                    kv.update_cluster_gc_states_cache(cluster_gc_states);
                }
                Err(err) => {
                    warn!(
                        "failed to update gc_safe_point (null_keyspace)";
                        "err" => ?err
                    );
                }
            }
        };
        self.remote.spawn(f);
    }

    fn handle_report_region_buckets(&mut self, mut region_buckets: BucketStat) {
        let store_id = self.store_id;
        let region_id = region_buckets.meta.region_id;
        region_buckets.prepare_report();
        self.merge_buckets(region_buckets);
        let report_buckets = self.region_buckets.get_mut(&region_id).unwrap();
        let last_report_ts = if report_buckets.last_report_ts.is_zero() {
            self.start_ts
        } else {
            report_buckets.last_report_ts
        };
        let now = UnixSecs::now();
        let interval_second = now.into_inner() - last_report_ts.into_inner();
        let delta = report_buckets.new_report(now);
        let resp = self
            .pd_client
            .report_region_buckets(&delta, Duration::from_secs(interval_second));
        let f = async move {
            let tag = || {
                PeerTag::new(
                    store_id,
                    RegionIdVer::new(region_id, delta.meta.region_epoch.version),
                )
            };
            if let Err(e) = resp.await {
                debug!(
                    "{} failed to send buckets", tag();
                    "region_id" => region_id,
                    "version" => delta.meta.version,
                    "region_epoch" => ?delta.meta.region_epoch,
                    "err" => ?e
                );
            } else {
                debug!("{} report_region_buckets", tag();
                    "version" => delta.meta.version,
                    "count" => delta.count(),
                );
            }
        };
        self.remote.spawn(f);
    }

    fn merge_buckets(&mut self, mut buckets: BucketStat) {
        let region_id = buckets.meta.region_id;
        self.region_buckets
            .entry(region_id)
            .and_modify(|report_bucket| {
                let current = &mut report_bucket.current_stat;
                if current.meta < buckets.meta {
                    mem::swap(current, &mut buckets);
                }
                merge_bucket_stats(
                    &current.meta.keys,
                    &mut current.stats,
                    &buckets.meta.keys,
                    &buckets.stats,
                );
            })
            .or_insert_with(|| ReportBucket::new(buckets));
    }

    fn handle_sync_region(
        &self,
        start: Vec<u8>,
        mut end: Vec<u8>,
        limit: usize,
        reverse: bool,
        callback: Box<dyn FnOnce(SyncRegionResponse) + Send>,
    ) {
        if end.is_empty() {
            end.extend_from_slice(GLOBAL_SHARD_END_KEY);
        }
        let regions = self.region_map.scan_regions(start, end, limit, reverse);
        let resp = self.make_sync_region_resp(regions);
        callback(resp);
    }

    fn handle_sync_region_by_id(
        &self,
        region_id: u64,
        callback: Box<dyn FnOnce(SyncRegionResponse) + Send>,
    ) {
        let region = self.region_map.regions.get(&region_id);
        let resp = self.make_sync_region_resp(region.into_iter().collect());
        callback(resp);
    }

    fn make_sync_region_resp(&self, regions: Vec<&Region>) -> SyncRegionResponse {
        let mut resp_regions = Vec::with_capacity(regions.len());
        let mut resp_stats = Vec::with_capacity(regions.len());
        let mut resp_leaders = Vec::with_capacity(regions.len());
        let mut resp_buckets = Vec::with_capacity(regions.len());
        let mut resp_down_peers: Vec<pdpb::PeersStats> = Vec::with_capacity(regions.len());
        let mut resp_pending_peers: Vec<Peers> = Vec::with_capacity(regions.len());
        for region in regions {
            resp_regions.push(region.clone());
            // The stats is used along with region, we need to push a default one if not
            // found.
            let mut region_stat = pdpb::RegionStat::new();
            let mut down_peers = pdpb::PeersStats::new();
            let mut pending_peers = Peers::new();
            if let Some(stats) = self.region_peers.get(&region.id) {
                region_stat.set_bytes_read(stats.read_bytes);
                region_stat.set_keys_read(stats.read_keys);
                region_stat.set_bytes_written(stats.last_region_report_written_bytes);
                region_stat.set_keys_written(stats.last_region_report_written_keys);
                down_peers.peers = stats.down_peers.clone().into();
                pending_peers.peers = stats.pending_peers.clone().into();
            }
            resp_stats.push(region_stat);
            resp_down_peers.push(down_peers);
            resp_pending_peers.push(pending_peers);
            let leader_peer = find_peer(region, self.store_id)
                .cloned()
                .unwrap_or_default();
            resp_leaders.push(leader_peer);
            if !self.region_buckets.is_empty() {
                // If there is any bucket, then all regions must push a bucket even if it's not
                // reported yet.
                let mut bucket = metapb::Buckets::new();
                if let Some(report_bucket) = self.region_buckets.get(&region.id) {
                    bucket.set_region_id(region.id);
                    bucket.set_version(report_bucket.current_stat.meta.version);
                    bucket.set_keys(report_bucket.current_stat.meta.keys.clone().into());
                    bucket.set_stats(report_bucket.current_stat.stats.clone());
                }
                resp_buckets.push(bucket);
            }
        }
        let mut resp = SyncRegionResponse::new();
        resp.mut_header().set_cluster_id(self.cluster_id);
        resp.set_regions(resp_regions.into());
        resp.set_region_leaders(resp_leaders.into());
        resp.set_region_stats(resp_stats.into());
        resp.set_buckets(resp_buckets.into());
        resp.set_down_peers(resp_down_peers.into());
        resp.set_pending_peers(resp_pending_peers.into());
        resp
    }

    fn handle_role_changed(&mut self, region_id: u64, keyspace_id: Option<u32>, role: StateRole) {
        let peer_stat = self.region_peers.entry(region_id).or_default();
        peer_stat.role = role;
        if role != StateRole::Leader {
            peer_stat.down_peers.clear();
            peer_stat.pending_peers.clear();
            Self::set_storage_size_metric(region_id, keyspace_id, None, false, None);
        }
    }

    fn handle_inspect_latency(&mut self, factor: InspectFactor) {
        let slow_score_tick_result = self
            .health_reporter
            .tick(self.store_stat.maybe_busy(), factor);
        if let Some(score) = slow_score_tick_result.updated_score {
            flush_store_inspect_slow_score_metrics(factor, score);
        }
        let id = slow_score_tick_result.tick_id;
        let scheduler = self.scheduler.clone();

        let inspector = {
            match factor {
                InspectFactor::RaftDisk => {
                    // If the last slow_score already reached abnormal state and was delayed for
                    // reporting by `store-heartbeat` to PD, we should report it here manually as
                    // a FAKE `store-heartbeat`.
                    if slow_score_tick_result.should_force_report_slow_store
                        && self.is_store_heartbeat_delayed()
                    {
                        self.handle_fake_store_heartbeat();
                    }
                    LatencyInspector::new(
                        id,
                        Box::new(move |id, duration| {
                            flush_store_inspect_disk_duration_metrics(factor, duration.clone());
                            if let Err(e) = scheduler.schedule(PdTask::UpdateSlowScore {
                                id,
                                factor,
                                duration,
                            }) {
                                warn!("schedule pd task failed"; "err" => ?e);
                            }
                        }),
                    )
                }
                InspectFactor::KvDisk => LatencyInspector::new(
                    id,
                    Box::new(move |id, duration| {
                        flush_store_inspect_disk_duration_metrics(factor, duration.clone());
                        if let Err(e) = scheduler.schedule(PdTask::UpdateSlowScore {
                            id,
                            factor,
                            duration,
                        }) {
                            warn!("schedule pd task failed"; "err" => ?e);
                        }
                    }),
                ),
                InspectFactor::Network => {
                    let network_durations = self.health_reporter.record_network_duration(id);
                    for (store_id, network_duration) in &network_durations {
                        flush_store_inspect_network_duration_metrics(
                            *store_id,
                            tikv_util::time::duration_to_sec(*network_duration),
                        );
                    }
                    // Inspect on network is periodically triggered by pd worker, it's no need to
                    // trigger a new inspector to rfstore.
                    return;
                }
            }
        };
        // Send the inspector to rfstore to trigger the next-round inspection of latency
        // jitters.
        let msg = StoreMsg::LatencyInspect {
            factor,
            send_time: TiInstant::now(),
            inspector,
        };
        self.router.send_store(msg);
    }
}

impl Runnable for PdRunner {
    type Task = PdTask;

    fn run(&mut self, task: PdTask) {
        debug!("executing task"; "task" => %task);

        if !self.is_hb_receiver_scheduled {
            self.schedule_heartbeat_receiver();
        }

        match task {
            PdTask::AskBatchSplit {
                region,
                split_keys,
                peer,
                right_derive,
                callback,
            } => Self::handle_ask_batch_split(
                self.peer_tag(&region),
                self.router.clone(),
                self.scheduler.clone(),
                self.pd_client.clone(),
                region,
                split_keys,
                peer,
                right_derive,
                callback,
                String::from("batch_split"),
                self.remote.clone(),
                self.kv.fs.clone(),
            ),

            PdTask::Heartbeat(hb_task) => {
                tikv_util::set_current_region(hb_task.region.id);
                let (
                    read_bytes_delta,
                    read_keys_delta,
                    written_bytes_delta,
                    written_keys_delta,
                    last_report_ts,
                    query_stats,
                ) = {
                    let peer_stat = self
                        .region_peers
                        .entry(hb_task.region.get_id())
                        .or_default();
                    peer_stat.approximate_size = hb_task.approximate_size;
                    peer_stat.approximate_keys = hb_task.approximate_keys;
                    peer_stat.approximate_kv_size = hb_task.approximate_kv_size;

                    let read_bytes_delta =
                        peer_stat.read_bytes - peer_stat.last_region_report_read_bytes;
                    let read_keys_delta =
                        peer_stat.read_keys - peer_stat.last_region_report_read_keys;
                    let written_bytes_delta =
                        hb_task.written_bytes - peer_stat.last_region_report_written_bytes;
                    let written_keys_delta =
                        hb_task.written_keys - peer_stat.last_region_report_written_keys;
                    let query_stats = peer_stat
                        .query_stats
                        .sub_query_stats(&peer_stat.last_region_report_query_stats);
                    let mut last_report_ts = peer_stat.last_region_report_ts;
                    peer_stat.last_region_report_written_bytes = hb_task.written_bytes;
                    peer_stat.last_region_report_written_keys = hb_task.written_keys;
                    peer_stat.last_region_report_read_bytes = peer_stat.read_bytes;
                    peer_stat.last_region_report_read_keys = peer_stat.read_keys;
                    peer_stat.last_region_report_query_stats = peer_stat.query_stats.clone();
                    let unix_secs_now = UnixSecs::now();
                    peer_stat.last_region_report_ts = unix_secs_now;

                    peer_stat.down_peers = hb_task.down_peers.clone();
                    peer_stat.pending_peers = hb_task.pending_peers.clone();

                    if last_report_ts.is_zero() {
                        last_report_ts = self.start_ts;
                    }
                    (
                        read_bytes_delta,
                        read_keys_delta,
                        written_bytes_delta,
                        written_keys_delta,
                        last_report_ts,
                        query_stats.0,
                    )
                };
                self.handle_heartbeat(
                    hb_task.term,
                    hb_task.region,
                    hb_task.peer,
                    RegionStat {
                        down_peers: hb_task.down_peers,
                        pending_peers: hb_task.pending_peers,
                        written_bytes: written_bytes_delta,
                        written_keys: written_keys_delta,
                        read_bytes: read_bytes_delta,
                        read_keys: read_keys_delta,
                        query_stats,
                        approximate_size: hb_task.approximate_size,
                        approximate_keys: hb_task.approximate_keys,
                        approximate_kv_size: hb_task.approximate_kv_size,
                        approximate_columnar_kv_size: 0,
                        last_report_ts,
                        cpu_usage: 0,
                    },
                    hb_task.replication_status,
                );
                if let Some(bucket_stat) = hb_task.bucket_stat {
                    self.handle_report_region_buckets(bucket_stat);
                }
            }
            PdTask::StoreHeartbeat {
                stats,
                store_info,
                send_detailed_report,
            } => self.handle_store_heartbeat(stats, Some(store_info), send_detailed_report),
            PdTask::ReportBatchSplit { regions } => self.handle_report_batch_split(regions),
            PdTask::ValidatePeer { region, peer } => self.handle_validate_peer(region, peer),
            PdTask::ReadStats { read_stats } => self.handle_read_stats(read_stats),
            PdTask::WriteStats { write_stats } => self.handle_write_stats(write_stats),
            PdTask::RegionCpuRecords(records) => self.handle_region_cpu_records(records),
            PdTask::DestroyPeer {
                region_id,
                keyspace_id,
            } => self.handle_destroy_peer(region_id, keyspace_id),
            PdTask::StoreInfos {
                cpu_usages,
                read_io_rates,
                write_io_rates,
            } => self.handle_store_infos(cpu_usages, read_io_rates, write_io_rates),
            PdTask::UpdateMaxTimestamp {
                region_id,
                initial_status,
                txn_ext,
            } => self.handle_update_max_timestamp(region_id, initial_status, txn_ext),
            PdTask::UpdateGcSafePoint => self.handle_update_gc_safe_point(),
            PdTask::SyncRegion {
                start,
                end,
                limit,
                reverse,
                callback,
            } => {
                self.handle_sync_region(start, end, limit, reverse, callback);
            }
            PdTask::SyncRegionById {
                region_id,
                callback,
            } => {
                self.handle_sync_region_by_id(region_id, callback);
            }
            PdTask::RoleChanged {
                region_id,
                keyspace_id,
                role,
            } => {
                self.handle_role_changed(region_id, keyspace_id, role);
            }
            PdTask::UpdateRaftCpuUtil => {
                self.raft_cpu_collector.update();
            }
            PdTask::InspectLatency { factor } => {
                self.handle_inspect_latency(factor);
            }
            PdTask::UpdateSlowScore {
                id,
                factor,
                duration,
            } => {
                self.health_reporter.record_disk_duration(
                    id,
                    factor,
                    duration,
                    !self.store_stat.maybe_busy(),
                );
            }
        };
    }

    fn shutdown(&mut self) {
        self.stats_monitor.stop();
    }
}

fn calculate_region_cpu_records(
    store_id: u64,
    records: Arc<RawRecords>,
    region_cpu_records: &mut HashMap<u64, u32>,
) {
    for (tag, record) in &records.records {
        let record_store_id = tag.store_id;
        if record_store_id != store_id {
            continue;
        }
        // Reporting a region heartbeat later will clear the corresponding record.
        *region_cpu_records.entry(tag.region_id).or_insert(0) += record.cpu_time;
    }
}

fn collect_engine_size(store_info: Option<&StoreInfo>) -> Option<(u64, u64, u64)> {
    debug_assert!(store_info.is_some());
    Some((
        disk::get_disk_capacity(),
        disk::get_disk_used_size(),
        disk::get_disk_available_size(),
    ))
}

fn new_change_peer_request(change_type: ConfChangeType, peer: metapb::Peer) -> AdminRequest {
    let mut req = AdminRequest::default();
    req.set_cmd_type(AdminCmdType::ChangePeer);
    req.mut_change_peer().set_change_type(change_type);
    req.mut_change_peer().set_peer(peer);
    req
}

fn new_change_peer_v2_request(changes: Vec<pdpb::ChangePeer>) -> AdminRequest {
    let mut req = AdminRequest::default();
    req.set_cmd_type(AdminCmdType::ChangePeerV2);
    let change_peer_reqs = changes
        .into_iter()
        .map(|mut c| {
            let mut cp = ChangePeerRequest::default();
            cp.set_change_type(c.get_change_type());
            cp.set_peer(c.take_peer());
            cp
        })
        .collect();
    let mut cp = ChangePeerV2Request::default();
    cp.set_changes(change_peer_reqs);
    req.set_change_peer_v2(cp);
    req
}

fn new_batch_split_region_request(
    split_keys: Vec<Vec<u8>>,
    ids: Vec<pdpb::SplitId>,
    right_derive: bool,
) -> AdminRequest {
    let mut req = AdminRequest::default();
    req.set_cmd_type(AdminCmdType::BatchSplit);
    req.mut_splits().set_right_derive(right_derive);
    let mut requests = Vec::with_capacity(ids.len());
    for (mut id, key) in ids.into_iter().zip(split_keys) {
        let mut split = SplitRequest::default();
        split.set_split_key(key);
        split.set_new_region_id(id.get_new_region_id());
        split.set_new_peer_ids(id.take_new_peer_ids());
        requests.push(split);
    }
    req.mut_splits().set_requests(requests.into());
    req
}

fn new_transfer_leader_request(peer: metapb::Peer) -> AdminRequest {
    let mut req = AdminRequest::default();
    req.set_cmd_type(AdminCmdType::TransferLeader);
    req.mut_transfer_leader().set_peer(peer);
    req
}

fn new_merge_request(merge: pdpb::Merge) -> AdminRequest {
    let mut req = AdminRequest::default();
    req.set_cmd_type(AdminCmdType::PrepareMerge);
    req.mut_prepare_merge()
        .set_target(merge.get_target().to_owned());
    req
}

fn new_admin_command(
    region_id: u64,
    epoch: metapb::RegionEpoch,
    peer: metapb::Peer,
    request: AdminRequest,
) -> RaftCmdRequest {
    let mut req = RaftCmdRequest::default();
    req.mut_header().set_region_id(region_id);
    req.mut_header().set_region_epoch(epoch);
    req.mut_header().set_peer(peer);
    req.set_admin_request(request);
    req
}

fn send_admin_request(
    router: &RaftRouter,
    region_id: u64,
    epoch: metapb::RegionEpoch,
    peer: metapb::Peer,
    request: AdminRequest,
    callback: Callback,
) {
    let req = new_admin_command(region_id, epoch, peer, request);
    router.send_command(req, TraceContext::default(), callback);
}

/// Sends a raft message to destroy the specified stale Peer
fn send_destroy_peer_message(
    router: &RaftRouter,
    local_region: metapb::Region,
    peer: metapb::Peer,
    pd_region: metapb::Region,
) {
    let mut message = RaftMessage::default();
    message.set_region_id(local_region.get_id());
    message.set_from_peer(peer.clone());
    message.set_to_peer(peer);
    message.set_region_epoch(pd_region.get_region_epoch().clone());
    message.set_is_tombstone(true);
    router.send_raft_msg(message);
}

fn collect_report_read_peer_stats(
    capacity: usize,
    mut report_read_stats: HashMap<u64, pdpb::PeerStat>,
    mut stats: pdpb::StoreStats,
) -> pdpb::StoreStats {
    if report_read_stats.len() < capacity * 3 {
        for (_, read_stat) in report_read_stats {
            stats.peer_stats.push(read_stat);
        }
        return stats;
    }
    let mut keys_topn_report = TopN::new(capacity);
    let mut bytes_topn_report = TopN::new(capacity);
    let mut stats_topn_report = TopN::new(capacity);
    for read_stat in report_read_stats.values() {
        let mut cmp_stat = PeerCmpReadStat::default();
        cmp_stat.region_id = read_stat.region_id;
        let mut key_cmp_stat = cmp_stat.clone();
        key_cmp_stat.report_stat = read_stat.read_keys;
        keys_topn_report.push(key_cmp_stat);
        let mut byte_cmp_stat = cmp_stat.clone();
        byte_cmp_stat.report_stat = read_stat.read_bytes;
        bytes_topn_report.push(byte_cmp_stat);
        let mut query_cmp_stat = cmp_stat.clone();
        query_cmp_stat.report_stat = get_read_query_num(read_stat.get_query_stats());
        stats_topn_report.push(query_cmp_stat);
    }

    for x in keys_topn_report {
        if let Some(report_stat) = report_read_stats.remove(&x.region_id) {
            stats.peer_stats.push(report_stat);
        }
    }

    for x in bytes_topn_report {
        if let Some(report_stat) = report_read_stats.remove(&x.region_id) {
            stats.peer_stats.push(report_stat);
        }
    }

    for x in stats_topn_report {
        if let Some(report_stat) = report_read_stats.remove(&x.region_id) {
            stats.peer_stats.push(report_stat);
        }
    }
    stats
}

fn get_read_query_num(stat: &pdpb::QueryStats) -> u64 {
    stat.get_get() + stat.get_coprocessor() + stat.get_scan()
}

pub async fn prepare_and_persist_encryption_metas(
    pd: &Arc<dyn PdClient>,
    dfs: Arc<dyn kvengine::dfs::Dfs>,
    encryption_cfgs: &Vec<(u32, KeyspaceEncryptionConfig)>,
) -> Result<Vec<kvenginepb::EncryptionMeta>, String> {
    if encryption_cfgs.is_empty() {
        return Ok(Vec::new());
    }
    info!(
        "generating and persisting CMEK encryption metas, encryption_cfgs={:?}",
        encryption_cfgs
    );
    let count = encryption_cfgs.len();
    let mut encryption_metas = Vec::with_capacity(count);
    // Get one TSO for each encryption meta file, which will be used as its file ID.
    let tso = match pd.batch_get_tso(count as u32).await {
        Ok(tso) => tso,
        Err(e) => {
            return Err(format!("failed to get tso for encryption keys: {}", e));
        }
    };
    let first_file_id = tso.into_inner() - count as u64 + 1;
    let rt = dfs.get_runtime();
    // Generate a new encryption meta for each keyspace. This involves calling into
    // KMS to generate a new master key.
    for (idx, (keyspace_id, cfg)) in encryption_cfgs.iter().enumerate() {
        let file_id = first_file_id + idx as u64;
        let meta = build_encryption_meta(*keyspace_id, file_id, cfg, rt).await?;
        encryption_metas.push(meta);
    }

    // Persist the encryption configs configs onto S3.
    for meta in &encryption_metas {
        let mut opts = kvengine::dfs::Options::default();
        opts.file_type = kvengine::dfs::FileType::EncryptionDict;
        let file_id = meta.current.as_ref().unwrap().get_file_id();
        let dfs_clone = dfs.clone();
        let content = meta.write_to_bytes().unwrap();
        let res = rt.spawn(async move { dfs_clone.create(file_id, content.into(), opts).await });
        match res.await {
            Ok(Ok(())) => info!(
                "persisted CMEK encryption meta file in S3";
                "file_id" => file_id,
                "content" => format!("{:#?}", meta),
            ),
            Ok(Err(e)) => {
                return Err(format!("failed to create file in S3: {}", e));
            }
            Err(e) => {
                return Err(format!("failed Tokio join error when writing to S3: {}", e));
            }
        }
    }
    Ok(encryption_metas)
}

pub async fn build_encryption_meta(
    keyspace_id: u32,
    file_id: u64,
    cfg: &KeyspaceEncryptionConfig,
    rt: &tokio::runtime::Runtime,
) -> Result<kvenginepb::EncryptionMeta, String> {
    let master_key_config = MasterKeyConfig {
        key_id: cfg.cmek_id.clone().unwrap_or_default(),
        vendor: cfg.vendor.clone().unwrap_or_default(),
        endpoint: cfg.endpoint.clone().unwrap_or_default(),
        region: cfg.region.clone().unwrap_or_default(),
        cipher_text: "".into(),
    };
    let cfg = master_key_config.clone();
    let res = rt
        .spawn(async move { cfg.generate_new_master_key().await })
        .await;
    match res {
        Ok(Ok((master_key, encrypted_master_key))) => {
            info!(
                "generated CMEK master key for keyspace";
                "keyspace_id" => keyspace_id,
            );

            let data_key = master_key.generate_encryption_key();
            let encrypted_data_key = data_key.export();
            let data_key_id = data_key.get_key_id();

            let mut meta = kvenginepb::EncryptionMeta::default();
            meta.keyspace_id = keyspace_id;
            // Current encryption epoch
            meta.mut_current().set_file_id(file_id);
            meta.mut_current().set_data_key_id(data_key_id);
            meta.mut_current()
                .set_created_at(tikv_util::time::UnixSecs::now().into_inner());
            // Master key
            let mut master_key = kvenginepb::MasterKey::default();
            master_key.set_cmek_id(master_key_config.key_id.clone());
            master_key.set_vendor(master_key_config.vendor.clone());
            master_key.set_region(master_key_config.region.clone());
            master_key.set_endpoint(master_key_config.endpoint.clone());
            master_key.set_ciphertext(encrypted_master_key.to_vec());
            meta.set_master_key(master_key);
            // Data key map.
            let mut data_key = kvenginepb::DataKey::new();
            data_key.set_ciphertext(encrypted_data_key);
            meta.mut_data_keys().insert(data_key_id, data_key);

            Ok(meta)
        }
        Ok(Err(e)) => Err(format!(
            "failed to generate master key, keyspace_id={} {}",
            keyspace_id, e
        )),
        Err(e) => Err(format!(
            "tokio join error during master key generation: {}",
            e
        )),
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use cloud_encryption::KeyspaceEncryptionConfig;
    use futures::executor::block_on;
    use kvengine::dfs::Dfs;
    use pd_client::PdClient;
    use protobuf::Message;
    use security::SecurityConfig;
    use test_cloud_server::oss::ObjectStorageService;
    use test_pd_client::PdWrapper;

    use crate::store::worker::pd::{build_encryption_meta, prepare_and_persist_encryption_metas};

    fn new_sample_cfg() -> KeyspaceEncryptionConfig {
        let mut cfg = KeyspaceEncryptionConfig::default();
        cfg.enabled = true;
        cfg.cmek_id = Some("random".to_string());
        cfg.vendor = Some("test".to_string());
        cfg
    }

    #[test]
    fn test_build_encryption_meta() {
        let keyspace_id = 1;
        let file_id = 2;

        let cfg = new_sample_cfg();
        let rt = tokio::runtime::Runtime::new().unwrap();
        let meta = rt
            .block_on(build_encryption_meta(keyspace_id, file_id, &cfg, &rt))
            .expect("build meta");
        assert_eq!(meta.get_keyspace_id(), keyspace_id);
        assert_eq!(meta.get_current().get_file_id(), file_id);
        assert_eq!(
            meta.get_master_key().get_cmek_id(),
            cfg.cmek_id.as_ref().unwrap()
        );
        assert_eq!(
            meta.get_master_key().get_vendor(),
            cfg.vendor.as_ref().unwrap()
        );
        assert!(!meta.get_data_keys().is_empty());
    }

    #[test]
    fn test_prepare_and_persist_encryption_metas() {
        test_util::init_log_for_test();
        let base_dir = tempfile::Builder::new()
            .prefix("test_cmek")
            .tempdir()
            .unwrap();

        let oss_dir = base_dir.path().join("oss");
        let mut oss = ObjectStorageService::new(oss_dir);
        oss.start_server();
        let dfs_conf = kvengine::dfs::DFSConfig {
            prefix: "load_data".to_string(),
            s3_endpoint: format!("http://127.0.0.1:{}", oss.port()),
            s3_key_id: "admin".to_string(),
            s3_secret_key: "admin".to_string(),
            s3_bucket: "test_cmek".to_string(),
            s3_region: "local".to_string(),
            zstd_compression_level: "3".to_string(),
            ..Default::default()
        };
        let dfs = Arc::new(kvengine::dfs::S3Fs::new(
            dfs_conf.prefix,
            dfs_conf.s3_endpoint,
            dfs_conf.s3_key_id,
            dfs_conf.s3_secret_key,
            dfs_conf.s3_region,
            dfs_conf.s3_bucket,
        ));
        let pd_client: Arc<dyn PdClient> = PdWrapper::new_test(1, &SecurityConfig::default(), None)
            .test_client()
            .unwrap();
        let metas = block_on(prepare_and_persist_encryption_metas(
            &pd_client,
            dfs.clone(),
            &vec![
                (123 /* keyspace_id */, new_sample_cfg()),
                (456 /* keyspace_id */, new_sample_cfg()),
            ],
        ))
        .unwrap();

        for m in &metas {
            assert!(m.get_keyspace_id() > 0);
            assert!(!m.get_data_keys().is_empty());
            // Fetch from S3 and compare.
            let mut opts = kvengine::dfs::Options::default();
            opts.file_type = kvengine::dfs::FileType::EncryptionDict;
            let dfs = dfs.clone();
            let meta_bytes_s3 = dfs
                .get_runtime()
                .block_on(dfs.read_file(m.get_current().get_file_id(), opts))
                .unwrap();
            assert_eq!(m.write_to_bytes().unwrap(), meta_bytes_s3);
        }
    }
}
