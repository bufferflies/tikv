// Copyright 2016 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    collections::{
        BTreeMap,
        Bound::{Excluded, Included, Unbounded},
        HashMap, VecDeque,
    },
    fmt::{self, Display, Formatter},
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        mpsc::SyncSender,
        Arc,
    },
    time::Duration,
    u64,
};

use engine_traits::{DeleteStrategy, KvEngine, Mutable, Range, WriteBatch, CF_LOCK, CF_RAFT};
use fail::fail_point;
use file_system::{IoType, WithIoType};
use kvproto::raft_serverpb::{PeerState, RaftApplyState, RegionLocalState};
use pd_client::PdClient;
use raft::eraftpb::Snapshot as RaftSnapshot;
use tikv_util::{
    box_err, box_try, defer, error, info,
    time::Instant,
    warn,
    worker::{Runnable, RunnableWithTimer},
};
use yatp::{pool::ThreadPool, task::future::TaskCell};

use super::metrics::*;
use crate::{
    coprocessor::CoprocessorHost,
    store::{
        self, check_abort,
        peer_storage::{
            JOB_STATUS_CANCELLED, JOB_STATUS_CANCELLING, JOB_STATUS_FAILED, JOB_STATUS_FINISHED,
            JOB_STATUS_PENDING, JOB_STATUS_RUNNING,
        },
        snap::{plain_file_used, Error, Result, SNAPSHOT_CFS},
        transport::CasualRouter,
        ApplyOptions, CasualMessage, SnapEntry, SnapKey, SnapManager,
    },
};

const CLEANUP_MAX_REGION_COUNT: usize = 64;

const TIFLASH: &str = "tiflash";
const ENGINE: &str = "engine";

/// Region related task
#[derive(Debug)]
pub enum Task<S> {
    Gen {
        region_id: u64,
        last_applied_term: u64,
        last_applied_state: RaftApplyState,
        kv_snap: S,
        canceled: Arc<AtomicBool>,
        notifier: SyncSender<RaftSnapshot>,
        for_balance: bool,
        to_store_id: u64,
    },
    Apply {
        region_id: u64,
        status: Arc<AtomicUsize>,
        peer_id: u64,
    },
    /// Destroy data between [start_key, end_key).
    ///
    /// The actual deletion may be delayed if the engine is overloaded or a
    /// reader is still referencing the data.
    Destroy {
        region_id: u64,
        start_key: Vec<u8>,
        end_key: Vec<u8>,
    },
}

impl<S> Task<S> {
    pub fn destroy(region_id: u64, start_key: Vec<u8>, end_key: Vec<u8>) -> Task<S> {
        Task::Destroy {
            region_id,
            start_key,
            end_key,
        }
    }
}

impl<S> Display for Task<S> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match *self {
            Task::Gen { region_id, .. } => write!(f, "Snap gen for {}", region_id),
            Task::Apply { region_id, .. } => write!(f, "Snap apply for {}", region_id),
            Task::Destroy {
                region_id,
                ref start_key,
                ref end_key,
            } => write!(
                f,
                "Destroy {} [{}, {})",
                region_id,
                log_wrappers::Value::key(start_key),
                log_wrappers::Value::key(end_key)
            ),
        }
    }
}

#[derive(Clone)]
struct StalePeerInfo {
    // the start_key is stored as a key in PendingDeleteRanges
    // below are stored as a value in PendingDeleteRanges
    pub region_id: u64,
    pub end_key: Vec<u8>,
    // Once the oldest snapshot sequence exceeds this, it ensures that no one is
    // reading on this peer anymore. So we can safely call `delete_files_in_range`,
    // which may break the consistency of snapshot, of this peer range.
    pub stale_sequence: u64,
}

/// A structure records all ranges to be deleted with some delay.
/// The delay is because there may be some coprocessor requests related to these
/// ranges.
#[derive(Clone, Default)]
struct PendingDeleteRanges {
    ranges: BTreeMap<Vec<u8>, StalePeerInfo>, // start_key -> StalePeerInfo
}

impl PendingDeleteRanges {
    /// Finds ranges that overlap with [start_key, end_key).
    fn find_overlap_ranges(
        &self,
        start_key: &[u8],
        end_key: &[u8],
    ) -> Vec<(u64, Vec<u8>, Vec<u8>, u64)> {
        let mut ranges = Vec::new();
        // find the first range that may overlap with [start_key, end_key)
        let sub_range = self.ranges.range((Unbounded, Excluded(start_key.to_vec())));
        if let Some((s_key, peer_info)) = sub_range.last() {
            if peer_info.end_key > start_key.to_vec() {
                ranges.push((
                    peer_info.region_id,
                    s_key.clone(),
                    peer_info.end_key.clone(),
                    peer_info.stale_sequence,
                ));
            }
        }

        // find the rest ranges that overlap with [start_key, end_key)
        for (s_key, peer_info) in self
            .ranges
            .range((Included(start_key.to_vec()), Excluded(end_key.to_vec())))
        {
            ranges.push((
                peer_info.region_id,
                s_key.clone(),
                peer_info.end_key.clone(),
                peer_info.stale_sequence,
            ));
        }
        ranges
    }

    /// Gets ranges that overlap with [start_key, end_key).
    pub fn drain_overlap_ranges(
        &mut self,
        start_key: &[u8],
        end_key: &[u8],
    ) -> Vec<(u64, Vec<u8>, Vec<u8>, u64)> {
        let ranges = self.find_overlap_ranges(start_key, end_key);

        for (_, s_key, ..) in &ranges {
            self.ranges.remove(s_key).unwrap();
        }
        ranges
    }

    /// Removes and returns the peer info with the `start_key`.
    fn remove(&mut self, start_key: &[u8]) -> Option<(u64, Vec<u8>, Vec<u8>)> {
        self.ranges
            .remove(start_key)
            .map(|peer_info| (peer_info.region_id, start_key.to_owned(), peer_info.end_key))
    }

    /// Inserts a new range waiting to be deleted.
    ///
    /// Before an insert is called, it must call drain_overlap_ranges to clean
    /// the overlapping range.
    fn insert(
        &mut self,
        region_id: u64,
        start_key: Vec<u8>,
        end_key: Vec<u8>,
        stale_sequence: u64,
    ) {
        if !self.find_overlap_ranges(&start_key, &end_key).is_empty() {
            panic!(
                "[region {}] register deleting data in [{}, {}) failed due to overlap",
                region_id,
                log_wrappers::Value::key(&start_key),
                log_wrappers::Value::key(&end_key),
            );
        }
        let info = StalePeerInfo {
            region_id,
            end_key,
            stale_sequence,
        };
        self.ranges.insert(start_key, info);
    }

    /// Gets all stale ranges info.
    pub fn stale_ranges(&self, oldest_sequence: u64) -> impl Iterator<Item = (u64, &[u8], &[u8])> {
        self.ranges
            .iter()
            .filter(move |&(_, info)| info.stale_sequence < oldest_sequence)
            .map(|(start_key, info)| {
                (
                    info.region_id,
                    start_key.as_slice(),
                    info.end_key.as_slice(),
                )
            })
    }

    pub fn len(&self) -> usize {
        self.ranges.len()
    }
}

struct SnapGenContext<EK, R> {
    engine: EK,
    mgr: SnapManager,
    router: R,
}

impl<EK, R> SnapGenContext<EK, R>
where
    EK: KvEngine,
    R: CasualRouter<EK>,
{
    /// Generates the snapshot of the Region.
    fn generate_snap(
        &self,
        region_id: u64,
        last_applied_term: u64,
        last_applied_state: RaftApplyState,
        kv_snap: EK::Snapshot,
        notifier: SyncSender<RaftSnapshot>,
        for_balance: bool,
        allow_multi_files_snapshot: bool,
    ) -> Result<()> {
        // do we need to check leader here?
        let snap = box_try!(store::do_snapshot::<EK>(
            self.mgr.clone(),
            &self.engine,
            kv_snap,
            region_id,
            last_applied_term,
            last_applied_state,
            for_balance,
            allow_multi_files_snapshot,
        ));
        // Only enable the fail point when the region id is equal to 1, which is
        // the id of bootstrapped region in tests.
        fail_point!("region_gen_snap", region_id == 1, |_| Ok(()));
        if let Err(e) = notifier.try_send(snap) {
            info!(
                "failed to notify snap result, leadership may have changed, ignore error";
                "region_id" => region_id,
                "err" => %e,
            );
        }
        // The error can be ignored as snapshot will be sent in next heartbeat in the
        // end.
        let _ = self
            .router
            .send(region_id, CasualMessage::SnapshotGenerated);
        Ok(())
    }

    /// Handles the task of generating snapshot of the Region. It calls
    /// `generate_snap` to do the actual work.
    fn handle_gen(
        &self,
        region_id: u64,
        last_applied_term: u64,
        last_applied_state: RaftApplyState,
        kv_snap: EK::Snapshot,
        canceled: Arc<AtomicBool>,
        notifier: SyncSender<RaftSnapshot>,
        for_balance: bool,
        allow_multi_files_snapshot: bool,
    ) {
        fail_point!("before_region_gen_snap", |_| ());
        SNAP_COUNTER.generate.start.inc();
        if canceled.load(Ordering::Relaxed) {
            info!("generate snap is canceled"; "region_id" => region_id);
            SNAP_COUNTER.generate.abort.inc();
            return;
        }

        let start = Instant::now();
        let _io_type_guard = WithIoType::new(if for_balance {
            IoType::LoadBalance
        } else {
            IoType::Replication
        });

        if let Err(e) = self.generate_snap(
            region_id,
            last_applied_term,
            last_applied_state,
            kv_snap,
            notifier,
            for_balance,
            allow_multi_files_snapshot,
        ) {
            error!(%e; "failed to generate snap!!!"; "region_id" => region_id,);
            SNAP_COUNTER.generate.fail.inc();
            return;
        }

        SNAP_COUNTER.generate.success.inc();
        SNAP_HISTOGRAM
            .generate
            .observe(start.saturating_elapsed_secs());
    }
}

pub struct Runner<EK, R, T>
where
    EK: KvEngine,
    T: PdClient + 'static,
{
    batch_size: usize,
    use_delete_range: bool,
    clean_stale_tick: usize,
    clean_stale_check_interval: Duration,
    clean_stale_ranges_tick: usize,

    tiflash_stores: HashMap<u64, bool>,
    // we may delay some apply tasks if level 0 files to write stall threshold,
    // pending_applies records all delayed apply task, and will check again later
    pending_applies: VecDeque<Task<EK::Snapshot>>,
    // Ranges that have been logically destroyed at a specific sequence number. We can
    // assume there will be no reader (engine snapshot) newer than that sequence number. Therefore,
    // they can be physically deleted with `DeleteFiles` when we're sure there is no older
    // reader as well.
    // To protect this assumption, before a new snapshot is applied, the overlapping pending ranges
    // must first be removed.
    // The sole purpose of maintaining this list is to optimize deletion with `DeleteFiles`
    // whenever we can. Errors while processing them can be ignored.
    pending_delete_ranges: PendingDeleteRanges,

    engine: EK,
    mgr: SnapManager,
    coprocessor_host: CoprocessorHost<EK>,
    router: R,
    pd_client: Option<Arc<T>>,
    pool: ThreadPool<TaskCell>,
}

impl<EK, R, T> Runner<EK, R, T>
where
    EK: KvEngine,
    R: CasualRouter<EK>,
    T: PdClient + 'static,
{
    fn region_state(&self, region_id: u64) -> Result<RegionLocalState> {
        let region_key = keys::region_state_key(region_id);
        let region_state: RegionLocalState =
            match box_try!(self.engine.get_msg_cf(CF_RAFT, &region_key)) {
                Some(state) => state,
                None => {
                    return Err(box_err!(
                        "failed to get region_state from {}",
                        log_wrappers::Value::key(&region_key)
                    ));
                }
            };
        Ok(region_state)
    }

    fn apply_state(&self, region_id: u64) -> Result<RaftApplyState> {
        let state_key = keys::apply_state_key(region_id);
        let apply_state: RaftApplyState =
            match box_try!(self.engine.get_msg_cf(CF_RAFT, &state_key)) {
                Some(state) => state,
                None => {
                    return Err(box_err!(
                        "failed to get apply_state from {}",
                        log_wrappers::Value::key(&state_key)
                    ));
                }
            };
        Ok(apply_state)
    }

    /// Applies snapshot data of the Region.
    fn apply_snap(&mut self, region_id: u64, peer_id: u64, abort: Arc<AtomicUsize>) -> Result<()> {
        info!("begin apply snap data"; "region_id" => region_id, "peer_id" => peer_id);
        fail_point!("region_apply_snap", |_| { Ok(()) });
        check_abort(&abort)?;

        let mut region_state = self.region_state(region_id)?;
        let region = region_state.get_region().clone();
        let start_key = keys::enc_start_key(&region);
        let end_key = keys::enc_end_key(&region);
        check_abort(&abort)?;
        self.clean_overlap_ranges(start_key, end_key)?;
        check_abort(&abort)?;
        fail_point!("apply_snap_cleanup_range");

        // apply snapshot
        let apply_state = self.apply_state(region_id)?;
        let term = apply_state.get_truncated_state().get_term();
        let idx = apply_state.get_truncated_state().get_index();
        let snap_key = SnapKey::new(region_id, term, idx);
        self.mgr.register(snap_key.clone(), SnapEntry::Applying);
        defer!({
            self.mgr.deregister(&snap_key, &SnapEntry::Applying);
        });
        let mut s = box_try!(self.mgr.get_snapshot_for_applying(&snap_key));
        if !s.exists() {
            return Err(box_err!("missing snapshot file {}", s.path()));
        }
        check_abort(&abort)?;
        let timer = Instant::now();
        let options = ApplyOptions {
            db: self.engine.clone(),
            region: region.clone(),
            abort: Arc::clone(&abort),
            write_batch_size: self.batch_size,
            coprocessor_host: self.coprocessor_host.clone(),
        };
        s.apply(options)?;
        self.coprocessor_host
            .post_apply_snapshot(&region, peer_id, &snap_key, Some(&s));

        // delete snapshot state.
        let mut wb = self.engine.write_batch();
        region_state.set_state(PeerState::Normal);
        box_try!(wb.put_msg_cf(CF_RAFT, &keys::region_state_key(region_id), &region_state));
        box_try!(wb.delete_cf(CF_RAFT, &keys::snapshot_raft_state_key(region_id)));
        wb.write().unwrap_or_else(|e| {
            panic!("{} failed to save apply_snap result: {:?}", region_id, e);
        });
        info!(
            "apply new data";
            "region_id" => region_id,
            "time_takes" => ?timer.saturating_elapsed(),
        );
        Ok(())
    }

    /// Tries to apply the snapshot of the specified Region. It calls
    /// `apply_snap` to do the actual work.
    fn handle_apply(&mut self, region_id: u64, peer_id: u64, status: Arc<AtomicUsize>) {
        let _ = status.compare_exchange(
            JOB_STATUS_PENDING,
            JOB_STATUS_RUNNING,
            Ordering::SeqCst,
            Ordering::SeqCst,
        );
        SNAP_COUNTER.apply.start.inc();

        let start = Instant::now();

        match self.apply_snap(region_id, peer_id, Arc::clone(&status)) {
            Ok(()) => {
                status.swap(JOB_STATUS_FINISHED, Ordering::SeqCst);
                SNAP_COUNTER.apply.success.inc();
            }
            Err(Error::Abort) => {
                warn!("applying snapshot is aborted"; "region_id" => region_id);
                assert_eq!(
                    status.swap(JOB_STATUS_CANCELLED, Ordering::SeqCst),
                    JOB_STATUS_CANCELLING
                );
                SNAP_COUNTER.apply.abort.inc();
            }
            Err(e) => {
                error!(%e; "failed to apply snap!!!");
                status.swap(JOB_STATUS_FAILED, Ordering::SeqCst);
                SNAP_COUNTER.apply.fail.inc();
            }
        }

        SNAP_HISTOGRAM
            .apply
            .observe(start.saturating_elapsed_secs());
        let _ = self.router.send(region_id, CasualMessage::SnapshotApplied);
    }

    /// Tries to clean up files in pending ranges overlapping with the given
    /// bounds. These pending ranges will be removed. Returns an updated range
    /// that also includes these ranges. Caller must ensure the remaining keys
    /// in the returning range will be deleted properly.
    fn clean_overlap_ranges_roughly(
        &mut self,
        mut start_key: Vec<u8>,
        mut end_key: Vec<u8>,
    ) -> (Vec<u8>, Vec<u8>) {
        let overlap_ranges = self
            .pending_delete_ranges
            .drain_overlap_ranges(&start_key, &end_key);
        if overlap_ranges.is_empty() {
            return (start_key, end_key);
        }
        CLEAN_COUNTER_VEC.with_label_values(&["overlap"]).inc();
        let oldest_sequence = self
            .engine
            .get_oldest_snapshot_sequence_number()
            .unwrap_or(u64::MAX);
        let df_ranges: Vec<_> = overlap_ranges
            .iter()
            .filter_map(|(region_id, cur_start, cur_end, stale_sequence)| {
                info!(
                    "delete data in range because of overlap"; "region_id" => region_id,
                    "start_key" => log_wrappers::Value::key(cur_start),
                    "end_key" => log_wrappers::Value::key(cur_end)
                );
                if &start_key > cur_start {
                    start_key = cur_start.clone();
                }
                if &end_key < cur_end {
                    end_key = cur_end.clone();
                }
                if *stale_sequence < oldest_sequence {
                    Some(Range::new(cur_start, cur_end))
                } else {
                    SNAP_COUNTER_VEC
                        .with_label_values(&["overlap", "not_delete_files"])
                        .inc();
                    None
                }
            })
            .collect();
        self.engine
            .delete_ranges_cfs(DeleteStrategy::DeleteFiles, &df_ranges)
            .unwrap_or_else(|e| {
                error!("failed to delete files in range"; "err" => %e);
            });
        (start_key, end_key)
    }

    /// Cleans up data in the given range and all pending ranges overlapping
    /// with it.
    fn clean_overlap_ranges(&mut self, start_key: Vec<u8>, end_key: Vec<u8>) -> Result<()> {
        let (start_key, end_key) = self.clean_overlap_ranges_roughly(start_key, end_key);
        self.delete_all_in_range(&[Range::new(&start_key, &end_key)])
    }

    /// Inserts a new pending range, and it will be cleaned up with some delay.
    fn insert_pending_delete_range(
        &mut self,
        region_id: u64,
        start_key: Vec<u8>,
        end_key: Vec<u8>,
    ) {
        let (start_key, end_key) = self.clean_overlap_ranges_roughly(start_key, end_key);
        info!("register deleting data in range";
            "region_id" => region_id,
            "start_key" => log_wrappers::Value::key(&start_key),
            "end_key" => log_wrappers::Value::key(&end_key),
        );
        let seq = self.engine.get_latest_sequence_number();
        self.pending_delete_ranges
            .insert(region_id, start_key, end_key, seq);
    }

    /// Cleans up stale ranges.
    fn clean_stale_ranges(&mut self) {
        STALE_PEER_PENDING_DELETE_RANGE_GAUGE.set(self.pending_delete_ranges.len() as f64);
        if self.ingest_maybe_stall() {
            return;
        }
        let oldest_sequence = self
            .engine
            .get_oldest_snapshot_sequence_number()
            .unwrap_or(u64::MAX);
        let mut region_ranges: Vec<(u64, Vec<u8>, Vec<u8>)> = self
            .pending_delete_ranges
            .stale_ranges(oldest_sequence)
            .map(|(region_id, s, e)| (region_id, s.to_vec(), e.to_vec()))
            .collect();
        if region_ranges.is_empty() {
            return;
        }
        CLEAN_COUNTER_VEC.with_label_values(&["destroy"]).inc_by(1);
        region_ranges.sort_by(|a, b| a.1.cmp(&b.1));
        region_ranges.truncate(CLEANUP_MAX_REGION_COUNT);
        let ranges: Vec<_> = region_ranges
            .iter()
            .map(|(region_id, start, end)| {
                info!("delete data in range because of stale"; "region_id" => region_id,
                    "start_key" => log_wrappers::Value::key(start),
                    "end_key" => log_wrappers::Value::key(end));
                Range::new(start, end)
            })
            .collect();

        self.engine
            .delete_ranges_cfs(DeleteStrategy::DeleteFiles, &ranges)
            .unwrap_or_else(|e| {
                error!("failed to delete files in range"; "err" => %e);
            });
        if let Err(e) = self.delete_all_in_range(&ranges) {
            error!("failed to cleanup stale range"; "err" => %e);
            return;
        }
        self.engine
            .delete_ranges_cfs(DeleteStrategy::DeleteBlobs, &ranges)
            .unwrap_or_else(|e| {
                error!("failed to delete blobs in range"; "err" => %e);
            });

        for (_, key, _) in region_ranges {
            assert!(
                self.pending_delete_ranges.remove(&key).is_some(),
                "cleanup pending_delete_ranges {} should exist",
                log_wrappers::Value::key(&key)
            );
        }
    }

    /// Checks the number of files at level 0 to avoid write stall after
    /// ingesting sst. Returns true if the ingestion causes write stall.
    fn ingest_maybe_stall(&self) -> bool {
        for cf in SNAPSHOT_CFS {
            // no need to check lock cf
            if plain_file_used(cf) {
                continue;
            }
            if self.engine.ingest_maybe_slowdown_writes(cf).expect("cf") {
                return true;
            }
        }
        false
    }

    fn delete_all_in_range(&self, ranges: &[Range<'_>]) -> Result<()> {
        for cf in self.engine.cf_names() {
            // CF_LOCK usually contains fewer keys than other CFs, so we delete them by key.
            let strategy = if cf == CF_LOCK {
                DeleteStrategy::DeleteByKey
            } else if self.use_delete_range {
                DeleteStrategy::DeleteByRange
            } else {
                DeleteStrategy::DeleteByWriter {
                    sst_path: self.mgr.get_temp_path_for_ingest(),
                }
            };
            box_try!(self.engine.delete_ranges_cf(cf, strategy, ranges));
        }

        Ok(())
    }

    /// Calls observer `pre_apply_snapshot` for every task.
    /// Multiple task can be `pre_apply_snapshot` at the same time.
    fn pre_apply_snapshot(&self, task: &Task<EK::Snapshot>) -> Result<()> {
        let (region_id, abort, peer_id) = match task {
            Task::Apply {
                region_id,
                status,
                peer_id,
            } => (region_id, status.clone(), peer_id),
            _ => panic!("invalid apply snapshot task"),
        };

        let region_state = self.region_state(*region_id)?;
        let apply_state = self.apply_state(*region_id)?;

        check_abort(&abort)?;

        let term = apply_state.get_truncated_state().get_term();
        let idx = apply_state.get_truncated_state().get_index();
        let snap_key = SnapKey::new(*region_id, term, idx);
        let s = box_try!(self.mgr.get_snapshot_for_applying(&snap_key));
        if !s.exists() {
            self.coprocessor_host.pre_apply_snapshot(
                region_state.get_region(),
                *peer_id,
                &snap_key,
                None,
            );
            return Err(box_err!("missing snapshot file {}", s.path()));
        }
        check_abort(&abort)?;
        self.coprocessor_host.pre_apply_snapshot(
            region_state.get_region(),
            *peer_id,
            &snap_key,
            Some(&s),
        );
        Ok(())
    }

    /// Tries to apply pending tasks if there is some.
    fn handle_pending_applies(&mut self, is_timeout: bool) {
        fail_point!("apply_pending_snapshot", |_| {});
        let mut new_batch = true;
        while !self.pending_applies.is_empty() {
            // should not handle too many applies than the number of files that can be
            // ingested. check level 0 every time because we can not make sure
            // how does the number of level 0 files change.
            if self.ingest_maybe_stall() {
                break;
            }
            if let Some(Task::Apply { region_id, .. }) = self.pending_applies.front() {
                fail_point!("handle_new_pending_applies", |_| {});
                if !self.engine.can_apply_snapshot(
                    is_timeout,
                    new_batch,
                    *region_id,
                    self.pending_applies.len(),
                ) {
                    // KvEngine can't apply snapshot for other reasons.
                    break;
                }
                if let Some(Task::Apply {
                    region_id,
                    status,
                    peer_id,
                }) = self.pending_applies.pop_front()
                {
                    new_batch = false;
                    self.handle_apply(region_id, peer_id, status);
                }
            }
        }
    }
}

impl<EK, R, T> Runnable for Runner<EK, R, T>
where
    EK: KvEngine,
    R: CasualRouter<EK> + Send + Clone + 'static,
    T: PdClient,
{
    type Task = Task<EK::Snapshot>;

    fn run(&mut self, task: Task<EK::Snapshot>) {
        match task {
            Task::Gen {
                region_id,
                last_applied_term,
                last_applied_state,
                kv_snap,
                canceled,
                notifier,
                for_balance,
                to_store_id,
            } => {
                // It is safe for now to handle generating and applying snapshot concurrently,
                // but it may not when merge is implemented.
                let mut allow_multi_files_snapshot = false;
                // if to_store_id is 0, it means the to_store_id cannot be found
                if to_store_id != 0 {
                    if let Some(is_tiflash) = self.tiflash_stores.get(&to_store_id) {
                        allow_multi_files_snapshot = !is_tiflash;
                    } else {
                        let is_tiflash = self.pd_client.as_ref().map_or(false, |pd_client| {
                            if let Ok(s) = pd_client.get_store(to_store_id) {
                                if let Some(_l) = s.get_labels().iter().find(|l| {
                                    l.key.to_lowercase() == ENGINE
                                        && l.value.to_lowercase() == TIFLASH
                                }) {
                                    return true;
                                } else {
                                    return false;
                                }
                            }
                            true
                        });
                        self.tiflash_stores.insert(to_store_id, is_tiflash);
                        allow_multi_files_snapshot = !is_tiflash;
                    }
                }
                SNAP_COUNTER.generate.all.inc();
                let ctx = SnapGenContext {
                    engine: self.engine.clone(),
                    mgr: self.mgr.clone(),
                    router: self.router.clone(),
                };
                self.pool.spawn(async move {
                    tikv_alloc::add_thread_memory_accessor();
                    ctx.handle_gen(
                        region_id,
                        last_applied_term,
                        last_applied_state,
                        kv_snap,
                        canceled,
                        notifier,
                        for_balance,
                        allow_multi_files_snapshot,
                    );
                    tikv_alloc::remove_thread_memory_accessor();
                });
            }
            task @ Task::Apply { .. } => {
                fail_point!("on_region_worker_apply", true, |_| {});
                if self.coprocessor_host.should_pre_apply_snapshot() {
                    let _ = self.pre_apply_snapshot(&task);
                }
                SNAP_COUNTER.apply.all.inc();
                // to makes sure applying snapshots in order.
                self.pending_applies.push_back(task);
                self.handle_pending_applies(false);
                if !self.pending_applies.is_empty() {
                    // delay the apply and retry later
                    SNAP_COUNTER.apply.delay.inc()
                }
            }
            Task::Destroy {
                region_id,
                start_key,
                end_key,
            } => {
                fail_point!("on_region_worker_destroy", true, |_| {});
                // try to delay the range deletion because
                // there might be a coprocessor request related to this range
                self.insert_pending_delete_range(region_id, start_key, end_key);
                self.clean_stale_ranges();
            }
        }
    }

    fn shutdown(&mut self) {
        self.pool.shutdown();
    }
}

impl<EK, R, T> RunnableWithTimer for Runner<EK, R, T>
where
    EK: KvEngine,
    R: CasualRouter<EK> + Send + Clone + 'static,
    T: PdClient + 'static,
{
    fn on_timeout(&mut self) {
        self.handle_pending_applies(true);
        self.clean_stale_tick += 1;
        if self.clean_stale_tick >= self.clean_stale_ranges_tick {
            self.clean_stale_ranges();
            self.clean_stale_tick = 0;
        }
    }

    fn get_interval(&self) -> Duration {
        self.clean_stale_check_interval
    }
}
