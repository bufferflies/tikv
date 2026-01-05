// Copyright 2021 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    cell::RefCell,
    cmp::min,
    collections::{hash_map::Entry, HashMap, VecDeque},
    mem::{self, MaybeUninit},
    sync::{Arc, Mutex},
    task::Poll,
    time::{Duration, Instant as StdInstant},
};

use crossbeam::channel::RecvTimeoutError;
use fail::fail_point;
use kvproto::{errorpb, raft_cmdpb::RaftCmdResponse};
use memory::MEMTRACE_APPLY_INFLIGHT;
use pin_project::pin_project;
use raftstore::store::{
    metrics::{
        STORE_WRITE_MIN_WRITE_PAUSE_DURATION_HISTOGRAM, STORE_WRITE_RAFTDB_DURATION_HISTOGRAM,
        STORE_WRITE_SEND_DURATION_HISTOGRAM, STORE_WRITE_TRIGGER_SIZE_HISTOGRAM,
    },
    util,
};
use rfengine::WriteBatch;
use tikv_alloc::TraceEvent;
use tikv_util::{
    debug, error, info,
    mpsc::{Receiver, Sender},
    sys::{thread::StdThreadBuildWrapper, SysQuota},
    time::{duration_to_sec, Instant, InstantExt},
    warn,
    yatp_pool::FuturePool,
};

use super::{metrics::*, *};
use crate::RaftRouter;

// If an `ApplyFuture`'s running time exceeds this threshold, it should
// call `reschedule` to release the thread for other tasks.
const APPLY_FUTURE_RESCHEDURE_DURATION: Duration = Duration::from_millis(5);

#[derive(Clone)]
pub(crate) struct PeerStates {
    pub(crate) applier: Arc<Mutex<Applier>>,
    pub(crate) apply_task_state: Arc<Mutex<ApplyTaskState>>,
    pub(crate) peer_fsm: Arc<Mutex<PeerFsm>>,
}

impl PeerStates {
    pub(crate) fn new(applier: Applier, peer_fsm: PeerFsm) -> Self {
        Self {
            applier: Arc::new(Mutex::new(applier)),
            apply_task_state: Arc::default(),
            peer_fsm: Arc::new(Mutex::new(peer_fsm)),
        }
    }
}

#[allow(clippy::vec_box)]
pub(crate) struct PeerInbox {
    pub(crate) peer: PeerStates,
    pub(crate) msgs: Vec<Box<PeerMsg>>,
}

impl PeerInbox {
    pub(crate) fn process(
        &mut self,
        ctx: &mut RaftContext,
        apply_pool: &FuturePool,
        statistics: Option<&mut [InboxPeerStat]>,
    ) {
        if self.msgs.is_empty() {
            return;
        }
        let mut peer_fsm = self.peer.peer_fsm.lock().unwrap();
        if peer_fsm.stopped {
            return;
        }
        let msg_len = self.msgs.len();
        let region_id = peer_fsm.region_id();
        let start = Instant::now_coarse();
        tikv_util::set_current_region_thread_local(region_id);
        PeerMsgHandler::new(&mut peer_fsm, ctx).handle_msgs(&mut self.msgs);
        peer_fsm.peer.handle_raft_ready(ctx, None);
        if !ctx.apply_msgs.msgs.is_empty() {
            // Update the statistics of applied unpersisted logs.
            let apply_ahead_delta = peer_fsm
                .peer
                .last_applying_idx
                .saturating_sub(peer_fsm.peer.raft_group.raft.r.raft_log.persisted);
            STORE_RAFT_APPLY_AHEAD_PERSIST_HISTOGRAM.observe(apply_ahead_delta as f64);
            let msgs = mem::take(&mut ctx.apply_msgs.msgs);
            maybe_send_apply(
                &self.peer.apply_task_state,
                &self.peer.applier,
                apply_pool,
                msgs,
            );
        }
        let Some(statistics) = statistics else {
            return;
        };
        // Filter out the elapsed time longer than recorded and replace the minimum one
        // if any.
        let elapsed = start.saturating_elapsed();
        // If this peer handle cost is too short, skip the statistics to avoid iterate.
        if elapsed < Duration::from_millis(10) {
            return;
        }
        if let Some(min_index) = statistics
            .iter()
            .enumerate()
            .filter(|(_, &stat)| stat.elapsed < elapsed)
            .min_by_key(|&(_, &val)| val.elapsed)
            .map(|(i, _)| i)
        {
            statistics[min_index].region_id = region_id;
            statistics[min_index].elapsed = elapsed;
            statistics[min_index].msg_cnt = msg_len;
        }
    }
}

pub(crate) struct Inboxes {
    inboxes: HashMap<u64, PeerInbox>,
}

impl Inboxes {
    fn new() -> Self {
        Inboxes {
            inboxes: HashMap::new(),
        }
    }
}

#[derive(Debug, Default, Copy, Clone)]
pub(crate) struct InboxPeerStat {
    pub(crate) region_id: u64,
    pub(crate) elapsed: Duration,
    pub(crate) msg_cnt: usize,
}

pub(crate) struct RaftWorker {
    ctx: StoreContext,
    receiver: Receiver<(u64, Box<PeerMsg>)>,
    router: RaftRouter,
    io_sender: Sender<Option<IoWorkerTask>>,
    last_tick: StdInstant,
    tick_seg_idx: usize,
    tick_millis: u64,
    store_fsm: StoreFsm,
    batch_msg_count: usize,
    max_batch_msg_limit: usize,
    max_batch_msg_size: usize,
    aux_task_senders: Vec<Sender<Vec<PeerInbox>>>,
    aux_res_receivers: Vec<Receiver<()>>,
    sent_aux_task: Vec<bool>,
    aux_handles: Vec<std::thread::JoinHandle<()>>,
    // The cpu usage percent for sum of raft worker thread and aux threads.
    // It is updated by background threads and checked in raft thread to update active aux
    // worker count.
    cpu_util: CpuUtilRef,
    active_aux_count: usize,

    apply_pool: FuturePool,
}

const PEER_INBOX_STATISTIC_COUNT: usize = 5;

impl RaftWorker {
    pub(crate) fn new(
        ctx: StoreContext,
        receiver: Receiver<(u64, Box<PeerMsg>)>,
        router: RaftRouter,
        io_sender: Sender<Option<IoWorkerTask>>,
        store_fsm: StoreFsm,
        cpu_util: CpuUtilRef,
        apply_pool: FuturePool,
    ) -> Self {
        let tick_millis = ctx.cfg.raft_base_tick_interval.as_millis();
        let max_batch_msg_limit = ctx.cfg.store_batch_system.max_batch_size();
        Self {
            ctx,
            receiver,
            router,
            io_sender,
            last_tick: StdInstant::now(),
            tick_seg_idx: 0,
            tick_millis,
            store_fsm,
            batch_msg_count: 0,
            max_batch_msg_limit,
            max_batch_msg_size: max_batch_msg_limit * 1024, /* Use 1 KB as the approximate
                                                             * per-message size */
            aux_task_senders: vec![],
            aux_res_receivers: vec![],
            sent_aux_task: vec![],
            aux_handles: vec![],
            cpu_util,
            active_aux_count: 0,
            apply_pool,
        }
    }

    pub(crate) fn run(&mut self) {
        let mut inboxes: Inboxes = Inboxes::new();
        let mut inbox_peer_stats = vec![InboxPeerStat::default(); PEER_INBOX_STATISTIC_COUNT];
        let store_id = self.ctx.store_id();
        let max_aux_worker_count = SysQuota::cpu_cores_quota().ceil() as usize;
        for i in 0..max_aux_worker_count {
            let (mut aux_worker, aux_result_rx) = RaftAuxWorker::new(
                self.ctx.global.clone(),
                self.io_sender.clone(),
                self.apply_pool.clone(),
            );
            let aux_task_tx = aux_worker.task_tx.clone();
            self.aux_task_senders.push(aux_task_tx);
            self.aux_res_receivers.push(aux_result_rx);
            self.sent_aux_task.push(false);
            let aux_handle = std::thread::Builder::new()
                .name(format!("{}-{}", self.cpu_util.thread_prefix(), i + 1))
                .spawn_wrapper(move || aux_worker.run())
                .unwrap();
            self.aux_handles.push(aux_handle);
        }
        loop {
            let loop_start = Instant::now_coarse();

            self.maybe_update_cfg();
            self.handle_store_msg();
            // Using global histograms is OK since this loop is single-threaded.
            // Consider switching to local histograms if parallelized.
            STORE_STORE_MSG_DURATION_HISTOGRAM
                .observe(duration_to_sec(loop_start.saturating_elapsed()));

            if self.store_fsm.stopped {
                self.stop();
                return;
            }

            if self.receive_msgs(&mut inboxes).is_err() {
                return; // channel closed, exit loop
            }

            // If store msg is empty, we can sync aux_worker after receive latest messages.
            self.sync_aux_worker();

            let send_start = Instant::now_coarse();
            let mut inboxes_vec: Vec<PeerInbox> =
                inboxes.inboxes.drain().map(|(_, inbox)| inbox).collect();
            self.try_send_aux_task(&mut inboxes_vec);
            STORE_SEND_AUX_TASK_DURATION_HISTOGRAM
                .observe(duration_to_sec(send_start.saturating_elapsed()));

            let process_start = Instant::now_coarse();
            inboxes_vec.into_iter().for_each(|mut inbox| {
                inbox.process(&mut self.ctx, &self.apply_pool, Some(&mut inbox_peer_stats));
            });
            let process_inbox_duration = process_start.saturating_elapsed();
            STORE_PROC_MSGS_DURATION_HISTOGRAM.observe(duration_to_sec(process_inbox_duration));

            if process_inbox_duration > Duration::from_millis(50) {
                inbox_peer_stats.sort_by(|a, b| b.elapsed.cmp(&a.elapsed));
                warn!(
                    "store_id: {} raft worker batch loop takes too long, process_inbox_elapsed: {:?}, top_{} peers: {:?}",
                    store_id, process_inbox_duration, PEER_INBOX_STATISTIC_COUNT, inbox_peer_stats
                );
            }
            inbox_peer_stats.fill(InboxPeerStat::default());
            if self.ctx.global.trans.need_flush() {
                self.ctx.global.trans.flush();
            }
            persist_states(&mut self.ctx, &self.io_sender);
            batch_end(&mut self.ctx, loop_start.saturating_elapsed());
            self.update_active_aux_count();
        }
    }

    fn maybe_update_cfg(&mut self) {
        let new_cfg = if let Some(change) = self.ctx.cfg_tracker.any_new() {
            change.clone()
        } else {
            return;
        };
        fail_point!("rfstore_raft_worker_update_cfg");
        let max_batch_msg_limit = new_cfg.store_batch_system.max_batch_size();
        self.max_batch_msg_limit = max_batch_msg_limit;
        // /* Use 1 KB as the approximate per-message size.
        self.max_batch_msg_size = max_batch_msg_limit * 1024;
        self.ctx.cfg = new_cfg;
    }

    fn try_send_aux_task(&mut self, inboxes: &mut Vec<PeerInbox>) {
        if !self.ctx.raft_ctx.raft_wb.is_empty() {
            // The raft_wb is written by handling store messages, do not use aux worker to
            // avoid race.
            return;
        }
        if self.active_aux_count == 0 {
            return;
        }
        if inboxes.len() <= 1 {
            // The aux worker is helpful only when the main raft worker is busy.
            // When there only one inbox, redirect the task to aux worker doesn't
            // have any benefit but increase the latency.
            return;
        }
        let mut aux_inboxes = vec![];
        let mut aux_msg_count = 0;
        let mut aux_worker_idx = 0;
        let target_aux_msg_count = self.batch_msg_count / (self.active_aux_count + 1);
        while let Some(inbox) = inboxes.pop() {
            aux_msg_count += inbox.msgs.len();
            aux_inboxes.push(inbox);
            if aux_msg_count >= target_aux_msg_count {
                self.aux_task_senders[aux_worker_idx]
                    .send(mem::take(&mut aux_inboxes))
                    .unwrap();
                self.sent_aux_task[aux_worker_idx] = true;
                aux_msg_count = 0;
                aux_worker_idx += 1;
                if aux_worker_idx == self.active_aux_count {
                    break;
                }
            }
        }
        if !aux_inboxes.is_empty() {
            self.aux_task_senders[aux_worker_idx]
                .send(mem::take(&mut aux_inboxes))
                .unwrap();
            self.sent_aux_task[aux_worker_idx] = true;
        }
    }

    fn sync_aux_worker(&mut self) {
        let timer = if self.active_aux_count != 0 {
            Some(Instant::now_coarse())
        } else {
            None
        };
        for i in 0..self.ctx.cfg.aux_worker_count {
            if self.sent_aux_task[i] {
                self.aux_res_receivers[i].recv().unwrap();
                self.sent_aux_task[i] = false;
            }
        }
        if let Some(timer) = timer {
            let elapsed = timer.saturating_elapsed();
            if elapsed > Duration::from_millis(100) {
                warn!(
                    "raft worker sync aux worker takes too long";
                    "duration" => ?elapsed,
                );
            }
            STORE_SYNC_AUX_WORKER_DURATION_HISTOGRAM.observe(duration_to_sec(elapsed));
        }
    }

    pub(crate) fn stop(&mut self) {
        let _ = self.io_sender.send(None);
        for peer_map in &self.ctx.peers {
            for peer in peer_map.values() {
                let mut peer_fsm = peer.peer_fsm.lock().unwrap();
                peer_fsm.peer.pending_reads.clear_all(None);
            }
        }
        for aux_task_tx in self.aux_task_senders.drain(..) {
            let _ = aux_task_tx.send(vec![]);
        }
        for handle in self.aux_handles.drain(..) {
            let _ = handle.join();
        }
    }

    fn handle_store_msg(&mut self) {
        while let Ok(msg) = self.store_fsm.receiver.try_recv() {
            self.sync_aux_worker();
            let mut store_handler = StoreMsgHandler::new(&mut self.store_fsm, &mut self.ctx);
            if let Some(apply_region) = store_handler.handle_msg(msg) {
                if !self.ctx.apply_msgs.msgs.is_empty() {
                    let peer = self.ctx.get_peer(apply_region);
                    let msgs = mem::take(&mut self.ctx.apply_msgs.msgs);
                    maybe_send_apply(
                        &peer.apply_task_state,
                        &peer.applier,
                        &self.apply_pool,
                        msgs,
                    );
                }
            }
        }
        if self.store_fsm.last_tick.saturating_elapsed().as_millis() as u64
            > self.store_fsm.tick_millis
        {
            let mut store_handler = StoreMsgHandler::new(&mut self.store_fsm, &mut self.ctx);
            store_handler.handle_msg(StoreMsg::Tick);
            self.store_fsm.last_tick = StdInstant::now();
        }
    }

    /// return true means channel is disconnected, return outer loop.
    fn receive_msgs(&mut self, inboxes: &mut Inboxes) -> std::result::Result<(), RecvTimeoutError> {
        self.batch_msg_count = 0;
        let res = self.receiver.recv_timeout(Duration::from_millis(10));
        let receive_time = Instant::now_coarse();
        match res {
            Ok((region_id, msg)) => {
                let mut batch_size = msg.size();
                self.append_msg(inboxes, region_id, msg);
                while let Ok((region_id, msg)) = self.receiver.try_recv() {
                    batch_size += msg.size();
                    self.append_msg(inboxes, region_id, msg);
                    if self.batch_msg_count > self.max_batch_msg_limit
                        || batch_size > self.max_batch_msg_size
                    {
                        break;
                    }
                }
                // record batch size and batch count metrics
                STORE_RECV_MSGS_COUNT_HISTOGRAM.observe(self.batch_msg_count as f64);
                STORE_RECV_MSGS_SIZE_HISTOGRAM.observe(batch_size as f64);
            }
            Err(RecvTimeoutError::Disconnected) => return Err(RecvTimeoutError::Disconnected),
            Err(RecvTimeoutError::Timeout) => {}
        }
        let tick_elapsed_millis = self.last_tick.saturating_elapsed().as_millis() as u64;
        let next_tick_seg_idx = min(
            (tick_elapsed_millis * PEER_SEGMENTS as u64 / self.tick_millis) as usize,
            PEER_SEGMENTS,
        );
        let mut tick_cnt = 0;
        for seg_idx in self.tick_seg_idx..next_tick_seg_idx {
            let peer_map = &self.ctx.peers[seg_idx];
            peer_map.iter().for_each(
                |(&region_id, peer)| match inboxes.inboxes.entry(region_id) {
                    Entry::Occupied(mut entry) => {
                        entry.get_mut().msgs.push(Box::new(PeerMsg::Tick));
                        tick_cnt += 1;
                    }
                    Entry::Vacant(entry) => {
                        entry.insert(PeerInbox {
                            peer: peer.clone(),
                            msgs: vec![Box::new(PeerMsg::Tick)],
                        });
                        tick_cnt += 1;
                    }
                },
            );
        }
        self.batch_msg_count += tick_cnt;
        if next_tick_seg_idx == PEER_SEGMENTS {
            self.last_tick = StdInstant::now();
            self.tick_seg_idx = 0;
        } else {
            self.tick_seg_idx = next_tick_seg_idx;
        }

        STORE_RECV_MSGS_DURATION_HISTOGRAM
            .observe(duration_to_sec(receive_time.saturating_elapsed()));
        Ok(())
    }

    fn append_msg(&mut self, inboxes: &mut Inboxes, region_id: u64, msg: Box<PeerMsg>) {
        if let Some(inbox) = inboxes.inboxes.get_mut(&region_id) {
            inbox.msgs.push(msg);
            self.batch_msg_count += 1;
            return;
        }
        if let Some(peer) = self.ctx.try_get_peer(region_id) {
            inboxes.inboxes.insert(
                region_id,
                PeerInbox {
                    peer,
                    msgs: vec![msg],
                },
            );
            self.batch_msg_count += 1;
            return;
        }
        match *msg {
            PeerMsg::RaftMessage(msg) => {
                self.router.send_store(StoreMsg::RaftMessage(msg));
            }
            PeerMsg::RaftCommand(cmd) => {
                let mut resp = RaftCmdResponse::default();
                let mut err = errorpb::Error::default();
                err.set_message(format!("region {} is missing", region_id));
                err.mut_region_not_found().set_region_id(region_id);
                resp.mut_header().set_error(err);
                cmd.callback.invoke_with_response(resp);
            }
            PeerMsg::Tick => {}
            PeerMsg::Start => {}
            PeerMsg::ApplyResult(_) => {}
            PeerMsg::CasualMessage(_) => {}
            PeerMsg::SignificantMsg(_) => {}
            PeerMsg::GenerateEngineChangeSet(_) => {}
            PeerMsg::ApplySnapshotResult(_) => {}
            PeerMsg::PrepareChangeSetResult(..) => {}
            PeerMsg::Persisted(_) => {}
            PeerMsg::PrepareCommitMergeResult(..) => {}
            PeerMsg::PrepareTxnFileResult { .. } => {}
            PeerMsg::RaftlogFetched(_) => {}
        }
    }

    fn update_active_aux_count(&mut self) {
        let cfg = &self.ctx.cfg;
        if cfg.aux_worker_count == 0 {
            return;
        }
        let cpu_usage = self.cpu_util.get_cpu_util();
        let origin_active_aux_count = self.active_aux_count;
        if cpu_usage < cfg.main_worker_max_util {
            self.active_aux_count = 0;
        } else {
            let expect_aux_usage = cpu_usage - cfg.main_worker_max_util;
            self.active_aux_count = ((expect_aux_usage + cfg.aux_worker_max_util - 1)
                / cfg.aux_worker_max_util)
                .min(cfg.aux_worker_count);
        }
        if self.active_aux_count != origin_active_aux_count {
            debug!(
                "{}: update use aux worker count to {} on usage {}",
                self.ctx.store_id(),
                self.active_aux_count,
                cpu_usage
            );
        }
    }
}

fn maybe_send_apply(
    pending_state: &Arc<Mutex<ApplyTaskState>>,
    applier: &Arc<Mutex<Applier>>,
    apply_pool: &FuturePool,
    msgs: Vec<ApplyMsg>,
) {
    if msgs.is_empty() {
        return;
    }
    let estimated_size = msgs.iter().map(|m| m.estimated_size()).sum();

    let apply_task = ApplyTask {
        msgs,
        send_time: Instant::now(),
        estimated_size,
    };
    MEMTRACE_APPLY_INFLIGHT.trace(TraceEvent::Add(estimated_size));

    let mut need_spawn = false;
    {
        let mut pending_state_mut = pending_state.lock().unwrap();
        pending_state_mut.pending_msgs.push_back(apply_task);
        if !pending_state_mut.is_running {
            need_spawn = true;
            pending_state_mut.is_running = true;
        }
    }
    if need_spawn {
        let apply_fut = ApplyFuture {
            applier: applier.clone(),
            apply_task_state: pending_state.clone(),
        };

        // this call can never fail as the thread pool use
        // unlimited queue.
        apply_pool.spawn_untracked(apply_fut).unwrap();
    }
}

pub(crate) struct RaftAuxWorker {
    ctx: RaftContext,
    task_tx: Sender<Vec<PeerInbox>>,
    task_rx: Receiver<Vec<PeerInbox>>,
    result_tx: Sender<()>,
    io_sender: Sender<Option<IoWorkerTask>>,
    apply_pool: FuturePool,
}

impl RaftAuxWorker {
    fn new(
        ctx: GlobalContext,
        io_sender: Sender<Option<IoWorkerTask>>,
        apply_pool: FuturePool,
    ) -> (Self, Receiver<()>) {
        let ctx = RaftContext::new(ctx);
        let (task_tx, task_rx) = tikv_util::mpsc::bounded(1);
        let (result_tx, result_rx) = tikv_util::mpsc::bounded(1);
        (
            Self {
                ctx,
                task_tx,
                task_rx,
                result_tx,
                io_sender,
                apply_pool,
            },
            result_rx,
        )
    }

    fn run(&mut self) {
        let store_id = self.ctx.store_id();
        let mut inbox_peer_stats = vec![InboxPeerStat::default(); PEER_INBOX_STATISTIC_COUNT];
        while let Ok(inboxes) = self.task_rx.recv() {
            if inboxes.is_empty() {
                return;
            }
            self.ctx.maybe_refresh_raft_config();
            let process_start = Instant::now();
            for mut inbox in inboxes {
                inbox.process(&mut self.ctx, &self.apply_pool, Some(&mut inbox_peer_stats));
            }
            let process_inbox_duration = process_start.saturating_elapsed();
            if process_inbox_duration > Duration::from_millis(50) {
                inbox_peer_stats.sort_by(|a, b| b.elapsed.cmp(&a.elapsed));
                warn!(
                    "store_id: {} raft aux worker batch loop takes too long, process_inbox_elapsed: {:?}, top_{} peers: {:?}",
                    store_id, process_inbox_duration, PEER_INBOX_STATISTIC_COUNT, inbox_peer_stats
                );
            }
            inbox_peer_stats.fill(InboxPeerStat::default());
            if self.ctx.global.trans.need_flush() {
                self.ctx.global.trans.flush();
            }
            persist_states(&mut self.ctx, &self.io_sender);
            self.result_tx.send(()).unwrap();
            batch_end(&mut self.ctx, process_start.saturating_elapsed());
        }
    }
}

fn persist_states(ctx: &mut RaftContext, io_sender: &Sender<Option<IoWorkerTask>>) {
    if ctx.persist_readies.is_empty() && ctx.raft_wb.is_empty() {
        return;
    }
    let mut raft_wb = mem::take(&mut ctx.raft_wb);
    let remove_dependents = mem::take(&mut ctx.remove_dependents);
    ctx.global.engines.raft.apply(&mut raft_wb);
    let readies = mem::take(&mut ctx.persist_readies);
    let io_task = IoTask {
        raft_wb,
        readies,
        remove_dependents,
    };
    io_sender.send(Some(IoWorkerTask::IoTask(io_task))).unwrap();
}

fn batch_end(ctx: &mut RaftContext, batch_duration: Duration) {
    if batch_duration > Duration::from_millis(50) {
        info!("raft worker batch loop takes {:?}", batch_duration);
    }
    let dur = duration_to_sec(batch_duration);
    ctx.raft_metrics.store_time.observe(dur);
    ctx.raft_metrics.process_ready.observe(dur);
    ctx.raft_metrics.maybe_flush();
    ctx.current_time = None;
    ctx.global.destroying.clear();
}

pub(crate) struct IoWorker {
    // this field is used for failpoint test.
    #[allow(dead_code)]
    store_id: u64,
    engine: rfengine::RfEngine,
    receiver: Receiver<Option<IoWorkerTask>>,
    router: RaftRouter,
    trans: Box<dyn Transport>,
    wb: WriteBatch,
    max_batch_size: usize,
    // min write duration is used to avoid too frequent write to raft db to avoid write
    // amplification.
    min_write_duration: Duration,
}

impl IoWorker {
    pub(crate) fn new(
        store_id: u64,
        engine: rfengine::RfEngine,
        router: RaftRouter,
        trans: Box<dyn Transport>,
        max_batch_size: usize,
        min_write_duration: Duration,
        io_notify_capacity: usize,
    ) -> (Self, Sender<Option<IoWorkerTask>>) {
        let (sender, receiver) = if io_notify_capacity == 0 {
            // Represents that there is no limit for io notification channel.
            tikv_util::mpsc::unbounded()
        } else {
            tikv_util::mpsc::bounded(io_notify_capacity)
        };
        (
            Self {
                store_id,
                engine,
                receiver,
                router,
                trans,
                wb: Default::default(),
                max_batch_size,
                min_write_duration,
            },
            sender,
        )
    }

    pub(crate) fn run(&mut self) {
        while let Ok(Some(t)) = self.receiver.recv() {
            match t {
                IoWorkerTask::IoTask(io_task) => {
                    let len = self.receiver.len();
                    let mut tasks = Vec::with_capacity(len + 1);

                    let mut total_estimated_size: usize = io_task.raft_wb.estimated_size();
                    tasks.push(io_task);
                    for _ in 0..len {
                        if total_estimated_size >= self.max_batch_size {
                            break;
                        }

                        let task = self.receiver.recv().unwrap();
                        if task.is_none() {
                            return;
                        }
                        match task.unwrap() {
                            IoWorkerTask::IoTask(task) => {
                                total_estimated_size += task.raft_wb.estimated_size();
                                tasks.push(task);
                            }
                            IoWorkerTask::UpdateConfig {
                                max_batch_size,
                                min_write_duration,
                            } => {
                                self.update_config(max_batch_size, min_write_duration);
                            }
                        }
                    }
                    for task in &mut tasks {
                        if !task.raft_wb.is_empty() {
                            let wb = mem::take(&mut task.raft_wb);
                            self.wb.merge_write_batch(wb);
                        }
                    }
                    self.handle_tasks(tasks);
                }
                IoWorkerTask::UpdateConfig {
                    max_batch_size,
                    min_write_duration,
                } => {
                    self.update_config(max_batch_size, min_write_duration);
                }
            }
        }
    }

    fn update_config(
        &mut self,
        max_batch_size: Option<usize>,
        min_write_duration: Option<Duration>,
    ) {
        fail::fail_point!("rfstore_io_worker_update_cfg");
        if let Some(batch_size) = max_batch_size {
            self.max_batch_size = batch_size;
        }
        if let Some(dur) = min_write_duration {
            self.min_write_duration = dur;
        }
    }

    fn handle_tasks(&mut self, tasks: Vec<IoTask>) {
        fail_point!("rfstore_before_save_on_store_1", self.store_id == 1, |_| {});
        let timer = Instant::now();
        let mut after_write = timer;
        if !self.wb.is_empty() {
            let wb = mem::take(&mut self.wb);
            let write_size = self.engine.persist(wb).unwrap();
            after_write = Instant::now();
            let write_raft_db_dur = after_write.saturating_duration_since(timer);
            if write_raft_db_dur > Duration::from_millis(40) {
                info!(
                    "io worker write raft db takes too long {:?}， size {}",
                    write_raft_db_dur, write_size
                );
            }
            STORE_WRITE_RAFTDB_DURATION_HISTOGRAM.observe(duration_to_sec(write_raft_db_dur));
            STORE_WRITE_TRIGGER_SIZE_HISTOGRAM.observe(write_size as f64);
        }
        for task in tasks {
            for mut ready in task.readies {
                let raft_messages = mem::take(&mut ready.raft_messages);
                for msg in raft_messages {
                    debug!(
                        "follower send raft message";
                        "region_id" => msg.region_id,
                        "message_type" => %util::MsgType(&msg),
                        "from_peer_id" => msg.get_from_peer().get_id(),
                        "to_peer_id" => msg.get_to_peer().get_id(),
                    );
                    if let Err(err) = self.trans.send(msg) {
                        error!("failed to send persist raft message {:?}", err);
                    }
                }
                let region_id = ready.region_id;
                self.router.send(region_id, PeerMsg::Persisted(ready));
            }

            for (parent_id, dependent_id) in task.remove_dependents {
                remove_dependent(&self.engine, &self.router, parent_id, dependent_id);
            }
        }
        if self.trans.need_flush() {
            self.trans.flush();
        }
        let end_time = Instant::now();
        let handle_time = end_time.saturating_duration_since(timer);
        if self.min_write_duration > handle_time {
            let duration = self.min_write_duration - handle_time;
            std::thread::sleep(duration);
            STORE_WRITE_MIN_WRITE_PAUSE_DURATION_HISTOGRAM.observe(duration.as_secs_f64());
        }
        let send_time = duration_to_sec(end_time.saturating_duration_since(after_write));
        STORE_WRITE_SEND_DURATION_HISTOGRAM.observe(send_time);
    }
}

pub(crate) struct ApplyTask {
    pub(crate) msgs: Vec<ApplyMsg>,
    pub(crate) send_time: Instant,
    pub(crate) estimated_size: usize,
}

#[derive(Default)]
pub(crate) struct ApplyTaskState {
    pub(crate) pending_msgs: VecDeque<ApplyTask>,
    /// Indicates whether the corresponding `ApplyFuture` to this region is
    /// currently running.
    ///
    /// **Lifecycle & Responsibility:**
    /// - The `PeerWorker` must **only** set this to `true` and spawn a new
    ///   `ApplyFuture` when its current value is `false`.
    /// - The `ApplyFuture` itself **must** reset this flag to `false` upon
    ///   completion.
    pub(crate) is_running: bool,
}

thread_local! {
    pub(crate) static APPLY_CTX_STATE: RefCell<MaybeUninit<ApplyContext>> = RefCell::new(MaybeUninit::uninit());
}

#[pin_project]
pub(crate) struct ApplyFuture {
    pub(crate) applier: Arc<Mutex<Applier>>,
    apply_task_state: Arc<Mutex<ApplyTaskState>>,
}

impl std::future::Future for ApplyFuture {
    type Output = ();
    fn poll(self: std::pin::Pin<&mut Self>, cx: &mut std::task::Context<'_>) -> Poll<Self::Output> {
        let this = self.project();
        let start = Instant::now_coarse();

        let apply_ctx = APPLY_CTX_STATE.with_borrow_mut(|c| {
            // SAFETY: this call is safe because we have already init the ApplyContext when
            // build the apply pool.
            unsafe {
                // NOTE: we can't directly use `c.assume_init_mut()` because the compile will
                // argue the lifetime is not long enough for `apply_ctx`.
                &mut *c.as_mut_ptr()
            }
        });
        let mut count = 0;
        let mut res = Poll::Ready(());
        loop {
            let mut batch = {
                let mut pending_state = this.apply_task_state.lock().unwrap();
                assert_eq!(pending_state.is_running, true);
                match pending_state.pending_msgs.pop_front() {
                    Some(t) => {
                        // explicitly reschedule if this task runs for too long and there are still
                        // pending tasks.
                        if count > 0
                            && start.saturating_elapsed() > APPLY_FUTURE_RESCHEDURE_DURATION
                        {
                            let mut reschedule_fut = yatp::task::future::reschedule();
                            if std::pin::pin!(reschedule_fut).poll(cx).is_pending() {
                                pending_state.pending_msgs.push_front(t);
                                drop(pending_state);
                                res = Poll::Pending;
                                break;
                            }
                        }
                        t
                    }
                    None => {
                        pending_state.is_running = false;
                        break;
                    }
                }
            };
            count += 1;

            let mut applier = this.applier.lock().unwrap();

            MEMTRACE_APPLY_INFLIGHT.trace(TraceEvent::Sub(batch.estimated_size));
            let timer = Instant::now();
            apply_ctx.apply_wait.observe(duration_to_sec(
                timer.saturating_duration_since(batch.send_time),
            ));
            tikv_util::set_current_region_thread_local(applier.region_id());
            for msg in batch.msgs.drain(..) {
                applier.handle_msg(apply_ctx, msg);
            }

            applier.update_memory_trace(&mut Default::default());
            apply_ctx.maybe_flush_metrics();
        }
        let total_duration = start.saturating_elapsed_secs();
        apply_ctx.fut_handle_batch_count.observe(count as f64);
        apply_ctx.future_exec_dur.observe(total_duration);
        res
    }
}
