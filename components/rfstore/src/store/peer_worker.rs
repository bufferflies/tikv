// Copyright 2021 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    cmp::min,
    collections::{hash_map::Entry, HashMap},
    mem,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use crossbeam::channel::RecvTimeoutError;
use kvproto::{errorpb, raft_cmdpb::RaftCmdResponse};
use raftstore::store::{
    metrics::{
        STORE_WRITE_RAFTDB_DURATION_HISTOGRAM, STORE_WRITE_SEND_DURATION_HISTOGRAM,
        STORE_WRITE_TRIGGER_SIZE_HISTOGRAM,
    },
    util,
};
use rfengine::WriteBatch;
use tikv_util::{
    debug, error, info,
    mpsc::{Receiver, Sender},
    time::{duration_to_sec, InstantExt},
    warn,
};

use super::*;
use crate::RaftRouter;

#[derive(Clone)]
pub(crate) struct PeerStates {
    pub(crate) applier: Arc<Mutex<Applier>>,
    pub(crate) peer_fsm: Arc<Mutex<PeerFsm>>,
}

impl PeerStates {
    pub(crate) fn new(applier: Applier, peer_fsm: PeerFsm) -> Self {
        Self {
            applier: Arc::new(Mutex::new(applier)),
            peer_fsm: Arc::new(Mutex::new(peer_fsm)),
        }
    }
}

pub(crate) struct PeerInbox {
    pub(crate) peer: PeerStates,
    pub(crate) msgs: Vec<PeerMsg>,
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
    receiver: Receiver<(u64, PeerMsg)>,
    router: RaftRouter,
    apply_senders: Vec<Sender<Option<ApplyBatch>>>,
    io_sender: Sender<Option<IoTask>>,
    last_tick: Instant,
    tick_seg_idx: usize,
    tick_millis: u64,
    store_fsm: StoreFsm,
}

const MAX_BATCH_COUNT: usize = 1024;
const MAX_BATCH_SIZE: usize = 1024 * 1024;
const PEER_INBOX_STATISTIC_COUNT: usize = 5;

impl RaftWorker {
    pub(crate) fn new(
        ctx: StoreContext,
        receiver: Receiver<(u64, PeerMsg)>,
        router: RaftRouter,
        io_sender: Sender<Option<IoTask>>,
        store_fsm: StoreFsm,
    ) -> (Self, Vec<Receiver<Option<ApplyBatch>>>) {
        let apply_pool_size = ctx.cfg.apply_pool_size;
        let mut apply_senders = Vec::with_capacity(apply_pool_size);
        let mut apply_receivers = Vec::with_capacity(apply_pool_size);
        for _ in 0..apply_pool_size {
            let (sender, receiver) = tikv_util::mpsc::unbounded();
            apply_senders.push(sender);
            apply_receivers.push(receiver);
        }
        let tick_millis = ctx.cfg.raft_base_tick_interval.as_millis();
        (
            Self {
                ctx,
                receiver,
                router,
                apply_senders,
                io_sender,
                last_tick: Instant::now(),
                tick_seg_idx: 0,
                tick_millis,
                store_fsm,
            },
            apply_receivers,
        )
    }

    pub(crate) fn run(&mut self) {
        let mut inboxes: Inboxes = Inboxes::new();
        let mut inbox_peer_stats = vec![InboxPeerStat::default(); PEER_INBOX_STATISTIC_COUNT];
        let store_id = self.ctx.store_id();
        loop {
            self.handle_store_msg();
            if self.store_fsm.stopped {
                self.stop();
                return;
            }

            let loop_start = match self.receive_msgs(&mut inboxes) {
                Ok(start_time) => start_time,
                Err(_) => return,
            };
            inboxes.inboxes.iter_mut().for_each(|(_, inbox)| {
                self.process_inbox(inbox, &mut inbox_peer_stats);
            });
            let process_inbox_elapsed = loop_start.saturating_elapsed();
            if process_inbox_elapsed > Duration::from_millis(50) {
                inbox_peer_stats.sort_by(|a, b| b.elapsed.cmp(&a.elapsed));
                warn!(
                    "store_id: {} raft worker batch loop takes too long, process_inbox_elapsed: {:?}, top_{} peers: {:?}",
                    store_id, process_inbox_elapsed, PEER_INBOX_STATISTIC_COUNT, inbox_peer_stats
                );
            }
            inbox_peer_stats.fill(InboxPeerStat::default());
            if self.ctx.global.trans.need_flush() {
                self.ctx.global.trans.flush();
            }
            self.persist_state();
            self.batch_end(loop_start.saturating_elapsed());
        }
    }

    pub(crate) fn stop(&mut self) {
        self.apply_senders.iter().for_each(|sender| {
            let _ = sender.send(None);
        });
        let _ = self.io_sender.send(None);
        for peer_map in &self.ctx.peers {
            for peer in peer_map.values() {
                let mut peer_fsm = peer.peer_fsm.lock().unwrap();
                peer_fsm.peer.pending_reads.clear_all(None);
            }
        }
    }

    fn handle_store_msg(&mut self) {
        while let Ok(msg) = self.store_fsm.receiver.try_recv() {
            let mut store_handler = StoreMsgHandler::new(&mut self.store_fsm, &mut self.ctx);
            if let Some(apply_region) = store_handler.handle_msg(msg) {
                let peer = self.ctx.get_peer(apply_region);
                let peer_fsm = peer.peer_fsm.lock().unwrap();
                let applier = peer.applier.clone();
                self.maybe_send_apply(&applier, &peer_fsm);
            }
        }
        if self.store_fsm.last_tick.saturating_elapsed().as_millis() as u64
            > self.store_fsm.tick_millis
        {
            let mut store_handler = StoreMsgHandler::new(&mut self.store_fsm, &mut self.ctx);
            store_handler.handle_msg(StoreMsg::Tick);
            self.store_fsm.last_tick = Instant::now();
        }
    }

    /// return true means channel is disconnected, return outer loop.
    fn receive_msgs(
        &mut self,
        inboxes: &mut Inboxes,
    ) -> std::result::Result<tikv_util::time::Instant, RecvTimeoutError> {
        inboxes.inboxes.retain(|_, inbox| -> bool {
            if inbox.msgs.is_empty() {
                false
            } else {
                inbox.msgs.truncate(0);
                true
            }
        });
        let res = self.receiver.recv_timeout(Duration::from_millis(10));
        let receive_time = tikv_util::time::Instant::now();
        match res {
            Ok((region_id, msg)) => {
                let mut batch_size = msg.size();
                let mut batch_cnt = 1;
                self.append_msg(inboxes, region_id, msg);
                while let Ok((region_id, msg)) = self.receiver.try_recv() {
                    batch_size += msg.size();
                    batch_cnt += 1;
                    self.append_msg(inboxes, region_id, msg);
                    if batch_cnt > MAX_BATCH_COUNT || batch_size > MAX_BATCH_SIZE {
                        break;
                    }
                }
            }
            Err(RecvTimeoutError::Disconnected) => return Err(RecvTimeoutError::Disconnected),
            Err(RecvTimeoutError::Timeout) => {}
        }
        let tick_elapsed_millis = self.last_tick.saturating_elapsed().as_millis() as u64;
        let next_tick_seg_idx = min(
            (tick_elapsed_millis * PEER_SEGMENTS as u64 / self.tick_millis) as usize,
            PEER_SEGMENTS,
        );
        for seg_idx in self.tick_seg_idx..next_tick_seg_idx {
            let peer_map = &self.ctx.peers[seg_idx];
            peer_map.iter().for_each(
                |(&region_id, peer)| match inboxes.inboxes.entry(region_id) {
                    Entry::Occupied(mut entry) => {
                        entry.get_mut().msgs.push(PeerMsg::Tick);
                    }
                    Entry::Vacant(entry) => {
                        entry.insert(PeerInbox {
                            peer: peer.clone(),
                            msgs: vec![PeerMsg::Tick],
                        });
                    }
                },
            );
        }
        if next_tick_seg_idx == PEER_SEGMENTS {
            self.last_tick = Instant::now();
            self.tick_seg_idx = 0;
        } else {
            self.tick_seg_idx = next_tick_seg_idx;
        }
        Ok(receive_time)
    }

    fn append_msg(&mut self, inboxes: &mut Inboxes, region_id: u64, msg: PeerMsg) {
        if let Some(inbox) = inboxes.inboxes.get_mut(&region_id) {
            inbox.msgs.push(msg);
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
            return;
        }
        match msg {
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
        }
    }

    fn process_inbox(&mut self, inbox: &mut PeerInbox, statistics: &mut [InboxPeerStat]) {
        if inbox.msgs.is_empty() {
            return;
        }
        let mut peer_fsm = inbox.peer.peer_fsm.lock().unwrap();
        if peer_fsm.stopped {
            return;
        }
        let msg_len = inbox.msgs.len();
        let region_id = peer_fsm.region_id();
        let start = tikv_util::time::Instant::now_coarse();
        tikv_util::set_current_region(region_id);
        PeerMsgHandler::new(&mut peer_fsm, &mut self.ctx).handle_msgs(&mut inbox.msgs);
        peer_fsm.peer.handle_raft_ready(&mut self.ctx, None);
        self.maybe_send_apply(&inbox.peer.applier, &peer_fsm);

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

    fn maybe_send_apply(&mut self, applier: &Arc<Mutex<Applier>>, peer_fsm: &PeerFsm) {
        if !self.ctx.apply_msgs.msgs.is_empty() {
            let peer_batch = ApplyBatch {
                msgs: mem::take(&mut self.ctx.apply_msgs.msgs),
                applier: applier.clone(),
                applying_cnt: peer_fsm.applying_cnt.clone(),
                send_time: tikv_util::time::Instant::now(),
            };
            peer_batch
                .applying_cnt
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            self.apply_senders[peer_fsm.apply_worker_idx]
                .send(Some(peer_batch))
                .unwrap();
        }
    }

    fn persist_state(&mut self) {
        if self.ctx.persist_readies.is_empty() && self.ctx.raft_wb.is_empty() {
            return;
        }
        let mut raft_wb = mem::take(&mut self.ctx.raft_wb);
        self.ctx.global.engines.raft.apply(&mut raft_wb);
        let readies = mem::take(&mut self.ctx.persist_readies);
        let io_task = IoTask { raft_wb, readies };
        self.io_sender.send(Some(io_task)).unwrap();
    }

    fn batch_end(&mut self, batch_duration: Duration) {
        if batch_duration > Duration::from_millis(50) {
            info!("raft worker batch loop takes {:?}", batch_duration);
        }
        self.ctx
            .raft_metrics
            .store_time
            .observe(duration_to_sec(batch_duration));
        self.ctx.raft_metrics.maybe_flush();
        self.ctx.current_time = None;
        self.ctx.global.destroying.clear();
    }
}

pub(crate) struct ApplyWorker {
    ctx: ApplyContext,
    receiver: Receiver<Option<ApplyBatch>>,
}

impl ApplyWorker {
    pub(crate) fn new(
        engine: kvengine::Engine,
        router: RaftRouter,
        receiver: Receiver<Option<ApplyBatch>>,
    ) -> Self {
        let ctx = ApplyContext::new(engine, Some(router));
        Self { ctx, receiver }
    }

    pub(crate) fn run(&mut self) {
        let mut loop_cnt = 0u64;
        while let Ok(Some(mut batch)) = self.receiver.recv() {
            let timer = tikv_util::time::Instant::now();
            self.ctx.apply_wait.observe(duration_to_sec(
                timer.saturating_duration_since(batch.send_time),
            ));
            let mut applier = batch.applier.lock().unwrap();
            tikv_util::set_current_region(applier.region_id());
            for msg in batch.msgs.drain(..) {
                applier.handle_msg(&mut self.ctx, msg);
            }
            batch
                .applying_cnt
                .fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
            loop_cnt += 1;
            if loop_cnt % 128 == 0 {
                self.ctx.apply_wait.flush();
                self.ctx.apply_time.flush();
            }
        }
    }
}

pub(crate) struct IoWorker {
    engine: rfengine::RfEngine,
    receiver: Receiver<Option<IoTask>>,
    router: RaftRouter,
    trans: Box<dyn Transport>,
    wb: WriteBatch,
}

impl IoWorker {
    pub(crate) fn new(
        engine: rfengine::RfEngine,
        router: RaftRouter,
        trans: Box<dyn Transport>,
    ) -> (Self, Sender<Option<IoTask>>) {
        let (sender, receiver) = tikv_util::mpsc::bounded(256);
        (
            Self {
                engine,
                receiver,
                router,
                trans,
                wb: Default::default(),
            },
            sender,
        )
    }

    pub(crate) fn run(&mut self) {
        while let Ok(Some(task)) = self.receiver.recv() {
            let len = self.receiver.len();
            let mut tasks = Vec::with_capacity(len + 1);
            tasks.push(task);
            for _ in 0..len {
                let task = self.receiver.recv().unwrap();
                if task.is_none() {
                    return;
                }
                tasks.push(task.unwrap());
            }
            for task in &mut tasks {
                if !task.raft_wb.is_empty() {
                    let wb = mem::take(&mut task.raft_wb);
                    self.wb.merge_write_batch(wb);
                }
            }
            self.handle_tasks(tasks);
        }
    }

    fn handle_tasks(&mut self, tasks: Vec<IoTask>) {
        let timer = tikv_util::time::Instant::now();
        if !self.wb.is_empty() {
            let wb = mem::take(&mut self.wb);
            let write_size = self.engine.persist(wb).unwrap();
            let write_raft_db_dur = timer.saturating_elapsed();
            if write_raft_db_dur > Duration::from_millis(50) {
                info!("io worker write raft db takes {:?}", write_raft_db_dur);
            }
            STORE_WRITE_RAFTDB_DURATION_HISTOGRAM.observe(duration_to_sec(write_raft_db_dur));
            STORE_WRITE_TRIGGER_SIZE_HISTOGRAM.observe(write_size as f64);
        }
        let timer = tikv_util::time::Instant::now();
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
        }
        if self.trans.need_flush() {
            self.trans.flush();
        }
        let send_time = duration_to_sec(timer.saturating_elapsed());
        STORE_WRITE_SEND_DURATION_HISTOGRAM.observe(send_time);
    }
}
