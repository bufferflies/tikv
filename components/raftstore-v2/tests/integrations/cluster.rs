// Copyright 2022 TiKV Project Authors. Licensed under Apache-2.0.

use core::panic;
use std::{
    ops::{Deref, DerefMut},
    path::Path,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Mutex,
    },
    thread,
    time::{Duration, Instant},
};

use collections::HashSet;
use concurrency_manager::ConcurrencyManager;
use crossbeam::channel::{self, Receiver, Sender, TrySendError};
use engine_test::{
    ctor::{CfOptions, DbOptions},
    kv::{KvTestEngine, TestTabletFactoryV2},
    raft::RaftTestEngine,
};
use engine_traits::{OpenOptions, TabletFactory, ALL_CFS};
use futures::executor::block_on;
use kvproto::{
    metapb::{self, RegionEpoch, Store},
    raft_cmdpb::{RaftCmdRequest, RaftCmdResponse},
    raft_serverpb::RaftMessage,
};
use raft::{eraftpb::MessageType, StateRole};
use raftstore::{
    coprocessor::{RegionChangeEvent, RoleChange},
    store::{
        region_meta::{RegionLocalState, RegionMeta},
        util::LockManagerNotifier,
        Config, TabletSnapKey, TabletSnapManager, Transport, RAFT_INIT_LOG_INDEX,
    },
};
use raftstore_v2::{
    create_store_batch_system,
    router::{DebugInfoChannel, FlushChannel, PeerMsg, PeerTick, QueryResult, RaftRouter},
    Bootstrap, StoreMeta, StoreSystem,
};
use slog::{debug, o, Logger};
use tempfile::TempDir;
use test_pd_client::TestPdClient;
use tikv_util::{
    config::{ReadableDuration, VersionTrack},
    store::new_peer,
};
use txn_types::TimeStamp;

#[derive(Clone)]
pub struct TestRouter(RaftRouter<KvTestEngine, RaftTestEngine>);

impl Deref for TestRouter {
    type Target = RaftRouter<KvTestEngine, RaftTestEngine>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for TestRouter {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl TestRouter {
    pub fn query(&self, region_id: u64, req: RaftCmdRequest) -> Option<QueryResult> {
        let (msg, sub) = PeerMsg::raft_query(req);
        self.send(region_id, msg).unwrap();
        block_on(sub.result())
    }

    pub fn must_query_debug_info(&self, region_id: u64, timeout: Duration) -> Option<RegionMeta> {
        let timer = Instant::now();
        while timer.elapsed() < timeout {
            let (ch, sub) = DebugInfoChannel::pair();
            let msg = PeerMsg::QueryDebugInfo(ch);
            let res = self.send(region_id, msg);
            if res.is_err() {
                thread::sleep(Duration::from_millis(10));
                continue;
            }
            return block_on(sub.result());
        }
        None
    }

    pub fn command(&self, region_id: u64, req: RaftCmdRequest) -> Option<RaftCmdResponse> {
        let (msg, sub) = PeerMsg::raft_command(req);
        self.send(region_id, msg).unwrap();
        block_on(sub.result())
    }

    pub fn wait_flush(&self, region_id: u64, timeout: Duration) -> bool {
        let timer = Instant::now();
        while timer.elapsed() < timeout {
            let (ch, sub) = FlushChannel::pair();
            let res = self.send(region_id, PeerMsg::WaitFlush(ch));
            match res {
                Ok(_) => return block_on(sub.result()).is_some(),
                Err(TrySendError::Disconnected(_)) => return false,
                Err(TrySendError::Full(_)) => thread::sleep(Duration::from_millis(10)),
            }
        }
        panic!("unable to flush {}", region_id);
    }

    pub fn wait_applied_to_current_term(&self, region_id: u64, timeout: Duration) {
        let mut now = Instant::now();
        let deadline = now + timeout;
        let mut res = None;
        while now < deadline {
            res = self.must_query_debug_info(region_id, deadline - now);
            if let Some(info) = &res {
                // If term matches and apply to commit index, then it must apply to current
                // term.
                if info.raft_apply.applied_index == info.raft_apply.commit_index
                    && info.raft_apply.commit_term == info.raft_status.hard_state.term
                {
                    return;
                }
            }
            thread::sleep(Duration::from_millis(10));
            now = Instant::now();
        }
        panic!(
            "region {} is not applied to current term, {:?}",
            region_id, res
        );
    }

    pub fn new_request_for(&self, region_id: u64) -> RaftCmdRequest {
        let meta = self
            .must_query_debug_info(region_id, Duration::from_secs(1))
            .unwrap();
        let mut req = RaftCmdRequest::default();
        req.mut_header().set_region_id(region_id);
        let epoch = req.mut_header().mut_region_epoch();
        let epoch_meta = &meta.region_state.epoch;
        epoch.set_version(epoch_meta.version);
        epoch.set_conf_ver(epoch_meta.conf_ver);
        let target_peer = *meta
            .region_state
            .peers
            .iter()
            .find(|p| p.id == meta.raft_status.id)
            .unwrap();
        let mut peer = new_peer(target_peer.store_id, target_peer.id);
        peer.role = target_peer.role.into();
        req.mut_header().set_peer(peer);
        req.mut_header().set_term(meta.raft_status.hard_state.term);
        req
    }

    pub fn region_detail(&self, region_id: u64) -> metapb::Region {
        let RegionLocalState {
            id,
            start_key,
            end_key,
            epoch,
            peers,
            ..
        } = self
            .must_query_debug_info(region_id, Duration::from_secs(1))
            .unwrap()
            .region_state;
        let mut region = metapb::Region::default();
        region.set_id(id);
        region.set_start_key(start_key);
        region.set_end_key(end_key);
        let mut region_epoch = RegionEpoch::default();
        region_epoch.set_conf_ver(epoch.conf_ver);
        region_epoch.set_version(epoch.version);
        region.set_region_epoch(region_epoch);
        for peer in peers {
            region.mut_peers().push(new_peer(peer.store_id, peer.id));
        }
        region
    }
}

pub struct RunningState {
    store_id: u64,
    pub raft_engine: RaftTestEngine,
    pub factory: Arc<TestTabletFactoryV2>,
    pub system: StoreSystem<KvTestEngine, RaftTestEngine>,
    pub cfg: Arc<VersionTrack<Config>>,
    pub transport: TestTransport,
    // We need this to clear the ref counts of CachedTablet when shutdown
    store_meta: Arc<Mutex<StoreMeta<KvTestEngine>>>,
    pub root_region_id: u64,
}

impl RunningState {
    fn new(
        pd_client: &Arc<TestPdClient>,
        path: &Path,
        cfg: Arc<VersionTrack<Config>>,
        transport: TestTransport,
        logger: &Logger,
    ) -> (TestRouter, TabletSnapManager, Self) {
        let cf_opts = ALL_CFS
            .iter()
            .copied()
            .map(|cf| (cf, CfOptions::default()))
            .collect();
        let factory = Arc::new(TestTabletFactoryV2::new(
            path,
            DbOptions::default(),
            cf_opts,
        ));
        let raft_engine =
            engine_test::raft::new_engine(&format!("{}", path.join("raft").display()), None)
                .unwrap();
        let mut bootstrap = Bootstrap::new(&raft_engine, 0, pd_client.as_ref(), logger.clone());
        let store_id = bootstrap.bootstrap_store().unwrap();
        let mut store = Store::default();
        store.set_id(store_id);
        let mut region_id = 2;
        if let Some(region) = bootstrap.bootstrap_first_region(&store, store_id).unwrap() {
            if factory.exists(region.get_id(), RAFT_INIT_LOG_INDEX) {
                factory
                    .destroy_tablet(region.get_id(), RAFT_INIT_LOG_INDEX)
                    .unwrap();
            }
            factory
                .open_tablet(
                    region.get_id(),
                    Some(RAFT_INIT_LOG_INDEX),
                    OpenOptions::default().set_create_new(true),
                )
                .unwrap();
            region_id = region.get_id();
        }

        let (router, mut system) =
            create_store_batch_system::<KvTestEngine, RaftTestEngine>(&cfg.value(), logger.clone());

        let router = RaftRouter::new(store_id, router);
        let store_meta = router.store_meta().clone();
        let snap_mgr = TabletSnapManager::new(path.join("tablets_snap").to_str().unwrap());
        snap_mgr.init().unwrap();
        system
            .start(
                store_id,
                cfg.clone(),
                raft_engine.clone(),
                factory.clone(),
                transport.clone(),
                pd_client.clone(),
                router.store_router(),
                store_meta.clone(),
                snap_mgr.clone(),
                ConcurrencyManager::new(TimeStamp::zero()), // todo
                None,
                Arc::new(DummyLockManagerObserver {}),
            )
            .unwrap();

        let state = Self {
            store_id,
            raft_engine,
            factory,
            system,
            cfg,
            transport,
            store_meta,
            root_region_id: region_id,
        };
        (TestRouter(router), snap_mgr, state)
    }

    pub fn peer_id(&self, region_id: u64) -> Option<u64> {
        let meta = self.store_meta.lock().unwrap();
        let kv = meta.readers.get_key_value(&region_id);
        if let Some(kv) = kv {
            return Some(kv.1.peer_id);
        }
        None
    }
}

impl Drop for RunningState {
    fn drop(&mut self) {
        self.system.shutdown();
    }
}

pub struct TestNode {
    pd_client: Arc<TestPdClient>,
    path: TempDir,
    running_state: Option<RunningState>,
    logger: Logger,
    snap_mgr: Option<TabletSnapManager>,
}

impl TestNode {
    fn with_pd(pd_client: Arc<TestPdClient>, logger: Logger) -> TestNode {
        let path = TempDir::new().unwrap();

        TestNode {
            pd_client,
            path,
            running_state: None,
            snap_mgr: None,
            logger,
        }
    }

    fn start(&mut self, cfg: Arc<VersionTrack<Config>>, trans: TestTransport) -> TestRouter {
        let (router, snap_mgr, state) =
            RunningState::new(&self.pd_client, self.path.path(), cfg, trans, &self.logger);
        self.running_state = Some(state);
        self.snap_mgr = Some(snap_mgr);
        router
    }

    pub fn tablet_factory(&self) -> &Arc<TestTabletFactoryV2> {
        &self.running_state().unwrap().factory
    }

    pub fn pd_client(&self) -> &Arc<TestPdClient> {
        &self.pd_client
    }

    fn stop(&mut self) {
        if let Some(state) = std::mem::take(&mut self.running_state) {
            let mut meta = state.store_meta.lock().unwrap();
            meta.tablet_caches.clear();
        }
    }

    fn restart(&mut self) -> TestRouter {
        let state = self.running_state().unwrap();
        let prev_transport = state.transport.clone();
        let cfg = state.cfg.clone();
        self.stop();
        self.start(cfg, prev_transport)
    }

    pub fn running_state(&self) -> Option<&RunningState> {
        self.running_state.as_ref()
    }

    pub fn snap_mgr(&self) -> Option<&TabletSnapManager> {
        self.snap_mgr.as_ref()
    }

    pub fn id(&self) -> u64 {
        self.running_state().unwrap().store_id
    }

    pub fn root_region_id(&self) -> u64 {
        self.running_state().unwrap().root_region_id
    }

    pub fn peer_id(&self, region_id: u64) -> Option<u64> {
        self.running_state().unwrap().peer_id(region_id)
    }
}

impl Drop for TestNode {
    fn drop(&mut self) {
        self.stop();
    }
}

#[derive(Clone)]
pub struct TestTransport {
    tx: Sender<RaftMessage>,
    flush_cnt: Arc<AtomicUsize>,
}

pub fn new_test_transport() -> (TestTransport, Receiver<RaftMessage>) {
    let (tx, rx) = channel::unbounded();
    let flush_cnt = Default::default();
    (TestTransport { tx, flush_cnt }, rx)
}

impl Transport for TestTransport {
    fn send(&mut self, msg: RaftMessage) -> raftstore_v2::Result<()> {
        let _ = self.tx.send(msg);
        Ok(())
    }

    fn set_store_allowlist(&mut self, _stores: Vec<u64>) {}

    fn need_flush(&self) -> bool {
        !self.tx.is_empty()
    }

    fn flush(&mut self) {
        self.flush_cnt.fetch_add(1, Ordering::SeqCst);
    }
}

// TODO: remove following when we finally integrate it in tikv-server binary.
pub fn v2_default_config() -> Config {
    let mut config = Config::default();
    config.store_io_pool_size = 1;
    config
}

/// Disable all ticks, so test case can schedule manually.
pub fn disable_all_auto_ticks(cfg: &mut Config) {
    cfg.raft_base_tick_interval = ReadableDuration::ZERO;
    cfg.raft_log_gc_tick_interval = ReadableDuration::ZERO;
    cfg.raft_log_compact_sync_interval = ReadableDuration::ZERO;
    cfg.raft_engine_purge_interval = ReadableDuration::ZERO;
    cfg.split_region_check_tick_interval = ReadableDuration::ZERO;
    cfg.region_compact_check_interval = ReadableDuration::ZERO;
    cfg.pd_heartbeat_tick_interval = ReadableDuration::ZERO;
    cfg.pd_store_heartbeat_tick_interval = ReadableDuration::ZERO;
    cfg.snap_mgr_gc_tick_interval = ReadableDuration::ZERO;
    cfg.lock_cf_compact_interval = ReadableDuration::ZERO;
    cfg.peer_stale_state_check_interval = ReadableDuration::ZERO;
    cfg.consistency_check_interval = ReadableDuration::ZERO;
    cfg.report_region_flow_interval = ReadableDuration::ZERO;
    cfg.check_leader_lease_interval = ReadableDuration::ZERO;
    cfg.merge_check_tick_interval = ReadableDuration::ZERO;
    cfg.cleanup_import_sst_interval = ReadableDuration::ZERO;
    cfg.inspect_interval = ReadableDuration::ZERO;
    cfg.report_min_resolved_ts_interval = ReadableDuration::ZERO;
    cfg.reactive_memory_lock_tick_interval = ReadableDuration::ZERO;
    cfg.report_region_buckets_tick_interval = ReadableDuration::ZERO;
    cfg.check_long_uncommitted_interval = ReadableDuration::ZERO;
}

pub struct Cluster {
    pd_client: Arc<TestPdClient>,
    nodes: Vec<TestNode>,
    receivers: Vec<Receiver<RaftMessage>>,
    routers: Vec<TestRouter>,
    logger: Logger,
}

impl Default for Cluster {
    fn default() -> Cluster {
        Cluster::with_node_count(1, None)
    }
}

impl Cluster {
    pub fn with_config(config: Config) -> Cluster {
        Cluster::with_node_count(1, Some(config))
    }

    pub fn with_node_count(count: usize, config: Option<Config>) -> Self {
        let logger = slog_global::borrow_global().new(o!());
        let mut cluster = Cluster {
            pd_client: Arc::new(TestPdClient::new(1, false)),
            nodes: vec![],
            receivers: vec![],
            routers: vec![],
            logger,
        };
        let mut cfg = if let Some(config) = config {
            config
        } else {
            v2_default_config()
        };
        disable_all_auto_ticks(&mut cfg);
        for _ in 1..=count {
            let mut node = TestNode::with_pd(cluster.pd_client.clone(), cluster.logger.clone());
            let (tx, rx) = new_test_transport();
            let router = node.start(Arc::new(VersionTrack::new(cfg.clone())), tx);
            cluster.nodes.push(node);
            cluster.receivers.push(rx);
            cluster.routers.push(router);
        }
        cluster
    }

    pub fn restart(&mut self, offset: usize) {
        let router = self.nodes[offset].restart();
        self.routers[offset] = router;
    }

    pub fn node(&self, offset: usize) -> &TestNode {
        &self.nodes[offset]
    }

    pub fn router(&self, offset: usize) -> TestRouter {
        self.routers[offset].clone()
    }

    pub fn root_region_id(&self) -> u64 {
        self.nodes[0].root_region_id()
    }

    pub fn trig_heartbeat(&self, node_offset: usize, region_id: u64) {
        for _i in 1..=self
            .node(node_offset)
            .running_state()
            .unwrap()
            .cfg
            .value()
            .raft_heartbeat_ticks
        {
            self.router(node_offset)
                .send(region_id, PeerMsg::Tick(PeerTick::Raft))
                .unwrap()
        }
    }

    pub fn trig_split_check(&self, node_offset: usize, region_id: u64) {
        for _i in 1..=self
            .node(node_offset)
            .running_state()
            .unwrap()
            .cfg
            .value()
            .raft_heartbeat_ticks
        {
            self.router(node_offset)
                .send(region_id, PeerMsg::Tick(PeerTick::SplitRegionCheck))
                .unwrap()
        }
    }

    /// Send messages and wait for side effects are all handled.
    #[allow(clippy::vec_box)]
    pub fn dispatch(&self, region_id: u64, mut msgs: Vec<Box<RaftMessage>>) {
        let mut regions = HashSet::default();
        regions.insert(region_id);
        loop {
            for msg in msgs.drain(..) {
                let offset = match self
                    .nodes
                    .iter()
                    .position(|n| n.id() == msg.get_to_peer().get_store_id())
                {
                    Some(offset) => offset,
                    None => {
                        debug!(self.logger, "failed to find node"; "message" => ?msg);
                        continue;
                    }
                };
                regions.insert(msg.get_region_id());
                // Simulate already received the snapshot.
                if msg.get_message().get_msg_type() == MessageType::MsgSnapshot {
                    let from_offset = match self
                        .nodes
                        .iter()
                        .position(|n| n.id() == msg.get_from_peer().get_store_id())
                    {
                        Some(offset) => offset,
                        None => {
                            debug!(self.logger, "failed to find snapshot source node"; "message" => ?msg);
                            continue;
                        }
                    };
                    let key = TabletSnapKey::new(
                        region_id,
                        msg.get_to_peer().get_id(),
                        msg.get_message().get_snapshot().get_metadata().get_term(),
                        msg.get_message().get_snapshot().get_metadata().get_index(),
                    );
                    let from_snap_mgr = self.node(from_offset).snap_mgr().unwrap();
                    let to_snap_mgr = self.node(offset).snap_mgr().unwrap();
                    let gen_path = from_snap_mgr.tablet_gen_path(&key);
                    let recv_path = to_snap_mgr.final_recv_path(&key);
                    assert!(gen_path.exists());
                    println!(
                        "snapshot send, send_path:{},recv_path:{}",
                        gen_path.display(),
                        recv_path.display()
                    );
                    std::fs::rename(gen_path, recv_path.clone()).unwrap();
                    assert!(recv_path.exists());
                }
                if let Err(e) = self.routers[offset].send_raft_message(msg) {
                    debug!(self.logger, "failed to send raft message"; "err" => ?e);
                }
            }
            for (router, rx) in self.routers.iter().zip(&self.receivers) {
                for region_id in &regions {
                    router.wait_flush(*region_id, Duration::from_secs(3));
                }
                while let Ok(msg) = rx.try_recv() {
                    msgs.push(Box::new(msg));
                }
            }
            regions.clear();
            if msgs.is_empty() {
                return;
            }
        }
    }
}

struct DummyLockManagerObserver {}

impl LockManagerNotifier for DummyLockManagerObserver {
    fn on_region_changed(&self, _: &metapb::Region, _: RegionChangeEvent, _: StateRole) {}

    fn on_role_change(&self, _: &metapb::Region, _: RoleChange) {}
}
