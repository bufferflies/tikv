// Copyright 2020 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    collections::HashMap,
    convert::identity,
    sync::*,
    time::{Duration, Instant},
};

use causal_ts::CausalTsProvider;
use cdc::{recv_timeout, CdcObserver, Delegate, Task, Validate};
use cloud_server::CdcCreationContext;
use concurrency_manager::ConcurrencyManager;
use engine_rocks::RocksEngine;
use futures::executor::block_on;
use grpcio::{
    CallOption, ChannelBuilder, ClientDuplexReceiver, ClientDuplexSender, ClientUnaryReceiver,
    Environment, MetadataBuilder,
};
use kvengine::WRITE_CF;
use kvproto::{
    cdcpb::{ChangeDataClient, ChangeDataEvent, ChangeDataRequest},
    kvrpcpb::{PrewriteRequestPessimisticAction::*, *},
    metapb::RegionEpoch,
    tikvpb::TikvClient,
};
use online_config::OnlineConfig;
use pd_client::PdClient;
use test_cloud_server::{alloc_node_id_vec, ServerClusterExt};
use test_pd_client::{PdClientExt, TestPdClient};
use test_raftstore::*;
use test_rfstore::{spawn_recv_filter, SharedFilters};
use tikv::config::{CdcConfig, TikvConfig};
use tikv_util::{
    config::ReadableDuration,
    debug, warn,
    worker::{Runnable, Scheduler},
    Either, HandyRwLock,
};
use txn_types::TimeStamp;
static INIT: Once = Once::new();

pub fn init() {
    INIT.call_once(test_util::setup_for_ci);
}

#[derive(Clone)]
pub struct ClientReceiver {
    receiver: Arc<Mutex<Option<ClientDuplexReceiver<ChangeDataEvent>>>>,
}

impl ClientReceiver {
    pub fn replace(
        &self,
        rx: Option<ClientDuplexReceiver<ChangeDataEvent>>,
    ) -> Option<ClientDuplexReceiver<ChangeDataEvent>> {
        std::mem::replace(&mut *self.receiver.lock().unwrap(), rx)
    }
}
#[allow(clippy::type_complexity)]
pub fn new_event_feed(
    client: &ChangeDataClient,
) -> (
    ClientDuplexSender<ChangeDataRequest>,
    ClientReceiver,
    Box<dyn Fn(bool) -> ChangeDataEvent + Send>,
) {
    create_event_feed(client, false)
}

#[allow(clippy::type_complexity)]
pub fn new_event_feed_v2(
    client: &ChangeDataClient,
) -> (
    ClientDuplexSender<ChangeDataRequest>,
    ClientReceiver,
    Box<dyn Fn(bool) -> ChangeDataEvent + Send>,
) {
    create_event_feed(client, true)
}

#[allow(clippy::type_complexity)]
fn create_event_feed(
    client: &ChangeDataClient,
    stream_multiplexing: bool,
) -> (
    ClientDuplexSender<ChangeDataRequest>,
    ClientReceiver,
    Box<dyn Fn(bool) -> ChangeDataEvent + Send>,
) {
    let (req_tx, resp_rx) = if stream_multiplexing {
        let mut metadata = MetadataBuilder::with_capacity(1);
        metadata.add_str("features", "stream-multiplexing").unwrap();
        let opt = CallOption::default().headers(metadata.build());
        client.event_feed_v2_opt(opt).unwrap()
    } else {
        client.event_feed().unwrap()
    };
    let event_feed_wrap = Arc::new(Mutex::new(Some(resp_rx)));
    let event_feed_wrap_clone = event_feed_wrap.clone();

    let receive_event = move |keep_resolved_ts: bool| loop {
        let mut events;
        {
            let mut event_feed = event_feed_wrap_clone.lock().unwrap();
            events = event_feed.take();
        }
        let mut events_rx = if let Some(events_rx) = events.as_mut() {
            events_rx
        } else {
            return ChangeDataEvent::default();
        };
        let change_data =
            if let Some(event) = recv_timeout(&mut events_rx, Duration::from_secs(5)).unwrap() {
                event
            } else {
                return ChangeDataEvent::default();
            };
        {
            let mut event_feed = event_feed_wrap_clone.lock().unwrap();
            *event_feed = events;
        }
        let change_data_event = change_data.unwrap_or_default();
        if !keep_resolved_ts && change_data_event.has_resolved_ts() {
            continue;
        }
        tikv_util::info!("cdc receive event {:?}", change_data_event);
        break change_data_event;
    };
    (
        req_tx,
        ClientReceiver {
            receiver: event_feed_wrap,
        },
        Box::new(receive_event),
    )
}

pub struct CloudTestSuiteBuilder {
    cfg_fun: Box<dyn Fn(u16, &mut TikvConfig)>,
    num_nodes: usize,
    after_cluster_bootstraped: Vec<Box<dyn Fn(&mut ServerClusterExt)>>,
}

impl Default for CloudTestSuiteBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl CloudTestSuiteBuilder {
    pub fn new() -> CloudTestSuiteBuilder {
        CloudTestSuiteBuilder {
            cfg_fun: Box::new(|_id, cfg| {
                cfg.storage.check_backup_ts = false;
            }),
            num_nodes: 1,
            after_cluster_bootstraped: vec![],
        }
    }

    #[must_use]
    pub fn cfg_fun<F>(mut self, f: F) -> CloudTestSuiteBuilder
    where
        F: Fn(u16, &mut TikvConfig) + 'static,
    {
        self.cfg_fun = Box::new(f);
        self
    }

    #[must_use]
    pub fn num_nodes(mut self, num: usize) -> CloudTestSuiteBuilder {
        self.num_nodes = num;
        self
    }

    #[must_use]
    pub fn presplit_magic_keyspace(mut self) -> CloudTestSuiteBuilder {
        const MAGIC_KEYSPACE_IDS: [u32; 1] = [u32::from_be_bytes(*b"\x00key")];
        self.after_cluster_bootstraped.push(Box::new(|cluster| {
            let mut cli_for_init = cluster.cluster.new_client();
            for magic in MAGIC_KEYSPACE_IDS {
                cli_for_init.split_keyspace(magic);
            }
        }));
        self
    }

    #[must_use]
    pub fn after_cluster_bootstrapped<F>(mut self, f: F) -> CloudTestSuiteBuilder
    where
        F: Fn(&mut ServerClusterExt) + 'static,
    {
        self.after_cluster_bootstraped.push(Box::new(f));
        self
    }

    pub fn build(self) -> TestSuite {
        let ids = alloc_node_id_vec(self.num_nodes);
        let (tx, rx) = std::sync::mpsc::channel();
        let mut svc_ext = test_cloud_server::ServerClusterBuilder::new(ids, self.cfg_fun)
            .before_run_server(move |i, svc| {
                let cm = svc.get_concurrency_manager().clone();
                let tx = tx.clone();
                let send_filters = Arc::default();
                let send_filters2 = Arc::clone(&send_filters);
                let recv_filters = Arc::default();
                let recv_filters2 = Arc::clone(&recv_filters);

                svc.mut_system()
                    .replace_peer_receiver(|rec| spawn_recv_filter(rec, recv_filters2));
                svc.test_export.on_cdc_creation =
                    Some(Box::new(move |ctx: CdcCreationContext<'_>| {
                        let scheduler = ctx.scheduler.clone();
                        let res = tx.send((
                            i,
                            scheduler,
                            cm,
                            ctx.observer.clone(),
                            send_filters,
                            recv_filters,
                        ));
                        if res.is_err() {
                            // TODO? (if needed): save `tx` and provide a method to manually refresh the new started nodes.
                            warn!("!!!!!! now the test framework losts all extra tracking to a node after its restart.\
                                packet filters, endpoints references, etc. won't work for this node then."; 
                                "node_id" => i);
                        }
                        let mut updated_cfg = CdcConfig::default();
                        updated_cfg.min_ts_interval = ReadableDuration::millis(100);
                        ctx.endpoint
                            .run(Task::ChangeConfig(CdcConfig::default().diff(&updated_cfg)));
                        ctx.endpoint.set_max_scan_batch_size(2);
                    }));
                svc.test_export.override_transport = Some(Box::new(move |transport| {
                    Box::new(test_rfstore::SimulateTransport::with_filters(
                        transport,
                        send_filters2,
                    ))
                }));
            })
            .build_ext();

        let mut endpoints = HashMap::default();
        let mut obs = HashMap::default();
        let mut concurrency_managers = HashMap::default();
        let mut send_filters = HashMap::default();
        let mut recv_filters = HashMap::default();
        for _ in 0..self.num_nodes {
            let (id, sched, cm, ob, sf, rf) = rx.recv().unwrap();
            let store_id = svc_ext.cluster.get_store_id(id);
            endpoints.insert(store_id, sched);
            obs.insert(store_id, ob);
            concurrency_managers.insert(store_id, cm);
            send_filters.insert(store_id, sf);
            recv_filters.insert(store_id, rf);
        }

        for after_bootstrap_hook in self.after_cluster_bootstraped {
            after_bootstrap_hook(&mut svc_ext);
        }

        let pd_client = svc_ext.get_pd_client();
        test_util::init_log_for_test();

        let mut cse_store_id_to_node_id = HashMap::default();
        for node_id in svc_ext.get_nodes() {
            let store_id = svc_ext.get_store_id(node_id);
            cse_store_id_to_node_id.insert(store_id, node_id);
        }

        TestSuite {
            cluster: TestCluster::CloudEngine {
                server: svc_ext,
                send_filters,
                recv_filters,
                store_id_to_node_id: cse_store_id_to_node_id,
            },
            endpoints,
            obs,
            tikv_cli: Default::default(),
            cdc_cli: Default::default(),
            concurrency_managers,
            pd_client,
            env: Arc::new(Environment::new(1)),
        }
    }
}

pub struct TestSuiteBuilder {
    cluster: Option<Cluster<ServerCluster>>,
    memory_quota: Option<usize>,
}

impl TestSuiteBuilder {
    pub fn new() -> TestSuiteBuilder {
        TestSuiteBuilder {
            cluster: None,
            memory_quota: None,
        }
    }

    #[must_use]
    pub fn cluster(mut self, cluster: Cluster<ServerCluster>) -> TestSuiteBuilder {
        self.cluster = Some(cluster);
        self
    }

    #[must_use]
    pub fn memory_quota(mut self, memory_quota: usize) -> TestSuiteBuilder {
        self.memory_quota = Some(memory_quota);
        self
    }

    #[cfg(NGDISABLE)]
    pub fn build(self) -> TestSuite {
        self.build_with_cluster_runner(|cluster| cluster.run())
    }

    #[cfg(NGDISABLE)]
    pub fn build_with_cluster_runner<F>(self, mut runner: F) -> TestSuite
    where
        F: FnMut(&mut Cluster<ServerCluster>),
    {
        use raftstore::coprocessor::CoprocessorHost;
        use rfstore::CdcRaftRouter;

        init();
        let memory_quota = self.memory_quota.unwrap_or(usize::MAX);
        let mut cluster = self.cluster.unwrap();
        let count = cluster.count;
        let pd_cli = cluster.pd_client.clone();
        let mut endpoints = HashMap::default();
        let mut quotas = HashMap::default();
        let mut obs = HashMap::default();
        let mut concurrency_managers = HashMap::default();
        // Hack! node id are generated from 1..count+1.
        for id in 1..=count as u64 {
            // Create and run cdc endpoints.
            let worker = LazyWorker::new(format!("cdc-{}", id));
            let mut sim = cluster.sim.wl();

            // Register cdc service to gRPC server.
            let memory_quota = Arc::new(MemoryQuota::new(memory_quota));
            let memory_quota_ = memory_quota.clone();
            let scheduler = worker.scheduler();
            let pool = Arc::new(Builder::new("cdc-watchdog-test").thread_count(1).create());
            sim.pending_services
                .entry(id)
                .or_default()
                .push(Box::new(move || {
                    create_change_data(cdc::Service::new(
                        scheduler.clone(),
                        memory_quota_.clone(),
                        pool.clone(),
                    ))
                }));
            sim.txn_extra_schedulers.insert(
                id,
                Arc::new(cdc::CdcTxnExtraScheduler::new(
                    worker.scheduler().clone(),
                    memory_quota.clone(),
                )),
            );
            let scheduler = worker.scheduler();
            let cdc_ob = cdc::CdcObserver::new(scheduler.clone(), memory_quota.clone());
            obs.insert(id, cdc_ob.clone());
            sim.coprocessor_hooks.entry(id).or_default().push(Box::new(
                move |host: &mut CoprocessorHost<kvengine::Engine>| {
                    panic!("cdc_ob.register_to(host)");
                },
            ));
            endpoints.insert(id, worker);
            quotas.insert(id, memory_quota);
        }

        runner(&mut cluster);
        for (id, worker) in &mut endpoints {
            let sim = cluster.sim.wl();
            let raft_router = sim.get_server_router(*id);
            let cdc_ob = obs.get(id).unwrap().clone();
            let cm = sim.get_concurrency_manager(*id);
            let env = Arc::new(Environment::new(1));
            let cfg = CdcConfig::default();
            let mut cdc_endpoint = cdc::Endpoint::new(
                DEFAULT_CLUSTER_ID,
                &cfg,
                &ResolvedTsConfig::default(),
                cluster.cfg.storage.api_version(),
                pd_cli.clone(),
                worker.scheduler(),
                CdcRaftRouter(raft_router),
                cdc_ob,
                cluster.store_metas[id].clone(),
                cm.clone(),
                env,
                sim.security_mgr.clone(),
                quotas[id].clone(),
                sim.get_causal_ts_provider(*id),
            );
            let mut updated_cfg = cfg.clone();
            updated_cfg.min_ts_interval = ReadableDuration::millis(100);
            cdc_endpoint.run(Task::ChangeConfig(cfg.diff(&updated_cfg)));
            cdc_endpoint.set_max_scan_batch_size(2);
            concurrency_managers.insert(*id, cm);
            worker.start(cdc_endpoint);
        }

        TestSuite {
            cluster: Tikv(cluster),
            endpoints,
            obs,
            concurrency_managers,
            env: Arc::new(Environment::new(1)),
            tikv_cli: HashMap::default(),
            cdc_cli: HashMap::default(),
        }
    }
}

pub enum TestCluster {
    Tikv(Cluster<ServerCluster>),
    CloudEngine {
        server: test_cloud_server::ServerClusterExt,
        send_filters: HashMap<u64, SharedFilters>,
        recv_filters: HashMap<u64, SharedFilters>,
        store_id_to_node_id: HashMap<u64, u16>,
    },
}

pub fn must_get_kvengine(
    e: &kvengine::Engine,
    region_id: u64,
    cf: usize,
    key: &[u8],
    value: Option<&[u8]>,
) {
    let snap = e.get_snap_access(region_id).unwrap();
    for _ in 1..300 {
        let res = snap.get(cf, key, u64::MAX);
        if value.is_none() && res.is_value_empty() {
            return;
        }
        if !res.is_value_empty() {
            if let (Some(value), res) = (value, res.get_value()) {
                assert_eq!(value, res);
                return;
            }
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    debug!("last try to get {}", log_wrappers::hex_encode_upper(key));
    let res = snap.get(cf, key, u64::MAX);
    if value.is_none() && res.is_value_empty()
        || value.is_some() && !res.is_value_empty() && value.unwrap() == res.get_value()
    {
        return;
    }
    panic!(
        "can't get value {:?} for key {}",
        value,
        log_wrappers::hex_encode_upper(key)
    )
}

pub fn must_get_equal_kvengine(
    engine: &kvengine::Engine,
    region_id: u64,
    key: &[u8],
    value: &[u8],
) {
    must_get_kvengine(engine, region_id, WRITE_CF, key, Some(value));
}

impl TestCluster {
    pub fn pd_client(&self) -> Arc<TestPdClient> {
        match self {
            TestCluster::Tikv(c) => c.pd_client.clone(),
            TestCluster::CloudEngine { server: c, .. } => c.cluster.get_pd_client(),
        }
    }

    pub fn get_region_epoch(&self, region_id: u64) -> RegionEpoch {
        block_on(self.pd_client().get_region_by_id(region_id))
            .unwrap()
            .unwrap()
            .take_region_epoch()
    }

    pub fn get_engine(&self, store_id: u64) -> Either<RocksEngine, kvengine::Engine> {
        match self {
            TestCluster::Tikv(c) => Either::Left(c.get_engine(store_id)),
            TestCluster::CloudEngine {
                server: c,
                store_id_to_node_id,
                ..
            } => {
                let node_id = store_id_to_node_id[&store_id];
                Either::Right(c.get_kvengine(node_id))
            }
        }
    }

    pub fn stop_node(&mut self, store_id: u64) {
        match self {
            TestCluster::Tikv(c) => {
                c.stop_node(store_id);
            }
            TestCluster::CloudEngine {
                server: c,
                store_id_to_node_id,
                ..
            } => {
                let node_id = store_id_to_node_id[&store_id];
                c.stop_node(node_id);
            }
        }
    }

    pub fn run_node(&mut self, store_id: u64) {
        match self {
            TestCluster::Tikv(c) => {
                c.run_node(store_id).unwrap();
            }
            TestCluster::CloudEngine {
                server: c,
                store_id_to_node_id,
                ..
            } => {
                let node_id = store_id_to_node_id[&store_id];
                c.start_node(node_id, |_, _| {});
            }
        }
    }

    pub fn try_merge(&mut self, source: u64, target: u64) {
        match self {
            TestCluster::Tikv(c) => {
                c.try_merge(source, target);
            }
            TestCluster::CloudEngine { server: c, .. } => {
                let pd = c.get_pd_client();
                pd.merge_region(source, target);
            }
        }
    }

    pub fn must_merge(&mut self, source: u64, target: u64) {
        match self {
            TestCluster::Tikv(c) => {
                c.must_try_merge(source, target);
            }
            TestCluster::CloudEngine { server: c, .. } => {
                let pd = c.get_pd_client();
                pd.merge_region(source, target);
                for _ in 0..30 {
                    let region = block_on(pd.get_region_by_id(source)).unwrap();
                    if region.is_none() {
                        return;
                    }
                    sleep_ms(100);
                }
                let region = block_on(pd.get_region_by_id(source)).unwrap();
                panic!("taking too long to merge {source} to {target}; source = {region:?}")
            }
        }
    }

    pub fn must_put(&mut self, key: &[u8], value: &[u8]) {
        match self {
            TestCluster::Tikv(c) => c.must_put(key, value),
            TestCluster::CloudEngine { server: c, .. } => {
                let mut cli = c.cluster.new_client();
                cli.put_kv(0..1, |_| key.to_vec(), |_| value.to_vec());
            }
        }
    }

    pub fn leader_of_region(&mut self, region_id: u64) -> Option<kvproto::metapb::Peer> {
        match self {
            TestCluster::Tikv(c) => c.leader_of_region(region_id),
            TestCluster::CloudEngine { server: c, .. } => {
                block_on(c.cluster.get_pd_client().get_region_leader_by_id(region_id))
                    .unwrap()
                    .map(|v| v.1)
            }
        }
    }

    pub fn must_transfer_leader(&mut self, region_id: u64, leader: kvproto::metapb::Peer) {
        match self {
            TestCluster::Tikv(c) => c.must_transfer_leader(region_id, leader),
            TestCluster::CloudEngine { server: c, .. } => {
                let timer = Instant::now();
                let pdc = c.get_pd_client();
                loop {
                    let cur_leader = block_on(pdc.get_region_leader_by_id(region_id)).unwrap();

                    if let Some((ref _region, ref cur_leader)) = cur_leader {
                        if cur_leader.get_id() == leader.get_id()
                            && cur_leader.get_store_id() == leader.get_store_id()
                        {
                            return;
                        }
                        pdc.transfer_leader(region_id, leader.clone(), vec![]);
                        sleep_ms(500);
                    }
                    if timer.elapsed() > Duration::from_secs(5) {
                        panic!(
                            "failed to transfer leader to [{}] {:?}, (region, current leader): {:?}",
                            region_id, leader, cur_leader
                        );
                    }
                }
            }
        }
    }

    pub fn api_version(&self) -> ApiVersion {
        match self {
            TestCluster::Tikv(c) => c.cfg.storage.api_version(),
            TestCluster::CloudEngine { server: _c, .. } => ApiVersion::V2,
        }
    }

    pub fn get_addr(&self, store_id: u64) -> String {
        match self {
            TestCluster::Tikv(c) => c.sim.rl().get_addr(store_id),
            TestCluster::CloudEngine { server: c, .. } => {
                let store = c.get_pd_client().get_store(store_id).unwrap();
                store.address
            }
        }
    }

    pub async fn flush_causal_ts(&self, store_id: u64) {
        let c = match self {
            TestCluster::Tikv(c) => c.sim.rl().get_causal_ts_provider(store_id),
            TestCluster::CloudEngine { server: c, .. } => {
                c.cluster.get_causal_ts_provider(store_id).cloned()
            }
        };
        c.unwrap().async_flush().await.unwrap();
    }

    pub fn shutdown(&mut self) {
        match self {
            TestCluster::Tikv(c) => c.shutdown(),
            TestCluster::CloudEngine { server: c, .. } => c.stop(),
        }
    }

    pub fn get_region(&self, key: &[u8]) -> kvproto::metapb::Region {
        match self {
            TestCluster::Tikv(c) => c.get_region(key),
            TestCluster::CloudEngine { server: c, .. } => {
                for _ in 0..100 {
                    if let Ok(region) = c.cluster.get_pd_client().get_region(key) {
                        return region;
                    }
                    sleep_ms(20);
                }
                panic!("find no region for {}", log_wrappers::hex_encode_upper(key));
            }
        }
    }

    pub fn must_split(&mut self, region: &kvproto::metapb::Region, key: &[u8]) {
        match self {
            TestCluster::Tikv(c) => c.must_split(region, key),
            TestCluster::CloudEngine { server: c, .. } => {
                let mut cli = c.cluster.new_client();
                cli.try_split(key, 10).unwrap()
            }
        }
    }

    pub fn add_cloud_recv_filter(&mut self, node_id: u64, filter: Box<dyn test_rfstore::Filter>) {
        match self {
            // The `Filter` type doesn't acutally match due to the `Result` type they use...
            TestCluster::Tikv(_c) => panic!("add_cloud_recv_filter not supported for TiKV mode"),
            TestCluster::CloudEngine { recv_filters, .. } => {
                if let Some(filters) = recv_filters.get(&node_id) {
                    filters.wl().push(filter);
                } else {
                    panic!("node {} not found", node_id);
                }
            }
        }
    }

    pub fn clear_recv_filters(&mut self, node_id: u64) {
        match self {
            TestCluster::Tikv(c) => c.sim.wl().clear_recv_filters(node_id),
            TestCluster::CloudEngine { recv_filters, .. } => {
                if let Some(filters) = recv_filters.get(&node_id) {
                    filters.wl().clear();
                } else {
                    panic!("node {} not found", node_id);
                }
            }
        }
    }

    pub fn add_cloud_send_filter(&mut self, node_id: u64, filter: Box<dyn test_rfstore::Filter>) {
        match self {
            // The `Filter` type doesn't acutally match due to the `Result` type they use...
            TestCluster::Tikv(_c) => panic!("add_cloud_send_filter not supported for TiKV mode"),
            TestCluster::CloudEngine { send_filters, .. } => {
                if let Some(filters) = send_filters.get(&node_id) {
                    filters.wl().push(filter);
                } else {
                    panic!("node {} not found", node_id);
                }
            }
        }
    }

    pub fn clear_send_filters(&mut self, node_id: u64) {
        match self {
            TestCluster::Tikv(c) => c.sim.wl().clear_send_filters(node_id),
            TestCluster::CloudEngine { send_filters, .. } => {
                if let Some(filters) = send_filters.get(&node_id) {
                    filters.wl().clear();
                } else {
                    panic!("node {} not found", node_id);
                }
            }
        }
    }
}

pub struct TestSuite {
    pub cluster: TestCluster,
    pub endpoints: HashMap<u64, Scheduler<Task>>,
    pub obs: HashMap<u64, CdcObserver>,
    pub pd_client: Arc<TestPdClient>,
    tikv_cli: HashMap<u64, TikvClient>,
    cdc_cli: HashMap<u64, ChangeDataClient>,
    concurrency_managers: HashMap<u64, ConcurrencyManager>,

    env: Arc<Environment>,
}

impl Default for TestSuiteBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl TestSuite {
    pub fn new(count: usize, api_version: ApiVersion) -> TestSuite {
        assert_eq!(api_version, ApiVersion::V2);
        Self::with_extra_builder_fun(count, identity)
    }

    pub fn cfg_for_cdc(cfg: &mut TikvConfig) {
        // Increase the Raft tick interval to make this test case running reliably.
        configure_for_lease_read(cfg, Some(100), None);
        // Disable background renew to make timestamp predictable.
        configure_for_causal_ts(cfg, "99999999s", 1);
        cfg.cdc.hibernate_regions_compatible = false;
        cfg.storage.check_backup_ts = false;
    }

    pub fn with_extra_builder_fun(
        count: usize,
        ext: impl FnOnce(CloudTestSuiteBuilder) -> CloudTestSuiteBuilder,
    ) -> TestSuite {
        ext(CloudTestSuiteBuilder::new()
            .presplit_magic_keyspace()
            .num_nodes(count)
            .cfg_fun(|_id, cfg| {
                Self::cfg_for_cdc(cfg);
            }))
        .build()
    }

    pub fn stop(mut self) {
        #[cfg(NGDISABLE)]
        for (_, worker) in self.endpoints.drain() {
            worker.stop_worker();
        }
        self.cluster.shutdown();
    }

    pub fn new_changedata_request(&mut self, region_id: u64) -> ChangeDataRequest {
        let mut req = ChangeDataRequest {
            region_id,
            ..Default::default()
        };
        req.mut_header()
            .set_cluster_id(self.cluster.pd_client().get_cluster_id().unwrap());
        req.set_region_epoch(self.get_context(region_id).take_region_epoch());
        req
    }

    pub fn must_kv_prewrite(
        &mut self,
        region_id: u64,
        muts: Vec<Mutation>,
        pk: Vec<u8>,
        ts: TimeStamp,
    ) {
        self.must_kv_prewrite_with_source(region_id, muts, pk, ts, 0);
    }

    pub fn must_kv_prewrite_with_source(
        &mut self,
        region_id: u64,
        muts: Vec<Mutation>,
        pk: Vec<u8>,
        ts: TimeStamp,
        txn_source: u64,
    ) {
        let mut prewrite_req = PrewriteRequest::default();
        let mut context = self.get_context(region_id);
        context.set_txn_source(txn_source);
        prewrite_req.set_context(context);
        prewrite_req.set_mutations(muts.into_iter().collect());
        prewrite_req.primary_lock = pk;
        prewrite_req.start_version = ts.into_inner();
        prewrite_req.lock_ttl = prewrite_req.start_version + 1;
        let prewrite_resp = self
            .get_tikv_client(region_id)
            .kv_prewrite(&prewrite_req)
            .unwrap();
        assert!(
            !prewrite_resp.has_region_error(),
            "{:?}",
            prewrite_resp.get_region_error()
        );
        assert!(
            prewrite_resp.errors.is_empty(),
            "{:?}",
            prewrite_resp.get_errors()
        );
    }

    pub fn must_kv_flush(
        &mut self,
        region_id: u64,
        muts: Vec<Mutation>,
        pk: Vec<u8>,
        ts: TimeStamp,
        generation: u64,
    ) {
        self.must_kv_flush_with_source(region_id, muts, pk, ts, generation, 0);
    }

    pub fn must_kv_flush_with_source(
        &mut self,
        region_id: u64,
        muts: Vec<Mutation>,
        pk: Vec<u8>,
        ts: TimeStamp,
        generation: u64,
        txn_source: u64,
    ) {
        let mut flush_req = FlushRequest::default();
        let mut context = self.get_context(region_id);
        context.set_txn_source(txn_source);
        flush_req.set_context(context);
        flush_req.set_mutations(muts.into_iter().collect());
        flush_req.primary_key = pk;
        flush_req.start_ts = ts.into_inner();
        flush_req.generation = generation;
        flush_req.lock_ttl = flush_req.start_ts + 1;
        let flush_resp = self
            .get_tikv_client(region_id)
            .kv_flush(&flush_req)
            .unwrap();
        assert!(
            !flush_resp.has_region_error(),
            "{:?}",
            flush_resp.get_region_error()
        );
        assert!(
            flush_resp.errors.is_empty(),
            "{:?}",
            flush_resp.get_errors()
        );
    }

    pub fn must_kv_put(&mut self, region_id: u64, key: Vec<u8>, value: Vec<u8>) {
        let mut rawkv_req = RawPutRequest::default();
        rawkv_req.set_context(self.get_context(region_id));
        rawkv_req.set_key(key);
        rawkv_req.set_value(value);
        rawkv_req.set_ttl(u64::MAX);

        let rawkv_resp = self.get_tikv_client(region_id).raw_put(&rawkv_req).unwrap();
        assert!(
            !rawkv_resp.has_region_error(),
            "{:?}",
            rawkv_resp.get_region_error()
        );
        assert!(rawkv_resp.error.is_empty(), "{:?}", rawkv_resp.get_error());
    }

    pub fn must_kv_commit(
        &mut self,
        region_id: u64,
        keys: Vec<Vec<u8>>,
        start_ts: TimeStamp,
        commit_ts: TimeStamp,
    ) {
        self.must_kv_commit_with_source(region_id, keys, start_ts, commit_ts, 0);
    }

    pub fn must_kv_commit_with_source(
        &mut self,
        region_id: u64,
        keys: Vec<Vec<u8>>,
        start_ts: TimeStamp,
        commit_ts: TimeStamp,
        txn_source: u64,
    ) {
        let mut commit_req = CommitRequest::default();
        let mut context = self.get_context(region_id);
        context.set_txn_source(txn_source);
        commit_req.set_context(context);
        commit_req.start_version = start_ts.into_inner();
        commit_req.set_keys(keys.into_iter().collect());
        commit_req.commit_version = commit_ts.into_inner();
        let commit_resp = self
            .get_tikv_client(region_id)
            .kv_commit(&commit_req)
            .unwrap();
        assert!(
            !commit_resp.has_region_error(),
            "{:?}",
            commit_resp.get_region_error()
        );
        assert!(!commit_resp.has_error(), "{:?}", commit_resp.get_error());
    }

    pub fn must_kv_rollback(&mut self, region_id: u64, keys: Vec<Vec<u8>>, start_ts: TimeStamp) {
        let mut rollback_req = BatchRollbackRequest::default();
        rollback_req.set_context(self.get_context(region_id));
        rollback_req.start_version = start_ts.into_inner();
        rollback_req.set_keys(keys.into_iter().collect());
        let rollback_resp = self
            .get_tikv_client(region_id)
            .kv_batch_rollback(&rollback_req)
            .unwrap();
        assert!(
            !rollback_resp.has_region_error(),
            "{:?}",
            rollback_resp.get_region_error()
        );
        assert!(
            !rollback_resp.has_error(),
            "{:?}",
            rollback_resp.get_error()
        );
    }

    pub fn must_check_txn_status(
        &mut self,
        region_id: u64,
        primary_key: Vec<u8>,
        lock_ts: TimeStamp,
        caller_start_ts: TimeStamp,
        current_ts: TimeStamp,
        rollback_if_not_exist: bool,
    ) -> Action {
        let mut req = CheckTxnStatusRequest::default();
        req.set_context(self.get_context(region_id));
        req.set_primary_key(primary_key);
        req.set_lock_ts(lock_ts.into_inner());
        req.set_caller_start_ts(caller_start_ts.into_inner());
        req.set_current_ts(current_ts.into_inner());
        req.set_rollback_if_not_exist(rollback_if_not_exist);
        let resp = self
            .get_tikv_client(region_id)
            .kv_check_txn_status(&req)
            .unwrap();
        assert!(!resp.has_region_error(), "{:?}", resp.get_region_error());
        assert!(!resp.has_error(), "{:?}", resp.get_error());
        resp.get_action()
    }

    pub fn must_acquire_pessimistic_lock(
        &mut self,
        region_id: u64,
        muts: Vec<Mutation>,
        pk: Vec<u8>,
        start_ts: TimeStamp,
        for_update_ts: TimeStamp,
    ) {
        let mut lock_req = PessimisticLockRequest::default();
        lock_req.set_context(self.get_context(region_id));
        lock_req.set_mutations(muts.into_iter().collect());
        lock_req.start_version = start_ts.into_inner();
        lock_req.for_update_ts = for_update_ts.into_inner();
        lock_req.primary_lock = pk;
        let lock_resp = self
            .get_tikv_client(region_id)
            .kv_pessimistic_lock(&lock_req)
            .unwrap();
        assert!(
            !lock_resp.has_region_error(),
            "{:?}",
            lock_resp.get_region_error()
        );
        assert!(
            lock_resp.get_errors().is_empty(),
            "{:?}",
            lock_resp.get_errors()
        );
    }

    pub fn must_release_pessimistic_lock(
        &mut self,
        region_id: u64,
        pk: Vec<u8>,
        start_ts: TimeStamp,
        for_update_ts: TimeStamp,
    ) {
        let mut req = PessimisticRollbackRequest::default();
        req.set_context(self.get_context(region_id));
        req.start_version = start_ts.into_inner();
        req.for_update_ts = for_update_ts.into_inner();
        req.set_keys(vec![pk].into_iter().collect());
        let resp = self
            .get_tikv_client(region_id)
            .kv_pessimistic_rollback(&req)
            .unwrap();
        assert!(!resp.has_region_error(), "{:?}", resp.get_region_error());
        assert!(resp.errors.is_empty(), "{:?}", resp.get_errors());
    }

    pub fn must_kv_pessimistic_prewrite(
        &mut self,
        region_id: u64,
        muts: Vec<Mutation>,
        pk: Vec<u8>,
        ts: TimeStamp,
        for_update_ts: TimeStamp,
    ) {
        let mut prewrite_req = PrewriteRequest::default();
        prewrite_req.set_context(self.get_context(region_id));
        prewrite_req.set_mutations(muts.into_iter().collect());
        prewrite_req.primary_lock = pk;
        prewrite_req.start_version = ts.into_inner();
        prewrite_req.lock_ttl = prewrite_req.start_version + 1;
        prewrite_req.for_update_ts = for_update_ts.into_inner();
        prewrite_req
            .mut_pessimistic_actions()
            .push(DoPessimisticCheck);
        let prewrite_resp = self
            .get_tikv_client(region_id)
            .kv_prewrite(&prewrite_req)
            .unwrap();
        assert!(
            !prewrite_resp.has_region_error(),
            "{:?}",
            prewrite_resp.get_region_error()
        );
        assert!(
            prewrite_resp.errors.is_empty(),
            "{:?}",
            prewrite_resp.get_errors()
        );
    }

    pub fn must_kv_txn_heartbeat(
        &mut self,
        region_id: u64,
        pk: Vec<u8>,
        ts: TimeStamp,
        advise_lock_ttl: TimeStamp,
    ) {
        let mut heartbeat_req = TxnHeartBeatRequest::default();
        heartbeat_req.set_context(self.get_context(region_id));
        heartbeat_req.primary_lock = pk;
        heartbeat_req.start_version = ts.into_inner();
        heartbeat_req.advise_lock_ttl = advise_lock_ttl.into_inner();
        let heartbeat_resp = self
            .get_tikv_client(region_id)
            .kv_txn_heart_beat(&heartbeat_req)
            .unwrap();
        assert!(!heartbeat_resp.has_region_error());
        assert!(!heartbeat_resp.has_error());
        assert_eq!(heartbeat_resp.lock_ttl, advise_lock_ttl.into_inner());
    }

    pub fn async_kv_commit(
        &mut self,
        region_id: u64,
        keys: Vec<Vec<u8>>,
        start_ts: TimeStamp,
        commit_ts: TimeStamp,
    ) -> ClientUnaryReceiver<CommitResponse> {
        let mut commit_req = CommitRequest::default();
        commit_req.set_context(self.get_context(region_id));
        commit_req.start_version = start_ts.into_inner();
        commit_req.set_keys(keys.into_iter().collect());
        commit_req.commit_version = commit_ts.into_inner();
        self.get_tikv_client(region_id)
            .kv_commit_async(&commit_req)
            .unwrap()
    }

    pub fn async_kv_txn_heartbeat(
        &mut self,
        region_id: u64,
        pk: Vec<u8>,
        ts: TimeStamp,
        advise_lock_ttl: TimeStamp,
    ) -> ClientUnaryReceiver<TxnHeartBeatResponse> {
        let mut heartbeat_req = TxnHeartBeatRequest::default();
        heartbeat_req.set_context(self.get_context(region_id));
        heartbeat_req.primary_lock = pk;
        heartbeat_req.start_version = ts.into_inner();
        heartbeat_req.advise_lock_ttl = advise_lock_ttl.into_inner();
        self.get_tikv_client(region_id)
            .kv_txn_heart_beat_async(&heartbeat_req)
            .unwrap()
    }

    pub fn get_context(&mut self, region_id: u64) -> Context {
        let epoch = self.cluster.get_region_epoch(region_id);
        let leader = self.cluster.leader_of_region(region_id).unwrap();
        let api_version = self.cluster.api_version();
        let mut context = Context::default();
        context.set_region_id(region_id);
        context.set_peer(leader);
        context.set_region_epoch(epoch);
        context.set_api_version(api_version);
        context
    }

    pub fn get_tikv_client(&mut self, region_id: u64) -> &TikvClient {
        let leader = self.cluster.leader_of_region(region_id).unwrap();
        let store_id = leader.get_store_id();
        let addr = self.cluster.get_addr(store_id);
        let env = self.env.clone();
        self.tikv_cli
            .entry(leader.get_store_id())
            .or_insert_with(|| {
                let channel = ChannelBuilder::new(env).connect(&addr);
                TikvClient::new(channel)
            })
    }

    pub fn get_region_cdc_client(&mut self, region_id: u64) -> &ChangeDataClient {
        let leader = self.cluster.leader_of_region(region_id).unwrap();
        let store_id = leader.get_store_id();
        let addr = self.cluster.get_addr(store_id);
        let env = self.env.clone();
        self.cdc_cli.entry(store_id).or_insert_with(|| {
            let channel = ChannelBuilder::new(env)
                .max_receive_message_len(i32::MAX)
                .connect(&addr);
            ChangeDataClient::new(channel)
        })
    }

    pub fn get_store_cdc_client(&mut self, store_id: u64) -> &ChangeDataClient {
        let addr = self.cluster.get_addr(store_id);
        let env = self.env.clone();
        self.cdc_cli.entry(store_id).or_insert_with(|| {
            let channel = ChannelBuilder::new(env).connect(&addr);
            ChangeDataClient::new(channel)
        })
    }

    pub fn get_txn_concurrency_manager(&self, store_id: u64) -> Option<ConcurrencyManager> {
        self.concurrency_managers.get(&store_id).cloned()
    }

    pub fn set_tso(&self, ts: impl Into<TimeStamp>) {
        self.cluster.pd_client().set_tso(ts.into());
    }

    pub fn flush_causal_timestamp_for_region(&mut self, region_id: u64) {
        let leader = self.cluster.leader_of_region(region_id).unwrap();
        block_on(self.cluster.flush_causal_ts(leader.store_id));
    }

    pub fn must_wait_delegate_condition(
        &self,
        node_id: u64,
        region_id: u64,
        cond: Arc<dyn Fn(Option<&Delegate>) -> bool + Sync + Send>,
    ) {
        let scheduler = self.endpoints[&node_id].clone();
        let start = Instant::now();
        loop {
            sleep_ms(100);
            let (tx, rx) = mpsc::sync_channel(1);
            let c = cond.clone();
            let checker = move |d: Option<&Delegate>| {
                tx.send(c(d)).unwrap();
            };
            scheduler
                .schedule(Task::Validate(Validate::Region(
                    region_id,
                    Box::new(checker),
                )))
                .unwrap();
            if rx.recv().unwrap() {
                return;
            }
            if start.elapsed() > Duration::from_secs(5) {
                panic!("wait delegate timeout");
            }
        }
    }

    pub fn must_kv_prepare_flashback(
        &mut self,
        region_id: u64,
        start_key: &[u8],
        end_key: &[u8],
        start_ts: TimeStamp,
    ) {
        let mut prepare_flashback_req = PrepareFlashbackToVersionRequest::default();
        prepare_flashback_req.set_context(self.get_context(region_id));
        prepare_flashback_req.set_start_key(start_key.to_vec());
        prepare_flashback_req.set_end_key(end_key.to_vec());
        prepare_flashback_req.set_start_ts(start_ts.into_inner());
        let prepare_flashback_resp = self
            .get_tikv_client(region_id)
            .kv_prepare_flashback_to_version(&prepare_flashback_req)
            .unwrap();
        assert!(
            !prepare_flashback_resp.has_region_error(),
            "{:?}",
            prepare_flashback_resp.get_region_error()
        );
    }

    pub fn must_kv_flashback(
        &mut self,
        region_id: u64,
        start_key: &[u8],
        end_key: &[u8],
        start_ts: TimeStamp,
        commit_ts: TimeStamp,
        version: TimeStamp,
    ) {
        let mut flashback_req = FlashbackToVersionRequest::default();
        flashback_req.set_context(self.get_context(region_id));
        flashback_req.set_start_key(start_key.to_vec());
        flashback_req.set_end_key(end_key.to_vec());
        flashback_req.set_start_ts(start_ts.into_inner());
        flashback_req.set_commit_ts(commit_ts.into_inner());
        flashback_req.set_version(version.into_inner());
        let flashback_resp = self
            .get_tikv_client(region_id)
            .kv_flashback_to_version(&flashback_req)
            .unwrap();
        assert!(
            !flashback_resp.has_region_error(),
            "{:?}",
            flashback_resp.get_region_error()
        );
    }
}
