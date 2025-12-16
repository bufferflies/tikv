// Copyright 2016 TiKV Project Authors. Licensed under Apache-2.0.

use std::sync::{Arc, Mutex, RwLock};

use collections::{HashMap, HashSet};
use concurrency_manager::ConcurrencyManager;
use engine_rocks::{RocksEngine, RocksSnapshot};
use engine_test::raft::RaftTestEngine;
use kvproto::{
    kvrpcpb::ApiVersion,
    raft_cmdpb::*,
    raft_serverpb::{self, RaftMessage},
};
use raftstore::{
    coprocessor::CoprocessorHost,
    errors::Error as RaftError,
    router::{LocalReadRouter, RaftStoreRouter, ServerRaftStoreRouter},
    store::{fsm::RaftRouter, *},
    Result,
};
use tempfile::TempDir;
use test_pd_client::TestPdClient;
use tikv::config::ConfigController;
use tikv_util::time::ThreadReadId;

use super::*;

pub struct ChannelTransportCore {
    snap_paths: HashMap<u64, (SnapManager, TempDir)>,
    routers: HashMap<u64, SimulateTransport<ServerRaftStoreRouter<RocksEngine, RaftTestEngine>>>,
}

#[derive(Clone)]
pub struct ChannelTransport {
    core: Arc<Mutex<ChannelTransportCore>>,
}

impl ChannelTransport {
    pub fn new() -> ChannelTransport {
        ChannelTransport {
            core: Arc::new(Mutex::new(ChannelTransportCore {
                snap_paths: HashMap::default(),
                routers: HashMap::default(),
            })),
        }
    }
}

impl Default for ChannelTransport {
    fn default() -> Self {
        Self::new()
    }
}

impl Transport for ChannelTransport {
    fn send(&mut self, _msg: RaftMessage) -> Result<()> {
        unimplemented!()
    }

    fn set_store_allowlist(&mut self, _allowlist: Vec<u64>) {
        unimplemented!();
    }

    fn need_flush(&self) -> bool {
        false
    }

    fn flush(&mut self) {}
}

type SimulateChannelTransport = SimulateTransport<ChannelTransport>;

pub struct NodeCluster {
    trans: ChannelTransport,
    _pd_client: Arc<TestPdClient>,
    snap_mgrs: HashMap<u64, SnapManager>,
    cfg_controller: Option<ConfigController>,
    simulate_trans: HashMap<u64, SimulateChannelTransport>,
    concurrency_managers: HashMap<u64, ConcurrencyManager>,
    #[allow(clippy::type_complexity)]
    post_create_coprocessor_host: Option<Box<dyn Fn(u64, &mut CoprocessorHost<RocksEngine>)>>,
}

impl NodeCluster {
    pub fn new(pd_client: Arc<TestPdClient>) -> NodeCluster {
        NodeCluster {
            trans: ChannelTransport::new(),
            _pd_client: pd_client,
            snap_mgrs: HashMap::default(),
            cfg_controller: None,
            simulate_trans: HashMap::default(),
            concurrency_managers: HashMap::default(),
            post_create_coprocessor_host: None,
        }
    }
}

impl NodeCluster {
    // Set a function that will be invoked after creating each CoprocessorHost. The
    // first argument of `op` is the node_id.
    // Set this before invoking `run_node`.
    #[allow(clippy::type_complexity)]
    pub fn post_create_coprocessor_host(
        &mut self,
        op: Box<dyn Fn(u64, &mut CoprocessorHost<RocksEngine>)>,
    ) {
        self.post_create_coprocessor_host = Some(op)
    }

    pub fn get_concurrency_manager(&self, node_id: u64) -> ConcurrencyManager {
        self.concurrency_managers.get(&node_id).unwrap().clone()
    }

    pub fn get_cfg_controller(&self) -> Option<&ConfigController> {
        self.cfg_controller.as_ref()
    }
}

impl Simulator for NodeCluster {
    fn get_snap_dir(&self, node_id: u64) -> String {
        self.trans.core.lock().unwrap().snap_paths[&node_id]
            .1
            .path()
            .to_str()
            .unwrap()
            .to_owned()
    }

    fn get_snap_mgr(&self, node_id: u64) -> &SnapManager {
        self.snap_mgrs.get(&node_id).unwrap()
    }

    fn stop_node(&mut self, _node_id: u64) {
        unimplemented!()
    }

    fn get_node_ids(&self) -> HashSet<u64> {
        unimplemented!()
    }

    fn async_command_on_node_with_opts(
        &self,
        node_id: u64,
        request: RaftCmdRequest,
        cb: Callback<RocksSnapshot>,
        opts: RaftCmdExtraOpts,
    ) -> Result<()> {
        if !self
            .trans
            .core
            .lock()
            .unwrap()
            .routers
            .contains_key(&node_id)
        {
            return Err(box_err!("missing sender for store {}", node_id));
        }

        let router = self
            .trans
            .core
            .lock()
            .unwrap()
            .routers
            .get(&node_id)
            .cloned()
            .unwrap();
        router.send_command(request, cb, opts)
    }

    fn async_read(
        &mut self,
        node_id: u64,
        batch_id: Option<ThreadReadId>,
        request: RaftCmdRequest,
        cb: Callback<RocksSnapshot>,
    ) {
        if !self
            .trans
            .core
            .lock()
            .unwrap()
            .routers
            .contains_key(&node_id)
        {
            let mut resp = RaftCmdResponse::default();
            let e: RaftError = box_err!("missing sender for store {}", node_id);
            resp.mut_header().set_error(e.into());
            cb.invoke_with_response(resp);
            return;
        }
        let mut guard = self.trans.core.lock().unwrap();
        let router = guard.routers.get_mut(&node_id).unwrap();
        router.read(batch_id, request, cb).unwrap();
    }

    fn send_raft_msg(&mut self, msg: raft_serverpb::RaftMessage) -> Result<()> {
        self.trans.send(msg)
    }

    fn add_send_filter(&mut self, node_id: u64, filter: Box<dyn Filter>) {
        self.simulate_trans
            .get_mut(&node_id)
            .unwrap()
            .add_filter(filter);
    }

    fn clear_send_filters(&mut self, node_id: u64) {
        self.simulate_trans
            .get_mut(&node_id)
            .unwrap()
            .clear_filters();
    }

    fn add_recv_filter(&mut self, node_id: u64, filter: Box<dyn Filter>) {
        let mut trans = self.trans.core.lock().unwrap();
        trans.routers.get_mut(&node_id).unwrap().add_filter(filter);
    }

    fn clear_recv_filters(&mut self, node_id: u64) {
        let mut trans = self.trans.core.lock().unwrap();
        trans.routers.get_mut(&node_id).unwrap().clear_filters();
    }

    fn get_router(&self, _node_id: u64) -> Option<RaftRouter<RocksEngine, RaftTestEngine>> {
        unimplemented!()
    }
}

pub fn new_node_cluster(id: u64, count: usize) -> Cluster<NodeCluster> {
    let pd_client = Arc::new(TestPdClient::new(id, false));
    let sim = Arc::new(RwLock::new(NodeCluster::new(Arc::clone(&pd_client))));
    Cluster::new(id, count, sim, pd_client, ApiVersion::V1)
}

pub fn new_incompatible_node_cluster(id: u64, count: usize) -> Cluster<NodeCluster> {
    let pd_client = Arc::new(TestPdClient::new(id, true));
    let sim = Arc::new(RwLock::new(NodeCluster::new(Arc::clone(&pd_client))));
    Cluster::new(id, count, sim, pd_client, ApiVersion::V1)
}
