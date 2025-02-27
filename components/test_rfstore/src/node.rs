// Copyright 2025 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    collections::HashMap,
    path::Path,
    sync::{Arc, Mutex, RwLock},
};

use cloud_server::{node::Node, server::Result as ServerResult};
use collections::HashSet;
use concurrency_manager::ConcurrencyManager;
use kvproto::{raft_cmdpb::*, raft_serverpb::RaftMessage};
use raftstore::coprocessor::CoprocessorHost;
use rfstore::{
    router::{LocalReadRouter, RaftStoreRouter},
    store::{Callback, Engines, LocalReader, RaftBatchSystem, StoreMeta, StoreMsg, Transport},
    Error as RaftError, RaftRouter, Result, ServerRaftStoreRouter,
};
use sst_importer::SstImporter;
use test_pd_client::TestPdClient;
use test_raftstore::Config;
use tikv_util::{
    box_err,
    config::VersionTrack,
    time::ThreadReadId,
    worker::{Builder as WorkerBuilder, LazyWorker},
};

use crate::{Cluster, Filter, SimulateTransport, Simulator};

#[derive(Clone, Default)]
pub struct ChannelTransport {
    pub(crate) routers: Arc<Mutex<HashMap<u64, SimulateTransport<ServerRaftStoreRouter>>>>,
}

impl Transport for ChannelTransport {
    fn send(&mut self, msg: RaftMessage) -> Result<()> {
        let to_store = msg.get_to_peer().get_store_id();
        let routers = self.routers.lock().unwrap();
        match routers.get(&to_store) {
            Some(h) => {
                h.send_raft_msg(msg);
                Ok(())
            }
            _ => Err(box_err!("missing sender for store {}", to_store)),
        }
    }

    fn need_flush(&self) -> bool {
        false
    }

    fn flush(&mut self) {}
}

type SimulateChannelTransport = SimulateTransport<ChannelTransport>;

pub struct NodeCluster {
    trans: ChannelTransport,
    pd_client: Arc<TestPdClient>,
    nodes: HashMap<u64, Node>,
    simulate_trans: HashMap<u64, SimulateChannelTransport>,
    concurrency_managers: HashMap<u64, ConcurrencyManager>,
}

impl NodeCluster {
    pub fn new(pd_client: Arc<TestPdClient>) -> Self {
        Self {
            trans: ChannelTransport::default(),
            pd_client,
            nodes: HashMap::default(),
            simulate_trans: HashMap::default(),
            concurrency_managers: HashMap::default(),
        }
    }

    pub fn get_node_router(&self, node_id: u64) -> SimulateTransport<ServerRaftStoreRouter> {
        self.trans
            .routers
            .lock()
            .unwrap()
            .get(&node_id)
            .cloned()
            .unwrap()
    }

    pub fn get_node(&mut self, node_id: u64) -> Option<&mut Node> {
        self.nodes.get_mut(&node_id)
    }

    pub fn get_concurrency_manager(&self, node_id: u64) -> ConcurrencyManager {
        self.concurrency_managers.get(&node_id).unwrap().clone()
    }
}

impl Simulator for NodeCluster {
    fn run_node(
        &mut self,
        node_id: u64,
        cfg: Config,
        engines: Engines,
        store_meta: StoreMeta,
        router: RaftRouter,
        system: RaftBatchSystem,
    ) -> ServerResult<u64> {
        assert!(node_id == 0 || !self.nodes.contains_key(&node_id));
        let pd_worker = LazyWorker::new("test-pd-worker");

        let simulate_trans = SimulateTransport::new(self.trans.clone());
        let bg_worker = WorkerBuilder::new("background").thread_count(2).create();

        let raft_store = cfg.raft_store.clone();
        let rf_store_cfg = rfstore::store::Config::from_old(&raft_store, &cfg.coprocessor);
        let store_cfg_tracker = Arc::new(VersionTrack::new(rf_store_cfg));

        let mut node = Node::new(
            system,
            &cfg.server,
            store_cfg_tracker.clone(),
            self.pd_client.clone(),
            bg_worker,
            "",
        );
        node.try_bootstrap_store(engines.clone())
            .unwrap_or_else(|e| panic!("failed to bootstrap node id: {}", e));

        // Create coprocessor.
        let coprocessor_host = CoprocessorHost::default();

        let cm = ConcurrencyManager::new(1.into());
        let cm_clone = cm.clone();

        let store_path = Path::new(&cfg.storage.data_dir).to_owned();
        let importer = {
            let dir = store_path.join("import-sst");
            Arc::new(SstImporter::new(&cfg.import, dir, None, cfg.storage.api_version()).unwrap())
        };
        // self.importers.insert(node_id, importer.clone());

        let local_reader = LocalReader::new(
            engines.kv.clone(),
            store_meta.readers.clone(),
            router.clone(),
        );

        node.start(
            engines,
            Box::new(simulate_trans.clone()),
            pd_worker,
            store_meta,
            coprocessor_host,
            importer,
            cm,
        )?;
        assert!(node_id == 0 || node_id == node.id());

        let node_id = node.id();
        self.concurrency_managers.insert(node_id, cm_clone);

        let router = ServerRaftStoreRouter::new(router, local_reader);
        self.trans
            .routers
            .lock()
            .unwrap()
            .insert(node_id, SimulateTransport::new(router));
        self.nodes.insert(node_id, node);
        self.simulate_trans.insert(node_id, simulate_trans);

        Ok(node_id)
    }

    fn stop_node(&mut self, node_id: u64) {
        if let Some(mut node) = self.nodes.remove(&node_id) {
            node.stop();
        }
        self.trans.routers.lock().unwrap().remove(&node_id).unwrap();
    }
    fn get_node_ids(&self) -> HashSet<u64> {
        self.nodes.keys().cloned().collect()
    }

    fn async_command_on_node(
        &self,
        node_id: u64,
        request: RaftCmdRequest,
        cb: Callback,
    ) -> Result<()> {
        if !self.trans.routers.lock().unwrap().contains_key(&node_id) {
            return Err(box_err!("missing sender for store {}", node_id));
        }

        let router = self
            .trans
            .routers
            .lock()
            .unwrap()
            .get(&node_id)
            .cloned()
            .unwrap();

        // PrepareMerge must sent as store message.
        let is_store_msg = request.has_admin_request()
            && request.get_admin_request().get_cmd_type() == AdminCmdType::PrepareMerge;
        if !is_store_msg {
            router.send_command(request, cb);
        } else {
            router.send_store_msg(StoreMsg::PrepareMerge {
                region_id: request.get_header().region_id,
                req: request,
                callback: cb,
            });
        }
        Ok(())
    }

    fn send_raft_msg(&mut self, msg: RaftMessage) -> Result<()> {
        self.trans.send(msg)
    }

    fn get_router(&self, node_id: u64) -> Option<RaftRouter> {
        self.nodes.get(&node_id).map(|node| node.get_router())
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
        let mut trans = self.trans.routers.lock().unwrap();
        trans.get_mut(&node_id).unwrap().add_filter(filter);
    }

    fn clear_recv_filters(&mut self, node_id: u64) {
        let mut trans = self.trans.routers.lock().unwrap();
        trans.get_mut(&node_id).unwrap().clear_filters();
    }

    fn async_read(
        &mut self,
        node_id: u64,
        batch_id: Option<ThreadReadId>,
        request: RaftCmdRequest,
        cb: Callback,
    ) {
        if !self.trans.routers.lock().unwrap().contains_key(&node_id) {
            let mut resp = RaftCmdResponse::default();
            let e: RaftError = box_err!("missing sender for store {}", node_id);
            resp.mut_header().set_error(e.into());
            cb.invoke_with_response(resp);
            return;
        }
        let mut guard = self.trans.routers.lock().unwrap();
        let router = guard.get_mut(&node_id).unwrap();
        router.read(batch_id, request, cb).unwrap();
    }
}

// Compare to server cluster, node cluster does not have server layer and
// storage layer.
pub fn new_node_cluster(mut id: u16, count: usize) -> Cluster<NodeCluster> {
    // 0 is invalid cluster id.
    if id == 0 {
        id = 1;
    }
    let pd_client = Arc::new(TestPdClient::new(id as u64, false));
    let sim = Arc::new(RwLock::new(NodeCluster::new(Arc::clone(&pd_client))));
    Cluster::new(id, count, sim, pd_client)
}
