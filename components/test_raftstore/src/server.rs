// Copyright 2016 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    sync::{Arc, Mutex},
    usize,
};

use causal_ts::CausalTsProviderImpl;
use collections::{HashMap, HashSet};
use concurrency_manager::ConcurrencyManager;
use engine_rocks::{RocksEngine, RocksSnapshot};
use engine_test::raft::RaftTestEngine;
use grpcio::Service;
use grpcio_health::HealthService;
use kvproto::{kvrpcpb::ApiVersion, raft_cmdpb::*, raft_serverpb};
use raftstore::{
    coprocessor::{CoprocessorHost, RegionInfoAccessor},
    router::ServerRaftStoreRouter,
    store::{fsm::RaftRouter, msg::RaftCmdExtraOpts, Callback, RegionSnapshot, SnapManager},
    Result,
};
use security::SecurityManager;
use tikv::{
    import::SstImporter,
    server::{resolve::StoreAddrResolver, RaftKv, Result as ServerResult},
};
use tikv_util::time::ThreadReadId;
use txn_types::TxnExtraScheduler;

use super::*;

type SimulateStoreTransport = SimulateTransport<ServerRaftStoreRouter<RocksEngine, RaftTestEngine>>;

pub type SimulateEngine = RaftKv<RocksEngine, SimulateStoreTransport>;

#[derive(Default, Clone)]
pub struct AddressMap {
    addrs: Arc<Mutex<HashMap<u64, String>>>,
}

impl AddressMap {
    pub fn get(&self, store_id: u64) -> Option<String> {
        let addrs = self.addrs.lock().unwrap();
        addrs.get(&store_id).cloned()
    }

    pub fn insert(&mut self, store_id: u64, addr: String) {
        self.addrs.lock().unwrap().insert(store_id, addr);
    }
}

impl StoreAddrResolver for AddressMap {
    fn resolve(
        &self,
        store_id: u64,
        cb: Box<dyn FnOnce(ServerResult<String>) + Send>,
    ) -> ServerResult<()> {
        let addr = self.get(store_id);
        match addr {
            Some(addr) => cb(Ok(addr)),
            None => cb(Err(box_err!(
                "unable to find address for store {}",
                store_id
            ))),
        }
        Ok(())
    }
}

type PendingServices = Vec<Box<dyn Fn() -> Service>>;
type CopHooks = Vec<Box<dyn Fn(&mut CoprocessorHost<RocksEngine>)>>;

pub struct ServerCluster {
    pub storages: HashMap<u64, SimulateEngine>,
    pub region_info_accessors: HashMap<u64, RegionInfoAccessor>,
    pub importers: HashMap<u64, Arc<SstImporter>>,
    pub pending_services: HashMap<u64, PendingServices>,
    pub coprocessor_hooks: HashMap<u64, CopHooks>,
    pub health_services: HashMap<u64, HealthService>,
    pub security_mgr: Arc<SecurityManager>,
    pub txn_extra_schedulers: HashMap<u64, Arc<dyn TxnExtraScheduler>>,
    pub causal_ts_providers: HashMap<u64, Arc<CausalTsProviderImpl>>,
}

impl ServerCluster {
    pub fn get_addr(&self, _node_id: u64) -> String {
        unimplemented!()
    }

    pub fn get_server_router(&self, _node_id: u64) -> SimulateStoreTransport {
        unimplemented!()
    }

    pub fn get_concurrency_manager(&self, _node_id: u64) -> ConcurrencyManager {
        unimplemented!()
    }

    pub fn get_causal_ts_provider(&self, _node_id: u64) -> Option<Arc<CausalTsProviderImpl>> {
        unimplemented!()
    }
}

impl Simulator for ServerCluster {
    fn get_snap_dir(&self, _node_id: u64) -> String {
        unimplemented!()
    }

    fn get_snap_mgr(&self, _node_id: u64) -> &SnapManager {
        unimplemented!()
    }

    fn stop_node(&mut self, _node_id: u64) {
        unimplemented!()
    }

    fn get_node_ids(&self) -> HashSet<u64> {
        unimplemented!()
    }

    fn async_command_on_node_with_opts(
        &self,
        _node_id: u64,
        _request: RaftCmdRequest,
        _cb: Callback<RocksSnapshot>,
        _opts: RaftCmdExtraOpts,
    ) -> Result<()> {
        unimplemented!()
    }

    fn async_read(
        &mut self,
        _node_id: u64,
        _batch_id: Option<ThreadReadId>,
        _request: RaftCmdRequest,
        _cb: Callback<RocksSnapshot>,
    ) {
        unimplemented!()
    }

    fn send_raft_msg(&mut self, _raft_msg: raft_serverpb::RaftMessage) -> Result<()> {
        unimplemented!()
    }

    fn add_send_filter(&mut self, _node_id: u64, _filter: Box<dyn Filter>) {
        unimplemented!()
    }

    fn clear_send_filters(&mut self, _node_id: u64) {
        unimplemented!()
    }

    fn add_recv_filter(&mut self, _node_id: u64, _filter: Box<dyn Filter>) {
        unimplemented!()
    }

    fn clear_recv_filters(&mut self, _node_id: u64) {
        unimplemented!()
    }

    fn get_router(&self, _node_id: u64) -> Option<RaftRouter<RocksEngine, RaftTestEngine>> {
        unimplemented!()
    }
}

impl Cluster<ServerCluster> {
    pub fn must_get_snapshot_of_region(
        &mut self,
        _region_id: u64,
    ) -> RegionSnapshot<RocksSnapshot> {
        unimplemented!()
    }
    pub fn must_get_raft_engine(&self, _store_id: u64) -> SimulateEngine {
        unimplemented!()
    }
}

pub fn new_server_cluster(_id: u64, _count: usize) -> Cluster<ServerCluster> {
    unimplemented!()
}

pub fn new_server_cluster_with_api_ver(
    _id: u64,
    _count: usize,
    _api_ver: ApiVersion,
) -> Cluster<ServerCluster> {
    unimplemented!()
}
