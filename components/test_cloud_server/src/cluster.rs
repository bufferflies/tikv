// Copyright 2022 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    thread::sleep,
    time::Duration,
};

use cloud_server::TikvServer;
use dashmap::DashMap;
use futures::executor::block_on;
use grpcio::{Channel, ChannelBuilder, EnvBuilder, Environment};
use kvengine::{dfs::Dfs, ShardStats};
use kvproto::{
    kvrpcpb,
    kvrpcpb::{Mutation, Op},
    raft_cmdpb::RaftCmdRequest,
};
use pd_client::PdClient;
use rfstore::{store::Callback, RaftStoreRouter};
use security::SecurityManager;
use tempfile::TempDir;
use test_pd_client::TestPdClient;
use test_raftstore::find_peer;
use tikv::{config::TikvConfig, import::SstImporter};
use tikv_util::{
    config::{ReadableDuration, ReadableSize},
    info,
    thread_group::GroupProperties,
    time::Instant,
};

use crate::{
    client::{ApiV2NoPrefixCodec, ClusterClient, ClusterTxnClient, RefStore},
    keyspace::{ClusterKeyspaceClient, KeyspaceManager},
    scheduler::Scheduler,
    txnlock::lock_resolver::LockResolver,
};

#[allow(dead_code)]
pub struct ServerCluster {
    servers: HashMap<u16 /* node_id */, TikvServer>,
    tmp_dir: TempDir,
    env: Arc<Environment>,
    pd_client: Arc<TestPdClient>,
    pd_server: Option<test_pd::Server<test_pd_client::Service>>,
    security_mgr: Arc<SecurityManager>,
    dfs: Option<Arc<dyn Dfs>>,
    channels: HashMap<u64, Channel>,
    ref_store: Arc<Mutex<RefStore>>,
    schedule_lock: Arc<DashMap<u64, Arc<Mutex<()>>>>,
    confs: HashMap<u16 /* node_id */, TikvConfig>,
    keyspace_manager: KeyspaceManager,
}

impl ServerCluster {
    // The node id is statically assigned, the temp dir and server address are
    // calculated by the node id.
    pub fn new<F>(nodes: Vec<u16>, update_conf: F) -> ServerCluster
    where
        F: Fn(u16, &mut TikvConfig),
    {
        tikv_util::thread_group::set_properties(Some(GroupProperties::default()));
        // Use prefix to generate `tmp_dir` to indicate the usage of dir more clearly.
        let tmp_dir = tempfile::Builder::new()
            .prefix("cluster_")
            .tempdir()
            .unwrap();
        let mut cluster = Self {
            servers: HashMap::new(),
            tmp_dir,
            env: Arc::new(EnvBuilder::new().cq_count(2).build()),
            pd_client: Arc::new(TestPdClient::new(1, false)),
            pd_server: None,
            security_mgr: Arc::new(SecurityManager::new(&Default::default()).unwrap()),
            dfs: None,
            channels: HashMap::new(),
            ref_store: Arc::new(Mutex::new(RefStore::default())),
            schedule_lock: Arc::new(DashMap::new()),
            confs: Default::default(),
            keyspace_manager: Default::default(),
        };
        for node_id in nodes {
            cluster.start_node(node_id, &update_conf);
        }
        cluster.wait_pd_region_min_count(1);
        cluster
    }

    pub fn get_dfs(&self) -> Option<Arc<dyn Dfs>> {
        self.dfs.clone()
    }

    fn prepare_dfs(config: &TikvConfig) -> Arc<dyn Dfs> {
        let dfs_conf = &config.dfs;
        if dfs_conf.s3_bucket.is_empty() && dfs_conf.s3_endpoint.is_empty()
            || dfs_conf.s3_endpoint == "local"
        {
            let local_path = PathBuf::from(&config.storage.data_dir).join(Path::new("local"));
            Arc::new(kvengine::dfs::LocalFs::new(&local_path))
        } else if dfs_conf.s3_endpoint == "memory" {
            Arc::new(kvengine::dfs::InMemFs::new())
        } else {
            Arc::new(kvengine::dfs::S3Fs::new(
                dfs_conf.prefix.clone(),
                dfs_conf.s3_endpoint.clone(),
                dfs_conf.s3_key_id.clone(),
                dfs_conf.s3_secret_key.clone(),
                dfs_conf.s3_region.clone(),
                dfs_conf.s3_bucket.clone(),
            ))
        }
    }

    /// `update_conf` is based on the existed config.
    pub fn start_node<F>(&mut self, node_id: u16, update_conf: F)
    where
        F: Fn(u16, &mut TikvConfig),
    {
        let mut config = if let Some(config) = self.confs.remove(&node_id) {
            config
        } else {
            new_test_config(self.tmp_dir.path(), node_id)
        };
        update_conf(node_id, &mut config);
        self.confs.insert(node_id, config.clone());

        std::fs::create_dir_all(&config.storage.data_dir).unwrap();
        let dfs = self.dfs.get_or_insert_with(|| Self::prepare_dfs(&config));
        let mut server = TikvServer::setup(
            config,
            self.security_mgr.clone(),
            self.env.clone(),
            self.pd_client.clone(),
            dfs.clone(),
        );
        server.run();
        let store_id = server.get_store_id();
        if let std::collections::hash_map::Entry::Vacant(e) = self.channels.entry(store_id) {
            let addr = node_addr(node_id);
            let channel = ChannelBuilder::new(self.env.clone()).connect(&addr);
            e.insert(channel);
        }
        self.servers.insert(node_id, server);
    }

    pub fn get_stores(&self) -> Vec<u64> {
        self.channels.keys().copied().collect()
    }

    pub fn get_store_id(&self, node_id: u16) -> u64 {
        self.servers.get(&node_id).unwrap().get_store_id()
    }

    pub fn get_pd_client(&self) -> Arc<TestPdClient> {
        self.pd_client.clone()
    }

    pub fn get_nodes(&self) -> Vec<u16> {
        self.servers.keys().copied().collect()
    }

    pub fn get_node_config(&self, node_id: u16) -> &TikvConfig {
        self.confs.get(&node_id).unwrap()
    }

    pub fn stop(&mut self) {
        let nodes = self.get_nodes();
        for node_id in nodes {
            self.stop_node(node_id);
        }
    }

    // Stop node gracefully.
    pub fn stop_node(&mut self, node_id: u16) {
        if let Some(node) = self.servers.remove(&node_id) {
            // Force stop node to cover the case wal chunk recovery.
            node.force_stop(false);
        }
    }

    // Stop node without flush rfengine dfs worker if force is true.
    fn stop_node_force(&mut self, node_id: u16, force: bool) {
        if let Some(node) = self.servers.remove(&node_id) {
            // Force stop node to cover the case wal chunk recovery.
            node.force_stop(force);
        }
    }

    pub fn restart_node(&mut self, node_id: u16, stop_dur: Duration, force: bool) {
        let store_id = self.get_store_id(node_id);
        self.stop_node_force(node_id, force);
        info!(
            "node stopped"; "node" => node_id, "store" => store_id, "force" => force,
        );

        std::thread::sleep(stop_dur);
        self.start_node(node_id, |_, _| {});
        info!("node restarted"; "node" => node_id, "store" => store_id);
    }

    pub fn get_kvengine(&self, node_id: u16) -> kvengine::Engine {
        let server = self.servers.get(&node_id).unwrap();
        server.get_kv_engine()
    }

    pub fn get_rfengine(&self, node_id: u16) -> rfengine::RfEngine {
        let server = self.servers.get(&node_id).unwrap();
        server.get_raft_engine()
    }

    pub fn get_snap(&self, node_id: u16, key: &[u8]) -> kvengine::SnapAccess {
        let engine = self.get_kvengine(node_id);
        let region = self.pd_client.get_region(key).unwrap();
        engine.get_snap_access(region.id).unwrap()
    }

    pub fn get_sst_importer(&self, node_id: u16) -> Arc<SstImporter> {
        let server = self.servers.get(&node_id).unwrap();
        server.get_sst_importer()
    }

    pub fn send_raft_command(&self, cmd: RaftCmdRequest) {
        let store_id = cmd.get_header().get_peer().get_store_id();
        for server in self.servers.values() {
            if server.get_store_id() == store_id {
                server.get_raft_router().send_command(cmd, Callback::None);
                return;
            }
        }
    }

    pub fn wait_region_replicated(&self, key: &[u8], replica_cnt: usize) {
        for _ in 0..10 {
            let region_info = self.pd_client.get_region_info(key).unwrap();
            let region_id = region_info.id;
            let region_ver = region_info.get_region_epoch().version;
            if region_info.region.get_peers().len() >= replica_cnt {
                let all_applied_snapshot = region_info.get_peers().iter().all(|peer| {
                    let node_id = self.get_server_node_id(peer.store_id);
                    let kv = self.get_kvengine(node_id);
                    kv.get_shard_with_ver(region_id, region_ver).is_ok()
                });
                if all_applied_snapshot {
                    return;
                }
            }
            std::thread::sleep(Duration::from_millis(1000));
        }
        panic!("region is not replicated");
    }

    pub fn wait_pd_region_count(&self, count: usize) {
        self.wait_pd_region_count_opt(count, Duration::from_secs(5));
    }

    pub fn wait_pd_region_count_opt(&self, count: usize, timeout: Duration) {
        let start_time = Instant::now_coarse();
        while start_time.saturating_elapsed() < timeout {
            if self.pd_client.get_regions_number() == count {
                return;
            }
            std::thread::sleep(Duration::from_millis(100));
        }
        panic!("pd region count not match");
    }

    pub fn wait_pd_region_min_count(&self, min_count: usize) {
        let mut region_count = 0;
        for _ in 0..10 {
            region_count = self.pd_client.get_regions_number();
            if region_count >= min_count {
                return;
            }
            std::thread::sleep(Duration::from_millis(100));
        }
        panic!(
            "pd region count {} < min_count({})",
            region_count, min_count
        );
    }

    pub fn remove_node_peers(&mut self, node_id: u16) {
        let server = self.servers.get(&node_id).unwrap();
        let store_id = server.get_store_id();
        let all_id_vers = server.get_kv_engine().get_all_shard_id_vers();
        for id_ver in &all_id_vers {
            let (region, leader) = block_on(self.pd_client.get_region_leader_by_id(id_ver.id))
                .unwrap()
                .unwrap();
            let target = if leader.store_id != store_id {
                &leader
            } else {
                region
                    .get_peers()
                    .iter()
                    .find(|x| x.store_id != store_id)
                    .unwrap()
            };
            self.pd_client
                .transfer_leader(region.id, target.clone(), vec![]);
            self.pd_client
                .region_leader_must_be(region.id, target.clone());
            if let Some(peer) = find_peer(&region, store_id) {
                self.pd_client.must_remove_peer(region.id, peer.clone());
            }
        }
        let server = self.servers.get(&node_id).unwrap();
        for _ in 0..30 {
            if server.get_kv_engine().get_all_shard_id_vers().is_empty() {
                return;
            }
            std::thread::sleep(Duration::from_millis(100));
        }
        panic!("kvengine is not empty");
    }

    fn get_server_node_id(&self, store_id: u64) -> u16 {
        for (node_id, server) in &self.servers {
            if server.get_store_id() == store_id {
                return *node_id;
            }
        }
        panic!("server not found");
    }

    pub fn new_client(&self) -> ClusterClient {
        self.new_client_opt(true, kvrpcpb::ApiVersion::V2)
    }

    pub fn new_client_opt(
        &self,
        with_lock_resolver: bool,
        api_version: kvrpcpb::ApiVersion,
    ) -> ClusterClient {
        let lock_resolver = with_lock_resolver.then(|| Box::new(self.new_lock_resolver()));
        ClusterClient {
            pd_client: self.pd_client.clone(),
            channels: self.channels.clone(),
            region_ranges: Default::default(),
            regions: Default::default(),
            ref_store: self.ref_store.clone(),
            max_ts: Default::default(),
            lock_resolver,
            api_version,
        }
    }

    pub async fn new_keyspace_client(&self) -> ClusterKeyspaceClient {
        ClusterKeyspaceClient::new(self.new_txn_client().await, self.keyspace_manager.clone())
    }

    pub fn keyspace_manager(&self) -> &KeyspaceManager {
        &self.keyspace_manager
    }

    pub fn new_scheduler(&self) -> Scheduler {
        Scheduler {
            pd: self.pd_client.clone(),
            store_ids: self.get_stores(),
            lock: self.schedule_lock.clone(),
        }
    }

    pub fn new_lock_resolver(&self) -> LockResolver {
        // `with_lock_resolver` must be false, otherwise it will cause dead loop.
        LockResolver::new(self.new_client_opt(false, kvrpcpb::ApiVersion::V2))
    }

    pub fn get_data_stats(&self) -> ClusterDataStats {
        let mut stats = ClusterDataStats::default();
        for server in self.servers.values() {
            let store_id = server.get_store_id();
            let kv_engine = server.get_kv_engine();
            stats.add(store_id, kv_engine.get_all_shard_stats());
        }
        stats
    }

    pub fn set_dfs_delay(&self, delay: Duration) {
        self.dfs.as_ref().unwrap().set_delay(delay);
    }

    // Wait shard version match between PD & kvengine.
    pub fn wait_region_version_match(&self) {
        let pd_client = self.get_pd_client();
        let ok = try_wait(
            || {
                let data_stats = self.get_data_stats();
                data_stats.check_region_version_match(&pd_client).is_ok()
            },
            10,
        );
        let data_stats = self.get_data_stats();
        if !ok {
            data_stats.check_region_version_match(&pd_client).unwrap();
        }
    }

    pub fn start_pd_server(&mut self, pd_count: usize) {
        let pd_service = test_pd_client::Service::new(self.pd_client.clone());
        let pd_server = test_pd::Server::with_case(pd_count, Arc::new(pd_service));
        self.pd_server = Some(pd_server);
    }

    pub fn pd_addrs(&self) -> Vec<(String, u16)> {
        self.pd_server.as_ref().unwrap().bind_addrs()
    }

    pub async fn new_txn_client(&self) -> ClusterTxnClient {
        let pd_endpoints = self
            .pd_addrs()
            .into_iter()
            .map(|(host, port)| format!("{}:{}", host, port))
            .collect::<Vec<_>>();
        let client = tikv_client::TransactionClient::new_with_codec(
            pd_endpoints,
            tikv_client::Config::default(),
            ApiV2NoPrefixCodec::default(),
        )
        .await
        .unwrap();
        ClusterTxnClient::new(client, self.get_pd_client(), self.new_client())
    }

    pub fn set_gc_safe_point(&self, ts: u64) {
        let _ = self.pd_client.set_gc_safe_point(ts).unwrap();
        for node_id in self.get_nodes() {
            self.get_kvengine(node_id).update_managed_safe_ts(ts);
        }
    }
}

pub fn new_test_config(base_dir: &Path, node_id: u16) -> TikvConfig {
    let mut config = TikvConfig::default();
    config.storage.data_dir = format!("{}/{}", base_dir.to_str().unwrap(), node_id);
    config.storage.api_version = 2;
    config.storage.enable_ttl = true;
    config.server.cluster_id = 1;
    config.server.addr = node_addr(node_id);
    config.server.status_addr = node_status_addr(node_id);
    config.server.grpc_keepalive_time = ReadableDuration::secs(1);
    config.server.grpc_keepalive_timeout = ReadableDuration::secs(1);
    config.dfs.s3_endpoint = "memory".to_string();
    config.dfs.zstd_compression_level = "3".to_string();
    config.raft_store.raft_base_tick_interval = ReadableDuration::millis(10);
    config.raft_store.raft_election_timeout_ticks = 50;
    config.raft_store.raft_store_max_leader_lease = ReadableDuration::millis(20);
    config.raft_store.split_region_check_tick_interval = ReadableDuration::millis(100);
    config.raft_store.raft_log_gc_tick_interval = ReadableDuration::millis(100);
    config.raft_store.pd_heartbeat_tick_interval = ReadableDuration::millis(100);
    config.raft_store.pd_store_heartbeat_tick_interval = ReadableDuration::millis(100);
    config.raft_store.max_peer_down_duration = ReadableDuration::secs(4);
    config.rocksdb.writecf.write_buffer_size = ReadableSize::kb(16);
    config.rocksdb.writecf.block_size = ReadableSize::kb(4);
    config.rocksdb.writecf.target_file_size_base = ReadableSize::kb(32);
    config.rocksdb.max_background_jobs = 2;
    config.rocksdb.max_sub_compactions = 1;
    config.rfengine.target_file_size = ReadableSize::kb(128);
    config.rfengine.wal_sync_dir = format!("{}/{}/wal", base_dir.to_str().unwrap(), node_id);

    // Work around https://github.com/tidbcloud/cloud-storage-engine/issues/882.
    config.server.raft_client_initial_reconnect_backoff = ReadableDuration::millis(100);
    config.server.raft_client_max_backoff = ReadableDuration::millis(250);

    config
}

// Keep away from 20xxx ports to work around https://github.com/tidbcloud/cloud-storage-engine/issues/658.
// TODO: Remove this work around.
fn node_addr(node_id: u16) -> String {
    format!("127.0.0.1:{}", node_id + 21000)
}

// Keep away from 3xxxx ports to work around https://github.com/tidbcloud/cloud-storage-engine/issues/658.
// TODO: Remove this work around.
fn node_status_addr(node_id: u16) -> String {
    format!("127.0.0.1:{}", node_id + 25000)
}

pub fn put_mut(key: &str, val: &str) -> Mutation {
    let mut mutation = Mutation::new();
    mutation.op = Op::Put;
    mutation.key = key.as_bytes().to_vec();
    mutation.value = val.as_bytes().to_vec();
    mutation
}

pub fn must_wait<F>(mut f: F, seconds: usize, fail_msg: &str)
where
    F: FnMut() -> bool,
{
    let begin = Instant::now_coarse();
    let timeout = Duration::from_secs(seconds as u64);
    while begin.saturating_elapsed() < timeout {
        if f() {
            return;
        }
        sleep(Duration::from_millis(100))
    }
    panic!("{}", fail_msg);
}

pub fn try_wait<F>(mut f: F, seconds: usize) -> bool
where
    F: FnMut() -> bool,
{
    let begin = Instant::now_coarse();
    let timeout = Duration::from_secs(seconds as u64);
    while begin.saturating_elapsed() < timeout {
        if f() {
            return true;
        }
        sleep(Duration::from_millis(100))
    }
    false
}

/// A extended version of `try_wait` which can save and return an extra value
/// from `f`.
pub fn try_wait_result<F, T, E>(mut f: F, seconds: usize) -> (std::result::Result<(), E>, T)
where
    F: FnMut() -> (std::result::Result<(), E>, T),
{
    let begin = Instant::now_coarse();
    let timeout = Duration::from_secs(seconds as u64);
    let mut last_err = None;
    let mut last_val = None;
    while begin.saturating_elapsed() < timeout {
        match f() {
            (Ok(()), val) => return (Ok(()), val),
            (Err(err), val) => {
                last_err = Some(err);
                last_val = Some(val);
                sleep(Duration::from_millis(100));
            }
        }
    }
    (Err(last_err.unwrap()), last_val.unwrap())
}

#[derive(Default, Debug)]
pub struct ClusterDataStats {
    regions: HashMap<u64, RegionShardStats>,
}

impl ClusterDataStats {
    fn add(&mut self, store_id: u64, shard_stats: Vec<ShardStats>) {
        for shard_stat in shard_stats {
            let region_shard_stats = self
                .regions
                .entry(shard_stat.id)
                .or_insert_with(|| RegionShardStats::new(shard_stat.id));
            region_shard_stats.shard_stats.insert(store_id, shard_stat);
        }
    }

    pub fn check_data(&self) -> Result<(), String> {
        for stats in self.regions.values() {
            let map_err_fn = |e| format!("err {} stats: {:?}", e, stats);
            stats.check_consistency().map_err(map_err_fn)?;
            stats.check_healthy().map_err(map_err_fn)?;
        }
        Ok(())
    }

    pub fn check_leader(&self) -> Result<(), String> {
        for stats in self.regions.values() {
            let map_err_fn = |e| format!("err {} stats: {:?}", e, stats);
            stats.get_leader_stats().map_err(map_err_fn)?;
        }
        Ok(())
    }

    pub fn iter_shard_stats(&self, mut f: impl FnMut(u64, &ShardStats) -> bool) {
        for region in self.regions.values() {
            for (&store_id, shard) in &region.shard_stats {
                if f(store_id, shard) {
                    return;
                }
            }
        }
    }

    fn get_region_shard_stats(&self, region_id: u64) -> Option<&ShardStats> {
        self.regions
            .get(&region_id)
            .map(|stats| stats.shard_stats.values().next().unwrap())
    }

    pub fn check_region_version_match(&self, pd_client: &TestPdClient) -> Result<(), String> {
        let regions = pd_client.get_all_regions();
        for region in regions {
            let region_id = region.get_id();
            let region_shard_stats = self.get_region_shard_stats(region_id).unwrap();
            if region_shard_stats.total_size == 0 {
                continue;
            }

            let region_pd_version = region.get_region_epoch().get_version();
            let region_shard_version = region_shard_stats.ver;
            if region_pd_version != region_shard_version {
                return Err(format!(
                    "region {} version not match, pd: {}, shard: {}",
                    region_id, region_pd_version, region_shard_version
                ));
            }
        }
        Ok(())
    }

    pub fn check_buckets(&self, pd_client: &TestPdClient, bucket_size: u64) -> Result<(), String> {
        let regions = pd_client.get_all_regions();
        for region in regions {
            let region_id = region.get_id();
            let region_shard_stats = self.get_region_shard_stats(region_id).unwrap();
            let shard_size = region_shard_stats.total_size;
            if shard_size == 0 {
                continue;
            }
            if let Some(buckets) = pd_client.get_buckets(region_id) {
                for i in 1..buckets.meta.keys.len() {
                    let prev_key = &buckets.meta.keys[i - 1];
                    let key = &buckets.meta.keys[i];
                    if !key.is_empty() {
                        assert!(prev_key < key, "region {} buckets {:?}", region_id, buckets);
                    }
                }
                let expected_bucket_count = (shard_size + bucket_size - 1) / bucket_size;
                let actual_bucket_count = buckets.meta.sizes.len() as u64;
                let ratio = expected_bucket_count as f64 / actual_bucket_count as f64;
                if !(0.3..=3.0).contains(&ratio) {
                    return Err(format!(
                        "region {} buckets {:?}, shard size {}, expected {}, actual {}",
                        region_id, buckets, shard_size, expected_bucket_count, actual_bucket_count
                    ));
                };
            } else {
                return Err(format!(
                    "region {} no buckets, shard size {}",
                    region_id, shard_size
                ));
            }
        }
        Ok(())
    }
}

#[derive(Default, Debug)]
#[allow(dead_code)]
pub struct RegionShardStats {
    region_id: u64,
    // store_id -> ShardStats
    shard_stats: HashMap<u64, ShardStats>,
}

impl RegionShardStats {
    fn new(region_id: u64) -> Self {
        Self {
            region_id,
            shard_stats: Default::default(),
        }
    }

    fn check_consistency(&self) -> Result<(), String> {
        if self.shard_stats.len() <= 1 {
            return Ok(());
        }
        let store_ids: Vec<u64> = self.shard_stats.keys().copied().collect();
        let first_id = &store_ids[0];
        let first_stats = self.shard_stats.get(first_id).unwrap();
        for store_id in &store_ids[1..] {
            let stats = self.shard_stats.get(store_id).unwrap();
            if stats.total_size != first_stats.total_size
                || stats.mem_table_count != first_stats.mem_table_count
                || stats.mem_table_size != first_stats.mem_table_size
                || stats.entries != first_stats.entries
                || stats.l0_table_count != first_stats.l0_table_count
                || stats.ver != first_stats.ver
            {
                return Err("inconsistent stats".into());
            }
        }
        Ok(())
    }

    fn get_leader_stats(&self) -> Result<&ShardStats, String> {
        let item = self.shard_stats.values().find(|stats| stats.active);
        if item.is_none() {
            return Err("no leader".into());
        }
        Ok(item.unwrap())
    }

    fn check_healthy(&self) -> Result<(), String> {
        let stats = self.get_leader_stats()?;
        if stats.mem_table_count > 1 {
            return Err(format!(
                "mem table count {} too large",
                stats.mem_table_count
            ));
        }
        if !stats.flushed {
            return Err("not initial flushed".into());
        }
        if stats.compaction_score > 2.0 {
            return Err(format!(
                "compaction score too large: {}",
                stats.compaction_score
            ));
        }
        Ok(())
    }
}
