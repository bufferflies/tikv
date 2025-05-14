// Copyright 2025 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    collections::{HashMap, HashSet},
    fs,
    net::SocketAddr,
    path::PathBuf,
    str::FromStr,
    sync::Arc,
    time::Duration,
};

use bytes::Bytes;
use cdc::{Conn, ConnId};
use grpcio::{ChannelBuilder, EnvBuilder, ServerBuilder};
use grpcio_health::{create_health, HealthService, ServingStatus};
use kvengine::dfs::{Dfs, S3Fs};
use kvproto::{cdcpb::create_change_data, raft_cmdpb::AdminRequest, tikvpb::create_tikv};
use merged_engine::{MergedEngine, MergedEngineContext};
use native_br::common::get_latest_backup_meta;
use pd_client::{PdClient, RpcClient};
use rfstore::store::ApplyContext;
use security::{SecurityConfig, SecurityManager};
use tikv_util::{error, info, thd_name};
use txn_types::TimeStamp;

use crate::{
    apply_observer::RegionEvents, provisioned::KeyspaceProvisionedService,
    scheduler::ChangefeedRequest, CdcMsg, Error, KeyspaceService, KeyspaceStates,
    ReplicationScheduler, ReplicationService, ReplicationWorkerConfig, Result,
};

#[allow(dead_code)]
pub struct ReplicationWorker {
    data_dir: PathBuf,
    ctx: MergedEngineContext,
    config: ReplicationWorkerConfig,
    merged_engine: MergedEngine,
    grpc_server: Option<grpcio::Server>,

    keyspaces: HashMap<u32, Box<dyn KeyspaceService>>,
    cdc_addrs: Arc<dashmap::DashMap<u32, String>>,
    conns: HashMap<ConnId, Conn>,

    tx: tikv_util::mpsc::Sender<CdcMsg>,
    rx: tikv_util::mpsc::Receiver<CdcMsg>,

    apply_ctx: ApplyContext,

    conn_regions: HashMap<ConnId, HashSet<u64>>,
    region_to_keyspace: HashMap<u64, u32>,

    last_update_time: TimeStamp,
    stop: bool,
}

impl ReplicationWorker {
    pub fn new(
        pd: Arc<dyn PdClient>,
        fs: Arc<S3Fs>,
        data_dir: String,
        security: SecurityConfig,
        config: ReplicationWorkerConfig,
    ) -> Result<Self> {
        let data_dir = PathBuf::from(data_dir);
        let merged_engine_dir = data_dir.join("merged_engine");
        fs::create_dir_all(merged_engine_dir.as_path()).unwrap();
        let master_key = fs.get_runtime().block_on(security.new_master_key());
        let ctx = MergedEngineContext {
            pd,
            fs,
            local_dir: merged_engine_dir,
            security_config: Arc::new(security),
            config: config.merged_engine.clone(),
            master_key,
        };

        let cluster_id = ctx.pd.get_cluster_id().unwrap();
        let runtime = ctx.fs.get_runtime();
        let cluster_backup = runtime.block_on(get_latest_backup_meta(&ctx.fs, cluster_id))?;
        let backup_ts = TimeStamp::new(cluster_backup.backup_ts);
        let merged_engine = MergedEngine::new(ctx.clone(), cluster_backup)?;
        let keyspace_ids = merged_engine.get_keyspaces();
        let mut keyspace_services = HashMap::new();
        let cdc_addrs = Arc::new(dashmap::DashMap::new());
        for keyspace_id in keyspace_ids {
            let states_bin = merged_engine.get_keyspace_states(keyspace_id).unwrap();
            let states: KeyspaceStates = serde_json::from_slice(&states_bin).unwrap();
            let cdc_addr = states.cdc_addr.clone();
            let mut task_service: Box<dyn KeyspaceService> = Box::new(
                KeyspaceProvisionedService::new(keyspace_id, &config, &ctx.security_config, states),
            );
            if let Err(err) = runtime.block_on(task_service.start()) {
                error!("keyspace {} start service error {:?}", keyspace_id, err);
                continue;
            }
            cdc_addrs.insert(keyspace_id, cdc_addr);
            keyspace_services.insert(keyspace_id, task_service);
        }
        let apply_ctx = ApplyContext::new(merged_engine.get_kv(), None);
        let (tx, rx) = tikv_util::mpsc::unbounded();
        let mut worker = Self {
            data_dir,
            config,
            ctx,
            merged_engine,
            grpc_server: None,
            keyspaces: Default::default(),
            cdc_addrs,
            conns: Default::default(),
            tx: tx.clone(),
            rx,
            apply_ctx,
            conn_regions: Default::default(),
            region_to_keyspace: Default::default(),
            last_update_time: backup_ts,
            stop: false,
        };
        let env = Arc::new(
            EnvBuilder::new()
                .cq_count(2)
                .name_prefix(thd_name!("grpc-server"))
                .build(),
        );
        let channel_args = ChannelBuilder::new(env.clone())
            .stream_initial_window_size(2 * 1024 * 1024)
            .max_concurrent_stream(1024)
            .max_receive_message_len(-1)
            .max_send_message_len(-1)
            .http2_max_ping_strikes(i32::MAX) // For pings without data from clients.
            .keepalive_time(Duration::from_secs(10))
            .keepalive_timeout(Duration::from_secs(3))
            .build_args();
        let addr = SocketAddr::from_str(&worker.config.grpc_addr).unwrap();
        let security_mgr = worker.ctx.pd.get_security_mgr();
        let service = ReplicationService::new(worker.merged_engine.get_kv(), tx.clone());
        let health_service = HealthService::default();
        health_service.set_serving_status("", ServingStatus::Serving);
        let sb = ServerBuilder::new(env)
            .channel_args(channel_args)
            .register_service(create_change_data(service.clone()))
            .register_service(create_tikv(service))
            .register_service(create_health(health_service));
        let sb = security_mgr.bind(sb, &addr.ip().to_string(), addr.port());
        let mut grpc_server = sb.build().unwrap();
        grpc_server.start();
        worker.grpc_server = Some(grpc_server);
        Ok(worker)
    }

    pub fn scheduler(&self) -> ReplicationScheduler {
        ReplicationScheduler::new(self.tx.clone())
    }

    pub fn run(&mut self) {
        let _enter = self.ctx.fs.get_runtime().enter();
        loop {
            let res = self.rx.recv_timeout(Duration::from_millis(100));
            match res {
                Ok(msg) => {
                    self.handle_msg(msg);
                    while let Ok(msg) = self.rx.try_recv() {
                        self.handle_msg(msg);
                    }
                }
                Err(err) => {
                    if err.is_disconnected() {
                        return;
                    }
                }
            }
            if self.stop {
                info!("replication_worker stopped");
                return;
            }
        }
    }

    fn handle_msg(&mut self, msg: CdcMsg) {
        match msg {
            CdcMsg::AddKeyspace {
                keyspace_id,
                pd_url,
                cdc_addr,
                cb,
            } => {
                self.handle_add_keyspace(keyspace_id, pd_url, cdc_addr, cb);
            }
            CdcMsg::GetKeyspaces { cb } => {
                cb(self.keyspaces.keys().cloned().collect());
            }
            CdcMsg::AddKeyspaceResult {
                keyspace_id,
                result,
                cb,
            } => {
                let res = self.handle_add_keyspace_result(keyspace_id, result);
                cb(res);
            }
            CdcMsg::RemoveKeyspace { keyspace_id, cb } => {
                self.handle_remove_keyspace_service(keyspace_id, cb);
            }
            CdcMsg::NewTask {
                keyspace_id,
                request,
            } => self.handle_new_task(keyspace_id, request),
            CdcMsg::Applied {
                region_id,
                region_events,
            } => {
                let res = self.handle_applied(region_id, region_events);
                self.handle_result(res, "applied");
            }
            CdcMsg::AppliedAdmin {
                region_id,
                region_version,
                admin,
            } => {
                let res = self.handle_applied_admin(region_id, region_version, admin);
                self.handle_result(res, "applied_admin");
            }
            CdcMsg::RemoveTask {
                keyspace_id,
                change_feed_id,
            } => {
                info!("remove task {} {}", keyspace_id, change_feed_id);
            }
            CdcMsg::Stop => {
                self.stop = true;
            }
        }
    }

    fn handle_add_keyspace(
        &self,
        keyspace_id: u32,
        pd_url: String,
        cdc_addr: String,
        cb: Box<dyn FnOnce(Result<()>) + Send>,
    ) {
        if self.keyspaces.contains_key(&keyspace_id) {
            cb(Err(Error::OtherError("keyspace already exists".into())));
            return;
        }
        let mut task_service: Box<dyn KeyspaceService> = {
            let mut states = KeyspaceStates::default();
            states.pd_url = pd_url;
            states.cdc_addr = cdc_addr;
            Box::new(KeyspaceProvisionedService::new(
                keyspace_id,
                &self.config,
                &self.ctx.security_config,
                states,
            ))
        };
        let scheduler = self.scheduler();
        let merged_engine_ctx = self.merged_engine.ctx.clone();
        let kv = self.merged_engine.kv.clone();
        let recover_handler = self.merged_engine.recover_handler.clone();
        tokio::spawn(async move {
            let res = task_service.start().await;
            if res.is_err() {
                cb(res);
                return;
            }
            // When new keyspace added, loading shards takes long time, we need a dedicated
            // thread to do it.
            std::thread::spawn(move || {
                let mut states = HashMap::new();
                states.insert(keyspace_id, Bytes::new());
                let res =
                    MergedEngine::load_shards(&merged_engine_ctx, &kv, recover_handler, &states)
                        .map_err(|e| Error::from(e));
                scheduler.schedule(CdcMsg::AddKeyspaceResult {
                    keyspace_id,
                    result: res.map(|_| task_service),
                    cb,
                });
            });
        });
    }

    fn handle_add_keyspace_result(
        &mut self,
        keyspace_id: u32,
        result: Result<Box<dyn KeyspaceService>>,
    ) -> Result<()> {
        let svc = result?;
        let states = svc.get_states().marshal();
        self.keyspaces.insert(keyspace_id, svc);
        self.merged_engine
            .set_keyspace_states(keyspace_id, states)?;
        Ok(())
    }

    fn handle_remove_keyspace_service(
        &mut self,
        keyspace_id: u32,
        cb: Box<dyn FnOnce(Result<()>) + Send>,
    ) {
        self.cdc_addrs.remove(&keyspace_id);
        let runtime = self.ctx.fs.get_runtime();
        let Some(mut svc) = self.keyspaces.remove(&keyspace_id) else {
            cb(Err(Error::OtherError("keyspace not found".into())));
            return;
        };
        let keyspace_regions = self.merged_engine.get_keyspace_regions(keyspace_id);
        let kv = self.merged_engine.get_kv();
        keyspace_regions.iter().for_each(|region_id| {
            self.region_to_keyspace.remove(region_id);
            kv.remove_shard(*region_id);
        });
        self.merged_engine.remove_keyspace(keyspace_id);
        runtime.spawn(async move {
            let res = svc.destroy().await;
            cb(res);
        });
    }

    fn handle_new_task(&mut self, keyspace_id: u32, request: ChangefeedRequest) {
        // TODO: implement this
        info!("new task {} {:?}", keyspace_id, request);
    }

    fn handle_result(&mut self, res: Result<()>, tag: &str) {
        if let Err(err) = res {
            error!("handle {} error {:?}", tag, err);
        }
    }

    fn handle_applied(&mut self, region_id: u64, region_events: RegionEvents) -> Result<()> {
        // TODO: implement this
        info!(
            "applied {} with {} events",
            region_id,
            region_events.events.len()
        );
        Ok(())
    }

    fn handle_applied_admin(
        &mut self,
        region_id: u64,
        region_version: u64,
        admin: AdminRequest,
    ) -> Result<()> {
        // TODO: implement this
        info!("applied admin {} {} {:?}", region_id, region_version, admin);
        Ok(())
    }
}

pub(crate) async fn new_keyspace_pd_client(
    pd_url: String,
    sec_conf: &SecurityConfig,
) -> Arc<RpcClient> {
    let sec_mgr = Arc::new(SecurityManager::new(sec_conf).unwrap());
    Arc::new(
        RpcClient::new_async(&pd_client::Config::new(vec![pd_url]), None, sec_mgr)
            .await
            .unwrap(),
    )
}
