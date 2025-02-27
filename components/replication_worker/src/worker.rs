// Copyright 2025 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    collections::{HashMap, HashSet},
    fs,
    net::SocketAddr,
    path::PathBuf,
    str::FromStr,
    sync::{Arc, RwLock},
    time::Duration,
};

use cdc::{Conn, ConnId};
use grpcio::{ChannelBuilder, EnvBuilder, ServerBuilder};
use grpcio_health::{create_health, HealthService, ServingStatus};
use kvengine::dfs::{Dfs, S3Fs};
use kvproto::{
    cdcpb::{create_change_data, Event},
    tikvpb::create_tikv,
};
use merged_engine::{MergedEngine, MergedEngineContext};
use native_br::common::get_latest_backup_meta;
use pd_client::PdClient;
use rfstore::store::ApplyContext;
use security::SecurityConfig;
use tikv_util::{error, info, thd_name};
use txn_types::TimeStamp;

use crate::{
    scheduler::ChangefeedRequest, CdcMsg, ReplicationScheduler, ReplicationService,
    ReplicationWorkerConfig,
};

#[allow(dead_code)]
#[derive(Default)]
struct RegionChange {
    keyspace_id: u32,
    requests: HashMap<u64, ConnId>,
    events: Vec<Event>,
}

#[allow(dead_code)]
pub struct ReplicationWorker {
    data_dir: PathBuf,
    ctx: MergedEngineContext,
    config: ReplicationWorkerConfig,
    merged_engine: MergedEngine,
    grpc_server: Option<grpcio::Server>,

    conns: HashMap<ConnId, Conn>,

    tx: tikv_util::mpsc::Sender<CdcMsg>,
    rx: tikv_util::mpsc::Receiver<CdcMsg>,

    apply_ctx: ApplyContext,

    region_changes: HashMap<u64, RegionChange>,
    conn_regions: HashMap<ConnId, HashSet<u64>>,
    region_to_keyspace: HashMap<u64, u32>,
    registered: Arc<RwLock<HashSet<u64>>>,

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
    ) -> Option<Self> {
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
        let cluster_backup = runtime
            .block_on(get_latest_backup_meta(&ctx.fs, cluster_id))
            .map_err(|e| {
                error!("get latest backup meta failed: {}", e);
            })
            .ok()?;
        let backup_ts = TimeStamp::new(cluster_backup.backup_ts);
        let merged_engine = MergedEngine::new(ctx.clone(), cluster_backup).ok()?;
        let apply_ctx = ApplyContext::new(merged_engine.get_kv(), None);
        let (tx, rx) = tikv_util::mpsc::unbounded();
        let registered = Arc::new(RwLock::new(HashSet::new()));
        let mut worker = Self {
            data_dir,
            config,
            ctx,
            merged_engine,
            grpc_server: None,
            conns: Default::default(),
            tx: tx.clone(),
            rx,
            apply_ctx,
            region_changes: Default::default(),
            conn_regions: Default::default(),
            registered,
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
        let addr = SocketAddr::from_str(&worker.config.grpc_addr)
            .map_err(|e| {
                error!("parse grpc addr failed: {}", e);
            })
            .ok()?;
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
        let mut grpc_server = sb
            .build()
            .map_err(|e| {
                error!("build grpc server failed: {}", e);
            })
            .ok()?;
        grpc_server.start();
        worker.grpc_server = Some(grpc_server);
        Some(worker)
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
                    self.handle_msg(msg).unwrap();
                    while let Ok(msg) = self.rx.try_recv() {
                        self.handle_msg(msg).unwrap();
                    }
                }
                Err(err) => {
                    if err.is_disconnected() {
                        return;
                    }
                }
            }
            if self.stop {
                return;
            }
        }
    }

    fn handle_msg(&mut self, msg: CdcMsg) -> cdc::Result<()> {
        match msg {
            CdcMsg::NewTask {
                keyspace_id,
                request,
            } => self.handle_new_task(keyspace_id, request),
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
        Ok(())
    }

    fn handle_new_task(&mut self, keyspace_id: u32, request: ChangefeedRequest) {
        // TODO: implement this
        info!("new task {} {:?}", keyspace_id, request);
    }
}
