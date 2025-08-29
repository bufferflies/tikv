// Copyright 2025 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    collections::{hash_map::Entry, HashMap as StdHashMap},
    fs, mem,
    net::SocketAddr,
    ops::Deref,
    path::PathBuf,
    str::FromStr,
    sync::Arc,
    time::Duration,
};

use api_version::ApiV2;
use bytes::Bytes;
use cdc::{CdcEvent, Conn, ConnId};
use collections::{HashMap, HashSet};
use futures::executor::block_on;
use grpcio::{ChannelBuilder, EnvBuilder, ServerBuilder};
use grpcio_health::{create_health, HealthService, ServingStatus};
use http::Request;
use hyper::{http, Body, StatusCode};
use kvengine::{
    dfs::{Dfs, S3Fs},
    table::InnerKey,
    Engine, IdVer, ShardMeta, ShardTag, SnapAccess, UserMeta, LOCK_CF, WRITE_CF,
};
use kvproto::{
    cdcpb,
    cdcpb::{
        create_change_data, ChangeDataRequest, Event, EventLogType, EventRow, EventRowOpType,
        ResolvedTs,
    },
    metapb,
    pdpb::StoreStats,
    raft_cmdpb::AdminRequest,
    tikvpb::create_tikv,
};
use log_wrappers::Value as LogValue;
use merged_engine::{MergedEngine, MergedEngineContext};
use native_br::common::{
    collect_wal_chunks_with_retry, get_latest_backup_meta, CollectWalChunksContext,
};
use pd_client::{PdClient, RegionStat};
use rfengine::{assemble_wal_chunks, RfEngine, TRUNCATE_ALL_INDEX};
use rfstore::store::ApplyContext;
use security::{HttpClient, SecurityConfig};
use tikv_util::{
    box_err, codec, debug, error, info,
    mpsc::{Receiver, Sender},
    thd_name,
    time::Instant,
    warn,
};
use txn_types::{LockType, TimeStamp};

use crate::{
    apply_observer::{is_index_key, CdcApplyObserver, RegionEvents},
    delegate::{RegionDelegate, RegionResolver, RequestId, RequestKey},
    kube::{KeyspaceKubeService, KubeApi},
    provisioned::KeyspaceProvisionedService,
    scheduler::get_cdc_status,
    ticdc_util::TiCdcError,
    util::{
        build_request_range_for_keyspace, keyspace_prefix_len, post_to_ticdc, DISPATCH_CDC_TIMEOUT,
    },
    CdcMsg, Deregister, Error,
    Error::StoreTimeout,
    KeyspaceService, KeyspaceStates, ReplicationScheduler, ReplicationService,
    ReplicationWorkerConfig, Result,
};

const MAX_INITIALIZE_SCAN_BATCH_BYTES: usize = 1024 * 1024;
const FETCH_WAL_TIMEOUT: Duration = Duration::from_secs(30);
const UPDATE_STORES_TIMEOUT: Duration = Duration::from_secs(120);

pub struct ReplicationWorker {
    ctx: MergedEngineContext,
    http_client: Arc<HttpClient>,
    config: ReplicationWorkerConfig,
    merged_engine: MergedEngine,
    grpc_server: Option<grpcio::Server>,
    kube_api: Option<Arc<KubeApi>>,
    runtime: tokio::runtime::Runtime,

    keyspaces: HashMap<u32, Box<dyn KeyspaceService>>,
    cdc_addrs: Arc<dashmap::DashMap<u32, String>>,

    conns: HashMap<ConnId, Conn>,

    tx: Sender<CdcMsg>,
    rx: Receiver<CdcMsg>,

    apply_ctx: ApplyContext,

    region_delegates: HashMap<u64 /* region_id */, RegionDelegate>,
    conn_regions: HashMap<ConnId, HashSet<u64 /* region_id */>>,
    region_to_keyspace: HashMap<u64 /* region_id */, u32 /* keyspace_id */>,

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
        let http_client = ctx
            .pd
            .get_security_mgr()
            .http_client(hyper::Client::builder())
            .unwrap();
        let cluster_id = ctx.pd.get_cluster_id().unwrap();
        let runtime = ctx.fs.get_runtime();
        let cluster_backup = runtime.block_on(get_latest_backup_meta(&ctx.fs, cluster_id))?;
        let backup_ts = TimeStamp::new(cluster_backup.backup_ts);
        let merged_engine = MergedEngine::new(ctx.clone(), cluster_backup)?;
        let keyspace_ids = merged_engine.get_keyspaces();
        let mut keyspace_services = HashMap::default();
        let cdc_addrs = Arc::new(dashmap::DashMap::new());
        let kube_api = if config.is_kube_mode() {
            info!("init k8s api");
            let api = runtime.block_on(KubeApi::new(
                config.pd_sts_name.clone(),
                config.cdc_sts_name.clone(),
                config.namespace.clone(),
            ))?;
            Some(Arc::new(api))
        } else {
            None
        };
        for keyspace_id in keyspace_ids {
            let states_bin = merged_engine.get_keyspace_states(keyspace_id).unwrap();
            let states: KeyspaceStates = serde_json::from_slice(&states_bin).unwrap();
            let cdc_addr = states.cdc_addr.clone();
            let mut task_service: Box<dyn KeyspaceService> = if states.is_provisioned() {
                Box::new(KeyspaceProvisionedService::new(
                    keyspace_id,
                    &config,
                    &ctx.security_config,
                    states,
                ))
            } else {
                Box::new(KeyspaceKubeService::new(
                    keyspace_id,
                    kube_api.clone().unwrap(),
                    &config,
                    &ctx.security_config,
                ))
            };
            if let Err(err) = runtime.block_on(task_service.start()) {
                error!("keyspace {} start service error {:?}", keyspace_id, err);
                continue;
            }
            cdc_addrs.insert(keyspace_id, cdc_addr);
            keyspace_services.insert(keyspace_id, task_service);
        }
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap();
        let interval = config.report_region_interval.0;
        for (&keyspace_id, ks_svc) in &keyspace_services {
            let raft = merged_engine.get_raft();
            let kv = merged_engine.get_kv();
            let rep_pd_cli = ks_svc.get_pd_client();
            runtime.spawn(async move {
                Self::report_regions_loop(keyspace_id, raft, kv, rep_pd_cli, interval).await;
            });
        }
        let mut apply_ctx =
            ApplyContext::new(merged_engine.get_kv(), Some(merged_engine.get_router()));
        let (tx, rx) = tikv_util::mpsc::unbounded();
        let apply_observer =
            CdcApplyObserver::new(merged_engine.get_kv(), tx.clone(), runtime.handle().clone());
        apply_ctx.set_apply_observer(Box::new(apply_observer));
        let mut worker = Self {
            config,
            ctx,
            http_client: Arc::new(http_client),
            merged_engine,
            runtime,
            grpc_server: None,
            kube_api,
            keyspaces: keyspace_services,
            cdc_addrs,
            conns: Default::default(),
            tx: tx.clone(),
            rx,
            apply_ctx,
            region_delegates: Default::default(),
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

    fn get_region_tag(&self, region_id: u64) -> ShardTag {
        ShardTag::new(self.ctx.config.merged_store_id, IdVer::new(region_id, 0))
    }

    pub fn scheduler(&self) -> ReplicationScheduler {
        ReplicationScheduler::new(
            self.tx.clone(),
            self.cdc_addrs.clone(),
            self.http_client.clone(),
        )
    }

    pub fn run(&mut self) {
        let _enter = self.ctx.fs.get_runtime().enter();
        info!("replication worker started");
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
            if let Err(err) = self.send_resolved_ts() {
                error!("send resolved ts error"; "err" => ?err);
            }
            if let Err(err) = self.maybe_update_merged_engine() {
                error!("update merged engine error"; "err" => ?err);
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
            CdcMsg::LoadKeyspaceShards {
                keyspace_id,
                task_service,
                cb,
            } => {
                let res = self.handle_load_keyspace_shards(keyspace_id, task_service);
                cb(res);
            }
            CdcMsg::LoadKeyspaceShardMetas { keyspace_id, cb } => {
                let res = self.handle_load_keyspace_shard_metas(keyspace_id);
                cb(res);
            }
            CdcMsg::RemoveKeyspace { keyspace_id, cb } => {
                self.handle_remove_keyspace_service(keyspace_id, cb);
            }
            CdcMsg::NewTask {
                keyspace_id,
                changefeed_id,
                body,
                cb,
            } => {
                self.handle_new_task(keyspace_id, changefeed_id, body, cb);
            }
            CdcMsg::OpenConn(conn) => self.handle_open_conn(conn),
            CdcMsg::Register { request, conn_id } => {
                let res = self.handle_register(request, conn_id);
                self.handle_result(res, "register");
            }
            CdcMsg::RegisterResult {
                event,
                conn_id,
                initialized,
            } => {
                let res = self.handle_register_result(event, conn_id, initialized);
                self.handle_result(res, "register_result");
            }
            CdcMsg::ScanLocksResult { region_id, locks } => {
                let res = self.handle_scan_locks_result(region_id, locks);
                self.handle_result(res, "scan_locks_result");
            }
            CdcMsg::Deregister(deregister) => {
                self.handle_deregister(deregister);
            }
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
                cb,
            } => {
                let res = self.handle_remove_task(keyspace_id, change_feed_id);
                cb(res);
            }
            CdcMsg::Stop => {
                self.stop = true;
            }
        }
    }

    fn handle_new_task(
        &mut self,
        keyspace_id: u32,
        changefeed_id: String,
        body: Bytes,
        cb: Box<dyn FnOnce(Result<(StatusCode, Bytes)>) + Send>,
    ) {
        let tag = format!("{keyspace_id}:new_task");
        let body_string = String::from_utf8_lossy(&body).to_string();
        info!("handle_new_task"; "keyspace" => keyspace_id, "req" => &body_string);
        #[allow(clippy::map_entry)]
        if !self.keyspaces.contains_key(&keyspace_id) {
            cb(Err(Error::OtherError("keyspace not found".into())));
            return;
        }
        let task_svc = self.keyspaces.get_mut(&keyspace_id).unwrap();
        let pd_client = task_svc.get_pd_client();
        Self::report_store_to_pd(&pd_client, self.ctx.config.merged_store_id);
        let keyspace_region_ids = self.merged_engine.get_keyspace_regions(keyspace_id);
        let raft = self.merged_engine.get_raft();
        for region_id in keyspace_region_ids {
            Self::report_region_to_rep_pd_by_id(&raft, &pd_client, region_id);
        }
        let states = task_svc.get_states_mut();
        let req_body = match states.feeds.entry(changefeed_id) {
            Entry::Vacant(e) => {
                // Consider as a retry request.
                // TODO: verify the request parameter.
                e.insert(body_string);
                self.merged_engine
                    .set_keyspace_states(keyspace_id, states.marshal())
                    .unwrap();
                body
            }
            Entry::Occupied(e) => {
                // Use the saved body, to ensure that the request between replication worker &
                // TiCDC are the same.
                Bytes::copy_from_slice(e.get().as_bytes())
            }
        };

        // Still dispatch to TiCDC, as the previous request will failed.
        // TODO: Remove the changefeed in replication worker on error ?
        let sec_mgr = self.ctx.pd.get_security_mgr();
        let client = sec_mgr.http_client(hyper::Client::builder()).unwrap();
        let cdc_addr = states.cdc_addr.clone();
        self.cdc_addrs.insert(keyspace_id, cdc_addr.clone());
        tokio::spawn(async move {
            fn is_err_retryable(err: &TiCdcError) -> bool {
                matches!(err, TiCdcError::ServerIsNotReady(_))
            }
            let new_cdc_task_uri = sec_mgr
                .build_uri(format!("{}/api/v2/changefeeds", &cdc_addr))
                .unwrap();
            let res =
                post_to_ticdc(&tag, &client, &new_cdc_task_uri, req_body, is_err_retryable).await;
            cb(res);
        });
    }

    fn handle_open_conn(&mut self, conn: Conn) {
        let conn_id = conn.get_id();
        self.conns.insert(conn_id, conn);
    }

    fn handle_register(&mut self, request: ChangeDataRequest, conn_id: ConnId) -> Result<()> {
        let region_id = request.region_id;
        let Some(shard) = self.merged_engine.get_kv().get_shard(request.region_id) else {
            self.send_region_not_found(conn_id, &request);
            return Ok(());
        };
        if shard.ver != request.get_region_epoch().get_version() {
            self.send_epoch_not_match(conn_id, &request);
            return Ok(());
        }
        info!("cdc register"; "req" => ?request, "conn" => ?conn_id);
        let delegate = self
            .region_delegates
            .entry(region_id)
            .or_insert_with(|| RegionDelegate::new(shard.keyspace_id, region_id));

        let snap_access = shard.new_snap_access();
        if delegate.resolver.is_none() {
            delegate.resolver = Some(RegionResolver::new_pending());
            let snap_access = snap_access.clone();
            let mut locks_handler = ScanLocksHandler::new(snap_access, self.tx.clone());
            self.runtime.spawn_blocking(move || {
                locks_handler.scan_locks();
            });
        }

        delegate.requests.add(&request, conn_id);
        self.conn_regions
            .entry(conn_id)
            .or_default()
            .insert(region_id);
        let mut register_handler =
            RegisterHandler::new(conn_id, &request, snap_access, self.tx.clone());
        self.runtime
            .spawn(async move { register_handler.handle_register().await });
        Ok(())
    }

    fn handle_register_result(
        &mut self,
        event: Event,
        conn_id: ConnId,
        initialized: bool,
    ) -> Result<()> {
        let request_id: RequestId = event.get_request_id().into();
        let region_id = event.get_region_id();
        let tag = self.get_region_tag(region_id);
        let Some(delegate) = self.region_delegates.get_mut(&region_id) else {
            warn!("{} handle_register_result: region delegate not found", tag; "conn" => ?conn_id, "request" => %request_id);
            return Ok(());
        };
        let Some(request_info) = delegate
            .requests
            .get_mut(&RequestKey::new(conn_id, request_id))
        else {
            warn!("{} handle_register_result: request not found", tag; "conn" => ?conn_id, "request" => %request_id);
            return Ok(());
        };
        request_info.initialized = initialized;
        let Some(conn) = self.conns.get(&conn_id) else {
            warn!("{} handle_register_result: conn not found", tag; "conn" => ?conn_id, "request" => %request_id);
            return Ok(());
        };
        debug!("{} handle_register_result: send event {:?}", tag, event);
        let sink = conn.get_sink();
        sink.unbounded_send(CdcEvent::Event(event), false)
            .map_err(|e| cdc::Error::from(e))?;
        if request_info.initialized {
            let pending_events = mem::take(&mut request_info.pending_events);
            let pending_events_count = pending_events.len();
            for pending_event in pending_events {
                debug!(
                    "{} handle_register_result: send pending event {:?}",
                    tag, pending_event
                );
                sink.unbounded_send(CdcEvent::Event(pending_event), false)
                    .map_err(|e| cdc::Error::from(e))?;
            }
            info!("{} handle_register_result: initialized, pending_events: {}", tag, pending_events_count;
                 "conn" => ?conn_id, "request" => %request_id);
        }
        Ok(())
    }

    fn handle_scan_locks_result(
        &mut self,
        region_id: u64,
        locks: Result<Vec<(Vec<u8>, TimeStamp)>>,
    ) -> Result<()> {
        let Some(delegate) = self.region_delegates.get_mut(&region_id) else {
            warn!(
                "{} handle_scan_locks_result: region delegate not found",
                region_id
            );
            return Ok(());
        };

        match locks {
            Ok(locks) => {
                delegate.handle_scan_locks(locks);
            }
            Err(err) => {
                error!("{} handle_scan_locks_result: scan locks error", region_id; "err" => ?err);
                delegate.resolver = None;

                let mut cdc_err = cdcpb::Error::default();
                cdc_err
                    .mut_server_is_busy()
                    .set_reason(format!("scan locks failed: {:?}", err));
                delegate.broadcast_error(cdc_err, &self.conns);
            }
        }

        Ok(())
    }

    fn send_error_event(&self, conn_id: ConnId, request: &ChangeDataRequest, error: cdcpb::Error) {
        let Some(conn) = self.conns.get(&conn_id) else {
            warn!("send_error_event: conn not found"; "conn" => ?conn_id);
            return;
        };

        let mut event = Event::new();
        event.set_region_id(request.region_id);
        event.set_request_id(request.request_id);
        event.set_error(error);
        if let Err(err) = conn.get_sink().unbounded_send(CdcEvent::Event(event), true) {
            warn!("send_error_event: failed"; "conn" => ?conn_id, "err" => ?err);
        }
    }

    fn send_region_not_found(&self, conn_id: ConnId, request: &ChangeDataRequest) {
        let mut error = cdcpb::Error::new();
        error
            .mut_region_not_found()
            .set_region_id(request.region_id);
        self.send_error_event(conn_id, request, error);
    }

    fn send_epoch_not_match(&self, conn_id: ConnId, request: &ChangeDataRequest) {
        let mut error = cdcpb::Error::new();
        let epoch_not_match = error.mut_epoch_not_match();
        let rep_region =
            Self::get_region_for_rep(&self.merged_engine.get_raft(), request.region_id);
        if let Some(rep_region) = rep_region {
            epoch_not_match.mut_current_regions().push(rep_region);
        }
        self.send_error_event(conn_id, request, error);
    }

    fn handle_deregister(&mut self, deregister: Deregister) {
        match deregister {
            Deregister::Conn(conn_id) => self.handle_deregister_conn(conn_id),
            Deregister::Request {
                conn_id,
                request_id,
            } => self.handle_deregister_request(conn_id, request_id),
            Deregister::Region {
                conn_id,
                request_id,
                region_id,
            } => self.handle_deregister_region(conn_id, request_id, region_id),
        }
    }

    fn handle_deregister_conn(&mut self, conn_id: ConnId) {
        info!("deregister conn"; "conn" => ?conn_id);
        self.conns.remove(&conn_id);
        if let Some(conn_regions) = self.conn_regions.remove(&conn_id) {
            for region_id in conn_regions {
                if let Some(delegate) = self.region_delegates.get_mut(&region_id) {
                    delegate.requests.retain(|k, _| k.conn_id != conn_id);
                    if delegate.requests.is_empty() {
                        self.remove_region(region_id);
                    }
                }
            }
        }
    }

    fn handle_deregister_request(&mut self, conn_id: ConnId, request_id: RequestId) {
        info!("deregister request"; "conn" => ?conn_id, "request" => %request_id);
        let mut remove_regions = vec![];
        if let Some(conn_regions) = self.conn_regions.get_mut(&conn_id) {
            let sink = self.conns.get(&conn_id).map(|c| c.get_sink());
            for &region_id in conn_regions.iter() {
                if let Some(delegate) = self.region_delegates.get_mut(&region_id) {
                    delegate.unsubscribe(conn_id, request_id, sink);
                    if delegate.requests.is_empty() {
                        // To work around mutable borrow limitation.
                        remove_regions.push(region_id);
                    }
                }
            }
        }
        for region_id in remove_regions {
            self.remove_region(region_id);
        }
    }

    fn handle_deregister_region(&mut self, conn_id: ConnId, request_id: RequestId, region_id: u64) {
        info!("deregister request"; "conn" => ?conn_id, "request" => %request_id, "region" => region_id);
        if let Some(delegate) = self.region_delegates.get_mut(&region_id) {
            let sink = self.conns.get(&conn_id).map(|c| c.get_sink());
            delegate.unsubscribe(conn_id, request_id, sink);
            if delegate.requests.is_empty() {
                self.remove_region(region_id);
            }
        }
    }

    fn handle_result(&mut self, res: Result<()>, tag: &str) {
        if let Err(err) = res {
            error!("handle {} error {:?}", tag, err);
        }
    }

    fn get_keyspace_id(&mut self, region_id: u64) -> Option<u32> {
        Some(match self.region_to_keyspace.entry(region_id) {
            Entry::Occupied(e) => *e.get(),
            Entry::Vacant(e) => {
                let keyspace_id = self
                    .merged_engine
                    .get_region_progress(region_id)?
                    .keyspace_id;
                *e.insert(keyspace_id)
            }
        })
    }

    fn send_resolved_ts(&mut self) -> Result<()> {
        info!("send_resolved_ts"; "last_update_time" => self.last_update_time);
        for (&region_id, delegate) in &mut self.region_delegates {
            let Some(ts) = delegate
                .resolver
                .as_mut()
                .and_then(|r| r.resolve(self.last_update_time))
            else {
                continue;
            };
            debug!("{} resolved ts: {}", region_id, ts);
            for (req_key, req_info) in delegate.requests.iter_mut() {
                if req_info.resolved_ts != ts && req_info.initialized {
                    let mut resolved_ts = ResolvedTs::default();
                    resolved_ts.set_regions(vec![region_id]);
                    resolved_ts.set_request_id(req_key.request_id.into_inner());
                    resolved_ts.set_ts(ts.into_inner());
                    debug!("{} send resolved_ts {}", region_id, ts; "request" => %req_key.request_id);
                    let conn = self.conns.get(&req_key.conn_id).unwrap();
                    conn.get_sink()
                        .unbounded_send(CdcEvent::ResolvedTs(resolved_ts), false)
                        .map_err(|e| cdc::Error::from(e))?;
                    req_info.resolved_ts = ts;
                }
            }
        }
        Ok(())
    }

    fn maybe_update_merged_engine(&mut self) -> Result<()> {
        if TimeStamp::physical_now().saturating_sub(self.last_update_time.physical())
            < self.config.sync_interval.as_millis()
        {
            return Ok(());
        }
        let new_timestamp = self.update_stores_with_retry(UPDATE_STORES_TIMEOUT)?;
        self.merged_engine.sync_merged(&mut self.apply_ctx)?;
        self.apply_ctx.flush_observer();
        self.last_update_time = new_timestamp;
        Ok(())
    }

    async fn report_regions_loop(
        keyspace_id: u32,
        raft: RfEngine,
        kv: Engine,
        rep_pd_cli: Arc<dyn PdClient>,
        interval: Duration,
    ) {
        // run a loop to report regions in case that the rep pd is restarted and lost
        // the region leader.
        loop {
            let start = Instant::now();
            let keyspace_shards = kv.get_keyspace_shards(keyspace_id).unwrap_or_default();
            if keyspace_shards.is_empty() {
                // keyspace shards not found means the keyspace is removed.
                return;
            }
            for region_id in keyspace_shards {
                let Some(mut region_local_state) =
                    rfstore::store::load_last_peer_state(&raft, region_id)
                else {
                    // The region may have been merged.
                    continue;
                };
                let mut region = region_local_state.take_region();
                region.set_start_key(Self::trim_keyspace_prefix(region.get_start_key()));
                region.set_end_key(Self::trim_keyspace_prefix(region.get_end_key()));
                let leader = region.get_peers()[0].clone();
                let mut stats = RegionStat::default();
                stats.approximate_kv_size = 100 * 1024 * 1024;
                stats.approximate_keys = 1000000;
                stats.approximate_size = 100 * 1024 * 1024;
                let res = rep_pd_cli
                    .region_heartbeat(1, region, leader, stats, None)
                    .await;
                if let Err(err) = res {
                    if kv.get_keyspace_shards(keyspace_id).is_none() {
                        // The keyspace has been removed.
                        return;
                    }
                    warn!("failed to heartbeat region {} {:?}", region_id, err);
                    continue;
                }
            }
            let elapsed = start.saturating_elapsed();
            tokio::time::sleep(interval.saturating_sub(elapsed)).await;
        }
    }

    fn update_stores_with_retry(&mut self, timeout: Duration) -> Result<TimeStamp> {
        let mut last_err: Option<Error> = None;
        let start_time = Instant::now_coarse();
        while start_time.saturating_elapsed() < timeout {
            match self.update_stores() {
                Ok(ts) => return Ok(ts),
                Err(err) => {
                    last_err = Some(err);
                    std::thread::sleep(Duration::from_secs(1));
                }
            }
        }
        Err(last_err.unwrap())
    }

    fn update_stores(&mut self) -> Result<TimeStamp> {
        let _enter = self.ctx.fs.get_runtime().enter();
        let stores = native_br::common::get_all_stores_except_tiflash(self.ctx.pd.as_ref())?;
        let ts = self.ctx.pd.get_min_tso()?;
        let mut errors = vec![];
        for store in stores {
            let store_id = store.id;
            if let Err(err) = self.update_store_wal(store) {
                warn!("failed to update store {}, err {:?}", store_id, err);
                errors.push(err);
            }
        }
        if errors.len() <= 1 {
            return Ok(ts);
        }
        Err(errors.pop().unwrap())
    }

    fn update_store_wal(&mut self, store: metapb::Store) -> Result<()> {
        let security_mgr = self.ctx.pd.get_security_mgr();
        let store_id = store.get_id();
        let store_progress = self.merged_engine.get_or_insert_store_progress(store_id);
        let mut epoch = store_progress.epoch;
        let mut start_off = store_progress.offset;
        loop {
            let uri = security_mgr
                .build_uri(format!(
                    "{}/rfengine/wal_chunk?epoch_id={}&start_off={}&end_off=0",
                    &store.status_address, epoch, start_off
                ))
                .unwrap();
            let req = http::Request::get(uri.clone())
                .body(hyper::Body::empty())
                .unwrap();
            let http_client = self.http_client.clone();
            let (status, data) = block_on(Self::send_request_to_store(
                req,
                &http_client,
                FETCH_WAL_TIMEOUT,
            ))?;
            if status == StatusCode::GONE {
                self.update_store_wal_from_s3(store.get_id(), epoch, start_off)?;
                epoch += 1;
                start_off = 0;
                continue;
            }
            if !status.is_success() {
                let err_str = String::from_utf8_lossy(&data);
                return Err(box_err!("{}", err_str));
            }
            let data_len = data.len();
            self.merged_engine
                .update_wal(store_id, epoch, start_off, data.clone())?;
            if status == StatusCode::PARTIAL_CONTENT {
                self.merged_engine
                    .rotate_wal(store_id, epoch, start_off + data_len as u64)?;
                epoch += 1;
                start_off = 0;
                continue;
            }
            return Ok(());
        }
    }

    pub async fn send_request_to_store(
        req: Request<Body>,
        client: &HttpClient,
        timeout: Duration,
    ) -> Result<(StatusCode, Bytes)> {
        debug_assert!(!timeout.is_zero());
        let uri_str = format!("{}", req.uri());
        let resp_res = tokio::time::timeout(timeout, client.request(req))
            .await
            .map_err(|_| StoreTimeout(format!("send request to {uri_str}")))?;
        let resp = resp_res?;
        let status = resp.status();
        let body = tokio::time::timeout(timeout, hyper::body::to_bytes(resp.into_body()))
            .await
            .map_err(|_| StoreTimeout(format!("read response from {uri_str}")))??;
        Ok((status, body))
    }

    fn update_store_wal_from_s3(
        &mut self,
        store_id: u64,
        epoch_id: u32,
        start_off: u64,
    ) -> Result<()> {
        info!("update store wal from s3";
            "store_id" => store_id, "epoch_id" => epoch_id, "start_off" => start_off);
        let collect_ctx = CollectWalChunksContext {
            pd_client: self.ctx.pd.clone(),
            dfs: self.ctx.fs.clone(),
            store_id,
            complete_wal_chunks: true,
            fetch_wal_timeout: FETCH_WAL_TIMEOUT,
        };
        let tag = format!("{}:{}", store_id, epoch_id);
        // there is no online chunk for this epoch.
        let (chunks, _) =
            collect_wal_chunks_with_retry(&tag, &collect_ctx, epoch_id, epoch_id + 1, 0)
                .map_err(|e| Error::from(e))?;
        let wal_data = assemble_wal_chunks(chunks)?.freeze();
        let end_off = wal_data.len() as u64;
        let remained_wal_data = wal_data.slice((start_off as usize)..);
        self.merged_engine
            .update_wal(store_id, epoch_id, start_off, remained_wal_data)?;
        self.merged_engine.rotate_wal(store_id, epoch_id, end_off)?;
        Ok(())
    }

    fn report_store_to_pd(pd_client: &Arc<dyn PdClient>, store_id: u64) {
        let mut store_stat = StoreStats::new();
        store_stat.set_store_id(store_id);
        store_stat.set_region_count(0);
        store_stat.set_capacity(100 * 1024 * 1024 * 1024);
        store_stat.set_used_size(50 * 1024 * 1024 * 1024);
        let resp = pd_client.store_heartbeat(store_stat, None, None);

        let new_pd_clinet = pd_client.clone();
        tokio::spawn(async move {
            if let Err(err) = resp.await {
                warn!("store heartbeat failed"; "err" => ?err);
                tokio::time::sleep(Duration::from_secs(10)).await;
                Self::report_store_to_pd(&new_pd_clinet, store_id);
            }
        });
    }

    fn report_region_to_rep_pd_by_id(
        raft: &RfEngine,
        rep_pd_cli: &Arc<dyn PdClient>,
        region_id: u64,
    ) {
        let Some(rep_region) = Self::get_region_for_rep(raft, region_id) else {
            return;
        };
        Self::report_region_to_rep_pd(rep_pd_cli, rep_region);
    }

    fn report_region_to_rep_pd(rep_pd_cli: &Arc<dyn PdClient>, region: metapb::Region) {
        info!("{} report region to pd event {:?}", region.id, region);
        let leader = region.get_peers()[0].clone();
        let mut stats = RegionStat::default();
        stats.approximate_kv_size = 100 * 1024 * 1024;
        stats.approximate_keys = 1000000;
        stats.approximate_size = 100 * 1024 * 1024;
        let resp = rep_pd_cli.region_heartbeat(1, region, leader, stats, None);
        tokio::spawn(async move {
            if let Err(err) = resp.await {
                warn!("region heartbeat failed"; "err" => ?err);
            }
        });
    }

    // The keyspace prefix will be trimmed for `rep-pd` & `rep-cdc`.
    fn get_region_for_rep(raft: &RfEngine, region_id: u64) -> Option<metapb::Region> {
        if raft.get_truncated_index(region_id) == Some(TRUNCATE_ALL_INDEX) {
            // region has been merged.
            return None;
        }

        let mut region_local_state = rfstore::store::load_last_peer_state(raft, region_id)?;
        let mut region = region_local_state.take_region();
        region.set_start_key(Self::trim_keyspace_prefix(region.get_start_key()));
        region.set_end_key(Self::trim_keyspace_prefix(region.get_end_key()));
        Some(region)
    }

    fn trim_keyspace_prefix(mut region_key: &[u8]) -> Vec<u8> {
        let raw_key = codec::bytes::decode_bytes(&mut region_key, false).unwrap();
        if raw_key.len() == 4 {
            vec![]
        } else {
            codec::bytes::encode_bytes(&raw_key[4..])
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
        let mut task_service: Box<dyn KeyspaceService> = if self.config.is_kube_mode() {
            let kube_api = self.kube_api.clone().unwrap();
            Box::new(KeyspaceKubeService::new(
                keyspace_id,
                kube_api,
                &self.config,
                &self.ctx.security_config,
            ))
        } else if pd_url.is_empty() || cdc_addr.is_empty() {
            cb(Err(Error::OtherError("pd_url or cdc_addr is empty".into())));
            return;
        } else {
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
        let kv = self.merged_engine.kv.clone();
        let http_client = self.http_client.clone();
        tokio::spawn(tikv_util::init_task_local(async move {
            let start_time = Instant::now_coarse();
            // Start service.
            let res = task_service.start().await;
            if res.is_err() {
                cb(res);
                return;
            }
            let cdc_addr = task_service.get_states().cdc_addr.clone();
            let res = get_cdc_status(&http_client, &cdc_addr, DISPATCH_CDC_TIMEOUT).await;
            if let Err(e) = res {
                cb(Err(box_err!("cdc_status failed: {:?}", e)));
                return;
            }

            // Prepare shards.
            // Note that the rfengine will update at the same time. So during load shards,
            // we will prepare again.
            let prepare_time = Instant::now_coarse();
            let mut metas = HashMap::default();
            if let Err(e) =
                Self::prepare_keyspace_shard_metas(keyspace_id, &scheduler, &kv, &mut metas).await
            {
                cb(Err(box_err!("prepare keyspace metas failed: {:?}", e)));
                return;
            }
            // Prepare again in case some shards are changed during last prepare.
            if let Err(e) =
                Self::prepare_keyspace_shard_metas(keyspace_id, &scheduler, &kv, &mut metas).await
            {
                cb(Err(box_err!("prepare keyspace metas failed: {:?}", e)));
                return;
            }

            // Load shards.
            let load_shards_time = Instant::now_coarse();
            let cb_with_log = move |res: Result<()>| {
                let end_time = Instant::now_coarse();
                info!("{} add_keyspace: {:?}", keyspace_id, &res;
                    "takes" => ?end_time.saturating_duration_since(start_time),
                    "start_svc" => ?prepare_time.saturating_duration_since(start_time),
                    "prepare" => ?load_shards_time.saturating_duration_since(prepare_time),
                    "load_shards" => ?end_time.saturating_duration_since(load_shards_time),
                );
                cb(res);
            };
            scheduler.schedule(CdcMsg::LoadKeyspaceShards {
                keyspace_id,
                task_service,
                cb: Box::new(cb_with_log),
            });
        }));
    }

    async fn upsert_keyspace_shard_metas(
        keyspace_id: u32,
        scheduler: &ReplicationScheduler,
        metas: &mut HashMap<u64 /* region_id */, ShardMeta>,
    ) -> Result<()> {
        let (cb, fut) = tikv_util::future::paired_future_callback();
        scheduler.schedule(CdcMsg::LoadKeyspaceShardMetas { keyspace_id, cb });
        let new_metas = fut
            .await
            .map_err(|_| -> Error { box_err!("load keyspace metas canceled") })?
            .map_err(|e| -> Error { box_err!("load keyspace metas failed: {:?}", e) })?;

        for (region_id, meta) in new_metas {
            match metas.entry(region_id) {
                Entry::Vacant(e) => {
                    e.insert(meta);
                }
                Entry::Occupied(mut e) => {
                    let exist_meta = e.get_mut();
                    if exist_meta.seq < meta.seq {
                        *exist_meta = meta;
                    } else {
                        debug_assert_eq!(exist_meta.seq, meta.seq);
                        e.remove();
                    }
                }
            }
        }

        Ok(())
    }

    async fn prepare_keyspace_shard_metas(
        keyspace_id: u32,
        scheduler: &ReplicationScheduler,
        kv: &Engine,
        metas: &mut HashMap<u64 /* region_id */, ShardMeta>,
    ) -> Result<()> {
        let start_time = Instant::now_coarse();
        Self::upsert_keyspace_shard_metas(keyspace_id, scheduler, metas).await?;

        let prepare_time = Instant::now_coarse();
        if !metas.is_empty() {
            kv.prepare_shards(metas)
                .map_err(|e| -> Error { box_err!("prepare keyspace metas failed: {:?}", e) })?;
        }

        let end_time = Instant::now_coarse();
        info!("{} prepare_keyspace_shard_metas", keyspace_id;
            "count" => metas.len(),
            "load" => ?prepare_time.saturating_duration_since(start_time),
            "prepare" => ?end_time.saturating_duration_since(prepare_time),
        );
        Ok(())
    }

    fn handle_load_keyspace_shards(
        &mut self,
        keyspace_id: u32,
        task_service: Box<dyn KeyspaceService>,
    ) -> Result<()> {
        let mut states = HashMap::default();
        states.insert(keyspace_id, Bytes::new());
        self.merged_engine.load_shards(&states)?;

        let states = task_service.get_states().marshal();
        let rep_pd_cli = task_service.get_pd_client();
        let raft = self.merged_engine.get_raft();
        let kv = self.merged_engine.get_kv();
        let interval = self.config.report_region_interval.0;
        self.runtime.spawn(async move {
            Self::report_regions_loop(
                keyspace_id,
                raft.clone(),
                kv.clone(),
                rep_pd_cli.clone(),
                interval,
            )
            .await
        });
        self.keyspaces.insert(keyspace_id, task_service);
        self.merged_engine
            .set_keyspace_states(keyspace_id, states)?;
        Ok(())
    }

    fn handle_load_keyspace_shard_metas(
        &mut self,
        keyspace_id: u32,
    ) -> Result<StdHashMap<u64 /* region_id */, ShardMeta>> {
        self.merged_engine
            .load_keyspace_shard_metas(keyspace_id)
            .map_err(Into::into)
    }

    fn handle_remove_keyspace_service(
        &mut self,
        keyspace_id: u32,
        cb: Box<dyn FnOnce(Result<()>) + Send>,
    ) {
        self.cdc_addrs.remove(&keyspace_id);
        let Some(mut svc) = self.keyspaces.remove(&keyspace_id) else {
            cb(Err(Error::OtherError("keyspace not found".into())));
            return;
        };
        let keyspace_regions = self.merged_engine.get_keyspace_regions(keyspace_id);
        let kv = self.merged_engine.get_kv();
        keyspace_regions.iter().for_each(|&region_id| {
            self.remove_region(region_id);
            kv.remove_shard(region_id);
        });
        self.merged_engine.remove_keyspace(keyspace_id);
        self.ctx.fs.get_runtime().spawn(async move {
            let res = svc.destroy().await;
            cb(res);
        });
    }

    fn handle_remove_task(&mut self, keyspace_id: u32, change_feed_id: String) -> Result<()> {
        let Some(task_ctx) = self.keyspaces.get_mut(&keyspace_id) else {
            return Err(Error::OtherError("keyspace service not found".into()));
        };
        if task_ctx
            .get_states_mut()
            .feeds
            .remove(&change_feed_id)
            .is_some()
        {
            self.merged_engine
                .set_keyspace_states(keyspace_id, task_ctx.get_states().marshal())?
        };
        Ok(())
    }

    fn handle_applied(&mut self, region_id: u64, region_events: RegionEvents) -> Result<()> {
        let tag = self.get_region_tag(region_id);
        if let Some(delegate) = self.region_delegates.get_mut(&region_id) {
            for (req_key, req_info) in delegate.requests.iter_mut() {
                let Some(conn) = self.conns.get(&req_key.conn_id) else {
                    warn!("{} handle_applied: conn not found, skip", tag;
                        "conn" => ?req_key.conn_id, "request" => %req_key.request_id);
                    continue;
                };
                for event in &region_events.events {
                    let mut event_to_send = Event::new();
                    event_to_send.set_request_id(req_key.request_id.into_inner());
                    event_to_send.set_region_id(region_id);
                    event_to_send.set_index(event.get_index());
                    if event.has_entries() {
                        let entries_to_send = event_to_send.mut_entries().mut_entries();
                        for entry in event.get_entries().get_entries() {
                            if req_info.in_range(entry.get_key()) {
                                entries_to_send.push(entry.clone());

                                debug!("send event";
                                    "key" => LogValue::key(entry.get_key()),
                                    "commit_ts" => entry.get_commit_ts(),
                                    "region" => region_id,
                                    "request" => %req_key.request_id,
                                    "r_type" => ?entry.r_type,
                                    "op_type" => ?entry.op_type);
                            }
                        }
                    }
                    if req_info.initialized {
                        conn.get_sink()
                            .unbounded_send(CdcEvent::Event(event_to_send), false)
                            .map_err(|e| cdc::Error::from(e))?;
                    } else {
                        req_info.pending_events.push(event_to_send);
                    }
                }
            }

            if let Some(resolver) = delegate.resolver.as_mut() {
                for (track_key, start_ts) in region_events.tracked_locks {
                    if start_ts == 0 {
                        // TODO: handle error.
                        resolver.untrack_lock(&track_key).unwrap();
                    } else {
                        // TODO: handle error.
                        resolver.track_lock(start_ts.into(), track_key).unwrap();
                    }
                }
            }
        }
        Ok(())
    }

    fn handle_applied_admin(
        &mut self,
        region_id: u64,
        region_version: u64,
        admin: AdminRequest,
    ) -> Result<()> {
        let raft = self.merged_engine.get_raft();
        let rep_region_opt = Self::get_region_for_rep(&raft, region_id);

        if let Some(rep_region) = &rep_region_opt {
            let Some(keyspace_id) = self.get_keyspace_id(region_id) else {
                let err_msg = format!("handle_applied_admin: region {} not found", region_id);
                debug_assert!(false, "{}", &err_msg);
                return Err(box_err!("{}", err_msg));
            };
            let task_ctx = self.keyspaces.get(&keyspace_id).unwrap();
            let pd_client = task_ctx.get_pd_client();
            Self::report_region_to_rep_pd(&pd_client, rep_region.clone());
            if admin.has_splits() {
                let split = admin.get_splits();
                for req in split.get_requests() {
                    Self::report_region_to_rep_pd_by_id(&raft, &pd_client, req.get_new_region_id());
                }
            }
        } else {
            info!("{} handle_applied_admin: region is merged", region_id);
        }

        let mut error = cdcpb::Error::new();
        let epoch_not_match = error.mut_epoch_not_match();
        // Set `region_not_found` if `rep_region_opt` is `None` ?
        if let Some(rep_region) = rep_region_opt {
            epoch_not_match.mut_current_regions().push(rep_region);
        }

        if let Some(delegate) = self.region_delegates.get_mut(&region_id) {
            let mut requests_to_remove = vec![];
            for (req_key, request) in delegate.requests.iter() {
                if request.region_version <= region_version {
                    requests_to_remove.push(*req_key);
                }
            }
            for req_key in requests_to_remove {
                delegate.requests.remove(&req_key).unwrap();
                let Some(conn) = self.conns.get(&req_key.conn_id) else {
                    continue;
                };
                let mut event = Event::new();
                event.set_region_id(region_id);
                event.set_request_id(req_key.request_id.into_inner());
                event.set_error(error.clone());
                info!(
                    "{} handle_applied_admin: send error event {:?}",
                    region_id, event
                );
                conn.get_sink()
                    .unbounded_send(CdcEvent::Event(event), false)
                    .map_err(|e| cdc::Error::from(e))?;
            }
            if delegate.requests.is_empty() {
                self.remove_region(region_id);
            }
        }
        Ok(())
    }

    fn remove_region(&mut self, region_id: u64) {
        self.region_to_keyspace.remove(&region_id);
        self.region_delegates.remove(&region_id);
    }
}

struct RegisterHandler {
    conn_id: ConnId,
    request_id: RequestId,
    snap_access: SnapAccess,
    sender: Sender<CdcMsg>,
    start_key: Bytes,
    end_key: Bytes,
    checkpoint_ts: u64,
    event_rows: Vec<EventRow>,
    initialized: bool,
}

impl RegisterHandler {
    fn new(
        conn_id: ConnId,
        request: &ChangeDataRequest,
        snap_access: SnapAccess,
        sender: Sender<CdcMsg>,
    ) -> Self {
        let keyspace_id = snap_access.get_keyspace_id();
        let request_id = request.get_request_id().into();
        let (start_key, end_key) = build_request_range_for_keyspace(keyspace_id, request);
        let checkpoint_ts = request.get_checkpoint_ts();
        Self {
            conn_id,
            request_id,
            snap_access,
            sender,
            start_key: start_key.into(),
            end_key: end_key.into(),
            checkpoint_ts,
            event_rows: vec![],
            initialized: false,
        }
    }

    async fn handle_register(&mut self) {
        let region_id = self.snap_access.get_id();
        info!("{} cdc register", region_id; "request" => %self.request_id, "conn" => ?self.conn_id);
        let mut entries_bytes = 0;
        let keyspace_id = self.snap_access.get_keyspace_id();
        let keyspace_prefix_len = keyspace_prefix_len(keyspace_id);
        let mut lock_iter = self
            .snap_access
            .new_iterator(LOCK_CF, false, false, None, false);
        lock_iter.set_range(self.start_key.clone(), self.end_key.clone());
        while lock_iter.valid() {
            let mut event_row = EventRow::new();
            let lock_key = lock_iter.key();
            event_row.set_key(lock_key[keyspace_prefix_len..].to_vec());
            if is_index_key(event_row.get_key()) {
                lock_iter.next();
                continue;
            }
            let lock_val = lock_iter.val();
            let mut lock = txn_types::Lock::parse(lock_val).unwrap();
            match lock.lock_type {
                LockType::Put => {
                    event_row.set_op_type(EventRowOpType::Put);
                }
                LockType::Delete => event_row.set_op_type(EventRowOpType::Delete),
                _ => {
                    lock_iter.next();
                    continue;
                }
            }
            event_row.set_start_ts(lock.ts.into_inner());
            let val = lock.short_value.take().unwrap_or_default();
            event_row.set_value(val);
            event_row.set_type(EventLogType::Prewrite);
            entries_bytes += event_row.get_key().len() + event_row.get_value().len();
            self.event_rows.push(event_row);
            if entries_bytes > MAX_INITIALIZE_SCAN_BATCH_BYTES {
                self.send_rows();
                entries_bytes = 0;
            }
            lock_iter.next();
        }

        info!("{} start incremental scan", region_id; "request" => %self.request_id, "checkpoint_ts" => self.checkpoint_ts);
        // scan incremental write after checkpoint ts;
        let mut write_iter = self
            .snap_access
            .new_delta_write_iterator_async(self.checkpoint_ts)
            .await;
        write_iter
            .seek_async(InnerKey::from_outer_key(&self.start_key))
            .await;
        let end_key = self.end_key.clone();
        let inner_end_key = InnerKey::from_outer_end_key(&end_key);
        while write_iter.valid() {
            let key = write_iter.key();
            if key >= inner_end_key {
                break;
            }
            let val = write_iter.value();
            if is_index_key(key.deref()) || val.is_deleted() || val.version <= self.checkpoint_ts {
                write_iter.next_async().await;
                continue;
            }
            let um = UserMeta::from_slice(val.user_meta());
            let mut event_row = EventRow::new();
            event_row.set_start_ts(um.start_ts);
            event_row.set_commit_ts(um.commit_ts);
            event_row.set_key(key.deref().to_vec());
            if val.get_value().is_empty() {
                event_row.set_op_type(EventRowOpType::Delete);
            } else {
                event_row.set_op_type(EventRowOpType::Put);
            }
            event_row.set_value(val.get_value().to_vec());
            let mut outer_key = ApiV2::get_keyspace_prefix_by_id(keyspace_id);
            outer_key.extend_from_slice(key.deref());
            let old_value = self
                .snap_access
                .get_async(WRITE_CF, &outer_key, um.commit_ts - 1)
                .await;
            if !old_value.get_value().is_empty() {
                event_row.set_old_value(old_value.get_value().to_vec());
            }
            event_row.set_type(EventLogType::Committed);
            entries_bytes += event_row.get_key().len() + event_row.get_value().len();
            debug!("delta add event row {:?}", event_row);
            self.event_rows.push(event_row);
            if entries_bytes > MAX_INITIALIZE_SCAN_BATCH_BYTES {
                self.send_rows();
                entries_bytes = 0;
            }
            write_iter.next_all_version_async().await;
        }
        let mut init_row = EventRow::new();
        init_row.set_type(EventLogType::Initialized);
        self.event_rows.push(init_row);
        self.initialized = true;
        self.send_rows();
    }

    fn send_rows(&mut self) {
        let event_rows = mem::take(&mut self.event_rows);
        let mut new_event = Event::new();
        new_event.set_region_id(self.snap_access.get_id());
        new_event.set_request_id(self.request_id.into_inner());
        new_event.mut_entries().set_entries(event_rows.into());

        debug!("send event {:?}", new_event);
        let _ = self.sender.send(CdcMsg::RegisterResult {
            event: new_event,
            conn_id: self.conn_id,
            initialized: self.initialized,
        });
    }
}

struct ScanLocksHandler {
    snap_access: SnapAccess,
    sender: Sender<CdcMsg>,
    locks: Vec<(Vec<u8>, TimeStamp)>,
}

impl ScanLocksHandler {
    fn new(snap_access: SnapAccess, sender: Sender<CdcMsg>) -> Self {
        Self {
            snap_access,
            sender,
            locks: vec![],
        }
    }

    fn scan_locks(&mut self) {
        let region_id = self.snap_access.get_id();
        let res = self.scan_locks_impl();
        let locks = res.map(|_| mem::take(&mut self.locks));
        if let Err(e) = self
            .sender
            .send(CdcMsg::ScanLocksResult { region_id, locks })
        {
            warn!("{} failed to send scan locks result", self.snap_access.get_tag(); "err" => ?e);
        }
    }

    fn scan_locks_impl(&mut self) -> Result<()> {
        let keyspace_id = self.snap_access.get_keyspace_id();
        let tag = self.snap_access.get_tag();
        info!("{} cdc scan locks", tag; "keyspace" => keyspace_id);

        let keyspace_prefix_len = keyspace_prefix_len(keyspace_id);
        let mut lock_iter = self
            .snap_access
            .new_iterator(LOCK_CF, false, false, None, false);
        lock_iter.set_range(
            self.snap_access.clone_start_key(),
            self.snap_access.clone_end_key(),
        );
        while lock_iter.valid() {
            let inner_lock_key = &lock_iter.key()[keyspace_prefix_len..];
            if is_index_key(inner_lock_key) {
                lock_iter.next();
                continue;
            }
            let lock_val = lock_iter.val();
            let lock = txn_types::Lock::parse(lock_val).unwrap();
            if !matches!(lock.lock_type, LockType::Put | LockType::Delete) {
                lock_iter.next();
                continue;
            }
            self.locks
                .push((lock_iter.key()[keyspace_prefix_len..].to_vec(), lock.ts));
            lock_iter.next();
        }

        info!("{} cdc scan locks", tag; "keyspace" => keyspace_id, "locks" => self.locks.len());
        Ok(())
    }
}
