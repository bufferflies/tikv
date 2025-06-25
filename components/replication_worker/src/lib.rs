// Copyright 2025 TiKV Project Authors. Licensed under Apache-2.0.

mod apply_observer;
mod error;
mod kube;
mod provisioned;
mod scheduler;
mod worker;

use std::{collections::HashMap, sync::Arc};

use api_version::ApiV2;
pub use apply_observer::{CdcApplyObserver, RegionEvents};
use async_trait::async_trait;
use bytes::Bytes;
use cdc::{Conn, ConnId, MemoryQuota};
pub use error::{Error, Result};
use futures::{future, SinkExt, TryFutureExt, TryStreamExt};
use grpcio::{DuplexSink, RequestStream, RpcContext, RpcStatus, RpcStatusCode, UnarySink};
use http::StatusCode;
use kvengine::{Shard, WRITE_CF};
use kvproto::{
    cdcpb,
    cdcpb::{ChangeDataEvent, ChangeDataRequest},
    cdcpb_grpc::ChangeData,
    errorpb::EpochNotMatch,
    kvrpcpb::{
        GetRequest, GetResponse, ScanLockRequest, ScanLockResponse, ScanRequest, ScanResponse,
    },
    metapb,
    metapb::{NodeState, RegionEpoch},
    raft_cmdpb::AdminRequest,
    tikvpb_grpc::Tikv,
};
use merged_engine::MergedEngineConfig;
use pd_client::{PdClient, RpcClient};
pub use provisioned::LocalProvider;
use resolved_ts::Resolver;
pub use scheduler::*;
use serde_derive::{Deserialize, Serialize};
use tikv::tikv_build_version;
use tikv_util::{config::ReadableDuration, error, info, warn};
use txn_types::TimeStamp;
pub use worker::ReplicationWorker;

pub(crate) const K8S_SERVICE_HOST: &str = "KUBERNETES_SERVICE_HOST";

#[derive(Clone, Serialize, Deserialize, PartialEq, Debug)]
#[serde(default)]
#[serde(rename_all = "kebab-case")]
pub struct ReplicationWorkerConfig {
    pub enabled: bool,

    pub grpc_addr: String,
    pub advertise_addr: String,

    // used for k8s mode.
    pub pd_sts_name: String,
    pub cdc_sts_name: String,
    pub namespace: String,

    pub report_region_interval: ReadableDuration,
    pub merged_engine: MergedEngineConfig,
}

impl Default for ReplicationWorkerConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            grpc_addr: "".to_string(),
            advertise_addr: "".to_string(),
            pd_sts_name: "".to_string(),
            cdc_sts_name: "".to_string(),
            namespace: "".to_string(),
            report_region_interval: ReadableDuration::secs(60),
            merged_engine: Default::default(),
        }
    }
}

impl ReplicationWorkerConfig {
    pub fn override_from_env(&mut self) {
        Self::env_or_default("PD_STS_NAME", &mut self.pd_sts_name);
        Self::env_or_default("CDC_STS_NAME", &mut self.cdc_sts_name);
    }

    fn env_or_default(name: &str, val: &mut String) {
        if let Ok(v) = std::env::var(name) {
            *val = v;
        }
    }

    pub fn is_kube_mode(&self) -> bool {
        !self.pd_sts_name.is_empty()
            && !self.cdc_sts_name.is_empty()
            && !self.namespace.is_empty()
            && std::env::var(K8S_SERVICE_HOST).is_ok()
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct KeyspaceStates {
    pub(crate) feeds: HashMap<String, String>,
    pub(crate) pd_url: String,
    pub(crate) cdc_addr: String,

    // used in k8s.
    pub(crate) pd_sts_name: String,
    pub(crate) cdc_sts_name: String,
}

impl KeyspaceStates {
    pub(crate) fn marshal(&self) -> Bytes {
        serde_json::to_vec(self).unwrap().into()
    }

    pub(crate) fn is_provisioned(&self) -> bool {
        self.pd_sts_name.is_empty()
    }
}

#[async_trait]
pub trait KeyspaceService: Send {
    fn keyspace_id(&self) -> u32;

    async fn start(&mut self) -> Result<()>;

    async fn destroy(&mut self) -> Result<()>;

    fn get_states(&self) -> &KeyspaceStates;

    fn get_states_mut(&mut self) -> &mut KeyspaceStates;

    fn get_pd_client(&self) -> Arc<dyn PdClient>;

    fn get_resolver(&mut self) -> &mut Resolver;
}

pub enum CdcMsg {
    AddKeyspace {
        keyspace_id: u32,
        pd_url: String,
        cdc_addr: String,
        cb: Box<dyn FnOnce(Result<()>) + Send>,
    },
    AddKeyspaceResult {
        keyspace_id: u32,
        result: Result<Box<dyn KeyspaceService>>,
        cb: Box<dyn FnOnce(Result<()>) + Send>,
    },
    NewTask {
        keyspace_id: u32,
        changefeed_id: String,
        body: Bytes,
        cb: Box<dyn FnOnce(Result<(StatusCode, Bytes)>) + Send>,
    },
    OpenConn(cdc::Conn),
    Register {
        request: ChangeDataRequest,
        conn_id: ConnId,
    },
    RegisterResult {
        keyspace_id: u32,
        event: cdcpb::Event,
        tracked_locks: Vec<(Vec<u8>, TimeStamp)>,
        conn_id: ConnId,
        initialized: bool,
    },
    Applied {
        region_id: u64,
        region_events: RegionEvents,
    },
    AppliedAdmin {
        region_id: u64,
        region_version: u64,
        admin: AdminRequest,
    },
    Deregister(ConnId),
    RemoveTask {
        keyspace_id: u32,
        change_feed_id: String,
        cb: Box<dyn FnOnce(Result<()>) + Send>,
    },
    RemoveKeyspace {
        keyspace_id: u32,
        cb: Box<dyn FnOnce(Result<()>) + Send>,
    },
    GetKeyspaces {
        cb: Box<dyn FnOnce(Vec<u32>) + Send>,
    },
    Stop,
}

#[derive(Clone)]
struct ReplicationService {
    kv: kvengine::Engine,
    scheduler: tikv_util::mpsc::Sender<CdcMsg>,
    memory_quota: MemoryQuota,
}

impl ReplicationService {
    pub fn new(kv: kvengine::Engine, scheduler: tikv_util::mpsc::Sender<CdcMsg>) -> Self {
        Self {
            kv,
            scheduler,
            memory_quota: MemoryQuota::new(1024 * 1024 * 1024),
        }
    }

    fn prepend_keyspace_prefix(shard: &Shard, req_key: &[u8]) -> Option<Bytes> {
        if req_key.is_empty() {
            None
        } else {
            let mut outer_req_key = ApiV2::get_keyspace_prefix_by_id(shard.keyspace_id);
            outer_req_key.extend_from_slice(req_key);
            Some(outer_req_key.into())
        }
    }
}

static CDC_CHANNEL_CAPACITY: usize = 128;

impl ChangeData for ReplicationService {
    fn event_feed(
        &mut self,
        ctx: RpcContext<'_>,
        stream: RequestStream<ChangeDataRequest>,
        mut sink: DuplexSink<ChangeDataEvent>,
    ) {
        let (event_sink, mut event_drain) =
            cdc::channel(CDC_CHANNEL_CAPACITY, self.memory_quota.clone());
        let peer = ctx.peer();
        let conn = Conn::new(event_sink, peer);
        let conn_id = conn.get_id();

        if let Err(status) = self.scheduler.send(CdcMsg::OpenConn(conn)).map_err(|e| {
            RpcStatus::with_message(RpcStatusCode::INVALID_ARGUMENT, format!("{:?}", e))
        }) {
            error!("cdc connection initiate failed"; "error" => ?status);
            ctx.spawn(
                sink.fail(status)
                    .unwrap_or_else(|e| error!("cdc failed to send error"; "error" => ?e)),
            );
            return;
        }

        let scheduler = self.scheduler.clone();
        let recv_req = stream.try_for_each(move |request| {
            info!("got event feed request {:?}", request);
            let ret = scheduler
                .send(CdcMsg::Register { request, conn_id })
                .map_err(|e| {
                    grpcio::Error::RpcFailure(RpcStatus::with_message(
                        RpcStatusCode::INVALID_ARGUMENT,
                        format!("{:?}", e),
                    ))
                });
            future::ready(ret)
        });

        let peer = ctx.peer();
        let scheduler = self.scheduler.clone();
        ctx.spawn(async move {
            let res = recv_req.await;
            // Unregister this downstream only.
            if let Err(e) = scheduler.send(CdcMsg::Deregister(conn_id)) {
                error!("cdc deregister failed"; "error" => ?e, "conn_id" => ?conn_id);
            }
            match res {
                Ok(()) => {
                    info!("cdc receive closed"; "downstream" => peer, "conn_id" => ?conn_id);
                }
                Err(e) => {
                    warn!("cdc receive failed"; "error" => ?e, "downstream" => peer, "conn_id" => ?conn_id);
                }
            }
        });

        let peer = ctx.peer();
        let scheduler = self.scheduler.clone();

        ctx.spawn(async move {
            let res = event_drain.forward(&mut sink).await;
            // Unregister this downstream only.
            if let Err(e) = scheduler.send(CdcMsg::Deregister(conn_id)) {
                error!("cdc deregister failed"; "error" => ?e);
            }
            match res {
                Ok(_s) => {
                    info!("cdc send closed"; "downstream" => peer, "conn_id" => ?conn_id);
                    let _ = sink.close().await;
                }
                Err(e) => {
                    warn!("cdc send failed"; "error" => ?e, "downstream" => peer, "conn_id" => ?conn_id);
                }
            }
        });
        info!("cdc event feed started");
    }
}

impl Tikv for ReplicationService {
    fn kv_scan(&mut self, ctx: RpcContext<'_>, req: ScanRequest, sink: UnarySink<ScanResponse>) {
        if req.reverse {
            let status = RpcStatus::new(RpcStatusCode::INVALID_ARGUMENT);
            ctx.spawn(
                sink.fail(status.clone())
                    .unwrap_or_else(|e| error!("kv_scan failed"; "error" => ?e)),
            );
            return;
        }
        let mut resp = ScanResponse::default();
        let region_id = req.get_context().get_region_id();
        let region_version = req.get_context().get_region_epoch().get_version();
        let res = self.kv.get_shard_with_ver(region_id, region_version);
        if res.is_err() {
            let mut region_err = kvproto::errorpb::Error::default();
            let epoch_not_match = EpochNotMatch::default();
            region_err.set_epoch_not_match(epoch_not_match);
            resp.set_region_error(region_err);
            ctx.spawn(
                sink.success(resp)
                    .unwrap_or_else(|e| error!("kv_scan failed"; "error" => ?e)),
            );
            return;
        }
        let shard = res.unwrap();
        let outer_start_key = Self::prepend_keyspace_prefix(&shard, req.get_start_key())
            .unwrap_or_else(|| shard.outer_start.clone());
        let outer_end_key = Self::prepend_keyspace_prefix(&shard, req.get_end_key())
            .unwrap_or_else(|| shard.outer_end.clone());
        let read_ts = Some(req.get_version());
        let snap_access = shard.new_snap_access();
        let task = async move {
            // TODO: handle locks.
            let mut iter = snap_access
                .new_iterator_async(WRITE_CF, false, false, read_ts, false)
                .await;
            iter.set_range_async(outer_start_key, outer_end_key).await;
            let mut kv_pairs = vec![];
            let keyspace_prefix_len = ApiV2::get_keyspace_prefix_by_id(shard.keyspace_id).len();
            while iter.valid() {
                if kv_pairs.len() == req.get_limit() as usize {
                    break;
                }
                let mut kv_pair = kvproto::kvrpcpb::KvPair::default();
                let key = iter.key();
                // trim keyspace prefix.
                kv_pair.set_key(key[keyspace_prefix_len..].to_vec());
                let val = iter.val();
                if val.is_empty() {
                    iter.next_async().await;
                    continue;
                }
                kv_pair.set_value(val.to_vec());
                kv_pairs.push(kv_pair);
                iter.next_async().await;
            }
            resp.set_pairs(kv_pairs.into());
            sink.success(resp)
                .unwrap_or_else(|e| {
                    error!("kv_scan failed"; "error" => ?e);
                })
                .await
        };
        ctx.spawn(task);
    }

    fn kv_get(&mut self, ctx: RpcContext<'_>, req: GetRequest, sink: UnarySink<GetResponse>) {
        let mut resp = GetResponse::default();
        let region_id = req.get_context().get_region_id();
        let region_version = req.get_context().get_region_epoch().get_version();
        let res = self.kv.get_shard_with_ver(region_id, region_version);
        if res.is_err() {
            let mut region_err = kvproto::errorpb::Error::default();
            let epoch_not_match = EpochNotMatch::default();
            region_err.set_epoch_not_match(epoch_not_match);
            resp.set_region_error(region_err);
            ctx.spawn(
                sink.success(resp)
                    .unwrap_or_else(|e| error!("kv_get failed"; "error" => ?e)),
            );
            return;
        }
        let shard = res.unwrap();
        let snap_access = shard.new_snap_access();
        let key = Self::prepend_keyspace_prefix(&shard, req.get_key()).unwrap();
        let task = async move {
            // TODO: handle locks.
            let item = snap_access
                .get_async(WRITE_CF, &key, req.get_version())
                .await;
            if !item.get_value().is_empty() {
                resp.set_value(item.get_value().to_vec());
            } else {
                resp.set_not_found(true);
            }
            sink.success(resp)
                .unwrap_or_else(|e| {
                    error!("kv_get failed"; "error" => ?e);
                })
                .await
        };
        ctx.spawn(task);
    }

    fn kv_scan_lock(
        &mut self,
        ctx: RpcContext<'_>,
        req: ScanLockRequest,
        sink: UnarySink<ScanLockResponse>,
    ) {
        // TODO: forward resolve lock request to upstream.
        warn!("received scan lock from CDC {:?}", req);
        let resp = ScanLockResponse::new();
        ctx.spawn(sink.success(resp).unwrap_or_else(|e| {
            error!("kv_scan_lock failed"; "error" => ?e);
        }));
    }
}

async fn bootstrap(pd_client: Arc<RpcClient>, store_id: u64, advertise_addr: String) -> Result<()> {
    let bootstrapped = pd_client.is_cluster_bootstrapped()?;
    if !bootstrapped {
        let mut store = metapb::Store::new();
        store.set_id(store_id);
        store.set_node_state(NodeState::Serving);
        store.set_address(advertise_addr);
        store.set_version(tikv_build_version().to_string());
        let mut initial_region = metapb::Region::new();
        initial_region.set_id(1);
        let mut initial_peer = metapb::Peer::new();
        initial_peer.set_id(1);
        initial_peer.set_store_id(store_id);
        initial_region.set_peers(vec![initial_peer].into());
        let mut initial_epoch = RegionEpoch::new();
        initial_epoch.set_version(1);
        initial_epoch.set_conf_ver(1);
        initial_region.set_region_epoch(initial_epoch);
        pd_client.bootstrap_cluster(store, initial_region)?;
    }
    Ok(())
}
