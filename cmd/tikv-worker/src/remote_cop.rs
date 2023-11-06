// Copyright 2023 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    collections::HashMap,
    net::SocketAddr,
    str::FromStr,
    sync::{Arc, Mutex},
    time::Duration,
};

use cloud_encryption::MasterKey;
use futures::{FutureExt, TryFutureExt};
use grpcio::{ChannelBuilder, EnvBuilder, Environment, RpcStatus, RpcStatusCode, ServerBuilder};
use kvengine::dfs::Dfs;
use kvproto::{
    coprocessor::{DelegateRequest, Request, Response},
    tikvpb::{create_tikv, Tikv, TikvClient},
};
use pd_client::PdClient;
use tikv::coprocessor::parse_request_and_handle_remote_cop;
use tikv_util::{quota_limiter::QuotaLimiter, thd_name, warn};

#[derive(Clone, Default, Serialize, Deserialize, PartialEq, Debug)]
#[serde(default)]
#[serde(rename_all = "kebab-case")]
pub struct Config {
    pub addr: String,
    pub max_handle_duration: Duration,
}

pub struct RemoteCopServer {
    grpc_server: grpcio::Server,
}

impl RemoteCopServer {
    pub fn new(
        pd: Arc<dyn PdClient>,
        dfs: Arc<dyn Dfs>,
        cfg: Config,
        master_key: MasterKey,
    ) -> RemoteCopServer {
        let env = Arc::new(
            EnvBuilder::new()
                .cq_count(16)
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
        let addr = SocketAddr::from_str(&cfg.addr).unwrap();
        let cop_service = CopService::new(pd.clone(), dfs, env.clone(), cfg, master_key);
        let sb = ServerBuilder::new(env)
            .channel_args(channel_args)
            .register_service(create_tikv(cop_service));
        let sb = pd
            .get_security_mgr()
            .bind(sb, &addr.ip().to_string(), addr.port());
        let grpc_server = sb.build().unwrap();
        Self { grpc_server }
    }

    pub fn start(&mut self) {
        self.grpc_server.start()
    }
}

#[derive(Clone)]
pub struct CopService {
    pd: Arc<dyn PdClient>,
    dfs: Arc<dyn Dfs>,
    env: Arc<Environment>,
    store_addrs: Arc<Mutex<HashMap<u64, String>>>,
    channels: Arc<Mutex<HashMap<String, TikvClient>>>,
    cfg: Config,
    quota_limiter: Arc<QuotaLimiter>,
    master_key: MasterKey,
}

impl CopService {
    fn new(
        pd: Arc<dyn PdClient>,
        dfs: Arc<dyn Dfs>,
        env: Arc<Environment>,
        cfg: Config,
        master_key: MasterKey,
    ) -> Self {
        Self {
            pd,
            dfs,
            env,
            store_addrs: Arc::new(Mutex::new(HashMap::new())),
            channels: Arc::new(Mutex::new(HashMap::new())),
            cfg,
            quota_limiter: Arc::new(QuotaLimiter::default()),
            master_key,
        }
    }
}

impl Tikv for CopService {
    fn coprocessor(
        &mut self,
        ctx: grpcio::RpcContext<'_>,
        req: Request,
        sink: grpcio::UnarySink<Response>,
    ) {
        let store_id = req.get_context().get_peer().get_store_id();
        let client_res = self.get_client(store_id);
        if let Err(err) = client_res {
            warn!("get client failed"; "err" => ?err);
            ctx.spawn(
                sink.fail(RpcStatus::with_message(
                    RpcStatusCode::INTERNAL,
                    err.to_string(),
                ))
                .unwrap_or_else(|e| {
                    warn!("failed to send rpc status"; "err" => ?e);
                }),
            );
            return;
        }
        let client = client_res.unwrap();
        let mut delegate_req = DelegateRequest::new();
        let key_ranges = req.get_ranges().to_vec();
        delegate_req.set_context(req.get_context().clone());
        delegate_req.set_ranges(key_ranges.into());
        delegate_req.set_start_ts(req.get_start_ts());
        let max_handle_duration = self.cfg.max_handle_duration;
        let quota_limit = self.quota_limiter.clone();
        let peer = Some(ctx.peer());
        let dfs = self.dfs.clone();
        let master_key = self.master_key.clone();
        let future = async move {
            let mut resp = client
                .delegate_coprocessor_async(&delegate_req)
                .unwrap()
                .await
                .map_err(|e| tikv::coprocessor::Error::Other(format!("{:?}", e)))?;
            let snap_access = kvengine::SnapAccess::construct_snapshot(
                dfs,
                &resp.take_mem_table_data(),
                &resp.take_snapshot(),
                &master_key,
            )
            .await
            .map_err(|e| tikv::coprocessor::Error::Other(format!("{:?}", e)))?;
            let snapshot = rfstore::store::RegionSnapshot::from_snapshot(snap_access);
            parse_request_and_handle_remote_cop(
                req,
                peer,
                max_handle_duration,
                quota_limit,
                snapshot,
            )
            .await
        };
        let task = async move {
            match future.await {
                Ok(mut resp) => sink.success(resp.consume()).await?,
                Err(err) => {
                    sink.fail(RpcStatus::with_message(
                        RpcStatusCode::INTERNAL,
                        err.to_string(),
                    ))
                    .await?
                }
            }
            Ok(())
        }
        .map_err(|e: grpcio::Error| {
            warn!("remote cop failed";
                "request" => "coprocessor",
                "err" => ?e
            );
        })
        .map(|_| ());
        ctx.spawn(task);
    }
}

impl CopService {
    fn resolve_store_addr(&mut self, store_id: u64) -> pd_client::Result<String> {
        let mut addrs = self.store_addrs.lock().unwrap();
        if let Some(addr) = addrs.get(&store_id) {
            return Ok(addr.clone());
        }
        let store = self.pd.get_store(store_id)?;
        addrs.insert(store_id, store.address.clone());
        Ok(store.address)
    }

    fn get_client(&mut self, store_id: u64) -> pd_client::Result<TikvClient> {
        let addr = self.resolve_store_addr(store_id)?;
        let mut channels = self.channels.lock().unwrap();
        if let Some(channel) = channels.get(&addr) {
            return Ok(channel.clone());
        }
        let builder = ChannelBuilder::new(self.env.clone());
        let channel = builder.connect(&addr);
        let client = TikvClient::new(channel);
        channels.insert(addr, client.clone());
        Ok(client)
    }
}
