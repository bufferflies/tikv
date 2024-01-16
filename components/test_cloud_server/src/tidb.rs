// Copyright 2023 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    fs, ops,
    path::{Path, PathBuf},
    process::{
        Command, {self},
    },
    sync::Arc,
    time::Duration,
};

use bstr::ByteSlice;
use bytes::Bytes;
use dashmap::DashMap;
use futures::executor::block_on;
use grpcio::EnvBuilder;
use hyper::{Body, Method, Request};
use pd_client::{
    pd_control::{PdControl, PdScheduleConfig},
    PdClient,
};
use security::{HttpClient, SecurityConfig, SecurityManager};
use serde_derive::Serialize;
use tempfile::TempDir;
use tikv_util::{box_err, debug, info};

use crate::try_wait_result_async;

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Sync + Send>>;

pub enum PdServerMode {
    Normal,
    MicroServices { tso_count: u16 },
}

/// Starts/stops pd-servers and provides interfaces such as PD endpoints.
pub struct PdServers {
    mode: PdServerMode,
    bin_path: PathBuf,
    data_path: PathBuf,
    port_base: u16,
    pre_alloc_keyspaces: u16,
    schedule_config: PdScheduleConfig,

    security_mgr: Arc<SecurityManager>,

    servers: DashMap<u16 /* idx */, process::Child>,
    tso_svcs: DashMap<u16 /* idx */, process::Child>,
}

impl PdServers {
    pub fn new(
        mode: PdServerMode,
        bin_path: PathBuf,
        data_path: PathBuf,
        port_base: u16,
        pre_alloc_keyspaces: u16,
        schedule_config: PdScheduleConfig,
        security_mgr: Arc<SecurityManager>,
    ) -> Self {
        Self {
            mode,
            bin_path,
            data_path,
            port_base,
            pre_alloc_keyspaces,
            schedule_config,
            security_mgr,
            servers: DashMap::new(),
            tso_svcs: DashMap::new(),
        }
    }

    pub fn start(&self, count: u16) {
        for idx in 0..count {
            self.start_single(idx, count);
        }

        if let PdServerMode::MicroServices { tso_count } = self.mode {
            for idx in 0..tso_count {
                self.start_single_tso_service(idx);
            }
        }
    }

    fn start_single(&self, idx: u16, total_count: u16) {
        let data_dir = self.data_path.join(format!("pd-{idx}"));
        let log_file = self.data_path.join(format!("pd-{idx}.log"));
        let initial_cluster = (0..total_count)
            .map(|i| format!("pd-{}=http://127.0.0.1:{}", i, self.peer_port(i)))
            .collect::<Vec<_>>()
            .join(",");

        let config_file = self.data_path.join(format!("pd-{idx}.toml"));
        let config = PdConfig {
            keyspace: PdKeyspaceConfig {
                pre_alloc: (1..=self.pre_alloc_keyspaces)
                    .map(|i| keyspace_name_by_idx(i))
                    .collect(),
            },
            schedule: self.schedule_config.clone(),
            ..Default::default()
        };
        let toml = toml::to_string(&config).unwrap();
        fs::write(&config_file, toml).unwrap();

        let mut cmd = Command::new(&self.bin_path);
        if matches!(self.mode, PdServerMode::MicroServices { .. }) {
            cmd.arg("services").arg("api");
        }
        cmd.arg(format!("--name=pd-{}", idx))
            .arg(format!("--data-dir={}", data_dir.display()))
            .arg(format!("--log-file={}", log_file.display()))
            .arg(format!(
                "--client-urls=http://127.0.0.1:{}",
                self.client_port(idx)
            ))
            .arg(format!(
                "--peer-urls=http://127.0.0.1:{}",
                self.peer_port(idx)
            ))
            .arg(format!("--initial-cluster={}", initial_cluster))
            .arg(format!("--config={}", config_file.display()));
        info!("start pd-server"; "cmd" => ?cmd);
        let child = cmd.spawn().unwrap();
        let old = self.servers.insert(idx, child);
        assert!(old.is_none(), "pd-{} already started (but dead ?)", idx);
    }

    fn start_single_tso_service(&self, idx: u16) {
        let log_file = self.data_path.join(format!("pd-tso-{idx}.log"));
        let mut cmd = Command::new(&self.bin_path);
        cmd.arg("services")
            .arg("tso")
            .arg(format!(
                "--backend-endpoints={}",
                self.endpoints_with_scheme().join(",")
            ))
            .arg(format!(
                "--listen-addr=http://127.0.0.1:{}",
                self.tso_svc_port(idx)
            ))
            .arg(format!(
                "--advertise-listen-addr=http://127.0.0.1:{}",
                self.tso_svc_port(idx)
            ))
            .arg(format!("--log-file={}", log_file.display()));
        info!("start pd-server tso"; "cmd" => ?cmd);
        let child = cmd.spawn().unwrap();
        let old = self.tso_svcs.insert(idx, child);
        assert!(old.is_none(), "pd-tso-{} already started (but dead ?)", idx);
    }

    pub fn client_port(&self, idx: u16) -> u16 {
        self.port_base + idx * 10
    }

    pub fn peer_port(&self, idx: u16) -> u16 {
        self.client_port(idx) + 1
    }

    pub fn tso_svc_port(&self, idx: u16) -> u16 {
        self.client_port(idx) + 100
    }

    pub fn endpoints(&self) -> Vec<String> {
        self.servers
            .iter()
            .map(|kv| {
                let idx = *kv.key();
                format!("127.0.0.1:{}", self.client_port(idx))
            })
            .collect()
    }

    pub fn endpoints_with_scheme(&self) -> Vec<String> {
        self.servers
            .iter()
            .map(|kv| {
                let idx = *kv.key();
                format!("http://127.0.0.1:{}", self.client_port(idx))
            })
            .collect()
    }

    pub async fn must_healthy(&self, timeout: Duration) {
        let pd_ctl = Arc::new(self.get_pd_control());
        try_wait_result_async(
            || {
                let pd_ctl = pd_ctl.clone();
                Box::pin(async move {
                    match pd_ctl.health().await {
                        Ok(health) if health => Ok(()),
                        Ok(_) => Err("PD unhealthy".to_string()),
                        Err(e) => {
                            let err = Err(format!("check PD healthy failed: {:?}", e));
                            info!("{:?}", err);
                            err
                        }
                    }
                })
            },
            timeout.as_secs() as usize,
        )
        .await
        .unwrap_or_else(|e| panic!("wait PD healthy timeout: {:?}", e));

        let pd_client = Arc::new(self.get_client_async().await);
        try_wait_result_async(
            || {
                let pd_client = pd_client.clone();
                Box::pin(async move {
                    match pd_client.get_tso().await {
                        Ok(ts) => {
                            info!("get_tso: {:?}", ts);
                            Ok(())
                        }
                        Err(e) => {
                            let err = Err(format!("get_tso failed: {:?}", e));
                            info!("{:?}", err);
                            err
                        }
                    }
                })
            },
            timeout.as_secs() as usize,
        )
        .await
        .unwrap_or_else(|e| panic!("wait PD get_tso timeout: {:?}", e));
        info!("PD is ready");
    }

    pub fn get_pd_control(&self) -> PdControl {
        let endpoints = self.endpoints();
        assert!(
            !endpoints.is_empty(),
            "no PD endpoints, invoke start() first"
        );
        let cfg = pd_client::Config::new(endpoints);
        let mgr = self.security_mgr.clone();
        PdControl::new(cfg, mgr).unwrap()
    }

    pub fn get_client(&self) -> pd_client::RpcClient {
        block_on(self.get_client_async())
    }

    pub async fn get_client_async(&self) -> pd_client::RpcClient {
        let endpoints = self.endpoints();
        let env = Arc::new(EnvBuilder::new().cq_count(1).build());
        let mgr = self.security_mgr.clone();
        let cfg = pd_client::Config::new(endpoints);
        cfg.validate().unwrap();
        pd_client::RpcClient::new_async(&cfg, Some(env), mgr)
            .await
            .unwrap_or_else(|e| panic!("failed to create rpc client: {:?}", e))
    }

    pub fn stop(&self, idx: u16, children: &DashMap<u16, process::Child>) {
        let (_, mut child) = children.remove(&idx).unwrap();
        // TODO: gracefully stop by SIGINT
        child.kill().unwrap_or_else(|err| {
            panic!("pd-{} has exited unexpectedly: {}", idx, err);
        });
    }

    pub fn stop_all(&self) {
        let tso_svcs = self.tso_svcs.iter().map(|kv| *kv.key()).collect::<Vec<_>>();
        for idx in tso_svcs {
            self.stop(idx, &self.tso_svcs);
        }

        let servers = self.servers.iter().map(|kv| *kv.key()).collect::<Vec<_>>();
        for idx in servers {
            self.stop(idx, &self.servers);
        }
    }
}

/// Starts/stops tidb-servers and provides interfaces such as host, port,
/// username, and password.
pub struct TidbServers {
    bin_path: PathBuf,
    data_path: PathBuf,
    port_base: u16,
    status_port_base: u16,

    security_mgr: Arc<SecurityManager>,

    children: DashMap<u16 /* idx */, process::Child>,
}

impl TidbServers {
    pub fn new(
        bin_path: PathBuf,
        data_path: PathBuf,
        port_base: u16,
        status_port_base: u16,
        security_mgr: Arc<SecurityManager>,
    ) -> Self {
        Self {
            bin_path,
            data_path,
            port_base,
            status_port_base,
            security_mgr,
            children: DashMap::new(),
        }
    }

    pub fn root(&self, idx: u16) -> String {
        format!("{}.root", keyspace_name_by_idx(idx))
    }

    pub fn port(&self, idx: u16) -> u16 {
        self.port_base + idx
    }

    pub fn status_port(&self, idx: u16) -> u16 {
        self.status_port_base + idx
    }

    pub fn conn_params(&self, idx: u16) -> ConnParams {
        ConnParams {
            host: "127.0.0.1".to_string(),
            port: self.port(idx),
            user: self.root(idx),
            password: "".to_string(),
        }
    }

    pub fn start(&self, idx: u16, pd_endpoints: &[String]) {
        let pd_endpoints = pd_endpoints.join(",");
        let log_file = self.data_path.join(format!("tidb-{idx}.log"));
        let slow_log_file = self.data_path.join(format!("tidb-slow-{idx}.log"));

        let config_file = self.data_path.join(format!("tidb-{idx}.toml"));
        let config = TidbConfig {
            keyspace_name: keyspace_name_by_idx(idx),
        };
        let toml = toml::to_string(&config).unwrap();
        fs::write(&config_file, toml).unwrap();

        let mut cmd = Command::new(&self.bin_path);
        cmd.arg("-L=info")
            .arg("--store=tikv")
            .arg("--host=127.0.0.1")
            .arg(format!("--path={}", pd_endpoints))
            .arg(format!("-P={}", self.port(idx)))
            .arg(format!("--status={}", self.status_port(idx)))
            .arg(format!("--log-file={}", log_file.display()))
            .arg(format!("--log-slow-query={}", slow_log_file.display()))
            .arg(format!("--config={}", config_file.display()));
        info!("start tidb-server"; "cmd" => ?cmd);
        let child = cmd.spawn().unwrap();
        let old = self.children.insert(idx, child);
        assert!(old.is_none(), "tidb-{} already started", idx);
    }

    pub fn get_tidb_control(&self, idx: u16) -> TidbControl {
        let endpoint = format!("127.0.0.1:{}", self.status_port(idx));
        TidbControl::new(endpoint, self.security_mgr.clone())
    }

    pub async fn must_healthy(&self, idx: u16, timeout: Duration) {
        let tidb_ctl = self.get_tidb_control(idx);
        try_wait_result_async(
            || {
                let tidb_ctl = tidb_ctl.clone();
                Box::pin(async move {
                    match tidb_ctl.health().await {
                        Ok(_) => Ok(()),
                        Err(e) => {
                            let err = Err(format!("check TiDB healthy failed: {:?}", e));
                            info!("{:?}", err);
                            err
                        }
                    }
                })
            },
            timeout.as_secs() as usize,
        )
        .await
        .unwrap_or_else(|e| panic!("wait TiDB-{} healthy timeout: {:?}", idx, e));
        info!("TiDB-{idx} is ready");
    }

    pub async fn must_all_healthy(&self, timeout: Duration) {
        let all = self.get_all_indexes();
        for idx in all {
            self.must_healthy(idx, timeout).await;
        }
    }

    pub fn stop(&self, idx: u16) {
        let (_, mut child) = self.children.remove(&idx).unwrap();
        // TODO: gracefully kill by SIGINT
        child.kill().unwrap_or_else(|err| {
            panic!("tidb-{} has exited unexpectedly: {}", idx, err);
        })
    }

    pub fn stop_all(&self) {
        let all = self.get_all_indexes();
        for idx in all {
            self.stop(idx);
        }
    }

    fn get_all_indexes(&self) -> Vec<u16> {
        self.children.iter().map(|kv| *kv.key()).collect::<Vec<_>>()
    }
}

#[derive(Clone)]
pub struct TidbCluster {
    inner: Arc<TidbClusterCore>,
}

impl ops::Deref for TidbCluster {
    type Target = TidbClusterCore;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl TidbCluster {
    pub fn new(
        pd_mode: PdServerMode,
        pd_bin: PathBuf,
        pd_port_base: u16,
        pd_schedule_config: PdScheduleConfig,
        tidb_bin: PathBuf,
        tidb_port_base: u16,
        tidb_status_port_base: u16,
        pre_alloc_keyspaces: u16,
        security_conf: &SecurityConfig,
    ) -> Self {
        check_binary("pd_server", &pd_bin);
        check_binary("tidb-server", &tidb_bin);

        let security_mgr = Arc::new(SecurityManager::new(security_conf).unwrap());
        let base_path = tempfile::Builder::new().prefix("tc_").tempdir().unwrap();

        let pd = PdServers::new(
            pd_mode,
            pd_bin,
            base_path.path().to_owned(),
            pd_port_base,
            pre_alloc_keyspaces,
            pd_schedule_config,
            security_mgr.clone(),
        );
        let tidb = TidbServers::new(
            tidb_bin,
            base_path.path().to_owned(),
            tidb_port_base,
            tidb_status_port_base,
            security_mgr,
        );

        let inner = TidbClusterCore {
            _data_path: base_path,
            pd,
            tidb,
        };

        Self {
            inner: Arc::new(inner),
        }
    }

    pub fn keyspace_name(idx: u16) -> String {
        keyspace_name_by_idx(idx)
    }

    pub fn get_idx_by_keyspace_name(name: &str) -> u16 {
        get_idx_from_keyspace_name(name)
    }
}

pub struct TidbClusterCore {
    _data_path: TempDir,

    pub pd: PdServers,
    pub tidb: TidbServers,
}

impl TidbClusterCore {
    pub async fn start_pd(&self, count: u16, timeout: Duration) {
        self.pd.start(count);
        // tikv-servers should be started after PD is healthy. Otherwise it will panic.
        self.pd.must_healthy(timeout).await;
    }

    pub async fn start_tidb(&self, count: u16, timeout: Duration) {
        let pd_endpoints = self.pd.endpoints();
        // Start from 1 as keyspace 0 is reserved.
        for idx in 1..=count {
            self.tidb.start(idx, &pd_endpoints);
        }
        self.tidb.must_all_healthy(timeout).await;
    }
}

pub struct ConnParams {
    pub host: String,
    pub port: u16,
    pub user: String,
    pub password: String,
}

impl ConnParams {
    pub fn conn_string(&self, db: &str) -> String {
        // mysql://user:password@host:port/db
        format!(
            "mysql://{}:{}@{}:{}/{db}",
            self.user, self.password, self.host, self.port
        )
    }
}

#[derive(Default, Serialize)]
#[serde(rename_all = "kebab-case")]
struct PdConfig {
    replication: PdReplicationConfig,
    keyspace: PdKeyspaceConfig,
    schedule: PdScheduleConfig,
}

#[derive(Serialize)]
#[serde(rename_all = "kebab-case")]
struct PdReplicationConfig {
    max_replicas: usize,
}

impl Default for PdReplicationConfig {
    fn default() -> Self {
        Self { max_replicas: 3 }
    }
}

#[derive(Default, Serialize)]
#[serde(rename_all = "kebab-case")]
struct PdKeyspaceConfig {
    pre_alloc: Vec<String>,
}

#[derive(Default, Serialize)]
#[serde(rename_all = "kebab-case")]
struct TidbConfig {
    keyspace_name: String,
}

const TIDB_STATUS_PATH: &str = "status";

/// TidbControl provides access to HTTP APIs of TiDB, which are not included in
/// gRPC interface. It's also expected to act like the tool `tidb-ctl`.
#[derive(Clone)]
pub struct TidbControl {
    security_mgr: Arc<SecurityManager>,
    endpoint: String,
    client: HttpClient,
}

impl TidbControl {
    pub fn new(endpoint: String, security_mgr: Arc<SecurityManager>) -> Self {
        let client = security_mgr.http_client(hyper::Client::builder()).unwrap();
        Self {
            endpoint,
            security_mgr,
            client,
        }
    }

    async fn request_restful(
        &self,
        path: String,
        method: Method,
        body_data: Option<Vec<u8>>,
    ) -> Result<Bytes> {
        let uri = self
            .security_mgr
            .build_uri(format!("{}/{}", self.endpoint, path))?;
        let req = Request::builder()
            .method(method.clone())
            .uri(uri)
            .body(match body_data {
                Some(ref data) => Body::from(data.to_owned()),
                None => Body::empty(),
            })
            .unwrap();
        let resp = self.client.request(req).await;
        match resp {
            Err(e) => Err(box_err!("request {} failed {:?}", path, e)),
            Ok(resp) => {
                let status = resp.status();
                let body = hyper::body::to_bytes(resp.into_body()).await.unwrap();
                if status.is_success() {
                    Ok(body)
                } else {
                    Err(box_err!(
                        "TiDB({}) path {} return error: {}: {}",
                        self.endpoint,
                        path,
                        status,
                        body.to_str_lossy()
                    ))
                }
            }
        }
    }

    pub async fn health(&self) -> Result<()> {
        match self
            .request_restful(TIDB_STATUS_PATH.to_string(), Method::GET, None)
            .await
        {
            Ok(_) => Ok(()),
            Err(e) => {
                let err = Err(box_err!("check TiDB healthy failed: {:?}", e));
                debug!("{:?}", err);
                err
            }
        }
    }
}

/// On real PD, keyspaces are indexed by name other than id.
/// So we need the conversion between keyspace id & name, to locate TiDB in
/// TidbCluster.
const KEYSPACE_NAME_PREFIX: &str = "ks";

fn keyspace_name_by_idx(idx: u16) -> String {
    format!("{KEYSPACE_NAME_PREFIX}{idx}")
}

fn get_idx_from_keyspace_name(name: &str) -> u16 {
    assert_eq!(&name[..KEYSPACE_NAME_PREFIX.len()], KEYSPACE_NAME_PREFIX);
    name[KEYSPACE_NAME_PREFIX.len()..].parse().unwrap()
}

/// Check binaries by running with "-V".
fn check_binary(name: &str, bin_path: &Path) {
    let mut cmd = Command::new(bin_path);
    cmd.arg("-V");
    let output = cmd.output().unwrap();
    assert!(
        output.status.success(),
        "{} --version failed: {:?}",
        name,
        output
    );
    info!(
        "{} version {}",
        name,
        String::from_utf8_lossy(&output.stdout).as_ref()
    );
}
