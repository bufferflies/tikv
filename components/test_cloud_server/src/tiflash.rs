// Copyright 2024 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    fs,
    path::PathBuf,
    process::{self, Command, Stdio},
    sync::{Arc, Mutex},
    time::Duration,
};

use dashmap::DashMap;
use hyper::Method;
use kvengine::dfs::DFSConfig;
use security::{RestfulClient, SecurityManager};
use serde_derive::Serialize;
use tikv_util::info;

use crate::try_wait_result_async;

const TIFLASH_HTTP_PORT_BASE: u16 = 8123;
const TIFLASH_TCP_PORT_BASE: u16 = 9000;
const TIFLASH_SERVICE_PORT_BASE: u16 = 3930;
const TIFLASH_METRICS_PORT_BASE: u16 = 8234;
const TIFLASH_PROXY_PORT_BASE: u16 = 20170;
const TIFLASH_PROXY_STATUS_PORT_BASE: u16 = 20292;
const MINIO_PORT: u16 = 29000;

pub enum TiFlashRole {
    Legacy,
    Compute,
    Write,
}

pub struct TiFlashServers {
    bin_path: PathBuf,
    data_path: PathBuf,
    security_mgr: Arc<SecurityManager>,
    servers: DashMap<u16 /* idx */, process::Child>,
    minio: Arc<Mutex<Option<process::Child>>>,
}

impl Drop for TiFlashServers {
    fn drop(&mut self) {
        self.stop_all();
    }
}

impl TiFlashServers {
    pub fn new(bin_path: PathBuf, data_path: PathBuf, security_mgr: Arc<SecurityManager>) -> Self {
        Self {
            bin_path,
            data_path,
            security_mgr,
            servers: DashMap::new(),
            minio: Arc::new(Mutex::new(None)),
        }
    }

    #[inline]
    pub fn status_addr(&self, idx: u16) -> String {
        format!("127.0.0.1:{}", TIFLASH_PROXY_STATUS_PORT_BASE + idx)
    }

    pub fn start(&self, idx: u16, dfs: DFSConfig, pd_endpoints: &[String], role: TiFlashRole) {
        let data_dir = self.data_path.join(format!("tiflash-{idx}"));
        let log_file = self.data_path.join(format!("tiflash-{idx}.log"));
        let error_log_file = self.data_path.join(format!("tiflash-error-{idx}.log"));
        let config_file = self.data_path.join(format!("tiflash-{idx}.toml"));

        let proxy_data_dir = self.data_path.join(format!("tiflash-proxy-{idx}"));
        let proxy_log_file = self.data_path.join(format!("tiflash-proxy-{idx}.log"));
        let proxy_config_file = self.data_path.join(format!("tiflash-proxy-{idx}.toml"));

        let service_addr = format!("127.0.0.1:{}", TIFLASH_SERVICE_PORT_BASE + idx);
        let role_str = match role {
            TiFlashRole::Legacy => "".to_string(),
            TiFlashRole::Compute => "tiflash_compute".to_string(),
            TiFlashRole::Write => "tiflash_write".to_string(),
        };
        let use_columnar = match role {
            TiFlashRole::Legacy => None,
            TiFlashRole::Compute => Some(true),
            TiFlashRole::Write => Some(false),
        };
        let s3 = match role {
            TiFlashRole::Legacy => None,
            TiFlashRole::Compute | TiFlashRole::Write => Some(StorageS3Config::default()),
        };

        let config = TiFlashConfig {
            http_port: TIFLASH_HTTP_PORT_BASE + idx,
            tcp_port: TIFLASH_TCP_PORT_BASE + idx,
            flash: FlashConfig {
                service_addr: service_addr.clone(),
                disaggregated_mode: role_str,
                use_columnar,
                proxy: ProxyConfig {
                    addr: format!("127.0.0.1:{}", TIFLASH_PROXY_PORT_BASE + idx),
                    status_addr: self.status_addr(idx),
                    data_dir: proxy_data_dir.to_str().unwrap().to_owned(),
                    config: proxy_config_file.to_str().unwrap().to_owned(),
                    log_file: proxy_log_file.to_str().unwrap().to_owned(),
                },
            },
            logger: LoggerConfig {
                log: log_file.to_str().unwrap().to_owned(),
                errorlog: error_log_file.to_str().unwrap().to_owned(),
                ..Default::default()
            },
            raft: RaftConfig {
                pd_addr: pd_endpoints.join(","),
            },
            status: StatusConfig {
                metrics_port: TIFLASH_METRICS_PORT_BASE + idx,
            },
            storage: StorageConfig {
                main: StorageMainConfig {
                    dir: vec![data_dir.to_str().unwrap().to_owned()],
                },
                s3,
                ..Default::default()
            },
            ..Default::default()
        };
        let config_toml = toml::to_string(&config).unwrap();
        fs::write(&config_file, config_toml).unwrap();

        let proxy_config = TiFlashProxyConfig {
            dfs,
            server: ProxyServerConfig {
                engine_addr: service_addr,
            },
            ..Default::default()
        };
        let proxy_toml = toml::to_string(&proxy_config).unwrap();
        fs::write(&proxy_config_file, proxy_toml).unwrap();

        let mut cmd = Command::new(&self.bin_path);
        cmd.arg("server")
            .arg(format!("--config-file={}", config_file.display()));
        // Drop stdout & stderr as TiFlash will duplicate logs to them even if log file
        // is specified.
        cmd.stdout(Stdio::null()).stderr(Stdio::null());
        info!("start tiflash"; "cmd" => ?cmd);
        let child = cmd.spawn().unwrap();
        let old = self.servers.insert(idx, child);
        assert!(old.is_none(), "tiflash-{} already started", idx);
    }

    pub async fn must_healthy(&self, idx: u16, timeout: Duration) {
        let client = RestfulClient::new(
            "tiflash-ctl",
            vec![self.status_addr(idx)],
            self.security_mgr.clone(),
        )
        .unwrap();
        try_wait_result_async(
            || {
                let client = client.clone();
                Box::pin(async move {
                    match client.request("status", Method::GET, None).await {
                        Ok(_) => Ok(()),
                        Err(e) => {
                            let err = Err(format!("check TiFlash status failed: {:?}", e));
                            info!("{:?}", err);
                            err
                        }
                    }
                })
            },
            timeout.as_secs() as usize,
        )
        .await
        .unwrap_or_else(|e| panic!("wait TiFlash-{} healthy timeout: {:?}", idx, e));
        info!("TiFlash-{idx} is ready");
    }

    pub async fn must_all_healthy(&self, timeout: Duration) {
        let all = self.get_all_indexes();
        for idx in all {
            self.must_healthy(idx, timeout).await;
        }
    }

    pub fn stop(&self, idx: u16) {
        let (_, mut child) = self.servers.remove(&idx).unwrap();
        // TODO: gracefully kill by SIGINT
        child.kill().unwrap_or_else(|err| {
            panic!("TiFlash-{} has exited unexpectedly: {}", idx, err);
        })
    }

    pub fn stop_all(&self) {
        let all = self.get_all_indexes();
        for idx in all {
            self.stop(idx);
        }
        let mut minio = self.minio.lock().unwrap();
        if let Some(mut child) = minio.take() {
            child.kill().unwrap_or_else(|err| {
                panic!("MinIO has exited unexpectedly: {}", err);
            });
        }
    }

    fn get_all_indexes(&self) -> Vec<u16> {
        self.servers.iter().map(|kv| *kv.key()).collect::<Vec<_>>()
    }

    // TiFlash use XML API to access S3.
    pub fn start_minio(&self) {
        let minio_data_dir = self.data_path.join("minio");
        fs::create_dir_all(&minio_data_dir).unwrap();

        let mut cmd = Command::new("minio");
        cmd.arg("server")
            .arg("--address")
            .arg(format!(":{}", MINIO_PORT))
            .arg(minio_data_dir.display().to_string())
            .env("MINIO_BROWSER", "off"); // Disable web UI for testing

        info!("starting minio server"; "cmd" => ?cmd);

        let mut child = cmd.spawn().unwrap_or_else(|e| {
            panic!("failed to start minio server: {}", e);
        });

        // Wait a moment for MinIO to start
        std::thread::sleep(Duration::from_secs(1));

        // Verify MinIO is running
        if let Ok(None) = child.try_wait() {
            info!("minio server started successfully");
        } else {
            panic!("minio server failed to start");
        }

        let mut minio = self.minio.lock().unwrap();
        *minio = Some(child);

        // Configure mc client
        let status = Command::new("mc")
            .arg("alias")
            .arg("set")
            .arg("local") // alias name
            .arg(format!("http://127.0.0.1:{}", MINIO_PORT))
            .arg("minioadmin")
            .arg("minioadmin")
            .status()
            .unwrap_or_else(|e| panic!("failed to configure mc: {}", e));

        if !status.success() {
            panic!("failed to configure mc client");
        }

        // Create bucket if it doesn't exist
        let status = Command::new("mc")
            .arg("mb")
            .arg("--ignore-existing")
            .arg(format!("local/{}", "tiflash"))
            .status()
            .unwrap_or_else(|e| panic!("failed to create bucket: {}", e));

        if !status.success() {
            panic!("failed to create bucket {}", "tiflash");
        }

        info!("created minio bucket successfully");
    }
}

#[derive(Serialize)]
#[serde(rename_all = "snake_case")]
struct TiFlashConfig {
    default_profile: String,
    display_name: String,
    http_port: u16,
    tcp_port: u16,
    listen_host: String,
    flash: FlashConfig,
    logger: LoggerConfig,
    raft: RaftConfig,
    status: StatusConfig,
    storage: StorageConfig,
}

impl Default for TiFlashConfig {
    fn default() -> Self {
        Self {
            default_profile: "default".to_owned(),
            display_name: "TiFlash".to_owned(),
            http_port: TIFLASH_HTTP_PORT_BASE,
            tcp_port: TIFLASH_TCP_PORT_BASE,
            listen_host: "127.0.0.1".to_owned(),
            flash: FlashConfig::default(),
            logger: LoggerConfig::default(),
            raft: RaftConfig::default(),
            status: StatusConfig::default(),
            storage: StorageConfig::default(),
        }
    }
}

#[derive(Default, Serialize)]
#[serde(rename_all = "snake_case")]
struct FlashConfig {
    service_addr: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    disaggregated_mode: String,
    use_columnar: Option<bool>,
    proxy: ProxyConfig,
}

#[derive(Default, Serialize)]
#[serde(rename_all = "kebab-case")]
struct ProxyConfig {
    addr: String,
    status_addr: String,
    data_dir: String,
    config: String,
    log_file: String,
}

#[derive(Serialize)]
#[serde(rename_all = "snake_case")]
struct LoggerConfig {
    count: usize,
    errorlog: String,
    level: String,
    log: String,
    size: String,
}

impl Default for LoggerConfig {
    fn default() -> Self {
        Self {
            count: 10,
            errorlog: "error.log".to_owned(),
            log: "info.log".to_owned(),
            level: "info".to_owned(),
            size: "300M".to_owned(),
        }
    }
}

#[derive(Default, Serialize)]
#[serde(rename_all = "snake_case")]
struct RaftConfig {
    pd_addr: String,
}

#[derive(Serialize)]
#[serde(rename_all = "snake_case")]
struct StatusConfig {
    metrics_port: u16,
}

impl Default for StatusConfig {
    fn default() -> Self {
        Self {
            metrics_port: TIFLASH_SERVICE_PORT_BASE,
        }
    }
}

#[derive(Serialize)]
#[serde(rename_all = "snake_case")]
struct StorageConfig {
    api_version: u16,
    main: StorageMainConfig,
    s3: Option<StorageS3Config>,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            api_version: 2,
            main: StorageMainConfig::default(),
            s3: None,
        }
    }
}

#[derive(Default, Serialize)]
#[serde(rename_all = "snake_case")]
struct StorageMainConfig {
    dir: Vec<String>,
}

#[derive(Serialize)]
#[serde(rename_all = "snake_case")]
struct StorageS3Config {
    endpoint: String,
    access_key_id: String,
    secret_access_key: String,
    bucket: String,
    root: String,
}

impl Default for StorageS3Config {
    fn default() -> Self {
        Self {
            endpoint: format!("http://127.0.0.1:{}", MINIO_PORT),
            access_key_id: "minioadmin".to_owned(),
            secret_access_key: "minioadmin".to_owned(),
            bucket: "tiflash".to_owned(),
            root: "/".to_owned(),
        }
    }
}

#[derive(Default, Serialize)]
#[serde(rename_all = "kebab-case")]
struct TiFlashProxyConfig {
    dfs: DFSConfig,
    server: ProxyServerConfig,
    storage: ProxyStorageConfig,
    raft_engine: ProxyRaftEngineConfig,
}

#[derive(Default, Serialize)]
#[serde(rename_all = "kebab-case")]
struct ProxyServerConfig {
    engine_addr: String,
}

#[derive(Serialize)]
#[serde(rename_all = "kebab-case")]
struct ProxyStorageConfig {
    api_version: u16,
    enable_ttl: bool,
    reserve_space: String,
}

impl Default for ProxyStorageConfig {
    fn default() -> Self {
        Self {
            api_version: 2,
            enable_ttl: true,
            reserve_space: "0".to_owned(),
        }
    }
}

#[derive(Default, Serialize)]
#[serde(rename_all = "kebab-case")]
struct ProxyRaftEngineConfig {
    enable: bool,
}
