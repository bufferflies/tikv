// Copyright 2022 TiKV Project Authors. Licensed under Apache-2.0.

mod common;
mod error;
mod load_data;
mod metrics;
mod native_br;
mod remote_cop;
mod schema_manager;
mod server;
mod txn_chunk;
mod worker_limiter;
mod worker_scaler;

use std::{
    fs,
    future::Future,
    path::{Path, PathBuf},
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
    time::Duration,
};

use ::load_data::task::ResourceGroupConfig;
use ::native_br::{backup::BackupConfig, restore::RestoreConfig};
use kvengine::{
    dfs::{DFSConfig, Dfs, S3Fs},
    table::ChecksumType,
    txn_chunk_manager::{with_pool_handle, TxnChunkManager},
    BLOCK_CACHE_KEY_SIZE,
};
use kvproto::metapb::Store;
#[cfg(feature = "testexport")]
pub use metrics::REMOTE_COMPACT_REQ_HANDLE_HISTOGRAM;
use pd_client::PdClient;
pub use schema_manager::broadcast_schema_update_to_all_stores;
use schema_manager::{SchemaManager, SchemaManagerConfig};
use security::{SecurityConfig, SecurityManager};
pub use server::get_cop_req_tag;
use slog_global::{error, info};
use tikv_util::{
    config::{ReadableDuration, ReadableSize},
    quota_limiter::QuotaLimiter,
    sys::{record_global_memory_usage, SysQuota},
    time::Instant,
};
use tokio::{runtime::Runtime, task::JoinHandle};
pub use txn_chunk::CreateTxnChunkResp;

use crate::{
    load_data::{LoadDataManager, MAX_IN_MEM_SIZE},
    native_br::{NativeBrConfig, NativeBrManager},
    remote_cop::RemoteCopServer,
    txn_chunk::TxnChunkHandler,
    worker_limiter::{WorkerLimiter, WorkerLimiterConfig},
    worker_scaler::{WorkerScaler, WorkerScalerConfig, LOAD_DATA_WORKER_ENV},
};

const BACKGROUND_WORKER_INTERVAL: Duration = Duration::from_secs(60); //1min
const RG_CONFIG_PATH: &str = "resource_group/controller";

struct ServerFuture {
    http_server: Box<dyn Future<Output = hyper::Result<()>> + Send + Unpin>,
    _grpc_server: Option<RemoteCopServer>,
}

impl ServerFuture {
    pub fn new(
        http_server: Box<dyn Future<Output = hyper::Result<()>> + Send + Unpin>,
        grpc_server: Option<RemoteCopServer>,
    ) -> Self {
        ServerFuture {
            http_server,
            _grpc_server: grpc_server,
        }
    }
}

impl Future for ServerFuture {
    type Output = hyper::Result<()>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        Pin::new(&mut self.http_server).poll(cx)
    }
}

pub fn run_cloud_worker(config: Config, config_file_path: Option<PathBuf>, pd: Arc<dyn PdClient>) {
    let thread_pool = Arc::new(
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .worker_threads(SysQuota::cpu_cores_quota() as usize)
            .thread_name("worker-server")
            .build()
            .unwrap(),
    );

    let server = start_server(config, config_file_path, thread_pool.clone(), pd);
    let (tx, rx) = std::sync::mpsc::sync_channel(1);
    thread_pool.spawn(async move {
        let res = server.await;
        tx.send(res).unwrap();
    });
    rx.recv().unwrap().unwrap();
}

// `config_file_path`: optional path to the config file which cloud_worker will
// periodically reload config from it.
fn start_server(
    config: Config,
    config_file_path: Option<PathBuf>,
    thread_pool: Arc<Runtime>,
    pd: Arc<dyn PdClient>,
) -> ServerFuture {
    let dfs_config = config.dfs.clone();
    let s3fs = Arc::new(kvengine::dfs::S3Fs::new(
        dfs_config.prefix,
        dfs_config.s3_endpoint,
        dfs_config.s3_key_id,
        dfs_config.s3_secret_key,
        dfs_config.s3_region,
        dfs_config.s3_bucket,
    ));

    // CacheFs is only used for remote coprocessor.
    let cache_fs = {
        Arc::new(kvengine::dfs::CacheFs::new(
            config.cop_cache_size.0,
            s3fs.clone(),
        ))
    };

    let block_cache = if config.cop_block_cache_size.0 == 0 {
        None
    } else {
        let cache = moka::sync::SegmentedCache::builder(256)
            .weigher(|_k, v: &bytes::Bytes| (BLOCK_CACHE_KEY_SIZE + v.len()) as u32)
            .max_capacity(config.cop_block_cache_size.0)
            .build();
        Some(cache)
    };

    let addr = config.addr.parse().expect("Unable to parse socket address");

    let compression_lvl: i32 = config
        .dfs
        .zstd_compression_level
        .parse()
        .expect("Unable to parse zstd compression level");
    let checksum_type = config.checksum_type;

    let incoming = {
        let _enter = thread_pool.enter();
        hyper::server::conn::AddrIncoming::bind(&addr)
    }
    .unwrap();
    let security_mgr = pd.get_security_mgr();
    let master_key = s3fs
        .get_runtime()
        .block_on(config.security.new_master_key());
    let is_load_data_worker = std::env::var(LOAD_DATA_WORKER_ENV).is_ok();
    let mut worker_scaler_opt: Option<WorkerScaler> = None;
    if config.worker_scaler.run && !is_load_data_worker {
        let scaler_cfg = config.worker_scaler.clone();
        let cluster_id = pd.get_cluster_id().unwrap();
        let worker_scaler = thread_pool
            .block_on(WorkerScaler::new(
                &scaler_cfg,
                cluster_id,
                security_mgr.clone(),
            ))
            .unwrap();
        worker_scaler_opt = Some(worker_scaler.clone());
        thread_pool.spawn(async move {
            worker_scaler.run().await;
        });
    }

    if !config.data_dir.is_empty() {
        fs::create_dir_all(&config.data_dir).unwrap();
    }

    let rg_config = if config.report_wru {
        let rg_config = get_rg_config_from_pd(&pd, &thread_pool)
            .unwrap_or_else(|e| panic!("failed to ru config: {:?}", e));
        info!("get rg config from pd: {:?}", rg_config);
        Some(rg_config)
    } else {
        None
    };
    let load_manager = Arc::new(LoadDataManager::new(
        pd.clone(),
        config.data_dir.clone().into(),
        s3fs.clone(),
        thread_pool.clone(),
        MAX_IN_MEM_SIZE,
        checksum_type,
        master_key.clone(),
        worker_scaler_opt,
        config.worker_scaler.clone(),
        config.enable_load_data_check_point,
        rg_config,
    ));
    let br_manager = Arc::new(NativeBrManager::new(
        thread_pool.clone(),
        pd.clone(),
        s3fs.clone(),
        Some(config.data_dir.clone()),
        config.clone(),
    ));
    spawn_br_background_worker(br_manager.clone(), config_file_path);
    let txn_chunk_handler = Arc::new(TxnChunkHandler::default());

    let worker_limiter = WorkerLimiter::new(config.worker_limiter.clone());

    // Create `TxnChunkManager` using `thread_pool`. Otherwise, as `TxnChunkManager`
    // is hold in async context, we will meet the panic of dropping tokio
    // runtime in async context.
    let txn_chunk_manager = TxnChunkManager::new(
        None,
        s3fs.clone(),
        block_cache.clone(),
        with_pool_handle(thread_pool.handle().clone()),
    );

    let ctx = Arc::new(server::Context {
        compression_lvl,
        checksum_type,
        s3fs: s3fs.clone(),
        cache_fs,
        pd: pd.clone(),
        load_manager: load_manager.clone(),
        br_manager,
        txn_chunk_handler,
        master_key,
        quota_limiter: Arc::new(QuotaLimiter::default()),
        block_cache,
        worker_limiter,
        txn_chunk_manager,
    });
    let acceptor = security_mgr.acceptor(incoming).unwrap();
    let server = start_serve!(ctx.clone(), acceptor);

    if config.schema_manager.enabled {
        let schema_manager = SchemaManager::new(
            ctx.clone(),
            security_mgr.clone(),
            config.schema_manager.clone(),
            config.pd.endpoints.as_ref(),
        );
        schema_manager.run(thread_pool.clone());
    }

    // try recover task from checkpoint
    load_manager.try_recover_or_clean_tasks_by_check_point();

    if config.register {
        let remote_compact_url = security_mgr
            .build_uri(format!("{}/compact", config.addr))
            .unwrap()
            .to_string();
        let duration = config.update_interval.0;
        std::thread::spawn(move || {
            loop {
                register_compactor_to_all_stores(
                    pd.clone(),
                    s3fs.clone(),
                    remote_compact_url.clone(),
                    security_mgr.clone(),
                );
                std::thread::sleep(duration);
            }
        });
    }

    let mut cop_server_opt = None;
    if !config.cop_addr.is_empty() {
        let cop_config = remote_cop::Config {
            addr: config.cop_addr,
            max_handle_duration: Duration::from_secs(60),
        };
        let cop_server = remote_cop::RemoteCopServer::new(ctx, cop_config);
        cop_server_opt = Some(cop_server);
    }
    if let Some(server) = cop_server_opt.as_mut() {
        server.start();
        info!("remote cop server started");
    }

    thread_pool.spawn(async move {
        loop {
            tokio::time::sleep(Duration::from_millis(100)).await;
            record_global_memory_usage();
        }
    });

    ServerFuture::new(server, cop_server_opt)
}

pub struct CloudWorker {
    config: Config,
    config_file_path: Option<PathBuf>,
    thread_pool: Arc<Runtime>,
    pd: Arc<dyn PdClient>,

    svc_handle: Option<JoinHandle<()>>,
    notify: Arc<tokio::sync::Notify>,
}

impl CloudWorker {
    pub fn new(
        config: Config,
        config_file_path: Option<PathBuf>,
        threads_cnt: usize,
        pd: Arc<dyn PdClient>,
    ) -> Self {
        let thread_pool = Arc::new(
            tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .worker_threads(threads_cnt)
                .thread_name("worker-server")
                .build()
                .unwrap(),
        );
        CloudWorker {
            config,
            config_file_path,
            thread_pool,
            pd,
            svc_handle: None,
            notify: Arc::new(tokio::sync::Notify::new()),
        }
    }

    pub fn addr(&self) -> &str {
        self.config.addr.as_str()
    }

    pub fn start(&mut self) {
        let server = start_server(
            self.config.clone(),
            self.config_file_path.clone(),
            self.thread_pool.clone(),
            self.pd.clone(),
        );
        let addr = self.addr().to_string();
        info!("{} cloud_worker server start", addr; "config" => ?self.config);

        let notify = self.notify.clone();
        let svc_handle = self.thread_pool.spawn(async move {
            tokio::select! {
                _ = notify.notified() => {
                    info!("{} cloud_worker server shutdown", addr);
                }
                res = server => {
                    if let Err(e) = res {
                        error!("{} cloud_worker server error: {:?}", addr, e);
                    } else {
                        info!("{} cloud_worker server graceful shutdown", addr);
                    }
                }
            }
        });
        self.svc_handle = Some(svc_handle);
    }

    pub fn shutdown(mut self) {
        if let Some(handle) = self.svc_handle.take() {
            self.notify.notify_waiters();
            self.thread_pool.block_on(async { handle.await.unwrap() })
        }
    }
}

fn spawn_br_background_worker(br_manager: Arc<NativeBrManager>, path: Option<PathBuf>) {
    std::thread::spawn(move || {
        info!("start br background worker");
        loop {
            if let Some(path) = path.as_ref() {
                update_native_br_config(br_manager.clone(), path.as_path());
            }

            br_manager.cleanup_expired_restores();

            std::thread::sleep(BACKGROUND_WORKER_INTERVAL);
        }
    });
}

fn update_native_br_config(br_manager: Arc<NativeBrManager>, path: &Path) {
    match std::fs::read(path) {
        Ok(data) => match toml::from_slice::<Config>(&data) {
            Ok(config) => br_manager.update_native_br_config(config.native_br),
            Err(e) => error!("failed to parse config file {:?}", e),
        },
        Err(e) => error!("failed to read config file {:?}", e),
    }
}

fn get_rg_config_from_pd(
    pd_client: &Arc<dyn PdClient>,
    runtime: &Arc<tokio::runtime::Runtime>,
) -> Result<ResourceGroupConfig, error::Error> {
    let configs =
        runtime.block_on(pd_client.load_global_config_by_path(RG_CONFIG_PATH.to_string()))?;
    if let Some(config) = configs.get(RG_CONFIG_PATH) {
        let rg_config = serde_json::from_slice(config)?;
        Ok(rg_config)
    } else {
        Err(error::Error::PdError(
            pd_client::Error::GlobalConfigNotFound("ru config not found".to_string()),
        ))
    }
}

pub(crate) fn get_all_stores_except_tiflash(
    pd_client: &Arc<dyn PdClient>,
) -> Result<Vec<Store>, pd_client::Error> {
    Ok(pd_client
        .get_all_stores(true)?
        .into_iter()
        .filter(|s| {
            !s.get_labels().iter().any(|l| {
                // including "tiflash" & "tiflash_compute"
                l.key.to_lowercase() == "engine" && l.value.to_lowercase().starts_with("tiflash")
            })
        })
        .collect())
}

fn register_compactor_to_all_stores(
    pd: Arc<dyn PdClient>,
    dfs: Arc<S3Fs>,
    remote_url: String,
    security_mgr: Arc<SecurityManager>,
) {
    let all_stores = match get_all_stores_except_tiflash(&pd) {
        Ok(stores) => stores,
        Err(e) => {
            error!("failed to get all stores {:?}", e);
            return;
        }
    };
    let start_time = Instant::now();
    let stores_len = all_stores.len();
    let (tx, rx) = std::sync::mpsc::sync_channel(stores_len);
    for store in all_stores {
        let tx = tx.clone();
        let remote_url = remote_url.clone();
        let security_mgr = security_mgr.clone();
        dfs.get_runtime().spawn(async move {
            tx.send(register_compactor_to_store(store, remote_url.clone(), security_mgr).await)
                .unwrap();
        });
    }
    let mut finish_cnt = 0;
    for _ in 0..stores_len {
        let ok = rx.recv().unwrap();
        if ok {
            finish_cnt += 1;
        }
    }
    let elapsed = start_time.saturating_elapsed();
    let remain = stores_len - finish_cnt;
    info!(
        "register compactor to {} stores in {:?}, remain {} stores",
        stores_len, elapsed, remain
    );
}

async fn register_compactor_to_store(
    store: Store,
    remote_url: String,
    security_mgr: Arc<SecurityManager>,
) -> bool {
    let uri = security_mgr
        .build_uri(format!("{}/kvengine/compactor", store.status_address))
        .unwrap();
    let client = security_mgr.http_client(hyper::Client::builder()).unwrap();
    let req = hyper::Request::builder()
        .method(hyper::Method::POST)
        .uri(uri)
        .body(hyper::Body::from(remote_url))
        .expect("request builder");
    match client.request(req).await {
        Ok(resp) => {
            let ok = resp.status() == hyper::StatusCode::OK;
            if !ok {
                error!(
                    "failed to register compactor to store";
                    "store" => ?store, "status" => ?resp.status()
                );
            }
            ok
        }
        Err(err) => {
            error!("failed to register compactor to store"; "store" => ?store, "err" => ?err);
            false
        }
    }
}

#[macro_use]
extern crate serde_derive;

#[derive(Clone, Serialize, Deserialize, PartialEq, Debug)]
#[serde(default)]
#[serde(rename_all = "kebab-case")]
pub struct Config {
    pub addr: String,
    pub pd: pd_client::Config,
    pub security: SecurityConfig,
    pub update_interval: ReadableDuration,
    pub dfs: DFSConfig,
    pub log_file: String,
    pub log_level: String,
    pub data_dir: String,
    pub register: bool,
    pub native_br: NativeBrConfig,
    pub cop_addr: String,
    pub cop_cache_size: ReadableSize,
    pub cop_block_cache_size: ReadableSize,
    pub worker_scaler: WorkerScalerConfig,
    pub report_wru: bool,
    pub enable_load_data_check_point: bool,
    pub checksum_type: ChecksumType,
    pub worker_limiter: WorkerLimiterConfig,
    pub schema_manager: SchemaManagerConfig,
}

impl Default for Config {
    fn default() -> Self {
        let mut pd = pd_client::Config::default();
        pd.endpoints.clear();
        Config {
            addr: String::from("0.0.0.0:19000"),
            pd,
            security: SecurityConfig::default(),
            update_interval: ReadableDuration::minutes(10),
            dfs: DFSConfig::default(),
            log_file: String::default(),
            log_level: String::default(),
            data_dir: String::default(),
            register: false,
            native_br: NativeBrConfig::default(),
            cop_addr: String::from("0.0.0.0:9500"),
            cop_cache_size: ReadableSize::gb(1),
            cop_block_cache_size: ReadableSize::default(),
            worker_scaler: WorkerScalerConfig::default(),
            report_wru: false,
            enable_load_data_check_point: false,
            checksum_type: ChecksumType::Crc32c,
            worker_limiter: WorkerLimiterConfig::default(),
            schema_manager: SchemaManagerConfig::default(),
        }
    }
}

impl Config {
    pub fn to_backup_config(&self) -> BackupConfig {
        let tolerate_err = usize::from(self.native_br.backup_tolerate_err);
        BackupConfig {
            pd: self.pd.clone(),
            security: self.security.clone(),
            dfs: self.dfs.clone(),
            tolerate_err,
            skip_keyspace_meta: false,
        }
    }
    pub fn to_restore_config(&self) -> RestoreConfig {
        RestoreConfig {
            pd: self.pd.clone(),
            security: self.security.clone(),
            dfs: self.dfs.clone(),
            // resolve_lock is to solve the Async Commit locks. Async Commit is not supported
            // for now. TODO: Change it to false when Async Commit is enabled, and impl resolve
            // locks in restore_keyspace.
            skip_resolve_lock: true,
            timeout_wait_flush: self.native_br.restore_timeout_wait_flush,
            timeout_restore_snapshot: self.native_br.restore_timeout_restore_snapshot,
            timeout_fetch_wal: self.native_br.restore_timeout_fetch_wal,
            max_retry: self.native_br.restore_max_retry,
            ..Default::default()
        }
    }

    pub fn validate(&self) -> Result<(), Box<dyn std::error::Error>> {
        self.worker_limiter.validate()?;
        Ok(())
    }
}
