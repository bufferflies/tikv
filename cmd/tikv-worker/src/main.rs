// Copyright 2022 TiKV Project Authors. Licensed under Apache-2.0.

mod common;
mod error;
mod load_data;
mod metrics;
mod native_br;
mod remote_cop;
mod worker_scaler;

use std::{
    io,
    path::{Path, PathBuf},
    str::FromStr,
    sync::Arc,
    time::Duration,
};

use ::native_br::{backup::BackupConfig, restore::RestoreConfig};
use clap::{App, Arg, ArgMatches};
use cloud_encryption::MasterKey;
use flate2::{write::GzEncoder, Compression};
use grpcio::EnvBuilder;
use http::{
    header::{ACCEPT_ENCODING, CONTENT_ENCODING, CONTENT_TYPE},
    HeaderValue, Request, Response, Uri,
};
use hyper::{
    body::Buf,
    service::{make_service_fn, service_fn},
    Body,
};
use kvengine::{
    dfs,
    dfs::{DFSConfig, Dfs, S3Fs},
    SnapAccess,
};
use kvproto::metapb::Store;
use pd_client::{PdClient, RpcClient};
use prometheus::TEXT_FORMAT;
use protobuf::Message;
use rfstore::store::{PdIdAllocator, RegionSnapshot};
use security::{SecurityConfig, SecurityManager};
use slog::Level;
use slog_global::{error, info};
use tikv::{
    coprocessor::remote_dispatcher::decode_remote_cop_request, server::status_server::StatusServer,
};
use tikv_util::{
    config::{ReadableDuration, ReadableSize},
    metrics::{dump, dump_to},
    quota_limiter::QuotaLimiter,
    sys::SysQuota,
    time::Instant,
};

use crate::{
    load_data::{handle_load_data, LoadDataManager, MAX_IN_MEM_SIZE},
    metrics::CPU_CORES_QUOTA_GAUGE,
    native_br::{NativeBrConfig, NativeBrManager},
    worker_scaler::{WorkerScaler, WorkerScalerConfig, LOAD_DATA_WORKER_ENV},
};

const ZSTD_COMPRESSION_LEVEL_FOR_REMOTE: &str = "5";
const DEFAULT_LOG_LEVEL: Level = Level::Info;
const BACKGROUND_WORKER_INTERVAL: Duration = Duration::from_secs(60); //1min

fn main() {
    init_logger(io::stdout(), DEFAULT_LOG_LEVEL);
    tikv_util::metrics::monitor_process()
        .unwrap_or_else(|e| panic!("failed to start process monitor: {}", e));
    CPU_CORES_QUOTA_GAUGE.set(SysQuota::cpu_cores_quota());
    let matches = App::new("tikv-worker")
        .about("tikv remote worker")
        .arg(
            Arg::with_name("config")
                .short("C")
                .long("config")
                .value_name("FILE")
                .help("Set the configuration file")
                .takes_value(true)
                .required(false),
        )
        .arg(
            Arg::with_name("addr")
                .short("A")
                .long("addr")
                .takes_value(true)
                .value_name("IP:PORT")
                .help("Set the listening address"),
        )
        .arg(
            Arg::with_name("log-file")
                .short("f")
                .long("log-file")
                .value_name("LOGFILE")
                .help("Sets log file")
                .long_help("Set the log file path. If not set, logs will output to stderr"),
        )
        .arg(
            Arg::with_name("log-level")
                .short("L")
                .long("log-level")
                .value_name("LOGLEVEL")
                .help("Sets log level")
                .long_help("Set the log level [debug,info,warn,error]"),
        )
        .arg(
            Arg::with_name("pd-endpoints")
                .long("pd-endpoints")
                .takes_value(true)
                .value_name("PD_URL")
                .multiple(true)
                .use_delimiter(true)
                .require_delimiter(true)
                .value_delimiter(",")
                .help("Sets PD endpoints")
                .long_help("Set the PD endpoints to use. Use `,` to separate multiple PDs"),
        )
        .arg(
            Arg::with_name("cacert")
                .long("cacert")
                .takes_value(true)
                .value_name("CERT")
                .help("Path of file that contains list of trusted SSL CAs"),
        )
        .arg(
            Arg::with_name("cert")
                .long("cert")
                .takes_value(true)
                .value_name("CERT")
                .help("Path of file that contains X509 certificate in PEM format"),
        )
        .arg(
            Arg::with_name("key")
                .long("key")
                .takes_value(true)
                .value_name("KEY")
                .help("Path of file that contains X509 key in PEM format"),
        )
        .arg(
            Arg::with_name("update-interval")
                .long("update-interval")
                .takes_value(true)
                .value_name("INTERVAL")
                .help("Sets registration update interval"),
        )
        .arg(
            Arg::with_name("register")
                .long("register")
                .takes_value(true)
                .value_name("Bool")
                .help("register compactor to stores"),
        )
        .arg(
            Arg::with_name("data-dir")
                .long("data-dir")
                .takes_value(true)
                .value_name("DIR")
                .help("data dir for load_data"),
        )
        .arg(
            Arg::with_name("cop-addr")
                .long("cop-addr")
                .takes_value(true)
                .value_name("IP:PORT")
                .help("Set the coprocessor listening address"),
        )
        .arg(
            Arg::with_name("run-worker-scaler")
                .long("run-worker-scaler")
                .takes_value(true)
                .value_name("Bool")
                .help("run worker-scaler"),
        )
        .get_matches();

    let mut config_file_path = None;
    let mut config: Config = match matches.value_of_os("config") {
        Some(config_path) => {
            let path = PathBuf::from(config_path);
            config_file_path = Some(path.clone());
            let result = std::fs::read(path);
            if result.is_err() {
                error!("failed to read config file {:?}", result.unwrap_err());
                return;
            }
            let data = result.unwrap();
            toml::from_slice(&data).unwrap()
        }
        None => Config::default(),
    };

    override_from_args(&mut config, &matches);
    let log_level =
        tikv_util::logger::get_level_by_string(&config.log_level).unwrap_or(DEFAULT_LOG_LEVEL);
    if !config.log_file.is_empty() {
        let log = tikv_util::logger::file_writer(&config.log_file, 300, 0, 0, rename_by_timestamp)
            .unwrap();
        init_logger(log, log_level);
    } else if log_level != DEFAULT_LOG_LEVEL {
        init_logger(io::stdout(), log_level);
    }
    config.dfs.override_from_env();
    config.security.master_key.override_from_env();

    // If zstd_compression_level is not set, set it to default value
    if config.dfs.zstd_compression_level.is_empty() {
        config.dfs.zstd_compression_level = ZSTD_COMPRESSION_LEVEL_FOR_REMOTE.to_string();
    }

    info!("config is {:?}", &config);
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

    let thread_pool = Arc::new(
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .worker_threads(SysQuota::cpu_cores_quota() as usize)
            .thread_name("worker-server")
            .build()
            .unwrap(),
    );
    let addr = config.addr.parse().expect("Unable to parse socket address");

    let compression_lvl: i32 = config
        .dfs
        .zstd_compression_level
        .parse()
        .expect("Unable to parse zstd compression level");

    let incoming = {
        let _enter = thread_pool.enter();
        hyper::server::conn::AddrIncoming::bind(&addr)
    }
    .unwrap();
    let security_mgr = Arc::new(
        SecurityManager::new(&config.security)
            .unwrap_or_else(|e| panic!("failed to create security manager: {:?}", e)),
    );
    let master_key = s3fs
        .get_runtime()
        .block_on(config.security.new_master_key());
    let env = Arc::new(EnvBuilder::new().cq_count(1).build());
    let pd = Arc::new(
        RpcClient::new(&config.pd, Some(env), security_mgr)
            .unwrap_or_else(|e| panic!("failed to create rpc client: {:?}", e)),
    );
    let is_load_data_worker = std::env::var(LOAD_DATA_WORKER_ENV).is_ok();
    let mut worker_scaler_opt: Option<WorkerScaler> = None;
    if config.worker_scaler.run && !is_load_data_worker {
        let scaler_cfg = config.worker_scaler.clone();
        let cluster_id = pd.get_cluster_id().unwrap();
        let worker_scaler = thread_pool
            .block_on(WorkerScaler::new(&scaler_cfg, cluster_id))
            .unwrap();
        worker_scaler_opt = Some(worker_scaler.clone());
        thread_pool.spawn(async move {
            worker_scaler.run().await;
        });
    }
    let load_manager = Arc::new(LoadDataManager::new(
        pd.clone(),
        config.data_dir.clone().into(),
        s3fs.clone(),
        thread_pool.clone(),
        MAX_IN_MEM_SIZE,
        master_key.clone(),
        worker_scaler_opt,
        config.worker_scaler.clone(),
    ));
    let br_manager = Arc::new(NativeBrManager::new(
        thread_pool.clone(),
        pd.clone(),
        s3fs.clone(),
        Some(config.data_dir.clone()),
        config.clone(),
    ));
    spawn_br_background_worker(br_manager.clone(), config_file_path);

    let server_builder = hyper::Server::builder(incoming);
    let s3fs_clone = s3fs.clone();
    let cache_fs_clone = cache_fs.clone();
    let pd_clone = pd.clone();
    let master_key_clone = master_key.clone();
    let quota_limiter = Arc::new(QuotaLimiter::default());
    let server = server_builder.serve(make_service_fn(move |_| {
        let s3fs = s3fs_clone.clone();
        let cache_fs = cache_fs_clone.clone();
        let load_manager = load_manager.clone();
        let br_manager = br_manager.clone();
        let pd = pd_clone.clone();
        let master_key = master_key_clone.clone();
        let quota_limiter = quota_limiter.clone();
        async move {
            // Create a status service.
            Ok::<_, hyper::Error>(service_fn(move |req: hyper::Request<hyper::Body>| {
                let s3fs = s3fs.clone();
                let cache_fs = cache_fs.clone();
                let load_manager = load_manager.clone();
                let br_manager = br_manager.clone();
                let pd = pd.clone();
                let master_key = master_key.clone();
                let quota_limiter = quota_limiter.clone();
                async move {
                    let path = req.uri().path().to_owned();
                    match path.as_ref() {
                        "/healthz" => Ok(hyper::Response::builder()
                            .status(200)
                            .body(hyper::Body::from("ok"))
                            .unwrap()),
                        "/compact" => {
                            let allocator = Arc::new(PdIdAllocator::new(pd));
                            kvengine::handle_remote_compaction(
                                s3fs,
                                req,
                                compression_lvl,
                                allocator,
                                master_key,
                            )
                            .await
                        }
                        "/analyze" => handle_remote_analysis(cache_fs, req, master_key).await,
                        "/coprocessor" => {
                            handle_remote_coprocessor(cache_fs, req, master_key, quota_limiter)
                                .await
                        }
                        "/load_data" => handle_load_data(load_manager, req).await,
                        "/metrics" => handle_get_metrics(req).await,
                        "/debug/pprof/profile" => {
                            StatusServer::<u8, u8>::dump_cpu_prof_to_resp(req).await
                        }
                        native_br::BACKUPS_API_PATH => {
                            native_br::handle_backup(br_manager, req).await
                        }
                        path if path.starts_with(native_br::RESTORE_KEYSPACE_API_PATH) => {
                            native_br::handle_restore_keyspace(br_manager, req).await
                        }
                        path if path.starts_with(native_br::WHITELIST_API_PATH) => {
                            native_br::handle_native_br_whitelist(br_manager, req).await
                        }
                        _ => Ok(hyper::Response::builder()
                            .status(404)
                            .body(hyper::Body::from("Not Found"))
                            .unwrap()),
                    }
                }
            }))
        }
    }));

    if config.register {
        let remote_compact_url = format!("http://{}/compact", config.addr);
        let duration = config.update_interval.0;
        let pd_clone = pd.clone();
        std::thread::spawn(move || {
            loop {
                register_compactor_to_all_stores(
                    pd_clone.clone(),
                    s3fs.clone(),
                    remote_compact_url.clone(),
                );
                std::thread::sleep(duration);
            }
        });
    }

    let mut cop_server_opt = None;
    if !config.cop_addr.is_empty() {
        let cop_config = remote_cop::Config {
            addr: config.cop_addr.clone(),
            max_handle_duration: Duration::from_secs(60),
        };
        let cop_server = remote_cop::RemoteCopServer::new(pd, cache_fs, cop_config, master_key);
        cop_server_opt = Some(cop_server);
    }
    if let Some(server) = cop_server_opt.as_mut() {
        server.start();
        info!("remote cop server started");
    }
    let (tx, rx) = std::sync::mpsc::sync_channel(1);
    thread_pool.spawn(async move {
        let res = server.await;
        tx.send(res).unwrap();
    });
    rx.recv().unwrap().unwrap();
}

fn init_logger<W: 'static + io::Write + Send>(writer: W, level: Level) {
    use slog::Drain;
    let decorator = slog_term::PlainDecorator::new(writer);
    let drain = slog_term::CompactFormat::new(decorator).build();
    let drain = std::sync::Mutex::new(drain).filter_level(level).fuse();
    let logger = slog::Logger::root(drain, slog::o!());
    slog_global::set_global(logger);
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

async fn handle_get_metrics(req: Request<Body>) -> hyper::Result<Response<Body>> {
    let gz_encoding = client_accept_gzip(&req);
    let metrics = if gz_encoding {
        // gzip can reduce the body size to less than 1/10.
        let mut encoder = GzEncoder::new(vec![], Compression::default());
        dump_to(&mut encoder, true);
        encoder.finish().unwrap()
    } else {
        dump(true).into_bytes()
    };
    let mut resp = Response::new(metrics.into());
    resp.headers_mut()
        .insert(CONTENT_TYPE, HeaderValue::from_static(TEXT_FORMAT));
    if gz_encoding {
        resp.headers_mut()
            .insert(CONTENT_ENCODING, HeaderValue::from_static("gzip"));
    }
    Ok(resp)
}

// check if the client allow return response with gzip compression
// the following logic is port from prometheus's golang:
// https://github.com/prometheus/client_golang/blob/24172847e35ba46025c49d90b8846b59eb5d9ead/prometheus/promhttp/http.go#L155-L176
fn client_accept_gzip(req: &Request<Body>) -> bool {
    let encoding = req
        .headers()
        .get(ACCEPT_ENCODING)
        .map(|enc| enc.to_str().unwrap_or_default())
        .unwrap_or_default();
    encoding
        .split(',')
        .map(|s| s.trim())
        .any(|s| s == "gzip" || s.starts_with("gzip;"))
}

async fn handle_remote_analysis(
    dfs: Arc<dyn dfs::Dfs>,
    req: hyper::Request<hyper::Body>,
    master_key: MasterKey,
) -> hyper::Result<hyper::Response<hyper::Body>> {
    let mut start_time = Instant::now();
    let req_body = hyper::body::to_bytes(req.into_body()).await?;
    let result = serde_json::from_slice(req_body.chunk());
    if result.is_err() {
        let err_str = result.unwrap_err().to_string();
        return Ok(hyper::Response::builder()
            .status(400)
            .body(err_str.into())
            .unwrap());
    }
    let remote_req: tikv::coprocessor::RemoteAnalysisRequest = result.unwrap();
    let tag = format!("[:{}]", remote_req.key);
    let mut change_set = kvenginepb::ChangeSet::default();
    change_set.merge_from_bytes(&remote_req.snap_bytes).unwrap();
    let snap_access = SnapAccess::from_change_set(dfs, change_set, true, &master_key).await;
    info!(
        "start analyzing for {}, prepare snap time {:?}",
        tag,
        start_time.saturating_elapsed()
    );
    start_time = Instant::now();
    let snap = RegionSnapshot::from_snapshot(snap_access);
    let (tx, rx) = tokio::sync::oneshot::channel();
    std::thread::spawn(move || {
        let result =
            tikv::coprocessor::parse_request_and_remote_analyze::<RegionSnapshot>(remote_req, snap);
        tx.send(result).unwrap();
    });
    match rx.await.unwrap() {
        Ok(data) => {
            info!(
                "finish analyzing for {}, takes {:?}, data size {}",
                tag,
                start_time.saturating_elapsed(),
                data.len(),
            );
            Ok(hyper::Response::builder()
                .status(200)
                .body(data.into())
                .unwrap())
        }
        Err(err) => {
            let err_str = format!("{:?}", err);
            error!("failed to analyze for {}, error {}", tag, err_str);
            let body = hyper::Body::from(err_str);
            Ok(hyper::Response::builder().status(500).body(body).unwrap())
        }
    }
}

async fn handle_remote_coprocessor(
    dfs: Arc<dyn dfs::Dfs>,
    req: hyper::Request<hyper::Body>,
    master_key: MasterKey,
    quota_limiter: Arc<QuotaLimiter>,
) -> hyper::Result<hyper::Response<hyper::Body>> {
    let req_body = hyper::body::to_bytes(req.into_body()).await?;
    let decode_res = decode_remote_cop_request(req_body.chunk());
    if let Err(err) = decode_res {
        let body = hyper::Body::from(format!("{:?}", err));
        return Ok(hyper::Response::builder().status(500).body(body).unwrap());
    }
    let (req_data, mem_data, snap_data) = decode_res.unwrap();
    let mut cop_req = kvproto::coprocessor::Request::default();
    if let Err(err) = cop_req.merge_from_bytes(req_data) {
        let body = hyper::Body::from(format!("{:?}", err));
        return Ok(hyper::Response::builder().status(500).body(body).unwrap());
    }
    let snap_access_res =
        SnapAccess::construct_snapshot(dfs, mem_data, snap_data, &master_key).await;
    if let Err(err) = snap_access_res.as_ref() {
        let body = hyper::Body::from(format!("{:?}", err));
        return Ok(hyper::Response::builder().status(500).body(body).unwrap());
    }
    let snap_access = snap_access_res.unwrap();
    let tag = format!(
        "ks{}:{}:{}",
        snap_access.get_keyspace_id(),
        snap_access.get_id(),
        snap_access.get_version()
    );
    let snap = RegionSnapshot::from_snapshot(snap_access);
    let result = tikv::coprocessor::parse_request_and_handle_remote_cop(
        cop_req,
        None,
        Duration::from_secs(60),
        quota_limiter,
        snap,
    )
    .await;
    if let Err(err) = result {
        error!("{} remote coprocessor failed, error {:?}", tag, err);
        let body = hyper::Body::from(format!("{:?}", err));
        return Ok(hyper::Response::builder().status(500).body(body).unwrap());
    }
    let response = result.unwrap();
    info!(
        "{} finished remote coprocessor resp size {}",
        tag,
        response.data.len()
    );
    let response_data = response.write_to_bytes().unwrap();
    Ok(hyper::Response::builder()
        .status(200)
        .body(response_data.into())
        .unwrap())
}

pub(crate) fn get_all_stores_except_tiflash(
    pd_client: &Arc<RpcClient>,
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

fn register_compactor_to_all_stores(pd: Arc<RpcClient>, dfs: Arc<S3Fs>, remote_url: String) {
    let all_stores = get_all_stores_except_tiflash(&pd)
        .unwrap_or_else(|e| panic!("failed get all stores {:?}", e));
    let start_time = Instant::now();
    let stores_len = all_stores.len();
    let (tx, rx) = std::sync::mpsc::sync_channel(stores_len);
    for store in all_stores {
        let tx = tx.clone();
        let remote_url = remote_url.clone();
        dfs.get_runtime().spawn(async move {
            tx.send(register_compactor_to_store(store, remote_url.clone()).await)
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

async fn register_compactor_to_store(store: Store, remote_url: String) -> bool {
    let uri = Uri::from_str(&format!(
        "http://{}/kvengine/compactor",
        &store.status_address
    ))
    .unwrap();
    let client = hyper::Client::new();
    let req = hyper::Request::builder()
        .method(hyper::Method::POST)
        .uri(uri)
        .body(hyper::Body::from(remote_url))
        .expect("request builder");
    let resp = client.request(req).await.unwrap();
    resp.status() == hyper::StatusCode::OK
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
    pub worker_scaler: WorkerScalerConfig,
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
            worker_scaler: WorkerScalerConfig::default(),
        }
    }
}

impl Config {
    pub fn to_backup_config(&self) -> BackupConfig {
        BackupConfig {
            pd: self.pd.clone(),
            security: self.security.clone(),
            dfs: self.dfs.clone(),
            tolerate_err: 0,
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
        }
    }
}

fn rename_by_timestamp(path: &Path) -> io::Result<PathBuf> {
    let mut new_path = path.parent().unwrap().to_path_buf();
    let mut new_fname = path.file_stem().unwrap().to_os_string();
    let dt = chrono::Local::now().format("%Y-%m-%dT%H-%M-%S%.3f");
    new_fname.push(format!("-{}", dt));
    if let Some(ext) = path.extension() {
        new_fname.push(".");
        new_fname.push(ext);
    };
    new_path.push(new_fname);
    Ok(new_path)
}

fn override_from_args(config: &mut Config, matches: &ArgMatches<'_>) {
    if let Some(file) = matches.value_of("log-file") {
        config.log_file = file.to_owned();
    }

    if let Some(addr) = matches.value_of("addr") {
        config.addr = addr.to_owned();
    }

    if let Some(endpoints) = matches.values_of("pd-endpoints") {
        config.pd.endpoints = endpoints.map(ToOwned::to_owned).collect();
    }

    if let Some(cacert) = matches.value_of("cacert") {
        config.security.ca_path = cacert.to_owned();
    }

    if let Some(cert) = matches.value_of("cert") {
        config.security.cert_path = cert.to_owned();
    }

    if let Some(key) = matches.value_of("key") {
        config.security.key_path = key.to_owned();
    }

    if let Some(interval) = matches.value_of("interval") {
        config.update_interval = ReadableDuration::secs(interval.parse().unwrap());
    }

    if let Some(register) = matches.value_of("register") {
        config.register = register == "true";
    }

    if let Some(dir) = matches.value_of("data-dir") {
        config.data_dir = dir.to_string();
    }

    if let Some(log_file) = matches.value_of("log-file") {
        config.log_file = log_file.to_string();
    }

    if let Some(log_level) = matches.value_of("log-level") {
        config.log_level = log_level.to_string();
    }

    if let Some(cop_addr) = matches.value_of("cop-addr") {
        config.cop_addr = cop_addr.to_string();
    }

    if let Some(run_worker_scaler) = matches.value_of("run-worker-scaler") {
        config.worker_scaler.run = run_worker_scaler == "true";
    }
}
