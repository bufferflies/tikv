// Copyright 2022 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    io,
    path::{Path, PathBuf},
    str::FromStr,
    sync::Arc,
};

use clap::{App, Arg, ArgMatches};
use grpcio::EnvBuilder;
use http::Uri;
use hyper::{
    body::Buf,
    service::{make_service_fn, service_fn},
};
use kvengine::{
    dfs,
    dfs::{DFSConfig, DFS, S3FS},
    SnapAccess,
};
use kvproto::metapb::Store;
use pd_client::{PdClient, RpcClient};
use protobuf::Message;
use rfstore::store::RegionSnapshot;
use security::{SecurityConfig, SecurityManager};
use slog_global::{error, info};
use tikv_util::{config::ReadableDuration, time::Instant};

const ZSTD_COMPRESSION_LEVEL_FOR_REMOTE: &str = "5";

fn main() {
    init_logger(io::stdout());
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
        .get_matches();

    let mut config: Config = match matches.value_of_os("config") {
        Some(config_path) => {
            let result = std::fs::read(PathBuf::from(config_path));
            if result.is_err() {
                error!("failed to read config file {:?}", result.unwrap_err());
                return;
            }
            let data = result.unwrap();
            toml::from_slice(&data).unwrap()
        }
        None => Config::default(),
    };

    if !config.log_file.is_empty() {
        let log = tikv_util::logger::file_writer(&config.log_file, 300, 0, 0, rename_by_timestamp)
            .unwrap();
        init_logger(log);
    }
    override_from_args(&mut config, &matches);
    config.dfs.override_from_env();

    // If zstd_compression_level is not set, set it to default value
    if config.dfs.zstd_compression_level.is_empty() {
        config.dfs.zstd_compression_level = ZSTD_COMPRESSION_LEVEL_FOR_REMOTE.to_string();
    }

    info!("config is {:?}", &config);
    let dfs = Arc::new(kvengine::dfs::S3FS::new(
        config.dfs.prefix,
        config.dfs.s3_endpoint,
        config.dfs.s3_key_id,
        config.dfs.s3_secret_key,
        config.dfs.s3_region,
        config.dfs.s3_bucket,
    ));
    let dfs_clone = dfs.clone();
    let thread_pool = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .worker_threads(16)
        .thread_name("worker-server")
        .build()
        .unwrap();
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
    let server_builder = hyper::Server::builder(incoming);
    let server = server_builder.serve(make_service_fn(move |_| {
        let dfs = dfs.clone();
        async move {
            // Create a status service.
            Ok::<_, hyper::Error>(service_fn(move |req: hyper::Request<hyper::Body>| {
                let dfs = dfs.clone();
                async move {
                    match req.uri().path() {
                        "/healthz" => Ok(hyper::Response::builder()
                            .status(200)
                            .body(hyper::Body::from("ok"))
                            .unwrap()),
                        "/compact" => {
                            kvengine::handle_remote_compaction(dfs, req, compression_lvl).await
                        }
                        "/analyze" => handle_remote_analysis(dfs, req).await,
                        _ => Ok(hyper::Response::builder()
                            .status(404)
                            .body(hyper::Body::from("Not Found"))
                            .unwrap()),
                    }
                }
            }))
        }
    }));

    if !config.pd.endpoints.is_empty() {
        let security_mgr = Arc::new(
            SecurityManager::new(&config.security)
                .unwrap_or_else(|e| panic!("failed to create security manager: {:?}", e)),
        );
        let env = Arc::new(EnvBuilder::new().cq_count(1).build());
        let pd_client = Arc::new(
            RpcClient::new(&config.pd, Some(env), security_mgr)
                .unwrap_or_else(|e| panic!("failed to create rpc client: {:?}", e)),
        );
        let remote_compact_url = format!("http://{}/compact", config.addr);
        let duration = config.update_interval.0;
        std::thread::spawn(move || {
            loop {
                register_compactor_to_all_stores(
                    pd_client.clone(),
                    dfs_clone.clone(),
                    remote_compact_url.clone(),
                );
                std::thread::sleep(duration);
            }
        });
    }

    let (tx, rx) = std::sync::mpsc::sync_channel(1);
    thread_pool.spawn(async move {
        let res = server.await;
        tx.send(res).unwrap();
    });
    rx.recv().unwrap().unwrap();
}

fn init_logger<W: 'static + io::Write + Send>(writer: W) {
    use slog::Drain;
    let decorator = slog_term::PlainDecorator::new(writer);
    let drain = slog_term::CompactFormat::new(decorator).build();
    let drain = std::sync::Mutex::new(drain).fuse();
    let logger = slog::Logger::root(drain, slog::o!());
    slog_global::set_global(logger);
}

async fn handle_remote_analysis(
    dfs: Arc<dyn dfs::DFS>,
    req: hyper::Request<hyper::Body>,
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
    let snap_access = SnapAccess::from_change_set(dfs, change_set).await;
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

pub(crate) fn get_all_stores_except_tiflash(
    pd_client: &Arc<RpcClient>,
) -> Result<Vec<Store>, pd_client::Error> {
    Ok(pd_client
        .get_all_stores(true)?
        .into_iter()
        .filter(|s| {
            !s.get_labels()
                .iter()
                .any(|l| l.key.to_lowercase() == "engine" && l.value.to_lowercase() == "tiflash")
        })
        .collect())
}

fn register_compactor_to_all_stores(pd: Arc<RpcClient>, dfs: Arc<S3FS>, remote_url: String) {
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
}
