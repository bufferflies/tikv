// Copyright 2022 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    io,
    path::{Path, PathBuf},
    str::FromStr,
    sync::Arc,
    time::Instant,
};

use clap::{App, Arg};
use grpcio::EnvBuilder;
use http::Uri;
use hyper::service::{make_service_fn, service_fn};
use kvengine::dfs::{DFSConfig, DFS, S3FS};
use kvproto::metapb::Store;
use pd_client::{PdClient, RpcClient};
use security::{SecurityConfig, SecurityManager};
use slog_global::{error, info};
use tikv_util::config::ReadableDuration;

fn main() {
    init_logger(io::stdout());
    let matches = App::new("tikv-compactor")
        .about("tikv remote compactor")
        .arg(
            Arg::with_name("config")
                .short("C")
                .long("config")
                .value_name("FILE")
                .help("Set the configuration file")
                .takes_value(true)
                .required(true),
        )
        .get_matches();
    let config_path = matches.value_of_os("config").unwrap();
    let result = std::fs::read(PathBuf::from(config_path));
    if result.is_err() {
        error!("failed to read config file {:?}", result.unwrap_err());
        return;
    }
    let data = result.unwrap();
    let config: Config = toml::from_slice(&data).unwrap();
    if !config.log_file.is_empty() {
        let log = tikv_util::logger::file_writer(&config.log_file, 300, 0, 0, rename_by_timestamp)
            .unwrap();
        init_logger(log);
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
    let dfsclone = dfs.clone();

    let thread_pool = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .worker_threads(16)
        .thread_name("compaction-server")
        .build()
        .unwrap();
    let addr = config.addr.parse().expect("Unable to parse socket address");

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
                async move { kvengine::handle_remote_compaction(dfs, req).await }
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
        let remote_url = format!("http://{}", config.addr);
        let duration = config.update_interval.0;
        std::thread::spawn(move || {
            loop {
                register_to_all_stores(pd_client.clone(), dfsclone.clone(), remote_url.clone());
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

fn register_to_all_stores(pd: Arc<RpcClient>, dfs: Arc<S3FS>, remote_url: String) {
    let all_stores = pd
        .get_all_stores(true)
        .unwrap_or_else(|e| panic!("failed get all stores {:?}", e));
    let start_time = Instant::now();
    let stores_len = all_stores.len();
    let (tx, rx) = std::sync::mpsc::sync_channel(stores_len);
    for store in all_stores {
        let tx = tx.clone();
        let remote_url = remote_url.clone();
        dfs.get_runtime().spawn(async move {
            tx.send(register_to_store(store, remote_url.clone()).await)
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
    let elapsed = start_time.elapsed();
    let remain = stores_len - finish_cnt;
    info!(
        "register compactor to {} stores in {:?}, remain {} stores",
        stores_len, elapsed, remain
    );
}

async fn register_to_store(store: Store, remote_url: String) -> bool {
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
            addr: String::from("127.0.0.1:19000"),
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
