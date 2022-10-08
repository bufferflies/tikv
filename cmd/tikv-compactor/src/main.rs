// Copyright 2022 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    io,
    path::{Path, PathBuf},
    sync::Arc,
};

use clap::{App, Arg};
use hyper::service::{make_service_fn, service_fn};
use kvengine::dfs::DFSConfig;
use slog_global::{error, info};

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
    let mut config: Config = toml::from_slice(&data).unwrap();
    if config.port == 0 {
        config.port = 19000;
    }
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

    let thread_pool = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .worker_threads(16)
        .thread_name("compaction-server")
        .build()
        .unwrap();
    let addr = ([0, 0, 0, 0], config.port).into();

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

#[macro_use]
extern crate serde_derive;

#[derive(Clone, Serialize, Deserialize, PartialEq, Debug, Default)]
#[serde(default)]
#[serde(rename_all = "kebab-case")]
pub struct Config {
    pub port: u16,
    pub log_file: String,
    pub dfs: DFSConfig,
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
