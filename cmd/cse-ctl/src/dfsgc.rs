// Copyright 2022 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    collections::HashSet,
    fs,
    path::PathBuf,
    str::FromStr,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};

use bytes::Buf;
use clap::ArgMatches;
use grpcio::EnvBuilder;
use http::Uri;
use kvengine::{
    dfs,
    dfs::{DFSConfig, DFS, S3FS},
};
use kvproto::metapb::Store;
use pd_client::{PdClient, RpcClient};
use security::{SecurityConfig, SecurityManager};
use slog_global::error;
use tikv_util::info;

pub(crate) fn execute_dfsgc(matches: &ArgMatches) {
    let config_path = matches.value_of_os("config").unwrap();
    let start_after = matches.value_of("start").unwrap();
    let result = std::fs::read(PathBuf::from(config_path));
    if result.is_err() {
        error!("failed to read config file {:?}", result.unwrap_err());
        return;
    }
    let data = result.unwrap();
    let mut config: DFSGCConfig = toml::from_slice(&data).unwrap();
    if config.data_dir.is_empty() {
        config.data_dir = ".".to_string();
    }
    let security_mgr = Arc::new(
        SecurityManager::new(&config.security)
            .unwrap_or_else(|e| panic!("failed to create security manager: {:?}", e)),
    );
    let env = Arc::new(EnvBuilder::new().cq_count(1).build());
    let pd_client = RpcClient::new(&config.pd, Some(env), security_mgr)
        .unwrap_or_else(|e| panic!("failed to create rpc client: {:?}", e));
    let s3fs = S3FS::new(
        config.dfs.prefix,
        config.dfs.s3_endpoint,
        config.dfs.s3_key_id,
        config.dfs.s3_secret_key,
        config.dfs.s3_region,
        config.dfs.s3_bucket,
    );
    let progress_file_path = PathBuf::from(format!("{}/{}", &config.data_dir, "dfsgc.progress"));
    let mut gc_worker = GcWorker::new(pd_client, s3fs, progress_file_path);
    gc_worker.collect_valid_files();
    gc_worker.remove_garbage_files(start_after.to_string());
}

static REMOVED: AtomicUsize = AtomicUsize::new(0);

#[derive(Clone, Serialize, Deserialize, PartialEq, Debug, Default)]
#[serde(default)]
#[serde(rename_all = "kebab-case")]
pub struct DFSGCConfig {
    pub pd: pd_client::Config,
    pub security: SecurityConfig,
    pub dfs: DFSConfig,

    // If the task is not finished, a progress file is stored in the data dir, the next run
    // will load the progress and continue the task.
    pub data_dir: String,
}

struct GcWorker {
    pd: RpcClient,
    s3fs: S3FS,
    progress_file_path: PathBuf,
    valid_files: HashSet<u64>,
}

impl GcWorker {
    fn new(pd: RpcClient, s3fs: S3FS, progress_file_path: PathBuf) -> Self {
        Self {
            pd,
            s3fs,
            progress_file_path,
            valid_files: HashSet::default(),
        }
    }

    fn collect_valid_files(&mut self) {
        let all_stores = self
            .pd
            .get_all_stores(true)
            .unwrap_or_else(|e| panic!("failed get all stores {:?}", e));
        let start_time = Instant::now();
        let stores_len = all_stores.len();
        let (tx, rx) = std::sync::mpsc::sync_channel(stores_len);
        for store in all_stores {
            let tx = tx.clone();
            self.s3fs.get_runtime().spawn(async move {
                tx.send(Self::get_store_files(store).await).unwrap();
            });
        }
        for _ in 0..stores_len {
            let store_files_ids = rx.recv().unwrap();
            for (_, region_file_ids) in store_files_ids {
                self.valid_files.extend(region_file_ids.into_iter());
            }
        }
        let elapsed = start_time.elapsed();
        info!(
            "collect {} valid files from {} stores in {:?}",
            self.valid_files.len(),
            stores_len,
            elapsed
        );
        // If all peers of a region are moved during the collection, it is possible that some files
        // are not collected, cause data lost.
        // As it takes time to move peer, we concurrently collect the store files to make the the
        // collection faster and assume it takes at least 30s to move all peers.
        assert!(elapsed < Duration::from_secs(30));
    }

    async fn get_store_files(store: Store) -> Vec<(u64, Vec<u64>)> {
        let uri =
            Uri::from_str(&format!("http://{}/kvengine/files", &store.status_address)).unwrap();
        let client = hyper::Client::new();
        let resp = client.get(uri).await.unwrap();
        let body = hyper::body::to_bytes(resp.into_body()).await.unwrap();
        serde_json::from_slice(body.chunk()).unwrap()
    }

    fn remove_garbage_files(&self, mut start_after: String) {
        if let Ok(data) = fs::read_to_string(self.progress_file_path.as_path()) {
            let state_start_after = data;
            if start_after < state_start_after {
                info!(
                    "update start_after {} to {} from state file",
                    start_after, state_start_after
                );
                start_after = state_start_after;
            }
        }
        info!("start remove garbage files after {}", start_after);
        let mut checked = 0;
        loop {
            let s3fs = self.s3fs.clone();
            let (files, has_more) = self
                .s3fs
                .get_runtime()
                .block_on(s3fs.list(start_after.as_str()))
                .unwrap();
            info!("listed {} files", files.len());
            for file_suffix in &files {
                let file_id = self.s3fs.parse_file_id(file_suffix.as_str());
                if !self.valid_files.contains(&file_id) {
                    // Remove the files one by one to prevent reach API rate limit.
                    self.remove_garbage_file(file_id);
                    checked += 1;
                }
            }
            if !files.is_empty() {
                start_after = (*files.last().unwrap()).clone();
                fs::write(self.progress_file_path.as_path(), start_after.as_bytes()).unwrap();
            }
            if !has_more || files.is_empty() {
                break;
            }
        }
        if self.progress_file_path.exists() {
            fs::remove_file(self.progress_file_path.as_path()).unwrap();
        }

        info!(
            "{} files valid, {} files checked, {} files removed",
            self.valid_files.len(),
            checked,
            REMOVED.load(Ordering::SeqCst)
        );
        info!("finished");
    }

    fn remove_garbage_file(&self, id: u64) {
        let opts = dfs::Options::new(0, 0);
        let s3fs = self.s3fs.clone();
        self.s3fs.get_runtime().block_on(async move {
            if !s3fs.is_removed(id).await {
                s3fs.remove(id, opts).await;
                REMOVED.fetch_add(1, Ordering::SeqCst);
                info!("removed {}", id);
            }
        });
    }
}
