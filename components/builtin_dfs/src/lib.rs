// Copyright 2024 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    collections::HashMap,
    sync::{Arc, RwLock},
    time::Duration,
};

use async_trait::async_trait;
use bytes::Bytes;
use http::{StatusCode, Uri};
use hyper::client::HttpConnector;
use kvengine::{
    dfs,
    dfs::{Dfs, FileType, Options},
};
use kvproto::{
    metapb,
    metapb::{Region, Store},
};
use pd_client::PdClient;
use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};
use tikv_util::{error, sys::thread::ThreadBuildWrapper, time::Instant, HandyRwLock};
use tokio::runtime::Runtime;

/// File metadata returned by meta query API.
///
/// Unlike S3 which returns file size in the Content-Length header with empty
/// body, we return metadata as JSON in the response body. This is because the
/// hyper HTTP framework automatically sets Content-Length based on actual body
/// size, making it impossible to set Content-Length to the file size while
/// keeping the body empty. Using JSON body provides better extensibility for
/// future metadata fields.
#[derive(Serialize, Deserialize, Debug)]
pub struct BuiltinDfsFileMeta {
    pub size: u64,
}

const PD_CLIENT_TIMEOUT: Duration = Duration::from_secs(30);

pub struct BuiltinDfs {
    pd: Arc<dyn PdClient>,
    runtime: Runtime,
    http_client: Arc<hyper::Client<HttpConnector>>,
    store_cache: RwLock<HashMap<u64, Store>>,
    region_cache: RwLock<HashMap<u64, Region>>,
}

impl BuiltinDfs {
    pub fn new(pd: Arc<dyn PdClient>) -> Self {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .thread_name("builtin-dfs")
            .enable_all()
            .worker_threads(2)
            .with_sys_hooks()// TODO: add IO type
            .build()
            .unwrap();
        Self {
            pd,
            runtime,
            http_client: Arc::new(hyper::Client::new()),
            store_cache: RwLock::new(HashMap::new()),
            region_cache: RwLock::new(HashMap::new()),
        }
    }

    async fn get_store_addr(&self, store_id: u64) -> dfs::Result<String> {
        {
            let guard = self.store_cache.rl();
            if let Some(store) = guard.get(&store_id) {
                // TODO: handle store address refresh.
                return Ok(store.get_status_address().to_string());
            }
        }
        let start_time = Instant::now_coarse();
        let mut retry = 0;
        while start_time.saturating_elapsed() < PD_CLIENT_TIMEOUT {
            match self.pd.get_store_async(store_id).await {
                Err(err) => {
                    error!(
                        "builtin_dfs get store {} failed {} retry {}",
                        store_id, err, retry
                    );
                    retry += 1;
                    tokio::time::sleep(Duration::from_secs(2)).await;
                    continue;
                }
                Ok(store) => {
                    let mut guard = self.store_cache.wl();
                    guard.insert(store_id, store.clone());
                    return Ok(store.status_address);
                }
            }
        }
        Err(dfs::Error::Other("get store addr timed out".to_string()))
    }

    async fn get_stores(&self, opts: &Options) -> dfs::Result<Vec<u64>> {
        if opts.shard_id == 0 {
            // opts.end_off.is_some(): IA segments.
            debug_assert!(
                opts.file_type == FileType::TxnChunk
                    || opts.file_type == FileType::Schema
                    || opts.end_off.is_some(),
                "shard_id is missing: {:?}",
                opts
            );
            let store_cache_rl = self.store_cache.rl();
            let empty = store_cache_rl.is_empty();
            drop(store_cache_rl);
            if empty {
                let mut store_cache = self.store_cache.wl();
                let mut stores = self
                    .pd
                    .get_all_stores(true)
                    .map_err(|e| dfs::Error::Other(e.to_string()))?;
                stores.retain(|store| {
                    !store
                        .get_labels()
                        .iter()
                        .any(|l| l.key == "engine" && l.value == "tiflash")
                        && store.state == metapb::StoreState::Up
                });
                for store in stores {
                    store_cache.insert(store.id, store);
                }
            }
            let store_cache = self.store_cache.rl();
            let mut store_ids: Vec<u64> = store_cache.keys().cloned().collect();
            drop(store_cache);
            store_ids.sort();
            store_ids.truncate(3);
            return Ok(store_ids);
        }

        {
            let guard = self.region_cache.read().unwrap();
            if let Some(region) = guard.get(&opts.shard_id) {
                let region_ver = region.get_region_epoch().get_version();
                if region_ver >= opts.shard_ver {
                    return Ok(region.get_peers().iter().map(|p| p.store_id).collect());
                }
            }
        }
        self.update_region(opts.shard_id).await
    }

    async fn update_region(&self, region_id: u64) -> dfs::Result<Vec<u64>> {
        let start_time = Instant::now_coarse();
        let mut retry = 0;
        while start_time.saturating_elapsed() < PD_CLIENT_TIMEOUT {
            match self.pd.get_region_by_id(region_id).await {
                Err(err) => {
                    error!(
                        "builtin_dfs get region {} failed {}, retry {}",
                        region_id, err, retry
                    );
                    retry += 1;
                    tokio::time::sleep(Duration::from_secs(2)).await;
                    continue;
                }
                Ok(region_opt) => {
                    if region_opt.is_none() {
                        return Err(dfs::Error::Other(format!("region {} not found", region_id)));
                    }
                    let region = region_opt.unwrap();
                    let mut guard = self.region_cache.wl();
                    guard.insert(region_id, region.clone());
                    return Ok(region.get_peers().iter().map(|p| p.store_id).collect());
                }
            }
        }
        Err(dfs::Error::Other("get region timed out".to_string()))
    }

    async fn read_file_from_store(
        &self,
        file_id: u64,
        opts: Options,
        store_id: u64,
    ) -> dfs::Result<Bytes> {
        let store_addr = self.get_store_addr(store_id).await?;
        let end_off = opts
            .end_off
            .map(|end| format!("&end_off={}", end))
            .unwrap_or_default();
        let uri = Uri::try_from(format!(
            "http://{}/dfs/{}?file_type={}&start_off={}{}",
            store_addr, file_id, opts.file_type, opts.start_off, end_off
        ))
        .unwrap();
        let resp = self.http_client.get(uri).await?;
        if !resp.status().is_success() {
            return Err(dfs::Error::Other(format!(
                "read file failed: {}",
                resp.status()
            )));
        }
        let data = hyper::body::to_bytes(resp.into_body()).await?;
        Ok(data)
    }

    async fn read_file_inner(&self, file_id: u64, opts: Options) -> dfs::Result<Bytes> {
        let mut stores = self.get_stores(&opts).await?;
        stores.shuffle(&mut rand::thread_rng());
        let mut errs = vec![];
        for store_id in stores {
            match self.read_file_from_store(file_id, opts, store_id).await {
                Ok(data) => return Ok(data),
                Err(err) => errs.push(format!("read file failed: {}", err)),
            }
        }
        Err(dfs::Error::Other(errs.join(", ")))
    }

    async fn create_file_inner(&self, file_id: u64, data: Bytes, opts: Options) -> dfs::Result<()> {
        let stores = self.get_stores(&opts).await?;
        let (tx, mut rx) = tokio::sync::mpsc::channel(stores.len());
        for &store_id in &stores {
            let store_addr = self.get_store_addr(store_id).await?;
            let url = format!(
                "http://{}/dfs/{}?file_type={}",
                store_addr, file_id, opts.file_type
            );
            let req = hyper::Request::builder()
                .method("POST")
                .uri(url)
                .body(data.clone().into())
                .unwrap();
            let http_client = self.http_client.clone();
            let tx = tx.clone();
            self.get_runtime().spawn(async move {
                let res = http_client.request(req).await;
                let _ = tx.send(res).await;
            });
        }
        let mut errors = vec![];
        let mut success_cnt = 0;
        for _ in 0..stores.len() {
            let res = rx.recv().await.unwrap();
            if res.is_err() {
                errors.push(dfs::Error::Other(format!(
                    "create file failed: {}",
                    res.err().unwrap()
                )));
                continue;
            }
            let resp = res.unwrap();
            let status = resp.status();
            if !status.is_success() {
                let err_msg = hyper::body::to_bytes(resp.into_body())
                    .await
                    .map(|b| String::from_utf8_lossy(&b).to_string())?;
                errors.push(dfs::Error::Other(format!(
                    "create file failed: {}:{}",
                    status, err_msg,
                )));
            } else {
                success_cnt += 1;
                if success_cnt == 2 {
                    break;
                }
            }
        }
        if errors.len() >= 2 {
            return Err(errors.pop().unwrap());
        }
        Ok(())
    }

    /// Helper function for file metadata queries.
    ///
    /// This function queries multiple stores using meta=true and returns
    /// file metadata from JSON response body.
    ///
    /// # Arguments
    /// * `file_id` - The file ID to query
    /// * `opts` - Query options including file type and shard information
    async fn query_meta(&self, file_id: u64, opts: &Options) -> dfs::Result<BuiltinDfsFileMeta> {
        let mut stores = self.get_stores(opts).await?;
        stores.shuffle(&mut rand::thread_rng());
        let mut errors = Vec::new();

        for store_id in stores {
            let store_addr = self.get_store_addr(store_id).await?;
            let uri = Uri::try_from(format!(
                "http://{}/dfs/{}?file_type={}&meta=true",
                store_addr, file_id, opts.file_type
            ))
            .unwrap();

            match self.http_client.get(uri).await {
                Ok(resp) => {
                    let status = resp.status();
                    if status.is_success() {
                        // Parse JSON response body
                        match hyper::body::to_bytes(resp.into_body()).await {
                            Ok(body) => match serde_json::from_slice::<BuiltinDfsFileMeta>(&body) {
                                Ok(meta) => return Ok(meta),
                                Err(e) => {
                                    errors.push(dfs::Error::Other(format!(
                                        "failed to parse meta response: {}",
                                        e
                                    )));
                                }
                            },
                            Err(e) => {
                                errors.push(dfs::Error::Other(format!(
                                    "failed to read response body: {}",
                                    e
                                )));
                            }
                        }
                    } else if status != StatusCode::NOT_FOUND {
                        // Collect non-404 errors
                        let err_msg = hyper::body::to_bytes(resp.into_body())
                            .await
                            .map(|b| String::from_utf8_lossy(&b).to_string())
                            .unwrap_or_else(|e| format!("failed to read error message: {}", e));
                        errors.push(dfs::Error::Other(format!(
                            "meta query failed: {}:{}",
                            status, err_msg
                        )));
                    }
                    // Silently skip 404 errors - file might exist on other
                    // stores
                }
                Err(e) => {
                    errors.push(dfs::Error::Other(format!("meta request failed: {}", e)));
                }
            }
        }

        // Return the last error if any, otherwise a generic error
        if errors.is_empty() {
            Err(dfs::Error::Other(format!(
                "meta query for file {} failed: file not found on any store",
                file_id
            )))
        } else {
            Err(errors.pop().unwrap())
        }
    }
}

#[async_trait]
impl dfs::Dfs for BuiltinDfs {
    async fn size(&self, file_id: u64, opts: Options) -> dfs::Result<u64> {
        let meta = self.query_meta(file_id, &opts).await?;
        Ok(meta.size)
    }

    async fn exists(&self, file_id: u64, opts: Options) -> dfs::Result<bool> {
        match self.query_meta(file_id, &opts).await {
            Ok(_meta) => Ok(true), // Got metadata, file exists
            Err(dfs::Error::Other(ref msg)) if msg.contains("file not found on any store") => {
                // All stores returned 404, file doesn't exist
                Ok(false)
            }
            Err(e) => Err(e), // Other errors should propagate
        }
    }

    async fn read_file(&self, file_id: u64, opts: Options) -> dfs::Result<Bytes> {
        self.read_file_inner(file_id, opts).await
    }

    async fn create(&self, file_id: u64, data: Bytes, opts: Options) -> dfs::Result<()> {
        self.create_file_inner(file_id, data, opts).await
    }

    async fn remove(&self, _file_id: u64, _file_len: Option<u64>, _opts: Options) {
        // Do nothing, file is removed by GC.
    }

    async fn permanently_remove(&self, _file_id: u64, _opts: Options) -> kvengine::dfs::Result<()> {
        // Do nothing, file is removed by GC.
        Ok(())
    }

    fn get_runtime(&self) -> &Runtime {
        &self.runtime
    }
}
