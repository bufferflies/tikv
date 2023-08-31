// Copyright 2023 TiKV Project Authors. Licensed under Apache-2.0.

use std::{collections::HashMap, path::PathBuf, sync::Arc};

use bytes::Bytes;
use cloud_encryption::MasterKey;
use http::{header, Method, Response, StatusCode};
use hyper::Body;
use kvengine::{
    dfs,
    table::sstable::{LZ4_COMPRESSION, NO_COMPRESSION, ZSTD_COMPRESSION},
};
use load_data::task::{
    LoadDataConfig, LoadDataContext, LoadTaskMsg, LoadTaskScheduler, LoadTaskStates,
    LoadTaskWorker, TaskContext,
};
use pd_client::PdClient;

use crate::{
    common::{get_body, get_u64_param, make_response},
    worker_scaler::{WorkerScaler, WorkerScalerConfig},
};

pub(crate) const MAX_IN_MEM_SIZE: usize = 256 * 1024 * 1024;

/// Remote load data worker API:
///
/// 1. init task:
///   POST /load_data?cluster_id=%d&start_ts=%d&commit_ts=%d
///
/// 2. put chunk:
///   PUT /load_data?cluster_id=%d&start_ts=%d&chunk_id=%d
///   key_len(2) + key(key_len) + val_len(4) + value(val_len)
///   key_len(2) + key(key_len) + val_len(4) + value(val_len)
///   ...
///
/// 3. build task:
///   POST /load_data?cluster_id=%d&start_ts=%d&build=true&compression=zstd&
///        split_size=%d&split_keys=%d
///
/// 4. get task states:
///   GET /load_data?cluster_id=%d&start_ts=%d
///   {"canceled": false, "finished": false, "error": "", "created-files": 10,
///   "ingested-regions": 3}
///
/// 5. clean up task:
///   DELETE /load_data?cluster_id=%d&start_ts=%d
pub(crate) async fn handle_load_data(
    manager: Arc<LoadDataManager>,
    req: hyper::Request<hyper::Body>,
) -> hyper::Result<hyper::Response<hyper::Body>> {
    let query = req.uri().query().unwrap_or("");
    let query_pairs: HashMap<_, _> = url::form_urlencoded::parse(query.as_bytes()).collect();
    let cluster_id = get_u64_param(&query_pairs, "cluster_id").unwrap_or_default();
    let pd_cluster_id = manager.ctx.pd.get_cluster_id().unwrap();
    if cluster_id != pd_cluster_id {
        return Ok(make_response(
            StatusCode::BAD_REQUEST,
            format!(
                "cluster id mismatch, got {}, expected {}",
                cluster_id, pd_cluster_id
            ),
        ));
    }
    let start_ts = get_u64_param(&query_pairs, "start_ts").unwrap_or_default();
    if start_ts == 0 {
        if *req.method() == Method::GET {
            let tasks = manager.list_tasks();
            let json = serde_json::to_string(&tasks).unwrap();
            return Ok(Response::builder()
                .header(header::CONTENT_TYPE, "application/json")
                .body(json.into())
                .unwrap());
        }
        return Ok(make_response(
            StatusCode::BAD_REQUEST,
            "start_ts is missing",
        ));
    }
    match *req.method() {
        Method::GET => {
            if let Some(states) = manager.get_task_states(start_ts) {
                let json = serde_json::to_string(&states).unwrap();
                Ok(Response::builder()
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(json.into())
                    .unwrap())
            } else {
                Ok(make_response(StatusCode::NOT_FOUND, ""))
            }
        }
        Method::POST => {
            if query_pairs.get("build").map(|x| x.as_ref()) == Some("true") {
                if !manager.has_task(start_ts) {
                    Ok(make_response(StatusCode::NOT_FOUND, ""))
                } else {
                    let compression = query_pairs
                        .get("compression")
                        .map(|x| x.to_string())
                        .unwrap_or_default();
                    let body = get_body(req).await?;
                    match serde_json::from_slice(&body) {
                        Ok(chunk_ids) => {
                            manager.build(start_ts, &compression, chunk_ids);
                            Ok(make_response(StatusCode::OK, ""))
                        }
                        Err(err) => {
                            Ok(make_response(StatusCode::BAD_REQUEST, format!("{:?}", err)))
                        }
                    }
                }
            } else if manager.has_task(start_ts) {
                Ok(make_response(StatusCode::BAD_REQUEST, "task exists"))
            } else {
                let commit_ts = get_u64_param(&query_pairs, "commit_ts").unwrap_or_default();
                let data_size = get_u64_param(&query_pairs, "data_size").unwrap_or_default();
                if data_size > manager.worker_scaler_conf.max_size.0 {
                    return Ok(make_response(
                        StatusCode::BAD_REQUEST,
                        format!(
                            "data size {} exceeds max data size {}",
                            data_size, manager.worker_scaler_conf.max_size.0
                        ),
                    ));
                }
                let spawn_load_data_worker = manager.worker_scaler.is_some()
                    && (data_size > manager.worker_scaler_conf.spawn_data_size.0
                        || manager.running_tasks.len()
                            > manager.worker_scaler_conf.spawn_running_tasks);
                if spawn_load_data_worker {
                    let worker_scaler = manager.worker_scaler.as_ref().unwrap();
                    let data_size_gb = data_size / 1024 / 1024 / 1024;
                    let worker_pod_res = worker_scaler
                        .create_worker(start_ts, data_size_gb as usize)
                        .await;
                    match worker_pod_res {
                        Ok(worker_pod) => {
                            let worker_addr = worker_scaler.get_worker_addr(&worker_pod).unwrap();
                            let resp = Response::builder()
                                .header("Location", worker_addr)
                                .status(StatusCode::FOUND)
                                .body(Body::empty())
                                .unwrap();
                            return Ok(resp);
                        }
                        Err(err) => {
                            return Ok(make_response(
                                StatusCode::INTERNAL_SERVER_ERROR,
                                format!("{:?}", err),
                            ));
                        }
                    }
                }
                let task_ctx = TaskContext {
                    start_ts,
                    commit_ts,
                    inner_key_off: None,
                    key_prefix: vec![],
                    encryption_key: None,
                };
                // step 1: on start, client call init task
                manager.init_task(task_ctx);
                Ok(make_response(StatusCode::OK, ""))
            }
        }
        Method::PUT => {
            let chunk_id = get_u64_param(&query_pairs, "chunk_id").unwrap_or_default();
            if chunk_id == 0 {
                return Ok(make_response(
                    StatusCode::BAD_REQUEST,
                    "chunk id is missing",
                ));
            }
            let body = get_body(req).await?;
            manager.put_chunk(start_ts, chunk_id, body.into());
            Ok(make_response(StatusCode::OK, ""))
        }
        Method::DELETE => {
            if !manager.has_task(start_ts) {
                Ok(make_response(StatusCode::NOT_FOUND, ""))
            } else {
                // step 4: on finish, client call DELETE task
                manager.delete(start_ts);
                Ok(make_response(StatusCode::OK, ""))
            }
        }
        _ => Ok(make_response(StatusCode::BAD_REQUEST, "invalid method")),
    }
}

pub(crate) struct LoadDataManager {
    running_tasks: Arc<dashmap::DashMap<u64, LoadTaskScheduler>>,
    config: LoadDataConfig,
    ctx: LoadDataContext,
    worker_scaler: Option<WorkerScaler>,
    worker_scaler_conf: WorkerScalerConfig,
}

impl LoadDataManager {
    pub(crate) fn new(
        pd: Arc<dyn PdClient>,
        dir: PathBuf,
        dfs: Arc<dyn dfs::Dfs>,
        runtime: Arc<tokio::runtime::Runtime>,
        max_in_mem_size: usize,
        master_key: MasterKey,
        worker_scaler: Option<WorkerScaler>,
        worker_scaler_conf: WorkerScalerConfig,
    ) -> Self {
        let config = LoadDataConfig::default();
        let context = LoadDataContext {
            pd,
            dir,
            dfs,
            runtime,
            max_in_mem_size,
            master_key,
        };
        Self {
            running_tasks: Arc::new(dashmap::DashMap::default()),
            config,
            ctx: context,
            worker_scaler,
            worker_scaler_conf,
        }
    }

    pub(crate) fn get_task_states(&self, start_ts: u64) -> Option<LoadTaskStates> {
        self.running_tasks.get(&start_ts).map(|x| {
            let thread_finished = x
                .thread_handle
                .as_ref()
                .unwrap()
                .lock()
                .unwrap()
                .is_finished();
            if thread_finished {
                x.cancel("task thread finished unexpectedly".to_string());
            }
            x.states.lock().unwrap().clone()
        })
    }

    pub(crate) fn list_tasks(&self) -> Vec<LoadTaskStates> {
        self.running_tasks
            .iter()
            .map(|x| {
                let thread_finished = x
                    .thread_handle
                    .as_ref()
                    .unwrap()
                    .lock()
                    .unwrap()
                    .is_finished();
                if thread_finished {
                    x.cancel("task thread finished unexpectedly".to_string());
                }
                x.states.lock().unwrap().clone()
            })
            .collect()
    }

    pub(crate) fn has_task(&self, start_ts: u64) -> bool {
        self.running_tasks.contains_key(&start_ts)
    }

    pub(crate) fn init_task(&self, task_ctx: TaskContext) {
        let mut worker =
            LoadTaskWorker::new(self.config.clone(), self.ctx.clone(), task_ctx.clone());
        let mut scheduler = worker.get_scheduler();
        let thread_handle = std::thread::spawn(move || {
            worker.run();
        });
        scheduler.set_thread_handle(thread_handle);
        self.running_tasks.insert(task_ctx.start_ts, scheduler);
    }

    pub(crate) fn build(&self, start_ts: u64, compression: &str, chunk_ids: Vec<u64>) {
        let compression_type = match compression {
            "lz4" => LZ4_COMPRESSION,
            "zstd" => ZSTD_COMPRESSION,
            _ => NO_COMPRESSION,
        };
        let scheduler = self.running_tasks.get(&start_ts).unwrap().clone();
        scheduler
            .sender
            .send(LoadTaskMsg::Build {
                chunk_ids,
                compression_type,
            })
            .unwrap();
    }

    pub(crate) fn put_chunk(&self, start_ts: u64, chunk_id: u64, chunk_data: Bytes) {
        let scheduler = self.running_tasks.get(&start_ts).unwrap().clone();
        scheduler
            .sender
            .send(LoadTaskMsg::AddChunk {
                chunk_id,
                chunk_data,
            })
            .unwrap();
    }

    pub(crate) fn delete(&self, start_ts: u64) {
        if let Some((_, scheduler)) = self.running_tasks.remove(&start_ts) {
            scheduler.cancel("deleted".to_string());
            scheduler.sender.send(LoadTaskMsg::Cleanup).unwrap();
        }
    }
}
