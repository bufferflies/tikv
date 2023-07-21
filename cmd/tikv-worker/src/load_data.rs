// Copyright 2023 TiKV Project Authors. Licensed under Apache-2.0.

use std::{collections::HashMap, path::PathBuf, sync::Arc};

use bytes::Bytes;
use cloud_encryption::MasterKey;
use http::{header, Method, Response, StatusCode};
use kvengine::{
    dfs,
    table::sstable::{LZ4_COMPRESSION, NO_COMPRESSION, ZSTD_COMPRESSION},
};
use load_data::task::{
    LoadDataContext, LoadTaskMsg, LoadTaskScheduler, LoadTaskStates, LoadTaskWorker, TaskContext,
};
use pd_client::PdClient;

use crate::common::{get_body, get_u64_param, make_response};

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
    if cluster_id != manager.ctx.pd.get_cluster_id().unwrap() {
        return Ok(make_response(
            StatusCode::BAD_REQUEST,
            "cluster id mismatch",
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
    ctx: LoadDataContext,
}

impl LoadDataManager {
    pub(crate) fn new(
        pd: Arc<dyn PdClient>,
        dir: PathBuf,
        dfs: Arc<dyn dfs::Dfs>,
        runtime: Arc<tokio::runtime::Runtime>,
        max_in_mem_size: usize,
        master_key: MasterKey,
    ) -> Self {
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
            ctx: context,
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
        let mut worker = LoadTaskWorker::new(self.ctx.clone(), task_ctx.clone());
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
