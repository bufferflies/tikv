// Copyright 2023 TiKV Project Authors. Licensed under Apache-2.0.

use std::{collections::HashMap, fs, path::PathBuf, sync::Arc};

use bytes::Bytes;
use cloud_encryption::MasterKey;
use http::{header, Method, Response, StatusCode};
use hyper::Body;
use kvengine::{
    dfs,
    table::sstable::{LZ4_COMPRESSION, NO_COMPRESSION, ZSTD_COMPRESSION},
};
use load_data::{
    check_point_storage,
    check_point_storage::{
        LoadDataCheckPointCtx, LoadDataWorkerState::BuildingSst, LocalFileCheckPointStorage,
    },
    task::{
        FlushResult, LoadDataConfig, LoadDataContext, LoadTaskMsg, LoadTaskScheduler,
        LoadTaskStates, LoadTaskWorker, PutChunkResult, TaskContext,
    },
};
use pd_client::PdClient;
use tikv_util::{debug, info};

use crate::{
    common::{get_body, get_param, make_response},
    worker_scaler::{WorkerScaler, WorkerScalerConfig},
};

pub(crate) const MAX_IN_MEM_SIZE: usize = 256 * 1024 * 1024;

/// Remote load data worker API:
///
/// 1. init task:
///   POST /load_data?cluster_id=%d&task_id=%s&start_ts=%d&commit_ts=%d
///
/// 2. put chunk:
///   PUT /load_data?cluster_id=%d&task_id=%s&writer_id=%d&chunk_id=%d
///   key_len(2) + key(key_len) + val_len(4) + value(val_len)
///   key_len(2) + key(key_len) + val_len(4) + value(val_len)
///   ...
///
/// 3. flush:
///   POST /load_data?cluster_id=%d&task_id=%s&flush=true
///
/// 4. build task:
///   POST /load_data?cluster_id=%d&task_id=%s&build=true&compression=zstd&
///        split_size=%d&split_keys=%d
///
/// 5. get task states:
///   GET /load_data?cluster_id=%d&task_id=%s
///   {"canceled": false, "finished": false, "error": "", "created-files": 10,
///   "ingested-regions": 3}
///
/// 6. clean up task:
///   DELETE /load_data?cluster_id=%d&task_id=%s
pub(crate) async fn handle_load_data(
    manager: Arc<LoadDataManager>,
    req: hyper::Request<hyper::Body>,
) -> hyper::Result<hyper::Response<hyper::Body>> {
    let query = req.uri().query().unwrap_or("");
    let query_pairs: HashMap<_, _> = url::form_urlencoded::parse(query.as_bytes()).collect();
    let cluster_id = get_param::<u64>(&query_pairs, "cluster_id").unwrap_or_default();
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
    let task_id = query_pairs
        .get("task_id")
        .map(|x| x.to_string())
        .unwrap_or_default();
    if task_id.is_empty() {
        if *req.method() == Method::GET {
            let tasks = manager.list_tasks();
            let json = serde_json::to_string(&tasks).unwrap();
            return Ok(Response::builder()
                .header(header::CONTENT_TYPE, "application/json")
                .body(json.into())
                .unwrap());
        }
        return Ok(make_response(StatusCode::BAD_REQUEST, "task_id is missing"));
    }
    match *req.method() {
        Method::GET => {
            if let Some(states) = manager.get_task_states(&task_id) {
                let json = serde_json::to_string(&states).unwrap();
                return Ok(Response::builder()
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(json.into())
                    .unwrap());
            }
            if manager.worker_scaler.is_some() {
                let worker_scaler = manager.worker_scaler.as_ref().unwrap();
                let worker_pod_addr = worker_scaler.get_worker_addr_by_task_id(&task_id).await;
                if let Some(addr) = worker_pod_addr {
                    info!("Get worker addr from cache:{}", addr);
                    let resp = Response::builder()
                        .header("Location", addr)
                        .status(StatusCode::FOUND)
                        .body(Body::empty())
                        .unwrap();
                    return Ok(resp);
                }
            }
            Ok(make_response(StatusCode::NOT_FOUND, ""))
        }
        Method::POST => {
            if query_pairs.get("build").map(|x| x.as_ref()) == Some("true") {
                if !manager.has_task(&task_id) {
                    Ok(make_response(StatusCode::NOT_FOUND, ""))
                } else {
                    let compression = query_pairs
                        .get("compression")
                        .map(|x| x.to_string())
                        .unwrap_or_default();
                    manager.build(&task_id, &compression);
                    Ok(make_response(StatusCode::OK, ""))
                }
            } else if query_pairs.get("flush").map(|x| x.as_ref()) == Some("true") {
                if !manager.has_task(&task_id) {
                    Ok(make_response(StatusCode::NOT_FOUND, ""))
                } else {
                    let flush_res = manager.flush(&task_id).await;
                    let json = serde_json::to_string(&flush_res).unwrap();
                    Ok(Response::builder()
                        .header(header::CONTENT_TYPE, "application/json")
                        .body(json.into())
                        .unwrap())
                }
            } else if manager.has_task(&task_id) {
                Ok(make_response(StatusCode::BAD_REQUEST, "task exists"))
            } else {
                let start_ts = get_param::<u64>(&query_pairs, "start_ts").unwrap_or_default();
                let commit_ts = get_param::<u64>(&query_pairs, "commit_ts").unwrap_or_default();
                let data_size = get_param::<u64>(&query_pairs, "data_size").unwrap_or_default();
                if data_size > manager.worker_scaler_conf.max_size.0 {
                    return Ok(make_response(
                        StatusCode::BAD_REQUEST,
                        format!(
                            "data size {} exceeds max data size {}",
                            data_size, manager.worker_scaler_conf.max_size.0
                        ),
                    ));
                }

                if manager.worker_scaler.is_some() {
                    let worker_scaler = manager.worker_scaler.as_ref().unwrap();
                    let worker_addr = worker_scaler.get_worker_addr_by_task_id(&task_id).await;
                    if let Some(addr) = worker_addr {
                        info!("Get worker addr from cache:{}", addr);
                        let resp = Response::builder()
                            .header("Location", addr)
                            .status(StatusCode::FOUND)
                            .body(Body::empty())
                            .unwrap();
                        return Ok(resp);
                    }
                }

                let spawn_load_data_worker = manager.worker_scaler.is_some()
                    && (data_size > manager.worker_scaler_conf.spawn_data_size.0
                        || (manager.running_tasks.len()
                            > manager.worker_scaler_conf.spawn_running_tasks
                            && data_size != 0));
                if spawn_load_data_worker {
                    let worker_scaler = manager.worker_scaler.as_ref().unwrap();
                    let data_size_gb = data_size / 1024 / 1024 / 1024;
                    let worker_pod_res = worker_scaler
                        .create_worker(&task_id, data_size_gb as usize)
                        .await;
                    match worker_pod_res {
                        Ok(worker_pod) => {
                            let worker_addr = worker_scaler.get_worker_addr(&worker_pod).unwrap();
                            info!("Get worker addr:{}", worker_addr);
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
                    task_id,
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
            let chunk_id = get_param::<u64>(&query_pairs, "chunk_id");
            if chunk_id.is_none() {
                return Ok(make_response(
                    StatusCode::BAD_REQUEST,
                    "chunk id is missing",
                ));
            }
            // To ensure compatibility, writer id can be None.
            // TODO: check writer_id after all remote backends are upgraded.
            let writer_id = get_param::<u64>(&query_pairs, "writer_id").unwrap_or_default();
            let body = get_body(req).await?;
            let put_chunk_res = manager
                .put_chunk(&task_id, writer_id, chunk_id.unwrap(), body.into())
                .await;

            let json = serde_json::to_string(&put_chunk_res).unwrap();
            Ok(Response::builder()
                .header(header::CONTENT_TYPE, "application/json")
                .body(json.into())
                .unwrap())
        }
        Method::DELETE => {
            if !manager.has_task(&task_id) {
                Ok(make_response(StatusCode::NOT_FOUND, ""))
            } else {
                // step 4: on finish, client call DELETE task
                manager.delete(&task_id);
                Ok(make_response(StatusCode::OK, ""))
            }
        }
        _ => Ok(make_response(StatusCode::BAD_REQUEST, "invalid method")),
    }
}

pub(crate) struct LoadDataManager {
    running_tasks: Arc<dashmap::DashMap<String, LoadTaskScheduler>>,
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
        enable_check_point: bool,
    ) -> Self {
        let mut config = LoadDataConfig::default();
        config.enable_check_point = enable_check_point;
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

    fn recover_task_by_check_point_file(&self, path: PathBuf) {
        let file_data = LocalFileCheckPointStorage::read_file(path);
        let mut check_point_ctx =
            LocalFileCheckPointStorage::binary_to_check_point(file_data.as_str());
        check_point_ctx.set_is_recover(true);
        debug!(
            "[check point] try recover task from checkpoint, file exists {}",
            file_data
        );
        self.exec_task_by_check_point(check_point_ctx.clone());

        if check_point_ctx.get_state() >= BuildingSst {
            self.build(
                check_point_ctx.get_task_id().as_str(),
                LoadDataManager::compression_num_to_str(check_point_ctx.clone().get_compression()),
            );
        }
    }

    pub fn try_recover_tasks_by_check_point(&self) {
        if !self.config.enable_check_point {
            return;
        }

        let mut check_point_dir = self.ctx.dir.clone();

        if check_point_dir.as_os_str().is_empty() {
            // Use current dir.
            check_point_dir = PathBuf::from(".");
        }

        let files = fs::read_dir(check_point_dir).unwrap();
        for file in files.filter_map(Result::ok) {
            let file_name = file.file_name();
            let str_file_name = file_name.to_string_lossy();

            if str_file_name.starts_with(check_point_storage::CHECKPOINT_WORKER_PREFIX) {
                let path = file.path();
                self.recover_task_by_check_point_file(path.clone());
            }
        }
    }

    pub(crate) fn get_task_states(&self, task_id: &str) -> Option<LoadTaskStates> {
        self.running_tasks.get(task_id).map(|x| {
            x.check_task_thread_finished();
            x.states.lock().unwrap().clone()
        })
    }

    pub(crate) fn list_tasks(&self) -> Vec<LoadTaskStates> {
        self.running_tasks
            .iter()
            .map(|x| {
                x.check_task_thread_finished();
                x.states.lock().unwrap().clone()
            })
            .collect()
    }

    pub(crate) fn has_task(&self, task_id: &str) -> bool {
        self.running_tasks.contains_key(task_id)
    }

    pub(crate) fn exec_task_by_check_point(&self, check_point_ctx: LoadDataCheckPointCtx) {
        let task_context = TaskContext {
            task_id: check_point_ctx.get_task_id(),
            start_ts: check_point_ctx.get_start_ts(),
            commit_ts: check_point_ctx.get_commit_ts(),
            inner_key_off: None,
            key_prefix: vec![],
            encryption_key: None,
        };
        let mut worker = LoadTaskWorker::new(
            self.config.clone(),
            self.ctx.clone(),
            task_context.clone(),
            check_point_ctx,
        );
        let mut scheduler = worker.get_scheduler();
        let thread_handle = std::thread::spawn(move || {
            worker.run();
        });

        scheduler.set_thread_handle(thread_handle);
        self.running_tasks.insert(task_context.task_id, scheduler);
    }

    pub(crate) fn init_task(&self, task_ctx: TaskContext) {
        let check_point = LoadDataCheckPointCtx::new(task_ctx.clone());
        let mut worker = LoadTaskWorker::new(
            self.config.clone(),
            self.ctx.clone(),
            task_ctx.clone(),
            check_point,
        );
        let mut scheduler = worker.get_scheduler();
        let thread_handle = std::thread::spawn(move || {
            worker.run();
        });
        scheduler.set_thread_handle(thread_handle);
        self.running_tasks.insert(task_ctx.task_id, scheduler);
    }

    fn compression_num_to_str(compression_type: u8) -> &'static str {
        if compression_type == LZ4_COMPRESSION {
            return "lz4";
        } else if compression_type == ZSTD_COMPRESSION {
            return "zstd";
        }
        ""
    }

    pub(crate) fn build(&self, task_id: &str, compression: &str) {
        let compression_type = match compression {
            "lz4" => LZ4_COMPRESSION,
            "zstd" => ZSTD_COMPRESSION,
            _ => NO_COMPRESSION,
        };
        let scheduler = self.running_tasks.get(task_id).unwrap().clone();
        scheduler
            .sender
            .send(LoadTaskMsg::Build { compression_type })
            .unwrap();
    }

    pub(crate) async fn flush(&self, task_id: &str) -> FlushResult {
        let scheduler = self.running_tasks.get(task_id).unwrap().clone();
        let (cb, fut) = tikv_util::future::paired_future_callback();
        scheduler.sender.send(LoadTaskMsg::Flush { cb }).unwrap();
        fut.await.unwrap()
    }

    pub(crate) async fn put_chunk(
        &self,
        task_id: &str,
        writer_id: u64,
        chunk_id: u64,
        chunk_data: Bytes,
    ) -> PutChunkResult {
        let scheduler = self.running_tasks.get(task_id).unwrap().clone();
        let (cb, fut) = tikv_util::future::paired_future_callback();
        scheduler
            .sender
            .send(LoadTaskMsg::AddChunk {
                writer_id,
                chunk_id,
                chunk_data,
                cb,
            })
            .unwrap();
        fut.await.unwrap()
    }

    pub(crate) fn delete(&self, task_id: &str) {
        if let Some((_, scheduler)) = self.running_tasks.remove(task_id) {
            scheduler.cancel("deleted".to_string());
            scheduler.sender.send(LoadTaskMsg::Cleanup).unwrap();
        }
    }
}
