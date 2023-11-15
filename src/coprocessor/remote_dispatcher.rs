// Copyright 2023 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    ops::{Add, Deref},
    sync::Arc,
    time::Duration,
};

use async_trait::async_trait;
use bytes::{Buf, BufMut, Bytes};
use futures_util::compat::Future01CompatExt;
use kvengine::{SnapAccess, UserMeta, LOCK_CF};
use kvproto::kvrpcpb::ExecDetailsV2;
use protobuf::Message;
use security::SecurityManager;
use tidb_query_common::execute_stats::ExecSummary;
use tikv_alloc::MemoryTraceGuard;
use tikv_kv::{Engine, Statistics};
use tikv_util::{deadline::Deadline, time::Instant, timer::GLOBAL_TIMER_HANDLE};
use tipb::DagRequest;
use txn_types::{TimeStamp, TsSet};

use crate::{
    coprocessor::{
        metrics::COPR_REMOTE_DAG_ESTIMATE_BLOCKS_HISTOGRAM, Error, ReqContext, RequestHandler,
        Result, MEMTRACE_ROOT, REQ_TYPE_DAG,
    },
    storage::txn::check_locks,
};

const ANALYZE_CACHE_CAPACITY: u64 = 64;

pub const REMOTE_ANALYZE_TIMEOUT: Duration = Duration::from_secs(300);

pub const REMOTE_COP_FORMAT_V1: u32 = 1;

#[derive(Default, Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
#[serde(rename_all = "kebab-case")]
pub struct RemoteAnalysisRequest {
    pub key: String,
    pub req_bytes: Vec<u8>,
    pub snap_bytes: Vec<u8>,
    pub max_handle_duration: Duration,
    pub peer: String,
}

#[derive(Clone)]
pub struct RemoteContext {
    pub core: Arc<RemoteContextCore>,
}

impl Deref for RemoteContext {
    type Target = RemoteContextCore;
    fn deref(&self) -> &Self::Target {
        &self.core
    }
}

pub struct RemoteContextCore {
    pub remote_analyze_url: String,
    pub cop_worker_provider: Arc<dyn CopWorkerProvider>,
    pub cop_min_blocks: usize,
    pub cop_white_list: Vec<u32>,
    pub runtime: Arc<tokio::runtime::Runtime>,
    pub analyze_cache: moka::future::Cache<String, Result<Vec<u8>>>,
    pub client: security::HttpClient,
}

pub trait CopWorkerProvider: Send + Sync {
    fn get(&self, keyspace_id: u32, start_ts: u64) -> Option<String>;
}

pub struct StaticCopWorkerProvider {
    worker_url: String,
}

impl CopWorkerProvider for StaticCopWorkerProvider {
    fn get(&self, _keyspace_id: u32, _start_ts: u64) -> Option<String> {
        if self.worker_url.is_empty() {
            return None;
        }
        Some(self.worker_url.clone())
    }
}

impl RemoteContext {
    pub fn new(
        remote_analyze_url: String,
        cop_worker_url: String,
        cop_min_blocks: usize,
        cop_white_list: Vec<u32>,
        security_mgr: Arc<SecurityManager>,
    ) -> Option<Self> {
        if remote_analyze_url.is_empty() && cop_worker_url.is_empty() {
            return None;
        }
        let runtime = Arc::new(
            tokio::runtime::Builder::new_multi_thread()
                .worker_threads(2)
                .enable_all()
                .thread_name("remote_coprocessor")
                .build()
                .unwrap(),
        );
        let client = security_mgr
            .http_client(hyper::Client::builder().pool_max_idle_per_host(0).clone())
            .unwrap();
        let analyze_cache = moka::future::Cache::builder()
            .max_capacity(ANALYZE_CACHE_CAPACITY)
            .time_to_live(REMOTE_ANALYZE_TIMEOUT * 2)
            .build();
        let cop_worker_provider = Arc::new(StaticCopWorkerProvider {
            worker_url: cop_worker_url,
        });
        Some(Self {
            core: Arc::new(RemoteContextCore {
                remote_analyze_url,
                cop_min_blocks,
                cop_worker_provider,
                cop_white_list,
                runtime,
                analyze_cache,
                client,
            }),
        })
    }
}

pub(crate) fn try_remote_dag_handler<E: Engine>(
    snap: Option<&kvengine::SnapAccess>,
    dag: &DagRequest,
    req_ctx: &ReqContext,
    remote_ctx: Option<RemoteContext>,
) -> Option<Box<dyn RequestHandler>> {
    let remote_ctx = remote_ctx?;
    let start_ts = req_ctx.txn_start_ts.into_inner();
    let snap = snap.cloned()?;
    let worker_addr = remote_ctx
        .cop_worker_provider
        .get(snap.get_keyspace_id(), start_ts)?;
    let keyspace_id = snap.get_keyspace_id();
    if !remote_ctx.cop_white_list.is_empty() && !remote_ctx.cop_white_list.contains(&keyspace_id) {
        return None;
    }
    let last_executor = dag.get_executors().last().unwrap();
    if last_executor.has_limit() {
        let limit = last_executor.get_limit();
        if limit.get_partition_by().is_empty() {
            // Do not offload coprocessor with limit because the actual cost may be much
            // smaller.
            return None;
        }
    }
    let mut ranges = Vec::with_capacity(req_ctx.ranges.len());
    for ran in &req_ctx.ranges {
        let start = Bytes::copy_from_slice(&ran.start);
        let end = Bytes::copy_from_slice(&ran.end);
        ranges.push((start, end))
    }
    let num_blocks = snap.estimated_range_blocks(&ranges);
    let min_blocks = calc_min_blocks(req_ctx.txn_start_ts, remote_ctx.cop_min_blocks);
    if num_blocks < min_blocks {
        return None;
    }
    COPR_REMOTE_DAG_ESTIMATE_BLOCKS_HISTOGRAM.observe(num_blocks as f64);
    let tag = format!("ks{}:{}:{}", keyspace_id, snap.get_id(), snap.get_version());
    info!("{} send remote coprocessor blocks:{}", tag, num_blocks);
    // reassemble a coprocessor request.
    let mut cop_req = kvproto::coprocessor::Request::default();
    cop_req.set_context(req_ctx.context.clone());
    cop_req.tp = REQ_TYPE_DAG;
    cop_req.start_ts = req_ctx.txn_start_ts.into_inner();
    cop_req.data = dag.write_to_bytes().unwrap();
    cop_req.ranges = req_ctx.ranges.clone().into();
    Some(
        RemoteDagDispatcher::new(
            cop_req,
            snap,
            ranges,
            req_ctx.bypass_locks.clone(),
            remote_ctx,
            worker_addr,
            tag,
            req_ctx.deadline,
        )
        .into_boxed(),
    )
}

pub struct RemoteDagDispatcher {
    req: kvproto::coprocessor::Request,
    snap: SnapAccess,
    ranges: Vec<(Bytes, Bytes)>,
    bypass_locks: TsSet,
    remote_ctx: RemoteContext,
    worker_addr: String,
    tag: String,
    exec_details: Option<ExecDetailsV2>,
    deadline: Deadline,
}

impl RemoteDagDispatcher {
    fn new(
        req: kvproto::coprocessor::Request,
        snap: SnapAccess,
        ranges: Vec<(Bytes, Bytes)>,
        bypass_locks: TsSet,
        remote_ctx: RemoteContext,
        worker_addr: String,
        tag: String,
        deadline: Deadline,
    ) -> Self {
        Self {
            req,
            snap,
            ranges,
            bypass_locks,
            remote_ctx,
            worker_addr,
            tag,
            exec_details: None,
            deadline,
        }
    }

    fn check_locks(&self) -> Result<()> {
        let mut stats = Statistics::default();
        let mut lock_iter =
            self.snap
                .new_iterator(LOCK_CF, false, false, Some(self.req.start_ts), true);
        for (start, end) in &self.ranges {
            lock_iter.set_range(start.clone(), end.clone());
            check_locks(
                &mut lock_iter,
                &self.bypass_locks,
                self.req.start_ts,
                &mut stats,
            )?;
        }
        Ok(())
    }

    async fn dispatch(&self, deadline: Instant) -> Result<Vec<u8>> {
        let cop_req: Vec<u8> = self.req.write_to_bytes().unwrap();
        let mem_data = self.build_mem_data();
        let (_, snap_data) = self.snap.marshal(&self.ranges, true, false);
        let req_body = encode_remote_cop_request(cop_req, mem_data, snap_data);
        let req = hyper::Request::builder()
            .method(hyper::Method::POST)
            .uri(&self.worker_addr)
            .header("content-type", "application/octet-stream")
            .body(hyper::Body::from(req_body))
            .map_err(|e| Error::Other(e.to_string()))?;
        let client = self.remote_ctx.client.clone();
        let (tx, rx) = tokio::sync::oneshot::channel();
        self.remote_ctx.runtime.spawn(async move {
            let res = tokio::time::timeout(
                deadline.saturating_duration_since(Instant::now_coarse()),
                async move {
                    let response = client
                        .request(req)
                        .await
                        .map_err(|e| Error::Other(e.to_string()))?;
                    let success = response.status().is_success();
                    let body = hyper::body::to_bytes(response.into_body())
                        .await
                        .map_err(|e| Error::Other(e.to_string()))?;
                    if !success {
                        return Err(Error::Other(
                            String::from_utf8_lossy(body.chunk()).to_string(),
                        ));
                    }
                    Ok(body.to_vec())
                },
            )
            .await
            .unwrap_or_else(|_| Err(Error::DeadlineExceeded));
            tx.send(res).unwrap();
        });
        rx.await.unwrap()
    }

    async fn dispatch_with_retry(&self, deadline: Deadline) -> Result<Vec<u8>> {
        let mut retry = 0;
        loop {
            let res = self.dispatch(deadline.inner()).await;
            match res {
                Ok(data) => {
                    return Ok(data);
                }
                Err(Error::DeadlineExceeded) => return Err(Error::DeadlineExceeded),
                Err(err) => {
                    if deadline.check().is_err() {
                        return Err(Error::DeadlineExceeded);
                    }
                    retry += 1;
                    error!(
                        "{} remote coprocessor error {:?}, retry {}",
                        self.tag, err, retry
                    );
                    let _ = GLOBAL_TIMER_HANDLE
                        .delay(std::time::Instant::now().add(Duration::from_secs(5)))
                        .compat()
                        .await;
                }
            }
        }
    }

    fn build_mem_data(&self) -> Vec<u8> {
        let mut mem_iterator =
            self.snap
                .new_memtable_iterator(0, false, false, Some(self.req.start_ts));
        let mut rows = vec![];
        for (range_start, range_end) in &self.ranges {
            mem_iterator.seek(range_start.chunk());
            while mem_iterator.valid() {
                let key = mem_iterator.key();
                if key >= range_end.chunk() {
                    break;
                }
                rows.push(kvengine::table::Row {
                    key: key.to_vec(),
                    user_meta: UserMeta::from_slice(mem_iterator.user_meta()),
                    value: mem_iterator.val().to_vec(),
                });
                mem_iterator.next();
            }
        }
        let mem_size = bincode::serialized_size(&rows)
            .map_err(|e| Error::Other(e.to_string()))
            .unwrap();
        let mut mem_data = Vec::with_capacity(mem_size as usize + 4);
        mem_data.put_u32_le(REMOTE_COP_FORMAT_V1);
        bincode::serialize_into(&mut mem_data, &rows).unwrap();
        mem_data
    }
}

#[async_trait]
impl RequestHandler for RemoteDagDispatcher {
    async fn handle_request(&mut self) -> Result<MemoryTraceGuard<kvproto::coprocessor::Response>> {
        self.check_locks()?;
        let ret = self.dispatch_with_retry(self.deadline).await;
        match ret {
            Ok(data) => {
                let memory_size = data.capacity();
                let mut resp = kvproto::coprocessor::Response::default();
                resp.merge_from_bytes(&data).unwrap();
                self.exec_details = resp.exec_details_v2.take();
                Ok(MEMTRACE_ROOT.trace_guard(resp, memory_size))
            }
            Err(Error::Other(e)) => {
                error!("{} remote coprocessor failed, error {}", self.tag, e);
                let mut resp = kvproto::coprocessor::Response::default();
                resp.set_other_error(e);
                Ok(resp.into())
            }
            Err(e) => {
                error!("{} remote coprocessor failed, error {:?}", self.tag, e);
                Err(e)
            }
        }
    }

    async fn handle_streaming_request(
        &mut self,
    ) -> Result<(Option<kvproto::coprocessor::Response>, bool)> {
        unimplemented!()
    }

    fn collect_scan_statistics(&mut self, dest: &mut Statistics) {
        if let Some(exec_details_v2) = self.exec_details.as_ref() {
            if let Some(scan_detail_v2) = exec_details_v2.scan_detail_v2.as_ref() {
                dest.processed_size = scan_detail_v2.processed_versions_size as usize;
                dest.write.processed_keys = scan_detail_v2.processed_versions as usize;
            }
        }
    }

    fn collect_scan_summary(&mut self, dest: &mut ExecSummary) {
        if let Some(exec_details_v2) = self.exec_details.as_ref() {
            if let Some(time_details) = exec_details_v2.time_detail.as_ref() {
                dest.time_processed_ns = time_details.process_wall_time_ms as usize * 1000000;
            }
        }
    }
}

/// If the query has already run for a long time, the additional latency for
/// offloading to remote coprocessor is non-significant, we can decreases the
/// min blocks to reduce the tikv-server resource consumption.
fn calc_min_blocks(start_ts: TimeStamp, config_min_blocks: usize) -> usize {
    const LONG_QUERY_MS: u64 = 15 * 1000;
    let elapsed_ms = TimeStamp::physical_now().saturating_sub(start_ts.physical());
    if elapsed_ms > LONG_QUERY_MS * 4 {
        // 60s
        config_min_blocks / 8
    } else if elapsed_ms > LONG_QUERY_MS * 2 {
        // 30s
        config_min_blocks / 4
    } else if elapsed_ms > LONG_QUERY_MS {
        // 15s
        config_min_blocks / 2
    } else {
        config_min_blocks
    }
}

pub fn encode_remote_cop_request(
    cop_req: Vec<u8>,
    mem_data: Vec<u8>,
    snap_data: Vec<u8>,
) -> Vec<u8> {
    let extra_len = 4 * 4;
    let mut req_body: Vec<u8> =
        Vec::with_capacity(extra_len + cop_req.len() + mem_data.len() + snap_data.len());
    req_body.put_u32_le(REMOTE_COP_FORMAT_V1);
    req_body.put_u32_le(cop_req.len() as u32);
    req_body.extend_from_slice(&cop_req);
    req_body.put_u32_le(mem_data.len() as u32);
    req_body.extend_from_slice(&mem_data);
    req_body.put_u32_le(snap_data.len() as u32);
    req_body.extend_from_slice(&snap_data);
    req_body
}

pub fn decode_remote_cop_request(body: &[u8]) -> Result<(&[u8], &[u8], &[u8])> {
    let mut body_buf = body;
    let err = Error::Other("failed to decode cop request".to_string());
    if body_buf.len() < 4 {
        return Err(err);
    }
    let format = body_buf.get_u32_le();
    if format != REMOTE_COP_FORMAT_V1 {
        return Err(err);
    }
    if body_buf.len() < 4 {
        return Err(err);
    }
    let req_data_len = body_buf.get_u32_le() as usize;
    if body_buf.len() < req_data_len {
        return Err(err);
    }
    let req_data = &body_buf[..req_data_len];
    body_buf = &body_buf[req_data_len..];
    if body_buf.len() < 4 {
        return Err(Error::Other("invalid cop request".to_string()));
    }
    let mem_data_len = body_buf.get_u32_le() as usize;
    if body_buf.len() < mem_data_len {
        return Err(err);
    }
    let mem_data = &body_buf[..mem_data_len];
    body_buf = &body_buf[mem_data_len..];
    if body_buf.len() < 4 {
        return Err(Error::Other("invalid cop request".to_string()));
    }
    let snap_data_len = body_buf.get_u32_le() as usize;
    if body_buf.len() < snap_data_len {
        return Err(err);
    }
    let snap_data = &body_buf[..snap_data_len];
    Ok((req_data, mem_data, snap_data))
}

#[test]
fn test_remote_cop_coded() {
    let cop_req = b"cop_req".to_vec();
    let mem_data = b"mem_data".to_vec();
    let snap_data = b"snap_data".to_vec();
    let req_body = encode_remote_cop_request(cop_req, mem_data, snap_data);
    let (cop_req, mem_data, snap_data) = decode_remote_cop_request(&req_body).unwrap();
    assert_eq!(cop_req, "cop_req".as_bytes());
    assert_eq!(mem_data, "mem_data".as_bytes());
    assert_eq!(snap_data, "snap_data".as_bytes());
}

#[test]
fn test_calc_min_blocks() {
    let conf_min_blocks = 512usize;
    let ts = TimeStamp::default();
    assert_eq!(calc_min_blocks(ts, conf_min_blocks), 64);
    let ts = TimeStamp::max();
    assert_eq!(calc_min_blocks(ts, conf_min_blocks), 512);

    let phys_now = TimeStamp::physical_now();
    let ts = TimeStamp::compose(phys_now - 70 * 1000, 1);
    assert_eq!(calc_min_blocks(ts, conf_min_blocks), 64);
    let ts = TimeStamp::compose(phys_now - 40 * 1000, 1);
    assert_eq!(calc_min_blocks(ts, conf_min_blocks), 128);
    let ts = TimeStamp::compose(phys_now - 20 * 1000, 1);
    assert_eq!(calc_min_blocks(ts, conf_min_blocks), 256);
    let ts = TimeStamp::compose(phys_now - 10 * 1000, 1);
    assert_eq!(calc_min_blocks(ts, conf_min_blocks), 512);
    let ts = TimeStamp::compose(phys_now + 20 * 1000, 1);
    assert_eq!(calc_min_blocks(ts, conf_min_blocks), 512);
}
