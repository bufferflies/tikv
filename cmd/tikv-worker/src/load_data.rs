// Copyright 2023 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    collections::{HashMap, HashSet},
    fs,
    fs::File,
    io::{BufReader, Read, Seek, SeekFrom, Write},
    mem,
    path::PathBuf,
    str::FromStr,
    sync::{Arc, Mutex},
    time::Duration,
};

use bytes::{Buf, BufMut, Bytes, BytesMut};
use http::{header, Method, Request, Response, StatusCode, Uri};
use hyper::Body;
use kvengine::{
    dfs,
    dfs::Options,
    table::{
        sstable::{Builder, LZ4_COMPRESSION, NO_COMPRESSION, ZSTD_COMPRESSION},
        Value,
    },
    UserMeta,
};
use kvproto::metapb;
use pd_client::PdClient;
use protobuf::Message;
use tikv_util::{
    codec::bytes::encode_bytes,
    error, info,
    mpsc::{Receiver, Sender},
    time::Instant,
    warn,
};

use crate::{
    common::{get_body, get_u64_param, make_response},
    error::Error,
};

type Result<T> = std::result::Result<T, Error>;

pub(crate) const MAX_IN_MEM_SIZE: usize = 256 * 1024 * 1024;

const SST_FILE_SIZE: usize = 16 * 1024 * 1024;
const REGION_SIZE: usize = 1024 * 1024 * 1024; // 1GB
const COARSE_SPLIT_SIZE: usize = 32 * 1024 * 1024 * 1024; // 32GB
const BLOCK_SIZE: usize = 64 * 1024;
const ZSTD_COMPRESSION_LEVEL: i32 = 3;
const FLUSH_FILE_CONCURRENCY: usize = 8;
const CREATE_FILE_CONCURRENCY: usize = 32;
const INGEST_CONCURRENCY: usize = 4;

const ALLOCATE_ID_TIMEOUT: Duration = Duration::from_secs(10 * 60);
const RETRY_SLEEP_DURATION: Duration = Duration::from_millis(100);
const MAX_RETRY_TIMES: usize = 10;
const MAX_SLEEP_DURATION: Duration = Duration::from_secs(30);

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

pub struct KvPair {
    key: Bytes,
    val: Bytes,
}

impl KvPair {
    fn new(key: Bytes, val: Bytes) -> KvPair {
        Self { key, val }
    }
}

pub struct KvPairsReader {
    key_buf: Vec<u8>,
    val_buf: Vec<u8>,
    val_base_len: usize,
    count: usize,
    idx: usize,
    buf_reader: BufReader<File>,
}

impl KvPairsReader {
    fn new(start_ts: u64, commit_ts: u64, count: usize, mut file: File) -> Self {
        file.seek(SeekFrom::Start(0)).unwrap();
        let buf_reader = BufReader::with_capacity(64 * 1024, file);
        let um = UserMeta::new(start_ts, commit_ts);
        let val_buf = Value::encode_buf(0, &um.to_array(), commit_ts, &[]);
        let val_base_len = val_buf.len();
        Self {
            key_buf: vec![],
            val_buf,
            val_base_len,
            count,
            idx: 0,
            buf_reader,
        }
    }

    fn key(&self) -> &[u8] {
        &self.key_buf
    }

    fn valid(&self) -> bool {
        self.idx <= self.count
    }

    fn next(&mut self) {
        self.idx += 1;
        if self.idx > self.count {
            return;
        }
        let mut key_len_buf = [0u8; 2];
        self.buf_reader.read_exact(&mut key_len_buf[..]).unwrap();
        let key_len = u16::from_le_bytes(key_len_buf);
        self.key_buf.resize(key_len as usize, 0);
        self.buf_reader.read_exact(&mut self.key_buf[..]).unwrap();
        let mut val_len_buf = [0u8; 4];
        self.buf_reader.read_exact(&mut val_len_buf[..]).unwrap();
        let val_len = u32::from_le_bytes(val_len_buf);
        self.val_buf.resize(self.val_base_len + val_len as usize, 0);
        self.buf_reader
            .read_exact(&mut self.val_buf[self.val_base_len..])
            .unwrap();
    }
}

#[derive(Default, Debug, Serialize, Deserialize, Clone)]
#[serde(default)]
#[serde(rename_all = "kebab-case")]
pub struct SstMeta {
    pub id: u64,
    pub smallest: Vec<u8>,
    pub biggest: Vec<u8>,
    pub size: usize,
    pub keys: usize,
    pub encoded_smallest: Vec<u8>,
    pub encoded_biggest: Vec<u8>,
}

pub(crate) struct LoadTaskWorker {
    ctx: LoadDataContext,
    task_ctx: TaskContext,
    task_dir: PathBuf,
    kv_pairs: Vec<KvPair>,
    in_mem_size: usize,
    file_idx: usize,
    readers: Vec<KvPairsReader>,
    reader_errs: Vec<Error>,
    scheduler: LoadTaskScheduler,
    receiver: Receiver<LoadTaskMsg>,
    handled_chunks: HashSet<u64>,
    cached_file_ids: Vec<u64>,
    file_tx: Sender<Result<KvPairsReader>>,
    file_rx: Receiver<Result<KvPairsReader>>,
}

pub(crate) enum LoadTaskMsg {
    AddChunk {
        chunk_id: u64,
        chunk_data: Bytes,
    },
    Build {
        chunk_ids: Vec<u64>,
        compression_type: u8,
    },
    Cleanup,
}

#[derive(Clone)]
pub(crate) struct LoadTaskScheduler {
    sender: Sender<LoadTaskMsg>,
    states: Arc<Mutex<LoadTaskStates>>,
}

impl LoadTaskScheduler {
    pub(crate) fn cancel(&self, err: String) {
        warn!("canceled {}", err);
        let mut states = self.states.lock().unwrap();
        states.canceled = true;
        states.error = err;
    }

    pub(crate) fn is_canceled(&self) -> bool {
        let states = self.states.lock().unwrap();
        states.canceled
    }

    pub(crate) fn add_created_files_count(&self) {
        let mut states = self.states.lock().unwrap();
        states.created_files += 1;
    }

    pub(crate) fn add_ingested_regions(&self) {
        let mut states = self.states.lock().unwrap();
        states.ingested_regions += 1;
    }

    pub(crate) fn set_finished(&self) {
        let mut states = self.states.lock().unwrap();
        states.finished = true;
    }

    pub(crate) fn is_finished(&self) -> bool {
        let states = self.states.lock().unwrap();
        states.finished
    }
}

#[derive(Clone)]
pub(crate) struct LoadDataContext {
    dir: PathBuf,
    dfs: Arc<dyn dfs::Dfs>,
    pd: Arc<dyn PdClient>,
    runtime: Arc<tokio::runtime::Runtime>,
    max_in_mem_size: usize,
}

#[derive(Copy, Clone)]
pub(crate) struct TaskContext {
    start_ts: u64,
    commit_ts: u64,
}

#[derive(Default, Debug, Serialize, Deserialize, Clone)]
#[serde(default)]
#[serde(rename_all = "kebab-case")]
pub struct LoadTaskStates {
    pub start_ts: u64,
    pub canceled: bool,
    pub finished: bool,
    pub error: String,
    pub created_files: usize,
    pub ingested_regions: usize,
}

impl LoadTaskWorker {
    pub(crate) fn new(context: LoadDataContext, task_ctx: TaskContext) -> Self {
        let (sender, receiver) = tikv_util::mpsc::unbounded();
        let mut states = LoadTaskStates::default();
        states.start_ts = task_ctx.start_ts;
        let scheduler = LoadTaskScheduler {
            sender,
            states: Arc::new(Mutex::new(states)),
        };
        let (file_tx, file_rx) = tikv_util::mpsc::unbounded();
        let task_dir = context.dir.join(format!("{}", task_ctx.start_ts));
        Self {
            ctx: context,
            task_ctx,
            task_dir,
            kv_pairs: vec![],
            in_mem_size: 0,
            file_idx: 0,
            readers: vec![],
            reader_errs: vec![],
            scheduler,
            receiver,
            handled_chunks: Default::default(),
            cached_file_ids: vec![],
            file_tx,
            file_rx,
        }
    }

    pub(crate) fn run(&mut self) {
        self.init_task_dir();
        while let Ok(msg) = self.receiver.recv() {
            match msg {
                LoadTaskMsg::AddChunk {
                    chunk_id,
                    chunk_data,
                } => {
                    self.handle_add_chunk(chunk_id, chunk_data);
                    if !self.reader_errs.is_empty() {
                        self.scheduler
                            .cancel(self.reader_errs.first().unwrap().to_string());
                        let remained = self.file_idx - self.reader_errs.len() - self.readers.len();
                        for _ in 0..remained {
                            self.recv_reader();
                        }
                    }
                }
                LoadTaskMsg::Build {
                    chunk_ids,
                    compression_type,
                } => {
                    if self.scheduler.is_canceled() {
                        warn!("task {} is canceled, do not build", self.task_ctx.start_ts);
                        continue;
                    }
                    if self.scheduler.is_finished() {
                        warn!("task {} is finished, skip build", self.task_ctx.start_ts);
                        continue;
                    }
                    if let Err(err) = self.build(chunk_ids, compression_type) {
                        error!("build failed {:?}", err);
                        self.scheduler.cancel(err.to_string());
                    }
                    continue;
                }
                LoadTaskMsg::Cleanup => {
                    self.remove_local_files();
                    return;
                }
            }
        }
    }

    pub(crate) fn get_scheduler(&self) -> LoadTaskScheduler {
        self.scheduler.clone()
    }

    pub(crate) fn handle_add_chunk(&mut self, chunk_id: u64, chunk_data: Bytes) {
        info!("handle add chunk {}, len {}", chunk_id, chunk_data.len());
        if self.handled_chunks.contains(&chunk_id) {
            warn!(
                "{} skip duplicated chunk {}",
                self.task_ctx.start_ts, chunk_id
            );
            return;
        }
        if self.scheduler.is_canceled() {
            warn!(
                "task {} is canceled, skip add chunk {}",
                self.task_ctx.start_ts, chunk_id
            );
            return;
        }
        if self.scheduler.is_finished() {
            warn!(
                "task {} is finished, skip add chunk {}",
                self.task_ctx.start_ts, chunk_id
            );
            return;
        }
        self.handled_chunks.insert(chunk_id);
        let mut offset = 0;
        while offset < chunk_data.len() {
            let key_len = (&chunk_data[offset..]).get_u16_le();
            offset += 2;
            let key = chunk_data.slice(offset..offset + key_len as usize);
            offset += key_len as usize;
            let val_len = (&chunk_data[offset..]).get_u32_le();
            offset += 4;
            let val = chunk_data.slice(offset..offset + val_len as usize);
            offset += val_len as usize;
            self.kv_pairs.push(KvPair::new(key, val));
            self.in_mem_size += 2 + key_len as usize + 4 + val_len as usize;
        }
        if self.in_mem_size > self.ctx.max_in_mem_size {
            info!(
                "{} flush to local file on in_mem_size {}",
                self.task_ctx.start_ts, self.in_mem_size
            );
            let kv_pairs = mem::take(&mut self.kv_pairs);
            let tx = self.file_tx.clone();
            let task_ctx = self.task_ctx;
            let file_path = self.file_path(self.file_idx);
            let in_mem_size = self.in_mem_size;
            self.file_idx += 1;
            std::thread::spawn(move || {
                let res = flush_to_local_file(kv_pairs, task_ctx, file_path, in_mem_size);
                tx.send(res).unwrap();
            });
            if self.file_idx + FLUSH_FILE_CONCURRENCY > self.readers.len() {
                self.recv_reader();
            }
            self.in_mem_size = 0;
        }
    }

    fn recv_reader(&mut self) {
        match self.file_rx.recv().unwrap() {
            Err(err) => {
                self.reader_errs.push(err);
            }
            Ok(reader) => {
                self.readers.push(reader);
            }
        }
    }

    fn alloc_file_id(&mut self) -> u64 {
        if let Some(id) = self.cached_file_ids.pop() {
            return id;
        }
        let start = Instant::now();
        let count = 64;
        loop {
            match futures::executor::block_on(self.ctx.pd.batch_get_tso(count as u32)) {
                Ok(ts) => {
                    let last = ts.into_inner();
                    let first = last - count as u64 + 1;
                    self.cached_file_ids = (first..=last).rev().collect();
                    return self.cached_file_ids.pop().unwrap();
                }
                Err(err) => {
                    error!("failed to allocate file id from PD {:?}", err);
                    std::thread::sleep(Duration::from_secs(1));
                    if start.saturating_elapsed() > ALLOCATE_ID_TIMEOUT {
                        panic!("allocate file id timeout");
                    }
                }
            }
        }
    }

    fn build(&mut self, chunk_ids: Vec<u64>, compression_type: u8) -> Result<()> {
        for id in chunk_ids {
            if !self.handled_chunks.contains(&id) {
                return Err(Error::CheckError(format!(
                    "{} chunk {} is not handled",
                    self.task_ctx.start_ts, id
                )));
            }
        }
        while self.readers.len() + self.reader_errs.len() < self.file_idx {
            self.recv_reader();
        }
        if !self.reader_errs.is_empty() {
            return Err(self.reader_errs.pop().unwrap());
        }
        if !self.kv_pairs.is_empty() {
            let kv_pairs = mem::take(&mut self.kv_pairs);
            let reader = flush_to_local_file(
                kv_pairs,
                self.task_ctx,
                self.file_path(self.file_idx),
                self.in_mem_size,
            )?;
            self.readers.push(reader);
            self.file_idx += 1;
        }
        if self.readers.is_empty() {
            info!("{} build empty data", self.task_ctx.start_ts);
            self.scheduler.set_finished();
            return Ok(());
        }

        info!("{} start build", self.task_ctx.start_ts);
        let (tx, rx) = tikv_util::mpsc::unbounded();
        let mut sent_count = 0;
        let mut recv_count = 0;
        let readers = mem::take(&mut self.readers);
        let mut merge_iter = MergeIterator::new(readers);
        let mut sst_metas = vec![];
        let mut errs = vec![];
        while merge_iter.valid() {
            let batch = self.read_batch(&mut merge_iter)?;
            if self.scheduler.is_canceled() {
                break;
            }
            if batch.is_empty() {
                break;
            }
            let file_id = self.alloc_file_id();
            self.spawn_build_file(file_id, batch, tx.clone(), compression_type);
            sent_count += 1;
            if sent_count > CREATE_FILE_CONCURRENCY {
                recv_count += 1;
                match rx.recv().unwrap() {
                    Err(err) => {
                        errs.push(err);
                        break;
                    }
                    Ok(sst_meta) => {
                        sst_metas.push(sst_meta);
                        self.scheduler.add_created_files_count();
                    }
                }
            }
        }
        for _ in 0..(sent_count - recv_count) {
            match rx.recv().unwrap() {
                Err(err) => {
                    error!("{} create file failed {}", self.task_ctx.start_ts, err);
                    errs.push(err);
                }
                Ok(sst_meta) => {
                    sst_metas.push(sst_meta);
                    self.scheduler.add_created_files_count();
                }
            }
        }
        if !errs.is_empty() {
            return Err(errs.pop().unwrap());
        }
        info!("{} finish build", self.task_ctx.start_ts);
        self.ingest(sst_metas)
    }

    fn spawn_build_file(
        &self,
        file_id: u64,
        mut batch: BytesMut,
        sender: Sender<Result<SstMeta>>,
        compression_type: u8,
    ) {
        let ctx = self.ctx.clone();
        let start_ts = self.task_ctx.start_ts;
        self.ctx.runtime.spawn(async move {
            info!("{} start build sst file {}", start_ts, file_id);
            let mut builder = Builder::new(
                file_id,
                BLOCK_SIZE,
                compression_type,
                ZSTD_COMPRESSION_LEVEL,
            );
            let mut entries = 0;
            let mut offset = 0;
            while offset < batch.len() {
                let key_len = (&batch[offset..]).get_u16_le() as usize;
                offset += 2;
                let key = &batch[offset..offset + key_len];
                offset += key_len;
                let val_len = (&batch[offset..]).get_u32_le() as usize;
                offset += 4;
                let val = &batch[offset..offset + val_len];
                offset += val_len;
                builder.add(key, &Value::decode(val), None);
                entries += 1;
            }
            batch.clear();
            builder.finish(0, &mut batch);
            let data = batch.freeze();
            let sst_meta = SstMeta {
                id: file_id,
                smallest: builder.get_smallest().to_vec(),
                biggest: builder.get_biggest().to_vec(),
                size: data.len(),
                keys: entries,
                encoded_smallest: encode_bytes(builder.get_smallest()),
                encoded_biggest: encode_bytes(builder.get_biggest()),
            };
            info!("{} finish build sst file {:?}", start_ts, sst_meta);
            let opts = Options::new(0, 0);
            let res = ctx
                .dfs
                .create(file_id, data, opts)
                .await
                .map(|_| sst_meta)
                .map_err(|e| Error::from(e));
            sender.send(res).unwrap();
        });
    }

    fn read_batch(&mut self, merge_iter: &mut MergeIterator) -> Result<BytesMut> {
        let mut buf = BytesMut::with_capacity(SST_FILE_SIZE);
        while merge_iter.valid() {
            let key = merge_iter.key();
            let key_len = key.len();
            let val = merge_iter.value();
            let val_len = val.len();
            if buf.len() + 2 + key_len + 4 + val_len > buf.capacity() {
                return Ok(buf);
            }
            buf.put_u16_le(key_len as u16);
            buf.extend_from_slice(key);
            buf.put_u32_le(val_len as u32);
            buf.extend_from_slice(val);
            merge_iter.next()?;
        }
        Ok(buf)
    }

    fn split_regions(&self, split_keys: &[Vec<u8>]) -> Result<Vec<u64>> {
        info!("{} start split", self.task_ctx.start_ts);
        let mut retry = 0;
        let mut split_keys = split_keys.to_owned();
        let mut new_regions_id = Vec::with_capacity(split_keys.len());
        loop {
            let mut unprocessed_keys = Vec::with_capacity(split_keys.len());
            for split_key in &split_keys {
                let region = self.ctx.pd.get_region(split_key)?;
                let start_key = region.get_start_key();
                if start_key == split_key {
                    new_regions_id.push(region.get_id());
                    continue;
                }
                unprocessed_keys.push(split_key.clone());
            }
            if unprocessed_keys.is_empty() {
                break;
            }

            let result = self
                .ctx
                .runtime
                .block_on(self.ctx.pd.split_regions(unprocessed_keys.clone()));
            match result {
                Err(e) => {
                    error!("{} split failed {:?}", self.task_ctx.start_ts, e);
                    if retry >= MAX_RETRY_TIMES {
                        return Err(Error::PdError(e));
                    }
                    std::thread::sleep(std::cmp::max(
                        MAX_SLEEP_DURATION,
                        2_u32.pow(retry as u32) * RETRY_SLEEP_DURATION,
                    ));
                }
                Ok(regions_id) => {
                    new_regions_id.extend_from_slice(&regions_id);
                    break;
                }
            }
            split_keys = unprocessed_keys.clone();
            retry += 1;
        }
        new_regions_id.sort();
        new_regions_id.dedup();
        info!(
            "{} finish split, new regions_id {:?}",
            self.task_ctx.start_ts, new_regions_id
        );
        Ok(new_regions_id)
    }

    fn ingest(&mut self, mut sst_metas: Vec<SstMeta>) -> Result<()> {
        if sst_metas.is_empty() {
            return Ok(());
        }
        info!("{} start ingest", self.task_ctx.start_ts);
        sst_metas.sort_by(|a, b| a.id.cmp(&b.id));
        let mut coarse_split_keys = gen_split_keys(&sst_metas, COARSE_SPLIT_SIZE);
        // split at last key so the last region will not be split by other concurrent
        // load_data and get epoch not match error.
        let mut last_key = sst_metas.last().unwrap().biggest.clone();
        last_key.push(0);
        coarse_split_keys.push(encode_bytes(&last_key));
        let new_regions_id = self.split_regions(&coarse_split_keys)?;
        for i in 0..coarse_split_keys.len() {
            let start_key = coarse_split_keys[i].clone();
            let end_key = if i + 1 == coarse_split_keys.len() {
                let mut last = sst_metas.last().unwrap().encoded_biggest.clone();
                last.push(0);
                last
            } else {
                coarse_split_keys[i + 1].clone()
            };
            let group_ssts = get_ssts_in_range(&sst_metas, &start_key, &end_key);
            self.ingest_group(group_ssts)?;
        }
        let result = self.ctx.pd.scatter_regions_by_id(new_regions_id);
        if let Err(err) = result {
            error!(
                "{} scatter regions failed {:?}",
                self.task_ctx.start_ts, err
            );
        }
        self.scheduler.set_finished();
        info!("{} finished ingest", self.task_ctx.start_ts);
        Ok(())
    }

    fn ingest_group(&self, sst_metas: Vec<SstMeta>) -> Result<()> {
        let split_keys = gen_split_keys(&sst_metas, REGION_SIZE);
        self.split_regions(&split_keys)?;
        let first_key = sst_metas.first().unwrap().encoded_smallest.clone();
        let mut last_key = sst_metas.last().unwrap().encoded_biggest.clone();
        last_key.push(0);
        let regions =
            self.ctx
                .runtime
                .block_on(self.ctx.pd.scan_regions(first_key, last_key, usize::MAX))?;
        info!("scanned regions {:?}", regions);
        let (tx, rx) = tikv_util::mpsc::unbounded();
        let mut msg_cnt = 0;
        for mut pd_region in regions {
            let region = pd_region.get_region();
            let cs = build_ingest_files(region, &sst_metas, self.task_ctx.start_ts);
            if cs.get_ingest_files().get_table_creates().is_empty() {
                continue;
            }
            if self.scheduler.is_canceled() {
                return Err(Error::Canceled);
            }
            let pd_cli = self.ctx.pd.clone();
            let tx = tx.clone();
            self.ctx.runtime.spawn(async move {
                info!("ingest file {:?}", cs);
                let region = pd_region.take_region();
                let leader = pd_region.take_leader();
                let res = ingest_files_to_leader(pd_cli, cs, region, leader).await;
                let _ = tx.send(res);
            });
            if msg_cnt < INGEST_CONCURRENCY {
                msg_cnt += 1;
            } else {
                rx.recv().unwrap()?;
                self.scheduler.add_ingested_regions();
            }
        }
        for _ in 0..msg_cnt {
            rx.recv().unwrap()?;
            self.scheduler.add_ingested_regions();
        }
        Ok(())
    }

    fn file_path(&self, file_idx: usize) -> PathBuf {
        self.task_dir.join(format!("kv_pairs_{}", file_idx))
    }

    fn init_task_dir(&self) {
        if let Err(err) = fs::create_dir(&self.task_dir) {
            self.scheduler.cancel(format!("{:?}", err))
        }
    }

    fn remove_local_files(&self) {
        if let Err(err) = fs::remove_dir_all(&self.task_dir) {
            error!(
                "failed to remove task {}, {:?}",
                self.task_ctx.start_ts, err
            );
        } else {
            info!("removed local files for task {}", self.task_ctx.start_ts);
        }
    }
}

async fn ingest_files_to_leader(
    pd: Arc<dyn PdClient>,
    cs: kvenginepb::ChangeSet,
    region: metapb::Region,
    mut leader: metapb::Peer,
) -> Result<()> {
    let http_client = hyper::client::Client::new();
    loop {
        let store_id = if leader.store_id > 0 {
            leader.store_id
        } else {
            get_leader_store(&pd, region.get_id()).await?
        };
        let store = pd.get_store_async(store_id).await?;
        let uri = Uri::from_str(&format!(
            "http://{}/ingest_files?cluster_id={}",
            &store.status_address,
            pd.get_cluster_id().unwrap()
        ))
        .unwrap();
        let body = cs.write_to_bytes().unwrap();
        let req = Request::post(uri).body(Body::from(body))?;
        let resp = http_client.request(req).await?;
        if !resp.status().is_success() {
            let body = hyper::body::to_bytes(resp.into_body()).await?;
            let mut errpb = kvproto::errorpb::Error::new();
            errpb.merge_from_bytes(&body).unwrap();
            if errpb.has_region_not_initialized() {
                tokio::time::sleep(Duration::from_secs(1)).await;
                continue;
            } else if errpb.has_not_leader() {
                leader = errpb.mut_not_leader().take_leader();
                if leader.store_id == 0 {
                    tokio::time::sleep(Duration::from_secs(1)).await;
                }
                continue;
            } else {
                return Err(Error::IngestFiles(format!("{:?}", errpb)));
            }
        }
        return Ok(());
    }
}

async fn get_leader_store(pd: &Arc<dyn PdClient>, region_id: u64) -> Result<u64> {
    let start = Instant::now();
    let timeout = Duration::from_secs(60);
    while start.saturating_elapsed() < timeout {
        let res = pd.get_region_leader_by_id(region_id).await?;
        if res.is_none() {
            return Err(Error::RegionNotFound(region_id));
        }
        let (_, leader) = res.unwrap();
        if leader.store_id > 0 {
            return Ok(leader.store_id);
        }
        warn!("leader not found, retry get region leader by id");
        tokio::time::sleep(Duration::from_secs(1)).await;
    }
    Err(Error::LeaderNotFound(region_id))
}

fn build_ingest_files(
    region: &metapb::Region,
    sst_metas: &[SstMeta],
    start_ts: u64,
) -> kvenginepb::ChangeSet {
    let mut cs = kvenginepb::ChangeSet::default();
    cs.set_shard_id(region.get_id());
    cs.set_shard_ver(region.get_region_epoch().get_version());
    let ingest_files = cs.mut_ingest_files();
    let properties = ingest_files.mut_properties();
    properties.mut_keys().push(kvengine::INGEST_ID_KEY.into());
    properties
        .mut_values()
        .push(start_ts.to_le_bytes().to_vec());
    let table_creates = ingest_files.mut_table_creates();
    for sst_meta in sst_metas {
        if sst_meta.encoded_biggest < region.start_key {
            continue;
        }
        if !region.end_key.is_empty() && sst_meta.encoded_smallest >= region.end_key {
            break;
        }
        let mut table_create = kvenginepb::TableCreate::new();
        table_create.set_id(sst_meta.id);
        table_create.set_cf(0);
        table_create.set_level(3);
        table_create.set_smallest(sst_meta.smallest.clone());
        table_create.set_biggest(sst_meta.biggest.clone());
        table_creates.push(table_create);
    }
    cs
}

fn gen_split_keys(ssts: &[SstMeta], split_size: usize) -> Vec<Vec<u8>> {
    let mut keys = vec![ssts.first().unwrap().encoded_smallest.clone()];
    let mut size = 0;
    for sst in ssts {
        if size > split_size {
            keys.push(sst.encoded_smallest.clone());
            size = 0;
        }
        size += sst.size;
    }
    keys
}

fn get_ssts_in_range(ssts: &[SstMeta], start: &[u8], end: &[u8]) -> Vec<SstMeta> {
    let position = ssts
        .binary_search_by(|sst| sst.encoded_smallest.as_slice().cmp(start))
        .unwrap();
    let mut matched = vec![];
    for i in position..ssts.len() {
        let sst = &ssts[i];
        if sst.encoded_smallest.as_slice() >= end {
            break;
        }
        matched.push(sst.clone())
    }
    matched
}

fn flush_to_local_file(
    mut kv_pairs: Vec<KvPair>,
    task_ctx: TaskContext,
    path: PathBuf,
    in_mem_size: usize,
) -> Result<KvPairsReader> {
    kv_pairs.sort_by(|a, b| a.key.cmp(&b.key));
    let mut file = fs::OpenOptions::new()
        .create(true)
        .write(true)
        .read(true)
        .open(path)?;
    let mut buf: Vec<u8> = Vec::with_capacity(in_mem_size);
    for pair in &kv_pairs {
        buf.put_u16_le(pair.key.len() as u16);
        buf.extend_from_slice(pair.key.chunk());
        buf.put_u32_le(pair.val.len() as u32);
        buf.extend_from_slice(pair.val.chunk());
        if buf.len() >= 128 * 1024 {
            let _ = file.write(&buf)?;
            buf.clear();
        }
    }
    if !buf.is_empty() {
        let _ = file.write(&buf)?;
    }
    Ok(KvPairsReader::new(
        task_ctx.start_ts,
        task_ctx.commit_ts,
        kv_pairs.len(),
        file,
    ))
}

struct MergeIterator {
    #[allow(clippy::vec_box)]
    heap: Vec<Box<KvPairsReader>>,
    prev_key: Vec<u8>,
}

impl MergeIterator {
    fn new(readers: Vec<KvPairsReader>) -> Self {
        let mut heap = Vec::with_capacity(readers.len());
        for mut reader in readers {
            reader.next();
            heap.push(Box::new(reader));
        }
        let mut it = Self {
            heap,
            prev_key: vec![],
        };
        it.init_heap();
        it.prev_key = it.key().to_vec();
        it
    }

    fn init_heap(&mut self) {
        for i in (0..self.heap.len() / 2).rev() {
            self.down(i);
        }
    }

    fn down(&mut self, i0: usize) -> bool {
        let n = self.heap.len();
        let mut i = i0;
        loop {
            let left = 2 * i + 1;
            if left >= n {
                break;
            }
            let right = left + 1;
            let j = if right < n && self.less(right, left) {
                right
            } else {
                left
            };
            if !self.less(j, i) {
                break;
            }
            self.heap.swap(i, j);
            i = j;
        }
        i > i0
    }

    fn less(&mut self, a: usize, b: usize) -> bool {
        self.heap[a].key() < self.heap[b].key()
    }

    fn key(&self) -> &[u8] {
        self.heap[0].key()
    }

    fn value(&self) -> &[u8] {
        &self.heap[0].val_buf
    }

    fn valid(&self) -> bool {
        !self.heap.is_empty()
    }

    fn next(&mut self) -> Result<()> {
        let heap_len = self.heap.len();
        if heap_len == 0 {
            return Ok(());
        }
        let first = &mut self.heap[0];
        first.next();
        if !first.valid() {
            self.heap.swap(0, heap_len - 1);
            self.heap.pop();
            if !self.valid() {
                return Ok(());
            }
        }
        self.down(0);
        let key = self.heap[0].key();
        if key == self.prev_key.as_slice() {
            return Err(Error::DuplicatedKey(format!("{:?}", key)));
        }
        self.prev_key.truncate(0);
        self.prev_key.extend_from_slice(key);
        Ok(())
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
    ) -> Self {
        let context = LoadDataContext {
            pd,
            dir,
            dfs,
            runtime,
            max_in_mem_size,
        };
        Self {
            running_tasks: Arc::new(dashmap::DashMap::default()),
            ctx: context,
        }
    }

    pub(crate) fn get_task_states(&self, start_ts: u64) -> Option<LoadTaskStates> {
        self.running_tasks
            .get(&start_ts)
            .map(|x| x.states.lock().unwrap().clone())
    }

    pub(crate) fn list_tasks(&self) -> Vec<LoadTaskStates> {
        self.running_tasks
            .iter()
            .map(|x| x.states.lock().unwrap().clone())
            .collect()
    }

    pub(crate) fn has_task(&self, start_ts: u64) -> bool {
        self.running_tasks.contains_key(&start_ts)
    }

    pub(crate) fn init_task(&self, task_ctx: TaskContext) {
        let mut worker = LoadTaskWorker::new(self.ctx.clone(), task_ctx);
        let scheduler = worker.get_scheduler();
        std::thread::spawn(move || {
            worker.run();
        });
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
