// Copyright 2023 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    collections::HashSet,
    fs,
    io::Write,
    mem,
    path::PathBuf,
    str::FromStr,
    sync::{Arc, Mutex},
    time::Duration,
};

use bytes::{Buf, BufMut, Bytes, BytesMut};
use http::{Request, Uri};
use hyper::Body;
use kvengine::{
    dfs,
    dfs::Options,
    stats::ShardStats,
    table::{sstable::Builder, Value},
};
use kvproto::metapb;
use pd_client::PdClient;
use protobuf::Message;
use rfstore::store::{raw_end_key, raw_start_key};
use serde_derive::{Deserialize, Serialize};
use tikv_util::{
    codec::bytes::{decode_bytes, encode_bytes},
    error, info,
    mpsc::{Receiver, Sender},
    time::Instant,
    warn,
};

use crate::{
    error::{Error, Result},
    kv::{KvPair, KvPairsReader, MergeIterator, SstMeta},
};

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

pub struct LoadTaskWorker {
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

pub enum LoadTaskMsg {
    AddChunk {
        chunk_id: u64,
        chunk_data: Bytes,
    },
    Build {
        chunk_ids: Vec<u64>,
        compression_type: u8,
    },
    Cleanup,
    QueryUnhandledChunks {
        chunk_ids: Vec<u64>,
        cb: Box<dyn FnOnce(Vec<u64>) + Send>,
    },
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

#[derive(Clone)]
pub struct LoadDataContext {
    pub dir: PathBuf,
    pub dfs: Arc<dyn dfs::Dfs>,
    pub pd: Arc<dyn PdClient>,
    pub runtime: Arc<tokio::runtime::Runtime>,
    pub max_in_mem_size: usize,
}

#[derive(Clone)]
pub struct TaskContext {
    pub start_ts: u64,
    pub commit_ts: u64,
    pub inner_key_off: Option<usize>,
    pub key_prefix: Vec<u8>,
}

#[derive(Clone)]
pub struct LoadTaskScheduler {
    pub sender: Sender<LoadTaskMsg>,
    pub states: Arc<Mutex<LoadTaskStates>>,
    pub thread_handle: Option<Arc<Mutex<std::thread::JoinHandle<()>>>>,
}

impl LoadTaskScheduler {
    pub fn cancel(&self, err: String) {
        warn!("canceled {}", err);
        let mut states = self.states.lock().unwrap();
        states.canceled = true;
        states.error = err;
    }

    pub fn is_canceled(&self) -> bool {
        let states = self.states.lock().unwrap();
        states.canceled
    }

    pub fn error_msg(&self) -> String {
        let states = self.states.lock().unwrap();
        states.error.clone()
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

    pub fn is_finished(&self) -> bool {
        let states = self.states.lock().unwrap();
        states.finished
    }

    pub fn set_thread_handle(&mut self, thread_handle: std::thread::JoinHandle<()>) {
        self.thread_handle = Some(Arc::new(Mutex::new(thread_handle)))
    }

    pub async fn query_unhandled_chunks(&self, chunk_ids: Vec<u64>) -> Vec<u64> {
        let (cb, fut) = tikv_util::future::paired_future_callback();
        self.sender
            .send(LoadTaskMsg::QueryUnhandledChunks { chunk_ids, cb })
            .unwrap();
        fut.await.unwrap()
    }
}

impl LoadTaskWorker {
    pub fn new(context: LoadDataContext, task_ctx: TaskContext) -> Self {
        let (sender, receiver) = tikv_util::mpsc::unbounded();
        let mut states = LoadTaskStates::default();
        states.start_ts = task_ctx.start_ts;
        let scheduler = LoadTaskScheduler {
            sender,
            states: Arc::new(Mutex::new(states)),
            thread_handle: None,
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

    pub fn run(&mut self) {
        self.init_task_dir();
        while let Ok(msg) = self.receiver.recv() {
            match msg {
                LoadTaskMsg::AddChunk {
                    chunk_id,
                    chunk_data,
                } => {
                    let res = self.handle_add_chunk(chunk_id, chunk_data);
                    if res.is_err() || !self.reader_errs.is_empty() {
                        let err_msg = if let Err(e) = res {
                            e.to_string()
                        } else {
                            self.reader_errs.first().unwrap().to_string()
                        };
                        self.scheduler.cancel(err_msg);
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
                LoadTaskMsg::QueryUnhandledChunks { mut chunk_ids, cb } => {
                    chunk_ids.drain_filter(|x| self.handled_chunks.contains(x));
                    cb(chunk_ids);
                }
            }
        }
    }

    pub fn get_scheduler(&self) -> LoadTaskScheduler {
        self.scheduler.clone()
    }

    pub(crate) fn handle_add_chunk(&mut self, chunk_id: u64, chunk_data: Bytes) -> Result<()> {
        info!("handle add chunk {}, len {}", chunk_id, chunk_data.len());
        if self.handled_chunks.contains(&chunk_id) {
            warn!(
                "{} skip duplicated chunk {}",
                self.task_ctx.start_ts, chunk_id
            );
            return Ok(());
        }
        if self.scheduler.is_canceled() {
            warn!(
                "task {} is canceled, skip add chunk {}",
                self.task_ctx.start_ts, chunk_id
            );
            return Ok(());
        }
        if self.scheduler.is_finished() {
            warn!(
                "task {} is finished, skip add chunk {}",
                self.task_ctx.start_ts, chunk_id
            );
            return Ok(());
        }
        self.handled_chunks.insert(chunk_id);

        if self.task_ctx.inner_key_off.is_none() {
            let key_len = (&chunk_data[0..]).get_u16_le();
            let first_key = chunk_data.slice(2..2 + key_len as usize);
            let region = self.ctx.pd.get_region(first_key.chunk())?;
            let region_id = region.get_id();
            let shard_stats = self
                .ctx
                .runtime
                .block_on(get_shard_stats(&self.ctx.pd, region_id))?;

            self.task_ctx.inner_key_off = Some(shard_stats.inner_key_off);
            self.task_ctx.key_prefix = first_key.slice(..shard_stats.inner_key_off).to_vec();
        }
        let inner_key_off = self.task_ctx.inner_key_off.unwrap();
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

            let key_prefix = key.slice(..inner_key_off);
            if key_prefix.chunk() != self.task_ctx.key_prefix.as_slice() {
                let err_msg = format!(
                    "{} chunk data key prefix inconsistent, first chunk: {:?}, chunk: {:?}",
                    self.task_ctx.start_ts,
                    self.task_ctx.key_prefix,
                    key_prefix.chunk()
                );
                error!("{}", err_msg);
                return Err(Error::CheckError(err_msg));
            }

            let inner_key = key.slice(inner_key_off..);
            self.kv_pairs.push(KvPair::new(inner_key, val));
            self.in_mem_size += 2 + key_len as usize + 4 + val_len as usize;
        }
        if self.in_mem_size > self.ctx.max_in_mem_size {
            info!(
                "{} flush to local file on in_mem_size {}",
                self.task_ctx.start_ts, self.in_mem_size
            );
            let kv_pairs = mem::take(&mut self.kv_pairs);
            let tx = self.file_tx.clone();
            let task_ctx = self.task_ctx.clone();
            let file_path = self.file_path(self.file_idx);
            let in_mem_size = self.in_mem_size;
            self.file_idx += 1;
            std::thread::spawn(move || {
                let res = flush_to_local_file(kv_pairs, task_ctx, file_path, in_mem_size);
                tx.send(res).unwrap();
            });
            if self.file_idx > self.readers.len() + self.reader_errs.len() + FLUSH_FILE_CONCURRENCY
            {
                self.recv_reader();
            }
            self.in_mem_size = 0;
        }
        Ok(())
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

    fn alloc_file_id(&mut self) -> Result<u64> {
        if let Some(id) = self.cached_file_ids.pop() {
            return Ok(id);
        }
        let start = Instant::now();
        let count = 64;
        loop {
            match futures::executor::block_on(self.ctx.pd.batch_get_tso(count as u32)) {
                Ok(ts) => {
                    let last = ts.into_inner();
                    let first = last - count as u64 + 1;
                    self.cached_file_ids = (first..=last).rev().collect();
                    return Ok(self.cached_file_ids.pop().unwrap());
                }
                Err(err) => {
                    error!("failed to allocate file id from PD {:?}", err);
                    std::thread::sleep(Duration::from_secs(1));
                    if start.saturating_elapsed() > ALLOCATE_ID_TIMEOUT {
                        return Err(Error::PdError(err));
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
                self.task_ctx.clone(),
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
            if let Err(err) = file_id {
                errs.push(err);
                break;
            }
            self.spawn_build_file(file_id.unwrap(), batch, tx.clone(), compression_type);
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
        info!(
            "{} start split, keys {:?}",
            self.task_ctx.start_ts, split_keys
        );
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
        let key_prefix = self.task_ctx.key_prefix.to_vec();
        let inner_key_off = self.task_ctx.inner_key_off.unwrap();
        let coarse_split_keys = gen_split_keys(&key_prefix, &sst_metas, COARSE_SPLIT_SIZE, true);
        let new_regions_id = self.split_regions(&coarse_split_keys)?;
        for i in 0..coarse_split_keys.len() - 1 {
            let mut encoded_start_key = coarse_split_keys[i].as_slice();
            let raw_start_key = decode_bytes(&mut encoded_start_key, false).unwrap();
            let mut encoded_end_key = coarse_split_keys[i + 1].as_slice();
            let raw_end_key = decode_bytes(&mut encoded_end_key, false).unwrap();
            let inner_start_key = &raw_start_key[inner_key_off..];
            let inner_end_key = &raw_end_key[inner_key_off..];
            let group_ssts = get_ssts_in_range(&sst_metas, inner_start_key, inner_end_key);
            assert!(
                !group_ssts.is_empty(),
                "raw start {:?}, raw end {:?}, inner start {:?}, inner end {:?}, ssts {:?}",
                raw_start_key,
                raw_end_key,
                inner_start_key,
                inner_end_key,
                sst_metas
            );
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
        let key_prefix = self.task_ctx.key_prefix.to_vec();
        let split_keys = gen_split_keys(&key_prefix, &sst_metas, REGION_SIZE, false);
        if !split_keys.is_empty() {
            self.split_regions(&split_keys)?;
        }
        let outer_first_key =
            new_region_key(&key_prefix, sst_metas.first().unwrap().smallest.as_slice());
        let outer_last_key = {
            let mut last_key = sst_metas.last().unwrap().biggest.clone();
            last_key.push(0);
            new_region_key(&key_prefix, &last_key)
        };
        let regions = self.ctx.runtime.block_on(self.ctx.pd.scan_regions(
            outer_first_key,
            outer_last_key,
            usize::MAX,
        ))?;
        info!("scanned regions {:?}", regions);
        // TODO: verify regions. In case the regions is not intact, we will silently
        // miss to ingest some chunks.
        let (tx, rx) = tikv_util::mpsc::unbounded();
        let mut msg_cnt = 0;
        for mut pd_region in regions {
            let region = pd_region.get_region();
            let cs =
                build_ingest_files(key_prefix.len(), region, &sst_metas, self.task_ctx.start_ts);
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

async fn get_shard_stats(pd: &Arc<dyn PdClient>, shard_id: u64) -> Result<ShardStats> {
    let http_client = hyper::client::Client::new();
    let mut retry = 0;
    loop {
        if retry > MAX_RETRY_TIMES {
            return Err(Error::Other(format!(
                "get_shard_stats failed, shard_id: {}",
                shard_id
            )));
        }
        if retry > 0 {
            tokio::time::sleep(Duration::from_secs(6)).await;
        }
        retry += 1;

        let store_id_res = get_leader_store(pd, shard_id).await;
        if store_id_res.is_err() {
            error!("get_shard_stats error: {:?}", store_id_res.unwrap_err());
            continue;
        }
        let store_id = store_id_res.unwrap();
        let store_res = pd.get_store_async(store_id).await;
        if store_res.is_err() {
            error!("get_shard_stats error: {:?}", store_res.unwrap_err());
            continue;
        }
        let store = store_res.unwrap();
        let uri = Uri::from_str(&format!(
            "http://{}/kvengine/{}",
            &store.status_address, shard_id
        ))
        .unwrap();
        let req = Request::get(uri).body(Body::from(""))?;
        match http_client.request(req).await {
            Ok(resp) => {
                if resp.status().is_success() {
                    let body = hyper::body::to_bytes(resp.into_body()).await?;
                    let shard_stats: ShardStats = serde_json::from_slice(&body).unwrap();
                    if shard_stats.id == 0 {
                        continue;
                    }
                    return Ok(shard_stats);
                } else {
                    continue;
                }
            }
            Err(e) => {
                error!(
                    "get_shard_stats failed, shard_id: {}, error: {:?}",
                    shard_id, e
                );
                continue;
            }
        }
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
    inner_key_off: usize,
    region: &metapb::Region,
    sst_metas: &[SstMeta],
    start_ts: u64,
) -> kvenginepb::ChangeSet {
    let raw_start_key = raw_start_key(region);
    let inner_start_key = &raw_start_key[inner_key_off..];
    let raw_end_key = raw_end_key(region);
    let inner_end_key = &raw_end_key[inner_key_off..];
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
        if sst_meta.biggest.as_slice() < inner_start_key {
            continue;
        }
        if !inner_end_key.is_empty() && sst_meta.smallest.as_slice() >= inner_end_key {
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

fn gen_split_keys(
    key_prefix: &[u8],
    ssts: &[SstMeta],
    split_size: usize,
    include_bound: bool,
) -> Vec<Vec<u8>> {
    let mut keys = vec![];
    if include_bound {
        keys.push(new_region_key(
            key_prefix,
            ssts.first().unwrap().smallest.as_slice(),
        ));
    }
    let mut size = 0;
    for sst in ssts {
        if size > split_size {
            keys.push(new_region_key(key_prefix, sst.smallest.as_slice()));
            size = 0;
        }
        size += sst.size;
    }
    if include_bound {
        // split at last key so the last region will not be split by other concurrent
        // load_data and get epoch not match error.
        let mut last_key = ssts.last().unwrap().biggest.to_vec();
        last_key.push(0);
        keys.push(new_region_key(key_prefix, last_key.as_slice()));
    }
    keys
}

fn new_region_key(key_prefix: &[u8], raw_key: &[u8]) -> Vec<u8> {
    let mut key = key_prefix.to_vec();
    key.extend_from_slice(raw_key);
    encode_bytes(&key)
}

fn get_ssts_in_range(ssts: &[SstMeta], start: &[u8], end: &[u8]) -> Vec<SstMeta> {
    let position = ssts
        .binary_search_by(|sst| sst.smallest.as_slice().cmp(start))
        .unwrap();
    let mut matched = vec![];
    for i in position..ssts.len() {
        let sst = &ssts[i];
        if !end.is_empty() && sst.smallest.as_slice() >= end {
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
