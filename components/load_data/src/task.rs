// Copyright 2023 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    collections::HashMap,
    fs,
    io::Write,
    mem,
    ops::Deref,
    path::PathBuf,
    sync::{Arc, Mutex, MutexGuard},
    time::Duration,
};

use api_version::api_v2::KEYSPACE_PREFIX_LEN;
use bytes::{Buf, BufMut, Bytes};
use cloud_encryption::{EncryptionKey, MasterKey};
use encryption::{DecrypterReader, EncrypterWriter, Iv};
use http::Request;
use hyper::Body;
use kvengine::{
    dfs,
    dfs::Options,
    get_shard_property,
    table::{sstable::Builder, InnerKey, Value},
    IdVer, ShardTag, ENCRYPTION_KEY, WRITE_CF, WRITE_CF_BOTTOM_LEVEL,
};
use kvproto::{encryptionpb::EncryptionMethod, metapb, pdpb};
use pd_client::PdClient;
use protobuf::Message;
use rfstore::store::{raw_end_key, raw_start_key};
use serde_derive::{Deserialize, Serialize};
use tidb_query_datatype::codec::table;
use tikv_util::{
    box_err,
    codec::bytes::{decode_bytes, encode_bytes},
    debug, error, info,
    merge_range::MergeRanges,
    mpsc::{Receiver, Sender},
    time::Instant,
    warn,
};

use crate::{
    check_point_storage::{
        LoadDataCheckPointCtx, LoadDataWorkerState, LocalFileCheckPointStorage, LocalFileInfo,
    },
    error::{Error, Result},
    kv::{DuplicateEntry, KvPair, KvPairsReader, MergeIterator, SstMeta},
};

const DEFAULT_BLOCK_SIZE: usize = 64 * 1024; // 64KB
const DEFAULT_SST_FILE_SIZE: usize = 48 * 1024 * 1024; // 48MB
const DEFAULT_REGION_SIZE: usize = 750 * 1024 * 1024; // 750MB
const DEFAULT_COARSE_SPLIT_SIZE: usize = 32 * 1024 * 1024 * 1024; // 32GB
const DEFAULT_ENABLE_CHECK_POINT: bool = false;

const ZSTD_COMPRESSION_LEVEL: i32 = 3;
const FLUSH_FILE_CONCURRENCY: usize = 4;
const CREATE_FILE_CONCURRENCY: usize = 32;
const INGEST_CONCURRENCY: usize = 4;

const ALLOCATE_ID_TIMEOUT: Duration = Duration::from_secs(10 * 60);
const RETRY_SLEEP_DURATION: Duration = Duration::from_millis(100);
const MAX_RETRY_TIMES: usize = 10;
const MAX_SLEEP_DURATION: Duration = Duration::from_secs(30);
const GET_SHARD_META_TIMEOUT: Duration = Duration::from_secs(60);

pub struct LoadTaskWorker {
    config: LoadDataConfig,
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
    unhandled_readers: Vec<UnhandledReader>,
    cached_file_ids: Vec<u64>,
    file_tx: Sender<UnhandledReader>,
    file_rx: Receiver<UnhandledReader>,
    check_point_store: Arc<Mutex<LocalFileCheckPointStorage>>,
}

pub struct UnhandledReader {
    pub reader: Result<KvPairsReader>,
    pub handled_chunk_ids: HashMap<u64 /* writer_id */, u64 /* chunk_id */>,
    pub file_idx: usize, // used to keep readers in order as kv pairs are flush asynchronously.
    pub path: PathBuf,   // It is recorded in checkpoint, and used to recover the reader.
    pub kv_count: usize, // It is recorded in checkpoint, and used to recover the reader.
}

pub enum LoadTaskMsg {
    AddChunk {
        writer_id: u64,
        chunk_id: u64,
        chunk_data: Bytes,
        cb: Box<dyn FnOnce(PutChunkResult) + Send>,
    },
    Build {
        compression_type: u8,
    },
    Flush {
        cb: Box<dyn FnOnce(FlushResult) + Send>,
    },
    Cleanup,
    QueryUnhandledChunks {
        chunk_ids: HashMap<u64 /* writer_id */, u64 /* chunk_id */>,
        cb: Box<dyn FnOnce(HashMap<u64, u64>) + Send>,
    },
}

#[derive(Default, Debug, Serialize, Deserialize, Clone)]
#[serde(default)]
#[serde(rename_all = "kebab-case")]
pub struct WritersStates {
    pub handled_chunk_ids: HashMap<u64 /* writer_id */, u64 /* chunk_id */>,
    pub flushed_chunk_ids: HashMap<u64 /* writer_id */, u64 /* chunk_id */>,
}

#[derive(Default, Debug, Serialize, Deserialize, Clone)]
#[serde(default)]
#[serde(rename_all = "kebab-case")]
pub struct PutChunkResult {
    pub handled_chunk_id: u64,
    pub flushed_chunk_id: u64,
    pub canceled: bool,
    pub finished: bool,
    pub error: String,
}

#[derive(Default, Debug, Serialize, Deserialize, Clone)]
#[serde(default)]
#[serde(rename_all = "kebab-case")]
pub struct FlushResult {
    pub flushed_chunk_ids: HashMap<u64 /* writer_id */, u64 /* chunk_id */>,
    pub canceled: bool,
    pub finished: bool,
    pub error: String,
}

#[derive(Default, Debug, Serialize, Deserialize, Clone)]
#[serde(default)]
#[serde(rename_all = "kebab-case")]
pub struct LoadTaskStates {
    pub task_id: String,
    pub canceled: bool,
    pub finished: bool,
    pub error: String,
    pub flushed_files: usize,
    pub created_files: usize,
    pub ingested_regions: usize,
    pub duplicated_entries: Vec<DuplicateEntry>,
}

#[derive(Clone, Serialize, Deserialize, PartialEq, Debug)]
pub struct LoadDataConfig {
    pub block_size: usize,
    pub sst_file_size: usize,
    pub region_size: usize,
    pub coarse_split_size: usize,
    pub enable_check_point: bool,
}

impl Default for LoadDataConfig {
    fn default() -> Self {
        Self {
            block_size: DEFAULT_BLOCK_SIZE,
            sst_file_size: DEFAULT_SST_FILE_SIZE,
            region_size: DEFAULT_REGION_SIZE,
            coarse_split_size: DEFAULT_COARSE_SPLIT_SIZE,
            enable_check_point: DEFAULT_ENABLE_CHECK_POINT,
        }
    }
}

#[derive(Clone)]
pub struct LoadDataContext {
    pub dir: PathBuf,
    pub dfs: Arc<dyn dfs::Dfs>,
    pub pd: Arc<dyn PdClient>,
    pub runtime: Arc<tokio::runtime::Runtime>,
    pub max_in_mem_size: usize,
    pub master_key: MasterKey,
}

#[derive(Clone, Default)]
pub struct TaskContext {
    pub task_id: String,
    pub start_ts: u64,
    pub commit_ts: u64,
    pub inner_key_off: Option<usize>,
    pub key_prefix: Vec<u8>,
    pub encryption_key: Option<EncryptionKey>,
}

#[derive(Clone)]
pub struct LoadTaskScheduler {
    pub sender: Sender<LoadTaskMsg>,
    pub states: Arc<Mutex<LoadTaskStates>>,
    pub writers: Arc<Mutex<WritersStates>>,
    pub thread_handle: Option<Arc<Mutex<std::thread::JoinHandle<()>>>>,
    pub check_point_store: Arc<Mutex<LocalFileCheckPointStorage>>,
}

impl LoadTaskScheduler {
    pub fn cancel(&self, err: String) {
        warn!("canceled {}", err);
        let mut states = self.states.lock().unwrap();
        self.check_point_store
            .lock()
            .unwrap()
            .clean_check_point_data();
        states.canceled = true;
        states.error = err;
    }

    pub fn check_task_thread_finished(&self) {
        let thread_finished = self
            .thread_handle
            .as_ref()
            .unwrap()
            .lock()
            .unwrap()
            .is_finished();
        if thread_finished {
            self.cancel("task thread finished unexpectedly".to_string());
        }
    }

    pub fn is_canceled(&self) -> bool {
        let states = self.states.lock().unwrap();
        states.canceled
    }

    pub fn error_msg(&self) -> String {
        let states = self.states.lock().unwrap();
        states.error.clone()
    }

    pub fn states(&self) -> LoadTaskStates {
        let states = self.states.lock().unwrap();
        states.clone()
    }

    pub(crate) fn add_created_files_count(&self) {
        let mut states = self.states.lock().unwrap();
        states.created_files += 1;
    }

    pub(crate) fn add_ingested_regions(&self) {
        let mut states = self.states.lock().unwrap();
        states.ingested_regions += 1;
    }

    pub(crate) fn set_finished(&self, dup_entries: Vec<DuplicateEntry>) {
        let mut states = self.states.lock().unwrap();
        states.finished = true;
        states.duplicated_entries = dup_entries;
    }

    pub fn is_finished(&self) -> bool {
        let states = self.states.lock().unwrap();
        states.finished
    }

    pub(crate) fn set_flushed_files(&self, flushed_files: usize) {
        let mut states = self.states.lock().unwrap();
        states.flushed_files = flushed_files;
    }

    pub(crate) fn get_handled_chunks(&self) -> HashMap<u64, u64> {
        let writers = self.writers.lock().unwrap();
        writers.handled_chunk_ids.clone()
    }

    pub(crate) fn get_flushed_chunks(&self) -> HashMap<u64, u64> {
        let writers = self.writers.lock().unwrap();
        writers.flushed_chunk_ids.clone()
    }

    pub fn get_handled_chunk(&self, writer_id: &u64) -> u64 {
        let writers = self.writers.lock().unwrap();
        *writers.handled_chunk_ids.get(writer_id).unwrap_or(&0)
    }

    pub fn get_flushed_chunk(&self, writer_id: &u64) -> u64 {
        let writers = self.writers.lock().unwrap();
        *writers.flushed_chunk_ids.get(writer_id).unwrap_or(&0)
    }

    pub(crate) fn update_handled_chunk(&self, writer_id: u64, chunk_id: u64) {
        let mut writers = self.writers.lock().unwrap();
        let handled_chunk_id = writers.handled_chunk_ids.entry(writer_id).or_insert(0);
        assert!(*handled_chunk_id == chunk_id - 1);
        *handled_chunk_id = chunk_id;
    }

    pub(crate) fn update_flushed_chunk(&self, writer_id: u64, chunk_id: u64) {
        let mut writers = self.writers.lock().unwrap();
        let flushed_chunk_id = writers.flushed_chunk_ids.entry(writer_id).or_insert(0);
        assert!(*flushed_chunk_id <= chunk_id);
        *flushed_chunk_id = chunk_id;
        let states = self.states.lock().unwrap();
        debug!(
            "{} update flushed chunk by writer_id:{},chunk_id:{}",
            states.task_id, writer_id, chunk_id
        )
    }

    pub fn set_thread_handle(&mut self, thread_handle: std::thread::JoinHandle<()>) {
        self.thread_handle = Some(Arc::new(Mutex::new(thread_handle)))
    }

    pub async fn query_unhandled_chunks(&self, chunk_ids: HashMap<u64, u64>) -> HashMap<u64, u64> {
        let (cb, fut) = tikv_util::future::paired_future_callback();
        self.sender
            .send(LoadTaskMsg::QueryUnhandledChunks { chunk_ids, cb })
            .unwrap();
        fut.await.unwrap()
    }
}

impl LoadTaskWorker {
    pub fn new(
        config: LoadDataConfig,
        context: LoadDataContext,
        task_ctx: TaskContext,
        check_point_ctx: LoadDataCheckPointCtx,
    ) -> Self {
        let (sender, receiver) = tikv_util::mpsc::unbounded();
        let mut states = LoadTaskStates::default();
        let mut writers = WritersStates::default();
        let mut file_idx = 0;
        if check_point_ctx.get_is_recover()
            && check_point_ctx.get_state() > LoadDataWorkerState::InitTask
        {
            writers.flushed_chunk_ids = check_point_ctx.get_flushed_chunk_ids();
            writers.handled_chunk_ids = check_point_ctx.get_flushed_chunk_ids();
            file_idx = check_point_ctx.get_flushed_file_idx() + 1;
            info!(
                "{} [check point] recover: writers.flushed_chunk_ids:{:?},writers.handled_chunk_ids:{:?},file_idx:{}",
                task_ctx.task_id, writers.flushed_chunk_ids, writers.handled_chunk_ids, file_idx
            );
        }

        // Init checkpoint info.
        let mut check_point_store =
            LocalFileCheckPointStorage::new(check_point_ctx.clone()).unwrap();
        if !check_point_ctx.get_is_recover() {
            check_point_store.flush_check_point_ctx().unwrap();
            check_point_store.print_log();
        }
        let check_point_store_arc = Arc::new(Mutex::new(check_point_store));

        states.task_id = task_ctx.task_id.clone();
        let scheduler = LoadTaskScheduler {
            sender,
            states: Arc::new(Mutex::new(states)),
            writers: Arc::new(Mutex::new(writers)),
            thread_handle: None,
            check_point_store: Arc::clone(&check_point_store_arc),
        };
        let (file_tx, file_rx) = tikv_util::mpsc::unbounded();
        let task_dir = context.dir.join(task_ctx.task_id.as_str());

        Self {
            config,
            ctx: context,
            task_ctx,
            task_dir,
            kv_pairs: vec![],
            in_mem_size: 0,
            file_idx,
            readers: vec![],
            reader_errs: vec![],
            scheduler,
            receiver,
            unhandled_readers: vec![],
            cached_file_ids: vec![],
            file_tx,
            file_rx,
            check_point_store: Arc::clone(&check_point_store_arc),
        }
    }

    pub fn run(&mut self) {
        self.init_task_dir();
        while let Ok(msg) = self.receiver.recv() {
            match msg {
                LoadTaskMsg::AddChunk {
                    writer_id,
                    chunk_id,
                    chunk_data,
                    cb,
                } => {
                    let res = self.handle_add_chunk(writer_id, chunk_id, chunk_data, cb);
                    if res.is_err() || !self.reader_errs.is_empty() {
                        let err_msg = if let Err(e) = res {
                            e.to_string()
                        } else {
                            self.reader_errs.first().unwrap().to_string()
                        };
                        self.scheduler.cancel(err_msg);
                        if self.file_idx > self.reader_errs.len() + self.readers.len() {
                            let recv_count = self.file_idx
                                - self.reader_errs.len()
                                - self.readers.len()
                                - self.unhandled_readers.len();
                            let check_point_store_mutex = Arc::clone(&self.check_point_store);
                            let mut check_point_store_guard =
                                check_point_store_mutex.lock().unwrap();
                            let recv_res =
                                self.recv_reader(recv_count, &mut check_point_store_guard);
                            if recv_res.is_err() || !self.reader_errs.is_empty() {
                                let err_msg = if let Err(recv_err) = recv_res {
                                    recv_err.to_string()
                                } else {
                                    self.reader_errs.first().unwrap().to_string()
                                };
                                self.scheduler.cancel(err_msg);
                            }
                        }
                    }
                }
                LoadTaskMsg::Build { compression_type } => {
                    if self.scheduler.is_canceled() {
                        warn!("task {} is canceled, do not build", self.task_ctx.task_id);
                        continue;
                    }
                    if self.scheduler.is_finished() {
                        warn!("task {} is finished, skip build", self.task_ctx.task_id);
                        continue;
                    }
                    if let Err(err) = self.build(compression_type) {
                        error!("build failed {:?}", err);
                        self.scheduler.cancel(err.to_string());
                    }
                    continue;
                }
                LoadTaskMsg::Flush { cb } => {
                    let mut flush_res = FlushResult::default();
                    if self.scheduler.is_canceled() {
                        warn!("task {} is canceled, do not build", self.task_ctx.task_id);
                        flush_res.canceled = true;
                        flush_res.error = self.scheduler.error_msg();
                        cb(flush_res);
                        continue;
                    }
                    if self.scheduler.is_finished() {
                        warn!("task {} is finished, skip flush", self.task_ctx.task_id);
                        flush_res.finished = true;
                        flush_res.error = self.scheduler.error_msg();
                        flush_res.flushed_chunk_ids = self.scheduler.get_flushed_chunks();
                        cb(flush_res);
                        continue;
                    }
                    let check_point_store_mutex = Arc::clone(&self.check_point_store);
                    let mut check_point_store_guard = check_point_store_mutex.lock().unwrap();
                    if let Err(err) = self.flush(&mut check_point_store_guard) {
                        error!("flush failed {:?}", err);
                        self.scheduler.cancel(err.to_string());
                    }
                    flush_res.error = self.scheduler.error_msg();
                    flush_res.flushed_chunk_ids = self.scheduler.get_flushed_chunks();
                    cb(flush_res);
                    continue;
                }
                LoadTaskMsg::Cleanup => {
                    self.remove_local_files();
                    return;
                }
                LoadTaskMsg::QueryUnhandledChunks { mut chunk_ids, cb } => {
                    let mut res: HashMap<u64, u64> = HashMap::new();
                    for (writer_id, chunk_id) in chunk_ids.drain() {
                        if self.scheduler.get_flushed_chunk(&writer_id) < chunk_id {
                            res.insert(writer_id, chunk_id);
                        }
                    }
                    cb(res);
                }
            }
        }
    }

    pub fn get_scheduler(&self) -> LoadTaskScheduler {
        self.scheduler.clone()
    }

    pub fn set_inner_key_off_and_encrytion_key(&mut self, first_key: Bytes) -> Result<()> {
        if self.task_ctx.inner_key_off.is_none() {
            let shard_meta = self.ctx.runtime.block_on(get_shard_meta(
                self.ctx.pd.clone(),
                first_key.chunk(),
                GET_SHARD_META_TIMEOUT,
            ))?;
            let snapshot = shard_meta.get_snapshot();
            self.task_ctx.inner_key_off = Some(snapshot.inner_key_off as usize);
            self.task_ctx.key_prefix = first_key.slice(..snapshot.inner_key_off as usize).to_vec();
            self.task_ctx.encryption_key =
                get_shard_property(ENCRYPTION_KEY, snapshot.get_properties()).map(|exported_key| {
                    self.ctx
                        .master_key
                        .decrypt_encryption_key(&exported_key)
                        .unwrap()
                });
        }
        Ok(())
    }

    pub(crate) fn handle_add_chunk(
        &mut self,
        writer_id: u64,
        chunk_id: u64,
        chunk_data: Bytes,
        cb: Box<dyn FnOnce(PutChunkResult) + Send>,
    ) -> Result<()> {
        info!(
            "{} handle add chunk {} with len {} for writer {}",
            self.task_ctx.task_id,
            chunk_id,
            chunk_data.len(),
            writer_id
        );
        let handled_chunk_id = self.scheduler.get_handled_chunk(&writer_id);
        let flushed_chunk_id = self.scheduler.get_flushed_chunk(&writer_id);
        let mut put_chunk_res = PutChunkResult {
            handled_chunk_id,
            flushed_chunk_id,
            canceled: self.scheduler.is_canceled(),
            finished: self.scheduler.is_finished(),
            error: self.scheduler.error_msg(),
        };
        if self.scheduler.is_canceled() {
            warn!(
                "{} is canceled, skip add chunk {}",
                self.task_ctx.task_id, chunk_id
            );
            cb(put_chunk_res);
            return Ok(());
        }
        if self.scheduler.is_finished() {
            warn!(
                "{} is finished, skip add chunk {}",
                self.task_ctx.task_id, chunk_id
            );
            cb(put_chunk_res);
            return Ok(());
        }

        if chunk_data.is_empty() {
            warn!(
                "{} skip add chunk, chunk {} is empty for writer {}",
                self.task_ctx.task_id, chunk_id, writer_id
            );
            cb(put_chunk_res);
            return Ok(());
        }
        if handled_chunk_id != chunk_id - 1 {
            warn!(
                "{} skip chunk {} for writer {}, expect chunk {}",
                self.task_ctx.task_id,
                chunk_id,
                writer_id,
                handled_chunk_id + 1
            );
            put_chunk_res.error = format!(
                "skip chunk {} for writer {}, expect chunk {}",
                chunk_id,
                writer_id,
                handled_chunk_id + 1
            );
            cb(put_chunk_res);
            return Ok(());
        }
        self.scheduler.update_handled_chunk(writer_id, chunk_id);
        put_chunk_res.handled_chunk_id = chunk_id;
        // call `cb` early to avoid waiting for flush
        cb(put_chunk_res);

        if self.task_ctx.inner_key_off.is_none() {
            let key_len = (&chunk_data[0..]).get_u16_le();
            let first_key = chunk_data.slice(2..2 + key_len as usize);
            self.set_inner_key_off_and_encrytion_key(first_key.clone())?;
            let check_point_store_mutex = Arc::clone(&self.check_point_store);
            let mut check_point_store_guard = check_point_store_mutex.lock().unwrap();
            check_point_store_guard.update_first_key(first_key)?;
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
                    self.task_ctx.task_id,
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

        let check_point_store_mutex = Arc::clone(&self.check_point_store);
        let mut check_point_store_guard = check_point_store_mutex.lock().unwrap();
        if self.in_mem_size > self.ctx.max_in_mem_size {
            self.flush_mem_buf();
            if self.file_idx
                > self.readers.len()
                    + self.reader_errs.len()
                    + self.unhandled_readers.len()
                    + FLUSH_FILE_CONCURRENCY
            {
                self.recv_reader(1, &mut check_point_store_guard)?;
            }
            self.in_mem_size = 0;
        }
        self.try_recv_reader(&mut check_point_store_guard)?;
        Ok(())
    }

    fn handle_readers(
        &mut self,
        check_point_store_guard: &mut MutexGuard<'_, LocalFileCheckPointStorage>,
    ) -> Result<()> {
        self.unhandled_readers
            .sort_by(|a, b| a.file_idx.cmp(&b.file_idx));

        let handled_file_idx = self.readers.len() + self.reader_errs.len();
        let mut need_handled = self.unhandled_readers.len();
        for (idx, unhandled_reader) in self.unhandled_readers.iter().enumerate() {
            assert!(unhandled_reader.file_idx >= handled_file_idx + idx);
            if unhandled_reader.file_idx > handled_file_idx + idx {
                need_handled = idx;
                break;
            }
        }
        debug!("received need handle {} readers", need_handled);
        if need_handled == 0 {
            return Ok(());
        }
        let mut handled_chunk_ids: HashMap<u64, u64> = HashMap::new();
        let mut max_file_idx = 0;
        let mut local_file_infos = vec![];
        for mut unhandled_reader in self.unhandled_readers.drain(0..need_handled) {
            match unhandled_reader.reader {
                Ok(reader) => {
                    self.readers.push(reader);
                    handled_chunk_ids = mem::take(&mut unhandled_reader.handled_chunk_ids);
                }
                Err(err) => {
                    let err_str = err.to_string();
                    self.reader_errs
                        .push(Error::HandleReaderError(err_str.clone()));
                    return Err(Error::HandleReaderError(err_str));
                }
            }
            if max_file_idx < unhandled_reader.file_idx {
                max_file_idx = unhandled_reader.file_idx;
            }
            local_file_infos.push(LocalFileInfo {
                path: unhandled_reader.path,
                kv_count: unhandled_reader.kv_count,
            })
        }
        for (writer_id, chunk_id) in handled_chunk_ids.clone() {
            self.scheduler.update_flushed_chunk(writer_id, chunk_id);
        }

        // Due to flush is async,
        // we need to update handled_chunk_ids and max_file_idx in handle_readers,
        // to make sure the value of handled_chunk_ids and max_file_idx are always
        // increasing.
        check_point_store_guard.update_flushed_info(
            handled_chunk_ids,
            max_file_idx,
            local_file_infos,
        )?;

        info!("{} handle {} readers", self.task_ctx.task_id, need_handled);
        Ok(())
    }

    fn try_recv_reader(
        &mut self,
        check_point_store_guard: &mut MutexGuard<'_, LocalFileCheckPointStorage>,
    ) -> Result<()> {
        while let Ok(reader) = self.file_rx.try_recv() {
            self.unhandled_readers.push(reader);
        }
        self.handle_readers(check_point_store_guard)?;
        Ok(())
    }

    fn recv_reader(
        &mut self,
        mut recv_count: usize,
        check_point_store_guard: &mut MutexGuard<'_, LocalFileCheckPointStorage>,
    ) -> Result<()> {
        while recv_count != 0 {
            let reader = self.file_rx.recv().unwrap();
            self.unhandled_readers.push(reader);
            recv_count -= 1;
        }
        self.handle_readers(check_point_store_guard)?;
        Ok(())
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

    // Recover readers from checkpoint.
    fn recover_readers(&mut self, check_point_ctx: LoadDataCheckPointCtx) {
        let paths = check_point_ctx.get_local_file_infos();
        for local_file_info in paths.iter() {
            let reader = reload_reader(local_file_info, self.task_ctx.clone());
            self.readers.push(reader);
        }
    }

    fn build(&mut self, compression_type: u8) -> Result<()> {
        info!("{} start build.", self.task_ctx.task_id);

        let check_point_store_mutex = Arc::clone(&self.check_point_store);
        let mut check_point_store_guard = check_point_store_mutex.lock().unwrap();

        // to be compatible with old remote backend, we need to flush all kv pairs
        // before build. TODO: remove it after all remote backend upgraded.
        self.flush(&mut check_point_store_guard)?;

        let mut sst_metas = vec![];

        if check_point_store_guard.get_state() < LoadDataWorkerState::BuildingSst {
            check_point_store_guard.update_build_msg(compression_type)?;
        }

        if check_point_store_guard.get_state() == LoadDataWorkerState::BuildingSst {
            // Begin to build sst.
            self.build_sst(
                &mut check_point_store_guard,
                &mut sst_metas,
                compression_type,
            )?;
            check_point_store_guard
                .flush_check_point_ctx_with_state(LoadDataWorkerState::IngestingSst)?;
        }

        // Begin ingest sst.
        if check_point_store_guard.get_state() == LoadDataWorkerState::IngestingSst {
            if check_point_store_guard.check_point_ctx.get_is_recover() {
                sst_metas = check_point_store_guard.get_sst_meta();
            }
            let res = self.ingest(
                sst_metas,
                check_point_store_guard
                    .check_point_ctx
                    .get_duplicated_entries(),
                &mut check_point_store_guard,
            );
            info!("{} ingest end.", self.task_ctx.task_id);
            return res;
        }
        Ok(())
    }

    fn flush_mem_buf(&mut self) {
        info!(
            "{} flush to local file on in_mem_size {}",
            self.task_ctx.task_id, self.in_mem_size
        );
        let kv_pairs = mem::take(&mut self.kv_pairs);
        let kv_count = kv_pairs.len();
        let tx = self.file_tx.clone();
        let task_ctx = self.task_ctx.clone();
        let file_path = self.file_path(self.file_idx);
        let in_mem_size = self.in_mem_size;
        let handled_chunk_ids = self.scheduler.get_handled_chunks();
        let file_idx = self.file_idx;
        self.file_idx += 1;
        self.scheduler.set_flushed_files(self.file_idx);
        std::thread::spawn(move || {
            let start = Instant::now();
            let task_id = task_ctx.task_id.clone();
            let res = flush_to_local_file(kv_pairs, task_ctx, file_path.clone(), in_mem_size);
            info!(
                "{} flush to local file {} takes {:?}",
                task_id,
                file_idx,
                start.saturating_elapsed()
            );
            tx.send(UnhandledReader {
                reader: res,
                handled_chunk_ids,
                file_idx,
                path: file_path.clone(),
                kv_count,
            })
            .unwrap();
        });
    }

    fn flush(
        &mut self,
        check_point_store_guard: &mut MutexGuard<'_, LocalFileCheckPointStorage>,
    ) -> Result<()> {
        if !self.reader_errs.is_empty() {
            return Err(self.reader_errs.pop().unwrap());
        }

        if !self.kv_pairs.is_empty() {
            self.flush_mem_buf();
        }
        if self.readers.len() + self.reader_errs.len() < self.file_idx {
            let recv_count = self.file_idx
                - self.readers.len()
                - self.reader_errs.len()
                - self.unhandled_readers.len();
            info!(
                "{} still needs to receive {} readers, file_idx:{},self.readers.len():{},self.reader_errs.len():{},self.unhandled_readers.len():{}",
                self.task_ctx.task_id,
                recv_count,
                self.file_idx,
                self.readers.len(),
                self.reader_errs.len(),
                self.unhandled_readers.len()
            );
            self.recv_reader(recv_count, check_point_store_guard)?;
        }
        if !self.reader_errs.is_empty() {
            return Err(self.reader_errs.pop().unwrap());
        }

        self.in_mem_size = 0;
        Ok(())
    }

    fn build_sst(
        &mut self,
        check_point_store_guard: &mut MutexGuard<'_, LocalFileCheckPointStorage>,
        sst_metas: &mut Vec<SstMeta>,
        compression_type: u8,
    ) -> Result<()> {
        if !self.kv_pairs.is_empty() || self.readers.len() < self.file_idx {
            debug!(
                "{} has {} not yet flushed kv pairs, {} unhandled files",
                self.task_ctx.task_id,
                self.kv_pairs.len(),
                self.file_idx - self.readers.len()
            );
            return Err(Error::CheckError(format!(
                "{} has {} not yet flushed kv pairs, {} unhandled files",
                self.task_ctx.task_id,
                self.kv_pairs.len(),
                self.file_idx - self.readers.len()
            )));
        }

        if self.readers.is_empty() {
            info!("{} build empty data", self.task_ctx.task_id);
            self.scheduler.set_finished(vec![]);
            return Ok(());
        }

        info!("{} start build sst.", self.task_ctx.task_id);
        let (tx, rx) = tikv_util::mpsc::unbounded();
        let mut sent_count = 0;
        let mut recv_count = 0;
        let readers = mem::take(&mut self.readers);
        let mut merge_iter = MergeIterator::new(readers, &self.task_ctx.key_prefix);

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
                    error!("{} create file failed {}", self.task_ctx.task_id, err);
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
        if !merge_iter.duplicated_entries.is_empty() {
            info!(
                "got {} duplicated entries, size {}",
                merge_iter.duplicated_entries.len(),
                merge_iter.duplicated_entries_size
            );
        }
        info!("{} finish build", self.task_ctx.task_id);
        check_point_store_guard
            .update_build_result(sst_metas.clone(), merge_iter.duplicated_entries)?;
        Ok(())
    }

    fn spawn_build_file(
        &self,
        file_id: u64,
        mut batch: Vec<u8>,
        sender: Sender<Result<SstMeta>>,
        compression_type: u8,
    ) {
        let ctx = self.ctx.clone();
        let block_size = self.config.block_size;
        let task_id = self.task_ctx.task_id.clone();
        let encryption_key = self.task_ctx.encryption_key.clone();

        self.ctx.runtime.spawn(async move {
            info!("{} start build sst file {}", task_id, file_id);
            let mut builder = Builder::new(
                file_id,
                block_size,
                compression_type,
                ZSTD_COMPRESSION_LEVEL,
                encryption_key,
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
                // The key is already trimmed prefix, so we can use `from_inner_buf` here.
                let inner_key = InnerKey::from_inner_buf(key);
                builder.add(inner_key, &Value::decode(val), None);
                entries += 1;
            }
            batch.clear();
            builder.finish(0, &mut batch);
            let data: Bytes = batch.into();
            let sst_meta = SstMeta {
                id: file_id,
                smallest: builder.get_smallest().to_vec(),
                biggest: builder.get_biggest().to_vec(),
                size: data.len(),
                keys: entries,
            };
            info!("{} finish build sst file {:?}", task_id, sst_meta);
            let opts = Options::new(0, 0);
            let res = ctx
                .dfs
                .create(file_id, data, opts)
                .await
                .map(|_| sst_meta.clone())
                .map_err(|e| Error::from(e));
            sender.send(res).unwrap();
            debug!(
                "{} finish dfs create sst file {:?}",
                task_id,
                sst_meta.clone()
            );
        });
    }

    fn read_batch(&mut self, merge_iter: &mut MergeIterator) -> Result<Vec<u8>> {
        let mut buf = Vec::with_capacity(self.config.sst_file_size);
        let mut pre_table_id = 0;
        while merge_iter.valid() {
            let key = merge_iter.key();
            let key_len = key.len();
            let val = merge_iter.value();
            let val_len = val.len();
            if buf.len() + 2 + key_len + 4 + val_len > buf.capacity() {
                return Ok(buf);
            }
            let table_id = if key.starts_with(table::TABLE_PREFIX) {
                table::decode_table_id(key).unwrap()
            } else {
                table::decode_table_id(&key[KEYSPACE_PREFIX_LEN..]).unwrap()
            };
            if pre_table_id != 0 && pre_table_id != table_id {
                return Ok(buf);
            }
            pre_table_id = table_id;
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
            self.task_ctx.task_id, split_keys
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
                    error!("{} split failed {:?}", self.task_ctx.task_id, e);
                    if retry >= MAX_RETRY_TIMES {
                        return Err(Error::PdError(e));
                    }
                    std::thread::sleep(std::cmp::min(
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
            self.task_ctx.task_id, new_regions_id
        );
        Ok(new_regions_id)
    }

    fn ingest(
        &mut self,
        mut sst_metas: Vec<SstMeta>,
        dup_entries: Vec<DuplicateEntry>,
        check_point_store_guard: &mut MutexGuard<'_, LocalFileCheckPointStorage>,
    ) -> Result<()> {
        if sst_metas.is_empty() {
            return Ok(());
        }
        info!("{} start ingest", self.task_ctx.task_id);
        sst_metas.sort_by(|a, b| a.id.cmp(&b.id));
        let key_prefix = self.task_ctx.key_prefix.to_vec();
        let inner_key_off = self.task_ctx.inner_key_off.unwrap();
        let coarse_split_keys =
            gen_split_keys(&key_prefix, &sst_metas, self.config.coarse_split_size, true);
        let new_regions_id = self.split_regions(&coarse_split_keys)?;
        let result = self.ctx.pd.scatter_regions_by_id(new_regions_id);
        if let Err(err) = result {
            error!("{} scatter regions failed {:?}", self.task_ctx.task_id, err);
        }
        for i in 0..coarse_split_keys.len() - 1 {
            let mut encoded_start_key = coarse_split_keys[i].as_slice();
            let raw_start_key = decode_bytes(&mut encoded_start_key, false).unwrap();
            let mut encoded_end_key = coarse_split_keys[i + 1].as_slice();
            let raw_end_key = decode_bytes(&mut encoded_end_key, false).unwrap();
            let inner_start_key = InnerKey::from_outer_key(&raw_start_key, inner_key_off);
            let inner_end_key = InnerKey::from_outer_end_key(&raw_end_key, inner_key_off);
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
        self.scheduler.set_finished(dup_entries);
        info!("{} finished ingest", self.task_ctx.task_id);
        check_point_store_guard
            .flush_check_point_ctx_with_state(LoadDataWorkerState::IngestedSst)?;
        Ok(())
    }

    fn is_ingest_error_retryable(err: &Error) -> bool {
        match err {
            Error::RegionNotFound(_)
            | Error::LeaderNotFound(_)
            | Error::RegionError(..)
            | Error::PdError(_)
            | Error::HyperError(_)
            | Error::RegionsIntegrityError(_) => true,
            Error::MultiErrors(errs) => errs.iter().all(Self::is_ingest_error_retryable),
            _ => false,
        }
    }

    fn ingest_group(&self, sst_metas: Vec<SstMeta>) -> Result<()> {
        let key_prefix = self.task_ctx.key_prefix.to_vec();
        let split_keys = gen_split_keys(&key_prefix, &sst_metas, self.config.region_size, false);
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

        let mut success_ranges = MergeRanges::default(); // keys of `success_ranges` are encoded. 
        let mut last_error: Option<Error> = None;
        for retry in 0..MAX_RETRY_TIMES {
            match self.ingest_group_to_range(
                &key_prefix,
                &sst_metas,
                outer_first_key.clone(),
                outer_last_key.clone(),
                &mut success_ranges,
            ) {
                Ok(_) => {
                    debug_assert!(success_ranges.covered(&outer_first_key, &outer_last_key));
                    return Ok(());
                }
                Err(err) if Self::is_ingest_error_retryable(&err) => {
                    warn!(
                        "{} ingest_group_to_range failed {:?}, retry {}",
                        self.task_ctx.task_id, err, retry
                    );
                    last_error = Some(err);
                    std::thread::sleep(std::cmp::min(
                        MAX_SLEEP_DURATION,
                        2_u32.pow(retry as u32) * RETRY_SLEEP_DURATION,
                    ));
                }
                Err(err) => return Err(err),
            }
        }
        Err(last_error.unwrap())
    }

    fn ingest_group_to_range(
        &self,
        key_prefix: &[u8],
        sst_metas: &[SstMeta],
        outer_first_key: Vec<u8>,
        outer_last_key: Vec<u8>,
        success_ranges: &mut MergeRanges,
    ) -> Result<()> {
        let mut regions = self.ctx.runtime.block_on(self.ctx.pd.scan_regions(
            outer_first_key.clone(),
            outer_last_key.clone(),
            usize::MAX,
        ))?;
        verify_regions_boundary(&outer_first_key, &outer_last_key, &regions)?;
        debug!("scanned regions {:?}", regions);
        if !success_ranges.is_empty() {
            regions.drain_filter(|region| {
                success_ranges.covered(&region.get_region().start_key, &region.get_region().end_key)
            });
        }
        info!("scanned and filtered regions {:?}", regions);

        let mut errors = vec![];
        let mut handle_ingest_res = |res: Result<metapb::Region>| match res {
            Ok(mut region) => {
                success_ranges.insert(region.take_start_key(), region.take_end_key());
                self.scheduler.add_ingested_regions();
            }
            Err(err) => errors.push(err),
        };

        let (tx, rx) = tikv_util::mpsc::unbounded();
        let mut msg_cnt = 0;
        for mut pd_region in regions {
            let region = pd_region.get_region();
            let cs =
                build_ingest_files(key_prefix.len(), region, sst_metas, self.task_ctx.commit_ts);
            if cs.get_ingest_files().get_table_creates().is_empty() {
                continue;
            }
            if self.scheduler.is_canceled() {
                return Err(Error::Canceled);
            }
            let pd_cli = self.ctx.pd.clone();
            let tx = tx.clone();
            self.ctx.runtime.spawn(async move {
                let region = pd_region.take_region();
                let leader = pd_region.take_leader();
                info!(
                    "ingest_group_to_range: region: {:?}, leader: {:?}, cs: {:?}",
                    region, leader, cs
                );
                let res = ingest_files_to_leader(pd_cli, cs, &region, leader).await;
                let _ = tx.send(res.map(|_| region));
            });
            if msg_cnt < INGEST_CONCURRENCY {
                msg_cnt += 1;
            } else {
                handle_ingest_res(rx.recv().unwrap());
            }
        }
        for _ in 0..msg_cnt {
            handle_ingest_res(rx.recv().unwrap());
        }

        if !errors.is_empty() {
            return Err(Error::MultiErrors(errors));
        }
        Ok(())
    }

    fn file_path(&self, file_idx: usize) -> PathBuf {
        self.task_dir.join(format!("kv_pairs_{}", file_idx))
    }

    fn init_task_dir(&mut self) {
        if !self.task_dir.is_dir() {
            if let Err(err) = fs::create_dir(&self.task_dir) {
                self.scheduler.cancel(format!("{:?}", err))
            }
        } else {
            if !self.config.enable_check_point {
                return;
            }

            let check_point_store_mutex = Arc::clone(&self.check_point_store);
            let check_point_store_guard = check_point_store_mutex.lock().unwrap();
            if check_point_store_guard.check_point_ctx.get_is_recover() {
                let check_point_ctx = check_point_store_guard.load_check_point_ctx();
                info!("{} [check point] recover readers", self.task_ctx.task_id);
                if let Err(err) =
                    self.set_inner_key_off_and_encrytion_key(check_point_ctx.get_first_key())
                {
                    self.scheduler.cancel(format!("{:?}", err))
                }
                self.recover_readers(check_point_ctx);
            }
        }
    }

    fn remove_local_files(&self) {
        self.check_point_store
            .lock()
            .unwrap()
            .clean_check_point_data();

        if let Err(err) = fs::remove_dir_all(&self.task_dir) {
            error!("failed to remove task {}, {:?}", self.task_ctx.task_id, err);
        } else {
            info!("removed local files for task {}", self.task_ctx.task_id);
        }
    }
}

async fn ingest_files_to_leader(
    pd: Arc<dyn PdClient>,
    cs: kvenginepb::ChangeSet,
    region: &metapb::Region,
    mut leader: metapb::Peer,
) -> Result<()> {
    let security_mgr = pd.get_security_mgr();
    let http_client = security_mgr.http_client(hyper::Client::builder())?;
    // Loop for retry on "not leader".
    loop {
        let store = get_leader_store(
            pd.clone() as Arc<dyn PdClient>,
            region.get_id(),
            Some(&leader),
        )
        .await?;
        let uri = security_mgr.build_uri(format!(
            "{}/ingest_files?cluster_id={}",
            &store.status_address,
            pd.get_cluster_id().unwrap()
        ))?;
        let body = cs.write_to_bytes().unwrap();
        let req = Request::post(uri).body(Body::from(body))?;
        let resp = http_client.request(req).await?;
        if !resp.status().is_success() {
            let body = hyper::body::to_bytes(resp.into_body()).await?;
            let mut errpb = kvproto::errorpb::Error::new();
            errpb.merge_from_bytes(&body).unwrap();
            let tag = ShardTag::new(
                store.get_id(),
                IdVer::new(region.get_id(), region.get_region_epoch().get_version()),
            );
            warn!("{} ingest_files_to_leader failed {:?}", tag, errpb);
            if errpb.has_not_leader() {
                leader = errpb.mut_not_leader().take_leader();
                if leader.store_id == 0 {
                    return Err(Error::LeaderNotFound(region.get_id()));
                }
                continue;
            } else if errpb
                .get_message()
                .starts_with(rfstore::errors::INGEST_OVERLAP_ERROR_TAG)
            {
                return Err(Error::IngestOverlap(errpb.take_message()));
            } else {
                return Err(Error::RegionError(region.get_id(), errpb));
            }
        }
        return Ok(());
    }
}

async fn get_shard_meta(
    pd: Arc<dyn PdClient>,
    shard_key: &[u8],
    timeout: Duration,
) -> Result<kvenginepb::ChangeSet> {
    let security_mgr = pd.get_security_mgr();
    let http_client = security_mgr.http_client(hyper::Client::builder())?;
    let start_time = Instant::now_coarse();
    let mut retry = 0;
    loop {
        if start_time.saturating_elapsed() >= timeout {
            return Err(Error::Other(box_err!(
                "get_shard_meta failed, key: {:?}",
                shard_key
            )));
        }
        if retry > 0 {
            tokio::time::sleep(Duration::from_millis(500)).await;
        }
        retry += 1;

        let region_res = pd.get_region_async(shard_key).await;
        if region_res.is_err() {
            error!(
                "get_shard_meta: get region error: {:?}, key: {:?}",
                region_res.unwrap_err(),
                shard_key
            );
            continue;
        }
        let shard_id = region_res.unwrap().get_id();

        let store_res = get_leader_store(pd.clone() as Arc<dyn PdClient>, shard_id, None).await;
        if store_res.is_err() {
            error!(
                "get_shard_meta: get leader error: {:?}, key: {:?}, shard_id: {}",
                store_res.unwrap_err(),
                shard_key,
                shard_id
            );
            continue;
        }
        let store = store_res.unwrap();
        let uri = security_mgr.build_uri(format!(
            "{}/kvengine/meta/{}",
            &store.status_address, shard_id
        ))?;
        let req = Request::get(uri).body(Body::from(""))?;
        match http_client.request(req).await {
            Ok(resp) => {
                if resp.status().is_success() {
                    let body = hyper::body::to_bytes(resp.into_body()).await?;
                    let mut cs = kvenginepb::ChangeSet::default();
                    cs.merge_from_bytes(&body)?;
                    if cs.shard_id == 0 {
                        continue;
                    }
                    return Ok(cs);
                } else {
                    continue;
                }
            }
            Err(e) => {
                error!(
                    "get_shard_meta failed, shard_id: {}, error: {:?}",
                    shard_id, e
                );
                continue;
            }
        }
    }
}

async fn get_leader_store(
    pd: Arc<dyn PdClient>,
    region_id: u64,
    leader: Option<&metapb::Peer>,
) -> Result<metapb::Store> {
    let store_id = if leader.is_none() || leader.unwrap().store_id == 0 {
        let res = pd.get_region_leader_by_id(region_id).await?;
        if res.is_none() {
            return Err(Error::RegionNotFound(region_id));
        }
        let (_, leader) = res.unwrap();
        if leader.store_id == 0 {
            return Err(Error::LeaderNotFound(region_id));
        }
        leader.store_id
    } else {
        leader.unwrap().store_id
    };
    Ok(pd.get_store_async(store_id).await?)
}

fn build_ingest_files(
    inner_key_off: usize,
    region: &metapb::Region,
    sst_metas: &[SstMeta],
    commit_ts: u64,
) -> kvenginepb::ChangeSet {
    let raw_start_key = raw_start_key(region);
    let inner_start_key = &raw_start_key[inner_key_off..];
    let raw_end_key = raw_end_key(region);
    let inner_end_key = &raw_end_key[inner_key_off..];
    let mut cs = kvenginepb::ChangeSet::default();
    cs.set_shard_id(region.get_id());
    cs.set_shard_ver(region.get_region_epoch().get_version());
    let ingest_files = cs.mut_ingest_files();
    ingest_files.set_max_ts(commit_ts);
    // Don't set INGEST_ID_KEY property to indicate that it's from load data.
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
        table_create.set_cf(WRITE_CF as i32);
        table_create.set_level(WRITE_CF_BOTTOM_LEVEL);
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

fn get_ssts_in_range(ssts: &[SstMeta], start: InnerKey<'_>, end: InnerKey<'_>) -> Vec<SstMeta> {
    let position = ssts
        .binary_search_by(|sst| sst.smallest.as_slice().cmp(start.deref()))
        .unwrap();
    let mut matched = vec![];
    for i in position..ssts.len() {
        let sst = &ssts[i];
        if !end.is_empty() && sst.smallest.as_slice() >= end.deref() {
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

    let file = fs::OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .read(true)
        .open(path.as_path())?;
    let mut buf: Vec<u8> = Vec::with_capacity(in_mem_size);
    let iv = if task_ctx.encryption_key.is_some() {
        let mut iv_buf = Vec::with_capacity(16);
        iv_buf.put_u64(task_ctx.start_ts);
        iv_buf.put_u64(task_ctx.commit_ts);
        Iv::from_slice(&iv_buf).unwrap()
    } else {
        Iv::Empty
    };
    let (method, key) = if let Some(key) = &task_ctx.encryption_key {
        (EncryptionMethod::Aes256Ctr, key.current_key.as_slice())
    } else {
        (EncryptionMethod::Plaintext, "".as_bytes())
    };
    let mut writer = EncrypterWriter::new(file, method, key, iv).unwrap();
    for pair in &kv_pairs {
        buf.put_u16_le(pair.key.len() as u16);
        buf.extend_from_slice(pair.key.chunk());
        buf.put_u32_le(pair.val.len() as u32);
        buf.extend_from_slice(pair.val.chunk());
        if buf.len() >= 128 * 1024 {
            writer.write_all(&buf)?;
            buf.clear();
        }
    }
    if !buf.is_empty() {
        writer.write_all(&buf)?;
    }
    writer.flush()?;
    let file = fs::File::open(path)?;
    let reader = DecrypterReader::new(file, method, key, iv).unwrap();
    Ok(KvPairsReader::new(
        task_ctx.start_ts,
        task_ctx.commit_ts,
        kv_pairs.len(),
        reader,
    ))
}

fn reload_reader(local_file_info: &LocalFileInfo, task_ctx: TaskContext) -> KvPairsReader {
    let file = fs::File::open(local_file_info.clone().path).unwrap();

    let (method, key) = if let Some(key) = &task_ctx.encryption_key {
        (EncryptionMethod::Aes256Ctr, key.current_key.as_slice())
    } else {
        (EncryptionMethod::Plaintext, "".as_bytes())
    };

    let iv = if task_ctx.encryption_key.is_some() {
        let mut iv_buf = Vec::with_capacity(16);
        iv_buf.put_u64(task_ctx.start_ts);
        iv_buf.put_u64(task_ctx.commit_ts);
        Iv::from_slice(&iv_buf).unwrap()
    } else {
        Iv::Empty
    };

    let reader = DecrypterReader::new(file, method, key, iv).unwrap();
    KvPairsReader::new(
        task_ctx.clone().start_ts,
        task_ctx.clone().commit_ts,
        local_file_info.kv_count,
        reader,
    )
}

fn verify_regions_boundary(
    start_key: &[u8],
    end_key: &[u8],
    regions: &[pdpb::Region],
) -> Result<()> {
    if regions.is_empty() {
        return Err(box_err!("no region"));
    }

    let first_region = regions.first().unwrap();
    let last_region = regions.last().unwrap();
    if first_region.get_region().get_start_key() > start_key {
        return Err(Error::RegionsIntegrityError(format!(
            "unexpected start key of first region: {:?}, start_key: {:?}",
            first_region, start_key
        )));
    } else if last_region.get_region().get_end_key() < end_key {
        return Err(Error::RegionsIntegrityError(format!(
            "unexpected end key of last region: {:?}, end_key: {:?}",
            last_region, end_key
        )));
    }

    for region in regions.windows(2) {
        if region[0].get_region().get_end_key() != region[1].get_region().get_start_key() {
            return Err(Error::RegionsIntegrityError(format!(
                "region boundary not match: {:?}, {:?}",
                region[0], region[1]
            )));
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_verify_regions_boundary() {
        let make_key = |key: u64| -> Vec<u8> { format!("k{:02}", key).into_bytes() };
        let make_region = |start: u64, end: u64| -> pdpb::Region {
            let mut region = metapb::Region::default();
            region.set_start_key(make_key(start));
            region.set_end_key(make_key(end));

            let mut pd_region = pdpb::Region::default();
            pd_region.set_region(region);
            pd_region
        };

        let regions1 = vec![
            make_region(1, 2),
            make_region(2, 4),
            make_region(4, 6),
            make_region(6, 10),
            make_region(10, 14),
        ];
        let regions2 = vec![make_region(1, 2), make_region(4, 6)];

        let cases = vec![
            (&regions1, 1, 14, true), // start, end, expect_is_ok
            (&regions1, 1, 2, true),
            (&regions1, 0, 2, false),
            (&regions1, 2, 15, false),
            (&regions2, 1, 6, false),
        ];
        for (idx, (regions, start, end, expect_is_ok)) in cases.into_iter().enumerate() {
            let res = verify_regions_boundary(&make_key(start), &make_key(end), regions);
            assert_eq!(res.is_ok(), expect_is_ok, "case {}: {:?}", idx, res);
        }
    }
}
