// Copyright 2024 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    cmp::{max, min, Ordering},
    collections::HashMap,
    fs,
    fs::File,
    io::Write,
    mem,
    path::PathBuf,
    sync::{Arc, Mutex},
    time::Duration,
};

use api_version::api_v2::KEYSPACE_PREFIX_LEN;
use bytes::{Buf, BufMut, Bytes};
use encryption::{DecrypterReader, EncrypterWriter, Iv};
use kvengine::{
    dfs::Options,
    table::{sstable::Builder, InnerKey, Value},
    UserMeta,
};
use kvproto::{encryptionpb::EncryptionMethod, metapb};
use rfengine::compress_lz4;
use tikv_util::{
    codec::bytes::decode_bytes,
    error, info,
    merge_range::MergeRanges,
    mpsc::{Receiver, Sender},
    time::Instant,
    warn,
};

use crate::{
    checkpoint::{FileMeta, LocalFileCheckpointStorage},
    error::{Error, Result},
    kv::{DuplicateEntry, KvPair, KvPairsReader, MergeIterator, SstMeta},
    metrics::LOAD_DATA_WRU_COST_COUNTER,
    task::{
        build_ingest_files, gen_split_keys, get_common_prefix, get_ssts_in_range,
        ingest_files_to_leader, new_region_key, verify_regions_boundary, FlushResult, FlushStates,
        LoadDataConfig, LoadDataContext, LoadTaskScheduler, PutChunkResult, TaskContext,
        ALLOCATE_ID_TIMEOUT, CREATE_FILE_CONCURRENCY, DEFAULT_AVG_BATCH_PROPORTION,
        FLUSH_FILE_CONCURRENCY, INGEST_CONCURRENCY, MAX_RETRY_TIMES, MAX_SLEEP_DURATION,
        REPLICA_NUMS, RETRY_SLEEP_DURATION, TXN_FILE_RU_DISCOUNT_RATIO, ZSTD_COMPRESSION_LEVEL,
    },
};

pub enum KvPairsWorkerMsg {
    AddChunk {
        writer_id: u64,
        chunk_id: u64,
        chunk_data: Bytes,
        cb: Box<dyn FnOnce(PutChunkResult) + Send>,
    },
    Flush {
        writer_id: u64,
        flush_file_count: Option<usize>,
        cb: Box<dyn FnOnce(FlushStates) + Send>,
    },
    CollectFileMetas {
        skip_sort: bool,
        cb: Box<dyn FnOnce((Vec<FileMeta>, Vec<DuplicateEntry>, Vec<u8>)) + Send>,
    },
    Cleanup,
}

pub struct UnhandledFlushFile {
    pub handled_chunk_ids: HashMap<u64 /* writer_id */, u64 /* chunk_id */>,
    pub file_idx: usize,
    pub key_comm_prefix: Vec<u8>,
    pub file_meta: FileMeta,
}

pub struct KvPairsWorker {
    worker_id: u64,
    config: LoadDataConfig,
    task_ctx: TaskContext,
    l0_data_dir: PathBuf,
    l1_data_dir: PathBuf,

    kv_pairs: Vec<KvPair>,
    in_mem_size: usize,
    l1_file_idx: usize,
    flush_file_errs: Vec<Error>,
    unhandled_flush_files: Vec<UnhandledFlushFile>,

    scheduler: LoadTaskScheduler,
    receiver: Receiver<KvPairsWorkerMsg>,
    file_tx: Sender<Result<UnhandledFlushFile>>,
    file_rx: Receiver<Result<UnhandledFlushFile>>,
    checkpoint: Arc<Mutex<LocalFileCheckpointStorage>>,

    // the following fields need to be persisted
    l0_file_idx: usize,
    l0_file_metas: Vec<FileMeta>,
    l1_file_metas: Vec<FileMeta>,
    dup_entries: Vec<DuplicateEntry>,
    key_comm_prefix: Vec<u8>,
    handled_chunk_ids: HashMap<u64, u64>,
    flushed_chunk_ids: HashMap<u64, u64>,
}

impl KvPairsWorker {
    pub fn new(
        worker_id: u64,
        config: LoadDataConfig,
        task_ctx: TaskContext,
        task_dir: PathBuf,
        receiver: Receiver<KvPairsWorkerMsg>,
        scheduler: LoadTaskScheduler,
        checkpoint: Arc<Mutex<LocalFileCheckpointStorage>>,
    ) -> Self {
        let mut checkpoint_guard = checkpoint.lock().unwrap();
        let worker_ctx = checkpoint_guard
            .checkpoint_ctx
            .get_kvpairs_worker_ctx(worker_id);

        let l0_file_metas = worker_ctx.l0_file_metas.clone();
        let l0_file_idx = l0_file_metas.len();
        let handled_chunk_ids = worker_ctx.flushed_chunk_ids.clone();
        let flushed_chunk_ids = worker_ctx.flushed_chunk_ids.clone();
        let l1_file_metas = worker_ctx.l1_file_metas.clone();
        let dup_entries = worker_ctx.duplicated_entries.clone();

        let mut key_comm_prefix = worker_ctx.key_comm_prefix.clone();
        if key_comm_prefix.is_empty() {
            let first_key = checkpoint_guard.checkpoint_ctx.get_first_key();
            key_comm_prefix = first_key.slice(task_ctx.inner_key_off.unwrap()..).to_vec();
        }
        drop(checkpoint_guard);

        let (file_tx, file_rx) = tikv_util::mpsc::unbounded();
        let l0_data_dir = task_dir
            .join(format!("worker-{}", worker_id))
            .join("l0-files");
        let l1_data_dir = task_dir
            .join(format!("worker-{}", worker_id))
            .join("l1-files");

        Self {
            worker_id,
            config,
            task_ctx,
            l0_data_dir,
            l1_data_dir,
            kv_pairs: vec![],
            in_mem_size: 0,
            l0_file_idx,
            l1_file_idx: 0,
            l0_file_metas,
            l1_file_metas,
            flush_file_errs: vec![],
            scheduler,
            receiver,
            unhandled_flush_files: vec![],
            file_tx,
            file_rx,
            checkpoint,
            key_comm_prefix,
            handled_chunk_ids,
            flushed_chunk_ids,
            dup_entries,
        }
    }

    pub fn run(&mut self) {
        self.init_task_dir();
        info!(
            "{} run kvpairs worker-{}, key comm prefix: {:?}, l0 file idx: {}, l0 file metas: {}, l1 file metas: {}, duplicated entries: {}",
            self.task_ctx.task_id,
            self.worker_id,
            self.key_comm_prefix,
            self.l0_file_idx,
            self.l0_file_metas.len(),
            self.l1_file_metas.len(),
            self.dup_entries.len(),
        );

        while let Ok(msg) = self.receiver.recv() {
            match msg {
                KvPairsWorkerMsg::AddChunk {
                    writer_id,
                    chunk_id,
                    chunk_data,
                    cb,
                } => {
                    if let Err(err) = self.handle_add_chunk(writer_id, chunk_id, chunk_data, cb) {
                        error!(
                            "{} worker-{} failed to handle add chunk, error: {:?}",
                            self.task_ctx.task_id, self.worker_id, err
                        );
                        self.scheduler.cancel(format!(
                            "{} worker-{} error: {:?}",
                            self.task_ctx.task_id, self.worker_id, err
                        ));
                        if self.l0_file_idx
                            > self.flush_file_errs.len()
                                + self.l0_file_metas.len()
                                + self.unhandled_flush_files.len()
                        {
                            let recv_count = self.l0_file_idx
                                - self.l0_file_metas.len()
                                - self.flush_file_errs.len()
                                - self.unhandled_flush_files.len();
                            let _ = self.recv_flush_file(recv_count);
                        }
                    }
                }

                KvPairsWorkerMsg::Flush {
                    writer_id: _,
                    flush_file_count,
                    cb,
                } => {
                    self.handle_flush_msg(flush_file_count, cb);
                }

                KvPairsWorkerMsg::CollectFileMetas { skip_sort, cb } => {
                    match self.collect_file_metas(skip_sort) {
                        Ok((file_metas, dup_entries)) => {
                            cb((file_metas, dup_entries, self.key_comm_prefix.clone()));
                        }
                        Err(err) => {
                            error!(
                                "{} worker-{} failed to collect file metas, error: {:?}",
                                self.task_ctx.task_id, self.worker_id, err
                            );
                            self.scheduler.cancel(format!(
                                "{} worker-{} error: {:?}",
                                self.task_ctx.task_id, self.worker_id, err
                            ));
                            cb((vec![], vec![], vec![]));
                        }
                    }
                }
                KvPairsWorkerMsg::Cleanup => {
                    return;
                }
            }
        }
    }

    fn init_task_dir(&mut self) {
        if !self.l0_data_dir.is_dir() {
            if let Err(err) = fs::create_dir_all(&self.l0_data_dir) {
                error!(
                    "{} worker-{} failed to create dir {:?}, error {:?}",
                    self.task_ctx.task_id, self.worker_id, self.l0_data_dir, err
                );
                self.scheduler.cancel(format!(
                    "{} worker-{} error: {:?}",
                    self.task_ctx.task_id, self.worker_id, err
                ));
                return;
            }
        }

        // If worker restarts during the building l1 files phase, should clean
        // up the previously created files after restarting.
        if self.l1_data_dir.is_dir() && self.l1_file_metas.is_empty() {
            if let Err(err) = fs::remove_dir_all(&self.l1_data_dir) {
                error!(
                    "{} worker-{} failed to remove l1 data dir, {:?}",
                    self.task_ctx.task_id, self.worker_id, err,
                );
                self.scheduler.cancel(format!(
                    "{} worker-{} error: {:?}",
                    self.task_ctx.task_id, self.worker_id, err
                ));
                return;
            }
        }

        if let Err(err) = fs::create_dir_all(&self.l1_data_dir) {
            error!(
                "{} worker-{} failed to create dir {:?}, error {:?}",
                self.task_ctx.task_id, self.worker_id, self.l1_data_dir, err
            );
            self.scheduler.cancel(format!(
                "{} worker-{} error: {:?}",
                self.task_ctx.task_id, self.worker_id, err
            ));
        }
    }

    fn handle_add_chunk(
        &mut self,
        writer_id: u64,
        chunk_id: u64,
        chunk_data: Bytes,
        cb: Box<dyn FnOnce(PutChunkResult) + Send>,
    ) -> Result<()> {
        let handled_chunk_id = self.get_handled_chunk_id(writer_id);
        let flushed_chunk_id = self.get_flushed_chunk_id(writer_id);
        let mut result = PutChunkResult {
            handled_chunk_id,
            flushed_chunk_id,
            canceled: self.scheduler.is_canceled(),
            finished: self.scheduler.is_finished(),
            error: self.scheduler.error_msg(),
        };

        if result.canceled || result.finished {
            warn!(
                "{} worker-{} skip chunk {}, canceled: {}, finished: {}, error message: {}",
                self.task_ctx.task_id,
                self.worker_id,
                chunk_id,
                result.canceled,
                result.finished,
                result.error,
            );
            cb(result);
            return Err(Error::Canceled);
        }

        if chunk_data.is_empty() {
            warn!(
                "{} worker-{} skip chunk, chunk {} is empty for writer {}",
                self.task_ctx.task_id, self.worker_id, chunk_id, writer_id
            );
            cb(result);
            return Ok(());
        }

        if handled_chunk_id != chunk_id - 1 {
            warn!(
                "{} worker-{} skip chunk {} for writer {}, expect chunk {}",
                self.task_ctx.task_id,
                self.worker_id,
                chunk_id,
                writer_id,
                handled_chunk_id + 1
            );
            result.error = format!(
                "skip chunk {} for writer {}, expect chunk {}",
                chunk_id,
                writer_id,
                handled_chunk_id + 1
            );
            cb(result);
            return Ok(());
        }

        info!(
            "{} worker-{} handle add chunk {} with len {} for writer {}",
            self.task_ctx.task_id,
            self.worker_id,
            chunk_id,
            chunk_data.len(),
            writer_id,
        );

        self.handled_chunk_ids.insert(writer_id, chunk_id);
        result.handled_chunk_id = chunk_id;
        // call `cb` early to avoid waiting for flush
        cb(result);

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
            let row_id_len = (&chunk_data[offset..]).get_u16_le();
            offset += 2;
            let row_id = chunk_data.slice(offset..offset + row_id_len as usize);
            offset += row_id_len as usize;

            let outer_key_prefix = key.slice(..inner_key_off);
            if outer_key_prefix.chunk() != self.task_ctx.outer_key_prefix.as_slice() {
                let err_msg = format!(
                    "key prefix inconsistent, first key: {:?}, current key: {:?}",
                    self.task_ctx.outer_key_prefix,
                    outer_key_prefix.chunk()
                );
                error!(
                    "{} worker-{} {}",
                    self.task_ctx.task_id, self.worker_id, err_msg
                );
                return Err(Error::CheckError(err_msg));
            }

            let inner_key = key.slice(inner_key_off..);
            self.kv_pairs.push(KvPair::new(inner_key, val, row_id));
            self.in_mem_size += 2 + key_len as usize + 4 + val_len as usize;
        }

        if self.in_mem_size > self.config.max_in_mem_size {
            self.flush_mem_buf();
            if self.l0_file_idx
                > self.l0_file_metas.len()
                    + self.flush_file_errs.len()
                    + self.unhandled_flush_files.len()
                    + FLUSH_FILE_CONCURRENCY
            {
                self.recv_flush_file(1)?;
            }
        }
        self.try_recv_flush_file()
    }

    fn l0_file_path(&self, file_idx: usize) -> PathBuf {
        self.l0_data_dir.join(format!("l0_kv_pairs_{}", file_idx))
    }

    fn l1_file_path(&self, file_idx: usize) -> PathBuf {
        self.l1_data_dir.join(format!("l1_kv_pairs_{}", file_idx))
    }

    fn try_recv_flush_file(&mut self) -> Result<()> {
        while let Ok(flush_file_result) = self.file_rx.try_recv() {
            match flush_file_result {
                Ok(flush_file) => {
                    self.unhandled_flush_files.push(flush_file);
                }
                Err(err) => {
                    self.flush_file_errs.push(err);
                }
            }
        }
        if !self.flush_file_errs.is_empty() {
            return Err(self.flush_file_errs.pop().unwrap());
        }
        if self.unhandled_flush_files.is_empty() {
            return Ok(());
        }
        self.handle_flush_files()
    }

    fn handle_flush_files(&mut self) -> Result<()> {
        self.unhandled_flush_files
            .sort_by(|a, b| a.file_idx.cmp(&b.file_idx));

        let handled_file_idx = self.l0_file_metas.len() + self.flush_file_errs.len();
        let mut need_handled = self.unhandled_flush_files.len();
        for (idx, unhandled_flush_file) in self.unhandled_flush_files.iter().enumerate() {
            assert!(unhandled_flush_file.file_idx >= handled_file_idx + idx);
            if unhandled_flush_file.file_idx > handled_file_idx + idx {
                need_handled = idx;
                break;
            }
        }
        if need_handled == 0 {
            return Ok(());
        }
        info!(
            "{} worker-{} need handle {} flush files",
            self.task_ctx.task_id, self.worker_id, need_handled
        );

        let mut handled_chunk_ids: HashMap<u64, u64> = HashMap::new();
        let mut file_metas = vec![];
        let mut last_key_comm_prefix = self.key_comm_prefix.clone();
        for mut unhandled_flush_file in self.unhandled_flush_files.drain(0..need_handled) {
            handled_chunk_ids = mem::take(&mut unhandled_flush_file.handled_chunk_ids);
            last_key_comm_prefix =
                get_common_prefix(&last_key_comm_prefix, &unhandled_flush_file.key_comm_prefix);

            file_metas.push(unhandled_flush_file.file_meta);
        }
        self.scheduler.add_flushed_files(need_handled);

        let mut checkpoint_guard = self.checkpoint.lock().unwrap();
        checkpoint_guard.update_l0_flushed_info(
            self.worker_id,
            handled_chunk_ids.clone(),
            file_metas.clone(),
            last_key_comm_prefix.clone(),
        )?;

        self.flushed_chunk_ids = handled_chunk_ids;
        self.key_comm_prefix = last_key_comm_prefix;
        self.l0_file_metas.append(&mut file_metas);

        info!(
            "{} worker-{} handle {} flush files, current key common prefix: {:?}",
            self.task_ctx.task_id, self.worker_id, need_handled, self.key_comm_prefix
        );
        Ok(())
    }

    fn handle_flush_msg(
        &mut self,
        flush_file_count: Option<usize>,
        cb: Box<dyn FnOnce(FlushStates) + Send>,
    ) {
        if self.scheduler.is_canceled() || self.scheduler.is_finished() {
            let flush_result = FlushResult {
                flushed_chunk_ids: HashMap::default(),
                canceled: self.scheduler.is_canceled(),
                finished: self.scheduler.is_finished(),
                error: self.scheduler.error_msg(),
            };
            warn!(
                "{} worker-{} skip flushing, canceled: {}, finished: {}, error message: {}",
                self.task_ctx.task_id,
                self.worker_id,
                flush_result.canceled,
                flush_result.finished,
                flush_result.error,
            );
            cb(FlushStates::FlushResult { flush_result });
            return;
        }

        let file_count;
        match flush_file_count {
            Some(flush_file_count) => {
                let result = if self.l0_file_idx < flush_file_count {
                    self.flush()
                } else {
                    self.try_recv_flush_file()
                };
                if let Err(err) = result {
                    error!(
                        "{} worker-{} failed to flush {:?}",
                        self.task_ctx.task_id, self.worker_id, err
                    );
                    self.scheduler.cancel(format!(
                        "{} worker-{} error: {:?}",
                        self.task_ctx.task_id, self.worker_id, err
                    ));
                    let flush_result = FlushResult {
                        flushed_chunk_ids: HashMap::new(),
                        canceled: true,
                        finished: false,
                        error: self.scheduler.error_msg(),
                    };
                    cb(FlushStates::FlushResult { flush_result });
                    if self.l0_file_idx
                        > self.flush_file_errs.len()
                            + self.l0_file_metas.len()
                            + self.unhandled_flush_files.len()
                    {
                        let recv_count = self.l0_file_idx
                            - self.flush_file_errs.len()
                            - self.l0_file_metas.len()
                            - self.unhandled_flush_files.len();
                        let _ = self.recv_flush_file(recv_count);
                    }
                    return;
                }
                file_count = flush_file_count;
            }
            None => {
                file_count = self.l0_file_idx + if self.kv_pairs.is_empty() { 0 } else { 1 };
                info!(
                    "{} worker-{} flush memory buffer, got file count {}",
                    self.task_ctx.task_id, self.worker_id, file_count
                );
            }
        }

        if self.l0_file_metas.len() + self.flush_file_errs.len() >= file_count {
            info!(
                "{} worker-{} has flushed files {}",
                self.task_ctx.task_id, self.worker_id, file_count
            );
            let flush_result = FlushResult {
                flushed_chunk_ids: self.flushed_chunk_ids.clone(),
                canceled: false,
                finished: false,
                error: self.scheduler.error_msg(),
            };
            cb(FlushStates::FlushResult { flush_result });
        } else {
            cb(FlushStates::FlushFileCount {
                flush_file_count: file_count,
            });
        }
    }

    fn flush(&mut self) -> Result<()> {
        if !self.flush_file_errs.is_empty() {
            return Err(self.flush_file_errs.pop().unwrap());
        }

        if !self.kv_pairs.is_empty() {
            self.flush_mem_buf();
        }

        self.try_recv_flush_file()
    }

    fn recv_flush_file(&mut self, mut recv_count: usize) -> Result<()> {
        while recv_count != 0 {
            match self.file_rx.recv().unwrap() {
                Ok(flush_file) => {
                    self.unhandled_flush_files.push(flush_file);
                }
                Err(err) => {
                    self.flush_file_errs.push(err);
                }
            }
            recv_count -= 1;
        }
        if !self.flush_file_errs.is_empty() {
            return Err(self.flush_file_errs.pop().unwrap());
        }
        self.handle_flush_files()
    }

    fn collect_file_metas(
        &mut self,
        skip_sort: bool,
    ) -> Result<(Vec<FileMeta>, Vec<DuplicateEntry>)> {
        if self.scheduler.is_canceled() || self.scheduler.is_finished() {
            warn!(
                "{} worker-{} skip collecting file meats, canceled: {}, finished: {}, error message: {}",
                self.task_ctx.task_id,
                self.worker_id,
                self.scheduler.is_canceled(),
                self.scheduler.is_finished(),
                self.scheduler.error_msg(),
            );
            return Err(Error::Canceled);
        }

        if !self.kv_pairs.is_empty() || self.l0_file_metas.len() < self.l0_file_idx {
            error!(
                "{} worker-{} has {} not yet flushed kv pairs, {} unhandled files",
                self.task_ctx.task_id,
                self.worker_id,
                self.kv_pairs.len(),
                self.l0_file_idx - self.l0_file_metas.len()
            );
            return Err(Error::CheckError(format!(
                "has {} not yet flushed kv pairs, {} unhandled files",
                self.kv_pairs.len(),
                self.l0_file_idx - self.l0_file_metas.len()
            )));
        }

        if skip_sort {
            return Ok((mem::take(&mut self.l0_file_metas), vec![]));
        }

        if !self.l1_file_metas.is_empty() {
            return Ok((
                mem::take(&mut self.l1_file_metas),
                mem::take(&mut self.dup_entries),
            ));
        }

        if self.l0_file_metas.is_empty() {
            info!(
                "{} worker-{} skip sorting empty data",
                self.task_ctx.task_id, self.worker_id
            );
            return Ok((vec![], vec![]));
        }

        info!(
            "{} worker-{} start to sort with key common prefix {:?}",
            self.task_ctx.task_id, self.worker_id, self.key_comm_prefix
        );
        let start = Instant::now();
        let (tx, rx) = tikv_util::mpsc::unbounded();
        let mut sent_count = 0;
        let mut recv_count = 0;

        let file_metas = mem::take(&mut self.l0_file_metas);
        let readers = build_readers(
            &self.task_ctx,
            file_metas,
            self.key_comm_prefix.len(),
            vec![],
            vec![],
        );
        let mut merge_iter = MergeIterator::new(
            readers,
            &self.task_ctx.outer_key_prefix,
            self.key_comm_prefix.len(),
        )?;

        let mut errs = vec![];
        let mut batches = vec![];
        let mut kv_count = 0;
        let mut kv_size = 0;
        while merge_iter.valid() {
            let (batch, batch_kv_count) =
                self.read_batch(&mut merge_iter, self.config.flush_batch_size)?;
            if batch.is_empty() {
                break;
            }

            kv_size += batch.len();
            kv_count += batch_kv_count;
            batches.push(batch);
            if kv_size > self.config.max_in_mem_size {
                self.flush_batches(
                    &mut batches,
                    kv_size,
                    kv_count,
                    merge_iter.prev_key().to_vec(),
                    tx.clone(),
                );
                kv_size = 0;
                kv_count = 0;
                sent_count += 1;
                if sent_count > FLUSH_FILE_CONCURRENCY {
                    recv_count += 1;
                    match rx.recv().unwrap() {
                        Ok(l1_file) => {
                            self.l1_file_metas.push(l1_file);
                        }
                        Err(err) => {
                            errs.push(err);
                            break;
                        }
                    }
                }
            }
        }
        if !batches.is_empty() {
            self.flush_batches(
                &mut batches,
                kv_size,
                kv_count,
                merge_iter.prev_key().to_vec(),
                tx.clone(),
            );
            sent_count += 1;
        }
        for _ in 0..(sent_count - recv_count) {
            match rx.recv().unwrap() {
                Ok(l1_file) => {
                    self.l1_file_metas.push(l1_file);
                }
                Err(err) => {
                    errs.push(err);
                }
            }
        }
        if !errs.is_empty() {
            return Err(errs.pop().unwrap());
        }

        if !merge_iter.duplicated_entries.is_empty() {
            info!(
                "{} worker-{} got {} duplicated entries, size {}",
                self.task_ctx.task_id,
                self.worker_id,
                merge_iter.duplicated_entries.len(),
                merge_iter.duplicated_entries_size
            );
            self.dup_entries = merge_iter.duplicated_entries;
        }
        self.scheduler.add_flushed_files(self.l1_file_metas.len());

        let mut checkpoint_guard = self.checkpoint.lock().unwrap();
        checkpoint_guard.update_l1_flushed_info(
            self.worker_id,
            self.l1_file_metas.clone(),
            self.dup_entries.clone(),
        )?;
        info!(
            "{} worker-{} finish sorting, takes {:?}",
            self.task_ctx.task_id,
            self.worker_id,
            start.saturating_elapsed(),
        );

        Ok((
            mem::take(&mut self.l1_file_metas),
            mem::take(&mut self.dup_entries),
        ))
    }

    fn flush_mem_buf(&mut self) {
        let kv_pairs = mem::take(&mut self.kv_pairs);
        let kv_count = kv_pairs.len();
        let tx = self.file_tx.clone();
        let task_ctx = self.task_ctx.clone();
        let file_path = self.l0_file_path(self.l0_file_idx);
        let batch_size = self.config.flush_batch_size;
        let handled_chunk_ids = self.handled_chunk_ids.clone();
        let file_idx = self.l0_file_idx;
        let worker_id = self.worker_id;
        let in_mem_size = self.in_mem_size;
        self.in_mem_size = 0;
        self.l0_file_idx += 1;
        std::thread::spawn(move || {
            let task_id = task_ctx.task_id.clone();
            let start = Instant::now();
            match flush_l0_file_to_local(kv_pairs, task_ctx, file_path.clone(), batch_size) {
                Ok((first_key, last_key, key_comm_prefix, kv_size)) => {
                    tx.send(Ok(UnhandledFlushFile {
                        handled_chunk_ids,
                        file_idx,
                        key_comm_prefix,
                        file_meta: FileMeta {
                            file_path,
                            kv_count,
                            kv_size,
                            first_key,
                            last_key,
                        },
                    }))
                    .unwrap();
                }
                Err(err) => {
                    tx.send(Err(err)).unwrap();
                }
            }
            info!(
                "{} worker-{} flush to local file on in_mem_size {}, file index: {}, takes {:?}",
                task_id,
                worker_id,
                in_mem_size,
                file_idx,
                start.saturating_elapsed(),
            );
        });
    }

    fn flush_batches(
        &mut self,
        batches: &mut Vec<Vec<u8>>,
        kv_size: usize,
        kv_count: usize,
        last_key: Vec<u8>,
        tx: Sender<Result<FileMeta>>,
    ) {
        let batches = mem::take(batches);
        let task_ctx = self.task_ctx.clone();
        let file_idx = self.l1_file_idx;
        let file_path = self.l1_file_path(file_idx);
        let worker_id = self.worker_id;

        self.l1_file_idx += 1;
        std::thread::spawn(move || {
            let first_batch = &batches[0];
            let first_key_len = (&first_batch[0..]).get_u16_le();
            let first_key = first_batch[2..2 + first_key_len as usize].to_vec();
            let task_id = task_ctx.task_id.clone();
            let start = Instant::now();
            match flush_l1_file_to_local(batches, task_ctx, file_path.clone()) {
                Ok(()) => {
                    tx.send(Ok(FileMeta {
                        file_path,
                        kv_count,
                        kv_size,
                        first_key,
                        last_key,
                    }))
                    .unwrap();
                }
                Err(err) => {
                    tx.send(Err(err)).unwrap();
                }
            }
            info!(
                "{} worker-{} flush batches to local disk, file index: {}, takes {:?}",
                task_id,
                worker_id,
                file_idx,
                start.saturating_elapsed(),
            );
        });
    }

    fn read_batch(
        &mut self,
        merge_iter: &mut MergeIterator,
        batch_size: usize,
    ) -> Result<(Vec<u8>, usize)> {
        let mut buf = Vec::with_capacity(batch_size);
        let mut kv_count = 0;
        while merge_iter.valid() {
            let key = merge_iter.key();
            let key_len = key.len();
            let val = merge_iter.value();
            let val_len = val.len();
            let row_id = merge_iter.row_id();
            let row_id_len = row_id.len();
            if buf.len() + 2 + key_len + 4 + val_len + 2 + row_id_len > buf.capacity() {
                return Ok((buf, kv_count));
            }

            buf.put_u16_le(key_len as u16);
            buf.extend_from_slice(key);
            buf.put_u32_le(val_len as u32);
            buf.extend_from_slice(val);
            buf.put_u16_le(row_id_len as u16);
            buf.extend_from_slice(row_id);
            kv_count += 1;
            merge_iter.next()?;
        }
        Ok((buf, kv_count))
    }

    fn get_handled_chunk_id(&mut self, writer_id: u64) -> u64 {
        let chunk_id = self.handled_chunk_ids.entry(writer_id).or_default();
        *chunk_id
    }

    fn get_flushed_chunk_id(&mut self, writer_id: u64) -> u64 {
        let chunk_id = self.flushed_chunk_ids.entry(writer_id).or_default();
        *chunk_id
    }
}

fn new_file_writer(task_ctx: TaskContext, path: PathBuf) -> Result<EncrypterWriter<File>> {
    let file = fs::OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .read(true)
        .open(path.as_path())?;
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
    Ok(EncrypterWriter::new(file, method, key, iv).unwrap())
}

fn flush_l1_file_to_local(
    batches: Vec<Vec<u8>>,
    task_ctx: TaskContext,
    path: PathBuf,
) -> Result<()> {
    let mut writer = new_file_writer(task_ctx, path)?;
    let mut compressed_buf = vec![];
    for batch in &batches {
        let compressed_size = compress_lz4(batch, &mut compressed_buf)? as u32;
        writer.write_all(&compressed_size.to_le_bytes())?;
        writer.write_all(&compressed_buf)?;
        compressed_buf.clear();
    }
    writer.flush()?;
    Ok(())
}

fn flush_l0_file_to_local(
    mut kv_pairs: Vec<KvPair>,
    task_ctx: TaskContext,
    path: PathBuf,
    batch_size: usize,
) -> Result<(Vec<u8>, Vec<u8>, Vec<u8>, usize)> {
    kv_pairs.sort_by(|a, b| {
        let order = a.key.cmp(&b.key);
        if order == Ordering::Equal {
            return a.row_id.cmp(&b.row_id);
        }
        order
    });

    let first_key = kv_pairs.first().unwrap().key.to_vec();
    let last_key = kv_pairs.last().unwrap().key.to_vec();
    let key_comm_prefix = get_common_prefix(&first_key, &last_key);

    let mut writer = new_file_writer(task_ctx, path)?;
    let mut kv_size = 0;
    let mut buf: Vec<u8> = Vec::with_capacity(batch_size + batch_size / 8);
    let mut compressed_buf: Vec<u8> = Vec::with_capacity(batch_size + batch_size / 8);
    for pair in &kv_pairs {
        buf.put_u16_le(pair.key.len() as u16);
        buf.extend_from_slice(pair.key.chunk());
        buf.put_u32_le(pair.val.len() as u32);
        buf.extend_from_slice(pair.val.chunk());
        buf.put_u16_le(pair.row_id.len() as u16);
        buf.extend_from_slice(pair.row_id.chunk());

        if buf.len() >= batch_size {
            let compressed_size = compress_lz4(&buf, &mut compressed_buf)? as u32;
            writer.write_all(&compressed_size.to_le_bytes())?;
            writer.write_all(&compressed_buf)?;
            kv_size += buf.len();
            buf.clear();
            compressed_buf.clear();
        }
    }
    if !buf.is_empty() {
        let compressed_size = compress_lz4(&buf, &mut compressed_buf)? as u32;
        writer.write_all(&compressed_size.to_le_bytes())?;
        writer.write_all(&compressed_buf)?;
        kv_size += buf.len();
    }
    writer.flush()?;
    Ok((first_key, last_key, key_comm_prefix, kv_size))
}

pub enum BuildingWorkerMsg {
    Build {
        start_key: Vec<u8>,
        end_key: Vec<u8>,
        file_metas: Vec<FileMeta>,
        compression_type: u8,
        cb: Box<dyn FnOnce(Vec<DuplicateEntry>) + Send>,
    },
    Ingest {
        cb: Box<dyn FnOnce(()) + Send>,
    },
    Cleanup,
}

pub struct BuildingWorker {
    worker_id: u64,
    config: LoadDataConfig,
    ctx: LoadDataContext,
    task_ctx: TaskContext,
    scheduler: LoadTaskScheduler,
    receiver: Receiver<BuildingWorkerMsg>,
    cached_file_ids: Vec<u64>,

    key_comm_prefix: Vec<u8>,
    checkpoint_store: Arc<Mutex<LocalFileCheckpointStorage>>,

    // the following fields need to be persisted
    sst_metas: Vec<SstMeta>,
    dup_entries: Vec<DuplicateEntry>,
    ingested: bool,
}

impl BuildingWorker {
    pub fn new(
        worker_id: u64,
        config: LoadDataConfig,
        ctx: LoadDataContext,
        task_ctx: TaskContext,
        key_comm_prefix: Vec<u8>,
        receiver: Receiver<BuildingWorkerMsg>,
        scheduler: LoadTaskScheduler,
        checkpoint_store: Arc<Mutex<LocalFileCheckpointStorage>>,
    ) -> Self {
        let mut checkpoint_guard = checkpoint_store.lock().unwrap();
        let worker_ctx = checkpoint_guard
            .checkpoint_ctx
            .get_building_worker_ctx(worker_id);

        let sst_metas = worker_ctx.sst_metas.clone();
        let ingested = worker_ctx.ingested;
        let dup_entries = worker_ctx.duplicated_entries.clone();
        drop(checkpoint_guard);

        Self {
            worker_id,
            config,
            ctx,
            task_ctx,
            receiver,
            scheduler,
            cached_file_ids: vec![],
            key_comm_prefix,
            checkpoint_store,
            sst_metas,
            dup_entries,
            ingested,
        }
    }

    pub fn run(&mut self) {
        info!(
            "{} run building worker-{}, key comm prefix: {:?}, sst metas: {}, duplicated entries: {}",
            self.task_ctx.task_id,
            self.worker_id,
            self.key_comm_prefix,
            self.sst_metas.len(),
            self.dup_entries.len(),
        );
        while let Ok(msg) = self.receiver.recv() {
            match msg {
                BuildingWorkerMsg::Build {
                    start_key,
                    end_key,
                    file_metas,
                    compression_type,
                    cb,
                } => match self.build(file_metas, start_key, end_key, compression_type) {
                    Ok(dup_entries) => {
                        cb(dup_entries);
                    }
                    Err(err) => {
                        error!(
                            "{} worker-{} failed to build sst files, error: {:?}",
                            self.task_ctx.task_id, self.worker_id, err
                        );
                        cb(vec![]);
                        self.scheduler.cancel(format!(
                            "{} worker-{} error: {:?}",
                            self.task_ctx.task_id, self.worker_id, err
                        ));
                    }
                },
                BuildingWorkerMsg::Ingest { cb } => {
                    if let Err(err) = self.ingest() {
                        error!(
                            "{} worker-{} failed to ingest sst files, error: {:?}",
                            self.task_ctx.task_id, self.worker_id, err
                        );
                        self.scheduler.cancel(format!(
                            "{} worker-{} error: {:?}",
                            self.task_ctx.task_id, self.worker_id, err
                        ));
                    }
                    cb(());
                }
                BuildingWorkerMsg::Cleanup => {
                    return;
                }
            }
        }
    }

    fn ingest(&mut self) -> Result<()> {
        if self.scheduler.is_canceled() || self.scheduler.is_finished() {
            warn!(
                "{} worker-{} skip building sst, canceled: {}, finished: {}, error message: {}",
                self.task_ctx.task_id,
                self.worker_id,
                self.scheduler.is_canceled(),
                self.scheduler.is_finished(),
                self.scheduler.error_msg(),
            );
            return Err(Error::Canceled);
        }

        if self.ingested {
            return Ok(());
        }

        if self.sst_metas.is_empty() {
            info!(
                "{} worker-{} skip ingesting empty data",
                self.task_ctx.task_id, self.worker_id
            );
            let mut checkpoint_guard = self.checkpoint_store.lock().unwrap();
            checkpoint_guard.set_worker_ingested(self.worker_id)?;
            return Ok(());
        }

        let start = Instant::now();
        let mut data_size = 0;
        let mut total_kvs = 0;
        for sst_meta in &self.sst_metas {
            data_size += sst_meta.uncompressed_size;
            total_kvs += sst_meta.keys;
        }
        let sst_metas = mem::take(&mut self.sst_metas);
        self.ingest_sst(sst_metas)?;
        self.scheduler.add_total_kvs(total_kvs);

        let mut wru = 0.0;
        if let (Some(ru_config), Some(keyspace_id)) =
            (&self.config.rg_config, self.task_ctx.keyspace_id)
        {
            let request_unit = &ru_config.request_unit;
            // The calculation formula is a reference to the pd's ru consumption,
            // ref https://github.com/tikv/pd/blob/master/client/resource_group/controller/model.go#L103.
            //
            // `write_base_cost + write_per_batch_base_cost * avg_batch_proportion +
            // write_cost_per_byte * data_size * replica_nums`
            //
            // In the formula, we use the default value of `avg_batch_proportion` and
            // `replica_nums`, which are 0.5 (same as the default value of
            // `avg_batch_proportion` in pd) and 3.0 respectively.
            //
            // Set import billing same as txn file, though the underlying mechanisms are not
            // the same.
            wru = request_unit.write_base_cost
                + request_unit.write_per_batch_base_cost * DEFAULT_AVG_BATCH_PROPORTION
                + request_unit.write_cost_per_byte
                    * data_size as f64
                    * TXN_FILE_RU_DISCOUNT_RATIO
                    * REPLICA_NUMS;

            LOAD_DATA_WRU_COST_COUNTER
                .with_label_values(&[&keyspace_id.to_string(), &self.task_ctx.task_id])
                .inc_by(wru as u64);
        }

        let mut checkpoint_guard = self.checkpoint_store.lock().unwrap();
        checkpoint_guard.set_worker_ingested(self.worker_id)?;
        info!(
            "{} worker-{} finish ingesting, data size: {}, kvs: {}, ru consumption: {}, keyspace id: {:?}, takes {:?}",
            self.task_ctx.task_id,
            self.worker_id,
            data_size,
            total_kvs,
            wru,
            self.task_ctx.keyspace_id,
            start.saturating_elapsed(),
        );
        Ok(())
    }

    fn build(
        &mut self,
        file_metas: Vec<FileMeta>,
        start_key: Vec<u8>,
        end_key: Vec<u8>,
        compression_type: u8,
    ) -> Result<Vec<DuplicateEntry>> {
        if self.scheduler.is_canceled() || self.scheduler.is_finished() {
            warn!(
                "{} worker-{} skip building sst, canceled: {}, finished: {}, error message: {}",
                self.task_ctx.task_id,
                self.worker_id,
                self.scheduler.is_canceled(),
                self.scheduler.is_finished(),
                self.scheduler.error_msg(),
            );
            return Err(Error::Canceled);
        }
        if !self.sst_metas.is_empty() {
            return Ok(mem::take(&mut self.dup_entries));
        }
        if file_metas.is_empty() {
            info!(
                "{} worker-{} skip building empty data",
                self.task_ctx.task_id, self.worker_id
            );
            return Ok(vec![]);
        }

        info!(
            "{} worker-{} start to build sst with key common prefix {:?}",
            self.task_ctx.task_id, self.worker_id, self.key_comm_prefix
        );
        let start = Instant::now();
        let readers = build_readers(
            &self.task_ctx,
            file_metas,
            self.key_comm_prefix.len(),
            start_key,
            end_key,
        );
        let (tx, rx) = tikv_util::mpsc::unbounded();
        let mut sent_count = 0;
        let mut recv_count = 0;
        let mut merge_iter = MergeIterator::new(
            readers,
            &self.task_ctx.outer_key_prefix,
            self.key_comm_prefix.len(),
        )?;

        let mut errs = vec![];
        while merge_iter.valid() {
            let batch = self.read_batch(&mut merge_iter, self.config.sst_file_size)?;
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
                        self.sst_metas.push(sst_meta);
                    }
                }
            }
        }
        for _ in 0..(sent_count - recv_count) {
            match rx.recv().unwrap() {
                Err(err) => {
                    error!(
                        "{} worker-{} failed to create sst file, error: {}",
                        self.task_ctx.task_id, self.worker_id, err
                    );
                    errs.push(err);
                }
                Ok(sst_meta) => {
                    self.sst_metas.push(sst_meta);
                }
            }
        }
        if !errs.is_empty() {
            return Err(errs.pop().unwrap());
        }
        if !merge_iter.duplicated_entries.is_empty() {
            info!(
                "{} worker-{} got {} duplicated entries, size {}",
                self.task_ctx.task_id,
                self.worker_id,
                merge_iter.duplicated_entries.len(),
                merge_iter.duplicated_entries_size
            );
            self.dup_entries = merge_iter.duplicated_entries;
        }
        self.scheduler.add_created_files(self.sst_metas.len());

        let mut checkpoint_guard = self.checkpoint_store.lock().unwrap();
        checkpoint_guard.update_sst_metas(
            self.worker_id,
            self.sst_metas.clone(),
            self.dup_entries.clone(),
        )?;

        info!(
            "{} worker-{} finish building, sst metas: {}, duplicated entries: {}, takes {:?}",
            self.task_ctx.task_id,
            self.worker_id,
            self.sst_metas.len(),
            self.dup_entries.len(),
            start.saturating_elapsed(),
        );
        Ok(mem::take(&mut self.dup_entries))
    }

    fn read_batch(&mut self, merge_iter: &mut MergeIterator, batch_size: usize) -> Result<Vec<u8>> {
        let mut buf = Vec::with_capacity(batch_size);
        let mut pre_table_id = vec![];
        while merge_iter.valid() {
            let key = merge_iter.key();
            let key_len = key.len();
            let val = merge_iter.value();
            let val_len = val.len();

            if buf.len() + 2 + key_len + 4 + val_len > buf.capacity() {
                return Ok(buf);
            }
            let table_id = merge_iter.table_id();
            if !pre_table_id.is_empty() && pre_table_id.as_slice() != table_id {
                return Ok(buf);
            }
            pre_table_id = table_id.to_vec();

            buf.put_u16_le(key_len as u16);
            buf.extend_from_slice(key);
            buf.put_u32_le(val_len as u32);
            buf.extend_from_slice(val);
            merge_iter.next()?;
        }
        Ok(buf)
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
                    error!(
                        "{} worker-{} failed to allocate file id from PD {:?}",
                        self.task_ctx.task_id, self.worker_id, err
                    );
                    std::thread::sleep(Duration::from_secs(1));
                    if start.saturating_elapsed() > ALLOCATE_ID_TIMEOUT {
                        return Err(Error::PdError(err));
                    }
                }
            }
        }
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
        let worker_id = self.worker_id;
        let checksum_type = self.config.checksum_type;
        let encryption_key = self.task_ctx.encryption_key.clone();
        let um = UserMeta::new(self.task_ctx.start_ts, self.task_ctx.commit_ts);
        let mut val_buf = Value::encode_buf(0, &um.to_array(), self.task_ctx.commit_ts, &[]);
        let base_val_len = val_buf.len();
        let prepend_keyspace_id = self.task_ctx.prepend_keyspace_id;

        self.ctx.runtime.spawn(async move {
            info!(
                "{} worker-{} start to build sst file {}",
                task_id, worker_id, file_id
            );
            let mut builder = Builder::new(
                file_id,
                block_size,
                compression_type,
                ZSTD_COMPRESSION_LEVEL,
                checksum_type,
                encryption_key,
                prepend_keyspace_id,
            );
            let mut entries = 0;
            let mut offset = 0;
            let mut uncompressed_size = 0;
            while offset < batch.len() {
                let key_len = (&batch[offset..]).get_u16_le() as usize;
                offset += 2;
                let key = &batch[offset..offset + key_len];
                offset += key_len;
                uncompressed_size += key_len;
                let val_len = (&batch[offset..]).get_u32_le() as usize;
                offset += 4;
                let val = &batch[offset..offset + val_len];
                offset += val_len;
                uncompressed_size += base_val_len + val_len;
                val_buf.resize(base_val_len, 0);
                val_buf.extend_from_slice(val);

                // The key is already trimmed prefix, so we can use `from_inner_buf` here.
                let inner_key = InnerKey::from_inner_buf(key);
                builder.add(inner_key, &Value::decode(&val_buf), None);
                entries += 1;
            }
            batch.clear();
            let res = builder.finish(0, &mut batch);
            let data: Bytes = batch.into();
            let sst_meta = SstMeta {
                id: file_id,
                smallest: builder.get_smallest().to_vec(),
                biggest: builder.get_biggest().to_vec(),
                size: data.len(),
                meta_offset: res.meta_offset,
                uncompressed_size,
                keys: entries,
            };
            info!(
                "{} worker-{} finish building sst file {:?}",
                task_id, worker_id, sst_meta
            );
            let opts = Options::default();
            let res = ctx
                .dfs
                .create(file_id, data, opts)
                .await
                .map(|_| sst_meta.clone())
                .map_err(|e| Error::from(e));

            sender.send(res).unwrap();
        });
    }

    fn ingest_sst(&mut self, mut sst_metas: Vec<SstMeta>) -> Result<()> {
        if sst_metas.is_empty() {
            return Ok(());
        }
        info!(
            "{} worker-{} start to ingest, sst metas: {}",
            self.task_ctx.task_id,
            self.worker_id,
            sst_metas.len()
        );
        sst_metas.sort_by(|a, b| a.id.cmp(&b.id));
        let start = Instant::now();
        let outer_key_prefix = self.task_ctx.outer_key_prefix.to_vec();
        let coarse_split_keys = gen_split_keys(
            &outer_key_prefix,
            &sst_metas,
            self.config.coarse_split_size,
            true,
        );
        let new_regions_id = self.split_regions(&coarse_split_keys)?;
        let result = self.ctx.pd.scatter_regions_by_id(new_regions_id);
        if let Err(err) = result {
            error!(
                "{} worker-{} scatter regions failed {:?}",
                self.task_ctx.task_id, self.worker_id, err
            );
        }
        for i in 0..coarse_split_keys.len() - 1 {
            let mut encoded_start_key = coarse_split_keys[i].as_slice();
            let raw_start_key = decode_bytes(&mut encoded_start_key, false).unwrap();
            let mut encoded_end_key = coarse_split_keys[i + 1].as_slice();
            let raw_end_key = decode_bytes(&mut encoded_end_key, false).unwrap();
            let inner_start_key = InnerKey::from_outer_key(&raw_start_key);
            let inner_end_key = InnerKey::from_outer_end_key(&raw_end_key);
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
        info!(
            "{} worker-{} finish ingesting, takes {:?}",
            self.task_ctx.task_id,
            self.worker_id,
            start.saturating_elapsed()
        );
        Ok(())
    }

    fn split_regions(&self, split_keys: &[Vec<u8>]) -> Result<Vec<u64>> {
        info!(
            "{} worker-{} start split, keys {:?}",
            self.task_ctx.task_id, self.worker_id, split_keys
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
                Err(err) => {
                    error!(
                        "{} worker-{} split failed {:?}",
                        self.task_ctx.task_id, self.worker_id, err
                    );
                    if retry >= MAX_RETRY_TIMES {
                        return Err(Error::PdError(err));
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
            "{} worker-{} finish split, new regions_id {:?}",
            self.task_ctx.task_id, self.worker_id, new_regions_id
        );
        Ok(new_regions_id)
    }

    fn ingest_group(&self, sst_metas: Vec<SstMeta>) -> Result<()> {
        let outer_key_prefix = self.task_ctx.outer_key_prefix.to_vec();
        let split_keys = gen_split_keys(
            &outer_key_prefix,
            &sst_metas,
            self.config.region_size,
            false,
        );
        if !split_keys.is_empty() {
            self.split_regions(&split_keys)?;
        }
        let outer_first_key = new_region_key(
            &outer_key_prefix,
            sst_metas.first().unwrap().smallest.as_slice(),
        );
        let outer_last_key = {
            let mut last_key = sst_metas.last().unwrap().biggest.clone();
            last_key.push(0);
            new_region_key(&outer_key_prefix, &last_key)
        };

        let mut success_ranges = MergeRanges::default(); // keys of `success_ranges` are encoded.
        let mut last_error: Option<Error> = None;
        for retry in 0..MAX_RETRY_TIMES {
            match self.ingest_group_to_range(
                &outer_key_prefix,
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
                        "{} worker-{} ingest_group_to_range failed {:?}, retry {}",
                        self.task_ctx.task_id, self.worker_id, err, retry
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

    fn ingest_group_to_range(
        &self,
        outer_key_prefix: &[u8],
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
        if !success_ranges.is_empty() {
            regions.retain(|region| {
                !success_ranges
                    .covered(&region.get_region().start_key, &region.get_region().end_key)
            });
        }
        info!(
            "{} worker-{} scanned and filtered regions {:?}",
            self.task_ctx.task_id, self.worker_id, regions
        );

        let mut errors = vec![];
        let mut handle_ingest_res = |res: Result<metapb::Region>| match res {
            Ok(region) => {
                let success_start = max(region.get_start_key(), outer_first_key.as_slice());
                let success_end = min(region.get_end_key(), outer_last_key.as_slice());
                success_ranges.insert(success_start.to_vec(), success_end.to_vec());
                self.scheduler.add_ingested_regions();
            }
            Err(err) => errors.push(err),
        };

        let (tx, rx) = tikv_util::mpsc::unbounded();
        let mut msg_cnt = 0;
        for mut pd_region in regions {
            let region = pd_region.get_region();
            let cs = build_ingest_files(
                outer_key_prefix.len(),
                region,
                sst_metas,
                self.task_ctx.commit_ts,
            );
            if cs.get_ingest_files().get_table_creates().is_empty() {
                continue;
            }
            if self.scheduler.is_canceled() {
                return Err(Error::Canceled);
            }
            let pd_cli = self.ctx.pd.clone();
            let task_id = self.task_ctx.task_id.clone();
            let worker_id = self.worker_id;
            let tx = tx.clone();
            self.ctx.runtime.spawn(async move {
                let region = pd_region.take_region();
                let leader = pd_region.take_leader();
                info!(
                    "{} worker-{} ingest_group_to_range: region: {:?}, leader: {:?}, cs: {:?}",
                    task_id, worker_id, region, leader, cs
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
}

fn build_readers(
    task_ctx: &TaskContext,
    file_metas: Vec<FileMeta>,
    key_comm_prefix: usize,
    lower_bound: Vec<u8>,
    upper_bound: Vec<u8>,
) -> Vec<KvPairsReader> {
    let table_prefix_offset = KEYSPACE_PREFIX_LEN - task_ctx.inner_key_off.unwrap();
    let mut readers = Vec::with_capacity(file_metas.len());
    for file_meta in file_metas {
        let file = File::open(file_meta.file_path).unwrap();
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
        let decrypter_reader = DecrypterReader::new(file, method, key, iv).unwrap();

        let lower_bound_suffix = if !lower_bound.is_empty() && lower_bound > file_meta.first_key {
            lower_bound.as_slice()[key_comm_prefix..].to_vec()
        } else {
            vec![]
        };
        let upper_bound_suffix = if !upper_bound.is_empty() && upper_bound <= file_meta.last_key {
            upper_bound.as_slice()[key_comm_prefix..].to_vec()
        } else {
            vec![]
        };

        let reader = KvPairsReader::new(
            file_meta.kv_count,
            decrypter_reader,
            lower_bound_suffix,
            upper_bound_suffix,
            table_prefix_offset,
        );
        readers.push(reader);
    }
    readers
}
