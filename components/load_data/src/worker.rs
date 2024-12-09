// Copyright 2024 TiKV Project Authors. Licensed under Apache-2.0.
//
use std::{
    cmp::Ordering,
    collections::HashMap,
    fs,
    fs::File,
    io::Write,
    mem,
    path::PathBuf,
    sync::{Arc, Mutex},
};

use bytes::{Buf, BufMut, Bytes};
use encryption::{DecrypterReader, EncrypterWriter, Iv};
use kvproto::encryptionpb::EncryptionMethod;
use rfengine::compress_lz4;
use tikv_util::{
    error, info,
    mpsc::{Receiver, Sender},
    time::Instant,
    warn,
};

use crate::{
    checkpoint::{FileMeta, LocalFileCheckpointStorage},
    error::{Error, Result},
    kv::{DuplicateEntry, KvPair, KvPairsReader, MergeIterator},
    task::{
        get_common_prefix, FlushResult, FlushStates, LoadDataConfig, LoadDataContext,
        LoadTaskScheduler, PutChunkResult, TaskContext, FLUSH_FILE_CONCURRENCY,
    },
};

#[allow(dead_code)]
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

#[allow(dead_code)]
pub struct UnhandledFlushFile {
    pub handled_chunk_ids: HashMap<u64 /* writer_id */, u64 /* chunk_id */>,
    pub file_idx: usize,
    pub key_comm_prefix: Vec<u8>,
    pub file_meta: FileMeta,
}

#[allow(dead_code)]
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

#[allow(dead_code)]
impl KvPairsWorker {
    pub fn new(
        worker_id: u64,
        config: LoadDataConfig,
        ctx: LoadDataContext,
        task_ctx: TaskContext,
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
        let l0_data_dir = ctx
            .dir
            .join(task_ctx.task_id.as_str())
            .join(format!("worker-{}", worker_id))
            .join("l0-files");
        let l1_data_dir = ctx
            .dir
            .join(task_ctx.task_id.as_str())
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
                    if let Err(err) = fs::remove_dir_all(&self.l0_data_dir) {
                        error!(
                            "{} worker-{} failed to remove data dir {:?}, error {:?}",
                            self.task_ctx.task_id, self.worker_id, self.l0_data_dir, err
                        );
                    }
                    if let Err(err) = fs::remove_dir_all(&self.l1_data_dir) {
                        error!(
                            "{} worker-{} failed to remove data dir {:?}, error {:?}",
                            self.task_ctx.task_id, self.worker_id, self.l1_data_dir, err
                        );
                    }
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
            return Ok(());
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

        let mut checkpoint_guard = self.checkpoint.lock().unwrap();
        checkpoint_guard.update_l0_flushed_info(
            self.worker_id,
            handled_chunk_ids.clone(),
            file_metas.clone(),
            last_key_comm_prefix.clone(),
        )?;
        self.scheduler.add_flushed_files(need_handled);

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
            return Ok((vec![], vec![]));
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

        let task_ctx = self.task_ctx.clone();
        let file_metas = mem::take(&mut self.l0_file_metas);
        let readers = build_readers(task_ctx, file_metas);
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

        let mut checkpoint_guard = self.checkpoint.lock().unwrap();
        checkpoint_guard.update_l1_flushed_info(
            self.worker_id,
            self.l1_file_metas.clone(),
            self.dup_entries.clone(),
        )?;
        self.scheduler.add_flushed_files(self.l1_file_metas.len());
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

#[allow(dead_code)]
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

#[allow(dead_code)]
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

#[allow(dead_code)]
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

#[allow(dead_code)]
fn build_readers(task_ctx: TaskContext, file_metas: Vec<FileMeta>) -> Vec<KvPairsReader> {
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
        let reader = KvPairsReader::new(file_meta.kv_count, decrypter_reader);
        readers.push(reader);
    }
    readers
}
