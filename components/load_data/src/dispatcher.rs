// Copyright 2024 TiKV Project Authors. Licensed under Apache-2.0.
use std::{
    collections::{HashMap, HashSet},
    sync::{Arc, Mutex, RwLock},
};

use api_version::ApiV2;
use bytes::{Buf, Bytes};
use kvengine::{get_shard_property, ENCRYPTION_KEY};
use tikv_util::{
    error, info,
    mpsc::{Receiver, Sender},
    time::Instant,
};

use crate::{
    checkpoint::{
        FileMeta, LoadDataCheckpointCtx, LoadDataWorkerState, LocalFileCheckpointStorage,
    },
    error::Result,
    kv::DuplicateEntry,
    metrics::remove_metrics,
    task::{
        get_common_prefix, get_shard_meta, FlushResult, FlushStates, LoadDataConfig,
        LoadDataContext, LoadTaskMsg, LoadTaskScheduler, LoadTaskStates, PutChunkResult,
        TaskContext, WritersStates, GET_SHARD_META_TIMEOUT,
    },
    worker::{BuildingWorker, BuildingWorkerMsg, KvPairsWorker, KvPairsWorkerMsg},
};

pub struct Dispatcher {
    config: LoadDataConfig,
    ctx: LoadDataContext,
    task_ctx: TaskContext,
    kvpairs_worker_num: u64,
    building_worker_num: u64,
    scheduler: LoadTaskScheduler,
    receiver: Receiver<LoadTaskMsg>,
    kvpairs_worker_senders: HashMap<u64, Sender<KvPairsWorkerMsg>>,
    building_worker_senders: HashMap<u64, Sender<BuildingWorkerMsg>>,
    writer2worker: HashMap<u64, u64>,
    next_worker: u64,
    checkpoint_store: Arc<Mutex<LocalFileCheckpointStorage>>,
}

impl Dispatcher {
    pub fn new(
        config: LoadDataConfig,
        ctx: LoadDataContext,
        task_ctx: TaskContext,
        mut checkpoint_ctx: LoadDataCheckpointCtx,
    ) -> Self {
        let kvpairs_worker_nums = config.kvpairs_worker_num as u64;
        let building_worker_nums = config.building_worker_num as u64;

        let mut writer2worker = HashMap::default();
        let mut flushed_files = 0;
        let mut created_files = 0;
        let mut total_kvs = 0;

        // recover `writer2worker` from checkpoint to ensure that the mapping from
        // `writer` to `worker` is consistent across restart.
        for worker_id in 0..kvpairs_worker_nums {
            let worker_ctx = checkpoint_ctx.get_kvpairs_worker_ctx(worker_id);
            for writer_id in worker_ctx.flushed_chunk_ids.keys() {
                writer2worker.insert(*writer_id, worker_id);
            }
            flushed_files += worker_ctx.l0_file_metas.len() + worker_ctx.l1_file_metas.len();
        }

        for worker_id in 0..building_worker_nums {
            let worker_ctx = checkpoint_ctx.get_building_worker_ctx(worker_id);
            if worker_ctx.ingested {
                for sst_meta in &worker_ctx.sst_metas {
                    total_kvs += sst_meta.keys
                }
            }
            created_files += worker_ctx.sst_metas.len();
        }
        let finished = checkpoint_ctx.get_state() == LoadDataWorkerState::IngestedSst;
        let duplicated_entries = checkpoint_ctx.get_duplicated_entries();

        let states = LoadTaskStates {
            task_id: task_ctx.task_id.clone(),
            canceled: checkpoint_ctx.canceled,
            finished,
            error: checkpoint_ctx.error.clone(),
            flushed_files,
            created_files,
            total_kvs,
            ingested_regions: 0,
            duplicated_entries,
        };

        let mut checkpoint_store =
            LocalFileCheckpointStorage::new(checkpoint_ctx, ctx.dir.clone()).unwrap();
        if !checkpoint_store.get_is_recover() {
            checkpoint_store.flush_checkpoint_ctx().unwrap();
        }
        let checkpoint_store = Arc::new(Mutex::new(checkpoint_store));

        let (sender, receiver) = tikv_util::mpsc::unbounded();
        let scheduler = LoadTaskScheduler {
            sender,
            states: Arc::new(RwLock::new(states)),
            writers: Arc::new(Mutex::new(WritersStates::default())),
            checkpoint_store: checkpoint_store.clone(),
            thread_handle: None,
        };

        Self {
            config,
            ctx,
            task_ctx,
            kvpairs_worker_num: kvpairs_worker_nums,
            building_worker_num: building_worker_nums,
            scheduler,
            writer2worker,
            next_worker: 0,
            receiver,
            kvpairs_worker_senders: HashMap::default(),
            building_worker_senders: HashMap::default(),
            checkpoint_store,
        }
    }

    pub fn run(&mut self) {
        self.init();
        while let Ok(msg) = self.receiver.recv() {
            match msg {
                LoadTaskMsg::AddChunk {
                    writer_id,
                    chunk_id,
                    chunk_data,
                    cb,
                } => {
                    if self.scheduler.is_canceled() || self.scheduler.is_finished() {
                        let result = PutChunkResult {
                            handled_chunk_id: 0,
                            flushed_chunk_id: 0,
                            canceled: self.scheduler.is_canceled(),
                            finished: self.scheduler.is_finished(),
                            error: self.scheduler.error_msg(),
                        };
                        info!(
                            "{} dispatcher skip chunk, canceled: {}, finished: {}, error message: {}",
                            self.task_ctx.task_id, result.canceled, result.finished, result.error,
                        );
                        cb(result);
                        continue;
                    }
                    if self.task_ctx.inner_key_off.is_none() {
                        if let Err(err) = self.init_inner_key_from_chunk(&chunk_data) {
                            error!(
                                "{} dispatcher failed to set inner key off, error: {:?}",
                                self.task_ctx.task_id, err
                            );
                            self.scheduler.cancel(format!(
                                "{} dispatcher error: {:?}",
                                self.task_ctx.task_id, err
                            ));
                            cb(PutChunkResult {
                                handled_chunk_id: 0,
                                flushed_chunk_id: 0,
                                canceled: self.scheduler.is_canceled(),
                                finished: self.scheduler.is_finished(),
                                error: self.scheduler.error_msg(),
                            });
                            continue;
                        }
                    }

                    let worker_id = self.pick_kvpairs_worker(writer_id);
                    let sender = self.get_kvpairs_worker_sender(worker_id);
                    sender
                        .send(KvPairsWorkerMsg::AddChunk {
                            writer_id,
                            chunk_id,
                            chunk_data,
                            cb,
                        })
                        .unwrap();
                }
                LoadTaskMsg::Build { compression_type } => {
                    if self.scheduler.is_canceled() || self.scheduler.is_finished() {
                        continue;
                    }
                    self.build(compression_type);
                }
                LoadTaskMsg::Flush {
                    writer_id,
                    flush_file_count,
                    cb,
                } => {
                    if self.scheduler.is_canceled() || self.scheduler.is_finished() {
                        let flush_result = FlushResult {
                            flushed_chunk_ids: HashMap::default(),
                            canceled: self.scheduler.is_canceled(),
                            finished: self.scheduler.is_finished(),
                            error: self.scheduler.error_msg(),
                        };
                        info!(
                            "{} dispatcher skip flushing, canceled: {}, finished: {}, error message: {}",
                            self.task_ctx.task_id,
                            flush_result.canceled,
                            flush_result.finished,
                            flush_result.error,
                        );
                        cb(FlushStates::FlushResult { flush_result });
                        continue;
                    }

                    let worker_id = self.pick_kvpairs_worker(writer_id);
                    let sender = self.get_kvpairs_worker_sender(worker_id);
                    sender
                        .send(KvPairsWorkerMsg::Flush {
                            writer_id,
                            flush_file_count,
                            cb,
                        })
                        .unwrap();
                }
                LoadTaskMsg::Cleanup => {
                    info!("{} dispatcher cleanup", self.task_ctx.task_id);
                    for sender in self.kvpairs_worker_senders.values() {
                        sender.send(KvPairsWorkerMsg::Cleanup).unwrap();
                    }
                    for sender in self.building_worker_senders.values() {
                        sender.send(BuildingWorkerMsg::Cleanup).unwrap();
                    }
                    self.checkpoint_store
                        .lock()
                        .unwrap()
                        .clean_checkpoint_data();

                    let keyspace_id = self
                        .task_ctx
                        .keyspace_id
                        .map(|keyspace_id| keyspace_id.to_string());
                    std::thread::sleep(self.config.metrics_gather_interval);
                    remove_metrics(&self.task_ctx.task_id, keyspace_id);
                    return;
                }
            }
        }
    }

    fn init(&mut self) {
        let checkpoint_guard = self.checkpoint_store.lock().unwrap();
        let first_key = checkpoint_guard.checkpoint_ctx.get_first_key();
        drop(checkpoint_guard);
        if !first_key.is_empty() {
            if let Err(err) = self.set_inner_key_off_and_encryption_key(&first_key) {
                error!(
                    "{} dispatcher failed to set inner key off, error: {:?}",
                    self.task_ctx.task_id, err
                );
                self.scheduler.cancel(format!(
                    "{} dispatcher error: {:?}",
                    self.task_ctx.task_id, err
                ));
            }
        }
    }

    fn pick_kvpairs_worker(&mut self, writer_id: u64) -> u64 {
        let worker = self.writer2worker.entry(writer_id).or_insert_with(|| {
            let worker = self.next_worker;
            self.next_worker = (worker + 1) % self.kvpairs_worker_num;
            worker
        });
        *worker
    }

    fn get_kvpairs_worker_sender(&mut self, worker_id: u64) -> &Sender<KvPairsWorkerMsg> {
        let sender = self
            .kvpairs_worker_senders
            .entry(worker_id)
            .or_insert_with(|| {
                let (sender, receiver) = tikv_util::mpsc::unbounded();
                let mut worker = KvPairsWorker::new(
                    worker_id,
                    self.config.clone(),
                    self.ctx.clone(),
                    self.task_ctx.clone(),
                    receiver,
                    self.scheduler.clone(),
                    self.checkpoint_store.clone(),
                );
                std::thread::spawn(move || {
                    worker.run();
                });
                sender
            });
        sender
    }

    pub fn get_scheduler(&self) -> LoadTaskScheduler {
        self.scheduler.clone()
    }

    fn init_inner_key_from_chunk(&mut self, chunk_data: &Bytes) -> Result<()> {
        let key_len = (&chunk_data[0..]).get_u16_le();
        let first_key = chunk_data.slice(2..2 + key_len as usize);
        self.set_inner_key_off_and_encryption_key(&first_key)?;

        let mut checkpoint_guard = self.checkpoint_store.lock().unwrap();
        checkpoint_guard.update_first_key(first_key)
    }

    fn set_inner_key_off_and_encryption_key(&mut self, first_key: &Bytes) -> Result<()> {
        if self.task_ctx.inner_key_off.is_none() {
            self.task_ctx.keyspace_id = ApiV2::get_u32_keyspace_id_by_key(first_key.chunk());
            let shard_meta = self.ctx.runtime.block_on(get_shard_meta(
                self.ctx.pd.clone(),
                first_key.chunk(),
                GET_SHARD_META_TIMEOUT,
            ))?;
            let snapshot = shard_meta.get_snapshot();
            let mut prepend_keyspace_id = None;
            if snapshot.inner_key_off == 0 {
                prepend_keyspace_id = ApiV2::get_u32_keyspace_id_by_key(&snapshot.outer_start)
            }
            self.task_ctx.inner_key_off = Some(snapshot.inner_key_off as usize);
            self.task_ctx.outer_key_prefix =
                first_key.slice(..snapshot.inner_key_off as usize).to_vec();
            self.task_ctx.encryption_key =
                get_shard_property(ENCRYPTION_KEY, snapshot.get_properties()).map(|exported_key| {
                    self.ctx
                        .master_key
                        .decrypt_encryption_key(&exported_key)
                        .unwrap()
                });
            self.task_ctx.prepend_keyspace_id = prepend_keyspace_id;
        }
        Ok(())
    }

    fn collect_file_metas(&mut self) -> (Vec<FileMeta>, Vec<DuplicateEntry>, Vec<u8>) {
        let kvpairs_worker_ids: Vec<u64> = self
            .writer2worker
            .values()
            .cloned()
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();

        // `kvpairs_worker_ids` is empty means no data
        if kvpairs_worker_ids.is_empty() {
            return (vec![], vec![], vec![]);
        }
        let mut futs = Vec::with_capacity(kvpairs_worker_ids.len());
        let skip_sort = self.kvpairs_worker_num == 1;
        for kvpairs_worker_id in kvpairs_worker_ids {
            let (cb, fut) = tikv_util::future::paired_future_callback();
            futs.push(fut);
            let sender = self.get_kvpairs_worker_sender(kvpairs_worker_id);
            sender
                .send(KvPairsWorkerMsg::CollectFileMetas { skip_sort, cb })
                .unwrap();
        }

        let mut file_metas = vec![];
        let mut dup_entries = vec![];
        let checkpoint_guard = self.checkpoint_store.lock().unwrap();
        let first_key = checkpoint_guard.checkpoint_ctx.get_first_key();
        let mut key_comm_prefix = first_key
            .slice(self.task_ctx.inner_key_off.unwrap()..)
            .to_vec();
        drop(checkpoint_guard);

        for fut in futs {
            let mut result = self.ctx.runtime.block_on(fut).unwrap();
            file_metas.append(&mut result.0);
            dup_entries.append(&mut result.1);
            key_comm_prefix = get_common_prefix(&key_comm_prefix, &result.2);
        }

        (file_metas, dup_entries, key_comm_prefix)
    }

    fn build(&mut self, compression_type: u8) {
        let mut checkpoint_guard = self.checkpoint_store.lock().unwrap();
        if checkpoint_guard.get_state() < LoadDataWorkerState::BuildingSst {
            if let Err(err) = checkpoint_guard.update_build_msg(compression_type) {
                error!(
                    "{} dispatcher failed to update build msg: {:?}",
                    self.task_ctx.task_id, err
                );
                self.scheduler.cancel(format!(
                    "{} dispatcher error: {:?}",
                    self.task_ctx.task_id, err
                ));
                return;
            }
        }
        drop(checkpoint_guard);

        let start = Instant::now();
        let (file_metas, mut dup_entries0, key_comm_prefix) = self.collect_file_metas();
        if self.scheduler.is_canceled() {
            info!(
                "{} is canceled, error message: {}",
                self.task_ctx.task_id,
                self.scheduler.error_msg(),
            );
            return;
        }

        let mut dup_entries1 = self.build_sst(file_metas, key_comm_prefix, compression_type);
        if self.scheduler.is_canceled() {
            info!(
                "{} is canceled, error message: {}",
                self.task_ctx.task_id,
                self.scheduler.error_msg(),
            );
            return;
        }

        self.ingest_sst();
        if self.scheduler.is_canceled() {
            info!(
                "{} is canceled, error message: {}",
                self.task_ctx.task_id,
                self.scheduler.error_msg(),
            );
            return;
        }

        // merge duplicated entries
        dup_entries0.append(&mut dup_entries1);
        dup_entries0.sort_by(|a, b| b.key.cmp(&a.key));
        let mut dup_entries = Vec::with_capacity(dup_entries0.len());
        while let Some(mut dup_entry) = dup_entries0.pop() {
            while !dup_entries0.is_empty() && dup_entry.key == dup_entries0.last().unwrap().key {
                dup_entry
                    .values
                    .append(&mut dup_entries0.pop().unwrap().values);
            }
            dup_entries.push(dup_entry);
        }

        let mut checkpoint_guard = self.checkpoint_store.lock().unwrap();
        if let Err(err) = checkpoint_guard.set_ingested(dup_entries.clone()) {
            error!(
                "{} dispatcher failed to update checkpoint state, error: {:?}",
                self.task_ctx.task_id, err
            );
        }
        info!(
            "{} dispatcher finish building, duplicated entries: {}, takes {:?}",
            self.task_ctx.task_id,
            dup_entries.len(),
            start.saturating_elapsed(),
        );
        self.scheduler.set_finished(dup_entries);
    }

    fn build_sst(
        &mut self,
        file_metas: Vec<FileMeta>,
        key_comm_prefix: Vec<u8>,
        compression_type: u8,
    ) -> Vec<DuplicateEntry> {
        if file_metas.is_empty() {
            return vec![];
        }

        let ranges_splitter = RangesSplitter::new(file_metas, self.building_worker_num as usize);
        let ranges_groups = ranges_splitter.split_ranges_groups();
        let mut futs = Vec::with_capacity(ranges_groups.len());
        for (worker_id, ranges_group) in ranges_groups.into_iter().enumerate() {
            let worker_id = worker_id as u64;
            let (start_key, end_key, file_metas) = (
                ranges_group.start_key,
                ranges_group.end_key,
                ranges_group.ranges,
            );
            info!(
                "{} dispatcher splits one group for worker-{}, start_key: {:?}, end_key: {:?}, file metas: {}",
                self.task_ctx.task_id,
                worker_id,
                start_key,
                end_key,
                file_metas.len()
            );
            let (cb, fut) = tikv_util::future::paired_future_callback();
            futs.push(fut);
            let sender = self
                .building_worker_senders
                .entry(worker_id)
                .or_insert_with(|| {
                    let (sender, receiver) = tikv_util::mpsc::unbounded();
                    let mut worker = BuildingWorker::new(
                        worker_id,
                        self.config.clone(),
                        self.ctx.clone(),
                        self.task_ctx.clone(),
                        key_comm_prefix.clone(),
                        receiver,
                        self.scheduler.clone(),
                        self.checkpoint_store.clone(),
                    );
                    std::thread::spawn(move || {
                        worker.run();
                    });
                    sender
                });

            sender
                .send(BuildingWorkerMsg::Build {
                    start_key,
                    end_key,
                    file_metas,
                    compression_type,
                    cb,
                })
                .unwrap();
        }

        let mut dup_entries = vec![];
        for fut in futs {
            let mut cur_dup_entries = self.ctx.runtime.block_on(fut).unwrap();
            dup_entries.append(&mut cur_dup_entries);
        }
        dup_entries
    }

    fn ingest_sst(&mut self) {
        let mut futs = Vec::with_capacity(self.building_worker_num as usize);
        for sender in self.building_worker_senders.values() {
            let (cb, fut) = tikv_util::future::paired_future_callback();
            futs.push(fut);
            sender.send(BuildingWorkerMsg::Ingest { cb }).unwrap();
        }

        for fut in futs {
            self.ctx.runtime.block_on(fut).unwrap();
        }
    }
}

#[derive(Debug)]
struct RangesGroup {
    ranges: Vec<FileMeta>,
    start_key: Vec<u8>,
    end_key: Vec<u8>,
}

struct RangesSplitter {
    group_num: usize,
    file_metas: Vec<FileMeta>,
}

impl RangesSplitter {
    fn new(mut file_metas: Vec<FileMeta>, group_num: usize) -> RangesSplitter {
        file_metas.sort_by(|a, b| a.first_key.cmp(&b.first_key));
        Self {
            group_num,
            file_metas,
        }
    }

    fn split_ranges_groups(&self) -> Vec<RangesGroup> {
        let mut group_len = self.file_metas.len() / self.group_num;
        if self.file_metas.len() % self.group_num != 0 {
            group_len += 1;
        }

        let mut ranges_groups: Vec<RangesGroup> = Vec::with_capacity(self.group_num);
        for file_meta in self.file_metas.iter().step_by(group_len) {
            if let Some(last) = ranges_groups.last_mut() {
                if last.start_key == file_meta.first_key {
                    continue;
                }
                last.end_key = file_meta.first_key.clone();
            }
            ranges_groups.push(RangesGroup {
                ranges: vec![],
                start_key: file_meta.first_key.clone(),
                end_key: vec![],
            });
        }

        for file_meta in &self.file_metas {
            for group in &mut ranges_groups {
                if file_meta.last_key < group.start_key
                    || (!group.end_key.is_empty() && file_meta.first_key >= group.end_key)
                {
                    continue;
                }
                group.ranges.push(file_meta.clone());
            }
        }

        ranges_groups
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::HashSet, mem, path::PathBuf};

    use proptest::prelude::*;
    use rand::prelude::*;
    use tidb_query_datatype::codec::table;

    use super::*;
    use crate::checkpoint::FileMeta;

    prop_compose! {
        fn arb_file_metas(min_file_meta_num: usize, max_file_meta_num: usize)
            (file_meta_num in min_file_meta_num..=max_file_meta_num)
            -> Vec<FileMeta> {
                let mut file_metas = Vec::with_capacity(file_meta_num);
                for i in 0..file_meta_num {
                    let mut first_key = get_random_key();
                    let mut last_key = get_random_key();
                    if first_key > last_key {
                        mem::swap(&mut first_key, &mut last_key);
                    }
                    file_metas.push(
                        FileMeta {
                            file_path: PathBuf::from(format!("path{}", i)),
                            kv_count: 10,
                            kv_size: 200,
                            first_key,
                            last_key,
                        }
                    );
                }
                file_metas
        }
    }

    #[test]
    fn test_range_splitter() {
        proptest!(|(
            file_metas in arb_file_metas(1, 100)
        )| {
            let mut rng = thread_rng();
            let group_num: usize = rng.gen_range(4..16);
            let splitter = RangesSplitter::new(file_metas.clone(), group_num);
            let groups = splitter.split_ranges_groups();

            prop_assert!(groups.len() <= group_num);
            let mut prev_end_key = groups.first().unwrap().start_key.clone();
            let mut all_file_metas = HashSet::new();
            for group in &groups {
                if !group.end_key.is_empty() {
                    prop_assert!(group.start_key < group.end_key);
                }
                prop_assert_eq!(group.start_key.as_slice(), prev_end_key.as_slice());
                prev_end_key = group.end_key.clone();

                let mut group_file_metas = HashSet::new();
                // The file meta in the group should overlap with `[group.start_key,
                // group.end_key)`.
                for file_meta in &group.ranges {
                    prop_assert!(file_meta.last_key >= group.start_key);
                    if !group.end_key.is_empty() {
                        prop_assert!(file_meta.first_key < group.end_key);
                    }

                    all_file_metas.insert(file_meta.file_path.to_str().unwrap().to_string());
                    group_file_metas.insert(file_meta.file_path.to_str().unwrap().to_string());
                }
                prop_assert_eq!(group_file_metas.len(), group.ranges.len());

                // The file meta that overlaps with `[group.start_key, group.end_key)` should be
                // in group.
                for file_meta in &file_metas {
                    if (!group.end_key.is_empty() && file_meta.first_key >= group.end_key) ||
                        (file_meta.last_key < group.start_key) {
                            continue;
                    }
                    prop_assert!(group_file_metas.contains(file_meta.file_path.to_str().unwrap()));
                }
            }
            prop_assert_eq!(all_file_metas.len(), file_metas.len());
            prop_assert!(prev_end_key.is_empty());
        });
    }

    fn get_random_key() -> Vec<u8> {
        let mut rng = thread_rng();
        let i: i64 = rng.gen_range(0..50);
        table::encode_row_key(1, i)
    }
}
