// Copyright 2023 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    collections::HashMap,
    fs,
    fs::OpenOptions,
    io::Write,
    path::PathBuf,
    sync::{Arc, Mutex},
    time::Duration,
};

use bytes::Bytes;
use chrono::Utc;
use dashmap::DashMap;
use serde_derive::{Deserialize, Serialize};
use tikv_client::Value;
use tikv_util::{debug, error, info};

use crate::{
    kv::{DuplicateEntry, SstMeta},
    metrics::LOAD_DATA_TASK_STATE,
    task::{LoadTaskMsg, LoadTaskScheduler, LoadTaskStates, TaskContext},
    Error,
};

pub type Result<T> = std::result::Result<T, Error>;
pub const CHECKPOINT_WORKER_PREFIX: &str = "LOAD_DATA_CHECK_POINT_";

pub const CANCELLED_TASK_EXPIRE_SEC: i64 = 3 * 60 * 60; // 3h
pub const IDLE_TASK_EXPIRE_SEC: i64 = 3 * 24 * 60 * 60; // 3d
pub const CLEANUP_INTERVAL_SEC: u64 = 10 * 60; // 10m

lazy_static::lazy_static! {
    static ref FILE_LOCK: Mutex<()> = Mutex::new(());
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, PartialOrd, Eq, Ord)]
pub enum LoadDataWorkerState {
    InitTask = 0,
    AddingChunks = 20,
    BuildingSst = 60,
    IngestingSst = 90,
    IngestedSst = 100,
}

impl Default for LoadDataWorkerState {
    fn default() -> Self {
        Self::InitTask
    }
}

impl LoadDataWorkerState {
    pub fn transition(&mut self, new_state: LoadDataWorkerState) -> bool {
        if !self.check_state(new_state) {
            return false;
        }

        *self = new_state;
        true
    }

    pub fn check_state(&self, new_state: LoadDataWorkerState) -> bool {
        if *self == new_state {
            return true;
        }

        match self {
            LoadDataWorkerState::InitTask => {
                if new_state == LoadDataWorkerState::AddingChunks
                    || new_state == LoadDataWorkerState::BuildingSst
                {
                    return true;
                }
            }
            LoadDataWorkerState::AddingChunks => {
                if new_state == LoadDataWorkerState::BuildingSst {
                    return true;
                }
            }
            LoadDataWorkerState::BuildingSst => {
                if new_state == LoadDataWorkerState::IngestingSst
                    || new_state == LoadDataWorkerState::IngestedSst
                {
                    return true;
                }
            }
            LoadDataWorkerState::IngestingSst => {
                if new_state == LoadDataWorkerState::IngestedSst {
                    return true;
                }
            }
            LoadDataWorkerState::IngestedSst => {
                if new_state == LoadDataWorkerState::IngestedSst {
                    return true;
                }
            }
        }
        false
    }

    pub(crate) fn as_str(&self) -> &str {
        match *self {
            LoadDataWorkerState::InitTask => "InitTask",
            LoadDataWorkerState::AddingChunks => "AddingChunks",
            LoadDataWorkerState::BuildingSst => "BuildingSst",
            LoadDataWorkerState::IngestingSst => "IngestingSst",
            LoadDataWorkerState::IngestedSst => "IngestedSst",
        }
    }
}

#[derive(Clone, Serialize, Deserialize, Debug, Default)]
#[serde(default)]
pub struct FileMeta {
    pub file_path: PathBuf,
    pub kv_count: usize,
    pub kv_size: usize,
    pub first_key: Vec<u8>,
    pub last_key: Vec<u8>,
}

#[derive(Clone, Serialize, Deserialize, Debug, Default)]
#[serde(default)]
pub struct KvPairsWorkerCtx {
    pub key_comm_prefix: Vec<u8>,
    pub flushed_chunk_ids: HashMap<u64, u64>,
    pub l0_file_metas: Vec<FileMeta>,
    pub l1_file_metas: Vec<FileMeta>,
    pub duplicated_entries: Vec<DuplicateEntry>,
}

#[derive(Clone, Serialize, Deserialize, Debug, Default)]
#[serde(default)]
pub struct BuildingWorkerCtx {
    pub sst_metas: Vec<SstMeta>,
    pub duplicated_entries: Vec<DuplicateEntry>,
    pub ingested: bool,
}

#[derive(Clone, Serialize, Deserialize, Debug, Default)]
#[serde(default)]
pub struct LoadDataCheckpointCtx {
    // TaskContext
    pub task_id: String,
    start_ts: u64,
    commit_ts: u64,
    first_key: Bytes,

    // LoadDataWorker
    local_file_infos: Vec<LocalFileInfo>,
    sst_metas: Vec<SstMeta>,
    flushed_chunk_ids: HashMap<u64 /* writer_id */, u64 /* chunk_id */>,
    flushed_file_idx: usize,
    key_comm_prefix: Vec<u8>,

    // KVPairsWorker & BuildingWorker
    kvpairs_workers_ctx: HashMap<u64 /* worker_id */, KvPairsWorkerCtx>,
    building_workers_ctx: HashMap<u64 /* worker_id */, BuildingWorkerCtx>,

    // common
    compression: u8,
    state: LoadDataWorkerState,
    duplicated_entries: Vec<DuplicateEntry>,
    is_recover: bool,
    pub canceled: bool,
    pub error: String,
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct LocalFileInfo {
    pub path: PathBuf,
    pub kv_count: usize,
}

impl LoadDataCheckpointCtx {
    pub fn new(task_ctx: TaskContext) -> Self {
        let now = Utc::now();
        let millis = now.timestamp_millis();
        LOAD_DATA_TASK_STATE
            .with_label_values(&[&task_ctx.task_id, LoadDataWorkerState::InitTask.as_str()])
            .set(millis as f64);

        Self {
            task_id: task_ctx.clone().task_id,
            start_ts: task_ctx.start_ts,
            commit_ts: task_ctx.commit_ts,
            compression: 0,
            state: LoadDataWorkerState::InitTask,
            sst_metas: vec![],
            local_file_infos: vec![],
            duplicated_entries: vec![],
            is_recover: false,
            first_key: Default::default(),
            flushed_chunk_ids: Default::default(),
            flushed_file_idx: 0,
            key_comm_prefix: vec![],
            canceled: false,
            error: "".to_string(),
            kvpairs_workers_ctx: HashMap::default(),
            building_workers_ctx: HashMap::default(),
        }
    }

    pub fn get_first_key(&self) -> Bytes {
        self.first_key.clone()
    }

    pub fn get_local_file_infos(&self) -> Vec<LocalFileInfo> {
        self.local_file_infos.clone()
    }

    pub fn get_compression(&self) -> u8 {
        self.compression
    }

    pub fn get_state(&self) -> LoadDataWorkerState {
        self.state
    }

    pub fn get_sst_metas(&self) -> Vec<SstMeta> {
        self.sst_metas.clone()
    }

    pub fn get_duplicated_entries(&self) -> Vec<DuplicateEntry> {
        self.duplicated_entries.clone()
    }

    pub fn get_flushed_chunk_ids(&self) -> HashMap<u64 /* writer_id */, u64 /* chunk_id */> {
        self.flushed_chunk_ids.clone()
    }

    pub fn get_flushed_file_idx(&self) -> usize {
        self.flushed_file_idx
    }

    pub fn get_key_comm_prefix(&self) -> Vec<u8> {
        self.key_comm_prefix.clone()
    }

    pub fn get_is_recover(&self) -> bool {
        self.is_recover
    }

    pub fn set_is_recover(&mut self, is_recover: bool) {
        self.is_recover = is_recover;
    }

    pub fn get_commit_ts(&self) -> u64 {
        self.commit_ts
    }

    pub fn get_start_ts(&self) -> u64 {
        self.start_ts
    }

    pub fn get_task_id(&self) -> String {
        self.task_id.clone()
    }

    pub fn get_kvpairs_worker_ctx(&mut self, worker_id: u64) -> &KvPairsWorkerCtx {
        self.kvpairs_workers_ctx.entry(worker_id).or_default()
    }

    pub fn get_building_worker_ctx(&mut self, worker_id: u64) -> &BuildingWorkerCtx {
        self.building_workers_ctx.entry(worker_id).or_default()
    }

    pub fn recover_from_old_model(&self) -> bool {
        self.is_recover && !self.key_comm_prefix.is_empty()
    }
}

pub struct LocalFileCheckpointStorage {
    data_path: PathBuf,
    file_name: String,
    pub checkpoint_ctx: LoadDataCheckpointCtx,
}

impl LocalFileCheckpointStorage {
    pub fn get_file_name_by_taskid(task_id: String) -> String {
        CHECKPOINT_WORKER_PREFIX.to_string() + &task_id
    }
    pub fn new(checkpoint_ctx: LoadDataCheckpointCtx, data_path: PathBuf) -> Result<Self> {
        let file_name =
            LocalFileCheckpointStorage::get_file_name_by_taskid(checkpoint_ctx.clone().task_id);

        Ok(Self {
            data_path,
            file_name,
            checkpoint_ctx,
        })
    }

    fn checkpoint_ctx_to_binary(&self) -> Vec<u8> {
        let res = serde_json::to_string(&self.checkpoint_ctx.clone()).unwrap();
        res.as_bytes().to_owned()
    }

    pub fn binary_to_checkpoint(json_str: &str) -> LoadDataCheckpointCtx {
        serde_json::from_str(json_str).unwrap()
    }

    fn write_atomic_file(&mut self, content: &[u8]) -> Result<()> {
        // Get mutex lock.
        let _lock = FILE_LOCK.lock();

        let tmp_file_path = self.file_name.clone() + ".tmp";
        let tmp_file = self.data_path.join(tmp_file_path);

        let mut file = OpenOptions::new()
            .write(true)
            .truncate(true)
            .create(true)
            .open(&tmp_file)?;
        file.write_all(content)?;
        file.flush()?;

        fs::rename(tmp_file, self.get_file_path())?;
        Ok(())
    }

    pub fn clean_checkpoint_data(&self) {
        let task_id = &self.checkpoint_ctx.task_id;
        let file_path = self.get_file_path();
        info!("{} remove checkpoint data :{:?}", task_id, file_path);
        if let Err(err) = fs::remove_file(file_path) {
            if err.kind() != std::io::ErrorKind::NotFound {
                error!("{} failed to delete checkpoint file: {}", task_id, err);
            }
        }
    }

    pub fn update_cancel_and_errmsg(&mut self, canceled: bool, errmsg: String) -> Result<()> {
        self.checkpoint_ctx.canceled = canceled;
        self.checkpoint_ctx.error = errmsg;
        self.flush_checkpoint_ctx()
    }

    fn get_file_path(&self) -> PathBuf {
        self.data_path.join(self.file_name.clone())
    }

    pub fn read_file(path: PathBuf) -> String {
        fs::read_to_string(path).unwrap()
    }

    fn transition(&mut self, new_state: LoadDataWorkerState) -> bool {
        let old_state = self.checkpoint_ctx.state;
        let is_succ = self.checkpoint_ctx.state.transition(new_state);
        if old_state != new_state {
            let now = Utc::now();
            let millis = now.timestamp_millis();
            LOAD_DATA_TASK_STATE
                .with_label_values(&[&self.checkpoint_ctx.task_id, new_state.as_str()])
                .set(millis as f64);
        }
        if !is_succ {
            error!(
                "{} [checkpoint] transition try check state failed, from {:?} to {:?}",
                self.checkpoint_ctx.task_id, old_state, new_state
            );
        } else {
            info!(
                "{} [checkpoint] transition try check state succeed, from {:?} to {:?}",
                self.checkpoint_ctx.task_id, old_state, new_state
            );
        }
        is_succ
    }

    pub fn get_sst_meta(&self) -> Vec<SstMeta> {
        self.checkpoint_ctx.sst_metas.clone()
    }

    pub fn update_build_msg(&mut self, compression_type: u8) -> Result<()> {
        self.checkpoint_ctx.compression = compression_type;
        self.flush_checkpoint_ctx_with_state(LoadDataWorkerState::BuildingSst)?;
        Ok(())
    }

    pub fn update_flushed_info(
        &mut self,
        handled_chunk_ids: HashMap<u64, u64>,
        file_idx: usize,
        local_file_infos: Vec<LocalFileInfo>,
        key_comm_prefix: Vec<u8>,
    ) -> Result<()> {
        // Update flushed chunk ids.
        for (writer_id, chunk_id) in handled_chunk_ids {
            let flushed_chunk_id = self
                .checkpoint_ctx
                .flushed_chunk_ids
                .entry(writer_id)
                .or_insert(0);
            assert!(*flushed_chunk_id <= chunk_id);
            *flushed_chunk_id = chunk_id;
        }

        // Update flushed file idx.
        self.checkpoint_ctx.flushed_file_idx = file_idx;

        // Update local file infos.
        for local_file_info in local_file_infos {
            self.checkpoint_ctx.local_file_infos.push(local_file_info);
            debug!(
                "{} [checkpoint store] update local_file_infos {:?}",
                self.checkpoint_ctx.task_id,
                self.checkpoint_ctx.local_file_infos.clone()
            );
        }
        self.checkpoint_ctx.key_comm_prefix = key_comm_prefix;

        self.flush_checkpoint_ctx_with_state(LoadDataWorkerState::AddingChunks)?;
        Ok(())
    }

    pub fn update_first_key_and_prefix(
        &mut self,
        first_key: Bytes,
        key_comm_prefix: Vec<u8>,
    ) -> Result<()> {
        self.checkpoint_ctx.first_key = first_key;
        self.checkpoint_ctx.key_comm_prefix = key_comm_prefix;
        self.flush_checkpoint_ctx()?;
        Ok(())
    }

    pub fn update_build_result(
        &mut self,
        sst_metas: Vec<SstMeta>,
        duplicated_entries: Vec<DuplicateEntry>,
    ) -> Result<()> {
        self.checkpoint_ctx.sst_metas = sst_metas;
        self.checkpoint_ctx.duplicated_entries = duplicated_entries;
        self.flush_checkpoint_ctx()?;
        Ok(())
    }

    pub fn get_local_file_infos(&self) -> Vec<LocalFileInfo> {
        self.checkpoint_ctx.local_file_infos.clone()
    }

    pub fn get_state(&self) -> LoadDataWorkerState {
        self.checkpoint_ctx.get_state()
    }

    pub fn flush_checkpoint_ctx_with_state(
        &mut self,
        new_state: LoadDataWorkerState,
    ) -> Result<()> {
        let is_succ = self.transition(new_state);

        if !is_succ {
            return Err(Error::CheckError("transition state err".to_string()));
        }
        self.checkpoint_ctx.state = new_state;
        self.flush_checkpoint_ctx()?;
        Ok(())
    }

    pub fn flush_checkpoint_ctx(&mut self) -> Result<()> {
        let value: Value = self.checkpoint_ctx_to_binary().to_vec();
        self.write_atomic_file(value.as_slice())?;
        self.print_log();
        Ok(())
    }

    pub fn print_log(&self) {
        let cp = self.checkpoint_ctx.clone();
        let duplicated_entries_size = cp.duplicated_entries.len();
        debug!(
            "{} [checkpoint store] checkpoint context: {:?}, duplicated_entries_size: {}, sst_metas.len(): {},",
            self.checkpoint_ctx.task_id,
            cp,
            duplicated_entries_size,
            cp.sst_metas.len()
        );
    }

    pub fn load_checkpoint_ctx(&self) -> LoadDataCheckpointCtx {
        let file_data = LocalFileCheckpointStorage::read_file(self.get_file_path());
        let checkpoint = LocalFileCheckpointStorage::binary_to_checkpoint(file_data.as_str());
        debug!(
            "{} [checkpoint store] loaded checkpoint: {:?},",
            self.checkpoint_ctx.task_id, checkpoint
        );
        checkpoint
    }
}

// The following methods are used by KvPairsWorker & BuildingWorker.
impl LocalFileCheckpointStorage {
    pub fn get_is_recover(&self) -> bool {
        self.checkpoint_ctx.get_is_recover()
    }

    pub fn update_first_key(&mut self, first_key: Bytes) -> Result<()> {
        self.checkpoint_ctx.first_key = first_key;
        self.flush_checkpoint_ctx()
    }

    pub fn update_l0_flushed_info(
        &mut self,
        worker_id: u64,
        handled_chunk_ids: HashMap<u64, u64>,
        mut l0_file_metas: Vec<FileMeta>,
        key_comm_prefix: Vec<u8>,
    ) -> Result<()> {
        let worker_ctx = self
            .checkpoint_ctx
            .kvpairs_workers_ctx
            .get_mut(&worker_id)
            .unwrap();
        worker_ctx.flushed_chunk_ids = handled_chunk_ids;
        worker_ctx.key_comm_prefix = key_comm_prefix;
        worker_ctx.l0_file_metas.append(&mut l0_file_metas);

        self.flush_checkpoint_ctx_with_state(LoadDataWorkerState::AddingChunks)
    }

    pub fn update_l1_flushed_info(
        &mut self,
        worker_id: u64,
        l1_file_metas: Vec<FileMeta>,
        duplicated_entries: Vec<DuplicateEntry>,
    ) -> Result<()> {
        let worker_ctx = self
            .checkpoint_ctx
            .kvpairs_workers_ctx
            .get_mut(&worker_id)
            .unwrap();
        worker_ctx.l1_file_metas = l1_file_metas;
        worker_ctx.duplicated_entries = duplicated_entries;

        self.flush_checkpoint_ctx_with_state(LoadDataWorkerState::BuildingSst)
    }

    pub fn update_sst_metas(
        &mut self,
        worker_id: u64,
        sst_metas: Vec<SstMeta>,
        duplicated_entries: Vec<DuplicateEntry>,
    ) -> Result<()> {
        let worker_ctx = self
            .checkpoint_ctx
            .building_workers_ctx
            .get_mut(&worker_id)
            .unwrap();

        worker_ctx.sst_metas = sst_metas;
        worker_ctx.duplicated_entries = duplicated_entries;
        self.flush_checkpoint_ctx_with_state(LoadDataWorkerState::BuildingSst)
    }

    pub fn set_worker_ingested(&mut self, worker_id: u64) -> Result<()> {
        let worker_ctx = self
            .checkpoint_ctx
            .building_workers_ctx
            .get_mut(&worker_id)
            .unwrap();
        worker_ctx.ingested = true;
        self.flush_checkpoint_ctx_with_state(LoadDataWorkerState::BuildingSst)
    }

    pub fn set_ingested(&mut self, duplicated_entries: Vec<DuplicateEntry>) -> Result<()> {
        self.checkpoint_ctx.duplicated_entries = duplicated_entries;
        self.checkpoint_ctx.state = LoadDataWorkerState::IngestedSst;
        self.flush_checkpoint_ctx()
    }
}

struct TracingTaskState {
    updated_at: i64,
    canceled_at: i64,
    canceled: bool,
    flushed_files: usize,
    created_files: usize,
    ingested_regions: usize,
}

pub struct LoadDataCleanupWorker {
    running_tasks: Arc<DashMap<String, LoadTaskScheduler>>,
    tracing_task_states: HashMap<String, TracingTaskState>,
    cleanup_interval_secs: u64,
    cancelled_task_expire_secs: i64,
    idle_task_expire_secs: i64,
}

impl LoadDataCleanupWorker {
    pub fn new(
        running_tasks: Arc<DashMap<String, LoadTaskScheduler>>,
        cleanup_interval_secs: u64,
        cancelled_task_expire_secs: i64,
        idle_task_expire_secs: i64,
    ) -> Self {
        Self {
            running_tasks,
            tracing_task_states: HashMap::default(),
            cleanup_interval_secs,
            cancelled_task_expire_secs,
            idle_task_expire_secs,
        }
    }

    pub fn run(&mut self) {
        let interval = Duration::from_secs(self.cleanup_interval_secs);
        info!("start to run cleanup worker");
        loop {
            let task_states: Vec<LoadTaskStates> = self
                .running_tasks
                .iter()
                .map(|x| {
                    x.check_task_thread_finished();
                    x.states.read().unwrap().clone()
                })
                .collect();

            let now_timestamp = chrono::Utc::now().timestamp();
            for task_state in task_states {
                let task_id = task_state.task_id;
                let tracing_task_state =
                    self.tracing_task_states
                        .entry(task_id.clone())
                        .or_insert(TracingTaskState {
                            updated_at: now_timestamp,
                            canceled_at: 0,
                            canceled: false,
                            flushed_files: 0,
                            created_files: 0,
                            ingested_regions: 0,
                        });

                if tracing_task_state.flushed_files != task_state.flushed_files
                    || tracing_task_state.created_files != task_state.created_files
                    || tracing_task_state.ingested_regions != task_state.ingested_regions
                {
                    tracing_task_state.flushed_files = task_state.flushed_files;
                    tracing_task_state.created_files = task_state.created_files;
                    tracing_task_state.ingested_regions = task_state.ingested_regions;
                    tracing_task_state.updated_at = now_timestamp;
                }

                if tracing_task_state.canceled != task_state.canceled {
                    tracing_task_state.canceled = task_state.canceled;
                    tracing_task_state.canceled_at = now_timestamp;
                }

                if tracing_task_state.canceled_at > 0
                    && now_timestamp - tracing_task_state.canceled_at
                        > self.cancelled_task_expire_secs
                {
                    info!(
                        "clean up canceled task {}, duration {}",
                        task_id,
                        now_timestamp - tracing_task_state.canceled_at
                    );
                    if let Some((_, scheduler)) = self.running_tasks.remove(&task_id) {
                        scheduler.sender.send(LoadTaskMsg::Cleanup).unwrap();
                    }
                } else if now_timestamp - tracing_task_state.updated_at > self.idle_task_expire_secs
                {
                    info!(
                        "clean up idle task {}, duration {}",
                        task_id,
                        now_timestamp - tracing_task_state.updated_at
                    );
                    if let Some((_, scheduler)) = self.running_tasks.remove(&task_id) {
                        scheduler.cancel("gc by cleanup worker".to_string());
                        scheduler.sender.send(LoadTaskMsg::Cleanup).unwrap();
                    }
                }
            }

            let task_ids: Vec<String> = self
                .tracing_task_states
                .iter()
                .map(|x| x.0.clone())
                .collect();
            for task_id in &task_ids {
                if !self.running_tasks.contains_key(task_id) {
                    self.tracing_task_states.remove(task_id);
                }
            }
            std::thread::sleep(interval);
        }
    }
}

#[cfg(test)]
mod tests {

    use std::sync::RwLock;

    use tempfile::TempDir;

    use super::*;
    use crate::task::WritersStates;

    #[test]
    fn test_local_file_store() {
        let expect_task_id = "task_id_001".to_string();
        let expect_state = LoadDataWorkerState::AddingChunks;
        let task_ctx = TaskContext {
            task_id: expect_task_id.clone(),
            start_ts: 1_u64,
            commit_ts: 1_u64,
            inner_key_off: None,
            outer_key_prefix: vec![],
            encryption_key: None,
            keyspace_id: None,
        };
        let checkpoint = LoadDataCheckpointCtx::new(task_ctx);

        let checkpoint_dir = TempDir::new().unwrap();
        let mut store =
            LocalFileCheckpointStorage::new(checkpoint, checkpoint_dir.path().to_owned()).unwrap();
        store.flush_checkpoint_ctx_with_state(expect_state).unwrap();
        let loaddata_checkpoint_msg = store.load_checkpoint_ctx();
        let res_task_id = loaddata_checkpoint_msg.task_id;

        assert_eq!(
            expect_task_id, res_task_id,
            "case {}: {}",
            expect_task_id, res_task_id
        );
        assert_eq!(
            expect_state, loaddata_checkpoint_msg.state,
            "case {:?}: {:?}",
            expect_state, loaddata_checkpoint_msg.state
        );

        // Add chunks

        // update_first_key
        let expect_first_key = Bytes::from_static(b"test_first_key");
        store
            .update_first_key_and_prefix(expect_first_key.clone(), expect_first_key.to_vec())
            .unwrap();
        let loaddata_checkpoint_msg = store.load_checkpoint_ctx();
        assert_eq!(expect_first_key, loaddata_checkpoint_msg.first_key);

        let expect_is_recover = true;
        store.checkpoint_ctx.set_is_recover(true);
        store.flush_checkpoint_ctx().unwrap();
        let loaddata_checkpoint_msg = store.load_checkpoint_ctx();
        assert_eq!(expect_is_recover, loaddata_checkpoint_msg.is_recover);

        // update_flushed_info
        let mut handled_chunk_ids: HashMap<u64, u64> = HashMap::new();
        handled_chunk_ids.insert(1, 100);
        handled_chunk_ids.insert(2, 200);
        handled_chunk_ids.insert(3, 300);

        let max_file_idx = 10;

        let local_file_infos: Vec<LocalFileInfo> = vec![
            LocalFileInfo {
                path: PathBuf::from("/path/to/file1"),
                kv_count: 10,
            },
            LocalFileInfo {
                path: PathBuf::from("/path/to/file2"),
                kv_count: 20,
            },
            LocalFileInfo {
                path: PathBuf::from("/path/to/file3"),
                kv_count: 30,
            },
        ];
        let key_comm_prefix = "test_".as_bytes().to_vec();

        store
            .update_flushed_info(
                handled_chunk_ids,
                max_file_idx,
                local_file_infos.clone(),
                key_comm_prefix.clone(),
            )
            .unwrap();
        let loaddata_checkpoint_msg = store.load_checkpoint_ctx();
        assert_eq!(max_file_idx, loaddata_checkpoint_msg.flushed_file_idx);

        let get_local_file_infos = loaddata_checkpoint_msg.local_file_infos;
        assert_eq!(local_file_infos.len(), get_local_file_infos.len());

        for i in 0..get_local_file_infos.len() {
            assert_eq!(local_file_infos[i].path, get_local_file_infos[i].path);
            assert_eq!(
                local_file_infos[i].kv_count,
                get_local_file_infos[i].kv_count
            );
        }
        assert_eq!(key_comm_prefix, loaddata_checkpoint_msg.key_comm_prefix);

        // build sst
        let compression_type = 1;

        // update_build_msg
        store.update_build_msg(compression_type).unwrap();
        let loaddata_checkpoint_msg = store.load_checkpoint_ctx();
        assert_eq!(compression_type, loaddata_checkpoint_msg.compression);

        // update_sst_meta
        let smallest = vec![1, 2, 3];
        let biggest = vec![4, 5, 6];

        let sst_meta = SstMeta {
            id: 1,
            smallest,
            biggest,
            size: 3,
            meta_offset: 0,
            uncompressed_size: 3,
            keys: 3,
        };

        let sst_metas = vec![sst_meta.clone()];

        // update_duplicated_entries
        let key = "test_key".to_string();
        let values = vec!["value1".to_string(), "value2".to_string()];
        let entry = DuplicateEntry { key, values };
        let entries = vec![entry.clone()];

        store.update_build_result(sst_metas, entries).unwrap();
        let loaddata_checkpoint_msg = store.load_checkpoint_ctx();
        assert_eq!(sst_meta, loaddata_checkpoint_msg.sst_metas[0]);
        assert_eq!(entry, loaddata_checkpoint_msg.duplicated_entries[0]);
    }

    #[test]
    fn test_load_data_worker_state() {
        let mut state = LoadDataWorkerState::InitTask;
        let is_succ = state.transition(LoadDataWorkerState::InitTask);
        assert_eq!(true, is_succ);

        let is_succ = state.transition(LoadDataWorkerState::AddingChunks);
        assert_eq!(true, is_succ);

        let is_succ = state.transition(LoadDataWorkerState::BuildingSst);
        assert_eq!(true, is_succ);

        let is_succ = state.transition(LoadDataWorkerState::IngestingSst);
        assert_eq!(true, is_succ);

        let is_succ = state.transition(LoadDataWorkerState::IngestedSst);
        assert_eq!(true, is_succ);

        let is_succ = state.transition(LoadDataWorkerState::IngestedSst);
        assert_eq!(true, is_succ);

        let is_succ = state.transition(LoadDataWorkerState::IngestingSst);
        assert_eq!(false, is_succ);
    }

    #[test]
    fn test_checkpoint_default() {
        let _ = LocalFileCheckpointStorage::binary_to_checkpoint("{}");
    }

    #[test]
    fn test_cleanup_worker() {
        let running_tasks: Arc<DashMap<String, LoadTaskScheduler>> = Arc::new(DashMap::new());
        let checkpoint_dir = TempDir::new().unwrap();
        let path = checkpoint_dir.path();

        // canceled task
        let task_id = "task_id1";
        let checkpoint_store = make_test_checkpoint_storage(path.to_owned(), task_id.to_string());
        let (sender, receiver1) = tikv_util::mpsc::unbounded();
        let thread_handle = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_secs(60));
        });
        let scheduler = LoadTaskScheduler {
            sender,
            states: Arc::new(RwLock::new(LoadTaskStates::default())),
            writers: Arc::new(Mutex::new(WritersStates::default())),
            thread_handle: Some(Arc::new(Mutex::new(thread_handle))),
            checkpoint_store: Arc::new(Mutex::new(checkpoint_store)),
        };
        let mut states = scheduler.states.write().unwrap();
        states.task_id = task_id.to_string();
        drop(states);
        scheduler.cancel("cancel for test".to_string());
        running_tasks.insert(task_id.to_owned(), scheduler);

        // idle task
        let task_id = "task_id2";
        let checkpoint_store = make_test_checkpoint_storage(path.to_owned(), task_id.to_string());
        let (sender, receiver2) = tikv_util::mpsc::unbounded();
        let thread_handle = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_secs(60));
        });
        let scheduler2 = LoadTaskScheduler {
            sender,
            states: Arc::new(RwLock::new(LoadTaskStates::default())),
            writers: Arc::new(Mutex::new(WritersStates::default())),
            thread_handle: Some(Arc::new(Mutex::new(thread_handle))),
            checkpoint_store: Arc::new(Mutex::new(checkpoint_store)),
        };
        let mut states = scheduler2.states.write().unwrap();
        states.task_id = task_id.to_string();
        drop(states);
        running_tasks.insert(task_id.to_owned(), scheduler2.clone());

        // normal task
        let task_id = "task_id3";
        let checkpoint_store = make_test_checkpoint_storage(path.to_owned(), task_id.to_string());
        let (sender, receiver3) = tikv_util::mpsc::unbounded();
        let thread_handle = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_secs(60));
        });
        let scheduler3 = LoadTaskScheduler {
            sender,
            states: Arc::new(RwLock::new(LoadTaskStates::default())),
            writers: Arc::new(Mutex::new(WritersStates::default())),
            thread_handle: Some(Arc::new(Mutex::new(thread_handle))),
            checkpoint_store: Arc::new(Mutex::new(checkpoint_store)),
        };
        let mut states = scheduler3.states.write().unwrap();
        states.task_id = task_id.to_string();
        drop(states);
        running_tasks.insert(task_id.to_owned(), scheduler3.clone());

        let mut cleanup_worker = LoadDataCleanupWorker::new(running_tasks.clone(), 5, 10, 10);
        std::thread::spawn(move || {
            cleanup_worker.run();
        });

        for i in 0..10 {
            let mut states = scheduler3.states.write().unwrap();
            states.flushed_files += i;
            drop(states);
            std::thread::sleep(Duration::from_secs(2));
        } // takes 20s = 10 * 2s

        receiver1.try_recv().unwrap();
        receiver2.try_recv().unwrap();
        let msg = receiver3.try_recv();
        assert!(msg.is_err());

        assert!(scheduler2.states.read().unwrap().canceled);
        assert!(!scheduler3.states.read().unwrap().canceled);
        assert!(running_tasks.len() == 1);
    }

    fn make_test_checkpoint_storage(
        checkpoint_dir: PathBuf,
        task_id: String,
    ) -> LocalFileCheckpointStorage {
        let task_ctx = TaskContext {
            task_id,
            start_ts: 1_u64,
            commit_ts: 1_u64,
            inner_key_off: None,
            outer_key_prefix: vec![],
            encryption_key: None,
            keyspace_id: None,
        };

        let checkpoint = LoadDataCheckpointCtx::new(task_ctx);
        let mut store = LocalFileCheckpointStorage::new(checkpoint, checkpoint_dir).unwrap();
        store.flush_checkpoint_ctx().unwrap();
        store
    }
}
