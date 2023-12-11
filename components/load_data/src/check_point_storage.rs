// Copyright 2023 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    collections::HashMap,
    fs,
    fs::OpenOptions,
    io::Write,
    path::{Path, PathBuf},
    sync::Mutex,
};

use bytes::Bytes;
use serde_derive::{Deserialize, Serialize};
use tikv_client::Value;
use tikv_util::{debug, error, info};

use crate::{
    kv::{DuplicateEntry, SstMeta},
    task::TaskContext,
    Error,
};

pub type Result<T> = std::result::Result<T, Error>;
pub const CHECKPOINT_WORKER_PREFIX: &str = "LOAD_DATA_CHECK_POINT_";
pub const CHECKPOINT_FILE_DIR: &str = "load_data_checkpoint";
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
                if new_state == LoadDataWorkerState::IngestingSst {
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
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct LoadDataCheckPointCtx {
    // From TaskContext.
    pub task_id: String, // Comes from TaskContext.
    start_ts: u64,       // Comes from TaskContext, Used to recover readers.
    commit_ts: u64,      // Comes from TaskContext, Used to recover readers.

    local_file_infos: Vec<LocalFileInfo>, // Used to recover readers.

    first_key: Bytes,

    compression: u8, // Comes from build request, Used to build sst.

    state: LoadDataWorkerState,
    sst_metas: Vec<SstMeta>,                 // Used to ingest.
    duplicated_entries: Vec<DuplicateEntry>, // Used to ingest.
    is_recover: bool,                        /* If is_recover is true, it means that the
                                              * task is recovered using checkpoint
                                              * information. */

    flushed_chunk_ids: HashMap<u64 /* writer_id */, u64 /* chunk_id */>,
    flushed_file_idx: usize,
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct LocalFileInfo {
    pub path: PathBuf,
    pub kv_count: usize,
}

impl LoadDataCheckPointCtx {
    pub fn new(task_ctx: TaskContext) -> Self {
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
}

pub struct LocalFileCheckPointStorage {
    data_path: PathBuf,
    file_name: String,
    pub check_point_ctx: LoadDataCheckPointCtx,
}

impl LocalFileCheckPointStorage {
    pub fn get_check_point_file_dir() -> PathBuf {
        return Path::new(CHECKPOINT_FILE_DIR).to_path_buf();
    }

    pub fn get_file_name_by_taskid(task_id: String) -> String {
        CHECKPOINT_WORKER_PREFIX.to_string() + &task_id
    }
    pub fn new(check_point_ctx: LoadDataCheckPointCtx) -> Result<Self> {
        let data_path = LocalFileCheckPointStorage::get_check_point_file_dir();
        let file_name =
            LocalFileCheckPointStorage::get_file_name_by_taskid(check_point_ctx.clone().task_id);
        if !data_path.is_dir() {
            LocalFileCheckPointStorage::init_data_dir(data_path.clone())?;
        }

        Ok(Self {
            data_path,
            file_name,
            check_point_ctx,
        })
    }

    fn init_data_dir(data_path: PathBuf) -> Result<()> {
        fs::create_dir_all(data_path)?;
        Ok(())
    }

    fn check_point_ctx_to_binary(&self) -> Vec<u8> {
        let res = serde_json::to_string(&self.check_point_ctx.clone()).unwrap();
        res.as_bytes().to_owned()
    }

    pub fn binary_to_check_point(json_str: &str) -> LoadDataCheckPointCtx {
        serde_json::from_str(json_str).unwrap()
    }

    fn write_atomic_file(&mut self, content: &[u8]) -> Result<()> {
        // Get mutex lock.
        let _lock = FILE_LOCK.lock();

        if !Path::new(CHECKPOINT_FILE_DIR).to_path_buf().is_dir() {
            LocalFileCheckPointStorage::init_data_dir(self.data_path.clone())?;
        }

        let mut file = OpenOptions::new()
            .write(true)
            .truncate(true)
            .create(true)
            .open(self.get_file_path())?;
        file.write_all(content)?;
        Ok(())
    }

    fn get_file_path(&self) -> PathBuf {
        self.data_path.join(self.file_name.clone())
    }

    pub fn read_file(path: PathBuf) -> String {
        fs::read_to_string(path).unwrap()
    }

    fn transition(&mut self, new_state: LoadDataWorkerState) -> bool {
        let old_state = self.check_point_ctx.state;
        let is_succ = self.check_point_ctx.state.transition(new_state);
        if !is_succ {
            error!(
                "{} [check point] transition try check state failed, from {:?} to {:?}",
                self.check_point_ctx.task_id, old_state, new_state
            );
        } else {
            info!(
                "{} [check point] transition try check state succeed, from {:?} to {:?}",
                self.check_point_ctx.task_id, old_state, new_state
            );
        }
        is_succ
    }

    pub fn get_sst_meta(&self) -> Vec<SstMeta> {
        self.check_point_ctx.sst_metas.clone()
    }

    pub fn update_build_msg(&mut self, compression_type: u8) -> Result<()> {
        self.check_point_ctx.compression = compression_type;
        self.flush_check_point_ctx_with_state(LoadDataWorkerState::BuildingSst)?;
        Ok(())
    }

    pub fn update_flushed_info(
        &mut self,
        handled_chunk_ids: HashMap<u64, u64>,
        file_idx: usize,
        local_file_infos: Vec<LocalFileInfo>,
    ) -> Result<()> {
        // Update flushed chunk ids.
        for (writer_id, chunk_id) in handled_chunk_ids {
            let flushed_chunk_id = self
                .check_point_ctx
                .flushed_chunk_ids
                .entry(writer_id)
                .or_insert(0);
            assert!(*flushed_chunk_id <= chunk_id);
            *flushed_chunk_id = chunk_id;
        }

        // Update flushed file idx.
        self.check_point_ctx.flushed_file_idx = file_idx;

        // Update local file infos.
        for local_file_info in local_file_infos {
            self.check_point_ctx.local_file_infos.push(local_file_info);
            info!(
                "{} [check point store] update local_file_infos {:?}",
                self.check_point_ctx.task_id,
                self.check_point_ctx.local_file_infos.clone()
            );
        }

        self.flush_check_point_ctx_with_state(LoadDataWorkerState::AddingChunks)?;
        Ok(())
    }

    pub fn update_first_key(&mut self, first_key: Bytes) -> Result<()> {
        self.check_point_ctx.first_key = first_key;
        self.flush_check_point_ctx()?;
        Ok(())
    }

    pub fn update_build_result(
        &mut self,
        sst_metas: Vec<SstMeta>,
        duplicated_entries: Vec<DuplicateEntry>,
    ) -> Result<()> {
        self.check_point_ctx.sst_metas = sst_metas;
        self.check_point_ctx.duplicated_entries = duplicated_entries;
        self.flush_check_point_ctx()?;
        Ok(())
    }

    pub fn get_local_file_infos(&self) -> Vec<LocalFileInfo> {
        self.check_point_ctx.local_file_infos.clone()
    }

    pub fn get_state(&self) -> LoadDataWorkerState {
        self.check_point_ctx.get_state()
    }

    pub fn flush_check_point_ctx_with_state(
        &mut self,
        new_state: LoadDataWorkerState,
    ) -> Result<()> {
        let is_succ = self.transition(new_state);

        if !is_succ {
            return Err(Error::CheckError("transition state err".to_string()));
        }
        self.check_point_ctx.state = new_state;
        self.flush_check_point_ctx()?;
        Ok(())
    }

    pub fn flush_check_point_ctx(&mut self) -> Result<()> {
        let value: Value = self.check_point_ctx_to_binary().to_vec();
        self.write_atomic_file(value.as_slice())?;
        self.print_log();
        Ok(())
    }

    pub fn print_log(&self) {
        let cp = self.check_point_ctx.clone();
        let duplicated_entries_size = cp.duplicated_entries.len();
        debug!(
            "{} [check point store] check point context:{:?},duplicated_entries_size:{},sst_metas.len():{},",
            self.check_point_ctx.task_id,
            cp,
            duplicated_entries_size,
            cp.sst_metas.len()
        );
    }

    pub fn load_check_point_ctx(&self) -> LoadDataCheckPointCtx {
        // When restart tikv worker, it need read check point ctx from storage.
        if !self.data_path.is_dir() {
            info!(
                "{} [check point store] loaded file not exists {},",
                self.check_point_ctx.task_id,
                self.data_path
                    .clone()
                    .into_os_string()
                    .into_string()
                    .unwrap()
            );
            let task_ctx = TaskContext::default();

            let check_point = LoadDataCheckPointCtx::new(task_ctx);
            info!(
                "{} [check point store] loaded check point:{:?},",
                self.check_point_ctx.task_id, check_point
            );
            return check_point;
        }

        let file_data = LocalFileCheckPointStorage::read_file(self.get_file_path());
        let check_point = LocalFileCheckPointStorage::binary_to_check_point(file_data.as_str());
        info!(
            "{} [check point store] loaded check point:{:?},",
            self.check_point_ctx.task_id, check_point
        );
        check_point
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_local_file_store() {
        let expect_task_id = "task_id_001".to_string();
        let expect_state = LoadDataWorkerState::AddingChunks;
        let task_ctx = TaskContext {
            task_id: expect_task_id.clone(),
            start_ts: 1_u64,
            commit_ts: 1_u64,
            inner_key_off: None,
            key_prefix: vec![],
            encryption_key: None,
        };
        let check_point = LoadDataCheckPointCtx::new(task_ctx);

        let mut store = LocalFileCheckPointStorage::new(check_point).unwrap();
        store
            .flush_check_point_ctx_with_state(expect_state)
            .unwrap();
        let loaddata_checkpoint_msg = store.load_check_point_ctx();
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
        store.update_first_key(expect_first_key.clone()).unwrap();
        let loaddata_checkpoint_msg = store.load_check_point_ctx();
        assert_eq!(expect_first_key, loaddata_checkpoint_msg.first_key);

        let expect_is_recover = true;
        store.check_point_ctx.set_is_recover(true);
        store.flush_check_point_ctx().unwrap();
        let loaddata_checkpoint_msg = store.load_check_point_ctx();
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

        store
            .update_flushed_info(handled_chunk_ids, max_file_idx, local_file_infos.clone())
            .unwrap();
        let loaddata_checkpoint_msg = store.load_check_point_ctx();
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

        // build sst
        let compression_type = 1;

        // update_build_msg
        store.update_build_msg(compression_type).unwrap();
        let loaddata_checkpoint_msg = store.load_check_point_ctx();
        assert_eq!(compression_type, loaddata_checkpoint_msg.compression);

        // update_sst_meta
        let smallest = vec![1, 2, 3];
        let biggest = vec![4, 5, 6];

        let sst_meta = SstMeta {
            id: 1,
            smallest,
            biggest,
            size: 3,
            keys: 3,
        };

        let sst_metas = vec![sst_meta.clone()];

        // update_duplicated_entries
        let key = "test_key".to_string();
        let values = vec!["value1".to_string(), "value2".to_string()];
        let entry = DuplicateEntry { key, values };
        let entries = vec![entry.clone()];

        store.update_build_result(sst_metas, entries).unwrap();
        let loaddata_checkpoint_msg = store.load_check_point_ctx();
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
}
