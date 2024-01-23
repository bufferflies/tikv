// Copyright 2023 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    collections::HashMap,
    fs,
    fs::OpenOptions,
    io::Write,
    path::PathBuf,
    sync::Mutex,
    time::{Duration, SystemTime},
};

use bytes::Bytes;
use chrono::Utc;
use serde_derive::{Deserialize, Serialize};
use tikv_client::Value;
use tikv_util::{debug, error, info};

use crate::{
    kv::{DuplicateEntry, SstMeta},
    metrics::LOAD_DATA_TASK_STATE,
    task::TaskContext,
    Error,
};

pub type Result<T> = std::result::Result<T, Error>;
pub const CHECKPOINT_WORKER_PREFIX: &str = "LOAD_DATA_CHECK_POINT_";

// The expiration time of the canceled task file.
pub const CANCELLED_CHECK_POINT_FILE_EXPIRE_SEC: u64 = 30 * 60;

// The interval between each attempt to clean the checkpoint file.
pub const CLEAN_CHECK_POINT_FILE_INTERVAL_SEC: u64 = 30 * 60;
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

    fn as_str(&self) -> &str {
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
pub struct LoadDataCheckPointCtx {
    // From TaskContext.
    pub task_id: String, // Comes from TaskContext.
    start_ts: u64,       // Comes from TaskContext, Used to recover readers.
    commit_ts: u64,      // Comes from TaskContext, Used to recover readers.

    local_file_infos: Vec<LocalFileInfo>, // Used to recover readers.

    first_key: Bytes, // Used to update inner_key_off and encrytion_key.

    compression: u8, // Comes from build request, Used to build sst.

    state: LoadDataWorkerState,
    sst_metas: Vec<SstMeta>,                 // Used to ingest.
    duplicated_entries: Vec<DuplicateEntry>, // Used to ingest.
    is_recover: bool,                        /* If is_recover is true, it means that the
                                              * task is recovered using checkpoint
                                              * information. */

    flushed_chunk_ids: HashMap<u64 /* writer_id */, u64 /* chunk_id */>,
    flushed_file_idx: usize,

    // Used to recover LoadTaskStates.The lightning service periodically obtains the execution
    // progress from LoadTaskStates.
    pub canceled: bool,
    pub error: String,
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct LocalFileInfo {
    pub path: PathBuf,
    pub kv_count: usize,
}

impl LoadDataCheckPointCtx {
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
            canceled: false,
            error: "".to_string(),
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
    pub fn get_file_name_by_taskid(task_id: String) -> String {
        CHECKPOINT_WORKER_PREFIX.to_string() + &task_id
    }
    pub fn new(check_point_ctx: LoadDataCheckPointCtx, data_path: PathBuf) -> Result<Self> {
        let file_name =
            LocalFileCheckPointStorage::get_file_name_by_taskid(check_point_ctx.clone().task_id);

        Ok(Self {
            data_path,
            file_name,
            check_point_ctx,
        })
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

        let tmp_file_path = self.file_name.clone() + ".tmp";
        let tmp_file = self.data_path.join(tmp_file_path);

        let mut file = OpenOptions::new()
            .write(true)
            .truncate(true)
            .create(true)
            .open(&tmp_file)?;
        file.write_all(content)?;

        fs::rename(tmp_file, self.get_file_path())?;
        Ok(())
    }

    pub fn clean_check_point_data(&self) {
        let file_path = self.get_file_path();
        info!("remove check point data :{:?}", file_path);
        if let Err(e) = fs::remove_file(file_path) {
            if e.kind() != std::io::ErrorKind::NotFound {
                error!("failed to delete check point file: {}", e);
            }
        }
    }

    pub fn update_cancel_and_errmsg(&mut self, canceled: bool, errmsg: String) -> Result<()> {
        self.check_point_ctx.canceled = canceled;
        self.check_point_ctx.error = errmsg;
        self.flush_check_point_ctx()?;
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
        if old_state != new_state {
            let now = Utc::now();
            let millis = now.timestamp_millis();
            LOAD_DATA_TASK_STATE
                .with_label_values(&[&self.check_point_ctx.task_id, new_state.as_str()])
                .set(millis as f64);
        }
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
            debug!(
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
        let file_data = LocalFileCheckPointStorage::read_file(self.get_file_path());
        let check_point = LocalFileCheckPointStorage::binary_to_check_point(file_data.as_str());
        info!(
            "{} [check point store] loaded check point:{:?},",
            self.check_point_ctx.task_id, check_point
        );
        check_point
    }
}

// spawn_clean_check_point_files_worker scan the check point files under the
// directory and clean up the cancel status files which have exceeded the
// waiting time.
pub fn spawn_clean_check_point_files_worker(
    check_point_dir: PathBuf,
    cancelled_task_check_point_file_expire_sec: u64,
    clean_check_point_file_interval_sec: u64,
) {
    std::thread::spawn(move || {
        loop {
            try_clean_check_point_files(
                check_point_dir.clone(),
                cancelled_task_check_point_file_expire_sec,
            );
            std::thread::sleep(Duration::from_secs(clean_check_point_file_interval_sec));
        }
    });
}

fn try_clean_check_point_files(
    check_point_dir: PathBuf,
    cancelled_task_check_point_file_expire_sec: u64,
) {
    let files = fs::read_dir(check_point_dir).unwrap();
    let dir_entries: Vec<fs::DirEntry> = files.filter_map(|r| r.ok()).collect();
    for file in &dir_entries {
        let file_name = file.file_name();
        let str_file_name = file_name.to_string_lossy();
        if str_file_name.starts_with(CHECKPOINT_WORKER_PREFIX) {
            let path = file.path();

            let file_data = LocalFileCheckPointStorage::read_file(path.clone());
            let check_point_ctx =
                LocalFileCheckPointStorage::binary_to_check_point(file_data.as_str());

            if check_point_ctx.canceled {
                try_clean_check_point_file(
                    path.clone(),
                    cancelled_task_check_point_file_expire_sec,
                );
            }
        }
    }
}

// Clean up files that have exceeded the wait time.
pub fn try_clean_check_point_file(path: PathBuf, cancelled_task_check_point_file_expire_sec: u64) {
    let metadata = fs::metadata(path.clone()).unwrap();
    let modified_time = metadata.modified().unwrap();
    let time_since_modified = SystemTime::now().duration_since(modified_time).unwrap();
    let time_since_modified = time_since_modified.as_secs();
    if time_since_modified > cancelled_task_check_point_file_expire_sec {
        fs::remove_file(path.clone()).unwrap();
        info!(
            "[check point store] {:?} check point file has been removed due to being modified over {} sec ago.",
            path, cancelled_task_check_point_file_expire_sec
        );
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

        let data_dir = "tikv_worker_dir";
        fs::create_dir_all(data_dir).unwrap();
        let mut store =
            LocalFileCheckPointStorage::new(check_point, PathBuf::from(data_dir)).unwrap();
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

        fs::remove_dir_all(data_dir).unwrap();
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

    // Test whether the file can be cleaned properly after the expire time is
    // exceeded.
    #[test]
    fn test_clean_file() {
        let check_point_dir = String::from("/tmp/test_check_point/");
        if PathBuf::from(check_point_dir.clone()).is_dir() {
            fs::remove_dir_all(check_point_dir.clone()).unwrap();
        }
        fs::create_dir(check_point_dir.clone()).unwrap();
        let cancelled_task_check_point_file_expire_sec = 1;

        let task_id1 = "task_id_001".to_string();
        let store1 = make_test_check_point_storage(check_point_dir.clone(), task_id1);

        // Sleep a while, wait file update time exceeds the expected wait time.
        std::thread::sleep(Duration::from_secs(
            cancelled_task_check_point_file_expire_sec + 2,
        ));

        // The file corresponding to path2 did not pass the wait time and was not
        // cleaned
        let task_id2 = "task_id_002".to_string();
        let store2 = make_test_check_point_storage(check_point_dir.clone(), task_id2);

        try_clean_check_point_files(
            PathBuf::from(check_point_dir.clone()),
            cancelled_task_check_point_file_expire_sec,
        );

        // The file of task_id1 should be deleted.
        assert_eq!(false, store1.get_file_path().exists());
        // The file of task_id2 was not cleaned up because the wait time was not
        // reached.
        assert_eq!(true, store2.get_file_path().exists());

        fs::remove_dir_all(check_point_dir).unwrap();
    }

    fn make_test_check_point_storage(
        check_point_dir: String,
        task_id: String,
    ) -> LocalFileCheckPointStorage {
        let task_ctx = TaskContext {
            task_id,
            start_ts: 1_u64,
            commit_ts: 1_u64,
            inner_key_off: None,
            key_prefix: vec![],
            encryption_key: None,
        };

        let mut check_point = LoadDataCheckPointCtx::new(task_ctx);
        check_point.canceled = true;
        let mut store =
            LocalFileCheckPointStorage::new(check_point, PathBuf::from(check_point_dir)).unwrap();
        store.flush_check_point_ctx().unwrap();
        store
    }

    #[test]
    fn test_check_point_default() {
        let _ = LocalFileCheckPointStorage::binary_to_check_point("{}");
    }
}
