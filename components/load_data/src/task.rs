// Copyright 2023 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    collections::HashMap,
    path::PathBuf,
    sync::{Arc, Mutex, RwLock},
};

use bytes::Bytes;
use chrono::Utc;
use cloud_encryption::{EncryptionKey, MasterKey};
use kvengine::{dfs, table::ChecksumType};
use pd_client::PdClient;
use serde_derive::{Deserialize, Serialize};
use tikv_util::{mpsc::Sender, warn};

use crate::{
    checkpoint::LocalFileCheckpointStorage, kv::DuplicateEntry, metrics::LOAD_DATA_TASK_STATE,
};

const DEFAULT_MAX_IN_MEM_SIZE: usize = 256 * 1024 * 1024; // 256MB
const DEFAULT_FLUSH_BATCH_SIZE: usize = 1024 * 1024; // 1MB
const DEFAULT_KVPAIRS_WORKER_NUM: usize = 1;
const DEFAULT_BUILDING_WORKER_NUM: usize = 1;
const DEFAULT_BLOCK_SIZE: usize = 64 * 1024; // 64KB
const DEFAULT_SST_FILE_SIZE: usize = 48 * 1024 * 1024; // 48MB
const DEFAULT_REGION_SIZE: usize = 750 * 1024 * 1024; // 750MB
const DEFAULT_COARSE_SPLIT_SIZE: usize = 32 * 1024 * 1024 * 1024; // 32GB
const DEFAULT_ENABLE_CHECKPOINT: bool = false;

pub enum LoadTaskMsg {
    AddChunk {
        writer_id: u64,
        chunk_id: u64,
        chunk_data: Bytes,
        cb: Box<dyn FnOnce(PutChunkResult) + Send>,
    },
    Build {
        compression_type: u8,
        cb: Box<dyn FnOnce(()) + Send>,
    },
    Flush {
        writer_id: u64,
        flush_file_count: Option<usize>,
        cb: Box<dyn FnOnce(FlushStates) + Send>,
    },
    Cleanup,
}

pub enum FlushStates {
    FlushFileCount { flush_file_count: usize },
    FlushResult { flush_result: FlushResult },
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
    pub total_kvs: usize,
    pub duplicated_entries: Vec<DuplicateEntry>,
}

#[derive(Clone, Serialize, Deserialize, PartialEq, Debug)]
pub struct LoadDataConfig {
    pub kvpairs_worker_num: usize,
    pub building_worker_num: usize,
    pub max_in_mem_size: usize,
    pub flush_batch_size: usize,
    pub block_size: usize,
    pub sst_file_size: usize,
    pub region_size: usize,
    pub coarse_split_size: usize,
    pub enable_checkpoint: bool,
    pub rg_config: Option<ResourceGroupConfig>,
    pub checksum_type: ChecksumType,
}

impl Default for LoadDataConfig {
    fn default() -> Self {
        Self {
            kvpairs_worker_num: DEFAULT_KVPAIRS_WORKER_NUM,
            building_worker_num: DEFAULT_BUILDING_WORKER_NUM,
            max_in_mem_size: DEFAULT_MAX_IN_MEM_SIZE,
            flush_batch_size: DEFAULT_FLUSH_BATCH_SIZE,
            block_size: DEFAULT_BLOCK_SIZE,
            sst_file_size: DEFAULT_SST_FILE_SIZE,
            region_size: DEFAULT_REGION_SIZE,
            coarse_split_size: DEFAULT_COARSE_SPLIT_SIZE,
            enable_checkpoint: DEFAULT_ENABLE_CHECKPOINT,
            rg_config: None,
            checksum_type: ChecksumType::Crc32,
        }
    }
}

#[derive(Default, Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(default)]
#[serde(rename_all = "kebab-case")]
pub struct ResourceGroupConfig {
    pub request_unit: RequestUnit,
}

#[derive(Default, Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(default)]
#[serde(rename_all = "kebab-case")]
pub struct RequestUnit {
    pub write_base_cost: f64,
    pub write_per_batch_base_cost: f64,
    pub write_cost_per_byte: f64,
}

#[derive(Clone)]
pub struct LoadDataContext {
    pub dir: PathBuf,
    pub dfs: Arc<dyn dfs::Dfs>,
    pub pd: Arc<dyn PdClient>,
    pub runtime: Arc<tokio::runtime::Runtime>,
    pub master_key: MasterKey,
    pub encryption_key_manager: Arc<cloud_encryption::EncryptionKeyManager>,
}

#[derive(Clone, Default)]
pub struct TaskContext {
    pub task_id: String,
    pub start_ts: u64,
    pub commit_ts: u64,
    pub inner_key_off: Option<usize>,
    pub outer_key_prefix: Vec<u8>,
    pub encryption_key: Option<EncryptionKey>,
    pub keyspace_id: Option<u32>,
}

#[derive(Clone)]
pub struct LoadTaskScheduler {
    pub sender: Sender<LoadTaskMsg>,
    pub states: Arc<RwLock<LoadTaskStates>>,
    pub thread_handle: Option<Arc<Mutex<std::thread::JoinHandle<()>>>>,
    pub checkpoint_store: Arc<Mutex<LocalFileCheckpointStorage>>,
}

impl LoadTaskScheduler {
    pub fn cancel(&self, err: String) {
        let mut states = self.states.write().unwrap();
        if states.canceled {
            return;
        }
        warn!("{} canceled {}", states.task_id, err);
        states.canceled = true;
        states.error = err.clone();
        drop(states);

        let mut checkpoint_guard = self.checkpoint_store.lock().unwrap();
        checkpoint_guard
            .update_cancel_and_errmsg(true, err)
            .unwrap();
        let task_id = &checkpoint_guard.checkpoint_ctx.task_id;

        let ts = Utc::now().timestamp();
        LOAD_DATA_TASK_STATE
            .with_label_values(&[task_id, "cancel"])
            .set(ts as f64);
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
        let states = self.states.read().unwrap();
        states.canceled
    }

    pub fn error_msg(&self) -> String {
        let states = self.states.read().unwrap();
        states.error.clone()
    }

    pub fn states(&self) -> LoadTaskStates {
        let states = self.states.read().unwrap();
        states.clone()
    }

    pub(crate) fn add_ingested_regions(&self) {
        let mut states = self.states.write().unwrap();
        states.ingested_regions += 1;
    }

    pub(crate) fn set_finished(&self, dup_entries: Vec<DuplicateEntry>) {
        let mut states = self.states.write().unwrap();
        states.finished = true;
        states.duplicated_entries = dup_entries;
    }

    pub fn is_finished(&self) -> bool {
        let states = self.states.read().unwrap();
        states.finished
    }

    pub(crate) fn add_created_files(&self, n: usize) {
        let mut states = self.states.write().unwrap();
        states.created_files += n;
    }

    pub(crate) fn add_flushed_files(&self, n: usize) {
        let mut states = self.states.write().unwrap();
        states.flushed_files += n;
    }

    pub(crate) fn add_total_kvs(&self, total_kvs: usize) {
        let mut states = self.states.write().unwrap();
        states.total_kvs += total_kvs;
    }

    pub fn set_thread_handle(&mut self, thread_handle: std::thread::JoinHandle<()>) {
        self.thread_handle = Some(Arc::new(Mutex::new(thread_handle)))
    }
}
