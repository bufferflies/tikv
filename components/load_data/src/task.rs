// Copyright 2023 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    cmp::Ordering,
    collections::HashMap,
    fs,
    io::Write,
    path::PathBuf,
    sync::{Arc, Mutex, RwLock},
    time::Duration,
};

use api_version::api_v2::KEYSPACE_PREFIX_LEN;
use bytes::{Buf, BufMut, Bytes};
use chrono::Utc;
use cloud_encryption::{EncryptionKey, MasterKey};
use encryption::{DecrypterReader, EncrypterWriter, Iv};
use http::Request;
use hyper::Body;
use kvengine::{
    dfs,
    table::{ChecksumType, InnerKey},
    IdVer, ShardTag, WRITE_CF, WRITE_CF_BOTTOM_LEVEL,
};
use kvproto::{encryptionpb::EncryptionMethod, metapb, pdpb};
use pd_client::PdClient;
use protobuf::Message;
use rfengine::compress_lz4;
use rfstore::store::{raw_end_key, raw_start_key};
use serde_derive::{Deserialize, Serialize};
use tikv_util::{box_err, codec::bytes::encode_bytes, error, mpsc::Sender, time::Instant, warn};

use crate::{
    checkpoint::LocalFileCheckpointStorage,
    error::{Error, Result},
    kv::{DuplicateEntry, KvPair, KvPairsReader, SstMeta},
    metrics::LOAD_DATA_TASK_STATE,
};

pub const DEFAULT_MAX_IN_MEM_SIZE: usize = 256 * 1024 * 1024; // 256MB
const DEFAULT_FLUSH_BATCH_SIZE: usize = 2 * 1024 * 1024; // 2MB
const DEFAULT_KVPAIRS_WORKER_NUM: usize = 1;
const DEFAULT_BUILDING_WORKER_NUM: usize = 1;
const DEFAULT_BLOCK_SIZE: usize = 64 * 1024; // 64KB
const DEFAULT_SST_FILE_SIZE: usize = 48 * 1024 * 1024; // 48MB
const DEFAULT_REGION_SIZE: usize = 750 * 1024 * 1024; // 750MB
const DEFAULT_COARSE_SPLIT_SIZE: usize = 32 * 1024 * 1024 * 1024; // 32GB
const DEFAULT_ENABLE_CHECKPOINT: bool = false;

pub const ZSTD_COMPRESSION_LEVEL: i32 = 3;
pub const FLUSH_FILE_CONCURRENCY: usize = 8;
pub const CREATE_FILE_CONCURRENCY: usize = 32;
pub const INGEST_CONCURRENCY: usize = 4;

pub const ALLOCATE_ID_TIMEOUT: Duration = Duration::from_secs(10 * 60);
pub const RETRY_SLEEP_DURATION: Duration = Duration::from_millis(100);
pub const MAX_RETRY_TIMES: usize = 10;
pub const MAX_SLEEP_DURATION: Duration = Duration::from_secs(30);
pub const GET_SHARD_META_TIMEOUT: Duration = Duration::from_secs(60);

// the following constants are used to calculate RU consumption
pub const DEFAULT_AVG_BATCH_PROPORTION: f64 = 0.5;
pub const REPLICA_NUMS: f64 = 3.0;
pub const TXN_FILE_RU_DISCOUNT_RATIO: f64 = 0.125;

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
        warn!("canceled {}", err);
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

pub async fn ingest_files_to_leader(
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

// Note: also used by `TxnChunkHandler`.
// TODO: find a better place for this method.
pub async fn get_shard_meta(
    pd: Arc<dyn PdClient>,
    shard_raw_key: &[u8],
    timeout: Duration,
) -> Result<kvenginepb::ChangeSet> {
    let security_mgr = pd.get_security_mgr();
    let http_client = security_mgr.http_client(hyper::Client::builder())?;
    let encoded_key = encode_bytes(shard_raw_key);
    let start_time = Instant::now_coarse();
    let mut retry = 0;
    loop {
        if start_time.saturating_elapsed() >= timeout {
            return Err(Error::Other(box_err!(
                "get_shard_meta failed, key: {:?}",
                shard_raw_key
            )));
        }
        if retry > 0 {
            tokio::time::sleep(Duration::from_millis(500)).await;
        }
        retry += 1;

        let region_res = pd.get_region_async(&encoded_key).await;
        if region_res.is_err() {
            error!(
                "get_shard_meta: get region error: {:?}, key: {:?}",
                region_res.unwrap_err(),
                encoded_key
            );
            continue;
        }
        let shard_id = region_res.unwrap().get_id();

        let store_res = get_leader_store(pd.clone() as Arc<dyn PdClient>, shard_id, None).await;
        if store_res.is_err() {
            error!(
                "get_shard_meta: get leader error: {:?}, key: {:?}, shard_id: {}",
                store_res.unwrap_err(),
                shard_raw_key,
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

#[allow(clippy::unnecessary_unwrap)]
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

pub fn build_ingest_files(
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
        table_create.set_meta_offset(sst_meta.meta_offset);
        table_creates.push(table_create);
    }
    cs
}

pub fn gen_split_keys(
    outer_key_prefix: &[u8],
    ssts: &[SstMeta],
    split_size: usize,
    include_bound: bool,
) -> Vec<Vec<u8>> {
    let mut keys = vec![];
    if include_bound {
        keys.push(new_region_key(
            outer_key_prefix,
            ssts.first().unwrap().smallest.as_slice(),
        ));
    }
    let mut size = 0;
    for sst in ssts {
        if size > split_size {
            keys.push(new_region_key(outer_key_prefix, sst.smallest.as_slice()));
            size = 0;
        }
        size += sst.size;
    }
    if include_bound {
        // split at last key so the last region will not be split by other concurrent
        // load_data and get epoch not match error.
        let mut last_key = ssts.last().unwrap().biggest.to_vec();
        last_key.push(0);
        keys.push(new_region_key(outer_key_prefix, last_key.as_slice()));
    }
    keys
}

pub fn new_region_key(outer_key_prefix: &[u8], raw_key: &[u8]) -> Vec<u8> {
    let mut key = outer_key_prefix.to_vec();
    key.extend_from_slice(raw_key);
    encode_bytes(&key)
}

pub fn get_ssts_in_range(ssts: &[SstMeta], start: InnerKey<'_>, end: InnerKey<'_>) -> Vec<SstMeta> {
    let position = ssts
        .binary_search_by(|sst| InnerKey::from_inner_buf(&sst.smallest).cmp(&start))
        .unwrap();
    let mut matched = vec![];
    for i in position..ssts.len() {
        let sst = &ssts[i];
        if !end.is_empty() && InnerKey::from_inner_buf(&sst.smallest) >= end {
            break;
        }
        matched.push(sst.clone())
    }
    matched
}

pub fn get_common_prefix(k1: &[u8], k2: &[u8]) -> Vec<u8> {
    let len = std::cmp::min(k1.len(), k2.len());
    let mut offset = len;
    for i in 0..len {
        if k1[i] != k2[i] {
            offset = i;
            break;
        }
    }
    k1[..offset].to_vec()
}

pub fn flush_to_local_file(
    mut kv_pairs: Vec<KvPair>,
    task_ctx: TaskContext,
    path: PathBuf,
    batch_size: usize,
) -> Result<(KvPairsReader, Vec<u8>)> {
    kv_pairs.sort_by(|a, b| {
        let order = a.key.cmp(&b.key);
        if order == Ordering::Equal {
            return a.row_id.cmp(&b.row_id);
        }
        order
    });

    let first = &kv_pairs.first().unwrap().key;
    let last = &kv_pairs.last().unwrap().key;
    let key_comm_prefix = get_common_prefix(first.chunk(), last.chunk());

    let file = fs::OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .read(true)
        .open(path.as_path())?;
    let mut buf: Vec<u8> = Vec::with_capacity(batch_size + batch_size / 8);
    let mut compressed_buf: Vec<u8> = Vec::with_capacity(batch_size + batch_size / 8);
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
        buf.put_u16_le(pair.row_id.len() as u16);
        buf.extend_from_slice(pair.row_id.chunk());

        if buf.len() >= batch_size {
            let compressed_size = compress_lz4(&buf, &mut compressed_buf)? as u32;
            writer.write_all(&compressed_size.to_le_bytes())?;
            writer.write_all(&compressed_buf)?;
            buf.clear();
            compressed_buf.clear();
        }
    }
    if !buf.is_empty() {
        let compressed_size = compress_lz4(&buf, &mut compressed_buf)? as u32;
        writer.write_all(&compressed_size.to_le_bytes())?;
        writer.write_all(&compressed_buf)?;
    }
    writer.flush()?;
    let file = fs::File::open(path)?;
    let reader = DecrypterReader::new(file, method, key, iv).unwrap();
    let table_prefix_offset = KEYSPACE_PREFIX_LEN - task_ctx.inner_key_off.unwrap();
    Ok((
        KvPairsReader::new(kv_pairs.len(), reader, vec![], vec![], table_prefix_offset),
        key_comm_prefix,
    ))
}

pub fn verify_regions_boundary(
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

    #[test]
    fn test_get_common_prefix() {
        let keys = vec![
            vec![b't', 128, 0, 0, 0, 0, 0, 0, 1, b'_', 1],
            vec![b't', 128, 0, 0, 0, 0, 0, 0, 1, b'_', 2],
            vec![b't', 128, 0, 0, 0, 0, 0, 0, 1, b'_', 3],
        ];
        let mut key_comm_prefix = keys[0].clone();

        for key in keys.iter() {
            key_comm_prefix = get_common_prefix(&key_comm_prefix, key);
        }

        let target_key_comm_prefix = vec![b't', 128, 0, 0, 0, 0, 0, 0, 1, b'_'];
        assert_eq!(key_comm_prefix, target_key_comm_prefix);
    }
}
