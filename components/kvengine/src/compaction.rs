// Copyright 2021 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    cmp::Ordering as CmpOrdering,
    collections::{hash_map::Entry, HashMap, HashSet},
    hash::Hash,
    iter::Iterator as StdIterator,
    ops::{Deref, Sub},
    path::{Path, PathBuf},
    sync::{atomic::Ordering, Arc, Mutex},
    time::{Duration, Instant},
};

use api_version::{
    api_v2::{KEYSPACE_PREFIX_LEN, TXN_KEY_PREFIX},
    ApiV2,
};
use bstr::ByteSlice;
use bytes::{Buf, Bytes, BytesMut};
use cloud_encryption::{EncryptionKey, MasterKey};
use futures::executor::block_on;
use http::StatusCode;
use hyper::Client;
use itertools::{Either, Itertools};
use kvenginepb::{self as pb, ColumnarCreate};
use pb::{BlobCreate, TableCreate};
use protobuf::Message;
use security::SecurityManager;
use slog_global::error;
use table::columnar::{ColumnarFile, ColumnarTableReader, Schema};
use tidb_query_common::util::convert_to_prefix_next;
use tidb_query_datatype::{
    codec::table::{
        decode_common_handle, decode_int_handle, decode_table_id, encode_row_key,
        encode_row_key_prefix, ID_LEN, TABLE_PREFIX_LEN,
    },
    VECTOR_INDEX_SPEC_KEY_DISTANCE_METRIC,
};
use tikv_util::mpsc;

use crate::{
    dfs,
    dfs::FileType,
    table::{
        blobtable::{
            blobtable::BlobTable,
            builder::{BlobTableBuildOptions, BlobTableBuilder},
        },
        columnar::{
            Block, ColumnarCompactReader, ColumnarConcatReader, ColumnarFileBuilder,
            ColumnarFilterReader, ColumnarMergeReader, ColumnarReader, ColumnarRowTableReader,
            ColumnarTableBuildOptions, ColumnarTableBuilder, ColumnarTruncateTsReader, SchemaFile,
            GLOBAL_COMMON_HANDLE_END,
        },
        file::{File, InMemFile, LocalFile},
        sstable::{self, builder::TableBuilderOptions, BlockCache, L0Builder, SsTable},
        vector_index::VectorIndexBuilder,
        BoundedDataSet, ChecksumType, DataBound, InnerKey,
    },
    Error::{
        CompactionNotRetryable, FallbackLocalCompactorDisabled, IncompatibleRemoteCompactor,
        RemoteCompaction, TableError,
    },
    Iterator, EXTRA_CF, LOCK_CF, WRITE_CF, *,
};

const MAJOR_COMPACTION_MIN_REQUEST_VERSION: u32 = 3;

// Do not skip L1 tables if there are too many small L1 tables.
const MAX_SKIP_L1_TABLES: usize = 16;
pub const INITIAL_SNAP_VERSION: u64 = 1;

static RETRY_INTERVAL: Duration = Duration::from_secs(600);

#[derive(Default, Eq, PartialEq, Hash, Clone)]
pub struct RemoteCompactor {
    remote_url: String,
    permanent: bool,
}

impl RemoteCompactor {
    pub fn new(remote_url: String, permanent: bool) -> Self {
        Self {
            remote_url,
            permanent,
        }
    }
}

pub struct RemoteCompactors {
    remote_urls: Vec<RemoteCompactor>,
    index: usize,
    last_failure: Instant,
}

impl RemoteCompactors {
    pub fn new(remote_url: String) -> Self {
        let mut remote_urls = Vec::new();
        if !remote_url.is_empty() {
            remote_urls.push(RemoteCompactor::new(remote_url, true));
        };
        Self {
            remote_urls,
            index: 0,
            last_failure: Instant::now().sub(RETRY_INTERVAL),
        }
    }
}

#[derive(Clone)]
pub struct CompactionClient {
    dfs: Arc<dyn dfs::Dfs>,
    id_allocator: Arc<dyn IdAllocator>,
    remote_compactors: Arc<Mutex<RemoteCompactors>>,
    client: Option<security::HttpClient>,
    compression_lvl: i32,
    pub(crate) checksum_type: ChecksumType,
    allow_fallback_local: bool,
    master_key: MasterKey,
    local_dir: PathBuf,
    // Whether the compaction client is used during restore (trim over bound & truncate ts).
    for_restore: bool,
}

impl CompactionClient {
    pub(crate) fn new(
        dfs: Arc<dyn dfs::Dfs>,
        remote_url: String,
        compression_lvl: i32,
        checksum_type: ChecksumType,
        allow_fallback_local: bool,
        id_allocator: Arc<dyn IdAllocator>,
        master_key: MasterKey,
        security_mgr: Arc<SecurityManager>,
        local_dir: PathBuf,
        for_restore: bool,
    ) -> Self {
        let remote_compactors = RemoteCompactors::new(remote_url);
        let client = security_mgr
            .http_client(Client::builder().pool_max_idle_per_host(0).clone())
            .unwrap();
        Self {
            dfs,
            remote_compactors: Arc::new(Mutex::new(remote_compactors)),
            client: Some(client),
            compression_lvl,
            checksum_type,
            allow_fallback_local,
            id_allocator,
            master_key,
            local_dir,
            for_restore,
        }
    }

    pub fn add_remote_compactor(&mut self, remote_url: String) {
        if !remote_url.is_empty() {
            let mut remote_compactors = self.remote_compactors.lock().unwrap();
            remote_compactors.last_failure = Instant::now().sub(RETRY_INTERVAL);
            if remote_compactors
                .remote_urls
                .iter()
                .any(|x| x.remote_url == remote_url.clone())
            {
                return;
            }
            remote_compactors
                .remote_urls
                .push(RemoteCompactor::new(remote_url.clone(), false));
            info!(
                "add remote compactor {}, total {}",
                remote_url,
                remote_compactors.remote_urls.len()
            );
        };
    }

    pub fn delete_remote_compactor(&self, remote_compactor: &RemoteCompactor) {
        let mut remote_compactors = self.remote_compactors.lock().unwrap();
        if !remote_compactor.permanent {
            remote_compactors
                .remote_urls
                .retain(|x| x.remote_url != remote_compactor.remote_url);
            info!(
                "delete remote compactor {}, total {}",
                remote_compactor.remote_url,
                remote_compactors.remote_urls.len()
            );
        } else if remote_compactors.remote_urls.len() == 1 {
            remote_compactors.last_failure = Instant::now();
        }
    }

    pub fn get_remote_compactors(&self) -> Vec<String> {
        self.remote_compactors
            .lock()
            .unwrap()
            .remote_urls
            .clone()
            .into_iter()
            .map(|r| r.remote_url)
            .collect()
    }

    pub fn get_remote_compactor(&self) -> RemoteCompactor {
        let mut remote_compactors = self.remote_compactors.lock().unwrap();
        if remote_compactors.remote_urls.is_empty()
            || remote_compactors.last_failure.elapsed().le(&RETRY_INTERVAL)
        {
            RemoteCompactor::default()
        } else {
            remote_compactors.index += 1;
            if remote_compactors.index >= remote_compactors.remote_urls.len() {
                remote_compactors.index = 0;
            }
            remote_compactors.remote_urls[remote_compactors.index].clone()
        }
    }

    pub(crate) fn compact(&self, req: CompactionRequest) -> Result<pb::ChangeSet> {
        let encryption_key = if req.exported_encryption_key.is_empty() {
            None
        } else {
            Some(
                self.master_key
                    .decrypt_encryption_key(&req.exported_encryption_key)
                    .unwrap(),
            )
        };
        let ctx = CompactionCtx {
            req: Arc::new(req),
            dfs: self.dfs.clone(),
            compression_lvl: self.compression_lvl,
            checksum_type: self.checksum_type,
            id_allocator: self.id_allocator.clone(),
            encryption_key,
            local_dir: Some(self.local_dir.clone()),
            for_restore: self.for_restore,
        };
        let req = &ctx.req;
        let mut remote_compactor = self.get_remote_compactor();
        if remote_compactor.remote_url.is_empty() {
            local_compact(&ctx)
        } else {
            let (tx, rx) = tikv_util::mpsc::bounded(1);
            let mut retry_cnt = 0;
            loop {
                let tx = tx.clone();
                let req_clone = req.clone();
                let client = self.clone();
                let remote_url = remote_compactor.remote_url.clone();
                self.dfs.get_runtime().spawn(async move {
                    let result = client.remote_compact(&req_clone, remote_url).await;
                    tx.send(result).unwrap();
                });
                match rx.recv().unwrap() {
                    result @ Ok(_) => break result,
                    Err(e @ IncompatibleRemoteCompactor { .. }) => {
                        if self.allow_fallback_local {
                            warn!("fall back to local compactor due to error: {:?}", e);
                            break local_compact(&ctx);
                        } else {
                            warn!(
                                "remote compactor is incompatible and local compaction is not allowed"
                            );
                            break Err(FallbackLocalCompactorDisabled);
                        }
                    }
                    Err(e) => {
                        retry_cnt += 1;
                        let tag = ShardTag::from_comp_req(req);
                        error!(
                            "shard {}, req {:?}, remote compaction failed {:?}, retrying {} remote compactor {}",
                            tag, req, e, retry_cnt, remote_compactor.remote_url
                        );

                        if remote_compactor.permanent && retry_cnt >= 5
                            || !remote_compactor.permanent && retry_cnt >= 3
                        {
                            retry_cnt = 0;
                            self.delete_remote_compactor(&remote_compactor);
                            remote_compactor = self.get_remote_compactor();
                        }
                        if remote_compactor.remote_url.is_empty() {
                            if self.allow_fallback_local {
                                break local_compact(&ctx);
                            } else {
                                warn!(
                                    "no remote compactor available and local compaction is not allowed"
                                );
                                break Err(FallbackLocalCompactorDisabled);
                            }
                        }
                        std::thread::sleep(Duration::from_secs(1));
                    }
                }
            }
        }
    }

    async fn remote_compact(
        &self,
        comp_req: &CompactionRequest,
        remote_url: String,
    ) -> Result<pb::ChangeSet> {
        let body_str = serde_json::to_string(&comp_req).unwrap();
        let req = hyper::Request::builder()
            .method(hyper::Method::POST)
            .uri(remote_url.clone())
            .header("content-type", "application/json")
            .body(hyper::Body::from(body_str))?;
        let tag = ShardTag::from_comp_req(comp_req);
        info!("{} send request to remote compactor", tag);
        let response = self.client.as_ref().unwrap().request(req).await?;
        info!("{} got response from remote compactor", tag);
        let status = response.status();
        let body = hyper::body::to_bytes(response.into_body()).await?;
        if !status.is_success() {
            let err_msg = String::from_utf8_lossy(body.chunk()).to_string();
            return if status == INCOMPATIBLE_COMPACTOR_ERROR_CODE {
                Err(IncompatibleRemoteCompactor {
                    url: remote_url,
                    msg: err_msg,
                })
            } else {
                Err(RemoteCompaction(err_msg))
            };
        }
        let mut cs = pb::ChangeSet::new();
        if let Err(err) = cs.merge_from_bytes(&body) {
            return Err(RemoteCompaction(err.to_string()));
        }
        Ok(cs)
    }
}

/// `CURRENT_COMPACTOR_VERSION` is used for version compatibility checking of
/// remote compactor. NOTE: Increase `CURRENT_COMPACTOR_VERSION` by 1 when add
/// new feature to remote compactor.
const CURRENT_COMPACTOR_VERSION: u32 = 3;

const INCOMPATIBLE_COMPACTOR_ERROR_CODE: StatusCode = StatusCode::NOT_IMPLEMENTED;

#[derive(Default, Debug, Serialize, Deserialize)]
#[serde(default)]
#[serde(rename_all = "kebab-case")]
pub struct CompactionRequest {
    pub engine_id: u64,
    pub shard_id: u64,
    pub shard_ver: u64,

    #[serde(rename = "start")]
    pub outer_start: Vec<u8>,
    #[serde(rename = "end")]
    pub outer_end: Vec<u8>,
    pub inner_key_off: usize,

    pub file_ids: Vec<u64>,

    pub compaction_tp: CompactionType,
    /// Required version of remote compactor.
    /// Must be set to `CURRENT_COMPACTOR_VERSION`.
    pub compactor_version: u32,

    pub safe_ts: u64,
    pub exported_encryption_key: Vec<u8>,
}

impl CompactionRequest {
    pub fn inner_start(&self) -> InnerKey<'_> {
        InnerKey::from_outer_key(&self.outer_start)
    }

    pub fn inner_end(&self) -> InnerKey<'_> {
        InnerKey::from_outer_end_key(&self.outer_end)
    }

    pub fn get_tag(&self) -> String {
        format!("[{}:{}:{}]", self.engine_id, self.shard_id, self.shard_ver)
    }

    pub fn prepend_keyspace_id(&self) -> Option<u32> {
        if self.inner_key_off > 0 {
            return None;
        }
        ApiV2::get_u32_keyspace_id_by_key(&self.outer_start)
    }

    pub fn keyspace_id(&self) -> u32 {
        ApiV2::get_u32_keyspace_id_by_key(&self.outer_start).unwrap_or_default()
    }
}

#[derive(Debug, Serialize, Deserialize, Default)]
pub struct MajorCompaction {
    safe_ts: u64,
    l0_tables: Vec<u64>,
    // L1 plus sstables of all cfs, map from cf id to sstables that are organized by level.
    ln_tables: HashMap<usize, Vec<(usize, Vec<u64>)>>,
    blob_tables: Vec<u64>,
    sst_config: TableBuilderOptions,
    bt_config: Option<BlobTableBuildOptions>,
}

#[derive(Debug, Serialize, Deserialize, Default)]
pub struct L0Compaction {
    safe_ts: u64,
    l0_tables: Vec<u64>,
    multi_cf_l1_tables: Vec<Vec<u64>>,
    sst_config: TableBuilderOptions,
    bt_config: Option<BlobTableBuildOptions>,
}

#[derive(Debug, Serialize, Deserialize, Default)]
pub struct L1PlusCompaction {
    cf: isize,
    level: usize,
    safe_ts: u64,
    // Whether to keep the latest tombstone before the safe ts.
    keep_latest_obsolete_tombstone: bool,
    upper_level: Vec<u64>,
    lower_level: Vec<u64>,
    sst_config: TableBuilderOptions,
}

#[derive(Debug, Serialize, Deserialize, Default)]
pub struct ColumnarCompaction {
    level: u32,
    safe_ts: u64,
    snap_version: u64,
    source_row_files: Vec<(u32, u64)>,      // (level, id)
    source_columnar_files: Vec<(u32, u64)>, // (level, id)
    schema_file_id: u64,
    columnar_config: ColumnarTableBuildOptions,
}

#[derive(Debug, Serialize, Deserialize, Default)]
pub struct ColumnarMajorCompaction {
    safe_ts: u64,
    l0_tables: Vec<u64>,
    ln_tables: Vec<(usize /* level */, Vec<u64> /* file ids */)>,
    blob_tables: Vec<u64>,
    old_columnar_tables: Vec<(usize /* level */, u64 /* file id */)>,
    schema_file_id: u64,
    columnar_config: ColumnarTableBuildOptions,
    snap_version: u64,
}

#[derive(Debug, Serialize, Deserialize, Default)]
pub enum InPlaceCompaction {
    #[default]
    Unknown,
    TruncateTs(u64),
    TrimOverBound,
    DestroyRange(Vec<u8>),
}

#[derive(Debug, Serialize, Deserialize, Default)]
pub struct InPlaceCompactionCtx {
    file_ids: Vec<(u64, u32, i32)>,
    col_file_ids: Vec<(u64, u32)>,
    block_size: usize,
    columnar_build_opts: ColumnarTableBuildOptions,
    schema_file_id: Option<u64>,
    spec: InPlaceCompaction,
}

#[derive(Debug, Serialize, Deserialize, Default)]
pub struct VectorIndexUpdate {
    table_id: i64,
    index_id: i64,
    col_id: i64,
    snap_version: u64,
    schema_file_id: u64,
    col_file_ids: Vec<(u64, u32)>,
    remove_file_ids: Vec<u64>,
}

#[derive(Debug, Serialize, Deserialize, Default)]
pub enum CompactionType {
    #[default]
    Unknown,
    Major(MajorCompaction),
    L0(L0Compaction),
    L1Plus(L1PlusCompaction),
    InPlace {
        file_ids: Vec<(u64, u32, i32)>,
        block_size: usize,
        spec: InPlaceCompaction,
    },
    InPlaceWithColumnar(InPlaceCompactionCtx),
    Columnar(ColumnarCompaction),
    ColumnarMajor(ColumnarMajorCompaction),
    VectorIndex(VectorIndexUpdate),
}

const MAX_COMPACTION_EXPAND_SIZE: u64 = 256 * 1024 * 1024;

impl Engine {
    pub fn update_managed_safe_ts(&self, ts: u64) {
        loop {
            let old = load_u64(&self.managed_safe_ts);
            if old < ts
                && self
                    .managed_safe_ts
                    .compare_exchange(old, ts, Ordering::Release, Ordering::Relaxed)
                    .is_err()
            {
                continue;
            }
            break;
        }
    }

    pub fn get_managed_safe_ts(&self, keyspace_id: u32) -> u64 {
        let gc_safe_point_ts = load_u64(&self.managed_safe_ts);
        debug!(
            "Get gc safe point v1, keyspace_id:{:?}, gc safepoint:{}",
            keyspace_id, gc_safe_point_ts,
        );
        gc_safe_point_ts
    }

    pub fn get_keyspace_gc_safepoint_v2(&self, keyspace_id: u32) -> u64 {
        match &self.ks_safepoint_v2 {
            Some(sp_map) => {
                if keyspace_id > 0 {
                    // Api v2 key.
                    let keyspace_sp_ts = sp_map.get(&keyspace_id);
                    match keyspace_sp_ts {
                        None => {
                            if self.opts.disable_safe_point_fallback_v1 {
                                debug!(
                                    "Can not get gc safe point v2, and refuse to get gc safe point v1, keyspace_id:{}",
                                    keyspace_id
                                );
                                return 0;
                            }
                            debug!(
                                "Can not get gc safe point v2, get gc safe point v1, keyspace_id:{}",
                                keyspace_id
                            );
                            // Api v1 key.
                            self.get_managed_safe_ts(keyspace_id)
                        }
                        Some(ks2sp) => {
                            let ks_gc_sp = *ks2sp.value();
                            debug!(
                                "Get gc safe point v2, keyspace_id:{}, gc safepoint:{}",
                                keyspace_id, ks_gc_sp
                            );
                            ks_gc_sp
                        }
                    }
                } else {
                    // Api v1 key.
                    self.get_managed_safe_ts(keyspace_id)
                }
            }
            None => {
                // Api v1 key.
                self.get_managed_safe_ts(keyspace_id)
            }
        }
    }

    pub(crate) fn run_compaction(&self, compact_rx: mpsc::Receiver<CompactMsg>) {
        let mut runner = CompactRunner::new(self.clone(), compact_rx);
        runner.run();
    }

    pub(crate) fn compact(&self, shard: Arc<Shard>) -> Option<Result<pb::ChangeSet>> {
        let tag = shard.tag();
        if !shard.ready_to_compact() {
            info!("Shard {} is not ready for compaction", tag);
            return None;
        }
        store_bool(&shard.compacting, true);
        match shard.get_compaction_priority() {
            Some(CompactionPriority::L0 { .. }) => self.trigger_l0_compaction(&shard),
            Some(CompactionPriority::L1Plus { cf, level, .. }) => {
                self.trigger_l1_plus_compaction(&shard, cf, level)
            }
            Some(CompactionPriority::Major { .. }) => self.trigger_major_compaction(&shard),
            Some(CompactionPriority::DestroyRange) => self.destroy_range(&shard).transpose(),
            Some(CompactionPriority::TruncateTs) => self.truncate_ts(&shard).transpose(),
            Some(CompactionPriority::TrimOverBound) => self.trim_over_bound(&shard).transpose(),
            Some(CompactionPriority::L0ToColumnar) => self.trigger_l0_to_columnar(&shard),
            Some(CompactionPriority::ColumnarL0 { .. }) => {
                self.trigger_columnar_l0_compaction(&shard)
            }
            Some(CompactionPriority::ColumnarL1 { .. }) => {
                self.trigger_columnar_l1_compaction(&shard)
            }
            Some(CompactionPriority::ColumnarMajor { .. }) => {
                self.trigger_columnar_major_compaction(&shard)
            }
            Some(CompactionPriority::ColumnarClear) => {
                self.trigger_remove_columnar_compaction(&shard)
            }
            Some(CompactionPriority::UpdateVectorIndex {
                table_id,
                index_id,
                col_id,
                rebuild,
                ..
            }) => self.trigger_vector_index_update(&shard, table_id, index_id, col_id, rebuild),
            None => {
                info!("Shard {} is not urgent for compaction", tag);
                store_bool(&shard.compacting, false);
                None
            }
        }
    }

    pub(crate) fn set_alloc_ids_for_request(
        &self,
        req: &mut CompactionRequest,
        cur_num_files: usize,
        num_files_at_most: usize,
    ) {
        let ids = self
            .id_allocator
            .alloc_id(num_files_at_most * 2 + 16 + cur_num_files)
            .unwrap();
        req.file_ids = ids;
    }

    pub(crate) fn new_compact_request_with_shard(&self, shard: &Shard) -> CompactionRequest {
        let exported_encryption_key = shard
            .properties
            .get(ENCRYPTION_KEY)
            .unwrap_or_default()
            .to_vec();
        self.new_compact_request(
            shard.engine_id,
            shard.id,
            shard.ver,
            shard.range.clone(),
            exported_encryption_key,
        )
    }

    pub(crate) fn new_compact_request_with_meta(&self, meta: &ShardMeta) -> CompactionRequest {
        let exported_encryption_key = meta
            .properties
            .get(ENCRYPTION_KEY)
            .unwrap_or_default()
            .to_vec();
        self.new_compact_request(
            meta.engine_id,
            meta.id,
            meta.ver,
            meta.range.clone(),
            exported_encryption_key,
        )
    }

    fn new_compact_request(
        &self,
        engine_id: u64,
        shard_id: u64,
        shard_ver: u64,
        range: ShardRange,
        exported_encryption_key: Vec<u8>,
    ) -> CompactionRequest {
        CompactionRequest {
            engine_id,
            shard_id,
            shard_ver,
            outer_start: range.outer_start.to_vec(),
            outer_end: range.outer_end.to_vec(),
            inner_key_off: range.inner_key_off,
            file_ids: vec![],
            compaction_tp: CompactionType::Unknown,
            compactor_version: self.opts.compaction_request_version,
            safe_ts: self.get_keyspace_gc_safepoint_v2(range.keyspace_id),
            exported_encryption_key,
        }
    }

    fn destroy_range(&self, shard: &Shard) -> Result<Option<pb::ChangeSet>> {
        let del_prefixes = shard.get_del_prefixes();
        if del_prefixes.is_empty() {
            info!(
                "{} Engine::destroy_range: del_prefixes is empty, skip",
                shard.tag()
            );
            return Ok(None);
        }
        let data = shard.get_data();
        // Tables that full covered by delete-prefixes.
        let mut deletes = vec![];
        // Tables that partially covered by delete-prefixes.
        let mut overlaps = vec![];
        let mut col_overlaps = vec![];
        for t in &data.l0_tbls {
            if del_prefixes.cover_range(t.smallest(), t.biggest()) {
                let mut delete = pb::TableDelete::default();
                delete.set_id(t.id());
                delete.set_level(0);
                delete.set_cf(-1);
                deletes.push(delete);
            } else if del_prefixes
                .inner_delete_bounds()
                .any(|bound| t.has_data_in_bound(bound))
            {
                overlaps.push((t.id(), 0, -1));
            }
        }
        data.for_each_level(|cf, lh| {
            for t in lh.tables.as_slice() {
                if del_prefixes.cover_range(t.smallest(), t.biggest()) {
                    let mut delete = pb::TableDelete::default();
                    delete.set_id(t.id());
                    delete.set_level(lh.level as u32);
                    delete.set_cf(cf as i32);
                    deletes.push(delete);
                } else if del_prefixes
                    .inner_delete_bounds()
                    .any(|bound| t.has_overlap(bound))
                {
                    overlaps.push((t.id(), lh.level as u32, cf as i32));
                }
            }
            false
        });

        let mut columnar_deletes = vec![];
        data.for_each_columnar_level(|cl| {
            for t in cl.files.iter() {
                if del_prefixes.cover_range(t.get_smallest(), t.get_biggest()) {
                    let mut delete = pb::ColumnarDelete::default();
                    delete.set_id(t.id());
                    delete.set_level(cl.level as u32);
                    columnar_deletes.push(delete);
                } else if del_prefixes.inner_delete_bounds().any(|bound| {
                    let start = bound.lower_bound;
                    let trim_keyspace_start =
                        if shard.inner_key_off == 0 && start[0] == TXN_KEY_PREFIX {
                            &start[KEYSPACE_PREFIX_LEN..]
                        } else {
                            start.as_ref()
                        };
                    match decode_table_id(trim_keyspace_start) {
                        Ok(table_id) => {
                            trim_keyspace_start.len() == TABLE_PREFIX_LEN + ID_LEN
                                && t.has_table(table_id)
                        }
                        Err(_) => {
                            error!(
                                "{}, can not decode table_id from start key: {:?}",
                                shard.tag(),
                                trim_keyspace_start
                            );
                            false
                        }
                    }
                }) {
                    col_overlaps.push((t.id(), cl.level as u32));
                }
            }
            false
        });

        info!(
            "start destroying range for {}, {:?}, destroyed: {}, overlaps: {}, columnar overlaps: {}",
            shard.tag(),
            del_prefixes,
            deletes.len() + columnar_deletes.len(),
            overlaps.len(),
            col_overlaps.len(),
        );

        let mut cs = if overlaps.is_empty() && col_overlaps.is_empty() {
            let mut cs = pb::ChangeSet::default();
            cs.mut_destroy_range().set_table_deletes(deletes.into());
            cs
        } else {
            let mut req = self.new_compact_request_with_shard(shard);
            req.file_ids = self
                .id_allocator
                .alloc_id(overlaps.len() + col_overlaps.len())
                .unwrap();
            let in_place_compaction_type = InPlaceCompaction::DestroyRange(del_prefixes.marshal());
            let in_place_compaction = InPlaceCompactionCtx {
                file_ids: overlaps,
                col_file_ids: col_overlaps,
                block_size: self.opts.table_builder_options.block_size,
                columnar_build_opts: self.opts.columnar_build_options,
                schema_file_id: shard.get_schema_file().map(|f| f.get_file_id()),
                spec: in_place_compaction_type,
            };
            req.compaction_tp = CompactionType::InPlaceWithColumnar(in_place_compaction);
            let mut cs = self.comp_client.compact(req)?;
            let dr = cs.mut_destroy_range();
            deletes.extend(dr.take_table_deletes().into_iter());
            dr.set_table_deletes(deletes.into());
            cs
        };
        cs.set_shard_id(shard.id);
        cs.set_shard_ver(shard.ver);
        cs.set_property_key(DEL_PREFIXES_KEY.to_string());
        cs.set_property_value(del_prefixes.marshal());
        Ok(Some(cs))
    }

    pub(crate) fn truncate_ts(&self, shard: &Shard) -> Result<Option<pb::ChangeSet>> {
        self.truncate_with_ts(shard, shard.get_truncate_ts().unwrap())
    }

    // Also used by keyspace restore
    pub fn truncate_with_ts(
        &self,
        shard: &Shard,
        truncate_ts: TruncateTs,
    ) -> Result<Option<pb::ChangeSet>> {
        let data = shard.get_data();
        // TODO: record min_ts to directly delete a SSTable.

        let mut overlaps = vec![];
        let mut col_overlaps = vec![];
        for t in &data.l0_tbls {
            if truncate_ts.inner() < t.max_ts() {
                overlaps.push((t.id(), 0, -1));
            }
        }
        data.for_each_level(|cf, lh| {
            for t in lh.tables.iter() {
                if truncate_ts.inner() < t.max_ts {
                    overlaps.push((t.id(), lh.level as u32, cf as i32));
                }
            }
            false
        });

        data.for_each_columnar_level(|cl| {
            for t in cl.files.iter() {
                if truncate_ts.inner() < t.get_max_version() {
                    col_overlaps.push((t.id(), cl.level as u32));
                }
            }
            false
        });

        info!(
            "start truncate ts for {}, truncate_ts: {:?}, overlaps: {}, columnar overlaps: {}",
            shard.tag(),
            truncate_ts,
            overlaps.len(),
            col_overlaps.len(),
        );

        let mut cs = if overlaps.is_empty() && col_overlaps.is_empty() {
            let mut cs = pb::ChangeSet::default();
            // Must `set_truncate_ts` for checking the type of ChangeSet properly.
            cs.set_truncate_ts(pb::TableChange::default());
            cs
        } else {
            let mut req = self.new_compact_request_with_shard(shard);
            req.file_ids = self
                .id_allocator
                .alloc_id(overlaps.len() + col_overlaps.len())
                .unwrap();
            let in_place_compaction = InPlaceCompaction::TruncateTs(truncate_ts.inner());
            let in_place_compaction_ctx = InPlaceCompactionCtx {
                file_ids: overlaps,
                col_file_ids: col_overlaps,
                block_size: self.opts.table_builder_options.block_size,
                columnar_build_opts: self.opts.columnar_build_options,
                schema_file_id: shard.get_schema_file().map(|f| f.get_file_id()),
                spec: in_place_compaction,
            };
            req.compaction_tp = CompactionType::InPlaceWithColumnar(in_place_compaction_ctx);
            self.comp_client.compact(req)?
        };
        cs.set_shard_id(shard.id);
        cs.set_shard_ver(shard.ver);
        cs.set_property_key(TRUNCATE_TS_KEY.to_string());
        cs.set_property_value(truncate_ts.marshal().to_vec());
        Ok(Some(cs))
    }

    fn trim_over_bound(&self, shard: &Shard) -> Result<Option<pb::ChangeSet>> {
        let data = shard.get_data();
        if !shard.get_trim_over_bound() {
            // `data.trim_over_bound is possible to be false.
            // See https://github.com/tidbcloud/cloud-storage-engine/issues/663.
            info!(
                "{} shard.data.trim_over_bound is false, skip trim_over_bound",
                shard.tag()
            );
            return Ok(None);
        }

        // Tables that are entirely over bound.
        let mut deletes = vec![];
        // Tables that are partially over bound.
        let mut overlaps = vec![];
        let mut col_overlaps = vec![];
        let shard_bound = shard.data_bound();
        for t in &data.l0_tbls {
            let table_bound = t.data_bound();
            if !shard_bound.overlap_bound(table_bound) {
                // -----smallest-----biggest-----[start----------end)
                // [start----------end)-----smallest-----biggest-----
                let mut delete = pb::TableDelete::default();
                delete.set_id(t.id());
                delete.set_level(0);
                delete.set_cf(-1);
                deletes.push(delete);
            } else if !shard_bound.contains_bound(table_bound) {
                // -----smallest-----[start----------end)-----biggest-----
                overlaps.push((t.id(), 0, -1));
            }
        }
        data.for_each_level(|cf, lh| {
            for t in lh.tables.iter() {
                let table_bound = t.data_bound();
                if !shard_bound.overlap_bound(table_bound) {
                    // -----smallest-----biggest-----[start----------end)
                    // [start----------end)-----smallest-----biggest-----
                    let mut delete = pb::TableDelete::default();
                    delete.set_id(t.id());
                    delete.set_level(lh.level as u32);
                    delete.set_cf(cf as i32);
                    deletes.push(delete);
                } else if !shard_bound.contains_bound(table_bound) {
                    // -----smallest-----[start----------end)-----biggest-----
                    overlaps.push((t.id(), lh.level as u32, cf as i32));
                }
            }
            false
        });
        let mut columnar_deletes = vec![];
        data.for_each_columnar_level(|cl| {
            for t in cl.files.iter() {
                let table_bound = t.data_bound();
                if !shard_bound.overlap_bound(table_bound) {
                    // -----smallest-----biggest-----[start----------end)
                    // [start----------end)-----smallest-----biggest-----
                    let mut delete = pb::ColumnarDelete::default();
                    delete.set_id(t.id());
                    delete.set_level(cl.level as u32);
                    columnar_deletes.push(delete);
                } else if !shard_bound.contains_bound(table_bound) {
                    // -----smallest-----[start----------end)-----biggest-----
                    col_overlaps.push((t.id(), cl.level as u32));
                }
            }
            false
        });

        info!(
            "start trim_over_bound for {}, destroyed: {}, overlaps: {}, columnar overlaps: {}",
            shard.tag(),
            deletes.len(),
            overlaps.len(),
            col_overlaps.len(),
        );

        let mut cs = if overlaps.is_empty() && col_overlaps.is_empty() {
            let mut cs = pb::ChangeSet::default();
            cs.mut_trim_over_bound().set_table_deletes(deletes.into());
            cs
        } else {
            let mut req = self.new_compact_request_with_shard(shard);
            req.file_ids = self
                .id_allocator
                .alloc_id(overlaps.len() + col_overlaps.len())
                .unwrap();
            let inplace_compaction = InPlaceCompactionCtx {
                file_ids: overlaps,
                col_file_ids: col_overlaps,
                block_size: self.opts.table_builder_options.block_size,
                columnar_build_opts: self.opts.columnar_build_options,
                schema_file_id: shard.get_schema_file().map(|f| f.get_file_id()),
                spec: InPlaceCompaction::TrimOverBound,
            };
            req.compaction_tp = CompactionType::InPlaceWithColumnar(inplace_compaction);
            let mut cs = self.comp_client.compact(req)?;
            let tc = cs.mut_trim_over_bound();
            deletes.extend(tc.take_table_deletes().into_iter());
            tc.set_table_deletes(deletes.into());
            cs
        };
        cs.set_shard_id(shard.id);
        cs.set_shard_ver(shard.ver);
        cs.set_property_key(TRIM_OVER_BOUND.to_string());
        cs.set_property_value(TRIM_OVER_BOUND_DISABLE.to_vec());
        Ok(Some(cs))
    }

    pub fn trim_over_bound_by_meta(&self, meta: &ShardMeta) -> Result<pb::ChangeSet> {
        // Tables that are entirely over bound.
        let mut deletes = vec![];
        // Tables that are partially over bound.
        let mut over_bounds = vec![];
        for (&id, f) in &meta.files {
            let shard_bound = meta.range.data_bound();
            let file_bound = f.data_bound();
            if !shard_bound.overlap_bound(file_bound) {
                let mut delete = pb::TableDelete::default();
                delete.set_id(id);
                delete.set_level(f.level as u32);
                delete.set_cf(f.cf as i32);
                deletes.push(delete);
            } else if !shard_bound.contains_bound(file_bound) {
                over_bounds.push((id, f.level as u32, f.cf as i32));
            }
        }

        debug!(
            "start trim_over_bound_by_meta for {}:{}, destroyed: {}, overlapping: {}",
            meta.id,
            meta.ver,
            deletes.len(),
            over_bounds.len()
        );

        let mut res_cs = if over_bounds.is_empty() {
            let mut cs = pb::ChangeSet::default();
            if !deletes.is_empty() {
                cs.mut_trim_over_bound().set_table_deletes(deletes.into());
            }
            cs
        } else {
            let mut req = self.new_compact_request_with_meta(meta);
            req.file_ids = self.id_allocator.alloc_id(over_bounds.len()).unwrap();
            let in_place_compaction_ctx = InPlaceCompactionCtx {
                file_ids: over_bounds,
                col_file_ids: vec![],
                block_size: self.opts.table_builder_options.block_size,
                columnar_build_opts: self.opts.columnar_build_options,
                schema_file_id: Some(meta.schema_file_id),
                spec: InPlaceCompaction::TrimOverBound,
            };
            req.compaction_tp = CompactionType::InPlaceWithColumnar(in_place_compaction_ctx);
            let mut cs = self.comp_client.compact(req)?;
            let tc = cs.mut_trim_over_bound();
            deletes.extend(tc.take_table_deletes().into_iter());
            tc.set_table_deletes(deletes.into());
            cs
        };
        res_cs.set_shard_id(meta.id);
        res_cs.set_shard_ver(meta.ver);
        res_cs.set_property_key(TRIM_OVER_BOUND.to_string());
        res_cs.set_property_value(TRIM_OVER_BOUND_DISABLE.to_vec());
        Ok(res_cs)
    }

    fn get_blob_table_build_options_or_none(&self, shard: &Shard) -> Option<BlobTableBuildOptions> {
        if shard.keyspace_id > 0 {
            let conf = self.per_keyspace_configs.get(&shard.keyspace_id)?;
            if conf.enable_blob {
                return Some(self.opts.blob_table_build_options);
            }
        }
        None
    }

    pub(crate) fn trigger_l0_compaction(&self, shard: &Shard) -> Option<Result<pb::ChangeSet>> {
        let tag = shard.tag();
        let data = shard.get_data();
        if data.l0_tbls.is_empty() {
            info!("{} zero L0 tables", tag);
            store_bool(&shard.compacting, false);
            return None;
        }
        if let Some(res) = self.try_move_down_l0(shard, &data) {
            info!("{} move down l0", tag);
            return Some(Ok(res));
        }
        let mut req = self.new_compact_request_with_shard(shard);
        let mut l0_tbls = vec![];
        let mut multi_cfs_l1_tbls = vec![];
        let mut total_size = 0;
        let mut smallest = data.l0_tbls[0].smallest();
        let mut biggest = data.l0_tbls[0].biggest();
        let mut estimated_blob_size = 0;
        let columnar_snap_version = shard.get_columnar_snap_version();
        let unconverted_l0s: HashSet<u64> = data.get_unconverted_l0s().iter().cloned().collect();
        for l0 in &data.l0_tbls {
            // Skip unconverted l0s.
            // After region merge there may have l0s with l0.version() <
            // columnar_snap_version already converted exists and will not be added
            // to unconverted_l0s. So we need check the unconverted_l0s and
            // assert the l0.version().
            if unconverted_l0s.contains(&l0.id()) {
                info!("{} skip unconverted l0 {}", tag, l0.id());
                assert!(
                    l0.version() > columnar_snap_version,
                    "{} unconverted l0 {} snap_version: {} columnar_snap_version: {} l0 version: {}, unconverted_l0s: {:?}",
                    tag,
                    l0.id(),
                    shard.get_snap_version(),
                    columnar_snap_version,
                    l0.version(),
                    data.get_unconverted_l0s()
                );
                continue;
            }

            if smallest > l0.smallest() {
                smallest = l0.smallest();
            }
            if biggest < l0.biggest() {
                biggest = l0.biggest();
            }
            l0_tbls.push(l0.id());
            if l0.size() > l0.entries() * self.opts.blob_table_build_options.min_blob_size as u64 {
                estimated_blob_size += l0.size()
                    - l0.entries() * self.opts.blob_table_build_options.min_blob_size as u64;
            }
            total_size += l0.size();
        }
        if l0_tbls.is_empty() {
            store_bool(&shard.compacting, false);
            info!(
                "{} trigger_l0_compaction, none l0 table after filtering",
                tag
            );
            return None;
        }

        for cf in 0..NUM_CFS {
            let lh = data.get_cf(cf).get_level(1);
            let mut l1_tbls = vec![];
            for tbl in lh.tables.as_slice() {
                let non_overlapping = tbl.biggest() < smallest || tbl.smallest() > biggest;
                // Do not skip if there are too many L1 tables.
                if lh.tables.len() < MAX_SKIP_L1_TABLES && non_overlapping {
                    info!(
                        "{} skip L1 table {} for L0 compaction, tbl smallest {:?}, tbl biggest {:?}, L0 smallest {:?}, L0 biggest {:?}",
                        tag,
                        tbl.id(),
                        tbl.smallest(),
                        tbl.biggest(),
                        smallest,
                        biggest,
                    );
                    continue;
                }
                l1_tbls.push(tbl.id());
                total_size += tbl.size();
                if tbl.size()
                    > tbl.entries as u64 * self.opts.blob_table_build_options.min_blob_size as u64
                {
                    estimated_blob_size += tbl.size()
                        - tbl.entries as u64
                            * self.opts.blob_table_build_options.min_blob_size as u64;
                }
            }
            multi_cfs_l1_tbls.push(l1_tbls);
        }
        let sst_config = self.opts.table_builder_options;

        // Only opt-in users will use blob store.
        let mut bt_config = self.get_blob_table_build_options_or_none(shard);
        if let Some(build_opts) = &bt_config {
            if estimated_blob_size < build_opts.target_blob_table_size as u64 {
                bt_config = None;
            }
        }
        info!(
            "{} build blob in L0 compaction, estimated blob size {}, blob table build options {:?}",
            tag, estimated_blob_size, bt_config
        );
        let estimated_num_files = total_size as usize
            / bt_config.map_or(sst_config.max_table_size, |c| {
                std::cmp::min(c.max_blob_table_size, sst_config.max_table_size)
            });
        self.set_alloc_ids_for_request(
            &mut req,
            l0_tbls.len() + multi_cfs_l1_tbls.len(),
            estimated_num_files,
        );
        let l0_compaction = L0Compaction {
            safe_ts: self.get_keyspace_gc_safepoint_v2(shard.keyspace_id),
            l0_tables: l0_tbls,
            multi_cf_l1_tables: multi_cfs_l1_tbls,
            sst_config,
            bt_config,
        };
        req.compaction_tp = CompactionType::L0(l0_compaction);
        info!("start compact L0 for {}", tag);
        Some(self.comp_client.compact(req))
    }

    fn try_move_down_l0(&self, shard: &Shard, shard_data: &ShardData) -> Option<pb::ChangeSet> {
        if !shard_data.l0_tbls.iter().any(|l0| l0.is_write_cf_only()) {
            return None;
        }
        let mut move_down_l0s = vec![];
        let columnar_snap_version = shard.get_columnar_snap_version();
        let tag = shard.tag();
        let unconverted_l0s: HashSet<u64> =
            shard_data.get_unconverted_l0s().iter().cloned().collect();
        for (i, l0_tbl) in shard_data.l0_tbls.iter().enumerate() {
            if !l0_tbl.is_write_cf_only() {
                continue;
            }
            // Avoid unconverted l0s compaction.
            if unconverted_l0s.contains(&l0_tbl.id()) {
                info!("{} skip unconverted l0 {}", tag, l0_tbl.id());
                assert!(
                    l0_tbl.version() > columnar_snap_version,
                    "{} unconverted l0 {} snap_version: {} columnar_snap_version: {} l0 version: {}, unconverted_l0s: {:?}",
                    tag,
                    l0_tbl.id(),
                    shard.get_snap_version(),
                    columnar_snap_version,
                    l0_tbl.version(),
                    shard_data.get_unconverted_l0s()
                );
                continue;
            }
            let l0_write_cf_tbl = l0_tbl.get_cf(WRITE_CF).as_ref().unwrap();
            let l0_write_cf_bound = l0_write_cf_tbl.data_bound();
            let overlap_below_l0 = shard_data.l0_tbls[i + 1..].iter().any(|below_l0| {
                if let Some(below_write_cf) = below_l0.get_cf(WRITE_CF) {
                    return below_write_cf.data_bound().overlap_bound(l0_write_cf_bound);
                }
                false
            });
            if overlap_below_l0 {
                continue;
            }
            let l1 = shard_data.cfs[WRITE_CF].get_level(1);
            if l0_write_cf_bound
                .find_overlap(l1.tables.as_slice(), true)
                .is_some()
            {
                continue;
            }
            let mut table_create = pb::TableCreate::default();
            table_create.set_id(l0_tbl.id());
            table_create.set_level(1);
            table_create.set_cf(WRITE_CF as i32);
            table_create.set_smallest(l0_write_cf_tbl.smallest().to_vec());
            table_create.set_biggest(l0_write_cf_tbl.biggest().to_vec());
            table_create.set_meta_offset(l0_write_cf_tbl.meta_offset());
            move_down_l0s.push(table_create);
        }
        if move_down_l0s.is_empty() {
            info!("{} no l0 to move down", tag);
            return None;
        }
        let mut cs = new_change_set(shard.id, shard.ver);
        let comp = cs.mut_compaction();
        comp.set_cf(-1);
        comp.set_level(0);
        comp.set_top_deletes(
            move_down_l0s
                .iter()
                .map(|tbl_create| tbl_create.id)
                .collect(),
        );
        comp.set_table_creates(move_down_l0s.into());
        Some(cs)
    }

    fn trigger_l1_plus_compaction(
        &self,
        shard: &Shard,
        cf: isize,
        level: usize,
    ) -> Option<Result<pb::ChangeSet>> {
        let tag = shard.tag();
        let data = shard.get_data();
        let scf = data.get_cf(cf as usize);
        let upper_level = &scf.levels[level - 1];
        let lower_level = &scf.levels[level];
        if upper_level.tables.len() == 0 {
            store_bool(&shard.compacting, false);
            info!("{} no upper level {} table", tag, level - 1);
            return None;
        }
        let shard_bound = shard.data_bound();
        let upper_level_bound = upper_level.get_data_bound().unwrap();
        let upper_level_candidates = if !shard_bound.contains_bound(upper_level_bound) {
            if upper_level.tables.first().unwrap().smallest() < shard_bound.lower_bound {
                Arc::new(vec![upper_level.tables.first().unwrap().clone()])
            } else {
                Arc::new(vec![upper_level.tables.last().unwrap().clone()])
            }
        } else {
            upper_level.tables.clone()
        };

        let sum_tbl_size = |tbls: &[sstable::SsTable]| tbls.iter().map(|tbl| tbl.size()).sum();

        let calc_ratio = |upper_size: u64, lower_size: u64| {
            if lower_size == 0 {
                return upper_size as f64;
            }
            upper_size as f64 / lower_size as f64
        };
        // First pick one table has max topSize/bottomSize ratio.
        let mut candidate_ratio = 0f64;
        let mut upper_left_idx = 0;
        let mut upper_right_idx = 0;
        let mut upper_size = 0;
        let mut lower_left_idx = 0;
        let mut lower_right_idx = 0;
        let mut lower_size = 0;
        for (i, tbl) in upper_level_candidates.iter().enumerate() {
            let (left, right) = tbl.data_bound().get_overlap_data_sets(&lower_level.tables);
            let new_lower_size: u64 = sum_tbl_size(&lower_level.tables[left..right]);
            let ratio = calc_ratio(tbl.size(), new_lower_size);
            if candidate_ratio < ratio {
                candidate_ratio = ratio;
                upper_left_idx = i;
                upper_right_idx = i + 1;
                upper_size = tbl.size();
                lower_left_idx = left;
                lower_right_idx = right;
                lower_size = new_lower_size;
            }
        }
        if upper_left_idx == upper_right_idx {
            store_bool(&shard.compacting, false);
            info!("{} no candidate for l1_plus_compaction", tag);
            return None;
        }
        // Expand to left to include more tops as long as the ratio doesn't decrease and
        // the total size do not exceed maxCompactionExpandSize.
        let cur_upper_left_idx = upper_left_idx;
        for i in (0..cur_upper_left_idx).rev() {
            let t = &upper_level_candidates[i];
            let (left, right) = t.data_bound().get_overlap_data_sets(&lower_level.tables);
            if right < lower_left_idx {
                // A bottom table is skipped, we can compact in another run.
                break;
            }
            let new_upper_size = t.size() + upper_size;
            let new_lower_size =
                sum_tbl_size(&lower_level.tables[left..lower_left_idx]) + lower_size;
            let new_ratio = calc_ratio(new_upper_size, new_lower_size);
            if new_ratio > candidate_ratio {
                upper_left_idx -= 1;
                lower_left_idx = left;
                upper_size = new_upper_size;
                lower_size = new_lower_size;
            } else {
                break;
            }
        }
        // Expand to right to include more tops as long as the ratio doesn't decrease
        // and the total size do not exceeds maxCompactionExpandSize.
        let cur_upper_right_idx = upper_right_idx;
        for i in cur_upper_right_idx..upper_level_candidates.len() {
            let t = &upper_level_candidates[i];
            let (left, right) = t.data_bound().get_overlap_data_sets(&lower_level.tables);
            if left > lower_right_idx {
                // A bottom table is skipped, we can compact in another run.
                break;
            }
            let new_upper_size = t.size() + upper_size;
            let new_lower_size =
                sum_tbl_size(&lower_level.tables[lower_right_idx..right]) + lower_size;
            let new_ratio = calc_ratio(new_upper_size, new_lower_size);
            if new_ratio > candidate_ratio
                && (new_upper_size + new_lower_size) < MAX_COMPACTION_EXPAND_SIZE
            {
                upper_right_idx += 1;
                lower_right_idx = right;
                upper_size = new_upper_size;
                lower_size = new_lower_size;
            } else {
                break;
            }
        }
        let upper_level_table_ids: Vec<_> = upper_level_candidates[upper_left_idx..upper_right_idx]
            .iter()
            .map(|t| t.id())
            .collect();
        let lower_level_table_ids: Vec<_> = lower_level.tables[lower_left_idx..lower_right_idx]
            .iter()
            .map(|t| t.id())
            .collect();
        if lower_level_table_ids.is_empty() && cf as usize == WRITE_CF {
            info!("move down L{} CF{} for {}", level, cf, tag);
            // Move down. only write CF benefits from this optimization.
            let mut comp = pb::Compaction::new();
            comp.set_cf(cf as i32);
            comp.set_level(level as u32);
            comp.set_top_deletes(upper_level_table_ids);
            let tbl_creates = upper_level_candidates[upper_left_idx..upper_right_idx]
                .iter()
                .map(|top_tbl| {
                    let mut tbl_create = pb::TableCreate::new();
                    tbl_create.set_id(top_tbl.id());
                    tbl_create.set_cf(cf as i32);
                    tbl_create.set_level(level as u32 + 1);
                    tbl_create.set_smallest(top_tbl.smallest().to_vec());
                    tbl_create.set_biggest(top_tbl.biggest().to_vec());
                    tbl_create.set_meta_offset(top_tbl.meta_offset());
                    tbl_create
                })
                .collect::<Vec<_>>();
            comp.set_table_creates(tbl_creates.into());
            let mut cs = new_change_set(shard.id, shard.ver);
            cs.set_compaction(comp);
            return Some(Ok(cs));
        }
        let mut has_overlap = false;
        let mut kr = KeyRange::default();
        // The range for overlapping check should include both upper level and lower
        // level.
        kr.update(&upper_level_candidates[upper_left_idx..upper_right_idx]);
        kr.update(&lower_level.tables[lower_left_idx..lower_right_idx]);
        for lvl_idx in (level + 1)..scf.levels.len() {
            let lh = &scf.levels[lvl_idx];
            let (left, right) = kr.data_bound().get_overlap_data_sets(&lh.tables);
            if left < right {
                has_overlap = true;
            }
        }
        let mut req = self.new_compact_request_with_shard(shard);
        let sst_config = self.opts.table_builder_options;
        let estimated_num_files = (upper_size + lower_size) as usize / sst_config.max_table_size;
        self.set_alloc_ids_for_request(
            &mut req,
            upper_level_table_ids.len() + lower_level_table_ids.len(),
            estimated_num_files,
        );
        let l1_plus: L1PlusCompaction = L1PlusCompaction {
            cf,
            level,
            safe_ts: self.get_keyspace_gc_safepoint_v2(shard.keyspace_id),
            keep_latest_obsolete_tombstone: has_overlap,
            upper_level: upper_level_table_ids,
            lower_level: lower_level_table_ids,
            sst_config,
        };
        req.compaction_tp = CompactionType::L1Plus(l1_plus);
        info!(
            "start compact L{} CF{} for {}, num_ids: {}, input_size: {}",
            level,
            cf,
            tag,
            req.file_ids.len(),
            upper_size + lower_size,
        );
        Some(self.comp_client.compact(req))
    }

    pub(crate) fn trigger_major_compaction(&self, shard: &Shard) -> Option<Result<pb::ChangeSet>> {
        if self.opts.compaction_request_version < MAJOR_COMPACTION_MIN_REQUEST_VERSION {
            return Some(Err(CompactionNotRetryable(format!(
                "trigger_major_compaction: compaction_request_version must >= {}",
                MAJOR_COMPACTION_MIN_REQUEST_VERSION
            ))));
        }
        let data = shard.get_data();
        // We checked the unconverted_l0s in `refresh_compaction_priority`, but new l0s
        // may be added after the check. The comapct is running in the background, so we
        // need to check it again.
        if data.has_unconverted_l0s() {
            info!(
                "{} trigger_major_compaction skipped due to unconverted_l0s {:?}",
                shard.tag(),
                data.get_unconverted_l0s()
            );
            store_bool(&shard.compacting, false);
            return None;
        }
        let mut req = self.new_compact_request_with_shard(shard);
        let mut ln_tables: HashMap<usize, Vec<(usize, Vec<u64>)>> = HashMap::new();
        let mut total_size = 0;
        let mut num_ln_files = 0;
        for (cf, shard_cf) in data.cfs.iter().enumerate() {
            let mut ssts_for_cf = vec![];
            for lh in &shard_cf.levels {
                if lh.tables.is_empty() {
                    continue;
                }
                num_ln_files += lh.tables.len();
                ssts_for_cf.push((lh.level, lh.tables.iter().map(|t| t.id()).collect()));
                total_size += lh.tables.iter().map(|t| t.size()).sum::<u64>();
            }
            ln_tables.insert(cf, ssts_for_cf);
        }
        total_size += data.l0_tbls.iter().map(|t| t.size()).sum::<u64>();
        total_size += data.blob_tbl_map.values().map(|t| t.size()).sum::<u64>();
        let sst_config = self.opts.table_builder_options;
        let bt_config = self.get_blob_table_build_options_or_none(shard);
        let estimated_num_files = total_size as usize
            / std::cmp::min(
                bt_config.map_or(usize::MAX, |bt_config| bt_config.max_blob_table_size),
                sst_config.max_table_size,
            );
        self.set_alloc_ids_for_request(
            &mut req,
            data.l0_tbls.len() + num_ln_files + data.blob_tbl_map.len(),
            estimated_num_files,
        );
        let major_compaction = MajorCompaction {
            safe_ts: self.get_keyspace_gc_safepoint_v2(shard.keyspace_id),
            l0_tables: data.l0_tbls.iter().map(|t| t.id()).collect(),
            ln_tables,
            blob_tables: data.blob_tbl_map.keys().copied().collect(),
            sst_config,
            bt_config,
        };
        req.compaction_tp = CompactionType::Major(major_compaction);
        info!(
            "start major compact for {}, num_ids: {}, input_size: {}",
            shard.tag(),
            req.file_ids.len(),
            total_size,
        );
        Some(self.comp_client.compact(req))
    }

    pub(crate) fn trigger_remove_columnar_compaction(
        &self,
        shard: &Shard,
    ) -> Option<Result<pb::ChangeSet>> {
        let tag = shard.tag();
        let data = shard.get_data();
        let mut old_columnar_tables = vec![];
        data.col_levels.levels.iter().for_each(|l| {
            l.files.iter().for_each(|f| {
                old_columnar_tables.push((l.level, f.id()));
            })
        });
        let mut req = self.new_compact_request_with_shard(shard);
        let columnar_major_compaction = ColumnarMajorCompaction {
            safe_ts: 0,
            l0_tables: vec![],
            ln_tables: vec![],
            blob_tables: vec![],
            old_columnar_tables,
            schema_file_id: 0,
            snap_version: 0,
            columnar_config: self.opts.columnar_build_options,
        };
        req.compaction_tp = CompactionType::ColumnarMajor(columnar_major_compaction);
        info!("start remove columnar for {}", tag);
        Some(self.comp_client.compact(req))
    }

    pub(crate) fn trigger_l0_to_columnar(&self, shard: &Shard) -> Option<Result<pb::ChangeSet>> {
        let tag = shard.tag();
        let data = shard.get_data();
        if data.col_levels.unconverted_l0s.is_empty() {
            info!("{} no unconverted l0s to convert to columnar", tag);
            store_bool(&shard.compacting, false);
            return None;
        }
        let schema_file = shard.get_schema_file()?;
        let mut req = self.new_compact_request_with_shard(shard);
        let mut source_row_tables = vec![];
        let mut snap_version = 0;
        for l0 in &data.col_levels.unconverted_l0s {
            source_row_tables.push((0, l0.id()));
            snap_version = snap_version.max(l0.version());
        }
        let num_l0s = source_row_tables.len();
        let columnar_compaction = ColumnarCompaction {
            level: 0,
            safe_ts: self.get_keyspace_gc_safepoint_v2(shard.keyspace_id),
            snap_version,
            source_row_files: source_row_tables,
            source_columnar_files: vec![],
            schema_file_id: schema_file.get_file_id(),
            columnar_config: self.opts.columnar_build_options,
        };
        req.compaction_tp = CompactionType::Columnar(columnar_compaction);
        self.set_alloc_ids_for_request(&mut req, num_l0s, num_l0s);
        info!(
            "start covert L0 to columnar for {}, num_l0s {}",
            tag, num_l0s
        );
        Some(self.comp_client.compact(req))
    }

    pub(crate) fn trigger_columnar_major_compaction(
        &self,
        shard: &Shard,
    ) -> Option<Result<pb::ChangeSet>> {
        info!("trigger_major_compaction for {}", shard.tag());
        let mut req = self.new_compact_request_with_shard(shard);
        let data = shard.get_data();
        if data.schema_file.is_none() {
            store_bool(&shard.compacting, false);
            warn!(
                "trigger_major_compaction: no schema file found for {}, skip",
                shard.tag()
            );
            return None;
        }
        let schema_file_id = shard.get_data().schema_file.as_ref().unwrap().get_file_id();
        let mut total_size = 0;
        let mut num_ln_files = 0;
        let shard_cf = data.get_cf(WRITE_CF);
        let mut ln_tables = vec![];
        for lh in &shard_cf.levels {
            if lh.tables.is_empty() {
                continue;
            }
            num_ln_files += lh.tables.len();
            ln_tables.push((lh.level, lh.tables.iter().map(|t| t.id()).collect()));
            total_size += lh.tables.iter().map(|t| t.size()).sum::<u64>();
        }

        let mut old_columnar_tables = vec![];

        data.col_levels.levels.iter().for_each(|col_level| {
            col_level.files.iter().for_each(|table| {
                old_columnar_tables.push((col_level.level, table.get_file().id()));
            });
        });

        let snap_version = data
            .l0_tbls
            .iter()
            .map(|t| t.version())
            .max()
            .unwrap_or(INITIAL_SNAP_VERSION);
        total_size += data.l0_tbls.iter().map(|t| t.size()).sum::<u64>();
        total_size += data.blob_tbl_map.values().map(|t| t.size()).sum::<u64>();
        let columnar_config = self.opts.columnar_build_options;
        let estimated_num_files = total_size as usize / columnar_config.max_columnar_table_size;
        self.set_alloc_ids_for_request(
            &mut req,
            data.l0_tbls.len() + num_ln_files + data.blob_tbl_map.len(),
            estimated_num_files,
        );
        let major_compaction = ColumnarMajorCompaction {
            safe_ts: self.get_keyspace_gc_safepoint_v2(shard.keyspace_id),
            l0_tables: data.l0_tbls.iter().map(|t| t.id()).collect(),
            ln_tables,
            blob_tables: data.blob_tbl_map.keys().copied().collect(),
            old_columnar_tables,
            schema_file_id,
            columnar_config,
            snap_version,
        };
        req.compaction_tp = CompactionType::ColumnarMajor(major_compaction);
        info!(
            "start major compact for columnar {}, num_ids: {}, input_size: {}",
            shard.tag(),
            req.file_ids.len(),
            total_size,
        );

        Some(self.comp_client.compact(req))
    }

    pub(crate) fn trigger_columnar_l0_compaction(
        &self,
        shard: &Shard,
    ) -> Option<Result<pb::ChangeSet>> {
        let tag = shard.tag();
        let data = shard.get_data();
        let l0_tbls = &data.col_levels.levels[0].files;
        if l0_tbls.is_empty() {
            info!(
                "{} no columnar base level tables, skip trigger_columnar_l0_compaction",
                tag
            );
            store_bool(&shard.compacting, false);
            return None;
        }
        if data.schema_file.is_none() {
            info!(
                "{} no schema file found, skip trigger_columnar_l0_compaction",
                tag
            );
            store_bool(&shard.compacting, false);
            return None;
        }
        let mut req = self.new_compact_request_with_shard(shard);
        let mut total_size = 0;
        let mut smallest = l0_tbls[0].get_smallest();
        let mut biggest = l0_tbls[0].get_biggest();
        let mut l0_tbl_ids = Vec::with_capacity(l0_tbls.len());
        let mut snap_version = 0;
        for col in l0_tbls {
            if smallest > col.get_smallest() {
                smallest = col.get_smallest();
            }
            if biggest < col.get_biggest() {
                biggest = col.get_biggest();
            }
            let l0_version = col.get_l0_version().unwrap();
            snap_version = snap_version.max(l0_version);
            l0_tbl_ids.push((0, col.get_file().id()));
            total_size += col.get_file().size();
        }

        let schema_file_id = data.schema_file.as_ref().unwrap().get_file_id();
        let columnar_config = self.opts.columnar_build_options;
        let estimated_num_files = total_size as usize / columnar_config.max_columnar_table_size;
        self.set_alloc_ids_for_request(&mut req, l0_tbl_ids.len(), estimated_num_files);
        let col_compaction = ColumnarCompaction {
            level: 0,
            safe_ts: self.get_keyspace_gc_safepoint_v2(shard.keyspace_id),
            snap_version,
            source_row_files: vec![],
            source_columnar_files: l0_tbl_ids,
            schema_file_id,
            columnar_config,
        };
        req.compaction_tp = CompactionType::Columnar(col_compaction);
        info!("start columnar compact L0 for {}", tag);
        Some(self.comp_client.compact(req))
    }

    pub(crate) fn trigger_columnar_l1_compaction(
        &self,
        shard: &Shard,
    ) -> Option<Result<pb::ChangeSet>> {
        let tag = shard.tag();
        let data = shard.get_data();
        let level = 1;
        let l1_tbls = &data.col_levels.levels[1].files;
        if l1_tbls.is_empty() {
            info!(
                "{} no columnar level 1 tables, skip trigger_columnar_l1_compaction",
                tag
            );
            store_bool(&shard.compacting, false);
            return None;
        }
        if data.schema_file.is_none() {
            info!(
                "{} no schema file found, skip trigger_columnar_l1_compaction",
                tag
            );
            store_bool(&shard.compacting, false);
            return None;
        }

        let mut total_size = 0;

        let mut smallest = l1_tbls[0].get_smallest();
        let mut biggest = l1_tbls[0].get_biggest();
        let mut l1_tbl_ids = Vec::with_capacity(l1_tbls.len());
        let mut snap_version = 0;
        for col in l1_tbls {
            if smallest > col.get_smallest() {
                smallest = col.get_smallest();
            }
            if biggest < col.get_biggest() {
                biggest = col.get_biggest();
            }
            let l0_version = col.get_l0_version().unwrap();
            snap_version = snap_version.max(l0_version);
            l1_tbl_ids.push((1, col.get_file().id()));
            total_size += col.get_file().size();
        }

        let mut l2_tbl_ids = vec![];
        let l2_tbls = &data.col_levels.levels[2].files;
        for tbl in l2_tbls {
            if tbl.get_biggest() < smallest || tbl.get_smallest() > biggest {
                info!(
                    "{} skip columnar table {} for compaction, level {}, tbl smallest {:?}, tbl biggest {:?}, base smallest {:?}, base biggest {:?}",
                    tag,
                    tbl.get_file().id(),
                    level,
                    tbl.get_smallest(),
                    tbl.get_biggest(),
                    smallest,
                    biggest,
                );
                continue;
            }
            l2_tbl_ids.push((2, tbl.get_file().id()));
            total_size += tbl.get_file().size();
        }

        let mut req = self.new_compact_request_with_shard(shard);

        let columnar_config = self.opts.columnar_build_options;
        let estimated_num_files = total_size as usize / columnar_config.max_columnar_table_size;
        self.set_alloc_ids_for_request(
            &mut req,
            l1_tbl_ids.len() + l2_tbl_ids.len(),
            estimated_num_files,
        );
        let schema_file_id = data.schema_file.as_ref().unwrap().get_file_id();
        let columnar_compaction = ColumnarCompaction {
            level,
            safe_ts: self.get_keyspace_gc_safepoint_v2(shard.keyspace_id),
            snap_version,
            source_row_files: vec![],
            source_columnar_files: [l1_tbl_ids, l2_tbl_ids].concat(),
            schema_file_id,
            columnar_config,
        };
        req.compaction_tp = CompactionType::Columnar(columnar_compaction);
        info!(
            "start compact columnar L{} for {}, num_ids: {}, input_size: {}",
            level,
            tag,
            req.file_ids.len(),
            total_size,
        );
        Some(self.comp_client.compact(req))
    }

    pub(crate) fn trigger_vector_index_update(
        &self,
        shard: &Shard,
        table_id: i64,
        index_id: i64,
        col_id: i64,
        rebuild: bool,
    ) -> Option<Result<pb::ChangeSet>> {
        let tag = shard.tag();
        info!("{} trigger vector index update", tag);
        let data = shard.get_data();
        let snap_version = shard.get_columnar_snap_version();
        let schema_file = data.schema_file.as_ref()?;
        let table_schema = schema_file.get_table(table_id)?;
        table_schema
            .vector_indexes
            .iter()
            .find(|idx| idx.index_id == index_id && idx.col_id == col_id)?;
        let mut remove_file_ids = vec![];
        let old_idx_ver =
            if let Some(old_vec_idx) = data.vector_indexes.get(table_id, index_id, col_id) {
                if rebuild {
                    for file in &old_vec_idx.files {
                        remove_file_ids.push(file.file_id());
                    }
                    0
                } else {
                    old_vec_idx.snap_version()
                }
            } else {
                0
            };
        let mut col_file_ids = vec![];
        for col_lvl in &data.col_levels.levels {
            for col_file in &col_lvl.files {
                let l0_version = col_file.get_l0_version().unwrap_or(1);
                if col_file.has_table(table_id) && l0_version > old_idx_ver {
                    col_file_ids.push((col_file.get_file().id(), col_lvl.level as u32));
                }
            }
        }
        let mut req = self.new_compact_request_with_shard(shard);
        req.compaction_tp = CompactionType::VectorIndex(VectorIndexUpdate {
            table_id,
            index_id,
            col_id,
            snap_version,
            schema_file_id: schema_file.get_file_id(),
            col_file_ids,
            remove_file_ids,
        });
        Some(self.comp_client.compact(req))
    }

    pub(crate) fn handle_compact_response(&self, cs: pb::ChangeSet) {
        self.meta_change_listener.on_change_set(cs);
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) enum CompactionPriority {
    L0 {
        score: f64,
    },
    L1Plus {
        cf: isize,
        score: f64,
        level: usize,
    },
    Major {
        score: f64,
    },
    DestroyRange,
    TruncateTs,
    TrimOverBound,
    L0ToColumnar,
    ColumnarL0 {
        score: f64,
    },
    ColumnarL1 {
        score: f64,
    },
    ColumnarMajor {
        score: f64,
    },
    ColumnarClear,
    UpdateVectorIndex {
        score: f64,
        table_id: i64,
        index_id: i64,
        col_id: i64,
        rebuild: bool,
    },
}

impl CompactionPriority {
    pub(crate) fn score(&self) -> f64 {
        match self {
            CompactionPriority::L0 { score } => *score,
            CompactionPriority::L1Plus { score, .. } => *score,
            CompactionPriority::Major { score } => *score,
            CompactionPriority::DestroyRange => f64::MAX,
            CompactionPriority::TruncateTs => f64::MAX,
            CompactionPriority::TrimOverBound => f64::MAX,
            CompactionPriority::L0ToColumnar => 2.0,
            CompactionPriority::ColumnarL0 { score } => *score,
            CompactionPriority::ColumnarL1 { score } => *score,
            CompactionPriority::ColumnarMajor { score } => *score,
            CompactionPriority::ColumnarClear => f64::MAX,
            CompactionPriority::UpdateVectorIndex { score, .. } => *score,
        }
    }

    pub(crate) fn level(&self) -> isize {
        match self {
            CompactionPriority::L0 { .. } => 0,
            CompactionPriority::L1Plus { level, .. } => *level as isize,
            CompactionPriority::Major { .. } => -1,
            CompactionPriority::DestroyRange => -1,
            CompactionPriority::TruncateTs => -1,
            CompactionPriority::TrimOverBound => -1,
            CompactionPriority::L0ToColumnar => 0,
            CompactionPriority::ColumnarL0 { .. } => 0,
            CompactionPriority::ColumnarL1 { .. } => 1,
            CompactionPriority::ColumnarMajor { .. } => -1,
            CompactionPriority::ColumnarClear => -1,
            CompactionPriority::UpdateVectorIndex { .. } => -1,
        }
    }

    pub(crate) fn cf(&self) -> isize {
        match self {
            CompactionPriority::L0 { .. } => -1,
            CompactionPriority::L1Plus { cf, .. } => *cf,
            CompactionPriority::Major { .. } => -1,
            CompactionPriority::DestroyRange => -1,
            CompactionPriority::TruncateTs => -1,
            CompactionPriority::TrimOverBound => -1,
            CompactionPriority::L0ToColumnar => 0,
            CompactionPriority::ColumnarL0 { .. } => -1,
            CompactionPriority::ColumnarL1 { .. } => -1,
            CompactionPriority::ColumnarMajor { .. } => -1,
            CompactionPriority::ColumnarClear => -1,
            CompactionPriority::UpdateVectorIndex { .. } => -1,
        }
    }
}

impl Ord for CompactionPriority {
    fn cmp(&self, other: &Self) -> CmpOrdering {
        self.score()
            .partial_cmp(&other.score())
            .unwrap_or(CmpOrdering::Equal)
    }
}

impl PartialOrd for CompactionPriority {
    fn partial_cmp(&self, other: &Self) -> Option<CmpOrdering> {
        self.score().partial_cmp(&other.score())
    }
}

impl PartialEq for CompactionPriority {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == CmpOrdering::Equal
    }
}

impl Eq for CompactionPriority {}

#[derive(Clone, Default)]
pub(crate) struct KeyRange {
    pub left: Bytes,
    pub right: Bytes,
}

impl KeyRange {
    pub(crate) fn left_key(&self) -> InnerKey<'_> {
        InnerKey::from_inner_buf(&self.left)
    }

    pub(crate) fn right_key(&self) -> InnerKey<'_> {
        InnerKey::from_inner_buf(&self.right)
    }

    pub(crate) fn update(&mut self, tables: &[SsTable]) {
        if tables.is_empty() {
            return;
        }
        let lower_smallest = tables.first().unwrap().smallest();
        if self.left.is_empty() || lower_smallest < self.left_key() {
            self.left = Bytes::copy_from_slice(lower_smallest.deref());
        }
        let lower_biggest = tables.last().unwrap().biggest();
        if lower_biggest > self.right_key() {
            self.right = Bytes::copy_from_slice(lower_biggest.deref());
        }
    }
}

impl BoundedDataSet for KeyRange {
    fn data_bound(&self) -> DataBound<'_> {
        DataBound::new(self.left_key(), self.right_key(), true)
    }
}

fn load_table_files(
    tbl_ids: &[u64],
    fs: Arc<dyn dfs::Dfs>,
    opts: dfs::Options,
    local_dir: Option<&PathBuf>,
    for_restore: bool,
) -> Result<Vec<Arc<dyn File>>> {
    if local_dir.is_none() {
        return load_table_files_from_dfs(tbl_ids, fs, opts);
    }

    let (mut files_loaded, files_failed) =
        load_table_files_from_local(tbl_ids, local_dir.unwrap(), for_restore, opts);
    if !files_failed.is_empty() {
        files_loaded.extend(load_table_files_from_dfs(&files_failed, fs, opts)?);
    }
    Ok(files_loaded)
}

fn load_table_files_from_local(
    tbl_ids: &[u64],
    local_dir: &Path,
    for_restore: bool,
    opts: dfs::Options,
) -> (
    Vec<Arc<dyn File>>, // files_loaded
    Vec<u64>,           // files_failed
) {
    let mut files_loaded: Vec<Arc<dyn File>> = vec![];
    let mut files_failed: Vec<u64> = vec![];
    for &id in tbl_ids {
        let file_name = match opts.file_type {
            FileType::Sst => new_sst_filename(id),
            FileType::Schema => new_schema_filename(id),
            FileType::Columnar => new_columnar_filename(id),
            FileType::TxnChunk => unreachable!("txn chunk should not be loaded from local"),
            FileType::VectorIndex => new_vector_index_filename(id),
            FileType::Blob => new_blob_filename(id),
        };
        let file_path = local_dir.join(file_name);
        match LocalFile::open(id, file_path.as_path(), false) {
            Ok(f) => files_loaded.push(Arc::new(f)),
            Err(e) => {
                files_failed.push(id);
                if !for_restore {
                    warn!(
                        "load_table_files_from_local: failed to open {}: {:?}",
                        id, e
                    );
                    metrics::ENGINE_LOAD_TABLE_FILES_ERROR
                        .with_label_values(&["compaction"])
                        .inc();
                }
            }
        }
    }
    (files_loaded, files_failed)
}

fn load_table_files_from_dfs(
    tbl_ids: &[u64],
    fs: Arc<dyn dfs::Dfs>,
    opts: dfs::Options,
) -> Result<Vec<Arc<dyn File>>> {
    let mut files = vec![];
    let (tx, rx) = std::sync::mpsc::sync_channel::<Result<(u64, Bytes)>>(tbl_ids.len());
    for id in tbl_ids {
        let aid = *id;
        let atx = tx.clone();
        let afs = fs.clone();
        fs.get_runtime().spawn(async move {
            let res = afs
                .read_file(aid, opts)
                .await
                .map(|data| (aid, data))
                .map_err(|e| Error::DfsError(e));
            atx.send(res).map_err(|_| "send file data failed").unwrap();
        });
    }
    let mut errors = vec![];
    for _ in tbl_ids {
        match rx.recv().unwrap() {
            Err(err) => errors.push(err),
            Ok((id, data)) => {
                let file: Arc<dyn File> = Arc::new(InMemFile::new(id, data));
                files.push(file);
            }
        }
    }
    if !errors.is_empty() {
        return Err(errors.pop().unwrap());
    }
    Ok(files)
}

fn files_to_l0_tables(
    files: Vec<Arc<dyn File>>,
    encryption_key: Option<EncryptionKey>,
) -> Vec<sstable::L0Table> {
    files
        .into_iter()
        .map(|f| {
            sstable::L0Table::new(f, BlockCache::None, false, encryption_key.clone())
                .unwrap()
                .unwrap()
        })
        .collect()
}

fn files_to_tables(
    files: Vec<Arc<dyn File>>,
    encryption_key: Option<EncryptionKey>,
) -> Vec<sstable::SsTable> {
    files
        .into_iter()
        .map(|f| sstable::SsTable::new(f, BlockCache::None, encryption_key.clone()).unwrap())
        .collect()
}

fn files_to_columnar_tables(files: Vec<Arc<dyn File>>) -> Vec<ColumnarFile> {
    files
        .into_iter()
        .map(|f| ColumnarFile::open(f).unwrap())
        .collect()
}

fn load_blob_tables(
    fs: Arc<dyn dfs::Dfs>,
    blob_table_ids: &[u64],
    opts: dfs::Options,
) -> Result<HashMap<u64, BlobTable>> {
    let mut blob_tables = HashMap::new();
    let (tx, rx) = std::sync::mpsc::sync_channel::<Result<(u64, Bytes)>>(blob_table_ids.len());
    for id in blob_table_ids {
        let aid = *id;
        let atx = tx.clone();
        let afs = fs.clone();
        fs.get_runtime().spawn(async move {
            let res = afs
                .read_file(aid, opts)
                .await
                .map(|data| (aid, data))
                .map_err(|e| Error::DfsError(e));
            atx.send(res).map_err(|_| "send file data failed").unwrap();
        });
    }
    let mut errors = vec![];
    for _ in blob_table_ids {
        match rx.recv().unwrap() {
            Err(err) => errors.push(err),
            Ok((id, data)) => {
                blob_tables.insert(id, BlobTable::from_bytes(data)?);
            }
        }
    }
    if !errors.is_empty() {
        return Err(errors.pop().unwrap());
    }
    Ok(blob_tables)
}

pub async fn handle_remote_compaction(
    dfs: Arc<dyn dfs::Dfs>,
    req: hyper::Request<hyper::Body>,
    compression_lvl: i32,
    checksum_type: ChecksumType,
    id_allocator: Arc<dyn IdAllocator>,
    master_key: MasterKey,
) -> hyper::Result<hyper::Response<hyper::Body>> {
    let req_body = hyper::body::to_bytes(req.into_body()).await?;
    let result = serde_json::from_slice(req_body.chunk());
    if result.is_err() {
        let err_str = result.unwrap_err().to_string();
        return Ok(hyper::Response::builder()
            .status(400)
            .body(err_str.into())
            .unwrap());
    }
    let comp_req: CompactionRequest = result.unwrap();

    if comp_req.compactor_version > CURRENT_COMPACTOR_VERSION {
        warn!(
            "received incompatible compactor-version({}). Upgrade tikv-worker (version:{}). Request: {:?}",
            comp_req.compactor_version, CURRENT_COMPACTOR_VERSION, comp_req,
        );
        let err_str = format!("incompatible compactor version ({CURRENT_COMPACTOR_VERSION})");
        return Ok(hyper::Response::builder()
            .status(INCOMPATIBLE_COMPACTOR_ERROR_CODE)
            .body(err_str.into())
            .unwrap());
    }
    let encryption_key = if comp_req.exported_encryption_key.is_empty() {
        None
    } else {
        Some(
            master_key
                .decrypt_encryption_key(&comp_req.exported_encryption_key)
                .unwrap(),
        )
    };
    let ctx = CompactionCtx {
        req: Arc::new(comp_req),
        dfs,
        compression_lvl,
        id_allocator,
        encryption_key,
        local_dir: None,
        for_restore: false,
        checksum_type,
    };
    let (tx, rx) = tokio::sync::oneshot::channel();
    std::thread::spawn(move || {
        tikv_util::set_current_region(ctx.req.shard_id);
        let result = local_compact(&ctx);
        if let Err(err) = tx.send(result) {
            // Send failed only when `rx` is dropped, should happen only when the server is
            // shutting down.
            let tag = ShardTag::from_comp_req(&ctx.req);
            warn!("{} failed to send compaction result: {:?}", tag, err);
        }
    });
    match rx.await.unwrap() {
        Ok(cs) => {
            let data = cs.write_to_bytes().unwrap();
            Ok(hyper::Response::builder()
                .status(200)
                .body(data.into())
                .unwrap())
        }
        Err(err) => {
            let err_str = format!("{:?}", err);
            error!("compaction failed {}", err_str);
            let body = hyper::Body::from(err_str);
            Ok(hyper::Response::builder().status(500).body(body).unwrap())
        }
    }
}

pub(crate) struct CompactionCtx {
    pub(crate) req: Arc<CompactionRequest>,
    pub(crate) dfs: Arc<dyn dfs::Dfs>,
    pub(crate) compression_lvl: i32,
    pub(crate) checksum_type: ChecksumType,
    pub(crate) id_allocator: Arc<dyn IdAllocator>,
    pub(crate) encryption_key: Option<EncryptionKey>,
    pub(crate) local_dir: Option<PathBuf>,
    for_restore: bool,
}

fn merge_table_change(
    mut row_tbl: pb::TableChange,
    mut col_tbl: pb::TableChange,
) -> pb::TableChange {
    let mut tb = pb::TableChange::new();
    tb.set_table_deletes(row_tbl.take_table_deletes());
    tb.set_table_creates(row_tbl.take_table_creates());
    tb.set_columnar_deletes(col_tbl.take_columnar_deletes());
    tb.set_columnar_creates(col_tbl.take_columnar_creates());
    tb.file_ids_map = row_tbl.file_ids_map;
    tb.file_ids_map.extend(col_tbl.file_ids_map);
    tb
}

struct LocalIdAllocator {
    id_allocator: Arc<dyn IdAllocator>,
    file_ids: Vec<u64>,
    num_files_quota: usize,
}

impl LocalIdAllocator {
    fn new(id_allocator: Arc<dyn IdAllocator>, file_ids: Vec<u64>) -> Self {
        let num_files_quota = 4096usize.saturating_sub(file_ids.len());
        Self {
            id_allocator,
            file_ids,
            num_files_quota,
        }
    }

    async fn alloc_id(&mut self) -> u64 {
        if self.file_ids.is_empty() && self.num_files_quota > 0 {
            let allocated = self
                .id_allocator
                .alloc_id_async(std::cmp::min(64, self.num_files_quota))
                .await
                .unwrap_or_else(|e| {
                    panic!("failed to allocate id: {:?}", e);
                });
            self.file_ids.extend_from_slice(&allocated);
            self.num_files_quota -= allocated.len();
        }
        self.file_ids.pop().unwrap_or_else(|| {
            panic!("compaction runs out of file ids");
        })
    }

    fn block_on_alloc_id(&mut self) -> u64 {
        block_on(self.alloc_id())
    }
}

fn local_compact(ctx: &CompactionCtx) -> Result<pb::ChangeSet> {
    let req = &ctx.req;
    if req.compactor_version != CURRENT_COMPACTOR_VERSION {
        return Err(CompactionNotRetryable(format!(
            "invalid compactor_version {}",
            req.compactor_version
        )));
    }
    let mut cs = pb::ChangeSet::new();
    cs.set_shard_id(req.shard_id);
    cs.set_shard_ver(req.shard_ver);
    let tag = ShardTag::from_comp_req(req);
    info!("start compaction for {}, req {:?}", tag, req);
    let mut id_allocator = LocalIdAllocator::new(ctx.id_allocator.clone(), req.file_ids.clone());
    match &req.compaction_tp {
        // For backward compatibility, remove InPlace compaction when tikv-server is upgraded
        CompactionType::InPlace {
            file_ids,
            block_size,
            spec,
        } => match spec {
            InPlaceCompaction::DestroyRange(del_prefix) => {
                cs.set_destroy_range(compact_destroy_range(
                    ctx,
                    *block_size,
                    file_ids,
                    del_prefix,
                )?);
            }
            InPlaceCompaction::TrimOverBound => {
                cs.set_trim_over_bound(compact_trim_over_bound(ctx, *block_size, file_ids)?);
            }
            InPlaceCompaction::TruncateTs(truncate_ts) => {
                cs.set_truncate_ts(compact_truncate_ts(
                    ctx,
                    *block_size,
                    file_ids,
                    *truncate_ts,
                )?);
            }
            InPlaceCompaction::Unknown => unreachable!(),
        },
        CompactionType::InPlaceWithColumnar(InPlaceCompactionCtx {
            file_ids,
            col_file_ids,
            block_size,
            columnar_build_opts,
            schema_file_id,
            spec,
        }) => match spec {
            InPlaceCompaction::DestroyRange(del_prefix) => {
                let row_tb = if !file_ids.is_empty() {
                    compact_destroy_range(ctx, *block_size, file_ids, del_prefix)?
                } else {
                    pb::TableChange::new()
                };
                let col_tb = if !col_file_ids.is_empty() {
                    block_on(compact_destroy_range_for_columnar(
                        ctx,
                        columnar_build_opts,
                        col_file_ids,
                        &mut id_allocator,
                        *schema_file_id,
                        del_prefix,
                    ))?
                } else {
                    pb::TableChange::new()
                };
                cs.set_destroy_range(merge_table_change(row_tb, col_tb));
            }
            InPlaceCompaction::TrimOverBound => {
                let row_tb = if !file_ids.is_empty() {
                    compact_trim_over_bound(ctx, *block_size, file_ids)?
                } else {
                    pb::TableChange::new()
                };
                let col_tb = if !col_file_ids.is_empty() {
                    block_on(compact_trim_over_bound_for_columnar(
                        ctx,
                        columnar_build_opts,
                        col_file_ids,
                        &mut id_allocator,
                        *schema_file_id,
                    ))?
                } else {
                    pb::TableChange::new()
                };
                cs.set_trim_over_bound(merge_table_change(row_tb, col_tb));
            }
            InPlaceCompaction::TruncateTs(truncate_ts) => {
                let row_tb = if !file_ids.is_empty() {
                    compact_truncate_ts(ctx, *block_size, file_ids, *truncate_ts)?
                } else {
                    pb::TableChange::new()
                };
                let col_tb = if !col_file_ids.is_empty() {
                    block_on(compact_truncate_ts_for_columnar(
                        ctx,
                        columnar_build_opts,
                        col_file_ids,
                        &mut id_allocator,
                        *schema_file_id,
                        *truncate_ts,
                    ))?
                } else {
                    pb::TableChange::new()
                };
                cs.set_truncate_ts(merge_table_change(row_tb, col_tb));
            }
            InPlaceCompaction::Unknown => unreachable!(),
        },
        CompactionType::L0(l0_compaction) => {
            cs.set_compaction(l0_compact_v3(ctx, l0_compaction, &mut id_allocator)?);
        }
        CompactionType::L1Plus(l1_plus_compaction) => {
            cs.set_compaction(l1_plus_compact_v3(
                ctx,
                l1_plus_compaction,
                &mut id_allocator,
            )?);
        }
        CompactionType::Major(major_compaction) => {
            cs.set_major_compaction(major_compact_v3(ctx, major_compaction, &mut id_allocator)?);
        }
        CompactionType::Columnar(columnar_compaction) => {
            cs.set_columnar_compaction(block_on(columnar_compact(
                ctx,
                columnar_compaction,
                &mut id_allocator,
            ))?);
        }
        CompactionType::ColumnarMajor(columnar_major_compaction) => {
            cs.set_columnar_compaction(block_on(columnar_major_compact(
                ctx,
                columnar_major_compaction,
                &mut id_allocator,
            ))?);
        }
        CompactionType::VectorIndex(update_vec_idx) => {
            cs.set_update_vector_index(block_on(update_vector_index(
                ctx,
                update_vec_idx,
                &mut id_allocator,
            ))?);
        }
        CompactionType::Unknown => unreachable!(),
    }
    Ok(cs)
}

/// Compact files in place to remove data covered by delete prefixes.
fn compact_destroy_range(
    ctx: &CompactionCtx,
    block_size: usize,
    files: &[(u64, u32, i32)],
    del_prefix: &[u8],
) -> Result<pb::TableChange> {
    let req = &ctx.req;
    let dfs = &ctx.dfs;
    let compression_lvl = ctx.compression_lvl;
    let checksum_type = ctx.checksum_type;
    assert!(!del_prefix.is_empty() && !files.is_empty());
    assert!(files.len() <= req.file_ids.len());

    let opts = dfs::Options::default().with_shard(req.shard_id, req.shard_ver);
    let mut table_files: HashMap<u64, Arc<dyn File>> = load_table_files(
        &files.iter().map(|(id, ..)| *id).collect::<Vec<_>>(),
        dfs.clone(),
        opts,
        ctx.local_dir.as_ref(),
        ctx.for_restore,
    )?
    .into_iter()
    .map(|file| (file.id(), file))
    .collect();

    let mut destroy = pb::TableChange::new();
    let mut deletes = vec![];
    let mut creates = vec![];
    let del_prefixes = DeletePrefixes::unmarshal(del_prefix, req.keyspace_id());
    let (tx, rx) = tikv_util::mpsc::bounded(req.file_ids.len());
    for (&(id, level, cf), &new_id) in files.iter().zip(req.file_ids.iter()) {
        let mut delete = pb::TableDelete::new();
        delete.set_id(id);
        delete.set_level(level);
        delete.set_cf(cf);
        deletes.push(delete);
        let file = table_files.remove(&id).unwrap();
        let (data, smallest, biggest, meta_offset) = if level == 0 {
            let t =
                sstable::L0Table::new(file, BlockCache::None, false, ctx.encryption_key.clone())
                    .unwrap()
                    .unwrap();
            let mut builder = L0Builder::new(
                new_id,
                block_size,
                t.version(),
                checksum_type,
                ctx.encryption_key.clone(),
                ctx.req.prepend_keyspace_id(),
            );
            for cf in 0..NUM_CFS {
                if let Some(cf_t) = t.get_cf(cf) {
                    let mut iter = cf_t.new_iterator(false, false);
                    iter.seek(req.inner_start());
                    while iter.valid() {
                        let key = iter.key();
                        if key >= req.inner_end() {
                            break;
                        }
                        if !del_prefixes.cover_prefix(key) {
                            builder.add(cf, key, &iter.value(), None);
                        }
                        iter.next_all_version();
                    }
                }
            }
            if builder.is_empty() {
                continue;
            }
            let (mut l0_create, data) = builder.finish();
            (data, l0_create.take_smallest(), l0_create.take_biggest(), 0)
        } else {
            let t =
                sstable::SsTable::new(file, BlockCache::None, ctx.encryption_key.clone()).unwrap();
            let mut builder = sstable::Builder::new(
                new_id,
                block_size,
                t.compression_type(),
                compression_lvl,
                ctx.checksum_type,
                ctx.encryption_key.clone(),
                ctx.req.prepend_keyspace_id(),
            );
            let mut iter = t.new_iterator(false, false);
            iter.seek(req.inner_start());
            while iter.valid() {
                let key = iter.key();
                if key >= req.inner_end() {
                    break;
                }
                if !del_prefixes.cover_prefix(key) {
                    builder.add(key, &iter.value(), None);
                }
                iter.next_all_version();
            }
            if builder.is_empty() {
                continue;
            }
            let mut buf = Vec::with_capacity(builder.estimated_size());
            let res = builder.finish(0, &mut buf);
            (buf.into(), res.smallest, res.biggest, res.meta_offset)
        };
        let tx = tx.clone();
        let dfs_clone = dfs.clone();
        dfs.get_runtime().spawn(async move {
            tx.send(dfs_clone.create(new_id, data, opts).await).unwrap();
        });
        let mut create = pb::TableCreate::new();
        create.set_id(new_id);
        create.set_level(level);
        create.set_cf(cf);
        create.set_smallest(smallest);
        create.set_biggest(biggest);
        create.set_meta_offset(meta_offset);
        creates.push(create);

        destroy.mut_file_ids_map().push(id);
        destroy.mut_file_ids_map().push(new_id);
    }
    let mut errors = creates
        .iter()
        .filter_map(|_| match rx.recv().unwrap() {
            Ok(_) => None,
            Err(e) => Some(e),
        })
        .collect::<Vec<_>>();
    if !errors.is_empty() {
        return Err(errors.pop().unwrap().into());
    }

    destroy.set_table_deletes(deletes.into());
    destroy.set_table_creates(creates.into());
    Ok(destroy)
}

async fn compact_destroy_range_for_columnar(
    ctx: &CompactionCtx,
    columnar_build_opts: &ColumnarTableBuildOptions,
    files: &[(u64, u32)],
    id_allocator: &mut LocalIdAllocator,
    schema_file_id: Option<u64>,
    del_prefix: &[u8],
) -> Result<pb::TableChange> {
    let req = &ctx.req;
    let dfs = &ctx.dfs;
    assert!(!del_prefix.is_empty() && !files.is_empty());

    let opts = dfs::Options::default()
        .with_shard(req.shard_id, req.shard_ver)
        .with_type(FileType::Columnar);
    let mut columnar_files: HashMap<u64, Arc<dyn File>> = load_table_files(
        &files.iter().map(|(id, ..)| *id).collect::<Vec<_>>(),
        dfs.clone(),
        opts,
        ctx.local_dir.as_ref(),
        ctx.for_restore,
    )?
    .into_iter()
    .map(|file| (file.id(), file))
    .collect();

    let schema_file = if schema_file_id.is_some() {
        let file = load_table_files(
            &[schema_file_id.unwrap()],
            dfs.clone(),
            dfs::Options::default().with_type(FileType::Schema),
            ctx.local_dir.as_ref(),
            false,
        )?
        .pop()
        .unwrap();
        Some(SchemaFile::open(file).unwrap())
    } else {
        None
    };

    let mut deletes = vec![];
    let mut creates = vec![];
    let del_prefixes = Arc::new(DeletePrefixes::unmarshal(del_prefix, req.keyspace_id()));
    let (tx, rx) = tikv_util::mpsc::bounded(req.file_ids.len());
    let mut cnt = 0;
    let keyspace_id = ApiV2::get_u32_keyspace_id_by_key(&req.outer_start).unwrap_or_default();
    for &(id, level) in files.iter() {
        let schema_file = schema_file.as_ref().unwrap();
        let file = columnar_files.remove(&id).unwrap();
        let columnar_file = ColumnarFile::open(file).unwrap();
        let overlap_tables =
            schema_file.overlap_tables(columnar_file.get_smallest(), columnar_file.get_biggest());
        let mut delete = pb::ColumnarDelete::new();
        delete.set_id(id);
        delete.set_level(level);
        deletes.push(delete);
        if overlap_tables.is_empty() {
            continue;
        }
        let mut file_builder = ColumnarFileBuilder::new(
            id_allocator.alloc_id().await,
            columnar_file.get_l0_version(),
            ctx.encryption_key.clone(),
        );
        for table_id in overlap_tables {
            let table_row_key_prefix = [
                api_version::ApiV2::get_txn_keyspace_prefix(keyspace_id),
                encode_row_key(table_id, 0),
            ]
            .concat();
            // destroy range used by drop table in TiDB, if the row key be covered by delete
            // prefixes, it means the table is dropped.
            let inner_key = InnerKey::from_outer_key(&table_row_key_prefix);
            if del_prefixes.cover_prefix(inner_key) {
                continue;
            }
            if columnar_file.has_table(table_id) {
                continue;
            }
            let schema = schema_file.get_table(table_id).unwrap();
            let reader = ColumnarTableReader::new(
                &columnar_file,
                schema.clone(),
                ctx.encryption_key.clone(),
            );
            let mut compact_reader =
                ColumnarCompactReader::new(Box::new(reader), level, schema, req.safe_ts);
            compact_reader.set_unbounded_handle_range().await?;
            compact_table_for_columnar(
                ctx,
                &mut compact_reader,
                &mut file_builder,
                schema,
                level,
                &mut cnt,
                tx.clone(),
                columnar_build_opts,
                id_allocator,
            )
            .await?;
        }
        if file_builder.num_tables() > 0 {
            cnt += 1;
            persist_columnar_file(level, &mut file_builder, tx.clone(), ctx.dfs.clone(), opts);
        }
    }
    let mut errors = vec![];
    for _ in 0..cnt {
        match rx.recv().unwrap() {
            Err(err) => errors.push(err),
            Ok(create) => {
                creates.push(create);
            }
        }
    }
    if !errors.is_empty() {
        return Err(errors.pop().unwrap().into());
    }
    let mut destroy = pb::TableChange::new();
    destroy.set_columnar_deletes(deletes.into());
    destroy.set_columnar_creates(creates.into());
    Ok(destroy)
}

fn compact_truncate_ts(
    ctx: &CompactionCtx,
    block_size: usize,
    files: &[(u64, u32, i32)],
    truncate_ts: u64,
) -> Result<pb::TableChange> {
    let req = &ctx.req;
    let dfs = &ctx.dfs;
    let compression_lvl = ctx.compression_lvl;
    let checksum_type = ctx.checksum_type;
    assert!(!files.is_empty());

    let opts = dfs::Options::default().with_shard(req.shard_id, req.shard_ver);
    let mut table_files: HashMap<u64, Arc<dyn File>> = load_table_files(
        &files.iter().map(|(id, ..)| *id).collect::<Vec<_>>(),
        dfs.clone(),
        opts,
        ctx.local_dir.as_ref(),
        ctx.for_restore,
    )?
    .into_iter()
    .map(|file| (file.id(), file))
    .collect();

    let mut table_change = pb::TableChange::new();
    let mut deletes = vec![];
    let mut creates = vec![];
    let (tx, rx) = tikv_util::mpsc::bounded(req.file_ids.len());
    for (&(id, level, cf), &new_id) in files.iter().zip(req.file_ids.iter()) {
        let file = table_files.remove(&id).unwrap();
        let mut delete = pb::TableDelete::new();
        delete.set_id(id);
        delete.set_level(level);
        delete.set_cf(cf);
        deletes.push(delete);

        let (data, smallest, biggest, meta_offset) = if level == 0 {
            let t =
                sstable::L0Table::new(file, BlockCache::None, false, ctx.encryption_key.clone())
                    .unwrap()
                    .unwrap();
            let mut builder = L0Builder::new(
                new_id,
                block_size,
                t.version(),
                checksum_type,
                ctx.encryption_key.clone(),
                ctx.req.prepend_keyspace_id(),
            );
            for cf in 0..NUM_CFS {
                if let Some(cf_t) = t.get_cf(cf) {
                    let mut iter = cf_t.new_iterator(false, false);
                    iter.rewind();
                    while iter.valid() {
                        let key = iter.key();
                        let value = iter.value();
                        // Keep data in LOCK_CF. Locks would be resolved by TiDB.
                        // TODO: handle async commit
                        if cf == LOCK_CF || value.version <= truncate_ts {
                            builder.add(cf, key, &value, None);
                        }
                        iter.next_all_version();
                    }
                }
            }
            if builder.is_empty() {
                continue;
            }
            let (mut l0_create, data) = builder.finish();
            (data, l0_create.take_smallest(), l0_create.take_biggest(), 0)
        } else {
            let t =
                sstable::SsTable::new(file, BlockCache::None, ctx.encryption_key.clone()).unwrap();
            let mut builder = sstable::Builder::new(
                new_id,
                block_size,
                t.compression_type(),
                compression_lvl,
                checksum_type,
                ctx.encryption_key.clone(),
                ctx.req.prepend_keyspace_id(),
            );
            let mut iter = t.new_iterator(false, false);
            iter.rewind();
            while iter.valid() {
                let key = iter.key();
                let value = iter.value();
                // Keep data in LOCK_CF. Locks would be resolved by TiDB.
                // TODO: handle async commit
                if cf as usize == LOCK_CF || value.version <= truncate_ts {
                    builder.add(key, &value, None);
                }
                iter.next_all_version();
            }
            if builder.is_empty() {
                continue;
            }
            let mut buf = Vec::with_capacity(builder.estimated_size());
            let res = builder.finish(0, &mut buf);
            (buf.into(), res.smallest, res.biggest, res.meta_offset)
        };

        let tx = tx.clone();
        let dfs_clone = dfs.clone();
        dfs.get_runtime().spawn(async move {
            tx.send(dfs_clone.create(new_id, data, opts).await).unwrap();
        });

        let mut create = pb::TableCreate::new();
        create.set_id(new_id);
        create.set_level(level);
        create.set_cf(cf);
        create.set_smallest(smallest);
        create.set_biggest(biggest);
        create.set_meta_offset(meta_offset);
        creates.push(create);

        table_change.mut_file_ids_map().push(id);
        table_change.mut_file_ids_map().push(new_id);
    }

    let mut errors = creates
        .iter()
        .filter_map(|_| match rx.recv().unwrap() {
            Ok(_) => None,
            Err(e) => Some(e),
        })
        .collect::<Vec<_>>();
    if !errors.is_empty() {
        return Err(errors.pop().unwrap().into());
    }

    table_change.set_table_deletes(deletes.into());
    table_change.set_table_creates(creates.into());
    Ok(table_change)
}

async fn compact_truncate_ts_for_columnar(
    ctx: &CompactionCtx,
    columnar_build_opts: &ColumnarTableBuildOptions,
    files: &[(u64, u32)],
    id_allocator: &mut LocalIdAllocator,
    schema_file_id: Option<u64>,
    truncate_ts: u64,
) -> Result<pb::TableChange> {
    let req = &ctx.req;
    let dfs = &ctx.dfs;
    assert!(!files.is_empty());

    let opts = dfs::Options::default()
        .with_shard(req.shard_id, req.shard_ver)
        .with_type(FileType::Columnar);
    let mut columnar_files: HashMap<u64, Arc<dyn File>> = load_table_files(
        &files.iter().map(|(id, ..)| *id).collect::<Vec<_>>(),
        dfs.clone(),
        opts,
        ctx.local_dir.as_ref(),
        ctx.for_restore,
    )?
    .into_iter()
    .map(|file| (file.id(), file))
    .collect();

    let schema_file = if schema_file_id.is_some() {
        let file = load_table_files(
            &[schema_file_id.unwrap()],
            dfs.clone(),
            dfs::Options::default().with_type(FileType::Schema),
            ctx.local_dir.as_ref(),
            false,
        )?
        .pop()
        .unwrap();
        Some(SchemaFile::open(file).unwrap())
    } else {
        None
    };

    let mut deletes = vec![];
    let mut creates = vec![];
    let (tx, rx) = tikv_util::mpsc::bounded(req.file_ids.len());
    let mut cnt = 0;
    for &(id, level) in files.iter() {
        let schema_file = schema_file.as_ref().unwrap();
        let file = columnar_files.remove(&id).unwrap();
        let columnar_file = ColumnarFile::open(file).unwrap();
        let overlap_tables =
            schema_file.overlap_tables(columnar_file.get_smallest(), columnar_file.get_biggest());
        let mut delete = pb::ColumnarDelete::new();
        delete.set_id(id);
        delete.set_level(level);
        deletes.push(delete);

        if overlap_tables.is_empty() {
            continue;
        }
        let mut file_builder = ColumnarFileBuilder::new(
            id_allocator.alloc_id().await,
            columnar_file.get_l0_version(),
            ctx.encryption_key.clone(),
        );
        for table_id in overlap_tables {
            let schema = schema_file.get_table(table_id).unwrap();
            if !columnar_file.has_table(table_id) {
                continue;
            }
            let reader = ColumnarTableReader::new(
                &columnar_file,
                schema.clone(),
                ctx.encryption_key.clone(),
            );
            let mut truncate_ts_reader =
                ColumnarTruncateTsReader::new(Box::new(reader), schema, truncate_ts);
            truncate_ts_reader.set_unbounded_handle_range().await?;
            compact_table_for_columnar(
                ctx,
                &mut truncate_ts_reader,
                &mut file_builder,
                schema,
                level,
                &mut cnt,
                tx.clone(),
                columnar_build_opts,
                id_allocator,
            )
            .await?;
        }
        if file_builder.num_tables() > 0 {
            cnt += 1;
            persist_columnar_file(level, &mut file_builder, tx.clone(), ctx.dfs.clone(), opts);
        }
    }
    let mut errors = vec![];
    for _ in 0..cnt {
        match rx.recv().unwrap() {
            Err(err) => errors.push(err),
            Ok(create) => {
                creates.push(create);
            }
        }
    }
    if !errors.is_empty() {
        return Err(errors.pop().unwrap().into());
    }
    let mut table_change = pb::TableChange::new();
    table_change.set_columnar_deletes(deletes.into());
    table_change.set_columnar_creates(creates.into());
    info!(
        "finish columnar truncate_ts compaction, table_change {:?}",
        table_change
    );
    Ok(table_change)
}

/// Compact files in place to remove data out of shard bound.
fn compact_trim_over_bound(
    ctx: &CompactionCtx,
    block_size: usize,
    files: &[(u64, u32, i32)],
) -> Result<pb::TableChange> {
    let req = &ctx.req;
    let dfs = &ctx.dfs;
    let compression_lvl = ctx.compression_lvl;
    let checksum_tp = ctx.checksum_type;
    assert!(!files.is_empty());
    assert!(files.len() <= req.file_ids.len());

    let opts = dfs::Options::default().with_shard(req.shard_id, req.shard_ver);
    let mut table_files: HashMap<u64, Arc<dyn File>> = load_table_files(
        &files.iter().map(|(id, ..)| *id).collect::<Vec<_>>(),
        dfs.clone(),
        opts,
        ctx.local_dir.as_ref(),
        ctx.for_restore,
    )?
    .into_iter()
    .map(|file| (file.id(), file))
    .collect();

    let mut table_change = pb::TableChange::new();
    let mut deletes = vec![];
    let mut creates = vec![];
    let (tx, rx) = tikv_util::mpsc::bounded(req.file_ids.len());
    for (&(id, level, cf), &new_id) in files.iter().zip(req.file_ids.iter()) {
        let file = table_files.remove(&id).unwrap();

        let mut delete = pb::TableDelete::new();
        delete.set_id(id);
        delete.set_level(level);
        delete.set_cf(cf);
        deletes.push(delete);

        let (data, smallest, biggest, meta_offset) = if level == 0 {
            let t =
                sstable::L0Table::new(file, BlockCache::None, false, ctx.encryption_key.clone())
                    .unwrap()
                    .unwrap();
            let mut builder = L0Builder::new(
                new_id,
                block_size,
                t.version(),
                checksum_tp,
                ctx.encryption_key.clone(),
                ctx.req.prepend_keyspace_id(),
            );
            for cf in 0..NUM_CFS {
                if let Some(cf_t) = t.get_cf(cf) {
                    let mut iter = cf_t.new_iterator(false, false);
                    iter.seek(req.inner_start());
                    while iter.valid() {
                        let key = iter.key();
                        if key >= req.inner_end() {
                            break;
                        }
                        builder.add(cf, key, &iter.value(), None);
                        iter.next_all_version();
                    }
                }
            }
            if builder.is_empty() {
                continue;
            }
            let (mut l0_create, data) = builder.finish();
            (data, l0_create.take_smallest(), l0_create.take_biggest(), 0)
        } else {
            let t =
                sstable::SsTable::new(file, BlockCache::None, ctx.encryption_key.clone()).unwrap();
            let mut builder = sstable::Builder::new(
                new_id,
                block_size,
                t.compression_type(),
                compression_lvl,
                checksum_tp,
                ctx.encryption_key.clone(),
                ctx.req.prepend_keyspace_id(),
            );
            let mut iter = t.new_iterator(false, false);
            iter.seek(req.inner_start());
            while iter.valid() {
                let key = iter.key();
                if key >= req.inner_end() {
                    break;
                }
                builder.add(key, &iter.value(), None);
                iter.next_all_version();
            }
            if builder.is_empty() {
                continue;
            }
            let mut buf = Vec::with_capacity(builder.estimated_size());
            let res = builder.finish(0, &mut buf);
            (buf.into(), res.smallest, res.biggest, res.meta_offset)
        };
        let tx = tx.clone();
        let dfs_clone = dfs.clone();
        dfs.get_runtime().spawn(async move {
            tx.send(dfs_clone.create(new_id, data, opts).await).unwrap();
        });

        let mut create = pb::TableCreate::new();
        create.set_id(new_id);
        create.set_level(level);
        create.set_cf(cf);
        create.set_smallest(smallest);
        create.set_biggest(biggest);
        create.set_meta_offset(meta_offset);
        creates.push(create);

        table_change.mut_file_ids_map().push(id);
        table_change.mut_file_ids_map().push(new_id);
    }

    let mut errors = creates
        .iter()
        .filter_map(|_| match rx.recv().unwrap() {
            Ok(_) => None,
            Err(e) => Some(e),
        })
        .collect::<Vec<_>>();
    if !errors.is_empty() {
        return Err(errors.pop().unwrap().into());
    }

    table_change.set_table_deletes(deletes.into());
    table_change.set_table_creates(creates.into());
    Ok(table_change)
}

async fn compact_trim_over_bound_for_columnar(
    ctx: &CompactionCtx,
    columnar_build_opts: &ColumnarTableBuildOptions,
    files: &[(u64, u32)],
    id_allocator: &mut LocalIdAllocator,
    schema_file_id: Option<u64>,
) -> Result<pb::TableChange> {
    let req = &ctx.req;
    let dfs = &ctx.dfs;
    assert!(!files.is_empty());
    assert!(files.len() <= req.file_ids.len());

    let opts = dfs::Options::default()
        .with_shard(req.shard_id, req.shard_ver)
        .with_type(FileType::Columnar);
    let mut columnar_files: HashMap<u64, Arc<dyn File>> = load_table_files(
        &files.iter().map(|(id, ..)| *id).collect::<Vec<_>>(),
        dfs.clone(),
        opts,
        ctx.local_dir.as_ref(),
        ctx.for_restore,
    )?
    .into_iter()
    .map(|file| (file.id(), file))
    .collect();

    let schema_file = if schema_file_id.is_some() {
        let file = load_table_files(
            &[schema_file_id.unwrap()],
            dfs.clone(),
            dfs::Options::default().with_type(FileType::Schema),
            ctx.local_dir.as_ref(),
            false,
        )?
        .pop()
        .unwrap();
        Some(SchemaFile::open(file).unwrap())
    } else {
        None
    };

    let mut deletes = vec![];
    let mut creates = vec![];
    let (tx, rx) = tikv_util::mpsc::bounded(req.file_ids.len());
    let mut cnt = 0;
    for &(id, level) in files.iter() {
        let schema_file = schema_file.as_ref().unwrap();
        let file = columnar_files.remove(&id).unwrap();
        let columnar_file = ColumnarFile::open(file).unwrap();
        let overlap_tables =
            schema_file.overlap_tables(columnar_file.get_smallest(), columnar_file.get_biggest());
        let mut delete = pb::ColumnarDelete::new();
        delete.set_id(id);
        delete.set_level(level);
        deletes.push(delete);
        if overlap_tables.is_empty() {
            continue;
        }
        let bound_start =
            if let Some(keyspace_prefix) = ApiV2::get_keyspace_prefix(&req.outer_start) {
                &req.outer_start[keyspace_prefix.len()..]
            } else {
                &req.outer_start
            };
        let bound_end = if let Some(keyspace_prefix) = ApiV2::get_keyspace_prefix(&req.outer_end) {
            if req.outer_end.len() == keyspace_prefix.len() {
                GLOBAL_SHARD_END_KEY
            } else {
                &req.outer_end[keyspace_prefix.len()..]
            }
        } else {
            &req.outer_end
        };
        let mut file_builder = ColumnarFileBuilder::new(
            id_allocator.alloc_id().await,
            columnar_file.get_l0_version(),
            ctx.encryption_key.clone(),
        );
        for table_id in overlap_tables {
            let row_key_prefix = encode_row_key_prefix(table_id);
            let mut row_key_prefix_next = row_key_prefix.clone();
            convert_to_prefix_next(&mut row_key_prefix_next);
            if bound_start >= row_key_prefix_next.as_slice()
                || bound_end <= row_key_prefix.as_slice()
            {
                // The whole table is over bound.
                continue;
            }
            if !columnar_file.has_table(table_id) {
                continue;
            }
            let schema = schema_file.get_table(table_id).unwrap();
            let reader = ColumnarTableReader::new(
                &columnar_file,
                schema.clone(),
                ctx.encryption_key.clone(),
            );
            let mut compact_reader =
                ColumnarCompactReader::new(Box::new(reader), level, schema, req.safe_ts);
            if schema.is_common_handle() {
                let start_handle = if bound_start > row_key_prefix.as_slice() {
                    decode_common_handle(bound_start).unwrap()
                } else {
                    &[]
                };
                let end_handle = if bound_end < row_key_prefix_next.as_slice() {
                    decode_common_handle(bound_end).unwrap()
                } else {
                    GLOBAL_COMMON_HANDLE_END
                };
                compact_reader
                    .set_handle_range(start_handle, end_handle)
                    .await?;
            } else {
                let start_handle = if bound_start > row_key_prefix.as_slice() {
                    decode_int_handle(bound_start).unwrap()
                } else {
                    i64::MIN
                };
                let end_handle = if bound_end < row_key_prefix_next.as_slice() {
                    Some(decode_int_handle(bound_end).unwrap())
                } else {
                    None
                };
                compact_reader
                    .set_int_handle_range(start_handle, end_handle)
                    .await?;
            }
            compact_table_for_columnar(
                ctx,
                &mut compact_reader,
                &mut file_builder,
                schema,
                level,
                &mut cnt,
                tx.clone(),
                columnar_build_opts,
                id_allocator,
            )
            .await?;
        }
        if file_builder.num_tables() > 0 {
            cnt += 1;
            persist_columnar_file(level, &mut file_builder, tx.clone(), ctx.dfs.clone(), opts);
        }
    }

    let mut errors = vec![];
    for _ in 0..cnt {
        match rx.recv().unwrap() {
            Err(err) => errors.push(err),
            Ok(create) => {
                creates.push(create);
            }
        }
    }
    if !errors.is_empty() {
        return Err(errors.pop().unwrap().into());
    }
    let mut table_change = pb::TableChange::new();
    table_change.set_columnar_deletes(deletes.into());
    table_change.set_columnar_creates(creates.into());
    info!(
        "finish columnar trim over bound compaction, table_change: {:?}",
        table_change
    );
    Ok(table_change)
}

enum FilePersistResult {
    TableCreate(pb::TableCreate),
    BlobTableCreate(pb::BlobCreate),
}

fn persist_sst(
    id: u64,
    cf: usize,
    target_lvl: u32,
    builder: &mut sstable::Builder,
    tx: mpsc::Sender<dfs::Result<FilePersistResult>>,
    fs: Arc<dyn dfs::Dfs>,
    opts: dfs::Options,
) {
    let mut buf = Vec::with_capacity(builder.estimated_size());
    let res = builder.finish(0, &mut buf);
    let mut tbl_create = pb::TableCreate::new();
    tbl_create.set_id(id);
    tbl_create.set_cf(cf as i32);
    tbl_create.set_level(target_lvl);
    tbl_create.set_smallest(res.smallest);
    tbl_create.set_biggest(res.biggest);
    tbl_create.set_meta_offset(res.meta_offset);

    let fs_clone = fs.clone();
    fs.get_runtime().spawn(async move {
        tx.send(
            fs_clone
                .create(id, buf.into(), opts)
                .await
                .map(|_| FilePersistResult::TableCreate(tbl_create)),
        )
        .unwrap();
    });
}

fn persist_blob_table(
    id: u64,
    builder: &mut BlobTableBuilder,
    tx: mpsc::Sender<dfs::Result<FilePersistResult>>,
    fs: Arc<dyn dfs::Dfs>,
    opts: dfs::Options,
) {
    let buf = builder.finish();
    let mut blob_table_create = pb::BlobCreate::new();
    blob_table_create.set_id(id);
    let (smallest, biggest) = builder.smallest_biggest_key();
    blob_table_create.set_smallest(smallest.to_vec());
    blob_table_create.set_biggest(biggest.to_vec());
    let fs_clone = fs.clone();
    fs.get_runtime().spawn(async move {
        tx.send(
            fs_clone
                .create(id, buf, opts)
                .await
                .map(|_| FilePersistResult::BlobTableCreate(blob_table_create)),
        )
        .unwrap();
    });
}

fn persist_columnar_file(
    target_lvl: u32,
    builder: &mut ColumnarFileBuilder,
    tx: mpsc::Sender<dfs::Result<pb::ColumnarCreate>>,
    fs: Arc<dyn dfs::Dfs>,
    opts: dfs::Options,
) {
    let id = builder.file_id;
    let (buf, meta_offset) = builder.build();
    let mut columnar_create = pb::ColumnarCreate::new();
    columnar_create.set_id(id);
    columnar_create.set_smallest(builder.smallest.clone());
    columnar_create.set_biggest(builder.biggest.clone());
    columnar_create.set_level(target_lvl);
    columnar_create.set_meta_offset(meta_offset as u32);
    let fs_clone = fs.clone();
    fs.get_runtime().spawn(async move {
        tx.send(
            fs_clone
                .create(id, buf.into(), opts.with_type(FileType::Columnar))
                .await
                .map(|_| columnar_create),
        )
        .unwrap();
    });
}

fn compact_for_cf(
    ctx: &CompactionCtx,
    iter: &mut Box<dyn table::Iterator>,
    safe_ts: u64,
    opts: dfs::Options,
    cf: usize,
    target_lvl: u32,
    sst_config: &TableBuilderOptions,
    bt_config: &Option<BlobTableBuildOptions>,
    keep_latest_obsolete_tombstone: bool,
    blob_tables: &HashMap<u64, BlobTable>,
    id_allocator: &mut LocalIdAllocator,
) -> Result<(Vec<TableCreate>, Vec<BlobCreate>)> {
    let shard_id = ctx.req.shard_id;
    let start = ctx.req.inner_start();
    let end = ctx.req.inner_end();
    let fs = &ctx.dfs;
    let compression_lvl = ctx.compression_lvl;
    let checksum_tp = ctx.checksum_type;
    let (tx, rx) = tikv_util::mpsc::bounded(ctx.req.file_ids.len());
    let mut cur_sst_id = id_allocator.block_on_alloc_id();

    let mut sst_builder = sstable::Builder::new(
        cur_sst_id,
        sst_config.block_size,
        sst_config.compression_tps[target_lvl as usize - 1],
        compression_lvl,
        checksum_tp,
        ctx.encryption_key.clone(),
        ctx.req.prepend_keyspace_id(),
    );
    let mut cur_blob_table_id = 0;
    // Owns the decompressed blob value while reading from the orginal blob
    // table, to reduce the memory re-allocation.
    let mut decompressed_blob_buf = vec![];
    let mut decryption_buf = vec![];
    let mut blob_table_builder = if let Some(config) = bt_config {
        cur_blob_table_id = id_allocator.block_on_alloc_id();
        Some((
            config,
            BlobTableBuilder::new(
                cur_blob_table_id,
                config.compression_type,
                compression_lvl,
                config.min_blob_size,
                ctx.encryption_key.clone(),
            ),
        ))
    } else {
        None
    };
    let mut cnt = 0;
    let mut last_key = BytesMut::new();
    // Whether to skip the entry if it has the same key as the last entry iterated.
    // This will be set to true when:
    //   * The last entry is the latest version of a key that is before the GC safe
    //     point.
    //   * Or this function is iterating the LOCK CF.
    let mut skip_same_key = false;
    iter.seek(start);
    while iter.valid() && iter.key() < end {
        let mut val = iter.value();
        let key = iter.key();
        // See if we need to skip this key.
        if key.deref() == last_key {
            if skip_same_key {
                iter.next_all_version();
                continue;
            }
        } else {
            // A new key is met.
            if !last_key.is_empty() {
                // Check if we need to rotate to a new SST or blob table.
                if sst_builder.estimated_size() > sst_config.max_table_size {
                    cnt += 1;
                    persist_sst(
                        cur_sst_id,
                        cf,
                        target_lvl,
                        &mut sst_builder,
                        tx.clone(),
                        fs.clone(),
                        opts,
                    );
                    cur_sst_id = id_allocator.block_on_alloc_id();
                    sst_builder.reset(cur_sst_id);
                }
                if let Some((bt_config, bt_builder)) = &mut blob_table_builder {
                    if bt_builder.total_blob_size() as usize > bt_config.max_blob_table_size {
                        cnt += 1;
                        persist_blob_table(
                            cur_blob_table_id,
                            bt_builder,
                            tx.clone(),
                            fs.clone(),
                            opts,
                        );
                        cur_blob_table_id = id_allocator.block_on_alloc_id();
                        bt_builder.reset(cur_blob_table_id);
                    }
                }
            }
            last_key.clear();
            last_key.extend_from_slice(key.deref());
            skip_same_key = false;
        }

        // Only consider the versions which are below the minReadTs, otherwise, we might
        // end up discarding the only valid version for a running transaction.
        if cf == LOCK_CF || val.version <= safe_ts {
            // key is the latest readable version of this key, so we simply discard all the
            // rest of the versions.

            skip_same_key = true;

            if val.is_deleted() {
                if !keep_latest_obsolete_tombstone {
                    iter.next_all_version();
                    continue;
                }
            } else {
                let user_meta = val.user_meta();
                if user_meta.len() == USER_META_SIZE {
                    let um = UserMeta::from_slice(user_meta);
                    if cf == WRITE_CF && um.commit_ts < safe_ts && val.is_value_empty() {
                        if keep_latest_obsolete_tombstone {
                            sst_builder.add(key, &table::Value::new_tombstone(val.version), None);
                        }
                        iter.next_all_version();
                        continue;
                    }
                    if cf == EXTRA_CF && um.start_ts < safe_ts {
                        iter.next_all_version();
                        continue;
                    }
                }
            }
        }
        if let Some((bt_config, bt_builder)) = &mut blob_table_builder {
            let mut need_recompress_blob = false;
            if val.is_blob_ref() {
                if blob_tables.is_empty() {
                    sst_builder.add(key, &val, None);
                } else {
                    // If it is a blob link, we need to deref it and may need to decompress it as
                    // well.
                    let blob_ref = val.get_blob_ref();
                    let blob_table = blob_tables.get(&blob_ref.fid).unwrap_or_else(|| {
                        panic!(
                            "[{}] blob table {} not found for key {:?}",
                            shard_id, blob_ref.fid, key,
                        )
                    });
                    // If the encryption ver is changed, we should decrypt the blob value and
                    // encrypt with the new encryption ver.
                    let need_re_encrypt = if let Some(encryption_key) = &ctx.encryption_key {
                        blob_table.encryption_ver != encryption_key.current_ver
                    } else {
                        false
                    };
                    if blob_table.compression_tp() != bt_config.compression_type
                        || blob_table.compression_lvl() != compression_lvl
                        || blob_table.min_blob_size() != bt_config.min_blob_size
                    {
                        need_recompress_blob = true;
                    }
                    // The blob value is either the original blob value or the compressed blob
                    // depending on whether need_recompress_blob is true.
                    let blob_or_compressed_blob = blob_table
                        .get_from_preloaded(
                            &blob_ref,
                            need_recompress_blob,
                            need_re_encrypt,
                            &mut decompressed_blob_buf,
                            &mut decryption_buf,
                            ctx.encryption_key.clone(),
                        )
                        .unwrap();
                    // The value needs to be added to the blob table, if:
                    //   1. need_recompress_blob is false, which means the val was already in the
                    //      blob table, and it is guaranteed that the blob size is larger than
                    //      min_blob_size.
                    //   2. need_recompress_blob is true, but the blob size is larger than
                    //      min_blob_size.
                    if !need_recompress_blob
                        || blob_or_compressed_blob.len() >= bt_config.min_blob_size as usize
                    {
                        let new_blob_ref = bt_builder.add_blob(
                            key,
                            blob_or_compressed_blob,
                            Some(blob_ref.original_len),
                            need_re_encrypt,
                        );
                        sst_builder.add(key, &val, Some(new_blob_ref));
                    } else {
                        val.fill_in_blob(blob_or_compressed_blob);
                        sst_builder.add(key, &val, None);
                    }
                }
            } else if val.value_len() >= bt_config.min_blob_size as usize {
                let blob_ref = bt_builder.add(key, &val);
                val.set_blob_ref();
                sst_builder.add(key, &val, Some(blob_ref));
            } else {
                sst_builder.add(key, &val, None);
            }
        } else {
            // If we are not building blob tables, we can write to the SST blindly.
            sst_builder.add(key, &val, None);
        }
        iter.next_all_version();
    }
    if !sst_builder.is_empty() {
        cnt += 1;
        persist_sst(
            cur_sst_id,
            cf,
            target_lvl,
            &mut sst_builder,
            tx.clone(),
            fs.clone(),
            opts,
        );
    }
    if let Some((_, bt_builder)) = &mut blob_table_builder {
        if !bt_builder.is_empty() {
            cnt += 1;
            persist_blob_table(cur_blob_table_id, bt_builder, tx, fs.clone(), opts);
        }
    }
    let mut errors = vec![];
    let mut sst_creates = vec![];
    let mut blob_table_creates = vec![];
    for _ in 0..cnt {
        match rx.recv().unwrap() {
            Err(err) => errors.push(err),
            Ok(FilePersistResult::TableCreate(tbl_create)) => {
                sst_creates.push(tbl_create);
            }
            Ok(FilePersistResult::BlobTableCreate(blob_table_create)) => {
                blob_table_creates.push(blob_table_create);
            }
        }
    }
    if !errors.is_empty() {
        return Err(errors.pop().unwrap().into());
    }
    Ok((sst_creates, blob_table_creates))
}

fn l0_compact_v3(
    ctx: &CompactionCtx,
    l0_compaction: &L0Compaction,
    id_allocator: &mut LocalIdAllocator,
) -> Result<pb::Compaction> {
    let req = &ctx.req;
    let fs = &ctx.dfs;
    let opts = dfs::Options::default().with_shard(req.shard_id, req.shard_ver);
    let l0_files = load_table_files(
        &l0_compaction.l0_tables,
        fs.clone(),
        opts,
        ctx.local_dir.as_ref(),
        ctx.for_restore,
    )?;
    let mut l0_tbls = files_to_l0_tables(l0_files, ctx.encryption_key.clone());
    l0_tbls.sort_by(|a, b| b.version().cmp(&a.version()));
    let mut comp = pb::Compaction::new();
    comp.set_top_deletes(l0_compaction.l0_tables.clone());
    comp.set_level(0_u32);
    comp.set_bottom_deletes(
        l0_compaction
            .multi_cf_l1_tables
            .clone()
            .into_iter()
            .flatten()
            .collect(),
    );
    let mut all_sst_creates = vec![];
    let mut all_bt_creates = vec![];
    for cf in 0..NUM_CFS {
        let l1_ids = &l0_compaction.multi_cf_l1_tables[cf];
        let l1_files = load_table_files(
            l1_ids,
            fs.clone(),
            opts,
            ctx.local_dir.as_ref(),
            ctx.for_restore,
        )?;
        let mut l1_tbls = files_to_tables(l1_files, ctx.encryption_key.clone());
        l1_tbls.sort_by(|a, b| a.smallest().cmp(&b.smallest()));
        let mut iters: Vec<Box<dyn table::Iterator>> = vec![];
        for l0_tbl in &l0_tbls {
            if let Some(tbl) = l0_tbl.get_cf(cf) {
                let iter = tbl.new_iterator(false, false);
                iters.push(iter);
            }
        }
        if !l1_tbls.is_empty() {
            let iter = ConcatIterator::new_with_tables(l1_tbls, false, false);
            iters.push(Box::new(iter));
        }
        let mut iter = table::new_merge_iterator(iters, false);
        let (sst_creates, bt_creates) = compact_for_cf(
            ctx,
            &mut iter,
            l0_compaction.safe_ts,
            opts,
            cf,
            1,
            &l0_compaction.sst_config,
            &l0_compaction.bt_config,
            //&None,
            true,
            &HashMap::new(),
            id_allocator,
        )?;
        all_sst_creates.extend(sst_creates);
        all_bt_creates.extend(bt_creates);
    }
    comp.set_table_creates(all_sst_creates.into());
    comp.set_blob_tables(all_bt_creates.into());
    Ok(comp)
}

fn l1_plus_compact_v3(
    ctx: &CompactionCtx,
    l1_plus_compaction: &L1PlusCompaction,
    id_allocator: &mut LocalIdAllocator,
) -> Result<pb::Compaction> {
    let req = &ctx.req;
    let fs = &ctx.dfs;
    let opts = dfs::Options::default().with_shard(req.shard_id, req.shard_ver);
    let upper_files = load_table_files(
        &l1_plus_compaction.upper_level,
        fs.clone(),
        opts,
        ctx.local_dir.as_ref(),
        ctx.for_restore,
    )?;
    let mut upper_tables = files_to_tables(upper_files, ctx.encryption_key.clone());
    upper_tables.sort_by(|a, b| a.smallest().cmp(&b.smallest()));
    let lower_files = load_table_files(
        &l1_plus_compaction.lower_level,
        fs.clone(),
        opts,
        ctx.local_dir.as_ref(),
        ctx.for_restore,
    )?;
    let mut lower_tables = files_to_tables(lower_files, ctx.encryption_key.clone());
    lower_tables.sort_by(|a, b| a.smallest().cmp(&b.smallest()));
    let upper_iter = Box::new(ConcatIterator::new_with_tables(upper_tables, false, false));
    let lower_iter = Box::new(ConcatIterator::new_with_tables(lower_tables, false, false));
    let mut iter = table::new_merge_iterator(vec![upper_iter, lower_iter], false);

    let (sst_creates, _) = compact_for_cf(
        ctx,
        &mut iter,
        l1_plus_compaction.safe_ts,
        opts,
        l1_plus_compaction.cf as usize,
        l1_plus_compaction.level as u32 + 1,
        &l1_plus_compaction.sst_config,
        &None,
        l1_plus_compaction.keep_latest_obsolete_tombstone,
        &HashMap::new(),
        id_allocator,
    )?;
    let mut comp = pb::Compaction::new();
    comp.set_top_deletes(l1_plus_compaction.upper_level.clone());
    comp.set_cf(l1_plus_compaction.cf as i32);
    comp.set_level(l1_plus_compaction.level as u32);
    comp.set_table_creates(sst_creates.into());
    comp.set_bottom_deletes(l1_plus_compaction.lower_level.clone());
    Ok(comp)
}

fn major_compact_v3(
    ctx: &CompactionCtx,
    major_compaction: &MajorCompaction,
    id_allocator: &mut LocalIdAllocator,
) -> Result<pb::MajorCompaction> {
    let req = &ctx.req;
    let fs = &ctx.dfs;
    let opts = dfs::Options::default().with_shard(req.shard_id, req.shard_ver);
    let mut ret = pb::MajorCompaction::new();
    ret.mut_old_blob_tables()
        .extend_from_slice(&major_compaction.blob_tables);
    let blob_tables = load_blob_tables(fs.clone(), &major_compaction.blob_tables, opts)?;
    let l0_files = load_table_files(
        &major_compaction.l0_tables,
        fs.clone(),
        opts,
        ctx.local_dir.as_ref(),
        ctx.for_restore,
    )?;
    let mut l0_tbls = files_to_l0_tables(l0_files, ctx.encryption_key.clone());
    l0_tbls.sort_by(|a, b| b.version().cmp(&a.version()));
    l0_tbls.iter().for_each(|tbl| {
        let mut tbl_delete = pb::TableDelete::new();
        tbl_delete.set_id(tbl.id());
        ret.mut_sstable_change()
            .mut_table_deletes()
            .push(tbl_delete);
    });
    for cf in 0..NUM_CFS {
        let mut iters: Vec<Box<dyn table::Iterator>> = vec![];
        for l0_tbl in &l0_tbls {
            if let Some(tbl) = l0_tbl.get_cf(cf) {
                let iter = tbl.new_iterator(false, false);
                iters.push(iter);
            }
        }
        if let Some(sstables) = major_compaction.ln_tables.get(&cf) {
            for (level, table_ids) in sstables.iter() {
                table_ids.iter().for_each(|t| {
                    let mut tbl_delete = pb::TableDelete::new();
                    tbl_delete.set_cf(cf as i32);
                    tbl_delete.set_level(*level as u32);
                    tbl_delete.set_id(*t);
                    ret.mut_sstable_change()
                        .mut_table_deletes()
                        .push(tbl_delete);
                });
                let files = load_table_files(
                    table_ids,
                    fs.clone(),
                    opts,
                    ctx.local_dir.as_ref(),
                    ctx.for_restore,
                )?;
                let mut tables = files_to_tables(files, ctx.encryption_key.clone());
                tables.sort_by(|a, b| a.smallest().cmp(&b.smallest()));
                let level_concat_iter =
                    Box::new(ConcatIterator::new_with_tables(tables, false, false));
                iters.push(level_concat_iter);
            }
        }
        if iters.is_empty() {
            continue;
        }
        let mut iter = table::new_merge_iterator(iters, false);
        let (sst_creates, blob_table_creates) = compact_for_cf(
            ctx,
            &mut iter,
            major_compaction.safe_ts,
            opts,
            cf,
            CF_LEVELS[cf] as u32,
            &major_compaction.sst_config,
            &major_compaction.bt_config,
            false,
            &blob_tables,
            id_allocator,
        )?;
        for sst_create in sst_creates {
            ret.mut_sstable_change()
                .mut_table_creates()
                .push(sst_create);
        }
        for blob_table_create in blob_table_creates {
            ret.mut_new_blob_tables().push(blob_table_create);
        }
    }
    Ok(ret)
}

async fn compact_table_for_columnar(
    ctx: &CompactionCtx,
    reader: &mut dyn ColumnarFilterReader,
    file_builder: &mut ColumnarFileBuilder,
    schema: &Schema,
    target_lvl: u32,
    cnt: &mut usize,
    tx: mpsc::Sender<dfs::Result<pb::ColumnarCreate>>,
    columnar_config: &ColumnarTableBuildOptions,
    id_allocator: &mut LocalIdAllocator,
) -> Result<()> {
    let fs = &ctx.dfs;
    let opts = dfs::Options::default()
        .with_type(FileType::Columnar)
        .with_shard(ctx.req.shard_id, ctx.req.shard_ver);
    let mut block = Block::new(schema);
    let mut res = reader
        .read_block(&mut block, columnar_config.pack_max_row_count)
        .await?;
    let mut row_count = 0;
    let mut tbl_builder = ColumnarTableBuilder::new(
        schema.clone(),
        *columnar_config,
        false,
        ctx.encryption_key.clone(),
        file_builder.file_id,
    );
    let mut block_offset = 0;
    while res > 0 {
        row_count += res;
        block_offset = tbl_builder.append_block(&block, block_offset);
        if block_offset < block.length() {
            res = block.length() - block_offset;
            let estimated_size = tbl_builder.get_estimated_size() + file_builder.estimated_size;
            if estimated_size > columnar_config.max_columnar_table_size {
                file_builder.add_table(tbl_builder);
                *cnt += 1;
                persist_columnar_file(target_lvl, file_builder, tx.clone(), fs.clone(), opts);
                file_builder.reset(id_allocator.alloc_id().await);
                tbl_builder = ColumnarTableBuilder::new(
                    schema.clone(),
                    *columnar_config,
                    false,
                    ctx.encryption_key.clone(),
                    file_builder.file_id,
                );
            }
        } else {
            block.reset();
            block_offset = 0;
            res = reader
                .read_block(&mut block, columnar_config.pack_max_row_count)
                .await?;
        }
    }
    if row_count > 0 {
        file_builder.add_table(tbl_builder);
    }
    Ok(())
}

async fn transform_for_columnar(
    ctx: &CompactionCtx,
    tbls: &Vec<SsTable>,
    blob_tbls: Option<Arc<HashMap<u64, BlobTable>>>,
    overlap_tables: Vec<i64>,
    schema_file: &SchemaFile,
    target_lvl: u32,
    columnar_config: &ColumnarTableBuildOptions,
    id_allocator: &mut LocalIdAllocator,
) -> Result<Vec<ColumnarCreate>> {
    let fs = &ctx.dfs;
    let opts = dfs::Options::default()
        .with_shard(ctx.req.shard_id, ctx.req.shard_ver)
        .with_type(FileType::Columnar);
    let (tx, rx) = tikv_util::mpsc::bounded(ctx.req.file_ids.len());
    let mut file_builder = ColumnarFileBuilder::new(
        id_allocator.alloc_id().await,
        None,
        ctx.encryption_key.clone(),
    );
    let mut cnt = 0;
    for table_id in overlap_tables {
        let schema = schema_file.get_table(table_id).unwrap();
        let mut columnar_readers: Vec<Box<dyn ColumnarReader>> = vec![];
        for tbl in tbls {
            let iter = tbl.new_iterator(false, false);
            let reader = ColumnarRowTableReader::new(
                schema.clone(),
                iter,
                blob_tbls.clone(),
                true,
                ctx.encryption_key.clone(),
            );
            columnar_readers.push(Box::new(reader));
        }
        let merge_reader = ColumnarMergeReader::new(schema.clone(), columnar_readers);
        let mut compact_reader =
            ColumnarCompactReader::new(Box::new(merge_reader), target_lvl, schema, ctx.req.safe_ts);
        compact_reader.set_unbounded_handle_range().await?;
        compact_table_for_columnar(
            ctx,
            &mut compact_reader,
            &mut file_builder,
            schema,
            target_lvl,
            &mut cnt,
            tx.clone(),
            columnar_config,
            id_allocator,
        )
        .await?;
    }
    if file_builder.num_tables() > 0 {
        cnt += 1;
        persist_columnar_file(target_lvl, &mut file_builder, tx, fs.clone(), opts);
    }
    let mut errors = vec![];
    let mut columnar_creates = Vec::with_capacity(cnt);
    for _ in 0..cnt {
        match rx.recv().unwrap() {
            Err(err) => errors.push(err),
            Ok(tbl_create) => {
                columnar_creates.push(tbl_create);
            }
        }
    }
    if !errors.is_empty() {
        return Err(errors.pop().unwrap().into());
    }

    Ok(columnar_creates)
}

async fn columnar_major_compact(
    ctx: &CompactionCtx,
    major_compaction: &ColumnarMajorCompaction,
    id_allocator: &mut LocalIdAllocator,
) -> Result<pb::ColumnarCompaction> {
    let mut ret = pb::ColumnarCompaction::new();
    ret.set_snap_version(major_compaction.snap_version);
    ret.set_target_level(2);
    if major_compaction.snap_version == 0 {
        let columnar_changes = ret.mut_columnar_change();
        for &(level, file_id) in major_compaction.old_columnar_tables.iter() {
            let mut tbl_delete = pb::ColumnarDelete::new();
            tbl_delete.set_level(level as u32);
            tbl_delete.set_id(file_id);
            columnar_changes.mut_columnar_deletes().push(tbl_delete);
        }
        return Ok(ret);
    }
    // Set row l0s for filter the new flushed l0 tables during apply major
    // compaction
    ret.set_row_l0s(major_compaction.l0_tables.clone());
    let req = &ctx.req;
    let tag = req.get_tag();
    let fs = &ctx.dfs;
    let opts = dfs::Options::default().with_shard(req.shard_id, req.shard_ver);
    let schema_file_data = load_table_files(
        &[major_compaction.schema_file_id],
        fs.clone(),
        dfs::Options::default().with_type(FileType::Schema),
        ctx.local_dir.as_ref(),
        false,
    )?
    .pop()
    .unwrap();
    let schema_file = SchemaFile::open(schema_file_data)?;
    let columnar_changes = ret.mut_columnar_change();
    let blob_tbls = load_blob_tables(fs.clone(), &major_compaction.blob_tables, opts)?;
    let l0_files = load_table_files(
        &major_compaction.l0_tables,
        fs.clone(),
        opts,
        ctx.local_dir.as_ref(),
        ctx.for_restore,
    )?;
    let mut l0_tbls = files_to_l0_tables(l0_files, ctx.encryption_key.clone());
    l0_tbls.sort_by(|a, b| b.version().cmp(&a.version()));
    let mut tbls = vec![];
    let mut smallest: Option<Vec<u8>> = None;
    let mut biggest: Option<Vec<u8>> = None;
    for l0_tbl in &l0_tbls {
        if let Some(tbl) = l0_tbl.get_cf(WRITE_CF) {
            if smallest.is_none() || smallest.as_ref().unwrap().deref() > tbl.smallest().deref() {
                smallest = Some(tbl.smallest().deref().to_vec());
            }
            if biggest.is_none() || biggest.as_ref().unwrap().deref() < tbl.biggest().deref() {
                biggest = Some(tbl.biggest().deref().to_vec());
            }
            tbls.push(tbl.clone());
        }
    }
    for (_, table_ids) in major_compaction.ln_tables.iter() {
        let files = load_table_files(
            table_ids,
            fs.clone(),
            opts,
            ctx.local_dir.as_ref(),
            ctx.for_restore,
        )?;
        let mut tables = files_to_tables(files, ctx.encryption_key.clone());
        tables.sort_by(|a, b| a.smallest().cmp(&b.smallest()));
        tables.iter().for_each(|tbl| {
            if smallest.is_none() || smallest.as_ref().unwrap().deref() > tbl.smallest().deref() {
                smallest = Some(tbl.smallest().deref().to_vec());
            }
            if biggest.is_none() || biggest.as_ref().unwrap().deref() < tbl.biggest().deref() {
                biggest = Some(tbl.biggest().deref().to_vec());
            }
            tbls.push(tbl.clone());
        });
    }
    if tbls.is_empty() {
        return Ok(ret);
    }
    let overlap_tables = schema_file.overlap_tables(
        InnerKey::from_inner_buf(smallest.as_ref().unwrap()),
        InnerKey::from_inner_buf(biggest.as_ref().unwrap()),
    );
    if overlap_tables.is_empty() {
        return Ok(ret);
    }

    for &(level, file_id) in major_compaction.old_columnar_tables.iter() {
        let mut tbl_delete = pb::ColumnarDelete::new();
        tbl_delete.set_level(level as u32);
        tbl_delete.set_id(file_id);
        columnar_changes.mut_columnar_deletes().push(tbl_delete);
    }

    let columnar_creates = transform_for_columnar(
        ctx,
        &tbls,
        Some(Arc::new(blob_tbls)),
        overlap_tables,
        &schema_file,
        2,
        &major_compaction.columnar_config,
        id_allocator,
    )
    .await?;
    for columnar_create in columnar_creates {
        columnar_changes
            .mut_columnar_creates()
            .push(columnar_create);
    }
    info!(
        "{} columnar_major_compaction, tbl_changes: {:?}",
        tag, columnar_changes
    );
    Ok(ret)
}

async fn columnar_compact(
    ctx: &CompactionCtx,
    columnar_compaction: &ColumnarCompaction,
    id_allocator: &mut LocalIdAllocator,
) -> Result<pb::ColumnarCompaction> {
    if !columnar_compaction.source_row_files.is_empty() {
        convert_row_file_to_columnar_file(ctx, columnar_compaction, id_allocator).await
    } else {
        compact_columnar_files(ctx, columnar_compaction, id_allocator).await
    }
}

async fn convert_row_file_to_columnar_file(
    ctx: &CompactionCtx,
    columnar_compaction: &ColumnarCompaction,
    id_allocator: &mut LocalIdAllocator,
) -> Result<pb::ColumnarCompaction> {
    let mut ret = pb::ColumnarCompaction::new();
    ret.set_snap_version(columnar_compaction.snap_version);
    ret.set_target_level(columnar_compaction.level);
    let row_l0s: Vec<u64> = columnar_compaction
        .source_row_files
        .iter()
        .filter(|(lvl, _)| *lvl == 0)
        .map(|(_, id)| *id)
        .collect();
    ret.set_row_l0s(row_l0s);
    let opts = dfs::Options::default()
        .with_type(FileType::Schema)
        .with_shard(ctx.req.shard_id, ctx.req.shard_ver);
    let runtime = ctx.dfs.get_runtime();
    let schema_file_id = columnar_compaction.schema_file_id;
    let schema_file_data = runtime.block_on(ctx.dfs.read_file(schema_file_id, opts))?;
    let schema_file = SchemaFile::open(Arc::new(InMemFile::new(schema_file_id, schema_file_data)))?;
    let l0_ids: Vec<u64> = columnar_compaction
        .source_row_files
        .iter()
        .map(|(_, id)| *id)
        .collect();
    let opts = dfs::Options::default().with_shard(ctx.req.shard_id, ctx.req.shard_ver);
    let l0_files = load_table_files(
        &l0_ids,
        ctx.dfs.clone(),
        opts,
        ctx.local_dir.as_ref(),
        ctx.for_restore,
    )?;
    let l0_tbls = files_to_l0_tables(l0_files, ctx.encryption_key.clone());
    let smallest = l0_tbls.iter().map(|l0| l0.smallest()).min().unwrap();
    let biggest = l0_tbls.iter().map(|l0| l0.biggest()).max().unwrap();
    let overlap_tables = schema_file.overlap_tables(smallest, biggest);
    if overlap_tables.is_empty() {
        return Ok(ret);
    }
    let mut file_builder = ColumnarFileBuilder::new(
        id_allocator.alloc_id().await,
        Some(columnar_compaction.snap_version),
        ctx.encryption_key.clone(),
    );
    let mut cnt = 0;
    let (tx, rx) = mpsc::bounded(ctx.req.file_ids.len());
    for overlap_table in overlap_tables {
        let schema = schema_file.get_table(overlap_table).unwrap();
        let mut columnar_readers: Vec<Box<dyn ColumnarReader>> = vec![];
        for l0_tbl in &l0_tbls {
            if let Some(tbl) = l0_tbl.get_cf(WRITE_CF) {
                let iter = tbl.new_iterator(false, false);
                let reader = ColumnarRowTableReader::new(
                    schema.clone(),
                    iter,
                    None,
                    true,
                    ctx.encryption_key.clone(),
                );
                columnar_readers.push(Box::new(reader));
            }
        }
        if columnar_readers.is_empty() {
            continue;
        }
        let merge_reader = ColumnarMergeReader::new(schema.clone(), columnar_readers);
        let mut compact_reader =
            ColumnarCompactReader::new(Box::new(merge_reader), 0, schema, ctx.req.safe_ts);
        compact_reader.set_unbounded_handle_range().await?;
        compact_table_for_columnar(
            ctx,
            &mut compact_reader,
            &mut file_builder,
            schema,
            0,
            &mut cnt,
            tx.clone(),
            &columnar_compaction.columnar_config,
            id_allocator,
        )
        .await?;
    }
    if file_builder.num_tables() > 0 {
        cnt += 1;
        persist_columnar_file(0, &mut file_builder, tx, ctx.dfs.clone(), opts);
    }
    let change = ret.mut_columnar_change();
    let mut errors = vec![];
    for _ in 0..cnt {
        match rx.recv().unwrap() {
            Err(err) => errors.push(err),
            Ok(col_create) => {
                change.mut_columnar_creates().push(col_create);
            }
        }
    }
    if !errors.is_empty() {
        return Err(errors.pop().unwrap().into());
    }
    Ok(ret)
}

async fn compact_columnar_files(
    ctx: &CompactionCtx,
    columnar_compaction: &ColumnarCompaction,
    id_allocator: &mut LocalIdAllocator,
) -> Result<pb::ColumnarCompaction> {
    if columnar_compaction.level == 0 {
        compact_columnar_l0_files(ctx, columnar_compaction, id_allocator).await
    } else {
        compact_columnar_l1_files(ctx, columnar_compaction, id_allocator).await
    }
}

async fn compact_columnar_l0_files(
    ctx: &CompactionCtx,
    columnar_compaction: &ColumnarCompaction,
    id_allocator: &mut LocalIdAllocator,
) -> Result<pb::ColumnarCompaction> {
    let mut ret = pb::ColumnarCompaction::default();
    ret.set_snap_version(columnar_compaction.snap_version);
    ret.set_target_level(1);
    let tag = ctx.req.get_tag();
    let fs = &ctx.dfs;
    let schema_file_data = load_table_files(
        &[columnar_compaction.schema_file_id],
        fs.clone(),
        dfs::Options::default().with_type(FileType::Schema),
        ctx.local_dir.as_ref(),
        false,
    )?
    .pop()
    .unwrap();
    let schema_file = SchemaFile::open(schema_file_data)?;
    let opts = dfs::Options::default()
        .with_type(FileType::Columnar)
        .with_shard(ctx.req.shard_id, ctx.req.shard_ver);
    let tbl_changes = ret.mut_columnar_change();
    let col_file_ids: Vec<u64> = columnar_compaction
        .source_columnar_files
        .iter()
        .map(|f| {
            assert_eq!(f.0, 0);
            f.1
        })
        .collect();
    for id in &col_file_ids {
        let mut col_delete = pb::ColumnarDelete::default();
        col_delete.set_id(*id);
        col_delete.set_level(0);
        tbl_changes.mut_columnar_deletes().push(col_delete);
    }
    let col_files = load_table_files(
        &col_file_ids,
        fs.clone(),
        opts,
        ctx.local_dir.as_ref(),
        false,
    )?;
    let col_tbls = files_to_columnar_tables(col_files);
    let snap_version = col_tbls
        .iter()
        .map(|f| f.get_l0_version().unwrap())
        .max()
        .unwrap();
    let mut smallest = col_tbls[0].get_smallest();
    let mut biggest = col_tbls[0].get_biggest();
    for columnar_file in &col_tbls {
        if smallest > columnar_file.get_smallest() {
            smallest = columnar_file.get_smallest();
        }
        if biggest < columnar_file.get_biggest() {
            biggest = columnar_file.get_biggest();
        }
    }
    let overlap_tables = schema_file.overlap_tables(smallest, biggest);
    if overlap_tables.is_empty() {
        return Ok(ret);
    }
    let mut file_builder = ColumnarFileBuilder::new(
        id_allocator.alloc_id().await,
        Some(snap_version),
        ctx.encryption_key.clone(),
    );
    let (tx, rx) = mpsc::bounded(ctx.req.file_ids.len());
    let mut cnt = 0;
    for table_id in overlap_tables {
        let schema = schema_file.get_table(table_id).unwrap();
        let mut readers: Vec<Box<dyn ColumnarReader>> = vec![];
        for columnar_file in &col_tbls {
            if !columnar_file.has_table(table_id) {
                continue;
            }
            let reader =
                ColumnarTableReader::new(columnar_file, schema.clone(), ctx.encryption_key.clone());
            readers.push(Box::new(reader));
        }
        if readers.is_empty() {
            continue;
        }
        let merge_reader = ColumnarMergeReader::new(schema.clone(), readers);
        let mut compact_reader =
            ColumnarCompactReader::new(Box::new(merge_reader), 1, schema, ctx.req.safe_ts);
        compact_reader.set_unbounded_handle_range().await?;
        compact_table_for_columnar(
            ctx,
            &mut compact_reader,
            &mut file_builder,
            schema,
            1,
            &mut cnt,
            tx.clone(),
            &columnar_compaction.columnar_config,
            id_allocator,
        )
        .await?;
    }
    if file_builder.num_tables() > 0 {
        cnt += 1;
        persist_columnar_file(1, &mut file_builder, tx, ctx.dfs.clone(), opts);
    }
    let mut errors = vec![];
    for _ in 0..cnt {
        match rx.recv().unwrap() {
            Err(err) => errors.push(err),
            Ok(col_create) => {
                tbl_changes.mut_columnar_creates().push(col_create);
            }
        }
    }
    if !errors.is_empty() {
        return Err(errors.pop().unwrap().into());
    }
    info!(
        "{} compact_columnar_l0_files done, tbl_changes: {:?}",
        tag, tbl_changes
    );
    Ok(ret)
}

async fn compact_columnar_l1_files(
    ctx: &CompactionCtx,
    columnar_compaction: &ColumnarCompaction,
    id_allocator: &mut LocalIdAllocator,
) -> Result<pb::ColumnarCompaction> {
    let mut ret = pb::ColumnarCompaction::default();
    ret.set_snap_version(columnar_compaction.snap_version);
    ret.set_target_level(2);
    let tag = ctx.req.get_tag();
    let fs = &ctx.dfs;
    let schema_file_data = load_table_files(
        &[columnar_compaction.schema_file_id],
        fs.clone(),
        dfs::Options::default().with_type(FileType::Schema),
        ctx.local_dir.as_ref(),
        false,
    )?
    .pop()
    .unwrap();
    let schema_file = SchemaFile::open(schema_file_data)?;
    let opts = dfs::Options::default()
        .with_type(FileType::Columnar)
        .with_shard(ctx.req.shard_id, ctx.req.shard_ver);
    let tbl_changes = ret.mut_columnar_change();
    let (l1_col_file_ids, l2_col_file_ids): (Vec<u64>, Vec<u64>) = columnar_compaction
        .source_columnar_files
        .iter()
        .partition_map(|f| {
            if f.0 == 1 {
                Either::Left(f.1)
            } else if f.0 == 2 {
                Either::Right(f.1)
            } else {
                panic!("invalid file level {}", f.0);
            }
        });
    let l1_tbl_files = load_table_files(
        &l1_col_file_ids,
        fs.clone(),
        opts,
        ctx.local_dir.as_ref(),
        false,
    )?;
    if l1_tbl_files.is_empty() {
        return Ok(ret);
    }
    let l1_tbls = files_to_columnar_tables(l1_tbl_files);
    let mut smallest = l1_tbls.iter().map(|f| f.get_smallest()).min().unwrap();
    let mut biggest = l1_tbls.iter().map(|f| f.get_biggest()).max().unwrap();
    for tbl in &l1_tbls {
        let mut col_delete = pb::ColumnarDelete::default();
        col_delete.set_id(tbl.get_file().id());
        col_delete.set_level(1);
        tbl_changes.mut_columnar_deletes().push(col_delete);
    }

    let l2_tbl_files = load_table_files(
        &l2_col_file_ids,
        fs.clone(),
        opts,
        ctx.local_dir.as_ref(),
        false,
    )?;
    let mut l2_tbls = files_to_columnar_tables(l2_tbl_files);
    l2_tbls.sort_by(|a, b| a.get_smallest().cmp(&b.get_smallest()));
    for tbl in &l2_tbls {
        smallest = smallest.min(tbl.get_smallest());
        biggest = biggest.max(tbl.get_biggest());
        let mut col_delete = pb::ColumnarDelete::default();
        col_delete.set_id(tbl.get_file().id());
        col_delete.set_level(2);
        tbl_changes.mut_columnar_deletes().push(col_delete);
    }

    let mut overlap_tables = schema_file.overlap_tables(smallest, biggest);
    if overlap_tables.is_empty() {
        return Ok(ret);
    }
    overlap_tables.sort();
    let mut file_builder = ColumnarFileBuilder::new(
        id_allocator.alloc_id().await,
        None,
        ctx.encryption_key.clone(),
    );
    let (tx, rx) = mpsc::bounded(ctx.req.file_ids.len());
    let mut cnt = 0;
    for table_id in overlap_tables {
        let schema = schema_file.get_table(table_id).unwrap();
        let mut readers: Vec<Box<dyn ColumnarReader>> = vec![];
        for columnar_file in &l1_tbls {
            if !columnar_file.has_table(table_id) {
                continue;
            }
            let reader =
                ColumnarTableReader::new(columnar_file, schema.clone(), ctx.encryption_key.clone());
            readers.push(Box::new(reader));
        }
        if readers.is_empty() {
            continue;
        }
        let concat_reader =
            ColumnarConcatReader::new(&l2_tbls, schema.clone(), ctx.encryption_key.clone());
        readers.push(Box::new(concat_reader));
        let merge_reader = ColumnarMergeReader::new(schema.clone(), readers);
        let mut compact_reader =
            ColumnarCompactReader::new(Box::new(merge_reader), 2, schema, ctx.req.safe_ts);
        compact_reader.set_unbounded_handle_range().await?;
        compact_table_for_columnar(
            ctx,
            &mut compact_reader,
            &mut file_builder,
            schema,
            2,
            &mut cnt,
            tx.clone(),
            &columnar_compaction.columnar_config,
            id_allocator,
        )
        .await?;
    }
    if file_builder.num_tables() > 0 {
        cnt += 1;
        persist_columnar_file(2, &mut file_builder, tx, ctx.dfs.clone(), opts);
    }
    let mut errors = vec![];
    for _ in 0..cnt {
        match rx.recv().unwrap() {
            Err(err) => errors.push(err),
            Ok(col_create) => {
                tbl_changes.mut_columnar_creates().push(col_create);
            }
        }
    }
    if !errors.is_empty() {
        return Err(errors.pop().unwrap().into());
    }
    info!(
        "{} compact_columnar_l1_files done, tbl_changes: {:?}",
        tag, tbl_changes
    );

    Ok(ret)
}

async fn update_vector_index(
    ctx: &CompactionCtx,
    update_vec_idx: &VectorIndexUpdate,
    id_allocator: &mut LocalIdAllocator,
) -> Result<pb::UpdateVectorIndex> {
    let fs = &ctx.dfs;
    let schema_file_data = load_table_files(
        &[update_vec_idx.schema_file_id],
        fs.clone(),
        dfs::Options::default().with_type(FileType::Schema),
        ctx.local_dir.as_ref(),
        false,
    )?
    .pop()
    .unwrap();
    let schema_file = SchemaFile::open(schema_file_data)?;
    let full_schema = schema_file.get_table(update_vec_idx.table_id).unwrap();
    let vec_idx_def = full_schema
        .vector_indexes
        .iter()
        .find(|idx| idx.index_id == update_vec_idx.index_id)
        .unwrap();
    let mut schema_buf = full_schema.to_schema_buf();
    schema_buf
        .columns
        .retain(|c| c.get_column_id() == vec_idx_def.col_id);
    let vector_col_schema = Schema::new(schema_buf);
    let dimension = vector_col_schema.columns[0].get_column_len() as usize;
    let metric = vec_idx_def
        .specs
        .get(VECTOR_INDEX_SPEC_KEY_DISTANCE_METRIC)
        .unwrap()
        .to_str_lossy();
    let opts = dfs::Options::default()
        .with_type(FileType::Columnar)
        .with_shard(ctx.req.shard_id, ctx.req.shard_ver);
    let col_file_ids: Vec<u64> = update_vec_idx
        .col_file_ids
        .iter()
        .map(|&(id, _)| id)
        .collect();
    let files = load_table_files(
        &col_file_ids,
        fs.clone(),
        opts,
        ctx.local_dir.as_ref(),
        false,
    )?;
    let mut columnar_files = vec![];
    for file in files {
        let columnar_file = ColumnarFile::open(file)?;
        columnar_files.push(columnar_file);
    }
    let mut readers: Vec<Box<dyn ColumnarReader>> = vec![];
    for columnar_file in &columnar_files {
        let reader = ColumnarTableReader::new(
            columnar_file,
            vector_col_schema.clone(),
            ctx.encryption_key.clone(),
        );
        readers.push(Box::new(reader));
    }
    let mut vec_builder = VectorIndexBuilder::new(
        dimension,
        metric.as_ref(),
        update_vec_idx.snap_version,
        update_vec_idx.table_id,
        update_vec_idx.index_id,
        update_vec_idx.col_id,
        vector_col_schema.is_common_handle(),
    )?;
    let mut block = Block::new(&vector_col_schema);
    let mut merge_reader = ColumnarMergeReader::new(vector_col_schema, readers);
    merge_reader.seek(&[]).await?;
    let mut res = merge_reader.read(&mut block, 1024).await?;
    while res > 0 {
        vec_builder.add_block(&block, 0)?;
        block.reset();
        res = merge_reader.read(&mut block, 1024).await?;
    }
    let buf = vec_builder.build()?;
    let file_id = id_allocator.alloc_id().await;
    fs.create(file_id, buf.into(), opts.with_type(FileType::VectorIndex))
        .await?;
    let mut vec_idx_file = pb::VectorIndexFile::new();
    vec_idx_file.id = file_id;
    vec_idx_file.snap_version = update_vec_idx.snap_version;
    vec_idx_file.smallest = vec_builder.smallest.clone();
    vec_idx_file.biggest = vec_builder.biggest.clone();
    let mut ret = pb::UpdateVectorIndex::new();
    ret.set_table_id(update_vec_idx.table_id);
    ret.set_index_id(update_vec_idx.index_id);
    ret.set_removed(update_vec_idx.remove_file_ids.clone());
    ret.set_added(vec![vec_idx_file].into());
    info!("update vector index result {:?}", ret);
    Ok(ret)
}

pub(crate) enum CompactMsg {
    /// The shard is ready to be compacted / destroyed range / truncated ts.
    Compact(IdVer),

    /// Compaction finished.
    Finish {
        task_id: u64,
        id_ver: IdVer,
        // This variant is too large(256 bytes). Box it to reduce size.
        result: Box<Option<Result<pb::ChangeSet>>>,
    },

    /// The compaction change set is applied to the engine, so the shard is
    /// ready for next compaction.
    Applied(IdVer),

    /// Clear message is sent when a shard changed its version or set to
    /// inactive. Then all the previous tasks will be discarded.
    /// This simplifies the logic, avoid race condition.
    Clear(IdVer),

    /// Pause message is used to pause compaction when a shard becomes leader
    /// but there are scheduled compactions.
    ///
    /// In this scene, we need to prevent new compaction tasks. Otherwise,
    /// compaction based on stale shard meta may be conflict with scheduled
    /// compactions.
    ///
    /// `seq` is the maximum log index sequence of scheduled compactions.
    Pause { id_ver: IdVer, seq: u64 },

    /// Resume compaction for previously blocked keyspace shards.
    UnblockKeyspace,

    /// Stop background compact thread.
    Stop,
}

pub(crate) struct CompactRunner {
    engine: Engine,
    rx: mpsc::Receiver<CompactMsg>,
    /// task_id is the identity of each compaction job.
    ///
    /// When a shard is set to inactive, all the compaction jobs of this shard
    /// should be discarded, then when it is set to active again, old result
    /// may arrive and conflict with the new tasks. So we use task_id to
    /// detect and discard obsolete tasks.
    task_id: u64,
    /// `running` contains shards' running compaction `task_id`s.
    running: HashMap<IdVer, u64 /* task_id */>,
    /// `notified` contains shards that have finished a compaction job and been
    /// notified to apply it.
    notified: HashSet<IdVer>,
    /// `pending` contains shards that want to do compaction but the job queue
    /// is full now.
    pending: HashSet<IdVer>,
    /// `paused_seq` is used to indicate that compaction should be paused until
    /// `Shard::meta_seq >= paused_seq`.
    paused_seq: HashMap<IdVer, u64>,

    /// blocked contains the keyspace shards that are blocked for compaction.
    blocked: HashMap<u32, HashSet<IdVer>>,
}

impl CompactRunner {
    pub(crate) fn new(engine: Engine, rx: mpsc::Receiver<CompactMsg>) -> Self {
        Self {
            engine,
            rx,
            task_id: 0,
            running: Default::default(),
            notified: Default::default(),
            pending: Default::default(),
            paused_seq: Default::default(),
            blocked: Default::default(),
        }
    }

    fn run(&mut self) {
        while let Ok(msg) = self.rx.recv() {
            match msg {
                CompactMsg::Compact(id_ver) => self.compact(id_ver),

                CompactMsg::Finish {
                    task_id,
                    id_ver,
                    result,
                } => self.compaction_finished(id_ver, task_id, *result),

                CompactMsg::Applied(id_ver) => self.compaction_applied(id_ver),

                CompactMsg::Clear(id_ver) => self.clear(id_ver),

                CompactMsg::Pause { id_ver, seq } => {
                    let old_seq = self.paused_seq.insert(id_ver, seq);

                    let tag = ShardTag::new(self.engine.get_engine_id(), id_ver);
                    info!("{} compaction paused", tag; "seq" => seq, "old_seq" => ?old_seq);
                    debug_assert!(
                        old_seq.map_or(true, |old| old <= seq),
                        "{}: compaction paused old_seq {:?} > seq {}",
                        tag,
                        old_seq,
                        seq
                    );
                }

                CompactMsg::UnblockKeyspace => self.schedule_unblocked_compaction(),

                CompactMsg::Stop => {
                    info!(
                        "Engine {} compaction worker receive stop msg and stop now",
                        self.engine.get_engine_id()
                    );
                    break;
                }
            }
        }
    }

    fn compact(&mut self, id_ver: IdVer) {
        // The shard is compacting, and following compaction will be trigger after
        // apply.
        if self.is_compacting(id_ver) {
            return;
        }
        if self.is_full() {
            self.pending.insert(id_ver);
            return;
        }

        let shard = match self.engine.get_shard_with_ver(id_ver.id, id_ver.ver) {
            Ok(shard) => shard,
            Err(_) => {
                self.pending.remove(&id_ver);
                return;
            }
        };
        if is_compaction_blocked(shard.keyspace_id) {
            self.blocked
                .entry(shard.keyspace_id)
                .or_default()
                .insert(id_ver);
            let tag = ShardTag::new(self.engine.get_engine_id(), id_ver);
            info!("{} compaction blocked", tag);
            return;
        }
        if self.is_paused(id_ver, shard.get_meta_sequence()) {
            return;
        }

        self.task_id += 1;
        let task_id = self.task_id;
        self.running.insert(id_ver, task_id);
        self.pending.remove(&id_ver); // Remove from pending if it's there.
        let engine = self.engine.clone();
        std::thread::spawn(move || {
            tikv_util::set_current_region(id_ver.id);
            let result = Box::new(engine.compact(shard));
            engine.send_compact_msg(CompactMsg::Finish {
                task_id,
                id_ver,
                result,
            });
        });
    }

    fn compaction_finished(
        &mut self,
        id_ver: IdVer,
        task_id: u64,
        result: Option<Result<pb::ChangeSet>>,
    ) {
        let tag = ShardTag::new(self.engine.get_engine_id(), id_ver);
        if self
            .running
            .get(&id_ver)
            .map(|id| *id != task_id)
            .unwrap_or(true)
        {
            info!(
                "shard {} discard old term compaction result {:?}",
                tag, result
            );
            return;
        }
        self.running.remove(&id_ver);
        match result {
            Some(Ok(resp)) => {
                self.engine.handle_compact_response(resp);
                self.notified.insert(id_ver);
                self.schedule_pending_compaction();
            }
            Some(Err(FallbackLocalCompactorDisabled)) => {
                error!("shard {} local compaction disabled, no need retry", tag);
            }
            Some(Err(CompactionNotRetryable(e))) => {
                error!("shard {} compaction failed {}, not retryable", tag, e);
            }
            Some(Err(TableError(table::Error::SchemaOutOfDate(e)))) => {
                error!("shard {} decode row to columnar failed {}", tag, e);
                if let Ok(shard) = self.engine.get_shard_with_ver(id_ver.id, id_ver.ver) {
                    if let Some(schema_file) = shard.get_schema_file() {
                        shard.set_outdated_schema_ver(schema_file.get_version());
                    }
                }
            }
            Some(Err(e)) => {
                error!("shard {} compaction failed {}, retrying", tag, e);
                self.compact(id_ver);
            }
            None => {
                info!("shard {} got empty compaction result", tag);
                // The compaction result can be none under complex situations. To avoid
                // compaction not making progress, we trigger it actively.
                if let Ok(shard) = self.engine.get_shard_with_ver(id_ver.id, id_ver.ver) {
                    self.engine.refresh_shard_states(&shard);
                }
                self.schedule_pending_compaction();
            }
        }
    }

    fn compaction_applied(&mut self, id_ver: IdVer) {
        // It's possible the shard is not being pending applied, because it may apply a
        // compaction from the former leader, so we use both `running` and
        // `notified` to detect whether it's compacting. It can avoid spawning
        // more compaction jobs, e.g., a new leader spawns one but not yet
        // finished, and it applies the compaction from the former leader, then the
        // new leader spawns another one. However, it's possible the new compaction is
        // finished and not yet applied, it can result in duplicated compaction
        // but no exceeding compaction jobs.
        self.notified.remove(&id_ver);
    }

    fn clear(&mut self, id_ver: IdVer) {
        self.running.remove(&id_ver);
        self.notified.remove(&id_ver);
        self.pending.remove(&id_ver);
        self.schedule_pending_compaction()
    }

    fn schedule_pending_compaction(&mut self) {
        while let Some(id_ver) = self.pick_highest_pri_pending_shard() {
            if self.is_full() {
                break;
            }
            self.pending.remove(&id_ver);
            self.compact(id_ver);
        }
    }

    fn pick_highest_pri_pending_shard(&mut self) -> Option<IdVer> {
        let engine = self.engine.clone();
        self.pending
            .retain(|id_ver| engine.get_shard_with_ver(id_ver.id, id_ver.ver).is_ok());

        // Find the shard with highest compaction priority.
        self.pending
            .iter()
            .max_by_key(|&&id_ver| {
                self.engine
                    .get_shard_with_ver(id_ver.id, id_ver.ver)
                    .ok()
                    .and_then(|shard| shard.get_compaction_priority())
            })
            .copied()
    }

    fn is_full(&self) -> bool {
        self.running.len() >= self.engine.opts.num_compactors
    }

    fn is_compacting(&self, id_ver: IdVer) -> bool {
        self.running.contains_key(&id_ver) || self.notified.contains(&id_ver)
    }

    fn is_paused(&mut self, id_ver: IdVer, meta_seq: u64) -> bool {
        match self.paused_seq.entry(id_ver) {
            Entry::Occupied(entry) => {
                let paused_seq = *entry.get();
                if meta_seq < paused_seq {
                    true
                } else {
                    entry.remove();
                    let tag = ShardTag::new(self.engine.get_engine_id(), id_ver);
                    info!("{} compaction resumed", tag; "meta_seq" => meta_seq, "paused_seq" => paused_seq);
                    false
                }
            }
            Entry::Vacant(_) => false,
        }
    }

    fn schedule_unblocked_compaction(&mut self) {
        if self.blocked.is_empty() {
            return;
        }
        for (_, blocked_ids) in self
            .blocked
            .extract_if(|keyspace_id, _| !is_compaction_blocked(*keyspace_id))
        {
            for id_ver in blocked_ids {
                let tag = ShardTag::new(self.engine.get_engine_id(), id_ver);
                info!("{} compaction unblocked", tag);
                self.pending.insert(id_ver);
            }
        }
        self.schedule_pending_compaction();
    }
}

fn is_compaction_blocked(keyspace_id: u32) -> bool {
    recovery::is_in_recovery_mode() && !recovery::in_white_list(keyspace_id)
}
