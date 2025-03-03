// Copyright 2025 TiKV Project Authors. Licensed under Apache-2.0.

mod error;
mod manifest;
mod preprocessor;

use std::{
    cmp::max,
    collections::{HashMap, HashSet},
    mem,
    path::{Path, PathBuf},
    sync::Arc,
};

use api_version::ApiV2;
use bytes::{Buf, BufMut, Bytes};
use cloud_encryption::MasterKey;
pub use error::{Error, Result};
use file_system::{IoRateLimitMode, IoRateLimiter};
use kvengine::{dfs::S3Fs, limiter::StoreLimiter, MetaIterator, ShardMeta};
use kvenginepb::ChangeSet;
use kvproto::{
    metapb,
    metapb::Peer,
    raft_cmdpb::AdminRequest,
    raft_serverpb::{RegionLocalState, StoreIdent},
};
use native_br::common::{
    collect_snapshot_meta_rlog_files, replay_wal_logs_from_backup, ReplayWalLogsContext,
};
use pd_client::PdClient;
use protobuf::Message;
use raft_proto::{eraftpb, eraftpb::Entry};
use rfengine::{
    iterator::WalIterator, RaftLogOp, RfEngine, RAFT_STATE_KEY_BYTE, REGION_META_KEY_BYTE,
    REGION_META_KEY_PREFIX, TRUNCATE_ALL_INDEX,
};
use rfenginepb::{ClusterBackupMeta, StoreBackupMeta};
use rfstore::store::{
    get_preprocess_cmd, state::RaftState, ApplyContext, ApplyMsgs, MetaChangeListener,
    PdIdAllocator, PreprocessContext, RecoverHandler, RAFT_INIT_LOG_INDEX,
};
use security::SecurityConfig;
use serde_derive::{Deserialize, Serialize};
use tikv::config::TikvConfig;
use tikv_util::{
    config::{ReadableDuration, ReadableSize},
    info, mpsc, warn,
};

use crate::{manifest::Manifest, preprocessor::Preprocessor};

#[derive(Clone)]
pub struct MergedEngineContext {
    pub pd: Arc<dyn PdClient>,
    pub fs: Arc<S3Fs>,
    pub local_dir: PathBuf,
    pub master_key: MasterKey,
    pub config: MergedEngineConfig,
    pub security_config: Arc<SecurityConfig>,
}

#[derive(Clone, Serialize, Deserialize, PartialEq, Debug)]
#[serde(default)]
#[serde(rename_all = "kebab-case")]
pub struct MergedEngineConfig {
    pub block_cache_size: ReadableSize,
    pub timeout_fetch_wal: ReadableDuration,
    pub merged_store_id: u64,
    pub mem_table_size: ReadableSize,
}

impl Default for MergedEngineConfig {
    fn default() -> Self {
        Self {
            block_cache_size: ReadableSize::mb(128),
            timeout_fetch_wal: ReadableDuration::secs(30),
            merged_store_id: 1024,
            mem_table_size: ReadableSize::mb(128),
        }
    }
}

#[derive(Clone, Debug)]
pub struct RegionProgress {
    pub keyspace_id: u32,
    pub entries: HashMap<u64, RaftLogOp>,
    pub synced_index: u64,
    pub commit_index: u64,
    pub truncated_index: u64,
    pub version: u64,
}

impl RegionProgress {
    pub fn new(keyspace_id: u32) -> Self {
        Self {
            keyspace_id,
            entries: HashMap::new(),
            synced_index: 0,
            commit_index: 0,
            truncated_index: 0,
            version: 0,
        }
    }
}

#[derive(Clone, Copy, Default, Debug)]
pub struct StoreProgress {
    pub store_id: u64,
    pub epoch: u32,
    pub offset: u64,
}

impl StoreProgress {
    pub(crate) fn encode(&self, buf: &mut Vec<u8>) {
        buf.put_u64_le(self.store_id);
        buf.put_u32_le(self.epoch);
        buf.put_u64_le(self.offset);
    }

    pub(crate) fn decode(buf: &mut impl Buf) -> Result<Self> {
        if (buf.remaining()) < 20 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "store progress buffer is too short",
            )
            .into());
        }
        let store_id = buf.get_u64_le();
        let epoch = buf.get_u32_le();
        let offset = buf.get_u64_le();
        Ok(Self {
            store_id,
            epoch,
            offset,
        })
    }
}

struct EmptyMetaIterator {}

impl MetaIterator for EmptyMetaIterator {
    fn iterate<F>(&mut self, _: F) -> kvengine::Result<()>
    where
        F: FnMut(ChangeSet),
    {
        Ok(())
    }

    fn engine_id(&self) -> u64 {
        0
    }
}

pub struct MergedEngine {
    ctx: MergedEngineContext,
    manifest: Manifest,
    region_progresses: HashMap<u64, RegionProgress>,
    updated_regions: HashMap<u64, u64>, // region_id -> region_version
    raft: RfEngine,
    kv: kvengine::Engine,
    recover_handler: RecoverHandler,
    preprocessors: HashMap<u64, Preprocessor>,
}

impl MergedEngine {
    pub fn new(ctx: MergedEngineContext, backup_meta: ClusterBackupMeta) -> Result<Self> {
        let merged_dir = ctx.local_dir.join("merged");
        let merged_cfg = rfengine::RfEngineConfig::default();
        let raft = RfEngine::open(merged_dir.as_path(), &merged_cfg, None, None)?;
        let merged_store_id = ctx.config.merged_store_id;
        if let Some(store_ident) = rfengine::load_store_ident(&raft) {
            if store_ident.get_store_id() != merged_store_id {
                panic!(
                    "store id mismatch, expect {}, got {}",
                    merged_store_id,
                    store_ident.get_store_id()
                );
            }
        } else {
            let mut store_ident = StoreIdent::new();
            store_ident.set_store_id(merged_store_id);
            rfengine::save_store_ident(&raft, &store_ident);
        }
        raft.set_engine_id(merged_store_id);
        let manifest_dir = ctx.local_dir.join("manifest");
        let mut manifest = Manifest::open(&manifest_dir)?;
        let region_progresses = if manifest.store_progresses.is_empty() {
            let (region_progresses, store_progresses) =
                Self::recover_from_backup(&ctx, &backup_meta, &raft)?;
            manifest.store_progresses = store_progresses;
            manifest.persist()?;
            region_progresses
        } else {
            Self::recover_from_merged_raft_engine(&raft)?
        };
        let mut preprocessors = HashMap::new();
        for (region_id, _) in raft.get_region_peer_map() {
            if region_id == 0 {
                continue;
            }
            let processor = Preprocessor::new(&raft, merged_store_id, region_id, &ctx.master_key);
            preprocessors.insert(region_id, processor);
        }
        let io_rate_limiter = Arc::new(IoRateLimiter::new(IoRateLimitMode::WriteOnly, true, true));
        let store_limiter = Arc::new(StoreLimiter::dummy());
        let mut recover_handler = RecoverHandler::new(raft.clone());
        recover_handler.set_merged_engine(true);
        let mut meta_iter = EmptyMetaIterator {};
        let kv = Self::init_kv_engine(
            &ctx,
            io_rate_limiter,
            store_limiter,
            &mut meta_iter,
            recover_handler.clone(),
        )
        .unwrap();
        Self::load_shards(
            &ctx,
            &kv,
            recover_handler.clone(),
            manifest.keyspace_ids.clone(),
        )?;
        Ok(Self {
            ctx,
            manifest,
            region_progresses,
            updated_regions: HashMap::new(),
            raft,
            kv,
            recover_handler,
            preprocessors,
        })
    }

    pub fn load_keyspaces(&mut self, keyspace_ids: Vec<u32>) -> Result<()> {
        let mut new_keyspaces = HashSet::new();
        for keyspace_id in keyspace_ids {
            if self.manifest.add_keyspace_id(keyspace_id) {
                new_keyspaces.insert(keyspace_id);
            }
        }
        info!("load keyspaces {:?}", new_keyspaces);
        self.manifest.persist()?;
        Self::load_shards(
            &self.ctx,
            &self.kv,
            self.recover_handler.clone(),
            new_keyspaces,
        )
    }

    fn load_shards(
        ctx: &MergedEngineContext,
        kv: &kvengine::Engine,
        mut recoverer: RecoverHandler,
        keyspace_ids: HashSet<u32>,
    ) -> Result<()> {
        let engine_id = ctx.config.merged_store_id;
        let mut metas = HashMap::new();
        recoverer.iterate(|cs| {
            let meta = ShardMeta::new(engine_id, &cs);
            if keyspace_ids.contains(&meta.range.keyspace_id) {
                info!("load keyspace insert cs {:?}", cs);
                metas.insert(meta.id, meta);
            }
        })?;
        kv.load_shards(metas, recoverer, None)?;
        Ok(())
    }

    pub fn close(&self) {
        self.raft.stop_worker(false);
        self.kv.close();
    }

    fn recover_from_backup(
        ctx: &MergedEngineContext,
        backup_meta: &ClusterBackupMeta,
        merged_raft: &RfEngine,
    ) -> Result<(HashMap<u64, RegionProgress>, HashMap<u64, StoreProgress>)> {
        let mut region_progresses = HashMap::new();
        let mut store_progresses = HashMap::new();
        let mut raftdb_pathes = Vec::new();
        for store in backup_meta.get_stores() {
            let store_progress = StoreProgress {
                store_id: store.get_store_id(),
                epoch: store.get_epoch(),
                offset: store.get_offset(),
            };
            store_progresses.insert(store.get_store_id(), store_progress);
            let mut store_config = TikvConfig::default();
            let store_path = ctx.local_dir.join(store.store_id.to_string());
            store_config.storage.data_dir = store_path.to_str().unwrap().to_string();
            store_config.raft_store.raftdb_path =
                store_path.join("raft").to_str().unwrap().to_string();
            store_config.rfengine.lightweight_backup = false;
            let origin = Self::setup_raft_engine(ctx, backup_meta, store).unwrap();
            raftdb_pathes.push(store_config.raft_store.raftdb_path);
            let region_peers_map = origin.get_region_peer_map();
            for (region_id, peer_id) in region_peers_map {
                if region_id == 0 {
                    continue;
                }
                let region_state = rfstore::store::load_last_peer_state(&origin, peer_id).unwrap();
                let keyspace_id =
                    ApiV2::get_u32_keyspace_id_by_key(region_state.get_region().get_start_key())
                        .unwrap_or_default();
                let region_version = region_state.get_region().get_region_epoch().get_version();
                let raft_state =
                    rfstore::store::load_peer_raft_state(&origin, peer_id, region_version).unwrap();
                let commit = raft_state.get_commit();
                let region_progress = region_progresses
                    .entry(region_id)
                    .or_insert(RegionProgress::new(keyspace_id));
                if raft_state.get_last_index() > commit {
                    // Fetch uncommitted entries, and insert them into region progress, so that they
                    // will be replayed when commit index advances (during sync_merged).
                    let mut entry_buf = Vec::new();
                    let low_idx = commit + 1;
                    let high_idx = raft_state.get_last_index() + 1;
                    if let Err(err) = origin.fetch_raft_entries_to(
                        peer_id,
                        low_idx,
                        high_idx,
                        None,
                        &mut entry_buf,
                    ) {
                        panic!(
                            "fetch raft entries failed for region {}, low: {}, high: {}, err: {}",
                            region_id, low_idx, high_idx, err
                        );
                    }
                    for entry in entry_buf.iter() {
                        if let Some(existing_op) = region_progress.entries.get(&entry.index) {
                            if existing_op.term >= entry.term as u32 {
                                continue;
                            }
                        }
                        region_progress
                            .entries
                            .insert(entry.index, RaftLogOp::new(entry));
                    }
                }
                // Committed entries will be replayed right away during the recovery process
                // below.
                if region_progress.commit_index >= commit {
                    continue;
                }
                let merged_commit_index = region_progress.commit_index;
                region_progress.commit_index = commit;
                region_progress.synced_index = commit;
                let truncated_index = max(
                    region_progress.truncated_index,
                    origin
                        .get_truncated_index(peer_id)
                        .unwrap_or(RAFT_INIT_LOG_INDEX),
                )
                .max(RAFT_INIT_LOG_INDEX);
                region_progress.truncated_index = truncated_index;
                region_progress.version = region_version;
                // merge states
                let mut batch = rfengine::WriteBatch::new();
                origin.iterate_peer_states(peer_id, false, |k, v| {
                    update_peer_state(&mut batch, k, v, merged_raft.get_engine_id(), region_id);
                });
                // merge raft logs
                let mut entry_buf = Vec::new();
                let low_idx = merged_commit_index.max(truncated_index) + 1;
                let high_idx = commit + 1;
                if let Err(err) =
                    origin.fetch_raft_entries_to(peer_id, low_idx, high_idx, None, &mut entry_buf)
                {
                    panic!(
                        "fetch raft entries failed for region {}, low: {}, high: {}, err: {}",
                        region_id, low_idx, high_idx, err
                    );
                }
                for entry in entry_buf.iter() {
                    batch.append_raft_log(region_id, region_id, entry);
                }
                batch.truncate_raft_log(region_id, region_id, truncated_index);
                merged_raft.write(batch).unwrap();
            }
        }
        // destroy original raft engines
        for raftdb_path in raftdb_pathes {
            let raft_path = Path::new(&raftdb_path);
            // clean up dir
            std::fs::remove_dir_all(raft_path).unwrap();
        }
        Ok((region_progresses, store_progresses))
    }

    fn recover_from_merged_raft_engine(
        merged_raft: &RfEngine,
    ) -> Result<HashMap<u64, RegionProgress>> {
        let mut region_progersses = HashMap::new();

        // Get region progresses from RfEngine.
        let region_peers_map = merged_raft.get_region_peer_map();
        for (region_id, peer_id) in region_peers_map {
            if region_id == 0 {
                continue;
            }
            let region_state = rfstore::store::load_last_peer_state(merged_raft, peer_id).unwrap();
            let region_version = region_state.get_region().get_region_epoch().get_version();
            let raft_state =
                rfstore::store::load_peer_raft_state(merged_raft, peer_id, region_version).unwrap();
            let commit = raft_state.get_commit();
            let keyspace_id =
                ApiV2::get_u32_keyspace_id_by_key(region_state.get_region().get_start_key())
                    .unwrap_or_default();
            let region_progress = region_progersses
                .entry(region_id)
                .or_insert(RegionProgress::new(keyspace_id));
            region_progress.commit_index = commit;
            region_progress.synced_index = commit;
            region_progress.truncated_index = merged_raft
                .get_truncated_index(region_id)
                .unwrap_or(RAFT_INIT_LOG_INDEX);
            region_progress.version = region_version;
        }

        Ok(region_progersses)
    }

    fn setup_raft_engine(
        ctx: &MergedEngineContext,
        backup_meta: &ClusterBackupMeta,
        store: &StoreBackupMeta,
    ) -> Result<RfEngine> {
        let mut store_config = TikvConfig::default();
        let store_path = ctx.local_dir.join(store.store_id.to_string());
        store_config.storage.data_dir = store_path.to_str().unwrap().to_string();
        store_config.raft_store.raftdb_path = store_path.join("raft").to_str().unwrap().to_string();
        store_config.rfengine.lightweight_backup = false;
        let store_id = store.get_store_id();
        let rlog_files = collect_snapshot_meta_rlog_files(
            ctx.fs.clone(),
            &ctx.fs.get_prefix(),
            backup_meta,
            store_id,
        )?;
        rfengine::lightweight_restore(
            Path::new(&store_config.raft_store.raftdb_path),
            rlog_files.snap_epoch,
            rlog_files.snap_meta,
            rlog_files.snap_rlog,
        )?;
        let raft_db_path = Path::new(&store_config.raft_store.raftdb_path);
        let data_dir = Path::new(&store_config.storage.data_dir);
        let rf_engine = RfEngine::open(
            raft_db_path,
            &store_config.rfengine,
            Some(data_dir),
            Some(store_config.dfs.clone()),
        )?;
        let ctx = ReplayWalLogsContext {
            pd_client: ctx.pd.clone(),
            dfs: ctx.fs.clone(),
            store_id,
            cluster_backup: backup_meta,
            rf_engine: &rf_engine,
            complete_wal_chunks: false,
            full_restore: true,
            fetch_wal_timeout: ctx.config.timeout_fetch_wal.0,
        };
        let tag = &format!("merged_{}", store_id);
        replay_wal_logs_from_backup(tag, &ctx, rlog_files.snap_epoch)?;
        Ok(rf_engine)
    }

    fn init_kv_engine(
        ctx: &MergedEngineContext,
        rate_limiter: Arc<IoRateLimiter>,
        store_limiter: Arc<StoreLimiter>,
        meta_iter: &mut impl kvengine::MetaIterator,
        recoverer: impl kvengine::RecoverHandler + 'static,
    ) -> Result<kvengine::Engine> {
        let kv_engine_path = ctx.local_dir.join("db");
        let mut kv_opts = kvengine::Options::default();
        kv_opts.local_dir = kv_engine_path;
        kv_opts.max_mem_table_size = ctx.config.mem_table_size.0;
        kv_opts.max_block_cache_size = ctx.config.block_cache_size.0 as i64;
        kv_opts.for_restore = true;
        let kv_conf = kvengine::KvEngineConfig::default();
        let opts = Arc::new(kv_opts);
        let id_allocator = Arc::new(PdIdAllocator::new(ctx.pd.clone()));
        let (sender, _) = mpsc::unbounded();
        let meta_change_listener = Box::new(MetaChangeListener { sender });
        let kv_engine = kvengine::Engine::open(
            ctx.fs.clone(),
            opts,
            kv_conf,
            meta_iter,
            recoverer,
            id_allocator,
            meta_change_listener,
            rate_limiter,
            store_limiter,
            None,
            ctx.master_key.clone(),
            ctx.pd.get_security_mgr(),
        )?;
        kv_engine.set_engine_id(ctx.config.merged_store_id);
        Ok(kv_engine)
    }

    pub fn get_kv(&self) -> kvengine::Engine {
        self.kv.clone()
    }

    pub fn get_raft(&self) -> RfEngine {
        self.raft.clone()
    }

    pub fn get_region_progress(&self, region_id: u64) -> Option<RegionProgress> {
        self.region_progresses.get(&region_id).cloned()
    }

    pub fn get_store_progress(&self, store_id: u64) -> Option<StoreProgress> {
        self.manifest.store_progresses.get(&store_id).cloned()
    }

    pub fn get_keyspace_regions(&self, keyspace_id: u32) -> Vec<u64> {
        let mut regions = Vec::new();
        for (&region_id, progress) in &self.region_progresses {
            if progress.keyspace_id == keyspace_id {
                regions.push(region_id);
            }
        }
        regions
    }

    pub fn update_wal(
        &mut self,
        store_id: u64,
        epoch_id: u32,
        offset: u64,
        data: Bytes,
    ) -> Result<()> {
        let cur_offset = if let Some(store_progress) = self.manifest.store_progresses.get(&store_id)
        {
            if store_progress.epoch != epoch_id || store_progress.offset != offset {
                return Err(rfengine::Error::Other(format!(
                    "{} epoch or offset mismatch, expect ({}, {}), got ({}, {}), wal_len: {}",
                    store_id,
                    epoch_id,
                    offset,
                    store_progress.epoch,
                    store_progress.offset,
                    data.len(),
                ))
                .into());
            }
            store_progress.offset
        } else {
            return Err(rfengine::Error::Other(format!(
                "store {} not found in store progresses",
                store_id
            ))
            .into());
        };
        let new_offset = cur_offset + data.len() as u64;
        let mut wal_iterator = WalIterator::new_from_chunks(data, epoch_id, offset);
        let mut origin_batches = Vec::new();
        wal_iterator.iterate_write_batch(|origin_wb| {
            origin_batches.push(origin_wb);
        })?;
        for mut origin_wb in origin_batches {
            let region_peer_map = origin_wb.get_region_peer_map();
            for (&region_id, &peer_id) in &region_peer_map {
                let progress = self.region_progresses.entry(region_id).or_insert_with(|| {
                    let bin = origin_wb
                        .get_latest_state(peer_id, region_id, REGION_META_KEY_PREFIX)
                        .unwrap_or_else(|| {
                            panic!("store {} region {} meta not found", store_id, region_id);
                        });
                    let mut region_local_state = RegionLocalState::new();
                    region_local_state.merge_from_bytes(bin).unwrap();
                    let keyspace_id = ApiV2::get_u32_keyspace_id_by_key(
                        region_local_state.get_region().get_start_key(),
                    )
                    .unwrap_or_default();
                    RegionProgress::new(keyspace_id)
                });
                if let Some(truncated_idx) = origin_wb.get_truncated_idx(peer_id) {
                    if progress.truncated_index < truncated_idx
                        && truncated_idx != TRUNCATE_ALL_INDEX
                    {
                        progress.truncated_index = truncated_idx;
                    }
                }
                if let Some(v) =
                    origin_wb.get_latest_state(peer_id, region_id, &[RAFT_STATE_KEY_BYTE])
                {
                    if !v.is_empty() {
                        let mut raft_state = RaftState::default();
                        raft_state.unmarshal(v);
                        if raft_state.get_commit() > progress.commit_index {
                            progress.commit_index = raft_state.get_commit();
                        }
                    }
                }
                if let Some(v) =
                    origin_wb.get_latest_state(peer_id, region_id, &[REGION_META_KEY_BYTE])
                {
                    if !v.is_empty() {
                        let mut region_local_state =
                            kvproto::raft_serverpb::RegionLocalState::new();
                        region_local_state.merge_from_bytes(v).unwrap();
                        let new_version = region_local_state
                            .get_region()
                            .get_region_epoch()
                            .get_version();
                        if progress.version < new_version {
                            progress.version = new_version;
                        }
                    }
                }
                origin_wb.read_peer_logs(peer_id, |logs| {
                    for log_op in logs {
                        if let Some(existing_op) = progress.entries.get(&log_op.index) {
                            if existing_op.term >= log_op.term {
                                continue;
                            }
                        }
                        progress.entries.insert(log_op.index, log_op.clone());
                    }
                });
                self.updated_regions.insert(region_id, progress.version);
            }
        }
        self.manifest
            .update_store_progress(store_id, epoch_id, new_offset);
        Ok(())
    }

    pub fn sync_merged(&mut self, apply_ctx: &mut ApplyContext) -> Result<()> {
        // Prepare context.
        let mut raft_wb = rfengine::WriteBatch::new();
        let mut remove_dependents = Vec::new();
        let mut apply_msgs = ApplyMsgs::default();
        let raft_cfg = rfstore::store::Config::default();
        let mut destroying = HashSet::new();
        let raft_engine = self.raft.clone();
        let mut ctx = PreprocessContext {
            store_id: self.ctx.config.merged_store_id,
            kv: None,
            raft: &raft_engine,
            raft_wb: &mut raft_wb,
            remove_dependents: &mut remove_dependents,
            apply_msgs: &mut apply_msgs,
            cfg: &raft_cfg,
            router: None,
            destroying: &mut destroying,
        };
        let mut updated_regions_with_ver: Vec<(u64, u64)> = self.updated_regions.drain().collect();
        updated_regions_with_ver.sort_by_key(|(_, version)| *version);
        let updated_regions = updated_regions_with_ver
            .iter()
            .map(|(region_id, _)| *region_id)
            .collect::<Vec<_>>();
        self.sync_merged_for_regions(&mut ctx, apply_ctx, &updated_regions)?;
        let mut raft_wb = rfengine::WriteBatch::new();
        for updated_region in updated_regions {
            let progress = self.region_progresses.get_mut(&updated_region).unwrap();
            if let Some(truncated) = self.raft.get_truncated_index(updated_region) {
                if progress.truncated_index > truncated {
                    raft_wb.truncate_raft_log(
                        updated_region,
                        updated_region,
                        progress.truncated_index,
                    );
                }
            }
            progress.entries.clear();
            progress.synced_index = progress.commit_index;
        }
        if !raft_wb.is_empty() {
            self.raft.write(raft_wb)?;
        }
        self.manifest.persist()?;
        Ok(())
    }

    fn sync_merged_for_regions(
        &mut self,
        ctx: &mut PreprocessContext<'_>,
        apply_ctx: &mut ApplyContext,
        updated_regions: &[u64],
    ) -> Result<()> {
        let mut new_regions = vec![];
        let merged_store_id = self.ctx.config.merged_store_id;
        for &updated_region in updated_regions {
            match self.raft.get_truncated_index(updated_region) {
                Some(truncated_idx) => {
                    if truncated_idx == TRUNCATE_ALL_INDEX {
                        // region is merged.
                        continue;
                    }
                }
                None => {
                    // newly split region is handled after parent regions.
                    new_regions.push(updated_region);
                    continue;
                }
            }
            let progress = self.region_progresses.get_mut(&updated_region).unwrap();
            let low = progress.synced_index.max(RAFT_INIT_LOG_INDEX) + 1;
            let high: u64 = progress.commit_index + 1;
            if low >= high {
                continue;
            }
            let mut preprocessor = self.preprocessors.entry(updated_region).or_insert_with(|| {
                Preprocessor::new(
                    &self.raft,
                    ctx.store_id,
                    updated_region,
                    &self.ctx.master_key,
                )
            });
            let mut hs = eraftpb::HardState::default();
            let mut preprocessor_ref = preprocessor.as_ref();
            for log_index in low..high {
                let mut entry = progress
                    .entries
                    .get(&log_index)
                    .unwrap_or_else(|| {
                        panic!(
                            "entry not found for region {}, log index {}",
                            updated_region, log_index
                        )
                    })
                    .to_entry();
                let admin_req = update_entry(&mut entry, merged_store_id);
                let err = preprocessor_ref.preprocess_committed_entry(ctx, &entry);
                if let Some(err) = err {
                    warn!("preprocess committed entry failed"; "region_id" => updated_region, "err" => ?err);
                }
                preprocessor_ref
                    .raft_state
                    .set_last_preprocessed_index(*preprocessor_ref.preprocessed_index);
                hs.set_term(1);
                hs.set_vote(updated_region);
                hs.set_commit(progress.commit_index);
                preprocessor_ref.raft_state.set_hard_state(&hs);
                preprocessor_ref.raft_state.set_last_index(log_index);
                ctx.raft_wb
                    .append_raft_log(updated_region, updated_region, &entry);
                let last_change_set = ctx.apply_msgs.get_last_change_set();
                if last_change_set.is_some() || admin_req.is_some() {
                    let shard_meta = rfstore::store::load_engine_meta(
                        &self.raft,
                        merged_store_id,
                        updated_region,
                    )
                    .unwrap();
                    preprocessor_ref.write_raft_state(ctx);
                    let wb = mem::take(ctx.raft_wb);
                    ctx.raft.write(wb)?;
                    let shard = self.kv.get_shard(updated_region).unwrap();
                    shard.sync_data_sequence(&shard_meta);
                    self.recover_handler
                        .recover_with_apply_ctx(apply_ctx, &shard, &shard_meta)?;
                }
                if let Some(admin_req) = admin_req {
                    let shard = self.kv.get_shard(updated_region).unwrap();
                    if admin_req.has_splits() {
                        let pending_split = last_change_set.unwrap();
                        self.kv.split(pending_split, RAFT_INIT_LOG_INDEX)?;
                    } else if admin_req.has_prepare_merge() {
                        self.kv.prepare_merge(shard.id, shard.ver, entry.index)?;
                    } else if admin_req.has_rollback_merge() {
                        self.kv.rollback_merge(shard.id, shard.ver, entry.index);
                    } else if admin_req.has_commit_merge() {
                        let source = last_change_set.unwrap();
                        ctx.raft
                            .iterate_peer_states(source.shard_id, false, |k, _| {
                                ctx.raft_wb
                                    .set_state(source.shard_id, source.shard_id, k, &[]);
                            });
                        ctx.raft_wb.truncate_raft_log(
                            source.shard_id,
                            source.shard_id,
                            TRUNCATE_ALL_INDEX,
                        );
                        let source_cs = self.kv.prepare_change_set(
                            source,
                            false,
                            false,
                            None,
                            None,
                            shard.get_encryption_key(),
                        )?;
                        self.kv
                            .commit_merge(shard.id, shard.ver, &source_cs, entry.index)?
                    }
                    self.preprocessors.remove(&updated_region);
                    preprocessor = self.preprocessors.entry(updated_region).or_insert_with(|| {
                        Preprocessor::new(
                            &self.raft,
                            ctx.store_id,
                            updated_region,
                            &self.ctx.master_key,
                        )
                    });
                    preprocessor_ref = preprocessor.as_ref();
                    preprocessor_ref.raft_state.set_hard_state(&hs);
                    preprocessor_ref.raft_state.set_last_index(log_index);
                    preprocessor_ref
                        .raft_state
                        .set_last_preprocessed_index(*preprocessor_ref.preprocessed_index);
                }
            }
            preprocessor_ref.write_raft_state(ctx);
        }
        if !ctx.raft_wb.is_empty() {
            let remain_updated_regions = ctx.raft_wb.get_region_peer_map();
            for updated_region in remain_updated_regions.keys() {
                let preprocessor = self.preprocessors.get_mut(updated_region).unwrap();
                let mut preprocessor_ref = preprocessor.as_ref();
                preprocessor_ref.write_raft_state(ctx);
            }
            let wb = mem::take(ctx.raft_wb);
            ctx.raft.write(wb)?;
            for &updated_region in remain_updated_regions.keys() {
                let preprocessor = self.preprocessors.get_mut(&updated_region).unwrap();
                let preprocessor_ref = preprocessor.as_ref();
                let shard_meta = preprocessor_ref.shard_meta.as_ref().unwrap();
                if let Some(shard) = self.kv.get_shard(updated_region) {
                    shard.sync_data_sequence(shard_meta);
                    self.recover_handler
                        .recover_with_apply_ctx(apply_ctx, &shard, shard_meta)
                        .unwrap();
                } else {
                    debug_assert!(
                        !self
                            .manifest
                            .keyspace_ids
                            .contains(&shard_meta.range.keyspace_id)
                    );
                }
            }
        }
        if !new_regions.is_empty() {
            if new_regions.len() == updated_regions.len() {
                panic!("all updated regions are new regions {:?}", new_regions);
            }
            self.sync_merged_for_regions(ctx, apply_ctx, &new_regions)
        } else {
            Ok(())
        }
    }
}

fn update_peer_state(
    wb: &mut rfengine::WriteBatch,
    k: &[u8],
    v: &[u8],
    store_id: u64,
    region_id: u64,
) {
    if k.starts_with(rfengine::REGION_META_KEY_PREFIX) {
        let mut origin_state = kvproto::raft_serverpb::RegionLocalState::new();
        origin_state.merge_from_bytes(v).unwrap();
        let region_version = origin_state.get_region().get_region_epoch().get_version();
        let merged_region_state = merge_region_local_state(&origin_state, store_id);
        let data = merged_region_state.write_to_bytes().unwrap();
        wb.set_state(
            region_id,
            region_id,
            &rfengine::region_state_key(region_version),
            &data,
        );
    } else {
        wb.set_state(region_id, region_id, k, v);
    }
}

// update the entry if needed and return AdminRequest for further processing
// if the entry is admin command.
fn update_entry(entry: &mut Entry, merged_store_id: u64) -> Option<AdminRequest> {
    if entry.get_entry_type() != raft_proto::eraftpb::EntryType::EntryNormal {
        // We don't need to handle conf change, set it to empty.
        entry.set_entry_type(raft_proto::eraftpb::EntryType::EntryNormal);
        entry.set_data(Bytes::new());
        return None;
    }
    if entry.get_data().is_empty() {
        return None;
    }
    let mut cmd = get_preprocess_cmd(entry)?;
    if !cmd.has_admin_request() {
        return None;
    }
    let header = cmd.mut_header();
    let region_id = header.get_region_id();
    let peer = header.mut_peer();
    peer.set_store_id(merged_store_id);
    peer.set_id(region_id);
    let epoch = header.mut_region_epoch();
    epoch.set_conf_ver(1);
    let admin_cmd = cmd.mut_admin_request();
    if admin_cmd.has_splits() {
        let splits = admin_cmd.mut_splits().mut_requests();
        for req in splits.iter_mut() {
            req.set_new_peer_ids(vec![req.new_region_id]);
        }
    } else if admin_cmd.has_prepare_merge() {
        let prepare_merge = admin_cmd.mut_prepare_merge();
        let new_target = merged_region_meta(prepare_merge.get_target(), merged_store_id);
        prepare_merge.set_target(new_target);
    } else if admin_cmd.has_commit_merge() {
        let commit_merge = admin_cmd.mut_commit_merge();
        let new_source = merged_region_meta(commit_merge.get_source(), merged_store_id);
        commit_merge.set_source(new_source);
    }
    let new_cmd = cmd.write_to_bytes().unwrap();
    entry.set_data(new_cmd.into());
    Some(cmd.take_admin_request())
}

// merged region meta has a single peer with id same as region id.
fn merged_region_meta(origin_region: &metapb::Region, store_id: u64) -> metapb::Region {
    let mut merged_peer = Peer::new();
    merged_peer.set_id(origin_region.get_id());
    merged_peer.set_store_id(store_id);
    let mut merged_region = origin_region.clone();
    merged_region.set_peers(vec![merged_peer].into());
    merged_region
}

fn merge_region_local_state(
    origin: &kvproto::raft_serverpb::RegionLocalState,
    store_id: u64,
) -> kvproto::raft_serverpb::RegionLocalState {
    let mut merged = origin.clone();
    merged.set_region(merged_region_meta(origin.get_region(), store_id));
    if merged.has_merge_state() {
        let merge_state = merged.get_merge_state();
        if merge_state.has_target() {
            let new_target = merged_region_meta(merge_state.get_target(), store_id);
            merged.mut_merge_state().set_target(new_target);
        }
    }
    merged
}
