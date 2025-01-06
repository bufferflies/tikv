// Copyright 2025 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    cmp::max,
    collections::{HashMap, HashSet},
    path::{Path, PathBuf},
    sync::Arc,
};

use bytes::Bytes;
use cloud_encryption::MasterKey;
use file_system::{IoRateLimitMode, IoRateLimiter};
use kvengine::{
    dfs, dfs::S3Fs, limiter::StoreLimiter, RecoverHandler as RecoverHandlerTrait, ShardMeta,
};
use kvproto::{metapb, metapb::Peer, raft_serverpb::StoreIdent};
use native_br::common::{
    collect_snapshot_meta_rlog_files, replay_wal_logs_from_backup, ReplayWalLogsContext,
};
use pd_client::PdClient;
use protobuf::Message;
use rfengine::{iterator::WalIterator, RfEngine, KV_ENGINE_META_KEY};
use rfenginepb::{ClusterBackupMeta, StoreBackupMeta};
use rfstore::store::{MetaChangeListener, PdIdAllocator, RecoverHandler, RAFT_INIT_LOG_INDEX};
use security::SecurityConfig;
use tikv::config::TikvConfig;
use tikv_util::{
    config::{ReadableDuration, ReadableSize},
    mpsc,
};

pub type Result<T> = std::result::Result<T, native_br::error::Error>;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Io error {0}")]
    IoError(std::io::Error),
    #[error("DFS error {0}")]
    DfsError(dfs::Error),
    #[error("Br error {0}")]
    BrError(native_br::error::Error),
    #[error("Other error {0}")]
    Other(#[from] Box<dyn std::error::Error + Send + Sync>),
}

#[derive(Clone)]
pub struct MergedEngineContext {
    pub pd: Arc<dyn PdClient>,
    pub fs: Arc<S3Fs>,
    pub local_dir: PathBuf,
    pub master_key: MasterKey,
    pub config: MergedEngineConfig,
    pub security_config: Arc<SecurityConfig>,
}

#[derive(Clone)]
pub struct MergedEngineConfig {
    pub block_cache_size: ReadableSize,
    pub timeout_fetch_wal: ReadableDuration,
    pub merged_store_id: u64,
}

#[derive(Clone, Copy)]
struct RegionProgress {
    commit_index: u64,
    truncated_index: u64,
}

impl Default for RegionProgress {
    fn default() -> Self {
        Self {
            commit_index: RAFT_INIT_LOG_INDEX,
            truncated_index: RAFT_INIT_LOG_INDEX,
        }
    }
}

pub struct MergedEngine {
    ctx: MergedEngineContext,
    origins: HashMap<u64, RfEngine>,
    region_progresses: HashMap<u64, RegionProgress>,
    raft: RfEngine,
    kv: kvengine::Engine,
    recover_handler: RecoverHandler,
}

impl MergedEngine {
    pub fn new(ctx: MergedEngineContext, backup_meta: ClusterBackupMeta) -> Self {
        let merged_dir = ctx.local_dir.join("merged");
        let merged_cfg = rfengine::RfEngineConfig::default();
        let raft = RfEngine::open(merged_dir.as_path(), &merged_cfg, None, None).unwrap();
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
        let mut region_progresses = HashMap::new();
        let mut origins = HashMap::new();
        for store in backup_meta.get_stores() {
            let mut store_config = TikvConfig::default();
            let store_path = ctx.local_dir.join(store.store_id.to_string());
            store_config.storage.data_dir = store_path.to_str().unwrap().to_string();
            store_config.raft_store.raftdb_path =
                store_path.join("raft").to_str().unwrap().to_string();
            store_config.rfengine.lightweight_backup = false;
            let origin = Self::setup_raft_engine(&ctx, &backup_meta, store).unwrap();
            let region_peers_map = origin.get_region_peer_map();
            for (region_id, peer_id) in region_peers_map {
                if region_id == 0 {
                    continue;
                }
                let region_state = rfstore::store::load_last_peer_state(&origin, peer_id).unwrap();
                let region_version = region_state.get_region().get_region_epoch().get_version();
                let raft_state =
                    rfstore::store::load_peer_raft_state(&origin, peer_id, region_version).unwrap();
                let commit = raft_state.get_commit();
                let region_progress = region_progresses
                    .entry(region_id)
                    .or_insert(RegionProgress::default());
                if region_progress.commit_index >= commit {
                    continue;
                }
                let merged_commit_index = region_progress.commit_index;
                region_progress.commit_index = commit;
                let truncated_index = max(
                    region_progress.truncated_index,
                    origin
                        .get_truncated_index(peer_id)
                        .unwrap_or(RAFT_INIT_LOG_INDEX),
                );
                region_progress.truncated_index = truncated_index;
                // merge states
                let mut batch = rfengine::WriteBatch::new();
                origin.iterate_peer_states(peer_id, false, |k, v| {
                    update_peer_state(&mut batch, k, v, merged_store_id, region_id);
                });
                // merge raft logs
                let mut entry_buf = Vec::new();
                let low_idx = merged_commit_index.max(truncated_index) + 1;
                let high_idx = commit + 1;
                origin
                    .fetch_raft_entries_to(peer_id, low_idx, high_idx, None, &mut entry_buf)
                    .unwrap();
                for entry in entry_buf.iter() {
                    batch.append_raft_log(region_id, region_id, entry);
                }
                batch.truncate_raft_log(region_id, region_id, truncated_index);
                raft.write(batch).unwrap();
            }
            origins.insert(store.store_id, origin);
        }
        let io_rate_limiter = Arc::new(IoRateLimiter::new(IoRateLimitMode::WriteOnly, true, true));
        let store_limiter = Arc::new(StoreLimiter::dummy());
        let recover_handler = RecoverHandler::new(raft.clone());
        let mut meta_iter = recover_handler.clone();
        let kv = Self::init_kv_engine(
            &ctx,
            io_rate_limiter,
            store_limiter,
            &mut meta_iter,
            recover_handler.clone(),
        )
        .unwrap();
        Self {
            ctx,
            origins,
            region_progresses,
            raft,
            kv,
            recover_handler,
        }
    }

    fn setup_raft_engine(
        ctx: &MergedEngineContext,
        backup_meta: &ClusterBackupMeta,
        store: &StoreBackupMeta,
    ) -> native_br::Result<RfEngine> {
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
    ) -> kvengine::Result<kvengine::Engine> {
        let kv_engine_path = ctx.local_dir.join("db");
        let mut kv_opts = kvengine::Options::default();
        kv_opts.local_dir = kv_engine_path;
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

    pub fn update_wal(
        &mut self,
        store_id: u64,
        epoch_id: u32,
        offset: u64,
        data: Bytes,
    ) -> rfengine::Result<()> {
        let origin = self.origins.get(&store_id).unwrap();
        let (cur_epoch, cur_offset) = origin.get_epoch_offset();
        if cur_epoch != epoch_id || cur_offset != offset {
            return Err(rfengine::Error::Other(format!(
                "{} epoch or offset mismatch, expect ({}, {}), got ({}, {}), wal_len: {}",
                store_id,
                epoch_id,
                offset,
                cur_epoch,
                cur_offset,
                data.len(),
            )));
        }
        let mut wal_iterator = WalIterator::new_from_chunks(data, epoch_id, offset);
        let mut err: Option<rfengine::Error> = None;
        let mut batch = rfengine::WriteBatch::new();
        let mut updated_regions = HashSet::new();
        wal_iterator.iterate_write_batch(|mut origin_wb| {
            if err.is_some() {
                return;
            }
            let region_peer_map = origin_wb.get_region_peer_map();
            origin.apply(&mut origin_wb);
            for (region_id, peer_id) in region_peer_map {
                let region_state = rfstore::store::load_last_peer_state(origin, peer_id).unwrap();
                let region_version = region_state.get_region().get_region_epoch().get_version();
                let raft_state =
                    rfstore::store::load_peer_raft_state(origin, peer_id, region_version).unwrap();
                let commit = raft_state.get_commit();
                let region_progress = self.region_progresses.entry(region_id).or_default();
                if region_progress.commit_index >= commit {
                    continue;
                }
                let merged_commit_index = region_progress.commit_index;
                region_progress.commit_index = commit;
                let truncated_index = max(
                    region_progress.truncated_index,
                    origin
                        .get_truncated_index(peer_id)
                        .unwrap_or(RAFT_INIT_LOG_INDEX),
                );
                region_progress.truncated_index = truncated_index;
                origin_wb.iterate_peer_states(peer_id, |k, v| {
                    update_peer_state(&mut batch, k, v, self.ctx.config.merged_store_id, region_id);
                });
                let low_idx = merged_commit_index + 1;
                let high_idx = commit + 1;
                let mut entry_buf = Vec::with_capacity((high_idx - low_idx) as usize);
                origin
                    .fetch_raft_entries_to(peer_id, low_idx, high_idx, None, &mut entry_buf)
                    .unwrap();
                for entry in entry_buf.iter() {
                    batch.append_raft_log(region_id, region_id, entry);
                }
                updated_regions.insert(region_id);
            }
            let res = origin.persist(origin_wb);
            if res.is_err() {
                err = Some(res.unwrap_err());
            }
        })?;
        if let Some(err) = err {
            return Err(err);
        }
        self.raft.write(batch)?;
        let mut truncate_batch = rfengine::WriteBatch::new();
        for region_id in updated_regions {
            let shard_meta_bin = self.raft.get_state(region_id, KV_ENGINE_META_KEY).unwrap();
            let mut cs = kvenginepb::ChangeSet::new();
            cs.merge_from_bytes(&shard_meta_bin).unwrap();
            let shard_meta = ShardMeta::new(self.ctx.config.merged_store_id, &cs);
            let shard = self.kv.get_shard(region_id).unwrap();
            self.recover_handler
                .recover(&self.kv, &shard, &shard_meta)
                .unwrap();
            let progress = self.region_progresses.get(&region_id).unwrap();
            truncate_batch.truncate_raft_log(region_id, region_id, progress.truncated_index);
        }
        // truncate raft logs after recover kvengine in case of raft log truncated
        // before apply.
        self.raft.write(truncate_batch)?;
        Ok(())
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
