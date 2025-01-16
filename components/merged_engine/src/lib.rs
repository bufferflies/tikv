// Copyright 2025 TiKV Project Authors. Licensed under Apache-2.0.

mod error;
mod preprocessor;

use std::{
    cmp::max,
    collections::{HashMap, HashSet},
    mem,
    path::{Path, PathBuf},
    sync::Arc,
};

use bytes::Bytes;
use cloud_encryption::MasterKey;
pub use error::{Error, Result};
use file_system::{IoRateLimitMode, IoRateLimiter};
use kvengine::{dfs::S3Fs, limiter::StoreLimiter, RecoverHandler as RecoverHandlerTrait};
use kvproto::{
    metapb,
    metapb::Peer,
    raft_cmdpb::AdminRequest,
    raft_serverpb::{PeerState, StoreIdent},
};
use native_br::common::{
    collect_snapshot_meta_rlog_files, replay_wal_logs_from_backup, ReplayWalLogsContext,
};
use pd_client::PdClient;
use protobuf::Message;
use raft_proto::eraftpb::Entry;
use rfengine::{iterator::WalIterator, RfEngine, TRUNCATE_ALL_INDEX};
use rfenginepb::{ClusterBackupMeta, StoreBackupMeta};
use rfstore::store::{
    get_preprocess_cmd, ApplyMsgs, MetaChangeListener, PdIdAllocator, PreprocessContext,
    RecoverHandler, RAFT_INIT_LOG_INDEX,
};
use security::SecurityConfig;
use tikv::config::TikvConfig;
use tikv_util::{
    config::{ReadableDuration, ReadableSize},
    mpsc, warn,
};

use crate::preprocessor::Preprocessor;

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
    pub mem_table_size: ReadableSize,
}

#[derive(Clone, Copy, Default, Debug)]
pub struct RegionProgress {
    pub leader_store: u64,
    pub leader_peer: u64,
    pub commit_index: u64,
    pub truncated_index: u64,
}

pub struct MergedEngine {
    ctx: MergedEngineContext,
    origins: HashMap<u64, RfEngine>,
    region_progresses: HashMap<u64, RegionProgress>,
    updated_regions: HashSet<u64>,
    raft: RfEngine,
    kv: kvengine::Engine,
    recover_handler: RecoverHandler,
    preprocessors: HashMap<u64, Preprocessor>,
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
        let mut preprocessors = HashMap::new();
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
                )
                .max(RAFT_INIT_LOG_INDEX);
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
                raft.write(batch).unwrap();
            }
            origins.insert(store.store_id, origin);
        }
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
            updated_regions: HashSet::new(),
            raft,
            kv,
            recover_handler,
            preprocessors,
        }
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

    pub fn update_wal(
        &mut self,
        store_id: u64,
        epoch_id: u32,
        offset: u64,
        data: Bytes,
    ) -> Result<()> {
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
            ))
            .into());
        }
        let mut wal_iterator = WalIterator::new_from_chunks(data, epoch_id, offset);
        let mut origin_batches = Vec::new();
        wal_iterator.iterate_write_batch(|origin_wb| {
            origin_batches.push(origin_wb);
        })?;
        for mut origin_wb in origin_batches {
            let region_peer_map = origin_wb.get_region_peer_map();
            for (&region_id, &peer_id) in &region_peer_map {
                let progress = self.region_progresses.entry(region_id).or_default();
                if let Some(truncated_idx) = origin_wb.reset_truncated_idx(peer_id) {
                    if progress.truncated_index < truncated_idx {
                        progress.truncated_index = truncated_idx;
                    }
                }
            }
            origin.write(origin_wb)?;
            for (region_id, peer_id) in region_peer_map {
                let progress = self.region_progresses.get_mut(&region_id).unwrap();
                if let Some(raft_state) = rfstore::store::load_last_raft_state(origin, peer_id) {
                    if progress.commit_index < raft_state.get_commit() {
                        progress.commit_index = raft_state.get_commit();
                        progress.leader_store = store_id;
                        progress.leader_peer = peer_id;
                    }
                    self.updated_regions.insert(region_id);
                } else {
                    let peer_state = rfstore::store::load_last_peer_state(origin, peer_id).unwrap();
                    assert_eq!(peer_state.get_state(), PeerState::Tombstone);
                    progress.truncated_index = TRUNCATE_ALL_INDEX;
                }
            }
        }
        Ok(())
    }

    pub fn sync_merged(&mut self) -> Result<()> {
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
        let updated_regions: Vec<u64> = self.updated_regions.iter().copied().collect();
        self.sync_merged_for_regions(&mut ctx, &updated_regions)?;
        let mut raft_wb = rfengine::WriteBatch::new();
        for updated_region in updated_regions {
            let progress = self.region_progresses.get(&updated_region).unwrap();
            if let Some(truncated) = self.raft.get_truncated_index(updated_region) {
                if progress.truncated_index > truncated {
                    raft_wb.truncate_raft_log(
                        updated_region,
                        updated_region,
                        progress.truncated_index,
                    );
                }
            }
        }
        self.raft.write(raft_wb)?;
        Ok(())
    }

    fn sync_merged_for_regions(
        &mut self,
        ctx: &mut PreprocessContext<'_>,
        updated_regions: &[u64],
    ) -> Result<()> {
        let mut new_regions = vec![];
        let merged_store_id = self.ctx.config.merged_store_id;
        for &updated_region in updated_regions {
            if self.raft.get_truncated_index(updated_region).is_none() {
                // newly split region is handled after parent regions.
                new_regions.push(updated_region);
                continue;
            }
            let last_index = self
                .raft
                .get_last_index(updated_region)
                .unwrap_or(RAFT_INIT_LOG_INDEX)
                .max(RAFT_INIT_LOG_INDEX);
            let progress = self.region_progresses.get_mut(&updated_region).unwrap();
            let origin = self.origins.get(&progress.leader_store).unwrap();
            let low = last_index + 1;
            let high = progress.commit_index + 1;
            let mut entries = vec![];
            if let Err(err) =
                origin.fetch_raft_entries_to(progress.leader_peer, low, high, None, &mut entries)
            {
                panic!(
                    "fetch raft entries failed for region {}, low: {}, high: {}, err: {}",
                    updated_region, low, high, err
                );
            }
            let preprocessor = self.preprocessors.entry(updated_region).or_insert_with(|| {
                Preprocessor::new(
                    &self.raft,
                    ctx.store_id,
                    updated_region,
                    &self.ctx.master_key,
                )
            });
            let mut preprocessor_ref = preprocessor.as_ref();
            for entry in &mut entries {
                let admin_req = update_entry(entry, merged_store_id);
                let err = preprocessor_ref.preprocess_committed_entry(ctx, entry);
                if let Some(err) = err {
                    warn!("preprocess committed entry failed"; "region_id" => updated_region, "err" => ?err);
                }
                preprocessor_ref
                    .raft_state
                    .set_last_preprocessed_index(*preprocessor_ref.preprocessed_index);
                ctx.raft_wb
                    .append_raft_log(updated_region, updated_region, entry);
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
                    self.recover_handler
                        .recover(&self.kv, &shard, &shard_meta)?;
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
                        let source_cs = self.kv.prepare_change_set(
                            source,
                            false,
                            None,
                            shard.get_encryption_key(),
                        )?;
                        self.kv
                            .commit_merge(shard.id, shard.ver, &source_cs, entry.index)?
                    }
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
                let shard = self.kv.get_shard(updated_region).unwrap();
                let preprocessor = self.preprocessors.get_mut(&updated_region).unwrap();
                let preprocessor_ref = preprocessor.as_ref();
                let shard_meta = preprocessor_ref.shard_meta.as_ref().unwrap();
                self.recover_handler
                    .recover(&self.kv, &shard, shard_meta)
                    .unwrap();
            }
        }
        if !new_regions.is_empty() {
            self.sync_merged_for_regions(ctx, &new_regions)
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
