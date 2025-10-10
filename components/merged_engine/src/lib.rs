// Copyright 2025 TiKV Project Authors. Licensed under Apache-2.0.

mod error;
mod manifest;
mod preprocessor;

use std::{
    cmp,
    cmp::max,
    collections::{
        hash_map::Entry as HashMapEntry, HashMap as StdHashMap, HashSet as StdHashSet, VecDeque,
    },
    fmt, fs, io, mem, ops,
    path::{Path, PathBuf},
    sync::Arc,
    time::Duration,
};

use api_version::ApiV2;
use bytes::{Buf, BufMut, Bytes};
use cloud_encryption::MasterKey;
use collections::{HashMap, HashMapExt, HashSet};
pub use error::{Error, Result};
use file_system::{IoRateLimitMode, IoRateLimiter};
use kvengine::{
    dfs::S3Fs, ia::util::IaConfig, limiter::StoreLimiter, IdVer, MetaIterator, Shard, ShardMeta,
    ShardTag, TERM_KEY,
};
use kvenginepb::ChangeSet;
use kvproto::{metapb, metapb::Peer, raft_cmdpb::AdminRequest, raft_serverpb::StoreIdent};
use native_br::common::{
    collect_snapshot_meta_rlog_files, replay_wal_logs_from_backup, ReplayWalLogsContext,
};
use pd_client::PdClient;
use protobuf::Message;
use raft_proto::{eraftpb, eraftpb::Entry};
use rfengine::{iterator::WalIterator, RaftLogOp, RfEngine, WriteBatch, TRUNCATE_ALL_INDEX};
use rfenginepb::{ClusterBackupMeta, StoreBackupMeta};
use rfstore::{
    store::{
        get_preprocess_cmd, load_last_raft_state_from_wb, state::RaftApplyState, write_engine_meta,
        Applier, ApplyContext, ApplyMsgs, MetaChangeListener, PdIdAllocator, PeerMsg, PeerTag,
        PreprocessContext, PreprocessRef, RecoverHandler, RegionIdVer, StoreMsg,
        RAFT_INIT_LOG_INDEX,
    },
    RaftRouter,
};
use security::SecurityConfig;
use serde_derive::{Deserialize, Serialize};
use tikv::config::TikvConfig;
use tikv_util::{
    box_try,
    config::{AbsoluteOrPercentSize, ReadableDuration, ReadableSize},
    debug, error, info, mpsc, warn,
};

use crate::{
    manifest::{Manifest, UncommittedEntries},
    preprocessor::Preprocessor,
};

macro_rules! try_force_stop {
    ($self:ident, $expr:expr) => {{
        #[cfg(feature = "testexport")]
        if $self.ctx.force_stop.get() {
            info!("merged engine force stopped");
            return $expr;
        }
    }};
}

macro_rules! try_force_stop_err {
    ($self:ident) => {{
        try_force_stop!($self, Err(Error::ForceStopped));
    }};
}

// The quorum size when replicas number is 3.
// Used to check whether the Raft log is committed.
const QUORUM_SIZE: u8 = 2;

#[derive(Clone)]
pub struct MergedEngineContext {
    pub pd: Arc<dyn PdClient>,
    pub fs: Arc<S3Fs>,
    pub local_dir: PathBuf,
    pub master_key: MasterKey,
    pub config: MergedEngineConfig,
    pub security_config: Arc<SecurityConfig>,

    #[allow(dead_code)]
    pub force_stop: ForceStop, // For test purpose.
}

#[derive(Clone, Serialize, Deserialize, PartialEq, Debug)]
#[serde(default)]
#[serde(rename_all = "kebab-case")]
pub struct MergedEngineConfig {
    pub block_cache_size: AbsoluteOrPercentSize,
    pub timeout_fetch_wal: ReadableDuration,
    pub merged_store_id: u64,
    pub mem_table_size: ReadableSize,
    pub raft_write_batch_size: ReadableSize,
    pub force_ia: bool,
}

impl Default for MergedEngineConfig {
    fn default() -> Self {
        Self {
            block_cache_size: AbsoluteOrPercentSize::Percent(10.0),
            timeout_fetch_wal: ReadableDuration::secs(30),
            merged_store_id: 1024,
            mem_table_size: ReadableSize::mb(128),
            raft_write_batch_size: ReadableSize::mb(4),
            force_ia: true,
        }
    }
}

#[derive(Clone, Debug)]
pub struct RegionProgress {
    pub keyspace_id: u32,
    pub region_id: u64,
    pub entries: HashMap<u64 /* log_index */, RaftLogOpWithCounter>,
    pub synced_index: u64,
    pub commit_index: u64,
    pub truncated_index: u64,
}

impl RegionProgress {
    pub fn new(keyspace_id: u32, region_id: u64) -> Self {
        Self {
            keyspace_id,
            region_id,
            entries: HashMap::default(),
            synced_index: 0,
            commit_index: 0,
            truncated_index: 0,
        }
    }

    pub fn upsert_entry<F>(&mut self, log_index: u64, term: u32, or_insert: F)
    where
        F: FnOnce() -> RaftLogOpWithCounter,
    {
        use std::cmp::Ordering::{Equal, Greater, Less};

        if let Some(existing_op) = self.entries.get_mut(&log_index) {
            debug_assert_eq!(log_index, existing_op.index);
            match existing_op.term.cmp(&term) {
                Greater => return,
                Equal => {
                    existing_op.inc_counter();
                    if existing_op.counter() >= QUORUM_SIZE && existing_op.index > self.commit_index
                    {
                        self.commit_index = existing_op.index;
                        debug!("upsert_entry: advance commit index: {}", self.commit_index; "region" => self.region_id);
                    }
                    return;
                }
                Less => {}
            }
        }
        self.entries.insert(log_index, or_insert());
    }
}

#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub struct StoreProgress {
    pub store_id: u64,
    pub epoch: u32,
    pub offset: u64,
}

impl fmt::Display for StoreProgress {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{{ store: {}, epoch: {}, offset: {} }}",
            self.store_id, self.epoch, self.offset
        )
    }
}

impl PartialOrd for StoreProgress {
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        (self.store_id == other.store_id).then(|| {
            self.epoch
                .cmp(&other.epoch)
                .then_with(|| self.offset.cmp(&other.offset))
        })
    }
}

impl Ord for StoreProgress {
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
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
    pub ctx: MergedEngineContext,
    manifest: Manifest,
    region_progresses: HashMap<u64, RegionProgress>,
    updated_regions: HashSet<u64>,
    raft: RfEngine,
    pub kv: kvengine::Engine,
    pub recover_handler: RecoverHandler,
    preprocessors: HashMap<u64, Preprocessor>,
    appliers: HashMap<u64, Applier>,
    peer_receiver: mpsc::Receiver<(u64, Box<PeerMsg>)>,
    _store_receiver: mpsc::Receiver<StoreMsg>, // applier never send store message.
    router: RaftRouter,
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
            // Note: manifest is not persisted here to avoid saving all entries. If
            // replication worker restart before next loop, we will recover from backup
            // again.
            region_progresses
        } else {
            Self::recover_from_merged_raft_engine(&raft, &manifest.uncommitted_entries)?
        };
        let mut preprocessors = HashMap::default();
        for (region_id, _) in raft.get_region_peer_map() {
            if region_id == 0 {
                continue;
            }
            tikv_util::set_current_region(region_id);
            let Some(processor) =
                Preprocessor::new(&raft, merged_store_id, region_id, &ctx.master_key)
            else {
                error!("{} failed to create preprocessor", region_id);
                debug_assert!(false);
                continue;
            };
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
        tikv_util::init_task_local_sync(|| {
            Self::load_shards_impl(
                &ctx,
                &kv,
                recover_handler.clone(),
                &manifest.keyspace_states,
            )
        })?;

        let (store_sender, store_receiver) = mpsc::unbounded();
        let (peer_sender, peer_receiver) = mpsc::unbounded();
        let router = RaftRouter::new(peer_sender, store_sender);
        Ok(Self {
            ctx,
            manifest,
            region_progresses,
            updated_regions: HashSet::default(),
            raft,
            kv,
            recover_handler,
            preprocessors,
            appliers: HashMap::default(),
            _store_receiver: store_receiver,
            peer_receiver,
            router,
        })
    }

    fn merged_store_id(&self) -> u64 {
        self.ctx.config.merged_store_id
    }

    pub fn set_keyspace_states(&mut self, keyspace_id: u32, states: Bytes) -> Result<()> {
        let old = self
            .manifest
            .set_keyspace_states(keyspace_id, states.clone());
        if old == Some(states.clone()) {
            return Ok(()); // no change
        }
        self.manifest.persist()
    }

    pub fn remove_keyspace(&mut self, keyspace_id: u32) {
        self.manifest.keyspace_states.remove(&keyspace_id);
    }

    pub fn get_keyspaces(&self) -> Vec<u32> {
        self.manifest.keyspace_states.keys().cloned().collect()
    }

    pub fn get_keyspace_states(&self, keyspace_id: u32) -> Option<Bytes> {
        self.manifest.keyspace_states.get(&keyspace_id).cloned()
    }

    pub fn get_router(&self) -> RaftRouter {
        self.router.clone()
    }

    fn load_shards_impl(
        ctx: &MergedEngineContext,
        kv: &kvengine::Engine,
        mut recoverer: RecoverHandler,
        keyspace_states: &HashMap<u32, Bytes>,
    ) -> Result<()> {
        let metas = Self::load_shard_metas_impl(ctx, &mut recoverer, keyspace_states)?;
        kv.load_shards(metas, recoverer, None)?;
        Ok(())
    }

    fn load_shard_metas_impl(
        ctx: &MergedEngineContext,
        recoverer: &mut RecoverHandler,
        keyspace_states: &HashMap<u32, Bytes>,
    ) -> Result<StdHashMap<u64 /* region_id */, ShardMeta>> {
        let engine_id = ctx.config.merged_store_id;
        let mut metas = StdHashMap::default();
        recoverer.iterate(|cs| {
            debug_assert!(cs.has_snapshot());
            let keyspace_id = get_keyspace_id_of_snapshot(cs.get_snapshot());
            if keyspace_states.contains_key(&keyspace_id) {
                let meta = ShardMeta::new(engine_id, &cs);
                info!("{} load shard meta: {:?}", meta.tag(), cs);
                metas.insert(meta.id, meta);
            }
        })?;
        Ok(metas)
    }

    pub fn load_shards(&self, keyspace_states: &HashMap<u32, Bytes>) -> Result<()> {
        Self::load_shards_impl(
            &self.ctx,
            &self.kv,
            self.recover_handler.clone(),
            keyspace_states,
        )
    }

    pub fn load_keyspace_shard_metas(
        &self,
        keyspace_id: u32,
    ) -> Result<StdHashMap<u64 /* region_id */, ShardMeta>> {
        let mut states = HashMap::with_capacity(1);
        states.insert(keyspace_id, Bytes::new());
        let mut recoverer = self.recover_handler.clone();
        Self::load_shard_metas_impl(&self.ctx, &mut recoverer, &states)
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
        let mut region_progresses = HashMap::default();
        let mut store_progresses = HashMap::default();
        let mut raftdb_paths = Vec::new();
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
            raftdb_paths.push(store_config.raft_store.raftdb_path);
            let region_peers_map = origin.get_region_peer_map();
            for (region_id, peer_id) in region_peers_map {
                if region_id == 0 {
                    continue;
                }
                let Some(region_state) = rfstore::store::load_last_peer_state(&origin, peer_id)
                else {
                    let mut states = HashMap::default();
                    origin.iterate_peer_states(peer_id, false, |k, v| {
                        states.insert(Bytes::copy_from_slice(k), Bytes::copy_from_slice(v));
                        true
                    });
                    warn!("{}:{} recover_from_backup: no peer state for region", store.store_id, region_id;
                        "peer" => peer_id, "states" => ?states);
                    debug_assert!(false);
                    continue;
                };
                let keyspace_id =
                    ApiV2::get_u32_keyspace_id_by_key(region_state.get_region().get_start_key())
                        .unwrap_or_default();
                let region_version = region_state.get_region().get_region_epoch().get_version();
                let tag = ShardTag::new(store.store_id, IdVer::new(region_id, region_version));
                let raft_state =
                    rfstore::store::load_peer_raft_state(&origin, peer_id, region_version).unwrap();
                let preprocess_index = raft_state.get_last_preprocessed_index();
                let region_progress = region_progresses
                    .entry(region_id)
                    .or_insert(RegionProgress::new(keyspace_id, region_id));
                let merged_commit_index = region_progress.commit_index;

                if raft_state.get_last_index() > preprocess_index {
                    // Fetch uncommitted entries, and insert them into region progress, so that they
                    // will be replayed when commit index advances (during sync_merged).
                    let mut entry_buf = Vec::new();
                    let low_idx = preprocess_index + 1;
                    let high_idx = raft_state.get_last_index() + 1;
                    debug!(
                        "{} recover_from_backup: fetch uncommitted entries: [{}, {})",
                        tag, low_idx, high_idx
                    );
                    if let Err(err) = origin.fetch_raft_entries_to(
                        peer_id,
                        low_idx,
                        high_idx,
                        None,
                        &mut entry_buf,
                    ) {
                        panic!(
                            "{} fetch raft entries failed for region, low: {}, high: {}, err: {}",
                            tag, low_idx, high_idx, err
                        );
                    }
                    for entry in entry_buf.iter() {
                        debug!(
                            "{} recover from backup: insert log index {}",
                            tag, entry.index
                        );
                        region_progress.upsert_entry(entry.index, entry.term as u32, || {
                            RaftLogOp::new(entry).into()
                        });
                    }
                }

                // Committed entries will be replayed right away during the recovery process
                // below.
                let commit = raft_state.get_commit().max(region_progress.commit_index);
                if merged_commit_index >= commit {
                    continue;
                }
                region_progress.commit_index = commit;
                region_progress.synced_index = preprocess_index;
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
                    update_peer_state(&mut batch, k, v, merged_raft.get_engine_id(), region_id);
                    true
                });

                if batch
                    .get_state(region_id, rfengine::KV_ENGINE_META_KEY)
                    .is_some()
                {
                    for k in [
                        rfengine::KV_ENGINE_META_DIFF_KEY,
                        rfengine::KV_ENGINE_META_SNAP_DIFF_KEY,
                    ] {
                        if batch.get_state(region_id, k).is_none() {
                            batch.set_state(region_id, region_id, k, &[]);
                        }
                    }
                }

                // merge raft logs
                let mut entry_buf = Vec::new();
                let low_idx = merged_commit_index.max(truncated_index) + 1;
                let high_idx = commit + 1;
                debug!(
                    "{} recover_from_backup: fetch committed entries: [{}, {})",
                    tag, low_idx, high_idx
                );
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
        for raftdb_path in raftdb_paths {
            let raft_path = Path::new(&raftdb_path);
            // clean up dir
            std::fs::remove_dir_all(raft_path).unwrap();
        }
        Ok((region_progresses, store_progresses))
    }

    fn recover_from_merged_raft_engine(
        merged_raft: &RfEngine,
        uncommitted_entries: &UncommittedEntries,
    ) -> Result<HashMap<u64, RegionProgress>> {
        let mut region_progresses = HashMap::default();

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
            let region_progress = region_progresses
                .entry(region_id)
                .or_insert(RegionProgress::new(keyspace_id, region_id));
            region_progress.commit_index = commit;
            region_progress.synced_index = commit;
            region_progress.truncated_index = merged_raft
                .get_truncated_index(region_id)
                .unwrap_or(RAFT_INIT_LOG_INDEX);
            if let Some(entries) = uncommitted_entries.get_region_entries(region_id) {
                region_progress.entries = entries.clone();
            }
        }

        Ok(region_progresses)
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
            store_id,
            None,
            Path::new(&store_config.raft_store.raftdb_path),
            rlog_files.snap_epoch,
            rlog_files.snap_meta,
            rlog_files.snap_rlog,
        )?;
        let raft_db_path = Path::new(&store_config.raft_store.raftdb_path);
        let data_dir = Path::new(&store_config.storage.data_dir);
        let rf_engine = RfEngine::open(raft_db_path, &store_config.rfengine, Some(data_dir), None)?;
        let cache_dir = store_path.join("cache");
        box_try!(fs::create_dir_all(&cache_dir));
        let ctx = ReplayWalLogsContext {
            pd_client: ctx.pd.clone(),
            dfs: ctx.fs.clone(),
            store_id,
            cluster_backup: backup_meta,
            rf_engine: &rf_engine,
            complete_wal_chunks: false,
            full_restore: false,
            fetch_wal_timeout: ctx.config.timeout_fetch_wal.0,
            cache_dir: Some(cache_dir),
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
        if !kv_engine_path.exists() {
            fs::create_dir_all(&kv_engine_path)?;
        }
        let mut kv_opts = kvengine::Options::default();
        kv_opts.local_dirs = vec![kv_engine_path];
        kv_opts.max_mem_table_size = ctx.config.mem_table_size.0;
        kv_opts.max_block_cache_size = ctx.config.block_cache_size.as_memory_size() as i64;
        kv_opts.for_restore = true;
        if ctx.config.force_ia {
            kv_opts.ia = IaConfig {
                mem_cap: AbsoluteOrPercentSize::Percent(10.0),
                disk_cap: AbsoluteOrPercentSize::Percent(60.0),
                dynamic_capacity: false,
                force_ia: true,
                ..Default::default()
            };
        }
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

    pub fn get_or_insert_store_progress(&mut self, store_id: u64) -> StoreProgress {
        *self
            .manifest
            .store_progresses
            .entry(store_id)
            .or_insert_with(|| StoreProgress {
                store_id,
                epoch: 1,
                offset: 0,
            })
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

    pub fn update_wal<R: io::Read>(
        &mut self,
        store_id: u64,
        epoch_id: u32,
        start_off: u64,
        end_off: u64,
        reader: R,
    ) -> Result<()> {
        if let Some(store_progress) = self.manifest.store_progresses.get(&store_id) {
            if store_progress.epoch != epoch_id || store_progress.offset != start_off {
                let err_msg = format!(
                    "store {} expect ({}, {}), got ({}, {}), end_off: {}",
                    store_id,
                    epoch_id,
                    start_off,
                    store_progress.epoch,
                    store_progress.offset,
                    end_off,
                );
                error!("{}", &err_msg);
                debug_assert!(false);
                return Err(Error::StoreProgressMismatch(err_msg));
            }
        } else {
            return Err(Error::StoreProgressNotFound(store_id));
        };
        let mut wal_iterator = WalIterator::new_from_reader(reader, epoch_id, start_off);
        let mut origin_batches = Vec::new();
        wal_iterator.iterate_write_batch(|origin_wb| {
            origin_batches.push(origin_wb);
        })?;
        for origin_wb in origin_batches {
            let region_peer_map = origin_wb.get_region_peer_map();
            for (&region_id, &peer_id) in &region_peer_map {
                if region_id == 0 {
                    continue;
                }
                tikv_util::set_current_region(region_id);
                let tag = ShardTag::new(store_id, IdVer::new(region_id, 0));

                let progress = match self.region_progresses.entry(region_id) {
                    HashMapEntry::Occupied(e) => e.into_mut(),
                    HashMapEntry::Vacant(e) => {
                        let Some(region_local_state) = origin_wb.get_latest_peer_state(peer_id)
                        else {
                            info!("{} update_wal: no peer state in wb", tag);
                            continue;
                        };
                        let keyspace_id = ApiV2::get_u32_keyspace_id_by_key(
                            region_local_state.get_region().get_start_key(),
                        )
                        .unwrap_or_default();
                        e.insert(RegionProgress::new(keyspace_id, region_id))
                    }
                };
                if progress.truncated_index == TRUNCATE_ALL_INDEX {
                    debug!("{} update_wal: truncate all", tag);
                    continue;
                }
                if let Some(truncated_idx) = origin_wb.get_truncated_idx(peer_id) {
                    if progress.truncated_index < truncated_idx
                        && truncated_idx != TRUNCATE_ALL_INDEX
                    {
                        progress.truncated_index = truncated_idx;
                    }
                }
                if let Some(raft_state) = load_last_raft_state_from_wb(&origin_wb, peer_id) {
                    if raft_state.get_commit() > progress.commit_index {
                        progress.commit_index = raft_state.get_commit();
                        debug!(
                            "{} update_wal: advance commit index {}",
                            tag, progress.commit_index
                        );
                    }
                }
                origin_wb.read_peer_logs(peer_id, |logs| {
                    for log_op in logs {
                        debug!("{} update_wal: insert log index {}", tag, log_op.index);
                        progress.upsert_entry(log_op.index, log_op.term, || log_op.clone().into());
                    }
                });
                self.updated_regions.insert(region_id);
            }
        }
        self.manifest
            .update_store_progress(store_id, epoch_id, end_off);
        Ok(())
    }

    pub fn rotate_wal(&mut self, store_id: u64, epoch_id: u32, offset: u64) -> Result<()> {
        if let Some(store_progress) = self.manifest.store_progresses.get_mut(&store_id) {
            if store_progress.epoch != epoch_id || store_progress.offset != offset {
                return Err(Error::StoreProgressMismatch(format!(
                    "store {} expect epoch {}, offset {}, got epoch {}, offset {}",
                    store_id, epoch_id, offset, store_progress.epoch, store_progress.offset
                )));
            }
            info!(
                "rotate store {} at epoch {}, offset {}",
                store_id, epoch_id, offset
            );
            store_progress.epoch += 1;
            store_progress.offset = 0;
        } else {
            return Err(Error::StoreProgressNotFound(store_id));
        }
        Ok(())
    }

    pub fn sync_merged(&mut self, apply_ctx: &mut ApplyContext) -> Result<()> {
        // Prepare context.
        let mut raft_wb = rfengine::WriteBatch::new();
        let mut remove_dependents = Vec::new();
        let mut apply_msgs = ApplyMsgs::default();
        let raft_cfg = rfstore::store::Config::default();
        let mut destroying = StdHashSet::default();
        let raft_engine = self.raft.clone();
        let router = self.router.clone();
        let pre_ctx = PreprocessContext {
            store_id: self.merged_store_id(),
            kv: None,
            raft: &raft_engine,
            raft_wb: &mut raft_wb,
            remove_dependents: &mut remove_dependents,
            apply_msgs: &mut apply_msgs,
            cfg: &raft_cfg,
            router: Some(&router),
            destroying: &mut destroying,
        };
        let mut ctx = SyncRegionsContext {
            pre_ctx,
            apply_ctx,
            prepared_msgs: HashMap::default(),
            destroyed_regions: HashSet::default(),
        };
        let updated_regions: Vec<u64> = self.updated_regions.drain().collect();
        self.sync_merged_with_ctx(&mut ctx, updated_regions)
            .map_err(|e| {
                debug!("sync_merged: clear context on error: {:?}", ctx; "err" => ?e);
                ctx.clear();
                e
            })
    }

    fn sync_merged_with_ctx(
        &mut self,
        ctx: &mut SyncRegionsContext<'_>,
        updated_regions: Vec<u64>,
    ) -> Result<()> {
        self.sync_merged_for_regions(ctx, &updated_regions)?;
        self.handle_prepared_msgs(ctx);
        self.update_progress_and_truncate(&updated_regions, ctx.raft_wb);
        self.destroy_regions(ctx);
        if !ctx.raft_wb.is_empty() {
            try_force_stop_err!(self);
            self.raft.write(mem::take(ctx.raft_wb))?;
        }
        self.manifest
            .update_region_progresses(&self.region_progresses);
        try_force_stop_err!(self);
        self.manifest.persist()?;
        Ok(())
    }

    fn sync_merged_for_regions(
        &mut self,
        ctx: &mut SyncRegionsContext<'_>,
        updated_regions: &[u64],
    ) -> Result<()> {
        let merged_store_id = self.merged_store_id();
        info!("sync merged for regions {:?}", updated_regions; "store" => merged_store_id);
        let mut update_queue = VecDeque::from(updated_regions.to_vec());
        let mut merged_wb = rfengine::WriteBatch::new();
        let mut merged_wb_estimated_size = 0;
        let mut finished_regions = HashSet::default();
        while let Some(updated_region) = update_queue.pop_front() {
            try_force_stop_err!(self);
            tikv_util::set_current_region(updated_region);
            match self.sync_region(
                ctx,
                updated_region,
                &finished_regions,
                &mut merged_wb,
                &mut merged_wb_estimated_size,
            )? {
                SyncRegionResult::Finished => {
                    finished_regions.insert(updated_region);
                }
                SyncRegionResult::Postponed => {
                    update_queue.push_back(updated_region);
                }
                SyncRegionResult::Resume => {
                    update_queue.push_front(updated_region);
                }
                SyncRegionResult::Dropped => continue,
            }
        }
        if !merged_wb.is_empty() {
            self.raft.persist(merged_wb)?;
        }
        Ok(())
    }

    fn sync_region(
        &mut self,
        ctx: &mut SyncRegionsContext<'_>,
        updated_region: u64,
        finished_regions: &HashSet<u64>,
        merged_wb: &mut rfengine::WriteBatch,
        merged_wb_estimated_size: &mut usize,
    ) -> Result<SyncRegionResult> {
        let merged_store_id = self.merged_store_id();
        let mut tag = PeerTag::new(merged_store_id, RegionIdVer::new(updated_region, 0));

        if self.raft.get_truncated_index(updated_region).is_none() {
            // region is newly inserted, should process parent first.
            info!("{} sync_merged: region is newly inserted", tag);
            return Ok(SyncRegionResult::Postponed);
        }

        let progress = self.region_progresses.get_mut(&updated_region).unwrap();
        let low = progress.synced_index.max(RAFT_INIT_LOG_INDEX) + 1;
        let high: u64 = progress.commit_index + 1;
        if low >= high {
            return Ok(SyncRegionResult::Finished);
        }
        let preprocessor = match self.preprocessors.entry(updated_region) {
            HashMapEntry::Occupied(e) => e.into_mut(),
            HashMapEntry::Vacant(e) => {
                let Some(preprocessor) = Preprocessor::new(
                    &self.raft,
                    ctx.store_id,
                    updated_region,
                    &self.ctx.master_key,
                ) else {
                    info!("{} sync_merged: region is merged or destroyed", tag);
                    return Ok(SyncRegionResult::Dropped);
                };
                e.insert(preprocessor)
            }
        };
        let mut preprocessor_ref = preprocessor.as_ref();
        tag = preprocessor_ref.tag();
        let mut entries = Vec::new();
        let mut wb_encoded_len = 0;
        let mut res = SyncRegionResult::Finished;
        // preprocess entries.
        for log_index in low..high {
            try_force_stop_err!(self);
            let mut entry = progress
                .entries
                .get(&log_index)
                .unwrap_or_else(|| {
                    panic!(
                        "{} entry not found for region {}, log index {}",
                        tag, updated_region, log_index
                    )
                })
                .to_entry();
            let mut admin_req = update_entry(&mut entry, merged_store_id);
            if let Some(admin) = admin_req.as_ref() {
                if admin.has_commit_merge() {
                    let commit_merge = admin.get_commit_merge();
                    if !finished_regions.contains(&commit_merge.get_source().get_id()) {
                        // need to process source region first.
                        progress.synced_index = log_index - 1;
                        res = SyncRegionResult::Postponed;
                        info!(
                            "{} commit merge postponed at {}, low {}, high {}",
                            tag, log_index, low, high
                        );
                        break;
                    }
                }
            }
            let err = preprocessor_ref.preprocess_committed_entry(ctx, &entry);
            if let Some(err) = err {
                warn!("{} preprocess committed entry failed: {:?}", tag, err);
                // clear failed command.
                admin_req = None;
                entry.set_data(Bytes::new());
            }
            preprocessor_ref
                .raft_state
                .set_last_preprocessed_index(*preprocessor_ref.preprocessed_index);
            let mut hs = eraftpb::HardState::default();
            hs.set_term(1);
            hs.set_vote(updated_region);
            hs.set_commit(progress.commit_index);
            preprocessor_ref.raft_state.set_hard_state(&hs);
            preprocessor_ref.raft_state.set_last_index(log_index);
            wb_encoded_len += ctx
                .raft_wb
                .append_raft_log(updated_region, updated_region, &entry);
            if let Some(admin_req) = admin_req {
                if admin_req.has_commit_merge() {
                    let last_change_set = ctx.apply_msgs.get_last_change_set();
                    let source = last_change_set.unwrap();
                    ctx.destroyed_regions.insert(source.shard_id);
                    ctx.raft
                        .iterate_peer_states(source.shard_id, false, |k, _| {
                            ctx.raft_wb
                                .set_state(source.shard_id, source.shard_id, k, &[]);
                            true
                        });
                    wb_encoded_len += ctx.raft_wb.truncate_raft_log(
                        source.shard_id,
                        source.shard_id,
                        TRUNCATE_ALL_INDEX,
                    );
                }
            }
            entries.push(entry);

            if log_index + 1 < high
                && wb_encoded_len >= self.ctx.config.raft_write_batch_size.0 as i64
            {
                debug!("{} sync_region: wb exceed size limit, break at {}", tag, log_index;
                    "wb_size" => wb_encoded_len, "low" => low, "high" => high);
                progress.synced_index = log_index;
                res = SyncRegionResult::Resume;
                break;
            }
        }
        if ctx.raft_wb.is_empty() {
            return Ok(res);
        }
        preprocessor_ref.write_raft_state(ctx);
        let mut wb = mem::take(ctx.raft_wb);
        ctx.raft.apply(&mut wb);
        *merged_wb_estimated_size += wb.estimated_size();
        merged_wb.merge_write_batch(wb);
        if *merged_wb_estimated_size > self.ctx.config.raft_write_batch_size.0 as usize {
            ctx.raft.persist(mem::take(merged_wb))?;
            *merged_wb_estimated_size = 0;
        }
        let shard = self.kv.get_shard(updated_region);
        if shard.is_none() {
            // shard is not in the keyspace range, skip apply.
            debug!("{} sync_merged_for_regions: skip apply", tag);
            preprocessor.sync_region();
            ctx.skip_apply();
            return Ok(res);
        }
        let shard = shard.unwrap();
        // apply committed entries.
        let applier = self
            .appliers
            .entry(updated_region)
            .or_insert_with(|| Self::new_applier(&shard, preprocessor_ref, low - 1));
        ctx.build_apply_msg_for_replication(entries);
        ctx.pre_ctx
            .handle_apply_msgs_for_replication(applier, ctx.apply_ctx);
        // We keep waiting for paused region because later region may depend on it.
        while applier.is_paused() {
            try_force_stop_err!(self);
            let msgs = if let Some(msgs) = ctx.prepared_msgs.remove(&updated_region) {
                // received by previous region, handle it now.
                msgs
            } else {
                let (id, peer_msg) = match self.peer_receiver.recv_timeout(Duration::from_secs(3)) {
                    Ok((id, msg)) => (id, msg),
                    Err(err) => {
                        if err.is_timeout() {
                            warn!("{} waiting for region to unpause", tag);
                            continue;
                        }
                        return Err(Error::Other(Box::new(err)));
                    }
                };
                if id != updated_region {
                    // For another region, handle it later.
                    ctx.prepared_msgs
                        .entry(id)
                        .or_default()
                        .push((id, peer_msg));
                    continue;
                }
                vec![(id, peer_msg)]
            };
            Self::apply_prepared_msgs(ctx, applier, msgs);
        }
        preprocessor.sync_region();

        Ok(res)
    }

    fn handle_prepared_msgs(&mut self, ctx: &mut SyncRegionsContext<'_>) {
        let msg_count = self.peer_receiver.len();
        for _ in 0..msg_count {
            let (id, peer_msg) = self.peer_receiver.recv().unwrap();
            ctx.prepared_msgs
                .entry(id)
                .or_default()
                .push((id, peer_msg));
        }
        let prepared_msgs = mem::take(&mut ctx.prepared_msgs);
        for (region_id, msgs) in prepared_msgs {
            tikv_util::set_current_region(region_id);
            let Some(applier) = self.appliers.get_mut(&region_id) else {
                continue;
            };
            Self::apply_prepared_msgs(ctx, applier, msgs);
        }
    }

    fn update_progress_and_truncate(&mut self, regions: &[u64], raft_wb: &mut WriteBatch) {
        for &region_id in regions {
            tikv_util::set_current_region(region_id);
            let tag = ShardTag::new(self.merged_store_id(), IdVer::new(region_id, 0));
            let progress = self.region_progresses.get_mut(&region_id).unwrap();
            let commit_index = progress.commit_index;
            progress.synced_index = commit_index;
            // We need to keep the uncommitted index for the next round.
            debug!(
                "{} update_progress_and_truncate: truncate <= {}",
                tag, commit_index
            );
            progress.entries.retain(|&index, _| index > commit_index);

            let truncated_idx = self.raft.get_truncated_index(region_id).unwrap();
            if progress.truncated_index > truncated_idx {
                if let Some(preprocessor) = self.preprocessors.get_mut(&region_id) {
                    if let Some(shard_meta) = preprocessor.as_ref().shard_meta {
                        if shard_meta.parent.is_some() {
                            // skip truncating region with parent, the parent may need the old
                            // raft logs on recover.
                            continue;
                        }
                        if shard_meta.data_sequence < progress.truncated_index {
                            shard_meta.data_sequence = progress.truncated_index;
                            write_engine_meta(raft_wb, region_id, shard_meta);
                        }
                    }
                }
                raft_wb.truncate_raft_log(region_id, region_id, progress.truncated_index);
            }
        }
    }

    fn destroy_regions(&mut self, ctx: &mut SyncRegionsContext<'_>) {
        for region_id in ctx.destroyed_regions.drain() {
            tikv_util::set_current_region(region_id);
            self.raft.iterate_peer_states(region_id, false, |k, _| {
                ctx.pre_ctx.raft_wb.set_state(region_id, region_id, k, &[]);
                true
            });
            ctx.pre_ctx
                .raft_wb
                .truncate_raft_log(region_id, region_id, TRUNCATE_ALL_INDEX);
            self.appliers.remove(&region_id);
            self.kv.remove_shard(region_id);
            self.preprocessors.remove(&region_id);
            let progress = self.region_progresses.get_mut(&region_id).unwrap();
            progress.truncated_index = TRUNCATE_ALL_INDEX;
        }
    }

    fn new_applier(
        shard: &Shard,
        preprocess_ref: PreprocessRef<'_>,
        applied_index: u64,
    ) -> Applier {
        let encryption_key = shard.get_encryption_key();
        let term_val = shard.get_property(TERM_KEY).unwrap();
        let term = term_val.chunk().get_u64_le();
        Applier::new_for_replication(
            preprocess_ref.region.clone(),
            encryption_key,
            RaftApplyState::new(applied_index, term),
        )
    }

    fn apply_prepared_msgs(
        ctx: &mut SyncRegionsContext<'_>,
        applier: &mut Applier,
        msgs: Vec<(u64, Box<PeerMsg>)>,
    ) {
        for (_, msg) in msgs {
            ctx.build_prepared_msg_for_replication(msg);
        }
        ctx.pre_ctx
            .handle_apply_msgs_for_replication(applier, ctx.apply_ctx);
    }
}

enum SyncRegionResult {
    Finished,
    Postponed,
    Resume,
    Dropped,
}

struct SyncRegionsContext<'a> {
    pre_ctx: PreprocessContext<'a>,
    apply_ctx: &'a mut ApplyContext,
    prepared_msgs: HashMap<u64 /* region_id */, Vec<(u64 /* region_id */, Box<PeerMsg>)>>,
    destroyed_regions: HashSet<u64>,
}

impl<'a> ops::Deref for SyncRegionsContext<'a> {
    type Target = PreprocessContext<'a>;

    fn deref(&self) -> &Self::Target {
        &self.pre_ctx
    }
}

impl ops::DerefMut for SyncRegionsContext<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.pre_ctx
    }
}

impl Drop for SyncRegionsContext<'_> {
    fn drop(&mut self) {
        debug_assert!(self.apply_msgs.is_empty());
        debug_assert!(self.prepared_msgs.is_empty());
        debug_assert!(self.destroyed_regions.is_empty());
        debug_assert!(self.raft_wb.is_empty());
    }
}

impl SyncRegionsContext<'_> {
    fn skip_apply(&mut self) {
        self.apply_msgs.clear();
    }

    fn clear(&mut self) {
        self.apply_msgs.clear();
        self.prepared_msgs.clear();
        self.destroyed_regions.clear();
        self.raft_wb.reset();
    }
}

impl fmt::Debug for SyncRegionsContext<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SyncRegionsContext")
            .field("apply_msgs", &self.apply_msgs.len())
            .field("prepared_msgs", &self.prepared_msgs.len())
            .field("destroyed_regions", &self.destroyed_regions.len())
            .field("raft_wb", &self.raft_wb.len())
            .finish()
    }
}

fn update_peer_state(
    wb: &mut rfengine::WriteBatch,
    k: &[u8],
    v: &[u8],
    store_id: u64,
    region_id: u64,
) {
    // TODO: Compare and overwrite state only when it's newer.
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

#[derive(Clone, Debug, PartialEq)]
pub struct RaftLogOpWithCounter {
    pub op: RaftLogOp,
    counter: u8,
}

impl ops::Deref for RaftLogOpWithCounter {
    type Target = RaftLogOp;

    fn deref(&self) -> &Self::Target {
        &self.op
    }
}

impl ops::DerefMut for RaftLogOpWithCounter {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.op
    }
}

impl From<RaftLogOp> for RaftLogOpWithCounter {
    fn from(op: RaftLogOp) -> Self {
        RaftLogOpWithCounter { op, counter: 1 }
    }
}

impl RaftLogOpWithCounter {
    pub fn inc_counter(&mut self) {
        self.counter = self.counter.saturating_add(1);
    }

    pub fn counter(&self) -> u8 {
        self.counter
    }
}

fn get_keyspace_id_of_snapshot(snap: &kvenginepb::Snapshot) -> u32 {
    ApiV2::get_u32_keyspace_id_by_key(snap.get_outer_start()).unwrap_or_default()
}

#[cfg(feature = "testexport")]
#[derive(Default, Clone)]
pub struct ForceStop(Arc<std::sync::atomic::AtomicBool>);

#[cfg(feature = "testexport")]
impl ForceStop {
    pub fn set(&self) {
        use std::sync::atomic::Ordering;
        self.0.store(true, Ordering::Relaxed);
    }

    pub fn get(&self) -> bool {
        use std::sync::atomic::Ordering;
        self.0.load(Ordering::Relaxed)
    }
}

#[cfg(not(feature = "testexport"))]
pub type ForceStop = ();
