// Copyright 2023 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    borrow::Cow,
    cmp,
    collections::{HashMap, HashSet},
    default::Default,
    fmt, mem,
    path::{Path, PathBuf},
    str::FromStr,
    sync::{Arc, RwLock},
    thread,
    time::Duration,
};

use api_version::ApiV2;
use cloud_server::TikvServer;
use file_system::{IoRateLimitMode, IoRateLimiter};
use http::{Request, Uri};
use hyper::Body;
use itertools::Itertools;
use kvengine::{dfs::S3Fs, IdVer, ShardMeta, ShardStats, ShardTag};
use kvenginepb as pb;
use pd_client::PdClient;
use protobuf::Message;
use rfengine::RfEngine;
use rfenginepb::ClusterBackupMeta;
use rfstore::store::StoreMsg;
use slog_global::{debug, error, info, warn};
use tempdir::TempDir;
use tikv::{config::TikvConfig, storage::mvcc::Key};
use tikv_util::{box_err, mpsc, time::Instant, HandyRwLock};
use tokio::runtime::Runtime;

use crate::{
    common::{
        create_pd_client, load_peer_raft_state, load_rf_engine_meta, now, send_request_to_store,
        RawRegion,
    },
    pd_control::PdControl,
    restore::{get_cluster_backup_meta, RestoreConfig, RestoreKeyspaceArgs},
    step, step_error,
};

const WORKING_PATH_PREFIX: &str = "tenant-restore";
const ZSTD_COMPRESSION_LEVEL: &str = "5"; // The same as ZSTD_COMPRESSION_LEVEL_FOR_REMOTE.

const REQUEST_RESTORE_SNAPSHOT_RETRY_LIMIT: usize = 10;

type RestoreResult<T> = std::result::Result<T, Box<dyn std::error::Error + Sync + Send>>;

pub(crate) fn execute_restore_keyspace(args: RestoreKeyspaceArgs) {
    match execute_restore_keyspace_impl(&args) {
        Ok(()) => {
            step!("Restore tenant {} succeed", args.keyspace_name);
        }
        Err(err) => {
            step_error!("Restore tenant {} error: {:?}", args.keyspace_name, err);
        }
    }
}

fn execute_restore_keyspace_impl(args: &RestoreKeyspaceArgs) -> RestoreResult<()> {
    let config: RestoreConfig = get_restore_keyspace_config_from_args(args);
    let pd_client: Arc<dyn PdClient> = Arc::new(create_pd_client(&config.security, &config.pd));
    let pd_control = PdControl::new(config.pd.clone(), config.security.clone());

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(8)
        .enable_all()
        .build()
        .unwrap();

    let keyspace_id = {
        let keyspace = runtime.block_on(pd_control.get_keyspace_by_name(&args.keyspace_name))?;
        assert_eq!(args.keyspace_name, keyspace.name);
        keyspace.id
    };

    restore_keyspace(
        keyspace_id,
        &args.name,
        args.working_path.as_deref(),
        &config,
        pd_client,
        &runtime,
    )
}

pub fn restore_keyspace(
    keyspace_id: u32,
    backup_name: &str,
    working_path: Option<&str>,
    config: &RestoreConfig,
    pd_client: Arc<dyn PdClient>,
    runtime: &Runtime,
) -> RestoreResult<()> {
    let working_dir = match working_path {
        Some(p) => TempDir::new_in(p, WORKING_PATH_PREFIX),
        None => TempDir::new(WORKING_PATH_PREFIX),
    }
    .unwrap();
    let working_path = working_dir.into_path();

    let (keyspace_start, keyspace_end) = ApiV2::get_txn_keyspace_range(keyspace_id);
    step!(
        "Start restore tenant {} from backup <{}>, tenant id:{} range:[{:?},{:?})",
        keyspace_id,
        backup_name,
        keyspace_id,
        keyspace_start,
        keyspace_end
    );

    let (cluster_backup, s3fs) = get_cluster_backup_meta(config, backup_name.to_owned());
    let dfs = Arc::new(s3fs);
    let mut cluster = BackupCluster::new(
        &cluster_backup,
        working_path,
        pd_client.clone(),
        dfs,
        keyspace_id,
    )?;
    step!("Restore {} shards from backup", cluster.shards_count());

    // trigger initial flush & flush mem-table to S3
    // NOTE: `cluster.shards` is NOT available before `flush_shards` finished.
    let flush_cnt = cluster.flush_shards()?;
    step!("Flush {} shards", flush_cnt);

    let truncate_ts_cnt = cluster.truncate_backup_ts()?;
    step!(
        "Truncate {} shards to ts {}",
        truncate_ts_cnt,
        cluster.backup_ts
    );

    // align target regions
    let target_regions = runtime.block_on(get_target_regions(
        &pd_client,
        &keyspace_start,
        &keyspace_end,
    ))?;
    let (aligned_regions, trimmed_shards_cnt) = cluster.align_target_regions(target_regions)?;
    step!(
        "Align {} backup shards to {} target regions and trim {} over bound shards",
        cluster.shards_count(),
        aligned_regions.len(),
        trimmed_shards_cnt
    );

    // gather SSTables for target regions
    let (target_shards, sstables_cnt) = cluster.gather_sstables(aligned_regions);
    step!(
        "Gather {} SSTables for {} target regions",
        sstables_cnt,
        target_shards.len(),
    );

    // TODO: remove replicas on TiFlash
    // TODO: retry on EpochNotMatch from `get_target_regions`.
    let snapshots = cluster.generate_snapshots(target_shards);
    let snapshots_count = snapshots.len();
    restore_snapshots(runtime, pd_client.clone(), snapshots)?;
    step!("Restore {snapshots_count} regions");

    // untag deleted S3 files

    Ok(())
}

#[derive(Default, Clone)]
struct BackupShard {
    pub region_id: u64,
    pub store_id: u64,
    pub peer_id: u64,
    pub need_flush: bool,
    pub meta: ShardMeta,
    pub(crate) raw_meta: Option<pb::ChangeSet>, // used for kv engine recovery only.
    raft_commit_index: u64,
}

impl fmt::Debug for BackupShard {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BackupShard")
            .field("tag", &format!("{}", self.tag()))
            .field("region_id", &self.region_id)
            .field("store_id", &self.store_id)
            .field("peer_id", &self.peer_id)
            .field("need_flush", &self.need_flush)
            .field("meta", &self.meta.to_change_set())
            .field("raw_meta", &self.raw_meta)
            .field("raft_commit_index", &self.raft_commit_index)
            .finish()
    }
}

#[cfg(test)]
impl PartialEq for BackupShard {
    fn eq(&self, other: &Self) -> bool {
        self.region_id == other.region_id
            && self.store_id == other.store_id
            && self.peer_id == other.peer_id
            && self.need_flush == other.need_flush
            && self.meta.to_change_set() == other.meta.to_change_set()
            && self.raw_meta == other.raw_meta
            && self.raft_commit_index == other.raft_commit_index
    }
}

impl BackupShard {
    pub fn tag(&self) -> ShardTag {
        ShardTag::new(self.store_id, IdVer::new(self.region_id, self.ver()))
    }

    pub fn ver(&self) -> u64 {
        self.meta.ver
    }

    pub fn start(&self) -> &[u8] {
        &self.meta.start
    }

    pub fn end(&self) -> &[u8] {
        &self.meta.end
    }

    pub fn table_version(&self) -> u64 {
        self.meta.base_version + self.meta.data_sequence
    }
}

#[derive(Debug, PartialEq)]
struct AlignedRegion {
    target_region: RawRegion,
    backup_shards_id: Vec<u64>,
}

struct BackupCluster {
    path: PathBuf,
    pd_client: Arc<dyn PdClient>,
    dfs: Arc<S3Fs>,
    keyspace_id: u32,
    keyspace_prefix: Vec<u8>,
    backup_ts: u64,

    // store_id -> rf_engine.
    raft_engines: HashMap<u64, RfEngine>,
    // must be `Some` after `new()`.
    kv_engine: Option<kvengine::Engine>,

    // region_id -> shard.
    shards: HashMap<u64, BackupShard>,
    // region_id -> raw_meta, used for kv_engine recover.
    raw_metas: HashMap<u64, pb::ChangeSet>,
    // store_id -> Vec<shard_id>.
    store_shards: HashMap<u64, Vec<u64>>,
    // shard_id sorted by start_key.
    sorted_shards: Vec<u64>,
    // store_id -> Vec<shard_id>.
    shards_need_flush: HashMap<u64, Vec<u64>>,
    // store_id -> Vec<shard_id>.
    shards_need_truncate: HashMap<u64, Vec<u64>>,

    meta_applier: Option<Arc<MetaApplier>>,
}

impl Drop for BackupCluster {
    fn drop(&mut self) {
        self.stop_engines();
    }
}

impl BackupCluster {
    pub fn new(
        cluster_meta: &ClusterBackupMeta,
        path: PathBuf,
        pd_client: Arc<dyn PdClient>,
        dfs: Arc<S3Fs>,
        keyspace_id: u32,
    ) -> RestoreResult<BackupCluster> {
        let (keyspace_prefix, _) = ApiV2::get_txn_keyspace_range(keyspace_id);
        let mut cluster = Self {
            path,
            pd_client,
            dfs,
            keyspace_id,
            keyspace_prefix,
            backup_ts: cluster_meta.backup_ts,
            shards: Default::default(),
            raw_metas: Default::default(),
            raft_engines: Default::default(),
            kv_engine: Default::default(),
            store_shards: Default::default(),
            sorted_shards: Default::default(),
            shards_need_flush: Default::default(),
            shards_need_truncate: Default::default(),
            meta_applier: None,
        };

        let mut store_configs = HashMap::with_capacity(cluster_meta.stores.len());

        for store in &cluster_meta.stores {
            let store_id = store.get_store_id();
            let store_config = cluster.generate_store_config(store_id);
            store_configs.insert(store_id, store_config);

            cluster.setup_raft_engine(
                store_id,
                cluster_meta,
                store_configs.get(&store_id).unwrap(),
            )?;
        }
        cluster.load_shards()?;

        let stores_id: Vec<_> = cluster.get_all_stores_id().collect();
        for store_id in stores_id {
            cluster.setup_kv_engine(store_id, store_configs.get(&store_id).unwrap())?;
        }
        Ok(cluster)
    }

    fn setup_raft_engine(
        &mut self,
        store_id: u64,
        cluster_backup: &ClusterBackupMeta,
        conf: &TikvConfig,
    ) -> RestoreResult<()> {
        rfengine::restore(
            self.dfs.clone(),
            cluster_backup,
            store_id,
            Path::new(&conf.raft_store.raftdb_path),
            None, // TODO: pass `Some(keyspace_id)` in.
        );

        let rf_engine = TikvServer::init_raft_engine(conf)?;
        self.raft_engines.insert(store_id, rf_engine);
        Ok(())
    }

    // Take all raw metas to release memory.
    fn take_raw_metas(&mut self, store_id: u64) -> HashMap<u64, pb::ChangeSet> {
        let shards = self.store_shards.get(&store_id).unwrap().to_owned();
        let mut raw_metas = HashMap::with_capacity(shards.len());
        for shard_id in shards {
            raw_metas.insert(shard_id, self.raw_metas.remove(&shard_id).unwrap());
        }
        raw_metas
    }

    fn get_shards_need_flush_and_truncate(&self, store_id: u64) -> Vec<u64> {
        let mut ret_shards = self.shards_need_flush.get(&store_id).unwrap().to_owned();
        let mut shard_need_truncate = self.shards_need_truncate.get(&store_id).unwrap().to_owned();
        ret_shards.append(&mut shard_need_truncate);
        ret_shards.dedup();
        ret_shards
    }

    fn setup_kv_engine(&mut self, store_id: u64, conf: &TikvConfig) -> RestoreResult<()> {
        let rf_engine = self.raft_engines.get(&store_id).unwrap();
        let recoverer = rfstore::store::RecoverHandler::new(rf_engine.clone());

        let shards_need_load = self.get_shards_need_flush_and_truncate(store_id);
        let raw_metas = self.take_raw_metas(store_id);

        if let Some(kv_engine) = self.kv_engine.as_mut() {
            if !shards_need_load.is_empty() {
                let mut meta_iter =
                    MetaIterator::new(kv_engine.get_engine_id(), shards_need_load, raw_metas);
                let metas = kv_engine.read_meta(&mut meta_iter)?;
                info!("kv_engine load {} shards in restore tenant", metas.len());
                kv_engine.load_shards(metas, recoverer)?;
            }
        } else {
            // Even if `shards_need_load.is_empty()`, `self.kv_engine` should be
            // constructed for later use.
            let io_rate_limiter =
                Arc::new(IoRateLimiter::new(IoRateLimitMode::WriteOnly, true, true));
            io_rate_limiter
                .set_io_rate_limit(conf.storage.io_rate_limit.max_bytes_per_sec.0 as usize);

            let mut meta_iter = MetaIterator::new(store_id, shards_need_load, raw_metas);
            let (kv_engine, _, receiver) = TikvServer::init_kv_engine(
                self.pd_client.clone(),
                conf,
                self.dfs.clone(),
                io_rate_limiter,
                &mut meta_iter,
                recoverer,
            )?;

            // Move shards to meta_applier to apply meta change in `flush_mem_table`.
            // Move back after flush finished.
            let shards = mem::take(&mut self.shards);
            let meta_applier = Arc::new(MetaApplier::new(kv_engine.clone(), shards, receiver));
            self.meta_applier = Some(meta_applier.clone());
            thread::spawn(move || {
                meta_applier.run();
            });

            self.kv_engine = Some(kv_engine);
        }
        Ok(())
    }

    fn stop_engines(&mut self) {
        let raft_engines = mem::take(&mut self.raft_engines);
        for rf in raft_engines.into_values() {
            rf.stop_worker();
            drop(rf);
        }
        let kv_engine = self.kv_engine.take().unwrap();
        drop(kv_engine);
    }

    fn generate_store_config(&self, store_id: u64) -> TikvConfig {
        let store_path = self.path.join(store_id.to_string());
        let rf_engine_path = store_path.join("raft");

        let mut config = TikvConfig::default();
        config.storage.data_dir = store_path.to_str().unwrap().to_string();
        config.raft_store.raftdb_path = rf_engine_path.to_str().unwrap().to_string();
        config.dfs.zstd_compression_level = ZSTD_COMPRESSION_LEVEL.to_string();
        config.rocksdb.max_background_jobs = 2;
        config.rocksdb.max_sub_compactions = 1;

        TikvServer::init_config(config).get_current()
    }

    fn collect_prefix_shards(store_id: u64, rf: &RfEngine, prefix: &[u8]) -> Vec<BackupShard> {
        let region_peers = rf.get_region_peer_map();
        let mut prefix_shards = vec![];
        for (region_id, peer_id) in region_peers {
            if region_id == 0 {
                continue;
            }
            let meta = match load_rf_engine_meta(rf, peer_id) {
                Some(meta) => meta,
                None => {
                    warn!(
                        "region {} peer {} has no rf_engine meta",
                        region_id, peer_id
                    );
                    continue;
                }
            };
            assert_eq!(region_id, meta.shard_id);

            let snap = meta.get_snapshot();
            if snap.get_start().starts_with(prefix) {
                let raft_last_index = rf.get_last_index(peer_id);
                let need_flush_mem_table = raft_last_index.map_or(false, |last_idx| {
                    assert!(last_idx >= snap.get_data_sequence());
                    last_idx > snap.get_data_sequence()
                });
                let need_initial_flush = meta.has_parent();

                let raft_state = load_peer_raft_state(rf, peer_id, meta.shard_ver).unwrap();
                let raft_commit_index = raft_state.get_hard_state().commit;

                let shard = BackupShard {
                    region_id,
                    store_id,
                    peer_id,
                    need_flush: need_flush_mem_table || need_initial_flush,
                    meta: ShardMeta::new(rf.get_engine_id(), &meta),
                    raw_meta: Some(meta),
                    raft_commit_index,
                };
                debug!("collect_prefix_shard: {:?}", shard);
                prefix_shards.push(shard);
            }
        }
        prefix_shards
    }

    fn load_shards(&mut self) -> RestoreResult<()> {
        let all_shards =
            HashMap::from_iter(self.raft_engines.iter().map(|(store_id, rf_engine)| {
                (
                    *store_id,
                    Self::collect_prefix_shards(*store_id, rf_engine, &self.keyspace_prefix),
                )
            }));

        let (mut leader_shards, mut sorted_shards_id) = Self::get_leader_shards(all_shards)?;
        self.shards = mem::take(&mut leader_shards);
        self.sorted_shards = mem::take(&mut sorted_shards_id);
        debug!("BackupCluster.load_shards: {:?}", self.shards);
        debug!("BackupCluster.sorted_shards: {:?}", self.sorted_shards);

        for (&shard_id, shard) in self.shards.iter_mut() {
            self.store_shards
                .entry(shard.store_id)
                .and_modify(|ids| ids.push(shard_id))
                .or_insert_with(|| vec![shard_id]);

            // Take `raw_meta`s out from `shards`, as `shards` will be moved into
            // `MetaApplier` for flush shards.
            self.raw_metas
                .insert(shard_id, shard.raw_meta.take().unwrap());
        }

        for (&store_id, shards) in &self.store_shards {
            let shards_need_flush = shards
                .iter()
                .filter(|&&shard_id| self.get_shard(shard_id).unwrap().need_flush)
                .copied()
                .collect();
            self.shards_need_flush.insert(store_id, shards_need_flush);
            let shards_need_truncate = shards
                .iter()
                .filter(|&&id| self.get_shard(id).unwrap().meta.max_ts >= self.backup_ts)
                .copied()
                .collect();
            self.shards_need_truncate
                .insert(store_id, shards_need_truncate);
        }

        self.verify_shards()
    }

    fn get_leader_shards(
        all_shards: HashMap<u64, Vec<BackupShard>>,
    ) -> RestoreResult<(
        HashMap<u64, BackupShard>, // shard_id -> BackupShard
        Vec<u64>,                  // Vec<shard_id> sorted by BackupShard.start()
    )> {
        // Assume 3 replicas here.
        let shards_cnt = all_shards.values().map(|x| x.len()).sum::<usize>() / 3;
        // leader_shards: shard_id -> BackupShard.
        let mut leader_shards = HashMap::with_capacity(shards_cnt);

        for mut shard in all_shards.into_values().flatten() {
            leader_shards
                .entry(shard.region_id)
                .and_modify(|old: &mut BackupShard| {
                    if shard.raft_commit_index > old.raft_commit_index {
                        std::mem::swap(old, &mut shard);
                    }
                })
                .or_insert(shard);
        }

        let mut sorted_shards: Vec<u64> = leader_shards
            .values()
            .sorted_by(|a, b| a.start().cmp(b.start()))
            .map(|x| x.region_id)
            .collect();

        // Handle overlapping shards.
        // Shards overlap will happen when some followers had not finished split or
        // merge.
        if sorted_shards.len() > 1 {
            let mut shards_to_remove: HashSet<u64> = HashSet::default();

            'outer: for i in 0..sorted_shards.len() - 1 {
                let left = leader_shards.get(&sorted_shards[i]).unwrap();
                if shards_to_remove.contains(&left.region_id) {
                    continue;
                }

                'inner: for j in i + 1..sorted_shards.len() {
                    let right = leader_shards.get(&sorted_shards[j]).unwrap();
                    if shards_to_remove.contains(&right.region_id) {
                        continue 'inner;
                    }
                    if right.start() >= left.end() {
                        continue 'outer;
                    }

                    match Ord::cmp(&left.ver(), &right.ver()) {
                        cmp::Ordering::Equal => {
                            return Err(box_err!(
                                "overlapping shards should not have same version, left:{:?}, right:{:?}",
                                left,
                                right
                            ));
                        }
                        cmp::Ordering::Less => {
                            shards_to_remove.insert(left.region_id);
                            continue 'outer;
                        }
                        cmp::Ordering::Greater => {
                            shards_to_remove.insert(right.region_id);
                            continue 'inner;
                        }
                    }
                }
            }

            if !shards_to_remove.is_empty() {
                for shard_id in &shards_to_remove {
                    leader_shards.remove(shard_id);
                }
                sorted_shards.retain(|shard_id| !shards_to_remove.contains(shard_id));
            }
        }

        Ok((leader_shards, sorted_shards))
    }

    fn verify_shards(&self) -> RestoreResult<()> {
        let (start, end) = ApiV2::get_txn_keyspace_range(self.keyspace_id);
        if self.sorted_shards.is_empty() {
            return Err(box_err!("no shard"));
        }

        let first_shard = self.get_sorted_shard(0);
        let last_shard = self.get_sorted_shard(self.sorted_shards.len() - 1);
        if first_shard.start() != start {
            return Err(box_err!(
                "start key of first region not match: {:?}",
                first_shard
            ));
        }
        if last_shard.end() != end {
            return Err(box_err!(
                "end key of last region not match: {:?}",
                last_shard
            ));
        }

        for shards_id in self.sorted_shards.windows(2) {
            let left = self.get_shard(shards_id[0]).unwrap();
            let right = self.get_shard(shards_id[1]).unwrap();
            if left.end() != right.start() {
                return Err(box_err!(
                    "region boundary not match: {:?}, {:?}",
                    left,
                    right
                ));
            }
        }

        Ok(())
    }

    fn get_sorted_shard(&self, sorted_idx: usize) -> &BackupShard {
        self.get_shard(self.sorted_shards[sorted_idx]).unwrap()
    }

    pub fn get_shard(&self, shard_id: u64) -> Option<&BackupShard> {
        self.shards.get(&shard_id)
    }

    pub fn get_shard_mut(&mut self, shard_id: u64) -> Option<&mut BackupShard> {
        self.shards.get_mut(&shard_id)
    }

    pub fn shards_count(&self) -> usize {
        self.sorted_shards.len()
    }

    #[inline]
    fn get_all_stores_id(&self) -> impl ExactSizeIterator<Item = u64> + '_ {
        self.store_shards.keys().copied()
    }

    pub fn flush_shards(&mut self) -> RestoreResult<usize /* number of flushed shards */> {
        let mut flush_cnt = 0_usize;
        for store_id in self.get_all_stores_id() {
            let shards_id = self.shards_need_flush.get(&store_id).unwrap();
            // TODO: run in parallel.
            flush_cnt += self.flush_shards_for_store(shards_id)?;
        }

        self.check_flushed(Duration::from_secs(30))?;

        // Take back from meta_applier.
        let mut shards = self.meta_applier.as_ref().unwrap().take_shards().unwrap();
        self.shards = mem::take(&mut shards);

        Ok(flush_cnt)
    }

    fn flush_shards_for_store(
        &self,
        shards_id: &[u64],
    ) -> RestoreResult<usize /* number of flushed shards */> {
        let kv_engine = self.kv_engine.as_ref().unwrap();
        for shard_id in shards_id {
            let engine_shard = kv_engine.get_shard(*shard_id).unwrap();
            kv_engine.flush_shard_for_restore(&engine_shard);
        }

        Ok(shards_id.len())
    }

    fn check_flushed(&self, timeout: Duration) -> RestoreResult<()> {
        let is_flushed = |stats: &ShardStats| {
            stats.flushed && stats.mem_table_size == 0 && stats.mem_table_count == 1
        };

        let kv_engine = self.kv_engine.as_ref().unwrap();
        let begin = Instant::now_coarse();
        while begin.saturating_elapsed() < timeout {
            let done = kv_engine
                .get_all_shard_stats()
                .into_iter()
                .all(|stats| is_flushed(&stats));
            if done {
                return Ok(());
            }
            thread::sleep(Duration::from_millis(500));
        }

        let stats: Vec<_> = kv_engine
            .get_all_shard_stats()
            .into_iter()
            .filter(|stats| !is_flushed(stats))
            .collect();
        error!("wait_for_mem_table_flush timeout, stats: {:?}", stats);
        Err(box_err!("wait_for_mem_table_flush timeout"))
    }

    fn truncate_backup_ts(&mut self) -> RestoreResult<usize> {
        let mut truncate_cnt = 0_usize;
        let stores_id: Vec<u64> = self.get_all_stores_id().collect();
        for store_id in stores_id {
            let shards_need_truncate = self.get_shards_need_flush_and_truncate(store_id);
            truncate_cnt += shards_need_truncate.len();
            for id in shards_need_truncate {
                self.truncate_ts(id)?;
            }
        }
        Ok(truncate_cnt)
    }

    // NOTE: `sorted_backup_shards_id` & `sorted_target_regions` must be sorted by
    // start_key.
    fn align_target_regions_impl(
        sorted_backup_shards_id: &[u64],
        backup_shards: &HashMap<u64, BackupShard>,
        sorted_target_regions: Vec<RawRegion>,
    ) -> Vec<AlignedRegion> {
        let mut aligned_regions = Vec::with_capacity(sorted_target_regions.len());
        let mut idx = 0;

        let key_in_shard =
            |key: &[u8], shard: &BackupShard| key >= shard.start() && key < shard.end();

        let get_backup_shard =
            |idx: usize| backup_shards.get(&sorted_backup_shards_id[idx]).unwrap();

        for target_region in sorted_target_regions {
            // Check overlapping with previous backup_region.
            if idx > 0 && key_in_shard(target_region.get_start_key(), get_backup_shard(idx - 1)) {
                idx -= 1;
            }

            let mut aligned_shards_id = vec![];
            while idx < sorted_backup_shards_id.len()
                && get_backup_shard(idx).start() < target_region.get_end_key()
            {
                aligned_shards_id.push(sorted_backup_shards_id[idx]);
                idx += 1;
            }

            aligned_regions.push(AlignedRegion {
                target_region,
                backup_shards_id: aligned_shards_id,
            });
        }

        aligned_regions
    }

    pub fn align_target_regions(
        &mut self,
        target_regions: Vec<RawRegion>,
    ) -> RestoreResult<(
        Vec<AlignedRegion>,
        usize, // number of trimmed shards
    )> {
        let aligned_regions =
            Self::align_target_regions_impl(&self.sorted_shards, &self.shards, target_regions);
        let trimmed_shards_cnt = self.trim_over_bound_shards(&aligned_regions)?;
        Ok((aligned_regions, trimmed_shards_cnt))
    }

    pub fn gather_sstables(
        &self,
        aligned_regions: Vec<AlignedRegion>,
    ) -> (Vec<ShardMeta>, usize /* number of sstables */) {
        let mut sstables_cnt = 0;
        let mut target_shards = Vec::with_capacity(aligned_regions.len());
        for mut region in aligned_regions {
            let mut meta = ShardMeta::default();
            meta.id = region.target_region.id;
            meta.ver = region.target_region.epoch.version;
            meta.start = region.target_region.take_start_key();
            meta.end = region.target_region.take_end_key();

            for shard_id in region.backup_shards_id {
                let shard = self.get_shard(shard_id).unwrap();
                for (&file_id, file_meta) in shard.meta.all_files() {
                    if meta.overlap_table(file_meta.smallest.as_ref(), file_meta.biggest.as_ref()) {
                        meta.add_file(
                            file_id,
                            file_meta.cf as i32,
                            file_meta.level as u32,
                            &file_meta.smallest,
                            &file_meta.biggest,
                        );
                        sstables_cnt += 1;
                    }
                }
                // Use `base_version` as table version, and `data_sequence` is 0.
                // And they will be adjusted at server side in `restore_shard` procedure.
                meta.base_version = cmp::max(meta.base_version, shard.table_version());
            }

            target_shards.push(meta);
        }
        (target_shards, sstables_cnt)
    }

    fn truncate_ts(&mut self, shard_id: u64) -> RestoreResult<bool /* has_truncate_ts */> {
        let shard_meta = &self.get_shard(shard_id).unwrap().meta;
        let kvengine = self.kv_engine.as_ref().unwrap();
        let shard = kvengine
            .get_shard_with_ver(shard_meta.id, shard_meta.ver)
            .expect("Could not find shard with meta");
        let res_cs = kvengine
            .truncate_with_ts(&shard, self.backup_ts.into())?
            .unwrap();
        if res_cs.has_truncate_ts() {
            let shard = self.get_shard_mut(shard_id).unwrap();
            info!(
                "before truncate ts: {:?}, table change: {:?}",
                shard,
                res_cs.get_truncate_ts()
            );
            shard.meta.apply_change_set(&res_cs);
            info!("after apply_truncate_ts: {:?}", shard);
            Ok(true)
        } else {
            Ok(false)
        }
    }

    fn trim_over_bound(&mut self, shard_id: u64) -> RestoreResult<bool /* has_trim_over_bound */> {
        let shard = self.get_shard(shard_id).unwrap();
        let kvengine = self.kv_engine.as_ref().unwrap();

        let res_cs = kvengine.trim_over_bound_by_meta(&shard.meta)?;
        if res_cs.has_trim_over_bound() {
            let shard = self.get_shard_mut(shard_id).unwrap();
            debug!(
                "before trim_over_bound: {:?}, table change: {:?}",
                shard,
                res_cs.get_trim_over_bound()
            );
            shard.meta.apply_change_set(&res_cs);
            debug!("after apply_trim_over_bound: {:?}", shard);
            Ok(true)
        } else {
            Ok(false)
        }
    }

    fn trim_over_bound_shards(
        &mut self,
        aligned_regions: &[AlignedRegion],
    ) -> RestoreResult<usize /* number of trimmed shards */> {
        let mut trim_shards_cnt = 0_usize;
        let mut unique_shards = HashSet::new();
        for region in aligned_regions
            .iter()
            .filter(|x| x.backup_shards_id.len() > 1)
        {
            for &shard_id in &region.backup_shards_id {
                if unique_shards.contains(&shard_id) {
                    continue;
                }
                unique_shards.insert(shard_id);

                // TODO: Run in parallel.
                if self.trim_over_bound(shard_id)? {
                    trim_shards_cnt += 1;
                }
            }
        }

        Ok(trim_shards_cnt)
    }

    pub fn generate_snapshots(&mut self, target_shards: Vec<ShardMeta>) -> Vec<pb::ChangeSet> {
        target_shards
            .into_iter()
            .map(|meta| {
                let mut cs = meta.to_change_set();
                let snap = cs.take_snapshot();
                cs.set_restore_shard(snap);
                cs
            })
            .collect()
    }
}

struct MetaApplier {
    engine: kvengine::Engine,
    shards: RwLock<Option<HashMap<u64, BackupShard>>>,
    store_rx: mpsc::Receiver<StoreMsg>,
}

impl MetaApplier {
    fn new(
        engine: kvengine::Engine,
        shards: HashMap<u64, BackupShard>,
        store_rx: mpsc::Receiver<StoreMsg>,
    ) -> Self {
        Self {
            engine,
            shards: RwLock::new(Some(shards)),
            store_rx,
        }
    }

    pub fn take_shards(&self) -> Option<HashMap<u64, BackupShard>> {
        self.shards.wl().take()
    }

    fn run(&self) {
        'outer: loop {
            let msg = match self.store_rx.recv() {
                Ok(msg) => msg,
                Err(err) => {
                    error!("applier recv task error: {:?}", err);
                    return;
                }
            };
            match msg {
                StoreMsg::GenerateEngineChangeSet(mut cs) => {
                    let tag =
                        ShardTag::new(self.engine.get_engine_id(), IdVer::from_change_set(&cs));
                    let engine_shard = self.engine.get_shard(cs.shard_id).unwrap();
                    let seq = cmp::max(
                        engine_shard.get_write_sequence(),
                        engine_shard.get_meta_sequence(),
                    ) + 1;
                    cs.set_sequence(seq);
                    debug!("{} apply change set: {:?}", tag, cs);

                    {
                        let mut shards = self.shards.wl();
                        if let Some(shards) = shards.as_mut() {
                            let shard = shards.get_mut(&cs.shard_id).unwrap();
                            shard.meta.apply_change_set(&cs);
                        } else {
                            if cs.has_compaction() {
                                debug!("MetaApplier: ignore changeset: {:?}", cs);
                                self.engine.meta_committed(&cs, true);
                                continue 'outer;
                            }
                            panic!(
                                "MetaApplier: changeset is lost due to shards map had be taken: {:?}",
                                cs
                            );
                        }
                    }

                    self.engine.meta_committed(&cs, false);
                    match self
                        .engine
                        .prepare_change_set(cs, false)
                        .and_then(|cs| self.engine.apply_change_set(cs))
                    {
                        Ok(()) => debug!("{} MetaApplier apply change set successfully", tag),
                        Err(err) => {
                            error!("{} MetaApplier apply change set failed: {:?}", tag, err)
                        }
                    }
                }
                _ => {
                    error!("unexpected msg");
                }
            }
        }
    }
}

struct MetaIterator {
    store_id: u64,
    shards: Vec<u64>,
    raw_metas: HashMap<u64, pb::ChangeSet>,
}

impl MetaIterator {
    pub fn new(store_id: u64, shards: Vec<u64>, raw_metas: HashMap<u64, pb::ChangeSet>) -> Self {
        Self {
            store_id,
            shards,
            raw_metas,
        }
    }
}

impl kvengine::MetaIterator for MetaIterator {
    fn iterate<F>(&mut self, mut f: F) -> kvengine::Result<()>
    where
        F: FnMut(kvenginepb::ChangeSet),
    {
        for shard_id in &self.shards {
            f(self.raw_metas.get(shard_id).unwrap().to_owned());
        }
        Ok(())
    }

    fn engine_id(&self) -> u64 {
        self.store_id
    }
}

async fn get_target_regions(
    pd_client: &Arc<dyn PdClient>,
    start_key: &[u8],
    end_key: &[u8],
) -> RestoreResult<Vec<RawRegion>> {
    let encoded_start_key = Key::from_raw(start_key).into_encoded();
    let encoded_end_key = Key::from_raw(end_key).into_encoded();

    let mut regions = vec![];
    let mut next_key = encoded_start_key.clone();
    while next_key < encoded_end_key {
        let region = pd_client.get_region_async(&next_key).await?;
        next_key = region.get_end_key().to_vec();
        regions.push(region.into());
    }
    debug!("target_regions: {:?}", regions);

    verify_regions_boundary(start_key, end_key, &regions)?;
    Ok(regions)
}

fn verify_regions_boundary(
    start_key: &[u8],
    end_key: &[u8],
    regions: &[RawRegion],
) -> RestoreResult<()> {
    if regions.is_empty() {
        return Err(box_err!("no region"));
    }

    let first_region = regions.first().unwrap();
    let last_region = regions.last().unwrap();
    if first_region.get_start_key() != start_key {
        return Err(box_err!(
            "unexpected start key of first region: {:?}",
            first_region
        ));
    } else if last_region.get_end_key() != end_key {
        return Err(box_err!(
            "unexpected end key of last region: {:?}",
            last_region
        ));
    }

    for region in regions.windows(2) {
        if region[0].get_end_key() != region[1].get_start_key() {
            return Err(box_err!(
                "region boundary not match: {:?}, {:?}",
                region[0],
                region[1]
            ));
        }
    }

    Ok(())
}

fn restore_snapshots(
    runtime: &Runtime,
    pd_client: Arc<dyn PdClient>,
    snapshots: Vec<pb::ChangeSet>,
) -> RestoreResult<()> {
    let mut handles = Vec::with_capacity(snapshots.len());
    for snap in snapshots {
        let pd_client = pd_client.clone();
        let task = async move { request_restore_snapshot(pd_client, &snap).await };
        handles.push(runtime.spawn(task));
    }

    let errors: Vec<_> = runtime
        .block_on(futures::future::join_all(handles))
        .into_iter()
        .filter_map(|x| x.unwrap().err())
        .collect();
    if !errors.is_empty() {
        Err(box_err!("{:?}", errors))
    } else {
        Ok(())
    }
}

async fn request_restore_snapshot(
    pd_client: Arc<dyn PdClient>,
    cs: &pb::ChangeSet,
) -> RestoreResult<()> {
    let post_data = Cow::from(cs.write_to_bytes().unwrap());
    let mut err = None;
    'retry: for i in 0..REQUEST_RESTORE_SNAPSHOT_RETRY_LIMIT {
        let (region, leader) = match pd_client.get_region_leader_by_id(cs.shard_id).await? {
            Some((region, leader)) => (region, leader),
            None => {
                let err_msg = format!("no leader of region {}", cs.shard_id);
                warn!("{}", err_msg);
                err = box_err!(err_msg);
                tokio::time::sleep(Duration::from_millis(1000)).await;
                continue 'retry;
            }
        };

        let region_ver = region.get_region_epoch().get_version();
        let shard_ver = cs.get_shard_ver();
        if region_ver != shard_ver {
            return Err(box_err!(
                "region version not match (region.ver {}, cs.ver {}), try again",
                region_ver,
                shard_ver
            ));
        }

        let store = pd_client.get_store_async(leader.get_store_id()).await?;
        let tag = ShardTag::new(store.get_id(), IdVer::new(cs.shard_id, cs.shard_ver));

        let uri =
            Uri::from_str(&format!("http://{}/restore-shard", &store.status_address)).unwrap();
        let req = Request::post(uri)
            .body(Body::from(post_data.clone()))
            .unwrap();
        match send_request_to_store(req, &store).await {
            Ok(_resp) => {
                debug!("{} request_restore_snapshot succeed", tag);
                return Ok(());
            }
            Err(e) => {
                let err_msg = format!("{} request_restore_snapshot #{i} error: {:?}", tag, e);
                warn!("{}", err_msg);
                err = box_err!(err_msg);
                tokio::time::sleep(Duration::from_millis(1000)).await;
                continue 'retry;
            }
        }
    }
    Err(err.expect("there must be error"))
}

#[cfg(test)]
mod tests {
    use std::collections::{hash_map::Entry, BTreeMap};

    use super::*;

    #[test]
    fn test_get_leader_shards() {
        let make_backup_shard = |tuple: (
            u64,  // shard_id
            u64,  // ver
            &str, // start
            &str, // end
        )|
         -> BackupShard {
            let mut shard = BackupShard::default();
            shard.region_id = tuple.0;
            shard.meta.ver = tuple.1;
            shard.meta.start = tuple.2.as_bytes().to_vec();
            shard.meta.end = tuple.3.as_bytes().to_vec();
            // Use `ver` as raft log index, assume that the newer ver, the faster raft
            // progress.
            shard.raft_commit_index = shard.ver();
            shard
        };

        let make_store_shards =
            |all_shards: &mut HashMap<u64, Vec<BackupShard>>,
             store_id: u64,
             shards: Vec<(u64, u64, &str, &str)>| {
                for shard_tuple in shards {
                    let shard = make_backup_shard(shard_tuple);
                    match all_shards.entry(store_id) {
                        Entry::Occupied(o) => {
                            o.into_mut().push(shard);
                        }
                        Entry::Vacant(v) => {
                            v.insert(vec![shard]);
                        }
                    }
                }
            };

        let cases = vec![
            (
                vec![(1, 100, "00", "01")], // store0: Vec<(shard_id, ver, start, end)>
                vec![(1, 100, "00", "01")], // store1
                vec![(1, 100, "00", "01")], // store2
                Some(vec![(1, 100, "00", "01")]), /* expected Option<Vec<(shard_id, ver, start,
                                             * end)>> */
            ),
            (
                vec![(1, 100, "00", "01"), (2, 200, "01", "02")],
                vec![(1, 100, "00", "01"), (2, 200, "01", "02")],
                vec![(1, 100, "00", "01"), (2, 200, "01", "02")],
                Some(vec![(1, 100, "00", "01"), (2, 200, "01", "02")]),
            ),
            (
                vec![(1, 100, "00", "01"), (2, 200, "01", "02")],
                vec![(1, 201, "00", "02")], // merge from shard 1 & 2
                vec![(1, 100, "00", "01"), (2, 200, "01", "02")],
                Some(vec![(1, 201, "00", "02")]),
            ),
            (
                vec![(1, 100, "00", "01"), (2, 200, "01", "02")],
                vec![(2, 201, "00", "02")], // merge from shard 1 & 2
                vec![(1, 100, "00", "01"), (2, 200, "01", "02")],
                Some(vec![(2, 201, "00", "02")]),
            ),
            (
                vec![(1, 101, "00", "01"), (2, 101, "01", "02")], // split from shard 1
                vec![(1, 100, "00", "02")],
                vec![(1, 100, "00", "02")],
                Some(vec![(1, 101, "00", "01"), (2, 101, "01", "02")]),
            ),
            (
                vec![(2, 100, "01", "02")],
                vec![(1, 100, "00", "01")],
                vec![(3, 100, "02", "03")],
                Some(vec![
                    (1, 100, "00", "01"),
                    (2, 100, "01", "02"),
                    (3, 100, "02", "03"),
                ]),
            ),
            (
                vec![(1, 100, "00", "01")],
                vec![(2, 100, "00", "02")], // error as overlapping with same ver
                vec![(1, 100, "00", "01")],
                None,
            ),
            #[cfg_attr(rustfmt, rustfmt_skip)]
            (
                vec![(1, 100, "00", "01"), (2, 100, "01", "03"), (3, 100, "03", "04"), (4, 100, "04", "06"), (7, 102, "06", "07"), (8, 102, "07", "08")],
                vec![(1, 100, "00", "01"), (2, 101, "01", "02"), (3, 101, "02", "04"), (4, 100, "04", "05"), (5, 100, "05", "08")],
                vec![(1, 100, "00", "01"), (2, 100, "01", "03"), (3, 100, "03", "04"), (5, 101, "04", "06"), (6, 101, "06", "07")],
                Some(vec![
                    (1, 100, "00", "01"), (2, 101, "01", "02"), (3, 101, "02", "04"), (5, 101, "04", "06"), (7, 102, "06", "07"), (8, 102, "07", "08"),
                ]),
            ),
        ];

        for (case_idx, (store0, store1, store2, expected)) in cases.into_iter().enumerate() {
            let mut all_shards = HashMap::default();
            make_store_shards(&mut all_shards, 0, store0);
            make_store_shards(&mut all_shards, 1, store1);
            make_store_shards(&mut all_shards, 2, store2);

            let res = BackupCluster::get_leader_shards(all_shards);
            if let Some(expected) = expected {
                let expected_sorted_shards: Vec<u64> = expected.iter().map(|x| x.0).collect();
                let expected_shards: HashMap<u64, BackupShard> =
                    HashMap::from_iter(expected.into_iter().map(|x| (x.0, make_backup_shard(x))));

                let (shards, sorted_shards) = res.unwrap();
                assert_eq!(
                    BTreeMap::from_iter(shards.into_iter()),
                    BTreeMap::from_iter(expected_shards.into_iter()),
                    "case: {}",
                    case_idx
                );
                assert_eq!(sorted_shards, expected_sorted_shards, "case: {}", case_idx);
            } else {
                assert!(res.is_err(), "case: {}", case_idx);
            }
        }
    }

    #[test]
    fn test_align_target_regions() {
        let cases = vec![
            // backup_regions, target_regions, align_regions_id
            (vec![0, 10], vec![0, 10], vec![vec![0]]),
            (vec![0, 5, 10], vec![0, 5, 10], vec![vec![0], vec![5]]),
            (
                vec![0, 3, 6, 10],
                vec![0, 5, 10],
                vec![vec![0, 3], vec![3, 6]],
            ),
            (
                vec![0, 5, 10],
                vec![0, 3, 6, 10],
                vec![vec![0], vec![0, 5], vec![5]],
            ),
            #[cfg_attr(rustfmt, rustfmt_skip)]
            (
            vec![   0,       2,                4,       6, 7, 8, 9, 10],
            vec![   0,       2,       3,       4,       6,          10],
            vec![vec![0], vec![2], vec![2], vec![4], vec![6,7,8,9]],
            ),
        ];

        let make_regions = |keys: Vec<u64>| -> Vec<RawRegion> {
            let mut regions = Vec::with_capacity(keys.len() - 1);
            for w in keys.as_slice().windows(2) {
                regions.push(RawRegion {
                    id: w[0],
                    raw_start: w[0].to_be_bytes().to_vec(),
                    raw_end: w[1].to_be_bytes().to_vec(),
                    ..Default::default()
                });
            }

            regions
        };

        let make_backup_shards = |regions: Vec<RawRegion>| {
            let mut shards = HashMap::with_capacity(regions.len());
            let mut shards_id = Vec::with_capacity(regions.len());
            for mut r in regions {
                let mut shard = BackupShard::default();
                shard.region_id = r.id;
                shard.meta.start = r.take_start_key();
                shard.meta.end = r.take_end_key();

                shards_id.push(shard.region_id);
                shards.insert(shard.region_id, shard);
            }

            (shards, shards_id)
        };

        for (case_idx, (backup_shards_key, target_regions_key, expected_regions_id)) in
            cases.into_iter().enumerate()
        {
            let backup_regions = make_regions(backup_shards_key);
            let (backup_shards, backup_shards_id) = make_backup_shards(backup_regions);

            let target_regions = make_regions(target_regions_key);

            let mut expected = Vec::with_capacity(target_regions.len());
            for (i, target_region) in target_regions.clone().into_iter().enumerate() {
                expected.push(AlignedRegion {
                    target_region,
                    backup_shards_id: expected_regions_id[i].clone(),
                });
            }

            let aligned_regions = BackupCluster::align_target_regions_impl(
                &backup_shards_id,
                &backup_shards,
                target_regions,
            );
            assert_eq!(aligned_regions, expected, "case: {}", case_idx);
        }
    }
}

pub fn get_restore_keyspace_config_from_args(args: &RestoreKeyspaceArgs) -> RestoreConfig {
    let mut config = RestoreConfig::default();
    if args.config.exists() {
        let data = std::fs::read(args.config.clone()).expect("failed to read config file");
        config = toml::from_slice(&data).unwrap();
    }
    // override from args and ENV
    if !args.pd.is_empty() {
        config.pd.endpoints = args.pd.split(',').map(|x| x.to_owned()).collect();
    }
    if args.cacert.exists() {
        config.security.ca_path = args.cacert.to_str().unwrap().to_owned();
    }
    if args.cert.exists() {
        config.security.cert_path = args.cert.to_str().unwrap().to_owned();
    }
    if args.key.exists() {
        config.security.key_path = args.key.to_str().unwrap().to_owned();
    }
    config.dfs.override_from_env();
    config.skip_resolve_lock = false;
    config
}
