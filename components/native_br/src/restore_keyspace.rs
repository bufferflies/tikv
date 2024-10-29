// Copyright 2023 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    borrow::Cow,
    cmp,
    collections::{HashMap, HashSet},
    default::Default,
    fmt, mem,
    ops::Deref,
    path::{Path, PathBuf},
    sync::{Arc, Mutex, RwLock},
    thread,
    time::Duration,
};

use api_version::ApiV2;
use bytes::Bytes;
use cloud_encryption::{EncryptionKey, MasterKey};
use cloud_server::{RestoreShardResponse, TikvServer};
use file_system::{IoRateLimitMode, IoRateLimiter};
use http::Request;
use hyper::Body;
use itertools::Itertools;
use kvengine::{
    dfs::{self, Dfs, FileType, S3Fs},
    limiter::StoreLimiter,
    table::{BoundedDataSet, DataBound, InnerKey},
    IdAllocator, IdVer, LoadTableFilterFn, ShardMeta, ShardRange, ShardStats, ShardTag,
    ENCRYPTION_KEY, GLOBAL_SHARD_END_KEY, INITIAL_SNAP_VERSION,
};
use kvenginepb as pb;
use kvproto::{metapb, metapb::PeerRole, raft_serverpb::MergeState};
use pd_client::{pd_control::PdControl, PdClient};
use protobuf::Message;
use raft::eraftpb;
use rfengine::RfEngine;
use rfenginepb::ClusterBackupMeta;
use rfstore::store::{
    parse_raft_cmd, rlog, state::RaftState, ApplyMsgs, PdIdAllocator, PeerTag, PreprocessContext,
    RegionIdVer, StoreMsg,
};
use security::SecurityConfig;
use slog_global::{debug, error, info, warn};
use tempdir::TempDir;
use tikv::{config::TikvConfig, storage::mvcc::Key};
use tikv_util::{box_err, box_try, merge_range::MergeRanges, mpsc, time::Instant, HandyRwLock};
use tokio::runtime::Runtime;

use crate::{
    archive::{
        get_cluster_backup_file_and_meta, get_incremental_backup_with_name, get_not_found_files,
        ArchiveReader, StoreMeta,
    },
    common::{
        collect_store_wal_rlog_files, load_peer_raft_state, load_rf_engine_meta, now,
        replay_wal_logs, retain_sst_files, send_request_to_store, RawRegion, RegionMetaGetter,
        StorePeer, TableFile,
    },
    error::{
        Error,
        Error::{KeyspaceInnerKeyOffNotEnabled, MetaNotFound, RetryLimitExceeded},
        Result,
    },
    lock::LockResolver,
    restore::RestoreConfig,
    step,
    tiflash::remove_tiflash_replia_of_keyspace,
};

const WORKING_PATH_PREFIX: &str = "keyspace-restore";
const ZSTD_COMPRESSION_LEVEL: &str = "5"; // The same as ZSTD_COMPRESSION_LEVEL_FOR_REMOTE.

const REPLICAS: usize = 3; // Number of replicas for each region.

const RESOLVE_LOCKS_BATCH_SIZE: usize = 1024;

pub const WAL_CHUNK_INTEGRITY_CHECK_COUNT: usize = 3;

pub const RESTORE_RFENGINE_CONCURRENCY: usize = 3;

#[derive(Clone, Debug, Default)]
pub struct RestoredKeyspace {
    pub keyspace_id: u32,
    pub target_keyspace_id: u32,
    pub ts: u64,
    pub restore_bytes: u64,
    pub tolerated_err: usize,
}

#[derive(Clone, Debug, Default)]
pub struct RestoredSnapshots {
    pub count: u64,
    pub restore_bytes: u64,
}

#[derive(Clone, Debug, Default)]
pub struct RestoredShard {
    pub shard_id: u64,
    pub restore_bytes: u64,
}

/// RestoreStep is the steps of restore process.
///
/// Some steps are performed in retry loop, and will be reported multiple times.
///
/// Must keep the order of enums the same as they happen, then reporters can
/// avoid the progress jumping back and forth.
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub enum RestoreStep {
    Pending,
    Init,
    InstantBackup,
    RemoveTiFlashReplicas,
    LoadBackupMeta,
    ExtractBackupShards,
    ResolveLocks,
    FlushShards,
    TruncateTs,
    SplitRegions,
    AlignRegions,
    RestoreSnapshotsToServers,
    RetainSstFiles,
    Finalize,
}

pub trait ReportRestoreStepTrait {
    fn report_step(&self, step: RestoreStep);
}

pub fn restore_keyspace_with_cfg(
    config: RestoreConfig,
    keyspace_name: &str,
    target_keyspace_name: &str,
    backup_name: &str,
    working_path: Option<&str>,
    s3fs: Arc<S3Fs>,
    pd_client: Arc<dyn PdClient>,
    runtime: &Runtime,
    truncate_ts: Option<u64>,
    reporter: Arc<dyn ReportRestoreStepTrait>,
) -> Result<RestoredKeyspace> {
    let mut pd_control = PdControl::new(config.pd.clone(), pd_client.get_security_mgr())?;
    pd_control.set_retry_timeout(config.timeout_pd_control.0);
    let keyspace_id = {
        let keyspace = runtime.block_on(pd_control.get_keyspace_by_name(keyspace_name))?;
        assert_eq!(keyspace_name, keyspace.name);
        keyspace.id
    };
    step!("Keyspace name {keyspace_name}'s id is {keyspace_id}");

    let target_keyspace_id = if keyspace_name != target_keyspace_name {
        let target_id = {
            let target_keyspace =
                runtime.block_on(pd_control.get_keyspace_by_name(target_keyspace_name))?;
            assert_eq!(target_keyspace_name, target_keyspace.name);
            target_keyspace.id
        };
        step!("Target keyspace name {target_keyspace_name}'s id is {target_id}");
        target_id
    } else {
        reporter.report_step(RestoreStep::RemoveTiFlashReplicas);
        if let Err(e) = runtime.block_on(remove_tiflash_replia_of_keyspace(
            keyspace_id,
            &pd_control,
            pd_client.clone(),
        )) {
            error!("Keyspace {keyspace_id} remove tiflash replica err {:?}", e);
            return Err(e);
        }
        step!("Removed keyspace {keyspace_name} tiflash replica");
        keyspace_id
    };

    restore_keyspace(
        keyspace_id,
        target_keyspace_id,
        backup_name,
        working_path,
        s3fs,
        config,
        pd_client,
        runtime,
        truncate_ts,
        reporter,
    )
}

pub fn restore_keyspace(
    keyspace_id: u32,
    target_keyspace_id: u32,
    backup_name: &str,
    working_path: Option<&str>,
    s3fs: Arc<S3Fs>,
    config: RestoreConfig,
    pd_client: Arc<dyn PdClient>,
    runtime: &Runtime,
    truncate_ts: Option<u64>,
    reporter: Arc<dyn ReportRestoreStepTrait>,
) -> Result<RestoredKeyspace> {
    let working_dir = match working_path {
        Some(p) => TempDir::new_in(p, WORKING_PATH_PREFIX),
        None => TempDir::new(WORKING_PATH_PREFIX),
    }
    .unwrap();
    let working_path = working_dir.path().to_path_buf();

    let inplace_restore = target_keyspace_id == keyspace_id;

    let (keyspace_start, keyspace_end) = ApiV2::get_txn_keyspace_range(keyspace_id);
    let (target_keyspace_start, target_keyspace_end) =
        ApiV2::get_txn_keyspace_range(target_keyspace_id);

    let keyspace_tag = make_keyspace_tag(keyspace_id, target_keyspace_id);

    reporter.report_step(RestoreStep::LoadBackupMeta);
    let (cluster_backup, archive_reader) =
        match get_cluster_backup_file_and_meta(&s3fs, backup_name.to_owned()) {
            Ok((_, backup_meta)) => (backup_meta, None),
            Err(dfs::Error::NoSuchKey(err)) => {
                warn!(
                    "The cluster backup meta {} is not found: {:?}, try the archive reader",
                    backup_name.to_owned(),
                    err
                );
                let backup_file =
                    get_incremental_backup_with_name(s3fs.get_prefix(), backup_name.to_owned());
                let backup_date = backup_file.created_at().date_naive();
                let archive_reader = ArchiveReader::new(s3fs.clone(), &backup_date)?;
                let backup_meta = archive_reader
                    .read_meta_file()
                    .map_err(|_| MetaNotFound(backup_file.id()))?;
                (backup_meta, Some(archive_reader))
            }
            Err(err) => {
                error!("get cluster backup meta {} failed: {:?}", backup_name, err);
                return Err(Error::DfsError(err));
            }
        };
    let truncate_ts = if archive_reader.is_none() {
        truncate_ts
    } else {
        Some(cluster_backup.backup_ts)
    };
    step!(
        "Start restore keyspace {} from backup <{}>, lightweight:{} range:[{},{}), target_range:[{},{}), backup ts:{}, safe ts:{} truncate ts:{:?}",
        keyspace_tag,
        backup_name,
        cluster_backup.is_lightweight,
        log_wrappers::hex_encode_upper(keyspace_start),
        log_wrappers::hex_encode_upper(keyspace_end),
        log_wrappers::hex_encode_upper(&target_keyspace_start),
        log_wrappers::hex_encode_upper(&target_keyspace_end),
        cluster_backup.backup_ts,
        cluster_backup.safe_ts,
        truncate_ts,
    );
    info!("{} restore config", keyspace_tag;
        "skip_resolve_lock" => config.skip_resolve_lock,
        "wal_target_size" => ?config.wal_target_size,
        "new_store_id_delta" => config.new_store_id_delta,
        "timeout_wait_flush" => ?config.timeout_wait_flush,
        "timeout_restore_snapshot" => ?config.timeout_restore_snapshot,
        "timeout_fetch_wal" => ?config.timeout_fetch_wal,
        "tolerate_err" => ?config.tolerate_err,
        "max_retry" => config.max_retry);
    debug!(
        "Keyspace {} get cluster backup meta: {:?}",
        keyspace_tag, cluster_backup
    );

    reporter.report_step(RestoreStep::ExtractBackupShards);
    if truncate_ts.is_some() {
        check_backup_meta_ts(&cluster_backup, truncate_ts.unwrap())?;
    }
    let truncate_ts = truncate_ts.unwrap_or(cluster_backup.backup_ts);
    let mut cluster = BackupCluster::new(
        &cluster_backup,
        working_path,
        pd_client.clone(),
        s3fs.clone(),
        config.clone(),
        keyspace_id,
        target_keyspace_id,
        truncate_ts,
        false,
        archive_reader,
        false,
    )?;
    step!(
        "Keyspace {} restore {} shards from backup",
        keyspace_tag,
        cluster.shards_count()
    );

    // Resolve locks
    // In order to avoid data inconsistency, it is necessary to commit all locks of
    // secondary keys of a committed transaction that are in the backup.
    // If this is not done, uncommitted secondary keys may be dropped when the
    // committed record of primary key is compacted after restoration.
    // This is likely to happen because the timestamp of restored data is generally
    // smaller than the GC safepoint at that time.
    reporter.report_step(RestoreStep::ResolveLocks);
    let lock_resolver = LockResolver::new(
        &keyspace_tag,
        cluster.get_kvengine(),
        RESOLVE_LOCKS_BATCH_SIZE,
        cluster.get_shard_meta_getter(),
    );
    let resolved_locks = runtime.block_on(lock_resolver.resolve_locks())?;
    cluster.add_shards_need_flush(&resolved_locks.resolved_shards);
    step!(
        "Keyspace {keyspace_tag} resolve {} locks / {} lock_txn_files of shards {:?}",
        resolved_locks.total_normal_locks_cnt,
        resolved_locks.total_lock_txn_files_cnt,
        resolved_locks.resolved_shards
    );

    // Trigger initial flush & flush mem-table to S3
    // NOTE: `cluster.shards` is NOT available before `flush_shards` finished.
    reporter.report_step(RestoreStep::FlushShards);
    let flush_cnt = cluster.flush_shards(config.timeout_wait_flush.0)?;
    step!("Keyspace {keyspace_tag} flush {flush_cnt} shards");

    reporter.report_step(RestoreStep::TruncateTs);
    let truncate_ts_cnt = cluster.truncate_ts()?;
    step!(
        "Keyspace {} truncate {} shards to ts {}",
        keyspace_tag,
        truncate_ts_cnt,
        cluster.truncate_ts
    );

    // Check if the backup enable inner_key_offset.
    if !inplace_restore && !cluster.is_inner_key_off_enabled() {
        error!("Keyspace {} inner_key_offset not enabled", keyspace_tag);
        return Err(KeyspaceInnerKeyOffNotEnabled(keyspace_id));
    }

    let inner_split_keys = cluster.get_region_split_keys(&target_keyspace_start);

    let mut success_ranges = MergeRanges::default();
    let mut retry = 0;
    // TODO: split regions for inplace restore.
    let mut need_split_regions = !inplace_restore;
    let mut last_error = None;
    let mut restore_bytes = 0;
    loop {
        if retry > config.max_retry {
            return Err(RetryLimitExceeded(Box::new(last_error.unwrap())));
        }
        retry += 1;

        let mut target_regions = match runtime.block_on(get_target_regions(
            pd_client.as_ref() as &dyn PdClient,
            &target_keyspace_start,
            &target_keyspace_end,
        )) {
            Ok(regions) => regions,
            Err(err) => {
                warn!("get target regions failed: {:?}, retry again", err);
                last_error = Some(err);
                thread::sleep(Duration::from_millis(200));
                continue;
            }
        };

        if need_split_regions {
            reporter.report_step(RestoreStep::SplitRegions);

            // Try split target keyspace regions according to backup keyspace regions if
            // needed.
            // No need check keyspace split in retry.
            split_target_keyspace(
                &inner_split_keys,
                &target_regions,
                &keyspace_tag,
                pd_client.clone(),
                config.timeout_split_regions.0,
                runtime,
            )?;
            need_split_regions = false;

            // Update target regions after split.
            target_regions = match runtime.block_on(get_target_regions(
                pd_client.as_ref(),
                &target_keyspace_start,
                &target_keyspace_end,
            )) {
                Ok(regions) => regions,
                Err(err) => {
                    warn!("get target regions failed: {:?}, retry again", err);
                    last_error = Some(err);
                    thread::sleep(Duration::from_millis(200));
                    continue;
                }
            };
        }

        let target_regions_total_cnt = target_regions.len();
        if !success_ranges.is_empty() {
            target_regions.retain(|r| !success_ranges.covered(&r.raw_start, &r.raw_end));
        }
        step!(
            "Keyspace {} target regions: {} ({} in total)",
            keyspace_tag,
            target_regions.len(),
            target_regions_total_cnt
        );

        // Align target regions.
        reporter.report_step(RestoreStep::AlignRegions);
        let (aligned_regions, trimmed_shards_cnt) = cluster.align_target_regions(target_regions)?;
        step!(
            "Keyspace {} align {} backup shards to {} target regions and trim {} over bound shards",
            keyspace_tag,
            cluster.shards_count(),
            aligned_regions.len(),
            trimmed_shards_cnt
        );

        // Gather sstables for target regions.
        let (target_shards, sstables_cnt) = cluster.gather_all_tables(aligned_regions)?;
        step!(
            "Keyspace {} gather {} SSTables for {} target regions",
            keyspace_tag,
            sstables_cnt,
            target_shards.len(),
        );

        reporter.report_step(RestoreStep::RestoreSnapshotsToServers);
        let snapshots = cluster.generate_snapshots(target_shards);
        let snapshots_count = snapshots.len();
        let ret = restore_snapshots(
            target_keyspace_id,
            runtime,
            pd_client.clone(),
            snapshots,
            &mut success_ranges,
            config.timeout_restore_snapshot.0,
        )?;
        restore_bytes += ret.restore_bytes;
        step!(
            "Keyspace {keyspace_tag} restore {snapshots_count} regions, result: {:?}",
            ret,
        );

        if success_ranges.covered(&target_keyspace_start, &target_keyspace_end) {
            break;
        } else {
            step!("Keyspace {keyspace_tag} restore retry {retry}");
        }
    }

    reporter.report_step(RestoreStep::RetainSstFiles);
    let files = cluster.get_all_shard_files(None);
    match retain_sst_files(files, &s3fs) {
        Err(e) => {
            return Err(box_err!(
                "Keyspace {} fail to retain restored sst files in s3, {:?}",
                keyspace_tag,
                e
            ));
        }
        Ok(file_cnt) => {
            step!("Keyspace {keyspace_tag} retain {file_cnt} files successfully");
        }
    }

    let restore_ret = RestoredKeyspace {
        keyspace_id,
        target_keyspace_id,
        ts: truncate_ts,
        restore_bytes,
        tolerated_err: cluster.tolerated_err(),
    };
    Ok(restore_ret)
}

fn check_backup_meta_ts(meta: &ClusterBackupMeta, truncate_ts: u64) -> Result<()> {
    if truncate_ts < meta.safe_ts || truncate_ts > meta.backup_ts {
        return Err(Error::PitrTsError(
            truncate_ts,
            meta.safe_ts,
            meta.backup_ts,
        ));
    }
    Ok(())
}

fn split_target_keyspace(
    split_keys: &[Vec<u8>],
    target_regions: &[RawRegion],
    keyspace_tag: &str,
    pd_client: Arc<dyn PdClient>,
    timeout: Duration,
    runtime: &Runtime,
) -> Result<()> {
    if split_keys.is_empty() {
        return Ok(());
    }

    // Filter out the already splitted keys.
    let mut split_keys = split_keys.to_vec();
    split_keys.retain(|key| {
        target_regions
            .iter()
            .all(|region| key.as_slice() != region.get_start_key())
    });
    step!(
        "Keyspace {} split target regions, split keys count {}",
        keyspace_tag,
        split_keys.len(),
    );
    let enc_split_keys = split_keys
        .iter()
        .map(|key| Key::from_raw(key.as_slice()).into_encoded())
        .collect::<Vec<_>>();
    if !enc_split_keys.is_empty() {
        runtime.block_on(pd_client.split_regions_with_retry(enc_split_keys, timeout))?;
    }

    Ok(())
}

#[derive(Default, Clone)]
pub struct BackupShard {
    pub region_id: u64,
    pub store_id: u64,
    pub peer_id: u64,
    pub need_flush: bool,
    pub meta: ShardMeta,
    pub(crate) raw_meta: Option<pb::ChangeSet>, // used for kv engine recovery only.
    raft_state: RaftState,
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
            .field("raft_state", &self.raft_state)
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
            && self.raft_state == other.raft_state
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
        &self.meta.range.outer_start
    }

    pub fn end(&self) -> &[u8] {
        &self.meta.range.outer_end
    }

    pub fn inner_start(&self) -> InnerKey<'_> {
        self.meta.range.inner_start()
    }

    pub fn inner_end(&self) -> InnerKey<'_> {
        self.meta.range.inner_end()
    }

    pub fn inner_key_off(&self) -> usize {
        self.meta.range.inner_key_off
    }

    pub fn table_version(&self) -> u64 {
        self.meta.base_version + self.meta.data_sequence
    }
}

impl BoundedDataSet for BackupShard {
    fn data_bound(&self) -> DataBound<'_> {
        self.meta.range.data_bound()
    }
}

#[derive(Debug, PartialEq)]
struct AlignedRegion {
    target_region: RawRegion,
    backup_shards_id: Vec<u64>,
}

pub struct BackupCluster {
    tag: String,
    path: PathBuf,
    pd_client: Arc<dyn PdClient>,
    dfs: Arc<S3Fs>,
    security_conf: SecurityConfig,
    master_key: MasterKey,
    keyspace_id: u32,
    target_keyspace_id: u32,
    keyspace_start: Vec<u8>,
    keyspace_end: Vec<u8>,
    truncate_ts: u64,
    archive_reader: Option<ArchiveReader>,

    // store_id -> rf_engine.
    raft_engines: HashMap<u64, RfEngine>,
    // store_id -> store_config.
    store_configs: HashMap<u64, TikvConfig>,
    // must be `Some` after `new()`.
    kv_engine: Option<kvengine::Engine>,

    // region_id -> shard.
    shards: HashMap<u64, BackupShard>,
    // region_id -> raw_meta, used for kv_engine recover.
    raw_metas: HashMap<u64, pb::ChangeSet>,
    // store_id -> Vec<shard_id>.
    store_shards: HashMap<u64, Vec<u64>>,
    // shard_id -> StorePeer{store_id, peer_id}. Used to resolve locks.
    shards_store_map: Option<Arc<HashMap<u64 /* shard_id */, StorePeer>>>,
    // shard_id sorted by start_key.
    sorted_shards: Vec<u64>,
    // Set<shard_id>.
    shards_need_flush: HashSet<u64>,
    // Set<shard_id>.
    shards_need_truncate: HashSet<u64>,

    meta_applier: Option<Arc<MetaApplier>>,

    meta_sender: Option<mpsc::Sender<StoreMsg>>,

    tolerated_err: usize,

    // Wrap by `Option` to detect use before filled.
    txn_chunk_ids_in_wal: Option<Vec<u64>>,

    // `false` for restoration.
    // `true` for "check_table" to read data from backup directly.
    load_all_tables: bool,
    // NOTE: New members need to check if need to clear in reset_keyspace.
    id_allocator: Arc<dyn IdAllocator>,
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
        restore_conf: RestoreConfig,
        keyspace_id: u32,
        target_keyspace_id: u32,
        truncate_ts: u64,
        archiving: bool,
        archive_reader: Option<ArchiveReader>,
        load_all_tables: bool, // `true` for "check_table" ONLY.
    ) -> Result<BackupCluster> {
        let (keyspace_start, keyspace_end) = if archiving {
            (Vec::default(), GLOBAL_SHARD_END_KEY.to_vec())
        } else {
            ApiV2::get_txn_keyspace_range(keyspace_id)
        };
        let security_conf = restore_conf.security.clone();
        let master_key = dfs.get_runtime().block_on(security_conf.new_master_key());
        let mut cluster = Self {
            tag: make_keyspace_tag(keyspace_id, target_keyspace_id),
            path,
            pd_client: pd_client.clone(),
            dfs,
            security_conf,
            master_key,
            keyspace_id,
            target_keyspace_id,
            keyspace_start,
            keyspace_end,
            truncate_ts,
            archive_reader,
            shards: Default::default(),
            raw_metas: Default::default(),
            raft_engines: Default::default(),
            store_configs: Default::default(),
            kv_engine: Default::default(),
            store_shards: Default::default(),
            shards_store_map: Default::default(),
            sorted_shards: Default::default(),
            shards_need_flush: Default::default(),
            shards_need_truncate: Default::default(),
            meta_applier: None,
            meta_sender: None,
            tolerated_err: 0,
            txn_chunk_ids_in_wal: None,
            load_all_tables,
            id_allocator: Arc::new(PdIdAllocator::new(pd_client)),
        };

        let mut store_configs = HashMap::with_capacity(cluster_meta.stores.len());

        // If the cluster backup already has tolerate error, we can't tolerate more
        // errors in restoration.
        let mut tolerate_err = if cluster_meta.get_tolerated_err() > 0 {
            0
        } else {
            restore_conf.tolerate_err
        };
        let fetch_wal_timeout = restore_conf.timeout_fetch_wal.0;

        let is_error_can_tolerate = |err: &Error| -> bool {
            if matches!(err, Error::RfengineHttpError(_)) {
                true
            } else if !restore_conf.strict_tolerate {
                crate::metrics::NATIVE_BR_RESTORE_ERROR
                    .with_label_values(&["setup_raft_engine"])
                    .inc();
                true
            } else {
                false
            }
        };
        let mut raft_engines: HashMap<u64, RfEngine> = Default::default();
        let mut cluster_tolerated_err = 0;
        let mut recv_restore_rfengine =
            |result_rx: &mpsc::Receiver<Result<(u64, TikvConfig, RfEngine)>>| -> Result<()> {
                // We can tolerate one store failure for lightweight restoration during fetch
                // latest wal chunk from store.
                match result_rx.recv().unwrap() {
                    Err(err) if is_error_can_tolerate(&err) => {
                        if tolerate_err == 0 {
                            return Err(err);
                        }
                        warn!(
                            "Keyspace {} setup raft engine failed, tolerate it: {:?}",
                            cluster.tag(),
                            err
                        );
                        tolerate_err -= 1;
                        cluster_tolerated_err += 1;
                    }
                    Err(err) => return Err(err),
                    Ok((store_id, store_config, rf_engine)) => {
                        store_configs.insert(store_id, store_config);
                        raft_engines.insert(store_id, rf_engine);
                        tikv_util::info!(
                            "Keyspace {} setup raft engine for store {} done",
                            cluster.tag(),
                            store_id
                        );
                    }
                }
                Ok(())
            };
        let (result_tx, result_rx) = tikv_util::mpsc::bounded(cluster_meta.stores.len());
        let mut msg_count = 0;

        for store in &cluster_meta.stores {
            let store_id = store.get_store_id();
            let store_config = cluster.generate_store_config(store_id);

            let tx = result_tx.clone();
            let cluster_backup_meta = cluster_meta.clone();
            let pd_client = cluster.pd_client.clone();
            let s3fs = cluster.dfs.clone();
            let keyspace_tag = cluster.tag().to_string();
            let archive_store_meta = if let Some(archive_reader) = &cluster.archive_reader {
                let store_meta = archive_reader.get_store_wal_rlog_meta(store_id)?;
                Some((archive_reader.get_start_date(), store_meta))
            } else {
                None
            };
            std::thread::spawn(move || {
                let res = BackupCluster::setup_raft_engine(
                    store_id,
                    &cluster_backup_meta,
                    &store_config,
                    pd_client,
                    s3fs,
                    fetch_wal_timeout,
                    archiving,
                    archive_store_meta,
                );
                if res.is_err() {
                    warn!(
                        "Keyspace {} setup raft engine for store {} failed",
                        keyspace_tag, store_id
                    );
                }
                let _ = tx.send(res.map(|rf_engine| (store_id, store_config, rf_engine)));
            });

            if msg_count < RESTORE_RFENGINE_CONCURRENCY {
                msg_count += 1;
            } else {
                recv_restore_rfengine(&result_rx)?;
            }
        }
        for _ in 0..msg_count {
            recv_restore_rfengine(&result_rx)?;
        }
        cluster.raft_engines = raft_engines;
        cluster.store_configs = store_configs;
        cluster.tolerated_err = cluster_tolerated_err;
        cluster.load_shards()?;
        cluster.collect_txn_chunks_in_wal()?;

        if archiving {
            return Ok(cluster);
        }
        cluster.check_all_shard_files()?;

        for store_id in cluster.get_all_stores_id() {
            cluster.setup_kv_engine(store_id)?;
        }
        Ok(cluster)
    }

    pub fn tag(&self) -> &str {
        &self.tag
    }

    pub fn setup_raft_engine(
        store_id: u64,
        cluster_backup: &ClusterBackupMeta,
        conf: &TikvConfig,
        pd_client: Arc<dyn PdClient>,
        dfs: Arc<S3Fs>,
        fetch_wal_timeout: Duration,
        archiving: bool,
        archive_store_meta: Option<(String, StoreMeta)>, // archive date, archive store meta
    ) -> Result<RfEngine> {
        let wals = if cluster_backup.is_lightweight {
            let wal_rlog_files = if let Some((date, store_meta)) = archive_store_meta {
                ArchiveReader::read_store_wal_rlog_files(&dfs, date, store_meta)?
            } else {
                collect_store_wal_rlog_files(dfs.clone(), cluster_backup, store_id)?
            };
            rfengine::lightweight_restore(
                Path::new(&conf.raft_store.raftdb_path),
                wal_rlog_files.snap_epoch,
                wal_rlog_files.snap_meta,
                wal_rlog_files.snap_rlog,
            )
            .map_err(|x| Error::RfEngine(x))?;
            Some(wal_rlog_files.wals)
        } else {
            rfengine::restore(
                dfs.clone(),
                cluster_backup,
                store_id,
                Path::new(&conf.raft_store.raftdb_path),
                None, // TODO: pass `Some(keyspace_id)` in.
            );
            None
        };

        let rf_engine = TikvServer::init_raft_engine(conf)?;

        if let Some(wals) = wals {
            // When archiving, the dfs should have complete wal chunks.
            let complete_wal_chunks = archiving;
            replay_wal_logs(
                pd_client.clone(),
                dfs,
                store_id,
                cluster_backup,
                &rf_engine,
                complete_wal_chunks,
                false,
                fetch_wal_timeout,
                wals,
            )?;
        }
        Ok(rf_engine)
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

    fn get_shards_need_flush_and_truncate(&self) -> Vec<u64> {
        self.shards_need_flush
            .union(&self.shards_need_truncate)
            .cloned()
            .collect()
    }

    fn setup_kv_engine(&mut self, store_id: u64) -> Result<()> {
        let conf = self.store_configs.get(&store_id).unwrap();
        let rf_engine = self.raft_engines.get(&store_id).unwrap();
        let recoverer = rfstore::store::RecoverHandler::new(rf_engine.clone());

        if self.kv_engine.is_none() {
            let io_rate_limiter =
                Arc::new(IoRateLimiter::new(IoRateLimitMode::WriteOnly, true, true));
            io_rate_limiter
                .set_io_rate_limit(conf.storage.io_rate_limit.max_bytes_per_sec.0 as usize);
            // TODO: enable flow control to avoid OOM during recovery.
            let store_limiter = Arc::new(StoreLimiter::dummy());

            let mut meta_iter = MetaIterator::new(store_id, Vec::new(), HashMap::new());
            let (kv_engine, sender, receiver) = TikvServer::init_kv_engine(
                self.pd_client.clone(),
                conf,
                self.dfs.clone(),
                io_rate_limiter,
                store_limiter,
                &mut meta_iter,
                recoverer.clone(),
                true,
                self.master_key.clone(),
                self.pd_client.get_security_mgr(),
            )?;

            let encryption_key = self.get_keyspace_exported_encryption_key().map(|exported| {
                kv_engine
                    .get_master_key()
                    .decrypt_encryption_key(&exported)
                    .unwrap()
            });

            // Move shards to meta_applier to apply meta change in `flush_mem_table`.
            // Move back after flush finished.
            let shards = mem::take(&mut self.shards);
            let meta_applier = Arc::new(MetaApplier::new(
                self.keyspace_id,
                kv_engine.clone(),
                encryption_key,
                shards,
                receiver,
            ));
            self.meta_applier = Some(meta_applier.clone());
            self.meta_sender = Some(sender);
            thread::spawn(move || {
                meta_applier.run();
            });

            self.kv_engine = Some(kv_engine);
        }

        // Load L0 & LOCK_CF of all shards for scanning locks.
        // TODO: L0 & LOCK_CF with `max_ts` < `safe_ts` can be skipped.
        // (Currently there is no `max_ts` for LOCK_CF of L0)
        let store_shards = self.store_shards.get(&store_id).unwrap().to_owned();
        let raw_metas = self.take_raw_metas(store_id);
        let kv_engine = self.kv_engine.as_mut().unwrap();
        if !store_shards.is_empty() {
            let mut meta_iter =
                MetaIterator::new(kv_engine.get_engine_id(), store_shards, raw_metas);
            let (metas, _) = kvengine::EngineCore::read_meta(&mut meta_iter)?;
            info!(
                "Keyspace {} kv_engine load {} shards in restore keyspace",
                self.keyspace_id,
                metas.len()
            );

            let table_filter: Option<LoadTableFilterFn> = if self.load_all_tables {
                None
            } else {
                // Load the following tables from DFS:
                // 1. Tables of shards need truncate, to get the `max_ts`.
                // 2. Tables has locks (L0 or LOCK_CF) to replay commit requests.
                let shard_need_truncate = self.shards_need_truncate.clone();
                let table_filter = move |shard_id: u64, tb: &kvengine::FileMeta| -> bool {
                    shard_need_truncate.contains(&shard_id) || tb.has_locks()
                };
                Some(Arc::new(table_filter))
            };
            kv_engine.load_shards(metas, recoverer, table_filter)?;
        }
        Ok(())
    }

    fn stop_engines(&mut self) {
        let raft_engines = mem::take(&mut self.raft_engines);
        for rf in raft_engines.into_values() {
            rf.stop_worker(true);
            drop(rf);
        }
        if let Some(kv_engine) = self.kv_engine.take() {
            // `self.kv_engine` will be None when `BackupCluster` fail to initialize.
            kv_engine.close();
            drop(kv_engine);
        }
        // Receiver cannot get err msg even close_sender, send Stop msg instead.
        if let Some(sender) = &self.meta_sender {
            sender.send(StoreMsg::Stop).unwrap();
        }
    }

    fn generate_store_config(&self, store_id: u64) -> TikvConfig {
        let (store_path, rf_engine_path) = (
            self.path.join(store_id.to_string()),
            self.path.join(store_id.to_string()).join("raft"),
        );

        let mut config = TikvConfig::default();
        config.storage.data_dir = store_path.to_str().unwrap().to_string();
        config.raft_store.raftdb_path = rf_engine_path.to_str().unwrap().to_string();
        config.dfs.zstd_compression_level = ZSTD_COMPRESSION_LEVEL.to_string();
        config.rocksdb.max_background_jobs = 2;
        config.rocksdb.max_sub_compactions = 1;
        config.security = self.security_conf.clone();

        config.raft_engine.enable = false;
        config.rfengine.lightweight_backup = false;

        TikvServer::init_config(config).get_current()
    }

    fn collect_keyspace_shards(
        store_id: u64,
        rf: &RfEngine,
        keyspace_start: &[u8],
        keyspace_end: &[u8],
    ) -> Result<Vec<BackupShard>> {
        let region_peers = rf.get_region_peer_map();
        let mut prefix_shards = vec![];
        let mut first_inner_key_off = None;
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
            if keyspace_start <= snap.get_outer_start() && snap.get_outer_end() <= keyspace_end {
                if let Some(x) = first_inner_key_off {
                    assert_eq!(x, snap.inner_key_off)
                } else {
                    first_inner_key_off = Some(snap.inner_key_off);
                }

                let shard = Self::create_backup_shard(rf, store_id, region_id, peer_id, meta)?;
                prefix_shards.push(shard);
            }
        }
        Ok(prefix_shards)
    }

    fn collect_full_shards(store_id: u64, rf: &RfEngine) -> Result<Vec<BackupShard>> {
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

            let shard = Self::create_backup_shard(rf, store_id, region_id, peer_id, meta)?;
            prefix_shards.push(shard);
        }
        Ok(prefix_shards)
    }

    pub fn is_full_range(&self) -> bool {
        self.keyspace_start.is_empty() && self.keyspace_end.eq(GLOBAL_SHARD_END_KEY)
    }

    fn create_backup_shard(
        rf: &RfEngine,
        store_id: u64,
        region_id: u64,
        peer_id: u64,
        mut meta: kvenginepb::ChangeSet,
    ) -> Result<BackupShard> {
        // Fix backups before pull/1732 in which the `data_sequence` in meta is not
        // correct.
        if meta.has_parent() {
            meta.mut_snapshot().data_sequence = meta.sequence;
        }

        let snap = meta.get_snapshot();
        let raft_last_index = rf.get_last_index(peer_id);
        let need_flush_mem_table = raft_last_index.map_or(false, |last_idx| {
            assert!(last_idx >= snap.get_data_sequence());
            last_idx > snap.get_data_sequence()
        });
        let need_initial_flush = meta.has_parent();

        let raft_state = load_peer_raft_state(rf, peer_id, meta.shard_ver).ok_or_else(|| -> Error {
            let mut states = vec![];
            rf.iterate_peer_states(peer_id, false, |k, v| {
                states.push((k.to_vec(), v.to_vec()));
            });
            error!(
                "failed to load peer raft state, store_id: {}, region_id: {}, peer_id: {}, state: {:?}",
                store_id, region_id, peer_id, states
            );

            box_err!("failed to load peer raft state, store_id: {}, region_id: {} peer_id: {}",
                store_id, region_id, peer_id)
        })?;

        let shard = BackupShard {
            region_id,
            store_id,
            peer_id,
            need_flush: need_flush_mem_table || need_initial_flush,
            meta: ShardMeta::new(rf.get_engine_id(), &meta),
            raw_meta: Some(meta),
            raft_state,
        };
        debug!(
            "create_backup_shard: raft_last_index {:?}, {:?}",
            raft_last_index, shard
        );
        Ok(shard)
    }

    fn load_shards(&mut self) -> Result<()> {
        // Will need to retry when `get_leader_shards_and_preprocess` produce new shards
        // after preprocess.
        let mut leader_shards = HashMap::new();
        let mut retry = 0_usize;
        let is_full_range = self.is_full_range();
        loop {
            let mut all_shards = HashMap::new();
            for (&store_id, rf_engine) in &self.raft_engines {
                let store_shards = if is_full_range {
                    Self::collect_full_shards(store_id, rf_engine)?
                } else {
                    Self::collect_keyspace_shards(
                        store_id,
                        rf_engine,
                        &self.keyspace_start,
                        &self.keyspace_end,
                    )?
                };
                all_shards.insert(store_id, store_shards);
            }

            // Check whether backup is empty (for this keyspace).
            // Empty backup should be invalid or before the creation of this
            // keyspace, so the restore is not allowed.
            if all_shards.iter().all(|(_, shards)| shards.is_empty()) {
                return Err(Error::BackupEmptyForKeyspace(self.keyspace_id));
            }

            match self.get_leader_shards_and_preprocess(all_shards)? {
                (_, true) => {
                    if retry > 20 {
                        // Should never happen. Just a safe guard to avoid infinite loop.
                        return Err(box_err!(
                            "get_leader_shards_and_preprocess retry too many times"
                        ));
                    }
                    retry += 1;
                    continue;
                }
                (shards, false) => {
                    let _ = mem::replace(&mut leader_shards, shards);
                    break;
                }
            }
        }

        let mut sorted_shards_id = Self::handle_overlapping_shards(&mut leader_shards)?;
        info!(
            "leader shard cnt {}, sorted shards cnt {}",
            leader_shards.len(),
            sorted_shards_id.len()
        );
        self.shards = mem::take(&mut leader_shards);
        self.sorted_shards = mem::take(&mut sorted_shards_id);
        debug!(
            "Keyspace {} BackupCluster.load_shards: {:?}",
            self.tag(),
            self.shards
        );
        debug!(
            "Keyspace {} BackupCluster.sorted_shards: {:?}",
            self.tag(),
            self.sorted_shards
        );

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

        self.shards_store_map = Some(Arc::new(
            self.shards
                .iter()
                .map(|(&shard_id, shard)| {
                    (
                        shard_id,
                        StorePeer {
                            store_id: shard.store_id,
                            peer_id: shard.peer_id,
                        },
                    )
                })
                .collect(),
        ));
        self.shards_need_flush.extend(
            self.shards
                .iter()
                .filter(|(_, shard)| shard.need_flush)
                .map(|(&shard_id, _)| shard_id),
        );
        self.shards_need_truncate.extend(
            self.shards
                .iter()
                .filter(|(_, shard)| shard.meta.max_ts >= self.truncate_ts)
                .map(|(&shard_id, _)| shard_id),
        );

        self.verify_shards()
    }

    // Should be called after load_shard & collect_txn_chunks_in_wal.
    pub fn get_all_shard_files(&self, skip_shards: Option<HashSet<u64>>) -> Vec<TableFile> {
        let mut files = vec![];
        for shard in self.shards.values() {
            if let Some(skip_shards) = &skip_shards {
                if skip_shards.contains(&shard.region_id) {
                    continue;
                }
            }
            files.extend(shard.meta.all_files().iter().map(|(&id, fm)| TableFile {
                id,
                ftype: fm.file_type,
            }));
            files.extend(
                shard
                    .meta
                    .all_txn_chunk_ids()
                    .into_iter()
                    .map(|id| TableFile {
                        id,
                        ftype: FileType::TxnChunk,
                    }),
            );
        }
        files.extend(
            self.txn_chunk_ids_in_wal
                .as_ref()
                .unwrap()
                .iter()
                .map(|&id| TableFile {
                    id,
                    ftype: FileType::TxnChunk,
                }),
        );
        files
    }

    // Should be called after load_shard and before setup_kv_engine.
    pub fn check_all_shard_files(&self) -> Result<()> {
        if let Some(archive_reader) = &self.archive_reader {
            let files = self.get_all_shard_files(None);
            let not_found_files = get_not_found_files(&self.dfs, files)?;
            archive_reader.restore_files(not_found_files)?;
        }
        Ok(())
    }

    fn get_leader_shards_and_preprocess(
        &self,
        all_shards: HashMap<u64, Vec<BackupShard>>,
    ) -> Result<(
        HashMap<u64, BackupShard>, // leader_shards, shard_id -> BackupShard
        bool,                      // has_new_peer
    )> {
        let shards_cnt = all_shards.values().map(|x| x.len()).sum::<usize>() / REPLICAS;
        // shard_peers: shard_id -> Vec<(store_id, BackupShard)>
        let mut shard_peers: HashMap<u64, Vec<BackupShard>> = HashMap::with_capacity(shards_cnt);

        for shard in all_shards.into_values().flatten() {
            shard_peers.entry(shard.region_id).or_default().push(shard);
        }
        let mut leader_shards = HashMap::with_capacity(shard_peers.len());
        for (&shard_id, peers) in shard_peers.iter_mut() {
            // Since we enabled raft pre-vote, the correct leader must have the max term and
            // then max last index.
            // Also sort by store_id (descending) to get peer with smallest store id as
            // leader, to make the result determined and improve efficiency for
            // retry.
            peers.sort_by(|a, b| {
                let term_last_a = (
                    a.raft_state.get_hard_state().get_term(),
                    a.raft_state.get_last_index(),
                    !a.store_id,
                );
                let term_last_b = (
                    b.raft_state.get_hard_state().get_term(),
                    b.raft_state.get_last_index(),
                    !b.store_id,
                );
                term_last_a.cmp(&term_last_b)
            });
            let mut leader = peers.pop().unwrap();
            if leader.raft_state.get_commit() < leader.raft_state.get_last_index() {
                // If we tolerate error, the actual leader maybe missing, it's possible that the
                // most up-to-date follower's last_index is already committed.
                // It's possible that the original cluster eventually not committed to
                // the last index, and the last index contains commit primary key timestamp
                // lower than backup ts. But the probability is very low and the
                // data is still consistent.
                let mut hard_state = leader.raft_state.get_hard_state();
                hard_state.set_commit(leader.raft_state.get_last_index());
                leader.raft_state.set_hard_state(&hard_state);
            }
            leader_shards.insert(shard_id, leader);
        }

        let mut has_new_peer = false;
        for shard in leader_shards.values_mut() {
            // Preprocess if needed.
            if shard.raft_state.get_commit() > shard.raft_state.get_last_preprocessed_index() {
                info!(
                    "Keyspace {} shard {} need preprocess, commit {}, last_preprocessed_index {}",
                    self.tag(),
                    shard.region_id,
                    shard.raft_state.get_commit(),
                    shard.raft_state.get_last_preprocessed_index()
                );
                let rf_engine = self.raft_engines.get(&shard.store_id).unwrap();
                if self.preprocess_shard(rf_engine, shard)? {
                    has_new_peer = true;
                }
            }
        }
        Ok((leader_shards, has_new_peer))
    }

    /// Handle overlapping shards.
    /// Shards overlap will happen when some followers had not finished split or
    /// merge.
    fn handle_overlapping_shards(
        leader_shards: &mut HashMap<u64, BackupShard>,
    ) -> Result<
        Vec<u64>, // Vec<shard_id> sorted by BackupShard.start()
    > {
        let mut sorted_shards: Vec<u64> = leader_shards
            .values()
            .sorted_by(|a, b| a.start().cmp(b.start()))
            .map(|x| x.region_id)
            .collect();

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

        Ok(sorted_shards)
    }

    fn preprocess_shard(
        &self,
        rf_engine: &RfEngine,
        old_shard: &mut BackupShard,
    ) -> Result<bool /* has_new_peer */> {
        let mut preprocessor = PeerPreprocessor::new(rf_engine, old_shard);

        // Get raft logs.
        let low_idx = preprocessor.preprocessed_index + 1;
        let high_idx = old_shard.raft_state.get_commit() + 1;
        info!(
            "Keyspace {} shard {} preprocess Raft log [{}, {}), raft_state {:?}",
            self.tag(),
            old_shard.tag(),
            low_idx,
            high_idx,
            preprocessor.raft_state,
        );

        let mut entries = Vec::with_capacity((high_idx.saturating_sub(low_idx)) as usize);
        let peer_id = old_shard.peer_id;
        rf_engine
            .fetch_raft_entries_to(peer_id, low_idx, high_idx, None, &mut entries)
            .map_err(|e| -> Error {
                let stats = rf_engine.get_peer_stats(peer_id);
                let truncated_state = rfstore::store::load_raft_truncated_state(rf_engine, peer_id);
                box_err!(
                    "{} entries unavailable err: {:?}, stats {:?}, truncated_state: {:?}, low: {}, high {}",
                    old_shard.tag(),
                    e,
                    stats,
                    truncated_state,
                    low_idx,
                    high_idx
                )
            })?;

        // Prepare context.
        let mut raft_wb = rfengine::WriteBatch::new();
        let mut remove_dependents = Vec::new();
        let mut apply_msgs = ApplyMsgs::default();
        let raft_cfg = rfstore::store::Config::default();
        let mut destroying = HashSet::new();
        let mut ctx = PreprocessContext {
            store_id: old_shard.store_id,
            kv: None,
            raft: rf_engine,
            raft_wb: &mut raft_wb,
            remove_dependents: &mut remove_dependents,
            apply_msgs: &mut apply_msgs,
            cfg: &raft_cfg,
            router: None,
            destroying: &mut destroying,
        };

        // Preprocess
        let mut preprocess_ref = preprocessor.as_ref();
        for entry in &entries {
            preprocess_ref.preprocess_committed_entry(&mut ctx, entry);
        }

        // Apply to rf_engine
        preprocess_ref
            .raft_state
            .set_last_preprocessed_index(*preprocess_ref.preprocessed_index);
        preprocess_ref.write_raft_state(&mut ctx); // update `last_preprocessed_index` to rf_engine.
        rf_engine.apply(ctx.raft_wb); // Do not need persist.

        // Get new shard.
        // TODO: In-place update `old_shard`, other than get a new one from rf_engine.
        let meta = load_rf_engine_meta(rf_engine, peer_id).unwrap();
        let new_shard = Self::create_backup_shard(
            rf_engine,
            old_shard.store_id,
            old_shard.region_id,
            peer_id,
            meta,
        )?;
        info!(
            "Keyspace {} preprocessed shard {}, old {:?}, new {:?}",
            self.tag(),
            old_shard.tag(),
            old_shard,
            new_shard
        );

        // Verify.
        let old_commit = old_shard.raft_state.get_commit();
        let new_commit = new_shard.raft_state.get_commit();
        let new_last_preprocessed_index = new_shard.raft_state.get_last_preprocessed_index();
        assert!(
            old_commit == new_commit && new_commit == new_last_preprocessed_index,
            "Keyspace {} shard {} preprocess: commit_index(old:{}, new:{})/last_preprocessed_index({}) not match",
            self.tag(),
            old_shard.tag(),
            old_commit,
            new_commit,
            new_last_preprocessed_index,
        );

        // When new shard has smaller range, there must be new peer due to split.
        // Raise error to retry from `load_shards` for simplicity, as it is a very rare
        // case.
        // Do not need to care about region merge, and `handle_overlapping_shards` will
        // handle the overlapping caused by merge.
        let has_new_peer =
            old_shard.start() < new_shard.start() || new_shard.end() < old_shard.end();
        if has_new_peer {
            info!(
                "{} split detected, retry load_shards, {} old_shard {:?}, new_shard {:?}",
                self.tag,
                old_shard.tag(),
                old_shard,
                new_shard
            );
        }

        let _ = mem::replace(old_shard, new_shard);
        Ok(has_new_peer)
    }

    fn verify_shards(&self) -> Result<()> {
        debug_assert!(!self.sorted_shards.is_empty());
        if self.is_full_range() {
            self.verify_full_shards();
            return Ok(());
        }

        let first_shard = self.get_sorted_shard(0);
        let last_shard = self.get_sorted_shard(self.sorted_shards.len() - 1);
        if first_shard.start() != self.keyspace_start {
            return Err(box_err!(
                "start key of first region not match: {:?}",
                first_shard
            ));
        }
        if last_shard.end() != self.keyspace_end {
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

    fn verify_full_shards(&self) {
        let first_shard = self.get_sorted_shard(0);
        let last_shard = self.get_sorted_shard(self.sorted_shards.len() - 1);
        if first_shard.start() != self.keyspace_start {
            warn!("start key of first region not match: {:?}", first_shard);
        }
        if last_shard.end() != self.keyspace_end {
            warn!("end key of last region not match: {:?}", last_shard);
        }

        for shards_id in self.sorted_shards.windows(2) {
            let left = self.get_shard(shards_id[0]).unwrap();
            let right = self.get_shard(shards_id[1]).unwrap();
            if left.end() != right.start() {
                warn!("region boundary not match: {:?}, {:?}", left, right);
            }
        }
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

    pub fn is_inner_key_off_enabled(&self) -> bool {
        self.get_sorted_shard(0).meta.range.inner_key_off != 0
    }

    pub fn get_shard_metas_before_flush(&self) -> Vec<ShardMeta> {
        let mut shards = Vec::with_capacity(self.sorted_shards.len());
        let guard = self.meta_applier.as_ref().unwrap().shards.read().unwrap();
        let backup_shards = guard.as_ref().unwrap();
        for &id in &self.sorted_shards {
            let backup_shard = backup_shards.get(&id).unwrap();
            shards.push(backup_shard.meta.clone());
        }
        shards
    }

    pub fn get_kvengine(&self) -> kvengine::Engine {
        self.kv_engine.as_ref().unwrap().clone()
    }

    pub fn get_rfengine(&self, store_id: u64) -> Option<RfEngine> {
        self.raft_engines.get(&store_id).cloned()
    }

    pub fn get_shard_meta_getter(&self) -> RegionMetaGetter {
        RegionMetaGetter {
            shard_store_map: self.shards_store_map.as_ref().unwrap().clone(),
            raft_engines: Arc::new(self.raft_engines.clone()),
        }
    }

    // The returned id of stores are sorted.
    // This will help to reuse downloaded table files when re-running the
    // restoration process.
    // See `BackupCluster::setup_kv_engine`.
    fn get_all_stores_id(&self) -> Vec<u64> {
        let mut stores_id: Vec<_> = self.store_shards.keys().copied().collect();
        stores_id.sort();
        stores_id
    }

    pub fn add_shards_need_flush(&mut self, shard_ids: &[u64]) {
        self.shards_need_flush.extend(shard_ids);
    }

    pub fn flush_shards(
        &mut self,
        timeout: Duration,
    ) -> Result<usize /* number of flushed shards */> {
        let kv_engine = self.kv_engine.as_ref().unwrap();
        // TODO: run in parallel.
        for &shard_id in &self.shards_need_flush {
            let engine_shard = kv_engine
                .get_shard(shard_id)
                .unwrap_or_else(|| panic!("shard not found: {}", shard_id));
            kv_engine.flush_shard_for_restore(&engine_shard)?;
        }

        self.check_flushed(timeout)?;

        let errors = self.meta_applier.as_ref().unwrap().take_errors();
        if !errors.is_empty() {
            return Err(box_err!(
                "{} meta_applier meet errors: {:?}",
                self.tag,
                errors
            ));
        }

        // Take back from meta_applier.
        let mut shards = self.meta_applier.as_ref().unwrap().take_shards().unwrap();
        self.shards = mem::take(&mut shards);

        Ok(self.shards_need_flush.len())
    }

    fn check_flushed(&self, timeout: Duration) -> Result<()> {
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
        error!(
            "Keyspace {} wait_for_mem_table_flush timeout, stats: {:?}",
            self.keyspace_id, stats
        );
        Err(box_err!("wait_for_mem_table_flush timeout"))
    }

    fn truncate_ts(&mut self) -> Result<usize> {
        let mut truncate_cnt = 0_usize;
        for shard_id in self.get_shards_need_flush_and_truncate() {
            truncate_cnt += self.truncate_shard_ts(shard_id)? as usize;
        }
        Ok(truncate_cnt)
    }

    // NOTE: `sorted_backup_shards_id` & `sorted_target_regions` must be sorted by
    // start_key.
    fn align_target_regions_impl(
        sorted_backup_shards_id: &[u64],
        backup_shards: &HashMap<u64, BackupShard>,
        sorted_target_regions: Vec<RawRegion>,
        inner_key_off: usize,
    ) -> Vec<AlignedRegion> {
        let mut aligned_regions = Vec::with_capacity(sorted_target_regions.len());
        let mut idx = 0;

        // NOTE: use inner_key to check overlapping, target_regions may belongs to
        // a new keyspace.
        let key_in_shard = |key: &[u8], shard: &BackupShard| {
            let inner_key = InnerKey::from_outer_key(key, inner_key_off);
            shard.data_bound().overlap_key(inner_key)
        };

        let get_backup_shard =
            |idx: usize| backup_shards.get(&sorted_backup_shards_id[idx]).unwrap();

        for target_region in sorted_target_regions {
            // Check overlapping with previous backup_region.
            if idx > 0 && key_in_shard(target_region.get_start_key(), get_backup_shard(idx - 1)) {
                idx -= 1;
            }
            let target_inner_end =
                InnerKey::from_outer_end_key(target_region.get_end_key(), inner_key_off);
            let mut aligned_shards_id = vec![];
            while idx < sorted_backup_shards_id.len()
                && get_backup_shard(idx).inner_start() < target_inner_end
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

    fn align_target_regions(
        &mut self,
        target_regions: Vec<RawRegion>,
    ) -> Result<(
        Vec<AlignedRegion>,
        usize, // number of trimmed shards
    )> {
        let inner_key_off = self.inner_key_off();
        let aligned_regions = Self::align_target_regions_impl(
            &self.sorted_shards,
            &self.shards,
            target_regions,
            inner_key_off,
        );
        let trimmed_shards_cnt = self.trim_over_bound_shards(&aligned_regions)?;
        Ok((aligned_regions, trimmed_shards_cnt))
    }

    pub fn get_region_split_keys(&self, keyspace_prefix: &[u8]) -> Vec<Vec<u8>> {
        let mut split_keys = Vec::with_capacity(self.sorted_shards.len() - 1);
        for shard_id in &self.sorted_shards {
            let shard = self.get_shard(*shard_id).unwrap();
            debug!(
                "get_region_split_keys, shard_id: {} start key {:?} end key {:?}",
                shard_id,
                shard.start().to_vec(),
                shard.end().to_vec()
            );
            let mut split_key = keyspace_prefix.to_vec();
            let inner_start = shard.inner_start();

            // Skip the keyspace boundary key.
            if inner_start.is_empty() {
                continue;
            }
            split_key.extend_from_slice(inner_start.deref());

            split_keys.push(split_key);
        }

        split_keys
    }

    fn update_schema_file_restore_version(&self, schema_file_id: u64) -> Result<u64> {
        let runtime = self.dfs.get_runtime();
        let schema_file = runtime.block_on(
            self.kv_engine
                .as_ref()
                .unwrap()
                .load_schema_file(schema_file_id),
        )?;
        info!(
            "Keyspace {} set schema file {schema_file_id} restore version to {}",
            self.tag(),
            self.truncate_ts
        );
        let new_schema_file = schema_file.set_restore_version(self.truncate_ts);
        let new_schema_file_id = *self.id_allocator.alloc_id(1)?.first().unwrap();
        let schema_data = new_schema_file.to_bytes();
        let opts = dfs::Options::default().with_type(dfs::FileType::Schema);
        runtime.block_on(
            self.dfs
                .create(new_schema_file_id, Bytes::from(schema_data), opts),
        )?;
        info!(
            "Keyspace {} update schema file {schema_file_id} restore version to {}, new file id {new_schema_file_id}",
            self.tag(),
            self.truncate_ts
        );
        Ok(new_schema_file_id)
    }

    fn gather_all_tables(
        &self,
        aligned_regions: Vec<AlignedRegion>,
    ) -> Result<(Vec<ShardMeta>, usize /* number of sstables */)> {
        let mut sstables_cnt = 0;
        let mut target_shards = Vec::with_capacity(aligned_regions.len());
        let inner_key_off = self.inner_key_off();
        let mut new_schema_file_id = None;
        for region in aligned_regions {
            let mut meta = ShardMeta::default();
            meta.id = region.target_region.id;
            meta.ver = region.target_region.epoch.version;
            meta.range = ShardRange::new(
                region.target_region.get_start_key(),
                region.target_region.get_end_key(),
                inner_key_off,
            );
            if let Some(encryption_key) = self.get_keyspace_exported_encryption_key() {
                meta.set_property(ENCRYPTION_KEY, encryption_key.as_slice());
            }
            let mut properties_helper =
                kvengine::util::PropertiesHelper::new_from_shard_meta(&meta);
            properties_helper.set_rewrite_range_prefix(!self.is_inplace_restore());
            for shard_id in region.backup_shards_id {
                let shard = self.get_shard(shard_id).unwrap();
                let mut all_l0_ssts: HashSet<u64> = HashSet::default();
                for (&file_id, file_meta) in shard.meta.all_files() {
                    if meta.overlap_bound(file_meta.data_bound()) {
                        meta.add_file(file_id, file_meta.clone());
                        sstables_cnt += 1;
                        if file_meta.get_level() == 0 && file_meta.is_sst_file() {
                            all_l0_ssts.insert(file_id);
                        }
                    }
                }
                // Only add the filtered l0s from ShardMeta.unconverted_l0s
                // to meta.unconverted_l0s
                meta.unconverted_l0s.extend(
                    shard
                        .meta
                        .unconverted_l0s
                        .iter()
                        .filter(|id| all_l0_ssts.contains(id)),
                );
                // Use `base_version` as table version, and `data_sequence` is 0.
                // And they will be adjusted at server side in `restore_shard` procedure.
                meta.base_version = cmp::max(meta.base_version, shard.table_version());
                properties_helper.merge_shard_meta(&shard.meta);
                if shard.meta.schema_file_id > 0 {
                    if new_schema_file_id.is_none() {
                        // Get the schema file and update schema_restore_version. Then put the new
                        // schema file to dfs.
                        new_schema_file_id = Some(
                            self.update_schema_file_restore_version(shard.meta.schema_file_id)?,
                        );
                    }
                    meta.schema_file_id = new_schema_file_id.unwrap();
                    meta.schema_file_ver = shard.meta.schema_file_ver;
                    meta.schema_restore_ver = self.truncate_ts;
                    // Update columnar_snap_version to initial value to guarantee the new flushed
                    // l0s can be added to unconverted_l0s in target shard.
                    meta.columnar_snap_version = INITIAL_SNAP_VERSION;
                }
            }

            properties_helper.build_to_shard_meta(&mut meta);
            target_shards.push(meta);
        }
        Ok((target_shards, sstables_cnt))
    }

    fn truncate_shard_ts(&mut self, shard_id: u64) -> Result<bool /* has_truncate_ts */> {
        let shard_meta = &self.get_shard(shard_id).unwrap().meta;
        let kvengine = self.kv_engine.as_ref().unwrap();
        let shard = kvengine
            .get_shard_with_ver(shard_meta.id, shard_meta.ver)
            .expect("Could not find shard with meta");
        let keyspace_id = self.keyspace_id;
        let mut res_cs = kvengine
            .truncate_with_ts(&shard, self.truncate_ts.into())?
            .unwrap();
        if res_cs.has_truncate_ts() && !ShardMeta::is_empty_table_change(res_cs.get_truncate_ts()) {
            let shard = self.get_shard_mut(shard_id).unwrap();
            res_cs.set_sequence(shard.meta.seq + 1);
            debug!(
                "Keyspace {} before truncate ts: {:?}, table change: {:?}",
                keyspace_id,
                shard,
                res_cs.get_truncate_ts()
            );
            shard.meta.apply_change_set(&res_cs);
            debug!(
                "Keyspace {} after apply_truncate_ts: {:?}",
                keyspace_id, shard
            );
            Ok(true)
        } else {
            Ok(false)
        }
    }

    fn trim_over_bound(&mut self, shard_id: u64) -> Result<bool /* has_trim_over_bound */> {
        let shard = self.get_shard(shard_id).unwrap();
        let kvengine = self.kv_engine.as_ref().unwrap();

        let mut res_cs = kvengine.trim_over_bound_by_meta(&shard.meta)?;
        let keyspace_id = self.keyspace_id;
        if res_cs.has_trim_over_bound() {
            let shard = self.get_shard_mut(shard_id).unwrap();
            res_cs.set_sequence(shard.meta.seq + 1);
            debug!(
                "Keyspace {} before trim_over_bound: {:?}, table change: {:?}",
                keyspace_id,
                shard,
                res_cs.get_trim_over_bound()
            );
            shard.meta.apply_change_set(&res_cs);
            debug!(
                "Keyspace {} after apply_trim_over_bound: {:?}",
                keyspace_id, shard
            );
            Ok(true)
        } else {
            Ok(false)
        }
    }

    fn trim_over_bound_shards(
        &mut self,
        aligned_regions: &[AlignedRegion],
    ) -> Result<usize /* number of trimmed shards */> {
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
                info!("{} generate_snapshot cs: {:?}", self.tag, cs);
                cs
            })
            .collect()
    }

    fn inner_key_off(&self) -> usize {
        self.get_shard(*self.sorted_shards.first().unwrap())
            .unwrap()
            .meta
            .range
            .inner_key_off
    }

    fn is_inplace_restore(&self) -> bool {
        self.keyspace_id == self.target_keyspace_id
    }

    pub fn reset_keyspace(&mut self, keyspace_id: u32, target_keyspace_id: u32) -> Result<()> {
        let (keyspace_start, keyspace_end) = ApiV2::get_txn_keyspace_range(keyspace_id);
        self.tag = make_keyspace_tag(keyspace_id, target_keyspace_id);
        self.keyspace_id = keyspace_id;
        self.target_keyspace_id = target_keyspace_id;
        self.keyspace_start = keyspace_start;
        self.keyspace_end = keyspace_end;

        let kv_engine = self.kv_engine.take().unwrap();
        kv_engine.close();
        drop(kv_engine);

        if let Some(sender) = self.meta_sender.take() {
            sender.send(StoreMsg::Stop).unwrap();
        }
        self.meta_applier.take();
        self.sorted_shards.clear();
        self.raw_metas.clear();
        self.store_shards.clear();
        self.shards_store_map = None;
        self.shards.clear();
        self.shards_need_flush.clear();
        self.shards_need_truncate.clear();
        self.tolerated_err = 0;
        self.txn_chunk_ids_in_wal = None;
        self.load_shards()?;
        self.collect_txn_chunks_in_wal()?;

        self.check_all_shard_files()?;

        for store_id in self.get_all_stores_id() {
            self.setup_kv_engine(store_id)?;
        }
        Ok(())
    }

    fn get_keyspace_exported_encryption_key(&self) -> Option<Vec<u8>> {
        self.get_sorted_shard(0)
            .meta
            .get_property(ENCRYPTION_KEY)
            .map(|x| x.to_vec())
    }

    fn tolerated_err(&self) -> usize {
        self.tolerated_err
    }

    // Should be called after `load_shards`.
    fn collect_txn_chunks_in_wal(&mut self) -> Result<()> {
        let mut total_chunk_ids = Vec::new();

        let master_key = self
            .dfs
            .get_runtime()
            .block_on(self.security_conf.new_master_key());
        let mut decryption_buf = Vec::new();

        for store_id in self.get_all_stores_id() {
            let rf_engine = self.raft_engines.get(&store_id).unwrap();
            for &shard_id in self.store_shards.get(&store_id).unwrap() {
                let shard = self.get_shard(shard_id).unwrap();

                let meta = self.raw_metas.get(&shard_id).unwrap();
                let snap = meta.get_snapshot();

                let encryption_key =
                    kvengine::get_shard_property(ENCRYPTION_KEY, snap.get_properties()).map(
                        |exported_key| master_key.decrypt_encryption_key(&exported_key).unwrap(),
                    );

                let tag = ShardTag::new(store_id, IdVer::from_change_set(meta));
                let peer_tag =
                    PeerTag::new(store_id, RegionIdVer::new(meta.shard_id, meta.shard_ver));

                let applied_index = snap.get_data_sequence();
                let commit_index = shard.raft_state.get_commit();
                let low_idx = applied_index + 1;
                let high_idx = commit_index + 1;
                debug_assert!(
                    low_idx <= high_idx,
                    "invalid entry index, low_idx {}, high_idx {}",
                    low_idx,
                    high_idx
                );
                if low_idx >= high_idx {
                    continue;
                }

                let mut entries = Vec::with_capacity((high_idx.saturating_sub(low_idx)) as usize);
                rf_engine
                    .fetch_raft_entries_to(shard.peer_id, low_idx, high_idx, None, &mut entries)
                    .map_err(|e| -> Error {
                        box_err!(
                            "{} entries unavailable err: {:?}, low: {}, high {}",
                            tag,
                            e,
                            low_idx,
                            high_idx
                        )
                    })?;

                let mut chunk_ids = Vec::new();
                for e in entries {
                    if e.data.is_empty() || e.entry_type != eraftpb::EntryType::EntryNormal {
                        continue;
                    }

                    let req =
                        parse_raft_cmd(&peer_tag, &e, encryption_key.as_ref(), &mut decryption_buf);
                    if req.get_header().get_region_epoch().version != meta.shard_ver {
                        continue;
                    }

                    if let Some(custom) = rlog::get_custom_log(&req) {
                        if custom.is_txn_file_ref() {
                            let mut txn_file_ref = box_try!(custom.get_txn_file_ref());
                            chunk_ids.extend(txn_file_ref.take_chunk_ids());
                        }
                    }
                }

                if !chunk_ids.is_empty() {
                    info!("{} collect txn chunks in wal [{}, {})", tag, low_idx, high_idx;
                        "chunk_ids" => ?chunk_ids);
                    total_chunk_ids.extend(chunk_ids);
                }
            }
        }

        total_chunk_ids.sort();
        total_chunk_ids.dedup();
        self.txn_chunk_ids_in_wal = Some(total_chunk_ids);
        Ok(())
    }
}

struct MetaApplier {
    keyspace_id: u32,
    engine: kvengine::Engine,
    encryption_key: Option<EncryptionKey>,
    shards: RwLock<Option<HashMap<u64, BackupShard>>>,
    store_rx: mpsc::Receiver<StoreMsg>,
    errors: Mutex<Vec<Error>>,
}

impl Drop for MetaApplier {
    fn drop(&mut self) {
        info!(
            "Meta applier is dropped, engine {}",
            self.engine.get_engine_id()
        );
    }
}

impl MetaApplier {
    fn new(
        keyspace_id: u32,
        engine: kvengine::Engine,
        encryption_key: Option<EncryptionKey>,
        shards: HashMap<u64, BackupShard>,
        store_rx: mpsc::Receiver<StoreMsg>,
    ) -> Self {
        Self {
            keyspace_id,
            engine,
            encryption_key,
            shards: RwLock::new(Some(shards)),
            store_rx,
            errors: Default::default(),
        }
    }

    pub fn take_shards(&self) -> Option<HashMap<u64, BackupShard>> {
        self.shards.wl().take()
    }

    pub fn take_errors(&self) -> Vec<Error> {
        let mut errors = self.errors.lock().unwrap();
        mem::take(&mut *errors)
    }

    fn run(&self) {
        'outer: loop {
            let msg = match self.store_rx.recv() {
                Ok(msg) => msg,
                Err(err) => {
                    error!(
                        "Keyspace {} applier recv task error: {:?}",
                        self.keyspace_id, err
                    );
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
                    debug!(
                        "Keyspace {} shard {} apply change set: {:?}",
                        self.keyspace_id, tag, cs
                    );

                    {
                        let mut shards = self.shards.wl();
                        if let Some(shards) = shards.as_mut() {
                            let shard = shards.get_mut(&cs.shard_id).unwrap();
                            shard.meta.apply_change_set(&cs);
                        } else {
                            if cs.has_compaction() {
                                debug!(
                                    "Keyspace {} MetaApplier: ignore changeset: {:?}",
                                    self.keyspace_id, cs
                                );
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
                        .prepare_change_set(cs, false, None, self.encryption_key.clone())
                        .and_then(|cs| self.engine.apply_change_set(cs))
                    {
                        Ok(()) => debug!(
                            "Keyspace {} shard {} MetaApplier apply change set successfully",
                            self.keyspace_id, tag
                        ),
                        Err(err) => {
                            let msg = format!(
                                "Keyspace {} shard {} MetaApplier apply change set failed: {:?}",
                                self.keyspace_id, tag, err
                            );
                            error!("{}", msg);
                            self.errors
                                .lock()
                                .unwrap()
                                .push(Error::Other(box_err!("{}", msg)));
                        }
                    }
                }
                StoreMsg::Stop => {
                    info!(
                        "Engine {} restore keyspace {} meta applier receive stop msg and stop now",
                        self.engine.get_engine_id(),
                        self.keyspace_id
                    );
                    break 'outer;
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
    pd_client: &dyn PdClient,
    start_key: &[u8],
    end_key: &[u8],
) -> Result<Vec<RawRegion>> {
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

fn verify_regions_boundary(start_key: &[u8], end_key: &[u8], regions: &[RawRegion]) -> Result<()> {
    if regions.is_empty() {
        return Err(box_err!("no region"));
    }

    let first_region = regions.first().unwrap();
    let last_region = regions.last().unwrap();
    if first_region.get_start_key() != start_key {
        return Err(box_err!(
            "unexpected start key of first region: {:?}, start_key: {:?}",
            first_region,
            start_key
        ));
    } else if last_region.get_end_key() != end_key {
        return Err(box_err!(
            "unexpected end key of last region: {:?}, end_key: {:?}",
            last_region,
            end_key
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
    keyspace_id: u32,
    runtime: &Runtime,
    pd_client: Arc<dyn PdClient>,
    snapshots: Vec<pb::ChangeSet>,
    success_ranges: &mut MergeRanges,
    timeout: Duration,
) -> Result<RestoredSnapshots> {
    let mut handles = Vec::with_capacity(snapshots.len());
    for snap in snapshots {
        let pd_client = pd_client.clone();
        let start = snap.get_restore_shard().get_outer_start().to_vec();
        let end = snap.get_restore_shard().get_outer_end().to_vec();
        let task = async move { request_restore_snapshot(pd_client, &snap, timeout).await };
        handles.push((runtime.spawn(task), start, end));
    }

    let is_error_retryable = |err: &Error| {
        matches!(
            err,
            Error::RegionVerNotMatch { .. } | Error::RegionNotFoundOrNoLeader(_)
        )
    };

    let mut restored = RestoredSnapshots::default();
    for (h, start, end) in handles {
        match runtime.block_on(h).unwrap() {
            Ok(resp) => {
                success_ranges.insert(start, end);
                restored.count += 1;
                restored.restore_bytes += resp.restore_bytes;
            }
            Err(e) if is_error_retryable(&e) => {
                info!(
                    "Keyspace {} request_restore_snapshot error: {:?}, retry in next loop",
                    keyspace_id, e
                );
            }
            Err(e) => {
                error!(
                    "Keyspace {} request_restore_snapshot error: {:?}",
                    keyspace_id, e
                );
                return Err(e);
            }
        }
    }
    Ok(restored)
}

async fn request_restore_snapshot(
    pd_client: Arc<dyn PdClient>,
    cs: &pb::ChangeSet,
    timeout: Duration,
) -> Result<RestoreShardResponse> {
    let post_data = Cow::from(cs.write_to_bytes().unwrap());
    let mut last_err = None;
    let mut retry_cnt = 0;
    let start_time = Instant::now();
    'retry: while start_time.saturating_elapsed() < timeout {
        retry_cnt += 1;
        let (region, leader) = match pd_client.get_region_leader_by_id(cs.shard_id).await? {
            Some((region, leader)) => (region, leader),
            None => {
                let e = Err(Error::RegionNotFoundOrNoLeader(cs.shard_id));
                warn!("{}:{}: {:?}", cs.shard_id, cs.shard_ver, e);
                last_err = Some(e);
                tokio::time::sleep(Duration::from_millis(200)).await;
                continue 'retry;
            }
        };

        let region_ver = region.get_region_epoch().get_version();
        let shard_ver = cs.get_shard_ver();
        if region_ver != shard_ver {
            return Err(Error::RegionVerNotMatch {
                expected: shard_ver,
                actual: region_ver,
            });
        }

        let store = pd_client.get_store_async(leader.get_store_id()).await?;
        let tag = ShardTag::new(store.get_id(), IdVer::new(cs.shard_id, cs.shard_ver));

        let security_mgr = pd_client.get_security_mgr();
        let uri = security_mgr.build_uri(format!("{}/restore-shard", &store.status_address))?;
        let req = Request::post(uri)
            .body(Body::from(post_data.clone()))
            .unwrap();
        match send_request_to_store(req, &store, security_mgr).await {
            Ok(resp) => {
                let resp: RestoreShardResponse = serde_json::from_slice(&resp).unwrap();
                debug!("{} request_restore_snapshot succeed", tag);
                return Ok(resp);
            }
            Err(e) => {
                let err_msg = format!(
                    "{} request_restore_snapshot #{retry_cnt} error: {:?}, region: {:?}, leader: {:?}",
                    tag, e, region, leader
                );
                warn!("{}", err_msg);
                last_err = Some(Err(box_err!(err_msg)));
                tokio::time::sleep(Duration::from_millis(500)).await;
                continue 'retry;
            }
        }
    }
    last_err.expect("there must be error")
}

fn make_keyspace_tag(source_keyspace_id: u32, target_keyspace_id: u32) -> String {
    format!("{}->{}", source_keyspace_id, target_keyspace_id)
}

struct PeerPreprocessor {
    preprocessed_index: u64,
    region: metapb::Region,
    preprocessed_region: Option<metapb::Region>,
    peer: metapb::Peer,
    shard_meta: Option<ShardMeta>,
    last_committed_split_idx: u64,
    pending_truncate: Option<(u64 /* term */, u64 /* index */)>,
    raft_hard_state: eraftpb::HardState,
    raft_state: RaftState,
    pending_merge_state: Option<MergeState>,
    first_no_kv_idx: u64,
    last_no_kv_idx: u64,
    learner_skip_idx: u64,
    encryption_key: Option<EncryptionKey>,
}

impl PeerPreprocessor {
    fn new(rf_engine: &RfEngine, shard: &mut BackupShard) -> Self {
        let peer = metapb::Peer {
            id: shard.peer_id,
            store_id: shard.store_id,
            role: PeerRole::Voter,
            ..Default::default()
        };
        let mut region_state = rf_engine
            .load_region_state(shard.peer_id, shard.ver())
            .unwrap_or_else(|| {
                panic!("failed to load region state for peer {}", shard.peer_id);
            });
        let region = region_state.take_region();
        let merge_state = region_state
            .has_merge_state()
            .then(|| region_state.take_merge_state());

        // Set `preprocessed_index` to the `data_sequence` to replay from last flush and
        // restore memory-based fields of `ShardMeta`.
        // See https://github.com/tidbcloud/cloud-storage-engine/issues/1680.
        //
        // Also note that when the shard is not yet initial flushed, in theory we should
        // preprocess from parent. But the only memory field `txn_file_locks` so far is
        // used for ignore split/merge, which will not happen when the shard is not
        // initial flushed. So currently we do not preprocess from parent for
        // simplicity.
        // See https://github.com/tidbcloud/cloud-storage-engine/issues/1862.
        let preprocessed_index = shard.meta.data_sequence;

        Self {
            preprocessed_index,
            region,
            preprocessed_region: None,
            peer,
            shard_meta: Some(shard.meta.clone()), // TODO: mem::take()
            last_committed_split_idx: 0,
            pending_truncate: None,
            raft_hard_state: shard.raft_state.get_hard_state(),
            raft_state: shard.raft_state,
            pending_merge_state: merge_state,
            first_no_kv_idx: 0, // truncated ?
            last_no_kv_idx: 0,  // truncated ?
            learner_skip_idx: 0,
            encryption_key: None,
        }
    }

    fn as_ref(&mut self) -> rfstore::store::peer::PreprocessRef<'_> {
        rfstore::store::peer::PreprocessRef {
            preprocessed_index: &mut self.preprocessed_index,
            region: &mut self.region,
            preprocessed_region: &mut self.preprocessed_region,
            peer: &mut self.peer,
            shard_meta: &mut self.shard_meta,
            last_committed_split_idx: &mut self.last_committed_split_idx,
            pending_truncate: &mut self.pending_truncate,
            raft_hard_state: self.raft_hard_state.clone(),
            raft_state: &mut self.raft_state,
            pending_merge_state: &mut self.pending_merge_state,
            first_no_kv_idx: &mut self.first_no_kv_idx,
            last_no_kv_idx: &mut self.last_no_kv_idx,
            learner_skip_idx: &mut self.learner_skip_idx,
            encryption_key: &mut self.encryption_key,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{hash_map::Entry, BTreeMap};

    use api_version::api_v2::KEYSPACE_PREFIX_LEN;

    use super::*;

    #[test]
    fn test_handle_overlapping_shards() {
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
            shard.meta.range = ShardRange::new(tuple.2.as_bytes(), tuple.3.as_bytes(), 0);
            shard
        };

        let get_leader_shards_by_ver =
            |leader_shards: &mut HashMap<u64, BackupShard>,
             _store_id: u64,
             shards: Vec<(u64, u64, &str, &str)>| {
                for shard_tuple in shards {
                    let shard = make_backup_shard(shard_tuple);
                    match leader_shards.entry(shard.region_id) {
                        Entry::Occupied(mut o) => {
                            if shard.ver() > o.get().ver() {
                                o.insert(shard);
                            }
                        }
                        Entry::Vacant(v) => {
                            v.insert(shard);
                        }
                    }
                }
            };

        let cases = vec![
            (
                vec![(1, 100, "00", "01")], // store0: Vec<(shard_id, ver, start, end)>
                vec![(1, 100, "00", "01")], // store1
                vec![(1, 100, "00", "01")], // store2
                // expected: Option<Vec<(shard_id, ver, start,end)>>, None means error
                Some(vec![(1, 100, "00", "01")]),
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
            let mut leader_shards = HashMap::default();
            get_leader_shards_by_ver(&mut leader_shards, 0, store0);
            get_leader_shards_by_ver(&mut leader_shards, 1, store1);
            get_leader_shards_by_ver(&mut leader_shards, 2, store2);

            let res = BackupCluster::handle_overlapping_shards(&mut leader_shards);
            if let Some(expected) = expected {
                let expected_sorted_shards: Vec<u64> = expected.iter().map(|x| x.0).collect();
                let expected_leader_shards: HashMap<u64, BackupShard> =
                    HashMap::from_iter(expected.into_iter().map(|x| (x.0, make_backup_shard(x))));

                let sorted_shards = res.unwrap();
                assert_eq!(
                    BTreeMap::from_iter(leader_shards.into_iter()),
                    BTreeMap::from_iter(expected_leader_shards.into_iter()),
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
        test_align_target_regions_helper(false);
        test_align_target_regions_helper(true);
    }

    fn test_align_target_regions_helper(inner_key_off_enabled: bool) {
        let inner_key_off = if inner_key_off_enabled {
            KEYSPACE_PREFIX_LEN
        } else {
            0
        };
        let key_prefix = b"x00";
        let cases: Vec<(Vec<&[u8]>, Vec<&[u8]>, Vec<Vec<u64>>)> = vec![
            (
                vec![b"0", b"1"], // backup_regions_boundary_keys, the index is the region id.
                vec![b"0", b"1"], // target_regions_boundary_keys
                vec![vec![0]],    // align_regions_id
            ),
            (
                vec![b"0", b"05", b"1"],
                vec![b"0", b"05", b"1"],
                vec![vec![0], vec![1]],
            ),
            (
                vec![b"1", b"13", b"16", b"2"],
                vec![b"1", b"15", b"2"],
                vec![vec![0, 1], vec![1, 2]],
            ),
            (
                vec![b"2", b"25", b"3"],
                vec![b"2", b"23", b"26", b"3"],
                vec![vec![0], vec![0, 1], vec![1]],
            ),
            #[cfg_attr(rustfmt, rustfmt_skip)]
            (
            vec![ b"3",    b"32",            b"34",  b"36", b"37", b"38", b"39", b"4"],
            vec![ b"3",    b"32",   b"33",   b"34",  b"36",                      b"4"],
            vec![vec![0], vec![1], vec![1], vec![2],         vec![3,4,5,6]],
            ),
        ];

        let make_regions = |keys: Vec<&[u8]>| -> Vec<RawRegion> {
            let mut regions = Vec::with_capacity(keys.len() - 1);
            for (idx, w) in keys.as_slice().windows(2).enumerate() {
                let (mut raw_start, mut raw_end) = (key_prefix.to_vec(), key_prefix.to_vec());
                raw_start.extend_from_slice(w[0]);
                raw_end.extend_from_slice(w[1]);
                regions.push(RawRegion {
                    id: idx as u64,
                    raw_start,
                    raw_end,
                    ..Default::default()
                });
            }

            regions
        };

        let make_backup_shards = |regions: Vec<RawRegion>| {
            let mut shards = HashMap::with_capacity(regions.len());
            let mut shards_id = Vec::with_capacity(regions.len());
            for r in regions {
                let mut shard = BackupShard::default();
                shard.region_id = r.id;
                shard.meta.range =
                    ShardRange::new(r.get_start_key(), r.get_end_key(), inner_key_off);

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
                inner_key_off,
            );
            assert_eq!(aligned_regions, expected, "case: {}", case_idx);
        }
    }
}
