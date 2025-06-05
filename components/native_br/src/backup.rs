// Copyright 2023 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    path::Path,
    time::{Duration, Instant as StdInstant, SystemTime, UNIX_EPOCH},
};

use bytes::Bytes;
use chrono::{DateTime, NaiveDate, NaiveDateTime, NaiveTime, Utc};
use fail::fail_point;
use futures::{compat::Stream01CompatExt, executor::block_on, StreamExt};
use http::Request;
use hyper::Body;
use kvengine::dfs::{DFSConfig, S3Fs};
use kvproto::metapb::Store;
use pd_client::PdClient;
use protobuf::Message;
use regex::Regex;
use rfenginepb::{ClusterBackupMeta, StoreBackupMeta};
use security::{SecurityConfig, SecurityManager};
use slog_global::{error, info, warn};
use tikv::storage::mvcc::TimeStamp;
use tikv_util::{
    backoff, config::ReadableDuration, retry::try_wait_result_async, time::Instant,
    timer::GLOBAL_TIMER_HANDLE,
};

use crate::{
    common::{
        create_pd_client, generate_etcd_connect_opt, get_all_stores_except_tiflash,
        get_latest_backup_meta, send_request_to_store, INCREMENTAL_BACKUP_FILE_NAME_FORMAT,
        INCREMENTAL_BACKUP_FOLDER_FORMAT,
    },
    error::{Error, SharedError},
};

const MAX_BATCH_GET_CNT: i64 = 1024;
// `PD_KEY_SPACE_META_PATH` is an array of tuples each containing:
// 1. A string representing a keyspace meta path.
// 2. A boolean indicating whether the path is prefixed by 'pd/$cluster_id/'.
//
// Keyspace meta paths in PD are:
// 1. keyspace meta: "/pd/$cluster_id/keyspaces/"
// 2. keyspace region label: "/pd/$cluster_id/region_label/keyspaces/"
// 3. keyspace placement rules: "/pd/$cluster_id/rules/"
// 4. keyspace group membership:
//    "/pd/$cluster_id/tso/keyspace_groups/membership/"
// 5. resource group information: "resource_group/" (Note: no leading slash)
// 6. tidb worker keys: "/tidb/remote/worker/"
const PD_KEY_SPACE_META_PATH: [(&str, bool); 6] = [
    ("keyspaces/", true),
    ("region_label/keyspaces/", true),
    ("rules/", true),
    ("tso/keyspace_groups/membership/", true),
    ("resource_group/", false),
    ("/tidb/remote/worker/", false),
];

const BACKUP_GC_SERVICE_NAME: &str = "native_br";
// `BACKUP_SERVICE_SAFEPOINT_TTL` should not be too long, as failure of backup
// will block GC.
const BACKUP_SERVICE_SAFEPOINT_TTL: Duration = Duration::from_secs(12 * 60 * 60); // 12 hour.

// Backup must not have more than 1 tolerated error of store.
const BACKUP_MAX_TOLERATED_ERROR: usize = 1;

pub type Result<T> = std::result::Result<T, Error>;
pub type SharedResult<T> = std::result::Result<T, SharedError>;

#[derive(Clone, Copy, PartialEq)]
pub enum BackupType {
    Full,
    Incremental,
    Lightweight,
}

/// Generate full path `/<prefix>/backup/<name>`.
/// When `name` is empty, `backup_ts` must be some.
pub fn backup_file_full_path(prefix: String, name: String, backup_ts: Option<u64>) -> String {
    let name = if name.is_empty() {
        IncrementalBackupFile::from_backup_ts(backup_ts.unwrap()).into_name()
    } else {
        name
    };
    // Use `Path` to handle path separators.
    Path::new(&prefix)
        .join("backup")
        .join(name)
        .to_str()
        .unwrap()
        .to_string()
}

pub fn execute_incremental_backup(
    config: BackupConfig,
    name: String,
    interval: Duration,
) -> Result<()> {
    if !name.is_empty() {
        return Err(Error::BackupError(
            "Don't support non-empty name for incremental backup.".to_string(),
        ));
    }

    if interval.is_zero() {
        return Err(Error::BackupError(
            "Interval must be positive for incremental backup. Use full or lightweight instead to backup once.".to_string(),
        ));
    }

    let duration = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
    let gap = interval.as_secs() - duration.as_secs() % interval.as_secs();
    let start_time = StdInstant::now()
        .checked_add(Duration::from_secs(gap))
        .unwrap();
    let mut interval = GLOBAL_TIMER_HANDLE.interval(start_time, interval).compat();
    let pd_client = create_pd_client(&config.security, &config.pd);
    let mut cluster_backup_meta = None;
    while let Some(Ok(_)) = block_on(interval.next()) {
        match backup_cluster(
            config.clone(),
            BackupType::Incremental,
            name.clone(),
            &pd_client,
            cluster_backup_meta.clone(),
        ) {
            Ok((_, meta)) => {
                cluster_backup_meta = Some(meta);
            }
            Err(e) => {
                // For other errors, retry incremental backup later.
                if need_full_backup(&e) {
                    warn!("Incremental backup fails {:?}, fallback to full backup", e);
                    // If incremental backup fails, restart full backup automatically.
                    match backup_cluster(
                        config.clone(),
                        BackupType::Full,
                        name.clone(),
                        &pd_client,
                        None,
                    ) {
                        Ok((_, meta)) => cluster_backup_meta = Some(meta),
                        Err(e) => {
                            return Err(Error::BackupError(format!(
                                "Full backup still fail {:?}",
                                e
                            )));
                        }
                    }
                } else {
                    warn!("Incremental backup fails {:?}, retry later", e);
                }
            }
        }
    }
    Ok(())
}

pub fn execute_full_backup(config: BackupConfig, name: String) -> Result<()> {
    // TODO: Set safepoint before backup and delete it after backup.
    let pd_client = create_pd_client(&config.security, &config.pd);
    backup_cluster(config, BackupType::Full, name, &pd_client, None).map(|_| ())
}

// Backup once if interval is 0.
pub fn execute_lightweight_backup(
    config: BackupConfig,
    name: String,
    interval: Duration,
) -> Result<()> {
    let pd_client = create_pd_client(&config.security, &config.pd);

    // Once lightweight backup.
    if interval.is_zero() {
        return backup_cluster(config, BackupType::Lightweight, name, &pd_client, None).map(|_| ());
    }

    // Cron lightweight backup.
    let duration = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
    let gap = interval.as_secs() - duration.as_secs() % interval.as_secs();
    let start_time = StdInstant::now()
        .checked_add(Duration::from_secs(gap))
        .unwrap();
    let mut interval = GLOBAL_TIMER_HANDLE.interval(start_time, interval).compat();
    let mut cluster_backup_meta = None;
    while let Some(Ok(_)) = block_on(interval.next()) {
        match backup_cluster(
            config.clone(),
            BackupType::Lightweight,
            name.clone(),
            &pd_client,
            cluster_backup_meta.clone(),
        ) {
            Ok((path, meta)) => {
                info!("lightweight backup succeed"; "path" => path, "meta" => %meta);
                cluster_backup_meta = Some(meta);
            }
            Err(e) => {
                error!("lightweight backup fail, {:?}", e);
            }
        }
    }
    Ok(())
}

pub fn update_service_safe_point(pd_client: &dyn PdClient, safepoint: u64) -> Result<()> {
    if let Err(e) = block_on(pd_client.update_service_safe_point(
        BACKUP_GC_SERVICE_NAME.to_string(),
        TimeStamp::from(safepoint),
        BACKUP_SERVICE_SAFEPOINT_TTL,
    )) {
        error!(
            "fail to update gc service safepoint {safepoint}, err {:?}",
            e
        );
        return Err(Error::PdError(e));
    }
    info!("update backup gc service safe point"; "safepoint" => safepoint);
    Ok(())
}

pub fn get_backup_ts(pd_client: &dyn PdClient) -> Result<u64> {
    match pd_client.get_min_tso() {
        Ok(ts) => Ok(ts.into_inner()),
        Err(e) => {
            if pd_client::grpc_error_is_unimplemented(&e) {
                info!("get_min_tso is unimplemented, fall back to get_tso");
                Ok(block_on(pd_client.get_tso())?.into_inner())
            } else {
                Err(e.into())
            }
        }
    }
}

// return backup file key(full path) and ClusterBackupMeta
pub fn backup_cluster(
    config: BackupConfig,
    backup_type: BackupType,
    name: String,
    pd_client: &dyn PdClient,
    last_backup_meta: Option<ClusterBackupMeta>,
) -> Result<(String, ClusterBackupMeta)> {
    let backup_ts = get_backup_ts(pd_client)?;
    let ret = backup_cluster_with_ts(
        config,
        backup_type,
        name,
        pd_client,
        backup_ts,
        last_backup_meta,
    )?;
    // Incremental backup keeps running in production env, so we only update service
    // safepoint when backup succeed. Therefore the first backup cannot be used as
    // we don't set safepoint before backup for logical simplicity.
    update_service_safe_point(pd_client, backup_ts)?;
    Ok(ret)
}

pub fn backup_cluster_with_ts(
    config: BackupConfig,
    backup_type: BackupType,
    name: String,
    pd_client: &dyn PdClient,
    backup_ts: u64,
    last_backup_meta: Option<ClusterBackupMeta>,
) -> Result<(String, ClusterBackupMeta)> {
    let mut stores = get_all_stores_except_tiflash(pd_client)?;
    let cluster_id = pd_client.get_cluster_id()?;

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .thread_name("backup-cluster")
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap();

    let dfs_conf = config.dfs.clone();
    let s3fs = S3Fs::new_from_config(dfs_conf);
    let mut cluster_backup_meta = match backup_type {
        BackupType::Full => ClusterBackupMeta::new(),
        BackupType::Incremental => {
            if let Some(meta) = last_backup_meta {
                // Cluster topology may be changed between two loop, so it's necessary to check
                // consistency.
                check_backup_meta_consistency(&meta, &stores)?;
                meta
            } else {
                // If no input backup meta, load latest one from s3.
                let meta = runtime.block_on(get_latest_backup_meta(&s3fs, cluster_id))?;
                if meta.is_lightweight {
                    info!("latest cluster backup meta is lightweight, fallback to full backup");
                    return Err(Error::MetaNotFound(cluster_id));
                }
                check_backup_meta_consistency(&meta, &stores)?;
                meta
            }
        }
        BackupType::Lightweight => {
            let mut meta = ClusterBackupMeta::new();
            let last_backup_meta = last_backup_meta.or_else(|| {
                match runtime.block_on(get_latest_backup_meta(&s3fs, cluster_id)) {
                    Ok(latest) => Some(latest),
                    Err(err) => {
                        warn!("get latest backup meta failed"; "err" => ?err);
                        None
                    }
                }
            });
            if let Some(mut last_backup_meta) = last_backup_meta {
                // Incremental backup of keyspace meta.
                meta.set_meta_revision(last_backup_meta.meta_revision);
                meta.set_keyspace_meta(last_backup_meta.take_keyspace_meta());
            }
            meta
        }
    };
    cluster_backup_meta.set_backup_ts(backup_ts);
    cluster_backup_meta.set_cluster_id(cluster_id);
    cluster_backup_meta.set_is_lightweight(backup_type == BackupType::Lightweight);

    runtime.block_on(backup_pd_keyspace_meta(&config, &mut cluster_backup_meta))?;

    let start_time = Instant::now_coarse();
    let mut bo = backoff::ExponentialBackoff::new(
        Duration::from_secs(1),
        Duration::from_secs(10),
        usize::MAX,
    );
    let tolerated_errs = loop {
        match runtime.block_on(backup_stores(
            &config,
            backup_type,
            pd_client,
            stores.clone(),
            config.timeout.0 / 2,
            &mut cluster_backup_meta,
        )) {
            Ok(tolerated_errs) => {
                break tolerated_errs;
            }
            Err(err) => {
                if start_time.saturating_elapsed() > config.timeout.0 {
                    return Err(err);
                }
                let next_delay = bo.next_delay();
                if next_delay.is_err() {
                    return Err(err);
                }
                if let Error::BackupErrorOnStores(_, missing_stores) = err {
                    stores = missing_stores;
                }
                std::thread::sleep(next_delay.unwrap());
            }
        }
    };

    let alloc_id = pd_client.alloc_id()?;
    let safe_ts = runtime.block_on(pd_client.get_gc_safe_point())?;
    if safe_ts > backup_ts {
        return Err(Error::TsError(safe_ts, backup_ts));
    }
    cluster_backup_meta.set_alloc_id(alloc_id);
    cluster_backup_meta.set_safe_ts(safe_ts);
    cluster_backup_meta.set_tolerated_err(tolerated_errs as u32);

    let backup_key = backup_file_full_path(s3fs.get_prefix(), name, Some(backup_ts));
    let backup_data = Bytes::from(cluster_backup_meta.write_to_bytes().unwrap());
    runtime
        .block_on(s3fs.put_object(backup_key.clone(), backup_data, backup_key.clone()))
        .unwrap();
    info!(
        "cluster backup cluster_id:{}, backup_ts:{}, alloc_id:{}, safe_ts:{}, num_stores:{}, tolerance_errors:{}, path:{}",
        cluster_backup_meta.cluster_id,
        cluster_backup_meta.backup_ts,
        cluster_backup_meta.alloc_id,
        cluster_backup_meta.safe_ts,
        cluster_backup_meta.get_stores().len(),
        cluster_backup_meta.get_tolerated_err(),
        backup_key,
    );
    Ok((backup_key, cluster_backup_meta))
}

async fn backup_stores(
    config: &BackupConfig,
    backup_type: BackupType,
    pd_client: &dyn PdClient,
    stores: Vec<Store>,
    timeout: Duration,
    cluster_backup_meta: &mut ClusterBackupMeta,
) -> Result<usize /* tolerated_errs */> {
    let mut tasks = tokio::task::JoinSet::new();
    let cluster_id = cluster_backup_meta.cluster_id;
    debug_assert!(cluster_id > 0);
    for store in stores {
        let rf_config = get_backup_config(cluster_backup_meta, cluster_id, store.id, backup_type);
        if let Some(rf_config) = rf_config {
            let security_mgr = pd_client.get_security_mgr();
            let task = async move {
                let res = backup_store(rf_config, &store, &security_mgr, timeout).await;
                (store, res)
            };
            tasks.spawn(task);
        } else {
            // Treat tolerated error in incremental backup as error, to make sure that
            // the `config.tolerate_err` will not be violated.
            // TODO: remove incremental backup.
            info!("incremental backup: last_backup.tolerated_err > 0"; "store_id" => store.id, "backup" => ?cluster_backup_meta);
            return Err(Error::IncrementalBackupToleratedError(store.id));
        }
    }

    let mut error_stores = vec![];
    let mut errors = vec![];
    while let Some(res) = tasks.join_next().await {
        let (store, res) =
            res.map_err(|e| Error::BackupError(format!("backup stores: join task failed: {e:?}")))?;
        match res {
            Ok(store_backup_meta) => {
                merge_store_backup_meta(cluster_backup_meta, store_backup_meta);
            }
            Err(err) => {
                warn!("backup store failed"; "store" => store.id, "err" => ?err);
                error_stores.push(store);
                errors.push(err);
            }
        }
    }

    let stores = get_all_stores_except_tiflash(pd_client)?;
    if let Err(missing_stores) = check_backup_meta_intact(cluster_backup_meta, stores) {
        for store in missing_stores {
            if !error_stores.iter().any(|s| s.id == store.id) {
                error_stores.push(store);
            }
        }
    }

    if error_stores.is_empty() {
        Ok(0)
    } else {
        let store_ids = error_stores.iter().map(|s| s.id).collect::<Vec<_>>();
        if error_stores.len() <= config.tolerate_err {
            info!("backup stores: tolerated error: {}", error_stores.len();
                "error_stores" => ?store_ids, "errors" => ?errors);
            Ok(error_stores.len())
        } else {
            warn!("backup stores: not intact"; "error_stores" => ?store_ids, "errors" => ?errors);
            Err(Error::BackupErrorOnStores(errors, error_stores))
        }
    }
}

async fn backup_store(
    config: rfengine::BackupConfig,
    store: &Store,
    security_mgr: &SecurityManager,
    timeout: Duration,
) -> Result<StoreBackupMeta> {
    fail_point!("native_br::backup_store");

    let uri = security_mgr
        .build_uri(format!("{}/rfengine/backup", &store.status_address))
        .unwrap();
    info!("Start backup store with config {}", config);
    let json_string = serde_json::to_string(&config).unwrap();
    let req = Request::post(uri.clone())
        .body(Body::from(json_string.clone()))
        .unwrap();
    let res = match send_request_to_store(req, store, security_mgr, timeout).await {
        Ok((_, resp)) => {
            let mut store_backup_meta = StoreBackupMeta::default();
            store_backup_meta.merge_from_bytes(&resp).unwrap();
            Ok(store_backup_meta)
        }
        Err(Error::HttpError(_, err_msg))
            if err_msg.contains(rfengine::RFENGINE_DFS_WORKER_UNHEALTHY_ERR_MSG) =>
        {
            Err(Error::RfengineDfsWorkerUnhealthy(err_msg))
        }
        Err(e) => Err(e),
    };

    fail_point!("native_br::backup_store::ret");
    res
}

fn merge_store_backup_meta(
    cluster_backup_meta: &mut ClusterBackupMeta,
    store_backup_meta: StoreBackupMeta,
) {
    // Only full backup has manifest. Full backup and lightweight backup need merge
    // store meta.
    //
    // BackupType::Full and BackupType::Lightweight.
    if store_backup_meta.has_manifest() || cluster_backup_meta.is_lightweight {
        // remove the old one and add the new one.
        if let Some(index) = cluster_backup_meta
            .stores
            .iter()
            .position(|s| s.store_id == store_backup_meta.store_id)
        {
            cluster_backup_meta.stores.remove(index);
        }
        cluster_backup_meta.mut_stores().push(store_backup_meta);
    } else {
        // For incremental backup, only WAL is backed up.
        // Append new WAL chunks to original StoreBackupMeta.
        //
        // BackupType::Incremental
        let store = cluster_backup_meta
            .mut_stores()
            .iter_mut()
            .find(|s| s.store_id == store_backup_meta.store_id)
            .unwrap(); // store existence is checked before.
        for chunk in &store_backup_meta.wal_chunks {
            store.mut_wal_chunks().push(chunk.clone());
        }
    }
}

fn get_backup_config(
    backup_meta: &ClusterBackupMeta,
    cluster_id: u64,
    store_id: u64,
    backup_type: BackupType,
) -> Option<rfengine::BackupConfig> {
    let incremental = backup_type == BackupType::Incremental;
    let lightweight = backup_type == BackupType::Lightweight;
    let mut config = rfengine::BackupConfig {
        cluster_id,
        store_id,
        incremental,
        wal_epoch: 0,
        start_offset: 0,
        lightweight,
    };
    if incremental {
        // store id existence is checked in check_backup_meta_consistency
        let store_meta = backup_meta
            .get_stores()
            .iter()
            .find(|s| s.get_store_id() == store_id)?;
        let last_wal = store_meta.get_wal_chunks().last().unwrap();
        config.wal_epoch = last_wal.epoch;
        config.start_offset = last_wal.get_end_off();
    }
    Some(config)
}

fn check_backup_meta_consistency(backup_meta: &ClusterBackupMeta, stores: &[Store]) -> Result<()> {
    if backup_meta.tolerated_err > BACKUP_MAX_TOLERATED_ERROR as u32 {
        error!("check_backup_meta_consistency: Tolerated error exceeds limit"; "backup" => ?backup_meta);
        return Err(Error::BackupError(format!(
            "Tolerated error {} exceeds limit {}",
            backup_meta.tolerated_err, BACKUP_MAX_TOLERATED_ERROR
        )));
    }

    if stores.len() != backup_meta.stores.len() + backup_meta.tolerated_err as usize {
        return Err(Error::TopoChanged(format!(
            "Stores' count changed during backup, cur: {}, backed up: {}",
            stores.len(),
            backup_meta.stores.len()
        )));
    }
    let remain_stores: Vec<u64> = stores
        .iter()
        .filter(|s| {
            !backup_meta
                .stores
                .iter()
                .any(|s_meta| s_meta.store_id == s.id)
        })
        .map(|s| s.id)
        .collect();
    if remain_stores.len() == backup_meta.tolerated_err as usize {
        Ok(())
    } else {
        Err(Error::TopoChanged(format!(
            "Check backup meta fails, have no meta for store: {:?}",
            remain_stores
        )))
    }
}

fn check_backup_meta_intact(
    backup_meta: &ClusterBackupMeta,
    stores: Vec<Store>,
) -> std::result::Result<(), Vec<Store> /* missing_stores */> {
    let missing_stores = stores
        .into_iter()
        .filter(|s| {
            !backup_meta
                .stores
                .iter()
                .any(|s_meta| s_meta.store_id == s.id)
        })
        .collect::<Vec<_>>();
    if missing_stores.is_empty() {
        Ok(())
    } else {
        Err(missing_stores)
    }
}

pub fn need_full_backup(err: &Error) -> bool {
    matches!(err, Error::TopoChanged(_) | Error::MetaNotFound(_))
}

async fn connect_etcd(config: &BackupConfig) -> Result<etcd_client::Client> {
    let option = generate_etcd_connect_opt(&config.security).unwrap();
    etcd_client::Client::connect(&config.pd.endpoints, Some(option))
        .await
        .map_err(|err| {
            warn!("connect etcd failed"; "err" => ?err);
            Error::EtcdError(err)
        })
}

// Get keyspace meta from etcd and populate them to ClusterBackupMeta
async fn backup_pd_keyspace_meta(
    config: &BackupConfig,
    cluster_backup_meta: &mut ClusterBackupMeta,
) -> Result<()> {
    #[cfg(feature = "testexport")]
    if config.skip_keyspace_meta {
        warn!("skip backup keyspace meta");
        return Ok(());
    }

    info!("start backup PD keyspace meta"; "backup_meta" => %cluster_backup_meta);
    let cluster_id = cluster_backup_meta.cluster_id;

    let mut etcd_client = {
        let mut backoff = Duration::ZERO;
        try_wait_result_async(
            || Box::pin(connect_etcd(config)),
            Duration::from_secs(10),
            move || {
                backoff += Duration::from_millis(200);
                backoff
            },
        )
        .await?
    };

    let old_revision = cluster_backup_meta.meta_revision;
    // Keyspace meta will not be deleted even keyspace is deleted
    // So incremental backup can be used.
    let get_option = etcd_client::GetOptions::new()
        .with_from_key()
        .with_min_mod_revision(old_revision)
        .with_limit(MAX_BATCH_GET_CNT);
    let mut min_revision = i64::MAX;
    let mut new_meta_cnt = 0;
    // Only backup raw key-value pairs in etcd.
    // Content is not parsed as it's hard to align to the format with PD repo.
    // User cannot set placement rule in serverless cluster except the tiflash
    // replica. So all placement rules are created inner, backup all of them.
    for (path, prefixed) in PD_KEY_SPACE_META_PATH {
        let prefix = if prefixed {
            format!("/pd/{}/{}", cluster_id, path).as_bytes().to_owned()
        } else {
            path.as_bytes().to_owned()
        };
        let mut seek_key = prefix.clone();
        loop {
            let resp = etcd_client.get(seek_key, Some(get_option.clone())).await?;
            let mut more = resp.more();
            for kv in resp.kvs() {
                if !kv.key().starts_with(&prefix) {
                    more = false;
                    break;
                }
                new_meta_cnt += 1;
                cluster_backup_meta
                    .mut_keyspace_meta()
                    .insert(kv.key().to_vec(), kv.value().to_vec());
            }
            min_revision = std::cmp::min(min_revision, resp.header().map_or(0, |h| h.revision()));
            if !more {
                break;
            }
            seek_key = resp.kvs().last().unwrap().key().to_owned();
            seek_key.push(0); // exclude the last key
        }
    }
    if min_revision != i64::MAX {
        cluster_backup_meta.set_meta_revision(min_revision);
    }
    info!(
        "Backed up {} pd meta kvs, including {} new meta, revision: {} -> {}",
        cluster_backup_meta.keyspace_meta.len(),
        new_meta_cnt,
        old_revision,
        cluster_backup_meta.meta_revision
    );
    Ok(())
}

#[derive(Clone, Serialize, Deserialize, PartialEq, Debug)]
#[serde(default)]
#[serde(rename_all = "kebab-case")]
pub struct BackupConfig {
    pub pd: pd_client::Config,
    pub security: SecurityConfig,
    pub dfs: DFSConfig,
    pub tolerate_err: usize,
    pub timeout: ReadableDuration,
    pub backup_delay: ReadableDuration,
    #[cfg(feature = "testexport")]
    pub skip_keyspace_meta: bool,
}

impl Default for BackupConfig {
    fn default() -> Self {
        Self {
            pd: pd_client::Config::default(),
            security: SecurityConfig::default(),
            dfs: DFSConfig::default(),
            tolerate_err: 0,
            timeout: ReadableDuration::secs(30),
            backup_delay: ReadableDuration::ZERO,
            #[cfg(feature = "testexport")]
            skip_keyspace_meta: false,
        }
    }
}

/// Incremental Backup ID, name, and S3 path
///
/// Backup ID (`backup_id`) is the UNIX timestamp when the backup is created.
///
/// Backup Name is in the format of "YYYYmmdd/HHMMSS.meta", in which "YYYYmmdd"
/// & "HHMMSS" are the UTC data & time respectively, when the backup is created.
///
/// Backup S3 path is `/<s3_prefix>/backup/{{backup_name}}`.
///
/// Manual backups without name also follow this rule.

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IncrementalBackupFile {
    name: String,
    created_at: DateTime<Utc>,
}

impl IncrementalBackupFile {
    pub fn from_datetime(utc: DateTime<Utc>) -> Self {
        let name = Self::generate_name_from_utc(&utc);
        Self {
            name,
            created_at: utc,
        }
    }

    pub fn from_backup_ts(backup_ts: u64) -> Self {
        let physical_seconds = TimeStamp::from(backup_ts).physical() / 1000; // Unit of physical is millisecond.
        let datetime = NaiveDateTime::from_timestamp_opt(physical_seconds as i64, 0).unwrap();
        let utc = DateTime::<Utc>::from_utc(datetime, Utc);
        Self::from_datetime(utc)
    }

    pub fn try_from_full_path(path: &str) -> Option<Self> {
        lazy_static::lazy_static! {
            static ref RE: Regex = Regex::new(r"/backup/([0-9]{8})/([0-9]{6})\.meta$").unwrap();
        }
        let caps = RE.captures(path)?;
        let date = NaiveDate::parse_from_str(&caps[1], INCREMENTAL_BACKUP_FOLDER_FORMAT).ok()?;
        let time = NaiveTime::parse_from_str(&caps[2], INCREMENTAL_BACKUP_FILE_NAME_FORMAT).ok()?;
        let datetime = NaiveDateTime::new(date, time);
        let utc = DateTime::<Utc>::from_utc(datetime, Utc);
        Some(Self::from_datetime(utc))
    }

    pub fn from_id(backup_id: u64) -> Self {
        let datetime = NaiveDateTime::from_timestamp_opt(backup_id as i64, 0).unwrap();
        let utc = DateTime::<Utc>::from_utc(datetime, Utc);
        Self::from_datetime(utc)
    }

    pub fn id(&self) -> u64 {
        self.created_at.timestamp() as u64
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn into_name(self) -> String {
        self.name
    }

    pub fn created_at(&self) -> &DateTime<Utc> {
        &self.created_at
    }

    // `/<s3_prefix>/backup/{{backup_name}}`
    pub fn full_path(&self, s3_prefix: &str) -> String {
        Path::new(s3_prefix)
            .join("backup")
            .join(&self.name)
            .to_str()
            .unwrap()
            .to_string()
    }

    fn generate_name_from_utc(utc: &DateTime<Utc>) -> String {
        format!(
            "{}/{}.meta",
            utc.format(INCREMENTAL_BACKUP_FOLDER_FORMAT),
            utc.format(INCREMENTAL_BACKUP_FILE_NAME_FORMAT)
        )
    }
}

#[cfg(test)]
mod tests {
    use chrono::{DateTime, NaiveDateTime, Utc};
    use kvengine::dfs::{DFSConnOptions, Dfs};
    use kvproto::metapb::Store;
    use rfenginepb::{ChangeSet, ClusterBackupMeta, StoreBackupMeta, WalChunk};
    use test_cloud_server::oss::ObjectStorageService;
    use test_pd_client::TestPdClient;

    use super::*;

    #[test]
    fn test_merge_store_backup_meta() {
        let mut cluster_meta = ClusterBackupMeta::new();
        let mut store_meta = StoreBackupMeta::new();
        let store_id = 1;
        let wal_chunk_cnt = 3;
        store_meta.set_store_id(store_id);
        store_meta.set_manifest(ChangeSet::default());
        for i in 0..wal_chunk_cnt {
            store_meta.mut_wal_chunks().push(WalChunk {
                epoch: 1,
                start_off: i * 10,
                end_off: i * 20,
                ..Default::default()
            });
        }
        // cluster_meta is empty, store meta has manifest, it is added.
        merge_store_backup_meta(&mut cluster_meta, store_meta.clone());
        assert_eq!(cluster_meta.stores.len(), 1);
        assert_eq!(cluster_meta.stores.last().unwrap().clone(), store_meta);

        // Add a new store meta
        store_meta.set_store_id(store_id + 1);
        merge_store_backup_meta(&mut cluster_meta, store_meta.clone());
        assert_eq!(cluster_meta.stores.len(), 2);
        assert_eq!(cluster_meta.stores.last().unwrap().clone(), store_meta);

        store_meta.set_store_id(store_id);
        for chunk in store_meta.mut_wal_chunks().iter_mut() {
            chunk.epoch = 2;
        }
        // replace the old one.
        merge_store_backup_meta(&mut cluster_meta, store_meta.clone());
        assert_eq!(cluster_meta.stores.len(), 2);
        assert_eq!(cluster_meta.stores.last().unwrap().clone(), store_meta);

        // incremental backup, append wal chunks.
        store_meta.set_store_id(store_id);
        store_meta.clear_manifest();
        let mut store_meta = StoreBackupMeta::new();
        store_meta.set_store_id(store_id);
        for i in 0..wal_chunk_cnt {
            store_meta.mut_wal_chunks().push(WalChunk {
                epoch: 2,
                start_off: (i + wal_chunk_cnt) * 10,
                end_off: (i + wal_chunk_cnt) * 20,
                ..Default::default()
            });
        }
        merge_store_backup_meta(&mut cluster_meta, store_meta.clone());
        assert_eq!(cluster_meta.stores.len(), 2);
        let last_meta = cluster_meta.stores.last().unwrap();
        assert_eq!(last_meta.wal_chunks.len(), 2 * wal_chunk_cnt as usize);
        for (i, chunk) in last_meta.wal_chunks.iter().enumerate() {
            assert_eq!(chunk.epoch, 2);
            assert_eq!(chunk.start_off, i as u64 * 10);
            assert_eq!(chunk.end_off, i as u64 * 20);
        }
    }

    #[test]
    fn test_check_backup_meta_consistency() {
        let mut meta = ClusterBackupMeta::new();
        let mut stores = vec![];
        for i in 0..3 {
            stores.push(Store {
                id: i,
                ..Default::default()
            });
        }
        check_backup_meta_consistency(&meta, &stores).unwrap_err();

        for i in 0..3 {
            meta.mut_stores().push(StoreBackupMeta {
                store_id: i + 1,
                ..Default::default()
            });
        }
        check_backup_meta_consistency(&meta, &stores).unwrap_err();

        meta.clear_stores();
        for i in 0..3 {
            meta.mut_stores().push(StoreBackupMeta {
                store_id: i,
                ..Default::default()
            });
        }
        check_backup_meta_consistency(&meta, &stores).unwrap();

        stores.push(Store {
            id: 10,
            ..Default::default()
        });
        check_backup_meta_consistency(&meta, &stores).unwrap_err();
        meta.set_tolerated_err(1);
        check_backup_meta_consistency(&meta, &stores).unwrap();
        stores.first_mut().unwrap().id = 11;
        check_backup_meta_consistency(&meta, &stores).unwrap_err();
    }

    #[test]
    fn test_incremental_backup_file() {
        let datetime =
            NaiveDateTime::parse_from_str("2023-03-21 11:22:33", "%Y-%m-%d %H:%M:%S").unwrap();
        let utc = DateTime::<Utc>::from_utc(datetime, Utc);

        let backup = IncrementalBackupFile::from_datetime(utc);
        assert_eq!(backup.name(), "20230321/112233.meta");
        assert_eq!(backup.id(), 1679397753);
        let full_path = backup.full_path("cse-local");
        assert_eq!(&full_path, "cse-local/backup/20230321/112233.meta");
        assert_eq!(
            full_path,
            backup_file_full_path("cse-local".to_owned(), backup.name.to_owned(), None)
        );

        let backup1 = IncrementalBackupFile::try_from_full_path(&full_path).unwrap();
        assert_eq!(backup, backup1);

        let backup2 = IncrementalBackupFile::from_id(backup.id());
        assert_eq!(backup, backup2);

        let backup_ts = TimeStamp::compose(datetime.timestamp_millis() as u64, 0);
        let backup3 = IncrementalBackupFile::from_backup_ts(backup_ts.into_inner());
        assert_eq!(backup, backup3);
    }

    #[test]
    fn test_get_latest_backup_meta() {
        test_util::init_log_for_test();

        let base_dir = tempfile::Builder::new()
            .prefix("test_get_latest_backup_meta_")
            .tempdir()
            .unwrap();

        let mut oss = ObjectStorageService::new(base_dir.path());
        oss.start_server();

        let s3fs = S3Fs::new(
            "pfx".to_string(),
            format!("http://127.0.0.1:{}", oss.port()),
            "admin".to_string(),
            "admin".to_string(),
            "local".to_string(),
            "bkt".to_string(),
            DFSConnOptions::default(),
        );
        let pd_client = TestPdClient::new(1, false);

        // Use cluster_id to distinguish with different backup meta.
        const CLUSTER_ID_INCREMENTAL: u64 = 1;
        const CLUSTER_ID_MANUAL: u64 = 2;

        let prefix = s3fs.get_prefix();
        s3fs.get_runtime().block_on(async {
            {
                let backup_ts = pd_client.get_tso().await.unwrap().into_inner();
                let mut backup_meta = ClusterBackupMeta::new();
                backup_meta.set_cluster_id(CLUSTER_ID_INCREMENTAL);
                let backup_key =
                    backup_file_full_path(prefix.clone(), "".to_string(), Some(backup_ts));
                let backup_data = Bytes::from(backup_meta.write_to_bytes().unwrap());
                s3fs.put_object(backup_key.clone(), backup_data, backup_key)
                    .await
                    .unwrap();
            }

            {
                // Write manual backup file named as "check_table".
                // See https://github.com/tidbcloud/cloud-storage-engine/issues/867.
                let mut backup_meta = ClusterBackupMeta::new();
                backup_meta.set_cluster_id(CLUSTER_ID_MANUAL);
                let backup_key =
                    backup_file_full_path(prefix.clone(), "check_table".to_string(), None);
                let backup_data = Bytes::from(backup_meta.write_to_bytes().unwrap());
                s3fs.put_object(backup_key.clone(), backup_data, backup_key)
                    .await
                    .unwrap();
            }

            // `check_table` will be the latest one if we don't filter incremental backup
            // metas.
            let (objects, ..) = s3fs.list("", Some("backup/"), None).await.unwrap();
            assert_eq!(objects.len(), 2);
            assert_eq!(
                objects.last().unwrap().key,
                format!("{}/backup/check_table", prefix)
            );

            let _ = get_latest_backup_meta(&s3fs, CLUSTER_ID_INCREMENTAL)
                .await
                .unwrap();
        });

        oss.shutdown();
    }
}
