// Copyright 2023 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    path::Path,
    str::FromStr,
    sync::mpsc::SyncSender,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use bytes::Bytes;
use chrono::{DateTime, NaiveDate, NaiveDateTime, NaiveTime, Utc};
use futures::{compat::Stream01CompatExt, executor::block_on, StreamExt};
use http::{Request, Uri};
use hyper::Body;
use kvengine::dfs::{self, DFSConfig, S3Fs};
use kvproto::metapb::Store;
use pd_client::PdClient;
use protobuf::Message;
use regex::Regex;
use rfenginepb::{ClusterBackupMeta, StoreBackupMeta};
use security::SecurityConfig;
use slog_global::{error, info, warn};
use tikv_util::timer::GLOBAL_TIMER_HANDLE;

use crate::{
    common::{
        create_pd_client, generate_etcd_connect_opt, get_all_stores_except_tiflash,
        send_request_to_store,
    },
    error::Error,
};

const MAX_BATCH_GET_CNT: i64 = 1024;
// keyspace meta in pd is "/pd/$cluster_id/PD_KEYSPACE_META_PATH"
const PD_KEY_SPACE_META_PATH: [&str; 4] = [
    "keyspaces/",
    "region_label/keyspaces/",
    "rules/",
    "tso/keyspace_groups/membership/",
];

pub type Result<T> = std::result::Result<T, Error>;

/// Generate full path `/<prefix>/backup/<name>`.
pub fn backup_file_full_path(prefix: String, name: String) -> String {
    let name = if name.is_empty() {
        IncrementalBackupFile::default().into_name()
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

pub fn execute_incremental_backup(config: BackupConfig, name: String, interval: Duration) {
    if !name.is_empty() {
        error!("Don't support non-empty name for incremental backup.");
        return;
    }
    let duration = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
    let gap = interval.as_secs() - duration.as_secs() % interval.as_secs();
    let start_time = Instant::now()
        .checked_add(Duration::from_secs(gap))
        .unwrap();
    let mut interval = GLOBAL_TIMER_HANDLE.interval(start_time, interval).compat();
    let pd_client = create_pd_client(&config.security, &config.pd);
    let mut cluster_backup_meta = None;
    while let Some(Ok(_)) = block_on(interval.next()) {
        match backup_cluster(
            config.clone(),
            true,
            name.clone(),
            &pd_client,
            cluster_backup_meta.clone(),
        ) {
            Ok(meta) => {
                cluster_backup_meta = Some(meta);
            }
            Err(e) => {
                // For other errors, retry incremental backup later.
                if need_full_backup(&e) {
                    warn!("Incremental backup fails {:?}, fallback to full backup", e);
                    // If incremental backup fails, restart full backup automatically.
                    match backup_cluster(config.clone(), false, name.clone(), &pd_client, None) {
                        Ok(meta) => cluster_backup_meta = Some(meta),
                        Err(e) => {
                            error!("Full backup still fail {:?}", e);
                            return;
                        }
                    }
                } else {
                    warn!("Incremental backup fails {:?}, retry later", e);
                }
            }
        }
    }
}

pub fn execute_full_backup(config: BackupConfig, name: String) {
    let pd_client = create_pd_client(&config.security, &config.pd);
    if let Err(e) = backup_cluster(config, false, name, &pd_client, None) {
        error!("Full backup fail, {:?}", e)
    }
}

fn backup_cluster(
    config: BackupConfig,
    incremental: bool,
    name: String,
    pd_client: &dyn PdClient,
    last_backup_meta: Option<ClusterBackupMeta>,
) -> Result<ClusterBackupMeta> {
    let backup_ts = block_on(pd_client.get_tso())?.into_inner();
    backup_cluster_with_ts(
        config,
        incremental,
        name,
        pd_client,
        backup_ts,
        last_backup_meta,
    )
}

pub fn backup_cluster_with_ts(
    config: BackupConfig,
    incremental: bool,
    name: String,
    pd_client: &dyn PdClient,
    backup_ts: u64,
    last_backup_meta: Option<ClusterBackupMeta>,
) -> Result<ClusterBackupMeta> {
    let stores = get_all_stores_except_tiflash(pd_client)?;
    let cluster_id = pd_client.get_cluster_id()?;

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(8)
        .enable_all()
        .build()
        .unwrap();

    let dfs_conf = config.dfs.clone();
    let s3fs = S3Fs::new(
        dfs_conf.prefix,
        dfs_conf.s3_endpoint,
        dfs_conf.s3_key_id,
        dfs_conf.s3_secret_key,
        dfs_conf.s3_region,
        dfs_conf.s3_bucket,
    );
    let mut cluster_backup_meta = if let Some(meta) = last_backup_meta {
        meta
    } else if !incremental {
        ClusterBackupMeta::new()
    } else {
        // If no input backup meta, load latest one from s3.
        let meta = runtime.block_on(get_latest_backup_meta(&s3fs, cluster_id))?;
        check_backup_meta_consistency(&meta, &stores)?;
        meta
    };
    cluster_backup_meta.set_backup_ts(backup_ts);
    cluster_backup_meta.set_cluster_id(cluster_id);

    if !config.skip_keyspace_meta {
        runtime.block_on(backup_pd_keyspace_meta(&config, &mut cluster_backup_meta))?;
    }

    let num_stores = stores.len();
    let (tx, rx) = std::sync::mpsc::sync_channel(num_stores);
    for store in stores {
        let config = get_backup_config(&cluster_backup_meta, cluster_id, store.id, incremental);
        runtime.spawn(backup_store(config, store, tx.clone()));
    }
    let mut errs = vec![];
    for _ in 0..num_stores {
        match rx.recv().unwrap() {
            Ok(store_backup_meta) => {
                merge_store_backup_meta(&mut cluster_backup_meta, store_backup_meta);
            }
            Err(err) => {
                errs.push(err);
            }
        }
    }
    if !errs.is_empty() {
        error!(
            "backup errors {:?}, tolerance {}",
            errs, config.tolerate_err
        );
        if errs.len() > config.tolerate_err {
            return Err(errs.pop().unwrap());
        }
    }
    let alloc_id = pd_client.alloc_id()?;
    let safe_ts = runtime.block_on(pd_client.get_gc_safe_point())?;
    if safe_ts > backup_ts {
        return Err(Error::TsError(safe_ts, backup_ts));
    }
    cluster_backup_meta.set_alloc_id(alloc_id);
    cluster_backup_meta.set_safe_ts(safe_ts);
    let stores = get_all_stores_except_tiflash(pd_client)?;
    check_backup_meta_consistency(&cluster_backup_meta, &stores)?;
    info!(
        "cluster backup cluster_id:{}, backup_ts:{}, alloc_id:{}, safe_ts:{}, num_stores:{}",
        cluster_backup_meta.cluster_id,
        cluster_backup_meta.backup_ts,
        cluster_backup_meta.alloc_id,
        cluster_backup_meta.safe_ts,
        cluster_backup_meta.get_stores().len(),
    );
    let backup_key = backup_file_full_path(s3fs.get_prefix(), name);
    let backup_data = Bytes::from(cluster_backup_meta.write_to_bytes().unwrap());
    runtime
        .block_on(s3fs.put_object(backup_key.clone(), backup_data, backup_key.clone()))
        .unwrap();
    info!("finished build backup file {}", backup_key);
    Ok(cluster_backup_meta)
}

async fn backup_store(
    config: rfengine::BackupConfig,
    store: Store,
    tx: SyncSender<Result<StoreBackupMeta>>,
) {
    let uri = Uri::from_str(&format!("http://{}/rfengine/backup", &store.status_address)).unwrap();
    info!("Start backup with config {}", config);
    let json_string = serde_json::to_string(&config).unwrap();
    let req = Request::post(uri).body(Body::from(json_string)).unwrap();
    match send_request_to_store(req, &store).await {
        Ok(resp) => {
            let mut store_backup_meta = StoreBackupMeta::default();
            store_backup_meta.merge_from_bytes(&resp).unwrap();
            tx.send(Ok(store_backup_meta)).unwrap()
        }
        Err(e) => tx.send(Err(e)).unwrap(),
    }
}

fn merge_store_backup_meta(
    cluster_backup_meta: &mut ClusterBackupMeta,
    store_backup_meta: StoreBackupMeta,
) {
    // only full backup has manifest
    if store_backup_meta.has_manifest() {
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
    incremental: bool,
) -> rfengine::BackupConfig {
    let mut config = rfengine::BackupConfig {
        cluster_id,
        store_id,
        incremental,
        wal_epoch: 0,
        start_offset: 0,
    };
    if incremental {
        let store_meta = backup_meta
            .get_stores()
            .iter()
            .find(|s| s.get_store_id() == store_id)
            .unwrap(); // store id existence is checked in check_backup_meta_consistency
        let last_wal = store_meta.get_wal_chunks().last().unwrap();
        config.wal_epoch = last_wal.epoch;
        config.start_offset = last_wal.get_end_off();
    }
    config
}

/// Return full path of incremental backups in S3.
pub async fn get_all_incremental_backups(
    s3fs: &S3Fs,
    start_date: &chrono::Date<Utc>,
    start_time: Option<&NaiveTime>,
    max_count: usize,
) -> dfs::Result<(Vec<String>, bool)> {
    let mut files = Vec::with_capacity(std::cmp::min(max_count, 1000));
    let mut start_key = format!(
        "{}/{}",
        start_date.format(INCREMENTAL_BACKUP_FOLDER_FORMAT),
        start_time
            .map(|t| t.format(INCREMENTAL_BACKUP_FILE_NAME_FORMAT).to_string())
            .unwrap_or_default()
    );
    let prefix = "backup/";

    let mut reach_limit = false;
    loop {
        match s3fs.list(&start_key, Some(prefix)).await {
            Ok((backup_files, more, next_start_after)) => {
                let mut file_keys = backup_files.into_iter().map(|f| f.key).collect::<Vec<_>>();
                files.append(&mut file_keys);
                if files.len() > max_count {
                    files.truncate(max_count);
                    reach_limit = true;
                }
                if reach_limit || !more {
                    break;
                }
                start_key = next_start_after.unwrap();
            }
            Err(e) => {
                return Err(e);
            }
        }
    }
    Ok((files, reach_limit))
}

// If backup exist, return the latest one, else create a new ClusterBackupMeta.
async fn get_latest_backup_meta(s3fs: &S3Fs, cluster_id: u64) -> Result<ClusterBackupMeta> {
    let now = Utc::now();
    let (files, _) = get_all_incremental_backups(s3fs, &now.date(), None, usize::MAX).await?;
    if files.is_empty() {
        return Err(Error::MetaNotFound(cluster_id));
    }
    // Incremental backup file name is generated with `backup_file_full_path`.
    // The last should be the latest one in most cases.
    let last_file = files.last().unwrap();
    let object = s3fs
        .get_object(
            last_file.clone(),
            last_file.clone(),
            engine_traits::GetObjectOptions::default(),
        )
        .await?;
    let mut meta = ClusterBackupMeta::new();
    meta.merge_from_bytes(&object).unwrap();
    if meta.cluster_id != cluster_id {
        return Err(Error::MetaNotFound(cluster_id));
    }
    info!(
        "Get cluster {} latest backup meta {}, store cnt {}",
        meta.cluster_id,
        last_file,
        meta.stores.len()
    );
    Ok(meta)
}

fn check_backup_meta_consistency(backup_meta: &ClusterBackupMeta, stores: &[Store]) -> Result<()> {
    if stores.len() != backup_meta.stores.len() {
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
    if remain_stores.is_empty() {
        Ok(())
    } else {
        Err(Error::TopoChanged(format!(
            "Check backup meta fails, have no meta for store: {:?}",
            remain_stores
        )))
    }
}

fn need_full_backup(err: &Error) -> bool {
    matches!(err, Error::TopoChanged(_) | Error::MetaNotFound(_))
}

// Get keyspace meta from etcd and populate them to ClusterBackupMeta
async fn backup_pd_keyspace_meta(
    config: &BackupConfig,
    cluster_backup_meta: &mut ClusterBackupMeta,
) -> Result<()> {
    let cluster_id = cluster_backup_meta.cluster_id;
    let option = generate_etcd_connect_opt(&config.security).unwrap();
    let mut etcd_client = etcd_client::Client::connect(&config.pd.endpoints, Some(option)).await?;
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
    for path in PD_KEY_SPACE_META_PATH {
        let prefix = format!("/pd/{}/{}", cluster_id, path).as_bytes().to_owned();
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

#[derive(Clone, Serialize, Deserialize, PartialEq, Debug, Default)]
#[serde(default)]
#[serde(rename_all = "kebab-case")]
pub struct BackupConfig {
    pub pd: pd_client::Config,
    pub security: SecurityConfig,
    pub dfs: DFSConfig,
    pub tolerate_err: usize,
    pub skip_keyspace_meta: bool,
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

const INCREMENTAL_BACKUP_FOLDER_FORMAT: &str = "%Y%m%d";
const INCREMENTAL_BACKUP_FILE_NAME_FORMAT: &str = "%H%M%S";

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IncrementalBackupFile {
    name: String,
    created_at: DateTime<Utc>,
}

impl IncrementalBackupFile {
    pub fn new() -> Self {
        let created_at = Utc::now();
        Self::from_datetime(created_at)
    }

    pub fn from_datetime(utc: DateTime<Utc>) -> Self {
        let name = Self::generate_name_from_utc(&utc);
        Self {
            name,
            created_at: utc,
        }
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
        let datetime = NaiveDateTime::from_timestamp(backup_id as i64, 0);
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

impl Default for IncrementalBackupFile {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use chrono::{DateTime, NaiveDateTime, Utc};
    use kvproto::metapb::Store;
    use rfenginepb::{ChangeSet, ClusterBackupMeta, StoreBackupMeta, WalChunk};

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
            backup_file_full_path("cse-local".to_owned(), backup.name.to_owned())
        );

        let backup1 = IncrementalBackupFile::try_from_full_path(&full_path).unwrap();
        assert_eq!(backup, backup1);

        let backup2 = IncrementalBackupFile::from_id(backup.id());
        assert_eq!(backup, backup2);
    }
}
