// Copyright 2023 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    collections::{HashMap, HashSet},
    path::PathBuf,
    sync::Arc,
    time::Duration,
};

use bstr::ByteSlice;
use bytes::{Buf, BufMut, Bytes};
use chrono::NaiveDate;
use engine_traits::GetObjectOptions;
use kvengine::dfs::{self, DFSConfig, Dfs, Options, S3Fs, STORAGE_CLASS_GLACIER_IR};
use kvproto::keyspacepb::{KeyspaceMeta, KeyspaceState};
use pd_client::PdClient;
use protobuf::Message;
use rfenginepb::ClusterBackupMeta;
use security::SecurityConfig;
use tikv_util::{error, info, mpsc::Receiver, time::Instant, warn};

use crate::{
    backup::{backup_file_full_path, IncrementalBackupFile},
    common::{create_pd_client, INCREMENTAL_BACKUP_FOLDER_FORMAT},
    error::{Error, Result},
    restore_keyspace::BackupCluster,
};

pub const DEFAULT_MAX_ARCHIVE_FILE_SIZE: u64 = 1024 * 1024 * 1024;
const ARCHIVE_PATH_PREFIX: &str = "archive";

pub const LOAD_FILE_CONCURRENCY: usize = 256;
pub const OBJECT_ADDR_SIZE: usize = 20;

/// Magic Number of the Archive index file. It's picked by running
///    echo archive.idx | sha1sum
/// and taking the leading 32 bits.
const ARCHIVE_MAGIC_NUMBER: u32 = 0x923d4deb;

pub const ARCHIVE_INDEX_FORMAT_V1: u32 = 1;

// Object address format:
//
// +---------------------------------+
// |         package id: u32         |
// +---------------------------------+
// |           offset: u64           |
// +---------------------------------+
// |           length: u64           |
// +---------------------------------+
//
// Archive index format:
//
// +---------------------------------+
// |        magic number: u32        |
// +---------------------------------+
// |          version: u32           |
// +---------------------------------+
// |   backup meta object address    |
// +---------------------------------+
// |      num of file ids: u32       |
// +---------------------------------+
// |            file id 1            |
// +---------------------------------+
// |            file id 2            |
// +---------------------------------+
// |             ...                 |
// +---------------------------------+
// |            file id n            |
// +---------------------------------+
// |       sst object address 1      |
// +---------------------------------+
// |       sst object address 2      |
// +---------------------------------+
// |             ...                 |
// +---------------------------------+
// |       sst object address n      |
// +---------------------------------+
//
// Archive package format:
//
// +---------------------------------+
// |          object file 1          |
// +---------------------------------+
// |          object file 2          |
// +---------------------------------+
// |             ...                 |
// +---------------------------------+
// |          object file n          |
// +---------------------------------+

pub fn archive_with_cfg(config: ArchiveConfig) {
    let pd_client = Arc::new(create_pd_client(&config.security, &config.pd));
    let dfs_conf = config.dfs.clone();
    let s3fs = Arc::new(S3Fs::new(
        dfs_conf.prefix,
        dfs_conf.s3_endpoint,
        dfs_conf.s3_key_id,
        dfs_conf.s3_secret_key,
        dfs_conf.s3_region,
        dfs_conf.s3_bucket,
    ));
    let expiration_date = NaiveDate::parse_from_str(
        config.expiration_date.as_str(),
        INCREMENTAL_BACKUP_FOLDER_FORMAT,
    )
    .unwrap();
    let start_archive_duration = chrono::Duration::from_std(config.start_archive_duration).unwrap();
    let end_archive_date = (chrono::Utc::now() - start_archive_duration).date_naive();
    if expiration_date >= end_archive_date {
        panic!(
            "start archive duration is invalid. expiration_date {}, start_archive_duration {}",
            expiration_date, start_archive_duration
        );
    }
    let begin_archive_date = expiration_date
        .checked_add_days(chrono::Days::new(1))
        .unwrap();
    if let Err(e) = archive_cluster_backup(
        config,
        pd_client,
        s3fs,
        begin_archive_date,
        end_archive_date,
        None,
    ) {
        error!("failed to archive cluster backup, err {:?}", e)
    }
}

#[derive(Clone, Serialize, Deserialize, PartialEq, Debug, Default)]
#[serde(default)]
#[serde(rename_all = "kebab-case")]
pub struct ArchiveConfig {
    pub pd: pd_client::Config,
    pub security: SecurityConfig,
    pub dfs: DFSConfig,
    pub data_dir: String,
    pub max_archive_file_size: u64,
    pub start_archive_duration: Duration,
    pub expiration_date: String,
    pub dry_run: bool,
}

impl ArchiveConfig {
    pub fn default() -> Self {
        let expiration_date_naive = chrono::Utc::now().date_naive() - chrono::Days::new(1);
        let expiration_date = archive_format_date(&expiration_date_naive);
        Self {
            pd: pd_client::Config::default(),
            security: SecurityConfig::default(),
            dfs: DFSConfig::default(),
            data_dir: String::default(),
            max_archive_file_size: DEFAULT_MAX_ARCHIVE_FILE_SIZE,
            start_archive_duration: Duration::from_secs(0),
            expiration_date,
            dry_run: true,
        }
    }

    pub fn check_data_dir(&mut self) {
        if self.data_dir.is_empty() {
            let data_dir = tempdir::TempDir::new(ARCHIVE_PATH_PREFIX).unwrap();
            let data_path = data_dir.path().to_path_buf();
            self.data_dir = data_path.to_str().unwrap().to_string();
        }
    }
}

#[derive(Default, Clone)]
pub struct ArchiveBackup {
    pub date: NaiveDate,
    pub meta_data: Bytes,
    pub files: HashSet<u64>,
}

impl ArchiveBackup {
    pub fn new(date: NaiveDate, meta_data: Bytes, files: HashSet<u64>) -> Self {
        Self {
            date,
            meta_data,
            files,
        }
    }
}

pub fn archive_cluster_backup(
    config: ArchiveConfig,
    pd_client: Arc<dyn PdClient>,
    s3fs: Arc<S3Fs>,
    begin_archive_date: NaiveDate,
    end_archive_date: NaiveDate,
    keyspace_ids: Option<Vec<u32>>,
) -> Result<()> {
    let cluster_id = pd_client.get_cluster_id()?;
    let mut backup_date = match get_latest_archive_date(&s3fs, &begin_archive_date) {
        Ok(latest_archive_date) => latest_archive_date
            .checked_add_days(chrono::Days::new(1))
            .unwrap(),
        Err(Error::DfsError(e)) => return Err(Error::DfsError(e)),
        Err(e) => {
            warn!(
                "failed to get latest archive date from date {}, err {}",
                begin_archive_date,
                e.to_string()
            );
            begin_archive_date
        }
    };

    info!(
        "archive from date {} to end date {}",
        backup_date, end_archive_date,
    );

    let path = PathBuf::from(&config.data_dir).join("stores");
    let mut old: Option<ArchiveBackup> = None;
    loop {
        if backup_date > end_archive_date {
            break;
        }
        let new = archive_backup_files(
            config.clone(),
            pd_client.clone(),
            s3fs.clone(),
            cluster_id,
            path.clone(),
            backup_date,
            old,
            keyspace_ids.clone(),
        )?;
        old = Some(new);
        std::fs::remove_dir_all(&path).unwrap();
        backup_date += chrono::Duration::days(1);
    }
    Ok(())
}

fn archive_backup_files(
    config: ArchiveConfig,
    pd_client: Arc<dyn PdClient>,
    s3fs: Arc<S3Fs>,
    cluster_id: u64,
    path: PathBuf,
    backup_date: NaiveDate,
    old: Option<ArchiveBackup>,
    keyspace_ids: Option<Vec<u32>>,
) -> Result<ArchiveBackup> {
    let runtime = s3fs.get_runtime();
    let (backups, _) = runtime.block_on(get_daily_incremental_backups(&s3fs, &backup_date, 1))?;
    if backups.is_empty() {
        return Err(Error::ArchiveError(format!(
            "failed to get first backup meta on {}",
            backup_date,
        )));
    }
    let back_file = backups.first().unwrap();
    let file_name = back_file.name().to_string();
    let (meta_file_data, cluster_backup) =
        get_cluster_backup_file_and_meta(&s3fs, file_name.clone())
            .map_err(|e| Error::DfsError(e))?;
    let files = get_cluster_backup_files(
        pd_client.clone(),
        s3fs.clone(),
        cluster_id,
        file_name,
        cluster_backup,
        path,
        config.security.clone(),
        keyspace_ids,
    )?;
    if let Some(old_archive_backup) = old {
        if old_archive_backup
            .date
            .checked_add_days(chrono::Days::new(1))
            .unwrap()
            .eq(&backup_date)
        {
            write_archive_packages_and_index(config, s3fs, old_archive_backup, &files)?
        } else {
            return Err(Error::ArchiveError(format!(
                "old backup {} is not {}'s last day",
                old_archive_backup.date, backup_date,
            )));
        }
    }
    Ok(ArchiveBackup::new(backup_date, meta_file_data, files))
}

fn write_archive_packages_and_index(
    config: ArchiveConfig,
    s3fs: Arc<S3Fs>,
    archive_backup: ArchiveBackup,
    next_day_files: &HashSet<u64>,
) -> Result<()> {
    let deleted = get_sorted_deleted_files(&archive_backup.files, next_day_files);
    info!(
        "get deleted files {} on {}",
        deleted.len(),
        archive_backup.date,
    );
    let format_date = archive_format_date(&archive_backup.date);
    if !config.dry_run {
        let mut writer = ArchiveWriter::new(
            config.max_archive_file_size,
            s3fs,
            format_date,
            archive_backup.meta_data,
        );
        writer.append_files(deleted)?;
        writer.finish();
    }
    Ok(())
}

fn get_sorted_deleted_files(old: &HashSet<u64>, new: &HashSet<u64>) -> Vec<u64> {
    let mut deleted: Vec<u64> = old.iter().filter(|x| !new.contains(*x)).copied().collect();
    deleted.sort();
    deleted
}

fn get_cluster_backup_files(
    pd_client: Arc<dyn PdClient>,
    s3fs: Arc<S3Fs>,
    cluster_id: u64,
    backup_name: String,
    cluster_backup: ClusterBackupMeta,
    path: PathBuf,
    security_conf: SecurityConfig,
    keyspace_ids: Option<Vec<u32>>,
) -> Result<HashSet<u64>> {
    if cluster_backup.cluster_id != cluster_id {
        return Err(Error::ArchiveError(format!(
            "cluster id not match, pd cluster id {}, meta cluster id {}",
            cluster_id, cluster_backup.cluster_id,
        )));
    }
    let start_time = Instant::now();
    let keyspace_ids = if let Some(keyspace_ids) = keyspace_ids {
        keyspace_ids
    } else {
        let mut keyspace_ids = vec![];
        for (k, v) in &cluster_backup.keyspace_meta {
            let key_str = String::from_utf8_lossy(k);
            if key_str.contains("/keyspaces/meta/") {
                let mut keyspace_meta = KeyspaceMeta::default();
                let res = keyspace_meta.merge_from_bytes(v);
                if res.is_ok() && keyspace_meta.state == KeyspaceState::Enabled {
                    keyspace_ids.push(keyspace_meta.id);
                }
            }
        }
        keyspace_ids
    };

    info!("keyspace ids {}", keyspace_ids.len());
    if keyspace_ids.is_empty() {
        return Err(Error::ArchiveError("keyspace ids is empty".to_string()));
    }
    let keyspace_id = keyspace_ids[0];
    let mut cluster = BackupCluster::new(
        &cluster_backup,
        path,
        pd_client.clone(),
        s3fs,
        security_conf,
        keyspace_id,
        keyspace_id,
        cluster_backup.backup_ts,
        true,
        None,
    )?;
    let mut all_files = HashSet::new();
    collect_all_files(&mut all_files, &cluster, keyspace_id);
    for i in 1..keyspace_ids.len() {
        let keyspace_id = keyspace_ids[i];
        match cluster.reset_keyspace(&cluster_backup, keyspace_id, keyspace_id) {
            Ok(_) => {}
            Err(e) => {
                return Err(Error::ArchiveError(format!(
                    "failed to reset keyspace {}, err {}",
                    keyspace_id, e
                )));
            }
        };
        collect_all_files(&mut all_files, &cluster, keyspace_id);
    }
    drop(cluster);
    info!(
        "backup {} all_files {} takes {:?}",
        backup_name,
        all_files.len(),
        start_time.saturating_elapsed()
    );
    Ok(all_files)
}

fn collect_all_files(all_files: &mut HashSet<u64>, cluster: &BackupCluster, keyspace_id: u32) {
    let files = cluster.get_all_shard_files();
    info!("keyspace id {} files {}", keyspace_id, files.len());
    all_files.extend(files.into_iter());
}

/// Return full path of daily incremental backups in S3.
pub async fn get_daily_incremental_backups(
    s3fs: &S3Fs,
    date: &chrono::NaiveDate,
    max_count: usize,
) -> dfs::Result<(Vec<IncrementalBackupFile>, bool)> {
    let mut files = Vec::with_capacity(std::cmp::min(max_count, 1000));
    let mut start_key = "".to_string();
    let prefix = format!("backup/{}/", date.format(INCREMENTAL_BACKUP_FOLDER_FORMAT));
    let mut reach_limit = false;
    loop {
        match s3fs.list(&start_key, Some(&prefix), None).await {
            Ok((backup_files, more, next_start_after)) => {
                let mut inc_files = backup_files
                    .into_iter()
                    .filter_map(|f| IncrementalBackupFile::try_from_full_path(&f.key))
                    .collect::<Vec<_>>();
                files.append(&mut inc_files);
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

pub fn get_incremental_backup_with_name(prefix: String, name: String) -> IncrementalBackupFile {
    let backup_key = backup_file_full_path(prefix, name, None);
    IncrementalBackupFile::try_from_full_path(&backup_key).unwrap()
}

pub fn get_cluster_backup_file_and_meta(
    s3fs: &S3Fs,
    name: String,
) -> dfs::Result<(Bytes, ClusterBackupMeta)> {
    let backup_key = backup_file_full_path(s3fs.get_prefix(), name.clone(), None);
    let runtime = s3fs.get_runtime();
    let data = runtime.block_on(s3fs.get_object(
        backup_key.clone(),
        name,
        engine_traits::GetObjectOptions::default(),
    ))?;
    let mut cluster_backup = ClusterBackupMeta::new();
    cluster_backup.merge_from_bytes(&data).map_err(|e| {
        dfs::Error::Other(format!(
            "Incorrect encoded data from s3 {}, err {}",
            backup_key, e
        ))
    })?;
    info!(
        "Restore cluster_id {}, alloc_id {}, backup_ts {}, safe_ts {}, store cnt {}",
        cluster_backup.cluster_id,
        cluster_backup.alloc_id,
        cluster_backup.backup_ts,
        cluster_backup.safe_ts,
        cluster_backup.stores.len()
    );
    Ok((data, cluster_backup))
}

pub async fn get_all_archive_index_paths(
    s3fs: &S3Fs,
    start_date: String,
    max_count: usize,
) -> dfs::Result<(Vec<String>, bool)> {
    let mut indexes = Vec::with_capacity(std::cmp::min(max_count, 1000));
    let mut start_key = start_date;
    let prefix = "archive/index/";

    let mut reach_limit = false;
    loop {
        // TODO: pass in `max_count` for limit.
        match s3fs.list(&start_key, Some(prefix), None).await {
            Ok((archive_indexes, more, next_start_after)) => {
                let mut inc_indexes = archive_indexes
                    .into_iter()
                    .map(|f| f.key)
                    .collect::<Vec<_>>();
                indexes.append(&mut inc_indexes);
                if indexes.len() > max_count {
                    indexes.truncate(max_count);
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
    Ok((indexes, reach_limit))
}

pub fn get_latest_archive_date(s3fs: &S3Fs, start_date: &chrono::NaiveDate) -> Result<NaiveDate> {
    let (indexes, _) = s3fs
        .get_runtime()
        .block_on(get_all_archive_index_paths(
            s3fs,
            archive_format_date(start_date),
            usize::MAX,
        ))
        .map_err(|e| {
            error!(
                "failed to get latest archive date from date {}, err {}",
                start_date,
                e.to_string()
            );
            Error::DfsError(e)
        })?;
    if indexes.is_empty() {
        return Err(Error::ArchiveError(format!(
            "No archives have been found since that day {}",
            start_date
        )));
    }
    let latest_archive_date = NaiveDate::parse_from_str(
        parse_index_date(indexes.last().unwrap()).as_str(),
        INCREMENTAL_BACKUP_FOLDER_FORMAT,
    )
    .ok()
    .unwrap();
    Ok(latest_archive_date)
}

pub fn get_archive_index(s3fs: &S3Fs, date: String) -> Result<(ArchiveIndex, Bytes)> {
    let index_key = archive_index_key(s3fs.get_prefix(), date.clone());
    let data = s3fs
        .get_runtime()
        .block_on(s3fs.get_object(
            index_key.clone(),
            index_key.clone(),
            GetObjectOptions::default(),
        ))
        .map_err(|e| {
            error!(
                "failed to get archived index with date {}, err {}",
                date,
                e.to_string()
            );
            Error::DfsError(e)
        })?;
    let archive_index = ArchiveIndex::unmarshal(&data).map_err(|e| {
        dfs::Error::Other(format!(
            "Incorrect archive index data from s3 {}, err {}",
            index_key, e
        ))
    })?;
    Ok((archive_index, data))
}

pub async fn get_archived_object(s3fs: &S3Fs, archive_addr: ArchiveAddress) -> Result<Bytes> {
    let package_key = archive_package_key(
        s3fs.get_prefix(),
        archive_addr.date.clone(),
        archive_addr.object_addr.package_id,
    );
    let opts = GetObjectOptions {
        start_off: archive_addr.object_addr.offset,
        end_off: Some(archive_addr.object_addr.offset + archive_addr.object_addr.length),
    };
    return s3fs
        .get_object(package_key.clone(), package_key, opts)
        .await
        .map_err(|e| {
            error!(
                "failed to get archived object with archive addr {:?}, err {}",
                archive_addr,
                e.to_string()
            );
            Error::DfsError(e)
        });
}

pub fn get_not_found_files(s3fs: &S3Fs, file_ids: Vec<u64>) -> Result<Vec<u64>> {
    let (result_tx, result_rx) = tikv_util::mpsc::bounded(file_ids.len());
    let mut not_found_files = Vec::default();
    let recv_sst_existence = |not_found_files: &mut Vec<u64>,
                              result_tx: &Receiver<dfs::Result<(u64, bool)>>|
     -> Result<()> {
        let (file_id, exist) = result_tx.recv().unwrap()?;
        if !exist {
            not_found_files.push(file_id);
        }
        Ok(())
    };
    let mut msg_count = 0;
    for file_id in file_ids {
        let dfs = s3fs.clone();
        let tx = result_tx.clone();
        s3fs.get_runtime().spawn(async move {
            let res = dfs
                .exist(dfs.file_key(file_id), format!("{}", file_id))
                .await;
            let _ = tx.send(res.map(|exist| (file_id, exist)));
        });
        if msg_count < LOAD_FILE_CONCURRENCY {
            msg_count += 1;
        } else {
            recv_sst_existence(&mut not_found_files, &result_rx)?;
        }
    }
    for _ in 0..msg_count {
        recv_sst_existence(&mut not_found_files, &result_rx)?;
    }
    Ok(not_found_files)
}

#[derive(Default, Clone, Copy, Debug)]
pub struct ObjectAddress {
    pub package_id: u32,
    pub offset: u64,
    pub length: u64,
}

impl ObjectAddress {
    pub fn new(package_id: u32, offset: u64, length: u64) -> Self {
        Self {
            package_id,
            offset,
            length,
        }
    }

    pub fn marshal(&self, buf: &mut Vec<u8>) {
        buf.put_u32_le(self.package_id);
        buf.put_u64_le(self.offset);
        buf.put_u64_le(self.length);
    }

    pub fn unmarshal(mut buf: &[u8]) -> Self {
        let package_id = buf.get_u32_le();
        let offset = buf.get_u64_le();
        let length = buf.get_u64_le();
        Self {
            package_id,
            offset,
            length,
        }
    }
}

#[derive(Default)]
pub struct ArchiveIndex {
    meta_address: ObjectAddress,
    sst_file_ids: Vec<u64>,
    sst_addresses: Vec<ObjectAddress>,
}

#[allow(dead_code)]
impl ArchiveIndex {
    pub fn new(meta_address: ObjectAddress) -> Self {
        Self {
            meta_address,
            sst_file_ids: Vec::new(),
            sst_addresses: Vec::new(),
        }
    }

    pub fn append(&mut self, file_id: u64, file_address: ObjectAddress) {
        self.sst_file_ids.push(file_id);
        self.sst_addresses.push(file_address);
    }

    pub fn marshal(&self, buf: &mut Vec<u8>) {
        buf.put_u32_le(ARCHIVE_MAGIC_NUMBER);
        buf.put_u32_le(ARCHIVE_INDEX_FORMAT_V1);
        self.meta_address.marshal(buf);
        let num_file_ids = self.sst_file_ids.len();
        buf.put_u32_le(num_file_ids as u32);
        for i in 0..num_file_ids {
            buf.put_u64_le(self.sst_file_ids[i]);
        }
        for i in 0..num_file_ids {
            self.sst_addresses[i].marshal(buf);
        }
    }

    pub fn unmarshal(mut buf: &[u8]) -> Result<Self> {
        let magic_number = buf.get_u32_le();
        if magic_number != ARCHIVE_MAGIC_NUMBER {
            return Err(Error::ArchiveError(
                "archive magic number mismatch".to_owned(),
            ));
        }
        let version = buf.get_u32_le();
        if version != ARCHIVE_INDEX_FORMAT_V1 {
            return Err(Error::ArchiveError(format!(
                "archive version is not match, got {}, expect {}",
                version, ARCHIVE_INDEX_FORMAT_V1
            )));
        }
        let meta_address = ObjectAddress::unmarshal(buf);
        buf.advance(OBJECT_ADDR_SIZE);
        let num_file_ids = buf.get_u32_le();
        let mut sst_file_ids = Vec::new();
        for _i in 0..num_file_ids {
            sst_file_ids.push(buf.get_u64_le());
        }
        let mut sst_addresses = Vec::new();
        for _i in 0..num_file_ids {
            sst_addresses.push(ObjectAddress::unmarshal(buf));
            buf.advance(OBJECT_ADDR_SIZE);
        }
        Ok(Self {
            meta_address,
            sst_file_ids,
            sst_addresses,
        })
    }

    pub fn get_meta_address(&self) -> ObjectAddress {
        self.meta_address
    }
}

struct ArchiveWriter {
    max_size: u64,
    package_id: u32,
    index: ArchiveIndex,
    buf: Vec<u8>,
    s3fs: Arc<S3Fs>,
    date: String,
}

impl ArchiveWriter {
    fn new(max_size: u64, s3fs: Arc<S3Fs>, date: String, meta_data: Bytes) -> Self {
        let package_id: u32 = 0;
        let mut buf = Vec::new();
        let meta_address = ObjectAddress::new(package_id, buf.len() as u64, meta_data.len() as u64);
        buf.extend_from_slice(meta_data.as_bytes());
        Self {
            max_size,
            package_id,
            index: ArchiveIndex::new(meta_address),
            buf,
            s3fs,
            date,
        }
    }

    fn append_files(&mut self, file_ids: Vec<u64>) -> Result<()> {
        let (result_tx, result_rx) = tikv_util::mpsc::bounded(file_ids.len());
        let mut msg_count = 0;
        for file_id in file_ids {
            let s3fs = self.s3fs.clone();
            let tx = result_tx.clone();
            self.s3fs.get_runtime().spawn(async move {
                let res = s3fs.read_file(file_id, Options::new(0, 0)).await;
                let _ = tx.send(res.map(|sst_data| (file_id, sst_data)));
            });
            if msg_count < LOAD_FILE_CONCURRENCY {
                msg_count += 1;
            } else {
                self.recv_sst_data(&result_rx)?;
            }
        }
        for _ in 0..msg_count {
            self.recv_sst_data(&result_rx)?;
        }
        Ok(())
    }

    fn recv_sst_data(&mut self, result_tx: &Receiver<dfs::Result<(u64, Bytes)>>) -> Result<()> {
        let (file_id, sst_data) = result_tx.recv().unwrap()?;
        self.append_file(file_id, sst_data);
        Ok(())
    }

    fn append_file(&mut self, file_id: u64, sst_data: Bytes) {
        if self.should_rotate() {
            self.rotate();
        }
        self.index.append(
            file_id,
            ObjectAddress::new(
                self.package_id,
                self.buf.len() as u64,
                sst_data.len() as u64,
            ),
        );
        self.buf.extend_from_slice(sst_data.as_bytes());
    }

    fn should_rotate(&self) -> bool {
        self.buf.len() as u64 > self.max_size
    }

    fn rotate(&mut self) {
        if self.buf.is_empty() {
            return;
        }
        let runtime = self.s3fs.get_runtime();
        let key = archive_package_key(self.s3fs.get_prefix(), self.date.clone(), self.package_id);
        let data = Bytes::from(self.buf.to_vec());
        runtime
            .block_on(self.s3fs.put_object_with_options(
                key.clone(),
                data,
                key.clone(),
                None,
                Some(STORAGE_CLASS_GLACIER_IR),
            ))
            .unwrap();
        info!("cluster archive package {} on {}", key, self.date.clone());
        self.package_id += 1;
        self.buf.truncate(0);
    }

    fn finish(&mut self) {
        self.rotate();
        self.index.marshal(&mut self.buf);
        let runtime = self.s3fs.get_runtime();
        let key = archive_index_key(self.s3fs.get_prefix(), self.date.clone());
        let data = Bytes::from(self.buf.to_vec());
        runtime
            .block_on(self.s3fs.put_object(key.clone(), data, key.clone()))
            .unwrap();
        info!("cluster archive index {} on {}", key, self.date.clone());
    }
}

#[derive(Default, Clone, Debug)]
pub struct ArchiveAddress {
    date: String,
    object_addr: ObjectAddress,
}

impl ArchiveAddress {
    pub fn new(date: String, object_addr: ObjectAddress) -> Self {
        Self { date, object_addr }
    }
}

pub struct ArchiveReader {
    start_date: String,
    meta_archive_address: Option<ArchiveAddress>,
    archive_addresses: HashMap<u64, ArchiveAddress>,
    s3fs: Arc<S3Fs>,
}

impl ArchiveReader {
    pub fn new(s3fs: Arc<S3Fs>, date: &NaiveDate) -> Result<Self> {
        let start_date = archive_format_date(date);
        let (index_keys, _) = s3fs.get_runtime().block_on(get_all_archive_index_paths(
            &s3fs,
            start_date.clone(),
            usize::MAX,
        ))?;
        if index_keys.is_empty() {
            return Err(Error::ArchiveError(format!(
                "failed to get archive reader on {}",
                start_date,
            )));
        }
        let mut meta_archive_address: Option<ArchiveAddress> = None;
        let mut archive_addresses: HashMap<u64, ArchiveAddress> = HashMap::new();
        for index_key in index_keys {
            let date = parse_index_date(&index_key.clone());
            let (archive_index, _) = get_archive_index(&s3fs, date.clone())?;
            if date.eq(&start_date) {
                meta_archive_address = Some(ArchiveAddress::new(
                    date.clone(),
                    archive_index.meta_address,
                ))
            }
            let num_file_ids = archive_index.sst_addresses.len();
            for i in 0..num_file_ids {
                archive_addresses.insert(
                    archive_index.sst_file_ids[i],
                    ArchiveAddress::new(date.clone(), archive_index.sst_addresses[i]),
                );
            }
        }
        Ok(Self {
            start_date,
            meta_archive_address,
            archive_addresses,
            s3fs,
        })
    }

    pub fn get_meta_archive_addr(&self) -> Option<ArchiveAddress> {
        self.meta_archive_address.clone()
    }

    pub fn read_meta_file(&self) -> Result<ClusterBackupMeta> {
        if let Some(archive_addr) = self.get_meta_archive_addr() {
            return self
                .s3fs
                .get_runtime()
                .block_on(get_archived_object(&self.s3fs, archive_addr))
                .and_then(|data| {
                    let mut cluster_backup_meta = ClusterBackupMeta::new();
                    cluster_backup_meta.merge_from_bytes(&data).map_err(|e| {
                        Error::ArchiveError(format!(
                            "failed to decode cluster backup meta on {}, err {}",
                            self.start_date, e,
                        ))
                    })?;
                    Ok(cluster_backup_meta)
                });
        }
        return Err(Error::ArchiveError(format!(
            "failed to find archive meta file on {}",
            self.start_date
        )));
    }

    pub fn get_file_archive_addr(&self, file_id: u64) -> Option<ArchiveAddress> {
        self.archive_addresses.get(&file_id).cloned()
    }

    pub fn read_file(&self, file_id: u64) -> Result<Bytes> {
        if let Some(archive_addr) = self.get_file_archive_addr(file_id) {
            return self
                .s3fs
                .get_runtime()
                .block_on(get_archived_object(&self.s3fs, archive_addr.clone()))
                .map_err(|e| {
                    error!(
                        "{}, file id {}, archive addr {:?}",
                        e.to_string(),
                        file_id,
                        archive_addr
                    );
                    e
                });
        }
        return Err(Error::ArchiveError(format!(
            "failed to find archive file {}",
            file_id,
        )));
    }

    pub fn restore_file(&self, file_id: u64) -> Result<()> {
        let date = self.read_file(file_id)?;
        return self
            .s3fs
            .get_runtime()
            .block_on(self.s3fs.create(file_id, date, Options::new(0, 0)))
            .map_err(|e| {
                error!(
                    "failed to restore archive file {}, err {}",
                    file_id,
                    e.to_string()
                );
                Error::DfsError(e)
            });
    }

    pub fn restore_files(&self, file_ids: Vec<u64>) -> Result<()> {
        let mut archive_addrs = Vec::default();
        for file_id in file_ids {
            if let Some(archive_addr) = self.get_file_archive_addr(file_id) {
                archive_addrs.push((file_id, archive_addr));
            } else {
                return Err(Error::ArchiveError(format!(
                    "failed to find archive file {}",
                    file_id,
                )));
            }
        }
        let recv_restore_file = |result_tx: &Receiver<Result<u64>>| -> Result<()> {
            let file_id = result_tx.recv().unwrap()?;
            info!("restore archived sst {} done", file_id);
            Ok(())
        };
        let (result_tx, result_rx) = tikv_util::mpsc::bounded(archive_addrs.len());
        let mut msg_count = 0;
        for (file_id, archive_addr) in archive_addrs {
            let s3fs = self.s3fs.clone();
            let tx = result_tx.clone();
            self.s3fs.get_runtime().spawn(async move {
                let res = get_archived_object(&s3fs, archive_addr.clone())
                    .await
                    .map_err(|e| {
                        error!(
                            "{}, file id {}, archive addr {:?}",
                            e.to_string(),
                            file_id,
                            archive_addr
                        );
                        e
                    });
                if res.is_err() {
                    let _ = tx.send(res.map(|_| file_id));
                    return;
                }
                let data = res.unwrap();
                let res = s3fs
                    .create(file_id, data, Options::new(0, 0))
                    .await
                    .map_err(|e| {
                        error!(
                            "{}, file id {}, archive addr {:?}",
                            e.to_string(),
                            file_id,
                            archive_addr
                        );
                        Error::DfsError(e)
                    });
                let _ = tx.send(res.map(|_| file_id));
            });
            if msg_count < LOAD_FILE_CONCURRENCY {
                msg_count += 1;
            } else {
                recv_restore_file(&result_rx)?;
            }
        }
        for _ in 0..msg_count {
            recv_restore_file(&result_rx)?;
        }
        Ok(())
    }

    pub fn get_file_ids(&self) -> Vec<u64> {
        self.archive_addresses.keys().copied().collect::<Vec<_>>()
    }
}

pub fn archive_format_date(date: &chrono::NaiveDate) -> String {
    date.format(INCREMENTAL_BACKUP_FOLDER_FORMAT).to_string()
}

pub fn archive_index_key(prefix: String, date: String) -> String {
    format!("{}/archive/index/{}.idx", prefix, date)
}

pub fn archive_package_key(prefix: String, date: String, package_id: u32) -> String {
    format!(
        "{}/archive/package/{}/{:08x}.pack",
        prefix, date, package_id
    )
}

pub fn parse_index_date(index_key: &str) -> String {
    let end_idx = index_key.len() - 4;
    let start_idx = end_idx - 8;
    let data_part = &index_key[start_idx..end_idx];
    data_part.to_string()
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use bytes::Bytes;
    use protobuf::Message;
    use rfenginepb::{ClusterBackupMeta, StoreBackupMeta};
    use test_cloud_server::oss::ObjectStorageService;

    use super::*;

    #[test]
    fn test_archive_writer() {
        test_util::init_log_for_test();

        const CLUSTER_ID: u64 = 1;
        const NUM_STORES: u64 = 4;
        const NUM_FILE_IDS: u64 = 24;

        let base_dir = tempfile::Builder::new()
            .prefix("test_archive_writer_")
            .tempdir()
            .unwrap();

        let mut oss = ObjectStorageService::new(base_dir.path());
        oss.start_server();

        let s3fs = Arc::new(S3Fs::new(
            "pfx".to_string(),
            format!("http://127.0.0.1:{}", oss.port()),
            "admin".to_string(),
            "admin".to_string(),
            "local".to_string(),
            "bkt".to_string(),
        ));
        let get_file_id = |i: u64| i;
        let get_sst_data = |file_id: u64| Bytes::from(b"x".repeat(100 + file_id as usize).to_vec());
        s3fs.get_runtime().block_on(async {
            for i in 0..NUM_FILE_IDS {
                let file_id = get_file_id(i);
                let opts = dfs::Options::new(0, 0);
                let sst_data = get_sst_data(file_id);
                s3fs.create(file_id, sst_data, opts).await.unwrap();
            }
        });

        let mut cluster_meta = ClusterBackupMeta::new();
        cluster_meta.set_cluster_id(CLUSTER_ID);
        for i in 0..NUM_STORES {
            cluster_meta.mut_stores().push(StoreBackupMeta {
                store_id: i,
                ..Default::default()
            });
        }
        assert_eq!(cluster_meta.stores.len(), NUM_STORES as usize);
        let meta_data = Bytes::from(cluster_meta.write_to_bytes().unwrap());
        assert!(!meta_data.is_empty());
        let backup_date = chrono::Utc::now().date_naive();
        let format_date = archive_format_date(&backup_date);
        let mut writer = ArchiveWriter::new(1024, s3fs.clone(), format_date.clone(), meta_data);
        let mut file_ids = Vec::with_capacity(NUM_FILE_IDS as usize);
        for file_id in 0..NUM_FILE_IDS {
            file_ids.push(file_id)
        }
        writer.append_files(file_ids).unwrap();
        writer.finish();
        let num_packages = writer.package_id;
        s3fs.get_runtime().block_on(async {
            let (objects, ..) = s3fs.list("", None, None).await.unwrap();
            assert_eq!(
                objects.len(),
                NUM_FILE_IDS as usize + 1 /* archive_index */+ num_packages as usize
            );
        });
        let (archive_index, data) = get_archive_index(&s3fs, format_date.clone()).unwrap();
        assert!(!data.is_empty());
        assert_eq!(archive_index.sst_file_ids.len(), NUM_FILE_IDS as usize);
        {
            let meta_address = archive_index.get_meta_address();
            let archive_address = ArchiveAddress::new(format_date.clone(), meta_address);
            let data = s3fs
                .get_runtime()
                .block_on(get_archived_object(&s3fs, archive_address))
                .unwrap();
            let mut cluster_backup = ClusterBackupMeta::new();
            cluster_backup.merge_from_bytes(&data).unwrap();
            assert_eq!(cluster_backup.cluster_id, CLUSTER_ID);
            assert_eq!(cluster_backup.stores.len(), NUM_STORES as usize);
            for i in 0..NUM_STORES {
                assert_eq!(cluster_backup.stores[i as usize].store_id, i);
            }
        }

        for i in 0..NUM_FILE_IDS as usize {
            let file_id = archive_index.sst_file_ids[i];
            let address = archive_index.sst_addresses[i];
            let archive_address = ArchiveAddress::new(format_date.clone(), address);
            let data = s3fs
                .get_runtime()
                .block_on(get_archived_object(&s3fs, archive_address))
                .unwrap();
            assert_eq!(data.len(), address.length as usize);
            let sst_data = get_sst_data(file_id);
            assert!(data.eq(&sst_data));
        }

        oss.shutdown();
    }

    #[test]
    fn test_archive_reader() {
        test_util::init_log_for_test();

        const CLUSTER_ID: u64 = 1;
        const NUM_DATES: u64 = 8;

        let base_dir = tempfile::Builder::new()
            .prefix("test_archive_reader_")
            .tempdir()
            .unwrap();

        let mut oss = ObjectStorageService::new(base_dir.path().join("oss").as_path());
        oss.start_server();

        let s3fs = Arc::new(S3Fs::new(
            "pfx".to_string(),
            format!("http://127.0.0.1:{}", oss.port()),
            "admin".to_string(),
            "admin".to_string(),
            "local".to_string(),
            "bkt".to_string(),
        ));
        let first_date = chrono::Utc::now().date_naive() - chrono::Duration::days(NUM_DATES as i64);
        let get_date = |j: u64| first_date + chrono::Duration::days(j as i64);
        let get_num_stores = |j: u64| j + 8;
        let get_num_file_ids = |j: u64| j + 3;
        let get_file_id = |j: u64, i: u64| j * 1000 + i + 200;
        let get_sst_data =
            |file_id: u64| Bytes::from(b"x".repeat(file_id as usize % 1000).to_vec());
        for j in 0..NUM_DATES {
            let num_file_ids = get_num_file_ids(j);
            s3fs.get_runtime().block_on(async {
                for i in 0..num_file_ids {
                    let file_id = get_file_id(j, i);
                    let opts = dfs::Options::new(0, 0);
                    let sst_data = get_sst_data(file_id);
                    s3fs.create(file_id, sst_data, opts).await.unwrap();
                }
            });
            let mut cluster_meta = ClusterBackupMeta::new();
            cluster_meta.set_cluster_id(CLUSTER_ID);
            let num_stores = get_num_stores(j);
            for i in 0..num_stores {
                cluster_meta.mut_stores().push(StoreBackupMeta {
                    store_id: i,
                    ..Default::default()
                });
            }
            let meta_data = Bytes::from(cluster_meta.write_to_bytes().unwrap());
            let backup_date = get_date(j);
            let format_date = archive_format_date(&backup_date);
            let mut writer = ArchiveWriter::new(512, s3fs.clone(), format_date.clone(), meta_data);
            let mut file_ids = Vec::with_capacity(num_file_ids as usize);
            for i in 0..num_file_ids {
                let file_id = get_file_id(j, i);
                file_ids.push(file_id)
            }
            writer.append_files(file_ids).unwrap();
            writer.finish();
        }
        let start_date = get_date(0);
        let reader = ArchiveReader::new(s3fs, &start_date).unwrap();
        {
            let cluster_backup_meta = reader.read_meta_file().unwrap();
            assert_eq!(cluster_backup_meta.cluster_id, CLUSTER_ID);
            let num_stores = get_num_stores(0);
            assert_eq!(cluster_backup_meta.stores.len(), num_stores as usize);
            for i in 0..num_stores {
                assert_eq!(cluster_backup_meta.stores[i as usize].store_id, i);
            }
        }
        for j in 0..NUM_DATES {
            let num_file_ids = get_num_file_ids(j);
            for i in 0..num_file_ids {
                let file_id = get_file_id(j, i);
                let data = reader.read_file(file_id).unwrap();
                let sst_data = get_sst_data(file_id);
                assert!(data.eq(&sst_data));
            }
        }

        oss.shutdown();
    }

    #[test]
    fn test_get_all_archive_index_paths() {
        test_util::init_log_for_test();

        const NUM_INDEXES: i64 = 3;

        let base_dir = tempfile::Builder::new()
            .prefix("test_get_all_archive_index_paths_")
            .tempdir()
            .unwrap();

        let mut oss = ObjectStorageService::new(base_dir.path());
        oss.start_server();

        let s3fs = Arc::new(S3Fs::new(
            "pfx".to_string(),
            format!("http://127.0.0.1:{}", oss.port()),
            "admin".to_string(),
            "admin".to_string(),
            "local".to_string(),
            "bkt".to_string(),
        ));
        let first_date = chrono::Utc::now().date_naive() - chrono::Duration::days(NUM_INDEXES);
        let get_date = |i: i64| first_date + chrono::Duration::days(i);
        s3fs.get_runtime().block_on(async {
            for i in 0..NUM_INDEXES {
                let date = get_date(i);
                let format_date = archive_format_date(&date);
                let key = archive_index_key(s3fs.get_prefix(), format_date);
                let sst_data = Bytes::from(b"x".repeat(100 + i as usize).to_vec());
                s3fs.put_object(key.clone(), sst_data, key.clone())
                    .await
                    .unwrap();
            }
            for i in 0..NUM_INDEXES {
                let date = get_date(i);
                let start_date = archive_format_date(&date);
                let (index_keys, _) = get_all_archive_index_paths(&s3fs, start_date, usize::MAX)
                    .await
                    .unwrap();
                assert_eq!(index_keys.len(), (NUM_INDEXES - i) as usize);
            }
        });

        oss.shutdown();
    }
}
