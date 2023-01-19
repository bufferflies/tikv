// Copyright 2022 TiKV Project Authors. Licensed under Apache-2.0.

use std::{path::PathBuf, str::FromStr, sync::mpsc::SyncSender, time::Duration};

use bytes::Bytes;
use clap::Args;
use futures::executor::block_on;
use http::{Request, Uri};
use hyper::Body;
use kvengine::dfs::{self, DFSConfig, S3FS};
use kvproto::metapb::Store;
use pd_client::PdClient;
use protobuf::Message;
use rfenginepb::{ClusterBackupMeta, StoreBackupMeta};
use security::SecurityConfig;
use slog_global::{error, info, warn};

use crate::common::{
    create_pd_client, generate_etcd_connect_opt, get_all_stores_except_tiflash,
    send_request_to_store,
};
const INCREMENTAL_BACKUP_INTERVAL: u64 = 30; // seconds.
const BACKUP_FOLDER_FORMAT: &str = "%Y%m%d";
const MAX_BATCH_GET_CNT: i64 = 1024;
// keyspace meta in pd is "/pd/$cluster_id/PD_KEYSPACE_META_PATH"
const PD_KEY_SPACE_META_PATH: [&str; 3] = ["keyspaces/", "region_label/keyspaces/", "rules/"];

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Cluster topology error {0}")]
    TopoChanged(String),
    #[error("Backup meta of cluster {0} is not found")]
    MetaNotFound(u64),
    #[error("DFS error {0}")]
    DFSError(dfs::Error),
    #[error("Server error {0}")]
    ServerError(String),
    #[error("Safe ts {0} is greater than backup ts {1}")]
    TsError(u64, u64),
    #[error("PD error {0}")]
    PDError(pd_client::Error),
    #[error("Etcd error {0}")]
    EtcdError(etcd_client::Error),
}

impl From<dfs::Error> for Error {
    fn from(e: dfs::Error) -> Self {
        Error::DFSError(e)
    }
}

impl From<pd_client::Error> for Error {
    fn from(e: pd_client::Error) -> Self {
        Error::PDError(e)
    }
}

impl From<etcd_client::Error> for Error {
    fn from(e: etcd_client::Error) -> Self {
        Error::EtcdError(e)
    }
}

#[derive(Args)]
pub struct BackupArgs {
    /// The path of the config file.
    #[clap(long, default_value = "")]
    pub config: PathBuf,
    /// The name of the backup file, if empty, a system generated name will be used.
    #[clap(long, default_value_t = String::new())]
    pub name: String,
    /// Incremental backup or full backup.
    #[clap(long)]
    pub incremental: bool,
    /// Incremental backup interval, in seconds.
    #[clap(long, default_value_t = INCREMENTAL_BACKUP_INTERVAL)]
    pub interval: u64,
    /// PD endpoints, use `,` to separate multiple PDs
    #[clap(long, default_value_t = String::new())]
    pub pd: String,
    /// Path of file that contains list of trusted SSL CAs
    #[clap(long, default_value = "")]
    pub cacert: PathBuf,
    /// Path of file that contains X509 certificate in PEM format
    #[clap(long, default_value = "")]
    pub cert: PathBuf,
    /// Path of file that contains X509 key in PEM format
    #[clap(long, default_value = "")]
    pub key: PathBuf,
}

fn backup_file_name(prefix: String, name: String, backup_ts: u64) -> String {
    if name.is_empty() {
        let folder = format!(
            "backup/{}",
            chrono::Local::now().format(BACKUP_FOLDER_FORMAT)
        );
        format!("{}/{}/{}.meta", prefix, folder, backup_ts)
    } else {
        format!("{}/backup/{}", prefix, name)
    }
}

pub fn execute_backup(args: BackupArgs) {
    let config: BackupConfig = get_backup_config_from_args(&args);
    if args.incremental {
        execute_incremental_backup(config, args.name, Duration::from_secs(args.interval))
    } else {
        execute_full_backup(config, args.name)
    }
}

fn execute_incremental_backup(config: BackupConfig, name: String, interval: Duration) {
    if !name.is_empty() {
        error!("Don't support non-empty name for incremental backup.");
        return;
    }
    let pd_client = create_pd_client(&config.security, &config.pd);
    let mut cluster_backup_meta = None;
    loop {
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
        std::thread::sleep(interval)
    }
}

fn execute_full_backup(config: BackupConfig, name: String) {
    let pd_client = create_pd_client(&config.security, &config.pd);
    if let Err(e) = backup_cluster(config, false, name, &pd_client, None) {
        error!("Full backup fail, {:?}", e)
    }
}

pub fn backup_cluster(
    config: BackupConfig,
    incremental: bool,
    name: String,
    pd_client: &dyn PdClient,
    last_backup_meta: Option<ClusterBackupMeta>,
) -> Result<ClusterBackupMeta> {
    let stores = get_all_stores_except_tiflash(pd_client)?;
    let backup_ts = block_on(pd_client.get_tso())?.into_inner();
    let cluster_id = pd_client.get_cluster_id()?;

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(8)
        .enable_all()
        .build()
        .unwrap();

    let dfs_conf = config.dfs.clone();
    let s3fs = S3FS::new(
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
    runtime.block_on(backup_pd_keyspace_meta(&config, &mut cluster_backup_meta))?;
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
        error!("backup errors {:?}", errs);
        if errs.len() > config.tolerate_err {
            return Err(Error::ServerError(format!("backup errors {:?}", errs)));
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
    let backup_key = backup_file_name(config.dfs.prefix, name, backup_ts);
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
    info!("Start backup with config {:?}", config);
    let json_string = serde_json::to_string(&config).unwrap();
    let req = Request::post(uri).body(Body::from(json_string)).unwrap();
    match send_request_to_store(req, store.clone()).await {
        Ok(resp) => {
            let mut store_backup_meta = StoreBackupMeta::default();
            store_backup_meta.merge_from_bytes(&resp).unwrap();
            tx.send(Ok(store_backup_meta)).unwrap()
        }
        Err(e) => tx.send(Err(Error::ServerError(e))).unwrap(),
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

async fn get_all_backup_files(s3fs: &S3FS) -> dfs::Result<Vec<String>> {
    let mut files = vec![];
    let mut start_key = format!(
        "backup/{}",
        chrono::Local::now().format(BACKUP_FOLDER_FORMAT)
    );
    let prefix = format!("{}/{}", s3fs.get_prefix(), start_key);
    loop {
        match s3fs.list(&start_key).await {
            Ok((backup_files, mut more)) => {
                for file in backup_files {
                    if !file.starts_with(&prefix) {
                        more = false;
                        break;
                    }
                    files.push(file);
                }
                if !more {
                    break;
                }
                start_key = files.last().unwrap().clone();
            }
            Err(e) => {
                return Err(e);
            }
        }
    }
    Ok(files)
}

// If backup exist, return the latest one, else create a new ClusterBackupMeta.
async fn get_latest_backup_meta(s3fs: &S3FS, cluster_id: u64) -> Result<ClusterBackupMeta> {
    let files = get_all_backup_files(s3fs).await?;
    if files.is_empty() {
        return Err(Error::MetaNotFound(cluster_id));
    }
    // Incremental backup file name is generated with `backup_file_name`.
    // The last should be the latest one in most cases.
    let last_file = files.last().unwrap();
    let object = s3fs
        .get_object(last_file.clone(), last_file.clone())
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
    // User cannot set placement rule in serverless cluster except the tiflash replica.
    // So all placement rules are created inner, backup all of them.
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
}

fn get_backup_config_from_args(args: &BackupArgs) -> BackupConfig {
    let mut config = BackupConfig::default();
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
    config
}

#[cfg(test)]
mod tests {
    use kvproto::metapb::Store;
    use rfenginepb::{ChangeSet, ClusterBackupMeta, StoreBackupMeta, WalChunk};

    use super::{check_backup_meta_consistency, merge_store_backup_meta};

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
        assert!(check_backup_meta_consistency(&meta, &stores).is_err());

        for i in 0..3 {
            meta.mut_stores().push(StoreBackupMeta {
                store_id: i + 1,
                ..Default::default()
            });
        }
        assert!(check_backup_meta_consistency(&meta, &stores).is_err());

        meta.clear_stores();
        for i in 0..3 {
            meta.mut_stores().push(StoreBackupMeta {
                store_id: i,
                ..Default::default()
            });
        }
        assert!(check_backup_meta_consistency(&meta, &stores).is_ok());
    }
}
