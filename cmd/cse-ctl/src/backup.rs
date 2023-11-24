// Copyright 2022 TiKV Project Authors. Licensed under Apache-2.0.

use std::{path::PathBuf, sync::Arc, time::Duration};

use clap::Args;
use kvengine::dfs::{DFSConfig, S3Fs};
use native_br::backup::{
    execute_full_backup, execute_incremental_backup, execute_lightweight_backup, BackupConfig,
};
use rfengine::parse_epoch_from_snapshot_key;
use tikv_util::info;
const INCREMENTAL_BACKUP_INTERVAL: u64 = 30; // seconds.

#[derive(Args)]
pub struct BackupArgs {
    /// The path of the config file.
    #[clap(long, default_value = "")]
    pub config: PathBuf,
    /// The name of the backup file, if empty, a system generated name will be
    /// used.
    #[clap(long, default_value_t = String::new())]
    pub name: String,
    /// Lightweight backup or legacy backup.
    #[clap(long)]
    pub lightweight: bool,
    /// Incremental backup or full backup.
    #[clap(long)]
    pub incremental: bool,
    /// Backup interval for incremental or lightweight, in seconds. For
    /// lightweight backup, interval set to 0 means only backup once.
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
    #[clap(long)]
    pub skip_keyspace_meta: bool,
    /// The tolerate num of stores' backup failure.
    #[clap(long, default_value_t = 0)]
    pub tolerate_err: usize,
}

pub fn execute_backup(args: BackupArgs) {
    let config: BackupConfig = get_backup_config_from_args(&args);
    info!("Begin backup with config {:?}", config);
    if args.lightweight {
        execute_lightweight_backup(config, args.name, Duration::from_secs(args.interval));
    } else if args.incremental {
        execute_incremental_backup(config, args.name, Duration::from_secs(args.interval))
    } else {
        execute_full_backup(config, args.name)
    }
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
    config.skip_keyspace_meta = args.skip_keyspace_meta;
    if args.tolerate_err > 0 {
        config.tolerate_err = args.tolerate_err;
    }
    config.dfs.override_from_env();
    config.security.master_key.override_from_env();
    config
}

#[derive(Args)]
pub struct ShowBackupArgs {
    /// The path of the config file.
    #[clap(long, default_value = "")]
    pub config: PathBuf,
    /// The name of the backup file.
    #[clap(long)]
    pub name: String,
    /// Enable verbose mode.
    #[clap(long)]
    pub verbose: bool,
}

#[derive(Clone, Serialize, Deserialize, PartialEq, Debug, Default)]
#[serde(default)]
#[serde(rename_all = "kebab-case")]
pub struct ShowBackupConfig {
    pub dfs: DFSConfig,
}

pub fn execute_show_backup(args: ShowBackupArgs) {
    let mut config = ShowBackupConfig::default();
    if args.config.exists() {
        let data = std::fs::read(args.config.clone()).expect("failed to read config file");
        config = toml::from_slice(&data).unwrap();
    }
    config.dfs.override_from_env();

    let s3fs = S3Fs::new(
        config.dfs.prefix.clone(),
        config.dfs.s3_endpoint,
        config.dfs.s3_key_id,
        config.dfs.s3_secret_key,
        config.dfs.s3_region,
        config.dfs.s3_bucket,
    );

    let cluster_backup = native_br::restore::get_cluster_backup_meta(&s3fs, args.name.clone());

    let store_ids = cluster_backup
        .get_stores()
        .iter()
        .map(|store| store.get_store_id())
        .collect::<Vec<_>>();
    let peers_count = cluster_backup
        .get_stores()
        .iter()
        .map(|store| store.get_manifest().get_peers().len())
        .sum::<usize>();
    let keyspaces_count = cluster_backup.keyspace_meta.len();
    let is_lightweight = cluster_backup.is_lightweight;

    println!("[Backup {}]", args.name);
    println!("  cluster_id: {}", cluster_backup.cluster_id);
    println!("  backup_ts: {}", cluster_backup.backup_ts);
    println!("  safe_ts: {}", cluster_backup.safe_ts);
    println!("  stores: {:?}", store_ids);
    println!("  peers count: {}", peers_count);
    println!("  alloc_id: {}", cluster_backup.alloc_id);
    println!("  keyspaces count: {}", keyspaces_count);
    println!("  lightweight: {}", is_lightweight);

    if args.verbose {
        for store in &cluster_backup.stores {
            println!();
            println!("[Store {}]", store.store_id);
            if is_lightweight {
                let object_storage = Arc::new(s3fs.clone());
                let snap_key = rfengine::find_latest_snapshot(
                    object_storage.clone(),
                    &config.dfs.prefix,
                    store,
                )
                .unwrap();
                let snap_epoch =
                    parse_epoch_from_snapshot_key(snap_key.as_deref()).unwrap_or_default();
                println!(
                    "  Backup epoch: {}, latest snapshot epoch: {}, offset: {}, {} epoch need to replay",
                    store.epoch,
                    snap_epoch,
                    store.offset,
                    store.epoch - snap_epoch
                );
                // Get latest snapshot epoch
            } else {
                println!("  WAL chunks:");
                for file_key in store.get_wal_chunks().iter().map(|chunk| {
                    rfengine::wal_file_key(
                        store.store_id,
                        chunk.epoch,
                        chunk.start_off,
                        chunk.end_off,
                    )
                }) {
                    println!(
                        "    Backup epoch: {}, file_key: {}",
                        store.get_manifest().epoch_id,
                        file_key
                    );
                }
                // TODO: output some readable information of `store.manifest`.
            }
        }
    }
}
