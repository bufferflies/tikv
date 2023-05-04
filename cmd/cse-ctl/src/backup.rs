// Copyright 2022 TiKV Project Authors. Licensed under Apache-2.0.

use std::{path::PathBuf, time::Duration};

use clap::Args;
use native_br::backup::{execute_full_backup, execute_incremental_backup, BackupConfig};
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
    #[clap(long)]
    pub skip_keyspace_meta: bool,
    /// The tolerate num of stores' backup failure.
    #[clap(long, default_value_t = 0)]
    pub tolerate_err: usize,
}

pub fn execute_backup(args: BackupArgs) {
    let config: BackupConfig = get_backup_config_from_args(&args);
    info!("Begin backup with config {:?}", config);
    if args.incremental {
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
    config
}
