// Copyright 2023 TiKV Project Authors. Licensed under Apache-2.0.

use std::{path::PathBuf, time::Duration};

use clap::Args;
use native_br::archive::{
    archive_with_cfg, ArchiveConfig, DEFAULT_MAX_ARCHIVE_FILE_SIZE, LOAD_FILE_CONCURRENCY,
};
use tikv_util::{config::ReadableDuration, info};

#[derive(Args)]
pub struct ArchiveArgs {
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
    /// Data dir for load_data
    #[clap(long, default_value_t = String::new())]
    pub data_dir: String,
    #[clap(long, default_value_t = DEFAULT_MAX_ARCHIVE_FILE_SIZE)]
    pub max_archive_file_size: u64,
    /// Start archive duration, in duration string (see `ReadableDuration`).
    /// `start_archive_time` = now() - `start-archive-duration`.
    #[clap(long, default_value = "91d")]
    pub start_archive_duration: ReadableDuration,
    /// The expiration date of backups. The date format is
    /// "%Y%m%d". e.g. 20060102
    #[clap(long, default_value_t = String::new())]
    pub expiration_date: String,
    /// Concurrently do s3 requests.
    #[clap(long, default_value_t = LOAD_FILE_CONCURRENCY)]
    pub concurrency: usize,
    #[clap(long)]
    pub dry_run: bool,
}

pub fn execute_archive(args: ArchiveArgs) {
    let config: ArchiveConfig = get_archive_config_from_args(&args);
    info!("Begin archive with config {:?}", config);
    if let Err(e) = archive_with_cfg(config) {
        panic!("failed to archive cluster backup, err {:?}", e)
    }
}

fn get_archive_config_from_args(args: &ArchiveArgs) -> ArchiveConfig {
    let mut config = ArchiveConfig::default();
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
    config.max_archive_file_size = args.max_archive_file_size;
    config.start_archive_duration = Duration::from(args.start_archive_duration);
    config.expiration_date = args.expiration_date.clone();
    config.concurrency = args.concurrency;
    config.dry_run = args.dry_run;
    config.dfs.override_from_env();
    config.security.master_key.override_from_env();
    config.data_dir = args.data_dir.clone();
    config.check_data_dir();
    config
}
