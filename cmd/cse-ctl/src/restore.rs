// Copyright 2022 TiKV Project Authors. Licensed under Apache-2.0.

use std::{path::PathBuf, sync::Arc};

use clap::{Args, Subcommand};
use kvengine::dfs::S3Fs;
use native_br::{
    common::{create_pd_client, now},
    restore,
    restore::{restore_pd, restore_tikv, RestoreConfig},
    restore_keyspace::{
        restore_keyspace_with_cfg, ReportRestoreStepTrait, RestoreStep, RestoredKeyspace,
    },
    step, step_error,
};
use pd_client::PdClient;
use slog_global::{error, info};
use tikv_util::config::ReadableSize;

use crate::restore::Commands::{Keyspace, Pd, Tikv};

#[derive(Args)]
pub struct RestoreCommand {
    #[clap(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Restore TiKV raft store data.
    Tikv(RestoreTikvArgs),
    /// Restore PD meta data.
    Pd(RestorePdArgs),
    /// Restore Keyspace data.
    Keyspace(RestoreKeyspaceArgs),
}

#[derive(Args)]
pub struct RestoreTikvArgs {
    /// The path of the config file.
    #[clap(long, default_value = "")]
    pub config: PathBuf,
    /// The name of the backup file.
    #[clap(long)]
    pub name: String,
    /// The local path to load store files.
    #[clap(long)]
    pub path: String,
    /// The store id in backup meta to restore.
    #[clap(long)]
    pub store_id: u64,
    /// Add delta to generate the new store_id. Please keep the delta same in
    /// the same cluster. This should be kept unchanged unless in particular
    /// cases.
    #[clap(long, default_value = "0")]
    pub new_store_id_delta: u64,
    /// WAL target file size.
    #[clap(long, default_value = "512MB")]
    pub wal_target_size: ReadableSize,
}

#[derive(Args)]
struct RestorePdArgs {
    /// The path of the config file.
    #[clap(long, default_value = "")]
    pub config: PathBuf,
    /// The name of the backup file.
    #[clap(long)]
    pub name: String,
    /// PD endpoints, use `,` to separate multiple PDs.
    #[clap(long, default_value_t = String::new())]
    pub pd: String,
    /// Path of file that contains list of trusted SSL CAs.
    #[clap(long, default_value = "")]
    pub cacert: PathBuf,
    /// Path of file that contains X509 certificate in PEM format.
    #[clap(long, default_value = "")]
    pub cert: PathBuf,
    /// Path of file that contains X509 key in PEM format.
    #[clap(long, default_value = "")]
    pub key: PathBuf,
    /// Add delta to generate the new store_id. Please keep the delta same with
    /// restore tikv. This should be kept unchanged unless in particular
    /// cases.
    #[clap(long, default_value = "0")]
    pub new_store_id_delta: u64,
}

#[derive(Args)]
pub struct RestoreKeyspaceArgs {
    /// The path of the config file.
    #[clap(long, default_value = "")]
    pub config: PathBuf,
    /// The name of the backup file.
    #[clap(long)]
    pub name: String,
    /// The keyspace to restore.
    #[clap(long)]
    pub keyspace_name: String,
    /// The target keyspace.
    #[clap(long)]
    pub target_keyspace_name: Option<String>,
    /// The local working path for temporary files during restore.
    #[clap(long)]
    pub working_path: Option<String>,
    /// PD endpoints, use `,` to separate multiple PDs.
    #[clap(long, default_value_t = String::new())]
    pub pd: String,
    /// Path of file that contains list of trusted SSL CAs.
    #[clap(long, default_value = "")]
    pub cacert: PathBuf,
    /// Path of file that contains X509 certificate in PEM format.
    #[clap(long, default_value = "")]
    pub cert: PathBuf,
    /// Path of file that contains X509 key in PEM format.
    #[clap(long, default_value = "")]
    pub key: PathBuf,
}

pub fn execute_restore_command(cmd: RestoreCommand) {
    match cmd.command {
        Tikv(args) => execute_restore_tikv(args),
        Pd(args) => execute_restore_pd(args),
        Keyspace(args) => execute_restore_keyspace(args),
    }
}

fn execute_restore_tikv(args: RestoreTikvArgs) {
    let config = get_restore_tikv_config_from_args(&args);
    restore_tikv(
        &config,
        args.name,
        args.store_id,
        args.new_store_id_delta,
        &args.path,
    );
}

fn execute_restore_pd(args: RestorePdArgs) {
    let config = get_restore_pd_config_from_args(&args);
    restore_pd(config, args.name);
}

fn execute_restore_keyspace(args: RestoreKeyspaceArgs) {
    let target_keyspace = args
        .target_keyspace_name
        .as_deref()
        .unwrap_or(&args.keyspace_name);
    match execute_restore_keyspace_impl(&args) {
        Ok(_) => {
            step!(
                "Restore keyspace {}->{} succeed",
                args.keyspace_name,
                target_keyspace
            );
        }
        Err(err) => {
            step_error!(
                "Restore keyspace {}->{} error: {:?}",
                args.keyspace_name,
                target_keyspace,
                err
            );
            panic!("execute restore keyspace error: {:?}", err);
        }
    }
}

fn execute_restore_keyspace_impl(
    args: &RestoreKeyspaceArgs,
) -> native_br::Result<RestoredKeyspace> {
    let config: restore::RestoreConfig = get_restore_keyspace_config_from_args(args);
    let pd_client: Arc<dyn PdClient> = Arc::new(create_pd_client(&config.security, &config.pd));
    let dfs_config = config.dfs.clone();
    let s3fs = S3Fs::new(
        dfs_config.prefix,
        dfs_config.s3_endpoint,
        dfs_config.s3_key_id,
        dfs_config.s3_secret_key,
        dfs_config.s3_region,
        dfs_config.s3_bucket,
    );
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(8)
        .enable_all()
        .build()
        .unwrap();
    let reporter = Arc::new(CliRestoreStepReporter::default());

    let target_keyspace_name = args
        .target_keyspace_name
        .as_deref()
        .unwrap_or(&args.keyspace_name);
    restore_keyspace_with_cfg(
        config,
        &args.keyspace_name,
        target_keyspace_name,
        &args.name,
        args.working_path.as_deref(),
        Arc::new(s3fs),
        pd_client,
        &runtime,
        None,
        reporter,
    )
}

fn get_restore_pd_config_from_args(args: &RestorePdArgs) -> RestoreConfig {
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
    config.new_store_id_delta = args.new_store_id_delta;
    config.dfs.override_from_env();
    config.security.master_key.override_from_env();
    config
}

fn get_restore_tikv_config_from_args(args: &RestoreTikvArgs) -> RestoreConfig {
    let mut config = RestoreConfig::default();
    if args.config.exists() {
        let data = std::fs::read(args.config.clone()).expect("failed to read config file");
        config = toml::from_slice(&data).unwrap();
    }
    // Override config file from args
    config.wal_target_size = args.wal_target_size;
    config.dfs.override_from_env();
    config.security.master_key.override_from_env();
    config.skip_resolve_lock = false;
    config
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
    config.security.master_key.override_from_env();
    config.skip_resolve_lock = false;
    config
}

#[derive(Default)]
struct CliRestoreStepReporter {}

impl ReportRestoreStepTrait for CliRestoreStepReporter {
    fn report_step(&self, _step: RestoreStep) {
        // TODO: friendly output for cli use.
    }
}
