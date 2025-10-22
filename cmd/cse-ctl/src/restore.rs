// Copyright 2022 TiKV Project Authors. Licensed under Apache-2.0.

use std::{path::PathBuf, sync::Arc};

use chrono::{DateTime, NaiveDateTime, Utc};
use clap::{Args, Subcommand};
use kvengine::dfs::{Dfs, S3Fs};
use native_br::{
    common::{create_pd_client, now},
    restore,
    restore::{restore_pd, restore_tikv, RestoreConfig},
    restore_keyspace::{
        restore_keyspace_with_cfg, ReportRestoreStepTrait, RestoreStep, RestoredKeyspace,
    },
    step, step_error,
};
use pd_client::{
    pd_control::{KeyspaceMeta, PdControl},
    PdClient,
};
use security::SecurityManager;
use slog_global::{error, info};
use tikv_util::{box_try, config::ReadableSize};

use crate::{
    backup::show_backup_summary,
    restore::Commands::{Keyspace, Pd, Tikv},
};

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
    /// Truncate newer data than given ts.
    #[clap(long)]
    pub truncate_ts: Option<u64>,
    /// The local working path for temporary files during restore.
    ///
    /// Use current path if not specified.
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
    /// Keep silent and NO confirmation when restore. Use with caution.
    #[clap(long)]
    pub silent: bool,
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
    if std::env::var("LOG_FILE").is_err() {
        std::env::set_var("LOG_FILE", "cse-ctl.log");
    }
    // The logger for test is enough.
    ::test_util::init_log_for_test();

    let target_keyspace = args
        .target_keyspace_name
        .as_deref()
        .unwrap_or(&args.keyspace_name);
    let config: restore::RestoreConfig = get_restore_keyspace_config_from_args(&args);

    if !args.silent {
        show_restore_keyspace_info(&args, &config, target_keyspace);
        if !warning_prompt("WARNING: this action will OVERWRITE user data", "YES") {
            return;
        }
    }

    match execute_restore_keyspace_impl(&args, config) {
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
            std::process::exit(1);
        }
    }
}

fn execute_restore_keyspace_impl(
    args: &RestoreKeyspaceArgs,
    config: restore::RestoreConfig,
) -> native_br::Result<RestoredKeyspace> {
    let working_path = match &args.working_path {
        Some(p) => {
            let path = PathBuf::from(p);
            box_try!(std::fs::create_dir_all(&path));
            path
        }
        None => box_try!(std::env::current_dir()),
    };

    let pd_client: Arc<dyn PdClient> = Arc::new(create_pd_client(&config.security, &config.pd));
    let dfs_config = config.dfs.clone();
    let s3fs = S3Fs::new_from_config(dfs_config);
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
        Some(working_path),
        Arc::new(s3fs),
        pd_client,
        &runtime,
        args.truncate_ts,
        reporter,
        None,
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
    config.security.override_from_env();
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
    config.security.override_from_env();
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
    config.security.override_from_env();
    config
}

#[derive(Default)]
struct CliRestoreStepReporter {}

impl ReportRestoreStepTrait for CliRestoreStepReporter {
    fn report_step(&self, _step: RestoreStep) {
        // TODO: friendly output for cli use.
    }
}

fn show_restore_keyspace_info(
    args: &RestoreKeyspaceArgs,
    config: &restore::RestoreConfig,
    target_keyspace_name: &str,
) {
    let security_mgr = Arc::new(
        SecurityManager::new(&config.security)
            .unwrap_or_else(|e| panic!("failed to create security manager: {:?}", e)),
    );
    let pd_ctl = PdControl::new(config.pd.clone(), security_mgr).unwrap();

    let dfs_config = config.dfs.clone();
    let s3fs = S3Fs::new_from_config(dfs_config);
    let rt = s3fs.get_runtime();

    println!("Restore Keyspace");
    println!();

    let cluster_backup = native_br::restore::get_cluster_backup_meta(&s3fs, args.name.clone());
    show_backup_summary(&cluster_backup, &args.name, 0);
    println!();

    let inplace = args.keyspace_name == target_keyspace_name;
    let keyspace = rt
        .block_on(pd_ctl.get_keyspace_by_name(&args.keyspace_name))
        .unwrap();
    if inplace {
        println!("[Keyspace]");
    } else {
        println!("[Keyspace (source)]");
    }
    show_keyspace(&keyspace, 2);
    println!();
    if !inplace {
        let target_keyspace = rt
            .block_on(pd_ctl.get_keyspace_by_name(target_keyspace_name))
            .unwrap();
        println!("[Keyspace (target)]");
        show_keyspace(&target_keyspace, 2);
        println!();
    }

    if let Some(truncate_ts) = args.truncate_ts {
        println!("truncate_ts: {}", truncate_ts);
        println!();
    }
}

fn show_keyspace(ks: &KeyspaceMeta, indent: usize) {
    let indent = " ".repeat(indent);
    println!("{}id: {}", indent, ks.id);
    println!("{}name: {}", indent, ks.name);
    println!("{}state: {}", indent, ks.state);

    let created_at = NaiveDateTime::from_timestamp_opt(ks.created_at as i64, 0)
        .expect("parse keyspace.created_at failed");
    let created_at = DateTime::<Utc>::from_utc(created_at, Utc);
    println!("{}created_at: {}", indent, created_at.to_rfc3339());

    println!("{}config:", indent);
    for (k, v) in &ks.config {
        println!("{}  {}: {}", indent, k, v);
    }
}

fn warning_prompt(message: &str, expected: &str) -> bool {
    println!("{}", message);
    println!("Type \"{}\" to continue, anything else to exit", expected);
    let mut answer = String::new();
    std::io::stdin().read_line(&mut answer).unwrap();
    if answer.trim_end_matches('\n') == expected {
        true
    } else {
        println!("exit.");
        false
    }
}
