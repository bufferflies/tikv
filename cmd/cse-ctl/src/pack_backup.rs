// Copyright 2023 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    path::PathBuf,
    sync::{Arc, Mutex},
};

use clap::Args;
use native_br::packing::{self, PackBackupStep, PackConfig, ReportPackBackupStepTrait};
use tikv_util::{crit, info, time::Instant};

#[derive(Args)]
/// Arguments for the pack backup command
pub struct PackBackupArgs {
    /// The path of the config file
    #[clap(long, default_value = "")]
    pub config: PathBuf,

    /// The name of the backup to pack
    #[clap(long)]
    pub backup_name: String,

    /// The name of the keyspace
    #[clap(long)]
    pub keyspace_name: String,

    /// Use offline PD mode, which is unsafe but allows packing backups
    /// without connecting to a PD instance. When this were set,
    /// will bypass PD and allocate from current local TSO to the new flushed
    /// SSTs.
    #[clap(long)]
    pub offline: bool,

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

    /// The directory to store temporary data
    #[clap(long)]
    pub data_dir: Option<PathBuf>,
}

impl PackBackupArgs {
    fn verify(&self) -> Result<(), String> {
        if self.offline && !self.pd.is_empty() {
            return Err(format!(
                "you specified both `--offline`({}) and `--pd`({:?}), please remove one of them",
                self.offline, self.pd
            ));
        }

        Ok(())
    }
}

pub struct CliReportPackBackupStep {
    last_step_start_time: Mutex<Instant>,
}

impl ReportPackBackupStepTrait for CliReportPackBackupStep {
    fn report_step(&self, step: PackBackupStep) {
        let mut last_step_time = self.last_step_start_time.lock().unwrap();
        let elapsed = last_step_time.saturating_elapsed();
        *last_step_time = Instant::now();
        info!("Starting a new step."; "step" => ?step, "last_step_takes" => ?elapsed);
    }
}

impl Default for CliReportPackBackupStep {
    fn default() -> Self {
        Self {
            last_step_start_time: Mutex::new(Instant::now()),
        }
    }
}

impl Drop for CliReportPackBackupStep {
    fn drop(&mut self) {
        let last_step_time = self.last_step_start_time.get_mut().unwrap();
        let elapsed = last_step_time.saturating_elapsed();
        info!("Finishing pack backup."; "takes" => ?elapsed);
    }
}

pub fn execute_pack_backup(args: PackBackupArgs) {
    if let Err(err) = args.verify() {
        crit!("Failed to verify arguments."; "err" => err);
    }

    let mut config = PackConfig::default();
    if args.config.exists() {
        let data = std::fs::read(args.config.clone()).expect("failed to read config file");
        config = toml::from_slice(&data).unwrap();
    }

    // override from args and ENV
    config.backup_name = args.backup_name.clone();
    config.keyspace_name = args.keyspace_name;
    config.offline = args.offline;
    config.data_dir = args.data_dir;

    if !args.pd.is_empty() {
        config.pd_config.endpoints = args.pd.split(',').map(|x| x.to_owned()).collect();
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

    info!("Begin packing backup with config {:?}", config);

    let reporter = Arc::new(CliReportPackBackupStep::default());
    match packing::pack_backup_with_cfg(reporter, config) {
        Ok(meta) => {
            info!("Successfully packed backup: {}", args.backup_name);
            info!("Copyable path: {}", meta.copyable_path);
        }
        Err(e) => {
            panic!("Failed to pack backup: {:?}", e);
        }
    }
}
