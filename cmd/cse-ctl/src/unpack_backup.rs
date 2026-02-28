// Copyright 2023 TiKV Project Authors. Licensed under Apache-2.0.

use std::{path::PathBuf, sync::Arc};

use clap::Args;
use kvengine::dfs::Dfs;
use native_br::packing::{MigratePackEnv, NoopReporter, UnpackRun};
use tikv_util::{config::ReadableSize, info};

use crate::common::CommonConfig;

#[derive(Args)]
/// Arguments for the unpack backup command
pub struct UnpackBackupArgs {
    /// The path of the config file
    #[clap(long, default_value = "")]
    pub config: PathBuf,

    /// The S3 path to the exotic (source) packed backup
    /// Format: bucket/path/to/packed_backup.meta
    #[clap(long)]
    pub exotic_path: String,

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

    /// Max throughput for unpack copy process, e.g. 500MiB. 0 to disable the
    /// limit.
    #[clap(long, default_value = "0")]
    pub max_throughput: ReadableSize,
}

pub fn execute_unpack_backup(args: UnpackBackupArgs) {
    info!(
        "Begin unpacking backup with exotic path: {}",
        args.exotic_path
    );

    let mut config = CommonConfig::from_args(&args.config, &args.pd);

    // Override security settings from args if provided
    if args.cacert.exists() {
        config.security.ca_path = args.cacert.to_str().unwrap().to_owned();
    }
    if args.cert.exists() {
        config.security.cert_path = args.cert.to_str().unwrap().to_owned();
    }
    if args.key.exists() {
        config.security.key_path = args.key.to_str().unwrap().to_owned();
    }

    let ctx = config.create_context();
    let s3fs = Arc::new(ctx.s3fs);
    let pd_client = ctx.pd_client;

    // Use the S3FS runtime to execute the async operations
    let handle = s3fs.get_runtime().handle().clone();

    let load_exotic_backup = async {
        info!("Loading exotic packed backup from: {}", args.exotic_path);

        let migrate_env = MigratePackEnv::load_exotic(s3fs.clone(), &args.exotic_path)
            .await?
            .with_rate_limit(args.max_throughput, "unpack", 0);
        let mut unpack_run = UnpackRun::new(migrate_env, pd_client, Arc::new(NoopReporter));
        let result = unpack_run.execute().await?;
        Ok::<String, native_br::error::Error>(result)
    };
    match handle.block_on(load_exotic_backup) {
        Ok(backup_name) => {
            info!("Successfully unpacked backup!");
            info!("New packed backup saved as: {}", backup_name);
        }
        Err(e) => {
            panic!("Failed to unpack backup: {:?}", e);
        }
    }
}
