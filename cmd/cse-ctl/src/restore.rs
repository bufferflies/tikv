// Copyright 2022 TiKV Project Authors. Licensed under Apache-2.0.

use std::path::PathBuf;

use clap::Args;
use kvengine::dfs::{DFSConfig, DFS, S3FS};
use protobuf::Message;
use rfenginepb::ClusterBackupMeta;
use security::SecurityConfig;
use slog_global::error;

#[derive(Args)]
pub struct RestoreArgs {
    /// The path of the config file.
    #[clap(long)]
    pub config: PathBuf,
    /// The name of the backup file.
    #[clap(long)]
    pub name: String,
    /// The local path to load store files.
    #[clap(long, default_value_t)]
    pub path: String,
    /// The store id to restore.
    #[clap(long, default_value_t)]
    pub store_id: u64,
}

pub(crate) fn execute_restore(args: RestoreArgs) {
    let result = std::fs::read(args.config);
    if result.is_err() {
        error!("failed to read config file {:?}", result.unwrap_err());
        return;
    }
    let data = result.unwrap();
    let config: RestoreConfig = toml::from_slice(&data).unwrap();
    let dfs_conf = config.dfs.clone();
    let backup_key = format!("{}/backup/{}", dfs_conf.prefix, args.name);
    let s3fs = S3FS::new(
        dfs_conf.prefix,
        dfs_conf.s3_endpoint,
        dfs_conf.s3_key_id,
        dfs_conf.s3_secret_key,
        dfs_conf.s3_region,
        dfs_conf.s3_bucket,
    );
    let runtime = s3fs.get_runtime();
    let data = runtime
        .block_on(s3fs.get_object(backup_key, args.name))
        .unwrap();
    let mut cluster_backup = ClusterBackupMeta::new();
    cluster_backup.merge_from_bytes(&data).unwrap();
    println!(
        "cluster_id {}, alloc_id {}, use pd recover",
        cluster_backup.cluster_id, cluster_backup.alloc_id
    );
    if args.store_id > 0 {
        rfengine::restore(
            Box::new(s3fs),
            &cluster_backup,
            args.store_id,
            &PathBuf::from(&args.path),
        );
    }
}

#[derive(Clone, Serialize, Deserialize, PartialEq, Debug, Default)]
#[serde(default)]
#[serde(rename_all = "kebab-case")]
pub struct RestoreConfig {
    pub pd: pd_client::Config,
    pub security: SecurityConfig,
    pub dfs: DFSConfig,
}
