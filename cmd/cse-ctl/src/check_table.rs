// Copyright 2023 TiKV Project Authors. Licensed under Apache-2.0.

use std::path::PathBuf;

use check_table::check_table::{CheckTableConfig, CheckTableParams};
use clap::Args;

#[derive(Args)]
pub struct CheckTableArgs {
    /// The path of the config file.
    #[clap(long, default_value = "")]
    pub config: PathBuf,
    /// PD endpoints, use `,` to separate multiple PDs
    #[clap(long, default_value_t = String::new())]
    pub pd: String,
    /// The backup name to check table.
    #[clap(long)]
    pub backup_name: String,
    /// The keyspace id to check table.
    #[clap(long, default_value_t = 0)]
    pub keyspace_id: u32,
    #[clap(long, default_value_t = 0)]
    pub timestamp: u64,
    /// Effective only when `all` is true.
    #[clap(long)]
    pub starts_from_keyspace_id: Option<u32>,
    /// Effective only on first keyspace. Used as partition id as well.
    #[clap(long)]
    pub starts_from_table_id: Option<i64>,
    /// Effective only on first keyspace. Used as partition id as well.
    #[clap(long)]
    pub ends_to_table_id: Option<i64>,
}

fn get_config_from_args(args: &CheckTableArgs) -> CheckTableConfig {
    let mut config = CheckTableConfig::default();
    if args.config.exists() {
        let data = std::fs::read(args.config.as_path()).expect("failed to read config file");
        config = toml::from_slice(&data).unwrap();
    }
    // override from args and ENV
    if !args.pd.is_empty() {
        config.pd.endpoints = args.pd.split(',').map(|x| x.to_owned()).collect();
    }
    if !args.backup_name.is_empty() {
        config.backup_name = args.backup_name.clone();
    }
    if args.keyspace_id != 0 && !config.keyspace_ids.contains(&args.keyspace_id) {
        config.keyspace_ids.push(args.keyspace_id);
    }
    if args.timestamp != 0 {
        config.timestamp = args.timestamp;
    }
    config.dfs.override_from_env();
    config.security.override_from_env();
    if config.data_dir.is_empty() {
        config.data_dir = ".".to_string();
    }
    config
}

pub(crate) fn execute_check_table(args: CheckTableArgs) {
    let config = get_config_from_args(&args);
    let params = CheckTableParams {
        starts_from_keyspace_id: args.starts_from_keyspace_id,
        starts_from_table_id: args.starts_from_table_id,
        ends_to_table_id: args.ends_to_table_id,
    };
    check_table::execute_check_table(config, params);
}
