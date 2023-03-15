// Copyright 2022 TiKV Project Authors. Licensed under Apache-2.0.

use std::{path::PathBuf, sync::Arc, time::Duration};

use clap::Args;
use native_br::{
    common::create_pd_client,
    truncate_ts::{truncate_ts_with_cfg, TruncateTsConfig},
};

const DEFAULT_TRUNCATE_TS_TIMEOUT: u64 = 5 * 60; // 5 min

#[derive(Args)]
pub struct TruncateTsArgs {
    /// The path of the config file.
    #[clap(long, default_value = "")]
    pub config: PathBuf,
    /// The truncate ts
    #[clap(long)]
    pub truncate_ts: u64,
    /// The timeout in seconds
    #[clap(long, default_value_t = DEFAULT_TRUNCATE_TS_TIMEOUT)]
    pub timeout: u64,
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

pub fn execute_truncate_ts(args: TruncateTsArgs) {
    let config = get_truncate_ts_config_from_args(&args);
    let pd_client = create_pd_client(&config.security, &config.pd);
    truncate_ts_with_cfg(
        config,
        Arc::new(pd_client),
        args.truncate_ts,
        Duration::from_secs(args.timeout),
        None,
    )
    .unwrap();
}

fn get_truncate_ts_config_from_args(args: &TruncateTsArgs) -> TruncateTsConfig {
    let mut config = TruncateTsConfig::default();
    if args.config.exists() {
        let data = std::fs::read(args.config.clone()).expect("failed to read config file");
        config = toml::from_slice(&data).unwrap();
    }
    // override from args
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
    config.skip_resolve_lock = false;
    config
}
