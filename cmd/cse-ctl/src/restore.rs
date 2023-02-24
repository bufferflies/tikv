// Copyright 2022 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    path::PathBuf,
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use clap::{Args, Subcommand};
use etcd_client::{Compare, CompareOp, Txn, TxnOp};
use kvengine::dfs::{DFSConfig, DFS, S3FS};
use protobuf::Message;
use rfenginepb::ClusterBackupMeta;
use security::SecurityConfig;
use slog_global::info;

use crate::{
    common::generate_etcd_connect_opt,
    restore::Commands::{Keyspace, Tikv, PD},
    restore_tenant::execute_restore_keyspace,
};

const PD_ROOT_PATH: &str = "/pd";
const PD_CLUSTER_ID_PATH: &str = "/pd/cluster_id";
const MAX_TXN_OPTS: usize = 128; // Default configuration in etcd server.

#[derive(Args)]
pub struct RestoreCommand {
    #[clap(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Restore TiKV raft store data.
    Tikv(RestoreTiKVArgs),
    /// Restore PD meta data.
    PD(RestorePDArgs),
    /// Restore Keyspace data.
    Keyspace(RestoreKeyspaceArgs),
}

#[derive(Args)]
pub struct RestoreTiKVArgs {
    /// The path of the config file.
    #[clap(long, default_value = "")]
    pub config: PathBuf,
    /// The name of the backup file.
    #[clap(long)]
    pub name: String,
    /// The local path to load store files.
    #[clap(long)]
    pub path: String,
    /// The store id to restore.
    #[clap(long)]
    pub store_id: u64,
}

#[derive(Args)]
struct RestorePDArgs {
    /// The path of the config file.
    #[clap(long, default_value = "")]
    pub config: PathBuf,
    /// The name of the backup file.
    #[clap(long)]
    pub name: String,
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

#[derive(Args)]
pub struct RestoreKeyspaceArgs {
    /// The path of the config file.
    #[clap(long)]
    pub config: PathBuf,
    /// The name of the backup file.
    #[clap(long)]
    pub name: String,
    /// The keyspace to restore.
    #[clap(long)]
    pub keyspace_name: String,
    /// The local working path for temporary files during restore.
    #[clap(long)]
    pub working_path: Option<String>,
}

pub fn execute_restore_command(cmd: RestoreCommand) {
    match cmd.command {
        Tikv(args) => execute_restore_tikv(args),
        PD(args) => execute_restore_pd(args),
        Keyspace(args) => execute_restore_keyspace(args),
    }
}

fn execute_restore_tikv(args: RestoreTiKVArgs) {
    let config = get_restore_tikv_config_from_args(&args);
    restore_tikv(&config, args.name, args.store_id, &args.path);
}

pub fn restore_tikv(config: &RestoreConfig, name: String, store_id: u64, path: &str) {
    let (cluster_backup, s3fs) = get_cluster_backup_meta(config, name);
    if store_id > 0 {
        rfengine::restore(
            Arc::new(s3fs),
            &cluster_backup,
            store_id,
            &PathBuf::from(path),
            None,
        );
    }
}

fn execute_restore_pd(args: RestorePDArgs) {
    let config = get_restore_pd_config_from_args(&args);
    let (cluster_backup, _) = get_cluster_backup_meta(&config, args.name);
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(restore_pd_keyspace_meta(&config, &cluster_backup));
}

pub(crate) fn get_cluster_backup_meta(
    config: &RestoreConfig,
    name: String,
) -> (ClusterBackupMeta, S3FS) {
    let dfs_conf = config.dfs.clone();
    let backup_key = format!("{}/backup/{}", dfs_conf.prefix, name);
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
        .block_on(s3fs.get_object(backup_key, name, engine_traits::GetObjectOptions::default()))
        .unwrap();
    let mut cluster_backup = ClusterBackupMeta::new();
    cluster_backup.merge_from_bytes(&data).unwrap();
    info!(
        "Restore cluster_id {}, alloc_id {}, backup_ts {}, safe_ts {}, store cnt {}",
        cluster_backup.cluster_id,
        cluster_backup.alloc_id,
        cluster_backup.backup_ts,
        cluster_backup.safe_ts,
        cluster_backup.stores.len()
    );
    (cluster_backup, s3fs)
}

// Mainly ref `recoverFromNewPDCluster` in `pd-recover`.
async fn restore_pd_keyspace_meta(config: &RestoreConfig, meta: &ClusterBackupMeta) {
    let option = generate_etcd_connect_opt(&config.security).unwrap();
    let mut etcd_client = etcd_client::Client::connect(&config.pd.endpoints, Some(option))
        .await
        .unwrap();
    let mut txn_opts = Vec::with_capacity(4);
    let root_path = format!("{}/{}", PD_ROOT_PATH, meta.cluster_id);
    // recover cluster_id
    txn_opts.push(TxnOp::put(
        PD_CLUSTER_ID_PATH.as_bytes().to_vec(),
        meta.cluster_id.to_be_bytes().to_vec(),
        None,
    ));
    // recover alloc id
    let alloc_id_path = format!("{}/{}", root_path, "alloc_id");
    txn_opts.push(TxnOp::put(
        alloc_id_path.as_bytes().to_vec(),
        meta.alloc_id.to_be_bytes().to_vec(),
        None,
    ));
    // recover meta of cluster
    let cluster_raft_path = format!("{}/{}", root_path, "raft");
    let cluster_meta = kvproto::metapb::Cluster {
        id: meta.cluster_id,
        ..Default::default()
    };
    txn_opts.push(TxnOp::put(
        cluster_raft_path.as_bytes().to_vec(),
        cluster_meta.write_to_bytes().unwrap(),
        None,
    ));
    // set raft bootstrap time
    let raft_bootstrap_time_path = format!(
        "{}/{}/{}",
        cluster_raft_path, "status", "raft_bootstrap_time"
    );
    let cur_nano = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64;
    txn_opts.push(TxnOp::put(
        raft_bootstrap_time_path.as_bytes().to_vec(),
        cur_nano.to_be_bytes().to_vec(),
        None,
    ));
    let resp = etcd_client
        .txn(
            Txn::new()
                .when(
                    &[Compare::create_revision(
                        cluster_raft_path,
                        CompareOp::Equal,
                        0,
                    )][..],
                )
                .and_then(txn_opts),
        )
        .await
        .unwrap();
    if !resp.succeeded() {
        panic!(
            "Failed to restore pd keyspace meta, please JUST start new pd-server(s) without tikv nodes."
        );
    }
    // recover key space meta
    let meta_cnt = meta.keyspace_meta.len();
    let mut txn_opts = Vec::with_capacity(std::cmp::min(meta_cnt, MAX_TXN_OPTS));
    let mut idx = 0;
    for (key, value) in &meta.keyspace_meta {
        txn_opts.push(TxnOp::put(key.to_owned(), value.to_owned(), None));
        idx += 1;
        if txn_opts.len() == MAX_TXN_OPTS || idx == meta_cnt {
            // There is no batch put interface now, use txn to do batch.
            let batch_cnt = txn_opts.len();
            let resp = etcd_client
                .txn(Txn::new().and_then(txn_opts))
                .await
                .unwrap();
            if !resp.succeeded() {
                panic!(
                    "Fail to restore pd meta, cur idx {} batch {} total {}",
                    idx - batch_cnt,
                    batch_cnt,
                    meta_cnt
                );
            }
            txn_opts = Vec::with_capacity(std::cmp::min(meta_cnt - idx, MAX_TXN_OPTS));
        }
    }
    info!(
        "Restore PD {} meta data(revision: {}) of cluster {} successfully",
        meta_cnt, meta.meta_revision, meta.cluster_id
    );
}

#[derive(Clone, Serialize, Deserialize, PartialEq, Debug, Default)]
#[serde(default)]
#[serde(rename_all = "kebab-case")]
pub struct RestoreConfig {
    pub pd: pd_client::Config,
    pub security: SecurityConfig,
    pub dfs: DFSConfig,
}

fn get_restore_pd_config_from_args(args: &RestorePDArgs) -> RestoreConfig {
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
    config
}

fn get_restore_tikv_config_from_args(args: &RestoreTiKVArgs) -> RestoreConfig {
    let mut config = RestoreConfig::default();
    if args.config.exists() {
        let data = std::fs::read(args.config.clone()).expect("failed to read config file");
        config = toml::from_slice(&data).unwrap();
    }
    config.dfs.override_from_env();
    config
}
