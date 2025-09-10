// Copyright 2025 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::Arc,
    time::Duration,
};

use clap::Args;
use cloud_encryption::MasterKey;
use cloud_worker::{SchemaManager, SchemaManagerConfig, SchemaMgrContext};
use colored::*;
use http::{Request, StatusCode};
use hyper::Body;
use kvengine::{
    context::{new_meta_file_cache, IaCtx, PrepareType, SnapCtx},
    dfs::{DFSConfig, Dfs, S3Fs},
    ia::{manager::IaManager, util::IaConfig},
    table::{
        columnar::{Block, ColumnarFilterReader, GLOBAL_COMMON_HANDLE_END},
        file::FdCache,
        sstable::BlockCache,
    },
    txn_chunk_manager::{TxnChunkManager, TxnChunkManagerConfig},
    ShardStatsLite, SnapAccess,
};
use kvproto::{coprocessor::DelegateResponse, metapb::Store};
use native_br::common::{create_pd_client, send_request_to_store};
use pd_client::PdClient;
use protobuf::Message;
use security::{SecurityConfig, SecurityManager};
use tikv_util::{
    config::AbsoluteOrPercentSize, error, info, memory::MemoryLimiter, worker_pool::WorkerPool,
};
use tokio::{fs::OpenOptions, io::AsyncWriteExt};

const CHECK_RESULT_FILE: &str = "check_columnar_result.txt";

#[derive(Args)]
pub struct CheckColumnarArgs {
    /// The path of the config file.
    #[clap(long, default_value = "")]
    pub config: PathBuf,
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
    /// The keyspace id to check columnar, if not set, check all keyspaces.
    #[clap(long, default_value_t = 0)]
    pub keyspace_id: u32,
    /// The keyspace id start to check columnar, if not set, check from 0.
    #[clap(long, default_value_t = 0)]
    pub keyspace_id_start: u32,
    /// The path of the schema file.
    #[clap(long, default_value = "")]
    pub schemas_path: PathBuf,
    /// The path of the working directory.
    #[clap(long, default_value = "/tmp/cse-ctl-check-columnar")]
    pub working_dir: PathBuf,
}

// This command is used to check the pk column loss in columnar files.
// See https://github.com/tidbcloud/cloud-storage-engine/pull/3210 for more details.
pub(crate) fn execute_check_columnar(args: CheckColumnarArgs) {
    let config = CheckColumnarConfig::from_args(&args);
    let pd_client = Arc::new(create_pd_client(&config.security, &config.pd));
    let dfs_cfg = config.dfs.clone();
    let s3fs = Arc::new(S3Fs::new_from_config(dfs_cfg));
    let runtime = s3fs.get_runtime();
    let ctx = Arc::new(SchemaMgrContext {
        s3fs: s3fs.clone(),
        pd: pd_client,
    });
    let security_mgr = Arc::new(SecurityManager::new(&config.security).unwrap());
    let mut schema_mgr_config = SchemaManagerConfig::default();
    if args.schemas_path.exists() {
        schema_mgr_config.dir = PathBuf::from(args.schemas_path.to_str().unwrap());
    }
    let schema_manager = SchemaManager::new(
        ctx.clone(),
        security_mgr.clone(),
        config.security.clone(),
        schema_mgr_config,
        &config.pd.endpoints,
    );

    runtime.block_on(check_columnar(ctx, &schema_manager, security_mgr, &config));
}

async fn check_columnar(
    ctx: Arc<SchemaMgrContext>,
    schema_manager: &SchemaManager,
    security_mgr: Arc<SecurityManager>,
    config: &CheckColumnarConfig,
) {
    let (stores, _) = schema_manager.get_tikv_stores();
    if stores.is_empty() {
        panic!("no tikv stores found");
    }
    let mut keyspace_stats = HashMap::new();
    if let Err(e) = schema_manager
        .refresh_keyspace_stats(&mut keyspace_stats, &stores)
        .await
    {
        panic!("refresh keyspace stats error: {:?}", e);
    }

    // sort keyspace_stats by keyspace_id, only collect shards has columnar_tables
    let mut sorted_keyspace_stats = keyspace_stats
        .into_iter()
        .filter_map(|(keyspace_id, shard_stats)| {
            let filtered_shards: Vec<_> = shard_stats
                .into_iter()
                .filter(|shard| shard.columnar_tables > 0)
                .collect();
            if filtered_shards.is_empty() {
                None
            } else {
                Some((keyspace_id, filtered_shards))
            }
        })
        .collect::<Vec<_>>();
    sorted_keyspace_stats.sort_by_key(|(keyspace_id, _)| *keyspace_id);
    if config.keyspace_id > 0 {
        sorted_keyspace_stats.retain(|(keyspace_id, _)| *keyspace_id == config.keyspace_id);
    }
    if config.keyspace_id_start > 0 {
        sorted_keyspace_stats.retain(|(keyspace_id, _)| *keyspace_id >= config.keyspace_id_start);
    }
    let total_keyspace_count = sorted_keyspace_stats.len();
    let mut total_region_idx = 0;
    let total_regions = sorted_keyspace_stats
        .iter()
        .map(|(_, shard_stats)| shard_stats.len())
        .sum::<usize>();
    let master_key = config.security.new_master_key().await;
    let txn_chunk_manager = TxnChunkManager::new(
        vec![],
        ctx.s3fs.clone(),
        BlockCache::None,
        None,
        WorkerPool::Handle(ctx.s3fs.get_runtime().handle().clone()),
        TxnChunkManagerConfig::default(),
    );
    let ia_config = IaConfig {
        mem_cap: AbsoluteOrPercentSize::Percent(20.0),
        disk_cap: AbsoluteOrPercentSize::Percent(50.0),
        ..Default::default()
    };
    let ia_mgr = build_ia_mgr(
        ctx.s3fs.clone(),
        ctx.s3fs.get_runtime(),
        &config.working_dir,
        &ia_config,
    );
    for (i, (keyspace_id, shard_stats)) in sorted_keyspace_stats.into_iter().enumerate() {
        let total_region_count_in_keyspace = shard_stats.len();
        for (shard_idx, shard) in shard_stats.iter().enumerate() {
            total_region_idx += 1;
            info!(
                "check columnar for keyspace_id: {} ({}/{}), shard {} ({}/{}), total regions {}/{}",
                keyspace_id,
                i + 1,
                total_keyspace_count,
                shard.id,
                shard_idx + 1,
                total_region_count_in_keyspace,
                total_region_idx,
                total_regions,
            );
            match check_columnar_for_shard(
                ctx.clone(),
                schema_manager,
                security_mgr.clone(),
                txn_chunk_manager.clone(),
                ia_mgr.clone(),
                &config.working_dir,
                &master_key,
                keyspace_id,
                shard,
                &stores,
            )
            .await
            {
                Ok(true) => {
                    info!(
                        "check {} for shard {}:{}:{}",
                        "SUCCESS".green().bold(),
                        keyspace_id,
                        shard.id,
                        shard.ver
                    );
                }
                Ok(false) => {
                    error!(
                        "check {} for shard {}:{}:{}",
                        "FAILED".red().bold(),
                        keyspace_id,
                        shard.id,
                        shard.ver
                    );
                    // append keyspace_id to result file
                    let mut file = OpenOptions::new()
                        .create(true)
                        .append(true)
                        .open(CHECK_RESULT_FILE)
                        .await
                        .unwrap();
                    file.write_all(
                        format!("{}:{}:{}\n", keyspace_id, shard.id, shard.ver).as_bytes(),
                    )
                    .await
                    .unwrap();
                }
                Err(e) => {
                    error!(
                        "check {} for shard {}:{}:{}: {}",
                        "ERROR".bright_yellow().bold(),
                        keyspace_id,
                        shard.id,
                        shard.ver,
                        e
                    );
                }
            }
        }
    }
}

async fn check_columnar_for_shard(
    ctx: Arc<SchemaMgrContext>,
    schema_manager: &SchemaManager,
    security_mgr: Arc<SecurityManager>,
    txn_chunk_manager: TxnChunkManager,
    ia_mgr: IaManager,
    working_dir: &Path,
    master_key: &MasterKey,
    keyspace_id: u32,
    shard: &ShardStatsLite,
    stores: &[Store],
) -> Result<bool, String> {
    let Some(schema_file) = schema_manager
        .get_schema_file_from_local(keyspace_id)
        .unwrap()
    else {
        return Err(format!(
            "no schema file found for keyspace_id: {}",
            keyspace_id
        ));
    };
    // Check if common index in schema file. Skip if no.
    let table_ids = schema_file
        .iter_tables()
        .filter_map(|(table_id, schema)| {
            if schema.is_common_handle() {
                Some(table_id)
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    if table_ids.is_empty() {
        return Ok(true);
    }
    let memory_limiter = MemoryLimiter::new(u64::MAX, None);
    let Some(leader_store_id) = get_leader_store(ctx.pd.clone(), shard.id).await else {
        return Err(format!(
            "get leader store failed for shard {}:{}, ignore, region has no leader",
            shard.id, shard.ver
        ));
    };
    let store = stores
        .iter()
        .find(|store| store.get_id() == leader_store_id);
    let Some(store) = store else {
        return Err(format!(
            "store {} not found in stores for shard {}:{}",
            leader_store_id, shard.id, shard.ver
        ));
    };
    let mut delegate_resp = DelegateResponse::default();
    let status_addr = store.get_status_address();
    let uri = security_mgr
        .build_uri(format!(
            "{}/kvengine/snapshot/{}?shard_ver={}&start_ts={}",
            status_addr,
            shard.id,
            shard.ver,
            u64::MAX,
        ))
        .unwrap();
    let req = Request::get(uri.clone()).body(Body::empty()).unwrap();
    let Ok((resp_code, resp)) =
        send_request_to_store(req, store, security_mgr.as_ref(), Duration::from_secs(10)).await
    else {
        return Err(format!(
            "get snapshot failed for shard {}:{}",
            shard.id, shard.ver
        ));
    };
    if resp_code != StatusCode::OK {
        return Err(format!(
            "get snapshot failed for shard {}:{}, status code: {}",
            shard.id, shard.ver, resp_code
        ));
    }
    delegate_resp.merge_from_bytes(resp.as_ref()).unwrap();
    if delegate_resp.has_region_error()
        && (delegate_resp.get_region_error().has_not_leader()
            || delegate_resp.get_region_error().has_epoch_not_match())
    {
        return Err(format!(
            "get snapshot failed for shard {}:{} due to region error: {:?}, skip",
            shard.id,
            shard.ver,
            delegate_resp.get_region_error()
        ));
    }
    if delegate_resp.has_locked() {
        return Err(format!(
            "get snapshot failed for shard {}:{} due to locked {:?}, skip",
            shard.id,
            shard.ver,
            delegate_resp.get_locked()
        ));
    }
    let tag = format!("{}:{}", shard.id, shard.ver);
    let snap_ctx = SnapCtx {
        dfs: ctx.s3fs.clone(),
        master_key: master_key.clone(),
        block_cache: BlockCache::None,
        vector_index_cache: None,
        meta_file_cache: new_meta_file_cache(0),
        schema_files: None,
        txn_chunk_manager,
        ia_ctx: IaCtx::Enabled(ia_mgr, Arc::new(vec![working_dir.to_path_buf()])),
        prepare_type: PrepareType::ColumnarOnly,
        read_columnar: true,
    };
    let mut delegate_resp = DelegateResponse::default();
    delegate_resp.merge_from_bytes(resp.as_ref()).unwrap();
    let (snap, _) = SnapAccess::construct_snapshot(
        &tag,
        &snap_ctx,
        delegate_resp.get_mem_table_data(),
        delegate_resp.get_snapshot(),
        memory_limiter.clone(),
    )
    .await
    .unwrap();

    for table_id in snap.get_columnar_table_ids() {
        let Some(schema) = schema_file.get_table(table_id) else {
            error!(
                "table {} not found in schema file for shard {}:{}",
                table_id, shard.id, shard.ver
            );
            continue;
        };
        if !schema.with_columnar() || !schema.is_common_handle() {
            continue;
        }

        let mut columns = vec![];
        for col_id in &schema.pk_col_ids {
            let column = schema.find_column_by_id(*col_id).unwrap();
            columns.push(column.clone());
        }
        let schema = snap.new_schema_from_columns(table_id, &columns).unwrap();
        let mut columnar_reader = snap
            .new_columnar_mvcc_reader(table_id, &columns, None, u64::MAX, None)
            .unwrap()
            .unwrap();
        let mut block = Block::new(&schema);
        columnar_reader
            .set_handle_range(&[], GLOBAL_COMMON_HANDLE_END)
            .await
            .unwrap();
        let mut read_rows = columnar_reader.read_block(&mut block, 1024).await.unwrap();
        while read_rows > 0 {
            // check if pk col empty
            for i in 0..read_rows {
                let handle = block.get_handle_buf().get_not_null_value(i);
                let version = block.get_version_buf().get_not_null_value(i);
                for col in block.get_columns().iter() {
                    let col_data = col.get_not_null_value(i);
                    if col_data.is_empty() {
                        error!(
                            "check_failed pk col empty: handle: {}, version: {}, col_id: {}",
                            log_wrappers::hex_encode_upper(handle),
                            log_wrappers::hex_encode_upper(version),
                            col.col_id()
                        );
                        // Check failed.
                        return Ok(false);
                    }
                }
            }
            block.reset();
            read_rows = columnar_reader.read_block(&mut block, 1024).await.unwrap();
        }
    }
    Ok(true)
}

#[derive(Clone, Serialize, Deserialize, PartialEq, Debug, Default)]
#[serde(default)]
#[serde(rename_all = "kebab-case")]
pub struct CheckColumnarConfig {
    pub pd: pd_client::Config,
    pub security: SecurityConfig,
    pub dfs: DFSConfig,
    pub keyspace_id: u32,
    pub keyspace_id_start: u32,
    pub working_dir: PathBuf,
}

impl CheckColumnarConfig {
    pub fn from_args(args: &CheckColumnarArgs) -> Self {
        let mut config = Self::default();
        if args.config.exists() {
            let data = std::fs::read(args.config.as_path()).expect("failed to read config file");
            config = toml::from_slice(&data).unwrap();
        }
        // override from args and ENV
        if !args.pd.is_empty() {
            config.pd.endpoints = args.pd.split(',').map(|x| x.to_owned()).collect();
        }
        if !args.working_dir.display().to_string().is_empty() {
            config.working_dir = args.working_dir.clone();
        }
        if args.keyspace_id > 0 {
            config.keyspace_id = args.keyspace_id;
        }
        if args.keyspace_id_start > 0 {
            config.keyspace_id_start = args.keyspace_id_start;
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
}

fn build_ia_mgr(
    dfs: Arc<dyn Dfs>,
    runtime: &tokio::runtime::Runtime,
    data_dir: &Path,
    ia: &IaConfig,
) -> IaManager {
    // Create the data directory if it doesn't exist.
    if !data_dir.exists() {
        std::fs::create_dir_all(data_dir).unwrap();
    }
    let options = ia.to_manager_options(vec![data_dir.to_path_buf()]).unwrap();
    let handle = runtime.handle().clone();
    let fd_cache = FdCache::new(ia.fd_cache_capacity);
    IaManager::new(options, dfs, Some(fd_cache), handle.into()).unwrap()
}

async fn get_leader_store(pd_client: Arc<dyn PdClient>, region_id: u64) -> Option<u64> {
    let (_, peer) = pd_client
        .get_region_leader_by_id(region_id)
        .await
        .unwrap()?;
    Some(peer.get_store_id())
}
