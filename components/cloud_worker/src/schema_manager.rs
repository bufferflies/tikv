// Copyright 2024 TiKV Project Authors. Licensed under Apache-2.0.
use std::{
    collections::HashMap,
    fs,
    ops::Deref,
    path::{Path, PathBuf},
    sync::Arc,
    time::Duration,
};

use api_version::{ApiV2, KeyMode, KvFormat};
use async_trait::async_trait;
use bytes::{Buf, BufMut, Bytes};
use dashmap::DashMap;
use http::Request;
use hyper::Body;
use kvengine::{
    dfs,
    dfs::Dfs,
    table::{
        columnar::{
            builder::{
                new_common_handle_column_info, new_int_handle_column_info, new_txn_id_column_info,
                new_version_column_info,
            },
            columnar::Schema,
            schema_file::{self, SchemaFile},
        },
        sstable::{File, LocalFile, NO_COMPRESSION},
        ChecksumType,
    },
    ShardStatsLite,
};
use kvproto::metapb::Store;
use native_br::common::send_request_to_store_with_retry;
use schema::schema::{convert_column_infos_to_tipb, TableInfo};
use security::SecurityManager;
use tikv_client::{BoundRange, Key, TransactionOptions, Value};
use tikv_util::{box_err, config::ReadableDuration, debug, error, info};

use crate::{error::Result, get_all_stores_except_tiflash, server::Context};

const DEFAULT_TIMEOUT: ReadableDuration = ReadableDuration::secs(5);
const KEYSPACE_REFRESH_INTERVAL: ReadableDuration = ReadableDuration::secs(60);
const SCHEMA_REFRESH_THRESHOLD: u64 = 256 * 1024 * 1024;

const META_FILE_MAGIC: u32 = 0x5E9EDFF4;
const META_FILE_FORMAT_VER: u16 = 1;
const META_FILE_NAME: &str = "schemas.meta";

#[derive(Clone, Default)]
pub struct ApiV2NoPrefixCodec {}

impl tikv_client::codec::Codec for ApiV2NoPrefixCodec {
    fn encode_request<R: tikv_client::request::KvRequest>(&self, req: &mut R) {
        req.set_api_version(tikv_client::proto::kvrpcpb::ApiVersion::V2);
    }
}

type TxnClient = tikv_client::TransactionClient<ApiV2NoPrefixCodec>;

#[derive(Clone)]
struct MetaFile {
    core: Arc<MetaFileCore>,
}

struct MetaFileCore {
    files: DashMap<u32 /* keyspace_id */, Vec<(u64 /* file_id */, i64 /* schema_version */)>>,
    checked_version: DashMap<u32 /* keyspace_id */, i64 /* schema_version */>,
}

#[derive(Clone, Copy, Debug)]
struct MetaFileFooter {
    pub checksum: u32,
    pub checksum_type: u8,
    pub compression_type: u8,
    pub format_version: u16,
    pub magic: u32,
}

impl MetaFileFooter {
    fn new() -> Self {
        MetaFileFooter {
            checksum: 0,
            checksum_type: ChecksumType::Crc32.value(),
            compression_type: NO_COMPRESSION,
            format_version: META_FILE_FORMAT_VER,
            magic: META_FILE_MAGIC,
        }
    }

    fn parse(mut buf: &[u8]) -> Self {
        MetaFileFooter {
            checksum: buf.get_u32_le(),
            checksum_type: buf.get_u8(),
            compression_type: buf.get_u8(),
            format_version: buf.get_u16_le(),
            magic: buf.get_u32_le(),
        }
    }

    fn write_to(&self, data: &mut Vec<u8>) {
        data.put_u32_le(self.checksum);
        data.put_u8(self.checksum_type);
        data.put_u8(self.compression_type);
        data.put_u16_le(self.format_version);
        data.put_u32_le(self.magic);
    }
}

impl MetaFile {
    fn new() -> Self {
        let core = MetaFileCore {
            files: DashMap::default(),
            checked_version: DashMap::default(),
        };
        Self {
            core: Arc::new(core),
        }
    }

    fn open(file: LocalFile) -> Result<Self> {
        let file_data = file.read(0, file.size() as usize)?;
        let footer_size = std::mem::size_of::<MetaFileFooter>();
        if file_data.len() < footer_size {
            return Err(crate::error::Error::FileCorrupted);
        }
        let footer_offset = file_data.len() - footer_size;
        let footer = MetaFileFooter::parse(&file_data[footer_offset..]);
        let mut data = &file_data[..footer_offset];
        if footer.magic != META_FILE_MAGIC {
            return Err(crate::error::Error::FileCorrupted);
        }
        let checksum_type = ChecksumType::from(footer.checksum_type);
        let got_checksum = checksum_type.checksum(data);
        if got_checksum != footer.checksum {
            return Err(crate::error::Error::FileCorrupted);
        }
        let keyspace_count = data.get_u64_le();
        let files = DashMap::with_capacity(keyspace_count as usize);
        for _ in 0..keyspace_count {
            let keyspace_id = data.get_u32_le();
            let file_count = data.get_u64_le();
            let mut keyspace_files = Vec::with_capacity(file_count as usize);
            for _ in 0..file_count {
                let file_id = data.get_u64_le();
                let schema_version = data.get_i64_le();
                keyspace_files.push((file_id, schema_version));
            }
            files.insert(keyspace_id, keyspace_files);
        }
        let checked_count = data.get_u64_le();
        let checked_version = DashMap::with_capacity(checked_count as usize);
        for _ in 0..checked_count {
            let keyspace_id = data.get_u32_le();
            let version = data.get_i64_le();
            checked_version.insert(keyspace_id, version);
        }
        let core = MetaFileCore {
            files,
            checked_version,
        };
        Ok(MetaFile {
            core: Arc::new(core),
        })
    }

    fn write(&self) -> Vec<u8> {
        let mut data = Vec::new();
        let keyspace_count = self.core.files.len();
        data.put_u64_le(keyspace_count as u64);
        for kv in self.core.files.iter() {
            let keyspace_id = kv.key();
            let v = kv.value();
            data.put_u32_le(*keyspace_id);
            data.put_u64_le(v.len() as u64);
            for (file_id, schema_version) in v {
                data.put_u64_le(*file_id);
                data.put_i64_le(*schema_version);
            }
        }
        let checked_count = self.core.checked_version.len();
        data.put_u64_le(checked_count as u64);
        for kv in self.core.checked_version.iter() {
            let keyspace_id = kv.key();
            let ver = kv.value();
            data.put_u32_le(*keyspace_id);
            data.put_i64_le(*ver);
        }
        let mut footer = MetaFileFooter::new();
        let checksum_type = ChecksumType::Crc32;
        footer.checksum = checksum_type.checksum(&data);
        footer.write_to(&mut data);
        data
    }

    fn add_file(&self, keyspace_id: u32, file_id: u64, schema_version: i64) {
        if let Some((_, version)) = self.get_latest_file(keyspace_id) {
            assert!(version < schema_version, "schema version must be in order");
        }

        self.core
            .files
            .entry(keyspace_id)
            .and_modify(|v| v.push((file_id, schema_version)))
            .or_insert(vec![(file_id, schema_version)]);
    }

    fn add_checked_version(&self, keyspace_id: u32, version: i64) {
        self.core.checked_version.insert(keyspace_id, version);
    }

    fn get_checked_version(&self, keyspace_id: u32) -> Option<i64> {
        self.core
            .checked_version
            .get(&keyspace_id)
            .as_deref()
            .cloned()
    }

    #[allow(dead_code)]
    fn get_files(&self, keyspace_id: u32) -> Option<Vec<(u64, i64)>> {
        self.core.files.get(&keyspace_id).as_deref().cloned()
    }

    fn get_latest_file(&self, keyspace_id: u32) -> Option<(u64, i64)> {
        self.core
            .files
            .get(&keyspace_id)
            .map(|m| m.last().cloned().unwrap())
    }
}

#[derive(Clone, Serialize, Deserialize, PartialEq, Debug)]
#[serde(default)]
#[serde(rename_all = "kebab-case")]
pub struct SchemaManagerConfig {
    pub dir: PathBuf,
    pub keyspace_refresh_interval: ReadableDuration,
    pub schema_refresh_threshold: u64,
    pub http_timeout: ReadableDuration,
    pub enabled: bool,
}

impl Default for SchemaManagerConfig {
    fn default() -> Self {
        Self {
            dir: tempfile::tempdir().unwrap().into_path(),
            keyspace_refresh_interval: KEYSPACE_REFRESH_INTERVAL,
            schema_refresh_threshold: SCHEMA_REFRESH_THRESHOLD,
            http_timeout: DEFAULT_TIMEOUT,
            enabled: false,
        }
    }
}

impl SchemaManagerConfig {
    pub fn new(
        dir: PathBuf,
        keyspace_refresh_interval: ReadableDuration,
        schema_refresh_threshold: u64,
        http_timeout: ReadableDuration,
        enabled: bool,
    ) -> Self {
        Self {
            dir,
            keyspace_refresh_interval,
            schema_refresh_threshold,
            http_timeout,
            enabled,
        }
    }
}

#[derive(Clone)]
pub struct SchemaManager {
    core: Arc<SchemaManagerCore>,
}

impl Deref for SchemaManager {
    type Target = SchemaManagerCore;

    fn deref(&self) -> &Self::Target {
        &self.core
    }
}

impl SchemaManager {
    pub(crate) fn new(
        ctx: Arc<Context>,
        security_mgr: Arc<SecurityManager>,
        config: SchemaManagerConfig,
        endpoints: &[String],
    ) -> Self {
        let runtime = ctx.s3fs.get_runtime();
        let txn_client = runtime
            .block_on(tikv_client::TransactionClient::new_with_codec(
                endpoints.to_vec(),
                tikv_client::Config::default(),
                ApiV2NoPrefixCodec::default(),
            ))
            .unwrap();
        info!("SchemaManager started with config: {:?}", config);
        Self {
            core: Arc::new(SchemaManagerCore::new(
                ctx,
                security_mgr,
                config,
                txn_client,
            )),
        }
    }

    pub(crate) fn run(&self, runtime: Arc<tokio::runtime::Runtime>) {
        let self_clone = self.clone();
        runtime.spawn(async move {
            loop {
                // Get all keyspaces stats from store.
                let stores = get_all_stores_except_tiflash(&self_clone.ctx.pd).unwrap_or_default();
                if stores.is_empty() {
                    tokio::time::sleep(self_clone.config.keyspace_refresh_interval.0).await;
                    continue;
                }

                let mut keyspace_stats = HashMap::new();
                if let Err(e) = self_clone
                    .refresh_keyspace_stats(&mut keyspace_stats, &stores)
                    .await
                {
                    error!("refresh keyspace stats error: {:?}", e);
                    tokio::time::sleep(self_clone.config.keyspace_refresh_interval.0).await;
                    continue;
                }
                // Refresh keyspaces schema version.
                if let Err(e) = self_clone
                    .refresh_keyspace_schema(&keyspace_stats, &stores)
                    .await
                {
                    error!("refresh schema version error: {:?}", e);
                }
                tokio::time::sleep(self_clone.config.keyspace_refresh_interval.0).await;
            }
        });
    }

    pub(crate) async fn refresh_keyspace_stats(
        &self,
        keyspace_stats: &mut HashMap<u32, Vec<ShardStatsLite>>,
        stores: &[Store],
    ) -> Result<()> {
        let mut all_shard_stats = vec![];
        for store in stores.iter() {
            let shard_stats = get_keyspace_stats_from_store(
                store,
                self.security_mgr.clone(),
                self.config.http_timeout.0,
            )
            .await?;
            all_shard_stats.extend(shard_stats);
        }

        for shard_stats in all_shard_stats {
            let start_key = shard_stats.start.to_vec();
            if ApiV2::parse_key_mode(&start_key) != KeyMode::Txn {
                continue;
            }
            let keyspace_id = ApiV2::get_u32_keyspace_id_by_key(&start_key);
            if keyspace_id.is_none() {
                continue;
            }
            let keyspace_id = keyspace_id.unwrap();
            keyspace_stats
                .entry(keyspace_id)
                .or_default()
                .push(shard_stats);
        }
        Ok(())
    }

    pub(crate) async fn refresh_keyspace_schema(
        &self,
        keyspace_stats: &HashMap<u32, Vec<ShardStatsLite>>,
        stores: &[Store],
    ) -> Result<()> {
        let (tx, rx) = tikv_util::mpsc::unbounded();
        let runtime = self.ctx.s3fs.get_runtime();
        let mut spawn_task_count = 0;
        for (&keyspace_id, keyspace_shard_stats) in keyspace_stats.iter().filter(|(_, v)| {
            v.iter().map(|s| s.total_size).sum::<u64>() > self.config.schema_refresh_threshold
        }) {
            // 1. Try to read schema file from local.
            let local_schema_file =
                read_schema_file_from_local(&self.config.dir, &self.meta_file, keyspace_id);
            let cur_schema_version = local_schema_file
                .as_ref()
                .map(|f| {
                    f.as_ref()
                        .map(|s| Some(s.get_version()))
                        .unwrap_or_default()
                })
                .unwrap_or_default();

            // Check the remote schema_version in store shard_stats to ensure the state
            // applied to kvengine.
            if let Ok(Some(schema_file)) = &local_schema_file {
                if self
                    .check_store_schema_version(
                        keyspace_id,
                        cur_schema_version.unwrap(),
                        schema_file,
                        keyspace_shard_stats,
                        stores,
                    )
                    .await
                {
                    continue;
                }
            }

            // 2. Check the latest schema version compare with cache, if any
            // update, fetch all the new schemas and update to S3.
            let kv_scanner = Arc::new(self.clone());
            let kv_getter = Arc::new(self.clone());
            let (schema_version, table_infos) =
                schema::sync_schema(kv_getter, kv_scanner, keyspace_id, cur_schema_version)
                    .await
                    .map_err(|e| crate::error::Error::SchemaError(e))?;
            let cur_schema_version = cur_schema_version.unwrap_or(0);
            if let Some(checked_schema_version) = self.meta_file.get_checked_version(keyspace_id) {
                if cur_schema_version < checked_schema_version
                    && checked_schema_version == schema_version
                {
                    debug!(
                        "skip checked schema_version, keyspace_id: {} cur_schema_version: {} checked_schema_version: {}",
                        keyspace_id, cur_schema_version, checked_schema_version
                    );
                    continue;
                }
            }
            if cur_schema_version < schema_version {
                if table_infos.is_empty() {
                    continue;
                }
                info!(
                    "schema need to be rebuilt, keyspace: {} cur_schema_version: {} schema_version: {}",
                    keyspace_id, cur_schema_version, schema_version
                );
                // 3. Build the schema file and upload to S3.
                let schemas = self.build_new_schema(
                    keyspace_id,
                    &local_schema_file,
                    schema_version,
                    table_infos,
                );
                if schemas.is_none() {
                    continue;
                }

                let new_schema_file_data =
                    schema_file::build_schema_file(keyspace_id, schema_version, schemas.unwrap());
                let file_id = self.ctx.pd.alloc_id()?;
                let dfs = self.ctx.s3fs.clone();
                let tx_clone = tx.clone();
                spawn_task_count += 1;
                runtime.spawn(async move {
                    let data = Bytes::from(new_schema_file_data.clone());
                    let opts = dfs::Options::default().with_type(dfs::FileType::Schema);
                    if let Err(err) = dfs.create(file_id, data.clone(), opts).await {
                        tx_clone
                            .send((
                                keyspace_id,
                                file_id,
                                schema_version,
                                data,
                                Err(crate::error::Error::DfsError(err)),
                            ))
                            .unwrap();
                    } else {
                        tx_clone
                            .send((keyspace_id, file_id, schema_version, data, Ok(file_id)))
                            .unwrap();
                    }
                });
            } else {
                debug!("schema has no changes, keyspace_id: {}", keyspace_id);
            }
        }

        for _ in 0..spawn_task_count {
            let (keyspace_id, file_id, schema_version, data, res) = rx.recv().unwrap();
            if let Err(err) = res {
                error!(
                    "failed to update schema file, keyspace_id: {} file_id: {} err: {:?}",
                    keyspace_id, file_id, err
                );
                continue;
            }
            // 4. Callback TiKV to update the new schema file to shard meta.
            info!(
                "broadcast schema update to stores, keyspace_id: {} file_id: {} schema_version: {}",
                keyspace_id, file_id, schema_version
            );
            if let Err(err) = broadcast_schema_update_to_all_stores(
                stores,
                self.security_mgr.clone(),
                self.config.http_timeout.0,
                keyspace_id,
                file_id,
            )
            .await
            {
                error!(
                    "failed to broadcast schema update, keyspace_id: {} file_id: {} err: {:?}",
                    keyspace_id, file_id, err
                );
                continue;
            }
            if let Err(err) =
                write_schema_file_to_local(&self.config.dir, keyspace_id, file_id, data)
            {
                error!(
                    "failed to write schema file to local, keyspace_id: {} file_id: {} err: {:?}",
                    keyspace_id, file_id, err
                );
                continue;
            }
            self.meta_file
                .add_file(keyspace_id, file_id, schema_version);
        }
        // Save meta_file.
        let meta = self.meta_file.write();
        write_meta_file_to_local(&self.config.dir, Bytes::from(meta)).unwrap();
        Ok(())
    }

    // return true if sent broadcast to stores
    async fn check_store_schema_version(
        &self,
        keyspace_id: u32,
        cur_schema_version: i64,
        schema_file: &SchemaFile,
        keyspace_shard_stats: &[ShardStatsLite],
        stores: &[Store],
    ) -> bool {
        let mut need_broadcast = false;
        for shard_stats in keyspace_shard_stats {
            let start_key = shard_stats.start.to_vec();
            let end_key = shard_stats.end.to_vec();
            if schema_file.overlap(&start_key, &end_key, keyspace_id)
                && cur_schema_version > shard_stats.schema_version
            {
                info!(
                    "store has stale schema version, keyspace_id: {} set cur_schema_version from {} to {}",
                    keyspace_id, cur_schema_version, shard_stats.schema_version,
                );
                need_broadcast = true;
            }
        }
        // Broadcast schema update to stores without building schema again.
        if need_broadcast {
            let file_id = schema_file.get_file_id();
            if let Err(err) = broadcast_schema_update_to_all_stores(
                stores,
                self.security_mgr.clone(),
                self.config.http_timeout.0,
                keyspace_id,
                file_id,
            )
            .await
            {
                error!(
                    "failed to broadcast schema update, keyspace_id: {} file_id: {} err: {:?}",
                    keyspace_id, file_id, err
                );
            }
            info!(
                "broadcast schema update to all stores, keyspace_id: {} file_id: {}",
                keyspace_id, file_id
            );
            return true;
        }
        false
    }

    fn build_new_schema(
        &self,
        keyspace_id: u32,
        local_schema_file: &Result<Option<SchemaFile>>,
        schema_version: i64,
        table_infos: Vec<TableInfo>,
    ) -> Option<Vec<Schema>> {
        // 3. Build the schema file and upload to S3.
        let mut schemas = if let Ok(Some(schema_file)) = local_schema_file {
            Vec::with_capacity(table_infos.len() + schema_file.schema_count())
        } else {
            Vec::with_capacity(table_infos.len())
        };
        let mut to_be_removed = vec![];
        for ti in table_infos {
            if ti.tiflash_replica.map(|t| t.count == 0).unwrap_or(true) {
                to_be_removed.push(ti.id);
                continue;
            }
            if ti.cols.as_ref().map(|c| c.is_empty()).unwrap_or(true) {
                to_be_removed.push(ti.id);
                continue;
            }

            let columns = convert_column_infos_to_tipb(ti.cols.as_ref().unwrap(), ti.pk_is_handle);
            let handle_column = if ti.is_common_handle {
                new_common_handle_column_info()
            } else {
                new_int_handle_column_info()
            };
            let schema = Schema {
                table_id: ti.id,
                handle_column,
                version_column: new_version_column_info(),
                txn_id_column: Some(new_txn_id_column_info()),
                columns,
            };
            schemas.push(schema);
        }

        if let Ok(Some(schema_file)) = &local_schema_file {
            // Check if the schemas contains in schema file to avoid useless update.
            if schema_file.contains(&schemas) && !schema_file.has_overlap_ids(&to_be_removed) {
                // The schema change is not related to columnar, skip.
                // Record the schema version to skip in later loop.
                self.meta_file
                    .add_checked_version(keyspace_id, schema_version);
                return None;
            }

            // Merge schemas in file to build the new one.
            let base = schema_file.export_schemas();
            schemas = merge_schema_diffs(base, schemas, &to_be_removed);
        }
        Some(schemas)
    }
}

pub struct SchemaManagerCore {
    ctx: Arc<Context>,
    security_mgr: Arc<SecurityManager>,
    config: SchemaManagerConfig,
    txn_client: TxnClient,
    meta_file: MetaFile,
}

impl SchemaManagerCore {
    pub(crate) fn new(
        ctx: Arc<Context>,
        security_mgr: Arc<SecurityManager>,
        config: SchemaManagerConfig,
        txn_client: TxnClient,
    ) -> Self {
        let meta_file_path = config.dir.join(META_FILE_NAME);
        let meta_file = if meta_file_path.exists() {
            MetaFile::open(LocalFile::open(0, meta_file_path.as_path(), false).unwrap()).unwrap()
        } else {
            MetaFile::new()
        };
        Self {
            ctx,
            security_mgr,
            config,
            txn_client,
            meta_file,
        }
    }
}

#[async_trait]
impl schema::KvScanner for SchemaManager {
    async fn scan(
        &self,
        start: &[u8],
        end: &[u8],
    ) -> std::result::Result<Vec<(Vec<u8>, Vec<u8>)>, String> {
        let start_ts = self
            .txn_client
            .current_timestamp()
            .await
            .map_err(|e| e.to_string())?;
        let mut snapshot = self
            .txn_client
            .snapshot(start_ts, TransactionOptions::new_pessimistic());
        let scan_range: BoundRange = (start.to_vec()..end.to_vec()).into();
        let kv_pairs = snapshot
            .scan(scan_range, u32::MAX)
            .await
            .map_err(|e| e.to_string())?;
        let mut pairs = vec![];
        for kv_pair in kv_pairs {
            let key = kv_pair.key().clone();
            let val = kv_pair.into_value();
            pairs.push((key.into(), val));
        }
        Ok(pairs)
    }
}

#[async_trait]
impl schema::KvGetter for SchemaManager {
    async fn get(&self, key: &[u8]) -> std::result::Result<Option<Vec<u8>>, String> {
        let start_ts = self
            .txn_client
            .current_timestamp()
            .await
            .map_err(|e| e.to_string())?;
        let mut snapshot = self
            .txn_client
            .snapshot(start_ts.clone(), TransactionOptions::new_pessimistic());
        let val = snapshot
            .get(key.to_vec())
            .await
            .map_err(|e| e.to_string())?;
        Ok(val)
    }

    async fn batch_get(
        &self,
        keys: &[Vec<u8>],
    ) -> std::result::Result<Vec<Option<Vec<u8>>>, String> {
        let start_ts = self
            .txn_client
            .current_timestamp()
            .await
            .map_err(|e| e.to_string())?;
        let mut snapshot = self
            .txn_client
            .snapshot(start_ts, TransactionOptions::new_pessimistic());
        let pairs: HashMap<Key, Value> = snapshot
            .batch_get(keys.to_vec())
            .await
            .map_err(|e| e.to_string())?
            .map(|pair| (pair.0, pair.1))
            .collect();
        let mut vals = Vec::with_capacity(keys.len());
        for key in keys {
            if let Some(val) = pairs.get(&Key::from(key.to_vec())) {
                vals.push(Some(val.clone()));
            } else {
                vals.push(None);
            }
        }
        Ok(vals)
    }
}

fn find_latest_schema_file<P: AsRef<Path>>(dir_path: P) -> Result<Option<String>> {
    let mut latest_file: Option<String> = None;
    info!(
        "searching latest schema file in {}",
        dir_path.as_ref().display()
    );
    let entries = fs::read_dir(&dir_path)?;
    for entry in entries {
        let entry = entry?;
        if let Ok(file_name) = entry.file_name().into_string() {
            if latest_file.is_none()
                || (file_name.ends_with(".schema") && &file_name > latest_file.as_ref().unwrap())
            {
                latest_file = Some(file_name);
            }
        }
    }
    Ok(latest_file)
}

fn merge_schema_diffs(
    mut base: HashMap<i64, Schema>,
    added: Vec<Schema>,
    removed_ids: &[i64],
) -> Vec<Schema> {
    for schema in added {
        base.insert(schema.table_id, schema);
    }
    for id in removed_ids {
        base.remove(id);
    }
    base.values().cloned().collect::<Vec<_>>()
}

fn read_schema_file_from_local<P: AsRef<Path>>(
    base_dir: P,
    meta_file: &MetaFile,
    keyspace_id: u32,
) -> Result<Option<SchemaFile>> {
    let dir = base_dir.as_ref().join(keyspace_id.to_string());
    // Try get file_id from meta_file.
    let file_id = if let Some((file_id, _)) = meta_file.get_latest_file(keyspace_id) {
        file_id
    } else {
        let latest_schema_filename = find_latest_schema_file(&dir)?;
        // If no schema file found, return None.
        if latest_schema_filename.is_none() {
            info!("no schema file found, keyspace_id: {}", keyspace_id);
            return Ok(None);
        }
        let latest_schema_filename = latest_schema_filename.unwrap();
        u64::from_str_radix(latest_schema_filename.strip_suffix(".schema").unwrap(), 16)?
    };

    let file_path = dir.join(format!("{:016x}.schema", file_id));
    let local_file = Arc::new(LocalFile::open(file_id, file_path.as_path(), false)?);
    let file = SchemaFile::open(local_file)?;
    Ok(Some(file))
}

fn write_schema_file_to_local<P: AsRef<Path>>(
    base_dir: P,
    keyspace_id: u32,
    id: u64,
    data: Bytes,
) -> Result<()> {
    let dir = base_dir.as_ref().join(keyspace_id.to_string());
    let filename = format!("{:016x}.schema", id);
    let tmp_file = format!("{:016x}.schema.tmp", id);
    let file_path = dir.join(filename);
    let tmp_file_path = dir.join(tmp_file);
    fs::create_dir_all(dir)?;
    fs::write(tmp_file_path.as_path(), data)?;
    fs::rename(tmp_file_path.as_path(), file_path)?;
    Ok(())
}

fn write_meta_file_to_local<P: AsRef<Path>>(dir: P, data: Bytes) -> Result<()> {
    let tmp_file = format!("{}.tmp", META_FILE_NAME);
    let file_path = dir.as_ref().join(META_FILE_NAME);
    let tmp_file_path = dir.as_ref().join(tmp_file);
    fs::create_dir_all(dir)?;
    fs::write(tmp_file_path.as_path(), data)?;
    fs::rename(tmp_file_path.as_path(), file_path)?;
    Ok(())
}

async fn broadcast_schema_update_to_all_stores(
    stores: &[Store],
    security_mgr: Arc<SecurityManager>,
    timeout: Duration,
    keyspace_id: u32,
    file_id: u64,
) -> Result<()> {
    for store in stores {
        let security_mgr = security_mgr.clone();
        let status_addr = store.get_status_address();
        let uri = security_mgr
            .build_uri(format!(
                "{}/schema_file?keyspace_id={}&file_id={}",
                status_addr, keyspace_id, file_id
            ))
            .unwrap();
        let req = || Request::post(uri.clone()).body(Body::empty()).unwrap();
        if let Err(err) = send_request_to_store_with_retry(req, store, security_mgr, timeout).await
        {
            return Err(box_err!(
                "broadcase schema update to store {} failed: {:?}",
                status_addr,
                err
            ));
        }
    }
    Ok(())
}

async fn get_keyspace_stats_from_store(
    store: &Store,
    security_mgr: Arc<SecurityManager>,
    timeout: Duration,
) -> Result<Vec<ShardStatsLite>> {
    let status_addr = store.get_status_address();
    let uri = security_mgr
        .build_uri(format!("{}/kvengine/active_lite", status_addr))
        .unwrap();
    let req = || Request::get(uri.clone()).body(Body::empty()).unwrap();
    let resp_bytes = send_request_to_store_with_retry(req, store, security_mgr, timeout).await?;
    let resp: Vec<ShardStatsLite> = serde_json::from_slice(&resp_bytes)?;
    Ok(resp)
}

#[cfg(test)]
mod tests {
    use std::fs;

    use bytes::Bytes;
    use kvengine::table::{
        columnar::{
            builder::{new_int_handle_column_info, new_version_column_info},
            columnar::Schema,
            schema_file,
        },
        sstable::LocalFile,
    };
    use tikv_util::info;

    use super::{
        read_schema_file_from_local, write_meta_file_to_local, write_schema_file_to_local,
        META_FILE_NAME,
    };
    use crate::schema_manager::{find_latest_schema_file, MetaFile};

    #[test]
    fn test_find_latest_schema_file() {
        ::test_util::init_log_for_test();

        let dir = tempfile::tempdir().unwrap();
        let _ = fs::create_dir_all(dir.path());
        for i in 50..=100 {
            let filename = format!("{:016x}.schema", i);
            fs::write(dir.path().join(filename), "test").unwrap();
        }
        for i in 0..50 {
            let filename = format!("{:016x}.schema", i);
            fs::write(dir.path().join(filename), "test").unwrap();
        }
        let latest_file = find_latest_schema_file(dir.path()).unwrap();
        assert_eq!(latest_file, Some(format!("{:016x}.schema", 100)));
    }

    #[test]
    fn test_local_schema_file() {
        ::test_util::init_log_for_test();

        let dir = tempfile::tempdir().unwrap();
        let mut schemas = vec![];
        for i in 0..=10 {
            let schema = Schema {
                table_id: i,
                handle_column: new_int_handle_column_info(),
                version_column: new_version_column_info(),
                txn_id_column: None,
                columns: vec![new_int_handle_column_info()],
            };
            schemas.push(schema);
        }
        let schema_file_data = schema_file::build_schema_file(1234, 100, schemas.clone());
        write_schema_file_to_local(dir.path(), 1234, 1000, Bytes::from(schema_file_data)).unwrap();
        schemas.push(Schema {
            table_id: 11,
            handle_column: new_int_handle_column_info(),
            version_column: new_version_column_info(),
            txn_id_column: None,
            columns: vec![new_int_handle_column_info()],
        });
        let schema_file_data = schema_file::build_schema_file(1234, 201, schemas);
        write_schema_file_to_local(dir.path(), 1234, 1001, Bytes::from(schema_file_data)).unwrap();

        // schema_file is the newest schema file of the keyspace.
        let schema_file = read_schema_file_from_local(dir.path(), &MetaFile::new(), 1234)
            .unwrap()
            .unwrap();
        assert_eq!(schema_file.get_keyspace_id(), 1234);
        assert_eq!(schema_file.get_version(), 201);
        info!(
            "schema file keyspace_id: {}, schema_version: {}, file_id: {}",
            schema_file.get_keyspace_id(),
            schema_file.get_version(),
            schema_file.get_file_id()
        );
        assert_eq!(schema_file.get_file_id(), 1001);
    }

    #[test]
    fn test_meta_file() {
        let dir = tempfile::tempdir().unwrap();
        let meta = MetaFile::new();
        for i in 1..100 {
            meta.add_file(i, (i * 10) as u64, (i + i * 10) as i64);
            meta.add_file(i, (i * 10 + 1) as u64, (i + i * 10 + 1) as i64);
        }
        for i in 1..10 {
            meta.add_checked_version(i, (i + i * 11) as i64);
        }
        let data = meta.write();
        write_meta_file_to_local(&dir, Bytes::from(data)).unwrap();
        let meta_file_path = dir.as_ref().join(META_FILE_NAME);
        let meta_file = LocalFile::open(0, &meta_file_path, false).unwrap();
        let read_meta = MetaFile::open(meta_file).unwrap();
        for i in 1..100 {
            let (file_id, schema_version) = read_meta.get_latest_file(i).unwrap();
            assert_eq!(file_id, (i * 10 + 1) as u64);
            assert_eq!(schema_version, (i + i * 10 + 1) as i64);
        }
        for i in 1..10 {
            let checked_version = read_meta.get_checked_version(i).unwrap();
            assert_eq!(checked_version, (i + i * 11) as i64);
        }
    }
}
