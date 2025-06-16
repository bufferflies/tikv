// Copyright 2024 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    collections::{BTreeMap, HashMap, HashSet},
    fs,
    ops::Deref,
    path::{Path, PathBuf},
    sync::Arc,
    time::Duration,
};

use api_version::{
    api_v2::{is_whole_keyspace_range, DEFAULT_KEYSPACE_ID, KEYSPACE_PREFIX_LEN},
    ApiV2,
};
use async_trait::async_trait;
use bytes::{Buf, BufMut, Bytes};
use dashmap::DashMap;
use http::Request;
use hyper::Body;
use kvengine::{
    dfs,
    dfs::Dfs,
    table::{
        columnar,
        columnar::{
            new_common_handle_column_info, new_int_handle_column_info, new_version_column_info,
            Schema, SchemaBufBuilder, SchemaFile, VectorIndexDef,
        },
        file::{File, LocalFile},
        ChecksumType, NO_COMPRESSION,
    },
    IdAllocator, ShardStatsLite,
};
use kvproto::metapb::Store;
use native_br::common::send_request_to_store_with_retry;
use rfstore::store::PdIdAllocator;
use schema::schema::{
    convert_column_infos_to_tipb, ColumnInfo, IndexInfo, StorageClassSpec, TableInfo,
    VectorIndexInfo,
};
use security::{SecurityConfig, SecurityManager};
use tidb_query_datatype::VECTOR_INDEX_SPEC_KEY_DISTANCE_METRIC;
use tikv_client::{BoundRange, Key, KvPair, TransactionOptions, Value};
use tikv_util::{box_err, config::ReadableDuration, debug, error, info, warn};

use crate::{
    error::{Error, Error::SchemaError, Result},
    get_all_stores_except_tiflash,
    server::Context,
};

const DEFAULT_TIMEOUT: ReadableDuration = ReadableDuration::secs(5);
const DEFAULT_GRPC_MAX_DECODING_MESSAGE_SIZE: usize = 32 * 1024 * 1024; // 32MB
const KEYSPACE_REFRESH_INTERVAL: ReadableDuration = ReadableDuration::secs(30);

const META_FILE_MAGIC: u32 = 0x5E9EDFF4;
const META_FILE_FORMAT_VER: u16 = 1;
const META_FILE_FORMAT_VER_V2: u16 = 2;
const META_FILE_NAME: &str = "schemas.meta";

const TIKV_STORE_LABEL_TIER_KEY: &str = "serverless.tidbcloud.com/tier";

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
    checked_versions: DashMap<u32 /* keyspace_id */, i64 /* schema_version */>,
    write_sequences: DashMap<u32 /* keyspace_id */, u64 /* write_sequence */>,
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
            format_version: META_FILE_FORMAT_VER_V2,
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

    fn is_valid_version(&self) -> bool {
        matches!(
            self.format_version,
            META_FILE_FORMAT_VER_V2 | META_FILE_FORMAT_VER
        )
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
            checked_versions: DashMap::default(),
            write_sequences: DashMap::default(),
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
        if !footer.is_valid_version() {
            return Err(crate::error::Error::CheckError(
                "invalid meta file version".to_string(),
            ));
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
        let checked_versions = DashMap::with_capacity(checked_count as usize);
        for _ in 0..checked_count {
            let keyspace_id = data.get_u32_le();
            let version = data.get_i64_le();
            checked_versions.insert(keyspace_id, version);
        }
        let write_sequences = if footer.format_version == META_FILE_FORMAT_VER_V2 {
            let seq_count = data.get_u64_le();
            let write_sequences = DashMap::with_capacity(seq_count as usize);
            for _ in 0..seq_count {
                let keyspace_id = data.get_u32_le();
                let seq = data.get_u64_le();
                write_sequences.insert(keyspace_id, seq);
            }
            write_sequences
        } else {
            DashMap::default()
        };
        let core = MetaFileCore {
            files,
            checked_versions,
            write_sequences,
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
        let checked_count = self.core.checked_versions.len();
        data.put_u64_le(checked_count as u64);
        for kv in self.core.checked_versions.iter() {
            let keyspace_id = kv.key();
            let ver = kv.value();
            data.put_u32_le(*keyspace_id);
            data.put_i64_le(*ver);
        }
        let write_sequence_count = self.core.write_sequences.len();
        data.put_u64_le(write_sequence_count as u64);
        for kv in self.core.write_sequences.iter() {
            let keyspace_id = kv.key();
            let seq = kv.value();
            data.put_u32_le(*keyspace_id);
            data.put_u64_le(*seq);
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
        self.core.checked_versions.insert(keyspace_id, version);
    }

    fn get_checked_version(&self, keyspace_id: u32) -> Option<i64> {
        self.core
            .checked_versions
            .get(&keyspace_id)
            .as_deref()
            .cloned()
    }

    fn add_write_sequence(&self, keyspace_id: u32, seq: u64) {
        self.core.write_sequences.insert(keyspace_id, seq);
    }

    fn get_write_sequence(&self, keyspace_id: u32) -> Option<u64> {
        self.core
            .write_sequences
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

    fn remove_keyspace(&self, keyspace_id: u32) -> Option<(u32, Vec<(u64, i64)>)> {
        // The schema file will be removed by gc.
        self.core.checked_versions.remove(&keyspace_id);
        self.core.write_sequences.remove(&keyspace_id);
        self.core.files.remove(&keyspace_id)
    }
}

#[derive(Clone, Serialize, Deserialize, PartialEq, Debug)]
#[serde(default)]
#[serde(rename_all = "kebab-case")]
pub struct SchemaManagerConfig {
    pub dir: PathBuf,
    pub keyspace_refresh_interval: ReadableDuration,
    pub http_timeout: ReadableDuration,
    pub enabled: bool,
    // `whitelist_file` is a json file contains a list of keyspace_id.
    pub whitelist_file: PathBuf,
    // The tier of TiKV stores to push schema file. Used for canary release.
    pub tikv_stores_tier: String,
}

impl Default for SchemaManagerConfig {
    fn default() -> Self {
        Self {
            dir: tempfile::tempdir().unwrap().into_path(),
            keyspace_refresh_interval: KEYSPACE_REFRESH_INTERVAL,
            http_timeout: DEFAULT_TIMEOUT,
            enabled: false,
            whitelist_file: PathBuf::new(), // Empty means no whitelist filtering.
            tikv_stores_tier: "".to_string(), // Empty means match all stores.
        }
    }
}

impl SchemaManagerConfig {
    pub fn new(
        dir: PathBuf,
        keyspace_refresh_interval: ReadableDuration,
        http_timeout: ReadableDuration,
        enabled: bool,
        whitelist_file: PathBuf,
        tikv_stores_tier: String,
    ) -> Self {
        Self {
            dir,
            keyspace_refresh_interval,
            http_timeout,
            enabled,
            whitelist_file,
            tikv_stores_tier,
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
        security_config: SecurityConfig,
        config: SchemaManagerConfig,
        endpoints: &[String],
    ) -> Self {
        let runtime = ctx.s3fs.get_runtime();
        let mut client_config = tikv_client::Config::default()
            .with_grpc_max_decoding_message_size(DEFAULT_GRPC_MAX_DECODING_MESSAGE_SIZE);
        if !security_config.ca_path.is_empty()
            || !security_config.cert_path.is_empty()
            || !security_config.key_path.is_empty()
        {
            let SecurityConfig {
                ca_path,
                cert_path,
                key_path,
                ..
            } = security_config;
            client_config = client_config.with_security(ca_path, cert_path, key_path);
        }
        let txn_client = runtime
            .block_on(tikv_client::TransactionClient::new_with_codec(
                endpoints.to_vec(),
                client_config,
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
                let (stores, _) = self_clone.get_tikv_stores();
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
                debug!("schema manager: keyspace stats: {:?}", keyspace_stats);
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
            Self::update_keyspace_stats(keyspace_stats, shard_stats)
        }
        Ok(())
    }

    fn update_keyspace_stats(
        keyspace_stats: &mut HashMap<u32, Vec<ShardStatsLite>>,
        shard_stats: ShardStatsLite,
    ) {
        let Some(keyspace_id) = ApiV2::get_u32_keyspace_id_by_key(&shard_stats.start) else {
            return;
        };
        let Some(end_keyspace_id) = ApiV2::get_u32_keyspace_id_by_key(&shard_stats.end) else {
            return;
        };
        if keyspace_id != end_keyspace_id
            && !is_whole_keyspace_range(&shard_stats.start[..KEYSPACE_PREFIX_LEN], &shard_stats.end)
        {
            return;
        }
        // Skip the empty keyspace or tombstone keyspace.
        if is_whole_keyspace_range(&shard_stats.start, &shard_stats.end)
            && shard_stats.total_size == 0
        {
            return;
        }
        keyspace_stats
            .entry(keyspace_id)
            .or_default()
            .push(shard_stats);
    }

    pub(crate) async fn refresh_keyspace_schema(
        &self,
        keyspace_stats: &HashMap<u32, Vec<ShardStatsLite>>,
        stores: &[Store],
    ) -> Result<()> {
        let (tx, rx) = tikv_util::mpsc::unbounded();
        let runtime = self.ctx.s3fs.get_runtime();
        let mut spawn_task_count = 0;
        for (&keyspace_id, keyspace_shard_stats) in keyspace_stats {
            // Skip the default keyspace. The tikv-client not support the default keyspace
            // with ApiV2NoPrefixCodec.
            if !self.in_whitelist(keyspace_id) || keyspace_id == DEFAULT_KEYSPACE_ID {
                continue;
            }
            // If the keyspace has only one shard, we check the write sequence if changed.
            // If the write sequence is not changed, we skip the refresh. This is useful to
            // avoid large number of infrequent small keyspaces to check schema metadata
            // from TiKV.
            let update_write_sequence = if keyspace_shard_stats.len() == 1 {
                let seq = self.meta_file.get_write_sequence(keyspace_id);
                let shard_stats = keyspace_shard_stats.first().unwrap();
                if seq.is_some_and(|seq| seq == shard_stats.write_sequence) {
                    debug!(
                        "{}: write sequence is not changed, skip",
                        keyspace_id;
                        "seq" => ?seq, "shard_stats" => ?shard_stats
                    );
                    continue;
                }
                Some(shard_stats.write_sequence)
            } else {
                None
            };
            debug!(
                "{}: update write sequence: {}",
                keyspace_id,
                update_write_sequence.is_some()
            );

            let keyspace_total_size = keyspace_shard_stats
                .iter()
                .map(|s| s.total_size)
                .sum::<u64>();
            if keyspace_total_size == 0 {
                debug!("{}: keyspace total size is 0, ignore keyspace", keyspace_id);
                continue;
            }

            if self.check_if_keyspace_restore_in_progress(keyspace_shard_stats) {
                info!("{}: keyspace restore in progress, skip", keyspace_id);
                continue;
            }

            // 1. Try to read schema file from local.
            let local_schema_file =
                match read_schema_file_from_local(&self.config.dir, &self.meta_file, keyspace_id) {
                    Ok(Some(local_schema_file)) => Some(local_schema_file),
                    Ok(None) => None,
                    Err(err) => {
                        warn!("read schema file from local failed: {:?}", err);
                        None
                    }
                };

            // Check the remote schema_version in store shard_stats to ensure the state
            // applied to kvengine.
            let cur_schema_version = if let Some(schema_file) = &local_schema_file {
                let cur_schema_version = schema_file.get_version();
                let cur_restore_version = schema_file.get_restore_version();
                debug!(
                    "{}: schema_file_id: {}, schema_version: {}, restore_version: {}",
                    keyspace_id,
                    schema_file.get_file_id(),
                    schema_file.get_version(),
                    cur_restore_version,
                );
                if self.check_if_keyspace_restored(
                    keyspace_id,
                    keyspace_shard_stats,
                    cur_schema_version,
                    cur_restore_version,
                ) {
                    // If the keyspace is just restored, the schema_restore_version in shard will be
                    // reset to backup_ts after restoration. In this case, we should remove the old
                    // schema file and try to rebuild it in next loop.
                    if let Some((_, files)) = self.meta_file.remove_keyspace(keyspace_id) {
                        let file_ids: Vec<u64> = files.iter().map(|f| f.0).collect();
                        remove_schema_file_from_local(&self.config.dir, keyspace_id, &file_ids)
                            .map_err(|err| -> Error {
                                box_err!(
                                    "{}: remove_schema_file_from_local failed {:?}",
                                    keyspace_id,
                                    err
                                )
                            })?;
                    }
                    info!(
                        "{}: keyspace is restored, remove local schema files",
                        keyspace_id
                    );
                    continue;
                }
                if self
                    .check_store_schema_version(
                        keyspace_id,
                        cur_schema_version,
                        schema_file,
                        keyspace_shard_stats,
                        stores,
                    )
                    .await
                {
                    continue;
                }
                Some(cur_schema_version)
            } else {
                None
            };

            // 2. Check the latest schema version compare with cache, if any
            // update, fetch all the new schemas and update to S3.
            let checked_version = self.meta_file.get_checked_version(keyspace_id);
            if let (Some(checked), Some(cur)) = (checked_version, cur_schema_version) {
                debug_assert!(checked >= cur, "{} {}", checked, cur);
            }
            let checked_version = checked_version.or(cur_schema_version);

            let kv_scanner = Arc::new(self.clone());
            let kv_getter = Arc::new(self.clone());
            let (schema_version, table_infos) = match schema::sync_schema(
                kv_getter,
                kv_scanner,
                keyspace_id,
                checked_version,
            )
            .await
            {
                Ok(result) => result,
                Err(err) => {
                    // TODO: report metrics and alarm.
                    error!("{}: sync schema failed, skip", keyspace_id; "err" => ?err);
                    continue;
                }
            };

            if checked_version.is_some_and(|v| v == schema_version) {
                debug!("{}: schema is up-to-date, skip", keyspace_id; "schema_ver" => schema_version,
                    "cur_ver" => ?cur_schema_version, "checked_ver" => ?checked_version);
                if self.meta_file.get_checked_version(keyspace_id).is_none() {
                    self.meta_file
                        .add_checked_version(keyspace_id, schema_version);
                }
                if let Some(write_sequence) = update_write_sequence {
                    self.meta_file
                        .add_write_sequence(keyspace_id, write_sequence);
                }
                continue;
            }
            info!(
                "{}: sync schema: schema_version: {}, checked_version: {:?}, table_infos: {}",
                keyspace_id,
                schema_version,
                checked_version,
                table_infos.len()
            );
            debug!("{}: sync schema", keyspace_id; "schema_ver" => schema_version, "tables" => ?table_infos);

            let old_storage_class_tables = local_schema_file
                .as_ref()
                .map(|schema_file| schema_file.tables_with_storage_class());
            // If the old specified storage class becomes unspecified, the storage class is
            // removed from the schema file.
            let sc_need_update_schema = table_infos.iter().any(|ti| {
                ti.with_storage_class_spec()
                    || old_storage_class_tables
                        .as_ref()
                        .is_some_and(|tables| tables.contains(&ti.id))
            });
            let columnar_need_update_schema = table_infos.iter().any(|ti| {
                // local schema file not exist, need to update.
                if local_schema_file.is_none() {
                    return true;
                }
                // If ti has columnar or schema has columnar, need to update.
                let local_schema = local_schema_file.as_ref().unwrap().get_table(ti.id);
                ti.with_columnar() || local_schema.map(|s| s.with_columnar()).unwrap_or_default()
            });

            if local_schema_file.is_some() && !sc_need_update_schema && !columnar_need_update_schema
            {
                debug!("{}: schema has no required changes, skip", keyspace_id;
                    "schema_version" => schema_version, "tables" => ?table_infos, "old_sc_tables" => ?old_storage_class_tables);
                self.meta_file
                    .add_checked_version(keyspace_id, schema_version);
                if let Some(write_sequence) = update_write_sequence {
                    self.meta_file
                        .add_write_sequence(keyspace_id, write_sequence);
                }
                continue;
            }

            let schema_restore_version = keyspace_shard_stats
                .first()
                .map(|s| s.schema_restore_version)
                .unwrap_or(0);
            info!("{}: sync schema: rebuild schema", keyspace_id;
                "cur_schema_ver" => ?cur_schema_version,
                "schema_ver" => schema_version,
                "schema_restore_ver" => schema_restore_version,
                "table_infos" => table_infos.len());
            // 3. Build the schema file and upload to S3.
            let schemas = match self.build_new_schema(local_schema_file.as_ref(), table_infos) {
                Ok(schema) => schema,
                Err(err) => {
                    // TODO: report metrics and alarm.
                    error!("{}: build new schema failed", keyspace_id; "err" => ?err);
                    continue;
                }
            };
            debug!("{}: build new schema", keyspace_id; "schemas" => ?schemas, "cur_schema_ver" => ?cur_schema_version, "schema_ver" => schema_version);
            if schemas.is_none() {
                self.meta_file
                    .add_checked_version(keyspace_id, schema_version);
                if let Some(write_sequence) = update_write_sequence {
                    self.meta_file
                        .add_write_sequence(keyspace_id, write_sequence);
                }
                continue;
            }

            // build schema file with the schema restore version from shard stats.
            let new_schema_file_data = columnar::build_schema_file(
                keyspace_id,
                schema_version,
                schemas.unwrap(),
                schema_restore_version,
            );
            let file_id = *self
                .id_allocator
                .alloc_id(1)
                .map_err(|e| Error::Other(box_err!(e.to_string())))?
                .first()
                .unwrap();
            let dfs = self.ctx.s3fs.clone();
            let tx_clone = tx.clone();
            spawn_task_count += 1;
            runtime.spawn(async move {
                let data = Bytes::from(new_schema_file_data.clone());
                let opts = dfs::Options::default().with_type(dfs::FileType::Schema);
                let res: Result<u64> = dfs
                    .create(file_id, data.clone(), opts)
                    .await
                    .map(|()| file_id)
                    .map_err(Into::into);
                let _ = tx_clone
                    .send((keyspace_id, file_id, schema_version, data, res))
                    .map_err(|err| {
                        // Should happen only when `refresh_keyspace_schema` is aborted.
                        warn!("{} refresh keyspace schema: send failed: {:?}", keyspace_id, err;
                            "file_id" => file_id, "schema_ver" => schema_version);
                    });
            });
        }

        for _ in 0..spawn_task_count {
            let (keyspace_id, file_id, schema_version, data, res) = rx.recv().unwrap();
            if let Err(err) = res {
                error!("{}: failed to update schema file", keyspace_id;
                    "file_id" => file_id, "schema_ver" => schema_version, "err" => ?err);
                continue;
            }
            if let Err(err) =
                write_schema_file_to_local(&self.config.dir, keyspace_id, file_id, data)
            {
                error!("{}: failed to write schema file to local", keyspace_id;
                    "file_id" => file_id, "schema_ver" => schema_version, "err" => ?err);
                continue;
            }

            // 4. Callback TiKV to update the new schema file to shard meta.
            info!("{}: broadcast schema update to stores", keyspace_id;
                "file_id" => file_id, "schema_ver" => schema_version);
            if let Err(err) = broadcast_schema_update_to_all_stores(
                stores,
                self.security_mgr.clone(),
                self.config.http_timeout.0,
                keyspace_id,
                file_id,
            )
            .await
            {
                // Go on to save checked version on error as it would be partial successful.
                // `check_store_schema_version` in next round will update the failed stores.
                error!("{}: failed to broadcast schema update", keyspace_id;
                    "file_id" => file_id, "schema_ver" => schema_version, "err" => ?err);
            }

            self.meta_file
                .add_file(keyspace_id, file_id, schema_version);
            self.meta_file
                .add_checked_version(keyspace_id, schema_version);
            // Note: we don't update the write sequence here in case of
            // broadcast some stores failure. We can retry check the store
            // status to broadcast again for small keyspaces. The write sequence
            // will be updated if schema up to date.
        }
        // Save meta_file.
        let meta = self.meta_file.write();
        write_meta_file_to_local(&self.config.dir, Bytes::from(meta))
            .map_err(|err| -> Error { box_err!("write_meta_file_to_local failed: {:?}", err) })?;
        Ok(())
    }

    // return true if the keyspace is just restored.
    fn check_if_keyspace_restored(
        &self,
        keyspace_id: u32,
        keyspace_shard_stats: &[ShardStatsLite],
        cur_schema_version: i64,
        cur_restore_version: u64,
    ) -> bool {
        let stats_schema_version = keyspace_shard_stats
            .iter()
            .map(|s| s.schema_version)
            .max()
            .unwrap_or_default();
        // NOTE: schema_restore_version is the same for all shards.
        let stats_restore_version = keyspace_shard_stats
            .first()
            .map(|s| s.schema_restore_version)
            .unwrap_or_default();
        // If the schema_version in local schema file is larger than shard stats, or the
        // restore_version is inconsistent, it means the keyspace is just restored.
        if stats_restore_version > cur_restore_version {
            info!("{}: keyspace restored", keyspace_id;
                "stats_restore_ver" => stats_restore_version,
                "cur_restore_ver" => cur_restore_version,
                "stats_schema_ver" => stats_schema_version,
                "cur_schema_ver" => cur_schema_version,
                "keyspace_shard_stats" => ?keyspace_shard_stats,
            );
            return true;
        }
        false
    }

    fn check_if_keyspace_restore_in_progress(
        &self,
        keyspace_shard_stats: &[ShardStatsLite],
    ) -> bool {
        let first_restore_version = keyspace_shard_stats
            .first()
            .map(|s| s.schema_restore_version)
            .unwrap_or(0);
        keyspace_shard_stats
            .iter()
            .any(|s| s.schema_restore_version != first_restore_version)
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
            if cur_schema_version <= shard_stats.schema_version {
                // Schema version fallback will happen in production env, e.g. re-deploy the
                // schema manager.
                debug_assert_eq!(
                    cur_schema_version, shard_stats.schema_version,
                    "schema version fallback: {:?}, cur: {}",
                    shard_stats, cur_schema_version,
                );
                continue;
            }

            // When shard is `with_schema` but not overlapped with schema file, it means
            // that the schema has changed to having no required changes (i.e.
            // storage class & columnar) for the shard, but the relevant data
            // has not been cleared.
            if shard_stats.with_schema()
                || schema_file.overlap(&shard_stats.start, &shard_stats.end, keyspace_id)
            {
                info!(
                    "{}: store has stale schema version", keyspace_id;
                    "shard_ver" => shard_stats.schema_version, "cur_ver" => cur_schema_version, "shard" => ?shard_stats,
                );
                need_broadcast = true;
                break;
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
        local_schema_file: Option<&SchemaFile>,
        table_infos: Vec<TableInfo>,
    ) -> Result<Option<Vec<Schema>>> {
        // 3. Build the schema file and upload to S3.
        let mut schemas = if let Some(schema_file) = local_schema_file {
            Vec::with_capacity(table_infos.len() + schema_file.schema_count())
        } else {
            Vec::with_capacity(table_infos.len())
        };
        let mut to_be_removed = vec![];
        for ti in table_infos {
            if !ti.with_required_changes() {
                to_be_removed.push(ti.id);
                continue;
            }
            schemas.push(table_info_to_schema(&ti).map_err(|err| {
                error!("convert table info to schema failed"; "err" => ?err, "table" => ?ti);
                SchemaError(format!("{err:?}"))
            })?);
        }

        if let Some(schema_file) = &local_schema_file {
            // Check if the schemas contains in schema file to avoid useless update.
            if schema_file.contains(&schemas) && !schema_file.has_overlap_ids(&to_be_removed) {
                // The schema has no change or not relevant, skip.
                return Ok(None);
            }

            // Merge schemas in file to build the new one.
            let base = schema_file.export_schemas();
            schemas = merge_schema_diffs(base, schemas, &to_be_removed);
        }
        Ok(Some(schemas))
    }
}

fn table_info_to_partition_sc_spec(ti: &TableInfo) -> Option<Vec<(i64, StorageClassSpec)>> {
    ti.partition.as_ref().map(|p| {
        p.definitions
            .iter()
            .map(|d| (d.id, d.storage_class_spec()))
            .collect::<Vec<_>>()
    })
}

fn table_info_to_schema(ti: &TableInfo) -> Result<Schema> {
    let mut builder = SchemaBufBuilder::new(ti.id);
    builder
        .storage_class_spec(ti.storage_class_spec())
        .partitions(table_info_to_partition_sc_spec(ti));

    if ti.with_columnar() {
        let ti_cols = ti.cols.as_ref().unwrap();
        let mut ti_pk_cols = vec![];
        if let Some(idx_info) = ti.index_info.as_ref() {
            let pk_idx = idx_info.iter().find(|idx| idx.is_primary);
            if let Some(pk_idx) = pk_idx {
                for idx_col in &pk_idx.idx_cols {
                    ti_pk_cols.push(ti_cols[idx_col.offset as usize].clone());
                }
            }
        }
        let pk_col_ids: Vec<i64> = ti_pk_cols.iter().map(|c| c.id).collect();
        let pk_cols = convert_column_infos_to_tipb(&ti_pk_cols, ti.pk_is_handle)?;
        let mut columns = convert_column_infos_to_tipb(ti.cols.as_ref().unwrap(), ti.pk_is_handle)?;
        columns.retain(|c| !pk_col_ids.contains(&c.get_column_id()));
        if !ti.pk_is_handle {
            // make sure the common handle columns are ordered by offset.
            columns.extend_from_slice(&pk_cols);
        }
        let handle_column = if ti.is_common_handle {
            new_common_handle_column_info()
        } else if ti.pk_is_handle {
            let pk_handle_col = columns.iter().find(|c| c.get_pk_handle()).unwrap().clone();
            columns.retain(|c| !c.get_pk_handle());
            pk_handle_col
        } else {
            new_int_handle_column_info()
        };
        let vector_indexes = parse_vector_indexes(ti_cols, ti.index_info.as_ref());

        builder.columns(
            handle_column,
            new_version_column_info(),
            columns,
            pk_col_ids,
            vector_indexes,
        );
    }

    Ok(builder.build().into())
}

#[derive(Default, Deserialize)]
#[serde(default)]
struct WhiteListKeyspace {
    keyspace_ids: Vec<u32>,
}

pub struct SchemaManagerCore {
    ctx: Arc<Context>,
    security_mgr: Arc<SecurityManager>,
    config: SchemaManagerConfig,
    txn_client: TxnClient,
    meta_file: MetaFile,
    id_allocator: Arc<dyn IdAllocator>,
    whitelist_keyspaces: Option<HashSet<u32>>,
}

impl SchemaManagerCore {
    pub(crate) fn new(
        ctx: Arc<Context>,
        security_mgr: Arc<SecurityManager>,
        config: SchemaManagerConfig,
        txn_client: TxnClient,
    ) -> Self {
        let whitelist_keyspaces: Option<HashSet<u32>> =
            (!config.whitelist_file.as_os_str().is_empty()).then(|| {
                let data = fs::read_to_string(&config.whitelist_file).unwrap();
                let whitelist: WhiteListKeyspace = serde_json::from_str(&data).unwrap();
                let whitelist_keyspaces: HashSet<u32> =
                    whitelist.keyspace_ids.iter().cloned().collect();
                info!(
                    "whitelist keyspaces count: {}, keyspaces: {:?}",
                    whitelist_keyspaces.len(),
                    whitelist_keyspaces
                );
                whitelist_keyspaces
            });
        let meta_file_path = config.dir.join(META_FILE_NAME);
        let meta_file = if meta_file_path.exists() {
            MetaFile::open(LocalFile::open(0, meta_file_path, None, false).unwrap()).unwrap()
        } else {
            MetaFile::new()
        };
        let id_allocator = Arc::new(PdIdAllocator::new(ctx.pd.clone()));
        let mgr = Self {
            ctx,
            security_mgr,
            config,
            txn_client,
            meta_file,
            id_allocator,
            whitelist_keyspaces,
        };

        if !mgr.config.tikv_stores_tier.is_empty() {
            let (stores, stores_not_match) = mgr.get_tikv_stores();
            info!("schema manager: TiKV stores with matched tier: {:?}", stores_status_addr(&stores);
                "tier" => &mgr.config.tikv_stores_tier,
                "not_match" => ?stores_status_addr(&stores_not_match),
            );
        }

        mgr
    }

    // Return whether the keyspace_id is in the whitelist, return true if whitelist
    // not configured.
    fn in_whitelist(&self, keyspace_id: u32) -> bool {
        self.whitelist_keyspaces
            .as_ref()
            .map_or(true, |whitelist| whitelist.contains(&keyspace_id))
    }

    fn get_tikv_stores(&self) -> (Vec<Store>, Vec<Store> /* stores_not_match */) {
        let all_stores = match get_all_stores_except_tiflash(&self.ctx.pd) {
            Ok(stores) => stores,
            Err(err) => {
                warn!("schema manager: get stores failed: {:?}", err);
                return (vec![], vec![]);
            }
        };

        if !self.config.tikv_stores_tier.is_empty() {
            all_stores.into_iter().partition(|store| {
                store.get_labels().iter().any(|label| {
                    label.key.eq_ignore_ascii_case(TIKV_STORE_LABEL_TIER_KEY)
                        && label
                            .value
                            .eq_ignore_ascii_case(&self.config.tikv_stores_tier)
                })
            })
        } else {
            (all_stores, vec![])
        }
    }
}

const SCAN_BATCH_SIZE: u32 = 4096;

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

        let mut pairs = Vec::new();
        let mut current_key = start.to_vec();

        loop {
            let scan_range: BoundRange = (current_key.clone()..end.to_vec()).into();
            let batch: Vec<KvPair> = snapshot
                .scan(scan_range, SCAN_BATCH_SIZE)
                .await
                .map_err(|e| e.to_string())?
                .collect();
            let batch_len = batch.len();
            if batch_len == 0 {
                // end of scan
                break;
            }
            current_key = batch.last().unwrap().key().clone().into();
            current_key.push(0);

            for KvPair(key, val) in batch {
                pairs.push((key.into(), val));
            }
            if batch_len < SCAN_BATCH_SIZE as usize {
                // end of scan, no need to continue
                break;
            }
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
            .map_err(|e| format!("schema manager: get timestamp failed: {e:?})"))?;
        let mut snapshot = self
            .txn_client
            .snapshot(start_ts.clone(), TransactionOptions::new_pessimistic());
        let val = snapshot.get(key.to_vec()).await.map_err(|e| {
            format!(
                "schema manager: kv get failed: {}: {e:?})",
                log_wrappers::Value::key(key)
            )
        })?;
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
            .map_err(|e| format!("schema manager: get timestamp failed: {e:?})"))?;
        let mut snapshot = self
            .txn_client
            .snapshot(start_ts, TransactionOptions::new_pessimistic());
        let pairs: HashMap<Key, Value> = snapshot
            .batch_get(keys.to_vec())
            .await
            .map_err(|e| format!("schema manager: kv batch get failed: {e:?})"))?
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

fn parse_vector_indexes(
    ti_cols: &[ColumnInfo],
    idx_infos: Option<&Vec<IndexInfo>>,
) -> Vec<VectorIndexDef> {
    let mut vec_idxes = vec![];
    for col in ti_cols {
        if let Some(vec_idx_info) = &col.vector_index {
            vec_idxes.push(new_vec_idx_def(vec_idx_info, 0, col.id));
        }
    }
    if let Some(idx_infos) = idx_infos {
        for idx_info in idx_infos {
            if let Some(vec_idx_info) = &idx_info.vector_index {
                let column_offset = idx_info.idx_cols[0].offset as usize;
                let col_id = ti_cols[column_offset].id;
                vec_idxes.push(new_vec_idx_def(vec_idx_info, idx_info.id, col_id));
            }
        }
    }
    vec_idxes
}

fn new_vec_idx_def(info: &VectorIndexInfo, index_id: i64, col_id: i64) -> VectorIndexDef {
    let mut vec_idx_def = VectorIndexDef::default();
    vec_idx_def.index_id = index_id;
    vec_idx_def.col_id = col_id;
    vec_idx_def.dimension = info.dimension as usize;
    vec_idx_def.index_kind = info.kind.clone();
    vec_idx_def.specs.insert(
        VECTOR_INDEX_SPEC_KEY_DISTANCE_METRIC.to_string(),
        info.distance_metric.as_bytes().to_vec(),
    );
    vec_idx_def
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
    mut base: BTreeMap<i64, Schema>,
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
    let fd = Arc::new(fs::File::open(file_path.as_path())?);
    let local_file = Arc::new(LocalFile::from_file(file_id, file_path, fd)?);
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

fn remove_schema_file_from_local<P: AsRef<Path>>(
    base_dir: P,
    keyspace_id: u32,
    file_ids: &[u64],
) -> Result<()> {
    let dir = base_dir.as_ref().join(keyspace_id.to_string());
    for file_id in file_ids {
        let filename = format!("{:016x}.schema", file_id);
        let file_path = dir.join(filename);
        fs::remove_file(file_path.as_path())?;
    }
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

pub async fn broadcast_schema_update_to_all_stores(
    stores: &[Store],
    security_mgr: Arc<SecurityManager>,
    timeout: Duration,
    keyspace_id: u32,
    file_id: u64,
) -> Result<()> {
    for store in stores {
        let status_addr = store.get_status_address();
        let uri = security_mgr
            .build_uri(format!(
                "{}/schema_file?keyspace_id={}&file_id={}",
                status_addr, keyspace_id, file_id
            ))
            .unwrap();
        let req = || Request::post(uri.clone()).body(Body::empty()).unwrap();
        if let Err(err) =
            send_request_to_store_with_retry(req, store, security_mgr.as_ref(), timeout).await
        {
            return Err(box_err!(
                "broadcast schema update to store {} failed: {:?}",
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
    let resp_bytes =
        send_request_to_store_with_retry(req, store, security_mgr.as_ref(), timeout).await?;
    let resp: Vec<ShardStatsLite> = serde_json::from_slice(&resp_bytes)?;
    Ok(resp)
}

#[inline]
fn stores_status_addr(stores: &[Store]) -> Vec<&str> {
    stores.iter().map(|s| s.get_status_address()).collect()
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, fs};

    use bytes::Bytes;
    use kvengine::{
        table::{
            columnar::{
                build_schema_file, new_int_handle_column_info, new_version_column_info, SchemaBuf,
            },
            file::LocalFile,
        },
        ShardStatsLite,
    };
    use schema::schema::StorageClassSpec;
    use tikv_util::info;

    use super::*;

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
            let schema = SchemaBuf::new(
                i,
                new_int_handle_column_info(),
                new_version_column_info(),
                vec![new_int_handle_column_info()],
                vec![],
                vec![],
                StorageClassSpec::default(),
                None,
            );
            schemas.push(schema.into());
        }
        let schema_file_data = build_schema_file(1234, 100, schemas.clone(), 0);
        write_schema_file_to_local(dir.path(), 1234, 1000, Bytes::from(schema_file_data)).unwrap();
        schemas.push(
            SchemaBuf::new(
                11,
                new_int_handle_column_info(),
                new_version_column_info(),
                vec![new_int_handle_column_info()],
                vec![],
                vec![],
                StorageClassSpec::default(),
                None,
            )
            .into(),
        );
        let schema_file_data = build_schema_file(1234, 201, schemas, 12345);
        write_schema_file_to_local(dir.path(), 1234, 1001, Bytes::from(schema_file_data)).unwrap();

        // schema_file is the newest schema file of the keyspace.
        let schema_file = read_schema_file_from_local(dir.path(), &MetaFile::new(), 1234)
            .unwrap()
            .unwrap();
        assert_eq!(schema_file.get_keyspace_id(), 1234);
        assert_eq!(schema_file.get_version(), 201);
        assert_eq!(schema_file.get_restore_version(), 12345);
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
        let meta_file = LocalFile::open(0, meta_file_path, None, false).unwrap();
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

    #[test]
    fn test_update_shard_stats() {
        let make_shard_stats = |id: u64, start: Vec<u8>, end: Vec<u8>| -> ShardStatsLite {
            ShardStatsLite {
                id,
                ver: 1,
                start: start.into(),
                end: end.into(),
                inner_key_off: 0,
                total_size: 0,
                schema_version: 1000,
                schema_restore_version: 0,
                write_sequence: 0,
                storage_class_spec: Default::default(),
                columnar_tables: 0,
            }
        };
        let mut keyspace_stats = HashMap::new();
        let shard_stats = make_shard_stats(1, vec![120, 255, 255, 255], vec![]);
        SchemaManager::update_keyspace_stats(&mut keyspace_stats, shard_stats);
        assert!(keyspace_stats.is_empty());
        let shard_stats = make_shard_stats(1, vec![120, 0, 0, 1], vec![120, 0, 0, 3]);
        SchemaManager::update_keyspace_stats(&mut keyspace_stats, shard_stats);
        assert!(keyspace_stats.is_empty());
        let shard_stats = make_shard_stats(1, vec![120, 0, 0, 2], vec![120, 0, 0, 3, 4]);
        assert!(keyspace_stats.is_empty());
        SchemaManager::update_keyspace_stats(&mut keyspace_stats, shard_stats);
        let shard_stats = make_shard_stats(1, vec![120, 0, 0, 1, 3], vec![120, 0, 0, 1, 4]);
        SchemaManager::update_keyspace_stats(&mut keyspace_stats, shard_stats);
        assert_eq!(keyspace_stats.len(), 1);
        let shard_stats = make_shard_stats(2, vec![120, 0, 0, 2], vec![120, 0, 0, 3]);
        SchemaManager::update_keyspace_stats(&mut keyspace_stats, shard_stats);
        assert_eq!(keyspace_stats.len(), 1);
        let mut shard_stats = make_shard_stats(2, vec![120, 0, 0, 2], vec![120, 0, 0, 3]);
        shard_stats.total_size = 100;
        SchemaManager::update_keyspace_stats(&mut keyspace_stats, shard_stats);
        assert_eq!(keyspace_stats.len(), 2);
        let shard_stats = make_shard_stats(3, vec![120, 0, 0, 3, 3], vec![120, 0, 0, 4]);
        SchemaManager::update_keyspace_stats(&mut keyspace_stats, shard_stats);
        assert_eq!(keyspace_stats.len(), 3);
    }
}
