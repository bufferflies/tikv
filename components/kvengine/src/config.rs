// Copyright 2023 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    collections::{HashMap, HashSet},
    error::Error,
};

use tikv_util::config::{AbsoluteOrPercentSize, ReadableDuration};

use crate::{
    ia::util::IaConfig,
    table::{
        blobtable::builder::BlobTableBuildOptions, columnar::ColumnarTableBuildOptions,
        sstable::BlockCacheType, vector_index::VectorIndexBuildOptions, ChecksumType,
    },
};

pub(crate) const DEFAULT_COMPACTION_REQUEST_VERSION: u32 = 3;
pub(crate) const DEFAULT_COMPACTION_TOMBS_RATIO: f64 = 0.3;
pub(crate) const DEFAULT_COMPACTION_TOMBS_COUNT: u64 = 10000;

/// The maximum size of a memtable is limited to 128MB. Otherwise it's possible
/// to use up arena blocks and lead to panic.
/// See `kvengine::table::memtable::arena::block_cap`.
pub const MEM_TABLE_MAX_SIZE: u64 = 128 * 1024 * 1024;

pub const DEFAULT_SOFT_REGION_MEM_USAGE_LIMIT_MB: u64 = 256;
pub const DEFAULT_HARD_REGION_MEM_USAGE_LIMIT_MB: u64 = 512;
pub const DEFAULT_MAX_REGION_SPEED_LIMIT_MB_PER_SEC: u64 = 50;
pub const DEFAULT_MIN_REGION_SPEED_LIMIT_MB_PER_SEC: u64 = 1;

pub const DEFAULT_FD_CACHE_CAPCITY: usize = 200000;

// If the number of panic regions of a table exceeds this threshold, the table
// will be auto blacklisted.
pub const DEFAULT_TABLE_AUTO_BLACKLIST_THRESHOLD: u64 = 3;
// If the number of blacklisted tables of a keyspace exceeds this threshold,
// the keyspace will be auto blacklisted.
pub const DEFAULT_KEYSPACE_AUTO_BLACKLIST_THRESHOLD: u64 = 4;

#[derive(Clone, Serialize, Deserialize, PartialEq, Debug)]
#[serde(default)]
#[serde(rename_all = "kebab-case")]
pub struct PerKeyspaceConfig {
    pub keyspace: u32,

    /// The factor of default split size for the keyspace, valid range from 0.1
    /// to 10.0.
    pub split_size_factor: f64,

    /// enable blob table for the keyspace.
    pub enable_blob: bool,
}

impl Default for PerKeyspaceConfig {
    fn default() -> Self {
        Self {
            keyspace: 0,
            split_size_factor: 1.0,
            enable_blob: false,
        }
    }
}

#[derive(Clone, Serialize, Deserialize, PartialEq, Debug)]
#[serde(default)]
#[serde(rename_all = "kebab-case")]
pub struct Config {
    /// The maximum delay duration for delete range.
    pub max_del_range_delay: ReadableDuration,
    pub compaction_request_version: u32,
    /// The ratio threshold of tombstone entries to trigger compaction.
    pub compaction_tombs_ratio: f64,
    /// The number threshold of tombstone entries to trigger compaction.
    pub compaction_tombs_count: u64,

    /// The remote worker address to run analyze and checksum requests.
    pub remote_worker_addr: String,

    /// The remote coprocessor address to run heavy coprocessor requests.
    pub remote_coprocessor_addr: String,

    /// The minimum number of blocks for a coprocessor request to be run on
    /// remote worker.
    pub remote_coprocessor_min_blocks_size: usize,

    /// Always send to remote coprocessor if the number of ranges reach this
    /// value.
    pub remote_coprocessor_num_ranges: usize,

    /// If enabled, flush large L0 file will split into multiple files.
    pub flush_split_l0: bool,
    /// If enabled, major compaction will update inner key offset from 0 to 4.
    pub update_inner_key_offset: bool,

    /// If enabled, columnar table will not be loaded. Used for troubleshooting.
    pub ignore_columnar_table_load: bool,
    /// Enable building columnar table. Default is false.
    pub build_columnar: bool,
    /// Enable columnar table read. Default is false.
    pub read_columnar: bool,

    pub txn_file_worker_pool_size: Option<usize>,

    /// concurrency per core for loading dfs files.
    pub dfs_load_concurrency_per_core: usize,

    /// the capacity of open fd cache.
    pub fd_cache_capacity: usize,

    /// The panic regions threshold of the table to auto blacklist.
    pub table_auto_blacklist_threshold: u64,

    /// The panic tables threshold of the keyspace to auto blacklist.
    pub keyspace_auto_blacklist_threshold: u64,

    pub checksum_type: ChecksumType,

    pub block_cache_type: BlockCacheType,

    pub per_keyspace_configs: Vec<PerKeyspaceConfig>,

    /// Enable IA by setting `ia.mem_cap > 0 && ia.disk_cap > 0`.
    pub ia: IaConfig,

    pub blob_table_build_options: BlobTableBuildOptions,

    pub columnar_table_build_options: ColumnarTableBuildOptions,

    pub vector_index_build_options: VectorIndexBuildOptions,
    // Note: Fields of simple (not structure) type can not be the last. Otherwise serializing the
    // config will meet the "ValueAfterTable" error.
    // See https://docs.rs/toml/0.5.11/toml/ser/enum.Error.html#variant.ValueAfterTable.
}

impl Default for Config {
    fn default() -> Self {
        Self {
            max_del_range_delay: ReadableDuration::secs(3600),
            compaction_request_version: DEFAULT_COMPACTION_REQUEST_VERSION,
            compaction_tombs_ratio: DEFAULT_COMPACTION_TOMBS_RATIO,
            compaction_tombs_count: DEFAULT_COMPACTION_TOMBS_COUNT,
            table_auto_blacklist_threshold: DEFAULT_TABLE_AUTO_BLACKLIST_THRESHOLD,
            keyspace_auto_blacklist_threshold: DEFAULT_KEYSPACE_AUTO_BLACKLIST_THRESHOLD,
            remote_worker_addr: "".to_string(),
            remote_coprocessor_addr: "".to_string(),
            remote_coprocessor_min_blocks_size: 32 * 1024 * 1024,
            remote_coprocessor_num_ranges: 2048,
            flush_split_l0: true,
            update_inner_key_offset: false,
            txn_file_worker_pool_size: None,
            // 8GB memory per core, 64 * 16MB files consumes 1GB at max.
            dfs_load_concurrency_per_core: 64,
            fd_cache_capacity: DEFAULT_FD_CACHE_CAPCITY,
            checksum_type: ChecksumType::Crc32,
            block_cache_type: BlockCacheType::Moka,
            blob_table_build_options: Default::default(),
            per_keyspace_configs: vec![],
            ia: IaConfig {
                mem_cap: AbsoluteOrPercentSize::Percent(15.0),
                disk_cap: AbsoluteOrPercentSize::Percent(70.0),
                dynamic_capacity: true,
                ..Default::default()
            },
            columnar_table_build_options: Default::default(),
            vector_index_build_options: Default::default(),
            ignore_columnar_table_load: false,
            build_columnar: false,
            read_columnar: false,
        }
    }
}

impl Config {
    pub fn validate(&self) -> Result<(), Box<dyn Error>> {
        let mut keyspace_set = HashSet::new();
        for cfg in &self.per_keyspace_configs {
            if cfg.split_size_factor > 10.0 || cfg.split_size_factor < 0.1 {
                return Err(format!(
                    "invalid split size factor {} for keyspace {}",
                    cfg.split_size_factor, cfg.keyspace
                )
                .into());
            }
            if !keyspace_set.insert(cfg.keyspace) {
                return Err(format!("duplicate keyspace {}", cfg.keyspace).into());
            }
        }
        Ok(())
    }

    pub fn get_per_keyspace_configs(&self) -> HashMap<u32, PerKeyspaceConfig> {
        let mut map = HashMap::new();
        for cfg in &self.per_keyspace_configs {
            map.insert(cfg.keyspace, cfg.clone());
        }
        map
    }
}
