// Copyright 2023 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    collections::{HashMap, HashSet},
    error::Error,
};

use tikv_util::config::ReadableDuration;

use crate::table::blobtable::builder::BlobTableBuildOptions;

pub(crate) const DEFAULT_COMPACTION_TOMBS_RATIO: f64 = 0.2;
pub(crate) const DEFAULT_COMPACTION_TOMBS_COUNT: u64 = 10000;

#[derive(Default, Clone, Serialize, Deserialize, PartialEq, Debug)]
#[serde(default)]
#[serde(rename_all = "kebab-case")]
pub struct PerKeyspaceConfig {
    pub keyspace: u32,
    pub blob_table_build_options: BlobTableBuildOptions,
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

    pub per_keyspace_configs: Vec<PerKeyspaceConfig>,
    // Note: `per_keyspace_configs` must be the last field. Otherwise serializing the config
    // will meet a "ValueAfterTable" error.
    // See https://docs.rs/toml/0.5.11/toml/ser/enum.Error.html#variant.ValueAfterTable.
}

impl Default for Config {
    fn default() -> Self {
        Self {
            max_del_range_delay: ReadableDuration::secs(3600),
            compaction_request_version: 2,
            compaction_tombs_ratio: DEFAULT_COMPACTION_TOMBS_RATIO,
            compaction_tombs_count: DEFAULT_COMPACTION_TOMBS_COUNT,
            per_keyspace_configs: vec![],
        }
    }
}

impl Config {
    pub fn validate(&self) -> Result<(), Box<dyn Error>> {
        let mut keyspace_set = HashSet::new();
        for cfg in &self.per_keyspace_configs {
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
