// Copyright 2025 TiKV Project Authors. Licensed under Apache-2.0.

use std::{ops::Deref, sync::Arc};

use quick_cache::sync::Cache;
use tikv_util::config::AbsoluteOrPercentSize;

use crate::table::columnar::ColumnarFile;

#[derive(Clone)]
pub struct ColumnarFileCache {
    cache: Arc<Cache<u64, ColumnarFile, ColumnarFileCacheWeighter>>,
}

impl Deref for ColumnarFileCache {
    type Target = Cache<u64, ColumnarFile, ColumnarFileCacheWeighter>;
    fn deref(&self) -> &Self::Target {
        &self.cache
    }
}

impl ColumnarFileCache {
    // NOTE: The `capacity` is not accurate. We use items_capacity to limit the
    // cache usage.
    pub fn new(items_capacity: usize, capacity: u64) -> Self {
        Self {
            cache: Arc::new(Cache::with_weighter(
                items_capacity,
                capacity,
                ColumnarFileCacheWeighter {},
            )),
        }
    }
}

#[derive(Clone)]
pub struct ColumnarFileCacheWeighter;

impl quick_cache::Weighter<u64, ColumnarFile> for ColumnarFileCacheWeighter {
    fn weight(&self, _: &u64, file: &ColumnarFile) -> u64 {
        // NOTE: The mem_size is not accurate.
        file.mem_size()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(default)]
#[serde(rename_all = "kebab-case")]
pub struct ColumnarFileCacheConfig {
    pub cache_size: usize,
    pub cache_cap: AbsoluteOrPercentSize,
}

impl Default for ColumnarFileCacheConfig {
    fn default() -> Self {
        Self {
            cache_size: 4096,
            cache_cap: AbsoluteOrPercentSize::Percent(1.0),
        }
    }
}
