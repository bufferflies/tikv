// Copyright 2022 TiKV Project Authors. Licensed under Apache-2.0.

use tikv_util::config::ReadableSize;

#[derive(Clone, Serialize, Deserialize, PartialEq, Debug)]
#[serde(default)]
#[serde(rename_all = "kebab-case")]
pub struct Config {
    /// Compress a log batch if its size exceeds this value. Setting it to zero
    /// disables compression.
    ///
    /// Default: "8KB"
    pub batch_compression_threshold: ReadableSize,
    /// Target file size for rotating log files.
    ///
    /// Default: "512MB"
    pub target_file_size: ReadableSize,

    /// Limit the worker io
    ///
    /// Default: "125MB"
    pub worker_rate_limit: ReadableSize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            batch_compression_threshold: ReadableSize::kb(8),
            target_file_size: ReadableSize::mb(512),
            worker_rate_limit: ReadableSize::mb(125),
        }
    }
}

impl Config {
    pub fn new(target_file_size: usize) -> Self {
        let mut cfg = Self::default();
        cfg.target_file_size = ReadableSize(target_file_size as u64);
        cfg
    }
}
