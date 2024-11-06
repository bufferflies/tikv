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

    /// The directory to store the wal files for synchronous write.
    /// It's used to reduce the latency of writing wal.
    pub wal_sync_dir: String,

    /// Whether to enable the lightweight backup.
    ///
    /// Default: false
    pub lightweight_backup: bool,

    /// Target file size for wal chunk files.
    ///
    /// Default: "128MB"
    pub wal_chunk_target_file_size: ReadableSize,

    /// Open RfEngine in cli mode used by tools. Skip serde for this field to
    /// avoid misconfiguration. If cli_mode is set, the sync WAL writer will
    /// avoid sync to improve performance.
    ///
    /// Default: false
    #[serde(skip)]
    pub cli_mode: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            batch_compression_threshold: ReadableSize::kb(8),
            target_file_size: ReadableSize::mb(512),
            worker_rate_limit: ReadableSize::mb(125),
            wal_sync_dir: "".to_owned(),
            lightweight_backup: false,
            wal_chunk_target_file_size: ReadableSize::mb(128),
            cli_mode: false,
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
