// Copyright 2020 TiKV Project Authors. Licensed under Apache-2.0.

use std::sync::Arc;

use async_trait::async_trait;
use bytes::Bytes;
use file_system::{get_io_rate_limiter, get_io_type, IoOp, IoRateLimiter};
use futures::Stream;
use serde::{Deserialize, Serialize};

use crate::Result;

pub trait FileSystemInspector: Sync + Send {
    fn read(&self, len: usize) -> Result<usize>;
    fn write(&self, len: usize) -> Result<usize>;
}

pub struct EngineFileSystemInspector {
    limiter: Option<Arc<IoRateLimiter>>,
}

impl EngineFileSystemInspector {
    #[allow(dead_code)]
    pub fn new() -> Self {
        EngineFileSystemInspector {
            limiter: get_io_rate_limiter(),
        }
    }

    pub fn from_limiter(limiter: Option<Arc<IoRateLimiter>>) -> Self {
        EngineFileSystemInspector { limiter }
    }
}

impl Default for EngineFileSystemInspector {
    fn default() -> Self {
        Self::new()
    }
}

impl FileSystemInspector for EngineFileSystemInspector {
    fn read(&self, len: usize) -> Result<usize> {
        if let Some(limiter) = &self.limiter {
            let io_type = get_io_type();
            Ok(limiter.request(io_type, IoOp::Read, len))
        } else {
            Ok(len)
        }
    }

    fn write(&self, len: usize) -> Result<usize> {
        if let Some(limiter) = &self.limiter {
            let io_type = get_io_type();
            Ok(limiter.request(io_type, IoOp::Write, len))
        } else {
            Ok(len)
        }
    }
}

/// (`start_off`, `end_off`):
/// * (None, None): read total object.
/// * (Some(s), Some(e)): read range [s, e).
/// * (Some(s), None): read from `s` to end of object.
/// * (None, Some(n)): read last `n` bytes.
#[derive(Debug, Default)]
pub struct GetObjectOptions {
    pub start_off: Option<u64>,
    pub end_off: Option<u64>,
}

impl GetObjectOptions {
    pub fn is_full_range(&self) -> bool {
        self.start_off.unwrap_or_default() == 0 && self.end_off.is_none()
    }

    pub fn range_string(&self) -> String {
        // https://docs.aws.amazon.com/AmazonS3/latest/API/API_GetObject.html
        // For example:
        // The first 10 bytes: Range: bytes=0-9
        // The last 10 bytes: Range: bytes=-10
        format!(
            "{}-{}",
            self.start_off.map_or(String::new(), |s| format!("{}", s)),
            self.end_off.map_or(String::new(), |e| format!("{}", e - 1))
        )
    }
}

#[derive(Clone, Default, Debug, Serialize, Deserialize)]
#[serde(default)]
#[serde(rename_all = "PascalCase")]
pub struct ListObjectContent {
    pub key: String,
    pub last_modified: String,
    pub storage_class: String,
    pub size: u64, // in bytes.
}

#[async_trait]
pub trait ObjectStorage: Sync + Send {
    fn put_objects(&self, objects: Vec<(String, Bytes)>) -> std::result::Result<(), String>;
    fn get_objects(
        &self,
        keys: Vec<(String, GetObjectOptions)>,
    ) -> std::result::Result<Vec<(String, Bytes)>, String>;
    fn list_objects(
        &self,
        start_after: &str, // The key to start after when listing objects (exclusive).
        prefix: Option<&str>,
        max_keys: Option<u32>,
    ) -> std::result::Result<(Vec<ListObjectContent>, Option<String>), String>;

    async fn download_objects(
        &self,
        items: Vec<(String, GetObjectOptions)>,
        concurrency: usize,
        dest_dir: std::path::PathBuf,
    ) -> std::pin::Pin<
        Box<dyn Stream<Item = std::result::Result<(String, std::path::PathBuf), String>> + Send>,
    >;
}
