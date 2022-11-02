// Copyright 2020 TiKV Project Authors. Licensed under Apache-2.0.

use std::sync::Arc;

use bytes::Bytes;
use file_system::{get_io_rate_limiter, get_io_type, IOOp, IORateLimiter};

pub trait FileSystemInspector: Sync + Send {
    fn read(&self, len: usize) -> Result<usize, String>;
    fn write(&self, len: usize) -> Result<usize, String>;
}

pub struct EngineFileSystemInspector {
    limiter: Option<Arc<IORateLimiter>>,
}

impl EngineFileSystemInspector {
    #[allow(dead_code)]
    pub fn new() -> Self {
        EngineFileSystemInspector {
            limiter: get_io_rate_limiter(),
        }
    }

    pub fn from_limiter(limiter: Option<Arc<IORateLimiter>>) -> Self {
        EngineFileSystemInspector { limiter }
    }
}

impl Default for EngineFileSystemInspector {
    fn default() -> Self {
        Self::new()
    }
}

impl FileSystemInspector for EngineFileSystemInspector {
    fn read(&self, len: usize) -> Result<usize, String> {
        if let Some(limiter) = &self.limiter {
            let io_type = get_io_type();
            Ok(limiter.request(io_type, IOOp::Read, len))
        } else {
            Ok(len)
        }
    }

    fn write(&self, len: usize) -> Result<usize, String> {
        if let Some(limiter) = &self.limiter {
            let io_type = get_io_type();
            Ok(limiter.request(io_type, IOOp::Write, len))
        } else {
            Ok(len)
        }
    }
}

pub trait ObjectStorage: Sync + Send {
    fn put_objects(&self, objects: Vec<(String, Bytes)>) -> Result<(), String>;
    fn get_objects(&self, keys: Vec<String>) -> Result<Vec<(String, Bytes)>, String>;
}
