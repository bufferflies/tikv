// Copyright 2025 TiKV Project Authors. Licensed under Apache-2.0.

use std::sync::Arc;

use async_trait::async_trait;
use bytes::Bytes;
use kvengine::dfs::{Dfs, LocalFs, Options, Result};
use tempfile::TempDir;

// TempDirDfs is a wrapper of `LocalFs` with a temp dir.
// It's only used for integration/failpoints tests.
#[derive(Clone)]
pub struct TempDirFs {
    _dir: Arc<TempDir>,
    fs: LocalFs,
}

impl Default for TempDirFs {
    fn default() -> Self {
        let dir = TempDir::new().unwrap();
        let fs = LocalFs::new(dir.path());
        Self {
            _dir: Arc::new(dir),
            fs,
        }
    }
}

#[async_trait]
impl Dfs for TempDirFs {
    async fn read_file(&self, file_id: u64, opts: Options) -> Result<Bytes> {
        self.fs.read_file(file_id, opts).await
    }

    async fn create(&self, file_id: u64, data: Bytes, opts: Options) -> Result<()> {
        self.fs.create(file_id, data, opts).await
    }

    async fn remove(&self, file_id: u64, file_len: Option<u64>, opts: Options) {
        self.fs.remove(file_id, file_len, opts).await
    }

    async fn permanently_remove(&self, file_id: u64, opts: Options) -> Result<()> {
        self.fs.permanently_remove(file_id, opts).await
    }

    fn get_runtime(&self) -> &tokio::runtime::Runtime {
        self.fs.get_runtime()
    }
}
