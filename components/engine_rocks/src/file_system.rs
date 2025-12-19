// Copyright 2020 TiKV Project Authors. Licensed under Apache-2.0.

use std::sync::Arc;

use engine_traits::{EngineFileSystemInspector, FileSystemInspector};
use rocksdb::FileSystemInspector as DBFileSystemInspector;

use crate::{e2r, r2e, raw::Env};

// Use engine::Env directly since Env is not abstracted.
pub(crate) fn get_env(
    base_env: Option<Arc<Env>>,
    limiter: Option<Arc<file_system::IoRateLimiter>>,
) -> engine_traits::Result<Arc<Env>> {
    let base_env = base_env.unwrap_or_else(|| Arc::new(Env::default()));
    Ok(Arc::new(
        Env::new_file_system_inspected_env(
            base_env,
            WrappedFileSystemInspector {
                inspector: EngineFileSystemInspector::from_limiter(limiter),
            },
        )
        .map_err(r2e)?,
    ))
}

pub struct WrappedFileSystemInspector<T: FileSystemInspector> {
    inspector: T,
}

impl<T: FileSystemInspector> DBFileSystemInspector for WrappedFileSystemInspector<T> {
    fn read(&self, len: usize) -> Result<usize, String> {
        self.inspector.read(len).map_err(e2r)
    }

    fn write(&self, len: usize) -> Result<usize, String> {
        self.inspector.write(len).map_err(e2r)
    }
}
