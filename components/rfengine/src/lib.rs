// Copyright 2021 TiKV Project Authors. Licensed under Apache-2.0.
#![feature(let_chains)]
#![cfg_attr(test, feature(test))]
// Bytes as map key
#![allow(clippy::mutable_key_type)]

#[cfg(test)]
extern crate test;
#[allow(unused_extern_crates)]
extern crate tikv_alloc;
#[macro_use]
extern crate serde_derive;

mod config;

pub use config::Config as RfEngineConfig;

pub mod compact_worker;
pub mod dfs_worker;
pub mod engine;
pub mod iterator;
pub mod load;
mod log_batch;
pub mod manifest;
mod metrics;
pub use metrics::RFENGINE_DFS_WORKER_HEALTHY_GAUGE; // For test purpose.
pub mod service_worker;
pub mod traits;
pub mod utils;
mod write_batch;
pub mod writer;

use std::num::ParseIntError;

pub use compact_worker::*;
pub use dfs_worker::*;
pub use engine::*;
use iterator::*;
pub use log_batch::RaftLogOp;
use metrics::*;
use thiserror::Error as ThisError;
use tikv_util::errors::IoError;
pub use traits::*;
pub use utils::*;
pub use write_batch::WriteBatch;
pub use writer::*;

pub type Result<T> = std::result::Result<T, Error>;

/// Information about an async fetch request that needs to be handled by upper
/// layer
#[derive(Debug, Clone)]
pub struct AsyncFetchInfo {
    pub peer_id: u64,
    pub low: u64,
    pub high: u64,
    pub max_size: Option<usize>,
    pub region_id: u64,
}

impl std::fmt::Display for AsyncFetchInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "AsyncFetch(peer_id={}, region_id={}, range=[{},{}))",
            self.peer_id, self.region_id, self.low, self.high,
        )
    }
}

impl std::error::Error for AsyncFetchInfo {}

#[derive(Debug, ThisError)]
pub enum Error {
    #[error("IO error: {0}")]
    Io(#[from] IoError),
    #[error("EOF")]
    Eof,
    #[error("parse error")]
    ParseError,
    #[error("Open error: {0}")]
    Open(String),
    #[error("Corruption: {msg}, epoch_id {epoch_id}, offset {offset}")]
    Corruption {
        msg: String,
        epoch_id: u32,
        offset: u64,
        data: Vec<u8>,
    },
    #[error("WAL Epoch {epoch_id} is overwritten")]
    WalEpochOverwritten { epoch_id: u32 },
    #[error("Snapshot is oversize: {0}")]
    SnapshotOversize(u64),
    #[error("Memory limit exceed, request {request}, available {available}")]
    MemoryLimitExceed { request: usize, available: i64 },
    #[error("Async fetch required: {0}")]
    AsyncFetch(AsyncFetchInfo),
    #[error("The entries of region is unavailable")]
    EntriesUnavailable,
    #[error("The entries of region is compacted")]
    EntriesCompacted,
    #[error("Other error: {0}")]
    Other(String),
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        if e.kind() == std::io::ErrorKind::UnexpectedEof {
            return Error::Eof;
        }
        Error::Io(IoError::new(e, "".to_string()))
    }
}

impl From<ParseIntError> for Error {
    fn from(_: ParseIntError) -> Self {
        Error::ParseError
    }
}

impl From<String> for Error {
    fn from(msg: String) -> Self {
        Error::Other(msg)
    }
}

impl From<Error> for raft::Error {
    fn from(e: Error) -> raft::Error {
        match e {
            Error::EntriesUnavailable => raft::Error::Store(raft::StorageError::Unavailable),
            Error::EntriesCompacted => raft::Error::Store(raft::StorageError::Compacted),
            Error::AsyncFetch(async_info) => {
                let boxed = Box::new(async_info) as Box<dyn std::error::Error + Sync + Send>;
                raft::Error::Store(raft::StorageError::Other(boxed))
            }
            e => {
                let boxed = Box::new(e) as Box<dyn std::error::Error + Sync + Send>;
                raft::Error::Store(raft::StorageError::Other(boxed))
            }
        }
    }
}
