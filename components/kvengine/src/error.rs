// Copyright 2021 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    fmt::{Debug, Display},
    io,
};

use crate::{dfs, table};

pub const MERGE_REGION_WITH_TXN_FILE_LOCKS_ERR_MSG: &str =
    "fail to merge source region with txn file locks";

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("key not found")]
    KeyNotFound,
    #[error("shard not found")]
    ShardNotFound,
    #[error("shard version not match")]
    ShardNotMatch,
    #[error("already splitting")]
    AlreadySplitting,
    #[error("alloc id error {0}")]
    ErrAllocId(String),
    #[error("open error {0}")]
    ErrOpen(String),
    #[error("table error {0}")]
    TableError(table::Error),
    #[error("dfs error {0}")]
    DfsError(dfs::Error),
    #[error("IO error {ctx}: {err}")]
    Io { err: io::Error, ctx: String },
    #[error("remote compaction {0}")]
    RemoteCompaction(String),
    #[error("apply change set {0}")]
    ApplyChangeSet(String),
    #[error("ingest files {0}")]
    IngestFiles(String),
    #[error("check merge {0}")]
    CheckMerge(String),
    #[error("incompatible remote compactor {}:{}", .url, .msg)]
    IncompatibleRemoteCompactor { url: String, msg: String },
    #[error("fallback to local compactor disabled")]
    FallbackLocalCompactorDisabled,
    #[error("not retryable compaction error {0}")]
    CompactionNotRetryable(String),
    #[error("remote read error {0}")]
    RemoteRead(String),
    #[error("Other error {0}")]
    Other(#[from] Box<dyn std::error::Error + Send + Sync>),
}

impl From<table::Error> for Error {
    fn from(e: table::Error) -> Self {
        Error::TableError(e)
    }
}

impl From<dfs::Error> for Error {
    fn from(e: dfs::Error) -> Self {
        Error::DfsError(e)
    }
}

impl From<hyper::Error> for Error {
    fn from(e: hyper::Error) -> Self {
        Error::RemoteCompaction(e.to_string())
    }
}

impl From<http::Error> for Error {
    fn from(e: http::Error) -> Self {
        Error::RemoteCompaction(e.to_string())
    }
}

// Ref: [`anyhow::Context`](https://github.com/dtolnay/anyhow/blob/1.0.26/src/lib.rs#L543)
pub trait IoContext<T> {
    fn ctx<C>(self, ctx: C) -> Result<T>
    where
        C: Display + Send + Sync + 'static;

    fn with_ctx<C, F>(self, f: F) -> Result<T>
    where
        C: Display + Send + Sync + 'static,
        F: FnOnce() -> C;

    fn table_ctx<C>(self, file_id: u64, ctx: C) -> table::Result<T>
    where
        C: Display + Send + Sync + 'static;

    fn with_table_ctx<C, F>(self, f: F) -> table::Result<T>
    where
        C: Display + Send + Sync + 'static,
        F: FnOnce() -> (u64, C);

    fn dfs_ctx<C>(self, file_id: u64, ctx: C) -> dfs::Result<T>
    where
        C: Display + Send + Sync + 'static;
}

impl<T> IoContext<T> for io::Result<T> {
    fn ctx<C>(self, ctx: C) -> Result<T>
    where
        C: Display + Send + Sync + 'static,
    {
        self.map_err(|err| Error::Io {
            err,
            ctx: format!("{ctx}"),
        })
    }

    fn with_ctx<C, F>(self, ctx_fn: F) -> Result<T>
    where
        C: Display + Send + Sync + 'static,
        F: FnOnce() -> C,
    {
        self.map_err(|err| Error::Io {
            err,
            ctx: format!("{}", ctx_fn()),
        })
    }

    fn table_ctx<C>(self, file_id: u64, ctx: C) -> table::Result<T>
    where
        C: Display + Send + Sync + 'static,
    {
        self.map_err(|err| table::Error::Io(table_ctx_to_str(file_id, ctx, err)))
    }

    fn with_table_ctx<C, F>(self, ctx_fn: F) -> table::Result<T>
    where
        C: Display + Send + Sync + 'static,
        F: FnOnce() -> (u64, C),
    {
        self.map_err(|err| {
            let (file_id, ctx) = ctx_fn();
            table::Error::Io(table_ctx_to_str(file_id, ctx, err))
        })
    }

    fn dfs_ctx<C>(self, file_id: u64, ctx: C) -> dfs::Result<T>
    where
        C: Display + Send + Sync + 'static,
    {
        self.map_err(|err| dfs::Error::Io(table_ctx_to_str(file_id, ctx, err)))
    }
}

#[inline]
fn table_ctx_to_str<C, E>(file_id: u64, ctx: C, err: E) -> String
where
    C: Display + Send + Sync + 'static,
    E: Debug,
{
    format!("{ctx}:{file_id}: {err:?}")
}
