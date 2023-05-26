// Copyright 2023 TiKV Project Authors. Licensed under Apache-2.0.

use kvengine::dfs;

#[derive(Debug, thiserror::Error)]
pub(crate) enum Error {
    #[error("check {0}")]
    CheckError(String),
    #[error("canceled")]
    Canceled,
    #[error("pd error {0}")]
    PdError(pd_client::Error),
    #[error("ingest files {0}")]
    IngestFiles(String),
    #[error("dfs error {0}")]
    DfsError(dfs::Error),
    #[error("hyper error {0}")]
    HyperError(hyper::Error),
    #[error("http error {0}")]
    HttpError(http::Error),
    #[error("io error {0}")]
    IoError(std::io::Error),
    #[error("region {0} not found")]
    RegionNotFound(u64),
    #[error("leader of region {0} not found")]
    LeaderNotFound(u64),
    #[error("duplicated key {0}")]
    DuplicatedKey(String),
    #[error("native backup/restore error {0}")]
    NativeBackupRestoreError(#[from] native_br::error::Error),
    #[error("restore keyspace task conflict with id {0}")]
    RestoreKeyspaceTaskConflict(u64),
    #[error("datetime parse error {0}")]
    DateTimeParseError(#[from] chrono::ParseError),
    #[error("ReachLimit {0}")]
    ReachConcurrencyLimit(usize),
    #[error("Other {0}")]
    Other(String),
}

impl From<dfs::Error> for Error {
    fn from(e: dfs::Error) -> Self {
        Error::DfsError(e)
    }
}

impl From<pd_client::Error> for Error {
    fn from(e: pd_client::Error) -> Self {
        Error::PdError(e)
    }
}

impl From<hyper::Error> for Error {
    fn from(e: hyper::Error) -> Self {
        Error::HyperError(e)
    }
}

impl From<http::Error> for Error {
    fn from(e: http::Error) -> Self {
        Error::HttpError(e)
    }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::IoError(e)
    }
}
