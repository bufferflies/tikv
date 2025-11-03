// Copyright 2023 TiKV Project Authors. Licensed under Apache-2.0.

use kvengine::dfs;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("check {0}")]
    CheckError(String),
    #[error("pd error {0}")]
    PdError(#[from] pd_client::Error),
    #[error("serde_json error {0}")]
    SerdeJsonError(#[from] serde_json::Error),
    #[error("dfs error {0}")]
    DfsError(#[from] dfs::Error),
    #[error("hyper error {0}")]
    HyperError(#[from] hyper::Error),
    #[error("http error {0}")]
    HttpError(#[from] http::Error),
    #[error("io error {0}")]
    Io(#[from] std::io::Error),
    #[error("io error {0}")]
    IoError(#[from] tikv_util::errors::IoError),
    #[error("native backup/restore error {0}")]
    NativeBackupRestoreError(#[from] native_br::error::Error),
    #[error("restore keyspace task conflict with id {0}")]
    RestoreKeyspaceTaskConflict(u64),
    #[error("datetime parse error {0}")]
    DateTimeParseError(#[from] chrono::ParseError),
    #[error("reach concurrency limit (current {current}, limit {limit})")]
    ReachConcurrencyLimit { current: usize, limit: usize },
    #[error("parse int error {0}")]
    ParseIntError(#[from] std::num::ParseIntError),
    #[error("table format error")]
    TableFormatError(#[from] kvengine::table::Error),
    #[error("schema error {0}")]
    SchemaError(String),
    #[error("file corrupted")]
    FileCorrupted,
    #[error("k8s error {0}")]
    K8sError(String),
    #[error("Other error {0}")]
    Other(#[from] Box<dyn std::error::Error + Send + Sync>),
}

pub type Result<T> = std::result::Result<T, Error>;
