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
    #[error("resource {resource} not found")]
    NotFound {
        resource: String,
        identity: Option<String>,
        notes: Option<String>,
    },
    #[error("resource {0} already exists")]
    Existed(String),
    #[error("state trans {from:?} => {to:?} isn't avaliable in resource {resource:?}")]
    InvalidStateTrans {
        resource: String,
        from: String,
        to: String,
    },
    #[error("datetime parse error {0}")]
    DateTimeParseError(#[from] chrono::ParseError),
    #[error("ReachLimit {0}")]
    ReachConcurrencyLimit(usize),
    #[error("parse int error {0}")]
    ParseIntError(#[from] std::num::ParseIntError),
    #[error("table format error")]
    TableFormatError(#[from] kvengine::table::Error),
    #[error("schema error {0}")]
    SchemaError(String),
    #[error("file corrupted")]
    FileCorrupted,
    #[error("Other error {0}")]
    Other(#[from] Box<dyn std::error::Error + Send + Sync>),
}

pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    pub fn identitied_not_found(resource: impl ToString, identity: impl ToString) -> Self {
        Self::not_found(resource, Some(identity), None::<u8>)
    }

    pub fn noted_not_found(resource: impl ToString, notes: impl ToString) -> Self {
        Self::not_found(resource, None::<u8>, Some(notes))
    }

    pub fn not_found(
        resourece: impl ToString,
        identity: Option<impl ToString>,
        notes: Option<impl ToString>,
    ) -> Self {
        Self::NotFound {
            resource: resourece.to_string(),
            identity: identity.map(|s| s.to_string()),
            notes: notes.map(|s| s.to_string()),
        }
    }
}
