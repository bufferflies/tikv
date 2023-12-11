// Copyright 2023 TiKV Project Authors. Licensed under Apache-2.0.

use kvengine::dfs;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("check {0}")]
    CheckError(String),
    #[error("handle reader {0}")]
    HandleReaderError(String),
    #[error("canceled")]
    Canceled,
    #[error("pd error {0}")]
    PdError(#[from] pd_client::Error),
    #[error("ingest files {0}")]
    IngestFiles(String),
    #[error("dfs error {0}")]
    DfsError(#[from] dfs::Error),
    #[error("hyper error {0}")]
    HyperError(#[from] hyper::Error),
    #[error("http error {0}")]
    HttpError(#[from] http::Error),
    #[error("io error {0}")]
    IoError(#[from] std::io::Error),
    #[error("protobuf error {0}")]
    ProtobufError(#[from] protobuf::ProtobufError),
    #[error("region {0} not found")]
    RegionNotFound(u64),
    #[error("leader of region {0} not found")]
    LeaderNotFound(u64),
    #[error("region {0} error {1:?}")]
    RegionError(u64, kvproto::errorpb::Error),
    #[error("too many duplicated keys {0}")]
    TooManyDuplicatedKeys(String),
    #[error("reach limit {0}")]
    ReachConcurrencyLimit(usize),
    #[error("regions are not intact in range: {0}")]
    RegionsIntegrityError(String),
    #[error("Ingest is overlapped with region existed data: {0}")]
    IngestOverlap(String),
    #[error("Multiply errors: {0:?}")]
    MultiErrors(Vec<Error>),
    #[error("other {0}")]
    Other(#[from] Box<dyn std::error::Error + Send + Sync>),
}

pub type Result<T> = std::result::Result<T, Error>;
