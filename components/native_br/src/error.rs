// Copyright 2022 TiKV Project Authors. Licensed under Apache-2.0.

use std::sync::Arc;

use kvengine::dfs;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Clone, thiserror::Error)]
#[error(transparent)]
pub struct SharedError(pub Arc<Error>);

impl From<Error> for SharedError {
    fn from(e: Error) -> Self {
        Self(Arc::new(e))
    }
}

impl SharedError {
    pub fn inner(&self) -> &Error {
        &self.0
    }
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Other error {0}")]
    Other(#[from] Box<dyn std::error::Error + Send + Sync>),
    #[error("Cluster topology error {0}")]
    TopoChanged(String),
    #[error("Backup meta of cluster {0} is not found")]
    MetaNotFound(u64),
    #[error("DFS error {0}")]
    DfsError(dfs::Error),
    #[error("Server error {0}")]
    ServerError(String),
    #[error("Safe ts {0} is greater than backup ts {1}")]
    TsError(u64, u64),
    #[error("PiTR ts {0} is out of safe ts {1} and backup ts {2}")]
    PitrTsError(u64, u64, u64),
    #[error("PD error {0}")]
    PdError(pd_client::Error),
    #[error("Etcd error {0}")]
    EtcdError(etcd_client::Error),
    #[error("{0} timeout {0}s")]
    Timeout(String, u64),
    #[error("TiKV error {0}")]
    TikvError(tikv_client::Error),
    #[error("KvEngine error {0}")]
    KvEngine(kvengine::Error),
    #[error("RfEngine error {0}")]
    RfEngine(rfengine::Error),
    #[error("Region version not match expected:{} actual:{}", .expected, .actual)]
    RegionVerNotMatch { expected: u64, actual: u64 },
    #[error("Region {0} not found or no leader")]
    RegionNotFoundOrNoLeader(u64 /* region id */),
    #[error("HTTP error {0}")]
    HttpError(#[from] hyper::Error),
    #[error("Retry limit exceeded, last error {0}")]
    RetryLimitExceeded(Box<Error>),
    #[error("Keyspace {0} inner_key_off not enabled")]
    KeyspaceInnerKeyOffNotEnabled(u32 /* region id */),
    #[error("Backup for keyspace {0} is empty")]
    BackupEmptyForKeyspace(u32 /* keyspace id */),
    #[error("Reach concurrency limit {0}")]
    ReachConcurrencyLimit(usize),
    #[error(transparent)]
    SharedError(#[from] SharedError),
    #[error("Archive error {0}")]
    ArchiveError(String),
    #[error("Mvcc error {0}")]
    MvccError(#[from] tikv::storage::mvcc::Error),
    #[error("Backup error {0}")]
    BackupError(String),
    #[error("Wal chunk integrity error {0}")]
    WalChunkIntegrityError(String),
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

impl From<etcd_client::Error> for Error {
    fn from(e: etcd_client::Error) -> Self {
        Error::EtcdError(e)
    }
}

impl From<tikv_client::Error> for Error {
    fn from(e: tikv_client::Error) -> Self {
        Error::TikvError(e)
    }
}

impl From<kvengine::Error> for Error {
    fn from(e: kvengine::Error) -> Self {
        Error::KvEngine(e)
    }
}

impl From<rfengine::Error> for Error {
    fn from(e: rfengine::Error) -> Self {
        Error::RfEngine(e)
    }
}
