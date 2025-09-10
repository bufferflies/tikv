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
    #[error("PD control error {0}")]
    PdCtlError(#[from] pd_client::pd_control::Error),
    #[error("Split regions error {0}")]
    SpitRegionsError(pd_client::Error),
    #[error("Etcd error {0}")]
    EtcdError(etcd_client::Error),
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
    #[error("TiKV store disk full {0:?}")]
    StoreDiskFull(Vec<u64> /* stores id */),
    #[error(transparent)]
    HttpRequestError(#[from] HttpRequestError),
    #[error("HTTP error {0}:{1}")]
    HttpError(http::StatusCode, String),
    #[error("HTTP error {0}:{1:?}")]
    HttpPbError(http::StatusCode, kvproto::errorpb::Error),
    #[error("Security client error {0}")]
    SecurityClientError(#[from] security::HttpClientError),
    #[error("Retry limit exceeded, last error {0}")]
    RetryLimitExceeded(Box<Error>),
    #[error("Restore other keyspace from/to default keyspace")]
    RestoreWithDefaultKeyspace,
    #[error("Restore snapshot error")]
    RestoreSnapshot,
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
    #[error("Backup error on stores {0:?}")]
    BackupErrorOnStores(Vec<Error>, Vec<kvproto::metapb::Store>),
    #[error("No snapshot available error {0}")]
    NoSnapshotAvailableError(String),
    #[error("WAL chunk integrity error {0}")]
    WalChunkIntegrityError(String),
    // IncrementalBackupToleratedError means that we are performing an incremental backup with a
    // last backup having tolerated error of one store, but we meet the error of another store.
    #[error("Incremental backup tolerated error for store {0}")]
    IncrementalBackupToleratedError(u64 /* store id */),
    #[error("Fetch RfEngine WAL chunk HTTP request error {0}")]
    RfengineHttpRequestError(HttpRequestError),
    #[error("Fetch RfEngine WAL chunk service error {0}")]
    RfengineHttpSvrError(String),
    #[error("RfEngine DFS worker unhealthy {0}")]
    RfengineDfsWorkerUnhealthy(String),
    #[error("Fetch RfEngine WAL chunk error due to epoch {epoch_id} overwritten")]
    RfengineWalEpochOverwritten { epoch_id: u32 },
}

#[derive(Debug, thiserror::Error)]
pub enum HttpRequestError {
    #[error("{0} timeout({1:?})")]
    Timeout(String /* message */, std::time::Duration),
    #[error("HTTP request error {0}: {1}")]
    Http(String /* uri */, hyper::Error),
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
