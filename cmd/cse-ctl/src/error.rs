// Copyright 2022 TiKV Project Authors. Licensed under Apache-2.0.

use kvengine::dfs;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Cluster topology error {0}")]
    TopoChanged(String),
    #[error("Backup meta of cluster {0} is not found")]
    MetaNotFound(u64),
    #[error("DFS error {0}")]
    DFSError(dfs::Error),
    #[error("Server error {0}")]
    ServerError(String),
    #[error("Safe ts {0} is greater than backup ts {1}")]
    TsError(u64, u64),
    #[error("PD error {0}")]
    PDError(pd_client::Error),
    #[error("Etcd error {0}")]
    EtcdError(etcd_client::Error),
    #[error("Timeout {0}s")]
    Timeout(u64),
    #[error("TiKV error {0}")]
    TiKVError(tikv_client::Error),
}

impl From<dfs::Error> for Error {
    fn from(e: dfs::Error) -> Self {
        Error::DFSError(e)
    }
}

impl From<pd_client::Error> for Error {
    fn from(e: pd_client::Error) -> Self {
        Error::PDError(e)
    }
}

impl From<etcd_client::Error> for Error {
    fn from(e: etcd_client::Error) -> Self {
        Error::EtcdError(e)
    }
}

impl From<tikv_client::Error> for Error {
    fn from(e: tikv_client::Error) -> Self {
        Error::TiKVError(e)
    }
}
