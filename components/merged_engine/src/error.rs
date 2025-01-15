// Copyright 2025 TiKV Project Authors. Licensed under Apache-2.0.

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Other error {0}")]
    Other(#[from] Box<dyn std::error::Error + Send + Sync>),
    #[error("KvEngine error {0}")]
    KvEngine(#[from] kvengine::Error),
    #[error("RfEngine error {0}")]
    RfEngine(#[from] rfengine::Error),
    #[error("Io error {0}")]
    IoError(#[from] std::io::Error),
    #[error("DFS error {0}")]
    DfsError(#[from] kvengine::dfs::Error),
    #[error("Br error {0}")]
    BrError(#[from] native_br::error::Error),
}

pub type Result<T> = std::result::Result<T, crate::Error>;
