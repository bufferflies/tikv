// Copyright 2025 TiKV Project Authors. Licensed under Apache-2.0.

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("cdc error {0}")]
    CdcError(#[from] cdc::Error),
    #[error("nix error {0}")]
    NixError(#[from] nix::Error),
    #[error("native_br error {0}")]
    BrError(#[from] native_br::error::Error),
    #[error("merged engine error {0}")]
    MergedEngineError(#[from] merged_engine::Error),
    #[error("kube error {0}")]
    KubeError(#[from] kube::Error),
    #[error("pd client error {0}")]
    PdClientError(#[from] pd_client::Error),
    #[error("hyper error {0}")]
    HyperError(#[from] hyper::Error),
    #[error("rfengine error {0}")]
    RfEngineError(#[from] rfengine::Error),
    #[error("store timeout {0}")]
    StoreTimeout(String),
    #[error("other error {0}")]
    OtherError(String),
}

pub type Result<T> = std::result::Result<T, Error>;
