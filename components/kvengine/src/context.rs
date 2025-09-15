// Copyright 2024 TiKV Project Authors. Licensed under Apache-2.0.

use std::{path::PathBuf, sync::Arc};

use cloud_encryption::MasterKey;
use dashmap::DashMap;

use crate::{
    dfs,
    ia::manager::IaManager,
    table::{columnar::SchemaFile, sstable::BlockCache},
    txn_chunk_manager::TxnChunkManager,
};

#[derive(Clone)]
pub struct SnapCtx {
    pub dfs: Arc<dyn dfs::Dfs>,
    pub master_key: MasterKey,
    pub block_cache: BlockCache,
    pub schema_files: Option<Arc<DashMap<u64, SchemaFile>>>,
    pub txn_chunk_manager: TxnChunkManager,
    pub ia_ctx: IaCtx,
    pub prepare_type: PrepareType,
    pub read_columnar: bool,
    pub encryption_key_manager: Arc<cloud_encryption::EncryptionKeyManager>,
}

#[derive(Clone, Copy, PartialEq)]
pub enum PrepareType {
    SstOnly,      // For sst reader.
    ColumnarOnly, // For columnar reader.
    All,
}

#[derive(Clone)]
pub enum IaCtx {
    Disabled,
    Enabled(IaManager, Arc<PathBuf>),
}

impl IaCtx {
    pub fn is_enabled(&self) -> bool {
        matches!(self, Self::Enabled(..))
    }
}
