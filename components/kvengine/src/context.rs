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
