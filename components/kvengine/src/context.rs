// Copyright 2024 TiKV Project Authors. Licensed under Apache-2.0.

use std::{path::PathBuf, sync::Arc};

use bytes::Bytes;
use cloud_encryption::MasterKey;
use dashmap::DashMap;
use quick_cache::sync::Cache;

use crate::{
    dfs,
    ia::manager::IaManager,
    table::{schema_file::SchemaFile, sstable::BlockCache, vector_index::VectorIndexCache},
    txn_chunk_manager::TxnChunkManager,
};

#[derive(Clone)]
pub struct SnapCtx {
    pub dfs: Arc<dyn dfs::Dfs>,
    pub master_key: MasterKey,
    pub block_cache: BlockCache,
    pub vector_index_cache: Option<VectorIndexCache>,
    pub schema_files: Option<Arc<DashMap<u64, SchemaFile>>>,
    pub txn_chunk_manager: TxnChunkManager,
    pub ia_ctx: IaCtx,
    pub prepare_type: PrepareType,
    pub read_columnar: bool,
    pub meta_file_cache: Arc<Cache<u64, Bytes, MetaFileCacheWeighter>>,
}

const ESTIMATED_META_FILE_SIZE: u64 = 64 * 1024; // 64KB

pub fn new_meta_file_cache(capacity: u64) -> Arc<Cache<u64, Bytes, MetaFileCacheWeighter>> {
    let estimated_items = (capacity / ESTIMATED_META_FILE_SIZE).max(64);
    Arc::new(Cache::with_weighter(
        estimated_items as usize,
        capacity,
        MetaFileCacheWeighter {},
    ))
}

#[derive(Clone)]
pub struct MetaFileCacheWeighter;

impl quick_cache::Weighter<u64, Bytes> for MetaFileCacheWeighter {
    fn weight(&self, _: &u64, value: &Bytes) -> u64 {
        value.len() as u64 + 8 // 8 bytes for the key
    }
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
