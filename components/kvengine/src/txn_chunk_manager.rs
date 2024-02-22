// Copyright 2024 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    default::Default,
    fs,
    ops::Deref,
    path::PathBuf,
    sync::{Arc, RwLock},
};

use bytes::{Buf, Bytes};
use dashmap::DashMap;
use moka::sync::SegmentedCache;
use regex::Regex;
use tikv_util::HandyRwLock;

use crate::{
    dfs::Dfs,
    table::{
        sstable::{BlockCacheKey, LocalFile},
        txn_file::TxnChunk,
    },
    Result,
};

#[derive(Clone)]
pub struct TxnChunkManager {
    core: Arc<TxnChunkManagerCore>,
}

impl TxnChunkManager {
    pub fn new(
        local_path: PathBuf,
        dfs: Arc<dyn Dfs>,
        cache: SegmentedCache<BlockCacheKey, Bytes>,
    ) -> Self {
        let manager = Self {
            core: Arc::new(TxnChunkManagerCore {
                local_path,
                dfs,
                txn_chunks: DashMap::new(),
                cache,
            }),
        };
        manager.init().unwrap();
        manager
    }
}

impl Deref for TxnChunkManager {
    type Target = TxnChunkManagerCore;

    fn deref(&self) -> &Self::Target {
        &self.core
    }
}

pub struct TxnChunkManagerCore {
    local_path: PathBuf,
    dfs: Arc<dyn Dfs>,
    txn_chunks: DashMap<u64, Arc<TxnChunkEntry>>,
    cache: SegmentedCache<BlockCacheKey, Bytes>,
}

struct TxnChunkEntry {
    chunk_data: RwLock<Option<TxnChunk>>,
}

impl Default for TxnChunkEntry {
    fn default() -> Self {
        Self {
            chunk_data: RwLock::new(None),
        }
    }
}

impl TxnChunkManagerCore {
    fn init(&self) -> Result<()> {
        if !self.local_path.exists() {
            fs::create_dir_all(&self.local_path)?;
        }
        if self.local_path.is_dir() {
            let read_dir = fs::read_dir(&self.local_path)?;
            for entry in read_dir.flatten() {
                let file_name = entry.file_name();
                if let Some(txn_file_id) = parse_txn_chunk_id(file_name.to_str().unwrap()) {
                    let txn_chunk = self.load_txn_chunk(txn_file_id, entry.path())?;
                    let files_entry = self.txn_chunks.entry(txn_file_id).or_default().clone();
                    let mut guard = files_entry.chunk_data.wl();
                    *guard = Some(txn_chunk)
                }
            }
        }
        Ok(())
    }

    pub fn prepare(&self, txn_chunk_id: u64) -> Result<()> {
        let entry = self.txn_chunks.entry(txn_chunk_id).or_default().clone();
        let mut guard = entry.chunk_data.write().unwrap();
        if guard.is_some() {
            return Ok(());
        }
        let local_file_path = self.local_file_path(txn_chunk_id);
        if !local_file_path.exists() {
            let file_name = txn_chunk_id.to_string();
            let runtime = self.dfs.get_runtime();
            let file_data = runtime.block_on(self.dfs.read_txn_chunk(txn_chunk_id))?;
            let txn_file_tmp_path = self.local_path.join(format!("{}.tmp", file_name));
            fs::write(&txn_file_tmp_path, file_data.chunk())?;
            let local_file_path = self.local_file_path(txn_chunk_id);
            fs::rename(&txn_file_tmp_path, local_file_path)?;
        }
        let txn_chunk = self.load_txn_chunk(txn_chunk_id, local_file_path)?;
        *guard = Some(txn_chunk);
        Ok(())
    }

    fn load_txn_chunk(&self, txn_chunk_id: u64, path: PathBuf) -> Result<TxnChunk> {
        let file = LocalFile::open(txn_chunk_id, path.as_path(), false)?;
        let txn_chunk = TxnChunk::new(Arc::new(file), Some(self.cache.clone()))?;
        Ok(txn_chunk)
    }

    pub fn all_chunks_exists(&self, chunk_ids: &[u64]) -> bool {
        for chunk_id in chunk_ids {
            if let Some(entry) = self.txn_chunks.get(chunk_id) {
                if let Ok(guard) = entry.value().chunk_data.try_read() {
                    if guard.is_none() {
                        // load chunk failed.
                        return false;
                    }
                } else {
                    // The chunk is loading by another thread.
                    return false;
                }
            } else {
                return false;
            }
        }
        true
    }

    pub fn get(&self, txn_chunk_id: u64) -> Option<TxnChunk> {
        let entry = self.txn_chunks.get(&txn_chunk_id)?.clone();
        let guard = entry.chunk_data.rl();
        guard.clone()
    }

    pub fn remove(&self, txn_chunk_id: u64) -> bool {
        let _ = fs::remove_file(self.local_file_path(txn_chunk_id));
        self.txn_chunks.remove(&txn_chunk_id).is_some()
    }

    fn local_file_path(&self, txn_chunk_id: u64) -> PathBuf {
        self.local_path.join(format!("{:016x}.txn", txn_chunk_id))
    }
}

pub fn parse_txn_chunk_id(txn_chunk_name: &str) -> Option<u64> {
    lazy_static::lazy_static! {
        static ref RE: Regex = Regex::new(r"([0-9a-fA-F]{16})\.txn$").unwrap();
    }
    let caps = RE.captures(txn_chunk_name)?;
    let chunk_id = u64::from_str_radix(&caps[1], 16).unwrap();
    Some(chunk_id)
}

#[cfg(test)]
mod tests {
    use tempfile::TempDir;

    use super::*;
    use crate::{
        dfs::InMemFs,
        table::{TxnChunkBuilder, OP_PUT},
        BLOCK_CACHE_KEY_SIZE,
    };

    #[test]
    fn test_txn_chunk_manager() {
        let tmp_dir = TempDir::new().unwrap();
        let dfs: Arc<dyn Dfs> = Arc::new(InMemFs::new());
        let cache: SegmentedCache<BlockCacheKey, Bytes> = SegmentedCache::builder(256)
            .weigher(|_k: &BlockCacheKey, v: &Bytes| (BLOCK_CACHE_KEY_SIZE + v.len()) as u32)
            .max_capacity(1024 * 1024u64)
            .build();
        let txn_chunk_manager =
            TxnChunkManager::new(tmp_dir.path().to_path_buf(), dfs.clone(), cache.clone());
        let runtime = dfs.get_runtime();
        for chunk_id in 1u64..=3 {
            let mut chunk_builder = TxnChunkBuilder::new(10);
            for i in 0..100 {
                let key = format!("{:02}/{:02}", chunk_id, i);
                chunk_builder.add_entry(key.as_bytes(), OP_PUT, key.as_bytes());
            }
            let mut buf = vec![];
            chunk_builder.finish(&mut buf);
            runtime
                .block_on(dfs.create_txn_chunk(chunk_id, buf.into()))
                .unwrap();
        }
        for chunk_id in 1u64..=3 {
            txn_chunk_manager.prepare(chunk_id).unwrap();
            assert!(txn_chunk_manager.get(chunk_id).is_some());
        }
        txn_chunk_manager.remove(1);
        assert!(txn_chunk_manager.get(1).is_none());
        drop(txn_chunk_manager);
        // After process restart, the remained txn chunks are all loaded.
        let txn_chunk_manager = TxnChunkManager::new(tmp_dir.path().to_path_buf(), dfs, cache);
        assert!(txn_chunk_manager.get(1).is_none());
        assert!(txn_chunk_manager.all_chunks_exists(&[2, 3]));
    }
}
