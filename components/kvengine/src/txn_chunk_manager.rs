// Copyright 2024 TiKV Project Authors. Licensed under Apache-2.0.

use std::{default::Default, fs, ops::Deref, path::PathBuf, sync::Arc};

use bytes::{Buf, Bytes};
use cloud_encryption::EncryptionKey;
use dashmap::DashMap;
use futures::executor::block_on;
use moka::sync::SegmentedCache;
use regex::Regex;
use tikv_util::box_err;
use tokio::sync::{OwnedRwLockWriteGuard, RwLock};

use crate::{
    dfs,
    dfs::{Dfs, FileType},
    error::IoContext,
    table,
    table::{
        file::{InMemFile, LocalFile},
        sstable::BlockCacheKey,
        txn_file::TxnChunk,
        TxnCtx, TxnFile, TxnFileId,
    },
    Error, Result,
};

const READ_DFS_CONCURRENCY: usize = 4;

#[derive(Clone)]
pub struct TxnChunkManager {
    core: Arc<TxnChunkManagerCore>,
}

impl TxnChunkManager {
    pub fn new(
        local_path: Option<PathBuf>,
        dfs: Arc<dyn Dfs>,
        cache: Option<SegmentedCache<BlockCacheKey, Bytes>>,
        worker_pool: WorkerPool,
    ) -> Self {
        info!("create txn chunk manager"; "worker_pool" => ?worker_pool);
        let manager = Self {
            core: Arc::new(TxnChunkManagerCore {
                local_path,
                dfs,
                txn_chunks: DashMap::new(),
                cache,
                worker_pool,
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

// If `TxnChunkManager` will be hold in an async context, create worker pool
// outside and pass the handle to create it. Otherwise, it will panic when the
// runtime is dropped in the async context.
#[derive(Debug)]
pub enum WorkerPool {
    Pool(tokio::runtime::Runtime),
    Handle(tokio::runtime::Handle),
}

impl WorkerPool {
    pub fn handle(&self) -> &tokio::runtime::Handle {
        match self {
            WorkerPool::Pool(pool) => pool.handle(),
            WorkerPool::Handle(handle) => handle,
        }
    }
}

pub fn with_pool_size(pool_size: usize) -> WorkerPool {
    WorkerPool::Pool(
        tokio::runtime::Builder::new_multi_thread()
            .thread_name("txn-chunk-worker")
            .worker_threads(1) // currently not used.
            .max_blocking_threads(pool_size)
            .enable_all()
            .build()
            .unwrap(),
    )
}

pub fn with_pool_handle(handle: tokio::runtime::Handle) -> WorkerPool {
    WorkerPool::Handle(handle)
}

// Memory based if `local_path` is None.
// TODO: unify the process of file-based & memory-based to eliminate duplicated
// codes.
pub struct TxnChunkManagerCore {
    local_path: Option<PathBuf>,
    dfs: Arc<dyn Dfs>,
    txn_chunks: DashMap<u64, TxnChunkEntry>,
    cache: Option<SegmentedCache<BlockCacheKey, Bytes>>,
    worker_pool: WorkerPool,
}

#[derive(Clone)]
struct TxnChunkEntry {
    chunk_data: Arc<RwLock<Option<TxnChunk>>>,
}

impl Default for TxnChunkEntry {
    fn default() -> Self {
        Self {
            chunk_data: Arc::new(RwLock::new(None)),
        }
    }
}

impl TxnChunkManagerCore {
    fn init(&self) -> Result<()> {
        if let Some(local_path) = self.local_path.as_ref() {
            if !local_path.exists() {
                fs::create_dir_all(local_path).ctx("txn_chunk_mgr.init.create_dir")?;
            }
            if local_path.is_dir() {
                let read_dir = fs::read_dir(local_path).ctx("txn_chunk_mgr.init.read_dir")?;
                for entry in read_dir.flatten() {
                    let file_name = entry.file_name();
                    if let Some(txn_file_id) = parse_txn_chunk_id(file_name.to_str().unwrap()) {
                        // We don't have the encryption key here, we can do nothing but skip the
                        // encrypted files.
                        // As the number of encrypted files is expected to be small, this should be
                        // fine.
                        let txn_chunk = match self.load_txn_chunk(txn_file_id, entry.path(), None) {
                            Ok(txn_chunk) => txn_chunk,
                            Err(Error::TableError(table::Error::NeedEncryptionKey { .. })) => {
                                continue;
                            }
                            Err(err) => return Err(err),
                        };
                        let files_entry = self.txn_chunks.entry(txn_file_id).or_default().clone();
                        let mut guard = block_on(files_entry.chunk_data.write());
                        *guard = Some(txn_chunk)
                    }
                }
            }
        }
        Ok(())
    }

    pub fn worker_pool(&self) -> &tokio::runtime::Handle {
        self.worker_pool.handle()
    }

    pub fn prepare(
        &self,
        txn_chunk_id: u64,
        encryption_key: Option<EncryptionKey>,
    ) -> Result<TxnChunk> {
        let entry = self.txn_chunks.entry(txn_chunk_id).or_default().clone();
        let mut guard = block_on(entry.chunk_data.write());
        if let Some(txn_chunk) = guard.as_ref() {
            return Ok(txn_chunk.clone());
        }

        if let Some(local_file_path) = self.local_file_path(txn_chunk_id) {
            if local_file_path.exists() {
                let txn_chunk =
                    self.load_txn_chunk(txn_chunk_id, local_file_path, encryption_key)?;
                *guard = Some(txn_chunk.clone());
                return Ok(txn_chunk);
            }
        }

        let runtime = self.dfs.get_runtime();
        let opts = dfs::Options::default().with_type(FileType::TxnChunk);
        let file_data = runtime.block_on(self.dfs.read_file(txn_chunk_id, opts))?;

        let txn_chunk = if let Some(local_path) = self.local_path.as_ref() {
            let file_name = txn_chunk_id.to_string();
            let txn_file_tmp_path = local_path.join(format!("{}.tmp", file_name));
            fs::write(&txn_file_tmp_path, file_data.chunk())
                .table_ctx(txn_chunk_id, "txn_chunk_mgr.prepare.write_tmp")?;
            let local_file_path = self.local_file_path(txn_chunk_id).unwrap();
            fs::rename(&txn_file_tmp_path, &local_file_path)
                .table_ctx(txn_chunk_id, "txn_chunk.prepare.rename")?;
            self.load_txn_chunk(txn_chunk_id, local_file_path, encryption_key)?
        } else {
            let file = InMemFile::new(txn_chunk_id, file_data);
            TxnChunk::new(Arc::new(file), self.cache.clone(), encryption_key)?
        };
        *guard = Some(txn_chunk.clone());
        Ok(txn_chunk)
    }

    pub fn prepare_txn_chunks(
        &self,
        mut txn_chunks_id: Vec<u64>,
        encryption_key: Option<EncryptionKey>,
    ) -> Result<()> {
        // Sort to avoid deadlocks.
        txn_chunks_id.sort();
        info!("prepare txn chunks: {:?}", txn_chunks_id);

        let (tx, rx) = tikv_util::mpsc::bounded(READ_DFS_CONCURRENCY);
        let runtime = self.dfs.get_runtime();
        let mut msg_count: usize = 0;
        for chunk_id in txn_chunks_id {
            let entry = self.txn_chunks.entry(chunk_id).or_default().clone();
            let mut guard = block_on(entry.chunk_data.write_owned());
            if guard.is_some() {
                continue;
            }

            if let Some(local_file_path) = self.local_file_path(chunk_id) {
                if local_file_path.exists() {
                    let txn_chunk =
                        self.load_txn_chunk(chunk_id, local_file_path, encryption_key.clone())?;
                    *guard = Some(txn_chunk);
                    continue;
                }
            }

            let dfs = self.dfs.clone();
            let tx = tx.clone();
            runtime.spawn(async move {
                let opts = dfs::Options::default().with_type(FileType::TxnChunk);
                let file_data = dfs.read_file(chunk_id, opts).await;
                if let Err(err) = tx.send((chunk_id, file_data, guard)) {
                    // Error should happen only when prepare_txn_chunks exit with error.
                    warn!("prepare_txn_chunks: send error: {:?}", err; "chunk_id" => chunk_id);
                }
            });
            msg_count += 1;

            if msg_count >= READ_DFS_CONCURRENCY {
                self.recv_txn_chunk_file_data(&rx, encryption_key.clone())?;
                msg_count -= 1;
            }
        }
        for _ in 0..msg_count {
            self.recv_txn_chunk_file_data(&rx, encryption_key.clone())?;
        }
        Ok(())
    }

    fn recv_txn_chunk_file_data(
        &self,
        rx: &tikv_util::mpsc::Receiver<(
            u64,
            crate::dfs::Result<Bytes>,
            OwnedRwLockWriteGuard<Option<TxnChunk>>,
        )>,
        encryption_key: Option<EncryptionKey>,
    ) -> Result<()> {
        let (chunk_id, file_data, mut guard) = rx.recv().unwrap();
        let file_data = file_data.map_err(|err| -> Error {
            box_err!("read_txn_chunk failed: {:?}, chunk_id {}", err, chunk_id)
        })?;
        let txn_chunk = if let Some(local_path) = self.local_path.as_ref() {
            let local_file_path = self.local_file_path(chunk_id).unwrap();
            if !local_file_path.exists() {
                let file_name = chunk_id.to_string();
                let txn_file_tmp_path = local_path.join(format!("{}.tmp", file_name));
                fs::write(&txn_file_tmp_path, file_data.chunk())
                    .table_ctx(chunk_id, "txn_chunk_mgr.recv_chunk.write_tmp")?;
                fs::rename(&txn_file_tmp_path, &local_file_path)
                    .table_ctx(chunk_id, "txn_chunk_mgr.recv_chunk.rename")?;
            }
            self.load_txn_chunk(chunk_id, local_file_path, encryption_key)?
        } else {
            let file = InMemFile::new(chunk_id, file_data);
            TxnChunk::new(Arc::new(file), self.cache.clone(), encryption_key)?
        };
        *guard = Some(txn_chunk);
        Ok(())
    }

    fn load_txn_chunk(
        &self,
        txn_chunk_id: u64,
        path: PathBuf,
        encryption_key: Option<EncryptionKey>,
    ) -> Result<TxnChunk> {
        let file = LocalFile::open(txn_chunk_id, path.as_path(), false)?;
        let txn_chunk = TxnChunk::new(Arc::new(file), self.cache.clone(), encryption_key)?;
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
        let guard = block_on(entry.chunk_data.read());
        guard.clone()
    }

    pub fn remove(&self, txn_chunk_id: u64) -> bool {
        if let Some(local_file_path) = self.local_file_path(txn_chunk_id) {
            let _ = fs::remove_file(local_file_path);
        }
        self.txn_chunks.remove(&txn_chunk_id).is_some()
    }

    fn local_file_path(&self, txn_chunk_id: u64) -> Option<PathBuf> {
        Some(
            self.local_path
                .as_ref()?
                .join(format!("{:016x}.txn", txn_chunk_id)),
        )
    }

    // `encryption_key` is required only when `is_prepared` is false.
    pub fn load_txn_file_from_ref(
        &self,
        shard_id: u64,
        shard_ver: u64,
        txn_file_ref: &kvenginepb::TxnFileRef,
        is_prepared: bool,
        encryption_key: Option<EncryptionKey>,
    ) -> Result<TxnFile> {
        let mut chunks = Vec::with_capacity(txn_file_ref.chunk_ids.len());
        for &chunk_id in &txn_file_ref.chunk_ids {
            if !is_prepared {
                self.prepare(chunk_id, encryption_key.clone())?;
            }
            let txn_chunk = self.get(chunk_id).ok_or_else(|| -> Error {
                box_err!("txn chunk is not prepared, chunk_id {}", chunk_id)
            })?;
            chunks.push(txn_chunk);
        }
        let txn_ctx = TxnCtx::from_txn_file_ref(txn_file_ref);
        let txn_file_id = TxnFileId::new(shard_id, shard_ver, txn_file_ref.start_ts);
        Ok(TxnFile::new(txn_file_id, chunks, txn_ctx)?)
    }

    pub fn load_txn_files_from_refs(
        &self,
        shard_id: u64,
        shard_ver: u64,
        txn_file_refs: &[kvenginepb::TxnFileRef],
        encryption_key: Option<EncryptionKey>,
    ) -> Result<Vec<TxnFile>> {
        let txn_chunks_id = txn_file_refs
            .iter()
            .flat_map(|txn_file_ref| txn_file_ref.chunk_ids.iter())
            .copied()
            .collect::<Vec<_>>();
        self.prepare_txn_chunks(txn_chunks_id, encryption_key)?;
        txn_file_refs
            .iter()
            .map(|txn_file_ref| {
                self.load_txn_file_from_ref(shard_id, shard_ver, txn_file_ref, true, None)
            })
            .collect()
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
    use rstest::rstest;
    use tempfile::TempDir;

    use super::*;
    use crate::{
        dfs::InMemFs,
        table::{NoPrefixKey, TxnChunkBuilder, OP_PUT},
        BLOCK_CACHE_KEY_SIZE,
    };

    #[rstest]
    #[case(Some(TempDir::new().unwrap()))]
    #[case::in_mem(None)]
    fn test_txn_chunk_manager(#[case] tmp_dir: Option<TempDir>) {
        let local_path = tmp_dir.as_ref().map(|dir| dir.path().to_path_buf());
        let dfs: Arc<dyn Dfs> = Arc::new(InMemFs::new());
        let cache: SegmentedCache<BlockCacheKey, Bytes> = SegmentedCache::builder(256)
            .weigher(|_k: &BlockCacheKey, v: &Bytes| (BLOCK_CACHE_KEY_SIZE + v.len()) as u32)
            .max_capacity(1024 * 1024u64)
            .build();
        let txn_chunk_manager = TxnChunkManager::new(
            local_path.clone(),
            dfs.clone(),
            Some(cache.clone()),
            with_pool_size(2),
        );
        let runtime = dfs.get_runtime();
        let opts = dfs::Options::default().with_type(FileType::TxnChunk);
        for chunk_id in 1u64..=6 {
            let mut chunk_builder = TxnChunkBuilder::new(chunk_id, 10, None, 0, true);
            for i in 0..100 {
                let key = format!("{:02}/{:02}", chunk_id, i);
                chunk_builder.add_entry(NoPrefixKey(key.as_bytes()), OP_PUT, key.as_bytes());
            }
            let mut buf = vec![];
            chunk_builder.finish(&mut buf);
            runtime
                .block_on(dfs.create(chunk_id, buf.into(), opts))
                .unwrap();
        }

        for chunk_id in 1u64..=3 {
            txn_chunk_manager.prepare(chunk_id, None).unwrap();
            assert!(txn_chunk_manager.get(chunk_id).is_some());
        }
        txn_chunk_manager
            .prepare_txn_chunks(vec![4, 5, 6], None)
            .unwrap();
        assert!(txn_chunk_manager.all_chunks_exists(&[4, 5, 6]));

        txn_chunk_manager.remove(1);
        assert!(txn_chunk_manager.get(1).is_none());
        drop(txn_chunk_manager);

        let txn_chunk_manager =
            TxnChunkManager::new(local_path.clone(), dfs, Some(cache), with_pool_size(2));
        if local_path.is_some() {
            // After process restart, the remained txn chunks are all loaded.
            assert!(txn_chunk_manager.get(1).is_none());
            assert!(txn_chunk_manager.all_chunks_exists(&[2, 3]));
        } else {
            assert!(!txn_chunk_manager.all_chunks_exists(&[2, 3]));
        }
    }
}
