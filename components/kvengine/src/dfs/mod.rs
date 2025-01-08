// Copyright 2021 TiKV Project Authors. Licensed under Apache-2.0.

mod config;
mod metrics;
mod s3;

use std::{
    any::Any,
    convert::TryFrom,
    fmt::{Debug, Display, Formatter},
    io::{self, BufReader, Read, Seek, SeekFrom, Write},
    ops::Deref,
    os::unix::fs::FileExt,
    path::{Path, PathBuf},
    result,
    sync::{
        atomic,
        atomic::{AtomicBool, AtomicU64},
        Arc,
    },
    time::Duration,
};

use async_trait::async_trait;
use bytes::Bytes;
pub use config::Config as DFSConfig;
use file_system;
use metrics::*;
use moka::future::ConcurrentCacheExt;
pub use s3::*;
use thiserror::Error;
use tikv_util::time::Instant;
use tokio::runtime::Runtime;

use crate::{
    table::{
        blobtable::blobtable::BlobTable,
        columnar::{ColumnarFileFooter, SchemaFileFooter},
        sstable::{L0Table, SsTable},
        TxnChunk,
    },
    IoContext,
};

// DFS represents a distributed file system.
#[async_trait]
pub trait Dfs: Any + Sync + Send {
    /// read_file reads the whole file to memory.
    /// It can be used by remote compaction server that doesn't have local disk.
    async fn read_file(&self, file_id: u64, opts: Options) -> Result<Bytes>;

    /// Create creates a new File.
    /// The shard_id and shard_ver can be used determine where to write the
    /// file.
    async fn create(&self, file_id: u64, data: Bytes, opts: Options) -> Result<()>;

    /// remove removes the file from the DFS.
    /// `file_len` is used to choose proper storage class for cost
    /// efficiency, in bytes.
    /// `None` if file_len is unknown when invoke this method.
    async fn remove(&self, file_id: u64, file_len: Option<u64>, opts: Options);

    /// Remove the file from DFS permanently.
    async fn permanently_remove(&self, file_id: u64, opts: Options) -> Result<()>;

    /// get_runtime gets the tokio runtime for the DFS.
    fn get_runtime(&self) -> &tokio::runtime::Runtime;
}

const REMOVE_DELAY: Duration = Duration::from_secs(90);

pub struct InMemFs {
    files: dashmap::DashMap<u64, Bytes>,
    pending_remove: dashmap::DashMap<u64, Instant>,
    runtime: tokio::runtime::Runtime,
}

impl Default for InMemFs {
    fn default() -> Self {
        Self::new()
    }
}

impl InMemFs {
    pub fn new() -> Self {
        Self {
            files: Default::default(),
            pending_remove: Default::default(),
            runtime: tokio::runtime::Builder::new_multi_thread()
                .worker_threads(1)
                .enable_all()
                .build()
                .unwrap(),
        }
    }
}

#[async_trait]
impl Dfs for InMemFs {
    async fn read_file(&self, file_id: u64, opts: Options) -> Result<Bytes> {
        if let Some(file) = self.files.get(&file_id).as_deref() {
            if let Some(end_off) = opts.end_off {
                return Ok(file.slice(opts.start_off as usize..end_off as usize));
            } else {
                return Ok(file.slice(opts.start_off as usize..));
            }
        }
        Err(Error::NotExists(file_id))
    }

    async fn create(&self, file_id: u64, data: Bytes, _opts: Options) -> Result<()> {
        self.files.insert(file_id, data);
        Ok(())
    }

    async fn remove(&self, file_id: u64, _file_len: Option<u64>, _opts: Options) {
        if self.pending_remove.contains_key(&file_id) {
            return;
        }
        let now = Instant::now_coarse();
        self.pending_remove.insert(file_id, now);
        self.pending_remove.retain(|id, &mut remove_time| {
            if now.saturating_duration_since(remove_time) > REMOVE_DELAY {
                self.files.remove(id);
                false
            } else {
                true
            }
        });
    }

    async fn permanently_remove(&self, file_id: u64, _opts: Options) -> Result<()> {
        self.files.remove(&file_id);
        Ok(())
    }

    fn get_runtime(&self) -> &Runtime {
        &self.runtime
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(u8)]
pub enum FileType {
    Sst = 0,
    TxnChunk = 1,
    Schema = 2,
    Columnar = 3,
    Blob = 4,
    VectorIndex = 5,
}

impl FileType {
    pub fn from_u8(t: u8) -> Option<FileType> {
        match t {
            0 => Some(FileType::Sst),
            1 => Some(FileType::TxnChunk),
            2 => Some(FileType::Schema),
            3 => Some(FileType::Columnar),
            4 => Some(FileType::Blob),
            5 => Some(FileType::VectorIndex),
            _ => None,
        }
    }

    pub fn suffix(&self) -> &'static str {
        match self {
            FileType::Sst => "sst",
            FileType::TxnChunk => "txn",
            FileType::Schema => "schema",
            FileType::Columnar => "col",
            FileType::Blob => "blob",
            FileType::VectorIndex => "vec",
        }
    }

    pub fn footer_size(&self) -> usize {
        match self {
            FileType::Sst => std::cmp::max(SsTable::footer_size(), L0Table::footer_size()),
            FileType::TxnChunk => TxnChunk::footer_size(),
            FileType::Schema => SchemaFileFooter::footer_size(),
            FileType::Columnar => ColumnarFileFooter::compute_size(),
            FileType::Blob => BlobTable::footer_size(),
            FileType::VectorIndex => unimplemented!(), // TODO
        }
    }
}

impl Display for FileType {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.suffix())
    }
}

impl TryFrom<&str> for FileType {
    type Error = String;

    fn try_from(value: &str) -> std::result::Result<Self, Self::Error> {
        Ok(match value {
            "sst" => FileType::Sst,
            "txn" => FileType::TxnChunk,
            "schema" => FileType::Schema,
            "col" => FileType::Columnar,
            "blob" => FileType::Blob,
            "vec" => FileType::VectorIndex,
            _ => {
                return Err(format!("invalid suffix: {value}"));
            }
        })
    }
}

#[derive(Clone)]
pub struct CacheFs {
    cache: moka::future::Cache<u64, Bytes>,
    s3_fs: Arc<S3Fs>,
}

impl CacheFs {
    pub fn new(cache_size: u64, s3_fs: Arc<S3Fs>) -> Self {
        let builder = moka::future::CacheBuilder::new(cache_size);
        // Do not set TTL to make the memory usage stable over time.
        let builder = builder.weigher(|_, v: &Bytes| v.len() as u32);
        let cache = builder.build();
        Self { cache, s3_fs }
    }

    async fn read_file_inner(&self, file_id: u64, opts: Options) -> Result<Bytes> {
        let s3_fs = self.s3_fs.clone();
        let cache_miss = Arc::new(AtomicBool::new(false));
        let cache_miss_clone = cache_miss.clone();
        let file = self
            .cache
            .try_get_with(file_id, async move {
                cache_miss_clone.store(true, atomic::Ordering::Relaxed);
                let full_range_opts = Options {
                    file_type: opts.file_type,
                    shard_id: opts.shard_id,
                    shard_ver: opts.shard_ver,
                    start_off: 0,
                    end_off: None,
                };
                s3_fs.read_file(file_id, full_range_opts).await
            })
            .await
            .map_err(|e| e.as_ref().clone())?;
        if cache_miss.load(atomic::Ordering::Acquire) {
            // In case the CacheFS exceeds its capacity, we explicitly sync it when new
            // entry is added.
            self.cache.sync();
            KVENGINE_CACHEFS_REQ_COUNTER_VEC
                .with_label_values(&["miss"])
                .inc();
        } else {
            KVENGINE_CACHEFS_REQ_COUNTER_VEC
                .with_label_values(&["hit"])
                .inc();
        }
        if let Some(end_off) = opts.end_off {
            Ok(file.slice(opts.start_off as usize..end_off as usize))
        } else {
            Ok(file.slice(opts.start_off as usize..))
        }
    }
}

#[async_trait]
impl Dfs for CacheFs {
    /// Note: `CacheFs` will cache the whole object, even if `read_file` just
    /// read part of it.
    async fn read_file(&self, file_id: u64, opts: Options) -> Result<Bytes> {
        self.read_file_inner(file_id, opts).await
    }

    async fn create(&self, _file_id: u64, _data: Bytes, _opts: Options) -> Result<()> {
        panic!("Do not call");
    }

    async fn remove(&self, _file_id: u64, _file_len: Option<u64>, _opts: Options) {
        panic!("Do not call");
    }

    async fn permanently_remove(&self, _file_id: u64, _opts: Options) -> Result<()> {
        panic!("Do not call");
    }

    fn get_runtime(&self) -> &Runtime {
        self.s3_fs.get_runtime()
    }
}

#[derive(Clone)]
pub struct LocalFs {
    core: Arc<LocalFsCore>,
}

impl LocalFs {
    pub fn new(dir: &Path) -> Self {
        let core = Arc::new(LocalFsCore::new(dir));
        Self { core }
    }
    pub fn local_sst_file_path(&self, file_id: u64) -> PathBuf {
        self.dir.join(self.sst_filename(file_id))
    }
    pub fn local_blob_file_path(&self, file_id: u64) -> PathBuf {
        self.dir.join(self.blob_filename(file_id))
    }
    pub fn local_columnar_file_path(&self, file_id: u64) -> PathBuf {
        self.dir.join(self.columnar_filename(file_id))
    }
    pub fn local_vector_index_file_path(&self, file_id: u64) -> PathBuf {
        self.dir.join(self.vector_index_filename(file_id))
    }
    pub fn sst_filename(&self, file_id: u64) -> PathBuf {
        PathBuf::from(format!("{:016x}.sst", file_id))
    }
    pub fn blob_filename(&self, file_id: u64) -> PathBuf {
        PathBuf::from(format!("{:016x}.blob", file_id))
    }
    pub fn columnar_filename(&self, file_id: u64) -> PathBuf {
        PathBuf::from(format!("{:016x}.col", file_id))
    }
    pub fn vector_index_filename(&self, file_id: u64) -> PathBuf {
        PathBuf::from(format!("{:016x}.vec", file_id))
    }
    pub fn tmp_file_path(&self, file_id: u64) -> PathBuf {
        let tmp_id = self
            .tmp_file_id
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        self.dir.join(self.new_tmp_filename(file_id, tmp_id))
    }
    pub fn new_tmp_filename(&self, file_id: u64, tmp_id: u64) -> PathBuf {
        PathBuf::from(format!("{:016x}.{}.tmp", file_id, tmp_id))
    }
    pub fn local_txn_chunk_path(&self, id: u64) -> PathBuf {
        self.dir.join("txn").join(format!("{:016x}.txn", id))
    }
    pub fn local_schema_file_path(&self, id: u64) -> PathBuf {
        self.dir.join("schema").join(format!("{:016x}.schema", id))
    }
    pub fn local_file_path(&self, file_id: u64, file_type: FileType) -> PathBuf {
        match file_type {
            FileType::Sst => self.local_sst_file_path(file_id),
            FileType::TxnChunk => self.local_txn_chunk_path(file_id),
            FileType::Schema => self.local_schema_file_path(file_id),
            FileType::Columnar => self.local_columnar_file_path(file_id),
            FileType::Blob => self.local_blob_file_path(file_id),
            FileType::VectorIndex => self.local_vector_index_file_path(file_id),
        }
    }
}

impl Deref for LocalFs {
    type Target = LocalFsCore;

    fn deref(&self) -> &Self::Target {
        &self.core
    }
}

pub struct LocalFsCore {
    dir: PathBuf,
    tmp_file_id: AtomicU64,
    runtime: tokio::runtime::Runtime,
}

impl LocalFsCore {
    pub fn new(dir: &Path) -> Self {
        if !dir.exists() {
            std::fs::create_dir_all(dir).unwrap();
        }
        if !dir.is_dir() {
            panic!("path {:?} is not dir", dir);
        }
        Self {
            dir: dir.to_owned(),
            tmp_file_id: AtomicU64::new(0),
            runtime: tokio::runtime::Builder::new_multi_thread()
                .worker_threads(8)
                .enable_all()
                .build()
                .unwrap(),
        }
    }
}

#[async_trait]
impl Dfs for LocalFs {
    async fn read_file(&self, file_id: u64, opts: Options) -> Result<Bytes> {
        let local_file_name = self.local_file_path(file_id, opts.file_type);
        let mut fd = std::fs::File::open(local_file_name).dfs_ctx(file_id, "open")?;
        let buf = if let Some(end_off) = opts.end_off {
            let mut buf = vec![0; (end_off - opts.start_off) as usize];
            fd.read_at(&mut buf, opts.start_off)
                .dfs_ctx(file_id, "read")?;
            buf
        } else {
            if opts.start_off > 0 {
                fd.seek(SeekFrom::Start(opts.start_off))
                    .dfs_ctx(file_id, "seek")?;
            }
            let mut reader = BufReader::new(fd);
            let mut buf = Vec::new();
            reader.read_to_end(&mut buf).dfs_ctx(file_id, "read")?;
            buf
        };
        KVENGINE_DFS_THROUGHPUT_VEC
            .with_label_values(&["read"])
            .inc_by(buf.len() as u64);
        Ok(Bytes::from(buf))
    }

    async fn create(&self, file_id: u64, data: Bytes, opts: Options) -> Result<()> {
        let local_file_name = self.local_file_path(file_id, opts.file_type);
        let tmp_file_name = self.tmp_file_path(file_id);
        let mut file = std::fs::File::create(&tmp_file_name)?;
        let mut start_off = 0;
        let write_batch_size = 256 * 1024;
        while start_off < data.len() {
            let end_off = std::cmp::min(start_off + write_batch_size, data.len());
            file.write_all(&data[start_off..end_off])?;
            file.sync_data()?;
            start_off = end_off;
        }
        std::fs::rename(&tmp_file_name, local_file_name)?;
        file_system::sync_dir(&self.dir)?;
        KVENGINE_DFS_THROUGHPUT_VEC
            .with_label_values(&["write"])
            .inc_by(data.len() as u64);
        Ok(())
    }

    async fn remove(&self, file_id: u64, _file_len: Option<u64>, opts: Options) {
        let local_file_path = self.local_file_path(file_id, opts.file_type);
        if let Err(err) = std::fs::remove_file(local_file_path) {
            error!("failed to remove local file {:?}", err);
        }
    }

    async fn permanently_remove(&self, file_id: u64, opts: Options) -> Result<()> {
        self.remove(file_id, None, opts).await;
        Ok(())
    }

    fn get_runtime(&self) -> &Runtime {
        &self.runtime
    }
}

#[derive(Clone, Copy)]
pub struct Options {
    pub file_type: FileType,
    pub shard_id: u64,
    pub shard_ver: u64,
    pub start_off: u64,
    pub end_off: Option<u64>,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            file_type: FileType::Sst,
            shard_id: 0,
            shard_ver: 0,
            start_off: 0,
            end_off: None,
        }
    }
}

impl Options {
    pub fn with_shard(mut self, shard_id: u64, shard_ver: u64) -> Self {
        self.shard_id = shard_id;
        self.shard_ver = shard_ver;
        self
    }

    pub fn with_type(mut self, file_type: FileType) -> Self {
        self.file_type = file_type;
        self
    }

    pub fn with_start_off(mut self, start_off: u64) -> Self {
        self.start_off = start_off;
        self
    }

    pub fn with_end_off(mut self, end_off: u64) -> Self {
        self.end_off = Some(end_off);
        self
    }
}

pub type Result<T> = result::Result<T, Error>;

#[derive(Debug, Error, Clone)]
pub enum Error {
    #[error("IO error: {0}")]
    Io(String),
    #[error("File {0} not exists")]
    NotExists(u64),
    #[error("Txn Chunk {0} not exists")]
    TxnChunkNotExists(u64),
    #[error("S3 error {0}")]
    S3(String),
    #[error("Other error {0}")]
    Other(String),
    #[error("The specified key {0} does not exist.")]
    NoSuchKey(String),
    #[error("hyper error {0}")]
    Hyper(String),
}

impl From<io::Error> for Error {
    #[inline]
    fn from(e: io::Error) -> Error {
        Error::Io(e.to_string())
    }
}

impl From<hyper::Error> for Error {
    #[inline]
    fn from(e: hyper::Error) -> Error {
        Error::Hyper(e.to_string())
    }
}

impl<E: Debug> From<rusoto_core::RusotoError<E>> for Error {
    fn from(err: rusoto_core::RusotoError<E>) -> Self {
        Error::S3(format!("{:?}", err))
    }
}

#[cfg(test)]
mod tests {
    use std::os::unix::fs::MetadataExt;

    use super::*;
    use crate::dfs::LocalFs;

    #[test]
    fn test_local_fs() {
        ::test_util::init_log_for_test();

        let local_dir = tempfile::tempdir().unwrap();
        let file_data = "abcdefgh".to_string().into_bytes();
        let localfs = LocalFs::new(local_dir.path());
        let (tx, rx) = tikv_util::mpsc::bounded(1);
        let file_id = 321u64;
        let fs = localfs.clone();
        let file_data_clone = file_data.clone();
        let f = async move {
            match fs
                .create(
                    file_id,
                    bytes::Bytes::from(file_data_clone),
                    Options::default(),
                )
                .await
            {
                Ok(_) => {
                    tx.send(true).unwrap();
                    println!("create ok");
                }
                Err(err) => {
                    tx.send(false).unwrap();
                    println!("create error {:?}", err)
                }
            }
        };
        localfs.runtime.spawn(f);
        assert!(rx.recv().unwrap());
        let fs = localfs.clone();
        let (tx, rx) = tikv_util::mpsc::bounded(1);
        let f = async move {
            let opts = Options::default();
            match fs.read_file(file_id, opts).await {
                Ok(data) => {
                    assert_eq!(&data, &file_data);
                    tx.send(true).unwrap();
                    println!("prefetch ok");
                }
                Err(err) => {
                    tx.send(false).unwrap();
                    println!("prefetch failed {:?}", err)
                }
            }
        };
        localfs.runtime.spawn(f);
        assert!(rx.recv().unwrap());
        let local_file = localfs.local_sst_file_path(file_id);
        let fd = std::fs::File::open(&local_file).unwrap();
        let meta = fd.metadata().unwrap();
        assert_eq!(meta.size(), 8u64);
        let fs = localfs.clone();
        let (tx, rx) = tikv_util::mpsc::bounded(1);
        let f = async move {
            fs.remove(file_id, None, Options::default()).await;
            tx.send(true).unwrap();
        };
        localfs.runtime.spawn(f);
        assert!(rx.recv().unwrap());
        std::fs::File::open(&local_file).unwrap_err();
    }
}
