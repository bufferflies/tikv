// Copyright 2022 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    fmt,
    ops::Deref,
    os::unix::fs::{FileExt, MetadataExt},
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicU64, Ordering::Relaxed},
        Arc, Mutex,
    },
};

use bytes::Bytes;
use memmap2::Mmap;

use crate::{error::IoContext, ia::types::FileSegmentIdent, table::table};

// 30 minutes idle file would be closed.
const FILE_TTL: u64 = 30 * 60;

#[async_trait::async_trait]
pub trait File: Sync + Send {
    // id returns the id of the file.
    fn id(&self) -> u64;

    /// `size` returns the size of the file.
    fn size(&self) -> u64;

    /// `path` returns the file path.
    fn path(&self) -> Option<PathBuf> {
        None
    }

    /// `is_sync` indicates that whether the whole file is local and can always
    /// be read by sync methods.
    fn is_sync(&self) -> bool {
        true
    }

    /// `read` reads the data at given offset.
    fn read(&self, off: u64, length: usize) -> table::Result<Bytes>;

    /// `read_at` reads the data to the buffer.
    fn read_at(&self, buf: &mut [u8], offset: u64) -> table::Result<()>;

    /// `read_table_meta` read meta (e.g, index, filter) of tables.
    ///
    /// `read_table_meta` has the same result of `read`. But some implementation
    /// can have better performance (e.g. `IaFile`).
    fn read_table_meta(&self, off: u64, length: usize) -> table::Result<Bytes> {
        self.read(off, length)
    }

    fn read_footer(&self, footer_length: usize) -> table::Result<Bytes> {
        let size = self.size();
        let Some(off) = size.checked_sub(footer_length as u64) else {
            error!("invalid file size"; "file_id" => self.id(), "size" => size, "footer_length" => footer_length);
            return Err(table::Error::InvalidFileSize);
        };
        self.read_table_meta(off, footer_length)
    }

    /// `read_async` is async version of `read`.
    async fn read_async(&self, off: u64, length: usize) -> table::Result<Bytes> {
        self.read(off, length)
    }

    /// `read_at_async` is async version of `read_at`.
    async fn read_at_async(&self, buf: &mut [u8], offset: u64) -> table::Result<()> {
        self.read_at(buf, offset)
    }

    /// `expire_open_file` closes the file if it's idle for a long time.
    ///
    /// It will be reopened and cached on next read.
    fn expire_open_file(&self) {}

    fn is_open(&self) -> bool {
        false
    }

    fn mmap(&self) -> table::Result<MmapData>;

    /// `get_remote_segments` returns the remote segments of the file. Used for
    /// prefetching. Available for IA files only.
    fn get_remote_segments(
        &self,
        _start_off: u64,
        _end_off: u64,
    ) -> table::Result<(Vec<FileSegmentIdent>, usize /* total_segments */)> {
        Ok((vec![], 0))
    }
}

pub enum MmapData {
    Local(Arc<Mmap>),
    InMem(Bytes),
}

impl Deref for MmapData {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        match self {
            MmapData::Local(mmap) => mmap.deref(),
            MmapData::InMem(data) => data.deref(),
        }
    }
}

pub struct LocalFile {
    id: u64,
    size: u64,
    path: PathBuf,
    fd: TtlCache<std::fs::File>,
    mmap: Mutex<Option<Arc<Mmap>>>,
}

impl LocalFile {
    pub fn open(id: u64, path: &Path, set_mtime: bool) -> table::Result<LocalFile> {
        if set_mtime {
            filetime::set_file_mtime(path, filetime::FileTime::now())
                .table_ctx(id, "local.open.set_file_mtime")?;
        }
        let meta = std::fs::metadata(path).table_ctx(id, "local.open.metadata")?;
        let local_file = LocalFile {
            id,
            size: meta.size(),
            path: path.to_path_buf(),
            fd: TtlCache::default(),
            mmap: Mutex::new(None),
        };
        Ok(local_file)
    }

    pub fn from_file(id: u64, path: PathBuf, file: Arc<std::fs::File>) -> table::Result<LocalFile> {
        let meta = std::fs::metadata(&path).table_ctx(id, "local.from_file.metadata")?;
        let local_file = LocalFile {
            id,
            size: meta.size(),
            path,
            fd: TtlCache::new(file),
            mmap: Mutex::new(None),
        };
        Ok(local_file)
    }

    fn get_file(&self) -> table::Result<Arc<std::fs::File>> {
        self.fd
            .get(|| std::fs::File::open(self.path.as_path()).table_ctx(self.id, "local.get_file"))
    }
}

impl File for LocalFile {
    fn id(&self) -> u64 {
        self.id
    }

    fn size(&self) -> u64 {
        self.size
    }

    fn path(&self) -> Option<PathBuf> {
        Some(self.path.clone())
    }

    fn read(&self, off: u64, length: usize) -> table::Result<Bytes> {
        let mut buf = vec![0; length];
        let fd = self.get_file()?;
        fd.read_at(&mut buf, off)
            .table_ctx(self.id(), "local.read")?;
        Ok(Bytes::from(buf))
    }

    fn read_at(&self, buf: &mut [u8], offset: u64) -> table::Result<()> {
        let fd = self.get_file()?;
        fd.read_at(buf, offset)
            .table_ctx(self.id(), "local.read_at")?;
        Ok(())
    }

    fn expire_open_file(&self) {
        self.fd.expire(FILE_TTL)
    }

    fn is_open(&self) -> bool {
        self.fd.is_loaded()
    }

    fn mmap(&self) -> table::Result<MmapData> {
        let mut gurad = self.mmap.lock().unwrap();
        if gurad.is_none() {
            let fd = self.get_file()?;
            let mmap = unsafe { Mmap::map(&fd).table_ctx(self.id(), "local.mmap")? };
            *gurad = Some(Arc::new(mmap));
        }
        let mmap = gurad.as_ref().unwrap().clone();
        Ok(MmapData::Local(mmap))
    }
}

#[derive(Clone)]
pub struct InMemFile {
    pub id: u64,
    data: Bytes,
    pub size: u64,
    is_sync: bool,
}

impl fmt::Debug for InMemFile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("InMemFile")
            .field("id", &self.id)
            .field("size", &self.size)
            .finish()
    }
}

impl InMemFile {
    pub fn new(id: u64, data: Bytes) -> Self {
        let size = data.len() as u64;
        Self {
            id,
            data,
            size,
            is_sync: true,
        }
    }

    #[cfg(test)]
    pub async fn new_async(id: u64, data: Bytes) -> Self {
        let size = data.len() as u64;
        Self {
            id,
            data,
            size,
            is_sync: false,
        }
    }

    fn read_inner(&self, off: u64, length: usize) -> table::Result<Bytes> {
        let off_usize = off as usize;
        Ok(self.data.slice(off_usize..off_usize + length))
    }

    fn read_at_inner(&self, buf: &mut [u8], offset: u64) -> table::Result<()> {
        let off_usize = offset as usize;
        let length = buf.len();
        buf.copy_from_slice(&self.data[off_usize..off_usize + length]);
        Ok(())
    }
}

#[async_trait::async_trait]
impl File for InMemFile {
    fn id(&self) -> u64 {
        self.id
    }

    fn size(&self) -> u64 {
        self.size
    }

    fn is_sync(&self) -> bool {
        self.is_sync
    }

    #[inline]
    fn read(&self, off: u64, length: usize) -> table::Result<Bytes> {
        debug_assert!(self.is_sync);
        self.read_inner(off, length)
    }

    #[inline]
    fn read_at(&self, buf: &mut [u8], offset: u64) -> table::Result<()> {
        debug_assert!(self.is_sync);
        self.read_at_inner(buf, offset)
    }

    fn read_table_meta(&self, off: u64, length: usize) -> table::Result<Bytes> {
        // Skip the `is_sync` checking.
        self.read_inner(off, length)
    }

    #[cfg(test)]
    #[inline]
    async fn read_async(&self, off: u64, length: usize) -> table::Result<Bytes> {
        debug_assert!(!self.is_sync);
        self.read_inner(off, length)
    }

    #[cfg(test)]
    #[inline]
    async fn read_at_async(&self, buf: &mut [u8], offset: u64) -> table::Result<()> {
        debug_assert!(!self.is_sync);
        self.read_at_inner(buf, offset)
    }

    fn mmap(&self) -> table::Result<MmapData> {
        Ok(MmapData::InMem(self.data.clone()))
    }
}

pub struct TtlCache<T> {
    access_ns: AtomicU64,
    data: Mutex<Option<Arc<T>>>,
}

impl<T> Default for TtlCache<T> {
    fn default() -> Self {
        Self {
            access_ns: AtomicU64::new(0),
            data: Default::default(),
        }
    }
}

impl<T> TtlCache<T> {
    pub fn new(t: Arc<T>) -> Self {
        let now_ns = time::precise_time_ns();
        Self {
            access_ns: AtomicU64::new(now_ns),
            data: Mutex::new(Some(t)),
        }
    }

    pub fn get(&self, init: impl FnOnce() -> table::Result<T>) -> table::Result<Arc<T>> {
        let now_ns = time::precise_time_ns();
        self.access_ns.store(now_ns, Relaxed);
        let mut guard = self.data.lock().unwrap();
        if guard.is_none() {
            let data = init()?;
            *guard = Some(Arc::new(data));
        }
        Ok(guard.as_ref().unwrap().clone())
    }

    pub fn expire(&self, dur_secs: u64) {
        let access_ns = self.access_ns.load(Relaxed);
        let now_ns = time::precise_time_ns();
        let dur_nanos = dur_secs * 1_000_000_000;
        if access_ns > 0 && now_ns.saturating_sub(access_ns) > dur_nanos {
            if let Ok(mut data) = self.data.try_lock() {
                data.take();
                self.access_ns.store(0, Relaxed);
            }
        }
    }

    pub fn is_loaded(&self) -> bool {
        self.access_ns.load(Relaxed) > 0
    }
}

#[cfg(test)]
mod tests {
    use std::{intrinsics::black_box, ops::Deref, time::Duration};

    use crate::table::file::TtlCache;

    #[test]
    fn test_ttl_cache() {
        let cache: TtlCache<Vec<u8>> = TtlCache::default();
        assert!(!cache.is_loaded());
        let data = cache.get(|| Ok(vec![1, 2, 3, 4])).unwrap();
        assert_eq!(data.deref(), &[1, 2, 3, 4]);
        assert!(cache.is_loaded());
        cache.expire(1);
        assert!(cache.is_loaded());
        std::thread::sleep(Duration::from_millis(1500));
        cache.expire(1);
        assert!(!cache.is_loaded());
    }

    #[bench]
    fn bench_ttl_cache(b: &mut test::Bencher) {
        let cache: TtlCache<u64> = TtlCache::default();
        b.iter(|| {
            for _ in 0..1000 {
                black_box(cache.get(|| Ok(1)).unwrap());
            }
        });
    }
}
