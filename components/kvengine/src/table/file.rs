// Copyright 2022 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    fmt,
    future::Future,
    ops::Deref,
    os::unix::fs::{FileExt, MetadataExt},
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering::Relaxed},
        Arc, Mutex,
    },
};

use bytes::Bytes;
use memmap2::Mmap;

use crate::{
    error::IoContext,
    table::{table, Error},
};

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

    /// `read_footer` reads last `footer_length` bytes of the file by default.
    ///
    /// Some implementation can have better performance (e.g. `IaFile`).
    fn read_footer(&self, footer_length: usize) -> table::Result<Bytes> {
        let size = self.size();
        if size < footer_length as u64 {
            error!("invalid file size"; "file_id" => self.id(), "size" => size, "footer_length" => footer_length);
            return Err(table::Error::InvalidFileSize);
        }
        self.read(size - footer_length as u64, footer_length)
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
            fd: TtlCache::new(true),
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

    fn read_footer(&self, footer_length: usize) -> table::Result<Bytes> {
        let Some(off) = self.size().checked_sub(footer_length as u64) else {
            error!("read footer: invalid size"; "file" => ?self, "footer_length" => footer_length);
            return Err(Error::InvalidFileSize);
        };
        self.read_inner(off, footer_length)
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

enum TtlCacheMutex<T> {
    Sync(std::sync::Mutex<T>),
    Async(tokio::sync::Mutex<T>),
}

pub struct TtlCache<T> {
    access_ns: AtomicU64,
    data: TtlCacheMutex<Option<Arc<T>>>,
    has_type_error: AtomicBool,
}

impl<T> TtlCache<T> {
    pub fn new(is_sync: bool) -> Self {
        Self {
            access_ns: AtomicU64::new(0),
            data: if is_sync {
                TtlCacheMutex::Sync(Default::default())
            } else {
                TtlCacheMutex::Async(Default::default())
            },
            has_type_error: AtomicBool::new(false),
        }
    }

    pub fn get(&self, init: impl FnOnce() -> table::Result<T>) -> table::Result<Arc<T>> {
        let now_ns = time::precise_time_ns();
        self.access_ns.store(now_ns, Relaxed);
        match &self.data {
            TtlCacheMutex::Sync(mu) => {
                let mut guard = mu.lock().unwrap();
                if guard.is_none() {
                    let data = init()?;
                    *guard = Some(Arc::new(data));
                }
                Ok(guard.as_ref().unwrap().clone())
            }
            _ => {
                self.handle_type_error(false);
                let data = init()?;
                Ok(Arc::new(data))
            }
        }
    }

    pub async fn get_async(
        &self,
        init: impl Future<Output = table::Result<T>>,
    ) -> table::Result<Arc<T>> {
        let now_ns = time::precise_time_ns();
        self.access_ns.store(now_ns, Relaxed);
        match &self.data {
            TtlCacheMutex::Async(mu) => {
                let mut guard = mu.lock().await;
                if guard.is_none() {
                    let data = init.await?;
                    *guard = Some(Arc::new(data));
                }
                Ok(guard.as_ref().unwrap().clone())
            }
            _ => {
                self.handle_type_error(true);
                let data = init.await?;
                Ok(Arc::new(data))
            }
        }
    }

    // TODO: remove when stable enough
    fn handle_type_error(&self, is_sync: bool) {
        debug_assert!(
            false,
            "incorrect TtlCache type: is_sync: {}, bt: {:?}",
            is_sync,
            backtrace::Backtrace::new(),
        );
        if self
            .has_type_error
            .compare_exchange_weak(false, true, Relaxed, Relaxed)
            .is_ok()
        {
            let bt = backtrace::Backtrace::new();
            error!("incorrect TtlCache type"; "is_sync" => is_sync, "bt" => ?bt);
        }
    }

    pub fn expire(&self, dur_secs: u64) {
        let access_ns = self.access_ns.load(Relaxed);
        let now_ns = time::precise_time_ns();
        let dur_nanos = dur_secs * 1_000_000_000;
        if access_ns > 0 && now_ns.saturating_sub(access_ns) > dur_nanos {
            match &self.data {
                TtlCacheMutex::Sync(mu) => {
                    if let Ok(mut data) = mu.try_lock() {
                        data.take();
                        self.access_ns.store(0, Relaxed);
                    }
                }
                TtlCacheMutex::Async(mu) => {
                    if let Ok(mut data) = mu.try_lock() {
                        data.take();
                        self.access_ns.store(0, Relaxed);
                    }
                }
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

    use futures::{executor::block_on, future};

    use crate::table::file::TtlCache;

    #[test]
    fn test_ttl_cache() {
        let cache: TtlCache<Vec<u8>> = TtlCache::new(true);
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

    #[tokio::test]
    async fn test_ttl_cache_async() {
        let cache: TtlCache<Vec<u8>> = TtlCache::new(false);
        assert!(!cache.is_loaded());
        let data = cache.get_async(future::ok(vec![1, 2, 3, 4])).await.unwrap();
        assert_eq!(data.deref(), &[1, 2, 3, 4]);
        assert!(cache.is_loaded());
        cache.expire(1);
        assert!(cache.is_loaded());
        tokio::time::sleep(Duration::from_millis(1500)).await;
        cache.expire(1);
        assert!(!cache.is_loaded());
    }

    #[bench]
    fn bench_ttl_cache(b: &mut test::Bencher) {
        let cache: TtlCache<u64> = TtlCache::new(true);
        b.iter(|| {
            for _ in 0..1000 {
                black_box(cache.get(|| Ok(1)).unwrap());
            }
        });
    }

    #[bench]
    fn bench_ttl_cache_async(b: &mut test::Bencher) {
        let cache: TtlCache<u64> = TtlCache::new(false);
        b.iter(|| {
            block_on(async {
                for _ in 0..1000 {
                    black_box(cache.get_async(future::ok(1)).await.unwrap());
                }
            });
        });
    }
}
