// Copyright 2024 TiKV Project Authors. Licensed under Apache-2.0.

#![allow(dead_code)]

use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicU64, Ordering::Relaxed},
        Arc,
    },
    time::Duration,
};

use async_trait::async_trait;
use bytes::{Buf, Bytes};
use dashmap::DashMap;
use nix::NixPath;
use sysinfo::{DiskExt, System as Sys, SystemExt};
use tikv_util::sys::SysQuota;
use tokio::{
    fs,
    io::{AsyncReadExt, AsyncSeekExt, AsyncWriteExt, SeekFrom},
};

use crate::{
    ia::{
        manager::{IaManagerOptions, QueueOptions},
        types::FileSegmentIdent,
    },
    table::{Error, Result},
    IoContext,
};

/// LocalStore is used to provide uniform access to both local disk and memory.
#[async_trait]
pub trait LocalStore: Send + Sync {
    /// Return the path of store. Will be `None` when it's in memory.
    fn path(&self) -> Option<&Path>;

    async fn init(&self) -> Result<()>;

    /// Return the existed keys & suffixes in the store from last startup.
    async fn scan(&self) -> Result<HashMap<String /* suffix */, Vec<String> /* keys */>>;

    async fn save(&self, file_id: u64, key: &str, data: Bytes) -> Result<()>;

    /// Read exactly `buf.len()` bytes into `buf` starting at `offset`.
    async fn read_at(
        &self,
        file_id: u64,
        key: &str,
        buf: &mut [u8],
        offset: u64,
    ) -> Result<Option<()>>;

    async fn read(
        &self,
        file_id: u64,
        key: &str,
        start_off: u64,
        end_off: u64,
    ) -> Result<Option<Bytes>>;

    async fn read_all(&self, file_id: u64, key: &str, buf: &mut Vec<u8>) -> Result<Option<()>>;

    async fn remove(&self, file_id: u64, key: &str) -> Result<Option<()>>;

    async fn exists(&self, file_id: u64, key: &str) -> bool;
}

pub fn new_local_store(path: Option<PathBuf>) -> Arc<dyn LocalStore> {
    if let Some(path) = path {
        Arc::new(LocalFileStore::new(path)) as _
    } else {
        Arc::new(LocalMemoryStore::default()) as _
    }
}

macro_rules! try_open {
    ($path:expr) => {{
        match tokio::fs::File::open($path).await {
            Ok(f) => Ok(f),
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
                return Ok(None);
            }
            Err(err) => Err(err),
        }
    }};
}

macro_rules! try_remove {
    ($path:expr) => {{
        match tokio::fs::remove_file($path).await {
            Ok(()) => Ok(()),
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
                return Ok(None);
            }
            Err(err) => Err(err),
        }
    }};
}

#[macro_export]
macro_rules! try_some {
    ($expr:expr) => {{
        match $expr {
            Some(v) => v,
            None => return Ok(None),
        }
    }};
}

pub struct LocalFileStore {
    dir: PathBuf,
}

impl LocalFileStore {
    pub fn new(dir: PathBuf) -> Self {
        Self { dir }
    }
}

#[async_trait]
impl LocalStore for LocalFileStore {
    fn path(&self) -> Option<&Path> {
        Some(&self.dir)
    }

    async fn init(&self) -> Result<()> {
        fs::create_dir_all(&self.dir)
            .await
            .table_ctx(0, format!("create_dir.{:?}", self.dir))
    }

    async fn scan(&self) -> Result<HashMap<String /* suffix */, Vec<String> /* keys */>> {
        let mut entries = fs::read_dir(&self.dir)
            .await
            .table_ctx(0, format!("read_dir.{:?}", self.dir))?;
        let mut map = HashMap::new();
        while let Some(entry) = entries.next_entry().await.table_ctx(0, "next_entry")? {
            let path = entry.path();
            if path.is_file() {
                let key = path.file_name().unwrap().to_str().unwrap().to_owned();
                let suffix = path.extension().unwrap().to_str().unwrap().to_owned();
                map.entry(suffix).or_insert_with(Vec::new).push(key);
            }
        }
        Ok(map)
    }

    async fn save(&self, file_id: u64, key: &str, data: Bytes) -> Result<()> {
        lazy_static::lazy_static! {
            static ref TMP_ID: AtomicU64 = AtomicU64::new(0);
        }

        let tmp_filename = format!("{}.{}.tmp", key, TMP_ID.fetch_add(1, Relaxed));
        let tmp_path = self.dir.join(tmp_filename);
        let mut f = fs::File::create(&tmp_path)
            .await
            .table_ctx(file_id, format!("create_tmp.{key}"))?;

        f.write_all(&data)
            .await
            .table_ctx(file_id, format!("write_tmp.{key}"))?;

        let path = self.dir.join(key);
        fs::rename(&tmp_path, &path)
            .await
            .table_ctx(file_id, format!("rename.{key}"))?;

        debug!("FileDataStore.save"; "file_id" => file_id, "key" => key, "path" => ?path);
        Ok(())
    }

    async fn read_at(
        &self,
        file_id: u64,
        key: &str,
        buf: &mut [u8],
        offset: u64,
    ) -> Result<Option<()>> {
        let path = self.dir.join(key);
        debug!("FileDataStore.read_at"; "file_id" => file_id, "key" => key, "path" => ?path);
        let mut f = try_open!(&path).table_ctx(file_id, format!("open.{key}"))?;
        if offset > 0 {
            f.seek(SeekFrom::Start(offset))
                .await
                .table_ctx(file_id, format!("seek.{key}"))?;
        }
        f.read_exact(buf)
            .await
            .table_ctx(file_id, format!("read_exact.{key}"))?;
        Ok(Some(()))
    }

    async fn read(
        &self,
        file_id: u64,
        key: &str,
        start_off: u64,
        end_off: u64,
    ) -> Result<Option<Bytes>> {
        let mut buf = vec![0; (end_off - start_off) as usize];
        try_some!(self.read_at(file_id, key, &mut buf, start_off).await?);
        Ok(Some(Bytes::from(buf)))
    }

    async fn read_all(&self, file_id: u64, key: &str, buf: &mut Vec<u8>) -> Result<Option<()>> {
        let path = self.dir.join(key);
        let mut f = try_open!(&path).table_ctx(file_id, format!("open.{key}"))?;
        f.read_to_end(buf)
            .await
            .table_ctx(file_id, format!("read_to_end.{key}"))?;
        Ok(Some(()))
    }

    async fn remove(&self, file_id: u64, key: &str) -> Result<Option<()>> {
        let path = self.dir.join(key);
        try_remove!(&path).table_ctx(file_id, format!("remove.{key}"))?;
        Ok(Some(()))
    }

    async fn exists(&self, _file_id: u64, key: &str) -> bool {
        let path = self.dir.join(key);
        tokio::fs::metadata(&path).await.is_ok_and(|m| m.is_file())
    }
}

#[derive(Default)]
pub(crate) struct LocalMemoryStore {
    m: DashMap<String, Bytes>,
}

impl LocalMemoryStore {
    fn get(&self, key: &str) -> Option<Bytes> {
        self.m.get(key).map(|x| x.value().clone())
    }

    fn get_with_check(&self, key: &str, end_off: u64) -> Result<Option<Bytes>> {
        let data = try_some!(self.get(key));
        if end_off > data.len() as u64 {
            return Err(Error::Io(format!(
                "read out of range, key: {}, end_off: {}, data length: {}",
                key,
                end_off,
                data.len()
            )));
        }
        Ok(Some(data))
    }
}

#[async_trait]
impl LocalStore for LocalMemoryStore {
    fn path(&self) -> Option<&Path> {
        None
    }

    async fn init(&self) -> Result<()> {
        Ok(())
    }

    async fn scan(&self) -> Result<HashMap<String /* suffix */, Vec<String> /* keys */>> {
        Ok(HashMap::new())
    }

    async fn save(&self, _file_id: u64, key: &str, data: Bytes) -> Result<()> {
        self.m.insert(key.to_owned(), data);
        Ok(())
    }

    async fn read_at(
        &self,
        _file_id: u64,
        key: &str,
        buf: &mut [u8],
        offset: u64,
    ) -> Result<Option<()>> {
        let end_off = offset + buf.len() as u64;
        let data = try_some!(self.get_with_check(key, end_off)?);
        buf.copy_from_slice(&data[offset as usize..end_off as usize]);
        Ok(Some(()))
    }

    async fn read(
        &self,
        _file_id: u64,
        key: &str,
        start_off: u64,
        end_off: u64,
    ) -> Result<Option<Bytes>> {
        let data = try_some!(self.get_with_check(key, end_off)?);
        Ok(Some(Bytes::copy_from_slice(
            data.slice(start_off as usize..end_off as usize).chunk(),
        )))
    }

    async fn read_all(&self, _file_id: u64, key: &str, buf: &mut Vec<u8>) -> Result<Option<()>> {
        let data = try_some!(self.get(key));
        buf.extend_from_slice(&data);
        Ok(Some(()))
    }

    async fn remove(&self, _file_id: u64, key: &str) -> Result<Option<()>> {
        Ok(self.m.remove(key).map(|_| ()))
    }

    async fn exists(&self, _file_id: u64, key: &str) -> bool {
        self.m.contains_key(key)
    }
}

/// Split the range [start_off, end_off) into segments with size `segment_size`.
///
/// Align to integer times of `segment_size`.
///
/// The last segment is rounded to the total_size as half segment size.
///
/// E.g.: With regard to segment size of 100, last segment of file with total
/// size of 1024 [900, 1024], while for file with total size of 1050 it is
/// [1000, 1050].
pub(crate) fn split_to_segments(
    file_id: u64,
    start_off: u64,
    end_off: u64,
    total_size: u64,
    segment_size: u64,
) -> Vec<FileSegmentIdent> {
    debug_assert!(start_off < end_off && end_off <= total_size);

    let start_floor = start_off / segment_size * segment_size;
    let end_ceil = (end_off + segment_size - 1) / segment_size * segment_size;
    let mut segs = (start_floor..end_ceil)
        .step_by(segment_size as usize)
        .map(|start_off| FileSegmentIdent {
            file_id,
            start_off,
            end_off: start_off + segment_size,
        })
        .collect::<Vec<_>>();

    debug_assert!(!segs.is_empty());
    if total_size < end_ceil {
        segs.last_mut().unwrap().end_off = total_size;
    }
    segs
}

const FILE_SEGMENT_SIZE_DEF: i64 = 1 << 20; // 1MiB
const MAIN_QUEUE_CAPACITY_FACTOR: i64 = 10; // Main queue is 10x larger than small queue.
const FREQ_UPDATE_INTERVAL: Duration = Duration::from_secs(60);

pub enum IaCapacity {
    Manual {
        small_queue: QueueOptions,
        main_queue: QueueOptions,
    },
    MemoryRatio(f64),
    MemoryCap(i64),
    /// Small queue in memory & main queue in disk.
    MemoryAndDiskRatio(f64 /* mem_ratio */, PathBuf, f64 /* disk_ratio */),
    MemoryAndDiskCap(i64 /* mem_cap */, PathBuf, i64 /* disk_ratio */),
}

impl Default for IaCapacity {
    fn default() -> Self {
        IaCapacity::MemoryRatio(0.2)
    }
}

impl IaCapacity {
    fn build_options(self, options: &mut IaManagerOptions) -> Result<()> {
        match self {
            IaCapacity::Manual {
                small_queue,
                main_queue,
            } => {
                options.small_queue = small_queue;
                options.main_queue = main_queue;
            }
            IaCapacity::MemoryRatio(ratio) => {
                let mem_cap = (SysQuota::memory_limit_in_bytes() as f64 * ratio) as i64;
                Self::set_options_by_mem_cap(mem_cap, options);
            }
            IaCapacity::MemoryCap(cap) => {
                Self::set_options_by_mem_cap(cap, options);
            }
            IaCapacity::MemoryAndDiskRatio(mem_ratio, local_dir, disk_ratio) => {
                let mem_cap = (SysQuota::memory_limit_in_bytes() as f64 * mem_ratio) as i64;
                options.small_queue.path = None;
                options.small_queue.cap = mem_cap;

                let disk_cap = (get_disk_capacity(&local_dir)? as f64 * disk_ratio) as i64;
                options.main_queue.path = Some(local_dir);
                options.main_queue.cap = disk_cap;
            }
            IaCapacity::MemoryAndDiskCap(mem_cap, local_dir, disk_cap) => {
                options.small_queue.path = None;
                options.small_queue.cap = mem_cap;
                options.main_queue.path = Some(local_dir);
                options.main_queue.cap = disk_cap;
            }
        }
        Ok(())
    }

    fn set_options_by_mem_cap(mem_cap: i64, options: &mut IaManagerOptions) {
        options.small_queue.path = None;
        options.small_queue.cap = mem_cap / MAIN_QUEUE_CAPACITY_FACTOR;
        options.main_queue.path = None;
        options.main_queue.cap = mem_cap - options.small_queue.cap;
    }

    #[cfg(any(test, feature = "testexport"))]
    pub fn set_parent_dir(&mut self, parent: PathBuf) {
        match self {
            IaCapacity::Manual {
                small_queue,
                main_queue,
            } => {
                if let Some(ref mut path) = small_queue.path {
                    *path = parent.join(&path);
                }
                if let Some(ref mut path) = main_queue.path {
                    *path = parent.join(&path);
                }
            }
            IaCapacity::MemoryAndDiskRatio(_, dir, _) | IaCapacity::MemoryAndDiskCap(_, dir, _) => {
                *dir = parent.join(&dir);
            }
            IaCapacity::MemoryCap(..) | IaCapacity::MemoryRatio(..) => {}
        }
    }
}

#[derive(Default)]
pub struct IaManagerOptionsBuilder {
    capacity: Option<IaCapacity>,
    segment_size: Option<i64>,
    freq_update_interval: Option<Duration>,
}

impl IaManagerOptionsBuilder {
    pub fn capacity(mut self, capacity: IaCapacity) -> Self {
        self.capacity = Some(capacity);
        self
    }

    pub fn segment_size(mut self, size: i64) -> Self {
        self.segment_size = Some(size);
        self
    }

    pub fn freq_update_interval(mut self, interval: Duration) -> Self {
        self.freq_update_interval = Some(interval);
        self
    }

    pub fn build(mut self) -> Result<IaManagerOptions> {
        let mut options = IaManagerOptions::default();

        let cap = self.capacity.take().unwrap_or_default();
        cap.build_options(&mut options)?;

        options.segment_size = self.segment_size.unwrap_or(FILE_SEGMENT_SIZE_DEF);
        options.freq_update_interval = self.freq_update_interval.unwrap_or(FREQ_UPDATE_INTERVAL);

        Ok(options)
    }
}

fn get_disk_capacity(dir: &Path) -> Result<u64> {
    let sys = Sys::new_all();
    // find the mounted disk of the data dir.
    let mut data_disk = None;
    let mut mount_point_len = 0;
    for disk in sys.disks() {
        let mp = disk.mount_point();
        if dir.starts_with(mp) && mp.len() > mount_point_len {
            data_disk = Some(disk);
            mount_point_len = mp.len();
        }
    }
    data_disk
        .map(|disk| disk.total_space())
        .ok_or_else(|| Error::Io(format!("Unable to find disk for dir: {:?}", dir)))
}

#[cfg(test)]
mod tests {
    use tempfile::TempDir;

    use super::*;

    #[test]
    fn test_split_to_segments() {
        let segment_size = 10;
        let cases = vec![
            // start_off, end_off, total_size, expected
            (0, 1, 5, vec![(0, 5)]),
            (1, 5, 5, vec![(0, 5)]),
            (0, 1, 100, vec![(0, 10)]),
            (0, 1, 100, vec![(0, 10)]),
            (0, 10, 100, vec![(0, 10)]),
            (0, 5, 100, vec![(0, 10)]),
            (0, 15, 100, vec![(0, 10), (10, 20)]),
            (5, 15, 20, vec![(0, 10), (10, 20)]),
            (5, 15, 30, vec![(0, 10), (10, 20)]),
            (5, 25, 28, vec![(0, 10), (10, 20), (20, 28)]),
            (5, 25, 30, vec![(0, 10), (10, 20), (20, 30)]),
            (5, 25, 34, vec![(0, 10), (10, 20), (20, 30)]),
            (5, 25, 35, vec![(0, 10), (10, 20), (20, 30)]),
            (1020, 1021, 1024, vec![(1020, 1024)]),
            (1010, 1021, 1021, vec![(1010, 1020), (1020, 1021)]),
            (1010, 1024, 1024, vec![(1010, 1020), (1020, 1024)]),
            (1010, 1025, 1025, vec![(1010, 1020), (1020, 1025)]),
        ];
        for (start, end, total_size, expected) in cases {
            let expected = expected
                .into_iter()
                .map(|(start_off, end_off)| FileSegmentIdent {
                    file_id: 1,
                    start_off,
                    end_off,
                })
                .collect::<Vec<_>>();
            assert_eq!(
                split_to_segments(1, start, end, total_size, segment_size),
                expected,
            );
        }
    }

    #[test]
    fn test_ia_capacity() {
        let mut options = IaManagerOptions::default();

        let ia_cap = IaCapacity::MemoryRatio(0.2);
        ia_cap.build_options(&mut options).unwrap();
        println!("options for MemoryRatio(0.2): {:?}", options);

        let temp_dir = TempDir::new().unwrap();
        let ia_cap = IaCapacity::MemoryAndDiskRatio(0.1, temp_dir.path().to_path_buf(), 0.1);
        ia_cap.build_options(&mut options).unwrap();
        println!("options for DiskRatio(0.1): {:?}", options);
    }
}
