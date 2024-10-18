// Copyright 2024 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering::Relaxed},
};

use async_trait::async_trait;
use bytes::{Buf, Bytes};
use dashmap::DashMap;
use tokio::{
    fs,
    io::{AsyncReadExt, AsyncSeekExt, AsyncWriteExt, SeekFrom},
};

use crate::{
    table::{Error, Result},
    IoContext,
};

/// LocalStore is used to provide uniform access to both local disk and memory.
#[async_trait]
pub(crate) trait LocalStore: Send + Sync {
    /// Return the path of store. Will be `None` when it's in memory.
    fn path(&self) -> Option<&Path>;

    /// Return the existed keys & suffixes in the store from last startup.
    async fn init(&self) -> Result<HashMap<String /* suffix */, Vec<String> /* keys */>>;

    async fn save(&self, file_id: u64, key: &str, data: Bytes) -> Result<()>;

    /// Read exactly `buf.len()` bytes into `buf` starting at `offset`.
    async fn read_at(
        &self,
        file_id: u64,
        key: &str,
        buf: &mut [u8],
        offset: u64,
        on_open: Option<Box<dyn FnOnce() + Send>>,
    ) -> Result<()>;

    async fn read(
        &self,
        file_id: u64,
        key: &str,
        start_off: u64,
        end_off: u64,
        on_open: Option<Box<dyn FnOnce() + Send>>,
    ) -> Result<Bytes>;

    async fn read_all(
        &self,
        file_id: u64,
        key: &str,
        buf: &mut Vec<u8>,
        on_open: Option<Box<dyn FnOnce() + Send>>,
    ) -> Result<()>;

    async fn remove(&self, file_id: u64, key: &str) -> Result<()>;
}

pub(crate) struct LocalFileStore {
    dir: PathBuf,
}

impl LocalFileStore {
    pub(crate) fn new(dir: PathBuf) -> Self {
        Self { dir }
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
}

#[async_trait]
impl LocalStore for LocalFileStore {
    fn path(&self) -> Option<&Path> {
        Some(&self.dir)
    }

    async fn init(&self) -> Result<HashMap<String /* suffix */, Vec<String> /* keys */>> {
        fs::create_dir_all(&self.dir)
            .await
            .table_ctx(0, format!("create_dir.{:?}", self.dir))?;
        self.scan().await
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
        on_open: Option<Box<dyn FnOnce() + Send>>,
    ) -> Result<()> {
        let path = self.dir.join(key);
        debug!("FileDataStore.read_at"; "file_id" => file_id, "key" => key, "path" => ?path);
        let f = fs::File::open(&path)
            .await
            .table_ctx(file_id, format!("open.{key}"));
        if let Some(cb) = on_open {
            cb();
        }

        let mut f = f?;
        if offset > 0 {
            f.seek(SeekFrom::Start(offset))
                .await
                .table_ctx(file_id, format!("seek.{key}"))?;
        }
        f.read_exact(buf)
            .await
            .table_ctx(file_id, format!("read_exact.{key}"))?;
        Ok(())
    }

    async fn read(
        &self,
        file_id: u64,
        key: &str,
        start_off: u64,
        end_off: u64,
        on_open: Option<Box<dyn FnOnce() + Send>>,
    ) -> Result<Bytes> {
        let mut buf = vec![0; (end_off - start_off) as usize];
        self.read_at(file_id, key, &mut buf, start_off, on_open)
            .await?;
        Ok(Bytes::from(buf))
    }

    async fn read_all(
        &self,
        file_id: u64,
        key: &str,
        buf: &mut Vec<u8>,
        on_open: Option<Box<dyn FnOnce() + Send>>,
    ) -> Result<()> {
        let path = self.dir.join(key);
        let f = fs::File::open(&path)
            .await
            .table_ctx(file_id, format!("open.{key}"));
        if let Some(cb) = on_open {
            cb();
        }

        let mut f = f?;
        f.read_to_end(buf)
            .await
            .table_ctx(file_id, format!("read_to_end.{key}"))?;
        Ok(())
    }

    async fn remove(&self, file_id: u64, key: &str) -> Result<()> {
        let path = self.dir.join(key);
        fs::remove_file(&path)
            .await
            .table_ctx(file_id, format!("remove.{key}"))?;
        Ok(())
    }
}

#[derive(Default)]
pub(crate) struct LocalMemoryStore {
    m: DashMap<String, Bytes>,
}

impl LocalMemoryStore {
    fn get(&self, key: &str) -> Result<Bytes> {
        if let Some(data) = self.m.get(key) {
            Ok(data.clone())
        } else {
            Err(Error::Io(format!("key not found in store: {}", key)))
        }
    }

    fn get_with_check(&self, key: &str, end_off: u64) -> Result<Bytes> {
        let data = self.get(key)?;
        if end_off > data.len() as u64 {
            return Err(Error::Io(format!(
                "read out of range, key: {}, end_off: {}, data length: {}",
                key,
                end_off,
                data.len()
            )));
        }
        Ok(data)
    }
}

#[async_trait]
impl LocalStore for LocalMemoryStore {
    fn path(&self) -> Option<&Path> {
        None
    }

    async fn init(&self) -> Result<HashMap<String /* suffix */, Vec<String> /* keys */>> {
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
        on_open: Option<Box<dyn FnOnce() + Send>>,
    ) -> Result<()> {
        let end_off = offset + buf.len() as u64;
        let data = self.get_with_check(key, end_off);
        if let Some(cb) = on_open {
            cb();
        }

        let data = data?;
        buf.copy_from_slice(&data[offset as usize..end_off as usize]);
        Ok(())
    }

    async fn read(
        &self,
        _file_id: u64,
        key: &str,
        start_off: u64,
        end_off: u64,
        on_open: Option<Box<dyn FnOnce() + Send>>,
    ) -> Result<Bytes> {
        let data = self.get_with_check(key, end_off);
        if let Some(cb) = on_open {
            cb();
        }

        let data = data?;
        Ok(Bytes::copy_from_slice(
            data.slice(start_off as usize..end_off as usize).chunk(),
        ))
    }

    async fn read_all(
        &self,
        _file_id: u64,
        key: &str,
        buf: &mut Vec<u8>,
        on_open: Option<Box<dyn FnOnce() + Send>>,
    ) -> Result<()> {
        let data = self.m.get(key).map(|r| r.value().clone());
        if let Some(cb) = on_open {
            cb();
        }

        let data = data.ok_or_else(|| Error::Io(format!("key not found in store: {}", key)))?;
        buf.extend_from_slice(&data);
        Ok(())
    }

    async fn remove(&self, _file_id: u64, key: &str) -> Result<()> {
        self.m.remove(key);
        Ok(())
    }
}
