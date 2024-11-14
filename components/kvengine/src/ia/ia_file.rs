// Copyright 2024 TiKV Project Authors. Licensed under Apache-2.0.

use std::fmt;

use async_trait::async_trait;
use bytes::{Bytes, BytesMut};

use crate::{
    dfs::FileType,
    ia::manager::{IaManager, ReadAt},
    table::{
        file::{File, MmapData},
        Error, Result,
    },
};

pub struct IaFile {
    pub(crate) id: u64,
    pub(crate) size: u64,
    pub(crate) ftype: FileType,
    pub(crate) footer: Bytes,
    pub(crate) mgr: IaManager,
}

impl fmt::Debug for IaFile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("IaFile")
            .field("id", &self.id)
            .field("size", &self.size)
            .field("ftype", &self.ftype)
            .field("footer_len", &self.footer.len())
            .finish()
    }
}

impl IaFile {
    pub async fn open(id: u64, ftype: FileType, mgr: &IaManager) -> Result<Self> {
        mgr.open_file(id, ftype).await
    }
}

#[async_trait]
impl File for IaFile {
    fn id(&self) -> u64 {
        self.id
    }

    fn size(&self) -> u64 {
        self.size
    }

    fn is_sync(&self) -> bool {
        false
    }

    fn read(&self, _off: u64, _length: usize) -> Result<Bytes> {
        unimplemented!()
    }

    fn read_at(&self, _buf: &mut [u8], _offset: u64) -> Result<()> {
        unimplemented!()
    }

    fn read_footer(&self, length: usize) -> Result<Bytes> {
        if self.footer.len() < length {
            return Err(Error::IaMgr(format!(
                "invalid footer length, expect {}, got {}, file_id {}",
                length,
                self.footer.len(),
                self.id,
            )));
        }
        Ok(self.footer.slice(self.footer.len() - length..))
    }

    async fn read_async(&self, off: u64, length: usize) -> Result<Bytes> {
        let end_off = off + length as u64;
        if end_off == self.size && length <= self.footer.len() {
            return self.read_footer(length);
        }

        let mut buf = BytesMut::new();
        buf.resize(length, 0);
        let read_at = ReadAt::new(buf.as_mut(), off);
        self.mgr
            .read_range(self.id, self.ftype, self.size, read_at)
            .await?;
        Ok(buf.freeze())
    }

    async fn read_at_async(&self, buf: &mut [u8], offset: u64) -> Result<()> {
        let buf_len = buf.len();
        let end_off = offset + buf_len as u64;
        if end_off == self.size && buf_len <= self.footer.len() {
            buf.copy_from_slice(&self.read_footer(buf_len)?);
            return Ok(());
        }

        let read_at = ReadAt::new(buf, offset);
        self.mgr
            .read_range(self.id, self.ftype, self.size, read_at)
            .await
    }

    fn mmap(&self) -> Result<MmapData> {
        unimplemented!()
    }
}
