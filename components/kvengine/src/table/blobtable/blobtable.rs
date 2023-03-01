// Copyright 2023 TiKV Project Authors. Licensed under Apache-2.0.

use std::{cmp, mem, sync::Arc};

use byteorder::{ByteOrder, LittleEndian};
use bytes::{Buf, Bytes};

use super::builder::*;
use crate::table::{
    sstable::{File, LZ4_COMPRESSION, NO_COMPRESSION, ZSTD_COMPRESSION},
    Error, Result,
};

pub struct BlobTable {
    file: Arc<dyn File>,
    footer: BlobFooter,
    smallest_key: Bytes,
    biggest_key: Bytes,
}

impl BlobTable {
    pub fn new(file: Arc<dyn File>) -> Result<Self> {
        let mut footer = BlobFooter::default();
        let size = file.size();
        if size < BLOB_FOOTER_SIZE as u64 {
            return Err(Error::InvalidFileSize);
        }
        let footer_data = file.read(size - BLOB_FOOTER_SIZE as u64, BLOB_FOOTER_SIZE)?;
        footer.unmarshal(&footer_data);
        let props_data = file.read(
            footer.properties_offset as u64,
            footer.properties_len(size as usize),
        )?;
        let mut prop_slice = props_data.chunk();
        let mut smallest_key = Bytes::new();
        let mut biggest_key = Bytes::new();
        while !prop_slice.is_empty() {
            let (key, val, remain) = parse_prop_data(prop_slice);
            prop_slice = remain;
            if key == PROP_KEY_SMALLEST.as_bytes() {
                smallest_key = Bytes::copy_from_slice(val);
            } else if key == PROP_KEY_BIGGEST.as_bytes() {
                biggest_key = Bytes::copy_from_slice(val);
            }
        }
        Ok(Self {
            file,
            footer,
            smallest_key,
            biggest_key,
        })
    }

    pub fn get(&self, offset: u32, size: u32) -> Result<Bytes> {
        let mut data = self.file.read(
            offset as u64,
            size as usize + mem::size_of::<Checksum>() + mem::size_of::<ValueLength>(),
        )?;
        let checksum = LittleEndian::read_u32(&data);
        data.advance(4);
        let val_len = LittleEndian::read_u32(&data);
        data.advance(4);
        assert!(val_len == size);
        let val = data.chunk();
        if self.footer.checksum_type == CRC32C && checksum != crc32c::crc32c(val) {
            return Err(Error::InvalidChecksum("blob checkusm mismatch".to_owned()));
        }

        Ok(Bytes::copy_from_slice(val))
    }

    pub fn decode(&self, data: &[u8], size: u32, decompressed_buf: &mut Vec<u8>) -> Result<bool> {
        let total_data_size =
            size as usize + mem::size_of::<Checksum>() + mem::size_of::<ValueLength>();
        assert!(data.len() >= total_data_size);
        let checksum = LittleEndian::read_u32(data);
        let compressed_len = LittleEndian::read_u32(&data[mem::size_of::<Checksum>()..]);
        assert_eq!(compressed_len, size);
        let compressed_data = &data[mem::size_of::<Checksum>() + mem::size_of::<ValueLength>()..];
        if self.footer.checksum_type == CRC32C && checksum != crc32c::crc32c(compressed_data) {
            return Err(Error::InvalidChecksum("blob checkusm mismatch".to_owned()));
        }
        return match self.footer.compression_type {
            NO_COMPRESSION => Ok(false), // in place decoding
            LZ4_COMPRESSION => unsafe {
                let decompressed = lz4::block::decompress(data, None)?;
                if decompressed_buf.capacity() < decompressed.len() {
                    decompressed_buf.reserve(decompressed.len() - decompressed_buf.capacity());
                }
                decompressed_buf.copy_from_slice(&decompressed);
                decompressed_buf.set_len(decompressed.len());
                Ok(true)
            },
            ZSTD_COMPRESSION => unsafe {
                let capacity = zstd_sys::ZSTD_getFrameContentSize(
                    compressed_data.as_ptr() as *const libc::c_void,
                    compressed_data.len(),
                ) as usize;
                if decompressed_buf.capacity() < capacity {
                    decompressed_buf.reserve(capacity - decompressed_buf.capacity());
                }
                let result = zstd_sys::ZSTD_decompress(
                    decompressed_buf.as_mut_ptr() as *mut libc::c_void,
                    capacity,
                    compressed_data.as_ptr() as *const libc::c_void,
                    compressed_data.len(),
                );
                assert_eq!(zstd_sys::ZSTD_isError(result), 0u32);
                decompressed_buf.set_len(capacity);
                Ok(true)
            },
            _ => panic!("unknown compression type {}", self.footer.compression_type),
        };
    }

    pub fn smallest_biggest_key(&self) -> (&[u8], &[u8]) {
        (self.smallest_key.chunk(), self.biggest_key.chunk())
    }

    pub fn total_blob_size(&self) -> u32 {
        self.footer.total_blob_size
    }
}

fn parse_prop_data(mut prop_data: &[u8]) -> (&[u8], &[u8], &[u8]) {
    let key_len = LittleEndian::read_u16(prop_data) as usize;
    prop_data = &prop_data[2..];
    let key = &prop_data[..key_len];
    prop_data = &prop_data[key_len..];
    let val_len = LittleEndian::read_u32(prop_data) as usize;
    prop_data = &prop_data[4..];
    let val = &prop_data[..val_len];
    let remained = &prop_data[val_len..];
    (key, val, remained)
}

pub struct BlobPrefetcher {
    blob_table: BlobTable,
    raw_buffer: Bytes,
    offset: u32,
    prefetch_size: usize,
    decompressed_buffer: Vec<u8>,
}

impl BlobPrefetcher {
    pub fn new(blob_table: BlobTable, prefetch_size: usize) -> Self {
        Self {
            blob_table,
            raw_buffer: Default::default(),
            offset: 0,
            prefetch_size,
            decompressed_buffer: vec![],
        }
    }

    pub fn get(&mut self, offset: u32, size: u32) -> Result<&[u8]> {
        let meta_size = mem::size_of::<Checksum>() + mem::size_of::<ValueLength>();
        let data_size = size as usize + meta_size;
        if self.raw_buffer.is_empty()
            || !(offset >= self.offset
                && offset + data_size as u32 <= self.offset + self.raw_buffer.len() as u32)
        {
            self.raw_buffer = self.blob_table.file.read(
                offset as u64,
                cmp::min(
                    cmp::max(self.prefetch_size, data_size),
                    self.blob_table.file.size() as usize - offset as usize,
                ),
            )?;
            self.offset = offset;
        }
        let data = &self.raw_buffer[(offset - self.offset) as usize..];
        if self
            .blob_table
            .decode(data, size, &mut self.decompressed_buffer)?
        {
            return Ok(&self.decompressed_buffer);
        }
        Ok(&self.raw_buffer[(offset - self.offset) as usize + meta_size
            ..(offset - self.offset) as usize + data_size])
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::table::{sstable, Value};

    #[test]
    fn test_basic() {
        let mut builder = super::BlobTableBuilder::new(0, 0, 0);
        let mut offsets = Vec::new();
        for i in 0..100 {
            let key_str = format!("key_{}", i);
            let val_str = format!("val_{}", i);
            let val_buf = Value::encode_buf(b'A', &[0], 0, val_str.as_bytes());
            let (offset, _) = builder.add(key_str.as_bytes(), Value::decode(val_buf.as_slice()));
            offsets.push(offset);
        }
        let file = sstable::InMemFile::new(1, builder.finish());
        let table = super::BlobTable::new(Arc::new(file)).unwrap();
        for i in 0..100 {
            let expected_val = format!("val_{}", i);
            let val = table.get(offsets[i], expected_val.len() as u32).unwrap();
            assert_eq!(val, expected_val.as_bytes());
        }
        assert_eq!(table.smallest_key, format!("key_{}", 0).as_bytes());
        assert_eq!(table.biggest_key, format!("key_{}", 99).as_bytes());
    }

    #[test]
    fn test_prefetcher() {
        let mut builder = super::BlobTableBuilder::new(0, 0, 0);
        let mut offsets = Vec::new();
        for i in 0..100 {
            let key_str = format!("key_{}", i);
            let val_str = format!("val_{}", i);
            let val_buf = Value::encode_buf(b'A', &[0], 0, val_str.as_bytes());
            let (offset, _) = builder.add(key_str.as_bytes(), Value::decode(val_buf.as_slice()));
            offsets.push(offset);
        }
        let file = sstable::InMemFile::new(1, builder.finish());
        let table = super::BlobTable::new(Arc::new(file)).unwrap();
        let mut prefetcher = super::BlobPrefetcher::new(table, 1000);
        for i in 0..100 {
            let expected_val = format!("val_{}", i);
            let val = prefetcher
                .get(offsets[i], expected_val.len() as u32)
                .unwrap();
            assert_eq!(val, expected_val.as_bytes());
        }
    }
}
