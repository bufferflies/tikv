// Copyright 2023 TiKV Project Authors. Licensed under Apache-2.0.

use std::{mem, slice};

use byteorder::{ByteOrder, LittleEndian};
use bytes::{BufMut, Bytes, BytesMut};

use crate::table::{
    sstable::{LZ4_COMPRESSION, NO_COMPRESSION, ZSTD_COMPRESSION},
    Value,
};

pub type ValueLength = u32; // Max value length is 4GB
pub type BlobOffset = u32; // Max blob file size is 4GB
pub type Checksum = u32;

pub const BLOB_FORMAT_V1: u16 = 1;
pub const BLOB_MAGIC_NUMBER: u32 = 0xdeadbeef;
pub const CRC32C: u8 = 1;
pub const PROP_KEY_BIGGEST: &str = "biggest_key";
pub const PROP_KEY_SMALLEST: &str = "smallest_key";

// Blob file format:
//
// +---------------------------------+
// |          blob record 1          |
// +---------------------------------+
// |          blob record 2          |
// +---------------------------------+
// |             ...                 |
// +---------------------------------+
// |          blob record n          |
// +---------------------------------+
// |         blob properties         |
// +---------------------------------+
// |           blob footer           |
// +---------------------------------+
//
// Blob record format:
//
// +---------------------------------+
// |          checksum: u32          |
// +---------------------------------+
// |        value length: u32        |
// +---------------------------------+
// |          value: bytes           |
// +---------------------------------+

#[repr(C)]
#[derive(Default, Clone, Copy)]
pub struct BlobFooter {
    pub properties_offset: u32,
    pub version: u64,
    pub total_blob_size: u32,
    pub compression_type: u8,
    pub checksum_type: u8,
    pub blob_format_version: u16,
    pub magic: u32,
}

pub const BLOB_FOOTER_SIZE: usize = mem::size_of::<BlobFooter>();

impl BlobFooter {
    pub fn data_len(&self) -> usize {
        self.properties_offset as usize
    }

    pub fn properties_len(&self, table_size: usize) -> usize {
        table_size - BLOB_FOOTER_SIZE - self.properties_offset as usize
    }

    pub fn unmarshal(&mut self, data: &[u8]) {
        let footer_ptr = data.as_ptr() as *const BlobFooter;
        *self = unsafe { *footer_ptr };
    }

    pub fn marshal(&self) -> &[u8] {
        let footer_ptr = self as *const BlobFooter as *const u8;
        unsafe { slice::from_raw_parts(footer_ptr, BLOB_FOOTER_SIZE) }
    }
}

#[derive(Default)]
pub struct BlobTableBuilder {
    fid: u64,
    buf: Vec<u8>,
    checksum_tp: u8,
    compression_tp: u8,
    compression_lvl: i32,
    total_blob_size: u32,
    smallest_key: Vec<u8>,
    last_key: Vec<u8>,
}

impl BlobTableBuilder {
    pub fn new(fid: u64, checksum_tp: u8, compression_tp: u8, compression_lvl: i32) -> Self {
        Self {
            fid,
            buf: vec![],
            checksum_tp,
            compression_tp,
            compression_lvl,
            total_blob_size: 0,
            smallest_key: vec![],
            last_key: vec![],
        }
    }

    pub fn reset(&mut self, fid: u64) {
        self.fid = fid;
        self.buf.clear();
        self.total_blob_size = 0;
        self.smallest_key.clear();
        self.last_key.clear();
        self.smallest_key.clear();
    }

    pub fn add(&mut self, key: &[u8], value: &Value) -> (BlobOffset, ValueLength) {
        if value.value_len() > BlobOffset::max_value() as usize {
            panic!(
                "value length {} exceeds max value length {}",
                value.value_len(),
                BlobOffset::max_value()
            );
        }
        if self.total_blob_size as usize + value.value_len() > BlobOffset::max_value() as usize {
            panic!(
                "total blob size {} exceeds max blob size {}",
                self.total_blob_size as usize + value.value_len(),
                BlobOffset::max_value()
            );
        }
        if self.smallest_key.is_empty() {
            self.smallest_key = key.to_vec();
        }
        if !key.eq(&self.last_key) {
            // TODO: move old versions to a different section.
            self.last_key = key.to_vec();
        }
        self.total_blob_size += value.value_len() as BlobOffset;

        let begin_off = self.buf.len();
        self.buf.resize(
            self.buf.len() + mem::size_of::<Checksum>() + mem::size_of::<ValueLength>(),
            0,
        );
        let compressed_len = match self.compression_tp {
            NO_COMPRESSION => {
                self.buf.extend_from_slice(value.get_value());
                value.value_len()
            }
            LZ4_COMPRESSION => Self::compress_lz4(value.get_value(), &mut self.buf),
            ZSTD_COMPRESSION => {
                Self::compress_zstd(value.get_value(), self.compression_lvl, &mut self.buf)
            }
            _ => panic!("unexpected compression type {}", self.compression_tp),
        };

        self.buf.put_u32_le(value.value_len() as ValueLength);
        self.buf.extend_from_slice(value.get_value());
        let mut checksum = 0u32;
        if self.checksum_tp == CRC32C {
            checksum = crc32c::crc32c(
                &self.buf
                    [(begin_off + mem::size_of::<Checksum>() + mem::size_of::<ValueLength>())..],
            );
        }
        let slice = self.buf.as_mut_slice();
        LittleEndian::write_u32(&mut slice[begin_off..], checksum); // put checksum at the reserved place.
        LittleEndian::write_u32(
            &mut slice[begin_off + mem::size_of::<Checksum>()..],
            compressed_len as ValueLength,
        ); // put compressed length at the reserved place.
        (begin_off as BlobOffset, compressed_len as ValueLength)
    }

    fn compress_lz4(uncompressed: &[u8], compressed_buf: &mut Vec<u8>) -> usize {
        unsafe {
            let uncompressed_len = uncompressed.len() as i32;
            let compress_bound = lz4::liblz4::LZ4_compressBound(uncompressed_len);
            let original_len = compressed_buf.len();
            compressed_buf.resize(original_len + compress_bound as usize, 0);
            let dst = &mut compressed_buf[original_len..];
            let size = lz4::liblz4::LZ4_compress_default(
                uncompressed.as_ptr() as *const libc::c_char,
                dst.as_mut_ptr() as *mut libc::c_char,
                uncompressed_len,
                compress_bound,
            ) as usize;
            compressed_buf.set_len(original_len + size);
            size
        }
    }

    fn compress_zstd(
        uncompressed: &[u8],
        compression_lvl: i32,
        compressed_buf: &mut Vec<u8>,
    ) -> usize {
        unsafe {
            let uncompressed_len = uncompressed.len();
            let compress_bound = zstd_sys::ZSTD_compressBound(uncompressed_len);
            let original_len = compressed_buf.len();
            compressed_buf.resize(original_len + compress_bound, 0);
            let dst = &mut compressed_buf[original_len..];
            let size = zstd_sys::ZSTD_compress(
                dst.as_mut_ptr() as *mut libc::c_void,
                compress_bound,
                uncompressed.as_ptr() as *const libc::c_void,
                uncompressed_len,
                compression_lvl as libc::c_int,
            );
            compressed_buf.set_len(original_len + size);
            size
        }
    }

    fn add_property(buf: &mut BytesMut, key: &[u8], val: &[u8]) {
        buf.put_u16_le(key.len() as u16);
        buf.put_slice(key);
        buf.put_u32_le(val.len() as u32);
        buf.put_slice(val);
    }

    // Smallest and biggest key are used to indicate the key range of the blob file.
    pub fn finish(&self) -> Bytes {
        let mut buf = BytesMut::from(&self.buf[..]);

        let properties_offset = buf.len();
        BlobTableBuilder::add_property(&mut buf, PROP_KEY_SMALLEST.as_bytes(), &self.smallest_key);
        BlobTableBuilder::add_property(&mut buf, PROP_KEY_BIGGEST.as_bytes(), &self.last_key);

        let mut footer = BlobFooter::default();
        footer.properties_offset = properties_offset as u32;
        footer.total_blob_size = self.total_blob_size;
        footer.compression_type = self.compression_tp;
        footer.checksum_type = self.checksum_tp;
        footer.blob_format_version = BLOB_FORMAT_V1;
        footer.magic = BLOB_MAGIC_NUMBER;
        buf.extend_from_slice(footer.marshal());
        buf.freeze()
    }

    pub fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }

    pub fn get_fid(&self) -> u64 {
        self.fid
    }

    pub fn smallest_biggest_key(&self) -> (&[u8], &[u8]) {
        (&self.smallest_key, &self.last_key)
    }
}
