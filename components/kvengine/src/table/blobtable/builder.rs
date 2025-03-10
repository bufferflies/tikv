// Copyright 2023 TiKV Project Authors. Licensed under Apache-2.0.

use std::{mem, ops::Deref};

use byteorder::{ByteOrder, LittleEndian};
use bytes::{Buf, BufMut, Bytes, BytesMut};
use cloud_encryption::EncryptionKey;

use super::BlobRef;
use crate::table::{
    sstable::PROP_KEY_ENCRYPTION_VER, ChecksumType, InnerKey, Value, LZ4_COMPRESSION,
    NO_COMPRESSION, ZSTD_COMPRESSION,
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
    pub padding: u32,
    pub version: u64,
    pub total_blob_size: u64,
    pub compression_type: u8,
    pub checksum_type: u8,
    pub blob_format_version: u16,
    pub compression_lvl: i32,
    pub min_blob_size: u32,
    pub magic: u32,
}

pub const BLOB_TABLE_FOOTER_SIZE: usize = mem::size_of::<BlobFooter>();
pub const BLOB_ENTRY_META_SIZE: usize = mem::size_of::<Checksum>() + mem::size_of::<ValueLength>();
pub const BLOB_ENTRY_LENGTH_OFFSET: usize = mem::size_of::<Checksum>();
pub const BLOB_ENTRY_VALUE_OFFSET: usize = BLOB_ENTRY_META_SIZE;

impl BlobFooter {
    pub fn properties_len(&self, table_size: usize) -> usize {
        table_size - BLOB_TABLE_FOOTER_SIZE - self.properties_offset as usize
    }

    pub fn unmarshal(&mut self, mut data: &[u8]) {
        self.properties_offset = data.get_u32_le();
        self.padding = data.get_u32_le();
        self.version = data.get_u64_le();
        self.total_blob_size = data.get_u64_le();
        self.compression_type = data.get_u8();
        self.checksum_type = data.get_u8();
        self.blob_format_version = data.get_u16_le();
        self.compression_lvl = data.get_i32_le();
        self.min_blob_size = data.get_u32_le();
        self.magic = data.get_u32_le();
    }

    pub fn marshal(&self, buf: &mut BytesMut) {
        buf.put_u32_le(self.properties_offset);
        buf.put_u32_le(self.padding);
        buf.put_u64_le(self.version);
        buf.put_u64_le(self.total_blob_size);
        buf.put_u8(self.compression_type);
        buf.put_u8(self.checksum_type);
        buf.put_u16_le(self.blob_format_version);
        buf.put_i32_le(self.compression_lvl);
        buf.put_u32_le(self.min_blob_size);
        buf.put_u32_le(self.magic);
    }

    pub fn is_match(&self) -> bool {
        self.magic == BLOB_MAGIC_NUMBER
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq)]
#[serde(default)]
#[serde(rename_all = "kebab-case")]
pub struct BlobTableBuildOptions {
    pub compression_type: u8,
    pub min_blob_size: u32,
    pub max_blob_table_size: usize,
    // Do not bother creating a blob table if the target blob table size is not reached.
    pub target_blob_table_size: usize,
}

impl Default for BlobTableBuildOptions {
    fn default() -> Self {
        Self {
            compression_type: LZ4_COMPRESSION,
            min_blob_size: 4096,
            max_blob_table_size: 64 * 1024 * 1024,
            target_blob_table_size: 2 * 1024 * 1024,
        }
    }
}

#[derive(Default)]
pub struct BlobTableBuilder {
    fid: u64,
    buf: Vec<u8>,
    checksum_tp: u8,
    compression_tp: u8,
    compression_lvl: i32,
    min_blob_size: u32,
    total_blob_size: u64,
    smallest_key: Vec<u8>,
    biggest_key: Vec<u8>,
    encryption_key: Option<EncryptionKey>,
    compressed_buf: Vec<u8>,
}

impl BlobTableBuilder {
    pub fn new(
        fid: u64,
        compression_tp: u8,
        compression_lvl: i32,
        min_blob_size: u32,
        encryption_key: Option<EncryptionKey>,
    ) -> Self {
        Self {
            fid,
            buf: vec![],
            checksum_tp: 0,
            compression_tp,
            compression_lvl,
            min_blob_size,
            total_blob_size: 0,
            smallest_key: vec![],
            biggest_key: vec![],
            encryption_key,
            compressed_buf: vec![],
        }
    }

    pub fn new_with_checksum_type(
        fid: u64,
        compression_tp: u8,
        compression_lvl: i32,
        min_blob_size: u32,
        encryption_key: Option<EncryptionKey>,
        checksum_type: ChecksumType,
    ) -> Self {
        Self {
            fid,
            buf: vec![],
            checksum_tp: checksum_type as u8,
            compression_tp,
            compression_lvl,
            min_blob_size,
            total_blob_size: 0,
            smallest_key: vec![],
            biggest_key: vec![],
            encryption_key,
            compressed_buf: vec![],
        }
    }

    pub fn reset(&mut self, fid: u64) {
        self.fid = fid;
        self.buf.clear();
        self.total_blob_size = 0;
        self.smallest_key.clear();
        self.biggest_key.clear();
        self.smallest_key.clear();
        self.compressed_buf.clear();
    }

    pub fn add_blob(
        &mut self,
        inner_key: InnerKey<'_>,
        blob: &[u8],
        already_compressed: Option<ValueLength>,
        need_encrypt: bool,
    ) -> BlobRef {
        let key = inner_key.deref();
        assert!(blob.len() <= ValueLength::max_value() as usize);
        assert!(self.total_blob_size as usize + blob.len() <= BlobOffset::max_value() as usize);
        if self.smallest_key.is_empty() || self.smallest_key.as_slice() > key {
            self.smallest_key.clear();
            self.smallest_key.extend_from_slice(key);
        }
        if self.biggest_key.is_empty() || self.biggest_key.as_slice() < key {
            self.biggest_key.clear();
            self.biggest_key.extend_from_slice(key);
        }
        self.total_blob_size += blob.len() as u64;

        let begin_off = self.buf.len();
        self.buf.resize(self.buf.len() + BLOB_ENTRY_META_SIZE, 0);
        let mut original_len = blob.len() as ValueLength;

        let compressed_blob = if let Some(len) = already_compressed {
            original_len = len;
            blob
        } else {
            match self.compression_tp {
                NO_COMPRESSION => blob,
                LZ4_COMPRESSION => {
                    self.compressed_buf.clear();
                    let _ = Self::compress_lz4(blob, &mut self.compressed_buf);
                    &self.compressed_buf
                }
                ZSTD_COMPRESSION => {
                    self.compressed_buf.clear();
                    let _ =
                        Self::compress_zstd(blob, self.compression_lvl, &mut self.compressed_buf);
                    &self.compressed_buf
                }
                _ => panic!("unexpected compression type {}", self.compression_tp),
            }
        };
        if let Some(encryption_key) = &self.encryption_key {
            if need_encrypt {
                encryption_key.encrypt(compressed_blob, self.fid, begin_off as u32, &mut self.buf);
            } else {
                self.buf.extend_from_slice(compressed_blob);
            }
        } else {
            self.buf.extend_from_slice(compressed_blob);
        }
        let data_len = self.buf.len() - begin_off - BLOB_ENTRY_VALUE_OFFSET;
        let checksum = ChecksumType::from(self.checksum_tp)
            .checksum(&self.buf[(begin_off + BLOB_ENTRY_VALUE_OFFSET)..]);
        let slice = self.buf.as_mut_slice();
        LittleEndian::write_u32(&mut slice[begin_off..], checksum); // put checksum at the reserved place.
        LittleEndian::write_u32(
            &mut slice[begin_off + BLOB_ENTRY_LENGTH_OFFSET..],
            data_len as ValueLength,
        ); // put compressed length at the reserved place.
        BlobRef::new(
            self.fid,
            begin_off as BlobOffset,
            data_len as u32,
            original_len,
        )
    }

    pub fn add(&mut self, key: InnerKey<'_>, value: &Value) -> BlobRef {
        self.add_blob(key, value.get_value(), None, true)
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
        BlobTableBuilder::add_property(&mut buf, PROP_KEY_BIGGEST.as_bytes(), &self.biggest_key);
        if let Some(encryption_key) = &self.encryption_key {
            BlobTableBuilder::add_property(
                &mut buf,
                PROP_KEY_ENCRYPTION_VER.as_bytes(),
                &encryption_key.current_ver.to_le_bytes(),
            )
        }

        let mut footer = BlobFooter::default();
        footer.properties_offset = properties_offset as u32;
        footer.version = BLOB_FORMAT_V1 as u64;
        footer.total_blob_size = self.total_blob_size;
        footer.compression_type = self.compression_tp;
        footer.checksum_type = self.checksum_tp;
        footer.blob_format_version = BLOB_FORMAT_V1;
        footer.compression_lvl = self.compression_lvl;
        footer.min_blob_size = self.min_blob_size;
        footer.magic = BLOB_MAGIC_NUMBER;
        footer.marshal(&mut buf);
        buf.freeze()
    }

    pub fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }

    pub fn get_fid(&self) -> u64 {
        self.fid
    }

    pub fn smallest_biggest_key(&self) -> (&[u8], &[u8]) {
        (&self.smallest_key, &self.biggest_key)
    }

    pub fn total_blob_size(&self) -> u64 {
        self.total_blob_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::table::Value;

    #[test]
    fn test_blob_table_builder_getters_and_reset() {
        // Create builder with specific properties
        let mut builder = BlobTableBuilder::new(
            42,               // file id
            ZSTD_COMPRESSION, // compression type
            3,                // compression level
            1024,             // min blob size
            None,             // encryption key
        );

        // Add test data
        let test_data = vec![b'x'; 2048];
        let encoded = Value::encode_buf(0, &[0], 0, &test_data);
        let value = Value::decode(encoded.as_slice());
        let small_key = InnerKey::from_inner_buf(b"aaa_key");
        let big_key = InnerKey::from_inner_buf(b"zzz_key");

        builder.add(big_key, &value);
        builder.add(small_key, &value);

        // Test getters
        assert_eq!(
            builder.get_fid(),
            42,
            "get_fid() should return correct file id"
        );
        assert_eq!(
            builder.total_blob_size(),
            test_data.len() as u64 * 2,
            "total_blob_size() should reflect sum of all blob sizes"
        );
        assert!(
            !builder.is_empty(),
            "is_empty() should return false after adding data"
        );

        // Test boundary keys
        let (smallest, biggest) = builder.smallest_biggest_key();
        assert_eq!(
            smallest,
            small_key.deref(),
            "smallest_key should be lexicographically smallest"
        );
        assert_eq!(
            biggest,
            big_key.deref(),
            "biggest_key should be lexicographically biggest"
        );

        // Test reset
        builder.reset(43);

        // Verify state after reset
        assert_eq!(
            builder.get_fid(),
            43,
            "get_fid() should return new file id after reset"
        );
        assert_eq!(
            builder.total_blob_size(),
            0,
            "total_blob_size() should be 0 after reset"
        );
        assert!(
            builder.is_empty(),
            "is_empty() should return true after reset"
        );

        let (smallest, biggest) = builder.smallest_biggest_key();
        assert!(
            smallest.is_empty(),
            "smallest_key should be empty after reset"
        );
        assert!(
            biggest.is_empty(),
            "biggest_key should be empty after reset"
        );
    }
}
