// Copyright 2023 TiKV Project Authors. Licensed under Apache-2.0.
use std::{cmp, collections::HashMap, sync::Arc};

use byteorder::{ByteOrder, LittleEndian};
use bytes::{Buf, Bytes};
use cloud_encryption::EncryptionKey;
use kvenginepb::BlobCreate;

use super::{builder::*, BlobRef};
use crate::{
    error::IoContext,
    table::{
        file::File, sstable::PROP_KEY_ENCRYPTION_VER, BoundedDataSet, ChecksumType, DataBound,
        Error, InnerKey, Result, LZ4_COMPRESSION, NO_COMPRESSION, ZSTD_COMPRESSION,
    },
};

#[derive(Clone)]
pub struct BlobTable {
    file: Option<Arc<dyn File>>,
    preloaded_data: Option<Bytes>,
    footer: BlobFooter,
    smallest_key: Bytes,
    biggest_key: Bytes,
    pub(crate) encryption_ver: u32,
}

impl BlobTable {
    pub fn new(file: Arc<dyn File>) -> Result<Self> {
        let mut footer = BlobFooter::default();
        let size = file.size();
        let footer_data = file.read_footer(Self::footer_size())?;
        footer.unmarshal(&footer_data);
        let props_data = file.read(
            footer.properties_offset as u64,
            footer.properties_len(size as usize),
        )?;
        let mut prop_slice = props_data.chunk();
        let mut smallest_key = Bytes::new();
        let mut biggest_key = Bytes::new();
        let mut encryption_ver = 0;
        while !prop_slice.is_empty() {
            let (key, val, remain) = parse_prop_data(prop_slice);
            prop_slice = remain;
            if key == PROP_KEY_SMALLEST {
                smallest_key = Bytes::copy_from_slice(val);
            } else if key == PROP_KEY_BIGGEST {
                biggest_key = Bytes::copy_from_slice(val);
            } else if key == PROP_KEY_ENCRYPTION_VER {
                encryption_ver = LittleEndian::read_u32(val);
            }
        }
        Ok(Self {
            file: Some(file),
            preloaded_data: None,
            footer,
            smallest_key,
            biggest_key,
            encryption_ver,
        })
    }

    pub fn from_bytes(bytes: Bytes) -> Result<Self> {
        let mut footer = BlobFooter::default();
        let size = bytes.len();
        if size < BLOB_TABLE_FOOTER_SIZE {
            return Err(Error::InvalidFileSize);
        }
        let footer_data = &bytes[size - BLOB_TABLE_FOOTER_SIZE..size];
        footer.unmarshal(footer_data);
        let mut props_data = &bytes[footer.properties_offset as usize
            ..footer.properties_offset as usize + footer.properties_len(size)];
        let mut smallest_key = Bytes::new();
        let mut biggest_key = Bytes::new();
        let mut encryption_ver = 0;
        while !props_data.is_empty() {
            let (key, val, remain) = parse_prop_data(props_data);
            props_data = remain;
            if key == PROP_KEY_SMALLEST {
                smallest_key = Bytes::copy_from_slice(val);
            } else if key == PROP_KEY_BIGGEST {
                biggest_key = Bytes::copy_from_slice(val);
            } else if key == PROP_KEY_ENCRYPTION_VER {
                encryption_ver = LittleEndian::read_u32(val);
            }
        }
        Ok(Self {
            file: None,
            preloaded_data: Some(bytes),
            footer,
            smallest_key,
            biggest_key,
            encryption_ver,
        })
    }

    pub fn get(
        &self,
        blob_ref: &BlobRef,
        decryption_buf: &mut Vec<u8>,
        encryption_key: Option<EncryptionKey>,
    ) -> Result<Vec<u8>> {
        let data = self
            .file
            .as_ref()
            .unwrap_or_else(|| panic!("file is not set"))
            .read(
                blob_ref.offset as u64,
                blob_ref.len as usize + BLOB_ENTRY_META_SIZE,
            )?;
        let meta_slice = &data.chunk()[..BLOB_ENTRY_META_SIZE];
        let mut data_slice = &data.chunk()[BLOB_ENTRY_META_SIZE..];
        if let Some(encryption_key) = &encryption_key {
            decryption_buf.clear();
            encryption_key.decrypt(
                data_slice,
                self.id(),
                blob_ref.offset,
                self.encryption_ver,
                decryption_buf,
            );
            data_slice = decryption_buf;
        }
        if self.footer.compression_type != NO_COMPRESSION {
            let mut decompressed = Vec::with_capacity(blob_ref.original_len as usize);
            self.decompress(
                meta_slice,
                data_slice,
                blob_ref.len,
                blob_ref.original_len,
                &mut decompressed,
            )?;
            Ok(decompressed)
        } else {
            Ok(data_slice.to_vec())
        }
    }

    // If data is compressed and the caller want it to be decpressed, the buf will
    // be used to store decompressed data, and the returned slice is a reference
    // to the decompressed data. Otherwise, the returned slice is a reference to
    // the original data.
    pub fn get_from_preloaded<'a>(
        &'a self,
        blob_ref: &BlobRef,
        need_decompress: bool,
        need_decrypt: bool,
        buf: &'a mut Vec<u8>,
        decryption_buf: &'a mut Vec<u8>,
        encryption_key: Option<EncryptionKey>,
    ) -> Result<&[u8]> {
        let data = &self.preloaded_data.as_ref().unwrap()[blob_ref.offset as usize
            ..blob_ref.offset as usize + BLOB_ENTRY_VALUE_OFFSET + blob_ref.len as usize];
        let meta_slice = &data[..BLOB_ENTRY_META_SIZE];
        let mut data_slice = &data[BLOB_ENTRY_META_SIZE..];
        if let Some(encryption_key) = &encryption_key {
            if need_decrypt {
                decryption_buf.clear();
                encryption_key.decrypt(
                    data_slice,
                    self.id(),
                    blob_ref.offset,
                    self.encryption_ver,
                    decryption_buf,
                );
                data_slice = decryption_buf;
            }
        }
        if need_decompress
            && self.decompress(
                meta_slice,
                data_slice,
                blob_ref.len,
                blob_ref.original_len,
                buf,
            )?
        {
            Ok(buf.as_slice())
        } else {
            Ok(data_slice)
        }
    }

    pub fn decompress(
        &self,
        meta_slice: &[u8],
        compressed_data: &[u8],
        size: u32,
        original_len: u32,
        decompressed_buf: &mut Vec<u8>,
    ) -> Result<bool> {
        let checksum = LittleEndian::read_u32(meta_slice);
        let compressed_len = LittleEndian::read_u32(&meta_slice[BLOB_ENTRY_LENGTH_OFFSET..]);
        assert_eq!(compressed_len, size);
        let got_checksum = ChecksumType::from(self.footer.checksum_type).checksum(compressed_data);
        if checksum != got_checksum {
            return Err(Error::InvalidChecksum("blob checksum mismatch".to_owned()));
        }
        match self.footer.compression_type {
            NO_COMPRESSION => Ok(false), // in place decoding
            LZ4_COMPRESSION => {
                decompressed_buf.resize(original_len as usize, 0);
                lz4::block::decompress_to_buffer(
                    compressed_data,
                    Some(original_len as i32),
                    decompressed_buf,
                )
                .table_ctx(0, "blob.lz4_decompress")?;
                Ok(true)
            }
            ZSTD_COMPRESSION => unsafe {
                decompressed_buf.resize(original_len as usize, 0);
                let result = zstd_sys::ZSTD_decompress(
                    decompressed_buf.as_mut_ptr() as *mut libc::c_void,
                    original_len as usize,
                    compressed_data.as_ptr() as *const libc::c_void,
                    compressed_data.len(),
                );
                assert_eq!(zstd_sys::ZSTD_isError(result), 0u32);
                Ok(true)
            },
            _ => panic!("unknown compression type {}", self.footer.compression_type),
        }
    }

    pub fn id(&self) -> u64 {
        self.file
            .as_ref()
            .unwrap_or_else(|| panic!("file is not set"))
            .id()
    }

    pub fn version(&self) -> u64 {
        self.footer.version
    }

    pub fn smallest_key(&self) -> InnerKey<'_> {
        InnerKey::from_inner_buf(self.smallest_key.chunk())
    }

    pub fn biggest_key(&self) -> InnerKey<'_> {
        InnerKey::from_inner_buf(self.biggest_key.chunk())
    }

    pub fn size(&self) -> u64 {
        self.file
            .as_ref()
            .unwrap_or_else(|| panic!("file is not set"))
            .size()
    }
    pub fn smallest_biggest_key(&self) -> (InnerKey<'_>, InnerKey<'_>) {
        (self.smallest_key(), self.biggest_key())
    }

    pub fn total_blob_size(&self) -> u64 {
        self.footer.total_blob_size
    }

    pub fn compression_tp(&self) -> u8 {
        self.footer.compression_type
    }

    pub fn compression_lvl(&self) -> i32 {
        self.footer.compression_lvl
    }

    pub fn min_blob_size(&self) -> u32 {
        self.footer.min_blob_size
    }

    pub const fn footer_size() -> usize {
        BLOB_TABLE_FOOTER_SIZE
    }

    pub fn to_blob_create(&self) -> BlobCreate {
        let mut blob_create = BlobCreate::new();
        blob_create.set_id(self.id());
        blob_create.set_smallest(self.smallest_key().to_vec());
        blob_create.set_biggest(self.biggest_key().to_vec());
        blob_create
    }
}

impl BoundedDataSet for BlobTable {
    fn data_bound(&self) -> DataBound<'_> {
        DataBound::new(self.smallest_key(), self.biggest_key(), true)
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
    tables: Arc<HashMap<u64, BlobTable>>,
    tbl_buffers: HashMap<u64, (u32, Vec<u8>)>,
    prefetch_size: usize,
    decompressed_buffer: Vec<u8>,
    decryption_buffer: Vec<u8>,
    encryption_key: Option<EncryptionKey>,
}

impl BlobPrefetcher {
    pub fn new(
        tables: Arc<HashMap<u64, BlobTable>>,
        prefetch_size: usize,
        encryption_key: Option<EncryptionKey>,
    ) -> Self {
        Self {
            tables,
            tbl_buffers: Default::default(),
            prefetch_size,
            decompressed_buffer: vec![],
            decryption_buffer: vec![],
            encryption_key,
        }
    }

    pub fn get(&mut self, blob_ref: &BlobRef) -> Result<&[u8]> {
        let blob_table = self
            .tables
            .get(&blob_ref.fid)
            .ok_or_else(|| Error::Other(format!("blob table not found, fid: {}", blob_ref.fid)))?;
        let data_size = blob_ref.len as usize + BLOB_ENTRY_META_SIZE;
        let (buffer_offset, buffer) = self
            .tbl_buffers
            .entry(blob_ref.fid)
            .or_insert_with(|| (blob_ref.offset, vec![]));
        if !(blob_ref.offset >= *buffer_offset
            && blob_ref.offset + data_size as u32 <= *buffer_offset + buffer.len() as u32)
        {
            let file = blob_table.file.as_ref().unwrap_or_else(|| {
                panic!(
                    "blob table file not set, blob table id: {}",
                    blob_table.id()
                )
            });
            let len = cmp::min(
                cmp::max(self.prefetch_size, data_size),
                file.size() as usize - blob_ref.offset as usize,
            );
            if len != buffer.len() {
                buffer.resize(len, 0);
            }
            file.read_at(buffer, blob_ref.offset as u64)?;
            *buffer_offset = blob_ref.offset;
        }
        let start_off = (blob_ref.offset - *buffer_offset) as usize;
        let data = &buffer[start_off..start_off + BLOB_ENTRY_VALUE_OFFSET + blob_ref.len as usize];
        let meta_slice = &data[..BLOB_ENTRY_META_SIZE];
        let mut data_slice = &data[BLOB_ENTRY_META_SIZE..];
        if let Some(encryption_key) = &self.encryption_key {
            let encryption_ver = blob_table.encryption_ver;
            self.decryption_buffer.clear();
            encryption_key.decrypt(
                data_slice,
                blob_table.id(),
                blob_ref.offset,
                encryption_ver,
                &mut self.decryption_buffer,
            );
            data_slice = &self.decryption_buffer;
        }
        if blob_table.decompress(
            meta_slice,
            data_slice,
            blob_ref.len,
            blob_ref.original_len,
            &mut self.decompressed_buffer,
        )? {
            return Ok(&self.decompressed_buffer);
        }
        Ok(data_slice)
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, sync::Arc};

    use cloud_encryption::EncryptionKey;
    use rand::{distributions::Alphanumeric, rngs::ThreadRng, Rng};
    use rstest::rstest;
    use test_util::init_log_for_test;

    use super::BlobTable;
    use crate::table::{blobtable::BlobRef, file::InMemFile, InnerKey, Value, NO_COMPRESSION};

    fn get_blob_text(max_len: usize, rng: &mut ThreadRng) -> String {
        let len = rng.gen_range(1..max_len);
        rng.sample_iter(&Alphanumeric)
            .take(len)
            .map(char::from)
            .collect()
    }

    struct TestData {
        blob: String,
        blob_ref: BlobRef,
    }

    #[rstest]
    #[case::enable_encryption(true)]
    #[case::disable_encryption(false)]
    fn test_basic(#[case] enable_encryption: bool) {
        init_log_for_test();
        let mut rng = rand::thread_rng();
        let encryption_key = if enable_encryption {
            Some(new_test_encryption_key())
        } else {
            None
        };
        let mut builder =
            super::BlobTableBuilder::new(1, NO_COMPRESSION, 0, 0, encryption_key.clone());
        let mut test_data = Vec::new();
        let meta: u8 = 0;

        for i in 0..128 {
            let key = format!("key_{:03}", i);
            let blob = get_blob_text(64, &mut rng);
            let encoded = Value::encode_buf(meta, &[0], 0, blob.as_bytes());
            let value = Value::decode(encoded.as_slice());
            // In this test assume that all values are converted to blob refs.
            let blob_ref = builder.add(InnerKey::from_inner_buf(key.as_bytes()), &value);
            test_data.push(TestData { blob, blob_ref });
        }

        let file = InMemFile::new(1, builder.finish());
        let table = super::BlobTable::new(Arc::new(file)).unwrap();
        let mut decryption_buf = vec![];
        for td in test_data {
            let blob = table
                .get(&td.blob_ref, &mut decryption_buf, encryption_key.clone())
                .unwrap();
            assert_eq!(td.blob.as_bytes(), blob);
        }

        assert_eq!(table.smallest_key, format!("key_{:03}", 0));
        assert_eq!(table.biggest_key, format!("key_{:03}", 127));
    }

    fn new_test_encryption_key() -> EncryptionKey {
        EncryptionKey::new(b"cipher".to_vec(), b"plain".to_vec(), 0)
    }

    #[rstest]
    #[case::enable_encryption(true)]
    #[case::disable_encryption(false)]
    fn test_prefetcher(#[case] enable_encryption: bool) {
        init_log_for_test();
        let encryption_key = if enable_encryption {
            Some(new_test_encryption_key())
        } else {
            None
        };
        let mut builder =
            super::BlobTableBuilder::new(1, NO_COMPRESSION, 0, 0, encryption_key.clone());
        let mut offsets = Vec::new();
        for i in 0..100 {
            let key_str = format!("key_{:03}", i);
            let val_str = format!("val_{:03}", i);
            let val_buf = Value::encode_buf(b'A', &[0], 0, val_str.as_bytes());
            let blob_ref = builder.add(
                InnerKey::from_inner_buf(key_str.as_bytes()),
                &Value::decode(val_buf.as_slice()),
            );
            offsets.push(blob_ref);
        }
        let file = InMemFile::new(1, builder.finish());
        let table = super::BlobTable::new(Arc::new(file)).unwrap();
        let blob_tables: HashMap<u64, BlobTable> = [(1, table)].into();
        let mut prefetcher =
            super::BlobPrefetcher::new(Arc::new(blob_tables), 1000, encryption_key);
        for i in 0..100 {
            let expected_val = format!("val_{:03}", i);
            let val = prefetcher.get(&offsets[i]).unwrap();
            assert_eq!(val, expected_val.as_bytes());
        }
    }
}
