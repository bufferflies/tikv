// Copyright 2025 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    convert::{TryFrom, TryInto},
    fmt, ops,
};

use bitflags::bitflags;
use bytes::{Buf, BufMut, Bytes};
use log_wrappers::Value as LogValue;
use tikv_util::{
    box_try,
    codec::number::{U32_SIZE, U64_SIZE, U8_SIZE},
};

use crate::{
    dfs::FileType,
    table::{
        sstable,
        sstable::{validate_checksum, SsTable},
        Error, Result,
    },
};

bitflags! {
    #[derive(Default)]
    struct TinyMetaFlags: u8 {
        const SEGMENT_OFFSETS = 1 << 0;
    }
}

#[derive(Clone)]
pub struct TinyMeta {
    pub file_id: u64,
    pub footer_and_properties: Bytes,      // Footer + properties.
    pub segment_offsets: Option<Vec<u64>>, // For IA only.
}

impl fmt::Debug for TinyMeta {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TinyMeta")
            .field("id", &self.file_id)
            .field("footer_and_properties", &self.footer_and_properties.len())
            .field(
                "segment_offsets",
                &self.segment_offsets.as_ref().map(Vec::len),
            )
            .finish()
    }
}

impl TinyMeta {
    pub fn try_convert_to(self, ftype: FileType) -> TypedTinyMeta {
        match ftype {
            FileType::Sst => SstTinyMeta::try_from(self)
                .map(TypedTinyMeta::Sst)
                .unwrap_or_default(),
            _ => TypedTinyMeta::None,
        }
    }

    fn read_footer(&self, footer_length: usize) -> Result<Bytes> {
        let data_len = self.footer_and_properties.len();
        let Some(off) = data_len.checked_sub(footer_length) else {
            error!("TinyMeta: read_footer: invalid file size";
                "file" => self.file_id, "data_len" => data_len, "footer_length" => footer_length);
            return Err(Error::InvalidFileSize);
        };
        Ok(self.footer_and_properties.slice(off..))
    }

    pub fn marshal(&self) -> Result<Vec<u8>> {
        let include_segment_offsets = self.segment_offsets.is_some();
        let segment_offsets_len = self.segment_offsets.as_ref().map_or(0, Vec::len);
        let flag = include_segment_offsets
            .then_some(TinyMetaFlags::SEGMENT_OFFSETS)
            .unwrap_or_default();

        // file_id + flag + footer_and_properties.len + footer_and_properties +
        // [segment_offsets.len + segment_offsets]
        let cap = U64_SIZE
            + U8_SIZE
            + U32_SIZE
            + self.footer_and_properties.len()
            + include_segment_offsets as usize * (U32_SIZE + segment_offsets_len * U64_SIZE);
        let mut buf = Vec::with_capacity(cap);
        buf.put_u64_le(self.file_id);
        buf.put_u8(flag.bits());
        buf.put_u32_le(box_try!(self.footer_and_properties.len().try_into()));
        buf.put_slice(&self.footer_and_properties);
        if let Some(segment_offsets) = &self.segment_offsets {
            buf.put_u32_le(box_try!(segment_offsets.len().try_into()));
            for offset in segment_offsets {
                buf.put_u64_le(*offset);
            }
        }
        Ok(buf)
    }

    pub fn unmarshal(buf: &mut &[u8]) -> Result<Self> {
        let expected_len = U64_SIZE + U8_SIZE + U32_SIZE; // file_id + flag + footer_and_properties.len
        if buf.remaining() < expected_len {
            return Err(Error::CorruptedMetaPack(format!(
                "length mismatch: {} < {}",
                buf.remaining(),
                expected_len
            )));
        }
        let file_id = buf.get_u64_le();
        let flag = TinyMetaFlags::from_bits_truncate(buf.get_u8());
        let footer_and_properties_len = buf.get_u32_le() as usize;
        if buf.remaining() < footer_and_properties_len {
            return Err(Error::CorruptedMetaPack(format!(
                "data length mismatch: {} < {}",
                buf.remaining(),
                footer_and_properties_len
            )));
        }
        let footer_and_properties = Bytes::copy_from_slice(&buf[..footer_and_properties_len]);
        buf.advance(footer_and_properties_len);

        let segment_offsets = if !flag.contains(TinyMetaFlags::SEGMENT_OFFSETS) {
            None
        } else {
            if buf.remaining() < U32_SIZE {
                return Err(Error::CorruptedMetaPack(format!(
                    "segment offsets length mismatch: {} < {}",
                    buf.remaining(),
                    U32_SIZE
                )));
            }
            let segment_offsets_len = buf.get_u32_le() as usize;
            let segment_offsets_bytes =
                segment_offsets_len.checked_mul(U64_SIZE).ok_or_else(|| {
                    Error::CorruptedMetaPack(format!(
                        "segment offsets size overflow: {} * {}",
                        segment_offsets_len, U64_SIZE
                    ))
                })?;
            if buf.remaining() < segment_offsets_bytes {
                return Err(Error::CorruptedMetaPack(format!(
                    "segment offsets data length mismatch: {} < {}",
                    buf.remaining(),
                    segment_offsets_bytes
                )));
            }
            let mut segment_offsets = Vec::with_capacity(segment_offsets_len);
            for _ in 0..segment_offsets_len {
                segment_offsets.push(buf.get_u64_le());
            }
            Some(segment_offsets)
        };

        Ok(TinyMeta {
            file_id,
            footer_and_properties,
            segment_offsets,
        })
    }
}

#[derive(Default, Clone, Debug)]
pub enum TypedTinyMeta {
    #[default]
    None,
    Sst(SstTinyMeta),
}

impl TypedTinyMeta {
    pub fn meta_size(&self) -> Option<u64> {
        match self {
            TypedTinyMeta::None => None,
            TypedTinyMeta::Sst(sst_meta) => Some(sst_meta.meta_size()),
        }
    }

    pub fn file_size(&self) -> Option<u64> {
        match self {
            TypedTinyMeta::None => None,
            TypedTinyMeta::Sst(sst_meta) => Some(sst_meta.file_size()),
        }
    }

    pub fn into_sst(self) -> Option<SstTinyMeta> {
        match self {
            TypedTinyMeta::None => None,
            TypedTinyMeta::Sst(sst_meta) => Some(sst_meta),
        }
    }

    pub fn as_sst(&self) -> Option<&SstTinyMeta> {
        match self {
            TypedTinyMeta::None => None,
            TypedTinyMeta::Sst(sst_meta) => Some(sst_meta),
        }
    }

    pub fn into_inner(self) -> Option<TinyMeta> {
        match self {
            TypedTinyMeta::None => None,
            TypedTinyMeta::Sst(sst_meta) => Some(sst_meta.inner),
        }
    }
}

#[derive(Clone)]
pub struct SstTinyMeta {
    pub inner: TinyMeta,
    pub footer: sstable::Footer,
}

impl ops::Deref for SstTinyMeta {
    type Target = TinyMeta;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl fmt::Debug for SstTinyMeta {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SstTinyMeta")
            .field("inner", &self.inner)
            .field("footer", &self.footer)
            .finish()
    }
}

impl TryFrom<TinyMeta> for SstTinyMeta {
    type Error = Error;

    fn try_from(tiny_meta: TinyMeta) -> Result<Self> {
        let footer_data = tiny_meta.read_footer(SsTable::footer_size())?;
        let mut footer = sstable::Footer::default();
        footer.unmarshal(&footer_data);
        if !footer.is_match() {
            error!("SstTinyMeta: footer not match";
                "file" => tiny_meta.file_id, "footer" => LogValue::value(&footer_data));
            return Err(Error::CorruptedMetaPack(format!(
                "SstTinyMeta: footer not match: {}",
                tiny_meta.file_id
            )));
        }

        Ok(SstTinyMeta {
            inner: tiny_meta,
            footer,
        })
    }
}

impl From<SstTinyMeta> for TinyMeta {
    fn from(sst_tiny_meta: SstTinyMeta) -> TinyMeta {
        sst_tiny_meta.inner
    }
}

impl SstTinyMeta {
    pub fn from_sstable(
        sstable: &SsTable,
        footer: sstable::Footer,
        properties_data: &[u8],
    ) -> Self {
        debug_assert_eq!(sstable.footer(), footer);

        let mut buf = Vec::with_capacity(properties_data.len() + SsTable::footer_size());
        buf.put_slice(properties_data);
        footer.marshal(&mut buf);

        let segment_offsets = sstable
            .try_get_ia_file()
            .map(|f| f.segment_offsets.clone())
            .or_else(|| {
                sstable
                    .try_get_auto_ia_file()
                    .map(|f| f.ia_file.segment_offsets.clone())
            });

        Self {
            inner: TinyMeta {
                file_id: sstable.id(),
                footer_and_properties: buf.into(),
                segment_offsets,
            },
            footer,
        }
    }

    fn try_read(&self, off: u64, length: usize) -> Option<Bytes> {
        let tiny_meta_off = self.footer.tiny_meta_offset() as u64;
        if off >= tiny_meta_off && off + length as u64 <= self.file_size() {
            let start = (off - tiny_meta_off) as usize;
            Some(self.footer_and_properties.slice(start..start + length))
        } else {
            None
        }
    }

    pub fn get_footer_and_properties(&self) -> Result<(sstable::Footer, Bytes)> {
        let props_off = self.footer.properties_offset;
        let props_len = self.footer.properties_len(self.file_size() as usize);

        let Some(props_data) = self.try_read(props_off as u64, props_len) else {
            warn!("SstTinyMeta: get_properties_data: invalid properties range";
                "file_id" => self.file_id, "props_off" => props_off, "props_len" => props_len);
            debug_assert!(false);
            return Err(Error::CorruptedMetaPack(format!(
                "invalid properties range: off={} len={}",
                props_off, props_len
            )));
        };

        box_try!(validate_checksum(self.file_id, &props_data, &self.footer));
        Ok((self.footer, props_data))
    }

    #[inline]
    pub fn file_size(&self) -> u64 {
        self.footer.tiny_meta_offset() as u64 + self.inner.footer_and_properties.len() as u64
    }

    #[inline]
    pub fn meta_size(&self) -> u64 {
        self.file_size() - self.footer.meta_offset() as u64
    }
}

#[cfg(test)]
mod tests {
    use bytes::Bytes;
    use rand::prelude::*;

    use super::*;
    use crate::table::{
        sstable::builder::{Footer, FOOTER_SIZE, MAGIC_NUMBER, TABLE_FORMAT_V1},
        ChecksumType, InnerKey, Value, NO_COMPRESSION,
    };

    #[test]
    fn test_tiny_meta() {
        // Test round-trip marshaling/unmarshaling.
        let footer_and_properties = Bytes::from_static(b"footer and properties data");
        let cases = [
            (98765, Some(vec![3, 7, 11])),
            (123, None),
            (456, Some(vec![])),
        ];

        for (file_id, segment_offsets) in cases {
            let original = TinyMeta {
                file_id,
                footer_and_properties: footer_and_properties.clone(),
                segment_offsets,
            };

            let marshaled = original.marshal().expect("marshal should succeed");
            assert!(!marshaled.is_empty());

            let mut slice: &[u8] = &marshaled;
            let restored = TinyMeta::unmarshal(&mut slice).expect("unmarshal should succeed");

            assert_eq!(restored.file_id, original.file_id);
            assert_eq!(
                restored.footer_and_properties,
                original.footer_and_properties
            );
            assert_eq!(restored.segment_offsets, original.segment_offsets);
            assert!(slice.is_empty()); // All data should be consumed.
        }
    }

    #[test]
    fn test_tiny_meta_unmarshal_insufficient_data() {
        {
            // Test unmarshal with insufficient data
            let mut slice: &[u8] = &[0u8; 5]; // Less than needed (U64_SIZE + U8_SIZE + U32_SIZE = 13)
            let result = TinyMeta::unmarshal(&mut slice);
            assert!(result.is_err());
            match result {
                Err(Error::CorruptedMetaPack(msg)) => {
                    assert!(msg.contains("length mismatch"));
                }
                _ => panic!("Expected CorruptedMetaPack error"),
            }
        }

        {
            // Test unmarshal with data length mismatch
            let mut buf = Vec::new();
            buf.put_u64_le(12345); // file_id
            buf.put_u8(0); // reserved
            buf.put_u32_le(100); // data_len (claiming 100 bytes)
            buf.put_slice(&[1u8; 50]); // But only provide 50 bytes

            let mut slice: &[u8] = &buf;
            let result = TinyMeta::unmarshal(&mut slice);
            assert!(result.is_err());
            match result {
                Err(Error::CorruptedMetaPack(msg)) => {
                    assert!(msg.contains("data length mismatch"));
                }
                _ => panic!("Expected CorruptedMetaPack error"),
            }
        }
    }

    #[test]
    fn test_typed_tiny_meta() {
        // Test TypedTinyMeta::None variant
        let none_meta = TypedTinyMeta::None;
        assert!(none_meta.meta_size().is_none());
        assert!(none_meta.file_size().is_none());
        assert!(none_meta.as_sst().is_none());
        assert!(none_meta.clone().into_sst().is_none());
        assert!(none_meta.into_inner().is_none());
    }

    #[test]
    fn test_sst_tiny_meta() {
        // Test successful conversion with valid SST data
        let file_id = 12345;
        let (tiny_meta, table_data, table_meta_off) = make_tiny_meta_ext(file_id, 20, false);
        let table_file_size = table_data.len() as u64;

        // Test TypedTinyMeta conversion
        let typed = tiny_meta.clone().try_convert_to(FileType::Sst);
        match &typed {
            TypedTinyMeta::Sst(sst_meta) => {
                assert_eq!(sst_meta.file_id, file_id);
                assert_eq!(
                    sst_meta.footer_and_properties,
                    tiny_meta.footer_and_properties
                );
                assert_eq!(sst_meta.segment_offsets, tiny_meta.segment_offsets);

                assert_eq!(sst_meta.file_size(), table_file_size);
                assert_eq!(sst_meta.meta_size(), table_file_size - table_meta_off);
                assert_eq!(
                    sst_meta.footer.tiny_meta_offset(),
                    table_file_size as u32 - tiny_meta.footer_and_properties.len() as u32
                );
            }
            _ => panic!("Should convert to Sst variant"),
        }
    }

    #[test]
    fn test_sst_tiny_meta_try_from_invalid() {
        // Test failed conversion with invalid SST data
        let tiny_meta = make_invalid_tiny_meta(99999);

        let result = SstTinyMeta::try_from(tiny_meta);
        assert!(result.is_err());
        match result {
            Err(Error::CorruptedMetaPack(msg)) => {
                assert!(msg.contains("footer not match"));
            }
            _ => panic!("Expected CorruptedMetaPack error for invalid footer"),
        }
    }

    #[test]
    fn test_sst_tiny_meta_try_read() {
        let file_id = 77777;
        let tiny_meta = make_tiny_meta(file_id, 20);
        let sst_meta = SstTinyMeta::try_from(tiny_meta).expect("Should convert");

        // Test try_read with valid offset (within tiny meta range)
        let props_off = sst_meta.footer.properties_offset as u64;
        let props_len = sst_meta
            .footer
            .properties_len(sst_meta.file_size() as usize);

        // Should be able to read properties data
        let read_result = sst_meta.try_read(props_off, props_len);
        assert!(read_result.is_some());
        let read_data = read_result.unwrap();
        assert_eq!(read_data.len(), props_len);

        // Test try_read with offset outside range
        let invalid_read = sst_meta.try_read(sst_meta.file_size() + 100, 10);
        assert!(invalid_read.is_none());

        // Test try_read with offset+length exceeding file size
        let overflow_read = sst_meta.try_read(sst_meta.file_size() - 5, 10);
        assert!(overflow_read.is_none());
    }

    #[test]
    fn test_sst_tiny_meta_get_footer_and_properties() {
        let file_id = 88888;
        let tiny_meta = make_tiny_meta(file_id, 20);
        let sst_meta = SstTinyMeta::try_from(tiny_meta).expect("Should convert");

        // Test get_footer_and_properties
        let result = sst_meta.get_footer_and_properties();
        assert!(result.is_ok());
        let (footer, props_data) = result.unwrap();

        assert!(footer.is_match());
        assert_eq!(footer.magic, MAGIC_NUMBER);
        assert_eq!(footer.table_format_version, TABLE_FORMAT_V1);
        assert_eq!(
            props_data.len(),
            sst_meta
                .footer
                .properties_len(sst_meta.file_size() as usize)
        );
    }

    fn make_sstable_data(
        file_id: u64,
        n: usize,
        key_len: usize,
        val_len: usize,
        multi_ver: bool,
    ) -> (
        Bytes, // file_data
        u64,   // tiny_meta_off
        u64,   // meta_off
    ) {
        let mut rng = thread_rng();

        let mut builder = sstable::Builder::new(
            file_id,
            4096,
            NO_COMPRESSION,
            0,
            ChecksumType::default(),
            None,
        );
        let mut val = vec![0; val_len];
        let mut ver = n as u64;
        let mut i = 0;
        for _ in 0..n {
            let key = format!("{:0key_len$}", i).into_bytes();
            rng.fill_bytes(val.as_mut_slice());
            let value_buf = Value::encode_buf(0u8, &[0], ver, &val);
            let value = Value::decode(&value_buf);
            builder.add(InnerKey::from_inner_buf(&key), &value, None);

            if multi_ver && rng.gen_ratio(1, 4) {
                ver -= 1;
            } else {
                i += 1;
                ver = n as u64;
            }
        }

        let mut buf = Vec::with_capacity(builder.estimated_size());
        let res = builder.finish(0, &mut buf);
        let file_data = Bytes::from(buf);
        (
            file_data,
            res.tiny_meta_offset as u64,
            res.meta_offset as u64,
        )
    }

    // Helper function to create a valid TinyMeta
    fn make_tiny_meta(file_id: u64, n: usize) -> TinyMeta {
        let with_offsets = thread_rng().gen_ratio(1, 2);
        let (tiny_meta, ..) = make_tiny_meta_ext(file_id, n, with_offsets);
        tiny_meta
    }

    fn make_tiny_meta_ext(
        file_id: u64,
        n: usize,
        with_offsets: bool,
    ) -> (
        TinyMeta,
        Bytes, // table_data
        u64,   // meta_off
    ) {
        let (table_data, tiny_meta_off, meta_off) = make_sstable_data(file_id, n, 7, 10, false);
        let mut rng = thread_rng();
        let amount = rng.gen_range(0..5);
        let segment_offsets = with_offsets.then(|| (0..meta_off).choose_multiple(&mut rng, amount));
        (
            TinyMeta {
                file_id,
                footer_and_properties: table_data.slice(tiny_meta_off as usize..),
                segment_offsets,
            },
            table_data,
            meta_off,
        )
    }

    // Helper function to create a TinyMeta with invalid SST footer data
    fn make_invalid_tiny_meta(file_id: u64) -> TinyMeta {
        // Create footer with wrong magic number
        let mut footer = Footer::default();
        footer.magic = 0xDEADBEEF; // Invalid magic
        footer.table_format_version = TABLE_FORMAT_V1;

        // Use same structure as valid footer for consistency
        let data_size = 100;
        let index_size = 50;
        let properties_size = 80;

        footer.old_data_offset = 0;
        footer.index_offset = data_size as u32;
        footer.old_index_offset = 0;
        footer.aux_index_offset = 0;
        footer.properties_offset = (data_size + index_size) as u32;
        footer.compression_type = 0;
        footer.checksum_type = 1;

        let mut footer_data = Vec::new();
        footer.marshal(&mut footer_data);

        let total_size = data_size + index_size + properties_size + FOOTER_SIZE;
        let mut sst_data = vec![0u8; total_size];
        let footer_start = total_size - FOOTER_SIZE;
        sst_data[footer_start..].copy_from_slice(&footer_data);

        TinyMeta {
            file_id,
            footer_and_properties: Bytes::from(sst_data),
            segment_offsets: None,
        }
    }
}
