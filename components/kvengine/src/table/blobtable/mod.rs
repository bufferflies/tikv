// Copyright 2023 TiKV Project Authors. Licensed under Apache-2.0.

pub mod blobtable;
pub mod builder;

use std::mem::size_of;

use byteorder::{ByteOrder, LittleEndian};

#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct BlobRef {
    /// ID of file where the data is stored.
    pub(crate) fid: u64,
    /// Absolute offset within the file. Apparently SST max size is <= 4GB.
    pub(crate) offset: builder::BlobOffset,
    /// Compressed value length.
    pub(crate) len: builder::ValueLength,
    /// Original value length.
    pub(crate) original_len: builder::ValueLength,
}

impl BlobRef {
    pub(crate) fn new(
        fid: u64,
        offset: builder::BlobOffset,
        len: builder::ValueLength,
        original_len: builder::ValueLength,
    ) -> Self {
        Self {
            fid,
            offset,
            original_len,
            len,
        }
    }

    pub(crate) fn deserialize(buf: &[u8]) -> Self {
        let mut offset: usize = 0;
        let fid = LittleEndian::read_u64(&buf[offset..]);
        offset += size_of::<u64>();
        let file_offset = LittleEndian::read_u32(&buf[offset..]);
        offset += size_of::<builder::BlobOffset>();
        let len = LittleEndian::read_u32(&buf[offset..]);
        offset += size_of::<builder::ValueLength>();
        let original_len = LittleEndian::read_u32(&buf[offset..]);
        Self {
            fid,
            offset: file_offset,
            len,
            original_len,
        }
    }

    pub(crate) fn serialize(&self, buf: &mut [u8]) {
        let mut offset: usize = 0;
        LittleEndian::write_u64(&mut buf[offset..], self.fid);
        offset += size_of::<u64>();
        LittleEndian::write_u32(&mut buf[offset..], self.offset);
        offset += size_of::<builder::BlobOffset>();
        LittleEndian::write_u32(&mut buf[offset..], self.len);
        offset += size_of::<builder::ValueLength>();
        LittleEndian::write_u32(&mut buf[offset..], self.original_len);
    }
}

#[test]
fn test_blob_ref_creation() {
    let blob_ref = BlobRef::new(42, 1000, 500, 1000);

    assert_eq!(blob_ref.fid, 42);
    assert_eq!(blob_ref.offset, 1000);
    assert_eq!(blob_ref.len, 500);
    assert_eq!(blob_ref.original_len, 1000);
}

#[test]
fn test_blob_ref_serialization() {
    let blob_ref = BlobRef::new(0xDEADBEEF, 0x1234, 0x5678, 0x9ABC);

    // Buffer size should be exactly what we need for serialization
    let buf_size = size_of::<u64>()
        + size_of::<builder::BlobOffset>()
        + size_of::<builder::ValueLength>()
        + size_of::<builder::ValueLength>();
    assert_eq!(buf_size, 20); // 8 + 4 + 4 + 4 but u32 is often padded to 4 bytes

    let mut buf = vec![0u8; buf_size];
    blob_ref.serialize(&mut buf);

    // Verify serialized data
    assert_eq!(LittleEndian::read_u64(&buf[0..]), 0xDEADBEEF);
    assert_eq!(LittleEndian::read_u32(&buf[8..]), 0x1234);
    assert_eq!(LittleEndian::read_u32(&buf[12..]), 0x5678);
    assert_eq!(LittleEndian::read_u32(&buf[16..]), 0x9ABC);
}

#[test]
fn test_blob_ref_deserialization() {
    // Prepare serialized data
    let mut buf = vec![0u8; 20]; // A bit more space than needed

    LittleEndian::write_u64(&mut buf[0..], 0xCAFEBABE);
    LittleEndian::write_u32(&mut buf[8..], 0x4321);
    LittleEndian::write_u32(&mut buf[12..], 0x8765);
    LittleEndian::write_u32(&mut buf[16..], 0xFEDC);

    let blob_ref = BlobRef::deserialize(&buf);

    assert_eq!(blob_ref.fid, 0xCAFEBABE);
    assert_eq!(blob_ref.offset, 0x4321);
    assert_eq!(blob_ref.len, 0x8765);
    assert_eq!(blob_ref.original_len, 0xFEDC);
}

#[test]
fn test_blob_ref_roundtrip() {
    // Test with various values including boundary cases
    let test_cases = vec![
        BlobRef::new(0, 0, 0, 0),                                 // All zeros
        BlobRef::new(u64::MAX, 0, 0, 0),                          // Max file ID
        BlobRef::new(0, u32::MAX, 0, 0),                          // Max offset
        BlobRef::new(0, 0, u32::MAX, 0),                          // Max compressed length
        BlobRef::new(0, 0, 0, u32::MAX),                          // Max original length
        BlobRef::new(0x1A2B3C4D5E6F7890, 0x2345, 0x6789, 0xABCD), // Random values
    ];

    for original in test_cases {
        let mut buf = vec![0u8; 20]; // Buffer for serialization
        original.serialize(&mut buf);

        let deserialized = BlobRef::deserialize(&buf);
        assert_eq!(deserialized.fid, original.fid);
        assert_eq!(deserialized.offset, original.offset);
        assert_eq!(deserialized.len, original.len);
        assert_eq!(deserialized.original_len, original.original_len);
    }
}

#[test]
fn test_blob_ref_debug_display() {
    let blob_ref = BlobRef::new(42, 1000, 500, 1000);
    let debug_str = format!("{:?}", blob_ref);

    // Verify the debug output contains all fields
    assert!(debug_str.contains("42"));
    assert!(debug_str.contains("1000"));
    assert!(debug_str.contains("500"));
}

#[test]
fn test_blob_ref_comparison() {
    let ref1 = BlobRef::new(1, 100, 50, 200);
    let ref2 = BlobRef::new(1, 100, 50, 200);
    let ref3 = BlobRef::new(2, 100, 50, 200);
    let ref4 = BlobRef::new(1, 101, 50, 200);

    // Test equality
    assert_eq!(ref1, ref2);

    // Test inequality
    assert_ne!(ref1, ref3);
    assert_ne!(ref1, ref4);

    // Test partial ordering
    assert!(ref1 < ref3);
    assert!(ref1 < ref4);
    assert!(ref3 > ref1);
}
