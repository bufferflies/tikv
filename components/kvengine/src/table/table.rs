// Copyright 2021 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    cmp::Ordering,
    fmt::{self, Debug, Formatter},
    iter::Iterator as StdIterator,
    mem::size_of,
    ops::Deref,
    ptr, result, slice,
};

use api_version::{api_v2, ApiV2, KeyMode, KvFormat};
use byteorder::{ByteOrder, LittleEndian};
use bytes::BufMut;
use log_wrappers::Value as LogValue;
use thiserror::Error;

use super::blobtable::BlobRef;
use crate::{dfs, GLOBAL_SHARD_END_KEY};

#[derive(Serialize, Deserialize, Debug)]
pub struct Row {
    pub key: Vec<u8>,
    pub user_meta: crate::UserMeta,
    pub value: Vec<u8>,
}

pub trait Iterator: Send {
    // next returns the next entry with different key on the latest version.
    // If old version is needed, call next_version.
    fn next(&mut self);

    // next_version set the current entry to an older version.
    // The iterator must be valid to call this method.
    // It returns true if there is an older version, returns false if there is no
    // older version. The iterator is still valid and on the same key.
    fn next_version(&mut self) -> bool;

    fn rewind(&mut self);

    fn seek(&mut self, key: InnerKey<'_>);

    fn key(&self) -> InnerKey<'_>;

    fn value(&self) -> Value;

    fn valid(&self) -> bool;

    fn seek_to_version(&mut self, version: u64) -> bool {
        if version >= self.value().version {
            return true;
        }
        while self.next_version() {
            if version >= self.value().version {
                return true;
            }
        }
        false
    }

    fn next_all_version(&mut self) {
        if !self.next_version() {
            self.next()
        }
    }
}

pub const NO_COMPRESSION: u8 = 0;
pub const LZ4_COMPRESSION: u8 = 1;
pub const ZSTD_COMPRESSION: u8 = 2;

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
#[derive(Default)]
pub enum ChecksumType {
    None,
    #[default]
    Crc32c,
    Crc32,
}

impl ChecksumType {
    pub fn value(&self) -> u8 {
        match self {
            ChecksumType::None => 0,
            ChecksumType::Crc32c => 1,
            ChecksumType::Crc32 => 2,
        }
    }

    pub fn checksum(&self, data: &[u8]) -> u32 {
        match self {
            ChecksumType::None => 0,
            ChecksumType::Crc32c => crc32c::crc32c(data),
            ChecksumType::Crc32 => crc32fast::hash(data),
        }
    }

    pub fn append(&self, checksum: u32, data: &[u8]) -> u32 {
        match self {
            ChecksumType::None => 0,
            ChecksumType::Crc32c => crc32c::crc32c_append(checksum, data),
            ChecksumType::Crc32 => {
                let mut hasher = crc32fast::Hasher::new_with_initial(checksum);
                hasher.update(data);
                hasher.finalize()
            }
        }
    }
}

impl From<u8> for ChecksumType {
    fn from(v: u8) -> Self {
        match v {
            0 => ChecksumType::None,
            1 => ChecksumType::Crc32c,
            2 => ChecksumType::Crc32,
            _ => {
                error!("unknown checksum type {}", v);
                debug_assert!(false, "unknown checksum type {}", v);
                ChecksumType::None
            }
        }
    }
}

pub const BIT_DELETE: u8 = 1;
pub const BIT_HAS_OLD_VERSION: u8 = 2;
pub const BIT_BLOB_REF: u8 = 4;

pub fn is_deleted(meta: u8) -> bool {
    meta & BIT_DELETE > 0
}

pub fn is_blob_ref(meta: u8) -> bool {
    meta & BIT_BLOB_REF > 0
}

pub fn is_old_version(meta: u8) -> bool {
    meta & BIT_HAS_OLD_VERSION > 0
}

/// [ meta: u8, user_meta_len: u8]
pub const VALUE_VERSION_OFF: usize = 2;

// Size in bytes of the version field.
pub const VALUE_VERSION_LEN: usize = std::mem::size_of::<u64>();

pub struct Version(pub u64);

impl Version {
    pub fn serialize(buf: &mut [u8], version: u64) -> usize {
        LittleEndian::write_u64(buf, version);
        size_of::<u64>()
    }

    pub fn deserialize(buf: &[u8]) -> u64 {
        LittleEndian::read_u64(buf)
    }
}

unsafe impl Send for Value {}

// Value is a short life struct used to pass value across iterators.
// It is valid until iterator call next or next_version.
// As long as the value is never escaped, there will be no dangling pointer.
#[derive(Debug, Copy, Clone)]
pub struct Value {
    /// Points to start of user_meta_len at offset VALUE_VERSION_OFF +
    /// serialized(version).
    ptr: *const u8,
    /// Bit flags
    pub meta: u8,
    /// User defined opaque meta data,
    user_meta_len: u8,
    /// Length of the data.
    len: u32,
    /// The row version
    pub version: u64,

    blob_ptr: *const u8,
}

impl Value {
    pub(crate) fn new() -> Self {
        Self {
            ptr: ptr::null(),
            meta: Default::default(),
            user_meta_len: Default::default(),
            len: Default::default(),
            version: Default::default(),
            blob_ptr: ptr::null(),
        }
    }

    fn encode_preamble(buf: &mut [u8], meta: u8, version: u64, user_meta: &[u8]) -> usize {
        buf[0] = meta;
        buf[1] = user_meta.len() as u8;
        let mut offset = VALUE_VERSION_OFF;
        offset += Version::serialize(&mut buf[offset..], version);
        if !user_meta.is_empty() {
            buf[offset..offset + user_meta.len()].copy_from_slice(user_meta);
        }
        offset + user_meta.len()
    }

    pub fn encode_buf(meta: u8, user_meta: &[u8], version: u64, val: &[u8]) -> Vec<u8> {
        assert!(!is_blob_ref(meta));
        let mut buf = vec![0; VALUE_VERSION_OFF + VALUE_VERSION_LEN + user_meta.len() + val.len()];
        let offset = Self::encode_preamble(&mut buf, meta, version, user_meta);
        let buf_slice = buf.as_mut_slice();
        buf_slice[offset..offset + val.len()].copy_from_slice(val);
        buf
    }

    /// Encode the contents in buf.
    pub(crate) fn encode(&self, buf: &mut [u8]) {
        let offset = Self::encode_preamble(buf, self.meta, self.version, self.user_meta());
        unsafe {
            ptr::copy(
                self.ptr.add(self.user_meta_len()),
                buf[offset..].as_mut_ptr(),
                self.value_len(),
            );
        }
    }

    pub fn encode_with_blob_ref(&self, buf: &mut [u8], blob_ref: BlobRef) {
        assert!(self.is_blob_ref());
        let offset = Self::encode_preamble(buf, self.meta, self.version, self.user_meta());
        blob_ref.serialize(&mut buf[offset..offset + size_of::<BlobRef>()]);
    }

    pub fn decode(buf: &[u8]) -> Self {
        let meta = buf[0];
        let user_meta_len = buf[1];
        let mut offset = VALUE_VERSION_OFF;
        let version = Version::deserialize(&buf[offset..]);
        offset += VALUE_VERSION_LEN;
        Self {
            ptr: buf[offset..].as_ptr(),
            meta,
            user_meta_len,
            len: (buf.len() - offset - user_meta_len as usize) as u32,
            version,
            blob_ptr: ptr::null(),
        }
    }

    pub(crate) fn new_with_meta_version(
        meta: u8,
        version: u64,
        user_meta_len: u8,
        buf: &[u8],
    ) -> Self {
        assert!(buf.len() >= user_meta_len as usize);
        Self {
            ptr: buf.as_ptr(),
            meta,
            user_meta_len,
            len: buf.len() as u32 - user_meta_len as u32,
            version,
            blob_ptr: ptr::null(),
        }
    }

    pub(crate) fn new_tombstone(version: u64) -> Self {
        Self {
            ptr: ptr::null(),
            meta: BIT_DELETE,
            user_meta_len: 0,
            len: 0,
            version,
            blob_ptr: ptr::null(),
        }
    }

    #[inline(always)]
    pub(crate) fn set_blob_ref(&mut self) {
        assert!(!self.is_blob_ref());
        self.meta |= BIT_BLOB_REF;
    }

    #[inline(always)]
    pub(crate) fn is_empty(&self) -> bool {
        self.meta == 0 && self.ptr.is_null()
    }

    #[inline(always)]
    pub(crate) fn is_valid(&self) -> bool {
        !self.is_empty()
    }

    #[inline(always)]
    pub(crate) fn is_deleted(&self) -> bool {
        is_deleted(self.meta)
    }

    #[inline(always)]
    pub(crate) fn is_blob_ref(&self) -> bool {
        is_blob_ref(self.meta)
    }

    #[inline(always)]
    pub fn value_len(&self) -> usize {
        self.len as usize
    }

    #[inline(always)]
    pub fn user_meta_len(&self) -> usize {
        self.user_meta_len as usize
    }

    #[inline(always)]
    pub fn user_meta(&self) -> &[u8] {
        unsafe { slice::from_raw_parts::<u8>(self.ptr, self.user_meta_len()) }
    }

    #[inline(always)]
    pub fn is_value_empty(&self) -> bool {
        self.value_len() == 0
    }

    #[inline(always)]
    pub fn get_value(&self) -> &[u8] {
        if self.is_value_empty() {
            return &[];
        }
        unsafe {
            if self.blob_ptr.is_null() {
                slice::from_raw_parts::<u8>(self.ptr.add(self.user_meta_len()), self.value_len())
            } else {
                slice::from_raw_parts::<u8>(self.blob_ptr, self.value_len())
            }
        }
    }

    #[inline(always)]
    fn encoded_size_of_preamble(&self) -> usize {
        VALUE_VERSION_OFF + size_of::<u64>() + self.user_meta_len()
    }

    #[inline(always)]
    pub fn encoded_size(&self) -> usize {
        self.encoded_size_of_preamble() + self.value_len()
    }

    #[inline(always)]
    pub fn encoded_size_with_blob_ref(&self) -> usize {
        self.encoded_size_of_preamble() + size_of::<BlobRef>()
    }

    pub fn get_blob_ref(&self) -> BlobRef {
        assert!(self.is_blob_ref());
        BlobRef::deserialize(self.get_value())
    }

    #[inline(always)]
    pub fn get_version(buf: &[u8]) -> u64 {
        Version::deserialize(buf)
    }

    pub fn fill_in_blob(&mut self, blob: &[u8]) {
        assert!(self.is_blob_ref());
        self.blob_ptr = blob.as_ptr();
        self.len = blob.len() as u32;
        self.meta &= !BIT_BLOB_REF;
    }
}

pub struct EmptyIterator;

impl Iterator for EmptyIterator {
    fn next(&mut self) {}

    fn next_version(&mut self) -> bool {
        false
    }

    fn rewind(&mut self) {}

    fn seek(&mut self, _: InnerKey<'_>) {}

    fn key(&self) -> InnerKey<'_> {
        InnerKey { key: &[] }
    }

    fn value(&self) -> Value {
        Value::new()
    }

    fn valid(&self) -> bool {
        false
    }
}

#[derive(Debug, Error, Clone)]
pub enum Error {
    #[error("Key not found")]
    NotFound,
    #[error("Invalid checksum {0}")]
    InvalidChecksum(String),
    #[error("Invalid filename")]
    InvalidFileName,
    #[error("Invalid file size")]
    InvalidFileSize,
    #[error("Invalid magic number")]
    InvalidMagicNumber,
    #[error("IO error: {0}")]
    Io(String),
    #[error("EOF")]
    Eof,
    #[error("Schema out of date: {0}")]
    SchemaOutOfDate(String),
    #[error("Need encryption key for txn chunk {chunk_id}, encryption version {encryption_ver}")]
    NeedEncryptionKey { chunk_id: u64, encryption_ver: u32 },
    #[error("{0}")]
    Other(String),
}

impl From<dfs::Error> for Error {
    #[inline]
    fn from(e: dfs::Error) -> Error {
        Error::Io(e.to_string())
    }
}

pub type Result<T> = result::Result<T, Error>;

/// Simple rewrite of golang sort.Search
/// Return i, f(x) == false when x in [0, i), f(x) == true when x in [i, n)
pub fn search<F>(n: usize, mut f: F) -> usize
where
    F: FnMut(usize) -> bool,
{
    let mut i = 0;
    let mut j = n;
    while i < j {
        let h = (i + j) / 2;
        if !f(h) {
            i = h + 1;
        } else {
            j = h;
        }
    }
    i
}

pub(crate) fn parse_prop_data(mut prop_data: &[u8]) -> (&[u8], &[u8], &[u8]) {
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

#[derive(Clone, Copy, Default)]
pub struct LocalAddr {
    pub start: usize,
    pub end: usize,
}

impl LocalAddr {
    pub fn new(start: usize, end: usize) -> Self {
        Self { start, end }
    }

    pub fn get(self, buf: &[u8]) -> &[u8] {
        &buf[self.start..self.end]
    }

    pub fn len(self) -> usize {
        self.end - self.start
    }

    pub fn is_empty(self) -> bool {
        self.len() == 0
    }
}

pub fn new_merge_iterator<'a>(
    mut iters: Vec<Box<dyn Iterator + 'a>>,
    reverse: bool,
) -> Box<dyn Iterator + 'a> {
    match iters.len() {
        0 => Box::new(EmptyIterator {}),
        1 => iters.pop().unwrap(),
        2 => {
            let second_iter: Box<dyn Iterator + 'a> = iters.pop().unwrap();
            let first_iter: Box<dyn Iterator + 'a> = iters.pop().unwrap();
            let first: Box<super::MergeIteratorChild<'a>> =
                Box::new(super::MergeIteratorChild::new(first_iter));
            let second = Box::new(super::MergeIteratorChild::new(second_iter));
            let merge_iter = super::MergeIterator::new(first, second, reverse);
            Box::new(merge_iter)
        }
        _ => {
            let mid = iters.len() / 2;
            let mut second = vec![];
            for _ in 0..mid {
                second.push(iters.pop().unwrap())
            }
            second.reverse();
            let first_it = new_merge_iterator(iters, reverse);
            let second_it = new_merge_iterator(second, reverse);
            new_merge_iterator(vec![first_it, second_it], reverse)
        }
    }
}

#[derive(Clone, Copy, PartialOrd, PartialEq, Ord, Eq)]
pub struct InnerKey<'a> {
    key: &'a [u8],
}

impl Debug for InnerKey<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", log_wrappers::hex_encode_upper(self.key))
    }
}

impl<'a> InnerKey<'a> {
    pub fn from_inner_buf(key: &'a [u8]) -> Self {
        Self { key }
    }

    pub fn from_outer_key<'b: 'a>(outer_key: &'b [u8], inner_offset: usize) -> Self {
        debug_assert!(inner_offset <= outer_key.len());
        let key = &outer_key[inner_offset..];
        Self { key }
    }

    pub fn from_outer_end_key<'b: 'a>(outer_end_key: &'b [u8], inner_offset: usize) -> Self {
        debug_assert!(inner_offset <= outer_end_key.len());
        if outer_end_key.len() == inner_offset {
            Self {
                key: GLOBAL_SHARD_END_KEY,
            }
        } else {
            let key = &outer_end_key[inner_offset..];
            Self { key }
        }
    }
}

impl Deref for InnerKey<'_> {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        self.key
    }
}

#[derive(Clone)]
pub struct OwnedInnerKey {
    inner: bytes::Bytes,
}

impl Debug for OwnedInnerKey {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", &self.inner)
    }
}

impl OwnedInnerKey {
    pub fn new(inner: bytes::Bytes) -> Self {
        Self { inner }
    }

    pub fn as_ref(&self) -> InnerKey<'_> {
        InnerKey::from_inner_buf(&self.inner)
    }

    pub fn into_inner(self) -> bytes::Bytes {
        self.inner
    }

    pub fn to_vec(&self) -> Vec<u8> {
        self.inner.to_vec()
    }
}

/// The user key without keyspace prefix.
///
/// A simple wrapper to make sure that the key and interface is properly used.
///
/// Currently used for building txn chunks as the hash index is always using
/// keys without keyspace prefix, no matter whether inner key offset is enabled
/// or not.
///
/// No prefix key is required to be a valid TiDB key (starts with "t" or "m") to
/// make conversion from `InnerKey` work.
#[derive(PartialEq)]
pub struct NoPrefixKey<'a>(pub &'a [u8]);

impl fmt::Debug for NoPrefixKey<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", LogValue::key(self.0))
    }
}

impl Deref for NoPrefixKey<'_> {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        self.0
    }
}

impl<'a> NoPrefixKey<'a> {
    pub fn from_inner_key(inner_key: &'a InnerKey<'a>) -> Option<Self> {
        match ApiV2::parse_key_mode(inner_key.deref()) {
            KeyMode::Txn => Some(Self(&inner_key.deref()[api_v2::KEYSPACE_PREFIX_LEN..])),
            KeyMode::Tidb => Some(Self(inner_key.deref())),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DataBound<'a> {
    pub lower_bound: InnerKey<'a>,
    pub upper_bound: InnerKey<'a>,
    pub upper_inclusive: bool,
}

impl<'a> DataBound<'a> {
    pub fn new(
        lower_bound: InnerKey<'a>,
        upper_bound: InnerKey<'a>,
        upper_inclusive: bool,
    ) -> Self {
        Self {
            lower_bound,
            upper_bound,
            upper_inclusive,
        }
    }

    pub fn overlap_key(&self, key: InnerKey<'_>) -> bool {
        self.lower_bound <= key && !self.less_than_key(key)
    }

    pub fn overlap_bound(&self, bound: DataBound<'_>) -> bool {
        !self.less_than_key(bound.lower_bound) && !bound.less_than_key(self.lower_bound)
    }

    pub fn contains_bound(&self, other: DataBound<'_>) -> bool {
        self.lower_bound <= other.lower_bound
            && match other.upper_bound.cmp(&self.upper_bound) {
                Ordering::Less => true,
                Ordering::Equal => !other.upper_inclusive || self.upper_inclusive,
                Ordering::Greater => false,
            }
    }

    pub fn less_than_key(&self, key: InnerKey<'_>) -> bool {
        match self.upper_bound.cmp(&key) {
            Ordering::Less => true,
            Ordering::Equal => !self.upper_inclusive,
            Ordering::Greater => false,
        }
    }

    /// Find for the first one of `data_sets` that overlaps with self.
    ///
    /// Pass `is_sorted` to `None` if it's not sure whether `tables` is sorted
    /// or not.
    ///
    /// Caller should ensure that `data_sets` should have no overlapped bounds.
    pub fn find_overlap<T: BoundedDataSet>(
        &self,
        data_sets: &[T],
        is_sorted: bool,
    ) -> Option<usize> {
        if is_sorted {
            let (left, right) = self.get_overlap_data_sets(data_sets);
            (left < right).then_some(left)
        } else {
            for (i, table) in data_sets.iter().enumerate() {
                if self.overlap_bound(table.data_bound()) {
                    return Some(i);
                }
            }
            None
        }
    }

    pub fn get_overlap_data_sets<T: BoundedDataSet>(&self, data_sets: &[T]) -> (usize, usize) {
        let left = search(data_sets.len(), |i| {
            !data_sets[i].data_bound().less_than_key(self.lower_bound)
        });
        let right = search(data_sets.len(), |i| {
            self.less_than_key(data_sets[i].lower_bound())
        });
        (left, right)
    }
}

/// `RangedDataSet` is used to get the bound of the data set.
pub trait BoundedDataSet: Sized {
    fn data_bound(&self) -> DataBound<'_>;

    fn lower_bound(&self) -> InnerKey<'_> {
        self.data_bound().lower_bound
    }

    fn overlap_bound(&self, bound: DataBound<'_>) -> bool {
        self.data_bound().overlap_bound(bound)
    }

    fn contains_bound(&self, bound: DataBound<'_>) -> bool {
        self.data_bound().contains_bound(bound)
    }
}

impl BoundedDataSet for kvenginepb::TableCreate {
    fn data_bound(&self) -> DataBound<'_> {
        DataBound::new(
            InnerKey::from_inner_buf(self.get_smallest()),
            InnerKey::from_inner_buf(self.get_biggest()),
            true,
        )
    }
}

impl BoundedDataSet for kvenginepb::BlobCreate {
    fn data_bound(&self) -> DataBound<'_> {
        DataBound::new(
            InnerKey::from_inner_buf(self.get_smallest()),
            InnerKey::from_inner_buf(self.get_biggest()),
            true,
        )
    }
}

pub fn encode_val_to_outer_val_owner(v: Value, outer_val_owner: &mut Vec<u8>) -> Value {
    outer_val_owner.resize(v.encoded_size(), 0);
    v.encode(outer_val_owner.as_mut_slice());
    Value::decode(outer_val_owner)
}

pub fn add_property(buf: &mut Vec<u8>, key: &[u8], val: &[u8]) {
    buf.put_u16_le(key.len() as u16);
    buf.extend_from_slice(key);
    buf.put_u32_le(val.len() as u32);
    buf.extend_from_slice(val);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data_bound_overlap() {
        let cases = vec![
            (
                (1, 2), // t1,
                (3, 4), // t2
                false,  // expected is_overlapping
            ),
            ((0, 0), (0, 0), true),
            ((0, 0), (0, 5), true),
            ((0, 0), (5, 5), false),
            ((0, 10), (0, 0), true),
            ((0, 10), (0, 5), true),
            ((0, 10), (5, 5), true),
            ((0, 10), (5, 10), true),
            ((0, 10), (5, 20), true),
            ((0, 10), (10, 20), true),
            ((0, 10), (20, 20), false),
            ((10, 10), (20, 20), false),
        ];

        for (idx, (t1, t2, expected)) in cases.into_iter().enumerate() {
            let t1 = make_table(t1);
            let t2 = make_table(t2);
            assert_eq!(
                t1.data_bound().overlap_bound(t2.data_bound()),
                expected,
                "case {}",
                idx
            );
            assert_eq!(
                t2.data_bound().overlap_bound(t1.data_bound()),
                expected,
                "case {}",
                idx
            );
        }
    }

    #[test]
    fn test_data_bound_search_overlap() {
        let sorted_tables = vec![(1, 1), (2, 5), (10, 10), (20, 30)]
            .into_iter()
            .map(make_table)
            .collect::<Vec<_>>();
        let unsorted_tables = vec![(10, 10), (1, 1), (2, 5), (20, 30)]
            .into_iter()
            .map(make_table)
            .collect::<Vec<_>>();
        let cases = vec![
            (
                (0, 0), // t,
                None,   // expect_sorted
                None,   // expect_unsorted
            ),
            ((0, 1), Some(0), Some(1)),
            ((0, 10), Some(0), Some(0)),
            ((1, 1), Some(0), Some(1)),
            ((1, 2), Some(0), Some(1)),
            ((1, 10), Some(0), Some(0)),
            ((2, 2), Some(1), Some(2)),
            ((2, 5), Some(1), Some(2)),
            ((2, 10), Some(1), Some(0)),
            ((5, 5), Some(1), Some(2)),
            ((5, 6), Some(1), Some(2)), // overlap with previous one
            ((5, 10), Some(1), Some(0)),
            ((6, 6), None, None),
            ((6, 10), Some(2), Some(0)), // overlap with next one
            ((10, 10), Some(2), Some(0)),
            ((11, 11), None, None),
            ((11, 20), Some(3), Some(3)),
            ((20, 40), Some(3), Some(3)),
            ((40, 40), None, None),
        ];

        for (idx, (t, expect_sorted, expect_unsorted)) in cases.into_iter().enumerate() {
            let t = make_table(t);
            assert_eq!(
                t.data_bound().find_overlap(&sorted_tables, true),
                expect_sorted,
                "case {}",
                idx
            );
            assert_eq!(
                t.data_bound().find_overlap(&unsorted_tables, false),
                expect_unsorted,
                "case {}",
                idx
            );
        }
    }

    fn make_table(t: (usize, usize)) -> kvenginepb::TableCreate {
        let mut table = kvenginepb::TableCreate::default();
        table.set_smallest(format!("{:04}", t.0).into_bytes());
        table.set_biggest(format!("{:04}", t.1).into_bytes());
        table
    }

    #[test]
    fn test_inner_key_debug() {
        let buf = vec![128, 0, 255];
        let inner_key = InnerKey::from_inner_buf(&buf);
        assert_eq!(format!("{:?}", inner_key), "8000FF".to_string());
    }

    #[test]
    fn test_no_prefix_key() {
        let cases: Vec<(&[u8], Option<&[u8]>)> = vec![
            (b"tkey1", Some(b"tkey1")),
            (b"m0n0n", Some(b"m0n0n")),
            (b"x123tkey1", Some(b"tkey1")),
            (b"", None),
            (b"a", None),
            (b"r123", None),
        ];

        for (inner_key, expected) in cases {
            let inner_key = InnerKey::from_inner_buf(inner_key);
            let no_prefix = NoPrefixKey::from_inner_key(&inner_key);
            let expected = expected.map(|x| NoPrefixKey(x));
            assert_eq!(no_prefix, expected);
        }
    }
}
