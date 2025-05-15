// Copyright 2021 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    cmp::Ordering,
    fmt::{Debug, Formatter},
    iter::Iterator as StdIterator,
    mem::size_of,
    ops::Deref,
    ptr, result, slice,
};

use api_version::{api_v2::KEYSPACE_PREFIX_LEN, ApiV2};
use byteorder::{ByteOrder, LittleEndian};
use bytes::{Buf, BufMut};
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

// Do nothing extra but just to make `maybe_async` works.
#[macro_export]
macro_rules! next {
    ($expr:expr) => {{ $expr.next() }};
}

#[macro_export]
macro_rules! next_async {
    ($expr:expr) => {{
        if $expr.is_next_sync() {
            $expr.next()
        } else {
            $expr.next_async().await
        }
    }};
}

// Do nothing extra but just to make `maybe_async` works.
#[macro_export]
macro_rules! next_version {
    ($expr:expr) => {{ $expr.next_version() }};
}

#[macro_export]
macro_rules! next_version_async {
    ($expr:expr) => {{
        if $expr.is_next_version_sync() {
            $expr.next_version()
        } else {
            $expr.next_version_async().await
        }
    }};
}

#[maybe_async::async_trait]
pub trait Iterator: Send {
    // next returns the next entry with different key on the latest version.
    // If old version is needed, call next_version.
    // TODO: add `async`. The method without `async` is a special case that generate
    // stub default implementation as `unimplemented!()`.
    #[maybe_async]
    fn next(&mut self);

    /// `is_next_sync` indicates that sync version of `next` can be used.
    fn is_next_sync(&self) -> bool {
        // TODO: remove default implementation.
        true
    }

    // next_version set the current entry to an older version.
    // The iterator must be valid to call this method.
    // It returns true if there is an older version, returns false if there is no
    // older version. The iterator is still valid and on the same key.
    #[maybe_async]
    fn next_version(&mut self) -> bool;

    /// `is_next_version_sync` indicates that sync version of `next_version` can
    /// be used.
    fn is_next_version_sync(&self) -> bool {
        // TODO: remove default implementation.
        true
    }

    fn rewind(&mut self);
    async fn rewind_async(&mut self) {
        self.rewind();
    }

    fn seek(&mut self, key: InnerKey<'_>);
    async fn seek_async(&mut self, key: InnerKey<'_>) {
        self.seek(key);
    }

    fn key(&self) -> InnerKey<'_>;

    fn value(&self) -> Value;

    fn valid(&self) -> bool;

    #[maybe_async]
    async fn seek_to_version(&mut self, version: u64) -> bool {
        if version >= self.value().version {
            return true;
        }
        while next_version!(self).await {
            if version >= self.value().version {
                return true;
            }
        }
        false
    }

    #[maybe_async]
    async fn next_all_version(&mut self) {
        if !next_version!(self).await {
            next!(self).await
        }
    }

    // NOTE: `rewind_and_dump` has side effect. Use with Caution.
    #[cfg(debug_assertions)]
    fn rewind_and_dump(&mut self) -> Vec<(String, Vec<DumpKv>)> {
        use bytes::Bytes;
        let mut kvs = vec![];
        self.rewind();
        while self.valid() {
            let key = OwnedInnerKey::from(self.key());
            let v = self.value();
            let value = if v.is_deleted() {
                None
            } else {
                Some(Bytes::copy_from_slice(v.get_value()))
            };
            kvs.push(DumpKv {
                key,
                ver: v.version,
                value,
            });
            self.next_all_version();
        }
        vec![(self.tag(), kvs)]
    }

    #[cfg(not(debug_assertions))]
    fn rewind_and_dump(&self) {}

    #[cfg(debug_assertions)]
    fn tag(&self) -> String {
        String::new()
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
//
// The Value format is as follows:
// | meta   | user_meta_len | version |      user_meta      |     value       |
// | 1 byte |    1 byte     | 8 bytes | user_meta_len bytes | value_len bytes |
#[derive(Debug, Copy, Clone)]
pub struct Value {
    /// Points to start of user_meta at offset VALUE_VERSION_OFF +
    /// serialized(version).
    ptr: *const u8,
    /// Bit flags representing the state of the value.
    ///
    /// Each bit in the `meta` field indicates a specific property:
    /// - Bit 1 (BIT_DELETE): Value has been marked for deletion.
    /// - Bit 2 (BIT_HAS_OLD_VERSION): Value has an old version available.
    /// - Bit 3 (BIT_BLOB_REF): Value is a reference to a blob (set if
    ///   `blob_ptr` points to a `BlobRef`).
    pub meta: u8,
    /// User defined opaque meta data,
    user_meta_len: u8,
    /// Length of the data.
    len: u32,
    /// The row version
    pub version: u64,

    /// Pointer to blob data, which can represent two states:
    /// 1. If `BIT_BLOB_REF` is set, this pointer points to a `BlobRef`,
    ///    indicating that the value is a reference to external blob data.
    /// 2. If `BIT_BLOB_REF` is not set, this pointer points to the actual blob
    ///    data stored in memory.
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

    #[cfg(test)]
    pub fn new_with_version(data: &[u8], version: u64) -> Self {
        let len = data.len() as u32;
        let ptr = data.as_ptr();

        Value {
            ptr,
            meta: 0,
            user_meta_len: 0,
            len,
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
    pub fn is_deleted(&self) -> bool {
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
    #[error("IA manager error: {0}")]
    IaMgr(String),
    #[error("Deadline is exceeded: {0}")]
    DeadlineExceeded(String),
    #[error("out of order, previous: {:?}, current: {:?}", .0, .1)]
    OutOfOrder(Vec<u8>, Vec<u8>),
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

#[derive(Clone, Copy, Default, Debug)]
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

#[cfg(test)]
pub async fn new_merge_iterator_async<'a>(
    iters: Vec<Box<dyn Iterator + 'a>>,
    reverse: bool,
) -> Box<dyn Iterator + 'a> {
    use crate::table::AsyncMergeIterator;
    Box::new(AsyncMergeIterator::new(iters, reverse, false))
}

/// Used with caution. The `end_key` must be the EXCLUSIVE end key of a range.
fn is_keyspace_end_key(end_key: &[u8]) -> bool {
    end_key.len() == KEYSPACE_PREFIX_LEN && ApiV2::is_txn_key(end_key)
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
        Self { key }.trim_keyspace()
    }

    pub fn from_outer_key<'b: 'a>(outer_key: &'b [u8]) -> Self {
        Self { key: outer_key }.trim_keyspace()
    }

    /// NOTE: The `outer_end_key` must be the EXCLUSIVE end key of a range.
    pub fn from_outer_end_key<'b: 'a>(outer_end_key: &'b [u8]) -> Self {
        if outer_end_key.is_empty() || is_keyspace_end_key(outer_end_key) {
            Self {
                key: GLOBAL_SHARD_END_KEY,
            }
        } else {
            Self::from_outer_key(outer_end_key)
        }
    }

    fn trim_keyspace(&self) -> Self {
        if ApiV2::is_txn_key(self.key) {
            Self {
                key: &self.key[KEYSPACE_PREFIX_LEN..],
            }
        } else {
            *self
        }
    }

    pub fn slice(&self, start: usize, end: usize) -> Self {
        Self {
            key: &self.key[start..end],
        }
    }
}

impl Deref for InnerKey<'_> {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        self.key
    }
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct OwnedInnerKey {
    inner: bytes::Bytes,
}

impl Debug for OwnedInnerKey {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", &LogValue::key(&self.inner))
    }
}

impl OwnedInnerKey {
    pub fn new(mut inner: bytes::Bytes) -> Self {
        if ApiV2::is_txn_key(inner.chunk()) {
            inner = inner.slice(KEYSPACE_PREFIX_LEN..);
        }
        Self { inner }
    }

    /// NOTE: The `inner_end_key` must be the EXCLUSIVE end key of a range.
    pub fn new_end_key(inner_end_key: bytes::Bytes) -> Self {
        if inner_end_key.is_empty() || is_keyspace_end_key(&inner_end_key) {
            Self {
                inner: bytes::Bytes::copy_from_slice(GLOBAL_SHARD_END_KEY),
            }
        } else {
            Self::new(inner_end_key)
        }
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

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

impl From<InnerKey<'_>> for OwnedInnerKey {
    fn from(key: InnerKey<'_>) -> Self {
        OwnedInnerKey {
            inner: bytes::Bytes::copy_from_slice(key.deref()),
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

    /// Whether the key overlaps the range of data bound.
    ///
    /// ```rust
    /// use kvengine::table::{DataBound, InnerKey};
    /// let k00 = b"k00";
    /// let k10 = b"k10";
    /// let k15 = b"k15";
    /// let k20 = b"k20";
    /// let k30 = b"k30";
    /// let data_bound = DataBound::new(
    ///     InnerKey::from_inner_buf(k10.as_slice()),
    ///     InnerKey::from_inner_buf(k20.as_slice()),
    ///     false,
    /// );
    /// for k in &[k10, k15] {
    ///     assert!(data_bound.overlap_key(InnerKey::from_inner_buf(k.as_slice())));
    /// }
    /// for k in &[k00, k20, k30] {
    ///     assert!(!data_bound.overlap_key(InnerKey::from_inner_buf(k.as_slice())));
    /// }
    /// ```
    pub fn overlap_key(&self, key: InnerKey<'_>) -> bool {
        self.lower_bound <= key && !self.less_than_key(key)
    }

    /// Whether the key overlaps, but not equal to the lower and upper bound.
    ///
    /// ```rust
    /// use kvengine::table::{DataBound, InnerKey};
    /// let k00 = b"k00";
    /// let k10 = b"k10";
    /// let k15 = b"k15";
    /// let k20 = b"k20";
    /// let k30 = b"k30";
    /// let data_bound = DataBound::new(
    ///     InnerKey::from_inner_buf(k10.as_slice()),
    ///     InnerKey::from_inner_buf(k20.as_slice()),
    ///     false,
    /// );
    /// assert!(data_bound.exclusive_overlap_key(InnerKey::from_inner_buf(k15.as_slice())));
    /// for k in &[k00, k10, k20, k30] {
    ///     assert!(!data_bound.exclusive_overlap_key(InnerKey::from_inner_buf(k.as_slice())));
    /// }
    /// ```
    pub fn exclusive_overlap_key(&self, key: InnerKey<'_>) -> bool {
        self.lower_bound < key && !self.less_or_equal_key(key)
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

    /// Whether the range is less than the key.
    ///
    /// ```rust
    /// use kvengine::table::{DataBound, InnerKey};
    /// let k00 = b"k00";
    /// let k10 = b"k10";
    /// let k15 = b"k15";
    /// let k20 = b"k20";
    /// let k30 = b"k30";
    /// let data_bound = DataBound::new(
    ///     InnerKey::from_inner_buf(k10.as_slice()),
    ///     InnerKey::from_inner_buf(k20.as_slice()),
    ///     false,
    /// );
    /// for k in &[k20, k30] {
    ///     assert!(data_bound.less_than_key(InnerKey::from_inner_buf(k.as_slice())));
    /// }
    /// for k in &[k00, k10, k15] {
    ///     assert!(!data_bound.less_than_key(InnerKey::from_inner_buf(k.as_slice())));
    /// }
    ///
    /// let data_bound = DataBound::new(
    ///     InnerKey::from_inner_buf(k10.as_slice()),
    ///     InnerKey::from_inner_buf(k20.as_slice()),
    ///     true,
    /// );
    /// assert!(!data_bound.less_than_key(InnerKey::from_inner_buf(k20.as_slice())));
    /// ```
    pub fn less_than_key(&self, key: InnerKey<'_>) -> bool {
        match self.upper_bound.cmp(&key) {
            Ordering::Less => true,
            Ordering::Equal => !self.upper_inclusive,
            Ordering::Greater => false,
        }
    }

    /// Whether the range is less than the key, or (upper bound) equal to the
    /// key.
    ///
    /// ```rust
    /// use kvengine::table::{DataBound, InnerKey};
    /// let k00 = b"k00";
    /// let k10 = b"k10";
    /// let k15 = b"k15";
    /// let k20 = b"k20";
    /// let k30 = b"k30";
    /// let data_bound = DataBound::new(
    ///     InnerKey::from_inner_buf(k10.as_slice()),
    ///     InnerKey::from_inner_buf(k20.as_slice()),
    ///     true,
    /// );
    /// for k in &[k20, k30] {
    ///     assert!(data_bound.less_or_equal_key(InnerKey::from_inner_buf(k.as_slice())));
    /// }
    /// for k in &[k00, k10, k15] {
    ///     assert!(!data_bound.less_or_equal_key(InnerKey::from_inner_buf(k.as_slice())));
    /// }
    /// ```
    pub fn less_or_equal_key(&self, key: InnerKey<'_>) -> bool {
        match self.upper_bound.cmp(&key) {
            Ordering::Less | Ordering::Equal => true,
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

impl BoundedDataSet for kvenginepb::VectorIndexFile {
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

#[cfg(debug_assertions)]
pub struct DumpKv {
    pub key: OwnedInnerKey,
    pub ver: u64,
    pub value: Option<bytes::Bytes>,
}

#[cfg(debug_assertions)]
impl std::fmt::Debug for DumpKv {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}:{}:{:?}", self.key, self.ver, self.value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_value_new() {
        let value = Value::new(); // Create a new Value instance

        // Assert the fields
        assert_eq!(value.meta, 0); // Meta should be default (0)
        assert_eq!(value.user_meta_len, 0); // User metadata length should be 0
        assert_eq!(value.len, 0); // Length should be 0
        assert_eq!(value.version, 0); // Version should be default (0)
        assert!(value.ptr.is_null()); // Pointer should be null
        assert!(value.blob_ptr.is_null()); // Blob pointer should be null
        assert!(value.is_empty()); // Value should be empty
    }

    #[test]
    fn test_value_new_with_version() {
        let value = Value::new_with_version(b"test_value", 100);
        assert_eq!(value.version, 100);
        assert_eq!(value.value_len(), 10); // Length of "test_value"
        assert!(!value.is_empty());
    }

    #[test]
    fn test_value_new_with_meta_and_version() {
        let user_meta = b"meta_data";
        let meta = BIT_DELETE; // Example meta flag
        let value = Value::new_with_meta_version(meta, 100, user_meta.len() as u8, user_meta);

        // Assert the fields
        assert_eq!(value.version, 100);
        assert_eq!(value.user_meta_len, user_meta.len() as u8); // Length of user_meta
        assert_eq!(value.meta, meta); // Check the meta field
        assert_eq!(value.value_len(), 0); // Length of value should be 0 since we didn't set it
        assert!(!value.is_empty());
    }

    #[test]
    fn test_value_new_tombstone() {
        let version = 100; // Example version
        let value = Value::new_tombstone(version); // Create a new tombstone Value instance

        // Assert the fields
        assert_eq!(value.meta, BIT_DELETE); // Meta should indicate deletion
        assert_eq!(value.user_meta_len, 0); // User metadata length should be 0
        assert_eq!(value.len, 0); // Length should be 0
        assert_eq!(value.version, version); // Version should be the one provided
        assert!(value.ptr.is_null()); // Pointer should be null
        assert!(value.blob_ptr.is_null()); // Blob pointer should be null
    }

    #[test]
    fn test_value_is_deleted() {
        let mut value = Value::new_with_version(b"", 100);
        value.meta |= BIT_DELETE; // Mark as deleted
        assert!(value.is_deleted());
    }

    #[test]
    fn test_value_is_blob_ref() {
        let mut value = Value::new_with_version(b"", 100);
        value.set_blob_ref(); // Set as blob reference
        assert!(value.is_blob_ref());
    }

    #[test]
    fn test_value_encoded_size() {
        let user_meta = b"user_meta"; // Example user metadata
        let data = b"test_value"; // Actual value

        // Create a combined data buffer that includes user_meta and data
        let combined_data = [&user_meta[..], &data[..]].concat();

        // Create a Value instance with user metadata
        let value = Value::new_with_meta_version(
            1,                        // Meta (set to 1 to indicate some state)
            100,                      // Version
            user_meta.len() as u8,    // Length of user metadata
            combined_data.as_slice(), // Combined data
        );

        // Calculate the expected encoded size
        let expected_size = 1 + 1 + VALUE_VERSION_LEN + user_meta.len() + data.len(); // Meta + user_meta_len + version + user_meta + value

        // Assert the encoded size
        assert_eq!(value.encoded_size(), expected_size); // Check the encoded size
    }

    #[test]
    fn test_encode_buf() {
        let meta: u8 = 1; // Example meta value
        let user_meta = b"user_meta"; // Example user metadata
        let version: u64 = 100; // Example version
        let value = b"test_value"; // Actual value

        // Call the encode_buf function
        let encoded = Value::encode_buf(meta, user_meta, version, value);

        // Calculate the expected size
        let expected_size = 1 + 1 + VALUE_VERSION_LEN + user_meta.len() + value.len(); // Meta + user_meta_len + version + user_meta + value

        // Assert the encoded size
        assert_eq!(encoded.len(), expected_size);

        // Verify the encoded data
        assert_eq!(encoded[0], meta); // Check meta
        assert_eq!(encoded[1], user_meta.len() as u8); // Check user metadata length

        let version_from_encoded = LittleEndian::read_u64(
            &encoded[VALUE_VERSION_OFF..VALUE_VERSION_OFF + VALUE_VERSION_LEN],
        );
        assert_eq!(version_from_encoded, version); // Check version

        // Check user metadata
        assert_eq!(
            &encoded[VALUE_VERSION_OFF + VALUE_VERSION_LEN
                ..VALUE_VERSION_OFF + VALUE_VERSION_LEN + user_meta.len()],
            user_meta
        );

        // Check value data
        assert_eq!(
            &encoded[VALUE_VERSION_OFF + VALUE_VERSION_LEN + user_meta.len()..],
            value
        );
    }

    #[test]
    fn test_value_encode() {
        let user_meta = b"user_meta"; // Example user metadata
        let data = b"test_value";
        let combined_data = [&user_meta[..], &data[..]].concat();
        let value = Value::new_with_meta_version(
            1,                     // Meta
            100,                   // Version
            user_meta.len() as u8, // Length of user metadata
            combined_data.as_slice(),
        );

        let mut buf = vec![0; value.encoded_size()];
        value.encode(&mut buf);

        // Verify the encoded data
        assert_eq!(buf[0], 1); // Meta
        assert_eq!(buf[1], user_meta.len() as u8); // User metadata length
        let version =
            LittleEndian::read_u64(&buf[VALUE_VERSION_OFF..VALUE_VERSION_OFF + VALUE_VERSION_LEN]);
        assert_eq!(version, 100); // Compare the extracted version with 100
        assert_eq!(
            &buf[VALUE_VERSION_OFF + VALUE_VERSION_LEN
                ..VALUE_VERSION_OFF + VALUE_VERSION_LEN + user_meta.len()],
            user_meta
        ); // User metadata
        assert_eq!(
            &buf[VALUE_VERSION_OFF + VALUE_VERSION_LEN + user_meta.len()..],
            b"test_value"
        ); // Value data
    }

    #[test]
    fn test_value_decode() {
        let user_meta = b"user_meta"; // Example user metadata
        let data = b"test_value"; // Actual value

        // Create a buffer with enough space for meta, user_meta_len, version, and value
        let mut buf = vec![0; 29];
        buf[0] = 1; // Meta
        buf[1] = user_meta.len() as u8; // User meta length
        LittleEndian::write_u64(&mut buf[VALUE_VERSION_OFF..], 100); // Version
        let user_meta_start = VALUE_VERSION_OFF + VALUE_VERSION_LEN;
        buf[user_meta_start..user_meta_start + user_meta.len()].copy_from_slice(user_meta); // User metadata
        let value_start = user_meta_start + user_meta.len();
        buf[value_start..value_start + data.len()].copy_from_slice(data); // Value data

        // Decode the value from the buffer
        let value = Value::decode(&buf);

        // Assert the fields
        assert_eq!(value.meta, 1); // Check the meta field
        assert_eq!(value.user_meta_len, user_meta.len() as u8); // Length of user_meta
        assert_eq!(value.version, 100);
        assert_eq!(value.user_meta(), user_meta); // Check user metadata
        assert_eq!(value.get_value(), data); // Check the actual value
    }

    #[test]
    fn test_value_encode_with_blob_ref() {
        let mut value = Value::new_with_version(b"test_value", 100);
        value.set_blob_ref(); // Set as blob reference
        let mut buf = vec![0; value.encoded_size_with_blob_ref()];
        value.encode_with_blob_ref(
            &mut buf,
            BlobRef {
                fid: 42,
                offset: 100,
                len: 100,
                original_len: 100,
            },
        );

        // Verify the encoded data
        assert_eq!(buf[0], 4); // Meta with blob ref
        assert_eq!(buf[1], 0); // User meta length
        let version =
            LittleEndian::read_u64(&buf[VALUE_VERSION_OFF..VALUE_VERSION_OFF + VALUE_VERSION_LEN]);
        assert_eq!(version, 100); // Compare the extracted version with 100
        // Check blob reference data
        let blob_ref = BlobRef::deserialize(&buf[VALUE_VERSION_OFF + VALUE_VERSION_LEN..]);
        assert_eq!(blob_ref.fid, 42);
        assert_eq!(blob_ref.offset, 100);
        assert_eq!(blob_ref.len, 100);
        assert_eq!(blob_ref.original_len, 100);
    }

    #[test]
    fn test_value_get_blob_ref() {
        let mut value = Value::new_with_version(b"", 100);
        value.set_blob_ref(); // Set as blob reference
        let buf = vec![0; 100];
        value.fill_in_blob(&buf);
        assert!(!value.is_blob_ref());
    }

    #[test]
    fn test_get_version() {
        // Prepare a buffer with a known version
        let version_value: u64 = 12345;
        let mut buf = vec![0; VALUE_VERSION_LEN];
        LittleEndian::write_u64(&mut buf, version_value); // Write the version into the buffer

        // Call the get_version function
        let version = Value::get_version(&buf);

        // Assert that the retrieved version matches the expected value
        assert_eq!(version, version_value);
    }

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

    #[test]
    fn test_data_bound_exclusive_overlap_key() {
        let start = make_key(10);
        let end = make_key(20);
        let range_inc = DataBound::new(start.as_ref(), end.as_ref(), true);
        let range_exc = DataBound::new(start.as_ref(), end.as_ref(), false);

        let cases = vec![
            (0, false), // (key, expected)
            (10, false),
            (11, true),
            (19, true),
            (20, false),
            (21, false),
        ];

        for (idx, (key, expected)) in cases.into_iter().enumerate() {
            let key = make_key(key);
            for range in &[range_inc, range_exc] {
                assert_eq!(
                    range.exclusive_overlap_key(key.as_ref()),
                    expected,
                    "case {}",
                    idx
                );
            }
        }
    }

    fn make_table(t: (usize, usize)) -> kvenginepb::TableCreate {
        let mut table = kvenginepb::TableCreate::default();
        table.set_smallest(format!("{:04}", t.0).into_bytes());
        table.set_biggest(format!("{:04}", t.1).into_bytes());
        table
    }

    fn make_key(i: usize) -> OwnedInnerKey {
        OwnedInnerKey::new(format!("{:04}", i).into_bytes().into())
    }

    #[test]
    fn test_inner_key() {
        let buf = vec![128, 0, 255];
        let inner_key = InnerKey::from_inner_buf(&buf);
        assert_eq!(format!("{:?}", inner_key), "8000FF".to_string());

        let keyspace_end_key = vec![0x78, 0, 0, 2];
        let inner_key = InnerKey::from_outer_end_key(&keyspace_end_key);
        assert_eq!(GLOBAL_SHARD_END_KEY, inner_key.deref());

        let owned_inner_key = OwnedInnerKey::new_end_key(bytes::Bytes::from(keyspace_end_key));
        assert_eq!(GLOBAL_SHARD_END_KEY, owned_inner_key.as_ref().deref());
    }
}
