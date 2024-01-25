// Copyright 2023 TiKV Project Authors. Licensed under Apache-2.0.
#![allow(dead_code)]

use std::{ops::Deref, sync::Arc};

use bytes::{Buf, BufMut, Bytes};
use moka::sync::SegmentedCache;

use crate::{
    table,
    table::{
        search,
        sstable::{key_diff_idx, BlockCacheKey, EntrySlice, File, TtlCache},
        Error, InnerKey, Result,
    },
};

const TXN_FILE_PROP_CHECK_NON_EXIST_COUNT: &str = "check_ne";
const TXN_FILE_PROP_INSERT_COUNT: &str = "insert";

const TXN_FILE_FORMAT: u16 = 1;
const TXN_FILE_CHECKSUM_TYPE: u8 = 1;
const TXN_FILE_MAGIC: u32 = 2785588940;
const U32_SIZE: usize = std::mem::size_of::<u32>();

// Op values reference kvproto::kvrpcpb::Op.
pub const OP_PUT: u8 = 0;
pub const OP_DELETE: u8 = 1;
pub const OP_LOCK: u8 = 2;
pub const OP_INSERT: u8 = 4;
pub const OP_CHECK_NOT_EXIST: u8 = 6;

#[derive(Clone)]
pub struct TxnChunk {
    inner: Arc<TxnChunkInner>,
}

impl TxnChunk {
    pub fn new(
        file: Arc<dyn File>,
        cache: Option<SegmentedCache<BlockCacheKey, Bytes>>,
    ) -> Result<Self> {
        let inner = TxnChunkInner::new(file, cache)?;
        Ok(Self {
            inner: Arc::new(inner),
        })
    }

    fn get_value(&self, key: InnerKey<'_>) -> Option<TxnChunkIterator> {
        if key < self.index.smallest() || key > self.index.biggest() {
            return None;
        }
        let key_hash = farmhash::fingerprint64(key.as_ref());
        let hash_index = self.load_hash_index().unwrap();
        if let Some(key_addr) = hash_index.get_entry(key_hash) {
            let mut iter = TxnChunkIterator::new(self.clone(), false);
            iter.locate_key(key_addr);
            if iter.key() != key {
                // There may be hash conflict.
                warn!("hash conflict");
                iter.seek(key);
            }
            if iter.valid() && iter.key() == key {
                return Some(iter);
            }
        }
        None
    }
}

impl Deref for TxnChunk {
    type Target = TxnChunkInner;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

pub struct TxnChunkInner {
    file: Arc<dyn File>,
    cache: Option<SegmentedCache<BlockCacheKey, Bytes>>,
    footer: TxnChunkFooter,
    index: TxnChunkIndex,
    hash_index: TtlCache<TxnChunkHashIndex>,
    inserts: u32,
    check_non_exists: u32,
}

#[derive(Clone)]
struct TxnChunkIndex {
    block_offs: Bytes,
    key_offs: Bytes,
    keys: Bytes,
    num_blocks: usize,
}

impl TxnChunkIndex {
    fn new(mut idx_data: Bytes) -> Self {
        let num_blocks = idx_data.get_u32_le() as usize;
        let block_offs = idx_data.slice(..(num_blocks + 1) * U32_SIZE);
        idx_data.advance((num_blocks + 1) * U32_SIZE);
        let key_offs = idx_data.slice(..(num_blocks + 1) * U32_SIZE);
        idx_data.advance((num_blocks + 1) * U32_SIZE);
        Self {
            block_offs,
            key_offs,
            keys: idx_data,
            num_blocks,
        }
    }

    fn get_block_key_off(&self, i: usize) -> usize {
        (&self.key_offs[i * U32_SIZE..]).get_u32_le() as usize
    }

    fn block_key(&self, i: usize) -> &[u8] {
        let start_off = self.get_block_key_off(i);
        let end_off = self.get_block_key_off(i + 1);
        &self.keys[start_off..end_off]
    }

    fn seek_block(&self, key: &[u8]) -> usize {
        search(self.num_blocks, |i| self.block_key(i) > key)
    }

    fn get_block_off(&self, i: usize) -> usize {
        (&self.block_offs[i * U32_SIZE..]).get_u32_le() as usize
    }

    pub fn smallest(&self) -> InnerKey<'_> {
        InnerKey::from_inner_buf(self.block_key(0))
    }

    pub fn biggest(&self) -> InnerKey<'_> {
        let off = self.get_block_key_off(self.num_blocks);
        InnerKey::from_inner_buf(&self.keys[off..])
    }
}

impl TxnChunkInner {
    pub fn new(
        file: Arc<dyn File>,
        cache: Option<SegmentedCache<BlockCacheKey, Bytes>>,
    ) -> Result<Self> {
        let footer = Self::load_footer(&file)?;
        let idx_length = footer.hash_index_offset - footer.index_offset;
        let raw_idx_data = file.read(footer.index_offset as u64, idx_length as usize)?;
        let idx_data = Self::validate_and_trim_checksum(raw_idx_data)?;
        let properties_length =
            file.size() as usize - footer.properties_offset as usize - TXN_FILE_CHUNK_FOOTER_SIZE;
        let raw_properties = file.read(footer.properties_offset as u64, properties_length)?;
        let properties = Self::validate_and_trim_checksum(raw_properties)?;
        let mut inserts = 0;
        let mut check_non_exists = 0;
        let mut prop_slice = properties.chunk();
        while !prop_slice.is_empty() {
            let (key, mut val, remained) = Self::parse_prop_data(prop_slice);
            if key == TXN_FILE_PROP_INSERT_COUNT.as_bytes() {
                inserts = val.get_u32_le();
            } else if key == TXN_FILE_PROP_CHECK_NON_EXIST_COUNT.as_bytes() {
                check_non_exists = val.get_u32_le();
            }
            prop_slice = remained;
        }
        let index = TxnChunkIndex::new(idx_data);
        let chunk = Self {
            file,
            cache,
            footer,
            index,
            hash_index: TtlCache::default(),
            check_non_exists,
            inserts,
        };
        chunk.load_hash_index()?;
        Ok(chunk)
    }

    fn parse_prop_data(mut prop_data: &[u8]) -> (&[u8], &[u8], &[u8]) {
        let key_len = prop_data.get_u16_le() as usize;
        let key = &prop_data[..key_len];
        prop_data.advance(key_len);
        let val_len = prop_data.get_u32_le() as usize;
        let val = &prop_data[..val_len];
        prop_data.advance(val_len);
        (key, val, prop_data)
    }

    fn load_footer(file: &Arc<dyn File>) -> Result<TxnChunkFooter> {
        let mut footer = TxnChunkFooter::default();
        let footer_off = file.size() - TXN_FILE_CHUNK_FOOTER_SIZE as u64;
        let footer_buf = file.read(footer_off, TXN_FILE_CHUNK_FOOTER_SIZE)?;
        footer.unmarshal(&footer_buf);
        assert_eq!(footer.magic, TXN_FILE_MAGIC);
        assert_eq!(footer.format_version, TXN_FILE_FORMAT);
        Ok(footer)
    }

    pub fn id(&self) -> u64 {
        self.file.id()
    }

    pub fn size(&self) -> usize {
        self.file.size() as usize
    }

    pub fn get_check_non_exists(&self) -> u32 {
        self.check_non_exists
    }

    pub fn get_inserts(&self) -> u32 {
        self.inserts
    }

    pub fn load_block(&self, pos: usize) -> Result<Bytes> {
        let block_off = self.index.get_block_off(pos);
        let next_block_off = self.index.get_block_off(pos + 1);
        let length = next_block_off - block_off;
        match &self.cache {
            Some(cache) => {
                let cache_key = BlockCacheKey::new(self.file.id(), block_off as u32);
                cache
                    .try_get_with(cache_key, || {
                        self.read_block_from_file(block_off as u64, length)
                    })
                    .map_err(|err| err.as_ref().clone())
            }
            None => self.read_block_from_file(block_off as u64, length),
        }
    }

    fn read_block_from_file(&self, offset: u64, length: usize) -> Result<Bytes> {
        let raw_block = self.file.read(offset, length)?;
        Self::validate_and_trim_checksum(raw_block)
    }

    fn validate_and_trim_checksum(data: Bytes) -> Result<Bytes> {
        if data.len() < 4 {
            return Err(Error::InvalidChecksum(String::from("data is too short")));
        }
        let content_len = data.len() - 4;
        let checksum = (&data[content_len..]).get_u32_le();
        let content = data.slice(..data.len() - 4);
        let got_checksum = crc32c::crc32c(&content);
        if checksum != got_checksum {
            return Err(Error::InvalidChecksum(format!(
                "checksum mismatch expect {} got {}",
                checksum, got_checksum
            )));
        }
        Ok(content)
    }

    fn load_hash_index(&self) -> Result<Arc<TxnChunkHashIndex>> {
        self.hash_index
            .get(|| -> Result<TxnChunkHashIndex> { self.init_hash_index() })
    }

    fn init_hash_index(&self) -> Result<TxnChunkHashIndex> {
        let offset = self.footer.hash_index_offset as u64;
        let length = (self.footer.properties_offset as usize) - offset as usize;
        let raw_hash_idx_data = self.file.read(offset, length)?;
        let hash_idx_data = Self::validate_and_trim_checksum(raw_hash_idx_data).unwrap();
        Ok(TxnChunkHashIndex::new(hash_idx_data))
    }
}

struct TxnChunkIterator {
    chunk: TxnChunk,
    block_iter: TxnChunkBlockIterator,
    num_blocks: usize,
    block_pos: usize,
    reverse: bool,
}

impl TxnChunkIterator {
    fn new(chunk: TxnChunk, reverse: bool) -> Self {
        let num_blocks = chunk.index.num_blocks;
        Self {
            chunk,
            block_iter: TxnChunkBlockIterator::default(),
            num_blocks,
            block_pos: 0,
            reverse,
        }
    }

    fn seek_inner(&mut self, key: &[u8]) {
        self.block_pos = self.chunk.index.seek_block(key).saturating_sub(1);
        if self.load_block() {
            self.block_iter.seek(key);
            if self.block_iter.err.is_some() && self.block_pos + 1 < self.num_blocks {
                self.block_pos += 1;
                if self.load_block() {
                    self.block_iter.seek(key);
                }
            }
        }
    }

    fn load_block(&mut self) -> bool {
        match self.chunk.load_block(self.block_pos) {
            Ok(block) => {
                let block_key = self.chunk.index.block_key(self.block_pos);
                self.block_iter.set_block(block_key, block);
                true
            }
            Err(err) => {
                self.block_iter.err = Some(err);
                false
            }
        }
    }

    fn next_inner(&mut self) {
        self.block_iter.next();
        if self.valid() {
            return;
        }
        if self.block_pos + 1 < self.num_blocks {
            self.block_pos += 1;
            if self.load_block() {
                self.block_iter.set_idx(0);
            }
        }
    }

    fn prev_inner(&mut self) {
        self.block_iter.prev();
        if self.valid() {
            return;
        }
        if self.block_pos > 0 {
            self.block_pos -= 1;
            if self.load_block() {
                self.block_iter.set_idx(self.block_iter.num_keys as i32 - 1);
            }
        }
    }

    fn locate_key(&mut self, key_addr: KeyAddr) {
        self.block_pos = key_addr.block_idx as usize;
        if self.load_block() {
            self.block_iter.set_idx(key_addr.key_idx as i32);
        }
    }

    fn next(&mut self) {
        if self.reverse {
            self.prev_inner();
        } else {
            self.next_inner();
        }
    }

    fn rewind(&mut self) {
        if self.reverse {
            self.block_pos = self.num_blocks - 1;
            if self.load_block() {
                self.block_iter.set_idx(self.block_iter.num_keys as i32 - 1);
            }
        } else {
            self.block_pos = 0;
            if self.load_block() {
                self.block_iter.set_idx(0);
            }
        }
    }

    fn seek(&mut self, key: InnerKey<'_>) {
        if self.reverse {
            if self.chunk.index.biggest() < key {
                self.rewind();
                return;
            }
            if key < self.chunk.index.smallest() {
                self.block_iter.err = Some(Error::Eof);
                return;
            }
        } else {
            if key < self.chunk.index.smallest() {
                self.rewind();
                return;
            }
            if self.chunk.index.biggest() < key {
                self.block_iter.err = Some(Error::Eof);
                return;
            }
        }
        self.seek_inner(key.deref());
        if self.reverse && self.key() > key {
            self.prev_inner();
        }
    }

    fn key(&self) -> InnerKey<'_> {
        InnerKey::from_inner_buf(&self.block_iter.key_buf)
    }

    fn get_value(&self) -> &[u8] {
        self.block_iter.get_val()
    }

    fn valid(&self) -> bool {
        self.block_iter.num_keys > 0 && self.block_iter.err.is_none()
    }
}

#[derive(Default)]
struct TxnChunkBlockIterator {
    num_keys: usize,
    key_buf: Vec<u8>,
    common_prefix_len: usize,
    entry_offs: Bytes,
    block_data: Bytes,
    entry_idx: i32,
    op: u8,
    val_start: usize,
    val_end: usize,
    err: Option<table::Error>,
}

impl TxnChunkBlockIterator {
    fn set_block(&mut self, block_key: &[u8], mut block: Bytes) {
        self.num_keys = block.get_u32_le() as usize;
        let common_prefix_len = block.get_u16_le() as usize;
        self.key_buf.truncate(0);
        self.key_buf
            .extend_from_slice(&block_key[..common_prefix_len]);
        self.common_prefix_len = common_prefix_len;
        self.entry_offs = block.slice(..(self.num_keys + 1) * U32_SIZE);
        block.advance(self.entry_offs.len());
        self.block_data = block;
    }

    fn seek(&mut self, key: &[u8]) {
        let common_prefix = &self.key_buf[..self.common_prefix_len];
        if key.len() <= common_prefix.len() {
            if key <= common_prefix {
                self.set_idx(0);
            } else {
                self.set_idx(self.num_keys as i32);
            }
            return;
        }
        use std::cmp::Ordering::*;
        match &key[..common_prefix.len()].cmp(common_prefix) {
            Less => {
                self.set_idx(0);
                return;
            }
            Greater => {
                self.set_idx(self.num_keys as i32);
                return;
            }
            Equal => {}
        };
        let diff_key = &key[common_prefix.len()..];
        let found_idx = search(self.num_keys, |i| {
            let entry_start = self.get_entry_off(i);
            let entry_end = self.get_entry_off(i + 1);
            let mut entry = &self.block_data[entry_start..entry_end];
            let key_len = entry.get_u16_le() as usize;
            &entry[..key_len] >= diff_key
        });
        self.set_idx(found_idx as i32);
    }

    fn set_idx(&mut self, i: i32) {
        self.entry_idx = i;
        if self.entry_idx < 0 || self.entry_idx >= self.num_keys as i32 {
            self.err = Some(table::Error::Eof);
        } else {
            self.err = None;
        }
        if self.err.is_none() {
            let entry_start = self.get_entry_off(self.entry_idx as usize);
            let entry_end = self.get_entry_off(self.entry_idx as usize + 1);
            let mut entry = &self.block_data[entry_start..entry_end];
            let key_len = entry.get_u16_le() as usize;
            let diff_key = &entry[..key_len];
            entry.advance(key_len);
            self.key_buf.truncate(self.common_prefix_len);
            self.key_buf.extend_from_slice(diff_key);
            self.op = entry.get_u8();
            self.val_start = entry_start + 2 + key_len + 1;
            self.val_end = entry_end;
        }
    }

    fn get_entry_off(&self, i: usize) -> usize {
        (&self.entry_offs[i * U32_SIZE..]).get_u32_le() as usize
    }

    fn get_val(&self) -> &[u8] {
        &self.block_data[self.val_start..self.val_end]
    }

    fn next(&mut self) {
        self.set_idx(self.entry_idx + 1);
    }

    fn prev(&mut self) {
        self.set_idx(self.entry_idx - 1);
    }
}

#[derive(Default)]
pub struct TxnChunkBuilder {
    data_buf: Vec<u8>,
    block: TxnChunkBlockBuffer,
    block_keys: EntrySlice,
    block_offsets: Vec<u32>,
    idx_buf: Vec<u8>,
    target_block_entries: usize,
    biggest_key: Vec<u8>,
    hash_idx_builder: HashIndexBuilder,
    insert_count: u32,
    check_not_exist_count: u32,
}

impl TxnChunkBuilder {
    pub fn new(target_block_entries: usize) -> Self {
        let mut builder = Self::default();
        builder.target_block_entries = target_block_entries;
        builder
    }

    pub fn add_entry(&mut self, key: &[u8], op: u8, val: &[u8]) {
        let key_hash = farmhash::fingerprint64(key);
        let block_idx = self.block_offsets.len() as u16;
        let key_idx = self.block.tmp_keys.length() as u16;
        self.hash_idx_builder
            .add_key_hash(key_hash, KeyAddr::new(block_idx, key_idx));
        self.block.tmp_ops.push(op);
        self.block.tmp_keys.append(key);
        self.block.tmp_vals.append(val);
        if self.block.tmp_keys.length() == self.target_block_entries {
            self.finish_block();
        }
    }

    fn build_properties(&self, buf: &mut Vec<u8>) {
        if self.check_not_exist_count > 0 {
            Self::add_property(
                buf,
                TXN_FILE_PROP_CHECK_NON_EXIST_COUNT.as_bytes(),
                &self.check_not_exist_count.to_le_bytes(),
            );
        }
        if self.insert_count > 0 {
            Self::add_property(
                buf,
                TXN_FILE_PROP_INSERT_COUNT.as_bytes(),
                &self.insert_count.to_le_bytes(),
            );
        }
        let checksum = crc32c::crc32c(buf);
        buf.put_u32_le(checksum);
    }

    fn add_property(buf: &mut Vec<u8>, key: &[u8], val: &[u8]) {
        buf.put_u16_le(key.len() as u16);
        buf.put_slice(key);
        buf.put_u32_le(val.len() as u32);
        buf.put_slice(val);
    }

    fn finish_block(&mut self) {
        self.block_keys.append(self.block.tmp_keys.get_entry(0));
        self.block_offsets.push(self.data_buf.len() as u32);
        let common_prefix_len = self.block.common_prefix_len();
        let num_entries = self.block.tmp_keys.length();
        let block_offset = self.data_buf.len();
        self.data_buf.put_u32_le(num_entries as u32);
        self.data_buf.put_u16_le(common_prefix_len as u16);
        let mut entry_offset = 0u32;
        for i in 0..num_entries {
            self.data_buf.put_u32_le(entry_offset);
            let key = self.block.tmp_keys.get_entry(i);
            let val = self.block.tmp_vals.get_entry(i);
            entry_offset += (2 + key.len() + 1 + val.len() - common_prefix_len) as u32;
        }
        self.data_buf.put_u32_le(entry_offset);
        for i in 0..num_entries {
            let key = self.block.tmp_keys.get_entry(i);
            self.data_buf
                .put_u16_le((key.len() - common_prefix_len) as u16);
            self.data_buf.extend_from_slice(&key[common_prefix_len..]);
            let op = self.block.tmp_ops[i];
            match op {
                OP_INSERT => self.insert_count += 1,
                OP_CHECK_NOT_EXIST => self.check_not_exist_count += 1,
                _ => {}
            }
            self.data_buf.push(op);
            let val = self.block.tmp_vals.get_entry(i);
            self.data_buf.extend_from_slice(val);
        }
        let checksum = crc32c::crc32c(&self.data_buf[block_offset..]);
        self.data_buf.put_u32_le(checksum);
        self.biggest_key.truncate(0);
        self.biggest_key
            .extend_from_slice(self.block.tmp_keys.get_last());
        self.block.reset();
    }

    fn build_index(&mut self) {
        let num_blocks = self.block_keys.length();
        self.idx_buf.put_u32_le(num_blocks as u32);
        for i in 0..num_blocks {
            let block_offset = self.block_offsets[i];
            self.idx_buf.put_u32_le(block_offset);
        }
        self.idx_buf.put_u32_le(self.data_buf.len() as u32);
        let mut offset = 0u32;
        for i in 0..num_blocks {
            self.idx_buf.put_u32_le(offset);
            let block_key = self.block_keys.get_entry(i);
            offset += block_key.len() as u32;
        }
        self.idx_buf.put_u32_le(offset);
        for i in 0..num_blocks {
            let block_key = self.block_keys.get_entry(i);
            self.idx_buf.extend_from_slice(block_key)
        }
        self.idx_buf.extend_from_slice(&self.biggest_key);
        let checksum = crc32c::crc32c(&self.idx_buf);
        self.idx_buf.put_u32_le(checksum);
    }

    pub fn finish(&mut self, data_buf: &mut Vec<u8>) {
        if self.block.length() > 0 {
            self.finish_block();
        }
        self.build_index();
        let (bucket_buf, entry_buf) = self.hash_idx_builder.build();
        let mut hash_index_checksum = crc32c::crc32c(&bucket_buf);
        hash_index_checksum = crc32c::crc32c_append(hash_index_checksum, &entry_buf);
        let hash_index_len = bucket_buf.len() + entry_buf.len() + 4;
        let mut props = vec![];
        self.build_properties(&mut props);
        let size = self.data_buf.len()
            + self.idx_buf.len()
            + hash_index_len
            + props.len()
            + TXN_FILE_CHUNK_FOOTER_SIZE;
        data_buf.reserve(size);
        data_buf.extend_from_slice(&self.data_buf);
        data_buf.extend_from_slice(&self.idx_buf);
        data_buf.extend_from_slice(&bucket_buf);
        data_buf.extend_from_slice(&entry_buf);
        data_buf.put_u32_le(hash_index_checksum);
        data_buf.extend_from_slice(&props);
        let index_offset = self.data_buf.len() as u32;
        let hash_index_offset = index_offset + self.idx_buf.len() as u32;
        let properties_offset = hash_index_offset + hash_index_len as u32;
        let footer = TxnChunkFooter {
            index_offset,
            hash_index_offset,
            properties_offset,
            reserved: 0,
            checksum_type: TXN_FILE_CHECKSUM_TYPE,
            format_version: TXN_FILE_FORMAT,
            magic: TXN_FILE_MAGIC,
        };
        data_buf.extend_from_slice(footer.marshal());
    }
}

#[derive(Default)]
struct TxnChunkBlockBuffer {
    tmp_keys: EntrySlice,
    tmp_ops: Vec<u8>,
    tmp_vals: EntrySlice,
}

impl TxnChunkBlockBuffer {
    fn length(&self) -> usize {
        self.tmp_keys.length()
    }

    fn common_prefix_len(&self) -> usize {
        let first_key = self.tmp_keys.get_entry(0);
        let last_key = self.tmp_keys.get_last();
        key_diff_idx(first_key, last_key)
    }

    fn reset(&mut self) {
        self.tmp_keys.reset();
        self.tmp_ops.truncate(0);
        self.tmp_vals.reset();
    }
}

pub const TXN_FILE_CHUNK_FOOTER_SIZE: usize = std::mem::size_of::<TxnChunkFooter>();

#[repr(C)]
#[derive(Default, Clone, Copy, Debug)]
struct TxnChunkFooter {
    index_offset: u32,
    hash_index_offset: u32,
    properties_offset: u32,
    reserved: u8,
    checksum_type: u8,
    format_version: u16,
    magic: u32,
}

impl TxnChunkFooter {
    pub fn marshal(&self) -> &[u8] {
        let footer_ptr = self as *const TxnChunkFooter as *const u8;
        unsafe { std::slice::from_raw_parts(footer_ptr, TXN_FILE_CHUNK_FOOTER_SIZE) }
    }

    pub fn unmarshal(&mut self, data: &[u8]) {
        let footer_ptr = data.as_ptr() as *const TxnChunkFooter;
        *self = unsafe { *footer_ptr };
    }
}

pub(crate) struct TxnChunkHashIndex {
    data: Bytes,
    num_buckets: usize,
    entries_base_off: usize,
}

impl TxnChunkHashIndex {
    pub(crate) fn new(data: Bytes) -> Self {
        let num_buckets = data.chunk().get_u32_le() as usize;
        let entries_base_off = 4 + num_buckets * 4 + 4;
        Self {
            data,
            num_buckets,
            entries_base_off,
        }
    }

    pub(crate) fn get_entry(&self, key_hash: u64) -> Option<KeyAddr> {
        let bucket_idx = (key_hash & (self.num_buckets as u64 - 1)) as usize;
        let bucket_entry_start_off =
            self.entries_base_off + (&self.data[4 + bucket_idx * 4..]).get_u32_le() as usize;
        let bucket_entry_end_off =
            self.entries_base_off + (&self.data[8 + bucket_idx * 4..]).get_u32_le() as usize;
        let mut entry_buf = &self.data[bucket_entry_start_off..bucket_entry_end_off];
        while !entry_buf.is_empty() {
            let entry_key_hash = entry_buf.get_u64_le();
            let block_idx = entry_buf.get_u16_le();
            let key_idx = entry_buf.get_u16_le();
            if entry_key_hash == key_hash {
                return Some(KeyAddr { block_idx, key_idx });
            }
        }
        None
    }

    pub(crate) fn any_exists(&self, key_hashes: &[u64]) -> bool {
        for &key_hash in key_hashes {
            if self.get_entry(key_hash).is_some() {
                return true;
            }
        }
        false
    }
}

#[derive(Default)]
pub struct HashIndexBuilder {
    key_hashes: Vec<(u64, KeyAddr)>,
}

#[derive(Clone, Copy)]
pub struct KeyAddr {
    pub block_idx: u16,
    pub key_idx: u16,
}

impl KeyAddr {
    pub fn new(block_idx: u16, key_idx: u16) -> Self {
        Self { block_idx, key_idx }
    }
}

impl HashIndexBuilder {
    pub(crate) fn add_key_hash(&mut self, key_hash: u64, key_addr: KeyAddr) {
        self.key_hashes.push((key_hash, key_addr))
    }

    pub(crate) fn build(&mut self) -> (Vec<u8>, Vec<u8>) {
        let num_buckets = (self.key_hashes.len().next_power_of_two() / 4).max(2);
        let bucket_mask = (num_buckets - 1) as u64;
        self.key_hashes.sort_by(|&(hash_a, _), &(hash_b, _)| {
            (hash_a & bucket_mask).cmp(&(hash_b & bucket_mask))
        });
        let mut bucket_buf = Vec::with_capacity(4 + num_buckets * 4 + 4);
        bucket_buf.put_u32_le(num_buckets as u32);
        bucket_buf.put_u32_le(0u32);
        let mut entry_buf = Vec::with_capacity(self.key_hashes.len() * 12);
        let mut current_bucket = 0;
        for &(key_hash, key_addr) in &self.key_hashes {
            let bucket_num = (key_hash & bucket_mask) as usize;
            while bucket_num > current_bucket {
                bucket_buf.put_u32_le(entry_buf.len() as u32);
                current_bucket += 1;
            }
            entry_buf.put_u64_le(key_hash);
            entry_buf.put_u16_le(key_addr.block_idx);
            entry_buf.put_u16_le(key_addr.key_idx);
        }
        while current_bucket < num_buckets {
            bucket_buf.put_u32_le(entry_buf.len() as u32);
            current_bucket += 1;
        }
        (bucket_buf, entry_buf)
    }
}

#[cfg(test)]
mod tests {
    use std::{iter::Iterator, ops::Deref, sync::Arc};

    use crate::table::{
        sstable::{get_test_key, get_test_value, InMemFile},
        txn_file::{
            TxnChunk, TxnChunkBuilder, TxnChunkIterator, OP_CHECK_NOT_EXIST, OP_INSERT, OP_PUT,
        },
        InnerKey,
    };

    #[test]
    fn test_txn_chunk() {
        let mut chunk_builder = TxnChunkBuilder::new(10);
        for i in (0..100).step_by(2) {
            let key = get_test_key("batch", i);
            let val = get_test_value(i);
            let op = if i < 50 {
                OP_INSERT
            } else if i < 60 {
                OP_CHECK_NOT_EXIST
            } else {
                OP_PUT
            };
            chunk_builder.add_entry(key.as_bytes(), op, val.as_bytes());
        }
        let mut chunk_data = vec![];
        chunk_builder.finish(&mut chunk_data);
        let chunk_file = Arc::new(InMemFile::new(1, chunk_data.into()));
        let chunk = TxnChunk::new(chunk_file, None).unwrap();
        assert_eq!(chunk.id(), 1);
        assert_eq!(chunk.get_inserts(), 25);
        assert_eq!(chunk.get_check_non_exists(), 5);
        assert!(chunk.size() > 0);
        let mut hashes = vec![1];
        let hash_idx = chunk.load_hash_index().unwrap();
        assert!(!hash_idx.any_exists(&hashes));
        hashes.push(farmhash::fingerprint64(get_test_key("batch", 0).as_bytes()));
        assert!(hash_idx.any_exists(&hashes));

        assert_eq!(
            chunk.index.smallest().deref(),
            get_test_key("batch", 0).as_bytes()
        );
        assert_eq!(
            chunk.index.biggest().deref(),
            get_test_key("batch", 98).as_bytes()
        );
        for i in 0..100 {
            let key = get_test_key("batch", i);
            let inner_key = InnerKey::from_inner_buf(key.as_bytes());
            let iter_opt = chunk.get_value(inner_key);
            if i % 2 == 0 {
                let iter = iter_opt.unwrap();
                let test_val = get_test_value(i);
                assert_eq!(iter.get_value(), test_val.as_bytes());
                if i < 50 {
                    assert_eq!(iter.block_iter.op, OP_INSERT, "{}", i);
                } else if i < 60 {
                    assert_eq!(iter.block_iter.op, OP_CHECK_NOT_EXIST, "{}", i);
                } else {
                    assert_eq!(iter.block_iter.op, OP_PUT, "{}", i);
                }
            } else {
                assert!(iter_opt.is_none());
            }
        }
        let mut iter = TxnChunkIterator::new(chunk.clone(), false);
        iter.rewind();
        let mut i = 0;
        while iter.valid() {
            let key = iter.key();
            let val = iter.get_value();
            assert_eq!(get_test_key("batch", i).as_bytes(), key.deref());
            assert_eq!(get_test_value(i).as_bytes(), val);
            iter.next();
            i += 2;
        }
        assert_eq!(i, 100);
        for i in 0..99 {
            let key = get_test_key("batch", i);
            iter.seek(InnerKey::from_inner_buf(key.as_bytes()));
            let expect_key = get_test_key("batch", i + i % 2);
            assert!(iter.valid(), "{}", i);
            assert_eq!(expect_key.as_bytes(), iter.key().deref(), "{}", i);
        }
        iter.seek(InnerKey::from_inner_buf(b"a"));
        assert!(iter.valid());
        assert_eq!(get_test_key("batch", 0).as_bytes(), iter.key().deref());
        assert_eq!(get_test_value(0).as_bytes(), iter.get_value());

        iter.seek(InnerKey::from_inner_buf(b"c"));
        assert!(!iter.valid());

        iter = TxnChunkIterator::new(chunk, true);
        iter.rewind();
        i = 98;
        while iter.valid() {
            let key = iter.key();
            let val = iter.get_value();
            assert_eq!(get_test_key("batch", i).as_bytes(), key.deref());
            assert_eq!(get_test_value(i).as_bytes(), val);
            iter.next();
            i = i.saturating_sub(2);
        }
        assert_eq!(i, 0);
        for i in 0..100 {
            let key = get_test_key("batch", i);
            iter.seek(InnerKey::from_inner_buf(key.as_bytes()));
            let expect_key = get_test_key("batch", i - i % 2);
            assert!(iter.valid());
            assert_eq!(expect_key.as_bytes(), iter.key().deref());
        }
        iter.seek(InnerKey::from_inner_buf(b"a"));
        assert!(!iter.valid());

        iter.seek(InnerKey::from_inner_buf(b"c"));
        assert!(iter.valid());
        assert_eq!(get_test_key("batch", 98).as_bytes(), iter.key().deref());
        assert_eq!(get_test_value(98).as_bytes(), iter.get_value());
    }
}
