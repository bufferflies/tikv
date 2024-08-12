// Copyright 2023 TiKV Project Authors. Licensed under Apache-2.0.

use std::{cmp, fmt, iter::Iterator as StdIterator, ops::Deref, sync::Arc};

use api_version::{api_v2::KEYSPACE_PREFIX_LEN, ApiV2};
use bytes::{Buf, BufMut, Bytes};
use cloud_encryption::EncryptionKey;
use itertools::Itertools;
use log_wrappers::Value as LogValue;
use moka::sync::SegmentedCache;
use tikv_util::codec::number::NumberEncoder;

use crate::{
    table,
    table::{
        encode_val_to_outer_val_owner,
        file::{File, TtlCache},
        search,
        sstable::{key_diff_idx, BlockCacheKey, EntrySlice},
        ChecksumType, Error, InnerKey, Iterator, NoPrefixKey, Result, Value,
    },
    UserMeta, USER_META_SIZE,
};

const TXN_FILE_PROP_CHECK_NON_EXIST_COUNT: &str = "check_ne";
const TXN_FILE_PROP_INSERT_COUNT: &str = "insert";
const TXN_FILE_PROP_ENCRYPTION_VER: &str = "encryption_ver";

const TXN_FILE_FORMAT: u16 = 1;
const TXN_FILE_MAGIC: u32 = 2785588940;
const U32_SIZE: usize = std::mem::size_of::<u32>();

// Op values reference kvproto::kvrpcpb::Op.
pub const OP_PUT: u8 = 0;
pub const OP_DELETE: u8 = 1;
pub const OP_LOCK: u8 = 2;
pub const OP_INSERT: u8 = 4;
pub const OP_CHECK_NOT_EXIST: u8 = 6;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct TxnFileId {
    pub shard_id: u64,
    pub shard_ver: u64,
    pub start_ts: u64,
}

impl TxnFileId {
    pub fn new(shard_id: u64, shard_ver: u64, start_ts: u64) -> Self {
        Self {
            shard_id,
            shard_ver,
            start_ts,
        }
    }
}

#[derive(Clone)]
pub struct TxnCtx {
    user_meta: Bytes,
    lock_val_prefix: Bytes,
    // The version of data, i.e. data_sequence for lock CF, commit_ts for write CF.
    version: u64,
    lower_bound_buf: Bytes,
    upper_bound_buf: Bytes,
}

impl fmt::Debug for TxnCtx {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut de = f.debug_struct("TxnCtx");
        if !self.user_meta.is_empty() {
            let um = UserMeta::from_slice(&self.user_meta);
            de.field("user_meta", &um);
        }
        de.field("lock_val_prefix", &LogValue::value(&self.lock_val_prefix))
            .field("version", &self.version)
            .field("lower_bound", &self.lower_bound())
            .field("upper_bound", &self.upper_bound())
            .finish()
    }
}

impl TxnCtx {
    pub fn new(
        user_meta: Bytes,
        lock_val_prefix: Bytes,
        version: u64,
        lower_bound: InnerKey<'_>,
        upper_bound: InnerKey<'_>,
    ) -> Self {
        Self {
            user_meta,
            lock_val_prefix,
            version,
            lower_bound_buf: Bytes::copy_from_slice(lower_bound.deref()),
            upper_bound_buf: Bytes::copy_from_slice(upper_bound.deref()),
        }
    }

    pub fn from_txn_file_ref(txn_file_ref: &kvenginepb::TxnFileRef) -> Self {
        let version = if !txn_file_ref.user_meta.is_empty() {
            let um = UserMeta::from_slice(&txn_file_ref.user_meta);
            um.commit_ts
        } else {
            txn_file_ref.version
        };
        Self {
            user_meta: txn_file_ref.user_meta.clone().into(),
            lock_val_prefix: txn_file_ref.lock_val_prefix.clone().into(),
            version,
            lower_bound_buf: txn_file_ref.inner_lower_bound.clone().into(),
            upper_bound_buf: txn_file_ref.inner_upper_bound.clone().into(),
        }
    }

    pub fn to_txn_file_ref(&self, txn_file_ref: &mut kvenginepb::TxnFileRef) {
        txn_file_ref.set_user_meta(self.user_meta.to_vec());
        txn_file_ref.set_lock_val_prefix(self.lock_val_prefix.to_vec());
        txn_file_ref.set_version(self.version);
        txn_file_ref.set_inner_lower_bound(self.lower_bound_buf.to_vec());
        txn_file_ref.set_inner_upper_bound(self.upper_bound_buf.to_vec());
    }

    pub fn is_lock(&self) -> bool {
        !self.lock_val_prefix.is_empty()
    }

    pub fn lower_bound(&self) -> InnerKey<'_> {
        InnerKey::from_inner_buf(&self.lower_bound_buf)
    }

    pub fn upper_bound(&self) -> InnerKey<'_> {
        InnerKey::from_inner_buf(&self.upper_bound_buf)
    }

    pub fn set_lock_val_prefix(&mut self, lock_val_prefix: Bytes) {
        self.lock_val_prefix = lock_val_prefix;
    }
}

#[derive(Clone)]
pub struct TxnFile {
    inner: Arc<TxnFileInner>,
}

impl TxnFile {
    pub fn new(id: TxnFileId, mut chunks: Vec<TxnChunk>, txn_ctx: TxnCtx) -> Result<Self> {
        debug_assert!(
            !txn_ctx.upper_bound().deref().is_empty(),
            "txn_ctx {:?}",
            txn_ctx
        );
        chunks.retain(|chunk| {
            chunk.index.smallest() < txn_ctx.upper_bound()
                && txn_ctx.lower_bound() <= chunk.index.biggest()
        });
        if chunks.is_empty() {
            return Ok(Self {
                inner: Arc::new(TxnFileInner::new(id, chunks, txn_ctx, 0)),
            });
        }
        let mut total_size = 0;
        let mut total_blocks = 0;
        for chunk in &chunks {
            total_size += chunk.size();
            total_blocks += chunk.index.num_blocks;
        }
        let avg_block_size = total_size / total_blocks;
        let first_chunk = chunks.first().unwrap();
        let first_block_idx = first_chunk
            .index
            .seek_block(txn_ctx.lower_bound().deref())
            .saturating_sub(1);
        let last_chunk = chunks.last().unwrap();
        let last_block_idx = last_chunk.index.seek_block(txn_ctx.upper_bound().deref());
        let overlapped_num_blocks =
            total_blocks - first_block_idx - (last_chunk.index.num_blocks - last_block_idx);
        let size = avg_block_size * overlapped_num_blocks;
        Ok(Self {
            inner: Arc::new(TxnFileInner::new(id, chunks, txn_ctx, size)),
        })
    }

    pub fn get_value(&self, key: InnerKey<'_>, outer_val_owner: &mut Vec<u8>) -> (u8, Value) {
        if self.is_empty() || key < self.txn_ctx.lower_bound() || key >= self.txn_ctx.upper_bound()
        {
            return (0, Value::new());
        }
        for (i, chunk) in self.chunks.iter().enumerate() {
            if let Some(chunk_iter) = chunk.get_value(key) {
                let mut file_iter = TxnFileIterator::new(self.clone(), false);
                file_iter.chunk_iter = Some(chunk_iter);
                file_iter.chunk_idx = i;
                file_iter.sync_val();
                let op = file_iter.get_op();
                let value = file_iter.value();
                return (op, encode_val_to_outer_val_owner(value, outer_val_owner));
            }
        }
        (0, Value::new())
    }

    pub fn key_hash_exists(&self, key_hashes: &[u64]) -> bool {
        for chunk in &self.chunks {
            let hash_index = chunk.load_hash_index().unwrap();
            if hash_index.any_exists(key_hashes) {
                return true;
            }
        }
        false
    }

    pub fn expire_ttl_cache(&self) {
        for chunk in &self.chunks {
            chunk.hash_index.expire(0);
        }
    }

    pub fn get_lock_val_prefix(&self) -> &[u8] {
        self.txn_ctx.lock_val_prefix.chunk()
    }

    pub fn lock_val_prefix(&self) -> Bytes {
        self.txn_ctx.lock_val_prefix.clone()
    }

    pub fn get_inserts(&self) -> u32 {
        self.chunks.iter().map(|chunk| chunk.inserts).sum()
    }

    pub fn get_check_not_exists(&self) -> u32 {
        self.chunks.iter().map(|chunk| chunk.check_non_exists).sum()
    }

    pub fn smallest(&self) -> InnerKey<'_> {
        self.chunks.first().unwrap().index.smallest()
    }

    pub fn biggest(&self) -> InnerKey<'_> {
        self.chunks.last().unwrap().index.biggest()
    }

    pub fn merge_chunks(lhs: &Self, rhs: &Self) -> Vec<TxnChunk> {
        let merge_iter =
            lhs.chunks
                .clone()
                .into_iter()
                .merge_join_by(rhs.chunks.clone(), |m, n| {
                    if m.id() == n.id() {
                        cmp::Ordering::Equal
                    } else {
                        m.get_index().smallest().cmp(&n.get_index().smallest())
                    }
                });
        merge_iter
            .into_iter()
            .map(|x| match x {
                itertools::EitherOrBoth::Both(x, _) => x,
                itertools::EitherOrBoth::Left(x) => x,
                itertools::EitherOrBoth::Right(x) => x,
            })
            .collect()
    }

    pub fn validate(&self) -> Result<()> {
        for chunks in self.chunks.windows(2) {
            if chunks[0].index.biggest() >= chunks[1].index.smallest() {
                error!("txn chunks overlap"; "txn_file" => ?self, "chunk0" => ?chunks[0], "chunk1" => ?chunks[1]);
                return Err(Error::Other("txn chunks overlap".to_string()));
            }
        }
        Ok(())
    }

    pub fn to_txn_file_ref(
        &self,
        ranges: Option<&[(InnerKey<'_>, InnerKey<'_>)]>,
    ) -> Option<kvenginepb::TxnFileRef> {
        let chunk_ids = if let Some(ranges) = ranges {
            self.chunk_ids_in_ranges(ranges)
        } else {
            self.chunk_ids()
        };
        if chunk_ids.is_empty() {
            return None;
        }

        let mut txn_file_ref = kvenginepb::TxnFileRef::default();
        txn_file_ref.set_start_ts(self.id.start_ts);
        txn_file_ref.set_shard_ver(self.id.shard_ver);
        self.txn_ctx.to_txn_file_ref(&mut txn_file_ref);
        txn_file_ref.set_chunk_ids(chunk_ids);
        Some(txn_file_ref)
    }

    pub fn has_data_in_range(&self, start: InnerKey<'_>, end: InnerKey<'_>) -> bool {
        let mut iter = TxnFileIterator::new(self.clone(), false);
        iter.seek(start);
        iter.valid() && iter.key() < end
    }

    pub fn has_over_bound_data(&self, start: InnerKey<'_>, end: InnerKey<'_>) -> bool {
        self.lower_bound() < start || self.upper_bound() > end
    }
}

impl fmt::Debug for TxnFile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TxnFile")
            .field("id", &self.id)
            .field("txn_ctx", &self.txn_ctx)
            .field("size", &self.size)
            .field("chunks", &self.chunk_ids())
            .finish()
    }
}

impl Deref for TxnFile {
    type Target = TxnFileInner;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

pub struct TxnFileInner {
    id: TxnFileId,
    txn_ctx: TxnCtx,
    chunks: Vec<TxnChunk>,
    size: usize,
}

impl TxnFileInner {
    fn new(id: TxnFileId, chunks: Vec<TxnChunk>, txn_ctx: TxnCtx, size: usize) -> Self {
        Self {
            id,
            chunks,
            txn_ctx,
            size,
        }
    }

    pub fn version(&self) -> u64 {
        self.txn_ctx.version
    }

    pub fn txn_ctx(&self) -> &TxnCtx {
        &self.txn_ctx
    }

    pub fn shard_ver(&self) -> u64 {
        self.id.shard_ver
    }

    pub fn start_ts(&self) -> u64 {
        self.id.start_ts
    }

    pub fn id(&self) -> TxnFileId {
        self.id
    }

    pub fn chunk_ids(&self) -> Vec<u64> {
        self.chunks.iter().map(|chunk| chunk.id()).collect()
    }

    pub fn chunk_ids_in_ranges(&self, ranges: &[(InnerKey<'_>, InnerKey<'_>)]) -> Vec<u64> {
        self.chunks
            .iter()
            .filter(|chunk| ranges.iter().any(|range| chunk.in_range(*range)))
            .map(|chunk| chunk.id())
            .collect()
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn is_empty(&self) -> bool {
        self.chunks.is_empty()
    }

    pub fn lower_bound(&self) -> InnerKey<'_> {
        self.txn_ctx.lower_bound()
    }

    pub fn upper_bound(&self) -> InnerKey<'_> {
        self.txn_ctx.upper_bound()
    }
}

#[derive(Clone)]
pub struct TxnChunk {
    inner: Arc<TxnChunkInner>,
}

impl fmt::Debug for TxnChunk {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let chunk_index = self.get_index();
        f.debug_struct("TxnChunk")
            .field("id", &self.id())
            .field("size", &self.size())
            .field("inserts", &self.get_inserts())
            .field("check_not_exists", &self.get_check_non_exists())
            .field("num_blocks", &chunk_index.num_blocks())
            .field("smallest", &chunk_index.smallest())
            .field("biggest", &chunk_index.biggest())
            .finish()
    }
}

impl TxnChunk {
    pub fn new(
        file: Arc<dyn File>,
        cache: Option<SegmentedCache<BlockCacheKey, Bytes>>,
        encryption_key: Option<EncryptionKey>,
    ) -> Result<Self> {
        let inner = TxnChunkInner::new(file, cache, encryption_key)?;
        Ok(Self {
            inner: Arc::new(inner),
        })
    }

    fn get_value(&self, key: InnerKey<'_>) -> Option<TxnChunkIterator> {
        let no_prefix_key = NoPrefixKey::from_inner_key(&key)?;
        let key_hash = farmhash::fingerprint64(no_prefix_key.deref());
        let hash_index = self.load_hash_index().unwrap();
        if let Some(key_addr) = hash_index.get_entry(key_hash) {
            let mut iter = TxnChunkIterator::new(self.clone(), false);
            iter.locate_key(key_addr);
            debug_assert!(
                iter.valid(),
                "txn chunk: {:?}, iter: {:?}, key: {:?}, no_prefix_key: {:?}, addr: {:?}",
                self,
                iter,
                key,
                no_prefix_key,
                key_addr,
            );
            if iter.key() != key {
                // There may be hash conflict.
                warn!("hash conflict"; "iter.key" => ?iter.key(), "key" => ?key, "no_prefix_key" => ?no_prefix_key, "file" => self.file.id());
                iter.seek(key);
            }
            if iter.valid() && iter.key() == key {
                return Some(iter);
            }
        }
        None
    }

    pub fn in_range(&self, range: (InnerKey<'_>, InnerKey<'_>)) -> bool {
        self.index.smallest() < range.1 && range.0 <= self.index.biggest()
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
    encryption_key: Option<EncryptionKey>,
    encryption_ver: u32,
}

#[derive(Clone)]
pub struct TxnChunkIndex {
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

    pub fn num_blocks(&self) -> usize {
        self.num_blocks
    }
}

impl TxnChunkInner {
    pub fn new(
        file: Arc<dyn File>,
        cache: Option<SegmentedCache<BlockCacheKey, Bytes>>,
        encryption_key: Option<EncryptionKey>,
    ) -> Result<Self> {
        let footer = Self::load_footer(&file)?;
        let idx_length = footer.hash_index_offset - footer.index_offset;
        let raw_idx_data = file.read(footer.index_offset as u64, idx_length as usize)?;
        let idx_data =
            Self::validate_and_trim_checksum(file.as_ref(), raw_idx_data, footer.checksum_type)?;
        let properties_length =
            file.size() as usize - footer.properties_offset as usize - TXN_FILE_CHUNK_FOOTER_SIZE;
        let raw_properties = file.read(footer.properties_offset as u64, properties_length)?;
        let properties =
            Self::validate_and_trim_checksum(file.as_ref(), raw_properties, footer.checksum_type)?;
        let mut inserts = 0;
        let mut check_non_exists = 0;
        let mut encryption_ver = 0;
        let mut prop_slice = properties.chunk();
        while !prop_slice.is_empty() {
            let (key, mut val, remained) = Self::parse_prop_data(prop_slice);
            if key == TXN_FILE_PROP_INSERT_COUNT.as_bytes() {
                inserts = val.get_u32_le();
            } else if key == TXN_FILE_PROP_CHECK_NON_EXIST_COUNT.as_bytes() {
                check_non_exists = val.get_u32_le();
            } else if key == TXN_FILE_PROP_ENCRYPTION_VER.as_bytes() {
                encryption_ver = val.get_u32_le();
                if encryption_key.is_none() {
                    return Err(Error::NeedEncryptionKey {
                        chunk_id: file.id(),
                        encryption_ver,
                    });
                }
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
            encryption_key,
            encryption_ver,
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

    pub fn get_index(&self) -> &TxnChunkIndex {
        &self.index
    }

    pub fn load_block(&self, pos: usize, decryption_buf: &mut Vec<u8>) -> Result<Bytes> {
        let block_off = self.index.get_block_off(pos);
        let next_block_off = self.index.get_block_off(pos + 1);
        let length = next_block_off - block_off;
        match &self.cache {
            Some(cache) => {
                let cache_key = BlockCacheKey::new(self.file.id(), block_off as u32);
                cache
                    .try_get_with(cache_key, || {
                        crate::metrics::ENGINE_CACHE_MISS.inc_by(1);
                        self.read_block_from_file(block_off as u64, length, decryption_buf)
                    })
                    .map_err(|err| err.as_ref().clone())
            }
            None => self.read_block_from_file(block_off as u64, length, decryption_buf),
        }
    }

    fn read_block_from_file(
        &self,
        offset: u64,
        length: usize,
        decryption_buf: &mut Vec<u8>,
    ) -> Result<Bytes> {
        let raw_block = if let Some(encryption_key) = &self.encryption_key {
            decryption_buf.resize(length, 0);
            self.file.read_at(decryption_buf, offset)?;
            let mut block = Vec::with_capacity(length + encryption_key.encryption_block_size());
            encryption_key.decrypt(
                decryption_buf,
                self.file.id(),
                offset as u32,
                self.encryption_ver,
                &mut block,
            );
            Bytes::from(block)
        } else {
            self.file.read(offset, length)?
        };
        Self::validate_and_trim_checksum(self.file.as_ref(), raw_block, self.footer.checksum_type)
    }

    fn validate_and_trim_checksum(
        file: &dyn File,
        data: Bytes,
        checksum_type: u8,
    ) -> Result<Bytes> {
        if data.len() < 4 {
            return Err(Error::InvalidChecksum(String::from("data is too short")));
        }
        let content_len = data.len() - 4;
        let checksum = (&data[content_len..]).get_u32_le();
        let content = data.slice(..data.len() - 4);
        let got_checksum = ChecksumType::from(checksum_type).checksum(&content);
        if checksum != got_checksum {
            if let Some(path) = file.path().as_ref() {
                let res = std::fs::remove_file(path);
                warn!("remove corrupted txn file"; "file_id" => file.id(), "path" => ?path, "res" => ?res);
            }
            return Err(Error::InvalidChecksum(format!(
                "checksum mismatch expect {} got {} file_id {}",
                checksum,
                got_checksum,
                file.id()
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
        let hash_idx_data = Self::validate_and_trim_checksum(
            self.file.as_ref(),
            raw_hash_idx_data,
            self.footer.checksum_type,
        )
        .unwrap();
        Ok(TxnChunkHashIndex::new(hash_idx_data))
    }
}

pub struct SkipOpTxnFileIterator {
    inner: TxnFileIterator,
    skip_lock: bool,
    skip_check_not_exist: bool,
}

impl SkipOpTxnFileIterator {
    pub fn new(
        txn_file_iter: TxnFileIterator,
        skip_lock: bool,
        skip_check_not_exist: bool,
    ) -> Self {
        Self {
            inner: txn_file_iter,
            skip_lock,
            skip_check_not_exist,
        }
    }

    fn should_skip(&self) -> bool {
        if self.inner.valid() {
            let op = self.inner.get_op();
            if self.skip_lock && op == OP_LOCK {
                return true;
            }
            if self.skip_check_not_exist && op == OP_CHECK_NOT_EXIST {
                return true;
            }
        }
        false
    }
}

impl Iterator for SkipOpTxnFileIterator {
    fn next(&mut self) {
        loop {
            self.inner.next();
            if self.should_skip() {
                continue;
            }
            break;
        }
    }

    fn next_version(&mut self) -> bool {
        false
    }

    fn rewind(&mut self) {
        self.inner.rewind();
        if self.should_skip() {
            self.next();
        }
    }

    fn seek(&mut self, key: InnerKey<'_>) {
        self.inner.seek(key);
        if self.should_skip() {
            self.next();
        }
    }

    fn key(&self) -> InnerKey<'_> {
        self.inner.key()
    }

    fn value(&self) -> Value {
        self.inner.value()
    }

    fn valid(&self) -> bool {
        self.inner.valid()
    }
}

pub struct TxnFileIterator {
    file: TxnFile,
    reverse: bool,
    chunk_iter: Option<TxnChunkIterator>,
    chunk_idx: usize,
    val_buf: Vec<u8>,
    over_bounded: bool,
}

impl fmt::Debug for TxnFileIterator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TxnFileIterator")
            .field("file", &self.file)
            .field("reverse", &self.reverse)
            .field("chunk_iter", &self.chunk_iter)
            .field("chunk_idx", &self.chunk_idx)
            .field("val_buf", &LogValue::value(&self.val_buf))
            .field("over_bounded", &self.over_bounded)
            .finish()
    }
}

impl TxnFileIterator {
    pub fn new(file: TxnFile, reverse: bool) -> Self {
        let val_buf = if file.txn_ctx.is_lock() {
            file.txn_ctx.lock_val_prefix.to_vec()
        } else {
            file.txn_ctx.user_meta.to_vec()
        };
        Self {
            file,
            reverse,
            chunk_iter: None,
            chunk_idx: 0,
            val_buf,
            over_bounded: false,
        }
    }

    pub fn get_op(&self) -> u8 {
        self.chunk_iter.as_ref().unwrap().block_iter.op
    }

    fn seek_chunk(&mut self, chunk_idx: usize, chunk: TxnChunk, key: InnerKey<'_>) {
        self.chunk_idx = chunk_idx;
        let mut chunk_iter = TxnChunkIterator::new(chunk, self.reverse);
        if self.reverse && chunk_idx + 1 == self.file.chunks.len() && key >= self.file.upper_bound()
        {
            chunk_iter.seek(self.file.upper_bound());
            if chunk_iter.valid() && chunk_iter.key() == self.file.upper_bound() {
                chunk_iter.next();
            }
        } else if !self.reverse && chunk_idx == 0 && key < self.file.lower_bound() {
            chunk_iter.seek(self.file.lower_bound());
        } else {
            chunk_iter.seek(key);
        }
        self.chunk_iter = Some(chunk_iter);
        self.sync_val();
    }

    fn sync_over_bound(&mut self) -> bool /* over_bounded */ {
        let chunk_iter = self.chunk_iter.as_ref().unwrap();
        if self.chunk_idx == 0 && chunk_iter.key() < self.file.lower_bound()
            || self.chunk_idx + 1 == self.file.chunks.len()
                && chunk_iter.key() >= self.file.upper_bound()
        {
            self.over_bounded = true;
        }
        self.over_bounded
    }

    fn sync_val(&mut self) {
        if !self.valid() {
            return;
        }
        if self.sync_over_bound() {
            return;
        }

        let chunk_iter = self.chunk_iter.as_ref().unwrap();
        let op = chunk_iter.block_iter.op;
        let val = chunk_iter.get_value();
        if self.file.txn_ctx.is_lock() {
            // op of the lock may be different for each entry, it's at the first byte of the
            // lock prefix we need to sync it.
            self.val_buf[0] = Self::op_to_lock_op(op);
            self.val_buf
                .truncate(self.file.txn_ctx.lock_val_prefix.len());
            self.val_buf.push(txn_types::SHORT_VALUE_PREFIX);
            self.val_buf.encode_var_u64(val.len() as u64).unwrap();
            self.val_buf.extend_from_slice(val);
        } else {
            self.val_buf.truncate(USER_META_SIZE);
            self.val_buf.extend_from_slice(val);
        }
    }

    fn op_to_lock_op(op: u8) -> u8 {
        match op {
            OP_LOCK => txn_types::LockType::Lock.to_u8(),
            OP_DELETE => txn_types::LockType::Delete.to_u8(),
            _ => txn_types::LockType::Put.to_u8(),
        }
    }
}

impl Iterator for TxnFileIterator {
    fn next(&mut self) {
        if !self.valid() {
            return;
        }
        let chunk_iter = self.chunk_iter.as_mut().unwrap();
        chunk_iter.next();
        if !chunk_iter.valid() {
            if self.reverse {
                if self.chunk_idx == 0 {
                    self.chunk_iter = None;
                    return;
                }
                self.chunk_idx -= 1;
            } else {
                if self.chunk_idx == self.file.chunks.len() - 1 {
                    self.chunk_iter = None;
                    return;
                }
                self.chunk_idx += 1;
            }
            let mut new_chunk_iter =
                TxnChunkIterator::new(self.file.chunks[self.chunk_idx].clone(), self.reverse);
            new_chunk_iter.rewind();
            self.chunk_iter = Some(new_chunk_iter);
        }
        self.sync_val();
    }

    fn next_version(&mut self) -> bool {
        false
    }

    fn rewind(&mut self) {
        if self.file.is_empty() {
            return;
        }
        self.over_bounded = false;
        let file = self.file.clone();
        if self.reverse {
            self.seek(file.upper_bound());
        } else {
            self.seek(file.lower_bound());
        }
    }

    fn seek(&mut self, key: InnerKey<'_>) {
        if self.file.is_empty() {
            return;
        }
        self.over_bounded = false;
        if self.reverse {
            for (i, chunk) in self.file.chunks.iter().enumerate().rev() {
                if key >= chunk.index.smallest() {
                    self.seek_chunk(i, chunk.clone(), key);
                    return;
                }
            }
            // key is smaller than first chunk smallest.
        } else {
            for (i, chunk) in self.file.chunks.iter().enumerate() {
                if key <= chunk.index.biggest() {
                    self.seek_chunk(i, chunk.clone(), key);
                    return;
                }
            }
            // key is greater than last chunk biggest.
        }
        self.chunk_iter = None;
    }

    fn key(&self) -> InnerKey<'_> {
        self.chunk_iter.as_ref().unwrap().key()
    }

    fn value(&self) -> Value {
        Value::new_with_meta_version(
            0,
            self.file.version(),
            self.file.txn_ctx.user_meta.len() as u8,
            &self.val_buf,
        )
    }

    fn valid(&self) -> bool {
        !self.file.is_empty()
            && !self.over_bounded
            && self
                .chunk_iter
                .as_ref()
                .map(|it| it.valid())
                .unwrap_or_default()
    }
}

pub struct TxnChunkIterator {
    chunk: TxnChunk,
    block_iter: TxnChunkBlockIterator,
    num_blocks: usize,
    block_pos: usize,
    reverse: bool,
    decryption_buf: Vec<u8>,
}

impl fmt::Debug for TxnChunkIterator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TxnChunkIterator")
            .field("chunk", &self.chunk)
            .field("block_iter", &self.block_iter)
            .field("num_blocks", &self.num_blocks)
            .field("block_pos", &self.block_pos)
            .field("reverse", &self.reverse)
            .finish()
    }
}

impl TxnChunkIterator {
    pub fn new(chunk: TxnChunk, reverse: bool) -> Self {
        let num_blocks = chunk.index.num_blocks;
        Self {
            chunk,
            block_iter: TxnChunkBlockIterator::default(),
            num_blocks,
            block_pos: 0,
            reverse,
            decryption_buf: Vec::new(),
        }
    }

    fn seek_inner(&mut self, key: &[u8]) {
        self.block_pos = self.chunk.index.seek_block(key).saturating_sub(1);
        self.load_block();
        self.block_iter.seek(key);
        if !self.block_iter.valid() && self.block_pos + 1 < self.num_blocks {
            self.block_pos += 1;
            self.load_block();
            self.block_iter.seek(key);
        }
    }

    fn load_block(&mut self) {
        match self
            .chunk
            .load_block(self.block_pos, &mut self.decryption_buf)
        {
            Ok(block) => {
                let block_key = self.chunk.index.block_key(self.block_pos);
                self.block_iter.set_block(block_key, block);
            }
            Err(err) => {
                panic!("load block failed, err {:?}, chunk {:?}", err, self.chunk);
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
            self.load_block();
            self.block_iter.set_idx(0);
        }
    }

    fn prev_inner(&mut self) {
        self.block_iter.prev();
        if self.valid() {
            return;
        }
        if self.block_pos > 0 {
            self.block_pos -= 1;
            self.load_block();
            self.block_iter.set_idx(self.block_iter.num_keys as i32 - 1);
        }
    }

    fn locate_key(&mut self, key_addr: KeyAddr) {
        self.block_pos = key_addr.block_idx as usize;
        self.load_block();
        self.block_iter.set_idx(key_addr.key_idx as i32);
    }

    pub fn next(&mut self) {
        if self.reverse {
            self.prev_inner();
        } else {
            self.next_inner();
        }
    }

    pub fn rewind(&mut self) {
        if self.reverse {
            self.block_pos = self.num_blocks - 1;
            self.load_block();
            self.block_iter.set_idx(self.block_iter.num_keys as i32 - 1);
        } else {
            self.block_pos = 0;
            self.load_block();
            self.block_iter.set_idx(0);
        }
    }

    pub fn seek(&mut self, key: InnerKey<'_>) {
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
        if self.reverse {
            if self.block_iter.valid() && self.key() > key {
                self.prev_inner();
            } else if !self.block_iter.valid() {
                self.rewind();
            }
        }
    }

    pub fn key(&self) -> InnerKey<'_> {
        InnerKey::from_inner_buf(&self.block_iter.key_buf)
    }

    pub fn get_value(&self) -> &[u8] {
        self.block_iter.get_val()
    }

    pub fn valid(&self) -> bool {
        self.block_iter.num_keys > 0 && self.block_iter.valid()
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

impl fmt::Debug for TxnChunkBlockIterator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TxnChunkBlockIterator")
            .field("num_keys", &self.num_keys)
            .field("key_buf", &LogValue::key(&self.key_buf))
            .field("common_prefix_len", &self.common_prefix_len)
            .field("entry_idx", &self.entry_idx)
            .field("op", &self.op)
            .field("val_start", &self.val_start)
            .field("val_end", &self.val_end)
            .field("err", &self.err)
            .finish()
    }
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

    fn valid(&self) -> bool {
        self.err.is_none()
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
    chunk_id: u64,
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
    checksum_type: ChecksumType,
    encryption_key: Option<EncryptionKey>,

    // `entry_key_buf` is used to build txn chunks for keyspaces with inner key offset disabled.
    // Note that hash index is ALWAYS without prefix.
    entry_key_buf: Option<Vec<u8>>,
}

impl TxnChunkBuilder {
    pub fn new(
        chunk_id: u64,
        target_block_entries: usize,
        encryption_key: Option<EncryptionKey>,
        keyspace_id: u32,
        enable_inner_key_off: bool,
    ) -> Self {
        let entry_key_buf =
            (!enable_inner_key_off).then(|| ApiV2::get_txn_keyspace_prefix(keyspace_id));
        Self {
            chunk_id,
            target_block_entries,
            checksum_type: ChecksumType::Crc32,
            encryption_key,
            entry_key_buf,
            ..Default::default()
        }
    }

    pub fn add_entry(&mut self, no_prefix_key: NoPrefixKey<'_>, op: u8, val: &[u8]) {
        let key_hash = farmhash::fingerprint64(no_prefix_key.deref());

        let key = if let Some(entry_key_buf) = self.entry_key_buf.as_mut() {
            entry_key_buf.truncate(KEYSPACE_PREFIX_LEN);
            entry_key_buf.extend_from_slice(no_prefix_key.deref());
            entry_key_buf.as_slice()
        } else {
            no_prefix_key.deref()
        };

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
        if let Some(encryption_key) = &self.encryption_key {
            Self::add_property(
                buf,
                TXN_FILE_PROP_ENCRYPTION_VER.as_bytes(),
                &encryption_key.current_ver.to_le_bytes(),
            )
        }
        let checksum = self.checksum_type.checksum(buf);
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
        let checksum = self.checksum_type.checksum(&self.data_buf[block_offset..]);
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
        let checksum = self.checksum_type.checksum(&self.idx_buf);
        self.idx_buf.put_u32_le(checksum);
    }

    fn build_blocks(&self, output: &mut Vec<u8>) {
        if let Some(encryption_key) = &self.encryption_key {
            for (i, &start) in self.block_offsets.iter().enumerate() {
                let end = if i + 1 < self.block_offsets.len() {
                    self.block_offsets[i + 1]
                } else {
                    self.data_buf.len() as u32
                };
                let data = &self.data_buf[start as usize..end as usize];
                encryption_key.encrypt(data, self.chunk_id, start, output);
            }
        } else {
            output.extend_from_slice(&self.data_buf);
        }
    }

    pub fn finish(&mut self, data_buf: &mut Vec<u8>) {
        if self.block.length() > 0 {
            self.finish_block();
        }
        self.build_index();
        let (bucket_buf, entry_buf) = self.hash_idx_builder.build();
        let mut hash_index_checksum = self.checksum_type.checksum(&bucket_buf);
        hash_index_checksum = self.checksum_type.append(hash_index_checksum, &entry_buf);
        let hash_index_len = bucket_buf.len() + entry_buf.len() + 4;
        let mut props = vec![];
        self.build_properties(&mut props);
        let size = self.data_buf.len()
            + self.idx_buf.len()
            + hash_index_len
            + props.len()
            + TXN_FILE_CHUNK_FOOTER_SIZE;
        data_buf.reserve(size);
        self.build_blocks(data_buf);
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
            checksum_type: self.checksum_type.value(),
            format_version: TXN_FILE_FORMAT,
            magic: TXN_FILE_MAGIC,
        };
        footer.marshal(data_buf);
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
    pub fn marshal(&self, buf: &mut Vec<u8>) {
        buf.put_u32_le(self.index_offset);
        buf.put_u32_le(self.hash_index_offset);
        buf.put_u32_le(self.properties_offset);
        buf.put_u8(self.reserved);
        buf.put_u8(self.checksum_type);
        buf.put_u16_le(self.format_version);
        buf.put_u32_le(self.magic);
    }

    pub fn unmarshal(&mut self, mut data: &[u8]) {
        self.index_offset = data.get_u32_le();
        self.hash_index_offset = data.get_u32_le();
        self.properties_offset = data.get_u32_le();
        self.reserved = data.get_u8();
        self.checksum_type = data.get_u8();
        self.format_version = data.get_u16_le();
        self.magic = data.get_u32_le();
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

#[derive(Clone, Copy, Debug)]
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
    use std::{collections::HashSet, iter::Iterator as StdIterator, sync::Arc};

    use bstr::ByteSlice;
    use cloud_encryption::EncryptionKey;
    use proptest::prelude::*;
    use rstest::rstest;
    use txn_types::LockType;

    use crate::{
        table::{
            file::InMemFile,
            sstable::get_test_value,
            txn_file::{
                TxnChunk, TxnChunkBuilder, TxnChunkIterator, OP_CHECK_NOT_EXIST, OP_INSERT, OP_PUT,
            },
            InnerKey, Iterator, NoPrefixKey, OwnedInnerKey, SkipOpTxnFileIterator, TxnCtx, TxnFile,
            TxnFileId, TxnFileIterator, OP_DELETE, OP_LOCK,
        },
        tests::generate_encryption_key,
        util::test_util::KeyBuilder,
        UserMeta, GLOBAL_SHARD_END_KEY,
    };

    const KEYSPACE_ID: u32 = 42;

    fn new_key_builder(enable_inner_key_off: bool) -> KeyBuilder {
        KeyBuilder::new(KEYSPACE_ID, enable_inner_key_off, "t_batch")
    }

    fn new_key_builder_prefix(enable_inner_key_off: bool, prefix: &str) -> KeyBuilder {
        KeyBuilder::new(KEYSPACE_ID, enable_inner_key_off, &format!("t_{prefix}"))
    }

    #[rstest]
    #[case::basic(None, true)]
    #[case::enc(Some(generate_encryption_key()), true)]
    #[case::disable_key_off(None, false)]
    #[case::enc_disable_key_off(Some(generate_encryption_key()), false)]
    fn test_txn_chunk(#[case] enc_key: Option<EncryptionKey>, #[case] enable_inner_key_off: bool) {
        let kb = new_key_builder(enable_inner_key_off);
        let get_inner_key = |prefix: &str, i: usize| -> OwnedInnerKey {
            let kb = new_key_builder_prefix(enable_inner_key_off, prefix);
            kb.i_to_inner_key(i)
        };

        let chunk = build_txn_chunk(
            0,
            100,
            1,
            |i| {
                if i < 50 {
                    OP_INSERT
                } else if i < 60 {
                    OP_CHECK_NOT_EXIST
                } else {
                    OP_PUT
                }
            },
            enc_key.as_ref(),
            enable_inner_key_off,
        );
        assert_eq!(chunk.id(), 1);
        assert_eq!(chunk.get_inserts(), 25);
        assert_eq!(chunk.get_check_non_exists(), 5);
        assert!(chunk.size() > 0);
        let mut hashes = vec![1];
        let hash_idx = chunk.load_hash_index().unwrap();
        assert!(!hash_idx.any_exists(&hashes));
        hashes.push(farmhash::fingerprint64(&kb.i_to_key(0)));
        assert!(hash_idx.any_exists(&hashes));

        assert_eq!(chunk.index.smallest(), kb.i_to_inner_key(0).as_ref());
        assert_eq!(chunk.index.biggest(), kb.i_to_inner_key(98).as_ref());
        for i in 0..100 {
            let key = kb.i_to_inner_key(i);
            let iter_opt = chunk.get_value(key.as_ref());
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
            assert_eq!(kb.i_to_inner_key(i).as_ref(), key);
            assert_eq!(get_test_value(i).as_bytes(), val);
            iter.next();
            i += 2;
        }
        assert_eq!(i, 100);
        for i in 0..99 {
            let key = kb.i_to_inner_key(i);
            iter.seek(key.as_ref());
            let expect_key = kb.i_to_inner_key(i + i % 2);
            assert!(iter.valid(), "{}", i);
            assert_eq!(expect_key.as_ref(), iter.key(), "{}", i);
        }
        iter.seek(get_inner_key("a", 0).as_ref());
        assert!(iter.valid());
        assert_eq!(kb.i_to_inner_key(0).as_ref(), iter.key());
        assert_eq!(get_test_value(0).as_bytes(), iter.get_value());

        iter.seek(get_inner_key("c", 0).as_ref());
        assert!(!iter.valid());

        iter = TxnChunkIterator::new(chunk, true);
        iter.rewind();
        i = 98;
        while iter.valid() {
            let key = iter.key();
            let val = iter.get_value();
            assert_eq!(kb.i_to_inner_key(i).as_ref(), key);
            assert_eq!(get_test_value(i).as_bytes(), val);
            iter.next();
            i = i.saturating_sub(2);
        }
        assert_eq!(i, 0);
        for i in 0..100 {
            let key = kb.i_to_inner_key(i);
            iter.seek(key.as_ref());
            let expect_key = kb.i_to_inner_key(i - i % 2);
            assert!(iter.valid());
            assert_eq!(expect_key.as_ref(), iter.key());
        }
        iter.seek(get_inner_key("a", 0).as_ref());
        assert!(!iter.valid());

        iter.seek(get_inner_key("c", 0).as_ref());
        assert!(iter.valid());
        assert_eq!(kb.i_to_inner_key(98).as_ref(), iter.key());
        assert_eq!(get_test_value(98).as_bytes(), iter.get_value());
    }

    #[rstest]
    #[case::enable_key_off(true)]
    #[case::disable_key_off(false)]
    fn test_txn_chunk_in_range(#[case] enable_inner_key_off: bool) {
        let kb = new_key_builder(enable_inner_key_off);
        let chunk = build_txn_chunk(10, 100, 1, |_| OP_PUT, None, enable_inner_key_off);

        let cases: Vec<(
            usize, // range start
            usize, // range end
            bool,  // expected in range
        )> = vec![
            (0, 1, false),
            (0, 10, false),
            (0, 50, true),
            (0, 100, true),
            (0, 200, true),
            (10, 50, true),
            (10, 100, true),
            (10, 200, true),
            (50, 100, true),
            (50, 200, true),
            (100, 200, false),
        ];
        for (start, end, expected) in cases {
            let start_key = kb.i_to_inner_key(start);
            let end_key = kb.i_to_inner_key(end);
            assert_eq!(
                chunk.in_range((start_key.as_ref(), end_key.as_ref())),
                expected
            );
        }
    }

    fn build_txn_chunk<F>(
        start: usize,
        end: usize,
        id: u64,
        op_fn: F,
        enc_key: Option<&EncryptionKey>,
        enable_inner_key_off: bool,
    ) -> TxnChunk
    where
        F: Fn(usize) -> u8,
    {
        let mut chunk_builder =
            TxnChunkBuilder::new(id, 10, enc_key.cloned(), KEYSPACE_ID, enable_inner_key_off);
        let kb = new_key_builder(enable_inner_key_off);
        for i in (start..end).step_by(2) {
            let key = kb.i_to_key(i);
            let val = get_test_value(i);
            let op = op_fn(i);
            chunk_builder.add_entry(NoPrefixKey(&key), op, val.as_bytes());
        }
        let mut chunk_data = vec![];
        chunk_builder.finish(&mut chunk_data);
        let chunk_file = Arc::new(InMemFile::new(id, chunk_data.into()));
        TxnChunk::new(chunk_file, None, enc_key.cloned()).unwrap()
    }

    #[rstest]
    #[case::base(None, true)]
    #[case::enc(Some(generate_encryption_key()), true)]
    #[case::disable_key_off(None, false)]
    #[case::enc_disable_key_off(Some(generate_encryption_key()), false)]
    fn test_txn_file(#[case] enc_key: Option<EncryptionKey>, #[case] enable_inner_key_off: bool) {
        let enc_key = enc_key.as_ref();
        let kb = new_key_builder(enable_inner_key_off);
        let op_fn = |i: usize| {
            if i < 100 {
                OP_LOCK
            } else if i < 150 {
                OP_DELETE
            } else if i < 200 {
                OP_CHECK_NOT_EXIST
            } else if i < 250 {
                OP_INSERT
            } else {
                OP_PUT
            }
        };
        let chunk_1 = build_txn_chunk(50, 100, 1, op_fn, enc_key, enable_inner_key_off);
        let chunk_2 = build_txn_chunk(100, 150, 2, op_fn, enc_key, enable_inner_key_off);
        let chunk_3 = build_txn_chunk(150, 200, 3, op_fn, enc_key, enable_inner_key_off);
        let chunk_4 = build_txn_chunk(200, 250, 4, op_fn, enc_key, enable_inner_key_off);
        let chunk_5 = build_txn_chunk(250, 300, 5, op_fn, enc_key, enable_inner_key_off);
        let id = TxnFileId::new(10, 1, 3);
        let lower_bound = InnerKey::from_inner_buf(b"");
        let upper_bound = InnerKey::from_inner_buf(GLOBAL_SHARD_END_KEY);
        let txn_ctx = TxnCtx::new(
            UserMeta::new(3, 5).to_array().to_vec().into(),
            Default::default(),
            3,
            lower_bound,
            upper_bound,
        );
        // test get
        let txn_file = TxnFile::new(
            id,
            vec![chunk_1, chunk_2, chunk_3, chunk_4, chunk_5],
            txn_ctx,
        )
        .unwrap();
        assert_eq!(txn_file.get_inserts(), 25);
        assert_eq!(txn_file.get_check_not_exists(), 25);
        assert!(txn_file.size > 0);

        let mut outer_owner = vec![];
        for i in 50..300 {
            let key = kb.i_to_inner_key(i);
            let (op, got_val) = txn_file.get_value(key.as_ref(), &mut outer_owner);
            if i % 2 == 0 {
                assert!(got_val.is_valid());
                let val = get_test_value(i);
                assert_eq!(got_val.get_value(), val.as_bytes());
                assert_eq!(op_fn(i), op);
            } else {
                assert!(!got_val.is_valid());
            }
        }
        // test iterate forward
        let mut iter = TxnFileIterator::new(txn_file.clone(), false);
        iter.rewind();
        let mut i = 50;
        while iter.valid() {
            let key = kb.i_to_inner_key(i);
            let val = get_test_value(i);
            assert_eq!(iter.key(), key.as_ref());
            assert_eq!(iter.get_op(), op_fn(i));
            let iter_val = iter.value();
            assert_eq!(iter_val.get_value(), val.as_bytes());
            iter.next();
            i += 2;
        }
        assert_eq!(i, 300);
        for i in 50..300 {
            let seek_key = kb.i_to_inner_key(i);
            iter.seek(seek_key.as_ref());
            let expect_key = kb.i_to_inner_key(i + i % 2);
            let expect_val = get_test_value(i + i % 2);
            if i == 299 {
                assert!(!iter.valid());
            } else {
                assert!(iter.valid());
                assert_eq!(iter.key(), expect_key.as_ref());
                assert_eq!(iter.value().get_value(), expect_val.as_bytes());
                let next_i = i + i % 2 + 2;
                if next_i < 300 {
                    // next after seek.
                    let next_key = kb.i_to_inner_key(next_i);
                    iter.next();
                    assert!(iter.valid());
                    assert_eq!(iter.key(), next_key.as_ref());
                }
            }
        }
        // test reverse iterate.
        iter = TxnFileIterator::new(txn_file.clone(), true);
        iter.rewind();
        let mut i = 300;
        while iter.valid() {
            i -= 2;
            let key = kb.i_to_inner_key(i);
            let val = get_test_value(i);
            assert_eq!(iter.key(), key.as_ref());
            assert_eq!(iter.get_op(), op_fn(i));
            let iter_val = iter.value();
            assert_eq!(iter_val.get_value(), val.as_bytes());
            iter.next();
        }
        assert_eq!(i, 50);
        for i in 49..300 {
            let seek_key = kb.i_to_inner_key(i);
            iter.seek(seek_key.as_ref());
            let expect_key = kb.i_to_inner_key(i - i % 2);
            let expect_val = get_test_value(i - i % 2);
            if i == 49 {
                assert!(!iter.valid());
            } else {
                assert!(iter.valid(), "{}", i);
                assert_eq!(iter.key(), expect_key.as_ref());
                assert_eq!(iter.value().get_value(), expect_val.as_bytes());
                let next_i = i - i % 2 - 2;
                if next_i >= 50 {
                    // next after seek.
                    let next_key = kb.i_to_inner_key(next_i);
                    iter.next();
                    assert!(iter.valid());
                    assert_eq!(iter.key(), next_key.as_ref(), "{}", i);
                }
            }
        }
        // test SkipOpTxnFileIterator.
        iter = TxnFileIterator::new(txn_file.clone(), false);
        let mut skip_op_iter = SkipOpTxnFileIterator::new(iter, false, true);
        skip_op_iter.rewind();
        let mut cnt = 0;
        while skip_op_iter.valid() {
            skip_op_iter.next();
            cnt += 1;
        }
        assert_eq!(cnt, 100);
        let check_not_exist_start_key = kb.i_to_inner_key(150);
        skip_op_iter.seek(check_not_exist_start_key.as_ref());
        let expect_key = kb.i_to_inner_key(200);
        assert!(skip_op_iter.valid());
        assert_eq!(skip_op_iter.key(), expect_key.as_ref());

        iter = TxnFileIterator::new(txn_file.clone(), false);
        let mut skip_op_iter = SkipOpTxnFileIterator::new(iter, true, true);
        skip_op_iter.rewind();
        cnt = 0;
        while skip_op_iter.valid() {
            skip_op_iter.next();
            cnt += 1;
        }
        assert_eq!(cnt, 75);

        // test lock cf txn file.
        let mut lock_prefix = txn_types::Lock::new(
            LockType::Put,
            vec![1],
            3.into(),
            12,
            None,
            0.into(),
            100,
            1.into(),
        );
        lock_prefix.is_txn_file = true;
        lock_prefix.to_bytes();
        let txn_ctx = TxnCtx::new(
            vec![].into(),
            lock_prefix.to_bytes().into(),
            1,
            lower_bound,
            upper_bound,
        );
        let txn_file = TxnFile::new(id, txn_file.chunks.clone(), txn_ctx).unwrap();
        let check_lock = |lock: &txn_types::Lock, i: usize| {
            assert_eq!(lock.primary, vec![1]);
            assert_eq!(lock.ttl, 12);
            assert_eq!(lock.ts.into_inner(), 3);
            assert_eq!(lock.for_update_ts.into_inner(), 0);
            assert_eq!(lock.min_commit_ts.into_inner(), 1);
            assert_eq!(lock.txn_size, 100);
            assert!(lock.is_txn_file);
            let expect_val = get_test_value(i);
            let short_val = lock.short_value.as_ref().unwrap();
            assert_eq!(short_val.as_bytes(), expect_val.as_bytes());
            let op = op_fn(i);
            match op {
                OP_PUT | OP_INSERT => {
                    assert_eq!(lock.lock_type, LockType::Put);
                }
                OP_LOCK => {
                    assert_eq!(lock.lock_type, LockType::Lock);
                }
                OP_DELETE => {
                    assert_eq!(lock.lock_type, LockType::Delete)
                }
                _ => {}
            }
        };
        for i in 50..300 {
            let key = kb.i_to_inner_key(i);
            let (_, val) = txn_file.get_value(key.as_ref(), &mut outer_owner);
            if i % 2 == 0 {
                assert!(val.is_valid());
                let lock = txn_types::Lock::parse(val.get_value()).unwrap();
                check_lock(&lock, i);
            } else {
                assert!(!val.is_valid());
            }
        }
        iter = TxnFileIterator::new(txn_file, false);
        iter.rewind();
        let mut i = 50;
        while iter.valid() {
            let key = kb.i_to_inner_key(i);
            assert_eq!(iter.key(), key.as_ref());
            let val = iter.value();
            let lock = txn_types::Lock::parse(val.get_value()).unwrap();
            check_lock(&lock, i);
            i += 2;
            iter.next();
        }
    }

    #[rstest]
    #[case::enable_key_off(true)]
    #[case::disable_key_off(false)]
    fn test_txn_file_in_ranges(#[case] enable_inner_key_off: bool) {
        let kb = new_key_builder(enable_inner_key_off);
        let chunk_1 = build_txn_chunk(50, 100, 1, |_| OP_PUT, None, enable_inner_key_off);
        let chunk_2 = build_txn_chunk(100, 150, 2, |_| OP_PUT, None, enable_inner_key_off);
        let chunk_5 = build_txn_chunk(250, 300, 5, |_| OP_PUT, None, enable_inner_key_off);
        let id = TxnFileId::new(10, 1, 3);
        let lower_bound = InnerKey::from_inner_buf(b"");
        let upper_bound = InnerKey::from_inner_buf(GLOBAL_SHARD_END_KEY);
        let txn_ctx = TxnCtx::new(
            UserMeta::new(3, 5).to_array().to_vec().into(),
            Default::default(),
            3,
            lower_bound,
            upper_bound,
        );
        // test get
        let txn_file = TxnFile::new(id, vec![chunk_1, chunk_2, chunk_5], txn_ctx).unwrap();

        let cases: Vec<(
            Vec<(usize /* start */, usize /* end */)>, // range
            Vec<u64>,                                  // chunks id in ranges
        )> = vec![
            (vec![(0, 100)], vec![1]),
            (vec![(100, 250)], vec![2]),
            (vec![(275, 325)], vec![5]),
            (vec![(75, 275)], vec![1, 2, 5]),
            (vec![(0, 400)], vec![1, 2, 5]),
            (vec![(0, 50), (150, 250), (300, 400)], vec![]),
        ];

        for (ranges, expect_ids) in cases {
            let owned_ranges = ranges
                .into_iter()
                .map(|(start, end)| (kb.i_to_inner_key(start), kb.i_to_inner_key(end)))
                .collect::<Vec<_>>();
            let ranges = owned_ranges
                .iter()
                .map(|(start, end)| (start.as_ref(), end.as_ref()))
                .collect::<Vec<_>>();
            assert_eq!(txn_file.chunk_ids_in_ranges(&ranges), expect_ids);
        }
    }

    #[rstest]
    #[case::enable_key_off(true)]
    #[case::disable_key_off(false)]
    fn test_txn_file_size(#[case] enable_inner_key_off: bool) {
        let kb = new_key_builder(enable_inner_key_off);
        let chunk_1 = build_txn_chunk(50, 100, 1, |_| OP_PUT, None, enable_inner_key_off);
        let chunk_2 = build_txn_chunk(100, 150, 2, |_| OP_PUT, None, enable_inner_key_off);
        let chunk_5 = build_txn_chunk(250, 300, 5, |_| OP_PUT, None, enable_inner_key_off);
        let id = TxnFileId::new(10, 1, 3);
        let cases: Vec<(
            (usize /* start */, usize /* end */), // range
            (usize, usize),                       // txn file size inner_key_off on/off.
        )> = vec![
            ((50, 301), (2394, 2448)),
            ((75, 301), (2128, 2176)),
            ((75, 281), (1862, 1904)),
            ((100, 281), (1345, 1375)),
            ((100, 170), (807, 825)),
            ((100, 149), (807, 825)),
            ((119, 121), (538, 550)),
            ((120, 121), (269, 275)),
            ((160, 170), (0, 0)),
        ];
        let chunks = vec![chunk_1, chunk_2, chunk_5];
        for ((start, end), (txn_file_size_enable, txn_file_size_disable)) in cases {
            let lower_bound = kb.i_to_inner_key(start);
            let upper_bound = kb.i_to_inner_key(end);
            let txn_ctx = TxnCtx::new(
                UserMeta::new(3, 5).to_array().to_vec().into(),
                Default::default(),
                3,
                lower_bound.as_ref(),
                upper_bound.as_ref(),
            );
            let txn_file = TxnFile::new(id, chunks.clone(), txn_ctx).unwrap();
            if enable_inner_key_off {
                assert_eq!(txn_file.size(), txn_file_size_enable);
            } else {
                assert_eq!(txn_file.size(), txn_file_size_disable);
            }
        }
    }

    #[derive(Debug)]
    struct ArbitraryChunks {
        chunks_start: usize,
        chunks_end: usize,
        chunks: Vec<TxnChunk>,
        chunk_ref: TxnFileRefStore,
        enable_inner_key_off: bool,
    }

    prop_compose! {
        // Choose small values to cover corner case easily.
        fn arb_chunks(max_chunk_size: usize, max_chunks_num: usize)
            (
                chunks_size in prop::collection::vec(1..max_chunk_size, 1..max_chunks_num),
                enable_enc in any::<bool>(),
                enable_inner_key_off in any::<bool>(),
            )
            -> ArbitraryChunks
        {
            const CHUNKS_START: usize = 100;
            let mut next_chunk: usize = CHUNKS_START;
            let mut chunks = Vec::with_capacity(chunks_size.len());
            let mut chunk_ref = TxnFileRefStore::new(enable_inner_key_off);
            let enc_key = enable_enc.then(generate_encryption_key);
            for chunk_size in chunks_size {
                let chunk = build_txn_chunk(next_chunk, next_chunk + chunk_size, next_chunk as u64, |_| OP_PUT, enc_key.as_ref(), enable_inner_key_off);
                chunks.push(chunk);
                chunk_ref.put_batch(next_chunk, next_chunk + chunk_size, |_| OP_PUT);
                next_chunk += chunk_size;
            }
            ArbitraryChunks {
                chunks_start: CHUNKS_START,
                chunks_end: next_chunk,
                chunks,
                chunk_ref,
                enable_inner_key_off,
            }
        }
    }
    prop_compose! {
        fn arb_bounded_args(chunk_start: usize, chunk_end: usize)
            (lower_bound in chunk_start-1..=chunk_end+1)
            (
                lower_bound in Just(lower_bound),
                upper_bound in lower_bound+1..=chunk_end+2,
                reverse in any::<bool>(),
                keys in prop::collection::hash_set(chunk_start-2..=chunk_end+3, 1..(chunk_end-chunk_start+6).min(10)),
            )
            -> (usize, usize, bool, HashSet<usize>)
        {
            (lower_bound, upper_bound, reverse, keys)
        }
    }
    prop_compose! {
        fn arb_range_args(chunk_start: usize, chunk_end: usize)
            (lower_bound in chunk_start-1..=chunk_end+1)
            (
                lower_bound in Just(lower_bound),
                upper_bound in lower_bound+1..=chunk_end+2,
            )
            -> (usize, usize)
        {
            (lower_bound, upper_bound)
        }
    }
    proptest! {
        #[test]
        fn test_txn_file_bounded( ArbitraryChunks { chunks_start, chunks_end, chunks, chunk_ref, enable_inner_key_off } in arb_chunks(50, 5)) {
            proptest!(|(
                (lower_bound, upper_bound, reverse, keys) in arb_bounded_args(chunks_start, chunks_end)
            )| {
                let kb = new_key_builder(enable_inner_key_off);
                let lower_bound_key = kb.i_to_inner_key(lower_bound);
                let upper_bound_key = kb.i_to_inner_key(upper_bound);

                let id = TxnFileId::new(10, 1, 3);
                let txn_ctx = TxnCtx::new(
                    UserMeta::new(3, 5).to_array().to_vec().into(),
                    Default::default(),
                    3,
                    lower_bound_key.as_ref(),
                    upper_bound_key.as_ref(),
                );
                let txn_file = TxnFile::new(id, chunks.clone(), txn_ctx).unwrap();

                let mut iter = TxnFileIterator::new(txn_file.clone(), reverse);
                iter.rewind();

                let chunk_ref = chunk_ref.slice(lower_bound_key.as_ref(), upper_bound_key.as_ref());
                for r in chunk_ref.iter(reverse) {
                    prop_assert!(iter.valid());
                    let key = iter.key();
                    let val = iter.value();
                    prop_assert_eq!(key, r.key.as_ref());
                    prop_assert_eq!(iter.get_op(), r.op);
                    prop_assert_eq!(val.get_value(), r.val.as_bytes());
                    iter.next();
                }
                prop_assert!(!iter.valid(), "iter: {:?}, chunk_ref: {:?}", iter, chunk_ref);

                let mut outer_owner = vec![];
                for (i, key) in keys.into_iter().enumerate() {
                    let key = kb.i_to_inner_key(key);

                    {
                        let (op, got_val) = txn_file.get_value(key.as_ref(), &mut outer_owner);
                        if let Some(r) = chunk_ref.get_value(key.as_ref()) {
                            prop_assert!(got_val.is_valid());
                            prop_assert_eq!(got_val.get_value(), r.val.as_bytes());
                            prop_assert_eq!(op, r.op);
                        } else {
                            prop_assert!(!got_val.is_valid());
                        }
                    }

                    {
                        iter.seek(key.as_ref());
                        for r in chunk_ref.seek(key.as_ref(), reverse) {
                            prop_assert!(iter.valid());
                            let key = iter.key();
                            let val = iter.value();
                            prop_assert_eq!(key, r.key.as_ref());
                            prop_assert_eq!(iter.get_op(), r.op);
                            prop_assert_eq!(val.get_value(), r.val.as_bytes());
                            iter.next();
                        }
                        prop_assert!(!iter.valid(), "case {}, key {:?}, iter {:?}", i, key, iter);
                    }
                }
            });
        }

        #[test]
        fn test_txn_file_has_data_in_range( ArbitraryChunks { chunks_start, chunks_end, chunks, chunk_ref, enable_inner_key_off } in arb_chunks(50, 5)) {
            proptest!(|(
                (lower_bound, upper_bound) in arb_range_args(chunks_start, chunks_end)
            )| {
                let kb = new_key_builder(enable_inner_key_off);
                let lower_bound_key = kb.i_to_inner_key(lower_bound);
                let upper_bound_key = kb.i_to_inner_key(upper_bound);

                let id = TxnFileId::new(10, 1, 3);
                let txn_ctx = TxnCtx::new(
                    UserMeta::new(3, 5).to_array().to_vec().into(),
                    Default::default(),
                    3,
                    InnerKey::from_inner_buf(b""),
                    InnerKey::from_inner_buf(GLOBAL_SHARD_END_KEY),
                );
                let txn_file = TxnFile::new(id, chunks.clone(), txn_ctx).unwrap();

                let chunk_ref = chunk_ref.slice(lower_bound_key.as_ref(), upper_bound_key.as_ref());
                prop_assert_eq!(txn_file.has_data_in_range(lower_bound_key.as_ref(), upper_bound_key.as_ref()), !chunk_ref.is_empty());
            });
        }
    }

    #[rstest]
    #[case::enable_key_off(true)]
    #[case::disable_key_off(false)]
    fn test_txn_file_has_over_bound_data(#[case] enable_inner_key_off: bool) {
        let kb = new_key_builder(enable_inner_key_off);
        let chunk = build_txn_chunk(0, 50, 1, |_| OP_PUT, None, enable_inner_key_off);
        let id = TxnFileId::new(10, 1, 3);
        let txn_ctx = TxnCtx::new(
            UserMeta::new(3, 5).to_array().to_vec().into(),
            Default::default(),
            3,
            kb.i_to_inner_key(10).as_ref(),
            kb.i_to_inner_key(20).as_ref(),
        );
        let txn_file = TxnFile::new(id, vec![chunk], txn_ctx).unwrap();

        let cases = [
            (0, 99, false),
            (10, 20, false),
            (11, 20, true),
            (10, 19, true),
            (11, 19, true),
        ];
        for (start, end, expect) in cases {
            assert_eq!(
                txn_file.has_over_bound_data(
                    kb.i_to_inner_key(start).as_ref(),
                    kb.i_to_inner_key(end).as_ref()
                ),
                expect,
                "start: {}, end: {}, expect {}",
                start,
                end,
                expect
            );
        }
    }

    #[derive(Debug)]
    struct TxnFileRefStoreItem {
        key: OwnedInnerKey,
        op: u8,
        val: String,
    }

    #[derive(Debug)]
    struct TxnFileRefStore {
        vec: Vec<TxnFileRefStoreItem>,
        key_builder: KeyBuilder,
    }

    impl TxnFileRefStore {
        pub fn new(enable_inner_key_off: bool) -> Self {
            Self {
                vec: vec![],
                key_builder: new_key_builder(enable_inner_key_off),
            }
        }

        // NOTE: All batches must be in order and no overlapping.
        pub fn put_batch<F>(&mut self, start: usize, end: usize, op_fn: F)
        where
            F: Fn(usize) -> u8,
        {
            self.vec.reserve((end - start) / 2);
            for i in (start..end).step_by(2) {
                let key = self.key_builder.i_to_inner_key(i);
                let val = get_test_value(i);
                let op = op_fn(i);
                self.vec.push(TxnFileRefStoreItem { key, op, val });
            }
        }

        pub fn slice(
            &self,
            lower_bound_key: InnerKey<'_>,
            upper_bound_key: InnerKey<'_>,
        ) -> TxnFileRefStoreSlice<'_> {
            let lower_bound = self
                .vec
                .binary_search_by_key(&lower_bound_key, |x| x.key.as_ref())
                .unwrap_or_else(|x| x);
            let upper_bound = self
                .vec
                .binary_search_by_key(&upper_bound_key, |x| x.key.as_ref())
                .unwrap_or_else(|x| x);
            TxnFileRefStoreSlice {
                slice: &self.vec[lower_bound..upper_bound],
            }
        }
    }

    #[derive(Debug)]
    struct TxnFileRefStoreSlice<'a> {
        slice: &'a [TxnFileRefStoreItem],
    }

    impl<'a> TxnFileRefStoreSlice<'a> {
        pub fn iter(
            &self,
            reverse: bool,
        ) -> Box<dyn StdIterator<Item = &TxnFileRefStoreItem> + '_> {
            if reverse {
                Box::new(self.slice.iter().rev())
            } else {
                Box::new(self.slice.iter())
            }
        }

        fn seek_pos(&self, key: InnerKey<'_>, reverse: bool) -> Option<usize> {
            match self.slice.binary_search_by_key(&key, |x| x.key.as_ref()) {
                Ok(x) => Some(x),
                #[allow(clippy::unnecessary_lazy_evaluations)]
                Err(x) if reverse => (x > 0).then(|| x - 1),
                Err(x) if !reverse => (x < self.slice.len()).then_some(x),
                _ => unreachable!(),
            }
        }

        pub fn seek(
            &self,
            key: InnerKey<'_>,
            reverse: bool,
        ) -> Box<dyn StdIterator<Item = &TxnFileRefStoreItem> + '_> {
            match self.seek_pos(key, reverse) {
                Some(x) if reverse => Box::new(self.slice[..=x].iter().rev()),
                Some(x) if !reverse => Box::new(self.slice[x..].iter()),
                None => Box::new(self.slice[0..0].iter()),
                _ => unreachable!(),
            }
        }

        pub fn get_value(&self, key: InnerKey<'_>) -> Option<&'a TxnFileRefStoreItem> {
            self.seek_pos(key, false)
                .filter(|&x| self.slice[x].key.as_ref() == key)
                .map(|x| &self.slice[x])
        }

        pub fn is_empty(&self) -> bool {
            self.slice.is_empty()
        }
    }
}
