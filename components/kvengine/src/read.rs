// Copyright 2021 TiKV Project Authors. Licensed under Apache-2.0.

use core::panic;
use std::{
    fmt::{Debug, Formatter},
    iter::Iterator as _,
    marker::PhantomData,
    ops::Deref,
    sync::{Arc, Mutex},
};

use bytes::{Buf, BufMut, Bytes, BytesMut};
use cloud_encryption::{EncryptionKey, MasterKey};
use dashmap::DashMap;
use kvenginepb as pb;
use kvenginepb::TxnFileRefs;
use log_wrappers::Value as LogValue;
use pb::SchemaMeta;
use protobuf::Message;
use tikv_util::{
    box_try,
    codec::number::{U32_SIZE, U64_SIZE},
};
use tipb::ColumnInfo;
use txn_types::Lock;

use crate::{
    limiter::RegionLimiter,
    table::{
        blobtable::blobtable::BlobPrefetcher,
        columnar::{
            ColumnarConcatReader, ColumnarMergeReader, ColumnarMvccReader, ColumnarReader,
            ColumnarRowTableReader, ColumnarTableReader, Schema, SchemaBuf, SchemaFile,
            HANDLE_COL_ID,
        },
        memtable::{CfTable, Hint, SkipList, WriteBatch},
        sstable::BlockCache,
        table, BoundedDataSet, DataBound, InnerKey, Iterator as TableIterator,
        SkipOpTxnFileIterator, TxnFile, TxnFileIterator,
    },
    txn_chunk_manager::TxnChunkManager,
    *,
};

/// Mem data format V1: format_ver,skls_data
const MEM_DATA_FORMAT_V1: u32 = 1;
/// Mem data format V2:
/// format_ver,skls_length,skls_data,txn_file_refs_length,txn_file_refs_data
const MEM_DATA_FORMAT_V2: u32 = 2;

pub struct Item<'a> {
    val: table::Value,
    pub path: AccessPath,
    phantom: PhantomData<&'a i32>,

    // Uses to hold the value's memory when necessary, so that the life time of val can be at least
    // long as the life time of Item itself. This is necessary when the caller is not responsible
    // (or impossible) to manage the life time. e.g. During point get, caller does not hold the
    // memory as in scan (held by the iterator).
    owned_val: Option<Vec<u8>>,
    owned_blob: Option<Vec<u8>>,
}

impl std::ops::Deref for Item<'_> {
    type Target = table::Value;

    fn deref(&self) -> &Self::Target {
        &self.val
    }
}

impl Item<'_> {
    fn new() -> Self {
        Self {
            val: table::Value::new(),
            path: AccessPath::default(),
            phantom: Default::default(),
            owned_val: None,
            owned_blob: None,
        }
    }
}

impl Default for Item<'_> {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Default, Debug, Clone, Copy)]
pub struct AccessPath {
    pub mem_table: u8,
    pub l0: u8,
    pub ln: u8,
}

#[derive(Clone)]
pub struct SnapAccess {
    pub core: Arc<SnapAccessCore>,
}

impl SnapAccess {
    pub fn new(shard: &Shard) -> Self {
        let core = Arc::new(SnapAccessCore::new(shard));
        Self { core }
    }

    pub async fn from_change_set(
        tag: String,
        dfs: Arc<dyn dfs::Dfs>,
        change_set: pb::ChangeSet,
        ignore_lock: bool,
        master_key: &MasterKey,
        block_cache: BlockCache,
        schema_files: Option<Arc<DashMap<u64, SchemaFile>>>,
        txn_chunk_manager: TxnChunkManager,
    ) -> Result<Self> {
        let core = Arc::new(
            SnapAccessCore::from_change_set(
                tag,
                dfs,
                change_set,
                vec![],
                ignore_lock,
                master_key,
                block_cache,
                schema_files,
                txn_chunk_manager,
            )
            .await?,
        );
        Ok(Self { core })
    }

    async fn from_change_set_and_memtable_data(
        tag: String,
        dfs: Arc<dyn dfs::Dfs>,
        change_set: pb::ChangeSet,
        mem_tbls: Vec<CfTable>,
        master_key: &MasterKey,
        block_cache: BlockCache,
        schema_files: Option<Arc<DashMap<u64, SchemaFile>>>,
        txn_chunk_manager: TxnChunkManager,
    ) -> Result<Self> {
        let core = Arc::new(
            SnapAccessCore::from_change_set(
                tag,
                dfs,
                change_set,
                mem_tbls,
                true,
                master_key,
                block_cache,
                schema_files,
                txn_chunk_manager,
            )
            .await?,
        );
        Ok(Self { core })
    }

    pub async fn construct_snapshot<'a>(
        tag: String,
        dfs: Arc<dyn dfs::Dfs>,
        mem_table_data: &[u8],
        snapshot: &[u8],
        master_key: &MasterKey,
        block_cache: BlockCache,
        schema_files: Option<Arc<DashMap<u64, SchemaFile>>>, // schema files cache
        txn_chunk_manager: TxnChunkManager,
    ) -> Result<Self> {
        let mut change_set = kvenginepb::ChangeSet::default();
        change_set.merge_from_bytes(snapshot).unwrap();
        if !change_set.has_snapshot() {
            error!("no snapshot in change set: {:?}", change_set);
            return Err(Error::RemoteRead("no snapshot in change set".to_string()));
        }

        let snap = change_set.get_snapshot();
        let inner_key_off = snap.get_inner_key_off() as usize;
        let encryption_key = get_shard_property(ENCRYPTION_KEY, snap.get_properties())
            .map(|v| master_key.decrypt_encryption_key(&v).unwrap());

        let shard_id = change_set.shard_id;
        let shard_ver = change_set.shard_ver;
        let mem_tbls = Self::construct_memtables(
            shard_id,
            shard_ver,
            mem_table_data,
            inner_key_off,
            txn_chunk_manager.clone(),
            encryption_key,
        )
        .await?;
        Self::from_change_set_and_memtable_data(
            tag,
            dfs,
            change_set,
            mem_tbls,
            master_key,
            block_cache,
            schema_files,
            txn_chunk_manager,
        )
        .await
    }

    async fn construct_memtables(
        shard_id: u64,
        shard_ver: u64,
        mut mem_table_data: &[u8],
        inner_key_off: usize,
        txn_chunk_manager: TxnChunkManager,
        encryption_key: Option<EncryptionKey>,
    ) -> Result<Vec<CfTable>> {
        if mem_table_data.is_empty() {
            return Ok(vec![]);
        }

        let format_version = mem_table_data.get_u32_le();
        if format_version == MEM_DATA_FORMAT_V1 {
            let mut mem_tbl = CfTable::new();
            Self::construct_skip_list(mem_table_data, inner_key_off, &mut mem_tbl)?;
            Ok(vec![mem_tbl])
        } else if format_version == MEM_DATA_FORMAT_V2 {
            Self::construct_memtables_format_v2(
                shard_id,
                shard_ver,
                mem_table_data,
                inner_key_off,
                txn_chunk_manager,
                encryption_key,
            )
            .await
        } else {
            Err(Error::RemoteRead(format!(
                "unsupported mem data format {}",
                format_version
            )))
        }
    }

    fn construct_skip_list(
        skl_data: &[u8],
        inner_key_off: usize,
        mem_tbl: &mut CfTable,
    ) -> Result<()> {
        let mut wb = WriteBatch::new();
        let rows: Vec<table::Row> = bincode::deserialize(skl_data).map_err(|e| {
            Error::RemoteRead(format!("failed to deserialize mem table data: {}", e))
        })?;
        for row in rows {
            let key = InnerKey::from_outer_key(&row.key, inner_key_off);
            wb.put(
                key,
                0,
                &row.user_meta.to_array(),
                row.user_meta.commit_ts,
                &row.value,
            );
        }
        mem_tbl.get_cf(WRITE_CF).put_batch(&mut wb, None, WRITE_CF);
        Ok(())
    }

    async fn construct_memtables_format_v2(
        shard_id: u64,
        shard_ver: u64,
        mut mem_data: &[u8],
        inner_key_off: usize,
        txn_chunk_manager: TxnChunkManager,
        encryption_key: Option<EncryptionKey>,
    ) -> Result<Vec<CfTable>> {
        let mut mem_tbl = CfTable::new();

        // Skiplists:
        let mem_size = mem_data.get_u64_le() as usize;
        let skl_data = &mem_data[..mem_size];
        mem_data.advance(mem_size);
        Self::construct_skip_list(skl_data, inner_key_off, &mut mem_tbl)?;

        // Txn file refs:
        let msg_len = mem_data.get_u32_le() as usize;
        let msg_data = &mem_data[..msg_len];
        mem_data.advance(msg_len);

        let mut txn_file_refs = TxnFileRefs::default();
        box_try!(txn_file_refs.merge_from_bytes(msg_data));

        let worker_pool = txn_chunk_manager.worker_pool().clone();
        let txn_files = worker_pool
            .spawn_blocking(move || {
                txn_chunk_manager.load_txn_files_from_refs(
                    shard_id,
                    shard_ver,
                    txn_file_refs.get_txn_file_refs(),
                    encryption_key,
                )
            })
            .await
            .unwrap()?;
        mem_tbl = mem_tbl.add_write_cf_txn_files(&txn_files);

        debug_assert!(
            mem_data.is_empty(),
            "mem_data: {:?}",
            LogValue::value(mem_data)
        );
        Ok(vec![mem_tbl])
    }
}

impl Deref for SnapAccess {
    type Target = SnapAccessCore;

    fn deref(&self) -> &Self::Target {
        &self.core
    }
}

impl Debug for SnapAccess {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "snap access {}, seq: {}", self.tag, self.write_sequence,)
    }
}

impl BoundedDataSet for SnapAccess {
    fn data_bound(&self) -> DataBound<'_> {
        self.data.range.data_bound()
    }
}

pub struct SnapAccessCore {
    tag: ShardTag,
    managed_ts: u64,
    base_version: u64,
    meta_seq: u64,
    write_sequence: u64,
    columnar_snap_version: u64,
    data: ShardData,
    get_hint: Mutex<Hint>,
    blob_table_prefetch_size: usize,
    deleting_prefixes: Arc<DeletePrefixes>,
    encryption_key: Option<EncryptionKey>,
}

impl SnapAccessCore {
    pub fn new(shard: &Shard) -> Self {
        let base_version = shard.get_base_version();
        let meta_seq = shard.get_meta_sequence();
        let write_sequence = shard.get_write_sequence();
        let columnar_snap_version = shard.get_columnar_snap_version();
        let data = shard.get_data();
        Self {
            tag: shard.tag(),
            write_sequence,
            columnar_snap_version,
            meta_seq,
            base_version,
            managed_ts: 0,
            data,
            get_hint: Mutex::new(Hint::new()),
            blob_table_prefetch_size: shard.opt.blob_prefetch_size,
            deleting_prefixes: shard.get_del_prefixes(),
            encryption_key: shard.encryption_key.clone(),
        }
    }

    pub async fn from_change_set(
        tag: String,
        dfs: Arc<dyn dfs::Dfs>,
        change_set: pb::ChangeSet,
        mem_tbls: Vec<CfTable>,
        ignore_lock: bool,
        master_key: &MasterKey,
        block_cache: BlockCache,
        schema_files: Option<Arc<DashMap<u64, SchemaFile>>>,
        txn_chunk_manager: TxnChunkManager,
    ) -> Result<Self> {
        let shard = Shard::from_change_set(
            tag,
            dfs,
            change_set,
            mem_tbls,
            ignore_lock,
            master_key,
            block_cache,
            schema_files,
            txn_chunk_manager,
        )
        .await?;
        Ok(Self::new(&shard))
    }

    pub fn new_iterator_skip_blob(
        &self,
        cf: usize,
        reversed: bool,
        all_versions: bool,
        read_ts: Option<u64>,
        fill_cache: bool,
    ) -> Iterator {
        let read_ts = if let Some(ts) = read_ts {
            ts
        } else if CF_MANAGED[cf] && self.managed_ts != 0 {
            self.managed_ts
        } else {
            u64::MAX
        };
        let data = self.data.clone();
        let mut key = BytesMut::new();
        key.extend_from_slice(data.prefix());
        Iterator {
            all_versions,
            reversed,
            read_ts,
            key,
            val: table::Value::new(),
            inner: self.new_table_iterator(cf, reversed, fill_cache, None),
            blob_prefetcher: None,
            data,
            range: None,
        }
    }

    pub fn new_iterator(
        &self,
        cf: usize,
        reversed: bool,
        all_versions: bool,
        read_ts: Option<u64>,
        fill_cache: bool,
    ) -> Iterator {
        let blob_prefetcher = Some(BlobPrefetcher::new(
            self.data.blob_tbl_map.clone(),
            self.blob_table_prefetch_size,
            self.encryption_key.clone(),
        ));
        let data = self.data.clone();
        let mut key = BytesMut::new();
        key.extend_from_slice(data.prefix());
        Iterator {
            all_versions,
            reversed,
            read_ts: self.get_read_ts(cf, read_ts),
            key,
            val: table::Value::new(),
            inner: self.new_table_iterator(cf, reversed, fill_cache, None),
            blob_prefetcher,
            data,
            range: None,
        }
    }

    pub fn new_memtable_iterator(
        &self,
        cf: usize,
        reversed: bool,
        all_versions: bool,
        read_ts: Option<u64>,
    ) -> Iterator {
        let data = self.data.clone();
        let mut key = BytesMut::new();
        key.extend_from_slice(data.prefix());
        Iterator {
            all_versions,
            reversed,
            read_ts: self.get_read_ts(cf, read_ts),
            key,
            val: table::Value::new(),
            inner: self.new_mem_table_iterator(cf, reversed),
            blob_prefetcher: None,
            data,
            range: None,
        }
    }

    fn get_read_ts(&self, cf: usize, read_ts: Option<u64>) -> u64 {
        if let Some(ts) = read_ts {
            ts
        } else if CF_MANAGED[cf] && self.managed_ts != 0 {
            self.managed_ts
        } else if !CF_MANAGED[cf] {
            self.get_mem_table_version()
        } else {
            u64::MAX
        }
    }

    fn fetch_blob(&self, key: InnerKey<'_>, val: &table::Value) -> Vec<u8> {
        assert!(val.is_blob_ref());
        let blob_ref = val.get_blob_ref();
        let blob_table = self
            .data
            .blob_tbl_map
            .get(&blob_ref.fid)
            .unwrap_or_else(|| {
                panic!(
                    "[{}] blob table not found {:?}, blob table id: {}",
                    self.tag, key, blob_ref.fid
                )
            });
        // TODO: avoid reallocation
        let mut decryption_buf = vec![];
        blob_table
            .get(&blob_ref, &mut decryption_buf, self.encryption_key.clone())
            .unwrap_or_else(|e| panic!("[{}] blob table get failed {:?} {:?}", self.tag, key, e))
    }

    /// get an Item by key. Caller need to call is_some() before get_value.
    /// We don't return Option because we may need AccessPath even if the item
    /// is none.
    pub fn get(&self, cf: usize, key: &[u8], version: u64) -> Item<'_> {
        let mut version = version;
        if version == 0 {
            version = u64::MAX;
        }

        debug_assert_eq!(self.data.prefix(), &key[..self.data.inner_key_off]);
        let inner_key = InnerKey::from_outer_key(key, self.data.inner_key_off);
        let mut item = Item::new();
        item.owned_val = Some(vec![]);
        item.val = self.get_value(
            cf,
            inner_key,
            version,
            &mut item.path,
            item.owned_val.as_mut().unwrap(),
            &self.data.lock_txn_files,
        );
        if item.val.is_blob_ref() {
            item.owned_blob = Some(self.fetch_blob(inner_key, &item.val));
            item.val.fill_in_blob(item.owned_blob.as_ref().unwrap());
        }
        item
    }

    pub fn get_non_txn_file_lock(&self, key: &[u8]) -> Item<'_> {
        let inner_key = InnerKey::from_outer_key(key, self.data.inner_key_off);
        let mut item = Item::new();
        item.owned_val = Some(vec![]);
        item.val = self.get_value(
            LOCK_CF,
            inner_key,
            u64::MAX,
            &mut item.path,
            item.owned_val.as_mut().unwrap(),
            &[],
        );
        item
    }

    fn get_value(
        &self,
        cf: usize,
        inner_key: InnerKey<'_>,
        version: u64,
        path: &mut AccessPath,
        out_val_owner: &mut Vec<u8>,
        with_lock_txn_files: &[TxnFile],
    ) -> table::Value {
        if cf == LOCK_CF {
            for txn_file in with_lock_txn_files {
                if txn_file.version() > version {
                    continue;
                }
                let (_, val) = txn_file.get_value(inner_key, out_val_owner);
                if val.is_valid() {
                    return val;
                }
            }
        }
        for i in 0..self.data.mem_tbls.len() {
            let tbl = self.data.mem_tbls.as_slice()[i].get_cf(cf);
            let v = if i == 0 && cf == 0 {
                // only use hint for the first mem-table and cf 0.
                let mut hint = self.get_hint.lock().unwrap();
                tbl.get_with_hint(inner_key, version, &mut hint, out_val_owner)
            } else {
                tbl.get(inner_key, version, out_val_owner)
            };
            path.mem_table = path.mem_table.saturating_add(1);
            if v.is_valid() {
                return v;
            }
        }
        let key_hash = farmhash::fingerprint64(inner_key.deref());
        for l0 in &self.data.l0_tbls {
            if let Some(tbl) = &l0.get_cf(cf) {
                if inner_key < tbl.smallest() || tbl.biggest() < inner_key {
                    continue;
                }
                let v = tbl.get(inner_key, version, key_hash, out_val_owner, 0);
                path.l0 = path.l0.saturating_add(1);
                if v.is_valid() {
                    return v;
                }
            }
        }
        let scf = self.data.get_cf(cf);
        for lh in &scf.levels {
            let v = lh.get(inner_key, version, key_hash, out_val_owner);
            path.ln += 1;
            if v.is_valid() {
                return v;
            }
        }
        table::Value::new()
    }

    pub fn multi_get(&self, cf: usize, keys: &[Vec<u8>], version: u64) -> Vec<Item<'_>> {
        let mut items = Vec::with_capacity(keys.len());
        for key in keys {
            let item = self.get(cf, key, version);
            items.push(item);
        }
        items
    }

    pub fn set_managed_ts(&mut self, managed_ts: u64) {
        self.managed_ts = managed_ts;
    }

    fn new_table_iterator(
        &self,
        cf: usize,
        reversed: bool,
        fill_cache: bool,
        skip_txn_file_with_start_ts: Option<u64>,
    ) -> Box<dyn table::Iterator> {
        let mut iters: Vec<Box<dyn table::Iterator>> = Vec::new();
        if cf == LOCK_CF && !self.data.lock_txn_files.is_empty() {
            for txn_file in &self.data.lock_txn_files {
                if skip_txn_file_with_start_ts
                    .map_or(false, |start_ts| txn_file.start_ts() == start_ts)
                {
                    continue;
                }
                let txn_file_iter = TxnFileIterator::new(txn_file.clone(), reversed);
                let skip_op_iter = SkipOpTxnFileIterator::new(txn_file_iter, false, true);
                iters.push(Box::new(skip_op_iter))
            }
        }
        for mem_tbl in &self.data.mem_tbls {
            iters.push(mem_tbl.get_cf(cf).new_iterator(reversed));
        }
        for l0 in &self.data.l0_tbls {
            if let Some(tbl) = &l0.get_cf(cf) {
                iters.push(tbl.new_iterator(reversed, fill_cache));
            }
        }
        let scf = self.data.get_cf(cf);
        for lh in scf.levels.as_slice() {
            if lh.tables.len() == 0 {
                continue;
            }
            if lh.tables.len() == 1 {
                iters.push(lh.tables[0].new_iterator(reversed, fill_cache));
                continue;
            }
            iters.push(Box::new(ConcatIterator::new(
                lh.clone(),
                reversed,
                fill_cache,
            )));
        }
        table::new_merge_iterator(iters, reversed)
    }

    pub fn new_mem_table_iterator(&self, cf: usize, reversed: bool) -> Box<dyn table::Iterator> {
        let mut iters: Vec<Box<dyn table::Iterator>> = Vec::new();
        for mem_tbl in &self.data.mem_tbls {
            iters.push(mem_tbl.get_cf(cf).new_iterator(reversed));
        }
        table::new_merge_iterator(iters, reversed)
    }

    pub fn new_delta_write_iterator(&self, since_ts: u64) -> Box<dyn table::Iterator> {
        let mut iters: Vec<Box<dyn table::Iterator>> = Vec::new();
        for mem_tbl in &self.data.mem_tbls {
            if mem_tbl.data_max_ts() > since_ts {
                iters.push(mem_tbl.get_cf(WRITE_CF).new_delta_write_iterator(since_ts));
            }
        }
        for l0 in &self.data.l0_tbls {
            if let Some(tbl) = &l0.get_cf(WRITE_CF) {
                if tbl.max_ts > since_ts {
                    iters.push(tbl.new_iterator(false, true));
                }
            }
        }
        let scf = self.data.get_cf(WRITE_CF);
        for lh in scf.levels.as_slice() {
            if lh.tables.len() == 0 || lh.max_ts < since_ts {
                continue;
            }
            if lh.tables.len() == 1 {
                iters.push(lh.tables[0].new_iterator(false, true));
                continue;
            }
            iters.push(Box::new(ConcatIterator::new(lh.clone(), false, true)));
        }
        table::new_merge_iterator(iters, false)
    }

    pub fn get_write_sequence(&self) -> u64 {
        self.write_sequence
    }

    pub fn get_mem_table_version(&self) -> u64 {
        self.base_version + self.write_sequence
    }

    pub fn get_start_key(&self) -> &[u8] {
        self.data.outer_start.chunk()
    }

    pub fn get_end_key(&self) -> &[u8] {
        self.data.outer_end.chunk()
    }

    pub fn key_is_in_range(&self, key: &[u8]) -> bool {
        self.get_start_key() <= key && key < self.get_end_key()
    }

    pub fn get_inner_start(&self) -> InnerKey<'_> {
        self.data.inner_start()
    }

    pub fn get_inner_end(&self) -> InnerKey<'_> {
        self.data.inner_end()
    }

    pub fn clone_end_key(&self) -> Bytes {
        self.data.outer_end.clone()
    }

    pub fn get_inner_key_offset(&self) -> usize {
        self.data.range.inner_key_off
    }

    pub fn get_tag(&self) -> ShardTag {
        self.tag
    }

    pub fn get_id(&self) -> u64 {
        self.tag.id_ver.id
    }

    pub fn get_version(&self) -> u64 {
        self.tag.id_ver.ver
    }

    pub fn get_meta_seq(&self) -> u64 {
        self.meta_seq
    }

    pub(crate) fn contains_in_older_table(&self, key: InnerKey<'_>, cf: usize) -> bool {
        let key_hash = farmhash::fingerprint64(key.deref());
        let mut outer_val_owner = vec![];
        for tbl in &self.data.mem_tbls[1..] {
            let val = tbl.get_cf(cf).get(key, u64::MAX, &mut outer_val_owner);
            if val.is_valid() {
                return !val.is_deleted();
            }
        }
        for l0 in &self.data.l0_tbls {
            let l0_cf = l0.get_cf(cf);
            if l0_cf.is_none() {
                continue;
            }
            let l0_cf = l0_cf.as_ref().unwrap();
            let mut owned_val = vec![];
            let val = l0_cf.get(key, u64::MAX, key_hash, &mut owned_val, 0);
            if val.is_valid() {
                return !val.is_deleted();
            }
        }
        for l in self.data.get_cf(cf).levels.as_slice() {
            if let Some(tbl) = l.get_table(key) {
                let mut owned_val = vec![];
                let val = tbl.get(key, u64::MAX, key_hash, &mut owned_val, l.level);
                if val.is_valid() {
                    return !val.is_deleted();
                }
            }
        }
        false
    }

    /// NOTE: `ChangeSet.snapshot.data_sequence` is not set, as it's not able to
    /// get the accurate data sequence of ShardMeta from Shard.
    fn to_change_set(&self, outer_ranges: &[(Bytes, Bytes)], ignore_locks: bool) -> pb::ChangeSet {
        let mut cs = new_change_set(self.get_tag().id_ver.id, self.get_tag().id_ver.ver);
        cs.set_sequence(self.meta_seq);

        let snap = cs.mut_snapshot();
        let mut properties = pb::Properties::new();
        properties.shard_id = self.get_tag().id_ver.id;
        if let Some(encryption_key) = &self.encryption_key {
            properties.mut_keys().push(ENCRYPTION_KEY.to_string());
            properties.mut_values().push(encryption_key.export());
        }

        snap.set_outer_start(self.get_start_key().to_vec());
        snap.set_outer_end(self.get_end_key().to_vec());
        snap.set_inner_key_off(self.data.range.inner_key_off as u32);
        snap.set_properties(properties);
        let mut count = 0;
        let mut overlapped_count = 0;
        for v in &self.data.l0_tbls {
            count += 1;
            if ignore_locks {
                if let Some(cf) = v.get_cf(WRITE_CF) {
                    if cf.size() == 0 {
                        continue;
                    }
                } else {
                    continue;
                }
            }
            let mut overlap = false;
            for (outer_start, outer_end) in outer_ranges {
                let inner_start =
                    InnerKey::from_outer_key(outer_start, self.data.range.inner_key_off);
                let inner_end =
                    InnerKey::from_outer_end_key(outer_end, self.data.range.inner_key_off);
                let bound = DataBound::new(inner_start, inner_end, false);
                if v.has_data_in_bound(bound) {
                    overlap = true;
                    break;
                }
            }
            if !overlap {
                continue;
            }
            overlapped_count += 1;
            let mut l0 = pb::L0Create::new();
            l0.set_id(v.id());
            l0.set_smallest(v.smallest().to_vec());
            l0.set_biggest(v.biggest().to_vec());
            snap.mut_l0_creates().push(l0);
        }
        for (k, v) in self.data.blob_tbl_map.iter() {
            assert_eq!(k, &v.id());
            count += 1;
            // FIXME: Overlap check
            let mut blob = pb::BlobCreate::new();
            blob.set_id(v.id());
            blob.set_smallest(v.smallest_key().to_vec());
            blob.set_biggest(v.biggest_key().to_vec());
            snap.mut_blob_creates().push(blob);
        }
        self.data.for_each_level(|cf, lh| {
            if cf == LOCK_CF {
                return false;
            }
            for v in lh.tables.iter() {
                count += 1;
                if ignore_locks && cf == WRITE_CF && v.size() == 0 {
                    continue;
                }
                let mut overlap = false;
                for (outer_start, outer_end) in outer_ranges {
                    let inner_start =
                        InnerKey::from_outer_key(outer_start, self.data.range.inner_key_off);
                    let inner_end =
                        InnerKey::from_outer_end_key(outer_end, self.data.range.inner_key_off);
                    let data_bound = DataBound::new(inner_start, inner_end, false);
                    if v.has_overlap(data_bound) {
                        overlap = true;
                        break;
                    }
                }
                if !overlap {
                    continue;
                }
                overlapped_count += 1;
                let mut tbl = pb::TableCreate::new();
                tbl.set_id(v.id());
                tbl.set_cf(cf as i32);
                tbl.set_level(lh.level as u32);
                tbl.set_smallest(v.smallest().to_vec());
                tbl.set_biggest(v.biggest().to_vec());
                snap.mut_table_creates().push(tbl);
            }
            false
        });
        self.data.for_each_columnar_level(|cl| {
            for col_file in cl.files.iter() {
                count += 1;
                // check overlap
                let mut overlap = false;
                for (outer_start, outer_end) in outer_ranges {
                    let inner_start =
                        InnerKey::from_outer_key(outer_start, self.data.range.inner_key_off);
                    let inner_end =
                        InnerKey::from_outer_end_key(outer_end, self.data.range.inner_key_off);
                    if col_file.has_data_in_range(inner_start, inner_end) {
                        overlap = true;
                        break;
                    }
                }
                if !overlap {
                    continue;
                }
                overlapped_count += 1;
                let mut tbl = pb::TableCreate::new();
                tbl.set_id(col_file.id());
                tbl.set_cf(WRITE_CF as i32);
                tbl.set_level(cl.level as u32);
                tbl.set_smallest(col_file.get_smallest().to_vec());
                tbl.set_biggest(col_file.get_biggest().to_vec());
                tbl.set_columnar_tables(col_file.table_count() as u32);
                snap.mut_columnar_creates().push(tbl);
            }
            false
        });
        if let Some(schema_file) = &self.data.schema_file {
            let mut schema_meta = SchemaMeta::default();
            schema_meta.set_file_id(schema_file.get_file_id());
            schema_meta.set_keyspace_id(self.get_keyspace_id());
            schema_meta.set_version(schema_file.get_version());
            snap.set_schema_meta(schema_meta);
        }
        snap.set_columnar_snap_version(self.columnar_snap_version);

        info!(
            "convert snap access to change set for {}, total files {}, overlapped files {}",
            self.get_tag(),
            count,
            overlapped_count,
        );
        cs
    }

    pub fn marshal(
        &self,
        ranges: &[(Bytes, Bytes)],
        ignore_locks: bool,
        cache_key: bool,
    ) -> (String, Vec<u8>) {
        let cs = self.to_change_set(ranges, ignore_locks);
        let key = if cache_key {
            let mut ranges_bytes = Vec::new();
            for (start, end) in ranges {
                ranges_bytes.append(start.to_vec().as_mut());
                ranges_bytes.append(end.to_vec().as_mut());
            }
            let ranges_key = hex::encode(ranges_bytes);
            format!(
                "{}:{}:{}:{}",
                cs.shard_id, cs.shard_ver, cs.sequence, ranges_key,
            )
        } else {
            "".to_string()
        };
        (key, cs.write_to_bytes().unwrap())
    }

    pub fn build_mem_data(&self, outer_ranges: &[(Bytes, Bytes)], start_ts: u64) -> Vec<u8> {
        let data_bounds: Vec<DataBound<'_>> = outer_ranges
            .iter()
            .map(|range| {
                DataBound::new(
                    InnerKey::from_outer_key(&range.0, self.data.inner_key_off),
                    InnerKey::from_outer_key(&range.1, self.data.inner_key_off),
                    false,
                )
            })
            .collect();

        let mut mem_data = Vec::with_capacity(U32_SIZE /* format */);
        let (skls, txn_file_refs) = self.get_mem_tables_group_by_type(WRITE_CF, &data_bounds);

        if txn_file_refs.get_txn_file_refs().is_empty() {
            // For backward compatible.
            // TODO: remove after all tikv-workers are upgraded.
            mem_data.put_u32_le(MEM_DATA_FORMAT_V1);
            self.build_skl_data(outer_ranges, start_ts, skls, &mut mem_data, false);
            return mem_data;
        }

        mem_data.put_u32_le(MEM_DATA_FORMAT_V2);
        self.build_skl_data(outer_ranges, start_ts, skls, &mut mem_data, true);
        self.build_txn_file_data(txn_file_refs, &mut mem_data);
        mem_data
    }

    fn build_skl_data(
        &self,
        ranges: &[(Bytes, Bytes)],
        start_ts: u64,
        skls: Vec<SkipList>,
        mem_data: &mut Vec<u8>,
        encode_meta: bool,
    ) {
        let skl_iters = skls
            .iter()
            .map(|skl| Box::new(skl.new_iterator(false)) as _)
            .collect();
        let merge_iter = table::new_merge_iterator(skl_iters, false);

        let data = self.data.clone();
        let mut key = BytesMut::new();
        key.extend_from_slice(data.prefix());
        let mut mem_iterator = Iterator {
            all_versions: false,
            reversed: false,
            read_ts: start_ts,
            key,
            val: table::Value::new(),
            inner: merge_iter,
            blob_prefetcher: None,
            data,
            range: None,
        };

        let mut rows = vec![];
        for (range_start, range_end) in ranges {
            mem_iterator.seek(range_start.chunk());
            while mem_iterator.valid() {
                let key = mem_iterator.key();
                if key >= range_end.chunk() {
                    break;
                }
                rows.push(table::Row {
                    key: key.to_vec(),
                    user_meta: UserMeta::from_slice(mem_iterator.user_meta()),
                    value: mem_iterator.val().to_vec(),
                });
                mem_iterator.next();
            }
        }
        let mem_size = bincode::serialized_size(&rows)
            .map_err(|e| Error::Other(e))
            .unwrap();
        if encode_meta {
            mem_data.reserve(U64_SIZE /* mem_size */ + mem_size as usize);
            mem_data.put_u64_le(mem_size);
        } else {
            mem_data.reserve(mem_size as usize);
        }
        bincode::serialize_into(mem_data, &rows).unwrap();
    }

    fn build_txn_file_data(&self, txn_file_refs: TxnFileRefs, mem_data: &mut Vec<u8>) {
        let msg_len = txn_file_refs.compute_size();
        mem_data.reserve(U32_SIZE /* msg_len */ + msg_len as usize);
        mem_data.put_u32_le(msg_len);
        txn_file_refs.write_to_vec(mem_data).unwrap();
    }

    fn get_mem_tables_group_by_type(
        &self,
        cf: usize,
        bounds: &[DataBound<'_>],
    ) -> (Vec<SkipList>, TxnFileRefs) {
        let mut skls = Vec::new();
        let mut txn_file_refs = TxnFileRefs::default();
        for mem_tbl in &self.data.mem_tbls {
            let skl_ext = mem_tbl.get_cf(cf);

            for txn_file_ref in skl_ext
                .get_txn_files()
                .into_iter()
                .filter_map(|txn_file| txn_file.to_txn_file_ref(Some(bounds)))
            {
                txn_file_refs.mut_txn_file_refs().push(txn_file_ref);
            }

            let skl = skl_ext.get_skl();
            if !skl.is_empty() {
                skls.push(skl);
            }
        }
        (skls, txn_file_refs)
    }

    pub fn get_all_sst_files(&self) -> Vec<u64> {
        self.data.get_all_sst_files()
    }

    pub fn has_schema_file(&self) -> bool {
        self.data.schema_file.is_some()
    }

    pub fn get_newer(&self, cf: usize, key: &[u8], version: u64) -> Item<'_> {
        let inner_key = InnerKey::from_outer_key(key, self.data.inner_key_off);
        let mut item = Item::new();
        item.owned_val = Some(vec![]);
        item.val = self.get_newer_val(cf, inner_key, version, item.owned_val.as_mut().unwrap());
        if item.val.is_blob_ref() {
            item.owned_blob = Some(self.fetch_blob(inner_key, &item.val));
            item.val.fill_in_blob(item.owned_blob.as_ref().unwrap());
        }
        item
    }

    fn get_newer_val(
        &self,
        cf: usize,
        inner_key: InnerKey<'_>,
        version: u64,
        out_val_owner: &mut Vec<u8>,
    ) -> table::Value {
        let key_hash = farmhash::fingerprint64(inner_key.deref());
        for i in 0..self.data.mem_tbls.len() {
            let tbl = self.data.mem_tbls.as_slice()[i].get_cf(cf);
            let v = tbl.get_newer(inner_key, version, out_val_owner);
            if v.is_valid() {
                return v;
            }
        }
        for l0 in &self.data.l0_tbls {
            if let Some(tbl) = &l0.get_cf(cf) {
                let v = tbl.get_newer(inner_key, version, key_hash, out_val_owner, 0);
                if v.is_valid() {
                    return v;
                }
            }
        }
        let scf = self.data.get_cf(cf);
        for lh in &scf.levels {
            let v = lh.get_newer(inner_key, version, key_hash, out_val_owner);
            if v.is_valid() {
                return v;
            }
        }
        table::Value::new()
    }

    pub fn has_data_in_prefix<'a>(&'a self, mut prefix: &'a [u8]) -> bool {
        let shard_prefix = self.data.prefix();

        let min_off = std::cmp::min(prefix.len(), self.data.inner_key_off);
        if min_off > 0 {
            if prefix[0..min_off] != shard_prefix[0..min_off] {
                return false;
            }
            if prefix.len() < self.data.inner_key_off {
                // If prefix is less than inner_key_off, the data in shard always have
                // `shard_prefix`, just update prefix to shard_prefix.
                prefix = shard_prefix;
            }
        }

        let inner_prefix = InnerKey::from_outer_key(prefix, self.data.inner_key_off);
        if self.deleting_prefixes.cover_prefix(inner_prefix) {
            return false;
        }
        let mut it = self.new_iterator(0, false, false, Some(u64::MAX), true);
        it.seek(prefix);
        if !it.valid() {
            return false;
        }
        it.key().starts_with(prefix)
    }

    pub fn has_unloaded_tables(&self) -> bool {
        !self.data.unloaded_tbls.is_empty()
    }

    pub fn get_encryption_key(&self) -> Option<EncryptionKey> {
        self.encryption_key.clone()
    }

    pub fn estimated_range_blocks_size(&self, ranges: &[(Bytes, Bytes)]) -> usize {
        // ignore L0 tables, only estimate L1+ for simplicity and performance.
        let inner_key_off = self.data.inner_key_off;
        let mut blocks_size = 0;
        let write_cf = &self.data.cfs[0];
        for lvl in &write_cf.levels {
            blocks_size += lvl.range_blocks_size(ranges, inner_key_off);
        }
        blocks_size
    }

    pub fn get_keyspace_id(&self) -> u32 {
        self.data.keyspace_id
    }

    fn seek_txn_file(&self, iter: &mut Box<dyn table::Iterator>, txn_file: &TxnFile) {
        if txn_file.smallest() < self.data.inner_start() {
            iter.seek(self.data.inner_start());
        } else {
            iter.seek(txn_file.smallest());
        }
    }

    fn get_upper_bound<'a>(&'a self, buf: &'a mut Vec<u8>, txn_file: &TxnFile) -> InnerKey<'_> {
        if txn_file.biggest() < self.data.inner_end() {
            buf.extend_from_slice(txn_file.biggest().deref());
            buf.push(0);
            InnerKey::from_inner_buf(buf)
        } else {
            self.data.inner_end()
        }
    }

    pub fn get_txn_file_conflict_lock(&self, txn_file: &TxnFile) -> Option<(Vec<u8>, Lock)> {
        if txn_file.is_empty() {
            return None;
        }
        if txn_file.size() < self.data.get_cf(LOCK_CF).size() as usize {
            // It is more efficient to iterate the txn file when it is small.
            let mut other_lock_txn_files = self.data.lock_txn_files.clone();
            other_lock_txn_files.retain(|f| f.start_ts() != txn_file.start_ts());
            let mut txn_file_iter = TxnFileIterator::new(txn_file.clone(), false);
            txn_file_iter.rewind();
            while txn_file_iter.valid() {
                let mut val = vec![];
                let txn_file_key = txn_file_iter.key();
                let version = self.get_mem_table_version();
                let lock_val = self.get_value(
                    LOCK_CF,
                    txn_file_key,
                    version,
                    &mut AccessPath::default(),
                    &mut val,
                    &other_lock_txn_files,
                );
                if lock_val.is_valid() && !lock_val.is_deleted() {
                    let conflict_lock = Lock::parse(lock_val.get_value()).unwrap();
                    // Return outer key.
                    let mut key = self.data.prefix().to_vec();
                    key.extend_from_slice(txn_file_key.deref());
                    return Some((key, conflict_lock));
                }
                txn_file_iter.next();
            }
            return None;
        }
        let mut lock_iter =
            self.new_table_iterator(LOCK_CF, false, true, Some(txn_file.start_ts()));
        self.seek_txn_file(&mut lock_iter, txn_file);
        let mut upper_bound_buf = vec![];
        let upper_bound = self.get_upper_bound(&mut upper_bound_buf, txn_file);
        let mut outer_val_buf = vec![];
        while lock_iter.valid() {
            if lock_iter.key() >= upper_bound {
                return None;
            }
            let lock_iter_val = lock_iter.value();
            if !lock_iter_val.is_deleted()
                && txn_file
                    .get_value(lock_iter.key(), &mut outer_val_buf)
                    .1
                    .is_valid()
            {
                let conflict_lock = Lock::parse(lock_iter_val.get_value()).unwrap();
                // Return outer key.
                let mut key = self.data.prefix().to_vec();
                key.extend_from_slice(lock_iter.key().as_ref());
                return Some((key, conflict_lock));
            }
            lock_iter.next();
        }
        None
    }

    // Get writes with `commit_ts` LARGER than `txn_file.start_ts()`.
    // These writes are not seen by clients and should be considered as conflicts.
    // Ref: `SnapAccessCore::get_newer`.
    pub fn get_txn_file_conflict_write(&self, txn_file: &TxnFile) -> Option<(Vec<u8>, UserMeta)> {
        if txn_file.is_empty() {
            return None;
        }
        let mut write_iter = self.new_delta_write_iterator(txn_file.start_ts());
        self.seek_txn_file(&mut write_iter, txn_file);
        let mut upper_bound_buf = vec![];
        let upper_bound = self.get_upper_bound(&mut upper_bound_buf, txn_file);
        let mut outer_val_buf = vec![];
        while write_iter.valid() {
            if write_iter.key() >= upper_bound {
                return None;
            }
            let write_iter_val = write_iter.value();
            if !write_iter_val.is_deleted()
                && txn_file
                    .get_value(write_iter.key(), &mut outer_val_buf)
                    .1
                    .is_valid()
            {
                debug_assert!(
                    !write_iter_val.user_meta().is_empty(),
                    "write_iter_val: {:?}",
                    write_iter_val,
                );
                let um = UserMeta::from_slice(write_iter_val.user_meta());
                if um.commit_ts > txn_file.start_ts() {
                    // Return outer key.
                    let mut key = self.data.prefix().to_vec();
                    key.extend_from_slice(write_iter.key().as_ref());
                    return Some((key, um));
                }
            }
            write_iter.next();
        }
        None
    }

    pub fn check_txn_file_constraint(
        &self,
        txn_file: &TxnFile,
        read_ts: u64,
    ) -> Option<Vec<u8> /* already_exist_key */> {
        let mut already_exist_key: Option<Vec<u8>> = None;

        let mut item = Item::new();
        item.owned_val = Some(vec![]);
        let cb = |key: InnerKey<'_>| -> bool {
            item.path = AccessPath::default();
            item.val = self.get_value(
                WRITE_CF,
                key,
                read_ts,
                &mut item.path,
                item.owned_val.as_mut().unwrap(),
                &[],
            );
            // Blob value is not needed here.

            if item.value_len() > 0 {
                let outer_key = self.data.to_outer_key(key);
                already_exist_key = Some(outer_key);
                return false; // Return false to stop iteration.
            }
            true
        };

        txn_file.iter_check_constraint_keys(self.data.data_bound(), cb);
        already_exist_key
    }

    pub fn get_lock_txn_file(&self, start_ts: u64) -> Option<TxnFile> {
        for txn_file in &self.data.lock_txn_files {
            if txn_file.start_ts() == start_ts {
                return Some(txn_file.clone());
            }
        }
        None
    }

    pub fn get_lock_txn_files(&self) -> &[TxnFile] {
        &self.data.lock_txn_files
    }

    pub fn get_limiter(&self) -> &RegionLimiter {
        &self.data.limiter
    }

    pub fn new_columnar_mvcc_reader(
        &self,
        table_id: i64,
        columns: &[ColumnInfo],
        read_ts: u64,
    ) -> Option<ColumnarMvccReader> {
        if self.columnar_snap_version == 0 {
            return None;
        }
        let schema_file = self.data.schema_file.as_ref()?;
        let table_schema = schema_file.get_table(table_id)?;
        let mut schema_buf = SchemaBuf {
            table_id,
            handle_column: table_schema.handle_column.clone(),
            version_column: table_schema.version_column.clone(),
            txn_id_column: None,
            columns: columns.to_vec(),
            pk_col_ids: table_schema.pk_col_ids.clone(),
            vector_indexes: vec![],
        };
        schema_buf
            .columns
            .retain(|c| !c.get_pk_handle() && c.get_column_id() != HANDLE_COL_ID as i64);
        let schema = Schema::new(schema_buf);
        let mut readers: Vec<Box<dyn ColumnarReader>> = vec![];
        for mem in &self.data.mem_tbls {
            let skl = mem.get_cf(WRITE_CF);
            if !skl.is_empty() {
                let iter = skl.new_iterator(false);
                let row_reader = ColumnarRowTableReader::new(
                    self.data.keyspace_id,
                    self.data.inner_key_off,
                    schema.clone(),
                    iter,
                    None,
                    false,
                    self.encryption_key.clone(),
                );
                readers.push(Box::new(row_reader));
            }
        }
        for l0 in &self.data.col_levels.unconverted_l0s {
            if let Some(l0_write) = l0.get_cf(WRITE_CF) {
                let iter = l0_write.new_iterator(false, true);
                let row_reader = ColumnarRowTableReader::new(
                    self.data.keyspace_id,
                    self.data.inner_key_off,
                    schema.clone(),
                    iter,
                    None,
                    false,
                    self.encryption_key.clone(),
                );
                readers.push(Box::new(row_reader));
            }
        }
        for columnar_level in &self.data.col_levels.levels {
            if columnar_level.level == 2 {
                let concat_reader = ColumnarConcatReader::new(
                    &columnar_level.files,
                    schema.clone(),
                    self.encryption_key.clone(),
                );
                readers.push(Box::new(concat_reader));
            } else {
                for col_file in &columnar_level.files {
                    if !col_file.has_table(schema.table_id) {
                        continue;
                    }
                    let col_reader = ColumnarTableReader::new(
                        col_file,
                        schema.clone(),
                        self.encryption_key.clone(),
                    );
                    readers.push(Box::new(col_reader));
                }
            }
        }
        let merged_reader = ColumnarMergeReader::new(schema.clone(), readers);
        let mvcc_reader = ColumnarMvccReader::new(Box::new(merged_reader), &schema, read_ts);
        Some(mvcc_reader)
    }
}

pub struct Iterator {
    all_versions: bool,
    reversed: bool,
    read_ts: u64,
    pub(crate) key: BytesMut,
    val: table::Value,
    pub(crate) inner: Box<dyn table::Iterator>,
    blob_prefetcher: Option<BlobPrefetcher>,
    data: ShardData,
    range: Option<(Bytes, Bytes)>, // [outer_lower_bound, outer_upper_bound)
}

impl Iterator {
    pub fn valid(&self) -> bool {
        self.val.is_valid()
    }

    pub fn key(&self) -> &[u8] {
        self.key.chunk()
    }

    pub fn val(&mut self) -> &[u8] {
        if self.val.is_blob_ref() {
            if let Some(prefetcher) = &mut self.blob_prefetcher {
                let blob_ref = self.val.get_blob_ref();
                return prefetcher.get(&blob_ref).unwrap_or_else(|e| {
                    panic!("failed to get blob, blob_ref: {:?}, err: {:?}", blob_ref, e)
                });
            }
        }
        self.val.get_value()
    }

    pub fn version(&self) -> u64 {
        self.val.version
    }

    pub fn meta(&self) -> u8 {
        self.val.meta
    }

    pub fn user_meta(&self) -> &[u8] {
        self.val.user_meta()
    }

    pub fn valid_for_prefix(&self, prefix: &[u8]) -> bool {
        self.key.starts_with(prefix)
    }

    pub fn next(&mut self) {
        if self.all_versions
            && self.valid()
            && self.inner.next_version()
            && !self.inner.value().is_deleted()
        {
            self.update_item();
            return;
        }
        self.inner.next();
        self.parse_item();
    }

    fn update_item(&mut self) {
        self.key.truncate(self.data.inner_key_off);
        self.key.extend_from_slice(self.inner.key().deref());
        self.val = self.inner.value();
    }

    fn parse_item(&mut self) {
        while self.inner.valid() {
            if self.is_inner_key_over_bound() {
                break;
            }
            let val = self.inner.value();
            if val.version > self.read_ts && !self.inner.seek_to_version(self.read_ts) {
                self.inner.next();
                continue;
            }
            if self.inner.value().is_deleted() {
                self.inner.next();
                continue;
            }
            self.update_item();
            return;
        }
        self.val = table::Value::new();
    }

    // seek would seek to the provided key if present. If absent, it would seek to
    // the next smallest key greater than provided if iterating in the forward
    // direction. Behavior would be reversed is iterating backwards.
    pub fn seek(&mut self, key: &[u8]) {
        if key.len() <= self.data.inner_key_off {
            self.inner.rewind();
        } else {
            self.inner
                .seek(InnerKey::from_outer_key(key, self.data.inner_key_off));
        }
        self.parse_item();
    }

    // rewind would rewind the iterator cursor all the way to zero-th position,
    // which would be the smallest key if iterating forward, and largest if
    // iterating backward. It does not keep track of whether the cursor started
    // with a seek().
    pub fn rewind(&mut self) {
        self.inner.rewind();
        if self.inner.valid() {
            if self.reversed {
                if self.inner.key() >= self.data.inner_end() {
                    self.inner.seek(self.data.inner_end());
                    if self.inner.key() == self.data.inner_end() {
                        self.inner.next();
                    }
                }
            } else if self.inner.key() < self.data.inner_start() {
                self.inner.seek(self.data.inner_start())
            }
        }
        self.parse_item();
    }

    pub fn set_all_versions(&mut self, all_versions: bool) {
        self.all_versions = all_versions;
    }

    pub fn is_reverse(&self) -> bool {
        self.reversed
    }

    // set the new range of the iterator, it the range is monotonic, we can avoid
    // seek. return true if seek is performed.
    #[allow(clippy::collapsible_else_if)]
    pub fn set_range(
        &mut self,
        outer_lower_bound_include: Bytes,
        outer_upper_bound_exclude: Bytes,
    ) -> bool {
        let inner_lower_bound =
            InnerKey::from_outer_key(&outer_lower_bound_include, self.data.inner_key_off);
        let inner_upper_bound =
            InnerKey::from_outer_end_key(&outer_upper_bound_exclude, self.data.inner_key_off);
        let mut seeked = false;
        // reset monotonic range can be optimized to avoid seek.
        if self.is_reset_monotonic_range(inner_lower_bound, inner_upper_bound) {
            // If inner is not valid, the iterator has reached the end, there is no more
            // data to return, we can avoid the seek.
            if self.inner.valid() {
                if self.reversed {
                    // If the new inner_upper_bound is greater than the current key, we can
                    // continue to use the current key to iterate backward, avoid the seek.
                    if self.inner.key() > inner_upper_bound {
                        self.inner.seek(inner_upper_bound);
                        seeked = true;
                    }
                    // the upper bound is exclusive, so we need to skip the current key.
                    if self.inner.key() == inner_upper_bound {
                        self.inner.next();
                    }
                } else {
                    // If the new inner_lower_bound is greater than or equal to the current key,
                    // we can continue to use the current key to iterate forward, avoid the seek.
                    if self.inner.key() < inner_lower_bound {
                        self.inner.seek(inner_lower_bound);
                        seeked = true;
                    }
                }
            }
        } else {
            // always seek if not reset monotonic range.
            if self.reversed {
                self.inner.seek(inner_upper_bound);
                if self.inner.valid() && self.inner.key() == inner_upper_bound {
                    self.inner.next();
                }
            } else {
                self.inner.seek(inner_lower_bound);
            }
            seeked = true;
        }
        self.range = Some((outer_lower_bound_include, outer_upper_bound_exclude));
        self.parse_item();
        seeked
    }

    fn is_reset_monotonic_range(
        &self,
        inner_lower_bound: InnerKey<'_>,
        inner_upper_bound: InnerKey<'_>,
    ) -> bool {
        if let Some((outer_old_lower, outer_old_upper)) = &self.range {
            if self.reversed {
                inner_upper_bound
                    <= InnerKey::from_outer_key(outer_old_lower, self.data.inner_key_off)
            } else {
                InnerKey::from_outer_end_key(outer_old_upper, self.data.inner_key_off)
                    <= inner_lower_bound
            }
        } else {
            false
        }
    }

    pub(crate) fn is_inner_key_over_bound(&self) -> bool {
        if let Some((outer_lower, outer_upper)) = &self.range {
            if self.inner.valid() {
                if self.reversed {
                    self.inner.key()
                        < InnerKey::from_outer_key(outer_lower, self.data.inner_key_off)
                } else {
                    self.inner.key()
                        >= InnerKey::from_outer_end_key(outer_upper, self.data.inner_key_off)
                }
            } else {
                true
            }
        } else if self.reversed {
            self.inner.key() < self.data.inner_start()
        } else {
            self.inner.key() >= self.data.inner_end()
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::BTreeMap, iter::Iterator, ops::Deref, sync::Arc};

    use api_version::{api_v2::KEYSPACE_PREFIX_LEN, ApiV2};
    use bytes::{Buf, Bytes};
    use cloud_encryption::{EncryptionKey, MasterKey};
    use futures::executor::block_on;
    use kvenginepb::TableCreate;
    use proptest::prelude::*;
    use protobuf::Message;

    use crate::{
        apply::create_snapshot_tables,
        dfs::{self, Dfs, InMemFs},
        read::MEM_DATA_FORMAT_V1,
        shard::ShardDataBuilder,
        table::{
            self,
            file::InMemFile,
            memtable::CfTable,
            sstable::{build_test_table_with_kvs, BlockCache},
            InnerKey, NoPrefixKey, OwnedInnerKey, TxnChunk, TxnChunkBuilder, TxnCtx, TxnFile,
            TxnFileId, OP_PUT,
        },
        txn_chunk_manager::{with_pool_size, TxnChunkManager, TxnChunkManagerConfig},
        util::test_util::KeyBuilder,
        ChangeSet, Shard, ShardRange, SnapAccess, UserMeta, ENCRYPTION_KEY, GLOBAL_SHARD_END_KEY,
        WRITE_CF,
    };

    const KEYSPACE_ID: u32 = 42;

    #[test]
    fn test_estimated_range_blocks_size() {
        let mut cs_pb = kvenginepb::ChangeSet::default();
        let mut cs = ChangeSet::new(cs_pb.clone());
        let snap = cs_pb.mut_snapshot();
        let mut build_table_fn =
            |start: i32, end: i32, level: u32, tbl_entries: usize, step: usize| {
                let mut kvs = vec![];
                for i in (start..end).step_by(step) {
                    let key = format!("key{:05x}", i);
                    let val = key.repeat(10);
                    kvs.push((key, val));
                    if kvs.len() == tbl_entries {
                        let tbl = build_test_table_with_kvs(&kvs);
                        let mut tbl_create = TableCreate::default();
                        tbl_create.id = tbl.id();
                        tbl_create.level = level;
                        tbl_create.smallest = tbl.smallest().to_vec();
                        tbl_create.biggest = tbl.biggest().to_vec();
                        kvs.truncate(0);
                        cs.ln_tables.insert(tbl.id(), tbl.clone());
                        snap.mut_table_creates().push(tbl_create);
                    }
                }
            };
        build_table_fn(0, 10000, 3, 1000, 1);
        build_table_fn(0, 10000, 2, 500, 5);
        build_table_fn(0, 10000, 1, 200, 20);
        cs.change_set = cs_pb;
        let range = ShardRange::new(&[], GLOBAL_SHARD_END_KEY, 0);
        let opt = Arc::new(crate::options::Options::default());
        let master_key = MasterKey::new(&[1u8; 32]);
        let shard = Shard::new(
            1,
            &kvenginepb::Properties::new(),
            1,
            range,
            opt,
            &master_key,
        );
        let mut builder = ShardDataBuilder::new(shard.get_data());
        create_snapshot_tables(&mut builder, cs.get_snapshot(), &cs, false);
        builder.set_schema_file(cs.schema_file.clone());
        shard.set_data(builder.build());
        let snap = shard.new_snap_access();

        let build_range_fn = |start: i32, end: i32| {
            (
                Bytes::from(format!("key{:05x}", start)),
                Bytes::from(format!("key{:05x}", end)),
            )
        };
        let total_blocks_size = snap.estimated_range_blocks_size(&[build_range_fn(0, 10000)]);
        assert_eq!(total_blocks_size, 1103598);

        // verify that many small ranges are properly deduplicated, num blocks never
        // exceed total.
        let mut many_small_ranges = vec![];
        for i in (0..10000).step_by(10) {
            let small_range = build_range_fn(i, i + 5);
            many_small_ranges.push(small_range);
        }
        let many_small_ranges_blocks_size = snap.estimated_range_blocks_size(&many_small_ranges);
        assert_eq!(many_small_ranges_blocks_size + 2, total_blocks_size);

        let half_num_blocks = snap.estimated_range_blocks_size(&[build_range_fn(5000, 10000)]);
        assert_eq!(half_num_blocks, 548238);

        for i in (100..10000).step_by(100) {
            let blocks_size = snap.estimated_range_blocks_size(&[build_range_fn(i, i + 1)]);
            // some range on level 1 doesn't overlap any table, so blocks_size may vary.
            assert!(blocks_size == 10543 || blocks_size == 6983);
        }

        let blocks_size = snap.estimated_range_blocks_size(&[
            build_range_fn(1, 2),
            build_range_fn(2, 3),
            build_range_fn(3, 4),
            build_range_fn(4, 5),
        ]);
        // each level only access one block.
        assert_eq!(blocks_size, 10543);

        let blocks_size = snap.estimated_range_blocks_size(&[
            build_range_fn(1, 2),
            build_range_fn(2000, 2001),
            build_range_fn(3000, 3001),
            build_range_fn(4000, 4001),
        ]);
        // each range on each level access one block.
        assert_eq!(blocks_size, 42172);
    }

    const MAX_I: usize = 100;

    fn mem_table_op_strategy() -> impl Strategy<Value = MemTableOp> {
        prop_oneof![
            (prop::collection::vec(1..MAX_I, 0..=20usize), any::<bool>())
                .prop_map(|(batch, switch)| MemTableOp::WriteBatch(batch, switch)),
            (prop::collection::vec(1..MAX_I, 0..=20usize), any::<bool>())
                .prop_map(|(batch, switch)| MemTableOp::WriteTxnFile(batch, switch)),
        ]
    }

    prop_compose! {
        fn arb_mem_table_ops(size: usize)
            (ops in prop::collection::vec(mem_table_op_strategy(), 0..=size))
            -> Vec<MemTableOp> {
            ops
        }
    }

    // [10, 20, 30, 40, 50] -> [(10, 20), (30, 40)]
    fn query_ranges() -> impl Strategy<Value = Vec<(usize, usize)>> {
        prop::collection::vec(0..MAX_I + 1, 0..=10usize).prop_map(|mut is| -> Vec<(usize, usize)> {
            is.sort();
            let ranges = is
                .chunks_exact(2)
                .filter_map(|chunk| (chunk[0] != chunk[1]).then_some((chunk[0], chunk[1])))
                .collect::<Vec<_>>();
            if ranges.is_empty() {
                vec![(0, MAX_I)]
            } else {
                ranges
            }
        })
    }

    proptest! {
        #[test]
        fn test_serde_mem_tables(
            ops in arb_mem_table_ops(10),
            ranges in query_ranges(),
            enable_enc in any::<bool>(),
            enable_inner_key_off in any::<bool>(),
        ) {
            let runtime = tokio::runtime::Runtime::new().unwrap();
            let _enter = runtime.enter();

            let engine_id = 1;
            let shard_id = 1;
            let shard_ver = 10;

            let kb = KeyBuilder::new(KEYSPACE_ID, enable_inner_key_off, "t_");
            let dfs: Arc<dyn crate::dfs::Dfs> = Arc::new(InMemFs::new());
            let txn_chunk_manager = TxnChunkManager::new(None, dfs.clone(), BlockCache::None, with_pool_size(2), TxnChunkManagerConfig::default());

            let master_key = MasterKey::new(&[1u8; 32]);
            let enc_key = enable_enc.then(||master_key.generate_encryption_key());

            let has_txn_file = ops.iter().any(|op| matches!(op, MemTableOp::WriteTxnFile(is, _) if !is.is_empty()));
            let (mem_tbls, ref_store) = make_mem_tables(ops, &kb, dfs.clone(), enc_key.as_ref());
            verify_mem_tables(&mem_tbls, &ref_store, &[(kb.i_to_inner_key(0).as_ref(), kb.i_to_inner_key(MAX_I).as_ref())])?;

            let (outer_start, outer_end) = ApiV2::get_txn_keyspace_range(KEYSPACE_ID);
            let range = ShardRange::new(&outer_start, &outer_end, KEYSPACE_PREFIX_LEN * enable_inner_key_off as usize);
            let opt = Arc::new(crate::options::Options::default());

            let inner_ranges = ranges.iter().map(|&(start, end)| {
                (kb.i_to_inner_key(start), kb.i_to_inner_key(end))
            }).collect::<Vec<_>>();
            let outer_ranges = ranges.into_iter().map(|(start, end)| {
                (Bytes::from(kb.i_to_outer_key(start)), Bytes::from(kb.i_to_outer_key(end)))
            }).collect::<Vec<_>>();

            // Serialize
            let mem_bin = {
                let mut props = kvenginepb::Properties {
                    shard_id,
                    ..Default::default()
                };
                if let Some(enc_key) = &enc_key {
                    props.mut_keys().push(ENCRYPTION_KEY.to_string());
                    props.mut_values().push(enc_key.export());
                }
                let shard = Shard::new(
                    engine_id,
                    &props,
                    shard_ver,
                    range,
                    opt,
                    &master_key,
                );
                let mut builder = ShardDataBuilder::new(shard.get_data());
                builder.set_mem_tbls(mem_tbls);
                shard.set_data(builder.build());
                let snap_access = shard.new_snap_access();

                snap_access.build_mem_data(&outer_ranges, u64::MAX)
            };
            if !has_txn_file {
                let format_ver = mem_bin.as_slice().get_u32_le();
                prop_assert_eq!(format_ver, MEM_DATA_FORMAT_V1);
            }

            // Deserialize
            let mut snap_pb = kvenginepb::Snapshot::default();
            snap_pb.set_inner_key_off(KEYSPACE_PREFIX_LEN as u32 * enable_inner_key_off as u32);
            let props = snap_pb.mut_properties();
            if let Some(enc_key) = &enc_key {
                props.mut_keys().push(ENCRYPTION_KEY.to_string());
                props.mut_values().push(enc_key.export());
            }
            let mut cs = kvenginepb::ChangeSet::default();
            cs.set_shard_id(shard_id);
            cs.set_shard_ver(shard_ver);
            cs.set_snapshot(snap_pb);
            let snap_bin = cs.write_to_bytes().unwrap();
            let remote_snap = block_on(SnapAccess::construct_snapshot("test".to_owned(), dfs, &mem_bin, &snap_bin, &master_key, BlockCache::None, None, txn_chunk_manager)).unwrap();

            let inner_ranges = inner_ranges.iter().map(|(start, end)| (start.as_ref(), end.as_ref())).collect::<Vec<_>>();
            let ref_store_in_ranges = ref_store.new_in_ranges(&inner_ranges);
            verify_mem_tables(remote_snap.data.mem_tbls.as_slice(), &ref_store_in_ranges, &inner_ranges)?;
        }
    }

    #[derive(Debug, Clone)]
    enum MemTableOp {
        WriteBatch(Vec<usize>, bool /* switch */),
        WriteTxnFile(Vec<usize>, bool /* switch */),
    }

    #[derive(Default, Debug)]
    struct RefStore {
        inner: BTreeMap<Bytes, String>,
    }

    impl RefStore {
        fn put(&mut self, key: OwnedInnerKey, val: String) {
            self.inner.insert(key.into_inner(), val);
        }

        fn get_value(&self, key: InnerKey<'_>) -> Option<&String> {
            self.inner.get(key.deref())
        }

        fn new_in_ranges(&self, ranges: &[(InnerKey<'_>, InnerKey<'_>)]) -> Self {
            let mut new_store = RefStore::default();
            for &(start, end) in ranges {
                let start = Bytes::copy_from_slice(start.deref());
                let end = Bytes::copy_from_slice(end.deref());
                for (k, v) in self.inner.range::<Bytes, _>(&start..&end) {
                    new_store.inner.insert(k.clone(), v.clone());
                }
            }
            new_store
        }

        fn len(&self) -> usize {
            self.inner.len()
        }
    }

    fn make_mem_tables(
        ops: Vec<MemTableOp>,
        kb: &KeyBuilder,
        dfs: Arc<dyn Dfs>,
        enc_key: Option<&EncryptionKey>,
    ) -> (Vec<CfTable>, RefStore) {
        let mut mem_tbls = vec![CfTable::new()];
        let mut ref_store = RefStore::default();
        let mut wb = crate::table::memtable::WriteBatch::new();
        let mut next_txn_chunk_id = 1000;

        let switch_mem_table = |mem_tbls: &mut Vec<CfTable>| {
            let mut new_mem_tbls = Vec::with_capacity(mem_tbls.len() + 1);
            new_mem_tbls.push(CfTable::new());
            new_mem_tbls.append(mem_tbls);
            *mem_tbls = new_mem_tbls;
        };

        for (idx, op) in ops.into_iter().enumerate() {
            let start_ts = idx as u64 * 100;
            let commit_ts = (idx as u64 + 1) * 100;
            let user_meta = UserMeta::new(start_ts, commit_ts);

            match op {
                MemTableOp::WriteBatch(is, switch) => {
                    for i in is {
                        let key = kb.i_to_inner_key(i);
                        let val = kb.i_to_val(start_ts as usize + i);
                        wb.put(
                            key.as_ref(),
                            0,
                            &user_meta.to_array(),
                            commit_ts,
                            val.as_bytes(),
                        );
                        ref_store.put(key, val);
                    }
                    let mem_tbl = mem_tbls[0].get_cf(WRITE_CF);
                    mem_tbl.put_batch(&mut wb, None, WRITE_CF);
                    wb.reset();

                    if switch {
                        switch_mem_table(&mut mem_tbls);
                    }
                }
                MemTableOp::WriteTxnFile(mut is, switch) => {
                    is.sort();
                    is.dedup();
                    let mut txn_chunks = vec![];
                    for chunk in is.chunks(3) {
                        next_txn_chunk_id += 1;
                        let mut builder = TxnChunkBuilder::new(
                            next_txn_chunk_id,
                            10,
                            enc_key.cloned(),
                            KEYSPACE_ID,
                            kb.get_enable_inner_key_off(),
                        );
                        for &i in chunk {
                            let key = kb.i_to_inner_key(i);
                            let val = kb.i_to_val(start_ts as usize + i);
                            builder.add_entry(NoPrefixKey(&kb.i_to_key(i)), OP_PUT, val.as_bytes());
                            ref_store.put(key, val);
                        }
                        let mut chunk_data = vec![];
                        builder.finish(&mut chunk_data);

                        let opts = dfs::Options::default().with_type(dfs::FileType::TxnChunk);
                        block_on(dfs.create(next_txn_chunk_id, chunk_data.clone().into(), opts))
                            .unwrap();

                        let chunk_file =
                            Arc::new(InMemFile::new(next_txn_chunk_id, chunk_data.into()));
                        let txn_chunk =
                            TxnChunk::new(chunk_file, BlockCache::None, enc_key.cloned()).unwrap();
                        txn_chunks.push(txn_chunk);
                    }

                    if !txn_chunks.is_empty() {
                        let id = TxnFileId::new(1, 1, start_ts);
                        let lower_bound = InnerKey::from_inner_buf(b"");
                        let upper_bound = InnerKey::from_inner_buf(GLOBAL_SHARD_END_KEY);
                        let txn_ctx = TxnCtx::new(
                            user_meta.to_array().to_vec().into(),
                            Default::default(),
                            commit_ts,
                            lower_bound,
                            upper_bound,
                        );
                        let txn_file = TxnFile::new(id, txn_chunks, txn_ctx).unwrap();
                        mem_tbls[0] = mem_tbls[0].add_write_cf_txn_files(&[txn_file]);
                        if switch {
                            switch_mem_table(&mut mem_tbls);
                        }
                    }
                }
            }
        }
        (mem_tbls, ref_store)
    }

    fn verify_mem_tables(
        mem_tbls: &[CfTable],
        ref_store: &RefStore,
        ranges: &[(InnerKey<'_>, InnerKey<'_>)],
    ) -> std::result::Result<(), TestCaseError> {
        let mem_iters = mem_tbls
            .iter()
            .map(|m| m.get_cf(WRITE_CF).new_iterator(false))
            .collect();
        let mut iter = table::new_merge_iterator(mem_iters, false);
        let mut count = 0;
        for &(start, end) in ranges {
            iter.seek(start);
            while iter.valid() && iter.key() < end {
                count += 1;
                let expect = ref_store.get_value(iter.key());
                prop_assert!(expect.is_some(), "key: {:?}", iter.key());
                let val = iter.value();
                prop_assert_eq!(val.get_value(), expect.unwrap().as_bytes());

                iter.next();
            }
        }
        prop_assert_eq!(count, ref_store.len());
        Ok(())
    }
}
