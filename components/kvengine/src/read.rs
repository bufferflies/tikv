// Copyright 2021 TiKV Project Authors. Licensed under Apache-2.0.

use core::panic;
use std::{
    collections::HashMap,
    fmt::{Debug, Formatter},
    marker::PhantomData,
    ops::Deref,
    sync::{Arc, Mutex},
};

use bytes::{Buf, BufMut, Bytes, BytesMut};
use cloud_encryption::{EncryptionKey, MasterKey};
use kvenginepb as pb;
use kvenginepb::TxnFileRef;
use moka::sync::SegmentedCache;
use protobuf::Message;
use txn_types::Lock;

use crate::{
    limiter::RegionLimiter,
    table::{
        blobtable::blobtable::{BlobPrefetcher, BlobTable},
        memtable::{CfTable, Hint, WriteBatch},
        sstable::{BlockCacheKey, InMemFile, L0Table, SsTable},
        table, InnerKey, SkipOpTxnFileIterator, TableExt, TxnFile, TxnFileIterator,
    },
    *,
};

const MEM_DATA_FORMAT_V1: u32 = 1;

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
        dfs: Arc<dyn dfs::Dfs>,
        change_set: pb::ChangeSet,
        ignore_lock: bool,
        master_key: &MasterKey,
        block_cache: Option<SegmentedCache<BlockCacheKey, Bytes>>,
    ) -> Self {
        let core = Arc::new(
            SnapAccessCore::from_change_set(
                dfs,
                change_set,
                None,
                ignore_lock,
                master_key,
                block_cache,
            )
            .await,
        );
        Self { core }
    }

    async fn from_change_set_and_memtable_data(
        dfs: Arc<dyn dfs::Dfs>,
        change_set: pb::ChangeSet,
        wb: &mut WriteBatch,
        master_key: &MasterKey,
        block_cache: Option<SegmentedCache<BlockCacheKey, Bytes>>,
    ) -> Self {
        let wb = if wb.is_empty() { None } else { Some(wb) };
        let core = Arc::new(
            SnapAccessCore::from_change_set(dfs, change_set, wb, true, master_key, block_cache)
                .await,
        );
        Self { core }
    }

    pub async fn construct_snapshot<'a>(
        dfs: Arc<dyn dfs::Dfs>,
        mut mem_table_data: &[u8],
        snapshot: &[u8],
        master_key: &MasterKey,
        block_cache: Option<SegmentedCache<BlockCacheKey, Bytes>>,
    ) -> Result<Self> {
        let mut change_set = kvenginepb::ChangeSet::default();
        change_set.merge_from_bytes(snapshot).unwrap();
        let inner_key_off = change_set.get_snapshot().get_inner_key_off() as usize;
        let mut wb = crate::table::memtable::WriteBatch::new();
        if !mem_table_data.is_empty() {
            let format_version = mem_table_data.get_u32_le();
            if format_version != MEM_DATA_FORMAT_V1 {
                return Err(Error::RemoteRead(format!(
                    "unsupported mem data format {}",
                    format_version
                )));
            }
            let rows: Vec<table::Row> = bincode::deserialize(mem_table_data).map_err(|e| {
                Error::RemoteRead(format!("failed to deserialize mem table data: {}", e))
            })?;
            for row in rows {
                let key = InnerKey::from_outer_key(&row.key, inner_key_off);
                wb.put(key, 0, &row.user_meta.to_array(), 0, &row.value);
            }
        }
        Ok(Self::from_change_set_and_memtable_data(
            dfs,
            change_set,
            &mut wb,
            master_key,
            block_cache,
        )
        .await)
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

pub struct SnapAccessCore {
    tag: ShardTag,
    managed_ts: u64,
    _base_version: u64,
    meta_seq: u64,
    write_sequence: u64,
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
        let data = shard.get_data();
        Self {
            tag: shard.tag(),
            write_sequence,
            meta_seq,
            _base_version: base_version,
            managed_ts: 0,
            data,
            get_hint: Mutex::new(Hint::new()),
            blob_table_prefetch_size: shard.opt.blob_prefetch_size,
            deleting_prefixes: shard.get_del_prefixes(),
            encryption_key: shard.encryption_key.clone(),
        }
    }

    pub async fn from_change_set(
        dfs: Arc<dyn dfs::Dfs>,
        change_set: pb::ChangeSet,
        wb: Option<&mut WriteBatch>,
        ignore_lock: bool,
        master_key: &MasterKey,
        block_cache: Option<SegmentedCache<BlockCacheKey, Bytes>>,
    ) -> Self {
        let mut cs = ChangeSet::new(change_set);
        let mut ids = HashMap::new();
        let mut _txn_file_refs: Vec<TxnFileRef> = vec![];
        let mem_tbls = vec![CfTable::new()];
        // FIXME: Iterate over all memtables
        if let Some(wb) = wb {
            let mem_tbl = mem_tbls[0].get_cf(WRITE_CF);
            mem_tbl.put_batch(wb, None, WRITE_CF);
            // Insert a dummy record, that should be ignored, so that we
            // don't have to fiddle too much with the code below.
            ids.insert(0, 0);
        }
        let encryption_key = if cs.has_snapshot() {
            let snap = cs.get_snapshot();
            for l0 in snap.get_l0_creates() {
                ids.insert(l0.id, 0);
            }
            for ln in snap.get_table_creates() {
                ids.insert(ln.id, ln.level);
            }
            for blob in snap.get_blob_creates() {
                ids.insert(blob.id, BLOB_LEVEL);
            }
            // TODO: load lock_txn_files
            // if !ignore_lock {
            //     _txn_file_refs = collect_snap_lock_txn_file_refs(snap);
            // }
            get_shard_property(ENCRYPTION_KEY, snap.get_properties())
                .map(|v| master_key.decrypt_encryption_key(&v).unwrap())
        } else {
            None
        };
        let (result_tx, mut result_rx) = tokio::sync::mpsc::unbounded_channel();
        let runtime = dfs.get_runtime();
        let opts = dfs::Options::new(cs.shard_id, cs.shard_ver);
        let mut msg_count = 0;
        for (&id, &level) in &ids {
            let tx = result_tx.clone();
            if id == 0 {
                tx.send(Ok((0, 0, None))).unwrap();
            } else {
                let fs = dfs.clone();
                let tx = result_tx.clone();
                runtime.spawn(async move {
                    let res = fs.read_file(id, opts).await;
                    tx.send(res.map(|data| (id, level, Some(data))))
                        .map_err(|_| "send file data failed")
                        .unwrap();
                });
            }
            msg_count += 1;
        }
        let mut errors = vec![];
        for _ in 0..msg_count {
            match result_rx.recv().await.unwrap() {
                Ok((id, _, None)) => {
                    assert_eq!(id, 0);
                }
                Ok((id, level, Some(data))) => {
                    assert!(id != 0);
                    let file = InMemFile::new(id, data);
                    if is_blob_file(level) {
                        let blob_table = BlobTable::new(Arc::new(file)).unwrap();
                        cs.blob_tables.insert(id, blob_table);
                    } else if level == 0 {
                        let l0_table = L0Table::new(
                            Arc::new(file),
                            block_cache.clone(),
                            ignore_lock,
                            encryption_key.clone(),
                        )
                        .unwrap();
                        if let Some(l0_table) = l0_table {
                            cs.l0_tables.insert(id, l0_table);
                        }
                    } else {
                        let ln_table = SsTable::new(
                            Arc::new(file),
                            block_cache.clone(),
                            level == 1,
                            encryption_key.clone(),
                        )
                        .unwrap();
                        cs.ln_tables.insert(id, ln_table);
                    }
                }
                Err(err) => {
                    error!("prefetch failed {:?}", &err);
                    errors.push(err);
                }
            }
        }
        if !errors.is_empty() {
            panic!("errors is not empty: {:?}", errors);
        }

        // TODO: load lock_txn_files

        let mut shard = Shard::new_for_ingest(0, &cs, Arc::new(Options::default()), master_key);
        let (l0s, blob_tbls, scfs, lock_txn_files) =
            create_snapshot_tables(cs.get_snapshot(), &cs, ignore_lock);
        let data = ShardData::new(
            shard.range.clone(),
            mem_tbls,
            l0s,
            Arc::new(blob_tbls),
            scfs,
            HashMap::new(),
            lock_txn_files,
            RegionLimiter::new((&shard.opt.flow_control).into()), // Note: limiter is disabled here
        );
        shard.id = cs.shard_id;
        shard.set_data(data);
        Self::new(&shard)
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
        blob_table
            .get(&blob_ref)
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
            false,
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
            true,
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
        ignore_txn_file: bool,
    ) -> table::Value {
        if cf == LOCK_CF && !ignore_txn_file {
            for txn_file in &self.data.lock_txn_files {
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
                iters.push(mem_tbl.get_cf(WRITE_CF).new_iterator(false));
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

    pub fn get_start_key(&self) -> &[u8] {
        self.data.outer_start.chunk()
    }

    pub fn get_end_key(&self) -> &[u8] {
        self.data.outer_end.chunk()
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
                if v.has_data_in_range(inner_start, inner_end) {
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
                    if v.has_overlap(inner_start, inner_end, false) {
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

    pub fn build_mem_data(&self, ranges: &[(Bytes, Bytes)], start_ts: u64) -> Vec<u8> {
        let mut mem_iterator = self.new_memtable_iterator(0, false, false, Some(start_ts));
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
        let mut mem_data = Vec::with_capacity(mem_size as usize + 4);
        mem_data.put_u32_le(MEM_DATA_FORMAT_V1);
        bincode::serialize_into(&mut mem_data, &rows).unwrap();
        mem_data
    }

    pub fn get_all_files(&self) -> Vec<u64> {
        self.data.get_all_files()
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
                out_val_owner.resize(v.encoded_size(), 0);
                v.encode(out_val_owner.as_mut_slice());
                return table::Value::decode(out_val_owner.as_slice());
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
    use std::iter::Iterator;

    use kvenginepb::TableCreate;

    use super::*;
    use crate::table::sstable::build_test_table_with_kvs;

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
                        let tbl = build_test_table_with_kvs(&kvs, false);
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

        let (l0s, blob_tbls, scfs, lock_txn_files) =
            create_snapshot_tables(cs.get_snapshot(), &cs, false);
        let data = ShardData::new(
            shard.range.clone(),
            vec![CfTable::new()],
            l0s,
            Arc::new(blob_tbls),
            scfs,
            HashMap::new(),
            lock_txn_files,
            RegionLimiter::dummy(),
        );
        shard.set_data(data);
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
}
