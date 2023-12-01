// Copyright 2021 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    cmp,
    collections::HashMap,
    fmt,
    iter::Iterator,
    ops::Deref,
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering::*, *},
        Arc, RwLock,
    },
};

use api_version::ApiV2;
use bytes::{Buf, BufMut, Bytes};
use cloud_encryption::{EncryptionKey, MasterKey};
use dashmap::DashMap;
use kvenginepb as pb;
use rand::Rng;
use slog_global::*;
use tikv_util::codec::number::U64_SIZE;

use crate::{
    table::{
        self,
        blobtable::blobtable::BlobTable,
        get_tables_in_range,
        memtable::{self, CfTable},
        search,
        sstable::{L0Table, SsTable},
        InnerKey, TableExt,
    },
    util::evenly_distribute,
    Iterator as TableIterator, *,
};

#[derive(Clone)]
pub(crate) struct ShardPendingOperations {
    pub(crate) del_prefixes: Arc<DeletePrefixes>,
    pub(crate) truncate_ts: Option<TruncateTs>,
    pub(crate) trim_over_bound: bool,
    pub(crate) manual_major_compaction: bool,
}

impl ShardPendingOperations {
    fn new(inner_key_off: usize) -> Self {
        Self {
            del_prefixes: Arc::new(DeletePrefixes::new_with_inner_key_off(inner_key_off)),
            truncate_ts: None,
            trim_over_bound: false,
            manual_major_compaction: false,
        }
    }
}

pub struct Shard {
    pub engine_id: u64,
    pub id: u64,
    pub ver: u64,
    pub range: ShardRange,
    pub parent_id: u64,
    pub(crate) data: RwLock<ShardData>,
    pub(crate) opt: Arc<Options>,

    // If the shard is not active, flush mem table and do compaction will ignore this shard.
    pub(crate) active: AtomicBool,

    pub(crate) properties: Properties,
    pub(crate) pending_ops: RwLock<ShardPendingOperations>,
    pub(crate) compacting: AtomicBool, // for statistics only
    pub(crate) initial_flushed: AtomicBool,

    pub(crate) base_version: AtomicU64,
    // Size of all SSTables + all blobs referred (note, not the blob table size).
    pub(crate) estimated_size: AtomicU64,
    pub(crate) estimated_entries: AtomicU64,
    pub(crate) max_ts: AtomicU64,
    pub(crate) estimated_kv_size: AtomicU64,

    pub(crate) sst_max_ts: AtomicU64, // the max_ts of sst files (mem-tables excluded)
    pub(crate) tombs: AtomicU64,      // number of tombstone entries
    pub(crate) entries_write_cf: AtomicU64, // number of entries in WRITE_CF

    // meta_seq is the raft log index of the applied change set.
    // Because change set are applied in the worker thread, the value is usually smaller
    // than write_sequence.
    pub(crate) meta_seq: AtomicU64,

    // write_sequence is the raft log index of the applied write batch.
    pub(crate) write_sequence: AtomicU64,

    pub(crate) compaction_priority: RwLock<Option<CompactionPriority>>,

    pub(crate) parent_snap: RwLock<Option<pb::Snapshot>>,

    pub(crate) encryption_key: Option<EncryptionKey>,
}

// Note: when add new property, consider whether to add it to following process:
// * `PropertiesHelper`
// * `is_property_need_flush`
// * `is_property_change_set`
pub const INGEST_ID_KEY: &str = "_ingest_id";
pub const DEL_PREFIXES_KEY: &str = "_del_prefixes";
pub const TRUNCATE_TS_KEY: &str = "_truncate_ts";
pub const ENCRYPTION_KEY: &str = "_encryption";

pub const TRIM_OVER_BOUND: &str = "_trim_over_bound";
pub const TRIM_OVER_BOUND_ENABLE: &[u8] = &[1];
pub const TRIM_OVER_BOUND_DISABLE: &[u8] = b"";

pub const MANUAL_MAJOR_COMPACTION: &str = "_manual_major_compaction";
pub const MANUAL_MAJOR_COMPACTION_ENABLE: &[u8] = &[1];
pub const MANUAL_MAJOR_COMPACTION_DISABLE: &[u8] = b"";

/// To indicate whether the property should be flush in mem-table flush process.
///
/// Most properties are applied to `Shard.properties` first. Then flush to
/// `ShardMeta` and persist.
///
/// But some other properties are set in `ShardMeta` first, then set or recover
/// to `Shard`. If `flush` operation get properties from mem-table and set to
/// `ShardMeta` on `apply_flush`, it will cause inconsistency, as properties in
/// mem-table would be applied lately than in `ShardMeta`.
#[inline]
pub fn is_property_need_flush(key: &str) -> bool {
    !(key == DEL_PREFIXES_KEY || key == TRUNCATE_TS_KEY)
}

impl Deref for Shard {
    type Target = ShardRange;

    fn deref(&self) -> &Self::Target {
        &self.range
    }
}

impl Shard {
    pub fn new(
        engine_id: u64,
        props: &pb::Properties,
        ver: u64,
        range: ShardRange,
        opt: Arc<Options>,
        master_key: &MasterKey,
    ) -> Self {
        let encryption_key = get_shard_property(ENCRYPTION_KEY, props)
            .map(|v| master_key.decrypt_encryption_key(&v).unwrap());
        let shard = Self {
            engine_id,
            id: props.shard_id,
            ver,
            range: range.clone(),
            parent_id: 0,
            pending_ops: RwLock::new(ShardPendingOperations::new(range.inner_key_off)),
            data: RwLock::new(ShardData::new_empty(range)),
            opt,
            active: Default::default(),
            properties: Properties::new().apply_pb(props),
            compacting: Default::default(),
            initial_flushed: Default::default(),
            base_version: Default::default(),
            estimated_size: Default::default(),
            estimated_entries: Default::default(),
            max_ts: Default::default(),
            estimated_kv_size: Default::default(),
            sst_max_ts: Default::default(),
            tombs: Default::default(),
            entries_write_cf: Default::default(),
            meta_seq: Default::default(),
            write_sequence: Default::default(),
            compaction_priority: RwLock::new(None),
            parent_snap: RwLock::new(None),
            encryption_key,
        };
        {
            let mut pending_ops = shard.pending_ops.write().unwrap();
            if let Some(val) = get_shard_property(DEL_PREFIXES_KEY, props) {
                let mut del_prefixes = DeletePrefixes::unmarshal(&val, shard.inner_key_off);
                del_prefixes.schedule_at = shard.gen_rand_schedule_del_range_time();
                pending_ops.del_prefixes = Arc::new(del_prefixes);
            }
            if let Some(val) = get_shard_property(TRUNCATE_TS_KEY, props) {
                // when load shard from Meta, the value maybe empty.
                if !val.is_empty() {
                    pending_ops.truncate_ts = Some(TruncateTs::unmarshal(&val));
                }
            }
            if let Some(val) = get_shard_property(TRIM_OVER_BOUND, props) {
                if !val.is_empty() {
                    pending_ops.trim_over_bound = true;
                }
            }
            if let Some(val) = get_shard_property(MANUAL_MAJOR_COMPACTION, props) {
                if !val.is_empty() {
                    pending_ops.manual_major_compaction = true;
                }
            }
        }
        shard
    }

    pub fn new_for_ingest(
        engine_id: u64,
        cs: &pb::ChangeSet,
        opt: Arc<Options>,
        master_key: &MasterKey,
    ) -> Self {
        let snap = cs.get_snapshot();
        let range = ShardRange::from_snap(snap);
        let mut shard = Self::new(
            engine_id,
            snap.get_properties(),
            cs.shard_ver,
            range,
            opt,
            master_key,
        );
        if !cs.has_parent() {
            store_bool(&shard.initial_flushed, true);
        } else {
            shard.parent_id = cs.get_parent().shard_id;
            let mut guard = shard.parent_snap.write().unwrap();
            *guard = Some(cs.get_parent().get_snapshot().clone());
        }
        shard.base_version.store(snap.base_version, Release);
        shard.meta_seq.store(cs.sequence, Release);
        shard.write_sequence.store(snap.data_sequence, Release);
        shard
    }

    pub fn id_ver(&self) -> IdVer {
        IdVer::new(self.id, self.ver)
    }

    pub(crate) fn set_active(&self, active: bool) {
        self.active.store(active, Release);
    }

    pub fn is_active(&self) -> bool {
        self.active.load(Acquire)
    }

    pub(crate) fn refresh_states(&self) {
        self.refresh_estimated_size_and_entries();
        self.refresh_compaction_priority();
    }

    fn refresh_estimated_size_and_entries(&self) {
        let data = self.get_data();

        let mut lv_stats = data.get_l0_stats();
        data.for_each_level(|cf, l| {
            lv_stats.add(&data.get_level_stats(cf, l), cf);
            false
        });

        store_u64(
            &self.estimated_size,
            lv_stats.data_size + lv_stats.blob_size,
        );
        store_u64(&self.estimated_entries, lv_stats.entries);
        store_u64(&self.sst_max_ts, lv_stats.max_ts);
        store_u64(
            &self.max_ts,
            std::cmp::max(lv_stats.max_ts, data.get_mem_table_max_ts()),
        );
        store_u64(&self.estimated_kv_size, lv_stats.kv_size);
        store_u64(&self.tombs, lv_stats.tombs);
        store_u64(&self.entries_write_cf, lv_stats.entries_write_cf);
    }

    pub(crate) fn gen_rand_schedule_del_range_time(&self) -> u64 {
        let now = time::precise_time_ns();
        let delay = rand::thread_rng().gen_range(0..self.opt.max_del_range_delay.as_nanos() as u64);
        now + delay
    }

    pub(crate) fn merge_del_prefix(&self, val: &[u8]) {
        info!("{} Shard::merge_del_prefix: {:?}", self.tag(), val);
        let mut pending_ops = self.pending_ops.write().unwrap();
        let mut del_prefixes = (*pending_ops.del_prefixes).merge_prefix(val);
        del_prefixes.schedule_at = self.gen_rand_schedule_del_range_time();
        pending_ops.del_prefixes = Arc::new(del_prefixes);
    }

    pub(crate) fn set_del_prefixes(&self, val: &[u8]) {
        info!("{} Shard::set_del_prefixes: {:?}", self.tag(), val);
        let mut pending_ops = self.pending_ops.write().unwrap();
        if val.is_empty() {
            pending_ops.del_prefixes =
                Arc::new(DeletePrefixes::new_with_inner_key_off(self.inner_key_off));
            return;
        }
        let mut del_prefixes = DeletePrefixes::unmarshal(val, self.inner_key_off);
        del_prefixes.schedule_at = self.gen_rand_schedule_del_range_time();
        pending_ops.del_prefixes = Arc::new(del_prefixes);
    }

    pub(crate) fn set_truncate_ts(&self, val: &[u8]) -> bool {
        let mut pending_ops = self.pending_ops.write().unwrap();
        let truncate_ts = TruncateTs::unmarshal(val);
        if let Some(curr_truncate_ts) = pending_ops.truncate_ts {
            if curr_truncate_ts <= truncate_ts {
                warn!("ignore another PiTR before the current one has completed"; "current truncated_ts" => ?curr_truncate_ts, "incoming truncated_ts" => ?truncate_ts, "shard" => ?self.tag());
                return false;
            } else {
                warn!("overwrite truncate_ts";"current truncated_ts" => ?curr_truncate_ts, "incoming truncated_ts" => ?truncate_ts, "shard" => ?self.tag());
                pending_ops.truncate_ts = Some(truncate_ts);
            }
        } else {
            pending_ops.truncate_ts = Some(truncate_ts);
        }
        true
    }

    pub(crate) fn set_trim_over_bound(&self, val: &[u8]) -> bool {
        let mut pending_ops = self.pending_ops.write().unwrap();
        if !val.is_empty() {
            pending_ops.trim_over_bound = true;
            return true;
        }
        false
    }

    pub(crate) fn set_manual_major_compaction(&self, val: &[u8]) -> bool {
        let mut pending_ops = self.pending_ops.write().unwrap();
        if !val.is_empty() {
            pending_ops.manual_major_compaction = true;
            return true;
        }
        pending_ops.manual_major_compaction = false;
        false
    }

    /// Get suggest key for region split.
    pub fn get_suggest_split_key(&self) -> Option<Bytes> {
        let data = self.get_data();
        let (max_level, left, right) = data.get_max_level_by_overlapping_tables_count()?;
        let tables = &max_level.tables.as_slice()[left..right];
        if tables.is_empty() {
            return None;
        }
        // Split on block boundaries only when there is one table.
        // As split on table boundaries would have lower cost for trim over bound data.
        if tables.len() == 1 {
            if let Some(inner_key) =
                tables[0].get_suggest_split_key(Some(data.inner_start()), Some(data.inner_end()))
            {
                return Some(
                    [self.key_prefix(), inner_key.to_vec().as_slice()]
                        .concat()
                        .into(),
                );
            }
            return None;
        }
        let tbl_idx = tables.len() * 2 / 3;
        Some(
            [self.key_prefix(), tables[tbl_idx].smallest().deref()]
                .concat()
                .into(),
        )
    }

    /// Get evenly split keys, mainly for acquiring bucket keys.
    pub fn get_evenly_split_keys(&self, count: usize) -> Option<Vec<Bytes>> {
        if count <= 1 {
            return None;
        }

        // Get tables in shard range.
        let data = self.get_data();
        let (max_level, left, right) = data.get_max_level_by_overlapping_tables_count()?;
        let tables = &max_level.tables.as_slice()[left..right];
        if tables.is_empty() {
            return None;
        }

        // Split at block boundaries when there are not enough tables.
        if tables.len() < count {
            let mut inner_keys = Vec::with_capacity(count);
            let counts_in_table = evenly_distribute(count, tables.len());
            debug_assert_eq!(counts_in_table.len(), tables.len());
            for (i, (tbl, count_in_table)) in tables.iter().zip(counts_in_table).enumerate() {
                let start = (i == 0).then_some(data.inner_start());
                let end = (i == tables.len() - 1).then_some(data.inner_end());
                inner_keys.extend(tbl.get_evenly_split_keys(start, end, count_in_table));
            }
            return Some(
                inner_keys
                    .into_iter()
                    .skip(1)
                    .map(|inner_key| [self.key_prefix(), inner_key.deref()].concat().into())
                    .collect(),
            );
        }

        // Split at table boundaries.
        let steps = evenly_distribute(tables.len(), count);
        let mut split_keys = Vec::with_capacity(count - 1);
        let mut tbl_idx = steps[0];
        for step in steps.into_iter().skip(1) {
            let tbl = &tables[tbl_idx];
            let split_key = [self.key_prefix(), tbl.smallest().deref()].concat().into();
            split_keys.push(split_key);

            tbl_idx += step;
        }
        Some(split_keys)
    }

    pub(crate) fn overlap_table(&self, smallest: InnerKey<'_>, biggest: InnerKey<'_>) -> bool {
        self.inner_start() <= biggest && smallest < self.inner_end()
    }

    pub(crate) fn cover_full_table(&self, smallest: InnerKey<'_>, biggest: InnerKey<'_>) -> bool {
        self.inner_start() <= smallest && biggest < self.inner_end()
    }

    #[cfg(test)]
    pub(crate) fn overlap_key(&self, inner_key: InnerKey<'_>) -> bool {
        self.inner_start() <= inner_key && inner_key < self.inner_end()
    }

    pub fn get_property(&self, key: &str) -> Option<Bytes> {
        self.properties.get(key)
    }

    pub fn set_property(&self, key: &str, val: &[u8]) {
        self.properties.set(key, val);
        // sync shard pending ops when set shard property.
        match key {
            DEL_PREFIXES_KEY => {
                self.set_del_prefixes(val);
            }
            TRUNCATE_TS_KEY => {
                let mut pending_ops = self.pending_ops.write().unwrap();
                if !val.is_empty() {
                    pending_ops.truncate_ts = Some(TruncateTs::unmarshal(val));
                } else {
                    pending_ops.truncate_ts = None;
                }
            }
            TRIM_OVER_BOUND => {
                let mut pending_ops = self.pending_ops.write().unwrap();
                if !val.is_empty() {
                    pending_ops.trim_over_bound = true;
                } else {
                    pending_ops.trim_over_bound = false;
                }
            }
            MANUAL_MAJOR_COMPACTION => {
                let mut pending_ops = self.pending_ops.write().unwrap();
                if !val.is_empty() {
                    pending_ops.manual_major_compaction = true;
                } else {
                    pending_ops.manual_major_compaction = false;
                }
            }
            _ => {}
        }
    }

    pub(crate) fn load_mem_table_version(&self) -> u64 {
        self.get_base_version() + self.write_sequence.load(Acquire)
    }

    pub fn get_all_files(&self) -> Vec<u64> {
        let data = self.get_data();
        data.get_all_files()
    }

    pub(crate) fn split_mem_tables(&self, parent_mem_tbls: &[CfTable]) -> Vec<CfTable> {
        let mut new_mem_tbls = vec![CfTable::new()];
        for old_mem_tbl in parent_mem_tbls {
            if old_mem_tbl.has_data_in_range(self.inner_start(), self.inner_end()) {
                new_mem_tbls.push(old_mem_tbl.new_split());
            }
        }
        new_mem_tbls
    }

    pub fn get_base_version(&self) -> u64 {
        self.base_version.load(Ordering::Acquire)
    }

    pub fn get_write_sequence(&self) -> u64 {
        self.write_sequence.load(Ordering::Acquire)
    }

    pub fn get_meta_sequence(&self) -> u64 {
        self.meta_seq.load(Ordering::Acquire)
    }

    pub fn get_estimated_size(&self) -> u64 {
        self.estimated_size.load(Ordering::Relaxed)
    }

    pub fn get_estimated_entries(&self) -> u64 {
        self.estimated_entries.load(Ordering::Relaxed)
    }

    // Note: `get_max_ts` would be not up-to-date.
    // Invoke `Shard::refresh_states` to refresh it.
    pub fn get_max_ts(&self) -> u64 {
        cmp::max(
            self.max_ts.load(Ordering::Relaxed),
            self.get_data().get_mem_table_max_ts(),
        )
    }

    pub fn get_estimated_kv_size(&self) -> u64 {
        self.estimated_kv_size.load(Ordering::Relaxed)
    }

    pub fn get_initial_flushed(&self) -> bool {
        self.initial_flushed.load(Acquire)
    }

    pub(crate) fn ready_to_destroy_range(
        del_prefixes: &DeletePrefixes,
        shard_data: &ShardData,
    ) -> bool {
        !del_prefixes.is_empty() && del_prefixes.after_scheduled_time()
        // No memtable contains data covered by deleted prefixes.
        && !shard_data.mem_tbls.iter().any(|mem_tbl| {
            del_prefixes
                .inner_delete_ranges()
                .any(|(start, end)| mem_tbl.has_data_in_range(start, end))
        })
    }

    fn ready_to_truncate_ts(truncate_ts: &Option<TruncateTs>, shard_data: &ShardData) -> bool {
        truncate_ts.is_some()
        // No memtable contains data with version > truncate_ts.
        && !shard_data.mem_tbls.iter().any(|mem_tbl| {
            mem_tbl.data_max_ts() > truncate_ts.unwrap().inner()
        })
    }

    fn ready_to_trim_over_bound(trim_over_bound: bool, shard_data: &ShardData) -> bool {
        trim_over_bound && !shard_data.has_mem_over_bound_data()
    }

    fn refresh_compaction_priority(&self) {
        let data = self.get_data();
        let pending_ops = self.pending_ops.read().unwrap();

        if Self::ready_to_destroy_range(&pending_ops.del_prefixes, &data) {
            *self.compaction_priority.write().unwrap() = Some(CompactionPriority::DestroyRange);
            return;
        } else if Self::ready_to_truncate_ts(&pending_ops.truncate_ts, &data) {
            *self.compaction_priority.write().unwrap() = Some(CompactionPriority::TruncateTs);
            return;
        } else if Self::ready_to_trim_over_bound(pending_ops.trim_over_bound, &data) {
            *self.compaction_priority.write().unwrap() = Some(CompactionPriority::TrimOverBound);
            return;
        } else if pending_ops.manual_major_compaction {
            // Set major compaction priority to 2.0 to make it less likely to be picked up
            // when there are other shards waiting to be compacted.
            *self.compaction_priority.write().unwrap() =
                Some(CompactionPriority::Major { score: 2.0 });
            return;
        }
        if !data.blob_tbl_map.is_empty() {
            let blob_table_utilization = {
                let mut in_use_blob_size = 0;
                let mut total_blob_size = 0;
                for l0 in &data.l0_tbls {
                    in_use_blob_size += l0.total_blob_size();
                }
                for cf in &data.cfs {
                    for lh in &cf.levels {
                        for sst in &(*lh.tables) {
                            in_use_blob_size += sst.total_blob_size();
                        }
                    }
                }

                for blob_table in data.blob_tbl_map.values() {
                    total_blob_size += blob_table.total_blob_size();
                }
                if total_blob_size == 0 {
                    1.0
                } else {
                    in_use_blob_size as f64 / total_blob_size as f64
                }
            };
            if blob_table_utilization < self.opt.blob_table_gc_ratio {
                let mut lock = self.compaction_priority.write().unwrap();
                *lock = Some(CompactionPriority::Major { score: f64::MAX });
                return;
            }
        }
        let mut score = 0.0;
        let mut cf_with_highest_score = -1;
        let mut level_with_highest_score = 0;
        if data.l0_tbls.len() > 2 {
            let size_score = data.get_l0_total_size() as f64 / self.opt.base_size as f64;
            let num_tbl_score = data.l0_tbls.len() as f64 / 4.0;
            score = size_score * 0.6 + num_tbl_score * 0.4;
        }
        for l0 in &data.l0_tbls {
            if !data.cover_full_table(l0.smallest(), l0.biggest()) {
                // set highest priority for newly split L0.
                score += 2.0;
                let mut lock = self.compaction_priority.write().unwrap();
                *lock = Some(CompactionPriority::L0 { score });
                return;
            }
        }
        for cf in 0..NUM_CFS {
            let scf = data.get_cf(cf);
            for lh in &scf.levels[..scf.levels.len() - 1] {
                let lvl_stats = data.get_level_stats(cf, lh);
                let lvl_score = lvl_stats.data_size as f64
                    / ((self.opt.base_size as f64) * 10f64.powf((lh.level - 1) as f64));
                if score < lvl_score {
                    score = lvl_score;
                    level_with_highest_score = lh.level;
                    cf_with_highest_score = cf as isize;
                }
            }
        }
        let priority = if score > 1.0 {
            let max_pri = if level_with_highest_score == 0 {
                CompactionPriority::L0 { score }
            } else {
                CompactionPriority::L1Plus {
                    cf: cf_with_highest_score,
                    level: level_with_highest_score,
                    score,
                }
            };
            Some(max_pri)
        } else {
            let handle_priority_none = || -> Option<CompactionPriority> {
                if !data.l0_tbls.is_empty() {
                    // Trigger L0 compaction for test purpose.
                    fail::fail_point!("refresh_compaction_priority_for_l0", |_| {
                        Some(CompactionPriority::L0 { score: 2.0 })
                    });
                }
                None
            };
            handle_priority_none()
        };
        let mut lock = self.compaction_priority.write().unwrap();
        *lock = priority;
    }

    pub(crate) fn get_compaction_priority(&self) -> Option<CompactionPriority> {
        self.compaction_priority.read().unwrap().clone()
    }

    pub(crate) fn is_compacting(&self) -> bool {
        load_bool(&self.compacting)
    }

    /// Check whether to trigger compaction for tombstones.
    ///
    /// Due to high costs of major compaction, the current strategies are
    /// relatively conservative.
    pub fn check_need_gc_tombstones(&self, safe_ts: u64) -> bool {
        if self.is_compacting()
            || self.get_compaction_priority().is_some()
            || self.pending_ops.read().unwrap().manual_major_compaction
        {
            return false;
        }

        let sst_max_ts = self.sst_max_ts.load(Ordering::Relaxed);
        if sst_max_ts > safe_ts {
            return false;
        }

        let tombs = self.tombs.load(Ordering::Relaxed);
        if tombs < self.opt.compaction_tombs_count {
            return false;
        }

        let write_entries = self.entries_write_cf.load(Ordering::Relaxed);
        let tombs_ratio = tombs as f64 / write_entries as f64;
        if tombs_ratio >= self.opt.compaction_tombs_ratio {
            let mut priority = self.compaction_priority.write().unwrap();
            if priority.is_some() {
                return false;
            }
            *priority = Some(CompactionPriority::Major { score: tombs_ratio });
            return true;
        }

        false
    }

    pub(crate) fn get_data(&self) -> ShardData {
        self.data.read().unwrap().clone()
    }

    pub(crate) fn set_data(&self, data: ShardData) {
        let mut guard = self.data.write().unwrap();
        *guard = data
    }

    pub fn new_snap_access(&self) -> SnapAccess {
        SnapAccess::new(self)
    }

    pub fn get_writable_mem_table_size(&self) -> u64 {
        let guard = self.data.read().unwrap();
        guard.mem_tbls[0].size()
    }

    pub fn has_over_bound_data(&self) -> bool {
        self.get_data().has_over_bound_data()
    }

    pub fn get_trim_over_bound(&self) -> bool {
        self.pending_ops.read().unwrap().trim_over_bound
    }

    pub fn get_del_prefixes(&self) -> Arc<DeletePrefixes> {
        self.pending_ops.read().unwrap().del_prefixes.clone()
    }

    pub fn get_truncate_ts(&self) -> Option<TruncateTs> {
        self.pending_ops.read().unwrap().truncate_ts
    }

    pub fn get_manual_major_compaction(&self) -> bool {
        self.pending_ops.read().unwrap().manual_major_compaction
    }

    pub fn data_all_persisted(&self) -> bool {
        self.data.read().unwrap().all_presisted()
    }

    pub fn tag(&self) -> ShardTag {
        ShardTag::new(self.engine_id, IdVer::new(self.id, self.ver))
    }

    pub(crate) fn ready_to_compact(&self) -> bool {
        self.is_active() && self.get_initial_flushed()
    }

    pub(crate) fn add_parent_mem_tbls(&self, parent: Arc<Shard>) {
        let shard_data = self.get_data();
        let parent_data = parent.get_data();
        let mem_tbls = self.split_mem_tables(&parent_data.mem_tbls);
        let mem_tbl_vers: Vec<u64> = mem_tbls.iter().map(|tbl| tbl.get_version()).collect();
        info!("{} add parent mem-tables {:?}", self.tag(), mem_tbl_vers);
        let new_data = ShardData::new(
            shard_data.range.clone(),
            mem_tbls,
            shard_data.l0_tbls.clone(),
            shard_data.blob_tbl_map.clone(),
            shard_data.cfs.clone(),
            shard_data.unloaded_tbls.clone(),
        );
        self.set_data(new_data);
    }

    pub(crate) fn inner_start(&self) -> InnerKey<'_> {
        InnerKey::from_outer_key(&self.outer_start, self.inner_key_off)
    }

    pub(crate) fn inner_end(&self) -> InnerKey<'_> {
        InnerKey::from_outer_end_key(&self.outer_end, self.inner_key_off)
    }

    pub(crate) fn key_prefix(&self) -> &[u8] {
        if self.inner_key_off > 0 {
            return &self.outer_start[0..self.inner_key_off];
        }
        &[]
    }

    /// Whether the shard is empty.
    ///
    /// NOTE: use with caution.
    /// The empty status of shard is not reliable, as the applying to kvengine
    /// would be late.
    pub(crate) fn is_empty(&self) -> bool {
        let data = self.get_data();
        if data.mem_tbls.iter().any(|mem_tbl| !mem_tbl.is_empty()) {
            return false;
        }
        self.get_estimated_size() == 0
    }
}

#[derive(Clone)]
pub(crate) struct ShardData {
    pub(crate) core: Arc<ShardDataCore>,
}

impl Deref for ShardData {
    type Target = ShardDataCore;

    fn deref(&self) -> &Self::Target {
        &self.core
    }
}

impl ShardData {
    pub(crate) fn new_empty(shard_range: ShardRange) -> Self {
        Self::new(
            shard_range,
            vec![CfTable::new()],
            vec![],
            Arc::new(HashMap::new()),
            [ShardCf::new(0), ShardCf::new(1), ShardCf::new(2)],
            HashMap::new(),
        )
    }

    pub(crate) fn new(
        range: ShardRange,
        mem_tbls: Vec<memtable::CfTable>,
        l0_tbls: Vec<L0Table>,
        blob_tbl_map: Arc<HashMap<u64, BlobTable>>,
        cfs: [ShardCf; 3],
        unloaded_tbls: HashMap<u64, FileMeta>,
    ) -> Self {
        assert!(!mem_tbls.is_empty());

        Self {
            core: Arc::new(ShardDataCore {
                range,
                mem_tbls,
                l0_tbls,
                blob_tbl_map,
                cfs,
                unloaded_tbls,
            }),
        }
    }
}

pub(crate) struct ShardDataCore {
    pub(crate) range: ShardRange,
    pub(crate) mem_tbls: Vec<memtable::CfTable>,
    pub(crate) l0_tbls: Vec<L0Table>,
    pub(crate) blob_tbl_map: Arc<HashMap<u64, BlobTable>>,
    pub(crate) cfs: [ShardCf; 3],
    /// Tables that are not loaded from DFS yet.
    pub(crate) unloaded_tbls: HashMap<u64, FileMeta>,
}

impl Deref for ShardDataCore {
    type Target = ShardRange;

    fn deref(&self) -> &Self::Target {
        &self.range
    }
}

impl ShardDataCore {
    pub fn get_writable_mem_table(&self) -> &memtable::CfTable {
        self.mem_tbls.first().unwrap()
    }

    pub(crate) fn get_cf(&self, cf: usize) -> &ShardCf {
        &self.cfs[cf]
    }

    pub(crate) fn get_all_files(&self) -> Vec<u64> {
        let mut files = Vec::new();
        for l0 in &self.l0_tbls {
            files.push(l0.id());
        }
        for blob_tbl_id in self.blob_tbl_map.keys() {
            files.push(*blob_tbl_id);
        }
        self.for_each_level(|_cf, lh| {
            for tbl in lh.tables.iter() {
                files.push(tbl.id())
            }
            false
        });
        files.sort_unstable();
        files
    }

    pub(crate) fn for_each_level<F>(&self, mut f: F)
    where
        F: FnMut(usize /* cf */, &LevelHandler) -> bool, // stopped
    {
        for cf in 0..NUM_CFS {
            let scf = self.get_cf(cf);
            for lh in scf.levels.as_slice() {
                if f(cf, lh) {
                    return;
                }
            }
        }
    }

    // Return (max_ts).
    pub(crate) fn get_mem_table_max_ts(&self) -> u64 {
        let mut max_ts = 0;
        self.mem_tbls.iter().for_each(|t| {
            max_ts = cmp::max(max_ts, t.data_max_ts());
        });
        max_ts
    }

    pub(crate) fn get_l0_total_size(&self) -> u64 {
        self.get_l0_stats().data_size
    }

    pub(crate) fn get_l0_stats(&self) -> LevelStatsLite {
        let mut stats = LevelStatsLite::default();
        self.l0_tbls.iter().for_each(|l0| {
            if self.cover_full_table(l0.smallest(), l0.biggest()) {
                stats.data_size += l0.size();
                stats.entries += l0.entries();
                stats.kv_size += l0.kv_size();
                stats.tombs += l0.tombs();
                stats.entries_write_cf += l0.entries_write_cf();
            } else {
                stats.data_size += l0.size() / 2;
                stats.entries += l0.entries() / 2;
                stats.kv_size += l0.kv_size() / 2;
                stats.tombs += l0.tombs() / 2;
                stats.entries_write_cf += l0.entries_write_cf() / 2;
            }
            stats.max_ts = cmp::max(stats.max_ts, l0.max_ts());
        });
        stats
    }

    pub(crate) fn get_level_stats(&self, cf: usize, level: &LevelHandler) -> LevelStatsLite {
        let mut stats = LevelStatsLite::default();
        level.tables.iter().enumerate().for_each(|(i, tbl)| {
            if self.is_over_bound_table(level, i, tbl) {
                stats.data_size +=
                    tbl.estimated_size_in_range(self.inner_start(), self.inner_end());
                stats.blob_size += tbl.in_use_total_blob_size / 2;
                stats.entries += tbl.entries as u64 / 2;
                if cf == WRITE_CF {
                    stats.kv_size += tbl.kv_size / 2;
                    stats.tombs += tbl.tombs as u64 / 2;
                    stats.entries_write_cf += tbl.entries as u64 / 2;
                }
            } else {
                stats.data_size += tbl.size();
                stats.blob_size += tbl.in_use_total_blob_size;
                stats.entries += tbl.entries as u64;
                if cf == WRITE_CF {
                    stats.kv_size += tbl.kv_size;
                    stats.tombs += tbl.tombs as u64;
                    stats.entries_write_cf += tbl.entries as u64;
                }
            }
            stats.max_ts = cmp::max(stats.max_ts, tbl.max_ts);
        });
        stats
    }

    fn is_over_bound_table(&self, level: &LevelHandler, i: usize, tbl: &SsTable) -> bool {
        let is_bound = i == 0 || i == level.tables.len() - 1;
        is_bound && !self.cover_full_table(tbl.smallest(), tbl.biggest())
    }

    pub(crate) fn cover_full_table(&self, smallest: InnerKey<'_>, biggest: InnerKey<'_>) -> bool {
        self.inner_start() <= smallest && biggest < self.inner_end()
    }

    pub fn all_presisted(&self) -> bool {
        self.mem_tbls.len() == 1 && self.mem_tbls[0].size() == 0
    }

    pub(crate) fn has_mem_over_bound_data(&self) -> bool {
        for mem_tbl in &self.mem_tbls {
            for cf in 0..NUM_CFS {
                let skl = mem_tbl.get_cf(cf);
                let mut iter = skl.new_iterator(false);
                iter.rewind();
                if iter.valid() && iter.key() < self.inner_start() {
                    return true;
                }
                let mut rev_iter = skl.new_iterator(true);
                rev_iter.rewind();
                if rev_iter.valid() && rev_iter.key() >= self.inner_end() {
                    return true;
                }
            }
        }
        false
    }

    pub(crate) fn has_file_over_bound_data(&self) -> bool {
        for l0 in &self.l0_tbls {
            if l0.smallest() < self.inner_start() || l0.biggest() >= self.inner_end() {
                return true;
            }
        }
        for cf in 0..NUM_CFS {
            let scf = &self.cfs[cf];
            for level in &scf.levels {
                if level.has_over_bound_data(self.inner_start(), self.inner_end()) {
                    return true;
                }
            }
        }
        false
    }

    pub fn has_over_bound_data(&self) -> bool {
        self.has_mem_over_bound_data() || self.has_file_over_bound_data()
    }

    /// Get max level by count of overlapping tables.
    /// Return level and range of the overlapping tables.
    pub fn get_max_level_by_overlapping_tables_count(
        &self,
    ) -> Option<(
        &LevelHandler,
        usize, // left table index in shard range
        usize, // exclusive right table index in shard range
    )> {
        self.get_cf(0)
            .levels
            .iter()
            .map(|level| {
                let (left, right) =
                    level.overlapping_tables_exclusive_end(self.inner_start(), self.inner_end());
                (level, left, right)
            })
            .max_by_key(|(_, left, right)| right - left)
    }
}

pub fn store_u64(ptr: &AtomicU64, val: u64) {
    ptr.store(val, Release);
}

pub fn load_u64(ptr: &AtomicU64) -> u64 {
    ptr.load(Acquire)
}

pub fn store_bool(ptr: &AtomicBool, val: bool) {
    ptr.store(val, Release)
}

pub fn load_bool(ptr: &AtomicBool) -> bool {
    ptr.load(Acquire)
}

pub(crate) struct ShardCfBuilder {
    levels: Vec<LevelHandlerBuilder>,
}

impl ShardCfBuilder {
    pub(crate) fn new(cf: usize) -> Self {
        Self {
            levels: vec![LevelHandlerBuilder::new(); CF_LEVELS[cf]],
        }
    }

    pub(crate) fn build(&mut self) -> ShardCf {
        let mut levels = Vec::with_capacity(self.levels.len());
        for i in 0..self.levels.len() {
            levels.push(self.levels[i].build(i + 1))
        }
        ShardCf { levels }
    }

    pub(crate) fn add_table(&mut self, tbl: SsTable, level: usize) {
        self.levels[level - 1].add_table(tbl)
    }
}

#[derive(Clone)]
struct LevelHandlerBuilder {
    tables: Option<Vec<SsTable>>,
}

impl LevelHandlerBuilder {
    fn new() -> Self {
        Self {
            tables: Some(vec![]),
        }
    }

    fn build(&mut self, level: usize) -> LevelHandler {
        let mut tables = self.tables.take().unwrap();
        tables.sort_by(|a, b| a.smallest().cmp(&b.smallest()));
        LevelHandler::new(level, tables)
    }

    fn add_table(&mut self, tbl: SsTable) {
        if self.tables.is_none() {
            self.tables = Some(vec![])
        }
        self.tables.as_mut().unwrap().push(tbl)
    }
}

#[derive(Clone)]
pub(crate) struct ShardCf {
    pub(crate) levels: Vec<LevelHandler>,
}

impl ShardCf {
    pub(crate) fn new(cf: usize) -> Self {
        let mut levels = vec![];
        for j in 1..=CF_LEVELS[cf] {
            levels.push(LevelHandler::new(j, vec![]));
        }
        Self { levels }
    }

    pub(crate) fn get_level(&self, level: usize) -> &LevelHandler {
        &self.levels[level - 1]
    }

    pub(crate) fn set_level(&mut self, level_handler: LevelHandler) {
        let level_idx = level_handler.level - 1;
        self.levels[level_idx] = level_handler
    }
}

#[derive(Default, Clone)]
pub struct LevelHandler {
    pub(crate) tables: Arc<Vec<SsTable>>,
    pub(crate) level: usize,
    pub(crate) max_ts: u64,
}

impl LevelHandler {
    pub fn new(level: usize, tables: Vec<SsTable>) -> Self {
        let mut max_ts = 0;
        for tbl in &tables {
            if max_ts < tbl.max_ts {
                max_ts = tbl.max_ts;
            }
        }
        Self {
            tables: Arc::new(tables),
            level,
            max_ts,
        }
    }

    pub(crate) fn overlapping_tables(&self, key_range: &KeyRange) -> (usize, usize) {
        get_tables_in_range(&self.tables, key_range.left_key(), key_range.right_key())
    }

    pub(crate) fn overlapping_tables_exclusive_end(
        &self,
        start: InnerKey<'_>,
        exclusive_end: InnerKey<'_>,
    ) -> (usize, usize) {
        let left = search(self.tables.len(), |i| start <= self.tables[i].biggest());
        let right = search(self.tables.len(), |i| {
            exclusive_end <= self.tables[i].smallest()
        });
        (left, right)
    }

    pub fn get(
        &self,
        key: InnerKey<'_>,
        version: u64,
        key_hash: u64,
        out_val_owner: &mut Vec<u8>,
    ) -> table::Value {
        self.get_in_table(key, version, key_hash, self.get_table(key), out_val_owner)
    }

    fn get_in_table(
        &self,
        key: InnerKey<'_>,
        version: u64,
        key_hash: u64,
        tbl: Option<&SsTable>,
        out_val_owner: &mut Vec<u8>,
    ) -> table::Value {
        if tbl.is_none() {
            return table::Value::new();
        }
        tbl.unwrap()
            .get(key, version, key_hash, out_val_owner, self.level)
    }

    // Note: the `key` may be on the left outside the returned sstable.
    pub(crate) fn get_table(&self, key: InnerKey<'_>) -> Option<&SsTable> {
        let idx = search(self.tables.len(), |i| self.tables[i].biggest() >= key);
        if idx >= self.tables.len() {
            return None;
        }
        Some(&self.tables[idx])
    }

    pub(crate) fn get_table_by_id(&self, id: u64) -> Option<SsTable> {
        for tbl in self.tables.as_slice() {
            if tbl.id() == id {
                return Some(tbl.clone());
            }
        }
        None
    }

    pub(crate) fn check_order(&self, cf: usize, tag: ShardTag) {
        let tables = &self.tables;
        if tables.len() <= 1 {
            return;
        }
        for i in 0..(tables.len() - 1) {
            let ti = &tables[i];
            let tj = &tables[i + 1];
            if ti.smallest() > ti.biggest()
                || ti.smallest() >= tj.smallest()
                || ti.biggest() >= tj.biggest()
            {
                panic!(
                    "[{}] check table fail table ti {} smallest {:?}, biggest {:?}, tj {} smallest {:?}, biggest {:?}, cf: {}, level: {}",
                    tag,
                    ti.id(),
                    ti.smallest(),
                    ti.biggest(),
                    tj.id(),
                    tj.smallest(),
                    tj.biggest(),
                    cf,
                    self.level,
                );
            }
        }
    }

    pub(crate) fn get_newer(
        &self,
        key: InnerKey<'_>,
        version: u64,
        key_hash: u64,
        out_val_owner: &mut Vec<u8>,
    ) -> table::Value {
        if self.max_ts < version {
            return table::Value::new();
        }
        if let Some(tbl) = self.get_table(key) {
            return tbl.get_newer(key, version, key_hash, out_val_owner, self.level);
        }
        table::Value::new()
    }

    pub(crate) fn has_over_bound_data(&self, start: InnerKey<'_>, end: InnerKey<'_>) -> bool {
        if self.tables.is_empty() {
            return false;
        }
        let first = self.tables.first().unwrap();
        let last = self.tables.last().unwrap();
        first.smallest() < start || last.biggest() >= end
    }

    pub(crate) fn range_blocks(&self, ranges: &[(Bytes, Bytes)], inner_key_off: usize) -> usize {
        let mut num_blocks = 0;
        let mut prev_table_id = 0;
        let mut prev_block_right = 0;
        for ran in ranges {
            let start_key = InnerKey::from_outer_key(&ran.0, inner_key_off);
            let end_key = InnerKey::from_outer_end_key(&ran.1, inner_key_off);
            let (left, right) = self.overlapping_tables_exclusive_end(start_key, end_key);
            if left == right {
                continue;
            }
            let overlap_tables = &self.tables[left..right];
            let first_table = overlap_tables.first().unwrap();
            let first_idx = first_table.load_index();
            let first_block_left = first_idx.seek_block(start_key.as_ref()).saturating_sub(1);
            let first_block_right = if overlap_tables.len() == 1 {
                first_idx.seek_block_bigger_or_equal(end_key.as_ref())
            } else {
                first_idx.num_blocks()
            };
            num_blocks += first_block_right - first_block_left;
            if first_table.id() == prev_table_id && first_block_left < prev_block_right {
                // do not count duplicated block.
                num_blocks -= 1;
            }
            if overlap_tables.len() == 1 {
                prev_table_id = first_table.id();
                prev_block_right = first_block_right;
                continue;
            }
            if overlap_tables.len() > 2 {
                let middle_tables = &overlap_tables[1..overlap_tables.len() - 1];
                for middle_table in middle_tables {
                    num_blocks += middle_table.load_index().num_blocks();
                }
            }
            let last_table = overlap_tables.last().unwrap();
            let last_idx = last_table.load_index();
            let last_block_right = last_idx.seek_block_bigger_or_equal(end_key.as_ref());
            num_blocks += last_block_right;
            prev_table_id = last_table.id();
            prev_block_right = last_block_right;
        }
        num_blocks
    }
}

#[derive(Default, Clone)]
pub struct Properties {
    m: DashMap<String, Bytes>,
}

impl Properties {
    pub fn new() -> Self {
        Self {
            m: dashmap::DashMap::new(),
        }
    }

    pub fn set(&self, key: &str, val: &[u8]) {
        self.m.insert(key.to_string(), Bytes::copy_from_slice(val));
    }

    pub fn get(&self, key: &str) -> Option<Bytes> {
        let bin = self.m.get(key)?;
        Some(bin.value().clone())
    }

    pub fn remove(&self, key: &str) {
        self.m.remove(key);
    }

    pub fn to_pb(&self, shard_id: u64) -> kvenginepb::Properties {
        let mut props = kvenginepb::Properties::new();
        props.shard_id = shard_id;
        self.m.iter().for_each(|r| {
            props.keys.push(r.key().clone());
            props.values.push(r.value().to_vec());
        });
        props
    }

    pub fn apply_pb(self, props: &kvenginepb::Properties) -> Self {
        let keys = props.get_keys();
        let vals = props.get_values();
        for i in 0..keys.len() {
            let key = &keys[i];
            let val = &vals[i];
            self.set(key, val.as_slice());
        }
        self
    }
}

pub fn get_shard_property(key: &str, props: &kvenginepb::Properties) -> Option<Vec<u8>> {
    let keys = props.get_keys();
    for i in 0..keys.len() {
        if key == keys[i] {
            return Some(props.get_values()[i].clone());
        }
    }
    None
}

/// Returns the ingest id if it is a legacy ingest request from the TiDB BR
/// tool.
#[inline]
pub fn is_legacy_ingest(ingest_files: &kvenginepb::IngestFiles) -> Option<Vec<u8>> {
    get_shard_property(INGEST_ID_KEY, ingest_files.get_properties())
}

pub fn get_splitting_start_end<'a: 'b, 'b>(
    start: &'a [u8],
    end: &'a [u8],
    split_keys: &'b [Vec<u8>],
    i: usize,
) -> (&'b [u8], &'b [u8]) {
    let start_key = if i != 0 {
        split_keys[i - 1].as_slice()
    } else {
        start as &'b [u8]
    };
    let end_key = if i == split_keys.len() {
        end
    } else {
        split_keys[i].as_slice()
    };
    (start_key, end_key)
}

#[derive(Clone)]
pub struct DeletePrefixes {
    pub(crate) prefixes: Vec<Vec<u8>>,
    // prefixes_nexts are prefix-next keys of prefixes, i.e., [prefixes[i], prefixes_nexts[i]) is
    // the range that should be destroyed.
    prefixes_nexts: Vec<Vec<u8>>,
    pub(crate) schedule_at: u64,
    pub(crate) inner_key_off: usize,
}

impl std::fmt::Debug for DeletePrefixes {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeletePrefixes")
            .field(
                "prefixes",
                &self
                    .prefixes
                    .iter()
                    .map(|p| log_wrappers::Value::key(p))
                    .collect::<Vec<_>>(),
            )
            .field(
                "prefixes_nexts",
                &self
                    .prefixes_nexts
                    .iter()
                    .map(|p| log_wrappers::Value::key(p))
                    .collect::<Vec<_>>(),
            )
            .field("schedule_at", &self.schedule_at)
            .field("inner_key_off", &self.inner_key_off)
            .finish()
    }
}

impl DeletePrefixes {
    pub fn new_with_inner_key_off(inner_key_off: usize) -> Self {
        Self {
            prefixes: vec![],
            prefixes_nexts: vec![],
            schedule_at: 0,
            inner_key_off,
        }
    }
    fn gen_prefixes_nexts(prefixes: &[Vec<u8>]) -> Vec<Vec<u8>> {
        prefixes
            .iter()
            .map(|p| {
                let mut p_n = p.to_vec();
                tidb_query_common::util::convert_to_prefix_next(&mut p_n);
                p_n
            })
            .collect()
    }

    pub fn is_empty(&self) -> bool {
        self.prefixes.is_empty()
    }

    pub fn after_scheduled_time(&self) -> bool {
        time::precise_time_ns() > self.schedule_at
    }

    #[must_use]
    pub fn merge_prefix(&self, prefix: &[u8]) -> Self {
        assert!(prefix.len() >= self.inner_key_off);

        let mut new_prefixes = vec![];
        for old_prefix in &self.prefixes {
            if prefix.starts_with(old_prefix) {
                // old prefix covers new prefix, do not need to change.
                return self.clone();
            }
            if old_prefix.starts_with(prefix) {
                // new prefix covers old prefix. old prefix can be dropped.
                continue;
            }
            new_prefixes.push(old_prefix.clone());
        }
        new_prefixes.push(prefix.to_vec());
        new_prefixes.sort_unstable();
        let new_prefixes_nexts = DeletePrefixes::gen_prefixes_nexts(&new_prefixes);
        Self {
            prefixes: new_prefixes,
            prefixes_nexts: new_prefixes_nexts,
            schedule_at: 0,
            inner_key_off: self.inner_key_off,
        }
    }

    pub fn merge_prefix_in_place(&mut self, prefix: &[u8]) {
        let merged = self.merge_prefix(prefix);
        *self = merged;
    }

    /// Merges multiple prefixes & self into a new one.
    ///
    /// `sorted_prefixes` must be sorted.
    ///
    /// If `sorted_prefixes` contains a prefix that is covered by another prefix
    /// in it, the covered one will not be dropped.
    ///
    /// `schedule_at` & `inner_key_off` are copied from self.
    #[must_use]
    fn merge_prefixes<'a, S: AsRef<[u8]> + 'a>(
        &self,
        sorted_prefixes: impl Iterator<Item = S>,
    ) -> Self {
        let mut new_prefixes = Vec::with_capacity(self.prefixes.len());

        let mut first_iter = self.prefixes.iter();
        let mut first_opt = first_iter.next();
        let mut second_iter = sorted_prefixes;
        let mut second_opt = second_iter.next();

        while let (Some(first), Some(second)) = (first_opt, second_opt.as_ref()) {
            let second = second.as_ref();
            if first.as_slice().starts_with(second) {
                // `second` covers `first`
                first_opt = first_iter.next();
            } else if second.starts_with(first) {
                // `first` covers `second`
                second_opt = second_iter.next();
            } else {
                // `first` and `second` are not covered by each other
                if first.as_slice() < second {
                    new_prefixes.push(first.clone());
                    first_opt = first_iter.next();
                } else {
                    new_prefixes.push(second.to_vec());
                    second_opt = second_iter.next();
                }
            }
        }

        while let Some(first) = first_opt {
            new_prefixes.push(first.clone());
            first_opt = first_iter.next();
        }
        while let Some(second) = second_opt {
            new_prefixes.push(second.as_ref().to_vec());
            second_opt = second_iter.next();
        }

        let new_prefixes_nexts = DeletePrefixes::gen_prefixes_nexts(&new_prefixes);
        Self {
            prefixes: new_prefixes,
            prefixes_nexts: new_prefixes_nexts,
            schedule_at: self.schedule_at,
            inner_key_off: self.inner_key_off,
        }
    }

    /// Merge another `DeletePrefixes` into self.
    ///
    /// `schedule_at` & `inner_key_off` are unchanged.
    pub fn merge(&mut self, other: &Self) {
        let merged = self.merge_prefixes(other.prefixes.iter().map(|p| p.as_slice()));
        *self = merged;
    }

    /// Removes all prefixes in the `other` one.
    pub fn split(&self, other: &DeletePrefixes) -> Self {
        let mut new = self.clone();
        new.prefixes.retain(|p| !other.prefixes.contains(p));
        new.prefixes_nexts = DeletePrefixes::gen_prefixes_nexts(&new.prefixes);
        new
    }

    pub fn marshal(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        for prefix in &self.prefixes {
            buf.put_u16_le(prefix.len() as u16);
            buf.extend_from_slice(prefix);
        }
        buf
    }

    pub fn unmarshal(mut data: &[u8], inner_key_off: usize) -> Self {
        let mut prefixes = vec![];
        while !data.is_empty() {
            let len = data.get_u16_le() as usize;
            prefixes.push(data[..len].to_vec());
            data.advance(len);
        }
        let prefixes_nexts = DeletePrefixes::gen_prefixes_nexts(&prefixes);
        Self {
            prefixes,
            prefixes_nexts,
            schedule_at: 0,
            inner_key_off,
        }
    }

    pub fn build_split(&self, start: &[u8], end: &[u8], inner_key_off: usize) -> Self {
        let prefixes: Vec<_> = self
            .prefixes
            .iter()
            .filter(|prefix| {
                (start <= prefix.as_slice() && prefix.as_slice() < end) || start.starts_with(prefix)
            })
            .cloned()
            .collect();
        let prefixes_nexts = DeletePrefixes::gen_prefixes_nexts(&prefixes);
        let scheduled_at = self.schedule_at;
        Self {
            prefixes,
            prefixes_nexts,
            schedule_at: scheduled_at,
            inner_key_off,
        }
    }

    pub(crate) fn cover_full_keyspace(&self, keyspace_prefix: &[u8]) -> bool {
        let inner_key_off = self.inner_key_off;
        self.prefixes.iter().any(|p| {
            if inner_key_off >= p.len() {
                return true;
            }
            keyspace_prefix.starts_with(p.as_slice())
        })
    }

    pub(crate) fn cover_prefix(&self, inner_prefix: InnerKey<'_>) -> bool {
        let inner_key_off = self.inner_key_off;
        self.prefixes.iter().any(|p| {
            if inner_key_off >= p.len() {
                return true;
            }
            let inner_p = InnerKey::from_outer_key(p, inner_key_off);
            inner_prefix.starts_with(inner_p.deref())
        })
    }

    pub(crate) fn cover_range(&self, inner_start: InnerKey<'_>, inner_end: InnerKey<'_>) -> bool {
        let inner_key_off = self.inner_key_off;
        self.prefixes.iter().any(|p| {
            if inner_key_off >= p.len() {
                return true;
            }
            let inner_p = InnerKey::from_outer_key(p, inner_key_off);
            inner_start.starts_with(inner_p.deref()) && inner_end.starts_with(inner_p.deref())
        })
    }

    pub(crate) fn inner_delete_ranges(&self) -> impl Iterator<Item = (InnerKey<'_>, InnerKey<'_>)> {
        let inner_key_off = self.inner_key_off;
        self.prefixes
            .iter()
            .map(move |p| {
                if inner_key_off >= p.len() {
                    InnerKey::from_inner_buf(&[])
                } else {
                    InnerKey::from_outer_key(p, inner_key_off)
                }
            })
            .zip(self.prefixes_nexts.iter().map(move |p| {
                if inner_key_off >= p.len() {
                    InnerKey::from_inner_buf(GLOBAL_SHARD_END_KEY)
                } else {
                    InnerKey::from_outer_end_key(p, inner_key_off)
                }
            }))
    }

    /// Get ranges that are not covered by the delete prefixes.
    pub fn get_complementary_ranges(
        &self,
        outer_start: &[u8],
        outer_end: &[u8],
    ) -> Vec<(Vec<u8>, Vec<u8>)> {
        let mut ranges = Vec::with_capacity(self.prefixes.len() + 1);
        let mut start = outer_start.to_vec();
        for (prefix, prefix_next) in self.prefixes.iter().zip(self.prefixes_nexts.iter()) {
            if prefix_next.as_slice() <= outer_start {
                continue;
            }
            if prefix.as_slice() >= outer_end {
                break;
            }
            // `start` is possible to be equal to `prefix` here.
            // e.q. prefixes: [00001, 00002] will generate iterations (00001, 00002),
            // (00002, 00003).
            if &start < prefix {
                ranges.push((start.clone(), prefix.clone()));
            }
            start = prefix_next.clone();
        }
        if start.as_slice() < outer_end {
            ranges.push((start, outer_end.to_vec()));
        }
        ranges
    }
}

pub fn merge_del_prefixes_if_needed(
    first: Option<Bytes>,
    second: Option<Bytes>,
    inner_key_off: usize,
) -> Option<Vec<u8>> {
    if first.is_none() && second.is_none() {
        return None;
    }

    let mut first_prefixes = first
        .map(|b| DeletePrefixes::unmarshal(b.chunk(), inner_key_off))
        .unwrap_or_else(|| DeletePrefixes::new_with_inner_key_off(inner_key_off));
    let second_prefixes = second
        .map(|b| DeletePrefixes::unmarshal(b.chunk(), inner_key_off))
        .unwrap_or_else(|| DeletePrefixes::new_with_inner_key_off(inner_key_off));
    for prefix in &second_prefixes.prefixes {
        first_prefixes = first_prefixes.merge_prefix(prefix);
    }
    Some(first_prefixes.marshal())
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Ord, PartialOrd, Hash)]
#[repr(transparent)]
pub struct TruncateTs(u64);

impl TruncateTs {
    pub fn marshal(&self) -> [u8; 8] {
        self.0.to_le_bytes()
    }

    pub fn unmarshal(mut data: &[u8]) -> Self {
        assert_eq!(data.len(), U64_SIZE);
        Self(data.get_u64_le())
    }

    #[inline]
    pub fn inner(&self) -> u64 {
        self.0
    }
}

impl From<u64> for TruncateTs {
    fn from(ts: u64) -> Self {
        Self(ts)
    }
}

pub(crate) fn need_update_truncate_ts(cur: Option<TruncateTs>, new: TruncateTs) -> bool {
    if cur.is_none() {
        return true;
    }
    new <= cur.unwrap()
}

#[derive(Clone, Default, PartialEq)]
pub struct ShardRange {
    pub outer_start: Bytes,
    pub outer_end: Bytes,
    pub inner_key_off: usize,
    pub keyspace_id: u32,
}

impl ShardRange {
    pub fn new(outer_start: &[u8], outer_end: &[u8], inner_key_off: usize) -> Self {
        let keyspace_id = ApiV2::get_keyspace_prefix(outer_start)
            .map(|prefix| ApiV2::get_u32_keyspace_id(ApiV2::get_keyspace_id(prefix)))
            .unwrap_or_default();
        Self {
            outer_start: Bytes::from(outer_start.to_vec()),
            outer_end: Bytes::from(outer_end.to_vec()),
            inner_key_off,
            keyspace_id,
        }
    }

    pub(crate) fn from_snap(snap: &kvenginepb::Snapshot) -> Self {
        Self::new(
            snap.get_outer_start(),
            snap.get_outer_end(),
            snap.inner_key_off as usize,
        )
    }

    pub fn inner_start(&self) -> InnerKey<'_> {
        InnerKey::from_outer_key(&self.outer_start, self.inner_key_off)
    }

    pub fn inner_end(&self) -> InnerKey<'_> {
        InnerKey::from_outer_end_key(&self.outer_end, self.inner_key_off)
    }

    pub(crate) fn prefix(&self) -> &[u8] {
        &self.outer_start[..self.inner_key_off]
    }
}

impl fmt::Debug for ShardRange {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ShardRange")
            .field(
                "outer_start",
                &log_wrappers::hex_encode_upper(&self.outer_start),
            )
            .field(
                "outer_end",
                &log_wrappers::hex_encode_upper(&self.outer_end),
            )
            .field("inner_key_off", &self.inner_key_off)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_del_prefixes() {
        for inner_key_off in [0, 4] {
            let assert_prefix_invariant = |del_prefix: &DeletePrefixes| {
                assert_eq!(del_prefix.prefixes.len(), del_prefix.prefixes_nexts.len());
                assert_eq!(
                    del_prefix.prefixes.len(),
                    del_prefix.inner_delete_ranges().count()
                );
                assert!(del_prefix.inner_delete_ranges().all(|(p, p_n)| {
                    let mut p_c = p.to_vec();
                    tidb_query_common::util::convert_to_prefix_next(&mut p_c);
                    p_c == p_n.deref()
                }));
            };

            let mut del_prefix = DeletePrefixes::new_with_inner_key_off(inner_key_off);
            assert_prefix_invariant(&del_prefix);
            assert!(del_prefix.is_empty());
            del_prefix = del_prefix.merge_prefix("0000101".as_bytes());
            assert!(!del_prefix.is_empty());
            assert_eq!(del_prefix.prefixes.len(), 1);
            assert_prefix_invariant(&del_prefix);
            for prefix in ["00001010", "0000101"] {
                let inner_prefix = InnerKey::from_outer_key(prefix.as_bytes(), inner_key_off);
                assert!(del_prefix.cover_prefix(inner_prefix));
            }
            for prefix in ["000010", "0000103"] {
                let inner_prefix = InnerKey::from_outer_key(prefix.as_bytes(), inner_key_off);
                assert!(!del_prefix.cover_prefix(inner_prefix));
            }

            del_prefix = del_prefix.merge_prefix("0000105".as_bytes());
            assert_prefix_invariant(&del_prefix);
            let bin = del_prefix.marshal();
            del_prefix = DeletePrefixes::unmarshal(&bin, inner_key_off);
            assert_prefix_invariant(&del_prefix);
            assert_eq!(del_prefix.prefixes.len(), 2);
            for prefix in ["00001010", "0000101", "00001050", "0000105"] {
                let inner_prefix = InnerKey::from_outer_key(prefix.as_bytes(), inner_key_off);
                assert!(del_prefix.cover_prefix(inner_prefix));
            }
            for prefix in ["000010", "0000103"] {
                let inner_prefix = InnerKey::from_outer_key(prefix.as_bytes(), inner_key_off);
                assert!(!del_prefix.cover_prefix(inner_prefix));
            }

            del_prefix = del_prefix.merge_prefix("000010".as_bytes());
            assert_prefix_invariant(&del_prefix);
            assert_eq!(del_prefix.prefixes.len(), 1);
            for prefix in ["000010", "0000101", "0000103", "0000104"] {
                let inner_prefix = InnerKey::from_outer_key(prefix.as_bytes(), inner_key_off);
                assert!(del_prefix.cover_prefix(inner_prefix));
            }
            for prefix in ["00001", "000011"] {
                let inner_prefix = InnerKey::from_outer_key(prefix.as_bytes(), inner_key_off);
                assert!(!del_prefix.cover_prefix(inner_prefix));
            }

            del_prefix = DeletePrefixes::new_with_inner_key_off(inner_key_off);
            del_prefix = del_prefix.merge_prefix("0000101".as_bytes());
            del_prefix = del_prefix.merge_prefix("0000102".as_bytes());
            assert_prefix_invariant(&del_prefix);
            for (start, end) in [("0000101", "00001011"), ("0000102", "00001022")] {
                let inner_start = InnerKey::from_outer_key(start.as_bytes(), inner_key_off);
                let inner_end = InnerKey::from_outer_key(end.as_bytes(), inner_key_off);
                assert!(del_prefix.cover_range(inner_start, inner_end));
            }
            for (start, end) in [
                ("000099", "0000100"),
                ("0000101", "0000102"),
                ("0000102", "0000103"),
            ] {
                let inner_start = InnerKey::from_outer_key(start.as_bytes(), inner_key_off);
                let inner_end = InnerKey::from_outer_key(end.as_bytes(), inner_key_off);
                assert!(!del_prefix.cover_range(inner_start, inner_end));
            }

            del_prefix = DeletePrefixes::new_with_inner_key_off(inner_key_off);
            del_prefix = del_prefix.merge_prefix("0000101".as_bytes());
            del_prefix = del_prefix.merge_prefix("00001033".as_bytes());
            del_prefix = del_prefix.merge_prefix("00001055".as_bytes());
            del_prefix = del_prefix.merge_prefix("0000107".as_bytes());
            assert_prefix_invariant(&del_prefix);

            let split_del_range =
                del_prefix.build_split("00001033".as_bytes(), "00001055".as_bytes(), inner_key_off);
            assert_prefix_invariant(&split_del_range);
            assert_eq!(split_del_range.prefixes.len(), 1);
            assert_eq!(&split_del_range.prefixes[0], "00001033".as_bytes());

            let split_del_range =
                del_prefix.build_split("00001034".as_bytes(), "00001055".as_bytes(), inner_key_off);
            assert_prefix_invariant(&split_del_range);
            assert_eq!(split_del_range.prefixes.len(), 0);

            let split_del_range = del_prefix.build_split(
                "000010334".as_bytes(),
                "000010555".as_bytes(),
                inner_key_off,
            );
            assert_prefix_invariant(&split_del_range);
            assert_eq!(split_del_range.prefixes.len(), 2);
            assert_eq!(&split_del_range.prefixes[0], "00001033".as_bytes());
            assert_eq!(&split_del_range.prefixes[1], "00001055".as_bytes());

            del_prefix = DeletePrefixes::new_with_inner_key_off(inner_key_off)
                .merge_prefix("0000100".as_bytes())
                .merge_prefix("0000200".as_bytes())
                .merge_prefix("0000300".as_bytes());
            assert_prefix_invariant(&del_prefix);
            del_prefix = del_prefix.split(
                &DeletePrefixes::new_with_inner_key_off(inner_key_off)
                    .merge_prefix("0000100".as_bytes()),
            );
            assert_prefix_invariant(&del_prefix);
            assert_eq!(del_prefix.prefixes.len(), 2);
            assert!(!del_prefix.cover_prefix(InnerKey::from_outer_key(
                "0000100".as_bytes(),
                inner_key_off
            )));
            assert!(del_prefix.cover_prefix(InnerKey::from_outer_key(
                "0000200".as_bytes(),
                inner_key_off
            )));
            assert!(del_prefix.cover_prefix(InnerKey::from_outer_key(
                "0000300".as_bytes(),
                inner_key_off
            )));
            del_prefix = del_prefix.split(
                &DeletePrefixes::new_with_inner_key_off(inner_key_off)
                    .merge_prefix("0000200".as_bytes())
                    .merge_prefix("0000300".as_bytes()),
            );
            assert_prefix_invariant(&del_prefix);
            assert_eq!(del_prefix.prefixes.len(), 0);
            assert!(!del_prefix.cover_prefix(InnerKey::from_outer_key(
                "0000100".as_bytes(),
                inner_key_off
            )));
            assert!(!del_prefix.cover_prefix(InnerKey::from_outer_key(
                "0000200".as_bytes(),
                inner_key_off
            )));
            assert!(!del_prefix.cover_prefix(InnerKey::from_outer_key(
                "0000300".as_bytes(),
                inner_key_off
            )));
        }
    }

    #[test]
    fn test_del_prefixes_merge_prefix() {
        for inner_key_off in [0, 4] {
            let mut first = DeletePrefixes::new_with_inner_key_off(inner_key_off);
            first = first.merge_prefix("0000101".as_bytes());
            first = first.merge_prefix("00001033".as_bytes());
            first = first.merge_prefix("00001055".as_bytes());
            first = first.merge_prefix("0000107".as_bytes());

            let mut second = DeletePrefixes::new_with_inner_key_off(inner_key_off);
            second = second.merge_prefix("0000101".as_bytes());
            second = second.merge_prefix("00001022".as_bytes());
            second = second.merge_prefix("00001044".as_bytes());
            second = second.merge_prefix("00001066".as_bytes());
            second = second.merge_prefix("0000107".as_bytes());

            let merged = merge_del_prefixes_if_needed(
                Some(Bytes::from(first.marshal())),
                Some(Bytes::from(second.marshal())),
                inner_key_off,
            )
            .unwrap();
            let merged_del_prefixes = DeletePrefixes::unmarshal(&merged, inner_key_off);
            assert_eq!(
                merged_del_prefixes.prefixes,
                vec![
                    "0000101".as_bytes(),
                    "00001022".as_bytes(),
                    "00001033".as_bytes(),
                    "00001044".as_bytes(),
                    "00001055".as_bytes(),
                    "00001066".as_bytes(),
                    "0000107".as_bytes()
                ]
            );

            let rev_merged = merge_del_prefixes_if_needed(
                Some(Bytes::from(second.marshal())),
                Some(Bytes::from(first.marshal())),
                inner_key_off,
            );
            let rev_merged_del_prefixes =
                DeletePrefixes::unmarshal(&rev_merged.unwrap(), inner_key_off);
            assert_eq!(
                merged_del_prefixes.prefixes,
                rev_merged_del_prefixes.prefixes,
            );
        }
    }

    #[test]
    fn test_del_prefixes_merge_prefixes() {
        let cases = vec![
            (
                vec![], // sorted_prefixes
                vec![], // expected prefixes
            ),
            (vec!["00005000"], vec!["00005000"]),
            (vec![], vec!["00005000"]),
            (
                vec!["00005000", "00005001", "00006"],
                vec!["00005000", "00005001", "00006"],
            ),
            (
                vec!["00004000", "0000500", "00007"],
                vec!["00004000", "0000500", "00006", "00007"],
            ),
            (vec!["0000", "0001"], vec!["0000", "0001"]),
            (vec!["00001"], vec!["0000", "0001"]),
        ];

        let mut del_prefix = DeletePrefixes::new_with_inner_key_off(4);
        for (idx, (sorted_prefixes, expected_prefixes)) in cases.into_iter().enumerate() {
            del_prefix.schedule_at = idx as u64;
            del_prefix = del_prefix.merge_prefixes(sorted_prefixes.iter().map(|p| p.as_bytes()));
            let expected_prefixes = expected_prefixes
                .iter()
                .map(|p| p.as_bytes().to_vec())
                .collect::<Vec<_>>();
            assert_eq!(del_prefix.prefixes, expected_prefixes, "case {}", idx);
            assert_eq!(del_prefix.schedule_at, idx as u64);
            assert_eq!(del_prefix.inner_key_off, 4);
        }
    }

    #[test]
    fn test_del_prefixes_complementary_ranges() {
        let mut del_prefix = DeletePrefixes::new_with_inner_key_off(4);
        del_prefix = del_prefix.merge_prefix("0000-1".as_bytes());
        del_prefix = del_prefix.merge_prefix("0000-2".as_bytes());
        del_prefix = del_prefix.merge_prefix("0000-33".as_bytes());
        del_prefix = del_prefix.merge_prefix("0000-5".as_bytes());

        let cases: Vec<(
            (&str, &str),      // outer_range
            Vec<(&str, &str)>, // expected
        )> = vec![
            (("0000-0", "0000-1"), vec![("0000-0", "0000-1")]),
            (("0000-0", "0000-2"), vec![("0000-0", "0000-1")]),
            (("0000-0", "0000-3"), vec![("0000-0", "0000-1")]),
            (
                ("0000-0", "0000-33"),
                vec![("0000-0", "0000-1"), ("0000-3", "0000-33")],
            ),
            (
                ("0000-0", "0000-333"),
                vec![("0000-0", "0000-1"), ("0000-3", "0000-33")],
            ),
            (
                ("0000-0", "0000-4"),
                vec![
                    ("0000-0", "0000-1"),
                    ("0000-3", "0000-33"),
                    ("0000-34", "0000-4"),
                ],
            ),
            (
                ("0000-0", "0000-9"),
                vec![
                    ("0000-0", "0000-1"),
                    ("0000-3", "0000-33"),
                    ("0000-34", "0000-5"),
                    ("0000-6", "0000-9"),
                ],
            ),
        ];

        for (i, (outer_range, expected)) in cases.into_iter().enumerate() {
            let ranges = del_prefix
                .get_complementary_ranges(outer_range.0.as_bytes(), outer_range.1.as_bytes());
            let expected = expected
                .into_iter()
                .map(|(start, end)| (start.as_bytes().to_vec(), end.as_bytes().to_vec()))
                .collect::<Vec<_>>();
            assert_eq!(ranges, expected, "case {}: {:?}", i, outer_range);
        }
    }

    #[test]
    fn test_truncate_ts() {
        let ts = 437598164238729283;
        let truncate_ts = TruncateTs(ts);
        assert_eq!(
            TruncateTs::unmarshal(truncate_ts.marshal().as_slice()),
            truncate_ts
        );
        assert_eq!(truncate_ts.inner(), ts);
    }

    #[test]
    fn test_range_keyspace_id() {
        let mut range = ShardRange::new(&[b'x', 0, 0, 1], &[b'x', 0, 0, 2], 4);
        assert_eq!(range.keyspace_id, 1);
        range = ShardRange::new(&[], GLOBAL_SHARD_END_KEY, 0);
        assert_eq!(range.keyspace_id, 0);
        range = ShardRange::new(&[b't', 0, 0, 0], &[], 0);
        assert_eq!(range.keyspace_id, 0);
    }
}
