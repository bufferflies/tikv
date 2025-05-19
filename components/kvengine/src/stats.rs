// Copyright 2022 TiKV Project Authors. Licensed under Apache-2.0.

use std::{cmp, collections::HashSet};

use api_version;
use bytes::Bytes;
use codec::{buffer::BufferWriter, number::NumberEncoder};
use schema::schema::StorageClass;

use crate::{
    metrics::{
        ENGINE_OPEN_FILES, ENGINE_REGION_HUGE_L0_TABLE_BYTES_HISTOGRAM,
        ENGINE_REGION_HUGE_MEM_TABLE_BYTES_HISTOGRAM,
    },
    shard::ShardData,
    table::{BoundedDataSet, DataBound, InnerKey},
    IdVer, COLUMNAR_LEVELS, EXTRA_CF, NUM_CFS, WRITE_CF,
};

#[derive(Default, Debug, Serialize, Deserialize)]
#[serde(default)]
#[serde(rename_all = "kebab-case")]
pub struct EngineStats {
    pub num_shards: usize,
    pub num_initial_flushed_shard: usize,
    pub num_active_shards: usize,
    pub num_compacting_shards: usize,
    pub num_has_del_prefixes_shards: usize,
    pub num_inner_key_shards: usize,
    pub ready_destroy_range_shards: Vec<IdVer>,
    pub mem_tables_count: usize,
    pub mem_tables_size: u64,
    pub l0_tables_count: usize,
    pub l0_tables_size: u64,
    pub blob_tables_count: usize,
    pub in_use_blob_size: u64,
    pub total_blob_size: u64,
    pub partial_l0_count: usize,
    pub partial_blob_count: usize,
    pub partial_ln_count: usize,
    pub cfs_num_files: Vec<usize>,
    pub cf_total_sizes: Vec<u64>,
    pub level_num_files: Vec<usize>,
    pub level_total_sizes: Vec<u64>,
    pub index_size: u64,
    pub in_mem_index_size: u64,
    pub filter_size: u64,
    pub in_mem_filter_size: u64,
    pub open_files: i64,
    pub max_ts: u64, // Use to check whether PiTR has completed.
    pub entries: usize,
    pub old_entries: usize,
    pub tombs: usize,
    pub kv_size: u64,
    pub txn_file_locks: usize,
    pub columnar_levels: Vec<ColumnarLevelStats>,
    pub vector_indexes: VectorIndexStats,
    pub top_10_write: Vec<ShardStats>,
    pub ia: StorageClassStats,
}

impl EngineStats {
    pub fn new() -> Self {
        let mut stats = EngineStats::default();
        stats.cfs_num_files = vec![0; 3];
        stats.cf_total_sizes = vec![0; 3];
        stats.level_num_files = vec![0; 3];
        stats.level_total_sizes = vec![0; 3];
        stats
    }
}

#[derive(Default, Debug, Deserialize, Serialize)]
#[serde(default)]
#[serde(rename_all = "kebab-case")]
pub struct ColumnarStatusResp {
    // Schema file already installed region count.
    pub ready: u64,
    // Total region count.
    pub total: u64,
}

impl super::Engine {
    pub fn get_all_shard_stats(&self) -> Vec<ShardStats> {
        self.get_all_shard_stats_ext(None)
    }

    pub fn get_all_shard_stats_ext(&self, skip_shards: Option<&HashSet<IdVer>>) -> Vec<ShardStats> {
        self.get_all_shard_id_vers()
            .into_iter()
            .filter_map(|id_ver| {
                if skip_shards.is_some_and(|skip_shards| skip_shards.contains(&id_ver)) {
                    return None;
                }
                self.get_shard(id_ver.id).map(|shard| shard.get_stats())
            })
            .collect()
    }

    pub fn get_all_active_shard_stats_lite(&self) -> Vec<ShardStatsLite> {
        self.get_all_shard_id_vers()
            .into_iter()
            .filter_map(|id_ver| {
                self.get_shard(id_ver.id).and_then(|shard| {
                    if shard.is_active() {
                        Some(shard.get_stats().into())
                    } else {
                        None
                    }
                })
            })
            .collect()
    }

    pub fn get_shard_stat(&self, region_id: u64) -> ShardStats {
        self.get_shard_stat_opt(region_id).unwrap_or_default()
    }

    pub fn get_shard_stat_opt(&self, region_id: u64) -> Option<ShardStats> {
        self.shards.get(&region_id).map(|shard| shard.get_stats())
    }

    pub fn get_engine_stats(&self, mut shard_stats: Vec<ShardStats>) -> EngineStats {
        let mut engine_stats = EngineStats::new();
        engine_stats.num_shards = shard_stats.len();
        engine_stats.open_files = self.fd_cache.size() as i64;
        engine_stats.columnar_levels = vec![ColumnarLevelStats::default(); COLUMNAR_LEVELS];
        for shard in &shard_stats {
            if shard.active {
                engine_stats.num_active_shards += 1;
            }
            if shard.compacting {
                engine_stats.num_compacting_shards += 1;
            }
            if shard.flushed {
                engine_stats.num_initial_flushed_shard += 1;
            }
            if shard.has_del_prefixes {
                engine_stats.num_has_del_prefixes_shards += 1;
            }
            if shard.ready_to_destroy_range {
                engine_stats
                    .ready_destroy_range_shards
                    .push(IdVer::new(shard.id, shard.ver));
            }
            if shard.inner_key_off > 0 {
                engine_stats.num_inner_key_shards += 1;
            }
            engine_stats.mem_tables_count += shard.mem_table_count;
            engine_stats.mem_tables_size += shard.mem_table_size;
            engine_stats.l0_tables_count += shard.l0_table_count;
            engine_stats.l0_tables_size += shard.l0_table_size;
            engine_stats.blob_tables_count += shard.blob_table_count;
            engine_stats.in_use_blob_size += shard.in_use_blob_size;
            engine_stats.total_blob_size += shard.total_blob_size;
            engine_stats.partial_l0_count += shard.partial_l0s;
            engine_stats.partial_blob_count += shard.shared_blob_tables;
            engine_stats.partial_ln_count += shard.partial_tbls;
            engine_stats.index_size += shard.index_size;
            engine_stats.in_mem_index_size += shard.in_mem_index_size;
            engine_stats.filter_size += shard.filter_size;
            engine_stats.in_mem_filter_size += shard.in_mem_filter_size;
            engine_stats.max_ts = cmp::max(engine_stats.max_ts, shard.max_ts);
            engine_stats.entries += shard.entries;
            engine_stats.old_entries += shard.old_entries;
            engine_stats.tombs += shard.tombs;
            engine_stats.kv_size += shard.kv_size;
            for cf in 0..NUM_CFS {
                let shard_cf_stat = &shard.cfs[cf];
                for (i, level_stat) in shard_cf_stat.levels.iter().enumerate() {
                    engine_stats.level_num_files[i] += level_stat.num_tables;
                    engine_stats.cfs_num_files[cf] += level_stat.num_tables;
                    engine_stats.level_total_sizes[i] += level_stat.data_size;
                    engine_stats.cf_total_sizes[cf] += level_stat.data_size;
                }
            }
            engine_stats.txn_file_locks += shard.txn_file_locks;
            for (i, col_level) in shard.columnar_levels.iter().enumerate() {
                let engine_col_level = &mut engine_stats.columnar_levels[i];
                engine_col_level.num_files += col_level.num_files;
                engine_col_level.data_size += col_level.data_size;
            }
            engine_stats.vector_indexes.data_size += shard.vector_indexes.data_size;
            engine_stats.vector_indexes.num_files += shard.vector_indexes.num_files;
            if matches!(shard.storage_class, StorageClass::Ia) {
                engine_stats.ia.add_ia_shard(shard);
            }
        }
        ENGINE_OPEN_FILES.set(engine_stats.open_files);
        shard_stats.sort_by(|a, b| {
            let a_size = a.mem_table_size + a.l0_table_size;
            let b_size = b.mem_table_size + b.l0_table_size;
            b_size.cmp(&a_size)
        });
        shard_stats.truncate(10);
        engine_stats.top_10_write = shard_stats;
        engine_stats
    }

    pub fn update_region_huge_table_bytes_metrics(
        shard_stats: &Vec<ShardStats>,
        max_mem_table_size: u64,
    ) {
        if max_mem_table_size == 0 {
            return;
        }
        let region_huge_mem_table_bytes = ENGINE_REGION_HUGE_MEM_TABLE_BYTES_HISTOGRAM.local();
        let region_huge_l0_table_bytes = ENGINE_REGION_HUGE_L0_TABLE_BYTES_HISTOGRAM.local();
        for shard in shard_stats {
            if shard.mem_table_size > max_mem_table_size * 2 {
                region_huge_mem_table_bytes.observe(shard.mem_table_size as f64);
            }
            if shard.l0_table_size > max_mem_table_size * 2 {
                region_huge_l0_table_bytes.observe(shard.l0_table_size as f64);
            }
        }
        region_huge_mem_table_bytes.flush();
        region_huge_l0_table_bytes.flush();
    }

    fn append_table_record_prefix(prefix: &mut Vec<u8>, table_id: i64) {
        prefix.write_bytes(b"t").unwrap();
        prefix.write_i64(table_id).unwrap();
        prefix.write_bytes(b"_r").unwrap();
    }

    pub fn collect_columnar_status(&self, keyspace_id: u32, table_id: i64) -> ColumnarStatusResp {
        let mut ready = 0;
        let mut total = 0;
        let prefix = api_version::ApiV2::get_txn_keyspace_prefix(keyspace_id);
        let mut table_lower_key = prefix.clone();
        Self::append_table_record_prefix(&mut table_lower_key, table_id);
        let table_upper_key = keys::next_key(&table_lower_key);
        self.get_all_shard_id_vers().into_iter().for_each(|id_ver| {
            if let Some(shard) = self.get_shard(id_ver.id) {
                let table_bound = DataBound::new(
                    InnerKey::from_outer_key(&table_lower_key),
                    InnerKey::from_outer_end_key(&table_upper_key),
                    false,
                );
                if shard.get_data().keyspace_id == keyspace_id && shard.overlap_bound(table_bound) {
                    if shard.has_columnar_table(table_id) {
                        ready += 1;
                    }
                    total += 1;
                }
            }
        });
        ColumnarStatusResp { ready, total }
    }
}

#[derive(Default, Serialize, Deserialize, Debug)]
#[serde(default)]
#[serde(rename_all = "kebab-case")]
pub struct ShardStats {
    pub id: u64,
    pub ver: u64,
    pub keyspace: u32,
    #[serde(serialize_with = "serialize_bytes_as_hex")]
    pub start: Bytes,
    #[serde(serialize_with = "serialize_bytes_as_hex")]
    pub end: Bytes,
    pub inner_key_off: usize,
    pub active: bool,
    pub compacting: bool,
    pub flushed: bool,
    pub mem_table_count: usize,
    pub mem_table_size: u64,
    pub mem_table_unpersisted_props_size: usize,
    pub l0_table_count: usize,
    pub blob_table_count: usize,
    pub in_use_blob_size: u64,
    pub total_blob_size: u64,

    pub l0_table_size: u64,
    pub l0_cf_table_size: [u64; NUM_CFS],
    pub cfs: Vec<CfStats>,
    pub col_levels_stats: Vec<LevelStats>,

    pub index_size: u64,
    pub in_mem_index_size: u64,
    pub filter_size: u64,
    pub in_mem_filter_size: u64,
    pub max_ts: u64,
    pub entries: usize,
    pub old_entries: usize,
    pub tombs: usize,
    pub kv_size: u64,
    pub keyspace_prefix_tables: u32,
    pub base_version: u64,
    pub meta_sequence: u64,
    pub write_sequence: u64,
    // Total size of all SST files and blobs referenced by the shard (not blob table file size).
    pub total_size: u64,
    pub partial_l0s: usize,
    pub shared_blob_tables: usize,
    pub partial_tbls: usize,
    pub compaction_cf: isize,
    pub compaction_level: isize,
    pub compaction_score: f64,
    pub has_over_bound_data: bool,
    pub has_del_prefixes: bool,
    pub ready_to_destroy_range: bool,
    pub trim_over_bound: bool,
    pub manual_major_compaction: bool,
    pub storage_class: StorageClass,
    pub is_sync: bool,
    pub checked_schema_version: i64,
    // Txn File Stats
    pub txn_file_locks: usize,
    // Columnar Stats
    pub schema_version: i64,
    pub schema_restore_version: u64,
    pub columnar_tables: usize,
    pub columnar_levels: Vec<ColumnarLevelStats>,
    pub vector_indexes: VectorIndexStats,
}

impl ShardStats {
    #[inline]
    pub fn mem_table_is_empty(&self) -> bool {
        self.mem_table_size == 0 && self.mem_table_count == 1
    }

    #[cfg(any(test, feature = "testexport"))]
    pub fn is_major_compacted(&self) -> bool {
        let bottom_most_level = self.cfs[WRITE_CF].levels.last().unwrap();
        self.mem_table_size == 0
            && self.mem_table_count == 1
            && self.l0_table_count == 0
            && bottom_most_level.level == crate::CF_LEVELS[WRITE_CF]
            && bottom_most_level.num_tables != 0
    }

    pub fn for_each_cf_level(&self, mut f: impl FnMut((usize, &LevelStats))) {
        for (cf, cf_stats) in self.cfs.iter().enumerate() {
            cf_stats.levels.iter().for_each(|lv| {
                f((cf, lv));
            });
        }
    }
}

#[derive(Default, Serialize, Deserialize, Debug)]
#[serde(default)]
#[serde(rename_all = "kebab-case")]
pub struct ShardStatsLite {
    pub id: u64,
    pub ver: u64,
    pub start: Bytes,
    pub end: Bytes,
    pub inner_key_off: usize,
    // Total size of all SST files and blobs referenced by the shard (not blob table file size).
    pub total_size: u64,
    pub schema_version: i64,
    pub schema_restore_version: u64,
    pub write_sequence: u64,
    pub storage_class: StorageClass,
    pub columnar_tables: usize,
}

impl From<ShardStats> for ShardStatsLite {
    fn from(s: ShardStats) -> Self {
        Self {
            id: s.id,
            ver: s.ver,
            start: s.start,
            end: s.end,
            inner_key_off: s.inner_key_off,
            total_size: s.total_size,
            schema_version: s.schema_version,
            schema_restore_version: s.schema_restore_version,
            write_sequence: s.write_sequence,
            storage_class: s.storage_class,
            columnar_tables: s.columnar_tables,
        }
    }
}

impl ShardStatsLite {
    pub fn with_schema(&self) -> bool {
        self.storage_class.is_specified() || self.columnar_tables > 0
    }
}

#[derive(Default, Serialize, Deserialize, Debug)]
#[serde(default)]
#[serde(rename_all = "kebab-case")]
pub struct CfStats {
    pub levels: Vec<LevelStats>,
}

#[derive(Default, Serialize, Deserialize, Debug, Clone)]
#[serde(default)]
#[serde(rename_all = "kebab-case")]
pub struct ColumnarLevelStats {
    pub num_files: usize,
    pub data_size: u64,
}

#[derive(Default, Serialize, Deserialize, Debug, Clone)]
#[serde(default)]
#[serde(rename_all = "kebab-case")]
pub struct VectorIndexStats {
    pub num_files: usize,
    pub data_size: u64,
}

#[derive(Default, Clone, Debug, PartialEq)]
pub struct LevelStatsLite {
    pub data_size: u64,
    pub blob_size: u64,
    pub entries: u64,
    // The followings are for WRITE_CF only.
    pub kv_size: u64,
    // The followings are for WRITE_CF & level 2+ only.
    pub lv2plus_max_ts: u64,
    pub lv2plus_tombs: u64,
    pub lv2plus_entries_write_cf: u64,
    // The following is for WRITE_CF & EXTRA_CF
    pub max_ts: u64,
    // The following is for Columnar.
    pub columnar_size: u64,
}

impl LevelStatsLite {
    pub fn add(&mut self, other: &LevelStatsLite, cf: usize) {
        self.data_size += other.data_size;
        self.blob_size += other.blob_size;
        self.entries += other.entries;
        if cf == WRITE_CF {
            self.kv_size += other.kv_size;
            self.lv2plus_max_ts = cmp::max(self.lv2plus_max_ts, other.lv2plus_max_ts);
            self.lv2plus_tombs += other.lv2plus_tombs;
            self.lv2plus_entries_write_cf += other.lv2plus_entries_write_cf;
            self.columnar_size += other.columnar_size;
        }
        self.max_ts = max_ts_by_cf(self.max_ts, cf, other.max_ts);
    }
}

#[derive(Default, Serialize, Deserialize, Debug)]
#[serde(default)]
#[serde(rename_all = "kebab-case")]
pub struct LevelStats {
    pub level: usize,
    pub num_tables: usize,
    pub data_size: u64,
    pub index_size: u64,
    pub in_mem_index_size: u64,
    pub filter_size: u64,
    pub in_mem_filter_size: u64,
    pub max_ts: u64,
    pub entries: usize,
    pub old_entries: usize,
    pub tombs: usize,
    pub kv_size: u64,
    pub in_use_blob_size: u64,
    pub columnar_size: u64,
}

#[derive(Default, Debug, Serialize, Deserialize)]
#[serde(default)]
#[serde(rename_all = "kebab-case")]
pub struct StorageClassStats {
    /// Number of shards using the specified storage class.
    pub num_shards: usize,
    /// Number of tables (e.g. SsTables, not TiDB tables).
    pub num_tables: usize,
    /// Total size of the tables.
    pub data_size: u64,
    /// Total kv size (before discount).
    pub kv_size: u64,
}

impl StorageClassStats {
    pub fn add_ia_shard(&mut self, shard: &ShardStats) {
        self.num_shards += 1;
        if !shard.is_sync {
            // Conversion to async would be later than storage class.
            shard.for_each_cf_level(|(cf, lv)| {
                if ShardData::can_use_ia(cf, lv.level) {
                    self.num_tables += lv.num_tables;
                    self.data_size += lv.data_size;
                    self.kv_size += lv.kv_size;
                }
            })
        }
    }
}

#[derive(Default, Serialize, Deserialize, Debug)]
#[serde(default)]
#[serde(rename_all = "kebab-case")]
pub struct ShardTruncateTsStats {
    pub id: u64,
    pub ver: u64,
    pub cur_max_ts: u64,
    pub truncate_ts: u64,
}

impl super::Shard {
    pub fn get_stats(&self) -> ShardStats {
        let mut total_size = 0;
        let mut tbl_index_size = 0;
        let mut in_mem_tbl_index_size = 0;
        let mut tbl_filter_size = 0;
        let mut in_mem_tbl_filter_size = 0;
        let mut max_ts = 0;
        let mut entries = 0;
        let mut old_entries = 0;
        let mut tombs = 0; // for WRITE_CF only
        let mut kv_size = 0; // for WRITE_CF only
        let data = self.get_data();
        let mem_table_count = data.mem_tbls.len();
        let mut mem_table_size = 0;
        let mut mem_table_unpersisted_props_size = 0;
        for mem_tbl in data.mem_tbls.as_slice() {
            mem_table_size += mem_tbl.size();
            mem_table_unpersisted_props_size = mem_tbl.unpersisted_props_size();
            max_ts = cmp::max(max_ts, mem_tbl.data_max_ts());
        }
        total_size += mem_table_size;
        let mut partial_l0s = 0;
        let mut shared_blob_tables = 0;
        let l0_table_count = data.l0_tbls.len();
        let mut l0_table_size = 0;
        let mut l0_cf_table_size = [0; NUM_CFS];
        let mut total_blob_size = 0;
        let mut in_use_blob_size = 0;
        let mut keyspace_prefix_tables = 0;
        let blob_table_count = data.blob_tbl_map.len();
        // FIXME: Calculate the total size of blob files.
        let mut blob_table_size = 0;
        let shard_bound = self.data_bound();
        for v in data.blob_tbl_map.values() {
            if shard_bound.contains_bound(v.data_bound()) {
                blob_table_size += v.size();
            } else {
                blob_table_size += v.size() / 2;
                shared_blob_tables += 1;
            }
        }
        total_blob_size += blob_table_size;
        for l0_tbl in data.l0_tbls.as_slice() {
            if shard_bound.contains_bound(l0_tbl.data_bound()) {
                l0_table_size += l0_tbl.size();
            } else {
                // TODO: estimate size by number of blocks in table.
                l0_table_size += l0_tbl.size() / 2;
                partial_l0s += 1;
            }
            let mut l0_keyspace_prefix_tables_counted = false;
            for cf in 0..NUM_CFS {
                if let Some(cf_tbl) = l0_tbl.get_cf(cf) {
                    tbl_index_size += cf_tbl.index_size();
                    tbl_filter_size += cf_tbl.filter_size();
                    in_mem_tbl_filter_size += cf_tbl.in_mem_filter_size();
                    max_ts = max_ts_by_cf(max_ts, cf, cf_tbl.max_ts);
                    entries += cf_tbl.entries as usize;
                    old_entries += cf_tbl.old_entries as usize;
                    if cf == WRITE_CF {
                        tombs += cf_tbl.tombs as usize;
                        kv_size += cf_tbl.kv_size;
                        cf_tbl.expire_cache(0);
                    }
                    if cf_tbl.keyspace_id.is_some() && !l0_keyspace_prefix_tables_counted {
                        keyspace_prefix_tables += 1;
                        l0_keyspace_prefix_tables_counted = true;
                    }
                    in_use_blob_size += cf_tbl.total_blob_size();

                    if shard_bound.contains_bound(cf_tbl.data_bound()) {
                        l0_cf_table_size[cf] += cf_tbl.size();
                    } else {
                        l0_cf_table_size[cf] += cf_tbl.size() / 2;
                    }
                }
            }
        }
        total_size += l0_table_size;
        let mut partial_tbls = 0;
        let mut cfs = vec![];
        for cf in 0..NUM_CFS {
            let scf = data.get_cf(cf);
            let mut cf_stat = CfStats { levels: vec![] };
            for l in scf.levels.as_slice() {
                let mut level_stats = LevelStats::default();
                level_stats.level = l.level;
                level_stats.num_tables = l.tables.len();
                for t in l.tables.as_slice() {
                    t.expire_cache(l.level);
                    if t.keyspace_id.is_some() {
                        keyspace_prefix_tables += 1;
                    }
                    if shard_bound.contains_bound(t.data_bound()) {
                        level_stats.data_size += t.size();
                        level_stats.index_size += t.index_size();
                        level_stats.in_mem_index_size += t.in_mem_index_size();
                        level_stats.filter_size += t.filter_size();
                        level_stats.in_mem_filter_size += t.in_mem_filter_size();
                        level_stats.entries += t.entries as usize;
                        level_stats.old_entries += t.old_entries as usize;
                        if cf == WRITE_CF {
                            level_stats.tombs += t.tombs as usize;
                            level_stats.kv_size += t.kv_size;
                        }
                        level_stats.in_use_blob_size += t.total_blob_size();
                    } else {
                        level_stats.data_size += t.size() / 2;
                        level_stats.index_size += t.index_size() / 2;
                        level_stats.filter_size += t.filter_size() / 2;
                        level_stats.entries += t.entries as usize / 2;
                        level_stats.old_entries += t.old_entries as usize / 2;
                        if cf == WRITE_CF {
                            level_stats.tombs += t.tombs as usize / 2;
                            level_stats.kv_size += t.kv_size / 2;
                        }
                        partial_tbls += 1;
                        level_stats.in_use_blob_size += t.total_blob_size() / 2;
                    }
                    level_stats.max_ts = max_ts_by_cf(level_stats.max_ts, cf, t.max_ts);
                }
                total_size += level_stats.data_size;
                tbl_index_size += level_stats.index_size;
                in_mem_tbl_index_size += level_stats.in_mem_index_size;
                tbl_filter_size += level_stats.filter_size;
                in_mem_tbl_filter_size += level_stats.in_mem_filter_size;
                max_ts = cmp::max(max_ts, level_stats.max_ts);
                entries += level_stats.entries;
                old_entries += level_stats.old_entries;
                tombs += level_stats.tombs;
                kv_size += level_stats.kv_size;
                in_use_blob_size += level_stats.in_use_blob_size;
                cf_stat.levels.push(level_stats);
            }
            cfs.push(cf_stat);
        }
        let mut col_levels_stats = vec![];
        for cl in data.col_levels.levels.as_slice() {
            let mut level_stats = LevelStats::default();
            level_stats.level = cl.level;
            for tbl in cl.files.as_slice() {
                level_stats.columnar_size += tbl.size();
            }
            col_levels_stats.push(level_stats);
        }
        total_size += in_use_blob_size;
        let priority = self.compaction_priority.read().unwrap().clone();
        let compaction_cf = priority.as_ref().map_or(0, |x| x.cf());
        let compaction_level = priority.as_ref().map_or(0, |x| x.level());
        let compaction_score = priority.as_ref().map_or(0f64, |x| x.score());
        let pending_ops = self.pending_ops.read().unwrap().clone();
        let txn_file_locks = data.lock_txn_files.len();
        let schema_version = data.schema_version;
        let schema_restore_version = data
            .schema_file
            .as_ref()
            .map(|sf| sf.get_restore_version())
            .unwrap_or_default();
        let columnar_tables = data.columnar_table_ids.len();
        let mut columnar_levels = vec![ColumnarLevelStats::default(); COLUMNAR_LEVELS];
        for (i, l) in data.col_levels.levels.iter().enumerate() {
            columnar_levels[i].num_files = l.files.len();
            columnar_levels[i].data_size = l.files.iter().map(|c| c.get_file().size()).sum();
        }
        let mut vector_indexes = VectorIndexStats::default();
        for vi in data.vector_indexes.get_all() {
            vector_indexes.num_files += vi.files.len();
            for f in &vi.files {
                vector_indexes.data_size += f.file_size();
            }
        }
        ShardStats {
            id: self.id,
            ver: self.ver,
            keyspace: self.keyspace_id,
            start: self.outer_start.clone(),
            end: self.outer_end.clone(),
            inner_key_off: data.inner_key_off,
            active: self.is_active(),
            compacting: self.is_compacting(),
            flushed: self.get_initial_flushed(),
            mem_table_count,
            mem_table_size,
            mem_table_unpersisted_props_size,
            l0_table_count,
            blob_table_count,
            l0_table_size,
            l0_cf_table_size,
            total_blob_size,
            in_use_blob_size,
            cfs,
            col_levels_stats,
            base_version: self.get_base_version(),
            meta_sequence: self.get_meta_sequence(),
            write_sequence: self.get_write_sequence(),
            total_size,
            index_size: tbl_index_size,
            in_mem_index_size: in_mem_tbl_index_size,
            filter_size: tbl_filter_size,
            in_mem_filter_size: in_mem_tbl_filter_size,
            max_ts,
            entries,
            old_entries,
            tombs,
            kv_size,
            keyspace_prefix_tables,
            partial_l0s,
            shared_blob_tables,
            partial_tbls,
            compaction_cf,
            compaction_level,
            compaction_score,
            has_over_bound_data: data.has_over_bound_data(),
            has_del_prefixes: !pending_ops.del_prefixes.is_empty(),
            ready_to_destroy_range: Self::ready_to_destroy_range(&pending_ops.del_prefixes, &data),
            trim_over_bound: pending_ops.trim_over_bound,
            manual_major_compaction: pending_ops.manual_major_compaction,
            storage_class: pending_ops.storage_class,
            is_sync: data.is_sync(),
            checked_schema_version: self.get_checked_schema_ver(),
            txn_file_locks,
            schema_version,
            schema_restore_version,
            columnar_tables,
            columnar_levels,
            vector_indexes,
        }
    }
}

#[inline]
pub fn max_ts_by_cf(max_ts: u64, cf: usize, cf_max_ts: u64) -> u64 {
    // Ignore LOCK_CF, as `ts` in LOCK_CF is not a TSO.
    if (cf == WRITE_CF || cf == EXTRA_CF) && max_ts < cf_max_ts {
        cf_max_ts
    } else {
        max_ts
    }
}

pub fn serialize_bytes_as_hex<S>(bytes: &Bytes, serializer: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    let hex_str = log_wrappers::hex_encode_upper(bytes);
    serializer.serialize_str(&hex_str)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::LOCK_CF;

    #[test]
    fn test_max_ts_by_cf() {
        let cases = vec![
            (100, WRITE_CF, 200, 200),
            (100, LOCK_CF, 200, 100),
            (100, EXTRA_CF, 200, 200),
            (200, WRITE_CF, 100, 200),
            (200, LOCK_CF, 100, 200),
            (200, EXTRA_CF, 100, 200),
        ];
        for (max_ts, cf, cf_max_ts, expected) in cases {
            assert_eq!(max_ts_by_cf(max_ts, cf, cf_max_ts), expected,);
        }
    }

    #[test]
    fn test_level_stats_lite() {
        let mut stats1 = LevelStatsLite {
            data_size: 10000,
            blob_size: 20000,
            kv_size: 8000,
            entries: 2000,
            lv2plus_max_ts: 50,
            lv2plus_tombs: 1000,
            lv2plus_entries_write_cf: 1500,
            max_ts: 100,
            columnar_size: 30000,
        };
        let mut stats2 = stats1.clone();

        stats2.max_ts = 110;
        stats2.lv2plus_max_ts = 60;
        stats1.add(&stats2, WRITE_CF);
        assert_eq!(
            stats1,
            LevelStatsLite {
                data_size: 20000,
                blob_size: 40000,
                kv_size: 16000,
                entries: 4000,
                lv2plus_max_ts: 60,
                lv2plus_tombs: 2000,
                lv2plus_entries_write_cf: 3000,
                max_ts: 110,
                columnar_size: 60000,
            }
        );

        stats2.max_ts = 120;
        stats2.lv2plus_max_ts = 70;
        stats1.add(&stats2, LOCK_CF);
        assert_eq!(
            stats1,
            LevelStatsLite {
                data_size: 30000,
                blob_size: 60000,
                kv_size: 16000, // unchanged
                entries: 6000,
                lv2plus_max_ts: 60,
                lv2plus_tombs: 2000,            // unchanged
                lv2plus_entries_write_cf: 3000, // unchanged
                max_ts: 110,                    // unchanged
                columnar_size: 60000,
            }
        );

        stats2.max_ts = 130;
        stats1.add(&stats2, EXTRA_CF);
        assert_eq!(
            stats1,
            LevelStatsLite {
                data_size: 40000,
                blob_size: 80000,
                kv_size: 16000, // unchanged
                entries: 8000,
                lv2plus_max_ts: 60,             // unchanged
                lv2plus_tombs: 2000,            // unchanged
                lv2plus_entries_write_cf: 3000, // unchanged
                max_ts: 130,
                columnar_size: 60000,
            }
        );

        let mut stats3 = LevelStatsLite::default();
        stats3.columnar_size = 10000;
        stats1.add(&stats3, WRITE_CF);
        assert_eq!(
            stats1,
            LevelStatsLite {
                data_size: 40000,               // unchanged
                blob_size: 80000,               // unchanged
                kv_size: 16000,                 // unchanged
                entries: 8000,                  // unchanged
                lv2plus_max_ts: 60,             // unchanged
                lv2plus_tombs: 2000,            // unchanged
                lv2plus_entries_write_cf: 3000, // unchanged
                max_ts: 130,                    // unchanged
                columnar_size: 70000,
            }
        );
    }
}
