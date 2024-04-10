// Copyright 2021 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    cmp::{max, min},
    collections::HashMap,
    sync::{
        atomic::{Ordering, Ordering::Release},
        Arc,
    },
};

use api_version::{
    api_v2::{is_whole_keyspace_range, KEYSPACE_PREFIX_LEN},
    ApiV2, KeyMode, KvFormat,
};
use bytes::{Buf, Bytes};
use dashmap::mapref::entry::Entry;
use kvenginepb as pb;
use slog_global::info;

use crate::{limiter::RegionLimiter, table::TableExt, *};

#[derive(Debug)]
pub struct CheckMergeResult {
    pub source_overbound: bool,
    pub target_overbound: bool,

    /// source/target region is required to be empty, but it's not.
    pub source_require_empty: bool,
    pub target_require_empty: bool,

    // source and target shards belong to same keyspace but with different encryption key.
    pub inconsistent_encryption_key: bool,
}

impl Engine {
    pub fn split(&self, mut cs: pb::ChangeSet, initial_seq: u64) -> Result<()> {
        let split = cs.take_split();
        let sequence = cs.get_sequence();

        let old_shard = self.get_shard_with_ver(cs.shard_id, cs.shard_ver)?;
        self.prepare_update_shard_version(&old_shard, sequence);

        let mut new_shards = vec![];
        let new_shard_props = split.get_new_shards();
        let new_ver = old_shard.ver + new_shard_props.len() as u64 - 1;
        let old_data = old_shard.get_data();
        assert!(
            old_data.lock_txn_files.is_empty(),
            "{} shard with txn files are not allowed to split, lock_txn_files {:?}",
            old_shard.tag(),
            old_data.lock_txn_files
        );
        let old_del_prefixes = old_shard.pending_ops.read().unwrap().del_prefixes.clone();
        for i in 0..=split.keys.len() {
            let (start_key, end_key) = get_splitting_start_end(
                old_shard.outer_start.chunk(),
                old_shard.outer_end.chunk(),
                split.get_keys(),
                i,
            );
            let range = if self.core.opts.enable_inner_key_offset
                && is_whole_keyspace_range(start_key, end_key)
            {
                ShardRange::new(start_key, end_key, KEYSPACE_PREFIX_LEN)
            } else {
                debug!(
                    "engine split";
                    "shard_id" => old_shard.id,
                    "start_key" => format!("{:?}", start_key),
                    "end_key" => format!("{:?}", end_key),
                    "inner_key_off" => old_shard.inner_key_off
                );
                ShardRange::new(start_key, end_key, old_shard.inner_key_off)
            };
            // Note: properties of new shards are processed in `build_split_pb`.
            let mut new_shard = Shard::new(
                self.get_engine_id(),
                &new_shard_props[i],
                new_ver,
                range,
                self.opts.clone(),
                &self.master_key,
            );
            new_shard.parent_id = old_shard.id;
            if new_shard.id == old_shard.id {
                new_shard.set_active(old_shard.is_active());
                store_u64(&new_shard.base_version, old_shard.get_base_version());
                store_u64(&new_shard.meta_seq, sequence);
                store_u64(&new_shard.write_sequence, sequence);
            } else {
                store_u64(
                    &new_shard.base_version,
                    old_shard.get_base_version() + sequence,
                );
                store_u64(&new_shard.meta_seq, initial_seq);
                store_u64(&new_shard.write_sequence, initial_seq);
            }
            if !old_del_prefixes.is_empty() {
                // We need to use the old shard's DEL_PREFIXES_KEY to overwrite the new shard's
                // DEL_PREFIXES_KEY. because the destroy_range compaction may have not
                // applied to the shard.
                let new_del_prefixes =
                    old_del_prefixes.build_split(start_key, end_key, old_shard.inner_key_off);
                if !new_del_prefixes.is_empty() {
                    new_shard.set_property(DEL_PREFIXES_KEY, &new_del_prefixes.marshal());
                }
            }
            new_shards.push(Arc::new(new_shard));
        }
        for new_shard in &new_shards {
            let new_mem_tbls = new_shard.split_mem_tables(&old_data.mem_tbls);
            let mut new_l0s = vec![];
            for l0 in &old_data.l0_tbls {
                if new_shard.overlap_table(l0.smallest(), l0.biggest()) {
                    new_l0s.push(l0.clone());
                }
            }
            let mut new_blob_tbl_map = HashMap::new();
            for blob_tbl in old_data.blob_tbl_map.values() {
                if new_shard.overlap_table(blob_tbl.smallest_key(), blob_tbl.biggest_key()) {
                    new_blob_tbl_map.insert(blob_tbl.id(), blob_tbl.clone());
                }
            }
            let mut new_cfs = [ShardCf::new(0), ShardCf::new(1), ShardCf::new(2)];
            for cf in 0..NUM_CFS {
                let old_scf = old_data.get_cf(cf);
                for lh in &old_scf.levels {
                    let mut new_level_tbls = vec![];
                    for tbl in lh.tables.as_slice() {
                        if new_shard.overlap_table(tbl.smallest(), tbl.biggest()) {
                            new_level_tbls.push(tbl.clone());
                        }
                    }
                    let new_level = LevelHandler::new(lh.level, new_level_tbls);
                    new_cfs[cf].set_level(new_level);
                }
            }
            let new_data = ShardData::new(
                new_shard.range.clone(),
                new_mem_tbls,
                new_l0s,
                Arc::new(new_blob_tbl_map),
                new_cfs,
                old_data.unloaded_tbls.clone(),
                vec![],
                RegionLimiter::new_from(&old_data.limiter),
            );
            new_shard.set_data(new_data);
        }
        for shard in new_shards.drain(..) {
            self.refresh_shard_states(&shard);
            let id = shard.id;
            if id != old_shard.id {
                match self.shards.entry(id) {
                    Entry::Occupied(_) => {
                        // The shard already exists, it must be created by ingest, and it maybe
                        // newer than this one, we avoid insert it.
                        continue;
                    }
                    Entry::Vacant(entry) => {
                        entry.insert(shard.clone());
                    }
                }
            } else {
                self.shards.insert(id, shard.clone());
            }
            let all_files = shard.get_all_files();
            info!(
                "split new shard {}, start {:x}, end {:x}, all files {:?}",
                shard.tag(),
                shard.outer_start,
                shard.outer_end,
                all_files
            );
        }
        Ok(())
    }

    pub(crate) fn prepare_update_shard_version(&self, shard: &Shard, sequence: u64) {
        shard.write_sequence.store(sequence, Release);
        let version = shard.load_mem_table_version();
        // Switch the old shard mem-table, so the first mem-table is always empty.
        // ignore the read-only mem-table to be flushed. let the new shard handle it.
        self.switch_mem_table(shard, version);
        self.send_flush_msg(FlushMsg::Clear(shard.id));
        self.send_compact_msg(CompactMsg::Clear(IdVer::new(shard.id, shard.ver)));
    }

    pub fn check_merge(
        &self,
        source_id: u64,
        source_ver: u64,
        target_id: u64,
        target_ver: u64,
    ) -> Result<CheckMergeResult> {
        let source_shard = self.get_shard_with_ver(source_id, source_ver)?;
        if !source_shard.get_initial_flushed() {
            return Err(Error::CheckMerge("source not initial flushed".to_string()));
        }
        if source_shard.has_txn_file_locks() {
            return Err(Error::CheckMerge("source has txn file locks".to_string()));
        }

        let target_shard = self.get_shard_with_ver(target_id, target_ver)?;
        if !target_shard.get_initial_flushed() {
            return Err(Error::CheckMerge("target not initial flushed".to_string()));
        }

        let belongs_to_same_keyspace = ApiV2::is_belongs_to_same_keyspace(
            &source_shard.outer_start,
            &target_shard.outer_start,
        );
        // Check if the source and target shards belong to the same keyspace but have
        // different encryption key property. This situation might occur during
        // keyspace restoration. In such cases, merging should be avoided.
        // If the source and target shards are from different keyspaces, the encryption
        // key property will be removed in `commit_merge`.
        let inconsistent_encryption_key = belongs_to_same_keyspace
            && source_shard
                .encryption_key
                .as_ref()
                .map(|k| k.cipher_text.clone())
                != target_shard
                    .encryption_key
                    .as_ref()
                    .map(|k| k.cipher_text.clone());

        let (clear_source, clear_target) =
            need_clear_region_data_on_merge(&source_shard.outer_start, &target_shard.outer_start);
        let shard_is_empty = |shard: &Shard| -> bool {
            shard.is_empty()
                || shard.get_del_prefixes().cover_full_keyspace(
                    ApiV2::get_keyspace_prefix(&shard.outer_start).unwrap_or_default(),
                )
        };
        let source_require_empty = clear_source && !shard_is_empty(source_shard.as_ref());
        let target_require_empty = clear_target && !shard_is_empty(target_shard.as_ref());

        Ok(CheckMergeResult {
            source_overbound: source_shard.has_over_bound_data(),
            target_overbound: target_shard.has_over_bound_data(),
            source_require_empty,
            target_require_empty,
            inconsistent_encryption_key,
        })
    }

    pub fn prepare_merge(&self, shard_id: u64, shard_ver: u64, sequence: u64) {
        let old_shard = self.get_shard_with_ver(shard_id, shard_ver).unwrap();
        self.prepare_update_shard_version(&old_shard, sequence);
        let mut new_shard = self.new_shard_version(&old_shard, sequence);
        // source shard may have non-empty mem-table, we need to flush them before
        // commit merge. The initial_flushed of the new shard is false, set the
        // parent for later initial flush.
        new_shard.parent_id = old_shard.id;
        info!("{} shard prepared merge", new_shard.tag());
        self.shards.insert(new_shard.id, Arc::new(new_shard));
    }

    pub fn rollback_merge(&self, shard_id: u64, shard_ver: u64, sequence: u64) {
        let old_shard = self.get_shard_with_ver(shard_id, shard_ver).unwrap();
        self.prepare_update_shard_version(&old_shard, sequence);
        let new_shard = self.new_shard_version(&old_shard, sequence);
        // There is no write during merging state, so we can directly set
        // initial_flushed to true.
        new_shard.initial_flushed.store(true, Ordering::Release);
        info!("{} shard rollback merge", new_shard.tag());
        self.shards.insert(new_shard.id, Arc::new(new_shard));
    }

    pub fn commit_merge(
        &self,
        shard_id: u64,
        shard_ver: u64,
        source: &ChangeSet,
        sequence: u64,
    ) -> Result<()> {
        let old_shard = self.get_shard_with_ver(shard_id, shard_ver)?;
        self.prepare_update_shard_version(&old_shard, sequence);
        let source_snap = source.get_snapshot();

        let belongs_to_same_keyspace =
            ApiV2::is_belongs_to_same_keyspace(&source_snap.outer_start, &old_shard.outer_start);
        if !belongs_to_same_keyspace {
            old_shard.del_property(ENCRYPTION_KEY);
        }

        let (clear_source, clear_target) =
            need_clear_region_data_on_merge(&source_snap.outer_start, &old_shard.outer_start);

        if clear_target {
            info!(
                "{} clear data of target shard on merge, target: {:?}",
                old_shard.tag(),
                old_shard.range,
            );
            old_shard.set_property(DEL_PREFIXES_KEY, &[]);
            old_shard.set_data(ShardData::new_empty(
                old_shard.range.clone(),
                old_shard.get_data().limiter.clone(),
            ));
        }
        let mut new_shard = self.new_shard_version(&old_shard, sequence);

        // TODO: Do we need to merge pending operations here?
        new_shard.range.outer_start = min(
            old_shard.outer_start.clone(),
            source_snap.outer_start.clone().into(),
        );
        new_shard.range.outer_end = max(
            old_shard.outer_end.clone(),
            source_snap.outer_end.clone().into(),
        );
        new_shard.ver = max(shard_ver, source.shard_ver) + 1;
        // make sure the new mem-table version is greater than source.
        let source_mem_tbl_version = source_snap.base_version + source.sequence;
        let target_mem_tbl_version = old_shard.get_base_version() + sequence;
        store_u64(
            &new_shard.base_version,
            max(source_mem_tbl_version, target_mem_tbl_version) - sequence,
        );

        let data = if !clear_source {
            if clear_target {
                // `inner_key_off` will be different when merge regions of different keyspaces.
                new_shard.range.inner_key_off = source_snap.inner_key_off as usize;
            }

            // merge source DEL_PREFIXES_KEY to new shard
            let source_del_prefixes =
                get_shard_property(DEL_PREFIXES_KEY, source_snap.get_properties())
                    .map(|v| Bytes::from(v));
            let old_del_prefixes = old_shard.get_property(DEL_PREFIXES_KEY);
            if let Some(new_del_prefixes) = merge_del_prefixes_if_needed(
                source_del_prefixes,
                old_del_prefixes,
                source_snap.inner_key_off as usize,
            ) {
                new_shard.set_property(DEL_PREFIXES_KEY, &new_del_prefixes);
            }

            // merge shard data
            let old_data = old_shard.get_data();
            let mem_tbls = old_data.mem_tbls.clone();
            let mut blob_tbl_map = old_data.blob_tbl_map.as_ref().clone();
            for v in source.blob_tables.values() {
                blob_tbl_map.insert(v.id(), v.clone());
            }
            let mut l0_tbls = old_data.l0_tbls.clone();
            for l0 in source.l0_tables.values() {
                l0_tbls.push(l0.clone())
            }
            l0_tbls.sort_by(|a, b| b.version().cmp(&a.version()));
            let mut new_cf_builders = [
                ShardCfBuilder::new(0),
                ShardCfBuilder::new(1),
                ShardCfBuilder::new(2),
            ];
            for cf in 0..NUM_CFS {
                let old_scf = old_data.get_cf(cf);
                for level in 1..=CF_LEVELS[cf] {
                    let old_level = old_scf.get_level(level);
                    let cf_builder = &mut new_cf_builders[cf];
                    for tbl in old_level.tables.as_slice() {
                        cf_builder.add_table(tbl.clone(), level);
                    }
                }
            }
            for tbl_create in source_snap.get_table_creates() {
                let tbl = source.ln_tables.get(&tbl_create.id).unwrap().clone();
                let cf_builder = &mut new_cf_builders[tbl_create.cf as usize];
                cf_builder.add_table(tbl, tbl_create.level as usize);
            }
            let new_cfs = [
                new_cf_builders[0].build(),
                new_cf_builders[1].build(),
                new_cf_builders[2].build(),
            ];
            let mut unloaded_tbls = source.unloaded_tables.clone();
            for (&id, tbl) in old_data.unloaded_tbls.iter() {
                unloaded_tbls.insert(id, tbl.clone());
            }
            let lock_txn_files = old_data.lock_txn_files.clone();
            assert!(
                source.lock_txn_files.is_empty(),
                "{} source shard with txn file locks are not allowed to be merged, source {:?}",
                old_shard.tag(),
                source
            );
            ShardData::new(
                new_shard.range.clone(),
                mem_tbls,
                l0_tbls,
                Arc::new(blob_tbl_map),
                new_cfs,
                unloaded_tbls,
                lock_txn_files,
                old_data.limiter.clone(),
            )
        } else {
            info!(
                "{} clear data of source shard on merge, source start: {}, end: {}",
                old_shard.tag(),
                log_wrappers::hex_encode_upper(&source_snap.outer_start),
                &log_wrappers::hex_encode_upper(&source_snap.outer_end),
            );

            let old_data = old_shard.get_data();
            ShardData::new(
                new_shard.range.clone(),
                old_data.mem_tbls.clone(),
                old_data.l0_tbls.clone(),
                old_data.blob_tbl_map.clone(),
                old_data.cfs.clone(),
                old_data.unloaded_tbls.clone(),
                old_data.lock_txn_files.clone(),
                old_data.limiter.clone(),
            )
        };
        new_shard.set_data(data);
        debug_assert_eq!(new_shard.range, new_shard.get_data().range);

        new_shard.parent_id = shard_id;
        let all_files = new_shard.get_all_files();
        info!(
            "merged new shard {}, start {:x}, end {:x}, all files {:?}",
            new_shard.tag(),
            new_shard.outer_start,
            new_shard.outer_end,
            all_files
        );
        self.refresh_shard_states(&new_shard);
        self.shards.insert(shard_id, Arc::new(new_shard));
        Ok(())
    }

    pub(crate) fn new_shard_version(&self, old_shard: &Shard, sequence: u64) -> Shard {
        let engine_id = self.get_engine_id();
        let new_shard = Shard::new(
            engine_id,
            &old_shard.properties.to_pb(old_shard.id),
            old_shard.ver + 1,
            old_shard.range.clone(),
            old_shard.opt.clone(),
            &self.master_key,
        );
        new_shard.set_data(old_shard.get_data());
        new_shard.set_active(old_shard.is_active());
        store_u64(&new_shard.base_version, old_shard.get_base_version());
        store_u64(&new_shard.meta_seq, sequence);
        store_u64(&new_shard.write_sequence, sequence);
        store_u64(&new_shard.estimated_size, old_shard.get_estimated_size());
        store_u64(
            &new_shard.estimated_entries,
            old_shard.get_estimated_entries(),
        );
        store_u64(&new_shard.max_ts, old_shard.get_max_ts());
        store_u64(
            &new_shard.estimated_kv_size,
            old_shard.get_estimated_kv_size(),
        );
        new_shard
    }
}

pub fn get_split_shard_index(split_keys: &[Vec<u8>], key: &[u8]) -> usize {
    for i in 0..split_keys.len() {
        if key < split_keys[i].as_slice() {
            return i;
        }
    }
    split_keys.len()
}

/// Whether to clear the data of source and/or target regions on merge.
///
/// Merging regions of different keyspaces only happens when the keyspace(s)
/// has been deleted. The validity of which was ensured by PD.
///
/// In this condition, we must clear the data of region in keyspace.
///
/// Otherwise, as SSTs in keyspace do not have prefix (when enable key
/// offset), the merged SSTs will violate data correctness.
pub fn need_clear_region_data_on_merge(
    source_outer_start: &[u8],
    target_outer_start: &[u8],
) -> (bool /* clear_source */, bool /* clear_target */) {
    let is_same_keyspace_merge =
        ApiV2::is_belongs_to_same_keyspace(source_outer_start, target_outer_start);
    let clear_region = |key: &[u8]| -> bool {
        let key_mode = ApiV2::parse_key_mode(key);
        let in_keyspace = key_mode == KeyMode::Txn || key_mode == KeyMode::Raw;
        !is_same_keyspace_merge && in_keyspace
    };
    (
        clear_region(source_outer_start),
        clear_region(target_outer_start),
    )
}

#[cfg(test)]
mod tests {
    use std::iter::Iterator;

    use super::*;

    #[test]
    fn test_need_clear_region_data_on_merge() {
        let cases: Vec<(&[u8], &[u8], (bool, bool))> = vec![
            // Txn
            (
                b"x0000",       // source_outer_start
                b"x0001",       // target_outer_start
                (false, false), // (clear_source, clear_target)
            ),
            (b"x0000", b"x0010", (true, true)),
            // Raw
            (b"r0000", b"r0001", (false, false)),
            (b"r0000", b"r0010", (true, true)),
            // Txn & Raw
            (b"x0000", b"r0000", (true, true)),
            // Txn & TiDB/Unknown
            (b"x0000", b"t", (true, false)),
            (b"t", b"x0000", (false, true)),
            (b"x0000", b"m", (true, false)),
            (b"m", b"x0000", (false, true)),
            (b"x0000", b"", (true, false)),
            (b"", b"x0000", (false, true)),
            // TiDB/Unknown
            (b"t", b"m", (false, false)),
            (b"m", b"", (false, false)),
            (b"", b"t", (false, false)),
        ];

        for (idx, (source_outer_start, target_outer_start, expected)) in
            cases.into_iter().enumerate()
        {
            let res = need_clear_region_data_on_merge(source_outer_start, target_outer_start);
            assert_eq!(res, expected, "case {}", idx);
        }
    }
}
