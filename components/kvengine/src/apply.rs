// Copyright 2021 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    collections::{HashMap, HashSet},
    fmt::{Debug, Formatter},
    iter::Iterator,
    ops::Deref,
    sync::Arc,
};

use bytes::{Buf, Bytes};
use cloud_encryption::EncryptionKey;
use kvenginepb as pb;
use moka::sync::SegmentedCache;

use crate::{
    meta::is_move_down,
    table::{
        blobtable::blobtable::BlobTable,
        columnar::{ColumnarFile, SchemaFile},
        file::LocalFile,
        sstable::{BlockCacheKey, L0Table, SsTable},
        InnerKey, TableExt, TxnFile,
    },
    *,
};

pub struct ChangeSet {
    pub change_set: kvenginepb::ChangeSet,
    pub l0_tables: HashMap<u64, L0Table>,
    pub ln_tables: HashMap<u64, SsTable>,
    pub blob_tables: HashMap<u64, BlobTable>,
    /// Tables that are not loaded from DFS.
    pub unloaded_tables: HashMap<u64, FileMeta>,
    pub lock_txn_files: Vec<TxnFile>,
    pub schema_file: Option<SchemaFile>,
    pub col_files: HashMap<u64, ColumnarFile>,
}

impl Deref for ChangeSet {
    type Target = kvenginepb::ChangeSet;

    fn deref(&self) -> &Self::Target {
        &self.change_set
    }
}

impl Debug for ChangeSet {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.change_set.fmt(f)
    }
}

impl ChangeSet {
    pub fn new(change_set: kvenginepb::ChangeSet) -> Self {
        Self {
            change_set,
            l0_tables: HashMap::new(),
            ln_tables: HashMap::new(),
            blob_tables: HashMap::new(),
            unloaded_tables: HashMap::new(),
            lock_txn_files: vec![],
            schema_file: None,
            col_files: HashMap::new(),
        }
    }

    pub fn add_file(
        &mut self,
        id: u64,
        file: LocalFile,
        meta: &FileMeta,
        cache: SegmentedCache<BlockCacheKey, Bytes>,
        encryption_key: Option<EncryptionKey>,
    ) -> Result<()> {
        if meta.is_blob_file() {
            let blob_table = BlobTable::new(Arc::new(file))?;
            self.blob_tables.insert(id, blob_table);
        } else if meta.is_columnar_file() {
            self.col_files
                .insert(id, ColumnarFile::open(Arc::new(file))?);
        } else if meta.get_level() == 0 {
            let l0_table = L0Table::new(Arc::new(file), Some(cache), false, encryption_key)?;
            self.l0_tables.insert(id, l0_table.unwrap());
        } else {
            let ln_table = SsTable::new(
                Arc::new(file),
                Some(cache),
                meta.get_level() == 1,
                encryption_key,
            )?;
            self.ln_tables.insert(id, ln_table);
        }
        Ok(())
    }
}

// `not_all_tables_loaded` means that some tables in `snap` are not loaded to
// `tables`. Should only happen in "ignore lock" or restoration (`for_restore`).
pub(crate) fn create_snapshot_tables(
    builder: &mut ShardDataBuilder,
    snap: &kvenginepb::Snapshot,
    tables: &ChangeSet,
    not_all_tables_loaded: bool,
) {
    // Note: Some tables in `snap` will not exist in `tables` if it's not necessary
    // to load from DFS.
    // Should only happen in restoration.
    let blob_creates = snap.get_blob_creates();
    let mut blob_tbl_map = HashMap::new();

    if !blob_creates.is_empty() {
        for blob_create in blob_creates {
            if let Some(blob_tbl) = tables.blob_tables.get(&blob_create.id) {
                blob_tbl_map.insert(blob_create.id, blob_tbl.clone());
            } else {
                assert!(
                    not_all_tables_loaded,
                    "blob_create: {:?}, tables: {:?}",
                    blob_create, tables,
                );
            }
        }
    };
    let mut l0_tbls = vec![];
    for l0_create in snap.get_l0_creates() {
        if let Some(l0_tbl) = tables.l0_tables.get(&l0_create.id) {
            l0_tbls.push(l0_tbl.clone());
        } else {
            assert!(
                not_all_tables_loaded,
                "l0_create: {:?}, tables: {:?}",
                l0_create, tables,
            );
        }
    }
    l0_tbls.sort_by(|a, b| b.version().cmp(&a.version()));

    let mut scf_builders = vec![];
    for cf in 0..NUM_CFS {
        let scf = ShardCfBuilder::new(cf);
        scf_builders.push(scf);
    }
    for table_create in snap.get_table_creates() {
        if let Some(tbl) = tables.ln_tables.get(&table_create.id) {
            let scf = &mut scf_builders.as_mut_slice()[table_create.cf as usize];
            scf.add_table(tbl.clone(), table_create.level as usize);
        } else {
            assert!(
                not_all_tables_loaded,
                "table_create: {:?}, tables: {:?}",
                table_create, tables,
            );
        }
    }
    let mut scfs = [ShardCf::new(0), ShardCf::new(1), ShardCf::new(2)];
    for cf in 0..NUM_CFS {
        let scf = &mut scf_builders.as_mut_slice()[cf];
        scfs[cf] = scf.build();
    }
    let mut col_levels = ColumnarLevels::new();
    for col_create in snap.get_columnar_creates() {
        if let Some(col_file) = tables.col_files.get(&col_create.id) {
            col_levels.add_file(col_create.level as usize, col_file.clone());
        } else {
            assert!(
                not_all_tables_loaded,
                "columnar_create: {:?}, tables: {:?}",
                col_create, tables,
            );
        }
    }
    for &l0_id in snap.get_unconverted_l0s() {
        let unconverted_l0 = l0_tbls.iter().find(|l0| l0.id() == l0_id).unwrap().clone();
        col_levels.unconverted_l0s.push(unconverted_l0);
    }
    col_levels.sort();
    builder.set_l0_tbls(l0_tbls);
    builder.set_blob_tbls(blob_tbl_map);
    builder.set_cfs(scfs);
    builder.set_lock_txn_files(tables.lock_txn_files.clone());
    builder.set_columnar_levels(col_levels);
}

impl EngineCore {
    pub fn apply_change_set(&self, cs: ChangeSet) -> Result<()> {
        let shard = self.get_shard(cs.shard_id);
        if shard.is_none() {
            return Err(Error::ShardNotFound);
        }
        let shard = shard.unwrap();
        info!(
            "{} kvengine apply change set sequence: {}",
            shard.tag(),
            cs.sequence
        );
        if shard.ver != cs.shard_ver {
            warn!(
                "{} kvengine::apply_change_set: shard not match, shard {}, cs {}, {:?}",
                shard.tag(),
                shard.ver,
                cs.shard_ver,
                cs
            );
            return Err(Error::ShardNotMatch);
        }
        let seq = load_u64(&shard.meta_seq);
        if seq >= cs.sequence {
            warn!(
                "{} skip duplicated shard seq:{}, change seq:{}",
                shard.tag(),
                seq,
                cs.sequence
            );
            return Ok(());
        }
        if cs.has_flush() {
            self.apply_flush(&shard, &cs);
        } else if cs.has_compaction()
            || cs.has_destroy_range()
            || cs.has_truncate_ts()
            || cs.has_trim_over_bound()
            || cs.has_major_compaction()
            || cs.has_columnar_compaction()
        {
            if cs.has_compaction() {
                self.apply_compaction(&shard, &cs);
            } else if cs.has_destroy_range() {
                self.apply_destroy_range(&shard, &cs);
            } else if cs.has_truncate_ts() {
                self.apply_truncate_ts(&shard, &cs);
            } else if cs.has_trim_over_bound() {
                self.apply_trim_over_bound(&shard, &cs);
            } else if cs.has_major_compaction() {
                self.apply_major_compaction(&shard, &cs);
            } else if cs.has_columnar_compaction() {
                self.apply_columnar_compaction(&shard, &cs);
            }
            store_bool(&shard.compacting, false);
            self.send_compact_msg(CompactMsg::Applied(IdVer::new(shard.id, shard.ver)));
        } else if cs.has_initial_flush() {
            self.apply_initial_flush(&shard, &cs);
        } else if cs.has_ingest_files() {
            self.apply_ingest_files(&shard, &cs)?;
        } else if cs.has_restore_shard() {
            self.apply_restore_shard(&shard, &cs)?;
        } else if cs.has_update_schema_meta() {
            self.apply_update_schema_meta(&shard, &cs);
        }
        debug!("{} finished applying change set: {:?}", shard.tag(), cs);

        // Get shard again as version may be changed after change set applied.
        // Note that the shard may have been destroyed (e.g. by merge, in rfstore
        // thread) at this point.
        if let Some(shard) = self.get_shard(cs.shard_id) {
            self.refresh_shard_states(&shard);
            // Since shard maybe replaced, we update the meta_seq here to avoid the
            // the case that the old shard's meta_sequence mismatch the old shard data.
            store_u64(&shard.meta_seq, cs.sequence);
        }

        Ok(())
    }

    fn apply_flush(&self, shard: &Shard, cs: &ChangeSet) {
        let flush = cs.get_flush();
        let old_data = shard.get_data();
        let mut new_mem_tbls = old_data.mem_tbls.clone();
        if flush.has_l0_create() || !flush.get_l0_creates().is_empty() {
            let mut l0s = vec![];
            if flush.has_l0_create() {
                let l0_id = flush.get_l0_create().get_id();
                let l0_tbl = cs.l0_tables.get(&l0_id).unwrap().clone();
                l0s.push(l0_tbl);
            }
            for l0_create in flush.get_l0_creates() {
                let l0_tbl = cs.l0_tables.get(&l0_create.get_id()).unwrap().clone();
                l0s.push(l0_tbl);
            }
            let l0_version = l0s.first().unwrap().version();
            let last = new_mem_tbls.pop().unwrap();
            let last_version = last.get_version();
            if last_version != l0_version {
                panic!(
                    "{} mem table last version {}, size {} not match L0 version {}, shard meta seq {}, flush seq {}",
                    shard.tag(),
                    last_version,
                    last.size(),
                    l0_version,
                    shard.get_meta_sequence(),
                    cs.sequence,
                );
            }
            let mut new_l0_tbls = Vec::with_capacity(old_data.l0_tbls.len() + l0s.len());
            new_l0_tbls.extend_from_slice(l0s.as_slice());
            new_l0_tbls.extend_from_slice(old_data.l0_tbls.as_slice());
            let mut col_levels = old_data.col_levels.clone();
            if old_data.schema_file.is_some() && shard.get_columnar_snap_version() > 0 {
                col_levels.unconverted_l0s.extend_from_slice(l0s.as_slice());
            }
            let mut builder = ShardDataBuilder::new(old_data);
            builder.set_mem_tbls(new_mem_tbls);
            builder.set_l0_tbls(new_l0_tbls);
            builder.set_columnar_levels(col_levels);
            shard.set_data(builder.build());
            self.send_free_mem_msg(FreeMemMsg::FreeMem(last));
        } else {
            // If there is no L0Create, it means the mem-table is empty during flush.
            // It's possible that the mem-table is not empty but doesn't have any data.
            // So we need to use has_data_in_range to check.
            let last = new_mem_tbls.last().unwrap();
            if new_mem_tbls.len() > 1
                && !last.has_data_in_range(shard.inner_start(), shard.inner_end())
            {
                let last = new_mem_tbls.pop().unwrap();
                let mut builder = ShardDataBuilder::new(old_data);
                builder.set_mem_tbls(new_mem_tbls);
                shard.set_data(builder.build());
                self.send_free_mem_msg(FreeMemMsg::FreeMem(last));
            }
        }
        store_u64(&shard.snap_version, flush.get_version());
        shard.clear_finished_txn_file_refs(flush.version);
    }

    fn apply_initial_flush(&self, shard: &Shard, cs: &ChangeSet) {
        let initial_flush = cs.get_initial_flush();
        let data = shard.get_data();
        let mut mem_tbls = data.mem_tbls.clone();

        let mut builder = ShardDataBuilder::new(data.clone());
        create_snapshot_tables(&mut builder, initial_flush, cs, self.opts.for_restore);
        // `lock_txn_files` is ignored because it does not depend on initial flush to
        // keep consistency between peers, as only target region has txn file locks.
        builder.set_lock_txn_files(data.lock_txn_files.clone());
        let mut max_flushed_mem_tbl_version = 0;
        mem_tbls.retain(|x| {
            let version = x.get_version();
            let flushed =
                version > 0 && version <= initial_flush.base_version + initial_flush.data_sequence;
            if flushed {
                if max_flushed_mem_tbl_version < version {
                    max_flushed_mem_tbl_version = version;
                }
                self.send_free_mem_msg(FreeMemMsg::FreeMem(x.clone()));
            }
            !flushed
        });
        builder.set_mem_tbls(mem_tbls);
        let new_data = builder.build();
        info!("{} apply_initial_flush", shard.tag();
            "seq" => cs.sequence,
            "lock_txn_files" => ?new_data.lock_txn_files);
        shard.set_data(new_data);
        shard.clear_finished_txn_file_refs(max_flushed_mem_tbl_version);

        store_bool(&shard.initial_flushed, true);
        // Switched memtables can't be flushed until initial flush finished, so we
        // trigger it actively.
        if let Err(err) = self.trigger_flush(shard) {
            warn!("{} trigger_flush error: {:?}", shard.tag(), err);
        }
    }

    fn apply_compaction(&self, shard: &Shard, cs: &ChangeSet) {
        let comp = cs.get_compaction();
        let mut del_files = HashMap::new();
        if comp.conflicted {
            if is_move_down(comp) {
                return;
            }
            for create in comp.get_table_creates() {
                let cover = shard.cover_full_table(
                    InnerKey::from_inner_buf(&create.smallest),
                    InnerKey::from_inner_buf(&create.biggest),
                );
                del_files.insert(create.id, cover);
            }
            self.remove_dfs_files(shard, del_files);
            return;
        }
        let data = shard.get_data();
        let mut new_cfs = data.cfs.clone();
        let mut new_l0s = data.l0_tbls.clone();
        let mut new_blob_tbl_map = data.blob_tbl_map.as_ref().clone();
        assert!(!is_blob_file(comp.level));
        if comp.level == 0 {
            let is_move_down = is_move_down(comp);
            new_l0s.retain(|x| {
                let is_deleted = comp.get_top_deletes().contains(&x.id());
                if is_deleted && !is_move_down {
                    del_files.insert(x.id(), shard.cover_full_table(x.smallest(), x.biggest()));
                }
                !is_deleted
            });
            new_blob_tbl_map.extend(cs.blob_tables.clone());
            for cf in 0..NUM_CFS {
                let new_l1 = self.new_level(shard, cs, &data, cf, &mut del_files, true);
                new_cfs[cf].set_level(new_l1);
            }
        } else {
            let cf = comp.cf as usize;
            let new_top_level = self.new_level(shard, cs, &data, cf, &mut del_files, false);
            new_cfs[cf].set_level(new_top_level);
            let new_bottom_level = self.new_level(shard, cs, &data, cf, &mut del_files, true);
            new_cfs[cf].set_level(new_bottom_level);
            // For move down operation, the TableCreates may contains TopDeletes, we don't
            // want to delete them.
            for create in comp.get_table_creates() {
                del_files.remove(&create.id);
            }
        }
        let mut builder = ShardDataBuilder::new(data);
        builder.set_l0_tbls(new_l0s);
        builder.set_blob_tbls(new_blob_tbl_map);
        builder.set_cfs(new_cfs);
        shard.set_data(builder.build());
        self.remove_dfs_files(shard, del_files);
    }

    fn apply_major_compaction(&self, shard: &Shard, cs: &ChangeSet) {
        let comp = cs.get_major_compaction();
        let mut del_file_is_subrange = HashMap::new();
        if comp.conflicted {
            for sst_create in comp.get_sstable_change().get_table_creates() {
                let is_subrange = shard.cover_full_table(
                    InnerKey::from_inner_buf(sst_create.get_smallest()),
                    InnerKey::from_inner_buf(sst_create.get_biggest()),
                );
                del_file_is_subrange.insert(sst_create.get_id(), is_subrange);
            }
            for blob_tbl_create in comp.get_new_blob_tables() {
                let is_subrange = shard.cover_full_table(
                    InnerKey::from_inner_buf(blob_tbl_create.get_smallest()),
                    InnerKey::from_inner_buf(blob_tbl_create.get_biggest()),
                );
                del_file_is_subrange.insert(blob_tbl_create.get_id(), is_subrange);
            }
            self.remove_dfs_files(shard, del_file_is_subrange);
            return;
        }
        let mut to_be_deleted = HashSet::new();
        let data = shard.get_data();

        for del in comp.get_sstable_change().get_table_deletes() {
            to_be_deleted.insert(del.get_id());
        }
        for del in comp.get_old_blob_tables() {
            to_be_deleted.insert(*del);
        }
        let mut new_blob_tbl_map = data.blob_tbl_map.as_ref().clone();
        let mut new_l0s = data.l0_tbls.clone();
        let mut new_cfs = data.cfs.clone();
        new_l0s.retain(|x| {
            let del = to_be_deleted.contains(&x.id());
            if del {
                del_file_is_subrange
                    .insert(x.id(), shard.cover_full_table(x.smallest(), x.biggest()));
            }
            !del
        });
        new_blob_tbl_map.retain(|id, v| {
            let del = to_be_deleted.contains(id);
            if del {
                del_file_is_subrange.insert(
                    *id,
                    shard.cover_full_table(v.smallest_key(), v.biggest_key()),
                );
            }
            !del
        });

        for new_blob_table in comp.get_new_blob_tables() {
            new_blob_tbl_map.insert(
                new_blob_table.get_id(),
                cs.blob_tables
                    .get(&new_blob_table.get_id())
                    .unwrap()
                    .clone(),
            );
        }
        for cf in 0..NUM_CFS {
            for level in 1..=CF_LEVELS[cf] {
                let mut tables = new_cfs[cf].get_level(level).tables.as_ref().clone();
                tables.retain(|x| {
                    let del = to_be_deleted.contains(&x.id());
                    if del {
                        del_file_is_subrange
                            .insert(x.id(), shard.cover_full_table(x.smallest(), x.biggest()));
                    }
                    !del
                });
                if level == CF_LEVELS[cf] {
                    for new_sstable in comp.get_sstable_change().get_table_creates() {
                        if new_sstable.get_cf() as usize == cf {
                            assert_eq!(new_sstable.get_level() as usize, CF_LEVELS[cf]);
                            tables.push(cs.ln_tables.get(&new_sstable.get_id()).unwrap().clone());
                        }
                    }
                }
                tables.sort_by(|a, b| a.smallest().cmp(&b.smallest()));
                let lh = LevelHandler::new(level, tables);
                new_cfs[cf].set_level(lh);
            }
        }
        let mut builder = ShardDataBuilder::new(data);
        builder.set_l0_tbls(new_l0s);
        builder.set_blob_tbls(new_blob_tbl_map);
        builder.set_cfs(new_cfs);
        shard.set_data(builder.build());
        self.remove_dfs_files(shard, del_file_is_subrange);
        shard.set_property(MANUAL_MAJOR_COMPACTION, MANUAL_MAJOR_COMPACTION_DISABLE);
    }

    fn get_tables_from_table_change(
        &self,
        buidler: &mut ShardDataBuilder,
        data: &ShardData,
        cs: &ChangeSet,
        tc: &pb::TableChange,
        del_files: &mut HashMap<u64, bool>,
    ) {
        let mut new_l0s = data.l0_tbls.clone();
        let mut new_cfs = data.cfs.clone();
        let mut new_col_levels = data.col_levels.clone();
        // Group files by cf and level.
        let mut grouped = HashMap::new();
        let mut all_deletes = HashSet::new();
        for deleted in tc.get_table_deletes() {
            grouped
                .entry((
                    deleted.get_cf(),
                    deleted.get_level() as usize,
                    deleted.columnar_tables > 0,
                ))
                .or_insert_with(|| (Vec::new(), Vec::new()))
                .0
                .push(deleted.get_id());
            all_deletes.insert(deleted.get_id());
        }
        for col_level in &mut new_col_levels.levels {
            col_level
                .files
                .retain(|c| !all_deletes.contains(&c.get_file().id()));
        }
        for created in tc.get_table_creates() {
            if created.columnar_tables > 0 {
                let col_file = cs.col_files.get(&created.get_id()).unwrap().clone();
                new_col_levels.add_file(created.get_level() as usize, col_file);
                continue;
            }
            grouped
                .entry((
                    created.get_cf(),
                    created.get_level() as usize,
                    created.columnar_tables > 0,
                ))
                .or_insert_with(|| (Vec::new(), Vec::new()))
                .1
                .push(created.get_id());
        }
        new_col_levels.sort();

        for ((cf, level, is_columnar), (deletes, creates)) in grouped {
            if is_columnar {
                new_col_levels.levels[level]
                    .files
                    .retain(|col| !deletes.contains(&col.id()));
                continue;
            }
            if level == 0 {
                new_l0s.retain(|l0| {
                    let is_deleted = deletes.contains(&l0.id());
                    if is_deleted {
                        del_files
                            .insert(l0.id(), data.cover_full_table(l0.smallest(), l0.biggest()));
                    }
                    !is_deleted
                });
                new_l0s.extend(
                    creates
                        .clone()
                        .into_iter()
                        .map(|id| cs.l0_tables.get(&id).unwrap().clone()),
                );
                new_l0s.sort_by(|a, b| b.version().cmp(&a.version()));
            } else {
                let old_level = new_cfs[cf as usize].get_level(level);
                let mut new_level_tables = old_level.tables.as_ref().clone();
                new_level_tables.retain(|t| {
                    let is_deleted = deletes.contains(&t.id());
                    if is_deleted {
                        del_files.insert(t.id(), data.cover_full_table(t.smallest(), t.biggest()));
                    }
                    !is_deleted
                });
                new_level_tables.extend(
                    creates
                        .into_iter()
                        .map(|id| cs.ln_tables.get(&id).unwrap().clone()),
                );
                new_level_tables.sort_by(|a, b| a.smallest().cmp(&b.smallest()));
                let new_level = LevelHandler::new(level, new_level_tables);
                new_cfs[cf as usize].set_level(new_level);
            }
        }
        buidler.set_l0_tbls(new_l0s);
        buidler.set_cfs(new_cfs);
        buidler.set_columnar_levels(new_col_levels);
    }

    fn apply_destroy_range(&self, shard: &Shard, cs: &ChangeSet) {
        assert!(cs.has_destroy_range());
        let data = shard.get_data();
        let tc = cs.get_destroy_range();
        let mut del_files = HashMap::new();
        let mut builder = ShardDataBuilder::new(data.clone());
        self.get_tables_from_table_change(&mut builder, &data, cs, tc, &mut del_files);
        assert_eq!(cs.get_property_key(), DEL_PREFIXES_KEY);
        let done = DeletePrefixes::unmarshal(cs.get_property_value(), shard.inner_key_off);
        shard.set_data(builder.build());
        let del_prefixes = shard.get_del_prefixes();
        let new_del_prefixes = del_prefixes.split(&done);
        shard.set_property(DEL_PREFIXES_KEY, &new_del_prefixes.marshal());
        self.remove_dfs_files(shard, del_files);
    }

    fn apply_truncate_ts(&self, shard: &Shard, cs: &ChangeSet) {
        debug!("apply truncate ts in engine, shard {:?}", shard.tag());
        assert!(cs.has_truncate_ts());
        let data = shard.get_data();
        let tc = cs.get_truncate_ts();
        let mut del_files = HashMap::new();
        let mut builder = ShardDataBuilder::new(data.clone());
        self.get_tables_from_table_change(&mut builder, &data, cs, tc, &mut del_files);
        assert_eq!(cs.get_property_key(), TRUNCATE_TS_KEY);
        shard.set_data(builder.build());
        let truncated_ts = TruncateTs::unmarshal(cs.get_property_value());
        // if applied truncate_ts is smaller than truncate ts in shard, remove it.
        let old_truncated_ts = shard.get_truncate_ts();
        if need_update_truncate_ts(old_truncated_ts, truncated_ts) {
            shard.set_property(TRUNCATE_TS_KEY, b"");
        }
        self.remove_dfs_files(shard, del_files);
    }

    fn apply_trim_over_bound(&self, shard: &Shard, cs: &ChangeSet) {
        debug!("{} apply changeset.trim_over_bound in engine", shard.tag());
        assert!(cs.has_trim_over_bound());
        let data = shard.get_data();
        let tc = cs.get_trim_over_bound();
        let mut del_files = HashMap::new();
        let mut builder = ShardDataBuilder::new(data.clone());
        self.get_tables_from_table_change(&mut builder, &data, cs, tc, &mut del_files);
        shard.set_data(builder.build());
        shard.set_property(TRIM_OVER_BOUND, TRIM_OVER_BOUND_DISABLE);
        self.remove_dfs_files(shard, del_files);
    }

    fn new_level(
        &self,
        shard: &Shard,
        cs: &ChangeSet,
        data: &ShardData,
        cf: usize,
        del_files: &mut HashMap<u64, bool>,
        is_bottom: bool,
    ) -> LevelHandler {
        let old_scf = data.get_cf(cf);
        let comp = cs.get_compaction();
        let level = if is_bottom {
            comp.get_level() as usize + 1
        } else {
            comp.get_level() as usize
        };
        let deletes = if is_bottom {
            comp.get_bottom_deletes()
        } else {
            comp.get_top_deletes()
        };
        let mut new_level_tables = old_scf.get_level(level).tables.as_ref().clone();
        new_level_tables.retain(|x| {
            let is_deleted = deletes.contains(&x.id());
            if is_deleted {
                del_files.insert(x.id(), shard.cover_full_table(x.smallest(), x.biggest()));
            }
            !is_deleted
        });
        if is_bottom {
            for new_tbl_create in comp.get_table_creates() {
                if new_tbl_create.cf as usize == cf {
                    let new_tbl = if is_move_down(comp) {
                        if comp.level == 0 {
                            let old_l0 = data
                                .l0_tbls
                                .iter()
                                .find(|l0| l0.id() == new_tbl_create.id)
                                .unwrap();
                            old_l0.get_cf(WRITE_CF).clone().unwrap()
                        } else {
                            let old_top_level = old_scf.get_level(level - 1);
                            old_top_level.get_table_by_id(new_tbl_create.id).unwrap()
                        }
                    } else {
                        cs.ln_tables.get(&new_tbl_create.get_id()).unwrap().clone()
                    };
                    new_level_tables.push(new_tbl);
                }
            }
            new_level_tables.sort_by(|a, b| a.smallest().cmp(&b.smallest()));
        }
        let new_level = LevelHandler::new(level, new_level_tables);
        new_level.check_order(cf, shard.tag());
        new_level
    }

    fn remove_dfs_files(&self, shard: &Shard, del_files: HashMap<u64, bool>) {
        if !shard.is_active() {
            return;
        }
        for (id, _cover) in del_files {
            self.set_local_file_mtime(id);
        }
    }

    // On remove local file, we need to retain the file for a while as SnapAccess
    // hold the file may reopen it, update the mtime so local file gc worker
    // will delay the remove.
    fn set_local_file_mtime(&self, file_id: u64) {
        let path = self.local_sst_file_path(file_id);
        if let Err(err) = filetime::set_file_mtime(path, filetime::FileTime::now()) {
            error!("failed to set local file mtime {} {:?}", file_id, err);
        }
    }

    fn _local_file_len(&self, file_id: u64) -> Option<u64> {
        let local_file_path = self.local_sst_file_path(file_id);
        match std::fs::metadata(local_file_path) {
            Ok(metadata) => Some(metadata.len()),
            Err(err) => {
                error!("failed to get local file len {:?}", err);
                None
            }
        }
    }

    fn apply_ingest_files(&self, shard: &Shard, cs: &ChangeSet) -> Result<()> {
        let ingest_files = cs.get_ingest_files();
        if let (Some(ingest_id), Some(old_ingest_id)) = (
            get_shard_property(INGEST_ID_KEY, ingest_files.get_properties()),
            shard.get_property(INGEST_ID_KEY),
        ) {
            if old_ingest_id.chunk() == ingest_id.as_slice() {
                // skip duplicated ingest files.
                return Ok(());
            }
        }
        let old_data = shard.get_data();
        let mut new_blob_tbl_map = old_data.blob_tbl_map.as_ref().clone();
        for blob_table_create in ingest_files.get_blob_creates() {
            let id = blob_table_create.get_id();
            new_blob_tbl_map.insert(id, cs.blob_tables.get(&id).unwrap().clone());
        }
        let mut new_l0s = old_data.l0_tbls.clone();
        for l0_create in ingest_files.get_l0_creates() {
            let l0_table = cs.l0_tables.get(&l0_create.get_id()).unwrap().clone();
            new_l0s.push(l0_table);
        }
        new_l0s.sort_unstable_by(|a, b| b.version().cmp(&a.version()));
        let mut scf_builder = ShardCfBuilder::new(0);
        for level in &old_data.cfs[0].levels {
            for old_tbl in level.tables.as_ref() {
                scf_builder.add_table(old_tbl.clone(), level.level);
            }
        }
        for tbl_create in ingest_files.get_table_creates() {
            let table = cs.ln_tables.get(&tbl_create.get_id()).unwrap().clone();
            scf_builder.add_table(table, tbl_create.level as usize);
        }
        let new_cf = scf_builder.build();
        let mut new_cfs = old_data.cfs.clone();
        new_cfs[0] = new_cf;
        let mut builder = ShardDataBuilder::new(old_data);
        builder.set_l0_tbls(new_l0s);
        builder.set_blob_tbls(new_blob_tbl_map);
        builder.set_cfs(new_cfs);
        shard.set_data(builder.build());
        Ok(())
    }

    fn apply_restore_shard(&self, old_shard: &Shard, cs: &ChangeSet) -> Result<()> {
        debug_assert!(cs.has_restore_shard());
        // TODO: skip duplicated.

        let snap = cs.get_restore_shard();
        assert_eq!(old_shard.outer_start, snap.outer_start);
        assert_eq!(old_shard.outer_end, snap.outer_end);

        self.send_flush_msg(FlushMsg::Clear(old_shard.id));
        self.send_compact_msg(CompactMsg::Clear(IdVer::new(old_shard.id, old_shard.ver)));

        let range = ShardRange::from_snap(snap);
        // Increase shard version to make change sets generated before restore shard
        // stale.
        let new_shard = Shard::new(
            self.get_engine_id(),
            snap.get_properties(),
            cs.shard_ver + 1,
            range,
            old_shard.opt.clone(),
            &self.master_key,
        );
        let snap_data = new_shard.get_data();
        let mut builder = ShardDataBuilder::new(snap_data);
        create_snapshot_tables(
            &mut builder,
            cs.get_restore_shard(),
            cs,
            self.opts.for_restore,
        );
        builder.set_schema_file(cs.schema_file.clone());
        new_shard.set_data(builder.build());
        new_shard.set_active(old_shard.is_active());

        store_u64(&new_shard.base_version, snap.base_version);
        store_u64(&new_shard.meta_seq, cs.sequence);
        store_u64(&new_shard.write_sequence, cs.sequence);
        debug_assert!(!cs.has_parent());
        store_bool(&new_shard.initial_flushed, true);

        let mut old_mem_tbls = old_shard.get_data().mem_tbls.clone();
        for mem_tbl in old_mem_tbls.drain(..) {
            self.send_free_mem_msg(FreeMemMsg::FreeMem(mem_tbl));
        }

        self.refresh_shard_states(&new_shard);
        info!(
            "restore shard {} mem_table_version {}, change {:?}",
            new_shard.tag(),
            new_shard.load_mem_table_version(),
            &cs,
        );
        self.shards.insert(new_shard.id, Arc::new(new_shard));

        Ok(())
    }

    fn apply_update_schema_meta(&self, shard: &Shard, cs: &ChangeSet) {
        let old_data = shard.get_data();
        let mut builder = ShardDataBuilder::new(old_data);
        builder.set_schema_file(cs.schema_file.clone());
        shard.set_data(builder.build());
    }

    fn apply_columnar_compaction(&self, shard: &Shard, cs: &ChangeSet) {
        let col_comp = cs.get_columnar_compaction();
        let col_change = col_comp.get_columnar_change();
        let old_data = shard.get_data();
        let mut new_col_levels = old_data.col_levels.clone();
        let deletes: HashSet<u64> = col_change
            .get_table_deletes()
            .iter()
            .map(|del| del.get_id())
            .collect();
        new_col_levels.retain(|c| !deletes.contains(&c.get_file().id()));
        for create in col_change.get_table_creates() {
            let col_file = cs.col_files.get(&create.get_id()).unwrap().clone();
            new_col_levels.add_file(create.get_level() as usize, col_file);
        }
        new_col_levels.sort();
        new_col_levels
            .unconverted_l0s
            .retain(|l0| !col_comp.row_l0s.contains(&l0.id()));
        if shard.get_columnar_snap_version() == 0 {
            let new_flushed_l0_tbls: Vec<L0Table> = old_data
                .l0_tbls
                .iter()
                .filter(|tbl| !col_comp.get_row_l0s().contains(&tbl.id()))
                .cloned()
                .collect();
            new_col_levels.unconverted_l0s.extend(new_flushed_l0_tbls);
        }
        let schema_file = if col_comp.get_snap_version() == 0 {
            new_col_levels.unconverted_l0s.clear();
            None
        } else {
            old_data.schema_file.clone()
        };
        let mut builder = ShardDataBuilder::new(old_data);
        builder.set_schema_file(schema_file);
        builder.set_columnar_levels(new_col_levels);
        shard.set_data(builder.build());
        store_u64(&shard.col_snap_version, col_comp.get_snap_version());
    }
}
