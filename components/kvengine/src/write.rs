// Copyright 2021 TiKV Project Authors. Licensed under Apache-2.0.

use std::{cmp, collections::HashMap, iter::Iterator};

use bytes::{Buf, BytesMut};
use kvenginepb::{TxnFileRef, TxnFileRefs};
use protobuf::Message;
use slog_global::info;

use crate::{
    table::{self, memtable, InnerKey, TxnFile},
    util::TxnFileRefPropertyHelper,
    *,
};

pub struct WriteBatch {
    shard_id: u64,
    cf_batches: [memtable::WriteBatch; NUM_CFS],
    properties: HashMap<String, BytesMut>,
    sequence: u64,
    switch_mem_table: bool,
    inner_key_off: usize,
}

impl WriteBatch {
    pub fn new(shard_id: u64, inner_key_off: usize) -> Self {
        let cf_batches = [
            memtable::WriteBatch::new(),
            memtable::WriteBatch::new(),
            memtable::WriteBatch::new(),
        ];
        Self {
            shard_id,
            cf_batches,
            properties: HashMap::new(),
            sequence: 1,
            switch_mem_table: false,
            inner_key_off,
        }
    }

    pub fn put(
        &mut self,
        cf: usize,
        key: &[u8],
        val: &[u8],
        meta: u8,
        user_meta: &[u8],
        version: u64,
    ) {
        self.validate_version(cf, version);
        let inner_key = InnerKey::from_outer_key(key, self.inner_key_off);
        self.get_cf_mut(cf)
            .put(inner_key, meta, user_meta, version, val);
    }

    fn validate_version(&self, cf: usize, version: u64) {
        if CF_MANAGED[cf] {
            if version == 0 {
                panic!("version is zero for managed CF")
            }
        } else if version != 0 {
            panic!("version is not zero for not managed CF")
        }
    }

    pub fn delete(&mut self, cf: usize, key: &[u8], version: u64) {
        self.validate_version(cf, version);
        let inner_key = InnerKey::from_outer_key(key, self.inner_key_off);
        self.get_cf_mut(cf)
            .put(inner_key, table::BIT_DELETE, &[], version, &[]);
    }

    pub fn set_property(&mut self, key: &str, val: &[u8]) {
        self.properties.insert(key.to_string(), BytesMut::from(val));
    }

    pub fn get_property(&self, key: &str) -> Option<&BytesMut> {
        self.properties.get(key)
    }

    pub fn set_sequence(&mut self, seq: u64) {
        self.sequence = seq;
    }

    pub fn set_switch_mem_table(&mut self) {
        self.switch_mem_table = true;
    }

    pub fn num_entries(&self) -> usize {
        let mut num = 0;
        for wb in &self.cf_batches {
            num += wb.len();
        }
        num
    }

    pub fn reset(&mut self) {
        for wb in &mut self.cf_batches {
            wb.reset();
        }
        self.sequence = 0;
        self.properties.clear();
        self.switch_mem_table = false;
    }

    pub fn get_cf_mut(&mut self, cf: usize) -> &mut memtable::WriteBatch {
        &mut self.cf_batches[cf]
    }

    pub fn cf_len(&self, cf: usize) -> usize {
        self.cf_batches[cf].len()
    }
}

impl Engine {
    // `force`: Should be set to `true` during split/merge, to help initial flush
    // get the properties need to be flush.
    // See https://github.com/tidbcloud/cloud-storage-engine/issues/1553.
    pub fn switch_mem_table(&self, shard: &Shard, version: u64, force: bool) {
        let data = shard.get_data();
        let mem_table = data.get_writable_mem_table();
        if !force && mem_table.size() == 0 {
            return;
        }
        mem_table.set_version(version);
        if force {
            mem_table.set_force_switch();
        }
        let new_tbl = memtable::CfTable::new();
        let mut new_mem_tbls = Vec::with_capacity(data.mem_tbls.len() + 1);
        new_mem_tbls.push(new_tbl);
        new_mem_tbls.extend_from_slice(data.mem_tbls.as_slice());
        let new_data = ShardData::new(
            data.range.clone(),
            new_mem_tbls,
            data.l0_tbls.clone(),
            data.blob_tbl_map.clone(),
            data.cfs.clone(),
            data.unloaded_tbls.clone(),
            data.lock_txn_files.clone(),
            data.limiter.clone(),
            data.update_counter + 1,
        );
        new_data.refresh_for_limiter(&shard.tag());
        shard.set_data(new_data);
        info!(
            "shard {} switch mem-table version {}, size {}, force {}",
            shard.tag(),
            version,
            mem_table.size(),
            force,
        );
        let props = shard.properties.to_pb(shard.id);
        mem_table.set_properties(props);
    }

    pub fn write(&self, wb: &mut WriteBatch) -> u64 {
        let shard = self.get_shard(wb.shard_id).unwrap_or_else(|| {
            let tag = ShardTag::new(self.get_engine_id(), IdVer::new(wb.shard_id, 0));
            panic!("{} unable to get shard", tag);
        });
        let version = shard.get_base_version() + wb.sequence;
        self.update_write_batch_version(wb, version);
        let mut data = shard.get_data();
        let snap = shard.new_snap_access();
        let mut mem_tbl = data.get_writable_mem_table();
        for cf in 0..NUM_CFS {
            mem_tbl
                .get_cf(cf)
                .put_batch(wb.get_cf_mut(cf), Some(&snap), cf);
        }
        let mut need_refresh_shard_states = false;

        // Property may be duplicated when `Shard.properties` is restored from
        // `ShardMeta.properties`, in scene of recover shard or restore snapshot. But
        // we must still apply the side effect of setting property (i.e. switch
        // mem-table), which has not been persisted. Otherwise peers of a region would
        // be inconsistent.
        let is_property_duplicated = wb.sequence <= shard.get_meta_sequence();

        for (k, v) in std::mem::take(&mut wb.properties) {
            match k.as_str() {
                DEL_PREFIXES_KEY => {
                    // Use property value, other than merged del_prefixes, to make switch mem-table
                    // determined. As shard.get_del_prefixes() among peers may not be the same.
                    let prefix = v.chunk();
                    let mut del_prefixes =
                        DeletePrefixes::new_with_inner_key_off(shard.inner_key_off);
                    del_prefixes.merge_prefix_in_place(prefix);
                    if del_prefixes
                        .inner_delete_ranges()
                        .any(|(start, end)| mem_tbl.has_data_in_range(start, end))
                    {
                        info!(
                            "{} kvengine::write set_switch_mem_table for del_prefixes, prefix {:?}",
                            shard.tag(),
                            prefix
                        );
                        wb.set_switch_mem_table();
                    }
                    need_refresh_shard_states = true;

                    if !is_property_duplicated {
                        shard.merge_del_prefix(prefix);
                        shard
                            .properties
                            .set(k.as_str(), &shard.get_del_prefixes().marshal());
                    } else {
                        info!(
                            "{} kvengine::write: del_prefixes is duplicated, skip merge prefix {:?}, current del_prefixes {:?}",
                            shard.tag(),
                            prefix,
                            shard.get_del_prefixes()
                        );
                    }
                }
                TRUNCATE_TS_KEY => {
                    // TODO: handle duplicated property.
                    if shard.set_truncate_ts(v.chunk()) {
                        wb.set_switch_mem_table();
                        if mem_tbl.data_max_ts() > shard.get_truncate_ts().unwrap().inner() {
                            wb.set_switch_mem_table();
                        }
                        need_refresh_shard_states = true;
                        shard.properties.set(k.as_str(), v.chunk());
                    }
                }
                TRIM_OVER_BOUND => {
                    // TODO: handle duplicated property.
                    shard.set_trim_over_bound(v.chunk());
                    need_refresh_shard_states = true;
                    shard.properties.set(k.as_str(), v.chunk());
                }
                MANUAL_MAJOR_COMPACTION => {
                    shard.set_manual_major_compaction(v.chunk());
                    need_refresh_shard_states = true;
                    shard.properties.set(k.as_str(), v.chunk());
                }
                TXN_FILE_REF => {
                    let need_switch = self.write_txn_file_ref(&shard, v.chunk());
                    if need_switch {
                        wb.set_switch_mem_table();
                    }
                    need_refresh_shard_states = true;

                    data = shard.get_data();
                    mem_tbl = data.get_writable_mem_table();
                }
                _ => {
                    shard.properties.set(k.as_str(), v.chunk());
                }
            }
        }
        if need_refresh_shard_states {
            self.refresh_shard_states(&shard);
        }
        store_u64(&shard.write_sequence, wb.sequence);
        let size = mem_tbl.size();
        if wb.switch_mem_table || size > self.opts.max_mem_table_size {
            self.switch_mem_table(&shard, version, false);
            if let Err(err) = self.trigger_flush(&shard) {
                warn!("{} trigger_flush error: {:?}", shard.tag(), err);
            }
            0
        } else {
            size
        }
    }

    fn write_txn_file_ref(&self, shard: &Shard, v: &[u8]) -> bool {
        let mut txn_file_refs = TxnFileRefs::new();
        txn_file_refs.merge_from_bytes(v).unwrap();
        debug_assert_eq!(txn_file_refs.txn_file_refs.len(), 1);
        let txn_file_ref = txn_file_refs.take_txn_file_refs().pop().unwrap();
        let old_data = shard.get_data();
        let mut lock_txn_files = old_data.lock_txn_files.clone();

        let txn_file = self
            .txn_chunk_mgr
            .load_txn_file_from_ref(shard.id, shard.ver, &txn_file_ref, true)
            .unwrap();

        let (is_commit, is_rollback) = Self::merge_txn_file_ref(shard, txn_file_ref);

        // lock txn files will be merged into ShardData.
        Self::merge_lock_txn_files(&mut lock_txn_files, &txn_file, is_commit || is_rollback);

        // commit txn files will be added to the writable mem-table.
        let mut mem_tbls = old_data.mem_tbls.clone();
        if is_commit {
            mem_tbls[0] = old_data
                .get_writable_mem_table()
                .add_write_cf_txn_files(txn_file);
        }
        let data = ShardData::new(
            old_data.range.clone(),
            mem_tbls,
            old_data.l0_tbls.clone(),
            old_data.blob_tbl_map.clone(),
            old_data.cfs.clone(),
            old_data.unloaded_tbls.clone(),
            lock_txn_files,
            old_data.limiter.clone(),
            old_data.update_counter + 1,
        );
        shard.set_data(data);
        is_commit
    }

    fn merge_txn_file_ref(
        shard: &Shard,
        wb_ref: TxnFileRef,
    ) -> (bool /* is_commit */, bool /* is_rollback */) {
        let tag = shard.tag();
        let mut prop =
            TxnFileRefPropertyHelper::from_property(shard.get_property(TXN_FILE_REF)).unwrap();
        let (is_commit, is_rollback) = prop.merge_txn_file_ref(&tag, &wb_ref);
        shard.properties.set(TXN_FILE_REF, &prop.marshall());
        info!("{} merge txn file ref", tag; "wb_ref" => ?wb_ref, "prop" => ?prop);
        (is_commit, is_rollback)
    }

    fn merge_lock_txn_files(
        shard_lock_txn_files: &mut Vec<TxnFile>,
        txn_file: &TxnFile,
        is_commit_or_rollback: bool,
    ) {
        if let Some(idx) = shard_lock_txn_files
            .iter()
            .position(|x| x.start_ts() == txn_file.start_ts())
        {
            if is_commit_or_rollback {
                shard_lock_txn_files.remove(idx);
            } else {
                shard_lock_txn_files[idx] = txn_file.clone()
            }
        } else {
            debug_assert!(!is_commit_or_rollback);
            shard_lock_txn_files.push(txn_file.clone());
        }
    }

    fn update_write_batch_version(&self, wb: &mut WriteBatch, version: u64) {
        for cf in 0..NUM_CFS {
            if !CF_MANAGED[cf] {
                wb.get_cf_mut(cf).iterate(|e, _| {
                    e.version = version;
                });
            };
        }
        if let Some(txn_file_refs_bin) = wb.get_property(TXN_FILE_REF) {
            let mut txn_file_refs = TxnFileRefs::new();
            txn_file_refs.merge_from_bytes(txn_file_refs_bin).unwrap();
            for txn_file_ref in txn_file_refs.mut_txn_file_refs().iter_mut() {
                txn_file_ref.version = version;
            }
            wb.set_property(TXN_FILE_REF, &txn_file_refs.write_to_bytes().unwrap());
        }
    }

    pub fn flush_shard_for_restore(&self, shard: &Shard) -> Result<()> {
        let ver = shard.get_base_version()
            + cmp::max(shard.get_write_sequence(), shard.get_meta_sequence())
            + 1;
        debug!(
            "{} flush_shard_for_restore, ver: {}, base_ver: {}, write_seq: {}, meta_seq: {}",
            shard.tag(),
            ver,
            shard.get_base_version(),
            shard.get_write_sequence(),
            shard.get_meta_sequence(),
        );

        if !shard.get_initial_flushed() {
            self.load_unloaded_tables(shard.id, shard.ver, false)?;
        }
        self.switch_mem_table(shard, ver, false);
        self.set_shard_active(shard.id, true);
        self.trigger_flush(shard)
    }
}
