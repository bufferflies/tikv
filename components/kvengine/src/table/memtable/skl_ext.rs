// Copyright 2023 TiKV Project Authors. Licensed under Apache-2.0.

use std::ops::Deref;

use crate::{
    table::{
        encode_val_to_outer_val_owner,
        memtable::{Hint, SkipList, WriteBatch},
        new_merge_iterator,
        txn_file::{SkipOpTxnFileIterator, TxnFile, TxnFileIterator, OP_CHECK_NOT_EXIST, OP_LOCK},
        InnerKey, Value,
    },
    Iterator, SnapAccess,
};

/// SkipListExt integrates txn files with SkipList, committed TxnFiles are added
/// to the WriteCF, and then switched, so txn files data in a mem-table always
/// have higher commit version. The Lock CF TxnFiles are handled separately in
/// SnapData.
#[derive(Clone)]
pub struct SkipListExt {
    skl: SkipList,
    txn_file: Option<TxnFile>,
}

impl SkipListExt {
    pub fn new(skl: SkipList) -> Self {
        Self {
            skl,
            txn_file: None,
        }
    }

    pub fn add_txn_file(&self, txn_file: TxnFile) -> Self {
        info!(
            "add txn_file id: {:?}, smallest: {:?}, biggest: {:?}",
            txn_file.id(),
            txn_file.smallest(),
            txn_file.biggest()
        );
        Self {
            skl: self.skl.clone(),
            txn_file: Some(txn_file),
        }
    }

    pub fn size(&self) -> usize {
        self.skl.size() as usize + self.txn_file.as_ref().map_or(0, |x| x.size())
    }

    pub fn is_empty(&self) -> bool {
        self.size() == 0
    }

    pub fn put_batch(&self, batch: &mut WriteBatch, snap: Option<&SnapAccess>, cf: usize) {
        self.skl.put_batch(batch, snap, cf);
    }

    pub fn new_iterator(&self, reverse: bool) -> Box<dyn Iterator> {
        let skl_iter = Box::new(self.skl.new_iterator(reverse));
        if self.txn_file.is_none() {
            return skl_iter;
        }
        let txn_file_iter = TxnFileIterator::new(self.txn_file.clone().unwrap(), reverse);
        let skip_op_iter = Box::new(SkipOpTxnFileIterator::new(txn_file_iter, true, true));
        new_merge_iterator(vec![skip_op_iter, skl_iter], reverse)
    }

    fn try_get_from_txn_file(
        &self,
        key: InnerKey<'_>,
        version: u64,
        outer_owner: &mut Vec<u8>,
    ) -> Option<Value> {
        if let Some(txn_file) = self.txn_file.as_ref() {
            if txn_file.version() > version {
                return None;
            }
            let (op, val) = txn_file.get_value(key, outer_owner);
            if val.is_valid() && op != OP_LOCK && op != OP_CHECK_NOT_EXIST {
                return Some(val);
            }
        }
        None
    }

    pub fn get_with_hint(
        &self,
        key: InnerKey<'_>,
        version: u64,
        h: &mut Hint,
        outer_owner: &mut Vec<u8>,
    ) -> Value {
        if let Some(v) = self.try_get_from_txn_file(key, version, outer_owner) {
            return v;
        }
        let v = self.skl.get_with_hint(key.deref(), version, h);
        if v.is_valid() {
            return encode_val_to_outer_val_owner(v, outer_owner);
        }
        Value::new()
    }

    pub fn get(&self, key: InnerKey<'_>, version: u64, outer_owner: &mut Vec<u8>) -> Value {
        if let Some(v) = self.try_get_from_txn_file(key, version, outer_owner) {
            return v;
        }
        let v = self.skl.get(key.deref(), version);
        if v.is_valid() {
            return encode_val_to_outer_val_owner(v, outer_owner);
        }
        Value::new()
    }

    pub fn get_newer(&self, key: InnerKey<'_>, version: u64, outer_owner: &mut Vec<u8>) -> Value {
        if let Some(txn_file) = self.txn_file.as_ref() {
            let (op, val) = txn_file.get_value(key, outer_owner);
            if val.is_valid() && val.version >= version && op != OP_LOCK && op != OP_CHECK_NOT_EXIST
            {
                return val;
            }
        }
        let v = self.skl.get_newer(key.deref(), version);
        if v.is_valid() {
            return encode_val_to_outer_val_owner(v, outer_owner);
        }
        Value::new()
    }

    pub fn data_max_ts(&self) -> u64 {
        let mut max_ts = self.skl.data_max_ts();
        if let Some(txn_file) = self.txn_file.as_ref() {
            max_ts = std::cmp::max(max_ts, txn_file.version())
        }
        max_ts
    }

    pub fn get_txn_file(&self) -> Option<TxnFile> {
        self.txn_file.clone()
    }
}

#[cfg(test)]
mod tests {
    use std::{ops::Deref, sync::Arc};

    use bytes::Bytes;

    use crate::{
        table::{
            memtable::{skl_ext::SkipListExt, SkipList, WriteBatch},
            sstable::InMemFile,
            txn_file::{TxnChunk, TxnChunkBuilder, TxnCtx, TxnFile, TxnFileId, OP_PUT},
            InnerKey,
        },
        UserMeta, WRITE_CF,
    };

    fn new_key(i: i32) -> String {
        format!("key{:05}", i)
    }

    fn new_val(i: i32) -> String {
        format!("val{:05}", i)
    }

    fn write_skl_write_cf(skl: &SkipList, keys: Vec<i32>, start_ts: u64, commit_ts: u64) {
        let mut wb = WriteBatch::new();
        for i in keys {
            let key = new_key(i);
            let val = new_val(i);
            wb.put(
                InnerKey::from_inner_buf(key.as_bytes()),
                0,
                &UserMeta::new(start_ts, commit_ts).to_array(),
                commit_ts,
                val.as_bytes(),
            );
        }
        skl.put_batch(&mut wb, None, WRITE_CF);
    }

    fn build_txn_file_chunk_data(keys: Vec<i32>) -> Bytes {
        let mut batch_builder = TxnChunkBuilder::new(10);
        for i in keys {
            let key = new_key(i);
            let val = new_val(i);
            batch_builder.add_entry(key.as_bytes(), OP_PUT, val.as_bytes());
        }
        let mut data_buf = vec![];
        batch_builder.finish(&mut data_buf);
        data_buf.into()
    }

    #[test]
    fn test_skl_ext_write_cf() {
        let skl = SkipList::new(None);
        write_skl_write_cf(&skl, vec![5, 10, 15], 100, 101);
        // write CF skip list data will always written before txn file data, because
        // mem-table will instantly switch after apply TxnFile commit.
        let txn_file_chunk_data = build_txn_file_chunk_data(vec![10, 12, 18]);
        let txn_file_chunk_file = Arc::new(InMemFile::new(1, txn_file_chunk_data));
        let txn_file_chunk = TxnChunk::new(txn_file_chunk_file, None).unwrap();
        let user_meta = UserMeta::new(102, 103).to_array().to_vec();
        let txn_ctx = TxnCtx::new(user_meta.into(), Bytes::new(), 103);
        let txn_file_id = TxnFileId::new(1, 1, 102);
        let txn_file = TxnFile::new(txn_file_id, vec![txn_file_chunk], txn_ctx).unwrap();
        let skl_ext = SkipListExt::new(skl).add_txn_file(txn_file);

        // test find key in skl.
        let key = new_key(5);
        let mut outer_key_owner = vec![];
        let val = skl_ext.get(
            InnerKey::from_inner_buf(key.as_bytes()),
            u64::MAX,
            &mut outer_key_owner,
        );
        assert_eq!(val.get_value(), new_val(5).as_bytes());
        assert_eq!(val.user_meta(), &UserMeta::new(100, 101).to_array());
        assert_eq!(val.version, 101);

        // test find key in txn file.
        let key = new_key(10);
        let mut outer_key_owner = vec![];
        let val = skl_ext.get(
            InnerKey::from_inner_buf(key.as_bytes()),
            u64::MAX,
            &mut outer_key_owner,
        );
        assert_eq!(val.get_value(), new_val(10).as_bytes());
        assert_eq!(val.user_meta(), &UserMeta::new(102, 103).to_array());
        assert_eq!(val.version, 103);

        // test get newer.
        let key = new_key(10);
        let inner_key = InnerKey::from_inner_buf(key.as_bytes());
        let mut outer_key_owner = vec![];
        let mut val = skl_ext.get_newer(inner_key, 102, &mut outer_key_owner);
        assert!(val.is_valid());
        assert_eq!(val.version, 103);
        val = skl_ext.get_newer(inner_key, 103, &mut outer_key_owner);
        assert!(val.is_valid());
        assert_eq!(val.version, 103);
        val = skl_ext.get_newer(inner_key, 104, &mut outer_key_owner);
        assert!(!val.is_valid());

        // test iterator.
        let mut iter = skl_ext.new_iterator(false);
        let mut count = 0;
        iter.rewind();
        let mut prev_key = vec![];
        let mut prev_version = 0;
        while iter.valid() {
            count += 1;
            let key = iter.key();
            if prev_key.as_slice() == key.deref() {
                assert!(prev_version > iter.value().version)
            } else {
                assert!(prev_key.as_slice() < key.deref());
            }
            prev_key = key.to_vec();
            prev_version = iter.value().version;
            iter.next_all_version();
        }
        assert_eq!(count, 6);
    }
}
