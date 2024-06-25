// Copyright 2023 TiKV Project Authors. Licensed under Apache-2.0.

use bytes::{Buf, BufMut, Bytes, BytesMut};
use collections::{HashMap, HashMapEntry};
use kvproto::kvrpcpb;
use log_wrappers::Value;
use protobuf::Message;
use tikv_util::{
    box_err,
    codec::number::{U16_SIZE, U32_SIZE, U64_SIZE},
};

use crate::{DeletePrefixes, ShardMeta, ShardTag, UserMeta, DEL_PREFIXES_KEY};

/// A helper function to evenly distribute `total` into `count` parts.
/// Note: when `total` <= `count`, return `[1; total]`.
pub fn evenly_distribute(total: usize, count: usize) -> Vec<usize> {
    debug_assert!(count > 0);
    if total <= count {
        return vec![1; total];
    }
    let quotient = total / count;
    let remainder = total % count;
    let mut ret = vec![quotient; count];
    for i in 0..remainder {
        ret[i] += 1;
    }
    ret
}

/// Helper for merging or splitting properties.
///
/// Currently only delete prefixes are handled.
///
/// Splitting properties is necessary for the following scenarios:
/// 1. `preprocess_pending_splits` (`build_split_pb`).
///
/// Merging properties is necessary for the following scenarios:
/// 1. `ShardMeta.commit_merge` (merge `ShardMeta` on region merge).
/// 2. `kvengine::Engine::commit_merge` (merge `Shard` on region merge).
/// 3. `restore_keyspace::BackupCluster::gather_sstables` (merge `ShardMeta` on
/// merging backup regions).
pub struct PropertiesHelper {
    del_prefixes: DeletePrefixes,

    /// Prefix of shard range, see `ShardRange::prefix()`.
    range_prefix: Vec<u8>,
    /// Whether to rewrite range prefix of `del_prefixes`.
    /// When enabled, range prefix of `del_prefixes` to be merged will be
    /// rewritten to `range_prefix`.
    rewrite_range_prefix: bool,
}

impl PropertiesHelper {
    pub fn new_from_shard_meta(meta: &ShardMeta) -> Self {
        Self::new(
            meta.get_property(DEL_PREFIXES_KEY),
            meta.range.inner_key_off,
            meta.range.prefix().to_owned(),
        )
    }

    fn new(del_prefixes_bytes: Option<Bytes>, inner_key_off: usize, range_prefix: Vec<u8>) -> Self {
        let del_prefixes = if let Some(bs) = del_prefixes_bytes {
            DeletePrefixes::unmarshal(bs.chunk(), inner_key_off)
        } else {
            DeletePrefixes::new_with_inner_key_off(inner_key_off)
        };
        Self {
            del_prefixes,
            range_prefix,
            rewrite_range_prefix: false,
        }
    }

    pub fn set_rewrite_range_prefix(&mut self, rewrite_range_prefix: bool) {
        self.rewrite_range_prefix = rewrite_range_prefix;
    }

    pub fn merge_shard_meta(&mut self, other: &ShardMeta) {
        self.merge(
            other
                .get_property(DEL_PREFIXES_KEY)
                .map(|prop| prop.to_vec()),
            other.range.inner_key_off,
        );
    }

    // TODO: pub fn merge_shard(..)

    fn merge(&mut self, del_prefixes_bytes: Option<Vec<u8>>, inner_key_off: usize) {
        // `self.del_prefixes.inner_key_off` & `inner_key_off` are not necessarily
        // equal.
        if let Some(bs) = del_prefixes_bytes {
            let mut del_prefixes = DeletePrefixes::unmarshal(&bs, inner_key_off);
            if self.rewrite_range_prefix
                && inner_key_off > 0
                && inner_key_off == self.range_prefix.len()
            {
                del_prefixes.rewrite_range_prefix(&self.range_prefix);
            }
            self.del_prefixes.merge(&del_prefixes);
        }
    }

    pub fn build_to_shard_meta(&self, meta: &mut ShardMeta) {
        if !self.del_prefixes.prefixes.is_empty() {
            info!(
                "{} PropertiesMerger.build_to_shard_meta: {:?}",
                meta.tag(),
                self.del_prefixes
            );
            meta.set_property(DEL_PREFIXES_KEY, &self.del_prefixes.marshal());
        }
    }

    fn split(&self, start_key: &[u8], end_key: &[u8]) -> DeletePrefixes {
        self.del_prefixes
            .build_split(start_key, end_key, self.del_prefixes.inner_key_off)
    }

    pub fn split_to_properties(
        &self,
        start_key: &[u8],
        end_key: &[u8],
        props: &mut kvenginepb::Properties,
    ) {
        let split_del_prefixes = self.split(start_key, end_key);
        if !split_del_prefixes.is_empty() {
            props.mut_keys().push(DEL_PREFIXES_KEY.to_owned());
            props.mut_values().push(split_del_prefixes.marshal());
        }
    }
}

#[derive(Debug)]
pub struct TxnFileRefPropertyHelper {
    txn_file_refs: kvenginepb::TxnFileRefs,
}

impl TxnFileRefPropertyHelper {
    pub fn from_property(data: Option<Bytes>) -> crate::Result<Self> {
        let mut txn_file_refs = kvenginepb::TxnFileRefs::new();
        if let Some(data) = data {
            txn_file_refs
                .merge_from_bytes(&data)
                .map_err(|err| -> crate::Error {
                    box_err!(
                        "failed to unmarshal TxnFileRefs: {:?}, data: {}",
                        err,
                        &Value::value(&data)
                    )
                })?;
        }
        Ok(Self { txn_file_refs })
    }

    pub fn marshall(&self) -> Vec<u8> {
        self.txn_file_refs.write_to_bytes().unwrap()
    }

    pub fn merge_txn_file_ref(
        &mut self,
        tag: &ShardTag,
        wb_ref: &kvenginepb::TxnFileRef,
    ) -> (bool /* is_commit */, bool /* is_rollback */) {
        let (is_commit, is_rollback) = if wb_ref.get_user_meta().is_empty() {
            (false, false)
        } else {
            let user_meta = UserMeta::from_slice(&wb_ref.user_meta);
            let is_rollback = user_meta.is_rollback();
            (!is_rollback, is_rollback)
        };

        let mut refs = self.txn_file_refs.take_txn_file_refs().into_vec();
        if let Some(idx) = refs.iter().position(|x| x.start_ts == wb_ref.start_ts) {
            if is_rollback {
                refs.remove(idx);
            } else {
                refs[idx] = wb_ref.clone();
            }
        } else {
            debug_assert!(
                !is_commit && !is_rollback,
                "{} merge txn file ref: lock not found: wb_ref: {:?}, current: {:?}",
                tag,
                wb_ref,
                self.txn_file_refs
            );
            refs.push(wb_ref.clone());
        }
        self.txn_file_refs.set_txn_file_refs(refs.into());

        (is_commit, is_rollback)
    }

    pub fn clear_finished(&mut self, version: u64) {
        let mut refs = self.txn_file_refs.take_txn_file_refs().into_vec();
        refs.retain(|r| r.get_version() > version || !r.get_lock_val_prefix().is_empty());
        self.txn_file_refs.set_txn_file_refs(refs.into());
    }
}

#[derive(Debug, Default, Clone)]
pub struct TxnFileLocks {
    // Shard meta sequence (log index) on changed. Used for de-duplication.
    seq: u64,
    inner: HashMap<u64 /* txn start_ts */, Bytes /* lock_val_prefix */>,
}

const TXN_FILE_LOCKS_FORMAT_VER: u16 = 1;

impl TxnFileLocks {
    pub fn marshall(&self) -> Bytes {
        let size: usize = self
            .inner
            .values()
            .map(|lock| U64_SIZE /* start_ts */ + U32_SIZE /* lock_len */ + lock.len())
            .sum();
        let capacity = U16_SIZE /* ver */ + U64_SIZE /* seq */ + size;

        let mut buf = BytesMut::with_capacity(capacity);
        buf.put_u16(TXN_FILE_LOCKS_FORMAT_VER);
        buf.put_u64(self.seq);
        for (&ts, lock) in &self.inner {
            buf.put_u64(ts);
            buf.put_u32(lock.len() as u32);
            buf.put(lock.as_ref());
        }
        buf.freeze()
    }

    pub fn unmarshall(mut buf: Bytes) -> Self {
        debug_assert!(buf.len() >= U16_SIZE + U64_SIZE, "buf: {:?}", buf);
        let ver = buf.get_u16();
        let seq = buf.get_u64();
        debug_assert_eq!(ver, TXN_FILE_LOCKS_FORMAT_VER, "buf: {:?}", buf);

        let mut locks = HashMap::default();
        while buf.has_remaining() {
            debug_assert!(buf.remaining() >= U64_SIZE + U32_SIZE, "buf: {:?}", buf);
            let start_ts = buf.get_u64();
            let lock_len = buf.get_u32() as usize;
            debug_assert!(buf.remaining() >= lock_len, "buf: {:?}", buf);
            let lock = buf.split_to(lock_len);
            locks.insert(start_ts, lock);
        }
        Self { seq, inner: locks }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    #[inline]
    pub fn seq(&self) -> u64 {
        self.seq
    }

    // Return whether modified, i.e., `start_ts` is newly inserted.
    // Duplication should be checked before insert.
    #[inline]
    pub fn insert(&mut self, seq: u64, start_ts: u64, lock_val_prefix: &[u8]) -> bool /* modified */
    {
        debug_assert!(self.seq < seq);
        self.seq = seq;

        match self.inner.entry(start_ts) {
            HashMapEntry::Vacant(e) => {
                e.insert(Bytes::copy_from_slice(lock_val_prefix));
                true
            }
            HashMapEntry::Occupied(_) => {
                // Value of `lock_val_prefix` may be different (for `min_commit_ts`), but can be
                // ignored, as it does not affect resolving the lock.
                false
            }
        }
    }

    // Return whether modified, i.e., an existed `start_ts` is removed.
    // Duplication should be checked before remove.
    #[inline]
    pub fn remove(&mut self, seq: u64, start_ts: u64) -> bool /* modified */ {
        debug_assert!(self.seq < seq);
        self.seq = seq;
        self.inner.remove(&start_ts).is_some()
    }

    // Note:
    // * `raw_key` is used to locate the region, it's not necessary to be in the txn
    //   file. Resolving txn file locks do not depend on the key.
    // * It's unreasonable that all locks from different transactions are of the
    //   same key. But currently TiDB/client-go does not complain about it.
    pub fn get_key_errors(&self, raw_key: &[u8]) -> crate::Result<Vec<kvrpcpb::KeyError>> {
        let mut key_errors = Vec::with_capacity(self.inner.len());
        for (start_ts, lock_val_prefix) in self.inner.iter() {
            let lock = txn_types::Lock::parse(lock_val_prefix).map_err(|err| {
                crate::Error::Other(box_err!(
                    "failed to parse lock: start_ts {}, lock_val_prefix {:?}, err {:?}",
                    start_ts,
                    lock_val_prefix,
                    err
                ))
            })?;
            let lock_info = lock.into_lock_info(raw_key.to_vec());
            let mut key_err = kvrpcpb::KeyError::default();
            key_err.set_locked(lock_info);
            key_errors.push(key_err);
        }
        Ok(key_errors)
    }
}

#[cfg(test)]
mod tests {
    use api_version::api_v2::KEYSPACE_PREFIX_LEN;

    use super::*;

    #[test]
    fn test_evenly_distribute() {
        assert_eq!(evenly_distribute(0, 1), Vec::<usize>::new());
        assert_eq!(evenly_distribute(10, 1), vec![10]);
        assert_eq!(evenly_distribute(10, 2), vec![5, 5]);
        assert_eq!(evenly_distribute(10, 3), vec![4, 3, 3]);
        assert_eq!(evenly_distribute(10, 4), vec![3, 3, 2, 2]);
        assert_eq!(evenly_distribute(10, 5), vec![2, 2, 2, 2, 2]);
        assert_eq!(evenly_distribute(10, 6), vec![2, 2, 2, 2, 1, 1]);
        assert_eq!(evenly_distribute(10, 7), vec![2, 2, 2, 1, 1, 1, 1]);
        assert_eq!(evenly_distribute(10, 8), vec![2, 2, 1, 1, 1, 1, 1, 1]);
        assert_eq!(evenly_distribute(10, 9), vec![2, 1, 1, 1, 1, 1, 1, 1, 1]);
        assert_eq!(evenly_distribute(10, 10), vec![1; 10]);
        assert_eq!(evenly_distribute(10, 11), vec![1; 10]);
        assert_eq!(evenly_distribute(10, 100), vec![1; 10]);
    }

    #[test]
    fn test_properties_helper() {
        let dp0 = DeletePrefixes::new_with_inner_key_off(KEYSPACE_PREFIX_LEN)
            .merge_prefix(b"x000101")
            .merge_prefix(b"x000103");

        let dp1 = DeletePrefixes::new_with_inner_key_off(KEYSPACE_PREFIX_LEN)
            .merge_prefix(b"x0101033")
            .merge_prefix(b"x0101066");

        let mut helper = PropertiesHelper::new(
            Some(dp0.marshal().into()),
            KEYSPACE_PREFIX_LEN,
            b"x000".to_vec(),
        );
        helper.set_rewrite_range_prefix(true);
        helper.merge(Some(dp1.marshal()), KEYSPACE_PREFIX_LEN);
        assert_eq!(
            helper.del_prefixes.prefixes,
            vec![
                b"x000101".to_vec(),
                b"x000103".to_vec(),
                b"x0001066".to_vec(),
            ]
        );

        helper.set_rewrite_range_prefix(false);
        let dp2 = DeletePrefixes::new_with_inner_key_off(KEYSPACE_PREFIX_LEN)
            .merge_prefix(b"x020101")
            .merge_prefix(b"x0201077");
        helper.merge(Some(dp2.marshal()), KEYSPACE_PREFIX_LEN);
        assert_eq!(
            helper.del_prefixes.prefixes,
            vec![
                b"x000101".to_vec(),
                b"x000103".to_vec(),
                b"x0001066".to_vec(),
                b"x020101".to_vec(),
                b"x0201077".to_vec(),
            ]
        );
    }

    #[test]
    fn test_txn_file_locks() {
        let mut locks = TxnFileLocks::default();
        assert!(locks.insert(1, 1, "a".as_bytes()));
        assert!(locks.insert(2, 2, "b".as_bytes()));
        assert!(!locks.insert(3, 1, "c".as_bytes()));

        let data = locks.marshall();
        let locks1 = TxnFileLocks::unmarshall(data);
        assert_eq!(locks.inner, locks1.inner);

        assert!(locks.remove(4, 1));
        assert!(!locks.remove(5, 1));
        assert!(locks.remove(6, 2));
        assert!(!locks.remove(7, 2));
        assert!(locks.is_empty());
    }
}
