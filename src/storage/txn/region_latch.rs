// Copyright 2023 TiKV Project Authors. Licensed under Apache-2.0.

use std::sync::Arc;

use crossbeam::utils::CachePadded;
use dashmap::DashMap;
use kvengine::table::TxnFile;
use parking_lot::{Mutex, MutexGuard};

use crate::storage::txn::{latch::Latch, Lock};

#[derive(Default)]
pub struct GlobalLatches {
    slots: Vec<CachePadded<Mutex<Latch>>>,
    size: usize,
    regions: DashMap<u64, Arc<Mutex<RegionTxnLatch>>>,
}

impl GlobalLatches {
    pub fn new(size: usize) -> Self {
        let size = usize::next_power_of_two(size);
        let mut slots = Vec::with_capacity(size);
        (0..size).for_each(|_| slots.push(Mutex::new(Latch::new()).into()));
        Self {
            slots,
            size,
            regions: DashMap::new(),
        }
    }

    pub fn acquire(&self, lock: &mut Lock, who: u64) -> bool {
        if !lock.count_added {
            let region_latch = self
                .regions
                .entry(lock.region_id)
                .or_insert(Arc::new(Mutex::new(RegionTxnLatch::default())))
                .clone();
            let mut guard = region_latch.lock();
            if lock.txn_file.is_some() {
                return guard.acquire_txn_lock(lock, who);
            }
            if guard.conflict_with_txn_latch(lock, who) {
                return false;
            }
            if let Some(txn_latch) = &guard.txn_file_latch {
                lock.checked_txn_cid = txn_latch.cid;
                guard.checked_txn_cmd_count += 1;
            } else {
                guard.before_txn_cmd_count += 1;
            }
            lock.count_added = true;
        }
        let mut acquired_count: usize = 0;
        for &key_hash in &lock.required_hashes[lock.owned_count..] {
            let mut latch = self.lock_latch(lock.keyspace_id, key_hash);
            match latch.get_first_req_by_hash(lock.keyspace_id, key_hash) {
                Some(cid) => {
                    if cid == who {
                        acquired_count += 1;
                    } else {
                        latch.wait_for_wake(lock.keyspace_id, key_hash, who);
                        break;
                    }
                }
                None => {
                    latch.wait_for_wake(lock.keyspace_id, key_hash, who);
                    acquired_count += 1;
                }
            }
        }
        lock.owned_count += acquired_count;
        lock.acquired()
    }

    pub fn release(
        &self,
        lock: &Lock,
        who: u64,
        _keep_latches_for_next_cmd: Option<(u64, &Lock)>,
    ) -> Vec<u64> {
        let region_latch = self
            .regions
            .entry(lock.region_id)
            .or_insert(Arc::new(Mutex::new(RegionTxnLatch::default())))
            .clone();
        let mut guard = region_latch.lock();
        if lock.txn_file.is_some() {
            return guard.release_txn_lock(lock, who);
        }
        let mut wakeup_list = vec![];
        if let Some(txn_file_latch) = &guard.txn_file_latch {
            let txn_latch_cid = txn_file_latch.cid;
            if txn_latch_cid == lock.checked_txn_cid {
                guard.checked_txn_cmd_count -= 1;
            } else {
                guard.before_txn_cmd_count -= 1;
                if guard.before_txn_cmd_count == 0 {
                    wakeup_list.push(txn_latch_cid);
                }
            }
        } else {
            guard.before_txn_cmd_count -= 1;
        }
        for &key_hash in &lock.required_hashes[..lock.owned_count] {
            let mut latch = self.lock_latch(lock.keyspace_id, key_hash);
            let (ks, v, front) = latch.pop_front(lock.keyspace_id, key_hash).unwrap();
            assert_eq!(front, who);
            assert_eq!(v, key_hash);
            if let Some(wakeup) = latch.get_first_req_by_hash(ks, key_hash) {
                wakeup_list.push(wakeup);
            }
        }
        wakeup_list
    }

    #[inline]
    fn lock_latch(&self, keyspace_id: u32, hash: u64) -> MutexGuard<'_, Latch> {
        self.slots[(keyspace_id as usize ^ hash as usize) & (self.size - 1)].lock()
    }
}

#[derive(Default, Debug)]
struct RegionTxnLatch {
    before_txn_cmd_count: usize,
    checked_txn_cmd_count: usize,
    txn_file_latch: Option<TxnFileLatch>,
}

#[derive(Debug)]
struct TxnFileLatch {
    start_ts: u64,
    txn_file: TxnFile,
    cid: u64,
    waiting_tasks: Vec<u64>,
}

impl RegionTxnLatch {
    fn acquire_txn_lock(&mut self, lock: &mut Lock, who: u64) -> bool {
        if let Some(txn_file_latch) = self.txn_file_latch.as_mut() {
            if txn_file_latch.cid == who && self.before_txn_cmd_count == 0 {
                return true;
            }
            txn_file_latch.waiting_tasks.push(who);
            return false;
        }
        let txn_file_latch = TxnFileLatch {
            start_ts: lock.start_ts,
            txn_file: lock.txn_file.clone().unwrap(),
            cid: who,
            waiting_tasks: vec![],
        };
        self.txn_file_latch = Some(txn_file_latch);
        self.before_txn_cmd_count == 0
    }

    fn conflict_with_txn_latch(&mut self, lock: &mut Lock, who: u64) -> bool {
        if let Some(txn_file_latch) = self.txn_file_latch.as_mut() {
            if txn_file_latch
                .txn_file
                .key_hash_exists(&lock.required_hashes)
            {
                // If the lock conflict with the txn latch, it waits for the txn file.
                txn_file_latch.waiting_tasks.push(who);
                return true;
            }
        }
        false
    }

    fn release_txn_lock(&mut self, lock: &Lock, who: u64) -> Vec<u64> {
        self.before_txn_cmd_count = self.checked_txn_cmd_count;
        self.checked_txn_cmd_count = 0;
        let txn_file_latch = self.txn_file_latch.take().unwrap_or_else(|| {
            panic!(
                "release_txn_lock: txn file latch not exist, region_id: {}, lock: {:?}, who: {}, self: {:?}",
                lock.region_id, lock, who, self
            );
        });
        assert_eq!(lock.start_ts, txn_file_latch.start_ts);
        assert_eq!(txn_file_latch.cid, who);
        txn_file_latch.waiting_tasks
    }
}

#[cfg(test)]
mod tests {
    use api_version::ApiV2;
    use kvengine::{
        table::{
            file::InMemFile, sstable::BlockCache, InnerKey, TxnChunk, TxnChunkBuilder, TxnCtx,
            TxnFileId, OP_PUT,
        },
        util::test_util::KeyBuilder,
        UserMeta, GLOBAL_SHARD_END_KEY,
    };
    use rstest::rstest;
    use txn_types::Key;

    use super::*;

    const KEYSPACE_ID: u32 = 42;

    #[test]
    fn test_region_latch_normal() {
        let global_latches = GlobalLatches::new(1024);

        let key_1_str = "x001abc";
        let keyspace_1 = ApiV2::get_u32_keyspace_id(ApiV2::get_keyspace_id(key_1_str.as_bytes()));
        let mut lock_1 = Lock::new(keyspace_1, &[Key::from_raw(key_1_str.as_bytes())]);
        assert!(global_latches.acquire(&mut lock_1, 1));

        let mut lock_1_conflict = Lock::new(keyspace_1, &[Key::from_raw(key_1_str.as_bytes())]);
        assert!(!global_latches.acquire(&mut lock_1_conflict, 2));

        let wakes = global_latches.release(&lock_1, 1, None);
        assert_eq!(wakes.len(), 1);
        assert_eq!(wakes[0], 2);
        assert!(global_latches.acquire(&mut lock_1_conflict, 2));

        let key_2_str = "x002abc";
        let keyspace_2 = ApiV2::get_u32_keyspace_id(ApiV2::get_keyspace_id(key_2_str.as_bytes()));
        let mut lock_2 = Lock::new(keyspace_2, &[Key::from_raw(key_2_str.as_bytes())]);

        // The key hash is calculated with keyspace prefix trimmed, so the hashes should
        // be equal.
        assert_eq!(
            lock_1_conflict.required_hashes.as_slice(),
            lock_2.required_hashes.as_slice()
        );
        // but they don't block each other.
        assert!(global_latches.acquire(&mut lock_2, 3));
    }

    #[rstest]
    #[case::enable_key_off(true)]
    #[case::disable_key_off(false)]
    fn test_region_latch_txn_file(#[case] enable_inner_key_off: bool) {
        let kb = KeyBuilder::new(KEYSPACE_ID, enable_inner_key_off, "t_");
        let global_latches = GlobalLatches::new(1024);

        let mut normal_lock = make_normal_lock(&kb.i_to_key(20), 999);
        let normal_cid = 1;
        assert!(global_latches.acquire(&mut normal_lock, normal_cid));

        let lower_bound = InnerKey::from_inner_buf(b"");
        let upper_bound = InnerKey::from_inner_buf(GLOBAL_SHARD_END_KEY);

        let txn_file = make_txn_file(
            100,
            200,
            1000,
            lower_bound,
            upper_bound,
            enable_inner_key_off,
            &kb,
        );
        let mut txn_lock = make_txn_lock(txn_file);
        let txn_lock_cid = 2;
        // The lock before the txn file blocks txn file lock.
        assert!(!global_latches.acquire(&mut txn_lock, txn_lock_cid));

        let wakes = global_latches.release(&normal_lock, normal_cid, None);
        assert_eq!(wakes, vec![2]);
        assert!(global_latches.acquire(&mut txn_lock, txn_lock_cid));

        // non conflicting lock will not block by txn lock.
        let mut normal_lock_no_conflict = make_normal_lock(&kb.i_to_key(50), 1001);
        let normal_no_conflict_cid = 3;
        assert!(global_latches.acquire(&mut normal_lock_no_conflict, normal_no_conflict_cid));

        // conflicting lock will be blocked by txn lock.
        let mut normal_lock_conflict = make_normal_lock(&kb.i_to_key(150), 1003);
        let normal_conflict_cid = 4;
        assert!(!global_latches.acquire(&mut normal_lock_conflict, normal_conflict_cid));

        // another txn file will be blocked by txn lock.
        let txn_file_2 = make_txn_file(
            300,
            400,
            1010,
            lower_bound,
            upper_bound,
            enable_inner_key_off,
            &kb,
        );
        let mut txn_lock_2 = make_txn_lock(txn_file_2);
        let txn_lock_2_cid = 5;
        assert!(!global_latches.acquire(&mut txn_lock_2, txn_lock_2_cid));

        let wakes = global_latches.release(&txn_lock, txn_lock_cid, None);
        assert_eq!(wakes, vec![normal_conflict_cid, txn_lock_2_cid]);

        assert!(global_latches.acquire(&mut normal_lock_conflict, normal_conflict_cid));

        // txn_lock_2 can not acquire because normal_lock_no_conflict is acquired before
        // txn_lock_2 and not released.
        assert!(!global_latches.acquire(&mut txn_lock_2, txn_lock_2_cid));

        let wakes = global_latches.release(&normal_lock_conflict, normal_conflict_cid, None);
        assert!(wakes.is_empty());
        assert!(!global_latches.acquire(&mut txn_lock_2, txn_lock_2_cid));

        let wakes = global_latches.release(&normal_lock_no_conflict, normal_no_conflict_cid, None);
        assert_eq!(wakes, vec![txn_lock_2_cid]);
        assert!(global_latches.acquire(&mut txn_lock_2, txn_lock_2_cid));
    }

    fn make_txn_file(
        start: usize,
        end: usize,
        start_ts: u64,
        lower_bound: InnerKey<'_>,
        upper_bound: InnerKey<'_>,
        enable_inner_key_off: bool,
        kb: &KeyBuilder,
    ) -> TxnFile {
        let txn_file_id = TxnFileId::new(1, 1, start_ts);
        let txn_chunk = make_txn_chunk(start, end, start_ts, enable_inner_key_off, kb);
        let user_meta = UserMeta::new(start_ts, start_ts + 1).to_array().to_vec();
        let txn_ctx = TxnCtx::new(
            user_meta.into(),
            vec![].into(),
            start_ts + 1,
            lower_bound,
            upper_bound,
        );
        TxnFile::new(txn_file_id, vec![txn_chunk], txn_ctx).unwrap()
    }

    fn make_txn_chunk(
        start: usize,
        end: usize,
        chunk_id: u64,
        enable_inner_key_off: bool,
        kb: &KeyBuilder,
    ) -> TxnChunk {
        let mut txn_chunk_builder =
            TxnChunkBuilder::new(chunk_id, 10, None, KEYSPACE_ID, enable_inner_key_off);
        for i in start..end {
            let key = kb.i_to_key(i);
            txn_chunk_builder.add_entry(InnerKey::from_outer_key(&key), OP_PUT, &key);
        }
        let mut buf = vec![];
        txn_chunk_builder.finish(&mut buf);
        let in_mem_file = Arc::new(InMemFile::new(chunk_id, buf.into()));
        TxnChunk::new(in_mem_file, BlockCache::None, None).unwrap()
    }

    fn make_normal_lock(raw_key: &[u8], start_ts: u64) -> Lock {
        let key = Key::from_raw(raw_key);
        let mut lock = Lock::new(KEYSPACE_ID, &[key]);
        lock.set_region_id_start_ts(1, start_ts);
        lock
    }

    fn make_txn_lock(txn_file: TxnFile) -> Lock {
        let mut lock = Lock::new(KEYSPACE_ID, &[]);
        lock.set_region_id_start_ts(1, txn_file.start_ts());
        lock.txn_file = Some(txn_file);
        lock
    }
}
