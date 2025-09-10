// Copyright 2022 TiKV Project Authors. Licensed under Apache-2.0.

use bytes::Bytes;
use kvengine::{UserMeta, EXTRA_CF, LOCK_CF, WRITE_CF};
use txn_types::{Key, Lock, OldValue, TimeStamp, Value, Write, WriteType};

use crate::storage::mvcc::{metrics::EXTRA_CF_SCAN_ITERATIONS, Result, TxnCommitRecord};

pub struct CloudReader {
    snapshot: kvengine::SnapAccess,
    fill_cache: bool,
    pub statistics: tikv_kv::Statistics,
    /// Cached EXTRA_CF iterator for reuse across multiple conflict checks
    cached_extra_cf_iter: Option<kvengine::read::Iterator>,
}

impl CloudReader {
    pub fn new(snapshot: kvengine::SnapAccess, fill_cache: bool) -> Self {
        Self {
            snapshot,
            fill_cache,
            statistics: tikv_kv::Statistics::default(),
            cached_extra_cf_iter: None,
        }
    }

    fn get_commit_by_item(
        user_meta: &UserMeta,
        value: &[u8],
        start_ts: TimeStamp,
    ) -> Option<TxnCommitRecord> {
        if user_meta.start_ts == start_ts.into_inner() {
            let (write_type, short_value) = if value.is_empty() {
                (WriteType::Delete, None)
            } else {
                (WriteType::Put, Some(value.to_vec()))
            };
            let write = Write::new(write_type, TimeStamp::new(user_meta.start_ts), short_value);
            return Some(TxnCommitRecord::SingleRecord {
                commit_ts: TimeStamp::new(user_meta.commit_ts),
                write,
            });
        }
        None
    }

    /// Note: This method is also used by resolving locks during restoring
    /// keyspace.
    #[maybe_async::both]
    pub async fn get_txn_commit_record(
        &mut self,
        key: &Key,
        start_ts: TimeStamp,
    ) -> Result<TxnCommitRecord> {
        let raw_key = key.to_raw()?;
        let item = self.snapshot.get(WRITE_CF, &raw_key, 0).await;
        if item.user_meta_len() > 0 {
            let user_meta = UserMeta::from_slice(item.user_meta());
            if let Some(record) = Self::get_commit_by_item(&user_meta, item.get_value(), start_ts) {
                return Ok(record);
            }
        }
        let mut data_iter = self
            .snapshot
            .new_iterator(WRITE_CF, false, true, None, self.fill_cache)
            .await;
        let mut next_key = Vec::with_capacity(raw_key.len() + 1);
        next_key.extend_from_slice(&raw_key);
        next_key.push(0);
        data_iter
            .set_range(Bytes::from(raw_key), Bytes::from(next_key))
            .await;
        while data_iter.valid() {
            debug_assert!(!kvengine::table::is_deleted(data_iter.meta()));
            // TODO: remove this check, iterator should not return deleted records.
            if kvengine::table::is_deleted(data_iter.meta()) {
                break;
            }

            let user_meta = UserMeta::from_slice(data_iter.user_meta());
            if user_meta.commit_ts < start_ts.into_inner() {
                // A transaction's commit_ts must be greater than start_ts, if current commit_ts
                // is already smaller than the start_ts, we don't need to look for older
                // version.
                break;
            }
            if let Some(record) = Self::get_commit_by_item(&user_meta, data_iter.val(), start_ts) {
                return Ok(record);
            }
            data_iter.next().await;
        }
        match self.get_extra(key, start_ts) {
            Some((commit_ts, write)) => Ok(TxnCommitRecord::SingleRecord { commit_ts, write }),
            None => Ok(TxnCommitRecord::None {
                overlapped_write: None,
            }),
        }
    }

    /// Find transaction status records using optimal strategy based on search
    /// criteria.
    /// Check for write conflicts in EXTRA_CF.
    /// Returns the first conflict found (either self-rollback or newer write).
    /// Always checks for self-rollback first (ExactMatch) before checking newer
    /// writes.
    ///
    /// # Parameters
    /// - `start_ts`: The start_ts of the transaction to be checked for
    ///   self-rollback
    /// - `max_allowed_write`: Exclusive threshold - any write with commit_ts >
    ///   this value is a conflict
    pub fn find_extra_cf_conflict_record(
        &mut self,
        key: &Key,
        start_ts: Option<TimeStamp>,
        max_allowed_write: Option<TimeStamp>,
    ) -> Result<Option<UserMeta>> {
        match (start_ts, max_allowed_write) {
            (Some(start_ts), None) => self.check_self_rollback(key, start_ts),
            _ => self.scan_extra_cf_for_write_conflict(key, start_ts, max_allowed_write),
        }
    }

    /// Check for self-rollback using point lookup optimization.
    ///
    /// Note: EXTRA_CF uses raw keys with encoded start_ts suffix (raw_key +
    /// reversed_start_ts), unlike WRITE_CF/LOCK_CF which use raw keys with
    /// timestamp as separate parameter. This encoding ensures consistent
    /// ordering across region splits/merges.
    fn check_self_rollback(&mut self, key: &Key, start_ts: TimeStamp) -> Result<Option<UserMeta>> {
        let raw_key = key.to_raw()?;
        let extra_key = kvengine::encode_extra_txn_status_key(&raw_key, start_ts.into_inner());
        let item = self.snapshot.get(EXTRA_CF, &extra_key, 0);

        // Check if key exists: non-existent keys return items with user_meta_len == 0
        if !item.exists() {
            return Ok(None);
        }

        let user_meta = UserMeta::from_slice(item.user_meta());
        Ok(Some(user_meta))
    }

    /// Scan EXTRA_CF for write conflicts.
    /// Returns immediately upon finding the first conflict.
    fn scan_extra_cf_for_write_conflict(
        &mut self,
        key: &Key,
        start_ts: Option<TimeStamp>,
        max_allowed_write: Option<TimeStamp>,
    ) -> Result<Option<UserMeta>> {
        let raw_key = key.to_raw()?;

        // Try to reuse cached iterator, otherwise create new one
        let mut extra_iter = self.cached_extra_cf_iter.take().unwrap_or_else(|| {
            self.snapshot
                .new_iterator(EXTRA_CF, false, false, None, self.fill_cache)
        });
        extra_iter.seek(&raw_key);

        let mut iterations = 0usize;
        // Raw keys can interleave in the EXTRA CF.
        // This loop handles it well by checking the prefix and the length match.
        // TODO: due to raw_key interleaving, the scan may face critical performance
        // degradation in extreme cases, e.g., a lot of keys share the same
        // prefix. We rely on the fix of the interleaving issue to solve this problem.
        while extra_iter.valid() && extra_iter.key().starts_with(&raw_key) {
            if extra_iter.key().len() == raw_key.len() + 8 {
                let um = UserMeta::from_slice(extra_iter.user_meta());
                let record_start_ts = TimeStamp::new(um.start_ts);
                iterations += 1;

                // Priority 1: Check for self-rollback (exact match)
                // Note: Self-rollback does not necessarily precede newer writes
                if let Some(target_start_ts) = start_ts
                    && um.is_rollback()
                    && record_start_ts == target_start_ts
                {
                    EXTRA_CF_SCAN_ITERATIONS.observe(iterations as f64);
                    // Cache iterator for potential reuse
                    self.cached_extra_cf_iter = Some(extra_iter);
                    return Ok(Some(um));
                }

                // Priority 2: Check for newer conflicts (only `Op_Lock` records, not Rollbacks)
                if let Some(threshold) = max_allowed_write {
                    if !um.is_rollback() && um.commit_ts > threshold.into_inner() {
                        EXTRA_CF_SCAN_ITERATIONS.observe(iterations as f64);
                        // Cache iterator for potential reuse
                        self.cached_extra_cf_iter = Some(extra_iter);
                        return Ok(Some(um));
                    }
                }
            }
            extra_iter.next();
        }
        EXTRA_CF_SCAN_ITERATIONS.observe(iterations as f64);
        // Cache iterator for potential reuse
        self.cached_extra_cf_iter = Some(extra_iter);
        Ok(None)
    }

    // do not use it for conflict check, use `check_write_conflict_in_extra_cf`
    // instead.
    fn get_extra(&mut self, key: &Key, start_ts: TimeStamp) -> Option<(TimeStamp, Write)> {
        let raw_key = key.to_raw().unwrap();
        let extra_key = kvengine::encode_extra_txn_status_key(&raw_key, start_ts.into_inner());
        let item = self.snapshot.get(EXTRA_CF, &extra_key, 0);
        if !item.exists() {
            return None;
        }
        let user_meta = UserMeta::from_slice(item.user_meta());
        Some(if user_meta.is_rollback() {
            (start_ts, Write::new(WriteType::Rollback, start_ts, None))
        } else {
            (
                user_meta.commit_ts.into(),
                Write::new(WriteType::Lock, start_ts, None),
            )
        })
    }

    pub fn load_lock(&mut self, key: &Key) -> Result<Option<Lock>> {
        let raw_key = key.to_raw().unwrap();
        let item = self.snapshot.get(LOCK_CF, &raw_key, 0);
        self.statistics.lock.get += 1;
        self.statistics.lock.flow_stats.read_keys += 1;
        self.statistics.lock.flow_stats.read_bytes += item.value_len();
        self.statistics.lock.processed_keys += 1;
        if item.value_len() == 0 {
            return Ok(None);
        }
        let lock = Lock::parse(item.get_value())?;
        Ok(Some(lock))
    }

    #[maybe_async::both]
    pub async fn get(
        &mut self,
        key: &Key,
        ts: TimeStamp,
        _gc_fence_limit: Option<TimeStamp>,
    ) -> Result<Option<Value>> {
        let raw_key = key.to_raw()?;
        let item = self.snapshot.get(WRITE_CF, &raw_key, ts.into_inner()).await;
        self.statistics.write.get += 1;
        self.statistics.write.flow_stats.read_bytes += raw_key.len() + item.value_len();
        self.statistics.write.flow_stats.read_keys += 1;
        self.statistics.write.processed_keys += 1;
        self.statistics.processed_size += raw_key.len() + item.value_len();
        if item.value_len() > 0 {
            return Ok(Some(item.get_value().to_vec()));
        }
        Ok(None)
    }

    #[maybe_async::both]
    pub async fn get_write(
        &mut self,
        key: &Key,
        ts: TimeStamp,
        _gc_fence_limit: Option<TimeStamp>,
    ) -> Result<Option<Write>> {
        self.seek_write(key, ts)
            .await
            .map(|opt| opt.map(|(_, write)| write))
    }

    #[maybe_async::both]
    pub async fn scan_write_for_key(
        &mut self,
        key: &Key,
        ts: TimeStamp,
    ) -> Result<Vec<(TimeStamp, Write)>> {
        let raw_key = key.to_raw()?;
        let mut res = vec![];
        let mut cursor = self
            .snapshot
            .new_iterator(WRITE_CF, false, true, None, self.fill_cache)
            .await;
        cursor.seek(&raw_key).await;

        while cursor.valid() {
            if cursor.key() != raw_key {
                break;
            }
            if !cursor.user_meta().is_empty() {
                let user_meta = UserMeta::from_slice(cursor.user_meta());
                // only keep the write whose commit ts less or equal than required ts.
                if user_meta.commit_ts <= ts.into_inner() {
                    let parsed = parse_write(&user_meta, cursor.val());
                    res.push(parsed);
                }
            }
            cursor.next().await;
        }
        Ok(res)
    }

    #[maybe_async::both]
    pub async fn seek_write(
        &mut self,
        key: &Key,
        ts: TimeStamp,
    ) -> Result<Option<(TimeStamp, Write)>> {
        let raw_key = key.to_raw()?;
        let item = self.snapshot.get(WRITE_CF, &raw_key, ts.into_inner()).await;
        self.statistics.write.seek += 1;
        self.statistics.write.flow_stats.read_keys += 1;
        self.statistics.write.flow_stats.read_bytes += raw_key.len() + item.value_len();
        self.statistics.write.processed_keys += 1;
        self.statistics.processed_size += raw_key.len() + item.value_len();
        if item.user_meta_len() > 0 {
            let user_meta = UserMeta::from_slice(item.user_meta());
            let (commit_ts, write) = parse_write(&user_meta, item.get_value());
            return Ok(Some((commit_ts, write)));
        }
        Ok(None)
    }

    #[maybe_async::both]
    #[inline(always)]
    pub async fn get_old_value(
        &mut self,
        key: &Key,
        start_ts: TimeStamp,
        prev_write: Option<Write>,
    ) -> Result<OldValue> {
        if let Some(write) = prev_write {
            if write.write_type == WriteType::Delete {
                return Ok(OldValue::None);
            }
            // Locks and Rolbacks are stored in extra CF, will not be seeked by seek_write.
            assert_eq!(write.write_type, WriteType::Put);
            return Ok(OldValue::value(write.short_value.unwrap()));
        }
        let raw_key = key.to_raw()?;
        let item = self
            .snapshot
            .get(WRITE_CF, &raw_key, start_ts.into_inner())
            .await;
        if item.value_len() > 0 {
            return Ok(OldValue::value(item.get_value().to_vec()));
        }
        Ok(OldValue::None)
    }

    /// Scan locks that satisfies `filter(lock)` returns true, from the given
    /// start key `start`. At most `limit` locks will be returned. If
    /// `limit` is set to `0`, it means unlimited.
    ///
    /// The return type is `(locks, is_remain)`. `is_remain` indicates whether
    /// there MAY be remaining locks that can be scanned.
    pub fn scan_locks<F>(
        &mut self,
        start: Option<&Key>,
        end: Option<&Key>,
        filter: F,
        limit: usize,
    ) -> Result<(Vec<(Key, Lock)>, bool)>
    where
        F: Fn(&Lock) -> bool,
    {
        let mut locks = vec![];
        let mut lock_iter =
            self.snapshot
                .new_iterator(LOCK_CF, false, false, None, self.fill_cache);
        let lower_bound: Bytes = if let Some(start) = start {
            Bytes::from(start.to_raw()?)
        } else {
            Bytes::copy_from_slice(self.snapshot.get_start_key())
        };
        let upper_bound = if let Some(k) = end {
            Bytes::from(k.to_raw()?)
        } else {
            self.snapshot.clone_end_key()
        };
        if lock_iter.set_range(lower_bound, upper_bound) {
            self.statistics.lock.seek += 1;
        }
        while lock_iter.valid() {
            let key = Key::from_raw(lock_iter.key());
            if let Some(end) = end {
                if key >= *end {
                    return Ok((locks, false));
                }
            }
            let val = lock_iter.val();
            self.statistics.lock.next += 1;
            self.statistics.lock.flow_stats.read_keys += 1;
            self.statistics.lock.flow_stats.read_bytes += val.len();
            self.statistics.lock.processed_keys += 1;
            let lock = Lock::parse(val)?;
            if filter(&lock) {
                locks.push((key, lock));
                if limit > 0 && locks.len() == limit {
                    return Ok((locks, true));
                }
            }
            lock_iter.next();
        }
        Ok((locks, false))
    }

    /// Returns an arbitrary write that is newer than `ts`, or None if no such
    /// write exists.
    #[maybe_async::both]
    pub async fn get_newer_from_write_cf(
        &mut self,
        key: &Key,
        ts: TimeStamp,
    ) -> Result<Option<(TimeStamp, Write)>> {
        let raw_key = key.to_raw()?;
        let item = self
            .snapshot
            .get_newer(WRITE_CF, &raw_key, ts.into_inner())
            .await;
        if item.user_meta_len() > 0 {
            let user_meta = UserMeta::from_slice(item.user_meta());
            let (commit_ts, write) = parse_write(&user_meta, item.get_value());
            return Ok(Some((commit_ts, write)));
        }
        Ok(None)
    }

    /// Return the first committed key for which `start_ts` equals to `ts`
    // TODO: support async.
    pub fn seek_ts(&mut self, ts: TimeStamp) -> Result<Option<Key>> {
        let mut it = self
            .snapshot
            .new_iterator(WRITE_CF, false, true, None, self.fill_cache);
        it.rewind();
        while it.valid() {
            debug_assert!(!kvengine::table::is_deleted(it.meta()));
            // TODO: remove this check, iterator should not return deleted records.
            if kvengine::table::is_deleted(it.meta()) {
                it.next();
                continue;
            }
            let user_meta = UserMeta::from_slice(it.user_meta());
            if user_meta.start_ts == ts.into_inner() {
                return Ok(Some(Key::from_raw(it.key())));
            }
            it.next()
        }
        Ok(None)
    }

    pub fn get_extras(&mut self, key: &Key) -> Vec<(TimeStamp, Write)> {
        let raw_key = key.to_raw().unwrap();
        let mut writes = vec![];
        let mut extra_iter = self
            .snapshot
            .new_iterator(EXTRA_CF, false, false, None, true);
        extra_iter.seek(&raw_key);
        while extra_iter.valid() {
            if !extra_iter.key().starts_with(&raw_key) {
                break;
            }
            if extra_iter.key().len() == raw_key.len() + 8 {
                let um = UserMeta::from_slice(extra_iter.user_meta());
                let (ts, write_type) = if um.is_rollback() {
                    (um.start_ts, WriteType::Rollback)
                } else {
                    (um.commit_ts, WriteType::Lock)
                };
                writes.push((ts.into(), Write::new(write_type, um.start_ts.into(), None)));
            }
            extra_iter.next();
        }
        writes
    }
}

pub fn parse_write(user_meta: &UserMeta, value: &[u8]) -> (TimeStamp, Write) {
    let commit_ts = user_meta.commit_ts;
    let write_type: WriteType;
    let short_value: Option<Value>;
    if value.is_empty() {
        write_type = WriteType::Delete;
        short_value = None;
    } else {
        write_type = WriteType::Put;
        short_value = Some(value.to_vec())
    }
    (
        TimeStamp::new(commit_ts),
        Write::new(write_type, TimeStamp::new(user_meta.start_ts), short_value),
    )
}
