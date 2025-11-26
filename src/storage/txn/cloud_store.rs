// Copyright 2022 TiKV Project Authors. Licensed under Apache-2.0.

use std::{borrow::Cow, marker::PhantomData};

use bytes::{Buf, Bytes};
use kvengine::{read, Item, SnapAccess, UserMeta};
use kvproto::kvrpcpb::IsolationLevel;
use tikv_kv::{Snapshot, Statistics};
use tikv_util::txn_debug;
use txn_types::{
    is_short_value, Key, Lock, LockType, OldValue, TimeStamp, TsSet, Value, Write, WriteType,
    SHORT_VALUE_MAX_LEN,
};

use crate::storage::{
    mvcc,
    mvcc::NewerTsCheckState,
    txn::{Error, ErrorInner, Result, TxnEntry, TxnEntryStore},
    REQUEST_EXCEED_BOUND,
};

pub struct CloudStore<S: Snapshot> {
    marker: PhantomData<S>,
    snapshot: kvengine::SnapAccess,
    start_ts: u64,
    bypass_locks: TsSet,
    fill_cache: bool,
    stats: Statistics,
}

const WRITE_CF: usize = 0;
const LOCK_CF: usize = 1;

#[maybe_async::async_trait]
impl<S: Snapshot> super::Store for CloudStore<S> {
    type Scanner = CloudStoreScanner;

    #[maybe_async]
    async fn get(&self, user_key: &Key, statistics: &mut Statistics) -> Result<Option<Value>> {
        txn_debug!(
            trace_event::types::Category::ReadDetails,
            "CloudStore::get entry";
            "key" => log_wrappers::Value::key(user_key.as_encoded()),
            "start_ts" => ?self.start_ts,
            "keyspace_id" => self.snapshot.get_keyspace_id(),
            "region_id" => self.snapshot.get_id()
        );

        let item = Self::get_inner(
            user_key,
            &self.snapshot,
            self.start_ts,
            &self.bypass_locks,
            statistics,
        )
        .await?;

        let result = if item.value_len() > 0 {
            Some(item.get_value().to_vec())
        } else {
            None
        };

        txn_debug!(
            trace_event::types::Category::ReadDetails,
            "CloudStore::get result";
            "key" => log_wrappers::Value::key(user_key.as_encoded()),
            "start_ts" => ?self.start_ts,
            "keyspace_id" => self.snapshot.get_keyspace_id(),
            "region_id" => self.snapshot.get_id(),
            "result" => ?result.as_ref().map(|v| log_wrappers::Value::value(v))
        );

        Ok(result)
    }

    #[maybe_async]
    async fn incremental_get(&mut self, user_key: &Key) -> Result<Option<Value>> {
        txn_debug!(
            trace_event::types::Category::ReadDetails,
            "CloudStore::incremental_get entry";
            "key" => log_wrappers::Value::key(user_key.as_encoded()),
            "start_ts" => ?self.start_ts,
            "keyspace_id" => self.snapshot.get_keyspace_id(),
            "region_id" => self.snapshot.get_id()
        );

        let stat = &mut self.stats;
        let item = Self::get_inner(
            user_key,
            &self.snapshot,
            self.start_ts,
            &self.bypass_locks,
            stat,
        )
        .await?;

        let result = if item.value_len() > 0 {
            Some(item.get_value().to_vec())
        } else {
            None
        };

        txn_debug!(trace_event::types::Category::ReadDetails,
            "CloudStore::incremental_get result";
            "key" => log_wrappers::Value::key(user_key.as_encoded()),
            "start_ts" => ?self.start_ts,
            "keyspace_id" => self.snapshot.get_keyspace_id(),
            "region_id" => self.snapshot.get_id(),
            "result" => ?result.as_ref().map(|v| log_wrappers::Value::value(v))
        );

        Ok(result)
    }

    fn incremental_get_take_statistics(&mut self) -> Statistics {
        std::mem::take(&mut self.stats)
    }

    fn incremental_get_met_newer_ts_data(&self) -> NewerTsCheckState {
        NewerTsCheckState::Unknown
    }

    #[maybe_async]
    async fn batch_get(
        &self,
        keys: &[Key],
        statistics: &mut Vec<Statistics>,
    ) -> Result<Vec<Result<Option<Value>>>> {
        txn_debug!(trace_event::types::Category::ReadDetails,
            "CloudStore::batch_get entry";
            "keys_count" => keys.len(),
            "keys" => ?keys,
            "start_ts" => ?self.start_ts,
            "keyspace_id" => self.snapshot.get_keyspace_id(),
            "region_id" => self.snapshot.get_id()
        );

        let mut res_vec = Vec::with_capacity(keys.len());
        for key in keys {
            let mut stats = Statistics::default();
            let res = self.get(key, &mut stats).await;
            res_vec.push(res);
            statistics.push(stats);
        }

        txn_debug!(trace_event::types::Category::ReadDetails,
            "CloudStore::batch_get result";
            "keys_count" => keys.len(),
            "start_ts" => ?self.start_ts,
            "keyspace_id" => self.snapshot.get_keyspace_id(),
            "region_id" => self.snapshot.get_id(),
            "result" => ?res_vec.iter().map(|r| {
                match r {
                    Ok(Some(value)) => Ok(log_wrappers::Value::value(value)),
                    Ok(None) => Ok(log_wrappers::Value::value(b"")),
                    Err(e) => Err(e)
                }
            }).collect::<Vec<_>>()
        );

        Ok(res_vec)
    }

    #[maybe_async]
    async fn scanner(
        &mut self,
        desc: bool,
        _key_only: bool,
        _check_has_newer_ts_data: bool,
        lower_bound: Option<Key>,
        upper_bound: Option<Key>,
    ) -> Result<Self::Scanner> {
        txn_debug!(trace_event::types::Category::ReadDetails,
            "CloudStore::scanner entry";
            "desc" => desc,
            "start_ts" => ?self.start_ts,
            "keyspace_id" => self.snapshot.get_keyspace_id(),
            "region_id" => self.snapshot.get_id(),
            "lower_bound" => ?lower_bound,
            "upper_bound" => ?upper_bound
        );

        self.scanner_inner(desc, lower_bound, upper_bound, false)
            .await
    }

    fn get_kvengine_snap(&self) -> Option<kvengine::SnapAccess> {
        Some(self.snapshot.clone())
    }

    fn is_sync(&self) -> bool {
        self.snapshot.is_sync()
    }

    fn get_read_ts(&self) -> u64 {
        self.start_ts
    }
}

impl<S: Snapshot> CloudStore<S> {
    pub fn new(snapshot: S, start_ts: u64, bypass_locks: TsSet, fill_cache: bool) -> Self {
        Self {
            marker: PhantomData,
            snapshot: snapshot.get_kvengine_snap().unwrap().clone(),
            start_ts,
            bypass_locks,
            fill_cache,
            stats: Statistics::default(),
        }
    }

    #[maybe_async::both]
    async fn get_inner<'a>(
        user_key: &Key,
        snap: &'a kvengine::SnapAccess,
        start_ts: u64,
        bypass_locks: &TsSet,
        statistics: &mut Statistics,
    ) -> mvcc::Result<Item<'a>> {
        tikv_util::set_current_region(snap.get_id());

        let raw_key = user_key.to_raw()?;
        let item = snap.get(LOCK_CF, &raw_key, snap.get_mem_table_version());
        statistics.lock.get += 1;
        statistics.lock.flow_stats.read_keys += 1;
        statistics.lock.flow_stats.read_bytes += raw_key.len() + item.value_len();
        statistics.lock.processed_keys += 1;
        if item.value_len() > 0 {
            let lock = Lock::parse(item.get_value()).unwrap();
            Lock::check_ts_conflict(
                Cow::Borrowed(&lock),
                user_key,
                TimeStamp::new(start_ts),
                bypass_locks,
                IsolationLevel::Si,
            )?;
        }
        if snap.get_start_key() > raw_key.as_slice() || snap.get_end_key() <= raw_key.as_slice() {
            error!(
                "get key {:?} out of snap range {:?}, {:?}, {}:{}",
                raw_key.as_slice(),
                snap.get_start_key(),
                snap.get_end_key(),
                snap.get_id(),
                snap.get_version()
            );
            return Ok(Item::default());
        }
        let item = snap.get(WRITE_CF, &raw_key, start_ts).await;
        statistics.write.get += 1;
        statistics.write.flow_stats.read_keys += 1;
        statistics.write.flow_stats.read_bytes += user_key.len() + item.value_len();
        statistics.write.processed_keys += 1;
        statistics.processed_size += user_key.len() + item.value_len();
        Ok(item)
    }

    pub fn check_locks_in_range(&self, lower_bound: &[u8], upper_bound: &[u8]) -> Result<()> {
        let lower_bound = Some(Key::from_raw(lower_bound));
        let upper_bound = Some(Key::from_raw(upper_bound));
        let (lower_bound, upper_bound) = verify_range(&self.snapshot, lower_bound, upper_bound)?;
        let mut stats = Statistics::default();
        let mut iter = self
            .snapshot
            .new_iterator(LOCK_CF, false, false, None, self.fill_cache);
        if iter.set_range(lower_bound, upper_bound) {
            stats.lock.seek += 1;
        }
        check_locks(&mut iter, &self.bypass_locks, self.start_ts, &mut stats)?;
        Ok(())
    }

    pub fn new_memtable_iterator(&self) -> kvengine::read::Iterator {
        self.snapshot
            .new_memtable_iterator(WRITE_CF, false, false, Some(self.start_ts))
    }

    #[maybe_async::both]
    async fn scanner_inner(
        &self,
        desc: bool,
        lower_bound: Option<Key>,
        upper_bound: Option<Key>,
        output_delete: bool,
    ) -> Result<CloudStoreScanner> {
        tikv_util::set_current_region(self.snapshot.get_id());
        CloudStoreScanner::new(
            self.snapshot.clone(),
            desc,
            self.fill_cache,
            self.bypass_locks.clone(),
            self.start_ts,
            lower_bound,
            upper_bound,
            output_delete,
        )
        .await
    }
}

pub fn check_locks(
    lock_iter: &mut read::Iterator,
    bypass_locks: &TsSet,
    start_ts: u64,
    stats: &mut Statistics,
) -> mvcc::Result<()> {
    while lock_iter.valid() {
        let raw_key = lock_iter.key();
        let key = Key::from_raw(raw_key);
        let raw_key_len = raw_key.len();
        let val = lock_iter.val();
        stats.lock.next += 1;
        stats.lock.flow_stats.read_keys += 1;
        stats.lock.flow_stats.read_bytes += raw_key_len + val.len();
        stats.lock.processed_keys += 1;
        let lock = Lock::parse(val)?;
        Lock::check_ts_conflict(
            Cow::Borrowed(&lock),
            &key,
            start_ts.into(),
            bypass_locks,
            IsolationLevel::Si,
        )?;
        lock_iter.next();
    }
    Ok(())
}

fn verify_range(
    snap: &SnapAccess,
    lower_bound: Option<Key>,
    upper_bound: Option<Key>,
) -> Result<(Bytes, Bytes)> {
    let lower = if let Some(lower_key) = &lower_bound {
        let lower = Bytes::from(lower_key.to_raw()?);
        if lower.chunk() < snap.get_start_key() {
            range_error(snap, &lower_bound, &upper_bound)?;
        }
        lower
    } else {
        snap.get_start_key().to_vec().into()
    };
    let upper = if let Some(upper_key) = &upper_bound {
        let upper = Bytes::from(upper_key.to_raw()?);
        if upper.chunk() > snap.get_end_key() {
            range_error(snap, &lower_bound, &upper_bound)?;
        }
        upper
    } else {
        snap.get_end_key().to_vec().into()
    };
    Ok((lower, upper))
}

fn range_error(
    snap: &SnapAccess,
    lower_bound: &Option<Key>,
    upper_bound: &Option<Key>,
) -> Result<()> {
    REQUEST_EXCEED_BOUND.inc();
    let start = lower_bound.as_ref().map(|k| k.as_encoded().clone());
    let end = upper_bound.as_ref().map(|k| k.as_encoded().clone());
    let snap_start = Key::from_raw(snap.get_start_key());
    let snap_end = Key::from_raw(snap.get_end_key());
    Err(Error::from(ErrorInner::InvalidReqRange {
        start,
        end,
        lower_bound: Some(snap_start.into_encoded()),
        upper_bound: Some(snap_end.into_encoded()),
    }))
}

/// DeltaScanner scans a key range and yields all MVCC versions of each key
/// after (but excluded) `start_ts` with representing:
/// - committed versions (from Write CF) as `TxnEntry::Commit` (including
///   deletes),
/// - uncommitted prewrite locks (from Lock CF) as `TxnEntry::Prewrite`.
pub struct CloudDeltaScanner {
    lock_iter: kvengine::read::Iterator,
    iter: kvengine::read::Iterator,
    start_ts: u64,
    pub stats: Statistics,
    lower_bound: Bytes,
    upper_bound: Bytes,
    filter_by_ts: bool,

    is_started: bool,
    current_raw_key: Vec<u8>,
    snap: SnapAccess,
}

pub enum DeltaEntry {
    Version {
        user_meta: UserMeta,
        value: Vec<u8>,
        encoded_key: Key,
        old_value: OldValue,
    },
    Lock {
        lock: Lock,
        raw_key: Vec<u8>,
        old_value: OldValue,
    },
}

impl std::fmt::Debug for DeltaEntry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Version {
                user_meta,
                value,
                encoded_key,
                old_value,
            } => f
                .debug_struct("Version")
                .field("user_meta", user_meta)
                .field("value", &format_args!("<{} byte(s)>", value.len()))
                .field("encoded_key", encoded_key)
                .field("old_value", old_value)
                .finish(),
            Self::Lock {
                lock,
                raw_key,
                old_value,
            } => f
                .debug_struct("Lock")
                .field("lock", lock)
                .field("raw_key", &log_wrappers::Value::key(raw_key))
                .field("old_value", old_value)
                .finish(),
        }
    }
}

impl DeltaEntry {
    fn into_txn_entry(self) -> TxnEntry {
        match self {
            DeltaEntry::Version {
                user_meta,
                value,
                encoded_key,
                old_value,
            } => {
                let write_key = encoded_key.clone().append_ts(user_meta.commit_ts.into());
                let write_ty = if value.is_empty() {
                    WriteType::Delete
                } else {
                    WriteType::Put
                };
                let default_key = encoded_key.append_ts(user_meta.start_ts.into());
                if value.len() < SHORT_VALUE_MAX_LEN {
                    let write_record = Write::new(write_ty, user_meta.start_ts.into(), Some(value));
                    TxnEntry::Commit {
                        default: (vec![], vec![]),
                        write: (write_key.into_encoded(), write_record.as_ref().to_bytes()),
                        old_value,
                    }
                } else {
                    let write_record = Write::new(write_ty, user_meta.start_ts.into(), None);
                    let default_value = value;
                    TxnEntry::Commit {
                        default: (default_key.into_encoded(), default_value),
                        write: (write_key.into_encoded(), write_record.as_ref().to_bytes()),
                        old_value,
                    }
                }
            }
            DeltaEntry::Lock {
                mut lock,
                raw_key,
                old_value,
            } => {
                assert!(
                    lock.short_value.is_some() || lock.lock_type == LockType::Delete,
                    "lock value in cloud engine without short value; lock = {:?}, key = {}",
                    lock,
                    log_wrappers::Value::key(&raw_key)
                );
                let encoded_key = Key::from_raw(&raw_key).into_encoded();
                if lock.short_value.as_ref().map(|v| v.len()).unwrap_or(0) > SHORT_VALUE_MAX_LEN {
                    let val = lock.short_value.take().unwrap();
                    let default_key = Key::from_raw(&raw_key).append_ts(lock.ts);
                    return TxnEntry::Prewrite {
                        default: (default_key.into_encoded(), val),
                        lock: (encoded_key, lock.to_bytes()),
                        old_value,
                    };
                }

                TxnEntry::Prewrite {
                    default: (vec![], vec![]),
                    lock: (encoded_key, lock.to_bytes()),
                    old_value,
                }
            }
        }
    }
}

impl CloudDeltaScanner {
    #[maybe_async::both]
    pub async fn new(
        snap: SnapAccess,
        start_ts: u64,
        lower_bound: Option<Key>,
        upper_bound: Option<Key>,
        filter_by_ts: bool,
    ) -> Self {
        let stats = Statistics::default();
        let lock_iter = snap.new_iterator(LOCK_CF, false, false, None, false);
        let iter = if filter_by_ts {
            snap.new_delta_iterator(false, start_ts)
        } else {
            snap.new_iterator(WRITE_CF, false, true, None, false)
        };
        let (lower_bound, upper_bound) =
            verify_range(&snap, lower_bound, upper_bound).unwrap_or((Bytes::new(), Bytes::new()));
        Self {
            lock_iter,
            start_ts,
            iter,
            lower_bound,
            upper_bound,
            stats,
            filter_by_ts,
            is_started: false,
            current_raw_key: vec![],
            snap,
        }
    }

    #[maybe_async::both]
    pub async fn init(&mut self) -> Result<()> {
        if self
            .lock_iter
            .set_range(self.lower_bound.clone(), self.upper_bound.clone())
        {
            self.stats.lock.seek += 1;
        }
        if self
            .iter
            .set_range(self.lower_bound.clone(), self.upper_bound.clone())
        {
            self.stats.write.seek += 1;
        }
        self.is_started = true;
        if self.iter.valid() {
            self.current_raw_key = self.iter.key().to_vec();
        }
        Ok(())
    }

    #[maybe_async::both]
    async fn get_old_value(&self, key: &[u8], before_ts: u64) -> OldValue {
        let item = self.snap.get(WRITE_CF, key, before_ts - 1).await;
        let v = item.get_value();
        if v.is_empty() {
            OldValue::None
        } else {
            OldValue::value(v.to_vec())
        }
    }

    #[maybe_async::both]
    pub async fn next_inner(&mut self) -> Result<Option<DeltaEntry>> {
        if !self.is_started {
            self.init().await?;
        }

        loop {
            // As we assume the lock cf's kv count should be relatively small, we use
            // `get_old_value` instead of reusing the write cf's iterator to
            // simplify the code.
            if self.lock_iter.valid() {
                let raw_key_len = self.lock_iter.key().len();
                let val_len = self.lock_iter.val().len();
                if self.current_raw_key.is_empty() // Write iter reachs the end...
                    // or it exceeds the lock iter. (We should yield Prewrite records before Commits.)
                    || self.lock_iter.key() <= self.current_raw_key.as_slice()
                {
                    let lock = Lock::parse(self.lock_iter.val()).map_err(Error::from_mvcc)?;
                    // ignore pessimistic or SELECT FOR UPDATE lock.
                    if lock.lock_type == LockType::Pessimistic || lock.lock_type == LockType::Lock {
                        self.lock_iter.next().await;
                        continue;
                    }

                    let lock_ts = std::cmp::max(lock.ts, lock.for_update_ts);
                    let entry = DeltaEntry::Lock {
                        lock,
                        raw_key: self.lock_iter.key().to_vec(),
                        // NOTE: maybe read it from the `WRITE_CF` state.
                        // if `lock.key == write.key` => old value is `write.val`
                        old_value: self
                            .get_old_value(self.lock_iter.key(), lock_ts.into_inner())
                            .await,
                    };

                    self.lock_iter.next().await;
                    self.stats.lock.next += 1;
                    self.stats.lock.flow_stats.read_keys += 1;
                    self.stats.lock.flow_stats.read_bytes += raw_key_len + val_len;
                    self.stats.lock.processed_keys += 1;

                    return Ok(Some(entry));
                }
            }

            if self.iter.valid() {
                let iter_key = self.iter.key();
                let key_len = iter_key.len();
                let key = Key::from_raw(iter_key);
                let user_meta = UserMeta::from_slice(self.iter.user_meta());

                if user_meta.commit_ts <= self.start_ts {
                    debug!("delta scanner get a commit record less than start_ts";
                        "key" => log_wrappers::Value::key(key.as_encoded()),
                        "user_meta" => ?user_meta,
                        "start_ts" => ?self.start_ts,
                    );
                    self.iter.next().await;
                    if !self.iter.valid() || self.iter.key() != self.current_raw_key {
                        self.current_raw_key.truncate(0);
                        if self.iter.valid() {
                            self.current_raw_key.extend_from_slice(self.iter.key());
                        }
                    }
                    continue;
                }

                let val = self.iter.val();
                let val_len = val.len();
                let val_vec = val.to_vec();

                self.iter.next().await;
                self.stats.write.next += 1;
                self.stats.write.flow_stats.read_keys += 1;
                self.stats.write.flow_stats.read_bytes += key_len + val_len;
                let old_value = if !self.iter.valid() || self.iter.key() != self.current_raw_key {
                    let old_value = if self.filter_by_ts {
                        self.get_old_value(&self.current_raw_key, user_meta.commit_ts)
                            .await
                    } else {
                        OldValue::None
                    };
                    self.stats.write.processed_keys += 1;
                    self.current_raw_key.truncate(0);
                    if self.iter.valid() {
                        self.current_raw_key.extend_from_slice(self.iter.key());
                    }
                    old_value
                } else {
                    // Here we are in the previous version.
                    let prev = self.iter.val();
                    if prev.is_empty() {
                        OldValue::None
                    } else {
                        OldValue::value(prev.to_vec())
                    }
                };
                self.stats.processed_size += key_len + val_len;

                let entry = DeltaEntry::Version {
                    user_meta,
                    value: val_vec,
                    encoded_key: key,
                    old_value,
                };
                return Ok(Some(entry));
            }

            return Ok(None);
        }
    }
}

impl super::TxnEntryScanner for CloudDeltaScanner {
    fn next_entry(&mut self) -> Result<Option<TxnEntry>> {
        Ok(self.next_inner()?.map(DeltaEntry::into_txn_entry))
    }

    fn take_statistics(&mut self) -> Statistics {
        std::mem::take(&mut self.stats)
    }
}

pub struct CloudStoreScanner {
    snap: SnapAccess,
    lock_iter: kvengine::read::Iterator,
    bypass_locks: TsSet,
    start_ts: u64,
    iter: kvengine::read::Iterator,
    stats: Statistics,
    is_started: bool,
    lower_bound: Bytes,
    upper_bound: Bytes,
    output_delete: bool,
}

impl CloudStoreScanner {
    #[maybe_async::both]
    pub async fn new(
        snap: SnapAccess,
        desc: bool,
        fill_cache: bool,
        bypass_locks: TsSet,
        start_ts: u64,
        lower_bound: Option<Key>,
        upper_bound: Option<Key>,
        output_delete: bool,
    ) -> Result<Self> {
        let stats = Statistics::default();
        let lock_iter = snap.new_iterator(LOCK_CF, desc, false, None, fill_cache);
        let iter = snap
            .new_iterator(WRITE_CF, desc, false, Some(start_ts), fill_cache)
            .await;
        let (lower_bound, upper_bound) = verify_range(&snap, lower_bound, upper_bound)?;
        Ok(Self {
            snap,
            lock_iter,
            bypass_locks,
            start_ts,
            iter,
            stats,
            is_started: false,
            lower_bound,
            upper_bound,
            output_delete,
        })
    }

    #[maybe_async::both]
    async fn init(&mut self) -> Result<()> {
        if self
            .lock_iter
            .set_range(self.lower_bound.clone(), self.upper_bound.clone())
        {
            self.stats.lock.seek += 1;
        };
        check_locks(
            &mut self.lock_iter,
            &self.bypass_locks,
            self.start_ts,
            &mut self.stats,
        )?;
        if self
            .iter
            .set_range(self.lower_bound.clone(), self.upper_bound.clone())
            .await
        {
            self.stats.write.seek += 1;
        }
        Ok(())
    }

    #[maybe_async::both]
    async fn next_inner(&mut self) -> Result<Option<(Key, UserMeta, Value)>> {
        if self.is_started {
            self.iter.next().await;
        } else {
            self.init().await?;
            self.is_started = true;
        }
        loop {
            if !self.iter.valid() {
                return Ok(None);
            }
            let iter_key = self.iter.key();
            let key_len = iter_key.len();
            let key = Key::from_raw(iter_key);
            let user_meta = UserMeta::from_slice(self.iter.user_meta());
            let val = self.iter.val();
            self.stats.write.next += 1;
            self.stats.write.flow_stats.read_keys += 1;
            self.stats.write.flow_stats.read_bytes += key_len + val.len();
            self.stats.write.processed_keys += 1;
            self.stats.processed_size += key_len + val.len();
            if !val.is_empty() || self.output_delete {
                return Ok(Some((key, user_meta, val.to_vec())));
            }
            self.stats.write.next_tombstone += 1;
            // Skip delete record.
            self.iter.next().await;
            continue;
        }
    }

    pub fn next_with_user_meta(&mut self) -> Result<Option<(Key, UserMeta, Value)>> {
        self.next_inner()
    }
}

#[maybe_async::async_trait]
impl super::Scanner for CloudStoreScanner {
    #[maybe_async]
    async fn next(&mut self) -> Result<Option<(Key, Value)>> {
        let result = self
            .next_inner()
            .await?
            .map(|(key, _user_meta, val)| (key, val));

        txn_debug!(trace_event::types::Category::ReadDetails,
            "CloudStoreScanner::next";
            "start_ts" => ?self.start_ts,
            "keyspace_id" => self.snap.get_keyspace_id(),
            "region_id" => self.snap.get_id(),
            "result" => ?result.as_ref().map(|(key, value)| {
                (key, log_wrappers::Value::value(value))
            })
        );

        Ok(result)
    }

    fn met_newer_ts_data(&self) -> NewerTsCheckState {
        NewerTsCheckState::Unknown
    }

    fn take_statistics(&mut self) -> Statistics {
        std::mem::take(&mut self.stats)
    }

    fn reset_range(
        &mut self,
        desc: bool,
        lower_bound: Option<Key>,
        upper_bound: Option<Key>,
    ) -> Result<bool> {
        if self.iter.is_reverse() != desc {
            return Ok(false);
        }
        let (lower_bound, upper_bound) = verify_range(&self.snap, lower_bound, upper_bound)?;
        self.lower_bound = lower_bound;
        self.upper_bound = upper_bound;
        self.is_started = false;
        Ok(true)
    }
}

impl super::TxnEntryScanner for CloudStoreScanner {
    fn next_entry(&mut self) -> Result<Option<TxnEntry>> {
        Ok(self.next_inner()?.map(|(key, user_meta, val)| {
            let write_key = key.append_ts(TimeStamp::new(user_meta.commit_ts));
            let write_type = if val.is_empty() {
                WriteType::Delete
            } else {
                WriteType::Put
            };
            let (write, default_key, default_val) = if is_short_value(&val) {
                (
                    Write::new(write_type, user_meta.start_ts.into(), Some(val)),
                    vec![],
                    vec![],
                )
            } else {
                let default_key = write_key
                    .clone()
                    .truncate_ts()
                    .unwrap()
                    .append_ts(user_meta.start_ts.into());
                (
                    Write::new(write_type, user_meta.start_ts.into(), None),
                    default_key.into_encoded(),
                    val,
                )
            };
            TxnEntry::Commit {
                default: (default_key, default_val),
                write: (write_key.into_encoded(), write.as_ref().to_bytes()),
                old_value: OldValue::None,
            }
        }))
    }

    fn take_statistics(&mut self) -> Statistics {
        std::mem::take(&mut self.stats)
    }
}

impl<S: Snapshot> TxnEntryStore for CloudStore<S> {
    type Scanner = CloudStoreScanner;
    fn entry_scanner(
        &self,
        lower_bound: Option<Key>,
        upper_bound: Option<Key>,
        _after_ts: TimeStamp,
        output_delete: bool,
    ) -> Result<Self::Scanner> {
        self.scanner_inner(false, lower_bound, upper_bound, output_delete)
    }
}
