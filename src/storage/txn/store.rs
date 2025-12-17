// Copyright 2016 TiKV Project Authors. Licensed under Apache-2.0.

use txn_types::{Key, KvPair, Lock, OldValue, TimeStamp, Value, WriteRef};

use super::{Error, ErrorInner, Result};
use crate::storage::{
    kv::Statistics,
    mvcc::{Error as MvccError, ErrorInner as MvccErrorInner, NewerTsCheckState},
};

#[maybe_async::async_trait]
pub trait Store: Send {
    /// The scanner type returned by `scanner()`.
    type Scanner: Scanner;

    /// Fetch the provided key.
    #[maybe_async]
    fn get(&self, key: &Key, statistics: &mut Statistics) -> Result<Option<Value>>;

    /// Re-use last cursor to incrementally (if possible) fetch the provided
    /// key.
    #[maybe_async]
    fn incremental_get(&mut self, key: &Key) -> Result<Option<Value>>;

    /// Take the statistics. Currently only available for `incremental_get`.
    fn incremental_get_take_statistics(&mut self) -> Statistics;

    /// Whether there was data > ts during previous incremental gets.
    fn incremental_get_met_newer_ts_data(&self) -> NewerTsCheckState;

    /// Whether checks the newer ts data
    fn is_check_has_newer_ts_data(&self) -> bool;

    /// Fetch the provided set of keys.
    #[maybe_async]
    fn batch_get(
        &self,
        keys: &[Key],
        statistics: &mut Vec<Statistics>,
    ) -> Result<Vec<Result<Option<Value>>>>;

    /// Retrieve a scanner over the bounds.
    #[maybe_async]
    fn scanner(
        &mut self,
        desc: bool,
        key_only: bool,
        check_has_newer_ts_data: bool,
        lower_bound: Option<Key>,
        upper_bound: Option<Key>,
    ) -> Result<Self::Scanner>;

    fn get_kvengine_snap(&self) -> Option<kvengine::SnapAccess> {
        None
    }

    fn is_sync(&self) -> bool {
        true
    }

    fn get_read_ts(&self) -> u64 {
        u64::MAX
    }
}

/// [`Scanner`]s allow retrieving items or batches from a scan result.
///
/// Commonly they are obtained as a result of a [`scanner`](Store::scanner)
/// operation.
#[maybe_async::async_trait]
pub trait Scanner: Send {
    /// Get the next [`KvPair`](KvPair) if it exists.
    #[maybe_async]
    fn next(&mut self) -> Result<Option<(Key, Value)>>;

    /// Get the next [`KvPair`](KvPair)s up to `limit` if they exist.
    /// If `sample_step` is greater than 0, skips `sample_step - 1` number of
    /// keys after each returned key.
    #[maybe_async]
    async fn scan(&mut self, limit: usize, sample_step: usize) -> Result<Vec<Result<KvPair>>> {
        debug!("scan limit {}", limit);
        let mut row_count = 0;
        let mut results = Vec::with_capacity(limit);
        while results.len() < limit {
            match self.next().await {
                Ok(Some((k, v))) => {
                    if sample_step > 0 {
                        row_count += 1;
                        if (row_count - 1) % sample_step != 0 {
                            continue;
                        }
                    }
                    results.push(Ok((k.to_raw()?, v)));
                }
                Ok(None) => break,
                Err(
                    e @ Error(box ErrorInner::Mvcc(MvccError(box MvccErrorInner::KeyIsLocked {
                        ..
                    }))),
                ) => {
                    // If we return key level error, TiDB will assume the key is in order, cause
                    // DDL skip a table range to build the table index.
                    // In cloud storage engine, the lock CF is checked first, then a large lock
                    // cf key maybe returned before write CF.
                    // So here we return a response level error, TiDB will resolve the lock and
                    // retry the scan request.
                    return Err(e);
                }
                Err(e) => return Err(e),
            }
        }
        Ok(results)
    }

    /// Whether there was data > ts during previous scans.
    fn met_newer_ts_data(&self) -> NewerTsCheckState;

    /// Take statistics.
    fn take_statistics(&mut self) -> Statistics;

    // Reset the range for the scanner, the caller need create a new
    // scanner if false is returned.
    fn reset_range(
        &mut self,
        _desc: bool,
        _lower_bound: Option<Key>,
        _upper_bound: Option<Key>,
    ) -> Result<bool> {
        Ok(false)
    }
}

pub trait TxnEntryStore: Send {
    /// The scanner type returned by `scanner()`.
    type Scanner: TxnEntryScanner;

    /// Retrieve a scanner over the bounds.
    fn entry_scanner(
        &self,
        lower_bound: Option<Key>,
        upper_bound: Option<Key>,
        after_ts: TimeStamp,
        output_delete: bool,
    ) -> Result<Self::Scanner>;
}

/// [`TxnEntryScanner`] allows retrieving items or batches from a scan result.
///
/// Commonly they are obtained as a result of a
/// [`entry_scanner`](TxnEntryStore::entry_scanner) operation.
pub trait TxnEntryScanner: Send {
    fn next_entry(&mut self) -> Result<Option<TxnEntry>>;

    fn scan_entries(&mut self, batch: &mut EntryBatch) -> Result<()> {
        while batch.entries.len() < batch.entries.capacity() {
            match self.next_entry() {
                Ok(Some(entry)) => {
                    batch.entries.push(entry);
                }
                Ok(None) => break,
                Err(e) => return Err(e),
            }
        }
        Ok(())
    }

    /// Take statistics.
    fn take_statistics(&mut self) -> Statistics;
}

/// A transaction entry in underlying storage.
#[derive(PartialEq, Debug, Clone)]
pub enum TxnEntry {
    Prewrite {
        default: KvPair,
        lock: KvPair,
        old_value: OldValue,
    },
    Commit {
        default: KvPair,
        write: KvPair,
        old_value: OldValue,
    },
    // TOOD: Add more entry if needed.
}

impl TxnEntry {
    pub fn old_value(&mut self) -> &mut OldValue {
        match self {
            TxnEntry::Prewrite {
                ref mut old_value, ..
            } => old_value,
            TxnEntry::Commit {
                ref mut old_value, ..
            } => old_value,
        }
    }

    pub fn erasing_last_change_ts(&self) -> TxnEntry {
        let mut e = self.clone();
        match &mut e {
            TxnEntry::Prewrite {
                lock: (_, value), ..
            } => {
                let l = Lock::parse(value).unwrap();
                *value = l.set_last_change(TimeStamp::zero(), 0).to_bytes();
            }
            TxnEntry::Commit {
                write: (_, value), ..
            } => {
                let mut w = WriteRef::parse(value).unwrap();
                w.last_change_ts = TimeStamp::zero();
                w.versions_to_last_change = 0;
                *value = w.to_bytes();
            }
        }
        e
    }
}

impl TxnEntry {
    /// This method will return a kv pair whose
    /// content and encode are same as a kv pair
    /// reture by ```StoreScanner::next```
    pub fn into_kvpair(self) -> Result<(Vec<u8>, Vec<u8>)> {
        match self {
            TxnEntry::Commit { default, write, .. } => {
                if !default.0.is_empty() {
                    let k = Key::from_encoded(default.0).truncate_ts()?;
                    let k = k.into_raw()?;
                    Ok((k, default.1))
                } else {
                    let k = Key::from_encoded(write.0).truncate_ts()?;
                    let k = k.into_raw()?;
                    let v = WriteRef::parse(&write.1)
                        .map_err(MvccError::from)?
                        .to_owned();
                    let v = v.short_value.unwrap_or_default();
                    Ok((k, v))
                }
            }
            // Prewrite are not support
            _ => unreachable!(),
        }
    }
    /// This method will generate this kv pair's key
    pub fn to_key(&self) -> Result<Key> {
        match self {
            TxnEntry::Commit { write, .. } => Ok(Key::from_encoded_slice(
                Key::truncate_ts_for(&write.0).unwrap(),
            )),
            // Prewrite are not support
            _ => unreachable!(),
        }
    }

    pub fn size(&self) -> usize {
        let mut size = 0;
        match self {
            TxnEntry::Commit {
                default,
                write,
                old_value,
            } => {
                size += default.0.len();
                size += default.1.len();
                size += write.0.len();
                size += write.1.len();
                size += old_value.value_size();
            }
            TxnEntry::Prewrite {
                default,
                lock,
                old_value,
            } => {
                size += default.0.len();
                size += default.1.len();
                size += lock.0.len();
                size += lock.1.len();
                size += old_value.value_size();
            }
        }
        size
    }
}

/// A batch of transaction entries.
pub struct EntryBatch {
    entries: Vec<TxnEntry>,
}

impl EntryBatch {
    pub fn with_capacity(cap: usize) -> EntryBatch {
        EntryBatch {
            entries: Vec::with_capacity(cap),
        }
    }

    pub fn clear(&mut self) {
        self.entries.clear();
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.len() == 0
    }

    pub fn iter(&self) -> impl Iterator<Item = &TxnEntry> {
        self.entries.iter()
    }

    pub fn drain(&mut self) -> std::vec::Drain<'_, TxnEntry> {
        self.entries.drain(..)
    }
}

/// A Store that reads on fixtures.
pub struct FixtureStore {
    data: std::collections::BTreeMap<Key, Result<Vec<u8>>>,
}

impl Clone for FixtureStore {
    fn clone(&self) -> Self {
        let data = self
            .data
            .iter()
            .map(|(k, v)| {
                let owned_k = k.clone();
                let owned_v = match v {
                    Ok(v) => Ok(v.clone()),
                    Err(e) => Err(e.maybe_clone().unwrap()),
                };
                (owned_k, owned_v)
            })
            .collect();
        Self { data }
    }
}

impl FixtureStore {
    pub fn new(data: std::collections::BTreeMap<Key, Result<Vec<u8>>>) -> Self {
        FixtureStore { data }
    }
}

impl Store for FixtureStore {
    type Scanner = FixtureStoreScanner;

    #[inline]
    fn get(&self, key: &Key, _statistics: &mut Statistics) -> Result<Option<Vec<u8>>> {
        let r = self.data.get(key);
        match r {
            None => Ok(None),
            Some(Ok(v)) => Ok(Some(v.clone())),
            Some(Err(e)) => Err(e.maybe_clone().unwrap()),
        }
    }

    #[inline]
    fn incremental_get(&mut self, key: &Key) -> Result<Option<Vec<u8>>> {
        let mut s = Statistics::default();
        self.get(key, &mut s)
    }

    #[inline]
    fn incremental_get_take_statistics(&mut self) -> Statistics {
        Statistics::default()
    }

    #[inline]
    fn incremental_get_met_newer_ts_data(&self) -> NewerTsCheckState {
        NewerTsCheckState::Unknown
    }

    fn is_check_has_newer_ts_data(&self) -> bool {
        false
    }

    #[inline]
    fn batch_get(
        &self,
        keys: &[Key],
        statistics: &mut Vec<Statistics>,
    ) -> Result<Vec<Result<Option<Vec<u8>>>>> {
        Ok(keys
            .iter()
            .map(|key| {
                statistics.push(Statistics::default());
                self.get(key, statistics.last_mut().unwrap())
            })
            .collect())
    }

    #[inline]
    fn scanner(
        &mut self,
        desc: bool,
        key_only: bool,
        _: bool,
        lower_bound: Option<Key>,
        upper_bound: Option<Key>,
    ) -> Result<FixtureStoreScanner> {
        use std::ops::Bound;

        let lower = lower_bound.as_ref().map_or(Bound::Unbounded, |v| {
            if !desc {
                Bound::Included(v)
            } else {
                Bound::Excluded(v)
            }
        });
        let upper = upper_bound.as_ref().map_or(Bound::Unbounded, |v| {
            if desc {
                Bound::Included(v)
            } else {
                Bound::Excluded(v)
            }
        });

        let mut vec: Vec<_> = self
            .data
            .range((lower, upper))
            .map(|(k, v)| {
                let owned_k = k.clone();
                let owned_v = if key_only {
                    match v {
                        Ok(_v) => Ok(vec![]),
                        Err(e) => Err(e.maybe_clone().unwrap()),
                    }
                } else {
                    match v {
                        Ok(v) => Ok(v.clone()),
                        Err(e) => Err(e.maybe_clone().unwrap()),
                    }
                };
                (owned_k, owned_v)
            })
            .collect();

        if desc {
            vec.reverse();
        }

        Ok(FixtureStoreScanner {
            // TODO: Remove clone when GATs is available. See rust-lang/rfcs#1598.
            data: vec.into_iter(),
        })
    }
}

/// A Scanner that scans on fixtures.
pub struct FixtureStoreScanner {
    data: std::vec::IntoIter<(Key, Result<Vec<u8>>)>,
}

impl Scanner for FixtureStoreScanner {
    #[inline]
    fn next(&mut self) -> Result<Option<(Key, Vec<u8>)>> {
        let value = self.data.next();
        match value {
            None => Ok(None),
            Some((k, Ok(v))) => Ok(Some((k, v))),
            Some((_k, Err(e))) => Err(e),
        }
    }

    #[inline]
    fn met_newer_ts_data(&self) -> NewerTsCheckState {
        NewerTsCheckState::Unknown
    }

    #[inline]
    fn take_statistics(&mut self) -> Statistics {
        Statistics::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gen_fixture_store() -> FixtureStore {
        use std::collections::BTreeMap;

        let mut data = BTreeMap::default();
        data.insert(Key::from_raw(b"abc"), Ok(b"foo".to_vec()));
        data.insert(Key::from_raw(b"ab"), Ok(b"bar".to_vec()));
        data.insert(Key::from_raw(b"abcd"), Ok(b"box".to_vec()));
        data.insert(Key::from_raw(b"b"), Ok(b"alpha".to_vec()));
        data.insert(Key::from_raw(b"bb"), Ok(b"alphaalpha".to_vec()));
        data.insert(
            Key::from_raw(b"bba"),
            Err(Error::from(ErrorInner::Mvcc(MvccError::from(
                MvccErrorInner::KeyIsLocked(kvproto::kvrpcpb::LockInfo::default()),
            )))),
        );
        data.insert(Key::from_raw(b"z"), Ok(b"beta".to_vec()));
        data.insert(Key::from_raw(b"ca"), Ok(b"hello".to_vec()));
        data.insert(
            Key::from_raw(b"zz"),
            Err(Error::from(ErrorInner::Mvcc(MvccError::from(
                txn_types::Error::from(txn_types::ErrorInner::BadFormatLock),
            )))),
        );

        FixtureStore::new(data)
    }

    #[test]
    fn test_fixture_get() {
        let store = gen_fixture_store();
        let mut statistics = Statistics::default();
        assert_eq!(
            store
                .get(&Key::from_raw(b"not exist"), &mut statistics)
                .unwrap(),
            None
        );
        assert_eq!(
            store.get(&Key::from_raw(b"c"), &mut statistics).unwrap(),
            None
        );
        assert_eq!(
            store.get(&Key::from_raw(b"ab"), &mut statistics).unwrap(),
            Some(b"bar".to_vec())
        );
        assert_eq!(
            store.get(&Key::from_raw(b"caa"), &mut statistics).unwrap(),
            None
        );
        assert_eq!(
            store.get(&Key::from_raw(b"ca"), &mut statistics).unwrap(),
            Some(b"hello".to_vec())
        );
        store
            .get(&Key::from_raw(b"bba"), &mut statistics)
            .unwrap_err();
        assert_eq!(
            store.get(&Key::from_raw(b"bbaa"), &mut statistics).unwrap(),
            None
        );
        assert_eq!(
            store.get(&Key::from_raw(b"abcd"), &mut statistics).unwrap(),
            Some(b"box".to_vec())
        );
        assert_eq!(
            store
                .get(&Key::from_raw(b"abcd\x00"), &mut statistics)
                .unwrap(),
            None
        );
        assert_eq!(
            store
                .get(&Key::from_raw(b"\x00abcd"), &mut statistics)
                .unwrap(),
            None
        );
        assert_eq!(
            store
                .get(&Key::from_raw(b"ab\x00cd"), &mut statistics)
                .unwrap(),
            None
        );
        assert_eq!(
            store.get(&Key::from_raw(b"ab"), &mut statistics).unwrap(),
            Some(b"bar".to_vec())
        );
        store
            .get(&Key::from_raw(b"zz"), &mut statistics)
            .unwrap_err();
        assert_eq!(
            store.get(&Key::from_raw(b"z"), &mut statistics).unwrap(),
            Some(b"beta".to_vec())
        );
    }

    #[test]
    fn test_fixture_scanner() {
        let mut store = gen_fixture_store();

        let mut scanner = store.scanner(false, false, false, None, None).unwrap();
        assert_eq!(
            scanner.next().unwrap(),
            Some((Key::from_raw(b"ab"), b"bar".to_vec()))
        );
        assert_eq!(
            scanner.next().unwrap(),
            Some((Key::from_raw(b"abc"), b"foo".to_vec()))
        );
        assert_eq!(
            scanner.next().unwrap(),
            Some((Key::from_raw(b"abcd"), b"box".to_vec()))
        );
        assert_eq!(
            scanner.next().unwrap(),
            Some((Key::from_raw(b"b"), b"alpha".to_vec()))
        );
        assert_eq!(
            scanner.next().unwrap(),
            Some((Key::from_raw(b"bb"), b"alphaalpha".to_vec()))
        );
        scanner.next().unwrap_err();
        assert_eq!(
            scanner.next().unwrap(),
            Some((Key::from_raw(b"ca"), b"hello".to_vec()))
        );
        assert_eq!(
            scanner.next().unwrap(),
            Some((Key::from_raw(b"z"), b"beta".to_vec()))
        );
        scanner.next().unwrap_err();
        // note: mvcc impl does not guarantee to work any more after meeting a non lock
        // error
        assert_eq!(scanner.next().unwrap(), None);

        let mut scanner = store.scanner(true, false, false, None, None).unwrap();
        scanner.next().unwrap_err();
        // note: mvcc impl does not guarantee to work any more after meeting a non lock
        // error
        assert_eq!(
            scanner.next().unwrap(),
            Some((Key::from_raw(b"z"), b"beta".to_vec()))
        );
        assert_eq!(
            scanner.next().unwrap(),
            Some((Key::from_raw(b"ca"), b"hello".to_vec()))
        );
        scanner.next().unwrap_err();
        assert_eq!(
            scanner.next().unwrap(),
            Some((Key::from_raw(b"bb"), b"alphaalpha".to_vec()))
        );
        assert_eq!(
            scanner.next().unwrap(),
            Some((Key::from_raw(b"b"), b"alpha".to_vec()))
        );
        assert_eq!(
            scanner.next().unwrap(),
            Some((Key::from_raw(b"abcd"), b"box".to_vec()))
        );
        assert_eq!(
            scanner.next().unwrap(),
            Some((Key::from_raw(b"abc"), b"foo".to_vec()))
        );
        assert_eq!(
            scanner.next().unwrap(),
            Some((Key::from_raw(b"ab"), b"bar".to_vec()))
        );
        assert_eq!(scanner.next().unwrap(), None);

        let mut scanner = store.scanner(false, true, false, None, None).unwrap();
        assert_eq!(
            scanner.next().unwrap(),
            Some((Key::from_raw(b"ab"), vec![]))
        );
        assert_eq!(
            scanner.next().unwrap(),
            Some((Key::from_raw(b"abc"), vec![]))
        );
        assert_eq!(
            scanner.next().unwrap(),
            Some((Key::from_raw(b"abcd"), vec![]))
        );
        assert_eq!(scanner.next().unwrap(), Some((Key::from_raw(b"b"), vec![])));
        assert_eq!(
            scanner.next().unwrap(),
            Some((Key::from_raw(b"bb"), vec![]))
        );
        scanner.next().unwrap_err();
        assert_eq!(
            scanner.next().unwrap(),
            Some((Key::from_raw(b"ca"), vec![]))
        );
        assert_eq!(scanner.next().unwrap(), Some((Key::from_raw(b"z"), vec![])));
        scanner.next().unwrap_err();
        // note: mvcc impl does not guarantee to work any more after meeting a non lock
        // error
        assert_eq!(scanner.next().unwrap(), None);

        let mut scanner = store
            .scanner(
                false,
                true,
                false,
                Some(Key::from_raw(b"abc")),
                Some(Key::from_raw(b"abcd")),
            )
            .unwrap();
        assert_eq!(
            scanner.next().unwrap(),
            Some((Key::from_raw(b"abc"), vec![]))
        );
        assert_eq!(scanner.next().unwrap(), None);

        let mut scanner = store
            .scanner(
                false,
                true,
                false,
                Some(Key::from_raw(b"abc")),
                Some(Key::from_raw(b"bba")),
            )
            .unwrap();
        assert_eq!(
            scanner.next().unwrap(),
            Some((Key::from_raw(b"abc"), vec![]))
        );
        assert_eq!(
            scanner.next().unwrap(),
            Some((Key::from_raw(b"abcd"), vec![]))
        );
        assert_eq!(scanner.next().unwrap(), Some((Key::from_raw(b"b"), vec![])));
        assert_eq!(
            scanner.next().unwrap(),
            Some((Key::from_raw(b"bb"), vec![]))
        );
        assert_eq!(scanner.next().unwrap(), None);

        let mut scanner = store
            .scanner(
                false,
                true,
                false,
                Some(Key::from_raw(b"b")),
                Some(Key::from_raw(b"c")),
            )
            .unwrap();
        assert_eq!(scanner.next().unwrap(), Some((Key::from_raw(b"b"), vec![])));
        assert_eq!(
            scanner.next().unwrap(),
            Some((Key::from_raw(b"bb"), vec![]))
        );
        scanner.next().unwrap_err();
        assert_eq!(scanner.next().unwrap(), None);

        let mut scanner = store
            .scanner(
                false,
                true,
                false,
                Some(Key::from_raw(b"b")),
                Some(Key::from_raw(b"b")),
            )
            .unwrap();
        assert_eq!(scanner.next().unwrap(), None);

        let mut scanner = store
            .scanner(
                true,
                true,
                false,
                Some(Key::from_raw(b"abc")),
                Some(Key::from_raw(b"abcd")),
            )
            .unwrap();
        assert_eq!(
            scanner.next().unwrap(),
            Some((Key::from_raw(b"abcd"), vec![]))
        );
        assert_eq!(scanner.next().unwrap(), None);

        let mut scanner = store
            .scanner(
                true,
                true,
                false,
                Some(Key::from_raw(b"abc")),
                Some(Key::from_raw(b"bba")),
            )
            .unwrap();
        scanner.next().unwrap_err();
        assert_eq!(
            scanner.next().unwrap(),
            Some((Key::from_raw(b"bb"), vec![]))
        );
        assert_eq!(scanner.next().unwrap(), Some((Key::from_raw(b"b"), vec![])));
        assert_eq!(
            scanner.next().unwrap(),
            Some((Key::from_raw(b"abcd"), vec![]))
        );
        assert_eq!(scanner.next().unwrap(), None);
    }

    #[test]
    fn test_txn_entry_size() {
        assert_eq!(
            TxnEntry::Prewrite {
                default: (vec![0; 10], vec![0; 10]),
                lock: (vec![0; 10], vec![0; 10]),
                old_value: OldValue::None,
            }
            .size(),
            40
        );

        assert_eq!(
            TxnEntry::Prewrite {
                default: (vec![0; 10], vec![0; 10]),
                lock: (vec![0; 10], vec![0; 10]),
                old_value: OldValue::value(vec![0; 10]),
            }
            .size(),
            50
        );

        assert_eq!(
            TxnEntry::Commit {
                default: (vec![0; 10], vec![0; 10]),
                write: (vec![0; 10], vec![0; 10]),
                old_value: OldValue::None,
            }
            .size(),
            40
        );

        assert_eq!(
            TxnEntry::Commit {
                default: (vec![0; 10], vec![0; 10]),
                write: (vec![0; 10], vec![0; 10]),
                old_value: OldValue::value(vec![0; 10]),
            }
            .size(),
            50
        );
    }
}

#[cfg(test)]
mod benches {
    use std::collections::BTreeMap;

    use rand::RngCore;

    use super::*;
    use crate::test;

    fn gen_payload(n: usize) -> Vec<u8> {
        let mut data = vec![0; n];
        rand::thread_rng().fill_bytes(&mut data);
        data
    }

    #[bench]
    fn bench_fixture_get(b: &mut test::Bencher) {
        let user_key = gen_payload(64);
        let mut data = BTreeMap::default();
        for i in 0..100 {
            let mut key = user_key.clone();
            key.push(i);
            data.insert(Key::from_raw(&key), Ok(gen_payload(100)));
        }
        let store = FixtureStore::new(data);
        let mut query_user_key = user_key;
        query_user_key.push(10);
        let query_key = Key::from_raw(&query_user_key);
        b.iter(|| {
            let store = test::black_box(&store);
            let mut statistics = Statistics::default();
            let value = store
                .get(test::black_box(&query_key), &mut statistics)
                .unwrap();
            test::black_box(value);
        })
    }

    #[bench]
    fn bench_fixture_batch_get(b: &mut test::Bencher) {
        let mut batch_get_keys = vec![];
        let mut data = BTreeMap::default();
        for _ in 0..100 {
            let user_key = gen_payload(64);
            let key = Key::from_raw(&user_key);
            batch_get_keys.push(key.clone());
            data.insert(key, Ok(gen_payload(100)));
        }
        let store = FixtureStore::new(data);
        b.iter(|| {
            let store = test::black_box(&store);
            let mut statistics = Vec::default();
            let value = store.batch_get(test::black_box(&batch_get_keys), &mut statistics);
            test::black_box(value.unwrap());
        })
    }

    #[bench]
    fn bench_fixture_scanner(b: &mut test::Bencher) {
        let mut data = BTreeMap::default();
        for _ in 0..2000 {
            let user_key = gen_payload(64);
            data.insert(Key::from_raw(&user_key), Ok(gen_payload(100)));
        }
        let mut store = FixtureStore::new(data);
        b.iter(|| {
            let store = test::black_box(&mut store);
            let scanner = store
                .scanner(
                    test::black_box(true),
                    test::black_box(false),
                    test::black_box(false),
                    test::black_box(None),
                    test::black_box(None),
                )
                .unwrap();
            test::black_box(scanner);
        })
    }

    #[bench]
    fn bench_fixture_scanner_next(b: &mut test::Bencher) {
        let mut data = BTreeMap::default();
        for _ in 0..2000 {
            let user_key = gen_payload(64);
            data.insert(Key::from_raw(&user_key), Ok(gen_payload(100)));
        }
        let mut store = FixtureStore::new(data);
        b.iter(|| {
            let store = test::black_box(&mut store);
            let mut scanner = store
                .scanner(
                    test::black_box(true),
                    test::black_box(false),
                    test::black_box(false),
                    test::black_box(None),
                    test::black_box(None),
                )
                .unwrap();
            for _ in 0..1000 {
                let v = scanner.next().unwrap();
                test::black_box(v);
            }
        })
    }

    #[bench]
    fn bench_fixture_scanner_scan(b: &mut test::Bencher) {
        let mut data = BTreeMap::default();
        for _ in 0..2000 {
            let user_key = gen_payload(64);
            data.insert(Key::from_raw(&user_key), Ok(gen_payload(100)));
        }
        let mut store = FixtureStore::new(data);
        b.iter(|| {
            let store = test::black_box(&mut store);
            let mut scanner = store
                .scanner(
                    test::black_box(true),
                    test::black_box(false),
                    test::black_box(false),
                    test::black_box(None),
                    test::black_box(None),
                )
                .unwrap();
            test::black_box(scanner.scan(1000, 0).unwrap());
        })
    }
}
