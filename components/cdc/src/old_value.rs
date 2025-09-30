// Copyright 2021 TiKV Project Authors. Licensed under Apache-2.0.

use getset::CopyGetters;
use kvengine::WRITE_CF;
use tikv::storage::Statistics;
use tikv_util::{
    config::ReadableSize,
    lru::{LruCache, SizePolicy},
    time::Instant,
};
use txn_types::{Key, MutationType, OldValue, TimeStamp, Value};

use crate::{metrics::*, Result};

pub(crate) type OldValueCallback = Box<
    dyn Fn(Key, TimeStamp, &mut OldValueCache, &mut Statistics) -> Result<Option<Vec<u8>>> + Send,
>;

#[derive(Default)]
pub struct OldValueCacheSizePolicy(usize);

impl SizePolicy<Key, (OldValue, Option<MutationType>)> for OldValueCacheSizePolicy {
    fn current(&self) -> usize {
        self.0
    }

    fn on_insert(&mut self, key: &Key, value: &(OldValue, Option<MutationType>)) {
        self.0 +=
            key.as_encoded().len() + value.0.size() + std::mem::size_of::<Option<MutationType>>();
    }

    fn on_remove(&mut self, key: &Key, value: &(OldValue, Option<MutationType>)) {
        self.0 -=
            key.as_encoded().len() + value.0.size() + std::mem::size_of::<Option<MutationType>>();
    }

    fn on_reset(&mut self, val: usize) {
        self.0 = val;
    }
}

#[derive(CopyGetters)]
pub struct OldValueCache {
    cache: LruCache<Key, (OldValue, Option<MutationType>), OldValueCacheSizePolicy>,
    #[getset(get_copy = "pub")]
    access_count: usize,
    #[getset(get_copy = "pub")]
    miss_count: usize,
    #[getset(get_copy = "pub")]
    miss_none_count: usize,
    #[getset(get_copy = "pub")]
    update_count: usize,
}

impl OldValueCache {
    pub fn new(capacity: ReadableSize) -> OldValueCache {
        CDC_OLD_VALUE_CACHE_MEMORY_QUOTA.set(capacity.0 as i64);
        OldValueCache {
            cache: LruCache::with_capacity_sample_and_trace(
                capacity.0 as usize,
                0,
                OldValueCacheSizePolicy(0),
            ),
            access_count: 0,
            miss_count: 0,
            miss_none_count: 0,
            update_count: 0,
        }
    }

    pub fn insert(&mut self, key: Key, old_value: (OldValue, Option<MutationType>)) {
        self.cache.insert(key, old_value);
        self.update_count += 1;
    }

    pub fn resize(&mut self, new_capacity: ReadableSize) {
        CDC_OLD_VALUE_CACHE_MEMORY_QUOTA.set(new_capacity.0 as i64);
        self.cache.resize(new_capacity.0 as usize);
    }

    pub fn flush_metrics(&mut self) {
        fail::fail_point!("cdc_flush_old_value_metrics", |_| {});
        CDC_OLD_VALUE_CACHE_BYTES.set(self.cache.size() as i64);
        CDC_OLD_VALUE_CACHE_LEN.set(self.cache.len() as i64);
        CDC_OLD_VALUE_CACHE_ACCESS.add(self.access_count as i64);
        CDC_OLD_VALUE_CACHE_MISS.add(self.miss_count as i64);
        CDC_OLD_VALUE_CACHE_MISS_NONE.add(self.miss_none_count as i64);
        self.access_count = 0;
        self.miss_count = 0;
        self.miss_none_count = 0;
        self.update_count = 0;
    }

    #[cfg(test)]
    pub(crate) fn capacity(&self) -> usize {
        self.cache.capacity()
    }
}

/// Fetch old value for `key`. If it can't be found in `old_value_cache`, seek
/// and retrieve it with `query_ts` from `snapshot`.
pub fn get_old_value(
    snapshot: &kvengine::SnapAccess,
    key: Key,
    query_ts: TimeStamp,
    old_value_cache: &mut OldValueCache,
    statistics: &mut Statistics,
) -> Result<Option<Vec<u8>>> {
    let start = Instant::now();
    tikv_util::defer!(
        CDC_OLD_VALUE_DURATION_HISTOGRAM
            .with_label_values(&["all"])
            .observe(start.saturating_elapsed().as_secs_f64())
    );

    old_value_cache.access_count += 1;
    if let Some((old_value, mutation_type)) = old_value_cache.cache.remove(&key) {
        return match mutation_type {
            // Old value of an Insert is guaranteed to be None.
            Some(MutationType::Insert) => {
                assert_eq!(old_value, OldValue::None);
                Ok(None)
            }
            // For Put, Delete or a mutation type we do not know,
            // we read old value from the cache.
            Some(MutationType::Put) | Some(MutationType::Delete) | None => {
                match old_value {
                    OldValue::None => Ok(None),
                    OldValue::Value { value } => Ok(Some(value)),
                    OldValue::ValueTimeStamp { start_ts } => {
                        let value = get_value_from_kvengine(
                            snapshot,
                            key,
                            start_ts.into_inner() - 1,
                            statistics,
                        );
                        Ok(value)
                    }
                    // Unspecified and SeekWrite should not be added into cache.
                    OldValue::Unspecified | OldValue::SeekWrite(_) => unreachable!(),
                }
            }
            _ => unreachable!(),
        };
    }

    // Cannot get old value from cache, seek for it in engine.
    old_value_cache.miss_count += 1;
    let value = get_value_from_kvengine(snapshot, key, query_ts.into_inner() - 1, statistics);
    if value.is_none() {
        old_value_cache.miss_none_count += 1;
    }
    Ok(value)
}

fn get_value_from_kvengine(
    snapshot: &kvengine::SnapAccess,
    key: Key,
    ts: u64,
    statistics: &mut Statistics,
) -> Option<Value> {
    let start = Instant::now();
    tikv_util::defer!(
        CDC_OLD_VALUE_DURATION_HISTOGRAM
            .with_label_values(&["get"])
            .observe(start.saturating_elapsed().as_secs_f64())
    );
    statistics.data.get += 1;

    let item = snapshot.get(WRITE_CF, &key.into_raw().unwrap(), ts);
    let val = item.get_value();
    if val.is_empty() {
        None
    } else {
        Some(val.to_vec())
    }
}

#[cfg(test)]
mod tests {
    use cloud_server::mock_kv_engine::TestKvEngine;
    use kvproto::kvrpcpb::PrewriteRequestPessimisticAction::*;
    use tikv::storage::txn::tests::*;

    use super::*;
    use crate::endpoint::tests::api_v2_key;

    #[track_caller]
    fn must_get_eq(kv_engine: &kvengine::Engine, key: &Key, ts: u64, value: Option<Value>) {
        let raw_key = key.clone().into_raw().unwrap();
        let snapshot = kv_engine.get_shard(1).unwrap().new_snap_access();

        let item = snapshot.get(WRITE_CF, &raw_key, ts - 1);
        match value {
            Some(v) => {
                assert_eq!(&*v, item.get_value());
            }
            None => {
                assert!(item.get_value().is_empty());
            }
        }
    }

    #[test]
    fn test_old_value_resize() {
        let capacity = 1024;

        let mut old_value_cache = OldValueCache::new(ReadableSize(capacity));
        let value = (
            OldValue::Value {
                value: b"value".to_vec(),
            },
            None,
        );

        // The size of each insert.
        let mut size_calc = OldValueCacheSizePolicy::default();
        size_calc.on_insert(&Key::from_raw(&0_usize.to_be_bytes()), &value);
        let size = size_calc.current();

        // Insert ten values.
        let cases = 10_usize;
        for i in 0..cases {
            let key = Key::from_raw(&i.to_be_bytes());
            old_value_cache.cache.insert(key, value.clone());
        }

        assert_eq!(old_value_cache.cache.size(), size * cases);
        assert_eq!(old_value_cache.cache.len(), cases);
        assert_eq!(old_value_cache.capacity(), capacity as usize);

        // Reduces capacity.
        let new_capacity = 256;
        // The memory usage that needs to be removed because of the capacity reduction.
        let dropped = old_value_cache.cache.size() - new_capacity;
        let dropped_count = if dropped % size != 0 {
            (dropped / size) + 1
        } else {
            dropped / size
        };
        // The remaining values count.
        let remaining_count = old_value_cache.cache.len() - dropped_count;
        old_value_cache.resize(ReadableSize(new_capacity as u64));

        assert_eq!(old_value_cache.cache.size(), size * remaining_count);
        assert_eq!(old_value_cache.cache.len(), remaining_count);
        assert_eq!(old_value_cache.capacity(), new_capacity);
        for i in dropped_count..cases {
            let key = Key::from_raw(&i.to_be_bytes());
            assert_eq!(old_value_cache.cache.get(&key).is_some(), true);
        }

        // Increases the capacity again.
        let new_capacity = 1024;
        old_value_cache.resize(ReadableSize(new_capacity));

        assert_eq!(old_value_cache.cache.size(), size * remaining_count);
        assert_eq!(old_value_cache.cache.len(), remaining_count);
        assert_eq!(old_value_cache.capacity(), new_capacity as usize);
        for i in dropped_count..cases {
            let key = Key::from_raw(&i.to_be_bytes());
            assert_eq!(old_value_cache.cache.get(&key).is_some(), true);
        }
    }

    #[test]
    fn test_old_value_reader() {
        let mut engine = TestKvEngine::new().unwrap();
        let kv_engine = engine.engine.clone();
        let raw_key = api_v2_key(b"k");
        let k = raw_key.as_ref();
        let key: Key = Key::from_raw(k);

        must_prewrite_put(&mut engine, k, b"v1", k, 1);
        must_get_eq(&kv_engine, &key, 2, None);
        must_get_eq(&kv_engine, &key, 1, None);
        must_commit(&mut engine, k, 1, 1);
        must_get_eq(&kv_engine, &key, 1, Some(b"v1".to_vec()));

        must_prewrite_put(&mut engine, k, b"v2", k, 2);
        must_get_eq(&kv_engine, &key, 2, Some(b"v1".to_vec()));
        must_rollback(&mut engine, k, 2, false);

        must_prewrite_put(&mut engine, k, b"v3", k, 3);
        must_get_eq(&kv_engine, &key, 3, Some(b"v1".to_vec()));
        must_commit(&mut engine, k, 3, 3);

        must_prewrite_delete(&mut engine, k, k, 4);
        must_get_eq(&kv_engine, &key, 4, Some(b"v3".to_vec()));
        must_commit(&mut engine, k, 4, 4);

        must_prewrite_put(&mut engine, k, vec![b'v'; 5120].as_slice(), k, 5);
        must_get_eq(&kv_engine, &key, 5, None);
        must_commit(&mut engine, k, 5, 5);

        must_prewrite_delete(&mut engine, k, k, 6);
        must_get_eq(&kv_engine, &key, 6, Some(vec![b'v'; 5120]));
        must_rollback(&mut engine, k, 6, false);

        must_prewrite_put(&mut engine, k, b"v4", k, 7);
        must_commit(&mut engine, k, 7, 9);

        must_acquire_pessimistic_lock(&mut engine, k, k, 8, 10);
        must_pessimistic_prewrite_put(&mut engine, k, b"v5", k, 8, 10, DoPessimisticCheck);
        must_get_eq(&kv_engine, &key, 10, Some(b"v4".to_vec()));
        must_commit(&mut engine, k, 8, 11);
    }

    #[test]
    #[ignore = "next gen does not support gc delete kv. old key can only be removed via compaction"]
    fn test_old_value_reader_check_gc_fence() {
        let mut engine = TestKvEngine::new().unwrap();
        let kv_engine = engine.engine.clone();

        // PUT,      Read
        //  `--------------^
        let k1 = api_v2_key(b"k1");
        must_prewrite_put(&mut engine, &k1, b"v1", &k1, 10);
        must_commit(&mut engine, &k1, 10, 20);
        must_get_eq(&kv_engine, &Key::from_raw(&k1), 40, Some(b"v1".to_vec()));
        must_cleanup_with_gc_fence(&mut engine, &k1, 20, 0, 50, true);
        must_get_eq(&kv_engine, &Key::from_raw(&k1), 40, Some(b"v1".to_vec()));

        // PUT,      Read
        //  `---------^
        let k2 = api_v2_key(b"k2");
        must_prewrite_put(&mut engine, &k2, b"v2", &k2, 11);
        must_commit(&mut engine, &k2, 11, 20);
        must_cleanup_with_gc_fence(&mut engine, &k2, 20, 0, 40, true);

        // PUT,      Read
        //  `-----^
        let k3 = api_v2_key(b"k3");
        must_prewrite_put(&mut engine, &k3, b"v3", &k3, 12);
        must_commit(&mut engine, &k3, 12, 20);
        must_cleanup_with_gc_fence(&mut engine, &k3, 20, 0, 30, true);

        // PUT,   PUT,       Read
        //  `-----^ `----^
        let k4 = api_v2_key(b"k4");
        must_prewrite_put(&mut engine, &k4, b"v4", &k4, 13);
        must_commit(&mut engine, &k4, 13, 14);
        must_prewrite_put(&mut engine, &k4, b"v4x", &k4, 15);
        must_commit(&mut engine, &k4, 15, 20);
        must_cleanup_with_gc_fence(&mut engine, &k4, 14, 0, 20, false);
        must_cleanup_with_gc_fence(&mut engine, &k4, 20, 0, 30, true);

        // PUT,   DEL,       Read
        //  `-----^ `----^
        let k5 = api_v2_key(b"k5");
        must_prewrite_put(&mut engine, &k5, b"v5", &k5, 13);
        must_commit(&mut engine, &k5, 13, 14);
        must_prewrite_delete(&mut engine, &k5, b"v5", 15);
        must_commit(&mut engine, &k5, 15, 20);
        must_cleanup_with_gc_fence(&mut engine, &k5, 14, 0, 20, false);
        must_cleanup_with_gc_fence(&mut engine, &k5, 20, 0, 30, true);

        // PUT, LOCK, LOCK,   Read
        //  `------------------------^
        let k6 = api_v2_key(b"k6");
        must_prewrite_put(&mut engine, &k6, b"v6", &k6, 16);
        must_commit(&mut engine, &k6, 16, 20);
        must_prewrite_lock(&mut engine, &k6, &k6, 25);
        must_commit(&mut engine, &k6, 25, 26);
        must_prewrite_lock(&mut engine, &k6, &k6, 28);
        must_commit(&mut engine, &k6, 28, 29);
        must_cleanup_with_gc_fence(&mut engine, &k6, 20, 0, 50, true);

        // PUT, LOCK,   LOCK,   Read
        //  `---------^
        let k7 = api_v2_key(b"k7");
        must_prewrite_put(&mut engine, &k7, b"v7", &k7, 16);
        must_commit(&mut engine, &k7, 16, 20);
        must_prewrite_lock(&mut engine, &k7, &k7, 25);
        must_commit(&mut engine, &k7, 25, 26);
        // gc_fence should >= commit_ts + 2
        must_cleanup_with_gc_fence(&mut engine, &k7, 20, 0, 28, true);
        must_prewrite_lock(&mut engine, &k7, &k7, 29);
        must_commit(&mut engine, &k7, 29, 30);

        // PUT,  Read
        //  * (GC fence ts is 0)
        let k8 = api_v2_key(b"k8");
        must_prewrite_put(&mut engine, &k8, b"v8", &k8, 17);
        must_commit(&mut engine, &k8, 17, 30);
        must_cleanup_with_gc_fence(&mut engine, &k8, 30, 0, 0, true);

        // PUT, LOCK,     Read
        // `-----------^
        let k9 = api_v2_key(b"k9");
        must_prewrite_put(&mut engine, &k9, b"v9", &k9, 18);
        must_commit(&mut engine, &k9, 18, 20);
        must_prewrite_lock(&mut engine, &k9, &k9, 25);
        must_commit(&mut engine, &k9, 25, 26);
        must_cleanup_with_gc_fence(&mut engine, &k9, 20, 0, 28, true);

        let expected_results = vec![
            (b"k1", Some(b"v1")),
            (b"k2", None),
            (b"k3", None),
            (b"k4", None),
            (b"k5", None),
            (b"k6", Some(b"v6")),
            (b"k7", None),
            (b"k8", Some(b"v8")),
            (b"k9", None),
        ];

        for (k, v) in expected_results {
            let enc_key = api_v2_key(k);
            must_get_eq(
                &kv_engine,
                &Key::from_raw(&enc_key),
                40,
                v.map(|v| v.to_vec()),
            );
        }
    }

    #[test]
    fn test_old_value_capacity_not_exceed_quota() {
        let mut cache = OldValueCache::new(ReadableSize(1000));
        fn short_val() -> OldValue {
            OldValue::Value {
                value: b"s".to_vec(),
            }
        }
        fn long_val() -> OldValue {
            OldValue::Value {
                value: vec![b'l'; 1024],
            }
        }
        fn enc(i: i32) -> Key {
            Key::from_encoded(i32::to_ne_bytes(i).to_vec())
        }

        for i in 0..100 {
            cache.insert(enc(i), (short_val(), None));
        }
        for i in 100..200 {
            // access the previous key for making it not be evicted
            cache.cache.get(&enc(i - 1));
            cache.insert(enc(i), (long_val(), None));
        }
        assert!(
            cache.cache.size() <= 1000,
            "but it is {}",
            cache.cache.size()
        );
    }
}
