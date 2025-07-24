// Copyright 2025 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    fmt::{Debug, Formatter},
    hash::Hash,
    sync::{atomic::AtomicBool, Arc},
};

use bytes::Bytes;
use quick_cache::{sync::GuardResult, Equivalent};

use crate::{
    metrics::{ENGINE_VALUE_CACHE_CACHE_FILL, ENGINE_VALUE_CACHE_CACHE_HIT},
    table::{InnerKey, OwnedInnerKey},
    SnapAccess,
};

// ValueCache caches the values in the engine.
// When the reader performs a get operation, it first checks the cache, if the
// cache exists and valid, it uses the cached value directly.
#[derive(Clone)]
pub struct ValueCache {
    inner: Arc<quick_cache::sync::Cache<ValueCacheKey, ValueCacheValue, Weighter>>,
    validation_tasks: Arc<papaya::HashMap<u64, Arc<crossbeam_queue::SegQueue<ValidationTask>>>>,
}

impl Debug for ValueCache {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "ValueCache")
    }
}

#[derive(Copy, Clone)]
struct Weighter;

impl quick_cache::Weighter<ValueCacheKey, ValueCacheValue> for Weighter {
    fn weight(&self, key: &ValueCacheKey, value: &ValueCacheValue) -> u64 {
        // The weight is the size of the key and value.
        key.inner_key.len() as u64 + value.value.len() as u64 + 64 // 64 bytes for metadata
    }
}

impl ValueCache {
    pub fn new(capacity: u64) -> Self {
        let estimated_items_capacity = (capacity / 128) as usize;
        let inner = Arc::new(quick_cache::sync::Cache::with_weighter(
            estimated_items_capacity,
            capacity,
            Weighter,
        ));
        ValueCache {
            inner,
            validation_tasks: Arc::new(papaya::HashMap::new()),
        }
    }

    pub fn get(
        &self,
        snap_access: &SnapAccess,
        outer_key: &[u8],
        start_ts: u64,
    ) -> Option<ValueCacheValue> {
        let keyspace_id = snap_access.get_keyspace_id();
        let inner_key = InnerKey::from_outer_key(outer_key);
        let key_ref = ValueCacheKeyRef {
            shard_ver: snap_access.get_version() as u32,
            keyspace_id,
            inner_key,
        };
        let val = self.inner.get(&key_ref)?;
        if val.version > start_ts {
            return None;
        }
        if !val.is_valid(
            snap_access.get_write_sequence(),
            snap_access.get_cache_invalidate_sequence(),
        ) {
            return None;
        }
        ENGINE_VALUE_CACHE_CACHE_HIT.inc();
        Some(val)
    }

    // set is called by the reader after get the row from the Shard.
    pub fn set(&self, snap_access: &SnapAccess, outer_key: &[u8], value: ValueCacheValue) {
        let inner_key = InnerKey::from_outer_key(outer_key);
        let cache_key = ValueCacheKey {
            shard_ver: snap_access.get_version() as u32,
            keyspace_id: snap_access.get_keyspace_id(),
            inner_key: inner_key.into(),
        };
        let write_seq = value.shard_write_seq;
        let shard_id = snap_access.get_id();
        match self.inner.get_value_or_guard(&cache_key, None) {
            GuardResult::Value(old_val) => {
                if old_val.version < value.version {
                    let val_state = value.validated.clone();
                    self.inner.insert(cache_key.clone(), value);
                    self.add_validation_task(snap_access.get_id(), cache_key, write_seq, val_state);
                }
            }
            GuardResult::Guard(g) => {
                let val_state = value.validated.clone();
                if g.insert(value).is_ok() {
                    self.add_validation_task(shard_id, cache_key, write_seq, val_state);
                }
            }
            GuardResult::Timeout => {}
        }
    }

    fn add_validation_task(
        &self,
        shard_id: u64,
        key: ValueCacheKey,
        write_seq: u64,
        validated: Arc<AtomicBool>,
    ) {
        ENGINE_VALUE_CACHE_CACHE_FILL.inc();
        let validation_task = ValidationTask {
            key,
            write_seq,
            validated,
        };
        let validates = self.validation_tasks.pin();
        let queue =
            validates.get_or_insert_with(shard_id, || Arc::new(crossbeam_queue::SegQueue::new()));
        queue.push(validation_task);
    }

    pub(crate) fn remove<Q>(&self, key: &Q) -> bool
    where
        Q: Hash + Debug + Equivalent<ValueCacheKey> + ?Sized,
    {
        self.inner.remove(key).is_some()
    }

    pub(crate) fn fetch_validate_keys(&self, shard_id: u64) -> Option<Vec<ValidationTask>> {
        let validates = self.validation_tasks.pin();
        let queue = validates.get(&shard_id)?;
        let length = queue.len();
        if length == 0 {
            return None;
        }
        Some((0..length).filter_map(|_| queue.pop()).collect::<Vec<_>>())
    }
}

#[derive(Eq, PartialEq, Clone, Hash)]
pub struct ValueCacheKey {
    pub shard_ver: u32,
    pub keyspace_id: u32,
    pub inner_key: OwnedInnerKey,
}

impl Debug for ValueCacheKey {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "ValueCacheKey(shard_ver: {}, keyspace_id: {}, inner_key: {:?})",
            self.shard_ver, self.keyspace_id, self.inner_key,
        )
    }
}

#[derive(Eq, PartialEq, Clone, Hash)]
pub struct ValueCacheKeyRef<'a> {
    pub shard_ver: u32,
    pub keyspace_id: u32,
    pub inner_key: InnerKey<'a>,
}

impl Equivalent<ValueCacheKey> for ValueCacheKeyRef<'_> {
    fn equivalent(&self, key: &ValueCacheKey) -> bool {
        key.shard_ver == self.shard_ver
            && key.keyspace_id == self.keyspace_id
            && key.inner_key.as_ref() == self.inner_key
    }
}

impl Debug for ValueCacheKeyRef<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "ValueCacheKeyRef(shard_ver: {}, keyspace_id: {}, inner_key: {:?})",
            self.shard_ver, self.keyspace_id, self.inner_key,
        )
    }
}

#[derive(Clone)]
pub struct ValueCacheValue {
    pub value: Bytes,
    pub start_ts: u64,
    pub version: u64,
    pub shard_write_seq: u64,
    pub validated: Arc<AtomicBool>,
}

impl Debug for ValueCacheValue {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "ValueCacheValue(val_len: {}, start_ts: {}, version: {}, shard_write_seq: {})",
            self.value.len(),
            self.start_ts,
            self.version,
            self.shard_write_seq
        )
    }
}

impl ValueCacheValue {
    pub fn new(value: Bytes, start_ts: u64, version: u64, shard_write_seq: u64) -> Self {
        let validated = Arc::new(AtomicBool::new(false));
        ValueCacheValue {
            value,
            start_ts,
            version,
            shard_write_seq,
            validated,
        }
    }

    pub fn is_valid(&self, shard_seq: u64, cache_invalidate_seq: u64) -> bool {
        if self.shard_write_seq == shard_seq {
            return true;
        }
        if self.shard_write_seq < cache_invalidate_seq {
            return false;
        }
        self.validated.load(std::sync::atomic::Ordering::Relaxed)
    }
}

pub(crate) struct ValidationTask {
    pub(crate) key: ValueCacheKey,
    pub(crate) write_seq: u64,
    pub(crate) validated: Arc<AtomicBool>,
}
