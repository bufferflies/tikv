// Copyright 2021 TiKV Project Authors. Licensed under Apache-2.0.

mod test_blob;
mod test_merge_iter;

use std::{cmp::Ordering::*, iter::Iterator as _, ops::Deref};

use bytes::{Buf, Bytes};
use rand::Rng;

use super::table::*;
use crate::{next, next_async};

#[derive(Debug)]
struct SimpleIterator {
    keys: Vec<Bytes>,
    vals: Vec<Vec<u8>>,
    // Indicate whether the value at the position can be retrieved by sync methods.
    vals_is_sync: Vec<bool>,
    idx: i32,
    reversed: bool,

    latest_offsets: Vec<usize>,
    ver_idx: usize,
}

impl SimpleIterator {
    fn new(keys: Vec<&'static str>, vals: Vec<&'static str>, reversed: bool, version: u64) -> Self {
        let length = keys.len();
        let mut ks: Vec<Bytes> = vec![];
        let mut vs = vec![];
        let mut syncs = vec![];
        let mut latest_off = vec![];
        for i in 0..length {
            ks.push(Bytes::from(keys[i]));
            let val = Value::encode_buf(0, &[], version, vals[i].as_bytes());
            vs.push(val);
            syncs.push(true);
            latest_off.push(i);
        }
        Self {
            keys: ks,
            vals: vs,
            vals_is_sync: syncs,
            idx: 0,
            reversed,
            latest_offsets: latest_off,
            ver_idx: 0,
        }
    }

    fn new_multi_version(max_ver: u64, min_ver: u64, reversed: bool) -> Self {
        let mut last_offs = vec![];
        let mut keys = vec![];
        let mut vals = vec![];
        let mut syncs = vec![];

        let mut rng = rand::thread_rng();
        for i in 0..100 {
            last_offs.push(keys.len());
            let key = Bytes::from(format!("key{:03}", i));
            for j in (min_ver..max_ver).rev() {
                keys.push(key.clone());
                let val = Value::encode_buf(0, &[], j, key.chunk());
                vals.push(val);
                syncs.push(true);
                if rng.gen_range(0..4) == 0 {
                    break;
                }
            }
        }
        Self {
            keys,
            vals,
            vals_is_sync: syncs,
            idx: 0,
            reversed,
            latest_offsets: last_offs,
            ver_idx: 0,
        }
    }

    /// kvs should be sorted by key ascending and version descending.
    fn from_kvs(kvs: Vec<(Bytes, Vec<u8>, bool)>, reversed: bool) -> Self {
        let mut keys = Vec::with_capacity(kvs.len());
        let mut vals = Vec::with_capacity(kvs.len());
        let mut syncs = Vec::with_capacity(kvs.len());
        let mut latest_offsets = Vec::with_capacity(kvs.len());
        let mut prev_key = None;
        for (key, value, is_sync) in kvs {
            let same_key = prev_key.as_ref().is_some_and(|x| x == &key);
            if !same_key {
                latest_offsets.push(keys.len());
                prev_key = Some(key.clone());
            }
            keys.push(key);
            vals.push(value);
            syncs.push(is_sync);
        }

        Self {
            keys,
            vals,
            vals_is_sync: syncs,
            idx: 0,
            reversed,
            latest_offsets,
            ver_idx: 0,
        }
    }

    fn entry_idx(&self) -> usize {
        self.entry_idx_inner(self.idx, self.ver_idx).unwrap()
    }

    fn entry_idx_inner(&self, idx: i32, ver_idx: usize) -> Option<usize> {
        if idx < 0 || idx >= self.latest_offsets.len() as i32 {
            return None;
        }

        let entry_idx = self.latest_offsets[idx as usize] + ver_idx;

        let next_entry_off = if idx + 1 < self.latest_offsets.len() as i32 {
            self.latest_offsets[idx as usize + 1]
        } else {
            self.keys.len()
        };

        (entry_idx < next_entry_off).then_some(entry_idx)
    }

    fn next_inner(&mut self) {
        if !self.reversed {
            self.idx += 1;
        } else {
            self.idx -= 1;
        }
        self.ver_idx = 0;
    }

    fn next_version_inner(&mut self) -> bool {
        let mut next_entry_off = self.keys.len();
        if self.idx + 1 < self.latest_offsets.len() as i32 {
            next_entry_off = self.latest_offsets[self.idx as usize + 1];
        }
        if self.entry_idx() + 1 < next_entry_off {
            self.ver_idx += 1;
            return true;
        }
        false
    }

    fn value_is_sync(&self) -> bool {
        self.vals_is_sync[self.entry_idx()]
    }
}

#[maybe_async::async_trait]
impl Iterator for SimpleIterator {
    fn next(&mut self) {
        self.next_inner();

        if self.valid() {
            assert!(self.value_is_sync(), "next: value is async: {:?}", self);
        }
    }

    async fn next_async(&mut self) {
        self.next_inner();

        if self.valid() {
            assert!(
                !self.value_is_sync(),
                "next_async: value is sync: {:?}",
                self
            );
        }
    }

    fn is_next_sync(&mut self) -> bool {
        let idx = if !self.reversed {
            self.idx + 1
        } else {
            self.idx - 1
        };
        // `is_next_sync` is usually implemented by check the boundary of blocks.
        // So the `idx` out of bound is considered as not sync.
        // This also ensure the correctness of `ConcatIterator`.
        self.entry_idx_inner(idx, 0)
            .map_or(false, |entry_idx| self.vals_is_sync[entry_idx])
    }

    fn next_version(&mut self) -> bool {
        let ok = self.next_version_inner();

        if ok {
            assert!(
                self.value_is_sync(),
                "next_version: value is async: {:?}",
                self
            );
        }
        ok
    }

    async fn next_version_async(&mut self) -> bool {
        let ok = self.next_version_inner();

        if ok {
            assert!(
                !self.value_is_sync(),
                "next_version_async: value is sync: {:?}",
                self
            );
        }
        ok
    }

    fn is_next_version_sync(&mut self) -> bool {
        // Versions of the same key are usually in the same block (old block).
        // So the `ver_idx` out of bound is considered as sync.
        self.entry_idx_inner(self.idx, self.ver_idx + 1)
            .map_or(true, |entry_idx| self.vals_is_sync[entry_idx])
    }

    #[maybe_async]
    async fn rewind(&mut self) {
        if !self.reversed {
            self.idx = 0;
            self.ver_idx = 0;
        } else {
            self.idx = self.latest_offsets.len() as i32 - 1;
            self.ver_idx = 0;
        }
    }

    #[maybe_async]
    async fn seek(&mut self, key: InnerKey<'_>) {
        self.idx = search(self.latest_offsets.len(), |idx| {
            self.keys[self.latest_offsets[idx]].chunk().cmp(key.deref()) != Less
        }) as i32;
        self.ver_idx = 0;
        if self.reversed && (!self.valid() || self.key().cmp(&key) != Equal) {
            self.idx -= 1;
        }
    }

    fn key(&self) -> InnerKey<'_> {
        InnerKey::from_inner_buf(self.keys[self.entry_idx()].as_ref())
    }

    fn value(&self) -> Value {
        let buf = self.vals[self.entry_idx()].as_slice();
        Value::decode(buf)
    }

    fn valid(&self) -> bool {
        self.idx >= 0 && self.idx < self.latest_offsets.len() as i32
    }
}

#[maybe_async::both]
async fn get_all(it: &mut dyn Iterator) -> (Vec<Bytes>, Vec<Bytes>) {
    let mut keys = vec![];
    let mut vals = vec![];
    while it.valid() {
        let key_b = Bytes::copy_from_slice(it.key().deref());
        keys.push(key_b);
        vals.push(Bytes::copy_from_slice(it.value().get_value()));
        next!(it).await;
    }
    (keys, vals)
}
#[allow(dead_code)]
#[maybe_async::both]
async fn get_all_versions(it: &mut dyn Iterator) -> (Vec<Bytes>, Vec<Bytes>, Vec<u64>) {
    let mut keys = vec![];
    let mut vals = vec![];
    let mut vers = vec![];
    while it.valid() {
        let key_b = Bytes::copy_from_slice(it.key().deref());
        keys.push(key_b);
        let val = it.value();
        vals.push(Bytes::copy_from_slice(val.get_value()));
        vers.push(val.version);
        it.next_all_version().await;
    }
    (keys, vals, vers)
}

fn i_to_key(i: usize) -> String {
    format!("{i:04}")
}

fn i_to_val(i: usize, ver: u64) -> String {
    format!("{i:04}-{ver:04}")
}

#[test]
fn test_simple_iterator() {
    let keys = vec!["1", "2", "3"];
    let vals = vec!["v1", "v2", "v3"];
    let mut it = Box::new(SimpleIterator::new(keys.clone(), vals.clone(), false, 1));
    it.rewind();
    let (n_keys, n_vals) = get_all(it.as_mut());
    for i in 0..keys.len() {
        assert_eq!(keys[i].as_bytes(), n_keys[i]);
        assert_eq!(vals[i].as_bytes(), n_vals[i]);
    }
}

#[tokio::test]
async fn test_simple_iterator_async() {
    let cases = [(1, 100), (2, 200), (2, 100), (3, 100)];
    let kvs: Vec<_> = cases
        .iter()
        .map(|&(i, ver)| {
            let val = Value::encode_buf(0, &[], ver, i_to_val(i, ver).as_bytes());
            (
                Bytes::from(i_to_key(i)),
                val,
                false, // is_sync
            )
        })
        .collect();

    {
        let mut it = Box::new(SimpleIterator::from_kvs(kvs.clone(), false));

        it.rewind_async().await;
        let (n_keys, n_vals) = get_all_async(it.as_mut()).await;
        assert_eq!(
            n_keys,
            [1, 2, 3]
                .map(|i| Bytes::from(i_to_key(i)))
                .iter()
                .collect::<Vec<_>>()
        );
        assert_eq!(
            n_vals,
            [(1, 100), (2, 200), (3, 100)]
                .map(|(i, ver)| Bytes::from(i_to_val(i, ver)))
                .iter()
                .collect::<Vec<_>>()
        );

        it.rewind_async().await;
        let (n_keys, n_vals, n_vers) = get_all_versions_async(it.as_mut()).await;
        assert_eq!(
            n_keys,
            [1, 2, 2, 3]
                .map(|i| Bytes::from(i_to_key(i)))
                .iter()
                .collect::<Vec<_>>()
        );
        assert_eq!(
            n_vals,
            [(1, 100), (2, 200), (2, 100), (3, 100)]
                .map(|(i, ver)| Bytes::from(i_to_val(i, ver)))
                .iter()
                .collect::<Vec<_>>()
        );
        assert_eq!(n_vers, vec![100, 200, 100, 100]);
    }

    // Reversed:
    {
        let mut it = Box::new(SimpleIterator::from_kvs(kvs, true));

        it.rewind_async().await;
        let (n_keys, n_vals) = get_all_async(it.as_mut()).await;
        assert_eq!(
            n_keys,
            [3, 2, 1]
                .map(|i| Bytes::from(i_to_key(i)))
                .iter()
                .collect::<Vec<_>>()
        );
        assert_eq!(
            n_vals,
            [(3, 100), (2, 200), (1, 100)]
                .map(|(i, ver)| Bytes::from(i_to_val(i, ver)))
                .iter()
                .collect::<Vec<_>>()
        );

        it.rewind_async().await;
        let (n_keys, n_vals, _) = get_all_versions_async(it.as_mut()).await;
        assert_eq!(
            n_keys,
            [3, 2, 2, 1]
                .map(|i| Bytes::from(i_to_key(i)))
                .iter()
                .collect::<Vec<_>>()
        );
        assert_eq!(
            n_vals,
            [(3, 100), (2, 200), (2, 100), (1, 100)]
                .map(|(i, ver)| Bytes::from(i_to_val(i, ver)))
                .iter()
                .collect::<Vec<_>>()
        );
    }
}
