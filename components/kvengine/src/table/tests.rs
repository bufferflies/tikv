// Copyright 2021 TiKV Project Authors. Licensed under Apache-2.0.

use std::{cmp::Ordering::*, iter::Iterator as _, mem, ops::Deref};

use bytes::{Buf, Bytes};
use proptest::prelude::*;
use rand::Rng;

use super::table::*;

#[derive(Debug)]
struct SimpleIterator {
    keys: Vec<Bytes>,
    vals: Vec<Vec<u8>>,
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
        let mut latest_off = vec![];
        for i in 0..length {
            ks.push(Bytes::from(keys[i]));
            let val = Value::encode_buf(0, &[], version, vals[i].as_bytes());
            vs.push(val);
            latest_off.push(i);
        }
        Self {
            keys: ks,
            vals: vs,
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

        let mut rng = rand::thread_rng();
        for i in 0..100 {
            last_offs.push(keys.len());
            let key = Bytes::from(format!("key{:03}", i));
            for j in (min_ver..max_ver).rev() {
                keys.push(key.clone());
                let val = Value::encode_buf(0, &[], j, key.chunk());
                vals.push(val);
                if rng.gen_range(0..4) == 0 {
                    break;
                }
            }
        }
        Self {
            keys,
            vals,
            idx: 0,
            reversed,
            latest_offsets: last_offs,
            ver_idx: 0,
        }
    }

    fn entry_idx(&self) -> usize {
        self.latest_offsets[self.idx as usize] + self.ver_idx
    }
}

impl Iterator for SimpleIterator {
    fn next(&mut self) {
        if !self.reversed {
            self.idx += 1;
        } else {
            self.idx -= 1;
        }
        self.ver_idx = 0;
    }

    fn next_version(&mut self) -> bool {
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

    fn rewind(&mut self) {
        if !self.reversed {
            self.idx = 0;
            self.ver_idx = 0;
        } else {
            self.idx = self.latest_offsets.len() as i32 - 1;
            self.ver_idx = 0;
        }
    }

    fn seek(&mut self, key: InnerKey<'_>) {
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

fn get_all(mut it: Box<dyn Iterator>) -> (Vec<Bytes>, Vec<Bytes>) {
    let mut keys = vec![];
    let mut vals = vec![];
    while it.valid() {
        let key_b = Bytes::copy_from_slice(it.key().deref());
        keys.push(key_b);
        vals.push(Bytes::copy_from_slice(it.value().get_value()));
        it.next();
    }
    (keys, vals)
}

#[test]
fn test_simple_iterator() {
    let keys = vec!["1", "2", "3"];
    let vals = vec!["v1", "v2", "v3"];
    let mut it = Box::new(SimpleIterator::new(keys.clone(), vals.clone(), false, 1));
    it.rewind();
    let (n_keys, n_vals) = get_all(it);
    for i in 0..keys.len() {
        assert_eq!(keys[i].as_bytes(), n_keys[i]);
        assert_eq!(vals[i].as_bytes(), n_vals[i]);
    }
}

#[test]
fn test_merge_single() {
    let keys = vec!["1", "2", "3"];
    let vals = vec!["v1", "v2", "v3"];
    let it = SimpleIterator::new(keys.clone(), vals.clone(), false, 1);
    let mut merge_it = new_merge_iterator(vec![Box::new(it)], false);
    merge_it.rewind();
    let (n_keys, n_vals) = get_all(merge_it);
    for i in 0..keys.len() {
        assert_eq!(keys[i].as_bytes(), n_keys[i]);
        assert_eq!(vals[i].as_bytes(), n_vals[i]);
    }
}

#[test]
fn test_merge_single_resversed() {
    let keys = vec!["1", "2", "3"];
    let vals = vec!["v1", "v2", "v3"];
    let it = SimpleIterator::new(keys.clone(), vals.clone(), true, 1);
    let mut merge_it = new_merge_iterator(vec![Box::new(it)], true);
    merge_it.rewind();
    let (n_keys, n_vals) = get_all(merge_it);
    for i in 0..keys.len() {
        let reverse_idx = keys.len() - 1 - i;
        assert_eq!(keys[reverse_idx].as_bytes(), n_keys[i]);
        assert_eq!(vals[reverse_idx].as_bytes(), n_vals[i]);
    }
}

#[test]
fn test_merge_more() {
    let it1 = Box::new(SimpleIterator::new(
        vec!["1", "3", "7"],
        vec!["a1", "a3", "a7"],
        false,
        10,
    ));
    let it2 = Box::new(SimpleIterator::new(
        vec!["2", "3", "5"],
        vec!["b2", "b3", "b5"],
        false,
        9,
    ));
    let it3 = Box::new(SimpleIterator::new(vec!["1"], vec!["c1"], false, 8));
    let it4 = Box::new(SimpleIterator::new(
        vec!["1", "7", "9"],
        vec!["d1", "d7", "d9"],
        false,
        7,
    ));
    let mut merge_it = new_merge_iterator(vec![it1, it2, it3, it4], false);
    let expected_keys = ["1", "2", "3", "5", "7", "9"];
    let expected_vals = ["a1", "b2", "a3", "b5", "a7", "d9"];
    merge_it.rewind();
    let (keys, vals) = get_all(merge_it);
    for i in 0..expected_keys.len() {
        assert_eq!(expected_keys[i].as_bytes(), keys[i]);
        assert_eq!(expected_vals[i].as_bytes(), vals[i]);
    }
}

#[test]
fn test_merge_iterator_nested() {
    let keys = vec!["1", "2", "3"];
    let vals = vec!["v1", "v2", "v3"];
    let it = Box::new(SimpleIterator::new(keys.clone(), vals.clone(), false, 1));
    let merge1 = new_merge_iterator(vec![it], false);
    let mut merge2 = new_merge_iterator(vec![merge1], false);
    merge2.rewind();
    let (n_keys, n_vals) = get_all(merge2);
    for i in 0..keys.len() {
        assert_eq!(keys[i].as_bytes(), n_keys[i]);
        assert_eq!(vals[i].as_bytes(), n_vals[i])
    }
}

#[test]
fn test_merge_iterator_seek() {
    let it1 = Box::new(SimpleIterator::new(
        vec!["1", "3", "7"],
        vec!["a1", "a3", "a7"],
        false,
        9,
    ));
    let it2 = Box::new(SimpleIterator::new(
        vec!["2", "3", "5"],
        vec!["b2", "b3", "b5"],
        false,
        8,
    ));
    let it3 = Box::new(SimpleIterator::new(vec!["1"], vec!["c1"], false, 7));
    let it4 = Box::new(SimpleIterator::new(
        vec!["1", "7", "9"],
        vec!["d1", "d7", "d9"],
        false,
        6,
    ));
    let mut merge_it = new_merge_iterator(vec![it1, it2, it3, it4], false);
    merge_it.seek(InnerKey::from_inner_buf("4".as_bytes()));
    let (keys, vals) = get_all(merge_it);
    let expected_keys = ["5", "7", "9"];
    let expected_vals = ["b5", "a7", "d9"];
    for i in 0..expected_keys.len() {
        assert_eq!(expected_keys[i].as_bytes(), keys[i]);
        assert_eq!(expected_vals[i].as_bytes(), vals[i]);
    }
}

#[test]
fn test_merge_iterator_seek_reversed() {
    let it1 = Box::new(SimpleIterator::new(
        vec!["1", "3", "7"],
        vec!["a1", "a3", "a7"],
        true,
        9,
    ));
    let it2 = Box::new(SimpleIterator::new(
        vec!["2", "3", "5"],
        vec!["b2", "b3", "b5"],
        true,
        8,
    ));
    let it3 = Box::new(SimpleIterator::new(vec!["1"], vec!["c1"], true, 7));
    let it4 = Box::new(SimpleIterator::new(
        vec!["1", "7", "9"],
        vec!["d1", "d7", "d9"],
        true,
        6,
    ));
    let mut merge_it = new_merge_iterator(vec![it1, it2, it3, it4], true);
    merge_it.seek(InnerKey::from_inner_buf("5".as_bytes()));
    let (keys, vals) = get_all(merge_it);
    let expected_keys = ["5", "3", "2", "1"];
    let expected_vals = ["b5", "a3", "b2", "a1"];
    for i in 0..expected_keys.len() {
        assert_eq!(expected_keys[i].as_bytes(), keys[i]);
        assert_eq!(expected_vals[i].as_bytes(), vals[i]);
    }
}

#[test]
fn test_merge_iterator_seek_invalid() {
    let it1 = Box::new(SimpleIterator::new(
        vec!["1", "3", "7"],
        vec!["a1", "a3", "a7"],
        false,
        9,
    ));
    let it2 = Box::new(SimpleIterator::new(
        vec!["2", "3", "5"],
        vec!["b2", "b3", "b5"],
        false,
        8,
    ));
    let it3 = Box::new(SimpleIterator::new(vec!["1"], vec!["c1"], false, 7));
    let it4 = Box::new(SimpleIterator::new(
        vec!["1", "7", "9"],
        vec!["d1", "d7", "d9"],
        false,
        6,
    ));
    let mut merge_it = new_merge_iterator(vec![it1, it2, it3, it4], false);
    merge_it.seek(InnerKey::from_inner_buf("f".as_bytes()));
    assert!(!merge_it.valid());
}

#[test]
fn test_merge_iterator_seek_invalid_reversed() {
    let it1 = Box::new(SimpleIterator::new(
        vec!["1", "3", "7"],
        vec!["a1", "a3", "a7"],
        true,
        9,
    ));
    let it2 = Box::new(SimpleIterator::new(
        vec!["2", "3", "5"],
        vec!["b2", "b3", "b5"],
        true,
        8,
    ));
    let it3 = Box::new(SimpleIterator::new(vec!["1"], vec!["c1"], true, 7));
    let it4 = Box::new(SimpleIterator::new(
        vec!["1", "7", "9"],
        vec!["d1", "d7", "d9"],
        true,
        6,
    ));
    let mut merge_it = new_merge_iterator(vec![it1, it2, it3, it4], true);
    merge_it.seek(InnerKey::from_inner_buf("0".as_bytes()));
    assert!(!merge_it.valid());
}

#[test]
fn merge_iterator_duplicated() {
    let it1 = Box::new(SimpleIterator::new(
        vec!["0", "1", "2"],
        vec!["0", "1", "2"],
        false,
        9,
    ));
    let it2 = Box::new(SimpleIterator::new(vec!["1"], vec!["1"], false, 8));
    let it3 = Box::new(SimpleIterator::new(vec!["2"], vec!["2"], false, 7));
    let mut merge_it = new_merge_iterator(vec![it1, it2, it3], false);
    merge_it.rewind();
    let mut cnt = 0;
    while merge_it.valid() {
        assert_eq!(cnt + 48, merge_it.key()[0] as i32);
        cnt += 1;
        merge_it.next();
    }
    assert_eq!(cnt, 3);
}

#[test]
fn test_multi_version_merge_iterator() {
    let mut rnd = rand::thread_rng();
    for &reversed in &[false, true] {
        let it1 = Box::new(SimpleIterator::new_multi_version(100, 90, reversed));
        let it2 = Box::new(SimpleIterator::new_multi_version(90, 80, reversed));
        let it3 = Box::new(SimpleIterator::new_multi_version(80, 70, reversed));
        let it4 = Box::new(SimpleIterator::new_multi_version(70, 60, reversed));
        let mut it = new_merge_iterator(vec![it1, it2, it3, it4], reversed);
        it.rewind();
        let mut cur_key = Bytes::copy_from_slice(it.key().deref());
        for _ in 1..100 {
            it.next();
            assert_eq!(it.valid(), true);
            assert_ne!(cur_key.chunk(), it.key().deref());
            cur_key = Bytes::copy_from_slice(it.key().deref());
            let cur_ver = it.value().version;
            while it.next_version() {
                assert_eq!(it.value().version < cur_ver, true);
            }
        }
        for _ in 0..100 {
            let key = Bytes::from(format!("key{:03}", rnd.gen_range::<u8, _>(0..100)));
            it.seek(InnerKey::from_inner_buf(&key));
            assert_eq!(it.valid(), true);
            assert_eq!(it.key().deref(), key.chunk());
            let mut cur_ver = it.value().version;
            while it.next_version() {
                assert_eq!(it.value().version < cur_ver, true);
                cur_ver = it.value().version;
            }
            assert_eq!(cur_ver <= 70, true);
        }
    }
}

prop_compose! {
    fn arb_keys(max_iters_count: usize, max_keys_count: usize, max_versions_count: usize)
    (iters_count in 2..=max_iters_count, versions_count in 1..=max_versions_count)
    (
        iters_count in Just(iters_count),
        keys in prop::collection::vec(0..max_keys_count, versions_count),
        keys_pos in prop::collection::vec(0..iters_count, versions_count),
    ) -> (usize, Vec<usize>, Vec<usize>) {
        (iters_count, keys, keys_pos)
    }
}
proptest! {
    #[test]
    fn test_multi_version_merge_iterator_rnd((iters_count, keys, keys_pos) in arb_keys(2, 5, 10), reversed in any::<bool>()) {
        // println!("iters_count: {}, keys: {:?}, keys_pos: {:?}, reversed: {}", iters_count, keys, keys_pos, reversed);

        let i_to_key = |i| format!("{i:04}");
        let i_to_val = |i, ver| format!("{i:04}-{ver:04}");

        let versions_count = keys.len();

        let mut iters_data = Vec::with_capacity(iters_count);
        let mut iters_last_offs = Vec::with_capacity(iters_count);
        let mut iters_keys = Vec::with_capacity(iters_count);
        let mut iters_vals = Vec::with_capacity(iters_count);
        for _ in 0..iters_count {
            iters_data.push(vec![]);
            iters_last_offs.push(vec![]);
            iters_keys.push(vec![]);
            iters_vals.push(vec![]);
        }

        for (idx, (k, pos)) in keys.into_iter().zip(keys_pos.into_iter()).enumerate() {
            let ver = versions_count - idx;
            let key = i_to_key(k);
            let val = i_to_val(k, ver);
            let value = Value::encode_buf(0, &[], ver as u64, val.as_bytes());
            iters_data[pos].push((Bytes::from(key), value))
        }
        for (pos, mut iter_data) in iters_data.into_iter().enumerate() {
            // Must be stable sort.
            iter_data.sort_by(|a, b| a.0.cmp(&b.0));

            let mut prev_key = None;
            for (key, value) in iter_data {
                let same_key = prev_key.as_ref().is_some_and(|x| x == &key);
                if !same_key {
                    iters_last_offs[pos].push(iters_keys[pos].len());
                    prev_key = Some(key.clone());
                }
                iters_keys[pos].push(key);
                iters_vals[pos].push(value);
            }
        }


        let mut iters: Vec<Box<dyn Iterator>> = Vec::with_capacity(iters_count);
        for _ in 0..iters_count {
            let mut iter = SimpleIterator {
                keys: iters_keys.pop().unwrap(),
                vals: iters_vals.pop().unwrap(),
                idx: 0,
                reversed,
                latest_offsets: iters_last_offs.pop().unwrap(),
                ver_idx: 0,
            };
            println!("iter {:?}", iter);
            iter.rewind();
            while iter.valid() {
                println!("{:?}: {}", iter.key(), iter.value().version);
                while iter.next_version() {
                    println!("ver: {:?}: {}", iter.key(), iter.value().version);
                }
                iter.next();
            }
            iters.push(Box::new(iter));
        }

        let mut merge_it = new_merge_iterator(iters, reversed);
        merge_it.rewind();

        let mut keys = Vec::with_capacity(versions_count);
        let mut vers = Vec::with_capacity(versions_count);
        while merge_it.valid() {
            keys.push(merge_it.key().to_vec());

            let mut cur_ver = merge_it.value().version;
            vers.push(cur_ver);

            while merge_it.next_version() {
                let key = merge_it.key();
                prop_assert_eq!(keys.last().unwrap().as_slice(), key.deref());

                let prev_ver = mem::replace(&mut cur_ver, merge_it.value().version);
                prop_assert!(prev_ver > cur_ver);
                vers.push(cur_ver);
            }
            merge_it.next();
        }

        let keys_is_sorted = keys.is_sorted_by(|a, b| {
            let r = a.cmp(b);
            Some(if reversed {
                r.reverse()
            } else {
                r
            })
        });
        prop_assert!(keys_is_sorted);

        prop_assert_eq!(vers.len(), versions_count);
        vers.sort();
        let expected_vers = (1..=versions_count as u64).collect::<Vec<_>>();
        prop_assert_eq!(vers, expected_vers);
    }
}

#[cfg(test)]
mod tests {

    use std::{mem::size_of, sync::Arc};

    use super::*;
    use crate::table::{
        blobtable::{blobtable::BlobTable, builder::BlobTableBuilder, BlobRef},
        sstable::{
            file::InMemFile,
            sstable::{
                get_test_key, new_table_builder_for_test, new_test_cache, SsTable, TEST_ID_ALLOC,
            },
        },
        Iterator,
    };

    #[cfg(test)]
    pub(crate) fn get_blob_test_value(n: usize) -> String {
        format!("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa - {}", n)
    }

    #[cfg(test)]
    pub(crate) fn generate_key_values(prefix: &str, n: usize) -> Vec<(String, String)> {
        assert!(n <= 10000);
        let mut results = Vec::with_capacity(n);
        for i in 0..n {
            let k = get_test_key(prefix, i);
            let v = get_blob_test_value(i);
            results.push((k, v));
        }
        results
    }

    #[cfg(test)]
    fn test_fetch_value_from_blob_table(bt: &BlobTable, v: Value) -> Result<Vec<u8>> {
        assert!(v.is_blob_ref());
        let blob_ref = v.get_blob_ref();
        assert_eq!(bt.id(), blob_ref.fid);
        bt.get(&blob_ref)
    }

    #[cfg(test)]
    fn test_store_value_in_blob_table(
        blob_builder: &mut BlobTableBuilder,
        k: &String,
        v: &mut Value,
    ) -> BlobRef {
        assert!(v.value_len() > size_of::<BlobRef>() + v.user_meta_len());
        v.set_blob_ref();
        blob_builder.add(InnerKey::from_inner_buf(k.as_bytes()), v)
    }

    #[cfg(test)]
    pub(crate) fn build_blob_test_table_with_kvs(
        kvs: &Vec<(String, String)>,
        load_filter: bool,
    ) -> (SsTable, BlobTable) {
        use crate::table::sstable::NO_COMPRESSION;

        let sst_fid = TEST_ID_ALLOC.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        let blob_fid = TEST_ID_ALLOC.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        let mut sst_builder = new_table_builder_for_test(sst_fid);
        let mut blob_builder = BlobTableBuilder::new(blob_fid, NO_COMPRESSION, 0, 0);
        let meta = 0u8;

        for (k, v) in kvs {
            let value_buf = Value::encode_buf(meta, &[0], 0, v.as_bytes());
            let mut v = Value::decode(value_buf.as_slice());
            let blob_ref = test_store_value_in_blob_table(&mut blob_builder, k, &mut v);
            sst_builder.add(InnerKey::from_inner_buf(k.as_bytes()), &v, Some(blob_ref));
        }

        let mut buf = Vec::with_capacity(sst_builder.estimated_size());

        sst_builder.finish(0, &mut buf);

        let bytes = blob_builder.finish();

        let sst_file = InMemFile::new(sst_fid, buf.into());
        let blob_file = InMemFile::new(blob_fid, bytes);

        (
            SsTable::new(Arc::new(sst_file), new_test_cache(), load_filter, None).unwrap(),
            BlobTable::new(Arc::new(blob_file)).unwrap(),
        )
    }

    #[cfg(test)]
    pub(crate) fn create_blob_sst_table(
        prefix: &str,
        n: usize,
        load_filter: bool,
    ) -> ((SsTable, BlobTable), Vec<(String, String)>) {
        let kvs = generate_key_values(prefix, n);
        (build_blob_test_table_with_kvs(&kvs, load_filter), kvs)
    }

    #[test]
    fn test_value_external_storage() {
        let ((t, bt), _) = create_blob_sst_table("key", 10000, true);
        let mut it = t.new_iterator(false, true);
        let mut count = 0;
        it.rewind();
        while it.valid() {
            let k = it.key();
            assert_eq!(k.deref(), get_test_key("key", count).as_bytes());
            let value = test_fetch_value_from_blob_table(&bt, it.value());
            assert_eq!(value.unwrap(), get_blob_test_value(count).as_bytes());
            count += 1;
            it.next()
        }
    }
}
