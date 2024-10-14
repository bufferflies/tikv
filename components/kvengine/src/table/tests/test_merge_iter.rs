// Copyright 2024 TiKV Project Authors. Licensed under Apache-2.0.

use std::{iter::Iterator as _, mem, ops::Deref};

use bytes::{Buf, Bytes};
use proptest::prelude::*;
use rand::Rng;

use super::{get_all, SimpleIterator};
use crate::table::*;

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
            let iter = SimpleIterator {
                keys: iters_keys.pop().unwrap(),
                vals: iters_vals.pop().unwrap(),
                idx: 0,
                reversed,
                latest_offsets: iters_last_offs.pop().unwrap(),
                ver_idx: 0,
            };
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
