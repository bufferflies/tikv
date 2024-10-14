// Copyright 2021 TiKV Project Authors. Licensed under Apache-2.0.

mod test_blob;
mod test_merge_iter;

use std::{cmp::Ordering::*, iter::Iterator as _, ops::Deref};

use bytes::{Buf, Bytes};
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
