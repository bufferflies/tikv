// Copyright 2017 TiKV Project Authors. Licensed under Apache-2.0.

#[allow(unused_extern_crates)]
extern crate tikv_alloc;

use std::{cmp::Eq, hash::Hash};

pub type HashMap<K, V> =
    std::collections::HashMap<K, V, std::hash::BuildHasherDefault<fxhash::FxHasher>>;
pub type HashSet<T> = std::collections::HashSet<T, std::hash::BuildHasherDefault<fxhash::FxHasher>>;
pub use std::collections::hash_map::Entry as HashMapEntry;

pub fn hash_set_with_capacity<T: Hash + Eq>(capacity: usize) -> HashSet<T> {
    HashSet::with_capacity_and_hasher(capacity, fxhash::FxBuildHasher::default())
}

// Modified from https://github.com/tkaitchuck/aHash/blob/4b73276f6b670bed9a7c9b5683eb9680df35183a/src/lib.rs#L153
pub trait HashMapExt {
    fn with_capacity(capacity: usize) -> Self;
}

impl<K, V> HashMapExt for HashMap<K, V>
where
    K: Hash + Eq,
{
    fn with_capacity(capacity: usize) -> Self {
        HashMap::with_capacity_and_hasher(capacity, fxhash::FxBuildHasher::default())
    }
}

pub trait HashSetExt {
    fn with_capacity(capacity: usize) -> Self;
}

impl<T> HashSetExt for HashSet<T>
where
    T: Hash + Eq,
{
    fn with_capacity(capacity: usize) -> Self {
        HashSet::with_capacity_and_hasher(capacity, fxhash::FxBuildHasher::default())
    }
}
