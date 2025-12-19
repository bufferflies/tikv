// Copyright 2019 TiKV Project Authors. Licensed under Apache-2.0.

//! Iteration over engines and snapshots.
//!
//! For the purpose of key/value iteration, TiKV defines its own `Iterator`
//! trait, and `Iterable` types that can create iterators.
//!
//! Both `KvEngine`s and `Snapshot`s are `Iterable`.
//!
//! Iteration is performed over consistent views into the database, even when
//! iterating over the engine without creating a `Snapshot`. That is, iterating
//! over an engine behaves implicitly as if a snapshot was created first, and
//! the iteration is being performed on the snapshot.
//!
//! Iterators can be in an _invalid_ state, in which they are not positioned at
//! a key/value pair. This can occur when attempting to move before the first
//! pair, past the last pair, or when seeking to a key that does not exist.
//! There may be other conditions that invalidate iterators (TODO: I don't
//! know).
//!
//! An invalid iterator cannot move forward or back, but may be returned to a
//! valid state through a successful "seek" operation.
//!
//! As TiKV inherits its iteration semantics from RocksDB,
//! the RocksDB documentation is the ultimate reference:
//!
//! - [RocksDB iterator API](https://github.com/facebook/rocksdb/blob/master/include/rocksdb/iterator.h).
//! - [RocksDB wiki on iterators](https://github.com/facebook/rocksdb/wiki/Iterator)

use tikv_util::keybuilder::KeyBuilder;

use crate::*;

/// Build an `IterOptions` using giving data key bound. Empty upper bound will
/// be ignored.
pub fn iter_option(lower_bound: &[u8], upper_bound: &[u8], fill_cache: bool) -> IterOptions {
    let lower_bound = Some(KeyBuilder::from_slice(lower_bound, 0, 0));
    let upper_bound = if upper_bound.is_empty() {
        None
    } else {
        Some(KeyBuilder::from_slice(upper_bound, 0, 0))
    };
    IterOptions::new(lower_bound, upper_bound, fill_cache)
}
