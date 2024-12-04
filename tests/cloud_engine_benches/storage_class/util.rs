// Copyright 2024 TiKV Project Authors. Licensed under Apache-2.0.

use std::sync::atomic::AtomicU16;

use kvengine::SnapAccess;
use rand::{thread_rng, Rng as _};
use tikv::storage::txn::CloudStoreScanner;
use tikv_util::info;
use txn_types::{Key, TsSet};

static NODE_ALLOCATOR: AtomicU16 = AtomicU16::new(1);

pub(crate) fn alloc_node_id() -> u16 {
    let node_id = NODE_ALLOCATOR.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    info!("allocated node_id {}", node_id);
    node_id
}

pub(crate) const KEY_PREFIX: &str = "x123";

pub(crate) fn i_to_key(i: usize) -> Vec<u8> {
    format!("{KEY_PREFIX}key_{:08}", i).into_bytes()
}

#[allow(dead_code)]
pub(crate) fn i_to_val(i: usize) -> Vec<u8> {
    format!("val_{:05}", i).into_bytes().repeat(100)
}

/// `value_size` should be factor of 8.
pub(crate) fn random_value(value_size: usize) -> Vec<u8> {
    let mut rng = thread_rng();
    let batch = value_size >> 3;
    (0..batch)
        .flat_map(|_| rng.gen::<u64>().to_le_bytes())
        .collect()
}

pub(crate) fn new_scanner(
    snap: &SnapAccess,
    desc: bool,
    read_ts: u64,
    lower_bound: Option<Key>,
    upper_bound: Option<Key>,
) -> CloudStoreScanner {
    CloudStoreScanner::new(
        snap.clone(),
        desc,
        true,
        TsSet::Empty,
        read_ts,
        lower_bound,
        upper_bound,
        false,
    )
    .unwrap()
}
