// Copyright 2022 TiKV Project Authors. Licensed under Apache-2.0.

#![feature(test)]
#![feature(box_patterns)]
#![feature(custom_test_frameworks)]
#![test_runner(test_util::run_tests)]

use std::sync::atomic::AtomicU16;

use tikv_util::info;

mod backup;
mod delete_range;
mod engine_basic;
mod gc;
mod load_data;
mod merge;
mod native_backup;
mod replica_read;
mod truncate_ts;

static NODE_ALLOCATOR: AtomicU16 = AtomicU16::new(1);

pub(crate) fn alloc_node_id() -> u16 {
    let node_id = NODE_ALLOCATOR.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    info!("allocated node_id {}", node_id);
    node_id
}

pub(crate) fn alloc_node_id_vec(count: usize) -> Vec<u16> {
    let mut nodes = vec![];
    nodes.resize_with(count, || alloc_node_id());
    nodes
}

pub(crate) fn get_keyspace_prefix(keyspace_id: u32) -> Vec<u8> {
    let mut prefix = keyspace_id.to_be_bytes();
    prefix[0] = b'x';
    prefix.to_vec()
}

pub(crate) fn generate_keyspace_key(keyspace_id: u32) -> impl Fn(usize) -> Vec<u8> {
    move |i: usize| -> Vec<u8> {
        let mut key = get_keyspace_prefix(keyspace_id);
        key.extend(i_to_key(i));
        key
    }
}

pub(crate) fn is_region_belongs_to_keyspace(
    region: &kvproto::metapb::Region,
    keyspace_id: u32,
) -> bool {
    let keyspace_prefix = get_keyspace_prefix(keyspace_id);
    let keypsace_next_prefix = get_keyspace_prefix(keyspace_id + 1);
    let start_key = region.get_start_key();
    let end_key = region.get_end_key();
    if start_key.is_empty() || end_key.is_empty() {
        return false;
    }
    start_key.starts_with(&keyspace_prefix)
        && (end_key.starts_with(&keyspace_prefix) || end_key == keypsace_next_prefix.as_slice())
}

pub(crate) fn i_to_key(i: usize) -> Vec<u8> {
    format!("xkey_{:03}", i).into_bytes()
}

pub(crate) fn i_to_val(i: usize) -> Vec<u8> {
    format!("val_{:03}", i).into_bytes().repeat(3)
}
