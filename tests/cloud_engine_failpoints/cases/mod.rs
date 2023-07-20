// Copyright 2022 TiKV Project Authors. Licensed under Apache-2.0.

use std::sync::atomic::AtomicU16;

use tikv_util::info;

mod test_compaction;
mod test_merge;
mod test_split;
mod test_trim_over_bound;

fn i_to_key(i: usize) -> Vec<u8> {
    format!("xkey{:08}", i).into_bytes()
}

fn i_to_val(i: usize) -> Vec<u8> {
    format!("val{:08}", i).into_bytes().repeat(10)
}

// Start from 400 to work around https://github.com/tidbcloud/cloud-storage-engine/issues/658.
static NODE_ALLOCATOR: AtomicU16 = AtomicU16::new(500);

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
