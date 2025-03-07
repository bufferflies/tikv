// Copyright 2022 TiKV Project Authors. Licensed under Apache-2.0.

use rand::Rng;

mod test_compaction;
mod test_merge;
mod test_trim_over_bound;

pub use test_cloud_server::{alloc_node_id, alloc_node_id_vec};

fn i_to_key(i: usize) -> Vec<u8> {
    format!("xkey{:08}", i).into_bytes()
}

fn i_to_val(i: usize) -> Vec<u8> {
    format!("val{:08}", i).into_bytes().repeat(10)
}

fn random_value_1kb(_: usize) -> Vec<u8> {
    let mut bytes = [0u8; 1024];
    rand::thread_rng().fill(&mut bytes);
    bytes.to_vec()
}
