// Copyright 2022 TiKV Project Authors. Licensed under Apache-2.0.

use api_version::ApiV2;
use rand::Rng;
use security::SecurityConfig;

mod gc;
mod test_async_fetch;
mod test_async_io;
mod test_compaction;
mod test_disk_usage;
mod test_load_data;
mod test_merge;
mod test_native_br;
mod test_rfengine;
mod test_rfstore_online_config;
mod test_split_region;
mod test_transaction;
mod test_trim_over_bound;

mod test_local_file_gc;

pub use test_cloud_server::{alloc_node_id, alloc_node_id_vec};
use tidb_query_datatype::codec::table;

fn i_to_key(i: usize) -> Vec<u8> {
    format!("xkey{:08}", i).into_bytes()
}

fn i_to_keyspace_key(keyspace_id: u32) -> impl Fn(usize) -> Vec<u8> {
    move |i: usize| -> Vec<u8> {
        let mut key = ApiV2::get_keyspace_prefix_by_id(keyspace_id);
        key.extend(format!("key{:08}", i).into_bytes());
        key
    }
}

fn i_to_val(i: usize) -> Vec<u8> {
    format!("val{:08}", i).into_bytes().repeat(10)
}

fn random_value<const N: usize>(_: usize) -> Vec<u8>
where
    [u8; N]: rand::Fill,
{
    let mut bytes = [0u8; N];
    rand::thread_rng().fill(&mut bytes);
    bytes.to_vec()
}

fn random_value_1kb(i: usize) -> Vec<u8> {
    random_value::<1024>(i)
}

fn i_to_row_key(keyspace_prefix: &[u8], table_id: i64, i: usize) -> Vec<u8> {
    let mut key = keyspace_prefix.to_vec();
    key.extend(table::encode_row_key(table_id, i as i64));
    key
}

fn new_security_config() -> SecurityConfig {
    let mut conf = SecurityConfig::default();
    conf.master_key.vendor = "test".to_string();
    conf.master_key.key_id = "random".to_string();
    conf
}
