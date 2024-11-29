// Copyright 2022 TiKV Project Authors. Licensed under Apache-2.0.

#![feature(test)]
#![feature(box_patterns)]
#![feature(custom_test_frameworks)]
#![feature(assert_matches)]
#![test_runner(test_util::run_tests)]

use std::{str::FromStr, sync::atomic::AtomicU16};

use api_version::ApiV2;
use bytes::Bytes;
use http::Uri;
use hyper::{Body, Request};
use kvproto::{kvrpcpb::UnsafeDestroyRangeRequest, metapb::Store};
use security::SecurityConfig;
use test_cloud_server::client::ClusterClient;
use tidb_query_common::util::convert_to_prefix_next;
use tikv_util::info;
mod backup;
mod columnar;
mod delete_range;
mod engine_basic;
mod gc;
mod ia_file;
mod load_data;
mod major_compaction;
mod merge;
mod native_backup;
mod replica_read;
mod transaction;
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
        key.extend(i_to_tidb_key(i));
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
    format!("xkey_{:08}", i).into_bytes()
}

pub(crate) fn i_to_tidb_key(i: usize) -> Vec<u8> {
    format!("t_key_{:08}", i).into_bytes()
}

pub(crate) fn i_to_val(i: usize) -> Vec<u8> {
    format!("val_{:03}", i).into_bytes().repeat(3)
}

pub(crate) fn i_to_val_opt(prefix: &str, repeat: usize) -> impl Fn(usize) -> Vec<u8> + '_ {
    move |i: usize| -> Vec<u8> { format!("{}{:03}", prefix, i).into_bytes().repeat(repeat) }
}

/// Generate keys of API v1 (TiDB metas)
pub(crate) fn i_to_key_v1(i: usize) -> Vec<u8> {
    format!("m_{:08}", i).into_bytes()
}

pub(crate) async fn request_major_compact_on_store(
    store: &Store,
    query: &str,
    permit_not_found: bool,
) {
    let uri = Uri::from_str(&format!(
        "http://{}/major-compact?{}",
        &store.status_address, query
    ))
    .unwrap();
    let req = Request::post(uri).body(Body::empty()).unwrap();
    let client = hyper::Client::new();
    let resp: http::Response<Body> = client.request(req).await.unwrap();
    let is_success = resp.status().is_success()
        || (permit_not_found && resp.status() == http::StatusCode::NOT_FOUND);
    assert!(
        is_success,
        "{:?}",
        hyper::body::to_bytes(resp.into_body()).await.unwrap()
    );
    hyper::body::to_bytes(resp.into_body()).await.unwrap();
}

pub(crate) fn i_to_key_with_keyspace(keyspace_id: u32) -> impl Fn(usize) -> Vec<u8> {
    move |i: usize| -> Vec<u8> {
        let mut key = ApiV2::get_txn_keyspace_prefix(keyspace_id);
        key.extend(format!("tkey{:08}", i).into_bytes());
        key
    }
}

pub(crate) fn i_to_val_with_size(size: usize) -> impl Fn(usize) -> Vec<u8> {
    move |i: usize| -> Vec<u8> { format!("{:0size$}", i, size = size).into_bytes() }
}

pub(crate) fn new_destroy_range_req(prefix: &[u8]) -> UnsafeDestroyRangeRequest {
    let mut req = UnsafeDestroyRangeRequest::default();
    let mut end_key = prefix.to_vec();
    convert_to_prefix_next(&mut end_key);
    req.set_start_key(prefix.to_vec());
    req.set_end_key(end_key);
    req
}

pub(crate) fn destroy_range(client: &mut ClusterClient, store_id: u64, prefix: &[u8]) {
    let req = new_destroy_range_req(prefix);
    let kv_client = client.get_kv_client(store_id);
    let resp = kv_client.unsafe_destroy_range(&req).unwrap();
    assert!(resp.get_error().is_empty(), "{:?}", resp.get_error());
    assert!(!resp.has_region_error());
}

pub(crate) async fn request_dump_snapshot_on_store(
    store: &Store,
    shard_id: u64,
    shard_ver: u64,
    start_ts: u64,
) -> Bytes {
    let uri = Uri::from_str(&format!(
        "http://{}/kvengine/snapshot/{}?start_ts={}&shard_ver={}",
        &store.status_address, shard_id, start_ts, shard_ver,
    ))
    .unwrap();
    let req = Request::get(uri).body(Body::empty()).unwrap();
    let client = hyper::Client::new();
    let resp: http::Response<Body> = client.request(req).await.unwrap();
    assert!(
        resp.status().is_success(),
        "{:?}",
        hyper::body::to_bytes(resp.into_body()).await.unwrap()
    );
    hyper::body::to_bytes(resp.into_body()).await.unwrap()
}

pub(crate) fn new_security_config() -> SecurityConfig {
    let mut conf = SecurityConfig::default();
    conf.master_key.vendor = "test".to_string();
    conf.master_key.key_id = "random".to_string();
    conf
}
