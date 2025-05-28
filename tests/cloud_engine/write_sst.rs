// Copyright 2025 TiKV Project Authors. Licensed under Apache-2.0.

use std::{iter::FromIterator, time::Duration};

use bytes::{BufMut, BytesMut};
use cloud_server::status_server::SstMeta;
use http::{Method, Request, StatusCode};
use hyper::{Body, Client};
use kvengine::dfs::DFSConfig;
use serde_json::{json, Value as JsonValue};
use test_cloud_server::{
    alloc_node_id_vec, oss::ObjectStorageService, ServerClusterBuilder, TikvWorkerOptions,
};
use tikv::config::TikvConfig;
use tikv_util::codec::bytes::encode_bytes;

use crate::{alloc_node_id, i_to_tidb_key, i_to_val};

const DATA_SIZE: usize = 50;
const NODES_SIZE: usize = 1;

#[test]
fn test_write_sst() {
    test_util::init_log_for_test();

    let base_dir = tempfile::Builder::new()
        .prefix("test_write_sst")
        .tempdir()
        .unwrap();

    // Start mock S3 service
    let oss_dir = base_dir.path().join("oss");
    let mut oss = ObjectStorageService::new(oss_dir.clone());
    oss.start_server();

    // Configure DFS (used by both TiKV and Worker)
    let dfs_config = DFSConfig {
        prefix: "test_write_sst".to_string(),
        s3_endpoint: format!("http://127.0.0.1:{}", oss.port()),
        s3_key_id: "admin".to_string(),
        s3_secret_key: "admin".to_string(),
        s3_bucket: "test_write_sst".to_string(),
        s3_region: "local".to_string(),
        zstd_compression_level: "3".to_string(),
        ..Default::default()
    };

    // Start TiKV cluster (provides PD)
    let nodes = Vec::from_iter((0..NODES_SIZE).map(|_| alloc_node_id()));
    let mut cluster = ServerClusterBuilder::new(nodes.clone(), |_, conf: &mut TikvConfig| {
        conf.dfs = dfs_config.clone(); // Use the same DFS config
    })
    .pd_server_cnt(1)
    .tikv_worker_cnt(1)
    .build();
    let dfs = cluster.get_dfs().unwrap();
    let runtime = dfs.get_runtime();
    cluster.start_tikv_workers(alloc_node_id_vec(1), TikvWorkerOptions::default());
    cluster.wait_region_replicated(&[], NODES_SIZE);
    let mut tikv_client = cluster.new_client(); // Get client for later use if needed
    let pd_client = tikv_client.pd_client.clone();
    let cluster_id = pd_client.get_cluster_id().unwrap();

    // Generate data [0..DATA_SIZE)
    let kvs: Vec<_> = (0..DATA_SIZE)
        .map(|i| (i_to_tidb_key(i), i_to_val(i)))
        .collect();

    // Encode data
    let mut body_buf = BytesMut::new();
    for (key, val) in &kvs {
        body_buf.put_u16_le(key.len() as u16);
        body_buf.put_slice(key);
        body_buf.put_u32_le(val.len() as u32);
        body_buf.put_slice(val);
    }
    let body_bytes = body_buf.freeze();

    // Send PUT /write_sst request
    // Use the cluster's PD client to get timestamp
    let commit_ts = runtime.block_on(pd_client.get_tso()).unwrap().into_inner();
    let write_url = format!(
        "http://{}/write_sst?cluster_id={}&commit_ts={}",
        cluster.tikv_worker_endpoints()[0],
        cluster_id,
        commit_ts
    );
    let http_client = Client::new();

    let req = Request::builder()
        .method(Method::PUT)
        .uri(&write_url)
        .header("Content-Type", "application/octet-stream")
        .body(Body::from(body_bytes.clone()))
        .unwrap();

    let resp = runtime.block_on(http_client.request(req)).unwrap();
    assert!(
        resp.status() == StatusCode::OK,
        "write_sst request failed: {:?}",
        resp.status()
    );

    // Read response body into bytes for JSON parsing
    let bytes = runtime
        .block_on(hyper::body::to_bytes(resp.into_body()))
        .unwrap();

    // Parse SstMeta from response
    let resp_body: JsonValue = serde_json::from_slice(&bytes).unwrap();
    let sst_meta_json = resp_body.get("sst_meta").unwrap();
    let sst_meta: SstMeta = serde_json::from_value(sst_meta_json.clone()).unwrap();

    // Get region info for ingestion
    let target_key = kvs[0].0.clone();
    // Use the PD client obtained earlier
    let region = pd_client.get_region(&target_key).unwrap();
    let region_id = region.get_id();
    let epoch = region.get_region_epoch().clone();

    // Send /ingest_s3 request to TiKV node
    let node_id = nodes[0];
    let status_addr = cluster.status_addr(node_id);
    let ingest_url = format!(
        "http://{}/ingest_s3?cluster_id={}&region_id={}&epoch_version={}",
        status_addr,
        cluster_id,
        region_id,
        epoch.get_version()
    );
    let req_body = json!(sst_meta).to_string();

    let ingest_req = Request::builder()
        .method(Method::POST)
        .uri(&ingest_url)
        .header("Content-Type", "application/json")
        .body(Body::from(req_body))
        .unwrap();

    let ingest_resp = runtime.block_on(http_client.request(ingest_req)).unwrap();
    assert!(
        ingest_resp.status().is_success(),
        "ingest_s3 request failed: {:?}",
        ingest_resp.body()
    );

    // Verify data in TiKV
    std::thread::sleep(Duration::from_secs(1));
    for (key, expected_val) in &kvs {
        let (val, _) = tikv_client.must_get_key(key, tikv_util::time::Instant::now());
        assert_eq!(
            val,
            *expected_val,
            "Value mismatch for key {:?} after ingestion",
            String::from_utf8_lossy(key)
        );
    }

    // --- Extra case: Ingest SST with non-overlapping range should fail
    // Generate New Data [DATA_SIZE, DATA_SIZE * 2)
    let new_kvs: Vec<_> = (DATA_SIZE..DATA_SIZE * 2)
        .map(|i| (i_to_tidb_key(i), i_to_val(i)))
        .collect();

    // Split Region at key(DATA_SIZE)
    let split_key = encode_bytes(&i_to_tidb_key(DATA_SIZE));
    runtime
        .block_on(tikv_client.pd_client.split_regions(vec![split_key]))
        .unwrap();
    cluster.wait_pd_region_count(2);

    // Get the regions after split
    let key_in_left = encode_bytes(&i_to_tidb_key(0));
    let key_in_right = encode_bytes(&i_to_tidb_key(DATA_SIZE));

    let left_region = pd_client.get_region(&key_in_left).unwrap();
    let right_region = pd_client.get_region(&key_in_right).unwrap();
    assert_ne!(
        left_region.get_id(),
        right_region.get_id(),
        "Split should result in two different regions"
    );

    let left_region_id = left_region.get_id();
    let left_epoch = left_region.get_region_epoch().clone();

    // Create SST for new data
    let mut body_buf = BytesMut::new();
    for (key, val) in &new_kvs {
        body_buf.put_u16_le(key.len() as u16);
        body_buf.put_slice(key);
        body_buf.put_u32_le(val.len() as u32);
        body_buf.put_slice(val);
    }
    let new_body_bytes = body_buf.freeze();

    // Send PUT /write_sst for new data
    let new_commit_ts = runtime.block_on(pd_client.get_tso()).unwrap().into_inner();
    let new_write_url = format!(
        "http://{}/write_sst?cluster_id={}&commit_ts={}",
        cluster.tikv_worker_endpoints()[0],
        cluster_id,
        new_commit_ts
    );

    let new_req = Request::builder()
        .method(Method::PUT)
        .uri(&new_write_url)
        .header("Content-Type", "application/octet-stream")
        .body(Body::from(new_body_bytes.clone()))
        .unwrap();

    let new_resp = runtime.block_on(http_client.request(new_req)).unwrap();
    assert!(
        new_resp.status() == StatusCode::OK,
        "New write_sst request failed: {:?}",
        new_resp.status()
    );

    // Parse SstMeta from new response
    let new_bytes = runtime
        .block_on(hyper::body::to_bytes(new_resp.into_body()))
        .unwrap();
    let new_resp_body: JsonValue = serde_json::from_slice(&new_bytes).unwrap();
    let new_sst_meta_json = new_resp_body.get("sst_meta").unwrap();
    let new_sst_meta: SstMeta = serde_json::from_value(new_sst_meta_json.clone()).unwrap();

    // Try to Ingest New SST to Left Region (Failure Case)
    let node_id = nodes[0];
    let status_addr = cluster.status_addr(node_id);
    let ingest_url = format!(
        "http://{}/ingest_s3?cluster_id={}&region_id={}&epoch_version={}",
        status_addr,
        cluster_id,
        left_region_id,
        left_epoch.get_version()
    );
    let req_body = json!(new_sst_meta).to_string();

    let ingest_req = Request::builder()
        .method(Method::POST)
        .uri(&ingest_url)
        .header("Content-Type", "application/json")
        .body(Body::from(req_body))
        .unwrap();

    let ingest_resp = runtime.block_on(http_client.request(ingest_req)).unwrap();
    let status = ingest_resp.status();
    // Expect failure because the SST range [key(DATA_SIZE), key(DATA_SIZE*2-1)]
    // does not overlap with the left region range [start_key, key(DATA_SIZE))
    assert!(
        !status.is_success(),
        "Ingest_s3 request to left region succeeded unexpectedly: {:?}",
        status
    );

    // Cleanup
    cluster.stop();
    oss.shutdown();
}
