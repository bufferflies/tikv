// Copyright 2025 TiKV Project Authors. Licensed under Apache-2.0.

use builtin_dfs::{BuiltinDfs, BuiltinDfsFileMeta};
use bytes::Bytes;
use http::{Request, StatusCode};
use hyper::{client::HttpConnector, Body, Client};
use kvengine::dfs::{Dfs, FileType, Options};
use pd_client::PdClient;
use test_cloud_server::{alloc_node_id_vec, ServerCluster, ServerClusterBuilder};
use test_util::init_log_for_test;
use tikv::config::TikvConfig;
use tikv_util::info;
use tokio::runtime::Runtime;

/// Test context for BuiltinDfs integration tests.
struct BuiltinDfsTestContext {
    cluster: ServerCluster,
    builtin_dfs: BuiltinDfs,
    status_addr: String,
    http_client: Client<HttpConnector>,
    runtime: Runtime,
    opts: Options,
}

impl BuiltinDfsTestContext {
    /// Create a new test context with a cluster and BuiltinDfs client.
    fn new() -> Self {
        test_util::init_log_for_test();

        // Start a cluster with builtin DFS
        let nodes = alloc_node_id_vec(1);
        let cluster = ServerClusterBuilder::new(nodes.clone(), |_, conf: &mut TikvConfig| {
            conf.dfs.s3_endpoint = "local".to_string();
        })
        .pd_server_cnt(1)
        .build();

        cluster.wait_region_replicated(&[], 1);

        let node_id = nodes[0];
        let status_addr = cluster.status_addr(node_id);
        let http_client = Client::new();
        let runtime = Runtime::new().unwrap();

        // Create BuiltinDfs using the cluster's PD client
        let pd_client = cluster.get_pd_client();
        let builtin_dfs = BuiltinDfs::new(pd_client.clone());

        // Get actual region info
        let region = pd_client.get_region(b"").unwrap();
        let region_id = region.get_id();
        let region_ver = region.get_region_epoch().get_version();
        let opts = Options::default().with_shard(region_id, region_ver);

        Self {
            cluster,
            builtin_dfs,
            status_addr,
            http_client,
            runtime,
            opts,
        }
    }

    /// Create a file via BuiltinDfs and wait for it to be written.
    fn create_file(&self, file_id: u64, file_data: &[u8]) {
        info!("Creating file {} with {} bytes", file_id, file_data.len());
        self.runtime
            .block_on(
                self.builtin_dfs
                    .create(file_id, Bytes::from(file_data.to_vec()), self.opts),
            )
            .expect("Failed to create file via BuiltinDfs");

        // Wait a bit for file to be written
        std::thread::sleep(std::time::Duration::from_millis(100));
    }

    /// Send an HTTP GET request and return the response.
    fn http_get(&self, url: &str) -> hyper::Response<Body> {
        let req = Request::get(url).body(Body::empty()).unwrap();
        self.runtime
            .block_on(self.http_client.request(req))
            .unwrap()
    }

    /// Stop the cluster.
    fn stop(mut self) {
        self.cluster.stop();
    }
}

/// Test status_server meta API via BuiltinDfs and HTTP.
///
/// This test validates the complete meta API functionality:
/// 1. HTTP API returns 404 for non-existent files
/// 2. HTTP API returns 200 with JSON metadata for existing files
/// 3. BuiltinDfs::size() correctly parses file size from response
/// 4. BuiltinDfs::exists() correctly determines file existence
#[test]
fn test_builtin_dfs_meta() {
    init_log_for_test();
    let ctx = BuiltinDfsTestContext::new();

    // Test parameters
    let file_id = 12345u64;
    let file_type = FileType::Sst;
    let file_data = b"Hello, status_server meta API test!";
    let expected_size = file_data.len() as u64;

    // HTTP endpoint for meta query
    let meta_url = format!(
        "http://{}/dfs/{}?file_type={}&meta=true",
        ctx.status_addr,
        file_id,
        file_type.suffix()
    );

    info!("Test 1: meta query before file creation - should return 404");
    let resp = ctx.http_get(&meta_url);
    assert_eq!(
        resp.status(),
        StatusCode::NOT_FOUND,
        "Should return 404 for non-existent file"
    );

    info!("Test 2: BuiltinDfs::exists before file creation - should return false");
    let exists = ctx
        .runtime
        .block_on(ctx.builtin_dfs.exists(file_id, ctx.opts))
        .expect("BuiltinDfs::exists should succeed");
    assert!(!exists, "File should not exist");

    // Create the file
    ctx.create_file(file_id, file_data);

    info!("Test 3: meta query after file creation - should return 200 with metadata");
    let resp = ctx.http_get(&meta_url);
    assert_eq!(
        resp.status(),
        StatusCode::OK,
        "Should return 200 for existing file"
    );

    // Parse JSON response body
    let body_bytes = ctx
        .runtime
        .block_on(hyper::body::to_bytes(resp.into_body()))
        .unwrap();

    let info: BuiltinDfsFileMeta =
        serde_json::from_slice(&body_bytes).expect("Failed to parse meta JSON response");
    assert_eq!(
        info.size, expected_size,
        "File size should match: expected {}, got {}",
        expected_size, info.size
    );

    info!("Test 4: BuiltinDfs::size - should return correct file size");
    let size = ctx
        .runtime
        .block_on(ctx.builtin_dfs.size(file_id, ctx.opts))
        .expect("BuiltinDfs::size should succeed");
    assert_eq!(size, expected_size, "File size should match");

    info!("Test 5: BuiltinDfs::exists after file creation - should return true");
    let exists = ctx
        .runtime
        .block_on(ctx.builtin_dfs.exists(file_id, ctx.opts))
        .expect("BuiltinDfs::exists should succeed");
    assert!(exists, "File should exist");

    info!("All tests completed successfully!");
    ctx.stop();
}
