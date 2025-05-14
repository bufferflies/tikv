// Copyright 2025 TiKV Project Authors. Licensed under Apache-2.0.

use std::{fs, path::PathBuf, time::Duration};

use cloud_worker::CloudWorker;
use futures::executor::block_on;
use native_br::backup;
use pd_client::pd_control::PdScheduleConfig;
use replication_worker::{KeyspacesResp, LocalProvider};
use security::{HttpClient, SecurityConfig};
use test_cloud_server::{
    must_wait,
    oss::prepare_dfs,
    tidb::{PdServerMode, StartTidbOptions, TidbCluster, PD_CLIENT_UPDATE_INTERVAL},
    ServerClusterBuilder,
};
use test_pd_client::PdWrapper;
use tikv_util::config::ReadableSize;

use crate::alloc_node_id_vec;

#[test]
fn test_replication_worker() {
    test_util::init_log_for_test();
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .worker_threads(1)
        .thread_name("test_replication_worker")
        .build()
        .unwrap();
    let _guard = runtime.enter();

    let (base_dir, _oss, dfs_conf) = prepare_dfs("test_merged_engine");
    let pd_schedule_config = PdScheduleConfig::default();
    let pd_bin = std::env::var("PD_BIN").expect("env PD_BIN is not set");
    let pd_bin_path = PathBuf::from(&pd_bin);
    let tidb_bin = std::env::var("TIDB_BIN").expect("env TIDB_BIN is not set");
    let tidb_bin_path = PathBuf::from(&tidb_bin);
    let tc = TidbCluster::new(
        PdServerMode::Normal,
        pd_bin_path.clone(),
        2379,
        pd_schedule_config,
        tidb_bin_path.clone(),
        4000,
        5000,
        pd_bin_path.clone(), // use pd bin path to bypass tiflash binary check.
        1,
        &Default::default(),
    );
    block_on(tc.start_pd(1, Duration::from_secs(5)));
    let security_conf = SecurityConfig::default();
    let pd = PdWrapper::new_real(tc.pd.endpoints(), &security_conf, PD_CLIENT_UPDATE_INTERVAL);
    let node_ids = alloc_node_id_vec(4);
    let mut cluster = ServerClusterBuilder::new(node_ids.clone(), |_, conf| {
        conf.dfs = dfs_conf.clone();
        conf.rfengine.lightweight_backup = true;
        conf.rfengine.target_file_size = ReadableSize::mb(8);
    })
    .pd(pd)
    .build();
    let pd_ctl = cluster.get_pd_control().unwrap();
    must_wait(
        || {
            let region_count = block_on(pd_ctl.get_regions_number()).unwrap();
            region_count == 4
        },
        10,
        || "wait for region merge".into(),
    );
    let tidb_opts = StartTidbOptions::default();
    block_on(tc.start_tidb(1, Duration::from_secs(10), "info", tidb_opts));

    let backup_config = backup::BackupConfig {
        dfs: dfs_conf.clone(),
        skip_keyspace_meta: true,
        ..Default::default()
    };
    let client = cluster.new_client();
    let backup_ts = client.get_ts().into_inner();
    backup::backup_cluster_with_ts(
        backup_config,
        backup::BackupType::Lightweight,
        "".into(),
        cluster.get_pure_pd_client().as_ref(),
        backup_ts,
        None,
    )
    .expect("backup::backup_cluster");

    let rep_dir = base_dir
        .path()
        .join("replication_worker")
        .to_str()
        .unwrap()
        .to_string();
    fs::create_dir_all(&rep_dir).unwrap();

    let mut worker_conf = cloud_worker::Config::default();
    worker_conf.data_dir = rep_dir.clone();
    worker_conf.addr = "127.0.0.1:5998".to_string();
    worker_conf.pd.endpoints = tc.pd.endpoints();
    worker_conf.security = security_conf.clone();
    worker_conf.dfs = dfs_conf.clone();
    let rep_config = &mut worker_conf.replication_worker;
    rep_config.enabled = true;
    rep_config.grpc_addr = "127.0.0.1:5999".to_string();
    rep_config.merged_engine.mem_table_size = ReadableSize::kb(16);

    let pd_client = cluster.get_pure_pd_client();
    let mut worker = CloudWorker::new(worker_conf.clone(), None, 2, pd_client.clone());
    worker.start();
    let worker_addr = worker.addr().to_string();
    let worker_client = pd_client
        .get_security_mgr()
        .http_client(hyper::Client::builder())
        .unwrap();
    let mut local_provider = LocalProvider::new(1, PathBuf::from(rep_dir.clone()), 6000);
    local_provider.start();
    let pd_url = local_provider.pd_client_url();
    let worker_base_url = format!("http://{}/cdc", worker_addr);
    let cdc_addr = local_provider.cdc_server_addr();
    // add keyspace before add task.
    let add_keyspace_url = format!("{}/keyspace?keyspace_id=1", worker_base_url);
    let add_keyspace_body = format!(r#"{{"pd_url":"{pd_url}","cdc_addr":"{cdc_addr}"}}"#);
    dispatch_http(&worker_client, add_keyspace_url, "POST", add_keyspace_body).unwrap();

    // verify keyspace is added
    let get_keyspace_url = format!("{}/keyspace", worker_base_url);
    let res = dispatch_http(&worker_client, get_keyspace_url, "GET", "".to_string()).unwrap();
    let keyspaces: KeyspacesResp = serde_json::from_slice(res.as_bytes()).unwrap();
    assert_eq!(keyspaces.keyspace_ids.len(), 1);

    // remove keyspace
    let remove_keyspace_url = format!("{worker_base_url}/keyspace?keyspace_id=1");
    dispatch_http(
        &worker_client,
        remove_keyspace_url,
        "DELETE",
        "".to_string(),
    )
    .unwrap();

    // verify keyspace is removed
    let get_keyspace_url = format!("{}/keyspace", worker_base_url);
    let res = dispatch_http(&worker_client, get_keyspace_url, "GET", "".to_string()).unwrap();
    let keyspaces: KeyspacesResp = serde_json::from_slice(res.as_bytes()).unwrap();
    assert!(keyspaces.keyspace_ids.is_empty());

    worker.shutdown();
    local_provider.destroy().unwrap();
    tc.tidb.stop_all();
    cluster.stop();
    tc.pd.stop_all();
}

fn dispatch_http(
    client: &HttpClient,
    url: String,
    method: &str,
    body: String,
) -> Result<String, String> {
    let req = http::Request::builder()
        .method(method)
        .uri(url)
        .body(body.into())
        .unwrap();
    let resp =
        block_on(client.request(req)).map_err(|e| format!("failed to send request: {}", e))?;
    let status = resp.status();
    let body = block_on(hyper::body::to_bytes(resp.into_body()))
        .map_err(|e| format!("failed to read response body: {}", e))?;
    let body_str = String::from_utf8(body.to_vec()).map_err(|e| format!("invalid utf-8: {}", e))?;
    if !status.is_success() {
        return Err(body_str);
    }
    Ok(body_str)
}
