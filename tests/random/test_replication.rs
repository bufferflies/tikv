// Copyright 2025 TiKV Project Authors. Licensed under Apache-2.0.

use std::{fs, path::PathBuf, thread, time::Duration};

use api_version::ApiV2;
use cloud_worker::CloudWorker;
use futures::executor::block_on;
use native_br::backup;
use pd_client::pd_control::PdScheduleConfig;
use replication_worker::{KeyspacesResp, LocalProvider, ReplicationWorkerConfig};
use security::{HttpClient, SecurityConfig};
use sqlx::Row;
use test_cloud_server::{
    must_wait,
    oss::prepare_dfs,
    tidb::{PdServerMode, StartTidbOptions, TidbCluster, PD_CLIENT_UPDATE_INTERVAL},
    ServerClusterBuilder,
};
use test_pd_client::PdWrapper;
use tidb_query_datatype::codec::table::encode_row_key;
use tikv_util::{codec::bytes::encode_bytes, config::ReadableSize, info};

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
    let params = tc.tidb.conn_params(1);
    let opts = sqlx::mysql::MySqlConnectOptions::new()
        .host(&params.host)
        .port(params.port)
        .username(&params.user)
        .database("test");
    let pool = block_on(sqlx::mysql::MySqlPoolOptions::new().connect_with(opts)).unwrap();

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
    let mut rep_config = ReplicationWorkerConfig::default();
    rep_config.override_from_env();
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

    let sink_uri = "mysql://root@127.0.0.1:9001".to_string();
    let start_ts = backup_ts;
    let changefeed_id = "rep-task";
    let add_task_url = format!("{}/api/v2/changefeeds?keyspace_id=1", worker_base_url);
    let add_task_body = format!(
        r#"{{"changefeed_id":"{changefeed_id}","sink_uri":"{sink_uri}","start_ts":{start_ts}}}"#
    );
    dispatch_http(&worker_client, add_task_url, "POST", add_task_body).unwrap();

    // get task list has rep-task.
    let get_task_list_url = format!("{}/api/v2/changefeeds?keyspace_id=1", worker_base_url);
    let resp = dispatch_http(&worker_client, get_task_list_url, "GET", "".to_string()).unwrap();
    assert!(resp.contains(changefeed_id));

    let table_name = "rep_table";
    let create_table =
        format!("create table {table_name} (id int primary key, col_i int, col_s varchar(255))");
    block_on(sqlx::query(&create_table).execute(&pool)).unwrap();

    let select_table_id = format!(
        "select tidb_table_id from information_schema.tables where table_schema = 'test' and table_name = '{table_name}'"
    );
    let row = block_on(sqlx::query(&select_table_id).fetch_one(&pool)).unwrap();
    let table_id: i64 = row.get("tidb_table_id");

    for i in 1..=10 {
        let sql = format!("insert into `{table_name}` values ({i}, {i}, 'test_{i}')",);
        block_on(sqlx::query(&sql).execute(&pool)).unwrap();
        thread::sleep(Duration::from_millis(500));
    }

    let pd_client = cluster.get_pure_pd_client();
    let row_key_5 = encode_pd_table_key(table_id, 5);
    let origin_region_id = pd_client.get_region(&row_key_5).unwrap().id;
    block_on(pd_client.split_regions(vec![row_key_5.clone()])).unwrap();
    let pd_ctl = cluster.get_pd_control().unwrap();
    must_wait(
        || {
            let region_count = block_on(pd_ctl.get_regions_number()).unwrap();
            region_count == 5
        },
        10,
        || "wait for region split".into(),
    );
    let row_key_1 = encode_pd_table_key(table_id, 1);
    let new_region_id = pd_client.get_region(&row_key_1).unwrap().id;
    for i in 1..=10 {
        let sql = format!("update `{table_name}` set col_i = col_i + 1 where id = {i}");
        block_on(sqlx::query(&sql).execute(&pool)).unwrap();
        thread::sleep(Duration::from_millis(500));
    }
    info!(
        "try to merge region {} to {}",
        origin_region_id, new_region_id
    );
    block_on(pd_ctl.merge_regions(origin_region_id, new_region_id)).unwrap();
    must_wait(
        || {
            let region_count = block_on(pd_ctl.get_regions_number()).unwrap();
            info!("region count {} after merge", region_count);
            region_count == 4
        },
        10,
        || "wait for region merge".into(),
    );
    worker.shutdown();
    thread::sleep(Duration::from_secs(1));

    // restart the replication worker.
    worker = CloudWorker::new(worker_conf.clone(), None, 2, pd_client.clone());
    worker.start();
    for i in 4..=8 {
        let sql = format!("delete from `{table_name}` where id = {i}");
        block_on(sqlx::query(&sql).execute(&pool)).unwrap();
        thread::sleep(Duration::from_millis(500));
    }
    thread::sleep(Duration::from_secs(5));
    let opts2 = sqlx::mysql::MySqlConnectOptions::new()
        .host("127.0.0.1")
        .port(9001)
        .username("root")
        .database("test");
    let pool_downstream =
        block_on(sqlx::mysql::MySqlPoolOptions::new().connect_with(opts2)).unwrap();
    let query2 = format!("select id, col_i from {table_name}");
    let result = block_on(sqlx::query(&query2).fetch_all(&pool_downstream)).unwrap();
    for row in result.iter() {
        let id: i32 = row.get("id");
        let col_i: i32 = row.get("col_i");
        info!("id: {}, col_i: {}", id, col_i);
        assert_eq!(id + 1, col_i);
    }
    assert_eq!(result.len(), 5);

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

fn encode_pd_table_key(table_id: i64, handle: i64) -> Vec<u8> {
    let mut raw_split_key = ApiV2::get_keyspace_prefix_by_id(1);
    let table_row_key = encode_row_key(table_id, handle);
    raw_split_key.extend_from_slice(&table_row_key);
    encode_bytes(&raw_split_key)
}
