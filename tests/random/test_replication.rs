// Copyright 2025 TiKV Project Authors. Licensed under Apache-2.0.

use std::{io::Write, path::PathBuf, sync::Arc, thread, time::Duration};

use api_version::ApiV2;
use cloud_worker::CloudWorker;
use futures::executor::block_on;
use log_wrappers::Value as LogValue;
use native_br::backup;
use pd_client::{PdClient, RpcClient};
use replication_worker::{KeyspacesResp, LocalProvider};
use security::{HttpClient, SecurityManager};
use sqlx::Row;
use test_cloud_server::{must_wait, must_wait_result, oss::prepare_dfs};
use tidb_query_datatype::codec::table::encode_row_key;
use tikv_util::{
    codec::bytes::encode_bytes,
    config::{ReadableDuration, ReadableSize},
    info,
    time::Instant,
};

use crate::{test_tidb::*, *};

const KEYSPACE_ID: u32 = 1;

#[test]
fn test_random_replication() {
    init_logger();
    let prepare_time = Instant::now_coarse();
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .worker_threads(1)
        .thread_name("test_random_rep")
        .build()
        .unwrap();
    let _guard = runtime.enter();

    let switches = Switches::from_env();
    info!("switches: {:?}", switches);

    let (_temp_dir, _oss, dfs_conf) = prepare_dfs("oss_");
    let security_conf = new_security_config();
    let tc = prepare_tidb_cluster(&security_conf, &switches);
    let mut cluster = prepare_cluster(
        &dfs_conf,
        &security_conf,
        NODES_COUNT,
        INITIAL_KEYSPACE_COUNT,
        &switches,
        &tc,
    );
    let keyspace_manager = cluster.keyspace_manager().clone();

    let tikv_worker_addr = cluster.tikv_worker_endpoints().pop().unwrap();
    start_components(&tc, tikv_worker_addr, &switches, &dfs_conf, &runtime);
    prepare_workloads(&tc, &keyspace_manager, &switches, &runtime);

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

    let pool = runtime.block_on(connect_tidb(&tc, &keyspace_manager, KEYSPACE_ID));

    info!("create table");
    let table_name = "rep_table";
    let create_table =
        format!("create table {table_name} (id int primary key, col_i int, col_s varchar(1024))");
    block_on(sqlx::query(&create_table).execute(&pool)).unwrap();

    let select_table_id = format!(
        "select tidb_table_id from information_schema.tables where table_schema = 'test' and table_name = '{table_name}'"
    );
    let row = block_on(sqlx::query(&select_table_id).fetch_one(&pool)).unwrap();
    let table_id: i64 = row.get("tidb_table_id");
    let val_fn = generate_random_string("rep".to_string());

    // Prepare workload.
    info!("prepare table");
    for i in 1..=5 {
        let val = String::from_utf8(val_fn(1000)).unwrap();
        let sql = format!("insert into `{table_name}` values ({i}, {i}, '{val}')");
        block_on(sqlx::query(&sql).execute(&pool)).unwrap();
    }

    // Start replication worker.
    info!("start replication worker");
    let rep_dir = tempfile::Builder::new().prefix("rep_").tempdir().unwrap();
    let rep_dir = rep_dir.path();

    let mut worker_conf = cloud_worker::Config::default();
    worker_conf.data_dir = rep_dir.to_str().unwrap().to_string();
    worker_conf.addr = "127.0.0.1:5998".to_string();
    worker_conf.pd.endpoints = tc.pd.endpoints();
    worker_conf.security = security_conf.clone();
    worker_conf.dfs = dfs_conf.clone();
    let rep_config = &mut worker_conf.replication_worker;
    rep_config.override_from_env();
    rep_config.enabled = true;
    rep_config.grpc_addr = "127.0.0.1:5999".to_string();
    rep_config.advertise_addr = "127.0.0.1:5999".to_string();
    rep_config.report_region_interval = ReadableDuration::secs(3);
    rep_config.merged_engine.mem_table_size = ReadableSize::kb(16);
    rep_config.merged_engine.raft_write_batch_size = ReadableSize::kb(256);

    let pd_client = cluster.get_pure_pd_client();
    let mut worker = CloudWorker::new(worker_conf.clone(), None, 2, pd_client.clone());
    worker.start();
    let worker_addr = worker.addr().to_string();
    let worker_client = pd_client
        .get_security_mgr()
        .http_client(hyper::Client::builder())
        .unwrap();
    let mut local_provider = LocalProvider::new(1, PathBuf::from(rep_dir), 6000);
    local_provider.start();
    let pd_url = local_provider.pd_client_url();
    let worker_base_url = format!("http://{}/cdc", worker_addr);
    let cdc_addr = local_provider.cdc_server_addr();
    // add keyspace before add task.
    let add_keyspace_url = format!("{}/keyspace?keyspace_id=1", worker_base_url);
    let add_keyspace_body = format!(r#"{{"pd_url":"{pd_url}","cdc_addr":"{cdc_addr}"}}"#);
    dispatch_http(&worker_client, add_keyspace_url, "POST", add_keyspace_body).unwrap();

    // Verify keyspace is added.
    let get_keyspace_url = format!("{}/keyspace", worker_base_url);
    let res = dispatch_http(&worker_client, get_keyspace_url, "GET", "".to_string()).unwrap();
    let keyspaces: KeyspacesResp = serde_json::from_slice(res.as_bytes()).unwrap();
    assert_eq!(keyspaces.keyspace_ids.len(), 1);

    // Add task.
    let sink_uri = "mysql://root@127.0.0.1:9001".to_string();
    let start_ts = backup_ts;
    let changefeed_id = "rep-task";
    let add_task_url = format!("{}/api/v2/changefeeds?keyspace_id=1", worker_base_url);
    let add_task_body = format!(
        r#"{{"changefeed_id":"{changefeed_id}","sink_uri":"{sink_uri}","start_ts":{start_ts}}}"#
    );
    dispatch_http(&worker_client, add_task_url, "POST", add_task_body).unwrap();

    // Get task list has rep-task.
    let get_task_list_url = format!("{}/api/v2/changefeeds?keyspace_id=1", worker_base_url);
    let resp = dispatch_http(&worker_client, get_task_list_url, "GET", "".to_string()).unwrap();
    assert!(resp.contains(changefeed_id));

    // Start workload.
    info!("start workload");
    let start_time = Instant::now();

    for i in 6..=10 {
        let val = String::from_utf8(val_fn(1000)).unwrap();
        let sql = format!("insert into `{table_name}` values ({i}, {i}, '{val}')");
        block_on(sqlx::query(&sql).execute(&pool)).unwrap();
        thread::sleep(Duration::from_millis(500));
    }

    // Pause the changefeed.
    info!("pause changefeed");
    let pause_task_url =
        format!("{worker_base_url}/api/v2/changefeeds/{changefeed_id}/pause?keyspace_id=1");
    dispatch_http(&worker_client, pause_task_url, "POST", "".to_string()).unwrap();

    // Restart the rep-pd and wait for the rep-pd region has leader.
    info!("restart rep-pd");
    local_provider.restart_local_pd().unwrap();
    let rep_pd_cli = new_rep_pd_client(local_provider.pd_client_url());
    must_wait(
        || {
            let region = match rep_pd_cli.get_region_info(&[]) {
                Ok(region) => region,
                Err(e) => {
                    // Should be caused by region role during split.
                    warn!("rep-pd get region failed: {:?}", e);
                    return false;
                }
            };
            info!("rep-pd region: {:?}", region);
            region.leader.is_some()
        },
        30,
        || "wait for rep-pd region leader".into(),
    );

    // Resume the changefeed.
    info!("resume changefeed");
    let resume_task_url =
        format!("{worker_base_url}/api/v2/changefeeds/{changefeed_id}/resume?keyspace_id=1");
    must_wait_result(
        || {
            // TiCDC may return error when PD is just up.
            dispatch_http(
                &worker_client,
                &resume_task_url,
                "POST",
                // r#"{"overwrite_checkpoint_ts": 0}"#.to_string(),
                r#"{}"#.to_string(),
            )
        },
        30,
        || "wait for resume changefeed".into(),
    );

    let pd_client = cluster.get_pd_client_ext();
    let pd_ctl = cluster.get_pd_control().unwrap();
    let row_key_5 = encode_pd_table_key(table_id, 5);
    block_on(pd_client.split_regions_with_retry(vec![row_key_5.clone()], Duration::from_secs(30)))
        .unwrap();
    let row_key_1 = encode_pd_table_key(table_id, 1);
    for i in 1..=10 {
        let sql = format!("update `{table_name}` set col_i = col_i + 1 where id = {i}");
        block_on(sqlx::query(&sql).execute(&pool)).unwrap();
        thread::sleep(Duration::from_millis(500));
    }
    info!(
        "try to merge region";
        "source" => LogValue::key(&row_key_5),
        "target" => LogValue::key(&row_key_1)
    );
    block_on(pd_ctl.merge_regions_by_key(&row_key_5, &row_key_1, Duration::from_secs(30))).unwrap();

    info!("shutdown replication worker");
    worker.shutdown();

    // write some data to make the wal rotate more than 4 times.
    info!("update workload");
    let update_count = 20;
    for _ in 0..update_count {
        let sql = format!("update `{table_name}` set col_i = col_i + 1");
        block_on(sqlx::query(&sql).execute(&pool)).unwrap();
    }

    // restart the replication worker.
    info!("restart replication worker");
    worker = CloudWorker::new(worker_conf.clone(), None, 2, pd_client.clone());
    worker.start();

    info!("delete workload");
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
        assert_eq!(id + 1 + update_count, col_i);
    }
    assert_eq!(result.len(), 5);

    // Remove task.
    info!("remove task");
    let remove_task_url = format!(
        "{}/api/v2/changefeeds/{changefeed_id}?keyspace_id=1",
        worker_base_url
    );
    dispatch_http(&worker_client, remove_task_url, "DELETE", "".to_string()).unwrap();

    // Get task list doesn't have rep-task.
    let get_task_list_rul = format!("{}/keyspace?keyspace_id=1", worker_base_url);
    let resp = dispatch_http(&worker_client, get_task_list_rul, "GET", "".to_string()).unwrap();
    assert!(!resp.contains(changefeed_id));

    // Remove keyspace.
    info!("remove keyspace");
    let remove_keyspace_url = format!("{worker_base_url}/keyspace?keyspace_id=1");
    dispatch_http(
        &worker_client,
        remove_keyspace_url,
        "DELETE",
        "".to_string(),
    )
    .unwrap();

    // Verify keyspace is removed.
    let get_keyspace_url = format!("{}/keyspace", worker_base_url);
    let res = dispatch_http(&worker_client, get_keyspace_url, "GET", "".to_string()).unwrap();
    let keyspaces: KeyspacesResp = serde_json::from_slice(res.as_bytes()).unwrap();
    assert!(keyspaces.keyspace_ids.is_empty());

    info!("stop components");
    worker.shutdown();
    local_provider.destroy().unwrap();
    tc.tidb.stop_all();
    cluster.stop();
    tc.pd.stop_all();

    // Statistics.
    let stdout = std::io::stdout();
    writeln!(
        stdout.lock(),
        "{} TEST SUCCEED: elapsed {:?},{:?}",
        test_id(),
        prepare_time.saturating_elapsed(),
        start_time.saturating_elapsed(),
    )
    .unwrap();
    stdout.lock().flush().unwrap();
}

fn new_rep_pd_client(pd_url: String) -> Arc<dyn PdClient> {
    let sec_mgr = Arc::new(SecurityManager::default());
    Arc::new(RpcClient::new(&pd_client::Config::new(vec![pd_url]), None, sec_mgr).unwrap())
}

fn dispatch_http<S: AsRef<str>>(
    client: &HttpClient,
    url: S,
    method: &str,
    body: String,
) -> std::result::Result<String, String> {
    let req = http::Request::builder()
        .method(method)
        .uri(url.as_ref())
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
