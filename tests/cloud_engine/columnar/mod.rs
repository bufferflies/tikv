// Copyright 2024 TiKV Project Authors. Licensed under Apache-2.0.

use std::time::Duration;

use api_version::ApiV2;
use hyper::Body;
use kvengine::{
    dfs,
    dfs::FileType,
    table::columnar::{
        builder::{new_int_handle_column_info, new_txn_id_column_info, new_version_column_info},
        columnar::Schema,
        schema_file::build_schema_file,
    },
};
use pd_client::PdClient;
use test_cloud_server::{must_wait, ServerCluster};
use tidb_query_datatype::{codec::table::TABLE_PREFIX, Collation, FieldTypeTp};
use tikv_util::codec::{bytes::encode_bytes, number::NumberEncoder};
use tipb::ColumnInfo;
use txn_types::Key;

use crate::alloc_node_id;

#[test]
fn test_schema_file() {
    test_util::init_log_for_test();
    let node_id = alloc_node_id();
    let mut cluster = ServerCluster::new(vec![node_id], |_, _| {});
    let dfs = cluster.get_dfs().unwrap();
    let keyspace_id = 9;
    let table_ids = dfs
        .get_runtime()
        .block_on(create_keyspace_and_split_tables(&mut cluster, keyspace_id));
    let schemas = build_schemas(vec![table_ids[1], table_ids[3]]);
    let schema_version = 10;
    let schema_file_data = build_schema_file(keyspace_id, schema_version, schemas);
    let schema_file_id = 100;
    let opts = dfs::Options::default().with_type(FileType::Schema);
    dfs.get_runtime()
        .block_on(dfs.create(schema_file_id, schema_file_data.into(), opts))
        .unwrap();
    let status_addr = cluster.status_addr(node_id);
    dfs.get_runtime().block_on(send_schema_file_request(
        &status_addr,
        keyspace_id,
        schema_file_id,
    ));
    let kvengine = cluster.get_kvengine(node_id);
    let all_ids_vers = kvengine.get_all_shard_id_vers();
    assert_eq!(all_ids_vers.len(), 7);
    must_wait(
        || {
            let mut shard_with_schema_file_count = 0;
            for &id_ver in &all_ids_vers {
                let shard = kvengine.get_shard(id_ver.id).unwrap();
                if shard.get_schema_file().is_some() {
                    shard_with_schema_file_count += 1;
                }
            }
            shard_with_schema_file_count == 2
        },
        10,
        || "failed to wait schema file".to_string(),
    );

    // test schema_file will be convert to tombstone if not longer overlap.
    let new_schemas = build_schemas(vec![table_ids[1]]);
    let new_schema_version = 11;
    let new_schema_file_data = build_schema_file(keyspace_id, new_schema_version, new_schemas);
    let new_schema_file_id = 101;
    dfs.get_runtime()
        .block_on(dfs.create(new_schema_file_id, new_schema_file_data.into(), opts))
        .unwrap();
    dfs.get_runtime().block_on(send_schema_file_request(
        &status_addr,
        keyspace_id,
        new_schema_file_id,
    ));
    must_wait(
        || {
            let mut shard_with_schema_file_ids = vec![];
            let mut tombstone_count = 0;
            for &id_ver in &all_ids_vers {
                let shard = kvengine.get_shard(id_ver.id).unwrap();
                if shard.get_schema_file().is_some() {
                    let schema_file_id = shard.get_schema_file().unwrap().get_file_id();
                    if schema_file_id == 0 {
                        tombstone_count += 1;
                    } else {
                        shard_with_schema_file_ids.push(schema_file_id);
                    }
                }
            }
            shard_with_schema_file_ids.len() == 1
                && shard_with_schema_file_ids[0] == new_schema_file_id
                && tombstone_count == 1
        },
        10,
        || "failed to wait schema file".to_string(),
    );

    // test schema file will not update if version is older.
    dfs.get_runtime().block_on(send_schema_file_request(
        &status_addr,
        keyspace_id,
        schema_file_id,
    ));
    std::thread::sleep(Duration::from_secs(3));
    for &id_ver in &all_ids_vers {
        let shard = kvengine.get_shard(id_ver.id).unwrap();
        if shard.get_schema_file().is_some() {
            let sf = shard.get_schema_file().unwrap();
            if sf.is_tombstone() {
                assert_eq!(sf.get_version(), new_schema_version);
            } else {
                assert_eq!(sf.get_file_id(), new_schema_file_id);
            }
            let stats = shard.get_stats();
            assert_eq!(stats.schema_version, new_schema_version);
        }
    }
}

fn build_schemas(table_ids: Vec<i64>) -> Vec<Schema> {
    let mut schemas = vec![];
    for &columnar_table_id in &table_ids {
        let mut c1 = ColumnInfo::new();
        c1.set_column_id(1);
        c1.set_tp(FieldTypeTp::LongLong.to_u8().unwrap() as i32);
        let mut c2 = ColumnInfo::new();
        c2.set_column_id(2);
        c2.set_tp(FieldTypeTp::VarChar.to_u8().unwrap() as i32);
        c2.set_column_len(255);
        c2.set_collation(Collation::Utf8Mb4Bin as i32);
        let schema = Schema {
            table_id: columnar_table_id,
            handle_column: new_int_handle_column_info(),
            version_column: new_version_column_info(),
            txn_id_column: Some(new_txn_id_column_info()),
            columns: vec![c1, c2],
        };
        schemas.push(schema);
    }
    schemas
}

async fn send_schema_file_request(status_addr: &str, keyspace_id: u32, schema_file_id: u64) {
    let request = hyper::http::Request::builder()
        .method(http::method::Method::POST)
        .uri(format!(
            "http://{}/schema_file?keyspace_id={}&file_id={}",
            status_addr, keyspace_id, schema_file_id
        ))
        .body(Body::empty())
        .unwrap();
    let http_client = hyper::client::Client::new();
    let resp = http_client.request(request).await.unwrap();
    assert!(resp.status().is_success());
}

async fn create_keyspace_and_split_tables(
    cluster: &mut ServerCluster,
    keyspace_id: u32,
) -> Vec<i64> {
    let km = cluster.keyspace_manager();
    km.create_single_keyspace(keyspace_id, format!("ks{}", keyspace_id), 4, 4, false)
        .await;
    let keyspace_split_keys = get_keyspace_split_keys(keyspace_id);
    let pd_client = cluster.get_pd_client();
    pd_client.split_regions(keyspace_split_keys).await.unwrap();
    let ks_meta = km.get_keyspace_meta(keyspace_id).unwrap();
    let table_ids = ks_meta.get_all_available_tables();
    let table_split_keys = get_table_split_keys(keyspace_id, &table_ids);
    pd_client.split_regions(table_split_keys).await.unwrap();
    table_ids
}

fn get_keyspace_split_keys(keyspace_id: u32) -> Vec<Vec<u8>> {
    vec![
        ApiV2::get_txn_keyspace_prefix(keyspace_id),
        ApiV2::get_txn_keyspace_prefix(keyspace_id + 1),
    ]
    .into_iter()
    .map(|k| Key::from_raw(&k).into_encoded())
    .collect()
}

fn get_table_split_keys(keyspace_id: u32, table_ids: &[i64]) -> Vec<Vec<u8>> {
    let keyspace_prefix = ApiV2::get_txn_keyspace_prefix(keyspace_id);
    table_ids
        .iter()
        .map(|&tbl_id| {
            let mut buf = vec![];
            buf.extend_from_slice(&keyspace_prefix);
            buf.extend_from_slice(TABLE_PREFIX);
            buf.encode_i64(tbl_id).unwrap();
            encode_bytes(&buf)
        })
        .collect()
}
