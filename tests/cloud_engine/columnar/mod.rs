// Copyright 2024 TiKV Project Authors. Licensed under Apache-2.0.

mod copr;
mod vector_index;

use std::{
    sync::{Arc, Mutex},
    time::Duration,
};

use api_version::ApiV2;
use bytes::Buf;
use dashmap::DashMap;
use futures::{future::ok, TryStreamExt};
use hyper::Body;
use kvengine::{
    dfs,
    dfs::FileType,
    table::{
        columnar,
        columnar::{
            build_schema_file, new_int_handle_column_info, new_txn_id_column_info,
            new_version_column_info, ColumnarFilterReader, Schema, SchemaBuf,
        },
        sstable::BlockCache,
    },
    ColumnarStatusResp, SnapAccess, WRITE_CF,
};
use kvproto::coprocessor::DelegateResponse;
use pd_client::PdClient;
use protobuf::Message;
use test_cloud_server::{
    client::{CommitAction, MutateOptions},
    must_wait,
    oss::prepare_dfs,
    ServerCluster,
};
use test_pd_client::PdClientExt;
use tidb_query_datatype::{
    codec::{
        row::v2::encoder_for_test::{Column, RowEncoder},
        table::{encode_row_key, TABLE_PREFIX},
    },
    expr::EvalContext,
    Collation, FieldTypeTp,
};
use tikv_util::{
    codec::{bytes::encode_bytes, number::NumberEncoder},
    info,
};
use tipb::ColumnInfo;
use txn_types::Key;

use crate::{alloc_node_id, request_dump_snapshot_on_store};

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
    let schema_file_data = build_schema_file(keyspace_id, schema_version, schemas, 0);
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

    // test schema_file will be set to None if not longer overlap.
    let new_schemas = build_schemas(vec![table_ids[1]]);
    let new_schema_version = 11;
    let new_schema_file_data = build_schema_file(keyspace_id, new_schema_version, new_schemas, 0);
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
            for &id_ver in &all_ids_vers {
                let shard = kvengine.get_shard(id_ver.id).unwrap();
                if shard.get_schema_file().is_some() {
                    let schema_file_id = shard.get_schema_file().unwrap().get_file_id();
                    shard_with_schema_file_ids.push(schema_file_id);
                }
            }
            shard_with_schema_file_ids.len() == 1
                && shard_with_schema_file_ids[0] == new_schema_file_id
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

    // test collect columnar status api
    let columnar_status = dfs
        .get_runtime()
        .block_on(send_collect_columnar_status_request(
            &status_addr,
            keyspace_id,
            table_ids[1],
        ));
    assert!(
        columnar_status.total > 0 && columnar_status.ready == columnar_status.total,
        "columnar status is not ready, columnar_status: {:#?}",
        columnar_status
    );
}

#[test]
fn test_covert_row_to_columnar() {
    test_util::init_log_for_test();
    let node_id = alloc_node_id();
    let mut cluster = ServerCluster::new(vec![node_id], |_, conf| {
        conf.enable_inner_key_offset = true;
        conf.kvengine
            .columnar_table_build_options
            .max_columnar_table_size = 1024;
        conf.kvengine
            .columnar_table_build_options
            .pack_max_row_count = 9;
    });
    let dfs = cluster.get_dfs().unwrap();
    let keyspace_id = 7;
    let table_ids = dfs
        .get_runtime()
        .block_on(create_keyspace_and_split_tables(&mut cluster, keyspace_id));
    let table_id = table_ids[1];
    let schemas = build_schemas(vec![table_id]);
    let mut schema_buf = schemas[0].to_schema_buf();
    schema_buf.txn_id_column = None;
    let schema = Schema::new(schema_buf);
    let schema_version = 10;
    let schema_file_data = build_schema_file(keyspace_id, schema_version, schemas, 0);
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
    must_wait(
        || {
            let all_id_vers = kvengine.get_all_shard_id_vers();
            for id_ver in all_id_vers {
                if let Ok(shard) = kvengine.get_shard_with_ver(id_ver.id, id_ver.ver) {
                    if shard.get_schema_file().is_some() {
                        return true;
                    }
                }
            }
            false
        },
        10,
        || "failed to build schema file".to_string(),
    );
    let mut client = cluster.new_client();
    let ctx = Mutex::new(EvalContext::default());
    client.put_kv(
        0..100,
        |i: usize| gen_row_key(keyspace_id, table_id, i),
        |i: usize| gen_row_val(&ctx, i),
    );
    client.put_kv(
        100..200,
        |i: usize| gen_row_key(keyspace_id, table_id, i),
        |i: usize| gen_row_val(&ctx, i),
    );
    let mut shard_id = None;
    must_wait(
        || {
            let all_id_vers = kvengine.get_all_shard_id_vers();
            for id_ver in all_id_vers {
                if let Ok(shard) = kvengine.get_shard_with_ver(id_ver.id, id_ver.ver) {
                    let snap_version = shard.get_snap_version();
                    let columnar_snap_version = shard.get_columnar_snap_version();
                    if snap_version == columnar_snap_version {
                        shard_id = Some(id_ver.id);
                        return true;
                    }
                }
            }
            false
        },
        10,
        || {
            // dump shard info
            let all_id_vers = kvengine.get_all_shard_id_vers();
            for id_ver in all_id_vers {
                if let Ok(shard) = kvengine.get_shard_with_ver(id_ver.id, id_ver.ver) {
                    let snap_version = shard.get_snap_version();
                    let columnar_snap_version = shard.get_columnar_snap_version();
                    info!(
                        "shard: {:?}, snap_version: {}, columnar_snap_version: {}",
                        id_ver, snap_version, columnar_snap_version
                    );
                }
            }
            "failed to build columnar file".to_string()
        },
    );
    let shard_id = shard_id.unwrap();
    let shard = kvengine.get_shard(shard_id).unwrap();
    let snap_access = shard.new_snap_access();
    let ts = client.get_ts().into_inner();
    let mut columnar_reader = snap_access
        .new_columnar_mvcc_reader(schema.table_id, &schema.columns, ts)
        .unwrap();
    columnar_reader.set_int_handle_range(0, Some(190)).unwrap();
    let mut block = columnar::Block::new(&schema);
    let read_rows = columnar_reader.read_block(&mut block, 200).unwrap();
    assert_eq!(read_rows, 190);
    for i in 0..read_rows {
        let handle = block.get_handle_buf().get_int_handle_value(i);
        assert_eq!(handle, i as i64);
        let columns = block.get_columns();
        assert_eq!(columns[0].get_not_null_value(i).get_i64_le(), i as i64);
        let str_val = gen_str_val(i);
        assert_eq!(columns[1].get_not_null_value(i), &str_val);
    }
}

#[test]
fn test_get_snapshot_from_leader_by_status_api() {
    test_util::init_log_for_test();
    let node_id = alloc_node_id();
    // prepare dfs
    let (_temp_dir, mut oss, dfs_config) =
        prepare_dfs("test_get_snapshot_from_leader_by_status_api");
    let mut cluster = ServerCluster::new(vec![node_id], |_, conf| {
        conf.enable_inner_key_offset = true;
        conf.kvengine
            .columnar_table_build_options
            .max_columnar_table_size = 1024;
        conf.kvengine
            .columnar_table_build_options
            .pack_max_row_count = 9;
        conf.dfs = dfs_config.clone();
    });
    let dfs = cluster.get_dfs().unwrap();
    let keyspace_id = 7;
    let table_ids = dfs
        .get_runtime()
        .block_on(create_keyspace_and_split_tables(&mut cluster, keyspace_id));
    let table_id = table_ids[1];
    let schemas = build_schemas(vec![table_id]);
    let mut schema_buf = schemas[0].to_schema_buf();
    schema_buf.txn_id_column = None;
    let schema_version = 10;
    let schema_file_data = build_schema_file(keyspace_id, schema_version, schemas, 0);
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
    must_wait(
        || {
            let all_id_vers = kvengine.get_all_shard_id_vers();
            for id_ver in all_id_vers {
                if let Ok(shard) = kvengine.get_shard_with_ver(id_ver.id, id_ver.ver) {
                    if shard.get_schema_file().is_some() {
                        return true;
                    }
                }
            }
            false
        },
        10,
        || "failed to build schema file".to_string(),
    );
    let master_key = kvengine.get_master_key();
    let mut client = cluster.new_client();
    let ctx = Mutex::new(EvalContext::default());
    client.put_kv(
        0..100,
        |i: usize| gen_row_key(keyspace_id, table_id, i),
        |i: usize| gen_row_val(&ctx, i),
    );
    client.put_kv(
        100..200,
        |i: usize| gen_row_key(keyspace_id, table_id, i),
        |i: usize| gen_row_val(&ctx, i),
    );
    must_wait(
        || {
            let all_id_vers = kvengine.get_all_shard_id_vers();
            for id_ver in all_id_vers {
                if let Ok(shard) = kvengine.get_shard_with_ver(id_ver.id, id_ver.ver) {
                    let snap_version = shard.get_snap_version();
                    let columnar_snap_version = shard.get_columnar_snap_version();
                    if snap_version == columnar_snap_version {
                        return true;
                    }
                }
            }
            false
        },
        10,
        || "failed to build columnar file".to_string(),
    );

    let pd_client = cluster.get_pd_client();
    let shard = pd_client
        .get_region(&encode_bytes(&gen_row_key(keyspace_id, table_id, 0)))
        .unwrap();

    let store_id = kvengine.get_engine_id();
    let store = pd_client.get_store(store_id).unwrap();
    let start_ts = client.get_ts().into_inner();

    // return region epoch not match in header
    let snapshot_from_remote = dfs.get_runtime().block_on(request_dump_snapshot_on_store(
        &store,
        shard.id,
        shard.region_epoch.as_ref().unwrap().version - 1,
        start_ts,
    ));
    let mut delegate_resp = DelegateResponse::default();
    delegate_resp
        .merge_from_bytes(&snapshot_from_remote)
        .unwrap();
    assert!(delegate_resp.get_region_error().has_epoch_not_match());
    assert!(delegate_resp.get_mem_table_data().is_empty());
    assert!(delegate_resp.get_snapshot().is_empty());

    let snapshot_from_remote = dfs.get_runtime().block_on(request_dump_snapshot_on_store(
        &store,
        shard.id,
        shard.region_epoch.as_ref().unwrap().version,
        start_ts,
    ));
    delegate_resp
        .merge_from_bytes(&snapshot_from_remote)
        .unwrap();
    let schema_files = Arc::new(DashMap::new());
    let snap_access = dfs
        .get_runtime()
        .block_on(SnapAccess::construct_snapshot(
            "test".to_owned(),
            dfs.clone(),
            delegate_resp.get_mem_table_data(),
            delegate_resp.get_snapshot(),
            &master_key,
            BlockCache::None,
            Some(schema_files.clone()),
            kvengine.get_txn_chunk_manager(),
        ))
        .unwrap();
    assert!(snap_access.has_schema_file());
    assert!(schema_files.contains_key(&schema_file_id));

    let mut iter = snap_access.new_iterator(WRITE_CF, false, true, Some(start_ts), false);
    iter.rewind();
    let mut i = 0;
    while iter.valid() {
        assert_eq!(iter.key(), &gen_row_key(keyspace_id, table_id, i));
        assert_eq!(iter.val(), &gen_row_val(&ctx, i));
        iter.next();
        i += 1;
    }
    assert_eq!(i, 200);

    // write some locks
    let mut opts = MutateOptions::default();
    opts.commit_action = CommitAction::NoCommit;
    client
        .try_put_kv(
            0..100,
            |i: usize| gen_row_key(keyspace_id, table_id, i),
            |i: usize| gen_row_val(&ctx, i),
            opts,
        )
        .unwrap();
    // use old start_ts should not return error
    let snapshot_from_remote = dfs.get_runtime().block_on(request_dump_snapshot_on_store(
        &store,
        shard.id,
        shard.region_epoch.as_ref().unwrap().version,
        start_ts,
    ));
    let mut delegate_resp = DelegateResponse::default();
    delegate_resp
        .merge_from_bytes(&snapshot_from_remote)
        .unwrap();
    assert!(!delegate_resp.has_locked());
    // use new start_ts should return lock error
    let snapshot_from_remote = dfs.get_runtime().block_on(request_dump_snapshot_on_store(
        &store,
        shard.id,
        shard.region_epoch.unwrap().version,
        client.get_ts().into_inner(),
    ));
    let mut delegate_resp = DelegateResponse::default();
    delegate_resp
        .merge_from_bytes(&snapshot_from_remote)
        .unwrap();
    assert!(delegate_resp.has_locked());

    cluster.stop();
    oss.shutdown();
}

#[test]
fn test_region_merge_with_columnar() {
    test_util::init_log_for_test();
    let node_id = alloc_node_id();
    let mut cluster = ServerCluster::new(vec![node_id], |_, conf| {
        conf.enable_inner_key_offset = true;
        conf.kvengine
            .columnar_table_build_options
            .max_columnar_table_size = 1024;
        conf.kvengine
            .columnar_table_build_options
            .pack_max_row_count = 9;
    });
    let dfs = cluster.get_dfs().unwrap();
    let keyspace_id = 7;
    let table_ids = dfs
        .get_runtime()
        .block_on(create_keyspace_and_split_tables(&mut cluster, keyspace_id));
    let table_id = table_ids[1];
    let schemas = build_schemas(vec![table_id]);
    let mut schema_buf = schemas[0].to_schema_buf();
    schema_buf.txn_id_column = None;
    let schema = Schema::new(schema_buf);
    let schema_version = 10;
    let schema_file_data = build_schema_file(keyspace_id, schema_version, schemas, 0);
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
    must_wait(
        || {
            let all_id_vers = kvengine.get_all_shard_id_vers();
            for id_ver in all_id_vers {
                if let Ok(shard) = kvengine.get_shard_with_ver(id_ver.id, id_ver.ver) {
                    if shard.get_schema_file().is_some() {
                        return true;
                    }
                }
            }
            false
        },
        10,
        || "failed to build schema file".to_string(),
    );
    let mut client = cluster.new_client();
    let ctx = Mutex::new(EvalContext::default());
    let pd_client = cluster.get_pd_client();
    let regions_count = pd_client.get_regions_number();
    let split_key = gen_row_key(keyspace_id, table_id, 100);
    client.split(&split_key);
    cluster.wait_pd_region_count(regions_count + 1);

    client.put_kv(
        0..100,
        |i: usize| gen_row_key(keyspace_id, table_id, i),
        |i: usize| gen_row_val(&ctx, i),
    );
    client.put_kv(
        100..200,
        |i: usize| gen_row_key(keyspace_id, table_id, i),
        |i: usize| gen_row_val(&ctx, i),
    );
    must_wait(
        || {
            let all_id_vers = kvengine.get_all_shard_id_vers();
            let mut columnar_count = 0;
            for id_ver in all_id_vers {
                if let Ok(shard) = kvengine.get_shard_with_ver(id_ver.id, id_ver.ver) {
                    let snap_version = shard.get_snap_version();
                    let columnar_snap_version = shard.get_columnar_snap_version();
                    if snap_version == columnar_snap_version {
                        columnar_count += 1;
                        if columnar_count == 2 {
                            return true;
                        }
                    }
                }
            }
            false
        },
        10,
        || {
            // dump shard info
            let all_id_vers = kvengine.get_all_shard_id_vers();
            for id_ver in all_id_vers {
                if let Ok(shard) = kvengine.get_shard_with_ver(id_ver.id, id_ver.ver) {
                    let snap_version = shard.get_snap_version();
                    let columnar_snap_version = shard.get_columnar_snap_version();
                    info!(
                        "shard: {:?}, snap_version: {}, columnar_snap_version: {}",
                        id_ver, snap_version, columnar_snap_version
                    );
                }
            }
            "failed to build columnar file".to_string()
        },
    );
    client.try_merge(&gen_row_key(keyspace_id, table_id, 0), &split_key);
    cluster.wait_pd_region_count(regions_count);

    let shard_id = pd_client
        .get_region(&encode_bytes(&split_key))
        .unwrap()
        .get_id();
    let shard = kvengine.get_shard(shard_id).unwrap();
    let snap_access = shard.new_snap_access();
    let ts = client.get_ts().into_inner();
    let mut columnar_reader = snap_access
        .new_columnar_mvcc_reader(schema.table_id, &schema.columns, ts)
        .unwrap();
    columnar_reader.set_int_handle_range(0, Some(190)).unwrap();
    let mut block = columnar::Block::new(&schema);
    let read_rows = columnar_reader.read_block(&mut block, 200).unwrap();
    assert_eq!(read_rows, 190);
    for i in 0..read_rows {
        let handle = block.get_handle_buf().get_int_handle_value(i);
        assert_eq!(handle, i as i64);
        let columns = block.get_columns();
        assert_eq!(columns[0].get_not_null_value(i).get_i64_le(), i as i64);
        let str_val = gen_str_val(i);
        assert_eq!(columns[1].get_not_null_value(i), &str_val);
    }
}

fn gen_row_key(keyspace_id: u32, table_id: i64, i: usize) -> Vec<u8> {
    let mut key = ApiV2::get_txn_keyspace_prefix(keyspace_id);
    let table_key = encode_row_key(table_id, i as i64);
    key.extend_from_slice(&table_key);
    key
}

fn gen_row_val(ctx: &Mutex<EvalContext>, i: usize) -> Vec<u8> {
    let mut row_val = vec![];
    let str_val = gen_str_val(i);
    let cols = vec![
        Column::new(1, Some(i as i64)),
        Column::new(2, Some(str_val)),
    ];
    let mut guard = ctx.lock().unwrap();
    row_val.write_row(&mut guard, cols).unwrap();
    row_val
}

fn gen_str_val(i: usize) -> Vec<u8> {
    let repeat = 1 + i % 16;
    format!("abc_{}", i).repeat(repeat).into_bytes()
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
        let schema = SchemaBuf {
            table_id: columnar_table_id,
            handle_column: new_int_handle_column_info(),
            version_column: new_version_column_info(),
            txn_id_column: Some(new_txn_id_column_info()),
            columns: vec![c1, c2],
            pk_col_ids: vec![],
            vector_indexes: vec![],
        }
        .into();
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

async fn send_collect_columnar_status_request(
    status_addr: &str,
    keyspace_id: u32,
    table_id: i64,
) -> ColumnarStatusResp {
    let request = hyper::http::Request::builder()
        .method(http::method::Method::GET)
        .uri(format!(
            "http://{}/kvengine/columnar_status?keyspace_id={}&table_id={}",
            status_addr, keyspace_id, table_id
        ))
        .body(Body::empty())
        .unwrap();
    let http_client = hyper::client::Client::new();
    let resp = http_client.request(request).await.unwrap();
    assert!(resp.status().is_success());
    let mut body = vec![];
    resp.into_body()
        .try_for_each(|bytes| {
            body.extend(bytes);
            ok(())
        })
        .await
        .unwrap();
    let columnar_status: ColumnarStatusResp = serde_json::from_slice(&body).unwrap();
    columnar_status
}

async fn create_keyspace_and_split_tables(
    cluster: &mut ServerCluster,
    keyspace_id: u32,
) -> Vec<i64> {
    let km = cluster.keyspace_manager();
    km.create_single_keyspace(keyspace_id, format!("ks{}", keyspace_id), 4, 4, 0.0, false)
        .await;
    let keyspace_split_keys = get_keyspace_split_keys(keyspace_id);
    let pd_client = cluster.get_pd_client();
    pd_client
        .split_regions_with_retry(keyspace_split_keys, Duration::from_secs(10))
        .await
        .unwrap();
    let ks_meta = km.get_keyspace_meta(keyspace_id).unwrap();
    let table_ids = ks_meta.get_all_available_tables();
    let table_split_keys = get_table_split_keys(keyspace_id, &table_ids);
    pd_client
        .split_regions_with_retry(table_split_keys, Duration::from_secs(10))
        .await
        .unwrap();
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
