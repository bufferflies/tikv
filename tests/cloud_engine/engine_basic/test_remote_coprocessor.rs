// Copyright 2023 TiKV Project Authors. Licensed under Apache-2.0.

use std::{collections::BTreeMap, sync::Arc};

use cloud_encryption::MasterKey;
use codec::prelude::NumberEncoder;
use kvengine::{table::table::Row, SnapAccess};
use kvproto::{
    coprocessor::{self as coppb, Request},
    kvrpcpb::{ApiVersion, Context},
};
use protobuf::Message;
use rfstore::store::RegionSnapshot;
use test_cloud_server::{
    client::{ClusterClient, TxnMutations},
    util::Mutation,
    ServerCluster,
};
use test_coprocessor::{
    next_id, offset_for_column, Column, ColumnBuilder, DagChunkSpliter, DagSelect, Table,
    TableBuilder, TYPE_LONG, TYPE_VAR_CHAR,
};
use tidb_query_datatype::{
    codec::{datum, table, Datum},
    expr::EvalContext,
};
use tikv::coprocessor::{REQ_TYPE_ANALYZE, REQ_TYPE_CHECKSUM, REQ_TYPE_DAG};
use tikv_util::{config::ReadableSize, info, quota_limiter::QuotaLimiter};
use tipb::{Chunk, Executor, Expr, ExprType, ScalarFuncSig};

use crate::alloc_node_id;

#[cfg(test)]
const FLAG_IGNORE_TRUNCATE: u64 = 1;
#[cfg(test)]
const FLAG_TRUNCATE_AS_WARNING: u64 = 1 << 1;
#[cfg(test)]
const MAX_INSERT_BATCH_SIZE: i64 = 200;

// sort_by sorts the `$v`(a vector of `Vec<Datum>`) by the $index elements in
/// `Vec<Datum>`
#[cfg(test)]
macro_rules! sort_by {
    ($v:ident, $index:expr, $t:ident) => {
        $v.sort_by(|a, b| match (&a[$index], &b[$index]) {
            (Datum::Null, Datum::Null) => std::cmp::Ordering::Equal,
            (Datum::$t(a), Datum::$t(b)) => a.cmp(&b),
            (Datum::Null, _) => std::cmp::Ordering::Less,
            (_, Datum::Null) => std::cmp::Ordering::Greater,
            _ => unreachable!(),
        });
    };
}

// FIXME: The break points don't work and so the test fails.
#[cfg(test)]
pub struct ProductTable(test_coprocessor::Table);

#[cfg(test)]
impl ProductTable {
    pub fn new() -> ProductTable {
        let id = ColumnBuilder::new()
            .col_type(TYPE_LONG)
            .primary_key(true)
            .build();
        let idx_id = next_id();
        let name = ColumnBuilder::new()
            .col_type(TYPE_VAR_CHAR)
            .index_key(idx_id)
            .build();
        let count = ColumnBuilder::new()
            .col_type(TYPE_LONG)
            .index_key(idx_id)
            .build();
        let table = TableBuilder::new()
            .add_col("id", id)
            .add_col("name", name)
            .add_col("count", count)
            .build();
        info!("idx_id: {}", idx_id);
        ProductTable(table)
    }
}

#[cfg(test)]
impl Default for ProductTable {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
impl std::ops::Deref for ProductTable {
    type Target = Table;

    fn deref(&self) -> &Table {
        &self.0
    }
}

#[cfg(test)]
fn check_chunk_datum_count(chunks: &[Chunk], datum_limit: usize) {
    let mut iter = chunks.iter();
    let res = iter.any(|x| datum::decode(&mut x.get_rows_data()).unwrap().len() != datum_limit);
    if res {
        assert!(iter.next().is_none());
    }
}

#[test]
fn test_basic() {
    test_util::init_log_for_test();

    let product = ProductTable::new();
    let mut dag_test = DagTest::new(&product);

    dag_test.insert_rows(1002);

    let select_key_ranges = vec![dag_test.get_key_range(10, 15)];
    let snapshot = dag_test.fetch_snapshot(dag_test.get_ts().into_inner(), select_key_ranges);

    snapshot.print_to_info();
}

#[test]
fn test_analyze() {
    test_util::init_log_for_test();

    let product = ProductTable::new();
    let mut dag_test = DagTest::new(&product);

    dag_test.insert_rows(1001);

    let mut col_req = tipb::AnalyzeColumnsReq::default();
    col_req.set_columns_info(dag_test.table.columns_info().into());

    let mut analyze_req = tipb::AnalyzeReq::default();
    analyze_req.set_tp(tipb::AnalyzeType::TypeColumn);
    analyze_req.set_col_req(col_req);

    let mut req = coppb::Request::default();
    req.set_context(Default::default());
    req.set_start_ts(dag_test.get_ts().into_inner());

    let select_key_ranges = vec![dag_test.get_key_range(10, 20)];

    req.set_ranges(select_key_ranges.clone().into());
    req.set_tp(REQ_TYPE_ANALYZE);
    req.set_data(analyze_req.write_to_bytes().unwrap());

    let snapshot = dag_test.fetch_snapshot(dag_test.get_ts().into_inner(), select_key_ranges);
    let quota_limiter = Arc::new(QuotaLimiter::default());

    match dag_test.execute(snapshot, quota_limiter, req) {
        Ok(_) => info!("Analyze Ok."),
        Err(e) => panic!("Analyze failed: {:?}!", e),
    }
}

#[test]
fn test_checksum() {
    test_util::init_log_for_test();

    let product = ProductTable::new();
    let mut dag_test = DagTest::new(&product);

    dag_test.insert_rows(1000);

    let checksum = tipb::ChecksumRequest::default();
    let mut req = coppb::Request::default();
    req.set_context(Default::default());
    req.set_start_ts(dag_test.get_ts().into_inner());

    let select_key_ranges = vec![dag_test.get_key_range(10, 20)];

    req.set_ranges(select_key_ranges.clone().into());
    req.set_tp(REQ_TYPE_CHECKSUM);
    req.set_data(checksum.write_to_bytes().unwrap());

    let snapshot = dag_test.fetch_snapshot(dag_test.get_ts().into_inner(), select_key_ranges);
    let quota_limiter = Arc::new(QuotaLimiter::default());

    match dag_test.execute(snapshot, quota_limiter, req) {
        Ok(_) => info!("Checksum Ok."),
        Err(e) => panic!("Checksum failed: {:?}!", e),
    }
}

#[test]
fn test_stack_guard() {
    test_util::init_log_for_test();

    let rows = vec![
        (1, Some("v:0"), 1),
        (2, Some("v:4"), 1),
        (3, Some("v:5"), 1),
        (4, Some("v:3"), 1),
        (5, Some("v:1"), 1),
    ];

    let product = ProductTable::new();
    let mut dag_test = DagTest::new(&product);

    dag_test.insert_and_commit(&rows);

    let req = {
        let mut expr = Expr::default();
        for _ in 0..101 {
            let mut e = Expr::default();
            e.mut_children().push(expr);
            expr = e;
        }
        let mut e = Executor::default();
        e.mut_selection().mut_conditions().push(expr);
        let mut dag = tipb::DagRequest::default();
        dag.mut_executors().push(e);
        let mut req = coppb::Request::default();
        req.set_tp(REQ_TYPE_DAG);
        req.set_data(dag.write_to_bytes().unwrap());
        req
    };

    let select_key_range = dag_test.get_key_range_all();
    let snapshot = dag_test.fetch_snapshot(dag_test.get_ts().into_inner(), vec![select_key_range]);
    let quota_limiter = Arc::new(QuotaLimiter::default());
    match dag_test.execute(snapshot, quota_limiter, req) {
        Ok(_) => panic!("Should fail"),
        Err(e) => info!("E: {}", e),
    }
}

#[test]
fn test_select_all_scan() {
    test_util::init_log_for_test();

    let rows = vec![
        (1, Some("v:1"), 1),
        (2, Some("v:2"), 1),
        (3, Some("v:3"), 1),
        (4, Some("v:4"), 1),
        (5, Some("v:5"), 1),
        (6, Some("v:6"), 1),
    ];

    let product = ProductTable::new();
    let mut dag_test = DagTest::new(&product);

    dag_test.insert_and_commit(&rows);

    let select_key_range = dag_test.get_key_range_all();
    let snapshot = dag_test.fetch_snapshot(dag_test.get_ts().into_inner(), vec![select_key_range]);

    let req = {
        DagSelect::from(&product)
            .output_offsets(Some(vec![0, 1, 2]))
            .key_ranges(vec![dag_test.get_key_range_all()])
            .build_with_start_ts(dag_test.client.get_ts())
    };

    let quota_limiter = Arc::new(QuotaLimiter::default());
    quota_limiter.set_read_bandwidth_limit(ReadableSize::kb(1), true);

    let cop_resp = match dag_test.execute(snapshot, quota_limiter.clone(), req) {
        Ok(resp) => resp,
        Err(e) => panic!("{}", e),
    };

    let mut resp = tipb::SelectResponse::default();
    resp.merge_from_bytes(cop_resp.get_data()).unwrap();

    let mut total_chunk_size = 0;
    for chunk in resp.get_chunks() {
        total_chunk_size += chunk.get_rows_data().len();
    }

    let spliter = DagChunkSpliter::new(resp.take_chunks().into(), 3);
    for (row, (id, name, cnt)) in spliter.zip(rows) {
        let name_datum = name.map(|s| s.as_bytes()).into();
        let expected_encoded = datum::encode_value(
            &mut EvalContext::default(),
            &[Datum::I64(id), name_datum, cnt.into()],
        )
        .unwrap();
        let result_encoded = datum::encode_value(&mut EvalContext::default(), &row).unwrap();
        assert_eq!(result_encoded, &*expected_encoded);
    }
    assert_eq!(
        quota_limiter.total_read_bytes_consumed(true),
        total_chunk_size
    ); // the consume_sample is called due to read bytes quota
}

#[test]
fn test_batch_row_limit() {
    test_util::init_log_for_test();

    let rows = vec![
        (1, Some("name:0"), 2),
        (2, Some("name:4"), 3),
        (4, Some("name:3"), 1),
        (5, Some("name:1"), 4),
    ];

    let batch_row_limit = 3;
    let chunk_datum_limit = batch_row_limit * 3; // we have 3 fields.
    let product = ProductTable::new();
    let mut dag_test = DagTest::new(&product);

    dag_test.insert_and_commit(&rows);

    // let mut cfg = Config::default();
    // cfg.end_point_batch_row_limit = batch_row_limit;

    let req = DagSelect::from(&product).build();

    let mut resp = dag_test.select_all(req, None);

    check_chunk_datum_count(resp.get_chunks(), chunk_datum_limit);

    let splitter = DagChunkSpliter::new(resp.take_chunks().into(), 3);

    for (row, (id, name, count)) in splitter.zip(rows) {
        let name_datum = name.map(|s| s.as_bytes()).into();
        let expected_encoded = datum::encode_value(
            &mut EvalContext::default(),
            &[Datum::I64(id), name_datum, count.into()],
        )
        .unwrap();
        let result_encoded = datum::encode_value(&mut EvalContext::default(), &row).unwrap();
        assert_eq!(result_encoded, &*expected_encoded);
    }
}

#[test]
fn test_group_by() {
    test_util::init_log_for_test();

    let rows = vec![
        (1, Some("name:0"), 2),
        (2, Some("name:2"), 3),
        (4, Some("name:0"), 1),
        (5, Some("name:1"), 4),
    ];

    let product = ProductTable::new();
    let mut dag_test = DagTest::new(&product);

    dag_test.insert_and_commit(&rows);

    // for dag
    let req = DagSelect::from(&product)
        .group_by(&[&product["name"]])
        .output_offsets(Some(vec![0]))
        .build();

    let mut resp = dag_test.select_all(req, None);

    // should only have name:0, name:2 and name:1
    let mut row_count = 0;
    let spliter = DagChunkSpliter::new(resp.take_chunks().into(), 1);
    let mut results = spliter.collect::<Vec<Vec<Datum>>>();
    sort_by!(results, 0, Bytes);
    for (row, name) in results.iter().zip(&[b"name:0", b"name:1", b"name:2"]) {
        let expected_encoded =
            datum::encode_value(&mut EvalContext::default(), &[Datum::Bytes(name.to_vec())])
                .unwrap();
        let result_encoded = datum::encode_value(&mut EvalContext::default(), row).unwrap();
        assert_eq!(result_encoded, &*expected_encoded);
        row_count += 1;
    }
    assert_eq!(row_count, 3);
}

#[test]
fn test_aggr_count() {
    test_util::init_log_for_test();

    let rows = vec![
        (1, Some("name:0"), 2),
        (2, Some("name:3"), 3),
        (4, Some("name:0"), 1),
        (5, Some("name:5"), 4),
        (6, Some("name:5"), 4),
        (7, None, 4),
    ];

    let product = ProductTable::new();
    let mut dag_test = DagTest::new(&product);

    dag_test.insert_and_commit(&rows);

    let exp = vec![
        (Datum::Null, 1),
        (Datum::Bytes(b"name:0".to_vec()), 2),
        (Datum::Bytes(b"name:3".to_vec()), 1),
        (Datum::Bytes(b"name:5".to_vec()), 2),
    ];

    // for dag
    let req = DagSelect::from(&product)
        .count(&product["count"])
        .group_by(&[&product["name"]])
        .output_offsets(Some(vec![0, 1]))
        .build();

    let mut resp = dag_test.select_all(req, None);

    let mut row_count = 0;
    let exp_len = exp.len();
    let spliter = DagChunkSpliter::new(resp.take_chunks().into(), 2);
    let mut results = spliter.collect::<Vec<Vec<Datum>>>();
    sort_by!(results, 1, Bytes);
    for (row, (name, cnt)) in results.iter().zip(exp) {
        let expected_datum = vec![Datum::U64(cnt), name];
        let expected_encoded =
            datum::encode_value(&mut EvalContext::default(), &expected_datum).unwrap();
        let result_encoded = datum::encode_value(&mut EvalContext::default(), row).unwrap();
        assert_eq!(&*result_encoded, &*expected_encoded);
        row_count += 1;
    }
    assert_eq!(row_count, exp_len);

    let exp = vec![
        (vec![Datum::Null, Datum::I64(4)], 1),
        (vec![Datum::Bytes(b"name:0".to_vec()), Datum::I64(1)], 1),
        (vec![Datum::Bytes(b"name:0".to_vec()), Datum::I64(2)], 1),
        (vec![Datum::Bytes(b"name:3".to_vec()), Datum::I64(3)], 1),
        (vec![Datum::Bytes(b"name:5".to_vec()), Datum::I64(4)], 2),
    ];

    // for dag
    let req = DagSelect::from(&product)
        .count(&product["id"])
        .group_by(&[&product["name"], &product["count"]])
        .build();

    let mut resp = dag_test.select_all(req, None);

    let mut row_count = 0;
    let exp_len = exp.len();
    let spliter = DagChunkSpliter::new(resp.take_chunks().into(), 3);
    let mut results = spliter.collect::<Vec<Vec<Datum>>>();
    sort_by!(results, 1, Bytes);
    for (row, (gk_data, cnt)) in results.iter().zip(exp) {
        let mut expected_datum = vec![Datum::U64(cnt)];
        expected_datum.extend_from_slice(gk_data.as_slice());
        let expected_encoded =
            datum::encode_value(&mut EvalContext::default(), &expected_datum).unwrap();
        let result_encoded = datum::encode_value(&mut EvalContext::default(), row).unwrap();
        assert_eq!(&*result_encoded, &*expected_encoded);
        row_count += 1;
    }
    assert_eq!(row_count, exp_len);
}

#[test]
fn test_aggr_first() {
    test_util::init_log_for_test();

    let rows = vec![
        (1, Some("name:0"), 2),
        (2, Some("name:3"), 3),
        (3, Some("name:5"), 3),
        (4, Some("name:0"), 1),
        (5, Some("name:5"), 4),
        (6, Some("name:5"), 4),
        (7, None, 4),
        (8, None, 5),
        (9, Some("name:5"), 5),
        (10, None, 6),
    ];

    let product = ProductTable::new();
    let mut dag_test = DagTest::new(&product);

    dag_test.insert_and_commit(&rows);

    let exp = vec![
        (Datum::Null, 7),
        (Datum::Bytes(b"name:0".to_vec()), 1),
        (Datum::Bytes(b"name:3".to_vec()), 2),
        (Datum::Bytes(b"name:5".to_vec()), 3),
    ];

    // for dag
    let req = DagSelect::from(&product)
        .first(&product["id"])
        .group_by(&[&product["name"]])
        .output_offsets(Some(vec![0, 1]))
        .build();

    let mut resp = dag_test.select_all(req, None);

    let mut row_count = 0;
    let exp_len = exp.len();
    let spliter = DagChunkSpliter::new(resp.take_chunks().into(), 2);
    let mut results = spliter.collect::<Vec<Vec<Datum>>>();
    sort_by!(results, 1, Bytes);
    for (row, (name, id)) in results.iter().zip(exp) {
        let expected_datum = vec![Datum::I64(id), name];
        let expected_encoded =
            datum::encode_value(&mut EvalContext::default(), &expected_datum).unwrap();
        let result_encoded = datum::encode_value(&mut EvalContext::default(), row).unwrap();
        assert_eq!(&*result_encoded, &*expected_encoded);
        row_count += 1;
    }
    assert_eq!(row_count, exp_len);

    let exp = vec![
        (5, Datum::Null),
        (6, Datum::Null),
        (2, Datum::Bytes(b"name:0".to_vec())),
        (1, Datum::Bytes(b"name:0".to_vec())),
        (3, Datum::Bytes(b"name:3".to_vec())),
        (4, Datum::Bytes(b"name:5".to_vec())),
    ];

    // for dag
    let req = DagSelect::from(&product)
        .first(&product["name"])
        .group_by(&[&product["count"]])
        .output_offsets(Some(vec![0, 1]))
        .build();

    let mut resp = dag_test.select_all(req, None);

    let mut row_count = 0;
    let exp_len = exp.len();
    let spliter = DagChunkSpliter::new(resp.take_chunks().into(), 2);
    let mut results = spliter.collect::<Vec<Vec<Datum>>>();
    sort_by!(results, 0, Bytes);
    for (row, (count, name)) in results.iter().zip(exp) {
        let expected_datum = vec![name, Datum::I64(count)];
        let expected_encoded =
            datum::encode_value(&mut EvalContext::default(), &expected_datum).unwrap();
        let result_encoded = datum::encode_value(&mut EvalContext::default(), row).unwrap();
        assert_eq!(&*result_encoded, &*expected_encoded);
        row_count += 1;
    }
    assert_eq!(row_count, exp_len);
}

#[test]
fn test_aggr_avg() {
    test_util::init_log_for_test();

    let rows = vec![
        (1, Some("name:0"), 2),
        (2, Some("name:3"), 3),
        (4, Some("name:0"), 1),
        (5, Some("name:5"), 4),
        (6, Some("name:5"), 4),
        (7, None, 4),
    ];

    let product = ProductTable::new();
    let mut dag_test = DagTest::new(&product);

    dag_test.insert_and_commit(&rows);

    let mut row_cache = dag_test.get_row_cache();

    let mut container = BTreeMap::<i64, Datum>::new();
    row_cache
        .add_col(&mut container, &product["id"], Datum::I64(8))
        .add_col(
            &mut container,
            &product["name"],
            Datum::Bytes(b"name:4".to_vec()),
        )
        .add_col(&mut container, &product["count"], Datum::Null)
        .push(container);

    Txn::auto_commit(dag_test.get_inserter(), row_cache);

    let exp = vec![
        (Datum::Null, (Datum::Dec(4.into()), 1)),
        (Datum::Bytes(b"name:0".to_vec()), (Datum::Dec(3.into()), 2)),
        (Datum::Bytes(b"name:3".to_vec()), (Datum::Dec(3.into()), 1)),
        (Datum::Bytes(b"name:4".to_vec()), (Datum::Null, 0)),
        (Datum::Bytes(b"name:5".to_vec()), (Datum::Dec(8.into()), 2)),
    ];

    // for dag
    let req = DagSelect::from(&product)
        .avg(&product["count"])
        .group_by(&[&product["name"]])
        .build();

    let mut resp = dag_test.select_all(req, None);

    let mut row_count = 0;
    let exp_len = exp.len();
    let spliter = DagChunkSpliter::new(resp.take_chunks().into(), 3);
    let mut results = spliter.collect::<Vec<Vec<Datum>>>();
    sort_by!(results, 2, Bytes);
    for (row, (name, (sum, cnt))) in results.iter().zip(exp) {
        let expected_datum = vec![Datum::U64(cnt), sum, name];
        let expected_encoded =
            datum::encode_value(&mut EvalContext::default(), &expected_datum).unwrap();
        let result_encoded = datum::encode_value(&mut EvalContext::default(), row).unwrap();
        assert_eq!(&*result_encoded, &*expected_encoded);
        row_count += 1;
    }
    assert_eq!(row_count, exp_len);
}

#[test]
fn test_aggr_sum() {
    test_util::init_log_for_test();

    let rows = vec![
        (1, Some("name:0"), 2),
        (2, Some("name:3"), 3),
        (4, Some("name:0"), 1),
        (5, Some("name:5"), 4),
        (6, Some("name:5"), 4),
        (7, None, 4),
    ];

    let product = ProductTable::new();
    let mut dag_test = DagTest::new(&product);

    dag_test.insert_and_commit(&rows);

    let exp = vec![
        (Datum::Null, 4),
        (Datum::Bytes(b"name:0".to_vec()), 3),
        (Datum::Bytes(b"name:3".to_vec()), 3),
        (Datum::Bytes(b"name:5".to_vec()), 8),
    ];
    // for dag
    let req = DagSelect::from(&product)
        .sum(&product["count"])
        .group_by(&[&product["name"]])
        .output_offsets(Some(vec![0, 1]))
        .build();
    let mut resp = dag_test.select_all(req, None);
    let mut row_count = 0;
    let exp_len = exp.len();
    let spliter = DagChunkSpliter::new(resp.take_chunks().into(), 2);
    let mut results = spliter.collect::<Vec<Vec<Datum>>>();
    sort_by!(results, 1, Bytes);
    for (row, (name, cnt)) in results.iter().zip(exp) {
        let expected_datum = vec![Datum::Dec(cnt.into()), name];
        let expected_encoded =
            datum::encode_value(&mut EvalContext::default(), &expected_datum).unwrap();
        let result_encoded = datum::encode_value(&mut EvalContext::default(), row).unwrap();
        assert_eq!(&*result_encoded, &*expected_encoded);
        row_count += 1;
    }
    assert_eq!(row_count, exp_len);
}

#[test]
fn test_aggr_extra() {
    test_util::init_log_for_test();

    let rows = vec![
        (1, Some("name:0"), 2),
        (2, Some("name:3"), 3),
        (4, Some("name:0"), 1),
        (5, Some("name:5"), 4),
        (6, Some("name:5"), 5),
        (7, None, 4),
    ];

    let product = ProductTable::new();
    let mut dag_test = DagTest::new(&product);

    dag_test.insert_and_commit(&rows);

    let mut txn = Txn::new();
    let mut row_cache = dag_test.get_row_cache();

    txn.begin(dag_test.get_ts());

    for &(id, name) in &[(8, b"name:5"), (9, b"name:6")] {
        let mut container = row_cache.container();

        row_cache
            .add_col(&mut container, &product["id"], Datum::I64(id))
            .add_col(
                &mut container,
                &product["name"],
                Datum::Bytes(name.to_vec()),
            )
            .add_col(&mut container, &product["count"], Datum::Null)
            .push(container);
    }

    txn.prewrite(dag_test.get_inserter(), row_cache);
    txn.commit();

    let exp = vec![
        (Datum::Null, Datum::I64(4), Datum::I64(4)),
        (
            Datum::Bytes(b"name:0".to_vec()),
            Datum::I64(2),
            Datum::I64(1),
        ),
        (
            Datum::Bytes(b"name:3".to_vec()),
            Datum::I64(3),
            Datum::I64(3),
        ),
        (
            Datum::Bytes(b"name:5".to_vec()),
            Datum::I64(5),
            Datum::I64(4),
        ),
        (Datum::Bytes(b"name:6".to_vec()), Datum::Null, Datum::Null),
    ];

    // for dag
    let req = DagSelect::from(&product)
        .max(&product["count"])
        .min(&product["count"])
        .group_by(&[&product["name"]])
        .build();

    let mut resp = dag_test.select_all(req, None);

    let mut row_count = 0;
    let exp_len = exp.len();
    let spliter = DagChunkSpliter::new(resp.take_chunks().into(), 3);
    let mut results = spliter.collect::<Vec<Vec<Datum>>>();
    sort_by!(results, 2, Bytes);
    for (row, (name, max, min)) in results.iter().zip(exp) {
        let expected_datum = vec![max, min, name];
        let expected_encoded =
            datum::encode_value(&mut EvalContext::default(), &expected_datum).unwrap();
        let result_encoded = datum::encode_value(&mut EvalContext::default(), row).unwrap();
        assert_eq!(result_encoded, &*expected_encoded);
        row_count += 1;
    }
    assert_eq!(row_count, exp_len);
}

#[test]
fn test_aggr_bit_ops() {
    test_util::init_log_for_test();

    let rows = vec![
        (1, Some("name:0"), 2),
        (2, Some("name:3"), 3),
        (4, Some("name:0"), 1),
        (5, Some("name:5"), 4),
        (6, Some("name:5"), 5),
        (7, None, 4),
    ];

    let product = ProductTable::new();
    let mut dag_test = DagTest::new(&product);

    dag_test.insert_and_commit(&rows);

    let mut txn = Txn::new();
    let mut row_cache = dag_test.get_row_cache();

    txn.begin(dag_test.get_ts());

    for &(id, name) in &[(8, b"name:5"), (9, b"name:6")] {
        let mut container = row_cache.container();

        row_cache
            .add_col(&mut container, &product["id"], Datum::I64(id))
            .add_col(
                &mut container,
                &product["name"],
                Datum::Bytes(name.to_vec()),
            )
            .add_col(&mut container, &product["count"], Datum::Null)
            .push(container);
    }

    txn.prewrite(dag_test.get_inserter(), row_cache);
    txn.commit();

    let exp = vec![
        (Datum::Null, Datum::I64(4), Datum::I64(4), Datum::I64(4)),
        (
            Datum::Bytes(b"name:0".to_vec()),
            Datum::I64(0),
            Datum::I64(3),
            Datum::I64(3),
        ),
        (
            Datum::Bytes(b"name:3".to_vec()),
            Datum::I64(3),
            Datum::I64(3),
            Datum::I64(3),
        ),
        (
            Datum::Bytes(b"name:5".to_vec()),
            Datum::I64(4),
            Datum::I64(5),
            Datum::I64(1),
        ),
        (
            Datum::Bytes(b"name:6".to_vec()),
            Datum::I64(-1),
            Datum::I64(0),
            Datum::I64(0),
        ),
    ];

    // for dag
    let req = DagSelect::from(&product)
        .bit_and(&product["count"])
        .bit_or(&product["count"])
        .bit_xor(&product["count"])
        .group_by(&[&product["name"]])
        .output_offsets(Some(vec![0, 1, 2, 3]))
        .build();
    let mut resp = dag_test.select_all(req, None);
    let mut row_count = 0;
    let exp_len = exp.len();
    let spliter = DagChunkSpliter::new(resp.take_chunks().into(), 4);
    let mut results = spliter.collect::<Vec<Vec<Datum>>>();
    sort_by!(results, 3, Bytes);
    for (row, (name, bitand, bitor, bitxor)) in results.iter().zip(exp) {
        let expected_datum = vec![bitand, bitor, bitxor, name];
        let expected_encoded =
            datum::encode_value(&mut EvalContext::default(), &expected_datum).unwrap();
        let result_encoded = datum::encode_value(&mut EvalContext::default(), row).unwrap();
        assert_eq!(result_encoded, &*expected_encoded);
        row_count += 1;
    }
    assert_eq!(row_count, exp_len);
}

#[test]
fn test_order_by_column() {
    test_util::init_log_for_test();

    let rows = vec![
        (1, Some("name:0"), 2),
        (2, Some("name:3"), 3),
        (4, Some("name:0"), 1),
        (5, Some("name:6"), 4),
        (6, Some("name:5"), 4),
        (7, Some("name:4"), 4),
        (8, None, 4),
    ];

    let product = ProductTable::new();
    let mut dag_test = DagTest::new(&product);

    dag_test.insert_and_commit(&rows);

    let exp = vec![
        (8, None, 4),
        (7, Some("name:4"), 4),
        (6, Some("name:5"), 4),
        (5, Some("name:6"), 4),
        (2, Some("name:3"), 3),
    ];

    // for dag
    let req = DagSelect::from(&product)
        .order_by(&product["count"], true)
        .order_by(&product["name"], false)
        .limit(5)
        .build();
    let mut resp = dag_test.select_all(req, None);
    let mut row_count = 0;
    let spliter = DagChunkSpliter::new(resp.take_chunks().into(), 3);
    for (row, (id, name, cnt)) in spliter.zip(exp) {
        let name_datum = name.map(|s| s.as_bytes()).into();
        let expected_encoded = datum::encode_value(
            &mut EvalContext::default(),
            &[i64::from(id).into(), name_datum, i64::from(cnt).into()],
        )
        .unwrap();
        let result_encoded = datum::encode_value(&mut EvalContext::default(), &row).unwrap();
        assert_eq!(&*result_encoded, &*expected_encoded);
        row_count += 1;
    }
    assert_eq!(row_count, 5);
}

#[test]
fn test_limit() {
    test_util::init_log_for_test();

    let mut rows = vec![
        (1, Some("name:0"), 2),
        (2, Some("name:3"), 3),
        (4, Some("name:0"), 1),
        (5, Some("name:5"), 4),
        (6, Some("name:5"), 4),
        (7, None, 4),
    ];

    let product = ProductTable::new();
    let mut dag_test = DagTest::new(&product);

    dag_test.insert_and_commit(&rows);

    let expect: Vec<_> = rows.drain(..5).collect();
    // for dag
    let req = DagSelect::from(&product).limit(5).build();
    let mut resp = dag_test.select_all(req, None);
    let mut row_count = 0;
    let spliter = DagChunkSpliter::new(resp.take_chunks().into(), 3);
    for (row, (id, name, cnt)) in spliter.zip(expect) {
        let name_datum = name.map(|s| s.as_bytes()).into();
        let expected_encoded = datum::encode_value(
            &mut EvalContext::default(),
            &[id.into(), name_datum, cnt.into()],
        )
        .unwrap();
        let result_encoded = datum::encode_value(&mut EvalContext::default(), &row).unwrap();
        assert_eq!(&*result_encoded, &*expected_encoded);
        row_count += 1;
    }
    assert_eq!(row_count, 5);
}

#[test]
fn test_reverse() {
    test_util::init_log_for_test();

    let mut rows = vec![
        (1, Some("name:0"), 2),
        (2, Some("name:3"), 3),
        (4, Some("name:0"), 1),
        (5, Some("name:5"), 4),
        (6, Some("name:5"), 4),
        (7, None, 4),
    ];

    let product = ProductTable::new();
    let mut dag_test = DagTest::new(&product);
    dag_test.insert_and_commit(&rows);

    rows.reverse();

    let expect: Vec<_> = rows.drain(..5).collect();

    // for dag
    let req = DagSelect::from(&product)
        .limit(5)
        .order_by(&product["id"], true)
        .build();
    let mut resp = dag_test.select_all(req, None);
    let mut row_count = 0;
    let spliter = DagChunkSpliter::new(resp.take_chunks().into(), 3);
    for (row, (id, name, cnt)) in spliter.zip(expect) {
        let name_datum = name.map(|s| s.as_bytes()).into();
        let expected_encoded = datum::encode_value(
            &mut EvalContext::default(),
            &[id.into(), name_datum, cnt.into()],
        )
        .unwrap();
        let result_encoded = datum::encode_value(&mut EvalContext::default(), &row).unwrap();
        assert_eq!(&*result_encoded, &*expected_encoded);
        row_count += 1;
    }
    assert_eq!(row_count, 5);
}

#[test]
fn test_limit_oom() {
    test_util::init_log_for_test();

    let rows = vec![
        (1, Some("name:0"), 2),
        (2, Some("name:3"), 3),
        (4, Some("name:0"), 1),
        (5, Some("name:5"), 4),
        (6, Some("name:5"), 4),
        (7, None, 4),
    ];

    let product = ProductTable::new();
    let mut dag_test = DagTest::new(&product);
    dag_test.insert_and_commit(&rows);

    // for dag
    let req = DagSelect::from_index(&product, &product["id"])
        .limit(100000000)
        .build();
    let mut resp = dag_test.select_all(req, Some(product["id"].index));
    let mut row_count = 0;
    let spliter = DagChunkSpliter::new(resp.take_chunks().into(), 1);
    for (row, (id, ..)) in spliter.zip(rows) {
        let expected_encoded =
            datum::encode_value(&mut EvalContext::default(), &[id.into()]).unwrap();
        let result_encoded = datum::encode_value(&mut EvalContext::default(), &row).unwrap();
        assert_eq!(&*result_encoded, &*expected_encoded);
        row_count += 1;
    }
    assert_eq!(row_count, 6);
}

#[test]
fn test_order_by_pk_with_select_from_index() {
    test_util::init_log_for_test();

    let mut rows = vec![
        (8, Some("name:0"), 2),
        (7, Some("name:3"), 3),
        (6, Some("name:0"), 1),
        (5, Some("name:6"), 4),
        (4, Some("name:5"), 4),
        (3, Some("name:4"), 4),
        (2, None, 4),
    ];

    let product = ProductTable::new();
    let mut dag_test = DagTest::new(&product);

    dag_test.insert_and_commit(&rows);

    let expect: Vec<_> = rows.drain(..5).collect();

    // for dag
    let req = DagSelect::from_index(&product, &product["name"])
        .order_by(&product["id"], true)
        .limit(5)
        .build();
    let mut resp = dag_test.select_all(req, Some(product["name"].index));
    let mut row_count = 0;
    let spliter = DagChunkSpliter::new(resp.take_chunks().into(), 3);
    for (row, (id, name, cnt)) in spliter.zip(expect) {
        let name_datum = name.map(|s| s.as_bytes()).into();
        let expected_encoded = datum::encode_value(
            &mut EvalContext::default(),
            &[name_datum, cnt.into(), id.into()],
        )
        .unwrap();
        let result_encoded = datum::encode_value(&mut EvalContext::default(), &row).unwrap();
        assert_eq!(&*result_encoded, &*expected_encoded);
        row_count += 1;
    }
    assert_eq!(row_count, 5);
}

#[test]
fn test_index() {
    test_util::init_log_for_test();

    let rows = vec![
        (1, Some("name:0"), 2),
        (2, Some("name:3"), 3),
        (4, Some("name:0"), 1),
        (5, Some("name:5"), 4),
        (6, Some("name:5"), 4),
        (7, None, 4),
    ];

    let product = ProductTable::new();
    let mut dag_test = DagTest::new(&product);
    dag_test.insert_and_commit(&rows);

    // for dag
    let req = DagSelect::from_index(&product, &product["id"]).build();
    let mut resp = dag_test.select_all(req, Some(product["id"].index));
    let mut row_count = 0;
    let spliter = DagChunkSpliter::new(resp.take_chunks().into(), 1);
    for (row, (id, ..)) in spliter.zip(rows) {
        let expected_encoded =
            datum::encode_value(&mut EvalContext::default(), &[id.into()]).unwrap();
        let result_encoded = datum::encode_value(&mut EvalContext::default(), &row).unwrap();
        assert_eq!(&*result_encoded, &*expected_encoded);
        row_count += 1;
    }
    assert_eq!(row_count, 6);
}

#[test]
fn test_index_reverse_limit() {
    test_util::init_log_for_test();

    let mut rows = vec![
        (1, Some("name:0"), 2),
        (2, Some("name:3"), 3),
        (4, Some("name:0"), 1),
        (5, Some("name:5"), 4),
        (6, Some("name:5"), 4),
        (7, None, 4),
    ];

    let product = ProductTable::new();
    let mut dag_test = DagTest::new(&product);
    dag_test.insert_and_commit(&rows);

    rows.reverse();
    let expect: Vec<_> = rows.drain(..5).collect();

    // for dag
    let req = DagSelect::from_index(&product, &product["id"])
        .limit(5)
        .order_by(&product["id"], true)
        .build();

    let mut resp = dag_test.select_all(req, Some(product["id"].index));
    let mut row_count = 0;
    let spliter = DagChunkSpliter::new(resp.take_chunks().into(), 1);
    for (row, (id, ..)) in spliter.zip(expect) {
        let expected_encoded =
            datum::encode_value(&mut EvalContext::default(), &[id.into()]).unwrap();
        let result_encoded = datum::encode_value(&mut EvalContext::default(), &row).unwrap();
        assert_eq!(&*result_encoded, &*expected_encoded);
        row_count += 1;
    }
    assert_eq!(row_count, 5);
}

#[test]
fn test_index_group_by() {
    test_util::init_log_for_test();

    let rows = vec![
        (1, Some("name:0"), 2),
        (2, Some("name:2"), 3),
        (4, Some("name:0"), 1),
        (5, Some("name:1"), 4),
    ];

    let product = ProductTable::new();
    let mut dag_test = DagTest::new(&product);
    dag_test.insert_and_commit(&rows);

    // for dag
    let req = DagSelect::from_index(&product, &product["name"])
        .group_by(&[&product["name"]])
        .output_offsets(Some(vec![0]))
        .build();

    let mut resp = dag_test.select_all(req, Some(product["name"].index));

    // should only have name:0, name:2 and name:1
    let mut row_count = 0;
    let spliter = DagChunkSpliter::new(resp.take_chunks().into(), 1);
    let mut results = spliter.collect::<Vec<Vec<Datum>>>();
    sort_by!(results, 0, Bytes);
    for (row, name) in results.iter().zip(&[b"name:0", b"name:1", b"name:2"]) {
        let expected_encoded =
            datum::encode_value(&mut EvalContext::default(), &[Datum::Bytes(name.to_vec())])
                .unwrap();
        let result_encoded = datum::encode_value(&mut EvalContext::default(), row).unwrap();
        assert_eq!(&*result_encoded, &*expected_encoded);
        row_count += 1;
    }
    assert_eq!(row_count, 3);
}

#[test]
fn test_index_aggr_count() {
    test_util::init_log_for_test();

    let rows = vec![
        (1, Some("name:0"), 2),
        (2, Some("name:3"), 3),
        (4, Some("name:0"), 1),
        (5, Some("name:5"), 4),
        (6, Some("name:5"), 4),
        (7, None, 4),
    ];

    let product = ProductTable::new();
    let mut dag_test = DagTest::new(&product);

    dag_test.insert_and_commit(&rows);

    // for dag
    let req = DagSelect::from_index(&product, &product["name"])
        .count(&product["id"])
        .output_offsets(Some(vec![0]))
        .build();
    let mut resp = dag_test.select_all(req, Some(product["name"].index));
    let mut spliter = DagChunkSpliter::new(resp.take_chunks().into(), 1);
    let expected_encoded = datum::encode_value(
        &mut EvalContext::default(),
        &[Datum::U64(rows.len() as u64)],
    )
    .unwrap();
    let ret_data = spliter.next();
    assert_eq!(ret_data.is_some(), true);
    let result_encoded =
        datum::encode_value(&mut EvalContext::default(), &ret_data.unwrap()).unwrap();
    assert_eq!(&*result_encoded, &*expected_encoded);
    assert_eq!(spliter.next().is_none(), true);

    let exp = vec![
        (Datum::Null, 1),
        (Datum::Bytes(b"name:0".to_vec()), 2),
        (Datum::Bytes(b"name:3".to_vec()), 1),
        (Datum::Bytes(b"name:5".to_vec()), 2),
    ];
    // for dag
    let req = DagSelect::from_index(&product, &product["name"])
        .count(&product["id"])
        .group_by(&[&product["name"]])
        .output_offsets(Some(vec![0, 1]))
        .build();
    resp = dag_test.select_all(req, Some(product["name"].index));
    let mut row_count = 0;
    let exp_len = exp.len();
    let spliter = DagChunkSpliter::new(resp.take_chunks().into(), 2);
    let mut results = spliter.collect::<Vec<Vec<Datum>>>();
    sort_by!(results, 1, Bytes);
    for (row, (name, cnt)) in results.iter().zip(exp) {
        let expected_datum = vec![Datum::U64(cnt), name];
        let expected_encoded =
            datum::encode_value(&mut EvalContext::default(), &expected_datum).unwrap();
        let result_encoded = datum::encode_value(&mut EvalContext::default(), row).unwrap();
        assert_eq!(&*result_encoded, &*expected_encoded);
        row_count += 1;
    }
    assert_eq!(row_count, exp_len);

    let exp = vec![
        (vec![Datum::Null, Datum::I64(4)], 1),
        (vec![Datum::Bytes(b"name:0".to_vec()), Datum::I64(1)], 1),
        (vec![Datum::Bytes(b"name:0".to_vec()), Datum::I64(2)], 1),
        (vec![Datum::Bytes(b"name:3".to_vec()), Datum::I64(3)], 1),
        (vec![Datum::Bytes(b"name:5".to_vec()), Datum::I64(4)], 2),
    ];
    let req = DagSelect::from_index(&product, &product["name"])
        .count(&product["id"])
        .group_by(&[&product["name"], &product["count"]])
        .build();
    resp = dag_test.select_all(req, Some(product["name"].index));
    let mut row_count = 0;
    let exp_len = exp.len();
    let spliter = DagChunkSpliter::new(resp.take_chunks().into(), 3);
    let mut results = spliter.collect::<Vec<Vec<Datum>>>();
    sort_by!(results, 1, Bytes);
    for (row, (gk_data, cnt)) in results.iter().zip(exp) {
        let mut expected_datum = vec![Datum::U64(cnt)];
        expected_datum.extend_from_slice(gk_data.as_slice());
        let expected_encoded =
            datum::encode_value(&mut EvalContext::default(), &expected_datum).unwrap();
        let result_encoded = datum::encode_value(&mut EvalContext::default(), row).unwrap();
        assert_eq!(&*result_encoded, &*expected_encoded);
        row_count += 1;
    }
    assert_eq!(row_count, exp_len);
}

#[test]
fn test_index_aggr_first() {
    test_util::init_log_for_test();

    let rows = vec![
        (1, Some("name:0"), 2),
        (2, Some("name:3"), 3),
        (4, Some("name:0"), 1),
        (5, Some("name:5"), 4),
        (6, Some("name:5"), 4),
        (7, None, 4),
    ];

    let product = ProductTable::new();
    let mut dag_test = DagTest::new(&product);

    dag_test.insert_and_commit(&rows);

    let exp = vec![
        (Datum::Null, 7),
        (Datum::Bytes(b"name:0".to_vec()), 4),
        (Datum::Bytes(b"name:3".to_vec()), 2),
        (Datum::Bytes(b"name:5".to_vec()), 5),
    ];
    // for dag
    let req = DagSelect::from_index(&product, &product["name"])
        .first(&product["id"])
        .group_by(&[&product["name"]])
        .output_offsets(Some(vec![0, 1]))
        .build();
    let mut resp = dag_test.select_all(req, Some(product["name"].index));
    let mut row_count = 0;
    let exp_len = exp.len();
    let spliter = DagChunkSpliter::new(resp.take_chunks().into(), 2);
    let mut results = spliter.collect::<Vec<Vec<Datum>>>();
    sort_by!(results, 1, Bytes);
    for (row, (name, id)) in results.iter().zip(exp) {
        let expected_datum = vec![Datum::I64(id), name];
        let expected_encoded =
            datum::encode_value(&mut EvalContext::default(), &expected_datum).unwrap();
        let result_encoded = datum::encode_value(&mut EvalContext::default(), row).unwrap();

        assert_eq!(
            &*result_encoded, &*expected_encoded,
            "exp: {:?}, got: {:?}",
            expected_datum, row
        );
        assert_eq!(&*result_encoded, &*expected_encoded);
        row_count += 1;
    }
    assert_eq!(row_count, exp_len);
}

#[test]
fn test_index_aggr_avg() {
    test_util::init_log_for_test();

    let rows = vec![
        (1, Some("name:0"), 2),
        (2, Some("name:3"), 3),
        (4, Some("name:0"), 1),
        (5, Some("name:5"), 4),
        (6, Some("name:5"), 4),
        (7, None, 4),
    ];

    let product = ProductTable::new();
    let mut dag_test = DagTest::new(&product);

    dag_test.insert_and_commit(&rows);

    let mut txn = Txn::new();
    let mut row_cache = dag_test.get_row_cache();
    let mut container = row_cache.container();

    txn.begin(dag_test.get_ts());

    row_cache
        .add_col(&mut container, &product["id"], Datum::I64(8))
        .add_col(
            &mut container,
            &product["name"],
            Datum::Bytes(b"name:4".to_vec()),
        )
        .add_col(&mut container, &product["count"], Datum::Null)
        .push(container);

    txn.prewrite(dag_test.get_inserter(), row_cache);
    txn.commit();

    let exp = vec![
        (Datum::Null, (Datum::Dec(4.into()), 1)),
        (Datum::Bytes(b"name:0".to_vec()), (Datum::Dec(3.into()), 2)),
        (Datum::Bytes(b"name:3".to_vec()), (Datum::Dec(3.into()), 1)),
        (Datum::Bytes(b"name:4".to_vec()), (Datum::Null, 0)),
        (Datum::Bytes(b"name:5".to_vec()), (Datum::Dec(8.into()), 2)),
    ];
    // for dag
    let req = DagSelect::from_index(&product, &product["name"])
        .avg(&product["count"])
        .group_by(&[&product["name"]])
        .build();
    let mut resp = dag_test.select_all(req, Some(product["name"].index));
    let mut row_count = 0;
    let exp_len = exp.len();
    let spliter = DagChunkSpliter::new(resp.take_chunks().into(), 3);
    let mut results = spliter.collect::<Vec<Vec<Datum>>>();
    sort_by!(results, 2, Bytes);
    for (row, (name, (sum, cnt))) in results.iter().zip(exp) {
        let expected_datum = vec![Datum::U64(cnt), sum, name];
        let expected_encoded =
            datum::encode_value(&mut EvalContext::default(), &expected_datum).unwrap();
        let result_encoded = datum::encode_value(&mut EvalContext::default(), row).unwrap();
        assert_eq!(&*result_encoded, &*expected_encoded);
        row_count += 1;
    }
    assert_eq!(row_count, exp_len);
}

#[test]
fn test_index_aggr_sum() {
    test_util::init_log_for_test();

    let rows = vec![
        (1, Some("name:0"), 2),
        (2, Some("name:3"), 3),
        (4, Some("name:0"), 1),
        (5, Some("name:5"), 4),
        (6, Some("name:5"), 4),
        (7, None, 4),
    ];

    let product = ProductTable::new();
    let mut dag_test = DagTest::new(&product);

    dag_test.insert_and_commit(&rows);

    let exp = vec![
        (Datum::Null, 4),
        (Datum::Bytes(b"name:0".to_vec()), 3),
        (Datum::Bytes(b"name:3".to_vec()), 3),
        (Datum::Bytes(b"name:5".to_vec()), 8),
    ];
    // for dag
    let req = DagSelect::from_index(&product, &product["name"])
        .sum(&product["count"])
        .group_by(&[&product["name"]])
        .output_offsets(Some(vec![0, 1]))
        .build();
    let mut resp = dag_test.select_all(req, Some(product["name"].index));
    let mut row_count = 0;
    let exp_len = exp.len();
    let spliter = DagChunkSpliter::new(resp.take_chunks().into(), 2);
    let mut results = spliter.collect::<Vec<Vec<Datum>>>();
    sort_by!(results, 1, Bytes);
    for (row, (name, cnt)) in results.iter().zip(exp) {
        let expected_datum = vec![Datum::Dec(cnt.into()), name];
        let expected_encoded =
            datum::encode_value(&mut EvalContext::default(), &expected_datum).unwrap();
        let result_encoded = datum::encode_value(&mut EvalContext::default(), row).unwrap();
        assert_eq!(&*result_encoded, &*expected_encoded);
        row_count += 1;
    }
    assert_eq!(row_count, exp_len);
}

#[test]
fn test_index_aggr_extre() {
    let rows = vec![
        (1, Some("name:0"), 2),
        (2, Some("name:3"), 3),
        (4, Some("name:0"), 1),
        (5, Some("name:5"), 4),
        (6, Some("name:5"), 5),
        (7, None, 4),
    ];

    let product = ProductTable::new();
    let mut dag_test = init_with_data(&product, &rows);

    let mut txn = Txn::new();
    let mut row_cache = dag_test.get_row_cache();

    txn.begin(dag_test.get_ts());

    for &(id, name) in &[(8, b"name:5"), (9, b"name:6")] {
        let mut container = row_cache.container();

        row_cache
            .add_col(&mut container, &product["id"], Datum::I64(id))
            .add_col(
                &mut container,
                &product["name"],
                Datum::Bytes(name.to_vec()),
            )
            .add_col(&mut container, &product["count"], Datum::Null)
            .push(container);
    }

    txn.prewrite(dag_test.get_inserter(), row_cache);
    txn.commit();

    let exp = vec![
        (Datum::Null, Datum::I64(4), Datum::I64(4)),
        (
            Datum::Bytes(b"name:0".to_vec()),
            Datum::I64(2),
            Datum::I64(1),
        ),
        (
            Datum::Bytes(b"name:3".to_vec()),
            Datum::I64(3),
            Datum::I64(3),
        ),
        (
            Datum::Bytes(b"name:5".to_vec()),
            Datum::I64(5),
            Datum::I64(4),
        ),
        (Datum::Bytes(b"name:6".to_vec()), Datum::Null, Datum::Null),
    ];
    // for dag
    let req = DagSelect::from_index(&product, &product["name"])
        .max(&product["count"])
        .min(&product["count"])
        .group_by(&[&product["name"]])
        .build();
    let mut resp = handle_select(&mut dag_test, req, Some(product["name"].index));
    let mut row_count = 0;
    let exp_len = exp.len();
    let spliter = DagChunkSpliter::new(resp.take_chunks().into(), 3);
    let mut results = spliter.collect::<Vec<Vec<Datum>>>();
    sort_by!(results, 2, Bytes);
    for (row, (name, max, min)) in results.iter().zip(exp) {
        let expected_datum = vec![max, min, name];
        let expected_encoded =
            datum::encode_value(&mut EvalContext::default(), &expected_datum).unwrap();
        let result_encoded = datum::encode_value(&mut EvalContext::default(), row).unwrap();
        assert_eq!(&*result_encoded, &*expected_encoded);
        row_count += 1;
    }
    assert_eq!(row_count, exp_len);
}

#[test]
fn test_where() {
    use tidb_query_datatype::{FieldTypeAccessor, FieldTypeTp};

    let rows = vec![
        (1, Some("name:0"), 2),
        (2, Some("name:4"), 3),
        (4, Some("name:3"), 1),
        (5, Some("name:1"), 4),
    ];

    let product = ProductTable::new();
    let mut endpoint = init_with_data(&product, &rows);
    let cols = product.columns_info();
    let cond = {
        let mut col = Expr::default();
        col.set_tp(tipb::ExprType::ColumnRef);
        let count_offset = test_coprocessor::offset_for_column(&cols, product["count"].id);
        col.mut_val().write_i64(count_offset).unwrap();
        col.mut_field_type()
            .as_mut_accessor()
            .set_tp(FieldTypeTp::LongLong);

        let mut value = Expr::default();
        value.set_tp(tipb::ExprType::String);
        value.set_val(String::from("2").into_bytes());
        value
            .mut_field_type()
            .as_mut_accessor()
            .set_tp(FieldTypeTp::VarString);

        let mut right = Expr::default();
        right.set_tp(tipb::ExprType::ScalarFunc);
        right.set_sig(tipb::ScalarFuncSig::CastStringAsInt);
        right
            .mut_field_type()
            .as_mut_accessor()
            .set_tp(FieldTypeTp::LongLong);
        right.mut_children().push(value);

        let mut cond = Expr::default();
        cond.set_tp(tipb::ExprType::ScalarFunc);
        cond.set_sig(tipb::ScalarFuncSig::LtInt);
        cond.mut_field_type()
            .as_mut_accessor()
            .set_tp(FieldTypeTp::LongLong);
        cond.mut_children().push(col);
        cond.mut_children().push(right);
        cond
    };

    let req = DagSelect::from(&product).where_expr(cond).build();
    let mut resp = handle_select(&mut endpoint, req, None);
    let mut spliter = DagChunkSpliter::new(resp.take_chunks().into(), 3);
    let row = spliter.next().unwrap();
    let (id, name, cnt) = rows[2];
    let name_datum = name.map(|s| s.as_bytes()).into();
    let expected_encoded = datum::encode_value(
        &mut EvalContext::default(),
        &[Datum::I64(id), name_datum, cnt.into()],
    )
    .unwrap();
    let result_encoded = datum::encode_value(&mut EvalContext::default(), &row).unwrap();
    assert_eq!(&*result_encoded, &*expected_encoded);
    assert_eq!(spliter.next().is_none(), true);
}

#[test]
fn test_handle_truncate() {
    use tidb_query_datatype::{FieldTypeAccessor, FieldTypeTp};

    let rows = vec![
        (1, Some("name:0"), 2),
        (2, Some("name:4"), 3),
        (4, Some("name:3"), 1),
        (5, Some("name:1"), 4),
    ];

    let product = ProductTable::new();
    let mut endpoint = init_with_data(&product, &rows);
    let cols = product.columns_info();
    let cases = vec![
        {
            // count > "2x"
            let mut col = Expr::default();
            col.set_tp(ExprType::ColumnRef);
            col.mut_field_type()
                .as_mut_accessor()
                .set_tp(FieldTypeTp::LongLong);
            let count_offset = offset_for_column(&cols, product["count"].id);
            col.mut_val().write_i64(count_offset).unwrap();

            // "2x" will be truncated.
            let mut value = Expr::default();
            value
                .mut_field_type()
                .as_mut_accessor()
                .set_tp(FieldTypeTp::String);
            value.set_tp(ExprType::String);
            value.set_val(String::from("2x").into_bytes());

            let mut right = Expr::default();
            right
                .mut_field_type()
                .as_mut_accessor()
                .set_tp(FieldTypeTp::LongLong);
            right.set_tp(ExprType::ScalarFunc);
            right.set_sig(ScalarFuncSig::CastStringAsInt);
            right.mut_children().push(value);

            let mut cond = Expr::default();
            cond.mut_field_type()
                .as_mut_accessor()
                .set_tp(FieldTypeTp::LongLong);
            cond.set_tp(ExprType::ScalarFunc);
            cond.set_sig(ScalarFuncSig::LtInt);
            cond.mut_children().push(col);
            cond.mut_children().push(right);
            cond
        },
        {
            // id
            let mut col_id = Expr::default();
            col_id
                .mut_field_type()
                .as_mut_accessor()
                .set_tp(FieldTypeTp::LongLong);
            col_id.set_tp(ExprType::ColumnRef);
            let id_offset = offset_for_column(&cols, product["id"].id);
            col_id.mut_val().write_i64(id_offset).unwrap();

            // "3x" will be truncated.
            let mut value = Expr::default();
            value
                .mut_field_type()
                .as_mut_accessor()
                .set_tp(FieldTypeTp::String);
            value.set_tp(ExprType::String);
            value.set_val(String::from("3x").into_bytes());

            let mut int_3 = Expr::default();
            int_3
                .mut_field_type()
                .as_mut_accessor()
                .set_tp(FieldTypeTp::LongLong);
            int_3.set_tp(ExprType::ScalarFunc);
            int_3.set_sig(ScalarFuncSig::CastStringAsInt);
            int_3.mut_children().push(value);

            // count
            let mut col_count = Expr::default();
            col_count
                .mut_field_type()
                .as_mut_accessor()
                .set_tp(FieldTypeTp::LongLong);
            col_count.set_tp(ExprType::ColumnRef);
            let count_offset = offset_for_column(&cols, product["count"].id);
            col_count.mut_val().write_i64(count_offset).unwrap();

            // "3x" + count
            let mut plus = Expr::default();
            plus.mut_field_type()
                .as_mut_accessor()
                .set_tp(FieldTypeTp::LongLong);
            plus.set_tp(ExprType::ScalarFunc);
            plus.set_sig(ScalarFuncSig::PlusInt);
            plus.mut_children().push(int_3);
            plus.mut_children().push(col_count);

            // id = "3x" + count
            let mut cond = Expr::default();
            cond.mut_field_type()
                .as_mut_accessor()
                .set_tp(FieldTypeTp::LongLong);
            cond.set_tp(ExprType::ScalarFunc);
            cond.set_sig(ScalarFuncSig::EqInt);
            cond.mut_children().push(col_id);
            cond.mut_children().push(plus);
            cond
        },
    ];

    for cond in cases {
        // Ignore truncate error.
        let req = DagSelect::from(&product)
            .where_expr(cond.clone())
            .build_with(Context::default(), &[FLAG_IGNORE_TRUNCATE]);
        let resp = handle_select(&mut endpoint, req, None);
        assert!(!resp.has_error());
        assert!(resp.get_warnings().is_empty());

        // truncate as warning
        let req = DagSelect::from(&product)
            .where_expr(cond.clone())
            .build_with(Context::default(), &[FLAG_TRUNCATE_AS_WARNING]);
        let mut resp = handle_select(&mut endpoint, req, None);
        assert!(!resp.has_error());
        assert!(!resp.get_warnings().is_empty());
        // check data
        let mut spliter = DagChunkSpliter::new(resp.take_chunks().into(), 3);
        let row = spliter.next().unwrap();
        let (id, name, cnt) = rows[2];
        let name_datum = name.map(|s| s.as_bytes()).into();
        let expected_encoded = datum::encode_value(
            &mut EvalContext::default(),
            &[Datum::I64(id), name_datum, cnt.into()],
        )
        .unwrap();
        let result_encoded = datum::encode_value(&mut EvalContext::default(), &row).unwrap();
        assert_eq!(&*result_encoded, &*expected_encoded);
        assert_eq!(spliter.next().is_none(), true);

        // Do NOT ignore truncate error.
        let req = DagSelect::from(&product).where_expr(cond.clone()).build();
        let resp = handle_select(&mut endpoint, req, None);
        assert!(resp.has_error());
        assert!(resp.get_warnings().is_empty());
    }
}

#[test]
fn test_default_val() {
    let mut rows = vec![
        (1, Some("name:0"), 2),
        (2, Some("name:3"), 3),
        (4, Some("name:0"), 1),
        (5, Some("name:5"), 4),
        (6, Some("name:5"), 4),
        (7, None, 4),
    ];

    let product = ProductTable::new();
    let added = ColumnBuilder::new()
        .col_type(TYPE_LONG)
        .default(Datum::I64(3))
        .build();
    let mut tbl = TableBuilder::new()
        .add_col("id", product["id"].clone())
        .add_col("name", product["name"].clone())
        .add_col("count", product["count"].clone())
        .add_col("added", added)
        .build();
    tbl.id = product.id;

    let mut endpoint = init_with_data(&product, &rows);
    let expect: Vec<_> = rows.drain(..5).collect();
    let req = DagSelect::from(&tbl).limit(5).build();
    let mut resp = handle_select(&mut endpoint, req, None);
    let mut row_count = 0;
    let spliter = DagChunkSpliter::new(resp.take_chunks().into(), 4);
    for (row, (id, name, cnt)) in spliter.zip(expect) {
        let name_datum = name.map(|s| s.as_bytes()).into();
        let expected_encoded = datum::encode_value(
            &mut EvalContext::default(),
            &[id.into(), name_datum, cnt.into(), Datum::I64(3)],
        )
        .unwrap();
        let result_encoded = datum::encode_value(&mut EvalContext::default(), &row).unwrap();
        assert_eq!(&*result_encoded, &*expected_encoded);
        row_count += 1;
    }
    assert_eq!(row_count, 5);
}

#[test]
fn test_output_offsets() {
    let rows = vec![
        (1, Some("name:0"), 2),
        (2, Some("name:4"), 3),
        (4, Some("name:3"), 1),
        (5, Some("name:1"), 4),
    ];

    let product = ProductTable::new();
    let mut endpoint = init_with_data(&product, &rows);

    let req = DagSelect::from(&product)
        .output_offsets(Some(vec![1]))
        .build();
    let mut resp = handle_select(&mut endpoint, req, None);
    let spliter = DagChunkSpliter::new(resp.take_chunks().into(), 1);
    for (row, (_, name, _)) in spliter.zip(rows) {
        let name_datum = name.map(|s| s.as_bytes()).into();
        let expected_encoded =
            datum::encode_value(&mut EvalContext::default(), &[name_datum]).unwrap();
        let result_encoded = datum::encode_value(&mut EvalContext::default(), &row).unwrap();
        assert_eq!(&*result_encoded, &*expected_encoded);
    }
}

#[test]
fn test_output_counts() {
    let rows = vec![
        (1, Some("name:0"), 2),
        (2, Some("name:4"), 3),
        (4, Some("name:3"), 1),
        (5, Some("name:1"), 4),
    ];

    let product = ProductTable::new();
    let mut endpoint = init_with_data(&product, &rows);

    let req = DagSelect::from(&product).build();
    let resp = handle_select(&mut endpoint, req, None);
    assert_eq!(resp.get_output_counts(), &[rows.len() as i64]);
}

#[cfg(test)]
struct RowCache<'a>(&'a Table, Vec<BTreeMap<i64, Datum>>);

impl<'a> RowCache<'a> {
    pub fn new(table: &'a Table) -> Self {
        Self(table, vec![])
    }

    fn push(&mut self, container: BTreeMap<i64, Datum>) {
        self.rows().push(container);
    }

    #[must_use]
    fn add_col(
        &mut self,
        container: &mut BTreeMap<i64, Datum>,
        col: &Column,
        value: Datum,
    ) -> &mut Self {
        assert!(self.table().column_by_id(col.id).is_some());
        container.insert(col.id, value);
        self
    }

    pub fn add_row(&mut self, row: &(i64, Option<&str>, i64)) -> &mut Self {
        let table = self.0;
        let mut container = self.container();

        self.add_col(&mut container, &table["id"], Datum::I64(row.0))
            .add_col(
                &mut container,
                &table["name"],
                row.1.map(str::as_bytes).into(),
            )
            .add_col(&mut container, &table["count"], Datum::I64(row.2))
            .push(container);
        self
    }

    pub fn add_rows(&mut self, rows: &[(i64, Option<&str>, i64)]) -> &Self {
        for row in rows {
            self.add_row(row);
        }
        self
    }

    #[must_use]
    fn container(&self) -> BTreeMap<i64, Datum> {
        BTreeMap::<i64, Datum>::new()
    }

    #[must_use]
    fn table(&self) -> &'a Table {
        self.0
    }

    #[must_use]
    fn rows(&mut self) -> &mut Vec<BTreeMap<i64, Datum>> {
        &mut self.1
    }
}

#[cfg(test)]
pub struct Insert<'a> {
    client: &'a mut ClusterClient,
}

impl<'a> Insert<'a> {
    pub fn new(client: &'a mut ClusterClient) -> Self {
        Self { client }
    }

    fn prewrite(
        &mut self,
        start_ts: txn_types::TimeStamp,
        mut row_cache: RowCache<'_>,
    ) -> Vec<Mutation> {
        assert!(!row_cache.rows().is_empty());

        let mut mutations = vec![];
        let table = row_cache.table();

        for row in row_cache.rows() {
            let handle = row
                .get(&table.handle_id)
                .cloned()
                .unwrap_or_else(|| Datum::I64(next_id()));

            let key = add_txn_prefix(&table::encode_row_key(table.id, handle.i64()));
            let keys: Vec<_> = row.keys().cloned().collect();
            let values: Vec<_> = row.values().cloned().collect();
            let value = table::encode_row(&mut EvalContext::default(), values, &keys).unwrap();
            let mut m = Mutation::default();

            m.set_op(kvproto::kvrpcpb::Op::Put);
            m.set_key(key.to_vec());
            m.set_value(value.to_vec());

            mutations.push(m);

            for (&id, idxs) in &table.idxs {
                let mut v: Vec<_> = idxs.iter().map(|id| row[id].clone()).collect();

                v.push(handle.clone());

                let encoded = datum::encode_key(&mut EvalContext::default(), &v).unwrap();
                let idx_key = add_txn_prefix(&table::encode_index_seek_key(table.id, id, &encoded));
                let mut m = Mutation::default();

                m.set_op(kvproto::kvrpcpb::Op::Put);
                m.set_key(idx_key.to_vec());
                m.set_value(vec![0]);

                mutations.push(m);
            }
        }

        let txn_muts = TxnMutations::from_normal(mutations.clone());
        self.client
            .kv_prewrite(txn_muts.primary(), None, txn_muts, start_ts)
            .expect("kv_prewrite");

        mutations
    }

    fn execute(mut self, row_cache: RowCache<'_>) -> txn_types::TimeStamp {
        let start_ts = self.client.get_ts();
        let mutations = self.prewrite(start_ts, row_cache);
        self.client.put_commit(start_ts, &mutations).unwrap()
    }

    pub fn commit(
        &mut self,
        start_ts: txn_types::TimeStamp,
        mutations: &[Mutation],
    ) -> txn_types::TimeStamp {
        self.client.put_commit(start_ts, mutations).unwrap()
    }
}

#[cfg(test)]
pub struct Delete<'a> {
    client: &'a mut ClusterClient,
}

#[allow(dead_code)]
#[cfg(test)]
impl<'a> Delete<'a> {
    pub fn new(client: &'a mut ClusterClient) -> Self {
        Self { client }
    }

    fn prewrite(
        &mut self,
        start_ts: txn_types::TimeStamp,
        mut row_cache: RowCache<'_>,
    ) -> Vec<Mutation> {
        let mut total_mutations = vec![];
        let table = row_cache.table();

        for row in row_cache.rows() {
            let handle = row
                .get(&table.handle_id)
                .cloned()
                .unwrap_or_else(|| Datum::I64(next_id()));

            let key = table::encode_row_key(table.id, handle.i64());
            let mut mutations = vec![];
            let mut m = Mutation::default();

            m.set_op(kvproto::kvrpcpb::Op::Del);
            m.set_key(key.to_vec());
            mutations.push(m);

            for (&id, idx_cols) in &table.idxs {
                let mut v: Vec<_> = idx_cols.iter().map(|id| row[id].clone()).collect();

                v.push(handle.clone());

                let encoded = datum::encode_key(&mut EvalContext::default(), &v).unwrap();
                let idx_key = table::encode_index_seek_key(table.id, id, &encoded);
                let mut mutations = vec![];
                let mut m = Mutation::default();

                m.set_op(kvproto::kvrpcpb::Op::Del);
                m.set_key(idx_key.to_vec());

                mutations.push(m);
            }

            total_mutations.extend(mutations.to_owned());
            let txn_muts = TxnMutations::from_normal(mutations.clone());
            self.client
                .kv_prewrite(txn_muts.primary(), None, txn_muts, start_ts)
                .expect("kv_prewrite");
        }

        total_mutations
    }

    fn execute(&mut self, row_cache: RowCache<'_>) {
        let start_ts = self.client.get_ts();
        let mutations = self.prewrite(start_ts, row_cache);
        self.client.del_table_rows_commit(start_ts, &mutations);
    }

    pub fn commit(&mut self, start_ts: txn_types::TimeStamp, mutations: &[Mutation]) {
        self.client.del_table_rows_commit(start_ts, mutations);
    }
}

#[cfg(test)]
#[derive(Debug, Clone)]
struct DagTestSnapshot {
    memtable_rows: Vec<u8>,
    cs: Vec<u8>,
    ctx: kvproto::kvrpcpb::Context,
}

impl DagTestSnapshot {
    fn print_to_info(&self) {
        let rows: Vec<Row> = bincode::deserialize(&self.memtable_rows[4..]).unwrap();
        let mut change_set = kvenginepb::ChangeSet::default();
        change_set.merge_from_bytes(&self.cs).unwrap();
        info!(
            "changeset: {:?}, ctx: {:?}, rows.len: {}",
            self.cs,
            self.ctx,
            rows.len()
        );
        for row in rows {
            info!(
                "{:?} : {:?} => {:?}",
                row.key,
                row.user_meta,
                std::str::from_utf8(&row.value).unwrap()
            );
        }
    }
}

#[cfg(test)]
enum Dml<'a> {
    Ins(Insert<'a>),
    #[allow(dead_code)]
    Del(Delete<'a>),
    Empty,
}

#[cfg(test)]
impl<'a> Default for Dml<'a> {
    fn default() -> Self {
        Dml::Empty
    }
}

#[cfg(test)]
struct Txn<'a> {
    started: bool,
    committed: bool,
    start_ts: txn_types::TimeStamp,
    prewrites: Vec<(Dml<'a>, Vec<Mutation>)>,
}

#[cfg(test)]
impl<'a> Txn<'a> {
    fn new() -> Self {
        Self {
            started: false,
            committed: false,
            start_ts: txn_types::TimeStamp::default(),
            prewrites: vec![],
        }
    }

    /// For auto-commit
    fn auto_commit(dml: Dml<'_>, row_cache: RowCache<'_>) {
        match dml {
            Dml::Ins(inserter) => {
                let _ = inserter.execute(row_cache);
            }
            Dml::Del(mut deleter) => {
                deleter.execute(row_cache);
            }
            Dml::Empty => panic!("Dml can't be Empty"),
        }
    }

    fn begin(&mut self, start_ts: txn_types::TimeStamp) -> &Self {
        assert!(!self.started);
        self.started = true;
        self.start_ts = start_ts;
        self
    }

    fn prewrite_execute(&self, dml: &mut Dml<'a>, row_cache: RowCache<'_>) -> Vec<Mutation> {
        match dml {
            Dml::Ins(inserter) => inserter.prewrite(self.start_ts, row_cache),
            Dml::Del(deleter) => deleter.prewrite(self.start_ts, row_cache),
            Dml::Empty => panic!("Dml can't be Empty"),
        }
    }

    fn prewrite(&mut self, mut dml: Dml<'a>, row_cache: RowCache<'_>) {
        assert!(self.started);
        assert!(!self.committed);
        let r = self.prewrite_execute(&mut dml, row_cache);
        self.prewrites.push((dml, r));
    }

    fn commit(&mut self) {
        assert!(self.started);
        assert!(!self.committed);

        if self.prewrites.is_empty() {
            return;
        }

        for (dml, mutations) in &mut self.prewrites {
            match dml {
                Dml::Ins(inserter) => {
                    let _ = inserter.commit(self.start_ts, mutations);
                }
                Dml::Del(deleter) => {
                    deleter.commit(self.start_ts, mutations);
                }
                Dml::Empty => panic!("Dml can't be Empty"),
            }
        }
        self.committed = true;
    }
}

#[cfg(test)]
struct DagTest<'a> {
    table: &'a ProductTable,
    cluster: ServerCluster,
    pub client: ClusterClient,
    master_key: MasterKey,
}

#[cfg(test)]
impl<'a> DagTest<'a> {
    pub fn new(table: &'a ProductTable) -> Self {
        let node_id = alloc_node_id();
        let mut dfs_cfg = kvengine::dfs::DFSConfig::default();

        dfs_cfg.s3_endpoint = "memory".to_owned();
        dfs_cfg.zstd_compression_level = "3".to_string();

        let cluster = ServerCluster::new(vec![node_id], |_, conf| {
            conf.dfs = dfs_cfg.clone();
            conf.enable_inner_key_offset = node_id % 2 == 0;
        });

        let mut client = cluster.new_client();
        client.split_keyspace(1);
        let master_key = cluster.get_kvengine(node_id).get_master_key();

        Self {
            table,
            cluster,
            client,
            master_key,
        }
    }

    pub fn get_ts(&self) -> txn_types::TimeStamp {
        self.client.get_ts()
    }

    fn get_inserter(&mut self) -> Dml<'_> {
        Dml::Ins(Insert::new(&mut self.client))
    }

    #[allow(dead_code)]
    fn get_deleter(&mut self) -> Dml<'_> {
        Dml::Del(Delete::new(&mut self.client))
    }

    fn get_row_cache(&self) -> RowCache<'a> {
        RowCache::new(self.table)
    }

    pub fn get_key_range_all(&self) -> coppb::KeyRange {
        let mut key_range = self.table.get_record_range_all();
        encode_key_range(&mut key_range);
        key_range
    }

    pub fn get_index_key_range_all(&self, index_id: i64) -> coppb::KeyRange {
        let mut key_range = self.table.get_index_range_all(index_id);
        encode_key_range(&mut key_range);
        key_range
    }

    pub fn get_key_range(&self, start: i64, end: i64) -> coppb::KeyRange {
        let mut key_range = self.table.get_record_range(start, end);
        encode_key_range(&mut key_range);
        key_range
    }

    pub fn _get_index_key_range(&self, index_id: i64, start: i64, end: i64) -> coppb::KeyRange {
        let mut key_range = self.table.get_index_record_range(index_id, start, end);
        encode_key_range(&mut key_range);
        key_range
    }

    #[allow(dead_code)]
    fn execute_select_range(
        &mut self,
        req: Request,
        index_id: Option<i64>,
        start: i64,
        end: i64,
    ) -> kvproto::coprocessor::Response {
        let select_key_range = if let Some(index_id) = index_id {
            self._get_index_key_range(index_id, start, end)
        } else {
            self.get_key_range(start, end)
        };
        let snapshot = self.fetch_snapshot(self.get_ts().into_inner(), vec![select_key_range]);
        let quota_limiter = Arc::new(QuotaLimiter::default());
        match self.execute(snapshot, quota_limiter, req) {
            Ok(resp) => resp,
            Err(e) => panic!("{}", e),
        }
    }

    pub fn insert_and_commit(&mut self, rows: &[(i64, Option<&str>, i64)]) {
        let mut row_cache = self.get_row_cache();
        row_cache.add_rows(rows);
        Txn::auto_commit(self.get_inserter(), row_cache);
    }

    pub fn insert_rows(&mut self, count: i64) {
        let mut to_insert = count;
        let (batches, mut batch_size) = if to_insert < MAX_INSERT_BATCH_SIZE {
            (1, to_insert)
        } else {
            let mut batch_size = to_insert / MAX_INSERT_BATCH_SIZE;
            if to_insert % MAX_INSERT_BATCH_SIZE > 0 {
                batch_size += 1;
            }
            (batch_size, MAX_INSERT_BATCH_SIZE)
        };

        for i in 0..batches {
            let mut row_cache = self.get_row_cache();

            batch_size = if batch_size > to_insert {
                to_insert
            } else {
                batch_size
            };

            for j in 0..batch_size {
                let mut container = row_cache.container();

                row_cache
                    .add_col(
                        &mut container,
                        &self.table["id"],
                        Datum::I64(count - to_insert),
                    )
                    .add_col(
                        &mut container,
                        &self.table["name"],
                        Datum::Bytes(format!("j: {}", j).into()),
                    )
                    .add_col(&mut container, &self.table["count"], Datum::I64(i))
                    .push(container);

                to_insert -= 1;
            }

            Txn::auto_commit(self.get_inserter(), row_cache);
        }
        assert_eq!(to_insert, 0);
    }

    fn fetch_snapshot(
        &mut self,
        start_ts: u64,
        key_ranges: Vec<coppb::KeyRange>,
    ) -> DagTestSnapshot {
        let dag_test_snapshot = match self
            .client
            .get_memtable_snapshot(None, start_ts, key_ranges)
        {
            Ok((memtable_rows, cs, ctx)) => DagTestSnapshot {
                memtable_rows,
                cs,
                ctx,
            },
            Err(e) => panic!("get_memtable_snapshot failed: {}", e),
        };
        dag_test_snapshot
    }

    fn execute_select_all_generic(
        &mut self,
        req: Request,
        index_id: Option<i64>,
    ) -> kvproto::coprocessor::Response {
        let select_key_range = if let Some(index_id) = index_id {
            self.get_index_key_range_all(index_id)
        } else {
            self.get_key_range_all()
        };
        let snapshot = self.fetch_snapshot(self.get_ts().into_inner(), vec![select_key_range]);
        let quota_limiter = Arc::new(QuotaLimiter::default());
        match self.execute(snapshot, quota_limiter, req) {
            Ok(resp) => resp,
            Err(e) => panic!("{}", e),
        }
    }

    fn select_all(&mut self, mut req: Request, index_id: Option<i64>) -> tipb::SelectResponse {
        let ranges = req.mut_ranges();
        for key_range in ranges.iter_mut() {
            encode_key_range(key_range);
        }
        let cop_resp = self.execute_select_all_generic(req, index_id);
        let mut resp = tipb::SelectResponse::default();
        resp.merge_from_bytes(cop_resp.get_data()).unwrap();
        resp
    }

    pub fn execute(
        &self,
        snapshot: DagTestSnapshot,
        quota_limiter: Arc<QuotaLimiter>,
        mut req: Request,
    ) -> Result<coppb::Response, tikv::coprocessor::Error> {
        req.mut_context().set_api_version(ApiVersion::V2);
        let tag = cloud_worker::get_cop_req_tag(&req);
        let f = async {
            let snap_access = SnapAccess::construct_snapshot(
                tag,
                self.cluster.get_dfs().unwrap(),
                &snapshot.memtable_rows,
                &snapshot.cs,
                &self.master_key,
                None,
            )
            .await
            .unwrap();
            let snap = RegionSnapshot::from_snapshot(snap_access);
            let (tx, rx) = tokio::sync::oneshot::channel();

            std::thread::spawn(move || {
                let f = tikv::coprocessor::parse_request_and_handle_remote_cop::<RegionSnapshot>(
                    req,
                    None,
                    std::time::Duration::new(1, 0),
                    quota_limiter,
                    snap,
                );
                let result = futures::executor::block_on(f);
                tx.send(result).unwrap();
            });

            match rx.await.unwrap() {
                Ok(mut data) => Ok(data.consume()),
                Err(e) => Err(e),
            }
        };
        futures::executor::block_on(f)
    }
}

impl Drop for DagTest<'_> {
    fn drop(&mut self) {
        self.cluster.stop();
    }
}

fn handle_select(
    dag_test: &mut DagTest<'_>,
    req: Request,
    index_id: Option<i64>,
) -> tipb::SelectResponse {
    dag_test.select_all(req, index_id)
}

fn init_with_data<'a>(tbl: &'a ProductTable, rows: &[(i64, Option<&str>, i64)]) -> DagTest<'a> {
    test_util::init_log_for_test();
    let mut dag_test = DagTest::new(tbl);
    dag_test.insert_and_commit(rows);
    dag_test
}

fn add_txn_prefix(key: &[u8]) -> Vec<u8> {
    let mut v = api_version::ApiV2::get_txn_keyspace_prefix(1);
    v.extend_from_slice(key);
    v
}

fn encode_key_range(key_range: &mut coppb::KeyRange) {
    key_range.set_start(add_txn_prefix(key_range.get_start()));
    key_range.set_end(add_txn_prefix(key_range.get_end()));
}
