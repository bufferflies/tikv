// Copyright 2024 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering::Relaxed},
        Arc,
    },
    time::Duration,
};

use anyhow::{Context, Result};
use futures::future::join_all;
use rand::prelude::*;
use sqlx::{
    types::{
        chrono::{DateTime, NaiveDate, NaiveDateTime, NaiveTime, Utc},
        BigDecimal,
    },
    Column, MySql, Pool, Row, TypeInfo,
};
use test_cloud_server::{keyspace::KeyspaceManager, tidb::TidbCluster};
use tikv_util::{debug, error, info, time::Instant};

use crate::{
    sql_util::{
        get_engine_hint, is_db_error_retryable, wait_tiflash_or_columnar_replicas_available,
        DEADLOCK_ERR_MSG,
    },
    Running, COLUMNAR_RETRY_COUNTER, COLUMNAR_WRITE_COUNTER,
};

const COLUMNAR_DB_NAME: &str = "columnar_db";
const COLUMNAR_TABLE_NAME: &str = "columnar_table";
const WORKLOAD_CONCURRENCY: usize = 1;

const COLUMNAR_REPLICAS_AVAILABLE_TIMEOUT: Duration = Duration::from_secs(30);

pub(crate) async fn prepare_columnar(
    tc: TidbCluster,
    keyspace_manager: KeyspaceManager,
    keyspace_id: u32,
) {
    let keyspace_name = keyspace_manager
        .get_keyspace_meta(keyspace_id)
        .unwrap()
        .name();
    let tag = format!("columnar-{}[{}]", keyspace_id, keyspace_name);
    let tidb_idx = TidbCluster::get_idx_by_keyspace_name(&keyspace_name);
    let params = tc.tidb.conn_params(tidb_idx);

    let conn_string = params.conn_string("test");
    let pool = sqlx::MySqlPool::connect(&conn_string).await.unwrap();

    info!("{} prepare columnar table", tag);

    let decimal_precision: u8 = rand::random::<u8>() % 50 + 10;
    let decimal_points: u8 = rand::random::<u8>() % decimal_precision.min(10);

    let sqls = vec![
        format!("drop database if exists `{COLUMNAR_DB_NAME}`"),
        format!("create database `{COLUMNAR_DB_NAME}`"),
        format!(
            "create table `{COLUMNAR_DB_NAME}`.`{COLUMNAR_TABLE_NAME}` \
            (id INT AUTO_INCREMENT PRIMARY KEY, \
            int_col INT, \
            tinyint_col TINYINT, \
            smallint_col SMALLINT, \
            mediumint_col MEDIUMINT, \
            bigint_col BIGINT, \
            float_col FLOAT, \
            double_col DOUBLE, \
            decimal_col DECIMAL({decimal_precision}, {decimal_points}), \
            date_col DATE, \
            datetime_col DATETIME, \
            timestamp_col TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP, \
            time_col TIME, \
            year_col YEAR, \
            char_col CHAR(10), \
            varchar_col VARCHAR(255), \
            text_col TEXT, \
            mediumtext_col MEDIUMTEXT, \
            longtext_col LONGTEXT, \
            binary_col BINARY(16), \
            varbinary_col VARBINARY(255), \
            blob_col BLOB, \
            mediumblob_col MEDIUMBLOB, \
            longblob_col LONGBLOB, \
            enum_col ENUM('enum-1', 'enum-2', 'enum-3'), \
            set_col SET('set-1', 'set-2', 'set-3'), \
            bool_col BOOLEAN, \
            json_col JSON, \
            default_col INT DEFAULT 100) COMMENT='columnar_engine'"
        ),
        // We should set tiflash replica to enable tidb plan tiflash replica.
        format!("alter table `{COLUMNAR_DB_NAME}`.`{COLUMNAR_TABLE_NAME}` set tiflash replica 1"),
    ];

    for sql in sqls {
        info!("{} prepare columnar table: executing sql", tag; "sql" => &sql);
        sqlx::query(&sql).execute(&pool).await.unwrap();
    }

    wait_tiflash_or_columnar_replicas_available(
        &tag,
        &pool,
        COLUMNAR_DB_NAME,
        COLUMNAR_TABLE_NAME,
        COLUMNAR_REPLICAS_AVAILABLE_TIMEOUT,
    )
    .await;
}

async fn trigger_columnar_major_compaction(pool: &Pool<MySql>) {
    // remove columnar replica.
    let sql =
        format!("alter table `{COLUMNAR_DB_NAME}`.`{COLUMNAR_TABLE_NAME}` set tiflash replica 0");
    sqlx::query(&sql).execute(pool).await.unwrap();
    // remove table comment
    let sql = format!("alter table `{COLUMNAR_DB_NAME}`.`{COLUMNAR_TABLE_NAME}` comment ''");
    sqlx::query(&sql).execute(pool).await.unwrap();
    // wait columnar replica cleared
    tokio::time::sleep(Duration::from_secs(10)).await;

    // add table comment
    let sql = format!(
        "alter table `{COLUMNAR_DB_NAME}`.`{COLUMNAR_TABLE_NAME}` comment 'columnar_engine'"
    );
    sqlx::query(&sql).execute(pool).await.unwrap();
    // add columnar replica.
    let sql =
        format!("alter table `{COLUMNAR_DB_NAME}`.`{COLUMNAR_TABLE_NAME}` set tiflash replica 1");
    sqlx::query(&sql).execute(pool).await.unwrap();
    // wait columnar replica ready
    wait_tiflash_or_columnar_replicas_available(
        "trigger_columnar_major_compaction",
        pool,
        COLUMNAR_DB_NAME,
        COLUMNAR_TABLE_NAME,
        COLUMNAR_REPLICAS_AVAILABLE_TIMEOUT * 2,
    )
    .await;
}

pub(crate) async fn run_columnar_workload(
    tc: TidbCluster,
    keyspace_manager: KeyspaceManager,
    keyspace_id: u32,
    running: Running,
) {
    info!("run workload with columnar");
    let keyspace_name = keyspace_manager
        .get_keyspace_meta(keyspace_id)
        .unwrap()
        .name();
    let tidb_idx = TidbCluster::get_idx_by_keyspace_name(&keyspace_name);
    let params = tc.tidb.conn_params(tidb_idx);
    let conn_string = params.conn_string(COLUMNAR_DB_NAME);
    let pool = sqlx::MySqlPool::connect(&conn_string).await.unwrap();

    let mut handles = Vec::with_capacity(WORKLOAD_CONCURRENCY);
    let max_id = Arc::new(AtomicU64::new(0));
    let pause_signal = Arc::new(AtomicBool::new(false));
    for tid in 0..WORKLOAD_CONCURRENCY {
        let pool = pool.clone();
        let running = running.clone();
        let max_id = max_id.clone();
        let pause_signal = pause_signal.clone();
        let handle = tokio::spawn(async move {
            let tag = format!("columnar-{}-{}", keyspace_id, tid);
            let start_time = Instant::now();
            while running.get() {
                if pause_signal.load(Relaxed) {
                    tokio::time::sleep(Duration::from_secs(1)).await;
                    continue;
                }
                let mut sqls = generate_insert_sqls(10);
                let del_sqls = generate_delete_sqls(10, max_id.load(Relaxed));
                sqls.extend(del_sqls);
                max_id.fetch_add(10, Relaxed);
                let ok = insert_delete_random_records(&tag, &pool, &sqls)
                    .await
                    .unwrap_or_else(|err| {
                        panic!("{} columnar insert/delete error: {:?}", tag, err);
                    });
                if ok {
                    COLUMNAR_WRITE_COUNTER.fetch_add(1, Relaxed);
                } else {
                    COLUMNAR_RETRY_COUNTER.fetch_add(1, Relaxed);
                }
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
            info!("columnar workload exit"; "tag" => tag, "dur" => ?start_time.saturating_elapsed());
        });
        handles.push(handle);
    }

    let pool_copy = pool.clone();
    let pause_signal_copy = pause_signal.clone();
    handles.push(tokio::spawn(async move {
        let start_time = Instant::now();
        while running.get() {
            if pause_signal_copy.load(Relaxed) {
                tokio::time::sleep(Duration::from_secs(1)).await;
                continue;
            }
            info!("verify_data randomly");
            match verify_data(&pool_copy, false).await {
                Ok(_) => {
                    info!("verify_data randomly success");
                }
                Err(err) => {
                    panic!("verify_data randomly error: {}", err);
                }
            }
            tokio::time::sleep(Duration::from_secs(5)).await;
        }
        info!("columnar verify thread exit"; "dur" => ?start_time.saturating_elapsed());
    }));

    let pool_copy = pool.clone();
    handles.push(tokio::spawn(async move {
        tokio::time::sleep(Duration::from_secs(60)).await;
        info!("pause columnar workload trigger columnar major compaction");
        pause_signal.store(true, Relaxed);
        trigger_columnar_major_compaction(&pool_copy).await;
        pause_signal.store(false, Relaxed);
    }));

    join_all(handles).await;

    match verify_data(&pool, true).await {
        Ok(_) => {
            info!("verify_data columnar success");
        }
        Err(err) => {
            panic!("verify_data columnar error: {}", err);
        }
    }
}

fn random_str(rng: &mut ThreadRng, len: usize, is_var: bool) -> String {
    let real_len = if is_var { rng.gen_range(1..=len) } else { len };
    (1..=real_len)
        .map(|_| rng.sample(rand::distributions::Alphanumeric) as char)
        .collect()
}

fn generate_insert_sqls(max_count: usize) -> Vec<String> {
    let mut rng = rand::thread_rng();
    let count = rng.gen_range(1..=max_count);
    let int_col: i32 = rng.gen();
    let smallint_col: i16 = rng.gen();
    let tinyint_col: i8 = rng.gen();
    let mediumint_col: i16 = rng.gen();
    let bigint_col: i64 = rng.gen();
    let float_col: f32 = rng.gen();
    let double_col: f64 = rng.gen();
    let decimal_col: f32 = rng.gen();
    let naive_date: NaiveDate = NaiveDate::from_ymd_opt(
        rng.gen_range(2000..2100),
        rng.gen_range(1..=12),
        rng.gen_range(1..=28),
    )
    .unwrap_or(NaiveDate::MIN);
    let naive_time: NaiveTime = NaiveTime::from_hms_opt(
        rng.gen_range(0..24),
        rng.gen_range(0..60),
        rng.gen_range(0..60),
    )
    .unwrap_or(NaiveTime::MIN);
    let date_col = naive_date.clone().to_string();
    let datetime_col = NaiveDateTime::new(naive_date, naive_time).to_string();
    let timestamp_col = "CURRENT_TIMESTAMP";
    let time_col = naive_time.to_string();
    let year_col = rng.gen_range(2000..2100).to_string();
    let char_col = random_str(&mut rng, 10, false);
    let varchar_col = random_str(&mut rng, 255, true);
    let text_col = random_str(&mut rng, 255, true);
    let mediumtext_col = random_str(&mut rng, 512, true);
    let longtext_col = random_str(&mut rng, 1024, true);
    let binary_col = random_str(&mut rng, 16, false);
    let varbinary_col = random_str(&mut rng, 255, true);
    let blob_col = random_str(&mut rng, 255, true);
    let mediumblob_col = random_str(&mut rng, 512, true);
    let longblob_col = random_str(&mut rng, 1024, true);
    let idx = rng.gen_range(1..=3);
    let enum_col = format!("enum-{}", idx);
    let set_col = format!("set-{}", idx);
    let bool_col = rng.gen_bool(0.5);
    let json_col = format!(
        r#"{{"key": "key", "value": "{}"}}"#,
        random_str(&mut rng, 10, true)
    );

    let mut sqls = vec![format!("begin pessimistic")];
    for _ in 0..count {
        let sql = format!(
            "INSERT INTO `{COLUMNAR_DB_NAME}`.`{COLUMNAR_TABLE_NAME}` (
                int_col, tinyint_col, smallint_col, mediumint_col, bigint_col,
                float_col, double_col, decimal_col,
                date_col, datetime_col, timestamp_col, time_col, year_col,
                char_col, varchar_col, text_col, mediumtext_col, longtext_col,
                binary_col, varbinary_col, blob_col, mediumblob_col, longblob_col,
                enum_col, set_col,
                bool_col,
                json_col
            ) VALUES (
                {int_col}, {tinyint_col}, {smallint_col}, {mediumint_col}, {bigint_col},
                {float_col}, {double_col}, {decimal_col},
                '{date_col}', '{datetime_col}', {timestamp_col}, '{time_col}', '{year_col}',
                '{char_col}', '{varchar_col}', '{text_col}', '{mediumtext_col}', '{longtext_col}',
                '{binary_col}', '{varbinary_col}', '{blob_col}', '{mediumblob_col}', '{longblob_col}',
                '{enum_col}', '{set_col}',
                {bool_col},
                '{json_col}'
            );"
        );

        sqls.push(sql);
    }
    sqls
}

fn generate_delete_sqls(count: usize, max_id: u64) -> Vec<String> {
    // Generate random count ids with max value max_id.
    let mut rng = rand::thread_rng();
    let ids: Vec<u64> = (1..=max_id).choose_multiple(&mut rng, count);
    let mut sqls = vec![];
    for id in ids {
        let sql =
            format!("delete from `{COLUMNAR_DB_NAME}`.`{COLUMNAR_TABLE_NAME}` where id = {id};");
        sqls.push(sql);
    }
    sqls
}

async fn insert_delete_random_records(
    tag: &str,
    pool: &Pool<MySql>,
    sqls: &[String],
) -> Result<bool> {
    let mut conn = pool.acquire().await.context("acquire")?;
    for sql in sqls {
        info!("{} columnar_workload: executing sql", tag; "sql" => sql);
        match sqlx::query(sql).execute(&mut conn).await {
            Ok(_) => {}
            Err(sqlx::Error::Database(err)) if err.message().contains(DEADLOCK_ERR_MSG) => {
                info!("{} columnar_workload: ignore deadlock, retry", tag; "sql" => sql, "err" => ?err);
                return Ok(false);
            }
            Err(err) => {
                error!("{} columnar_workload: execute failed", tag; "sql" => sql, "err" => ?err);
                return Err(err).with_context(|| format!("columnar_workload: sql: {}", sql));
            }
        }
    }
    match sqlx::query("commit").execute(&mut conn).await {
        Ok(_) => {
            info!(
                "{} columnar_workload: commit success, insert rows: {}",
                tag,
                sqls.len() - 1
            );
            Ok(true)
        }
        Err(err) if is_db_error_retryable(&err) => {
            info!("{} columnar_workload: commit ignore retryable error", tag; "err" => ?err);
            Ok(false)
        }
        Err(err) => {
            error!("{} columnar_workload: commit failed", tag; "err" => ?err);
            Err(err).context("columnar_workload commit")
        }
    }
}

async fn verify_data(pool: &Pool<MySql>, full_scan: bool) -> Result<()> {
    let mut tx = pool.begin().await.context("begin")?;
    let query_engine = |use_tiflash: bool| {
        format!(
            "select {} count(id) as count from `{COLUMNAR_DB_NAME}`.`{COLUMNAR_TABLE_NAME}`",
            get_engine_hint(use_tiflash, COLUMNAR_TABLE_NAME)
        )
    };
    // Verify the row count first.
    info!("verify_data: execute sql: {}", query_engine(false));
    let row_result = sqlx::query(&query_engine(false))
        .fetch_one(&mut tx)
        .await
        .context("select count(id) row engine")?;
    let row_count: i64 = row_result.get("count");
    info!("verify_data: execute sql: {}", query_engine(true));
    let col_result = sqlx::query(&query_engine(true))
        .fetch_one(&mut tx)
        .await
        .context("select count(id) col engine")?;
    let col_count: i64 = col_result.get("count");
    if row_count != col_count {
        // dump the data
        let scan_engine = |use_tiflash: bool| {
            format!(
                "select {} id from `{COLUMNAR_DB_NAME}`.`{COLUMNAR_TABLE_NAME}`",
                get_engine_hint(use_tiflash, COLUMNAR_TABLE_NAME)
            )
        };
        let row_result = sqlx::query(&scan_engine(false))
            .fetch_all(&mut tx)
            .await
            .context("select id row engine")?;
        let col_result = sqlx::query(&scan_engine(true))
            .fetch_all(&mut tx)
            .await
            .context("select id columnar engine")?;
        if row_result.len() != col_result.len() {
            for row in &row_result {
                let id: i64 = row.get("id");
                info!("verify_data: row_result id: {}", id);
            }
            for row in &col_result {
                let id: i64 = row.get("id");
                info!("verify_data: col_result id: {}", id);
            }
        }

        // Retry the query to check if the same with previous result.
        let row_result1 = sqlx::query(&query_engine(false))
            .fetch_one(&mut tx)
            .await
            .context("select count(id) row engine")?;
        let row_count1: i64 = row_result1.get("count");
        let col_result1 = sqlx::query(&query_engine(true))
            .fetch_one(&mut tx)
            .await
            .context("select count(id) col engine")?;
        let col_count1: i64 = col_result1.get("count");
        panic!(
            "row count {} != col count {}, row_result: {}, col_result: {}, retry row_count: {}, retry col_count: {}",
            row_count,
            col_count,
            row_result.len(),
            col_result.len(),
            row_count1,
            col_count1
        );
    }
    if row_count == 0 {
        info!("verify_data success, empty dataset");
        return Ok(());
    }
    info!("verify_data begin compare_rows for {} rows", row_count);

    // Verify all records data.
    let scan_engine = |use_tiflash: bool, offset: u64, limit: u64| {
        format!(
            "select {} id, int_col, tinyint_col, smallint_col, mediumint_col, bigint_col,
                float_col, double_col, decimal_col,
                date_col, datetime_col, timestamp_col, time_col, year_col,
                char_col, varchar_col, text_col, mediumtext_col, longtext_col,
                binary_col, varbinary_col, blob_col, mediumblob_col, longblob_col,
                enum_col, set_col,
                bool_col,
                json_col from `{COLUMNAR_DB_NAME}`.`{COLUMNAR_TABLE_NAME}` order by id limit {limit} offset {offset} ",
            get_engine_hint(use_tiflash, COLUMNAR_TABLE_NAME)
        )
    };

    let (offset, limit) = if full_scan {
        (0, row_count as u64)
    } else {
        (
            rand::random::<u64>() % row_count as u64 / 2,
            rand::random::<u64>() % row_count as u64 / 2,
        )
    };

    let row_result = sqlx::query(&scan_engine(false, offset, limit))
        .fetch_all(&mut tx)
        .await
        .context("select * row engine")?;
    let col_result = sqlx::query(&scan_engine(true, offset, limit))
        .fetch_all(&mut tx)
        .await
        .context("select * columnar engine")?;
    for (row1, row2) in row_result.iter().zip(col_result.iter()) {
        compare_rows(row1, row2).unwrap();
    }

    let scan_engine_with_primary_condition =
        |use_tiflash: bool, lower_bound: u32, upper_bound: u32| {
            format!(
            "select {} id, int_col, tinyint_col, smallint_col, mediumint_col, bigint_col,
                float_col, double_col, decimal_col,
                date_col, datetime_col, timestamp_col, time_col, year_col,
                char_col, varchar_col, text_col, mediumtext_col, longtext_col,
                binary_col, varbinary_col, blob_col, mediumblob_col, longblob_col,
                enum_col, set_col,
                bool_col,
                json_col from `{COLUMNAR_DB_NAME}`.`{COLUMNAR_TABLE_NAME}` where id >= {lower_bound} and id <= {upper_bound} order by id",
            get_engine_hint(use_tiflash, COLUMNAR_TABLE_NAME)
        )
        };

    let scan_engine_with_condition = |use_tiflash: bool, lower_bound: u32, upper_bound: u32| {
        format!(
            "select {} id, int_col, tinyint_col, smallint_col, mediumint_col, bigint_col,
                float_col, double_col, decimal_col,
                date_col, datetime_col, timestamp_col, time_col, year_col,
                char_col, varchar_col, text_col, mediumtext_col, longtext_col,
                binary_col, varbinary_col, blob_col, mediumblob_col, longblob_col,
                enum_col, set_col,
                bool_col,
                json_col from `{COLUMNAR_DB_NAME}`.`{COLUMNAR_TABLE_NAME}` where int_col >= {lower_bound} and int_col <= {upper_bound} order by id",
            get_engine_hint(use_tiflash, COLUMNAR_TABLE_NAME)
        )
    };

    // Generate random number between 0 and row_count.
    let lower_bound = rand::random::<u32>() % row_count as u32;
    // Generate random number between row_count and u32::MAX.
    let upper_bound = rand::random::<u32>() % (u32::MAX - row_count as u32) + row_count as u32;

    // Verify the data with condition on primary id.
    let row_result = sqlx::query(&scan_engine_with_primary_condition(
        false,
        lower_bound,
        upper_bound,
    ))
    .fetch_all(&mut tx)
    .await
    .context("select * row engine with primary condition")?;
    let col_result = sqlx::query(&scan_engine_with_primary_condition(
        true,
        lower_bound,
        upper_bound,
    ))
    .fetch_all(&mut tx)
    .await
    .context("select * columnar engine with primary condition")?;
    if row_result.len() != col_result.len() {
        for row in &row_result {
            let id: i64 = row.get("id");
            info!("verify_data: row_result id: {}", id);
        }
        for row in &col_result {
            let id: i64 = row.get("id");
            info!("verify_data: col_result id: {}", id);
        }
        panic!(
            "scan_with_primary_condition row count {} != col count {}",
            row_result.len(),
            col_result.len()
        );
    }
    for (row1, row2) in row_result.iter().zip(col_result.iter()) {
        compare_rows(row1, row2).unwrap();
    }

    // Verify the data with condition on int_col.
    let row_result = sqlx::query(&scan_engine_with_condition(false, lower_bound, upper_bound))
        .fetch_all(&mut tx)
        .await
        .context("select * row engine with primary condition")?;
    let col_result = sqlx::query(&scan_engine_with_condition(true, lower_bound, upper_bound))
        .fetch_all(&mut tx)
        .await
        .context("select * columnar engine with primary condition")?;
    if row_result.len() != col_result.len() {
        for row in &row_result {
            let id: i64 = row.get("id");
            let int_col: i64 = row.get("int_col");
            info!("verify_data: row_result id: {}, int_col: {}", id, int_col);
        }
        for row in &col_result {
            let id: i64 = row.get("id");
            let int_col: i64 = row.get("int_col");
            info!("verify_data: col_result id: {}, int_col: {}", id, int_col);
        }
        panic!(
            "scan_with_condition row count {} != col count {}",
            row_result.len(),
            col_result.len()
        );
    }
    for (row1, row2) in row_result.iter().zip(col_result.iter()) {
        compare_rows(row1, row2).unwrap();
    }

    Ok(())
}

fn compare_rows(row1: &sqlx::mysql::MySqlRow, row2: &sqlx::mysql::MySqlRow) -> Result<()> {
    assert_eq!(row1.len(), row2.len(), "Row length mismatch");
    debug!("verify_data compare_rows, cols: {}", row1.len());

    for (col1, col2) in row1.columns().iter().zip(row2.columns().iter()) {
        assert_eq!(col1.name(), col2.name(), "Column name mismatch");
        assert_eq!(col1.type_info(), col2.type_info(), "Column type mismatch");
        match col1.type_info().name() {
            "TINYINT" | "SMALLINT" | "INT" | "MEDIUMINT" | "BIGINT" => {
                let value1: i64 = row1.get::<i64, _>(col1.ordinal());
                let value2: i64 = row2.get::<i64, _>(col2.ordinal());
                assert_eq!(value1, value2, "Column {} value mismatch", col1.name());
            }
            "TINYINT UNSIGNED" | "SMALLINT UNSIGNED" | "INT UNSIGNED" | "MEDIUMINT UNSIGNED"
            | "BIGINT UNSIGNED" => {
                let value1: u64 = row1.get::<u64, _>(col1.ordinal());
                let value2: u64 = row2.get::<u64, _>(col2.ordinal());
                assert_eq!(value1, value2, "Column {} value mismatch", col1.name());
            }
            "FLOAT" => {
                let value1: f32 = row1.get::<f32, _>(col1.ordinal());
                let value2: f32 = row2.get::<f32, _>(col2.ordinal());
                assert_eq!(value1, value2, "Column {} value mismatch", col1.name());
            }
            "DOUBLE" => {
                let value1: f64 = row1.get::<f64, _>(col1.ordinal());
                let value2: f64 = row2.get::<f64, _>(col2.ordinal());
                assert_eq!(value1, value2, "Column {} value mismatch", col1.name());
            }
            "NULL" => {
                assert_eq!(col1.type_info().is_null(), col2.type_info().is_null());
            }
            "DATETIME" => {
                let value1: NaiveDateTime = row1.get::<NaiveDateTime, _>(col1.ordinal());
                let value2: NaiveDateTime = row2.get::<NaiveDateTime, _>(col2.ordinal());
                assert_eq!(value1, value2, "Column {} value mismatch", col1.name());
            }
            "TIMESTAMP" => {
                let value1: DateTime<Utc> = row1.get::<DateTime<Utc>, _>(col1.ordinal());
                let value2: DateTime<Utc> = row2.get::<DateTime<Utc>, _>(col2.ordinal());
                assert_eq!(value1, value2, "Column {} value mismatch", col1.name());
            }
            "DATE" => {
                let value1: NaiveDate = row1.get::<NaiveDate, _>(col1.ordinal());
                let value2: NaiveDate = row2.get::<NaiveDate, _>(col2.ordinal());
                assert_eq!(value1, value2, "Column {} value mismatch", col1.name());
            }
            "TIME" => {
                let value1: NaiveTime = row1.get::<NaiveTime, _>(col1.ordinal());
                let value2: NaiveTime = row2.get::<NaiveTime, _>(col2.ordinal());
                assert_eq!(value1, value2, "Column {} value mismatch", col1.name());
            }
            "YEAR" => {
                let value1: u16 = row1.get::<u16, _>(col1.ordinal());
                let value2: u16 = row2.get::<u16, _>(col2.ordinal());
                assert_eq!(value1, value2, "Column {} value mismatch", col1.name());
            }
            "ENUM" | "JSON" => {}
            "SET" | "CHAR" | "VARCHAR" | "TINYTEXT" | "TEXT" | "MEDIUMTEXT" | "LONGTEXT" => {
                let value1: &str = row1.get::<&str, _>(col1.ordinal());
                let value2: &str = row2.get::<&str, _>(col2.ordinal());
                assert_eq!(value1, value2, "Column {} value mismatch", col1.name());
            }
            "TINYBLOB" | "BLOB" | "MEDIUMBLOB" | "LONGBLOB" => {
                let value1: Vec<u8> = row1.get::<Vec<u8>, _>(col1.ordinal());
                let value2: Vec<u8> = row2.get::<Vec<u8>, _>(col2.ordinal());
                assert_eq!(value1, value2, "Column {} value mismatch", col1.name());
            }
            "DECIMAL" => {
                let value1: BigDecimal = row1.get::<BigDecimal, _>(col1.ordinal());
                let value2: BigDecimal = row2.get::<BigDecimal, _>(col2.ordinal());
                assert_eq!(value1, value2, "Column {} value mismatch", col1.name());
            }
            _ => {}
        }
    }
    Ok(())
}
