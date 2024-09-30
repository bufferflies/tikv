// Copyright 2024 TiKV Project Authors. Licensed under Apache-2.0.

//! Test for the scene that data have unique constraint.

use std::{fmt, sync::atomic::Ordering::Relaxed, time::Duration};

use anyhow::{Context, Result};
use futures::future::join_all;
use rand::prelude::*;
use sqlx::{Connection, MySql, Row};
use test_cloud_server::{keyspace::KeyspaceManager, tidb::TidbCluster};
use tikv_util::{debug, error, info, time::Instant, warn};

use crate::{
    sql_util::{
        gen_padding, is_duplicate_entry_err, is_error_retryable, retry_or_panic, Transaction,
        MAX_PADDING_SIZE,
    },
    UNIQUE_WORKLOAD_CONFLICT_COUNTER, UNIQUE_WORKLOAD_TXN_COUNTER,
};

const UNIQUE_WORKLOAD_CONCURRENCY: usize = 4;
const UNIQUE_DB_NAME: &str = "uniq";
const UNIQUE_TABLE_NAME: &str = "rows";

const VALUE0_MAX: i32 = 1000;
const VALUE1_MAX: i32 = VALUE0_MAX * 10; // To generate index key in a wider range.

/// 0.1% stock data which are inserted during preparation and will not be
/// deleted. So the probability of conflict for a insert with 100 rows will be
/// about (1-0.1%)^100 = 90%
const STOCK_DATA_NUM: i32 = VALUE0_MAX / 1000;

const INSERT_BATCH_SIZE_OPTS: &[i32] = &[
    VALUE0_MAX / 100,
    VALUE0_MAX / 20,
    VALUE0_MAX / 10,
    VALUE0_MAX / 5,
]; // 10, 50, 100, 200

pub(crate) async fn prepare_unique_workload(
    tc: TidbCluster,
    keyspace_manager: KeyspaceManager,
    keyspace_id: u32,
) {
    let keyspace_name = keyspace_manager
        .get_keyspace_meta(keyspace_id)
        .unwrap()
        .name();
    let tag = format!("unique-{}[{}]", keyspace_id, keyspace_name);
    let tidb_idx = TidbCluster::get_idx_by_keyspace_name(&keyspace_name);
    let params = tc.tidb.conn_params(tidb_idx);

    let conn_string = params.conn_string("test");
    let pool = sqlx::MySqlPool::connect(&conn_string).await.unwrap();

    info!("{} prepare_unique_workload", tag);

    let mut sqls = vec![
        format!("drop database if exists `{UNIQUE_DB_NAME}`"),
        format!("create database `{UNIQUE_DB_NAME}`"),
        format!(
            "create table `{UNIQUE_DB_NAME}`.`{UNIQUE_TABLE_NAME}` \
            (id int not null auto_increment, \
            val0 int not null, \
            val1 int not null, \
            padding varbinary({MAX_PADDING_SIZE}) not null default 0x0, \
            primary key (id), \
            unique key `val0_unique_idx` (val0), \
            key `val1_idx` (val1))"
        ),
    ];

    {
        // Insert stock data.
        let mut rng = thread_rng();
        let mut padding = [0u8; MAX_PADDING_SIZE];
        let padding_len = gen_padding(STOCK_DATA_NUM as usize, &mut rng, &mut padding);
        let hex_padding = hex::encode(&padding[..padding_len]);

        let mut values = Vec::with_capacity(STOCK_DATA_NUM as usize);
        for val0 in 0..STOCK_DATA_NUM {
            let val1 = rng.gen_range(0..VALUE1_MAX);
            values.push(format!("({val0}, {val1}, 0x{hex_padding})"));
        }

        let sql = format!(
            "insert into `{UNIQUE_DB_NAME}`.`{UNIQUE_TABLE_NAME}` (val0, val1, padding) values {}",
            values.join(", ")
        );
        sqls.push(sql);
    }

    for sql in sqls {
        info!("{} prepare_unique_workload", tag; "sql" => &sql);
        sqlx::query(&sql).execute(&pool).await.unwrap();
    }
}

pub(crate) async fn run_unique_workload(
    tc: TidbCluster,
    keyspace_manager: KeyspaceManager,
    keyspace_id: u32,
    use_txn_file: bool,
    timeout: Duration,
) {
    info!("run_unique_workload"; "use_txn_file" => use_txn_file);
    let keyspace_name = keyspace_manager
        .get_keyspace_meta(keyspace_id)
        .unwrap()
        .name();
    let tidb_idx = TidbCluster::get_idx_by_keyspace_name(&keyspace_name);
    let params = tc.tidb.conn_params(tidb_idx);
    let conn_string = params.conn_string(UNIQUE_DB_NAME);

    let mut handles = Vec::with_capacity(UNIQUE_WORKLOAD_CONCURRENCY);
    for tid in 0..UNIQUE_WORKLOAD_CONCURRENCY {
        let conn_string = conn_string.clone();
        let handle = tokio::spawn(async move {
            let tag = format!("unique-{}-{}", keyspace_id, tid);

            let mut padding = [0u8; MAX_PADDING_SIZE];
            let start_time = Instant::now();
            while start_time.saturating_elapsed() < timeout {
                let (insert_sql, delete_sql) = {
                    let mut rng = thread_rng();

                    let mut contains_stock_data = false;

                    let mut val0 = rng.gen_range(0..VALUE0_MAX);
                    let batch_size = *INSERT_BATCH_SIZE_OPTS.choose(&mut rng).unwrap() as usize;
                    // Generate padding based on 2 x smallest batch size, to generate hybrid
                    // workload with both pessimistic & txn file transactions.
                    let padding_len = gen_padding(
                        INSERT_BATCH_SIZE_OPTS[0] as usize * 2,
                        &mut rng,
                        &mut padding,
                    );
                    let hex_padding = hex::encode(&padding[..padding_len]);

                    let mut val0s = Vec::with_capacity(batch_size);
                    let mut values = Vec::with_capacity(batch_size);
                    for _ in 0..batch_size {
                        if val0 < STOCK_DATA_NUM {
                            contains_stock_data = true;
                        }
                        if !contains_stock_data {
                            val0s.push(val0);
                        }

                        let val1 = rng.gen_range(0..VALUE1_MAX);
                        values.push(format!("({val0}, {val1}, 0x{hex_padding})"));

                        val0 = (val0 + 1) % VALUE0_MAX;
                    }

                    let insert_sql = format!(
                        "insert into `{UNIQUE_DB_NAME}`.`{UNIQUE_TABLE_NAME}` (val0, val1, padding) values {}",
                        values.join(", ")
                    );
                    let delete_sql: Option<String> = (!contains_stock_data).then(|| format!(
                        "delete from `{UNIQUE_DB_NAME}`.`{UNIQUE_TABLE_NAME}` where val0 in ({})",
                        val0s
                            .iter()
                            .map(|v| v.to_string())
                            .collect::<Vec<_>>()
                            .join(", ")
                    ));
                    (insert_sql, delete_sql)
                };

                let expect_conflict = delete_sql.is_none();
                match do_sql(&tag, &conn_string, &insert_sql, use_txn_file).await {
                    Ok(()) => {
                        if expect_conflict {
                            let mut conn =
                                sqlx::MySqlConnection::connect(&conn_string).await.unwrap();
                            match list_rows(&mut conn).await {
                                Ok((rows, _)) => {
                                    info!("{} unique workload list rows", tag; "rows" => ?rows);
                                }
                                Err(err) => {
                                    warn!("{} unique workload list rows failed: {:?}", tag, err);
                                }
                            }
                            panic!(
                                "{} unique workload insert should meet conflict, sql {}",
                                tag, insert_sql
                            );
                        }
                        UNIQUE_WORKLOAD_TXN_COUNTER.fetch_add(1, Relaxed);
                    }
                    Err(err) if is_duplicate_entry_err(&err) => {
                        info!("{} unique workload meet conflict", tag; "err" => ?err, "expect_conflict" => expect_conflict);
                        UNIQUE_WORKLOAD_CONFLICT_COUNTER.fetch_add(1, Relaxed);
                    }
                    Err(err) if is_error_retryable(&err) => {
                        warn!("{} unique workload perform insert failed, retry", tag; "err" => ?err);
                    }
                    Err(err) => panic!(
                        "{} unique workload perform insert failed: {:?}, sql: {}",
                        tag, err, insert_sql
                    ),
                }

                if let Some(delete_sql) = delete_sql {
                    match do_sql(&tag, &conn_string, &delete_sql, use_txn_file).await {
                        Ok(()) => {}
                        Err(err) if is_error_retryable(&err) => {
                            warn!("{} unique workload perform delete failed, retry", tag; "err" => ?err);
                        }
                        Err(err) => {
                            panic!("{} unique workload perform delete failed: {:?}", tag, err)
                        }
                    }
                }
            }
        });
        handles.push(handle);
    }

    let conn_string_copy = conn_string.clone();
    handles.push(tokio::spawn(async move {
        let start_time = Instant::now();
        while start_time.saturating_elapsed() < timeout {
            retry_or_panic!(
                verify_unique(&conn_string_copy)
                    .await
                    .context("verify_unique")
            );
            tokio::time::sleep(Duration::from_secs(1)).await;
        }
    }));

    join_all(handles).await;

    let mut conn = sqlx::MySqlConnection::connect(&conn_string).await.unwrap();
    verify_unique(&conn_string).await.unwrap();
    let (rows, _) = list_rows(&mut conn).await.unwrap();
    info!("unique rows: {:?}", rows);
}

async fn do_sql(tag: &str, conn_string: &str, sql: &str, optimistic_txn: bool) -> Result<()> {
    let mut txn = Transaction::begin(tag, conn_string, optimistic_txn)
        .await
        .context("begin")?;
    debug!("{} unique_workload: do sql", tag; "sql" => sql, "start_ts" => txn.start_ts());
    sqlx::query(sql)
        .execute(txn.conn())
        .await
        .context(format!("query: {sql}"))?;
    txn.commit().await.context("do_sql_commit")
}

async fn verify_unique(conn_string: &str) -> Result<()> {
    let mut conn = sqlx::MySqlConnection::connect(conn_string)
        .await
        .context("connect")?;
    sqlx::query("begin")
        .execute(&mut conn)
        .await
        .context("verify_begin")?;
    let sql = format!(
        "select val0, count(*) as cnt, \
         TIDB_CURRENT_TSO() as tso \
         from `{UNIQUE_DB_NAME}`.`{UNIQUE_TABLE_NAME}` \
         group by val0 having cnt > 1 "
    );
    let rows = sqlx::query(&sql)
        .fetch_all(&mut conn)
        .await
        .context("select sum")?;
    if !rows.is_empty() {
        for row in rows {
            let val0: i32 = row.get("val0");
            let cnt: i64 = row.get("cnt");
            let tso: i64 = row.get("tso");
            error!("unique value has duplicated rows"; "val0" => val0, "cnt" => cnt, "tso" => tso);
        }

        let (rows, tso) = list_rows(&mut conn).await?;
        error!("duplicated unique rows"; "rows" => ?rows, "tso" => tso);

        panic!("unique value duplicated rows");
    }
    Ok(())
}

async fn list_rows<'a, E>(executor: E) -> Result<(Vec<UniqueRow>, i64 /* tso */)>
where
    E: sqlx::Executor<'a, Database = MySql>,
{
    let sql = format!(
        "select id, val0, val1, padding, TIDB_CURRENT_TSO() as tso from `{UNIQUE_DB_NAME}`.`{UNIQUE_TABLE_NAME}` order by id"
    );
    let mut tso: Option<i64> = None;
    let rows = sqlx::query(&sql)
        .fetch_all(executor)
        .await
        .context("select all")?;
    let unique_rows = rows
        .into_iter()
        .map(|row| {
            let id: i32 = row.get("id");
            let val0: i32 = row.get("val0");
            let val1: i32 = row.get("val1");
            let mut padding: Vec<u8> = row.get("padding");
            padding.truncate(16);
            let _ = tso.get_or_insert_with(|| row.get("tso"));
            UniqueRow {
                id,
                val0,
                val1,
                padding,
            }
        })
        .collect::<Vec<_>>();
    Ok((unique_rows, tso.unwrap()))
}

#[derive(PartialEq)]
struct UniqueRow {
    id: i32,
    val0: i32,
    val1: i32,
    padding: Vec<u8>,
}

impl fmt::Debug for UniqueRow {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("UniqueRow")
            .field("id", &self.id)
            .field("val0", &self.val0)
            .field("val1", &self.val1)
            .field("padding", &hex::encode(&self.padding))
            .finish()
    }
}
