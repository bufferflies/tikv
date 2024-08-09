// Copyright 2024 TiKV Project Authors. Licensed under Apache-2.0.

use std::{fmt, sync::atomic::Ordering::Relaxed, time::Duration};

use anyhow::{Context, Result};
use futures::future::join_all;
use rand::prelude::*;
use sqlx::{MySql, Pool, Row};
use test_cloud_server::{keyspace::KeyspaceManager, tidb::TidbCluster};
use tikv_util::{error, info, time::Instant};

use crate::{
    test_txn_file::{TXN_CHUNK_MAX_SIZE, TXN_FILE_MIN_SIZE},
    JEPSEN_BANK_TXN_COUNTER, JEPSEN_BANK_TXN_RETRY_COUNTER,
};

pub(crate) const JEPSEN_BANK_WORKLOAD_CONCURRENCY: usize = 4;
const BANK_DB_NAME: &str = "jepsen-bank";
const ACCOUNTS_TABLE_NAME: &str = "accounts";
const BANK_ACCOUNTS: usize = 10;
const BANK_TXN_FILE_RATIO: f64 = 0.5;

const MAX_PADDING_SIZE: usize = TXN_CHUNK_MAX_SIZE * 2;

// Ref: https://github.com/tidbcloud/tidb-cse/blob/release-7.1-keyspace/kv/error.go, TxnRetryableMark
const TIDB_TXN_RETRYABLE_MARK: &str = "[try again later]";
const DEADLOCK_ERR_MSG: &str = "Deadlock found";
const WRITE_CONFLICT_ERR_MSG: &str = "Write conflict";
const RETRYABLE_DB_ERR_MSGS: &[&str] = &[
    TIDB_TXN_RETRYABLE_MARK,
    WRITE_CONFLICT_ERR_MSG,
    DEADLOCK_ERR_MSG,
];

const TIFLASH_REPLICAS_AVAILABLE_TIMEOUT: Duration = Duration::from_secs(30);

fn is_db_error_retryable(err: &sqlx::Error) -> bool {
    match err {
        sqlx::Error::Database(db_err) => RETRYABLE_DB_ERR_MSGS
            .iter()
            .any(|msg| db_err.message().contains(msg)),
        _ => false,
    }
}

fn gen_padding(rng: &mut ThreadRng, buf: &mut [u8]) -> usize {
    static PADDING_LENS: [usize; 4] = [
        TXN_FILE_MIN_SIZE,
        TXN_CHUNK_MAX_SIZE / 2,
        TXN_CHUNK_MAX_SIZE,
        MAX_PADDING_SIZE,
    ];
    let len = PADDING_LENS.choose(rng).unwrap();
    rng.fill(&mut buf[..*len]);
    *len
}

pub(crate) async fn prepare_jepsen_bank(
    tc: TidbCluster,
    keyspace_manager: KeyspaceManager,
    keyspace_id: u32,
    tiflash_replicas: Option<usize>,
) {
    let keyspace_name = keyspace_manager
        .get_keyspace_meta(keyspace_id)
        .unwrap()
        .name();
    let tag = format!("bank-{}[{}]", keyspace_id, keyspace_name);
    let tidb_idx = TidbCluster::get_idx_by_keyspace_name(&keyspace_name);
    let params = tc.tidb.conn_params(tidb_idx);

    let conn_string = params.conn_string("test");
    let pool = sqlx::MySqlPool::connect(&conn_string).await.unwrap();

    info!("{} prepare_jepsen_bank", tag);

    let mut sqls = vec![
        format!("drop database if exists `{BANK_DB_NAME}`"),
        format!("create database `{BANK_DB_NAME}`"),
        format!(
            "create table `{BANK_DB_NAME}`.`{ACCOUNTS_TABLE_NAME}` \
            (id int not null primary key, \
            balance int not null default 0, \
            padding varbinary({MAX_PADDING_SIZE}) not null default 0x0)"
        ),
    ];
    if let Some(tiflash_replicas) = tiflash_replicas {
        sqls.push(format!(
            "alter table `{BANK_DB_NAME}`.`{ACCOUNTS_TABLE_NAME}` set tiflash replica {tiflash_replicas}"
        ));
    }
    for account_id in 0..BANK_ACCOUNTS {
        sqls.push(format!(
            "insert into `{BANK_DB_NAME}`.`{ACCOUNTS_TABLE_NAME}` (id) values ({account_id})"
        ));
    }
    for sql in sqls {
        info!("{} prepare_jepsen_bank", tag; "sql" => &sql);
        sqlx::query(&sql).execute(&pool).await.unwrap();
    }

    if tiflash_replicas.is_some() {
        wait_tiflash_replicas_available(
            &tag,
            &pool,
            BANK_DB_NAME,
            ACCOUNTS_TABLE_NAME,
            TIFLASH_REPLICAS_AVAILABLE_TIMEOUT,
        )
        .await;
    }
}

pub(crate) async fn run_jepsen_bank(
    tc: TidbCluster,
    keyspace_manager: KeyspaceManager,
    keyspace_id: u32,
    jepsen_use_txn_file: bool,
    use_tiflash: bool,
    timeout: Duration,
) {
    info!("run_jepsen_bank"; "use_txn_file" => jepsen_use_txn_file, "use_tiflash" => use_tiflash);
    let keyspace_name = keyspace_manager
        .get_keyspace_meta(keyspace_id)
        .unwrap()
        .name();
    let tidb_idx = TidbCluster::get_idx_by_keyspace_name(&keyspace_name);
    let params = tc.tidb.conn_params(tidb_idx);
    let conn_string = params.conn_string(BANK_DB_NAME);
    let pool = sqlx::MySqlPool::connect(&conn_string).await.unwrap();

    let mut handles = Vec::with_capacity(JEPSEN_BANK_WORKLOAD_CONCURRENCY);
    for tid in 0..JEPSEN_BANK_WORKLOAD_CONCURRENCY {
        let pool = pool.clone();
        let handle = tokio::spawn(async move {
            let tag = format!("bank-{}-{}", keyspace_id, tid);

            let mut padding = [0u8; MAX_PADDING_SIZE];
            let start_time = Instant::now();
            while start_time.saturating_elapsed() < timeout {
                let mut random = || {
                    let mut rng = thread_rng();
                    let from_to = (0..BANK_ACCOUNTS).choose_multiple(&mut rng, 2);
                    let amount = rng.gen_range(0..100);
                    let padding_len = gen_padding(&mut rng, &mut padding);
                    let use_txn_file = if jepsen_use_txn_file {
                        rng.gen_bool(BANK_TXN_FILE_RATIO)
                    } else {
                        false
                    };
                    (from_to[0], from_to[1], amount, padding_len, use_txn_file)
                };
                let (from, to, amount, padding_len, use_txn_file) = random();
                let padding = &padding[..padding_len];

                let ok = bank_transfer(&tag, &pool, from, to, amount, use_txn_file, padding)
                    .await
                    .unwrap_or_else(|err| {
                        panic!("{} bank_transfer error: {:?}", tag, err);
                    });
                if ok {
                    JEPSEN_BANK_TXN_COUNTER.fetch_add(1, Relaxed);
                } else {
                    JEPSEN_BANK_TXN_RETRY_COUNTER.fetch_add(1, Relaxed);
                }
            }
        });
        handles.push(handle);
    }

    let pool_copy = pool.clone();
    handles.push(tokio::spawn(async move {
        let start_time = Instant::now();
        while start_time.saturating_elapsed() < timeout {
            verify_bank_accounts(&pool_copy, false).await.unwrap();
            if use_tiflash {
                verify_bank_accounts(&pool_copy, true).await.unwrap();
            }
            tokio::time::sleep(Duration::from_secs(1)).await;
        }
    }));

    join_all(handles).await;

    verify_bank_accounts(&pool, false).await.unwrap();
    let accounts = list_bank_accounts(&pool, false).await.unwrap();
    info!("accounts: {:?}", accounts);
    if use_tiflash {
        verify_bank_accounts(&pool, true).await.unwrap();
        let tiflash_accounts = list_bank_accounts(&pool, true).await.unwrap();
        info!("accounts from TiFlash: {:?}", accounts);
        assert_eq!(accounts, tiflash_accounts);
    }
}

async fn bank_transfer(
    tag: &str,
    pool: &Pool<MySql>,
    from: usize,
    to: usize,
    amount: i32,
    use_txn_file: bool,
    padding: &[u8],
) -> Result<bool> {
    let txn_mode = if use_txn_file {
        "optimistic"
    } else {
        "pessimistic"
    };
    let hex_padding = hex::encode(padding);
    let sqls = vec![
        format!("begin {txn_mode}"),
        format!(
            "update `{BANK_DB_NAME}`.`{ACCOUNTS_TABLE_NAME}` \
                        set balance = balance - {amount}, padding = 0x{hex_padding} \
                        where id = {from}",
        ),
        format!(
            "update `{BANK_DB_NAME}`.`{ACCOUNTS_TABLE_NAME}` \
                        set balance = balance + {amount}, padding = 0x{hex_padding} \
                        where id = {to}",
        ),
    ];

    let mut conn = pool.acquire().await.context("acquire")?;
    for sql in &sqls {
        info!("{} bank_transfer: executing sql", tag; "sql" => sql);
        match sqlx::query(sql).execute(&mut conn).await {
            Ok(_) => {}
            Err(sqlx::Error::Database(err)) if err.message().contains(DEADLOCK_ERR_MSG) => {
                info!("{} bank_transfer: ignore deadlock, retry", tag; "sql" => sql, "err" => ?err);
                return Ok(false);
            }
            Err(err) => {
                error!("{} bank_transfer: execute failed", tag; "sql" => sql, "err" => ?err);
                return Err(err).with_context(|| format!("bank_transfer: sql: {}", sql));
            }
        }
    }
    match sqlx::query("commit").execute(&mut conn).await {
        Ok(_) => Ok(true),
        Err(err) if is_db_error_retryable(&err) => {
            info!("{} bank_transfer: commit ignore retryable error", tag; "err" => ?err);
            Ok(false)
        }
        Err(err) => {
            error!("{} bank_transfer: commit failed", tag; "err" => ?err);
            Err(err).context("bank_transfer commit")
        }
    }
}

async fn verify_bank_accounts(pool: &Pool<MySql>, use_tiflash: bool) -> Result<()> {
    let mut tx = pool.begin().await.context("begin")?;
    let engine_hint = get_engine_hint(use_tiflash, ACCOUNTS_TABLE_NAME);
    // Cast sum to "signed", as sum() return Decimal which is not easy to handle.
    let sql = format!(
        "select {engine_hint} cast(sum(balance) as signed) as sum from `{BANK_DB_NAME}`.`{ACCOUNTS_TABLE_NAME}`"
    );
    let row = sqlx::query(&sql)
        .fetch_one(&mut tx)
        .await
        .context("select sum")?;
    let sum: i32 = row.get("sum");
    if sum != 0 {
        let accounts = list_bank_accounts(&mut tx, use_tiflash).await?;
        panic!("sum is not zero: {}, accounts {:?}", sum, accounts);
    }
    Ok(())
}

#[derive(PartialEq)]
struct Account {
    id: i32,
    balance: i32,
    padding: Vec<u8>,
}

impl fmt::Debug for Account {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Account")
            .field("id", &self.id)
            .field("balance", &self.balance)
            .field("padding", &hex::encode(&self.padding))
            .finish()
    }
}

async fn list_bank_accounts<'a, E>(executor: E, use_tiflash: bool) -> Result<Vec<Account>>
where
    E: sqlx::Executor<'a, Database = MySql>,
{
    let engine_hint = get_engine_hint(use_tiflash, ACCOUNTS_TABLE_NAME);
    let sql = format!(
        "select {engine_hint} id, balance, padding from `{BANK_DB_NAME}`.`{ACCOUNTS_TABLE_NAME}` order by id"
    );
    let rows = sqlx::query(&sql)
        .fetch_all(executor)
        .await
        .context("select all")?;
    assert_eq!(rows.len(), BANK_ACCOUNTS);
    let accounts = rows
        .into_iter()
        .map(|row| {
            let id: i32 = row.get("id");
            let balance: i32 = row.get("balance");
            let mut padding: Vec<u8> = row.get("padding");
            padding.truncate(16);
            Account {
                id,
                balance,
                padding,
            }
        })
        .collect::<Vec<_>>();
    Ok(accounts)
}

// Hint: /*+ READ_FROM_STORAGE(TIFLASH[t1], TIKV[t2]) */
fn get_engine_hint(use_tiflash: bool, tb: &str) -> String {
    let engine = if use_tiflash { "TIFLASH" } else { "TIKV" };
    format!("/*+ READ_FROM_STORAGE({engine}[{tb}]) */")
}

async fn query_tiflash_progress<'a, E>(
    executor: E,
    db: &str,
    tb: &str,
) -> Result<(bool /* available */, f64 /* progress */)>
where
    E: sqlx::Executor<'a, Database = MySql>,
{
    let sql = format!(
        "select available, progress from information_schema.tiflash_replica where TABLE_SCHEMA='{db}' and TABLE_NAME='{tb}'"
    );
    let row = sqlx::query(&sql).fetch_one(executor).await.context(sql)?;
    let available: i32 = row.get("available");
    let progress: f64 = row.get("progress");
    Ok((available != 0, progress))
}

async fn wait_tiflash_replicas_available(
    tag: &str,
    pool: &sqlx::pool::Pool<MySql>,
    db: &str,
    tb: &str,
    timeout: Duration,
) {
    let start = Instant::now_coarse();
    while start.saturating_elapsed() < timeout {
        let (available, progress) = query_tiflash_progress(pool, db, tb).await.unwrap();
        if available {
            info!("{} TiFlash replicas available", tag; "db" => db, "tb" => tb, "progress" => progress);
            return;
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
    let (available, progress) = query_tiflash_progress(pool, db, tb).await.unwrap();
    panic!(
        "{} TiFlash replicas not available, db {}, tb {}, available {}, progress {}",
        tag, db, tb, available, progress,
    );
}
