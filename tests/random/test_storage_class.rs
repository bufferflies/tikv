// Copyright 2025 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    sync::{atomic::Ordering, Arc},
    time::Duration,
};

use anyhow::Context;
use collections::HashSet;
use kvengine::IdVer;
use rand::prelude::*;
use schema::schema::StorageClass;
use sqlx::Executor;
use test_cloud_server::{
    keyspace::{make_row_key, KeyspaceManager},
    table::TableMeta,
    tidb::TidbCluster,
    ServerCluster, TryWaiter,
};
use tikv_util::{codec::bytes::decode_bytes, debug, error, info, time::Instant};

use crate::{
    sql_util::retry_or_panic,
    test_tidb::{connect_tidb, query_tidb_table_schema},
    Running, ALTER_TABLE_IA_COUNTER, ALTER_TABLE_NON_IA_COUNTER, ASYNC_SHARD_COUNTER,
};

type Result<T> = anyhow::Result<T>;

pub(crate) async fn spawn_alter_storage_class(
    tc: TidbCluster,
    keyspace_manager: KeyspaceManager,
    tables: Vec<Arc<TableMeta>>,
    ia_table_ratio: f64,
    max_interval: Duration,
    running: Running,
) {
    let random = || {
        let mut rng = thread_rng();
        let interval = rng.gen_range(Duration::ZERO..=max_interval);
        let skip = !rng.gen_bool(ia_table_ratio);
        let table = tables.choose(&mut rng).unwrap();
        (interval, skip, table)
    };
    while running.get() {
        let (interval, skip, table) = random();
        tokio::time::sleep(interval).await;
        if skip {
            continue;
        }

        let target_sc = if table.storage_class() != StorageClass::Ia {
            StorageClass::Ia
        } else if thread_rng().gen_bool(0.5) {
            StorageClass::Standard
        } else {
            StorageClass::Unspecified
        };
        info!("alter storage class"; "table" => ?table, "target" => ?target_sc);
        retry_or_panic!(
            alter_storage_class(&tc, &keyspace_manager, table, target_sc)
                .await
                .context("alter_sc")
        );
        if target_sc == StorageClass::Ia {
            ALTER_TABLE_IA_COUNTER.fetch_add(1, Ordering::Relaxed);
        } else {
            ALTER_TABLE_NON_IA_COUNTER.fetch_add(1, Ordering::Relaxed);
        }
    }
}

async fn alter_storage_class(
    tc: &TidbCluster,
    keyspace_manager: &KeyspaceManager,
    table: &TableMeta,
    storage_class: StorageClass,
) -> Result<()> {
    let pool = connect_tidb(tc, keyspace_manager, table.keyspace_id()).await;
    let mut conn = pool.acquire().await.context("acquire")?;
    let storage_str = match storage_class {
        StorageClass::Unspecified => "",
        StorageClass::Standard => r#"{"storage_class": "STANDARD"}"#,
        StorageClass::Ia => r#"{"storage_class": "IA"}"#,
    };
    let expect_sc = match storage_class {
        StorageClass::Unspecified | StorageClass::Standard => StorageClass::Standard, /* TiDB will set STANDARD as default. */
        StorageClass::Ia => StorageClass::Ia,
    };

    let alter_table = format!(
        "ALTER TABLE `{}`.`{}` ENGINE_ATTRIBUTE = '{}'",
        table.db_name(),
        table.table_name(),
        storage_str
    );
    info!("alter table: {}", &alter_table);
    conn.execute(alter_table.as_str())
        .await
        .context("alter_table")?;

    let schema = query_tidb_table_schema(&mut conn, table.db_name(), table.table_name())
        .await
        .context("query_schema")?;
    assert_eq!(table.id(), schema.id);
    assert_eq!(expect_sc, schema.storage_class);
    table.set_storage_class(storage_class);
    Ok(())
}

pub(crate) fn check_storage_class(
    cluster: &ServerCluster,
    tables: &[Arc<TableMeta>],
    timeout: Duration,
) {
    let pd_client = cluster.get_pd_client_ext();
    let mut async_shards = 0;
    let mut success_shards: HashSet<IdVer> = HashSet::default();
    let mut last_errs: Vec<String> = vec![];

    let start_time = Instant::now_coarse();
    while start_time.saturating_elapsed() < timeout {
        last_errs.clear();
        for table in tables {
            let expect_sc = match table.storage_class() {
                StorageClass::Unspecified | StorageClass::Standard => StorageClass::Unspecified, /* CSE will convert STANDARD to UNSPECIFIED. */
                StorageClass::Ia => StorageClass::Ia,
            };
            let expect_sync = expect_sc.is_sync();

            let mut key = make_row_key(table.keyspace_id(), table.id(), b"");
            let end_key = make_row_key(table.keyspace_id(), table.id() + 1, b"");
            while key < end_key {
                debug!("check storage class";
                    "table" => ?table,
                    "expect_sc" => ?expect_sc,
                    "expect_sync" => expect_sync,
                    "start_key" => log_wrappers::Value::key(&key),
                    "end_key" => log_wrappers::Value::key(&end_key),
                );

                let mut next_key: Option<Vec<u8>> = None;
                let mut shard_id_ver: Option<IdVer> = None;
                let ok = TryWaiter::timeout(5).interval_dur(Duration::from_millis(500)).try_wait(
                    || {
                        let Some(shard) = cluster.get_active_shard_by_key(&key) else {
                            return false;
                        };
                        next_key = Some(shard.outer_end.to_vec());

                        let id_ver = IdVer::new(shard.id, shard.ver);
                        shard_id_ver = Some(id_ver);
                        if success_shards.contains(&id_ver) {
                            return true;
                        }

                        let snap = shard.new_snap_access();
                        let ok = expect_sc == shard.get_storage_class();
                        let check_sync = || {
                            if expect_sync {
                                snap.is_sync()
                            } else {
                                !snap.is_sync() || snap.write_cf_level_n_is_empty()
                            }
                        };
                        let ok = ok && check_sync();
                        info!("{} check storage class: ok: {}", id_ver, ok;
                            "table" => ?table, "snap" => ?snap.display(),
                            "expect_sc" => ?expect_sc, "sc" => ?shard.get_storage_class(),
                            "expect_sync" => expect_sync, "sync" => snap.is_sync(), "write_cf_level_n_is_empty" => snap.write_cf_level_n_is_empty(),
                        );
                        if ok {
                            success_shards.insert(id_ver);
                            if !expect_sync {
                                async_shards += 1;
                            }
                        }
                        ok
                    },
                );

                if !ok {
                    let snap = shard_id_ver.and_then(|id_ver| {
                        cluster
                            .get_active_shard(id_ver.id)
                            .map(|x| x.new_snap_access().display())
                    });
                    let err_msg = format!(
                        "{} check storage class failed: table: {:?}, shard: {:?}",
                        shard_id_ver.unwrap_or_default(),
                        table,
                        snap,
                    );
                    last_errs.push(err_msg);
                }

                key = match next_key {
                    Some(k) => k,
                    None => {
                        let region = pd_client.get_region(&key).unwrap_or_else(|err| {
                            panic!("get region failed: {:?}, {:?}", key, err)
                        });
                        decode_bytes(&mut region.end_key.as_slice(), false).unwrap()
                    }
                };
            }
        }

        if last_errs.is_empty() {
            break;
        }
    }

    if !last_errs.is_empty() {
        let errs_count = last_errs.len();
        for err_msg in last_errs {
            error!("{}", err_msg);
        }
        panic!("check storage class failed: {}", errs_count);
    }

    ASYNC_SHARD_COUNTER.fetch_add(async_shards, Ordering::Relaxed);
}
