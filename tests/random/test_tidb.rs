// Copyright 2023 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    ops::Div,
    path::PathBuf,
    sync::{atomic::Ordering, Arc},
    time::Duration,
};

use kvengine::dfs::DFSConfig;
use pd_client::{
    pd_control,
    pd_control::{OpKind, PdScheduleConfig},
};
use rand::prelude::*;
use security::SecurityConfig;
use test_cloud_server::{
    oss::prepare_dfs, tidb::*, tikv_worker_cop_url, tpc::*, try_wait_async, ServerCluster,
};
use test_pd_client::PdWrapper;
use tikv::config::TikvConfig;
use tikv_util::{
    config::{ReadableDuration, ReadableSize},
    info,
    time::Instant,
};

use crate::{
    test_jepsen::*,
    test_txn_file::{TXN_CHUNK_MAX_SIZE, TXN_FILE_MIN_SIZE},
    *,
};

const REGION_SIZE: ReadableSize = ReadableSize::mb(1);
// TiDB has records with 200kb+ size (see "mysql.stats_history"), so set bucket
// size to 256kb.
const REGION_BUCKET_SIZE: ReadableSize = ReadableSize::kb(256);
// Ref: https://docs.pingcap.com/tidb/stable/pd-configuration-file#split-merge-interval
const SPLIT_MERGE_INTERVAL: ReadableDuration = ReadableDuration::secs(10);

// Test on only one keyspace to simulate a heavy tenant. Scenes of multiple
// keyspaces are covered by test_random_all.
const INITIAL_KEYSPACE_COUNT: usize = 1;
const NODES_COUNT: usize = 4;
const TEST_DURATION: Duration = Duration::from_secs(120); // Test for longer as TiDB bootstrap may cost 30s+.

const TIKV_WORKERS_COUNT: usize = 2;
const TIKV_WORKERS_THREADS_COUNT: usize = 2;

const PD_COUNT: usize = 1;
const PD_BIN_ENV_KEY: &str = "PD_BIN";
const PD_PORT_ENV_KEY: &str = "PD_PORT";
const PD_PORT_DEFAULT: u16 = 2379;
const PD_HEALTHY_TIMEOUT: Duration = Duration::from_secs(30);
const PD_TSO_SVC_COUNT: usize = 2;

const TIDB_BIN_ENV_KEY: &str = "TIDB_BIN";
const TIDB_PORT_ENV_KEY: &str = "TIDB_PORT";
const TIDB_PORT_DEFAULT: u16 = 4000;
const TIDB_STATUS_PORT_ENV_KEY: &str = "TIDB_STATUS_PORT";
const TIDB_STATUS_PORT_DEFAULT: u16 = 10080;
const TIDB_HEALTHY_TIMEOUT: Duration = Duration::from_secs(120);
const TIDB_LOG_LEVEL: &str = "info";

const TIFLASH_SWITCH_ENV_KEY: &str = "USE_TIFLASH";
const TIFLASH_BIN_ENV_KEY: &str = "TIFLASH_BIN";
const TIFLASH_SERVER_COUNT: usize = 1;
const TIFLASH_HEALTHY_TIMEOUT: Duration = Duration::from_secs(120);

const TPC_WORKLOAD_SWITCH_ENV_KEY: &str = "TPC_WORKLOAD";
const TPC_BIN_ENV_KEY: &str = "TPC_BIN";
const TPC_USE_TXN_FILE_RATIO: f64 = 0.5;
const TPCC_WAREHOUSES: usize = 2;
const TPCC_MAX_PROCS: usize = 1;
const TPCC_THREADS: usize = 4; // Number of threads for each TPCC workload.
const TPCC_RUN_DURATION: Duration = Duration::from_secs(10); // Duration of each TPCC run.
const TPCC_WORKLOAD_CONCURRENCY: usize = 1;

const JEPSEN_WORKLOAD_SWITCH_ENV_KEY: &str = "JEPSEN_WORKLOAD";
const JEPSEN_WORKLOAD_USE_TXN_FILE_ENV_KEY: &str = "JEPSEN_TXN_FILE";
const JEPSEN_WORKLOAD_KEYSPACE: u32 = 1; // Keyspace starts from 1.

const VERIFY_HEALTHY_TIMEOUT: Duration = Duration::from_secs(120);

const ENABLE_INNER_KEY_OFF_RATIO: f64 = 0.8; // 80% chance to enable inner key offset

const USE_REMOTE_COP_ENV_KEY: &str = "USE_REMOTE_COP";

#[test]
fn test_random_with_tidb() {
    let _logger_guard = test_util::init_log_for_test_async();
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .worker_threads(4)
        .thread_name("random-workload")
        .build()
        .unwrap();
    let _guard = runtime.enter();

    let enable_inner_key_off: bool = thread_rng().gen_bool(ENABLE_INNER_KEY_OFF_RATIO);
    let use_remote_cop = env_switch(USE_REMOTE_COP_ENV_KEY);
    info!("switches";
        "enable_inner_key_off" => enable_inner_key_off,
        "use_remote_cop" => use_remote_cop,
    );

    // Prepare.
    let (_temp_dir, _oss, dfs_config) = prepare_dfs("oss_");
    let security_conf = new_security_config();
    let tc = prepare_tidb_cluster(&security_conf);
    let mut cluster = prepare_cluster(
        &dfs_config,
        &security_conf,
        NODES_COUNT,
        INITIAL_KEYSPACE_COUNT,
        enable_inner_key_off,
        use_remote_cop,
        Some(&tc),
    );
    let pd_client = cluster.get_pd_client_ext();
    let pd_ctl = Arc::new(cluster.get_pd_control().unwrap());
    let keyspace_manager = cluster.keyspace_manager().clone();

    let tpc_bin = std::env::var(TPC_BIN_ENV_KEY).expect("env TPC_BIN is not set");
    check_tpc_binary(&tpc_bin);

    let start_tidb = {
        let tc = tc.clone();
        let tikv_worker_addr = cluster.tikv_worker_endpoints().pop().unwrap();
        runtime.spawn(async move {
            tc.start_tidb(
                INITIAL_KEYSPACE_COUNT as u16,
                TIDB_HEALTHY_TIMEOUT,
                TIDB_LOG_LEVEL,
                StartTidbOptions {
                    tikv_worker_addr,
                    txn_chunk_max_size: TXN_CHUNK_MAX_SIZE as u64,
                    txn_file_min_mutation_size: Some(TXN_FILE_MIN_SIZE as u64),
                },
            )
            .await
        })
    };
    let start_tiflash = {
        let tc = tc.clone();
        runtime.spawn_blocking(move || {
            tc.start_tiflash(
                TIFLASH_SERVER_COUNT as u16,
                &dfs_config,
                TIFLASH_HEALTHY_TIMEOUT,
            );
        })
    };
    let (start_tidb, start_tiflash) =
        runtime.block_on(async move { futures::join!(start_tidb, start_tiflash) });
    start_tidb.unwrap();
    start_tiflash.unwrap();

    let tiflash_switch_on = env_switch(TIFLASH_SWITCH_ENV_KEY);
    let tpc_switch_on = env_switch(TPC_WORKLOAD_SWITCH_ENV_KEY);
    let jepsen_switch_on = env_switch(JEPSEN_WORKLOAD_SWITCH_ENV_KEY);
    let jepsen_use_txn_file = env_switch(JEPSEN_WORKLOAD_USE_TXN_FILE_ENV_KEY);

    let mut rng = thread_rng();
    let tpc_use_txn_file = rng.gen_bool(TPC_USE_TXN_FILE_RATIO);

    let mut prepare_tasks = vec![];
    if tpc_switch_on {
        let all_keyspaces = keyspace_manager.get_all_keyspaces();
        prepare_tasks.push(runtime.spawn(prepare_tpcc(
            tc.clone(),
            keyspace_manager.clone(),
            tpc_bin.clone(),
            all_keyspaces,
            tpc_use_txn_file,
        )));
    }
    if jepsen_switch_on {
        prepare_tasks.push(runtime.spawn(prepare_jepsen_bank(
            tc.clone(),
            keyspace_manager.clone(),
            JEPSEN_WORKLOAD_KEYSPACE,
            tiflash_switch_on.then_some(TIFLASH_SERVER_COUNT),
        )));
    }
    runtime.block_on(futures::future::join_all(prepare_tasks));

    let mut async_handles = vec![];
    if tpc_switch_on {
        for tpc_idx in 0..TPCC_WORKLOAD_CONCURRENCY {
            async_handles.push(spawn_tpcc(
                tc.clone(),
                keyspace_manager.clone(),
                &tpc_bin,
                tpc_idx,
                TPCC_RUN_DURATION,
                TEST_DURATION,
            ));
        }
    }
    if jepsen_switch_on {
        async_handles.push(runtime.spawn(run_jepsen_bank(
            tc.clone(),
            keyspace_manager,
            JEPSEN_WORKLOAD_KEYSPACE,
            jepsen_use_txn_file,
            tiflash_switch_on,
            TEST_DURATION,
        )));
    }

    assert!(!async_handles.is_empty(), "no workload to run");
    async_handles.push(spawn_restart_tso_svc(
        tc.clone(),
        Duration::from_secs(10),
        TEST_DURATION,
    ));

    // Main loop.
    let start_time = Instant::now();
    while start_time.saturating_elapsed() < TEST_DURATION {
        // Restart nodes.
        random_node_restart(&mut cluster);
    }

    // Finish.
    runtime.block_on(async {
        info!("test finished, stopping all workers");
        for handle in async_handles {
            handle.await.unwrap();
        }

        // Make stats stable
        info!("stop TiDB and schedulers");
        tc.pd.must_healthy(VERIFY_HEALTHY_TIMEOUT).await;
        tc.tidb.must_all_healthy(VERIFY_HEALTHY_TIMEOUT).await;
        tc.tiflash.must_all_healthy(VERIFY_HEALTHY_TIMEOUT).await;
        tc.tidb.stop_all(); // To stop background tasks.
        tc.tiflash.stop_all();
        stop_schedulers(pd_ctl).await;

        info!("verify cluster");
        verify_cluster(&mut cluster, tpc_switch_on, jepsen_switch_on).await;
    });

    // Stop cluster.
    info!("stopping cluster");
    cluster.stop();

    // Statistics.
    let total_write_count = WRITE_COUNTER.load(Ordering::SeqCst);
    let total_keyspace_count = KEYSPACE_COUNTER.load(Ordering::SeqCst);
    let total_table_count = TABLE_COUNTER.load(Ordering::SeqCst);
    let total_drop_table_count = DROP_TABLE_COUNTER.load(Ordering::SeqCst);
    let total_merge_count = MERGE_COUNTER.load(Ordering::SeqCst);
    let total_move_count = MOVE_COUNTER.load(Ordering::SeqCst);
    let total_transfer_count = TRANSFER_COUNTER.load(Ordering::SeqCst);
    let total_node_restart = NODE_RESTART_COUNTER.load(Ordering::SeqCst);
    let total_backup_count = BACKUP_COUNTER.load(Ordering::SeqCst);
    let total_restore_count = RESTORE_COUNTER.load(Ordering::SeqCst);
    let total_load_data_count = LOAD_DATA_COUNTER.load(Ordering::SeqCst);
    let total_manual_major_compact = MANUAL_MAJOR_COMPACT_COUNTER.load(Ordering::SeqCst);
    let total_gc_resolved_locks = GC_ADVANCE_SAFE_POINT_COUNTER.load(Ordering::SeqCst);
    let total_tpcc_txns = TPCC_COUNTER.load(Ordering::SeqCst);
    let total_jepsen_bank = JEPSEN_BANK_TXN_COUNTER.load(Ordering::SeqCst);
    let total_jepsen_bank_retry = JEPSEN_BANK_TXN_RETRY_COUNTER.load(Ordering::SeqCst);
    let region_number = pd_client.get_regions_number();
    info!(
        "TEST SUCCEED: write {}, keyspace {}, table {}, drop table {}, region {}, merge {}, move {}, transfer {}, node restart {}, backup {}, restore {}, load_data {}, manual_major_compact {}, gc {}, tpcc {}, jepsen_bank {} (retry {})",
        total_write_count,
        total_keyspace_count,
        total_table_count,
        total_drop_table_count,
        region_number,
        total_merge_count,
        total_move_count,
        total_transfer_count,
        total_node_restart,
        total_backup_count,
        total_restore_count,
        total_load_data_count,
        total_manual_major_compact,
        total_gc_resolved_locks,
        total_tpcc_txns,
        total_jepsen_bank,
        total_jepsen_bank_retry,
    );

    tc.pd.stop_all();
}

fn prepare_tidb_cluster(security_config: &SecurityConfig) -> TidbCluster {
    let pd_bin = std::env::var(PD_BIN_ENV_KEY).expect("env PD_BIN is not set");
    let pd_port_base = std::env::var(PD_PORT_ENV_KEY)
        .map(|s| s.parse().unwrap())
        .unwrap_or(PD_PORT_DEFAULT);

    let tidb_bin = std::env::var(TIDB_BIN_ENV_KEY).expect("env TIDB_BIN is not set");
    let tidb_port_base = std::env::var(TIDB_PORT_ENV_KEY)
        .map(|s| s.parse().unwrap())
        .unwrap_or(TIDB_PORT_DEFAULT);
    let tidb_status_port_base = std::env::var(TIDB_STATUS_PORT_ENV_KEY)
        .map(|s| s.parse().unwrap())
        .unwrap_or(TIDB_STATUS_PORT_DEFAULT);

    let tiflash_bin = std::env::var(TIFLASH_BIN_ENV_KEY).expect("env TIFLASH_BIN is not set");

    let max_merge_region_size = REGION_SIZE.div(5);
    let max_merge_region_keys = max_merge_region_size.0 / 100; // Assume 100 bytes per key, about 2000 keys.
    let pd_scheduler_config = PdScheduleConfig {
        max_merge_region_size: max_merge_region_size.as_mb().max(1),
        max_merge_region_keys,
        split_merge_interval: SPLIT_MERGE_INTERVAL,
        ..Default::default()
    };

    let mut rng = rand::thread_rng();
    let pd_mode = if rng.gen_ratio(1, 10) {
        PdServerMode::Normal
    } else {
        PdServerMode::MicroServices {
            tso_count: PD_TSO_SVC_COUNT as u16,
        }
    };
    let tc = TidbCluster::new(
        pd_mode,
        PathBuf::from(pd_bin),
        pd_port_base,
        pd_scheduler_config,
        PathBuf::from(tidb_bin),
        tidb_port_base,
        tidb_status_port_base,
        PathBuf::from(tiflash_bin),
        INITIAL_KEYSPACE_COUNT as u16,
        security_config,
    );
    block_on(tc.start_pd(PD_COUNT as u16, PD_HEALTHY_TIMEOUT));
    tc
}

// TODO: merge to `prepare_cluster` in `test_all.rs`.
fn prepare_cluster(
    dfs_config: &DFSConfig,
    security_conf: &SecurityConfig,
    nodes_count: usize,
    initial_keyspace_count: usize,
    enable_inner_key_off: bool,
    use_remote_cop: bool,
    tc: Option<&TidbCluster>,
) -> ServerCluster {
    let mut rng = rand::thread_rng();
    let nodes = alloc_node_id_vec(nodes_count);
    let dfs_config = Arc::new(dfs_config.clone());
    let update_conf_fn = move |node_id: u16, conf: &mut TikvConfig| {
        conf.dfs = (*dfs_config).clone();
        conf.coprocessor.region_split_size = REGION_SIZE;
        conf.coprocessor.region_bucket_size = REGION_BUCKET_SIZE;
        conf.raft_store.peer_stale_state_check_interval = ReadableDuration::secs(1);
        conf.raft_store.abnormal_leader_missing_duration = ReadableDuration::secs(3);
        conf.raft_store.max_leader_missing_duration = ReadableDuration::secs(5);
        conf.raft_store.pd_heartbeat_tick_interval = ReadableDuration::secs(1);
        conf.rocksdb.writecf.block_size = ReadableSize::kb(4);
        conf.rocksdb.writecf.target_file_size_base = ReadableSize::kb(16);
        conf.rfengine.target_file_size = ReadableSize::mb(8);
        conf.rfengine.batch_compression_threshold =
            ReadableSize::kb(rand::thread_rng().gen_range(0..2));
        conf.rfengine.lightweight_backup = true;
        conf.rfengine.wal_chunk_target_file_size = ReadableSize::kb(512);
        conf.enable_inner_key_offset = enable_inner_key_off;
        conf.security = security_conf.clone();
        conf.kvengine.compaction_tombs_count = 100;
        conf.kvengine.max_del_range_delay = ReadableDuration(Duration::from_secs(3));
        conf.storage.flow_control.enable = true;

        if use_remote_cop {
            let cop_worker_url = tikv_worker_cop_url(node_id % TIKV_WORKERS_COUNT as u16);
            conf.kvengine.remote_worker_addr = cop_worker_url.clone();
            conf.kvengine.remote_coprocessor_addr = cop_worker_url;
            conf.kvengine.remote_coprocessor_min_blocks_size = 1024 * 1024;
        }
    };

    let pd_wrapper = match tc {
        Some(tc) => PdWrapper::new_real(tc.pd.endpoints(), security_conf),
        None => PdWrapper::new_test(1, security_conf, None),
    };
    let mut cluster = ServerCluster::new_opt(nodes, update_conf_fn, pd_wrapper);
    cluster.start_tikv_workers(TIKV_WORKERS_COUNT, TIKV_WORKERS_THREADS_COUNT, true);
    cluster.wait_region_replicated(&[], 3);

    let mut keyspaces: Vec<u32> = vec![];
    let mut keyspace_names: Vec<String> = vec![];
    match tc {
        Some(tc) => {
            let pd_control = tc.pd.get_pd_control();

            // Initial keyspaces have been created by `pre_alloc_keyspaces` of PD.
            // Note: keyspaces allocation starts from 1.
            for idx in 1..=initial_keyspace_count {
                let keyspace_name = TidbCluster::keyspace_name(idx as u16);
                let keyspace = block_on(pd_control.get_keyspace_by_name(&keyspace_name)).unwrap();
                keyspaces.push(keyspace.id);
                keyspace_names.push(keyspace_name);
            }
        }
        None => {
            // TODO
            unimplemented!();
        }
    }

    // TODO: create by API to be uniform with test PD.
    // TODO: enable encryption.
    cluster.keyspace_manager().create_keyspaces(
        &keyspaces,
        keyspace_names,
        DEFAULT_INNER_KEY_OFFSET,
        0,
        Some(&mut rng),
    );
    KEYSPACE_COUNTER.store(initial_keyspace_count, Ordering::Relaxed);

    // TODO: scatter regions.
    // TODO: restart cluster with inner key offset enabled

    cluster
}

async fn stop_schedulers(pd_ctl: Arc<pd_control::PdControl>) {
    // Pause all schedulers to make stats stable.
    pd_ctl
        .pause_or_resume_scheduler("all", Duration::MAX)
        .await
        .expect("pause schedulers failed");
    let all_schedulers = pd_ctl.list_schedulers(None).await.unwrap();
    // `list_schedulers(None)` does not return `paused_at` field. So invoke
    // `list_schedulers(Paused)` again.
    let paused_schedulers = pd_ctl
        .list_schedulers(Some(pd_control::SchedulerStatus::Paused))
        .await
        .unwrap();
    info!(
        "all schedulers: {:?}, paused schedulers: {:?}",
        all_schedulers, paused_schedulers
    );

    // Wait operators finished.
    let ok = try_wait_async(
        || {
            let pd_ctl = pd_ctl.clone();
            Box::pin(async move {
                let is_region_op = |op: &pd_control::Operator| -> bool {
                    for kind in [OpKind::OpMerge, OpKind::OpSplit] {
                        if op.kind_mask & kind as u32 != 0 {
                            return true;
                        }
                    }
                    false
                };
                let operators = pd_ctl
                    .get_operators()
                    .await
                    .unwrap()
                    .into_iter()
                    .filter(is_region_op)
                    .collect::<Vec<_>>();
                if operators.is_empty() {
                    return true;
                }

                for op in operators {
                    if let Err(err) = pd_ctl.cancel_operator_by_region(op.region_id).await {
                        // Would fail when the operator has finished.
                        warn!("cancel operator failed"; "op" => ?op, "err" => ?err);
                    } else {
                        info!("operator canceled"; "op" => ?op);
                    }
                }
                false
            })
        },
        10,
    )
    .await;
    if !ok {
        // It's OK to fail here. `verify_cluster` will retry.
        warn!(
            "wait operators finished timeout: {:?}",
            pd_ctl.get_operators().await.unwrap()
        );
    }
}

// TODO: merge to `verify_cluster` in `test_all.rs`.
async fn verify_cluster(cluster: &mut ServerCluster, tpc_switch_on: bool, jepsen_switch_on: bool) {
    // Check statistics.
    // Check after verify data, to ensure that PD heartbeat have updated region
    // stats.
    verify_cluster_stats(cluster, REGION_BUCKET_SIZE.0, Duration::from_secs(60));

    if tpc_switch_on {
        check_tpc();
    }
    if jepsen_switch_on {
        check_jepsen();
    }
}

async fn prepare_tpcc(
    tc: TidbCluster,
    keyspace_manager: KeyspaceManager,
    tpc_bin: String,
    keyspace_ids: Vec<u32>,
    use_txn_file: bool,
) {
    let mut handles = Vec::with_capacity(TPCC_WORKLOAD_CONCURRENCY);
    for tpc_idx in 0..TPCC_WORKLOAD_CONCURRENCY {
        let tc = tc.clone();
        let keyspace_manager = keyspace_manager.clone();
        let tpc_bin = PathBuf::from_str(&tpc_bin).unwrap();
        let keyspace_ids = keyspace_ids.to_vec();
        let task = async move {
            let db = db_name_by_tpc_idx(tpc_idx);

            for keyspace_id in keyspace_ids {
                let keyspace_name = keyspace_manager
                    .get_keyspace_meta(keyspace_id)
                    .unwrap()
                    .name();
                let tag = format!("tpcc-{}[{}]-{}", keyspace_id, keyspace_name, tpc_idx);
                let tidb_idx = TidbCluster::get_idx_by_keyspace_name(&keyspace_name);
                let params = tc.tidb.conn_params(tidb_idx);

                let mut tpc = Tpc::new(tag.clone(), tpc_bin.clone());
                tpc.tpcc()
                    .host(&params.host)
                    .port(params.port)
                    .user(&params.user)
                    .password(&params.password)
                    .db(&db)
                    .warehouses(TPCC_WAREHOUSES)
                    .max_procs(TPCC_MAX_PROCS);

                let conn_string = params.conn_string("test");
                let pool = sqlx::MySqlPool::connect(&conn_string).await.unwrap();

                info!("{} prepare_tpcc", tag; "use_txn_file" => use_txn_file);

                let mut sqls = vec![format!("CREATE DATABASE IF NOT EXISTS `{}`", db)];
                if use_txn_file {
                    sqls.push("SET GLOBAL tidb_txn_mode = 'optimistic'".to_string());
                    sqls.push("SET GLOBAL tidb_enable_txn_file = 'ON'".to_string());
                }
                for sql in sqls {
                    info!("{} executing sql", tag; "sql" => &sql);
                    sqlx::query(&sql).execute(&pool).await.unwrap();
                }

                tpc.prepare(TPCC_THREADS).await.unwrap();
                tpc.check().await.unwrap();

                // Use txn file during preparation only. As using optimistic transaction for
                // TPC-C will meet lots of write conflicts.
                if use_txn_file {
                    let sql = "SET GLOBAL tidb_txn_mode = 'pessimistic'";
                    info!("{} executing sql", tag; "sql" => &sql);
                    sqlx::query(sql).execute(&pool).await.unwrap();
                }
            }
        };
        handles.push(tokio::spawn(task));
    }

    for handle in handles {
        handle.await.unwrap();
    }
}

fn spawn_tpcc(
    tc: TidbCluster,
    keyspace_manager: KeyspaceManager,
    tpc_bin: &str,
    tpc_idx: usize,
    run_duration: Duration,
    timeout: Duration,
) -> tokio::task::JoinHandle<()> {
    let tpc_bin = PathBuf::from_str(tpc_bin).unwrap();
    tokio::spawn(async move {
        let db = db_name_by_tpc_idx(tpc_idx);

        let start_time = Instant::now();
        while start_time.saturating_elapsed() < timeout {
            let random_keyspace = || {
                let mut rng = rand::thread_rng();
                keyspace_manager.get_zipf_random_keyspace(&mut rng)
            };
            let keyspace_id = random_keyspace();
            let keyspace_name = keyspace_manager
                .get_keyspace_meta(keyspace_id)
                .unwrap()
                .name();
            {
                let lock = keyspace_manager.get_keyspace_lock(keyspace_id);
                let guard = lock.try_shared_lock();
                if guard.is_none() {
                    tokio::task::yield_now().await;
                    continue;
                }
                let _guard = guard.unwrap();

                let tag = format!("tpcc-{}[{}]-{}", keyspace_id, keyspace_name, tpc_idx);
                let tidb_idx = TidbCluster::get_idx_by_keyspace_name(&keyspace_name);
                let params = tc.tidb.conn_params(tidb_idx);

                let mut tpc = Tpc::new(tag, tpc_bin.clone());
                tpc.tpcc()
                    .host(&params.host)
                    .port(params.port)
                    .user(&params.user)
                    .password(&params.password)
                    .db(&db)
                    .warehouses(TPCC_WAREHOUSES)
                    .max_procs(TPCC_MAX_PROCS);

                let txns = tpc
                    .run(TPCC_THREADS, false, true, run_duration)
                    .await
                    .unwrap();
                TPCC_COUNTER.fetch_add(txns, Ordering::Relaxed);
                tpc.check().await.unwrap();
            }
        }
    })
}

fn db_name_by_tpc_idx(tpc_idx: usize) -> String {
    format!("tpcc_{tpc_idx}")
}

fn check_tpc_binary(tpc_bin: &str) {
    let mut cmd = std::process::Command::new(tpc_bin);
    cmd.arg("version");
    let output = cmd.output().unwrap();
    assert!(
        output.status.success(),
        "tpc binary check failed: {:?}",
        output
    );

    info!("tpc binary check passed"; "output" => ?output);
}

fn check_tpc() {
    let tpc_txns = TPCC_COUNTER.load(Ordering::Relaxed);
    assert!(
        tpc_txns >= 100,
        "TPC-C transactions are too few: {}",
        tpc_txns
    );
}

fn check_jepsen() {
    let jepsen_txns = JEPSEN_BANK_TXN_COUNTER.load(Ordering::Relaxed);
    let threshold = 100;
    assert!(
        jepsen_txns >= threshold,
        "Jepsen transactions are too few: {}, threshold: {}",
        jepsen_txns,
        threshold
    );
}

fn spawn_restart_tso_svc(
    tc: TidbCluster,
    restart_interval: Duration,
    timeout: Duration,
) -> tokio::task::JoinHandle<()> {
    let task = async move {
        let tso_svc_count = match tc.pd.mode() {
            PdServerMode::Normal => return,
            PdServerMode::MicroServices { tso_count } => *tso_count,
        };
        let start_time = Instant::now_coarse();
        while start_time.saturating_elapsed() < timeout {
            let random = || {
                let mut rng = rand::thread_rng();
                let tso_svc_idx = rng.gen_range(0..tso_svc_count);
                let interval_secs = restart_interval.as_secs();
                let stop_dur =
                    Duration::from_secs(rng.gen_range(interval_secs / 2..interval_secs * 3 / 2));
                // Give some time for TSO service to campaign about the leader before next loop.
                let loop_interval =
                    Duration::from_secs(rng.gen_range(interval_secs / 2..interval_secs));
                (tso_svc_idx, stop_dur, loop_interval)
            };
            let (tso_svc_idx, stop_dur, loop_interval) = random();

            tc.pd
                .restart_tso_svc(tso_svc_idx, stop_dur, PD_HEALTHY_TIMEOUT)
                .await;
            tokio::time::sleep(loop_interval).await;
        }
    };
    tokio::spawn(task)
}
