// Copyright 2023 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    ops::Div,
    path::PathBuf,
    sync::{atomic::Ordering, Arc},
    time::Duration,
};

use kvengine::{dfs::DFSConfig, table::sstable::BlockCacheType};
use pd_client::{
    pd_control,
    pd_control::{OpKind, PdControl, PdScheduleConfig},
};
use rand::prelude::*;
use security::SecurityConfig;
use sqlx::ConnectOptions;
use test_cloud_server::{
    oss::prepare_dfs, tidb::*, tikv_worker_cop_url, try_wait_async, ServerCluster,
    ServerClusterBuilder, TikvWorkerOptions,
};
use test_pd_client::PdWrapper;
use tikv::config::TikvConfig;
use tikv_util::{
    config::{ReadableDuration, ReadableSize},
    info,
    sys::SysQuota,
    time::Instant,
};
use tokio::runtime::Runtime;

use crate::{
    test_columnar::{prepare_columnar, run_columnar_workload},
    test_jepsen::*,
    test_tpc::*,
    test_txn_file::{TXN_CHUNK_MAX_SIZE, TXN_FILE_MIN_SIZE},
    test_unique::*,
    *,
};

pub(crate) const REGION_SIZE: ReadableSize = ReadableSize::mb(1);
// TiDB has records with 200kb+ size (see "mysql.stats_history"), so set bucket
// size to 256kb.
pub(crate) const REGION_BUCKET_SIZE: ReadableSize = ReadableSize::kb(256);
// Ref: https://docs.pingcap.com/tidb/stable/pd-configuration-file#split-merge-interval
const SPLIT_MERGE_INTERVAL: ReadableDuration = ReadableDuration::secs(10);

// Test on only one keyspace to simulate a heavy tenant. Scenes of multiple
// keyspaces are covered by test_random_all.
pub(crate) const INITIAL_KEYSPACE_COUNT: usize = 1;
pub(crate) const NODES_COUNT: usize = 4;
const TEST_DURATION: Duration = Duration::from_secs(120); // Test for longer as TiDB bootstrap may cost 30s+.

pub(crate) const TIKV_WORKERS_COUNT: usize = 2;

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
pub(crate) const TIDB_HEALTHY_TIMEOUT: Duration = Duration::from_secs(600); // TODO: improve the efficiency of TiDB start up.
pub(crate) const TIDB_LOG_LEVEL: &str = "info";
pub(crate) const TIDB_GC_INTERVAL: &str = "60s";
pub(crate) const TIDB_GC_LIFETIME: &str = "90s";

pub(crate) const TIKV_SERVER_BIN_ENV_KEY: &str = "TIKV_SERVER_BIN";
pub(crate) const TIKV_WORKER_BIN_ENV_KEY: &str = "TIKV_WORKER_BIN";

pub(crate) const TIFLASH_SWITCH_ENV_KEY: &str = "USE_TIFLASH";
pub(crate) const TIFLASH_BIN_ENV_KEY: &str = "TIFLASH_BIN";
pub(crate) const TIFLASH_SERVER_COUNT: usize = 1;
pub(crate) const TIFLASH_HEALTHY_TIMEOUT: Duration = Duration::from_secs(120);

pub(crate) const TPC_WORKLOAD_SWITCH_ENV_KEY: &str = "TPC_WORKLOAD";
pub(crate) const TPC_BIN_ENV_KEY: &str = "TPC_BIN";
pub(crate) const TPCC_RUN_DURATION: Duration = Duration::from_secs(10); // Duration of each TPCC run.
pub(crate) const TPCC_WORKLOAD_CONCURRENCY: usize = 1;

pub(crate) const JEPSEN_WORKLOAD_SWITCH_ENV_KEY: &str = "JEPSEN_WORKLOAD";
pub(crate) const JEPSEN_WORKLOAD_USE_TXN_FILE_ENV_KEY: &str = "JEPSEN_TXN_FILE";
pub(crate) const JEPSEN_WORKLOAD_KEYSPACE: u32 = 1; // Keyspace starts from 1.

pub(crate) const UNIQUE_WORKLOAD_SWITCH_ENV_KEY: &str = "UNIQUE_WORKLOAD";
pub(crate) const UNIQUE_WORKLOAD_KEYSPACE: u32 = 1; // Keyspace starts from 1.

pub(crate) const COLUMNAR_WORKLOAD_SWITCH_ENV_KEY: &str = "COLUMNAR_WORKLOAD";
pub(crate) const COLUMNAR_WORKLOAD_KEYSPACE: u32 = 1;

pub(crate) const VERIFY_HEALTHY_TIMEOUT: Duration = Duration::from_secs(120);

pub(crate) const ENABLE_GLOBAL_TXN_FILE_RATIO: f64 = 0.8; // 80% chance enable txn file globally.
pub(crate) const ENABLE_GLOBAL_TXN_FILE_ENV_KEY: &str = "GLOBAL_TXN_FILE";

pub(crate) const USE_REMOTE_COP_ENV_KEY: &str = "USE_REMOTE_COP";
pub(crate) const REMOTE_COP_MIN_BLOCK_SIZE_OPTIONS: [usize; 3] =
    [64 * 1024, 512 * 1024, 1024 * 1024];
pub(crate) const COP_BLOCK_CACHE_SIZE: ReadableSize = ReadableSize::mb(16); // Small size to make eviction more frequent.

pub(crate) const RESTART_TSO_SVC_ENV_KEY: &str = "RESTART_TSO_SVC";

pub(crate) const MEMORY_CAPACITY_RATIO: f64 = 0.8; // Reserve 20% memory for PD, TiDB, and TiFlash.

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

    let switches = Switches::from_env();
    info!("switches: {:?}", switches);

    // Prepare.
    let (_temp_dir, _oss, dfs_config) = prepare_dfs("oss_");
    let security_conf = new_security_config();
    let tc = prepare_tidb_cluster(&security_conf);
    let mut cluster = prepare_cluster(
        &dfs_config,
        &security_conf,
        NODES_COUNT,
        INITIAL_KEYSPACE_COUNT,
        &switches,
        &tc,
    );
    let pd_client = cluster.get_pd_client_ext();
    let pd_ctl = Arc::new(cluster.get_pd_control().unwrap());
    let keyspace_manager = cluster.keyspace_manager().clone();

    let tikv_worker_addr = cluster.tikv_worker_endpoints().pop().unwrap();
    start_components(&tc, tikv_worker_addr, &switches, &dfs_config, &runtime);
    prepare_workloads(&tc, &keyspace_manager, &switches, &runtime);
    let running = Running::new_start();
    let async_handles =
        start_workloads(&tc, &keyspace_manager, &switches, &runtime, running.clone());

    // Main loop.
    let start_time = Instant::now();
    while start_time.saturating_elapsed() < TEST_DURATION {
        // Restart nodes.
        random_node_restart(&mut cluster, |_, _| {});
    }

    // Finish.
    runtime.block_on(async {
        info!("test finished, stopping all workers");
        running.stop();
        for handle in async_handles {
            handle.await.unwrap();
        }

        // Make stats stable
        info!("stop TiDB and schedulers");
        check_and_stop_components(&tc).await;
        stop_schedulers(pd_ctl).await;

        info!("verify cluster");
        verify_cluster(&mut cluster, &switches).await;
    });

    // Stop cluster.
    info!("stopping cluster");
    cluster.stop();

    let region_number = pd_client.get_regions_number();
    tc.pd.stop_all();

    // Statistics.
    let stats = WorkloadStats::collect();

    info!("TEST SUCCEED: region_number {}, {:?}", region_number, stats);
}

pub(crate) fn prepare_tidb_cluster(security_config: &SecurityConfig) -> TidbCluster {
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

fn prepare_cluster(
    dfs_config: &DFSConfig,
    security_conf: &SecurityConfig,
    nodes_count: usize,
    initial_keyspace_count: usize,
    switches: &Switches,
    tc: &TidbCluster,
) -> ServerCluster {
    let mut rng = rand::thread_rng();
    let nodes = alloc_node_id_vec(nodes_count);
    let tikv_worker_nodes = alloc_node_id_vec(TIKV_WORKERS_COUNT);
    let update_conf_fn =
        generate_update_conf_fn(dfs_config, security_conf, &tikv_worker_nodes, switches);
    let pd = PdWrapper::new_real(tc.pd.endpoints(), security_conf, PD_CLIENT_UPDATE_INTERVAL);
    let mut cluster = ServerClusterBuilder::new(nodes, update_conf_fn)
        .pd(pd)
        .memory_capacity_ratio(MEMORY_CAPACITY_RATIO)
        .build();
    cluster.start_tikv_workers(
        tikv_worker_nodes,
        TikvWorkerOptions {
            cop_block_cache_size: COP_BLOCK_CACHE_SIZE,
            cop_block_cache_type: switches.block_cache_type,
            ..Default::default()
        },
    );
    if switches.columnar_switch_on {
        cluster.start_schema_manager(alloc_node_id());
    }
    cluster.wait_region_replicated(&[], 3);

    let pd_control = tc.pd.get_pd_control();
    // Initial keyspaces have been created by `pre_alloc_keyspaces` of PD.
    let (keyspace_ids, keyspace_names) =
        get_pre_alloc_keyspaces(initial_keyspace_count, &pd_control);

    // TODO: create by API to be uniform with test PD.
    // TODO: enable encryption.
    cluster.keyspace_manager().create_keyspaces(
        &keyspace_ids,
        keyspace_names,
        &CreateKeyspaceOptions {
            enable_inner_key_off: switches.enable_inner_key_off,
            ..Default::default()
        },
        Some(&mut rng),
    );
    KEYSPACE_COUNTER.store(initial_keyspace_count, Ordering::Relaxed);

    // TODO: scatter regions.
    // TODO: restart cluster with inner key offset enabled

    cluster
}

pub(crate) fn generate_update_conf_fn<'a>(
    dfs_config: &'a DFSConfig,
    security_conf: &'a SecurityConfig,
    tikv_worker_nodes: &'a [u16],
    switches: &'a Switches,
) -> impl Fn(u16, &mut TikvConfig) + 'a {
    let cpu_cores = SysQuota::cpu_cores_quota() as usize;

    move |_node_id: u16, conf: &mut TikvConfig| {
        let mut rng = thread_rng();
        conf.dfs = dfs_config.clone();
        conf.enable_inner_key_offset = switches.enable_inner_key_off;
        conf.security = security_conf.clone();

        conf.coprocessor.region_split_size = REGION_SIZE;
        conf.coprocessor.region_bucket_size = REGION_BUCKET_SIZE;

        conf.raft_store.peer_stale_state_check_interval = ReadableDuration::secs(5);
        conf.raft_store.abnormal_leader_missing_duration = ReadableDuration::secs(15);
        conf.raft_store.max_leader_missing_duration = ReadableDuration::secs(25);
        conf.raft_store.split_region_check_tick_interval = ReadableDuration::millis(500);
        conf.raft_store.raft_log_gc_tick_interval = ReadableDuration::millis(500);
        conf.raft_store.pd_heartbeat_tick_interval = ReadableDuration::secs(5);
        conf.raft_store.pd_store_heartbeat_tick_interval = ReadableDuration::millis(500);

        conf.rocksdb.writecf.block_size = ReadableSize::kb(4);
        conf.rocksdb.writecf.target_file_size_base = ReadableSize::kb(16);

        conf.rfengine.target_file_size = ReadableSize::mb(8);
        conf.rfengine.batch_compression_threshold = ReadableSize::kb(rng.gen_range(0..2));
        conf.rfengine.lightweight_backup = true;
        conf.rfengine.wal_chunk_target_file_size = ReadableSize::kb(512);
        conf.rfengine.dfs_worker_memory_limit = (conf.rfengine.target_file_size * 8).into();

        conf.kvengine.compaction_tombs_count = 100;
        conf.kvengine.max_del_range_delay = ReadableDuration(Duration::from_secs(3));
        conf.kvengine.block_cache_type = switches.block_cache_type;
        conf.kvengine
            .columnar_table_build_options
            .pack_max_row_count = 32;
        conf.kvengine.columnar_table_build_options.pack_max_size = 32 * 128;
        conf.kvengine.vector_index_build_options.delta_size = 128;
        conf.kvengine.vector_index_build_options.rebuild_file_count = 2;

        conf.storage.flow_control.enable = true;
        conf.storage.scheduler_worker_pool_size = cpu_cores;
        conf.gc.enable_safe_point_v2 = true;

        if switches.remote_cop_min_block_size > 0 {
            let tikv_worker_idx = *tikv_worker_nodes.choose(&mut rng).unwrap();
            let cop_worker_url = tikv_worker_cop_url(tikv_worker_idx);
            conf.kvengine.remote_worker_addr = cop_worker_url.clone();
            conf.kvengine.remote_coprocessor_addr = cop_worker_url;
            conf.kvengine.remote_coprocessor_min_blocks_size = switches.remote_cop_min_block_size;
        }
    }
}

pub(crate) fn get_pre_alloc_keyspaces(
    initial_keyspace_count: usize,
    pd_control: &PdControl,
) -> (
    Vec<u32>,    // keyspace_ids
    Vec<String>, // keyspace_names
) {
    let mut ids = vec![];
    let mut names = vec![];
    // Note: keyspaces allocation starts from 1.
    for idx in 1..=initial_keyspace_count {
        let keyspace_name = TidbCluster::keyspace_name(idx as u16);
        let keyspace = block_on(pd_control.get_keyspace_by_name(&keyspace_name)).unwrap();
        ids.push(keyspace.id);
        names.push(keyspace_name);
    }
    (ids, names)
}

pub(crate) fn start_components(
    tc: &TidbCluster,
    tikv_worker_addr: String,
    switches: &Switches,
    dfs_config: &DFSConfig,
    runtime: &Runtime,
) {
    let start_tidb = {
        let tc = tc.clone();
        let columnar_switch_on = switches.columnar_switch_on;
        runtime.spawn(async move {
            tc.start_tidb(
                INITIAL_KEYSPACE_COUNT as u16,
                TIDB_HEALTHY_TIMEOUT,
                TIDB_LOG_LEVEL,
                StartTidbOptions {
                    tikv_worker_addr,
                    txn_chunk_max_size: TXN_CHUNK_MAX_SIZE as u64,
                    txn_file_min_mutation_size: Some(TXN_FILE_MIN_SIZE as u64),
                    gc_interval: TIDB_GC_INTERVAL.to_owned(),
                    gc_lifetime: TIDB_GC_LIFETIME.to_owned(),
                    tiflash_compute_mode: columnar_switch_on,
                },
            )
            .await
        })
    };
    let start_tiflash = {
        let tc = tc.clone();
        let columnar_switch_on = switches.columnar_switch_on;
        let dfs_config = dfs_config.clone();
        runtime.spawn_blocking(move || {
            tc.start_tiflash(
                TIFLASH_SERVER_COUNT as u16,
                &dfs_config,
                TIFLASH_HEALTHY_TIMEOUT,
                columnar_switch_on,
            );
        })
    };
    let (start_tidb, start_tiflash) =
        runtime.block_on(async move { futures::join!(start_tidb, start_tiflash) });
    start_tidb.unwrap();
    start_tiflash.unwrap();
}

pub(crate) fn prepare_workloads(
    tc: &TidbCluster,
    keyspace_manager: &KeyspaceManager,
    switches: &Switches,
    runtime: &Runtime,
) {
    if !switches.global_use_txn_file {
        runtime.block_on(async {
            for keyspace_id in keyspace_manager.get_all_keyspaces() {
                let pool = connect_tidb(tc, keyspace_manager, keyspace_id).await;
                sqlx::query("SET GLOBAL tidb_disable_txn_file = 'ON'")
                    .execute(&pool)
                    .await
                    .unwrap();
            }
        });
    }

    let mut prepare_tasks = vec![];
    if switches.tpc_switch_on {
        let tpc_bin = std::env::var(TPC_BIN_ENV_KEY).expect("env TPC_BIN is not set");
        check_tpc_binary(&tpc_bin);

        let all_keyspaces = keyspace_manager.get_all_keyspaces();
        prepare_tasks.push(runtime.spawn(prepare_tpcc(
            tc.clone(),
            keyspace_manager.clone(),
            tpc_bin.clone(),
            all_keyspaces,
            switches.global_use_txn_file,
            TPCC_WORKLOAD_CONCURRENCY,
        )));
    }
    if switches.jepsen_switch_on {
        prepare_tasks.push(runtime.spawn(prepare_jepsen_bank(
            tc.clone(),
            keyspace_manager.clone(),
            JEPSEN_WORKLOAD_KEYSPACE,
            switches.tiflash_switch_on.then_some(TIFLASH_SERVER_COUNT),
        )));
    }
    if switches.unique_workload_switch_on {
        prepare_tasks.push(runtime.spawn(prepare_unique_workload(
            tc.clone(),
            keyspace_manager.clone(),
            UNIQUE_WORKLOAD_KEYSPACE,
        )));
    }
    if switches.columnar_switch_on {
        info!("prepare_columnar");
        prepare_tasks.push(runtime.spawn(prepare_columnar(
            tc.clone(),
            keyspace_manager.clone(),
            COLUMNAR_WORKLOAD_KEYSPACE,
        )));
    }
    runtime
        .block_on(futures::future::try_join_all(prepare_tasks))
        .unwrap();
}

pub(crate) fn start_workloads(
    tc: &TidbCluster,
    keyspace_manager: &KeyspaceManager,
    switches: &Switches,
    runtime: &Runtime,
    running: Running,
) -> Vec<tokio::task::JoinHandle<()>> {
    let mut async_handles = vec![];
    if switches.tpc_switch_on {
        let tpc_bin = std::env::var(TPC_BIN_ENV_KEY).unwrap();
        for tpc_idx in 0..TPCC_WORKLOAD_CONCURRENCY {
            async_handles.push(spawn_tpcc(
                tc.clone(),
                keyspace_manager.clone(),
                &tpc_bin,
                tpc_idx,
                TPCC_RUN_DURATION,
                running.clone(),
            ));
        }
    }
    if switches.jepsen_switch_on {
        async_handles.push(runtime.spawn(run_jepsen_bank(
            tc.clone(),
            keyspace_manager.clone(),
            JEPSEN_WORKLOAD_KEYSPACE,
            switches.global_use_txn_file && switches.jepsen_use_txn_file,
            switches.tiflash_switch_on,
            running.clone(),
        )));
    }
    if switches.unique_workload_switch_on {
        async_handles.push(runtime.spawn(run_unique_workload(
            tc.clone(),
            keyspace_manager.clone(),
            UNIQUE_WORKLOAD_KEYSPACE,
            switches.global_use_txn_file,
            running.clone(),
        )));
    }
    if switches.columnar_switch_on {
        async_handles.push(runtime.spawn(run_columnar_workload(
            tc.clone(),
            keyspace_manager.clone(),
            COLUMNAR_WORKLOAD_KEYSPACE,
            running.clone(),
        )));
    }

    assert!(!async_handles.is_empty(), "no workload to run");
    if switches.restart_tso_svc {
        async_handles.push(spawn_restart_tso_svc(
            tc.clone(),
            Duration::from_secs(10),
            running,
        ));
    }

    async_handles
}

pub(crate) async fn stop_schedulers(pd_ctl: Arc<pd_control::PdControl>) {
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

pub(crate) async fn check_and_stop_components(tc: &TidbCluster) {
    tc.pd.must_healthy(VERIFY_HEALTHY_TIMEOUT).await;
    tc.tidb.must_all_healthy(VERIFY_HEALTHY_TIMEOUT).await;
    tc.tiflash.must_all_healthy(VERIFY_HEALTHY_TIMEOUT).await;
    tc.tidb.stop_all(); // To stop background tasks.
    tc.tiflash.stop_all();
}

// TODO: merge to `verify_cluster` in `test_all.rs`.
pub(crate) async fn verify_cluster(cluster: &mut ServerCluster, switches: &Switches) {
    // Check statistics.
    // Check after verify data, to ensure that PD heartbeat have updated region
    // stats.
    verify_cluster_stats(cluster, REGION_BUCKET_SIZE.0, Duration::from_secs(60));

    if switches.tpc_switch_on {
        check_tpc();
    }
    if switches.jepsen_switch_on {
        check_jepsen();
    }
}

pub(crate) fn spawn_restart_tso_svc(
    tc: TidbCluster,
    restart_interval: Duration,
    running: Running,
) -> tokio::task::JoinHandle<()> {
    let task = async move {
        let tso_svc_count = match tc.pd.mode() {
            PdServerMode::Normal => return,
            PdServerMode::MicroServices { tso_count } => *tso_count,
        };
        let start_time = Instant::now_coarse();
        while running.get() {
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
        info!("restart tso workload exit"; "dur" => ?start_time.saturating_elapsed());
    };
    tokio::spawn(task)
}

pub(crate) async fn connect_tidb(
    tc: &TidbCluster,
    keyspace_manager: &KeyspaceManager,
    keyspace_id: u32,
) -> sqlx::MySqlPool {
    let keyspace_name = keyspace_manager
        .get_keyspace_meta(keyspace_id)
        .unwrap()
        .name();
    let tidb_idx = TidbCluster::get_idx_by_keyspace_name(&keyspace_name);
    let params = tc.tidb.conn_params(tidb_idx);
    let mut opts = sqlx::mysql::MySqlConnectOptions::new()
        .host(&params.host)
        .port(params.port)
        .username(&params.user)
        .database("test");
    opts.log_statements(log::LevelFilter::Debug)
        .log_slow_statements(log::LevelFilter::Warn, Duration::from_secs(30));
    sqlx::mysql::MySqlPoolOptions::new()
        .connect_with(opts)
        .await
        .unwrap()
}

#[derive(Debug)]
pub(crate) struct Switches {
    pub enable_inner_key_off: bool,
    pub remote_cop_min_block_size: usize,
    pub block_cache_type: BlockCacheType,
    pub columnar_switch_on: bool,
    pub tiflash_switch_on: bool,
    pub tpc_switch_on: bool,
    pub jepsen_switch_on: bool,
    pub jepsen_use_txn_file: bool,
    pub unique_workload_switch_on: bool,
    pub global_use_txn_file: bool,
    pub restart_tso_svc: bool,
}

impl Switches {
    pub fn from_env() -> Self {
        let mut rng = thread_rng();

        let enable_inner_key_off: bool = rng.gen_bool(env_param("ENABLE_INNER_KEY_OFF_RATIO", 0.5));
        // Random min block size to generate more or less workloads for cop workers.
        let remote_cop_min_block_size = env_switch(USE_REMOTE_COP_ENV_KEY) as usize
            * (*REMOTE_COP_MIN_BLOCK_SIZE_OPTIONS.choose(&mut rng).unwrap());
        let block_cache_type = if rng.gen_ratio(1, 5) {
            BlockCacheType::Moka
        } else {
            BlockCacheType::Quick
        };
        let columnar_switch_on = env_switch_opt(COLUMNAR_WORKLOAD_SWITCH_ENV_KEY, 0);
        let tiflash_switch_on = env_switch(TIFLASH_SWITCH_ENV_KEY);
        let tpc_switch_on = env_switch(TPC_WORKLOAD_SWITCH_ENV_KEY);
        let jepsen_switch_on = env_switch(JEPSEN_WORKLOAD_SWITCH_ENV_KEY);
        let jepsen_use_txn_file = env_switch(JEPSEN_WORKLOAD_USE_TXN_FILE_ENV_KEY);
        let unique_workload_switch_on = env_switch_opt(UNIQUE_WORKLOAD_SWITCH_ENV_KEY, 0);

        let global_use_txn_file = env_switch(ENABLE_GLOBAL_TXN_FILE_ENV_KEY);
        let global_use_txn_file = global_use_txn_file && rng.gen_bool(ENABLE_GLOBAL_TXN_FILE_RATIO);

        let restart_tso_svc = env_switch(RESTART_TSO_SVC_ENV_KEY);

        Self {
            enable_inner_key_off,
            remote_cop_min_block_size,
            block_cache_type,
            columnar_switch_on,
            tiflash_switch_on,
            tpc_switch_on,
            jepsen_switch_on,
            jepsen_use_txn_file,
            unique_workload_switch_on,
            global_use_txn_file,
            restart_tso_svc,
        }
    }
}

#[allow(dead_code)]
#[derive(Debug)]
pub(crate) struct WorkloadStats {
    pub total_keyspace_count: usize,
    pub total_node_restart: usize,
    pub total_tpcc_txns: usize,
    pub total_jepsen_bank: usize,
    pub total_jepsen_bank_retry: usize,
    pub total_unique_workload: usize,
    pub total_unique_conflict: usize,
}

impl WorkloadStats {
    pub fn collect() -> Self {
        let total_keyspace_count = KEYSPACE_COUNTER.load(Ordering::SeqCst);
        let total_node_restart = NODE_RESTART_COUNTER.load(Ordering::SeqCst);
        let total_tpcc_txns = TPCC_COUNTER.load(Ordering::SeqCst);
        let total_jepsen_bank = JEPSEN_BANK_TXN_COUNTER.load(Ordering::SeqCst);
        let total_jepsen_bank_retry = JEPSEN_BANK_TXN_RETRY_COUNTER.load(Ordering::SeqCst);
        let total_unique_workload = UNIQUE_WORKLOAD_TXN_COUNTER.load(Ordering::SeqCst);
        let total_unique_conflict = UNIQUE_WORKLOAD_CONFLICT_COUNTER.load(Ordering::SeqCst);
        Self {
            total_keyspace_count,
            total_node_restart,
            total_tpcc_txns,
            total_jepsen_bank,
            total_jepsen_bank_retry,
            total_unique_workload,
            total_unique_conflict,
        }
    }
}
