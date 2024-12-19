// Copyright 2024 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    path::PathBuf,
    sync::{atomic::Ordering, Arc},
    time::Duration,
};

use kvengine::{dfs::DFSConfig, table::sstable::BlockCacheType};
use pd_client::pd_control::PdControl;
use rand::prelude::*;
use security::SecurityConfig;
use test_cloud_server::{
    must_wait,
    oss::prepare_dfs,
    tidb::*,
    tikv_bin::{TikvServers, TikvWorkers},
    tikv_worker_cop_url, ServerCluster, TikvWorkerOptions,
};
use test_pd_client::PdWrapper;
use tikv::config::TikvConfig;
use tikv_util::{
    config::{ReadableDuration, ReadableSize},
    info,
    sys::SysQuota,
    time::Instant,
};

use crate::{
    test_columnar::{prepare_columnar, run_columnar_workload},
    test_jepsen::*,
    test_tidb::*,
    test_txn_file::{TXN_CHUNK_MAX_SIZE, TXN_FILE_MIN_SIZE},
    test_unique::*,
    *,
};

const TIKV_STORE_UPGRADE_DOWNGRADE_INTERVAL: Duration = Duration::from_secs(10); // Interval between upgrade/downgrade TiKV stores.
const TEST_DURATION_BEFORE_UPGRADE: Duration = Duration::from_secs(60);
const TEST_DURATION_AFTER_UPGRADE: Duration = Duration::from_secs(60);
const TEST_DURATION_AFTER_DOWNGRADE: Duration = Duration::from_secs(60);

const EVICT_LEADERS_TIMEOUT: Duration = Duration::from_secs(30);

const WAIT_STORE_STATE_TIMEOUT: Duration = Duration::from_secs(60);
const WAIT_TIKV_SERVER_HEALTHY_TIMEOUT: Duration = Duration::from_secs(30);
const WAIT_TIKV_WORKER_HEALTHY_TIMEOUT: Duration = Duration::from_secs(15);

#[test]
fn test_random_upgrade() {
    let _logger_guard = test_util::init_log_for_test_async();
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .worker_threads(4)
        .thread_name("random-workload")
        .build()
        .unwrap();
    let _guard = runtime.enter();
    let mut rng = thread_rng();

    // TODO: eliminate duplicated codes with `test_random_with_tidb`.
    let enable_inner_key_off: bool = rng.gen_bool(ENABLE_INNER_KEY_OFF_RATIO);
    let use_remote_cop = env_switch(USE_REMOTE_COP_ENV_KEY);
    let block_cache_type = if rng.gen_ratio(1, 5) {
        BlockCacheType::Moka
    } else {
        BlockCacheType::Quick
    };
    let columnar_switch_on = env_switch_opt(COLUMNAR_WORKLOAD_SWITCH_ENV_KEY, 0);

    let test_dur_before_upgrade = env_param(
        "TEST_DUR_BEFORE_UPGRADE",
        ReadableDuration(TEST_DURATION_BEFORE_UPGRADE),
    );
    let test_dur_after_upgrade = env_param(
        "TEST_DUR_AFTER_UPGRADE",
        ReadableDuration(TEST_DURATION_AFTER_UPGRADE),
    );
    let test_dur_after_downgrade = env_param(
        "TEST_DUR_AFTER_DOWNGRADE",
        ReadableDuration(TEST_DURATION_AFTER_DOWNGRADE),
    );
    // +2 for estimated extra time for upgrade and downgrade.
    let dur_upgrade_downgrade =
        TIKV_STORE_UPGRADE_DOWNGRADE_INTERVAL * (NODES_COUNT as u32 + 2) * 2;
    let test_dur = test_dur_before_upgrade.0
        + test_dur_after_upgrade.0
        + test_dur_after_downgrade.0
        + dur_upgrade_downgrade;

    let evict_leader_switch = rng.gen_ratio(1, 5);

    info!("switches";
        "enable_inner_key_off" => enable_inner_key_off,
        "use_remote_cop" => use_remote_cop,
        "columnar_switch" => columnar_switch_on,
        "block_cache_type" => ?block_cache_type,
        "test_dur_before_upgrade" => ?test_dur_before_upgrade,
        "test_dur_after_upgrade" => ?test_dur_after_upgrade,
        "evict_leader" => evict_leader_switch,
    );

    // Prepare.
    let (_temp_dir, _oss, dfs_config) = prepare_dfs("oss_");
    let security_conf = new_security_config();
    let tc = prepare_tidb_cluster(&security_conf);
    let (mut cluster, mut tikv_servers, mut tikv_workers) = prepare_cluster(
        &dfs_config,
        &security_conf,
        NODES_COUNT,
        INITIAL_KEYSPACE_COUNT,
        enable_inner_key_off,
        use_remote_cop,
        columnar_switch_on,
        block_cache_type,
        &tc,
    );
    let pd_client = cluster.get_pd_client_ext();
    let pd_ctl = Arc::new(cluster.get_pd_control().unwrap());
    let keyspace_manager = cluster.keyspace_manager().clone();

    let start_tidb = {
        let tc = tc.clone();
        let tikv_worker_addr = tikv_workers.endpoints().pop().unwrap();
        runtime.spawn(async move {
            tc.start_tidb(
                INITIAL_KEYSPACE_COUNT as u16,
                TIDB_HEALTHY_TIMEOUT,
                TIDB_LOG_LEVEL,
                StartTidbOptions {
                    tikv_worker_addr,
                    txn_chunk_max_size: TXN_CHUNK_MAX_SIZE as u64,
                    txn_file_min_mutation_size: Some(TXN_FILE_MIN_SIZE as u64),
                    tiflash_compute_mode: columnar_switch_on,
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
                columnar_switch_on,
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
    let unique_workload_switch_on = env_switch_opt(UNIQUE_WORKLOAD_SWITCH_ENV_KEY, 0);
    let global_use_txn_file = env_switch(ENABLE_GLOBAL_TXN_FILE_ENV_KEY);

    let mut rng = thread_rng();
    let global_use_txn_file = global_use_txn_file && rng.gen_bool(ENABLE_GLOBAL_TXN_FILE_RATIO);

    info!("global_use_txn_file: {}", global_use_txn_file);
    if !global_use_txn_file {
        runtime.block_on(async {
            for keyspace_id in keyspace_manager.get_all_keyspaces() {
                let pool = connect_tidb(&tc, &keyspace_manager, keyspace_id).await;
                sqlx::query("SET GLOBAL tidb_disable_txn_file = 'ON'")
                    .execute(&pool)
                    .await
                    .unwrap();
            }
        });
    }

    let mut prepare_tasks = vec![];
    if tpc_switch_on {
        let tpc_bin = std::env::var(TPC_BIN_ENV_KEY).expect("env TPC_BIN is not set");
        check_tpc_binary(&tpc_bin);

        let all_keyspaces = keyspace_manager.get_all_keyspaces();
        prepare_tasks.push(runtime.spawn(prepare_tpcc(
            tc.clone(),
            keyspace_manager.clone(),
            tpc_bin.clone(),
            all_keyspaces,
            global_use_txn_file,
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
    if unique_workload_switch_on {
        prepare_tasks.push(runtime.spawn(prepare_unique_workload(
            tc.clone(),
            keyspace_manager.clone(),
            UNIQUE_WORKLOAD_KEYSPACE,
        )));
    }
    if columnar_switch_on {
        info!("prepare_columnar");
        prepare_tasks.push(runtime.spawn(prepare_columnar(
            tc.clone(),
            keyspace_manager.clone(),
            COLUMNAR_WORKLOAD_KEYSPACE,
        )));
    }
    runtime.block_on(futures::future::join_all(prepare_tasks));

    let mut async_handles = vec![];
    if tpc_switch_on {
        let tpc_bin = std::env::var(TPC_BIN_ENV_KEY).unwrap();
        for tpc_idx in 0..TPCC_WORKLOAD_CONCURRENCY {
            async_handles.push(spawn_tpcc(
                tc.clone(),
                keyspace_manager.clone(),
                &tpc_bin,
                tpc_idx,
                TPCC_RUN_DURATION,
                test_dur,
            ));
        }
    }
    if jepsen_switch_on {
        async_handles.push(runtime.spawn(run_jepsen_bank(
            tc.clone(),
            keyspace_manager.clone(),
            JEPSEN_WORKLOAD_KEYSPACE,
            global_use_txn_file && jepsen_use_txn_file,
            tiflash_switch_on,
            test_dur,
        )));
    }
    if unique_workload_switch_on {
        async_handles.push(runtime.spawn(run_unique_workload(
            tc.clone(),
            keyspace_manager.clone(),
            UNIQUE_WORKLOAD_KEYSPACE,
            global_use_txn_file,
            test_dur,
        )));
    }
    if columnar_switch_on {
        async_handles.push(runtime.spawn(run_columnar_workload(
            tc.clone(),
            keyspace_manager,
            COLUMNAR_WORKLOAD_KEYSPACE,
            test_dur,
        )));
    }

    assert!(!async_handles.is_empty(), "no workload to run");
    async_handles.push(spawn_restart_tso_svc(
        tc.clone(),
        Duration::from_secs(10),
        test_dur,
    ));

    let must_wait_store_down = |ctx: &str, store_id: u64| {
        must_wait_store_state(
            ctx,
            &pd_ctl,
            store_id,
            |store| store.store.state_name != "Up",
            WAIT_STORE_STATE_TIMEOUT,
        );
    };
    let must_wait_store_up = |ctx: &str, store_id: u64| {
        must_wait_store_state(
            ctx,
            &pd_ctl,
            store_id,
            |store| store.store.state_name == "Up",
            WAIT_STORE_STATE_TIMEOUT,
        );
    };

    // Before upgrade.
    let start_time = Instant::now();
    while start_time.saturating_elapsed() < test_dur_before_upgrade.0 {
        sleep(Duration::from_secs(1));
    }

    // Upgrade tikv-workers.
    // TODO: rolling upgrade.
    {
        block_on(tikv_workers.must_all_healthy(Duration::from_secs(1)));
        tikv_workers.stop_all();
        cluster.start_tikv_workers_on_existed_configs(2);
        block_on(cluster.tikv_workers_must_healthy(WAIT_TIKV_WORKER_HEALTHY_TIMEOUT));
        info!("tikv-worker: upgrade finished");
    }

    // Rolling upgrade tikv-servers.
    {
        block_on(tikv_servers.must_all_healthy(Duration::from_secs(1)));
        let nodes = tikv_servers.get_all_nodes();
        for node_id in nodes {
            // Sleep first to have longer test duration for different version between
            // tikv-server & tikv-worker.
            std::thread::sleep(TIKV_STORE_UPGRADE_DOWNGRADE_INTERVAL);

            let conf = cluster.get_node_config(node_id);
            let store = block_on(pd_ctl.find_store_by_status_address(&conf.server.status_addr))
                .unwrap()
                .unwrap();
            let store_id = store.store.id;

            if evict_leader_switch {
                info!("upgrade: evict leader"; "node_id" => node_id, "store_id" => store_id, "store" => ?store);
                let (store, scheduler_name) =
                    block_on(pd_ctl.evict_store_leaders(store_id, EVICT_LEADERS_TIMEOUT)).unwrap();
                if store.status.leader_count > 0 {
                    warn!("upgrade: store still has leaders after evicting"; "node_id" => node_id, "store" => ?store);
                }
                block_on(pd_ctl.remove_scheduler(&scheduler_name)).unwrap();
            }

            info!("upgrade: stop old version"; "node_id" => node_id);
            let exit_status = tikv_servers.stop_node(node_id);
            if !exit_status.success() {
                panic!(
                    "tikv-server exit with error, node_id {}, exit_status {:?}",
                    node_id, exit_status
                );
            }
            must_wait_store_down("upgrade: wait for store is down", store_id);

            info!("upgrade: start new version"; "node_id" => ?node_id);
            cluster.start_node(node_id, |_, _| {});
            let new_store_id = cluster.get_store_id(node_id);
            assert_eq!(new_store_id, store_id);
            must_wait_store_up("upgrade: wait for store is up", store_id);
        }
        info!("tikv-server: upgrade finished");
    }

    // After upgrade.
    let start_time = Instant::now();
    while start_time.saturating_elapsed() < test_dur_after_upgrade.0 {
        // Restart nodes.
        random_node_restart(&mut cluster);
    }

    // Downgrade tikv-workers.
    // TODO: rolling downgrade.
    {
        block_on(cluster.tikv_workers_must_healthy(Duration::from_secs(1)));
        cluster.stop_tikv_workers();
        tikv_workers.start_all(cluster.tikv_worker_configs());
        block_on(tikv_workers.must_all_healthy(WAIT_TIKV_WORKER_HEALTHY_TIMEOUT));
        info!("tikv-worker: downgrade finished");
    }

    // Rolling downgrade tikv-servers.
    {
        let nodes = cluster.get_nodes();
        for node_id in nodes {
            std::thread::sleep(TIKV_STORE_UPGRADE_DOWNGRADE_INTERVAL);

            let store_id = cluster.get_store_id(node_id);

            if evict_leader_switch {
                let (store, scheduler_name) =
                    block_on(pd_ctl.evict_store_leaders(store_id, EVICT_LEADERS_TIMEOUT)).unwrap();
                if store.status.leader_count > 0 {
                    warn!("downgrade: store still has leaders after evicting"; "node_id" => node_id, "store" => ?store);
                }
                block_on(pd_ctl.remove_scheduler(&scheduler_name)).unwrap();
            }

            let force = rng.gen_bool(0.2);
            info!("downgrade: stop new version"; "node_id" => node_id, "force" => force);
            cluster.stop_node_force(node_id, force);
            must_wait_store_down("downgrade: wait for store is down", store_id);

            info!("downgrade: start new version"; "node_id" => ?node_id);
            tikv_servers.start_node(node_id, |_, _| {});
            block_on(tikv_servers.must_healthy(node_id, WAIT_TIKV_SERVER_HEALTHY_TIMEOUT));
            must_wait_store_up("downgrade: wait for store is up", store_id);
        }
        info!("tikv-server: downgrade finished");
    }

    // After downgrade.
    let start_time = Instant::now();
    while start_time.saturating_elapsed() < test_dur_after_downgrade.0 {
        sleep(Duration::from_secs(1));
    }

    // Finish.
    info!("test finished, stopping all workers");
    block_on(futures::future::try_join_all(async_handles)).unwrap();

    // Start cluster again for `verify_cluster`.
    {
        block_on(tikv_servers.must_all_healthy(Duration::from_secs(1)));
        let nodes = tikv_servers.get_all_nodes();
        for node_id in nodes {
            info!("final: stop old version"; "node_id" => node_id);
            let exit_status = tikv_servers.stop_node(node_id);
            if !exit_status.success() {
                panic!(
                    "tikv-server exit with error, node_id {}, exit_status {:?}",
                    node_id, exit_status
                );
            }

            info!("final: start new version"; "node_id" => ?node_id);
            cluster.start_node(node_id, |_, _| {});
            let store_id = cluster.get_store_id(node_id);
            must_wait_store_up("final: wait for store is up", store_id);
        }
        info!("tikv-server: final startup finished");
    }

    runtime.block_on(async {
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
    let total_keyspace_count = KEYSPACE_COUNTER.load(Ordering::SeqCst);
    let total_node_restart = NODE_RESTART_COUNTER.load(Ordering::SeqCst);
    let total_tpcc_txns = TPCC_COUNTER.load(Ordering::SeqCst);
    let total_jepsen_bank = JEPSEN_BANK_TXN_COUNTER.load(Ordering::SeqCst);
    let total_jepsen_bank_retry = JEPSEN_BANK_TXN_RETRY_COUNTER.load(Ordering::SeqCst);
    let total_unique_workload = UNIQUE_WORKLOAD_TXN_COUNTER.load(Ordering::SeqCst);
    let total_unique_conflict = UNIQUE_WORKLOAD_CONFLICT_COUNTER.load(Ordering::SeqCst);
    let region_number = pd_client.get_regions_number();

    let rfstore_propose_switch_mem_table =
        rfstore::store::metrics::STORE_PROPOSE_SWITCH_MEM_TABLE_COUNTER.get();
    assert!(rfstore_propose_switch_mem_table > 0);

    info!(
        "TEST SUCCEED: keyspace {}, region {}, node restart {}, tpcc {}, jepsen_bank {} (retry {}), unique_workload {} (conflict {})",
        total_keyspace_count,
        region_number,
        total_node_restart,
        total_tpcc_txns,
        total_jepsen_bank,
        total_jepsen_bank_retry,
        total_unique_workload,
        total_unique_conflict;
        "rfstore_propose_switch_mem_table" => rfstore_propose_switch_mem_table,
    );

    tc.pd.stop_all();
}

fn prepare_tikv_servers(
    data_path: PathBuf,
    working_path: PathBuf,
    nodes_count: usize,
    security_conf: &SecurityConfig,
) -> TikvServers {
    let tikv_server_bin =
        std::env::var(TIKV_SERVER_BIN_ENV_KEY).expect("env TIKV_SERVER_BIN is not set");
    TikvServers::new(
        PathBuf::from(tikv_server_bin),
        data_path,
        working_path,
        nodes_count,
        security_conf,
    )
}

fn prepare_tikv_workers(working_path: PathBuf, security_conf: &SecurityConfig) -> TikvWorkers {
    let tikv_worker_bin =
        std::env::var(TIKV_WORKER_BIN_ENV_KEY).expect("env TIKV_WORKER_BIN is not set");
    TikvWorkers::new(PathBuf::from(tikv_worker_bin), working_path, security_conf)
}

fn prepare_cluster(
    dfs_config: &DFSConfig,
    security_conf: &SecurityConfig,
    nodes_count: usize,
    initial_keyspace_count: usize,
    enable_inner_key_off: bool,
    use_remote_cop: bool,
    enable_schema_manager: bool,
    block_cache_type: BlockCacheType,
    tc: &TidbCluster,
) -> (ServerCluster, TikvServers, TikvWorkers) {
    let mut rng = thread_rng();
    let nodes = alloc_node_id_vec(nodes_count);
    let tikv_worker_nodes = alloc_node_id_vec(TIKV_WORKERS_COUNT);
    let dfs_config = Arc::new(dfs_config.clone());
    let cpu_cores = SysQuota::cpu_cores_quota() as usize;
    let update_conf_fn = |_node_id: u16, conf: &mut TikvConfig| {
        let mut rng = thread_rng();
        conf.dfs = (*dfs_config).clone();
        conf.enable_inner_key_offset = enable_inner_key_off;
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
        conf.kvengine.block_cache_type = block_cache_type;

        conf.storage.flow_control.enable = true;
        conf.storage.scheduler_worker_pool_size = cpu_cores;

        if use_remote_cop {
            let tikv_worker_idx = *tikv_worker_nodes.choose(&mut rng).unwrap();
            let cop_worker_url = tikv_worker_cop_url(tikv_worker_idx);
            conf.kvengine.remote_worker_addr = cop_worker_url.clone();
            conf.kvengine.remote_coprocessor_addr = cop_worker_url;
            conf.kvengine.remote_coprocessor_min_blocks_size = 1024 * 1024;
        }
    };

    let pd_wrapper = PdWrapper::new_real(tc.pd.endpoints(), security_conf);
    let mut cluster = ServerCluster::new_opt(vec![], |_, _| {}, pd_wrapper);

    // Start tikv-servers.
    let mut tikv_servers = prepare_tikv_servers(
        cluster.data_dir().to_path_buf(),
        tc.data_path().to_path_buf(),
        NODES_COUNT,
        security_conf,
    );
    for node_id in nodes {
        // Start tikv-servers one by one to work around the conflict on bootstrap
        // cluster.
        tikv_servers.start_node(node_id, update_conf_fn);
        block_on(tikv_servers.must_healthy(node_id, WAIT_TIKV_SERVER_HEALTHY_TIMEOUT));
    }
    for (node_id, conf) in tikv_servers.configs() {
        cluster.update_node_config(*node_id, conf.clone());
    }

    // Start tikv-workers.
    cluster.generate_tikv_worker_configs(
        tikv_worker_nodes,
        TikvWorkerOptions {
            cop_block_cache_size: COP_BLOCK_CACHE_SIZE,
            cop_block_cache_type: block_cache_type,
            ..Default::default()
        },
    );
    let mut tikv_workers = prepare_tikv_workers(tc.data_path().to_path_buf(), security_conf);
    tikv_workers.start_all(cluster.tikv_worker_configs());
    block_on(tikv_workers.must_all_healthy(WAIT_TIKV_WORKER_HEALTHY_TIMEOUT));

    if enable_schema_manager {
        cluster.start_schema_manager(alloc_node_id());
    }
    cluster.wait_region_replicated(&[], 3);

    let mut keyspaces: Vec<u32> = vec![];
    let mut keyspace_names: Vec<String> = vec![];

    let pd_control = tc.pd.get_pd_control();

    // Initial keyspaces have been created by `pre_alloc_keyspaces` of PD.
    // Note: keyspaces allocation starts from 1.
    for idx in 1..=initial_keyspace_count {
        let keyspace_name = TidbCluster::keyspace_name(idx as u16);
        let keyspace = block_on(pd_control.get_keyspace_by_name(&keyspace_name)).unwrap();
        keyspaces.push(keyspace.id);
        keyspace_names.push(keyspace_name);
    }

    // TODO: create by API to be uniform with test PD.
    // TODO: enable encryption.
    cluster.keyspace_manager().create_keyspaces(
        &keyspaces,
        keyspace_names,
        DEFAULT_INNER_KEY_OFFSET,
        0,
        0.0,
        Some(&mut rng),
    );
    KEYSPACE_COUNTER.store(initial_keyspace_count, Ordering::Relaxed);

    // TODO: scatter regions.
    // TODO: restart cluster with inner key offset enabled

    (cluster, tikv_servers, tikv_workers)
}

fn must_wait_store_state<F>(
    ctx: &str,
    pd_ctl: &PdControl,
    store_id: u64,
    expect: F,
    timeout: Duration,
) where
    F: Fn(&pd_client::pd_control::StoreInfo) -> bool,
{
    must_wait(
        || {
            let store = block_on(pd_ctl.get_store(store_id)).unwrap();
            expect(&store)
        },
        timeout.as_secs() as usize,
        || {
            let store = block_on(pd_ctl.get_store(store_id)).unwrap();
            format!("{ctx}: store state not match: {store:?}")
        },
    );
}
