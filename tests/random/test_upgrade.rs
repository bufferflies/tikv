// Copyright 2024 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    path::PathBuf,
    sync::{atomic::Ordering, Arc},
    time::Duration,
};

use kvengine::dfs::DFSConfig;
use pd_client::pd_control::PdControl;
use rand::prelude::*;
use security::SecurityConfig;
use test_cloud_server::{
    must_wait,
    oss::prepare_dfs,
    tidb::*,
    tikv_bin::{TikvServers, TikvWorkers},
    ServerCluster, TikvWorkerOptions,
};
use test_pd_client::PdWrapper;
use tikv_util::{config::ReadableDuration, info, time::Instant};

use crate::{test_tidb::*, *};

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

    let temp_dir = std::env::temp_dir();
    tikv_util::set_panic_hook(false, temp_dir.to_str().unwrap()); // To prevent temp dirs from being dropped on error.

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .worker_threads(4)
        .thread_name("random-workload")
        .build()
        .unwrap();
    let _guard = runtime.enter();

    let switches = Switches::from_env();
    let upgrade_switches = UpgradeTestSwitches::from_env();
    info!("switches: {:?}, {:?}", switches, upgrade_switches);

    // Prepare.
    let (_temp_dir, _oss, dfs_config) = prepare_dfs("oss_");
    let security_conf = new_security_config();
    let tc = prepare_tidb_cluster(&security_conf);
    let (mut cluster, mut tikv_servers, mut tikv_workers) = prepare_cluster(
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

    let mut rng = thread_rng();

    let tikv_worker_addr = tikv_workers.endpoints().pop().unwrap();
    start_components(&tc, tikv_worker_addr, &switches, &dfs_config, &runtime);
    prepare_workloads(&tc, &keyspace_manager, &switches, &runtime);
    let running = Running::new_start();
    let async_handles =
        start_workloads(&tc, &keyspace_manager, &switches, &runtime, running.clone());

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
    while start_time.saturating_elapsed() < upgrade_switches.test_dur_before_upgrade.0 {
        sleep(Duration::from_secs(1));
    }
    info!("before upgrade: finished"; "stats" => ?WorkloadStats::collect());

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

            if upgrade_switches.evict_leader_switch {
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
    while start_time.saturating_elapsed() < upgrade_switches.test_dur_after_upgrade.0 {
        // Restart nodes.
        random_node_restart(&mut cluster);
    }
    info!("after upgrade: finished"; "stats" => ?WorkloadStats::collect());

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

            if upgrade_switches.evict_leader_switch {
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
    while start_time.saturating_elapsed() < upgrade_switches.test_dur_after_downgrade.0 {
        sleep(Duration::from_secs(1));
    }

    // Finish.
    info!("test finished, stopping all workers");
    running.stop();
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
        check_and_stop_components(&tc).await;
        stop_schedulers(pd_ctl).await;

        info!("verify cluster");
        verify_cluster(&mut cluster, &switches).await;
    });

    // Stop cluster.
    info!("stopping cluster");
    cluster.stop();

    let rfstore_propose_switch_mem_table =
        rfstore::store::metrics::STORE_PROPOSE_SWITCH_MEM_TABLE_COUNTER.get();
    assert!(rfstore_propose_switch_mem_table > 0);

    let region_number = pd_client.get_regions_number();
    tc.pd.stop_all();

    // Statistics.
    let stats = WorkloadStats::collect();

    info!("TEST SUCCEED: region_number {}, {:?}", region_number, stats);
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
    switches: &Switches,
    tc: &TidbCluster,
) -> (ServerCluster, TikvServers, TikvWorkers) {
    let mut rng = thread_rng();
    let nodes = alloc_node_id_vec(nodes_count);
    let tikv_worker_nodes = alloc_node_id_vec(TIKV_WORKERS_COUNT);
    let update_conf_fn =
        generate_update_conf_fn(dfs_config, security_conf, &tikv_worker_nodes, switches);
    let pd_wrapper =
        PdWrapper::new_real(tc.pd.endpoints(), security_conf, PD_CLIENT_UPDATE_INTERVAL);
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
        tikv_servers.start_node(node_id, &update_conf_fn);
        block_on(tikv_servers.must_healthy(node_id, WAIT_TIKV_SERVER_HEALTHY_TIMEOUT));
    }
    for (node_id, conf) in tikv_servers.configs() {
        cluster.update_node_config(*node_id, conf.clone());
    }

    // Start tikv-workers.
    cluster.generate_tikv_worker_configs(
        tikv_worker_nodes.clone(),
        TikvWorkerOptions {
            cop_block_cache_size: COP_BLOCK_CACHE_SIZE,
            cop_block_cache_type: switches.block_cache_type,
            ..Default::default()
        },
    );
    let mut tikv_workers = prepare_tikv_workers(tc.data_path().to_path_buf(), security_conf);
    tikv_workers.start_all(cluster.tikv_worker_configs());
    block_on(tikv_workers.must_all_healthy(WAIT_TIKV_WORKER_HEALTHY_TIMEOUT));

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

#[derive(Debug)]
struct UpgradeTestSwitches {
    test_dur_before_upgrade: ReadableDuration,
    test_dur_after_upgrade: ReadableDuration,
    test_dur_after_downgrade: ReadableDuration,
    evict_leader_switch: bool,
}

impl UpgradeTestSwitches {
    fn from_env() -> Self {
        let mut rng = thread_rng();

        let test_dur_before_upgrade = env_param(
            "TEST_DUR_BEFORE_UPGRADE",
            ReadableDuration(crate::test_upgrade::TEST_DURATION_BEFORE_UPGRADE),
        );
        let test_dur_after_upgrade = env_param(
            "TEST_DUR_AFTER_UPGRADE",
            ReadableDuration(crate::test_upgrade::TEST_DURATION_AFTER_UPGRADE),
        );
        let test_dur_after_downgrade = env_param(
            "TEST_DUR_AFTER_DOWNGRADE",
            ReadableDuration(crate::test_upgrade::TEST_DURATION_AFTER_DOWNGRADE),
        );

        let evict_leader_switch = rng.gen_ratio(1, 5);

        Self {
            test_dur_before_upgrade,
            test_dur_after_upgrade,
            test_dur_after_downgrade,
            evict_leader_switch,
        }
    }
}
