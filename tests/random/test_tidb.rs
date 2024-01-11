// Copyright 2023 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    ops::Div,
    path::PathBuf,
    sync::{atomic::Ordering, Arc},
    time::Duration,
};

use kvengine::dfs::DFSConfig;
use pd_client::pd_control::PdScheduleConfig;
use rand::Rng;
use security::SecurityConfig;
use test_cloud_server::{oss::prepare_dfs, tidb::*, try_wait_result, ServerCluster};
use test_pd_client::PdWrapper;
use tikv::config::TikvConfig;
use tikv_util::{
    config::{ReadableDuration, ReadableSize},
    info,
    time::Instant,
};

use crate::*;

const REGION_SIZE: ReadableSize = ReadableSize::mb(1);
// TiDB has records with 200kb+ size (see "mysql.stats_history"), so set bucket
// size to 256kb.
const REGION_BUCKET_SIZE: ReadableSize = ReadableSize::kb(256);
// Ref: https://docs.pingcap.com/tidb/stable/pd-configuration-file#split-merge-interval
const SPLIT_MERGE_INTERVAL: ReadableDuration = ReadableDuration::secs(10);

const INITIAL_KEYSPACE_COUNT: usize = 4;
const NODES_COUNT: usize = 4;

const PD_COUNT: usize = 1;
const PD_BIN_ENV_KEY: &str = "PD_BIN";
const PD_PORT_ENV_KEY: &str = "PD_PORT";
const PD_PORT_DEFAULT: u16 = 2379;

const TIDB_BIN_ENV_KEY: &str = "TIDB_BIN";
const TIDB_PORT_ENV_KEY: &str = "TIDB_PORT";
const TIDB_PORT_DEFAULT: u16 = 4000;
const TIDB_STATUS_PORT_ENV_KEY: &str = "TIDB_STATUS_PORT";
const TIDB_STATUS_PORT_DEFAULT: u16 = 10080;

#[test]
fn test_random_with_tidb() {
    test_util::init_log_for_test();
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .worker_threads(4)
        .thread_name("random-workload")
        .build()
        .unwrap();
    let _guard = runtime.enter();

    // Prepare.
    let (_temp_dir, _oss, dfs_config) = prepare_dfs("oss_");
    let security_conf = new_security_config();
    let tc = prepare_tidb_cluster(&security_conf);
    let mut cluster = prepare_cluster(
        &dfs_config,
        &security_conf,
        NODES_COUNT,
        INITIAL_KEYSPACE_COUNT,
        Some(&tc),
    );
    let pd_client = cluster.get_pd_client_ext();

    block_on(tc.start_tidb(INITIAL_KEYSPACE_COUNT as u16, Duration::from_secs(30)));

    // Main loop.
    let start_time = Instant::now();
    while start_time.saturating_elapsed() < Duration::from_secs(10) {
        // Restart nodes.
        random_node_restart(&mut cluster);
    }

    // Finish.
    info!("test finished, stopping all workers");

    // Verify.
    info!("verify cluster");
    let verified_records_count = runtime.block_on(verify_cluster(&mut cluster));

    // Stop cluster.
    info!("stopping cluster");
    tc.tidb.stop_all();
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
    let region_number = pd_client.get_regions_number();
    info!(
        "TEST SUCCEED: write {}, keyspace {}, table {}, drop table {}, region {}, merge {}, move {}, transfer {}, node restart {}, backup {}, restore {}, load_data {}, manual_major_compact {}, verified_records {}, gc {}",
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
        verified_records_count,
        total_gc_resolved_locks,
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

    let max_merge_region_size = REGION_SIZE.div(5);
    let max_merge_region_keys = max_merge_region_size.0 / 100; // Assume 100 bytes per key, about 2000 keys.
    let pd_scheduler_config = PdScheduleConfig {
        max_merge_region_size: max_merge_region_size.as_mb().max(1),
        max_merge_region_keys,
        split_merge_interval: SPLIT_MERGE_INTERVAL,
        ..Default::default()
    };

    let tc = TidbCluster::new(
        PathBuf::from(pd_bin),
        pd_port_base,
        pd_scheduler_config,
        PathBuf::from(tidb_bin),
        tidb_port_base,
        tidb_status_port_base,
        INITIAL_KEYSPACE_COUNT as u16,
        security_config,
    );
    block_on(tc.start_pd(PD_COUNT as u16, Duration::from_secs(30)));
    tc
}

// TODO: merge to `prepare_cluster` in `test_all.rs`.
fn prepare_cluster(
    dfs_config: &DFSConfig,
    security_conf: &SecurityConfig,
    nodes_count: usize,
    initial_keyspace_count: usize,
    tc: Option<&TidbCluster>,
) -> ServerCluster {
    let mut rng = rand::thread_rng();
    let nodes = alloc_node_id_vec(nodes_count);
    let dfs_config = Arc::new(dfs_config.clone());
    let update_conf_fn = move |_, conf: &mut TikvConfig| {
        conf.dfs = (*dfs_config).clone();
        conf.coprocessor.region_split_size = REGION_SIZE;
        conf.coprocessor.region_bucket_size = REGION_BUCKET_SIZE;
        conf.raft_store.peer_stale_state_check_interval = ReadableDuration::secs(1);
        conf.raft_store.abnormal_leader_missing_duration = ReadableDuration::secs(3);
        conf.raft_store.max_leader_missing_duration = ReadableDuration::secs(5);
        conf.rocksdb.writecf.block_size = ReadableSize::kb(4);
        conf.rocksdb.writecf.target_file_size_base = ReadableSize::kb(16);
        conf.rfengine.target_file_size = ReadableSize::mb(8);
        conf.rfengine.batch_compression_threshold =
            ReadableSize::kb(rand::thread_rng().gen_range(0..2));
        conf.rfengine.lightweight_backup = true;
        conf.rfengine.wal_chunk_target_file_size = ReadableSize::kb(512);
        // TODO: test for both enable and disable inner_key_offset
        conf.enable_inner_key_offset = true;
        conf.security = security_conf.clone();
        conf.kvengine.compaction_tombs_count = 100;
        conf.kvengine.max_del_range_delay = ReadableDuration(Duration::from_secs(3));
    };

    let pd_wrapper = match tc {
        Some(tc) => PdWrapper::new_real(tc.pd.endpoints(), security_conf),
        None => PdWrapper::new_test(1, security_conf),
    };
    let cluster = ServerCluster::new_opt(nodes, update_conf_fn, pd_wrapper);
    cluster.wait_region_replicated(&[], 3);

    let mut keyspaces: Vec<u32> = vec![];
    let mut keyspace_names: Vec<String> = vec![];
    match tc {
        Some(tc) => {
            let pd_control = tc.pd.get_pd_control();

            // Initial keyspaces have been created by `pre_alloc_keyspaces` of PD.
            // Note: starts from 1.
            for idx in 1..initial_keyspace_count {
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

// TODO: merge to `verify_cluster` in `test_all.rs`.
async fn verify_cluster(cluster: &mut ServerCluster) -> usize {
    // Check statistics.
    // Check after verify data, to ensure that PD heartbeat have updated region
    // stats.
    let (res, data_stats) = try_wait_result(
        || {
            let stats = cluster.get_data_stats();
            (stats.check_data(), stats)
        },
        20,
    );
    assert!(
        res.is_ok(),
        "check_data failed: {:?}, stats: {:?}",
        res,
        data_stats
    );
    cluster.wait_region_version_match();
    data_stats
        .check_buckets(cluster.get_pd_client_ext().as_ref(), REGION_BUCKET_SIZE.0)
        .unwrap();

    0
}
