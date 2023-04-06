// Copyright 2023 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    sync::{atomic::Ordering, Arc, Mutex, RwLock},
    thread::{sleep, JoinHandle},
    time::Duration,
};

use kvengine::dfs::{DFSConfig, S3Fs};
use native_br::{backup, restore_keyspace};
use pd_client::PdClient;
use rand::{prelude::SliceRandom, Rng};
use test_cloud_server::{client::ClusterClient, try_wait, ServerCluster};
use tikv::config::TikvConfig;
use tikv_util::{
    config::{ReadableDuration, ReadableSize},
    error, info,
    time::Instant,
};
use tokio::runtime::Runtime;

use crate::{
    alloc_node_id_vec, get_keyspace_prefix, prepare_dfs, spawn_gc_worker, spawn_keyspace_write,
    spawn_move, spawn_transfer, CONCURRENCY, MERGE_COUNTER, MOVE_COUNTER, TIMEOUT,
    TRANSFER_COUNTER, WRITE_COUNTER,
};

const KEYSPACE_COUNT: usize = 10;

#[test]
fn test_random_br() {
    test_util::init_log_for_test();

    let (_temp_dir, _oss, dfs_config) = prepare_dfs("random_br_");

    // Prepare cluster.
    let nodes = alloc_node_id_vec(5);
    let update_conf_fn = |_, conf: &mut TikvConfig| {
        conf.dfs = dfs_config.clone();
        conf.coprocessor.region_split_size = ReadableSize::kb(192);
        conf.coprocessor.region_bucket_size = ReadableSize::kb(64);
        conf.raft_store.peer_stale_state_check_interval = ReadableDuration::secs(1);
        conf.raft_store.abnormal_leader_missing_duration = ReadableDuration::secs(3);
        conf.raft_store.max_leader_missing_duration = ReadableDuration::secs(5);
        conf.rocksdb.writecf.target_file_size_base = ReadableSize::kb(16);
        conf.rfengine.target_file_size = ReadableSize::mb(1);
        conf.rfengine.batch_compression_threshold =
            ReadableSize::kb(rand::thread_rng().gen_range(0..2));
    };
    let mut cluster = ServerCluster::new(nodes, update_conf_fn);
    cluster.wait_region_replicated(&[], 3);
    let pd_client = cluster.get_pd_client();
    pd_client.disable_default_operator();
    let mut client = cluster.new_client();

    // Split keyspaces.
    for keyspace_id in 0..=KEYSPACE_COUNT {
        client.split(&get_keyspace_prefix(keyspace_id as u32));
    }
    cluster.wait_pd_region_min_count(KEYSPACE_COUNT + 2);

    let move_scheduler = cluster.new_scheduler();
    for _ in 0..20 {
        move_scheduler.move_random_region();
    }

    let backups = Arc::new(Mutex::new(vec![]));

    // Start workloads & schedulers.
    // TODO: restart nodes
    // TODO: add scheduler: merge inside keyspace
    let mut handles = vec![
        spawn_transfer(cluster.new_scheduler()),
        spawn_move(cluster.new_scheduler(), Arc::new(RwLock::new(()))),
        spawn_gc_worker(cluster.get_pd_client(), TIMEOUT),
        spawn_incremental_backup(
            backups.clone(),
            cluster.new_client(),
            cluster.get_pd_client(),
            dfs_config.clone(),
            TIMEOUT,
        ),
    ];
    for idx in 0..CONCURRENCY {
        handles.push(spawn_keyspace_write(
            idx,
            cluster.new_client(),
            KEYSPACE_COUNT,
            TIMEOUT,
        ));
    }

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(8)
        .enable_all()
        .thread_name("restore-keyspace")
        .build()
        .unwrap();
    let mut rng = rand::thread_rng();

    let start_time = Instant::now();
    while start_time.saturating_elapsed() < TIMEOUT {
        let backup_name = match backups.lock().unwrap().choose(&mut rng) {
            Some(name) => name.clone(),
            None => {
                sleep(Duration::from_millis(100));
                continue;
            }
        };

        let keyspace = rng.gen_range(0..KEYSPACE_COUNT) as u32;
        do_restore_keyspace(
            &mut cluster,
            &runtime,
            dfs_config.clone(),
            keyspace,
            &backup_name,
        );
        info!(
            "restore keyspace {} from backup {} success",
            keyspace, backup_name
        );
    }

    info!("test finished, stopping all workers");
    for handle in handles {
        handle.join().unwrap();
    }
    let ok = try_wait(
        || {
            let data_stats = cluster.get_data_stats();
            data_stats.check_data().is_ok()
        },
        20,
    );
    if !ok {
        if let Err(str) = cluster.get_data_stats().check_data() {
            if str.contains("compaction score too large") {
                // TODO: investigate why compaction score too large
                error!("{}", str);
            } else {
                panic!("{}", str);
            }
        }
    }

    // TODO: verify data
    // let mut client = cluster.new_client();
    // client.verify_data_with_ref_store();

    cluster.stop();
    let total_write_count = WRITE_COUNTER.load(Ordering::SeqCst);
    let total_merge_count = MERGE_COUNTER.load(Ordering::SeqCst);
    let total_move_count = MOVE_COUNTER.load(Ordering::SeqCst);
    let total_transfer_count = TRANSFER_COUNTER.load(Ordering::SeqCst);
    let region_number = pd_client.get_regions_number();
    info!(
        "total_write_count {}, region number {}, merge count {}, move count {}, transfer count {}",
        total_write_count, region_number, total_merge_count, total_move_count, total_transfer_count,
    );
}

fn do_restore_keyspace(
    cluster: &mut ServerCluster,
    runtime: &Runtime,
    dfs_config: DFSConfig,
    keyspace: u32,
    backup_name: &str,
) {
    let s3fs = Arc::new(S3Fs::new(
        dfs_config.prefix,
        dfs_config.s3_endpoint,
        dfs_config.s3_key_id,
        dfs_config.s3_secret_key,
        dfs_config.s3_region,
        dfs_config.s3_bucket,
    ));
    restore_keyspace::restore_keyspace(
        keyspace,
        backup_name,
        None,
        s3fs,
        cluster.get_pd_client(),
        runtime,
    )
    .unwrap();
}

fn spawn_incremental_backup(
    backups: Arc<Mutex<Vec<String>>>,
    client: ClusterClient,
    pd_client: Arc<dyn PdClient>,
    dfs_config: DFSConfig,
    timeout: Duration,
) -> JoinHandle<()> {
    std::thread::spawn(move || {
        let backup_config = backup::BackupConfig {
            dfs: dfs_config,
            tolerate_err: 1,
            skip_keyspace_meta: true,
            ..Default::default()
        };

        let start_time = Instant::now();
        let mut last_backup_meta = None;
        let mut idx = 0;
        while start_time.saturating_elapsed() < timeout {
            idx += 1;
            let backup_name = format!("backup_{}", idx);
            let backup_ts = client.get_ts().into_inner();
            let backup_meta = backup::backup_cluster_with_ts(
                backup_config.clone(),
                last_backup_meta.is_some(),
                backup_name.clone(),
                pd_client.as_ref(),
                backup_ts,
                last_backup_meta.take(),
            )
            .expect("backup::backup_cluster");
            info!("backup done: {}: {:?}", backup_name, backup_meta);

            backups.lock().unwrap().push(backup_name);
            last_backup_meta = Some(backup_meta);
            sleep(Duration::from_secs(3));
        }
        info!("incremental backup thread exit");
    })
}
