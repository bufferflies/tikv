// Copyright 2025 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    thread,
    time::Duration,
};

use chrono::Utc;
use cloud_worker::native_br::{test_utils::NativeBrSvcClient, BackupItem, RestoreState};
use collections::HashMap;
use futures::executor::block_on;
use kvengine::dfs::S3Fs;
use kvproto::{
    metapb,
    metapb::StoreState::{Offline, Tombstone, Up},
};
use native_br::{
    backup,
    restore::RestoreConfig,
    restore_keyspace,
    restore_keyspace::{ReportRestoreStepTrait, RestoreStep},
};
use pd_client::PdClient;
use security::{SecurityConfig, SecurityManager};
use test_cloud_server::{
    alloc_node_id, alloc_node_id_vec, client::RequestOptions, oss::prepare_dfs, ServerCluster,
    ServerClusterBuilder, TikvWorkerOptions, TryWaiter,
};
use test_pd_client::PdWrapper;
use tikv::config::TikvConfig;
use tikv_util::info;
use tokio::runtime::Runtime;

use crate::cases::{i_to_keyspace_key, random_value};

#[test]
fn test_backup_on_scaling_up() {
    test_util::init_log_for_test();
    const KEYSPACE_ID: u32 = 1;
    const DATA_LEN: usize = 100;
    const VALUE_SIZE: usize = 64;

    let get_all_stores_fp = "test_pd::get_all_stores";

    let (_temp_dir, mut oss, dfs_config) = prepare_dfs("t_");
    let s3fs = Arc::new(S3Fs::new_from_config(dfs_config.clone()));
    let reporter = Arc::new(DummyStepReporter::default());
    let runtime = Runtime::new().unwrap();

    let nodes = alloc_node_id_vec(6);
    let mut cluster = ServerCluster::new(nodes.clone(), |_, conf: &mut TikvConfig| {
        conf.dfs = dfs_config.clone();
        conf.rfengine.lightweight_backup = true;
        conf.enable_inner_key_offset = true;
    });
    cluster.wait_region_replicated(&[], 3);
    let pd_client = cluster.get_pd_client();
    let mut client = cluster.new_client();
    client.split_keyspace(KEYSPACE_ID);

    let stores = nodes
        .iter()
        .map(|&node_id| {
            let store = pd_client.get_store(cluster.get_store_id(node_id)).unwrap();
            (node_id, store)
        })
        .collect::<HashMap<_, _>>();

    for idx in [3, 4, 5] {
        stop_node(&mut cluster, nodes[idx], &stores, Tombstone);
    }

    let i_to_key = i_to_keyspace_key(KEYSPACE_ID);
    client.put_kv(0..DATA_LEN, &i_to_key, random_value::<VALUE_SIZE>);
    client.verify_data_with_ref_store();
    let origin_ref_store = client.dump_ref_store();

    let backup_config = backup::BackupConfig {
        dfs: dfs_config.clone(),
        tolerate_err: 1,
        skip_keyspace_meta: true,
        ..Default::default()
    };

    // The first invoke of `get_all_stores` during backup will succeed, then
    // blocked.
    fail::cfg(get_all_stores_fp, "1*off->pause").unwrap();

    // Perform backup.
    let backup_name = generate_backup_name();
    thread::scope(|s| {
        stop_node(&mut cluster, nodes[0], &stores, Offline);

        let backup_config = backup_config.clone();
        let backup_name = backup_name.clone();
        let pd_client = pd_client.clone();
        let backup_ts = client.get_ts().into_inner();
        let h = s.spawn(move || {
            backup::backup_cluster_with_ts(
                backup_config,
                backup::BackupType::Lightweight,
                backup_name,
                pd_client.as_ref(),
                backup_ts,
                None,
            )
        });

        thread::sleep(Duration::from_secs(2));

        // This topology change will be tolerated.
        start_node(&mut cluster, nodes[3], &stores, Up);
        fail::cfg(get_all_stores_fp, "off").unwrap();

        let (_, backup_meta) = h.join().unwrap().expect("backup");
        info!("backup cluster result: {}", backup_meta);
        assert_eq!(backup_meta.get_stores().len(), 3);
        assert_eq!(backup_meta.tolerated_err, 1);
    });

    // Put another data.
    client.put_kv(
        DATA_LEN / 4..DATA_LEN / 2,
        &i_to_key,
        random_value::<VALUE_SIZE>,
    );
    client.verify_data_with_ref_store();
    let origin_ref_store_2 = client.dump_ref_store();

    fail::cfg(get_all_stores_fp, "1*off->pause").unwrap();
    // Perform backup.
    let backup_name_2 = generate_backup_name();
    thread::scope(|s| {
        start_node(&mut cluster, nodes[0], &stores, Up);

        let backup_name = backup_name_2.clone();
        let backup_ts = client.get_ts().into_inner();
        let h = s.spawn(move || {
            backup::backup_cluster_with_ts(
                backup_config,
                backup::BackupType::Lightweight,
                backup_name,
                pd_client.as_ref(),
                backup_ts,
                None,
            )
        });

        thread::sleep(Duration::from_secs(2));

        // This topology change can not be tolerated, then retry to succeed.
        for idx in [4, 5] {
            start_node(&mut cluster, nodes[idx], &stores, Up);
        }
        fail::cfg(get_all_stores_fp, "off").unwrap();

        let (_, backup_meta) = h.join().unwrap().expect("backup");
        info!("backup cluster result: {}", backup_meta);
        assert_eq!(backup_meta.get_stores().len(), 6);
        assert_eq!(backup_meta.tolerated_err, 0);
    });

    // More data at last.
    client.put_kv(
        DATA_LEN / 3..DATA_LEN * 2 / 3,
        &i_to_key,
        random_value::<VALUE_SIZE>,
    );
    client.verify_data_with_ref_store();

    // Perform restore to verify backup.
    for (bk_name, ref_store) in [
        (backup_name, origin_ref_store),
        (backup_name_2, origin_ref_store_2),
    ] {
        let restore_config = RestoreConfig {
            tolerate_err: 0,
            strict_tolerate: true,
            ..Default::default()
        };
        let res = restore_keyspace::restore_keyspace(
            KEYSPACE_ID,
            KEYSPACE_ID,
            &bk_name,
            None,
            s3fs.clone(),
            restore_config.clone(),
            cluster.get_pd_client(),
            &runtime,
            None,
            reporter.clone(),
        )
        .expect("restore");
        info!("restore keyspace result for backup {}: {:?}", bk_name, res);
        let (existed, deleted) = client
            .verify_data_with_given_ref_store(&ref_store, None, &RequestOptions::default())
            .unwrap();
        assert_eq!(existed, DATA_LEN);
        assert_eq!(deleted, 0);
    }

    fail::remove(get_all_stores_fp);
    cluster.stop();
    oss.shutdown();
}

#[test]
fn test_restore_on_disk_full() {
    test_util::init_log_for_test();
    const KEYSPACE_ID: u32 = 1;
    const DATA_LEN: usize = 10;
    const VALUE_SIZE: usize = 64;

    let low_space_fp = "engine_is_low_space";

    let (_temp_dir, mut oss, dfs_config) = prepare_dfs("t_");
    let s3fs = Arc::new(S3Fs::new_from_config(dfs_config.clone()));
    let reporter = Arc::new(DummyStepReporter::default());
    let runtime = Runtime::new().unwrap();

    let nodes = alloc_node_id_vec(3);
    let mut cluster = ServerCluster::new(nodes.clone(), |_, conf: &mut TikvConfig| {
        conf.dfs = dfs_config.clone();
        conf.rfengine.lightweight_backup = true;
        conf.enable_inner_key_offset = true;
    });
    cluster.wait_region_replicated(&[], 3);
    let pd_client = cluster.get_pd_client();
    let mut client = cluster.new_client();
    client.split_keyspace(KEYSPACE_ID);

    let i_to_key = i_to_keyspace_key(KEYSPACE_ID);
    client.put_kv(0..DATA_LEN, &i_to_key, random_value::<VALUE_SIZE>);
    client.verify_data_with_ref_store();
    let origin_ref_store = client.dump_ref_store();

    let backup_config = backup::BackupConfig {
        dfs: dfs_config.clone(),
        skip_keyspace_meta: true,
        ..Default::default()
    };

    // Perform backup.
    let backup_name = generate_backup_name();
    let pd_client = pd_client.clone();
    let backup_ts = client.get_ts().into_inner();
    backup::backup_cluster_with_ts(
        backup_config,
        backup::BackupType::Lightweight,
        backup_name.clone(),
        pd_client.as_ref(),
        backup_ts,
        None,
    )
    .expect("backup");

    // Put another data.
    client.put_kv(
        DATA_LEN / 4..DATA_LEN / 2,
        &i_to_key,
        random_value::<VALUE_SIZE>,
    );
    client.verify_data_with_ref_store();

    fail::cfg(low_space_fp, "return").unwrap();

    thread::scope(|s| {
        let pd_client = cluster.get_pd_client();
        let h = s.spawn(move || {
            restore_keyspace::restore_keyspace(
                KEYSPACE_ID,
                KEYSPACE_ID,
                &backup_name,
                None,
                s3fs.clone(),
                RestoreConfig::default(),
                pd_client,
                &runtime,
                None,
                reporter,
            )
        });

        thread::sleep(Duration::from_secs(3));
        fail::cfg(low_space_fp, "off").unwrap();

        let res = h.join().unwrap().expect("restore");
        info!("restore keyspace result: {:?}", res);
    });
    let (existed, deleted) = client
        .verify_data_with_given_ref_store(&origin_ref_store, None, &RequestOptions::default())
        .unwrap();
    assert_eq!(existed, DATA_LEN);
    assert_eq!(deleted, 0);

    fail::remove(low_space_fp);
    cluster.stop();
    oss.shutdown();
}

#[test]
fn test_native_br_service() {
    test_util::init_log_for_test();
    const KEYSPACE_ID: u32 = 1;
    const DATA_LEN: usize = 100;
    const VALUE_SIZE: usize = 64;

    let mock_get_keyspace_fp = "pd_ctl::mock_get_keyspace_by_name";
    let mock_no_tiflash = "pd_ctl::mock_no_tiflash_placement_rule_group";

    let runtime = Runtime::new().unwrap();
    let _enter = runtime.enter();

    let (_temp_dir, mut oss, dfs_config) = prepare_dfs("t_");
    let pd_wrapper = PdWrapper::new_test(1, &SecurityConfig::default(), None);
    let mut cluster = ServerClusterBuilder::new(alloc_node_id_vec(3), |_, conf| {
        conf.dfs = dfs_config.clone();
        conf.rfengine.lightweight_backup = true;
        conf.enable_inner_key_offset = true;
    })
    .pd(pd_wrapper)
    .build();
    let tikv_worker_id = alloc_node_id();
    let tikv_worker_opts = TikvWorkerOptions {
        backup_interval: Duration::from_millis(1010),
        backup_skip_keyspace_meta: true,
        restore_timeout_pd_control: Duration::from_secs(3),
        ..Default::default()
    };
    cluster.start_tikv_workers(vec![tikv_worker_id], tikv_worker_opts.clone());
    cluster.wait_region_replicated(&[], 3);

    let pd_client = cluster.get_pd_client();
    let cluster_id = pd_client.get_cluster_id().unwrap();

    let mut client = cluster.new_client();
    client.split_keyspace(KEYSPACE_ID);

    let i_to_key = i_to_keyspace_key(KEYSPACE_ID);
    client.put_kv(0..DATA_LEN, &i_to_key, random_value::<VALUE_SIZE>);
    client.verify_data_with_ref_store();

    let datetime0 = Utc::now();
    let ref_store0 = client.dump_ref_store();

    let security_mgr = Arc::new(SecurityManager::default());
    let br_cli =
        NativeBrSvcClient::new(cluster_id, cluster.tikv_worker_endpoints(), security_mgr).unwrap();
    let backup = TryWaiter::timeout(10)
        .interval(1)
        .try_wait_result(|| {
            let mut backups = block_on(br_cli.list_backups(&datetime0)).unwrap();
            backups
                .items
                .pop()
                .ok_or(Err::<BackupItem, String>("no backup found".to_string()))
        })
        .unwrap();
    info!("backup: {:?}", backup);

    // More data.
    client.put_kv(
        DATA_LEN / 3..DATA_LEN * 2 / 3,
        &i_to_key,
        random_value::<VALUE_SIZE>,
    );
    client.verify_data_with_ref_store();

    // Restore.
    let keyspace_name = format!("ks{KEYSPACE_ID}");
    let progress = block_on(br_cli.restore_keyspace(1, keyspace_name.clone(), &backup)).unwrap();
    info!("create restore: {:?}", progress);

    // Must fail due to get keyspace name error.
    TryWaiter::timeout(30).interval(1).must_wait(
        || {
            let progress = block_on(br_cli.get_restore_progress(1, keyspace_name.clone())).unwrap();
            info!("restore progress: {:?}", progress);
            progress.status == RestoreState::Error
        },
        || "restore not error".to_string(),
    );

    // Mock PD control & retry.
    fail::cfg(mock_get_keyspace_fp, "return").unwrap();
    fail::cfg(mock_no_tiflash, "return").unwrap();

    // Restore tikv worker.
    cluster.stop_tikv_workers();
    cluster.start_tikv_workers(vec![tikv_worker_id], tikv_worker_opts.clone());

    // Check task persistence.
    let progress = block_on(br_cli.get_restore_progress(1, keyspace_name.clone())).unwrap();
    info!("restore progress (after restart): {:?}", progress);
    assert_eq!(progress.status, RestoreState::Error);
    assert_eq!(progress.error, "interrupted");

    // Retry restore.
    let progress = block_on(br_cli.restore_keyspace(1, keyspace_name.clone(), &backup)).unwrap();
    info!("retry restore: {:?}", progress);

    // Wait succeed.
    TryWaiter::timeout(30).interval(1).must_wait(
        || {
            let progress = block_on(br_cli.get_restore_progress(1, keyspace_name.clone())).unwrap();
            info!("restore progress: {:?}", progress);
            assert_ne!(
                progress.status,
                RestoreState::Error,
                "progress: {:?}",
                progress
            );
            progress.status == RestoreState::Succeed
        },
        || "restore not succeed".to_string(),
    );

    // Verify data.
    let (existed, deleted) = client
        .verify_data_with_given_ref_store(&ref_store0, None, &RequestOptions::default())
        .unwrap();
    assert_eq!(existed, DATA_LEN);
    assert_eq!(deleted, 0);

    fail::remove(mock_get_keyspace_fp);
    fail::remove(mock_no_tiflash);

    cluster.stop();
    oss.shutdown();
}

#[derive(Default)]
struct DummyStepReporter {}

impl ReportRestoreStepTrait for DummyStepReporter {
    fn report_step(&self, _step: RestoreStep) {}
}

fn generate_backup_name() -> String {
    static BACKUP_ID: AtomicUsize = AtomicUsize::new(0);
    format!("{:04}", BACKUP_ID.fetch_add(1, Ordering::Relaxed))
}

fn stop_node(
    cluster: &mut ServerCluster,
    node_id: u16,
    stores: &HashMap<u16, metapb::Store>,
    state: metapb::StoreState,
) {
    cluster.stop_node(node_id);
    let store_id = set_store_state(cluster, node_id, stores, state);
    info!("node stopped"; "node" => node_id, "store" => store_id, "state" => ?state);
}

fn start_node(
    cluster: &mut ServerCluster,
    node_id: u16,
    stores: &HashMap<u16, metapb::Store>,
    state: metapb::StoreState,
) {
    let store_id = set_store_state(cluster, node_id, stores, state);
    cluster.start_node(node_id, |_, _| {});
    info!("node started"; "node" => node_id, "store" => store_id, "state" => ?state);
}

fn set_store_state(
    cluster: &ServerCluster,
    node_id: u16,
    stores: &HashMap<u16, metapb::Store>,
    state: metapb::StoreState,
) -> u64 /* store_id */ {
    let mut store = stores.get(&node_id).unwrap().clone();
    let store_id = store.id;
    store.state = state;
    cluster.get_pd_client().put_store(store).unwrap();
    store_id
}
