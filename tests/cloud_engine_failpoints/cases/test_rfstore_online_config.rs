// Copyright 2025 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
    time::Duration,
};

use concurrency_manager::ConcurrencyManager;
use kvengine::dfs::Dfs;
use kvproto::metapb;
use online_config::{ConfigManager, ConfigValue};
use raftstore::coprocessor::CoprocessorHost;
use rfstore::store::{config_manager::RfstoreConfigManager, RaftBatchSystem, StoreMeta};
use security::SecurityManager;
use sst_importer::SstImporter;
use test_cloud_server::new_test_config;
use test_pd_client::TestPdClient;
use test_rfstore::{
    create_test_engine, new_node_cluster, ChannelTransport, SimulateTransport, TempDirFs,
};
use tikv_util::{
    config::{ReadableDuration, VersionTrack},
    worker::LazyWorker,
};

#[test]
fn test_rfstore_online_config() {
    let dir = test_util::temp_dir("test_rfstore_cfg", true);
    let test_cfg = new_test_config(dir.path(), 1, 1, 1.0);
    let pd_client = Arc::new(TestPdClient::new(1, false));
    let dfs = Arc::new(TempDirFs::default());
    let io_rate_limiter = Arc::new(
        test_cfg
            .storage
            .io_rate_limit
            .build(true /* enable_statistics */),
    );
    let master_key = dfs
        .get_runtime()
        .block_on(test_cfg.security.new_master_key());
    let security_manager = Arc::new(SecurityManager::new(&test_cfg.security).unwrap());
    let cfg = test_raftstore::Config::new(test_cfg, true);
    let engines = create_test_engine(
        &cfg,
        dir.path(),
        pd_client.clone(),
        dfs.clone(),
        io_rate_limiter,
        master_key,
        security_manager,
    );

    // Initialize raftstore channels.
    let conf = Arc::new(VersionTrack::new(cfg.raft_store.clone()));
    let mut system = RaftBatchSystem::new(&engines, conf);
    let store_meta = Arc::new(Mutex::new(StoreMeta::new(2)));

    let mut store = metapb::Store::default();
    store.set_id(1);
    let trans = ChannelTransport::default();
    let simulate_trans = SimulateTransport::new(trans);
    let pd_worker = LazyWorker::new("test-pd-worker");
    let coprocessor_host = CoprocessorHost::default();
    let cm = ConcurrencyManager::new(1.into());
    let importer = {
        let dir = dir.path().join("import-sst");
        Arc::new(SstImporter::new(&cfg.import, dir, None, cfg.storage.api_version()).unwrap())
    };

    system
        .spawn(
            store,
            engines,
            Box::new(simulate_trans),
            pd_client,
            pd_worker,
            store_meta,
            coprocessor_host,
            importer,
            cm,
        )
        .unwrap();

    let mut rfstore_cfg_mgr = system.get_rfstore_config_manager();

    let (raft_sender, raft_receiver) = tikv_util::mpsc::unbounded();
    fail::cfg_callback("rfstore_raft_worker_update_cfg", move || {
        raft_sender.send(()).unwrap();
    })
    .unwrap();

    let (io_sender, io_receiver) = tikv_util::mpsc::unbounded();
    fail::cfg_callback("rfstore_io_worker_update_cfg", move || {
        io_sender.send(()).unwrap();
    })
    .unwrap();

    // normal change, should only trigger raft worker config update.
    // TODO: also check aux worker.
    let mut change = HashMap::new();
    change.insert("raft_entry_max_size".to_string(), ConfigValue::Size(128));
    rfstore_cfg_mgr.dispatch(change).unwrap();
    raft_receiver
        .recv_timeout(Duration::from_millis(100))
        .unwrap();
    io_receiver
        .recv_timeout(Duration::from_millis(100))
        .unwrap_err();

    for cfg in [
        ("raft_worker_max_batch_size", ConfigValue::Size(10240)),
        ("io_worker_min_write_duration", ConfigValue::Duration(100)),
    ] {
        // update `raft_worker_max_batch_size` or `io_worker_min_write_duration`
        // should trigger both update even if it's not useful in raft worker.
        update_config(&mut rfstore_cfg_mgr, cfg.0, cfg.1);
        raft_receiver
            .recv_timeout(Duration::from_millis(100))
            .unwrap();
        io_receiver
            .recv_timeout(Duration::from_millis(100))
            .unwrap();
    }

    // test change apply pool size
    let apply_pool = rfstore_cfg_mgr.apply_pool();
    for size in [1, 2, 1, 2] {
        // NOTE: the origin input `apply_pool_size` is converted to
        // `apply_batch_system.pool_size` in function `serde_to_online_config`
        let apply_batch_change = [("pool_size".to_string(), ConfigValue::Usize(size))].into();
        update_config(
            &mut rfstore_cfg_mgr,
            "apply_batch_system",
            ConfigValue::Module(apply_batch_change),
        );
        assert_eq!(apply_pool.get_pool_size(), size);
    }

    // TODO: test more configs here.
}

fn test_change_ticker_interval(
    rfstore_cfg_mgr: &mut RfstoreConfigManager,
    fp: &str,
    cfg_name: &str,
    default_dur: Duration,
) {
    let (notify_sender, notify_receiver) = tikv_util::mpsc::bounded(1);
    fail::cfg_callback(fp, move || {
        // only cache 1 message.
        let _ = notify_sender.try_send(());
    })
    .unwrap();
    // empty the channel.
    _ = notify_receiver.try_recv();
    notify_receiver.recv_timeout(default_dur * 2).unwrap();

    // change the heartbeat interval to a very large value, should not receive
    // messge again.
    update_config(rfstore_cfg_mgr, cfg_name, ConfigValue::Duration(100000));
    // wait 1 more tick interval to ensure the tick interal is updated.
    _ = notify_receiver.recv_timeout(default_dur * 2);
    // wait a longer time but should still receive no notification.
    notify_receiver.recv_timeout(default_dur * 5).unwrap_err();
    // reset the config to a short duration.
    update_config(
        rfstore_cfg_mgr,
        cfg_name,
        ConfigValue::Duration(default_dur.as_millis() as u64),
    );
    // the notification should also reset.
    notify_receiver.recv_timeout(default_dur * 2).unwrap();
    fail::remove(fp);
}

fn update_config(cfg_mgr: &mut RfstoreConfigManager, key: &str, val: ConfigValue) {
    let changes = [(key.to_string(), val)].into();
    cfg_mgr.dispatch(changes).unwrap();
}

#[test]
fn test_rfstore_change_tick_interval() {
    let mut cluster = new_node_cluster(0, 1);
    cluster.cfg.raft_store.peer_long_check_interval = ReadableDuration::millis(200);
    cluster.cfg.raft_store.update_gc_safe_point_interval = ReadableDuration::millis(200);
    cluster.cfg.raft_store.local_file_gc_tick_interval = ReadableDuration::millis(200);
    cluster.cfg.raft_store.local_file_gc_timeout = ReadableDuration::millis(500);
    cluster.run();

    let rfstore_cfg_mgr = cluster.mut_rfstore_config_manager(1).unwrap();
    // test change raft ticker interval.
    test_change_ticker_interval(
        rfstore_cfg_mgr,
        "on_split_region_check_tick",
        "split_region_check_tick_interval",
        Duration::from_millis(100),
    );
    test_change_ticker_interval(
        rfstore_cfg_mgr,
        "on_pd_heartbeat_tick",
        "pd_heartbeat_tick_interval",
        Duration::from_millis(100),
    );
    test_change_ticker_interval(
        rfstore_cfg_mgr,
        "on_raft_log_gc_tick",
        "raft_log_gc_tick_interval",
        Duration::from_millis(100),
    );
    test_change_ticker_interval(
        rfstore_cfg_mgr,
        "on_check_long_tick",
        "peer_long_check_interval",
        Duration::from_millis(200),
    );

    // test change store ticker interval.
    test_change_ticker_interval(
        rfstore_cfg_mgr,
        "on_store_pd_heartbeat_tick",
        "pd_store_heartbeat_tick_interval",
        Duration::from_millis(100),
    );
    test_change_ticker_interval(
        rfstore_cfg_mgr,
        "on_update_gc_safe_point",
        "update_gc_safe_point_interval",
        Duration::from_millis(200),
    );
    test_change_ticker_interval(
        rfstore_cfg_mgr,
        "on_local_file_gc",
        "local_file_gc_tick_interval",
        Duration::from_millis(200),
    );
}
