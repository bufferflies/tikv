// Copyright 2023 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    time::Duration,
};

use kvengine::dfs::{DFSConfig, S3Fs};
use kvproto::metapb;
use native_br::{
    backup,
    common::now,
    restore_keyspace,
    restore_keyspace::{ReportRestoreStepTrait, RestoreStep},
    step,
};
use pd_client::PdClient;
use rand::Rng;
use test_cloud_server::{
    client::{RequestOptions, RequestPeerRole},
    oss::ObjectStorageService,
    try_wait, ServerCluster,
};
use tikv::config::TikvConfig;
use tikv_util::{config::ReadableSize, debug, info, store::new_learner_peer, warn};
use tokio::runtime::Runtime;

use crate::alloc_node_id_vec;

const BASIC_DATA_COUNT: usize = 10;
const RANDOM_VALUE_LEN: usize = 128;
const NODES_COUNT: usize = 4;
const KEYSPACE_COUNT: usize = 3;
const DEFAULT_LOOP_COUNT: usize = 5;

// The numbers have no special meaning, just to make them different.
// Require to be larger than 136, see `i_to_val`.
const BASIC_DATA_LEN: usize = 140;
const IMPORT_DATA_LEN: usize = 142;
const PRE_BACKUP_DATA_LEN: usize = 144;
const POST_BACKUP_DATA_LEN: usize = 146;
const PRE_PITR_DATA_LEN: usize = 148;
const FINAL_DATA_LEN: usize = 150;

#[test]
fn test_restore_keyspace() {
    test_util::init_log_for_test();
    let loop_count = std::env::var("LOOP")
        .unwrap_or_default()
        .parse::<usize>()
        .unwrap_or(DEFAULT_LOOP_COUNT);

    test_restore_keyspace_opt(loop_count, true);
    // Regression test for `inner_key_off` disabled.
    test_restore_keyspace_opt(DEFAULT_LOOP_COUNT, false);
}

fn test_restore_keyspace_opt(loop_count: usize, enable_inner_key_off: bool) {
    let cases = vec![
        // keyspace_id, data_count, shuffle_regions, has_learner, loop_count
        (1, 1, None, false, 1),
        (1, 100, None, false, 3),
        (1, 200, Some(10), false, loop_count),
        (1, 200, None, true, loop_count), // Don't shuffle regions for stability.
        (2, 1, None, false, 1),
    ];

    let base_dir = tempfile::Builder::new()
        .prefix("test_restore_keyspace_")
        .tempdir()
        .unwrap();

    let oss_dir = base_dir.path().join("oss");
    let mut oss = ObjectStorageService::new(oss_dir);
    oss.start_server();

    let dfs_config = DFSConfig {
        prefix: "pfx".to_string(),
        s3_endpoint: format!("http://127.0.0.1:{}", oss.port()),
        s3_key_id: "admin".to_string(),
        s3_secret_key: "admin".to_string(),
        s3_bucket: "bkt".to_string(),
        s3_region: "local".to_string(),
        zstd_compression_level: "3".to_string(),
        ..Default::default()
    };

    let mut cluster = ServerCluster::new(
        alloc_node_id_vec(NODES_COUNT),
        |_, conf: &mut TikvConfig| {
            conf.dfs = dfs_config.clone();
            // Set small mem-table size to make data reach L1 and generate over bound
            // shards.
            conf.rocksdb.writecf.write_buffer_size = ReadableSize::kb(1);
            conf.coprocessor.region_split_size = ReadableSize::kb(128); // kv_opts.base_size = 8kb
            conf.coprocessor.region_bucket_size = ReadableSize::kb(64);
            conf.rfengine.target_file_size = ReadableSize::mb(1);
            conf.enable_inner_key_offset = enable_inner_key_off;
        },
    );
    cluster.wait_region_replicated(&[], 3);
    let pd_client = cluster.get_pd_client();
    let mut client = cluster.new_client();

    // Split keyspaces.
    for keyspace_id in 0..=KEYSPACE_COUNT {
        client.split(&get_keyspace_prefix(keyspace_id as u32));
    }
    cluster.wait_pd_region_count(KEYSPACE_COUNT + 2);

    // Import basic data.
    for keyspace_id in 0..KEYSPACE_COUNT {
        client.put_kv(
            0..BASIC_DATA_COUNT,
            gen_keyspace_key(keyspace_id as u32),
            i_to_val(BASIC_DATA_LEN),
        );
    }

    let runtime = Runtime::new().unwrap();

    // Run cases.
    for (case_idx, &(keyspace_id, data_count, shuffle_regions, has_learner, loop_count)) in
        cases.iter().enumerate()
    {
        step!("case group: {case_idx}");

        if has_learner {
            pd_client.disable_default_operator(); // To prevent PD removing learners due to exceed `max-replicas`.
            let (start, end) = (
                get_keyspace_prefix(keyspace_id),
                get_keyspace_prefix(keyspace_id + 1),
            );
            add_learners(&mut cluster, &start, &end);
            check_learners(&cluster, &start, &end, NODES_COUNT - 3, 10);
            step!("add learner done");
        } else {
            pd_client.enable_default_operator(); // Remove learners by peer count check.
        }

        for i in 0..loop_count {
            test_restore_keyspace_impl(
                &mut cluster,
                &dfs_config,
                &format!("{case_idx}:{i}"),
                keyspace_id,
                data_count,
                shuffle_regions,
                has_learner,
                &runtime,
            );
        }
    }

    cluster.stop();
    // Don't graceful shutdown oss (`oss.shutdown()`), as some S3FS threads are
    // still alive and holding connections.
}

fn test_restore_keyspace_impl(
    cluster: &mut ServerCluster,
    dfs_config: &DFSConfig,
    case_name: &str,
    keyspace_id: u32,
    data_count: usize,
    shuffle_regions: Option<usize>,
    has_learner: bool,
    runtime: &Runtime,
) {
    step!("case: {case_name}");
    let mut rng = rand::thread_rng();
    let mut client = cluster.new_client();
    let i_to_key = gen_keyspace_key(keyspace_id);
    let (keyspace_start, keyspace_end) = (
        get_keyspace_prefix(keyspace_id),
        get_keyspace_prefix(keyspace_id + 1),
    );

    let backup_config = backup::BackupConfig {
        dfs: dfs_config.clone(),
        skip_keyspace_meta: true,
        ..Default::default()
    };
    let s3fs = Arc::new(S3Fs::new(
        dfs_config.prefix.clone(),
        dfs_config.s3_endpoint.clone(),
        dfs_config.s3_key_id.clone(),
        dfs_config.s3_secret_key.clone(),
        dfs_config.s3_region.clone(),
        dfs_config.s3_bucket.clone(),
    ));
    let reporter = Arc::new(DummyStepReporter::default());

    // Import data.
    client.put_kv(0..data_count, &i_to_key, i_to_val(IMPORT_DATA_LEN));
    let origin_ref_store = client.dump_ref_store();

    // Prepare for PiTR.
    let truncate_ts = if rng.gen_ratio(1, 3) {
        let ts = client.get_ts();
        client.put_kv(0..data_count, &i_to_key, i_to_val(PRE_PITR_DATA_LEN));
        step!("another writes for pitr done");
        client.verify_data_with_ref_store();
        Some(ts.into_inner())
    } else {
        None
    };

    // Perform snapshot backup.
    let snapshot_backup_name = generate_backup_name();
    let backup_meta = {
        let backup_ts = client.get_ts().into_inner();
        // Put more data before backup to verify truncate ts take affect.
        client.put_kv(0..data_count, &i_to_key, i_to_val(PRE_BACKUP_DATA_LEN));
        step!("another writes done");
        client.verify_data_with_ref_store();
        assert!(
            client
                .verify_data_with_given_ref_store(
                    &origin_ref_store,
                    None,
                    &RequestOptions::default()
                )
                .is_err(),
            "case: {}",
            case_name,
        );
        step!("verify before backup ok");
        let (_, backup_meta) = backup::backup_cluster_with_ts(
            backup_config.clone(),
            false,
            snapshot_backup_name.clone(),
            cluster.get_pd_client().as_ref(),
            backup_ts,
            None,
        )
        .expect("backup::backup_cluster");
        info!("backup_cluster result: {:?}", backup_meta);
        client.verify_data_with_ref_store();
        step!("backup done");

        backup_meta
    };

    // Shuffle regions.
    if let Some(shuffle_regions) = shuffle_regions {
        let keyspace_prefix = get_keyspace_prefix(keyspace_id);
        let region_count = client.pd_client.get_regions_number();
        let mut i = 0;
        while i < shuffle_regions {
            let split_key = rng.gen::<usize>() % data_count;
            if let Err(e) = client.try_split(&i_to_key(split_key)) {
                warn!("try split error: {:?}", e);
                continue;
            }
            i += 1;
        }

        cluster.wait_pd_region_min_count(region_count + shuffle_regions);
        let split_region_count = client.pd_client.get_regions_number();

        for _ in 0..shuffle_regions {
            let source_key = rng.gen::<usize>() % data_count;
            if let Err(e) = client.try_merge_adjacent_region(
                &i_to_key(source_key),
                Some(&keyspace_prefix),
                Duration::from_secs(3),
            ) {
                warn!("try_merge_adjacent_region fail: {:?}", e);
            }
        }

        let merge_region_count = client.pd_client.get_regions_number();
        step!(
            "shuffle regions done, regions {} -> {} -> {}",
            region_count,
            split_region_count,
            merge_region_count
        );
    }

    // Put another data.
    client.put_kv(0..data_count, &i_to_key, i_to_val(POST_BACKUP_DATA_LEN));
    step!("another writes done");
    client.verify_data_with_ref_store();
    assert!(
        client
            .verify_data_with_given_ref_store(&origin_ref_store, None, &RequestOptions::default())
            .is_err(),
        "case: {}",
        case_name,
    );
    step!("verify ok");

    // Test for "PiTR on restored data". For the following scenario:
    // T0: put data0.
    // T1: perform backup.
    // T2: put data1.
    // T3: restore to backup. Now data0 should be in store.
    // T4: PiTR to T2. Now data1 should be in store.
    let truncate_ts_pitr = client.get_ts().into_inner();
    let ref_store_pitr = client.dump_ref_store();

    // Instant backup before restore
    let instant_backup_name = {
        let backup_ts = client.get_ts().into_inner();
        let instant_backup_name = generate_backup_name();
        let (_, backup_meta) = backup::backup_cluster_with_ts(
            backup_config,
            true,
            instant_backup_name.clone(),
            cluster.get_pd_client().as_ref(),
            backup_ts,
            Some(backup_meta),
        )
        .expect("backup::backup_cluster");
        info!("backup_cluster result (instant): {:?}", backup_meta);
        step!("backup for pitr done");
        instant_backup_name
    };

    // Restore keyspace.
    restore_keyspace::restore_keyspace(
        keyspace_id,
        keyspace_id,
        &snapshot_backup_name,
        None,
        s3fs.clone(),
        cluster.get_pd_client(),
        runtime,
        truncate_ts,
        reporter.clone(),
    )
    .unwrap();
    step!("restore done");

    // Verify restored data.
    {
        let mut verify_options: Vec<(Option<(&[u8], &[u8])>, RequestOptions)> =
            vec![(None, RequestOptions::default())];
        if has_learner {
            verify_options.push((
                Some((&keyspace_start, &keyspace_end)),
                RequestOptions {
                    peer_role: RequestPeerRole::Learner,
                },
            ));
        }
        let mut verify_res = vec![];
        for (range, opt) in verify_options {
            verify_res.push(
                client
                    .verify_data_with_given_ref_store(&origin_ref_store, range, &opt)
                    .unwrap(),
            );
        }
        step!(
            "verify restore data done{}, kv count: {:?}",
            if has_learner { " (with learner)" } else { "" },
            verify_res,
        );
    }

    // PiTR on restored data.
    {
        restore_keyspace::restore_keyspace(
            keyspace_id,
            keyspace_id,
            &instant_backup_name,
            None,
            s3fs,
            cluster.get_pd_client(),
            runtime,
            Some(truncate_ts_pitr),
            reporter,
        )
        .unwrap();
        step!("restore (pitr on restored data) done");

        let verify_res = client
            .verify_data_with_given_ref_store(
                &ref_store_pitr,
                Some((&keyspace_start, &keyspace_end)),
                &RequestOptions::default(),
            )
            .unwrap();
        step!(
            "verify (pitr on restored data) done, kv count: {}",
            verify_res
        );
    }

    // Write more data to verify the sequences.
    for id in 0..KEYSPACE_COUNT {
        let count = if id as u32 == keyspace_id {
            data_count
        } else {
            BASIC_DATA_COUNT
        };
        client.put_kv(
            0..count,
            gen_keyspace_key(id as u32),
            i_to_val(FINAL_DATA_LEN),
        );
    }
    client.verify_data_with_ref_store();
    step!("verify more writes done");
}

fn i_to_key(i: usize) -> Vec<u8> {
    format!("xkey_{:08}", i).into_bytes()
}

fn random_val() -> Vec<u8> {
    let mut bytes = [0u8; RANDOM_VALUE_LEN];
    rand::thread_rng().fill(&mut bytes);
    bytes.to_vec()
}

// Generate `i_to_val` by specifying expected length of value, to make it easier
// to know where the data is written.
// `expected_len` must be > 136
// `expected_len` = 128(random_val) + i(8) + padding
fn i_to_val(expected_len: usize) -> impl Fn(usize) -> Vec<u8> {
    assert!(
        expected_len > 8 + RANDOM_VALUE_LEN,
        "expected_len: {} is not larger than 136",
        expected_len
    );
    move |i: usize| -> Vec<u8> {
        let mut val = format!("{:08}", i).into_bytes();
        let mut padding = vec![b'0'; expected_len - 8 - RANDOM_VALUE_LEN];
        val.append(&mut padding);
        val.append(&mut random_val());
        val
    }
}

fn get_keyspace_prefix(keyspace_id: u32) -> Vec<u8> {
    let mut prefix = keyspace_id.to_be_bytes();
    prefix[0] = b'x';
    prefix.to_vec()
}

fn gen_keyspace_key(keyspace_id: u32) -> impl Fn(usize) -> Vec<u8> {
    move |i: usize| -> Vec<u8> {
        let mut key = get_keyspace_prefix(keyspace_id);
        key.extend(i_to_key(i));
        key
    }
}

// Add learners to all stores with no peer of keyspace regions.
// TODO: add to `test_cloud_server` for reuse.
fn add_learners(cluster: &mut ServerCluster, start: &[u8], end: &[u8]) {
    let pd_client = cluster.get_pd_client();
    let mut client = cluster.new_client();
    let stores = cluster.get_stores();

    client.clear_region_cache();

    let mut next_key = start.to_owned();
    while next_key.as_slice() < end {
        let region = client.get_region_by_key(&next_key);
        next_key = region.raw_end().to_owned();

        let learner_stores = stores
            .iter()
            .filter(|&&store_id| region.peers().iter().all(|peer| peer.store_id != store_id))
            .collect::<Vec<_>>();
        for &store_id in learner_stores {
            let learner_peer = new_learner_peer(store_id, pd_client.alloc_id().unwrap());
            pd_client.must_add_peer(region.id(), learner_peer);
        }
    }
}

fn check_learners_impl(
    cluster: &ServerCluster,
    start: &[u8],
    end: &[u8],
    min_count: usize,
) -> Vec<test_cloud_server::client::RawRegion> {
    let mut client = cluster.new_client();

    let mut res = vec![];
    let mut next_key = start.to_owned();
    while next_key.as_slice() < end {
        let region = client.get_region_by_key(&next_key);
        next_key = region.raw_end().to_owned();

        let learner_cnt = region
            .peers()
            .iter()
            .filter(|peer| peer.get_role() == metapb::PeerRole::Learner)
            .count();
        debug!(
            "check region for learner: {:?}, count: {}",
            region, learner_cnt
        );
        if learner_cnt < min_count {
            res.push(region);
        }
    }
    res
}

fn check_learners(
    cluster: &ServerCluster,
    start: &[u8],
    end: &[u8],
    min_count: usize,
    timeout_seconds: usize,
) {
    let ok = try_wait(
        || check_learners_impl(cluster, start, end, min_count).is_empty(),
        timeout_seconds,
    );
    assert!(
        ok,
        "{:?}",
        check_learners_impl(cluster, start, end, min_count)
    );
}

fn generate_backup_name() -> String {
    static BACKUP_ID: AtomicUsize = AtomicUsize::new(0);
    format!("{:04}", BACKUP_ID.fetch_add(1, Ordering::Relaxed))
}

#[derive(Default)]
struct DummyStepReporter {}

impl ReportRestoreStepTrait for DummyStepReporter {
    fn report_step(&self, _step: RestoreStep) {}
}
