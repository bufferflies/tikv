// Copyright 2023 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    time::Duration,
};

use kvengine::{
    dfs::{DFSConfig, Dfs, S3Fs},
    ShardStats, WRITE_CF,
};
use kvproto::metapb;
use native_br::{
    archive, backup,
    common::now,
    restore::RestoreConfig,
    restore_keyspace,
    restore_keyspace::{ReportRestoreStepTrait, RestoreStep},
    step,
};
use pd_client::PdClient;
use rand::{seq::SliceRandom, Rng};
use test_cloud_server::{
    client::{CommitAction, RequestOptions, RequestPeerRole},
    oss::prepare_dfs,
    try_wait, ServerCluster,
};
use test_pd_client::TestPdClient;
use tikv::config::TikvConfig;
use tikv_util::{
    codec::bytes::encode_bytes,
    config::{ReadableDuration, ReadableSize},
    debug, info,
    store::new_learner_peer,
    warn,
};
use tokio::runtime::Runtime;
use txn_types::TimeStamp;

use crate::{alloc_node_id_vec, request_major_compact_on_store};

const BASIC_DATA_COUNT: usize = 10;
const RANDOM_VALUE_LEN: usize = 64;
const NODES_COUNT: usize = 4;
const KEYSPACE_COUNT: usize = 3;
const DEFAULT_LOOP_COUNT: usize = 3;
const DEFAULT_TARGET_REGIONS: usize = 4;
const BACKUP_DAYS: usize = 4;

// The numbers have no special meaning, just to make them different.
// Require to be larger than 72, see `i_to_val`.
const BASIC_DATA_LEN: usize = 80;
const IMPORT_DATA_LEN: usize = 82;
const PRE_BACKUP_DATA_LEN: usize = 84;
const POST_BACKUP_DATA_LEN: usize = 86;
const PRE_PITR_DATA_LEN: usize = 88;
const FINAL_DATA_LEN: usize = 90;

#[test]
fn test_restore_keyspace() {
    test_util::init_log_for_test();
    let loop_count = std::env::var("LOOP")
        .unwrap_or_default()
        .parse::<usize>()
        .unwrap_or(DEFAULT_LOOP_COUNT);
    let target_regions = std::env::var("TARGET_REGIONS")
        .unwrap_or_default()
        .parse::<usize>()
        .unwrap_or(DEFAULT_TARGET_REGIONS);

    test_restore_keyspace_opt(loop_count, target_regions, true, true, false);
    // Regression test for `inner_key_off` disabled.
    test_restore_keyspace_opt(DEFAULT_LOOP_COUNT, target_regions, false, false, false);
    // Regression test for `lightweight` disabled.
    test_restore_keyspace_opt(DEFAULT_LOOP_COUNT, target_regions, true, false, false);
    // Test archiving without `inner_key_off`and lightweight backup.
    test_restore_keyspace_opt(DEFAULT_LOOP_COUNT, target_regions, false, false, true);
    // Test archiving with `inner_key_off` and lightweight backup.
    test_restore_keyspace_opt(DEFAULT_LOOP_COUNT, target_regions, true, true, true);
}

fn test_restore_keyspace_opt(
    loop_count: usize,
    target_regions: usize,
    enable_inner_key_off: bool,
    lightweight: bool,
    archiving: bool,
) {
    let cases = vec![
        // keyspace_id, data_count, shuffle_regions, has_learner, loop_count
        (1, 1, None, false, 1),
        (1, 100, Some(target_regions), false, loop_count),
        (1, 100, None, true, 1), // Don't shuffle regions for stability.
        (2, 1, None, false, 1),
    ];

    let (_temp_dir, _oss, dfs_config) = prepare_dfs("test_restore_keyspace_");
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
            conf.rfengine.lightweight_backup = lightweight;
            conf.rfengine.wal_chunk_target_file_size = ReadableSize::kb(128);
            conf.enable_inner_key_offset = enable_inner_key_off;
        },
    );
    cluster.wait_region_replicated(&[], 3);
    let pd_client = cluster.get_pd_client();
    let mut client = cluster.new_client();

    if archiving {
        pd_client.set_tso(TimeStamp::compose(
            chrono::Utc::now().timestamp_millis() as u64,
            0,
        ));
    }

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

        for loop_idx in 0..loop_count {
            if archiving {
                test_restore_archived_keyspace_impl(
                    case_idx,
                    loop_idx,
                    &mut cluster,
                    &dfs_config,
                    keyspace_id,
                    data_count,
                    shuffle_regions,
                    has_learner,
                    lightweight,
                    &runtime,
                );
                continue;
            }
            test_restore_keyspace_impl(
                case_idx,
                loop_idx,
                &mut cluster,
                &dfs_config,
                keyspace_id,
                data_count,
                shuffle_regions,
                has_learner,
                lightweight,
                &runtime,
            );
        }
    }

    cluster.stop();
    // Don't graceful shutdown oss (`oss.shutdown()`), as some S3FS threads are
    // still alive and holding connections.
}

fn test_restore_keyspace_impl(
    case_idx: usize,
    loop_idx: usize,
    cluster: &mut ServerCluster,
    dfs_config: &DFSConfig,
    keyspace_id: u32,
    data_count: usize,
    shuffle_regions: Option<usize>,
    has_learner: bool,
    lightweight: bool,
    runtime: &Runtime,
) {
    step!("case: {}:{}", case_idx, loop_idx);
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
    let commit_action = if rng.gen_ratio(2, 3) {
        CommitAction::AsyncCommitSecondaryKeys(Duration::ZERO)
    } else {
        CommitAction::AsyncCommitSecondaryKeys(Duration::MAX)
    };
    client
        .try_put_kv(
            0..data_count,
            &i_to_key,
            i_to_val(IMPORT_DATA_LEN),
            commit_action,
        )
        .unwrap();
    let origin_ref_store = client.dump_ref_store();

    // Prepare for PiTR.
    let truncate_ts = if rng.gen_ratio(1, 2) {
        let ts = client.get_ts();
        client.put_kv(
            0..(data_count / 2).max(1),
            &i_to_key,
            i_to_val(PRE_PITR_DATA_LEN),
        );
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
        client.put_kv(
            0..(data_count / 2).max(1),
            &i_to_key,
            i_to_val(PRE_BACKUP_DATA_LEN),
        );
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
            "case: {}:{}",
            case_idx,
            loop_idx
        );
        step!("verify before backup ok");
        let backup_type = if lightweight {
            backup::BackupType::Lightweight
        } else {
            backup::BackupType::Full
        };
        let (_, backup_meta) = backup::backup_cluster_with_ts(
            backup_config.clone(),
            backup_type,
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
    if let (Some(shuffle_regions), 0) = (shuffle_regions, loop_idx) {
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

        for _ in 0..(shuffle_regions / 2) {
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
        "case: {}:{}",
        case_idx,
        loop_idx,
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
        let backup_type = if lightweight {
            backup::BackupType::Lightweight
        } else {
            backup::BackupType::Incremental
        };
        let (_, backup_meta) = backup::backup_cluster_with_ts(
            backup_config,
            backup_type,
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
        RestoreConfig::default(),
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
            RestoreConfig::default(),
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

fn test_restore_archived_keyspace_impl(
    case_idx: usize,
    loop_idx: usize,
    cluster: &mut ServerCluster,
    dfs_config: &DFSConfig,
    keyspace_id: u32,
    data_count: usize,
    shuffle_regions: Option<usize>,
    has_learner: bool,
    lightweight: bool,
    runtime: &Runtime,
) {
    step!("case: {}:{}", case_idx, loop_idx);
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

    let pd_client = cluster.get_pd_client();
    let mut archive_config = archive::ArchiveConfig {
        dfs: dfs_config.clone(),
        max_archive_file_size: 1024 * 32,
        dry_run: false,
        ..Default::default()
    };
    archive_config.check_data_dir();

    // Import, backup and archive every day.
    let begin_archive_date =
        chrono::NaiveDateTime::from_timestamp_millis(client.get_ts().physical() as i64).unwrap();
    let mut date_time = begin_archive_date;
    let mut backup_ts_list = Vec::with_capacity(BACKUP_DAYS);
    let mut ref_store_list = Vec::with_capacity(BACKUP_DAYS);
    let mut last_ref_store = client.dump_ref_store();
    for idx in 0..BACKUP_DAYS {
        // Import data.
        client.put_kv(0..data_count, &i_to_key, i_to_val(IMPORT_DATA_LEN + idx));
        step!("write done on {}", date_time.date());

        client.verify_data_with_ref_store();
        assert!(
            client
                .verify_data_with_given_ref_store(&last_ref_store, None, &RequestOptions::default())
                .is_err(),
            "case: {}:{}:{}",
            case_idx,
            loop_idx,
            date_time.date(),
        );
        last_ref_store = client.dump_ref_store();
        step!("verify ok on {}", date_time.date());

        // Perform snapshot backup.
        let backup_ts = client.get_ts().into_inner();
        backup_ts_list.push(backup_ts);
        ref_store_list.push(client.dump_ref_store());
        let backup_type = if lightweight {
            backup::BackupType::Lightweight
        } else {
            backup::BackupType::Full
        };
        let (_, backup_meta) = backup::backup_cluster_with_ts(
            backup_config.clone(),
            backup_type,
            String::default(),
            cluster.get_pd_client().as_ref(),
            backup_ts,
            None,
        )
        .expect("backup::backup_cluster");
        info!("backup_cluster result: {:?}", backup_meta);

        // Archive backup.
        archive::archive_cluster_backup(
            archive_config.clone(),
            cluster.get_pd_client(),
            s3fs.clone(),
            begin_archive_date.date(),
            date_time.date(),
        )
        .unwrap();
        step!("archive done on {}", date_time.date());

        date_time = date_time.checked_add_days(chrono::Days::new(1)).unwrap();
        pd_client.set_tso(TimeStamp::compose(date_time.timestamp_millis() as u64, 0));
    }

    // Delete old backup.
    {
        let snapshot_backup_ts = backup_ts_list[0];
        let snapshot_backup_date = chrono::NaiveDateTime::from_timestamp_millis(
            TimeStamp::from(snapshot_backup_ts).physical() as i64,
        )
        .unwrap()
        .date();
        let archive_reader =
            archive::ArchiveReader::new(s3fs.clone(), &snapshot_backup_date).unwrap();
        let old_file_ids = archive_reader.get_file_ids();
        s3fs.get_runtime().block_on(async {
            for idx in 0..BACKUP_DAYS - 1 {
                let backup_ts = backup_ts_list[idx];
                let backup_key = backup::backup_file_full_path(
                    s3fs.get_prefix(),
                    String::default(),
                    Some(backup_ts),
                );
                s3fs.delete_object(backup_key.clone(), backup_ts.to_string())
                    .await
                    .unwrap();
            }
            for file_id in old_file_ids {
                s3fs.delete_object(s3fs.file_key(file_id), file_id.to_string())
                    .await
                    .unwrap();
            }
        });
        step!(
            "delete old backup meta and sst files done. case: {}:{}",
            case_idx,
            loop_idx,
        );
    }

    for idx in 0..BACKUP_DAYS - 1 {
        // Shuffle regions.
        if let (Some(shuffle_regions), 0) = (shuffle_regions, loop_idx) {
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

            for _ in 0..(shuffle_regions / 2) {
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

        // Restore keyspace.
        {
            let snapshot_backup_ts = backup_ts_list[idx];
            let snapshot_backup_name =
                backup::IncrementalBackupFile::from_backup_ts(snapshot_backup_ts).into_name();
            restore_keyspace::restore_keyspace(
                keyspace_id,
                keyspace_id,
                &snapshot_backup_name,
                None,
                s3fs.clone(),
                RestoreConfig::default(),
                cluster.get_pd_client(),
                runtime,
                None,
                reporter.clone(),
            )
            .unwrap();
            step!("restore done. case: {}:{}:{}", case_idx, loop_idx, idx);
        }

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
                        .verify_data_with_given_ref_store(&ref_store_list[idx], range, &opt)
                        .unwrap(),
                );
            }
            step!(
                "verify restore data done{}, kv count: {:?}, case: {}:{}:{}",
                if has_learner { " (with learner)" } else { "" },
                verify_res,
                case_idx,
                loop_idx,
                idx
            );
        }
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

/// This test is to verify that `restore_keyspace` can resolve locks.
///
/// Resolving locks is necessary when a backup is performed after the primary
/// key is committed but the secondary keys are not in a transaction.
///
/// See https://github.com/tidbcloud/cloud-storage-engine/issues/1134.
#[test]
fn test_restore_keyspace_with_resolve_locks() {
    const KEYSPACE_ID: u32 = 1;

    test_util::init_log_for_test();
    let (_temp_dir, _oss, dfs_config) = prepare_dfs("test_restore_keyspace_");
    let s3fs = Arc::new(S3Fs::new(
        dfs_config.prefix.clone(),
        dfs_config.s3_endpoint.clone(),
        dfs_config.s3_key_id.clone(),
        dfs_config.s3_secret_key.clone(),
        dfs_config.s3_region.clone(),
        dfs_config.s3_bucket.clone(),
    ));
    let reporter = Arc::new(DummyStepReporter::default());
    let runtime = Runtime::new().unwrap();

    let mut cluster = ServerCluster::new(
        alloc_node_id_vec(NODES_COUNT),
        |_, conf: &mut TikvConfig| {
            conf.dfs = dfs_config.clone();
            // Set small mem-table size to make data reach L0+.
            conf.rocksdb.writecf.write_buffer_size = ReadableSize(128);
            conf.coprocessor.region_split_size = ReadableSize::kb(128); // kv_opts.base_size = 8kb
            conf.coprocessor.region_bucket_size = ReadableSize::kb(64);
            conf.rfengine.target_file_size = ReadableSize::mb(1);
            conf.rfengine.lightweight_backup = true;
            conf.rfengine.wal_chunk_target_file_size = ReadableSize::kb(128);
            conf.enable_inner_key_offset = true;
        },
    );
    cluster.wait_region_replicated(&[], 3);
    let pd_client = cluster.get_pd_client();
    let mut client = cluster.new_client();
    client.split(&get_keyspace_prefix(KEYSPACE_ID));
    client.split(&get_keyspace_prefix(KEYSPACE_ID + 1));

    // Import data
    let i_to_key = gen_keyspace_key(KEYSPACE_ID);
    client.put_kv(0..100, &i_to_key, i_to_val(BASIC_DATA_LEN));
    // Delete data without committing secondary keys, to simulate the case that if
    // we don't resolve locks during restoration, the secondary keys will be rolled
    // back unexpectedly after deletion of primary key is compacted.
    for range in vec![0..25, 25..50, 50..100] {
        client
            .try_del_kv(
                range,
                &i_to_key,
                CommitAction::AsyncCommitSecondaryKeys(Duration::MAX),
            )
            .unwrap();
    }
    let origin_ref_store = client.dump_ref_store();

    // Wait for data of WRITE_CF being compacted to L1+, and trigger
    // `kvengine::Engine::load_unloaded_tables`.
    wait_for_keyspace_stats(
        &runtime,
        &cluster,
        &pd_client,
        KEYSPACE_ID,
        |stats| stats.cfs[WRITE_CF].levels.iter().any(|l| l.num_tables > 0),
        Duration::from_secs(10),
    );

    // Perform backup.
    let snapshot_backup_name = generate_backup_name();
    {
        let backup_ts = client.get_ts().into_inner();
        let backup_config = backup::BackupConfig {
            dfs: dfs_config,
            skip_keyspace_meta: true,
            ..Default::default()
        };
        let (_, backup_meta) = backup::backup_cluster_with_ts(
            backup_config,
            backup::BackupType::Lightweight,
            snapshot_backup_name.clone(),
            cluster.get_pd_client().as_ref(),
            backup_ts,
            None,
        )
        .expect("backup::backup_cluster");
        info!("backup_cluster result: {:?}", backup_meta);
    }

    // Restore keyspace.
    restore_keyspace::restore_keyspace(
        KEYSPACE_ID,
        KEYSPACE_ID,
        &snapshot_backup_name,
        None,
        s3fs,
        RestoreConfig::default(),
        cluster.get_pd_client(),
        &runtime,
        None,
        reporter,
    )
    .unwrap();

    // Invoke major compaction to reproduce issue by compacting the deletion of
    // primary key.
    // If restore keyspace do not resolve locks, the secondary keys of "put_kv" will
    // not be compacted, and the following `wait_for_keyspace_stats` will fail.
    let ts = client.get_ts();
    cluster.set_gc_safe_point(ts.into_inner());
    request_major_compaction(&runtime, &pd_client, KEYSPACE_ID);
    wait_for_keyspace_stats(
        &runtime,
        &cluster,
        &pd_client,
        KEYSPACE_ID,
        |stats| stats.cfs[WRITE_CF].levels.iter().all(|l| l.num_tables == 0),
        Duration::from_secs(10),
    );
    std::thread::sleep(Duration::from_secs(10));

    // Verify restored data.
    client
        .verify_data_with_given_ref_store(&origin_ref_store, None, &RequestOptions::default())
        .unwrap();

    cluster.stop();
}

/// This test is to verify that `restore_keyspace` can handle the case that in
/// lightweight backup, the last epoch has no chunk in DFS.
///
/// See https://github.com/tidbcloud/cloud-storage-engine/issues/1319.
#[test]
fn test_restore_keyspace_with_no_chunk() {
    const KEYSPACE_ID: u32 = 1;

    test_util::init_log_for_test();
    let (_temp_dir, _oss, dfs_config) = prepare_dfs("test_restore_keyspace_");
    let s3fs = Arc::new(S3Fs::new(
        dfs_config.prefix.clone(),
        dfs_config.s3_endpoint.clone(),
        dfs_config.s3_key_id.clone(),
        dfs_config.s3_secret_key.clone(),
        dfs_config.s3_region.clone(),
        dfs_config.s3_bucket.clone(),
    ));
    let reporter = Arc::new(DummyStepReporter::default());
    let runtime = Runtime::new().unwrap();

    let mut cluster = ServerCluster::new(
        alloc_node_id_vec(NODES_COUNT),
        |_, conf: &mut TikvConfig| {
            conf.dfs = dfs_config.clone();
            conf.rfengine.lightweight_backup = true;
            conf.enable_inner_key_offset = true;
        },
    );
    cluster.wait_region_replicated(&[], 3);
    let mut client = cluster.new_client();
    client.split(&get_keyspace_prefix(KEYSPACE_ID));
    client.split(&get_keyspace_prefix(KEYSPACE_ID + 1));

    // Import small data
    let i_to_key = gen_keyspace_key(KEYSPACE_ID);
    client.put_kv(0..1, &i_to_key, i_to_val(BASIC_DATA_LEN));
    client.verify_data_with_ref_store();

    // Perform backup.
    let snapshot_backup_name = generate_backup_name();
    {
        let backup_ts = client.get_ts().into_inner();
        let backup_config = backup::BackupConfig {
            dfs: dfs_config,
            skip_keyspace_meta: true,
            ..Default::default()
        };
        let (_, backup_meta) = backup::backup_cluster_with_ts(
            backup_config,
            backup::BackupType::Lightweight,
            snapshot_backup_name.clone(),
            cluster.get_pd_client().as_ref(),
            backup_ts,
            None,
        )
        .expect("backup::backup_cluster");
        info!("backup_cluster result: {:?}", backup_meta);
    }

    // Restore keyspace.
    restore_keyspace::restore_keyspace(
        KEYSPACE_ID,
        KEYSPACE_ID,
        &snapshot_backup_name,
        None,
        s3fs,
        RestoreConfig::default(),
        cluster.get_pd_client(),
        &runtime,
        None,
        reporter,
    )
    .unwrap();

    // Verify restored data.
    client.verify_data_with_ref_store();
    cluster.stop();
}

#[test]
fn test_restore_keyspace_with_failed_store() {
    const KEYSPACE_ID: u32 = 1;

    test_util::init_log_for_test();
    let (_temp_dir, _oss, dfs_config) = prepare_dfs("test_restore_keyspace_");
    let s3fs = Arc::new(S3Fs::new(
        dfs_config.prefix.clone(),
        dfs_config.s3_endpoint.clone(),
        dfs_config.s3_key_id.clone(),
        dfs_config.s3_secret_key.clone(),
        dfs_config.s3_region.clone(),
        dfs_config.s3_bucket.clone(),
    ));
    let reporter = Arc::new(DummyStepReporter::default());
    let runtime = Runtime::new().unwrap();

    let mut cluster = ServerCluster::new(
        alloc_node_id_vec(NODES_COUNT),
        |_, conf: &mut TikvConfig| {
            conf.dfs = dfs_config.clone();
            conf.rfengine.lightweight_backup = true;
            conf.enable_inner_key_offset = true;
        },
    );
    cluster.wait_region_replicated(&[], 3);
    let mut client = cluster.new_client();
    client.split(&get_keyspace_prefix(KEYSPACE_ID));
    client.split(&get_keyspace_prefix(KEYSPACE_ID + 1));

    // Import small data
    let i_to_key = gen_keyspace_key(KEYSPACE_ID);
    client.put_kv(0..1, &i_to_key, i_to_val(BASIC_DATA_LEN));
    client.verify_data_with_ref_store();

    // Perform backup.
    let snapshot_backup_name = generate_backup_name();
    {
        let backup_ts = client.get_ts().into_inner();
        let backup_config = backup::BackupConfig {
            dfs: dfs_config,
            skip_keyspace_meta: true,
            ..Default::default()
        };
        let (_, backup_meta) = backup::backup_cluster_with_ts(
            backup_config,
            backup::BackupType::Lightweight,
            snapshot_backup_name.clone(),
            cluster.get_pd_client().as_ref(),
            backup_ts,
            None,
        )
        .expect("backup::backup_cluster");
        info!("backup_cluster result: {:?}", backup_meta);
    }

    // Force stop one node without flush the last wal chunk to dfs.
    let node_ids = cluster.get_nodes();
    let mut rng = rand::thread_rng();
    let failed_node_id = node_ids.choose(&mut rng).unwrap();
    cluster.stop_node_force(*failed_node_id, true);

    let mut restore_config = RestoreConfig {
        tolerate_err: 0,
        timeout_fetch_wal: ReadableDuration::secs(1), /* Set a small timeout to make the test
                                                       * faster. */
        ..Default::default()
    };

    // Restore keyspace will failed without tolerate error.
    restore_keyspace::restore_keyspace(
        KEYSPACE_ID,
        KEYSPACE_ID,
        &snapshot_backup_name,
        None,
        s3fs.clone(),
        restore_config.clone(),
        cluster.get_pd_client(),
        &runtime,
        None,
        reporter.clone(),
    )
    .unwrap_err();

    // Update tolerate error config and restore keyspace again.
    restore_config.tolerate_err = 1;

    // Restore keyspace.
    restore_keyspace::restore_keyspace(
        KEYSPACE_ID,
        KEYSPACE_ID,
        &snapshot_backup_name,
        None,
        s3fs,
        restore_config,
        cluster.get_pd_client(),
        &runtime,
        None,
        reporter,
    )
    .unwrap();

    // Verify restored data.
    client.verify_data_with_ref_store();
    cluster.stop();
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
// `expected_len` must be > 72
// `expected_len` = 64(random_val) + i(8) + padding
fn i_to_val(expected_len: usize) -> impl Fn(usize) -> Vec<u8> {
    assert!(
        expected_len > 8 + RANDOM_VALUE_LEN,
        "expected_len: {} is not larger than 72",
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

fn request_major_compaction(runtime: &Runtime, pd_client: &TestPdClient, keyspace_id: u32) {
    let stores = pd_client.get_all_stores(true).unwrap();
    let mut handles = Vec::with_capacity(stores.len());
    for store in stores {
        handles.push(runtime.spawn(async move {
            let query = format!("major_compact=true&keyspace_id={}", keyspace_id);
            request_major_compact_on_store(&store, query.as_str(), true).await;
        }));
    }
    for handle in handles {
        runtime.block_on(handle).unwrap();
    }
}

/// Wait for stats of all regions in a keyspace to be expected.
///
/// Note that when any peer has the expected stats, the region will be
/// considered as expected. In the scene of restoration, as we restore from
/// rfengine of leader peer, when any peer become expected, the rfengine meta of
/// leader must be expected.
fn wait_for_keyspace_stats(
    runtime: &Runtime,
    cluster: &ServerCluster,
    pd_client: &TestPdClient,
    keyspace_id: u32,
    expect: impl Fn(&ShardStats) -> bool,
    timeout: Duration,
) {
    let regions = runtime
        .block_on(pd_client.scan_regions(
            encode_bytes(&get_keyspace_prefix(keyspace_id)),
            encode_bytes(&get_keyspace_prefix(keyspace_id + 1)),
            100,
        ))
        .unwrap();
    info!("regions {:?}", regions);
    let node_ids = cluster.get_nodes();
    let get_stats = |region_id: u64| -> Vec<ShardStats> {
        node_ids
            .iter()
            .filter_map(|id| cluster.get_kvengine(*id).get_shard_stat_opt(region_id))
            .collect()
    };
    for region in regions {
        let region_id = region.get_region().id;
        let ok = try_wait(
            || get_stats(region_id).iter().any(|stats| expect(stats)),
            timeout.as_secs() as usize,
        );
        assert!(
            ok,
            "wait_for_keyspace_stats timeout, region_id: {}, stats: {:?}",
            region_id,
            get_stats(region_id)
        );
    }
}

#[derive(Default)]
struct DummyStepReporter {}

impl ReportRestoreStepTrait for DummyStepReporter {
    fn report_step(&self, _step: RestoreStep) {}
}
