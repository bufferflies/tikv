// Copyright 2023 TiKV Project Authors. Licensed under Apache-2.0.

use std::{sync::Arc, time::Duration};

use kvengine::dfs::{DFSConfig, S3Fs};
use kvproto::metapb;
use native_br::{backup, common::now, restore_keyspace, step};
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
const DEFAULT_LOOP_COUNT: usize = 3;

#[test]
fn test_restore_keyspace() {
    test_util::init_log_for_test();
    test_restore_keyspace_opt(false); // TODO: remove when wal sync dir is enabled by default.
    test_restore_keyspace_opt(true);
}

fn test_restore_keyspace_opt(enable_wal_sync_dir: bool) {
    let loop_count = std::env::var("LOOP")
        .unwrap_or_default()
        .parse::<usize>()
        .unwrap_or(DEFAULT_LOOP_COUNT);

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
    let base_dir_str = base_dir.path().to_str().unwrap();

    let oss_dir = base_dir.path().join("oss");
    let mut oss = ObjectStorageService::new(oss_dir);
    oss.start_server();

    let dfs_config = DFSConfig {
        prefix: "test_restore_keyspace_".to_string(),
        s3_endpoint: format!("http://127.0.0.1:{}", oss.port()),
        s3_key_id: "admin".to_string(),
        s3_secret_key: "admin".to_string(),
        s3_bucket: "test_restore_keyspace_".to_string(),
        s3_region: "local".to_string(),
        zstd_compression_level: "3".to_string(),
        ..Default::default()
    };

    let mut cluster = ServerCluster::new(
        alloc_node_id_vec(NODES_COUNT),
        |node_id: u16, conf: &mut TikvConfig| {
            conf.dfs = dfs_config.clone();
            // Set small mem-table size to make data reach L1 and generate over bound
            // shards.
            conf.rocksdb.writecf.write_buffer_size = ReadableSize::kb(1);
            conf.coprocessor.region_split_size = ReadableSize::kb(128); // kv_opts.base_size = 8kb
            conf.coprocessor.region_bucket_size = ReadableSize::kb(64);
            conf.rfengine.target_file_size = ReadableSize::mb(1);
            conf.rfengine.wal_sync_dir = enable_wal_sync_dir.then(|| {
                let dir = format!("{}/wal_sync/{}", base_dir_str, node_id);
                step!("enable wal sync dir: {}", dir);
                dir
            });
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
            i_to_val_138,
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
    let mut client = cluster.new_client();
    let i_to_key = gen_keyspace_key(keyspace_id);
    let (keyspace_start, keyspace_end) = (
        get_keyspace_prefix(keyspace_id),
        get_keyspace_prefix(keyspace_id + 1),
    );

    // Import data.
    client.put_kv(0..data_count, &i_to_key, i_to_val_140);
    let origin_ref_store = client.dump_ref_store();

    // Execute backup.
    let backup_name = format!("restore_keyspace_test_{}", rand::thread_rng().gen::<u64>());
    let backup_config = backup::BackupConfig {
        dfs: dfs_config.clone(),
        skip_keyspace_meta: true,
        ..Default::default()
    };
    let backup_ts = client.get_ts().into_inner();
    // Put more data before backup to verify truncate ts take affect.
    client.put_kv(0..data_count, &i_to_key, i_to_val_142);
    step!("another writes done");
    client.verify_data_with_ref_store();
    assert!(
        client
            .verify_data_with_given_ref_store(&origin_ref_store, None, &RequestOptions::default())
            .is_err(),
        "case: {}",
        case_name,
    );
    step!("verify before backup ok");
    let backup_meta = backup::backup_cluster_with_ts(
        backup_config,
        false,
        backup_name.clone(),
        cluster.get_pd_client().as_ref(),
        backup_ts,
        None,
    )
    .expect("backup::backup_cluster");
    info!("backup_cluster result: {:?}", backup_meta);
    client.verify_data_with_ref_store();
    step!("backup done");

    // Shuffle regions.
    if let Some(shuffle_regions) = shuffle_regions {
        let keyspace_prefix = get_keyspace_prefix(keyspace_id);
        let region_count = client.pd_client.get_regions_number();
        let mut i = 0;
        while i < shuffle_regions {
            let split_key = rand::thread_rng().gen::<usize>() % data_count;
            if let Err(e) = client.try_split(&i_to_key(split_key)) {
                warn!("try split error: {:?}", e);
                continue;
            }
            i += 1;
        }

        cluster.wait_pd_region_min_count(region_count + shuffle_regions);
        let split_region_count = client.pd_client.get_regions_number();

        for _ in 0..shuffle_regions {
            let source_key = rand::thread_rng().gen::<usize>() % data_count;
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
    client.put_kv(0..data_count, &i_to_key, i_to_val_142);
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

    // Restore keyspace.
    let dfs_config = dfs_config.clone();
    let s3fs = Arc::new(S3Fs::new(
        dfs_config.prefix,
        dfs_config.s3_endpoint,
        dfs_config.s3_key_id,
        dfs_config.s3_secret_key,
        dfs_config.s3_region,
        dfs_config.s3_bucket,
    ));
    restore_keyspace::restore_keyspace(
        keyspace_id,
        &backup_name,
        None,
        s3fs,
        cluster.get_pd_client(),
        runtime,
    )
    .unwrap();
    step!("restore done");

    // Verify restored data.
    // Retry as shard restore is asynchronously applied.
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
    let mut verify_res = Err("unknown".into());
    for (range, opt) in verify_options {
        let ok = try_wait(
            || {
                verify_res =
                    client.verify_data_with_given_ref_store(&origin_ref_store, range, &opt);
                verify_res.is_ok()
            },
            10,
        );
        if !ok {
            client
                .verify_data_with_given_ref_store(&origin_ref_store, range, &opt)
                .unwrap();
        }
    }
    step!(
        "verify restore data done{}, kv count: {}",
        if has_learner { " (with learner)" } else { "" },
        verify_res.unwrap(),
    );

    // Write more data to verify the sequences.
    for id in 0..KEYSPACE_COUNT {
        let count = if id as u32 == keyspace_id {
            data_count
        } else {
            BASIC_DATA_COUNT
        };
        client.put_kv(0..count, gen_keyspace_key(id as u32), i_to_val_138);
    }
    client.verify_data_with_ref_store();
    step!("verify more writes done");
}

fn i_to_key(i: usize) -> Vec<u8> {
    format!("key_{:08}", i).into_bytes()
}

fn random_val() -> Vec<u8> {
    let mut bytes = [0u8; RANDOM_VALUE_LEN];
    rand::thread_rng().fill(&mut bytes);
    bytes.to_vec()
}

// Distinguish different values by length.
fn i_to_val_138(i: usize) -> Vec<u8> {
    let mut val = format!("A_{:08}", i).into_bytes();
    val.append(&mut random_val());
    val
}

fn i_to_val_140(i: usize) -> Vec<u8> {
    let mut val = format!("B_{:010}", i).into_bytes();
    val.append(&mut random_val());
    val
}

fn i_to_val_142(i: usize) -> Vec<u8> {
    let mut val = format!("C_{:012}", i).into_bytes();
    val.append(&mut random_val());
    val
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
