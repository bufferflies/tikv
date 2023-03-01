// Copyright 2023 TiKV Project Authors. Licensed under Apache-2.0.

use std::time::Duration;

use cse_ctl::{backup, common::now, restore, restore_tenant, step};
use kvengine::dfs::DFSConfig;
use rand::Rng;
use test_cloud_server::{oss::ObjectStorageService, try_wait, ServerCluster};
use tikv::config::TiKvConfig;
use tikv_util::{config::ReadableSize, info, warn};
use tokio::runtime::Runtime;

use crate::alloc_node_id_vec;

const BASIC_DATA_COUNT: usize = 10;
const RANDOM_VALUE_LEN: usize = 128;
const NODES_COUNT: usize = 3;
const KEYSPACE_COUNT: usize = 3;
const DEFAULT_LOOP_COUNT: usize = 3;

#[test]
fn test_inplace_restore_tenant() {
    test_util::init_log_for_test();

    let loop_count = std::env::var("LOOP")
        .unwrap_or_default()
        .parse::<usize>()
        .unwrap_or(DEFAULT_LOOP_COUNT);

    let cases = vec![
        // keyspace_id, data_count, shuffle_regions, loop_count
        (1, 1, None, 1),
        (1, 100, None, 3),
        (1, 200, Some(10), loop_count),
        (2, 1, None, 1),
    ];

    let base_dir = tempfile::Builder::new()
        .prefix("test_inplace_restore_tenant_")
        .tempdir()
        .unwrap();

    let oss_dir = base_dir.path().join("oss");
    let mut oss = ObjectStorageService::new(oss_dir);
    oss.start_server();

    let dfs_config = DFSConfig {
        prefix: "test_inplace_restore_tenant_".to_string(),
        s3_endpoint: format!("http://127.0.0.1:{}", oss.port()),
        s3_key_id: "admin".to_string(),
        s3_secret_key: "admin".to_string(),
        s3_bucket: "test_inplace_restore_tenant_".to_string(),
        s3_region: "local".to_string(),
        zstd_compression_level: "3".to_string(),
        ..Default::default()
    };

    let mut cluster = ServerCluster::new(
        alloc_node_id_vec(NODES_COUNT),
        |_, conf: &mut TiKvConfig| {
            conf.dfs = dfs_config.clone();
            // Set small mem-table size to make data reach L1 and generate over bound shards.
            conf.rocksdb.writecf.write_buffer_size = ReadableSize::kb(1);
            conf.coprocessor.region_split_size = ReadableSize::kb(128); // kv_opts.base_size = 8kb
            conf.rfengine.target_file_size = ReadableSize::mb(1);
        },
    );
    cluster.wait_region_replicated(&[], 3);
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
    for (case_idx, &(keyspace_id, data_count, shuffle_regions, loop_count)) in
        cases.iter().enumerate()
    {
        for i in 0..loop_count {
            test_inplace_restore_tenant_impl(
                &mut cluster,
                &dfs_config,
                &format!("{case_idx}:{i}"),
                keyspace_id,
                data_count,
                shuffle_regions,
                &runtime,
            );
        }
    }

    cluster.stop();
    // Don't graceful shutdown, as some S3FS threads are still alive and holding connections.
    // oss.shutdown();
}

fn test_inplace_restore_tenant_impl(
    cluster: &mut ServerCluster,
    dfs_config: &DFSConfig,
    case_name: &str,
    keyspace_id: u32,
    data_count: usize,
    shuffle_regions: Option<usize>,
    runtime: &Runtime,
) {
    step!("case: {case_name}");
    let mut client = cluster.new_client();
    let i_to_key = gen_keyspace_key(keyspace_id);

    // Import data.
    client.put_kv(0..data_count, &i_to_key, i_to_val_140);
    let origin_ref_store = client.dump_ref_store();

    // Execute backup.
    let backup_name = format!("restore_tenant_test_{}", rand::thread_rng().gen::<u64>());
    let backup_config = backup::BackupConfig {
        dfs: dfs_config.clone(),
        skip_keyspace_meta: true,
        ..Default::default()
    };
    let backup_meta = backup::backup_cluster(
        backup_config,
        false,
        backup_name.clone(),
        cluster.get_pd_client().as_ref(),
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
            .verify_data_with_given_ref_store(&origin_ref_store)
            .is_err(),
        "case: {}",
        case_name,
    );
    step!("verify ok");

    // Restore tenant.
    let config = restore::RestoreConfig {
        dfs: dfs_config.clone(),
        skip_resolve_lock: true,
        ..Default::default()
    };
    restore_tenant::restore_keyspace(
        keyspace_id,
        &backup_name,
        None,
        &config,
        cluster.get_pd_client(),
        runtime,
    )
    .unwrap();
    step!("restore done");

    // Verify restored data.
    // Retry as shard restore is asynchronously applied.
    let ok = try_wait(
        || {
            client
                .verify_data_with_given_ref_store(&origin_ref_store)
                .is_ok()
        },
        10,
    );
    if !ok {
        client
            .verify_data_with_given_ref_store(&origin_ref_store)
            .unwrap();
    }
    step!("verify restore data done");

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
