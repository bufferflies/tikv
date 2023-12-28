// Copyright 2022 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    sync::{Arc, Mutex},
    time::Duration,
};

use futures::executor::block_on;
use native_br::truncate_ts::{truncate_ts_with_cfg, TruncateTsConfig};
use pd_client::PdClient;
use rand::Rng;
use test_cloud_server::{client::ClusterClient, ServerCluster};
use tikv_util::{info, time::Instant};
use txn_types::TimeStamp;

use crate::alloc_node_id;

const TEST_KEY_SPACE_CNT: usize = 10;

struct TruncateTsContext {
    client: ClusterClient,
    max_key_idx: usize,
    keyspace_id: Option<u32>,
    timeout: Duration,
}

#[test]
fn test_truncate_ts() {
    test_util::init_log_for_test();
    let mut nodes = vec![];
    let node_cnt = 3;
    for _ in 0..node_cnt {
        nodes.push(alloc_node_id());
    }
    let max_key_idx = 2000;
    let mut cluster = ServerCluster::new(nodes.clone(), |_, _| {});
    let mut client = cluster.new_client();
    // Split keyspaces.
    for keyspace_id in 0..=TEST_KEY_SPACE_CNT {
        client.split(&get_keyspace_prefix(keyspace_id as u32));
    }
    cluster.wait_pd_region_count(TEST_KEY_SPACE_CNT + 2);
    // preload some key-value pairs.
    client.put_kv(0..max_key_idx, i_to_key, i_to_random_value);

    let keyspaces_cases = vec![None, Some(random_keyspace(max_key_idx))];
    for keyspace in keyspaces_cases {
        let client = cluster.new_client();
        test_truncate_ts_impl(client, max_key_idx, keyspace, Duration::from_secs(30));
        // restart one node.
        let node_id = nodes[rand::thread_rng().gen_range(0..node_cnt)];
        cluster.stop_node(node_id);
        cluster.start_node(node_id, |_, _| {});
        info!("Cluster node {} is restarted.", node_id);

        // test truncate ts after restart
        let client = cluster.new_client();
        test_truncate_ts_impl(client, max_key_idx, keyspace, Duration::from_secs(30));
    }

    info!("Truncate ts test stopped.");
    cluster.stop();
}

fn random_keyspace(max_idx: usize) -> u32 {
    let mut rng = rand::thread_rng();
    i_to_keyspace(rng.gen_range(0..max_idx))
}

fn test_truncate_ts_impl(
    client: ClusterClient,
    max_key_idx: usize,
    keyspace_id: Option<u32>,
    timeout: Duration,
) {
    let context1 = Arc::new(Mutex::new(TruncateTsContext {
        client,
        max_key_idx,
        keyspace_id,
        timeout,
    }));
    let context2 = context1.clone();
    let handle1 = std::thread::spawn(move || random_update_kv(context1));
    let handle2 = std::thread::spawn(move || truncate_ts_and_verification(context2));
    handle1.join().unwrap();
    handle2.join().unwrap();
}

fn random_update_kv(context: Arc<Mutex<TruncateTsContext>>) {
    let min_batch_cnt = 10;
    let max_batch_cnt = 100;
    let start = Instant::now();
    let mut update_number = 0;
    let mut rng = rand::thread_rng();
    loop {
        {
            let mut ctx = context.lock().unwrap();
            if Instant::now().duration_since(start) > ctx.timeout {
                info!(
                    "Random update thread finish, update {} kvs totally",
                    update_number
                );
                break;
            }
            // update some key-value randomly.
            let batch_cnt = rng.gen_range(min_batch_cnt..max_batch_cnt);
            let start_idx = rng.gen_range(0..ctx.max_key_idx - batch_cnt);
            ctx.client.put_kv(
                start_idx..start_idx + batch_cnt,
                i_to_key,
                i_to_random_value,
            );
            update_number += batch_cnt;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
}

fn must_get_tso(pd_client: Arc<dyn PdClient>) -> TimeStamp {
    block_on(pd_client.get_tso()).unwrap()
}

fn truncate_ts_and_verification(context: Arc<Mutex<TruncateTsContext>>) {
    let start = Instant::now();
    let mut truncate_number = 0;
    let mut rng = rand::thread_rng();
    let max_sleep_time = 10; // in secs
    loop {
        {
            let ctx = context.lock().unwrap();
            if Instant::now().duration_since(start) > ctx.timeout {
                info!(
                    "Truncate ts thread finish, test {} times totally",
                    truncate_number
                );
                break;
            }
        }
        let saved_tso = must_get_tso(context.lock().unwrap().client.pd_client());
        info!("Get ts {} as the truncate ts", saved_tso);
        // wait a while to let update thread to write some data.
        std::thread::sleep(Duration::from_secs(rng.gen_range(0..max_sleep_time)));
        {
            let mut ctx = context.lock().unwrap();
            let mut values_before = vec![];
            if ctx.keyspace_id.is_some() {
                values_before = Vec::with_capacity(ctx.max_key_idx);
                let now = Instant::now();
                for i in 0..ctx.max_key_idx {
                    let key = i_to_key(i);
                    let (value, _) = ctx.client.must_get_key(&key, now);
                    values_before.push(value);
                }
            }
            info!("Begin truncate ts with {}", saved_tso);
            let truncate_ts = saved_tso.into_inner();
            execute_truncate_ts(&ctx.client, truncate_ts, ctx.keyspace_id);

            // do verification, compare the value with latest version with truncate_ts
            // version.
            let now = Instant::now();
            for i in 0..ctx.max_key_idx {
                let key = i_to_key(i);
                let (value, _) = ctx.client.must_get_key_version(&key, truncate_ts, now);
                let (cur_value, _) = ctx.client.must_get_key(&key, now);
                if ctx.keyspace_id.is_none() || ctx.keyspace_id == Some(i_to_keyspace(i)) {
                    assert_eq!(cur_value, value);
                } else {
                    // Verify the data out of keyspace is not truncated.
                    assert_eq!(&values_before[i], &cur_value);
                }
            }
            truncate_number += 1;
        }
    }
}

fn execute_truncate_ts(client: &ClusterClient, ts: u64, keyspace_id: Option<u32>) {
    let start = Instant::now();
    let truncate_ts_cfg = TruncateTsConfig {
        skip_resolve_lock: true,
        ..Default::default()
    };
    truncate_ts_with_cfg(
        truncate_ts_cfg,
        client.pd_client(),
        ts,
        Duration::from_secs(30),
        keyspace_id,
    )
    .unwrap();
    info!(
        "Truncate ts takes {:?} ",
        Instant::now().duration_since(start)
    );
}

fn get_keyspace_prefix(keyspace_id: u32) -> Vec<u8> {
    let mut prefix = keyspace_id.to_be_bytes();
    prefix[0] = b'x';
    prefix.to_vec()
}

fn i_to_keyspace(i: usize) -> u32 {
    (i % TEST_KEY_SPACE_CNT) as u32
}

fn i_to_key(i: usize) -> Vec<u8> {
    let mut keyspace_id = i_to_keyspace(i).to_be_bytes();
    keyspace_id[0] = api_version::api_v2::TXN_KEY_PREFIX;
    let mut key = keyspace_id.to_vec();
    key.append(&mut format!("key_{:03}", i).into_bytes());
    key
}

fn i_to_random_value(_i: usize) -> Vec<u8> {
    let random_number: usize = rand::thread_rng().gen();
    format!("value_{:20}", random_number).into_bytes()
}
