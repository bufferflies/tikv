// Copyright 2022 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    collections::HashMap,
    str::FromStr,
    sync::{mpsc::SyncSender, Arc, Mutex},
    time::Duration,
};

use futures::executor::block_on;
use hyper::{Body, Request, Uri};
use kvengine::{EngineStats, ShardTruncateTsStats};
use kvproto::metapb::Store;
use pd_client::PdClient;
use rand::Rng;
use test_cloud_server::{client::ClusterClient, ServerCluster};
use test_pd_client::TestPdClient;
use tikv_util::{info, time::Instant};
use tokio::runtime::Runtime;
use txn_types::TimeStamp;

use crate::alloc_node_id;

struct TruncateTsContext {
    client: ClusterClient,
    max_key_idx: usize,
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
    // preload some key-value pairs.
    client.put_kv(0..max_key_idx, i_to_key, i_to_random_value);

    test_truncate_ts_impl(client, max_key_idx, Duration::from_secs(30));

    // restart one node.
    let node_id = nodes[rand::thread_rng().gen_range(0..node_cnt)];
    cluster.stop_node(node_id);
    cluster.start_node(node_id, |_, _| {});
    info!("Cluster node {} is restarted.", node_id);

    // test truncate ts after restart
    let client = cluster.new_client();
    test_truncate_ts_impl(client, max_key_idx, Duration::from_secs(30));

    info!("Truncate ts test stopped.");
    cluster.stop();
}

fn test_truncate_ts_impl(client: ClusterClient, max_key_idx: usize, timeout: Duration) {
    let context1 = Arc::new(Mutex::new(TruncateTsContext {
        client,
        max_key_idx,
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

fn must_get_tso(pd_client: Arc<TestPdClient>) -> TimeStamp {
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
        let saved_tso = must_get_tso(context.lock().unwrap().client.pd_client.clone());
        info!("Get ts {} as the truncate ts", saved_tso);
        // wait a while to let update thread to write some data.
        std::thread::sleep(Duration::from_secs(rng.gen_range(0..max_sleep_time)));
        {
            info!("Begin truncate ts with {}", saved_tso);
            let truncate_ts = saved_tso.into_inner();
            let mut ctx = context.lock().unwrap();
            execute_truncate_ts(&ctx.client, truncate_ts);

            // do verification, compare the value with latest version with truncate_ts version.
            let now = Instant::now();
            for i in 0..ctx.max_key_idx {
                let key = i_to_key(i);
                let value = ctx.client.must_get_key_version(&key, truncate_ts, now);
                let cur_value = ctx.client.must_get_key(&key, now);
                assert_eq!(value, cur_value);
            }
            truncate_number += 1;
        }
    }
}

fn execute_truncate_ts(client: &ClusterClient, ts: u64) {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(3)
        .enable_all()
        .build()
        .unwrap();
    let cluster_id = client.pd_client.get_cluster_id().unwrap();
    let stores = client.pd_client.get_all_stores(true).unwrap();
    let mut remain_stores = HashMap::new();
    for store in stores {
        remain_stores.insert(store.id, store);
    }
    let shard_cnt =
        request_truncate_ts_on_all_stores(cluster_id, &remain_stores, ts, &runtime).unwrap();
    info!("{} shards begin truncate ts to {}", shard_cnt, ts);

    let start = Instant::now();
    // wait 200 * 50ms at maximum
    wait_truncate_ts_finish(
        &mut remain_stores,
        ts,
        &runtime,
        Duration::from_millis(50),
        200,
    )
    .unwrap();
    info!(
        "Truncate ts takes {:?} ",
        Instant::now().duration_since(start)
    );

    if !remain_stores.is_empty() {
        panic!("Some stores failed to truncate ts, {:?}", remain_stores);
    }
}

// send truncate ts request and return the count of shard which execute truncate ts.
fn request_truncate_ts_on_all_stores(
    cluster_id: u64,
    stores: &HashMap<u64, Store>,
    truncate_ts: u64,
    runtime: &Runtime,
) -> Result<usize, String> {
    let mut shard_cnt = 0;
    let store_cnt = stores.len();
    let (tx, rx) = std::sync::mpsc::sync_channel(store_cnt);
    for store in stores.values() {
        runtime.spawn(request_truncate_ts_store(
            cluster_id,
            store.clone(),
            truncate_ts,
            tx.clone(),
        ));
    }
    let mut errs = vec![];
    for _ in 0..store_cnt {
        match rx.recv().unwrap() {
            Ok((_, resp)) => {
                shard_cnt += resp.len();
            }
            Err(err) => {
                errs.push(err);
            }
        }
    }
    if !errs.is_empty() {
        return Err(errs.join(";"));
    }
    Ok(shard_cnt)
}

async fn request_truncate_ts_store(
    cluster_id: u64,
    store: Store,
    truncate_ts: u64,
    tx: SyncSender<Result<(u64, Vec<ShardTruncateTsStats>), String>>,
) {
    let uri = Uri::from_str(&format!("http://{}/truncate-ts", &store.status_address)).unwrap();
    let store_id = store.get_id();
    let mut body_map = HashMap::new();
    body_map.insert("cluster_id".to_string(), cluster_id.to_string());
    body_map.insert("truncate_ts".to_string(), truncate_ts.to_string());
    let json_string = serde_json::to_string(&body_map).unwrap();
    let req = Request::post(uri).body(Body::from(json_string)).unwrap();
    let client = hyper::Client::new();
    let resp = client.request(req).await.unwrap();
    if !resp.status().is_success() {
        tx.send(Err(format!("Request to store {:?} failed", store_id)))
            .unwrap();
        return;
    }
    let body = hyper::body::to_bytes(resp.into_body()).await.unwrap();
    let resp: Vec<ShardTruncateTsStats> = serde_json::from_slice(&body).unwrap();
    tx.send(Ok((store_id, resp))).unwrap();
}

// query max ts on all stores and remove the finished store in stores.
fn wait_truncate_ts_finish(
    stores: &mut HashMap<u64, Store>,
    truncate_ts: u64,
    runtime: &Runtime,
    interval: Duration,
    max_retry_cnt: u64,
) -> Result<(), String> {
    for _ in 0..max_retry_cnt {
        // wait a while for truncate finish.
        std::thread::sleep(interval);

        let cnt = stores.len();
        let (tx, rx) = std::sync::mpsc::sync_channel(cnt);
        for (_, store) in stores.clone() {
            runtime.spawn(query_max_ts_store(store, tx.clone()));
        }
        let mut errs = vec![];
        for _ in 0..cnt {
            match rx.recv().unwrap() {
                Ok((store_id, max_ts)) => {
                    if max_ts <= truncate_ts {
                        stores.remove(&store_id);
                    }
                }
                Err(err) => {
                    errs.push(err);
                }
            }
        }
        if !errs.is_empty() {
            return Err(errs.join(";"));
        }
        if stores.is_empty() {
            return Ok(());
        }
    }
    Ok(())
}

async fn query_max_ts_store(store: Store, tx: SyncSender<Result<(u64, u64), String>>) {
    let uri = Uri::from_str(&format!("http://{}/kvengine", &store.status_address)).unwrap();
    let store_id = store.get_id();
    let client = hyper::Client::new();
    match client.get(uri).await {
        Ok(resp) => {
            let body = hyper::body::to_bytes(resp.into_body()).await.unwrap();
            let engine_stats: EngineStats = serde_json::from_slice(&body).unwrap();
            tx.send(Ok((store_id, engine_stats.max_ts))).unwrap()
        }
        Err(e) => tx
            .send(Err(format!("Store {:?} failed, {:?}", store_id, e)))
            .unwrap(),
    }
}

fn i_to_key(i: usize) -> Vec<u8> {
    format!("key_{:03}", i).into_bytes()
}

fn i_to_random_value(_i: usize) -> Vec<u8> {
    let random_number: usize = rand::thread_rng().gen();
    format!("value_{:20}", random_number).into_bytes()
}
