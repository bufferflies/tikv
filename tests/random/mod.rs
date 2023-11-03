// Copyright 2022 TiKV Project Authors. Licensed under Apache-2.0.

mod test_all;
mod test_load_data;
mod test_native_br;

use std::{
    str::FromStr,
    sync::{
        atomic::{AtomicU16, AtomicUsize, Ordering},
        Arc, RwLock,
    },
    thread::{sleep, JoinHandle},
    time::Duration,
};

use api_version::ApiV2;
use futures::executor::block_on;
use http::{Request, StatusCode, Uri};
use hyper::Body;
use kvengine::dfs::DFSConfig;
use kvproto::{metapb::Store, pdpb::CheckPolicy};
use pd_client::PdClient;
use rand::{prelude::SliceRandom, Rng, RngCore};
use security::SecurityConfig;
use tempfile::TempDir;
use test_cloud_server::{
    client::ClusterClient,
    keyspace::{ClusterKeyspaceClient, KeyspaceManager},
    oss::ObjectStorageService,
    scheduler::Scheduler,
    try_wait, ServerCluster,
};
use test_pd_client::TestPdClient;
use tikv::config::TikvConfig;
use tikv_util::{
    box_err,
    config::{ReadableDuration, ReadableSize},
    info,
    time::Instant,
    warn,
};
use txn_types::Key;

pub(crate) type Error = Box<dyn std::error::Error + Send + Sync + 'static>;
pub(crate) type Result<T> = std::result::Result<T, Error>;

lazy_static::lazy_static! {
    pub static ref WRITE_COUNTER: AtomicUsize = AtomicUsize::new(0);
    pub static ref MOVE_COUNTER: AtomicUsize = AtomicUsize::new(0);
    pub static ref MERGE_COUNTER: AtomicUsize = AtomicUsize::new(0);
    pub static ref TRANSFER_COUNTER: AtomicUsize = AtomicUsize::new(0);
    pub static ref NODE_RESTART_COUNTER: AtomicUsize = AtomicUsize::new(0);
    pub static ref BACKUP_COUNTER: AtomicUsize = AtomicUsize::new(0);
    pub static ref RESTORE_COUNTER: AtomicUsize = AtomicUsize::new(0);
    pub static ref LOAD_DATA_COUNTER: AtomicUsize = AtomicUsize::new(0);
    pub static ref KEYSPACE_COUNTER: AtomicUsize = AtomicUsize::new(0);
    pub static ref TABLE_COUNTER: AtomicUsize = AtomicUsize::new(0);
    pub static ref MANUAL_MAJOR_COMPACT_COUNTER: AtomicUsize = AtomicUsize::new(0);
}

pub const TIMEOUT: Duration = Duration::from_secs(90);
pub const CONCURRENCY: usize = 4;

const DEFAULT_INNER_KEY_OFFSET: usize = 4;
const REQUEST_MAJOR_COMPACT_ON_STORE_TIMEOUT: Duration = Duration::from_secs(20);

static NODE_ALLOCATOR: AtomicU16 = AtomicU16::new(1);

pub(crate) fn alloc_node_id() -> u16 {
    let node_id = NODE_ALLOCATOR.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    info!("allocated node_id {}", node_id);
    node_id
}

pub(crate) fn alloc_node_id_vec(count: usize) -> Vec<u16> {
    let mut nodes = vec![];
    nodes.resize_with(count, || alloc_node_id());
    nodes
}

#[test]
fn test_random_merge() {
    test_util::init_log_for_test();
    // use 4 nodes for easier schedule merge.
    let nodes = vec![
        alloc_node_id(),
        alloc_node_id(),
        alloc_node_id(),
        alloc_node_id(),
    ];
    let bucket_size_kb = 64;
    let update_conf_fn = |_, conf: &mut TikvConfig| {
        conf.coprocessor.region_split_size = ReadableSize::kb(192);
        conf.coprocessor.region_bucket_size = ReadableSize::kb(bucket_size_kb);
        conf.raft_store.peer_stale_state_check_interval = ReadableDuration::secs(1);
        conf.raft_store.abnormal_leader_missing_duration = ReadableDuration::secs(3);
        conf.raft_store.max_leader_missing_duration = ReadableDuration::secs(5);
        conf.rocksdb.writecf.target_file_size_base = ReadableSize::kb(16);
        conf.rfengine.target_file_size = ReadableSize::mb(1);
        conf.rfengine.batch_compression_threshold =
            ReadableSize::kb(rand::thread_rng().gen_range(0..2));
        conf.kvengine.compaction_tombs_count = 100;
    };
    let mut cluster = ServerCluster::new(nodes.clone(), update_conf_fn);
    cluster.wait_region_replicated(&[], 3);
    cluster.get_pd_client().disable_default_operator();
    let region = cluster
        .get_pd_client()
        .get_all_regions()
        .first()
        .unwrap()
        .clone();
    let mut keys = vec![];
    for i in 0..20 {
        let key = i_to_key(i * 100);
        keys.push(Key::from_raw(&key).into_encoded());
    }
    cluster
        .get_pd_client()
        .must_split_region(region, CheckPolicy::Usekey, keys);
    let move_scheduler = cluster.new_scheduler();
    for _ in 0..20 {
        move_scheduler.move_random_region();
    }
    let handles = vec![
        spawn_write(0, cluster.new_client()),
        spawn_merge(cluster.new_scheduler(), true),
        spawn_transfer(cluster.new_scheduler()),
        spawn_move(cluster.new_scheduler(), Arc::new(RwLock::new(()))),
    ];
    let start_time = Instant::now();
    let pd_client = cluster.get_pd_client();
    while start_time.saturating_elapsed() < TIMEOUT {
        let ts = block_on(pd_client.get_tso()).unwrap();
        let mut rng = rand::thread_rng();
        let node_idx = rng.gen_range(0..nodes.len());
        let node_id = nodes[node_idx];
        info!("stop node {}", node_id);
        cluster.stop_node(node_id);
        info!("finish stop node {}", node_id);
        let sleep_sec = rng.gen_range(1..5);
        sleep(Duration::from_secs(sleep_sec));
        info!("start node {}", node_id);
        cluster.start_node(node_id, update_conf_fn);
        sleep(Duration::from_secs(10));
        pd_client.set_gc_safe_point(ts.into_inner());
    }
    info!("stop node thread exit");
    for handle in handles {
        handle.join().unwrap();
    }
    let ok = try_wait(
        || {
            let data_stats = cluster.get_data_stats();
            data_stats.check_data().is_ok()
        },
        10,
    );
    let data_stats = cluster.get_data_stats();
    if !ok {
        data_stats.check_data().unwrap();
    }

    cluster.wait_region_version_match();
    data_stats
        .check_buckets(&pd_client, bucket_size_kb * 1024)
        .unwrap();
    let mut client = cluster.new_client();
    client.verify_data_with_ref_store();
    cluster.stop();
    let total_write_count = WRITE_COUNTER.load(Ordering::SeqCst);
    let total_merge_count = MERGE_COUNTER.load(Ordering::SeqCst);
    let total_move_count = MOVE_COUNTER.load(Ordering::SeqCst);
    let total_transfer_count = TRANSFER_COUNTER.load(Ordering::SeqCst);
    let region_number = pd_client.get_regions_number();
    info!(
        "TEST SUCCEED: total_write_count {}, region number {}, merge count {}, move count {}, transfer count {}",
        total_write_count, region_number, total_merge_count, total_move_count, total_transfer_count,
    );
}

pub(crate) fn spawn_write(idx: usize, mut client: ClusterClient) -> JoinHandle<()> {
    std::thread::spawn(move || {
        // Make sure each write thread don't conflict with others.
        let begin = idx * 2000;
        let end = begin + 2000 - 10;
        let start_time = Instant::now();
        let mut rng = rand::thread_rng();
        while start_time.saturating_elapsed() < TIMEOUT {
            let i = rng.gen_range(begin..end);
            if rng.gen_ratio(2, 3) {
                client.put_kv(i..(i + 10), i_to_key, i_to_val);
            } else {
                client.del_kv(i..(i + 10), i_to_key);
            }
            WRITE_COUNTER.fetch_add(10, Ordering::SeqCst);
        }
        info!("write thread {} exit", idx);
    })
}

pub(crate) fn spawn_move(scheduler: Scheduler, two_node_down: Arc<RwLock<()>>) -> JoinHandle<()> {
    std::thread::spawn(move || {
        let start_time = Instant::now();
        while start_time.saturating_elapsed() < TIMEOUT {
            sleep(Duration::from_millis(1000));
            let guard = two_node_down.read().unwrap();
            scheduler.move_random_region();
            drop(guard);
            MOVE_COUNTER.fetch_add(1, Ordering::SeqCst);
        }
        info!("move thread exit");
    })
}

pub(crate) fn spawn_merge(scheduler: Scheduler, disallow_cross_keyspace: bool) -> JoinHandle<()> {
    std::thread::spawn(move || {
        let start_time = Instant::now();
        while start_time.saturating_elapsed() < TIMEOUT {
            if scheduler.merge_random_region(disallow_cross_keyspace) {
                MERGE_COUNTER.fetch_add(1, Ordering::SeqCst);
            }
        }
        info!("merge thread exit");
    })
}

pub(crate) fn spawn_transfer(scheduler: Scheduler) -> JoinHandle<()> {
    std::thread::spawn(move || {
        let start_time = Instant::now();
        while start_time.saturating_elapsed() < TIMEOUT {
            sleep(Duration::from_millis(100));
            if scheduler.transfer_random_leader() {
                TRANSFER_COUNTER.fetch_add(1, Ordering::SeqCst);
            }
        }
        info!("transfer thread exit");
    })
}

pub(crate) fn spawn_gc_worker(pd_client: Arc<TestPdClient>, timeout: Duration) -> JoinHandle<()> {
    std::thread::spawn(move || {
        let start_time = Instant::now();
        while start_time.saturating_elapsed() < timeout {
            let ts = block_on(pd_client.get_tso()).unwrap();
            sleep(Duration::from_secs(10));
            pd_client.set_gc_safe_point(ts.into_inner());
        }
        info!("gc worker thread exit");
    })
}

pub(crate) fn spawn_major_compact(
    pd_client: Arc<TestPdClient>,
    keyspace_manager: KeyspaceManager,
    timeout: Duration,
) -> JoinHandle<()> {
    std::thread::spawn(move || {
        let mut rng = rand::thread_rng();
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .thread_name("major-compact")
            .build()
            .unwrap();

        let start_time = Instant::now();
        while start_time.saturating_elapsed() < timeout {
            // Manual major compact should not conform the zipf distribution.
            let keyspace_id = keyspace_manager.get_uniform_random_keyspace(&mut rng);

            let stores = pd_client.get_all_stores(true).unwrap();
            {
                // To be mutual-exclusive with load_data.
                let lock = keyspace_manager.get_keyspace_lock(keyspace_id);
                let _guard = lock.extra_lock();

                let mut handles = Vec::with_capacity(stores.len());
                for store in stores {
                    handles.push(
                        runtime.spawn(request_major_compact_on_store(store.clone(), keyspace_id)),
                    );
                }
                for handle in handles {
                    runtime.block_on(handle).unwrap().unwrap();
                }
            }

            MANUAL_MAJOR_COMPACT_COUNTER.fetch_add(1, Ordering::SeqCst);
            sleep(Duration::from_secs(10));
        }
        info!("major compact worker thread exit");
    })
}

pub(crate) async fn request_major_compact_on_store(store: Store, keyspace_id: u32) -> Result<()> {
    let uri = Uri::from_str(&format!(
        "http://{}/major-compact?major_compact=true&keyspace_id={}",
        &store.status_address, keyspace_id
    ))
    .unwrap();
    let client = hyper::Client::new();
    let mut last_err: Option<Error> = None;
    let mut retry = 0;
    let start_time = Instant::now();
    while start_time.saturating_elapsed() < REQUEST_MAJOR_COMPACT_ON_STORE_TIMEOUT {
        retry += 1;
        let req = Request::post(&uri).body(Body::empty()).unwrap();
        match client.request(req).await {
            // Treat 404 as success.
            Ok(resp) if (resp.status().is_success() || resp.status() == StatusCode::NOT_FOUND) => {
                let body = hyper::body::to_bytes(resp.into_body()).await.unwrap();
                let msg = String::from_utf8_lossy(&body);
                info!(
                    "request_major_compact_on_store success, keyspace {}: {}",
                    keyspace_id,
                    msg.as_ref()
                );
                return Ok(());
            }
            Ok(resp) => {
                let body = hyper::body::to_bytes(resp.into_body()).await.unwrap();
                let msg = String::from_utf8_lossy(&body);
                // Return error only when there is bad request argument.
                panic!(
                    "request_major_compact_on_store failed: {}, retry {}",
                    msg.as_ref(),
                    retry
                );
            }
            Err(err) => {
                last_err = Some(box_err!(
                    "request_major_compact_on_store failed: {:?}, retry {}",
                    err,
                    retry
                ));
                warn!("{:?}", last_err.as_ref().unwrap());
                tokio::time::sleep(Duration::from_millis(500)).await;
                continue;
            }
        }
    }
    Err(last_err.unwrap())
}

pub(crate) fn spawn_keyspace_write(
    idx: usize,
    client: ClusterKeyspaceClient,
    timeout: Duration,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        // Make sure each write thread don't conflict with others.
        let begin = idx * 2000;
        let end = begin + 2000 - 10;
        let start_time = Instant::now();
        while start_time.saturating_elapsed() < timeout {
            let random = || {
                let mut rng = rand::thread_rng();
                let keyspace_id = client.keyspace_manager().get_zipf_random_keyspace(&mut rng);
                let table_id = client
                    .keyspace_manager()
                    .get_random_available_table(keyspace_id, &mut rng);
                let i = rng.gen_range(begin..end);
                let put_kv = rng.gen_ratio(2, 3);
                (keyspace_id, table_id, i, put_kv)
            };
            let (keyspace_id, table_id, i, put_kv) = random();
            let table_id = match table_id {
                Some(table_id) => table_id,
                None => continue,
            };

            {
                let lock = client.keyspace_manager().get_keyspace_lock(keyspace_id);
                let guard = lock.try_shared_lock();
                if guard.is_none() {
                    continue;
                }
                let _guard = guard.unwrap();
                info!("[{}] thread write on keyspace {}", idx, keyspace_id);

                if put_kv {
                    client
                        .keyspace_put_kv(keyspace_id, table_id, i..(i + 10), i_to_key, i_to_val)
                        .await
                        .unwrap();
                } else {
                    client
                        .keyspace_del_kv(keyspace_id, table_id, i..(i + 10), i_to_key)
                        .await
                        .unwrap();
                };
            }
            WRITE_COUNTER.fetch_add(10, Ordering::SeqCst);
        }
        info!("keyspace write thread {} exit", idx);
    })
}

pub fn spawn_create_keyspace(
    pd_client: Arc<TestPdClient>,
    keyspace_manager: KeyspaceManager,
    initial_table_count: usize,
    timeout: Duration,
) -> JoinHandle<()> {
    std::thread::spawn(move || {
        let mut rng = rand::thread_rng();
        let start_time = Instant::now();
        while start_time.saturating_elapsed() < timeout {
            sleep(Duration::from_secs(rng.gen_range(0..5)));

            let max_keyspace = keyspace_manager.max_keyspace_id().unwrap();
            let new_keyspace = rng.gen_range(max_keyspace + 1..=max_keyspace + 3);

            must_split_region_for_keyspace(&pd_client, new_keyspace);
            // Don't shuffle keyspaces. Otherwise we will not have a few big keyspaces.
            // Note that the new keyspace will has less chance to be written.
            keyspace_manager.create_keyspaces(
                &[new_keyspace],
                DEFAULT_INNER_KEY_OFFSET,
                initial_table_count,
                None,
            );

            KEYSPACE_COUNTER.fetch_add(1, Ordering::Relaxed);
            TABLE_COUNTER.fetch_add(initial_table_count, Ordering::Relaxed);
        }
        info!("create keyspace thread exit");
    })
}

fn must_split_region_for_keyspace(pd_client: &TestPdClient, keyspace_id: u32) {
    let keys = vec![
        ApiV2::get_txn_keyspace_prefix(keyspace_id),
        ApiV2::get_txn_keyspace_prefix(keyspace_id + 1),
    ];
    let mut split_keys = keys
        .into_iter()
        .map(|k| Key::from_raw(&k).into_encoded())
        .collect::<Vec<_>>();
    let region_key = split_keys[0].clone();

    let ok = try_wait(
        || {
            let region = pd_client.get_region(&split_keys[0]).unwrap();
            if region.get_start_key() == split_keys[0] {
                split_keys.remove(0);
            }
            if split_keys
                .last()
                .map_or(false, |last| region.get_end_key() == last.as_slice())
            {
                split_keys.pop();
            }
            if split_keys.is_empty() {
                return true;
            }
            pd_client.split_region(region, CheckPolicy::Usekey, split_keys.clone());
            false
        },
        10,
    );
    let new_region = pd_client.get_region(&region_key).unwrap();
    assert!(ok, "split region failed: {:?}", new_region);
    info!(
        "split region for keyspace {}: {:?}",
        keyspace_id, new_region
    );
}

pub(crate) fn random_node_restart(cluster: &mut ServerCluster) {
    let mut rng = rand::thread_rng();

    // Some regions would loss majority for a wile when the sleep duration is small.
    // Data corruption should not happen, and workloads should tolerate this.
    sleep(Duration::from_secs(rng.gen_range(3..17)));

    let nodes = cluster.get_nodes();
    let node_id = *nodes.choose(&mut rng).unwrap();
    let sleep_sec = rng.gen_range(0..3);
    let force_stop = rng.gen();
    cluster.restart_node(node_id, Duration::from_secs(sleep_sec), force_stop);
    NODE_RESTART_COUNTER.fetch_add(1, Ordering::Relaxed);
}

pub(crate) fn i_to_key(i: usize) -> Vec<u8> {
    format!("xkey{:08}", i).into_bytes()
}

pub(crate) fn i_to_val(i: usize) -> Vec<u8> {
    let mut rng = rand::thread_rng();
    let mut buf = vec![0; i % 512 + 1];
    rng.fill_bytes(&mut buf);
    buf
}

pub(crate) fn generate_keyspace_key(keyspace_id: u32) -> impl Fn(usize) -> Vec<u8> {
    move |i: usize| -> Vec<u8> {
        let mut key = ApiV2::get_txn_keyspace_prefix(keyspace_id);
        key.extend(i_to_key(i));
        key
    }
}

pub(crate) fn prepare_dfs(prefix: &str) -> (TempDir, ObjectStorageService, DFSConfig) {
    let base_dir = tempfile::Builder::new().prefix(prefix).tempdir().unwrap();

    let oss_dir = base_dir.path().join("oss");
    let mut oss = ObjectStorageService::new(oss_dir);
    oss.start_server();

    let dfs_config = DFSConfig {
        prefix: prefix.to_string(),
        s3_endpoint: format!("http://127.0.0.1:{}", oss.port()),
        s3_key_id: "admin".to_string(),
        s3_secret_key: "admin".to_string(),
        s3_bucket: prefix.to_string(),
        s3_region: "local".to_string(),
        zstd_compression_level: "3".to_string(),
        ..Default::default()
    };

    (base_dir, oss, dfs_config)
}

pub(crate) fn new_security_config() -> SecurityConfig {
    let mut conf = SecurityConfig::default();
    conf.master_key.vendor = "test".to_string();
    conf.master_key.key_id = "random".to_string();
    conf
}
