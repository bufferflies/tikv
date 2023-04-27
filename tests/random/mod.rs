// Copyright 2022 TiKV Project Authors. Licensed under Apache-2.0.

mod test_native_br;

use std::{
    sync::{
        atomic::{AtomicU16, AtomicUsize, Ordering},
        Arc, RwLock,
    },
    thread::{sleep, JoinHandle},
    time::Duration,
};

use futures::executor::block_on;
use kvengine::dfs::DFSConfig;
use kvproto::pdpb::CheckPolicy;
use pd_client::PdClient;
use rand::{Rng, RngCore};
use tempfile::TempDir;
use test_cloud_server::{
    client::ClusterClient, oss::ObjectStorageService, scheduler::Scheduler, try_wait, ServerCluster,
};
use test_pd_client::TestPdClient;
use tikv::config::TikvConfig;
use tikv_util::{
    config::{ReadableDuration, ReadableSize},
    error, info,
    time::Instant,
};
use txn_types::Key;

lazy_static::lazy_static! {
    pub static ref WRITE_COUNTER: AtomicUsize = AtomicUsize::new(0);
    pub static ref MOVE_COUNTER: AtomicUsize = AtomicUsize::new(0);
    pub static ref MERGE_COUNTER: AtomicUsize = AtomicUsize::new(0);
    pub static ref TRANSFER_COUNTER: AtomicUsize = AtomicUsize::new(0);
    pub static ref NODE_RESTART_COUNTER: AtomicUsize = AtomicUsize::new(0);
    pub static ref BACKUP_COUNTER: AtomicUsize = AtomicUsize::new(0);
    pub static ref RESTORE_COUNTER: AtomicUsize = AtomicUsize::new(0);
}

pub const TIMEOUT: Duration = Duration::from_secs(60);
pub const CONCURRENCY: usize = 4;

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
        spawn_merge(cluster.new_scheduler(), false),
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
        "total_write_count {}, region number {}, merge count {}, move count {}, transfer count {}",
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

pub(crate) fn spawn_keyspace_write(
    idx: usize,
    mut client: ClusterClient,
    keyspace_count: usize,
    timeout: Duration,
) -> JoinHandle<()> {
    std::thread::spawn(move || {
        // Make sure each write thread don't conflict with others.
        let begin = idx * 2000;
        let end = begin + 2000 - 10;
        let start_time = Instant::now();
        let mut rng = rand::thread_rng();
        while start_time.saturating_elapsed() < timeout {
            let keyspace = rng.gen_range(0..keyspace_count);
            let i_to_keyspace_key = generate_keyspace_key(keyspace as u32);

            let i = rng.gen_range(begin..end);
            let res = if rng.gen_ratio(2, 3) {
                client.try_put_kv(i..(i + 10), &i_to_keyspace_key, i_to_val)
            } else {
                client.del_kv(i..(i + 10), &i_to_keyspace_key);
                Ok(())
            };
            if res.is_err() {
                // TODO: raise the error
                error!("keyspace_write error: {:?}", res);
            }
            WRITE_COUNTER.fetch_add(10, Ordering::SeqCst);
        }
        info!("keyspace write thread {} exit", idx);
    })
}

pub(crate) fn i_to_key(i: usize) -> Vec<u8> {
    format!("key{:08}", i).into_bytes()
}

pub(crate) fn i_to_val(i: usize) -> Vec<u8> {
    let mut rng = rand::thread_rng();
    let mut buf = vec![0; i % 512 + 1];
    rng.fill_bytes(&mut buf);
    buf
}

pub(crate) fn get_keyspace_prefix(keyspace_id: u32) -> Vec<u8> {
    let mut prefix = keyspace_id.to_be_bytes();
    prefix[0] = b'x';
    prefix.to_vec()
}

pub(crate) fn generate_keyspace_key(keyspace_id: u32) -> impl Fn(usize) -> Vec<u8> {
    move |i: usize| -> Vec<u8> {
        let mut key = get_keyspace_prefix(keyspace_id);
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
