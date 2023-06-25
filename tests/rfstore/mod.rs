// Copyright 2023 TiKV Project Authors. Licensed under Apache-2.0.

use std::time::Duration;

use test_cloud_server::ServerCluster;
use tikv_util::{
    config::{ReadableDuration, ReadableSize},
    info,
    time::Instant,
};

#[test]
fn test_rfstore_latency() {
    test_util::init_log_for_test();
    let nodes = vec![1, 2, 3];
    let cluster = ServerCluster::new(nodes, |_, config| {
        config.server.grpc_keepalive_time = ReadableDuration::secs(5);
        config.raft_store.raft_base_tick_interval = ReadableDuration::secs(1);
        config.raft_store.raft_election_timeout_ticks = 10;
        config.raft_store.raft_store_max_leader_lease = ReadableDuration::secs(9);
        config.raft_store.split_region_check_tick_interval = ReadableDuration::secs(10);
        config.raft_store.raft_log_gc_tick_interval = ReadableDuration::secs(3);
        config.raft_store.pd_heartbeat_tick_interval = ReadableDuration::secs(60);
        config.raft_store.pd_store_heartbeat_tick_interval = ReadableDuration::secs(10);
        config.rocksdb.writecf.write_buffer_size = ReadableSize::kb(512);
        config.rocksdb.writecf.block_size = ReadableSize::kb(16);
        config.rocksdb.writecf.target_file_size_base = ReadableSize::kb(512);
        config.rfengine.target_file_size = ReadableSize::mb(16);
    });
    cluster.wait_region_replicated(&[], 3);
    cluster.get_pd_client().disable_default_operator();
    let mut client = cluster.new_client();
    let mut keyspace_id = 1;
    for _ in 0..10 {
        info!("spit keyspace {}", keyspace_id);
        let keyspaces = keyspace_id..(keyspace_id + 1000);
        client.split_keyspaces(keyspaces);
        keyspace_id += 1000;
    }
    std::thread::sleep(Duration::from_secs(5));
    for i in 0..10000 {
        let start = Instant::now();
        client.put_kv(i..i + 1, i_to_key, i_to_key);
        let latency = start.saturating_elapsed();
        if latency > Duration::from_millis(30) {
            info!("latency: {:?} at {}", start.saturating_elapsed(), i);
        }
    }
}

fn i_to_key(i: usize) -> Vec<u8> {
    let keyspace = i % 100 + 1;
    let mut key = api_version::ApiV2::get_txn_keyspace_prefix(keyspace as u32);
    key.extend_from_slice(format!("key_{:05}", i).as_bytes());
    key
}
