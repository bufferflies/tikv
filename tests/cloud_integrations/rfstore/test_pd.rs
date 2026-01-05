// Copyright 2025 TiKV Project Authors. Licensed under Apache-2.0.

use health_controller::types::{InspectFactor, LatencyInspector};
use rfstore::{store::StoreMsg, RaftStoreRouter};
use strum::IntoEnumIterator;
use test_rfstore::new_node_cluster;
use tikv_util::{config::ReadableDuration, time::Instant, HandyRwLock};

#[test]
fn test_latency_inspect() {
    let mut cluster = new_node_cluster(0, 1);
    cluster.cfg.raft_store.store_io_pool_size = 2;
    cluster.cfg.raft_store.inspect_kvdb_interval = ReadableDuration::millis(500);
    cluster.run();

    let binding = cluster.sim.wl();
    let router = binding.get_node_router(1);
    let (tx, rx) = std::sync::mpsc::sync_channel(10);
    // Inspect different disk_factors.
    for factor in InspectFactor::iter() {
        let cloned_tx = tx.clone();
        let inspector = LatencyInspector::new(
            1,
            Box::new(move |_, duration| {
                let dur = duration.sum(true);
                cloned_tx.send(dur).unwrap();
            }),
        );
        let msg = StoreMsg::LatencyInspect {
            factor,
            send_time: Instant::now(),
            inspector,
        };
        match factor {
            InspectFactor::RaftDisk => {
                router.send_store_msg(msg);
                rx.recv_timeout(std::time::Duration::from_secs(2)).unwrap();
            }
            InspectFactor::KvDisk | InspectFactor::Network => {
                // Since kvengine and rfengine use the same mount path in test
                // env, it should timeout. And as for InspectFactor::Network, it should not be
                // handled by StoreFsm.
                router.send_store_msg(msg);
                rx.recv_timeout(std::time::Duration::from_secs(1))
                    .unwrap_err();
            }
        }
    }
}
