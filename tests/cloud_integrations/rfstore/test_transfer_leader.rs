// Copyright 2025 TiKV Project Authors. Licensed under Apache-2.0.

use std::{sync::Arc, thread, time::Duration};

use pd_client::PdClient;
use test_pd_client::PdClientExt;
use test_raftstore::{
    new_admin_request, new_get_cmd, new_peer, new_request, new_transfer_leader_cmd, sleep_ms,
};
use test_rfstore::*;
use tikv_util::debug;

#[test]
fn test_server_basic_transfer_leader() {
    let mut cluster = new_node_cluster(0, 3);
    // heartbeat ticks can't be higher than election timeout ticks.
    cluster.cfg.raft_store.raft_heartbeat_ticks = 9;
    let reserved_time = Duration::from_millis(
        cluster.cfg.raft_store.raft_base_tick_interval.as_millis()
            * cluster.cfg.raft_store.raft_heartbeat_ticks as u64
            + cluster
                .cfg
                .raft_store
                .max_entry_cache_warmup_duration
                .as_millis(),
    );
    cluster.run();

    // transfer leader to (2, 2) first to make address resolve happen early.
    cluster.must_transfer_leader(1, new_peer(2, 2));
    cluster.must_transfer_leader(1, new_peer(1, 1));

    let mut region = cluster.get_region(b"k3");

    // ensure follower has latest entries before transfer leader.
    cluster.must_put(b"k1", b"v1");
    must_get_equal(&cluster.get_engine(2), region.id, b"k1", b"v1");

    // check if transfer leader is fast enough.
    let leader = cluster.leader_of_region(1).unwrap();
    let admin_req = new_transfer_leader_cmd(new_peer(2, 2));
    let mut req = new_admin_request(1, region.get_region_epoch(), admin_req);
    req.mut_header().set_peer(leader);
    cluster.call_command(req, Duration::from_secs(3)).unwrap();
    thread::sleep(reserved_time);
    assert_eq!(
        cluster.query_leader(2, 1, Duration::from_secs(5)),
        Some(new_peer(2, 2))
    );

    let mut req = new_write_request(
        region.get_id(),
        region.take_region_epoch(),
        new_put_cmd(b"k3", b"v3"),
    );
    req.mut_header().set_peer(new_peer(2, 2));
    // transfer leader to (1, 1)
    cluster.must_transfer_leader(1, new_peer(1, 1));
    // send request to old leader (2, 2) directly and verify it fails
    let resp = cluster.call_command(req, Duration::from_secs(5)).unwrap();
    assert!(resp.get_header().get_error().has_not_leader());
}

#[test]
fn test_server_pd_transfer_leader() {
    let mut cluster = new_node_cluster(0, 3);
    let pd_client = Arc::clone(&cluster.pd_client);
    pd_client.disable_default_operator();

    cluster.run();

    cluster.must_put(b"k", b"v");

    // call command on this leader directly, must successfully.
    let mut region = cluster.get_region(b"");
    let mut req = new_request(
        region.get_id(),
        region.take_region_epoch(),
        vec![new_get_cmd(b"k")],
        false,
    );

    for id in 1..4 {
        // select a new leader to transfer
        pd_client.transfer_leader(region.get_id(), new_peer(id, id), vec![]);

        for _ in 0..100 {
            // reset leader and wait transfer successfully.
            cluster.reset_leader_of_region(1);

            sleep_ms(20);

            if let Some(leader) = cluster.leader_of_region(1) {
                if leader.get_id() == id {
                    // make sure new leader apply an entry on its term
                    // so we can use its local reader safely
                    cluster.must_put(b"k1", b"v1");
                    break;
                }
            }
        }

        assert_eq!(cluster.leader_of_region(1), Some(new_peer(id, id)));
        req.mut_header().set_peer(new_peer(id, id));
        debug!("requesting {:?}", req);
        let resp = cluster
            .call_command(req.clone(), Duration::from_secs(5))
            .unwrap();
        assert!(!resp.get_header().has_error(), "{:?}", resp);
        assert_eq!(resp.get_responses()[0].get_get().get_value(), b"v");
    }
}

#[test]
#[ignore = "need to cherry pick https://github.com/tikv/tikv/pull/11063"]
fn test_server_pd_transfer_leader_multi_target() {
    let mut cluster = new_node_cluster(0, 3);
    let pd_client = Arc::clone(&cluster.pd_client);
    pd_client.disable_default_operator();

    cluster.run();

    let mut region = pd_client.get_region(b"k").unwrap();

    cluster.must_put(b"k1", b"v1");
    must_get_equal(&cluster.get_engine(3), region.id, b"k1", b"v1");
    cluster.add_send_filter(IsolationFilterFactory::new(3));
    cluster.must_put(b"k2", b"v2");
    // append index of the cluster will finally be 2, 2, 1
    must_get_equal(&cluster.get_engine(1), region.id, b"k2", b"v2");
    must_get_equal(&cluster.get_engine(2), region.id, b"k2", b"v2");
    must_get_none(&cluster.get_engine(3), region.id, b"k2");

    // select multi target leaders to transfer
    // set `peer` to 3, make sure the new leader comes from `peers`
    pd_client.transfer_leader(
        region.get_id(),
        new_peer(3, 3),
        vec![new_peer(2, 2), new_peer(3, 3)],
    );

    for _ in 0..100 {
        // reset leader and wait transfer successfully.
        cluster.reset_leader_of_region(1);

        sleep_ms(20);

        if let Some(leader) = cluster.leader_of_region(1) {
            if leader.get_id() == 2 {
                break;
            }
        }
    }

    // Give some time for leader to commit the first entry
    // todo: It shouldn't need this, but for now and for v2, without it, the test is
    // not stable.
    thread::sleep(Duration::from_millis(100));

    // call command on this leader directly, must successfully.
    let mut req = new_request(
        region.get_id(),
        region.take_region_epoch(),
        vec![new_get_cmd(b"k1")],
        false,
    );
    let leader = cluster.leader_of_region(1).unwrap();
    assert_eq!(leader, new_peer(2, 2), "leader {:?}", leader);
    req.mut_header()
        .set_peer(new_peer(leader.store_id, leader.id));
    let resp = cluster.call_command(req, Duration::from_secs(5)).unwrap();
    assert!(!resp.get_header().has_error(), "{:?}", resp);
    assert_eq!(resp.get_responses()[0].get_get().get_value(), b"v1");
}
