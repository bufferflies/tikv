use kvproto::kvrpcpb::Context;
use pd_client::PdClient;
use test_cloud_server::{alloc_node_id_vec, ServerCluster};
use test_pd_client::PdClientExt;
use tikv_kv::Engine;

use crate::i_to_key;

/// RaftKV precheck_write_with_ctx checks if the current role is leader.
/// When it is not, it should return NotLeader error during prechecking.
#[test]
fn test_raftkv_precheck_write_with_ctx() {
    test_util::init_log_for_test();
    let cluster = ServerCluster::new(alloc_node_id_vec(3), |_, _| {});
    cluster.wait_region_replicated(&[], 1);
    let mut client = cluster.new_client();
    let keyspace_id = api_version::ApiV2::get_u32_keyspace_id_by_key(b"xkey").unwrap();
    client.split_keyspace(keyspace_id);

    cluster.wait_region_replicated(&i_to_key(1), 3);
    // make sure leader has been elected.
    assert_eq!(client.kv_get(&i_to_key(1), u64::MAX).unwrap(), None);

    let region = cluster
        .get_pd_client()
        .get_region_info(&i_to_key(1))
        .unwrap();
    let leader_peer = region.leader.as_ref().unwrap();
    let follower_peer = region
        .peers
        .iter()
        .find(|p| p.id != leader_peer.id)
        .unwrap();
    let leader_storage = cluster
        .get_raft_kv(cluster.get_server_node_id(leader_peer.store_id).unwrap())
        .unwrap();
    let follower_storage = cluster
        .get_raft_kv(cluster.get_server_node_id(follower_peer.store_id).unwrap())
        .unwrap();

    // Assume this is a write request.
    let mut ctx = Context::default();
    ctx.set_region_id(region.get_id());
    ctx.set_region_epoch(region.get_region_epoch().clone());
    ctx.set_peer(region.get_peers()[0].clone());

    // The (write) request can be sent to the leader.
    leader_storage.precheck_write_with_ctx(&ctx).unwrap();
    // The (write) request should not be send to a follower.
    follower_storage.precheck_write_with_ctx(&ctx).unwrap_err();

    // Manually transfer leader so that the previous leader is no longer the leader.
    cluster
        .get_pd_client()
        .transfer_leader(region.id, follower_peer.clone(), vec![]);
    let mut transferred = false;
    for _retry in 0..20 {
        let new_region_info = cluster
            .get_pd_client()
            .get_region_info(&i_to_key(1))
            .unwrap();
        if new_region_info.leader.as_ref().unwrap().id != leader_peer.id {
            transferred = true;
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
    assert!(transferred);
    leader_storage.precheck_write_with_ctx(&ctx).unwrap_err();
}
