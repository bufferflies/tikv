// Copyright 2025 TiKV Project Authors. Licensed under Apache-2.0.

use kvproto::{
    metapb,
    raft_cmdpb::{CmdType, RaftCmdRequest, Request},
    raft_serverpb::RaftMessage,
};

use super::*;
use crate::{
    store::{initial_region, *},
    Error,
};

/// Sets up a minimal test environment with a Peer FSM and associated Raft
/// context.
fn build_test_env() -> (PeerFsm, RaftContext) {
    let (engines, _tmp_dir) = new_test_engines();
    let mut region = initial_region(1, 1, 1);
    region.mut_peers().push(new_peer(2, 2));
    region.mut_peers().push(new_peer(3, 3));

    // Build the peer FSM
    let fsm = new_test_peer_fsm(engines.clone(), &region).expect("Failed to create test PeerFsm");
    let raft_ctx = new_test_raft_ctx(engines, None);
    (fsm, raft_ctx)
}

#[test]
fn test_peer_fsm_create() {
    // Test that creating a PeerFsm with an empty region would fail.
    let empty_region = metapb::Region::default();
    let (engines, _tmp_dir) = new_test_engines();
    assert!(PeerFsm::create(1, &Config::default(), engines, &empty_region).is_err());
}

#[test]
fn test_validate_raft_msg() {
    test_util::init_log_for_test();
    let (mut fsm, mut raft_ctx) = build_test_env();

    let my_store_id = fsm.peer.peer.get_store_id();
    let my_peer_id = fsm.peer.peer.get_id();

    let mut handler = PeerMsgHandler::new(&mut fsm, &mut raft_ctx);
    let mut raft_msg = RaftMessage::default();
    raft_msg.set_to_peer(new_peer(my_store_id + 1, my_peer_id + 1));
    assert!(
        !handler.validate_raft_msg(&raft_msg),
        "msg with incorrect store_id should be invalid"
    );

    raft_msg.set_to_peer(new_peer(my_store_id, my_peer_id));
    assert!(
        !handler.validate_raft_msg(&raft_msg),
        "msg without epoch should be invalid"
    );
    assert_eq!(
        handler
            .ctx
            .raft_metrics
            .message_dropped
            .mismatch_region_epoch
            .get(),
        1
    );

    raft_msg.set_region_epoch(metapb::RegionEpoch::default());
    assert!(handler.validate_raft_msg(&raft_msg));
}

#[test]
fn test_check_msg() {
    test_util::init_log_for_test();
    let (mut fsm, mut raft_ctx) = build_test_env();

    let mut raft_msg = RaftMessage::default();
    raft_msg.mut_region_epoch().set_version(INIT_EPOCH_VER);
    raft_msg
        .mut_region_epoch()
        .set_conf_ver(INIT_EPOCH_CONF_VER);
    raft_msg.set_from_peer(new_peer(2, 2));

    // Missing to_peer
    let mut handler = PeerMsgHandler::new(&mut fsm, &mut raft_ctx);
    assert!(handler.check_msg(&raft_msg));
    assert_eq!(handler.ctx.raft_metrics.message_dropped.stale_msg.get(), 1);

    raft_msg.set_to_peer(new_peer(1, 1));
    assert!(!handler.check_msg(&raft_msg));
}

#[test]
fn test_pre_propose_raft_command() {
    test_util::init_log_for_test();
    let (mut fsm, mut raft_ctx) = build_test_env();

    let my_store_id = fsm.peer.peer.get_store_id();
    let my_peer_id = fsm.peer.peer.get_id();

    let mut base_req = RaftCmdRequest::default();
    base_req.mut_header().mut_peer().set_id(my_peer_id);
    base_req.mut_header().mut_peer().set_store_id(my_store_id);

    // ─────────────────────────────────────────────────────────────────────────────
    // SCENARIO 1: Store ID mismatch
    // ─────────────────────────────────────────────────────────────────────────────
    {
        let mut req = base_req.clone();
        req.mut_header().mut_peer().set_store_id(99999);

        let mut handler = PeerMsgHandler::new(&mut fsm, &mut raft_ctx);
        assert!(matches!(
            handler.pre_propose_raft_command(&req).unwrap_err(),
            Error::StoreNotMatch {
                to_store_id: 99999,
                my_store_id: 1,
            }
        ));
        // Validate metrics
        assert_eq!(
            handler
                .ctx
                .raft_metrics
                .invalid_proposal
                .mismatch_store_id
                .get(),
            1
        );
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // SCENARIO 2: Not leader
    // ─────────────────────────────────────────────────────────────────────────────
    {
        let mut req = base_req.clone();
        let mut r = Request::default();
        r.set_cmd_type(CmdType::Get);
        req.mut_requests().push(r);
        fsm.peer.raft_group.raft.r.state = raft::StateRole::Follower;

        let mut handler = PeerMsgHandler::new(&mut fsm, &mut raft_ctx);
        let err = handler.pre_propose_raft_command(&req).unwrap_err();
        assert!(matches!(err, Error::NotLeader(_, _)));
        // Validate metrics
        assert_eq!(
            handler.ctx.raft_metrics.invalid_proposal.not_leader.get(),
            1
        );

        // Make the peer a leader for the tests below
        fsm.peer.raft_group.raft.r.state = raft::StateRole::Leader;
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // SCENARIO 3: Peer ID mismatch
    // ─────────────────────────────────────────────────────────────────────────────
    {
        let mut req = base_req.clone();
        req.mut_header().mut_peer().set_id(99999); // mismatch peer_id

        let mut handler = PeerMsgHandler::new(&mut fsm, &mut raft_ctx);
        let err_str = handler
            .pre_propose_raft_command(&req)
            .unwrap_err()
            .to_string();
        assert!(
            err_str.contains("mismatch peer id"),
            "unexpected error: {:?}",
            err_str
        );
        // Validate metric
        assert_eq!(
            handler
                .ctx
                .raft_metrics
                .invalid_proposal
                .mismatch_peer_id
                .get(),
            1
        );
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // SCENARIO 4: Region not initialized
    // ─────────────────────────────────────────────────────────────────────────────
    {
        // Remove shard_meta => region is considered not initialized
        let shard_meta = fsm.peer.mut_store().shard_meta.take();

        let mut handler = PeerMsgHandler::new(&mut fsm, &mut raft_ctx);
        let err = handler.pre_propose_raft_command(&base_req).unwrap_err();
        assert!(
            matches!(err, Error::RegionNotInitialized(1)),
            "Expected RegionNotInitialized(1)"
        );
        // Validate metric
        assert_eq!(
            handler
                .ctx
                .raft_metrics
                .invalid_proposal
                .region_not_initialized
                .get(),
            1
        );

        // Restore shard_meta for the next scenario
        fsm.peer.mut_store().shard_meta = shard_meta;
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // SCENARIO 5: Snapshot applying => "peer is applying snapshot"
    // ─────────────────────────────────────────────────────────────────────────────
    {
        fsm.peer.mut_store().snap_state = SnapState::Applying;
        assert!(fsm.peer.mut_store().is_applying_snapshot());

        let mut handler = PeerMsgHandler::new(&mut fsm, &mut raft_ctx);
        let err_str = handler
            .pre_propose_raft_command(&base_req)
            .unwrap_err()
            .to_string();
        assert!(
            err_str.contains("peer is applying snapshot"),
            "unexpected error: {:?}",
            err_str
        );

        // Switch back to snap_state=Relax for next scenario
        fsm.peer.mut_store().snap_state = SnapState::Relax;
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // SCENARIO 6: Stale command => term mismatch
    // ─────────────────────────────────────────────────────────────────────────────
    {
        // The request header says term=1, but peer's actual term=3 => stale
        fsm.peer.raft_group.raft.term = 3;
        let mut req = base_req.clone();
        req.mut_header().set_term(1);

        let mut handler = PeerMsgHandler::new(&mut fsm, &mut raft_ctx);
        let err = handler.pre_propose_raft_command(&req).unwrap_err();
        assert!(matches!(err, Error::StaleCommand), "Expected StaleCommand");
        assert_eq!(
            handler
                .ctx
                .raft_metrics
                .invalid_proposal
                .stale_command
                .get(),
            1
        );
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // SCENARIO 7: Missing epoch => "missing epoch!"
    // ─────────────────────────────────────────────────────────────────────────────
    {
        // Now request has term=5, but missing epoch => "missing epoch!"
        fsm.peer.raft_group.raft.term = 5;
        let mut req = base_req.clone();
        req.mut_header().set_term(5);

        let mut handler = PeerMsgHandler::new(&mut fsm, &mut raft_ctx);
        let err_str = handler
            .pre_propose_raft_command(&req)
            .unwrap_err()
            .to_string();
        assert!(
            err_str.contains("missing epoch!"),
            "unexpected error: {:?}",
            err_str
        );
    }
}
