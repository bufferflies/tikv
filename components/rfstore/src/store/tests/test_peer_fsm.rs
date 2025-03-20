// Copyright 2025 TiKV Project Authors. Licensed under Apache-2.0.

use std::time::Duration;

use kvenginepb::TxnFileRef;
use kvproto::{
    metapb,
    pdpb::CheckPolicy,
    raft_cmdpb::{CmdType, RaftCmdRequest, Request},
    raft_serverpb::RaftMessage,
};

use super::*;
use crate::{
    store::{initial_region, *},
    Error,
};

#[derive(Default)]
struct TestConfig {
    pd_scheduler: Option<tikv_util::worker::Scheduler<PdTask>>,
}

/// Sets up a minimal test environment with a Peer FSM and associated Raft
/// context.
fn build_test_env(cfg: TestConfig) -> (PeerFsm, RaftContext, TempDir) {
    let (engines, tmp_dir) = new_test_engines();
    let mut region = initial_region(1, 1, 1);
    region.mut_peers().push(new_peer(2, 2));
    region.mut_peers().push(new_peer(3, 3));

    // Build the peer FSM
    let fsm = new_test_peer_fsm(engines.clone(), &region).expect("Failed to create test PeerFsm");
    let raft_ctx = new_test_raft_ctx(engines, cfg.pd_scheduler);
    (fsm, raft_ctx, tmp_dir)
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
    let (mut fsm, mut raft_ctx, _tmp_dir) = build_test_env(TestConfig::default());

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
    let (mut fsm, mut raft_ctx, _tmp_dir) = build_test_env(TestConfig::default());

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
    let (mut fsm, mut raft_ctx, _tmp_dir) = build_test_env(TestConfig::default());

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

#[test]
fn test_propose_raft_command() {
    test_util::init_log_for_test();
    let (mut fsm, mut raft_ctx, _tmp_dir) = build_test_env(TestConfig::default());

    // Test that a Raft command proposal will fail if the peer is pending removal.
    fsm.peer.pending_remove = true;
    let mut handler = PeerMsgHandler::new(&mut fsm, &mut raft_ctx);
    let cb = Callback::write(Box::new(|resp| {
        assert!(
            resp.response
                .get_header()
                .get_error()
                .has_region_not_found()
        );
    }));
    handler.propose_raft_command(RaftCmdRequest::default(), cb, None);
}

// Waits for the PD task to be processed.
fn wait_for_task_completion(pd_scheduler: &tikv_util::worker::Scheduler<PdTask>) {
    test_util::eventually(
        Duration::from_millis(100),
        Duration::from_millis(1000),
        || !pd_scheduler.is_busy(),
    );
}

// A test PD runner that sends a signal when a task is processed.
struct TestPdRunner {
    sender: tikv_util::mpsc::Sender<()>,
}

impl tikv_util::worker::Runnable for TestPdRunner {
    type Task = PdTask;

    fn run(&mut self, _task: Self::Task) {
        self.sender.send(()).unwrap()
    }
}

#[test]
fn test_half_split_region() {
    test_util::init_log_for_test();

    let mut pd_worker = LazyWorker::new("test-pd-worker");
    let pd_scheduler = pd_worker.scheduler();
    let (mut fsm, mut raft_ctx, _tmp_dir) = build_test_env(TestConfig {
        pd_scheduler: Some(pd_scheduler.clone()),
    });

    let (tx, rx) = tikv_util::mpsc::unbounded();
    assert!(pd_worker.start(TestPdRunner { sender: tx }));

    let region = fsm.peer.region().clone();

    // ─────────────────────────────────────────────────────────────────────────────
    // SCENARIO: Not leader => no task scheduled
    // ─────────────────────────────────────────────────────────────────────────────
    {
        fsm.peer.raft_group.raft.state = raft::StateRole::Follower;

        let mut handler = PeerMsgHandler::new(&mut fsm, &mut raft_ctx);
        handler.on_schedule_half_split_region(
            region.get_region_epoch().clone(),
            CheckPolicy::Usekey,
            "pd",
        );
        wait_for_task_completion(&pd_scheduler);
        assert_eq!(rx.len(), 0, "expected no task scheduled since not leader");
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // SCENARIO: Stale epoch => no task scheduled
    // ─────────────────────────────────────────────────────────────────────────────
    {
        // Make the peer a leader first.
        fsm.peer.raft_group.raft.state = raft::StateRole::Leader;
        // Assign a higher region version to simulate a stale command.
        let mut r = region.clone();
        r.mut_region_epoch().set_version(99);
        fsm.peer.mut_store().set_region(r);

        let mut handler = PeerMsgHandler::new(&mut fsm, &mut raft_ctx);
        handler.on_schedule_half_split_region(
            region.get_region_epoch().clone(),
            CheckPolicy::Usekey,
            "pd",
        );
        wait_for_task_completion(&pd_scheduler);
        assert_eq!(rx.len(), 0, "expected no task scheduled due to stale epoch");
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // SCENARIO: No buckets => split key not found
    // ─────────────────────────────────────────────────────────────────────────────
    {
        fsm.peer.buckets = None;
        fsm.peer.mut_store().set_region(region.clone());

        let mut handler = PeerMsgHandler::new(&mut fsm, &mut raft_ctx);
        handler.on_schedule_half_split_region(
            region.get_region_epoch().clone(),
            CheckPolicy::Usekey,
            "pd",
        );
        wait_for_task_completion(&pd_scheduler);
        assert_eq!(
            rx.len(),
            0,
            "expected no tasks scheduled due to empty bucket"
        );
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // SCENARIO: Buckets with length > 2 => split task scheduled
    // ─────────────────────────────────────────────────────────────────────────────
    {
        let mut bucket_stats = pd_client::BucketStat::default();
        let mut meta = pd_client::BucketMeta::default();
        meta.keys = vec![
            b"k0".to_vec(),
            b"k1".to_vec(),
            b"k2".to_vec(),
            b"k3".to_vec(),
        ];
        bucket_stats.meta = Arc::new(meta);

        fsm.peer.buckets = Some(bucket_stats);

        let mut handler = PeerMsgHandler::new(&mut fsm, &mut raft_ctx);
        handler.on_schedule_half_split_region(
            region.get_region_epoch().clone(),
            CheckPolicy::Usekey,
            "pd",
        );

        // A split key is found,
        wait_for_task_completion(&pd_scheduler);
        assert_eq!(rx.len(), 1, "expected a scheduled task");
    }
}

#[test]
fn test_validate_split_region() {
    test_util::init_log_for_test();
    let (mut fsm, mut raft_ctx, _tmp_dir) = build_test_env(TestConfig::default());
    let valid_epoch = fsm.peer.region().get_region_epoch().clone();
    let valid_split_keys = vec![tikv_util::codec::bytes::encode_bytes(b"valid_key")];

    // ─────────────────────────────────────────────────────────────────────────────
    // SCENARIO: invalid split_keys
    // ─────────────────────────────────────────────────────────────────────────────
    {
        let cases = vec![
            // (split_keys, expected_err_string)
            (vec![], "no split key is specified"),
            (vec![vec![]], "split key should not be empty"),
            (vec![b"\xFF\x00".to_vec()], "split key decode failed"),
        ];

        let mut handler = PeerMsgHandler::new(&mut fsm, &mut raft_ctx);
        for (invalid_split_keys, expected_msg) in cases {
            let err_str = handler
                .validate_split_region(&valid_epoch, &invalid_split_keys)
                .unwrap_err()
                .to_string();
            assert!(
                err_str.contains(expected_msg),
                "unexpected error: {}, expected_msg={}",
                err_str,
                expected_msg
            );
        }
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // SCENARIO: Not leader => returns `Error::NotLeader`
    // ─────────────────────────────────────────────────────────────────────────────
    {
        let mut handler = PeerMsgHandler::new(&mut fsm, &mut raft_ctx);
        let res = handler.validate_split_region(&valid_epoch, &valid_split_keys);
        assert!(
            matches!(res, Err(Error::NotLeader(..))),
            "unexpected result: {:?}",
            res
        );
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // SCENARIO: Epoch mismatch => returns `Error::EpochNotMatch`
    // ─────────────────────────────────────────────────────────────────────────────
    {
        fsm.peer.raft_group.raft.state = raft::StateRole::Leader;
        let mut invalid_epoch = valid_epoch.clone();
        invalid_epoch.set_version(fsm.peer.region().get_region_epoch().get_version() + 1);
        let mut handler = PeerMsgHandler::new(&mut fsm, &mut raft_ctx);
        let res = handler.validate_split_region(&invalid_epoch, &valid_split_keys);
        assert!(
            matches!(res, Err(Error::EpochNotMatch(..))),
            "unexpected result: {:?}",
            res
        );
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // SCENARIO: Transaction file lock => returns `Error::KeyErrors`
    // ─────────────────────────────────────────────────────────────────────────────
    {
        let mut new_meta = kvengine::ShardMeta::default();
        let mut txn_file = TxnFileRef::new();
        txn_file.set_lock_val_prefix(
            txn_types::Lock::new(
                txn_types::LockType::Pessimistic,
                b"primary".to_vec(),
                txn_types::TimeStamp::zero(),
                0,
                None,
                txn_types::TimeStamp::zero(),
                0,
                txn_types::TimeStamp::zero(),
            )
            .to_bytes(),
        );
        new_meta.merge_txn_file_ref(&txn_file, 1);
        fsm.peer.mut_store().shard_meta = Some(new_meta);

        let mut handler = PeerMsgHandler::new(&mut fsm, &mut raft_ctx);
        let res = handler.validate_split_region(&valid_epoch, &valid_split_keys);
        assert!(
            matches!(res, Err(Error::KeyErrors(_))),
            "unexpected result: {:?}",
            res
        );
    }
}

use rstest::rstest;

fn callback_expect_no_error() -> Callback {
    Callback::write(Box::new(|resp| {
        let header = resp.response.get_header();
        assert!(
            !header.has_error(),
            "expected no error, got: {:?}",
            header.get_error()
        );
    }))
}

fn callback_expect_epoch_not_match() -> Callback {
    Callback::write(Box::new(|resp| {
        let header = resp.response.get_header();
        assert!(
            header.get_error().has_epoch_not_match(),
            "expected EpochNotMatch error, got: {:?}",
            header.get_error()
        );
    }))
}

#[rstest]
#[case::not_leader(raft::StateRole::Follower, false, false, callback_expect_no_error)]
#[case::epoch_mismatch(raft::StateRole::Leader, true, false, callback_expect_epoch_not_match)]
#[case::success(raft::StateRole::Leader, false, true, callback_expect_no_error)]
fn test_on_delete_prefix(
    #[case] role: raft::StateRole,
    #[case] version_mismatch: bool,
    #[case] expected_proposal: bool,
    #[case] cb_factory: fn() -> Callback,
) {
    test_util::init_log_for_test();

    let (mut fsm, mut raft_ctx, _tmp_dir) = build_test_env(TestConfig::default());

    let mut region_version = fsm.peer.region().get_region_epoch().get_version();
    if version_mismatch {
        region_version += 1
    };

    // Set the peer's raft role
    fsm.peer.raft_group.raft.state = role;

    // Record the propose count before
    let old_propose_count = raft_ctx.raft_metrics.propose.all.get();

    // Call on_delete_prefix.
    let mut handler = PeerMsgHandler::new(&mut fsm, &mut raft_ctx);
    handler.on_delete_prefix(region_version, b"test_prefix".to_vec(), cb_factory());

    // Check if a new proposal was made
    let proposed = raft_ctx.raft_metrics.propose.all.get() > old_propose_count;
    assert_eq!(
        proposed, expected_proposal,
        "role={:?}, version_mismatch={} => expected_proposal={}, got {}",
        role, version_mismatch, expected_proposal, proposed
    );
}

#[rstest]
#[case::not_leader(raft::StateRole::Follower, false, false, callback_expect_no_error)]
#[case::version_mismatch(raft::StateRole::Leader, true, false, callback_expect_no_error)]
#[case::success(raft::StateRole::Leader, false, true, callback_expect_no_error)]
fn test_on_truncate_ts(
    #[case] role: raft::StateRole,
    #[case] version_mismatch: bool,
    #[case] expected_proposal: bool,
    #[case] cb_factory: fn() -> Callback,
) {
    test_util::init_log_for_test();
    let (mut fsm, mut raft_ctx, _tmp_dir) = build_test_env(TestConfig::default());
    let mut shard_ver = fsm.peer.region().get_region_epoch().get_version();
    if version_mismatch {
        shard_ver += 1
    }
    let truncate_ts = 1234;

    // Set the peer's raft role
    fsm.peer.raft_group.raft.state = role;

    // Record the propose count before
    let old_propose_count = raft_ctx.raft_metrics.propose.all.get();

    // Call on_truncate_ts
    let mut handler = PeerMsgHandler::new(&mut fsm, &mut raft_ctx);
    handler.on_truncate_ts(truncate_ts, shard_ver, cb_factory());

    // Check if a new proposal was made
    let proposed = raft_ctx.raft_metrics.propose.all.get() > old_propose_count;
    assert_eq!(
        proposed, expected_proposal,
        "role={:?}, version_mismatch={} => expected_proposal={}, but got {}",
        role, version_mismatch, expected_proposal, proposed
    );
}

#[rstest]
#[case::not_leader(raft::StateRole::Follower, true, false, callback_expect_no_error)]
#[case::leader_major_compact_true(raft::StateRole::Leader, true, true, callback_expect_no_error)]
#[case::leader_major_compact_false(raft::StateRole::Leader, false, true, callback_expect_no_error)]
fn test_on_manual_major_compact(
    #[case] role: raft::StateRole,
    #[case] major_compact: bool,
    #[case] expected_proposal: bool,
    #[case] cb_factory: fn() -> Callback,
) {
    test_util::init_log_for_test();
    let (mut fsm, mut raft_ctx, _tmp_dir) = build_test_env(TestConfig::default());

    // Set the peer's raft role
    fsm.peer.raft_group.raft.state = role;

    // Record the propose count before
    let old_propose_count = raft_ctx.raft_metrics.propose.all.get();

    // Call on_manual_major_compact
    let mut handler = PeerMsgHandler::new(&mut fsm, &mut raft_ctx);
    handler.on_manual_major_compact(major_compact, cb_factory());

    // Check if a new proposal was made
    let new_propose_count = raft_ctx.raft_metrics.propose.all.get();
    let proposed = new_propose_count > old_propose_count;
    assert_eq!(
        proposed, expected_proposal,
        "role={:?}, major_compact={} => expected_proposal={}, got {}",
        role, major_compact, expected_proposal, proposed
    );
}
