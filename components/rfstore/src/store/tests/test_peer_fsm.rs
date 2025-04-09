// Copyright 2025 TiKV Project Authors. Licensed under Apache-2.0.

use std::time::Duration;

use kvengine::table::{
    columnar::{Schema, SchemaBuf, SchemaFile},
    file::InMemFile,
};
use kvenginepb::TxnFileRef;
use kvproto::{
    metapb,
    pdpb::CheckPolicy,
    raft_cmdpb::{
        AdminCmdType, AdminRequest, CmdType, CommitMergeRequest, PrepareMergeRequest,
        RaftCmdRequest, Request,
    },
    raft_serverpb::{PeerState, RaftMessage, RegionLocalState},
};
use protobuf::Message;
use raft::StateRole::{Follower, Leader};

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
    append_peer(&mut region, 2, 2);
    append_peer(&mut region, 3, 3);

    // Build the peer FSM
    let fsm = new_test_peer_fsm(engines.clone(), &region).expect("Failed to create test PeerFsm");
    let raft_ctx = new_test_raft_ctx(engines, cfg.pd_scheduler);
    (fsm, raft_ctx, tmp_dir)
}

fn set_region_epoch(region: &mut metapb::Region, version: u64, conf_ver: u64) {
    let mut epoch = metapb::RegionEpoch::new();
    epoch.set_version(version);
    epoch.set_conf_ver(conf_ver);
    region.set_region_epoch(epoch);
}

fn append_peer(region: &mut metapb::Region, store_id: u64, peer_id: u64) {
    region.mut_peers().push(new_peer(store_id, peer_id));
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

    let mut msg = raft::eraftpb::Message::default();
    msg.set_term(5);
    raft_msg.set_message(msg);
    assert!(MsgDebug(&raft_msg).to_string().contains("term: 5"));
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
        fsm.peer.raft_group.raft.r.state = Follower;

        let mut handler = PeerMsgHandler::new(&mut fsm, &mut raft_ctx);
        let err = handler.pre_propose_raft_command(&req).unwrap_err();
        assert!(matches!(err, Error::NotLeader(_, _)));
        // Validate metrics
        assert_eq!(
            handler.ctx.raft_metrics.invalid_proposal.not_leader.get(),
            1
        );

        // Make the peer a leader for the tests below
        fsm.peer.raft_group.raft.r.state = Leader;
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
        fsm.peer.raft_group.raft.state = Follower;

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
        fsm.peer.raft_group.raft.state = Leader;
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
        fsm.peer.raft_group.raft.state = Leader;
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

use kvengine::table::columnar::build_schema_file;
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

fn callback_expect_not_leader() -> Callback {
    Callback::write(Box::new(|resp| {
        let header = resp.response.get_header();
        assert!(
            header.get_error().has_not_leader(),
            "expected NotLeader error, got: {:?}",
            header.get_error()
        );
    }))
}

fn callback_expect_region_not_initialized() -> Callback {
    Callback::write(Box::new(|resp| {
        let header = resp.response.get_header();
        assert!(
            header.get_error().has_region_not_initialized(),
            "expected RegionNotInitialized error, got: {:?}",
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

fn callback_expect_err_msg(expected_msg: &str) -> Callback {
    let expected_msg = expected_msg.to_string();
    Callback::write(Box::new(move |resp| {
        let header = resp.response.get_header();
        let err_msg = header.get_error().get_message();
        assert!(
            err_msg.contains(&expected_msg),
            "Expected error message to contain '{}', got: {}",
            expected_msg,
            err_msg
        );
    }))
}

#[rstest]
#[case::not_leader(Follower, false, false, callback_expect_no_error)]
#[case::epoch_mismatch(Leader, true, false, callback_expect_epoch_not_match)]
#[case::success(Leader, false, true, callback_expect_no_error)]
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
#[case::not_leader(Follower, true, false, callback_expect_no_error)]
#[case::leader_major_compact_true(Leader, true, true, callback_expect_no_error)]
#[case::leader_major_compact_false(Leader, false, true, callback_expect_no_error)]
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

#[rstest]
#[case::not_leader(Follower, false, false, false)]
#[case::mismatch_restore_ver(Leader, true, false, false)]
#[case::stale_file_ver(Leader, false, true, false)]
#[case::success(Leader, false, false, true)]
fn test_on_update_schema_file(
    #[case] role: raft::StateRole,
    #[case] mismatch_restore_ver: bool,
    #[case] stale_file_ver: bool,
    #[case] expected_proposal: bool,
) {
    test_util::init_log_for_test();
    let (mut fsm, mut raft_ctx, _tmp_dir) = build_test_env(TestConfig::default());
    let schema_file_ver = 10;
    let schema_restore_ver = 100;
    {
        let mut shard_meta = kvengine::ShardMeta::default();
        shard_meta.schema.schema_restore_ver = schema_restore_ver;
        shard_meta.schema.schema_file_ver = schema_file_ver;
        shard_meta.schema.schema_file_id = 99;
        shard_meta.range.outer_start = bytes::Bytes::from(b"k00000".to_vec());
        shard_meta.range.outer_end = bytes::Bytes::from(b"k99999".to_vec());
        assert!(shard_meta.initial_flushed());
        fsm.peer.mut_store().shard_meta = Some(shard_meta);
    }

    let file_restore_ver = if mismatch_restore_ver {
        schema_restore_ver + 1
    } else {
        schema_restore_ver
    };
    let file_version = if stale_file_ver {
        schema_file_ver - 1
    } else {
        schema_file_ver + 1
    };
    let data = build_schema_file(
        0,
        file_version,
        vec![Schema::new(SchemaBuf::default())],
        file_restore_ver,
    );

    fsm.peer.raft_group.raft.state = role;
    let old_propose_count = raft_ctx.raft_metrics.propose.all.get();
    let mut handler = PeerMsgHandler::new(&mut fsm, &mut raft_ctx);
    handler
        .on_update_schema_file(SchemaFile::open(Arc::new(InMemFile::new(0, data.into()))).unwrap());

    let new_propose_count = raft_ctx.raft_metrics.propose.all.get();
    let proposed = new_propose_count > old_propose_count;
    assert_eq!(
        proposed, expected_proposal,
        "role={:?}, mismatch_restore_ver={}, stale_file_ver={} => expected_proposal={}, got {}",
        role, mismatch_restore_ver, stale_file_ver, expected_proposal, proposed
    );
}

#[rstest]
#[case::not_leader(Follower, false, false, callback_expect_not_leader)]
#[case::leader_not_applied(Leader, true, false, callback_expect_not_leader)]
#[case::epoch_mismatch(Leader, false, true, callback_expect_epoch_not_match)]
#[case::leader_success(Leader, false, false, callback_expect_no_error)]
fn test_on_check_leader(
    #[case] role: raft::StateRole,
    #[case] not_applied_to_current_term: bool,
    #[case] version_mismatch: bool,
    #[case] cb_factory: fn() -> Callback,
) {
    test_util::init_log_for_test();
    let (mut fsm, mut raft_ctx, _tmp_dir) = build_test_env(TestConfig::default());
    let mut shard_ver = fsm.peer.region().get_region_epoch().version;
    if version_mismatch {
        shard_ver += 1;
    }

    if not_applied_to_current_term {
        let mut apply_state = fsm.peer.get_store().apply_state();
        apply_state.applied_index_term = fsm.peer.raft_group.raft.term.saturating_sub(1);
        fsm.peer.mut_store().set_applied_state(apply_state);
    }
    fsm.peer.raft_group.raft.state = role;

    let mut handler = PeerMsgHandler::new(&mut fsm, &mut raft_ctx);
    handler.on_check_leader(shard_ver, cb_factory());
}

#[rstest]
#[case::not_leader(Follower, false, false, callback_expect_not_leader)]
#[case::not_initial_flushed(Leader, true, false, callback_expect_region_not_initialized)]
#[case::epoch_mismatch(Leader, false, true, callback_expect_epoch_not_match)]
#[case::leader_success(Leader, false, false, callback_expect_no_error)]
fn test_on_ingest_files(
    #[case] role: raft::StateRole,
    #[case] not_initial_flushed: bool,
    #[case] shard_version_mismatch: bool,
    #[case] cb_factory: fn() -> Callback,
) {
    test_util::init_log_for_test();
    let (mut fsm, mut raft_ctx, _tmp_dir) = build_test_env(TestConfig::default());

    fsm.peer.raft_group.raft.state = role;

    let mut cs = kvenginepb::ChangeSet::default();
    cs.shard_ver = fsm.peer.region().get_region_epoch().version;
    if shard_version_mismatch {
        cs.shard_ver += 1;
    }

    if not_initial_flushed {
        fsm.peer.mut_store().shard_meta.as_mut().unwrap().parent = Some(Box::default());
        assert!(!fsm.get_peer().get_store().initial_flushed());
    }

    let mut handler = PeerMsgHandler::new(&mut fsm, &mut raft_ctx);
    handler.on_ingest_files(cs, cb_factory());
}

#[rstest]
#[case::invalid_cmd(
    Leader,
    true,
    false,
    false,
    Some("invalid changeset"),
    false,
    callback_expect_no_error
)]
#[case::invalid_snap_range(
    Leader,
    false,
    true,
    false,
    Some("invalid snapshot range"),
    false,
    callback_expect_no_error
)]
#[case::not_leader(Follower, false, false, false, None, false, callback_expect_not_leader)]
#[case::epoch_mismatch(
    Leader,
    false,
    false,
    true,
    None,
    false,
    callback_expect_epoch_not_match
)]
#[case::leader_success(Leader, false, false, false, None, true, callback_expect_no_error)]
fn test_on_restore_shard(
    #[case] role: raft::StateRole,
    #[case] invalid_command: bool,
    #[case] invalid_snap_range: bool,
    #[case] epoch_mismatch: bool,
    #[case] expect_err_msg: Option<&str>,
    #[case] expect_proposal: bool,
    #[case] cb_factory: fn() -> Callback,
) {
    test_util::init_log_for_test();
    let (mut fsm, mut raft_ctx, _tmp_dir) = build_test_env(TestConfig::default());
    fsm.peer.raft_group.raft.state = role;

    let mut cs = kvenginepb::ChangeSet::default();
    cs.shard_ver = fsm.peer.region().get_region_epoch().version;
    if epoch_mismatch {
        cs.shard_ver += 1;
    }

    let cb = if let Some(msg) = expect_err_msg {
        callback_expect_err_msg(msg)
    } else {
        cb_factory()
    };

    let mut region = fsm.peer.region().clone();
    region.set_start_key(tikv_util::codec::bytes::encode_bytes(b"k00000"));
    region.set_end_key(tikv_util::codec::bytes::encode_bytes(b"k99999"));
    fsm.peer.mut_store().set_region(region.clone());

    if !invalid_command {
        let rs = cs.mut_restore_shard();
        if invalid_snap_range {
            // Set wrong outer start/end keys.
            rs.set_outer_start(b"bad_start".to_vec());
            rs.set_outer_end(b"bad_end".to_vec());
        } else {
            rs.set_outer_start(raw_start_key(&region));
            rs.set_outer_end(raw_end_key(&region));
        }
    }

    let old_propose_count = raft_ctx.raft_metrics.propose.all.get();
    let mut handler = PeerMsgHandler::new(&mut fsm, &mut raft_ctx);
    handler.on_restore_shard(cs, cb);
    let proposed = raft_ctx.raft_metrics.propose.all.get() > old_propose_count;
    assert_eq!(
        proposed, expect_proposal,
        "expect_proposal={}, but got {}",
        expect_proposal, proposed
    );
}

#[rstest]
#[case::missing_local_state(None, false)]
#[case::local_epoch_is_newer(Some((PeerState::Normal, 10)), true)]
#[case::local_tombstone(Some((PeerState::Tombstone, 5)), true)]
#[case::local_not_tombstone(Some((PeerState::Normal, 5)), false)]
fn test_is_merge_target_region_stale(
    #[case] local_state_info: Option<(PeerState, u64)>, // peer_state, epoch version
    #[case] expected_is_stale: bool,
) {
    test_util::init_log_for_test();
    let (mut fsm, mut raft_ctx, _tmp_dir) = build_test_env(TestConfig::default());
    let target_peer_id = 1001;
    let store_id = fsm.peer.get_store().store_id;
    raft_ctx.global.store.set_id(store_id);

    // Write the local target peer state into rfengine.
    if let Some((peer_state, ver)) = local_state_info {
        let mut local_state = RegionLocalState::new();
        local_state.set_state(peer_state);

        let mut region = metapb::Region::new();
        set_region_epoch(&mut region, ver, ver);
        append_peer(&mut region, store_id, target_peer_id);
        local_state.set_region(region);

        let encoded = local_state.write_to_bytes().unwrap();
        let key = rfengine::region_state_key(target_peer_id);
        let mut wb = rfengine::WriteBatch::new();
        wb.set_state(target_peer_id, 0, &key, &encoded);
        raft_ctx.global.engines.raft.write(wb).unwrap();
    }

    let mut target_region = metapb::Region::new();
    set_region_epoch(&mut target_region, 5, 5);
    append_peer(&mut target_region, store_id, target_peer_id);

    let handler = PeerMsgHandler::new(&mut fsm, &mut raft_ctx);
    let res = handler
        .is_merge_target_region_stale(&target_region)
        .unwrap();
    assert_eq!(
        res, expected_is_stale,
        "local_state={:?}, expected_is_stale={}",
        local_state_info, expected_is_stale
    );
}

#[rstest]
#[case::mismatch(true)]
#[case::correct(false)]
fn test_on_prepared_txn_file(#[case] mismatch: bool) {
    test_util::init_log_for_test();
    let (mut fsm, mut raft_ctx, _tmp_dir) = build_test_env(TestConfig::default());
    let mut peer_id = fsm.peer.peer.get_id();
    if mismatch {
        peer_id += 1;
    }

    let mut handler = PeerMsgHandler::new(&mut fsm, &mut raft_ctx);
    handler.on_prepared_txn_file(123, peer_id);
    let pushed = !raft_ctx.apply_msgs.msgs.is_empty();
    assert_eq!(pushed, !mismatch);
}

#[rstest]
#[case::prepare_not_sibling(AdminCmdType::PrepareMerge, false)]
#[case::prepare_not_same_stores(AdminCmdType::PrepareMerge, true)]
#[case::commit_not_sibling(AdminCmdType::CommitMerge, false)]
#[case::commit_not_same_stores(AdminCmdType::CommitMerge, true)]
fn test_check_merge_proposal(#[case] cmd_type: AdminCmdType, #[case] is_sibling: bool) {
    test_util::init_log_for_test();
    let (mut fsm, mut raft_ctx, _tmp) = build_test_env(TestConfig::default());

    let mut fsm_region = fsm.peer.region().clone();
    fsm_region.set_start_key(tikv_util::codec::bytes::encode_bytes(b"k00000"));
    fsm_region.set_end_key(tikv_util::codec::bytes::encode_bytes(b"k00001"));
    fsm.peer.mut_store().set_region(fsm_region.clone());

    let mut store_meta = StoreMeta::new(0);
    store_meta.cop_host = Some(CoprocessorHost::default());

    let mut region = metapb::Region::default();
    if is_sibling {
        region.set_start_key(tikv_util::codec::bytes::encode_bytes(b"k00001"));
        region.set_end_key(tikv_util::codec::bytes::encode_bytes(b"k00002"));
    }

    let mut admin_req = AdminRequest::default();
    admin_req.set_cmd_type(cmd_type);
    match cmd_type {
        AdminCmdType::PrepareMerge => {
            let target_region = region.clone();
            let mut prep = PrepareMergeRequest::default();
            prep.set_target(target_region.clone());
            admin_req.set_prepare_merge(prep);
            store_meta.region_map.put(target_region);
        }
        AdminCmdType::CommitMerge => {
            let source_region = region.clone();
            let mut cm = CommitMergeRequest::default();
            cm.set_source(source_region);
            admin_req.set_commit_merge(cm);
        }
        _ => {}
    }

    let mut req = RaftCmdRequest::default();
    req.set_admin_request(admin_req);

    // In this test, an error is always expected, either due to regions not being
    // siblings or a mismatch in store configurations.
    let expect_err_substr = if !is_sibling { "sibling" } else { "match" };
    let mut handler = PeerMsgHandler::new(&mut fsm, &mut raft_ctx);
    let err_str = handler
        .check_merge_proposal(&req, Some(&mut store_meta))
        .unwrap_err()
        .to_string();
    assert!(
        err_str.contains(expect_err_substr),
        "unexpected error: {:?}",
        err_str
    );
}
