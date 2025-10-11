// Copyright 2019 TiKV Project Authors. Licensed under Apache-2.0.

use std::{sync::*, time::Duration};

use cdc::{Task, Validate};
use futures::{executor::block_on, SinkExt};
use grpcio::WriteFlags;
use kvproto::{cdcpb::*, kvrpcpb::*};
use pd_client::PdClient;
use tikv::config::TikvConfig;
use tikv_util::config::ReadableSize;

use crate::{new_event_feed, TestSuite};

fn cfg_for_mem_quota(cfg: &mut TikvConfig, memory_quota: usize) {
    TestSuite::cfg_for_cdc(cfg);
    cfg.cdc.sink_memory_quota = ReadableSize(memory_quota as u64);
}

fn prewite_commit(suite: &mut TestSuite, region_id: u64, k: Vec<u8>, v: Vec<u8>) {
    // Prewrite
    let start_ts = block_on(suite.cluster.pd_client().get_tso()).unwrap();
    let mut mutation = Mutation::default();
    mutation.set_op(Op::Put);
    mutation.key = k.clone();
    mutation.value = v;
    suite.must_kv_prewrite(region_id, vec![mutation], k.clone(), start_ts);
    // Commit
    let commit_ts = block_on(suite.cluster.pd_client().get_tso()).unwrap();
    suite.must_kv_commit(region_id, vec![k.clone()], start_ts, commit_ts);
}

#[test]
fn test_resolver_track_lock_memory_quota_exceeded() {
    let memory_quota = 1024usize;
    let region_id = 1001;
    let mut suite = TestSuite::with_extra_builder_fun(1, |builder| {
        builder.cfg_fun(move |_, cfg| {
            cfg_for_mem_quota(cfg, memory_quota);
        })
    });

    // Let CdcEvent size be 0 to effectively disable memory quota for CdcEvent.
    fail::cfg("cdc_event_size", "return(0)").unwrap();

    let req = suite.new_changedata_request(region_id);
    let (mut req_tx, _event_feed_wrap, receive_event) =
        new_event_feed(suite.get_region_cdc_client(region_id));

    block_on(req_tx.send((req, WriteFlags::default()))).unwrap();

    let event = receive_event(false);
    event.events.into_iter().for_each(|e| {
        match e.event.unwrap() {
            // Even if there is no write,
            // it should always outputs an Initialized event.
            Event_oneof_event::Entries(es) => {
                assert!(es.entries.len() == 1, "{:?}", es);
                let e = &es.entries[0];
                assert_eq!(e.get_type(), EventLogType::Initialized, "{:?}", es);
            }
            other => panic!("unknown event {:?}", other),
        }
    });

    // Client must receive messages when there is no congest error.
    let key_size = memory_quota / 2;
    let (mut k, v) = (vec![1; key_size], vec![5]);
    k[0] = b'x';
    prewite_commit(&mut suite, region_id, k, v);

    let mut events = receive_event(false).events.to_vec();
    assert_eq!(events.len(), 1, "{:?}", events);
    match events.pop().unwrap().event.unwrap() {
        Event_oneof_event::Entries(entries) => {
            assert_eq!(entries.entries.len(), 1);
            assert_eq!(entries.entries[0].get_type(), EventLogType::Committed);
        }
        other => panic!("unknown event {:?}", other),
    }

    // Trigger congest error.
    let key_size = memory_quota * 2;
    let (mut k, v) = (vec![2; key_size], vec![5]);
    k[0] = b'x';
    prewite_commit(&mut suite, region_id, k, v);
    let mut events = receive_event(false).events.to_vec();
    assert_eq!(events.len(), 1, "{:?}", events);
    match events.pop().unwrap().event.unwrap() {
        Event_oneof_event::Error(e) => {
            // Unknown errors are translated into region_not_found.
            assert!(e.has_congested(), "{:?}", e);
        }
        other => panic!("unknown event {:?}", other),
    }

    // The delegate must be removed.
    let scheduler = suite.endpoints.values().next().unwrap().clone();
    let (tx, rx) = mpsc::channel();
    scheduler
        .schedule(Task::Validate(Validate::Region(
            region_id,
            Box::new(move |delegate| {
                tx.send(delegate.is_none()).unwrap();
            }),
        )))
        .unwrap();

    assert!(
        rx.recv_timeout(Duration::from_millis(1000)).unwrap(),
        "find unexpected delegate"
    );

    suite.stop();
}

#[test]
fn test_pending_on_region_ready_memory_quota_exceeded() {
    let memory_quota = 1024usize;
    let region_id = 1001;
    let mut suite = TestSuite::with_extra_builder_fun(1, |builder| {
        builder.cfg_fun(move |_, cfg| {
            cfg_for_mem_quota(cfg, memory_quota);
        })
    });

    // Let CdcEvent size be 0 to effectively disable memory quota for CdcEvent.
    fail::cfg("cdc_event_size", "return(0)").unwrap();

    // Trigger memory quota exceeded error.
    fail::cfg("cdc_finish_scan_locks_memory_quota_exceed", "return").unwrap();
    let req = suite.new_changedata_request(region_id);
    let (mut req_tx, _event_feed_wrap, receive_event) =
        new_event_feed(suite.get_region_cdc_client(region_id));
    block_on(req_tx.send((req, WriteFlags::default()))).unwrap();

    // MemoryQuotaExceeded error is triggered.
    let mut events = receive_event(false).events.to_vec();
    assert_eq!(events.len(), 1, "{:?}", events);
    match events.pop().unwrap().event.unwrap() {
        Event_oneof_event::Error(e) => {
            // Unknown errors are translated into region_not_found.
            assert!(e.has_congested(), "{:?}", e);
        }
        other => panic!("unknown event {:?}", other),
    }

    // The delegate must be removed.
    let scheduler = suite.endpoints.values().next().unwrap().clone();
    let (tx, rx) = mpsc::channel();
    scheduler
        .schedule(Task::Validate(Validate::Region(
            region_id,
            Box::new(move |delegate| {
                tx.send(delegate.is_none()).unwrap();
            }),
        )))
        .unwrap();

    assert!(
        rx.recv_timeout(Duration::from_millis(1000)).unwrap(),
        "find unexpected delegate"
    );

    fail::remove("cdc_event_size");
    fail::remove("cdc_finish_scan_locks_memory_quota_exceed");
    suite.stop();
}

#[test]
fn test_pending_push_lock_memory_quota_exceeded() {
    let memory_quota = 1024usize;
    let region_id = 1001;
    let mut suite = TestSuite::with_extra_builder_fun(1, |builder| {
        builder.cfg_fun(move |_, cfg| {
            cfg_for_mem_quota(cfg, memory_quota);
        })
    });

    // Let CdcEvent size be 0 to effectively disable memory quota for CdcEvent.
    fail::cfg("cdc_event_size", "return(0)").unwrap();

    // Pause scan so that no region can be initialized, and all locks will be
    // put in pending locks.
    fail::cfg("cdc_incremental_scan_start", "pause").unwrap();

    let req = suite.new_changedata_request(region_id);
    let (mut req_tx, _event_feed_wrap, receive_event) =
        new_event_feed(suite.get_region_cdc_client(region_id));
    block_on(req_tx.send((req, WriteFlags::default()))).unwrap();

    // Trigger congest error.
    let key_size = memory_quota * 2;
    let (mut k, v) = (vec![1; key_size], vec![5]);
    k[0] = b'x';
    prewite_commit(&mut suite, region_id, k, v);
    let mut events = receive_event(false).events.to_vec();
    assert_eq!(events.len(), 1, "{:?}", events);
    match events.pop().unwrap().event.unwrap() {
        Event_oneof_event::Error(e) => {
            // Unknown errors are translated into region_not_found.
            assert!(e.has_congested(), "{:?}", e);
        }
        other => panic!("unknown event {:?}", other),
    }

    // The delegate must be removed.
    let scheduler = suite.endpoints.values().next().unwrap().clone();
    let (tx, rx) = mpsc::channel();
    scheduler
        .schedule(Task::Validate(Validate::Region(
            region_id,
            Box::new(move |delegate| {
                tx.send(delegate.is_none()).unwrap();
            }),
        )))
        .unwrap();

    assert!(
        rx.recv_timeout(Duration::from_millis(1000)).unwrap(),
        "find unexpected delegate"
    );

    fail::remove("cdc_event_size");
    fail::remove("cdc_incremental_scan_start");
    suite.stop();
}

#[test]
fn test_scan_lock_memory_quota_exceeded() {
    let memory_quota = 1024usize;
    let region_id = 1001;
    let mut suite = TestSuite::with_extra_builder_fun(1, |builder| {
        builder.cfg_fun(move |_, cfg| {
            cfg_for_mem_quota(cfg, memory_quota);
        })
    });

    // Let CdcEvent size be 0 to effectively disable memory quota for CdcEvent.
    fail::cfg("cdc_event_size", "return(0)").unwrap();

    // Put a lock that exceeds memory quota.
    let key_size = memory_quota * 2;
    let (mut k, v) = (vec![1; key_size], vec![5]);
    k[0] = b'x';
    let start_ts = block_on(suite.cluster.pd_client().get_tso()).unwrap();
    let mut mutation = Mutation::default();
    mutation.set_op(Op::Put);
    mutation.key = k.clone();
    mutation.value = v;
    suite.must_kv_prewrite(region_id, vec![mutation], k, start_ts);
    // prewite_commit(&mut suite, region_id, k, v);

    // No region can be initialized.
    let req = suite.new_changedata_request(region_id);
    let (mut req_tx, _event_feed_wrap, receive_event) =
        new_event_feed(suite.get_region_cdc_client(region_id));
    block_on(req_tx.send((req, WriteFlags::default()))).unwrap();
    let mut events = receive_event(false).events.to_vec();
    assert_eq!(events.len(), 1, "{:?}", events);
    match events.pop().unwrap().event.unwrap() {
        Event_oneof_event::Error(e) => {
            // Unknown errors are translated into region_not_found.
            assert!(e.has_congested(), "{:?}", e);
        }
        other => panic!("unknown event {:?}", other),
    }
    let scheduler = suite.endpoints.values().next().unwrap().clone();
    let (tx, rx) = mpsc::channel();
    scheduler
        .schedule(Task::Validate(Validate::Region(
            region_id,
            Box::new(move |delegate| {
                tx.send(delegate.is_none()).unwrap();
            }),
        )))
        .unwrap();

    assert!(
        rx.recv_timeout(Duration::from_millis(1000)).unwrap(),
        "find unexpected delegate"
    );

    fail::remove("cdc_event_size");
    suite.stop();
}
