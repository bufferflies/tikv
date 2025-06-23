// Copyright 2025 TiKV Project Authors. Licensed under Apache-2.0.

use kvproto::kvrpcpb;
use test_cloud_server::client::ClusterClient;
use txn_types::{LockType, TimeStamp, WriteType};

// Helper to get MVCC info for a key
pub fn get_mvcc_info(client: &mut ClusterClient, key: &[u8]) -> Option<kvrpcpb::MvccInfo> {
    client
        .kv_get_mvcc_by_key(key)
        .expect("kv_get_mvcc_by_key failed")
}

// Helper to assert lock properties
pub fn must_locked_with_properties(
    client: &mut ClusterClient,
    key: &[u8],
    primary_key: &[u8],
    expected_start_ts: impl Into<TimeStamp>,
    expected_lock_type: LockType,
) -> kvrpcpb::MvccLock {
    let expected_lock_type = match expected_lock_type {
        LockType::Put => kvrpcpb::Op::Put,
        LockType::Lock => kvrpcpb::Op::Lock,
        LockType::Delete => kvrpcpb::Op::Del,
        LockType::Pessimistic => kvrpcpb::Op::PessimisticLock,
    };

    let mut mvcc_info = get_mvcc_info(client, key).expect("MVCC info not found for locked key");
    assert!(
        mvcc_info.has_lock(),
        "Key {:?} should be locked but is not",
        key
    );
    let lock_info = mvcc_info.take_lock();
    assert_eq!(
        lock_info.get_primary(),
        primary_key,
        "Lock primary mismatch for key {:?}",
        key
    );
    assert_eq!(
        lock_info.get_start_ts(),
        expected_start_ts.into().into_inner(),
        "Lock start_ts mismatch for key {:?}",
        key
    );
    assert_eq!(
        lock_info.get_type(),
        expected_lock_type,
        "Lock type mismatch for key {:?}",
        key
    );
    lock_info
}

// Helper to assert absence of lock
pub fn must_unlocked(client: &mut ClusterClient, key: &[u8]) {
    match get_mvcc_info(client, key) {
        Some(info) => assert!(
            !info.has_lock(),
            "Key {:?} should be unlocked but has lock: {:?}",
            key,
            info.get_lock()
        ),
        None => { /* Key not found implies unlocked */ }
    }
}

// Helper to assert absence of write record
pub fn must_not_written(client: &mut ClusterClient, key: &[u8], commit_ts: impl Into<TimeStamp>) {
    let commit_ts_val = commit_ts.into().into_inner();
    match get_mvcc_info(client, key) {
        Some(info) => {
            let found = info
                .get_writes()
                .iter()
                .any(|w| w.commit_ts == commit_ts_val);
            assert!(
                !found,
                "Key {:?} should not have write record at commit_ts {}",
                key, commit_ts_val
            );
        }
        None => { /* Key not found implies no write record */ }
    }
}

// Helper to assert write properties
pub fn must_written_with_properties(
    client: &mut ClusterClient,
    key: &[u8],
    expected_start_ts: impl Into<TimeStamp>,
    expected_commit_ts: impl Into<TimeStamp>,
    expected_write_type: WriteType,
) -> kvrpcpb::MvccWrite {
    let expected_write_type = match expected_write_type {
        WriteType::Put => kvrpcpb::Op::Put,
        WriteType::Lock => kvrpcpb::Op::Lock,
        WriteType::Delete => kvrpcpb::Op::Del,
        WriteType::Rollback => kvrpcpb::Op::Rollback,
    };
    let info = get_mvcc_info(client, key).expect("MVCC info not found for written key");
    let commit_ts_val = expected_commit_ts.into().into_inner();
    let mvcc_write = info
        .get_writes()
        .iter()
        .find(|w| w.commit_ts == commit_ts_val)
        .unwrap_or_else(|| {
            panic!(
                "Write record not found for key {:?} at commit_ts {}",
                key, commit_ts_val
            )
        });
    assert_eq!(
        mvcc_write.get_start_ts(),
        expected_start_ts.into().into_inner(),
        "Write start_ts mismatch for key {:?}",
        key
    );
    assert_eq!(
        mvcc_write.get_type(),
        expected_write_type,
        "Write type mismatch for key {:?}",
        key
    );
    mvcc_write.clone()
}
