// Copyright 2025 TiKV Project Authors. Licensed under Apache-2.0.

//! Implementation-Based Write Conflict Detection Test Suite
//!
//! This module tests write conflict detection based on the ACTUAL
//! implementation from the recent commits (a9b416d20, 86c59fceb, e625771a0).
//!
//! ## Conflict Checking Implementation Details:
//!
//! ### 1. Prewrite Operations (prewrite.rs)
//! - **Optimistic transactions**: Uses start_ts for conflict detection
//!   - WRITE_CF: Conflicts if start_ts < commit_ts
//!   - EXTRA_CF: Checks self-rollback (exact start_ts) and newer LOCKs (>
//!     start_ts)
//! - **Pessimistic transactions**: Uses for_update_ts for conflict detection
//!   - WRITE_CF: Conflicts if for_update_ts < commit_ts
//!   - EXTRA_CF: Checks self-rollback and newer LOCKs (> for_update_ts)
//!   - Special case: DoConstraintCheck uses start_ts instead of for_update_ts
//!
//! ### 2. Amend Pessimistic Lock (prewrite.rs::amend_pessimistic_lock)
//! - Called when pessimistic lock is missing during prewrite
//! - EXTRA_CF: Any record with commit_ts >= start_ts means lock was lost
//!   - commit_ts == 0: Transaction was rolled back
//!   - commit_ts > 0: Another transaction acquired the lock
//! - WRITE_CF: Checks if commit_ts >= start_ts (lock was lost)
//!
//! ### 3. Acquire Pessimistic Lock (acquire_pessimistic_lock.rs)
//! - EXTRA_CF: Uses optimized single scan
//!   - Self-rollback: Checks exact start_ts match with commit_ts == 0
//!   - Newer conflicts: Checks for LOCKs with commit_ts > for_update_ts
//! - Can allow lock with conflict if configured (locked_with_conflict_ts)
//!
//! ## Key Implementation Patterns:
//! - CloudReader::find_extra_cf_conflict_record() performs unified scanning
//! - WRITE_CF checking was pre-existing; recent commits added EXTRA_CF checking
//! - Self-rollback always checked first (priority 1) before newer conflicts

use super::util::{PessimisticAction, *};

#[test]
fn test_optimistic_prewrite_conflicts() {
    // Optimistic prewrite uses start_ts for conflict detection
    let cases = vec![
        // WRITE CF
        case!(prewrite!(10), commit!(10, 15), prewrite!(12); fail!()),
        case!(prewrite!(10), commit!(10, 15), prewrite!(15); ok!()),
        case!(prewrite!(10), commit!(10, 15), prewrite!(20); ok!()),
        case!(lock!(10, 15), prewrite!(20); fail!()),
        case!(prewrite!(10).delete(), commit!(10, 15), prewrite!(12); fail!()),
        case!(prewrite!(10).delete(), commit!(10, 15), prewrite!(15); ok!()),
        case!(prewrite!(10).delete(), commit!(10, 15), prewrite!(20); ok!()),
        // EXTRA CF
        case!(prewrite!(10).lock(), commit!(10, 15), prewrite!(15); ok!()),
        case!(prewrite!(10).lock(), commit!(10, 15), prewrite!(20); ok!()),
        case!(prewrite!(10).lock(), commit!(10, 15), prewrite!(12); fail!()),
    ];
    run_test_cases(cases);
}

#[test]
fn test_self_rollback_detection_during_prewrite() {
    let cases = vec![
        case!(prewrite!(10), rollback!(10), prewrite!(10); fail!()),
        case!(prewrite!(10, expire), check_txn_status!(10, 15, 15), prewrite!(10); fail!()),
        case!(prewrite!(10), rollback!(10), prewrite!(15); ok!()),
        case!(prewrite!(10, expire), check_txn_status!(10, 15, 15), prewrite!(20); ok!()),
    ];
    run_test_cases(cases);
}

#[test]
fn test_acquire_pessimistic_lock_conflicts() {
    let cases = vec![
        // SELF-ROLLBACK: Same start_ts rollback detection
        case!(prewrite!(10), rollback!(10), lock!(20, 25); ok!()), // Different start_ts
        case!(prewrite!(10), rollback!(10), lock!(10, 25); fail!()), // Different start_ts
        // NEWER-LOCK: use for_update_ts for conflict detection
        case!(prewrite!(10), commit!(10, 15), lock!(12, 12); fail!()),
        case!(prewrite!(10), commit!(10, 15), lock!(12, 15); ok!()),
        case!(prewrite!(10), commit!(10, 15), lock!(12, 18); ok!()),
        // LOCKED
        case!(lock!(10, 15), lock!(11, 16); fail!()),
    ];
    run_test_cases(cases);
}

#[test]
fn test_amend_pessimistic_lock_conflicts() {
    let cases = vec![
        // SELF-ROLLBACK: Self-rollback detection during amend
        case!(lock!(10, 15), rollback!(10), pessimistic_prewrite!(10, 15); fail!()),
        // LOST-LOCK: Lock taken by another transaction by pessimistic lock
        case!(lock!(20, 20), pessimistic_prewrite!(10, 15); fail!()),
        // LOST-LOCK: Lock taken by another transaction by prwrite
        case!(prewrite!(20), commit!(20, 25), pessimistic_prewrite!(10, 22); fail!()),
        // KEEP-LOCK: Lock still held by same transaction
        case!(lock!(10, 15), pessimistic_prewrite!(10, 15); ok!()),
        case!(pessimistic_prewrite!(10, 15); ok!()),
    ];
    run_test_cases(cases);
}

#[test]
fn test_optimistic_skip_constraint_check() {
    let cases = vec![
        // Test 1: Basic conflict detection - should fail
        case!(prewrite!(10), commit!(10, 15), prewrite!(12); fail!()),
        // Test 2: With skip_constraint_check, it should be ignored for optimistic transactions
        // and still detect the conflict (behavior now disallowed for optimistic transactions)
        case!(prewrite!(10), commit!(10, 15), prewrite!(12).skip_constraint_check(); fail!()),
    ];
    run_test_cases(cases);
}

#[test]
fn test_pessimistic_action() {
    // This test verifies the key change in PR 3252: pessimistic transactions
    // use PessimisticAction to control constraint checking behavior.
    // Following the pattern from test_optimistic_skip_constraint_check.

    let cases = vec![
        // DoPessimisticCheck should detect conflicts - should fail
        case!(
            prewrite!(10),
            commit!(10, 20),
            pessimistic_prewrite!(12, 15)
                .pessimistic_action(kvproto::kvrpcpb::PrewriteRequestPessimisticAction::DoPessimisticCheck);
            fail!()
        ),
        // SkipPessimisticCheck should NOT skip conflict check - whatever is_retry is
        case!(
            prewrite!(10),
            commit!(10, 20),
            pessimistic_prewrite!(12, 15)
                .pessimistic_action(kvproto::kvrpcpb::PrewriteRequestPessimisticAction::SkipPessimisticCheck)
                .is_retry(false);
            fail!()
        ),
        case!(
            prewrite!(10),
            commit!(10, 20),
            pessimistic_prewrite!(12, 15)
                .pessimistic_action(kvproto::kvrpcpb::PrewriteRequestPessimisticAction::SkipPessimisticCheck)
                .is_retry(true);
            fail!()
        ),
        // DoConstraintCheck should detect conflicts - should fail
        case!(
            prewrite!(10),
            commit!(10, 20),
            pessimistic_prewrite!(12, 15)
                .pessimistic_action(PessimisticAction::DoConstraintCheck);
            fail!()
        ),
    ];
    run_test_cases(cases);
}
