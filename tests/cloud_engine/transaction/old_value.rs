// Copyright 2025 TiKV Project Authors. Licensed under Apache-2.0.

use serial_test::serial;

use super::util::*;

/// Tests old_value capture mechanism in transaction commands.
///
/// These tests use a probe to capture and verify the actual old_values that
/// were captured by the transaction layer through the get_txn_extra_op()
/// mechanism.

#[test]
#[serial(old_value)]
fn test_old_value_basic_optimistic() {
    let _extra_op = ExtraOpGuard::to_read_old_value();

    run_test_cases(vec![
        case!(
            prewrite!(10).value(b"v1"),
            assert_old_value!(None::<&[u8]>);
            ok!()
        ),
        case!(
            prewrite!(5).value(b"initial"),
            commit!(5, 6),
            prewrite!(10).value(b"v2"),
            assert_old_value!(Some(b"initial"));
            ok!()
        ),
        case!(
            prewrite!(15).value(b"to_delete"),
            commit!(15, 16),
            prewrite!(20).delete(),
            assert_old_value!(Some(b"to_delete"));
            ok!()
        ),
    ]);
}

#[test]
#[serial(old_value)]
fn test_old_value_acquire_pessimistic_lock() {
    let _extra_op = ExtraOpGuard::to_read_old_value();

    // Test standalone acquire pessimistic lock should capture old_value
    run_test_cases(vec![
        case!(
            prewrite!(5).value(b"initial"),
            commit!(5, 6),
            lock!(10, 15),
            assert_old_value!(Some(b"initial"));
            ok!()
        ),
        case!(
            lock!(20, 25),
            assert_old_value!(None::<&[u8]>);
            ok!()
        ),
    ]);
}

#[test]
#[serial(old_value)]
fn test_old_value_pessimistic_prewrite() {
    let _extra_op = ExtraOpGuard::to_read_old_value();

    run_test_cases(vec![
        // NOTE: this is different from classic TiKV.
        // Pessimistic prewrite without prior pessimistic lock
        // still captures old_value when ExtraOp::ReadOldValue is enabled.
        // This is due to next-gen tikv enforces constraint checks, even if the
        // pessimistic_action is SkipPessimisticCheck.
        case!(
            prewrite!(5).value(b"initial"),
            commit!(5, 6),
            pessimistic_prewrite!(10, 15).value(b"pessimistic_v2"),
            assert_old_value!(Some(b"initial"));
            ok!()
        ),
        case!(
            prewrite!(25).value(b"to_delete_pessimistic"),
            commit!(25, 26),
            pessimistic_prewrite!(30, 35).delete(),
            assert_old_value!(Some(b"to_delete_pessimistic"));
            ok!()
        ),
    ]);
}
