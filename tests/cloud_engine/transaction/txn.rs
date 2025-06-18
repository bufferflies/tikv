// Copyright 2025 TiKV Project Authors. Licensed under Apache-2.0.

use kvproto::kvrpcpb::{Assertion, Op};

use super::util::*;
use crate::i_to_val;

#[test]
fn test_no_conflict() {
    let cases = vec![
        case!(get!(10); ok!()),
        case!(lock!(10, 10); ok!()),
        case!(prewrite!(10); ok!()),
        case!(commit!(10, 15); fail!()),
    ];
    run_test_cases(cases);
}

#[test]
fn test_write_write_conflict() {
    let cases = vec![
        case!(prewrite!(10), prewrite!(20); fail!()),
        case!(prewrite!(10), lock!(20, 20); fail!()),
        case!(prewrite!(10, expire), check_txn_status!(10, 20, 20), prewrite!(20); ok!()),
        case!(prewrite!(10, expire), check_txn_status!(10, 20, 20), lock!(20, 20); ok!()),
        case!(lock!(10, 10), prewrite!(20); fail!()),
        case!(lock!(10, 10), lock!(20, 20); fail!()),
        case!(lock!(10, 10), check_txn_status!(10, 20000, 20000), prewrite!(20000); fail!()),
        case!(lock!(10, 10), check_txn_status!(10, 20050, 20050), prewrite!(20050); ok!()),
        case!(lock!(10, 10), check_txn_status!(10, 20050, 20050), lock!(20050, 20050); ok!()),
        case!(prewrite!(10), commit!(10, 20), prewrite!(20); ok!()),
        case!(prewrite!(10), commit!(10, 20), prewrite!(25); ok!()),
        case!(prewrite!(10), commit!(10, 20), prewrite!(15); fail!()),
        // This does work, different from the above case.
        case!(prewrite!(10), commit!(10, 20), lock!(20, 20); ok!()),
        case!(prewrite!(10), commit!(10, 20), lock!(25, 25); ok!()),
        case!(prewrite!(10), commit!(10, 20), lock!(15, 25); ok!()),
        case!(prewrite!(10), commit!(10, 20), lock!(15, 15); fail!()),
        // rollback doesn't protect
        case!(prewrite!(10), rollback!(10), prewrite!(20); ok!()),
        case!(prewrite!(10), rollback!(10), prewrite!(5); ok!()),
        case!(prewrite!(10, expire), check_txn_status!(10, 15, 15), prewrite!(20); ok!()),
        case!(prewrite!(10), rollback!(10), lock!(20, 20); ok!()),
        case!(prewrite!(10), rollback!(10), lock!(5, 5); ok!()),
        case!(prewrite!(10, expire), check_txn_status!(10, 15, 15), lock!(20, 20); ok!()),
        case!(prewrite!(10, expire), check_txn_status!(10, 15, 15), lock!(14, 20); ok!()),
        // ROLLBACK records are not in WRITE CF any more.
        // This is different from classic TiKV. Rollback records don't result in write conflict.
        // case!(prewrite!(10, expire), check_txn_status!(10, 15, 15), prewrite!(5); fail!()),
        // case!(prewrite!(10, expire), check_txn_status!(10, 15, 15), lock!(5, 5); ok!()),
    ];
    run_test_cases(cases);
}

#[test]
fn test_read_write_conflict() {
    let cases = vec![
        case!(prewrite!(10), get!(5); ok!()),
        case!(prewrite!(10), get!(10); ok!()),
        case!(prewrite!(10), get!(20); fail!()),
        case!(prewrite!(10), check_txn_status!(10, 20, 20), get!(20); ok!()),
        case!(prewrite!(10), check_txn_status!(10, 20, 20), commit!(10, 25); ok!()),
        // pessimistic locks don't block read
        case!(lock!(10, 15), get!(5); ok!()),
        case!(lock!(10, 15), get!(10); ok!()),
        case!(lock!(10, 15), get!(16); ok!()),
    ];
    run_test_cases(cases);
}

#[test]
fn test_snapshot_read() {
    let v1 = i_to_val(1);
    let v2 = i_to_val(2);
    let cases = vec![
        case!(prewrite!(10).value(&v1), commit!(10, 20), prewrite!(30).value(&v2), commit!(30, 40), get!(10).value(&[]); ok!()),
        case!(prewrite!(10).value(&v1), commit!(10, 20), prewrite!(30).value(&v2), commit!(30, 40), get!(20).value(&v1); ok!()),
        case!(prewrite!(10).value(&v1), commit!(10, 20), prewrite!(30).value(&v2), commit!(30, 40), get!(30).value(&v1); ok!()),
        case!(prewrite!(10).value(&v1), commit!(10, 20), prewrite!(30).value(&v2), commit!(30, 40), get!(40).value(&v2); ok!()),
        case!(prewrite!(10).value(&v1), commit!(10, 20), prewrite!(30).value(&v2), commit!(30, 40), get!(50).value(&v2); ok!()),
    ];
    run_test_cases(cases);
}

#[test]
fn test_idempotency() {
    let cases = vec![
        case!(prewrite!(10), prewrite!(10); ok!()),
        case!(prewrite!(10), rollback!(10), prewrite!(10); fail!()),
        // duplicate prewrite & commit does not overwrite
        case!(prewrite!(10), commit!(10, 20), prewrite!(10), commit!(10, 30), get!(20); ok!()),
        case!(lock!(10, 10), lock!(10, 10); ok!()),
        case!(lock!(10, 10), pessimistic_prewrite!(10, 10); ok!()),
        case!(lock!(10, 10), pessimistic_prewrite!(10, 10), lock!(10, 10); fail!()),
        case!(prewrite!(10), commit!(10, 20), commit!(10, 30), get!(20); ok!()),
        case!(lock!(10, 10), pessimistic_prewrite!(10, 10), commit!(10, 20), lock!(10, 10); fail!()),
    ];
    run_test_cases(cases);
}

#[test]
fn test_assertions() {
    let cases = vec![
        case!(prewrite!(10).assert(Assertion::NotExist); ok!()),
        case!(prewrite!(30).assert(Assertion::Exist); fail!()),
        case!(prewrite!(50), commit!(50, 55), prewrite!(60).assert(Assertion::NotExist); fail!()),
        case!(prewrite!(70), commit!(70, 75), prewrite!(80).assert(Assertion::Exist); ok!()),
        case!(prewrite!(90), commit!(90, 95), prewrite!(100).delete(), commit!(100, 105), prewrite!(110).assert(Assertion::Exist); fail!()),
        // deletion is treated as not exist
        case!(prewrite!(120), commit!(120, 125), prewrite!(130).delete(), commit!(130, 135), prewrite!(140).assert(Assertion::NotExist); ok!()),
        case!(prewrite!(150), commit!(150, 155), prewrite!(160).delete().assert(Assertion::Exist); ok!()),
        case!(prewrite!(170).delete().assert(Assertion::NotExist); ok!()),
        case!(prewrite!(180).delete().assert(Assertion::Exist); fail!()),
        // None should alwyas succeed
        case!(prewrite!(190).assert(Assertion::None); ok!()),
        case!(prewrite!(200), commit!(200, 205), prewrite!(210).assert(Assertion::None); ok!()),
    ];
    run_test_cases(cases);
}
