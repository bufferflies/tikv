// Copyright 2024 TiKV Project Authors. Licensed under Apache-2.0.

use lazy_static::lazy_static;
use prometheus::*;

lazy_static! {
    pub static ref NATIVE_BR_RESTORE_ERROR: IntCounterVec = register_int_counter_vec!(
        "native_br_restore_error",
        "Number of errors during restoration",
        &["type"],
    )
    .unwrap();
    pub static ref NATIVE_BR_RFENGINE_WAL_EPOCH_OVERWRITTEN_ERROR: IntCounter =
        register_int_counter!(
            "native_br_restore_rfengine_wal_epoch_overwritten_error",
            "Number of errors that epoch of rfengine WAL is overwritten"
        )
        .unwrap();
}
