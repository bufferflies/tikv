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
}
