// Copyright 2025 TiKV Project Authors. Licensed under Apache-2.0.

use prometheus::*;

lazy_static::lazy_static! {
    pub static ref HYPER_RELOAD_CERT_COUNTER: IntCounterVec = register_int_counter_vec!(
        "tikv_hyper_reload_cert_total",
        "Total number of hyper reload certificates",
        &["status", "type"]
    ).unwrap();
}
