// Copyright 2023 TiKV Project Authors. Licensed under Apache-2.0.

use lazy_static::lazy_static;
use prometheus::*;

lazy_static! {
    pub static ref NATIVE_BR_HISTOGRAM_VEC: HistogramVec = register_histogram_vec!(
        "tikv_worker_native_br_duration_seconds",
        "Bucketed histogram of native br duration",
        &["type"],
        // Start from 10ms.
        exponential_buckets(0.01, 2.0, 16).unwrap()
    )
    .unwrap();

    pub static ref NATIVE_BR_COUNTER_VEC: IntCounterVec = register_int_counter_vec!(
        "tikv_worker_native_br_counter",
        "The counter of native br operations",
        &["type"],
    )
    .unwrap();
}
