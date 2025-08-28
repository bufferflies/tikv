// Copyright 2025 TiKV Project Authors. Licensed under Apache-2.0.

use lazy_static::*;
use prometheus::*;

lazy_static! {
    pub static ref ACTIVE_KEYSPACE_READ_BYTES: IntCounterVec = register_int_counter_vec!(
        "tikv_active_keyspace_read_bytes",
        "Total bytes read by active keyspace",
        &["keyspace_id"]
    )
    .unwrap();
    pub static ref REQUEST_WAIT_HISTOGRAM_VEC: HistogramVec = register_histogram_vec!(
        "tikv_request_wait_duration_seconds",
        "Bucketed histogram of request wait duration",
        &["type"]
    )
    .unwrap();
}
