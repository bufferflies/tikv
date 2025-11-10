// Copyright 2025 TiKV Project Authors. Licensed under Apache-2.0.

use lazy_static::lazy_static;
use prometheus::*;

lazy_static! {
    pub static ref LOCAL_FILES: IntGauge = register_int_gauge!(
        "rfstore_local_file_on_disk",
        "Number of rfstore local files on disk.",
    )
    .unwrap();
    pub static ref LOCAL_ALIVE_FILES: IntGaugeVec = register_int_gauge_vec!(
        "rfstore_local_file_alive",
        "Number of rfstore local files used by kvengine.",
        &["type"]
    )
    .unwrap();
    pub static ref LOCAL_SKIP_GC_FILES: IntGauge = register_int_gauge!(
        "rfstore_local_file_skip_gc",
        "Number of rfstore local files skipped by gc.",
    )
    .unwrap();
    pub static ref LOCAL_FILE_GC: IntCounterVec = register_int_counter_vec!(
        "rfstore_local_file_gc_total",
        "Total number of rfstore garbage collected of local files.",
        &["type"]
    )
    .unwrap();
    pub static ref LOCAL_PENDING_GC_FILES: IntGauge = register_int_gauge!(
        "rfstore_local_file_pending_gc",
        "Number of rfstore local files pending for gc.",
    )
    .unwrap();
    pub static ref LOCAL_FILE_GC_ERRORS: IntCounter = register_int_counter!(
        "rfstore_local_file_error_total",
        "Total number of rfstore local file gc errors.",
    )
    .unwrap();
}
