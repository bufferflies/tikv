// Copyright 2024 TiKV Project Authors. Licensed under Apache-2.0.

use lazy_static::lazy_static;
use prometheus::*;

lazy_static! {
    pub static ref NATIVE_BR_BACKUP_SUCCESS: IntCounter =
        register_int_counter!("native_br_backup_success", "Number of success backup").unwrap();
    pub static ref NATIVE_BR_BACKUP_ERROR: IntCounter =
        register_int_counter!("native_br_backup_error", "Number of errors during backup").unwrap();
    pub static ref NATIVE_BR_BACKUP_MISSING_COMMIT_RECORD: IntCounter = register_int_counter!(
        "native_br_backup_missing_commit_record",
        "Number of backup missing commit record"
    )
    .unwrap();
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
    pub static ref NATIVE_BR_RESTORED_SNAPSHOT_KV_BYTES: IntCounter = register_int_counter!(
        "native_br_restored_snapshot_bytes",
        "Total bytes of restored snapshot"
    )
    .unwrap();
    pub static ref NATIVE_BR_RESTORE_SNAPSHOT_REQUEST_SECS: Histogram = register_histogram!(
        "native_br_restore_snapshot_request_secs",
        "Duration of snapshot restore requests in seconds",
        exponential_buckets(0.01, 2.0, 16).unwrap()
    )
    .unwrap();
    pub static ref NATIVE_BR_RESTORE_PENDING_DATA_SIZE: IntGauge = register_int_gauge!(
        "native_br_restore_pending_data_size",
        "Restore pending data size",
    )
    .unwrap();
    pub static ref NATIVE_BR_RESTORED_DATA_SIZE: IntCounter =
        register_int_counter!("native_br_restored_data_size", "Restored data size",).unwrap();
    pub static ref NATIVE_BR_RESTORED_KV_SIZE: IntCounter =
        register_int_counter!("native_br_restored_kv_size", "Restored kv size",).unwrap();
    pub static ref NATIVE_BR_RATE_LIMITER_WAIT_DURATION_SECS: HistogramVec =
        register_histogram_vec!(
            "native_br_rate_limiter_wait_duration_secs",
            "Duration of rate limiter wait in seconds",
            &["source"],
            exponential_buckets(0.001, 2.0, 20).unwrap()
        )
        .unwrap();
    pub static ref NATIVE_BR_RATE_LIMITER_BYTES_TOTAL: IntCounterVec = register_int_counter_vec!(
        "native_br_rate_limiter_bytes_total",
        "Total bytes consumed through rate limiter",
        &["source"],
    )
    .unwrap();
}
