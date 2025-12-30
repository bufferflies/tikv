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
    pub static ref NATIVE_BR_BACKUP_BATCH_SIZE: Histogram = register_histogram!(
        "native_br_backup_batch_size",
        "Histogram of backup batch size",
        exponential_buckets(1.0, 2.0, 16).unwrap()
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
    pub static ref NATIVE_BR_RFENGINE_CACHE_HIT: IntCounter = register_int_counter!(
        "native_br_rfengine_cache_hit",
        "Number of rfengine cache hits during restore"
    )
    .unwrap();
    pub static ref NATIVE_BR_RFENGINE_CACHE_MISS: IntCounter = register_int_counter!(
        "native_br_rfengine_cache_miss",
        "Number of rfengine cache misses during restore"
    )
    .unwrap();
}
