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
}
