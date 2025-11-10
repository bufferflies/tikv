// Copyright 2025 TiKV Project Authors. Licensed under Apache-2.0.

use lazy_static::lazy_static;
use prometheus::*;

lazy_static! {
    pub static ref REP_SYNC_WAL_TS: IntGauge = register_int_gauge!(
        "tikv_cdc_rep_sync_wal_ts",
        "The latest timestamp of sync WAL (milliseconds)"
    )
    .unwrap();
    pub static ref REP_SYNC_WAL_TS_LAG: IntGauge = register_int_gauge!(
        "tikv_cdc_rep_sync_wal_ts_lag",
        "The lag between the sync WAL timestamp and now (milliseconds)"
    )
    .unwrap();
    pub static ref REP_SYNC_WAL_TS_LAG_HISTOGRAM: Histogram = register_histogram!(
        "tikv_cdc_rep_sync_wal_ts_lag_seconds",
        "Bucketed histogram of the gap between sync WAL timestamp and now",
        exponential_buckets(0.001, 2.0, 24).unwrap()
    )
    .unwrap();

    pub static ref REP_UPDATE_STORE_COUNTER: IntCounterVec = register_int_counter_vec!(
        "tikv_cdc_rep_update_store_counter",
        "Counter of update store",
        &["store", "result"]
    )
    .unwrap();
    pub static ref REP_UPDATE_STORE_DURATION: HistogramVec = register_histogram_vec!(
        "tikv_cdc_rep_update_store_duration_seconds",
        "Bucketed histogram of update stores",
        &["store"],
        exponential_buckets(0.1, 2.0, 16).unwrap() // 0.1s ~ 100m
    ).unwrap();
    pub static ref REP_UPDATE_STORE_EPOCH_LAG: IntGaugeVec = register_int_gauge_vec!(
        "tikv_cdc_rep_update_store_epoch_lag",
        "Epoch lag between store progress and first target",
        &["store"],
    ).unwrap();

    pub static ref REP_SCAN_LOCKS_DURATION_HISTOGRAM: Histogram = register_histogram!(
        "tikv_cdc_rep_scan_locks_duration_seconds",
        "Bucketed histogram of replication worker scan locks duration",
        exponential_buckets(0.005, 2.0, 20).unwrap()
    )
    .unwrap();
    pub static ref REP_SCAN_LOCKS_BYTES: IntCounter = register_int_counter!(
        "tikv_cdc_rep_scan_locks_bytes_total",
        "Total fetched bytes of replication worker scan locks"
    )
    .unwrap();
    pub static ref REP_SCAN_LOCKS_TASKS: IntGaugeVec = register_int_gauge_vec!(
        "tikv_cdc_rep_scan_locks_tasks",
        "Total number of replication worker scan locks tasks",
        &["type"]
    )
    .unwrap();
}
