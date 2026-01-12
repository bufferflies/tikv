// Copyright 2022 TiKV Project Authors. Licensed under Apache-2.0.

use lazy_static::lazy_static;
use prometheus::*;
use prometheus_static_metric::*;

use crate::*;

make_static_metric! {
    pub struct ResourceUsageIntGaugeVec: IntGauge {
        "type" => {
            memory,
            disk,
        },
    }

    pub struct EntriesCountIntGaugeVec: IntGauge {
         "type" => {
             memory,
             offloaded,
         },
     }
}

pub fn flush_engine_properties(_engine: &RfEngine, _name: &str) {}

lazy_static! {
    pub static ref ENGINE_RESOUCE_USAGE: ResourceUsageIntGaugeVec = register_static_int_gauge_vec!(
        ResourceUsageIntGaugeVec,
        "raft_engine_resouce_usage_bytes",
        "Total number of resource usage",
        &["type"],
    )
    .unwrap();
    pub static ref ENGINE_ENTRIES_COUNT: EntriesCountIntGaugeVec = register_static_int_gauge_vec!(
         EntriesCountIntGaugeVec,
         "raft_engine_total_entries_count",
         "Total number of raft entries",
         &["type"],
     )
    .unwrap();
    pub static ref ENGINE_PERSIST_DURATION_HISTOGRAM: Histogram = register_histogram!(
        "raft_engine_persist_duration_seconds",
        "Bucketed histogram of Raft Engine persist duration",
        exponential_buckets(0.00005, 1.8, 26).unwrap()
    )
    .unwrap();
    pub static ref ENGINE_ROTATE_DURATION_HISTOGRAM: Histogram = register_histogram!(
        "raft_engine_rotate_duration_seconds",
        "Bucketed histogram of Raft Engine rotate duration",
        exponential_buckets(0.00005, 1.8, 26).unwrap()
    )
    .unwrap();
    pub static ref ENGINE_WAL_WRITE_THROTTLE_DURATION_HISTOGRAM: Histogram = register_histogram!(
        "raft_engine_wal_write_throttle_duration_seconds",
        "Bucketed histogram of Raft Engine wal write throttle duration",
        exponential_buckets(0.00005, 1.8, 26).unwrap()
    )
    .unwrap();
    pub static ref ENGINE_WAL_WRITE_DURATION_HISTOGRAM: Histogram = register_histogram!(
        "raft_engine_wal_write_duration_seconds",
        "Bucketed histogram of Raft Engine wal write duration",
        exponential_buckets(0.00005, 1.8, 26).unwrap()
    )
    .unwrap();
    pub static ref ENGINE_APPLY_DURATION_HISTOGRAM: Histogram = register_histogram!(
        "raft_engine_apply_duration_seconds",
        "Bucketed histogram of Raft Engine apply duration",
        exponential_buckets(0.00005, 1.8, 26).unwrap()
    )
    .unwrap();
    pub static ref ENGINE_TRUNCATE_DURATION_HISTOGRAM: Histogram = register_histogram!(
        "raft_engine_truncate_duration_seconds",
        "Bucketed histogram of Raft Engine truncate duration",
        exponential_buckets(0.00005, 1.8, 26).unwrap()
    )
    .unwrap();
    pub static ref ENGINE_FETCH_ENTRIES_DURATION_HISTOGRAM: Histogram = register_histogram!(
        "raft_engine_fetch_entries_duration_seconds",
        "Bucketed histogram of Raft Engine fetch entries duration",
        exponential_buckets(0.00005, 1.8, 26).unwrap()
    )
    .unwrap();
    pub static ref ENGINE_COMPACT_WAL_DURATION_HISTOGRAM: Histogram = register_histogram!(
        "raft_engine_compact_wal_duration_seconds",
        "Bucketed histogram of Raft Engine compact WAL duration",
        exponential_buckets(0.0005, 1.8, 26).unwrap()
    )
    .unwrap();
    pub static ref ENGINE_COMPACT_CACHE_WAL_SKIPPED_COUNTER: IntCounter = register_int_counter!(
        "raft_engine_compact_cached_wal_skipped_counter",
        "Counter of rfengine WAL caching for compact was skipped",
    )
    .unwrap();
    pub static ref ENGINE_TOTAL_WALS_GAUGE: IntGauge = register_int_gauge!(
        "raft_engine_total_wals_count",
        "Total number of raft WAL logs"
    )
    .unwrap();
    pub static ref ENGINE_PENDING_COMPACTION_WALS_GAUGE: IntGauge = register_int_gauge!(
        "raft_engine_pending_compaction_wals",
        "Total number of pending compaction wals"
    )
    .unwrap();
    pub static ref ENGINE_TAKE_SNAPSHOT_DURATION_HISTOGRAM: Histogram = register_histogram!(
        "raft_engine_take_snapshot_duration_seconds",
        "Bucketed histogram of Raft Engine take snapshot duration",
        exponential_buckets(0.0005, 1.8, 26).unwrap()
    )
    .unwrap();
    pub static ref ENGINE_REGION_WRITE_BATCH_SIZE_HISTOGRAM: Histogram = register_histogram!(
        "raft_engine_region_write_batch_size",
        "Bucketed histogram of Raft Engine region write batch size",
        exponential_buckets(16.0, 2.0, 20).unwrap()
    )
    .unwrap();
    pub static ref RFENGINE_RLOG_GC_SIZE: HistogramVec = register_histogram_vec!(
        "raft_engine_log_file_gc_size",
        "Bucketed histogram of raft log gc file size",
        &["status"],
        exponential_buckets(16.0, 2.0, 20).unwrap()
    )
    .unwrap();
    pub static ref RFENGINE_BACKUP_DURATION_HISTOGRAM: HistogramVec = register_histogram_vec!(
        "raft_engine_backup_duration_seconds",
        "Bucketed histogram of rfengine backup duration",
        &["type"], // full or incremental
        exponential_buckets(0.1, 2.0, 20).unwrap() // start from 0.1s
    )
    .unwrap();
    pub static ref RFENGINE_BACKUP_COUNTER: IntCounterVec = register_int_counter_vec!(
        "raft_engine_backup_counter",
        "Counter of rfengine backup",
        &["status"], // success or fail
    )
    .unwrap();
    pub static ref RFENGINE_DFS_WORKER_HEALTHY_GAUGE: IntGauge = register_int_gauge!(
        "raft_engine_dfs_worker_healthy",
        "Status of healthy dfs worker",
    ).unwrap();
    pub static ref RFENGINE_DFS_UPLOAD_BYTES: IntCounter = register_int_counter!(
        "raft_engine_dfs_uploaded_bytes_total",
        "Total number of bytes uploaded to DFS",
    ).unwrap();
    pub static ref RFENGINE_DFS_REQUESTS: IntCounter = register_int_counter!(
        "raft_engine_dfs_requests_total",
        "Total number of requests to DFS",
    ).unwrap();
    pub static ref RFENGINE_DFS_UPLOAD_BYTES_ACKED: IntCounter = register_int_counter!(
        "raft_engine_dfs_uploaded_bytes_acked",
        "Total number of bytes uploaded to DFS",
    ).unwrap();
    pub static ref RFENGINE_DFS_REQUESTS_ACKED: IntCounter = register_int_counter!(
        "raft_engine_dfs_requests_acked",
        "Total number of requests to DFS",
    ).unwrap();

    pub static ref RFENGINE_DOUBLE_WRITE_HEALTHY_GAUGE: IntGauge = register_int_gauge!(
        "raft_engine_double_write_healthy",
        "Status of healthy wal double write",
    )
    .unwrap();
    pub static ref RFENGINE_DFS_RUNNING_UPLOADS: IntGauge = register_int_gauge!(
        "raft_engine_dfs_running_uploads",
        "Number of running uploads to DFS",
    )
    .unwrap();
}

#[cfg(feature = "testexport")]
lazy_static! {
    pub static ref RFENGINE_DFS_WORKER_BECOME_UNHEALTHY_COUNTER: IntCounter =
        register_int_counter!(
            "raft_engine_dfs_worker_become_unhealthy_counter",
            "Counter of rfengine DFS worker become unhealthy",
        )
        .unwrap();
}
