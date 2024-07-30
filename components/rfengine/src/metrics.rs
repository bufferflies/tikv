// Copyright 2022 TiKV Project Authors. Licensed under Apache-2.0.

use lazy_static::lazy_static;
use prometheus::*;
use prometheus_static_metric::*;

use crate::*;

make_static_metric! {
    pub label_enum LogQueueKind {
        rewrite,
        append,
    }

    pub struct LogQueueHistogramVec: Histogram {
        "type" => LogQueueKind,
    }

    pub struct LogQueueCounterVec: IntCounter {
        "type" => LogQueueKind,
    }

    pub struct LogQueueGaugeVec: IntGauge {
        "type" => LogQueueKind,
    }
}

pub fn flush_engine_properties(_engine: &RfEngine, _name: &str) {}

lazy_static! {
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
}
