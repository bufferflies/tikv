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

    pub static ref REMOTE_COPR_DAG_REQ_COUNTER: IntCounter = register_int_counter!(
        "tikv_worker_remote_cop_dag_request_counter",
        "Total count of remote copr requests",
    )
    .unwrap();

    pub static ref REMOTE_COPR_DAG_RESP_SIZE: IntCounter = register_int_counter!(
        "tikv_worker_remote_cop_dag_response_size",
        "Total size of remote copr responses",
    )
    .unwrap();

    pub static ref REMOTE_ANALYZE_REQ_COUNTER: IntCounter = register_int_counter!(
        "tikv_worker_remote_analyze_request_counter",
        "Total count of remote analyze requests",
    )
    .unwrap();

    pub static ref REMOTE_ANALYZE_RESP_SIZE: IntCounter = register_int_counter!(
        "tikv_worker_remote_analyze_response_size",
        "Total size of remote analyze responses",
    )
    .unwrap();

    pub static ref REMOTE_CHECKSUM_REQ_COUNTER: IntCounter = register_int_counter!(
        "tikv_worker_remote_checksum_request_counter",
        "Total count of remote checksum requests",
    )
    .unwrap();

    pub static ref REMOTE_CHECKSUM_RESP_SIZE: IntCounter = register_int_counter!(
        "tikv_worker_remote_checksum_response_size",
        "Total size of remote checksum responses",
    )
    .unwrap();

    pub static ref REMOTE_COPR_SNAPSHOT_HISTOGRAM: Histogram = register_histogram!(
        "tikv_worker_remote_cop_snapshot_duration_seconds",
        "Bucketed histogram of remote copr snapshot duration",
        exponential_buckets(0.0005, 2.0, 20).unwrap()
    ).unwrap();

    pub static ref REMOTE_COPR_PREFETCH_HISTOGRAM: Histogram = register_histogram!(
        "tikv_worker_remote_cop_prefetch_duration_seconds",
        "Bucketed histogram of remote copr prefetch segments duration",
        exponential_buckets(0.0005, 2.0, 20).unwrap()
    ).unwrap();

    pub static ref REMOTE_COPR_PREFETCH_CACHE_HIT_PERCENT_HISTOGRAM: Histogram = register_histogram!(
        "tikv_worker_remote_cop_prefetch_cache_hit_percent",
        "Bucketed histogram of cache hit percent for remote copr prefetch segments",
        linear_buckets(50.0, 2.0, 25).unwrap()
    )
    .unwrap();

    pub static ref REMOTE_COPR_REQ_HANDLE_HISTOGRAM: Histogram = register_histogram!(
        "tikv_worker_remote_cop_request_duration_seconds",
        "Bucketed histogram of remote copr request duration",
        exponential_buckets(0.0005, 2.0, 20).unwrap()
    ).unwrap();

    pub static ref REMOTE_COMPACT_REQ_HANDLE_HISTOGRAM: Histogram = register_histogram!(
        "tikv_worker_remote_compact_request_duration_seconds",
        "Bucketed histogram of remote compaction request duration",
        exponential_buckets(0.0005, 2.0, 20).unwrap()
    ).unwrap();

    pub static ref WORKER_SCALER_QUERY_FAILURES_COUNTER_VEC: IntCounterVec = register_int_counter_vec!(
        "tikv_worker_worker_scaler_query_failures_counter",
        "Total count of worker scaler failures in querying task state",
        &["task_id"],
    )
    .unwrap();

    pub static ref WORKER_MEMORY_LIMITER_CURRENT_USED: IntGauge = register_int_gauge!(
        "tikv_worker_memory_limiter_current_used",
        "Current used memory reported by worker memory limiter",
    ).unwrap();

    pub static ref SCHEMA_MANAGER_SYNC_LOOP_COUNT: IntCounter = register_int_counter!(
        "tikv_worker_schema_manager_sync_loop_count",
        "Total count of schema manager sync loop",
    ).unwrap();

    pub static ref SCHEMA_MANAGER_SYNC_LOOP_ERROR_COUNT: IntCounterVec = register_int_counter_vec!(
        "tikv_worker_schema_manager_sync_loop_error_count",
        "Total count of schema manager sync loop errors",
        &["type"],
    ).unwrap();
}
