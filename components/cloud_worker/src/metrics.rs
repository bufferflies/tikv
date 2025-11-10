// Copyright 2023 TiKV Project Authors. Licensed under Apache-2.0.

use lazy_static::lazy_static;
use prometheus::*;
use prometheus_static_metric::*;

make_static_metric! {
    pub label_enum ServiceFailureType {
        queue_full,
        wait_timeout,
    }

    pub struct RemoteCompactFailedRequestsCounterVec: IntCounter {
        "type" => ServiceFailureType,
    }

    pub struct RemoteCopFailedRequestsCounterVec: IntCounter {
        "type" => ServiceFailureType,
    }
}

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

    pub static ref REMOTE_COPR_PROCESSING_REQ_COUNTER: IntGauge = register_int_gauge!(
        "tikv_worker_remote_cop_processing_requests_counter",
        "Number of remote coprocessor requests under processing"
    ).unwrap();

    pub static ref REMOTE_COPR_FAILED_REQUESTS_COUNTER_VEC: RemoteCopFailedRequestsCounterVec = register_static_int_counter_vec!(
        RemoteCopFailedRequestsCounterVec,
        "tikv_worker_remote_cop_failed_requests_counter",
        "Number of remote coprocessor requests that failed to acquire permit, by failure reason",
        &["type"]
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

    pub static ref REMOTE_COPR_PROCESS_HISTOGRAM: Histogram = register_histogram!(
        "tikv_worker_remote_cop_process_duration_seconds",
        "Bucketed histogram of remote copr process duration",
        exponential_buckets(0.0005, 2.0, 20).unwrap()
    ).unwrap();

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

    pub static ref REMOTE_COMPACT_PROCESSING_REQUESTS_COUNTER: IntGauge = register_int_gauge!(
        "tikv_worker_remote_compact_processing_requests_counter",
        "Number of remote compaction requests being processed"
    ).unwrap();

    pub static ref WORKER_LIMITER_REQUEST_WAIT_HISTOGRAM: HistogramVec = register_histogram_vec!(
        "tikv_worker_limiter_request_wait_duration_seconds",
        "Bucketed histogram of worker limiter request wait duration",
        & ["type"],
        exponential_buckets(0.0005, 2.0, 20).unwrap()
    ).unwrap();

    pub static ref WORKER_LIMITER_WAITING_REQUESTS_COUNTER_VEC: IntGaugeVec = register_int_gauge_vec!(
        "tikv_worker_limiter_waiting_requests_counter",
        "The number of requests waiting for permits",
        &["type"]
    )
    .unwrap();

    pub static ref REMOTE_COMPACT_FAILED_REQUESTS_COUNTER_VEC: RemoteCompactFailedRequestsCounterVec = register_static_int_counter_vec!(
        RemoteCompactFailedRequestsCounterVec,
        "tikv_worker_remote_compact_failed_requests_counter",
        "Number of remote compaction requests that failed to acquire permit, by failure reason",
        &["type"]
    )
    .unwrap();

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

    pub static ref NATIVE_BR_V1X_COUNTER_VEC: IntCounterVec = register_int_counter_vec!(
        "tikv_worker_native_br_v1x_counter",
        "The counter of native br v1x operations",
        &["type", "result"]
    )
    .unwrap();

    pub static ref NATIVE_BR_V1X_BACKGROUND_TASKS: IntGaugeVec = register_int_gauge_vec!(
        "tikv_worker_native_running_tasks",
        "The running tasks of v1x API",
        &["type"]
    ).unwrap();
}
