// Copyright 2022 TiKV Project Authors. Licensed under Apache-2.0.

use prometheus::{
    exponential_buckets, register_histogram_vec, register_int_counter_vec, HistogramVec,
    IntCounterVec,
};
use prometheus_static_metric::{auto_flush_from, make_auto_flush_static_metric};

make_auto_flush_static_metric! {
    pub label_enum DfsType {
        s3,
        local,
    }

    pub label_enum S3ApiType {
        get,
        put,
        delete,
        list,
        head,
        copy,
        get_tagging,
    }

    pub label_enum MetricsFileType {
        sst,
        txn,
        schema,
        col,
        blob,
        vec,
        unknown,
    }

    pub struct KvEngineDfsThroughput: LocalIntCounter {
        "dfs_type" => DfsType,
        "type" => S3ApiType,
        "file_type" => MetricsFileType,
    }

    pub struct KvEngineDfsLatency: LocalHistogram {
        "type" => S3ApiType,
        "file_type" => MetricsFileType,
    }

    pub struct KvEngineDfsLatencyWithRetry: LocalHistogram {
        "type" => S3ApiType,
        "file_type" => MetricsFileType,
    }

    pub struct KvEngineDfsRequestCounter: LocalIntCounter {
        "type" => S3ApiType,
        "file_type" => MetricsFileType,
    }

    pub struct KvEngineDfsRetryCounter: LocalIntCounter {
        "type" => S3ApiType,
        "file_type" => MetricsFileType,
    }
}

impl From<&str> for MetricsFileType {
    fn from(s: &str) -> Self {
        match s {
            "sst" => MetricsFileType::sst,
            "blob" => MetricsFileType::blob,
            "txn" => MetricsFileType::txn,
            "schema" => MetricsFileType::schema,
            "col" => MetricsFileType::col,
            "vec" => MetricsFileType::vec,
            _ => MetricsFileType::unknown,
        }
    }
}

lazy_static! {
    pub static ref KVENGINE_DFS_THROUGHPUT_VEC: IntCounterVec = register_int_counter_vec!(
        "kv_engine_dfs_throughput_bytes",
        "Throughput of kvengine dfs by dfs_type, S3 API type and file type",
        &["dfs_type", "type", "file_type"]
    )
    .unwrap();
    // TODO(hhwyt): The following metrics may also need to be tagged by dfs_type.
    pub static ref KVENGINE_DFS_LATENCY_VEC: HistogramVec = register_histogram_vec!(
        "kv_engine_dfs_latency_seconds",
        "Latency of kvengine dfs in seconds by S3 API type and file type",
        &["type", "file_type"],
        exponential_buckets(0.001, 2.0, 20).unwrap()
    )
    .unwrap();
    pub static ref KVENGINE_DFS_LATENCY_WITH_RETRY_VEC: HistogramVec = register_histogram_vec!(
        "kv_engine_dfs_latency_with_retry_seconds",
        "Latency of kvengine dfs including retry time in seconds by S3 API type and file type",
        &["type", "file_type"],
        exponential_buckets(0.001, 2.0, 20).unwrap()
    )
    .unwrap();
    pub static ref KVENGINE_DFS_REQUEST_COUNTER_VEC: IntCounterVec = register_int_counter_vec!(
        "kv_engine_dfs_request_counter",
        "Count of kvengine dfs requests by S3 API type and file type",
        &["type", "file_type"]
    )
    .unwrap();
    pub static ref KVENGINE_DFS_RETRY_COUNTER_VEC: IntCounterVec = register_int_counter_vec!(
        "kv_engine_dfs_retry_counter",
        "Retries counter of kvengine dfs by S3 API type and file type",
        &["type", "file_type"]
    )
    .unwrap();
    pub static ref KVENGINE_CACHEFS_REQ_COUNTER_VEC: IntCounterVec = register_int_counter_vec!(
        "kv_engine_cachefs_req_count",
        "Count of kvengine cachefs requests hit/miss",
        &["type"]
    )
    .unwrap();
}

lazy_static! {
    pub static ref KVENGINE_DFS_THROUGHPUT: KvEngineDfsThroughput =
        auto_flush_from!(KVENGINE_DFS_THROUGHPUT_VEC, KvEngineDfsThroughput);
    pub static ref KVENGINE_DFS_LATENCY: KvEngineDfsLatency =
        auto_flush_from!(KVENGINE_DFS_LATENCY_VEC, KvEngineDfsLatency);
    pub static ref KVENGINE_DFS_LATENCY_WITH_RETRY: KvEngineDfsLatencyWithRetry = auto_flush_from!(
        KVENGINE_DFS_LATENCY_WITH_RETRY_VEC,
        KvEngineDfsLatencyWithRetry
    );
    pub static ref KVENGINE_DFS_REQUEST_COUNTER: KvEngineDfsRequestCounter =
        auto_flush_from!(KVENGINE_DFS_REQUEST_COUNTER_VEC, KvEngineDfsRequestCounter);
    pub static ref KVENGINE_DFS_RETRY_COUNTER: KvEngineDfsRetryCounter =
        auto_flush_from!(KVENGINE_DFS_RETRY_COUNTER_VEC, KvEngineDfsRetryCounter);
}
