// Copyright 2020 TiKV Project Authors. Licensed under Apache-2.0.

use lazy_static::*;
use prometheus::*;
use prometheus_static_metric::*;

/// Installing a new capture contains 2 phases, one for incremental scanning and
/// one for fetching delta changes from raftstore. They can share some similar
/// metrics, in which case we can use this tag to distinct them.
pub const TAG_DELTA_CHANGE: &str = "delta_change";
pub const TAG_INCREMENTAL_SCAN: &str = "incremental_scan";

make_auto_flush_static_metric! {
    pub label_enum PerfMetric {
        user_key_comparison_count,
        block_cache_hit_count,
        block_read_count,
        block_read_byte,
        block_read_time,
        block_cache_index_hit_count,
        index_block_read_count,
        block_cache_filter_hit_count,
        filter_block_read_count,
        block_checksum_time,
        block_decompress_time,
        get_read_bytes,
        iter_read_bytes,
        internal_key_skipped_count,
        internal_delete_skipped_count,
        internal_recent_skipped_count,
        get_snapshot_time,
        get_from_memtable_time,
        get_from_memtable_count,
        get_post_process_time,
        get_from_output_files_time,
        seek_on_memtable_time,
        seek_on_memtable_count,
        next_on_memtable_count,
        prev_on_memtable_count,
        seek_child_seek_time,
        seek_child_seek_count,
        seek_min_heap_time,
        seek_max_heap_time,
        seek_internal_seek_time,
        db_mutex_lock_nanos,
        db_condition_wait_nanos,
        read_index_block_nanos,
        read_filter_block_nanos,
        new_table_block_iter_nanos,
        new_table_iterator_nanos,
        block_seek_nanos,
        find_table_nanos,
        bloom_memtable_hit_count,
        bloom_memtable_miss_count,
        bloom_sst_hit_count,
        bloom_sst_miss_count,
        get_cpu_nanos,
        iter_next_cpu_nanos,
        iter_prev_cpu_nanos,
        iter_seek_cpu_nanos,
        encrypt_data_nanos,
        decrypt_data_nanos,
    }

    pub struct PerfCounter: LocalIntCounter {
        "metric" => PerfMetric,
    }
}

lazy_static! {
    pub static ref CDC_ENDPOINT_PENDING_TASKS: IntGauge = register_int_gauge!(
        "tikv_cdc_endpoint_pending_tasks",
        "CDC endpoint pending tasks"
    ).unwrap();
    pub static ref CDC_RESOLVED_TS_GAP_HISTOGRAM: Histogram = register_histogram!(
        "tikv_cdc_resolved_ts_gap_seconds",
        "Bucketed histogram of the gap between cdc resolved ts and current tso",
        exponential_buckets(0.001, 2.0, 24).unwrap()
    )
    .unwrap();
    pub static ref CDC_SCAN_DURATION_HISTOGRAM: Histogram = register_histogram!(
        "tikv_cdc_scan_duration_seconds",
        "Bucketed histogram of cdc async scan duration",
        exponential_buckets(0.005, 2.0, 20).unwrap()
    )
    .unwrap();
    pub static ref CDC_SCAN_BYTES: IntCounter = register_int_counter!(
        "tikv_cdc_scan_bytes_total",
        "Total fetched bytes of CDC incremental scan"
    )
    .unwrap();
    pub static ref CDC_SCAN_TASKS: IntGaugeVec = register_int_gauge_vec!(
        "tikv_cdc_scan_tasks",
        "Total number of CDC incremental scan tasks",
        &["type"]
    )
    .unwrap();
    pub static ref CDC_SCAN_DISK_READ_BYTES: IntCounter = register_int_counter!(
        "tikv_cdc_scan_disk_read_bytes_total",
        "Total disk read bytes of CDC incremental scan"
    ).unwrap();
    pub static ref CDC_MIN_RESOLVED_TS_REGION: IntGauge = register_int_gauge!(
        "tikv_cdc_min_resolved_ts_region",
        "The region which has minimal resolved ts"
    )
    .unwrap();
    pub static ref CDC_MIN_RESOLVED_TS_LAG: IntGauge = register_int_gauge!(
        "tikv_cdc_min_resolved_ts_lag",
        "The lag between the minimal resolved ts and the current ts (milliseconds)"
    ).unwrap();
    pub static ref CDC_MIN_RESOLVED_TS: IntGauge = register_int_gauge!(
        "tikv_cdc_min_resolved_ts",
        "The minimal resolved ts for current regions (milliseconds)"
    )
    .unwrap();
    pub static ref CDC_PENDING_BYTES_GAUGE: IntGauge = register_int_gauge!(
        "tikv_cdc_pending_bytes",
        "Bytes in memory of a pending register",
    )
    .unwrap();
    pub static ref CDC_PENDING_LOCKS_BYTES_GAUGE: IntGauge = register_int_gauge!(
        "tikv_cdc_pending_locks_bytes",
        "Bytes of locks in memory of a pending region",
    )
    .unwrap();
    pub static ref CDC_CAPTURED_REGION_COUNT: IntGauge = register_int_gauge!(
        "tikv_cdc_captured_region_total",
        "Total number of CDC captured regions"
    )
    .unwrap();
    pub static ref CDC_OLD_VALUE_CACHE_LEN: IntGauge = register_int_gauge!(
        "tikv_cdc_old_value_cache_length",
        "Number of elements in old value cache"
    )
    .unwrap();
    pub static ref CDC_OLD_VALUE_CACHE_CAP: IntGauge = register_int_gauge!(
        "tikv_cdc_old_value_cache_capacity",
        "Capacity of old value cache"
    )
    .unwrap();
    pub static ref CDC_SINK_BYTES: IntGauge = register_int_gauge!(
        "tikv_cdc_sink_memory_bytes",
        "Total bytes of memory used in CDC sink"
    )
    .unwrap();
    pub static ref CDC_SINK_CAP: IntGauge = register_int_gauge!(
        "tikv_cdc_sink_memory_capacity",
        "Capacity of CDC sink capacity in bytes"
    )
    .unwrap();
    pub static ref CDC_REGION_RESOLVE_STATUS_GAUGE_VEC: IntGaugeVec = register_int_gauge_vec!(
        "tikv_cdc_region_resolve_status",
        "The status of CDC captured regions",
        &["status"]
    )
    .unwrap();
    pub static ref CDC_OLD_VALUE_CACHE_MISS: IntGauge = register_int_gauge!(
        "tikv_cdc_old_value_cache_miss",
        "Count of old value cache missing"
    )
    .unwrap();
    pub static ref CDC_OLD_VALUE_CACHE_MISS_NONE: IntGauge = register_int_gauge!(
        "tikv_cdc_old_value_cache_miss_none",
        "Count of None old value cache missing"
    )
    .unwrap();
    pub static ref CDC_OLD_VALUE_CACHE_ACCESS: IntGauge = register_int_gauge!(
        "tikv_cdc_old_value_cache_access",
        "Count of old value cache accessing"
    )
    .unwrap();
    pub static ref CDC_OLD_VALUE_CACHE_BYTES: IntGauge =
        register_int_gauge!("tikv_cdc_old_value_cache_bytes", "Bytes of old value cache").unwrap();
    pub static ref CDC_OLD_VALUE_CACHE_MEMORY_QUOTA: IntGauge =
        register_int_gauge!("tikv_cdc_old_value_cache_memory_quota", "Memory quota in bytes of old value cache").unwrap();
    pub static ref CDC_OLD_VALUE_SCAN_DETAILS: IntCounterVec = register_int_counter_vec!(
        "tikv_cdc_old_value_scan_details",
        "Bucketed counter of scan details for old value",
        // Two types: `incremental_scan` and `delta_change`.
        &["cf", "tag", "type"]
    )
    .unwrap();
    pub static ref CDC_OLD_VALUE_DURATION_HISTOGRAM: HistogramVec = register_histogram_vec!(
        "tikv_cdc_old_value_duration",
        "Bucketed histogram of cdc old value scan duration",
        &["tag"],
        exponential_buckets(0.0001, 2.0, 20).unwrap()
    )
    .unwrap();
    pub static ref CDC_RESOLVED_TS_ADVANCE_METHOD: IntGauge = register_int_gauge!(
        "tikv_cdc_resolved_ts_advance_method",
        "Resolved Ts advance method, 0 = advanced through raft command, 1 = advanced through store RPC"
    )
    .unwrap();
    pub static ref CDC_GRPC_ACCUMULATE_MESSAGE_BYTES: IntCounterVec = register_int_counter_vec!(
        "tikv_cdc_grpc_message_sent_bytes",
        "Accumulated bytes of sent CDC gRPC messages",
        &["type"]
    )
    .unwrap();

    pub static ref CDC_ROCKSDB_PERF_COUNTER: IntCounterVec = register_int_counter_vec!(
        "tikv_cdc_rocksdb_perf",
        "Total number of RocksDB internal operations from PerfContext",
        &["metric"]
    )
    .unwrap();

    pub static ref CDC_RAW_OUTLIER_RESOLVED_TS_GAP: Histogram = register_histogram!(
        "tikv_cdc_raw_outlier_resolved_ts_gap_seconds",
        "Bucketed histogram of the gap between cdc raw outlier resolver_ts and current tso",
        exponential_buckets(1.0, 2.0, 15).unwrap() // outlier threshold is 60s by default.
    )
    .unwrap();

    pub static ref CDC_ROCKSDB_PERF_COUNTER_STATIC: PerfCounter =
        auto_flush_from!(CDC_ROCKSDB_PERF_COUNTER, PerfCounter);
}
