// Copyright 2022 TiKV Project Authors. Licensed under Apache-2.0.

use std::time::Instant;

use lazy_static::lazy_static;
use prometheus::*;
use prometheus_static_metric::*;

use crate::*;

make_static_metric! {
    pub label_enum SeekType {
        seek,
        next,
        prev,
    }

    pub label_enum WriteFlowType {
        keys,
        bytes,
    }

    pub struct SeekDurationVec: Histogram {
        "type" => SeekType,
    }

    pub struct WriteFlowVec: Histogram {
        "type" => WriteFlowType,
    }
}

pub fn flush_engine_properties(_engine: &Engine, _name: &str) {}

lazy_static! {
    pub static ref ENGINE_GET_DURATION: Histogram = register_histogram!(
        "kv_engine_get_duration_seconds",
        "Bucketed histogram of KV Engine get duration",
        exponential_buckets(0.00005, 1.8, 26).unwrap()
    )
    .unwrap();
    pub static ref ENGINE_SEEK_DURATION: SeekDurationVec = register_static_histogram_vec!(
        SeekDurationVec,
        "kv_engine_seek_duration_seconds",
        "Bucketed histogram of KV Engine seek duration",
        &["type"],
        exponential_buckets(0.00005, 1.8, 26).unwrap()
    )
    .unwrap();
    pub static ref ENGINE_WRITE_DURATION: Histogram = register_histogram!(
        "kv_engine_write_duration_seconds",
        "Bucketed histogram of KV Engine write duration",
        exponential_buckets(0.00005, 1.8, 26).unwrap()
    )
    .unwrap();
    pub static ref ENGINE_WRITE_FLOW: WriteFlowVec = register_static_histogram_vec!(
        WriteFlowVec,
        "kv_engine_write_flow",
        "Bucketed histogram of KV Engine write flow",
        &["type"],
        exponential_buckets(8.0, 2.0, 20).unwrap()
    )
    .unwrap();
    pub static ref ENGINE_ARENA_GROW_DURATION_HISTOGRAM: Histogram = register_histogram!(
        "kv_engine_arena_grow_duration_seconds",
        "Bucketed histogram of KV Engine arena grow duration",
        exponential_buckets(0.00005, 1.8, 26).unwrap()
    )
    .unwrap();
    pub static ref ENGINE_CACHE_MISS: IntCounter =
        register_int_counter!("kv_engine_cache_miss", "kv engine cache miss",).unwrap();
    pub static ref ENGINE_LEVEL_WRITE_VEC: IntCounterVec = register_int_counter_vec!(
        "kv_engine_level_write_bytes",
        "Write bytes of kvengine of each level",
        &["level"]
    )
    .unwrap();
    pub static ref ENGINE_OPEN_FILES: IntGauge =
        register_int_gauge!("kv_engine_open_files", "kv engine open files",).unwrap();
    pub static ref ENGINE_LOAD_TABLE_FILES_ERROR: IntCounterVec = register_int_counter_vec!(
        "kv_engine_load_table_files_error",
        "Total number of kv engine load table files error",
        &["type"]
    )
    .unwrap();
    pub static ref ENGINE_THROTTLE_ACTION_COUNTER: IntCounterVec = register_int_counter_vec!(
        "kv_engine_throttle_action_total",
        "Total number of actions for flow control.",
        &["level", "type"]
    )
    .unwrap();
    pub static ref ENGINE_REGION_HUGE_MEM_TABLE_BYTES_HISTOGRAM: Histogram = register_histogram!(
        "kv_engine_region_huge_mem_table_bytes",
        "Histogram of huge mem table bytes for regions",
        exponential_buckets(1024.0 * 1024.0, 2.0, 20).unwrap()
    )
    .unwrap();
    pub static ref ENGINE_REGION_HUGE_L0_TABLE_BYTES_HISTOGRAM: Histogram = register_histogram!(
        "kv_engine_region_huge_l0_table_bytes",
        "Histogram of huge l0 table bytes for regions",
        exponential_buckets(1024.0 * 1024.0, 2.0, 20).unwrap()
    )
    .unwrap();
    pub static ref ENGINE_FREE_MEM_BYTES_HISTOGRAM: Histogram = register_histogram!(
        "kv_engine_free_mem_bytes",
        "Histogram of free mem bytes",
        exponential_buckets(1024.0 * 1024.0, 2.0, 20).unwrap() // 1MB ~ 1TB
    )
    .unwrap();
    pub static ref ENGINE_IA_MANAGER_SEGMENTS_DISK_SIZE: IntGauge = register_int_gauge!(
        "kv_engine_ia_manager_segments_disk_size",
        "Total disk usage size of IA manager segments",
    )
    .unwrap();
    pub static ref ENGINE_IA_MANAGER_SEGMENTS_MEMORY_SIZE: IntGauge = register_int_gauge!(
        "kv_engine_ia_manager_segments_memory_size",
        "Total memory size of IA manager segments",
    )
    .unwrap();
    pub static ref ENGINE_IA_MAIN_QUEUE_CAPACITY: IntGauge = register_int_gauge!(
        "kv_engine_ia_main_queue_capacity",
        "Capacity of IA main queue",
    )
    .unwrap();
    pub static ref ENGINE_IA_SMALL_QUEUE_CAPACITY: IntGauge = register_int_gauge!(
        "kv_engine_ia_small_queue_capacity",
        "Capacity of IA small queue",
    )
    .unwrap();
    pub static ref ENGINE_IA_READ_SEGMENT_DURATION_HISTOGRAM: Histogram = register_histogram!(
        "kv_engine_ia_read_segment_duration_seconds",
        "Histogram of read IA segment duration",
        exponential_buckets(5e-5, 2.0, 20).unwrap() // 50us ~ 26s
    )
    .unwrap();
    pub static ref ENGINE_IA_READ_SEGMENT_CACHE_MISS: IntCounter = register_int_counter!(
        "kv_engine_ia_read_segment_cache_miss",
        "Counter of read IA segment cache miss",
    )
    .unwrap();
    pub static ref ENGINE_COLUMNAR_TOO_MANY_UNCONVERTED_L0S: IntCounter = register_int_counter!(
        "kv_engine_columnar_too_many_unconverted_l0s",
        "Counter of columnar has too many unconverted L0s",
    )
    .unwrap();

    pub static ref ENGINE_REMOTE_COMPACT_EXCEED_MEMORY_LIMIT_COUNTER: IntCounter = register_int_counter!(
        "kv_engine_remote_compact_exceed_memory_limit_counter",
        "Total number of remote compaction requests that exceed memory limit",
    ).unwrap();

    pub static ref ENGINE_INGEST_LEVEL_HISTOGRAM: Histogram = register_histogram!(
        "kv_engine_ingest_level",
        "Histogram of levels where tables are ingested",
        vec![0.0, 1.0, 2.0, 3.0]
    ).unwrap();

    pub static ref ENGINE_PREPARE_LOAD_REMOTE_FILE: IntCounter = register_int_counter!(
        "kv_engine_prepare_load_remote_file",
        "Total number of remote files loaded during preparing changeset"
    ).unwrap();

    pub static ref ENGINE_PREPARE_USE_LOCAL_FILE: IntCounter = register_int_counter!(
        "kv_engine_prepare_use_local_file",
        "Total number local file hit during preparing changeset"
    ).unwrap();
}

pub(crate) fn elapsed_secs(t: Instant) -> f64 {
    let d = Instant::now().saturating_duration_since(t);
    let nanos = f64::from(d.subsec_nanos());
    d.as_secs() as f64 + (nanos / 1_000_000_000.0)
}
