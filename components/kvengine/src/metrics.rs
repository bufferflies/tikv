// Copyright 2022 TiKV Project Authors. Licensed under Apache-2.0.

use std::time::Instant;

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

pub fn flush_engine_properties(_engine: &Engine, _name: &str) {}

lazy_static! {
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
    pub static ref ENGINE_IA_MANAGER_SEGMENTS_MEMORY_SIZE: IntGauge = register_int_gauge!(
        "kv_engine_ia_manager_segments_memory_size",
        "Total memory size of IA manager segments",
    )
    .unwrap();
}

pub(crate) fn elapsed_secs(t: Instant) -> f64 {
    let d = Instant::now().saturating_duration_since(t);
    let nanos = f64::from(d.subsec_nanos());
    d.as_secs() as f64 + (nanos / 1_000_000_000.0)
}
