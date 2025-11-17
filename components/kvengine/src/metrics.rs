// Copyright 2022 TiKV Project Authors. Licensed under Apache-2.0.

use std::time::Instant;

use lazy_static::lazy_static;
use prometheus::*;
use prometheus_static_metric::*;

use crate::*;

pub fn flush_engine_properties(kv_engine: &Engine, _name: &str) {
    let kv_all_shard_stats = kv_engine.get_all_shard_stats();
    Engine::update_region_huge_table_bytes_metrics(
        &kv_all_shard_stats,
        kv_engine.opts.max_mem_table_size,
    );
    let EngineStats {
        num_shards,
        num_initial_flushed_shard,
        num_active_shards,
        num_compacting_shards,
        num_pending_compaction_shards,
        num_has_del_prefixes_shards,
        num_inner_key_shards,
        ready_destroy_range_shards,
        mem_tables_count,
        mem_tables_size: _, // Exported by PdRunner::handle_store_heartbeat
        l0_tables_count,
        l0_tables_size,
        blob_tables_count,
        in_use_blob_size,
        total_blob_size,
        partial_l0_count,
        partial_blob_count,
        partial_ln_count,
        cf_level_files: cf_level_num_files,
        cf_level_sizes: cf_level_total_sizes,
        index_size,
        in_mem_index_size: _, // Exported by PdRunner::handle_store_heartbeat
        filter_size,
        in_mem_filter_size: _, // Exported by PdRunner::handle_store_heartbeat
        open_files,
        max_ts,
        entries,
        old_entries,
        tombs,
        kv_size,
        txn_file_locks,
        columnar_levels,
        vector_indexes,
        top_10_write,
        ia:
            StorageClassStats {
                num_shards: ia_num_shards,
                num_tables: ia_num_tables,
                data_size: _, // Exported by PdRunner::handle_store_heartbeat
                kv_size: _,   // Exported by PdRunner::handle_store_heartbeat
            },
    } = Engine::get_engine_stats(kv_all_shard_stats);

    ENGINE_SHARDS.set(num_shards as i64);
    ENGINE_INITIAL_FLUSHED_SHARDS.set(num_initial_flushed_shard as i64);
    ENGINE_ACTIVE_SHARDS.set(num_active_shards as i64);
    ENGINE_COMPACTING_SHARDS.set(num_compacting_shards as i64);
    ENGINE_PENDING_COMPACTION_SHARDS.set(num_pending_compaction_shards as i64);
    ENGINE_HAS_DEL_PREFIXES_SHARDS.set(num_has_del_prefixes_shards as i64);
    ENGINE_INNER_KEY_SHARDS.set(num_inner_key_shards as i64);
    ENGINE_READY_DESTROY_RANGE_SHARDS.set(ready_destroy_range_shards.len() as i64);
    ENGINE_MEM_TABLES_COUNT.set(mem_tables_count as i64);
    ENGINE_L0_TABLES_COUNT.set(l0_tables_count as i64);
    ENGINE_L0_TABLES_SIZE.set(l0_tables_size as i64);
    ENGINE_BLOB_TABLES_COUNT.set(blob_tables_count as i64);
    ENGINE_IN_USE_BLOB_SIZE.set(in_use_blob_size as i64);
    ENGINE_TOTAL_BLOB_SIZE.set(total_blob_size as i64);
    ENGINE_PARTIAL_L0_COUNT.set(partial_l0_count as i64);
    ENGINE_PARTIAL_BLOB_COUNT.set(partial_blob_count as i64);
    ENGINE_PARTIAL_LN_COUNT.set(partial_ln_count as i64);
    for (cf, level_files) in cf_level_num_files.iter().enumerate() {
        for (level, num_files) in level_files.iter().enumerate() {
            ENGINE_CF_LEVEL_NUM_FILES
                .with_label_values(&[CF_NAMES[cf], &level.to_string()])
                .set(*num_files as i64);
        }
    }
    for (cf, level_size) in cf_level_total_sizes.iter().enumerate() {
        for (level, total_size) in level_size.iter().enumerate() {
            ENGINE_CF_LEVEL_TOTAL_SIZES
                .with_label_values(&[CF_NAMES[cf], &level.to_string()])
                .set(*total_size as i64);
        }
    }
    ENGINE_INDEX_SIZE.set(index_size as i64);
    ENGINE_FILTER_SIZE.set(filter_size as i64);
    ENGINE_OPEN_FILES.set(open_files);
    ENGINE_MAX_TS.set(max_ts as i64);
    ENGINE_ENTRIES.set(entries as i64);
    ENGINE_OLD_ENTRIES.set(old_entries as i64);
    ENGINE_TOMBSTONES.set(tombs as i64);
    ENGINE_KV_SIZE.set(kv_size as i64);
    ENGINE_TXN_FILE_LOCKS.set(txn_file_locks as i64);
    for (level, stats) in columnar_levels.iter().enumerate() {
        ENGINE_COLUMNAR_FILES_COUNT
            .with_label_values(&[&level.to_string()])
            .set(stats.num_files as i64);
        ENGINE_COLUMNAR_DATA_SIZE
            .with_label_values(&[&level.to_string()])
            .set(stats.data_size as i64);
    }
    ENGINE_VECTOR_INDEX_FILE_COUNT.set(vector_indexes.num_files as i64);
    ENGINE_VECTOR_INDEX_DATA_SIZE.set(vector_indexes.data_size as i64);
    for (rank, shard_stats) in top_10_write.iter().enumerate() {
        ENGINE_TOP_10_WRITE_SHARDS
            .with_label_values(&[&(rank + 1).to_string()])
            .set(shard_stats.id as i64);
    }
    ENGINE_IA_SHARDS.set(ia_num_shards as i64);
    ENGINE_IA_SST_TABLES.set(ia_num_tables as i64);
}

make_static_metric! {
    pub label_enum WriteFlowType {
        keys,
        bytes,
    }

    pub struct WriteFlowVec: Histogram {
        "type" => WriteFlowType,
    }
}

make_auto_flush_static_metric! {
    pub label_enum ChangeSetType {
        columnar_compaction,
        compaction,
        destroy_range,
        flush,
        ingest_files,
        initial_flush,
        major_compaction,
        restore_shard,
        trim_over_bound,
        truncate_ts,
        update_schema_meta,
        update_vector_index,
        update_storage_class,
    }
    pub struct ChangeSetVec: LocalIntCounter {
        "type" => ChangeSetType,
    }

    pub label_enum SeekType {
        seek,
        next,
        prev,
    }
    pub struct SeekDurationVec: LocalHistogram {
        "type" => SeekType,
    }
}

lazy_static! {
    pub static ref ENGINE_APPLY_CHANGE_SET_VEC: IntCounterVec = register_int_counter_vec!(
        "kvengine_apply_change_set_total",
        "Total number of kvengine apply change set.",
        &["type"]
    )
    .unwrap();
    pub static ref ENGINE_APPLY_CHANGE_SET_COUNTER: ChangeSetVec =
        auto_flush_from!(ENGINE_APPLY_CHANGE_SET_VEC, ChangeSetVec);
    pub static ref ENGINE_SHARDS: IntGauge = register_int_gauge!(
        "kv_engine_shards_total",
        "Total number of shards in the engine",
    )
    .unwrap();
    pub static ref ENGINE_INITIAL_FLUSHED_SHARDS: IntGauge = register_int_gauge!(
        "kv_engine_initial_flushed_shards",
        "Number of initial flushed shards in the engine",
    )
    .unwrap();
    pub static ref ENGINE_ACTIVE_SHARDS: IntGauge = register_int_gauge!(
        "kv_engine_active_shards",
        "Number of active shards in the engine",
    )
    .unwrap();
    pub static ref ENGINE_COMPACTING_SHARDS: IntGauge = register_int_gauge!(
        "kv_engine_compacting_shards",
        "Number of compacting shards in the engine",
    )
    .unwrap();
    pub static ref ENGINE_PENDING_COMPACTION_SHARDS: IntGauge = register_int_gauge!(
        "kv_engine_pending_compaction_shards",
        "Number of pending compaction shards in the engine",
    )
    .unwrap();
    pub static ref ENGINE_HAS_DEL_PREFIXES_SHARDS: IntGauge = register_int_gauge!(
        "kv_engine_has_del_prefixes_shards",
        "Number of shards with delete prefixes in the engine",
    )
    .unwrap();
    pub static ref ENGINE_INNER_KEY_SHARDS: IntGauge = register_int_gauge!(
        "kv_engine_inner_key_shards",
        "Number of shards with inner keys in the engine",
    )
    .unwrap();
    pub static ref ENGINE_READY_DESTROY_RANGE_SHARDS: IntGauge = register_int_gauge!(
        "kv_engine_ready_destroy_range_shards",
        "Number of shards ready to destroy range in the engine",
    )
    .unwrap();
    pub static ref ENGINE_MEM_TABLES_COUNT: IntGauge = register_int_gauge!(
        "kv_engine_mem_tables_count",
        "Number of memory tables in the engine",
    )
    .unwrap();
    pub static ref ENGINE_L0_TABLES_COUNT: IntGauge = register_int_gauge!(
        "kv_engine_l0_tables_count",
        "Number of L0 tables in the engine",
    )
    .unwrap();
    pub static ref ENGINE_L0_TABLES_SIZE: IntGauge = register_int_gauge!(
        "kv_engine_l0_tables_size_bytes",
        "Total size of L0 tables in bytes",
    )
    .unwrap();
    pub static ref ENGINE_BLOB_TABLES_COUNT: IntGauge = register_int_gauge!(
        "kv_engine_blob_tables_count",
        "Number of blob tables in the engine",
    )
    .unwrap();
    pub static ref ENGINE_IN_USE_BLOB_SIZE: IntGauge = register_int_gauge!(
        "kv_engine_in_use_blob_size_bytes",
        "In-use blob size in bytes",
    )
    .unwrap();
    pub static ref ENGINE_TOTAL_BLOB_SIZE: IntGauge = register_int_gauge!(
        "kv_engine_total_blob_size_bytes",
        "Total blob size in bytes",
    )
    .unwrap();
    pub static ref ENGINE_PARTIAL_L0_COUNT: IntGauge =
        register_int_gauge!("kv_engine_partial_l0_count", "Number of partial L0 tables",).unwrap();
    pub static ref ENGINE_PARTIAL_BLOB_COUNT: IntGauge = register_int_gauge!(
        "kv_engine_partial_blob_count",
        "Number of partial blob tables",
    )
    .unwrap();
    pub static ref ENGINE_PARTIAL_LN_COUNT: IntGauge =
        register_int_gauge!("kv_engine_partial_ln_count", "Number of partial Ln tables",).unwrap();
    pub static ref ENGINE_CF_LEVEL_NUM_FILES: IntGaugeVec = register_int_gauge_vec!(
        "kv_engine_cf_level_num_files",
        "Number of files in each level",
        &["cf", "level"]
    )
    .unwrap();
    pub static ref ENGINE_CF_LEVEL_TOTAL_SIZES: IntGaugeVec = register_int_gauge_vec!(
        "kv_engine_cf_level_total_sizes_bytes",
        "Total sizes of each level in bytes",
        &["cf", "level"]
    )
    .unwrap();
    pub static ref ENGINE_INDEX_SIZE: IntGauge =
        register_int_gauge!("kv_engine_index_size_bytes", "Total index size in bytes",).unwrap();
    pub static ref ENGINE_FILTER_SIZE: IntGauge =
        register_int_gauge!("kv_engine_filter_size_bytes", "Total filter size in bytes",).unwrap();
    pub static ref ENGINE_MAX_TS: IntGauge = register_int_gauge!(
        "kv_engine_max_ts",
        "Maximum timestamp for PiTR completion check",
    )
    .unwrap();
    pub static ref ENGINE_ENTRIES: IntGauge =
        register_int_gauge!("kv_engine_entries", "Total number of entries in the engine",).unwrap();
    pub static ref ENGINE_OLD_ENTRIES: IntGauge = register_int_gauge!(
        "kv_engine_old_entries",
        "Number of old entries in the engine",
    )
    .unwrap();
    pub static ref ENGINE_TOMBSTONES: IntGauge =
        register_int_gauge!("kv_engine_tombstones", "Number of tombstones in the engine",).unwrap();
    pub static ref ENGINE_KV_SIZE: IntGauge =
        register_int_gauge!("kv_engine_kv_size_bytes", "Total KV size in bytes",).unwrap();
    pub static ref ENGINE_TXN_FILE_LOCKS: IntGauge = register_int_gauge!(
        "kv_engine_txn_file_locks",
        "Number of transaction file locks",
    )
    .unwrap();
    pub static ref ENGINE_COLUMNAR_FILES_COUNT: IntGaugeVec = register_int_gauge_vec!(
        "kv_engine_columnar_files_count",
        "Number of columnar files in the engine",
        &["level"]
    )
    .unwrap();
    pub static ref ENGINE_COLUMNAR_DATA_SIZE: IntGaugeVec = register_int_gauge_vec!(
        "kv_engine_columnar_data_size_bytes",
        "Total size of columnar data in bytes",
        &["level"]
    )
    .unwrap();
    pub static ref ENGINE_VECTOR_INDEX_FILE_COUNT: IntGauge = register_int_gauge!(
        "kv_engine_vector_index_file_count",
        "Number of vector index files in the engine",
    )
    .unwrap();
    pub static ref ENGINE_VECTOR_INDEX_DATA_SIZE: IntGauge = register_int_gauge!(
        "kv_engine_vector_index_data_size_bytes",
        "Total size of vector indexes in bytes",
    )
    .unwrap();
    pub static ref ENGINE_TOP_10_WRITE_SHARDS: IntGaugeVec = register_int_gauge_vec!(
        "kv_engine_top_10_write_shards",
        "Number of top 10 write shards in the engine",
        &["rank"],
    )
    .unwrap();
}

lazy_static! {
    pub static ref ENGINE_GET_DURATION: Histogram = register_histogram!(
        "kv_engine_get_duration_seconds",
        "Bucketed histogram of KV Engine get duration",
        exponential_buckets(0.00005, 1.8, 26).unwrap()
    )
    .unwrap();
    pub static ref ENGINE_SEEK_DURATION_VEC: HistogramVec = register_histogram_vec!(
        "kv_engine_seek_duration_seconds",
        "Bucketed histogram of KV Engine seek duration",
        &["type"],
        exponential_buckets(0.00005, 1.8, 26).unwrap()
    )
    .unwrap();
    pub static ref ENGINE_SEEK_DURATION_STATIC: SeekDurationVec = auto_flush_from!(ENGINE_SEEK_DURATION_VEC, SeekDurationVec);
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
    pub static ref ENGINE_DFS_LOAD_MEMORY_USAGE: IntGauge = register_int_gauge!(
        "kv_engine_dfs_load_memory_usage_bytes",
        "kv engine dfs load memory usage in bytes",
    )
    .unwrap();
    pub static ref ENGINE_DFS_LOAD_MEMORY_WAIT_DURATION: Histogram = register_histogram!(
        "kv_engine_dfs_load_memory_wait_duration_seconds",
        "Histogram of wait duration when acquiring DFS load memory",
        exponential_buckets(0.001, 2.0, 20).unwrap() // 1ms ~ 524s
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
    pub static ref ENGINE_IA_SHARDS: IntGauge = register_int_gauge!(
        "kv_engine_ia_shards_total",
        "Total number of IA shards in the engine",
    )
    .unwrap();
    pub static ref ENGINE_IA_SST_TABLES: IntGauge = register_int_gauge!(
        "kv_engine_ia_sst_tables_total",
        "Total number of IA SST tables in the engine",
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
