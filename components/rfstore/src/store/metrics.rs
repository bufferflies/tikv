// Copyright 2024 TiKV Project Authors. Licensed under Apache-2.0.

use lazy_static::lazy_static;
use prometheus::*;
use prometheus_static_metric::*;

make_auto_flush_static_metric! {
    pub label_enum RaftEntryFetchKind {
        async_fetch,
        sync_fetch,
        fallback_fetch,
        fetch_invalid,
        fetch_unused,
    }

    pub struct RaftEntryFetches : LocalIntCounter {
        "kind" => RaftEntryFetchKind,
    }
}

lazy_static! {
    pub static ref STORE_PROPOSE_SWITCH_MEM_TABLE_COUNTER: IntCounter = register_int_counter!(
        "rfstore_propose_switch_mem_table_counter",
        "Counter of rfstore propose switch mem table",
    )
    .unwrap();
    pub static ref STORE_INGEST_CONVERT_TASK_STATUS: GaugeVec = register_gauge_vec!(
        "rfstore_ingest_convert_sst_task_status_counter",
        "Counter of rfstore ingest convert sst task status",
        &["status"]
    )
    .unwrap();
    pub static ref STORE_SYNC_AUX_WORKER_DURATION_HISTOGRAM: Histogram = register_histogram!(
        "rfstore_sync_aux_worker_duration_seconds",
        "Bucketed histogram of syncing aux workers",
        exponential_buckets(0.00001, 2.0, 26).unwrap()
    )
    .unwrap();

    // Raft entry fetches categorized by kind: async, sync, fallback, invalid, unused
    pub static ref RAFT_ENTRY_FETCHES_VEC: IntCounterVec = register_int_counter_vec!(
        "rfstore_raft_entry_fetches_total",
        "Number of raft entry fetches by type",
        &["kind"]
    )
    .unwrap();
    pub static ref RAFT_ENTRY_FETCHES: RaftEntryFetches =
        auto_flush_from!(RAFT_ENTRY_FETCHES_VEC, RaftEntryFetches);

    // Duration of async raft entry fetch tasks from schedule to result consumption
    pub static ref RAFT_ENTRY_FETCH_TASK_DURATION_HISTOGRAM: Histogram = register_histogram!(
        "rfstore_raft_entry_fetch_task_duration_seconds",
        "Duration of async raft entry fetch tasks",
        vec![
            0.001, 0.005, 0.01, 0.025, 0.05,
            0.1, 0.25, 0.5, 1.0, 2.5,
            5.0, 10.0
        ]
    )
    .unwrap();

    pub static ref RAFT_ENTRY_FETCHES_TASK_DURATION_HISTOGRAM: Histogram =
        register_histogram!(
            "tikv_rfstore_entry_fetches_task_duration_seconds",
            "Bucketed histogram of raft entry fetches task duration.",
            exponential_buckets(0.0005, 2.0, 21).unwrap()  // 500us ~ 8.7m
        ).unwrap();
}
