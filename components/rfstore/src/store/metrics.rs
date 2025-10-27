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
        "type" => RaftEntryFetchKind,
    }
}

make_static_metric! {
    pub struct StoreBusyOnApplyRegionsGaugeVec: IntGauge {
        "type" => {
            busy_apply_peers,
            completed_apply_peers,
        },
    }

    pub struct StoreBusyStateGaugeVec: IntGauge {
        "type" => {
            raftstore_busy,
            applystore_busy,
        },
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
        &["type"]
    )
    .unwrap();
    pub static ref RAFT_ENTRY_FETCHES: RaftEntryFetches =
        auto_flush_from!(RAFT_ENTRY_FETCHES_VEC, RaftEntryFetches);

    pub static ref RAFT_ENTRY_FETCHES_TASK_DURATION_HISTOGRAM: Histogram =
        register_histogram!(
            "rfstore_raft_entry_fetches_task_duration_seconds",
            "Bucketed histogram of raft entry fetches task duration.",
            exponential_buckets(0.0005, 2.0, 21).unwrap()  // 500us ~ 8.7m
        ).unwrap();


    pub static ref STORE_BUSY_ON_APPLY_REGIONS_GAUGE_VEC: StoreBusyOnApplyRegionsGaugeVec =
        register_static_int_gauge_vec!(
            StoreBusyOnApplyRegionsGaugeVec,
            "tikv_raftstore_busy_on_apply_region_total",
            "Total number of regions busy on apply or complete apply.",
            &["type"]
        ).unwrap();

    pub static ref STORE_PROCESS_BUSY_GAUGE_VEC: StoreBusyStateGaugeVec =
        register_static_int_gauge_vec!(
            StoreBusyStateGaugeVec,
            "tikv_raftstore_process_busy",
            "Is raft process busy or not",
            &["type"]
        ).unwrap();
}
