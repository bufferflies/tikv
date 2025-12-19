// Copyright 2023 TiKV Project Authors. Licensed under Apache-2.0.

use lazy_static::lazy_static;
use prometheus::*;
use tikv_util::info;

use crate::checkpoint::LoadDataWorkerState;

lazy_static! {
    pub static ref LOAD_DATA_CREATING_FILE_COUNTER: IntCounterVec = register_int_counter_vec!(
        "tikv_worker_load_data_creating_file_counter",
        "Total count file of the worker creating",
        &["task_id", "keyspace_id", "file_type"],
    )
    .unwrap();
    pub static ref LOAD_DATA_INGEST_SST_FILE_COUNTER: IntCounterVec = register_int_counter_vec!(
        "tikv_worker_load_data_ingest_sst_file_counter",
        "Total count sst of the worker ingesting to TiKV",
        &["task_id", "keyspace_id"],
    )
    .unwrap();
    pub static ref LOAD_DATA_WRU_COST_COUNTER: IntCounterVec = register_int_counter_vec!(
        "tikv_worker_load_data_wru_cost_counter",
        "Total count of the write request unit cost for load data",
        &["task_id", "keyspace_id"],
    )
    .unwrap();
    pub static ref LOAD_DATA_TASK_STATE: GaugeVec = register_gauge_vec!(
        "tikv_worker_load_data_task_state",
        "load data task state changes and corresponding times",
        &["task_id", "state"],
    )
    .unwrap();
    pub static ref LOAD_DATA_INGEST_RANEG_GROUP_FAILURES_COUNTER: IntCounterVec =
        register_int_counter_vec!(
            "tikv_worker_load_data_ingest_range_group_failures_counter",
            "Total count of worker scaler failures in ingesting range group",
            &["task_id"],
        )
        .unwrap();
    pub static ref LOAD_DATA_SPLIT_REGION_FAILURES_COUNTER: IntCounterVec =
        register_int_counter_vec!(
            "tikv_worker_load_data_split_region_failures_counter",
            "Total count of worker scaler failures in spliting region",
            &["task_id"],
        )
        .unwrap();
    pub static ref LOAD_DATA_GET_SHARD_META_FAILURES_COUNTER: IntCounter = register_int_counter!(
        "tikv_worker_load_data_get_shard_meta_failures_counter",
        "Total count of worker scaler failures in getting shard meta",
    )
    .unwrap();
}

pub fn remove_metrics(task_id: &str, keyspace_id: &str) {
    info!("remove cancelled task metrics, task_id: {}", task_id);

    let _ =
        LOAD_DATA_CREATING_FILE_COUNTER.remove_label_values(&[task_id, keyspace_id, "l0_kv_pairs"]);
    let _ =
        LOAD_DATA_CREATING_FILE_COUNTER.remove_label_values(&[task_id, keyspace_id, "l1_kv_pairs"]);
    let _ = LOAD_DATA_CREATING_FILE_COUNTER.remove_label_values(&[task_id, keyspace_id, "sst"]);

    let _ = LOAD_DATA_INGEST_SST_FILE_COUNTER.remove_label_values(&[task_id, keyspace_id]);
    let _ = LOAD_DATA_WRU_COST_COUNTER.remove_label_values(&[task_id, keyspace_id]);
    let _ = LOAD_DATA_INGEST_RANEG_GROUP_FAILURES_COUNTER.remove_label_values(&[task_id]);
    let _ = LOAD_DATA_SPLIT_REGION_FAILURES_COUNTER.remove_label_values(&[task_id]);

    let _ = LOAD_DATA_TASK_STATE
        .remove_label_values(&[task_id, LoadDataWorkerState::InitTask.as_str()]);
    let _ = LOAD_DATA_TASK_STATE
        .remove_label_values(&[task_id, LoadDataWorkerState::AddingChunks.as_str()]);
    let _ = LOAD_DATA_TASK_STATE
        .remove_label_values(&[task_id, LoadDataWorkerState::BuildingSst.as_str()]);
    let _ = LOAD_DATA_TASK_STATE
        .remove_label_values(&[task_id, LoadDataWorkerState::IngestingSst.as_str()]);
    let _ = LOAD_DATA_TASK_STATE
        .remove_label_values(&[task_id, LoadDataWorkerState::IngestedSst.as_str()]);
    let _ = LOAD_DATA_TASK_STATE.remove_label_values(&[task_id, "cancel"]);
}
