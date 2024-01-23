// Copyright 2023 TiKV Project Authors. Licensed under Apache-2.0.

use lazy_static::lazy_static;
use prometheus::*;

lazy_static! {
    pub static ref LOAD_DATA_HANDLE_ADD_CHUNK_TIME_MILLIS: IntCounterVec =
        register_int_counter_vec!(
            "tikv_worker_load_data_handle_add_chunk_time_millis",
            "Total time taken to handle add chunk time",
            &["task_id"],
        )
        .unwrap();
    pub static ref LOAD_DATA_HANDLE_ADD_CHUNK_COUNTER: IntCounterVec = register_int_counter_vec!(
        "tikv_worker_load_data_handle_add_chunk_counter",
        "Total count of handle add chunks",
        &["task_id"],
    )
    .unwrap();
    pub static ref LOAD_DATA_BUILD_SST_TIME_MILLIS: IntCounterVec = register_int_counter_vec!(
        "tikv_worker_load_data_build_sst_time_millis",
        "Total time taken to build the sst files",
        &["task_id"],
    )
    .unwrap();
    pub static ref LOAD_DATA_BUILD_SST_COUNTER: IntCounterVec = register_int_counter_vec!(
        "tikv_worker_load_data_build_sst_counter",
        "Total count of build sst files",
        &["task_id"],
    )
    .unwrap();
    pub static ref LOAD_DATA_TASK_STATE: GaugeVec = register_gauge_vec!(
        "tikv_worker_load_data_task_state",
        "load data task state changes and corresponding times",
        &["task_id", "state"],
    )
    .unwrap();
}
