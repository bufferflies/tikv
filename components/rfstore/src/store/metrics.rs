// Copyright 2024 TiKV Project Authors. Licensed under Apache-2.0.

use lazy_static::lazy_static;
use prometheus::*;

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
}
