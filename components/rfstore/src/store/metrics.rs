// Copyright 2024 TiKV Project Authors. Licensed under Apache-2.0.

use lazy_static::lazy_static;
use prometheus::*;

lazy_static! {
    pub static ref STORE_PROPOSE_SWITCH_MEM_TABLE_COUNTER: IntCounter = register_int_counter!(
        "rfstore_propose_switch_mem_table_counter",
        "Counter of rfstore propose switch mem table",
    )
    .unwrap();
    pub static ref IDLE_PEER_COUNT: IntGauge =
        register_int_gauge!("rfstore_idle_peers_count", "The number of idle peers").unwrap();
    pub static ref LOCK_CACHE_LEN: IntGauge =
        register_int_gauge!("rfstore_lock_cache_len", "The length of lock cache").unwrap();
    pub static ref LOCK_CACHE_CAPCITY: IntGauge =
        register_int_gauge!("rfstore_lock_cache_capacity", "The capacity of lock cache").unwrap();
}
