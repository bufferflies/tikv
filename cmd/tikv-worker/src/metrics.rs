// Copyright 2023 TiKV Project Authors. Licensed under Apache-2.0.

use lazy_static::lazy_static;
use prometheus::*;

lazy_static! {
    pub static ref CPU_CORES_QUOTA_GAUGE: Gauge = register_gauge!(
        "tikv_worker_cpu_cores_quota",
        "Total CPU cores quota for TiKV worker"
    )
    .unwrap();
}
