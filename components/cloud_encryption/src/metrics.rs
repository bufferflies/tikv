// Copyright 2025 TiKV Project Authors. Licensed under Apache-2.0.

use lazy_static::*;
use prometheus::*;
use prometheus_static_metric::*;

make_static_metric! {
    pub label_enum MasterKeyOpsType {
        generate,
        decrypt,
    }

    pub struct MasterKeyOpsDurationVec: Histogram {
        "type" => MasterKeyOpsType,
    }
}

lazy_static! {
    pub static ref MASTER_KEY_OPS_DURATION: MasterKeyOpsDurationVec =
        register_static_histogram_vec!(
            MasterKeyOpsDurationVec,
            "tikv_master_key_ops_duration_seconds",
            "Bucketed histogram of master key operation duration",
            &["type"],
            exponential_buckets(0.00005, 1.8, 26).unwrap()
        )
        .unwrap();
    pub static ref ENCRYPTION_KEY_GET_DURATION: Histogram = register_histogram!(
        "tikv_encryption_key_get_duration",
        "Bucketed histogram of encryption key get duration",
        exponential_buckets(0.00005, 1.8, 26).unwrap()
    )
    .unwrap();
    pub static ref ENCRYPTION_KEY_SWITCH_COUNTER: IntCounter = register_int_counter!(
        "tikv_encryption_key_switch_count",
        "Total number of encryption key switches",
    )
    .unwrap();
    pub static ref ENCRYPTION_REGION_COUNT_GAUGE: IntGauge = register_int_gauge!(
        "tikv_encryption_region_count",
        "Total number of regions with encryption enabled",
    )
    .unwrap();
    pub static ref ENCRYPTION_KEYSPACE_ACTIVE_COUNT_GAUGE: IntGauge = register_int_gauge!(
        "tikv_encryption_keyspace_active_count",
        "Number of keyspaces currently in use with encryption enabled",
    )
    .unwrap();
    pub static ref ENCRYPTION_KEYSPACE_REGISTERED_COUNT_GAUGE: IntGauge = register_int_gauge!(
        "tikv_encryption_keyspace_registered_count",
        "Cumulative number of keyspaces with encryption enabled that have ever registered",
    )
    .unwrap();
}
