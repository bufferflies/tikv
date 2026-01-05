// Copyright 2025 TiKV Project Authors. Licensed under Apache-2.0.

use lazy_static::lazy_static;
use prometheus::*;
use prometheus_static_metric::*;

use crate::types::{InspectDuration, InspectFactor};

make_static_metric! {
    pub label_enum Factor {
        kvdb,
        raft,
        network,
    }

    pub label_enum Stage {
        wait,
        process,
    }

    pub struct InspectDurationHistogramVec: Histogram {
        "type" => Factor,
        "stage" => Stage,
    }

    pub struct InspectSlowScoreGaugeVec: IntGauge {
        "type" => Factor,
    }
}

lazy_static! {
    pub static ref STORE_INSPECT_DISK_DURATION_HISTOGRAM: InspectDurationHistogramVec =
        register_static_histogram_vec!(
            InspectDurationHistogramVec,
            "store_inspect_disk_duration_seconds",
            "Bucketed histogram of inspect duration.",
            &["type", "stage"],
            exponential_buckets(0.00001, 2.0, 26).unwrap()
        )
        .unwrap();
    pub static ref STORE_INSPECT_NETWORK_DURATION_HISTOGRAM: HistogramVec =
        register_histogram_vec!(
            "store_inspect_network_duration_seconds",
            "Bucketed histogram of inspect network duration.",
            &["target"],
            exponential_buckets(0.00001, 2.0, 26).unwrap()
        )
        .unwrap();
    pub static ref STORE_SLOW_SCORE_GAUGE: InspectSlowScoreGaugeVec =
        register_static_int_gauge_vec!(
            InspectSlowScoreGaugeVec,
            "store_slow_score",
            "Slow score of the store.",
            &["type"]
        )
        .unwrap();
}

fn convert_to_factor(t: InspectFactor) -> Factor {
    match t {
        InspectFactor::KvDisk => Factor::kvdb,
        InspectFactor::RaftDisk => Factor::raft,
        InspectFactor::Network => Factor::network,
        #[allow(unreachable_patterns)]
        unexpected => panic!("unexpected name {:?}", unexpected),
    }
}

pub fn flush_store_inspect_disk_duration_metrics(t: InspectFactor, duration: InspectDuration) {
    let factor_enum = convert_to_factor(t);
    let inspector = STORE_INSPECT_DISK_DURATION_HISTOGRAM.get(factor_enum);
    inspector.wait.observe(tikv_util::time::duration_to_sec(
        duration.wait_duration.unwrap_or_default(),
    ));
    inspector.process.observe(tikv_util::time::duration_to_sec(
        duration.process_duration.unwrap_or_default(),
    ));
}

pub fn flush_store_inspect_network_duration_metrics(target: u64, dur_as_secs: f64) {
    STORE_INSPECT_NETWORK_DURATION_HISTOGRAM
        .with_label_values(&[&target.to_string()])
        .observe(dur_as_secs);
}

pub fn flush_store_inspect_slow_score_metrics(t: InspectFactor, value: f64) {
    let factor_enum = convert_to_factor(t);
    STORE_SLOW_SCORE_GAUGE.get(factor_enum).set(value as i64);
}
