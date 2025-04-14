use lazy_static::lazy_static;
use prometheus::*;
use prometheus_static_metric::*;

make_auto_flush_static_metric! {
    pub label_enum StatusReqKind {
        sync_region,
        sync_region_by_id,
        kvengine,
        rf_wal_chunk,
        rf_backup,
        truncate_ts,
        restore_shard,
        ingest_files,
        schema_file,
    }

    pub struct StatusReqHistogram: LocalHistogram {
        "req" => StatusReqKind,
    }
}

lazy_static! {
    pub static ref STATUS_REQ_HISTOGRAM_VEC: HistogramVec = register_histogram_vec!(
        "tikv_status_request_duration_seconds",
        "Bucketed histogram of status request duration",
        &["req"],
        exponential_buckets(0.00001, 2.0, 26).unwrap()
    )
    .unwrap();
    pub static ref STATUS_REQ_HISTOGRAM_STATIC: StatusReqHistogram =
        auto_flush_from!(STATUS_REQ_HISTOGRAM_VEC, StatusReqHistogram);
}
