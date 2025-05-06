use lazy_static::lazy_static;
use prometheus::*;

lazy_static! {
    pub static ref STATUS_REQUEST_DURATION: HistogramVec = register_histogram_vec!(
        "tikv_status_server_request_duration_seconds",
        "Bucketed histogram of TiKV status server request duration",
        &["method", "path"],
        exponential_buckets(0.0001, 2.0, 24).unwrap() // 0.1ms ~ 1677.7s
    )
    .unwrap();
}
