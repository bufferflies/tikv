use std::time::Duration;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

#[derive(Deserialize, Serialize, Default)]
pub struct ReplicaConfig {
    #[serde(default)]
    pub bdr_mode: bool,
    #[serde(default)]
    pub case_sensitive: bool,
    #[serde(default = "default_true")]
    pub check_gc_safe_point: bool,
    #[serde(default)]
    pub consistent: ConsistentConfig,
    #[serde(default)]
    pub enable_sync_point: bool,
    #[serde(default)]
    pub filter: FilterConfig,
    #[serde(default)]
    pub force_replicate: bool,
    #[serde(default)]
    pub ignore_ineligible_table: bool,
    #[serde(default)]
    pub memory_quota: u64,
    #[serde(default)]
    pub mounter: MounterConfig,
    #[serde(default)]
    pub sink: SinkConfig,
    #[serde(
        serialize_with = "duration_to_nanos_string",
        deserialize_with = "duration_from_nanos_string",
        default = "default_sync_point_interval"
    )]
    pub sync_point_interval: Duration,
    #[serde(
        serialize_with = "duration_to_nanos_string",
        deserialize_with = "duration_from_nanos_string",
        default = "default_sync_point_retention"
    )]
    pub sync_point_retention: Duration,
}

#[derive(Deserialize, Serialize, Default)]
pub struct ConsistentConfig {
    #[serde(default)]
    flush_interval: u64,
    #[serde(default)]
    level: String,
    #[serde(default)]
    max_log_size: u64,
    #[serde(default)]
    storage: String,
    #[serde(default)]
    use_file_backend: bool,
    #[serde(default)]
    encoding_worker_num: i32,
    #[serde(default)]
    flush_worker_num: i32,
    #[serde(default)]
    compression: String,
    #[serde(default = "default_flush_concurrency")]
    flush_concurrency: i32,
}

#[derive(Deserialize, Serialize, Default)]
pub struct FilterConfig {
    #[serde(default)]
    event_filters: Vec<EventFilter>,
    #[serde(default)]
    ignore_txn_start_ts: Vec<u64>,
    #[serde(default)]
    rules: Vec<String>,
}

#[derive(Deserialize, Serialize, Default)]
pub struct EventFilter {
    #[serde(default)]
    ignore_delete_value_expr: Vec<String>,
    #[serde(default)]
    ignore_event: Vec<String>,
    #[serde(default)]
    ignore_insert_value_expr: Vec<String>,
    #[serde(default)]
    ignore_sql: Vec<String>,
    #[serde(default)]
    ignore_update_new_value_expr: Vec<String>,
    #[serde(default)]
    ignore_update_old_value_expr: Vec<String>,
    #[serde(default)]
    matcher: Vec<String>,
}

#[derive(Deserialize, Serialize, Default)]
pub struct MounterConfig {
    #[serde(default = "default_mounter_worker_num")]
    worker_num: i32,
}

#[derive(Deserialize, Serialize, Default)]
pub struct SinkConfig {
    #[serde(default)]
    column_selectors: Vec<ColumnSelector>,
    #[serde(default)]
    csv: CsvConfig,
    #[serde(default)]
    date_separator: String,
    #[serde(default)]
    dispatchers: Vec<Dispatcher>,
    #[serde(default = "default_sink_encoder_concurrency")]
    encoder_concurrency: i32,
    #[serde(default)]
    protocol: String,
    #[serde(default)]
    schema_registry: String,
    #[serde(default)]
    terminator: String,
    #[serde(default)]
    transaction_atomicity: String,
    #[serde(default)]
    only_output_updated_columns: bool,
    #[serde(default)]
    cloud_storage_config: CloudStorageConfig,
    #[serde(default)]
    open: OpenConfig,
    #[serde(default)]
    debezium: DebeziumConfig,
}

#[derive(Deserialize, Serialize, Default)]
pub struct ColumnSelector {
    #[serde(default)]
    columns: Vec<String>,
    #[serde(default)]
    matcher: Vec<String>,
}

#[derive(Deserialize, Serialize, Default)]
pub struct CsvConfig {
    #[serde(default = "default_csv_delimiter")]
    delimiter: String,
    #[serde(default)]
    include_commit_ts: bool,
    #[serde(default = "default_csv_null")]
    null: String,
    #[serde(default = "default_csv_quote")]
    quote: String,
    #[serde(default = "default_binary_encoding")]
    binary_encoding_method: String,
}

#[derive(Deserialize, Serialize, Default)]
pub struct Dispatcher {
    #[serde(default)]
    matcher: Vec<String>,
    #[serde(default)]
    partition: String,
    #[serde(default)]
    topic: String,
}

#[derive(Deserialize, Serialize, Default)]
pub struct CloudStorageConfig {
    #[serde(default)]
    worker_count: i32,
    #[serde(default)]
    flush_interval: String,
    #[serde(default)]
    file_size: i32,
    #[serde(default)]
    file_expiration_days: i32,
    #[serde(default)]
    file_cleanup_cron_spec: String,
    #[serde(default)]
    flush_concurrency: i32,
    #[serde(default)]
    output_raw_change_event: bool,
}

#[derive(Deserialize, Serialize, Default)]
pub struct OpenConfig {
    #[serde(default = "default_true")]
    output_old_value: bool,
}

#[derive(Deserialize, Serialize, Default)]
pub struct DebeziumConfig {
    #[serde(default = "default_true")]
    output_old_value: bool,
}

fn default_true() -> bool {
    true
}
fn default_sync_point_interval() -> Duration {
    Duration::from_secs(10 * 60)
}
fn default_sync_point_retention() -> Duration {
    Duration::from_secs(24 * 60 * 60)
}
fn default_flush_concurrency() -> i32 {
    1
}
fn default_mounter_worker_num() -> i32 {
    16
}
fn default_sink_encoder_concurrency() -> i32 {
    16
}
fn default_csv_delimiter() -> String {
    ",".to_string()
}
fn default_csv_null() -> String {
    "\\N".to_string()
}
fn default_csv_quote() -> String {
    "\"".to_string()
}
fn default_binary_encoding() -> String {
    "base64".to_string()
}

fn duration_to_nanos_string<S>(d: &Duration, serializer: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    serializer.serialize_str(&d.as_nanos().to_string())
}

fn duration_from_nanos_string<'de, D>(deserializer: D) -> Result<Duration, D::Error>
where
    D: Deserializer<'de>,
{
    let s = String::deserialize(deserializer)?;
    s.parse::<u128>()
        .map(|nanos| Duration::from_nanos(nanos as u64))
        .map_err(serde::de::Error::custom)
}

impl ReplicaConfig {
    pub fn validate(&self) -> Result<(), String> {
        self.consistent.validate()?;
        self.filter.validate()?;
        self.mounter.validate()?;
        self.sink.validate()?;

        if self.sync_point_interval < Duration::from_secs(30) {
            return Err("sync_point_interval must be at least 30s".to_string());
        }

        Ok(())
    }
}

impl ConsistentConfig {
    pub fn validate(&self) -> Result<(), String> {
        Ok(())
    }
}

impl FilterConfig {
    pub fn validate(&self) -> Result<(), String> {
        Ok(())
    }
}

impl MounterConfig {
    pub fn validate(&self) -> Result<(), String> {
        Ok(())
    }
}

impl SinkConfig {
    pub fn validate(&self) -> Result<(), String> {
        Ok(())
    }
}
