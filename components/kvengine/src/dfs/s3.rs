// Copyright 2022 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    convert::TryFrom,
    fmt::{Debug, Formatter},
    ops::{Deref, DerefMut},
    path::{Path, PathBuf},
    sync::Arc,
    time::Duration,
};

use async_trait::async_trait;
use bstr::ByteSlice;
use bytes::{Buf, Bytes};
use engine_traits::{GetObjectOptions, ListObjectContent, ObjectStorage};
use fail::fail_point;
use farmhash::fingerprint64;
use futures::{Stream, StreamExt};
use http::{header::CONTENT_RANGE, StatusCode};
use hyper_tls::HttpsConnector;
use regex::Regex;
use rusoto_core::{
    param::{Params, ServiceParams},
    request::{BufferedHttpResponse, HttpResponse},
    signature::SignedRequest,
    HttpClient, HttpDispatchError, Region, RusotoError,
};
use rusoto_s3::{
    CopyObjectError, DeleteObjectError, GetObjectError, GetObjectTaggingError, HeadObjectError,
    ListObjectsV2Error, PutObjectError,
};
use tikv_util::{sys::thread::ThreadBuildWrapper, time::Instant};
use tokio::{io::AsyncWriteExt, runtime::Runtime};

use crate::dfs::{
    self, config::Config, Dfs, Error, FileType, MetricsFileType, Options, KVENGINE_DFS_LATENCY,
    KVENGINE_DFS_LATENCY_WITH_RETRY, KVENGINE_DFS_REQUEST_COUNTER, KVENGINE_DFS_RETRY_COUNTER,
    KVENGINE_DFS_THROUGHPUT,
};

const MAX_RETRY_COUNT: u32 = 9;
const RETRY_SLEEP_MS: u64 = 500;
const CONNECTION_TIMEOUT: Duration = Duration::from_secs(5);
const DISPATCH_TIMEOUT: Duration = Duration::from_secs(60);
const READ_BODY_TIMEOUT: Duration = Duration::from_secs(60);

pub const STORAGE_CLASS_DEFAULT: &str = STORAGE_CLASS_INTELLIGENT_TIERING;
pub const STORAGE_CLASS_INTELLIGENT_TIERING: &str = "INTELLIGENT_TIERING";
pub const STORAGE_CLASS_STANDARD: &str = "STANDARD";
pub const STORAGE_CLASS_STANDARD_IA: &str = "STANDARD_IA";
pub const STORAGE_CLASS_GLACIER_IR: &str = "GLACIER_IR";

const AWS_DOMAIN_STRING: &str = "amazonaws";

const SMALL_FILE_THRESHOLD_BYTES: u64 = 1024 * 1024; // 1MB

#[derive(Clone)]
pub struct S3Fs {
    core: Arc<S3FsCore>,
}

impl S3Fs {
    pub fn new(
        prefix: String,
        endpoint: String,
        key_id: String,
        secret_key: String,
        region: String,
        bucket: String,
    ) -> Self {
        let core = Arc::new(S3FsCore::new(
            endpoint, key_id, secret_key, region, bucket, prefix,
        ));
        Self { core }
    }

    pub fn new_from_config(conf: Config) -> Self {
        Self::new(
            conf.prefix,
            conf.s3_endpoint,
            conf.s3_key_id,
            conf.s3_secret_key,
            conf.s3_region,
            conf.s3_bucket,
        )
    }

    pub fn new_with_runtime_from_config(runtime: Arc<Runtime>, conf: Config) -> Self {
        let core = Arc::new(S3FsCore::new_with_runtime(
            runtime,
            conf.s3_endpoint,
            conf.s3_key_id,
            conf.s3_secret_key,
            conf.s3_region,
            conf.s3_bucket,
            conf.prefix,
        ));
        Self { core }
    }

    #[cfg(any(test, feature = "testexport"))]
    pub fn new_for_test(
        runtime: Runtime,
        rusoto_s3c: rusoto_core::Client,
        aws_s3c: aws_sdk_s3::Client,
        bucket: String,
        prefix: String,
    ) -> Self {
        Self {
            core: Arc::new(S3FsCore::new_with_runtime_and_s3_client(
                Arc::new(runtime),
                rusoto_s3c,
                aws_s3c,
                "".to_string(),
                false,
                "".to_string(),
                "local".to_string(),
                bucket,
                prefix,
            )),
        }
    }
}

impl Deref for S3Fs {
    type Target = S3FsCore;

    fn deref(&self) -> &Self::Target {
        &self.core
    }
}

pub struct S3FsCore {
    rusoto_s3c: rusoto_core::Client,
    aws_s3c: aws_sdk_s3::Client,
    hostname: String,
    region: Region,
    bucket: String,
    prefix: String,
    runtime: Arc<tokio::runtime::Runtime>,
    virtual_host: bool,
    // `rusoto` uses https by default, but cse uses http as default protocol.
    // this marker indicates the original endpoint has no protocol hence http should be used.
    use_http: bool,
}

type RusotoProvider = Arc<dyn rusoto_credential::ProvideAwsCredentials + Send + Sync>;
type AwsProvider = aws_credential_types::provider::SharedCredentialsProvider;

// Introduce the official AWS SDK to replace rusoto (now in maintenance mode).
// This base provider encapsulates both SDKs: existing code can continue using
// rusoto, while new code should adopt the official AWS SDK.
// The long-term goal is to fully migrate away from rusoto once all legacy usage
// is retired.
pub struct CredProviders {
    pub rusoto: BoxedRusotoProvider,
    pub aws: AwsProvider,
}

#[derive(Clone)]
pub struct BoxedRusotoProvider(RusotoProvider);

#[async_trait::async_trait]
impl rusoto_credential::ProvideAwsCredentials for BoxedRusotoProvider {
    async fn credentials(
        &self,
    ) -> Result<rusoto_credential::AwsCredentials, rusoto_credential::CredentialsError> {
        self.0.credentials().await
    }
}

fn run_async_on_fresh_thread<Fut, T>(fut: Fut) -> T
where
    Fut: std::future::Future<Output = T> + Send + 'static,
    T: Send + 'static,
{
    let (tx, rx) = std::sync::mpsc::sync_channel::<T>(1);

    std::thread::spawn(move || {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("build tokio current-thread runtime");

        let out = rt.block_on(fut);
        let _ = tx.send(out);
    });

    rx.recv().expect("async thread panicked or was killed")
}

impl S3FsCore {
    fn build_provider(endpoint: String, key_id: String, secret_key: String) -> CredProviders {
        let rusoto: RusotoProvider = if !key_id.is_empty() {
            Arc::new(rusoto_credential::StaticProvider::new(
                key_id.to_owned(),
                secret_key.to_owned(),
                None,
                None,
            ))
        } else if endpoint.contains(aliyun::DOMAIN_STRING) {
            Arc::new(aliyun::new_credential_provider().expect("failed to get aliyun provider"))
        } else {
            Arc::new(aws::new_credentials_provider_rusoto_wrapper())
        };

        let aws: AwsProvider = if !key_id.is_empty() {
            aws_credential_types::provider::SharedCredentialsProvider::new(
                aws_credential_types::Credentials::from_keys(key_id, secret_key, None),
            )
        } else {
            // use aws official default provider to handle both aliyun and aws.
            let aws_client = aws::new_http_client();
            aws_credential_types::provider::SharedCredentialsProvider::new(
                aws::new_credentials_provider(aws_client),
            )
        };
        CredProviders {
            rusoto: BoxedRusotoProvider(rusoto),
            aws,
        }
    }

    fn new_s3_client(
        endpoint: String,
        key_id: String,
        secret_key: String,
        region: String,
        bucket: String,
    ) -> (rusoto_core::Client, aws_sdk_s3::Client, String, bool) {
        let mut config = rusoto_core::HttpConfig::new();
        config.read_buf_size(256 * 1024);
        let endpoint = if endpoint.is_empty() {
            format!("http://s3.{}.amazonaws.com", region.as_str())
        } else {
            endpoint
        };
        let use_tls = endpoint.starts_with("https");
        let mut http_connector = hyper::client::connect::HttpConnector::new();
        http_connector.set_connect_timeout(Some(CONNECTION_TIMEOUT));
        let providers = Self::build_provider(endpoint.clone(), key_id.clone(), secret_key.clone());
        let rusoto_s3c = if use_tls {
            let https_connector = HttpsConnector::new_with_connector(http_connector);
            let http_client = HttpClient::from_connector_with_config(https_connector, config);
            rusoto_core::Client::new_with(providers.rusoto, http_client)
        } else {
            let http_client = HttpClient::from_connector_with_config(http_connector, config);
            rusoto_core::Client::new_with(providers.rusoto, http_client)
        };

        let no_schema_endpoint = endpoint
            .find("://")
            .map(|p| &endpoint[p + 3..])
            .unwrap_or(&endpoint);
        // local deployed s3 service like minio does not support virtual host
        // addressing, it always has a port defined at the end.
        let virtual_host = !no_schema_endpoint.contains(':');
        let hostname = if virtual_host {
            format!("{}.{}", &bucket, no_schema_endpoint)
        } else {
            no_schema_endpoint.to_string()
        };

        let aws_s3c = {
            let aws_client = aws::new_http_client();
            let mut loader = aws_config::defaults(aws_config::BehaviorVersion::latest())
                .credentials_provider(providers.aws);
            if !region.is_empty() {
                loader = loader.region(aws_config::Region::new(region.clone()));
            }
            if !endpoint.is_empty() {
                loader = loader.endpoint_url(endpoint.clone());
            }
            loader = loader.http_client(aws_client);
            let sdk_config = run_async_on_fresh_thread(loader.load());
            let builder =
                aws_sdk_s3::config::Builder::from(&sdk_config).force_path_style(!virtual_host);
            aws_sdk_s3::Client::from_conf(builder.build())
        };

        (rusoto_s3c, aws_s3c, hostname, virtual_host)
    }

    #[inline]
    fn new_with_runtime_and_s3_client(
        runtime: Arc<tokio::runtime::Runtime>,
        rusoto_s3c: rusoto_core::Client,
        aws_s3c: aws_sdk_s3::Client,
        hostname: String,
        virtual_host: bool,
        endpoint: String,
        region: String,
        bucket: String,
        mut prefix: String,
    ) -> Self {
        if prefix.is_empty() {
            prefix.push_str("default")
        }
        let use_http = !endpoint.starts_with("https://");
        let region = Region::Custom {
            name: region,
            endpoint,
        };
        Self {
            rusoto_s3c,
            aws_s3c,
            hostname,
            region,
            bucket,
            prefix,
            runtime,
            virtual_host,
            use_http,
        }
    }

    pub fn new(
        endpoint: String,
        key_id: String,
        secret_key: String,
        region: String,
        bucket: String,
        prefix: String,
    ) -> Self {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .thread_name("s3-client")
            .with_sys_hooks()
            .build()
            .unwrap();
        Self::new_with_runtime(
            Arc::new(runtime),
            endpoint,
            key_id,
            secret_key,
            region,
            bucket,
            prefix,
        )
    }

    pub fn new_with_runtime(
        runtime: Arc<tokio::runtime::Runtime>,
        endpoint: String,
        key_id: String,
        secret_key: String,
        region: String,
        bucket: String,
        prefix: String,
    ) -> Self {
        let (rusoto_s3, aws_s3c, hostname, virtual_host) = Self::new_s3_client(
            endpoint.clone(),
            key_id,
            secret_key,
            region.clone(),
            bucket.clone(),
        );
        Self::new_with_runtime_and_s3_client(
            runtime,
            rusoto_s3,
            aws_s3c,
            hostname,
            virtual_host,
            endpoint,
            region,
            bucket,
            prefix,
        )
    }

    pub fn rel_file_key(&self, file_id: u64, file_type: FileType) -> String {
        let idx = (fingerprint64(file_id.to_le_bytes().as_slice())) as u8;
        match file_type {
            FileType::Sst => {
                format!("{:02x}/{:016x}.sst", idx, file_id)
            }
            FileType::Blob => {
                format!("blob/{:02x}/{:016x}.blob", idx, file_id)
            }
            FileType::TxnChunk => {
                format!("txn/{:02x}/{:016x}.txn", idx, file_id)
            }
            FileType::Schema => {
                format!("schema/{:02x}/{:016x}.schema", idx, file_id)
            }
            FileType::Columnar => {
                format!("col/{:02x}/{:016x}.col", idx, file_id)
            }
            FileType::VectorIndex => {
                format!("vec/{:02x}/{:016x}.vec", idx, file_id)
            }
            FileType::EncryptionDict => {
                format!("encryption/{:02x}/{:016x}.dict", idx, file_id)
            }
        }
    }

    pub fn file_key(&self, file_id: u64, file_type: FileType) -> String {
        let idx = (fingerprint64(file_id.to_le_bytes().as_slice())) as u8;
        match file_type {
            FileType::Sst => {
                format!("{}/{:02x}/{:016x}.sst", self.prefix, idx, file_id)
            }
            FileType::Blob => {
                format!("{}/blob/{:02x}/{:016x}.blob", self.prefix, idx, file_id)
            }
            FileType::TxnChunk => {
                format!("{}/txn/{:02x}/{:016x}.txn", self.prefix, idx, file_id)
            }
            FileType::Schema => {
                format!("{}/schema/{:02x}/{:016x}.schema", self.prefix, idx, file_id)
            }
            FileType::Columnar => {
                format!("{}/col/{:02x}/{:016x}.col", self.prefix, idx, file_id)
            }
            FileType::VectorIndex => {
                format!("{}/vec/{:02x}/{:016x}.vec", self.prefix, idx, file_id)
            }
            FileType::EncryptionDict => {
                format!(
                    "{}/encryption/{:02x}/{:016x}.dict",
                    self.prefix, idx, file_id
                )
            }
        }
    }

    pub fn get_prefix(&self) -> String {
        self.prefix.clone()
    }

    pub fn get_bucket(&self) -> String {
        self.bucket.clone()
    }

    pub fn is_on_aws(&self) -> bool {
        self.hostname.contains(AWS_DOMAIN_STRING)
    }

    // parse the sst file's suffix with format {idx}/{file_id}.sst
    pub fn parse_sst_file_suffix(&self, key: &str) -> String {
        let end_idx = key.len();
        let start_idx = end_idx - 4 - 16 - 1 - 2;
        let suffix = &key[start_idx..end_idx];
        suffix.to_string()
    }

    // Try to parse the sst file id from file key.
    // Note: do NOT use in performance critical path as regex is used.
    pub fn try_parse_all_file_id(&self, key: &str) -> Option<(u64, FileType)> {
        self.try_parse_sst_file_id(key)
            .or_else(|| self.try_parse_other_file_id(key))
    }

    // Expected file key format: "/{prefix}/{idx}/{file_id}.sst".
    pub fn try_parse_sst_file_id(&self, key: &str) -> Option<(u64, FileType)> {
        if !key.ends_with(".sst") {
            return None;
        }

        lazy_static::lazy_static! {
            static ref RE: Regex = Regex::new(r"/[0-9a-f]{2}/([0-9a-f]{16})\.sst$").unwrap();
        }
        let caps = RE.captures(key)?;
        Some((u64::from_str_radix(&caps[1], 16).unwrap(), FileType::Sst))
    }

    // Expected file key format:
    // "/{prefix}/{file_type}/{idx}/{file_id}.{file_type}".
    pub fn try_parse_other_file_id(&self, key: &str) -> Option<(u64, FileType)> {
        lazy_static::lazy_static! {
            static ref RE: Regex = Regex::new(r"/(?<subdir>[a-z]+)/[0-9a-f]{2}/(?<fileid>[0-9a-f]{16})\.(?<filetype>[a-z]+)$").unwrap();
        }
        let caps = RE.captures(key)?;

        if caps["filetype"] != caps["subdir"] {
            return None;
        }
        let file_type = FileType::try_from(&caps["filetype"]).ok()?;
        let file_id = u64::from_str_radix(&caps["fileid"], 16).ok()?;
        Some((file_id, file_type))
    }

    fn is_err_retryable<T>(&self, rustoto_err: &RusotoError<T>) -> bool {
        match rustoto_err {
            RusotoError::Service(_) => true,
            RusotoError::HttpDispatch(_) => true,
            RusotoError::InvalidDnsName(_) => false,
            RusotoError::Credentials(cred) => cred.message.contains("Timeout"),
            RusotoError::Validation(_) => false,
            RusotoError::ParseError(_) => false,
            RusotoError::Unknown(resp) => resp.status.is_server_error(),
            RusotoError::Blocking => false,
        }
    }

    fn is_err_not_found<T>(&self, rustoto_err: &RusotoError<T>) -> bool {
        match rustoto_err {
            RusotoError::Unknown(resp) => resp.status == StatusCode::NOT_FOUND,
            _ => false,
        }
    }

    async fn sleep_for_retry(&self, retry_cnt: &mut u32, file_name: &str) -> bool {
        if *retry_cnt < MAX_RETRY_COUNT {
            *retry_cnt += 1;
            let retry_sleep = 2u64.pow(*retry_cnt) * RETRY_SLEEP_MS;
            tokio::time::sleep(Duration::from_millis(retry_sleep)).await;
            true
        } else {
            error!(
                "read file {}, reach max retry count {}",
                file_name, MAX_RETRY_COUNT
            );
            false
        }
    }

    /// list gets a list of file ids(full path) greater than `start_after`, with
    /// optional prefix `prefix`.
    ///
    /// The result contains:
    ///     A vector of file content with `key` & `last_modified` timestamp.
    ///     A boolean `has_more` indicate if there is more.
    ///     An optional `next_start_after` for next loop if `has_more` is true.
    ///
    /// Note:
    ///     `prefix` should NOT be contained in `start_after`.
    ///     The file ids in result are in full path, including `S3Fs.prefix` and
    /// `prefix`.
    ///
    /// Ref: https://docs.aws.amazon.com/AmazonS3/latest/API/API_ListObjectsV2.html
    pub async fn list(
        &self,
        start_after: &str,
        prefix: Option<&str>,
        max_keys: Option<u32>,
    ) -> crate::dfs::Result<(
        Vec<ListObjectContent>,
        bool,           // has_more. Deprecated, use `next_start_after`
        Option<String>, // next_start_after
    )> {
        let prefix = format!("{}/{}", self.prefix.clone(), prefix.unwrap_or_default());
        let start_after = format!("{}{}", prefix, start_after);
        let mut retry_cnt = 0;
        let start_time_with_retry = Instant::now_coarse();
        loop {
            let start_time = Instant::now_coarse();
            let mut req = self.new_request("GET", "");
            let mut params = Params::new();
            params.put("list-type", "2");
            params.put("start-after", &start_after);
            params.put("prefix", &prefix);
            if let Some(max_keys) = max_keys {
                params.put("max-keys", &max_keys.to_string());
            }
            req.set_params(params);
            let mut result = self.dispatch(req, ListObjectsV2Error::from_response).await;
            if result.is_ok() {
                let mut response = result.unwrap();
                let body_res = self.read_body(&mut response).await;
                if body_res.is_ok() {
                    let body = body_res.unwrap();
                    let body_str = body.to_str().unwrap();
                    let list: ListObjects = quick_xml::de::from_str(body_str).unwrap();
                    let next_start_after = list.is_truncated.then(|| {
                        list.contents.last().unwrap().key.as_str()[prefix.len()..].to_string()
                    });

                    let duration = start_time.saturating_elapsed();
                    KVENGINE_DFS_LATENCY
                        .list
                        .unknown
                        .observe(duration.as_secs_f64());
                    let duration_with_retry = start_time_with_retry.saturating_elapsed();
                    KVENGINE_DFS_LATENCY_WITH_RETRY
                        .list
                        .unknown
                        .observe(duration_with_retry.as_secs_f64());
                    KVENGINE_DFS_REQUEST_COUNTER.list.unknown.inc();

                    return Ok((list.contents, list.is_truncated, next_start_after));
                } else {
                    result = Err(body_res.unwrap_err().into());
                }
            }
            let err = result.unwrap_err();
            if self.is_err_not_found(&err) {
                return Ok((vec![], false, None));
            } else if self.is_err_retryable(&err) && retry_cnt < MAX_RETRY_COUNT {
                KVENGINE_DFS_RETRY_COUNTER.list.unknown.inc();
                retry_cnt += 1;
                let retry_sleep = 2u64.pow(retry_cnt) * RETRY_SLEEP_MS;
                tokio::time::sleep(Duration::from_millis(retry_sleep)).await;
                continue;
            }
            error!(
                "failed to list files start after {}, reach max retry count {}, err {:?}",
                start_after, MAX_RETRY_COUNT, err,
            );
            return Err(err.into());
        }
    }

    pub async fn list_folders(
        &self,
        prefix: &str,
        delimiter: Option<&str>,
    ) -> crate::dfs::Result<Vec<String>> {
        let prefix = format!("{}/{}", self.prefix.clone(), prefix);
        let delimiter = delimiter.unwrap_or("/");
        let mut retry_cnt = 0;
        let start_time_with_retry = Instant::now_coarse();
        loop {
            let start_time = Instant::now_coarse();
            let mut req = self.new_request("GET", "");
            let mut params = Params::new();
            params.put("list-type", "2");
            params.put("start-after", &prefix);
            params.put("prefix", &prefix);
            params.put("delimiter", delimiter);
            req.set_params(params);
            let mut result = self.dispatch(req, ListObjectsV2Error::from_response).await;
            if result.is_ok() {
                let mut response = result.unwrap();
                let body_res = self.read_body(&mut response).await;
                if body_res.is_ok() {
                    let body = body_res.unwrap();
                    let body_str = body.to_str().unwrap();
                    let list: ListObjects = quick_xml::de::from_str(body_str).unwrap();
                    let prefixes = list
                        .common_prefixes
                        .into_iter()
                        .map(|p| p.prefix)
                        .collect::<Vec<_>>();
                    let duration = start_time.saturating_elapsed();
                    KVENGINE_DFS_LATENCY
                        .list
                        .unknown
                        .observe(duration.as_secs_f64());
                    let duration_with_retry = start_time_with_retry.saturating_elapsed();
                    KVENGINE_DFS_LATENCY_WITH_RETRY
                        .list
                        .unknown
                        .observe(duration_with_retry.as_secs_f64());
                    KVENGINE_DFS_REQUEST_COUNTER.list.unknown.inc();
                    return Ok(prefixes);
                } else {
                    result = Err(body_res.unwrap_err().into());
                }
            }
            let err = result.unwrap_err();
            if self.is_err_not_found(&err) {
                return Ok(vec![]);
            } else if self.is_err_retryable(&err) && retry_cnt < MAX_RETRY_COUNT {
                KVENGINE_DFS_RETRY_COUNTER.list.unknown.inc();
                retry_cnt += 1;
                let retry_sleep = 2u64.pow(retry_cnt) * RETRY_SLEEP_MS;
                tokio::time::sleep(Duration::from_millis(retry_sleep)).await;
                continue;
            }
            error!(
                "failed to list folders prefix {}, reach max retry count {}, err {:?}",
                prefix, MAX_RETRY_COUNT, err,
            );
            return Err(err.into());
        }
    }

    pub async fn is_removed(&self, file_key: &str) -> Result<bool, dfs::Error> {
        let mut retry_cnt = 0;
        let file_type = self.get_file_type_from_key(file_key);
        let start_time_with_retry = Instant::now();
        loop {
            let start_time = Instant::now();
            let req = self.new_tagging_request("GET", file_key);
            let mut result = self
                .dispatch(req, GetObjectTaggingError::from_response)
                .await;
            if result.is_ok() {
                let mut resp = result.unwrap();
                let body_res = self.read_body(&mut resp).await;
                if body_res.is_ok() {
                    let body = body_res.unwrap();
                    let body_str = body.to_str().unwrap();
                    let tagging: Tagging = quick_xml::de::from_str(body_str).unwrap();

                    let duration = start_time.saturating_elapsed();
                    KVENGINE_DFS_LATENCY
                        .get_tagging
                        .get(MetricsFileType::from(file_type))
                        .observe(duration.as_secs_f64());
                    let duration_with_retry = start_time_with_retry.saturating_elapsed();
                    KVENGINE_DFS_LATENCY_WITH_RETRY
                        .get_tagging
                        .get(MetricsFileType::from(file_type))
                        .observe(duration_with_retry.as_secs_f64());
                    KVENGINE_DFS_REQUEST_COUNTER
                        .get_tagging
                        .get(MetricsFileType::from(file_type))
                        .inc();
                    return Ok(tagging.has_deleted_tag());
                } else {
                    result = Err(body_res.unwrap_err().into());
                }
            }
            let err = result.unwrap_err();
            if let RusotoError::Service(_) = err {
                return Err(dfs::Error::S3(format!(
                    "Get file {} tagging fail {:?}",
                    file_key, err
                )));
            }
            if self.is_err_retryable(&err) && retry_cnt < MAX_RETRY_COUNT {
                KVENGINE_DFS_RETRY_COUNTER
                    .get_tagging
                    .get(MetricsFileType::from(file_type))
                    .inc();
                retry_cnt += 1;
                let retry_sleep = 2u64.pow(retry_cnt) * RETRY_SLEEP_MS;
                warn!(
                    "Get file {} tagging fail {:?}, retry_cnt {}, retry after {}ms",
                    file_key, err, retry_cnt, retry_sleep
                );
                tokio::time::sleep(Duration::from_millis(retry_sleep)).await;
                continue;
            }
            return Err(dfs::Error::S3(format!(
                "Get file {} tagging, reach max retry count {}, err {:?}",
                file_key, MAX_RETRY_COUNT, err,
            )));
        }
    }

    pub async fn exist(&self, key: String, file_name: String) -> Result<bool, dfs::Error> {
        let mut retry_cnt = 0;
        let file_type = self.get_file_type_from_key(&key);
        let start_time_with_retry = Instant::now();
        loop {
            let start_time = Instant::now();
            let req = self.new_request("HEAD", &key);
            let result = self.dispatch(req, HeadObjectError::from_response).await;
            if result.is_ok() {
                let duration = start_time.saturating_elapsed();
                KVENGINE_DFS_LATENCY
                    .head
                    .get(MetricsFileType::from(file_type))
                    .observe(duration.as_secs_f64());
                let duration_with_retry = start_time_with_retry.saturating_elapsed();
                KVENGINE_DFS_LATENCY_WITH_RETRY
                    .head
                    .get(MetricsFileType::from(file_type))
                    .observe(duration_with_retry.as_secs_f64());
                KVENGINE_DFS_REQUEST_COUNTER
                    .head
                    .get(MetricsFileType::from(file_type))
                    .inc();
                return Ok(true);
            }
            let err = result.unwrap_err();
            if let RusotoError::Service(HeadObjectError::NoSuchKey(err_msg)) = &err {
                warn!("file {} not exist, err msg {}", &file_name, err_msg);
                return Ok(false);
            }
            if self.is_err_not_found(&err) {
                warn!("file {} not exist, err {}", &file_name, err.to_string());
                return Ok(false);
            }
            if self.is_err_retryable(&err) && self.sleep_for_retry(&mut retry_cnt, &file_name).await
            {
                KVENGINE_DFS_RETRY_COUNTER
                    .head
                    .get(MetricsFileType::from(file_type))
                    .inc();
                warn!("retry head file {}, error {:?}", &file_name, &err);
                continue;
            }
            return Err(dfs::Error::S3(err.to_string()));
        }
    }

    fn new_request(&self, method: &str, key: &str) -> SignedRequest {
        let path = if self.virtual_host {
            format!("/{}", key)
        } else {
            format!("/{}/{}", &self.bucket, key)
        };
        let mut req = SignedRequest::new(method, "s3", &self.region, &path);
        if self.use_http {
            req.scheme = Some("http".to_string());
        }
        req
    }

    fn new_tagging_request(&self, method: &str, key: &str) -> SignedRequest {
        let mut req = self.new_request(method, key);
        let mut params = Params::new();
        params.put_key("tagging");
        req.set_params(params);
        req
    }

    async fn dispatch<E>(
        &self,
        mut req: SignedRequest,
        from_response: fn(BufferedHttpResponse) -> RusotoError<E>,
    ) -> Result<Response, RusotoError<E>> {
        req.set_hostname(Some(self.hostname.clone()));
        let mut resp = self
            .rusoto_s3c
            .sign_and_dispatch_timeout(req, DISPATCH_TIMEOUT)
            .await?;
        if !resp.status.is_success() {
            let buffered = resp.buffer().await.map_err(RusotoError::HttpDispatch)?;
            return Err(from_response(buffered));
        }
        Ok(Response { resp })
    }

    async fn read_body(&self, resp: &mut Response) -> Result<Bytes, HttpDispatchError> {
        let cap = resp
            .headers
            .remove("Content-Length")
            .map(|value| value.parse::<usize>().unwrap())
            .unwrap_or_default();
        let mut buf = Vec::with_capacity(cap);
        while let Some(res) = tokio::time::timeout(READ_BODY_TIMEOUT, resp.body.next())
            .await
            .map_err(|e| HttpDispatchError::new(format!("read body timeout {:?}", e)))?
        {
            let chunk = res.map_err(|e| HttpDispatchError::new(format!("{:?}", e)))?;
            buf.extend_from_slice(chunk.chunk());
        }
        Ok(Bytes::from(buf))
    }

    pub async fn download_object(
        &self,
        full_key: &str,
        dst_path: impl AsRef<Path>,
    ) -> crate::dfs::Result<()> {
        let resp = self
            .aws_s3c
            .get_object()
            .bucket(&self.bucket)
            .key(full_key)
            .send()
            .await?;

        let path: &Path = dst_path.as_ref();
        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }
        let tmp_path: PathBuf = path.with_extension("part");

        let mut reader = resp.body.into_async_read();
        let mut file = tokio::fs::File::create(&tmp_path).await?;

        tokio::io::copy(&mut reader, &mut file).await?;

        file.flush().await?;

        tokio::fs::rename(&tmp_path, path).await?;

        Ok(())
    }

    pub async fn get_object(
        &self,
        key: String,
        file_name: String,
        opts: GetObjectOptions,
    ) -> crate::dfs::Result<Bytes> {
        let (data, _) = self.get_object_ext(key, file_name, opts, false).await?;
        Ok(data)
    }

    pub async fn get_object_ext(
        &self,
        key: String,
        file_name: String,
        opts: GetObjectOptions,
        need_complete_length: bool,
    ) -> crate::dfs::Result<(Bytes, Option<u64> /* complete_length */)> {
        let mut retry_cnt = 0;
        let file_type = self.get_file_type_from_key(&key);
        let start_time_with_retry = Instant::now_coarse();
        loop {
            let start_time = Instant::now_coarse();
            let mut req = self.new_request("GET", &key);
            if !opts.is_full_range() {
                req.add_header("Range", &format!("bytes={}", opts.range_string()));
            }
            let mut result = self.dispatch(req, GetObjectError::from_response).await;

            if result.is_ok() {
                let mut resp = result.unwrap();
                let body = self.read_body(&mut resp).await;
                match body {
                    Ok(data) => {
                        info!(
                            "read file {}, size {}, takes {:?}, retry {}",
                            &file_name,
                            data.len(),
                            start_time.saturating_elapsed(),
                            retry_cnt
                        );

                        let complete_length = if need_complete_length {
                            if opts.is_full_range() {
                                Some(data.len() as u64)
                            } else {
                                parse_content_range(resp.headers.get(CONTENT_RANGE))
                            }
                        } else {
                            None
                        };

                        let duration = start_time.saturating_elapsed();
                        KVENGINE_DFS_THROUGHPUT
                            .s3
                            .get
                            .get(MetricsFileType::from(file_type))
                            .inc_by(data.len() as u64);
                        KVENGINE_DFS_LATENCY
                            .get
                            .get(MetricsFileType::from(file_type))
                            .observe(duration.as_secs_f64());
                        let duration_with_retry = start_time_with_retry.saturating_elapsed();
                        KVENGINE_DFS_LATENCY_WITH_RETRY
                            .get
                            .get(MetricsFileType::from(file_type))
                            .observe(duration_with_retry.as_secs_f64());
                        KVENGINE_DFS_REQUEST_COUNTER
                            .get
                            .get(MetricsFileType::from(file_type))
                            .inc();

                        return Ok((data, complete_length));
                    }
                    Err(err) => result = Err(err.into()),
                }
            }
            let err = result.unwrap_err();
            if let RusotoError::Service(GetObjectError::NoSuchKey(err_msg)) = &err {
                error!("file {} not exist, err msg {}", &file_name, err_msg);
                return Err(dfs::Error::NoSuchKey(format!(
                    "file {} not exist,",
                    &file_name
                )));
            }
            if self.is_err_not_found(&err) {
                error!("file {} not exist, err {}", &file_name, err.to_string());
                return Err(dfs::Error::NoSuchKey(format!(
                    "file {} not exist,",
                    &file_name
                )));
            }
            if self.is_err_retryable(&err) && self.sleep_for_retry(&mut retry_cnt, &file_name).await
            {
                KVENGINE_DFS_RETRY_COUNTER
                    .get
                    .get(MetricsFileType::from(file_type))
                    .inc();
                warn!("retry read file {}, error {:?}", &file_name, &err);
                continue;
            }
            return Err(err.into());
        }
    }

    pub async fn put_object(
        &self,
        key: String,
        data: Bytes,
        file_name: String,
    ) -> crate::dfs::Result<()> {
        self.put_object_with_options(key, data, file_name, None, None, None)
            .await
    }

    pub async fn put_object_with_options(
        &self,
        key: String,
        data: Bytes,
        file_name: String,
        tagging: Option<&Tagging>,
        storage_class: Option<&str>,
        checksum: Option<u32>, // checksum saved in big-endian order
    ) -> crate::dfs::Result<()> {
        fail_point!(
            "s3fs_put_wal_chunks_error",
            key.contains("wal_chunks"),
            |_| Err(Error::Other(
                "s3fs_put_wal_chunks_error failpoint".to_string(),
            ))
        );

        let mut retry_cnt = 0;
        let start_time_with_retry = Instant::now();
        let data_len = data.len();
        let file_type = self.get_file_type_from_key(&key);
        loop {
            let start_time = Instant::now();
            let mut req = self.new_request("PUT", &key);
            req.add_header("Content-Length", &format!("{}", data.len()));
            if let Some(tagging) = tagging {
                req.add_header("x-amz-tagging", &tagging.to_url_encoded());
            }
            if let Some(storage_class) = storage_class.as_ref() {
                if self.is_on_aws() {
                    req.add_header("x-amz-storage-class", storage_class);
                } else {
                    debug!(
                        "{} ignore storage_class {} which is not supported by {}",
                        key, storage_class, self.hostname
                    );
                }
            }
            if let Some(checksum) = checksum {
                if self.is_on_aws() {
                    // `x-amz-checksum-crc32c` value is base64 encoded in big-endian order.
                    let checksum = base64::encode(checksum.to_be_bytes());
                    req.add_header("x-amz-checksum-crc32", &checksum);
                } else {
                    debug!(
                        "{} ignore x-amz-checksum-crc32c which is not supported by {}",
                        key, self.hostname
                    );
                }
            }
            let data = data.clone();
            let stream = futures::stream::once(async move { Ok(data) });
            req.set_payload_stream(rusoto_core::ByteStream::new(stream));
            let result = self.dispatch(req, PutObjectError::from_response).await;
            if result.is_ok() {
                info!(
                    "create file {}, size {}, takes {:?}, retry {}",
                    &file_name,
                    data_len,
                    start_time.saturating_elapsed(),
                    retry_cnt
                );
                KVENGINE_DFS_THROUGHPUT
                    .s3
                    .put
                    .get(MetricsFileType::from(file_type))
                    .inc_by(data_len as u64);
                KVENGINE_DFS_LATENCY
                    .put
                    .get(MetricsFileType::from(file_type))
                    .observe(start_time.saturating_elapsed().as_secs_f64());
                KVENGINE_DFS_LATENCY_WITH_RETRY
                    .put
                    .get(MetricsFileType::from(file_type))
                    .observe(start_time_with_retry.saturating_elapsed().as_secs_f64());
                KVENGINE_DFS_REQUEST_COUNTER
                    .put
                    .get(MetricsFileType::from(file_type))
                    .inc();

                return Ok(());
            }
            let err = result.unwrap_err();
            if self.is_err_retryable(&err) {
                if retry_cnt < MAX_RETRY_COUNT {
                    KVENGINE_DFS_RETRY_COUNTER
                        .put
                        .get(MetricsFileType::from(file_type))
                        .inc();
                    retry_cnt += 1;
                    let retry_sleep = 2u64.pow(retry_cnt) * RETRY_SLEEP_MS;
                    tokio::time::sleep(Duration::from_millis(retry_sleep)).await;
                    warn!("retry create file {}, error {:?}", &file_name, &err);
                    continue;
                } else {
                    error!(
                        "create file {}, takes {:?}, reach max retry count {}",
                        &file_name,
                        start_time_with_retry.saturating_elapsed(),
                        MAX_RETRY_COUNT
                    );
                }
            }
            return Err(err.into());
        }
    }

    // Ref: https://docs.aws.amazon.com/AmazonS3/latest/API/API_DeleteObject.html
    pub async fn delete_object(&self, key: String, file_name: String) -> crate::dfs::Result<()> {
        let mut retry_cnt = 0;
        let file_type = self.get_file_type_from_key(&key);
        let start_time_with_retry = Instant::now_coarse();
        loop {
            let start_time = Instant::now_coarse();
            let req = self.new_request("DELETE", &key);
            let result = self.dispatch(req, DeleteObjectError::from_response).await;
            if result.is_ok() {
                info!(
                    "delete file {}, takes {:?}, retry {}",
                    &file_name,
                    start_time.saturating_elapsed(),
                    retry_cnt
                );

                let duration = start_time.saturating_elapsed();
                KVENGINE_DFS_LATENCY
                    .delete
                    .get(MetricsFileType::from(file_type))
                    .observe(duration.as_secs_f64());
                let duration_with_retry = start_time_with_retry.saturating_elapsed();
                KVENGINE_DFS_LATENCY_WITH_RETRY
                    .delete
                    .get(MetricsFileType::from(file_type))
                    .observe(duration_with_retry.as_secs_f64());
                KVENGINE_DFS_REQUEST_COUNTER
                    .delete
                    .get(MetricsFileType::from(file_type))
                    .inc();
                return Ok(());
            }
            let err = result.unwrap_err();
            if self.is_err_retryable(&err) {
                if retry_cnt < MAX_RETRY_COUNT {
                    KVENGINE_DFS_RETRY_COUNTER
                        .delete
                        .get(MetricsFileType::from(file_type))
                        .inc();
                    retry_cnt += 1;
                    let retry_sleep = 2u64.pow(retry_cnt) * RETRY_SLEEP_MS;
                    tokio::time::sleep(Duration::from_millis(retry_sleep)).await;
                    warn!(
                        "retry delete file {}, error {:?}, retry cnt {}, retry after {}ms",
                        &file_name, &err, retry_cnt, retry_sleep
                    );
                    continue;
                } else {
                    error!(
                        "delete file {}, takes {:?}, reach max retry count {}",
                        &file_name,
                        start_time.saturating_elapsed(),
                        MAX_RETRY_COUNT
                    );
                }
            }
            return Err(err.into());
        }
    }

    /// Copy object from `source_file_id` to `target_file_id`.
    ///
    /// `source_file_id` and `target_file_id` can be the same. And it's the only
    /// way to update the object creation timestamp, which is used for
    /// expiration rules of lifecycle.
    ///
    /// `target_tags`: replace tags if some, otherwise copy from source.
    ///
    /// `target_storage_class`: copy to another storage class if some. Note than
    /// only AWS S3 support storage class.
    pub async fn copy_object(
        &self,
        source_key: &str,
        target_key: &str,
        target_tagging: Option<&Tagging>,
        target_storage_class: Option<&str>,
    ) -> Result<(), dfs::Error> {
        let full_source_key = format!("{}/{}", self.bucket, source_key);
        self.raw_copy_object(
            &full_source_key,
            target_key,
            target_tagging,
            target_storage_class,
        )
        .await
    }

    /// Copy object from `source_key` to `target_key`.
    ///
    /// "raw" means the source key will be directly put to `x-amz-copy-source`.
    ///
    /// So this can be used to copy objects across buckets. For now copying a
    /// packed backup uses this.
    pub async fn raw_copy_object(
        &self,
        source_key: &str,
        target_key: &str,
        target_tagging: Option<&Tagging>,
        target_storage_class: Option<&str>,
    ) -> Result<(), dfs::Error> {
        let start_time_with_retry = Instant::now();
        let file_type = self.get_file_type_from_key(target_key);
        let mut retry_cnt = 0;
        loop {
            let start_time = Instant::now_coarse();
            let mut req = self.new_request("PUT", target_key);
            req.add_header("x-amz-copy-source", source_key);
            req.add_header("x-amz-metadata-directive", "REPLACE");
            if let Some(target_tagging) = target_tagging {
                req.add_header("x-amz-tagging", &target_tagging.to_url_encoded());
                req.add_header("x-amz-tagging-directive", "REPLACE");
            }
            if let Some(target_storage_class) = target_storage_class.as_ref() {
                if self.is_on_aws() {
                    req.add_header("x-amz-storage-class", target_storage_class);
                } else {
                    debug!(
                        "{} ignore target_storage_class {} which is not supported by {}",
                        target_key, target_storage_class, self.hostname
                    );
                }
            }
            if let Err(err) = self.dispatch(req, CopyObjectError::from_response).await {
                if retry_cnt < MAX_RETRY_COUNT {
                    KVENGINE_DFS_RETRY_COUNTER
                        .copy
                        .get(MetricsFileType::from(file_type))
                        .inc();
                    retry_cnt += 1;
                    let retry_sleep = 2u64.pow(retry_cnt) * RETRY_SLEEP_MS;
                    warn!(
                        "retry copy file {}, retry count {}, retry after {}ms, err {:?}",
                        target_key, retry_cnt, retry_sleep, err,
                    );
                    tokio::time::sleep(Duration::from_millis(retry_sleep)).await;
                    continue;
                } else {
                    let err_msg = format!(
                        "failed to copy file {} from {}, reach max retry count {}, err {:?}",
                        target_key, source_key, MAX_RETRY_COUNT, err,
                    );
                    error!("{}", err_msg);
                    return Err(dfs::Error::S3(err_msg));
                }
            }

            let duration = start_time.saturating_elapsed();
            KVENGINE_DFS_LATENCY
                .copy
                .get(MetricsFileType::from(file_type))
                .observe(duration.as_secs_f64());
            let duration_with_retry = start_time_with_retry.saturating_elapsed();
            KVENGINE_DFS_LATENCY_WITH_RETRY
                .copy
                .get(MetricsFileType::from(file_type))
                .observe(duration_with_retry.as_secs_f64());
            KVENGINE_DFS_REQUEST_COUNTER
                .copy
                .get(MetricsFileType::from(file_type))
                .inc();
            return Ok(());
        }
    }

    pub async fn retain_file(&self, file_key: &str) -> Result<(), dfs::Error> {
        if self.is_removed(file_key).await? {
            let empty_tagging = Tagging::default();
            self.copy_object(
                file_key,
                file_key,
                Some(&empty_tagging),
                Some(STORAGE_CLASS_DEFAULT),
            )
            .await?;
        }
        Ok(())
    }

    /// Choose proper storage class for removed files.
    pub fn choose_storage_class_for_removed_files(&self, file_len: Option<u64>) -> &'static str {
        // STORAGE_CLASS_STANDARD_IA is more cost efficient than STORAGE_CLASS_STANDARD
        // for NOT small files.
        if file_len.is_some() && file_len.unwrap() > SMALL_FILE_THRESHOLD_BYTES {
            STORAGE_CLASS_STANDARD_IA
        } else {
            STORAGE_CLASS_DEFAULT
        }
    }

    // Add helper methods for getting file type and storage class
    fn get_file_type_from_key(&self, key: &str) -> &'static str {
        if key.ends_with(".sst") {
            "sst"
        } else if key.ends_with(".blob") {
            "blob"
        } else if key.ends_with(".txn") {
            "txn"
        } else if key.ends_with(".schema") {
            "schema"
        } else if key.ends_with(".col") {
            "col"
        } else if key.ends_with(".vec") {
            "vec"
        } else {
            "unknown"
        }
    }
}

#[async_trait]
impl ObjectStorage for S3Fs {
    fn put_objects(&self, objects: Vec<(String, Bytes)>) -> Result<(), String> {
        let put_object = |key: String, data: Bytes| {
            let full_key = format!("{}/{}", self.prefix, key);
            let fs = self.clone();
            let checksum = if self.is_on_aws() {
                Some(crc32fast::hash(&data))
            } else {
                None
            };
            async move {
                fs.put_object_with_options(
                    full_key,
                    data,
                    key.clone(),
                    None,
                    Some(STORAGE_CLASS_INTELLIGENT_TIERING),
                    checksum,
                )
                .await
                .map_err(|err| format!("put {} failed {:?}", &key, err))
            }
        };

        let runtime = self.get_runtime();
        let len = objects.len();

        if len == 1 {
            let (key, data) = objects.into_iter().next().unwrap();
            return runtime.block_on(put_object(key, data));
        }

        let mut handles = Vec::with_capacity(len);
        for (key, data) in objects {
            handles.push(runtime.spawn(put_object(key, data)));
        }

        let errs: Vec<_> = runtime
            .block_on(futures::future::join_all(handles))
            .into_iter()
            .filter_map(|r| r.unwrap().err())
            .collect();
        if !errs.is_empty() {
            return Err(format!("{:?}", errs));
        }
        Ok(())
    }

    fn get_objects(
        &self,
        keys: Vec<(String, GetObjectOptions)>,
    ) -> Result<Vec<(String, Bytes)>, String> {
        let runtime = self.get_runtime();
        let len = keys.len();
        let mut handles = Vec::with_capacity(len);
        for (key, opts) in keys {
            let full_key = format!("{}/{}", self.prefix, key);
            let fs = self.clone();
            handles.push(runtime.spawn(async move {
                fs.get_object(full_key, key.clone(), opts)
                    .await
                    .map_err(|err| format!("get {} failed {:?}", &key, err))
                    .map(|data| (key.clone(), data))
            }));
        }

        let mut objects = vec![];
        let mut errs = vec![];
        for res in runtime.block_on(futures::future::join_all(handles)) {
            match res.unwrap() {
                Ok((key, data)) => {
                    objects.push((key, data));
                }
                Err(err) => {
                    errs.push(err);
                }
            }
        }
        if !errs.is_empty() {
            return Err(format!("{:?}", errs));
        }
        Ok(objects)
    }

    fn list_objects(
        &self,
        start_after: &str,
        prefix: Option<&str>,
        max_keys: Option<u32>,
    ) -> Result<(Vec<ListObjectContent>, Option<String>), String> {
        let runtime = self.get_runtime();
        runtime
            .block_on(self.list(start_after, prefix, max_keys))
            .map(|v| (v.0, v.2))
            .map_err(|err| format!("list failed {:?}", err))
    }

    async fn download_objects(
        &self,
        items: Vec<(String, GetObjectOptions)>,
        concurrency: usize,
        dest_dir: PathBuf,
    ) -> std::pin::Pin<Box<dyn Stream<Item = Result<(String, PathBuf), String>> + Send>> {
        let s3fs = self.clone();

        let s = futures::stream::iter(items.into_iter().map(move |(key, _)| {
            let s3fs = s3fs.clone();
            let full_key = format!("{}/{}", s3fs.prefix, key);
            let dest_path = dest_dir.join(&s3fs.bucket).join(&full_key);
            let dest_path_str = dest_path.to_string_lossy().into_owned();

            async move {
                match s3fs.download_object(&full_key, &dest_path_str).await {
                    Ok(_) => Ok((key, dest_path)),
                    Err(err) => Err(err),
                }
                .map_err(|err| format!("downloads {} objects failed {:?}", dest_path_str, err))
            }
        }))
        .buffer_unordered(concurrency.max(1));
        Box::pin(s)
    }
}

#[async_trait]
impl Dfs for S3Fs {
    async fn exists(&self, file_id: u64, opts: Options) -> crate::dfs::Result<bool> {
        let file_key = self.file_key(file_id, opts.file_type);
        let file_name = format!("{}.{}", file_id, opts.file_type.suffix());
        self.exist(file_key, file_name).await
    }

    async fn read_file(&self, file_id: u64, opts: Options) -> crate::dfs::Result<Bytes> {
        let filename = if let Some(end_off) = opts.end_off {
            format!("{}-{}-{}.seg", file_id, opts.start_off, end_off)
        } else {
            format!("{}.{}", file_id, opts.file_type.suffix())
        };
        self.get_object(
            self.file_key(file_id, opts.file_type),
            filename,
            GetObjectOptions {
                start_off: Some(opts.start_off),
                end_off: opts.end_off,
            },
        )
        .await
    }

    async fn create(&self, file_id: u64, data: Bytes, opts: Options) -> crate::dfs::Result<()> {
        // Calculate checksum for the file. This can be used to ensure the data
        // integrity when saving to dfs (s3 only) and verify the data integrity when
        // getting objects from dfs.
        let checksum = if self.is_on_aws() {
            Some(crc32fast::hash(&data))
        } else {
            None
        };
        self.put_object_with_options(
            self.file_key(file_id, opts.file_type),
            data,
            format!("{}.{}", file_id, opts.file_type.suffix()),
            None,
            Some(STORAGE_CLASS_INTELLIGENT_TIERING),
            checksum,
        )
        .await
    }

    /// Logically remove the file on S3 by tagging with "deleted=true".
    /// And the file would be permanently removed after `gc_lifetime`. See
    /// `DfsGc`.
    async fn remove(&self, file_id: u64, file_len: Option<u64>, opts: Options) {
        // Only AWS supports storage class.
        let target_storage_class = if self.is_on_aws() {
            let new_storage_class = self.choose_storage_class_for_removed_files(file_len);
            (new_storage_class != STORAGE_CLASS_DEFAULT).then_some(new_storage_class)
        } else {
            None
        };
        let target_tagging = Tagging::new_single_deleted();
        let file_key = self.file_key(file_id, opts.file_type);
        let _ = self
            .copy_object(
                &file_key,
                &file_key,
                Some(&target_tagging),
                target_storage_class,
            )
            .await;
    }

    /// Permanently remove the file on S3.
    /// This method should be used by `DfsGc` ONLY to meet GC rules.
    async fn permanently_remove(&self, file_id: u64, opts: Options) -> crate::dfs::Result<()> {
        if self.is_on_aws() {
            return Err(Error::Other(format!(
                "{} permanently_remove is forbidden on AWS",
                file_id
            )));
        }

        self.delete_object(self.file_key(file_id, opts.file_type), file_id.to_string())
            .await
    }

    fn get_runtime(&self) -> &Runtime {
        &self.runtime
    }
}

#[derive(Default, Debug, Serialize, Deserialize)]
#[serde(default)]
#[serde(rename_all = "PascalCase")]
pub struct ListObjects {
    pub common_prefixes: Vec<CommonPrefix>,
    pub contents: Vec<ListObjectContent>,
    pub is_truncated: bool,
}

#[derive(Clone, Default, Debug, Serialize, Deserialize)]
#[serde(default)]
#[serde(rename_all = "PascalCase")]
pub struct CommonPrefix {
    pub prefix: String,
}

#[derive(Clone, Default, Debug, Serialize, Deserialize)]
#[serde(default)]
#[serde(rename_all = "PascalCase")]
pub struct Tagging {
    tag_set: TagSet,
}

impl Tagging {
    fn new_single_deleted() -> Tagging {
        Self {
            tag_set: TagSet {
                tag: vec![Tag::new_deleted()],
            },
        }
    }

    pub fn add_tag(&mut self, key: String, value: String) {
        self.tag_set.tag.push(Tag { key, value });
    }

    fn has_deleted_tag(&self) -> bool {
        self.tag_set.tag.iter().any(|tag| tag.is_deleted())
    }

    pub fn to_xml(&self) -> String {
        quick_xml::se::to_string(&self).unwrap()
    }

    pub fn to_url_encoded(&self) -> String {
        let mut se = url::form_urlencoded::Serializer::new(String::new());
        for tag in &self.tag_set.tag {
            se.append_pair(&tag.key, &tag.value);
        }
        se.finish()
    }

    pub fn from_url_encoded(query: &str) -> Tagging {
        let mut tagging = Self::default();
        for (key, value) in url::form_urlencoded::parse(query.as_bytes()) {
            tagging.tag_set.tag.push(Tag {
                key: key.into_owned(),
                value: value.into_owned(),
            });
        }
        tagging
    }
}

#[derive(Clone, Default, Debug, Serialize, Deserialize)]
#[serde(default)]
#[serde(rename_all = "PascalCase")]
struct TagSet {
    tag: Vec<Tag>,
}

#[derive(Clone, Default, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
#[serde(rename_all = "PascalCase")]
struct Tag {
    #[serde(rename = "$unflatten=Key")]
    key: String,
    #[serde(rename = "$unflatten=Value")]
    value: String,
}

impl Tag {
    fn new_deleted() -> Self {
        Self {
            key: "deleted".to_string(),
            value: "true".to_string(),
        }
    }

    fn is_deleted(&self) -> bool {
        self.key == "deleted" && self.value == "true"
    }
}

struct Response {
    resp: HttpResponse,
}

impl Deref for Response {
    type Target = HttpResponse;

    fn deref(&self) -> &Self::Target {
        &self.resp
    }
}

impl DerefMut for Response {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.resp
    }
}

impl Debug for Response {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.resp.status)
    }
}

/// Ref: https://www.rfc-editor.org/rfc/rfc9110.html#section-14.4, Content-Range
///
/// Examples:
///
/// - bytes 0-499/1234
/// - bytes 42-1233/*
/// - bytes */1234 (when response with 416 Range Not Satisfiable)
fn parse_content_range<S: AsRef<str>>(content_range: Option<S>) -> Option<u64> {
    let content_range = content_range?;
    let content_range = content_range.as_ref();
    if content_range.starts_with("bytes ") {
        let parts: Vec<&str> = content_range.split('/').collect();
        if parts.len() == 2 {
            return parts[1].parse::<u64>().ok();
        }
    }
    None
}

#[cfg(any(test, feature = "testexport"))]
pub mod test_util {
    use std::sync::{Arc, Mutex};

    use aws_sdk_s3::{config::Region, error::ConnectorError, Client as AwsS3Client};
    use aws_smithy_runtime_api::{
        box_error::BoxError,
        client::{
            http::{
                http_client_fn, HttpConnector as SmithyHttpConnectorTrait, HttpConnectorFuture,
                HttpConnectorSettings, SharedHttpClient, SharedHttpConnector,
            },
            orchestrator::{HttpRequest, HttpResponse},
            runtime_components::RuntimeComponents,
        },
    };
    use aws_smithy_types::body::SdkBody;
    use http::{Response as HttpResp, StatusCode as HttpStatus};
    use http_body_util::BodyExt;
    use rusoto_core::signature::SignedRequest;
    use rusoto_mock::{MockCredentialsProvider, MockRequestDispatcher};

    use crate::dfs::S3Fs;

    #[derive(Debug)]
    struct State {
        created: bool,
        file_data: Vec<u8>,
    }

    #[derive(Clone, Debug)]
    struct S3TestDispatcher {
        state: Arc<Mutex<State>>,
    }

    fn cx<E>(e: E) -> ConnectorError
    where
        E: std::error::Error + Send + Sync + 'static,
    {
        ConnectorError::other(BoxError::from(e), None)
    }
    fn cx_poison() -> ConnectorError {
        cx(std::io::Error::new(
            std::io::ErrorKind::Other,
            "state mutex poisoned",
        ))
    }

    impl SmithyHttpConnectorTrait for S3TestDispatcher {
        fn call(&self, request: HttpRequest) -> HttpConnectorFuture {
            let state = self.state.clone();
            let method = request.method().to_string();
            let mut req = request;
            let fut = async move {
                let res = match method.as_str() {
                    "PUT" => {
                        let buf = req
                            .body_mut()
                            .collect()
                            .await
                            .map_err(|e| ConnectorError::io(e))?
                            .to_bytes()
                            .to_vec();
                        {
                            let mut st: std::sync::MutexGuard<'_, State> =
                                state.lock().map_err(|_| cx_poison())?;
                            st.created = true;
                            st.file_data = buf;
                        }

                        HttpResp::builder()
                            .status(HttpStatus::OK)
                            .body(SdkBody::empty())
                    }
                    "HEAD" => {
                        let created = {
                            let st: std::sync::MutexGuard<'_, State> =
                                state.lock().map_err(|_| cx_poison())?;
                            st.created
                        };

                        if created {
                            HttpResp::builder()
                                .status(HttpStatus::OK)
                                .body(SdkBody::empty())
                        } else {
                            HttpResp::builder()
                                .status(HttpStatus::NOT_FOUND)
                                .body(SdkBody::from("Not Found"))
                        }
                    }
                    "GET" => {
                        let (created, data) = {
                            let st: std::sync::MutexGuard<'_, State> =
                                state.lock().map_err(|_| cx_poison())?;
                            (st.created, st.file_data.clone())
                        };
                        if created {
                            HttpResp::builder()
                                .status(HttpStatus::OK)
                                .body(SdkBody::from(data))
                        } else {
                            HttpResp::builder()
                                .status(HttpStatus::NOT_FOUND)
                                .body(SdkBody::from("Not Found"))
                        }
                    }
                    _ => HttpResp::builder()
                        .status(HttpStatus::OK)
                        .body(SdkBody::empty()),
                };
                let http_res = res.map_err(cx)?;
                let status = http_res.status().into();
                let body = http_res.into_body();

                let smithy_res: HttpResponse = HttpResponse::new(status, body);
                Ok::<HttpResponse, ConnectorError>(smithy_res)
            };

            HttpConnectorFuture::new(fut)
        }
    }

    impl rusoto_core::request::DispatchSignedRequest for S3TestDispatcher {
        fn dispatch(
            &self,
            request: SignedRequest,
            _timeout: Option<std::time::Duration>,
        ) -> rusoto_core::request::DispatchSignedRequestFuture {
            let method = request.method.clone();
            let state = self.state.clone();
            Box::pin(async move {
                match method.as_str() {
                    "HEAD" => {
                        let created = {
                            let state = state.lock().unwrap();
                            state.created
                        };
                        if created {
                            MockRequestDispatcher::with_status(200)
                                .dispatch(request, _timeout)
                                .await
                        } else {
                            MockRequestDispatcher::with_status(404)
                                .dispatch(request, _timeout)
                                .await
                        }
                    }
                    "PUT" => {
                        {
                            let mut state = state.lock().unwrap();
                            state.created = true;
                        }
                        MockRequestDispatcher::with_status(200)
                            .dispatch(request, _timeout)
                            .await
                    }
                    "GET" => {
                        let (created, file_data) = {
                            let state = state.lock().unwrap();
                            (state.created, state.file_data.clone())
                        };
                        if created {
                            MockRequestDispatcher::with_status(200)
                                .with_body(std::str::from_utf8(&file_data).unwrap())
                                .dispatch(request, _timeout)
                                .await
                        } else {
                            MockRequestDispatcher::with_status(404)
                                .dispatch(request, _timeout)
                                .await
                        }
                    }
                    _ => {
                        MockRequestDispatcher::with_status(200)
                            .dispatch(request, _timeout)
                            .await
                    }
                }
            })
        }
    }

    fn make_shared_http_client(dispatcher: S3TestDispatcher) -> SharedHttpClient {
        http_client_fn(
            move |_settings: &HttpConnectorSettings,
                  _rc: &RuntimeComponents|
                  -> SharedHttpConnector {
                SharedHttpConnector::new(dispatcher.clone())
            },
        )
    }

    pub fn new_test_s3fs(file_data: &[u8]) -> S3Fs {
        let state = Arc::new(Mutex::new(State {
            created: false,
            file_data: file_data.to_vec(),
        }));
        let dispatcher = S3TestDispatcher { state };
        let rusoto_s3c = rusoto_core::Client::new_with(MockCredentialsProvider, dispatcher.clone());

        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .thread_name("test-s3-client")
            .enable_all()
            .build()
            .unwrap();

        let base = runtime.block_on(
            aws_config::defaults(aws_sdk_s3::config::BehaviorVersion::latest())
                .http_client(make_shared_http_client(dispatcher.clone()))
                .region(Region::new("us-east-1"))
                .load(),
        );

        let conf = aws_sdk_s3::config::Builder::from(&base)
            .http_client(make_shared_http_client(dispatcher))
            .force_path_style(true)
            .build();

        let aws_s3c = AwsS3Client::from_conf(conf);
        S3Fs::new_for_test(
            runtime,
            rusoto_s3c,
            aws_s3c,
            "shard-db".into(),
            "prefix".into(),
        )
    }
}

#[cfg(test)]
mod tests {
    use std::{fs, io::Write};

    use bytes::Buf;
    use rand::random;

    use super::*;
    use crate::{
        dfs::test_util::new_test_s3fs,
        table::{
            file::{File, LocalFile},
            sstable::new_filename,
        },
    };

    #[test]
    fn test_s3() {
        ::test_util::init_log_for_test();

        let local_dir = tempfile::tempdir().unwrap();
        let file_data = "abcdefgh".to_string().into_bytes();
        let s3fs = new_test_s3fs(&file_data);
        let file_id = 321u64;
        let opts = Options::default();

        // Test exists before create
        let fs = s3fs.clone();
        let (tx_exists_before, rx_exists_before) = tikv_util::mpsc::bounded(1);
        let f_exists_before = async move {
            match fs.exists(file_id, opts).await {
                Ok(exists) => {
                    tx_exists_before.send(exists).unwrap();
                }
                Err(err) => {
                    println!("exists before create error {:?}", err);
                    tx_exists_before.send(true).unwrap(); // Fail test on error
                }
            }
        };
        s3fs.runtime.spawn(f_exists_before);
        assert!(!rx_exists_before.recv().unwrap());

        let fs = s3fs.clone();
        let (tx, rx) = tikv_util::mpsc::bounded(1);
        let file_data_clone_create = file_data.clone();
        let f_create = async move {
            match fs
                .create(file_id, bytes::Bytes::from(file_data_clone_create), opts)
                .await
            {
                Ok(_) => {
                    tx.send(true).unwrap();
                    println!("create ok");
                }
                Err(err) => {
                    tx.send(false).unwrap();
                    println!("create error {:?}", err)
                }
            }
        };
        s3fs.runtime.spawn(f_create);
        assert!(rx.recv().unwrap());

        // Test exists after create
        let fs = s3fs.clone();
        let (tx_exists_after, rx_exists_after) = tikv_util::mpsc::bounded(1);
        let f_exists_after = async move {
            match fs.exists(file_id, opts).await {
                Ok(exists) => {
                    tx_exists_after.send(exists).unwrap();
                }
                Err(err) => {
                    println!("exists after create error {:?}", err);
                    tx_exists_after.send(false).unwrap(); // Fail test on error
                }
            }
        };
        s3fs.runtime.spawn(f_exists_after);
        assert!(rx_exists_after.recv().unwrap());

        let fs = s3fs.clone();
        let (tx, rx) = tikv_util::mpsc::bounded(1);
        let local_file = new_filename(file_id, local_dir.path());
        let move_local_file = local_file.clone();
        let f = async move {
            let opts = Options::default();
            match fs.read_file(file_id, opts).await {
                Ok(data) => {
                    let mut file = std::fs::File::create(&move_local_file).unwrap();
                    file.write_all(data.chunk()).unwrap();
                    tx.send(true).unwrap();
                    println!("prefetch ok");
                }
                Err(err) => {
                    tx.send(false).unwrap();
                    println!("prefetch failed {:?}", err)
                }
            }
        };
        s3fs.runtime.spawn(f);
        assert!(rx.recv().unwrap());
        let data = std::fs::read(&local_file).unwrap();
        assert_eq!(&data, &file_data);
        let file = LocalFile::open(file_id, &local_file, false).unwrap();
        assert_eq!(file.size(), 8u64);
        assert_eq!(file.id(), 321u64);
        let data = file.read(0, 8).unwrap();
        assert_eq!(&data, &file_data);
        let fs = s3fs.clone();
        let (tx, rx) = tikv_util::mpsc::bounded(1);
        let f = async move {
            fs.remove(file_id, None, Options::default()).await;
            tx.send(true).unwrap();
        };
        s3fs.runtime.spawn(f);
        assert!(rx.recv().unwrap());
        let _ = fs::remove_file(local_file);
    }

    #[test]
    fn test_s3_schema_file() {
        ::test_util::init_log_for_test();

        let local_dir = tempfile::tempdir().unwrap();
        let file_data = "abcdefgh".to_string().into_bytes();
        let s3fs = new_test_s3fs(&file_data);
        let (tx, rx) = tikv_util::mpsc::bounded(1);

        let fs = s3fs.clone();
        let file_data2 = file_data.clone();
        let opts = Options::default().with_type(FileType::Schema);
        let f = async move {
            match fs.create(1234, bytes::Bytes::from(file_data2), opts).await {
                Ok(_) => {
                    tx.send(true).unwrap();
                    println!("create ok");
                }
                Err(err) => {
                    tx.send(false).unwrap();
                    println!("create error {:?}", err)
                }
            }
        };
        s3fs.runtime.spawn(f);
        assert!(rx.recv().unwrap());
        let fs = s3fs.clone();
        let (tx, rx) = tikv_util::mpsc::bounded(1);
        let local_file = local_dir.path().join(format!("{:016x}.schema", 1234));
        let move_local_file = local_file.clone();
        let f = async move {
            match fs.read_file(1234, opts).await {
                Ok(data) => {
                    let mut file = std::fs::File::create(&move_local_file).unwrap();
                    file.write_all(data.chunk()).unwrap();
                    tx.send(true).unwrap();
                    println!("prefetch ok");
                }
                Err(err) => {
                    tx.send(false).unwrap();
                    println!("prefetch failed {:?}", err)
                }
            }
        };
        s3fs.runtime.spawn(f);
        assert!(rx.recv().unwrap());
        let data = std::fs::read(&local_file).unwrap();
        assert_eq!(&data, &file_data);
        let file = LocalFile::open(1234, &local_file, false).unwrap();
        assert_eq!(file.size(), 8u64);
        assert_eq!(file.id(), 1234u64);
        let data = file.read(0, 8).unwrap();
        assert_eq!(&data, &file_data);
        let fs = s3fs.clone();
        let (tx, rx) = tikv_util::mpsc::bounded(1);
        let f = async move {
            fs.remove(1234, None, opts).await;
            tx.send(true).unwrap();
        };
        s3fs.runtime.spawn(f);
        assert!(rx.recv().unwrap());
        let _ = fs::remove_file(local_file);
    }

    #[test]
    fn test_parse_file_id() {
        let s3fs = new_test_s3fs(b"abcdefgh");

        let file_key = s3fs.file_key(random(), FileType::Sst);
        assert_eq!(
            format!("{}/{}", "prefix", s3fs.parse_sst_file_suffix(&file_key)),
            file_key
        );

        for file_id in [0, 42, 0x1_0000_0000, 0xffff_ffff_ffff_ffff] {
            let file_key = s3fs.file_key(file_id, FileType::Sst);
            assert_eq!(
                s3fs.try_parse_sst_file_id(&file_key),
                Some((file_id, FileType::Sst))
            );
            assert_eq!(
                s3fs.try_parse_all_file_id(&file_key),
                Some((file_id, FileType::Sst))
            );
        }

        for file_type in [
            FileType::Blob,
            FileType::TxnChunk,
            FileType::Schema,
            FileType::Columnar,
            FileType::VectorIndex,
        ] {
            for file_key in [
                "".to_string(),
                "cse/0000000000000001/e00000001/0000000000800000_00000000008e9000.wal".to_string(),
                s3fs.file_key(42, file_type),
            ] {
                assert_eq!(s3fs.try_parse_sst_file_id(&file_key), None);
            }

            for file_id in [0, 42, 0x1_0000_0000, 0xffff_ffff_ffff_ffff] {
                let file_key = s3fs.file_key(file_id, file_type);
                assert_eq!(
                    s3fs.try_parse_other_file_id(&file_key),
                    Some((file_id, file_type))
                );
                assert_eq!(
                    s3fs.try_parse_all_file_id(&file_key),
                    Some((file_id, file_type))
                );
            }
        }

        for file_key in [
            "".to_string(),
            "cse/0000000000000001/e00000001/0000000000800000_00000000008e9000.wal".to_string(),
            s3fs.file_key(42, FileType::Sst),
        ] {
            assert_eq!(s3fs.try_parse_other_file_id(&file_key), None);
        }
    }

    #[test]
    fn test_tagging() {
        let tagging_deleted = Tagging::new_single_deleted();

        assert_eq!(
            tagging_deleted.to_xml(),
            "<Tagging><TagSet><Tag><Key>deleted</Key><Value>true</Value></Tag></TagSet></Tagging>"
        );
        assert_eq!(tagging_deleted.to_url_encoded(), "deleted=true");
        assert_eq!(
            tagging_deleted.tag_set.tag,
            Tagging::from_url_encoded("deleted=true").tag_set.tag
        );
    }

    #[test]
    fn test_parse_content_range() {
        let cases = [
            (None, None),
            (Some(""), None),
            (Some("bytes"), None),
            (Some("bytes 0-499/1234"), Some(1234)),
            (Some("bytes 42-1233/*"), None),
            (Some("bytes */1234"), Some(1234)),
        ];
        for (content_range, expected) in cases {
            assert_eq!(parse_content_range(content_range), expected);
        }
    }
}
