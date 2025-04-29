// Copyright 2022 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    convert::TryFrom,
    fmt::{Debug, Formatter},
    ops::{Deref, DerefMut},
    sync::Arc,
    time::Duration,
};

use async_trait::async_trait;
use bstr::ByteSlice;
use bytes::{BufMut, Bytes, BytesMut};
use engine_traits::{GetObjectOptions, ListObjectContent, ObjectStorage};
use fail::fail_point;
use farmhash::fingerprint64;
use futures::StreamExt;
use http::StatusCode;
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
use tokio::runtime::Runtime;

use crate::dfs::{
    self, config::Config, Dfs, Error, FileType, MetricsFileType, Options, ReservableWriter,
    KVENGINE_DFS_LATENCY, KVENGINE_DFS_LATENCY_WITH_RETRY, KVENGINE_DFS_REQUEST_COUNTER,
    KVENGINE_DFS_RETRY_COUNTER, KVENGINE_DFS_THROUGHPUT,
};

const MAX_RETRY_COUNT: u32 = 9;
const RETRY_SLEEP_MS: u64 = 500;
const CONNECTION_TIMEOUT: Duration = Duration::from_secs(5);
const DISPATCH_TIMEOUT: Duration = Duration::from_secs(300);
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
        s3c: rusoto_core::Client,
        bucket: String,
        prefix: String,
    ) -> Self {
        Self {
            core: Arc::new(S3FsCore::new_with_runtime_and_s3_client(
                Arc::new(runtime),
                s3c,
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
    s3c: rusoto_core::Client,
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

impl S3FsCore {
    fn new_s3_client(
        endpoint: String,
        key_id: String,
        secret_key: String,
        region: String,
        bucket: String,
    ) -> (rusoto_core::Client, String, bool) {
        let mut config = rusoto_core::HttpConfig::new();
        config.read_buf_size(256 * 1024);
        let endpoint = if endpoint.is_empty() {
            format!("http://s3.{}.amazonaws.com", region.as_str())
        } else {
            endpoint
        };
        let use_tls = endpoint.starts_with("https");
        let default_provider = aws::new_credentials_provider_rusoto_wrapper();
        let static_provider =
            rusoto_credential::StaticProvider::new(key_id.clone(), secret_key, None, None);
        let mut http_connector = hyper::client::connect::HttpConnector::new();
        http_connector.set_connect_timeout(Some(CONNECTION_TIMEOUT));
        let s3c = if use_tls {
            let https_connector = HttpsConnector::new_with_connector(http_connector);
            let http_client = HttpClient::from_connector_with_config(https_connector, config);
            if key_id.is_empty() {
                if endpoint.contains(aliyun::DOMAIN_STRING) {
                    let ali_provider = aliyun::new_credential_provider().unwrap();
                    rusoto_core::Client::new_with(ali_provider, http_client)
                } else {
                    rusoto_core::Client::new_with(default_provider, http_client)
                }
            } else {
                rusoto_core::Client::new_with(static_provider, http_client)
            }
        } else {
            let http_client = HttpClient::from_connector_with_config(http_connector, config);
            if key_id.is_empty() {
                if endpoint.contains(aliyun::DOMAIN_STRING) {
                    let ali_provider = aliyun::new_credential_provider().unwrap();
                    rusoto_core::Client::new_with(ali_provider, http_client)
                } else {
                    rusoto_core::Client::new_with(default_provider, http_client)
                }
            } else {
                rusoto_core::Client::new_with(static_provider, http_client)
            }
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

        (s3c, hostname, virtual_host)
    }

    #[inline]
    fn new_with_runtime_and_s3_client(
        runtime: Arc<tokio::runtime::Runtime>,
        s3c: rusoto_core::Client,
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
            s3c,
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
        let (rusoto_s3, hostname, virtual_host) = Self::new_s3_client(
            endpoint.clone(),
            key_id,
            secret_key,
            region.clone(),
            bucket.clone(),
        );
        Self::new_with_runtime_and_s3_client(
            runtime,
            rusoto_s3,
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
            .s3c
            .sign_and_dispatch_timeout(req, DISPATCH_TIMEOUT)
            .await?;
        if !resp.status.is_success() {
            let buffered = resp.buffer().await.map_err(RusotoError::HttpDispatch)?;
            return Err(from_response(buffered));
        }
        Ok(Response { resp })
    }

    async fn read_body(&self, resp: &mut Response) -> Result<Bytes, HttpDispatchError> {
        let mut writer = BytesMut::new().writer();
        self.read_body_to_writer(resp, &mut writer).await?;
        Ok(writer.into_inner().freeze())
    }

    async fn read_body_to_writer<W>(
        &self,
        resp: &mut Response,
        writer: &mut W,
    ) -> Result<u64 /* read_len */, HttpDispatchError>
    where
        W: ReservableWriter + Send + 'static,
    {
        let cap = resp
            .headers
            .remove("Content-Length")
            .map(|value| value.parse::<u64>().unwrap())
            .unwrap_or_default();
        writer.reserve_capacity(cap);
        let mut read_len = 0;
        while let Some(res) = tokio::time::timeout(READ_BODY_TIMEOUT, resp.body.next())
            .await
            .map_err(|e| HttpDispatchError::new(format!("read body timeout {:?}", e)))?
        {
            let chunk = res.map_err(|e| HttpDispatchError::new(format!("{:?}", e)))?;
            read_len += chunk.len() as u64;
            writer
                .write_all(&chunk)
                .map_err(|e| HttpDispatchError::new(format!("write_all: {:?}", e)))?;
        }
        if read_len != cap {
            warn!("content length mismatch";
                "content_length" => cap, "read_len" => read_len);
            debug_assert!(false);
        }
        Ok(read_len)
    }

    pub async fn get_object(
        &self,
        key: String,
        file_name: String,
        opts: GetObjectOptions,
    ) -> crate::dfs::Result<Bytes> {
        let mut writer = BytesMut::new().writer();
        self.get_object_to_writer(key, file_name, opts, &mut writer)
            .await?;
        Ok(writer.into_inner().freeze())
    }

    pub async fn get_object_to_writer<W>(
        &self,
        key: String,
        file_name: String,
        opts: GetObjectOptions,
        writer: &mut W,
    ) -> crate::dfs::Result<u64>
    where
        W: ReservableWriter + Send + 'static,
    {
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
                let res = self.read_body_to_writer(&mut resp, writer).await;
                match res {
                    Ok(object_len) => {
                        info!(
                            "read file {}, size {}, takes {:?}, retry {}",
                            &file_name,
                            object_len,
                            start_time.saturating_elapsed(),
                            retry_cnt
                        );
                        let duration = start_time.saturating_elapsed();
                        KVENGINE_DFS_THROUGHPUT
                            .s3
                            .get
                            .get(MetricsFileType::from(file_type))
                            .inc_by(object_len);
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
                        return Ok(object_len);
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
        let get_object = |key: String, opts: GetObjectOptions| {
            let full_key = format!("{}/{}", self.prefix, key);
            let fs = self.clone();
            async move {
                match fs.get_object(full_key, key.clone(), opts).await {
                    Ok(data) => Ok((key, data)),
                    Err(err) => Err(format!("put {} failed {:?}", &key, err)),
                }
            }
        };

        let runtime = self.get_runtime();
        if keys.len() == 1 {
            let mut keys = keys;
            let (key, opts) = keys.pop().unwrap();
            let res = runtime.block_on(get_object(key, opts))?;
            return Ok(vec![res]);
        }

        let mut join_set = tokio::task::JoinSet::new();
        for (key, opts) in keys {
            join_set.spawn_on(get_object(key, opts), runtime.handle());
        }

        runtime.block_on(async move {
            let mut objects: Vec<(String, Bytes)> = vec![];
            while let Some(res) = join_set.join_next().await {
                objects.push(res.expect("task panic")?);
            }
            Ok(objects)
        })
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

    async fn size(&self, file_id: u64, opts: Options) -> crate::dfs::Result<u64> {
        // Use HEAD to check and get size from Content-Length where possible
        let key = self.file_key(file_id, opts.file_type);
        let name = format!("{}.{}", file_id, opts.file_type.suffix());
        let mut retry_cnt = 0;
        let file_type = self.get_file_type_from_key(&key);
        let start_time_with_retry = Instant::now();
        loop {
            let start_time = Instant::now();
            let req = self.new_request("HEAD", &key);
            let result = self.dispatch(req, HeadObjectError::from_response).await;
            match result {
                Ok(resp) => {
                    // Try to parse Content-Length
                    let len = match resp
                        .headers
                        .get("Content-Length")
                        .and_then(|v| v.parse::<u64>().ok())
                    {
                        Some(len) => len,
                        None => {
                            error!(
                                "missing or invalid Content-Length header for file {}",
                                &name
                            );
                            return Err(dfs::Error::Other(format!(
                                "missing or invalid Content-Length header for file {}",
                                &name
                            )));
                        }
                    };
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
                    return Ok(len);
                }
                Err(err) => {
                    if let RusotoError::Service(HeadObjectError::NoSuchKey(err_msg)) = &err {
                        error!("file {} not exist, err msg {}", &name, err_msg);
                        return Err(dfs::Error::NoSuchKey(format!("file {} not exist,", &name)));
                    }
                    if self.is_err_not_found(&err) {
                        error!("file {} not exist, err {}", &name, err.to_string());
                        return Err(dfs::Error::NoSuchKey(format!("file {} not exist,", &name)));
                    }
                    if self.is_err_retryable(&err)
                        && self.sleep_for_retry(&mut retry_cnt, &name).await
                    {
                        KVENGINE_DFS_RETRY_COUNTER
                            .head
                            .get(MetricsFileType::from(file_type))
                            .inc();
                        warn!("retry head file {}, error {:?}", &name, &err);
                        continue;
                    }
                    return Err(err.into());
                }
            }
        }
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

#[cfg(any(test, feature = "testexport"))]
pub mod test_util {
    use std::sync::{Arc, Mutex};

    use rusoto_core::signature::SignedRequest;
    use rusoto_mock::{MockCredentialsProvider, MockRequestDispatcher};

    use crate::dfs::S3Fs;

    struct State {
        created: bool,
        file_data: Vec<u8>,
    }

    #[derive(Clone)]
    struct S3TestDispatcher {
        state: Arc<Mutex<State>>,
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
                        let (created, file_len) = {
                            let state = state.lock().unwrap();
                            (state.created, state.file_data.len())
                        };
                        if created {
                            MockRequestDispatcher::with_status(200)
                                .with_header("Content-Length", &file_len.to_string())
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
                                .with_header(
                                    http::header::CONTENT_LENGTH.as_str(),
                                    &file_data.len().to_string(),
                                )
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

    pub fn new_test_s3fs(file_data: &[u8]) -> S3Fs {
        let state = Arc::new(Mutex::new(State {
            created: false,
            file_data: file_data.to_vec(),
        }));
        let dispatcher = S3TestDispatcher { state };
        let s3c = rusoto_core::Client::new_with(MockCredentialsProvider, dispatcher);
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .thread_name("test-s3-client")
            .enable_all()
            .build()
            .unwrap();

        S3Fs::new_for_test(runtime, s3c, "shard-db".into(), "prefix".into())
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

        // Test size before create - should return error
        let fs = s3fs.clone();
        let (tx_size_before, rx_size_before) = tikv_util::mpsc::bounded(1);
        let f_size_before = async move {
            match fs.size(file_id, opts).await {
                Ok(_) => {
                    tx_size_before.send(false).unwrap(); // Should not succeed
                }
                Err(_) => {
                    tx_size_before.send(true).unwrap(); // Expected: error
                }
            }
        };
        s3fs.runtime.spawn(f_size_before);
        assert!(rx_size_before.recv().unwrap());

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

        // Test size after create
        let fs = s3fs.clone();
        let (tx_size_after, rx_size_after) = tikv_util::mpsc::bounded(1);
        let expected_size = file_data.len() as u64;
        let f_size_after = async move {
            match fs.size(file_id, opts).await {
                Ok(size) => {
                    tx_size_after.send((true, size)).unwrap();
                }
                Err(_err) => {
                    tx_size_after.send((false, 0)).unwrap();
                }
            }
        };
        s3fs.runtime.spawn(f_size_after);
        let (success, size) = rx_size_after.recv().unwrap();
        assert!(success);
        assert_eq!(size, expected_size);

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
}
