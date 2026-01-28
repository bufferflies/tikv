// Copyright 2023 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    cmp,
    collections::HashMap,
    convert::Infallible,
    io::SeekFrom,
    net::SocketAddr,
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicU16, AtomicU32, Ordering},
        Arc, Mutex,
    },
    time::Duration,
};

use anyhow::{anyhow, bail, Context};
use bytes::Buf;
use chrono::{DateTime, Utc};
use dashmap::DashMap;
use engine_traits::ListObjectContent;
use filetime::FileTime;
use futures::{future::ok, StreamExt, TryStreamExt};
use glob::glob;
use hyper::{
    header,
    header::HeaderValue,
    service::{make_service_fn, service_fn},
    Body, HeaderMap, Method, Request, Response, Server, StatusCode,
};
use kvengine::dfs::{CommonPrefix, DFSConfig, ListObjects, Tagging, STORAGE_CLASS_DEFAULT};
use rand::Rng;
use regex::Regex;
use tempfile::TempDir;
use tikv_util::{debug, error, info, time::Instant};
use tokio::{
    fs,
    fs::File,
    io::{AsyncReadExt, AsyncSeekExt, AsyncWriteExt},
    runtime::Runtime,
    sync::oneshot,
    task::JoinHandle,
};
use url::form_urlencoded;

type Result<T> = std::result::Result<T, anyhow::Error>;
type HttpResult = std::result::Result<Response<Body>, hyper::Error>;
type DelayRules = Arc<DashMap<String /* path keyword */, u64 /* delay ms */>>;

struct ServiceContext {
    store_path: PathBuf,
    tagging: Mutex<HashMap<String, Tagging>>, // file path -> Tagging
    delay_ms: Arc<AtomicU32>,
    put_delay_rules: DelayRules,
    as_if_date: Arc<Mutex<Option<DateTime<Utc>>>>,
    // Throttle injection: returns HTTP 429 (Too Many Requests).
    throttle_count: Arc<AtomicU32>,
}

impl ServiceContext {
    fn _add_tag(&self, file_path: String, key: String, value: String) {
        self.tagging
            .lock()
            .unwrap()
            .entry(file_path)
            .and_modify(|t| t.add_tag(key.clone(), value.clone()))
            .or_insert({
                let mut tagging = Tagging::default();
                tagging.add_tag(key, value);
                tagging
            });
    }

    fn insert_tagging(&self, file_path: String, tagging: Tagging) {
        self.tagging.lock().unwrap().insert(file_path, tagging);
    }

    fn get_tag(&self, file_path: &String) -> Option<Tagging> {
        self.tagging.lock().unwrap().get(file_path).cloned()
    }

    fn remove_tags(&self, file_path: String) {
        self.tagging.lock().unwrap().remove(&file_path);
    }

    async fn do_delay(&self) {
        let delay_ms = self.delay_ms.load(Ordering::Relaxed);
        if delay_ms > 0 {
            tokio::time::sleep(Duration::from_millis(delay_ms as u64)).await;
        }
    }

    async fn do_put_delay(&self, path: &str) {
        let mut delay_ms: Option<u64> = None;
        for r in self.put_delay_rules.iter() {
            if path.contains(r.key()) {
                delay_ms = Some(*r.value());
                break;
            }
        }

        if let Some(delay_ms) = delay_ms {
            tokio::time::sleep(Duration::from_millis(delay_ms)).await;
        }
    }

    fn maybe_override_mtime(&self, file_path: &Path) -> Result<()> {
        if let Some(as_if_date) = *self.as_if_date.lock().unwrap() {
            let file_time = FileTime::from_system_time(as_if_date.into());
            filetime::set_file_mtime(file_path, file_time)?;
        }
        Ok(())
    }

    /// Check if request should be throttled. Returns 429 (Too Many Requests) if
    /// so.
    fn maybe_throttle(&self) -> Option<Response<Body>> {
        // Atomically decrement only if count > 0 to avoid race conditions.
        let res = self
            .throttle_count
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |count| {
                if count > 0 { Some(count - 1) } else { None }
            });
        if res.is_err() {
            return None;
        }
        Some(
            Response::builder()
                .status(StatusCode::TOO_MANY_REQUESTS)
                .body(Body::from("QpsLimitExceeded"))
                .unwrap(),
        )
    }
}

pub struct ObjectStorageService {
    store_path: PathBuf,
    svc_handle: Option<JoinHandle<()>>,
    notify: Arc<tokio::sync::Notify>,      // for shutdown
    close_tx: Option<oneshot::Sender<()>>, // for graceful shutdown
    port: Arc<AtomicU16>,
    runtime: Runtime,
    delay_ms: Arc<AtomicU32>,
    put_delay_rules: DelayRules,
    as_if_date: Arc<Mutex<Option<DateTime<Utc>>>>,
    // Throttle injection: returns 429.
    throttle_count: Arc<AtomicU32>,
}

impl ObjectStorageService {
    pub fn new(store_path: impl Into<PathBuf>) -> Self {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(4)
            .enable_all()
            .thread_name("oss")
            .build()
            .unwrap();
        Self {
            store_path: store_path.into(),
            svc_handle: None,
            notify: Arc::new(tokio::sync::Notify::new()),
            close_tx: None,
            port: Default::default(),
            runtime,
            delay_ms: Default::default(),
            put_delay_rules: Default::default(),
            as_if_date: Arc::new(Mutex::new(None)),
            throttle_count: Default::default(),
        }
    }

    pub fn port(&self) -> u16 {
        self.port.load(Ordering::Acquire)
    }

    pub fn set_delay(&self, delay: Duration) {
        self.delay_ms
            .store(delay.as_millis() as u32, Ordering::Relaxed);
    }

    pub fn set_put_delay(&self, keyword: &str, delay: Duration) {
        self.put_delay_rules
            .insert(keyword.to_string(), delay.as_millis() as u64);
    }

    pub fn set_as_if_date(&self, as_if_date: Option<DateTime<Utc>>) {
        let mut guard = self.as_if_date.lock().unwrap();
        *guard = as_if_date;
    }

    /// Configure mock S3 to return 429 for the next `count` requests.
    pub fn set_throttle(&self, count: u32) {
        self.throttle_count.store(count, Ordering::Relaxed);
    }

    /// Remove objects under `prefix` whose modified time is older than
    /// `expire_dur`.
    ///
    /// The current time is anchored to `as_if_date` if it is set; otherwise it
    /// falls back to `Utc::now()`. Returns the number of removed objects.
    pub fn expire_path(&self, prefix: &str, expire_dur: Duration) -> Result<usize> {
        let root = self
            .store_path
            .join(prefix.trim_start_matches('/'))
            .to_path_buf();

        if !root.exists() {
            return Ok(0);
        }

        let now = self.as_if_date.lock().unwrap().unwrap_or_else(Utc::now);
        let expire = chrono::Duration::from_std(expire_dur)
            .unwrap_or_else(|_| chrono::Duration::max_value());

        let mut removed = 0usize;
        let mut stack = vec![root];
        while let Some(path) = stack.pop() {
            if path.is_dir() {
                for entry in std::fs::read_dir(&path)? {
                    stack.push(entry?.path());
                }
                continue;
            }

            let metadata = match std::fs::metadata(&path) {
                Ok(meta) => meta,
                Err(e) => {
                    debug!("skip path metadata error"; "path" => %path.display(), "error" => %e);
                    continue;
                }
            };

            let modified = match metadata.modified() {
                Ok(m) => m,
                Err(e) => {
                    debug!("skip path get mtime error"; "path" => %path.display(), "error" => %e);
                    continue;
                }
            };

            let modified: DateTime<Utc> = modified.into();
            let age = now.signed_duration_since(modified);
            if age >= expire {
                match std::fs::remove_file(&path) {
                    Ok(_) => {
                        removed += 1;
                        debug!("expired object removed"; "path" => %path.display());
                    }
                    Err(e) => {
                        error!("failed to remove expired object"; "path" => %path.display(), "error" => %e);
                    }
                }
            }
        }

        Ok(removed)
    }

    fn make_file_path(store_path: &Path, uri: &str) -> PathBuf {
        store_path.join(uri.strip_prefix('/').unwrap())
    }

    async fn handle_put_object(
        ctx: Arc<ServiceContext>,
        req: Request<Body>,
    ) -> Result<Response<Body>> {
        let (parts, mut body) = req.into_parts();
        let file_path = Self::make_file_path(&ctx.store_path, parts.uri.path());
        let parent = file_path
            .parent()
            .ok_or(anyhow!("fail to get parent for {:?}", file_path))?;
        let tmp_file_path = {
            let file_name = file_path
                .file_name()
                .ok_or(anyhow!("fail to get file name for {:?}", file_path))?
                .to_str()
                .unwrap();
            parent.to_path_buf().join(format!(
                "{}.{}.tmp",
                file_name,
                rand::thread_rng().gen::<u16>()
            ))
        };

        ctx.do_delay().await;
        ctx.do_put_delay(parts.uri.path()).await;

        fs::create_dir_all(parent).await?;
        let mut file = File::create(&tmp_file_path).await?;
        debug!(
            "handle_put_object: ready to save object, store_path: {:?}, file_path: {}, tmp_file_path: {}",
            ctx.store_path,
            file_path.to_str().unwrap(),
            tmp_file_path.to_str().unwrap()
        );

        while let Some(chunk) = body.next().await {
            file.write_all(chunk?.chunk()).await?;
        }
        file.sync_all().await?;
        let len = file.metadata().await?.len();
        drop(file);
        fs::rename(&tmp_file_path, &file_path).await?;
        ctx.maybe_override_mtime(&file_path)?;
        // sync_dir, see `file_system::sync_dir`
        File::open(parent).await?.sync_all().await?;

        info!(
            "handle_put_object: object save succeed, local file: {}, len: {}",
            file_path.to_str().unwrap(),
            len
        );
        let resp = Response::new(Body::from(format!("file length {}", len)));
        Ok(resp)
    }

    /// Parse header "Range" and return the range to read.
    ///
    /// Note that the return `end` is exclusive, i.e. [start, end), while the
    /// "end" in "Range" header is inclusive.
    ///
    /// When `start` is None, it means read the last `end` bytes.
    ///
    /// Ref: https://www.rfc-editor.org/rfc/rfc9110.html#section-14.1.2, Byte Ranges.
    fn parse_get_object_req_range(
        headers: &HeaderMap<HeaderValue>,
    ) -> Result<(Option<u64>, Option<u64>)> {
        lazy_static::lazy_static! {
            static ref RE: Regex = Regex::new(r"bytes=(\d*)-(\d*)").unwrap();
        }

        let Some(v) = headers.get(header::RANGE) else {
            return Ok((None, None));
        };
        let Some(matches) = RE.captures(v.to_str().unwrap()) else {
            bail!("bad range: {:?}", v);
        };

        let start_str = matches.get(1).unwrap().as_str();
        let start = start_str.parse::<u64>().ok();
        let end_str = matches.get(2).unwrap().as_str();
        // Convert to exclusive end.
        let end = end_str
            .parse::<u64>()
            .map(|x| if start.is_some() { x + 1 } else { x })
            .ok();
        Ok((start, end))
    }

    async fn handle_head_object(
        ctx: Arc<ServiceContext>,
        req: Request<Body>,
    ) -> Result<Response<Body>> {
        let (parts, _) = req.into_parts();
        let file_path = Self::make_file_path(&ctx.store_path, parts.uri.path());
        info!(
            "handle_head_object: file_path {}",
            file_path.to_str().unwrap()
        );
        let res = match fs::metadata(file_path.to_str().unwrap()).await {
            Ok(_) => Response::new(Body::empty()),
            Err(_) => {
                info!("handle_get_object: path not found: {}", parts.uri.path());
                Self::not_found()
            }
        };
        Ok(res)
    }

    async fn handle_delete_object(
        ctx: Arc<ServiceContext>,
        req: Request<Body>,
    ) -> Result<Response<Body>> {
        let (parts, _) = req.into_parts();
        let file_path = Self::make_file_path(&ctx.store_path, parts.uri.path());
        info!(
            "handle_delete_object: file_path {}",
            file_path.to_str().unwrap()
        );
        let res = match fs::remove_file(file_path.to_str().unwrap()).await {
            Ok(_) => Response::new(Body::empty()),
            Err(_) => {
                info!("handle_get_object: path not found: {}", parts.uri.path());
                Self::not_found()
            }
        };
        Ok(res)
    }

    async fn handle_get_object(
        ctx: Arc<ServiceContext>,
        req: Request<Body>,
    ) -> Result<Response<Body>> {
        let (parts, _) = req.into_parts();
        let file_path = Self::make_file_path(&ctx.store_path, parts.uri.path());
        let Ok((start, end)) = Self::parse_get_object_req_range(&parts.headers) else {
            return Ok(Self::bad_request("bad range".to_string()));
        };
        let res = if let Ok(mut file) = File::open(&file_path).await {
            ctx.do_delay().await;

            let file_len = file.metadata().await.context("metadata")?.len();
            info!(
                "handle_get_object: file_path {}, len {}, range [{:?}, {:?})",
                file_path.to_str().unwrap(),
                file_len,
                start,
                end
            );

            let (body, content_range) = match (start, end) {
                (None, None) => {
                    let mut buf = Vec::with_capacity(file_len as usize);
                    let read_len = file.read_to_end(&mut buf).await.context("read_to_end")?;
                    assert_eq!(read_len, file_len as usize);
                    (Body::from(buf), None)
                }
                (start, end) => {
                    let (start, end) = match (start, end) {
                        (Some(s), Some(e)) => (s, cmp::min(e, file_len)),
                        (Some(s), None) => (s, file_len),
                        (None, Some(e)) => (file_len.saturating_sub(e), file_len),
                        (None, None) => unreachable!(),
                    };

                    if start >= end {
                        // Note: S3 accept "start >= end" and return the whole object. We are more
                        // strict here.
                        return Ok(Self::bad_request(format!(
                            "invalid range: [{}, {})",
                            start, end
                        )));
                    }

                    file.seek(SeekFrom::Start(start)).await.context("seek")?;
                    let read_total = end - start;
                    let mut buf = vec![0; read_total as usize];
                    file.read_exact(buf.as_mut_slice())
                        .await
                        .context("read_exact")?;

                    // Ref: https://www.rfc-editor.org/rfc/rfc9110.html#section-14.4, Content-Range.
                    let range_resp = format!("bytes {start}-{end}/{file_len}");

                    (Body::from(buf), Some(range_resp))
                }
            };
            let mut resp = Response::builder();
            if let Some(content_range) = content_range {
                resp = resp.header("Content-Range", content_range);
            }
            resp.status(StatusCode::OK).body(body).unwrap()
        } else {
            info!("handle_get_object: path not found: {}", parts.uri.path());
            Self::not_found()
        };
        Ok(res)
    }

    fn handle_get_object_tag(
        ctx: Arc<ServiceContext>,
        req: Request<Body>,
    ) -> Result<Response<Body>> {
        let (parts, _) = req.into_parts();
        let file_path = Self::make_file_path(&ctx.store_path, parts.uri.path());
        let file = file_path.as_os_str().to_str().unwrap().to_owned();
        let tagging = ctx.get_tag(&file).unwrap_or_default();
        info!("Get object {} tag: {:?}", file, tagging);
        let tagging_xml = quick_xml::se::to_string(&tagging).unwrap();
        Ok(Response::new(Body::from(tagging_xml)))
    }

    fn handle_delete_object_tag(
        ctx: Arc<ServiceContext>,
        req: Request<Body>,
    ) -> Result<Response<Body>> {
        let (parts, _) = req.into_parts();
        let file_path = Self::make_file_path(&ctx.store_path, parts.uri.path());
        let file = file_path.as_os_str().to_str().unwrap().to_owned();
        info!("Delete object {} tag", file);
        ctx.remove_tags(file);
        Ok(Response::default())
    }

    fn is_copy_object_request(req: &Request<Body>) -> bool {
        req.headers().contains_key("x-amz-copy-source")
    }

    async fn handle_copy_object(
        ctx: Arc<ServiceContext>,
        req: Request<Body>,
    ) -> Result<Response<Body>> {
        let (parts, _) = req.into_parts();

        let copy_source = parts
            .headers
            .get("x-amz-copy-source")
            .ok_or(anyhow!("x-amz-copy-source is missing"))?
            .to_str()
            .unwrap();
        let target = parts.uri.path().strip_prefix('/').unwrap();
        let file_path = Self::make_file_path(&ctx.store_path, parts.uri.path());
        let file = file_path.as_os_str().to_str().unwrap().to_owned();

        if copy_source != target {
            let source = Self::make_file_path(&ctx.store_path, &format!("/{}", copy_source));
            let parent = file_path.parent().unwrap();
            info!("moving local file"; "copy_source" => copy_source, "target" => target,
                "file_path" => %file_path.display(), "source" => %source.display());
            std::fs::create_dir_all(parent)
                .with_context(|| format!("Failed to create directory: {}", parent.display()))?;
            std::fs::copy(&source, &file_path).with_context(|| {
                format!(
                    "Failed to copy file from {} to {}",
                    source.display(),
                    file_path.display(),
                )
            })?;
        }

        ctx.maybe_override_mtime(&file_path)?;

        let tagging_directive = parts
            .headers
            .get("x-amz-tagging-directive")
            .map(|x| x.to_str().unwrap())
            .unwrap_or("COPY");
        if tagging_directive == "REPLACE" {
            let tagging_str = parts
                .headers
                .get("x-amz-tagging")
                .ok_or(anyhow!("x-amz-tagging is missing"))?;
            let tagging = Tagging::from_url_encoded(tagging_str.to_str().unwrap());
            info!("Copy object {} with replacing tagging: {:?}", file, tagging);
            ctx.insert_tagging(file, tagging);
        }

        // TODO: support storage class

        Ok(Response::default())
    }

    fn is_tagging_object_request(req: &Request<Body>) -> bool {
        if let Some(query) = req.uri().query() {
            let params = form_urlencoded::parse(query.as_bytes())
                .into_owned()
                .collect::<HashMap<String, String>>();
            return params.contains_key("tagging");
        }
        false
    }

    async fn handle_tagging_object(
        ctx: Arc<ServiceContext>,
        req: Request<Body>,
    ) -> Result<Response<Body>> {
        let (parts, body) = req.into_parts();
        let file_path = Self::make_file_path(&ctx.store_path, parts.uri.path());
        let file = file_path.as_os_str().to_str().unwrap().to_owned();
        let mut body_content = Vec::new();
        body.try_for_each(|bytes| {
            body_content.extend(bytes);
            ok(())
        })
        .await?;
        let tagging: Tagging = quick_xml::de::from_slice(&body_content).unwrap();
        info!("Put object {} tag {:?}", file, tagging);
        // Only support replace tagging for now. TODO: Support append tags.
        ctx.insert_tagging(file, tagging);
        Ok(Response::default())
    }

    fn is_list_objects_request(req: &Request<Body>) -> bool {
        if let Some(query) = req.uri().query() {
            form_urlencoded::parse(query.as_bytes()).any(|(k, _)| k == "list-type")
        } else {
            false
        }
    }

    /// Ref: https://docs.aws.amazon.com/AmazonS3/latest/API/API_ListObjectsV2.html
    ///
    /// list-type: must be 2.
    ///
    /// prefix: list keys begin with `prefix`, e.g. `backup/`.
    ///
    /// start-after: list keys after `start-after` key, e.g.
    /// `backup/20230701/000000.meta`.
    ///
    /// Note: "bucket" should not be included in key.
    async fn handle_list_objects(
        ctx: Arc<ServiceContext>,
        req: Request<Body>,
    ) -> Result<Response<Body>> {
        let query = req.uri().query().unwrap();
        let params = form_urlencoded::parse(query.as_bytes())
            .into_owned()
            .collect::<HashMap<String, String>>();
        info!("handle_list_objects: request: {:?}", params);

        assert_eq!(params.get("list-type").unwrap(), "2"); // Support ListObjectsV2 only.
        let prefix = params.get("prefix").cloned().unwrap_or_default();
        let start_after = params.get("start-after");
        let max_keys = params
            .get("max-keys")
            .map(|x| x.parse::<usize>().unwrap())
            .unwrap_or(1000);
        let delimiter = params.get("delimiter");

        let bucket_path = Self::make_file_path(&ctx.store_path, req.uri().path());
        let list_path = bucket_path.join(prefix);
        let list_pattern = list_path.to_str().unwrap().to_owned() + "*";

        let mut common_prefixes = vec![];
        let mut files = vec![];
        let mut add_file = |file: PathBuf| {
            files.push(file);
        };
        glob(&list_pattern) // use `glob` to support prefix when it's not a directory.
            .unwrap_or_else(|e| panic!("invalid pattern {}, {:?}", list_pattern, e))
            .filter_map(|x| x.ok())
            .for_each(|x| {
                if delimiter.is_some() {
                    if x.is_dir() {
                        let prefix = x
                            .strip_prefix(&bucket_path)
                            .unwrap()
                            .to_str()
                            .map(|x| {
                                if x.ends_with('/') {
                                    x.to_string()
                                } else {
                                    format!("{}/", x)
                                }
                            })
                            .unwrap();
                        common_prefixes.push(CommonPrefix { prefix });
                    } else {
                        add_file(x);
                    }
                } else {
                    visit_path(x, &mut add_file);
                }
            });
        files.sort_unstable();
        let start = if let Some(start_after) = start_after {
            let start_after = bucket_path.join(start_after);
            files
                .iter()
                .position(|x| x > &start_after)
                .unwrap_or(files.len())
        } else {
            0
        };

        let end = cmp::min(start + max_keys, files.len());
        let contents = files[start..end]
            .iter()
            .filter_map(|x| {
                // Filter out tmp files.
                if x.to_str().unwrap().ends_with(".tmp") {
                    None
                } else {
                    let metadata = x.metadata().unwrap();
                    let last_modified: DateTime<Utc> = metadata.modified().unwrap().into();
                    Some(ListObjectContent {
                        key: x
                            .strip_prefix(&bucket_path)
                            .unwrap()
                            .to_str()
                            .unwrap()
                            .to_owned(),
                        last_modified: last_modified.to_rfc3339(),
                        storage_class: STORAGE_CLASS_DEFAULT.to_owned(),
                        size: metadata.len(),
                    })
                }
            })
            .collect::<Vec<_>>();
        let list_objects = ListObjects {
            common_prefixes,
            contents,
            is_truncated: end < files.len(),
        };
        info!("handle_list_objects: result: {:?}", list_objects);
        let list_objects_xml = quick_xml::se::to_string(&list_objects).unwrap();
        Ok(Response::new(Body::from(list_objects_xml)))
    }

    async fn service(ctx: Arc<ServiceContext>, req: Request<Body>) -> HttpResult {
        // Check for throttle injection first.
        if let Some(throttle_resp) = ctx.maybe_throttle() {
            return Ok(throttle_resp);
        }
        let res: Result<Response<Body>> = match *req.method() {
            Method::PUT if Self::is_copy_object_request(&req) => {
                Self::handle_copy_object(ctx, req).await
            }
            Method::PUT if Self::is_tagging_object_request(&req) => {
                Self::handle_tagging_object(ctx, req).await
            }
            Method::PUT => Self::handle_put_object(ctx, req).await,
            Method::GET if Self::is_tagging_object_request(&req) => {
                Self::handle_get_object_tag(ctx, req)
            }
            Method::GET if Self::is_list_objects_request(&req) => {
                Self::handle_list_objects(ctx, req).await
            }
            Method::HEAD => Self::handle_head_object(ctx, req).await,
            Method::GET => Self::handle_get_object(ctx, req).await,
            Method::DELETE if Self::is_tagging_object_request(&req) => {
                Self::handle_delete_object_tag(ctx, req)
            }
            Method::DELETE => Self::handle_delete_object(ctx, req).await,
            _ => Ok(Self::method_not_found()),
        };
        res.or_else(|e| {
            Ok(Response::builder()
                .status(StatusCode::INTERNAL_SERVER_ERROR)
                .body(Body::from(format!("{:?}", e)))
                .unwrap())
        })
    }

    fn not_found() -> Response<Body> {
        Response::builder()
            .status(StatusCode::NOT_FOUND)
            .body(Body::from("Not Found"))
            .unwrap()
    }

    fn method_not_found() -> Response<Body> {
        Response::builder()
            .status(StatusCode::METHOD_NOT_ALLOWED)
            .body(Body::from("method not allowed"))
            .unwrap()
    }

    fn bad_request(msg: String) -> Response<Body> {
        Response::builder()
            .status(StatusCode::BAD_REQUEST)
            .body(Body::from(msg))
            .unwrap()
    }

    pub fn start_server(&mut self) {
        assert!(self.svc_handle.is_none(), "server has started");

        let addr = SocketAddr::from(([127, 0, 0, 1], 0));
        let ctx = Arc::new(ServiceContext {
            store_path: self.store_path.clone(),
            tagging: Default::default(),
            delay_ms: self.delay_ms.clone(),
            put_delay_rules: self.put_delay_rules.clone(),
            as_if_date: self.as_if_date.clone(),
            throttle_count: self.throttle_count.clone(),
        });
        let make_svc = make_service_fn(move |_conn| {
            let ctx = ctx.clone();
            async move {
                Ok::<_, Infallible>(service_fn(move |req| {
                    let ctx = ctx.clone();
                    async move { Self::service(ctx, req).await }
                }))
            }
        });

        let (close_tx, close_rx) = oneshot::channel();

        let port = self.port.clone();
        let notify = self.notify.clone();
        let svc_handle = self.runtime.spawn(async move {
            let server = Server::bind(&addr).serve(make_svc);
            port.store(server.local_addr().port(), Ordering::Release);
            let graceful = server.with_graceful_shutdown(async {
                close_rx.await.ok();
            });
            tokio::select! {
                _ = notify.notified() => {
                    info!("server shutdown");
                }
                res = graceful => {
                    if let Err(e) = res {
                        error!("server error: {}", e);
                    } else {
                        info!("server graceful shutdown");
                    }
                }
            }
        });
        self.svc_handle = Some(svc_handle);
        self.close_tx = Some(close_tx);

        let start = Instant::now();
        while self.port() == 0 {
            std::thread::sleep(Duration::from_millis(100));
            if start.saturating_elapsed() > Duration::from_secs(3) {
                panic!("start_server failed");
            }
        }
        info!("start_server on port {}", self.port());
    }

    // Note: In tests, drop all clients (e.g. `S3Fs`) before calling this method.
    pub fn graceful_shutdown(&mut self) {
        if let Some(handle) = self.svc_handle.take() {
            let close_tx = self.close_tx.take().unwrap();
            close_tx.send(()).unwrap();
            self.runtime.block_on(async { handle.await.unwrap() })
        }
    }

    pub fn shutdown(&mut self) {
        if let Some(handle) = self.svc_handle.take() {
            self.notify.notify_waiters();
            self.runtime.block_on(async { handle.await.unwrap() })
        }
    }
}

// Ref: https://doc.rust-lang.org/std/fs/fn.read_dir.html
fn visit_path(path: PathBuf, cb: &mut dyn FnMut(PathBuf)) {
    if path.is_dir() {
        for entry in std::fs::read_dir(&path).unwrap() {
            visit_path(entry.unwrap().path(), cb);
        }
    } else {
        cb(path);
    }
}

pub fn prepare_dfs(prefix: &str) -> (TempDir, ObjectStorageService, DFSConfig) {
    let base_dir = tempfile::Builder::new().prefix(prefix).tempdir().unwrap();

    let oss_dir = base_dir.path().join("oss");
    let mut oss = ObjectStorageService::new(oss_dir);
    oss.start_server();

    let dfs_config = DFSConfig {
        prefix: prefix.to_string(),
        s3_endpoint: format!("http://127.0.0.1:{}", oss.port()),
        s3_key_id: "admin".to_string(),
        s3_secret_key: "admin".to_string(),
        s3_bucket: prefix.to_string(),
        s3_region: "local".to_string(),
        zstd_compression_level: "3".to_string(),
        ..Default::default()
    };

    (base_dir, oss, dfs_config)
}

#[cfg(test)]
mod tests {
    use std::{
        io::{Read, Seek},
        os::unix::fs::FileExt,
        time::Duration,
    };

    use bytes::Bytes;
    use chrono::TimeZone;
    use kvengine::{
        dfs,
        dfs::{Dfs, FileType, Options, S3Fs},
    };
    use rand::prelude::*;
    use tempfile::tempfile;
    use tikv_util::time::Instant;

    use super::*;

    const TEST_COUNT: usize = 100;
    const TEST_DATA_SIZE: usize = 1024;

    fn random_range(rng: &mut ThreadRng) -> (usize, usize) {
        let len = rng.gen_range(1..TEST_DATA_SIZE / 2);
        let start = rng.gen_range(0..(TEST_DATA_SIZE - len));
        (start, start + len)
    }

    #[test]
    fn test_oss_basic() {
        test_util::init_log_for_test();

        let base_dir = tempfile::Builder::new()
            .prefix("test_oss_")
            .tempdir()
            .unwrap();

        let mut oss = ObjectStorageService::new(base_dir.path());
        oss.start_server();

        let s3fs = S3Fs::new(
            "oss_test".to_string(),
            format!("http://127.0.0.1:{}", oss.port()),
            "admin".to_string(),
            "admin".to_string(),
            "local".to_string(),
            "cse_test".to_string(),
        );

        let runtime = s3fs.get_runtime();
        let mut rng = rand::thread_rng();
        let mut handles = Vec::with_capacity(TEST_COUNT);
        for idx in 0..TEST_COUNT {
            let options = Options::default();
            let file_id = rng.gen::<u32>() as u64;
            let write_data = {
                let mut buf = [0u8; TEST_DATA_SIZE];
                rng.fill(&mut buf);
                Bytes::from(buf.to_vec())
            };
            let fs = s3fs.clone();
            let range = random_range(&mut rng);

            let handle = runtime.spawn(async move {
                fs.create(file_id, write_data.clone(), options)
                    .await
                    .unwrap();

                let read_data = fs.read_file(file_id, options).await.unwrap();
                assert_eq!(write_data, read_data);

                {
                    let partial_read_data = fs
                        .read_file(file_id, options.with_start_off(range.0 as u64))
                        .await
                        .unwrap();
                    assert_eq!(write_data.slice(range.0..), partial_read_data);
                }

                let key = fs.file_key(file_id, FileType::Sst);
                let exist = fs.exist(key.clone(), file_id.to_string()).await.unwrap();
                assert!(exist);
                let opts = engine_traits::GetObjectOptions {
                    start_off: Some(range.0 as u64),
                    end_off: Some(range.1 as u64),
                };
                let read_data = fs
                    .get_object(key.clone(), file_id.to_string(), opts)
                    .await
                    .unwrap();
                assert_eq!(write_data.slice(range.0..range.1), read_data);
                let opts = engine_traits::GetObjectOptions {
                    start_off: Some(range.0 as u64),
                    end_off: None,
                };
                let read_data = fs
                    .get_object(key.clone(), file_id.to_string(), opts)
                    .await
                    .unwrap();
                assert_eq!(write_data.slice(range.0..), read_data);

                if idx % 7 == 0 {
                    fs.remove(file_id, None, options).await;
                }

                if idx % 7 == 6 {
                    fs.delete_object(key.clone(), file_id.to_string())
                        .await
                        .unwrap();
                    let exist = fs.exist(key.clone(), file_id.to_string()).await.unwrap();
                    assert!(!exist);
                    let err = fs
                        .get_object(
                            key,
                            file_id.to_string(),
                            engine_traits::GetObjectOptions::default(),
                        )
                        .await
                        .err();
                    let no_such_key = matches!(err, Some(dfs::Error::NoSuchKey(_)));
                    assert!(no_such_key);
                }
            });
            handles.push(handle);
        }

        runtime.block_on(futures::future::join_all(handles));
        drop(s3fs);
        oss.graceful_shutdown();
    }

    #[test]
    fn test_oss_as_if_date_and_expire() {
        test_util::init_log_for_test();

        let base_dir = tempfile::Builder::new()
            .prefix("test_oss_as_if_date_")
            .tempdir()
            .unwrap();

        let mut oss = ObjectStorageService::new(base_dir.path());
        let create_date = Utc.with_ymd_and_hms(2020, 1, 2, 3, 4, 5).single().unwrap();
        oss.set_as_if_date(Some(create_date));
        oss.start_server();

        let bucket = "bkt";
        let prefix = "pfx";
        let s3fs = S3Fs::new(
            prefix.to_string(),
            format!("http://127.0.0.1:{}", oss.port()),
            "admin".to_string(),
            "admin".to_string(),
            "local".to_string(),
            bucket.to_string(),
        );
        let runtime = s3fs.get_runtime();

        // Put the first object with the initial as_if_date.
        let old_key = format!("{}/old.obj", prefix);
        runtime
            .block_on(s3fs.put_object(
                old_key.clone(),
                Bytes::from("hello".as_bytes().to_vec()),
                old_key.clone(),
            ))
            .unwrap();

        let old_path = oss.store_path.join(format!("{}/{}", bucket, old_key));
        let old_mtime: DateTime<Utc> = std::fs::metadata(&old_path)
            .unwrap()
            .modified()
            .unwrap()
            .into();
        assert_eq!(old_mtime, create_date);

        // Advance the reference time and create a second object.
        let ref_date = create_date + chrono::Duration::days(10);
        oss.set_as_if_date(Some(ref_date));
        let new_key = format!("{}/new.obj", prefix);
        runtime
            .block_on(s3fs.put_object(
                new_key.clone(),
                Bytes::from("world".as_bytes().to_vec()),
                new_key.clone(),
            ))
            .unwrap();
        let new_path = oss.store_path.join(format!("{}/{}", bucket, new_key));
        let new_mtime: DateTime<Utc> = std::fs::metadata(&new_path)
            .unwrap()
            .modified()
            .unwrap()
            .into();
        assert_eq!(new_mtime, ref_date);

        // Expire files older than 5 days relative to the current as_if_date.
        let removed = oss
            .expire_path(bucket, Duration::from_secs(5 * 24 * 3600))
            .unwrap();
        assert_eq!(removed, 1);
        assert!(!old_path.exists());
        assert!(new_path.exists());

        drop(s3fs);
        oss.graceful_shutdown();
    }

    #[test]
    fn test_oss_shutdown() {
        test_util::init_log_for_test();

        let base_dir = tempfile::Builder::new()
            .prefix("test_oss_shutdown")
            .tempdir()
            .unwrap();

        let mut oss = ObjectStorageService::new(base_dir.path());
        oss.start_server();

        let s3fs = S3Fs::new(
            "oss_test".to_string(),
            format!("http://127.0.0.1:{}", oss.port()),
            "admin".to_string(),
            "admin".to_string(),
            "local".to_string(),
            "cse_test".to_string(),
        );
        let runtime = s3fs.get_runtime();

        let file_id = 42;
        let key = s3fs.file_key(file_id, FileType::Sst);
        runtime
            .block_on(s3fs.create(
                file_id,
                Bytes::from("test_oss_shutdown".to_string()),
                Options::default(),
            ))
            .unwrap();

        let fs = s3fs.clone();
        let key_clone = key.clone();
        let handle: JoinHandle<bool /* request_timeout */> = runtime.spawn(async move {
            let start = Instant::now_coarse();
            while start.saturating_elapsed() < Duration::from_secs(5) {
                // When oss is shutdown, the exist request must timeout.
                match tokio::time::timeout(
                    Duration::from_secs(2),
                    fs.exist(key_clone.clone(), file_id.to_string()),
                )
                .await
                {
                    Ok(exist) => {
                        assert!(exist.unwrap());
                        tokio::time::sleep(Duration::from_millis(100)).await;
                    }
                    Err(_) => return true,
                }
            }
            false
        });

        let exist = runtime
            .block_on(s3fs.exist(key, file_id.to_string()))
            .unwrap();
        assert!(exist);
        std::thread::sleep(Duration::from_millis(500));
        oss.shutdown();
        let request_timeout = runtime.block_on(handle).unwrap();
        assert!(request_timeout);
    }

    #[test]
    fn test_oss_retain_file() {
        test_util::init_log_for_test();

        let base_dir = tempfile::Builder::new()
            .prefix("test_oss_tag_")
            .tempdir()
            .unwrap();

        let mut oss = ObjectStorageService::new(base_dir.path());
        oss.start_server();

        let s3fs = S3Fs::new(
            "oss_test".to_string(),
            format!("http://127.0.0.1:{}", oss.port()),
            "admin".to_string(),
            "admin".to_string(),
            "local".to_string(),
            "cse_test".to_string(),
        );

        let runtime = s3fs.get_runtime();
        let mut rng = rand::thread_rng();
        let mut handles = Vec::with_capacity(TEST_COUNT);
        for _ in 0..TEST_COUNT {
            let options = Options::default();
            let file_id = rng.gen::<u32>() as u64;
            let write_data = {
                let mut buf = [0u8; TEST_DATA_SIZE];
                rng.fill(&mut buf);
                Bytes::from(buf.to_vec())
            };
            let fs = s3fs.clone();
            let handle = runtime.spawn(async move {
                fs.create(file_id, write_data.clone(), options)
                    .await
                    .unwrap();
                fs.remove(file_id, None, options).await;
                let file_key = fs.file_key(file_id, FileType::Sst);
                assert!(fs.is_removed(&file_key).await.unwrap());
                fs.retain_file(&file_key).await.unwrap();
                assert!(!fs.is_removed(&file_key).await.unwrap());
            });
            handles.push(handle);
        }

        runtime.block_on(futures::future::join_all(handles));
        drop(s3fs);
        oss.graceful_shutdown();
    }

    #[test]
    fn test_oss_list_objects() {
        test_util::init_log_for_test();

        let base_dir = tempfile::Builder::new()
            .prefix("test_oss_list_objects_")
            .tempdir()
            .unwrap();

        let mut oss = ObjectStorageService::new(base_dir.path());
        oss.start_server();

        let s3fs = S3Fs::new(
            "pfx".to_string(),
            format!("http://127.0.0.1:{}", oss.port()),
            "admin".to_string(),
            "admin".to_string(),
            "local".to_string(),
            "bkt".to_string(),
        );

        let prefix = s3fs.get_prefix();
        s3fs.get_runtime().block_on(async {
            {
                let (objects, has_more, next_start_after) =
                    s3fs.list("", None, None).await.unwrap();
                assert!(objects.is_empty());
                assert!(!has_more);
                assert!(next_start_after.is_none());
            }

            let mut keys_a = Vec::new();
            for i in 0..10 {
                let key = format!("{}/A_{:04}.meta", prefix, i);
                keys_a.push(key.clone());
                s3fs.put_object(
                    key.clone(),
                    Bytes::from(b"x".repeat(i as usize).to_vec()),
                    key,
                )
                .await
                .unwrap();
            }
            let mut keys_b = Vec::new();
            for i in 0..20 {
                let key = format!("{}/B/{:04}.meta", prefix, i);
                keys_b.push(key.clone());
                s3fs.put_object(
                    key.clone(),
                    Bytes::from(b"xx".repeat(i as usize).to_vec()),
                    key,
                )
                .await
                .unwrap();
            }

            {
                // list all
                let (objects, has_more, next_start_after) =
                    s3fs.list("", None, None).await.unwrap();
                assert_eq!(objects.len(), 30);
                assert!(!has_more);
                assert!(next_start_after.is_none());
                for i in 0..10 {
                    assert_eq!(objects[i].key, keys_a[i]);
                    assert_eq!(objects[i].size, i as u64);
                    let _ = DateTime::parse_from_rfc3339(&objects[i].last_modified).unwrap();
                }
                for i in 0..20 {
                    assert_eq!(objects[i + 10].key, keys_b[i]);
                    assert_eq!(objects[i + 10].size, (i + i) as u64);
                    let _ = DateTime::parse_from_rfc3339(&objects[i + 10].last_modified).unwrap();
                }
            }

            {
                // list with prefix
                let (objects, has_more, ..) = s3fs.list("", Some("B/"), None).await.unwrap();
                assert_eq!(objects.len(), 20);
                assert!(!has_more);
                for i in 0..20 {
                    assert_eq!(objects[i].key, keys_b[i]);
                }
            }

            {
                // list with prefix & start_after
                let (objects, has_more, ..) =
                    s3fs.list("0009.meta", Some("B/"), None).await.unwrap();
                assert_eq!(objects.len(), 10);
                assert!(!has_more);
                for i in 10..20 {
                    assert_eq!(objects[i - 10].key, keys_b[i]);
                }
            }

            {
                // list with prefix & start_after & max_keys
                let mut start_after = "0009.meta".to_string();
                let mut objects = vec![];
                loop {
                    let (mut objs, _, next_start_after) =
                        s3fs.list(&start_after, Some("B/"), Some(3)).await.unwrap();
                    assert!(objs.len() <= 3);
                    objects.append(&mut objs);
                    if let Some(next) = next_start_after {
                        start_after = next;
                    } else {
                        break;
                    }
                }
                assert_eq!(objects.len(), 10);
                for i in 10..20 {
                    assert_eq!(objects[i - 10].key, keys_b[i]);
                }
            }

            {
                // list all with delimiter
                let folders = s3fs.list_folders("", None).await.unwrap();
                assert_eq!(folders.len(), 1);
                assert_eq!(folders[0], format!("{}/B/", s3fs.get_prefix()));
            }

            {
                // list with prefix and delimiter
                let folders = s3fs.list_folders("B/", None).await.unwrap();
                assert_eq!(folders.len(), 0);
            }
        });

        drop(s3fs);
        oss.graceful_shutdown();
    }

    #[test]
    fn test_oss_get_object() {
        const OBJECT_SIZE: usize = 64 * 1024;

        test_util::init_log_for_test();

        let base_dir = tempfile::Builder::new()
            .prefix("test_oss_")
            .tempdir()
            .unwrap();

        let mut oss = ObjectStorageService::new(base_dir.path());
        oss.start_server();

        let s3fs = S3Fs::new(
            "oss_test".to_string(),
            format!("http://127.0.0.1:{}", oss.port()),
            "admin".to_string(),
            "admin".to_string(),
            "local".to_string(),
            "cse_test".to_string(),
        );

        let mut data = vec![0u8; OBJECT_SIZE];
        thread_rng().fill_bytes(data.as_mut_slice());
        let data = Bytes::from(data);
        let data_len = data.len();
        assert_eq!(data_len, OBJECT_SIZE);

        let runtime = s3fs.get_runtime();
        runtime.block_on(async {
            s3fs.create(1, data.clone(), Options::default())
                .await
                .unwrap();
            assert_eq!(s3fs.read_file(1, Options::default()).await.unwrap(), data);

            let cases = [
                // start_off, end_off, expected
                (Some(10), None, data.slice(10..)),
                (Some(10), Some(20), data.slice(10..20)),
                (None, None, data.clone()),
                (None, Some(100), data.slice(data_len - 100..data_len)),
                (None, Some(data_len as u64 + 1), data.clone()),
            ];

            for (start_off, end_off, expected) in cases {
                let s3fs = s3fs.clone();
                let key = s3fs.file_key(1, FileType::Sst);
                let file_name = "1".to_string();
                let opts = engine_traits::GetObjectOptions { start_off, end_off };
                let read_data = s3fs
                    .get_object(key.clone(), file_name.clone(), opts)
                    .await
                    .unwrap();
                assert_eq!(read_data, expected);

                let build_writer = || {
                    let f = tempfile().unwrap();
                    Ok(std::io::BufWriter::new(f))
                };
                let (writer, len) = s3fs
                    .get_object_to_writer(key, file_name, opts, build_writer)
                    .await
                    .unwrap();
                let mut f = writer.into_inner().unwrap();
                assert_eq!(len as usize, expected.len());
                {
                    let mut read_data = Vec::with_capacity(len as usize);
                    f.rewind().unwrap();
                    f.read_to_end(&mut read_data).unwrap();
                    assert_eq!(&read_data, expected.as_ref());
                }
                {
                    // Test for `read_exact_at`:
                    let mut read_data = vec![0; len as usize];
                    f.read_exact_at(&mut read_data, 0).unwrap();
                    assert_eq!(&read_data, expected.as_ref());
                }
            }
        });

        drop(s3fs);
        oss.graceful_shutdown();
    }

    /// Test S3 client retries with exponential backoff on 429 (Too Many
    /// Requests) errors.
    ///
    /// T1: Configure OSS to return 429 for next 2 requests
    /// T2: Create file (should retry with exponential backoff, then succeed)
    /// T3: Verify data integrity
    #[test]
    fn test_oss_retry_exponential_backoff() {
        test_util::init_log_for_test();

        let base_dir = tempfile::Builder::new()
            .prefix("test_oss_exp_backoff_")
            .tempdir()
            .unwrap();

        let mut oss = ObjectStorageService::new(base_dir.path());
        oss.start_server();

        let s3fs = S3Fs::new(
            "oss_test".to_string(),
            format!("http://127.0.0.1:{}", oss.port()),
            "admin".to_string(),
            "admin".to_string(),
            "local".to_string(),
            "cse_test".to_string(),
        );

        let runtime = s3fs.get_runtime();
        let file_id = 54321u64;
        let write_data = Bytes::from(b"test_exp_backoff_data".to_vec());

        // T1: Configure OSS to return 429 for next 2 requests.
        oss.set_throttle(2);

        // T2: Create file with exponential backoff retry.
        // retry_cnt=1: 2^1 * 500ms = 1000ms
        // retry_cnt=2: 2^2 * 500ms = 2000ms
        // Total: ~3000ms
        let start = Instant::now_coarse();
        runtime
            .block_on(s3fs.create(file_id, write_data.clone(), Options::default()))
            .unwrap();
        let elapsed = start.saturating_elapsed();

        // Verify exponential backoff timing (~3s with some variance).
        assert!(
            elapsed >= Duration::from_millis(2500),
            "expected >= 2500ms exponential backoff, got {:?}",
            elapsed
        );

        // T3: Verify data integrity.
        let read_data = runtime
            .block_on(s3fs.read_file(file_id, Options::default()))
            .unwrap();
        assert_eq!(write_data, read_data);

        drop(s3fs);
        oss.graceful_shutdown();
    }
}
