// Copyright 2023 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    collections::HashMap,
    convert::Infallible,
    io::SeekFrom,
    net::SocketAddr,
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicU16, Ordering},
        Arc, Mutex,
    },
    time::Duration,
};

use bytes::Buf;
use futures::{future::ok, StreamExt, TryStreamExt};
use hyper::{
    header::HeaderValue,
    service::{make_service_fn, service_fn},
    Body, HeaderMap, Method, Request, Response, Server, StatusCode,
};
use kvengine::dfs::Tagging;
use rand::Rng;
use tikv_util::{debug, error, info, time::Instant};
use tokio::{
    fs,
    fs::File,
    io::{AsyncReadExt, AsyncSeekExt, AsyncWriteExt},
    runtime::Runtime,
    sync::oneshot,
    task::JoinHandle,
};
use tokio_util::codec::{BytesCodec, FramedRead};
use url::form_urlencoded;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
type HttpResult = std::result::Result<Response<Body>, hyper::Error>;

struct ServiceContext {
    store_path: PathBuf,
    tagging: HashMap<String, Tagging>, // file path -> Tagging
}

impl ServiceContext {
    fn _add_tag(&mut self, file_path: String, key: String, value: String) {
        self.tagging
            .entry(file_path)
            .and_modify(|t| t.add_tag(key.clone(), value.clone()))
            .or_insert({
                let mut tagging = Tagging::default();
                tagging.add_tag(key, value);
                tagging
            });
    }

    fn insert_tagging(&mut self, file_path: String, tagging: Tagging) {
        self.tagging.insert(file_path, tagging);
    }

    fn get_tag(&self, file_path: &String) -> Option<Tagging> {
        self.tagging.get(file_path).cloned()
    }

    fn remove_tags(&mut self, file_path: String) {
        self.tagging.remove(&file_path);
    }
}

pub struct ObjectStorageService {
    store_path: PathBuf,
    svc_handle: Option<JoinHandle<()>>,
    close_tx: Option<oneshot::Sender<()>>,
    port: Arc<AtomicU16>,
    runtime: Runtime,
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
            close_tx: None,
            port: Default::default(),
            runtime,
        }
    }

    pub fn port(&self) -> u16 {
        self.port.load(Ordering::Acquire)
    }

    fn make_file_path(store_path: &Path, uri: &str) -> PathBuf {
        store_path.join(uri.strip_prefix('/').unwrap())
    }

    async fn handle_put_object(
        ctx: Arc<Mutex<ServiceContext>>,
        req: Request<Body>,
    ) -> Result<Response<Body>> {
        let (parts, mut body) = req.into_parts();
        let path = ctx.lock().unwrap().store_path.to_owned();
        let file_path = Self::make_file_path(&path, parts.uri.path());
        let parent = file_path
            .parent()
            .ok_or(format!("fail to get parent for {:?}", file_path))?;
        let tmp_file_path = {
            let file_name = file_path
                .file_name()
                .ok_or(format!("fail to get file name for {:?}", file_path))?
                .to_str()
                .unwrap();
            parent.to_path_buf().join(format!(
                "{}.tmp.{}",
                file_name,
                rand::thread_rng().gen::<u16>()
            ))
        };

        fs::create_dir_all(parent).await?;
        let mut file = File::create(&tmp_file_path).await?;
        debug!(
            "handle_put_object: ready to save object, store_path: {}, file_path: {}, tmp_file_path: {}",
            path.to_str().unwrap(),
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

    // Return none means get all content in file.
    // End is none means read to end.
    fn parse_get_object_req_range(headers: &HeaderMap<HeaderValue>) -> Option<(u64, Option<u64>)> {
        if let Some(v) = headers.get("Range") {
            let regex = regex::Regex::new(r"bytes=(\d+)-(\d*)").unwrap();
            let matches = regex.captures(v.to_str().unwrap()).unwrap();
            let start_str = matches.get(1).unwrap().as_str();
            let start = start_str.parse::<u64>().unwrap();
            let end_str = matches.get(2).unwrap().as_str();
            let end = if end_str.is_empty() {
                None
            } else {
                Some(end_str.parse::<u64>().unwrap())
            };
            Some((start, end))
        } else {
            None
        }
    }

    async fn handle_get_object(
        ctx: Arc<Mutex<ServiceContext>>,
        req: Request<Body>,
    ) -> Result<Response<Body>> {
        let (parts, _) = req.into_parts();
        let path = ctx.lock().unwrap().store_path.to_owned();
        let file_path = Self::make_file_path(&path, parts.uri.path());
        let range = Self::parse_get_object_req_range(&parts.headers);
        let res = if let Ok(mut file) = File::open(file_path).await {
            let body = match range {
                None => {
                    let stream = FramedRead::new(file, BytesCodec::new());
                    Body::wrap_stream(stream)
                }
                Some(r) => {
                    let end = if r.1.is_none() {
                        file.metadata().await.unwrap().len()
                    } else {
                        r.1.unwrap()
                    };
                    assert!(end > r.0);
                    // TODO: implement range read with FramedRead.
                    file.seek(SeekFrom::Start(r.0)).await.unwrap();
                    let mut buf = vec![0u8; (end - r.0) as usize];
                    file.read_exact(&mut buf).await.unwrap();
                    Body::from(buf)
                }
            };
            Response::new(body)
        } else {
            Self::not_found()
        };
        Ok(res)
    }

    fn handle_get_object_tag(
        ctx: Arc<Mutex<ServiceContext>>,
        req: Request<Body>,
    ) -> Result<Response<Body>> {
        let (parts, _) = req.into_parts();
        let path = ctx.lock().unwrap().store_path.to_owned();
        let file_path = Self::make_file_path(&path, parts.uri.path());
        let file = file_path.as_os_str().to_str().unwrap().to_owned();
        let tagging = ctx.lock().unwrap().get_tag(&file).unwrap_or_default();
        info!("Get object {} tag: {:?}", file, tagging);
        let tagging_xml = quick_xml::se::to_string(&tagging).unwrap();
        Ok(Response::new(Body::from(tagging_xml)))
    }

    fn handle_delete_object_tag(
        ctx: Arc<Mutex<ServiceContext>>,
        req: Request<Body>,
    ) -> Result<Response<Body>> {
        let (parts, _) = req.into_parts();
        let path = ctx.lock().unwrap().store_path.to_owned();
        let file_path = Self::make_file_path(&path, parts.uri.path());
        let file = file_path.as_os_str().to_str().unwrap().to_owned();
        info!("Delete object {} tag", file);
        ctx.lock().unwrap().remove_tags(file);
        Ok(Response::default())
    }

    fn is_copy_object_request(req: &Request<Body>) -> bool {
        req.headers().contains_key("x-amz-copy-source")
    }

    async fn handle_copy_object(
        ctx: Arc<Mutex<ServiceContext>>,
        req: Request<Body>,
    ) -> Result<Response<Body>> {
        let (parts, _) = req.into_parts();

        let copy_source = parts
            .headers
            .get("x-amz-copy-source")
            .ok_or("x-amz-copy-source is missing")?
            .to_str()
            .unwrap();
        let target = parts.uri.path().strip_prefix('/').unwrap();
        if copy_source != target {
            return Err(format!(
                "support copy from same source only, source {}, target {}",
                copy_source, target
            )
            .into());
        }

        let path = ctx.lock().unwrap().store_path.to_owned();
        let file_path = Self::make_file_path(&path, parts.uri.path());
        let file = file_path.as_os_str().to_str().unwrap().to_owned();

        let tagging_directive = parts
            .headers
            .get("x-amz-tagging-directive")
            .map(|x| x.to_str().unwrap())
            .unwrap_or("COPY");
        if tagging_directive == "REPLACE" {
            let tagging_str = parts
                .headers
                .get("x-amz-tagging")
                .ok_or("x-amz-tagging is missing")?;
            let tagging = Tagging::from_url_encoded(tagging_str.to_str().unwrap());
            info!("Copy object {} with replacing tagging: {:?}", file, tagging);
            ctx.lock().unwrap().insert_tagging(file, tagging);
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
        ctx: Arc<Mutex<ServiceContext>>,
        req: Request<Body>,
    ) -> Result<Response<Body>> {
        let (parts, body) = req.into_parts();
        let path = ctx.lock().unwrap().store_path.to_owned();
        let file_path = Self::make_file_path(&path, parts.uri.path());
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
        ctx.lock().unwrap().insert_tagging(file, tagging);
        Ok(Response::default())
    }

    async fn service(ctx: Arc<Mutex<ServiceContext>>, req: Request<Body>) -> HttpResult {
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
            Method::GET => Self::handle_get_object(ctx, req).await,
            Method::DELETE if Self::is_tagging_object_request(&req) => {
                Self::handle_delete_object_tag(ctx, req)
            }
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

    pub fn start_server(&mut self) {
        assert!(self.svc_handle.is_none(), "server has started");

        let addr = SocketAddr::from(([127, 0, 0, 1], 0));
        let ctx = Arc::new(Mutex::new(ServiceContext {
            store_path: self.store_path.clone(),
            tagging: HashMap::default(),
        }));
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
        let svc_handle = self.runtime.spawn(async move {
            let server = Server::bind(&addr).serve(make_svc);
            port.store(server.local_addr().port(), Ordering::Release);
            let graceful = server.with_graceful_shutdown(async {
                close_rx.await.ok();
            });
            if let Err(e) = graceful.await {
                error!("server error: {}", e);
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

    pub fn shutdown(&mut self) {
        if let Some(handle) = self.svc_handle.take() {
            let close_tx = self.close_tx.take().unwrap();
            close_tx.send(()).unwrap();
            self.runtime.block_on(async { handle.await.unwrap() })
        }
    }
}

#[cfg(test)]
mod tests {
    use bytes::Bytes;
    use kvengine::dfs::{Dfs, Options, S3Fs};
    use rand::prelude::ThreadRng;

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
            let options = Options::new(0, 0);
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

                let key = fs.file_key(file_id);
                let opts = engine_traits::GetObjectOptions {
                    start_off: range.0 as u64,
                    end_off: Some(range.1 as u64),
                };
                let read_data = fs
                    .get_object(key.clone(), file_id.to_string(), opts)
                    .await
                    .unwrap();
                assert_eq!(write_data.slice(range.0..range.1), read_data);
                let opts = engine_traits::GetObjectOptions {
                    start_off: range.0 as u64,
                    end_off: None,
                };
                let read_data = fs.get_object(key, file_id.to_string(), opts).await.unwrap();
                assert_eq!(write_data.slice(range.0..), read_data);

                if idx % 7 == 0 {
                    fs.remove(file_id, options).await;
                }
            });
            handles.push(handle);
        }

        runtime.block_on(futures::future::join_all(handles));
        oss.shutdown();
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
            let options = Options::new(0, 0);
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
                fs.remove(file_id, options).await;
                assert!(fs.is_removed(file_id).await.unwrap());
                fs.retain_file(file_id).await.unwrap();
                assert!(!fs.is_removed(file_id).await.unwrap());
            });
            handles.push(handle);
        }

        runtime.block_on(futures::future::join_all(handles));
        oss.shutdown();
    }
}
