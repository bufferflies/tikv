// Copyright 2023 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    collections::HashMap,
    convert::Infallible,
    net::SocketAddr,
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicU16, Ordering},
        Arc,
    },
    time::Duration,
};

use bytes::Buf;
use futures::StreamExt;
use hyper::{
    service::{make_service_fn, service_fn},
    Body, Method, Request, Response, Server, StatusCode,
};
use rand::Rng;
use tikv_util::{debug, error, info, time::Instant, warn};
use tokio::{fs, fs::File, io::AsyncWriteExt, runtime::Runtime, sync::oneshot, task::JoinHandle};
use tokio_util::codec::{BytesCodec, FramedRead};
use url::form_urlencoded;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
type HttpResult = std::result::Result<Response<Body>, hyper::Error>;

struct ServiceContext {
    store_path: PathBuf,
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
        ctx: Arc<ServiceContext>,
        req: Request<Body>,
    ) -> Result<Response<Body>> {
        let (parts, mut body) = req.into_parts();
        let file_path = Self::make_file_path(&ctx.store_path, parts.uri.path());
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
            ctx.store_path.to_str().unwrap(),
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

    async fn handle_get_object(
        ctx: Arc<ServiceContext>,
        req: Request<Body>,
    ) -> Result<Response<Body>> {
        let (parts, _) = req.into_parts();
        let file_path = Self::make_file_path(&ctx.store_path, parts.uri.path());

        let res = if let Ok(file) = File::open(file_path).await {
            let stream = FramedRead::new(file, BytesCodec::new());
            let body = Body::wrap_stream(stream);
            Response::new(body)
        } else {
            Self::not_found()
        };
        Ok(res)
    }

    fn is_copy_object_request(req: &Request<Body>) -> bool {
        req.headers().contains_key("x-amz-copy-source")
    }

    async fn handle_copy_object(
        _ctx: Arc<ServiceContext>,
        _req: Request<Body>,
    ) -> Result<Response<Body>> {
        // TODO
        warn!("handle_copy_object: request ignored");
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
        _ctx: Arc<ServiceContext>,
        _req: Request<Body>,
    ) -> Result<Response<Body>> {
        // TODO
        warn!("handle_tagging_object: request ignored");
        Ok(Response::default())
    }

    async fn service(ctx: Arc<ServiceContext>, req: Request<Body>) -> HttpResult {
        let res: Result<Response<Body>> = match *req.method() {
            Method::PUT if Self::is_copy_object_request(&req) => {
                Self::handle_copy_object(ctx, req).await
            }
            Method::PUT if Self::is_tagging_object_request(&req) => {
                Self::handle_tagging_object(ctx, req).await
            }
            Method::PUT => Self::handle_put_object(ctx, req).await,
            Method::GET => Self::handle_get_object(ctx, req).await,
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
        let ctx = Arc::new(ServiceContext {
            store_path: self.store_path.clone(),
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
    use kvengine::dfs::{Options, DFS, S3FS};

    use super::*;

    const TEST_COUNT: usize = 100;

    #[test]
    fn test_oss() {
        test_util::init_log_for_test();

        let base_dir = tempfile::Builder::new()
            .prefix("test_oss_")
            .tempdir()
            .unwrap();

        let mut oss = ObjectStorageService::new(base_dir.path());
        oss.start_server();

        let s3fs = S3FS::new(
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
                let mut buf = [0u8; 1024];
                rng.fill(&mut buf);
                Bytes::from(buf.to_vec())
            };
            let fs = s3fs.clone();

            let handle = runtime.spawn(async move {
                fs.create(file_id, write_data.clone(), options)
                    .await
                    .unwrap();

                let read_data = fs.read_file(file_id, options).await.unwrap();
                assert_eq!(write_data, read_data);

                if idx % 7 == 0 {
                    fs.remove(file_id, options).await;
                }
            });
            handles.push(handle);
        }

        runtime.block_on(futures::future::join_all(handles));
        oss.shutdown();
    }
}
