// Copyright 2023 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    error::Error as StdError,
    future::Future,
    sync::Arc,
    time::{Duration, Instant},
};

use bytes::{Buf, Bytes};
use cloud_encryption::MasterKey;
use flate2::{write::GzEncoder, Compression};
use http::{
    header::{ACCEPT_ENCODING, CONTENT_ENCODING, CONTENT_TYPE},
    HeaderValue, Request, Response,
};
use hyper::{
    server::accept::Accept,
    service::{make_service_fn, service_fn},
    Body,
};
use kvengine::{
    dfs,
    dfs::{CacheFs, S3Fs},
    table::{sstable::BlockCacheKey, ChecksumType},
    SnapAccess,
};
use pd_client::PdClient;
use prometheus::TEXT_FORMAT;
use protobuf::Message;
use rfstore::store::{PdIdAllocator, RegionSnapshot};
use tikv::{
    coprocessor::{
        remote_dispatcher::decode_remote_cop_request, REQ_TYPE_ANALYZE, REQ_TYPE_CHECKSUM,
        REQ_TYPE_DAG,
    },
    server::status_server::StatusServer,
};
use tikv_util::{
    error, info,
    metrics::{dump, dump_to},
    quota_limiter::QuotaLimiter,
    time::InstantExt,
};
use tokio::io::{AsyncRead, AsyncWrite};

use crate::{
    cop_limiter::CopLimiter,
    load_data::{self, LoadDataManager},
    metrics::{
        REMOTE_ANALYZE_REQ_COUNTER, REMOTE_ANALYZE_RESP_SIZE, REMOTE_CHECKSUM_REQ_COUNTER,
        REMOTE_CHECKSUM_RESP_SIZE, REMOTE_COMPACT_REQ_HANDLE_HISTOGRAM,
        REMOTE_COPR_DAG_REQ_COUNTER, REMOTE_COPR_DAG_RESP_SIZE, REMOTE_COPR_REQ_HANDLE_HISTOGRAM,
        REMOTE_COPR_SNAPSHOT_HISTOGRAM,
    },
    native_br::{self, NativeBrManager},
    txn_chunk::handle_txn_chunk,
};

pub(crate) struct Context {
    pub compression_lvl: i32,
    pub checksum_type: ChecksumType,
    pub s3fs: Arc<S3Fs>,
    pub cache_fs: Arc<CacheFs>,
    pub load_manager: Arc<LoadDataManager>,
    pub br_manager: Arc<NativeBrManager>,
    pub pd: Arc<dyn PdClient>,
    pub master_key: MasterKey,
    pub quota_limiter: Arc<QuotaLimiter>,
    pub block_cache: Option<moka::sync::SegmentedCache<BlockCacheKey, Bytes>>,
    pub cop_limiter: CopLimiter,
}

#[macro_export]
macro_rules! start_serve {
    ($ctx:expr, $acceptor:expr) => {{
        match $acceptor {
            tikv_util::Either::Left(acceptor) => {
                $crate::server::start($ctx, hyper::server::Server::builder(acceptor))
            }
            tikv_util::Either::Right(acceptor) => {
                $crate::server::start($ctx, hyper::server::Server::builder(acceptor))
            }
        }
    }};
}

pub(crate) fn start<I, C>(
    ctx: Arc<Context>,
    builder: hyper::server::Builder<I>,
) -> Box<dyn Future<Output = hyper::Result<()>> + Send + Unpin>
where
    I: Accept<Conn = C, Error = std::io::Error> + Send + Unpin + 'static,
    I::Error: Into<Box<dyn StdError + Send + Sync>>,
    I::Conn: AsyncRead + AsyncWrite + Unpin + Send + 'static,
{
    let server = builder.serve(make_service_fn(move |_| {
        let ctx = ctx.clone();
        async move {
            // Create a status service.
            Ok::<_, hyper::Error>(service_fn(move |req: hyper::Request<hyper::Body>| {
                let ctx = ctx.clone();
                async move {
                    let path = req.uri().path().to_owned();
                    match path.as_ref() {
                        "/healthz" => Ok(hyper::Response::builder()
                            .status(200)
                            .body(hyper::Body::from("ok"))
                            .unwrap()),
                        "/compact" => {
                            let ob_start = Instant::now();

                            let allocator = Arc::new(PdIdAllocator::new(ctx.pd.clone()));
                            let resp = kvengine::handle_remote_compaction(
                                ctx.s3fs.clone(),
                                req,
                                ctx.compression_lvl,
                                ctx.checksum_type,
                                allocator,
                                ctx.master_key.clone(),
                            )
                            .await;

                            if resp.is_ok() && resp.as_ref().unwrap().status().is_success() {
                                REMOTE_COMPACT_REQ_HANDLE_HISTOGRAM
                                    .observe(ob_start.saturating_elapsed().as_secs_f64());
                            }
                            resp
                        }
                        "/coprocessor" => handle_remote_coprocessor(ctx, req).await,
                        "/load_data" => {
                            load_data::handle_load_data(ctx.load_manager.clone(), req).await
                        }
                        "/metrics" => handle_get_metrics(req).await,
                        "/debug/pprof/profile" => {
                            StatusServer::<u8, u8>::dump_cpu_prof_to_resp(req).await
                        }
                        native_br::BACKUPS_API_PATH => {
                            native_br::handle_backup(ctx.br_manager.clone(), req).await
                        }
                        path if path.starts_with(native_br::RESTORE_KEYSPACE_API_PATH) => {
                            native_br::handle_restore_keyspace(ctx.br_manager.clone(), req).await
                        }
                        "/txn_chunk" => handle_txn_chunk(ctx, req).await,
                        _ => Ok(hyper::Response::builder()
                            .status(404)
                            .body(hyper::Body::from("Not Found"))
                            .unwrap()),
                    }
                }
            }))
        }
    }));
    Box::new(server)
}

async fn handle_remote_coprocessor(
    ctx: Arc<Context>,
    req: hyper::Request<hyper::Body>,
) -> hyper::Result<hyper::Response<hyper::Body>> {
    let req_body = hyper::body::to_bytes(req.into_body()).await?;
    let decode_res = decode_remote_cop_request(req_body.chunk());
    if let Err(err) = decode_res {
        let body = hyper::Body::from(format!("{:?}", err));
        return Ok(hyper::Response::builder().status(500).body(body).unwrap());
    }
    let (req_data, mem_data, snap_data) = decode_res.unwrap();
    let mut cop_req = kvproto::coprocessor::Request::default();
    if let Err(err) = cop_req.merge_from_bytes(req_data) {
        let body = hyper::Body::from(format!("{:?}", err));
        return Ok(hyper::Response::builder().status(500).body(body).unwrap());
    }
    let cop_ctx = cop_req.get_context();
    let keyspace_id = cop_ctx.keyspace_id;
    let timeout = Duration::from_millis(cop_ctx.get_max_execution_duration_ms());
    if !ctx
        .cop_limiter
        .wait_for_high_mem_usage(keyspace_id, timeout)
        .await
    {
        let body = hyper::Body::from("memory pressure is too high");
        return Ok(hyper::Response::builder().status(500).body(body).unwrap());
    }
    let req_type = cop_req.get_tp();
    let dfs: Arc<dyn dfs::Dfs> = match req_type {
        REQ_TYPE_DAG => ctx.cache_fs.clone(),
        _ => ctx.s3fs.clone(),
    };

    let ob_start = Instant::now();
    let snap_access_res = SnapAccess::construct_snapshot(
        dfs,
        mem_data,
        snap_data,
        &ctx.master_key,
        ctx.block_cache.clone(),
    )
    .await;
    if let Err(err) = snap_access_res.as_ref() {
        let body = hyper::Body::from(format!("{:?}", err));
        return Ok(hyper::Response::builder().status(500).body(body).unwrap());
    }
    REMOTE_COPR_SNAPSHOT_HISTOGRAM.observe(ob_start.saturating_elapsed().as_secs_f64());
    let snap_access = snap_access_res.unwrap();
    let req_type_str = match req_type {
        REQ_TYPE_DAG => "dag".to_string(),
        REQ_TYPE_ANALYZE => "analyze".to_string(),
        REQ_TYPE_CHECKSUM => "checksum".to_string(),
        _ => "".to_string(),
    };
    let tag = format!(
        "{} ks{}:{}:{}",
        req_type_str,
        snap_access.get_keyspace_id(),
        snap_access.get_id(),
        snap_access.get_version()
    );
    let snap = RegionSnapshot::from_snapshot(snap_access);
    let ob_start = Instant::now();
    let result = tikv::coprocessor::parse_request_and_handle_remote_cop(
        cop_req,
        None,
        Duration::from_secs(60),
        ctx.quota_limiter.clone(),
        snap,
    )
    .await;
    if let Err(err) = result {
        error!("{} remote coprocessor failed, error {:?}", tag, err);
        let body = hyper::Body::from(format!("{:?}", err));
        return Ok(hyper::Response::builder().status(500).body(body).unwrap());
    }
    REMOTE_COPR_REQ_HANDLE_HISTOGRAM.observe(ob_start.saturating_elapsed().as_secs_f64());
    let response = result.unwrap();
    info!(
        "{} finished remote coprocessor resp size {}",
        tag,
        response.data.len()
    );

    match req_type {
        REQ_TYPE_DAG => {
            REMOTE_COPR_DAG_REQ_COUNTER.inc();
            REMOTE_COPR_DAG_RESP_SIZE.inc_by(response.data.len() as u64);
        }
        REQ_TYPE_ANALYZE => {
            REMOTE_ANALYZE_REQ_COUNTER.inc();
            REMOTE_ANALYZE_RESP_SIZE.inc_by(response.data.len() as u64);
        }
        REQ_TYPE_CHECKSUM => {
            REMOTE_CHECKSUM_REQ_COUNTER.inc();
            REMOTE_CHECKSUM_RESP_SIZE.inc_by(response.data.len() as u64);
        }
        _ => {}
    }
    ctx.cop_limiter
        .add_sample(keyspace_id, response.data.len() as u64, Instant::now());
    let response_data = response.write_to_bytes().unwrap();
    Ok(hyper::Response::builder()
        .status(200)
        .body(response_data.into())
        .unwrap())
}

async fn handle_get_metrics(req: Request<Body>) -> hyper::Result<Response<Body>> {
    let gz_encoding = client_accept_gzip(&req);
    let metrics = if gz_encoding {
        // gzip can reduce the body size to less than 1/10.
        let mut encoder = GzEncoder::new(vec![], Compression::default());
        dump_to(&mut encoder, true);
        encoder.finish().unwrap()
    } else {
        dump(true).into_bytes()
    };
    let mut resp = Response::new(metrics.into());
    resp.headers_mut()
        .insert(CONTENT_TYPE, HeaderValue::from_static(TEXT_FORMAT));
    if gz_encoding {
        resp.headers_mut()
            .insert(CONTENT_ENCODING, HeaderValue::from_static("gzip"));
    }
    Ok(resp)
}

// check if the client allow return response with gzip compression
// the following logic is port from prometheus's golang:
// https://github.com/prometheus/client_golang/blob/24172847e35ba46025c49d90b8846b59eb5d9ead/prometheus/promhttp/http.go#L155-L176
fn client_accept_gzip(req: &Request<Body>) -> bool {
    let encoding = req
        .headers()
        .get(ACCEPT_ENCODING)
        .map(|enc| enc.to_str().unwrap_or_default())
        .unwrap_or_default();
    encoding
        .split(',')
        .map(|s| s.trim())
        .any(|s| s == "gzip" || s.starts_with("gzip;"))
}
