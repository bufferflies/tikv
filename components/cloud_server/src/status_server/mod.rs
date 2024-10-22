// Copyright 2018 TiKV Project Authors. Licensed under Apache-2.0.

mod profile;

use std::{
    borrow::Cow,
    convert::TryInto,
    env::args,
    error::Error as StdError,
    net::SocketAddr,
    pin::Pin,
    str::{self, FromStr},
    sync::Arc,
    task::{Context, Poll},
    time::{Duration, Instant},
};

use api_version::ApiV2;
use async_stream::stream;
use bytes::{Buf, BufMut, BytesMut};
use collections::HashMap;
use flate2::{write::GzEncoder, Compression};
use futures::{
    compat::Compat01As03,
    future::{ok, poll_fn},
    prelude::*,
};
use hyper::{
    self, header,
    header::{HeaderValue, ACCEPT_ENCODING, CONTENT_ENCODING, CONTENT_TYPE},
    server::{
        accept::Accept,
        conn::{AddrIncoming, AddrStream},
        Builder as HyperBuilder,
    },
    service::{make_service_fn, service_fn},
    Body, Method, Request, Response, Server, StatusCode,
};
use kvengine::{
    dfs::{DFSConfig, FileType},
    Shard, ShardStats,
};
use kvproto::{coprocessor::DelegateResponse, raft_serverpb::StoreIdent};
use online_config::OnlineConfig;
use openssl::{
    ssl::{Ssl, SslAcceptor, SslFiletype, SslMethod, SslVerifyMode},
    x509::X509,
};
use pin_project::pin_project;
use profile::*;
use prometheus::TEXT_FORMAT;
use protobuf::Message;
use rfengine::{
    load_store_ident, raft_state_key, RfEngine, WriteBatch, KV_ENGINE_META_KEY,
    RAFT_TRUNCATED_STATE_KEY,
};
use rfstore::{
    store::{
        peer_storage::{collect_prefix_regions, load_raft_engine_meta, load_region_state},
        state::RaftState,
        Callback, CasualMessage, RegionSnapshot, StoreMsg, RAFT_INIT_LOG_INDEX, RAFT_INIT_LOG_TERM,
        TERM_KEY,
    },
    RaftRouter, RaftStoreRouter,
};
use security::{self, SecurityConfig};
use serde_json::Value;
use tikv::{
    config::{ConfigController, LogLevel},
    server::status_server::profile::start_one_cpu_profile,
    storage::CloudStore,
};
use tikv_util::{
    codec::bytes::decode_bytes,
    future::paired_future_callback,
    logger::set_log_level,
    metrics::{dump, dump_to},
    sys::thread::ThreadBuildWrapper,
    time::InstantExt,
    timer::GLOBAL_TIMER_HANDLE,
};
use tokio::{
    io::{AsyncRead, AsyncWrite},
    runtime::{Builder, Runtime},
    sync::oneshot::{self, Receiver, Sender},
};
use tokio_openssl::SslStream;
use txn_types::TsSet;

use crate::{
    server::Result, status_server::metrics::STATUS_REQ_HISTOGRAM_STATIC,
    tikv_util::codec::number::NumberEncoder,
};

mod metrics;

static TIMER_CANCELED: &str = "tokio timer canceled";

#[cfg(feature = "failpoints")]
static MISSING_NAME: &[u8] = b"Missing param name";
#[cfg(feature = "failpoints")]
static MISSING_ACTIONS: &[u8] = b"Missing param actions";
#[cfg(feature = "failpoints")]
static FAIL_POINTS_REQUEST_PATH: &str = "/fail";

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
struct LogLevelRequest {
    pub log_level: LogLevel,
}

#[derive(Deserialize, Default)]
#[serde(default)]
struct SyncRegionRequest {
    pub start: String,
    pub end: String,
    pub limit: usize,
    pub reverse: bool,
    pub debug: bool,
}

#[derive(Deserialize)]
#[serde(rename_all = "kebab-case")]
struct SyncRegionByIdRequest {
    pub region_id: u64,
    #[serde(default)]
    pub debug: bool,
}

pub struct StatusServer {
    thread_pool: Runtime,
    tx: Sender<()>,
    rx: Option<Receiver<()>>,
    addr: Option<SocketAddr>,
    cfg_controller: ConfigController,
    router: RaftRouter,
    security_config: Arc<SecurityConfig>,
    kvengine: kvengine::Engine,
    rfengine: rfengine::RfEngine,
}

impl StatusServer {
    pub fn new(
        status_thread_pool_size: usize,
        cfg_controller: ConfigController,
        security_config: Arc<SecurityConfig>,
        router: RaftRouter,
        kvengine: kvengine::Engine,
        rfengine: rfengine::RfEngine,
    ) -> Result<Self> {
        let thread_pool = Builder::new_multi_thread()
            .enable_all()
            .worker_threads(status_thread_pool_size)
            .thread_name("status-server")
            .after_start_wrapper(|| debug!("Status server started"))
            .before_stop_wrapper(|| debug!("stopping status server"))
            .build()?;

        let (tx, rx) = oneshot::channel::<()>();
        Ok(StatusServer {
            thread_pool,
            tx,
            rx: Some(rx),
            addr: None,
            cfg_controller,
            router,
            security_config,
            kvengine,
            rfengine,
        })
    }

    fn load_raft_state(rf: &RfEngine, peer_id: u64, ver: u64) -> Option<RaftState> {
        let key = raft_state_key(ver);
        let raft_state_val = match rf.get_state(peer_id, key.chunk()) {
            Some(val) => val,
            None => {
                error!("failed to load raft state");
                return None;
            }
        };
        let mut raft_state = RaftState::default();
        raft_state.unmarshal(&raft_state_val);
        Some(raft_state)
    }

    fn write_empty_engine_meta(
        rf: &RfEngine,
        wb: &mut WriteBatch,
        peer_id: u64,
        region_id: u64,
        ver: u64,
        index: u64,
        term: u64,
        inner_key_off: u32,
    ) -> Result<kvenginepb::ChangeSet> {
        let region_local_state = match load_region_state(rf, peer_id, ver) {
            Some(val) => val,
            None => {
                return Err(crate::server::Error::Other(box_err!(
                    "region state not found"
                )));
            }
        };
        let mut enc_start_key = region_local_state.get_region().get_start_key();
        let mut enc_end_key = region_local_state.get_region().get_end_key();
        let start_key = decode_bytes(&mut enc_start_key, false).unwrap();
        let end_key = decode_bytes(&mut enc_end_key, false).unwrap();

        let mut cs = kvenginepb::ChangeSet::new();
        cs.set_shard_id(region_id);
        cs.set_shard_ver(ver);
        cs.set_sequence(index);
        let snap = cs.mut_snapshot();
        snap.set_outer_start(start_key.to_vec());
        snap.set_outer_end(end_key.to_vec());
        snap.set_inner_key_off(inner_key_off);
        snap.set_data_sequence(index);
        let props = snap.mut_properties();
        props.set_shard_id(region_id);
        props.mut_keys().push(TERM_KEY.to_string());
        props.mut_values().push(term.to_le_bytes().to_vec());
        let cs_val = cs.write_to_bytes().unwrap();
        wb.set_state(peer_id, region_id, KV_ENGINE_META_KEY, &cs_val);
        Ok(cs)
    }

    fn write_truncated_state(
        wb: &mut WriteBatch,
        peer_id: u64,
        region_id: u64,
        index: u64,
        term: u64,
    ) {
        let mut ts_val = BytesMut::with_capacity(16);
        ts_val.put_u64_le(term);
        ts_val.put_u64_le(index);
        let ts = ts_val.freeze();
        wb.set_state(peer_id, region_id, RAFT_TRUNCATED_STATE_KEY, ts.chunk());
    }

    fn write_raft_state(
        wb: &mut WriteBatch,
        peer_id: u64,
        region_id: u64,
        ver: u64,
        index: u64,
        term: u64,
    ) -> RaftState {
        let mut rs_val = BytesMut::with_capacity(40);
        rs_val.put_u64_le(term);
        rs_val.put_u64_le(0);
        rs_val.put_u64_le(index);
        rs_val.put_u64_le(index);
        rs_val.put_u64_le(index);
        let rs = rs_val.freeze();
        let key = raft_state_key(ver);
        wb.set_state(peer_id, region_id, key.chunk(), rs.chunk());
        let mut raft_state = RaftState::default();
        raft_state.unmarshal(rs.chunk());
        raft_state
    }

    pub fn dump_heap_prof_to_resp(req: Request<Body>) -> hyper::Result<Response<Body>> {
        let query = req.uri().query().unwrap_or("");
        let query_pairs: HashMap<_, _> = url::form_urlencoded::parse(query.as_bytes()).collect();

        let use_jeprof = query_pairs.get("jeprof").map(|x| x.as_ref()) == Some("true");
        if use_jeprof {
            return Ok(make_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                "jeprof=true is not supported on CSE",
            ));
        }

        let result = {
            let file = match dump_one_heap_profile() {
                Ok(file) => file,
                Err(e) => return Ok(make_response(StatusCode::INTERNAL_SERVER_ERROR, e)),
            };
            let path = file.path();
            read_file(path.to_string_lossy().as_ref())
            // Note: file, which is a NamedTempFile, is supposed to be unlinked
            // when it is dropped here.
        };

        match result {
            Ok(body) => {
                info!("dump or get heap profile successfully");
                let mut response = Response::builder()
                    .header("X-Content-Type-Options", "nosniff")
                    .header("Content-Disposition", "attachment; filename=\"profile\"")
                    .header("Content-Length", body.len());
                response = if use_jeprof {
                    response.header("Content-Type", mime::IMAGE_SVG.to_string())
                } else {
                    response.header("Content-Type", mime::APPLICATION_OCTET_STREAM.to_string())
                };
                Ok(response.body(body.into()).unwrap())
            }
            Err(e) => {
                info!("dump or get heap profile fail: {}", e);
                Ok(make_response(StatusCode::INTERNAL_SERVER_ERROR, e))
            }
        }
    }

    async fn get_config(
        req: Request<Body>,
        cfg_controller: &ConfigController,
    ) -> hyper::Result<Response<Body>> {
        let mut full = false;
        if let Some(query) = req.uri().query() {
            let query_pairs: HashMap<_, _> =
                url::form_urlencoded::parse(query.as_bytes()).collect();
            full = match query_pairs.get("full") {
                Some(val) => match val.parse() {
                    Ok(val) => val,
                    Err(err) => return Ok(make_response(StatusCode::BAD_REQUEST, err.to_string())),
                },
                None => false,
            };
        }
        let encode_res = if full {
            // Get all config
            serde_json::to_string(&cfg_controller.get_current())
        } else {
            // Filter hidden config
            serde_json::to_string(&cfg_controller.get_current().get_encoder())
        };
        Ok(match encode_res {
            Ok(json) => Response::builder()
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(json))
                .unwrap(),
            Err(_) => make_response(StatusCode::INTERNAL_SERVER_ERROR, "Internal Server Error"),
        })
    }

    pub fn get_cmdline(_req: Request<Body>) -> hyper::Result<Response<Body>> {
        let args = args().fold(String::new(), |mut a, b| {
            a.push_str(&b);
            a.push('\x00');
            a
        });
        let response = Response::builder()
            .header("Content-Type", mime::TEXT_PLAIN.to_string())
            .header("X-Content-Type-Options", "nosniff")
            .body(args.into())
            .unwrap();
        Ok(response)
    }

    pub fn get_symbol_count(req: Request<Body>) -> hyper::Result<Response<Body>> {
        assert_eq!(req.method(), Method::GET);
        // We don't know how many symbols we have, but we
        // do have symbol information. pprof only cares whether
        // this number is 0 (no symbols available) or > 0.
        let text = "num_symbols: 1\n";
        let response = Response::builder()
            .header("Content-Type", mime::TEXT_PLAIN.to_string())
            .header("X-Content-Type-Options", "nosniff")
            .header("Content-Length", text.len())
            .body(text.into())
            .unwrap();
        Ok(response)
    }

    // The request and response format follows pprof remote server
    // https://gperftools.github.io/gperftools/pprof_remote_servers.html
    // Here is the go pprof implementation:
    // https://github.com/golang/go/blob/3857a89e7eb872fa22d569e70b7e076bec74ebbb/src/net/http/pprof/pprof.go#L191
    pub async fn get_symbol(req: Request<Body>) -> hyper::Result<Response<Body>> {
        assert_eq!(req.method(), Method::POST);
        let mut text = String::new();
        let body_bytes = hyper::body::to_bytes(req.into_body()).await?;
        let body = String::from_utf8(body_bytes.to_vec());
        if body.is_err() {
            return Ok(make_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                "Body is not a valid UTF8",
            ));
        }
        let body = body.unwrap();

        // The request body is a list of addr to be resolved joined by '+'.
        // Resolve addrs with addr2line and write the symbols each per line in
        // response.
        for pc in body.split('+') {
            let addr = usize::from_str_radix(pc.trim_start_matches("0x"), 16).unwrap_or(0);
            if addr == 0 {
                info!("invalid addr: {}", addr);
                continue;
            }

            // Would be multiple symbols if inlined.
            let mut syms = vec![];
            backtrace::resolve(addr as *mut std::ffi::c_void, |sym| {
                let name = sym
                    .name()
                    .unwrap_or_else(|| backtrace::SymbolName::new(b"<unknown>"));
                syms.push(name.to_string());
            });

            if !syms.is_empty() {
                // join inline functions with '--'
                let f = syms.join("--");
                // should be <hex address> <function name>
                text.push_str(format!("{:#x} {}\n", addr, f).as_str());
            } else {
                info!("can't resolve mapped addr: {:#x}", addr);
                text.push_str(format!("{:#x} ??\n", addr).as_str());
            }
        }
        let response = Response::builder()
            .header("Content-Type", mime::TEXT_PLAIN.to_string())
            .header("X-Content-Type-Options", "nosniff")
            .header("Content-Length", text.len())
            .body(text.into())
            .unwrap();
        Ok(response)
    }

    async fn update_config(
        cfg_controller: ConfigController,
        req: Request<Body>,
    ) -> hyper::Result<Response<Body>> {
        let mut body = Vec::new();
        req.into_body()
            .try_for_each(|bytes| {
                body.extend(bytes);
                ok(())
            })
            .await?;
        Ok(match decode_json(&body) {
            Ok(change) => match cfg_controller.update(change) {
                Err(e) => make_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("failed to update, error: {:?}", e),
                ),
                Ok(_) => {
                    let mut resp = Response::default();
                    *resp.status_mut() = StatusCode::OK;
                    resp
                }
            },
            Err(e) => make_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("failed to decode, error: {:?}", e),
            ),
        })
    }

    pub async fn dump_cpu_prof_to_resp(req: Request<Body>) -> hyper::Result<Response<Body>> {
        let query = req.uri().query().unwrap_or("");
        let query_pairs: HashMap<_, _> = url::form_urlencoded::parse(query.as_bytes()).collect();

        let seconds: u64 = match query_pairs.get("seconds") {
            Some(val) => match val.parse() {
                Ok(val) => val,
                Err(err) => return Ok(make_response(StatusCode::BAD_REQUEST, err.to_string())),
            },
            None => 10,
        };

        let frequency: i32 = match query_pairs.get("frequency") {
            Some(val) => match val.parse() {
                Ok(val) => val,
                Err(err) => return Ok(make_response(StatusCode::BAD_REQUEST, err.to_string())),
            },
            None => 99, /* Default frequency of sampling. 99Hz to avoid coincide with special
                         * periods */
        };

        let prototype_content_type: hyper::http::HeaderValue =
            hyper::http::HeaderValue::from_str("application/protobuf").unwrap();
        let output_protobuf = req.headers().get("Content-Type") == Some(&prototype_content_type);

        let timer = GLOBAL_TIMER_HANDLE.delay(Instant::now() + Duration::from_secs(seconds));
        let end = async move {
            Compat01As03::new(timer)
                .await
                .map_err(|_| TIMER_CANCELED.to_owned())
        };
        match start_one_cpu_profile(end, frequency, output_protobuf).await {
            Ok(body) => {
                info!("dump cpu profile successfully");
                Ok(make_response(StatusCode::OK, body))
            }
            Err(e) => {
                info!("dump cpu profile fail: {}", e);
                Ok(make_response(StatusCode::INTERNAL_SERVER_ERROR, e))
            }
        }
    }

    async fn change_log_level(req: Request<Body>) -> hyper::Result<Response<Body>> {
        let mut body = Vec::new();
        req.into_body()
            .try_for_each(|bytes| {
                body.extend(bytes);
                ok(())
            })
            .await?;

        let log_level_request: std::result::Result<LogLevelRequest, serde_json::error::Error> =
            serde_json::from_slice(&body);

        match log_level_request {
            Ok(req) => {
                set_log_level(req.log_level.into());
                Ok(Response::new(Body::empty()))
            }
            Err(err) => Ok(make_response(StatusCode::BAD_REQUEST, err.to_string())),
        }
    }

    // URI: /kvengine/snapshot/<shard_id>?start_ts=xxx&shard_ver=xxx
    // dump kvengine shard snapshot with start_ts
    async fn dump_kvengine_snapshot(
        req: Request<Body>,
        engine: kvengine::Engine,
        router: RaftRouter,
    ) -> hyper::Result<Response<Body>> {
        let path = req.uri().path();
        let last = get_last_path_segment(path);
        let query = req.uri().query().unwrap_or("");
        let query_pairs: HashMap<_, _> = url::form_urlencoded::parse(query.as_bytes()).collect();
        if !query_pairs.contains_key("start_ts") {
            return Ok(make_response(
                StatusCode::BAD_REQUEST,
                "start_ts is required".to_string(),
            ));
        }
        if !query_pairs.contains_key("shard_ver") {
            return Ok(make_response(
                StatusCode::BAD_REQUEST,
                "shard_ver is required".to_string(),
            ));
        }
        let shard_ver = match u64::from_str(query_pairs.get("shard_ver").unwrap()) {
            Ok(ver) => ver,
            Err(err) => return Ok(make_response(StatusCode::BAD_REQUEST, err.to_string())),
        };
        let shard_id = match u64::from_str(last) {
            Ok(id) => id,
            Err(err) => return Ok(make_response(StatusCode::BAD_REQUEST, err.to_string())),
        };
        let (cb, fut) = paired_future_callback();
        let callback = Callback::Read(Box::new(move |res| {
            cb(res);
        }));
        router.send_casual_msg(
            shard_id,
            CasualMessage::CheckLeader {
                shard_ver,
                callback,
            },
        );
        let mut res = match fut.await {
            Ok(res) => res,
            Err(e) => {
                let err_msg = format!("{} check leader channel error: {:?}", shard_id, e);
                error!("{}", err_msg);
                return Ok(make_response(StatusCode::INTERNAL_SERVER_ERROR, err_msg));
            }
        };
        let make_ok_response = |body: Vec<u8>| -> Response<Body> {
            Response::builder().body(Body::from(body)).unwrap()
        };
        let mut delegate_resp = DelegateResponse::default();
        let mut err = res.response.take_header().take_error();
        if err.has_not_leader() {
            delegate_resp
                .mut_region_error()
                .set_not_leader(err.take_not_leader());
            let body = delegate_resp.write_to_bytes().unwrap();
            return Ok(make_ok_response(body));
        } else if err.has_epoch_not_match() {
            delegate_resp
                .mut_region_error()
                .set_epoch_not_match(err.take_epoch_not_match());
            let body = delegate_resp.write_to_bytes().unwrap();
            return Ok(make_ok_response(body));
        }
        Ok(match engine.get_shard_with_ver(shard_id, shard_ver) {
            Ok(shard) => {
                let snap_access = shard.new_snap_access();
                let outer_ranges = vec![(shard.outer_start.clone(), shard.outer_end.clone())];
                let start_ts = u64::from_str(query_pairs.get("start_ts").unwrap()).unwrap();
                let region_snapshot = RegionSnapshot::from_snapshot(snap_access.clone());
                let cloud_store =
                    CloudStore::new(region_snapshot, start_ts, TsSet::default(), true);
                // Check if the shard contains locks belongs to the ranges. If there is any
                // lock, return the first key as LockInfo. The client should retry in this case.
                if let Err(tikv::coprocessor::Error::Locked(lock_info)) = cloud_store
                    .check_locks_in_range(&shard.outer_start, &shard.outer_end)
                    .map_err(tikv::coprocessor::Error::from)
                {
                    delegate_resp.set_locked(lock_info);
                    let body = delegate_resp.write_to_bytes().unwrap();
                    return Ok(make_ok_response(body));
                }
                let mem_data = snap_access.build_mem_data(&outer_ranges, start_ts);
                let (_, snap_data) = shard.new_snap_access().marshal(
                    &[(shard.outer_start.clone(), shard.outer_end.clone())],
                    false,
                    false,
                );
                delegate_resp.set_mem_table_data(mem_data);
                delegate_resp.set_snapshot(snap_data);
                let body = delegate_resp.write_to_bytes().unwrap();
                make_ok_response(body)
            }
            Err(e) => make_response(StatusCode::NOT_FOUND, e.to_string()),
        })
    }

    async fn dump_kvengine_meta(
        req: Request<Body>,
        engine: kvengine::Engine,
    ) -> hyper::Result<Response<Body>> {
        let path = req.uri().path();
        let last = get_last_path_segment(path);
        let meta_bin = u64::from_str(last)
            .map(|id| {
                engine
                    .get_shard(id)
                    .map(|shard| {
                        let (_, meta_bin) = shard.new_snap_access().marshal(
                            &[(shard.outer_start.clone(), shard.outer_end.clone())],
                            false,
                            false,
                        );
                        meta_bin
                    })
                    .unwrap_or_default()
            })
            .unwrap_or_default();
        Ok(Response::builder().body(Body::from(meta_bin)).unwrap())
    }

    async fn dump_kvengine_stats(
        req: Request<Body>,
        engine: kvengine::Engine,
    ) -> hyper::Result<Response<Body>> {
        let path = req.uri().path();
        let last = get_last_path_segment(path);
        let res;
        if let Ok(id) = u64::from_str(last) {
            // get all shard state in given keyspace id.
            if path.starts_with("/kvengine/keyspace") {
                let range = ApiV2::get_txn_keyspace_range(id as u32);
                match Self::get_covered_shards_by_range(Some(range), &engine) {
                    Ok(shards) => {
                        let states: Vec<ShardStats> =
                            shards.iter().map(|s| s.get_stats()).collect();
                        res = serde_json::to_string_pretty(&states);
                    }
                    Err(e) => {
                        return Ok(make_response(
                            StatusCode::INTERNAL_SERVER_ERROR,
                            e.to_string(),
                        ));
                    }
                }
            } else {
                let shard_stats = engine.get_shard_stat(id);
                res = serde_json::to_string_pretty(&shard_stats);
            }
        } else if path.starts_with("/kvengine/all") {
            let all_shard_stats = engine.get_all_shard_stats();
            res = serde_json::to_string_pretty(&all_shard_stats);
        } else if path.starts_with("/kvengine/active_lite") {
            // get all active shard stats lite.
            let all_active_lite = engine.get_all_active_shard_stats_lite();
            res = serde_json::to_string_pretty(&all_active_lite);
        } else if path.starts_with("/kvengine/files") {
            let mut all_shard_files: Vec<(u64, Vec<u64>)> = engine
                .get_all_shard_id_vers()
                .iter()
                .filter_map(|id_ver| {
                    engine
                        .get_shard(id_ver.id)
                        .map(|shard| (shard.id, shard.get_all_files()))
                })
                .collect();
            // add blacklist files to the collection of files in use,
            // for simplicity, zero shard id represents the blacklist files
            let blacklist_file_ids = engine
                .get_files_in_blacklist()
                .iter()
                .copied()
                .collect::<Vec<_>>();
            all_shard_files.push((0, blacklist_file_ids));
            res = serde_json::to_string(&all_shard_files);
        } else if path.starts_with("/kvengine/compactor") {
            let remote_urls = engine.comp_client.get_remote_compactors();
            res = serde_json::to_string_pretty(&remote_urls);
        } else {
            let all_shard_stats = engine.get_all_shard_stats();
            let engine_stats = kvengine::Engine::get_engine_stats(all_shard_stats);
            res = serde_json::to_string_pretty(&engine_stats);
        }
        Ok(match res {
            Ok(json) => Response::builder()
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(json))
                .unwrap(),
            Err(_) => make_response(StatusCode::INTERNAL_SERVER_ERROR, "Internal Server Error"),
        })
    }

    async fn add_remote_compactor(
        req: Request<Body>,
        mut comp_client: kvengine::CompactionClient,
    ) -> hyper::Result<Response<Body>> {
        let mut body = Vec::new();
        req.into_body()
            .try_for_each(|bytes| {
                body.extend(bytes);
                ok(())
            })
            .await?;
        Ok(match String::from_utf8(body) {
            Ok(remote_url) => {
                comp_client.add_remote_compactor(remote_url);
                let mut resp = Response::default();
                *resp.status_mut() = StatusCode::OK;
                resp
            }
            Err(e) => make_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("failed to decode, error: {:?}", e),
            ),
        })
    }

    async fn get_change_set_request(
        raw_req: Request<Body>,
    ) -> hyper::Result<kvenginepb::ChangeSet> {
        let mut body = Vec::new();
        raw_req
            .into_body()
            .try_for_each(|bytes| {
                body.extend(bytes);
                ok(())
            })
            .await?;
        let mut cs = kvenginepb::ChangeSet::default();
        cs.merge_from_bytes(&body).unwrap();
        Ok(cs)
    }

    async fn ingest_files(req: Request<Body>, router: RaftRouter) -> hyper::Result<Response<Body>> {
        let cs = Self::get_change_set_request(req).await?;
        let shard_id = cs.get_shard_id();
        info!("[{}] receive ingest_files request: {:?}", shard_id, cs);

        let (cb, fut) = paired_future_callback();
        let callback = Callback::write(Box::new(move |res| {
            cb(res);
        }));
        router.send_casual_msg(
            cs.get_shard_id(),
            CasualMessage::IngestFiles { cs, callback },
        );

        let res = match fut.await {
            Ok(res) => res,
            Err(e) => {
                let err_msg = format!("{} ingest_files channel error: {:?}", shard_id, e);
                error!("{}", err_msg);
                // Return "not leader" to let callers retry.
                let mut errpb = kvproto::errorpb::Error::default();
                errpb.set_not_leader(Default::default());
                errpb.set_message(err_msg);
                return Ok(make_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    errpb.write_to_bytes().unwrap(),
                ));
            }
        };
        if res.response.get_header().has_error() {
            error!(
                "{} ingest_files error: {:?}",
                shard_id,
                res.response.get_header().get_error()
            );
            let err_data = res
                .response
                .get_header()
                .get_error()
                .write_to_bytes()
                .unwrap();
            Ok(make_response(StatusCode::INTERNAL_SERVER_ERROR, err_data))
        } else {
            Ok(make_response(StatusCode::OK, ""))
        }
    }

    async fn dump_rfengine_stats(
        req: Request<Body>,
        engine: rfengine::RfEngine,
    ) -> hyper::Result<Response<Body>> {
        let path = req.uri().path();
        let last = get_last_path_segment(path);
        let res = if let Ok(peer_id) = u64::from_str(last) {
            let peer_stats = engine.get_peer_stats(peer_id);
            serde_json::to_string_pretty(&peer_stats)
        } else {
            let engine_stats = engine.get_engine_stats();
            serde_json::to_string_pretty(&engine_stats)
        };
        Ok(match res {
            Ok(json) => Response::builder()
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(json))
                .unwrap(),
            Err(_) => make_response(StatusCode::INTERNAL_SERVER_ERROR, "Internal Server Error"),
        })
    }

    // URI: /rfengine/wal_chunk?epoch_id=xxx&start_off=xxx&end_off=xxx
    async fn rfengine_wal_chunk(
        req: Request<Body>,
        engine: rfengine::RfEngine,
    ) -> hyper::Result<Response<Body>> {
        let bad_request_resp = |msg: &str| make_response(StatusCode::BAD_REQUEST, msg.to_owned());
        if !engine.is_lightweight_backup_enabled() {
            return Ok(bad_request_resp("lightweight backup not enabled"));
        }

        let query = req.uri().query().unwrap_or("");
        let query_pairs: HashMap<_, _> = url::form_urlencoded::parse(query.as_bytes()).collect();
        info!("/rfengine/wal_chunk {:?}", query_pairs);
        if !query_pairs.contains_key("epoch_id")
            || !query_pairs.contains_key("start_off")
            || !query_pairs.contains_key("end_off")
        {
            return Ok(bad_request_resp("query parameters invalid"));
        }
        let (callback, future) = paired_future_callback();
        let epoch_id = match u32::from_str(query_pairs.get("epoch_id").unwrap()) {
            Ok(epoch_id) => epoch_id,
            Err(err) => return Ok(bad_request_resp(err.to_string().as_str())),
        };
        let start_off = match u64::from_str(query_pairs.get("start_off").unwrap()) {
            Ok(start_off) => start_off,
            Err(err) => return Ok(bad_request_resp(err.to_string().as_str())),
        };
        let end_off = match u64::from_str(query_pairs.get("end_off").unwrap()) {
            Ok(end_off) => end_off,
            Err(err) => return Ok(bad_request_resp(err.to_string().as_str())),
        };
        engine.dump_wal_chunk(epoch_id, start_off, end_off, callback);

        Ok(match future.await {
            Ok(resp) => match resp {
                Ok(chunk) => Response::builder().body(Body::from(chunk)).unwrap(),
                Err(err) => {
                    error!("rfengine_wal_chunk error: {}", err);
                    make_response(StatusCode::BAD_REQUEST, format!("bad request {}", err))
                }
            },
            Err(e) => make_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Internal Server Error {}", e),
            ),
        })
    }

    // URI: /unsafe_recover/clear?cluster_id=xxx&[keyspace_id=xxx[&table_id=xxx]][&
    // region_id=xxx]
    async fn unsafe_recover(
        req: Request<Body>,
        rf: rfengine::RfEngine,
        engine: kvengine::Engine,
    ) -> hyper::Result<Response<Body>> {
        let bad_request_resp =
            |msg: &str| make_response(StatusCode::BAD_REQUEST, msg.to_owned() + "\n");
        let path = req.uri().path();
        if path != "/unsafe_recover/clear" {
            return Ok(bad_request_resp("bad request URI"));
        }
        let query = req.uri().query().unwrap_or("");
        let query_pairs: HashMap<_, _> = url::form_urlencoded::parse(query.as_bytes()).collect();
        let cluster_id = query_pairs.get("cluster_id");
        if cluster_id.is_none() {
            return Ok(bad_request_resp("cluster_id not found"));
        }
        let cluster_id = match u64::from_str(cluster_id.unwrap()) {
            Ok(cluster_id) => cluster_id,
            Err(e) => return Ok(bad_request_resp(e.to_string().as_str())),
        };
        let store_ident = load_store_ident(&rf).unwrap_or_default();
        if cluster_id != store_ident.cluster_id {
            return Ok(bad_request_resp("cluster_id not match"));
        }
        let region_id = query_pairs.get("region_id");
        let keyspace_id = query_pairs.get("keyspace_id");
        let table_id = query_pairs.get("table_id");

        let target_regions = if let Some(region_id) = region_id {
            let region_id = match u64::from_str(region_id) {
                Ok(region_id) => region_id,
                Err(err) => return Ok(bad_request_resp(err.to_string().as_str())),
            };
            match rf.get_region_peer_map().get(&region_id) {
                Some(&peer_id) => {
                    let cs = load_raft_engine_meta(&rf, peer_id);
                    if cs.is_none() {
                        return Ok(bad_request_resp(
                            format!("region {} peer {} not exists", region_id, peer_id).as_str(),
                        ));
                    }
                    vec![(peer_id, region_id, cs.unwrap().shard_ver)]
                }
                None => {
                    return Ok(bad_request_resp(
                        format!("region {} not exists", region_id).as_str(),
                    ));
                }
            }
        } else if let Some(keyspace_id) = keyspace_id {
            let keyspace_id = match u32::from_str(keyspace_id) {
                Ok(keyspace_id) => keyspace_id,
                Err(e) => return Ok(bad_request_resp(e.to_string().as_str())),
            };
            let mut prefix = ApiV2::get_txn_keyspace_prefix(keyspace_id);
            if let Some(table_id) = table_id {
                let table_id = match u64::from_str(table_id) {
                    Ok(table_id) => table_id,
                    Err(e) => return Ok(bad_request_resp(e.to_string().as_str())),
                };
                prefix.put_u8(b't');
                prefix.encode_i64(table_id as i64).unwrap();
            }
            if let Some(regions) = collect_prefix_regions(&rf, &prefix) {
                regions
            } else {
                return Ok(bad_request_resp("collect none region with prefix"));
            }
        } else {
            return Ok(bad_request_resp("query parameters invalid"));
        };

        let target_regions_len = target_regions.len();
        let mut wb = WriteBatch::new();
        for (peer_id, region_id, region_version) in target_regions {
            let shard_stats = engine.get_shard_stat(region_id);
            info!(
                "unsafe recover region {} peer {} shard_stats {:?}",
                region_id, peer_id, shard_stats
            );
            if shard_stats.id != 0 {
                return Ok(bad_request_resp(
                    format!("region {} not in blacklist", region_id).as_str(),
                ));
            }

            let old_raft_state = match Self::load_raft_state(&rf, peer_id, region_version) {
                Some(state) => state,
                None => {
                    return Ok(bad_request_resp(
                        format!(
                            "region {} peer {} raft state not exists",
                            region_id, peer_id
                        )
                        .as_str(),
                    ));
                }
            };
            let raft_state_last_index = old_raft_state.get_last_index();
            let raft_log_last_index = rf.get_last_index(peer_id).unwrap_or(RAFT_INIT_LOG_INDEX);
            let last_index = std::cmp::max(raft_state_last_index, raft_log_last_index);
            info!(
                "unsafe_recover region: {} peer: {} truncate raft log to {}",
                region_id, peer_id, last_index
            );
            wb.truncate_raft_log(peer_id, region_id, last_index);

            let raft_log_term = rf
                .get_term(peer_id, last_index)
                .unwrap_or(RAFT_INIT_LOG_TERM);
            let old_kv_engine_meta = match rf.get_state(peer_id, KV_ENGINE_META_KEY) {
                Some(meta) => meta,
                None => {
                    return Ok(bad_request_resp(
                        format!("peer {} kv engine meta not exists", peer_id).as_str(),
                    ));
                }
            };
            let mut old_cs = kvenginepb::ChangeSet::new();
            old_cs.merge_from_bytes(&old_kv_engine_meta).unwrap();
            let snap = old_cs.get_snapshot();
            let inner_key_off = snap.get_inner_key_off();
            let mut properties = kvengine::Properties::new();
            properties = properties.apply_pb(snap.get_properties());
            let meta_term = properties.get(TERM_KEY).unwrap().get_u64_le();

            // Add a delta to term 3 make sure it's greater than term in pd cache.
            let term = std::cmp::max(raft_log_term, meta_term) + 3;

            let _ = match Self::write_empty_engine_meta(
                &rf,
                &mut wb,
                peer_id,
                region_id,
                region_version,
                last_index,
                term,
                inner_key_off,
            ) {
                Ok(cs) => {
                    info!(
                        "unsafe_recover region: {} peer: {} reset kv engine meta {:?} -> {:?}",
                        region_id, peer_id, old_cs, cs
                    );
                    cs
                }
                Err(e) => return Ok(bad_request_resp(e.to_string().as_str())),
            };

            Self::write_truncated_state(&mut wb, peer_id, region_id, last_index, term);
            info!(
                "unsafe_recover region: {} peer: {} reset truncated state to truncated_index: {} truncated_term: {}",
                region_id, peer_id, last_index, term
            );

            let raft_state = Self::write_raft_state(
                &mut wb,
                peer_id,
                region_id,
                region_version,
                last_index,
                term,
            );
            info!(
                "unsafe_recover region: {} peer: {} reset raft state {:?} -> {:?}",
                region_id, peer_id, old_raft_state, raft_state
            );
        }

        // Write batch to raft engine only when it's not empty. Or it will cause the wal
        // iteration interrupted.
        if !wb.is_empty() {
            rf.write(wb).unwrap();
        }

        Ok(make_response(
            StatusCode::OK,
            format!("{} region(s) clear success\n", target_regions_len),
        ))
    }

    // URI: /major_compact?major_compact=xxx[&keyspace_id=xxx[&table_id=xxx]][&
    // region_id=xxx]
    // Note: return 404 when no match region is found.
    async fn major_compact(
        req: Request<Body>,
        rf: RfEngine,
        router: RaftRouter,
    ) -> hyper::Result<Response<Body>> {
        info!("major compact request: {:?}", req);
        let bad_request_resp = |msg: &str| make_response(StatusCode::BAD_REQUEST, msg.to_owned());
        let not_found_resp = |msg: &str| make_response(StatusCode::NOT_FOUND, msg.to_owned());

        let path = req.uri().path();
        if path != "/major-compact" {
            return Ok(bad_request_resp("bad request URI"));
        }
        let query = req.uri().query().unwrap_or("");
        let query_pairs: HashMap<_, _> = url::form_urlencoded::parse(query.as_bytes()).collect();
        let major_compact = query_pairs.get("major_compact");
        if major_compact.is_none() {
            return Ok(bad_request_resp("major_compact not found"));
        }
        let major_compact = match bool::from_str(major_compact.unwrap()) {
            Ok(major_compact) => major_compact,
            Err(e) => return Ok(bad_request_resp(e.to_string().as_str())),
        };
        let region_id = query_pairs.get("region_id");
        let keyspace_id = query_pairs.get("keyspace_id");
        let table_id = query_pairs.get("table_id");

        let target_regions = if let Some(region_id) = region_id {
            let region_id = match u64::from_str(region_id) {
                Ok(region_id) => region_id,
                Err(err) => return Ok(bad_request_resp(err.to_string().as_str())),
            };
            let region_to_peers = rf.get_region_peer_map();
            match region_to_peers.get(&region_id) {
                Some(&peer_id) => {
                    let cs = load_raft_engine_meta(&rf, peer_id);
                    if cs.is_none() {
                        return Ok(not_found_resp(
                            format!("region {} peer {} not exists", region_id, peer_id).as_str(),
                        ));
                    }
                    vec![region_id]
                }
                None => {
                    return Ok(not_found_resp(
                        format!("region {} not exists", region_id).as_str(),
                    ));
                }
            }
        } else if let Some(keyspace_id) = keyspace_id {
            let keyspace_id = match u32::from_str(keyspace_id) {
                Ok(keyspace_id) => keyspace_id,
                Err(e) => return Ok(bad_request_resp(e.to_string().as_str())),
            };
            let mut prefix = ApiV2::get_txn_keyspace_prefix(keyspace_id);
            if let Some(table_id) = table_id {
                let table_id = match u64::from_str(table_id) {
                    Ok(table_id) => table_id,
                    Err(e) => return Ok(bad_request_resp(e.to_string().as_str())),
                };
                prefix.put_u8(b't');
                prefix.encode_i64(table_id as i64).unwrap();
            }
            if let Some(regions) = collect_prefix_regions(&rf, &prefix) {
                regions
                    .into_iter()
                    .map(|(_, region_id, _)| region_id)
                    .collect::<Vec<_>>()
            } else {
                return Ok(not_found_resp("collect none region with prefix"));
            }
        } else {
            return Ok(bad_request_resp("query parameters invalid"));
        };
        if target_regions.is_empty() {
            return Ok(not_found_resp("target regions is empty"));
        } else {
            info!("manual major compact regions {:?}", target_regions);
        }
        let mut region_futures = Vec::with_capacity(target_regions.len());
        for region_id in target_regions.iter() {
            let (cb, fu) = paired_future_callback();
            let callback = Callback::write(Box::new(move |_| {
                cb(());
            }));
            region_futures.push(fu);
            router.send_casual_msg(
                *region_id,
                CasualMessage::MajorCompact {
                    major_compact,
                    callback,
                },
            );
        }
        let _ = futures::future::join_all(region_futures).await;
        Ok(make_response(
            StatusCode::OK,
            format!(
                "trigger manual major compaction on region(s) {:?} success",
                target_regions
            ),
        ))
    }

    fn get_dfs_file_id(req: &Request<Body>) -> Option<u64> {
        let path = req.uri().path();
        let last = get_last_path_segment(path);
        u64::from_str(last).ok()
    }

    fn get_dfs_file_type(req: &Request<Body>) -> FileType {
        if let Some(query) = req.uri().query() {
            let query_pairs: HashMap<_, _> =
                url::form_urlencoded::parse(query.as_bytes()).collect();
            if let Some(file_type_str) = query_pairs.get("file_type") {
                return file_type_str.as_ref().try_into().unwrap_or(FileType::Sst);
            }
        }
        FileType::Sst
    }

    async fn handle_dfs_file_read(
        req: Request<Body>,
        engine: kvengine::Engine,
    ) -> hyper::Result<Response<Body>> {
        let id_opt = Self::get_dfs_file_id(&req);
        if id_opt.is_none() {
            return Ok(make_response(StatusCode::BAD_REQUEST, "invalid file id"));
        }
        let id = id_opt.unwrap();
        let file_type = Self::get_dfs_file_type(&req);
        let (callback, future) = paired_future_callback();
        std::thread::spawn(move || {
            let res = match file_type {
                FileType::TxnChunk => engine.get_txn_chunk_manager().read_local_chunk(id),
                _ => engine.read_local_file(id, file_type),
            };
            callback(res);
        });
        let res = future.await.unwrap();
        Ok(match res {
            Ok(data) => Response::builder()
                .header(header::CONTENT_TYPE, "application/octet-stream")
                .body(Body::from(data))
                .unwrap(),
            Err(err) => make_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Internal Server Error {}", err),
            ),
        })
    }

    async fn handle_dfs_file_create(
        req: Request<Body>,
        engine: kvengine::Engine,
    ) -> hyper::Result<Response<Body>> {
        let id_opt = Self::get_dfs_file_id(&req);
        if id_opt.is_none() {
            return Ok(make_response(StatusCode::BAD_REQUEST, "invalid file id"));
        }
        let id = id_opt.unwrap();
        let file_type = Self::get_dfs_file_type(&req);
        let data = hyper::body::to_bytes(req.into_body()).await?;
        let (callback, future) = paired_future_callback();
        std::thread::spawn(move || {
            let res = match file_type {
                FileType::TxnChunk => engine.get_txn_chunk_manager().write_local_chunk(id, data),
                _ => engine.write_local_file_if_not_exists(id, data, file_type),
            };
            callback(res);
        });
        let res = future.await.unwrap();
        Ok(match res {
            Ok(_) => make_response(StatusCode::OK, ""),
            Err(err) => make_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Internal Server Error {}", err),
            ),
        })
    }

    async fn backup_rfengine(
        req: Request<Body>,
        engine: rfengine::RfEngine,
        dfs_conf: DFSConfig,
    ) -> hyper::Result<Response<Body>> {
        let body = hyper::body::to_bytes(req.into_body()).await?;
        let backup_config: serde_json::Result<rfengine::BackupConfig> =
            serde_json::from_slice(&body);
        if backup_config.is_err() {
            return Ok(make_response(StatusCode::BAD_REQUEST, "Bad request body"));
        }
        let backup_config = backup_config.unwrap();

        // Check if lightweight backup enabled.
        if backup_config.lightweight && !engine.is_lightweight_backup_enabled() {
            return Ok(make_response(
                StatusCode::BAD_REQUEST,
                "lightweight backup not enabled",
            ));
        }

        let cluster_id = backup_config.cluster_id;
        let mut store_ident = StoreIdent::default();
        let data = engine
            .get_state(0, rfengine::STORE_IDENT_KEY)
            .unwrap_or_default();
        store_ident.merge_from_bytes(data.chunk()).unwrap();
        if store_ident.cluster_id != cluster_id {
            warn!(
                "{}: backup id mismatch expect {:?}, got {}",
                store_ident.store_id, store_ident.cluster_id, cluster_id
            );
            return Ok(make_response(
                StatusCode::BAD_REQUEST,
                "cluster id mismatch",
            ));
        }

        let s3fs = kvengine::dfs::S3Fs::new(
            dfs_conf.prefix,
            dfs_conf.s3_endpoint,
            dfs_conf.s3_key_id,
            dfs_conf.s3_secret_key,
            dfs_conf.s3_region,
            dfs_conf.s3_bucket,
        );
        let (callback, future) = paired_future_callback();
        let task = rfengine::BackupTask::new(Box::new(s3fs), callback, backup_config);
        engine.backup(task);
        Ok(match future.await {
            Ok(resp) => match resp {
                Ok(meta) => {
                    info!("{}: backup finished", meta.store_id);
                    Response::builder()
                        .body(Body::from(meta.write_to_bytes().unwrap()))
                        .unwrap()
                }
                Err(err) => {
                    error!("{}: backup failed {:?}", store_ident.store_id, &err);
                    make_response(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        format!("Internal Server Error {}", err),
                    )
                }
            },
            Err(e) => make_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Internal Server Error {}", e),
            ),
        })
    }

    fn check_truncate_ts_req(
        rfengine: &rfengine::RfEngine,
        config: &TruncateTsConfig,
    ) -> Result<()> {
        let mut store_ident = StoreIdent::default();
        let data = rfengine
            .get_state(0, rfengine::STORE_IDENT_KEY)
            .unwrap_or_default();
        store_ident.merge_from_bytes(data.chunk()).unwrap();
        if store_ident.cluster_id != config.cluster_id {
            return Err(box_err!(
                "Cluster id mismatch, got {:?}, expect {:?}",
                config.cluster_id,
                store_ident.cluster_id
            ));
        }
        if config.ts == 0 {
            return Err(box_err!("Invalid truncate ts {:?}", config.ts));
        }
        if config.range.is_some() {
            let range = config.range.as_ref().unwrap();
            if range.0.is_empty() || range.1.is_empty() {
                return Err(box_err!("Unsupported range {:?}", range));
            }
        }
        Ok(())
    }

    fn get_covered_shards_by_range(
        range: Option<(Vec<u8>, Vec<u8>)>,
        engine: &kvengine::Engine,
    ) -> Result<Vec<Arc<Shard>>> {
        let shards = engine.get_all_shard_id_vers();
        let mut partial_covered_shard = None;
        let ret: Vec<Arc<Shard>> = shards
            .into_iter()
            .filter_map(|s| engine.get_shard(s.id))
            .filter(|s| {
                if range.is_none() {
                    return true;
                }
                let range = range.as_ref().unwrap();
                let full_cover = s.outer_start >= range.0 && s.outer_end <= range.1;
                if !full_cover && s.outer_start < range.1 && s.outer_end > range.0 {
                    partial_covered_shard = Some(s.clone());
                }
                full_cover
            })
            .collect();
        if let Some(s) = partial_covered_shard {
            return Err(box_err!(
                "Shard {} [{:?}-{:?}) is partial covered by range {:?}",
                s.tag(),
                s.outer_start,
                s.outer_end,
                range
            ));
        }
        Ok(ret)
    }

    async fn truncate_ts(
        req: Request<Body>,
        rfengine: rfengine::RfEngine,
        engine: kvengine::Engine,
        router: RaftRouter,
    ) -> hyper::Result<Response<Body>> {
        let body = hyper::body::to_bytes(req.into_body()).await?;
        let config: serde_json::Result<TruncateTsConfig> = serde_json::from_slice(&body);
        if config.is_err() {
            return Ok(make_response(StatusCode::BAD_REQUEST, "Bad request body"));
        }
        let config = config.unwrap();
        let cluster_id = config.cluster_id;
        let truncate_ts = config.ts;
        if let Err(e) = Self::check_truncate_ts_req(&rfengine, &config) {
            error!("Invalid cluster id or truncate ts, {:?}", e);
            return Ok(make_response(StatusCode::BAD_REQUEST, e.to_string()));
        }
        let all_shards = Self::get_covered_shards_by_range(config.range, &engine);
        if all_shards.is_err() {
            let msg = all_shards.err().unwrap().to_string();
            error!("Some shards are partial covered {}", msg);
            return Ok(make_response(StatusCode::BAD_REQUEST, msg));
        }
        let all_shards = all_shards.unwrap();
        let mut region_futures = vec![];
        let mut shards_stat = vec![];
        info!(
            "Begin truncate to ts {:?} for cluster {:?}, shard cnt {}",
            truncate_ts,
            cluster_id,
            all_shards.len()
        );
        for shard in all_shards {
            let max_ts = shard.get_max_ts();
            // if shard max_ts is smaller than truncate_ts, no need to send request.
            if max_ts <= truncate_ts {
                continue;
            }
            let (cb, fu) = paired_future_callback();
            let callback = Callback::write(Box::new(move |_| {
                cb(());
            }));
            region_futures.push(fu);
            router.send_casual_msg(
                shard.id,
                CasualMessage::TruncateTs {
                    ts: truncate_ts,
                    shard_ver: shard.ver,
                    callback,
                },
            );
            shards_stat.push(kvengine::ShardTruncateTsStats {
                id: shard.id,
                ver: shard.ver,
                cur_max_ts: max_ts,
                truncate_ts,
            });
        }
        let _ = futures::future::join_all(region_futures).await;
        let encode_res = serde_json::to_string_pretty(&shards_stat);
        Ok(match encode_res {
            Ok(json) => Response::builder()
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(json))
                .unwrap(),
            Err(_) => make_response(StatusCode::INTERNAL_SERVER_ERROR, "Internal Server Error"),
        })
    }

    async fn get_restore_shard_request(
        raw_req: Request<Body>,
    ) -> hyper::Result<RestoreShardRequest> {
        let mut body = Vec::new();
        raw_req
            .into_body()
            .try_for_each(|bytes| {
                body.extend(bytes);
                ok(())
            })
            .await?;
        let mut cs = kvenginepb::ChangeSet::default();
        cs.merge_from_bytes(&body).unwrap();
        Ok(RestoreShardRequest { cs })
    }

    async fn restore_shard(
        req: Request<Body>,
        router: RaftRouter,
        engine: kvengine::Engine,
    ) -> hyper::Result<Response<Body>> {
        let req = Self::get_restore_shard_request(req).await?;
        let shard_id = req.cs.get_shard_id();
        debug!("[{}] receive restore_shard request: {:?}", shard_id, req);

        let (cb, fut) = paired_future_callback();
        let callback = Callback::write(Box::new(move |res| {
            cb(res);
        }));
        router.send_casual_msg(
            req.cs.get_shard_id(),
            CasualMessage::RestoreShard {
                cs: req.cs,
                callback,
            },
        );

        let res = match fut.await {
            Ok(res) => res,
            Err(e) => {
                let err_msg = format!("{} restore_shard channel error: {:?}", shard_id, e);
                error!("{}", err_msg);
                return Ok(make_response(StatusCode::INTERNAL_SERVER_ERROR, err_msg));
            }
        };
        if res.response.get_header().has_error() {
            error!(
                "{} restore_shard error: {:?}",
                shard_id,
                res.response.get_header().get_error()
            );
            Ok(make_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                res.response
                    .get_header()
                    .get_error()
                    .get_message()
                    .to_string(),
            ))
        } else {
            let stat = engine.get_shard_stat(shard_id);
            let resp = RestoreShardResponse {
                shard_id,
                restore_bytes: stat.kv_size,
            };
            let json = serde_json::to_string_pretty(&resp).unwrap();
            Ok(Response::builder()
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(json))
                .unwrap())
        }
    }

    async fn handle_schema_file(
        req: Request<Body>,
        router: RaftRouter,
        engine: kvengine::Engine,
    ) -> hyper::Result<Response<Body>> {
        let bad_request_resp = |msg: &str| make_response(StatusCode::BAD_REQUEST, msg.to_owned());
        let query = req.uri().query().unwrap_or("");
        let query_pairs: HashMap<_, _> = url::form_urlencoded::parse(query.as_bytes()).collect();
        let keyspace_id = match get_uint_param(&query_pairs, "keyspace_id") {
            Some(keyspace_id) => keyspace_id,
            None => {
                return Ok(bad_request_resp("keyspace_id not found"));
            }
        };
        let file_id = match get_uint_param(&query_pairs, "file_id") {
            Some(file_id) => file_id,
            None => {
                return Ok(bad_request_resp("file_id not found"));
            }
        };
        let schema_file = match engine.load_schema_file(file_id).await {
            Ok(file) => file,
            Err(e) => {
                let msg = format!("failed to load schema file {:?}", e);
                return Ok(make_response(StatusCode::INTERNAL_SERVER_ERROR, msg));
            }
        };
        let (start, end) = ApiV2::get_txn_keyspace_range(keyspace_id as u32);
        let (callback, future) = paired_future_callback();
        router.send_store_msg(StoreMsg::GetRegionsInRange {
            start,
            end,
            callback,
        });
        let res = future.await;
        if res.is_err() {
            return Ok(make_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                "failed to get regions in range",
            ));
        }
        let region_id_vers = res.unwrap();
        for region_id_ver in region_id_vers {
            router.send_casual_msg(
                region_id_ver.id(),
                CasualMessage::UpdateSchemaFile(schema_file.clone()),
            );
        }
        Ok(Response::new(Body::empty()))
    }

    pub fn stop(self) {
        let _ = self.tx.send(());
        self.thread_pool.shutdown_timeout(Duration::from_secs(3));
    }

    // Return listening address, this may only be used for outer test
    // to get the real address because we may use "127.0.0.1:0"
    // in test to avoid port conflict.
    #[allow(unused)]
    pub fn listening_addr(&self) -> SocketAddr {
        self.addr.unwrap()
    }
}

fn get_last_path_segment(path: &str) -> &str {
    let mut last = "";
    for i in (0..path.len()).rev() {
        if path.as_bytes()[i] == b'/' {
            last = &path[i + 1..];
            break;
        }
    }
    last
}

fn get_uint_param(query_pairs: &HashMap<Cow<'_, str>, Cow<'_, str>>, key: &str) -> Option<u64> {
    query_pairs.get(key).and_then(|v| u64::from_str(v).ok())
}

impl StatusServer {
    pub async fn dump_region_meta(
        _req: Request<Body>,
        _router: RaftRouter,
    ) -> hyper::Result<Response<Body>> {
        // TODO(x)
        Ok(hyper::Response::new(Body::empty()))
    }

    pub async fn handle_sync_region(
        req: Request<Body>,
        router: RaftRouter,
    ) -> hyper::Result<Response<Body>> {
        let mut body = Vec::new();
        req.into_body()
            .try_for_each(|bytes| {
                body.extend(bytes);
                ok(())
            })
            .await?;

        let SyncRegionRequest {
            start,
            end,
            limit,
            reverse,
            debug,
        } = match serde_json::from_slice::<SyncRegionRequest>(&body) {
            Ok(req) => req,
            Err(e) => {
                return Ok(make_response(
                    StatusCode::BAD_REQUEST,
                    format!("invalid request body: {:?}", e),
                ));
            }
        };

        let (start, end) = match (hex::decode(&start), hex::decode(&end)) {
            (Ok(start), Ok(end)) if end.is_empty() || start <= end => (start, end),
            _ => {
                return Ok(make_response(
                    StatusCode::BAD_REQUEST,
                    format!("invalid range: [{}, {})", start, end),
                ));
            }
        };

        let (callback, future) = paired_future_callback();
        let store_msg = StoreMsg::SyncRegion {
            start,
            end,
            limit,
            reverse,
            callback,
        };
        router.send_store_msg(store_msg);
        match future.await {
            Ok(resp) => {
                let body = if debug {
                    format!("{:?}", resp).into_bytes()
                } else {
                    resp.write_to_bytes().unwrap()
                };
                Ok(Response::new(body.into()))
            }
            Err(e) => Ok(make_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Internal Server Error {}", e),
            )),
        }
    }

    pub async fn handle_sync_region_by_id(
        req: Request<Body>,
        router: RaftRouter,
    ) -> hyper::Result<Response<Body>> {
        let mut body = Vec::new();
        req.into_body()
            .try_for_each(|bytes| {
                body.extend(bytes);
                ok(())
            })
            .await?;

        let SyncRegionByIdRequest { region_id, debug } = match serde_json::from_slice(&body) {
            Ok(req) => req,
            Err(e) => {
                return Ok(make_response(
                    StatusCode::BAD_REQUEST,
                    format!("invalid request body: {:?}", e),
                ));
            }
        };

        let (callback, future) = paired_future_callback();
        let store_msg = StoreMsg::SyncRegionById {
            region_id,
            callback,
        };
        router.send_store_msg(store_msg);
        match future.await {
            Ok(resp) => {
                let body = if debug {
                    format!("{:?}", resp).into_bytes()
                } else {
                    resp.write_to_bytes().unwrap()
                };
                Ok(Response::new(body.into()))
            }
            Err(e) => Ok(make_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Internal Server Error {}", e),
            )),
        }
    }

    fn handle_get_metrics(
        req: Request<Body>,
        mgr: &ConfigController,
    ) -> hyper::Result<Response<Body>> {
        let should_simplify = mgr.get_current().server.simplify_metrics;
        let gz_encoding = client_accept_gzip(&req);
        let metrics = if gz_encoding {
            // gzip can reduce the body size to less than 1/10.
            let mut encoder = GzEncoder::new(vec![], Compression::default());
            dump_to(&mut encoder, should_simplify);
            encoder.finish().unwrap()
        } else {
            dump(should_simplify).into_bytes()
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

    fn start_serve<I, C>(&mut self, builder: HyperBuilder<I>)
    where
        I: Accept<Conn = C, Error = std::io::Error> + Send + 'static,
        I::Error: Into<Box<dyn StdError + Send + Sync>>,
        I::Conn: AsyncRead + AsyncWrite + Unpin + Send + 'static,
        C: ServerConnection,
    {
        let security_config = self.security_config.clone();
        let cfg_controller = self.cfg_controller.clone();
        let router = self.router.clone();
        let engine = self.kvengine.clone();
        let rfengine = self.rfengine.clone();
        // Start to serve.
        let server = builder.serve(make_service_fn(move |conn: &C| {
            let x509 = conn.get_x509();
            let security_config = security_config.clone();
            let cfg_controller = cfg_controller.clone();
            let router = router.clone();
            let engine = engine.clone();
            let rfengine = rfengine.clone();
            async move {
                // Create a status service.
                Ok::<_, hyper::Error>(service_fn(move |req: Request<Body>| {
                    let start = Instant::now();
                    let x509 = x509.clone();
                    let security_config = security_config.clone();
                    let cfg_controller = cfg_controller.clone();
                    let router = router.clone();
                    let engine = engine.clone();
                    let rfengine = rfengine.clone();
                    async move {
                        let path = req.uri().path().to_owned();
                        let method = req.method().to_owned();

                        #[cfg(feature = "failpoints")]
                        {
                            if path.starts_with(FAIL_POINTS_REQUEST_PATH) {
                                return handle_fail_points_request(req).await;
                            }
                        }

                        // 1. POST "/config" will modify the configuration of TiKV.
                        // 2. GET "/region" will get start key and end key. These keys could be
                        // actual user data since in some cases the data
                        // itself is stored in the key.
                        let should_check_cert = !matches!(
                            (&method, path.as_ref()),
                            (&Method::GET, "/metrics")
                                | (&Method::GET, "/status")
                                | (&Method::GET, "/config")
                                | (&Method::GET, "/debug/pprof/profile")
                        );

                        if should_check_cert && !check_cert(security_config, x509) {
                            return Ok(make_response(
                                StatusCode::FORBIDDEN,
                                "certificate role error",
                            ));
                        }

                        match (method, path.as_ref()) {
                            (Method::GET, "/metrics") => {
                                Self::handle_get_metrics(req, &cfg_controller)
                            }
                            (Method::GET, "/status") => Ok(Response::default()),
                            (Method::GET, "/debug/pprof/heap_list") => {
                                Ok(make_response(
                                    StatusCode::GONE,
                                    "Deprecated, heap profiling is always enabled by default, just use /debug/pprof/heap to get the heap profile when needed",
                                ))
                            }
                            (Method::GET, "/debug/pprof/heap_activate") => {
                                Ok(make_response(
                                    StatusCode::GONE,
                                    "Deprecated, use config `memory.enable_heap_profiling` to toggle",
                                ))
                            }
                            (Method::GET, "/debug/pprof/heap_deactivate") => {
                                Ok(make_response(
                                    StatusCode::GONE,
                                    "Deprecated, use config `memory.enable_heap_profiling` to toggle",
                                ))
                            }
                            (Method::GET, "/debug/pprof/heap") => {
                                Self::dump_heap_prof_to_resp(req)
                            }
                            (Method::GET, "/debug/pprof/cmdline") => Self::get_cmdline(req),
                            (Method::GET, "/debug/pprof/symbol") => {
                                Self::get_symbol_count(req)
                            }
                            (Method::POST, "/debug/pprof/symbol") => Self::get_symbol(req).await,
                            (Method::GET, "/config") => {
                                Self::get_config(req, &cfg_controller).await
                            }
                            (Method::POST, "/config") => {
                                Self::update_config(cfg_controller.clone(), req).await
                            }
                            (Method::GET, "/debug/pprof/profile") => {
                                Self::dump_cpu_prof_to_resp(req).await
                            }
                            (Method::GET, "/debug/fail_point") => {
                                info!("debug fail point API start");
                                fail_point!("debug_fail_point");
                                info!("debug fail point API finish");
                                Ok(Response::default())
                            }
                            (Method::GET, path) if path.starts_with("/region") => {
                                Self::dump_region_meta(req, router).await
                            }
                            (Method::GET, path) if path.starts_with("/sync_region_by_id") => {
                                let resp = Self::handle_sync_region_by_id(req, router).await?;
                                STATUS_REQ_HISTOGRAM_STATIC
                                    .sync_region_by_id
                                    .observe(start.saturating_elapsed().as_secs_f64());
                                Ok(resp)
                            }
                            (Method::GET, path) if path.starts_with("/sync_region") => {
                                let resp = Self::handle_sync_region(req, router).await?;
                                STATUS_REQ_HISTOGRAM_STATIC
                                    .sync_region
                                    .observe(start.saturating_elapsed().as_secs_f64());
                                Ok(resp)
                            }
                            (Method::PUT, path) if path.starts_with("/log-level") => {
                                Self::change_log_level(req).await
                            }
                            (Method::GET, path) if path.starts_with("/kvengine") => {
                                if path.starts_with("/kvengine/snapshot/") {
                                    Self::dump_kvengine_snapshot(req, engine, router).await
                                } else if path.starts_with("/kvengine/meta/") {
                                    Self::dump_kvengine_meta(req, engine).await
                                } else {
                                    Self::dump_kvengine_stats(req, engine).await
                                }
                            }
                            (Method::GET, path) if path.starts_with("/rfengine") => {
                                if path.starts_with("/rfengine/wal_chunk") {
                                    Self::rfengine_wal_chunk(req, rfengine).await
                                } else {
                                    Self::dump_rfengine_stats(req, rfengine).await
                                }
                            }
                            (Method::POST, path) if path.starts_with("/rfengine/backup") => {
                                Self::backup_rfengine(
                                    req,
                                    rfengine,
                                    cfg_controller.get_current().dfs.clone(),
                                )
                                .await
                            }
                            (Method::POST, path) if path.starts_with("/truncate-ts") => {
                                Self::truncate_ts(req, rfengine, engine, router).await
                            }

                            (Method::POST, path) if path.starts_with("/restore-shard") => {
                                Self::restore_shard(req, router, engine).await
                            }
                            (Method::POST, path) if path.starts_with("/kvengine/compactor") => {
                                Self::add_remote_compactor(req, engine.comp_client.clone()).await
                            }
                            (Method::POST, path) if path.starts_with("/ingest_files") => {
                                Self::ingest_files(req, router).await
                            }
                            (Method::POST, path) if path.starts_with("/unsafe_recover") => {
                                Self::unsafe_recover(req, rfengine, engine).await
                            }
                            (Method::POST, path) if path.starts_with("/major-compact") => {
                                Self::major_compact(req, rfengine, router).await
                            }
                            (Method::POST, path) if path.starts_with("/schema_file") => {
                                Self::handle_schema_file(req, router, engine).await
                            }
                            (Method::GET, path) if path.starts_with("/dfs/") => {
                                Self::handle_dfs_file_read(req, engine).await
                            }
                            (Method::POST, path) if path.starts_with("/dfs/") => {
                                Self::handle_dfs_file_create(req, engine).await
                            }
                            _ => Ok(make_response(StatusCode::NOT_FOUND, "path not found")),
                        }
                    }
                }))
            }
        }));

        let rx = self.rx.take().unwrap();
        let graceful = server
            .with_graceful_shutdown(async move {
                let _ = rx.await;
            })
            .map_err(|e| error!("Status server error: {:?}", e));
        self.thread_pool.spawn(graceful);
    }

    pub fn start(&mut self, status_addr: String) -> Result<()> {
        let addr = SocketAddr::from_str(&status_addr)?;

        let incoming = {
            let _enter = self.thread_pool.enter();
            AddrIncoming::bind(&addr)
        }?;
        self.addr = Some(incoming.local_addr());
        if !self.security_config.cert_path.is_empty()
            && !self.security_config.key_path.is_empty()
            && !self.security_config.ca_path.is_empty()
        {
            let mut acceptor = SslAcceptor::mozilla_modern(SslMethod::tls())?;
            acceptor.set_ca_file(&self.security_config.ca_path)?;
            acceptor.set_certificate_chain_file(&self.security_config.cert_path)?;
            acceptor.set_private_key_file(&self.security_config.key_path, SslFiletype::PEM)?;
            if !self.security_config.cert_allowed_cn.is_empty() {
                acceptor.set_verify(SslVerifyMode::PEER | SslVerifyMode::FAIL_IF_NO_PEER_CERT);
            }
            let acceptor = acceptor.build();
            let tls_incoming = tls_incoming(acceptor, incoming);
            let server = Server::builder(tls_incoming);
            self.start_serve(server);
        } else {
            let server = Server::builder(incoming);
            self.start_serve(server);
        }
        Ok(())
    }
}

// To unify TLS/Plain connection usage in start_serve function
trait ServerConnection {
    fn get_x509(&self) -> Option<X509>;
}

impl ServerConnection for SslStream<AddrStream> {
    fn get_x509(&self) -> Option<X509> {
        self.ssl().peer_certificate()
    }
}

impl ServerConnection for AddrStream {
    fn get_x509(&self) -> Option<X509> {
        None
    }
}

// Check if the peer's x509 certificate meets the requirements, this should
// be called where the access should be controlled.
//
// For now, the check only verifies the role of the peer certificate.
fn check_cert(security_config: Arc<SecurityConfig>, cert: Option<X509>) -> bool {
    // if `cert_allowed_cn` is empty, skip check and return true
    if !security_config.cert_allowed_cn.is_empty() {
        if let Some(x509) = cert {
            if let Some(name) = x509
                .subject_name()
                .entries_by_nid(openssl::nid::Nid::COMMONNAME)
                .next()
            {
                let data = name.data().as_slice();
                // Check common name in peer cert
                return security::match_peer_names(
                    &security_config.cert_allowed_cn,
                    std::str::from_utf8(data).unwrap(),
                );
            }
        }
        false
    } else {
        true
    }
}

fn tls_incoming(
    acceptor: SslAcceptor,
    mut incoming: AddrIncoming,
) -> impl Accept<Conn = SslStream<AddrStream>, Error = std::io::Error> {
    let context = acceptor.into_context();
    let s = stream! {
        loop {
            let stream = match poll_fn(|cx| Pin::new(&mut incoming).poll_accept(cx)).await {
                Some(Ok(stream)) => stream,
                Some(Err(e)) => {
                    yield Err(e);
                    continue;
                }
                None => break,
            };
            let ssl = match Ssl::new(&context) {
                Ok(ssl) => ssl,
                Err(err) => {
                    error!("Status server error: {}", err);
                    continue;
                }
            };
            match tokio_openssl::SslStream::new(ssl, stream) {
                Ok(mut ssl_stream) => match Pin::new(&mut ssl_stream).accept().await {
                    Err(e) => {
                        error!(
                            "Status server error: TLS handshake error";
                            "remote_addr" => ssl_stream.get_ref().remote_addr(),
                            "err" => e.to_string()
                        );
                        continue;
                    },
                    Ok(()) => {
                        yield Ok(ssl_stream);
                    },
                }
                Err(err) => {
                    error!("Status server error: {}", err);
                    continue;
                }
            };
        }
    };
    TlsIncoming(s)
}

#[pin_project]
struct TlsIncoming<S>(#[pin] S);

impl<S> Accept for TlsIncoming<S>
where
    S: Stream<Item = std::io::Result<SslStream<AddrStream>>>,
{
    type Conn = SslStream<AddrStream>;
    type Error = std::io::Error;

    fn poll_accept(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<std::io::Result<Self::Conn>>> {
        self.project().0.poll_next(cx)
    }
}

// For handling fail points related requests
#[cfg(feature = "failpoints")]
async fn handle_fail_points_request(req: Request<Body>) -> hyper::Result<Response<Body>> {
    let path = req.uri().path().to_owned();
    let method = req.method().to_owned();
    let fail_path = format!("{}/", FAIL_POINTS_REQUEST_PATH);
    let fail_path_has_sub_path: bool = path.starts_with(&fail_path);

    match (method, fail_path_has_sub_path) {
        (Method::PUT, true) => {
            let mut buf = Vec::new();
            req.into_body()
                .try_for_each(|bytes| {
                    buf.extend(bytes);
                    ok(())
                })
                .await?;
            let (_, name) = path.split_at(fail_path.len());
            if name.is_empty() {
                return Ok(Response::builder()
                    .status(StatusCode::UNPROCESSABLE_ENTITY)
                    .body(MISSING_NAME.into())
                    .unwrap());
            };

            let actions = String::from_utf8(buf).unwrap_or_default();
            if actions.is_empty() {
                return Ok(Response::builder()
                    .status(StatusCode::UNPROCESSABLE_ENTITY)
                    .body(MISSING_ACTIONS.into())
                    .unwrap());
            };

            if let Err(e) = fail::cfg(name.to_owned(), &actions) {
                return Ok(Response::builder()
                    .status(StatusCode::BAD_REQUEST)
                    .body(e.into())
                    .unwrap());
            }
            let body = format!("Added fail point with name: {}, actions: {}", name, actions);
            Ok(Response::new(body.into()))
        }
        (Method::DELETE, true) => {
            let (_, name) = path.split_at(fail_path.len());
            if name.is_empty() {
                return Ok(Response::builder()
                    .status(StatusCode::UNPROCESSABLE_ENTITY)
                    .body(MISSING_NAME.into())
                    .unwrap());
            };

            fail::remove(name);
            let body = format!("Deleted fail point with name: {}", name);
            Ok(Response::new(body.into()))
        }
        (Method::GET, _) => {
            // In this scope the path must be like /fail...(/...), which starts with
            // FAIL_POINTS_REQUEST_PATH and may or may not have a sub path
            // Now we return 404 when path is neither /fail nor /fail/
            if path != FAIL_POINTS_REQUEST_PATH && path != fail_path {
                return Ok(Response::builder()
                    .status(StatusCode::NOT_FOUND)
                    .body(Body::empty())
                    .unwrap());
            }

            // From here path is either /fail or /fail/, return lists of fail points
            let list: Vec<String> = fail::list()
                .into_iter()
                .map(move |(name, actions)| format!("{}={}", name, actions))
                .collect();
            let list = list.join("\n");
            Ok(Response::new(list.into()))
        }
        _ => Ok(Response::builder()
            .status(StatusCode::METHOD_NOT_ALLOWED)
            .body(Body::empty())
            .unwrap()),
    }
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

// Decode different type of json value to string value
fn decode_json(
    data: &[u8],
) -> std::result::Result<std::collections::HashMap<String, String>, Box<dyn std::error::Error>> {
    let json: Value = serde_json::from_slice(data)?;
    if let Value::Object(map) = json {
        let mut dst = std::collections::HashMap::new();
        for (k, v) in map.into_iter() {
            let v = match v {
                Value::Bool(v) => format!("{}", v),
                Value::Number(v) => format!("{}", v),
                Value::String(v) => v,
                Value::Array(_) => return Err("array type are not supported".to_owned().into()),
                _ => return Err("wrong format".to_owned().into()),
            };
            dst.insert(k, v);
        }
        Ok(dst)
    } else {
        Err("wrong format".to_owned().into())
    }
}

fn make_response<T>(status_code: StatusCode, message: T) -> Response<Body>
where
    T: Into<Body>,
{
    Response::builder()
        .status(status_code)
        .body(message.into())
        .unwrap()
}

#[derive(Debug)]
struct RestoreShardRequest {
    pub cs: kvenginepb::ChangeSet,
}

#[derive(Default, Serialize, Deserialize, Debug, PartialEq)]
#[serde(default)]
#[serde(rename_all = "kebab-case")]
pub struct RestoreShardResponse {
    pub shard_id: u64,
    pub restore_bytes: u64,
}

#[derive(Default, Serialize, Deserialize, Debug)]
#[serde(default)]
#[serde(rename_all = "kebab-case")]
pub struct TruncateTsConfig {
    pub cluster_id: u64,
    pub ts: u64,
    pub range: Option<(Vec<u8>, Vec<u8>)>, // None means apply to all keyspaces.
}
