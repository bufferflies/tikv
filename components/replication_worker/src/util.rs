// Copyright 2025 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::Duration,
};

use api_version::{api_v2::KEYSPACE_PREFIX_LEN, ApiV2};
use bytes::Bytes;
use http::{Request, StatusCode, Uri};
use hyper::Body;
use kvengine::GLOBAL_SHARD_END_KEY;
use kvproto::cdcpb::ChangeDataRequest;
use pd_client::RpcClient;
use security::{HttpClient, SecurityConfig, SecurityManager};
use tikv_util::{codec::bytes::decode_bytes, error, info, time::Instant, warn};
use txn_types::TimeStamp;

use crate::{ticdc_util, ticdc_util::TiCdcError, Error};

pub(crate) const DISPATCH_CDC_TIMEOUT: Duration = Duration::from_secs(30);

pub(crate) async fn send_request_to_store(
    req: Request<Body>,
    client: &HttpClient,
    timeout: Duration,
) -> crate::Result<(StatusCode, Bytes)> {
    debug_assert!(!timeout.is_zero());
    let uri_str = format!("{}", req.uri());
    let resp_res = tokio::time::timeout(timeout, client.request(req))
        .await
        .map_err(|_| Error::StoreTimeout(format!("send request to {uri_str}")))?;
    let resp = resp_res?;
    let status = resp.status();
    let body = tokio::time::timeout(timeout, hyper::body::to_bytes(resp.into_body()))
        .await
        .map_err(|_| Error::StoreTimeout(format!("read response from {uri_str}")))??;
    Ok((status, body))
}

async fn dispatch_http_post_with_retry(
    client: &HttpClient,
    uri: &Uri,
    body: Bytes,
    timeout: Duration,
) -> crate::Result<(StatusCode, Bytes)> {
    let mut last_err: Option<Error> = None;
    let start_time = Instant::now_coarse();
    while start_time.saturating_elapsed() < timeout {
        match dispatch_http_post(client, uri, body.clone()).await {
            Ok(resp) => {
                return Ok(resp);
            }
            Err(err) => {
                last_err = Some(err);
                tokio::time::sleep(Duration::from_millis(500)).await;
            }
        }
    }
    Err(last_err.unwrap())
}

async fn dispatch_http_post(
    client: &HttpClient,
    uri: &Uri,
    req_body: Bytes,
) -> crate::Result<(StatusCode, Bytes)> {
    let req = http::Request::builder()
        .uri(uri)
        .method("POST")
        .body(req_body.into())
        .unwrap();
    let resp = client.request(req).await?;
    let status = resp.status();
    let body = hyper::body::to_bytes(resp.into_body()).await?;
    Ok((status, body))
}

pub(crate) async fn post_to_ticdc<F>(
    tag: &str,
    client: &HttpClient,
    uri: &Uri,
    body: Bytes,
    is_err_retryable: F,
) -> crate::Result<(StatusCode, Bytes)>
where
    F: Fn(&TiCdcError) -> bool,
{
    info!("{} post_to_ticdc", tag;
        "uri" => ?uri, "body" => String::from_utf8_lossy(&body).as_ref());
    let mut last_resp: Option<(StatusCode, Bytes)> = None;
    let start_time = Instant::now_coarse();
    while start_time.saturating_elapsed() < DISPATCH_CDC_TIMEOUT {
        let (status, resp) =
            dispatch_http_post_with_retry(client, uri, body.clone(), DISPATCH_CDC_TIMEOUT).await?;
        if !status.is_success() {
            let ticdc_err = ticdc_util::parse_ticdc_response(&resp);
            if is_err_retryable(&ticdc_err) {
                warn!("{} post_to_ticdc error", tag; "resp" => String::from_utf8_lossy(&resp).as_ref());
                last_resp = Some((status, resp));
                tokio::time::sleep(Duration::from_millis(500)).await;
                continue;
            }
        }
        info!("{} post_to_ticdc success", tag; "resp" => String::from_utf8_lossy(&resp).as_ref());
        return Ok((status, resp));
    }
    let (status, resp) = last_resp.unwrap();
    error!("{} post_to_ticdc error", tag; "status" => ?status, "resp" => String::from_utf8_lossy(&resp).as_ref());
    Ok((status, resp))
}

pub(crate) fn build_request_range(request: &ChangeDataRequest) -> (Vec<u8>, Vec<u8>) {
    let mut encoded_start_key = request.get_start_key();
    let start_key = decode_bytes(&mut encoded_start_key, false).unwrap_or_default();
    let end_key = if request.get_end_key().is_empty() {
        GLOBAL_SHARD_END_KEY.to_vec()
    } else {
        let mut encoded_end_key = request.get_end_key();
        decode_bytes(&mut encoded_end_key, false).unwrap_or_default()
    };
    (start_key, end_key)
}

pub(crate) fn build_request_range_for_keyspace(
    keyspace_id: u32,
    request: &ChangeDataRequest,
) -> (Vec<u8>, Vec<u8>) {
    let (start_key, end_ekey) = build_request_range(request);
    let mut prepended_start_key = ApiV2::get_keyspace_prefix_by_id(keyspace_id);
    prepended_start_key.extend_from_slice(&start_key);
    let mut prepended_end_key = ApiV2::get_keyspace_prefix_by_id(keyspace_id);
    prepended_end_key.extend_from_slice(&end_ekey);
    (prepended_start_key, prepended_end_key)
}

pub(crate) async fn new_keyspace_pd_client(
    pd_url: String,
    sec_conf: &SecurityConfig,
) -> Arc<RpcClient> {
    let sec_mgr = Arc::new(SecurityManager::new(sec_conf).unwrap());
    Arc::new(
        RpcClient::new_async(&pd_client::Config::new(vec![pd_url]), None, sec_mgr)
            .await
            .unwrap(),
    )
}

pub(crate) fn keyspace_prefix_len(keyspace_id: u32) -> usize {
    if keyspace_id > 0 {
        KEYSPACE_PREFIX_LEN
    } else {
        0
    }
}

#[derive(Default)]
pub(crate) struct ResolvedTsStats {
    pub(crate) resolved_regions: usize,
    /// Count of regions are still scanning locks.
    pub(crate) unresolved_regions: usize,

    pub(crate) min_ts: TimeStamp,
    pub(crate) min_ts_region_id: u64,
}

impl ResolvedTsStats {
    pub(crate) fn record_resolved_region(&mut self, region_id: u64, resolved_ts: TimeStamp) {
        self.resolved_regions += 1;
        if self.min_ts.is_zero() || self.min_ts > resolved_ts {
            self.min_ts = resolved_ts;
            self.min_ts_region_id = region_id;
        }
    }

    pub(crate) fn record_unresolved_region(&mut self, _region_id: u64) {
        self.unresolved_regions += 1;
    }
}

#[derive(Clone)]
pub(crate) struct ArcTimeStamp(Arc<AtomicU64>);

impl ArcTimeStamp {
    pub(crate) fn get(&self) -> TimeStamp {
        TimeStamp::new(self.0.load(Ordering::Relaxed))
    }

    pub(crate) fn set(&self, ts: TimeStamp) {
        self.0.store(ts.into_inner(), Ordering::Relaxed);
    }
}

impl From<TimeStamp> for ArcTimeStamp {
    fn from(ts: TimeStamp) -> Self {
        ArcTimeStamp(Arc::new(AtomicU64::new(ts.into_inner())))
    }
}
