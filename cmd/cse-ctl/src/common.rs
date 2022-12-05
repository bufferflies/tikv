// Copyright 2022 TiKV Project Authors. Licensed under Apache-2.0.
use std::{result::Result, sync::Arc};

use bytes::Bytes;
use grpcio::EnvBuilder;
use http::Request;
use hyper::Body;
use kvproto::metapb::Store;
use pd_client::RpcClient;
use security::{SecurityConfig, SecurityManager};

pub(crate) fn create_pd_client(
    security_conf: &SecurityConfig,
    pd_conf: &pd_client::Config,
) -> RpcClient {
    let security_mgr = Arc::new(
        SecurityManager::new(security_conf)
            .unwrap_or_else(|e| panic!("failed to create security manager: {:?}", e)),
    );
    let env = Arc::new(EnvBuilder::new().cq_count(1).build());
    RpcClient::new(pd_conf, Some(env), security_mgr)
        .unwrap_or_else(|e| panic!("failed to create rpc client: {:?}", e))
}

pub(crate) async fn send_request_to_store(
    req: Request<Body>,
    store: Store,
) -> Result<Bytes, String> {
    let client = hyper::Client::new();
    let resp = client.request(req).await;
    if resp.is_err() {
        return Err(format!("{:?} {:?}", &store, resp.unwrap_err()));
    }
    let resp = resp.unwrap();
    if !resp.status().is_success() {
        return Err(format!("{:?} {:?}", &store, resp.status()));
    }
    match hyper::body::to_bytes(resp.into_body()).await {
        Ok(body) => Ok(body),
        Err(e) => Err(format!("{:?} {:?}", &store, e)),
    }
}
