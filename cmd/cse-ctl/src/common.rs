// Copyright 2022 TiKV Project Authors. Licensed under Apache-2.0.
use std::{error::Error, result::Result, sync::Arc};

use bytes::Bytes;
use etcd_client::ConnectOptions;
use grpcio::EnvBuilder;
use http::Request;
use hyper::Body;
use kvproto::metapb::Store;
use pd_client::{PdClient, RpcClient};
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

pub(crate) fn get_all_stores_except_tiflash(
    pd_client: &dyn PdClient,
) -> Result<Vec<Store>, pd_client::Error> {
    Ok(pd_client
        .get_all_stores(true)?
        .into_iter()
        .filter(|s| {
            !s.get_labels()
                .iter()
                .any(|l| l.key.to_lowercase() == "engine" && l.value.to_lowercase() == "tiflash")
        })
        .collect())
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
        let status = resp.status();
        let body = hyper::body::to_bytes(resp.into_body()).await.unwrap();
        return Err(format!("{:?} {:?}: {:?}", &store, status, body));
    }
    match hyper::body::to_bytes(resp.into_body()).await {
        Ok(body) => Ok(body),
        Err(e) => Err(format!("{:?} {:?}", &store, e)),
    }
}

pub(crate) fn generate_etcd_connect_opt(
    security: &SecurityConfig,
) -> Result<ConnectOptions, Box<dyn Error>> {
    let option = ConnectOptions::new();
    if !security.ca_path.is_empty() {
        // TODO: add tls support, the etcd-client tls has issue
        // See: https://github.com/etcdv3/etcd-client/issues/49
    }
    Ok(option)
}
