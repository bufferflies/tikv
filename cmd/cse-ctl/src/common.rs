// Copyright 2022 TiKV Project Authors. Licensed under Apache-2.0.
use std::{error::Error, result::Result, sync::Arc};

use bytes::Bytes;
use etcd_client::{ConnectOptions, OpenSslClientConfig};
use grpcio::EnvBuilder;
use http::Request;
use hyper::Body;
use kvproto::{metapb, metapb::Store};
use pd_client::{PdClient, RpcClient};
use protobuf::Message;
use rfstore::store::state::RaftState;
use security::{SecurityConfig, SecurityManager};
use tikv_util::codec::bytes::decode_bytes;

pub fn create_pd_client(security_conf: &SecurityConfig, pd_conf: &pd_client::Config) -> RpcClient {
    let security_mgr = Arc::new(
        SecurityManager::new(security_conf)
            .unwrap_or_else(|e| panic!("failed to create security manager: {:?}", e)),
    );
    let env = Arc::new(EnvBuilder::new().cq_count(1).build());
    RpcClient::new(pd_conf, Some(env), security_mgr)
        .unwrap_or_else(|e| panic!("failed to create rpc client: {:?}", e))
}

pub fn get_all_stores_except_tiflash(
    pd_client: &dyn PdClient,
) -> Result<Vec<Store>, pd_client::Error> {
    Ok(pd_client
        .get_all_stores(true)?
        .into_iter()
        .filter(|s| {
            !s.get_labels().iter().any(|l| {
                // including "tiflash" & "tiflash_compute"
                l.key.to_lowercase() == "engine" && l.value.to_lowercase().starts_with("tiflash")
            })
        })
        .collect())
}

pub async fn send_request_to_store(req: Request<Body>, store: &Store) -> Result<Bytes, String> {
    let client = hyper::Client::new();
    let resp = client.request(req).await;
    if resp.is_err() {
        return Err(format!("{:?} {:?}", store, resp.unwrap_err()));
    }
    let resp = resp.unwrap();
    if !resp.status().is_success() {
        let status = resp.status();
        let body = hyper::body::to_bytes(resp.into_body()).await.unwrap();
        return Err(format!("{:?} {:?}: {:?}", store, status, body));
    }
    match hyper::body::to_bytes(resp.into_body()).await {
        Ok(body) => Ok(body),
        Err(e) => Err(format!("{:?} {:?}", store, e)),
    }
}

pub fn generate_etcd_connect_opt(
    security: &SecurityConfig,
) -> Result<ConnectOptions, Box<dyn Error>> {
    let mut option = ConnectOptions::new();
    if !security.ca_path.is_empty() {
        let (ca, cert, key) = security.load_certs()?;
        option = option.with_openssl_tls(
            OpenSslClientConfig::default()
                .ca_cert_pem(&ca)
                .client_cert_pem_and_key(&cert, &key),
        );
    }
    Ok(option)
}

pub fn load_rf_engine_meta(rf: &rfengine::RfEngine, peer_id: u64) -> Option<kvenginepb::ChangeSet> {
    rf.get_state(peer_id, rfengine::KV_ENGINE_META_KEY)
        .map(|engine_meta_val| {
            let mut cs = kvenginepb::ChangeSet::new();
            cs.merge_from_bytes(&engine_meta_val).unwrap();
            cs
        })
}

pub fn load_peer_raft_state(
    rf: &rfengine::RfEngine,
    peer_id: u64,
    region_version: u64,
) -> Option<RaftState> {
    let raft_state_key = rfengine::raft_state_key(region_version);
    let raft_state_val = rf.get_state(peer_id, &raft_state_key)?;
    let mut raft_state = RaftState::default();
    raft_state.unmarshal(raft_state_val.as_ref());
    Some(raft_state)
}

#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct RawRegion {
    pub id: u64,
    pub raw_start: Vec<u8>,
    pub raw_end: Vec<u8>,
    pub epoch: metapb::RegionEpoch,
    pub peers: Vec<metapb::Peer>,
}

impl RawRegion {
    pub fn get_start_key(&self) -> &[u8] {
        self.raw_start.as_slice()
    }

    pub fn get_end_key(&self) -> &[u8] {
        self.raw_end.as_slice()
    }

    pub fn take_start_key(&mut self) -> Vec<u8> {
        std::mem::take(&mut self.raw_start)
    }

    pub fn take_end_key(&mut self) -> Vec<u8> {
        std::mem::take(&mut self.raw_end)
    }
}

impl From<metapb::Region> for RawRegion {
    fn from(mut region: metapb::Region) -> Self {
        let raw_start = if region.start_key.is_empty() {
            vec![]
        } else {
            let mut slice = region.start_key.as_slice();
            decode_bytes(&mut slice, false).unwrap()
        };
        let raw_end = if region.end_key.is_empty() {
            vec![255; 8]
        } else {
            let mut slice = region.end_key.as_slice();
            decode_bytes(&mut slice, false).unwrap()
        };
        RawRegion {
            id: region.id,
            raw_start,
            raw_end,
            epoch: region.take_region_epoch(),
            peers: region.take_peers().into_vec(),
        }
    }
}

#[inline]
pub fn now() -> String {
    chrono::Local::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, false)
}

#[macro_export]
macro_rules! step( ($($args:tt)+) => {
    let msg = format!($($args)+);
    info!("{}", msg);
    println!("[{}] {}", now(), msg);
};);

#[macro_export]
macro_rules! step_error( ($($args:tt)+) => {
    let msg = format!($($args)+);
    error!("{}", msg);
    eprintln!("[{}] {}", now(), msg);
};);
