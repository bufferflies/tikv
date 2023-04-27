// Copyright 2022 TiKV Project Authors. Licensed under Apache-2.0.
use std::{sync::Arc, time::Duration};

use bytes::Bytes;
use etcd_client::{ConnectOptions, OpenSslClientConfig};
use grpcio::EnvBuilder;
use http::Request;
use hyper::Body;
use kvengine::dfs::{Dfs, S3Fs};
use kvproto::{metapb, metapb::Store};
use pd_client::{PdClient, RpcClient};
use protobuf::Message;
use rfstore::store::state::RaftState;
use security::{SecurityConfig, SecurityManager};
use slog_global::error;
use tikv_util::{box_err, codec::bytes::decode_bytes};

use crate::error::Result;

const MAX_S3_REQ_BATCH_SIZE: usize = 1024;

pub fn create_pd_client(security_conf: &SecurityConfig, pd_conf: &pd_client::Config) -> RpcClient {
    let security_mgr = Arc::new(
        SecurityManager::new(security_conf)
            .unwrap_or_else(|e| panic!("failed to create security manager: {:?}", e)),
    );
    let env = Arc::new(EnvBuilder::new().cq_count(1).build());
    RpcClient::new(pd_conf, Some(env), security_mgr)
        .unwrap_or_else(|e| panic!("failed to create rpc client: {:?}", e))
}

pub fn get_all_stores_except_tiflash(pd_client: &dyn PdClient) -> Result<Vec<Store>> {
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

pub fn get_tiflash_storage_stores(pd_client: &dyn PdClient) -> Result<Vec<Store>> {
    Ok(pd_client
        .get_all_stores(true)?
        .into_iter()
        .filter(|s| {
            let mut is_tiflash = false;
            let mut is_write_role = false;
            for l in s.get_labels().iter() {
                if l.key.to_lowercase() == "engine" && l.value.to_lowercase() == "tiflash" {
                    is_tiflash = true;
                }
                // exclude the tiflash write node
                if l.key.to_lowercase() == "engine_role" && l.value.to_lowercase() == "write" {
                    is_write_role = true;
                }
            }
            is_tiflash && !is_write_role
        })
        .collect())
}

pub async fn send_request_to_store(req: Request<Body>, store: &Store) -> Result<Bytes> {
    let client = hyper::Client::new();
    let resp = client.request(req).await;
    if let Err(err) = resp {
        error!(
            "send request to store failed, store {:?}, err {:?}",
            store, err
        );
        return Err(err.into());
    }
    let resp = resp.unwrap();
    if !resp.status().is_success() {
        let status = resp.status();
        let body = hyper::body::to_bytes(resp.into_body()).await.unwrap();
        return Err(box_err!("{:?} {:?}: {:?}", store, status, body));
    }
    match hyper::body::to_bytes(resp.into_body()).await {
        Ok(body) => Ok(body),
        Err(e) => Err(box_err!("{:?} {:?}", store, e)),
    }
}

pub fn generate_etcd_connect_opt(security: &SecurityConfig) -> Result<ConnectOptions> {
    if security.ca_path.is_empty() {
        return Ok(ConnectOptions::new());
    }
    match security.load_certs() {
        Ok((ca, cert, key)) => Ok(ConnectOptions::new().with_openssl_tls(
            OpenSslClientConfig::default()
                .ca_cert_pem(&ca)
                .client_cert_pem_and_key(&cert, &key),
        )),
        Err(e) => Err(box_err!("Load security fail {:?}", e)),
    }
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

pub fn retain_sst_files(file_ids: Vec<u64>, s3fs: &S3Fs) -> Result<usize> {
    let mut idx = 0;
    let mut total_cnt = 0;
    while idx < file_ids.len() {
        let end_idx = std::cmp::min(file_ids.len(), idx + MAX_S3_REQ_BATCH_SIZE);
        total_cnt += retain_sst_files_in_batch(&file_ids[idx..end_idx], s3fs, idx == 0)?;
        idx = end_idx;
    }
    Ok(total_cnt)
}

fn retain_sst_files_in_batch(file_ids: &[u64], s3fs: &S3Fs, first_batch: bool) -> Result<usize> {
    let runtime = s3fs.get_runtime();
    let file_cnt = file_ids.len();
    let (tx, rx) = tikv_util::mpsc::unbounded();
    for id in file_ids {
        runtime.spawn(retain_s3_file(s3fs.clone(), id.to_owned(), tx.clone()));
    }
    // To avoid too much request to cause s3 SlowDown issue.
    if !first_batch {
        std::thread::sleep(Duration::from_secs(1));
    }
    let mut succeed_cnt = 0;
    for _ in 0..file_cnt {
        match rx.recv().unwrap() {
            Ok(_) => succeed_cnt += 1,
            Err(e) => error!("{}", e),
        }
    }
    if succeed_cnt != file_cnt {
        return Err(box_err!(
            "Error occurs, succeed {}, totally {}",
            succeed_cnt,
            file_cnt
        ));
    }
    Ok(succeed_cnt)
}

async fn retain_s3_file(fs: S3Fs, file_id: u64, tx: tikv_util::mpsc::Sender<Result<u64>>) {
    match fs.retain_file(file_id).await {
        Ok(()) => tx.send(Ok(file_id)).unwrap(),
        Err(e) => tx
            .send(Err(box_err!("Retain file {:?} fail {}", file_id, e)))
            .unwrap(),
    }
}
