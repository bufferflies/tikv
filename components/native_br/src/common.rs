// Copyright 2022 TiKV Project Authors. Licensed under Apache-2.0.
use std::{str::FromStr, sync::Arc, time::Duration};

use bytes::Bytes;
use engine_traits::{GetObjectOptions, ListObjectContent, ObjectStorage};
use etcd_client::{ConnectOptions, OpenSslClientConfig};
use grpcio::EnvBuilder;
use http::Request;
use hyper::{Body, Uri};
use kvengine::dfs::{Dfs, S3Fs};
use kvproto::{metapb, metapb::Store};
use pd_client::{PdClient, RpcClient};
use protobuf::Message;
use rfengine::{
    assemble_wal_chunks, verify_wal_chunks_integrity, wal_chunk_file_prefix, wal_chunk_file_suffix,
    RfEngine,
};
use rfenginepb::ClusterBackupMeta;
use rfstore::store::state::RaftState;
use security::{SecurityConfig, SecurityManager};
use slog_global::error;
use tikv_util::{box_err, codec::bytes::decode_bytes, info};

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
    let mut handles = Vec::with_capacity(file_cnt);
    for &id in file_ids {
        let s3fs = s3fs.clone();
        handles.push(runtime.spawn(async move { s3fs.retain_file(id).await }));
    }
    // To avoid too much request to cause s3 SlowDown issue.
    if !first_batch {
        std::thread::sleep(Duration::from_secs(1));
    }
    let mut succeed_cnt = 0;
    for (handle, &file_id) in handles.into_iter().zip(file_ids) {
        match runtime.block_on(handle).unwrap() {
            Ok(_) => succeed_cnt += 1,
            Err(e) => error!("Retain file {} fail {:?}", file_id, e),
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

async fn fetch_rfengine_wal_chunk(
    store: Store,
    epoch_id: u32,
    start_off: u64,
    end_off: u64,
) -> Result<Bytes> {
    let uri = Uri::from_str(&format!(
        "http://{}/rfengine/wal_chunk?epoch_id={}&start_off={}&end_off={}",
        &store.status_address, epoch_id, start_off, end_off
    ))
    .unwrap();
    let req = Request::get(uri).body(Body::empty()).unwrap();
    send_request_to_store(req, &store).await
}

// TODO: Filter out the write batches of specified keyspace to replay to save
// memory.
pub fn replay_wal_logs(
    pd_client: Arc<dyn PdClient>,
    dfs: Arc<S3Fs>,
    store_id: u64,
    cluster_backup: &ClusterBackupMeta,
    rf: &RfEngine,
    snap_epoch: u32,
    full_restore: bool,
) -> Result<()> {
    let store_meta = cluster_backup
        .get_stores()
        .iter()
        .find(|x| x.store_id == store_id)
        .expect("store not found");
    let backup_epoch = store_meta.get_epoch();
    let backup_offset = store_meta.get_offset();
    for replay_epoch in snap_epoch + 1..=backup_epoch {
        let scan_prefix = wal_chunk_file_prefix(store_id, replay_epoch);
        let scan_start = wal_chunk_file_suffix(0, 0);
        info!(
            "replay_wal_logs list chunks with prefix {} start_after {} replay_epoch {} backup meta epoch {}",
            scan_prefix, scan_start, replay_epoch, backup_epoch
        );

        let dfs_clone = dfs.clone();
        match dfs.list_objects(&scan_start, Some(&scan_prefix), None) {
            Ok((chunks, _)) => {
                replay_wal_chunks(
                    pd_client.clone(),
                    dfs_clone,
                    store_id,
                    chunks,
                    rf,
                    replay_epoch,
                    backup_epoch,
                    backup_offset,
                    full_restore,
                )?;
            }
            Err(err) => {
                error!("list wal chunk files failed: {:?}", err);
                return Err(box_err!("list wal chunk files failed: {:?}", err));
            }
        }
    }

    Ok(())
}

fn replay_wal_chunks(
    pd_client: Arc<dyn PdClient>,
    dfs: Arc<S3Fs>,
    store_id: u64,
    chunks: Vec<ListObjectContent>,
    rf: &RfEngine,
    epoch_id: u32,
    backup_epoch: u32,
    backup_offset: u64,
    full_restore: bool,
) -> Result<()> {
    let dfs_prefix = format!("{}/", dfs.get_prefix());
    let chunk_keys = chunks
        .into_iter()
        .map(|chunk| {
            chunk
                .key
                .as_str()
                .strip_prefix(&dfs_prefix)
                .unwrap_or_default()
                .to_string()
        })
        .collect::<Vec<_>>();
    if !verify_wal_chunks_integrity(&chunk_keys, epoch_id != backup_epoch) {
        let err_msg = format!(
            "wal chunk files integrity check failed, epoch_id: {} backup_epoch: {} chunk_keys: {:?}",
            epoch_id, backup_epoch, chunk_keys
        );
        return Err(box_err!(&err_msg));
    }
    let chunk_keys_with_option = chunk_keys
        .into_iter()
        .map(|chunk| (chunk, GetObjectOptions::default()))
        .collect::<Vec<_>>();
    // Assemble WAL chunks in memory.
    let mut chunk_objects = dfs.get_objects(chunk_keys_with_option).unwrap();
    // Sort objects by chunk name.
    chunk_objects.sort_by(|a, b| a.0.cmp(&b.0));
    info!(
        "wal chunk files in epoch {} {:?}",
        epoch_id,
        chunk_objects
            .iter()
            .map(|o| o.0.clone())
            .collect::<Vec<_>>()
    );
    let chunks = chunk_objects
        .into_iter()
        .map(|(_, chunk)| chunk)
        .collect::<Vec<_>>();
    let mut epoch_wal = assemble_wal_chunks(chunks)?;

    info!(
        "assemble wal from chunks done, epoch {} wal size {} backup offset {}",
        epoch_id,
        epoch_wal.len(),
        backup_offset
    );
    let end_offset = if backup_epoch == epoch_id && backup_offset <= epoch_wal.len() as u64 {
        // Replay finished.
        backup_offset
    } else if backup_epoch == epoch_id {
        // If the cluster is for full restore, the wal should be integrated.
        if full_restore {
            return Err(box_err!(
                "wal chunk is not integrated, choose a earlier backup"
            ));
        }
        // Need fetch the last chunk from server then append fetched data to epoch_wal.
        let store = pd_client.get_store(store_id).unwrap();
        let runtime = dfs.get_runtime();
        let last_chunk = runtime.block_on(fetch_rfengine_wal_chunk(
            store,
            epoch_id,
            epoch_wal.len() as u64,
            backup_offset,
        ))?;
        info!(
            "fetched last wal chunk start_off {} end_off {} data len {}",
            epoch_wal.len(),
            backup_offset,
            last_chunk.len()
        );
        epoch_wal.extend(last_chunk);
        epoch_wal.len() as u64
    } else {
        // This is a previous epoch, replay all chunks.
        u64::MAX
    };

    info!(
        "replay wal chunk file for epoch {} size {} end_offset {}",
        epoch_id,
        epoch_wal.len(),
        end_offset
    );
    rf.replay_wal_file(epoch_wal.freeze(), epoch_id, end_offset, full_restore)?;

    Ok(())
}
