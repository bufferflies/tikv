// Copyright 2022 TiKV Project Authors. Licensed under Apache-2.0.
use std::{collections::HashMap, fmt, sync::Arc, time::Duration};

use bytes::Bytes;
use chrono::{NaiveTime, Utc};
use engine_traits::{GetObjectOptions, ObjectStorage};
use etcd_client::{ConnectOptions, OpenSslClientConfig};
use grpcio::EnvBuilder;
use http::Request;
use hyper::Body;
use kvengine::dfs::{self, Dfs, S3Fs};
use kvproto::{metapb, metapb::Store};
use pd_client::{PdClient, RpcClient};
use protobuf::Message;
use rfengine::{
    assemble_wal_chunks, parse_wal_chunk_key, verify_wal_chunks_integrity, wal_chunk_file_prefix,
    wal_chunk_file_suffix, RfEngine,
};
use rfenginepb::ClusterBackupMeta;
use rfstore::store::state::RaftState;
use security::{SecurityConfig, SecurityManager};
use slog_global::error;
use tikv_util::{box_err, codec::bytes::decode_bytes, info, time::Instant};

use crate::{
    backup::IncrementalBackupFile,
    error::{Error, Result},
};

const MAX_S3_REQ_BATCH_SIZE: usize = 1024;
const FETCH_RFENGINE_WAL_CHUCK_TIMEOUT: Duration = Duration::from_secs(30);
pub const INCREMENTAL_BACKUP_FOLDER_FORMAT: &str = "%Y%m%d";
pub const INCREMENTAL_BACKUP_FILE_NAME_FORMAT: &str = "%H%M%S";

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

pub async fn send_request_to_store(
    req: Request<Body>,
    store: &Store,
    security_mgr: Arc<SecurityManager>,
) -> Result<Bytes> {
    let client = security_mgr.http_client(hyper::Client::builder())?;
    let uri_str = format!("{}", req.uri());
    let resp = client.request(req).await;
    if let Err(err) = resp {
        error!(
            "send request to store failed, store {:?}, err {:?}, uri {:?}",
            store, err, uri_str
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

pub async fn send_request_to_store_with_retry<F>(
    build_req: F,
    store: &Store,
    security_mgr: Arc<SecurityManager>,
    timeout: Duration,
) -> Result<Bytes>
where
    F: Fn() -> Request<Body>,
{
    let is_error_retryable = |err: &Error| matches!(err, Error::HttpError(_));
    let mut last_err: Option<Error> = None;
    let start_time = Instant::now_coarse();
    while start_time.saturating_elapsed() < timeout {
        let req = build_req();
        match send_request_to_store(req, store, security_mgr.clone()).await {
            Ok(resp) => return Ok(resp),
            Err(err) if is_error_retryable(&err) => {
                last_err = Some(err);
                tokio::time::sleep(Duration::from_millis(500)).await;
            }
            Err(err) => return Err(err),
        }
    }
    Err(last_err.unwrap())
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

#[derive(Clone, Default, PartialEq)]
pub(crate) struct RawRegion {
    pub id: u64,
    pub raw_start: Vec<u8>,
    pub raw_end: Vec<u8>,
    pub epoch: metapb::RegionEpoch,
    pub peers: Vec<metapb::Peer>,
}

impl fmt::Debug for RawRegion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RawRegion")
            .field("id", &self.id)
            .field(
                "raw_start",
                &log_wrappers::hex_encode_upper(&self.raw_start),
            )
            .field("raw_end", &log_wrappers::hex_encode_upper(&self.raw_end))
            .field("epoch", &self.epoch)
            .field("peers", &self.peers)
            .finish()
    }
}

impl RawRegion {
    pub fn get_start_key(&self) -> &[u8] {
        self.raw_start.as_slice()
    }

    pub fn get_end_key(&self) -> &[u8] {
        self.raw_end.as_slice()
    }

    /// Equal: `key` is within the boundary of the region.
    /// Less: region is to the left of `key`.
    /// Greater: region is to the right of `key`.
    pub fn compare_with_key(&self, key: &[u8]) -> std::cmp::Ordering {
        if self.raw_end.as_slice() <= key {
            std::cmp::Ordering::Less
        } else if self.raw_start.as_slice() > key {
            std::cmp::Ordering::Greater
        } else {
            std::cmp::Ordering::Equal
        }
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
    security_mgr: Arc<SecurityManager>,
) -> Result<Bytes> {
    let uri = security_mgr.build_uri(format!(
        "{}/rfengine/wal_chunk?epoch_id={}&start_off={}&end_off={}",
        &store.status_address, epoch_id, start_off, end_off
    ))?;
    let req = || Request::get(uri.clone()).body(Body::empty()).unwrap();
    send_request_to_store_with_retry(req, &store, security_mgr, FETCH_RFENGINE_WAL_CHUCK_TIMEOUT)
        .await
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
    let mut wal_chunks = vec![];
    let dfs_prefix = format!("{}/", dfs.get_prefix());
    for replay_epoch in snap_epoch + 1..=backup_epoch {
        let scan_prefix = wal_chunk_file_prefix(store_id, replay_epoch);
        let scan_start = wal_chunk_file_suffix(0, 0);
        info!(
            "replay_wal_logs list chunks with prefix {} start_after {} replay_epoch {} backup meta epoch {}",
            scan_prefix, scan_start, replay_epoch, backup_epoch
        );

        match dfs.list_objects(&scan_start, Some(&scan_prefix), None) {
            Ok((chunks, _)) => {
                let chunk_keys: Vec<String> = chunks
                    .iter()
                    .map(|chunk| {
                        chunk
                            .key
                            .as_str()
                            .strip_prefix(&dfs_prefix)
                            .unwrap_or_default()
                            .to_string()
                    })
                    .collect::<Vec<_>>();
                wal_chunks.push((replay_epoch, chunk_keys));
            }
            Err(err) => {
                error!("list wal chunk files failed: {:?}", err);
                return Err(box_err!("list wal chunk files failed: {:?}", err));
            }
        }
    }

    // Verify the wal chunk files integrity.
    for (epoch_id, chunk_keys) in wal_chunks.iter() {
        if !verify_wal_chunks_integrity(chunk_keys, *epoch_id != backup_epoch) {
            let err_msg = format!(
                "wal chunk files integrity check failed, epoch_id: {} backup_epoch: {} chunk_keys: {:?}",
                epoch_id, backup_epoch, chunk_keys
            );
            return Err(Error::WalChunkIntegrityError(err_msg));
        }
    }

    // Download the wal chunk files concurrently.
    let chunks_data = collect_all_chunk_files(dfs.clone(), wal_chunks)?;

    // Replay epoch wal chunk files in order.
    for (epoch_id, chunks) in chunks_data.into_iter() {
        replay_wal_chunks(
            pd_client.clone(),
            dfs.clone(),
            store_id,
            chunks,
            rf,
            epoch_id,
            backup_epoch,
            backup_offset,
            full_restore,
        )?;
    }

    Ok(())
}

// Collect wal chunk files concurrently and return the chunks data with order.
// The data is compressed with lz4 and all chunks data size about 1GB at most,
// so it's safe to keep all data in memory.
fn collect_all_chunk_files(
    dfs: Arc<S3Fs>,
    wal_chunks: Vec<(u32, Vec<String>)>,
) -> Result<Vec<(u32, Vec<Bytes>)>> {
    let epoch_cnt = wal_chunks.len();
    let objects_cnt = wal_chunks
        .iter()
        .map(|(_, chunks)| chunks.len())
        .sum::<usize>();
    let mut all_chunk_keys_with_option = Vec::with_capacity(objects_cnt);
    for (_, chunk_keys) in wal_chunks.into_iter() {
        all_chunk_keys_with_option.extend(
            chunk_keys
                .into_iter()
                .map(|key| (key, GetObjectOptions::default()))
                .collect::<Vec<_>>(),
        );
    }

    let chunks = dfs
        .get_objects(all_chunk_keys_with_option)
        .map_err(|e| Error::DfsError(dfs::Error::S3(e)))?;

    let mut chunks_map = HashMap::with_capacity(epoch_cnt);
    for (key, value) in chunks.into_iter() {
        let (epoch_id, ..) = parse_wal_chunk_key(Some(&key)).unwrap();

        info!(
            "wal chunk file {} epoch {} size {}",
            key,
            epoch_id,
            value.len()
        );
        chunks_map
            .entry(epoch_id)
            .or_insert_with(Vec::new)
            .push((key, value));
    }
    let mut sorted_by_epoch = chunks_map.into_iter().collect::<Vec<_>>();
    sorted_by_epoch.sort_by_key(|k| k.0);

    let chunks_data = sorted_by_epoch
        .into_iter()
        .map(|(epoch_id, mut chunks_vec)| {
            // Sort objects by chunk name.
            chunks_vec.sort_by(|a, b| a.0.cmp(&b.0));
            (
                epoch_id,
                chunks_vec
                    .into_iter()
                    .map(|(_, data)| data)
                    .collect::<Vec<_>>(),
            )
        })
        .collect::<Vec<_>>();
    Ok(chunks_data)
}

fn replay_wal_chunks(
    pd_client: Arc<dyn PdClient>,
    dfs: Arc<S3Fs>,
    store_id: u64,
    chunks: Vec<Bytes>,
    rf: &RfEngine,
    epoch_id: u32,
    backup_epoch: u32,
    backup_offset: u64,
    full_restore: bool,
) -> Result<()> {
    // Assemble WAL chunks in memory.
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
        let security_mgr = pd_client.get_security_mgr();
        let last_chunk = runtime.block_on(fetch_rfengine_wal_chunk(
            store,
            epoch_id,
            epoch_wal.len() as u64,
            backup_offset,
            security_mgr,
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

/// Return full path of incremental backups in S3.
pub async fn get_all_incremental_backups(
    s3fs: &S3Fs,
    start_date: &chrono::NaiveDate,
    start_time: Option<&NaiveTime>,
    max_count: usize,
) -> dfs::Result<(Vec<IncrementalBackupFile>, bool)> {
    let mut files = Vec::with_capacity(std::cmp::min(max_count, 1000));
    let mut start_key = format!(
        "{}/{}",
        start_date.format(INCREMENTAL_BACKUP_FOLDER_FORMAT),
        start_time
            .map(|t| t.format(INCREMENTAL_BACKUP_FILE_NAME_FORMAT).to_string())
            .unwrap_or_default()
    );
    let prefix = "backup/";

    let mut reach_limit = false;
    loop {
        // TODO: pass in `max_count` for limit.
        match s3fs.list(&start_key, Some(prefix), None).await {
            Ok((backup_files, more, next_start_after)) => {
                let mut inc_files = backup_files
                    .into_iter()
                    .filter_map(|f| IncrementalBackupFile::try_from_full_path(&f.key))
                    .collect::<Vec<_>>();
                files.append(&mut inc_files);
                if files.len() > max_count {
                    files.truncate(max_count);
                    reach_limit = true;
                }
                if reach_limit || !more {
                    break;
                }
                start_key = next_start_after.unwrap();
            }
            Err(e) => {
                return Err(e);
            }
        }
    }
    Ok((files, reach_limit))
}

// If backup exist, return the latest one, else create a new ClusterBackupMeta.
pub async fn get_latest_backup_meta(s3fs: &S3Fs, cluster_id: u64) -> Result<ClusterBackupMeta> {
    let now = Utc::now();
    let (files, _) = get_all_incremental_backups(s3fs, &now.date_naive(), None, usize::MAX).await?;
    if files.is_empty() {
        return Err(Error::MetaNotFound(cluster_id));
    }
    // Incremental backup file name is generated with `backup_file_full_path` named
    // by creation time. The last should be the latest one.
    let last_file = files.last().unwrap();
    let full_path = last_file.full_path(&s3fs.get_prefix());
    let object = s3fs
        .get_object(
            full_path.clone(),
            full_path.clone(),
            engine_traits::GetObjectOptions::default(),
        )
        .await?;
    let mut meta = ClusterBackupMeta::new();
    meta.merge_from_bytes(&object).unwrap();
    if meta.cluster_id != cluster_id {
        return Err(Error::MetaNotFound(cluster_id));
    }

    info!(
        "Get cluster {} latest backup meta {}, store cnt {}",
        meta.cluster_id,
        full_path,
        meta.stores.len()
    );
    Ok(meta)
}

pub fn check_store_id_exists(s3fs: &S3Fs, store_id: u64) -> Result<bool> {
    let prefix = format!("store_backup/{:016x}/", store_id);
    let (files, ..) = s3fs
        .get_runtime()
        .block_on(s3fs.list("", Some(&prefix), None))?;
    Ok(!files.is_empty())
}

#[derive(Debug)]
pub struct StorePeer {
    pub store_id: u64,
    pub peer_id: u64,
}

#[derive(Clone)]
pub struct RegionMetaGetter {
    pub shard_store_map: Arc<HashMap<u64 /* shard_id */, StorePeer>>,
    pub raft_engines: Arc<HashMap<u64 /* store_id */, rfengine::RfEngine>>,
}

impl fmt::Debug for RegionMetaGetter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ShardMetaGetter")
            .field("shard_store_map", &self.shard_store_map)
            .field(
                "raft_engines",
                &self.raft_engines.keys().collect::<Vec<_>>(),
            )
            .finish()
    }
}

impl RegionMetaGetter {
    pub fn load_region_meta(
        &self,
        shard_id: u64,
        shard_ver: u64,
        kvengine_store_id: u64,
    ) -> Option<metapb::Region> {
        self.shard_store_map.get(&shard_id).and_then(|sp| {
            self.raft_engines.get(&sp.store_id).map(|rf| {
                let mut region_state =
                    rf.load_region_state(sp.peer_id, shard_ver)
                        .unwrap_or_else(|| {
                            panic!(
                                "{}:{} failed to get region state, state key {:?}",
                                shard_id,
                                shard_ver,
                                rfengine::region_state_key(shard_ver),
                            );
                        });
                let mut region = region_state.take_region();

                // Filter the leader peer by store id.
                let mut peers = region.take_peers().into_vec();
                peers.retain(|p| p.store_id == sp.store_id);
                assert!(
                    peers.len() == 1,
                    "leader peer not found, region {:?}, store_id {}",
                    rf.load_region_state(sp.peer_id, shard_ver).unwrap(),
                    sp.store_id
                );
                // Move the peer to the same store with kvengine.
                // To keep the region meta consistent with kvengine during applying committing
                // locks.
                peers[0].store_id = kvengine_store_id;
                region.mut_peers().push(peers.pop().unwrap());

                region
            })
        })
    }
}
