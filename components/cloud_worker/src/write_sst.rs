use std::{collections::HashMap, sync::Arc};

use bytes::{Buf, Bytes, BytesMut};
use cloud_server::status_server::SstMeta;
use futures::StreamExt;
use http::StatusCode;
use hyper::{Body, Request, Response, Result};
use kvengine::{
    dfs::{Dfs, Options},
    table::{sstable::Builder, InnerKey, Value},
    UserMeta,
};
use pd_client::PdClient;
use rfstore::store::util::PdIdAllocator;
use serde_json::json;
use tikv_util::info;
use tokio::sync::Mutex;
use url::form_urlencoded;

use crate::{common::make_response, worker_limiter::WorkerLimiter};

/// Handle the /write_sst HTTP endpoint for chunked SST file writing and
/// building. TiDB calls this API to write kv pairs, returns the SST filename in
/// JSON.
///
/// - PUT /write_sst?cluster_id=%d&commit_ts=%d&compression=zstd
///   - Chunk encoding: key_len(2) + key + val_len(4) + value
///   - Only Write CF, key is with keyspace, value is plain after sql encoded
///   - Returns: {"filename": "..."} or {"error": "..."}
pub async fn handle_write_sst(
    ctx: Arc<crate::server::Context>,
    req: Request<Body>,
) -> Result<Response<Body>> {
    let method = req.method().clone();
    let uri = req.uri();
    let query = uri.query().unwrap_or("");
    let params: HashMap<_, _> = form_urlencoded::parse(query.as_bytes())
        .into_owned()
        .collect();
    let cluster_id = params.get("cluster_id").and_then(|v| v.parse::<u64>().ok());
    let commit_ts = params.get("commit_ts").and_then(|v| v.parse::<u64>().ok());
    let compression = match params.get("compression") {
        Some(v) if v == "lz4" => kvengine::table::LZ4_COMPRESSION,
        _ => kvengine::table::ZSTD_COMPRESSION,
    };

    if cluster_id.is_none() || commit_ts.is_none() {
        return Ok(make_response(
            StatusCode::BAD_REQUEST,
            "missing cluster_id or commit_ts".to_string(),
        ));
    }
    let cluster_id = cluster_id.unwrap();
    if cluster_id != ctx.cluster_id {
        return Ok(make_response(
            StatusCode::BAD_REQUEST,
            format!(
                "mismatched cluster_id: expected {}, got {}",
                ctx.cluster_id, cluster_id
            ),
        ));
    }
    let commit_ts = commit_ts.unwrap();

    let timeout = ctx.write_sst_manager.limiter.wait_timeout();
    let _permit =
        match tokio::time::timeout(timeout, ctx.compaction_limiter.acquire_permit(0)).await {
            Ok(permit) => permit,
            Err(_) => {
                return Ok(make_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "request wait timed out".to_string(),
                ));
            }
        };

    // TODO: support encryption
    let encryption_key: Option<cloud_encryption::EncryptionKey> = None;

    if method == http::Method::PUT {
        let file_id = match ctx.write_sst_manager.alloc_file_id().await {
            Ok(file_id) => file_id,
            Err(e) => {
                return Ok(make_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("failed to allocate file id: {}", e),
                ));
            }
        };
        let mut builder = Builder::new(
            file_id,
            ctx.block_size, // it's from `config.cop_block_size` which is the same as TiKV's
            compression,
            ctx.compression_lvl,
            ctx.checksum_type,
            encryption_key,
        );

        let mut entries = 0;
        let mut uncompressed_size = 0;
        let um = UserMeta::new(commit_ts, commit_ts);
        let mut val_buf = Value::encode_buf(0, &um.to_array(), commit_ts, &[]);
        let base_val_len = val_buf.len();

        let mut stream = req.into_body();
        let mut buffer = BytesMut::new();

        while let Some(chunk) = stream.next().await {
            // On client side, The chunk maybe truncated by the network layer, so chunk
            // could be incomplete. Stack the incomplete chunk to the buffer.
            buffer.extend_from_slice(&chunk?);

            // parse chunk: key_len(2), key, val_len(4), value
            loop {
                // At least 6 bytes are needed to parse key_len + val_len
                if buffer.len() < 6 {
                    break;
                }
                let key_len = (&buffer[..2]).get_u16_le() as usize;
                // Check if we have enough data for key + next 4 bytes (val_len)
                if buffer.len() < 2 + key_len + 4 {
                    break;
                }
                let key = &buffer[2..2 + key_len];
                let val_len = (&buffer[2 + key_len..2 + key_len + 4]).get_u32_le() as usize;
                let total_len = 2 + key_len + 4 + val_len;
                if buffer.len() < total_len {
                    break;
                }
                let val_start = 2 + key_len + 4;
                let val = &buffer[val_start..val_start + val_len];

                uncompressed_size += key_len + base_val_len + val_len;
                val_buf.resize(base_val_len, 0);
                val_buf.extend_from_slice(val);

                let inner_key = InnerKey::from_outer_key(key);
                if let Err(e) = builder.add(inner_key, &Value::decode(&val_buf), None) {
                    return Ok(make_response(
                        StatusCode::BAD_REQUEST,
                        format!("failed to add entry: {}", e),
                    ));
                }
                entries += 1;

                buffer.advance(total_len);
            }
        }
        if !buffer.is_empty() {
            // If there are remaining bytes in the buffer, we need to handle them.
            // This could be a partial entry or an error.
            return Ok(make_response(
                StatusCode::BAD_REQUEST,
                "incomplete chunk".to_string(),
            ));
        }
        if builder.is_empty() {
            return Ok(make_response(
                StatusCode::BAD_REQUEST,
                "empty chunk".to_string(),
            ));
        }
        let mut buf = vec![];
        let res = builder.finish(0, &mut buf);
        let data: Bytes = buf.into();
        let sst_meta = SstMeta {
            id: file_id,
            smallest: builder.get_smallest().to_vec(),
            biggest: builder.get_biggest().to_vec(),
            meta_offset: res.meta_offset,
            commit_ts,
            size: data.len(),
            uncompressed_size,
            keys: entries,
        };
        info!("finish building sst file {:?}", sst_meta);
        let opts = Options::default();
        // finish and return sst meta
        match ctx.s3fs.create(file_id, data, opts).await {
            Ok(_) => {
                let body = json!({"sst_meta": sst_meta}).to_string();
                Ok(Response::builder()
                    .status(200)
                    .header("Content-Type", "application/json")
                    .body(Body::from(body))
                    .unwrap())
            }
            Err(e) => Ok(make_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("failed to write sst file to s3: {}", e),
            )),
        }
    } else {
        Ok(make_response(
            StatusCode::METHOD_NOT_ALLOWED,
            "method not allowed".to_string(),
        ))
    }
}

/// Manages SST file writing sessions, including file ID allocation.
pub struct WriteSstManager {
    id_allocator: Arc<dyn kvengine::IdAllocator>,
    // Cache for pre-allocated file IDs obtained from PD TSO.
    cached_file_ids: Mutex<Vec<u64>>,
    pub limiter: WorkerLimiter,
}

impl WriteSstManager {
    /// Creates a new instance of SstWriteManager.
    pub fn new(pd_client: Arc<dyn PdClient>, limiter: WorkerLimiter) -> Self {
        WriteSstManager {
            id_allocator: Arc::new(PdIdAllocator::new(pd_client)),
            cached_file_ids: Mutex::new(Vec::new()),
            limiter,
        }
    }

    /// Allocates a unique file ID for a new SST file.
    ///
    /// It first attempts to use a cached ID. If the cache is empty,
    /// it requests a batch of timestamps (TSO) from PD to use as file IDs,
    /// caches them, and returns one. Retries on failure until a timeout.
    async fn alloc_file_id(&self) -> kvengine::Result<u64> {
        let mut cached_file_ids = self.cached_file_ids.lock().await;
        // Return a cached ID if available
        if let Some(id) = cached_file_ids.pop() {
            return Ok(id);
        }

        let count = 64; // Number of IDs to fetch per batch

        let ids = self.id_allocator.alloc_id_async(count).await?;
        cached_file_ids.extend(ids);
        Ok(cached_file_ids.pop().unwrap())
    }
}
