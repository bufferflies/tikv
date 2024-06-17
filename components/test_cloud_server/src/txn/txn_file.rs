// Copyright 2024 TiKV Project Authors. Licensed under Apache-2.0.

use core::slice::SlicePattern;
use std::{error::Error, fmt, mem, sync::Arc};

use api_version::ApiV2;
use bytes::{BufMut, Bytes, BytesMut};
use cloud_worker::CreateTxnChunkResp;
use hyper::Method;
use log_wrappers::Value;
use security::{RestfulClient, SecurityManager};

use crate::util::{Mutation, DEFAULT_INNER_KEY_OFFSET};

const TXN_ENTRY_OVERHEAD: usize =
    mem::size_of::<u16>() + mem::size_of::<u8>() + mem::size_of::<u32>();
// key_len + op + value_len
const TXN_CHUNK_OVERHEAD: usize = mem::size_of::<u32>(); // checksum

pub type Result<T> = std::result::Result<T, Box<dyn Error + Sync + Send>>;

#[derive(Clone)]
pub struct TxnFileHelper {
    max_chunk_size: usize,
    cli: RestfulClient,
}

impl TxnFileHelper {
    pub fn new(
        max_chunk_size: usize,
        tikv_worker_endpoints: Vec<String>,
        security_mgr: Arc<SecurityManager>,
    ) -> Result<Self> {
        Ok(TxnFileHelper {
            max_chunk_size,
            cli: RestfulClient::new("txn_file_helper", tikv_worker_endpoints, security_mgr)?,
        })
    }
}

impl TxnFileHelper {
    // `muts` should contain keyspace prefix.
    pub async fn build_txn_chunks(&self, muts: Vec<Mutation>) -> Result<Vec<TxnFileChunk>> {
        if muts.is_empty() {
            return Ok(vec![]);
        }

        let keyspace_id = ApiV2::get_u32_keyspace_id_by_key(&muts[0].key).unwrap();
        let total_size = muts
            .iter()
            .map(|m| m.key.len() - DEFAULT_INNER_KEY_OFFSET + m.value.len() + TXN_ENTRY_OVERHEAD)
            .sum::<usize>()
            + TXN_CHUNK_OVERHEAD;
        let mut buf = BytesMut::with_capacity(total_size.min(self.max_chunk_size));

        let mut outer_smallest = None;
        let mut chunks = vec![];

        for (i, m) in muts.iter().enumerate() {
            let key = m.key.slice(DEFAULT_INNER_KEY_OFFSET..);
            if !buf.is_empty()
                && buf.len() + key.len() + m.value.len() + TXN_ENTRY_OVERHEAD + TXN_CHUNK_OVERHEAD
                    > self.max_chunk_size
            {
                let buf_to_flush =
                    mem::replace(&mut buf, BytesMut::with_capacity(self.max_chunk_size));
                let chunk_id = self.flush_to_tikv_worker(keyspace_id, buf_to_flush).await?;
                chunks.push(TxnFileChunk {
                    chunk_id,
                    outer_smallest: outer_smallest.take().unwrap(),
                    outer_biggest: muts[i - 1].key.clone(),
                });
            }

            outer_smallest.get_or_insert_with(|| m.key.clone());

            buf.put_u16_le(key.len() as u16);
            buf.put(key);
            buf.put_u8(m.op as u8);
            buf.put_u32_le(m.value.len() as u32);
            buf.put(m.value.clone());
        }

        if !buf.is_empty() {
            let chunk_id = self.flush_to_tikv_worker(keyspace_id, buf).await?;
            chunks.push(TxnFileChunk {
                chunk_id,
                outer_smallest: outer_smallest.unwrap(),
                outer_biggest: muts.last().unwrap().key.clone(),
            });
        }

        Ok(chunks)
    }

    async fn flush_to_tikv_worker(
        &self,
        keyspace_id: u32,
        mut buf: BytesMut,
    ) -> Result<u64 /* chunk_id */> {
        let checksum = crc32fast::hash(buf.as_slice());
        buf.put_u32_le(checksum);

        let path = format!("txn_chunk?keyspace_id={keyspace_id}");
        let data = buf.freeze();
        let resp = self.cli.request(path, Method::POST, Some(data)).await?;
        let resp: CreateTxnChunkResp = serde_json::from_slice(&resp)?;
        Ok(resp.chunk_id)
    }
}

#[derive(Clone, PartialEq)]
pub struct TxnFileChunk {
    pub chunk_id: u64,
    pub outer_smallest: Bytes,
    pub outer_biggest: Bytes,
}

impl fmt::Debug for TxnFileChunk {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TxnFileChunk")
            .field("chunk_id", &self.chunk_id)
            .field("outer_smallest", &Value::key(&self.outer_smallest))
            .field("outer_biggest", &Value::key(&self.outer_biggest))
            .finish()
    }
}
