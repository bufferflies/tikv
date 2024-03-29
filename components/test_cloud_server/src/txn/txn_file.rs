// Copyright 2024 TiKV Project Authors. Licensed under Apache-2.0.

use core::slice::SlicePattern;
use std::{error::Error, fmt, mem, sync::Arc};

use bytes::{BufMut, Bytes, BytesMut};
use cloud_worker::CreateTxnChunkResp;
use hyper::Method;
use log_wrappers::Value;
use rfstore::store::RegionIdVer;
use security::{RestfulClient, SecurityManager};

use crate::util::{Mutation, RawRegion, DEFAULT_INNER_KEY_OFFSET};

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
                let chunk_id = self.flush_to_tikv_worker(buf_to_flush).await?;
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
            let chunk_id = self.flush_to_tikv_worker(buf).await?;
            chunks.push(TxnFileChunk {
                chunk_id,
                outer_smallest: outer_smallest.unwrap(),
                outer_biggest: muts.last().unwrap().key.clone(),
            });
        }

        Ok(chunks)
    }

    async fn flush_to_tikv_worker(&self, mut buf: BytesMut) -> Result<u64 /* chunk_id */> {
        let checksum = crc32c::crc32c(buf.as_slice());
        buf.put_u32_le(checksum);

        let data = buf.freeze();
        let resp = self
            .cli
            .request("txn_chunk", Method::POST, Some(data))
            .await?;
        let resp: CreateTxnChunkResp = serde_json::from_slice(&resp)?;
        Ok(resp.chunk_id)
    }

    // `chunks` & `regions` should be sorted and no overlapping.
    // The returned chunks should keep the order of input chunks.
    pub fn group_txn_chunks_by_regions(
        chunks: &[TxnFileChunk],
        regions: Vec<RawRegion>,
    ) -> Vec<(RegionIdVer, Vec<TxnFileChunk>)> {
        assert!(!chunks.is_empty());
        assert!(!regions.is_empty());
        assert!(chunks.first().unwrap().outer_smallest >= regions.first().unwrap().raw_start());
        assert!(chunks.last().unwrap().outer_biggest < regions.last().unwrap().raw_end());

        let mut chunks_by_regions = vec![];
        let mut chunks_idx = 0;
        for region in regions {
            let mut region_chunks = vec![];
            if chunks_idx > 0 && chunks[chunks_idx - 1].outer_biggest >= region.raw_start() {
                chunks_idx -= 1;
            }
            while chunks_idx < chunks.len() && chunks[chunks_idx].outer_smallest < region.raw_end()
            {
                region_chunks.push(chunks[chunks_idx].clone());
                chunks_idx += 1;
            }
            if !region_chunks.is_empty() {
                let region_id_ver = region.id_ver();
                chunks_by_regions.push((region_id_ver, region_chunks));
            }
        }

        debug_assert_eq!(
            chunks[0].chunk_id, chunks_by_regions[0].1[0].chunk_id,
            "group_txn_chunks_by_regions should keep the order"
        );
        chunks_by_regions
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

#[cfg(test)]
mod tests {
    use bytes::Bytes;

    use super::*;

    #[test]
    fn test_group_txn_chunks_by_regions() {
        let generate_chunk = |id: u64, smallest: u32, biggest: u32| TxnFileChunk {
            chunk_id: id,
            outer_smallest: Bytes::from(format!("{:04}", smallest)),
            outer_biggest: Bytes::from(format!("{:04}", biggest)),
        };

        let generate_region = |id: u64, start: u32, end: u32| {
            RawRegion::new_for_test(
                id,
                0,
                format!("{:04}", start).into_bytes(),
                format!("{:04}", end).into_bytes(),
            )
        };

        let test_cases = vec![{
            let chunks = vec![
                generate_chunk(0, 2, 3),
                generate_chunk(1, 4, 5),
                generate_chunk(2, 6, 8),
                generate_chunk(3, 20, 21),
                generate_chunk(4, 21, 22),
                generate_chunk(5, 22, 23),
            ];

            let regions_and_expected = vec![
                (generate_region(100, 0, 2), vec![]),
                (generate_region(101, 2, 4), vec![chunks[0].clone()]),
                (generate_region(102, 4, 5), vec![chunks[1].clone()]),
                (generate_region(103, 5, 6), vec![chunks[1].clone()]),
                (generate_region(104, 6, 7), vec![chunks[2].clone()]),
                (generate_region(105, 7, 9), vec![chunks[2].clone()]),
                (generate_region(106, 9, 12), vec![]),
                (
                    generate_region(107, 12, 30),
                    vec![chunks[3].clone(), chunks[4].clone(), chunks[5].clone()],
                ),
            ];

            (chunks, regions_and_expected)
        }];

        for (chunks, regions_and_expected) in test_cases {
            let regions = regions_and_expected
                .iter()
                .map(|x| x.0.clone())
                .collect::<Vec<_>>();
            let result = TxnFileHelper::group_txn_chunks_by_regions(&chunks, regions.clone());

            let expected = regions_and_expected
                .into_iter()
                .filter_map(|(r, c)| (!c.is_empty()).then_some((r.id_ver(), c)))
                .collect::<Vec<_>>();
            assert_eq!(
                result, expected,
                "chunks {:?}, regions {:?}",
                chunks, regions
            );
        }
    }
}
