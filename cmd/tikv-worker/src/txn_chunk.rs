// Copyright 2024 TiKV Project Authors. Licensed under Apache-2.0.

use std::sync::Arc;

use bytes::Buf;
use http::{header, Request, Response, StatusCode};
use hyper::Body;
use kvengine::{dfs::Dfs, table::txn_file::TxnChunkBuilder};
use pd_client::PdClient;

use crate::{
    common::{get_body, make_response},
    server::Context,
};

pub(crate) async fn handle_txn_chunk(
    ctx: Arc<Context>,
    req: Request<Body>,
) -> hyper::Result<Response<Body>> {
    let chunk_id = match ctx.pd.get_tso().await {
        Ok(tso) => tso.into_inner(),
        Err(err) => {
            return Ok(make_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("failed to create txn chunk file {:?}", err),
            ));
        }
    };
    create_txn_chunk(chunk_id, ctx.s3fs.clone(), req).await
}

#[derive(Default, Serialize, Deserialize, Debug)]
#[serde(default)]
pub struct CreateTxnChunkResp {
    pub chunk_id: u64,
}

pub(crate) async fn create_txn_chunk(
    chunk_id: u64,
    dfs: Arc<dyn Dfs>,
    req: Request<Body>,
) -> hyper::Result<Response<Body>> {
    if *req.method() != http::Method::POST {
        return Ok(make_response(StatusCode::BAD_REQUEST, "invalid method"));
    }
    let body = get_body(req).await?;
    if body.len() < 4 {
        return Ok(make_response(StatusCode::BAD_REQUEST, "body is invalid"));
    }
    let data_len = body.len() - 4;
    let checksum = (&body[data_len..]).get_u32_le();
    let mut body_buf = &body[..data_len];
    if crc32c::crc32c(body_buf) != checksum {
        return Ok(make_response(StatusCode::BAD_REQUEST, "checksum mismatch"));
    }
    let mut txn_chunk_builder = TxnChunkBuilder::new(4096);
    while !body_buf.is_empty() {
        let key_len = body_buf.get_u16_le() as usize;
        let key = &body_buf[..key_len];
        body_buf.advance(key_len);
        let op = body_buf.get_u8();
        let val_len = body_buf.get_u32_le() as usize;
        let val = &body_buf[..val_len];
        body_buf.advance(val_len);
        txn_chunk_builder.add_entry(key, op, val);
    }
    drop(body);
    let mut txn_chunk_buf = vec![];
    txn_chunk_builder.finish(&mut txn_chunk_buf);
    if let Err(err) = dfs.create_txn_chunk(chunk_id, txn_chunk_buf.into()).await {
        return Ok(make_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("failed to create txn chunk file {:?}", err),
        ));
    }
    let resp = CreateTxnChunkResp { chunk_id };
    let json = serde_json::to_string(&resp).unwrap();
    Ok(Response::builder()
        .header(header::CONTENT_TYPE, "application/json")
        .body(json.into())
        .unwrap())
}

#[cfg(test)]
mod tests {
    use std::{ops::Deref, sync::Arc};

    use bytes::{Buf, BufMut};
    use futures::StreamExt;
    use http::Method;
    use kvengine::{
        dfs::{Dfs, InMemFs},
        table::{sstable::InMemFile, TxnChunk, TxnCtx, TxnFile, TxnFileId, TxnFileIterator},
        Iterator, UserMeta,
    };

    use crate::txn_chunk::{create_txn_chunk, CreateTxnChunkResp};

    #[test]
    fn test_create_txn_chunk() {
        let mut req_body = vec![];
        let buf = &mut req_body;
        for i in 0..100 {
            let key = format!("key{:03}", i);
            let val = format!("val{:03}", i);
            buf.put_u16_le(key.len() as u16);
            buf.extend_from_slice(key.as_bytes());
            buf.put_u8(1);
            buf.put_u32_le(val.len() as u32);
            buf.extend_from_slice(val.as_bytes());
        }
        let check_sum = crc32c::crc32c(buf);
        buf.put_u32_le(check_sum);
        let chunk_id = 155;
        let dfs: Arc<dyn Dfs> = Arc::new(InMemFs::new());
        let req = http::Request::builder()
            .method(Method::POST)
            .body(hyper::Body::from(req_body))
            .unwrap();
        let mut res = dfs
            .get_runtime()
            .block_on(create_txn_chunk(chunk_id, dfs.clone(), req))
            .unwrap();
        assert!(res.status().is_success());
        let body = dfs
            .get_runtime()
            .block_on(res.body_mut().next())
            .unwrap()
            .unwrap();
        let resp: CreateTxnChunkResp = serde_json::from_slice(body.chunk()).unwrap();
        assert_eq!(resp.chunk_id, 155);
        let chunk_data = dfs
            .get_runtime()
            .block_on(dfs.read_txn_chunk(chunk_id))
            .unwrap();
        assert!(!chunk_data.is_empty());
        let txn_chunk = TxnChunk::new(Arc::new(InMemFile::new(155, chunk_data)), None).unwrap();
        let user_meta = UserMeta::new(1, 2).to_array().to_vec();
        let txn_ctx = TxnCtx::new(user_meta.into(), vec![].into(), 2);
        let txn_file = TxnFile::new(TxnFileId::new(1, 1, 1), vec![txn_chunk], txn_ctx).unwrap();
        let mut iter = TxnFileIterator::new(txn_file, false);
        iter.rewind();
        let mut i = 0;
        while iter.valid() {
            assert_eq!(iter.key().deref(), format!("key{:03}", i).as_bytes());
            assert_eq!(iter.value().get_value(), format!("val{:03}", i).as_bytes());
            assert_eq!(iter.get_op(), 1);
            iter.next();
            i += 1;
        }
        assert_eq!(i, 100);
    }
}
