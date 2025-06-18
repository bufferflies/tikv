// Copyright 2024 TiKV Project Authors. Licensed under Apache-2.0.

use std::{fmt, mem, str::FromStr, time::Duration};

use api_version::ApiV2;
use bytes::Bytes;
use codec::number::NumberEncoder;
use hyper::{http, Body, Request, Uri};
use kvengine::table::columnar::{
    new_int_handle_column_info, new_version_column_info, Schema, SchemaBuf,
};
use kvproto::{
    kvrpcpb, metapb,
    metapb::{Peer, RegionEpoch, Store},
};
use log_wrappers::Value;
use pd_client::PdClient;
use rfstore::store::RegionIdVer;
use schema::schema::StorageClass;
use test_pd_client::TestPdClient;
use tidb_query_datatype::{codec::table::TABLE_PREFIX, Collation, FieldTypeTp};
use tikv::storage::mvcc::Key;
use tikv_util::{
    codec::bytes::{decode_bytes, encode_bytes},
    time::Instant,
};
use tipb::ColumnInfo;
use tokio::runtime::Runtime;

pub(crate) const DEFAULT_INNER_KEY_OFFSET: usize = 4;

/// A cheaply cloneable version of `kvrpcpb::Mutation`.
#[derive(Default, Clone)]
pub struct Mutation {
    pub op: kvrpcpb::Op,
    pub key: Bytes,
    pub value: Bytes,
    pub assertion: kvrpcpb::Assertion,
}

impl fmt::Debug for Mutation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Mutation")
            .field("op", &self.op)
            .field("key", &Value::key(&self.key))
            .field("value", &Value::value(&self.value))
            .finish()
    }
}

impl From<&Mutation> for kvrpcpb::Mutation {
    fn from(m: &Mutation) -> Self {
        kvrpcpb::Mutation {
            op: m.op,
            key: m.key.to_vec(),
            value: m.value.to_vec(),
            assertion: m.assertion,
            ..Default::default()
        }
    }
}

impl Mutation {
    pub fn get_op(&self) -> kvrpcpb::Op {
        self.op
    }

    pub fn set_op(&mut self, op: kvrpcpb::Op) {
        self.op = op;
    }

    pub fn get_key(&self) -> &[u8] {
        &self.key
    }

    pub fn set_key(&mut self, key: Vec<u8>) {
        self.key = key.into();
    }

    pub fn get_value(&self) -> &[u8] {
        &self.value
    }

    pub fn set_value(&mut self, value: Vec<u8>) {
        self.value = value.into();
    }

    pub fn take_key(&mut self) -> Vec<u8> {
        mem::take(&mut self.key).into()
    }

    pub fn take_value(&mut self) -> Vec<u8> {
        mem::take(&mut self.value).into()
    }

    pub fn get_assertion(&self) -> kvrpcpb::Assertion {
        self.assertion
    }

    pub fn set_assertion(&mut self, assertion: kvrpcpb::Assertion) {
        self.assertion = assertion;
    }
}

#[derive(Clone)]
pub struct RawRegion {
    pub id: u64,
    pub raw_start: Vec<u8>,
    pub raw_end: Vec<u8>,
    pub epoch: RegionEpoch,
    pub peers: Vec<Peer>,
    pub leader_idx: usize,
}

impl fmt::Debug for RawRegion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RawRegion")
            .field("id", &self.id)
            .field("raw_start", &Value::key(&self.raw_start))
            .field("raw_end", &Value::key(&self.raw_end))
            .field("epoch", &self.epoch)
            .field("peers", &self.peers)
            .field("leader_idx", &self.leader_idx)
            .finish()
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
            leader_idx: 0,
        }
    }
}

impl RawRegion {
    pub fn get_leader(&self) -> &Peer {
        &self.peers[self.leader_idx]
    }

    pub fn id_ver(&self) -> RegionIdVer {
        RegionIdVer::new(self.id, self.epoch.version)
    }

    pub fn update_leader(&mut self, leader: &Peer) -> bool {
        if let Some(idx) = self.peers.iter().position(|p| p.id == leader.id) {
            self.leader_idx = idx;
            true
        } else {
            false
        }
    }

    pub fn raw_start(&self) -> &[u8] {
        &self.raw_start
    }

    pub fn raw_end(&self) -> &[u8] {
        &self.raw_end
    }

    pub fn equal(&self, other: &Self) -> bool {
        self.id == other.id
            && self.epoch == other.epoch
            && self.get_leader().id == other.get_leader().id
    }
}

impl pd_client::util::RegionLike for RawRegion {
    const KEY_ENCODED: bool = false;

    fn id(&self) -> u64 {
        self.id
    }

    fn epoch(&self) -> &metapb::RegionEpoch {
        &self.epoch
    }

    fn start_key(&self) -> &[u8] {
        &self.raw_start
    }

    fn end_key(&self) -> &[u8] {
        &self.raw_end
    }
}

#[derive(Default)]
pub struct TableSchemaOptions {
    pub table_id: i64,
    pub with_columns: bool,
    pub storage_class: StorageClass,
}

pub fn build_schemas(tables: &[TableSchemaOptions]) -> Vec<Schema> {
    let mut schemas = vec![];
    for opts in tables {
        let mut schema = SchemaBuf {
            table_id: opts.table_id,
            ..Default::default()
        };
        if opts.with_columns {
            let mut c1 = ColumnInfo::new();
            c1.set_column_id(1);
            c1.set_tp(FieldTypeTp::LongLong.to_u8().unwrap() as i32);
            let mut c2 = ColumnInfo::new();
            c2.set_column_id(2);
            c2.set_tp(FieldTypeTp::VarChar.to_u8().unwrap() as i32);
            c2.set_column_len(255);
            c2.set_collation(Collation::Utf8Mb4Bin as i32);

            schema = SchemaBuf::new(
                opts.table_id,
                new_int_handle_column_info(),
                new_version_column_info(),
                vec![c1, c2],
                vec![],
                vec![],
                StorageClass::default(),
                None,
            )
        }

        if opts.storage_class.is_specified() {
            schema.set_storage_class(opts.storage_class);
        }

        schemas.push(schema.into());
    }
    schemas
}

pub fn get_keyspace_split_keys(keyspace_id: u32) -> Vec<Vec<u8>> {
    vec![
        ApiV2::get_keyspace_prefix_by_id(keyspace_id),
        ApiV2::get_keyspace_prefix_by_id(keyspace_id + 1),
    ]
    .into_iter()
    .filter_map(|k| (!k.is_empty()).then(|| Key::from_raw(&k).into_encoded()))
    .collect()
}

pub fn get_table_split_keys(keyspace_id: u32, table_ids: &[i64]) -> Vec<Vec<u8>> {
    let keyspace_prefix = ApiV2::get_keyspace_prefix_by_id(keyspace_id);
    let mut dup_table_ids = table_ids
        .iter()
        .flat_map(|&id| [id, id + 1])
        .collect::<Vec<_>>();
    dup_table_ids.sort();
    dup_table_ids.dedup();
    dup_table_ids
        .into_iter()
        .map(|tbl_id| {
            let mut buf = vec![];
            buf.extend_from_slice(&keyspace_prefix);
            buf.extend_from_slice(TABLE_PREFIX);
            buf.write_i64(tbl_id).unwrap();
            encode_bytes(&buf)
        })
        .collect()
}

pub async fn broadcast_schema_file_request(
    stores: &[Store],
    keyspace_id: u32,
    schema_file_id: u64,
    timeout: Duration,
) {
    for store in stores {
        let status_addr = store.get_status_address();
        let http_client = hyper::client::Client::new();
        let start_time = Instant::now_coarse();
        while start_time.saturating_elapsed() < timeout {
            let request = hyper::http::Request::builder()
                .method(http::method::Method::POST)
                .uri(format!(
                    "http://{}/schema_file?keyspace_id={}&file_id={}",
                    status_addr, keyspace_id, schema_file_id
                ))
                .body(Body::empty())
                .unwrap();
            if let Ok(resp) = http_client.request(request).await {
                if resp.status().is_success() {
                    break;
                }
            }
            tokio::time::sleep(Duration::from_millis(500)).await;
        }
    }
}

pub async fn request_major_compact_on_store(store: &Store, query: &str, permit_not_found: bool) {
    let uri = Uri::from_str(&format!(
        "http://{}/major-compact?{}",
        &store.status_address, query
    ))
    .unwrap();
    let req = Request::post(uri).body(Body::empty()).unwrap();
    let client = hyper::Client::new();
    let resp: http::Response<Body> = client.request(req).await.unwrap();
    let is_success = resp.status().is_success()
        || (permit_not_found && resp.status() == http::StatusCode::NOT_FOUND);
    assert!(
        is_success,
        "{:?}",
        hyper::body::to_bytes(resp.into_body()).await.unwrap()
    );
    hyper::body::to_bytes(resp.into_body()).await.unwrap();
}

pub fn request_major_compaction(runtime: &Runtime, pd_client: &TestPdClient, keyspace_id: u32) {
    let stores = pd_client.get_all_stores(true).unwrap();
    let mut handles = Vec::with_capacity(stores.len());
    for store in stores {
        handles.push(runtime.spawn(async move {
            let query = format!("major_compact=true&keyspace_id={}", keyspace_id);
            request_major_compact_on_store(&store, query.as_str(), true).await;
        }));
    }
    for handle in handles {
        runtime.block_on(handle).unwrap();
    }
}
