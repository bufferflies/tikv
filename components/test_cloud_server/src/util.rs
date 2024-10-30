// Copyright 2024 TiKV Project Authors. Licensed under Apache-2.0.

use std::{fmt, mem};

use bytes::Bytes;
use kvengine::table::columnar::{
    new_int_handle_column_info, new_version_column_info, Schema, SchemaBuf,
};
use kvproto::{
    kvrpcpb, metapb,
    metapb::{Peer, RegionEpoch},
};
use log_wrappers::Value;
use rfstore::store::RegionIdVer;
use tidb_query_datatype::{Collation, FieldTypeTp};
use tikv_util::codec::bytes::decode_bytes;
use tipb::ColumnInfo;
pub(crate) const DEFAULT_INNER_KEY_OFFSET: usize = 4;

/// A cheaply cloneable version of `kvrpcpb::Mutation`.
#[derive(Default, Clone)]
pub struct Mutation {
    pub op: kvrpcpb::Op,
    pub key: Bytes,
    pub value: Bytes,
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

pub fn build_schemas(table_ids: Vec<i64>) -> Vec<Schema> {
    let mut schemas = vec![];
    for &columnar_table_id in &table_ids {
        let mut c1 = ColumnInfo::new();
        c1.set_column_id(1);
        c1.set_tp(FieldTypeTp::LongLong.to_u8().unwrap() as i32);
        let mut c2 = ColumnInfo::new();
        c2.set_column_id(2);
        c2.set_tp(FieldTypeTp::VarChar.to_u8().unwrap() as i32);
        c2.set_column_len(255);
        c2.set_collation(Collation::Utf8Mb4Bin as i32);
        let schema = SchemaBuf {
            table_id: columnar_table_id,
            handle_column: new_int_handle_column_info(),
            version_column: new_version_column_info(),
            txn_id_column: None,
            columns: vec![c1, c2],
            pk_col_ids: vec![],
            vector_indexes: vec![],
        }
        .into();
        schemas.push(schema);
    }
    schemas
}
