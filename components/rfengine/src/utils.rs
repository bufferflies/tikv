// Copyright 2021 TiKV Project Authors. Licensed under Apache-2.0.

use api_version::{
    api_v2::{self, KEYSPACE_ID_LEN},
    ApiV2, KeyMode, KvFormat,
};
use bytes::{BufMut, Bytes, BytesMut};
use kvproto::metapb;

pub(crate) const RAFT_STATE_KEY_BYTE: u8 = 1;
pub const REGION_META_KEY_BYTE: u8 = 2;
pub const REGION_META_KEY_PREFIX: &[u8] = &[REGION_META_KEY_BYTE];
pub const STORE_IDENT_KEY: &[u8] = &[3];
pub const PREPARE_BOOTSTRAP_KEY: &[u8] = &[4];
pub const KV_ENGINE_META_KEY: &[u8] = &[5];
pub const RAFT_TRUNCATED_STATE_KEY: &[u8] = &[6];

pub fn raft_state_key(version: u64) -> Bytes {
    let mut key = BytesMut::with_capacity(5);
    key.put_u8(RAFT_STATE_KEY_BYTE);
    key.put_u32(version as u32);
    key.freeze()
}

pub fn region_state_key(version: u64) -> Bytes {
    let mut key = BytesMut::with_capacity(5);
    key.put_u8(REGION_META_KEY_BYTE);
    key.put_u32(version as u32);
    key.freeze()
}

fn is_api_v2_region(region: &metapb::Region) -> bool {
    let startkey = region.start_key.as_slice();
    let endkey = region.end_key.as_slice();

    let start_key_mode = ApiV2::parse_key_mode(startkey);
    let end_key_mode = ApiV2::parse_key_mode(endkey);

    (start_key_mode == KeyMode::Raw || start_key_mode == KeyMode::Txn)
        && (end_key_mode == KeyMode::Raw || end_key_mode == KeyMode::Txn)
}

pub fn get_region_keyspace_id_str(region: &metapb::Region) -> Option<String> {
    if is_api_v2_region(region) {
        let keyspace_id_str = ApiV2::get_keyspace_id_str(region.start_key.as_slice());
        return Some(keyspace_id_str);
    }
    None
}

pub(crate) fn get_region_keyspace_id(region: &metapb::Region) -> [u8; KEYSPACE_ID_LEN] {
    if is_api_v2_region(region) {
        ApiV2::get_keyspace_id(region.start_key.as_slice())
    } else {
        api_v2::UNKNOWN_KEYSPACE_ID
    }
}

#[cfg(test)]
pub mod tests {
    use api_version::{
        api_v2::{self, TXN_KEY_PREFIX},
        ApiV2,
    };
    use byteorder::{BigEndian, ByteOrder};
    use kvproto::metapb::Region;

    use crate::{get_region_keyspace_id, get_region_keyspace_id_str};

    #[test]
    fn test_get_region_keyspace_id() {
        let keyspace_id = 1;
        let startkey = get_txn_startkey_prefix(keyspace_id);
        let endkey = get_txn_endkey_prefix(keyspace_id);

        let mut region = Region {
            id: keyspace_id as u64,
            start_key: startkey.to_vec(),
            end_key: endkey.to_vec(),
            ..Default::default()
        };

        assert_eq!(
            get_region_keyspace_id_str(&region).unwrap(),
            keyspace_id.to_string()
        );
        assert_eq!(
            get_region_keyspace_id(&region),
            ApiV2::get_keyspace_id(&keyspace_id.to_be_bytes())
        );
        region.start_key = vec![];
        assert!(get_region_keyspace_id_str(&region).is_none());
        assert_eq!(get_region_keyspace_id(&region), api_v2::UNKNOWN_KEYSPACE_ID);
    }

    pub fn get_txn_startkey_prefix(keyspace_id: u32) -> [u8; 4] {
        let mut keyspace_id_buf = [0u8; 4];
        BigEndian::write_u32(&mut keyspace_id_buf, keyspace_id);
        keyspace_id_buf[0] = TXN_KEY_PREFIX;
        keyspace_id_buf
    }

    pub fn get_txn_endkey_prefix(keyspace_id: u32) -> [u8; 4] {
        let mut keyspace_id_buf = get_txn_startkey_prefix(keyspace_id);
        keyspace_id_buf[3] += 1;
        keyspace_id_buf
    }
}
