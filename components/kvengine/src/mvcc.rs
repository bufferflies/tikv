// Copyright 2021 TiKV Project Authors. Licensed under Apache-2.0.

use byteorder::{ByteOrder, LittleEndian};
use bytes::Bytes;
use tikv_util::codec::number::NumberEncoder;

pub const WRITE_CF: usize = 0;
pub const LOCK_CF: usize = 1;
pub const EXTRA_CF: usize = 2;

pub const USER_META_FORMAT_V1: u8 = 1;

// format(1) + start_ts(8) + commit_ts(8)
pub const USER_META_SIZE: usize = 1 + std::mem::size_of::<UserMeta>();

#[derive(Clone, Copy, Serialize, Deserialize, Debug)]
pub struct UserMeta {
    pub start_ts: u64,
    pub commit_ts: u64,
}

impl UserMeta {
    pub fn from_slice(buf: &[u8]) -> Self {
        assert_eq!(buf[0], USER_META_FORMAT_V1);
        Self {
            start_ts: LittleEndian::read_u64(&buf[1..]),
            commit_ts: LittleEndian::read_u64(&buf[9..]),
        }
    }

    pub fn new(start_ts: u64, commit_ts: u64) -> Self {
        Self {
            start_ts,
            commit_ts,
        }
    }

    pub fn to_array(&self) -> [u8; USER_META_SIZE] {
        let mut array = [0u8; USER_META_SIZE];
        array[0] = USER_META_FORMAT_V1;
        LittleEndian::write_u64(&mut array[1..], self.start_ts);
        LittleEndian::write_u64(&mut array[9..], self.commit_ts);
        array
    }

    pub fn is_rollback(&self) -> bool {
        self.commit_ts == 0
    }
}

pub fn encode_extra_txn_status_key(key: &[u8], start_ts: u64) -> Bytes {
    // Attention: make the encoding format of txn key consistent with the
    // implementation in TiKV, the format should be `[raw_key +
    // <BigEndian>::(!ts)]`.
    //
    // Ref: https://github.com/tikv/tikv/blob/1deb3a135dc41c3ca227e3d5a29712526b492a4c/components/tikv_util/src/codec/number.rs#L73.
    let mut buf = vec![];
    buf.extend_from_slice(key);
    buf.encode_u64_desc(start_ts).unwrap();
    buf.into()
}

#[cfg(test)]
mod tests {

    use tikv_util::{codec::number::decode_u64_desc, time::Instant};

    use super::*;

    #[test]
    fn test_extra_txn_status_key_encoding() {
        let raw_key = b"test_tidb_123";

        let ts = Instant::now();
        let encoded_key = encode_extra_txn_status_key(raw_key, ts.second() as u64);
        let len = encoded_key.len();
        let mut decoded_key = &encoded_key[(len - 8)..];
        let decoded_ts = decode_u64_desc(&mut decoded_key).unwrap();
        assert_eq!(decoded_ts, ts.second() as u64);
    }
}
