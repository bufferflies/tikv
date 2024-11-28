// Copyright 2021 TiKV Project Authors. Licensed under Apache-2.0.

use std::path::{Path, PathBuf};

use api_version::{
    api_v2::{self, KEYSPACE_ID_LEN},
    ApiV2, KeyMode, KvFormat,
};
use bytes::{BufMut, Bytes, BytesMut};
use kvproto::metapb;
use regex::Regex;

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

pub fn compress_lz4(uncompressed: &[u8], compressed_buf: &mut Vec<u8>) -> std::io::Result<usize> {
    let compress_bound: i32 = unsafe { lz4::liblz4::LZ4_compressBound(uncompressed.len() as i32) };
    let existed_bytes = compressed_buf.len();
    compressed_buf.resize(existed_bytes + 4 + compress_bound as usize, 0);
    let size = lz4::block::compress_to_buffer(
        uncompressed,
        None,
        true,
        &mut compressed_buf[existed_bytes..],
    )?;
    compressed_buf.truncate(existed_bytes + size);
    Ok(existed_bytes + size)
}

pub fn decompress_lz4(content: &[u8]) -> std::io::Result<Vec<u8>> {
    lz4::block::decompress(content, None)
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

pub(crate) fn raft_log_file_name(dir: &Path, peer_id: u64, first: u64, last: u64) -> PathBuf {
    dir.join(format!(
        "{:016x}_{:016x}_{:016x}.rlog",
        peer_id, first, last,
    ))
}

pub(crate) fn store_raft_log_file_key(store_id: u64, epoch: u32) -> String {
    format!("{:016x}/r{:016x}.rlog", store_id, epoch)
}

pub fn wal_file_key(store_id: u64, epoch_id: u32, start_off: u64, end_off: u64) -> String {
    format!(
        "{:016x}/e{:08x}/{:016x}_{:016x}.wal",
        store_id, epoch_id, start_off, end_off
    )
}

pub fn snapshot_store_meta_key(store_id: u64, epoch: u32) -> String {
    format!(
        "store_backup/{:016x}/snapshots/m{:08x}.meta",
        store_id, epoch
    )
}

pub fn snapshot_rlog_key(store_id: u64, epoch: u32) -> String {
    format!(
        "store_backup/{:016x}/snapshots/r{:08x}.rlog",
        store_id, epoch
    )
}

pub(crate) fn snapshot_rlog_key_suffix(epoch: u32) -> String {
    format!("{:08x}.rlog", epoch)
}

pub(crate) fn snapshot_rlog_key_prefix(store_id: u64) -> String {
    format!("store_backup/{:016x}/snapshots/r", store_id)
}

pub fn parse_epoch_from_snapshot_key(key: Option<&str>) -> Option<u32> {
    key.and_then(|key| {
        let re = Regex::new(r"r([0-9a-fA-F]+)\.rlog").unwrap();
        if let Some(captures) = re.captures(key) {
            if let Some(epoch_hex) = captures.get(1) {
                let epoch = epoch_hex.as_str();
                return Some(u32::from_str_radix(epoch, 16).unwrap());
            }
        }
        None
    })
}

pub fn parse_wal_chunk_key(key: Option<&str>) -> Option<(u32, u64, u64, bool /* is last chunk */)> {
    key.and_then(|key| {
        let re = Regex::new(r"e([0-9a-fA-F]+)_([0-9a-fA-F]+)_([0-9a-fA-F]+)\.wal").unwrap();
        if let Some(captures) = re.captures(key) {
            let epoch_hex = captures.get(1).unwrap().as_str();
            let start_off_hex = captures.get(2).unwrap().as_str();
            let end_off_hex = captures.get(3).unwrap().as_str();
            let epoch = u32::from_str_radix(epoch_hex, 16).unwrap();
            let start_off = u64::from_str_radix(start_off_hex, 16).unwrap();
            let end_off = u64::from_str_radix(end_off_hex, 16).unwrap();
            return Some((epoch, start_off, end_off, key.ends_with(".last")));
        }
        None
    })
}

pub fn verify_wal_chunks_integrity(
    chunks: &[String],
    check_last: bool,
) -> std::result::Result<u64 /* last_end_off */, String> {
    if chunks.is_empty() {
        // `check_last` is true, it means the chunks belongs to a previous epoch, empty
        // chunks is invalid. `check_last` is false, it means the chunks belongs
        // to the current epoch, it is valid before the first chunk put to S3.
        return if check_last {
            Err("no chunk".to_string())
        } else {
            Ok(0)
        };
    }

    let wal_epoch;
    let mut last_end_off;
    let mut has_last_chunk;

    let first_chunk = chunks.first().unwrap();
    if let Some((epoch, start_off, end_off, last)) = parse_wal_chunk_key(Some(first_chunk)) {
        if start_off != 0 {
            return Err("miss first chunk".to_string());
        }
        wal_epoch = epoch;
        last_end_off = end_off;
        has_last_chunk = last;
    } else {
        return Err(format!("invalid pattern: {first_chunk}"));
    }

    for chunk in &chunks[1..] {
        if let Some((epoch, start_off, end_off, last)) = parse_wal_chunk_key(Some(chunk)) {
            if epoch != wal_epoch {
                return Err(format!("epoch mismatch: {epoch} != {wal_epoch}"));
            }
            if start_off != last_end_off {
                return Err(format!("miss chunk: offset {last_end_off}"));
            }
            last_end_off = end_off;
            has_last_chunk = last;
        } else {
            return Err(format!("invalid pattern: {chunk}"));
        }
    }
    if check_last && !has_last_chunk {
        Err("miss last chunk".to_string())
    } else {
        Ok(last_end_off)
    }
}

pub fn wal_chunk_file_key(store_id: u64, epoch_id: u32, start_off: u64, end_off: u64) -> String {
    wal_chunk_file_key_with_suffix(store_id, epoch_id, start_off, end_off, false)
}

pub fn last_wal_chunk_file_key(
    store_id: u64,
    epoch_id: u32,
    start_off: u64,
    end_off: u64,
) -> String {
    wal_chunk_file_key_with_suffix(store_id, epoch_id, start_off, end_off, true)
}

fn wal_chunk_file_key_with_suffix(
    store_id: u64,
    epoch_id: u32,
    start_off: u64,
    end_off: u64,
    last: bool,
) -> String {
    let suffix = if last { ".last" } else { "" };
    format!(
        "store_backup/{:016x}/wal_chunks/e{:08x}_{:016x}_{:016x}.wal{}",
        store_id, epoch_id, start_off, end_off, suffix,
    )
}

pub fn wal_chunk_file_prefix(store_id: u64, epoch_id: u32) -> String {
    format!(
        "store_backup/{:016x}/wal_chunks/e{:08x}_",
        store_id, epoch_id
    )
}

pub fn wal_chunk_file_suffix(start_off: u64, end_off: u64) -> String {
    format!("{:016x}_{:016x}.wal", start_off, end_off)
}

#[cfg(test)]
pub mod test_util {
    use std::{sync::Once, time::Duration};

    use api_version::api_v2::TXN_KEY_PREFIX;
    use byteorder::{BigEndian, ByteOrder};
    use bytes::{BufMut, BytesMut};
    use kvproto::raft_serverpb::RegionLocalState;
    use protobuf::Message;
    use raft_proto::{eraftpb, eraftpb::EntryType};
    use tikv_util::time::Instant;

    use crate::region_state_key;

    static INIT: Once = Once::new();

    pub fn init_logger() {
        INIT.call_once(test_util::init_log_for_test);
    }

    #[must_use]
    pub fn try_wait<F>(f: F, seconds: usize) -> bool
    where
        F: Fn() -> bool,
    {
        let begin = Instant::now_coarse();
        let timeout = Duration::from_secs(seconds as u64);
        while begin.saturating_elapsed() < timeout {
            if f() {
                return true;
            }
            std::thread::sleep(Duration::from_millis(100))
        }
        false
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

    pub fn make_log_data(index: u64, size: usize) -> eraftpb::Entry {
        let mut entry = eraftpb::Entry::new();
        entry.set_entry_type(eraftpb::EntryType::EntryConfChange);
        entry.set_index(index);
        entry.set_term(1);

        let mut data = BytesMut::with_capacity(size);
        data.resize(size, 0);
        entry.set_data(data.freeze());
        entry
    }

    pub fn make_state_kv(key_byte: u8, idx: u64) -> (BytesMut, BytesMut) {
        let mut key = BytesMut::new();
        key.put_u8(key_byte);
        let mut val = BytesMut::new();
        val.put_u64_le(idx);
        (key, val)
    }

    pub fn make_region_state(region_epoch: u64, keyspace_id: u32) -> (Vec<u8>, Vec<u8>) {
        let key = region_state_key(region_epoch).to_vec();

        let mut local_stat = RegionLocalState::default();
        local_stat.mut_region().start_key = get_txn_startkey_prefix(keyspace_id).to_vec();
        local_stat.mut_region().end_key = get_txn_endkey_prefix(keyspace_id).to_vec();
        let val = local_stat.write_to_bytes().unwrap();

        (key, val)
    }

    pub fn new_raft_entry(
        tp: EntryType,
        term: u64,
        index: u64,
        data: &[u8],
        context: u8,
    ) -> eraftpb::Entry {
        let mut entry = eraftpb::Entry::new();
        entry.set_entry_type(tp);
        entry.set_term(term);
        entry.set_index(index);
        entry.set_data(data.to_vec().into());
        if context > 0 {
            entry.set_context(vec![context].into());
        }
        entry
    }
}

#[cfg(test)]
mod tests {
    use api_version::{
        api_v2::{self},
        ApiV2,
    };
    use kvproto::metapb::Region;

    use crate::{
        get_region_keyspace_id, get_region_keyspace_id_str, last_wal_chunk_file_key,
        parse_epoch_from_snapshot_key, parse_wal_chunk_key, snapshot_rlog_key,
        test_util::{get_txn_endkey_prefix, get_txn_startkey_prefix},
        verify_wal_chunks_integrity, wal_chunk_file_key,
    };

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

    #[test]
    fn test_parse_snapshot_key() {
        let cases = vec![
            (snapshot_rlog_key(1, 2), Some(2)),
            (snapshot_rlog_key(1, 1000), Some(1000)),
            ("xxxxx".to_string(), None),
            ("".to_string(), None),
        ];
        for case in cases {
            assert_eq!(parse_epoch_from_snapshot_key(Some(case.0.as_str())), case.1);
        }
        assert_eq!(parse_epoch_from_snapshot_key(None), None);
    }

    #[test]
    fn test_parse_wal_chunk_key() {
        let cases = vec![
            (wal_chunk_file_key(1, 1, 10, 20), Some((1, 10, 20, false))),
            (
                wal_chunk_file_key(1, 1000, 10, 20),
                Some((1000, 10, 20, false)),
            ),
            (
                last_wal_chunk_file_key(1, 1000, 1000, 2000),
                Some((1000, 1000, 2000, true)),
            ),
            ("xxxxx".to_string(), None),
            ("".to_string(), None),
        ];
        for case in cases {
            assert_eq!(parse_wal_chunk_key(Some(case.0.as_str())), case.1);
        }
        assert_eq!(parse_wal_chunk_key(None), None);
    }

    #[test]
    fn test_lz4_compress() {
        let data = b"hello world".repeat(100);
        let mut compressed = Vec::new();
        let size = super::compress_lz4(&data, &mut compressed).unwrap();
        assert!(size < data.len());
        let decompressed = super::decompress_lz4(&compressed).unwrap();
        assert_eq!(data, decompressed);

        let mut compressed_with_data = b"header".to_vec();
        let _ = super::compress_lz4(&data, &mut compressed_with_data).unwrap();
        assert_eq!(compressed_with_data.len(), size + 6);
        assert_eq!(&compressed_with_data[0..6], b"header".as_slice());
        assert_eq!(&compressed_with_data[6..], compressed.as_slice());
    }

    #[test]
    fn test_verify_wal_chunks_integrity() {
        let cases: Vec<(
            Vec<String>,                  // keys
            bool,                         // check_last
            std::result::Result<u64, ()>, // expected
        )> = vec![
            (vec![], false, Ok(0)),                                   // empty
            (vec![wal_chunk_file_key(1, 1, 0, 100)], false, Ok(100)), // only has one chunk
            (vec![wal_chunk_file_key(1, 1, 1, 100)], false, Err(())), // start_off is not 0
            (
                // start_off is not 0
                vec![
                    wal_chunk_file_key(1, 1, 1, 100),
                    wal_chunk_file_key(1, 1, 100, 200),
                ],
                false,
                Err(()),
            ),
            (
                // offset is not continuous
                vec![
                    wal_chunk_file_key(1, 1, 0, 100),
                    wal_chunk_file_key(1, 1, 150, 200),
                    wal_chunk_file_key(1, 1, 200, 300),
                ],
                false,
                Err(()),
            ),
            (
                // epoch is not consistent
                vec![
                    wal_chunk_file_key(1, 1, 0, 100),
                    wal_chunk_file_key(1, 1, 100, 200),
                    wal_chunk_file_key(1, 2, 200, 300),
                ],
                false,
                Err(()),
            ),
            (
                // last chunk is not last, ignore the last check
                vec![
                    wal_chunk_file_key(1, 1, 0, 100),
                    wal_chunk_file_key(1, 1, 100, 200),
                    wal_chunk_file_key(1, 1, 200, 300),
                ],
                false,
                Ok(300),
            ),
            (
                // last chunk is not last, check the last
                vec![
                    wal_chunk_file_key(1, 1, 0, 100),
                    wal_chunk_file_key(1, 1, 100, 200),
                    wal_chunk_file_key(1, 1, 200, 300),
                ],
                true,
                Err(()),
            ),
            (
                // last chunk is last, ignore the last check
                vec![
                    wal_chunk_file_key(1, 1, 0, 100),
                    wal_chunk_file_key(1, 1, 100, 200),
                    last_wal_chunk_file_key(1, 1, 200, 300),
                ],
                false,
                Ok(300),
            ),
            (
                // last chunk is last, check the last
                vec![
                    wal_chunk_file_key(1, 1, 0, 100),
                    wal_chunk_file_key(1, 1, 100, 200),
                    last_wal_chunk_file_key(1, 1, 200, 300),
                ],
                true,
                Ok(300),
            ),
        ];

        for (keys, check_last, expected) in cases {
            assert_eq!(
                verify_wal_chunks_integrity(&keys, check_last).map_err(|_| ()),
                expected
            );
        }
    }
}
