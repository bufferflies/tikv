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

pub(crate) fn snapshot_store_meta_key(store_id: u64, epoch: u32) -> String {
    format!(
        "store_backup/{:016x}/snapshots/m{:08x}.meta",
        store_id, epoch
    )
}

pub(crate) fn snapshot_rlog_key(store_id: u64, epoch: u32) -> String {
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

pub fn verify_wal_chunks_integrity(chunks: &[String], check_last: bool) -> bool {
    if chunks.is_empty() {
        // `check_last` is true, it means the chunks belongs to a previous epoch, empty
        // chunks is invalid. `check_last` is false, it means the chunks belongs
        // to the current epoch, it is valid before the first chunk put to S3.
        return !check_last;
    }

    let wal_epoch;
    let mut last_end_off;
    let mut has_last_chunk;

    let first_chunk = chunks.first().unwrap();
    if let Some((epoch, start_off, end_off, last)) = parse_wal_chunk_key(Some(first_chunk)) {
        if start_off != 0 {
            return false;
        }
        wal_epoch = epoch;
        last_end_off = end_off;
        has_last_chunk = last;
    } else {
        return false;
    }

    for chunk in &chunks[1..] {
        if let Some((epoch, start_off, end_off, last)) = parse_wal_chunk_key(Some(chunk)) {
            if epoch != wal_epoch {
                return false;
            }
            if start_off != last_end_off {
                return false;
            }
            last_end_off = end_off;
            has_last_chunk = last;
        } else {
            return false;
        }
    }
    if !check_last { true } else { has_last_chunk }
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
pub mod tests {
    use std::time::Duration;

    use api_version::{
        api_v2::{self, TXN_KEY_PREFIX},
        ApiV2,
    };
    use byteorder::{BigEndian, ByteOrder};
    use kvproto::metapb::Region;
    use tikv_util::time::Instant;

    use crate::{
        get_region_keyspace_id, get_region_keyspace_id_str, last_wal_chunk_file_key,
        parse_epoch_from_snapshot_key, parse_wal_chunk_key, snapshot_rlog_key,
        verify_wal_chunks_integrity, wal_chunk_file_key,
    };

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
        let cases = vec![
            (vec![], false, true),                                  // empty
            (vec![wal_chunk_file_key(1, 1, 0, 100)], false, true),  // only has one chunk
            (vec![wal_chunk_file_key(1, 1, 1, 100)], false, false), // start_off is not 0
            (
                vec![
                    wal_chunk_file_key(1, 1, 1, 100),
                    wal_chunk_file_key(1, 1, 100, 200),
                ],
                false,
                false,
            ), // start_off is not 0
            (
                vec![
                    wal_chunk_file_key(1, 1, 0, 100),
                    wal_chunk_file_key(1, 1, 150, 200),
                    wal_chunk_file_key(1, 1, 200, 300),
                ],
                false,
                false,
            ), // offset is not continuous
            (
                vec![
                    wal_chunk_file_key(1, 1, 0, 100),
                    wal_chunk_file_key(1, 1, 100, 200),
                    wal_chunk_file_key(1, 2, 200, 300),
                ],
                false,
                false,
            ), // epoch is not consistent
            (
                vec![
                    wal_chunk_file_key(1, 1, 0, 100),
                    wal_chunk_file_key(1, 1, 100, 200),
                    wal_chunk_file_key(1, 1, 200, 300),
                ],
                false,
                true,
            ), /* last chunk is not last,
                                                                     * ignore the last check */
            (
                vec![
                    wal_chunk_file_key(1, 1, 0, 100),
                    wal_chunk_file_key(1, 1, 100, 200),
                    wal_chunk_file_key(1, 1, 200, 300),
                ],
                true,
                false,
            ), // last chunk is not last, check the last
            (
                vec![
                    wal_chunk_file_key(1, 1, 0, 100),
                    wal_chunk_file_key(1, 1, 100, 200),
                    last_wal_chunk_file_key(1, 1, 200, 300),
                ],
                false,
                true,
            ), // last chunk is last, ignore the last check
            (
                vec![
                    wal_chunk_file_key(1, 1, 0, 100),
                    wal_chunk_file_key(1, 1, 100, 200),
                    last_wal_chunk_file_key(1, 1, 200, 300),
                ],
                true,
                true,
            ), // last chunk is last, check the last
        ];

        for case in cases {
            assert_eq!(verify_wal_chunks_integrity(&case.0, case.1), case.2);
        }
    }
}
