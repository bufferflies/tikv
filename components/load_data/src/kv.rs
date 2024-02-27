// Copyright 2023 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    fs::File,
    io::{BufReader, Read},
};

use bytes::Bytes;
use encryption::DecrypterReader;
use kvengine::{table::Value, UserMeta};
use serde_derive::{Deserialize, Serialize};

use crate::error::Error;

const MAX_DUP_SIZE: usize = 64 * 1024 * 1024;

pub struct KvPair {
    pub key: Bytes,
    pub val: Bytes,
}

impl KvPair {
    pub fn new(key: Bytes, val: Bytes) -> KvPair {
        Self { key, val }
    }
}

pub struct KvPairsReader {
    key_buf: Vec<u8>,
    val_buf: Vec<u8>,
    val_base_len: usize,
    count: usize,
    idx: usize,
    buf_reader: BufReader<DecrypterReader<File>>,
}

impl KvPairsReader {
    pub fn new(start_ts: u64, commit_ts: u64, count: usize, reader: DecrypterReader<File>) -> Self {
        let buf_reader = BufReader::with_capacity(64 * 1024, reader);
        let um = UserMeta::new(start_ts, commit_ts);
        let val_buf = Value::encode_buf(0, &um.to_array(), commit_ts, &[]);
        let val_base_len = val_buf.len();
        Self {
            key_buf: vec![],
            val_buf,
            val_base_len,
            count,
            idx: 0,
            buf_reader,
        }
    }

    fn key(&self) -> &[u8] {
        &self.key_buf
    }

    fn valid(&self) -> bool {
        self.idx <= self.count
    }

    fn next(&mut self) {
        self.idx += 1;
        if self.idx > self.count {
            return;
        }
        let mut key_len_buf = [0u8; 2];
        self.buf_reader.read_exact(&mut key_len_buf[..]).unwrap();
        let key_len = u16::from_le_bytes(key_len_buf);
        self.key_buf.resize(key_len as usize, 0);
        self.buf_reader.read_exact(&mut self.key_buf[..]).unwrap();
        let mut val_len_buf = [0u8; 4];
        self.buf_reader.read_exact(&mut val_len_buf[..]).unwrap();
        let val_len = u32::from_le_bytes(val_len_buf);
        self.val_buf.resize(self.val_base_len + val_len as usize, 0);
        self.buf_reader
            .read_exact(&mut self.val_buf[self.val_base_len..])
            .unwrap();
    }
}

#[derive(Default, Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(default)]
#[serde(rename_all = "kebab-case")]
pub struct SstMeta {
    pub id: u64,
    pub smallest: Vec<u8>,
    pub biggest: Vec<u8>,
    pub size: usize,
    pub uncompressed_size: usize,
    pub keys: usize,
}

pub struct MergeIterator {
    #[allow(clippy::vec_box)]
    heap: Vec<Box<KvPairsReader>>,
    prev_key: Vec<u8>,
    prev_val: Vec<u8>,
    key_prefix: Vec<u8>,
    pub(crate) duplicated_entries: Vec<DuplicateEntry>,
    pub(crate) duplicated_entries_size: usize,
    last_dup_entry_key: Vec<u8>,
}

#[derive(Default, Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(default)]
#[serde(rename_all = "kebab-case")]
pub struct DuplicateEntry {
    pub key: String,
    pub values: Vec<String>,
}

impl MergeIterator {
    pub fn new(readers: Vec<KvPairsReader>, key_prefix: &[u8]) -> Self {
        let mut heap = Vec::with_capacity(readers.len());
        for mut reader in readers {
            reader.next();
            heap.push(Box::new(reader));
        }
        let mut it = Self {
            heap,
            prev_key: vec![],
            prev_val: vec![],
            key_prefix: key_prefix.to_vec(),
            duplicated_entries: vec![],
            duplicated_entries_size: 0,
            last_dup_entry_key: vec![],
        };
        it.init_heap();
        it.prev_key = it.key().to_vec();
        it.prev_val = it.value().to_vec();
        it
    }

    fn init_heap(&mut self) {
        for i in (0..self.heap.len() / 2).rev() {
            self.down(i);
        }
    }

    fn down(&mut self, i0: usize) -> bool {
        let n = self.heap.len();
        let mut i = i0;
        loop {
            let left = 2 * i + 1;
            if left >= n {
                break;
            }
            let right = left + 1;
            let j = if right < n && self.less(right, left) {
                right
            } else {
                left
            };
            if !self.less(j, i) {
                break;
            }
            self.heap.swap(i, j);
            i = j;
        }
        i > i0
    }

    fn less(&mut self, a: usize, b: usize) -> bool {
        self.heap[a].key() < self.heap[b].key()
    }

    pub fn key(&self) -> &[u8] {
        self.heap[0].key()
    }

    pub fn value(&self) -> &[u8] {
        &self.heap[0].val_buf
    }

    pub fn valid(&self) -> bool {
        !self.heap.is_empty()
    }

    pub fn next(&mut self) -> crate::Result<()> {
        loop {
            let dup = self.next_maybe_dup()?;
            if !dup {
                return Ok(());
            }
        }
    }

    pub fn next_maybe_dup(&mut self) -> crate::Result<bool> {
        let heap_len = self.heap.len();
        if heap_len == 0 {
            return Ok(false);
        }
        let first = &mut self.heap[0];
        first.next();
        if !first.valid() {
            self.heap.swap(0, heap_len - 1);
            self.heap.pop();
            if !self.valid() {
                return Ok(false);
            }
        }
        self.down(0);
        let key = self.heap[0].key();
        let val = self.heap[0].val_buf.as_slice();
        let val_base_len = self.heap[0].val_base_len;
        if key == self.prev_key.as_slice() {
            if self.duplicated_entries_size > MAX_DUP_SIZE {
                return Err(Error::TooManyDuplicatedKeys(
                    self.duplicated_entries.len().to_string(),
                ));
            }
            let val_str = hex::encode(&val[val_base_len..]);
            if let Some(entry) = self.duplicated_entries.last_mut() {
                if self.last_dup_entry_key.as_slice() == key {
                    self.duplicated_entries_size += val.len();
                    entry.values.push(val_str);
                    return Ok(true);
                }
            }
            self.duplicated_entries_size += key.len() + self.prev_val.len() + val.len();
            let mut dup_key = self.key_prefix.clone();
            dup_key.extend_from_slice(key);
            let dup_entry = DuplicateEntry {
                key: hex::encode(dup_key),
                values: vec![hex::encode(&self.prev_val[val_base_len..]), val_str],
            };
            self.duplicated_entries.push(dup_entry);
            self.last_dup_entry_key = key.to_vec();
            return Ok(true);
        }
        self.prev_key.truncate(0);
        self.prev_key.extend_from_slice(key);
        self.prev_val.truncate(0);
        self.prev_val.extend_from_slice(val);
        Ok(false)
    }
}
