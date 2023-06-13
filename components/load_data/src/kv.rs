// Copyright 2023 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    fs::File,
    io::{BufReader, Read, Seek, SeekFrom},
};

use bytes::Bytes;
use kvengine::{table::Value, UserMeta};
use serde_derive::{Deserialize, Serialize};

use crate::error::Error;

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
    buf_reader: BufReader<File>,
}

impl KvPairsReader {
    pub fn new(start_ts: u64, commit_ts: u64, count: usize, mut file: File) -> Self {
        file.seek(SeekFrom::Start(0)).unwrap();
        let buf_reader = BufReader::with_capacity(64 * 1024, file);
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

#[derive(Default, Debug, Serialize, Deserialize, Clone)]
#[serde(default)]
#[serde(rename_all = "kebab-case")]
pub struct SstMeta {
    pub id: u64,
    pub smallest: Vec<u8>,
    pub biggest: Vec<u8>,
    pub size: usize,
    pub keys: usize,
}

pub struct MergeIterator {
    #[allow(clippy::vec_box)]
    heap: Vec<Box<KvPairsReader>>,
    prev_key: Vec<u8>,
}

impl MergeIterator {
    pub fn new(readers: Vec<KvPairsReader>) -> Self {
        let mut heap = Vec::with_capacity(readers.len());
        for mut reader in readers {
            reader.next();
            heap.push(Box::new(reader));
        }
        let mut it = Self {
            heap,
            prev_key: vec![],
        };
        it.init_heap();
        it.prev_key = it.key().to_vec();
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
        let heap_len = self.heap.len();
        if heap_len == 0 {
            return Ok(());
        }
        let first = &mut self.heap[0];
        first.next();
        if !first.valid() {
            self.heap.swap(0, heap_len - 1);
            self.heap.pop();
            if !self.valid() {
                return Ok(());
            }
        }
        self.down(0);
        let key = self.heap[0].key();
        if key == self.prev_key.as_slice() {
            return Err(Error::DuplicatedKey(format!("{:?}", key)));
        }
        self.prev_key.truncate(0);
        self.prev_key.extend_from_slice(key);
        Ok(())
    }
}
