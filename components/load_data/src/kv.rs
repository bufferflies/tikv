// Copyright 2023 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    fs::File,
    io::{ErrorKind, Read},
    thread,
};

use bytes::Bytes;
use encryption::DecrypterReader;
use rfengine::decompress_lz4;
use serde_derive::{Deserialize, Serialize};
use tidb_query_datatype::codec::table::{TABLE_PREFIX_KEY_LEN, TABLE_PREFIX_LEN};
use tikv_util::mpsc::{bounded, Receiver};

use crate::error::{Error, Result};

const MAX_DUP_SIZE: usize = 64 * 1024 * 1024;

#[derive(Clone)]
pub struct KvPair {
    pub key: Bytes,
    pub val: Bytes,
    pub row_id: Bytes,
}

impl KvPair {
    pub fn new(key: Bytes, val: Bytes, row_id: Bytes) -> KvPair {
        Self { key, val, row_id }
    }
}

pub struct KvPairsReader {
    key_buf_len: usize,
    val_buf_len: usize,
    row_id_buf_len: usize,
    count: usize,
    idx: usize,
    buf: Vec<u8>,
    offset: usize,
    next_offset: usize,
    buf_rx: Receiver<Result<Vec<u8>>>,
    key_comm_prefix_len: usize,
    lower_bound_suffix: Vec<u8>,
    upper_bound_suffix: Vec<u8>,
    table_prefix_offset: usize,
}

impl KvPairsReader {
    pub fn new(
        count: usize,
        mut reader: DecrypterReader<File>,
        key_comm_prefix_len: usize,
        lower_bound_suffix: Vec<u8>,
        upper_bound_suffix: Vec<u8>,
        table_prefix_offset: usize,
    ) -> Self {
        let (buf_tx, buf_rx) = bounded(0);

        thread::spawn(move || {
            let mut compressed_size_buf = [0u8; 4];
            let mut compressed_buf: Vec<u8> = vec![];
            loop {
                if let Err(err) = reader.read_exact(&mut compressed_size_buf[..]) {
                    if err.kind() != ErrorKind::UnexpectedEof {
                        let _ = buf_tx.send(Err(Error::IoError(err)));
                    }
                    return;
                }

                let compressed_size = u32::from_le_bytes(compressed_size_buf) as usize;
                compressed_buf.resize(compressed_size, 0);
                if let Err(err) = reader.read_exact(&mut compressed_buf) {
                    let _ = buf_tx.send(Err(Error::IoError(err)));
                    return;
                }
                let buf = decompress_lz4(&compressed_buf[0..compressed_size]).unwrap();

                if buf_tx.send(Ok(buf)).is_err() {
                    return;
                }
            }
        });

        Self {
            key_buf_len: 0,
            val_buf_len: 0,
            row_id_buf_len: 0,
            count,
            idx: 0,
            // lazy init
            buf: vec![],
            offset: 0,
            next_offset: 0,
            buf_rx,
            key_comm_prefix_len,
            lower_bound_suffix,
            upper_bound_suffix,
            table_prefix_offset,
        }
    }

    fn init(&mut self) -> Result<()> {
        self.next()?;

        if !self.lower_bound_suffix.is_empty() {
            while self.valid() && self.key_suffix() < self.lower_bound_suffix.as_slice() {
                self.next()?;
            }
        }
        Ok(())
    }

    fn key(&self) -> &[u8] {
        &self.buf[self.offset + 2..self.offset + 2 + self.key_buf_len]
    }

    fn key_suffix(&self) -> &[u8] {
        &self.buf[self.offset + 2 + self.key_comm_prefix_len..self.offset + 2 + self.key_buf_len]
    }

    fn value(&self) -> &[u8] {
        let offset = self.offset + 2 + self.key_buf_len + 4;
        &self.buf[offset..offset + self.val_buf_len]
    }

    fn row_id(&self) -> &[u8] {
        let offset = self.offset + 2 + self.key_buf_len + 4 + self.val_buf_len + 2;
        &self.buf[offset..offset + self.row_id_buf_len]
    }

    fn table_id(&self) -> &[u8] {
        &self.buf[self.offset + 2 + self.table_prefix_offset + TABLE_PREFIX_LEN
            ..self.offset + 2 + self.table_prefix_offset + TABLE_PREFIX_KEY_LEN]
    }

    fn valid(&self) -> bool {
        self.idx <= self.count
            && (self.upper_bound_suffix.is_empty()
                || self.key_suffix() < self.upper_bound_suffix.as_slice())
    }

    fn next(&mut self) -> Result<()> {
        self.idx += 1;
        if self.idx > self.count {
            return Ok(());
        }

        self.offset = self.next_offset;
        if self.offset == self.buf.len() {
            let buf = self.buf_rx.recv().unwrap();
            match buf {
                Ok(buf) => {
                    self.buf = buf;
                }
                Err(err) => {
                    return Err(err);
                }
            }
            self.offset = 0;
        }

        let mut offset = self.offset;
        let buf = &self.buf;
        self.key_buf_len = u16::from_le_bytes(buf[offset..offset + 2].try_into().unwrap()) as usize;
        offset += 2 + self.key_buf_len;
        self.val_buf_len = u32::from_le_bytes(buf[offset..offset + 4].try_into().unwrap()) as usize;
        offset += 4 + self.val_buf_len;
        self.row_id_buf_len =
            u16::from_le_bytes(buf[offset..offset + 2].try_into().unwrap()) as usize;
        offset += 2 + self.row_id_buf_len;
        self.next_offset = offset;

        Ok(())
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
    pub meta_offset: u32,
    pub uncompressed_size: usize,
    pub keys: usize,
}

pub struct MergeIterator {
    #[allow(clippy::vec_box)]
    heap: Vec<Box<KvPairsReader>>,
    prev_key: Vec<u8>,
    prev_val: Vec<u8>,
    prev_row_id: Vec<u8>,
    outer_key_prefix: Vec<u8>,
    pub(crate) duplicated_entries: Vec<DuplicateEntry>,
    pub(crate) duplicated_entries_size: usize,
    last_dup_entry_key: Vec<u8>,
    last_dup_entry_row_id: Vec<u8>,
}

#[derive(Default, Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(default)]
#[serde(rename_all = "kebab-case")]
pub struct DuplicateEntry {
    pub key: String,
    pub values: Vec<String>,
}

impl MergeIterator {
    pub fn new(readers: Vec<KvPairsReader>, outer_key_prefix: &[u8]) -> Result<Self> {
        let mut heap = Vec::with_capacity(readers.len());
        for mut reader in readers {
            reader.init()?;
            if reader.valid() {
                heap.push(Box::new(reader));
            }
        }

        let mut it = Self {
            heap,
            prev_key: vec![],
            prev_val: vec![],
            prev_row_id: vec![],
            outer_key_prefix: outer_key_prefix.to_vec(),
            duplicated_entries: vec![],
            duplicated_entries_size: 0,
            last_dup_entry_key: vec![],
            last_dup_entry_row_id: vec![],
        };
        it.init_heap();
        it.prev_key = it.key().to_vec();
        it.prev_val = it.value().to_vec();
        it.prev_row_id = it.row_id().to_vec();
        Ok(it)
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
        match self.heap[a].key_suffix().cmp(self.heap[b].key_suffix()) {
            std::cmp::Ordering::Less => true,
            std::cmp::Ordering::Equal => self.heap[a].row_id() < self.heap[b].row_id(),
            std::cmp::Ordering::Greater => false,
        }
    }

    pub fn key(&self) -> &[u8] {
        self.heap[0].key()
    }

    pub fn value(&self) -> &[u8] {
        self.heap[0].value()
    }

    pub fn row_id(&self) -> &[u8] {
        self.heap[0].row_id()
    }

    pub fn table_id(&self) -> &[u8] {
        self.heap[0].table_id()
    }

    pub fn prev_key(&self) -> &[u8] {
        self.prev_key.as_slice()
    }

    pub fn valid(&self) -> bool {
        !self.heap.is_empty()
    }

    pub fn next(&mut self) -> Result<()> {
        loop {
            let dup = self.next_maybe_dup()?;
            if !dup {
                return Ok(());
            }
        }
    }

    pub fn next_maybe_dup(&mut self) -> Result<bool> {
        let heap_len = self.heap.len();
        if heap_len == 0 {
            return Ok(false);
        }
        let first = &mut self.heap[0];

        first.next()?;
        if !first.valid() {
            self.heap.swap(0, heap_len - 1);
            self.heap.pop();
            if !self.valid() {
                return Ok(false);
            }
        }
        self.down(0);

        let key = self.heap[0].key();
        let val = self.heap[0].value();
        let row_id = self.heap[0].row_id();
        if key == self.prev_key.as_slice() {
            if row_id == self.prev_row_id.as_slice() {
                return Ok(true);
            }

            if self.duplicated_entries_size > MAX_DUP_SIZE {
                return Err(Error::TooManyDuplicatedKeys(
                    self.duplicated_entries.len().to_string(),
                ));
            }
            let val_str = hex::encode(val);
            if let Some(entry) = self.duplicated_entries.last_mut() {
                if self.last_dup_entry_key.as_slice() == key {
                    if self.last_dup_entry_row_id.as_slice() != row_id {
                        self.duplicated_entries_size += val.len();
                        entry.values.push(val_str);
                    }
                    return Ok(true);
                }
            }
            self.duplicated_entries_size += key.len() + self.prev_val.len() + val.len();
            let mut dup_key = self.outer_key_prefix.clone();
            dup_key.extend_from_slice(key);
            let dup_entry = DuplicateEntry {
                key: hex::encode(dup_key),
                values: vec![hex::encode(&self.prev_val), val_str],
            };
            self.duplicated_entries.push(dup_entry);
            self.last_dup_entry_key = key.to_vec();
            self.last_dup_entry_row_id = row_id.to_vec();
            return Ok(true);
        }

        self.prev_key.truncate(0);
        self.prev_key.extend_from_slice(key);
        self.prev_val.truncate(0);
        self.prev_val.extend_from_slice(val);
        self.prev_row_id.truncate(0);
        self.prev_row_id.extend_from_slice(row_id);
        Ok(false)
    }
}

#[cfg(test)]
mod tests {
    use std::{path::PathBuf, thread::sleep, time::Duration};

    use api_version::api_v2::KEYSPACE_PREFIX_LEN;
    use bytes::Bytes;
    use chrono::Utc;
    use rand::{seq::SliceRandom, thread_rng};
    use tempfile::TempDir;
    use tidb_query_datatype::codec::table;

    use super::*;
    use crate::{
        checkpoint::FileMeta,
        task::TaskContext,
        worker::{build_readers, flush_l0_file_to_local, get_common_prefix},
    };

    #[test]
    fn test_kv_pairs_reader() {
        let kv_pair_size = get_kv_pair_size();
        let kv_count = 5;
        let mut ids: Vec<usize> = (0..kv_count).collect();
        ids.shuffle(&mut thread_rng());

        let mut kv_pairs = Vec::with_capacity(ids.len());
        for id in ids.iter() {
            kv_pairs.push(KvPair::new(
                Bytes::from(i_to_key(1, id)),
                Bytes::from(i_to_val(id)),
                Bytes::from(i_to_row_id(id)),
            ));
        }
        let tmp_dir = TempDir::new().unwrap();
        let path = tmp_dir.path().join(format!(
            "test_kv_pairs_reader_{}",
            Utc::now().timestamp_millis()
        ));

        let mock_task_ctx = generate_mock_task_ctx();
        let (file_meta, key_comm_prefix) =
            generate_file_meta(&mock_task_ctx, kv_pairs, path.clone(), kv_pair_size);
        let mut reader = build_readers(
            &mock_task_ctx,
            vec![file_meta],
            key_comm_prefix.len(),
            vec![],
            vec![],
        )
        .pop()
        .unwrap();
        let key_comm_prefix_len = key_comm_prefix.len();
        let mut key_buf = key_comm_prefix;

        reader.init().unwrap();
        ids.sort();
        for id in ids.iter() {
            key_buf.resize(key_comm_prefix_len, 0);
            key_buf.extend_from_slice(reader.key_suffix());
            let key = i_to_key(1, id);
            assert_eq!(key.as_slice(), reader.key());
            assert_eq!(key.as_slice(), key_buf.as_slice());
            assert_eq!(i_to_val(id).as_slice(), reader.value());
            assert_eq!(i_to_row_id(id).as_slice(), reader.row_id());
            reader.next().unwrap();
        }
        reader.next().unwrap();
        assert!(!reader.valid());
    }

    #[test]
    fn test_merge_iterator() {
        let reader_count = 3;
        let kv_count_per_reader = 5;
        let kv_pair_size = get_kv_pair_size();
        let mut key_ids: Vec<usize> = (0..reader_count * kv_count_per_reader).collect();
        key_ids.shuffle(&mut thread_rng());
        let mut kv_pairs = Vec::with_capacity(key_ids.len());
        for id in key_ids.iter() {
            kv_pairs.push(KvPair::new(
                Bytes::from(i_to_key(1, id)),
                Bytes::from(i_to_val(id)),
                Bytes::from(i_to_row_id(id)),
            ));
        }

        let mock_task_ctx = generate_mock_task_ctx();
        let tmp_dir = TempDir::new().unwrap();
        let mut file_metas = vec![];
        let mut last_key_comm_prefix = i_to_key(1, &key_ids[0]);
        for i in 0..reader_count {
            let kv_pairs_part =
                kv_pairs[i * kv_count_per_reader..(i + 1) * kv_count_per_reader].to_vec();
            let path = tmp_dir.path().join(format!(
                "test_merge_iterator_{}",
                Utc::now().timestamp_millis()
            ));
            let (file_meta, key_comm_prefix) = generate_file_meta(
                &mock_task_ctx,
                kv_pairs_part,
                path.clone(),
                kv_pair_size * (kv_count_per_reader / 2),
            );
            file_metas.push(file_meta);
            last_key_comm_prefix = get_common_prefix(&last_key_comm_prefix, &key_comm_prefix);

            sleep(Duration::from_millis(3));
        }
        let readers = build_readers(
            &mock_task_ctx,
            file_metas,
            last_key_comm_prefix.len(),
            vec![],
            vec![],
        );
        let mut merge_iter = MergeIterator::new(readers, "".as_bytes()).unwrap();

        key_ids.sort();
        for id in key_ids.iter() {
            assert_eq!(i_to_key(1, id), merge_iter.key());
            assert_eq!(i_to_val(id).as_slice(), merge_iter.value());
            assert_eq!(i_to_row_id(id).as_slice(), merge_iter.row_id());
            merge_iter.next().unwrap();
        }
        assert!(!merge_iter.valid());
    }

    fn generate_file_meta(
        task_ctx: &TaskContext,
        kv_pairs: Vec<KvPair>,
        file_path: PathBuf,
        batch_size: usize,
    ) -> (FileMeta, Vec<u8>) {
        let kv_count = kv_pairs.len();
        let (first_key, last_key, key_comm_prefix, kv_size) =
            flush_l0_file_to_local(kv_pairs, task_ctx.clone(), file_path.clone(), batch_size)
                .unwrap();
        let file_meta = FileMeta {
            file_path,
            kv_count,
            kv_size,
            first_key,
            last_key,
        };
        (file_meta, key_comm_prefix)
    }

    fn generate_mock_task_ctx() -> TaskContext {
        TaskContext {
            task_id: "mock_load_data_task_id".to_string(),
            start_ts: 0,
            commit_ts: 0,
            inner_key_off: Some(KEYSPACE_PREFIX_LEN),
            outer_key_prefix: vec![],
            encryption_key: None,
            keyspace_id: None,
        }
    }

    fn i_to_key(table_id: i64, i: &usize) -> Vec<u8> {
        table::encode_row_key(table_id, *i as i64) // 19 bytes
    }

    fn i_to_val(i: &usize) -> Vec<u8> {
        format!("val_{:08}", i).repeat(10).into_bytes() // 120 bytes
    }

    fn i_to_row_id(i: &usize) -> Vec<u8> {
        format!("row_{:08}", i).into_bytes() // 12 bytes
    }

    fn get_kv_pair_size() -> usize {
        19 + 120 + 12
    }
}
