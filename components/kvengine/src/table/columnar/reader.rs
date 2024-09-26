// Copyright 2024 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    cmp::{min, Ordering},
    collections::HashMap,
    mem,
    ops::Deref,
    sync::Arc,
};

use api_version::{
    api_v2::{KEYSPACE_ID_LEN, TXN_KEY_PREFIX},
    ApiV2,
};
use bytes::Buf;
use cloud_encryption::EncryptionKey;
use tidb_query_datatype::{
    codec::{
        datum,
        datum::{
            BYTES_FLAG, COMPACT_BYTES_FLAG, DECIMAL_FLAG, DURATION_FLAG, FLOAT_FLAG, INT_FLAG,
            JSON_FLAG, NIL_FLAG, UINT_FLAG, VAR_INT_FLAG, VAR_UINT_FLAG, VECTOR_FLOAT32_FLAG,
        },
        mysql::{DecimalDecoder, DecimalEncoder},
        row::v2::{decode_v2_i64, decode_v2_u64, RowSlice},
        table::{
            decode_common_handle, decode_int_handle, encode_common_handle_for_test, encode_row_key,
            PREFIX_LEN,
        },
    },
    FieldTypeFlag, FieldTypeTp,
};
use tikv_util::codec::{
    bytes::{decode_bytes, decode_compact_bytes},
    number::{decode_f64, decode_i64, decode_u64, decode_var_i64, decode_var_u64},
};
use tipb::ColumnInfo;

use crate::{
    table::{
        self,
        blobtable::blobtable::BlobTable,
        columnar::{
            columnar::{
                decompress_pack, get_fixed_size, Block, ColumnBuffer, ColumnMeta, ColumnarFile,
                Schema, TableMeta,
            },
            get_primary_key,
        },
        file::File,
        search, InnerKey,
    },
    UserMeta,
};

pub const GLOBAL_COMMON_HANDLE_END: &[u8] = &[255];

pub trait ColumnarReader: Send {
    fn schema(&self) -> &Schema;
    fn seek(&mut self, handle: &[u8]) -> crate::table::Result<()>;
    fn read(&mut self, block: &mut Block, limit: usize) -> crate::table::Result<usize>;
}

pub(crate) struct ColumnarTableReader {
    table_meta: Arc<TableMeta>,
    schema: Schema,
    handle_reader: ColumnarColumnReader,
    version_reader: ColumnarColumnReader,
    txn_id_reader: Option<ColumnarColumnReader>,
    columns_readers: Vec<ColumnarColumnReader>,
}

#[allow(dead_code)]
impl ColumnarTableReader {
    pub fn new(
        columnar_file: &ColumnarFile,
        schema: Schema,
        encryption_key: Option<EncryptionKey>,
    ) -> ColumnarTableReader {
        let table_meta = columnar_file.get_table(schema.table_id);
        debug_assert_eq!(table_meta.table_id, schema.table_id);
        let file = columnar_file.get_file();
        let encryption_ver = columnar_file.get_encryption_ver();
        let handle_reader = ColumnarColumnReader::new(
            file.clone(),
            table_meta.handle_column.clone(),
            false,
            encryption_key.clone(),
            encryption_ver,
        );
        let version_reader = ColumnarColumnReader::new(
            file.clone(),
            table_meta.version_column.clone(),
            false,
            encryption_key.clone(),
            encryption_ver,
        );
        let txn_id_reader = schema.txn_id_column.is_some().then(|| {
            ColumnarColumnReader::new(
                file.clone(),
                table_meta.txn_id_column.clone(),
                false,
                encryption_key.clone(),
                encryption_ver,
            )
        });
        let columns_readers = schema
            .columns
            .iter()
            .map(|col| {
                let col_id = col.get_column_id() as i32;
                if let Some(col_meta) = table_meta.columns.get(&col_id) {
                    ColumnarColumnReader::new(
                        file.clone(),
                        col_meta.clone(),
                        false,
                        encryption_key.clone(),
                        encryption_ver,
                    )
                } else {
                    let col_meta = ColumnMeta::new(col.clone(), false);
                    ColumnarColumnReader::new(
                        file.clone(),
                        Arc::new(col_meta),
                        true,
                        encryption_key.clone(),
                        encryption_ver,
                    )
                }
            })
            .collect();
        ColumnarTableReader {
            table_meta,
            schema,
            handle_reader,
            version_reader,
            txn_id_reader,
            columns_readers,
        }
    }
}

impl ColumnarReader for ColumnarTableReader {
    fn schema(&self) -> &Schema {
        &self.schema
    }

    fn seek(&mut self, handle: &[u8]) -> crate::table::Result<()> {
        let pack_idx = self.table_meta.handle_index.search_pack_idx(handle);
        self.handle_reader.load_pack(pack_idx)?;
        let handle_buffer = &self.handle_reader.pack_buffer;
        let row_idx_in_pack = if handle.is_empty() {
            0
        } else if self.handle_reader.col_meta.fixed_size > 0 {
            let int_handle = (&handle[..]).get_i64_le();
            search(handle_buffer.length(), |i| {
                handle_buffer.get_int_handle_value(i) >= int_handle
            })
        } else {
            search(handle_buffer.length(), |i| {
                handle_buffer.get_not_null_value(i) >= handle
            })
        };
        self.handle_reader.row_idx_in_pack = row_idx_in_pack;
        let row_idx = self.handle_reader.pack_row_start + row_idx_in_pack;
        self.version_reader.set_row_idx(row_idx)?;
        if let Some(txn_id_reader) = &mut self.txn_id_reader {
            txn_id_reader.set_row_idx(row_idx)?;
        }
        for col_reader in &mut self.columns_readers {
            col_reader.set_row_idx(row_idx)?;
        }
        Ok(())
    }

    fn read(&mut self, block: &mut Block, limit: usize) -> crate::table::Result<usize> {
        let read_row = self.handle_reader.read(&mut block.handles, limit)?;
        self.version_reader.read(&mut block.versions, limit)?;
        if let Some(txn_id_reader) = &mut self.txn_id_reader {
            txn_id_reader.read(block.txn_ids.as_mut().unwrap(), limit)?;
        }
        for (i, col) in self.columns_readers.iter_mut().enumerate() {
            col.read(&mut block.columns[i], limit)?;
        }
        Ok(read_row)
    }
}

pub(crate) struct ColumnarColumnReader {
    pack_loader: PackLoader,
    col_meta: Arc<ColumnMeta>,
    pack_buffer: ColumnBuffer,
    pack_idx: usize,
    pack_row_start: usize,
    pack_row_end: usize,
    row_idx_in_pack: usize,
    is_default_val: bool,
    default_val: Option<Vec<u8>>,
    decryption_buf: Vec<u8>,
}

impl ColumnarColumnReader {
    pub(crate) fn new(
        file: Arc<dyn File>,
        col_meta: Arc<ColumnMeta>,
        is_default_val: bool,
        encryption_key: Option<EncryptionKey>,
        encryption_ver: u32,
    ) -> ColumnarColumnReader {
        let pack_loader = PackLoader::new(file, encryption_key, encryption_ver);
        let pack_buffer = ColumnBuffer::new_from_col_info(&col_meta.col_info);
        let default_val = if is_default_val {
            parse_default_val(&col_meta.col_info)
        } else {
            None
        };
        ColumnarColumnReader {
            pack_loader,
            col_meta,
            pack_buffer,
            pack_idx: 0,
            pack_row_start: 0,
            pack_row_end: 0,
            row_idx_in_pack: 0,
            is_default_val,
            default_val,
            decryption_buf: vec![],
        }
    }

    pub(crate) fn set_row_idx(&mut self, row_idx: usize) -> crate::table::Result<()> {
        if self.is_default_val {
            self.row_idx_in_pack = row_idx;
            return Ok(());
        }
        if self.pack_row_start < row_idx && row_idx < self.pack_row_end {
            self.row_idx_in_pack = row_idx - self.pack_row_start;
            return Ok(());
        }
        let from_pack_idx = if self.pack_row_start < row_idx {
            self.pack_idx + 1
        } else {
            1
        };
        let pack_idx = self
            .col_meta
            .pack_offsets
            .search_pack_idx(from_pack_idx, row_idx as u32);
        self.load_pack(pack_idx)?;
        self.row_idx_in_pack = row_idx - self.pack_row_start;
        Ok(())
    }

    pub(crate) fn read(
        &mut self,
        to: &mut ColumnBuffer,
        limit: usize,
    ) -> crate::table::Result<usize> {
        if self.is_default_val {
            return self.read_default(to, limit);
        }
        let mut read_row = 0;
        let num_packs = self.col_meta.pack_offsets.num_packs();
        while read_row < limit {
            let remain = self.pack_buffer.length() - self.row_idx_in_pack;
            if remain == 0 {
                let next_pack_idx = self.pack_idx + 1;
                if next_pack_idx >= num_packs {
                    break;
                }
                self.load_pack(next_pack_idx)?;
                self.row_idx_in_pack = 0;
                continue;
            }
            let batch_size = min(remain, limit - read_row);
            to.append(
                &self.pack_buffer,
                self.row_idx_in_pack,
                self.row_idx_in_pack + batch_size,
            );
            self.row_idx_in_pack += batch_size;
            read_row += batch_size;
        }
        Ok(read_row)
    }

    fn read_default(&mut self, to: &mut ColumnBuffer, limit: usize) -> crate::table::Result<usize> {
        if let Some(val) = &self.default_val {
            for _ in 0..limit {
                to.push_value(val);
            }
        } else {
            for _ in 0..limit {
                to.push_null();
            }
        }
        Ok(limit)
    }

    fn load_pack(&mut self, pack_idx: usize) -> crate::table::Result<()> {
        let num_packs = self.col_meta.pack_offsets.num_packs();
        if pack_idx >= num_packs {
            self.pack_idx = num_packs;
            self.pack_buffer.reset();
            let (_, row_end_off) = self.col_meta.pack_offsets.end_offset();
            self.pack_row_start = row_end_off as usize;
            self.pack_row_end = row_end_off as usize;
            self.row_idx_in_pack = 0;
            return Ok(());
        }
        let ((pack_start, pack_row_start), (pack_end, pack_row_end)) =
            self.col_meta.get_pack_offset(pack_idx);
        self.pack_loader.load_pack(
            &mut self.pack_buffer,
            pack_start,
            pack_end,
            &mut self.decryption_buf,
        )?;
        self.pack_idx = pack_idx;
        self.pack_row_start = pack_row_start as usize;
        self.pack_row_end = pack_row_end as usize;
        Ok(())
    }
}

struct PackLoader {
    file: Arc<dyn File>,
    compressed_buf: Vec<u8>,
    uncompressed_buf: Vec<u8>,
    encryption_key: Option<EncryptionKey>,
    encryption_ver: u32,
}

impl PackLoader {
    pub fn new(
        file: Arc<dyn File>,
        encryption_key: Option<EncryptionKey>,
        encryption_ver: u32,
    ) -> PackLoader {
        PackLoader {
            file,
            encryption_key,
            encryption_ver,
            compressed_buf: vec![],
            uncompressed_buf: vec![],
        }
    }

    pub fn load_pack(
        &mut self,
        col_buf: &mut ColumnBuffer,
        pack_offset: u32,
        pack_end_offset: u32,
        decryption_buf: &mut Vec<u8>,
    ) -> crate::table::Result<()> {
        let length = (pack_end_offset - pack_offset) as usize;
        if let Some(encryption_key) = &self.encryption_key {
            decryption_buf.resize(length, 0);
            self.file.read_at(decryption_buf, pack_offset as u64)?;
            self.compressed_buf.clear();
            encryption_key.decrypt(
                decryption_buf,
                self.file.id(),
                pack_offset,
                self.encryption_ver,
                &mut self.compressed_buf,
            );
        } else {
            self.compressed_buf.resize(length, 0);
            self.file
                .read_at(&mut self.compressed_buf, pack_offset as u64)?;
        }
        decompress_pack(&self.compressed_buf, &mut self.uncompressed_buf);
        col_buf.parse(&self.uncompressed_buf);
        Ok(())
    }
}

pub trait ColumnarFilterReader: Send {
    fn set_handle_range(
        &mut self,
        start_handle: &[u8],
        end_handle: &[u8],
    ) -> crate::table::Result<()>;
    fn set_int_handle_range(
        &mut self,
        start_handle: i64,
        end_handle: Option<i64>,
    ) -> crate::table::Result<()>;
    fn get_schema(&self) -> &Schema;
    fn read_block(&mut self, block: &mut Block, limit: usize) -> crate::table::Result<usize>;
    fn set_unbounded_handle_range(&mut self) -> crate::table::Result<()> {
        if self.get_schema().is_common_handle() {
            self.set_handle_range(&[], GLOBAL_COMMON_HANDLE_END)?;
        } else {
            self.set_int_handle_range(i64::MIN, None)?;
        }
        Ok(())
    }
}

struct ColumnarFilter {
    ranges: Vec<(usize, usize)>,
    in_range: bool,
    range_start: usize,
    filter_block: Block,
}

impl ColumnarFilter {
    fn new(schema: &Schema) -> Self {
        Self {
            ranges: vec![],
            in_range: false,
            range_start: 0,
            filter_block: Block::new(schema),
        }
    }

    fn finish_range(&mut self, i: usize) {
        if self.in_range {
            self.ranges.push((self.range_start, i));
            self.in_range = false;
        }
    }

    fn start_range(&mut self, i: usize) {
        if !self.in_range {
            self.range_start = i;
            self.in_range = true;
        }
    }

    fn clear(&mut self) {
        self.ranges.clear();
        self.in_range = false;
    }

    fn do_filter(&mut self, read_row: usize, block: &mut Block) -> usize {
        if self.ranges.len() == 1 {
            let (start, end) = self.ranges[0];
            if start == 0 {
                return if end == read_row {
                    // All rows are valid, no need to filter.
                    read_row
                } else {
                    block.truncate(end);
                    end
                };
            }
        }
        self.filter_block.reset();
        let mut filtered_rows = 0;
        for &(start, end) in &self.ranges {
            self.filter_block.handles.append(&block.handles, start, end);
            self.filter_block
                .versions
                .append(&block.versions, start, end);
            if let Some(filter_txn_ids) = &mut self.filter_block.txn_ids {
                filter_txn_ids.append(block.txn_ids.as_ref().unwrap(), start, end);
            }
            for (i, col) in self.filter_block.columns.iter_mut().enumerate() {
                col.append(&block.columns[i], start, end);
            }
            filtered_rows += end - start;
        }
        mem::swap(block, &mut self.filter_block);
        filtered_rows
    }
}

pub struct ColumnarMvccReader {
    src: Box<dyn ColumnarReader>,
    read_ts: u64,
    filter: ColumnarFilter,
    end_handle: Vec<u8>,
    end_int_handle: Option<i64>,
    prev_int_handle: Option<i64>,
    prev_common_handle: Vec<u8>,
}

impl ColumnarMvccReader {
    pub fn new(src: Box<dyn ColumnarReader>, schema: &Schema, read_ts: u64) -> ColumnarMvccReader {
        ColumnarMvccReader {
            src,
            filter: ColumnarFilter::new(schema),
            read_ts,
            end_handle: vec![],
            end_int_handle: None,
            prev_int_handle: None,
            prev_common_handle: vec![],
        }
    }
}

impl ColumnarFilterReader for ColumnarMvccReader {
    fn set_handle_range(
        &mut self,
        start_handle: &[u8],
        end_handle: &[u8],
    ) -> crate::table::Result<()> {
        self.end_handle = end_handle.to_vec();
        self.src.seek(start_handle)?;
        self.filter.clear();
        Ok(())
    }

    fn set_int_handle_range(
        &mut self,
        start_handle: i64,
        end_handle: Option<i64>,
    ) -> crate::table::Result<()> {
        self.end_int_handle = end_handle;
        self.src.seek(&start_handle.to_le_bytes())?;
        self.filter.clear();
        Ok(())
    }

    fn get_schema(&self) -> &Schema {
        self.src.schema()
    }

    fn read_block(&mut self, block: &mut Block, limit: usize) -> crate::table::Result<usize> {
        self.filter.clear();
        block.reset();
        let read_row = self.src.read(block, limit)?;
        if read_row == 0 {
            return Ok(0);
        }
        if block.handles.fixed_size > 0 {
            for i in 0..block.handles.length() {
                let handle = block.handles.get_int_handle_value(i);
                let version = block.versions.get_version(i);
                if version > self.read_ts
                    || (self.prev_int_handle.is_some() && handle == self.prev_int_handle.unwrap())
                {
                    self.filter.finish_range(i);
                    continue;
                }
                if block.versions.is_null(i) {
                    self.prev_int_handle = Some(handle);
                    self.filter.finish_range(i);
                    continue;
                }
                if self.end_int_handle.is_some() && handle >= self.end_int_handle.unwrap() {
                    self.filter.finish_range(i);
                    break;
                }
                self.filter.start_range(i);
                self.prev_int_handle = Some(handle);
            }
        } else {
            let length = block.handles.length();
            let last_handle = block.handles.get_not_null_value(length - 1);
            let check_handle = last_handle >= self.end_handle.as_slice();
            for i in 0..block.handles.length() {
                let handle = block.handles.get_not_null_value(i);
                let version = block.versions.get_version(i);
                if version > self.read_ts || handle == self.prev_common_handle {
                    self.filter.finish_range(i);
                    continue;
                }
                if block.versions.is_null(i) {
                    self.prev_common_handle = handle.to_vec();
                    self.filter.finish_range(i);
                    continue;
                }
                if check_handle && handle >= self.end_handle.as_slice() {
                    self.filter.finish_range(i);
                    break;
                }
                self.filter.start_range(i);
                self.prev_common_handle = handle.to_vec();
            }
        }
        self.filter.finish_range(read_row);
        Ok(self.filter.do_filter(read_row, block))
    }
}

pub struct ColumnarCompactReader {
    src: Box<dyn ColumnarReader>,
    level: u32,
    safe_ts: u64,
    filter: ColumnarFilter,
    end_handle: Vec<u8>,
    end_int_handle: Option<i64>,
    prev_int_handle: Option<i64>,
    prev_common_handle: Vec<u8>,
}

impl ColumnarCompactReader {
    pub fn new(
        src: Box<dyn ColumnarReader>,
        level: u32,
        schema: &Schema,
        safe_ts: u64,
    ) -> ColumnarCompactReader {
        ColumnarCompactReader {
            src,
            level,
            filter: ColumnarFilter::new(schema),
            safe_ts,
            end_handle: vec![],
            end_int_handle: None,
            prev_int_handle: None,
            prev_common_handle: vec![],
        }
    }
}

impl ColumnarFilterReader for ColumnarCompactReader {
    fn set_handle_range(
        &mut self,
        start_handle: &[u8],
        end_handle: &[u8],
    ) -> crate::table::Result<()> {
        self.end_handle = end_handle.to_vec();
        self.src.seek(start_handle)?;
        self.filter.clear();
        Ok(())
    }

    fn set_int_handle_range(
        &mut self,
        start_handle: i64,
        end_handle: Option<i64>,
    ) -> crate::table::Result<()> {
        self.end_int_handle = end_handle;
        self.src.seek(&start_handle.to_le_bytes())?;
        self.filter.clear();
        Ok(())
    }

    fn get_schema(&self) -> &Schema {
        self.src.schema()
    }

    fn read_block(&mut self, block: &mut Block, limit: usize) -> crate::table::Result<usize> {
        self.filter.clear();
        block.reset();
        let read_row = self.src.read(block, limit)?;
        if read_row == 0 {
            return Ok(0);
        }
        if block.handles.fixed_size > 0 {
            for i in 0..block.handles.length() {
                let handle = block.handles.get_int_handle_value(i);
                let version = block.versions.get_version(i);
                if self.prev_int_handle.is_some()
                    && handle == self.prev_int_handle.unwrap()
                    && version < self.safe_ts
                {
                    self.filter.finish_range(i);
                    continue;
                }
                if self.level == 2 && version < self.safe_ts && block.versions.is_null(i) {
                    self.prev_int_handle = Some(handle);
                    self.filter.finish_range(i);
                    continue;
                }
                if self.end_int_handle.is_some() && handle >= self.end_int_handle.unwrap() {
                    self.filter.finish_range(i);
                    break;
                }
                self.filter.start_range(i);
                self.prev_int_handle = Some(handle);
            }
        } else {
            let length = block.handles.length();
            let last_handle = block.handles.get_not_null_value(length - 1);
            let check_handle = last_handle >= self.end_handle.as_slice();
            for i in 0..block.handles.length() {
                let handle = block.handles.get_not_null_value(i);
                let version = block.versions.get_version(i);
                if handle == self.prev_common_handle && version < self.safe_ts {
                    self.filter.finish_range(i);
                    continue;
                }
                if self.level == 2 && version < self.safe_ts && block.versions.is_null(i) {
                    self.prev_common_handle = handle.to_vec();
                    self.filter.finish_range(i);
                    continue;
                }
                if check_handle && handle >= self.end_handle.as_slice() {
                    self.filter.finish_range(i);
                    break;
                }
                self.filter.start_range(i);
                self.prev_common_handle = handle.to_vec();
            }
        }
        self.filter.finish_range(read_row);
        Ok(self.filter.do_filter(read_row, block))
    }
}

pub struct ColumnarTruncateTsReader {
    src: Box<dyn ColumnarReader>,
    truncate_ts: u64,
    filter: ColumnarFilter,
    end_handle: Vec<u8>,
    end_int_handle: Option<i64>,
}

impl ColumnarTruncateTsReader {
    pub fn new(
        src: Box<dyn ColumnarReader>,
        schema: &Schema,
        truncate_ts: u64,
    ) -> ColumnarTruncateTsReader {
        ColumnarTruncateTsReader {
            src,
            filter: ColumnarFilter::new(schema),
            truncate_ts,
            end_handle: vec![],
            end_int_handle: None,
        }
    }
}

impl ColumnarFilterReader for ColumnarTruncateTsReader {
    fn set_handle_range(
        &mut self,
        start_handle: &[u8],
        end_handle: &[u8],
    ) -> crate::table::Result<()> {
        self.end_handle = end_handle.to_vec();
        self.src.seek(start_handle)?;
        self.filter.clear();
        Ok(())
    }

    fn set_int_handle_range(
        &mut self,
        start_handle: i64,
        end_handle: Option<i64>,
    ) -> crate::table::Result<()> {
        self.end_int_handle = end_handle;
        self.src.seek(&start_handle.to_le_bytes())?;
        self.filter.clear();
        Ok(())
    }

    fn get_schema(&self) -> &Schema {
        self.src.schema()
    }

    fn read_block(&mut self, block: &mut Block, limit: usize) -> crate::table::Result<usize> {
        self.filter.clear();
        block.reset();
        let read_row = self.src.read(block, limit)?;
        if read_row == 0 {
            return Ok(0);
        }
        if block.handles.fixed_size > 0 {
            for i in 0..block.handles.length() {
                let handle = block.handles.get_int_handle_value(i);
                let version = block.versions.get_version(i);
                if version > self.truncate_ts {
                    self.filter.finish_range(i);
                    continue;
                }
                if self.end_int_handle.is_some() && handle >= self.end_int_handle.unwrap() {
                    self.filter.finish_range(i);
                    break;
                }
                self.filter.start_range(i);
            }
        } else {
            let length = block.handles.length();
            let last_handle = block.handles.get_not_null_value(length - 1);
            let check_handle = last_handle >= self.end_handle.as_slice();
            for i in 0..block.handles.length() {
                let handle = block.handles.get_not_null_value(i);
                let version = block.versions.get_version(i);
                if version > self.truncate_ts {
                    self.filter.finish_range(i);
                    continue;
                }
                if check_handle && handle >= self.end_handle.as_slice() {
                    self.filter.finish_range(i);
                    break;
                }
                self.filter.start_range(i);
            }
        }
        self.filter.finish_range(read_row);
        Ok(self.filter.do_filter(read_row, block))
    }
}

pub(crate) struct ColumnarMergeReader {
    schema: Schema,
    #[allow(clippy::vec_box)]
    heap: Vec<Box<ColumnarReaderBuffer>>,
    is_int_handle: bool,
    first_batch_end_row_idx: usize,
}

pub(crate) struct ColumnarReaderBuffer {
    reader: Box<dyn ColumnarReader>,
    block: Block,
    row_idx: usize,
}

impl ColumnarReaderBuffer {
    pub fn seek(&mut self, handle: &[u8]) -> crate::table::Result<()> {
        self.reader.seek(handle)?;
        self.reader.read(&mut self.block, 1024)?;
        self.row_idx = 0;
        Ok(())
    }

    pub fn handle(&self) -> &[u8] {
        self.block.handles.get_not_null_value(self.row_idx)
    }

    pub fn int_handle(&self) -> i64 {
        self.block.handles.get_int_handle_value(self.row_idx)
    }

    pub fn version(&self) -> u64 {
        self.block.versions.get_version(self.row_idx)
    }

    pub fn valid(&self) -> bool {
        self.row_idx < self.block.handles.length()
    }

    pub fn read_block(&mut self) -> crate::table::Result<()> {
        self.block.reset();
        self.reader.read(&mut self.block, 1024)?;
        self.row_idx = 0;
        Ok(())
    }
}

#[allow(dead_code)]
impl ColumnarMergeReader {
    pub fn new(schema: Schema, readers: Vec<Box<dyn ColumnarReader>>) -> ColumnarMergeReader {
        let heap: Vec<Box<ColumnarReaderBuffer>> = readers
            .into_iter()
            .map(|reader| {
                Box::new(ColumnarReaderBuffer {
                    reader,
                    block: Block::new(&schema),
                    row_idx: 0,
                })
            })
            .collect();
        let is_int_handle = !schema.is_common_handle();
        ColumnarMergeReader {
            schema,
            heap,
            is_int_handle,
            first_batch_end_row_idx: 0,
        }
    }

    fn init_heap(&mut self) {
        for i in (0..self.heap.len() / 2).rev() {
            self.down(i);
        }
        if !self.heap.is_empty() {
            self.update_first_block_end_row_idx();
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
        if self.is_int_handle {
            let a_handle = self.heap[a].int_handle();
            let b_handle = self.heap[b].int_handle();
            if a_handle != b_handle {
                a_handle < b_handle
            } else {
                self.heap[a].version() > self.heap[b].version()
            }
        } else {
            match self.heap[a].handle().cmp(self.heap[b].handle()) {
                Ordering::Less => true,
                Ordering::Equal => self.heap[a].version() > self.heap[b].version(),
                Ordering::Greater => false,
            }
        }
    }

    fn update_first_block_end_row_idx(&mut self) {
        let first = &self.heap[0];
        if self.heap.len() == 1 {
            self.first_batch_end_row_idx = first.block.handles.length();
            return;
        }
        let mut second_handle = if self.heap.len() >= 3 {
            if first.block.handles.fixed_size > 0 {
                if self.heap[1].int_handle() < self.heap[2].int_handle() {
                    self.heap[1].handle()
                } else {
                    self.heap[2].handle()
                }
            } else if self.heap[1].handle() < self.heap[2].handle() {
                self.heap[1].handle()
            } else {
                self.heap[2].handle()
            }
        } else {
            self.heap[1].handle()
        };
        if first.block.handles.fixed_size > 0 {
            let int_second_handle = second_handle.get_i64_le();
            for i in first.row_idx + 1..first.block.handles.length() {
                let handle = first.block.handles.get_int_handle_value(i);
                if handle >= int_second_handle {
                    self.first_batch_end_row_idx = i;
                    return;
                }
            }
        } else {
            for i in first.row_idx + 1..first.block.handles.length() {
                if first.block.handles.get_not_null_value(i) >= second_handle {
                    self.first_batch_end_row_idx = i;
                    return;
                }
            }
        }
        self.first_batch_end_row_idx = first.block.handles.length();
    }
}

impl ColumnarReader for ColumnarMergeReader {
    fn schema(&self) -> &Schema {
        &self.schema
    }

    fn seek(&mut self, handle: &[u8]) -> crate::table::Result<()> {
        for reader in &mut self.heap {
            reader.seek(handle)?;
        }
        self.heap.retain(|r| r.valid());
        if !self.heap.is_empty() {
            self.init_heap();
        }
        Ok(())
    }

    fn read(&mut self, block: &mut Block, limit: usize) -> crate::table::Result<usize> {
        let mut read_row = 0;
        while read_row < limit {
            if self.heap.is_empty() {
                return Ok(read_row);
            }
            let first = &mut self.heap[0];
            let remain = min(
                limit - read_row,
                self.first_batch_end_row_idx - first.row_idx,
            );
            block.append(&first.block, first.row_idx, first.row_idx + remain);
            first.row_idx += remain;
            read_row += remain;
            if first.row_idx == first.block.handles.length() {
                first.read_block()?;
                if !first.valid() {
                    self.heap.swap_remove(0);
                    if self.heap.is_empty() {
                        return Ok(read_row);
                    }
                }
                self.down(0);
                self.update_first_block_end_row_idx();
            } else if first.row_idx == self.first_batch_end_row_idx {
                self.down(0);
                self.update_first_block_end_row_idx();
            }
        }
        Ok(read_row)
    }
}

fn parse_default_val(col_info: &ColumnInfo) -> Option<Vec<u8>> {
    let mut default_val = col_info.get_default_val();
    if default_val.is_empty() {
        return None;
    }
    let flag = default_val.get_u8();
    let val = match flag {
        INT_FLAG | DURATION_FLAG => decode_i64(&mut default_val).unwrap().to_le_bytes().to_vec(),
        UINT_FLAG => decode_u64(&mut default_val).unwrap().to_le_bytes().to_vec(),
        BYTES_FLAG => decode_bytes(&mut default_val, false).unwrap(),
        COMPACT_BYTES_FLAG => decode_compact_bytes(&mut default_val).unwrap(),
        NIL_FLAG => {
            return None;
        }
        FLOAT_FLAG => decode_f64(&mut default_val).unwrap().to_le_bytes().to_vec(),
        DECIMAL_FLAG | JSON_FLAG => default_val.to_vec(),
        VAR_INT_FLAG => decode_var_i64(&mut default_val)
            .unwrap()
            .to_le_bytes()
            .to_vec(),
        VAR_UINT_FLAG => decode_var_u64(&mut default_val)
            .unwrap()
            .to_le_bytes()
            .to_vec(),
        VECTOR_FLOAT32_FLAG => {
            warn!("unimplemented vector default");
            return None;
        }
        f => {
            warn!("parse default val get unknown flag {}", f);
            return None;
        }
    };
    Some(val)
}

pub struct ColumnarRowTableReader {
    schema: Schema,
    iter: Box<dyn table::Iterator>,
    blob_tbls: Option<Arc<HashMap<u64, BlobTable>>>,
    prefix: Vec<u8>,
    default_vals: Vec<Option<Vec<u8>>>,
    is_int_handle: bool,
    check_schema: bool,
    keyspace_id: u32,
    max_col_id: i32,
    inner_key_off: usize,
    encryption_key: Option<EncryptionKey>,
    decryption_buf: Vec<u8>,
}

impl ColumnarRowTableReader {
    pub fn new(
        keyspace_id: u32,
        inner_key_off: usize,
        schema: Schema,
        iter: Box<dyn table::Iterator>,
        blob_tbls: Option<Arc<HashMap<u64, BlobTable>>>,
        check_schema: bool,
        encryption_key: Option<EncryptionKey>,
    ) -> ColumnarRowTableReader {
        let mut prefix = if inner_key_off > 0 {
            encode_row_key(schema.table_id, 0)
        } else {
            let mut keyspace_prefix = api_version::ApiV2::get_txn_keyspace_prefix(keyspace_id);
            let row_key = encode_row_key(schema.table_id, 0);
            keyspace_prefix.extend_from_slice(&row_key);
            keyspace_prefix
        };
        prefix.truncate(if inner_key_off > 0 {
            PREFIX_LEN
        } else {
            KEYSPACE_ID_LEN + PREFIX_LEN
        });
        let is_int_handle = get_fixed_size(&schema.handle_column) > 0;
        let default_vals = schema
            .columns
            .iter()
            .map(|col| parse_default_val(col))
            .collect();
        let max_col_id = schema
            .columns
            .iter()
            .map(|col| col.get_column_id())
            .max()
            .unwrap_or_default() as i32;
        ColumnarRowTableReader {
            keyspace_id,
            schema,
            iter,
            blob_tbls,
            prefix,
            default_vals,
            is_int_handle,
            check_schema,
            max_col_id,
            inner_key_off,
            encryption_key,
            decryption_buf: vec![],
        }
    }

    fn decode_row_columns(&self, block: &mut Block, row_value: &[u8]) -> table::Result<()> {
        let row_slice = RowSlice::from_bytes(row_value).unwrap();
        if self.check_schema {
            let row_max_col_id = row_slice.max_col_id();
            if self.max_col_id < row_max_col_id {
                let err_info = format!(
                    "ks:{} tbl:{} col:{}",
                    self.keyspace_id, self.schema.table_id, row_max_col_id
                );
                return Err(table::Error::SchemaOutOfDate(err_info));
            }
        }
        let values = row_slice.values();
        for (offset, col_info) in self.schema.columns.iter().enumerate() {
            let col_id = col_info.get_column_id();
            let col_buf = &mut block.columns[offset];
            if row_slice.search_in_null_ids(col_id) {
                col_buf.push_null();
                continue;
            }
            if let Some((start, end)) = row_slice.search_in_non_null_ids(col_id).unwrap() {
                let col_val = &values[start..end];
                Self::push_col_buf_with_field_type(col_buf, col_info, col_val);
            } else if !self.is_int_handle && get_primary_key(col_info) {
                // get value from common handle
                let mut common_handle =
                    block.handles.get_not_null_value(block.handles.length() - 1);
                for &pk_col_id in &self.schema.pk_col_ids {
                    let (datum, remain) = datum::split_datum(common_handle, false).unwrap();
                    if pk_col_id == col_id {
                        Self::push_col_buf_with_datum(col_buf, col_info, datum);
                        break;
                    }
                    common_handle = remain;
                }
            } else if let Some(default_val) = &self.default_vals[offset] {
                col_buf.push_value(default_val);
            } else {
                col_buf.push_null();
            }
        }
        Ok(())
    }

    fn is_unsigned(col_info: &ColumnInfo) -> bool {
        let flag = FieldTypeFlag::from_bits(col_info.get_flag() as u32).unwrap();
        flag.contains(FieldTypeFlag::UNSIGNED)
    }

    fn push_col_buf_with_field_type(
        col_buf: &mut ColumnBuffer,
        col_info: &ColumnInfo,
        col_val: &[u8],
    ) {
        let ft = FieldTypeTp::from_u8(col_info.get_tp() as u8).unwrap();
        match ft {
            FieldTypeTp::Tiny
            | FieldTypeTp::Short
            | FieldTypeTp::Int24
            | FieldTypeTp::Long
            | FieldTypeTp::LongLong => {
                if Self::is_unsigned(col_info) {
                    let v = decode_v2_u64(col_val).unwrap();
                    col_buf.push_value(&v.to_le_bytes());
                } else {
                    let v = decode_v2_i64(col_val).unwrap();
                    col_buf.push_value(&v.to_le_bytes());
                }
            }
            FieldTypeTp::Date
            | FieldTypeTp::DateTime
            | FieldTypeTp::Timestamp
            | FieldTypeTp::Enum
            | FieldTypeTp::Bit
            | FieldTypeTp::Set => {
                let v = decode_v2_u64(col_val).unwrap();
                col_buf.push_value(&v.to_le_bytes());
            }
            FieldTypeTp::Year | FieldTypeTp::Duration => {
                let v = decode_v2_i64(col_val).unwrap();
                col_buf.push_value(&v.to_le_bytes());
            }
            FieldTypeTp::Float | FieldTypeTp::Double => {
                let mut val = col_val;
                let v = decode_f64(&mut val).unwrap();
                col_buf.push_value(&v.to_le_bytes());
            }
            FieldTypeTp::Null => {
                col_buf.push_null();
            }
            FieldTypeTp::Unspecified
            | FieldTypeTp::NewDate
            | FieldTypeTp::VarChar
            | FieldTypeTp::Json
            | FieldTypeTp::NewDecimal
            | FieldTypeTp::TinyBlob
            | FieldTypeTp::MediumBlob
            | FieldTypeTp::LongBlob
            | FieldTypeTp::Blob
            | FieldTypeTp::VarString
            | FieldTypeTp::String
            | FieldTypeTp::Geometry
            | FieldTypeTp::TiDbVectorFloat32 => {
                col_buf.push_value(col_val);
            }
        }
    }

    fn push_col_buf_with_datum(
        col_buf: &mut ColumnBuffer,
        col_info: &ColumnInfo,
        mut datum: &[u8],
    ) {
        let flag = datum.get_u8();
        match flag {
            INT_FLAG | DURATION_FLAG => {
                let v = decode_i64(&mut datum).unwrap();
                col_buf.push_value(&v.to_le_bytes());
            }
            UINT_FLAG => {
                let v = decode_u64(&mut datum).unwrap();
                col_buf.push_value(&v.to_le_bytes());
            }
            BYTES_FLAG => {
                let v = decode_bytes(&mut datum, false).unwrap();
                col_buf.push_value(&v);
            }
            NIL_FLAG => {
                col_buf.push_null();
            }
            FLOAT_FLAG => {
                let v = decode_f64(&mut datum).unwrap();
                col_buf.push_value(&v.to_le_bytes());
            }
            DECIMAL_FLAG => {
                let v = datum.read_decimal().unwrap();
                let mut buf = vec![];
                let prec = col_info.get_column_len() as u8;
                let frac = col_info.get_decimal() as u8;
                buf.write_decimal(&v, prec, frac).unwrap();
                col_buf.push_value(&buf);
            }
            JSON_FLAG | VECTOR_FLOAT32_FLAG | VAR_UINT_FLAG | VAR_INT_FLAG | COMPACT_BYTES_FLAG => {
                unreachable!("invalid flag {} in common handle", flag)
            }
            _ => {
                unreachable!("unknown flag {} in common handle", flag)
            }
        }
    }
}

impl ColumnarReader for ColumnarRowTableReader {
    fn schema(&self) -> &Schema {
        &self.schema
    }

    fn seek(&mut self, handle: &[u8]) -> table::Result<()> {
        let row_key = if self.is_int_handle && !handle.is_empty() {
            encode_row_key(self.schema.table_id, (&handle[..]).get_i64_le())
        } else {
            encode_common_handle_for_test(self.schema.table_id, handle)
        };
        self.iter.seek(InnerKey::from_inner_buf(&row_key));
        Ok(())
    }

    fn read(&mut self, block: &mut Block, limit: usize) -> table::Result<usize> {
        let mut read_rows = 0;
        while self.iter.valid() && read_rows < limit {
            let key = self.iter.key();
            if !key.deref().starts_with(&self.prefix) {
                break;
            }
            let table_key = if self.inner_key_off == 0 && key[0] == TXN_KEY_PREFIX {
                &key.deref()[4..]
            } else {
                key.deref()
            };
            if self.is_int_handle {
                let int_handle = decode_int_handle(table_key).unwrap();
                block.handles.push_value(&int_handle.to_le_bytes());
            } else {
                let common_handle = decode_common_handle(table_key).unwrap();
                block.handles.push_value(common_handle);
            }
            let value = self.iter.value();
            let version = value.version;
            assert_eq!(self.schema.txn_id_column.is_some(), block.txn_ids.is_some());
            if let Some(txn_id_col) = block.txn_ids.as_mut() {
                let user_meta = value.user_meta();
                let txn_id = if user_meta.is_empty() {
                    0u64
                } else {
                    let um = UserMeta::from_slice(value.user_meta());
                    um.start_ts
                };
                txn_id_col.push_value(&txn_id.to_le_bytes());
            }
            let mut handle_row_value = |reader: &mut Self, row_value: &[u8]| -> table::Result<()> {
                let is_deleted = row_value.is_empty();
                block.versions.push_version(version, is_deleted);
                if is_deleted {
                    for col in &mut block.columns {
                        if col.nullable {
                            col.push_null();
                        } else {
                            col.push_zero();
                        }
                    }
                } else {
                    reader.decode_row_columns(block, row_value)?;
                }
                Ok(())
            };
            if value.is_blob_ref() {
                let blob_ref = value.get_blob_ref();
                let blob_tbl = self.blob_tbls.as_ref().unwrap().get(&blob_ref.fid).unwrap();
                let row_value = blob_tbl
                    .get(
                        &blob_ref,
                        &mut self.decryption_buf,
                        self.encryption_key.clone(),
                    )
                    .unwrap();
                handle_row_value(self, &row_value)?
            } else {
                let row_value = value.get_value();
                handle_row_value(self, row_value)?
            };
            read_rows += 1;
            self.iter.next_all_version();
        }
        Ok(read_rows)
    }
}

pub struct ColumnarConcatReader {
    schema: Schema,
    files: Vec<ColumnarFile>,
    reader: Option<ColumnarTableReader>,
    idx: usize,
    encryption_key: Option<EncryptionKey>,
}

impl ColumnarConcatReader {
    pub fn new(
        files: &[ColumnarFile],
        schema: Schema,
        encryption_key: Option<EncryptionKey>,
    ) -> ColumnarConcatReader {
        let mut reader_files = vec![];
        for file in files {
            if file.has_table(schema.table_id) {
                reader_files.push(file.clone());
            }
        }
        ColumnarConcatReader {
            schema,
            files: reader_files,
            reader: None,
            idx: 0,
            encryption_key,
        }
    }
}

impl ColumnarReader for ColumnarConcatReader {
    fn schema(&self) -> &Schema {
        &self.schema
    }

    fn seek(&mut self, handle: &[u8]) -> table::Result<()> {
        if self.files.is_empty() {
            return Ok(());
        }
        let mut row_key = if get_fixed_size(&self.schema.handle_column) > 0 {
            encode_row_key(self.schema.table_id, (&handle[..]).get_i64_le())
        } else {
            encode_common_handle_for_test(self.schema.table_id, handle)
        };
        if let Some(prefix) = ApiV2::get_keyspace_prefix(self.files[0].get_smallest().deref()) {
            let mut buf = Vec::with_capacity(prefix.len() + row_key.len());
            buf.extend_from_slice(prefix);
            buf.extend_from_slice(&row_key);
            row_key = buf;
        }
        for (i, file) in self.files.iter().enumerate() {
            if file.get_biggest().deref() < row_key.as_slice() {
                continue;
            }
            let mut reader =
                ColumnarTableReader::new(file, self.schema.clone(), self.encryption_key.clone());
            reader.seek(handle)?;
            self.reader = Some(reader);
            self.idx = i;
            return Ok(());
        }
        self.idx = self.files.len();
        Ok(())
    }

    fn read(&mut self, block: &mut Block, limit: usize) -> table::Result<usize> {
        let mut read_rows = 0;
        while self.idx < self.files.len() && read_rows < limit {
            let reader = self.reader.as_mut().unwrap();
            let cnt = reader.read(block, limit - read_rows)?;
            if cnt == 0 {
                self.idx += 1;
                if self.idx < self.files.len() {
                    let file = &self.files[self.idx];
                    let mut reader = ColumnarTableReader::new(
                        file,
                        self.schema.clone(),
                        self.encryption_key.clone(),
                    );
                    reader.seek(&[])?;
                    self.reader = Some(reader);
                }
                continue;
            }
            read_rows += cnt;
        }
        Ok(read_rows)
    }
}

#[cfg(test)]
pub mod tests {
    use std::sync::Arc;

    use proptest::{arbitrary::any, proptest};
    use rand::Rng;
    use rstest::rstest;
    use test_util::init_log_for_test;
    use tidb_query_datatype::{
        codec::row::v2::encoder_for_test::{Column, RowEncoder},
        expr::EvalContext,
        Collation::Utf8Mb4GeneralCi,
        FieldTypeTp,
    };

    use super::*;
    use crate::{
        table::{
            columnar::{
                builder::{
                    new_common_handle_column_info, new_int_handle_column_info,
                    new_txn_id_column_info, new_version_column_info, ColumnarFileBuilder,
                    ColumnarTableBuildOptions, ColumnarTableBuilder,
                },
                columnar::ColumnarFile,
                reader::{ColumnarMvccReader, ColumnarReader, ColumnarTableReader},
                SchemaBuf,
            },
            file::{File, InMemFile},
            memtable::{CfTable, WriteBatch},
        },
        UserMeta, WRITE_CF,
    };

    #[derive(Default, Clone)]
    pub struct RefRow {
        handle: Vec<u8>,
        is_common_handle: bool,
        version: u64,
        txn_id: u64,
        is_deleted: bool,
        c0: Option<i64>,
        c1: Option<Vec<u8>>,
    }

    impl RefRow {
        pub fn in_bound(&self, mut start: &[u8], mut end: &[u8]) -> bool {
            if self.is_common_handle {
                self.handle.as_slice() >= start && self.handle.as_slice() < end
            } else {
                let handle = self.handle.as_slice().get_i64_le();
                handle >= start.get_i64_le() && handle < end.get_i64_le()
            }
        }

        pub fn cmp(&self, other: &Self) -> Ordering {
            let order = if self.is_common_handle {
                self.handle.cmp(&other.handle)
            } else {
                self.handle
                    .as_slice()
                    .get_i64_le()
                    .cmp(&other.handle.as_slice().get_i64_le())
            };
            if order.is_eq() {
                other.version.cmp(&self.version)
            } else {
                order
            }
        }
    }

    fn new_column_info(col_id: i64) -> ColumnInfo {
        let mut col_info = ColumnInfo::new();
        col_info.set_column_id(col_id);
        col_info
    }

    pub fn new_schema(table_id: i64, common_handle: bool) -> Schema {
        let mut schema_buf = SchemaBuf::default();
        schema_buf.table_id = table_id;
        if common_handle {
            schema_buf.handle_column = new_common_handle_column_info();
        } else {
            schema_buf.handle_column = new_int_handle_column_info();
        }
        schema_buf.version_column = new_version_column_info();
        schema_buf.txn_id_column = Some(new_txn_id_column_info());
        let mut col_1 = new_column_info(1);
        col_1.set_tp(FieldTypeTp::LongLong as i32);
        let mut col_2 = new_column_info(2);
        col_2.set_tp(FieldTypeTp::VarChar as i32);
        col_2.set_collation(Utf8Mb4GeneralCi as i32);
        schema_buf.columns.push(col_1);
        schema_buf.columns.push(col_2);
        Schema::new(schema_buf)
    }

    pub fn build_table(
        file_id: u64,
        enable_inner_key_off: bool,
        schema: &Schema,
        start: i32,
        end: i32,
        version: u64,
    ) -> (Arc<dyn File>, Vec<RefRow>) {
        build_table_with_encryption(
            file_id,
            enable_inner_key_off,
            schema,
            start,
            end,
            version,
            None,
        )
    }

    fn new_test_encryption_key() -> EncryptionKey {
        EncryptionKey::new(b"cipher".to_vec(), b"plain".to_vec(), 0)
    }

    pub fn build_table_with_encryption(
        file_id: u64,
        enable_inner_key_off: bool,
        schema: &Schema,
        start: i32,
        end: i32,
        version: u64,
        encryption_key: Option<EncryptionKey>,
    ) -> (Arc<dyn File>, Vec<RefRow>) {
        let mut rng = rand::thread_rng();
        let mut ref_rows: Vec<RefRow> = vec![];
        let inner_key_off = if enable_inner_key_off { 4 } else { 0 };
        let keyspace_id = 1;
        let keyspace_prefix = api_version::ApiV2::get_txn_keyspace_prefix(keyspace_id);
        let is_common_handle = get_fixed_size(&schema.handle_column) == 0;
        for i in start..end {
            ref_rows.push(new_ref_row(i, version, is_common_handle));
            if rng.gen_ratio(1, 8) {
                ref_rows.push(new_ref_row(i, version - 1, is_common_handle));
            }
            if rng.gen_ratio(1, 16) {
                ref_rows.push(new_ref_row(i, version - 2, is_common_handle));
            }
        }
        let mut eval_ctx = EvalContext::default();
        let mut wb = WriteBatch::new();
        for ref_row in ref_rows.iter().rev() {
            let row_key = if is_common_handle {
                encode_common_handle_for_test(schema.table_id, ref_row.handle.as_slice())
            } else {
                encode_row_key(schema.table_id, ref_row.handle.as_slice().get_i64_le())
            };
            let inner_buf = if enable_inner_key_off {
                row_key
            } else {
                [keyspace_prefix.clone(), row_key].concat()
            };
            let inner_key = InnerKey::from_inner_buf(&inner_buf);
            let mut row_val = vec![];
            let cols = vec![
                Column::new(1, ref_row.c0),
                Column::new(2, ref_row.c1.clone()),
            ];
            if !ref_row.is_deleted {
                row_val.write_row(&mut eval_ctx, cols).unwrap();
            }
            let user_meta = UserMeta::new(ref_row.txn_id, ref_row.version).to_array();
            wb.put(inner_key, 0, &user_meta, ref_row.version, &row_val);
        }
        let cf_tbl = CfTable::new();
        cf_tbl.get_cf(WRITE_CF).put_batch(&mut wb, None, WRITE_CF);
        let iter = cf_tbl.get_cf(WRITE_CF).new_iterator(false);
        let mut row_tbl_reader =
            ColumnarRowTableReader::new(1, inner_key_off, schema.clone(), iter, None, false, None);
        let mut block = Block::new(schema);
        let mut opts = ColumnarTableBuildOptions::default();
        opts.pack_max_row_count = 8;
        opts.pack_max_size = 256;
        let mut table_builder =
            ColumnarTableBuilder::new(schema.clone(), opts, true, encryption_key, file_id);
        row_tbl_reader.seek(&ref_rows[0].handle).unwrap();
        let mut append_rows = 0;
        let mut block_off = 0;
        while append_rows < ref_rows.len() {
            if block_off == block.length() {
                block_off = 0;
                block.reset();
                let limit = rng.gen_range(2..10);
                let read = row_tbl_reader.read(&mut block, limit).unwrap();
                if read == 0 {
                    break;
                }
            }
            block_off = table_builder.append_block(&block, block_off);
            append_rows += block.length() - block_off;
        }
        let mut file_builder = ColumnarFileBuilder::new(
            file_id,
            keyspace_id,
            inner_key_off,
            Some(version), // use version as l0_version for test
            None,
        );
        file_builder.add_table(table_builder);
        let file_data = file_builder.build();
        (
            Arc::new(InMemFile::new(file_id, file_data.into())),
            ref_rows,
        )
    }

    fn new_ref_row(i: i32, version: u64, is_common_handle: bool) -> RefRow {
        let mut ref_row = RefRow::default();
        ref_row.is_common_handle = is_common_handle;
        if is_common_handle {
            ref_row.handle = i_to_common_handle(i);
        } else {
            ref_row.handle = (i as i64).to_le_bytes().to_vec();
        }
        ref_row.version = version;
        ref_row.txn_id = version - 1;
        let mut rng = rand::thread_rng();
        let is_delete = rng.gen_ratio(1, 10);
        ref_row.is_deleted = is_delete;
        let is_null = is_delete || rng.gen_ratio(1, 5);
        if !is_null {
            ref_row.c0 = Some(i as i64);
            ref_row.c1 = Some(i_to_string_col(i, version));
        }
        ref_row
    }

    fn i_to_string_col(i: i32, version: u64) -> Vec<u8> {
        let repeat = 1 + i % 16;
        format!("abc_{}_{}", i, version)
            .repeat(repeat as usize)
            .into_bytes()
    }

    pub fn i_to_common_handle(i: i32) -> Vec<u8> {
        format!("{:08x}", i).into_bytes()
    }

    #[test]
    fn test_columnar_builder() {
        init_log_for_test();
        for common_handle in [true, true] {
            let schema = new_schema(1, common_handle);
            let (file, ref_rows) = build_table(1, true, &schema, 100, 150, 100);
            let columnar_file = ColumnarFile::open(file).unwrap();
            let mut reader = ColumnarTableReader::new(&columnar_file, schema.clone(), None);
            reader.seek(&0u64.to_le_bytes()).unwrap();
            let mut block = Block::new(&schema);
            reader.read(&mut block, 100).unwrap();
            verify_with_ref_rows(&block, &ref_rows);
        }
    }

    fn new_mvcc_reader(
        schema: &Schema,
        files: &[Arc<dyn File>],
        read_ts: u64,
        encryption_key: Option<EncryptionKey>,
    ) -> ColumnarMvccReader {
        let mut readers: Vec<Box<dyn ColumnarReader>> = vec![];
        for file in files {
            let columnar_file = ColumnarFile::open(file.clone()).unwrap();
            let reader =
                ColumnarTableReader::new(&columnar_file, schema.clone(), encryption_key.clone());
            readers.push(Box::new(reader));
        }
        let merge_reader = ColumnarMergeReader::new(schema.clone(), readers);
        ColumnarMvccReader::new(Box::new(merge_reader), schema, read_ts)
    }

    fn new_compact_reader(
        schema: &Schema,
        level: u32,
        files: &[Arc<dyn File>],
        safe_ts: u64,
    ) -> ColumnarCompactReader {
        let mut readers: Vec<Box<dyn ColumnarReader>> = vec![];
        for file in files {
            let columnar_file = ColumnarFile::open(file.clone()).unwrap();
            if !columnar_file.has_table(schema.table_id) {
                continue;
            }
            let reader = ColumnarTableReader::new(&columnar_file, schema.clone(), None);
            readers.push(Box::new(reader));
        }
        let merge_reader = ColumnarMergeReader::new(schema.clone(), readers);
        ColumnarCompactReader::new(Box::new(merge_reader), level, schema, safe_ts)
    }

    pub fn verify_with_ref_rows(block: &Block, ref_rows: &[RefRow]) {
        for i in 0..ref_rows.len() {
            let ref_row = &ref_rows[i];
            assert_eq!(ref_row.handle, block.handles.get_not_null_value(i), "{}", i);
            assert_eq!(ref_row.version, block.versions.get_version(i));
            assert_eq!(ref_row.is_deleted, block.versions.is_null(i));
            assert_eq!(ref_row.c0.is_none(), block.columns[0].is_null(i));
            assert_eq!(ref_row.c1.is_none(), block.columns[1].is_null(i));
            if let Some(c0) = ref_row.c0 {
                assert_eq!(c0, block.columns[0].get_value(i).unwrap().get_i64_le());
            }
            if let Some(c1) = &ref_row.c1 {
                assert_eq!(c1.as_slice(), block.columns[1].get_value(i).unwrap());
            }
        }
    }

    pub fn merge_refs(
        refs: Vec<Vec<RefRow>>,
        level: u32,
        read_ts: Option<u64>,
        safe_ts: Option<u64>,
        bound: Option<(Vec<u8>, Vec<u8>)>,
    ) -> Vec<RefRow> {
        let mut merged = vec![];
        for ref_rows in refs {
            merged.extend_from_slice(&ref_rows);
        }
        if let Some(read_ts) = read_ts {
            merged.retain(|r| r.version <= read_ts);
        }
        if let Some((start, end)) = bound {
            merged.retain(|r| r.in_bound(&start, &end));
        }
        merged.sort_by(|a, b| a.cmp(b));
        if read_ts.is_some() {
            merged.dedup_by(|a, b| a.handle == b.handle);
            merged.retain(|r| !r.is_deleted);
        }
        if safe_ts.is_some() {
            let mut new_merged = vec![];
            let mut prev_handle = vec![];
            for r in merged.iter_mut() {
                let handle = r.handle.clone();
                let version = r.version;
                if prev_handle == handle && version < safe_ts.unwrap() {
                    continue;
                }
                if level == 2 && version < safe_ts.unwrap() && r.is_deleted {
                    prev_handle = handle;
                    continue;
                }
                prev_handle = handle;
                new_merged.push(r.clone());
            }
            merged = new_merged;
        }
        merged
    }

    #[rstest]
    #[case::enable_encryption(true)]
    #[case::disable_encryption(false)]
    fn test_reader(#[case] enable_encryption: bool) {
        init_log_for_test();
        let encryption_key = if enable_encryption {
            Some(new_test_encryption_key())
        } else {
            None
        };
        let mut options = vec![];
        let mut rng = rand::thread_rng();
        for _ in 0..100 {
            options.push(rng.gen_bool(0.5))
        }
        for common_handle in options {
            let schema = new_schema(1, common_handle);
            let (file_1, ref_1) = build_table_with_encryption(
                1,
                true,
                &schema,
                100,
                150,
                100,
                encryption_key.clone(),
            );
            let (file_2, ref_2) = build_table_with_encryption(
                2,
                true,
                &schema,
                140,
                190,
                110,
                encryption_key.clone(),
            );
            let (file_3, ref_3) = build_table_with_encryption(
                3,
                true,
                &schema,
                185,
                240,
                120,
                encryption_key.clone(),
            );
            let files = vec![file_1, file_2, file_3];
            let ref_rows = vec![ref_1, ref_2, ref_3];
            let mut block = Block::new(&schema);
            let mut rng = rand::thread_rng();
            for read_ts in [90, 100, 110, 120] {
                let start_handle = rng.gen_range(90i64..170i64);
                let end_handle = start_handle + rng.gen_range(1i64..200i64);
                let mut mvcc_reader =
                    new_mvcc_reader(&schema, &files, read_ts, encryption_key.clone());
                if common_handle {
                    let common_start_handle = i_to_common_handle(start_handle as i32);
                    let common_end_handle = i_to_common_handle(end_handle as i32);
                    mvcc_reader
                        .set_handle_range(&common_start_handle, &common_end_handle)
                        .unwrap();
                } else {
                    mvcc_reader
                        .set_int_handle_range(start_handle, Some(end_handle))
                        .unwrap();
                }
                mvcc_reader.read_block(&mut block, 500).unwrap();
                let range_bound = if common_handle {
                    (
                        i_to_common_handle(start_handle as i32),
                        i_to_common_handle(end_handle as i32),
                    )
                } else {
                    (
                        start_handle.to_le_bytes().to_vec(),
                        end_handle.to_le_bytes().to_vec(),
                    )
                };
                let merged_refs =
                    merge_refs(ref_rows.clone(), 0, Some(read_ts), None, Some(range_bound));
                verify_with_ref_rows(&block, &merged_refs);
            }
        }
    }

    proptest! {
        #[test]
        fn test_concat_reader(
            common_handle in any::<bool>(),
        ) {
            init_log_for_test();
            let schema = new_schema(1, common_handle);
            let (file_1, ref_1) = build_table(1, true, &schema, 100, 150, 100);
            let (file_2, ref_2) = build_table(2, true, &schema, 160, 190, 100);
            let (file_3, ref_3) = build_table(3, true, &schema, 191, 240, 100);
            let files = vec![file_1, file_2, file_3];
            let ref_rows = vec![ref_1, ref_2, ref_3];
            let col_files: Vec<ColumnarFile> = files.iter().map(|f| ColumnarFile::open(f.clone()).unwrap()).collect();
            for _ in 0..50 {
                let mut rng = rand::thread_rng();
                let start_handle = rng.gen_range(90i64..170i64);
                let end_handle = start_handle + rng.gen_range(1i64..200i64);
                let mut block = Block::new(&schema);
                let concat_reader = ColumnarConcatReader::new(&col_files, schema.clone(), None);
                let mut mvcc_reader = ColumnarMvccReader::new(Box::new(concat_reader), &schema, 100);
                if common_handle {
                    let common_start_handle = i_to_common_handle(start_handle as i32);
                    let common_end_handle = i_to_common_handle(end_handle as i32);
                    mvcc_reader
                        .set_handle_range(&common_start_handle, &common_end_handle)
                        .unwrap();
                } else {
                    mvcc_reader
                        .set_int_handle_range(start_handle, Some(end_handle))
                        .unwrap();
                }
                mvcc_reader.read_block(&mut block, 500).unwrap();
                let range_bound = if common_handle {
                    (
                        i_to_common_handle(start_handle as i32),
                        i_to_common_handle(end_handle as i32),
                    )
                } else {
                    (
                        start_handle.to_le_bytes().to_vec(),
                        end_handle.to_le_bytes().to_vec(),
                    )
                };
                let merged_refs = merge_refs(ref_rows.clone(), 0, Some(100), None, Some(range_bound));
                verify_with_ref_rows(&block, &merged_refs);
            }
        }
    }

    proptest! {
        #[test]
        fn test_compact_reader(
            common_handle in any::<bool>(),
            enable_inner_key_off in any::<bool>(),
        ) {
            init_log_for_test();
            let schema = new_schema(1, common_handle);
            let (file_1, ref_1) = build_table(1, enable_inner_key_off, &schema, 100, 200, 100);
            let (file_2, ref_2) = build_table(2, enable_inner_key_off, &schema, 150, 250, 200);
            let (file_3, ref_3) = build_table(3, enable_inner_key_off, &schema, 200, 300, 300);
            let files = vec![file_1, file_2, file_3];
            let ref_rows = vec![ref_1, ref_2, ref_3];
            for _ in 0..50 {
                let mut rng = rand::thread_rng();
                let start_handle = rng.gen_range(90i64..170i64);
                let end_handle = start_handle + rng.gen_range(1i64..200i64);
                let mut block = Block::new(&schema);
                let mut compact_reader = new_compact_reader(&schema, 2, &files, 250);
                if common_handle {
                    let common_start_handle = i_to_common_handle(start_handle as i32);
                    let common_end_handle = i_to_common_handle(end_handle as i32);
                    compact_reader
                        .set_handle_range(&common_start_handle, &common_end_handle)
                        .unwrap();
                } else {
                    compact_reader
                        .set_int_handle_range(start_handle, Some(end_handle))
                        .unwrap();
                }
                compact_reader.read_block(&mut block, 1000).unwrap();
                let range_bound = if common_handle {
                    (
                        i_to_common_handle(start_handle as i32),
                        i_to_common_handle(end_handle as i32),
                    )
                } else {
                    (
                        start_handle.to_le_bytes().to_vec(),
                        end_handle.to_le_bytes().to_vec(),
                    )
                };
                let merged_refs = merge_refs(ref_rows.clone(), 2, None, Some(250), Some(range_bound));
                verify_with_ref_rows(&block, &merged_refs);
            }
        }
    }
}
