// Copyright 2024 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    cmp::{min, Ordering},
    mem,
    sync::Arc,
};

use bytes::Buf;
use tidb_query_datatype::codec::datum::{
    BYTES_FLAG, COMPACT_BYTES_FLAG, DECIMAL_FLAG, DURATION_FLAG, FLOAT_FLAG, INT_FLAG, JSON_FLAG,
    NIL_FLAG, UINT_FLAG, VAR_INT_FLAG, VAR_UINT_FLAG, VECTOR_FLOAT32_FLAG,
};
use tikv_util::codec::{
    bytes::{decode_bytes, decode_compact_bytes},
    number::{decode_f64, decode_i64, decode_u64, decode_var_i64, decode_var_u64},
};
use tipb::ColumnInfo;

use crate::table::{
    columnar::columnar::{
        decompress_pack, Block, ColumnBuffer, ColumnMeta, ColumnarFile, Schema, TableMeta,
    },
    search,
    sstable::File,
};

pub trait ColumnarReader {
    fn schema(&self) -> &Schema;
    fn seek(&mut self, handle: &[u8]) -> crate::table::Result<()>;
    fn read(&mut self, block: &mut Block, limit: usize) -> crate::table::Result<usize>;
}

pub(crate) struct ColumnarTableReader {
    table_meta: Arc<TableMeta>,
    schema: Schema,
    handle_reader: ColumnarColumnReader,
    version_reader: ColumnarColumnReader,
    columns_readers: Vec<ColumnarColumnReader>,
}

#[allow(dead_code)]
impl ColumnarTableReader {
    pub fn new(
        columnar_file: &ColumnarFile,
        table_id: i64,
        columns: Vec<ColumnInfo>,
    ) -> ColumnarTableReader {
        let table_meta = columnar_file.get_table(table_id);
        debug_assert_eq!(table_meta.table_id, table_id);
        let file = columnar_file.get_file();
        let mut schema = Schema {
            table_id,
            handle_column: table_meta.handle_column.col_info.clone(),
            version_column: table_meta.version_column.col_info.clone(),
            columns,
        };
        let handle_column_id = schema.handle_column.get_column_id();
        schema
            .columns
            .retain(|col| col.get_column_id() != handle_column_id);
        let handle_reader =
            ColumnarColumnReader::new(file.clone(), table_meta.handle_column.clone(), false);
        let version_reader =
            ColumnarColumnReader::new(file.clone(), table_meta.version_column.clone(), false);
        let columns_readers = schema
            .columns
            .iter()
            .map(|col| {
                let col_id = col.get_column_id() as i32;
                if let Some(col_meta) = table_meta.columns.get(&col_id) {
                    ColumnarColumnReader::new(file.clone(), col_meta.clone(), false)
                } else {
                    let col_meta = ColumnMeta::new(col.clone(), false);
                    ColumnarColumnReader::new(file.clone(), Arc::new(col_meta), true)
                }
            })
            .collect();
        ColumnarTableReader {
            table_meta,
            schema,
            handle_reader,
            version_reader,
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
        let row_idx_in_pack = if self.handle_reader.col_meta.fixed_size > 0 {
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
        for col_reader in &mut self.columns_readers {
            col_reader.set_row_idx(row_idx)?;
        }
        Ok(())
    }

    fn read(&mut self, block: &mut Block, limit: usize) -> crate::table::Result<usize> {
        let read_row = self.handle_reader.read(&mut block.handles, limit)?;
        self.version_reader.read(&mut block.versions, limit)?;
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
}

impl ColumnarColumnReader {
    pub(crate) fn new(
        file: Arc<dyn File>,
        col_meta: Arc<ColumnMeta>,
        is_default_val: bool,
    ) -> ColumnarColumnReader {
        let pack_loader = PackLoader::new(file);
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
        self.pack_loader
            .load_pack(&mut self.pack_buffer, pack_start, pack_end)?;
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
}

impl PackLoader {
    pub fn new(file: Arc<dyn File>) -> PackLoader {
        PackLoader {
            file,
            compressed_buf: vec![],
            uncompressed_buf: vec![],
        }
    }

    pub fn load_pack(
        &mut self,
        col_buf: &mut ColumnBuffer,
        pack_offset: u32,
        pack_end_offset: u32,
    ) -> crate::table::Result<()> {
        let length = (pack_end_offset - pack_offset) as usize;
        self.compressed_buf.resize(length, 0);
        self.file
            .read_at(&mut self.compressed_buf, pack_offset as u64)?;
        decompress_pack(&self.compressed_buf, &mut self.uncompressed_buf);
        col_buf.parse(&self.uncompressed_buf);
        Ok(())
    }
}

pub struct ColumnarMvccReader {
    src: Box<dyn ColumnarReader>,
    read_ts: u64,
    ranges: Vec<(usize, usize)>,
    in_range: bool,
    range_start: usize,
    filter_block: Block,
    end_handle: Vec<u8>,
}

#[allow(dead_code)]
impl ColumnarMvccReader {
    pub fn new(src: Box<dyn ColumnarReader>, schema: &Schema, read_ts: u64) -> ColumnarMvccReader {
        ColumnarMvccReader {
            src,
            filter_block: Block::new(schema),
            read_ts,
            ranges: vec![],
            in_range: false,
            range_start: 0,
            end_handle: vec![],
        }
    }
}

impl ColumnarMvccReader {
    pub fn set_handle_range(
        &mut self,
        start_handle: &[u8],
        end_handle: &[u8],
    ) -> crate::table::Result<()> {
        self.end_handle = end_handle.to_vec();
        self.src.seek(start_handle)?;
        self.ranges.clear();
        Ok(())
    }

    pub fn read_block(&mut self, block: &mut Block, limit: usize) -> crate::table::Result<usize> {
        self.ranges.clear();
        self.in_range = false;
        block.reset();
        let read_row = self.src.read(block, limit)?;
        if block.handles.fixed_size > 0 {
            let end_handle = self.end_handle.as_slice().get_i64_le();
            let mut prev_handle = 0;
            for i in 0..block.handles.length() {
                let handle = block.handles.get_int_handle_value(i);
                let version = block.versions.get_version(i);
                if version > self.read_ts || handle == prev_handle {
                    self.finish_range(i);
                    continue;
                }
                if block.versions.is_null(i) {
                    prev_handle = handle;
                    self.finish_range(i);
                    continue;
                }
                if handle >= end_handle {
                    self.finish_range(i);
                    break;
                }
                self.start_range(i);
                prev_handle = handle;
            }
        } else {
            let mut prev_handle = [].as_slice();
            let length = block.handles.length();
            let last_handle = block.handles.get_not_null_value(length - 1);
            let check_handle = last_handle >= self.end_handle.as_slice();
            for i in 0..block.handles.length() {
                let handle = block.handles.get_not_null_value(i);
                let version = block.versions.get_version(i);
                if version > self.read_ts || handle == prev_handle {
                    self.finish_range(i);
                    continue;
                }
                if block.versions.is_null(i) {
                    prev_handle = handle;
                    self.finish_range(i);
                    continue;
                }
                if check_handle && handle >= self.end_handle.as_slice() {
                    self.finish_range(i);
                    break;
                }
                self.start_range(i);
                prev_handle = handle;
            }
        }
        self.finish_range(read_row);
        if self.ranges.len() == 1 {
            let (start, end) = self.ranges[0];
            if start == 0 {
                return if end == read_row {
                    // All rows are valid, no need to filter.
                    Ok(read_row)
                } else {
                    block.truncate(end);
                    Ok(end)
                };
            }
        }
        // filter invalid rows.
        self.filter_block.reset();
        let mut filtered_rows = 0;
        for &(start, end) in &self.ranges {
            self.filter_block.handles.append(&block.handles, start, end);
            self.filter_block
                .versions
                .append(&block.versions, start, end);
            for (i, col) in self.filter_block.columns.iter_mut().enumerate() {
                col.append(&block.columns[i], start, end);
            }
            filtered_rows += end - start;
        }
        mem::swap(block, &mut self.filter_block);
        Ok(filtered_rows)
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
        let is_int_handle = heap[0].block.handles.fixed_size > 0;
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
        self.update_first_block_end_row_idx();
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
            if self.heap[1].handle() < self.heap[2].handle() {
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
        self.init_heap();
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

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use bytes::Bytes;
    use rand::Rng;
    use test_util::init_log_for_test;
    use tidb_query_datatype::{Collation::Utf8Mb4GeneralCi, FieldTypeTp};

    use super::*;
    use crate::table::{
        columnar::{
            builder::{
                new_common_handle_column_info, new_int_handle_column_info, new_version_column_info,
                ColumnarFileBuilder, ColumnarTableBuilder,
            },
            columnar::ColumnarFile,
            reader::{ColumnarMvccReader, ColumnarReader, ColumnarTableReader},
        },
        sstable::InMemFile,
    };

    #[derive(Default, Clone)]
    struct RefRow {
        handle: Vec<u8>,
        is_common_handle: bool,
        version: u64,
        is_deleted: bool,
        c0: Option<i64>,
        c1: Option<Bytes>,
    }

    impl RefRow {
        fn in_bound(&self, mut start: &[u8], mut end: &[u8]) -> bool {
            if self.is_common_handle {
                self.handle.as_slice() >= start && self.handle.as_slice() < end
            } else {
                let handle = self.handle.as_slice().get_i64_le();
                handle >= start.get_i64_le() && handle < end.get_i64_le()
            }
        }

        fn cmp(&self, other: &Self) -> Ordering {
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

    fn new_schema(table_id: i64, common_handle: bool) -> Schema {
        let mut schema = Schema::default();
        schema.table_id = table_id;
        if common_handle {
            schema.handle_column = new_common_handle_column_info();
        } else {
            schema.handle_column = new_int_handle_column_info();
        }
        schema.version_column = new_version_column_info();
        let mut col_1 = new_column_info(1);
        col_1.set_tp(FieldTypeTp::LongLong as i32);
        let mut col_2 = new_column_info(2);
        col_2.set_tp(FieldTypeTp::VarChar as i32);
        col_2.set_collation(Utf8Mb4GeneralCi as i32);
        schema.columns.push(col_1);
        schema.columns.push(col_2);
        schema
    }

    fn build_table(
        file_id: u64,
        schema: &Schema,
        start: i32,
        end: i32,
        version: u64,
    ) -> (Arc<dyn File>, Vec<RefRow>) {
        let mut block = Block::new(schema);
        let mut table_builder = ColumnarTableBuilder::new(schema.clone(), 8, 256, true);
        let mut rng = rand::thread_rng();
        let mut ref_rows = vec![];
        for i in start..end {
            ref_rows.push(append_row(&mut block, i, version));
            if rng.gen_ratio(1, 8) {
                ref_rows.push(append_row(&mut block, i, version - 1));
            }
            if rng.gen_ratio(1, 16) {
                let is_tombstone = rng.gen_ratio(1, 10);
                if is_tombstone {
                    ref_rows.push(append_row(&mut block, i, 0));
                } else {
                    ref_rows.push(append_row(&mut block, i, version - 2));
                }
            }
            if block.handles.length() > rng.gen_range(4..16) {
                table_builder.add_block(&block);
                block.reset();
            }
        }
        if block.handles.length() > 0 {
            table_builder.add_block(&block);
        }
        let mut file_builder = ColumnarFileBuilder::new(file_id, None);
        file_builder.add_table(table_builder);
        let file_data = file_builder.build();
        (Arc::new(InMemFile::new(1, file_data.into())), ref_rows)
    }

    fn append_row(block: &mut Block, i: i32, version: u64) -> RefRow {
        let mut ref_row = RefRow::default();
        if block.handles.fixed_size > 0 {
            ref_row.handle = (i as i64).to_le_bytes().to_vec();
        } else {
            ref_row.handle = i_to_common_handle(i);
            ref_row.is_common_handle = true;
        }
        ref_row.version = version;
        if block.handles.fixed_size > 0 {
            block.handles.push_value(&(i as i64).to_le_bytes());
        } else {
            block.handles.push_value(&i_to_common_handle(i));
        }
        let mut rng = rand::thread_rng();
        let is_delete = version == 0 || rng.gen_ratio(1, 10);
        ref_row.is_deleted = is_delete;
        block.versions.push_version(version, is_delete);
        let is_null = is_delete || rng.gen_ratio(1, 5);
        if is_null {
            block.columns[0].push_null();
            block.columns[1].push_null();
        } else {
            ref_row.c0 = Some(i as i64);
            ref_row.c1 = Some(i_to_string_col(i, version).into());
            block.columns[0].push_value(&(i as i64).to_le_bytes());
            block.columns[1].push_value(&i_to_string_col(i, version));
        }
        ref_row
    }

    fn i_to_string_col(i: i32, version: u64) -> Vec<u8> {
        let repeat = 1 + i % 16;
        format!("abc_{}_{}", i, version)
            .repeat(repeat as usize)
            .into_bytes()
    }

    fn i_to_common_handle(i: i32) -> Vec<u8> {
        format!("{:08x}", i).into_bytes()
    }

    #[test]
    fn test_columnar_builder() {
        init_log_for_test();
        for common_handle in [true, true] {
            let schema = new_schema(1, common_handle);
            let (file, ref_rows) = build_table(1, &schema, 100, 150, 100);
            let columnar_file = ColumnarFile::open(file).unwrap();
            let mut reader = ColumnarTableReader::new(&columnar_file, 1, schema.columns.clone());
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
    ) -> ColumnarMvccReader {
        let mut readers: Vec<Box<dyn ColumnarReader>> = vec![];
        for file in files {
            let columnar_file = ColumnarFile::open(file.clone()).unwrap();
            let reader =
                ColumnarTableReader::new(&columnar_file, schema.table_id, schema.columns.clone());
            readers.push(Box::new(reader));
        }
        let merge_reader = ColumnarMergeReader::new(schema.clone(), readers);
        ColumnarMvccReader::new(Box::new(merge_reader), schema, read_ts)
    }

    fn verify_with_ref_rows(block: &Block, ref_rows: &[RefRow]) {
        if block.handles.length() != ref_rows.len() {
            println!("diff len {} {}", block.handles.length(), ref_rows.len());
            for i in 0..ref_rows.len() {
                let handle = ref_rows[i].handle.as_slice();
                println!("ref handle {:?}", handle);
            }
            for i in 0..block.handles.length() {
                let handle = block.handles.get_not_null_value(i);
                println!("block handle {:?}", handle);
            }
            panic!("diff len");
        }

        for i in 0..ref_rows.len() {
            let ref_row = &ref_rows[i];
            assert_eq!(ref_row.handle, block.handles.get_not_null_value(i), "{}", i);
            assert_eq!(ref_row.version, block.versions.get_version(i));
            assert_eq!(ref_row.is_deleted, block.versions.is_null(i));
            assert_eq!(ref_row.c0.is_none(), block.columns[0].is_null(i));
            assert_eq!(ref_row.c1.is_none(), block.columns[1].is_null(i));
            if let Some(c0) = ref_row.c0 {
                assert_eq!(
                    c0,
                    block.columns[0].get_nullable_value(i).unwrap().get_i64_le()
                );
            }
            if let Some(c1) = &ref_row.c1 {
                assert_eq!(c1.chunk(), block.columns[1].get_nullable_value(i).unwrap());
            }
        }
    }

    fn merge_refs(
        refs: Vec<Vec<RefRow>>,
        read_ts: Option<u64>,
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
        merged
    }

    #[test]
    fn test_reader() {
        init_log_for_test();
        let mut options = vec![];
        let mut rng = rand::thread_rng();
        for _ in 0..100 {
            options.push(rng.gen_bool(0.5))
        }
        for common_handle in options {
            let schema = new_schema(1, common_handle);
            let (file_1, ref_1) = build_table(1, &schema, 100, 150, 100);
            let (file_2, ref_2) = build_table(2, &schema, 140, 190, 110);
            let (file_3, ref_3) = build_table(3, &schema, 185, 240, 120);
            let files = vec![file_1, file_2, file_3];
            let ref_rows = vec![ref_1, ref_2, ref_3];
            let mut block = Block::new(&schema);
            let mut rng = rand::thread_rng();
            for read_ts in [90, 100, 110, 120] {
                let start_handle = rng.gen_range(90i64..170i64);
                let end_handle = start_handle + rng.gen_range(1i64..200i64);
                let mut mvcc_reader = new_mvcc_reader(&schema, &files, read_ts);
                if common_handle {
                    let common_start_handle = i_to_common_handle(start_handle as i32);
                    let common_end_handle = i_to_common_handle(end_handle as i32);
                    mvcc_reader
                        .set_handle_range(&common_start_handle, &common_end_handle)
                        .unwrap();
                } else {
                    mvcc_reader
                        .set_handle_range(&start_handle.to_le_bytes(), &end_handle.to_le_bytes())
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
                let merged_refs = merge_refs(ref_rows.clone(), Some(read_ts), Some(range_bound));
                verify_with_ref_rows(&block, &merged_refs);
            }
        }
    }
}
