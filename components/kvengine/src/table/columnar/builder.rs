// Copyright 2024 TiKV Project Authors. Licensed under Apache-2.0.

use std::cmp::max;

use bytes::{Buf, BufMut};
use tidb_query_datatype::{
    codec::table::{encode_common_handle_for_test, encode_row_key},
    Collation, FieldTypeFlag, FieldTypeTp,
};
use tipb::ColumnInfo;

use crate::table::{
    add_property,
    columnar::columnar::{
        compress_pack, get_unsigned, Block, ColumnBuffer, ColumnMeta, ColumnarFileFooter, Schema,
        COLUMNAR_MAGIC, HANDLE_COL_ID, TXN_ID_COL_ID, VERSION_COL_ID,
    },
    sstable::LZ4_COMPRESSION,
    ChecksumType,
};

pub const PACK_MAX_ROW_COUNT: usize = 8192;
pub const PACK_FORMAT: u16 = 1;
pub const ENCODING_TYPE_NONE: u16 = 0;
pub const MAX_COLUMNAR_LEVEL: usize = 2;

pub const PROP_KEY_SMALLEST: &str = "smallest";
pub const PROP_KEY_BIGGEST: &str = "biggest";
pub const PROP_KEY_MAX_VERSION: &str = "max_ver";
pub const PROP_KEY_SNAP_VERSION: &str = "snap_ver";

pub fn new_version_column_info() -> ColumnInfo {
    let mut col_info = ColumnInfo::new();
    col_info.set_column_id(VERSION_COL_ID as i64);
    col_info.set_tp(FieldTypeTp::LongLong as i32);
    let flag = FieldTypeFlag::UNSIGNED.bits();
    col_info.set_flag(flag as i32);
    col_info
}

pub fn new_txn_id_column_info() -> ColumnInfo {
    let mut col_info = ColumnInfo::new();
    col_info.set_column_id(TXN_ID_COL_ID as i64);
    col_info.set_tp(FieldTypeTp::LongLong as i32);
    let flag = FieldTypeFlag::UNSIGNED.bits() | FieldTypeFlag::NOT_NULL.bits();
    col_info.set_flag(flag as i32);
    col_info
}

pub fn new_int_handle_column_info() -> ColumnInfo {
    let mut col_info = ColumnInfo::new();
    col_info.set_column_id(HANDLE_COL_ID as i64);
    col_info.set_tp(FieldTypeTp::LongLong as i32);
    col_info.set_flag(FieldTypeFlag::NOT_NULL.bits() as i32);
    col_info
}

pub fn new_common_handle_column_info() -> ColumnInfo {
    let mut col_info = ColumnInfo::new();
    col_info.set_column_id(HANDLE_COL_ID as i64);
    col_info.set_tp(FieldTypeTp::VarChar as i32);
    col_info.set_collation(Collation::Binary as i32);
    col_info.set_flag(FieldTypeFlag::NOT_NULL.bits() as i32);
    col_info
}

pub struct ColumnarFileBuilder {
    pub file_id: u64,
    snap_version: Option<u64>,
    tables: Vec<ColumnarTableBuilder>,
    pub(crate) estimated_size: usize,
    pub(crate) smallest: Vec<u8>,
    pub(crate) biggest: Vec<u8>,
}

#[derive(Debug)]
pub(crate) struct TableOffset {
    pub(crate) table_id: i64,
    pub(crate) offset: u32,
    pub(crate) index_offset: u32,
    pub(crate) end_offset: u32,
}

impl TableOffset {
    fn new(table_id: i64, offset: usize, data_sizes: DataSizeTuple) -> Self {
        let index_offset = offset + data_sizes.packs_data_size;
        let end_offset = index_offset + data_sizes.index_size;
        Self {
            table_id,
            offset: offset as u32,
            index_offset: index_offset as u32,
            end_offset: end_offset as u32,
        }
    }

    pub(crate) fn parse(mut buf: &[u8]) -> Self {
        let table_id = buf.get_i64_le();
        let offset = buf.get_u32_le();
        let index_offset = buf.get_u32_le();
        let end_offset = buf.get_u32_le();
        TableOffset {
            table_id,
            offset,
            index_offset,
            end_offset,
        }
    }

    pub(crate) fn compute_size() -> usize {
        20
    }

    pub(crate) fn write_to(&self, buf: &mut Vec<u8>) {
        buf.put_i64_le(self.table_id);
        buf.put_u32_le(self.offset);
        buf.put_u32_le(self.index_offset);
        buf.put_u32_le(self.end_offset);
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq)]
#[serde(default)]
#[serde(rename_all = "kebab-case")]
pub struct ColumnarTableBuildOptions {
    pub compression_type: u8,
    pub checksum_type: u8,
    pub max_columnar_table_size: usize,
    pub pack_max_row_count: usize,
    pub pack_max_size: usize,
}

impl Default for ColumnarTableBuildOptions {
    fn default() -> Self {
        Self {
            compression_type: LZ4_COMPRESSION,
            checksum_type: ChecksumType::Crc32.value(),
            max_columnar_table_size: 32 * 1024 * 1024,
            pack_max_row_count: 8192,
            pack_max_size: 256 * 1024,
        }
    }
}

#[allow(dead_code)]
impl ColumnarFileBuilder {
    pub fn new(file_id: u64, snap_version: Option<u64>) -> Self {
        ColumnarFileBuilder {
            file_id,
            snap_version,
            estimated_size: 0,
            tables: vec![],
            smallest: vec![],
            biggest: vec![],
        }
    }

    pub fn add_table(&mut self, table: ColumnarTableBuilder) {
        self.estimated_size += table.get_estimated_size();
        self.tables.push(table);
    }

    pub fn build(&mut self) -> Vec<u8> {
        if self.tables.is_empty() {
            return vec![];
        }
        self.tables
            .sort_by(|a, b| a.schema.table_id.cmp(&b.schema.table_id));
        let mut tables_offsets = vec![];
        let mut total_size = 0;
        let mut max_version = 0;
        for table in &mut self.tables {
            table.finish_table();
            max_version = max_version.max(table.max_version);
            let table_size = table.compute_size();
            let sum_table_size = table_size.packs_data_size + table_size.index_size;
            let table_offset = TableOffset::new(table.schema.table_id, total_size, table_size);
            tables_offsets.push(table_offset);
            total_size += sum_table_size;
        }
        let mut property_buf = vec![];
        let (smallest, biggest) = self.build_smallest_biggest();
        self.smallest = smallest;
        self.biggest = biggest;
        add_property(
            &mut property_buf,
            PROP_KEY_SMALLEST.as_bytes(),
            &self.smallest,
        );
        add_property(
            &mut property_buf,
            PROP_KEY_BIGGEST.as_bytes(),
            &self.biggest,
        );
        add_property(
            &mut property_buf,
            PROP_KEY_MAX_VERSION.as_bytes(),
            &max_version.to_le_bytes(),
        );
        if let Some(snap_version) = self.snap_version {
            add_property(
                &mut property_buf,
                PROP_KEY_SNAP_VERSION.as_bytes(),
                &snap_version.to_le_bytes(),
            );
        }
        total_size += property_buf.len();
        total_size += tables_offsets.len() * TableOffset::compute_size();
        total_size += ColumnarFileFooter::compute_size();
        let mut file_buffer = Vec::with_capacity(total_size);
        for tbl in &mut self.tables {
            tbl.build(&mut file_buffer);
        }
        file_buffer.extend_from_slice(&property_buf);
        for table_offset in &tables_offsets {
            table_offset.write_to(&mut file_buffer);
        }
        let footer = ColumnarFileFooter {
            number_tables: self.tables.len() as u32,
            properties_size: property_buf.len() as u32,
            compression_type: LZ4_COMPRESSION,
            checksum_type: ChecksumType::Crc32.value(),
            format_version: 1,
            magic: COLUMNAR_MAGIC,
        };
        footer.write_to(&mut file_buffer);
        debug_assert_eq!(file_buffer.capacity(), file_buffer.len());
        file_buffer
    }

    fn build_smallest_biggest(&self) -> (Vec<u8>, Vec<u8>) {
        let first_table = self.tables.first().unwrap();
        let smallest_table_id = first_table.schema.table_id;
        let smallest_is_int_handle = first_table.handle_builder.col_meta.fixed_size > 0;
        let mut smallest_handle = first_table.handle_builder.handle_index[0].as_slice();
        let smallest_key = if smallest_is_int_handle {
            let smallest_int_handle = smallest_handle.get_i64_le();
            encode_row_key(smallest_table_id, smallest_int_handle)
        } else {
            encode_common_handle_for_test(smallest_table_id, smallest_handle)
        };
        let last_table = self.tables.last().unwrap();
        let biggest_table_id = last_table.schema.table_id;
        let biggest_is_int_handle = last_table.handle_builder.col_meta.fixed_size > 0;
        let mut biggest_handle = first_table.handle_builder.max_handle.as_slice();
        let biggest_key = if biggest_is_int_handle {
            let biggest_int_handle = biggest_handle.get_i64_le();
            encode_row_key(biggest_table_id, biggest_int_handle)
        } else {
            encode_common_handle_for_test(biggest_table_id, biggest_handle)
        };
        (smallest_key, biggest_key)
    }

    pub(crate) fn num_tables(&self) -> usize {
        self.tables.len()
    }

    pub(crate) fn reset(&mut self, id: u64) {
        self.file_id = id;
        self.estimated_size = 0;
        self.tables.clear();
    }
}

pub struct ColumnarTableBuilder {
    schema: Schema,
    handle_builder: ColumnarColumnBuilder,
    version_builder: ColumnarColumnBuilder,
    txn_id_builder: ColumnarColumnBuilder,
    column_builders: Vec<ColumnarColumnBuilder>,
    compressed_handle_index: Vec<u8>,
    properties: Vec<u8>,
    max_version: u64,
}

#[allow(dead_code)]
impl ColumnarTableBuilder {
    pub fn new(schema: Schema, opts: ColumnarTableBuildOptions, need_min_max: bool) -> Self {
        let pack_max_row_count = opts.pack_max_row_count;
        let pack_max_size = opts.pack_max_size;
        let mut column_builders = vec![];
        for col_info in &schema.columns {
            let col_builder = ColumnarColumnBuilder::new(
                col_info.clone(),
                pack_max_row_count,
                pack_max_size,
                need_min_max,
            );
            column_builders.push(col_builder);
        }
        let handle_col_info = schema.handle_column.clone();
        let handle_builder =
            ColumnarColumnBuilder::new(handle_col_info, pack_max_row_count, pack_max_size, false);
        let version_builder = ColumnarColumnBuilder::new(
            new_version_column_info(),
            pack_max_row_count,
            pack_max_size,
            false,
        );
        let txn_id_col = schema.txn_id_column.as_ref().unwrap().clone();
        let txn_id_builder =
            ColumnarColumnBuilder::new(txn_id_col, pack_max_row_count, pack_max_size, false);
        Self {
            schema,
            handle_builder,
            version_builder,
            txn_id_builder,
            column_builders,
            compressed_handle_index: vec![],
            properties: vec![],
            max_version: 0,
        }
    }

    pub(crate) fn append_block(&mut self, block: &Block, start_offset: usize) -> usize {
        let mut end_offset = block.length();
        end_offset = self
            .handle_builder
            .append_handle(&block.handles, start_offset, end_offset);
        self.version_builder
            .append(&block.versions, start_offset, end_offset);
        let txn_id_buf = block.txn_ids.as_ref().unwrap();
        self.txn_id_builder
            .append(txn_id_buf, start_offset, end_offset);
        for (i, col_buf) in block.columns.iter().enumerate() {
            let col_builder = &mut self.column_builders[i];
            col_builder.append(col_buf, start_offset, end_offset);
        }
        for i in start_offset..end_offset {
            self.max_version = max(self.max_version, block.versions.get_version(i))
        }
        end_offset
    }

    fn finish_table(&mut self) {
        self.handle_builder.finish_pack();
        // handle index add the handle next to the max handle
        // to avoid out of range seek load the last pack.
        if self.handle_builder.col_meta.fixed_size > 0 {
            let next_handle = self.handle_builder.max_handle.as_slice().get_i64_le();
            // if the handle is already i64::MAX, there is no next handle.
            if next_handle != i64::MAX {
                self.handle_builder
                    .handle_index
                    .push((next_handle + 1).to_le_bytes().to_vec());
            }
        } else {
            let mut next_handle = self.handle_builder.max_handle.clone();
            next_handle.push(0);
            self.handle_builder.handle_index.push(next_handle);
        };
        self.compressed_handle_index = self.handle_builder.build_handle_index();
        self.version_builder.finish_pack();
        self.txn_id_builder.finish_pack();
        for col_builder in &mut self.column_builders {
            col_builder.finish_pack();
        }
        for col_builder in &mut self.column_builders {
            col_builder.finish_min_max_pack();
        }
        add_property(
            &mut self.properties,
            PROP_KEY_MAX_VERSION.as_bytes(),
            &self.max_version.to_le_bytes(),
        );
    }

    pub(crate) fn compute_size(&self) -> DataSizeTuple {
        let mut total_size = DataSizeTuple::default();
        if self.handle_builder.row_count == 0 {
            return total_size;
        }
        let handle_col_size = self.handle_builder.compute_size();
        total_size.add(handle_col_size);
        let version_col_size = self.version_builder.compute_size();
        total_size.add(version_col_size);
        let txn_id_col_size = self.txn_id_builder.compute_size();
        total_size.add(txn_id_col_size);
        for col_builder in &self.column_builders {
            let col_size = col_builder.compute_size();
            total_size.add(col_size);
        }
        total_size.index_size += 4 + self.compressed_handle_index.len();
        // num columns(4) | table properties size(4) |
        total_size.index_size += 8;
        total_size.index_size += self.properties.len();
        total_size
    }

    pub(crate) fn build(&mut self, output_buf: &mut Vec<u8>) {
        // pack data part:
        self.handle_builder
            .col_meta
            .pack_offsets
            .update_base(output_buf.len() as u32);
        for pack in &self.handle_builder.compressed_packs {
            output_buf.extend_from_slice(pack);
        }

        self.version_builder
            .col_meta
            .pack_offsets
            .update_base(output_buf.len() as u32);
        for pack in &self.version_builder.compressed_packs {
            output_buf.extend_from_slice(pack);
        }

        self.txn_id_builder
            .col_meta
            .pack_offsets
            .update_base(output_buf.len() as u32);
        for pack in &self.txn_id_builder.compressed_packs {
            output_buf.extend_from_slice(pack);
        }

        for column in &mut self.column_builders {
            column
                .col_meta
                .pack_offsets
                .update_base(output_buf.len() as u32);
            for pack in &column.compressed_packs {
                output_buf.extend_from_slice(pack);
            }
        }

        // index part:
        let num_columns = self.column_builders.len() as u32 + 3;
        output_buf.put_u32_le(num_columns);
        self.handle_builder.col_meta.write_to(output_buf);
        self.version_builder.col_meta.write_to(output_buf);
        self.txn_id_builder.col_meta.write_to(output_buf);
        for column in &mut self.column_builders {
            column.col_meta.write_to(output_buf);
        }
        output_buf.put_u32_le(self.compressed_handle_index.len() as u32);
        output_buf.extend_from_slice(&self.compressed_handle_index);
        output_buf.put_u32_le(self.properties.len() as u32);
        output_buf.extend_from_slice(&self.properties);
    }

    pub(crate) fn get_estimated_size(&self) -> usize {
        let mut estimated_size = self.handle_builder.estimated_size
            + self.version_builder.estimated_size
            + self.txn_id_builder.estimated_size;
        for cb in &self.column_builders {
            estimated_size += cb.estimated_size;
        }
        estimated_size
    }
}

pub struct ColumnarColumnBuilder {
    col_meta: ColumnMeta,
    pack_buffer: ColumnBuffer,
    pack_max_row_count: usize,
    pack_max_size: usize,
    row_count: u32,
    is_handle: bool,
    handle_index: Vec<Vec<u8>>,
    max_handle: Vec<u8>,
    uncompressed_buf: Vec<u8>,
    compressed_buf: Vec<u8>,
    compressed_packs: Vec<Vec<u8>>,
    estimated_size: usize,
}

#[derive(Default, Copy, Clone, Debug)]
pub(crate) struct DataSizeTuple {
    packs_data_size: usize,
    index_size: usize,
}

impl DataSizeTuple {
    pub fn add(&mut self, other: Self) {
        self.packs_data_size += other.packs_data_size;
        self.index_size += other.index_size;
    }
}

impl ColumnarColumnBuilder {
    pub(crate) fn new(
        col_info: ColumnInfo,
        pack_max_row_count: usize,
        pack_max_size: usize,
        need_min_max: bool,
    ) -> Self {
        let col_id = col_info.get_column_id() as i32;
        let col_meta = ColumnMeta::new(col_info, need_min_max);
        let buffer = ColumnBuffer::new(col_id, col_meta.fixed_size, col_meta.nullable);
        let is_handle = col_id == HANDLE_COL_ID || col_meta.col_info.get_pk_handle();
        Self {
            col_meta,
            pack_buffer: buffer,
            pack_max_row_count,
            pack_max_size,
            row_count: 0,
            is_handle,
            handle_index: vec![],
            max_handle: vec![],
            uncompressed_buf: vec![],
            compressed_buf: vec![],
            compressed_packs: vec![],
            estimated_size: 0,
        }
    }

    pub(crate) fn append(&mut self, input: &ColumnBuffer, row_offset: usize, row_end_off: usize) {
        let mut current_offset = row_offset;
        while current_offset < row_end_off {
            let pack_remain = self.pack_max_row_count - self.pack_buffer.length();
            let end_idx_in_length_limit = current_offset + pack_remain;
            let pack_size_remain = self.pack_max_size - self.pack_buffer.data_size();
            let end_idx_in_size_limit =
                input.get_end_idx_in_size_limit(current_offset, pack_size_remain);
            let limited_end_offset = std::cmp::min(end_idx_in_length_limit, end_idx_in_size_limit);
            let current_end_offset = std::cmp::min(row_end_off, limited_end_offset);
            self.pack_buffer
                .append(input, current_offset, current_end_offset);
            self.row_count += (current_end_offset - current_offset) as u32;
            if self.pack_buffer.length() >= self.pack_max_row_count
                || self.pack_buffer.data_size() >= self.pack_max_size
            {
                self.finish_pack();
            }
            current_offset = current_end_offset;
        }
    }

    pub(crate) fn append_handle(
        &mut self,
        input: &ColumnBuffer,
        row_offset: usize,
        row_end_off: usize,
    ) -> usize {
        debug_assert!(self.is_handle);
        for i in row_offset..row_end_off {
            let handle = input.get_not_null_value(i);
            let current_length = self.pack_buffer.length();
            if current_length >= self.pack_max_row_count {
                let last_handle = self.pack_buffer.get_not_null_value(current_length - 1);
                if last_handle != handle {
                    self.finish_pack();
                    return i;
                }
            }
            self.pack_buffer.push_value(handle);
            self.row_count += 1;
        }
        row_end_off
    }

    fn finish_pack(&mut self) {
        let current_length = self.pack_buffer.length();
        if current_length == 0 {
            return;
        }
        if self.is_handle {
            let pack_first_handle = self.pack_buffer.get_not_null_value(0).to_vec();
            self.handle_index.push(pack_first_handle);
            let last_handle = self.pack_buffer.get_not_null_value(current_length - 1);
            self.max_handle.truncate(0);
            self.max_handle.extend_from_slice(last_handle);
        }
        self.fill_uncompressed_buf();
        if let Some(mut min_max) = self.col_meta.min_max.take() {
            if let Some((min, max)) = self.compute_min_max() {
                min_max.push_value(&min);
                min_max.push_value(&max);
            } else {
                min_max.push_null();
                min_max.push_null();
            }
            self.col_meta.min_max = Some(min_max);
        }
        let pack = self.compress_pack();
        let (total_len, _) = self.col_meta.pack_offsets.end_offset();
        self.col_meta
            .pack_offsets
            .push(total_len + pack.len() as u32, self.row_count);
        self.estimated_size += pack.len();
        self.compressed_packs.push(pack);
        self.pack_buffer.reset();
    }

    fn build_handle_index(&mut self) -> Vec<u8> {
        debug_assert!(self.is_handle);
        for handle_val in &self.handle_index {
            self.pack_buffer.push_value(handle_val);
        }
        self.fill_uncompressed_buf();
        self.compress_pack()
    }

    fn fill_uncompressed_buf(&mut self) {
        self.pack_buffer.write_to(&mut self.uncompressed_buf);
    }

    fn compress_pack(&mut self) -> Vec<u8> {
        compress_pack(&self.uncompressed_buf, &mut self.compressed_buf)
    }

    fn finish_min_max_pack(&mut self) {
        if let Some(min_max_buf) = self.col_meta.min_max.take() {
            self.pack_buffer = min_max_buf;
            self.fill_uncompressed_buf();
            let compressed_buf = self.compress_pack();
            self.col_meta.compressed_min_max_pack = compressed_buf;
        }
    }

    fn compute_min_max(&self) -> Option<(Vec<u8>, Vec<u8>)> {
        let tp = FieldTypeTp::from_u8(self.col_meta.col_info.get_tp() as u8).unwrap();
        let is_unsigned = get_unsigned(&self.col_meta.col_info);
        if self.pack_buffer.nullable && self.pack_buffer.nulls.iter().all(|&x| x == 1) {
            return None;
        }
        debug_assert!(self.pack_buffer.length() > 0);
        match tp {
            FieldTypeTp::Unspecified => {}
            FieldTypeTp::Float => {
                return self
                    .compute_min_max_fixed::<f32>()
                    .map(|(min, max)| (min.to_le_bytes().to_vec(), max.to_le_bytes().to_vec()));
            }
            FieldTypeTp::Double => {
                return self
                    .compute_min_max_fixed::<f64>()
                    .map(|(min, max)| (min.to_le_bytes().to_vec(), max.to_le_bytes().to_vec()));
            }
            FieldTypeTp::Null => {}
            FieldTypeTp::Tiny | FieldTypeTp::Short | FieldTypeTp::Int24 | FieldTypeTp::Long => {
                return if is_unsigned {
                    self.compute_min_max_fixed::<u32>()
                        .map(|(min, max)| (min.to_le_bytes().to_vec(), max.to_le_bytes().to_vec()))
                } else {
                    self.compute_min_max_fixed::<i32>()
                        .map(|(min, max)| (min.to_le_bytes().to_vec(), max.to_le_bytes().to_vec()))
                };
            }
            FieldTypeTp::LongLong => {
                return if is_unsigned {
                    self.compute_min_max_fixed::<u64>()
                        .map(|(min, max)| (min.to_le_bytes().to_vec(), max.to_le_bytes().to_vec()))
                } else {
                    self.compute_min_max_fixed::<i64>()
                        .map(|(min, max)| (min.to_le_bytes().to_vec(), max.to_le_bytes().to_vec()))
                };
            }
            FieldTypeTp::Duration | FieldTypeTp::Year => {
                return self
                    .compute_min_max_fixed::<i64>()
                    .map(|(min, max)| (min.to_le_bytes().to_vec(), max.to_le_bytes().to_vec()));
            }
            FieldTypeTp::Timestamp
            | FieldTypeTp::DateTime
            | FieldTypeTp::Date
            | FieldTypeTp::NewDate
            | FieldTypeTp::Bit
            | FieldTypeTp::Enum => {
                return self
                    .compute_min_max_fixed::<u64>()
                    .map(|(min, max)| (min.to_le_bytes().to_vec(), max.to_le_bytes().to_vec()));
            }
            FieldTypeTp::VarChar => {}
            FieldTypeTp::Json => {}
            FieldTypeTp::NewDecimal => {}
            FieldTypeTp::Set => {}
            FieldTypeTp::TinyBlob => {}
            FieldTypeTp::MediumBlob => {}
            FieldTypeTp::LongBlob => {}
            FieldTypeTp::Blob => {}
            FieldTypeTp::VarString => {}
            FieldTypeTp::String => {}
            FieldTypeTp::Geometry => {}
            FieldTypeTp::TiDbVectorFloat32 => {}
        }
        None
    }

    fn compute_min_max_fixed<T: bytemuck::AnyBitPattern + PartialOrd>(&self) -> Option<(T, T)> {
        let val_slice: &[T] = bytemuck::cast_slice(&self.pack_buffer.data_buf);
        let mut min_val = None;
        let mut max_val = None;
        for (i, val) in val_slice.iter().enumerate() {
            if self.pack_buffer.nullable && self.pack_buffer.nulls[i] == 1 {
                continue;
            }
            if min_val.is_none() {
                min_val = Some(*val);
                max_val = Some(*val);
                continue;
            }
            if min_val.unwrap() > *val {
                min_val = Some(*val);
            }
            if max_val.unwrap() < *val {
                max_val = Some(*val);
            }
        }
        let min_val = min_val?;
        let max_val = max_val?;
        Some((min_val, max_val))
    }

    // index part:
    // num_packs(4) | pack_offsets | num_old_packs(4) | old_pack_offsets |
    // col_info_len(4) | col_info |
    // min_max_idx_len(4) | min_max_idx
    pub(crate) fn compute_size(&self) -> DataSizeTuple {
        let packs_data_size = self.compressed_packs.iter().map(|buf| buf.len()).sum();
        let index_size = self.col_meta.compute_size();
        DataSizeTuple {
            packs_data_size,
            index_size,
        }
    }
}
