// Copyright 2024 TiKV Project Authors. Licensed under Apache-2.0.

use std::sync::Arc;

use bytes::{Buf, BufMut};
use collections::HashMap;
use protobuf::Message;
use tidb_query_datatype::{FieldTypeFlag, FieldTypeTp};
use tipb::ColumnInfo;

use crate::table::{
    columnar::builder::{
        TableOffset, ENCODING_TYPE_NONE, PACK_FORMAT, PROP_KEY_BIGGEST, PROP_KEY_MAX_VERSION,
        PROP_KEY_SMALLEST, PROP_KEY_SNAP_VERSION,
    },
    parse_prop_data, search,
    sstable::{File, LZ4_COMPRESSION},
    InnerKey,
};

pub(crate) const HANDLE_COL_ID: i32 = -1;
pub(crate) const VERSION_COL_ID: i32 = -1024;
pub(crate) const TXN_ID_COL_ID: i32 = -1034;

pub const COLUMNAR_MAGIC: u32 = 0xc01e32ae;

#[derive(Default, Clone, Debug, PartialEq)]
pub struct Schema {
    pub table_id: i64,
    pub handle_column: ColumnInfo,
    pub version_column: ColumnInfo,
    pub txn_id_column: Option<ColumnInfo>,
    pub columns: Vec<ColumnInfo>,
}

#[repr(C)]
pub struct ColumnarFileFooter {
    pub number_tables: u32,
    pub properties_size: u32,
    pub compression_type: u8,
    pub checksum_type: u8,
    pub format_version: u16,
    pub magic: u32,
}

impl ColumnarFileFooter {
    pub fn parse(mut buf: &[u8]) -> Self {
        let number_tables = buf.get_u32_le();
        let properties_size = buf.get_u32_le();
        let compression_type = buf.get_u8();
        let checksum_type = buf.get_u8();
        let format_version = buf.get_u16_le();
        let magic = buf.get_u32_le();
        Self {
            number_tables,
            properties_size,
            compression_type,
            checksum_type,
            format_version,
            magic,
        }
    }

    pub fn write_to(&self, buf: &mut Vec<u8>) {
        buf.put_u32_le(self.number_tables);
        buf.put_u32_le(self.properties_size);
        buf.put_u8(self.compression_type);
        buf.put_u8(self.checksum_type);
        buf.put_u16_le(self.format_version);
        buf.put_u32_le(self.magic);
    }

    pub fn compute_size() -> usize {
        4 + 4 + 1 + 1 + 2 + 4
    }
}

pub(crate) struct HandleIndex {
    pub(crate) buf: ColumnBuffer,
}

impl HandleIndex {
    fn new(buf: ColumnBuffer) -> Self {
        Self { buf }
    }

    pub(crate) fn search_pack_idx(&self, mut handle: &[u8]) -> usize {
        let idx = if self.buf.fixed_size > 0 {
            let int_handle = handle.get_i64_le();
            search(self.buf.length(), |i| {
                self.buf.get_int_handle_value(i) > int_handle
            })
        } else {
            search(self.buf.length(), |i| {
                self.buf.get_not_null_value(i) > handle
            })
        };
        if idx == 0 { 0 } else { idx - 1 }
    }
}

pub(crate) struct TableMeta {
    pub(crate) table_id: i64,
    pub(crate) handle_index: HandleIndex,
    pub(crate) handle_column: Arc<ColumnMeta>,
    pub(crate) version_column: Arc<ColumnMeta>,
    pub(crate) txn_id_column: Arc<ColumnMeta>,
    pub(crate) columns: HashMap<i32, Arc<ColumnMeta>>,
}

impl TableMeta {
    pub(crate) fn parse(table_id: i64, mut buf: &[u8]) -> Self {
        let num_cols = buf.get_u32_le();
        let (handle_column, remained) = ColumnMeta::parse(buf);
        buf = remained;
        let (version_column, remained) = ColumnMeta::parse(buf);
        buf = remained;
        let (txn_id_column, remained) = ColumnMeta::parse(buf);
        buf = remained;
        let mut columns = HashMap::default();
        for _ in 0..(num_cols - 3) {
            let (col, remained) = ColumnMeta::parse(buf);
            buf = remained;
            columns.insert(col.col_info.get_column_id() as i32, Arc::new(col));
        }
        let mut uncompressed_buf = vec![];
        let compressed_handle_len = buf.get_u32_le() as usize;
        let compressed_handle_idx_buf = &buf[..compressed_handle_len];
        buf = &buf[compressed_handle_len..];
        let handle_col_id = handle_column.col_info.get_column_id() as i32;
        let mut handle_index_buf = ColumnBuffer::new(
            handle_col_id,
            handle_column.fixed_size,
            handle_column.nullable,
        );
        decompress_pack(compressed_handle_idx_buf, &mut uncompressed_buf);
        handle_index_buf.parse(&uncompressed_buf);
        let handle_index = HandleIndex::new(handle_index_buf);
        let _table_props_len = buf.get_u32_le() as usize;
        let _properties = &buf[.._table_props_len];
        Self {
            table_id,
            handle_index,
            handle_column: Arc::new(handle_column),
            version_column: Arc::new(version_column),
            txn_id_column: Arc::new(txn_id_column),
            columns,
        }
    }
}

pub struct ColumnMeta {
    pub(crate) col_info: ColumnInfo,
    pub(crate) fixed_size: usize,
    pub(crate) nullable: bool,
    // The length is number_of_packs + 1, the last element is the end offset of the last pack.
    pub(crate) pack_offsets: PackOffsets,
    // table level min-max, contains just two values, can be used to filter the whole file.
    pub(crate) min_max: Option<ColumnBuffer>,
    pub(crate) compressed_min_max_pack: Vec<u8>,
}

impl ColumnMeta {
    pub(crate) fn new(col_info: ColumnInfo, need_min_max: bool) -> Self {
        let col_id = col_info.get_column_id() as i32;
        let fixed_size = get_fixed_size(&col_info);
        let nullable = col_info.get_flag() as u32 & FieldTypeFlag::NOT_NULL.bits() == 0;
        let min_max = (can_build_min_max(&col_info) && need_min_max)
            .then(|| ColumnBuffer::new(col_id, fixed_size, nullable));
        Self {
            col_info,
            fixed_size,
            nullable,
            pack_offsets: PackOffsets::default(),
            min_max,
            compressed_min_max_pack: vec![],
        }
    }

    pub(crate) fn parse(mut buf: &[u8]) -> (Self, &[u8]) {
        let pack_offsets = PackOffsets::parse(buf).unwrap();
        buf = &buf[pack_offsets.compute_size()..];
        let col_info_len = buf.get_u32_le() as usize;
        let col_info_buf = &buf[..col_info_len];
        let mut col_info = ColumnInfo::new();
        col_info.merge_from_bytes(col_info_buf).unwrap();
        buf = &buf[col_info_len..];
        let col_id = col_info.get_column_id() as i32;
        let fixed_size = get_fixed_size(&col_info);
        let nullable = get_nullable(&col_info);
        let min_max_idx_len = buf.get_u32_le() as usize;
        let min_max_opt = if min_max_idx_len > 0 {
            let compressed_min_max_idx_buf = &buf[..min_max_idx_len];
            buf = &buf[min_max_idx_len..];
            let mut uncompressed_min_max_buf = vec![];
            decompress_pack(compressed_min_max_idx_buf, &mut uncompressed_min_max_buf);
            let mut min_max = ColumnBuffer::new(col_id, fixed_size, nullable);
            min_max.parse(&uncompressed_min_max_buf);
            Some(min_max)
        } else {
            None
        };
        buf = &buf[4..]; // column props, currently not used.
        (
            Self {
                col_info,
                fixed_size,
                nullable,
                pack_offsets,
                min_max: min_max_opt,
                compressed_min_max_pack: vec![],
            },
            buf,
        )
    }

    // num_packs(4) | pack_offsets | num_old_packs(4) | old_pack_offsets |
    // col_info_len(4) | col_info |
    // min_max_idx_len(4) | min_max_idx | column_props_len(4) | column_props
    pub(crate) fn write_to(&self, buf: &mut Vec<u8>) {
        self.pack_offsets.write_to(buf);
        let col_info_len = self.col_info.compute_size();
        buf.put_u32_le(col_info_len);
        let col_info_bin = self.col_info.write_to_bytes().unwrap();
        buf.extend_from_slice(&col_info_bin);
        buf.put_u32_le(self.compressed_min_max_pack.len() as u32);
        buf.extend_from_slice(&self.compressed_min_max_pack);
        buf.put_u32_le(0);
    }

    pub(crate) fn compute_size(&self) -> usize {
        let pack_offsets_len = 4 + self.pack_offsets.offsets.len() * 8;
        let col_info_len = 4 + self.col_info.compute_size() as usize;
        let min_max_idx_len = 4 + self.compressed_min_max_pack.len();
        let column_props_len = 4;
        pack_offsets_len + col_info_len + min_max_idx_len + column_props_len
    }

    pub(crate) fn get_pack_offset(&self, pack_idx: usize) -> ((u32, u32), (u32, u32)) {
        if pack_idx >= self.pack_offsets.num_packs() {
            info!(
                "get pack offset idx {} off {:?} row {:?}",
                pack_idx, self.pack_offsets.offsets, self.pack_offsets.row_offsets
            );
        }
        let start = self.pack_offsets.get(pack_idx);
        let end = self.pack_offsets.get(pack_idx + 1);
        (start, end)
    }
}

pub struct PackOffsets {
    offsets: Vec<u32>,
    row_offsets: Vec<u32>,
}

impl Default for PackOffsets {
    fn default() -> Self {
        Self {
            offsets: vec![0],
            row_offsets: vec![0],
        }
    }
}

impl PackOffsets {
    pub fn parse(mut buf: &[u8]) -> Option<Self> {
        let offsets_len = buf.get_u32_le() as usize;
        if offsets_len == 0 {
            return None;
        }
        let mut offsets = Vec::with_capacity(offsets_len);
        let mut row_offsets = Vec::with_capacity(offsets_len);
        for _ in 0..offsets_len {
            offsets.push(buf.get_u32_le());
        }
        for _ in 0..offsets_len {
            row_offsets.push(buf.get_u32_le());
        }
        Some(Self {
            offsets,
            row_offsets,
        })
    }

    pub fn compute_size(&self) -> usize {
        4 + self.offsets.len() * 8
    }

    pub fn write_to(&self, buf: &mut Vec<u8>) {
        buf.put_u32_le(self.offsets.len() as u32);
        buf.extend_from_slice(bytemuck::cast_slice(&self.offsets));
        buf.extend_from_slice(bytemuck::cast_slice(&self.row_offsets));
    }

    pub fn push(&mut self, offset: u32, row_offset: u32) {
        self.offsets.push(offset);
        self.row_offsets.push(row_offset);
    }

    pub fn get(&self, idx: usize) -> (u32, u32) {
        (self.offsets[idx], self.row_offsets[idx])
    }

    pub fn end_offset(&self) -> (u32, u32) {
        let idx = self.offsets.len() - 1;
        (self.offsets[idx], self.row_offsets[idx])
    }

    pub fn update_base(&mut self, base: u32) {
        for offset in &mut self.offsets {
            *offset += base;
        }
    }

    pub fn num_packs(&self) -> usize {
        self.offsets.len() - 1
    }

    pub fn search_pack_idx(&self, from_pack_idx: usize, row_idx: u32) -> usize {
        for (i, &row_off) in self.row_offsets[from_pack_idx..].iter().enumerate() {
            if row_idx < row_off {
                return from_pack_idx + i - 1;
            }
        }
        self.num_packs()
    }
}

pub struct ColumnarFile {
    core: Arc<ColumnarFileCore>,
}

impl ColumnarFile {
    pub fn open(file: Arc<dyn File>) -> crate::table::Result<Self> {
        let file_len = file.size();
        let footer_len = ColumnarFileFooter::compute_size();
        let mut buf = vec![0; footer_len];
        let footer_offset = file_len - footer_len as u64;
        file.read_at(&mut buf, footer_offset)?;
        let footer = ColumnarFileFooter::parse(&buf);
        let table_offsets_size = footer.number_tables as u64 * TableOffset::compute_size() as u64;
        let table_offsets_offset = footer_offset - table_offsets_size;
        buf.resize(table_offsets_size as usize, 0);
        file.read_at(&mut buf, table_offsets_offset)?;
        let mut table_offsets = vec![];
        for i in 0..footer.number_tables {
            let offset = i as usize * TableOffset::compute_size();
            let table_offset = TableOffset::parse(&buf[offset..]);
            table_offsets.push(table_offset);
        }
        let mut smallest_key = vec![];
        let mut biggest_key = vec![];
        let mut max_version = 0;
        let mut l0_version = None;
        let property_offset = table_offsets_offset - footer.properties_size as u64;
        let mut property_buf = vec![0; footer.properties_size as usize];
        file.read_at(&mut property_buf, property_offset)?;
        let mut prop_remain = property_buf.as_slice();
        while !prop_remain.is_empty() {
            let (key, mut val, remain) = parse_prop_data(prop_remain);
            if key == PROP_KEY_SMALLEST.as_bytes() {
                smallest_key = val.to_vec();
            } else if key == PROP_KEY_BIGGEST.as_bytes() {
                biggest_key = val.to_vec();
            } else if key == PROP_KEY_MAX_VERSION.as_bytes() {
                max_version = val.get_u64_le();
            } else if key == PROP_KEY_SNAP_VERSION.as_bytes() {
                l0_version = Some(val.get_u64_le());
            }
            prop_remain = remain;
        }
        let mut tables = HashMap::default();
        for table_offset in table_offsets {
            let index_size = table_offset.end_offset - table_offset.index_offset;
            let mut table_index_buf = vec![0; index_size as usize];
            file.read_at(&mut table_index_buf, table_offset.index_offset as u64)?;
            let table_meta = TableMeta::parse(table_offset.table_id, &table_index_buf);
            tables.insert(table_offset.table_id, Arc::new(table_meta));
        }
        Ok(Self {
            core: Arc::new(ColumnarFileCore {
                file,
                smallest_key,
                biggest_key,
                max_version,
                l0_version,
                tables,
            }),
        })
    }

    pub(crate) fn get_table(&self, table_id: i64) -> Arc<TableMeta> {
        self.core.tables.get(&table_id).unwrap().clone()
    }

    pub fn get_file(&self) -> Arc<dyn File> {
        self.core.file.clone()
    }

    pub fn get_smallest(&self) -> InnerKey<'_> {
        InnerKey::from_inner_buf(&self.core.smallest_key)
    }

    pub fn get_biggest(&self) -> InnerKey<'_> {
        InnerKey::from_inner_buf(&self.core.biggest_key)
    }

    pub fn get_max_version(&self) -> u64 {
        self.core.max_version
    }

    pub fn get_l0_version(&self) -> Option<u64> {
        self.core.l0_version
    }
}

struct ColumnarFileCore {
    file: Arc<dyn File>,
    smallest_key: Vec<u8>,
    biggest_key: Vec<u8>,
    max_version: u64,
    l0_version: Option<u64>,
    tables: HashMap<i64, Arc<TableMeta>>,
}

pub(crate) struct ColumnBuffer {
    pub(crate) col_id: i32,
    pub(crate) nullable: bool,
    pub(crate) fixed_size: usize,
    pub(crate) data_buf: Vec<u8>,
    pub(crate) offsets: Vec<u32>,
    pub(crate) nulls: Vec<u8>,
}

impl ColumnBuffer {
    pub(crate) fn new(col_id: i32, fixed_size: usize, nullable: bool) -> Self {
        let offsets = if fixed_size > 0 { vec![] } else { vec![0] };
        Self {
            col_id,
            nullable,
            fixed_size,
            data_buf: vec![],
            offsets,
            nulls: vec![],
        }
    }

    pub(crate) fn new_from_col_info(col_info: &ColumnInfo) -> Self {
        let col_id = col_info.get_column_id() as i32;
        let fixed_size = get_fixed_size(col_info);
        let nullable = get_nullable(col_info);
        Self::new(col_id, fixed_size, nullable)
    }

    pub fn length(&self) -> usize {
        if self.fixed_size > 0 {
            self.data_buf.len() / self.fixed_size
        } else {
            self.offsets.len() - 1
        }
    }

    pub fn data_size(&self) -> usize {
        self.data_buf.len()
    }

    pub fn get_end_idx_in_size_limit(&self, from_idx: usize, size_limit: usize) -> usize {
        if self.fixed_size > 0 {
            return self.length();
        }
        let start = self.offsets[from_idx] as usize;
        let end = self.offsets.last().cloned().unwrap() as usize;
        if end - start > size_limit {
            for (i, &offset) in self.offsets[from_idx + 1..].iter().enumerate() {
                if offset as usize - start > size_limit {
                    return from_idx + 1 + i;
                }
            }
        }
        self.length()
    }

    pub fn reset(&mut self) {
        if self.fixed_size == 0 {
            self.offsets.truncate(1);
        }
        self.data_buf.truncate(0);
        self.nulls.truncate(0);
    }

    pub(crate) fn push_null(&mut self) {
        self.push_zero();
        self.nulls.push(1);
    }

    pub(crate) fn push_zero(&mut self) {
        if self.fixed_size > 0 {
            self.data_buf
                .resize(self.data_buf.len() + self.fixed_size, 0);
        } else {
            self.offsets.push(self.data_buf.len() as u32);
        }
    }

    // version is not 0 and delete is true represents mvcc delete.
    // version is 0 and delete is true represents tombstone.
    #[allow(dead_code)]
    pub(crate) fn push_version(&mut self, version: u64, is_delete: bool) {
        debug_assert_eq!(self.col_id, VERSION_COL_ID);
        self.data_buf.put_u64_le(version);
        self.nulls.push(is_delete as u8);
    }

    pub(crate) fn push_value(&mut self, data: &[u8]) {
        self.data_buf.extend_from_slice(data);
        if self.fixed_size == 0 {
            self.offsets.push(self.data_buf.len() as u32);
        } else {
            debug_assert_eq!(data.len(), self.fixed_size);
        }
        if self.nullable {
            self.nulls.push(0);
        }
    }

    #[allow(dead_code)]
    pub(crate) fn get_nullable_value(&self, idx: usize) -> Option<&[u8]> {
        debug_assert!(self.nullable);
        if self.nulls[idx] == 1 {
            return None;
        }
        if self.fixed_size == 0 {
            let start = self.offsets[idx] as usize;
            let end = self.offsets[idx + 1] as usize;
            Some(&self.data_buf[start..end])
        } else {
            let start = idx * self.fixed_size;
            let end = (idx + 1) * self.fixed_size;
            Some(&self.data_buf[start..end])
        }
    }

    #[inline]
    pub(crate) fn get_not_null_value(&self, idx: usize) -> &[u8] {
        debug_assert!(
            !self.nullable || self.nulls[idx] == 0,
            "id: {}, nulls: {:?}, idx {}",
            self.col_id,
            self.nulls,
            idx
        );
        if self.fixed_size == 0 {
            let start = self.offsets[idx] as usize;
            let end = self.offsets[idx + 1] as usize;
            &self.data_buf[start..end]
        } else {
            let start = idx * self.fixed_size;
            let end = (idx + 1) * self.fixed_size;
            &self.data_buf[start..end]
        }
    }

    #[inline]
    pub(crate) fn get_int_handle_value(&self, idx: usize) -> i64 {
        debug_assert!(self.fixed_size == 8);
        let start = idx * self.fixed_size;
        let end = (idx + 1) * self.fixed_size;
        (&self.data_buf[start..end]).get_i64_le()
    }

    pub(crate) fn get_version(&self, idx: usize) -> u64 {
        debug_assert!(self.col_id == VERSION_COL_ID || self.col_id == TXN_ID_COL_ID);
        (&self.data_buf[idx * 8..]).get_u64_le()
    }

    pub(crate) fn is_null(&self, idx: usize) -> bool {
        debug_assert!(self.nullable);
        self.nulls[idx] == 1
    }

    pub(crate) fn append(
        &mut self,
        other: &ColumnBuffer,
        row_offset: usize,
        row_end_offset: usize,
    ) {
        if self.fixed_size > 0 {
            let other_data_offset = row_offset * self.fixed_size;
            let other_data_end_offset = row_end_offset * self.fixed_size;
            self.data_buf
                .extend_from_slice(&other.data_buf[other_data_offset..other_data_end_offset]);
        } else {
            let other_data_offset = other.offsets[row_offset] as usize;
            let other_data_end_offset = other.offsets[row_end_offset] as usize;
            let data_base_offset = self.data_buf.len() as u32;
            self.data_buf
                .extend_from_slice(&other.data_buf[other_data_offset..other_data_end_offset]);
            let update_start = self.offsets.len();
            self.offsets
                .extend_from_slice(&other.offsets[row_offset + 1..=row_end_offset]);
            if other_data_offset > data_base_offset as usize {
                let delta = other_data_offset as u32 - data_base_offset;
                for offset in &mut self.offsets[update_start..] {
                    *offset -= delta;
                }
            } else {
                let delta = data_base_offset - other_data_offset as u32;
                for offset in &mut self.offsets[update_start..] {
                    *offset += delta;
                }
            }
        }
        if self.nullable {
            self.nulls
                .extend_from_slice(&other.nulls[row_offset..row_end_offset]);
        }
    }

    pub(crate) fn truncate(&mut self, length: usize) {
        debug_assert!(self.length() >= length);
        if self.fixed_size > 0 {
            self.data_buf.truncate(length * self.fixed_size);
        } else {
            self.data_buf.truncate(self.offsets[length] as usize);
            self.offsets.truncate(length + 1);
        }
        if self.nullable {
            self.nulls.truncate(length);
        }
    }

    pub(crate) fn parse(&mut self, mut uncompressed_pack: &[u8]) {
        let _pack_format = uncompressed_pack.get_u16();
        let _encoding_type = uncompressed_pack.get_u16();
        let length = uncompressed_pack.get_u32_le() as usize;
        let data_end_offset = if self.fixed_size == 0 {
            self.offsets.truncate(0);
            let offsets_end_idx = (length + 1) * 4;
            let offsets = bytemuck::cast_slice(&uncompressed_pack[..offsets_end_idx]);
            self.offsets.extend_from_slice(offsets);
            uncompressed_pack = &uncompressed_pack[offsets_end_idx..];
            self.offsets.last().copied().unwrap() as usize
        } else {
            length * self.fixed_size
        };
        self.data_buf.truncate(0);
        self.data_buf
            .extend_from_slice(&uncompressed_pack[..data_end_offset]);
        uncompressed_pack = &uncompressed_pack[data_end_offset..];
        if self.nullable {
            self.nulls.truncate(0);
            self.nulls.extend_from_slice(uncompressed_pack);
        }
    }

    pub(crate) fn write_to(&self, buf: &mut Vec<u8>) {
        buf.clear();
        buf.put_u16_le(PACK_FORMAT);
        buf.put_u16_le(ENCODING_TYPE_NONE);
        buf.put_u32_le(self.length() as u32);
        if self.fixed_size == 0 {
            buf.extend_from_slice(bytemuck::cast_slice(&self.offsets));
        }
        buf.extend_from_slice(&self.data_buf);
        buf.extend_from_slice(&self.nulls);
    }
}

pub struct Block {
    pub(crate) handles: ColumnBuffer,
    pub(crate) versions: ColumnBuffer,
    pub(crate) txn_ids: Option<ColumnBuffer>,
    pub(crate) columns: Vec<ColumnBuffer>,
}

impl Block {
    pub(crate) fn new(schema: &Schema) -> Self {
        let handles = ColumnBuffer::new_from_col_info(&schema.handle_column);
        let versions = ColumnBuffer::new_from_col_info(&schema.version_column);
        let txn_ids = schema
            .txn_id_column
            .as_ref()
            .map(|x| ColumnBuffer::new_from_col_info(x));
        let mut columns = vec![];
        for col_info in &schema.columns {
            columns.push(ColumnBuffer::new_from_col_info(col_info));
        }
        Self {
            handles,
            versions,
            txn_ids,
            columns,
        }
    }

    pub(crate) fn reset(&mut self) {
        self.handles.reset();
        self.versions.reset();
        if let Some(txn_ids) = self.txn_ids.as_mut() {
            txn_ids.reset();
        }
        self.columns.iter_mut().for_each(|col| col.reset());
    }

    pub(crate) fn truncate(&mut self, length: usize) {
        self.handles.truncate(length);
        self.versions.truncate(length);
        if let Some(txn_ids) = self.txn_ids.as_mut() {
            txn_ids.truncate(length);
        }
        for col in &mut self.columns {
            col.truncate(length);
        }
    }

    pub(crate) fn append(&mut self, other: &Block, row_offset: usize, row_end_offset: usize) {
        self.handles
            .append(&other.handles, row_offset, row_end_offset);
        self.versions
            .append(&other.versions, row_offset, row_end_offset);
        if let Some(txn_ids) = self.txn_ids.as_mut() {
            txn_ids.append(other.txn_ids.as_ref().unwrap(), row_offset, row_end_offset);
        }
        for (col, other_col) in self.columns.iter_mut().zip(&other.columns) {
            col.append(other_col, row_offset, row_end_offset);
        }
    }
}

pub(crate) fn get_fixed_size(col_info: &ColumnInfo) -> usize {
    let tp = FieldTypeTp::from_u8(col_info.get_tp() as u8).unwrap();
    match tp {
        FieldTypeTp::Tiny
        | FieldTypeTp::Short
        | FieldTypeTp::Int24
        | FieldTypeTp::Long
        | FieldTypeTp::Float => 4,
        FieldTypeTp::Double
        | FieldTypeTp::Timestamp
        | FieldTypeTp::LongLong
        | FieldTypeTp::Date
        | FieldTypeTp::Duration
        | FieldTypeTp::Year
        | FieldTypeTp::DateTime
        | FieldTypeTp::NewDate
        | FieldTypeTp::Enum
        | FieldTypeTp::Set
        | FieldTypeTp::Bit => 8,
        FieldTypeTp::NewDecimal => 40,
        _ => 0,
    }
}

pub(crate) fn can_build_min_max(col_info: &ColumnInfo) -> bool {
    let tp = FieldTypeTp::from_u8(col_info.get_tp() as u8).unwrap();
    matches!(
        tp,
        FieldTypeTp::Tiny
            | FieldTypeTp::Short
            | FieldTypeTp::Int24
            | FieldTypeTp::Long
            | FieldTypeTp::Float
            | FieldTypeTp::Double
            | FieldTypeTp::Timestamp
            | FieldTypeTp::LongLong
            | FieldTypeTp::Date
            | FieldTypeTp::Duration
            | FieldTypeTp::Year
            | FieldTypeTp::DateTime
            | FieldTypeTp::NewDate
            | FieldTypeTp::Enum
            | FieldTypeTp::NewDecimal
    )
}

pub(crate) fn get_nullable(col_info: &ColumnInfo) -> bool {
    col_info.get_flag() as u32 & FieldTypeFlag::NOT_NULL.bits() == 0
}

pub(crate) fn get_unsigned(col_info: &ColumnInfo) -> bool {
    col_info.get_flag() as u32 & FieldTypeFlag::UNSIGNED.bits() == 1
}

pub(crate) fn decompress_pack(mut compressed_pack: &[u8], out_buf: &mut Vec<u8>) {
    let check_sum_offset = compressed_pack.len() - 4;
    let checksum = (&compressed_pack[check_sum_offset..]).get_u32_le();
    compressed_pack = &compressed_pack[..check_sum_offset];
    if crc32fast::hash(compressed_pack) != checksum {
        panic!("checksum mismatch");
    }
    let last_offset = compressed_pack.len() - 1;
    let compression_type = compressed_pack[last_offset];
    compressed_pack = &compressed_pack[..last_offset];
    if compression_type == LZ4_COMPRESSION {
        let decompressed_len = compressed_pack.get_u32_le() as usize;
        out_buf.resize(decompressed_len, 0);
        lz4::block::decompress_to_buffer(compressed_pack, Some(decompressed_len as i32), out_buf)
            .unwrap();
    } else {
        panic!("unsupported compression type");
    }
}

pub(crate) fn compress_pack(uncompressed_buf: &[u8], compressed_buf: &mut Vec<u8>) -> Vec<u8> {
    let compress_bound: i32 =
        unsafe { lz4::liblz4::LZ4_compressBound(uncompressed_buf.len() as i32) };
    compressed_buf.resize(compress_bound as usize + 4, 0);
    let size =
        lz4::block::compress_to_buffer(uncompressed_buf, None, true, compressed_buf).unwrap();
    let mut compressed_pack = Vec::with_capacity(size + 1 + 4);
    compressed_pack.extend_from_slice(&compressed_buf[..size]);
    compressed_pack.put_u8(LZ4_COMPRESSION);
    let checksum = crc32fast::hash(&compressed_pack);
    compressed_pack.put_u32_le(checksum);
    compressed_pack
}
