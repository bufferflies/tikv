// Copyright 2024 TiKV Project Authors. Licensed under Apache-2.0.

use std::{convert::TryInto, ops::Deref, sync::Arc};

use aligned_vec::{avec, AVec};
use bytes::{Buf, BufMut};
use collections::HashMap;
use kvenginepb::ColumnarCreate;
use protobuf::Message;
use schema::schema::StorageClass;
use tidb_query_datatype::{FieldTypeAccessor, FieldTypeFlag, FieldTypeTp};
use tipb::ColumnInfo;

use crate::table::{
    columnar::builder::{
        TableOffsets, ENCODING_TYPE_NONE, PACK_FORMAT, PROP_KEY_BIGGEST, PROP_KEY_MAX_VERSION,
        PROP_KEY_SMALLEST, PROP_KEY_SNAP_VERSION,
    },
    file::File,
    parse_prop_data, search,
    sstable::{L0Table, PROP_KEY_ENCRYPTION_VER},
    BoundedDataSet, DataBound, InnerKey, LZ4_COMPRESSION,
};

pub const HANDLE_COL_ID: i32 = -1;
pub(crate) const VERSION_COL_ID: i32 = -1024;

pub const COLUMNAR_MAGIC: u32 = 0xc01e32ae;

#[derive(Default, Clone, Debug, PartialEq)]
pub struct SchemaBuf {
    pub inner: Arc<SchemaBufInner>,
    pub partitions: Option<Vec<(i64, StorageClass)>>,
    pub table_id: i64,
    pub property_sc: StorageClass,
    pub is_sub_partition: bool,
}

impl Deref for SchemaBuf {
    type Target = SchemaBufInner;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl SchemaBuf {
    pub fn new(
        table_id: i64,
        handle_column: ColumnInfo,
        version_column: ColumnInfo,
        columns: Vec<ColumnInfo>,
        pk_col_ids: Vec<i64>,
        vector_indexes: Vec<VectorIndexDef>,
        property_sc: StorageClass,
        partitions: Option<Vec<(i64, StorageClass)>>,
    ) -> Self {
        Self {
            table_id,
            inner: Arc::new(SchemaBufInner {
                handle_column,
                version_column,
                columns,
                pk_col_ids,
                vector_indexes,
            }),
            partitions,
            property_sc,
            is_sub_partition: false,
        }
    }

    pub fn retain_columns<F>(&self, f: F) -> Self
    where
        F: FnMut(&ColumnInfo) -> bool,
    {
        let mut columns = self.inner.columns.clone();
        columns.retain(f);
        SchemaBuf::new(
            self.table_id,
            self.handle_column.clone(),
            self.version_column.clone(),
            columns,
            self.pk_col_ids.clone(),
            self.vector_indexes.clone(),
            self.property_sc,
            self.partitions.clone(),
        )
    }

    pub fn to_partition_schema(&self, partition_id: i64, sc: StorageClass) -> SchemaBuf {
        SchemaBuf {
            table_id: partition_id,
            inner: self.inner.clone(),
            partitions: None,
            property_sc: sc,
            is_sub_partition: true,
        }
    }

    pub fn to_partition_schemas(&self) -> Vec<Schema> {
        self.partitions
            .as_ref()
            .map(|btree| {
                btree
                    .iter()
                    .map(|(id, sc)| self.to_partition_schema(*id, *sc).into())
                    .collect()
            })
            .unwrap_or_default()
    }

    #[inline]
    pub fn is_sub_partition(&self) -> bool {
        self.is_sub_partition
    }

    #[inline]
    pub fn set_storage_class(&mut self, storage_class: StorageClass) {
        self.property_sc = storage_class;
    }

    #[inline]
    pub fn get_storage_class(&self) -> StorageClass {
        self.property_sc
    }

    #[inline]
    pub fn is_partitioned_table(&self) -> bool {
        self.partitions.is_some()
    }

    pub fn remove_pk_col_from_columns(&mut self) {
        let handle_col_id = self.handle_column.get_column_id();
        let mut columns = self.columns.clone();
        columns.retain(|c| c.get_column_id() != handle_col_id);
        self.inner = Arc::new(SchemaBufInner {
            handle_column: self.handle_column.clone(),
            version_column: self.version_column.clone(),
            columns,
            pk_col_ids: self.pk_col_ids.clone(),
            vector_indexes: self.vector_indexes.clone(),
        });
    }
}

#[derive(Default, Clone, Debug, PartialEq)]
pub struct SchemaBufInner {
    pub handle_column: ColumnInfo,
    pub version_column: ColumnInfo,
    pub columns: Vec<ColumnInfo>,
    pub pk_col_ids: Vec<i64>,
    pub vector_indexes: Vec<VectorIndexDef>,
}

#[derive(Default, Clone, Debug, PartialEq)]
pub struct VectorIndexDef {
    pub index_id: i64,
    pub col_id: i64,
    pub dimension: usize,
    pub index_kind: String,
    pub specs: HashMap<String, Vec<u8>>,
}

impl VectorIndexDef {
    pub fn to_pb(&self) -> kvenginepb::VectorIndexDef {
        let mut vec_idx_pb = kvenginepb::VectorIndexDef::new();
        vec_idx_pb.set_index_id(self.index_id);
        vec_idx_pb.set_col_id(self.col_id);
        vec_idx_pb.set_index_kind(self.index_kind.clone());
        for (k, v) in &self.specs {
            vec_idx_pb.mut_spec_keys().push(k.clone());
            vec_idx_pb.mut_spec_values().push(v.clone());
        }
        vec_idx_pb
    }

    pub fn from_pb(dimension: usize, vec_idx_pb: &kvenginepb::VectorIndexDef) -> Self {
        let mut specs = HashMap::default();
        for i in 0..vec_idx_pb.spec_keys.len() {
            let key = vec_idx_pb.spec_keys[i].clone();
            let val = vec_idx_pb.spec_values[i].clone();
            specs.insert(key, val);
        }
        Self {
            index_id: vec_idx_pb.index_id,
            col_id: vec_idx_pb.col_id,
            dimension,
            index_kind: vec_idx_pb.index_kind.clone(),
            specs,
        }
    }
}

#[derive(Default, Clone, Debug, PartialEq)]
pub struct Schema {
    core: Arc<SchemaBuf>,
}

impl Deref for Schema {
    type Target = SchemaBuf;

    fn deref(&self) -> &Self::Target {
        &self.core
    }
}

impl From<SchemaBuf> for Schema {
    fn from(buf: SchemaBuf) -> Self {
        Self::new(buf)
    }
}

impl Schema {
    pub fn new(buf: SchemaBuf) -> Self {
        Self {
            core: Arc::new(buf),
        }
    }

    pub fn is_common_handle(&self) -> bool {
        get_fixed_size(&self.handle_column) == 0
    }

    pub fn with_columnar(&self) -> bool {
        self.handle_column.has_column_id()
            || !self.columns.is_empty()
            || !self.pk_col_ids.is_empty()
            || !self.vector_indexes.is_empty()
    }

    pub fn find_column_by_id(&self, id: i64) -> Option<&ColumnInfo> {
        if let Some(c) = self.columns.iter().find(|c| c.get_column_id() == id) {
            return Some(c);
        }
        if self.handle_column.get_column_id() == id {
            return Some(&self.handle_column);
        }
        None
    }

    pub fn to_schema_buf(&self) -> SchemaBuf {
        self.deref().clone()
    }
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

    pub const fn compute_size() -> usize {
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
        let idx = if handle.is_empty() {
            0
        } else if self.buf.fixed_size > 0 {
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
    pub(crate) columns: HashMap<i32, Arc<ColumnMeta>>,
}

impl TableMeta {
    pub(crate) fn parse(table_id: i64, mut buf: &[u8]) -> Self {
        let num_cols = buf.get_u32_le();
        let (handle_column, remained) = ColumnMeta::parse(buf);
        buf = remained;
        let (version_column, remained) = ColumnMeta::parse(buf);
        buf = remained;
        let mut columns = HashMap::default();
        let mut pk_col_ids = vec![];
        for _ in 0..(num_cols - 3) {
            let (col, remained) = ColumnMeta::parse(buf);
            buf = remained;
            if get_primary_key(&col.col_info) {
                pk_col_ids.push(col.col_info.get_column_id());
            }
            columns.insert(col.col_info.get_column_id() as i32, Arc::new(col));
        }
        let mut uncompressed_buf = avec![];
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
            columns,
        }
    }
}

pub struct MinMaxIndex {
    pub(crate) min_max: ColumnBuffer,
    pub(crate) has_null_marks: Vec<u8>,
    pub(crate) has_value_marks: Vec<u8>,
}

impl MinMaxIndex {
    pub(crate) fn new(col_id: i32, fixed_size: usize, nullable: bool) -> Self {
        let min_max = ColumnBuffer::new(col_id, fixed_size, nullable);
        Self {
            min_max,
            has_null_marks: vec![],
            has_value_marks: vec![],
        }
    }

    pub(crate) fn push_value(&mut self, data: &[u8]) {
        self.min_max.push_value(data);
    }

    pub(crate) fn push_null(&mut self) {
        self.min_max.push_null();
    }

    pub(crate) fn push_has_null(&mut self, has_null: bool) {
        self.has_null_marks.push(has_null as u8);
    }

    pub(crate) fn push_has_value(&mut self, has_value: bool) {
        self.has_value_marks.push(has_value as u8);
    }

    pub(crate) fn parse(&mut self, mut buf: &[u8]) {
        self.min_max.reset();
        buf = self.min_max.parse(buf);
        let length = self.min_max.length() / 2;
        self.has_null_marks.clear();
        self.has_null_marks.extend_from_slice(&buf[..length]);
        buf = &buf[length..];
        self.has_value_marks.clear();
        self.has_value_marks.extend_from_slice(&buf[..length]);
    }

    pub(crate) fn write_to(&self, buf: &mut Vec<u8>) {
        buf.clear();
        self.min_max.append_to(buf);
        buf.extend_from_slice(&self.has_null_marks);
        buf.extend_from_slice(&self.has_value_marks);
    }

    pub(crate) fn check_equal(
        &self,
        pack_idx: usize,
        value: &[u8],
        field_type: FieldTypeTp,
        is_unsigned: bool,
    ) -> bool {
        if self.has_value_marks[pack_idx] == 0 {
            return false;
        }
        let min = self.min_max.get_not_null_value(pack_idx * 2);
        let max = self.min_max.get_not_null_value(pack_idx * 2 + 1);
        // value >= min && value <= max
        field_cmp(value, min, field_type, is_unsigned) >= std::cmp::Ordering::Equal
            && field_cmp(value, max, field_type, is_unsigned) <= std::cmp::Ordering::Equal
    }

    pub(crate) fn check_not_equal(
        &self,
        pack_idx: usize,
        value: &[u8],
        field_type: FieldTypeTp,
        is_unsigned: bool,
    ) -> bool {
        if self.has_value_marks[pack_idx] == 0 {
            return false;
        }
        let min = self.min_max.get_not_null_value(pack_idx * 2);
        let max = self.min_max.get_not_null_value(pack_idx * 2 + 1);
        // value != min || value != max
        field_cmp(value, min, field_type, is_unsigned) != std::cmp::Ordering::Equal
            || field_cmp(value, max, field_type, is_unsigned) != std::cmp::Ordering::Equal
    }

    pub(crate) fn check_less(
        &self,
        pack_idx: usize,
        value: &[u8],
        field_type: FieldTypeTp,
        is_unsigned: bool,
    ) -> bool {
        if self.has_value_marks[pack_idx] == 0 {
            return false;
        }
        let min = self.min_max.get_not_null_value(pack_idx * 2);
        // value > min
        field_cmp(value, min, field_type, is_unsigned) == std::cmp::Ordering::Greater
    }

    pub(crate) fn check_less_equal(
        &self,
        pack_idx: usize,
        value: &[u8],
        field_type: FieldTypeTp,
        is_unsigned: bool,
    ) -> bool {
        if self.has_value_marks[pack_idx] == 0 {
            return false;
        }
        let min = self.min_max.get_not_null_value(pack_idx * 2);
        // value >= min
        field_cmp(value, min, field_type, is_unsigned) >= std::cmp::Ordering::Equal
    }

    pub(crate) fn check_greater(
        &self,
        pack_idx: usize,
        value: &[u8],
        field_type: FieldTypeTp,
        is_unsigned: bool,
    ) -> bool {
        if self.has_value_marks[pack_idx] == 0 {
            return false;
        }
        let max = self.min_max.get_not_null_value(pack_idx * 2 + 1);
        // value < max
        field_cmp(value, max, field_type, is_unsigned) == std::cmp::Ordering::Less
    }

    pub(crate) fn check_greater_equal(
        &self,
        pack_idx: usize,
        value: &[u8],
        field_type: FieldTypeTp,
        is_unsigned: bool,
    ) -> bool {
        if self.has_value_marks[pack_idx] == 0 {
            return false;
        }
        let max = self.min_max.get_not_null_value(pack_idx * 2 + 1);
        // value <= max
        field_cmp(value, max, field_type, is_unsigned) <= std::cmp::Ordering::Equal
    }

    pub(crate) fn check_in(
        &self,
        pack_idx: usize,
        values: &Vec<Vec<u8>>,
        field_type: FieldTypeTp,
        is_unsigned: bool,
    ) -> bool {
        if self.has_value_marks[pack_idx] == 0 {
            return false;
        }
        for value in values {
            if self.check_equal(pack_idx, value, field_type, is_unsigned) {
                return true;
            }
        }
        false
    }

    pub(crate) fn check_not_in(
        &self,
        pack_idx: usize,
        values: &Vec<Vec<u8>>,
        field_type: FieldTypeTp,
        is_unsigned: bool,
    ) -> bool {
        if self.has_value_marks[pack_idx] == 0 {
            return false;
        }
        !self.check_in(pack_idx, values, field_type, is_unsigned)
    }

    pub(crate) fn check_is_null(&self, pack_idx: usize) -> bool {
        if self.has_value_marks[pack_idx] == 0 {
            return false;
        }
        if self.has_null_marks[pack_idx] == 0 {
            return false;
        }
        true
    }
}

fn field_cmp(
    mut a: &[u8],
    mut b: &[u8],
    field_type: FieldTypeTp,
    is_unsigned: bool,
) -> std::cmp::Ordering {
    match field_type {
        FieldTypeTp::Unspecified => {
            panic!("field type is unspecified");
        }
        FieldTypeTp::Float | FieldTypeTp::Double => {
            let a = a.get_f64_le();
            let b = b.get_f64_le();
            a.partial_cmp(&b).unwrap()
        }
        FieldTypeTp::Null => std::cmp::Ordering::Equal,
        FieldTypeTp::Tiny
        | FieldTypeTp::Short
        | FieldTypeTp::Int24
        | FieldTypeTp::Long
        | FieldTypeTp::LongLong => {
            if is_unsigned {
                let a = a.get_u64_le();
                let b = b.get_u64_le();
                a.cmp(&b)
            } else {
                let a = a.get_i64_le();
                let b = b.get_i64_le();
                a.cmp(&b)
            }
        }
        FieldTypeTp::Duration | FieldTypeTp::Year => {
            let a = a.get_i64_le();
            let b = b.get_i64_le();
            a.cmp(&b)
        }
        FieldTypeTp::Timestamp
        | FieldTypeTp::DateTime
        | FieldTypeTp::Date
        | FieldTypeTp::NewDate
        | FieldTypeTp::Bit
        | FieldTypeTp::Enum => {
            let a = a.get_u64_le();
            let b = b.get_u64_le();
            a.cmp(&b)
        }
        _ => {
            unimplemented!("field type is not supported");
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
    pub(crate) min_max: Option<MinMaxIndex>,
    pub(crate) compressed_min_max_pack: Vec<u8>,
}

impl ColumnMeta {
    pub(crate) fn new(col_info: ColumnInfo, need_min_max: bool) -> Self {
        let col_id = col_info.get_column_id() as i32;
        let fixed_size = get_fixed_size(&col_info);
        let nullable = col_info.get_flag() as u32 & FieldTypeFlag::NOT_NULL.bits() == 0;
        let min_max = (can_build_min_max(&col_info) && need_min_max)
            .then(|| MinMaxIndex::new(col_id, fixed_size, nullable));
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
            let mut uncompressed_min_max_buf = avec![];
            decompress_pack(compressed_min_max_idx_buf, &mut uncompressed_min_max_buf);
            let mut min_max = MinMaxIndex::new(col_id, fixed_size, nullable);
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

#[derive(Clone)]
pub struct ColumnarFile {
    core: Arc<ColumnarFileCore>,
}

impl ColumnarFile {
    pub fn open(file: Arc<dyn File>) -> crate::table::Result<Self> {
        let file_len = file.size();
        let footer_len = ColumnarFileFooter::compute_size();
        let footer_offset = file_len - footer_len as u64;
        let footer_data = file.read_footer(footer_len)?;
        let footer = ColumnarFileFooter::parse(&footer_data);

        let table_offsets_size = TableOffsets::compute_size(footer.number_tables as usize) as u64;
        let table_offsets_offset = footer_offset - table_offsets_size;
        let buf = file.read_table_meta(table_offsets_offset, table_offsets_size as usize)?;
        let table_offsets = TableOffsets::parse(&buf, footer.number_tables);
        let mut smallest_key = vec![];
        let mut biggest_key = vec![];
        let mut max_version = 0;
        let mut l0_version = None;
        let mut encryption_ver = 0;
        let property_offset = table_offsets_offset - footer.properties_size as u64;
        let property_buf =
            file.read_table_meta(property_offset, footer.properties_size as usize)?;
        let mut prop_remain = property_buf.as_ref();
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
            } else if key == PROP_KEY_ENCRYPTION_VER.as_bytes() {
                encryption_ver = val.get_u32_le();
            }
            prop_remain = remain;
        }
        let mut tables = HashMap::default();
        let index_offset = table_offsets.index_offset();
        for i in 0..footer.number_tables as usize {
            let (idx_start, idx_end) = table_offsets.get_index_range(i);
            let table_index_buf = file.read_table_meta(
                (index_offset + idx_start) as u64,
                (idx_end - idx_start) as usize,
            )?;
            let table_meta = TableMeta::parse(table_offsets.table_ids[i], &table_index_buf);
            tables.insert(table_offsets.table_ids[i], Arc::new(table_meta));
        }
        Ok(Self {
            core: Arc::new(ColumnarFileCore {
                file,
                smallest_key,
                biggest_key,
                max_version,
                l0_version,
                tables,
                encryption_ver,
                index_offset,
            }),
        })
    }

    pub(crate) fn get_table(&self, table_id: i64) -> Arc<TableMeta> {
        self.core.tables.get(&table_id).unwrap().clone()
    }

    pub(crate) fn has_table(&self, table_id: i64) -> bool {
        self.core.tables.contains_key(&table_id)
    }

    pub fn get_file(&self) -> Arc<dyn File> {
        self.core.file.clone()
    }

    pub fn id(&self) -> u64 {
        self.core.file.id()
    }

    pub fn get_smallest(&self) -> InnerKey<'_> {
        InnerKey::from_inner_buf(&self.core.smallest_key)
    }

    pub fn get_biggest(&self) -> InnerKey<'_> {
        InnerKey::from_inner_buf(&self.core.biggest_key)
    }

    /// The offset of first table meta (index).
    #[inline]
    pub fn get_meta_offset(&self) -> u32 {
        self.get_index_offset()
    }

    pub fn get_index_offset(&self) -> u32 {
        self.core.index_offset
    }

    pub fn has_data_in_range(&self, start_key: InnerKey<'_>, end_key: InnerKey<'_>) -> bool {
        self.get_smallest() < end_key && self.get_biggest() >= start_key
    }

    pub fn get_max_version(&self) -> u64 {
        self.core.max_version
    }

    pub fn get_l0_version(&self) -> Option<u64> {
        self.core.l0_version
    }

    pub fn size(&self) -> u64 {
        self.core.file.size()
    }

    pub fn table_count(&self) -> usize {
        self.core.tables.len()
    }

    pub fn get_encryption_ver(&self) -> u32 {
        self.core.encryption_ver
    }

    pub fn to_columnar_create(&self, lvl: usize) -> ColumnarCreate {
        let mut columnar_create = ColumnarCreate::new();
        columnar_create.set_id(self.id());
        columnar_create.set_level(lvl as u32);
        columnar_create.set_smallest(self.get_smallest().to_vec());
        columnar_create.set_biggest(self.get_biggest().to_vec());
        columnar_create.set_meta_offset(self.get_meta_offset());
        columnar_create
    }
}

impl BoundedDataSet for ColumnarFile {
    fn data_bound(&self) -> DataBound<'_> {
        DataBound::new(self.get_smallest(), self.get_biggest(), true)
    }
}

struct ColumnarFileCore {
    file: Arc<dyn File>,
    smallest_key: Vec<u8>,
    biggest_key: Vec<u8>,
    max_version: u64,
    l0_version: Option<u64>,
    tables: HashMap<i64, Arc<TableMeta>>,
    encryption_ver: u32,
    index_offset: u32,
}

pub struct ColumnBuffer {
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

    // Used by proxy.
    pub fn col_id(&self) -> i32 {
        self.col_id
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

    pub(crate) fn reset(&mut self) {
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
    // version < safe_ts when delete is true represents tombstone.
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

    #[inline]
    pub fn get_not_null_value(&self, idx: usize) -> &[u8] {
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

    pub fn get_value(&self, idx: usize) -> Option<&[u8]> {
        if self.nullable && self.nulls[idx] == 1 {
            return None;
        }
        Some(self.get_not_null_value(idx))
    }

    #[inline]
    pub fn get_int_handle_value(&self, idx: usize) -> i64 {
        debug_assert!(self.fixed_size == 8);
        let start = idx * self.fixed_size;
        let end = (idx + 1) * self.fixed_size;
        (&self.data_buf[start..end]).get_i64_le()
    }

    pub fn get_version(&self, idx: usize) -> u64 {
        debug_assert!(self.col_id == VERSION_COL_ID);
        (&self.data_buf[idx * 8..]).get_u64_le()
    }

    pub fn is_null(&self, idx: usize) -> bool {
        debug_assert!(self.nullable);
        self.nulls[idx] == 1
    }

    pub fn is_nullable(&self) -> bool {
        self.nullable
    }

    pub fn get_fixed_size(&self) -> usize {
        self.fixed_size
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

    pub(crate) fn parse<'a>(&mut self, mut uncompressed_pack: &'a [u8]) -> &'a [u8] {
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
            self.nulls.extend_from_slice(&uncompressed_pack[..length]);
            uncompressed_pack = &uncompressed_pack[length..]
        }
        uncompressed_pack
    }

    pub(crate) fn write_to(&self, buf: &mut Vec<u8>) {
        buf.clear();
        self.append_to(buf);
    }

    pub(crate) fn append_to(&self, buf: &mut Vec<u8>) {
        buf.put_u16_le(PACK_FORMAT);
        buf.put_u16_le(ENCODING_TYPE_NONE);
        buf.put_u32_le(self.length() as u32);
        if self.fixed_size == 0 {
            buf.extend_from_slice(bytemuck::cast_slice(&self.offsets));
        }
        buf.extend_from_slice(&self.data_buf);
        buf.extend_from_slice(&self.nulls);
    }

    fn write_var_u64<B: BufMut>(buf: &mut B, mut value: u64) {
        loop {
            if value < 0x80 {
                buf.put_u8(value as u8);
                break;
            } else {
                buf.put_u8(((value & 0x7F) | 0x80) as u8);
                value >>= 7;
            }
        }
    }

    fn compute_size_for_tiflash(&self, tp: FieldTypeTp, column_len: usize) -> usize {
        let data_size = match tp {
            FieldTypeTp::NewDecimal => {
                let item_size = match column_len {
                    0..=9 => 4,
                    10..=18 => 8,
                    19..=38 => 16,
                    39..=65 => 32,
                    _ => panic!("unsupported precision: {}", column_len),
                };
                item_size * self.length()
            }
            FieldTypeTp::String
            | FieldTypeTp::VarChar
            | FieldTypeTp::VarString
            | FieldTypeTp::Blob
            | FieldTypeTp::TinyBlob
            | FieldTypeTp::MediumBlob
            | FieldTypeTp::LongBlob
            | FieldTypeTp::Json => self.offsets.last().copied().unwrap() as usize + self.length() * 4 /* varint */,
            FieldTypeTp::Tiny => self.length(),
            FieldTypeTp::Short => self.length() * 2,
            FieldTypeTp::Int24 | FieldTypeTp::Long => self.length() * 4,
            FieldTypeTp::LongLong
            | FieldTypeTp::Double
            | FieldTypeTp::Date
            | FieldTypeTp::DateTime
            | FieldTypeTp::Timestamp
            | FieldTypeTp::Bit
            | FieldTypeTp::Set
            | FieldTypeTp::Enum
            | FieldTypeTp::Duration => self.length() * 8,
            FieldTypeTp::Float => self.length() * 4,
            FieldTypeTp::Year => self.length() * 2,
            FieldTypeTp::Null => self.length(),
            FieldTypeTp::TiDbVectorFloat32 => self.length() * 8 + self.length() * column_len * 4,
            FieldTypeTp::NewDate => {
                unimplemented!();
            }
            FieldTypeTp::Unspecified => self.length(),
            _ => unreachable!(),
        };
        if self.is_nullable() {
            self.length() /* nulls map */ + data_size
        } else {
            data_size
        }
    }

    // Used by proxy.
    pub fn serialize_for_tiflash(
        &self,
        buf: &mut Vec<u8>,
        tp: i32,           // column type
        is_unsigned: bool, // used by integer number types
        column_len: usize,
    ) {
        buf.clear();
        let tp = FieldTypeTp::from_i32(tp).unwrap();
        let target_size = self.compute_size_for_tiflash(tp, column_len);
        buf.reserve(target_size);
        if self.is_nullable() {
            buf.extend_from_slice(&self.nulls);
        }
        match tp {
            FieldTypeTp::NewDecimal => {
                let null_bytes: &'static [u8] = match column_len {
                    0..=9 => &[0; 4],
                    10..=18 => &[0; 8],
                    19..=38 => &[0; 16],
                    39..=65 => &[0; 32],
                    _ => panic!("unsupported precision: {}", column_len),
                };
                for idx in 0..self.length() {
                    match self.get_value(idx) {
                        Some(v) => {
                            buf.extend_from_slice(v);
                        }
                        None => {
                            buf.extend_from_slice(null_bytes);
                        }
                    }
                }
            }
            FieldTypeTp::String
            | FieldTypeTp::VarChar
            | FieldTypeTp::VarString
            | FieldTypeTp::Blob
            | FieldTypeTp::TinyBlob
            | FieldTypeTp::MediumBlob
            | FieldTypeTp::LongBlob
            | FieldTypeTp::Json => {
                for idx in 0..self.length() {
                    match self.get_value(idx) {
                        Some(v) => {
                            let val_len = v.len() as u64;
                            Self::write_var_u64(buf, val_len);
                            buf.extend_from_slice(v);
                        }
                        None => {
                            buf.push(0);
                        }
                    }
                }
            }
            FieldTypeTp::Tiny => {
                for idx in 0..self.length() {
                    match self.get_value(idx) {
                        Some(v) => {
                            if is_unsigned {
                                let i_val = u64::from_le_bytes(v.try_into().unwrap()) as u8;
                                buf.extend_from_slice(&i_val.to_le_bytes());
                            } else {
                                let i_val = i64::from_le_bytes(v.try_into().unwrap()) as i8;
                                buf.extend_from_slice(&i_val.to_le_bytes());
                            }
                        }
                        None => {
                            buf.push(0);
                        }
                    }
                }
            }
            FieldTypeTp::Short => {
                for idx in 0..self.length() {
                    match self.get_value(idx) {
                        Some(v) => {
                            if is_unsigned {
                                let i_val = u64::from_le_bytes(v.try_into().unwrap()) as u16;
                                buf.extend_from_slice(&i_val.to_le_bytes());
                            } else {
                                let i_val = i64::from_le_bytes(v.try_into().unwrap()) as i16;
                                buf.extend_from_slice(&i_val.to_le_bytes());
                            }
                        }
                        None => {
                            buf.extend_from_slice(&[0; 2]);
                        }
                    }
                }
            }
            FieldTypeTp::Int24 | FieldTypeTp::Long => {
                for idx in 0..self.length() {
                    match self.get_value(idx) {
                        Some(v) => {
                            if is_unsigned {
                                let i_val = u64::from_le_bytes(v.try_into().unwrap()) as u32;
                                buf.extend_from_slice(&i_val.to_le_bytes());
                            } else {
                                let i_val = i64::from_le_bytes(v.try_into().unwrap()) as i32;
                                buf.extend_from_slice(&i_val.to_le_bytes());
                            }
                        }
                        None => {
                            buf.extend_from_slice(&[0; 4]);
                        }
                    }
                }
            }
            FieldTypeTp::LongLong
            | FieldTypeTp::Double
            | FieldTypeTp::Date
            | FieldTypeTp::DateTime
            | FieldTypeTp::Timestamp
            | FieldTypeTp::Bit
            | FieldTypeTp::Set
            | FieldTypeTp::Enum
            | FieldTypeTp::Duration => {
                buf.extend_from_slice(&self.data_buf);
            }

            FieldTypeTp::Float => {
                for idx in 0..self.length() {
                    match self.get_value(idx) {
                        Some(v) => {
                            let f_val = f64::from_le_bytes(v.try_into().unwrap()) as f32;
                            buf.extend_from_slice(&f_val.to_le_bytes());
                        }
                        None => {
                            buf.extend_from_slice(&[0; 4]);
                        }
                    }
                }
            }
            FieldTypeTp::Year => {
                for idx in 0..self.length() {
                    match self.get_value(idx) {
                        Some(v) => {
                            let i_val = i64::from_le_bytes(v.try_into().unwrap()) as u16;
                            buf.extend_from_slice(&i_val.to_le_bytes());
                        }
                        None => {
                            buf.extend_from_slice(&[0; 2]);
                        }
                    }
                }
            }
            FieldTypeTp::Null => {
                buf.resize(buf.len() + self.length(), 0);
            }
            FieldTypeTp::TiDbVectorFloat32 => {
                let dimension = column_len;
                let arr_len = (dimension as u64).to_le_bytes();
                for _ in 0..self.length() {
                    buf.extend_from_slice(&arr_len);
                }
                for idx in 0..self.length() {
                    match self.get_value(idx) {
                        Some(v) => {
                            // skip the first 4 bytes which is the length of the array
                            buf.extend_from_slice(&v[4..]);
                        }
                        None => {
                            buf.resize(buf.len() + dimension * 4, 0);
                        }
                    }
                }
            }
            FieldTypeTp::NewDate => {
                // TODO
                unimplemented!();
            }

            FieldTypeTp::Unspecified => {}
            _ => unreachable!(),
        }
        debug!(
            "serialize_for_tiflash tp: {} length: {} data size: {}",
            tp,
            self.length(),
            buf.len()
        );
    }
}

pub struct Block {
    pub(crate) handles: ColumnBuffer,
    pub(crate) versions: ColumnBuffer,
    pub(crate) columns: Vec<ColumnBuffer>,
}

impl Block {
    pub fn new(schema: &Schema) -> Self {
        let handles = ColumnBuffer::new_from_col_info(&schema.handle_column);
        let versions = ColumnBuffer::new_from_col_info(&schema.version_column);
        let mut columns = vec![];
        for col_info in &schema.columns {
            if col_info.get_column_id() == schema.handle_column.get_column_id() {
                continue;
            }
            columns.push(ColumnBuffer::new_from_col_info(col_info));
        }
        Self {
            handles,
            versions,
            columns,
        }
    }

    // Also used in proxy kvengine.
    pub fn reset(&mut self) {
        self.handles.reset();
        self.versions.reset();
        self.columns.iter_mut().for_each(|col| col.reset());
    }

    pub(crate) fn truncate(&mut self, length: usize) {
        self.handles.truncate(length);
        self.versions.truncate(length);
        for col in &mut self.columns {
            col.truncate(length);
        }
    }

    pub(crate) fn append(&mut self, other: &Block, row_offset: usize, row_end_offset: usize) {
        self.handles
            .append(&other.handles, row_offset, row_end_offset);
        self.versions
            .append(&other.versions, row_offset, row_end_offset);
        for (col, other_col) in self.columns.iter_mut().zip(&other.columns) {
            col.append(other_col, row_offset, row_end_offset);
        }
    }

    pub fn length(&self) -> usize {
        self.handles.length()
    }

    pub fn get_handle_buf(&self) -> &ColumnBuffer {
        &self.handles
    }

    pub fn get_version_buf(&self) -> &ColumnBuffer {
        &self.versions
    }

    pub fn get_columns(&self) -> &[ColumnBuffer] {
        &self.columns
    }

    pub fn get_column(&self, col_id: i64) -> &ColumnBuffer {
        for col in &self.columns {
            if col.col_id() == col_id as i32 {
                return col;
            }
        }
        unreachable!()
    }
}

pub(crate) fn get_fixed_size(col_info: &ColumnInfo) -> usize {
    let tp = FieldTypeTp::from_i32(col_info.get_tp()).unwrap();
    match tp {
        FieldTypeTp::Float
        | FieldTypeTp::Tiny
        | FieldTypeTp::Short
        | FieldTypeTp::Int24
        | FieldTypeTp::Long
        | FieldTypeTp::Double
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
        FieldTypeTp::NewDecimal => 0,
        _ => 0,
    }
}

pub(crate) fn can_build_min_max(col_info: &ColumnInfo) -> bool {
    let tp = FieldTypeTp::from_i32(col_info.get_tp()).unwrap();
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
    !col_info.flag().contains(FieldTypeFlag::NOT_NULL)
}

pub(crate) fn get_unsigned(col_info: &ColumnInfo) -> bool {
    col_info.flag().contains(FieldTypeFlag::UNSIGNED)
}

pub(crate) fn get_primary_key(col_info: &ColumnInfo) -> bool {
    col_info.flag().contains(FieldTypeFlag::PRIMARY_KEY)
}

pub(crate) fn decompress_pack(mut compressed_pack: &[u8], out_buf: &mut AVec<u8>) {
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

#[derive(Clone, Default)]
pub(crate) struct ColumnarLevel {
    pub(crate) level: usize,
    pub(crate) files: Vec<ColumnarFile>,
}

impl ColumnarLevel {
    pub(crate) fn new(level: usize) -> Self {
        Self {
            level,
            files: vec![],
        }
    }

    pub(crate) fn sort(&mut self) {
        if self.level < 2 {
            self.files.sort_by(|a, b| {
                let a_l0_version = a.get_l0_version().unwrap();
                let b_l0_version = b.get_l0_version().unwrap();
                b_l0_version.cmp(&a_l0_version)
            })
        } else {
            self.files
                .sort_by(|a, b| a.get_smallest().cmp(&b.get_smallest()))
        }
    }
}

#[derive(Clone)]
pub(crate) struct ColumnarLevels {
    pub(crate) unconverted_l0s: Vec<L0Table>,
    pub(crate) levels: Vec<ColumnarLevel>,
    pub(crate) l2_snap_version: u64,
}

impl ColumnarLevels {
    pub(crate) fn new() -> Self {
        Self {
            unconverted_l0s: vec![],
            levels: vec![
                ColumnarLevel::new(0),
                ColumnarLevel::new(1),
                ColumnarLevel::new(2),
            ],
            l2_snap_version: 0,
        }
    }

    pub(crate) fn sort(&mut self) {
        self.levels.iter_mut().for_each(|l| l.sort());
    }

    pub(crate) fn add_file(&mut self, level: usize, file: ColumnarFile) {
        self.levels[level].files.push(file);
    }

    pub(crate) fn retain(&mut self, f: impl Fn(&ColumnarFile) -> bool) {
        self.levels.iter_mut().for_each(|l| l.files.retain(&f));
    }
}
