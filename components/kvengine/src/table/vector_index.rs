// Copyright 2024 TiKV Project Authors. Licensed under Apache-2.0.

use std::{collections::HashSet, ops::Deref, sync::Arc};

use async_trait::async_trait;
use bytes::{Buf, BufMut};
use cloud_encryption::EncryptionKey;
use tidb_query_datatype::codec::{
    mysql::{VectorFloat32Encoder, VectorFloat32Ref},
    table::{encode_common_handle_row_key, encode_row_key},
};
use usearch::IndexOptions;

use crate::{
    table,
    table::{
        columnar::{
            get_fixed_size, Block, ColumnarConcatReader, ColumnarLevels, ColumnarMergeReader,
            ColumnarReader, ColumnarTableReader, Schema,
        },
        file::{File, MmapData},
        search, BoundedDataSet, DataBound, Error,
        Error::Other,
        InnerKey, Result,
    },
};

const MAGIC_NUMBER: u32 = 0x19504cf0;
const FORMAT_VERSION: u32 = 1;
const CONNECTIVITY: usize = 16;
const EXPANSION_ADD: usize = 128;
const EXPANSION_SEARCH: usize = 64;

const QUANTIZATION: usearch::ScalarKind = usearch::ScalarKind::F32;

#[derive(Default, Clone)]
pub struct VectorIndexes {
    indexes: Vec<VectorIndex>,
}

impl VectorIndexes {
    pub(crate) fn is_empty(&self) -> bool {
        self.indexes.is_empty()
    }

    pub(crate) fn get_all(&self) -> &[VectorIndex] {
        &self.indexes
    }

    pub fn add_index_file(&mut self, vec_idx_file: VectorIndexFile) {
        for index in &mut self.indexes {
            if index.table_id == vec_idx_file.table_id()
                && index.index_id == vec_idx_file.index_id()
                && index.col_id == vec_idx_file.column_id()
            {
                if !index
                    .files
                    .iter()
                    .any(|f| f.file_id() == vec_idx_file.file_id())
                {
                    index.files.push(vec_idx_file);
                    index.sort();
                }
                return;
            }
        }
        let mut index = VectorIndex::new(
            vec_idx_file.table_id(),
            vec_idx_file.index_id(),
            vec_idx_file.column_id(),
        );
        index.files.push(vec_idx_file);
        self.indexes.push(index);
    }

    pub fn remove_index_file(
        &mut self,
        table_id: i64,
        index_id: i64,
        col_id: i64,
        remove_file_ids: &[u64],
    ) {
        for index in &mut self.indexes {
            if index.table_id == table_id && index.index_id == index_id && index.col_id == col_id {
                index
                    .files
                    .retain(|f| !remove_file_ids.contains(&f.file_id()));
                break;
            }
        }
        self.indexes.retain(|index| !index.files.is_empty());
    }

    pub fn get(&self, table_id: i64, index_id: i64, col_id: i64) -> Option<&VectorIndex> {
        self.indexes.iter().find(|vec_idx| {
            vec_idx.table_id == table_id && vec_idx.index_id == index_id && vec_idx.col_id == col_id
        })
    }

    pub fn get_mut(
        &mut self,
        table_id: i64,
        index_id: i64,
        col_id: i64,
    ) -> Option<&mut VectorIndex> {
        self.indexes
            .iter_mut()
            .find(|i| i.table_id == table_id && i.index_id == index_id && i.col_id == col_id)
    }

    pub fn sort(&mut self) {
        for index in &mut self.indexes {
            index.sort();
        }
    }
}

#[derive(Clone)]
pub struct VectorIndex {
    pub table_id: i64,
    pub index_id: i64,
    pub col_id: i64,
    pub(crate) files: Vec<VectorIndexFile>,
}

impl VectorIndex {
    pub(crate) fn new(table_id: i64, index_id: i64, col_id: i64) -> VectorIndex {
        VectorIndex {
            table_id,
            index_id,
            col_id,
            files: vec![],
        }
    }

    pub(crate) fn sort(&mut self) {
        self.files
            .sort_by(|a, b| b.snap_version().cmp(&a.snap_version()));
    }

    pub fn snap_version(&self) -> u64 {
        self.files[0].snap_version()
    }

    pub fn search(&self, target: &[f32], count: usize, start_ts: u64) -> Result<Vec<VectorItem>> {
        let mut results = vec![];
        let mut handles_dedup = HashSet::new();
        for file in &self.files {
            let items = file.search(target, count, start_ts)?;
            for item in items {
                if handles_dedup.contains(&item.handle) {
                    continue;
                }
                handles_dedup.insert(item.handle.clone());
                results.push(item);
            }
        }
        Ok(results)
    }

    pub fn to_vector_index_pb(&self) -> kvenginepb::VectorIndex {
        let mut vec_idx_pb = kvenginepb::VectorIndex::new();
        vec_idx_pb.set_table_id(self.table_id);
        vec_idx_pb.set_index_id(self.index_id);
        vec_idx_pb.set_col_id(self.col_id);
        for vec_idx_file in &self.files {
            vec_idx_pb
                .mut_files()
                .push(vec_idx_file.to_vector_index_file_pb());
        }
        vec_idx_pb
    }
}

// index file format:
//   index | versions | handles | props | footer
#[derive(Debug)]
#[repr(C)]
pub struct VectorIndexFileFooter {
    index_size: u32,
    props_size: u32,
    format_ver: u32,
    magic_number: u32,
}

impl Default for VectorIndexFileFooter {
    fn default() -> Self {
        VectorIndexFileFooter {
            index_size: 0,
            props_size: 0,
            format_ver: FORMAT_VERSION,
            magic_number: MAGIC_NUMBER,
        }
    }
}

impl VectorIndexFileFooter {
    fn unmarshal(&mut self, mut data: &[u8]) {
        self.index_size = data.get_u32_le();
        self.props_size = data.get_u32_le();
        self.format_ver = data.get_u32_le();
        self.magic_number = data.get_u32_le();
    }

    fn marshal(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(Self::size());
        buf.put_u32_le(self.index_size);
        buf.put_u32_le(self.props_size);
        buf.put_u32_le(self.format_ver);
        buf.put_u32_le(MAGIC_NUMBER);
        buf
    }

    fn size() -> usize {
        std::mem::size_of::<VectorIndexFileFooter>()
    }
}

#[derive(Clone)]
pub struct VectorIndexFile {
    core: Arc<VectorIndexFileCore>,
}

impl VectorIndexFile {
    pub fn file_id(&self) -> u64 {
        self.core.file.id()
    }

    pub fn file_size(&self) -> u64 {
        self.core.file.size()
    }

    pub fn snap_version(&self) -> u64 {
        self.core.snap_version
    }

    pub fn table_id(&self) -> i64 {
        self.core.table_id
    }

    pub fn index_id(&self) -> i64 {
        self.core.index_id as i64
    }

    pub fn column_id(&self) -> i64 {
        self.core.column_id as i64
    }

    pub fn smallest(&self) -> InnerKey<'_> {
        InnerKey::from_inner_buf(&self.core.smallest)
    }

    pub fn biggest(&self) -> InnerKey<'_> {
        InnerKey::from_inner_buf(&self.core.biggest)
    }
}

impl Deref for VectorIndexFile {
    type Target = VectorIndexFileCore;

    fn deref(&self) -> &Self::Target {
        &self.core
    }
}

impl BoundedDataSet for VectorIndexFile {
    fn data_bound(&self) -> DataBound<'_> {
        DataBound::new(self.smallest(), self.biggest(), true)
    }
}

pub struct VectorIndexFileCore {
    file: Arc<dyn File>,
    snap_version: u64,
    table_id: i64,
    index_id: i32,
    column_id: i32,
    is_common_handle: bool,
    smallest: Vec<u8>,
    biggest: Vec<u8>,
    handles_start: usize,
    handles_end: usize,
    versions_start: usize,
    versions_end: usize,
    index: usearch::Index,
    file_data: MmapData,
}

impl VectorIndexFile {
    pub fn new(file: Arc<dyn File>) -> Result<Self> {
        let mut footer = VectorIndexFileFooter::default();
        let file_data = file.mmap()?;
        let footer_size = std::mem::size_of::<VectorIndexFileFooter>();
        let footer_data = &file_data[file_data.len() - footer_size..];
        footer.unmarshal(footer_data);
        if footer.magic_number != MAGIC_NUMBER {
            return Err(Error::InvalidMagicNumber);
        }
        if footer.format_ver != FORMAT_VERSION {
            return Err(Other(format!(
                "unsupported format version: {}",
                footer.format_ver
            )));
        }
        let prop_size = footer.props_size as usize;
        let prop_offset = file_data.len() - footer_size - prop_size;
        let mut prop_data_buf = &file_data[prop_offset..prop_offset + prop_size];
        let mut snap_version = 0;
        let mut table_id = 0;
        let mut index_id = 0;
        let mut column_id = 0;
        let mut metric_repr = 0;
        let mut is_common_handle = false;
        let mut smallest = vec![];
        let mut biggest = vec![];
        while prop_data_buf.len() > 2 {
            let prop_key_len = prop_data_buf.get_u16_le();
            if prop_key_len == 0 {
                break;
            }
            let prop_key = &prop_data_buf[..prop_key_len as usize];
            prop_data_buf.advance(prop_key_len as usize);
            let prop_val_len = prop_data_buf.get_u32_le();
            let mut prop_value = &prop_data_buf[..prop_val_len as usize];
            prop_data_buf.advance(prop_val_len as usize);
            if prop_key == PROP_SMALLEST.as_bytes() {
                smallest = prop_value.to_vec();
            } else if prop_key == PROP_BIGGEST.as_bytes() {
                biggest = prop_value.to_vec();
            } else if prop_key == PROP_TABLE_ID.as_bytes() {
                table_id = prop_value.get_i64_le();
            } else if prop_key == PROP_INDEX_ID.as_bytes() {
                index_id = prop_value.get_i32_le();
            } else if prop_key == PROP_COLUMN_ID.as_bytes() {
                column_id = prop_value.get_i32_le();
            } else if prop_key == PROP_IS_COMMON_HANDLE.as_bytes() {
                is_common_handle = prop_value.get_u8() == 1;
            } else if prop_key == PROP_METRIC_REPR.as_bytes() {
                metric_repr = prop_value.get_i32_le();
            } else if prop_key == PROP_SNAP_VERSION.as_bytes() {
                snap_version = prop_value.get_u64_le();
            }
        }
        let index_data = &file_data[..footer.index_size as usize];
        let mut opts = new_index_opts();
        opts.metric.repr = metric_repr;
        let index = usearch::Index::new(&opts).map_err(|e| Other(e.to_string()))?;
        unsafe {
            index
                .view_from_buffer(index_data)
                .map_err(|e| Other(e.to_string()))?;
        }
        let versions_start = (index_data.len() + 7) & !7; // aligned to 8 bytes
        let versions_end = versions_start + index.size() * 8;
        let mut handles_start = versions_end;
        let handles_end;
        if is_common_handle {
            handles_start = versions_end + (index.size() + 1) * 4;
            let handle_offsets = &file_data[versions_end..handles_start];
            let handle_data_length =
                (&handle_offsets[handle_offsets.len() - 4..]).get_u32_le() as usize;
            handles_end = handles_start + handle_data_length;
        } else {
            handles_end = handles_start + index.size() * 8;
        };
        Ok(VectorIndexFile {
            core: Arc::new(VectorIndexFileCore {
                file,
                smallest,
                biggest,
                snap_version,
                table_id,
                index_id,
                column_id,
                is_common_handle,
                versions_start,
                versions_end,
                handles_start,
                handles_end,
                index,
                file_data,
            }),
        })
    }

    fn get_versions(&self) -> &[u64] {
        bytemuck::cast_slice(&self.file_data[self.versions_start..self.versions_end])
    }

    fn get_handles_offsets(&self) -> &[u32] {
        bytemuck::cast_slice(&self.file_data[self.versions_end..self.handles_start])
    }

    fn get_handles_data(&self) -> &[u8] {
        &self.file_data[self.handles_start..self.handles_end]
    }

    fn get_handle(&self, key: u64) -> &[u8] {
        let handles = self.get_handles_data();
        if self.is_common_handle {
            let handle_offsets = self.get_handles_offsets();
            let start = handle_offsets[key as usize] as usize;
            let end = handle_offsets[key as usize + 1] as usize;
            &handles[start..end]
        } else {
            let offset = key as usize * 8;
            &handles[offset..offset + 8]
        }
    }

    pub fn search(&self, target: &[f32], count: usize, start_ts: u64) -> Result<Vec<VectorItem>> {
        let versions = self.get_versions();
        let mut results = vec![];
        let matches = self
            .index
            .filtered_search(target, count, |key| {
                let handle = self.get_handle(key);
                let version = versions[key as usize];
                if version > start_ts {
                    return false;
                }
                if key > 0 {
                    let prev_key = key - 1;
                    let prev_handle = self.get_handle(prev_key);
                    if prev_handle == handle {
                        let prev_version = versions[prev_key as usize];
                        debug_assert!(prev_version > version);
                        if prev_version > start_ts {
                            return false;
                        }
                    }
                }
                true
            })
            .map_err(|e| Other(e.to_string()))?;
        for (i, &key) in matches.keys.iter().enumerate() {
            let handle = self.get_handle(key);
            let version = versions[key as usize];
            let mut value = vec![0f32; self.index.dimensions()];
            self.index
                .get(key, &mut value)
                .map_err(|e| Other(e.to_string()))?;
            let item = VectorItem {
                handle: handle.to_vec(),
                version,
                distance: matches.distances[i],
                value,
            };
            results.push(item);
        }
        Ok(results)
    }

    pub fn to_vector_index_file_pb(&self) -> kvenginepb::VectorIndexFile {
        let mut vec_idx_file_pb = kvenginepb::VectorIndexFile::new();
        vec_idx_file_pb.set_id(self.file_id());
        vec_idx_file_pb.set_snap_version(self.snap_version());
        vec_idx_file_pb.set_smallest(self.smallest().to_vec());
        vec_idx_file_pb.set_biggest(self.biggest().to_vec());
        vec_idx_file_pb
    }
}

#[derive(Debug)]
pub struct VectorItem {
    pub handle: Vec<u8>,
    pub version: u64,
    pub distance: f32,
    pub value: Vec<f32>,
}

impl VectorItem {
    pub fn get_int_handle(&self) -> i64 {
        self.handle.as_slice().get_i64_le()
    }
}

pub(crate) struct VectorItemsReader {
    schema: Schema,
    vector_col_idx: usize,
    items: Vec<VectorItem>,
    idx: usize,
    // The vector item already has handle, version and vector column.
    // If we need to read other columns, we can use the inner reader to read them.
    // The inner reader's schema doesn't contains the vector column.
    inner_reader: Option<Box<dyn ColumnarReader>>,
}

impl VectorItemsReader {
    pub(crate) fn new(
        schema: Schema,
        vector_index: &VectorIndex,
        target: &[f32],
        top_k: usize,
        read_ts: u64,
        start_handle: Option<&[u8]>,
        end_handle: Option<&[u8]>,
        col_levels: &ColumnarLevels,
        encryption_key: Option<EncryptionKey>,
    ) -> Result<Self> {
        let items = Self::search_items(
            &schema,
            vector_index,
            target,
            top_k,
            read_ts,
            start_handle,
            end_handle,
        )?;
        let vector_col_idx = schema
            .columns
            .iter()
            .position(|c| c.get_column_id() == vector_index.col_id)
            .unwrap();
        let inner_reader =
            Self::build_inner_reader(&schema, vector_index, col_levels, encryption_key);
        Ok(Self {
            schema,
            vector_col_idx,
            items,
            idx: 0,
            inner_reader,
        })
    }

    fn search_items(
        schema: &Schema,
        vector_index: &VectorIndex,
        target: &[f32],
        top_k: usize,
        read_ts: u64,
        start_handle: Option<&[u8]>,
        end_handle: Option<&[u8]>,
    ) -> Result<Vec<VectorItem>> {
        // TODO: The items read from vector index may be mvcc deleted. Valid items may
        // be less than top_k.
        let mut items = vector_index.search(target, top_k, read_ts)?;
        if schema.is_common_handle() {
            if let Some(start_handle) = start_handle {
                let end_handle = end_handle.unwrap();
                items.retain(|item| {
                    item.handle.as_slice() >= start_handle && item.handle.as_slice() < end_handle
                });
            }
            items.sort_by(|a, b| a.handle.cmp(&b.handle));
        } else {
            if let Some(mut start_handle) = start_handle {
                let start_int_handle = start_handle.get_i64_le();
                let end_int_handle = end_handle.map(|mut h| h.get_i64_le());
                items.retain(|item| {
                    let item_handle = item.handle.as_slice().get_i64_le();
                    item_handle >= start_int_handle
                        && end_int_handle.map_or(true, |end| item_handle < end)
                });
            }
            items.sort_by(|a, b| {
                a.handle
                    .as_slice()
                    .get_i64_le()
                    .cmp(&b.handle.as_slice().get_i64_le())
            })
        }
        Ok(items)
    }

    fn build_inner_reader(
        schema: &Schema,
        vector_index: &VectorIndex,
        col_levels: &ColumnarLevels,
        encryption_key: Option<EncryptionKey>,
    ) -> Option<Box<dyn ColumnarReader>> {
        if schema.columns.len() == 1 {
            assert_eq!(schema.columns[0].get_column_id(), vector_index.col_id);
            return None;
        }
        let mut readers: Vec<Box<dyn ColumnarReader>> = vec![];
        let inner_schema: Schema = schema
            .to_schema_buf()
            .retain_columns(|c| c.get_column_id() != vector_index.col_id)
            .into();
        for columnar_level in &col_levels.levels {
            if columnar_level.level == 2 {
                let concat_reader = ColumnarConcatReader::new(
                    &columnar_level.files,
                    inner_schema.clone(),
                    None,
                    encryption_key.clone(),
                );
                readers.push(Box::new(concat_reader));
            } else {
                for file in &columnar_level.files {
                    if file.get_l0_version().unwrap_or_default() > vector_index.snap_version() {
                        continue;
                    }
                    if !file.has_table(schema.table_id) {
                        continue;
                    }
                    let col_reader = ColumnarTableReader::new(
                        file,
                        inner_schema.clone(),
                        None,
                        encryption_key.clone(),
                    );
                    readers.push(Box::new(col_reader));
                }
            }
        }
        if readers.len() == 1 {
            Some(readers.pop().unwrap())
        } else {
            Some(Box::new(ColumnarMergeReader::new(
                inner_schema.clone(),
                readers,
            )))
        }
    }
}

#[async_trait]
impl ColumnarReader for VectorItemsReader {
    fn schema(&self) -> &Schema {
        &self.schema
    }

    async fn seek(&mut self, mut handle: &[u8]) -> Result<()> {
        self.idx = if get_fixed_size(&self.schema.handle_column) > 0 {
            let int_handle = handle.get_i64_le();
            search(self.items.len(), |i| {
                self.items[i].handle.as_slice().get_i64_le() >= int_handle
            })
        } else {
            search(self.items.len(), |i| {
                self.items[i].handle.as_slice() >= handle
            })
        };
        Ok(())
    }

    async fn read(&mut self, block: &mut Block, limit: usize) -> Result<usize> {
        let mut vector_col = block.columns.remove(self.vector_col_idx);
        let mut vec_val_buf = vec![];
        let old_idx = self.idx;
        for _ in 0..limit {
            if self.idx >= self.items.len() {
                break;
            }
            let item = &self.items[self.idx];
            let data = VectorFloat32Ref::from_f32(&item.value);
            vec_val_buf.truncate(0);
            vec_val_buf.write_vector_float32(data).unwrap();
            vector_col.push_value(&vec_val_buf);
            if let Some(inner) = &mut self.inner_reader {
                inner.seek(&item.handle).await?;
                inner.read(block, 1).await?;
            } else {
                block.handles.push_value(&item.handle);
                block.versions.push_version(item.version, false);
            }
            self.idx += 1;
        }
        block.columns.insert(self.vector_col_idx, vector_col);
        Ok(self.idx - old_idx)
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq)]
#[serde(default)]
#[serde(rename_all = "kebab-case")]
pub struct VectorIndexBuildOptions {
    // When the size of the non-indexed vector data exceed this value,
    // a new vector index file is build.
    pub delta_size: usize,
    // When the file count of a vector index exceed this value, a new vector index is rebuild to
    // replace the old vector index files.
    pub rebuild_file_count: usize,
}

impl Default for VectorIndexBuildOptions {
    fn default() -> Self {
        VectorIndexBuildOptions {
            delta_size: 16 * 1024 * 1024,
            rebuild_file_count: 4,
        }
    }
}

pub struct VectorIndexBuilder {
    index: usearch::Index,
    int_handles: Vec<i64>,
    common_handles: Vec<Vec<u8>>,
    versions: Vec<i64>,
    footer: VectorIndexFileFooter,
    is_common_handle: bool,
    metric_repr: i32,
    snap_version: u64,
    table_id: i64,
    index_id: i32,
    col_id: i32,
    num_rows: u64,
    smallest_int_handle: Option<i64>,
    biggest_int_handle: i64,
    smallest_common_handle: Vec<u8>,
    biggest_common_handle: Vec<u8>,
    pub(crate) smallest: Vec<u8>,
    pub(crate) biggest: Vec<u8>,
}

impl VectorIndexBuilder {
    pub fn new(
        dimension: usize,
        metric: &str,
        snap_version: u64,
        table_id: i64,
        index_id: i64,
        col_id: i64,
        is_common_handle: bool,
    ) -> Result<Self> {
        let mut opts = new_index_opts();
        opts.dimensions = dimension;
        opts.metric = match metric {
            "l2" => usearch::MetricKind::L2sq,
            "cosine" => usearch::MetricKind::Cos,
            _ => usearch::MetricKind::Cos,
        };
        let mut header = VectorIndexFileFooter::default();
        header.magic_number = MAGIC_NUMBER;
        let index = usearch::Index::new(&opts).map_err(|e| Other(e.to_string()))?;
        Ok(VectorIndexBuilder {
            index,
            int_handles: vec![],
            common_handles: vec![],
            versions: vec![],
            footer: header,
            snap_version,
            table_id,
            index_id: index_id as i32,
            col_id: col_id as i32,
            is_common_handle,
            metric_repr: opts.metric.repr,
            num_rows: 0,
            smallest_int_handle: None,
            biggest_int_handle: i64::MIN,
            smallest_common_handle: vec![],
            biggest_common_handle: vec![],
            smallest: vec![],
            biggest: vec![],
        })
    }

    pub fn add_block(&mut self, block: &Block, vec_col_off: usize) -> table::Result<()> {
        let vec_column = &block.get_columns()[vec_col_off];
        self.index
            .reserve(self.index.size() + vec_column.length())
            .unwrap();
        for i in 0..vec_column.length() {
            let vec_val = vec_column.get_value(i);
            if self.is_common_handle {
                let common_handle = block.handles.get_not_null_value(i);
                if i == 0 || i == vec_column.length() - 1 {
                    self.update_common_handle(common_handle);
                }
                if vec_val.is_some() {
                    self.common_handles.push(common_handle.to_vec());
                }
            } else {
                let handle = block.handles.get_int_handle_value(i);
                if i == 0 || i == vec_column.length() - 1 {
                    self.update_int_handle(handle);
                }
                if vec_val.is_some() {
                    self.int_handles.push(handle);
                }
            }
            if let Some(vec_val) = vec_val {
                let vec_f32: &[f32] = bytemuck::cast_slice(&vec_val[4..]);
                self.index
                    .add(self.num_rows, vec_f32)
                    .map_err(|e| Other(e.to_string()))?;
                let version = block.versions.get_version(i);
                self.versions.push(version as i64);
                self.num_rows += 1;
            }
        }
        Ok(())
    }

    fn update_int_handle(&mut self, handle: i64) {
        if self.smallest_int_handle.is_none() {
            self.smallest_int_handle = Some(handle);
        }
        self.biggest_int_handle = handle;
    }

    fn update_common_handle(&mut self, common_handle: &[u8]) {
        if self.smallest_common_handle.is_empty() {
            self.smallest_common_handle = common_handle.to_vec();
        }
        self.biggest_common_handle.truncate(0);
        self.biggest_common_handle.extend_from_slice(common_handle);
    }

    pub fn build(&mut self) -> Result<Vec<u8>> {
        let entry_count = self.num_rows as usize;
        let handles_size = entry_count * 8;
        let versions_size = entry_count * 8;
        let index_size = self.index.serialized_length();
        self.footer.index_size = index_size as u32;
        if self.is_common_handle {
            self.smallest
                .extend_from_slice(&encode_common_handle_row_key(
                    self.table_id,
                    self.smallest_common_handle.as_slice(),
                ));
            self.biggest
                .extend_from_slice(&encode_common_handle_row_key(
                    self.table_id,
                    self.biggest_common_handle.as_slice(),
                ));
        } else {
            self.smallest.extend_from_slice(&encode_row_key(
                self.table_id,
                self.smallest_int_handle.unwrap(),
            ));
            self.biggest
                .extend_from_slice(&encode_row_key(self.table_id, self.biggest_int_handle));
        }
        let mut props_buf = vec![];
        let mut write_prop = |key: &str, val: &[u8]| {
            props_buf.put_u16_le(key.len() as u16);
            props_buf.extend_from_slice(key.as_bytes());
            props_buf.put_u32_le(val.len() as u32);
            props_buf.extend_from_slice(val);
        };
        write_prop(PROP_SNAP_VERSION, &self.snap_version.to_le_bytes());
        write_prop(PROP_TABLE_ID, &self.table_id.to_le_bytes());
        write_prop(PROP_INDEX_ID, &self.index_id.to_le_bytes());
        write_prop(PROP_COLUMN_ID, &self.col_id.to_le_bytes());
        write_prop(PROP_IS_COMMON_HANDLE, &[self.is_common_handle as u8]);
        write_prop(PROP_METRIC_REPR, &self.metric_repr.to_le_bytes());
        write_prop(PROP_SMALLEST, &self.smallest);
        write_prop(PROP_BIGGEST, &self.biggest);
        self.footer.props_size = props_buf.len() as u32;
        let footer_data = self.footer.marshal();
        let mut buf = Vec::with_capacity(
            index_size + versions_size + handles_size + props_buf.len() + footer_data.len(),
        );
        buf.resize(index_size, 0);
        self.index
            .save_to_buffer(&mut buf)
            .map_err(|e| Other(e.to_string()))?;
        buf.resize((buf.len() + 7) & !7, 0); // align to 8 bytes
        buf.extend_from_slice(bytemuck::cast_slice(&self.versions));
        if self.is_common_handle {
            let mut handle_offsets = Vec::with_capacity(entry_count + 1);
            handle_offsets.push(0);
            let total_handle_size = self.common_handles.iter().map(|h| h.len()).sum::<usize>();
            let mut handle_data = Vec::with_capacity(total_handle_size);
            for i in 0..entry_count {
                handle_data.extend_from_slice(&self.common_handles[i]);
                handle_offsets.push(handle_data.len() as u32);
            }
            buf.extend_from_slice(bytemuck::cast_slice(&handle_offsets));
            buf.extend_from_slice(&handle_data);
        } else {
            buf.extend_from_slice(bytemuck::cast_slice(&self.int_handles));
        }
        buf.extend_from_slice(&props_buf);
        buf.extend_from_slice(&footer_data);
        Ok(buf)
    }
}

const PROP_SMALLEST: &str = "smallest";
const PROP_BIGGEST: &str = "biggest";
const PROP_SNAP_VERSION: &str = "snap_ver";
const PROP_TABLE_ID: &str = "tbl_id";
const PROP_INDEX_ID: &str = "idx_id";
const PROP_COLUMN_ID: &str = "col_id";
const PROP_IS_COMMON_HANDLE: &str = "c_h";
const PROP_METRIC_REPR: &str = "m_r";

fn new_index_opts() -> IndexOptions {
    let mut opts = IndexOptions::default();
    opts.connectivity = CONNECTIVITY;
    opts.expansion_add = EXPANSION_ADD;
    opts.expansion_search = EXPANSION_SEARCH;
    opts.quantization = QUANTIZATION;
    opts
}

#[cfg(test)]
mod tests {
    use std::{fs, sync::Arc};

    use bstr::ByteSlice;
    use schema::schema::StorageClass;
    use tidb_query_datatype::{
        codec::{
            data_type::VectorFloat32,
            mysql::VectorFloat32Encoder,
            table::{encode_common_handle_row_key, encode_row_key},
        },
        FieldTypeAccessor, FieldTypeTp,
    };
    use tipb::ColumnInfo;

    use crate::table::{
        columnar::{
            new_common_handle_column_info, new_int_handle_column_info, new_version_column_info,
            Block, Schema, SchemaBuf,
        },
        file::LocalFile,
        vector_index::{VectorIndex, VectorIndexBuilder, VectorIndexFile},
    };

    const TEST_DIMENSION: usize = 3;

    #[test]
    fn test_vector_index_file() {
        for common_handle in [true, false] {
            let vec_idx = build_vector_index_file(common_handle, 100, 200, 1);
            assert_eq!(vec_idx.index.size(), 100);
            assert_eq!(vec_idx.table_id, 1);
            assert_eq!(vec_idx.index_id, 1);
            assert_eq!(vec_idx.column_id, 1);
            assert_eq!(vec_idx.snap_version, 1);
            assert_eq!(vec_idx.is_common_handle, common_handle);
            if common_handle {
                assert_eq!(
                    vec_idx.smallest.as_bytes(),
                    encode_common_handle_row_key(1, &i_to_common_handle(100)).as_slice()
                );
                assert_eq!(
                    vec_idx.biggest.as_bytes(),
                    encode_common_handle_row_key(1, &i_to_common_handle(199)).as_slice()
                );
            } else {
                assert_eq!(
                    vec_idx.smallest.as_bytes(),
                    encode_row_key(1, 100).as_slice()
                );
                assert_eq!(
                    vec_idx.biggest.as_bytes(),
                    encode_row_key(1, 199).as_slice()
                );
            }
            for key in 0u64..100 {
                let mut vec_val = vec![0f32; TEST_DIMENSION];
                let cnt = vec_idx.index.get(key, &mut vec_val).unwrap();
                assert_eq!(cnt, 1);
                if common_handle {
                    assert_eq!(
                        vec_idx.get_handle(key),
                        i_to_common_handle(100 + key as i64)
                    );
                } else {
                    assert_eq!(vec_idx.get_handle(key), &(100 + key as i64).to_le_bytes());
                }
            }
            let items = vec_idx
                .search(&[50.0f32, 150.0f32, 250.0f32], 3, u64::MAX)
                .unwrap();
            assert_eq!(items.len(), 3);
            assert_eq!(items[0].value, vec![50.0f32, 150.0f32, 250.0f32]);
            assert_eq!(items[1].value, vec![51.0f32, 151.0f32, 251.0f32]);
            assert_eq!(items[2].value, vec![49.0f32, 149.0f32, 249.0f32]);
        }
    }

    fn i_to_common_handle(i: i64) -> Vec<u8> {
        format!("{:08x}", i).as_bytes().to_vec()
    }

    fn build_vector_index_file(
        common_handle: bool,
        start: i64,
        end: i64,
        snap_version: u64,
    ) -> VectorIndexFile {
        let mut builder =
            VectorIndexBuilder::new(3, "cosine", snap_version, 1, 1, 1, common_handle).unwrap();
        let handle_column = if common_handle {
            new_common_handle_column_info()
        } else {
            new_int_handle_column_info()
        };
        let version_column = new_version_column_info();
        let mut vec_col_info = ColumnInfo::new();
        vec_col_info.set_column_id(1);
        vec_col_info.set_tp(FieldTypeTp::TiDbVectorFloat32 as i32);
        vec_col_info.set_flen(TEST_DIMENSION as isize);
        let columns = vec![vec_col_info];
        let schema_buf = SchemaBuf::new(
            1,
            handle_column,
            version_column,
            columns,
            vec![],
            vec![],
            StorageClass::default(),
            None,
        );
        let schema = Schema::new(schema_buf);
        let mut block = Block::new(&schema);
        for i in start..end {
            if common_handle {
                block.handles.push_value(&i_to_common_handle(i));
            } else {
                block.handles.push_value(&i.to_le_bytes());
            }
            block.versions.push_value(&snap_version.to_le_bytes());
        }
        let vec_col_buf = &mut block.columns[0];
        for i in start..end {
            let mut vec_f32_val = vec![];
            for _ in 0..TEST_DIMENSION {
                vec_f32_val.push(i as f32);
            }
            let vec_f32 =
                VectorFloat32::copy_from_f32(&[(i - 100) as f32, i as f32, (i + 100) as f32]);
            let mut buf = vec![];
            buf.write_vector_float32(vec_f32.as_ref()).unwrap();
            vec_col_buf.push_value(&buf);
        }
        builder.add_block(&block, 0).unwrap();
        let data = builder.build().unwrap();
        let temp_path = tempfile::NamedTempFile::new().unwrap().into_temp_path();
        fs::write(&temp_path, data).unwrap();
        let local_file = LocalFile::open(1, &temp_path, false).unwrap();
        VectorIndexFile::new(Arc::new(local_file)).unwrap()
    }

    #[test]
    fn test_vector_index_deduplication() {
        // Create multiple vector index files with duplicate handles
        let mut vector_index = VectorIndex::new(1, 1, 1);
        vector_index
            .files
            .push(build_vector_index_file(false, 100, 200, 1));
        vector_index
            .files
            .push(build_vector_index_file(false, 150, 200, 2));
        vector_index
            .files
            .push(build_vector_index_file(false, 151, 200, 3));
        vector_index.sort();

        // Perform a search
        let query_vector = vec![50.0f32, 150.0f32, 250.0f32];
        let mut items = vector_index.search(&query_vector, 3, u64::MAX).unwrap();

        // each file returns 3， total is 9, remains 5 after deduplicate.
        assert_eq!(items.len(), 5);
        items.sort_by(|a, b| a.distance.total_cmp(&b.distance));
        assert_eq!(items[0].value, vec![50.0f32, 150.0f32, 250.0f32]);
        assert_eq!(items[0].version, 2);
        assert_eq!(items[1].value, vec![51.0f32, 151.0f32, 251.0f32]);
        assert_eq!(items[1].version, 3);
        assert_eq!(items[2].value, vec![49.0f32, 149.0f32, 249.0f32]);
        assert_eq!(items[2].version, 1);
    }
}
