// Copyright 2024 TiKV Project Authors. Licensed under Apache-2.0.

use std::{collections::HashMap, ops::Deref, sync::Arc};

use api_version::api_v2::KEYSPACE_PREFIX_LEN;
use bytes::{Buf, BufMut};
use protobuf::Message;
use tidb_query_datatype::codec::table::{
    decode_table_id, INDEX_PREFIX_SEP, RECORD_PREFIX_SEP, TABLE_PREFIX, TABLE_PREFIX_KEY_LEN,
};
use tipb::TableInfo;

use crate::table::{
    self,
    columnar::{
        builder::{new_txn_id_column_info, new_version_column_info},
        columnar::Schema,
        get_primary_key, SchemaBuf,
    },
    file::File,
    ChecksumType, InnerKey, NO_COMPRESSION,
};

pub const SCHEMA_FILE_MAGIC: u32 = 0x5353484D;
pub const SCHEMA_FILE_FORMAT_VER: u16 = 1;

#[derive(Clone)]
pub struct SchemaFile {
    core: Arc<SchemaFileCore>,
}

pub(crate) struct SchemaFileCore {
    file_id: u64,
    keyspace_id: u32,
    version: i64,
    tables: HashMap<i64, Schema>,
}

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct SchemaFileFooter {
    pub checksum: u32,
    pub checksum_type: u8,
    pub compression_type: u8,
    pub format_version: u16,
    pub magic: u32,
}

impl SchemaFileFooter {
    pub(crate) fn new() -> Self {
        SchemaFileFooter {
            checksum: 0,
            checksum_type: ChecksumType::Crc32.value(),
            compression_type: NO_COMPRESSION,
            format_version: SCHEMA_FILE_FORMAT_VER,
            magic: SCHEMA_FILE_MAGIC,
        }
    }

    pub(crate) fn parse(mut buf: &[u8]) -> Self {
        SchemaFileFooter {
            checksum: buf.get_u32_le(),
            checksum_type: buf.get_u8(),
            compression_type: buf.get_u8(),
            format_version: buf.get_u16_le(),
            magic: buf.get_u32_le(),
        }
    }

    pub(crate) fn write_to(&self, data: &mut Vec<u8>) {
        data.put_u32_le(self.checksum);
        data.put_u8(self.checksum_type);
        data.put_u8(self.compression_type);
        data.put_u16_le(self.format_version);
        data.put_u32_le(self.magic);
    }

    pub const fn footer_size() -> usize {
        std::mem::size_of::<Self>()
    }
}

impl SchemaFile {
    pub fn open(file: Arc<dyn File>) -> table::Result<Self> {
        let file_id = file.id();
        let file_data = file.read(0, file.size() as usize)?;
        let footer_size = SchemaFileFooter::footer_size();
        if file_data.len() < footer_size {
            return Err(table::Error::InvalidFileSize);
        }
        let footer_offset = file_data.len() - footer_size;
        let footer = SchemaFileFooter::parse(&file_data[footer_offset..]);
        let mut data = &file_data[..footer_offset];
        if footer.magic != SCHEMA_FILE_MAGIC {
            return Err(table::Error::InvalidMagicNumber);
        }
        assert_eq!(footer.format_version, SCHEMA_FILE_FORMAT_VER);
        assert_eq!(footer.compression_type, NO_COMPRESSION);
        let checksum_type = ChecksumType::from(footer.checksum_type);
        let got_checksum = checksum_type.checksum(data);
        if got_checksum != footer.checksum {
            return Err(table::Error::InvalidChecksum(format!(
                "expect{}, got:{}",
                footer.checksum, got_checksum
            )));
        }
        let keyspace_id = data.get_u32_le();
        let schema_version = data.get_i64_le();
        let mut tables = HashMap::new();
        while !data.is_empty() {
            let table_info_len = data.get_u32_le() as usize;
            let mut table_info = TableInfo::new();
            let table_pb = &data[..table_info_len];
            table_info.merge_from_bytes(table_pb).unwrap();
            data.advance(table_info_len);
            let table_id = table_info.get_table_id();
            let mut columns = table_info.take_columns().into_vec();
            let pk_col_ids: Vec<i64> = columns
                .iter()
                .filter(|c| get_primary_key(c))
                .map(|c| c.get_column_id())
                .collect();
            let handle_column = columns.pop().unwrap();
            let schema_buf = SchemaBuf {
                table_id,
                handle_column,
                version_column: new_version_column_info(),
                txn_id_column: Some(new_txn_id_column_info()),
                columns,
                pk_col_ids,
            };
            tables.insert(table_id, Schema::new(schema_buf));
        }
        let core = SchemaFileCore {
            file_id,
            keyspace_id,
            version: schema_version,
            tables,
        };
        Ok(SchemaFile {
            core: Arc::new(core),
        })
    }

    pub fn get_table(&self, table_id: i64) -> Option<&Schema> {
        self.core.tables.get(&table_id)
    }

    pub fn get_keyspace_id(&self) -> u32 {
        self.core.keyspace_id
    }

    pub fn get_version(&self) -> i64 {
        self.core.version
    }

    pub fn get_file_id(&self) -> u64 {
        self.core.file_id
    }

    pub fn is_tombstone(&self) -> bool {
        self.core.file_id == 0
    }

    // overlap checks if the Shard contains any rows in any of the schema tables.
    pub fn overlap(&self, mut start_key: &[u8], mut end_key: &[u8], keyspace_id: u32) -> bool {
        if keyspace_id != self.get_keyspace_id() {
            return false;
        }
        if keyspace_id > 0 {
            start_key.advance(KEYSPACE_PREFIX_LEN);
            end_key.advance(KEYSPACE_PREFIX_LEN);
        }
        if !end_key.is_empty() && end_key < TABLE_PREFIX {
            // meta data region is not overlapped.
            return false;
        }
        let start_table_id = decode_table_id(start_key).unwrap_or(0);
        let mut end_table_id = decode_table_id(end_key).unwrap_or(i64::MAX);
        if end_key.len() >= TABLE_PREFIX_KEY_LEN
            && &end_key[TABLE_PREFIX_KEY_LEN..] < RECORD_PREFIX_SEP
        {
            // When end key is smaller than the table first record key, it doesn't overlap
            // the table.
            end_table_id -= 1;
        }
        if start_table_id == end_table_id && start_table_id > 0 && start_table_id < i64::MAX {
            // The shard only contains a single table's index is not overlapped.
            if start_key[TABLE_PREFIX_KEY_LEN..].starts_with(INDEX_PREFIX_SEP)
                && end_key[TABLE_PREFIX_KEY_LEN..].starts_with(INDEX_PREFIX_SEP)
            {
                return false;
            }
        }
        for &table_id in self.core.tables.keys() {
            if start_table_id <= table_id && table_id <= end_table_id {
                return true;
            }
        }
        false
    }

    pub fn overlap_tables(&self, smallest: InnerKey<'_>, biggest: InnerKey<'_>) -> Vec<i64> {
        let start_table_id = decode_table_id(smallest.deref()).unwrap_or(0);
        let end_table_id = decode_table_id(biggest.deref()).unwrap_or(i64::MAX);
        self.core
            .tables
            .keys()
            .filter(|&&tid| start_table_id <= tid && tid <= end_table_id)
            .copied()
            .collect()
    }

    pub fn contains(&self, others: &[Schema]) -> bool {
        for schema in others {
            if self.get_table(schema.table_id).is_none()
                || !self.get_table(schema.table_id).unwrap().eq(schema)
            {
                return false;
            }
        }
        true
    }

    pub fn has_overlap_ids(&self, others: &[i64]) -> bool {
        for tbl_id in others {
            if self.core.tables.contains_key(tbl_id) {
                return true;
            }
        }
        false
    }

    pub fn export_schemas(&self) -> HashMap<i64, Schema> {
        self.core.tables.clone()
    }

    pub fn schema_count(&self) -> usize {
        self.core.tables.len()
    }
}

pub fn build_schema_file(keyspace_id: u32, schema_version: i64, tables: Vec<Schema>) -> Vec<u8> {
    let mut data = Vec::new();
    data.put_u32_le(keyspace_id);
    data.put_i64_le(schema_version);
    for schema in &tables {
        let mut columns = Vec::with_capacity(schema.columns.len() + 1);
        columns.extend_from_slice(&schema.columns);
        columns.push(schema.handle_column.clone());
        let mut table_info = TableInfo::new();
        table_info.set_table_id(schema.table_id);
        table_info.set_columns(protobuf::RepeatedField::from_vec(columns));
        let table_info_data = table_info.write_to_bytes().unwrap();
        data.put_u32_le(table_info_data.len() as u32);
        data.extend_from_slice(&table_info_data);
    }
    let mut footer = SchemaFileFooter::new();
    let checksum_type = ChecksumType::Crc32;
    footer.checksum = checksum_type.checksum(&data);
    footer.write_to(&mut data);
    data
}

#[cfg(test)]
mod tests {
    use api_version::{api_v2::TIDB_META_KEY_PREFIX, ApiV2};
    use tidb_query_datatype::{
        codec::table::RECORD_PREFIX_SEP, Collation::Utf8Mb4GeneralCi, FieldTypeTp,
    };
    use tikv_util::codec::number::NumberEncoder;

    use super::*;
    use crate::table::{
        columnar::builder::{new_common_handle_column_info, new_int_handle_column_info},
        file::InMemFile,
    };

    fn new_column_info(id: i64, is_int: bool) -> tipb::ColumnInfo {
        let mut column_info = tipb::ColumnInfo::new();
        column_info.set_column_id(id);
        if is_int {
            column_info.set_tp(FieldTypeTp::LongLong as i32);
        } else {
            column_info.set_tp(FieldTypeTp::VarChar as i32);
            column_info.set_collation(Utf8Mb4GeneralCi as i32);
        }
        column_info
    }

    #[test]
    fn test_schema_file() {
        let keyspace_id = 1;
        let schema_version = 1234i64;
        let schema_1 = Schema::new(SchemaBuf {
            table_id: 10,
            handle_column: new_common_handle_column_info(),
            version_column: new_version_column_info(),
            txn_id_column: Some(new_txn_id_column_info()),
            columns: vec![new_column_info(3, true), new_column_info(4, false)],
            pk_col_ids: vec![],
        });
        let schema_2 = Schema::new(SchemaBuf {
            table_id: 20,
            handle_column: new_int_handle_column_info(),
            version_column: new_version_column_info(),
            txn_id_column: Some(new_txn_id_column_info()),
            columns: vec![new_column_info(3, false), new_column_info(4, true)],
            pk_col_ids: vec![],
        });
        let schemas = vec![schema_1, schema_2];
        let data = build_schema_file(keyspace_id, schema_version, schemas.clone());
        let file = Arc::new(InMemFile::new(100, data.into()));
        let schema_file = SchemaFile::open(file).unwrap();
        assert_eq!(&schemas[0], schema_file.get_table(10).unwrap());
        assert_eq!(&schemas[1], schema_file.get_table(20).unwrap());
        assert_eq!(schema_version, schema_file.get_version());

        for case in vec![
            OverlapCase {
                start_key: ApiV2::get_txn_keyspace_prefix(keyspace_id),
                end_key: encode_meta_key(keyspace_id, b"def"),
                overlap: false,
            },
            OverlapCase {
                start_key: ApiV2::get_txn_keyspace_prefix(keyspace_id),
                end_key: ApiV2::get_txn_keyspace_prefix(keyspace_id + 1),
                overlap: true,
            },
            OverlapCase {
                start_key: encode_table_key(keyspace_id, 20, false, b""),
                end_key: ApiV2::get_txn_keyspace_prefix(keyspace_id + 1),
                overlap: true,
            },
            OverlapCase {
                start_key: encode_table_key(keyspace_id, 21, true, b""),
                end_key: ApiV2::get_txn_keyspace_prefix(keyspace_id + 1),
                overlap: false,
            },
            OverlapCase {
                start_key: encode_meta_key(keyspace_id, b"abc"),
                end_key: encode_meta_key(keyspace_id, b"def"),
                overlap: false,
            },
            OverlapCase {
                start_key: encode_meta_key(keyspace_id, b"abc"),
                end_key: encode_table_key(keyspace_id, 13, true, b""),
                overlap: true,
            },
            OverlapCase {
                start_key: encode_table_key(keyspace_id, 1, true, b""),
                end_key: encode_table_key(keyspace_id, 10, false, b""),
                overlap: false,
            },
            OverlapCase {
                start_key: encode_table_key(keyspace_id, 1, true, b""),
                end_key: encode_table_key(keyspace_id, 10, true, b""),
                overlap: true,
            },
            OverlapCase {
                start_key: encode_table_key(keyspace_id, 10, true, b""),
                end_key: encode_table_key(keyspace_id, 13, false, b""),
                overlap: true,
            },
            OverlapCase {
                start_key: encode_table_key(keyspace_id, 10, false, b"012"),
                end_key: encode_table_key(keyspace_id, 10, false, b"123"),
                overlap: false,
            },
            OverlapCase {
                start_key: encode_table_key(keyspace_id, 10, false, b"012"),
                end_key: encode_table_key(keyspace_id, 10, true, b"123"),
                overlap: true,
            },
        ] {
            assert_eq!(
                schema_file.overlap(&case.start_key, &case.end_key, keyspace_id),
                case.overlap,
                "{:?}",
                case
            );
        }
    }

    #[test]
    fn test_contains_schema() {
        let keyspace_id = 1;
        let schema_version = 1234i64;
        let schema_1 = Schema::new(SchemaBuf {
            table_id: 10,
            handle_column: new_common_handle_column_info(),
            version_column: new_version_column_info(),
            txn_id_column: Some(new_txn_id_column_info()),
            columns: vec![new_column_info(3, true), new_column_info(4, false)],
            pk_col_ids: vec![],
        });
        let schema_2 = Schema::new(SchemaBuf {
            table_id: 20,
            handle_column: new_int_handle_column_info(),
            version_column: new_version_column_info(),
            txn_id_column: Some(new_txn_id_column_info()),
            columns: vec![new_column_info(3, false), new_column_info(4, true)],
            pk_col_ids: vec![],
        });
        let schema_3 = Schema::new(SchemaBuf {
            table_id: 30,
            handle_column: new_int_handle_column_info(),
            version_column: new_version_column_info(),
            txn_id_column: Some(new_txn_id_column_info()),
            columns: vec![new_column_info(3, false), new_column_info(4, true)],
            pk_col_ids: vec![],
        });
        let schemas = vec![schema_1.clone(), schema_2.clone()];
        let data = build_schema_file(keyspace_id, schema_version, schemas.clone());
        let file = Arc::new(InMemFile::new(100, data.into()));
        let schema_file = SchemaFile::open(file).unwrap();
        assert!(schema_file.contains(&schemas));
        let reverse_schemas = vec![schema_2.clone(), schema_1.clone()];
        assert!(schema_file.contains(&reverse_schemas));
        let single_schema = vec![schema_1.clone()];
        assert!(schema_file.contains(&single_schema));
        let more_schema = vec![schema_1, schema_2, schema_3];
        assert!(!schema_file.contains(&more_schema));
    }

    #[derive(Debug)]
    struct OverlapCase {
        start_key: Vec<u8>,
        end_key: Vec<u8>,
        overlap: bool,
    }

    fn encode_table_key(keyspace_id: u32, table_id: i64, is_row: bool, suffix: &[u8]) -> Vec<u8> {
        let mut key = Vec::with_capacity(KEYSPACE_PREFIX_LEN + TABLE_PREFIX_KEY_LEN);
        key.put(ApiV2::get_txn_keyspace_prefix(keyspace_id).as_slice());
        key.put(TABLE_PREFIX);
        key.encode_i64(table_id).unwrap();
        if is_row {
            key.put(RECORD_PREFIX_SEP);
        } else {
            key.put(INDEX_PREFIX_SEP);
        }
        key.put(suffix);
        key
    }

    fn encode_meta_key(keyspace_id: u32, suffix: &[u8]) -> Vec<u8> {
        let mut key = Vec::with_capacity(KEYSPACE_PREFIX_LEN + TABLE_PREFIX_KEY_LEN);
        key.put(ApiV2::get_txn_keyspace_prefix(keyspace_id).as_slice());
        key.push(TIDB_META_KEY_PREFIX);
        key.put(suffix);
        key
    }
}
