// Copyright 2024 TiKV Project Authors. Licensed under Apache-2.0.

use std::{collections::HashMap, sync::Arc};

use bytes::{Buf, BufMut};
use protobuf::Message;
use tipb::TableInfo;

use crate::{
    table,
    table::{
        columnar::{
            builder::{new_txn_id_column_info, new_version_column_info},
            columnar::Schema,
        },
        sstable::{File, NO_COMPRESSION},
        ChecksumType,
    },
};

pub const SCHEMA_FILE_MAGIC: u32 = 0x5353484D;
pub const SCHEMA_FILE_FORMAT_VER: u16 = 1;

pub struct SchemaFile {
    core: Arc<SchemaFileCore>,
}

pub(crate) struct SchemaFileCore {
    file_id: u64,
    keyspace_id: u32,
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
}

impl SchemaFile {
    pub fn open(file: Arc<dyn File>) -> table::Result<Self> {
        let file_id = file.id();
        let file_data = file.read(0, file.size() as usize)?;
        let footer_size = std::mem::size_of::<SchemaFileFooter>();
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
        let mut tables = HashMap::new();
        while !data.is_empty() {
            let table_info_len = data.get_u32_le() as usize;
            let mut table_info = TableInfo::new();
            let table_pb = &data[..table_info_len];
            table_info.merge_from_bytes(table_pb).unwrap();
            data.advance(table_info_len);
            let table_id = table_info.get_table_id();
            let mut columns = table_info.take_columns().into_vec();
            let handle_column = columns.pop().unwrap();
            let schema = Schema {
                table_id,
                handle_column,
                version_column: new_version_column_info(),
                txn_id_column: Some(new_txn_id_column_info()),
                columns,
            };
            tables.insert(table_id, schema);
        }
        let core = SchemaFileCore {
            file_id,
            keyspace_id,
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

    pub fn get_file_id(&self) -> u64 {
        self.core.file_id
    }
}

pub fn build_schema_file(keyspace_id: u32, tables: Vec<Schema>) -> Vec<u8> {
    let mut data = Vec::new();
    data.put_u32_le(keyspace_id);
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
    use tidb_query_datatype::{Collation::Utf8Mb4GeneralCi, FieldTypeTp};

    use super::*;
    use crate::table::{
        columnar::builder::{new_common_handle_column_info, new_int_handle_column_info},
        sstable::InMemFile,
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
        let schema_1 = Schema {
            table_id: 1,
            handle_column: new_common_handle_column_info(),
            version_column: new_version_column_info(),
            txn_id_column: Some(new_txn_id_column_info()),
            columns: vec![new_column_info(3, true), new_column_info(4, false)],
        };
        let schema_2 = Schema {
            table_id: 2,
            handle_column: new_int_handle_column_info(),
            version_column: new_version_column_info(),
            txn_id_column: Some(new_txn_id_column_info()),
            columns: vec![new_column_info(3, false), new_column_info(4, true)],
        };
        let schemas = vec![schema_1, schema_2];
        let data = build_schema_file(keyspace_id, schemas.clone());
        let file = Arc::new(InMemFile::new(100, data.into()));
        let schema_file = SchemaFile::open(file).unwrap();
        assert_eq!(&schemas[0], schema_file.get_table(1).unwrap());
        assert_eq!(&schemas[1], schema_file.get_table(2).unwrap());
    }
}
