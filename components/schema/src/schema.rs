// Copyright 2023 TiKV Project Authors. Licensed under Apache-2.0.

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DbInfo {
    #[serde(rename = "id")]
    pub id: i64,
    #[serde(rename = "db_name")]
    pub name: CiStr,
    pub charset: String,
    pub collate: String,
    #[serde(skip)]
    pub tables: Vec<TableInfo>,
    pub state: SchemaState,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CiStr {
    #[serde(rename = "O")]
    pub o: String,
    #[serde(rename = "L")]
    pub l: String,
}

pub const STATE_PUBLIC: SchemaState = SchemaState(5);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TableInfo {
    pub id: i64,
    pub name: CiStr,
    pub charset: String,
    pub collate: String,
    pub cols: Option<Vec<ColumnInfo>>,
    pub index_info: Option<Vec<IndexInfo>>,
    pub state: SchemaState,
    pub pk_is_handle: bool,
    pub is_common_handle: bool,
    pub common_handle_version: u16,
    pub comment: String,
    pub partition: Option<PartitionInfo>,
    pub version: u16,
    pub is_columnar: bool,
}

// ColumnInfo provides meta data describing of a table column.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnInfo {
    pub id: i64,
    pub name: CiStr,
    pub offset: i64,
    pub origin_default: Option<String>,
    pub origin_default_bit: Option<Vec<u8>>,
    pub default: Option<String>,
    pub default_bit: Option<Vec<u8>>,
    // DefaultIsExpr is indicates the default value string is expr.
    pub default_is_expr: bool,
    pub generated_expr_string: String,
    pub generated_stored: bool,
    #[serde(rename = "type")]
    pub field_type: FieldType,
    pub state: SchemaState,
    pub comment: String,
    // A hidden column is used internally(expression index) and are not accessible by users.
    pub hidden: bool,
    // Version means the version of the column info.
    // Version = 0: For OriginDefaultValue and DefaultValue of timestamp column will stores the
    // default time in system time zone.              That is a bug if multiple TiDB servers in
    // different system time zone. Version = 1: For OriginDefaultValue and DefaultValue of
    // timestamp column will stores the default time in UTC time zone.              This will
    // fix bug in version 0. For compatibility with version 0, we add version field in column info
    // struct.
    pub version: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexInfo {
    pub id: i64,
    pub idx_name: CiStr,
    pub tbl_name: CiStr,
    pub idx_cols: Vec<IndexColumn>,
    pub state: SchemaState,
    pub is_unique: bool,
    pub is_primary: bool,
    pub is_invisible: bool,
    pub is_global: bool,
    pub mv_index: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexColumn {
    pub name: CiStr,
    pub offset: isize,
    // Length of prefix when using column prefix
    // for indexing;
    // UnspecifedLength if not using prefix indexing
    pub length: isize,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "PascalCase")]
pub struct FieldType {
    pub tp: i64,
    pub flag: u64,
    pub flen: i64,
    pub decimal: i64,
    pub charset: String,
    pub collate: String,
    pub elems: Option<Vec<String>>, // replace `()` with the appropriate type
    pub elems_is_binary_lit: Option<Vec<bool>>, // replace `()` with the appropriate type
    pub array: Option<bool>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SchemaState(u8);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartitionType(usize);

// PartitionInfo provides table partition info.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartitionInfo {
    #[serde(rename = "type")]
    pub type_: PartitionType,
    pub expr: String,
    pub columns: Vec<CiStr>,
    // User may already create table with partition but table partition is not
    // yet supported back then. When Enable is true, write/read need use tid
    // rather than pid.
    pub enable: bool,
    pub definitions: Vec<PartitionDefinition>,
    // AddingDefinitions is filled when adding a partition that is in the mid state.
    pub adding_definitions: Option<Vec<PartitionDefinition>>,
    // DroppingDefinitions is filled when dropping a partition that is in the mid state.
    pub dropping_definitions: Option<Vec<PartitionDefinition>>,
    pub states: Option<Vec<PartitionState>>,
    pub num: u64,
}

// PartitionState is the state of the partition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartitionState {
    pub id: i64,
    pub state: SchemaState,
}

// PartitionDefinition defines a single partition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartitionDefinition {
    pub id: i64,
    pub name: CiStr,
    pub less_than: Option<Vec<String>>,
    pub in_values: Option<Vec<Vec<String>>>,
    pub comment: Option<String>,
}
