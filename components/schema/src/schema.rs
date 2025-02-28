// Copyright 2023 TiKV Project Authors. Licensed under Apache-2.0.

use std::fmt;

use tidb_query_datatype::{
    codec::{
        datum,
        mysql::{Decimal, Duration, Time, TimeType},
        Datum,
    },
    expr::EvalContext,
    Collation, FieldTypeFlag, FieldTypeTp,
};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
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

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CiStr {
    #[serde(rename = "O")]
    pub o: String,
    #[serde(rename = "L")]
    pub l: String,
}

pub const STATE_PUBLIC: SchemaState = SchemaState(5);

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
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
    pub tiflash_replica: Option<TiFlashReplica>,
    // `None`: the storage class is never set.
    pub storage_class_tier: Option<String>,
}

impl TableInfo {
    pub fn with_columnar(&self) -> bool {
        !self.cols.as_ref().map(|c| c.is_empty()).unwrap_or(true) && self.build_columnar()
    }

    pub fn build_columnar(&self) -> bool {
        if self.comment.contains("columnar_engine") {
            return true;
        }
        if let Some(tiflash_replica) = &self.tiflash_replica {
            if tiflash_replica.count > 0 {
                return true;
            }
        }
        false
    }

    pub fn with_storage_class(&self) -> bool {
        self.storage_class() != StorageClass::Unspecified
    }

    pub fn storage_class(&self) -> StorageClass {
        let mut storage_class = self
            .storage_class_tier
            .as_deref()
            .and_then(|x| x.try_into().ok())
            .unwrap_or(StorageClass::Unspecified);
        if storage_class == StorageClass::Standard {
            // The standard storage class is not set to schema file and shard, need to be
            // changed to unspecified.
            storage_class = StorageClass::Unspecified;
        }
        storage_class
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TiFlashReplica {
    #[serde(rename = "Count")]
    pub count: u64,
    #[serde(rename = "LocationLabels")]
    pub location_labels: Option<Vec<String>>,
    #[serde(rename = "Available")]
    pub available: bool,
    #[serde(rename = "AvailablePartitionIDs")]
    pub available_partition_ids: Option<Vec<i64>>,
}

// ColumnInfo provides meta data describing of a table column.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ColumnInfo {
    pub id: i64,
    pub name: CiStr,
    pub offset: i64,
    pub origin_default: Option<String>,
    pub origin_default_bit: Option<String>, // base64 encoded string.
    pub default: Option<String>,
    pub default_bit: Option<String>, // base64 encoded string.
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
    pub vector_index: Option<VectorIndexInfo>,
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
    pub vector_index: Option<VectorIndexInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorIndexInfo {
    pub kind: String,
    pub dimension: u64,
    pub distance_metric: String,
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

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
#[serde(rename_all = "PascalCase")]
pub struct FieldType {
    pub tp: i32,
    pub flag: i32,
    pub flen: i64,
    pub decimal: i32,
    pub charset: String,
    pub collate: String,
    pub elems: Option<Vec<String>>, // replace `()` with the appropriate type
    pub elems_is_binary_lit: Option<Vec<bool>>, // replace `()` with the appropriate type
    pub array: Option<bool>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct SchemaState(u8);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartitionType(usize);

// PartitionInfo provides table partition info.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartitionInfo {
    #[serde(rename = "type")]
    pub type_: PartitionType,
    pub expr: String,
    pub columns: Option<Vec<CiStr>>,
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
    // `None`: the storage class is never set.
    pub storage_class_tier: Option<String>,
}

// SchemaDiff contains the schema modification at a particular schema version.
// It is used to reduce schema reload cost.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SchemaDiff {
    pub version: i64,
    #[serde(rename = "type")]
    pub action_type: ActionType,
    pub schema_id: i64,
    pub table_id: i64,
    pub old_table_id: i64,
    pub old_schema_id: i64,
    pub regenerate_schema_map: bool,
    pub affected_opts: Option<Vec<AffectedOption>>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ActionType(u8);

// AffectedOption is used when a ddl affects multi tables.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AffectedOption {
    pub schema_id: i64,
    pub table_id: i64,
    pub old_table_id: i64,
    pub old_schema_id: i64,
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub struct SchemaVersionResponse {
    pub version: i64,
    pub schemas: Vec<TableInfo>,
}

const STORAGE_CLASS_TIER_STANDARD: &str = "STANDARD";
const STORAGE_CLASS_TIER_IA: &str = "IA";

#[repr(u8)]
#[derive(PartialEq, Clone, Copy, Default, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum StorageClass {
    /// User doesn't specify the storage class.
    ///
    /// Most tables and partitions should be of this. Used to bypass storage
    /// class relevant logic for performance.
    #[default]
    Unspecified = 0,

    /// Standard Tier for general purpose.
    Standard = 1,

    /// Infrequent Access Tier for warm / cold data.
    Ia = 2,
}

impl StorageClass {
    pub fn display(&self) -> &'static str {
        match self {
            Self::Unspecified => "UNSPECIFIED",
            Self::Standard => "STANDARD",
            Self::Ia => "IA",
        }
    }

    pub fn marshal(&self) -> Vec<u8> {
        match self {
            Self::Unspecified => vec![],
            _ => vec![*self as u8],
        }
    }

    pub fn unmarshal(val: Option<&[u8]>) -> Self {
        match val {
            Some(val) if !val.is_empty() => {
                debug_assert_eq!(val.len(), 1, "invalid val: {:?}", val);
                StorageClass::try_from(val[0]).unwrap_or(StorageClass::Unspecified)
            }
            _ => StorageClass::Unspecified,
        }
    }

    /// User has specified the storage class.
    #[inline]
    pub fn is_specified(&self) -> bool {
        *self != Self::Unspecified
    }

    /// The storage class requires to occupy a region exclusively.
    #[inline]
    pub fn require_exclusive_region(&self) -> bool {
        matches!(self, Self::Ia)
    }
}

impl TryFrom<&str> for StorageClass {
    type Error = String;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value {
            "" => Ok(StorageClass::Unspecified),
            STORAGE_CLASS_TIER_STANDARD => Ok(StorageClass::Standard),
            STORAGE_CLASS_TIER_IA => Ok(StorageClass::Ia),
            _ => {
                debug_assert!(false, "Unknown storage class: {}", value);
                Err(format!("Unknown storage class: {}", value))
            }
        }
    }
}

impl TryFrom<u8> for StorageClass {
    type Error = String;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        // `0` is not expected to be passed in.
        match value {
            0 => Ok(StorageClass::Unspecified),
            1 => Ok(StorageClass::Standard),
            2 => Ok(StorageClass::Ia),
            _ => {
                debug_assert!(false, "Unknown storage class: {}", value);
                Err(format!("Unknown storage class: {}", value))
            }
        }
    }
}

impl fmt::Debug for StorageClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}({})", self.display(), *self as u8)
    }
}

impl fmt::Display for StorageClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.display())
    }
}

pub fn convert_column_infos_to_tipb(
    column_infos: &[ColumnInfo],
    pk_is_handle: bool,
) -> Vec<tipb::ColumnInfo> {
    let mut tipb_column_infos = Vec::with_capacity(column_infos.len());
    let mut ctx = EvalContext::default();
    for column_info in column_infos {
        let mut ci = tipb::ColumnInfo::new();
        ci.set_column_id(column_info.id);
        ci.set_tp(column_info.field_type.tp);
        ci.set_flag(column_info.field_type.flag);
        ci.set_column_len(column_info.field_type.flen.min(i32::MAX as i64) as i32);
        ci.set_decimal(column_info.field_type.decimal);
        ci.set_elems(
            column_info
                .field_type
                .elems
                .as_ref()
                .cloned()
                .unwrap_or_default()
                .into(),
        );
        ci.set_array(column_info.field_type.array.unwrap_or_default());
        let collation = Collation::from_name(column_info.field_type.collate.as_str()).unwrap();
        ci.set_collation(collation as i32);
        let is_pk_handle = column_info.field_type.flag as u32 & FieldTypeFlag::PRIMARY_KEY.bits()
            != 0
            && pk_is_handle;
        ci.set_pk_handle(is_pk_handle);
        let default_val = decode_default_value_to_datum(&mut ctx, column_info)
            .map(|v| datum::encode_value(&mut EvalContext::default(), &[v]).unwrap())
            .unwrap_or_default();
        ci.set_default_val(default_val);
        tipb_column_infos.push(ci);
    }
    tipb_column_infos
}

// Ref: SetPBColumnsDefaultValue in TiDB.
// https://github.com/pingcap/tidb/blob/45318da24d8e4c0c6aab836d291a33f949dd18bf/pkg/table/tables/tables.go#L2303-L2329
// If origin_default is none, return None.
fn decode_default_value_to_datum(ctx: &mut EvalContext, c: &ColumnInfo) -> Option<Datum> {
    if !c.generated_expr_string.is_empty() && !c.generated_stored {
        return Some(Datum::Null);
    }

    // return None if origin_default is none.
    c.origin_default.as_ref()?;

    let field_type_tp = FieldTypeTp::from_i32(c.field_type.tp).unwrap();
    let field_flag = c.field_type.flag;

    let default = c.origin_default.as_ref().unwrap();
    let result = match field_type_tp {
        FieldTypeTp::Tiny
        | FieldTypeTp::Short
        | FieldTypeTp::Int24
        | FieldTypeTp::Long
        | FieldTypeTp::LongLong => {
            if field_flag as u32 & FieldTypeFlag::UNSIGNED.bits() == FieldTypeFlag::UNSIGNED.bits()
            {
                default.parse::<u64>().map(|v| Datum::U64(v)).map_err(|e| {
                    tidb_query_datatype::codec::Error::InvalidDataType(format!(
                        "Invalid unsigned integer: {:?}",
                        e
                    ))
                })
            } else {
                default.parse::<i64>().map(|v| Datum::I64(v)).map_err(|e| {
                    tidb_query_datatype::codec::Error::InvalidDataType(format!(
                        "Invalid signed integer: {:?}",
                        e
                    ))
                })
            }
        }
        FieldTypeTp::Year => default.parse::<u64>().map(|v| Datum::U64(v)).map_err(|e| {
            tidb_query_datatype::codec::Error::InvalidDataType(format!(
                "Invalid signed integer: {:?}",
                e
            ))
        }),

        FieldTypeTp::Float | FieldTypeTp::Double => {
            default.parse::<f64>().map(|v| Datum::F64(v)).map_err(|e| {
                tidb_query_datatype::codec::Error::InvalidDataType(format!(
                    "Invalid float number: {:?}",
                    e
                ))
            })
        }
        FieldTypeTp::Date | FieldTypeTp::DateTime | FieldTypeTp::Timestamp => {
            // TODO: handle timezone
            let x = default.to_lowercase();
            if x == "current_timestamp" || x == "current_data" {
                Time::parse_datetime(
                    ctx,
                    chrono::Utc::now()
                        .format("%Y-%m-%d %H:%M:%S")
                        .to_string()
                        .as_str(),
                    c.field_type.decimal as i8,
                    false,
                )
                .map(|t| Datum::Time(t))
                .map_err(|e| {
                    tidb_query_datatype::codec::Error::InvalidDataType(format!(
                        "Invalid datetime: {:?}",
                        e
                    ))
                })
            } else if x == "0000-00-00 00:00:00" {
                Time::parse_from_i64(
                    ctx,
                    0,
                    TimeType::try_from(field_type_tp).unwrap(),
                    c.field_type.decimal as i8,
                )
                .map(|t| Datum::Time(t))
                .map_err(|e| {
                    tidb_query_datatype::codec::Error::InvalidDataType(format!(
                        "Invalid datetime: {:?}",
                        e
                    ))
                })
            } else {
                Time::parse(
                    ctx,
                    default.as_str(),
                    TimeType::try_from(field_type_tp).unwrap(),
                    c.field_type.decimal as i8,
                    false,
                )
                .map(|t| Datum::Time(t))
                .map_err(|e| {
                    tidb_query_datatype::codec::Error::InvalidDataType(format!(
                        "Invalid datetime: {:?}",
                        e
                    ))
                })
            }
        }

        FieldTypeTp::NewDecimal => Decimal::from_bytes(default.as_bytes())
            .map(|v| Datum::Dec(v.unwrap()))
            .map_err(|_| {
                tidb_query_datatype::codec::Error::InvalidDataType("Invalid decimal".to_string())
            }),
        FieldTypeTp::String
        | FieldTypeTp::VarString
        | FieldTypeTp::VarChar
        | FieldTypeTp::TinyBlob => Ok(Datum::Bytes(default.as_bytes().to_vec())),
        FieldTypeTp::Duration => {
            parse_duration(default.as_str(), c.field_type.decimal as i8).map(|v| Datum::Dur(v))
        }
        _ => Ok(Datum::Bytes(vec![])),
    };
    if let Ok(datum) = result {
        Some(datum)
    } else {
        Some(Datum::Bytes(vec![]))
    }
}

fn parse_duration(input: &str, fsp: i8) -> tidb_query_datatype::codec::Result<Duration> {
    let parts: Vec<&str> = input.split(':').collect();
    if parts.len() != 3 {
        return Err(tidb_query_datatype::codec::Error::InvalidDataType(
            "Invalid input: must be in the format HH:MM:SS".to_string(),
        ));
    }

    let hours = parts[0].parse::<i64>().map_err(|_| {
        tidb_query_datatype::codec::Error::InvalidDataType("Invalid hours".to_string())
    })?;
    let minutes = parts[1].parse::<u64>().map_err(|_| {
        tidb_query_datatype::codec::Error::InvalidDataType("Invalid minutes".to_string())
    })?;
    let seconds = parts[2].parse::<u64>().map_err(|_| {
        tidb_query_datatype::codec::Error::InvalidDataType("Invalid seconds".to_string())
    })?;

    if minutes >= 60 || seconds >= 60 {
        return Err(tidb_query_datatype::codec::Error::InvalidDataType(
            "Minutes and seconds must be less than 60".to_string(),
        ));
    }

    let secs = if hours < 0 {
        -hours * 3600 - minutes as i64 * 60 - seconds as i64
    } else {
        hours * 3600 + minutes as i64 * 60 + seconds as i64
    };

    Duration::from_secs(secs, fsp)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tiflash_replica() {
        let str =
            r#"{"Count":1,"LocationLabels":["host"],"Available":true,"AvailablePartitionIDs":[1]}"#;
        let replica: TiFlashReplica = serde_json::from_str(str).unwrap();
        assert_eq!(replica.count, 1);
        assert_eq!(replica.location_labels, Some(vec!["host".to_string()]));
        assert_eq!(replica.available, true);
        assert_eq!(replica.available_partition_ids, Some(vec![1]));

        let str =
            r#"{"Count":2,"LocationLabels":null,"Available":true,"AvailablePartitionIDs":null}"#;
        let replica: TiFlashReplica = serde_json::from_str(str).unwrap();
        assert_eq!(replica.count, 2);
        assert_eq!(replica.location_labels, None);
        assert_eq!(replica.available, true);
        assert_eq!(replica.available_partition_ids, None);
    }

    #[test]
    fn test_storage_class() {
        let cases = vec![
            (StorageClass::Unspecified, None, vec![]),
            (StorageClass::Unspecified, Some(vec![]), vec![]),
            (StorageClass::Standard, Some(vec![1]), vec![1]),
            (StorageClass::Ia, Some(vec![2]), vec![2]),
        ];
        for (sc, input, output) in cases {
            assert_eq!(StorageClass::unmarshal(input.as_deref()), sc);
            assert_eq!(sc.marshal(), output);
        }
    }
}
