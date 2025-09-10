// Copyright 2024 TiKV Project Authors. Licensed under Apache-2.0.

use std::ops::{BitAnd, BitOr, Not};

use codec::number::NumberDecoder;
use tidb_query_datatype::{FieldTypeAccessor, FieldTypeTp};
use tipb::ColumnInfo;

use super::TableMeta;
use crate::table::columnar::is_unsigned;

#[derive(Debug, Default, Copy, Clone)]
pub enum FilterType {
    #[default]
    Unsupported = 0,

    // logical
    Not = 1,
    Or,
    And,
    // compare
    Equal,
    NotEqual,
    Greater,
    GreaterEqual,
    Less,
    LessEqual,

    In,
    NotIn,
    Like,
    NotLike,
    IsNull,
}

impl FilterType {
    fn new_from_sig(sig: tipb::ScalarFuncSig) -> Self {
        match sig {
            tipb::ScalarFuncSig::LtInt => FilterType::Less,
            tipb::ScalarFuncSig::LtReal => FilterType::Less,
            tipb::ScalarFuncSig::LtString => FilterType::Less,
            tipb::ScalarFuncSig::LtDecimal => FilterType::Less,
            tipb::ScalarFuncSig::LtTime => FilterType::Less,
            tipb::ScalarFuncSig::LtDuration => FilterType::Less,
            tipb::ScalarFuncSig::LtJson => FilterType::Less,

            tipb::ScalarFuncSig::LeInt => FilterType::LessEqual,
            tipb::ScalarFuncSig::LeReal => FilterType::LessEqual,
            tipb::ScalarFuncSig::LeString => FilterType::LessEqual,
            tipb::ScalarFuncSig::LeDecimal => FilterType::LessEqual,
            tipb::ScalarFuncSig::LeTime => FilterType::LessEqual,
            tipb::ScalarFuncSig::LeDuration => FilterType::LessEqual,
            tipb::ScalarFuncSig::LeJson => FilterType::LessEqual,

            tipb::ScalarFuncSig::GtInt => FilterType::Greater,
            tipb::ScalarFuncSig::GtReal => FilterType::Greater,
            tipb::ScalarFuncSig::GtString => FilterType::Greater,
            tipb::ScalarFuncSig::GtDecimal => FilterType::Greater,
            tipb::ScalarFuncSig::GtTime => FilterType::Greater,
            tipb::ScalarFuncSig::GtDuration => FilterType::Greater,
            tipb::ScalarFuncSig::GtJson => FilterType::Greater,

            tipb::ScalarFuncSig::GeInt => FilterType::GreaterEqual,
            tipb::ScalarFuncSig::GeReal => FilterType::GreaterEqual,
            tipb::ScalarFuncSig::GeString => FilterType::GreaterEqual,
            tipb::ScalarFuncSig::GeDecimal => FilterType::GreaterEqual,
            tipb::ScalarFuncSig::GeTime => FilterType::GreaterEqual,
            tipb::ScalarFuncSig::GeDuration => FilterType::GreaterEqual,
            tipb::ScalarFuncSig::GeJson => FilterType::GreaterEqual,

            tipb::ScalarFuncSig::EqInt => FilterType::Equal,
            tipb::ScalarFuncSig::EqReal => FilterType::Equal,
            tipb::ScalarFuncSig::EqString => FilterType::Equal,
            tipb::ScalarFuncSig::EqDecimal => FilterType::Equal,
            tipb::ScalarFuncSig::EqTime => FilterType::Equal,
            tipb::ScalarFuncSig::EqDuration => FilterType::Equal,
            tipb::ScalarFuncSig::EqJson => FilterType::Equal,

            tipb::ScalarFuncSig::NeInt => FilterType::NotEqual,
            tipb::ScalarFuncSig::NeReal => FilterType::NotEqual,
            tipb::ScalarFuncSig::NeString => FilterType::NotEqual,
            tipb::ScalarFuncSig::NeDecimal => FilterType::NotEqual,
            tipb::ScalarFuncSig::NeTime => FilterType::NotEqual,
            tipb::ScalarFuncSig::NeDuration => FilterType::NotEqual,
            tipb::ScalarFuncSig::NeJson => FilterType::NotEqual,

            tipb::ScalarFuncSig::LogicalAnd => FilterType::And,
            tipb::ScalarFuncSig::LogicalOr => FilterType::Or,

            tipb::ScalarFuncSig::UnaryNotDecimal => FilterType::Not,
            tipb::ScalarFuncSig::UnaryNotInt => FilterType::Not,
            tipb::ScalarFuncSig::UnaryNotReal => FilterType::Not,

            tipb::ScalarFuncSig::DecimalIsNull => FilterType::IsNull,
            tipb::ScalarFuncSig::DurationIsNull => FilterType::IsNull,
            tipb::ScalarFuncSig::RealIsNull => FilterType::IsNull,
            tipb::ScalarFuncSig::StringIsNull => FilterType::IsNull,
            tipb::ScalarFuncSig::TimeIsNull => FilterType::IsNull,
            tipb::ScalarFuncSig::IntIsNull => FilterType::IsNull,
            tipb::ScalarFuncSig::JsonIsNull => FilterType::IsNull,

            tipb::ScalarFuncSig::InInt => FilterType::In,
            tipb::ScalarFuncSig::InReal => FilterType::In,
            tipb::ScalarFuncSig::InString => FilterType::In,
            tipb::ScalarFuncSig::InDecimal => FilterType::In,
            tipb::ScalarFuncSig::InTime => FilterType::In,
            tipb::ScalarFuncSig::InDuration => FilterType::In,

            _ => FilterType::Unsupported,
        }
    }
}

// This is a pre-check for filter support type.
fn is_filter_support_type(field_type: FieldTypeTp) -> bool {
    match field_type {
        FieldTypeTp::Tiny
        | FieldTypeTp::Short
        | FieldTypeTp::Long
        | FieldTypeTp::LongLong
        | FieldTypeTp::Int24
        | FieldTypeTp::Year => true,
        // For these date-like types, they store UTC time and ignore time_zone
        FieldTypeTp::NewDate
        | FieldTypeTp::Date
        | FieldTypeTp::Duration
        | FieldTypeTp::DateTime
        | FieldTypeTp::Timestamp => true,
        // For these types, should take collation into consideration. Disable them.
        FieldTypeTp::VarChar
        | FieldTypeTp::Json
        | FieldTypeTp::TinyBlob
        | FieldTypeTp::MediumBlob
        | FieldTypeTp::LongBlob
        | FieldTypeTp::Blob
        | FieldTypeTp::VarString
        | FieldTypeTp::String => false,
        // Unknown.
        FieldTypeTp::TiDbVectorFloat32
        | FieldTypeTp::Unspecified
        | FieldTypeTp::NewDecimal
        | FieldTypeTp::Float
        | FieldTypeTp::Double
        | FieldTypeTp::Null
        | FieldTypeTp::Bit
        | FieldTypeTp::Enum
        | FieldTypeTp::Set
        | FieldTypeTp::Geometry => false,
    }
}

#[derive(Debug, Default, Clone)]
struct ParseCtx {
    columns: Vec<ColumnInfo>,
    // TODO: timezone info
}

#[derive(PartialEq)]
enum OperandType {
    Unknown = 0,
    Column,
    Literal,
}

fn parse_compare_expr(
    expr: &tipb::Expr,
    filter_type: FilterType,
    ctx: &ParseCtx,
) -> FilterOperator {
    if expr.get_children().len() != 2 && !matches!(filter_type, FilterType::In | FilterType::NotIn)
    {
        return FilterOperator::default();
    }
    let mut left_type = OperandType::Unknown;
    let mut right_type = OperandType::Unknown;
    // Only support `column` `op` `literal`, and `column` in (literal1, literal2,
    // ...) now.
    let is_timestamp_column = expr.get_children().iter().any(|child| {
        is_column_expr(child) && child.get_field_type().tp() == FieldTypeTp::Timestamp
    });
    let mut col_id = None;
    let mut values = vec![];
    let children_size = expr.get_children().len();
    for (i, child) in expr.get_children().iter().enumerate() {
        if is_column_expr(child) {
            let tp = child.get_field_type().tp();
            if !is_filter_support_type(tp) {
                return FilterOperator::default();
            }
            // Only support `column` in/not in (literal1, literal2, ...) now.
            if i != 0 && children_size != 2 {
                return FilterOperator::default();
            }
            col_id = Some(get_column_id(child, ctx));
            if i == 0 {
                left_type = OperandType::Column;
            } else if i == 1 {
                right_type = OperandType::Column;
            }
        } else if is_not_null_literal_expr(child) {
            if let Some(val) = decode_literal_value(child) {
                if i == 0 {
                    left_type = OperandType::Literal;
                } else if i == 1 {
                    right_type = OperandType::Literal;
                }
                if is_timestamp_column {
                    let literal_type = child.get_field_type().tp();
                    if literal_type != FieldTypeTp::Timestamp
                        && literal_type != FieldTypeTp::DateTime
                    {
                        return FilterOperator::default();
                    }
                    // TODO: convert with timezone
                }
                values.push(val);
            }
        } else {
            return FilterOperator::default();
        }
    }
    if col_id.is_none() {
        return FilterOperator::default();
    }

    let is_normal_cmp = left_type == OperandType::Column && right_type == OperandType::Literal;
    let is_inverse_cmp = left_type == OperandType::Literal && right_type == OperandType::Column;
    if !is_normal_cmp && !is_inverse_cmp {
        return FilterOperator::default();
    }

    // Adjust the filter type by the direction of operands
    let mut filter_type_with_direction = filter_type;
    if is_inverse_cmp {
        match filter_type {
            FilterType::Greater => {
                filter_type_with_direction = FilterType::Less;
            }
            FilterType::GreaterEqual => {
                filter_type_with_direction = FilterType::LessEqual;
            }
            FilterType::Less => {
                filter_type_with_direction = FilterType::Greater;
            }
            FilterType::LessEqual => {
                filter_type_with_direction = FilterType::GreaterEqual;
            }
            _ => {}
        }
    }
    let compare = CompareOperator::new(col_id.unwrap(), values);
    FilterOperator::new_compare(filter_type_with_direction, compare)
}

fn parse_ti_expr(expr: &tipb::Expr, ctx: &ParseCtx) -> FilterOperator {
    assert!(is_function_expr(expr));
    let op_unsupported = FilterOperator::default();
    if is_agg_function_expr(expr) {
        return op_unsupported;
    }
    let filter_type = FilterType::new_from_sig(expr.get_sig());
    match filter_type {
        // Not/And/Or only support when the child is FunctionExpr, thus expr like `a and null` can
        // not do filter here. If later we want to support Not/And/Or do filter not only
        // FunctionExpr but also ColumnExpr and Literal, we must take a special
        // consideration about null value, due to we just ignore the null value in other filter(such
        // as equal, greater, etc.) Therefore, null value may bring some extra correctness
        // problem if we expand Not/And/Or filtering areas.
        FilterType::Not => {
            if expr.get_children().len() != 1 {
                return op_unsupported;
            }
            let child = &expr.get_children()[0];
            if !is_function_expr(child) {
                return op_unsupported;
            }
            FilterOperator::new(FilterType::Not, vec![parse_ti_expr(child, ctx)])
        }
        FilterType::And | FilterType::Or => {
            FilterOperator::new(filter_type, parse_children(expr, ctx))
        }
        FilterType::Equal
        | FilterType::NotEqual
        | FilterType::Greater
        | FilterType::GreaterEqual
        | FilterType::Less
        | FilterType::LessEqual
        | FilterType::In
        | FilterType::NotIn => parse_compare_expr(expr, filter_type, ctx),
        FilterType::IsNull => {
            // for IsNULL filter, we only support do filter when the child is ColumnExpr.
            // That is, we only do filter when the statement likes `where a is null`, but
            // not `where (a > 1) is null` because in other filter
            // calculation(like Equal/Less/LessEqual/Greater/GreateEqual/NotEqual), we just
            // make filter ignoring the null value. Therefore, if we support
            // IsNull with sub expr, there could be correctness problem.
            // For example, we have a table t(a int, b int), and the data is: (1, 2), (0,
            // null), (null, 1) and then we execute `select * from t where (a >
            // 1) is null`, we want to get (null, 1) but in RSResult (a > 1), we
            // will get the result RSResult::None, and then we think the result is the empty
            // set.
            if expr.get_children().len() != 1 {
                return op_unsupported;
            }
            let child = &expr.get_children()[0];
            if !is_column_expr(child) {
                return op_unsupported;
            }
            if !is_filter_support_type(child.get_field_type().tp()) {
                return op_unsupported;
            }
            let compare = CompareOperator::new(get_column_id(child, ctx), vec![]);
            FilterOperator::new_compare(FilterType::IsNull, compare)
        }
        FilterType::Like | FilterType::NotLike | FilterType::Unsupported => op_unsupported,
    }
}

fn parse_children(expr: &tipb::Expr, ctx: &ParseCtx) -> Vec<FilterOperator> {
    expr.get_children()
        .iter()
        .map(|child| {
            if is_function_expr(child) {
                parse_ti_expr(child, ctx)
            } else {
                FilterOperator::default()
            }
        })
        .collect()
}

fn is_column_expr(expr: &tipb::Expr) -> bool {
    matches!(expr.get_tp(), tipb::ExprType::ColumnRef)
}

fn is_function_expr(expr: &tipb::Expr) -> bool {
    is_scalar_function_expr(expr) || is_agg_function_expr(expr) || is_window_function_expr(expr)
}

fn is_scalar_function_expr(expr: &tipb::Expr) -> bool {
    matches!(expr.get_tp(), tipb::ExprType::ScalarFunc)
}

fn is_agg_function_expr(expr: &tipb::Expr) -> bool {
    matches!(
        expr.get_tp(),
        tipb::ExprType::Count
            | tipb::ExprType::Sum
            | tipb::ExprType::Avg
            | tipb::ExprType::Min
            | tipb::ExprType::Max
            | tipb::ExprType::First
            | tipb::ExprType::GroupConcat
            | tipb::ExprType::AggBitAnd
            | tipb::ExprType::AggBitOr
            | tipb::ExprType::AggBitXor
            | tipb::ExprType::Std
            | tipb::ExprType::Stddev
            | tipb::ExprType::StddevPop
            | tipb::ExprType::StddevSamp
            | tipb::ExprType::VarPop
            | tipb::ExprType::VarSamp
            | tipb::ExprType::Variance
            | tipb::ExprType::JsonArrayAgg
            | tipb::ExprType::JsonObjectAgg
            | tipb::ExprType::ApproxCountDistinct
    )
}

fn is_window_function_expr(expr: &tipb::Expr) -> bool {
    matches!(
        expr.get_tp(),
        tipb::ExprType::RowNumber
            | tipb::ExprType::Rank
            | tipb::ExprType::DenseRank
            | tipb::ExprType::Lead
            | tipb::ExprType::Lag
            | tipb::ExprType::FirstValue
            | tipb::ExprType::LastValue
    )
}

fn is_not_null_literal_expr(expr: &tipb::Expr) -> bool {
    matches!(
        expr.get_tp(),
        tipb::ExprType::Int64
            | tipb::ExprType::Uint64
            | tipb::ExprType::Float32
            | tipb::ExprType::Float64
            | tipb::ExprType::String
            | tipb::ExprType::Bytes
            | tipb::ExprType::MysqlBit
            | tipb::ExprType::MysqlDecimal
            | tipb::ExprType::MysqlDuration
            | tipb::ExprType::MysqlEnum
            | tipb::ExprType::MysqlHex
            | tipb::ExprType::MysqlSet
            | tipb::ExprType::MysqlTime
            | tipb::ExprType::MysqlJson
            | tipb::ExprType::ValueList
            | tipb::ExprType::TiDbVectorFloat32
    )
}

fn get_column_id(expr: &tipb::Expr, ctx: &ParseCtx) -> i64 {
    let col_offset = expr.get_val().read_i64().unwrap() as usize;
    ctx.columns[col_offset].get_column_id()
}

fn decode_literal_value(expr: &tipb::Expr) -> Option<Vec<u8>> {
    let mut val = expr.get_val();
    let res = match expr.get_tp() {
        tipb::ExprType::Int64 => val.read_i64().unwrap().to_le_bytes().to_vec(),
        tipb::ExprType::Uint64 => val.read_u64().unwrap().to_le_bytes().to_vec(),
        tipb::ExprType::Float32 => (val.read_f64().unwrap()).to_le_bytes().to_vec(),
        tipb::ExprType::Float64 => val.read_f64().unwrap().to_le_bytes().to_vec(),
        tipb::ExprType::String | tipb::ExprType::Bytes => val.to_vec(),
        tipb::ExprType::MysqlTime => val.read_u64().unwrap().to_le_bytes().to_vec(),
        tipb::ExprType::MysqlDuration => val.read_i64().unwrap().to_le_bytes().to_vec(),
        _ => return None,
    };
    Some(res)
}

#[derive(Debug, Default, Clone)]
pub struct FilterOperator {
    op: FilterType,
    children: Vec<FilterOperator>,
    compare: Option<CompareOperator>,
}

#[derive(Debug, Default, Clone)]
pub struct CompareOperator {
    col_id: i64,
    values: Vec<Vec<u8>>,
}

impl CompareOperator {
    fn new(col_id: i64, values: Vec<Vec<u8>>) -> Self {
        CompareOperator { col_id, values }
    }
}

#[derive(PartialEq, Debug)]
pub enum FilterOpResult {
    Unknown,
    Some,
    None,
}

impl BitAnd for FilterOpResult {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self::Output {
        if self == FilterOpResult::None || rhs == FilterOpResult::None {
            FilterOpResult::None
        } else {
            FilterOpResult::Some
        }
    }
}

impl BitOr for FilterOpResult {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        if self == FilterOpResult::None && rhs == FilterOpResult::None {
            FilterOpResult::None
        } else {
            FilterOpResult::Some
        }
    }
}

impl Not for FilterOpResult {
    type Output = Self;

    fn not(self) -> Self::Output {
        match self {
            FilterOpResult::None => FilterOpResult::Some,
            FilterOpResult::Some => FilterOpResult::None,
            FilterOpResult::Unknown => FilterOpResult::Unknown,
        }
    }
}

pub type FilterOpResults = Vec<FilterOpResult>;

impl FilterOperator {
    pub fn new(op: FilterType, children: Vec<FilterOperator>) -> Self {
        FilterOperator {
            op,
            children,
            compare: None,
        }
    }

    pub fn new_compare(op: FilterType, compare: CompareOperator) -> Self {
        FilterOperator {
            op,
            children: vec![],
            compare: Some(compare),
        }
    }

    pub(crate) fn rough_check(&self, table_meta: &TableMeta) -> FilterOpResults {
        let pack_num = table_meta.handle_column.pack_offsets.num_packs();
        let mut res = Vec::with_capacity(pack_num);
        for pack_idx in 0..pack_num {
            res.push(self.rough_check_pack(pack_idx, table_meta));
        }
        res
    }

    pub(crate) fn rough_check_pack(
        &self,
        pack_idx: usize,
        table_meta: &TableMeta,
    ) -> FilterOpResult {
        match self.op {
            FilterType::And => self.handle_logical_and(pack_idx, table_meta),
            FilterType::Or => self.handle_logical_or(pack_idx, table_meta),
            FilterType::Not => self.handle_logical_not(pack_idx, table_meta),
            FilterType::Equal
            | FilterType::NotEqual
            | FilterType::Less
            | FilterType::LessEqual
            | FilterType::Greater
            | FilterType::GreaterEqual
            | FilterType::In
            | FilterType::NotIn
            | FilterType::IsNull => self.handle_comparison(pack_idx, table_meta),
            _ => FilterOpResult::Unknown,
        }
    }

    #[inline]
    fn handle_logical_and(&self, pack_idx: usize, table_meta: &TableMeta) -> FilterOpResult {
        self.children
            .iter()
            .fold(FilterOpResult::Some, |res, child| {
                res & child.rough_check_pack(pack_idx, table_meta)
            })
    }

    #[inline]
    fn handle_logical_or(&self, pack_idx: usize, table_meta: &TableMeta) -> FilterOpResult {
        self.children
            .iter()
            .fold(FilterOpResult::None, |res, child| {
                res | child.rough_check_pack(pack_idx, table_meta)
            })
    }

    #[inline]
    fn handle_logical_not(&self, pack_idx: usize, table_meta: &TableMeta) -> FilterOpResult {
        let new_op = self.children[0].transform_not_op();
        new_op.rough_check_pack(pack_idx, table_meta)
    }

    fn transform_not_op(&self) -> FilterOperator {
        match self.op {
            FilterType::And => FilterOperator::new(
                FilterType::Or,
                self.children
                    .iter()
                    .map(|child| child.transform_not_op())
                    .collect(),
            ),
            FilterType::Or => FilterOperator::new(
                FilterType::And,
                self.children
                    .iter()
                    .map(|child| child.transform_not_op())
                    .collect(),
            ),
            FilterType::Not => FilterOperator::new(
                self.children[0].op,
                self.children[0]
                    .children
                    .iter()
                    .map(|child| child.transform_not_op())
                    .collect(),
            ),
            FilterType::Equal => FilterOperator::new_compare(
                FilterType::NotEqual,
                self.compare.as_ref().unwrap().clone(),
            ),
            FilterType::NotEqual => FilterOperator::new_compare(
                FilterType::Equal,
                self.compare.as_ref().unwrap().clone(),
            ),
            FilterType::Less => FilterOperator::new_compare(
                FilterType::GreaterEqual,
                self.compare.as_ref().unwrap().clone(),
            ),
            FilterType::LessEqual => FilterOperator::new_compare(
                FilterType::Greater,
                self.compare.as_ref().unwrap().clone(),
            ),
            FilterType::Greater => FilterOperator::new_compare(
                FilterType::LessEqual,
                self.compare.as_ref().unwrap().clone(),
            ),
            FilterType::GreaterEqual => FilterOperator::new_compare(
                FilterType::Less,
                self.compare.as_ref().unwrap().clone(),
            ),
            FilterType::In => FilterOperator::new_compare(
                FilterType::NotIn,
                self.compare.as_ref().unwrap().clone(),
            ),
            FilterType::NotIn => {
                FilterOperator::new_compare(FilterType::In, self.compare.as_ref().unwrap().clone())
            }
            FilterType::Like => FilterOperator::new_compare(
                FilterType::NotLike,
                self.compare.as_ref().unwrap().clone(),
            ),
            FilterType::NotLike => FilterOperator::new_compare(
                FilterType::Like,
                self.compare.as_ref().unwrap().clone(),
            ),
            _ => FilterOperator::default(),
        }
    }

    fn handle_comparison(&self, pack_idx: usize, table_meta: &TableMeta) -> FilterOpResult {
        let compare = self
            .compare
            .as_ref()
            .expect("operator must have compare data");
        let Some(col_meta) = table_meta.columns.get(&(compare.col_id as i32)) else {
            // If the column is not exists in the table, we treat it as no min-max index.
            return FilterOpResult::Some;
        };

        if let Some(min_max) = &col_meta.min_max {
            let result = match self.op {
                FilterType::Equal => min_max.check_equal(
                    pack_idx,
                    &compare.values[0],
                    col_meta.col_info.tp(),
                    is_unsigned(&col_meta.col_info),
                ),
                FilterType::NotEqual => min_max.check_not_equal(
                    pack_idx,
                    &compare.values[0],
                    col_meta.col_info.tp(),
                    is_unsigned(&col_meta.col_info),
                ),
                FilterType::Less => min_max.check_less(
                    pack_idx,
                    &compare.values[0],
                    col_meta.col_info.tp(),
                    is_unsigned(&col_meta.col_info),
                ),
                FilterType::LessEqual => min_max.check_less_equal(
                    pack_idx,
                    &compare.values[0],
                    col_meta.col_info.tp(),
                    is_unsigned(&col_meta.col_info),
                ),
                FilterType::Greater => min_max.check_greater(
                    pack_idx,
                    &compare.values[0],
                    col_meta.col_info.tp(),
                    is_unsigned(&col_meta.col_info),
                ),
                FilterType::GreaterEqual => min_max.check_greater_equal(
                    pack_idx,
                    &compare.values[0],
                    col_meta.col_info.tp(),
                    is_unsigned(&col_meta.col_info),
                ),
                FilterType::In => min_max.check_in(
                    pack_idx,
                    &compare.values,
                    col_meta.col_info.tp(),
                    is_unsigned(&col_meta.col_info),
                ),
                FilterType::NotIn => min_max.check_not_in(
                    pack_idx,
                    &compare.values,
                    col_meta.col_info.tp(),
                    is_unsigned(&col_meta.col_info),
                ),
                FilterType::IsNull => min_max.check_is_null(pack_idx),
                _ => unreachable!("Invalid filter type for comparison"),
            };

            if result {
                FilterOpResult::Some
            } else {
                FilterOpResult::None
            }
        } else {
            // load pack if has no min-max
            FilterOpResult::Some
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct TableScanCtx {
    table_scan: tipb::Executor,
    filter_conditions: Vec<tipb::Expr>,
}

impl TableScanCtx {
    pub fn new(table_scan: tipb::Executor, filter_conditions: Vec<tipb::Expr>) -> Self {
        TableScanCtx {
            table_scan,
            filter_conditions,
        }
    }

    pub fn is_empty_filter(&self) -> bool {
        self.filter_conditions.is_empty()
            && self
                .table_scan
                .get_tbl_scan()
                .get_pushed_down_filter_conditions()
                .is_empty()
            && self
                .table_scan
                .get_partition_table_scan()
                .get_pushed_down_filter_conditions()
                .is_empty()
    }

    pub fn to_filter_operator(&self) -> FilterOperator {
        let is_partition_scan = self.table_scan.get_tp() == tipb::ExecType::TypePartitionTableScan;
        let columns = if is_partition_scan {
            self.table_scan.get_partition_table_scan().get_columns()
        } else {
            self.table_scan.get_tbl_scan().get_columns()
        };
        let ctx = ParseCtx {
            columns: columns.to_vec(),
        };
        let mut children = vec![];
        for expr in &self.filter_conditions {
            if is_function_expr(expr) {
                children.push(parse_ti_expr(expr, &ctx));
            }
        }
        let pushed_down_filter_conditions = if is_partition_scan {
            self.table_scan
                .get_partition_table_scan()
                .get_pushed_down_filter_conditions()
        } else {
            self.table_scan
                .get_tbl_scan()
                .get_pushed_down_filter_conditions()
        };
        for expr in pushed_down_filter_conditions {
            if is_function_expr(expr) {
                children.push(parse_ti_expr(expr, &ctx));
            }
        }
        // TODO: handle ann_query
        if children.is_empty() {
            return FilterOperator::default();
        }
        debug!("to_filter_operator children: {:?}", children);
        FilterOperator::new(FilterType::And, children)
    }
}
