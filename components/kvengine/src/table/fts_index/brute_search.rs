// Copyright 2024 TiKV Project Authors. Licensed under Apache-2.0.

use std::{borrow::Cow, sync::Arc};

use async_trait::async_trait;
use clara_fts::BruteScoredSearcher;
use tidb_query_datatype::{
    codec::collation::{encoding::*, Encoding},
    Collation, FieldTypeAccessor, FieldTypeFlag, FieldTypeTp,
};
use tipb::{ColumnInfo, FtsQueryInfo};

use crate::table::{
    columnar::{Block, ColumnBuffer, ColumnarFilterReader},
    schema_file::{Schema, SchemaBuf},
    Error, Result,
};

pub const VIRTUAL_SCORE_COLUMN_ID: i64 = -2050;

type StringifyFn = Box<dyn Fn(&[u8]) -> Result<Cow<'_, str>> + Send>;

/// Supported FTS query types in TiDB-CSE:
///
/// - TiDB-CSE currently supports two FTS query types from `tipb::FtsQueryType`:
///     - `FtsQueryTypeNoScore` (value = 1): Does not require a score column.
///     - `FtsQueryTypeWithScore` (value = 2): Requires a score column.
const SUPPORTED_FTS_TYPES: &[tipb::FtsQueryType] = &[
    tipb::FtsQueryType::FtsQueryTypeNoScore,
    tipb::FtsQueryType::FtsQueryTypeWithScore,
];

fn decode<E: Encoding>(input: &[u8]) -> Result<Vec<u8>> {
    E::decode(input).map_err(|e| Error::Other(e.to_string()))
}

pub struct FtsBruteForceReader {
    // `fts_col_idx` is used to find the fts_col in the schema.
    // it may be `None` if schema does not contain fts_col.
    fts_col_idx: Option<usize>,
    // `inner_reader` is the reader that actually reads fts_col,
    // it always reads the fts_col column and then score_col is calculated based on the result.
    inner_reader: Box<dyn ColumnarFilterReader>,
    // `string_fn` is used to convert the fts_col data to string.
    string_fn: StringifyFn,
    // `brute_searcher` is used to calculate the score for each row.
    brute_searcher: BruteScoredSearcher,
    fts_query_info: Arc<FtsQueryInfo>,
    // `score_results` is used to store the score results for each row.
    score_results: Vec<clara_fts::ScoredResult>,
}

impl FtsBruteForceReader {
    /// Checks whether this schema is a valid schema for fts reading.
    /// The score column and fts_col may not exist in the schema,
    /// need to check the validity of the schema.
    pub fn validate_schema(schema: &Schema, fts_query: &Arc<FtsQueryInfo>) -> Result<()> {
        if fts_query.get_query_type() == tipb::FtsQueryType::FtsQueryTypeNoScore {
            // Schema of type FtsQueryTypeNoScore should not contain score_col.
            if schema
                .columns
                .iter()
                .any(|c| c.get_column_id() == VIRTUAL_SCORE_COLUMN_ID)
            {
                return Err(Error::Other(format!(
                    "schema must not contain score_col with id {}.",
                    VIRTUAL_SCORE_COLUMN_ID
                )));
            }
        }
        if fts_query.get_query_type() == tipb::FtsQueryType::FtsQueryTypeWithScore {
            let last_col = schema
                .columns
                .last()
                .ok_or_else(|| Error::Other("schema must not be empty for score column".into()))?;

            if last_col.get_column_id() != VIRTUAL_SCORE_COLUMN_ID {
                return Err(Error::Other(format!(
                    "schema last column id must be {} for score column, but got {}",
                    VIRTUAL_SCORE_COLUMN_ID,
                    last_col.get_column_id()
                )));
            }
            if last_col.get_tp() != FieldTypeTp::Float as i32 {
                return Err(Error::Other(format!(
                    "schema last column type must be float32 for score column, but got {}",
                    last_col.get_tp()
                )));
            }
            if !last_col.flag().contains(FieldTypeFlag::NOT_NULL) {
                return Err(Error::Other(format!(
                    "schema last column type must be not nullable for score column, but got {}",
                    last_col.get_flag()
                )));
            }
        }

        Ok(())
    }

    /// Generates a inner schema which is used for reading necessary data
    /// without an index. More specifically, the differences are:
    /// - In FtsQueryTypeWithScore, the last column of schema is score_col,
    ///   which is not need to read. So we need to remove the last column.
    /// - In both FtsQueryTypeWithScore and FtsQueryTypeWithNoScore, fts column
    ///   may be not included in the TableScan schema (if user query does not
    ///   actually need its value). However as we are doing FTS without an index
    ///   we still need the data of fts column. So **fts column** will be added
    ///   back in inner schema.
    pub fn generate_inner_schema(schema: &Schema, fts_query: &Arc<FtsQueryInfo>) -> Result<Schema> {
        if schema.columns.is_empty() {
            return Err(Error::Other("schema must not be empty".into()));
        }

        Self::validate_schema(schema, fts_query)?;

        // Also verify whether fts_query.column is valid. We will use fts_query.column
        // to reconstruct the fts_col for reading.
        let query_type = fts_query.get_query_type();
        if !SUPPORTED_FTS_TYPES.contains(&query_type) {
            return Err(Error::Other(format!(
                "unsupported FTS query type: {:?}",
                query_type
            )));
        }

        if fts_query.get_columns().is_empty() {
            return Err(Error::Other("fts_query.columns is empty.".into()));
        }

        if !fts_query.get_columns()[0].has_column_id() {
            return Err(Error::Other(
                "missing fts_query.column.column_id for distance proj".into(),
            ));
        }
        if !fts_query.get_columns()[0].as_accessor().is_string_like() {
            return Err(Error::Other(format!(
                "fts_query.column.tp must be string for full text search, but got {}",
                fts_query.get_columns()[0].get_tp()
            )));
        }

        let mut add_fts_col = true;
        for col_info in &schema.columns {
            if col_info.get_column_id() == fts_query.get_columns()[0].get_column_id() {
                add_fts_col = false;
                break;
            }
        }

        let mut new_col = ColumnInfo::new();
        new_col.set_column_id(fts_query.get_columns()[0].get_column_id());
        new_col.set_tp(fts_query.get_columns()[0].get_tp());
        new_col.set_flag(fts_query.get_columns()[0].get_flag());
        new_col.set_collation(fts_query.get_columns()[0].get_collation());
        new_col.set_default_val(fts_query.get_columns()[0].get_default_val().into());

        let mut new_inner = (*schema.inner).clone();
        // when type is FtsQueryTypeWithScore, the last column is score_col,
        if fts_query.get_query_type() == tipb::FtsQueryType::FtsQueryTypeWithScore {
            new_inner.columns.pop();
        }

        // fts_col has existed, do not need to add fts_col again.
        if add_fts_col {
            new_inner.columns.push(new_col);
        }

        let buf = SchemaBuf {
            inner: Arc::new(new_inner),
            partitions: schema.partitions.clone(),
            table_id: schema.table_id,
            sc_spec: schema.sc_spec.clone(),
            is_sub_partition: schema.is_sub_partition,
        };
        Ok(Schema::new(buf))
    }

    pub(crate) fn new(
        inner_reader: Box<dyn ColumnarFilterReader>,
        schema: Schema,
        fts_query_info: Arc<FtsQueryInfo>,
    ) -> Result<Self> {
        let query_type = fts_query_info.get_query_type();
        if !SUPPORTED_FTS_TYPES.contains(&query_type) {
            return Err(Error::Other(format!(
                "unsupported FTS query type: {:?}",
                query_type
            )));
        }

        if fts_query_info.get_columns().is_empty() {
            return Err(Error::Other("fts_query_info.columns is empty.".into()));
        }

        let col_id = fts_query_info.get_columns()[0].get_column_id();
        let fts_col_idx = schema
            .columns
            .iter()
            .position(|c| c.get_column_id() == col_id);

        // Set the stringify function for the text encoding.
        let collation = if fts_col_idx.is_none() {
            inner_reader
                .get_schema()
                .columns
                .last()
                .unwrap()
                .as_accessor()
                .collation()
                .map_err(|e| Error::Other(e.to_string()))?
        } else {
            schema.columns[fts_col_idx.unwrap()]
                .as_accessor()
                .collation()
                .map_err(|e| Error::Other(e.to_string()))?
        };
        let string_fn: StringifyFn = match collation {
            Collation::Utf8Mb4Bin
            | Collation::Utf8Mb4BinNoPadding
            | Collation::Utf8Mb4GeneralCi
            | Collation::Utf8Mb4UnicodeCi
            | Collation::Utf8Mb40900AiCi
            | Collation::Utf8Mb40900Bin
            | Collation::Latin1Bin => Box::new(|a| {
                std::str::from_utf8(a)
                    .map(|s| Cow::Borrowed(s))
                    .map_err(|e| Error::Other(e.to_string()))
            }),

            Collation::Binary => Box::new(|a| {
                decode::<EncodingBinary>(a).and_then(|decoded_bytes| {
                    String::from_utf8(decoded_bytes)
                        .map(Cow::Owned)
                        .map_err(|e| Error::Other(e.to_string()))
                })
            }),

            Collation::GbkBin | Collation::GbkChineseCi => Box::new(|a| {
                decode::<EncodingGbk>(a).and_then(|decoded_bytes| {
                    String::from_utf8(decoded_bytes)
                        .map(Cow::Owned)
                        .map_err(|e| Error::Other(e.to_string()))
                })
            }),

            Collation::Gb18030Bin | Collation::Gb18030ChineseCi => Box::new(|a| {
                decode::<EncodingGb18030>(a).and_then(|decoded_bytes| {
                    String::from_utf8(decoded_bytes)
                        .map(Cow::Owned)
                        .map_err(|e| Error::Other(e.to_string()))
                })
            }),
        };

        let brute_searcher = BruteScoredSearcher::new(
            fts_query_info.get_query_tokenizer(),
            fts_query_info.get_query_text(),
        )
        .map_err(|e| Error::Other(e.to_string()))?;

        Ok(Self {
            fts_col_idx,
            inner_reader,
            string_fn,
            brute_searcher,
            fts_query_info,
            score_results: Vec::new(),
        })
    }

    /// `read_with_score` handles the case where fts_type is
    /// `FtsQueryTypeWithScore`.
    async fn read_with_score(
        &mut self,
        block: &mut Block,
        limit: usize,
    ) -> crate::table::Result<(usize /* read_row */, bool /* drained */)> {
        // don't need to read score column. we will fill back the score column later.
        let mut score_col =
            if block.columns.last().unwrap().col_id() as i64 == VIRTUAL_SCORE_COLUMN_ID {
                block.columns.pop().unwrap()
            } else {
                let mut col = ColumnInfo::new();
                col.set_column_id(VIRTUAL_SCORE_COLUMN_ID);
                col.set_tp(FieldTypeTp::Float as i32);
                col.set_flag(FieldTypeFlag::NOT_NULL.bits() as i32);
                ColumnBuffer::new_from_col_info(&col)
            };

        // If fts_col_idx is None, it means that there is no fts column in the schema,
        // so need to add a column in block to read the original data. When adding the
        // fts column, it will be added to the last column.
        if self.fts_col_idx.is_none() {
            let fts_col_src = ColumnBuffer::new_from_col_info(
                self.inner_reader.get_schema().columns.last().unwrap(),
            );
            block.columns.push(fts_col_src);
        }
        let (total_read_row, drained) = self.inner_reader.try_read_block(block, limit).await?;

        // get fts column
        let fts_col = if self.fts_col_idx.is_none() {
            &block.columns[block.columns.len() - 1]
        } else {
            &block.columns[self.fts_col_idx.unwrap()]
        };

        self.brute_searcher.clear();
        // add the fts data to brute_searcher.
        for row in 0..total_read_row {
            if let Some(fts_data) = fts_col.get_value(row) {
                let fts_text = (self.string_fn)(fts_data)?;
                self.brute_searcher.add_document(&fts_text);
            } else {
                self.brute_searcher.add_null();
            }
        }

        // get score results.
        let filter = clara_fts::BitmapFilter::all_match();
        self.score_results.clear();
        self.brute_searcher
            .search(&filter, &mut self.score_results)
            .map_err(|e| Error::Other(e.to_string()))?;

        let mut score_vec = vec![None; total_read_row];
        for result in &self.score_results {
            if (result.doc_id as usize) < score_vec.len() {
                score_vec[result.doc_id as usize] = Some(result.score);
            }
        }

        // fill in score_col.
        for row in 0..total_read_row {
            let score_f64 = score_vec[row].unwrap_or(0.0) as f64;
            score_col.push_value(&score_f64.to_le_bytes());
        }

        // When fts_col_idx is None, the last column of block.columns is the fts_col
        // which we added, and needs to be removed.
        if self.fts_col_idx.is_none() {
            block.columns.pop().unwrap();
        }
        block.columns.push(score_col);

        // Use retain to keep only rows that have matching scores
        block.retain_rows(|row_idx| score_vec[row_idx].is_some());

        Ok((block.length(), drained))
    }

    /// `read_no_score` handles the case where fts_type is
    /// `FtsQueryTypeNoScore`.
    /// We need to match the read fts_col column and
    /// return only the matching rows.
    async fn read_no_score(
        &mut self,
        block: &mut Block,
        limit: usize,
    ) -> crate::table::Result<(usize /* read_row */, bool /* drained */)> {
        let (total_read_row, drained) = self.inner_reader.try_read_block(block, limit).await?;

        let fts_col = &block.columns[self.fts_col_idx.unwrap()];

        self.brute_searcher.clear();
        for row in 0..total_read_row {
            if let Some(fts_data) = fts_col.get_value(row) {
                let fts_text = (self.string_fn)(fts_data)?;
                self.brute_searcher.add_document(&fts_text);
            } else {
                self.brute_searcher.add_null();
            }
        }

        let filter = clara_fts::BitmapFilter::all_match();
        self.score_results.clear();
        self.brute_searcher
            .search(&filter, &mut self.score_results)
            .map_err(|e| Error::Other(e.to_string()))?;

        let mut matched = vec![false; total_read_row];
        for result in &self.score_results {
            if (result.doc_id as usize) < matched.len() {
                matched[result.doc_id as usize] = true;
            }
        }

        // Use retain to keep only rows that have matching scores
        block.retain_rows(|row_idx| matched[row_idx]);

        Ok((block.length(), drained))
    }
}

#[async_trait]
impl ColumnarFilterReader for FtsBruteForceReader {
    async fn set_handle_range(
        &mut self,
        start_handle: &[u8],
        end_handle: &[u8],
    ) -> crate::table::Result<()> {
        self.inner_reader
            .set_handle_range(start_handle, end_handle)
            .await
    }

    async fn set_int_handle_range(
        &mut self,
        start_handle: i64,
        end_handle: Option<i64>,
    ) -> crate::table::Result<()> {
        self.inner_reader
            .set_int_handle_range(start_handle, end_handle)
            .await
    }

    // get schema of inner_reader
    fn get_schema(&self) -> &Schema {
        self.inner_reader.get_schema()
    }

    async fn try_read_block(
        &mut self,
        block: &mut Block,
        limit: usize,
    ) -> crate::table::Result<(usize /* read_row */, bool /* drained */)> {
        if self.fts_query_info.get_query_type() == tipb::FtsQueryType::FtsQueryTypeWithScore {
            self.read_with_score(block, limit).await
        } else {
            self.read_no_score(block, limit).await
        }
    }
}
