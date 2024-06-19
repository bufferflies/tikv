// Copyright 2024 TiKV Project Authors. Licensed under Apache-2.0.

use std::{ops::Range, sync::Arc};

use api_version::ApiV2;
use async_trait::async_trait;
use tikv_util::{
    codec::{bytes::encode_bytes, number::NumberEncoder},
    debug, warn,
};

use crate::{
    load_schema,
    schema::{SchemaDiff, TableInfo, STATE_PUBLIC},
    KvScanner,
};

const TIDB_SCHEMA_VERSION_KEY: &[u8] = b"SchemaVersionKey";
const TIDB_SCHEMA_DIFF_PREFIX: &[u8] = b"Diff";
const TIDB_META_KEY_PREFIX: u8 = b'm';
const STRING_DATA_TYPE: u8 = b's';
const HASH_DATA_TYPE: u8 = b'h';

// Sync schema for keyspace_id if needed.
pub async fn sync_schema(
    kv_getter: Arc<dyn KvGetter>,
    kv_scanner: Arc<dyn KvScanner>,
    keyspace_id: u32,
    cur_version: Option<i64>,
) -> Result<(i64, Vec<TableInfo>), String> {
    // Get schema version key from meta.
    let schema_version_data = kv_getter.get(&schema_version_key(keyspace_id)).await?;
    // This is a empty keyspace.
    if schema_version_data.is_none() {
        return Ok((0, vec![]));
    }
    let schema_version = String::from_utf8(schema_version_data.unwrap())
        .unwrap()
        .parse::<i64>()
        .unwrap();
    if let Some(cur_version) = cur_version {
        if cur_version == schema_version {
            return Ok((schema_version, vec![]));
        }
        let mut table_infos = vec![];
        if cur_version > schema_version {
            warn!(
                "sync_schema, but current version {} is greater than schema meta version {}, need full sync",
                cur_version, schema_version
            );
            return sync_all_schemas(kv_scanner, keyspace_id, schema_version).await;
        }

        debug!(
            "sync_schema, current version is {}, schema meta version {}",
            cur_version, schema_version
        );
        let version_range = cur_version + 1..schema_version + 1;
        let schema_diffs = get_schema_diff(kv_getter.clone(), version_range, keyspace_id).await?;
        debug!("sync_schema, schema diffs {:?}", schema_diffs);
        for diff in schema_diffs {
            if diff.regenerate_schema_map {
                return sync_all_schemas(kv_scanner, keyspace_id, schema_version).await;
            }
            let db_id = diff.schema_id;
            if db_id == 1 {
                continue;
            }

            let table_id = diff.table_id;
            let schema_data_key = schema_data_key(keyspace_id, db_id, table_id);
            let schema_data = kv_getter.get(&schema_data_key).await?.unwrap_or_default();
            if schema_data.is_empty() {
                warn!(
                    "sync_schema, schema data in diff keyspace: {} db: {} table: {}, diff: {} not found, skip",
                    keyspace_id, db_id, table_id, diff.version
                );
                continue;
            }
            let res: serde_json::Result<TableInfo> = serde_json::from_slice(&schema_data);
            match res {
                Ok(tbl) => {
                    debug!(
                        "sync_schema, keyspace {} table {} schema {:?}",
                        keyspace_id, table_id, tbl
                    );
                    if tbl.state == STATE_PUBLIC {
                        table_infos.push(tbl);
                    }
                }
                Err(err) => {
                    let val_str = String::from_utf8_lossy(&schema_data);
                    warn!(
                        "invalid key {:?}, val {} err {:?}",
                        schema_data_key, val_str, err
                    );
                }
            }
        }

        Ok((schema_version, table_infos))
    } else {
        sync_all_schemas(kv_scanner, keyspace_id, schema_version).await
    }
}

async fn sync_all_schemas(
    kv_scanner: Arc<dyn KvScanner>,
    keyspace_id: u32,
    schema_version: i64,
) -> Result<(i64, Vec<TableInfo>), String> {
    let keyspace_prefix = ApiV2::get_txn_keyspace_prefix(keyspace_id);
    let db_infos = load_schema(kv_scanner, &keyspace_prefix).await?;
    Ok((
        schema_version,
        db_infos
            .into_iter()
            .flat_map(|db_info| db_info.tables)
            .collect(),
    ))
}

async fn get_schema_diff(
    kv_getter: Arc<dyn KvGetter>,
    range: Range<i64>, // Schema version range (exclusive_start, inclusive_end]
    keyspace_id: u32,
) -> Result<Vec<SchemaDiff>, String> {
    let (start, end) = (range.start, range.end);
    let mut schema_diff_keys = Vec::with_capacity((end - start) as usize);
    range.into_iter().for_each(|ver| {
        schema_diff_keys.push(schema_diff_key(keyspace_id, ver));
    });
    let schema_diffs = kv_getter.batch_get(&schema_diff_keys).await?;
    let mut schema_diffs_res = Vec::with_capacity(schema_diffs.len());
    for (idx, schema_diff) in schema_diffs.into_iter().enumerate() {
        if schema_diff.is_none() {
            warn!(
                "sync_schema, keyspace {} schema diff for ver {} not found",
                keyspace_id,
                start + 1 + idx as i64
            );
            continue;
        }
        let schema_diff = schema_diff.unwrap();
        let res: serde_json::Result<SchemaDiff> = serde_json::from_slice(&schema_diff);
        match res {
            Ok(schema_diff) => {
                schema_diffs_res.push(schema_diff);
            }
            Err(err) => {
                return Err(format!(
                    "keyspace {} schema diff ver {} decode failed, err {:?}",
                    keyspace_id,
                    start + 1 + idx as i64,
                    err
                ));
            }
        }
    }
    Ok(schema_diffs_res)
}

fn schema_version_key(keyspace_id: u32) -> Vec<u8> {
    let mut key = api_version::ApiV2::get_txn_keyspace_prefix(keyspace_id);
    let encoded_key = encode_bytes(TIDB_SCHEMA_VERSION_KEY);
    key.reserve(encoded_key.len() + 1 + 8);
    key.push(TIDB_META_KEY_PREFIX);
    key.extend_from_slice(&encoded_key);
    key.encode_u64(STRING_DATA_TYPE as u64).unwrap();
    key
}

fn schema_diff_key(keyspace_id: u32, ver: i64) -> Vec<u8> {
    let ver_str = ver.to_string();
    let mut key = api_version::ApiV2::get_txn_keyspace_prefix(keyspace_id);
    let mut raw_key = TIDB_SCHEMA_DIFF_PREFIX.to_vec();
    raw_key.push(b':');
    raw_key.extend_from_slice(ver_str.as_bytes());
    key.reserve(raw_key.len() + 1 + 1 + 8);
    key.push(TIDB_META_KEY_PREFIX);
    key.extend_from_slice(&encode_bytes(&raw_key));
    key.encode_u64(STRING_DATA_TYPE as u64).unwrap();
    key
}

fn schema_data_key(keyspace_id: u32, db_id: i64, table_id: i64) -> Vec<u8> {
    let mut key = api_version::ApiV2::get_txn_keyspace_prefix(keyspace_id);
    let enc_db_key = encode_bytes(format!("DB:{}", db_id).as_bytes());
    let enc_table_key = encode_bytes(format!("Table:{}", table_id).as_bytes());
    key.reserve(enc_db_key.len() + enc_table_key.len() + 1);
    key.push(TIDB_META_KEY_PREFIX);
    key.extend_from_slice(&enc_db_key);
    key.encode_u64(HASH_DATA_TYPE as u64).unwrap();
    key.extend_from_slice(&enc_table_key);

    key
}

#[async_trait]
pub trait KvGetter: Send + Sync {
    async fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>, String>;
    async fn batch_get(&self, keys: &[Vec<u8>]) -> Result<Vec<Option<Vec<u8>>>, String>;
}
