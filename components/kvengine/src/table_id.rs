// Copyright 2025 TiKV Project Authors. Licensed under Apache-2.0.

//! Utilities for handling TiDB (not SST) table IDs.

use std::{collections::HashSet, ops::Deref};

use bytes::BufMut;
use tidb_query_datatype::codec::table::{
    decode_table_id, RECORD_PREFIX_SEP, TABLE_PREFIX, TABLE_PREFIX_KEY_LEN,
};
use tikv_util::codec::number::NumberEncoder;

use crate::table::{DataBound, InnerKey, OwnedInnerKey};

pub fn merge_columnar_table_ids(
    source_table_ids: &[i64],
    target_table_ids: &[i64],
    source_bound: DataBound<'_>,
    target_bound: DataBound<'_>,
) -> Vec<i64> {
    let (min_source_table_id, max_source_table_id) = get_table_id_from_data_bound(source_bound);
    let (min_target_table_id, max_target_table_id) = get_table_id_from_data_bound(target_bound);
    let source_table_ids: HashSet<_> = source_table_ids.iter().cloned().collect();
    let target_table_ids: HashSet<_> = target_table_ids.iter().cloned().collect();
    let common_table_ids: HashSet<_> = source_table_ids
        .intersection(&target_table_ids)
        .cloned()
        .collect();
    let mut only_in_source: HashSet<_> = source_table_ids
        .difference(&target_table_ids)
        .cloned()
        .collect();
    only_in_source
        .retain(|&table_id| table_id < min_target_table_id || table_id > max_target_table_id);
    let mut only_in_target: HashSet<_> = target_table_ids
        .difference(&source_table_ids)
        .cloned()
        .collect();
    only_in_target
        .retain(|&table_id| table_id < min_source_table_id || table_id > max_source_table_id);
    let mut merged_table_ids = common_table_ids
        .union(&only_in_source)
        .cloned()
        .collect::<HashSet<_>>()
        .union(&only_in_target)
        .cloned()
        .collect::<Vec<_>>();
    merged_table_ids.sort();
    merged_table_ids
}

/// Try to get the table id from the data bound.
///
/// If the data bound is not a table id, min table id is 0 and max table id is
/// i64::MAX. If the upper bound is starts with "m", max table id should be 0.
pub fn get_table_id_from_data_bound(data_bound: DataBound<'_>) -> (i64, i64) {
    let data_upper_bound = data_bound.upper_bound.deref();
    let start_table_id = decode_table_id(data_bound.lower_bound.deref()).unwrap_or(0);
    let end_table_id = if !data_upper_bound.is_empty() && data_upper_bound < TABLE_PREFIX {
        0
    } else {
        let mut table_id = decode_table_id(data_upper_bound).unwrap_or(i64::MAX);
        if data_upper_bound.len() >= TABLE_PREFIX_KEY_LEN
            && &data_upper_bound[TABLE_PREFIX_KEY_LEN..] <= RECORD_PREFIX_SEP
        {
            table_id -= 1;
        }
        table_id
    };

    (start_table_id, end_table_id)
}

/// Return true when both keys are table keys and they belong to the same table.
///
/// Should check `ApiV2::is_belongs_to_same_keyspace()` first if the keys can
/// belong to different keyspace.
pub fn keys_belong_to_same_table(k0: InnerKey<'_>, k1: InnerKey<'_>) -> bool {
    k0.len() >= TABLE_PREFIX_KEY_LEN
        && k1.len() >= TABLE_PREFIX_KEY_LEN
        && k0[..TABLE_PREFIX_KEY_LEN] == k1[..TABLE_PREFIX_KEY_LEN]
}

pub fn encode_table_prefix_key(table_id: i64) -> OwnedInnerKey {
    let mut key = Vec::with_capacity(TABLE_PREFIX_KEY_LEN);
    key.put(TABLE_PREFIX);
    key.encode_i64(table_id).unwrap();
    OwnedInnerKey::new(key.into())
}

/// Return whether the key is on a table boundary.
#[inline]
pub fn is_table_boundary_key(key: InnerKey<'_>) -> bool {
    key.len() == TABLE_PREFIX_KEY_LEN && decode_table_id(key.as_ref()).is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::table::InnerKey;

    #[test]
    fn test_keys_belong_to_same_table() {
        let cases: Vec<(&'static str, &'static str, bool)> = vec![
            (
                "7800002A7480000000000000015F728000000000000001", // table id before `5F72`
                "7800002A7480000000000000015F728000000000000002",
                true,
            ),
            (
                "7800002A7480000000000000015F728000000000000001",
                "7800002A7480000000000000025F728000000000000001",
                false,
            ),
            (
                "7480000000000000015F728000000000000001",
                "7480000000000000015F728000000000000001",
                true,
            ),
            (
                "7480000000000000015F728000000000000001",
                "7480000000000000025F728000000000000001",
                false,
            ),
            (
                "7800002A",
                "7800002A7480000000000000115F728000000000000001",
                false,
            ),
            (
                "7800002A7480000000000000015F728000000000000001",
                "7800002AFF",
                false,
            ),
        ];

        for (k0, k1, expected) in cases {
            let key0 = hex::decode(k0).unwrap();
            let key1 = hex::decode(k1).unwrap();
            assert_eq!(
                keys_belong_to_same_table(
                    InnerKey::from_outer_key(&key0),
                    InnerKey::from_outer_key(&key1)
                ),
                expected
            );
        }
    }

    #[test]
    fn test_is_table_boundary_key() {
        let cases: Vec<(&'static str, bool)> = vec![
            ("748000000000000001", true),
            ("7480000000000000015F728000000000000001", false),
            ("758000000000000001", false),
        ];

        for (k, expected) in cases {
            let key = hex::decode(k).unwrap();
            assert_eq!(
                is_table_boundary_key(InnerKey::from_inner_buf(&key)),
                expected
            );
        }
    }
}
