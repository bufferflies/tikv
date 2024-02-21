// Copyright 2023 TiKV Project Authors. Licensed under Apache-2.0.

use bytes::{Buf, Bytes};

use crate::{DeletePrefixes, ShardMeta, DEL_PREFIXES_KEY};

/// A helper function to evenly distribute `total` into `count` parts.
/// Note: when `total` <= `count`, return `[1; total]`.
pub fn evenly_distribute(total: usize, count: usize) -> Vec<usize> {
    debug_assert!(count > 0);
    if total <= count {
        return vec![1; total];
    }
    let quotient = total / count;
    let remainder = total % count;
    let mut ret = vec![quotient; count];
    for i in 0..remainder {
        ret[i] += 1;
    }
    ret
}

/// Helper for merging or splitting properties.
///
/// Currently only delete prefixes are handled.
///
/// Splitting properties is necessary for the following scenarios:
/// 1. `preprocess_pending_splits` (`build_split_pb`).
///
/// Merging properties is necessary for the following scenarios:
/// 1. `ShardMeta.commit_merge` (merge `ShardMeta` on region merge).
/// 2. `kvengine::Engine::commit_merge` (merge `Shard` on region merge).
/// 3. `restore_keyspace::BackupCluster::gather_sstables` (merge `ShardMeta` on
/// merging backup regions).
pub struct PropertiesHelper {
    del_prefixes: DeletePrefixes,

    /// Prefix of shard range, see `ShardRange::prefix()`.
    range_prefix: Vec<u8>,
    /// Whether to rewrite range prefix of `del_prefixes`.
    /// When enabled, range prefix of `del_prefixes` to be merged will be
    /// rewritten to `range_prefix`.
    rewrite_range_prefix: bool,
}

impl PropertiesHelper {
    pub fn new_from_shard_meta(meta: &ShardMeta) -> Self {
        Self::new(
            meta.get_property(DEL_PREFIXES_KEY),
            meta.range.inner_key_off,
            meta.range.prefix().to_owned(),
        )
    }

    fn new(del_prefixes_bytes: Option<Bytes>, inner_key_off: usize, range_prefix: Vec<u8>) -> Self {
        let del_prefixes = if let Some(bs) = del_prefixes_bytes {
            DeletePrefixes::unmarshal(bs.chunk(), inner_key_off)
        } else {
            DeletePrefixes::new_with_inner_key_off(inner_key_off)
        };
        Self {
            del_prefixes,
            range_prefix,
            rewrite_range_prefix: false,
        }
    }

    pub fn set_rewrite_range_prefix(&mut self, rewrite_range_prefix: bool) {
        self.rewrite_range_prefix = rewrite_range_prefix;
    }

    pub fn merge_shard_meta(&mut self, other: &ShardMeta) {
        self.merge(
            other
                .get_property(DEL_PREFIXES_KEY)
                .map(|prop| prop.to_vec()),
            other.range.inner_key_off,
        );
    }

    // TODO: pub fn merge_shard(..)

    fn merge(&mut self, del_prefixes_bytes: Option<Vec<u8>>, inner_key_off: usize) {
        // `self.del_prefixes.inner_key_off` & `inner_key_off` are not necessarily
        // equal.
        if let Some(bs) = del_prefixes_bytes {
            let mut del_prefixes = DeletePrefixes::unmarshal(&bs, inner_key_off);
            if self.rewrite_range_prefix
                && inner_key_off > 0
                && inner_key_off == self.range_prefix.len()
            {
                del_prefixes.rewrite_range_prefix(&self.range_prefix);
            }
            self.del_prefixes.merge(&del_prefixes);
        }
    }

    pub fn build_to_shard_meta(&self, meta: &mut ShardMeta) {
        if !self.del_prefixes.prefixes.is_empty() {
            info!(
                "{} PropertiesMerger.build_to_shard_meta: {:?}",
                meta.tag(),
                self.del_prefixes
            );
            meta.set_property(DEL_PREFIXES_KEY, &self.del_prefixes.marshal());
        }
    }

    fn split(&self, start_key: &[u8], end_key: &[u8]) -> DeletePrefixes {
        self.del_prefixes
            .build_split(start_key, end_key, self.del_prefixes.inner_key_off)
    }

    pub fn split_to_properties(
        &self,
        start_key: &[u8],
        end_key: &[u8],
        props: &mut kvenginepb::Properties,
    ) {
        let split_del_prefixes = self.split(start_key, end_key);
        if !split_del_prefixes.is_empty() {
            props.mut_keys().push(DEL_PREFIXES_KEY.to_owned());
            props.mut_values().push(split_del_prefixes.marshal());
        }
    }
}

#[cfg(test)]
mod tests {
    use api_version::api_v2::KEYSPACE_PREFIX_LEN;

    use super::*;

    #[test]
    fn test_evenly_distribute() {
        assert_eq!(evenly_distribute(0, 1), Vec::<usize>::new());
        assert_eq!(evenly_distribute(10, 1), vec![10]);
        assert_eq!(evenly_distribute(10, 2), vec![5, 5]);
        assert_eq!(evenly_distribute(10, 3), vec![4, 3, 3]);
        assert_eq!(evenly_distribute(10, 4), vec![3, 3, 2, 2]);
        assert_eq!(evenly_distribute(10, 5), vec![2, 2, 2, 2, 2]);
        assert_eq!(evenly_distribute(10, 6), vec![2, 2, 2, 2, 1, 1]);
        assert_eq!(evenly_distribute(10, 7), vec![2, 2, 2, 1, 1, 1, 1]);
        assert_eq!(evenly_distribute(10, 8), vec![2, 2, 1, 1, 1, 1, 1, 1]);
        assert_eq!(evenly_distribute(10, 9), vec![2, 1, 1, 1, 1, 1, 1, 1, 1]);
        assert_eq!(evenly_distribute(10, 10), vec![1; 10]);
        assert_eq!(evenly_distribute(10, 11), vec![1; 10]);
        assert_eq!(evenly_distribute(10, 100), vec![1; 10]);
    }

    #[test]
    fn test_properties_helper() {
        let dp0 = DeletePrefixes::new_with_inner_key_off(KEYSPACE_PREFIX_LEN)
            .merge_prefix(b"x000101")
            .merge_prefix(b"x000103");

        let dp1 = DeletePrefixes::new_with_inner_key_off(KEYSPACE_PREFIX_LEN)
            .merge_prefix(b"x0101033")
            .merge_prefix(b"x0101066");

        let mut helper = PropertiesHelper::new(
            Some(dp0.marshal().into()),
            KEYSPACE_PREFIX_LEN,
            b"x000".to_vec(),
        );
        helper.set_rewrite_range_prefix(true);
        helper.merge(Some(dp1.marshal()), KEYSPACE_PREFIX_LEN);
        assert_eq!(
            helper.del_prefixes.prefixes,
            vec![
                b"x000101".to_vec(),
                b"x000103".to_vec(),
                b"x0001066".to_vec(),
            ]
        );

        helper.set_rewrite_range_prefix(false);
        let dp2 = DeletePrefixes::new_with_inner_key_off(KEYSPACE_PREFIX_LEN)
            .merge_prefix(b"x020101")
            .merge_prefix(b"x0201077");
        helper.merge(Some(dp2.marshal()), KEYSPACE_PREFIX_LEN);
        assert_eq!(
            helper.del_prefixes.prefixes,
            vec![
                b"x000101".to_vec(),
                b"x000103".to_vec(),
                b"x0001066".to_vec(),
                b"x020101".to_vec(),
                b"x0201077".to_vec(),
            ]
        );
    }
}
