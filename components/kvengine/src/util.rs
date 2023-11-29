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
}

impl PropertiesHelper {
    pub fn new_from_shard_meta(meta: &ShardMeta) -> Self {
        Self::new(
            meta.get_property(DEL_PREFIXES_KEY),
            meta.range.inner_key_off,
        )
    }

    fn new(del_prefixes_bytes: Option<Bytes>, inner_key_off: usize) -> Self {
        let del_prefixes = if let Some(bs) = del_prefixes_bytes {
            DeletePrefixes::unmarshal(bs.chunk(), inner_key_off)
        } else {
            DeletePrefixes::new_with_inner_key_off(inner_key_off)
        };
        Self { del_prefixes }
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
            let del_prefixes = DeletePrefixes::unmarshal(&bs, inner_key_off);
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
}
