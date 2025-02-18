// Copyright 2022 TiKV Project Authors. Licensed under Apache-2.0.

#![allow(elided_lifetimes_in_paths)]

pub mod changeset;
pub use changeset::*;

// The generated `Debug` implementation is too verbose (especially the
// `keyspace_meta` field), so we provide the `Display`.
impl std::fmt::Display for ClusterBackupMeta {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ClusterBackupMeta")
            .field("stores", &self.get_stores())
            .field("cluster_id", &self.cluster_id)
            .field("backup_ts", &self.backup_ts)
            .field("alloc_id", &self.alloc_id)
            .field("safe_ts", &self.safe_ts)
            .field("keyspace_meta_cnt", &self.keyspace_meta.len())
            .field("meta_revision", &self.meta_revision)
            .field("is_lightweight", &self.is_lightweight)
            .field("tolerated_err", &self.tolerated_err)
            .finish()
    }
}
