// Copyright 2021 TiKV Project Authors. Licensed under Apache-2.0.

use engine_traits::Result;

use crate::{engine::RocksEngine, util};

impl RocksEngine {
    pub fn get_cf_num_files_at_level(&self, cf: &str, level: usize) -> Result<Option<u64>> {
        let handle = util::get_cf_handle(self.as_inner(), cf)?;
        Ok(crate::util::get_cf_num_files_at_level(
            self.as_inner(),
            handle,
            level,
        ))
    }

    pub fn get_cf_num_immutable_mem_table(&self, cf: &str) -> Result<Option<u64>> {
        let handle = util::get_cf_handle(self.as_inner(), cf)?;
        Ok(crate::util::get_cf_num_immutable_mem_table(
            self.as_inner(),
            handle,
        ))
    }

    pub fn get_cf_pending_compaction_bytes(&self, cf: &str) -> Result<Option<u64>> {
        let handle = util::get_cf_handle(self.as_inner(), cf)?;
        Ok(crate::util::get_cf_pending_compaction_bytes(
            self.as_inner(),
            handle,
        ))
    }
}
