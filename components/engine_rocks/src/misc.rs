// Copyright 2020 TiKV Project Authors. Licensed under Apache-2.0.

use engine_traits::Result;
use rocksdb::FlushOptions;

use crate::{engine::RocksEngine, r2e, util};

impl RocksEngine {
    pub fn flush_cf(&self, cf: &str, wait: bool) -> Result<()> {
        let handle = util::get_cf_handle(self.as_inner(), cf)?;
        let mut fopts = FlushOptions::default();
        fopts.set_wait(wait);
        fopts.set_allow_write_stall(true);
        self.as_inner().flush_cf(handle, &fopts).map_err(r2e)
    }
}
