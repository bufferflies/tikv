// Copyright 2025 TiKV Project Authors. Licensed under Apache-2.0.

//! Coprocessor online config manager.

use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc,
};

use online_config::{ConfigChange, ConfigManager, Result as CfgResult};
use tikv_util::config::ReadableSize;

pub(super) struct CopConfigManager {
    cop_max_resp_size: Arc<AtomicU64>,
}

impl CopConfigManager {
    pub fn new(cop_max_resp_size: Arc<AtomicU64>) -> Self {
        Self { cop_max_resp_size }
    }
}

impl ConfigManager for CopConfigManager {
    fn dispatch(&mut self, mut change: ConfigChange) -> CfgResult<()> {
        if let Some(s) = change.remove("cop_max_resp_size") {
            let new_size: ReadableSize = s.into();
            self.cop_max_resp_size.store(new_size.0, Ordering::Relaxed);
        }
        Ok(())
    }
}
