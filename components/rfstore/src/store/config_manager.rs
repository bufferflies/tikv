// Copyright 2025 TiKV Project Authors. Licensed under Apache-2.0.

use std::sync::Arc;

use online_config::{ConfigChange, ConfigManager, ConfigValue, OnlineConfig};
use tikv_util::{
    config::{ReadableDuration, ReadableSize, VersionTrack},
    mpsc::Sender,
    yatp_pool::FuturePool,
};

use crate::store::{Config, IoWorkerTask};

pub struct RfstoreConfigManager {
    config: Arc<VersionTrack<Config>>,
    io_worker_sender: Sender<Option<IoWorkerTask>>,
    apply_pool: FuturePool,
}

impl RfstoreConfigManager {
    pub(crate) fn new(
        config: Arc<VersionTrack<Config>>,
        io_worker_sender: Sender<Option<IoWorkerTask>>,
        apply_pool: FuturePool,
    ) -> Self {
        Self {
            config,
            io_worker_sender,
            apply_pool,
        }
    }

    #[cfg(any(test, feature = "testexport"))]
    pub fn get_config(&self) -> Config {
        self.config.value().clone()
    }

    #[cfg(any(test, feature = "testexport"))]
    pub fn apply_pool(&self) -> FuturePool {
        self.apply_pool.clone()
    }
}

impl ConfigManager for RfstoreConfigManager {
    fn dispatch(&mut self, change: ConfigChange) -> online_config::Result<()> {
        if change.contains_key("raft_worker_max_batch_size")
            || change.contains_key("io_worker_min_write_duration")
        {
            let max_batch_size = change
                .get("raft_worker_max_batch_size")
                .map(|v| ReadableSize::from(v.clone()).0 as usize);
            let min_write_duration = change
                .get("io_worker_min_write_duration")
                .map(|v| ReadableDuration::from(v.clone()).0);
            self.io_worker_sender
                .send(Some(IoWorkerTask::UpdateConfig {
                    max_batch_size,
                    min_write_duration,
                }))
                .unwrap();
        }
        if let Some(ConfigValue::Module(apply_change)) = change.get("apply_batch_system") {
            // currently, only support change pool size.
            if let Some(v) = apply_change.get("pool_size") {
                let new_size: usize = v.into();
                let current_size = self.apply_pool.get_pool_size();
                self.apply_pool.scale_pool_size(new_size);
                tikv_util::info!("apply pool thread change"; "current" => current_size, "new" => new_size);
            }
        }

        self.config.update(|c| c.update(change))
    }
}
