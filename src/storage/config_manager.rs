// Copyright 2016 TiKV Project Authors. Licensed under Apache-2.0.

//! Storage online config manager.

use std::{convert::TryInto, sync::Arc};

use concurrency_manager::ConcurrencyManager;
use file_system::{IoPriority, IoRateLimiter, IoType};
use online_config::{ConfigChange, ConfigManager, ConfigValue, Result as CfgResult};
use strum::IntoEnumIterator;
use tikv_kv::Engine;
use tikv_util::config::{ReadableDuration, ReadableSize};

use crate::storage::{lock_manager::LockManager, TxnScheduler};

pub struct StorageConfigManger<E: Engine, L: LockManager> {
    concurrency_manager: ConcurrencyManager,
    io_rate_limiter: Arc<IoRateLimiter>,
    scheduler: TxnScheduler<E, L>,
    // whether scheduler worker pool shares unified read pool.
    use_separated_scheduler_pool: bool,
}

impl<E: Engine, L: LockManager> StorageConfigManger<E, L> {
    pub fn new(
        concurrency_manager: ConcurrencyManager,
        io_rate_limiter: Arc<IoRateLimiter>,
        scheduler: TxnScheduler<E, L>,
        use_separated_scheduler_pool: bool,
    ) -> Self {
        StorageConfigManger {
            concurrency_manager,
            io_rate_limiter,
            scheduler,
            use_separated_scheduler_pool,
        }
    }
}

// Safety: We only access the `SchedulerPool` in TxnScheduler, so it's thread
// safe here.
unsafe impl<E: Engine, L: LockManager> Send for StorageConfigManger<E, L> {}
unsafe impl<E: Engine, L: LockManager> Sync for StorageConfigManger<E, L> {}

impl<E: Engine, L: LockManager> ConfigManager for StorageConfigManger<E, L> {
    fn dispatch(&mut self, mut change: ConfigChange) -> CfgResult<()> {
        if let Some(ConfigValue::Module(mut _block_cache)) = change.remove("block_cache") {
            // TODO: do not support change block-cache.capacity currnetly.
        } else if let Some(ConfigValue::Module(_flow_control)) = change.remove("flow_control") {
            // TODO: support OnlineConfig for kvengine's flow_control
        }
        if let Some(ConfigValue::Module(mut io_rate_limit)) = change.remove("io_rate_limit") {
            if let Some(limit) = io_rate_limit.remove("max_bytes_per_sec") {
                let limit: ReadableSize = limit.into();
                self.io_rate_limiter.set_io_rate_limit(limit.0 as usize);
            }

            for t in IoType::iter() {
                if let Some(priority) = io_rate_limit.remove(&(t.as_str().to_owned() + "_priority"))
                {
                    let priority: IoPriority = priority.try_into()?;
                    self.io_rate_limiter.set_io_priority(t, priority);
                }
            }
        }
        if let Some(v) = change.get("scheduler_worker_pool_size") {
            if self.use_separated_scheduler_pool {
                let pool_size: usize = v.into();
                self.scheduler.scale_pool_size(pool_size);
            } else {
                warn!(
                    "cannot scale scheduler worker pool when config 'use_separated_scheduler_pool' is false."
                );
            }
        }
        if let Some(ConfigValue::Module(mut max_ts)) = change.remove("max_ts") {
            if let Some(v) = max_ts.remove("action_on_invalid_update") {
                let str_v: String = v.into();
                let action: concurrency_manager::ActionOnInvalidMaxTs = str_v.try_into()?;
                self.concurrency_manager
                    .set_action_on_invalid_max_ts_update(action);
            }
            if let Some(v) = max_ts.remove("max_drift") {
                let dur_v: ReadableDuration = v.into();
                self.concurrency_manager.set_max_ts_drift_allowance(dur_v.0);
            }
        }

        Ok(())
    }
}
