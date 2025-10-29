// Copyright 2024 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    marker::PhantomData,
    sync::{Arc, Mutex},
    time::Duration,
};

use prometheus::IntGauge;
use tikv_util::{
    memory::MemoryLimiter,
    time::{Instant, Limiter},
};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use crate::{
    metrics::{ENGINE_DFS_LOAD_MEMORY_WAIT_DURATION, ENGINE_THROTTLE_ACTION_COUNTER},
    KvEngineConfig, Options, ShardTag,
};

/// All members are in bytes.
#[derive(Clone, Default, Debug)]
pub struct LimiterOptions {
    pub enable: bool,
    pub soft_limit: u64,
    pub hard_limit: u64,
    pub max_speed_limit: u64,
    pub min_speed_limit: u64,
}

pub trait LimiterLevel {
    const TAG: &'static str;
}

#[derive(Clone)]
pub struct StoreLevel {}

impl LimiterLevel for StoreLevel {
    const TAG: &'static str = "store";
}

pub type StoreLimiter = WriteRateLimiter<StoreLevel>;

#[derive(Clone)]
pub struct RegionLevel {}

impl LimiterLevel for RegionLevel {
    const TAG: &'static str = "region";
}

pub type RegionLimiter = WriteRateLimiter<RegionLevel>;

struct LimiterMetrics {
    speed_metric: IntGauge,
    last_record_time: Mutex<Instant>,
}

// `Lv` makes store & region limiters of different types and avoids misuse.
#[derive(Clone)]
pub struct WriteRateLimiter<Lv: LimiterLevel> {
    options: LimiterOptions,
    limiter: Arc<Limiter>,
    metrics: Option<Arc<LimiterMetrics>>,
    _phantom: PhantomData<Lv>,
}

impl WriteRateLimiter<StoreLevel> {
    pub fn new(options: LimiterOptions, speed_metric: IntGauge) -> Self {
        Self::new_impl(options, Some(speed_metric))
    }
}

impl WriteRateLimiter<RegionLevel> {
    pub fn new(options: LimiterOptions) -> Self {
        Self::new_impl(options, None)
    }

    /// New from another one.
    pub fn new_from(another: &WriteRateLimiter<RegionLevel>) -> Self {
        Self::new_impl(another.options.clone(), None)
    }
}

/// Implement interfaces for `storage::txn::flow_controller::FlowController`.
impl<Lv: LimiterLevel> WriteRateLimiter<Lv> {
    pub fn should_drop(&self, _region_id: u64) -> bool {
        // TODO: early drop ?
        false
    }

    pub fn discard_ratio(&self, _region_id: u64) -> f64 {
        0.0
    }

    pub fn enable(&self, _enable: bool) {
        // TODO: support online enable/disable
    }

    pub fn enabled(&self) -> bool {
        self.options.enable
    }

    pub fn consume(&self, _region_id: u64, bytes: usize) -> Duration {
        self.limiter.consume_duration(bytes)
    }

    pub fn unconsume(&self, _region_id: u64, bytes: usize) {
        self.limiter.unconsume(bytes);
    }

    pub fn is_unlimited(&self, _region_id: u64) -> bool {
        self.limiter.speed_limit() == f64::INFINITY
    }

    pub fn total_bytes_consumed(&self, _region_id: u64) -> usize {
        self.limiter.total_bytes_consumed()
    }

    pub fn speed_limit(&self) -> f64 {
        self.limiter.speed_limit()
    }

    pub fn set_speed_limit(&self, _region_id: u64, speed_limit: f64) {
        self.limiter.set_speed_limit(speed_limit);
    }
}

impl<Lv: LimiterLevel> WriteRateLimiter<Lv> {
    fn new_impl(options: LimiterOptions, speed_metric: Option<IntGauge>) -> Self {
        let limiter = Arc::new(
            <Limiter>::builder(f64::INFINITY)
                .refill(Duration::from_millis(1))
                .build(),
        );
        let metrics = speed_metric.map(|speed_metric| {
            Arc::new(LimiterMetrics {
                speed_metric,
                last_record_time: Mutex::new(Instant::now_coarse()),
            })
        });
        Self {
            options,
            limiter,
            metrics,
            _phantom: PhantomData,
        }
    }

    pub fn dummy() -> Self {
        let options = LimiterOptions {
            enable: false,
            ..Default::default()
        };
        Self::new_impl(options, None)
    }

    fn update_speed_limit(&self, tag: &ShardTag, throttle: f64) {
        let pre = self.limiter.speed_limit();
        self.limiter.set_speed_limit(throttle);

        if pre.is_infinite() && throttle.is_finite() {
            info!("{} WriteRateLimiter::start_throttle", tag; "throttle" => throttle);
            ENGINE_THROTTLE_ACTION_COUNTER
                .with_label_values(&[Lv::TAG, "start_throttle"])
                .inc();
        } else if pre.is_finite() && throttle.is_infinite() {
            info!("{} WriteRateLimiter::stop_throttle", tag; "pre_throttle" => pre);
            ENGINE_THROTTLE_ACTION_COUNTER
                .with_label_values(&[Lv::TAG, "stop_throttle"])
                .inc();
        }
        self.update_statistics();
    }

    fn update_statistics(&self) {
        if let Some(metrics) = self.metrics.as_ref() {
            let mut last_record_time = metrics.last_record_time.lock().unwrap();
            let dur = last_record_time.saturating_elapsed_secs();
            if dur < f64::EPSILON {
                return;
            }

            let total = self.limiter.total_bytes_consumed();
            self.limiter.reset_statistics();
            *last_record_time = Instant::now_coarse();
            drop(last_record_time);

            let rate = total as f64 / dur;
            debug!("WriteRateLimiter::update_statistics";
                "rate" => rate,
                "total" => total,
                "dur" => ?dur,
            );
            metrics.speed_metric.set(rate as i64);
        }
    }

    pub fn update_usage(&self, tag: &ShardTag, usage: u64) {
        if !self.options.enable {
            return;
        }

        let throttle = Self::calculate_throttle(
            usage,
            self.options.soft_limit,
            self.options.hard_limit,
            self.options.max_speed_limit,
            self.options.min_speed_limit,
        );
        debug!("{} ShardLimiter::update_usage", tag;
            "usage" => usage,
            "throttle" => throttle,
        );
        self.update_speed_limit(tag, throttle);
    }

    /// Calculate throttle according to current usage.
    ///
    /// All the parameters are in bytes or bytes/s.
    ///
    /// `usage` is the current usage of component (e.g. memtables) to be
    /// throttled.
    fn calculate_throttle(
        usage: u64,
        soft_limit: u64,
        hard_limit: u64,
        max_speed_limit: u64,
        min_speed_limit: u64,
    ) -> f64 {
        debug_assert!(hard_limit >= soft_limit);
        debug_assert!(max_speed_limit >= min_speed_limit);

        if usage < soft_limit {
            f64::INFINITY
        } else if usage >= hard_limit {
            min_speed_limit as f64
        } else {
            (hard_limit - usage) as f64 / (hard_limit - soft_limit) as f64
                * (max_speed_limit - min_speed_limit) as f64
                + min_speed_limit as f64
        }
    }
}

/// DfsLimiter is used to limit the memory used for dfs loading files.
#[derive(Clone)]
pub(crate) struct DfsLoadLimiter {
    semaphore: Arc<Semaphore>,
    memory_limiter: MemoryLimiter,
}

impl DfsLoadLimiter {
    pub(crate) fn new(cfg: &KvEngineConfig, opts: &Options) -> DfsLoadLimiter {
        let num_cores = tikv_util::sys::SysQuota::cpu_cores_quota();
        let global_concurrency = (num_cores.max(1.0) as usize) * cfg.dfs_load_concurrency_per_core;
        let semaphore = Arc::new(Semaphore::new(global_concurrency));
        let memory_limiter = MemoryLimiter::new(
            opts.dfs_load_memory_limit,
            Some(crate::metrics::ENGINE_DFS_LOAD_MEMORY_USAGE.clone()),
        );
        Self {
            semaphore,
            memory_limiter,
        }
    }

    pub(crate) async fn acquire_permit(&self) -> OwnedSemaphorePermit {
        let global_semaphore = self.semaphore.clone();
        // We never close the semaphore, so it is safe to unwrap.
        global_semaphore.acquire_owned().await.unwrap()
    }

    /// Acquire memory for loading a file of specified size, waiting until
    /// memory is available. This will block indefinitely until enough
    /// memory is released by other operations. For critical operations like
    /// file loading, we must wait until memory is available.
    ///
    /// If the file size exceeds the total memory capacity, this function
    /// will skip memory limiting and log a warning instead of deadlocking.
    pub(crate) async fn acquire_memory_blocking(
        &self,
        file_size: u64,
    ) -> tikv_util::memory::MemoryLimiterGuard {
        let capacity = self.memory_limiter.capacity();

        // Check if file size exceeds the total capacity - if so, we can never
        // acquire enough memory and would deadlock. Skip memory limiting for
        // this file and log a warning.
        if file_size > capacity {
            warn!(
                "file size exceeds dfs load memory capacity, skipping memory limit for this file";
                "file_size" => file_size,
                "capacity" => capacity,
                "file_size_mb" => file_size / (1024 * 1024),
                "capacity_mb" => capacity / (1024 * 1024)
            );
            // Return a dummy guard with 0 size to avoid deadlock
            return self
                .memory_limiter
                .acquire(0)
                .expect("acquiring 0 bytes should always succeed");
        }

        let start = Instant::now();
        let mut waited = false;

        loop {
            match self.memory_limiter.acquire(file_size) {
                Ok(guard) => {
                    // Only record wait time if we actually had to wait
                    if waited {
                        let wait_duration = start.saturating_elapsed_secs();
                        ENGINE_DFS_LOAD_MEMORY_WAIT_DURATION.observe(wait_duration);
                    }
                    return guard;
                }
                Err(_exceeded) => {
                    waited = true;
                    // Wait a bit before retrying - file loading is critical, so we must wait
                    tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculate_throttle() {
        let cases = vec![
            (0, f64::INFINITY), // (usage, expected throttle)
            (200, f64::INFINITY),
            (255, f64::INFINITY),
            (256, 50.0),
            (300, 45.8),
            (512, 25.0),
            (600, 17.0),
            (700, 7.5),
            (767, 1.0),
            (768, 1.0),
            (800, 1.0),
        ];

        for (usage, throttle_mb) in cases {
            let result = RegionLimiter::calculate_throttle(
                usage << 20,
                256 << 20,
                768 << 20,
                50 << 20,
                1 << 20,
            );

            let throttle = throttle_mb * 1024.0 * 1024.0;
            let result_mb = result / 1024.0 / 1024.0;
            let diff = (result_mb - throttle_mb).abs();
            assert!(
                result == throttle || diff < 1.0,
                "result: {}, throttle: {}, diff: {}",
                result_mb,
                throttle_mb,
                diff
            );
        }

        let corner_cases = vec![
            (
                512,  // usage
                256,  // soft_limit
                768,  // hard_limit
                50,   // max_speed_limit
                50,   // min_speed_limit
                50.0, // expected throttle
            ),
            (512, 512, 512, 50, 1, 1.0), // soft_limit == hard_limit
        ];
        for (usage, soft, hard, max, min, expected) in corner_cases {
            let result = RegionLimiter::calculate_throttle(usage, soft, hard, max, min);
            assert_eq!(result, expected);
        }
    }

    #[tokio::test]
    async fn test_acquire_memory_blocking() {
        use crate::Options;

        // Test case 1: file_size within capacity - should acquire memory successfully
        {
            let config = KvEngineConfig {
                dfs_load_concurrency_per_core: 64,
                dfs_load_memory_ratio: 0.25,
                ..Default::default()
            };
            let mut opts = Options::default();
            opts.dfs_load_memory_limit = 1024 * 1024; // 1 MB capacity

            let limiter = DfsLoadLimiter::new(&config, &opts);

            // Try to acquire 512 KB (within capacity)
            let file_size = 512 * 1024;
            let guard = limiter.acquire_memory_blocking(file_size).await;

            // Should succeed and track the memory
            assert_eq!(limiter.memory_limiter.used(), file_size);

            // Drop the guard to release memory
            drop(guard);
            assert_eq!(limiter.memory_limiter.used(), 0);
        }

        // Test case 2: file_size exceeds capacity - should skip memory limiting
        {
            let config = KvEngineConfig {
                dfs_load_concurrency_per_core: 64,
                dfs_load_memory_ratio: 0.25,
                ..Default::default()
            };
            let mut opts = Options::default();
            opts.dfs_load_memory_limit = 1024 * 1024; // 1 MB capacity

            let limiter = DfsLoadLimiter::new(&config, &opts);

            // Try to acquire 2 MB (exceeds capacity)
            let file_size = 2 * 1024 * 1024;
            let guard = limiter.acquire_memory_blocking(file_size).await;

            // Should succeed but not track the memory (returns 0-size guard)
            // Memory usage should be 0 because we skipped memory limiting
            assert_eq!(limiter.memory_limiter.used(), 0);

            // Drop the guard - should still be 0
            drop(guard);
            assert_eq!(limiter.memory_limiter.used(), 0);
        }

        // Test case 3: Multiple concurrent acquisitions within capacity
        {
            let config = KvEngineConfig {
                dfs_load_concurrency_per_core: 64,
                dfs_load_memory_ratio: 0.25,
                ..Default::default()
            };
            let mut opts = Options::default();
            opts.dfs_load_memory_limit = 1024 * 1024; // 1 MB capacity

            let limiter = DfsLoadLimiter::new(&config, &opts);

            // Acquire 400 KB three times in sequence (total 1.2 MB would exceed)
            let file_size = 400 * 1024;

            let guard1 = limiter.acquire_memory_blocking(file_size).await;
            assert_eq!(limiter.memory_limiter.used(), file_size);

            let guard2 = limiter.acquire_memory_blocking(file_size).await;
            assert_eq!(limiter.memory_limiter.used(), file_size * 2);

            // Release first guard
            drop(guard1);
            assert_eq!(limiter.memory_limiter.used(), file_size);

            // Now we can acquire another one
            let guard3 = limiter.acquire_memory_blocking(file_size).await;
            assert_eq!(limiter.memory_limiter.used(), file_size * 2);

            drop(guard2);
            drop(guard3);
            assert_eq!(limiter.memory_limiter.used(), 0);
        }

        // Test case 4: Verify wait duration metric is recorded when blocking occurs
        {
            let config = KvEngineConfig {
                dfs_load_concurrency_per_core: 64,
                dfs_load_memory_ratio: 0.25,
                ..Default::default()
            };
            let mut opts = Options::default();
            opts.dfs_load_memory_limit = 500 * 1024; // 500 KB capacity

            let limiter = DfsLoadLimiter::new(&config, &opts);

            // Get the initial metric count
            let initial_count =
                crate::metrics::ENGINE_DFS_LOAD_MEMORY_WAIT_DURATION.get_sample_count();

            // Acquire 400 KB - should succeed immediately (no wait)
            let guard1 = limiter.acquire_memory_blocking(400 * 1024).await;
            assert_eq!(limiter.memory_limiter.used(), 400 * 1024);

            // Spawn a task that will try to acquire 300 KB (total would be 700 KB > 500 KB)
            // This should block and wait
            let limiter_clone = limiter.clone();
            let handle =
                tokio::spawn(
                    async move { limiter_clone.acquire_memory_blocking(300 * 1024).await },
                );

            // Give the task time to start and begin waiting
            tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;

            // Release the first guard to allow the second task to proceed
            drop(guard1);

            // Wait for the second task to complete
            let guard2 = handle.await.unwrap();
            assert_eq!(limiter.memory_limiter.used(), 300 * 1024);

            // Verify the wait duration metric was recorded
            let final_count =
                crate::metrics::ENGINE_DFS_LOAD_MEMORY_WAIT_DURATION.get_sample_count();
            assert!(
                final_count > initial_count,
                "Wait duration metric should be recorded when blocking occurs. Initial: {}, Final: {}",
                initial_count,
                final_count
            );

            drop(guard2);
            assert_eq!(limiter.memory_limiter.used(), 0);
        }
    }
}
