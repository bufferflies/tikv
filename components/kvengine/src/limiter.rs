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

/// Default ratio applied to `hard_limit` to define the bursting threshold
/// (i.e. `hard_limit * (1.0 + DEFAULT_BURST_RATIO)`).
const DEFAULT_BURST_RATIO: f64 = 0.1;

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

    /// Returns the hard resource limit in bytes.
    ///
    /// Returns the configured hard limit when enabled, or `0` when disabled.
    pub fn resource_max_limit(&self) -> u64 {
        if self.enabled() {
            self.options.hard_limit
        } else {
            0
        }
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

    pub fn update_usage(&self, tag: &ShardTag, usage: u64, forcibly_throttling: impl Fn() -> bool) {
        if !self.options.enable {
            return;
        }

        let throttle = Self::calculate_throttle(
            usage,
            self.options.soft_limit,
            self.options.hard_limit,
            self.options.max_speed_limit,
            self.options.min_speed_limit,
            forcibly_throttling,
        );
        debug!("{} ShardLimiter::update_usage", tag;
            "usage" => usage,
            "throttle" => throttle,
        );
        self.update_speed_limit(tag, throttle);
    }

    /// Calculate throttle according to current usage using a smooth exponential
    /// curve with bursting mechanism.
    ///
    /// All the parameters are in bytes or bytes/s.
    ///
    /// `usage` is the current usage of component (e.g. memtables) to be
    /// throttled.
    ///
    /// The throttling algorithm has four zones:
    /// 1. Below soft_limit: no throttling (INFINITY)
    /// 2. Between soft_limit and hard_limit: exponential throttling curve
    /// 3. Between hard_limit and hard_limit * 1.1: bursting mechanism for
    ///    smoother throttling
    /// 4. Above hard_limit * 1.1: minimum speed limit (hard cutoff)
    pub(crate) fn calculate_throttle(
        usage: u64,
        soft_limit: u64,
        hard_limit: u64,
        max_speed_limit: u64,
        min_speed_limit: u64,
        forcibly_throttling: impl Fn() -> bool,
    ) -> f64 {
        debug_assert!(hard_limit >= soft_limit);
        debug_assert!(max_speed_limit >= min_speed_limit);

        // Early exit: if soft_limit is 0, no throttling
        if soft_limit == 0 {
            return f64::INFINITY;
        }

        // Calculate the bursting threshold (hard_limit plus a burst ratio),
        // using saturating_add to avoid potential overflow when converting
        // from floating point back to u64.
        let burst_extra = (hard_limit as f64 * DEFAULT_BURST_RATIO) as u64;
        let bursting_threshold = hard_limit.saturating_add(burst_extra);

        if usage < soft_limit {
            // Zone 1: No throttling below soft_limit
            f64::INFINITY
        } else if usage < hard_limit {
            // Zone 2: Exponential throttling curve between soft_limit and hard_limit
            ENGINE_THROTTLE_ACTION_COUNTER
                .with_label_values(&[Lv::TAG, "smoothing"])
                .inc();

            let range = (hard_limit - soft_limit) as f64;
            let position = (usage - soft_limit) as f64;
            let normalized = position / range.max(1.0);

            // Exponential curve: e^(-k*x) where k controls the curve steepness
            // Using k=2.0 provides a good balance between smoothness and responsiveness
            let exp_factor = (-normalized * 2.0).exp();

            // Interpolate from max_speed_limit to min_speed_limit using exponential curve
            let throttle = min_speed_limit as f64
                + (max_speed_limit as f64 - min_speed_limit as f64) * exp_factor;
            throttle.max(min_speed_limit as f64)
        } else if usage < bursting_threshold && !forcibly_throttling() {
            // Zone 3: Bursting mechanism between hard_limit and hard_limit * 1.1
            // This provides a smoother transition from the exponential curve result
            // at hard_limit down to min_speed_limit at bursting_threshold
            ENGINE_THROTTLE_ACTION_COUNTER
                .with_label_values(&[Lv::TAG, "bursting"])
                .inc();

            // Calculate the throttle value at hard_limit (from the exponential curve)
            let throttle_at_hard_limit = min_speed_limit as f64
                + (max_speed_limit as f64 - min_speed_limit as f64) * (-2.0f64).exp(); // exp(-2.0) when normalized = 1.0

            // Calculate the excess beyond hard_limit
            let excess = (usage - hard_limit) as f64;
            let bursting_range = (bursting_threshold - hard_limit) as f64;
            let bursting_ratio = excess / bursting_range.max(1.0);

            // Use a smooth sigmoid-like curve for the bursting mechanism
            // This provides a gradual deceleration from throttle_at_hard_limit to
            // min_speed_limit using the standard smoothstep (Hermite) interpolation:
            // S(t) = 3t^2 - 2t^3
            let t = bursting_ratio.min(1.0);
            let smooth_factor = 3.0 * t * t - 2.0 * t * t * t;

            // Interpolate smoothly from throttle_at_hard_limit to min_speed_limit
            let throttle = throttle_at_hard_limit
                - (throttle_at_hard_limit - min_speed_limit as f64) * smooth_factor;
            throttle.max(min_speed_limit as f64)
        } else {
            // Zone 4: Hard cutoff at bursting_threshold, use minimum speed limit
            ENGINE_THROTTLE_ACTION_COUNTER
                .with_label_values(&[Lv::TAG, "hard_cutoff"])
                .inc();

            min_speed_limit as f64
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
        let soft_limit = 256 << 20; // 256 MB
        let hard_limit = 768 << 20; // 768 MB
        let max_speed_limit = 50 << 20; // 50 MB/s
        let min_speed_limit = 1 << 20; // 1 MB/s
        let bursting_threshold = (hard_limit as f64 * 1.1) as u64; // 110% of hard_limit
        let forcibly_throttling = || false;

        // Test 1: Below soft_limit - should be INFINITY
        let result_below = RegionLimiter::calculate_throttle(
            0,
            soft_limit,
            hard_limit,
            max_speed_limit,
            min_speed_limit,
            forcibly_throttling,
        );
        assert_eq!(
            result_below,
            f64::INFINITY,
            "Below soft_limit should be INFINITY"
        );

        let result_just_below_soft = RegionLimiter::calculate_throttle(
            soft_limit - 1,
            soft_limit,
            hard_limit,
            max_speed_limit,
            min_speed_limit,
            forcibly_throttling,
        );
        assert_eq!(
            result_just_below_soft,
            f64::INFINITY,
            "Just below soft_limit should be INFINITY"
        );

        // Test 2: At soft_limit - should start throttling (not INFINITY)
        let result_at_soft = RegionLimiter::calculate_throttle(
            soft_limit,
            soft_limit,
            hard_limit,
            max_speed_limit,
            min_speed_limit,
            forcibly_throttling,
        );
        assert!(
            result_at_soft.is_finite()
                && result_at_soft <= max_speed_limit as f64
                && result_at_soft >= min_speed_limit as f64,
            "At soft_limit should start throttling between min and max: {}",
            result_at_soft
        );

        // Test 3: Between soft_limit and hard_limit - should use exponential curve
        let result_mid = RegionLimiter::calculate_throttle(
            (soft_limit + hard_limit) / 2,
            soft_limit,
            hard_limit,
            max_speed_limit,
            min_speed_limit,
            forcibly_throttling,
        );
        assert!(
            result_mid.is_finite()
                && result_mid < result_at_soft
                && result_mid >= min_speed_limit as f64,
            "Between soft_limit and hard_limit should throttle: {}",
            result_mid
        );

        // Test 4: At hard_limit - should be in bursting zone
        let result_at_hard = RegionLimiter::calculate_throttle(
            hard_limit,
            soft_limit,
            hard_limit,
            max_speed_limit,
            min_speed_limit,
            forcibly_throttling,
        );
        assert!(
            result_at_hard >= min_speed_limit as f64,
            "At hard_limit should be at least min_speed_limit: {}",
            result_at_hard
        );

        // Test 5: In bursting zone (between hard_limit and hard_limit * 1.1)
        let result_in_bursting = RegionLimiter::calculate_throttle(
            hard_limit + (bursting_threshold - hard_limit) / 2,
            soft_limit,
            hard_limit,
            max_speed_limit,
            min_speed_limit,
            forcibly_throttling,
        );
        assert!(
            result_in_bursting >= min_speed_limit as f64 && result_in_bursting <= result_at_hard,
            "In bursting zone should throttle smoothly: {}",
            result_in_bursting
        );

        // Test 6: At bursting threshold (hard_limit * 1.1) - should be min_speed_limit
        let result_at_bursting_threshold = RegionLimiter::calculate_throttle(
            bursting_threshold,
            soft_limit,
            hard_limit,
            max_speed_limit,
            min_speed_limit,
            forcibly_throttling,
        );
        assert_eq!(
            result_at_bursting_threshold, min_speed_limit as f64,
            "At bursting threshold should be min_speed_limit: {}",
            result_at_bursting_threshold
        );

        // Test 7: Above bursting threshold - should be min_speed_limit (hard cutoff)
        let result_above_bursting = RegionLimiter::calculate_throttle(
            bursting_threshold + (100 << 20), // 100 MB above bursting threshold
            soft_limit,
            hard_limit,
            max_speed_limit,
            min_speed_limit,
            forcibly_throttling,
        );
        assert_eq!(
            result_above_bursting, min_speed_limit as f64,
            "Above bursting threshold should be min_speed_limit: {}",
            result_above_bursting
        );

        // Test 8: Monotonicity - throttle should decrease as usage increases
        let usages = vec![
            soft_limit - 1,
            soft_limit,
            (soft_limit + hard_limit) / 2,
            hard_limit,
            hard_limit + (bursting_threshold - hard_limit) / 2,
            bursting_threshold,
            bursting_threshold + (100 << 20),
        ];
        let mut prev_throttle = f64::INFINITY;
        for usage in usages {
            let throttle = RegionLimiter::calculate_throttle(
                usage,
                soft_limit,
                hard_limit,
                max_speed_limit,
                min_speed_limit,
                forcibly_throttling,
            );
            assert!(
                throttle <= prev_throttle || (prev_throttle.is_infinite() && throttle.is_finite()),
                "Throttle should be monotonic decreasing: usage={}, prev={}, current={}",
                usage,
                prev_throttle,
                throttle
            );
            prev_throttle = throttle;
        }

        // Test 9: Corner case - soft_limit == hard_limit
        let result_equal = RegionLimiter::calculate_throttle(
            512 << 20,
            512 << 20,
            512 << 20,
            50 << 20,
            1 << 20,
            forcibly_throttling,
        );
        assert!(
            result_equal >= min_speed_limit as f64 && result_equal <= max_speed_limit as f64,
            "When soft_limit == hard_limit, should be between min and max: {}",
            result_equal
        );

        // Test 10: Corner case - soft_limit == 0 (should return INFINITY)
        let result_zero_soft = RegionLimiter::calculate_throttle(
            100 << 20,
            0,
            hard_limit,
            max_speed_limit,
            min_speed_limit,
            forcibly_throttling,
        );
        assert_eq!(
            result_zero_soft,
            f64::INFINITY,
            "When soft_limit is 0, should return INFINITY"
        );

        // Test 11: Forcibly throttling in bursting zone - should bypass bursting and
        // use min_speed_limit
        let forcibly_throttling_true = || true;
        let usage_in_bursting_zone = hard_limit + (bursting_threshold - hard_limit) / 2;

        let result_bursting_forced = RegionLimiter::calculate_throttle(
            usage_in_bursting_zone,
            soft_limit,
            hard_limit,
            max_speed_limit,
            min_speed_limit,
            forcibly_throttling_true,
        );
        assert_eq!(
            result_bursting_forced, min_speed_limit as f64,
            "When forcibly_throttling in bursting zone, should bypass bursting and use min_speed_limit: {}",
            result_bursting_forced
        );

        // Test 12: Forcibly throttling above hard_limit - should bypass all zones and
        // use min_speed_limit
        let usage_above_hard = hard_limit + (50 << 20);

        let result_above_hard_forced = RegionLimiter::calculate_throttle(
            usage_above_hard,
            soft_limit,
            hard_limit,
            max_speed_limit,
            min_speed_limit,
            forcibly_throttling_true,
        );
        assert_eq!(
            result_above_hard_forced, min_speed_limit as f64,
            "When forcibly_throttling above hard_limit, should use min_speed_limit immediately: {}",
            result_above_hard_forced
        );

        // Test 13: Forcibly throttling between soft and hard limits - should use
        // expotential curve.
        let usage_mid_zone = (soft_limit + hard_limit) / 2;

        let result_mid_forced = RegionLimiter::calculate_throttle(
            usage_mid_zone,
            soft_limit,
            hard_limit,
            max_speed_limit,
            min_speed_limit,
            forcibly_throttling_true,
        );

        assert!(
            result_mid_forced.is_finite() && result_mid_forced >= min_speed_limit as f64,
            "When forcibly_throttling between soft and hard limits, should use expotential curve {}",
            result_mid_forced
        );

        // Test 14: Verify forcibly_throttling bypasses all calculations
        // Even at extreme usage levels, forcibly_throttling should result in
        // min_speed_limit
        let extreme_usage = hard_limit * 2;

        let result_extreme_forced = RegionLimiter::calculate_throttle(
            extreme_usage,
            soft_limit,
            hard_limit,
            max_speed_limit,
            min_speed_limit,
            forcibly_throttling_true,
        );
        assert_eq!(
            result_extreme_forced, min_speed_limit as f64,
            "When forcibly_throttling at extreme usage, should use min_speed_limit: {}",
            result_extreme_forced
        );

        // Test 15: Compare forcibly throttling vs normal throttling at same usage level
        // This verifies the difference in behavior
        let comparison_usage = hard_limit + (bursting_threshold - hard_limit) / 3;

        let result_normal = RegionLimiter::calculate_throttle(
            comparison_usage,
            soft_limit,
            hard_limit,
            max_speed_limit,
            min_speed_limit,
            forcibly_throttling, // false
        );
        let result_forced = StoreLimiter::calculate_throttle(
            comparison_usage,
            soft_limit,
            hard_limit,
            max_speed_limit,
            min_speed_limit,
            forcibly_throttling_true,
        );

        assert!(
            result_forced < result_normal,
            "Forcibly throttling should result in stricter throttling than normal: forced={}, normal={}",
            result_forced,
            result_normal
        );
        assert_eq!(
            result_forced, min_speed_limit as f64,
            "Forcibly throttling should equal min_speed_limit: {}",
            result_forced
        );
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
