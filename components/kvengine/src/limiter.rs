// Copyright 2024 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    marker::PhantomData,
    sync::{Arc, Mutex},
    time::Duration,
};

use prometheus::IntGauge;
use tikv_util::time::{Instant, Limiter};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use crate::{metrics::ENGINE_THROTTLE_ACTION_COUNTER, KvEngineConfig, ShardTag};

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
}

impl DfsLoadLimiter {
    pub(crate) fn new(cfg: &KvEngineConfig) -> DfsLoadLimiter {
        let num_cores = tikv_util::sys::SysQuota::cpu_cores_quota();
        let global_concurrency = (num_cores.max(1.0) as usize) * cfg.dfs_load_concurrency_per_core;
        let semaphore = Arc::new(Semaphore::new(global_concurrency));
        Self { semaphore }
    }

    pub(crate) async fn acquire_permit(&self) -> OwnedSemaphorePermit {
        let global_semaphore = self.semaphore.clone();
        // We never close the semaphore, so it is safe to unwrap.
        global_semaphore.acquire_owned().await.unwrap()
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
}
