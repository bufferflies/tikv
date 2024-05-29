// Copyright 2024 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    collections::{HashMap, VecDeque},
    error::Error,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use rand::Rng;
use tikv_util::{
    config::ReadableDuration,
    info,
    sys::{get_global_memory_usage, SysQuota},
    time::InstantExt,
};

#[derive(Clone, Serialize, Deserialize, PartialEq, Debug)]
#[serde(default)]
#[serde(rename_all = "kebab-case")]
pub struct CopLimiterConfig {
    pub high_mem_ratio: f64,
    pub wait_min_duration: ReadableDuration,
    pub wait_max_duration: ReadableDuration,
    pub sample_ttl: ReadableDuration,
}

impl Default for CopLimiterConfig {
    fn default() -> Self {
        Self {
            high_mem_ratio: 0.8,
            wait_min_duration: ReadableDuration::millis(100),
            wait_max_duration: ReadableDuration::secs(3),
            sample_ttl: ReadableDuration::secs(60),
        }
    }
}

const DEFAULT_WAIT_TIMEOUT: Duration = Duration::from_secs(20);

impl CopLimiterConfig {
    pub fn validate(&self) -> Result<(), Box<dyn Error>> {
        if self.high_mem_ratio <= 0.0 || self.high_mem_ratio >= 1.0 {
            return Err("high_mem_ratio must be in (0, 1)".into());
        }
        if self.wait_min_duration >= self.wait_max_duration {
            return Err("wait_min_duration must be less than wait_max_duration".into());
        }
        Ok(())
    }
}

#[derive(Clone)]
pub(crate) struct CopLimiter {
    core: Arc<Mutex<CopLimiterCore>>,
    high_mem_usage: u64,
    wait_min_millis: u64,
    wait_max_millis: u64,
    sample_ttl: Duration,
}

impl CopLimiter {
    pub(crate) fn new(cfg: CopLimiterConfig) -> Self {
        let mem_limit = SysQuota::memory_limit_in_bytes();
        let high_mem_usage = (mem_limit as f64 * cfg.high_mem_ratio) as u64;
        assert!(cfg.wait_min_duration < cfg.wait_max_duration);
        Self {
            core: Arc::new(Mutex::new(CopLimiterCore {
                samples: VecDeque::new(),
                keyspace_throughputs: HashMap::new(),
                total_throughput: 0,
            })),
            high_mem_usage,
            wait_min_millis: cfg.wait_min_duration.as_millis(),
            wait_max_millis: cfg.wait_max_duration.as_millis(),
            sample_ttl: cfg.sample_ttl.0,
        }
    }
}

struct CopLimiterCore {
    samples: VecDeque<CopStatSample>,
    keyspace_throughputs: HashMap<u32, u64>,
    total_throughput: u64,
}

#[derive(Copy, Clone)]
struct CopStatSample {
    keyspace_id: u32,
    resp_size: u64,
    finished_at: Instant,
}

impl CopLimiter {
    pub(crate) fn add_sample(&self, keyspace_id: u32, resp_size: u64, finished_at: Instant) {
        let cop_stat_sample = CopStatSample {
            keyspace_id,
            resp_size,
            finished_at,
        };
        let mut core = self.core.lock().unwrap();
        core.samples.push_back(cop_stat_sample);
        let entry = core
            .keyspace_throughputs
            .entry(cop_stat_sample.keyspace_id)
            .or_insert(0);
        *entry += cop_stat_sample.resp_size;
        core.total_throughput += cop_stat_sample.resp_size;
        while let Some(&front) = core.samples.front() {
            let sample_elapsed = finished_at.saturating_duration_since(front.finished_at);
            if sample_elapsed > self.sample_ttl {
                let entry = core
                    .keyspace_throughputs
                    .get_mut(&front.keyspace_id)
                    .unwrap();
                *entry -= front.resp_size;
                if *entry == 0 {
                    core.keyspace_throughputs.remove(&front.keyspace_id);
                }
                core.total_throughput -= front.resp_size;
                core.samples.pop_front();
            } else {
                break;
            }
        }
    }

    fn is_high_throughput(&self, keyspace_id: u32) -> bool {
        let core = self.core.lock().unwrap();
        if core.keyspace_throughputs.is_empty() {
            return false;
        }
        let throughput = core
            .keyspace_throughputs
            .get(&keyspace_id)
            .cloned()
            .unwrap_or(0);
        let avg_throughput = core.total_throughput / core.keyspace_throughputs.len() as u64;
        throughput >= avg_throughput
    }

    pub(crate) async fn wait_for_high_mem_usage(
        &self,
        keyspace_id: u32,
        tag: &str,
        mut timeout: Duration,
    ) -> bool {
        let usage = get_global_memory_usage();
        if usage < self.high_mem_usage {
            return true;
        }
        if !self.is_high_throughput(keyspace_id) {
            return true;
        }
        if timeout.is_zero() {
            timeout = DEFAULT_WAIT_TIMEOUT;
        }
        let start = Instant::now();
        while start.saturating_elapsed() < timeout {
            // random sleep to avoid thundering herd
            let wait_millis =
                rand::thread_rng().gen_range(self.wait_min_millis..self.wait_max_millis);
            let wait_dur = Duration::from_millis(wait_millis);
            info!(
                "wait for memory high usage";
                "tag" => tag,
                "wait_dur" => ?wait_dur,
                "mem_usage" => usage,
            );
            tokio::time::sleep(wait_dur).await;
            if get_global_memory_usage() < self.high_mem_usage {
                return true;
            }
        }
        info!(
            "wait memory high usage timeout";
            "tag" => tag,
        );
        false
    }
}

#[cfg(test)]
mod tests {
    use test_util::init_log_for_test;
    use tikv_util::sys::set_memory_usage_for_test;

    use super::*;
    #[test]
    fn test_cop_limiter() {
        init_log_for_test();
        let config = CopLimiterConfig {
            high_mem_ratio: 0.8,
            wait_min_duration: ReadableDuration::millis(50),
            wait_max_duration: ReadableDuration::millis(200),
            sample_ttl: ReadableDuration::secs(1),
        };
        let cop_limiter = CopLimiter::new(config);
        let high_mem_usage = cop_limiter.high_mem_usage;
        set_memory_usage_for_test(high_mem_usage);
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let tag = "ks:1:1:1:1";
        let pass =
            runtime.block_on(cop_limiter.wait_for_high_mem_usage(1, tag, Duration::from_secs(1)));
        // pass because there is no samples.
        assert!(pass);

        cop_limiter.add_sample(1, 100, Instant::now());
        cop_limiter.add_sample(2, 200, Instant::now());
        let pass_1 =
            runtime.block_on(cop_limiter.wait_for_high_mem_usage(1, tag, Duration::from_secs(1)));
        // pass because the throughput is below average.
        assert!(pass_1);
        let pass_2 =
            runtime.block_on(cop_limiter.wait_for_high_mem_usage(2, tag, Duration::from_secs(1)));
        // not pass because the throughput is above average.
        assert!(!pass_2);

        cop_limiter.add_sample(1, 50, Instant::now());
        {
            let core = cop_limiter.core.lock().unwrap();
            // the first two samples are expired, so there is only the last sample.
            assert_eq!(core.samples.len(), 1);
            assert_eq!(core.keyspace_throughputs.len(), 1);
            assert_eq!(core.total_throughput, 50);
        }
        let pass = runtime.block_on(cop_limiter.wait_for_high_mem_usage(
            1,
            tag,
            Duration::from_millis(500),
        ));
        // not pass because only a single keyspace, throughput equal to average.
        assert!(!pass);
        runtime.spawn(async move {
            tokio::time::sleep(Duration::from_millis(200)).await;
            set_memory_usage_for_test(high_mem_usage / 2);
        });
        let pass = runtime.block_on(cop_limiter.wait_for_high_mem_usage(
            1,
            tag,
            Duration::from_millis(500),
        ));
        // the memory usage decreased, so the request should pass.
        assert!(pass);
    }
}
