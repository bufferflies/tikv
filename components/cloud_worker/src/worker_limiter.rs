// Copyright 2024 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    error::Error,
    sync::{
        atomic::{AtomicI64, Ordering},
        Arc,
    },
    time::Duration,
};

use dashmap::DashMap;
use tikv_util::{config::ReadableDuration, defer, sys::SysQuota, time::Instant};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use crate::metrics::{
    WORKER_LIMITER_REQUEST_WAIT_HISTOGRAM, WORKER_LIMITER_WAITING_REQUESTS_COUNTER_VEC,
};

#[derive(Clone, Serialize, Deserialize, PartialEq, Debug)]
#[serde(default)]
#[serde(rename_all = "kebab-case")]
pub struct CoprocessorLimiterConfig {
    pub global_concurrency_factor: f64,
    pub keyspace_concurrency_factor: f64,
}

impl Default for CoprocessorLimiterConfig {
    fn default() -> Self {
        Self {
            global_concurrency_factor: 6.0,
            keyspace_concurrency_factor: 2.0,
        }
    }
}

impl CoprocessorLimiterConfig {
    pub fn validate(&self) -> Result<(), Box<dyn Error>> {
        if self.global_concurrency_factor > 10.0 {
            return Err("global concurrency factor must be less than 10".into());
        }
        if self.global_concurrency_factor < self.keyspace_concurrency_factor {
            return Err(
                "global concurrency factor must be greater than keyspace concurrency_factor".into(),
            );
        }
        Ok(())
    }
}

#[derive(Clone, Serialize, Deserialize, PartialEq, Debug)]
#[serde(default)]
#[serde(rename_all = "kebab-case")]
#[serde(deny_unknown_fields)]
pub struct CompactionLimiterConfig {
    pub global_concurrency_factor: f64,
    pub max_queue_size: usize,
    pub wait_timeout: ReadableDuration,
}

impl Default for CompactionLimiterConfig {
    fn default() -> Self {
        Self {
            global_concurrency_factor: 6.0,
            max_queue_size: 32,
            wait_timeout: ReadableDuration::secs(5),
        }
    }
}

impl CompactionLimiterConfig {
    pub fn validate(&self) -> Result<(), Box<dyn Error>> {
        if self.global_concurrency_factor > 10.0 {
            return Err("global concurrency factor must be less than 10".into());
        }
        if self.max_queue_size == 0 {
            return Err("max queue size must be greater than 0".into());
        }
        Ok(())
    }
}

/// The type of worker, used for metrics and configuration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WorkerType {
    Coprocessor,
    Compaction,
}

impl WorkerType {
    /// Get the string representation of the worker type
    pub fn as_str(&self) -> &'static str {
        match self {
            WorkerType::Coprocessor => "coprocessor",
            WorkerType::Compaction => "compaction",
        }
    }
}

pub struct Permit {
    _global_permit: OwnedSemaphorePermit,
    _keyspace_permit: OwnedSemaphorePermit,
}

#[derive(Clone)]
pub(crate) struct WorkerLimiter {
    keyspace_concurrency_factor: f64,
    wait_timeout: Duration,
    max_queue_size: u64,

    global_semaphore: Arc<Semaphore>,
    keyspace_semaphores: Arc<DashMap<u32, Arc<Semaphore>>>,
    worker_type: WorkerType,
    waiting_count: Arc<AtomicI64>,
}

const MIN_CONCURRENCY: usize = 2;

impl WorkerLimiter {
    pub fn new(
        global_concurrency_factor: f64,
        keyspace_concurrency_factor: f64,
        wait_timeout: Duration,
        max_queue_size: u64,
        worker_type: WorkerType,
    ) -> Self {
        let cpu_cores = SysQuota::cpu_cores_quota();
        let global_concurrency =
            MIN_CONCURRENCY.max((cpu_cores * global_concurrency_factor) as usize);
        let global_semaphore = Arc::new(Semaphore::new(global_concurrency));

        Self {
            keyspace_concurrency_factor,
            wait_timeout,
            max_queue_size,
            global_semaphore,
            keyspace_semaphores: Arc::new(DashMap::new()),
            worker_type,
            waiting_count: Arc::new(AtomicI64::new(0)),
        }
    }
}

impl WorkerLimiter {
    /// Acquire a permit for the given keyspace id.
    /// Returns None if the waiting requests over the max queue size.
    pub(crate) async fn acquire_permit(&self, keyspace_id: u32) -> Option<Permit> {
        let wait_start = Instant::now_coarse();

        // Overflow check.
        debug_assert!(self.max_queue_size <= i64::MAX as u64);
        // Try to increment waiting count atomically if it's below max_queue_size
        if self
            .waiting_count
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |current| {
                if current >= self.max_queue_size as i64 {
                    None
                } else {
                    Some(current + 1)
                }
            })
            .is_err()
        {
            return None;
        }

        defer!({
            self.waiting_count.fetch_sub(1, Ordering::SeqCst);
        });

        // Update metrics
        WORKER_LIMITER_WAITING_REQUESTS_COUNTER_VEC
            .with_label_values(&[self.worker_type.as_str()])
            .inc();
        defer!({
            WORKER_LIMITER_WAITING_REQUESTS_COUNTER_VEC
                .with_label_values(&[self.worker_type.as_str()])
                .dec();
        });

        let semaphore = self
            .keyspace_semaphores
            .entry(keyspace_id)
            .or_insert_with(|| {
                let keyspace_concurrency = MIN_CONCURRENCY
                    .max((SysQuota::cpu_cores_quota() * self.keyspace_concurrency_factor) as usize);
                Arc::new(Semaphore::new(keyspace_concurrency))
            })
            .clone();
        let _keyspace_permit = semaphore.acquire_owned().await.unwrap();
        let global_semaphore = self.global_semaphore.clone();
        let _global_permit = global_semaphore.acquire_owned().await.unwrap();
        WORKER_LIMITER_REQUEST_WAIT_HISTOGRAM
            .with_label_values(&[self.worker_type.as_str()])
            .observe(wait_start.saturating_elapsed().as_secs_f64());
        Some(Permit {
            _global_permit,
            _keyspace_permit,
        })
    }

    pub fn wait_timeout(&self) -> Duration {
        self.wait_timeout
    }
}

#[cfg(test)]
mod tests {
    use std::{
        collections::HashMap,
        sync::{atomic::Ordering, Arc, Mutex},
        time::Duration,
    };

    use tikv_util::sys::SysQuota;

    use crate::worker_limiter::{CompactionLimiterConfig, CoprocessorLimiterConfig, WorkerType};

    #[derive(Default)]
    struct ConcurrencyCounter {
        running: usize,
        max_running: usize,
        max_waiting: usize,
    }

    #[test]
    fn test_coprocessor_limiter_concurrency() {
        let coprocessor_config = CoprocessorLimiterConfig::default();
        let coprocessor_limiter = super::WorkerLimiter::new(
            coprocessor_config.global_concurrency_factor,
            coprocessor_config.keyspace_concurrency_factor,
            Duration::from_secs(u64::MAX),
            i64::MAX as u64,
            WorkerType::Coprocessor,
        );

        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .worker_threads(4)
            .build()
            .unwrap();

        // Test compressor limiter
        test_limiter_concurrency(&coprocessor_limiter, &runtime);
    }

    #[test]
    fn test_compaction_limiter_concurrency() {
        let compaction_config = CompactionLimiterConfig::default();
        let compaction_limiter = super::WorkerLimiter::new(
            compaction_config.global_concurrency_factor,
            1_000_000.0,
            compaction_config.wait_timeout.0,
            compaction_config.max_queue_size as u64,
            WorkerType::Compaction,
        );

        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .worker_threads(4)
            .build()
            .unwrap();

        // Test compaction limiter
        test_limiter_concurrency(&compaction_limiter, &runtime);
    }

    fn test_limiter_concurrency(
        worker_limiter: &super::WorkerLimiter,
        runtime: &tokio::runtime::Runtime,
    ) {
        let global_counter = Arc::new(Mutex::new(ConcurrencyCounter::default()));
        let keyspace_counters = Arc::new(Mutex::new(HashMap::new()));
        let cpu_cores = SysQuota::cpu_cores_quota() as u32;
        let mut handles = vec![];
        for i in 0..(100 * cpu_cores) {
            let global_counter = global_counter.clone();
            let keyspace_counters = keyspace_counters.clone();
            let worker_limiter = worker_limiter.clone();
            let handle = runtime.spawn(async move {
                let keyspace_id = i % 4;
                let _permit = if let Some(permit) = worker_limiter.acquire_permit(keyspace_id).await
                {
                    permit
                } else {
                    // Record waiting count when permit acquisition fails
                    let waiting = worker_limiter.waiting_count.load(Ordering::SeqCst);
                    {
                        let mut guard = global_counter.lock().unwrap();
                        guard.max_waiting = guard.max_waiting.max(waiting as usize);
                    }
                    return;
                };
                {
                    let mut guard = global_counter.lock().unwrap();
                    guard.running += 1;
                    guard.max_running = guard.max_running.max(guard.running);
                }
                {
                    let mut guard = keyspace_counters.lock().unwrap();
                    let ks_counter = guard
                        .entry(keyspace_id)
                        .or_insert_with(|| ConcurrencyCounter::default());
                    ks_counter.running += 1;
                    ks_counter.max_running = ks_counter.max_running.max(ks_counter.running);
                }
                tokio::time::sleep(Duration::from_millis(50)).await;
                {
                    let mut guard = keyspace_counters.lock().unwrap();
                    let ks_counter = guard.get_mut(&keyspace_id).unwrap();
                    ks_counter.running -= 1;
                }
                {
                    let mut guard = global_counter.lock().unwrap();
                    guard.running -= 1;
                }
            });
            handles.push(handle);
        }
        for handle in handles {
            runtime.block_on(handle).unwrap();
        }
        let counter_guard = global_counter.lock().unwrap();
        assert_eq!(counter_guard.running, 0);
        assert_eq!(
            counter_guard.max_running,
            worker_limiter.global_semaphore.available_permits()
        );
        // Assert that max waiting count never exceeds max_queue_size
        assert!(
            counter_guard.max_waiting <= worker_limiter.max_queue_size as usize,
            "max waiting count {} exceeds max queue size {}",
            counter_guard.max_waiting,
            worker_limiter.max_queue_size
        );
        let ks_counter_guard = keyspace_counters.lock().unwrap();
        for (keyspace_id, counter) in ks_counter_guard.iter() {
            assert_eq!(counter.running, 0);
            assert!(
                counter.max_running
                    <= worker_limiter
                        .keyspace_semaphores
                        .get(keyspace_id)
                        .unwrap()
                        .available_permits()
            );
        }
    }
}
