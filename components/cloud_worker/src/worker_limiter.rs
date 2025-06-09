// Copyright 2024 TiKV Project Authors. Licensed under Apache-2.0.

use std::{error::Error, sync::Arc};

use dashmap::DashMap;
use tikv_util::sys::SysQuota;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

#[derive(Clone, Serialize, Deserialize, PartialEq, Debug)]
#[serde(default)]
#[serde(rename_all = "kebab-case")]
pub struct WorkerLimiterConfig {
    pub global_concurrency_factor: f64,
    pub keyspace_concurrency_factor: f64,
}

impl Default for WorkerLimiterConfig {
    fn default() -> Self {
        Self {
            global_concurrency_factor: 96.0,
            keyspace_concurrency_factor: 32.0,
        }
    }
}

impl WorkerLimiterConfig {
    pub fn validate(&self) -> Result<(), Box<dyn Error>> {
        if self.global_concurrency_factor > 256.0 {
            return Err("global concurrency factor must be less than 256".into());
        }
        if self.global_concurrency_factor < self.keyspace_concurrency_factor {
            return Err(
                "global concurrency factor must be greater than keyspace concurrency_factor".into(),
            );
        }
        Ok(())
    }
}

pub(crate) struct Permit {
    _global_permit: OwnedSemaphorePermit,
    _keyspace_permit: OwnedSemaphorePermit,
}

#[derive(Clone)]
pub(crate) struct WorkerLimiter {
    global_semaphore: Arc<Semaphore>,
    keyspace_semaphores: Arc<DashMap<u32, Arc<Semaphore>>>,
    cfg: Arc<WorkerLimiterConfig>,
}

const MIN_CONCURRENCY: usize = 2;

impl WorkerLimiter {
    pub(crate) fn new(cfg: WorkerLimiterConfig) -> Self {
        cfg.validate().unwrap();
        let cpu_cores = SysQuota::cpu_cores_quota();
        let global_concurrency =
            MIN_CONCURRENCY.max((cpu_cores * cfg.global_concurrency_factor) as usize);
        let global_semaphore = Arc::new(Semaphore::new(global_concurrency));
        Self {
            global_semaphore,
            keyspace_semaphores: Arc::new(DashMap::new()),
            cfg: Arc::new(cfg),
        }
    }
}

impl WorkerLimiter {
    pub(crate) async fn acquire_permit(&self, keyspace_id: u32) -> Permit {
        let semaphore = self
            .keyspace_semaphores
            .entry(keyspace_id)
            .or_insert_with(|| {
                let keyspace_concurrency = MIN_CONCURRENCY.max(
                    (SysQuota::cpu_cores_quota() * self.cfg.keyspace_concurrency_factor) as usize,
                );
                Arc::new(Semaphore::new(keyspace_concurrency))
            })
            .clone();
        let _keyspace_permit = semaphore.acquire_owned().await.unwrap();
        let global_semaphore = self.global_semaphore.clone();
        let _global_permit = global_semaphore.acquire_owned().await.unwrap();
        Permit {
            _global_permit,
            _keyspace_permit,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{
        collections::HashMap,
        sync::{Arc, Mutex},
        time::Duration,
    };

    use tikv_util::sys::SysQuota;

    use crate::worker_limiter::{WorkerLimiter, WorkerLimiterConfig};

    #[derive(Default)]
    struct ConcurrencyCounter {
        running: usize,
        max_running: usize,
    }

    #[test]
    fn test_worker_limiter_concurrency() {
        let config = WorkerLimiterConfig::default();
        let worker_limiter = WorkerLimiter::new(config);
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .worker_threads(4)
            .build()
            .unwrap();
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
                let _permit = worker_limiter.acquire_permit(keyspace_id).await;
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
