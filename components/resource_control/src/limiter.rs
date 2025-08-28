// Copyright 2025 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    ops::{Add, Deref, Sub},
    sync::{
        atomic::{AtomicBool, Ordering::Relaxed},
        Arc,
    },
    time::{Duration, Instant},
};

use dashmap::DashMap;
use tikv_util::{error, info};

use crate::{AtomicDuration, AtomicTime, Config, TimeUnit, ACTIVE_KEYSPACE_READ_BYTES};

const MIN_WAIT_TIME_INTERVAL: Duration = Duration::from_millis(1);

pub const MAX_WAIT_TIME: Duration = Duration::from_millis(50);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[allow(dead_code)]
pub enum Action {
    None,
    ScaleOut,
    Throttle,
    Reject,
}

#[derive(Clone)]
pub struct ReadLimiter {
    pub(crate) core: Arc<ReadLimiterCore>,
}

pub struct ReadLimiterCore {
    pub(crate) enabled: AtomicBool,
    pub(crate) timeout: AtomicDuration,
    pub(crate) stats_interval: AtomicDuration,
    pub(crate) max_wait_time: AtomicDuration,
    pub(crate) keyspace_limiters: DashMap<u32, (Instant, KeyspaceReadLimiter)>,
}

impl Deref for ReadLimiter {
    type Target = ReadLimiterCore;

    fn deref(&self) -> &Self::Target {
        &self.core
    }
}

impl ReadLimiterCore {
    pub fn new(config: Config) -> Self {
        let timeout = config.limiter_timeout.0;
        let stats_interval = config.limiter_stats_interval.0;
        Self {
            enabled: AtomicBool::from(config.enabled),
            timeout: AtomicDuration::new(timeout, TimeUnit::Millisecond),
            stats_interval: AtomicDuration::new(stats_interval, TimeUnit::Millisecond),
            max_wait_time: AtomicDuration::new(MAX_WAIT_TIME, TimeUnit::Millisecond),
            keyspace_limiters: DashMap::new(),
        }
    }
}

impl ReadLimiter {
    pub fn new(config: Config) -> Self {
        Self {
            core: Arc::new(ReadLimiterCore::new(config)),
        }
    }

    pub(crate) fn update_enabled(&self, enabled: bool) {
        self.enabled.store(enabled, Relaxed);
        if !enabled {
            self.clear_all_limiter();
        }
    }

    pub(crate) fn get_enabled(&self) -> bool {
        self.enabled.load(Relaxed)
    }

    pub(crate) fn update_timeout(&self, timeout: Duration) {
        self.timeout.store(timeout);
    }

    pub(crate) fn update_stats_interval(&self, stats_interval: Duration) {
        self.stats_interval.store(stats_interval);
    }

    pub(crate) fn update_max_wait_time(&self, max_wait_time: Duration) {
        self.max_wait_time.store(max_wait_time);
    }

    pub(crate) fn clear_all_limiter(&self) {
        self.keyspace_limiters.iter().for_each(|keyspace_limiter_ref|{
            let keyspace_id = *keyspace_limiter_ref.key();
            let keyspace_label = keyspace_id.to_string();
            let keyspace_str = keyspace_label.as_str();
            let _ =  ACTIVE_KEYSPACE_READ_BYTES.remove_label_values(&[keyspace_str]).map_err(
                |err| error!("failed to remove active keyspace read bytes metric"; "keyspace_id" => keyspace_str, "err" => %err),
            );
        });
        self.keyspace_limiters.clear();
    }

    pub(crate) fn update_limit(
        &self,
        keyspace_id: u32,
        req_speed_limit: Option<f64>,
        bytes_speed_limit: Option<f64>,
        instant: Instant,
    ) {
        if !self.get_enabled() {
            return;
        }
        self.keyspace_limiters
            .entry(keyspace_id)
            .and_modify(|(ts, keyspace_read_limiter)| {
                *ts = instant;
                keyspace_read_limiter.set_speed_limit(req_speed_limit, bytes_speed_limit);
                keyspace_read_limiter.update_stats_interval(self.stats_interval.load());
                keyspace_read_limiter.update_max_wait_time(self.max_wait_time.load());
            })
            .or_insert_with(|| {
                let keyspace_read_limiter = KeyspaceReadLimiter::new(
                    keyspace_id,
                    self.stats_interval.load(),
                    self.max_wait_time.load(),
                );
                keyspace_read_limiter.set_speed_limit(req_speed_limit, bytes_speed_limit);
                (instant, keyspace_read_limiter)
            });
    }

    pub fn get_limiter(&self, keyspace_id: u32) -> Option<KeyspaceReadLimiter> {
        if !self.get_enabled() {
            return None;
        }
        let timeout = self.timeout.load();
        if let Some((ts, keyspace_read_limiter)) =
            self.keyspace_limiters.get(&keyspace_id).map(|v| v.clone())
        {
            if ts.elapsed() < timeout {
                Some(keyspace_read_limiter.clone())
            } else {
                self.remove_limiter(keyspace_id);
                None
            }
        } else {
            None
        }
    }

    fn remove_limiter(&self, keyspace_id: u32) {
        self.keyspace_limiters.remove(&keyspace_id);
        let keyspace_label = keyspace_id.to_string();
        let keyspace_str = keyspace_label.as_str();
        let _ =  ACTIVE_KEYSPACE_READ_BYTES.remove_label_values(&[keyspace_str]).map_err(
            |err| error!("failed to remove active keyspace read bytes metric"; "keyspace_id" => keyspace_str, "err" => %err),
        );
    }
}

#[derive(Clone)]
pub struct KeyspaceReadLimiter {
    core: Arc<KeyspaceReadLimiterCore>,
}

pub struct KeyspaceReadLimiterCore {
    keyspace_id: u32,
    req_limiter: tikv_util::time::Limiter,
    bytes_limiter: tikv_util::time::Limiter,
    allowed_time: AtomicTime, // Requests after the allowed time do not need to wait.
    last_time: AtomicTime,
    stats_interval: AtomicDuration,
    max_wait_time: AtomicDuration,
}

impl Deref for KeyspaceReadLimiter {
    type Target = KeyspaceReadLimiterCore;

    fn deref(&self) -> &Self::Target {
        &self.core
    }
}

impl KeyspaceReadLimiterCore {
    pub fn new(keyspace_id: u32, stats_interval: Duration, max_wait_time: Duration) -> Self {
        let req_limiter = <tikv_util::time::Limiter>::builder(f64::INFINITY).build();
        let bytes_limiter = <tikv_util::time::Limiter>::builder(f64::INFINITY).build();
        let start = Instant::now();
        Self {
            keyspace_id,
            req_limiter,
            bytes_limiter,
            allowed_time: AtomicTime::new(start, TimeUnit::Microsecond),
            last_time: AtomicTime::new(start, TimeUnit::Millisecond),
            stats_interval: AtomicDuration::new(stats_interval, TimeUnit::Millisecond),
            max_wait_time: AtomicDuration::new(max_wait_time, TimeUnit::Millisecond),
        }
    }
}

impl Default for KeyspaceReadLimiter {
    fn default() -> Self {
        Self::new(0, Duration::from_secs(1), MAX_WAIT_TIME)
    }
}

impl KeyspaceReadLimiter {
    pub fn new(keyspace_id: u32, stats_interval: Duration, max_wait_time: Duration) -> Self {
        Self {
            core: Arc::new(KeyspaceReadLimiterCore::new(
                keyspace_id,
                stats_interval,
                max_wait_time,
            )),
        }
    }

    pub(crate) fn update_allowed_time<F>(&self, update: F) -> Duration
    where
        F: Fn(Instant) -> (Option<Instant>, Duration),
    {
        let mut allowed_time = self.allowed_time.load();
        loop {
            let (new_allowed_time, dur) = update(allowed_time);
            let Some(new_allowed_time) = new_allowed_time else {
                return dur;
            };
            match self
                .allowed_time
                .compare_exchange(allowed_time, new_allowed_time)
            {
                Some(current) => {
                    allowed_time = current;
                }
                None => {
                    return dur;
                }
            }
        }
    }

    pub fn take_wait_time(&self) -> Duration {
        let update = |allowed_time: Instant| {
            let now = Instant::now();
            let dur = allowed_time.duration_since(now);
            if dur < MIN_WAIT_TIME_INTERVAL {
                return (None, dur);
            }
            let new_allowed_time = now;
            (Some(new_allowed_time), dur)
        };
        self.update_allowed_time(update)
    }

    pub fn wait_time(&self) -> Duration {
        let update = |allowed_time: Instant| {
            let now = Instant::now();
            let dur = allowed_time.duration_since(now);
            if dur < MIN_WAIT_TIME_INTERVAL {
                return (None, dur);
            };
            let new_allowed_time = allowed_time.sub(MIN_WAIT_TIME_INTERVAL);
            (Some(new_allowed_time), MIN_WAIT_TIME_INTERVAL)
        };
        self.update_allowed_time(update)
    }

    pub async fn wait(&self) -> Duration {
        let mut wait_time = Duration::default();
        loop {
            let dur = self.wait_time();
            if dur.is_zero() {
                break;
            }
            tokio::time::sleep(dur).await;
            wait_time = wait_time.add(dur);
            if wait_time >= self.max_wait_time.load() {
                break;
            }
        }
        wait_time
    }

    pub fn consume(&self, bytes: usize) {
        let dur = self
            .req_limiter
            .consume_duration(1)
            .max(self.bytes_limiter.consume_duration(bytes));

        let update = |allowed_time: Instant| {
            let now = Instant::now();
            let new_allowed_time = if allowed_time.duration_since(now).is_zero() {
                now.add(dur)
            } else {
                allowed_time.add(dur)
            };
            (Some(new_allowed_time), dur)
        };
        self.update_allowed_time(update);
    }

    pub fn unconsume(&self, bytes: usize) {
        self.req_limiter.unconsume(1);
        self.bytes_limiter.unconsume(bytes);
    }

    pub fn is_unlimited(&self) -> bool {
        self.req_limiter.speed_limit() == f64::INFINITY
            && self.bytes_limiter.speed_limit() == f64::INFINITY
    }

    pub fn total_consumed(&self) -> (usize /* req */, usize /* bytes */) {
        (
            self.req_limiter.total_bytes_consumed(),
            self.bytes_limiter.total_bytes_consumed(),
        )
    }

    pub fn speed_limit(&self) -> (f64 /* req */, f64 /* bytes */) {
        (
            self.req_limiter.speed_limit(),
            self.bytes_limiter.speed_limit(),
        )
    }

    pub fn set_speed_limit(&self, req_speed_limit: Option<f64>, bytes_speed_limit: Option<f64>) {
        if let Some(req_speed_limit) = req_speed_limit {
            if req_speed_limit > 0.0 {
                self.req_limiter.set_speed_limit(req_speed_limit);
            } else {
                self.req_limiter.set_speed_limit(f64::INFINITY);
            }
        }
        if let Some(bytes_speed_limit) = bytes_speed_limit {
            if bytes_speed_limit > 0.0 {
                self.bytes_limiter.set_speed_limit(bytes_speed_limit);
            } else {
                self.bytes_limiter.set_speed_limit(f64::INFINITY);
            }
        }
        self.update_statistics();
    }

    fn update_statistics(&self) {
        let last_time = self.last_time.load();
        let dur = last_time.elapsed();
        if dur < self.stats_interval.load() {
            return;
        }
        self.last_time.store(Instant::now());
        let total_requests = self.req_limiter.total_bytes_consumed() as f64;
        let total_bytes = self.bytes_limiter.total_bytes_consumed() as f64;
        self.req_limiter.reset_statistics();
        self.bytes_limiter.reset_statistics();
        if total_requests == 0.0 {
            return;
        }
        let qps = total_requests / dur.as_secs_f64();
        let bytes_rate = total_bytes / dur.as_secs_f64();
        let bytes_per_req = total_bytes / total_requests;
        info!("resource control update_statistics dur {:?}", dur;
            "keyspace_id" => self.keyspace_id,
            "total_requests" => total_requests,
            "total_bytes" => total_bytes,
            "qps" => qps,
            "bytes_rate" => bytes_rate,
            "bytes_per_req" => bytes_per_req,
        );
    }

    fn update_stats_interval(&self, stats_interval: Duration) {
        self.stats_interval.store(stats_interval);
    }

    fn update_max_wait_time(&self, max_wait_time: Duration) {
        self.max_wait_time.store(max_wait_time);
    }
}
