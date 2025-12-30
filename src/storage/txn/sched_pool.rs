// Copyright 2019 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    cell::RefCell,
    mem,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Mutex,
    },
};

use collections::HashMap;
use file_system::{set_io_type, IoType};
use kvproto::pdpb::QueryKind;
use pd_client::{Feature, FeatureGate};
use prometheus::local::*;
use raftstore::store::WriteStats;
use tikv_util::{
    sys::{thread::ThreadBuildWrapper, SysQuota},
    yatp_pool::{
        self, FuturePool, PoolTicker, ScalableTokioHandle, ScalableTokioRuntime, YatpPoolBuilder,
    },
};

use crate::storage::{
    kv::{destroy_tls_engine, set_tls_engine, Engine, FlowStatsReporter, Statistics},
    metrics::{SCHED_POOL_RUNNING_TASKS_GAUGE, *},
    test_util::latest_feature_gate,
};

pub struct SchedLocalMetrics {
    local_scan_details: HashMap<&'static str, Statistics>,
    command_keyread_histogram_vec: LocalHistogramVec,
    local_write_stats: WriteStats,
}

thread_local! {
    static TLS_SCHED_METRICS: RefCell<SchedLocalMetrics> = RefCell::new(
        SchedLocalMetrics {
            local_scan_details: HashMap::default(),
            command_keyread_histogram_vec: KV_COMMAND_KEYREAD_HISTOGRAM_VEC.local(),
            local_write_stats:WriteStats::default(),
        }
    );

    static TLS_FEATURE_GATE: RefCell<FeatureGate> = RefCell::new(latest_feature_gate());
}

#[derive(Clone)]
pub enum SchedPool {
    Yatp {
        pool: FuturePool,
    },
    Tokio {
        handle: ScalableTokioHandle,
        task_monitor: tokio_metrics::TaskMonitor,
    },
}

#[derive(Clone)]
pub struct SchedTicker<R: FlowStatsReporter> {
    reporter: R,
}

impl<R: FlowStatsReporter> PoolTicker for SchedTicker<R> {
    fn on_tick(&mut self) {
        tls_flush(&self.reporter);
    }
}

impl SchedPool {
    pub fn new_yatp<E: Engine, R: FlowStatsReporter>(
        engine: E,
        pool_size: usize,
        reporter: R,
        feature_gate: FeatureGate,
        name_prefix: &str,
    ) -> Self {
        let engine = Arc::new(Mutex::new(engine));
        // for low cpu quota env, set the max-thread-count as 4 to allow potential cases
        // that we need more thread than cpu num.
        let max_pool_size = std::cmp::max(
            pool_size,
            std::cmp::max(4, SysQuota::cpu_cores_quota() as usize),
        );
        let pool = YatpPoolBuilder::new(SchedTicker {reporter:reporter.clone()})
            .thread_count(1, pool_size, max_pool_size)
            .name_prefix(name_prefix)
            // Safety: by setting `after_start` and `before_stop`, `FuturePool` ensures
            // the tls_engine invariants.
            .after_start(move || {
                set_tls_engine(engine.lock().unwrap().clone());
                set_io_type(IoType::ForegroundWrite);
                TLS_FEATURE_GATE.with(|c| *c.borrow_mut() = feature_gate.clone());
            })
            .before_stop(move || unsafe {
                // Safety: we ensure the `set_` and `destroy_` calls use the same engine type.
                destroy_tls_engine::<E>();
                tls_flush(&reporter);
            })
            .build_future_pool();
        SchedPool::Yatp { pool }
    }

    /// Build a Tokio-based scheduler pool.
    ///
    /// Returns the pool plus the owned runtime so callers can shut it down
    /// explicitly (e.g., during server shutdown) while cloned pools keep only a
    /// handle.
    pub fn new_tokio<E: Engine>(
        engine: E,
        pool_size: usize,
        feature_gate: FeatureGate,
        name_prefix: &str,
    ) -> (Self, ScalableTokioRuntime) {
        let engine = Arc::new(Mutex::new(engine));
        let thread_name_prefix = name_prefix.to_string();
        let props = tikv_util::thread_group::current_properties();
        let max_pool_size = std::cmp::max(1, SysQuota::cpu_cores_quota().ceil() as usize);
        assert!(max_pool_size >= pool_size);
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .thread_name_fn(move || {
                static ATOMIC_ID: AtomicUsize = AtomicUsize::new(0);
                let id = ATOMIC_ID.fetch_add(1, Ordering::SeqCst);
                format!("{}-{}", thread_name_prefix, id)
            })
            .worker_threads(max_pool_size)
            .after_start_wrapper(move || {
                let engine = engine.lock().unwrap().clone();
                set_tls_engine(engine);
                set_io_type(IoType::ForegroundWrite);
                TLS_FEATURE_GATE.with(|c| *c.borrow_mut() = feature_gate.clone());
                tikv_util::thread_group::set_properties(props.clone());
            })
            .before_stop_wrapper(|| unsafe {
                destroy_tls_engine::<E>();
            })
            .enable_all()
            .build()
            .unwrap();

        let scalable_runtime = ScalableTokioRuntime::new(runtime, max_pool_size, pool_size);
        let handle = scalable_runtime.handle().unwrap();
        // Create TaskMonitor for schedule wait time metrics
        let task_monitor = tokio_metrics::TaskMonitor::new();

        // Spawn background task to collect and export metrics
        let metrics_task_monitor = task_monitor.clone();
        handle.spawn(async move {
            use std::time::Duration;
            let mut intervals = metrics_task_monitor.intervals();
            loop {
                tokio::time::sleep(Duration::from_secs(1)).await;
                if let Some(metrics) = intervals.next() {
                    SCHED_TOKIO_POOL_MEAN_FIRST_POLL_DELAY
                        .set(metrics.mean_first_poll_delay().as_secs_f64());
                    SCHED_TOKIO_POOL_FIRST_POLL_COUNT.inc_by(metrics.first_poll_count);
                    SCHED_TOKIO_POOL_MEAN_IDLE_DURATION
                        .set(metrics.mean_idle_duration().as_secs_f64());
                    SCHED_TOKIO_POOL_MEAN_SCHEDULED_DURATION
                        .set(metrics.mean_scheduled_duration().as_secs_f64());
                    SCHED_TOKIO_POOL_MEAN_POLL_DURATION
                        .set(metrics.mean_poll_duration().as_secs_f64());
                }
            }
        });

        (
            SchedPool::Tokio {
                handle,
                task_monitor,
            },
            scalable_runtime,
        )
    }

    pub fn spawn<F>(&self, future: F) -> Result<(), yatp_pool::Full>
    where
        F: std::future::Future<Output = ()> + Send + 'static,
    {
        // Wrap future to track running tasks
        let tracked_future = async move {
            SCHED_POOL_RUNNING_TASKS_GAUGE.inc();
            future.await;
            SCHED_POOL_RUNNING_TASKS_GAUGE.dec();
        };

        // TODO: queue depth based backpressure
        // We want consistent behavior as the merged pool.
        // We rely solely on the running_write_bytes based backpressure.
        match self {
            SchedPool::Yatp { pool } => pool.spawn(tracked_future),
            SchedPool::Tokio {
                handle,
                task_monitor,
            } => {
                let future = async move { tikv_util::init_task_local(tracked_future).await };
                // Instrument with tokio-metrics to track schedule wait time
                let instrumented = task_monitor.instrument(future);
                handle.spawn(instrumented);
                Ok(())
            }
        }
    }
}

pub fn tls_collect_scan_details(cmd: &'static str, stats: &Statistics) {
    TLS_SCHED_METRICS.with(|m| {
        m.borrow_mut()
            .local_scan_details
            .entry(cmd)
            .or_default()
            .add(stats);
    });
}

pub fn tls_flush<R: FlowStatsReporter>(reporter: &R) {
    TLS_SCHED_METRICS.with(|m| {
        let mut m = m.borrow_mut();
        for (cmd, stat) in m.local_scan_details.drain() {
            for (cf, cf_details) in stat.details().iter() {
                for (tag, count) in cf_details.iter() {
                    KV_COMMAND_SCAN_DETAILS
                        .with_label_values(&[cmd, *cf, *tag])
                        .inc_by(*count as u64);
                }
            }
        }
        m.command_keyread_histogram_vec.flush();

        // Report PD metrics
        if !m.local_write_stats.is_empty() {
            let mut write_stats = WriteStats::default();
            mem::swap(&mut write_stats, &mut m.local_write_stats);
            reporter.report_write_stats(write_stats);
        }
    });
}

pub fn tls_collect_query(region_id: u64, kind: QueryKind) {
    TLS_SCHED_METRICS.with(|m| {
        let mut m = m.borrow_mut();
        m.local_write_stats.add_query_num(region_id, kind);
    });
}

pub fn tls_collect_keyread_histogram_vec(cmd: &str, count: f64) {
    TLS_SCHED_METRICS.with(|m| {
        m.borrow_mut()
            .command_keyread_histogram_vec
            .with_label_values(&[cmd])
            .observe(count);
    });
}

pub fn tls_can_enable(feature: Feature) -> bool {
    TLS_FEATURE_GATE.with(|feature_gate| feature_gate.borrow().can_enable(feature))
}

#[cfg(test)]
pub fn set_tls_feature_gate(feature_gate: FeatureGate) {
    TLS_FEATURE_GATE.with(|f| *f.borrow_mut() = feature_gate);
}
