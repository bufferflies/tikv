// Copyright 2022 TiKV Project Authors. Licensed under Apache-2.0.
use std::{
    collections::BTreeMap,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::Duration,
};

use crossbeam::atomic::AtomicCell;
use engine_rocks::{ReadPerfContext, ReadPerfInstant};
use fail::fail_point;
use kvengine::{
    table::{DataBound, InnerKey},
    SnapAccess, TableRef, WRITE_CF,
};
use kvproto::metapb::{Region, RegionEpoch};
use raftstore::{coprocessor::ObserveHandle, store::fsm::ChangeObserver};
use rfstore::{
    router::CdcHandle,
    store::{
        msg::{Callback, ReadResponse},
        RegionSnapshot,
    },
};
use tikv::storage::{
    kv::Snapshot,
    mvcc::CloudReader,
    txn::{CloudDeltaScanner, TxnEntry, TxnEntryScanner},
};
use tikv_util::{
    box_err, debug, defer, error, info,
    sys::inspector::{self_thread_inspector, ThreadInspector},
    time::{Instant, Limiter},
    warn,
    worker::Scheduler,
};
use tokio::sync::Semaphore;
use txn_types::{Key, LockType, OldValue, TimeStamp};

use crate::{
    channel::CdcEvent,
    delegate::{
        post_init_downstream, Delegate, DownstreamId, DownstreamState, MiniLock, ObservedRange,
    },
    endpoint::Deregister,
    metrics::*,
    service::{ConnId, RequestId},
    Error, Result, Task,
};

#[derive(Copy, Clone, Debug, Default)]
pub(crate) struct ScanStat {
    // Fetched bytes to the scanner.
    emit: usize,
    // Bytes from the device, `None` if not possible to get it.
    disk_read: Option<usize>,
    // Perf delta for RocksDB.
    perf_delta: ReadPerfContext,
}

pub(crate) struct Initializer {
    pub(crate) region_id: u64,
    pub(crate) conn_id: ConnId,
    pub(crate) request_id: RequestId,
    pub(crate) checkpoint_ts: TimeStamp,
    pub(crate) region_epoch: RegionEpoch,

    // `build_resolver` can only be determined after snapshot is acquired.
    // If a region is subscribed more than one times, the downstream with the
    // earliest snapshot will build the lock resolver.
    //
    // `build_resolver` won't be changed after set in `InitDownstream`.
    pub(crate) build_resolver: Arc<AtomicBool>,

    pub(crate) observed_range: ObservedRange,
    pub(crate) observe_handle: ObserveHandle,
    pub(crate) downstream_id: DownstreamId,
    pub(crate) downstream_state: Arc<AtomicCell<DownstreamState>>,
    pub(crate) scan_truncated: Arc<AtomicBool>,

    pub(crate) sched: Scheduler<Task>,
    pub(crate) sink: crate::channel::Sink,
    pub(crate) concurrency_semaphore: Arc<Semaphore>,

    pub(crate) scan_speed_limiter: Limiter,
    pub(crate) fetch_speed_limiter: Limiter,

    pub(crate) max_scan_batch_bytes: usize,
    pub(crate) max_scan_batch_size: usize,

    #[allow(dead_code)]
    pub(crate) ts_filter_ratio: f64,
    pub(crate) filter_loop: bool,
}

impl Initializer {
    pub(crate) async fn initialize<T>(&mut self, cdc_handle: T) -> Result<()>
    where
        T: 'static + CdcHandle,
    {
        fail_point!("cdc_before_initialize");
        let concurrency_semaphore = self.concurrency_semaphore.clone();
        let _permit = concurrency_semaphore.acquire().await;

        let region_id = self.region_id;
        let downstream_id = self.downstream_id;
        let observe_id = self.observe_handle.id;
        // when there are a lot of pending incremental scan tasks, they may be stopped,
        // check the state here to accelerate tasks cancel process.
        if self.downstream_state.load() == DownstreamState::Stopped {
            info!("cdc async incremental scan canceled before start";
                "downstream_id" => ?downstream_id,
                "request_id" => ?self.request_id,
                "region_id" => region_id,
                "conn_id" => ?self.conn_id);
            return Err(Error::Other(box_err!("scan canceled")));
        }

        // To avoid holding too many snapshots and holding them too long,
        // we need to acquire scan concurrency permit before taking snapshot.
        let sched = self.sched.clone();
        let region_epoch = self.region_epoch.clone();
        let downstream_state = self.downstream_state.clone();
        let (cb, fut) = tikv_util::future::paired_future_callback();
        let sink = self.sink.clone();
        let build_resolver = self.build_resolver.clone();
        let (incremental_scan_barrier_cb, incremental_scan_barrier_fut) =
            tikv_util::future::paired_future_callback();
        let barrier = CdcEvent::Barrier(Some(incremental_scan_barrier_cb));
        if let Err(e) = cdc_handle.capture_change(
            self.region_id,
            region_epoch,
            ChangeObserver::from_cdc(self.region_id, self.observe_handle.clone()),
            // NOTE: raftstore handles requests in serial for every region.
            // That's why we can determine whether to build a lock resolver or not
            // without check and compare snapshot sequence number.
            Callback::Read(Box::new(move |resp| {
                if let Err(e) = sched.schedule(Task::InitDownstream {
                    region_id,
                    observe_id,
                    downstream_id,
                    downstream_state,
                    sink,
                    build_resolver,
                    incremental_scan_barrier: barrier,
                    cb: Box::new(move || cb(resp)),
                }) {
                    error!("cdc schedule cdc task failed"; "error" => ?e);
                }
            })),
        ) {
            warn!("cdc send capture change cmd failed";
            "region_id" => self.region_id, "error" => ?e);
            return Err(Error::request(e.into()));
        }

        // Wait all delta changes earlier than the incremental scan snapshot be
        // sent to the downstream, so that they must be consumed before the
        // incremental scan result.
        if let Err(e) = incremental_scan_barrier_fut.await {
            return Err(Error::Other(box_err!(e)));
        }

        match fut.await {
            Ok(resp) => self.on_change_cmd_response(resp).await,
            Err(e) => Err(Error::Other(box_err!(e))),
        }
    }

    pub(crate) async fn on_change_cmd_response(&mut self, mut resp: ReadResponse) -> Result<()> {
        if let Some(region_snapshot) = resp.snapshot {
            let region = region_snapshot.region.as_deref().unwrap().clone();
            assert_eq!(self.region_id, region.get_id());
            self.async_incremental_scan(region_snapshot, region)
                .await
                .map(|_| ())
        } else {
            assert!(
                resp.response.get_header().has_error(),
                "no snapshot and no error? {:?}",
                resp.response
            );
            let err = resp.response.take_header().take_error();
            Err(Error::request(err))
        }
    }

    pub(crate) async fn async_incremental_scan(
        &mut self,
        snap: RegionSnapshot,
        region: Region,
    ) -> Result<ScanStat> {
        CDC_SCAN_TASKS.with_label_values(&["ongoing"]).inc();
        defer!(CDC_SCAN_TASKS.with_label_values(&["ongoing"]).dec());

        let region_id = self.region_id;
        let request_id = self.request_id;
        let downstream_id = self.downstream_id;
        let conn_id = self.conn_id;
        let on_cancel = || -> Result<ScanStat> {
            info!(
                "cdc async incremental scan canceled";
                "downstream_id" => ?downstream_id,
                "request_id" => ?request_id,
                "region_id" => region_id,
                "conn_id" =>?conn_id,
            );
            Err(box_err!("scan canceled"))
        };

        if self.downstream_state.load() == DownstreamState::Stopped {
            return on_cancel();
        }

        self.observed_range.update_region_key_range(&region);

        // Be compatible with old TiCDC clients, which won't give `observed_range`.
        let (start_key, end_key): (Key, Key);
        if self.observed_range.start_key_encoded.as_encoded() <= &region.start_key {
            start_key = Key::from_encoded_slice(&region.start_key);
        } else {
            start_key = self.observed_range.start_key_encoded.clone();
        }
        if self.observed_range.end_key_encoded.is_empty()
            || self.observed_range.end_key_encoded.as_encoded() >= &region.end_key
                && !region.end_key.is_empty()
        {
            end_key = Key::from_encoded_slice(&region.end_key);
        } else {
            end_key = self.observed_range.end_key_encoded.clone();
        }

        debug!(
            "cdc async incremental scan";
            "end_key" => log_wrappers::Value::key(end_key.as_encoded()),
            "start_key" => log_wrappers::Value::key(start_key.as_encoded()),
            "all_key_covered" => ?self.observed_range.all_key_covered,
            "downstream_id" => ?downstream_id,
            "request_id" => ?request_id,
            "region_id" => region_id,
            "conn_id" => ?conn_id,
        );

        if self.build_resolver.load(Ordering::Acquire) {
            // Scan and collect locks if build_resolver is true. The range
            // should be the whole region span instead of subscribed span,
            // because those locks will be shared between multiple Downstreams.
            let mut reader = CloudReader::new(snap.get_kvengine_snap().unwrap().clone(), false);
            let (key_locks, has_remain) = reader.scan_locks(None, None, |_| true, 0)?;
            assert!(!has_remain);
            let mut locks = BTreeMap::<Key, MiniLock>::new();
            for (key, lock) in key_locks {
                // When `decode_lock`, only consider `Put` and `Delete`
                if matches!(lock.lock_type, LockType::Put | LockType::Delete) {
                    let mini_lock = MiniLock::new(lock.ts, lock.txn_source);
                    locks.insert(key, mini_lock);
                }
            }
            self.finish_scan_locks(region, locks);
        };
        let filter_by_ts = self.ts_filter_is_helpful(&snap.snap, &start_key, &end_key);

        let upper_boundary = if end_key.as_encoded().is_empty() {
            // Region upper boundary could be an empty slice.
            None
        } else {
            Some(end_key)
        };
        // Time range: (checkpoint_ts, max]
        let mut scanner = CloudDeltaScanner::new(
            snap.snap.clone(),
            self.checkpoint_ts.into_inner(),
            Some(start_key),
            upper_boundary,
            filter_by_ts,
        );

        fail_point!("cdc_incremental_scan_start");
        let mut done = false;
        let start = Instant::now_coarse();
        let mut sink_time = Duration::default();
        let curr_state = self.downstream_state.load();
        assert!(matches!(
            curr_state,
            DownstreamState::Initializing | DownstreamState::Stopped
        ));

        let mut scan_stat = ScanStat::default();
        let scan_long_time = AtomicBool::new(false);
        defer!(if scan_long_time.load(Ordering::SeqCst) {
            CDC_SCAN_LONG_DURATION_REGIONS.dec();
        });

        let mut total_scanned_entries = 0;
        while !done {
            // Add metrics to observe long time incremental scan region count
            if !scan_long_time.load(Ordering::SeqCst)
                && start.saturating_elapsed() > Duration::from_secs(60)
            {
                CDC_SCAN_LONG_DURATION_REGIONS.inc();
                scan_long_time.store(true, Ordering::SeqCst);
                warn!(
                    "cdc incremental scan takes too long";
                    "scanned_bytes" => scan_stat.emit, "scanned_entries" => total_scanned_entries,
                    "sink_takes" => ?sink_time, "takes" => ?start.saturating_elapsed(),
                    "downstream_id" => ?self.downstream_id, "request_id" => ?request_id,
                    "region_id" => region_id, "conn_id" => ?self.conn_id,
                );
            }
            // When downstream_state is Stopped, it means the corresponding
            // delegate is stopped. The initialization can be safely canceled.
            if self.downstream_state.load() == DownstreamState::Stopped {
                return on_cancel();
            }
            let entries = self.scan_batch(&mut scanner, &mut scan_stat).await?;
            if let Some(None) = entries.last() {
                // If the last element is None, it means scanning is finished.
                done = true;
            }
            total_scanned_entries += entries.len();
            debug!("cdc scan entries"; "len" => entries.len(), "region_id" => region_id);
            fail_point!("before_schedule_incremental_scan");
            let start_sink = Instant::now_coarse();
            self.sink_scan_events(entries, done).await?;
            sink_time += start_sink.saturating_elapsed();
        }

        fail_point!("before_post_incremental_scan");
        if !post_init_downstream(&self.downstream_state) {
            return on_cancel();
        }
        let takes = start.saturating_elapsed();
        info!("cdc async incremental scan finished";
            "scanned_bytes" => scan_stat.emit, "scanned_entries" => total_scanned_entries,
            "sink_takes" => ?sink_time, "takes" => ?takes,
            "downstream_id" => ?downstream_id, "request_id" => ?self.request_id,
            "region_id" => region_id, "conn_id" => ?conn_id);

        CDC_SCAN_DURATION_HISTOGRAM.observe(takes.as_secs_f64());
        CDC_SCAN_SINK_DURATION_HISTOGRAM.observe(sink_time.as_secs_f64());
        Ok(scan_stat)
    }

    // It's extracted from `Initializer::scan_batch` to avoid becoming an
    // asynchronous block, so that we can limit scan speed based on the thread
    // disk I/O or RocksDB block read bytes.
    fn do_scan(
        &self,
        scanner: &mut CloudDeltaScanner,
        entries: &mut Vec<Option<TxnEntry>>,
    ) -> Result<ScanStat> {
        // This code block shouldn't be switched to other threads.
        let mut total_bytes = 0;
        let mut total_size = 0;
        let perf_instant = ReadPerfInstant::new();
        let inspector = self_thread_inspector().ok();
        let old_io_stat = inspector.as_ref().and_then(|x| x.io_stat().unwrap_or(None));
        while total_bytes <= self.max_scan_batch_bytes && total_size < self.max_scan_batch_size {
            total_size += 1;
            match scanner.next_entry()? {
                Some(mut entry) => {
                    assert!(*entry.old_value() != OldValue::Unspecified);
                    total_bytes += entry.size();
                    entries.push(Some(entry));
                }
                None => {
                    entries.push(None);
                    break;
                }
            }
        }

        let new_io_stat = inspector.as_ref().and_then(|x| x.io_stat().unwrap_or(None));
        let disk_read = match (old_io_stat, new_io_stat) {
            (Some(s1), Some(s2)) => Some((s2.read - s1.read) as usize),
            _ => None,
        };
        let perf_delta = perf_instant.delta();
        let emit = total_bytes;
        Ok(ScanStat {
            emit,
            disk_read,
            perf_delta,
        })
    }

    async fn scan_batch(
        &self,
        scanner: &mut CloudDeltaScanner,
        scan_stat: &mut ScanStat,
    ) -> Result<Vec<Option<TxnEntry>>> {
        let mut entries = Vec::with_capacity(self.max_scan_batch_size);
        let delta = self.do_scan(scanner, &mut entries)?;
        scan_stat.emit += delta.emit;
        scan_stat.perf_delta += delta.perf_delta;
        if let Some(disk_read) = delta.disk_read {
            *scan_stat.disk_read.get_or_insert(0) += disk_read;
        }

        TLS_CDC_PERF_STATS.with(|x| *x.borrow_mut() += delta.perf_delta);
        tls_flush_perf_stats();
        if let Some(bytes) = delta.disk_read {
            CDC_SCAN_DISK_READ_BYTES.inc_by(bytes as _);
            self.scan_speed_limiter.consume(bytes).await;
        }
        CDC_SCAN_BYTES.inc_by(delta.emit as _);
        self.fetch_speed_limiter.consume(delta.emit as _).await;

        Ok(entries)
    }

    async fn sink_scan_events(&mut self, entries: Vec<Option<TxnEntry>>, done: bool) -> Result<()> {
        let mut barrier = None;
        let mut events = Delegate::convert_to_grpc_events(
            self.region_id,
            self.request_id,
            entries,
            self.filter_loop,
            &self.observed_range,
        )?;
        if done {
            let (cb, fut) = tikv_util::future::paired_future_callback();
            events.push(CdcEvent::Barrier(Some(cb)));
            barrier = Some(fut);
        }
        if let Err(e) = self
            .sink
            .send_all(events, self.scan_truncated.clone())
            .await
        {
            warn!("cdc send scan event failed"; "err" => ?e, "downstream_id" => ?self.downstream_id,
                "request_id" => ?self.request_id, "region_id" => self.region_id, "conn_id" => ?self.conn_id);
            return Err(Error::Sink(e));
        }

        if let Some(barrier) = barrier {
            // CDC needs to make sure resolved ts events can only be sent after
            // incremental scan is finished.
            // Wait the barrier to ensure tikv sends out all scan events.
            let _ = barrier.await;
        }

        Ok(())
    }

    fn finish_scan_locks(&self, region: Region, locks: BTreeMap<Key, MiniLock>) {
        let observe_id = self.observe_handle.id;
        info!(
            "cdc has scanned all incremental scan locks";
            "lock_count" => locks.len(),
            "downstream_id" => ?self.downstream_id,
            "request_id" => ?self.request_id,
            "region_id" => region.get_id(),
            "conn_id" => ?self.conn_id,
        );

        fail_point!("before_schedule_resolver_ready");
        if let Err(e) = self.sched.schedule(Task::FinishScanLocks {
            observe_id,
            region,
            locks,
        }) {
            error!("cdc schedule task failed"; "error" => ?e);
        }
    }

    // Deregister downstream when the Initializer fails to initialize.
    pub(crate) fn deregister_downstream(&self, err: Error) {
        let build_resolver = self.build_resolver.load(Ordering::Acquire);
        let deregister = if build_resolver || err.has_region_error() {
            // Deregister delegate on the conditions,
            // * It fails to build a resolver. A delegate requires a resolver to advance
            //   resolved ts.
            // * A region error. It usually mean a peer is not leader or a leader meets an
            //   error and can not serve.
            Deregister::Delegate {
                region_id: self.region_id,
                observe_id: self.observe_handle.id,
                err,
            }
        } else {
            Deregister::Downstream {
                conn_id: self.conn_id,
                request_id: self.request_id,
                region_id: self.region_id,
                downstream_id: self.downstream_id,
                err: Some(err),
            }
        };

        if let Err(e) = self.sched.schedule(Task::Deregister(deregister)) {
            error!("cdc schedule cdc task failed"; "error" => ?e);
        }
    }

    fn ts_filter_is_helpful(&self, snap: &SnapAccess, start_key: &Key, end_key: &Key) -> bool {
        if self.ts_filter_ratio < f64::EPSILON {
            return false;
        }

        let hint_min_ts = self.checkpoint_ts.into_inner();
        let start = start_key.clone().into_raw().unwrap();
        let end = end_key.clone().into_raw().unwrap();
        let range = DataBound::new(
            InnerKey::from_outer_key(&start),
            InnerKey::from_outer_end_key(&end),
            false,
        );
        let (mut total_count, mut filtered_count, mut tables) = (0, 0, 0);
        snap.iter_tables(WRITE_CF, |tbl| {
            tables += 1;
            match tbl {
                TableRef::Memtable(m) => {
                    let count = m.get_cf(WRITE_CF).get_skl().entries();
                    if m.data_max_ts() < hint_min_ts || !m.has_data_in_bound(range) {
                        filtered_count += count;
                    }
                    total_count += count;
                }
                TableRef::L0(l) => {
                    if let Some(tbl) = l.get_cf(WRITE_CF) {
                        if tbl.max_ts < hint_min_ts || !tbl.has_overlap(range) {
                            filtered_count += tbl.entries as usize;
                        }
                        total_count += tbl.entries as usize;
                    }
                }
                TableRef::SsTable(tbl) => {
                    if tbl.max_ts < hint_min_ts || !tbl.has_overlap(range) {
                        filtered_count += tbl.entries as usize;
                    }
                    total_count += tbl.entries as usize;
                }
            }
        });

        let valid_count = total_count - filtered_count;
        let use_ts_filter = valid_count as f64 <= total_count as f64 * self.ts_filter_ratio;
        info!("cdc incremental scan uses ts filter: {}", use_ts_filter;
            "region_id" => self.region_id,
            "hint_min_ts" => hint_min_ts,
            "mvcc_versions" => total_count,
            "filtered_versions" => filtered_count,
            "tables" => tables);
        use_ts_filter
    }
}

#[cfg(test)]
mod tests {
    use std::{
        collections::BTreeMap,
        fmt::Display,
        sync::{
            atomic::AtomicBool,
            mpsc::{channel, sync_channel, Receiver, RecvTimeoutError, Sender},
            Arc,
        },
        time::Duration,
    };

    use cloud_server::mock_kv_engine::TestKvEngine;
    use futures::{executor::block_on, StreamExt};
    use kvproto::{
        cdcpb::{ChangeDataRequestKvApi, EventLogType, Event_oneof_event},
        errorpb::Error as ErrorHeader,
    };
    use raftstore::coprocessor::ObserveHandle;
    use rfstore::RaftRouter;
    use tikv::storage::{
        kv::Engine,
        txn::tests::{
            must_acquire_pessimistic_lock, must_commit, must_prewrite_delete, must_prewrite_put,
            must_prewrite_put_with_txn_soucre,
        },
    };
    use tikv_util::{
        memory::MemoryQuota,
        sys::thread::ThreadBuildWrapper,
        worker::{LazyWorker, Runnable},
    };
    use tokio::runtime::{Builder, Runtime};

    use super::*;
    use crate::{
        endpoint::tests::{api_v2_key, MockCdcRaftRouter},
        txn_source::TxnSource,
    };

    struct ReceiverRunnable<T: Display + Send> {
        tx: Sender<T>,
    }

    impl<T: Display + Send + 'static> Runnable for ReceiverRunnable<T> {
        type Task = T;

        fn run(&mut self, task: T) {
            let _ = self.tx.send(task);
        }
    }

    fn new_receiver_worker<T: Display + Send + 'static>() -> (LazyWorker<T>, Receiver<T>) {
        let (tx, rx) = channel();
        let runnable = ReceiverRunnable { tx };
        let mut worker = LazyWorker::new("test-receiver-worker");
        worker.start(runnable);
        (worker, rx)
    }

    fn mock_initializer(
        scan_limit: usize,
        fetch_limit: usize,
        buffer: usize,
        _engine: Option<kvengine::Engine>,
        _kv_api: ChangeDataRequestKvApi,
        filter_loop: bool,
    ) -> (
        LazyWorker<Task>,
        Runtime,
        Initializer,
        Receiver<Task>,
        crate::channel::Drain,
    ) {
        let (receiver_worker, rx) = new_receiver_worker();
        let quota = Arc::new(MemoryQuota::new(usize::MAX));
        let (sink, drain) = crate::channel::channel(ConnId::default(), buffer, quota);

        let pool = Builder::new_multi_thread()
            .thread_name("test-initializer-worker")
            .worker_threads(4)
            .with_sys_hooks()
            .build()
            .unwrap();
        let downstream_state = Arc::new(AtomicCell::new(DownstreamState::Initializing));
        let initializer = Initializer {
            region_id: 1,
            conn_id: ConnId::new(),
            request_id: RequestId(0),
            checkpoint_ts: 1.into(),
            region_epoch: RegionEpoch::default(),

            build_resolver: Arc::new(AtomicBool::new(true)),
            observed_range: ObservedRange::default(),
            observe_handle: ObserveHandle::new(),
            downstream_id: DownstreamId::new(),
            downstream_state,
            scan_truncated: Arc::new(Default::default()),

            sched: receiver_worker.scheduler(),
            sink,
            concurrency_semaphore: Arc::new(Semaphore::new(1)),

            scan_speed_limiter: Limiter::new(scan_limit as _),
            fetch_speed_limiter: Limiter::new(fetch_limit as _),
            max_scan_batch_bytes: 1024 * 1024,
            max_scan_batch_size: 1024,

            ts_filter_ratio: 1.0, // always enable it.
            filter_loop,
        };

        (receiver_worker, pool, initializer, rx, drain)
    }

    #[test]
    fn test_initializer_scan_locks() {
        let mut engine = TestKvEngine::new().unwrap();

        let mut expected_locks = BTreeMap::<Key, MiniLock>::new();

        // Only observe ["b\x00", "b\0x90"]
        let observed_range = ObservedRange::new(
            Key::from_raw(&api_v2_key(&[b'k', 0])).into_encoded(),
            Key::from_raw(&api_v2_key(&[b'k', 90])).into_encoded(),
        )
        .unwrap();

        // Pessimistic locks should not be tracked
        for i in 1..=10 {
            let (k, ts) = (&api_v2_key(&[b'k', i]), TimeStamp::new(i as _));
            must_acquire_pessimistic_lock(&mut engine, k, k, ts, ts);
        }

        for i in 11..=100 {
            let (k, v, ts) = (&api_v2_key(&[b'k', i]), &[b'v', i], TimeStamp::new(i as _));
            must_prewrite_put(&mut engine, k, v, k, ts);
            expected_locks.insert(Key::from_raw(k), MiniLock::from_ts(ts));
        }

        let region = Region::default();
        let snap = engine.snapshot(Default::default()).unwrap();
        let (mut worker, pool, mut initializer, rx, mut drain) = mock_initializer(
            usize::MAX,
            usize::MAX,
            1000,
            engine.kv_engine(),
            ChangeDataRequestKvApi::TiDb,
            false,
        );
        initializer.observed_range = observed_range.clone();
        initializer.build_resolver.store(true, Ordering::Release);
        initializer
            .downstream_state
            .store(DownstreamState::Initializing);

        let check_result = || {
            let task = rx.recv().unwrap();
            match task {
                Task::FinishScanLocks { locks, .. } => assert_eq!(locks, expected_locks),
                t => panic!("unexpected task {} received", t),
            }
        };

        pool.spawn(async move {
            let mut d = drain.drain();
            while let Some((e, _)) = d.next().await {
                if let CdcEvent::Event(e) = e {
                    for e in e.get_entries().get_entries() {
                        if e.r_type == EventLogType::Prewrite {
                            let key = Key::from_raw(&e.key).into_encoded();
                            assert!(observed_range.contains_encoded_key(&key));
                        }
                    }
                }
            }
        });

        block_on(initializer.async_incremental_scan(snap.clone(), region.clone())).unwrap();
        check_result();

        initializer.build_resolver.store(false, Ordering::Release);
        initializer
            .downstream_state
            .store(DownstreamState::Initializing);
        block_on(initializer.async_incremental_scan(snap.clone(), region.clone())).unwrap();
        match rx.recv_timeout(Duration::from_millis(100)) {
            Ok(t) => panic!("unexpected task {} received", t),
            Err(RecvTimeoutError::Timeout) => (),
            Err(e) => panic!("unexpected err {:?}", e),
        }

        // Test cancellation.
        initializer.downstream_state.store(DownstreamState::Stopped);
        block_on(initializer.async_incremental_scan(snap.clone(), region.clone())).unwrap_err();

        // Disconnect sink by dropping runtime (it also drops drain).
        initializer
            .downstream_state
            .store(DownstreamState::Initializing);
        drop(pool);
        block_on(initializer.async_incremental_scan(snap, region)).unwrap_err();

        worker.stop();
    }

    fn test_initializer_txn_source_filter(txn_source: TxnSource, filter_loop: bool) {
        let mut engine = TestKvEngine::new().unwrap();

        let mut total_bytes = 0;
        for i in 10..100 {
            let (k, v) = (&[b'k', i], &[b'v', i]);
            total_bytes += k.len();
            total_bytes += v.len();
            let ts = TimeStamp::new(i as _);
            must_prewrite_put_with_txn_soucre(&mut engine, k, v, k, ts, txn_source.into());
        }

        let snap = engine.snapshot(Default::default()).unwrap();
        // Buffer must be large enough to unblock async incremental scan.
        let buffer = 1000;
        let (mut worker, pool, mut initializer, _rx, mut drain) = mock_initializer(
            total_bytes,
            total_bytes,
            buffer,
            engine.kv_engine(),
            ChangeDataRequestKvApi::TiDb,
            filter_loop,
        );
        let th = pool.spawn(async move {
            initializer
                .async_incremental_scan(snap, Region::default())
                .await
                .unwrap();
        });
        let mut drain = drain.drain();
        while let Some((event, _)) = block_on(drain.next()) {
            let event = match event {
                CdcEvent::Event(x) if x.event.is_some() => x.event.unwrap(),
                _ => continue,
            };
            let entries = match event {
                Event_oneof_event::Entries(mut x) => x.take_entries().into_vec(),
                _ => continue,
            };
            assert_eq!(entries.len(), 1);
            assert_eq!(entries[0].get_type(), EventLogType::Initialized);
        }
        block_on(th).unwrap();
        worker.stop();
    }

    #[test]
    fn test_initializer_cdc_write_filter() {
        let mut txn_source = TxnSource::default();
        txn_source.set_cdc_write_source(1);
        test_initializer_txn_source_filter(txn_source, true);
    }

    #[test]
    fn test_initializer_lightning_physical_import_filter() {
        let mut txn_source = TxnSource::default();
        txn_source.set_lightning_physical_import();
        test_initializer_txn_source_filter(txn_source, false);
    }

    #[test]
    fn test_initializer_lossy_ddl_filter() {
        let mut txn_source = TxnSource::default();
        txn_source.set_lossy_ddl_reorg_source(1);
        test_initializer_txn_source_filter(txn_source, false);

        // With cdr write source and filter loop is false, we should still ignore lossy
        // ddl changes.
        let mut txn_source = TxnSource::default();
        txn_source.set_cdc_write_source(1);
        txn_source.set_lossy_ddl_reorg_source(1);
        test_initializer_txn_source_filter(txn_source, false);

        // With cdr write source and filter loop is true, we should still ignore all
        // events.
        let mut txn_source = TxnSource::default();
        txn_source.set_cdc_write_source(1);
        txn_source.set_lossy_ddl_reorg_source(1);
        test_initializer_txn_source_filter(txn_source, true);
    }

    // Test `hint_min_ts` works fine with `ExtraOp::ReadOldValue`.
    // Whether `DeltaScanner` emits correct old values or not is already tested by
    // another case `test_old_value_with_hint_min_ts`, so here we only care about
    // handling `OldValue::SeekWrite` with `OldValueReader`.
    #[test]
    fn test_incremental_scanner_with_hint_min_ts() {
        let mut engine = TestKvEngine::new().unwrap();

        let v_suffix = |suffix: usize| -> Vec<u8> {
            let suffix = suffix.to_string().into_bytes();
            let mut v = Vec::with_capacity(1000 + suffix.len());
            (0..100).for_each(|_| v.extend_from_slice(b"vvvvvvvvvv"));
            v.extend_from_slice(&suffix);
            v
        };

        fn check_handling_old_value_seek_write<F>(engine: &mut TestKvEngine, v_suffix: F)
        where
            F: Fn(usize) -> Vec<u8>,
        {
            // Do incremental scan with different `hint_min_ts` values.
            for checkpoint_ts in [200, 100, 150] {
                let (mut worker, pool, mut initializer, _rx, mut drain) = mock_initializer(
                    usize::MAX,
                    usize::MAX,
                    1000,
                    engine.kv_engine(),
                    ChangeDataRequestKvApi::TiDb,
                    false,
                );
                initializer.checkpoint_ts = checkpoint_ts.into();
                let mut drain = drain.drain();

                let snap = engine.snapshot(Default::default()).unwrap();
                let th = pool.spawn(async move {
                    initializer
                        .async_incremental_scan(snap, Region::default())
                        .await
                        .unwrap();
                });

                while let Some((event, _)) = block_on(drain.next()) {
                    let event = match event {
                        CdcEvent::Event(x) if x.event.is_some() => x.event.unwrap(),
                        _ => continue,
                    };
                    let entries = match event {
                        Event_oneof_event::Entries(mut x) => x.take_entries().into_vec(),
                        _ => continue,
                    };
                    for entry in entries.into_iter().filter(|x| x.start_ts == 200) {
                        // Check old value is expected in all cases.
                        assert_eq!(entry.get_old_value(), &v_suffix(100));
                    }
                }
                block_on(th).unwrap();
                worker.stop();
            }
        }

        // Create the initial data with CF_WRITE L0: |zkey_110, zkey1_160|
        must_prewrite_put(&mut engine, b"zkey", &v_suffix(100), b"zkey", 100);
        must_commit(&mut engine, b"zkey", 100, 110);
        must_prewrite_put(&mut engine, b"zzzz", &v_suffix(150), b"zzzz", 150);
        must_commit(&mut engine, b"zzzz", 150, 160);
        must_prewrite_delete(&mut engine, b"zkey", b"zkey", 200);
        check_handling_old_value_seek_write(&mut engine, v_suffix); // For TxnEntry::Prewrite.

        // CF_WRITE L0: |zkey_110, zkey1_160|, |zkey_210|
        must_commit(&mut engine, b"zkey", 200, 210);
        check_handling_old_value_seek_write(&mut engine, v_suffix); // For TxnEntry::Commit.
    }

    #[test]
    fn test_initializer_deregister_downstream() {
        let total_bytes = 1;
        let buffer = 1;
        let (mut worker, _pool, initializer, rx, _drain) = mock_initializer(
            total_bytes,
            total_bytes,
            buffer,
            None,
            ChangeDataRequestKvApi::TiDb,
            false,
        );

        // Errors reported by region should deregister region.
        initializer.build_resolver.store(false, Ordering::Release);
        initializer.deregister_downstream(Error::request(ErrorHeader::default()));
        let task = rx.recv_timeout(Duration::from_millis(100));
        match task {
            Ok(Task::Deregister(Deregister::Delegate { region_id, .. })) => {
                assert_eq!(region_id, initializer.region_id);
            }
            Ok(other) => panic!("unexpected task {:?}", other),
            Err(e) => panic!("unexpected err {:?}", e),
        }

        initializer.build_resolver.store(false, Ordering::Release);
        initializer.deregister_downstream(Error::Other(box_err!("test")));
        let task = rx.recv_timeout(Duration::from_millis(100));
        match task {
            Ok(Task::Deregister(Deregister::Downstream { region_id, .. })) => {
                assert_eq!(region_id, initializer.region_id);
            }
            Ok(other) => panic!("unexpected task {:?}", other),
            Err(e) => panic!("unexpected err {:?}", e),
        }

        // Test deregister region when resolver fails to build.
        initializer.build_resolver.store(true, Ordering::Release);
        initializer.deregister_downstream(Error::Other(box_err!("test")));
        let task = rx.recv_timeout(Duration::from_millis(100));
        match task {
            Ok(Task::Deregister(Deregister::Delegate { region_id, .. })) => {
                assert_eq!(region_id, initializer.region_id);
            }
            Ok(other) => panic!("unexpected task {:?}", other),
            Err(e) => panic!("unexpected err {:?}", e),
        }

        worker.stop();
    }

    #[test]
    fn test_initializer_initialize() {
        test_initializer_initialize_impl(ChangeDataRequestKvApi::TiDb);
        test_initializer_initialize_impl(ChangeDataRequestKvApi::RawKv);
    }

    fn test_initializer_initialize_impl(kv_api: ChangeDataRequestKvApi) {
        let total_bytes = 1;
        let buffer = 1;
        let (mut worker, pool, mut initializer, _rx, _drain) =
            mock_initializer(total_bytes, total_bytes, buffer, None, kv_api, false);

        let (raft_tx, _raft_rx) = tikv_util::mpsc::unbounded();
        let (store_tx, _store_rx) = tikv_util::mpsc::unbounded();
        let raft_router = RaftRouter::new(raft_tx, store_tx);
        let cdc_handle = MockCdcRaftRouter::new(raft_router.clone());

        initializer.downstream_state.store(DownstreamState::Stopped);
        block_on(initializer.initialize(cdc_handle.clone())).unwrap_err();

        let (tx, rx) = sync_channel(1);
        let concurrency_semaphore = initializer.concurrency_semaphore.clone();
        pool.spawn(async move {
            let _permit = concurrency_semaphore.acquire().await;
            tx.send(()).unwrap();
            tx.send(()).unwrap();
            tx.send(()).unwrap();
        });
        rx.recv_timeout(Duration::from_millis(200)).unwrap();

        let (tx1, rx1) = sync_channel(1);
        pool.spawn(async move {
            let res = initializer.initialize(cdc_handle).await;
            tx1.send(res).unwrap();
        });
        // Must timeout because there is no enough permit.
        rx1.recv_timeout(Duration::from_millis(200)).unwrap_err();

        // Release the permit
        rx.recv_timeout(Duration::from_millis(200)).unwrap();
        let res = rx1.recv_timeout(Duration::from_millis(200)).unwrap();
        res.unwrap_err();

        worker.stop();
    }

    #[test]
    fn test_initialize_scan_range() {
        let mut engine = TestKvEngine::new().unwrap();

        // Must start with 'z', otherwise table property collector doesn't work.
        let ka = Key::from_raw(b"zaaa").into_encoded();
        let km = Key::from_raw(b"zmmm").into_encoded();
        let ky = Key::from_raw(b"zyyy").into_encoded();
        let kz = Key::from_raw(b"zzzz").into_encoded();

        // Incremental scan iterator shouldn't access the key because it's out of range.
        must_prewrite_put(&mut engine, &ka, b"value", &ka, 200);
        must_commit(&mut engine, &ka, 200, 210);

        // Incremental scan iterator shouldn't access the key because it's skiped by ts
        // filter.
        must_prewrite_put(&mut engine, &km, b"value", &km, 100);
        must_commit(&mut engine, &km, 100, 110);

        must_prewrite_put(&mut engine, &ky, b"value", &ky, 200);
        must_commit(&mut engine, &ky, 200, 210);

        let (mut _worker, pool, mut initializer, _rx, mut drain) = mock_initializer(
            usize::MAX,
            usize::MAX,
            1000,
            engine.kv_engine(),
            ChangeDataRequestKvApi::TiDb,
            false,
        );

        initializer.observed_range = ObservedRange::new(km, kz).unwrap();
        initializer.checkpoint_ts = 150.into();

        let th = pool.spawn(async move {
            let snap = engine.snapshot(Default::default()).unwrap();
            let region = Region::default();
            let scan_stat = initializer
                .async_incremental_scan(snap, region)
                .await
                .unwrap();
            // FIXME: update ScanStat to fix kvengine better.
            assert_eq!(scan_stat.emit, 72);
        });
        while block_on(drain.drain().next()).is_some() {}
        block_on(th).unwrap();
    }
}
