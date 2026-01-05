// Copyright 2025 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    iter::FromIterator,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    time::Duration,
};

use collections::HashMap;
use concurrency_manager::ConcurrencyManager;
use futures::executor::block_on;
use health_controller::HealthController;
use kvproto::pdpb::RegionHeartbeatResponse;
use pd_client::{Error, PdClient, PdFuture};
use security::GetSecurityManager;
use tempfile::TempDir;
use tikv_util::{box_err, time::Instant};
use tokio::sync::Mutex;
use txn_types::{ClusterGcStates, GcState, NULL_KEYSPACE_ID};
use yatp::task::future;

use crate::{
    store::{
        tests::new_test_engines, Config as RfStoreConfig, CpuUtilCollector, Engines, PdRunner,
        PdTask,
    },
    RaftRouter,
};

struct TestPdRunnerSuite {
    pd_worker: tikv_util::worker::LazyWorker<PdTask>,
    engines: Engines,
    _pool: yatp::ThreadPool<future::TaskCell>,
    _temp_dir: TempDir,
}

impl TestPdRunnerSuite {
    fn with_pd_client(pd_client: Arc<dyn PdClient>) -> TestPdRunnerSuite {
        let (peer_sender, _) = tikv_util::mpsc::unbounded();
        let (store_sender, _) = tikv_util::mpsc::unbounded();
        let router = RaftRouter::new(peer_sender, store_sender);

        let mut pd_worker = tikv_util::worker::LazyWorker::new("pd-worker");

        let yatp_pool = yatp::Builder::new("test-pool")
            .max_thread_count(2)
            .build_future_pool();

        let (engine, temp_dir) = new_test_engines();

        let rfstore_cfg = RfStoreConfig::default();

        let pd_runner = PdRunner::new(
            &rfstore_cfg,
            1,
            pd_client,
            router,
            pd_worker.scheduler(),
            ConcurrencyManager::new(1.into()),
            yatp_pool.remote().clone(),
            engine.kv.clone(),
            CpuUtilCollector::new("test-pd-".into()),
            HealthController::default(),
        );

        pd_worker.start(pd_runner);

        Self {
            _temp_dir: temp_dir,
            engines: engine,
            _pool: yatp_pool,
            pd_worker,
        }
    }
}

impl Drop for TestPdRunnerSuite {
    fn drop(&mut self) {
        self.pd_worker.stop();
    }
}

#[test]
fn test_update_gc_safe_point_avoiding_concurrent_running() {
    struct MockPdClient {
        result_rx: Arc<Mutex<tokio::sync::mpsc::Receiver<pd_client::Result<ClusterGcStates>>>>,
        call_count: AtomicUsize,
    }

    impl GetSecurityManager for MockPdClient {}
    impl PdClient for MockPdClient {
        fn get_cluster_id(&self) -> pd_client::Result<u64> {
            Ok(1)
        }

        fn handle_region_heartbeat_response(
            &self,
            _store_id: u64,
            _f: Box<dyn Fn(RegionHeartbeatResponse) + Send + 'static>,
        ) -> PdFuture<()> {
            Box::pin(async { Ok(()) })
        }

        fn get_all_keyspaces_gc_states(&self) -> PdFuture<ClusterGcStates> {
            self.call_count.fetch_add(1, Ordering::SeqCst);
            let rx = self.result_rx.clone();
            Box::pin(async move {
                let mut rx = rx.try_lock().map_err(|e| {
                    Error::Other(box_err!("failed to lock result rx, get_all_keyspaces_gc_states might be called concurrenty, which in unexpected. err: {:?}", e))
                })?;
                rx.recv().await.unwrap()
            })
        }
    }
    let (tx, rx) = tokio::sync::mpsc::channel(10);
    let pd = Arc::new(MockPdClient {
        result_rx: Arc::new(Mutex::new(rx)),
        call_count: AtomicUsize::new(0),
    });

    let suite = TestPdRunnerSuite::with_pd_client(pd.clone());
    for _i in 0..5 {
        suite
            .pd_worker
            .scheduler()
            .schedule(PdTask::UpdateGcSafePoint)
            .unwrap();
    }
    assert_eq!(
        suite.engines.kv.get_gc_safe_point(NULL_KEYSPACE_ID),
        0.into()
    );

    for &value in &[10u64, 20] {
        let now = Instant::now();
        block_on(tx.send(Ok(ClusterGcStates::new(
            HashMap::from_iter(std::iter::once((
                NULL_KEYSPACE_ID,
                GcState::new(
                    NULL_KEYSPACE_ID,
                    false,
                    value.into(),
                    value.into(),
                    vec![],
                    now,
                ),
            ))),
            now,
        ))))
        .unwrap();
    }

    // The first message (10) is consumed and is used to update the GC states stored
    // in the kv engine.
    for _i in 0..50 {
        match suite
            .engines
            .kv
            .get_gc_safe_point(NULL_KEYSPACE_ID)
            .into_inner()
        {
            0 => std::thread::sleep(Duration::from_millis(50)),
            10 => break,
            20 => panic!(
                "the value for the second RPC call is used, which is expected never to happen"
            ),
            _ => unreachable!(),
        }
    }
    if suite.engines.kv.get_gc_safe_point(NULL_KEYSPACE_ID) == 0.into() {
        panic!("gc states not updated")
    }

    // The second message (20) is never used, as the subsequent
    // `PdTask::UpdateGcSafePoint` tasks are ignored.
    for _i in 0..2 {
        std::thread::sleep(Duration::from_millis(100));
        assert_eq!(
            suite.engines.kv.get_gc_safe_point(NULL_KEYSPACE_ID),
            10.into()
        );
    }

    drop(suite);
    // The RPC is called only once.
    assert_eq!(pd.call_count.load(Ordering::SeqCst), 1);
}
