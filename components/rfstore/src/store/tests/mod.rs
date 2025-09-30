// Copyright 2025 TiKV Project Authors. Licensed under Apache-2.0.

mod test_pd_worker;
mod test_peer_fsm;

use std::{
    collections::HashSet,
    sync::{atomic::AtomicU64, Arc, Mutex},
};

use cloud_encryption::MasterKey;
use file_system::IoRateLimiter;
use kvengine::limiter::StoreLimiter;
use kvproto::{
    kvrpcpb::ApiVersion,
    metapb::{self},
    raft_serverpb::RaftMessage,
};
use raftstore::coprocessor::CoprocessorHost;
use rfengine::{RfEngine, RfEngineConfig};
use security::SecurityManager;
use sst_importer::SstImporter;
use tempfile::TempDir;
use test_pd_client::TestPdClient;
use tikv_util::{config::VersionTrack, store::new_peer, worker::LazyWorker};

use crate::{
    store::{initial_region, prepare_bootstrap_cluster, Config, *},
    RaftRouter,
};

// Creates new engines (`kvengine` and `rf_engine`) for use in tests.
fn new_test_engines() -> (Engines, TempDir) {
    let tmp_dir = test_util::temp_dir("test_node_", true);
    let data_dir = tmp_dir.path();

    let mut rfengine_cfg = RfEngineConfig::default();
    rfengine_cfg.wal_sync_dir = format!("{}/wal", data_dir.display());

    let rf_engine = RfEngine::open(
        data_dir.join("rfengine").as_path(),
        &rfengine_cfg,
        Some(data_dir),
        None,
    )
    .unwrap();

    let rate_limiter = Arc::new(IoRateLimiter::new_for_test());
    let store_limiter = Arc::new(StoreLimiter::dummy());

    let (sender, _) = tikv_util::mpsc::unbounded();
    let meta_change_listener = Box::new(MetaChangeListener {
        sender: sender.clone(),
    });

    let dfs = Arc::new(kvengine::dfs::InMemFs::new());
    let recoverer = crate::store::RecoverHandler::new(rf_engine.clone());
    let mut meta_iter = recoverer.clone();
    let pd_client = Arc::new(TestPdClient::new(1, false));
    let id_allocator = Arc::new(PdIdAllocator::new(pd_client.clone()));

    let kv_cfg = kvengine::config::Config::default();
    let mut kv_opts = kvengine::Options::default();
    kv_opts.local_dir = data_dir.to_path_buf();

    let kvengine = kvengine::Engine::open(
        dfs,
        Arc::new(kv_opts),
        kv_cfg,
        &mut meta_iter,
        recoverer,
        id_allocator,
        meta_change_listener,
        rate_limiter,
        store_limiter,
        MasterKey::new(&[1u8; 32]),
        Arc::new(SecurityManager::default()),
    )
    .unwrap();

    let engines = Engines::new(kvengine, rf_engine, tikv_util::mpsc::unbounded(), None);

    let region = initial_region(1, 1, 1);
    prepare_bootstrap_cluster(&engines, &region, 1).unwrap();

    (engines, tmp_dir)
}

// A mock transport implementation that captures Raft messages sent by peers.
#[derive(Clone)]
pub struct MockTransport {
    messages: Arc<Mutex<Vec<RaftMessage>>>,
}

impl MockTransport {
    pub fn new() -> Self {
        MockTransport {
            messages: Arc::new(Mutex::new(Vec::new())),
        }
    }
}

impl Transport for MockTransport {
    fn send(&mut self, msg: RaftMessage) -> crate::Result<()> {
        self.messages.lock().unwrap().push(msg);
        Ok(())
    }

    fn need_flush(&self) -> bool {
        false
    }

    fn flush(&mut self) {}
}

// Initializes a `PeerFsm` instance for testing purposes.
fn new_test_peer_fsm(engines: Engines, region: &metapb::Region) -> Result<PeerFsm, crate::Error> {
    let read_worker = tikv_util::worker::Worker::new("test-read-worker");
    let read_scheduler = read_worker.start(
        "test-read-worker",
        crate::store::worker::ReadRunner::new(
            engines.raft.clone(),
            RaftRouter::new(
                tikv_util::mpsc::unbounded().0,
                tikv_util::mpsc::unbounded().0,
            ),
        ),
    );
    PeerFsm::create(1, &Config::default(), engines, region, read_scheduler)
}

// Builds a `RaftContext` for testing.
fn new_test_raft_ctx(
    engines: Engines,
    pd_scheduler: Option<tikv_util::worker::Scheduler<PdTask>>,
) -> RaftContext {
    let rf_store_cfg = Config::default();
    let cfg = Arc::new(VersionTrack::new(rf_store_cfg));

    let tempfile = tempfile::tempdir().unwrap();
    let sst_import_dir = tempfile.path();
    let importer = Arc::new(
        SstImporter::new(
            &sst_importer::Config::default(),
            sst_import_dir,
            None,
            ApiVersion::V2,
        )
        .unwrap(),
    );

    let pd_scheduler =
        pd_scheduler.unwrap_or_else(|| LazyWorker::new("test-pd-worker").scheduler());
    let gc_worker = LazyWorker::new("test-gc-worker");
    let schema_worker = LazyWorker::new("test-schema-worker");

    let (peer_sender, _) = tikv_util::mpsc::unbounded();
    let (store_sender, _) = tikv_util::mpsc::unbounded();
    let router = RaftRouter::new(peer_sender, store_sender);

    let read_worker = tikv_util::worker::Worker::new("test-read-worker");
    let read_scheduler = read_worker.start(
        "test-read-worker",
        crate::store::worker::ReadRunner::new(engines.raft.clone(), router.clone()),
    );

    let trans = MockTransport::new();

    let global_ctx: GlobalContext = GlobalContext {
        cfg,
        engines,
        store: metapb::Store::default(),
        readers: StoreMeta::new(PENDING_MSG_CAP).readers.clone(),
        router,
        trans: Box::new(trans),
        pd_scheduler,
        gc_scheduler: gc_worker.scheduler(),
        schema_scheduler: schema_worker.scheduler(),
        read_scheduler,
        coprocessor_host: CoprocessorHost::default(),
        importer,
        destroying: HashSet::default(),
        engine_total_bytes_written: Arc::new(AtomicU64::new(0)),
        engine_total_keys_written: Arc::new(AtomicU64::new(0)),
    };
    RaftContext::new(global_ctx)
}
