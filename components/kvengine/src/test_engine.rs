// Copyright 2025 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    env,
    ops::Deref,
    path::Path,
    sync::{atomic::AtomicU64, Arc},
    thread,
    time::Duration,
    u64, vec,
};

use api_version::{api_v2::KEYSPACE_PREFIX_LEN, ApiV2};
use async_trait::async_trait;
use cloud_encryption::MasterKey;
use file_system::IoRateLimiter;
use kvenginepb as pb;
use security::SecurityManager;
use tempfile::TempDir;
use tikv_util::mpsc;
use util::test_util::KeyBuilder;

use crate::{dfs::InMemFs, limiter::StoreLimiter, *};

macro_rules! unwrap_or_return {
    ($e:expr, $m:expr) => {
        match $e {
            Ok(x) => x,
            Err(y) => {
                error!("{:?} {:?}", y, $m);
                return;
            }
        }
    };
}

pub const KEYSPACE_ID: u32 = 1;

pub const DEF_BLOCK_SIZE: usize = 4 << 10;
pub const DEF_MIN_BLOB_SIZE: u32 = 64;

pub const TABLE_KEY_PREFIX: &str = "t_";

/// Wrap `Engine` to make sure that it will be closed after the test, and not
/// interfere with other tests.
pub struct TestEngine {
    pub engine: Engine,
    pub key_builder: KeyBuilder,
}

impl std::ops::Deref for TestEngine {
    type Target = Engine;

    fn deref(&self) -> &Self::Target {
        &self.engine
    }
}

impl Drop for TestEngine {
    fn drop(&mut self) {
        self.engine.close();
    }
}

impl TestEngine {
    pub fn key_builder(&self) -> &KeyBuilder {
        &self.key_builder
    }
}

pub fn new_test_engine() -> (TestEngine, mpsc::Sender<ApplyTask>) {
    new_test_engine_opt(false, DEF_BLOCK_SIZE, "")
}

pub fn new_test_engine_api_v2() -> (TestEngine, mpsc::Sender<ApplyTask>) {
    new_test_engine_opt(true, DEF_BLOCK_SIZE, "")
}

pub fn new_test_engine_opt(
    enable_inner_key_off: bool,
    block_size: usize,
    key_prefix: &str,
) -> (TestEngine, mpsc::Sender<ApplyTask>) {
    let (listener_tx, listener_rx) = mpsc::unbounded();
    let tester = EngineTester::new(enable_inner_key_off, block_size);
    let meta_change_listener = Box::new(TestMetaChangeListener {
        sender: listener_tx,
    });
    let rate_limiter = Arc::new(IoRateLimiter::new_for_test());
    let store_limiter = Arc::new(StoreLimiter::dummy());
    let mut meta_iter = tester.clone();
    let engine = Engine::open(
        tester.fs.clone(),
        tester.opts.clone(),
        tester.config.clone(),
        &mut meta_iter,
        tester.clone(),
        tester.core.clone(),
        meta_change_listener,
        rate_limiter,
        store_limiter,
        MasterKey::new(&[1u8; 32]),
        Arc::new(SecurityManager::default()),
    )
    .unwrap();
    {
        let shard = engine.get_shard(1).unwrap();
        store_bool(&shard.active, true);
    }
    let (applier_tx, applier_rx) = mpsc::unbounded();
    let meta_listener = MetaListener::new(listener_rx, applier_tx.clone());
    thread::spawn(move || {
        meta_listener.run();
    });
    let applier = Applier::new(engine.clone(), applier_rx);
    thread::spawn(move || {
        applier.run();
    });
    let keyspace_id = if enable_inner_key_off { 1 } else { 0 };
    let key_builder = KeyBuilder::new(keyspace_id, key_prefix);
    (
        TestEngine {
            engine,
            key_builder,
        },
        applier_tx,
    )
}

#[derive(Clone)]
struct TestMetaChangeListener {
    sender: mpsc::Sender<pb::ChangeSet>,
}

impl MetaChangeListener for TestMetaChangeListener {
    fn on_change_set(&self, cs: pb::ChangeSet) {
        info!("on meta change listener");
        self.sender.send(cs).unwrap();
    }
}

#[derive(Clone)]
pub struct EngineTester {
    core: Arc<EngineTesterCore>,
}

impl Deref for EngineTester {
    type Target = EngineTesterCore;
    fn deref(&self) -> &Self::Target {
        &self.core
    }
}

impl EngineTester {
    pub fn new(enable_inner_key_off: bool, block_size: usize) -> Self {
        let initial_cs = new_initial_cs(enable_inner_key_off);
        let initial_meta = ShardMeta::new(1, &initial_cs);
        let metas = dashmap::DashMap::new();
        metas.insert(1, Arc::new(initial_meta));
        let tmp_dir = TempDir::new().unwrap();
        let opts = new_test_options(tmp_dir.path(), enable_inner_key_off, block_size);
        let config = KvEngineConfig::default();

        Self {
            core: Arc::new(EngineTesterCore {
                _tmp_dir: tmp_dir,
                metas,
                fs: Arc::new(InMemFs::new()),
                opts: Arc::new(opts),
                config,
                id: AtomicU64::new(0),
            }),
        }
    }
}

pub struct EngineTesterCore {
    _tmp_dir: TempDir,
    metas: dashmap::DashMap<u64, Arc<ShardMeta>>,
    fs: Arc<dfs::InMemFs>,
    opts: Arc<Options>,
    config: KvEngineConfig,
    id: AtomicU64,
}

impl MetaIterator for EngineTester {
    fn iterate<F>(&mut self, mut f: F) -> Result<()>
    where
        F: FnMut(kvenginepb::ChangeSet),
    {
        for meta in &self.metas {
            f(meta.value().to_change_set())
        }
        Ok(())
    }

    fn engine_id(&self) -> u64 {
        1
    }
}

impl RecoverHandler for EngineTester {
    fn recover(&self, _engine: &Engine, _shard: &Arc<Shard>, _info: &ShardMeta) -> Result<()> {
        Ok(())
    }
}

#[async_trait]
impl IdAllocator for EngineTesterCore {
    fn alloc_id(&self, count: usize) -> Result<Vec<u64>> {
        let start_id = self
            .id
            .fetch_add(count as u64, std::sync::atomic::Ordering::Relaxed)
            + 1;
        let end_id = start_id + count as u64;
        let mut ids = Vec::with_capacity(count);
        for id in start_id..end_id {
            ids.push(id);
        }
        Ok(ids)
    }

    async fn alloc_id_async(&self, count: usize) -> Result<Vec<u64>> {
        self.alloc_id(count)
    }
}

struct MetaListener {
    applier_tx: mpsc::Sender<ApplyTask>,
    meta_rx: mpsc::Receiver<pb::ChangeSet>,
}

impl MetaListener {
    fn new(meta_rx: mpsc::Receiver<pb::ChangeSet>, applier_tx: mpsc::Sender<ApplyTask>) -> Self {
        Self {
            meta_rx,
            applier_tx,
        }
    }

    fn run(&self) {
        loop {
            let cs = unwrap_or_return!(self.meta_rx.recv(), "meta_listener_a");
            let (tx, rx) = mpsc::bounded(1);
            let task = ApplyTask::new_cs(cs, tx);
            self.applier_tx.send(task).unwrap();
            let res = unwrap_or_return!(rx.recv(), "meta_listener_b");
            unwrap_or_return!(res, "meta_listener_c");
        }
    }
}

struct Applier {
    engine: Engine,
    task_rx: mpsc::Receiver<ApplyTask>,
}

impl Applier {
    fn new(engine: Engine, task_rx: mpsc::Receiver<ApplyTask>) -> Self {
        Self { engine, task_rx }
    }

    fn run(&self) {
        let mut seq = 2;
        loop {
            let mut task = unwrap_or_return!(self.task_rx.recv(), "apply recv task");
            seq += 1;
            if let Some(wb) = task.wb.as_mut() {
                wb.set_sequence(seq);
                self.engine.write(wb, &[]);
            }
            if let Some(mut cs) = task.cs.take() {
                cs.set_sequence(seq);
                if cs.has_split() {
                    let mut ids = vec![];
                    for new_shard in cs.get_split().get_new_shards() {
                        ids.push(new_shard.shard_id);
                    }
                    unwrap_or_return!(self.engine.split(cs, 1), "apply split");
                    for id in ids {
                        let shard = self.engine.get_shard(id).unwrap();
                        shard.set_active(true);
                    }
                    info!("applier executed split");
                } else {
                    self.engine.meta_committed(&cs, false);
                    unwrap_or_return!(
                        self.engine.apply_change_set(
                            &self
                                .engine
                                .prepare_change_set(cs, false, false, None, None, None)
                                .unwrap()
                        ),
                        "applier apply changeset"
                    );
                }
            }
            task.result_tx.send(Ok(seq)).unwrap();
        }
    }
}

pub struct ApplyTask {
    pub wb: Option<WriteBatch>,
    pub cs: Option<pb::ChangeSet>,
    pub result_tx: mpsc::Sender<Result<u64 /* write_sequence */>>,
}

impl ApplyTask {
    pub fn new_cs(cs: pb::ChangeSet, result_tx: mpsc::Sender<Result<u64>>) -> Self {
        Self {
            wb: None,
            cs: Some(cs),
            result_tx,
        }
    }

    pub fn new_wb(wb: WriteBatch, result_tx: mpsc::Sender<Result<u64>>) -> Self {
        Self {
            wb: Some(wb),
            cs: None,
            result_tx,
        }
    }
}

struct Splitter {
    apply_sender: mpsc::Sender<ApplyTask>,
    keys: Vec<Vec<u8>>,
    current_shard_id: u64,
    shard_ver: u64,
    new_id: u64,
}

#[allow(dead_code)]
impl Splitter {
    fn new(
        keys: Vec<Vec<u8>>,
        current_shard_id_ver: IdVer,
        new_id_base: u64,
        apply_sender: mpsc::Sender<ApplyTask>,
    ) -> Self {
        Self {
            keys,
            apply_sender,
            current_shard_id: current_shard_id_ver.id,
            shard_ver: current_shard_id_ver.ver,
            new_id: new_id_base,
        }
    }

    fn run(&mut self) {
        let keys = self.keys.clone();
        for key in keys {
            thread::sleep(Duration::from_millis(200));
            self.new_id += 1;
            self.split(key.clone(), vec![self.new_id, self.current_shard_id]);
        }
    }

    fn send_task(&mut self, cs: pb::ChangeSet) {
        let (tx, rx) = mpsc::bounded(1);
        let task = ApplyTask {
            cs: Some(cs),
            wb: None,
            result_tx: tx,
        };
        self.apply_sender.send(task).unwrap();
        let res = unwrap_or_return!(rx.recv(), "splitter recv");
        res.unwrap();
    }

    fn split(&mut self, key: Vec<u8>, new_ids: Vec<u64>) {
        let mut cs = pb::ChangeSet::new();
        cs.set_shard_id(self.current_shard_id);
        cs.set_shard_ver(self.shard_ver);
        let mut finish_split = pb::Split::new();
        finish_split.set_keys(protobuf::RepeatedField::from_vec(vec![key]));
        let mut new_shards = Vec::new();
        for new_id in &new_ids {
            let mut new_shard = pb::Properties::new();
            new_shard.set_shard_id(*new_id);
            new_shards.push(new_shard);
        }
        finish_split.set_new_shards(protobuf::RepeatedField::from_vec(new_shards));
        cs.set_split(finish_split);
        self.send_task(cs);
        self.shard_ver += 1;
    }
}

fn new_initial_cs(enable_inner_key_off: bool) -> pb::ChangeSet {
    let mut cs = pb::ChangeSet::new();
    cs.set_shard_id(1);
    cs.set_shard_ver(1);
    cs.set_sequence(1);
    let mut snap = pb::Snapshot::new();
    snap.set_base_version(1);
    if enable_inner_key_off {
        let (start, end) = ApiV2::get_txn_keyspace_range(KEYSPACE_ID);
        snap.set_outer_start(start);
        snap.set_outer_end(end);
        snap.set_inner_key_off(KEYSPACE_PREFIX_LEN as u32);
    } else {
        snap.set_outer_end(GLOBAL_SHARD_END_KEY.to_vec());
    }
    let props = snap.mut_properties();
    props.shard_id = 1;
    cs.set_snapshot(snap);
    cs
}

fn new_test_options(
    path: impl AsRef<Path>,
    enable_inner_key_off: bool,
    block_size: usize,
) -> Options {
    let min_blob_size: u32 = match env::var("MIN_BLOB_SIZE") {
        Ok(val) => match val.trim().parse() {
            Ok(n) => n,
            Err(e) => {
                warn!("MIN_BLOB_SIZE=<number>, got {}", e);
                DEF_MIN_BLOB_SIZE
            }
        },
        Err(_) => DEF_MIN_BLOB_SIZE,
    };
    info!("MIN_BLOB_SIZE={}", min_blob_size);
    let mut opts = Options::default();
    opts.local_dir = path.as_ref().to_path_buf();
    opts.base_size = 64 << 10;
    opts.table_builder_options.block_size = block_size;
    opts.table_builder_options.max_table_size = 8 << 10;
    opts.table_builder_options.flush_split_l0 = true;
    opts.columnar_build_options.max_columnar_table_size = 1024;
    opts.columnar_build_options.pack_max_row_count = 9;
    opts.max_mem_table_size = 32 << 10; // mem-table size should be much larger than max_table_size.
    opts.num_compactors = 2;
    opts.blob_table_build_options.min_blob_size = min_blob_size;
    opts.max_del_range_delay = Duration::from_secs(1);
    opts.enable_inner_key_offset = enable_inner_key_off;
    opts.read_columnar = true;
    opts
}
