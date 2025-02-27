// Copyright 2025 TiKV Project Authors. Licensed under Apache-2.0.
use std::{
    fmt::Write,
    path::Path,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    thread,
    time::Duration,
};

use cloud_encryption::MasterKey;
use cloud_server::TikvServer;
use file_system::IoRateLimiter;
use kvengine::{dfs::Dfs, WRITE_CF};
use kvproto::{
    metapb::RegionEpoch,
    raft_cmdpb::{CmdType, CustomRequest, RaftCmdRequest, RaftCmdResponse},
    raft_serverpb::RegionLocalState,
};
use pd_client::PdClient;
use protobuf::Message;
use rand::RngCore;
use rfengine::{RfEngine, REGION_META_KEY_BYTE};
use rfstore::store::{
    rlog, Callback, CustomBuilder, Engines, ExtCallback, ReadResponse, WriteResponse,
};
use security::SecurityManager;
use test_raftstore::Config;
use tikv_util::{debug, escape, mpsc::future, warn};
use txn_types::Key;

use crate::{Cluster, Simulator};

fn gen_ts() -> u64 {
    // a counter to generate sequential timestamp value.
    static TS_COUNTER: AtomicU64 = AtomicU64::new(1);

    TS_COUNTER.fetch_add(1, Ordering::Relaxed)
}

pub fn create_test_engine(
    cfg: &Config,
    dir: &std::path::Path,
    pd: Arc<dyn PdClient>,
    dfs: Arc<dyn Dfs>,
    rate_limiter: Arc<IoRateLimiter>,
    master_key: MasterKey,
    security_mgr: Arc<SecurityManager>,
) -> Engines {
    let mut cfg: Config = cfg.clone();
    cfg.storage.data_dir = dir.to_str().unwrap().to_string();
    cfg.rfengine.wal_sync_dir = format!("{}/wal", dir.display());

    let rfengine_dir = dir.join("rfengine");
    let data_dir = Path::new(&cfg.storage.data_dir);
    let rf_engine = RfEngine::open(
        rfengine_dir.as_path(),
        &cfg.rfengine,
        Some(data_dir),
        Some(cfg.dfs.clone()),
    )
    .unwrap();

    let recoverer = rfstore::store::RecoverHandler::new(rf_engine.clone());
    let mut meta_iter = recoverer.clone();
    // TODO: This feature is risky when multiple stores are shut down improperly
    // if let Some(region_ids) = get_store_regions(pd.clone(), conf,
    // rf_engine.clone()) {     meta_iter.
    // set_contained_region_ids(region_ids); }
    let (_flow_ctl, store_limiter) = TikvServer::init_flow_control(&cfg.tikv);
    let (kv_engine, sender, receiver) = TikvServer::init_kv_engine(
        pd,
        &cfg,
        dfs,
        rate_limiter,
        store_limiter,
        &mut meta_iter,
        recoverer,
        false,
        master_key,
        security_mgr,
    )
    .unwrap();
    Engines::new(
        kv_engine,
        rf_engine,
        (sender, receiver),
        meta_iter.take_black_list(),
    )
}

#[derive(Default)]
struct CallbackLeakDetector {
    called: bool,
}

impl Drop for CallbackLeakDetector {
    fn drop(&mut self) {
        if self.called {
            return;
        }

        debug!("before capture");
        let bt = backtrace::Backtrace::new();
        warn!("callback is dropped"; "backtrace" => ?bt);
    }
}

pub fn check_raft_cmd_request(cmd: &RaftCmdRequest) -> bool {
    let mut is_read = cmd.has_status_request();
    let mut is_write = cmd.has_admin_request();
    if !cmd.get_requests().is_empty() {
        for req in cmd.get_requests() {
            match req.get_cmd_type() {
                CmdType::Get | CmdType::Snap | CmdType::ReadIndex => is_read = true,
                CmdType::Put | CmdType::Delete | CmdType::DeleteRange | CmdType::IngestSst => {
                    is_write = true
                }
                CmdType::Invalid | CmdType::Prewrite => panic!("Invalid RaftCmdRequest: {:?}", cmd),
            }
        }
    } else if cmd.has_custom_request() {
        is_write = true;
    }

    assert!(is_read ^ is_write, "Invalid RaftCmdRequest: {:?}", cmd);
    is_read
}

pub fn make_cb(cmd: &RaftCmdRequest) -> (Callback, future::Receiver<RaftCmdResponse>) {
    let is_read = check_raft_cmd_request(cmd);
    let (tx, rx) = future::bounded(1, future::WakePolicy::Immediately);
    let mut detector = CallbackLeakDetector::default();
    let cb = if is_read {
        Callback::Read(Box::new(move |resp: ReadResponse| {
            detector.called = true;
            // we don't care error actually.
            let _ = tx.send(resp.response);
        }))
    } else {
        Callback::write(Box::new(move |resp: WriteResponse| {
            detector.called = true;
            // we don't care error actually.
            let _ = tx.send(resp.response);
        }))
    };
    (cb, rx)
}

pub fn make_cb_ext(
    cmd: &RaftCmdRequest,
    proposed: Option<ExtCallback>,
    committed: Option<ExtCallback>,
) -> (Callback, future::Receiver<RaftCmdResponse>) {
    let (cb, receiver) = make_cb(cmd);
    if let Callback::Write { cb, .. } = cb {
        (Callback::write_ext(cb, proposed, committed), receiver)
    } else {
        (cb, receiver)
    }
}

pub fn must_get(
    engine: &kvengine::Engine,
    region_id: u64,
    cf: usize,
    key: &[u8],
    value: Option<&[u8]>,
) {
    for _ in 1..300 {
        if let Some(snapshot) = engine.get_snap_access(region_id) {
            let item = snapshot.get(cf, key, 0);
            let res = item.get_value();
            if let Some(value) = value {
                if !res.is_empty() {
                    assert_eq!(value, res);
                    return;
                }
            } else if res.is_empty() {
                return;
            }
        }

        thread::sleep(Duration::from_millis(20));
    }

    debug!("last try to get {}", log_wrappers::hex_encode_upper(key));
    let snapshot = match engine.get_snap_access(region_id) {
        Some(s) => s,
        None => {
            panic!("engine snap for {} is none.", region_id)
        }
    };
    let item = snapshot.get(cf, key, 0);
    let res = item.get_value();
    if (value.is_none() && res.is_empty()) || (value.is_some() && value.unwrap() == res) {
        return;
    }
    panic!(
        "can't get value {:?} for key {}, got: '{}'",
        value.map(escape),
        log_wrappers::hex_encode_upper(key),
        escape(res)
    )
}

pub fn must_get_equal(engine: &kvengine::Engine, region_id: u64, key: &[u8], value: &[u8]) {
    must_get(engine, region_id, WRITE_CF, key, Some(value));
}

pub fn must_get_none(engine: &kvengine::Engine, region_id: u64, key: &[u8]) {
    must_get(engine, region_id, WRITE_CF, key, None);
}

pub fn shard_must_not_exist(engine: &kvengine::Engine, region_id: u64) {
    for _ in 1..300 {
        if engine.get_snap_access(region_id).is_none() {
            return;
        }

        thread::sleep(Duration::from_millis(20));
    }
    if engine.get_snap_access(region_id).is_some() {
        panic!("engine snap for {} is not none.", region_id);
    }
}

pub fn new_put_cmd(key: &[u8], value: &[u8]) -> CustomRequest {
    let mut builder = CustomBuilder::new();
    builder.set_type(rlog::TYPE_ONE_PC);
    builder.append_one_pc(key, value, false, false, gen_ts(), gen_ts());
    builder.build()
}

pub fn new_delete_cmd(key: &[u8]) -> CustomRequest {
    new_put_cmd(key, &[])
}

pub fn new_write_request(
    region_id: u64,
    epoch: RegionEpoch,
    request: CustomRequest,
) -> RaftCmdRequest {
    let mut req = test_raftstore::new_base_request(region_id, epoch, false);
    req.set_custom_request(request);
    req
}

pub fn load_region_local_state(engine: &RfEngine, peer_id: u64) -> Option<RegionLocalState> {
    let mut state = None;
    engine.iterate_peer_states(peer_id, true, |key, val| {
        if key[0] != REGION_META_KEY_BYTE {
            return true;
        }
        let mut local_state = RegionLocalState::default();
        local_state.merge_from_bytes(val).unwrap();
        state = Some(local_state);
        false
    });
    state
}

pub fn put_till_size<T: Simulator>(
    cluster: &mut Cluster<T>,
    limit: u64,
    prefix: &str,
    range: &mut dyn Iterator<Item = u64>,
) -> Vec<u8> {
    assert!(limit > 0);
    let mut len = 0;
    let mut rng = rand::thread_rng();
    let mut key = String::new();
    let mut value = vec![0; 64];
    let mut count = 0;
    let mut key_len = 0;
    while len < limit {
        let batch_size = std::cmp::min(1024, limit - len);
        let mut builder = CustomBuilder::new();
        builder.set_type(rlog::TYPE_ONE_PC);
        for _ in 0..batch_size / 82 + 1 {
            key.clear();
            let key_id = range.next().unwrap();
            write!(key, "{}{:09}", prefix, key_id).unwrap();
            rng.fill_bytes(&mut value);
            // Encoded key match key format.
            let enc_key = Key::from_raw(key.as_bytes()).into_encoded();
            len += enc_key.len() as u64;
            key_len = enc_key.len();
            len += value.len() as u64;
            builder.append_one_pc(&enc_key, &value, false, false, 1, 2);
            count += 1;
        }
        let req = builder.build();
        cluster.put_custom(key.as_bytes(), req).unwrap();
    }
    println!(
        "put till size, key_len: {} len: {}, kv count: {}",
        key_len, len, count
    );
    key.into_bytes()
}
