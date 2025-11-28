// Copyright 2021 TiKV Project Authors. Licensed under Apache-2.0.

use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc,
};

use crossbeam::atomic::AtomicCell;
use fail::fail_point;
use kvproto::{
    kvrpcpb::ExtraOp as TxnExtraOp,
    metapb,
    raft_cmdpb::{CmdType, RaftCmdRequest, RaftCmdResponse, ReadIndexResponse, Request, Response},
};
use pd_client::BucketMeta;
use raftstore::store::{
    util::{LeaseState, RemoteLease},
    worker_metrics::*,
    TxnExt,
};
use tikv_util::{
    debug, error,
    lru::LruCache,
    time::{monotonic_raw_now, ThreadReadId},
};
use time::Timespec;
use trace_event::types::TraceContext;

use crate::{
    store::{
        cf_name_to_num, cmd_resp, util, Callback, Peer, PeerMsg, RaftCommand, ReadResponse,
        RegionSnapshot, RequestInspector, RequestPolicy,
    },
    Error, RaftRouter, Result,
};

#[derive(Debug)]
pub enum ReadProgress {
    Region(metapb::Region),
    Term(u64),
    AppliedIndexTerm(u64),
    LeaderLease(RemoteLease),
    RegionBuckets(Arc<BucketMeta>),
}

impl ReadProgress {
    pub fn region(region: metapb::Region) -> ReadProgress {
        ReadProgress::Region(region)
    }

    pub fn term(term: u64) -> ReadProgress {
        ReadProgress::Term(term)
    }

    pub fn applied_index_term(applied_index_term: u64) -> ReadProgress {
        ReadProgress::AppliedIndexTerm(applied_index_term)
    }

    pub fn leader_lease(lease: RemoteLease) -> ReadProgress {
        ReadProgress::LeaderLease(lease)
    }

    pub fn region_buckets(bucket_meta: Arc<BucketMeta>) -> ReadProgress {
        ReadProgress::RegionBuckets(bucket_meta)
    }
}

pub trait ReadExecutor {
    fn get_snapshot(&self, region_id: u64, region_ver: u64) -> Result<RegionSnapshot>;
    fn get_value(&self, req: &Request, region: &metapb::Region) -> Result<Response> {
        let key = req.get_get().get_key();
        // region key range has no data prefix, so we must use origin key to check.
        util::check_key_in_region(key, region)?;

        let region_snap =
            self.get_snapshot(region.get_id(), region.get_region_epoch().get_version())?;
        let cf_num = cf_name_to_num(req.get_get().get_cf());
        let item = region_snap.snap.get(cf_num, key, u64::MAX);
        let mut resp = Response::default();
        if item.value_len() > 0 {
            resp.mut_get().set_value(item.get_value().to_vec());
        }
        Ok(resp)
    }

    fn execute(
        &mut self,
        msg: &RaftCmdRequest,
        region: &Arc<metapb::Region>,
        read_index: Option<u64>,
        _ts: Option<ThreadReadId>,
    ) -> ReadResponse {
        let requests = msg.get_requests();
        let mut response = ReadResponse {
            response: RaftCmdResponse::default(),
            snapshot: None,
            txn_extra_op: TxnExtraOp::Noop,
        };
        let mut responses = Vec::with_capacity(requests.len());
        for req in requests {
            let cmd_type = req.get_cmd_type();
            let mut resp = match cmd_type {
                CmdType::Get => match self.get_value(req, region.as_ref()) {
                    Ok(resp) => resp,
                    Err(e) => {
                        error!(?e;
                            "failed to execute get command";
                            "region_id" => region.get_id(),
                        );
                        response.response = cmd_resp::new_error(e);
                        return response;
                    }
                },
                CmdType::Snap => {
                    match self
                        .get_snapshot(region.get_id(), region.get_region_epoch().get_version())
                    {
                        Ok(snapshot) => {
                            response.snapshot = Some(snapshot);
                            Response::default()
                        }
                        Err(e) => {
                            response.response = cmd_resp::new_error(e);
                            return response;
                        }
                    }
                }
                CmdType::ReadIndex => {
                    let mut resp = Response::default();
                    if let Some(read_index) = read_index {
                        let mut res = ReadIndexResponse::default();
                        res.set_read_index(read_index);
                        resp.set_read_index(res);
                    } else {
                        panic!("[region {}] can not get readindex", region.get_id());
                    }
                    resp
                }
                CmdType::Prewrite
                | CmdType::Put
                | CmdType::Delete
                | CmdType::DeleteRange
                | CmdType::IngestSst
                | CmdType::Invalid => unreachable!(),
            };
            resp.set_cmd_type(cmd_type);
            responses.push(resp);
        }
        response.response.set_responses(responses.into());
        response
    }
}

/// A read only delegate of `Peer`.
#[derive(Clone, Debug)]
pub struct ReadDelegate {
    pub region: Arc<metapb::Region>,
    pub peer_id: u64,
    pub store_id: u64,
    pub term: u64,
    pub applied_index_term: u64,
    pub leader_lease: Option<RemoteLease>,
    pub last_valid_ts: Timespec,

    pub tag: String,
    pub txn_ext: Arc<TxnExt>,
    pub bucket_meta: Option<Arc<BucketMeta>>,
    // TODO: finish txn_extra_op
    pub txn_extra_op: Arc<AtomicCell<TxnExtraOp>>,

    // `track_ver` used to keep the local `ReadDelegate` in `LocalReader`
    // up-to-date with the global `ReadDelegate` stored at `StoreMeta`
    pub track_ver: TrackVer,
}

impl ReadDelegate {
    pub(crate) fn from_peer(peer: &Peer) -> ReadDelegate {
        let region = peer.region().clone();
        let region_id = region.get_id();
        let peer_id = peer.peer.get_id();
        let store_id = peer.peer.get_store_id();
        ReadDelegate {
            region: Arc::new(region),
            peer_id,
            store_id,
            term: peer.term(),
            applied_index_term: peer.get_store().applied_index_term(),
            leader_lease: None,
            last_valid_ts: Timespec::new(0, 0),
            tag: format!("[region {}] {}", region_id, peer_id),
            txn_ext: peer.txn_ext.clone(),
            bucket_meta: peer.buckets.as_ref().map(|b| b.meta.clone()),
            txn_extra_op: peer.txn_extra_op.clone(),
            track_ver: TrackVer::new(),
        }
    }

    fn fresh_valid_ts(&mut self) {
        self.last_valid_ts = monotonic_raw_now();
    }

    pub fn update(&mut self, progress: ReadProgress) {
        self.fresh_valid_ts();
        self.track_ver.inc();
        match progress {
            ReadProgress::Region(region) => {
                self.region = Arc::new(region);
            }
            ReadProgress::Term(term) => {
                self.term = term;
            }
            ReadProgress::AppliedIndexTerm(applied_index_term) => {
                self.applied_index_term = applied_index_term;
            }
            ReadProgress::LeaderLease(leader_lease) => {
                self.leader_lease = Some(leader_lease);
            }
            ReadProgress::RegionBuckets(bucket_meta) => {
                self.bucket_meta = Some(bucket_meta);
            }
        }
    }

    fn is_in_leader_lease(&self, ts: Timespec) -> bool {
        if let Some(ref lease) = self.leader_lease {
            let term = lease.term();
            if term == self.term {
                if lease.inspect(Some(ts)) == LeaseState::Valid {
                    return true;
                } else {
                    TLS_LOCAL_READ_METRICS
                        .with(|m| m.borrow_mut().reject_reason.lease_expire.inc());
                    debug!("rejected by lease expire"; "tag" => &self.tag);
                }
            } else {
                TLS_LOCAL_READ_METRICS.with(|m| m.borrow_mut().reject_reason.term_mismatch.inc());
                debug!("rejected by term mismatch"; "tag" => &self.tag);
            }
        }
        false
    }

    /// Used in some external tests.
    pub fn mock(region_id: u64) -> Self {
        let mut region: metapb::Region = Default::default();
        region.set_id(region_id);
        ReadDelegate {
            region: Arc::new(region),
            peer_id: 1,
            store_id: 1,
            term: 1,
            applied_index_term: 1,
            leader_lease: None,
            last_valid_ts: Timespec::new(0, 0),
            tag: format!("[region {}] {}", region_id, 1),
            txn_extra_op: Default::default(),
            txn_ext: Default::default(),
            track_ver: TrackVer::new(),
            bucket_meta: None,
        }
    }
}

pub struct LocalReader {
    store_readers: Arc<dashmap::DashMap<u64, ReadDelegate>>,
    kv_engine: kvengine::Engine,
    // region id -> ReadDelegate
    // The use of `Arc` here is a workaround, see the comment at `get_delegate`
    delegates: LruCache<u64, Arc<ReadDelegate>>,
    // A channel to raftstore.
    router: RaftRouter,
}

impl ReadExecutor for LocalReader {
    fn get_snapshot(&self, region_id: u64, region_ver: u64) -> Result<RegionSnapshot> {
        if let Some(snap) = self.kv_engine.get_snap_access(region_id) {
            if snap.get_version() == region_ver {
                let extra_op = self
                    .delegates
                    .get_no_promote(&region_id)
                    .map(|d| d.txn_extra_op.load())
                    .unwrap_or(TxnExtraOp::Noop);
                let mut snapshot = RegionSnapshot::from_snapshot(snap, None);
                snapshot.txn_extra_op = extra_op;
                return Ok(snapshot);
            }
        }
        Err(Error::StaleCommand)
    }
}

impl LocalReader {
    pub fn new(
        kv_engine: kvengine::Engine,
        store_readers: Arc<dashmap::DashMap<u64, ReadDelegate>>,
        router: RaftRouter,
    ) -> Self {
        LocalReader {
            store_readers,
            kv_engine,
            router,
            delegates: LruCache::with_capacity_and_sample(0, 7),
        }
    }

    fn redirect(&mut self, cmd: RaftCommand) {
        debug!("localreader redirects command"; "command" => ?cmd);
        let region_id = cmd.request.get_header().get_region_id();
        self.router.send(region_id, PeerMsg::RaftCommand(cmd));
    }

    // Ideally `get_delegate` should return `Option<&ReadDelegate>`, but if so the
    // lifetime of the returned `&ReadDelegate` will bind to `self`, and make it
    // impossible to use `&mut self` while the `&ReadDelegate` is alive, a
    // better choice is use `Rc` but `LocalReader: Send` will be violated, which
    // is required by `LocalReadRouter: Send`, use `Arc` will introduce extra cost
    // but make the logic clear
    fn get_delegate(&mut self, region_id: u64) -> Option<Arc<ReadDelegate>> {
        match self.delegates.get(&region_id) {
            // The local `ReadDelegate` is up to date
            Some(d) if !d.track_ver.any_new() => Some(Arc::clone(d)),
            _ => {
                debug!("update local read delegate"; "region_id" => region_id);
                TLS_LOCAL_READ_METRICS.with(|m| m.borrow_mut().reject_reason.cache_miss.inc());
                let meta_len = self.store_readers.len();
                let meta_reader = self
                    .store_readers
                    .get(&region_id)
                    .map(|reader| Arc::new(reader.value().clone()));
                // Remove the stale delegate
                self.delegates.remove(&region_id);
                self.delegates.resize(meta_len);
                match meta_reader {
                    Some(reader) => {
                        self.delegates.insert(region_id, Arc::clone(&reader));
                        Some(reader)
                    }
                    None => None,
                }
            }
        }
    }

    fn pre_propose_raft_command(
        &mut self,
        req: &RaftCmdRequest,
    ) -> Result<Option<(Arc<ReadDelegate>, RequestPolicy)>> {
        // Check region id.
        let region_id = req.get_header().get_region_id();
        let delegate = match self.get_delegate(region_id) {
            Some(d) => d,
            None => {
                TLS_LOCAL_READ_METRICS.with(|m| m.borrow_mut().reject_reason.no_region.inc());
                debug!("rejected by no region"; "region_id" => region_id);
                return Ok(None);
            }
        };

        if let Err(e) = util::check_store_id(req, delegate.store_id) {
            TLS_LOCAL_READ_METRICS.with(|m| m.borrow_mut().reject_reason.store_id_mismatch.inc());
            debug!("rejected by store id not match"; "err" => %e);
            return Err(e);
        }

        fail_point!("localreader_on_find_delegate");

        // Check peer id.
        if let Err(e) = util::check_peer_id(req, delegate.peer_id) {
            TLS_LOCAL_READ_METRICS.with(|m| m.borrow_mut().reject_reason.peer_id_mismatch.inc());
            return Err(e);
        }

        // Check term.
        if let Err(e) = util::check_term(req, delegate.term) {
            debug!(
                "check term";
                "delegate_term" => delegate.term,
                "header_term" => req.get_header().get_term(),
            );
            TLS_LOCAL_READ_METRICS.with(|m| m.borrow_mut().reject_reason.term_mismatch.inc());
            return Err(e);
        }

        // Check region epoch.
        if util::check_region_epoch(req, &delegate.region, false).is_err() {
            TLS_LOCAL_READ_METRICS.with(|m| m.borrow_mut().reject_reason.epoch.inc());
            // Stale epoch, redirect it to raftstore to get the latest region.
            let delegate_ver = delegate.region.get_region_epoch().get_version();
            debug!("rejected by epoch not match req {:?}", req; "delegate_ver" => delegate_ver);
            return Ok(None);
        }

        let mut inspector = Inspector {
            delegate: &delegate,
        };
        match inspector.inspect(req) {
            RequestPolicy::ReadLocal => Ok(Some((delegate, RequestPolicy::ReadLocal))),
            // It can not handle other policies.
            _ => Ok(None),
        }
    }

    pub fn propose_raft_command(
        &mut self,
        mut read_id: Option<ThreadReadId>,
        req: RaftCmdRequest,
        trace_ctx: TraceContext,
        cb: Callback,
    ) {
        match self.pre_propose_raft_command(&req) {
            Ok(Some((delegate, policy))) => {
                let mut response = match policy {
                    // Leader can read local if and only if it is in lease.
                    RequestPolicy::ReadLocal => {
                        let snapshot_ts = match read_id.as_mut() {
                            // If this peer became Leader not long ago and just after the cached
                            // snapshot was created, this snapshot can not see all data of the peer.
                            Some(id) => {
                                if id.create_time <= delegate.last_valid_ts {
                                    id.create_time = monotonic_raw_now();
                                }
                                id.create_time
                            }
                            None => monotonic_raw_now(),
                        };
                        if !delegate.is_in_leader_lease(snapshot_ts) {
                            // Forward to raftstore.
                            self.redirect(RaftCommand::new(req, trace_ctx, cb));
                            return;
                        }
                        self.execute(&req, &delegate.region, None, read_id)
                    }
                    _ => unreachable!(),
                };
                response.txn_extra_op = delegate.txn_extra_op.load();
                cmd_resp::bind_term(&mut response.response, delegate.term);
                if let Some(snap) = response.snapshot.as_mut() {
                    snap.txn_ext = Some(delegate.txn_ext.clone());
                    snap.txn_extra_op = response.txn_extra_op;
                    snap.bucket_meta = delegate.bucket_meta.clone();
                }
                cb.invoke_read(response);
            }
            // Forward to raftstore.
            Ok(None) => self.redirect(RaftCommand::new(req, trace_ctx, cb)),
            Err(e) => {
                let mut response = cmd_resp::new_error(e);
                if let Some(delegate) = self.delegates.get(&req.get_header().get_region_id()) {
                    cmd_resp::bind_term(&mut response, delegate.term);
                }
                cb.invoke_read(ReadResponse {
                    response,
                    snapshot: None,
                    txn_extra_op: TxnExtraOp::Noop,
                });
            }
        }
    }

    /// If read requests are received at the same RPC request, we can create one
    /// snapshot for all of them and check whether the time when the
    /// snapshot was created is in lease. We use ThreadReadId to figure out
    /// whether this RaftCommand comes from the same RPC request with
    /// the last RaftCommand which left a snapshot cached in LocalReader.
    /// ThreadReadId is composed by thread_id and a thread_local incremental
    /// sequence.
    #[inline]
    pub fn read(&mut self, read_id: Option<ThreadReadId>, req: RaftCmdRequest, cb: Callback) {
        // TODO: Pass a proper trace context here.
        self.propose_raft_command(read_id, req, TraceContext::default(), cb);
        maybe_tls_local_read_metrics_flush();
    }
}

impl Clone for LocalReader {
    fn clone(&self) -> Self {
        LocalReader {
            store_readers: self.store_readers.clone(),
            kv_engine: self.kv_engine.clone(),
            router: self.router.clone(),
            delegates: LruCache::with_capacity_and_sample(0, 7),
        }
    }
}

struct Inspector<'r> {
    delegate: &'r ReadDelegate,
}

impl<'r> RequestInspector for Inspector<'r> {
    fn has_applied_to_current_term(&mut self) -> bool {
        if self.delegate.applied_index_term == self.delegate.term {
            true
        } else {
            debug!(
                "rejected by term check";
                "tag" => &self.delegate.tag,
                "applied_index_term" => self.delegate.applied_index_term,
                "delegate_term" => ?self.delegate.term,
            );

            // only for metric.
            TLS_LOCAL_READ_METRICS.with(|m| m.borrow_mut().reject_reason.applied_term.inc());
            false
        }
    }

    fn inspect_lease(&mut self) -> LeaseState {
        // TODO: disable localreader if we did not enable raft's check_quorum.
        if self.delegate.leader_lease.is_some() {
            // We skip lease check, because it is postponed until `handle_read`.
            LeaseState::Valid
        } else {
            debug!("rejected by leader lease"; "tag" => &self.delegate.tag);
            TLS_LOCAL_READ_METRICS.with(|m| m.borrow_mut().reject_reason.no_lease.inc());
            LeaseState::Expired
        }
    }
}

#[derive(Debug)]
pub struct TrackVer {
    version: Arc<AtomicU64>,
    local_ver: u64,
    // source set to `true` means the `TrackVer` is created by `TrackVer::new` instead
    // of `TrackVer::clone`, more specific, only the `ReadDelegate` created by `ReadDelegate::new`
    // will have source `TrackVer` and be able to increase `TrackVer::version`, because these
    // `ReadDelegate` are store at `StoreMeta` and only them will invoke `ReadDelegate::update`
    source: bool,
}

impl TrackVer {
    pub fn new() -> TrackVer {
        TrackVer {
            version: Arc::new(AtomicU64::from(0)),
            local_ver: 0,
            source: true,
        }
    }

    // Take `&mut self` to prevent calling `inc` and `clone` at the same time
    fn inc(&mut self) {
        // Only the source `TrackVer` can increase version
        if self.source {
            self.version.fetch_add(1, Ordering::Relaxed);
        }
    }

    fn any_new(&self) -> bool {
        self.version.load(Ordering::Relaxed) > self.local_ver
    }
}

impl Default for TrackVer {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for TrackVer {
    fn clone(&self) -> Self {
        TrackVer {
            version: Arc::clone(&self.version),
            local_ver: self.version.load(Ordering::Relaxed),
            source: false,
        }
    }
}

impl Drop for ReadDelegate {
    fn drop(&mut self) {
        // call `inc` to notify the source `ReadDelegate` is dropped
        self.track_ver.inc();
    }
}
