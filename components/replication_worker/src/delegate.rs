// Copyright 2025 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    fmt, mem,
    result::Result as StdResult,
    sync::atomic::{AtomicU64, Ordering},
};

use cdc::{CdcEvent, Conn, ConnId, Sink};
use collections::HashMap;
use kvengine::{table::SnapVersion, IdVer, ShardTag};
use kvproto::cdcpb;
use log_wrappers::Value as LogValue;
use protobuf::Message;
use resolved_ts::Resolver;
use tikv_util::{debug, error, info, warn};
use txn_types::TimeStamp;

use crate::{error::Result, util::build_request_range, Error};

/// An identifier of a ChangeDataRequest.
///
/// - ChangeDataRequest from the same changefeed on different connections or
///   different regions has the same RequestId.
/// - Different ChangeDataRequest on different TiCDC instances can have the same
///   RequestId.
#[derive(Clone, Copy, Eq, PartialEq, Hash)]
pub struct RequestId(u64);

impl RequestId {
    pub fn into_inner(self) -> u64 {
        self.0
    }
}

impl From<u64> for RequestId {
    fn from(id: u64) -> Self {
        RequestId(id)
    }
}

impl fmt::Debug for RequestId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl fmt::Display for RequestId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self, f)
    }
}

// Different connection (from different TiCDC instances) can have the same
// RequestId. So use (ConnId, RequestId) to make it unique.
#[derive(Clone, Copy, Eq, PartialEq, Hash)]
pub struct RequestKey {
    pub conn_id: ConnId,
    pub request_id: RequestId,
}

impl RequestKey {
    pub fn new(conn_id: ConnId, request_id: RequestId) -> Self {
        Self {
            conn_id,
            request_id,
        }
    }
}

impl fmt::Debug for RequestKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Request")
            .field("conn", &self.conn_id.into_inner())
            .field("request", &self.request_id)
            .finish()
    }
}

pub(crate) enum RequestState {
    Initializing {
        events: Vec<cdcpb::Event>,
        bytes: u64,

        /// The unique ID to identify an initialization.
        ///
        /// Other fields (e.g., snap_version + checkpoint_ts) is not safe enough
        /// when the sink is failed and the initialization is retried, and
        /// previous initialization is still running.
        init_id: u64,
    },
    Initialized,
}

impl fmt::Debug for RequestState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RequestState::Initializing {
                events,
                bytes,
                init_id,
            } => f
                .debug_struct("Initializing")
                .field("events", &events.len())
                .field("bytes", bytes)
                .field("init_id", init_id)
                .finish(),
            RequestState::Initialized => write!(f, "Initialized"),
        }
    }
}

impl RequestState {
    pub(crate) fn is_initializing_with_id(&self, req_init_id: u64) -> bool {
        match self {
            RequestState::Initializing { init_id, .. } => *init_id == req_init_id,
            _ => false,
        }
    }

    pub(crate) fn is_initialized(&self) -> bool {
        matches!(self, RequestState::Initialized)
    }

    pub(crate) fn must_push_pending(&mut self, event: cdcpb::Event) {
        match self {
            Self::Initializing { events, bytes, .. } => {
                *bytes = bytes.saturating_add(event.compute_size() as u64);
                events.push(event);
            }
            Self::Initialized => unreachable!(),
        }
    }

    pub(crate) fn must_finish_initialize(&mut self) -> (Vec<cdcpb::Event>, u64 /* bytes */) {
        match mem::replace(self, RequestState::Initialized) {
            RequestState::Initializing { events, bytes, .. } => (events, bytes),
            RequestState::Initialized => unreachable!(),
        }
    }
}

/// Information about a ChangeDataRequest.
pub(crate) struct RequestInfo {
    pub(crate) start_key: Vec<u8>,
    pub(crate) end_key: Vec<u8>,
    pub(crate) resolved_ts: TimeStamp,
    pub(crate) state: RequestState,
}

impl RequestInfo {
    pub(crate) fn in_range(&self, key: &[u8]) -> bool {
        key >= self.start_key.as_slice() && key < self.end_key.as_slice()
    }
}

#[derive(Default)]
pub(crate) struct RegionRequests {
    inner: HashMap<RequestKey, RequestInfo>,
}

impl std::ops::Deref for RegionRequests {
    type Target = HashMap<RequestKey, RequestInfo>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl std::ops::DerefMut for RegionRequests {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl RegionRequests {
    pub(crate) fn add(
        &mut self,
        request: &cdcpb::ChangeDataRequest,
        conn_id: ConnId,
    ) -> StdResult<u64 /* init_id */, String /* current_state */> {
        lazy_static::lazy_static! {
            static ref INIT_ID_ALLOC: AtomicU64 = AtomicU64::new(1);
        }

        let request_id: RequestId = request.get_request_id().into();
        let request_key = RequestKey::new(conn_id, request_id);
        if let Some(request_info) = self.inner.get(&request_key) {
            // Return string to work around for borrow check.
            return Err(format!("{:?}", request_info.state));
        }

        let (start_key, end_key) = build_request_range(request);
        let init_id = INIT_ID_ALLOC.fetch_add(1, Ordering::Relaxed);
        let request_info = RequestInfo {
            start_key,
            end_key,
            resolved_ts: TimeStamp::zero(),
            state: RequestState::Initializing {
                events: vec![],
                bytes: 0,
                init_id,
            },
        };
        let old = self.inner.insert(request_key, request_info);
        debug_assert!(old.is_none());
        Ok(init_id)
    }

    pub(crate) fn check_duplicated(
        &self,
        request: &cdcpb::ChangeDataRequest,
        conn_id: ConnId,
    ) -> Result<()> {
        let request_id: RequestId = request.get_request_id().into();
        let request_key = RequestKey::new(conn_id, request_id);
        if let Some(request_info) = self.inner.get(&request_key) {
            debug!("duplicated request";
                "conn" => ?conn_id, "request" => %request_id, "current" => ?request_info.state);
            Err(Error::DuplicatedRegister {
                conn_id,
                request_id,
            })
        } else {
            Ok(())
        }
    }
}

pub(crate) enum PendingLock {
    Track { key: Vec<u8>, start_ts: TimeStamp },
    Untrack { key: Vec<u8> },
}

pub(crate) enum RegionResolver {
    Resolver(Resolver),
    Pending {
        locks: Vec<PendingLock>,
        bytes: u64,
        snap_version: SnapVersion,
    },
}

impl RegionResolver {
    pub(crate) fn new_pending(snap_version: SnapVersion) -> Self {
        RegionResolver::Pending {
            locks: vec![],
            bytes: 0,
            snap_version,
        }
    }

    fn is_pending_with_snap_version(&self, expect_snap_version: SnapVersion) -> bool {
        match self {
            RegionResolver::Pending { snap_version, .. } => *snap_version == expect_snap_version,
            _ => false,
        }
    }

    pub fn track_lock(&mut self, start_ts: TimeStamp, key: Vec<u8>) -> Result<()> {
        match self {
            RegionResolver::Resolver(resolver) => resolver.track_lock(start_ts, key, None),
            RegionResolver::Pending { locks, bytes, .. } => {
                // TODO: handle OOM.
                *bytes =
                    bytes.saturating_add(key.len() as u64 + mem::size_of::<TimeStamp>() as u64);
                locks.push(PendingLock::Track { key, start_ts });
            }
        }
        Ok(())
    }

    pub fn untrack_lock(&mut self, key: &[u8]) -> Result<()> {
        match self {
            RegionResolver::Resolver(resolver) => resolver.untrack_lock(key, None),
            RegionResolver::Pending { locks, bytes, .. } => {
                *bytes = bytes.saturating_add(key.len() as u64);
                locks.push(PendingLock::Untrack { key: key.to_vec() });
            }
        }
        Ok(())
    }

    pub fn resolve(&mut self, min_ts: TimeStamp) -> Option<TimeStamp> {
        match self {
            RegionResolver::Resolver(resolver) => Some(resolver.resolve(min_ts)),
            RegionResolver::Pending { .. } => None,
        }
    }

    #[allow(dead_code)]
    pub fn resolved_ts(&self) -> Option<TimeStamp> {
        match self {
            RegionResolver::Resolver(resolver) => Some(resolver.resolved_ts()),
            RegionResolver::Pending { .. } => None,
        }
    }

    pub fn to_resolver(self, region_id: u64, tracked_locks: Vec<(Vec<u8>, TimeStamp)>) -> Self {
        let Self::Pending { locks, .. } = self else {
            unreachable!();
        };

        let mut resolver = Resolver::new(region_id);
        for (key, ts) in tracked_locks {
            debug!("{} to_resolver: track lock", region_id; "key" => LogValue::key(&key), "ts" => ts);
            resolver.track_lock(ts, key, None);
        }

        for lock in locks {
            match lock {
                PendingLock::Track { key, start_ts } => {
                    debug!("{} to_resolver: track lock", region_id; "key" => LogValue::key(&key), "ts" => start_ts);
                    resolver.track_lock(start_ts, key, None);
                }
                PendingLock::Untrack { key } => {
                    debug!("{} to_resolver: untrack lock", region_id; "key" => LogValue::key(&key));
                    resolver.untrack_lock(&key, None);
                }
            }
        }

        Self::Resolver(resolver)
    }
}

/// A CDC delegate of a region.
pub(crate) struct RegionDelegate {
    pub(crate) merged_store_id: u64,
    pub(crate) region_id: u64,
    pub(crate) region_ver: u64,
    pub(crate) requests: RegionRequests,
    pub(crate) resolver: Option<RegionResolver>,
}

impl RegionDelegate {
    pub(crate) fn new(merged_store_id: u64, region_id: u64, region_ver: u64) -> Self {
        Self {
            merged_store_id,
            region_id,
            region_ver,
            requests: RegionRequests::default(),
            resolver: None,
        }
    }

    fn tag(&self) -> ShardTag {
        ShardTag::new(self.merged_store_id, IdVer::new(self.region_id, 0))
    }

    pub(crate) fn broadcast_error(&self, error: cdcpb::Error, conns: &HashMap<ConnId, Conn>) {
        let mut event = cdcpb::Event {
            region_id: self.region_id,
            ..Default::default()
        };
        for req_key in self.requests.keys() {
            let Some(conn) = conns.get(&req_key.conn_id) else {
                continue;
            };

            event.set_request_id(req_key.request_id.into_inner());
            event.set_error(error.clone());
            if let Err(err) = conn
                .get_sink()
                .unbounded_send(CdcEvent::Event(event.clone()), true)
            {
                warn!("{} failed to send error event", self.tag();
                    "conn" => ?req_key.conn_id,
                    "request" => %req_key.request_id,
                    "err" => ?err);
            }
        }
    }

    pub(crate) fn handle_scan_locks(
        &mut self,
        locks: Vec<(Vec<u8>, TimeStamp)>,
        snap_version: SnapVersion,
    ) {
        let Some(cur_resolver) = self.resolver.take() else {
            // For the case when first scan failed then receive a second one.
            // As pending locks are dropped, the result can not be used.
            warn!("{} handle_scan_locks: drop scan result", self.tag());
            return;
        };
        if cur_resolver.is_pending_with_snap_version(snap_version) {
            let resolver = cur_resolver.to_resolver(self.region_id, locks);
            self.resolver = Some(resolver);
        } else {
            info!("{} handle_scan_locks: drop stale scan result", self.tag(); "snap_version" => snap_version);
        }
    }

    pub(crate) fn handle_scan_locks_error(
        &mut self,
        err: &Error,
        snap_version: SnapVersion,
        conns: &HashMap<ConnId, Conn>,
    ) {
        if self
            .resolver
            .as_ref()
            .is_some_and(|r| r.is_pending_with_snap_version(snap_version))
        {
            error!("{} handle_scan_locks: failed", self.tag(); "err" => ?err);
            self.resolver = None;

            let mut cdc_err = cdcpb::Error::default();
            cdc_err
                .mut_server_is_busy()
                .set_reason(format!("scan locks failed: {:?}", err));
            self.broadcast_error(cdc_err, conns);
        } else {
            info!("{} handle_scan_locks: drop stale scan error", self.tag(); "snap_version" => snap_version, "err" => ?err);
        }
    }

    pub(crate) fn unsubscribe(
        &mut self,
        conn_id: ConnId,
        request_id: RequestId,
        sink: Option<&Sink>,
        err_event: Option<cdcpb::Error>,
    ) {
        if self
            .requests
            .remove(&RequestKey::new(conn_id, request_id))
            .is_none()
        {
            return;
        }

        if let Some(sink) = sink {
            let err_event = err_event.unwrap_or_else(|| {
                let mut e = cdcpb::Error::new();
                e.mut_region_not_found().set_region_id(self.region_id);
                e
            });
            let event = cdcpb::Event {
                region_id: self.region_id,
                request_id: request_id.into_inner(),
                event: Some(cdcpb::Event_oneof_event::Error(err_event)),
                ..Default::default()
            };
            if let Err(e) = sink.unbounded_send(CdcEvent::Event(event), true) {
                warn!("{} unsubscribe: send event failed", self.tag(); "request" => %request_id, "err" => ?e);
            }
        }
    }
}
