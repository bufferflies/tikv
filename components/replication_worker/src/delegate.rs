// Copyright 2025 TiKV Project Authors. Licensed under Apache-2.0.

use std::{fmt, mem};

use cdc::{CdcEvent, Conn, ConnId, Sink};
use collections::HashMap;
use kvproto::cdcpb;
use log_wrappers::Value as LogValue;
use resolved_ts::Resolver;
use tikv_util::{debug, warn};
use txn_types::TimeStamp;

use crate::{error::Result, util::build_request_range};

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
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
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

/// Information about a ChangeDataRequest.
pub(crate) struct RequestInfo {
    pub(crate) region_version: u64,
    pub(crate) start_key: Vec<u8>,
    pub(crate) end_key: Vec<u8>,
    pub(crate) resolved_ts: TimeStamp,
    pub(crate) initialized: bool,
    pub(crate) pending_events: Vec<cdcpb::Event>,
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
    pub(crate) fn add(&mut self, request: &cdcpb::ChangeDataRequest, conn_id: ConnId) {
        let request_id: RequestId = request.get_request_id().into();
        let region_version = request.get_region_epoch().get_version();
        let (start_key, end_key) = build_request_range(request);
        let request_info = RequestInfo {
            region_version,
            start_key,
            end_key,
            resolved_ts: TimeStamp::zero(),
            initialized: false,
            pending_events: vec![],
        };
        self.inner
            .insert(RequestKey::new(conn_id, request_id), request_info);
    }
}

pub(crate) enum PendingLock {
    Track { key: Vec<u8>, start_ts: TimeStamp },
    Untrack { key: Vec<u8> },
}

pub(crate) enum RegionResolver {
    Resolver(Resolver),
    Pending { locks: Vec<PendingLock>, bytes: u64 },
}

impl RegionResolver {
    pub(crate) fn new_pending() -> Self {
        RegionResolver::Pending {
            locks: vec![],
            bytes: 0,
        }
    }

    fn is_pending(&self) -> bool {
        matches!(self, RegionResolver::Pending { .. })
    }

    pub fn track_lock(&mut self, start_ts: TimeStamp, key: Vec<u8>) -> Result<()> {
        match self {
            RegionResolver::Resolver(resolver) => resolver.track_lock(start_ts, key, None),
            RegionResolver::Pending { locks, bytes } => {
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
            RegionResolver::Pending { locks, bytes } => {
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
    pub(crate) keyspace_id: u32,
    pub(crate) region_id: u64,
    pub(crate) requests: RegionRequests,
    pub(crate) resolver: Option<RegionResolver>,
}

impl RegionDelegate {
    pub(crate) fn new(keyspace_id: u32, region_id: u64) -> Self {
        Self {
            keyspace_id,
            region_id,
            requests: RegionRequests::default(),
            resolver: None,
        }
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
                warn!("{} failed to send error event", self.region_id;
                    "keyspace" => self.keyspace_id,
                    "conn" => ?req_key.conn_id,
                    "request" => %req_key.request_id,
                    "err" => ?err);
            }
        }
    }

    pub(crate) fn handle_scan_locks(&mut self, locks: Vec<(Vec<u8>, TimeStamp)>) {
        let pending_resolver = self.resolver.take().unwrap_or_else(|| {
            panic!("{} handle_scan_locks: resolver is None", self.region_id);
        });
        if !pending_resolver.is_pending() {
            panic!("{} handle_scan_locks: resolver not pending", self.region_id);
        }
        let resolver = pending_resolver.to_resolver(self.region_id, locks);
        self.resolver = Some(resolver);
    }

    pub(crate) fn unsubscribe(
        &mut self,
        conn_id: ConnId,
        request_id: RequestId,
        sink: Option<&Sink>,
    ) {
        if self
            .requests
            .remove(&RequestKey::new(conn_id, request_id))
            .is_none()
        {
            return;
        }

        if let Some(sink) = sink {
            let mut err_event = cdcpb::Error::new();
            err_event
                .mut_region_not_found()
                .set_region_id(self.region_id);
            let event = cdcpb::Event {
                region_id: self.region_id,
                request_id: request_id.into_inner(),
                event: Some(cdcpb::Event_oneof_event::Error(err_event)),
                ..Default::default()
            };
            if let Err(e) = sink.unbounded_send(CdcEvent::Event(event), true) {
                warn!("{} unsubscribe: send event failed", self.region_id; "request" => %request_id, "err" => ?e);
            }
        }
    }
}
