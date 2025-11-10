// Copyright 2022 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    cell::Cell,
    future::Future,
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
};

use pin_project::pin_project;
use slog::{Record, Serializer, Value};

use crate::{slab::TrackerToken, Tracker, GLOBAL_TRACKERS, INVALID_TRACKER_TOKEN};

#[derive(Clone, Default)]
pub struct TraceId(pub Option<Arc<[u8]>>, pub u64);

impl TraceId {
    pub fn new(trace_id: &[u8], control_flags: u64) -> TraceId {
        if !trace_id.is_empty() {
            TraceId(Some(Arc::from(trace_id)), control_flags)
        } else {
            TraceId(None, control_flags)
        }
    }

    pub fn control_flags(&self) -> u64 {
        self.1
    }
}

impl Value for TraceId {
    fn serialize(
        &self,
        _record: &Record<'_>,
        key: slog::Key,
        serializer: &mut dyn Serializer,
    ) -> slog::Result {
        match &self.0 {
            Some(arc) => serializer.emit_str(key, &hex::encode(arc)),
            None => serializer.emit_str(key, "None"),
        }
    }
}

thread_local! {
    static TLS_TRACKER_TOKEN: Cell<TrackerToken> = Cell::new(INVALID_TRACKER_TOKEN);
    static TLS_TRACE_ID: Cell<TraceId> = Cell::new(TraceId(None, 0));
}

pub fn set_tls_tracker_token(token: TrackerToken) {
    TLS_TRACKER_TOKEN.with(|c| {
        c.set(token);
    })
}

pub fn set_tls_trace_id(v: TraceId) {
    TLS_TRACE_ID.with(|c| c.set(v))
}

pub fn clear_tls_tracker_token() {
    set_tls_tracker_token(INVALID_TRACKER_TOKEN);
}

pub fn get_tls_tracker_token() -> TrackerToken {
    TLS_TRACKER_TOKEN.with(|c| c.get())
}

pub fn get_tls_trace_id() -> TraceId {
    TLS_TRACE_ID.with(|c| {
        let v = c.take();
        let ret = v.clone();
        c.set(v);
        ret
    })
}

pub fn with_tls_tracker<F>(mut f: F)
where
    F: FnMut(&mut Tracker),
{
    TLS_TRACKER_TOKEN.with(|c| {
        GLOBAL_TRACKERS.with_tracker(c.get(), &mut f);
    });
}

#[pin_project]
pub struct TrackedFuture<F> {
    #[pin]
    future: F,
    tracker: TrackerToken,
    trace_id: TraceId,
}

impl<F> TrackedFuture<F> {
    pub fn new(future: F) -> TrackedFuture<F> {
        TrackedFuture {
            future,
            tracker: get_tls_tracker_token(),
            trace_id: get_tls_trace_id(),
        }
    }
}

impl<F: Future> Future for TrackedFuture<F> {
    type Output = F::Output;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.project();
        TLS_TRACKER_TOKEN.with(|c| {
            c.set(*this.tracker);
        });
        TLS_TRACE_ID.with(|c| {
            c.set(this.trace_id.clone());
        });

        let res = this.future.poll(cx);

        TLS_TRACKER_TOKEN.with(|c| c.set(INVALID_TRACKER_TOKEN));
        TLS_TRACE_ID.with(|c| c.set(TraceId(None, 0)));

        res
    }
}
