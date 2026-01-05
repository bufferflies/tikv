// Copyright 2024 TiKV Project Authors. Licensed under Apache-2.0.

use std::fmt::Debug;

use strum_macros::EnumIter;

/// Represents durations for the stages of a single write operation as
/// recorded by an inspector.
#[derive(Clone, Default, Debug)]
pub struct InspectDuration {
    pub wait_duration: Option<std::time::Duration>,
    /// Duration spent in processing stages, for example flushing Raft entries
    /// to the WAL or applying write batches into the KV engine.
    pub process_duration: Option<std::time::Duration>,
}

impl InspectDuration {
    #[inline]
    pub fn sum(&self, include_wait_duration: bool) -> std::time::Duration {
        let duration = self.process_duration.unwrap_or_default();
        if include_wait_duration {
            duration + self.wait_duration.unwrap_or_default()
        } else {
            duration
        }
    }
}

#[repr(u8)]
#[derive(Debug, Clone, Copy, Eq, PartialEq, EnumIter)]
pub enum InspectFactor {
    RaftDisk = 0,
    KvDisk,
    Network, // Unimplemented
}

impl InspectFactor {
    pub fn as_str(&self) -> &str {
        match *self {
            InspectFactor::RaftDisk => "raft",
            InspectFactor::KvDisk => "kvdb",
            InspectFactor::Network => "network",
        }
    }
}

/// Utility to collect and report durations for rfstore (write) processing
/// stages. When dropped or finished explicitly the provided callback is
/// invoked with the recorded durations.
pub struct LatencyInspector {
    id: u64,
    duration: InspectDuration,
    cb: Box<dyn FnOnce(u64, InspectDuration) + Send>,
}

impl Debug for LatencyInspector {
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            fmt,
            "LatencyInspector: id {} duration: {:?}",
            self.id, self.duration
        )
    }
}

impl LatencyInspector {
    pub fn new(id: u64, cb: Box<dyn FnOnce(u64, InspectDuration) + Send>) -> Self {
        Self {
            id,
            cb,
            duration: InspectDuration::default(),
        }
    }

    pub fn record_wait_duration(&mut self, duration: std::time::Duration) {
        self.duration.wait_duration = Some(duration);
    }

    pub fn record_process_duration(&mut self, duration: std::time::Duration) {
        self.duration.process_duration = Some(duration);
    }

    /// Call the callback.
    pub fn finish(self) {
        (self.cb)(self.id, self.duration);
    }
}
