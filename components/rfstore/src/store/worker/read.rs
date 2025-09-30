// Copyright 2025 TiKV Project Authors. Licensed under Apache-2.0.

use std::fmt;

use raft::GetEntriesContext;
use rfengine::RfEngine;
use tikv_util::worker::Runnable;

use crate::store::peer_storage::RaftlogFetchResult;

/// Task for async raft log reading.
pub enum ReadTask {
    FetchLogs {
        region_id: u64,
        peer_id: u64,
        context: GetEntriesContext,
        low: u64,
        high: u64,
        max_size: usize,
        tried_cnt: usize,
        term: u64,
    },
}

impl fmt::Display for ReadTask {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ReadTask::FetchLogs {
                region_id,
                peer_id,
                low,
                high,
                ..
            } => {
                write!(
                    f,
                    "FetchLogs(region_id={}, peer_id={}, range=[{}, {}))",
                    region_id, peer_id, low, high
                )
            }
        }
    }
}

#[derive(Debug)]
pub struct FetchedLogs {
    pub context: GetEntriesContext,
    pub logs: Box<RaftlogFetchResult>,
}

/// A router for receiving fetched result.
pub trait AsyncReadNotifier: Send {
    fn notify_logs_fetched(&self, region_id: u64, fetched: FetchedLogs);
}

pub struct ReadRunner<N>
where
    N: AsyncReadNotifier,
{
    notifier: N,
    raft_engine: RfEngine,
}

impl<N> ReadRunner<N>
where
    N: AsyncReadNotifier,
{
    pub fn new(raft_engine: RfEngine, notifier: N) -> Self {
        ReadRunner {
            notifier,
            raft_engine,
        }
    }
}

impl<N> Runnable for ReadRunner<N>
where
    N: AsyncReadNotifier,
{
    type Task = ReadTask;

    fn run(&mut self, task: ReadTask) {
        match task {
            ReadTask::FetchLogs {
                region_id,
                peer_id,
                low,
                high,
                max_size,
                context,
                tried_cnt,
                term,
            } => {
                let mut ents = Vec::with_capacity((high - low) as usize);
                let res = self.raft_engine.fetch_raft_entries_to(
                    peer_id,
                    low,
                    high,
                    Some(max_size),
                    &mut ents,
                    None, // No async context needed for sync execution
                );

                let hit_size_limit = res
                    .as_ref()
                    .map(|c| (*c as u64) != high - low)
                    .unwrap_or(false);

                self.notifier.notify_logs_fetched(
                    region_id,
                    FetchedLogs {
                        context,
                        logs: Box::new(RaftlogFetchResult {
                            ents: res.map(|_| ents).map_err(|e| e.into()),
                            low,
                            max_size: max_size as u64,
                            hit_size_limit,
                            tried_cnt,
                            term,
                        }),
                    },
                );
            }
        }
    }
}
