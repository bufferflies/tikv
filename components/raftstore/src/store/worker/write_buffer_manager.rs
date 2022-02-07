// Copyright 2021 TiKV Project Authors. Licensed under Apache-2.0.

use collections::HashMap;
use engine_traits::{Engines, GlobalWriteBufferStats, KvEngine, RaftEngine};
use serde::{Deserialize, Serialize};
use slog_global::{info, warn};
use std::collections::BinaryHeap;
use std::fmt::{self, Display};
use std::time::{Duration, Instant};
use tikv_util::config::{ReadableDuration, ReadableSize};
use tikv_util::time::InstantExt;
use tikv_util::worker::{Runnable, RunnableWithTimer};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(default)]
#[serde(rename_all = "kebab-case")]
pub struct Config {
    pub total_limit: ReadableSize,
    pub soft_limit: ReadableSize,
    pub flush_threshold: ReadableSize,
    pub evict_life_time: ReadableDuration,
    pub max_flush_batch: usize,
    pub check_interval: ReadableDuration,
}

impl Default for Config {
    #[inline]
    fn default() -> Config {
        Config {
            total_limit: ReadableSize::gb(5),
            soft_limit: ReadableSize::gb(2),
            flush_threshold: ReadableSize::mb(1),
            evict_life_time: ReadableDuration::minutes(30),
            max_flush_batch: 8,
            check_interval: ReadableDuration::secs(10),
        }
    }
}

pub struct WriteBufferManager {
    write_buffers: HashMap<u64, u64>,
    last_access: HashMap<u64, Instant>,
    global_mgr: Box<dyn GlobalWriteBufferStats>,
    cfg: Config,
}

impl WriteBufferManager {
    pub fn new(m: Box<dyn GlobalWriteBufferStats>, cfg: Config) -> WriteBufferManager {
        WriteBufferManager {
            global_mgr: m,
            cfg,
            write_buffers: HashMap::default(),
            last_access: HashMap::default(),
        }
    }

    #[inline]
    fn record_size(&mut self, region_id: u64, size: usize) {
        self.write_buffers.insert(region_id, size as u64);
    }

    fn pick_to_flush(&self) -> Vec<u64> {
        let mem_usage = self.global_mgr.memory_usage() as u64;
        if mem_usage < self.cfg.soft_limit.0 {
            return vec![];
        }

        let mutable_mem_usage = self.global_mgr.mutable_memtable_memory_usage() as u64;
        if mutable_mem_usage < self.cfg.soft_limit.0 && mem_usage < self.cfg.total_limit.0 {
            return vec![];
        }

        info!(
            "trigger tablet flush";
            "mem_usage" => mem_usage,
            "mutable_mem_usage" => mutable_mem_usage,
            "soft_limit" => self.cfg.soft_limit.0,
            "total_limit" => self.cfg.total_limit.0
        );
        if mutable_mem_usage < self.cfg.soft_limit.0 {
            match self.pick_one() {
                Some(id) => return vec![id],
                None => return vec![],
            }
        }

        self.pick_batch(mutable_mem_usage)
    }

    fn pick_one(&self) -> Option<u64> {
        let mut accesses: Vec<_> = self.last_access.iter().map(|(a, b)| (*a, *b)).collect();
        accesses.sort_by_key(|(_, time)| *time);
        let mut res = (0, 0);
        for (id, time) in &accesses {
            if time.saturating_elapsed() >= self.cfg.evict_life_time.0 {
                let size = match self.write_buffers.get(id) {
                    Some(s) => *s,
                    None => continue,
                };
                if size < self.cfg.flush_threshold.0 {
                    continue;
                }
                if size > res.1 {
                    res = (*id, size);
                }
            } else {
                break;
            }
        }
        if res.0 == 0 {
            accesses.get(0).map(|(id, _)| *id)
        } else {
            Some(res.0)
        }
    }

    fn fetch_by_access_time(&self, accesses: &[(u64, Instant)]) -> Vec<u64> {
        let mut candidates = BinaryHeap::with_capacity(self.cfg.max_flush_batch);
        for (id, time) in accesses {
            if time.saturating_elapsed() >= self.cfg.evict_life_time.0 {
                let size = match self.write_buffers.get(id) {
                    Some(s) => *s,
                    None => continue,
                };
                if size < self.cfg.flush_threshold.0 {
                    continue;
                }
                if candidates.len() < self.cfg.max_flush_batch {
                    candidates.push((-(size as i64), *id));
                } else {
                    let mut head = candidates.peek_mut().unwrap();
                    if head.0 > -(size as i64) {
                        *head = (-(size as i64), *id);
                    }
                }
            } else {
                break;
            }
        }
        candidates.iter().map(|(_, id)| *id).collect()
    }

    fn fetch_by_size(&self, accesses: &[(u64, Instant)], mutable_mem_usage: u64) -> Vec<u64> {
        let mut candidates = Vec::with_capacity(self.cfg.max_flush_batch);
        for (id, _) in accesses {
            let size = match self.write_buffers.get(id) {
                Some(s) => *s,
                None => continue,
            };
            if size < self.cfg.flush_threshold.0 {
                continue;
            }
            candidates.push((size, *id));
        }
        let mut choose_limit = if mutable_mem_usage > self.cfg.total_limit.0 {
            mutable_mem_usage / 2
        } else {
            self.cfg.soft_limit.0 / 2
        };
        info!("choose by size"; "candidates" => ?candidates, "choose_limit" => choose_limit);
        candidates.pop();
        let mut priority_arr = Vec::with_capacity(candidates.len());
        for (pos, (size, id)) in candidates.iter().enumerate() {
            priority_arr.push(((*size) + 1024 * (candidates.len() - pos) as u64, *size, *id))
        }
        let heap = BinaryHeap::from(priority_arr);
        let mut result = vec![];
        for (_, size, id) in heap.into_iter_sorted() {
            result.push(id);
            if size >= choose_limit {
                break;
            }
            choose_limit -= size;
        }
        result
    }

    fn pick_batch(&self, mutable_mem_usage: u64) -> Vec<u64> {
        let mut accesses: Vec<_> = self.last_access.iter().map(|(a, b)| (*a, *b)).collect();
        accesses.sort_by_key(|(_, time)| *time);
        let mut candidates = self.fetch_by_access_time(&accesses);
        if candidates.is_empty() {
            candidates = self.fetch_by_size(&accesses, mutable_mem_usage);
        }
        if !candidates.is_empty() {
            return candidates;
        }
        let count = std::cmp::min(accesses.len() / 2, self.cfg.max_flush_batch);
        accesses.iter().take(count).map(|(id, _)| *id).collect()
    }

    fn record_access(&mut self, region_id: u64, time: Instant) {
        self.last_access.insert(region_id, time);
    }

    fn mark_flush(&mut self, id: u64, time: Instant, size: usize) {
        if let Some(la) = self.last_access.remove(&id) {
            if la > time {
                self.last_access.insert(id, la);
                return;
            }
        }
        self.write_buffers.insert(id, size as u64);
    }
}

#[derive(Debug)]
pub enum Msg {
    RecordAccess { region_id: u64, time: Instant },
    RecordSize { region_id: u64, size: usize },
}

impl Display for Msg {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

pub struct Runner<EK, ER> {
    engines: Engines<EK, ER>,
    mgr: WriteBufferManager,
}

impl<EK: KvEngine, ER: RaftEngine> Runner<EK, ER> {
    pub fn new(engines: Engines<EK, ER>, cfg: Config) -> Runner<EK, ER> {
        Runner {
            mgr: WriteBufferManager::new(engines.tablets.write_buffer_states(), cfg),
            engines,
        }
    }
}

impl<EK: KvEngine, ER: RaftEngine + Send> Runnable for Runner<EK, ER> {
    type Task = Msg;

    fn run(&mut self, task: Msg) {
        match task {
            Msg::RecordAccess { region_id, time } => self.mgr.record_access(region_id, time),
            Msg::RecordSize { region_id, size } => self.mgr.record_size(region_id, size),
        }
    }
}

impl<EK: KvEngine, ER: RaftEngine> RunnableWithTimer for Runner<EK, ER> {
    fn on_timeout(&mut self) {
        let to_flush = self.mgr.pick_to_flush();
        if to_flush.is_empty() {
            return;
        }
        for region_id in to_flush {
            let tablet = match self.engines.tablets.open_tablet_cache_any(region_id) {
                Some(t) => t,
                None => return,
            };
            let before = tablet.get_engine_memory_usage();
            let time = Instant::now();
            if let Err(e) = tablet.flush(true) {
                warn!("failed to flush tablet"; "region_id" => region_id, "err" => ?e);
            }
            let after = tablet.get_engine_memory_usage();
            info!("tablet flushed"; "region_id" => region_id, "before" => before, "after" => after);
            let size = tablet.get_engine_memory_usage();
            self.mgr.mark_flush(region_id, time, size as usize);
        }
    }
    fn get_interval(&self) -> Duration {
        self.mgr.cfg.check_interval.0
    }
}
