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

        if mutable_mem_usage < self.cfg.soft_limit.0 {
            match self.pick_one() {
                Some(id) => return vec![id],
                None => return vec![],
            }
        }

        self.pick_batch()
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

    fn pick_batch(&self) -> Vec<u64> {
        let mut candidates = BinaryHeap::with_capacity(self.cfg.max_flush_batch);
        let mut accesses: Vec<_> = self.last_access.iter().map(|(a, b)| (*a, *b)).collect();
        accesses.sort_by_key(|(_, time)| *time);
        for (id, time) in &accesses {
            if time.saturating_elapsed() >= self.cfg.evict_life_time.0 {
                let size = match self.write_buffers.get(id) {
                    Some(s) => *s,
                    None => continue,
                };
                if size < self.cfg.flush_threshold.0 {
                    continue;
                }
                if candidates.len() < self.cfg.max_flush_batch {
                    candidates.push((*id, -(size as i64)));
                } else {
                    let mut head = candidates.peek_mut().unwrap();
                    if head.1 > -(size as i64) {
                        *head = (*id, -(size as i64));
                    }
                }
            } else {
                break;
            }
        }
        if !candidates.is_empty() {
            return candidates.iter().map(|(id, _)| *id).collect();
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
            if let Err(e) = tablet.flush(false) {
                warn!("failed to flush tablet"; "region_id" => region_id, "err" => ?e);
            }
            let after = tablet.get_engine_memory_usage();
            info!("tablet flushed"; "region_id" => region_id, "before" => before, "after" => after);
            let size = tablet.get_engine_memory_usage();
            self.mgr.mark_flush(region_id, time, size as usize);
        }
    }
    fn get_interval(&self) -> Duration {
        Duration::from_secs(10)
    }
}
