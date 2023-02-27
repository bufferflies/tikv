// Copyright 2022 TiKV Project Authors. Licensed under Apache-2.0.

use tracker::{TrackerToken, GLOBAL_TRACKERS};

#[derive(Debug)]
pub struct RaftEnginePerfContext;

impl engine_traits::PerfContext for RaftEnginePerfContext {
    fn start_observe(&mut self) {
        panic!()
    }

    fn report_metrics(&mut self, trackers: &[TrackerToken]) {
        panic!()
    }
}
