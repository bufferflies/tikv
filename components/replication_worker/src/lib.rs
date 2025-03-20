// Copyright 2025 TiKV Project Authors. Licensed under Apache-2.0.

mod apply_observer;
mod config;
mod scheduler;
mod worker;

pub use apply_observer::{CdcApplyObserver, RegionEvents};
use cdc::MemoryQuota;
use kvproto::{cdcpb_grpc::ChangeData, raft_cmdpb::AdminRequest, tikvpb_grpc::Tikv};
use merged_engine::MergedEngineConfig;
pub use scheduler::{handle_cdc_request, ReplicationScheduler};
use serde_derive::{Deserialize, Serialize};
pub use worker::ReplicationWorker;

use crate::scheduler::ChangefeedRequest;

#[derive(Clone, Serialize, Deserialize, PartialEq, Debug, Default)]
#[serde(default)]
#[serde(rename_all = "kebab-case")]
pub struct ReplicationWorkerConfig {
    pub enabled: bool,

    pub grpc_addr: String,
    // used for local mode.
    pub pd_bin_path: String,
    pub cdc_bin_path: String,
    pub base_port: u16,

    // used for test.
    pub tidb_bin_path: String,

    // used for k8s mode.
    pub pd_sts_name: String,
    pub cdc_sts_name: String,

    pub merged_engine: MergedEngineConfig,
}

impl ReplicationWorkerConfig {
    pub fn override_from_env(&mut self) {
        Self::env_or_default("PD_BIN", &mut self.pd_bin_path);
        Self::env_or_default("CDC_BIN", &mut self.cdc_bin_path);
        Self::env_or_default("TIDB_BIN", &mut self.tidb_bin_path);
        Self::env_or_default("PD_STS_NAME", &mut self.pd_sts_name);
        Self::env_or_default("CDC_STS_NAME", &mut self.cdc_sts_name);
    }

    fn env_or_default(name: &str, val: &mut String) {
        if let Ok(v) = std::env::var(name) {
            *val = v;
        }
    }
}

pub enum CdcMsg {
    NewTask {
        keyspace_id: u32,
        request: ChangefeedRequest,
    },
    Applied {
        region_id: u64,
        region_events: RegionEvents,
    },
    AppliedAdmin {
        region_id: u64,
        region_version: u64,
        admin: AdminRequest,
    },
    RemoveTask {
        keyspace_id: u32,
        change_feed_id: String,
    },
    Stop,
}

#[allow(dead_code)]
#[derive(Clone)]
struct ReplicationService {
    kv: kvengine::Engine,
    scheduler: tikv_util::mpsc::Sender<CdcMsg>,
    memory_quota: MemoryQuota,
}

impl ReplicationService {
    pub fn new(kv: kvengine::Engine, scheduler: tikv_util::mpsc::Sender<CdcMsg>) -> Self {
        Self {
            kv,
            scheduler,
            memory_quota: MemoryQuota::new(1024 * 1024 * 1024),
        }
    }
}

impl ChangeData for ReplicationService {}

impl Tikv for ReplicationService {}
