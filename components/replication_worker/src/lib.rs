// Copyright 2025 TiKV Project Authors. Licensed under Apache-2.0.

mod apply_observer;
mod config;
mod error;
mod provisioned;
mod scheduler;
mod worker;

use std::{collections::HashMap, sync::Arc};

pub use apply_observer::{CdcApplyObserver, RegionEvents};
use async_trait::async_trait;
use bytes::Bytes;
use cdc::MemoryQuota;
pub use error::{Error, Result};
use kvproto::{cdcpb_grpc::ChangeData, raft_cmdpb::AdminRequest, tikvpb_grpc::Tikv};
use merged_engine::MergedEngineConfig;
use pd_client::PdClient;
pub use provisioned::LocalProvider;
use resolved_ts::Resolver;
pub use scheduler::*;
use serde_derive::{Deserialize, Serialize};
pub use worker::ReplicationWorker;

#[derive(Clone, Serialize, Deserialize, PartialEq, Debug, Default)]
#[serde(default)]
#[serde(rename_all = "kebab-case")]
pub struct ReplicationWorkerConfig {
    pub enabled: bool,

    pub grpc_addr: String,

    // used for k8s mode.
    pub pd_sts_name: String,
    pub cdc_sts_name: String,

    pub merged_engine: MergedEngineConfig,
}

impl ReplicationWorkerConfig {
    pub fn override_from_env(&mut self) {
        Self::env_or_default("PD_STS_NAME", &mut self.pd_sts_name);
        Self::env_or_default("CDC_STS_NAME", &mut self.cdc_sts_name);
    }

    fn env_or_default(name: &str, val: &mut String) {
        if let Ok(v) = std::env::var(name) {
            *val = v;
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct KeyspaceStates {
    pub(crate) feeds: HashMap<String, ChangefeedRequest>,
    pub(crate) pd_url: String,
    pub(crate) cdc_addr: String,

    // used in k8s.
    pub(crate) pd_sts_name: String,
    pub(crate) cdc_sts_name: String,
}

impl KeyspaceStates {
    pub(crate) fn marshal(&self) -> Bytes {
        serde_json::to_vec(self).unwrap().into()
    }
}

#[async_trait]
pub trait KeyspaceService: Send {
    fn keyspace_id(&self) -> u32;

    async fn start(&mut self) -> Result<()>;

    async fn destroy(&mut self) -> Result<()>;

    fn get_states(&self) -> &KeyspaceStates;

    fn get_states_mut(&mut self) -> &mut KeyspaceStates;

    fn get_pd_client(&self) -> Arc<dyn PdClient>;

    fn get_resolver(&mut self) -> &mut Resolver;
}

pub enum CdcMsg {
    AddKeyspace {
        keyspace_id: u32,
        pd_url: String,
        cdc_addr: String,
        cb: Box<dyn FnOnce(Result<()>) + Send>,
    },
    AddKeyspaceResult {
        keyspace_id: u32,
        result: Result<Box<dyn KeyspaceService>>,
        cb: Box<dyn FnOnce(Result<()>) + Send>,
    },
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
    RemoveKeyspace {
        keyspace_id: u32,
        cb: Box<dyn FnOnce(Result<()>) + Send>,
    },
    GetKeyspaces {
        cb: Box<dyn FnOnce(Vec<u32>) + Send>,
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
