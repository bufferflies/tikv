// Copyright 2025 TiKV Project Authors. Licensed under Apache-2.0.

mod collector;
mod config;
mod controller;
mod event;
mod hub;
mod limiter;
mod metrics;
mod publisher;
mod resource;
mod subscriber;
mod usage;
mod util;

pub use collector::{ResourceCollector, ThreadCollector};
pub use config::{Config, ConfigManager};
pub use controller::ResourceController;
pub use event::{Metric, ResourceEvent, Scope, Severity, SeverityThreshold};
use hub::{EventHub, ResourceHub};
pub use limiter::{KeyspaceReadLimiter, ReadLimiter, TransferLeaderLimiter};
pub use metrics::{ACTIVE_KEYSPACE_READ_BYTES, REQUEST_WAIT_HISTOGRAM_VEC};
use publisher::EventPublisher;
pub use publisher::ResourcePublisher;
pub use resource::{CpuType, Resource, ResourceType, ResourceValue, Usage};
pub use subscriber::{ReadSubscriber, ResourceSubscriber, TransferLeaderSubscriber};
use usage::{GlobalInstantUsages, ResourceUsage, Usages};
pub use util::{AtomicDuration, AtomicTime, TimeUnit};

const MAX_BATCH_SIZE: usize = 1024 * 1024;
