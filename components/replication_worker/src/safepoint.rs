// Copyright 2025 TiKV Project Authors. Licensed under Apache-2.0.

//! ServiceSafepointManager manages service safepoints for changefeeds.
//!
//! Ref: https://github.com/pingcap/tiflow/blob/release-8.5/pkg/txnutil/gc/doc.go

use std::sync::Arc;

use pd_client::PdClient;
use tikv_util::{error, info, warn};

use crate::{
    metrics::REP_SAFEPOINT_EVENTS_COUNTER, Error, ReplicationWorkerConfig, Result, SafepointConfig,
};

pub(crate) struct ServiceSafepointManager {
    config: SafepointConfig,
    merged_store_id: u64,

    pd: Arc<dyn PdClient>,
    runtime: tokio::runtime::Handle,
}

impl ServiceSafepointManager {
    pub(crate) fn new(
        merged_store_id: u64,
        pd: Arc<dyn PdClient>,
        rep_config: &ReplicationWorkerConfig,
        runtime: tokio::runtime::Handle,
    ) -> Result<Self> {
        let config = rep_config.safepoint.clone();
        Ok(Self {
            config,
            merged_store_id,
            pd,
            runtime,
        })
    }

    pub(crate) fn shutdown(self) {}

    pub(crate) fn ensure_changefeed_start_ts_safety(
        &mut self,
        keyspace_id: u32,
        feed_id: &str,
        start_ts: u64,
    ) -> Result<()> {
        let keyspace_str = keyspace_id.to_string();
        let service_id = self.get_ensure_start_ts_service_id(feed_id);
        match self
            .runtime
            .block_on(self.pd.update_keyspace_service_safe_point(
                keyspace_id,
                service_id,
                start_ts.into(),
                self.config.create_changefeed_gc_ttl.0,
            )) {
            Ok(new_safepoint) => {
                info!("ServiceSafepointManager: ensure_changefeed_start_ts_safety ok";
                    "keyspace" => keyspace_id, "feed" => feed_id,
                    "start_ts" => start_ts, "new_safepoint" => new_safepoint);
                REP_SAFEPOINT_EVENTS_COUNTER
                    .with_label_values(&["ok", &keyspace_str, "ensure_start_ts_safepoint"])
                    .inc();
                Ok(())
            }
            Err(pd_client::Error::UnsafeServiceGcSafePoint {
                requested,
                current_minimal,
            }) => {
                debug_assert_eq!(requested.into_inner(), start_ts);
                warn!("ServiceSafepointManager: start_ts before safepoint";
                    "keyspace" => keyspace_id, "feed" => feed_id,
                    "start_ts" => start_ts, "current" => current_minimal);
                REP_SAFEPOINT_EVENTS_COUNTER
                    .with_label_values(&["warn", &keyspace_str, "start_ts_before_safepoint"])
                    .inc();
                Err(Error::StartTsBeforeSafepoint {
                    start_ts,
                    gc_safe_point: current_minimal.into_inner(),
                })
            }
            Err(err) => {
                error!("ServiceSafepointManager: ensure_changefeed_start_ts_safety failed: {:?}", err;
                    "keyspace" => keyspace_id, "feed" => feed_id, "start_ts" => start_ts);
                REP_SAFEPOINT_EVENTS_COUNTER
                    .with_label_values(&["error", &keyspace_str, "ensure_start_ts_safepoint"])
                    .inc();
                Err(err.into())
            }
        }
    }

    fn get_ensure_start_ts_service_id(&self, changefeed_id: &str) -> String {
        format!(
            "rep-worker-{}-creating-{}",
            self.merged_store_id, changefeed_id
        )
    }
}
