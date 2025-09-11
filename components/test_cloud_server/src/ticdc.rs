// Copyright 2025 TiKV Project Authors. Licensed under Apache-2.0.

use serde_derive::Serialize;

// Ref: https://docs.pingcap.com/tidb/stable/ticdc-open-api-v2/#parameter-descriptions.
#[rustfmt::skip]
/*
{
  "changefeed_id":"rep-task",
  "sink_uri":"mysql://root@127.0.0.1:4455",
  "start_ts": 460426866192809988,
  "replica_config":{
    "enable_sync_point":true,
    "sync_point_interval":"30s",
    "filter":{
      "rules":["sbtest.*"]
    }
  }
}
 */
#[derive(Serialize, Default)]
pub struct ChangefeedParams {
    pub changefeed_id: String,
    pub sink_uri: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub start_ts: Option<u64>,
    pub replica_config: ChangefeedReplicaConfig,
}

#[derive(Serialize, Default)]
pub struct ChangefeedReplicaConfig {
    pub enable_sync_point: bool,
    pub sync_point_interval: String, // Minimum is 30s.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub filter: Option<ChangefeedFilter>,
}

#[derive(Serialize, Default)]
pub struct ChangefeedFilter {
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub rules: Vec<String>,
}
