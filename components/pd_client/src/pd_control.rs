// Copyright 2023 TiKV Project Authors. Licensed under Apache-2.0.

use std::{collections::HashMap, sync::Arc, time::Duration};

use bstr::ByteSlice;
use http::Method;
use kvproto::metapb;
use security::{RestfulClient, SecurityManager};
use serde::Deserialize;
use slog_global::debug;
use tikv_util::config::ReadableDuration;

use crate::Config;

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Sync + Send>>;

const PD_CONFIG_PATH: &str = "pd/api/v1/config";
const PD_REGIONS_STORE_PATH: &str = "pd/api/v1/regions/store";
const PD_KEYSPACE_PATH: &str = "pd/api/v2/keyspaces";
const PD_PLACEMENT_RULE_GROUP_PATH: &str = "pd/api/v1/config/placement-rule";
const PD_PLACEMENT_RULE_PATH: &str = "pd/api/v1/config/rule";
const PD_STATS_REGION: &str = "pd/api/v1/stats/region";
const PD_HEALTH_PATH: &str = "health";
const PD_SCHEDULERS_PATH: &str = "pd/api/v1/schedulers";
const PD_OPERATORS_PATH: &str = "pd/api/v1/operators";

const TIFLASH_GROUP: &str = "tiflash";

#[derive(Default, Serialize, Deserialize, Debug)]
#[serde(default)]
pub struct RuleGroup {
    pub group_id: String,
    pub group_index: i64,
    pub group_override: bool,
    pub rules: Option<Vec<Rule>>,
}

#[derive(Default, Serialize, Deserialize, Debug)]
#[serde(default)]
pub struct Rule {
    pub group_id: String,
    pub id: String,
    pub index: i64,
    #[serde(rename = "override")]
    pub is_override: bool,
    pub start_key: String,
    pub end_key: String,
    pub role: String,
    pub is_witness: bool,
    pub count: u32,
}

/// PdControl provides access to HTTP APIs of PD, which are not included in gRPC
/// interface. It's also expected to act like the tool `pd-ctl`.
#[derive(Clone)]
pub struct PdControl {
    client: RestfulClient,
}

impl PdControl {
    pub fn new(config: Config, security_mgr: Arc<SecurityManager>) -> Result<Self> {
        let client = RestfulClient::new("pd_control", config.endpoints, security_mgr)?;
        Ok(Self { client })
    }

    pub async fn get_config(&self) -> Result<PdConfigFromApi> {
        self.client.get(PD_CONFIG_PATH).await
    }

    pub async fn get_store_regions(&self, store_id: u64) -> Result<RegionsInfo> {
        let query = format!("{}/{}", PD_REGIONS_STORE_PATH, store_id);
        self.client.get(query).await
    }

    pub async fn get_keyspace_by_name(&self, keyspace_name: &str) -> Result<KeyspaceMeta> {
        let query = format!("{PD_KEYSPACE_PATH}/{}", keyspace_name);
        self.client.get(query).await
    }

    pub async fn create_keyspace(&self, params: CreateKeyspaceParams) -> Result<KeyspaceMeta> {
        self.client.post(PD_KEYSPACE_PATH, &params).await
    }

    pub async fn get_tiflash_placement_rule_group(&self) -> Result<Option<RuleGroup>> {
        let path = format!("{PD_PLACEMENT_RULE_GROUP_PATH}/{TIFLASH_GROUP}");
        self.client.get(path).await
    }

    pub async fn remove_tiflash_placement_rule_by_id(&self, rule_id: &str) -> Result<()> {
        let path = format!("{PD_PLACEMENT_RULE_PATH}/{TIFLASH_GROUP}/{rule_id}");
        let _ = self.client.delete(path).await?;
        Ok(())
    }

    // Ref: https://github.com/tidbcloud/pd-cse/blob/release-7.1-keyspace/server/api/stats.go
    pub async fn get_regions_number(&self) -> Result<i32> {
        let path = format!("{PD_STATS_REGION}?start_key=&end_key=&count=true");
        let region_stats: RegionStats = self.client.get(path).await?;
        Ok(region_stats.count)
    }

    // Ref: https://github.com/tidbcloud/pd-cse/blob/release-7.1-keyspace/server/api/health.go
    pub async fn health(&self) -> Result<bool> {
        let health: Health = self.client.get(PD_HEALTH_PATH).await?;
        Ok(health.health == "true")
    }

    // Ref: https://github.com/tidbcloud/pd-cse/blob/release-7.1-keyspace/server/api/scheduler.go
    // `scheduler_name` can be "all" to pause or resume all schedulers.
    // Pass `dur` as `Duration::ZERO` to resume the scheduler.
    pub async fn pause_or_resume_scheduler(
        &self,
        scheduler_name: &str,
        dur: Duration,
    ) -> Result<()> {
        let path = format!("{PD_SCHEDULERS_PATH}/{scheduler_name}");
        let params = SchedulerDelay {
            delay: dur.as_secs() as i64,
        };
        let body_data = serde_json::to_vec(&params)?;
        let _ = self
            .client
            .request(path, Method::POST, Some(body_data))
            .await?;
        Ok(())
    }

    // Note: only when `status` is `Some(SchedulerStatus::Paused)` will return
    // schedulers with timestamps.
    pub async fn list_schedulers(&self, status: Option<SchedulerStatus>) -> Result<Vec<Scheduler>> {
        let status_str = match status {
            Some(SchedulerStatus::Paused) => "paused",
            Some(SchedulerStatus::Disabled) => "disabled",
            None => "",
        };
        let query = format!("{PD_SCHEDULERS_PATH}?status={status_str}&timestamp=1");
        match status {
            Some(SchedulerStatus::Paused) => self.client.get(query).await,
            Some(SchedulerStatus::Disabled) | None => {
                let scheduler_names: Vec<String> = self.client.get(query).await?;
                Ok(scheduler_names
                    .into_iter()
                    .map(|name| Scheduler {
                        name,
                        ..Default::default()
                    })
                    .collect())
            }
        }
    }

    pub async fn create_scheduler(&self, name: String) -> Result<()> {
        let param = CreateSchedulerParam { name };
        let body_data = serde_json::to_vec(&param)?;
        let resp = self
            .client
            .request(PD_SCHEDULERS_PATH, Method::POST, Some(body_data))
            .await?;
        debug!("pd_control::create_scheduler"; "resp" => resp.to_str_lossy().as_ref());
        Ok(())
    }

    // Ref: https://github.com/tidbcloud/pd-cse/blob/release-7.1-keyspace/server/api/operator.go
    pub async fn get_operators(&self) -> Result<Vec<Operator>> {
        let query = format!("{PD_OPERATORS_PATH}?object=1");
        self.client.get(query).await
    }

    pub async fn cancel_operator_by_region(&self, region_id: u64) -> Result<()> {
        let query = format!("{PD_OPERATORS_PATH}/{region_id}");
        let _ = self.client.delete(query).await?;
        Ok(())
    }
}

#[derive(Default, Serialize, Deserialize, Debug, PartialEq, Eq)]
#[serde(default)]
pub struct KeyspaceMeta {
    pub id: u32,
    pub name: String,
    pub state: String,
    pub created_at: u64,
    pub state_changed_at: u64,
    pub config: HashMap<String, String>,
}

#[derive(Default, Serialize, Deserialize, Debug)]
#[serde(default)]
pub struct RegionInfo {
    pub id: u64,
    pub start_key: String,
    pub end_key: String,
}

#[derive(Default, Serialize, Deserialize, Debug)]
#[serde(default)]
pub struct RegionsInfo {
    pub count: u64,
    pub regions: Vec<RegionInfo>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(default)]
#[serde(rename_all = "kebab-case")]
pub struct PdScheduleConfig {
    pub max_store_down_time: String,
    pub max_merge_region_size: u64, // in MB.
    pub max_merge_region_keys: u64,
    pub split_merge_interval: ReadableDuration,
}

impl Default for PdScheduleConfig {
    fn default() -> Self {
        Self {
            max_store_down_time: "30m".to_string(),
            max_merge_region_size: 96,
            max_merge_region_keys: 200000,
            split_merge_interval: ReadableDuration::hours(1),
        }
    }
}

#[derive(Default, Serialize, Deserialize, Debug)]
#[serde(default)]
#[serde(rename_all = "kebab-case")]
pub struct PdConfigFromApi {
    pub schedule: PdScheduleConfig,
}

#[derive(Default, Serialize, Deserialize, Debug)]
#[serde(default)]
pub struct RegionStats {
    pub count: i32,
}

#[derive(Default, Serialize, Deserialize, Debug)]
#[serde(default)]
pub struct Health {
    pub health: String, // Note: not a boolean.
}

#[derive(Default, Serialize, Deserialize, Debug)]
#[serde(default)]
pub struct CreateKeyspaceParams {
    pub name: String,
    pub config: HashMap<String, String>,
}

impl CreateKeyspaceParams {
    pub fn new(name: String) -> Self {
        Self {
            name,
            config: HashMap::new(),
        }
    }

    pub fn with_encryption(&mut self, enabled: bool) -> &mut Self {
        self.config.insert(
            KEYSPACE_CONFIG_ENCRYPTION_KEY.to_string(),
            serde_json::to_string(&KeyspaceConfigEncryption { enabled }).unwrap(),
        );
        self
    }
}

pub const KEYSPACE_CONFIG_ENCRYPTION_KEY: &str = "encryption";

#[derive(Default, Serialize, Deserialize, Debug)]
#[serde(default)]
pub struct KeyspaceConfigEncryption {
    pub enabled: bool,
}

#[derive(Default, Serialize, Deserialize, Debug)]
#[serde(default)]
pub struct SchedulerDelay {
    pub delay: i64,
}

pub enum SchedulerStatus {
    Paused,
    Disabled,
}

#[derive(Default, Serialize, Deserialize, Debug)]
#[serde(default)]
pub struct Scheduler {
    pub name: String,
    pub paused_at: String,
    pub resume_at: String,
}

#[derive(Default, Serialize, Deserialize, Debug)]
#[serde(default)]
pub struct CreateSchedulerParam {
    pub name: String,
}

#[derive(Default, Deserialize, Debug)]
pub struct RegionEpoch {
    pub conf_ver: u64,
    pub version: u64,
}

impl From<&RegionEpoch> for metapb::RegionEpoch {
    fn from(epoch: &RegionEpoch) -> Self {
        let mut region_epoch = metapb::RegionEpoch::default();
        region_epoch.set_conf_ver(epoch.conf_ver);
        region_epoch.set_version(epoch.version);
        region_epoch
    }
}

pub type OpKindMask = u32; // bit mask of OpKind

// Ref: https://github.com/tidbcloud/pd-cse/blob/release-7.1-keyspace/pkg/schedule/operator/operator.go
#[derive(Default, Deserialize)]
pub struct Operator {
    pub desc: String,
    pub brief: String,
    pub region_id: u64,
    pub region_epoch: RegionEpoch,
    #[serde(rename = "kind")]
    pub kind_mask: OpKindMask,
}

impl std::fmt::Debug for Operator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Operator")
            .field("desc", &self.desc)
            .field("brief", &self.brief)
            .field("region_id", &self.region_id)
            .field("region_epoch", &self.region_epoch)
            .field("kind", &self.kind())
            .finish()
    }
}

impl Operator {
    pub fn kind(&self) -> Vec<OpKind> {
        let mut v = vec![];
        if self.kind_mask & OpKind::OpAdmin as u32 != 0 {
            v.push(OpKind::OpAdmin);
        }
        if self.kind_mask & OpKind::OpMerge as u32 != 0 {
            v.push(OpKind::OpMerge);
        }
        if self.kind_mask & OpKind::OpRange as u32 != 0 {
            v.push(OpKind::OpRange);
        }
        if self.kind_mask & OpKind::OpReplica as u32 != 0 {
            v.push(OpKind::OpReplica);
        }
        if self.kind_mask & OpKind::OpSplit as u32 != 0 {
            v.push(OpKind::OpSplit);
        }
        if self.kind_mask & OpKind::OpHotRegion as u32 != 0 {
            v.push(OpKind::OpHotRegion);
        }
        if self.kind_mask & OpKind::OpRegion as u32 != 0 {
            v.push(OpKind::OpRegion);
        }
        if self.kind_mask & OpKind::OpLeader as u32 != 0 {
            v.push(OpKind::OpLeader);
        }
        if self.kind_mask & OpKind::OpWitnessLeader as u32 != 0 {
            v.push(OpKind::OpWitnessLeader);
        }
        if self.kind_mask & OpKind::OpWitness as u32 != 0 {
            v.push(OpKind::OpWitness);
        }
        if v.is_empty() {
            v.push(OpKind::OpUnknown);
        }
        v
    }
}

// Ref: https://github.com/tidbcloud/pd-cse/blob/release-7.1-keyspace/pkg/schedule/operator/kind.go
#[derive(Debug, PartialEq)]
#[repr(u32)]
pub enum OpKind {
    OpUnknown = 0,
    OpAdmin = 1 << 0,
    // Initiated by merge checker or merge scheduler. Note that it may not include region merge.
    // the order describe the operator's producer and is very helpful to decouple scheduler or
    // checker limit
    OpMerge = 1 << 1,
    // Initiated by range scheduler.
    OpRange = 1 << 2,
    // Initiated by replica checker.
    OpReplica = 1 << 3,
    // Include region split. Initiated by rule checker if `kind & OpAdmin == 0`.
    OpSplit = 1 << 4,
    // Initiated by hot region scheduler.
    OpHotRegion = 1 << 5,
    // Include peer addition or removal or switch witness. This means that this operator may take
    // a long time.
    OpRegion = 1 << 6,
    // Include leader transfer.
    OpLeader = 1 << 7,
    // Include witness leader transfer.
    OpWitnessLeader = 1 << 8,
    // Include witness transfer.
    OpWitness = 1 << 9,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_op_kind() {
        let mut op = Operator::default();
        assert_eq!(op.kind(), vec![OpKind::OpUnknown]);

        op.kind_mask = OpKind::OpMerge as u32;
        assert_eq!(op.kind(), vec![OpKind::OpMerge]);

        op.kind_mask = OpKind::OpSplit as u32;
        assert_eq!(op.kind(), vec![OpKind::OpSplit]);

        op.kind_mask = OpKind::OpRegion as u32;
        assert_eq!(op.kind(), vec![OpKind::OpRegion]);

        op.kind_mask = (OpKind::OpAdmin as u32) | (OpKind::OpReplica as u32);
        assert_eq!(op.kind(), vec![OpKind::OpAdmin, OpKind::OpReplica]);
    }
}
