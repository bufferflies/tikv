// Copyright 2023 TiKV Project Authors. Licensed under Apache-2.0.

use std::{collections::HashMap, sync::Arc, time::Duration};

use bstr::ByteSlice;
use bytes::Bytes;
use http::{Method, Request};
use hyper::Body;
use security::SecurityManager;
use serde::Deserialize;
use slog_global::debug;
use tikv_util::{box_err, config::ReadableDuration};

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
    security_mgr: Arc<SecurityManager>,
    endpoints: Vec<String>,
}

impl PdControl {
    pub fn new(config: Config, security_mgr: Arc<SecurityManager>) -> Result<Self> {
        let mut endpoints = Vec::with_capacity(config.endpoints.len());
        for endpoint in &config.endpoints {
            let url = if endpoint.starts_with("http://") {
                endpoint.strip_prefix("http://").unwrap()
            } else if endpoint.starts_with("https://") {
                endpoint.strip_prefix("https://").unwrap()
            } else {
                endpoint
            };
            endpoints.push(url.to_string());
        }
        Ok(Self {
            endpoints,
            security_mgr,
        })
    }

    async fn request_pd_restful(
        &self,
        path: impl AsRef<str>,
        method: Method,
        body_data: Option<Vec<u8>>,
    ) -> Result<Bytes> {
        let path = path.as_ref();
        let client = self.security_mgr.http_client(hyper::Client::builder())?;
        let mut err = None;
        for endpoint in self.endpoints.iter() {
            let uri = self.security_mgr.build_uri(format!("{endpoint}/{path}"))?;
            let req = Request::builder()
                .method(method.clone())
                .uri(uri.clone())
                .body(match body_data {
                    Some(ref data) => Body::from(data.to_owned()),
                    None => Body::empty(),
                })
                .unwrap();
            let resp = client.request(req).await;
            match resp {
                Err(e) => {
                    err = Some(box_err!(
                        "PD uri[{}] error: {}",
                        uri.to_string(),
                        e.to_string()
                    ))
                }
                Ok(resp) => {
                    let status = resp.status();
                    let body = hyper::body::to_bytes(resp.into_body()).await.unwrap();
                    if status.is_success() {
                        debug!("request_pd_restful success: {}", body.to_str_lossy());
                        return Ok(body);
                    } else {
                        err = Some(box_err!(
                            "PD({endpoint}) return error: {status}: {}",
                            body.to_str_lossy()
                        ));
                    }
                }
            }
        }
        Err(err.expect("there must be error"))
    }

    async fn get<T>(&self, path: impl AsRef<str>) -> Result<T>
    where
        T: std::fmt::Debug + for<'a> serde::de::Deserialize<'a>,
    {
        let path = path.as_ref();
        match self.request_pd_restful(path, Method::GET, None).await {
            Ok(resp) => {
                let t: T = serde_json::from_slice(&resp)?;
                debug!("pd_control::get"; "path" => path, "resp" => ?t);
                Ok(t)
            }
            Err(err) => Err(box_err!(
                "pd_control::get failed: path: {}, err: {:?}",
                path,
                err
            )),
        }
    }

    async fn post<Req, Resp>(&self, path: impl AsRef<str>, data: &Req) -> Result<Resp>
    where
        Req: serde::ser::Serialize,
        Resp: std::fmt::Debug + for<'a> serde::de::Deserialize<'a>,
    {
        let path = path.as_ref();
        let body_data = serde_json::to_vec(data)?;
        match self
            .request_pd_restful(path, Method::POST, Some(body_data))
            .await
        {
            Ok(resp) => {
                let t: Resp = serde_json::from_slice(&resp)?;
                debug!("pd_control::post"; "path" => path, "resp" => ?t);
                Ok(t)
            }
            Err(err) => Err(box_err!(
                "pd_control::post failed: path: {}, err: {:?}",
                path,
                err
            )),
        }
    }

    pub async fn get_config(&self) -> Result<PdConfigFromApi> {
        self.get(PD_CONFIG_PATH).await
    }

    pub async fn get_store_regions(&self, store_id: u64) -> Result<RegionsInfo> {
        let query = format!("{}/{}", PD_REGIONS_STORE_PATH, store_id);
        self.get(query).await
    }

    pub async fn get_keyspace_by_name(&self, keyspace_name: &str) -> Result<KeyspaceMeta> {
        let query = format!("{PD_KEYSPACE_PATH}/{}", keyspace_name);
        self.get(query).await
    }

    pub async fn create_keyspace(&self, params: CreateKeyspaceParams) -> Result<KeyspaceMeta> {
        self.post(PD_KEYSPACE_PATH, &params).await
    }

    pub async fn get_tiflash_placement_rule_group(&self) -> Result<Option<RuleGroup>> {
        let path = format!("{PD_PLACEMENT_RULE_GROUP_PATH}/{TIFLASH_GROUP}");
        self.get(path).await
    }

    pub async fn remove_tiflash_placement_rule_by_id(&self, rule_id: &str) -> Result<()> {
        let path = format!("{PD_PLACEMENT_RULE_PATH}/{TIFLASH_GROUP}/{rule_id}");
        match self.request_pd_restful(path, Method::DELETE, None).await {
            Ok(_) => {
                debug!("delete tiflash rule {}-{}", TIFLASH_GROUP, rule_id);
                Ok(())
            }
            Err(err) => Err(box_err!("delete tiflash rule error: {:?}", err)),
        }
    }

    // Ref: https://github.com/tidbcloud/pd-cse/blob/release-7.1-keyspace/server/api/stats.go
    pub async fn get_regions_number(&self) -> Result<i32> {
        let path = format!("{PD_STATS_REGION}?start_key=&end_key=&count=true");
        let region_stats: RegionStats = self.get(path).await?;
        Ok(region_stats.count)
    }

    // Ref: https://github.com/tidbcloud/pd-cse/blob/release-7.1-keyspace/server/api/health.go
    pub async fn health(&self) -> Result<bool> {
        let health: Health = self.get(PD_HEALTH_PATH).await?;
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
            .request_pd_restful(path, Method::POST, Some(body_data))
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
            Some(SchedulerStatus::Paused) => self.get(query).await,
            Some(SchedulerStatus::Disabled) | None => {
                let scheduler_names: Vec<String> = self.get(query).await?;
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

    // Ref: https://github.com/tidbcloud/pd-cse/blob/release-7.1-keyspace/server/api/operator.go
    // Return array of strings, see https://github.com/tidbcloud/pd-cse/blob/release-7.1-keyspace/pkg/schedule/operator/operator.go, Operator.MarshalJSON
    // E.g.:
    // ```
    // "balance-leader {transfer leader: store 5 to 6} (kind:leader, region:179(42,29), createAt:2024-01-09 08:31 :45.749305225 +0000 UTC m=+110.335827093,startAt:2024-01-09 08:31:45.749636244 +0000 UTC m=+110.336158110, currentStep:0, size:1, steps:[0:{transfer leader from store 5 to store 6}],timeout:[1m0s])"
    // ````
    pub async fn get_operators(&self) -> Result<Vec<String>> {
        self.get(PD_OPERATORS_PATH).await
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
