// Copyright 2023 TiKV Project Authors. Licensed under Apache-2.0.

use std::sync::Arc;

use bstr::ByteSlice;
use bytes::Bytes;
use http::{Method, Request};
use hyper::Body;
use security::SecurityManager;
use slog_global::debug;
use tikv_util::box_err;

use crate::Config;

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Sync + Send>>;

const PD_CONFIG_PATH: &str = "pd/api/v1/config";
const PD_REGIONS_STORE_PATH: &str = "pd/api/v1/regions/store";
const PD_KEYSPACE_PATH: &str = "pd/api/v2/keyspaces";
const PD_PLACEMENT_RULE_GROUP_PATH: &str = "pd/api/v1/config/placement-rule";
const PD_PLACEMENT_RULE_PATH: &str = "pd/api/v1/config/rule";
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
        Ok(Self {
            endpoints: config.endpoints,
            security_mgr,
        })
    }

    async fn request_pd_restful(
        &self,
        path: String,
        method: Method,
        body_data: Option<Vec<u8>>,
    ) -> Result<Bytes> {
        let client = self.security_mgr.http_client(hyper::Client::builder())?;
        let mut err = None;
        for endpoint in self.endpoints.iter() {
            let uri = self.security_mgr.build_uri(format!("{endpoint}/{path}"))?;
            let req = Request::builder()
                .method(method.clone())
                .uri(uri)
                .body(match body_data {
                    Some(ref data) => Body::from(data.to_owned()),
                    None => Body::empty(),
                })
                .unwrap();
            let resp = client.request(req).await;
            match resp {
                Err(e) => err = Some(box_err!(e)),
                Ok(resp) => {
                    let status = resp.status();
                    let body = hyper::body::to_bytes(resp.into_body()).await.unwrap();
                    if status.is_success() {
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

    pub async fn get_config(&self) -> Result<PdConfigFromApi> {
        let query = PD_CONFIG_PATH.to_string();
        match self.request_pd_restful(query, Method::GET, None).await {
            Ok(resp) => {
                let pd_config = serde_json::from_slice(&resp)?;
                debug!("pd_config: {:?}", pd_config);
                Ok(pd_config)
            }
            Err(err) => Err(box_err!("get_config error: {:?}", err)),
        }
    }

    pub async fn get_store_regions(&self, store_id: u64) -> Result<RegionsInfo> {
        let query = format!("{}/{}", PD_REGIONS_STORE_PATH, store_id);
        match self.request_pd_restful(query, Method::GET, None).await {
            Ok(resp) => {
                let regions_info = serde_json::from_slice(&resp)?;
                debug!("regions_info: {:?}", regions_info);
                Ok(regions_info)
            }
            Err(err) => Err(box_err!("get_store_regions error: {:?}", err)),
        }
    }

    pub async fn get_keyspace_by_name(&self, keyspace_name: &str) -> Result<KeyspaceMeta> {
        let query = format!("{PD_KEYSPACE_PATH}/{}", keyspace_name);
        match self.request_pd_restful(query, Method::GET, None).await {
            Ok(resp) => {
                let keyspace = serde_json::from_slice(&resp)?;
                debug!("get_keyspace: {:?}", keyspace);
                Ok(keyspace)
            }
            Err(err) => Err(box_err!("get_keyspace_by_name error: {:?}", err)),
        }
    }

    pub async fn get_tiflash_placement_rule_group(&self) -> Result<Option<RuleGroup>> {
        let path = format!("{PD_PLACEMENT_RULE_GROUP_PATH}/{TIFLASH_GROUP}");
        match self.request_pd_restful(path, Method::GET, None).await {
            Ok(resp) => {
                let tiflash_rule_group = serde_json::from_slice(&resp)?;
                debug!("get tiflash rule group: {:?}", tiflash_rule_group);
                Ok(tiflash_rule_group)
            }
            Err(err) => Err(box_err!("get tiflash rule group error: {:?}", err)),
        }
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
}

#[derive(Default, Serialize, Deserialize, Debug)]
#[serde(default)]
#[serde(rename_all = "kebab-case")]
pub struct KeyspaceMeta {
    pub id: u32,
    pub name: String,
    pub state: String,
    pub created_at: u64,
    pub state_changed_at: u64,
}

#[derive(Default, Serialize, Deserialize, Debug)]
#[serde(default)]
#[serde(rename_all = "kebab-case")]
pub struct RegionInfo {
    pub id: u64,
    pub start_key: String,
    pub end_key: String,
}

#[derive(Default, Serialize, Deserialize, Debug)]
#[serde(default)]
#[serde(rename_all = "kebab-case")]
pub struct RegionsInfo {
    pub count: u64,
    pub regions: Vec<RegionInfo>,
}

#[derive(Default, Serialize, Deserialize, Debug)]
#[serde(default)]
#[serde(rename_all = "kebab-case")]
pub struct PdScheduleConfig {
    pub max_store_down_time: String,
}

#[derive(Default, Serialize, Deserialize, Debug)]
#[serde(default)]
#[serde(rename_all = "kebab-case")]
pub struct PdConfigFromApi {
    pub schedule: PdScheduleConfig,
}
