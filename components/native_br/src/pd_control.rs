// Copyright 2023 TiKV Project Authors. Licensed under Apache-2.0.

use std::str::FromStr;

use bstr::ByteSlice;
use bytes::Bytes;
use http::{Request, Uri};
use hyper::Body;
use slog_global::debug;
use tikv_util::box_err;

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Sync + Send>>;

const PD_KEYSPACE_PATH: &str = "/pd/api/v2/keyspaces";

/// PdControl provides access to HTTP APIs of PD, which are not included in gRPC
/// interface. It's also expected to act like the tool `pd-ctl`.
#[derive(Clone)]
pub struct PdControl {
    config: pd_client::Config,
    // TODO: support TLS
    _security_config: security::SecurityConfig,
}

impl PdControl {
    pub fn new(config: pd_client::Config, security_config: security::SecurityConfig) -> Self {
        Self {
            config,
            _security_config: security_config,
        }
    }

    async fn request_pd_restful(&self, query: String, post_data: Option<Vec<u8>>) -> Result<Bytes> {
        let client = hyper::Client::new();
        let mut err = None;
        for endpoint in &self.config.endpoints {
            let uri = if !endpoint.starts_with("http") {
                format!("http://{endpoint}{query}")
            } else {
                format!("{endpoint}/{query}")
            };
            let uri = Uri::from_str(&uri).unwrap();
            let resp = match post_data {
                Some(ref post_data) => {
                    let req = Request::post(uri)
                        .body(Body::from(post_data.to_vec()))
                        .unwrap();
                    client.request(req).await
                }
                None => client.get(uri).await,
            };
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

    pub async fn get_keyspace_by_name(&self, keyspace_name: &str) -> Result<KeyspaceMeta> {
        let query = format!("{PD_KEYSPACE_PATH}/{}", keyspace_name);
        match self.request_pd_restful(query, None).await {
            Ok(resp) => {
                let keyspace = serde_json::from_slice(&resp)?;
                debug!("get_keyspace: {:?}", keyspace);
                Ok(keyspace)
            }
            Err(err) => Err(box_err!("get_keyspace_by_name error: {:?}", err)),
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
