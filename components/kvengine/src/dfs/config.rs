// Copyright 2022 TiKV Project Authors. Licensed under Apache-2.0.

use std::{env, error::Error};

#[derive(Clone, Serialize, Deserialize, PartialEq, Debug, Default)]
#[serde(default)]
#[serde(rename_all = "kebab-case")]
pub struct Config {
    pub prefix: String,

    pub s3_endpoint: String,

    pub s3_key_id: String,

    pub s3_secret_key: String,

    pub s3_bucket: String,

    pub s3_region: String,

    pub remote_compactor_addr: String,
}

impl Config {
    #[allow(dead_code)]
    fn validate(&self) -> Result<(), Box<dyn Error>> {
        // TODO(x) validate dfs config
        Ok(())
    }

    pub fn override_from_env(&mut self) {
        let dfs_s3_bucket = env::var("DFS_S3_BUCKET").unwrap_or_default();
        if self.s3_bucket.is_empty() && self.s3_endpoint.is_empty() && !dfs_s3_bucket.is_empty() {
            self.s3_bucket = dfs_s3_bucket;
            self.s3_endpoint = env::var("DFS_S3_ENDPOINT").unwrap_or_default();
            self.prefix = env::var("DFS_PREFIX").unwrap_or_default();
            self.s3_key_id = env::var("DFS_S3_KEY_ID").unwrap_or_default();
            self.s3_secret_key = env::var("DFS_S3_SECRET_KEY").unwrap_or_default();
            self.s3_region = env::var("DFS_S3_REGION").unwrap_or_default();
        }
    }
}
