// Copyright 2025 TiKV Project Authors. Licensed under Apache-2.0.

use std::{fs, time::Duration};

use async_trait::async_trait;
use chrono::Utc;
use crypto::{hmac::Hmac, mac::Mac, sha1::Sha1};
use hyper::client::HttpConnector;
use hyper_tls::HttpsConnector;
use rusoto_credential::{
    AutoRefreshingProvider, AwsCredentials, CredentialsError, ProvideAwsCredentials,
};
use serde_derive::{Deserialize, Serialize};

pub const OIDC_PROVIDER_ARN: &str = "ALIBABA_CLOUD_OIDC_PROVIDER_ARN";
pub const ROLE_ARN: &str = "ALIBABA_CLOUD_ROLE_ARN";
pub const TOKEN_FILE: &str = "ALIBABA_CLOUD_OIDC_TOKEN_FILE";
pub const REGION_ID: &str = "ALIBABA_CLOUD_REGION_ID";
pub const DOMAIN_STRING: &str = "aliyuncs.com";

pub const TIMEOUT: Duration = Duration::from_secs(60);

pub struct AssumeRoleWithOidcProvider {
    region: String,
    oidc_provider_arn: String,
    role_arn: String,
    oidc_token_file: String,
    client: hyper::Client<HttpsConnector<HttpConnector>>,
}

#[derive(Serialize, Deserialize, Default, Debug)]
#[serde(rename_all = "PascalCase")]
struct AssumeRoleResponse {
    credentials: Credentials,
}

#[derive(Serialize, Deserialize, Default, Debug)]
#[serde(rename_all = "PascalCase")]
struct Credentials {
    access_key_id: String,
    access_key_secret: String,
    security_token: String,
    expiration: String,
}

impl AssumeRoleWithOidcProvider {
    pub fn new() -> Result<AssumeRoleWithOidcProvider, CredentialsError> {
        let region = std::env::var(REGION_ID)
            .map_err(|_| CredentialsError::new("failed to get region id from env"))?;
        let oidc_provider_arn = std::env::var(OIDC_PROVIDER_ARN)
            .map_err(|_| CredentialsError::new("failed to get oidc provider arn from env"))?;
        let role_arn = std::env::var(ROLE_ARN)
            .map_err(|_| CredentialsError::new("failed to get role arn from env"))?;
        let oidc_token_file = std::env::var(TOKEN_FILE)
            .map_err(|_| CredentialsError::new("failed to get oidc token file from env"))?;
        let client = hyper::Client::builder().build(hyper_tls::HttpsConnector::new());
        Ok(AssumeRoleWithOidcProvider {
            region,
            oidc_provider_arn,
            role_arn,
            oidc_token_file,
            client,
        })
    }
}

#[async_trait]
impl ProvideAwsCredentials for AssumeRoleWithOidcProvider {
    async fn credentials(&self) -> Result<AwsCredentials, CredentialsError> {
        let token = String::from_utf8(fs::read(&self.oidc_token_file)?)?;
        let timestamp = Utc::now().to_rfc3339();
        let params = vec![
            ("Action", "AssumeRoleWithOIDC"),
            ("Format", "JSON"),
            ("Version", "2015-04-01"),
            ("RoleSessionName", "TiKV"),
            ("OIDCProviderArn", &self.oidc_provider_arn),
            ("RoleArn", &self.role_arn),
            ("Timestamp", &timestamp),
            ("OIDCToken", &token),
        ];
        let query_string = build_query(&params);
        let uri = format!(
            "https://sts.{}.{}/?{}",
            self.region, DOMAIN_STRING, query_string,
        );
        let request = hyper::Request::post(uri)
            .body(hyper::Body::empty())
            .unwrap();
        let resp = self.client.request(request).await?;
        if !resp.status().is_success() {
            return Err(CredentialsError::new(format!(
                "failed to AssumeRoleWithOIDC: {}",
                resp.status()
            )));
        }
        let body = hyper::body::to_bytes(resp.into_body()).await?;
        let resp: AssumeRoleResponse = serde_json::from_slice(&body)?;
        let expiration = chrono::DateTime::parse_from_rfc3339(&resp.credentials.expiration)?;
        Ok(AwsCredentials::new(
            resp.credentials.access_key_id,
            resp.credentials.access_key_secret,
            Some(resp.credentials.security_token),
            Some(expiration.with_timezone(&Utc)),
        ))
    }
}

pub fn new_credential_provider()
-> Result<AutoRefreshingProvider<AssumeRoleWithOidcProvider>, CredentialsError> {
    let assume_role_provider = AssumeRoleWithOidcProvider::new()?;
    let auto_refreshing_provider = AutoRefreshingProvider::new(assume_role_provider)?;
    Ok(auto_refreshing_provider)
}

pub async fn decrypt_master_key(cypher_text_blob: &str) -> Result<Vec<u8>, CredentialsError> {
    let credential_provider = AssumeRoleWithOidcProvider::new()?;
    let cred = credential_provider.credentials().await?;
    let resp =
        decrypt_key_with_credential(cypher_text_blob, &cred, &credential_provider.region).await?;
    base64::decode(resp.plaintext).map_err(|_| CredentialsError::new("failed to decode base64"))
}

async fn decrypt_key_with_credential(
    cypher_text_blob: &str,
    cred: &AwsCredentials,
    region: &str,
) -> Result<DecryptResponse, CredentialsError> {
    let params = vec![
        ("Action", "Decrypt"),
        ("CiphertextBlob", cypher_text_blob),
        ("Version", "2016-01-20"),
    ];
    let singed_query = build_signed_query("GET", &params, cred);
    let uri = format!("https://kms.{}.aliyuncs.com/?{}", region, singed_query);
    let client = hyper::Client::builder().build(hyper_tls::HttpsConnector::new());
    let request = hyper::Request::get(uri).body(hyper::Body::empty()).unwrap();
    let resp = client.request(request).await?;
    if !resp.status().is_success() {
        return Err(CredentialsError::new(format!(
            "failed to decrypt: {}",
            resp.status()
        )));
    }
    let body = hyper::body::to_bytes(resp.into_body()).await?;
    let resp: DecryptResponse = serde_json::from_slice(&body)?;
    Ok(resp)
}

#[derive(Serialize, Deserialize, Default, Debug)]
#[serde(rename_all = "PascalCase")]
struct DecryptResponse {
    plaintext: String,
}

fn build_signed_query(
    method: &str,
    input_params: &[(&str, &str)],
    cred: &AwsCredentials,
) -> String {
    let mut all_params = input_params.to_vec();
    all_params.push(("AccessKeyId", cred.aws_access_key_id()));
    let timestamp = Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string();
    all_params.push(("Timestamp", &timestamp));
    all_params.push(("Format", "JSON"));
    all_params.push(("SignatureMethod", "HMAC-SHA1"));
    all_params.push(("SignatureVersion", "1.0"));
    all_params.push(("SecurityToken", cred.token().as_ref().unwrap()));
    all_params.sort_by(|a, b| a.0.cmp(b.0));
    let canonical_query = build_query(&all_params);
    let string_to_sign = format!("{}&%2F&{}", method, percent_encode(&canonical_query));
    let key_secret = format!("{}&", cred.aws_secret_access_key());
    let mut mac = Hmac::new(Sha1::new(), key_secret.as_bytes());
    mac.input(string_to_sign.as_bytes());
    let signature = base64::encode(mac.result().code());
    all_params.push(("Signature", &signature));
    build_query(&all_params)
}

fn build_query(params: &[(&str, &str)]) -> String {
    params
        .iter()
        .map(|(k, v)| format!("{}={}", percent_encode(k), percent_encode(v)))
        .collect::<Vec<_>>()
        .join("&")
}

// Percent-encode for aliyun signature
fn percent_encode(input: &str) -> String {
    let mut encoded = String::new();
    for byte in input.as_bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                encoded.push(*byte as char);
            }
            _ => encoded.push_str(&format!("%{:02X}", byte)),
        }
    }
    encoded
}
