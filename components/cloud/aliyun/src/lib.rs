// Copyright 2025 TiKV Project Authors. Licensed under Apache-2.0.

use std::{fs, sync::Arc, time::Duration};

use async_trait::async_trait;
use aws::ActiveRefreshingProvider;
use chrono::{DateTime, Utc};
use crypto::{hmac::Hmac, mac::Mac, sha1::Sha1};
use hyper::{client::HttpConnector, header::HeaderValue, Body, Client, Method, Request};
use hyper_tls::HttpsConnector;
use rusoto_credential::{AwsCredentials, CredentialsError, ProvideAwsCredentials};
use serde_derive::{Deserialize, Serialize};

pub const OIDC_PROVIDER_ARN: &str = "ALIBABA_CLOUD_OIDC_PROVIDER_ARN";
pub const ROLE_ARN: &str = "ALIBABA_CLOUD_ROLE_ARN";
pub const TOKEN_FILE: &str = "ALIBABA_CLOUD_OIDC_TOKEN_FILE";
pub const REGION_ID: &str = "ALIBABA_CLOUD_REGION_ID";
pub const DOMAIN_STRING: &str = "aliyuncs.com";
pub const ECS_METADATA_DISABLED: &str = "ALIBABA_CLOUD_ECS_METADATA_DISABLED";
pub const ECS_METADATA_DISABLED_V1: &str = "ALIBABA_CLOUD_IMDSV1_DISABLED";
pub const ECS_ROLE_NAME: &str = "ALIBABA_CLOUD_ECS_METADATA";

const IMDS_BASE_URL: &str = "http://100.100.100.200";
const DEFAULT_METADATA_TOKEN_TTL_SECS: u64 = 21600;

pub const TIMEOUT: Duration = Duration::from_secs(60);

#[derive(Default)]
pub struct ChainProvider {
    providers: Vec<Arc<dyn ProvideAwsCredentials + Send + Sync>>,
}

impl ChainProvider {
    pub fn push<P>(&mut self, p: P)
    where
        P: ProvideAwsCredentials + Send + Sync + 'static,
    {
        self.providers.push(Arc::new(p));
    }
}

#[async_trait]
impl ProvideAwsCredentials for ChainProvider {
    async fn credentials(&self) -> Result<AwsCredentials, CredentialsError> {
        for p in self.providers.iter() {
            match p.credentials().await {
                Ok(c) => return Ok(c),
                Err(_) => {
                    // ignore this credential error, try next
                    continue;
                }
            }
        }
        Err(CredentialsError::new("all credential providers failed"))
    }
}

pub struct EcsRamRoleCredentialsProvider {
    role_name: Option<String>,
    disable_imdsv1: bool,
    client: hyper::Client<HttpsConnector<HttpConnector>>,
}
#[derive(Deserialize, Debug)]
struct EcsRamRoleCredentials {
    #[serde(rename = "Code")]
    code: Option<String>,
    #[serde(rename = "AccessKeyId")]
    access_key_id: Option<String>,
    #[serde(rename = "AccessKeySecret")]
    access_key_secret: Option<String>,
    #[serde(rename = "SecurityToken")]
    security_token: Option<String>,
    #[serde(rename = "Expiration")]
    expiration: Option<String>,
}

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

fn is_truthy(v: &str) -> bool {
    matches!(v.to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on")
}

impl EcsRamRoleCredentialsProvider {
    pub fn new() -> Result<EcsRamRoleCredentialsProvider, CredentialsError> {
        if let Ok(v) = std::env::var(ECS_METADATA_DISABLED) {
            if is_truthy(&v) {
                return Err(CredentialsError::new("ECS metadata is disabled by env"));
            }
        }

        let disable_imdsv1: bool = std::env::var(ECS_METADATA_DISABLED_V1)
            .ok()
            .map(|v| is_truthy(&v))
            .unwrap_or(false);

        let role_name: Option<String> = std::env::var(ECS_ROLE_NAME).ok();

        let https = HttpsConnector::new();
        let client: Client<_, Body> = Client::builder().build(https);

        Ok(EcsRamRoleCredentialsProvider {
            role_name,
            disable_imdsv1,
            client,
        })
    }

    async fn get_metadata_token(&self) -> Result<Option<String>, CredentialsError> {
        let url = format!("{IMDS_BASE_URL}/latest/api/token");
        let mut req = Request::builder()
            .method(Method::PUT)
            .uri(url)
            .body(Body::empty())
            .map_err(|e| CredentialsError::new(format!("build IMDS token request: {e}")))?;
        // X-aliyun-ecs-metadata-token-ttl-seconds
        req.headers_mut().insert(
            "X-aliyun-ecs-metadata-token-ttl-seconds",
            HeaderValue::from_str(&DEFAULT_METADATA_TOKEN_TTL_SECS.to_string())
                .map_err(|e| CredentialsError::new(format!("invalid ttl header: {e}")))?,
        );

        // Try IMDSv2 first; if it fails, attempt IMDSv1 unless explicitly disabled
        let resp = match tokio::time::timeout(TIMEOUT, self.client.request(req)).await {
            Ok(Ok(r)) => r,
            Ok(Err(e)) => {
                if self.disable_imdsv1 {
                    return Err(CredentialsError::new(format!(
                        "get metadata token failed: {e}"
                    )));
                } else {
                    return Ok(None);
                }
            }
            Err(_) => {
                if self.disable_imdsv1 {
                    return Err(CredentialsError::new("get metadata token timeout"));
                } else {
                    return Ok(None);
                }
            }
        };

        let status = resp.status();
        let body =
            match tokio::time::timeout(TIMEOUT, hyper::body::to_bytes(resp.into_body())).await {
                Ok(Ok(b)) => b,
                Ok(Err(e)) => {
                    if self.disable_imdsv1 {
                        return Err(CredentialsError::new(format!(
                            "read token body failed: {e}"
                        )));
                    } else {
                        return Ok(None);
                    }
                }
                Err(_) => {
                    if self.disable_imdsv1 {
                        return Err(CredentialsError::new("read token body timeout"));
                    } else {
                        return Ok(None);
                    }
                }
            };

        if !status.is_success() {
            if self.disable_imdsv1 {
                return Err(CredentialsError::new(format!(
                    "refresh ECS sts token err (IMDSv2 token), httpStatus: {status}, body={}",
                    String::from_utf8_lossy(&body)
                )));
            } else {
                return Ok(None);
            }
        }

        let token = String::from_utf8(body.to_vec())
            .map_err(|e| CredentialsError::new(format!("token utf8 error: {e}")))?;
        Ok(Some(token.trim().to_string()))
    }

    async fn get_role_name(&self) -> Result<String, CredentialsError> {
        if let Some(name) = &self.role_name {
            return Ok(name.clone());
        }

        let url = format!("{IMDS_BASE_URL}/latest/meta-data/ram/security-credentials/");
        let mut req = Request::builder()
            .method(Method::GET)
            .uri(url)
            .body(Body::empty())
            .map_err(|e| CredentialsError::new(format!("build roleName request: {e}")))?;

        if let Some(token) = self.get_metadata_token().await? {
            req.headers_mut().insert(
                "X-aliyun-ecs-metadata-token",
                HeaderValue::from_str(&token)
                    .map_err(|e| CredentialsError::new(format!("set token header: {e}")))?,
            );
        }

        let resp = tokio::time::timeout(TIMEOUT, self.client.request(req))
            .await
            .map_err(|_| CredentialsError::new("get role name timeout"))?
            .map_err(|e| CredentialsError::new(format!("get role name failed: {e}")))?;

        if resp.status() != hyper::StatusCode::OK {
            return Err(CredentialsError::new(format!(
                "get role name failed: http {}",
                resp.status()
            )));
        }

        let body = tokio::time::timeout(TIMEOUT, hyper::body::to_bytes(resp.into_body()))
            .await
            .map_err(|_| CredentialsError::new("read role name body timeout"))?
            .map_err(|e| CredentialsError::new(format!("read role name body: {e}")))?;

        let name = String::from_utf8(body.to_vec())
            .map_err(|e| CredentialsError::new(format!("role name utf8 error: {e}")))?;
        Ok(name.trim().to_string())
    }

    async fn get_session_credentials(
        &self,
    ) -> Result<(String, String, String, DateTime<Utc>), CredentialsError> {
        let role_name = self.get_role_name().await?;
        let url = format!("{IMDS_BASE_URL}/latest/meta-data/ram/security-credentials/{role_name}");
        let mut req = Request::builder()
            .method(Method::GET)
            .uri(url)
            .body(Body::empty())
            .map_err(|e| CredentialsError::new(format!("build creds request: {e}")))?;

        if let Some(token) = self.get_metadata_token().await? {
            req.headers_mut().insert(
                "X-aliyun-ecs-metadata-token",
                HeaderValue::from_str(&token)
                    .map_err(|e| CredentialsError::new(format!("set token header: {e}")))?,
            );
        }

        let resp = tokio::time::timeout(TIMEOUT, self.client.request(req))
            .await
            .map_err(|_| CredentialsError::new("get creds timeout"))?
            .map_err(|e| CredentialsError::new(format!("refresh ECS sts token err: {e}")))?;

        let status = resp.status();
        let body = tokio::time::timeout(TIMEOUT, hyper::body::to_bytes(resp.into_body()))
            .await
            .map_err(|_| CredentialsError::new("read creds body timeout"))?
            .map_err(|e| CredentialsError::new(format!("read creds body: {e}")))?;

        if status != hyper::StatusCode::OK {
            return Err(CredentialsError::new(format!(
                "refresh ECS sts token err, httpStatus: {status}, message={}",
                String::from_utf8_lossy(&body)
            )));
        }

        let data: EcsRamRoleCredentials = serde_json::from_slice(&body)
            .map_err(|e| CredentialsError::new(format!("json unmarshal fail: {e}")))?;

        if data.code.as_deref() != Some("Success") {
            return Err(CredentialsError::new(format!(
                "refresh ECS sts token err, Code:{} is not Success",
                data.code.as_deref().unwrap_or("None")
            )));
        }

        let (ak, sk, token, exp) = match (
            data.access_key_id,
            data.access_key_secret,
            data.security_token,
            data.expiration,
        ) {
            (Some(ak), Some(sk), Some(tok), Some(exp)) => (ak, sk, tok, exp),
            _ => {
                return Err(CredentialsError::new(
                    "refresh ECS sts token err, fail to get credentials",
                ));
            }
        };

        let expiration = chrono::DateTime::parse_from_rfc3339(&exp)
            .map_err(|e| CredentialsError::new(format!("parse expiration: {e}")))?
            .with_timezone(&Utc);

        Ok((ak, sk, token, expiration))
    }
}

#[async_trait]
impl ProvideAwsCredentials for EcsRamRoleCredentialsProvider {
    async fn credentials(&self) -> Result<AwsCredentials, CredentialsError> {
        let (ak, sk, token, expiration) = self.get_session_credentials().await?;
        Ok(AwsCredentials::new(ak, sk, Some(token), Some(expiration)))
    }
}

pub fn new_credential_provider() -> Result<ActiveRefreshingProvider, CredentialsError> {
    let mut chain_provider = ChainProvider::default();
    if let Ok(oidc_provider) = AssumeRoleWithOidcProvider::new() {
        chain_provider.push(oidc_provider);
    }
    if let Ok(ecs_ram_provider) = EcsRamRoleCredentialsProvider::new() {
        chain_provider.push(ecs_ram_provider);
    }
    let auto_refreshing_provider = ActiveRefreshingProvider::new(Arc::new(chain_provider));
    Ok(auto_refreshing_provider)
}

pub async fn decrypt_master_key(
    cypher_text_blob: &str,
    region: &str,
) -> Result<Vec<u8>, CredentialsError> {
    let credential_provider = new_credential_provider()?;
    let cred = credential_provider.credentials().await?;
    let resp = decrypt_key_with_credential(cypher_text_blob, &cred, region).await?;
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
