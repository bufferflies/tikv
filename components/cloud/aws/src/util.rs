// Copyright 2020 TiKV Project Authors. Licensed under Apache-2.0.
use std::{error::Error as StdError, io, sync::Arc};

use ::aws_smithy_runtime_api::client::orchestrator::HttpResponse;
use async_trait::async_trait;
use aws_config::{
    default_provider::credentials::DefaultCredentialsChain,
    environment::EnvironmentVariableRegionProvider,
    meta::region::{self, ProvideRegion, RegionProviderChain},
    profile::ProfileFileRegionProvider,
    provider_config::ProviderConfig,
    ConfigLoader, Region,
};
use aws_credential_types::provider::{error::CredentialsError, ProvideCredentials};
use aws_sdk_kms::config::SharedHttpClient;
use aws_sdk_s3::config::HttpClient;
use aws_smithy_runtime::client::http::hyper_014::HyperClientBuilder;
use chrono::{DateTime, Utc};
use cloud::metrics;
use futures::{Future, TryFutureExt};
use hyper::Client;
use hyper_tls::HttpsConnector;
use rusoto_credential::{AutoRefreshingProvider, AwsCredentials, ProvideAwsCredentials};
use tikv_util::{
    stream::{block_on_external_io, retry_ext, RetryError, RetryExt},
    warn,
};

const READ_BUF_SIZE: usize = 1024 * 1024 * 2;

const DEFAULT_REGION: &str = "us-east-1";

pub(crate) type SdkError<E, R = HttpResponse> =
    ::aws_smithy_runtime_api::client::result::SdkError<E, R>;

struct CredentialsErrorWrapper(CredentialsError);

impl From<CredentialsErrorWrapper> for CredentialsError {
    fn from(c: CredentialsErrorWrapper) -> CredentialsError {
        c.0
    }
}

impl std::fmt::Display for CredentialsErrorWrapper {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.0)?;
        Ok(())
    }
}

impl RetryError for CredentialsErrorWrapper {
    fn is_retryable(&self) -> bool {
        true
    }
}

pub fn new_http_client() -> SharedHttpClient {
    let mut hyper_builder = Client::builder();
    hyper_builder.http1_read_buf_exact_size(READ_BUF_SIZE);

    HyperClientBuilder::new()
        .hyper_builder(hyper_builder)
        .build(HttpsConnector::new())
}

pub fn new_credentials_provider_rusoto_wrapper() -> AutoRefreshingProvider<RusotoCredentialsWrapper>
{
    AutoRefreshingProvider::new(RusotoCredentialsWrapper(tokio::sync::OnceCell::new())).unwrap()
}

pub fn new_credentials_provider(http: impl HttpClient + 'static) -> DefaultCredentialsProvider {
    let fut = DefaultCredentialsProvider::new(http);
    if let Ok(hnd) = tokio::runtime::Handle::try_current() {
        tokio::task::block_in_place(move || hnd.block_on(fut))
    } else {
        block_on_external_io(fut)
    }
}

pub fn is_retryable<T>(error: &SdkError<T>) -> bool {
    match error {
        SdkError::TimeoutError(_) => true,
        SdkError::DispatchFailure(_) => true,
        SdkError::ResponseError(resp_err) => {
            let code = resp_err.raw().status();
            code.is_server_error() || code.as_u16() == http::StatusCode::REQUEST_TIMEOUT.as_u16()
        }
        _ => false,
    }
}

pub fn configure_endpoint(loader: ConfigLoader, endpoint: &str) -> ConfigLoader {
    if !endpoint.is_empty() {
        loader.endpoint_url(endpoint)
    } else {
        loader
    }
}

pub fn configure_region(loader: ConfigLoader, region: &str) -> io::Result<ConfigLoader> {
    if !region.is_empty() {
        Ok(loader.region(Region::new(region.to_owned())))
    } else {
        Ok(loader.region(DefaultRegionProvider::new()))
    }
}

pub async fn retry_and_count<G, T, F, E>(action: G, name: &'static str) -> Result<T, E>
where
    G: FnMut() -> F,
    F: Future<Output = Result<T, E>>,
    E: RetryError + std::fmt::Display,
{
    let id = uuid::Uuid::new_v4();
    retry_ext(
        action,
        RetryExt::default().with_fail_hook(move |err: &E| {
            warn!("aws request fails"; "err" => %err, "retry?" => %err.is_retryable(), "context" => %name, "uuid" => %id);
            metrics::CLOUD_ERROR_VEC.with_label_values(&["aws", name]).inc();
        }),
    ).await
}

#[derive(Debug)]
struct DefaultRegionProvider(RegionProviderChain);

impl DefaultRegionProvider {
    fn new() -> Self {
        let env_provider = EnvironmentVariableRegionProvider::new();
        let profile_provider = ProfileFileRegionProvider::builder().build();

        // same as default region resolving in rusoto
        let chain = RegionProviderChain::first_try(env_provider)
            .or_else(profile_provider)
            .or_else(Region::new(DEFAULT_REGION));

        Self(chain)
    }
}

impl ProvideRegion for DefaultRegionProvider {
    fn region(&self) -> region::future::ProvideRegion<'_> {
        ProvideRegion::region(&self.0)
    }
}

/// `RusotoCredentialsWrapper` is a compatibility layer used by the KVEngine S3
/// core, which still relies heavily on the Rusoto SDK for S3 interactions.
///
/// Instead of implementing credentials logic using Rusoto's native mechanisms,
/// this wrapper delegates to an AWS SDK credentials provider internally. This
/// allows us to unify credential sourcing logic while maintaining compatibility
/// with existing Rusoto-based code.
///
/// Since credential initialization may occur in both sync and async contexts
/// (e.g., via `block_on` in a sync path), the wrapper uses a `OnceCell` to
/// ensure safe, lazy, and thread-safe one-time initialization without risking
/// runtime panics.
///
/// In the future, once `s3fscore` is migrated off Rusoto, this layer can be
/// removed entirely.
pub struct RusotoCredentialsWrapper(tokio::sync::OnceCell<DefaultCredentialsProvider>);

#[async_trait]
impl rusoto_credential::ProvideAwsCredentials for RusotoCredentialsWrapper {
    async fn credentials(&self) -> Result<AwsCredentials, rusoto_credential::CredentialsError> {
        if !self.0.initialized() {
            let client = new_http_client();
            let default_provider = DefaultCredentialsProvider::new(client).await;
            if let Err(e) = self.0.set(default_provider) {
                return Err(rusoto_credential::CredentialsError::new(format!(
                    "cannot set oncecell {}",
                    e
                )));
            }
        }
        match self.0.get() {
            Some(c) => c
                .provide_credentials()
                .await
                .map(|a| {
                    warn!(
                        "provide cred expire at {:?}",
                        a.expiry().map(|t| DateTime::<Utc>::from(t))
                    );
                    rusoto_credential::AwsCredentials::new(
                        a.access_key_id().to_string(),
                        a.secret_access_key().to_string(),
                        a.session_token().map(|s| s.to_string()),
                        a.expiry().map(|t| DateTime::<Utc>::from(t)),
                    )
                })
                .map_err(|e| {
                    rusoto_credential::CredentialsError::new(format!(
                        "cannot translate credentials {}",
                        e
                    ))
                }),
            None => Err(rusoto_credential::CredentialsError::new(
                "cannot get oncecell",
            )),
        }
    }
}

#[derive(Debug)]
pub struct DefaultCredentialsProvider {
    default_provider: DefaultCredentialsChain,
}

unsafe impl Send for DefaultCredentialsProvider {}
unsafe impl Sync for DefaultCredentialsProvider {}

impl DefaultCredentialsProvider {
    async fn new(cli: impl HttpClient + 'static) -> Self {
        let cfg = ProviderConfig::default().with_http_client(cli);
        let default_provider = DefaultCredentialsChain::builder()
            .configure(cfg)
            .build()
            .await;
        Self { default_provider }
    }
}

impl ProvideCredentials for DefaultCredentialsProvider {
    fn provide_credentials<'a>(
        &'a self,
    ) -> aws_credential_types::provider::future::ProvideCredentials<'a>
    where
        Self: 'a,
    {
        aws_credential_types::provider::future::ProvideCredentials::new(async move {
            // Add exponential backoff for every error, because we cannot
            // distinguish the error type.
            let cred = retry_and_count(
                || {
                    #[cfg(test)]
                    fail::fail_point!("cred_err", |_| {
                        let cause: Box<dyn StdError + Send + Sync + 'static> =
                            String::from("injected error").into();
                        Box::pin(futures::future::err(CredentialsErrorWrapper(
                            CredentialsError::provider_error(cause),
                        )))
                            as std::pin::Pin<Box<dyn futures::Future<Output = _> + Send>>
                    });

                    Box::pin(
                        self.default_provider
                            .provide_credentials()
                            .map_err(|e| CredentialsErrorWrapper(e)),
                    )
                },
                "get_cred_on_premise",
            )
            .await
            .map_err(|e| e.0);

            cred.map_err(|e| {
                let msg = e
                    .source()
                    .map(|src_err| src_err.to_string())
                    .unwrap_or_else(|| e.to_string());
                let cause: Box<dyn StdError + Send + Sync + 'static> =
                    format_args!("Couldn't find AWS credentials in sources ({}).", msg)
                        .to_string()
                        .into();
                CredentialsError::provider_error(cause)
            })
        })
    }
}

/// ActiveRefreshingProvider try to refresh the credentials much early than the
/// expiration time. So we can tolerate credential service down for half of the
/// expire duration.
#[derive(Clone)]
pub struct ActiveRefreshingProvider {
    inner: Arc<dyn ProvideAwsCredentials + Send + Sync>,
    state: Arc<tokio::sync::Mutex<CredentialsState>>,
}

struct CredentialsState {
    current: Option<AwsCredentials>,
    refresh_at: Option<DateTime<Utc>>,
}

impl CredentialsState {
    fn set_credentials(&mut self, creds: AwsCredentials) {
        self.refresh_at = creds.expires_at().map(|expire_at| {
            let now = Utc::now();
            let expire_secs = expire_at - now;
            now + expire_secs / 2
        });
        self.current = Some(creds);
    }

    fn need_refresh(&self) -> bool {
        self.refresh_at.map(|t| t < Utc::now()).unwrap_or_default()
    }

    fn is_expired(&self) -> bool {
        match self.current.as_ref() {
            None => true,
            Some(creds) => {
                if let Some(expired_at) = creds.expires_at() {
                    *expired_at < Utc::now()
                } else {
                    false
                }
            }
        }
    }
}

impl ActiveRefreshingProvider {
    pub fn new(inner: Arc<dyn ProvideAwsCredentials + Send + Sync>) -> ActiveRefreshingProvider {
        ActiveRefreshingProvider {
            inner,
            state: Arc::new(tokio::sync::Mutex::new(CredentialsState {
                current: None,
                refresh_at: None,
            })),
        }
    }
}

#[async_trait]
impl ProvideAwsCredentials for ActiveRefreshingProvider {
    async fn credentials(&self) -> Result<AwsCredentials, rusoto_credential::CredentialsError> {
        let mut state = self.state.lock().await;
        if state.is_expired() {
            let creds = self.inner.credentials().await?;
            state.set_credentials(creds.clone());
        } else if state.need_refresh() {
            let creds_res = self.inner.credentials().await;
            match creds_res {
                Ok(creds) => {
                    state.set_credentials(creds.clone());
                }
                Err(err) => {
                    warn!("failed to refresh credentials {:?}", err);
                    state.refresh_at = Some(Utc::now() + chrono::Duration::seconds(3));
                }
            }
        }
        Ok(state.current.as_ref().unwrap().clone())
    }
}

#[cfg(test)]
mod tests {
    use std::{sync::Mutex, time::Duration};

    #[allow(unused_imports)]
    use super::*;

    #[cfg(feature = "failpoints")]
    #[tokio::test]
    async fn test_default_provider() {
        const AWS_WEB_IDENTITY_TOKEN_FILE: &str = "AWS_WEB_IDENTITY_TOKEN_FILE";

        let default_provider = DefaultCredentialsProvider::new(new_http_client()).await;
        std::env::set_var(AWS_WEB_IDENTITY_TOKEN_FILE, "tmp");
        // mock k8s env with web_identitiy_provider
        fail::cfg("cred_err", "return").unwrap();
        fail::cfg("retry_count", "return(1)").unwrap();
        let res = default_provider.provide_credentials().await;
        assert_eq!(res.is_err(), true);

        let err = res.unwrap_err();

        match err {
            CredentialsError::ProviderError(_) => {
                assert_eq!(
                    err.source().unwrap().to_string(),
                    "Couldn't find AWS credentials in sources (injected error)."
                )
            }
            err => panic!("unexpected error type: {}", err),
        }

        fail::remove("cred_err");
        fail::remove("retry_count");
        std::env::remove_var(AWS_WEB_IDENTITY_TOKEN_FILE);
    }

    #[derive(Clone)]
    struct MockProvider {
        creds: Arc<Mutex<Option<AwsCredentials>>>,
    }

    impl MockProvider {
        fn new() -> Self {
            MockProvider {
                creds: Arc::new(Mutex::new(None)),
            }
        }

        fn set_credentials(&self, creds: Option<AwsCredentials>) {
            let mut guard = self.creds.lock().unwrap();
            *guard = creds;
        }
    }

    #[async_trait]
    impl rusoto_credential::ProvideAwsCredentials for MockProvider {
        async fn credentials(&self) -> Result<AwsCredentials, rusoto_credential::CredentialsError> {
            let creds = self.creds.lock().unwrap();
            if creds.is_none() {
                return Err(rusoto_credential::CredentialsError::new("no credentials"));
            }
            Ok(creds.as_ref().unwrap().clone())
        }
    }

    fn new_mock_credential(expire_at: DateTime<Utc>) -> AwsCredentials {
        AwsCredentials::new(
            "access_key".to_string(),
            "access_secret".to_string(),
            None,
            Some(expire_at),
        )
    }

    #[tokio::test]
    async fn test_active_refreshing_provider() {
        let mock_provider = MockProvider::new();
        let provider = ActiveRefreshingProvider::new(Arc::new(mock_provider.clone()));
        let res = provider.credentials().await;
        res.unwrap_err();
        let expire_at_1 = Utc::now() + chrono::Duration::seconds(2);
        mock_provider.set_credentials(Some(new_mock_credential(expire_at_1)));
        let res = provider.credentials().await;
        res.unwrap();
        let state = provider.state.lock().await;
        assert!(state.refresh_at.is_some());
        let refresh_at = state.refresh_at.unwrap();
        drop(state);
        assert!(refresh_at < expire_at_1);

        let expire_at_2 = Utc::now() + chrono::Duration::seconds(3);
        mock_provider.set_credentials(Some(new_mock_credential(expire_at_2)));
        // Before refresh_at, we still get the original credential.
        let res = provider.credentials().await;
        let creds = res.unwrap();
        assert_eq!(creds.expires_at().as_ref().unwrap().clone(), expire_at_1);

        tokio::time::sleep(Duration::from_millis(1200)).await;

        // When after refresh at, we get the latest credential.
        let expire_at_3 = Utc::now() + chrono::Duration::seconds(2);
        mock_provider.set_credentials(Some(new_mock_credential(expire_at_3)));
        let res = provider.credentials().await;
        let creds = res.unwrap();
        assert_eq!(creds.expires_at().as_ref().unwrap().clone(), expire_at_3);

        tokio::time::sleep(Duration::from_millis(1200)).await;
        mock_provider.set_credentials(None);

        // After refresh at, we failed to refresh the credential, still use the old one.
        let creds = provider.credentials().await.unwrap();
        assert_eq!(creds.expires_at().as_ref().unwrap().clone(), expire_at_3);

        tokio::time::sleep(Duration::from_millis(1000)).await;

        // After expire, and the provider is still unavailable, we get the error.
        let res = provider.credentials().await;
        res.unwrap_err();

        // When inner provider is available, we get the latest credential.
        let expire_at_4 = Utc::now() + chrono::Duration::seconds(2);
        mock_provider.set_credentials(Some(new_mock_credential(expire_at_4)));
        let creds = provider.credentials().await.unwrap();
        assert_eq!(*creds.expires_at().as_ref().unwrap(), expire_at_4);
    }
}
