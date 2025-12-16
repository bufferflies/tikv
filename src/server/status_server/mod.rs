// Copyright 2018 TiKV Project Authors. Licensed under Apache-2.0.

/// Provides profilers for TiKV.
pub mod profile;
use std::{
    marker::PhantomData,
    time::{Duration, Instant},
};

use collections::HashMap;
use futures::compat::Compat01As03;
use hyper::{Body, Request, Response, StatusCode};
pub use profile::start_one_cpu_profile;
use tikv_util::timer::GLOBAL_TIMER_HANDLE;

use crate::config::LogLevel;

static TIMER_CANCELED: &str = "tokio timer canceled";

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
struct LogLevelRequest {
    pub log_level: LogLevel,
}

pub struct StatusServer<E, R> {
    _router: R,
    _snap: PhantomData<E>,
}

impl<E, R> StatusServer<E, R>
where
    E: 'static,
    R: 'static + Send,
{
    pub async fn dump_cpu_prof_to_resp(req: Request<Body>) -> hyper::Result<Response<Body>> {
        let query = req.uri().query().unwrap_or("");
        let query_pairs: HashMap<_, _> = url::form_urlencoded::parse(query.as_bytes()).collect();

        let seconds: u64 = match query_pairs.get("seconds") {
            Some(val) => match val.parse() {
                Ok(val) => val,
                Err(err) => return Ok(make_response(StatusCode::BAD_REQUEST, err.to_string())),
            },
            None => 10,
        };

        let frequency: i32 = match query_pairs.get("frequency") {
            Some(val) => match val.parse() {
                Ok(val) => val,
                Err(err) => return Ok(make_response(StatusCode::BAD_REQUEST, err.to_string())),
            },
            None => 99, /* Default frequency of sampling. 99Hz to avoid coincide with special
                         * periods */
        };

        let prototype_content_type: hyper::http::HeaderValue =
            hyper::http::HeaderValue::from_str("application/protobuf").unwrap();
        let output_protobuf = req.headers().get("Content-Type") == Some(&prototype_content_type);

        let timer = GLOBAL_TIMER_HANDLE.delay(Instant::now() + Duration::from_secs(seconds));
        let end = async move {
            Compat01As03::new(timer)
                .await
                .map_err(|_| TIMER_CANCELED.to_owned())
        };
        match start_one_cpu_profile(end, frequency, output_protobuf).await {
            Ok(body) => {
                info!("dump cpu profile successfully");
                let mut response = Response::builder()
                    .header(
                        "Content-Disposition",
                        "attachment; filename=\"cpu_profile\"",
                    )
                    .header("Content-Length", body.len());
                response = if output_protobuf {
                    response.header("Content-Type", mime::APPLICATION_OCTET_STREAM.to_string())
                } else {
                    response.header("Content-Type", mime::IMAGE_SVG.to_string())
                };
                Ok(response.body(body.into()).unwrap())
            }
            Err(e) => {
                info!("dump cpu profile fail: {}", e);
                Ok(make_response(StatusCode::INTERNAL_SERVER_ERROR, e))
            }
        }
    }
}

fn make_response<T>(status_code: StatusCode, message: T) -> Response<Body>
where
    T: Into<Body>,
{
    Response::builder()
        .status(status_code)
        .body(message.into())
        .unwrap()
}
