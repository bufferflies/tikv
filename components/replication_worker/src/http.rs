use chrono::Utc;
use hyper::{Body, Method, Request, Response, Result, StatusCode};
use serde::{Deserialize, Serialize};

use crate::{config::ReplicaConfig, ReplicationWorker};

#[derive(Serialize)]
struct ErrorResponse {
    error_msg: String,
    error_code: String,
}

#[derive(Serialize)]
struct ChangefeedListResponse {
    total: usize,
    items: Vec<ChangefeedItem>,
}

#[derive(Serialize)]
struct ChangefeedItem {
    id: String,
    state: String,
    checkpoint_tso: u64,
    checkpoint_time: String,
    error: Option<ErrorInfo>,
}

#[derive(Deserialize)]
struct ChangefeedRequest {
    changefeed_id: Option<String>,
    replica_config: Option<ReplicaConfig>,
    sink_uri: String,
    start_ts: Option<u64>,
    target_ts: Option<u64>,
}

// Keep Serialize for response types
#[derive(Serialize)]
struct ChangefeedResponse {
    admin_job_type: i32,
    checkpoint_time: String,
    checkpoint_ts: u64,
    config: ReplicaConfig,
    create_time: String,
    creator_version: String,
    error: Option<ErrorInfo>,
    id: String,
    resolved_ts: u64,
    sink_uri: String,
    start_ts: u64,
    state: String,
    target_ts: u64,
    task_status: Vec<TaskStatus>,
}

#[derive(Serialize)]
struct ErrorInfo {
    addr: String,
    code: String,
    message: String,
}

#[derive(Serialize)]
struct TaskStatus {
    capture_id: String,
    table_ids: Vec<u64>,
}

impl ReplicationWorker {
    fn error_response(status: StatusCode, msg: &str, code: &str) -> Response<Body> {
        let error = ErrorResponse {
            error_msg: msg.to_string(),
            error_code: code.to_string(),
        };
        Response::builder()
            .status(status)
            .body(Body::from(serde_json::to_string(&error).unwrap()))
            .unwrap()
    }

    pub async fn handle_http_request(&self, req: Request<Body>) -> Result<Response<Body>> {
        let keyspace_id = {
            if let Some(keyspace_str) = extract_param(req.uri(), "keyspace_id") {
                match keyspace_str.parse::<u64>() {
                    Ok(keyspace_id) => keyspace_id,
                    Err(_) => {
                        return Ok(Self::error_response(
                            StatusCode::BAD_REQUEST,
                            "Invalid keyspace_id",
                            "CDC:ErrInvalidParam",
                        ));
                    }
                }
            } else {
                return Ok(Self::error_response(
                    StatusCode::BAD_REQUEST,
                    "keyspace_id is required",
                    "CDC:ErrInvalidParam",
                ));
            }
        };
        let response = match (req.method(), req.uri().path()) {
            (&Method::GET, "/cdc/api/v2/changefeeds") => {
                Self::handle_list_changefeeds(keyspace_id, req.uri())
            }
            (&Method::GET, path) if path.starts_with("/cdc/api/v2/changefeeds/") => {
                Self::handle_get_changefeed(keyspace_id, path)
            }
            (&Method::POST, "/cdc/api/v2/changefeeds") => {
                let body_bytes = hyper::body::to_bytes(req.into_body()).await?;
                Self::handle_create_changefeed(keyspace_id, &body_bytes)
            }

            (&Method::DELETE, path) if path.starts_with("/cdc/api/v2/changefeeds/") => {
                Self::handle_delete_changefeed(keyspace_id, path)
            }

            _ => Self::error_response(
                StatusCode::NOT_FOUND,
                "Route not found",
                "CDC:ErrAPIRouteNotFound",
            ),
        };

        Ok(response)
    }

    fn handle_delete_changefeed(_keyspace_id: u64, path: &str) -> Response<Body> {
        if let Some(_change_id) = path.strip_prefix("/cdc/api/v2/changefeeds/") {
            // TODO: Implement actual deletion logic
            Response::builder()
                .status(StatusCode::OK)
                .body(Body::empty())
                .unwrap()
        } else {
            Self::error_response(
                StatusCode::BAD_REQUEST,
                "Invalid changefeed ID",
                "CDC:ErrInvalidChangefeedID",
            )
        }
    }

    fn handle_list_changefeeds(_keyspace_id: u64, uri: &hyper::Uri) -> Response<Body> {
        fn is_valid_state(state: &str) -> bool {
            matches!(
                state,
                "all" | "normal" | "stopped" | "error" | "failed" | "finished"
            )
        }

        let state = extract_param(uri, "state");
        if let Some(state) = state {
            if !is_valid_state(state) {
                return Self::error_response(
                    StatusCode::BAD_REQUEST,
                    "Invalid state parameter",
                    "CDC:ErrInvalidParam",
                );
            }
        }
        // TODO: Implement actual filtering based on state
        let response = ChangefeedListResponse {
            total: 2,
            items: vec![
                ChangefeedItem {
                    id: "test".to_string(),
                    state: "normal".to_string(),
                    checkpoint_tso: 439749918821711874,
                    checkpoint_time: "2023-02-27 23:46:52.888".to_string(),
                    error: None,
                },
                ChangefeedItem {
                    id: "test2".to_string(),
                    state: "normal".to_string(),
                    checkpoint_tso: 439749918821711874,
                    checkpoint_time: "2023-02-27 23:46:52.888".to_string(),
                    error: None,
                },
            ],
        };

        Response::builder()
            .status(StatusCode::OK)
            .body(Body::from(serde_json::to_string(&response).unwrap()))
            .unwrap()
    }

    fn handle_get_changefeed(_keyspace_id: u64, path: &str) -> Response<Body> {
        if let Some(changefeed_id) = path.strip_prefix("/cdc/api/v2/changefeeds/") {
            if changefeed_id.is_empty() {
                return Self::error_response(
                    StatusCode::BAD_REQUEST,
                    "Invalid changefeed ID",
                    "CDC:ErrInvalidChangefeedID",
                );
            }
            // TODO: Implement actual changefeed lookup
            let response = ChangefeedResponse {
                admin_job_type: 0,
                checkpoint_time: "2023-02-27 23:46:52.888".to_string(),
                checkpoint_ts: 439749918821711874,
                config: ReplicaConfig::default(),
                create_time: Utc::now().to_string(),
                creator_version: "v8.5.1".to_string(),
                error: None,
                id: changefeed_id.to_string(),
                resolved_ts: 439749918821711874,
                sink_uri: "blackhole://".to_string(),
                start_ts: 439749918821711874,
                state: "normal".to_string(),
                target_ts: 0,
                task_status: vec![],
            };
            Response::builder()
                .status(StatusCode::OK)
                .body(Body::from(serde_json::to_string(&response).unwrap()))
                .unwrap()
        } else {
            Self::error_response(
                StatusCode::BAD_REQUEST,
                "Invalid changefeed ID",
                "CDC:ErrInvalidChangefeedID",
            )
        }
    }

    fn handle_create_changefeed(_keyspace_id: u64, body_bytes: &[u8]) -> Response<Body> {
        match serde_json::from_slice::<ChangefeedRequest>(body_bytes) {
            Ok(changefeed_req) => {
                if changefeed_req.sink_uri.is_empty() {
                    return Self::error_response(
                        StatusCode::BAD_REQUEST,
                        "sink_uri is required",
                        "CDC:ErrInvalidRequestBody",
                    );
                }
                changefeed_req.replica_config.as_ref().and_then(|config| {
                    if let Err(msg) = config.validate() {
                        return Some(Self::error_response(
                            StatusCode::BAD_REQUEST,
                            &msg,
                            "CDC:ErrInvalidRequestBody",
                        ));
                    }
                    None
                });
                let response = ChangefeedResponse {
                    admin_job_type: 0,
                    checkpoint_time: "0".to_string(),
                    checkpoint_ts: 0,
                    config: changefeed_req.replica_config.unwrap_or_default(),
                    create_time: Utc::now().to_string(),
                    creator_version: "0.0.0".to_owned(),
                    error: None,
                    id: changefeed_req.changefeed_id.unwrap_or_default(),
                    resolved_ts: 0,
                    sink_uri: changefeed_req.sink_uri,
                    start_ts: changefeed_req.start_ts.unwrap_or(0),
                    state: "normal".to_string(),
                    target_ts: changefeed_req.target_ts.unwrap_or(0),
                    task_status: vec![],
                };
                Response::builder()
                    .status(StatusCode::OK)
                    .body(Body::from(serde_json::to_string(&response).unwrap()))
                    .unwrap()
            }
            Err(e) => Self::error_response(
                StatusCode::BAD_REQUEST,
                &e.to_string(),
                "CDC:ErrInvalidRequestBody",
            ),
        }
    }
}

fn extract_param<'a>(uri: &'a hyper::Uri, key: &str) -> Option<&'a str> {
    uri.query().and_then(|q| {
        q.split('&')
            .find(|p| p.starts_with(key))
            .map(|p| p.split('=').nth(1).unwrap_or(""))
    })
}

#[cfg(test)]
mod tests {
    use hyper::http::request;

    use super::*;

    impl ReplicationWorker {
        fn new_test() -> Self {
            // TODO: Add proper initialization when needed
            Self {}
        }
    }

    #[tokio::test]
    async fn test_create_changefeed() {
        let worker = ReplicationWorker::new_test();
        let req = request::Builder::new()
            .method(Method::POST)
            .uri("/cdc/api/v2/changefeeds?keyspace_id=1")
            .body(Body::from(
                r#"{
                    "changefeed_id": "test1",
                    "sink_uri": "blackhole://"
                }"#,
            ))
            .unwrap();

        let resp = worker.handle_http_request(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_list_changefeeds() {
        let worker = ReplicationWorker::new_test();
        let req = request::Builder::new()
            .method(Method::GET)
            .uri("/cdc/api/v2/changefeeds?keyspace_id=1&state=normal")
            .body(Body::empty())
            .unwrap();

        let resp = worker.handle_http_request(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_get_changefeed() {
        let worker = ReplicationWorker::new_test();
        let req = request::Builder::new()
            .method(Method::GET)
            .uri("/cdc/api/v2/changefeeds/test1?keyspace_id=1")
            .body(Body::empty())
            .unwrap();

        let resp = worker.handle_http_request(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_delete_changefeed() {
        let worker = ReplicationWorker::new_test();
        let req = request::Builder::new()
            .method(Method::DELETE)
            .uri("/cdc/api/v2/changefeeds/test1?keyspace_id=1")
            .body(Body::empty())
            .unwrap();

        let resp = worker.handle_http_request(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_invalid_request() {
        let worker = ReplicationWorker::new_test();
        // Missing keyspace_id for list changefeeds
        let req = request::Builder::new()
            .method(Method::GET)
            .uri("/cdc/api/v2/changefeeds")
            .body(Body::empty())
            .unwrap();
        let resp = worker.handle_http_request(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);

        // Missing keyspace_id for get changefeed
        let req = request::Builder::new()
            .method(Method::GET)
            .uri("/cdc/api/v2/changefeeds/test1")
            .body(Body::empty())
            .unwrap();
        let resp = worker.handle_http_request(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);

        // Missing keyspace_id for create changefeed
        let req = request::Builder::new()
            .method(Method::POST)
            .uri("/cdc/api/v2/changefeeds")
            .body(Body::from(
                r#"{
                "changefeed_id": "test1",
                "sink_uri": "blackhole://"
            }"#,
            ))
            .unwrap();
        let resp = worker.handle_http_request(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);

        // Missing keyspace_id for delete changefeed
        let req = request::Builder::new()
            .method(Method::DELETE)
            .uri("/cdc/api/v2/changefeeds/test1")
            .body(Body::empty())
            .unwrap();
        let resp = worker.handle_http_request(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        // Invalid state parameter
        let req = request::Builder::new()
            .method(Method::GET)
            .uri("/cdc/api/v2/changefeeds?keyspace_id=1&state=invalid")
            .body(Body::empty())
            .unwrap();
        let resp = worker.handle_http_request(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);

        // Invalid changefeed ID format
        let req = request::Builder::new()
            .method(Method::GET)
            .uri("/cdc/api/v2/changefeeds/?keyspace_id=1")
            .body(Body::empty())
            .unwrap();
        let resp = worker.handle_http_request(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);

        // Missing required field in create request
        let req = request::Builder::new()
            .method(Method::POST)
            .uri("/cdc/api/v2/changefeeds?keyspace_id=1")
            .body(Body::from("{}"))
            .unwrap();
        let resp = worker.handle_http_request(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }
}
