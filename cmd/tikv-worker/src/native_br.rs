// Copyright 2023 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    collections::HashMap,
    sync::{Arc, RwLock},
    thread,
    time::{Duration, Instant},
};

use http::{header, Method, Response, StatusCode};
use kvengine::dfs::S3Fs;
use native_br::{
    backup::get_all_backup_files, restore::RestoreConfig,
    restore_keyspace::restore_keyspace_with_cfg,
};
use pd_client::PdClient;
use security::SecurityConfig;
use tokio::runtime::Runtime;

use crate::{
    common::{get_u64_param, make_response},
    error::Error,
};

type Result<T> = std::result::Result<T, Error>;

/// Backup and restore keyspace API:
///
/// 1. backup:
///   GET    /backups?cluster_id=%date=%s
/// 2. restore keyspace:
///   PUT    /restore_keyspace?cluster_id=%d&name=%s&keyspace=%s
///   GET    /restore_keyspace?cluster_id=%d&keyspace=%s
///   DELETE /restore_keyspace?cluster_id=%d&keyspace=%s

pub(crate) async fn handle_backup(
    manager: Arc<NativeBrManger>,
    req: hyper::Request<hyper::Body>,
) -> hyper::Result<hyper::Response<hyper::Body>> {
    let query = req.uri().query().unwrap_or("");
    let query_pairs: HashMap<_, _> = url::form_urlencoded::parse(query.as_bytes()).collect();
    let cluster_id = get_u64_param(&query_pairs, "cluster_id").unwrap_or_default();
    if cluster_id != manager.get_cluster_id().unwrap() {
        return Ok(make_response(
            StatusCode::BAD_REQUEST,
            "cluster id mismatch",
        ));
    }
    match *req.method() {
        Method::GET => {
            let prefix = query_pairs.get("date");
            if prefix.is_none() {
                return Ok(make_response(
                    StatusCode::BAD_REQUEST,
                    "Backup Date is none",
                ));
            }
            match manager.list_backups(prefix.unwrap().to_string()).await {
                Ok(backups) => {
                    let json = serde_json::to_string(&backups).unwrap();
                    Ok(Response::builder()
                        .header(header::CONTENT_TYPE, "application/json")
                        .body(json.into())
                        .unwrap())
                }
                Err(e) => Ok(make_response(StatusCode::NOT_FOUND, e.to_string())),
            }
        }
        _ => Ok(make_response(StatusCode::BAD_REQUEST, "invalid method")),
    }
}

pub(crate) async fn handle_restore_keyspace(
    manager: Arc<NativeBrManger>,
    req: hyper::Request<hyper::Body>,
) -> hyper::Result<hyper::Response<hyper::Body>> {
    let query = req.uri().query().unwrap_or("");
    let query_pairs: HashMap<_, _> = url::form_urlencoded::parse(query.as_bytes()).collect();
    let cluster_id = get_u64_param(&query_pairs, "cluster_id").unwrap_or_default();
    if cluster_id != manager.get_cluster_id().unwrap() {
        return Ok(make_response(
            StatusCode::BAD_REQUEST,
            "cluster id mismatch",
        ));
    }
    let keyspace = query_pairs.get("keyspace");
    if keyspace.is_none() {
        return Ok(make_response(
            StatusCode::BAD_REQUEST,
            "Keyspace name is none",
        ));
    }
    let keyspace = keyspace.unwrap().to_string();
    match *req.method() {
        Method::GET => {
            if let Some(status) = manager.restore_status(keyspace) {
                let json = serde_json::to_string(&status).unwrap();
                Ok(Response::builder()
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(json.into())
                    .unwrap())
            } else {
                Ok(make_response(StatusCode::NOT_FOUND, ""))
            }
        }
        Method::PUT => {
            let backup_name = query_pairs.get("name");
            if backup_name.is_none() {
                return Ok(make_response(
                    StatusCode::BAD_REQUEST,
                    "Backup name is none",
                ));
            }
            let backup_name = backup_name.unwrap().to_string();
            match manager.restore_keyspace(keyspace, backup_name) {
                Ok(()) => Ok(make_response(StatusCode::OK, "Restore task is triggered")),
                Err(e) => Ok(make_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    e.to_string(),
                )),
            }
        }
        Method::DELETE => {
            if manager.delete_restore(keyspace).is_some() {
                Ok(make_response(StatusCode::OK, "Restore task is deleted"))
            } else {
                Ok(make_response(StatusCode::NOT_FOUND, ""))
            }
        }
        _ => Ok(make_response(StatusCode::BAD_REQUEST, "invalid method")),
    }
}

#[derive(Default, Debug, Serialize, Deserialize, Clone, PartialEq, PartialOrd)]
enum RestoreState {
    #[default]
    Init,
    Running,
    Succeed,
    Error,
}

pub(crate) struct RestoreStatus {
    state: RestoreState,
    start: Instant,
}

#[derive(Clone, Serialize, Deserialize, Debug)]
#[serde(rename_all = "kebab-case")]
pub(crate) struct ReadableRestoreStatus {
    state: RestoreState,
    duration: Duration,
}

pub(crate) struct BrContext {
    pub pd: pd_client::Config,
    pub security: SecurityConfig,
    pub pd_client: Arc<dyn PdClient>,
    pub working_path: Option<String>,
    pub s3fs: Arc<S3Fs>,
    pub runtime: Arc<Runtime>,
    pub restore_task: RwLock<HashMap<String, RestoreStatus>>,
}

impl BrContext {
    // Return false if the state if falling back.
    fn change_restore_state(&self, keyspace_name: &String, state: RestoreState) -> bool {
        let mut states = self.restore_task.write().unwrap();
        match states.get_mut(keyspace_name) {
            Some(restore_status) => {
                if restore_status.state >= state {
                    return false;
                }
                restore_status.state = state;
            }
            None => {
                debug_assert_eq!(state, RestoreState::Init);
                states.insert(
                    keyspace_name.to_owned(),
                    RestoreStatus {
                        state: RestoreState::Init,
                        start: Instant::now(),
                    },
                );
            }
        }
        true
    }

    fn restore_keyspace(&self, keyspace_name: String, backup_name: String) {
        self.change_restore_state(&keyspace_name, RestoreState::Running);
        let config = RestoreConfig {
            pd: self.pd.clone(),
            security: self.security.clone(),
            ..Default::default()
        };
        match restore_keyspace_with_cfg(
            config,
            &keyspace_name,
            &backup_name,
            self.working_path.as_deref(),
            self.s3fs.clone(),
            self.pd_client.clone(),
            &self.runtime,
        ) {
            Ok(()) => self.change_restore_state(&keyspace_name, RestoreState::Succeed),
            Err(_) => self.change_restore_state(&keyspace_name, RestoreState::Error),
        };
    }
}

pub(crate) struct NativeBrManger {
    pub context: Arc<BrContext>,
}

impl NativeBrManger {
    pub(crate) fn new(
        runtime: Arc<Runtime>,
        pd: pd_client::Config,
        security: SecurityConfig,
        pd_client: Arc<dyn PdClient>,
        s3fs: Arc<S3Fs>,
        working_path: Option<String>,
    ) -> Self {
        Self {
            context: Arc::new(BrContext {
                pd,
                security,
                pd_client,
                s3fs,
                working_path,
                runtime,
                restore_task: RwLock::new(HashMap::new()),
            }),
        }
    }

    async fn list_backups(&self, date: String) -> Result<Vec<String>> {
        Ok(get_all_backup_files(&self.context.s3fs, date).await?)
    }

    fn get_cluster_id(&self) -> Result<u64> {
        Ok(self.context.pd_client.get_cluster_id()?)
    }

    fn restore_keyspace(&self, keyspace_name: String, backup_name: String) -> Result<()> {
        if !self
            .context
            .change_restore_state(&keyspace_name, RestoreState::Init)
        {
            return Err(Error::RestoreError(
                "Restore task is already running".to_string(),
            ));
        }
        let context = self.context.clone();
        thread::spawn(move || context.restore_keyspace(keyspace_name, backup_name));
        Ok(())
    }

    fn restore_status(&self, keyspace: String) -> Option<ReadableRestoreStatus> {
        self.context
            .restore_task
            .read()
            .unwrap()
            .get(&keyspace)
            .map(|s| ReadableRestoreStatus {
                state: s.state.clone(),
                duration: Instant::now().duration_since(s.start),
            })
    }

    fn delete_restore(&self, keyspace: String) -> Option<RestoreStatus> {
        self.context.restore_task.write().unwrap().remove(&keyspace)
    }
}
