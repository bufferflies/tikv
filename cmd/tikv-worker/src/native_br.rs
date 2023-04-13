// Copyright 2023 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    collections::{HashMap, HashSet},
    sync::{Arc, RwLock},
    thread,
};

use chrono::{DateTime, NaiveDateTime, Utc};
use http::{Method, StatusCode};
use hyper::{Body, Response};
use kvengine::dfs::S3Fs;
use native_br::{
    backup, backup::IncrementalBackupFile, restore::RestoreConfig,
    restore_keyspace::restore_keyspace_with_cfg,
};
use pd_client::PdClient;
use security::SecurityConfig;
use serde::Deserialize;
use tikv_util::{debug, error, info, HandyRwLock};
use tokio::runtime::Runtime;

use crate::{
    common::{get_u64_param, make_json_response, make_response},
    error::Error,
};

type Result<T> = std::result::Result<T, Error>;

const MAX_RESTORE_CONCURRENCY: usize = 20;
const MAX_BACKUP_COUNT_PER_PAGE: usize = 1000; // Same with dfs list.
const JSON_TIME_FORMAT: &str = "%Y-%m-%d %H:%M:%S%.3f"; // e.g. 2006-01-02 15:04:05.000
const BACKUP_NAME_FORMAT: &str = "%Y%m%d%H%M%S";

pub(crate) const BACKUPS_API_PATH: &str = "/api/v1/backups";
pub(crate) const RESTORE_KEYSPACE_API_PATH: &str = "/api/v1/restore_keyspace/";
pub(crate) const WHITELIST_API_PATH: &str = "/api/v1/native_br/whitelist/";

/// Backup and restore keyspace API:
///
/// 1. backup:
///   * GET    /api/v1/backups?cluster_id=%d&last_backup_time=<JSON_TIME_FORMAT>
///
/// 2. restore keyspace:
///   * PUT    /api/v1/restore_keyspace/<restore_id>?cluster_id=%d&keyspace=%s&
///     backup_id=%d&backup_name=%s
///   * GET    /api/v1/restore_keyspace/<restore_id>?cluster_id=%d&keyspace=%s
///   * DELETE /api/v1/restore_keyspace/<restore_id>?cluster_id=%d&keyspace=%s
///   * GET    /api/v1/restore_keyspace/?cluster_id=%d
/// 3. whitelist
///   * GET    /api/v1/native_br/whitelist/?cluster_id=%d
///   * GET    /api/v1/native_br/whitelist/<keyspace>?cluster_id=%d

pub(crate) async fn handle_backup(
    manager: Arc<NativeBrManger>,
    req: hyper::Request<hyper::Body>,
) -> hyper::Result<hyper::Response<hyper::Body>> {
    let query = req.uri().query().unwrap_or("");
    let query_pairs: HashMap<_, _> = url::form_urlencoded::parse(query.as_bytes()).collect();
    match get_u64_param(&query_pairs, "cluster_id") {
        Some(cluster_id) if cluster_id == manager.get_cluster_id().unwrap() => {}
        _ => {
            return Ok(make_response(
                StatusCode::BAD_REQUEST,
                "Cluster ID mismatch",
            ));
        }
    }
    match *req.method() {
        Method::GET => {
            let last_backup_time = query_pairs.get("last_backup_time").and_then(|s| {
                NaiveDateTime::parse_from_str(s, JSON_TIME_FORMAT)
                    .ok()
                    .map(|t| DateTime::<Utc>::from_utc(t, Utc))
            });
            if last_backup_time.is_none() {
                return Ok(make_response(
                    StatusCode::BAD_REQUEST,
                    "Last backup time is none or invalid",
                ));
            }
            let start_backup_time = last_backup_time
                .unwrap()
                .checked_add_signed(chrono::Duration::seconds(1))
                .unwrap();
            let max_count = if let Some(count) = query_pairs.get("max_count") {
                match count.parse::<usize>() {
                    Ok(c) => c,
                    Err(e) => {
                        return Ok(make_response(
                            StatusCode::BAD_REQUEST,
                            format!("max count is invalid {:?}", e),
                        ));
                    }
                }
            } else {
                MAX_BACKUP_COUNT_PER_PAGE
            };

            match manager.list_backups(&start_backup_time, max_count).await {
                Ok((backups, has_more)) => {
                    let resp = ListBackupResponse {
                        items: backups.into_iter().map(Into::into).collect(),
                        has_more,
                    };
                    Ok(make_json_response(StatusCode::OK, &resp))
                }
                Err(e) => Ok(make_response(
                    StatusCode::NOT_FOUND,
                    format!("Backups not found: {:?}", e),
                )),
            }
        }
        _ => Ok(make_response(StatusCode::BAD_REQUEST, "Invalid method")),
    }
}

pub(crate) async fn handle_restore_keyspace(
    manager: Arc<NativeBrManger>,
    req: hyper::Request<hyper::Body>,
) -> hyper::Result<hyper::Response<hyper::Body>> {
    let query = req.uri().query().unwrap_or("");
    let query_pairs: HashMap<_, _> = url::form_urlencoded::parse(query.as_bytes()).collect();

    match get_u64_param(&query_pairs, "cluster_id") {
        Some(cluster_id) if cluster_id == manager.get_cluster_id().unwrap() => {}
        _ => {
            return Ok(make_response(
                StatusCode::BAD_REQUEST,
                "Cluster ID mismatch",
            ));
        }
    }

    let sub_path = req
        .uri()
        .path()
        .strip_prefix(RESTORE_KEYSPACE_API_PATH)
        .unwrap();
    if sub_path.is_empty() && *req.method() == Method::GET {
        return handle_get_all_restore_task(&manager);
    }

    let restore_id = match sub_path.parse::<u64>() {
        Ok(id) => id,
        Err(_) => {
            return Ok(make_response(StatusCode::BAD_REQUEST, "Invalid restore id"));
        }
    };

    let keyspace = query_pairs.get("keyspace");
    if keyspace.is_none() {
        return Ok(make_response(
            StatusCode::BAD_REQUEST,
            "Keyspace name is none",
        ));
    }
    let keyspace = keyspace.unwrap().to_string();
    if !manager.is_keyspace_allowed(&keyspace) {
        return Ok(make_response(
            StatusCode::FORBIDDEN,
            format!("Keyspace {} is not in whitelist", keyspace),
        ));
    }

    match *req.method() {
        Method::GET => {
            debug!(
                "{} request to GET restore_keyspace, restore_id {}",
                keyspace, restore_id
            );
            handle_restore_status(&manager, restore_id, &keyspace)
        }
        Method::PUT => {
            let backup_id = match get_u64_param(&query_pairs, "backup_id") {
                Some(id) => id,
                None => {
                    return Ok(make_response(
                        StatusCode::BAD_REQUEST,
                        "Backup ID is invalid",
                    ));
                }
            };
            let backup_name = query_pairs
                .get("backup_name")
                .map(|s| s.to_string())
                .unwrap_or_default();
            let backup = IncrementalBackupFile::from_id(backup_id);
            if backup_name != backup.created_at().format(BACKUP_NAME_FORMAT).to_string() {
                return Ok(make_response(
                    StatusCode::BAD_REQUEST,
                    "Backup ID & name mismatch",
                ));
            }

            debug!(
                "{} request to PUT restore_keyspace, restore_id {}, backup {:?}",
                keyspace, restore_id, backup
            );
            match manager.restore_keyspace(restore_id, keyspace.clone(), backup.name().to_string())
            {
                Ok(true) => {
                    info!(
                        "{} restore keyspace started, restore_id {}, backup {:?}",
                        keyspace, restore_id, backup
                    );
                    let resp = RestoreProgressResponse {
                        status: RestoreState::Pending,
                        error: String::new(),
                        duration: 0,
                        id: restore_id,
                        keyspace,
                    };
                    Ok(make_json_response(StatusCode::CREATED, &resp))
                }
                Ok(false) => {
                    info!(
                        "{} restore keyspace ignored, restore_id {}, backup {:?}",
                        keyspace, restore_id, backup
                    );
                    // Return current status when `restore_keyspace` request is ignored.
                    handle_restore_status(&manager, restore_id, &keyspace)
                }
                Err(Error::RestoreKeyspaceTaskConflict(conflict_restore_id)) => {
                    info!(
                        "{} restore keyspace conflict, restore_id {}, backup {:?}, conflict restore_id {}",
                        keyspace, restore_id, backup, conflict_restore_id,
                    );
                    let resp = RestoreConflictResponse {
                        keyspace,
                        id: restore_id,
                        conflict_restore_id,
                    };
                    Ok(make_json_response(StatusCode::CONFLICT, &resp))
                }
                Err(e) => {
                    error!(
                        "{} restore keyspace error, restore_id {}, backup {:?}, error {:?}",
                        keyspace, restore_id, backup, e
                    );
                    handle_error(e)
                }
            }
        }
        Method::DELETE => {
            debug!(
                "{} request to DELETE restore_keyspace, restore_id {}",
                keyspace, restore_id
            );
            match manager.delete_restore(restore_id, &keyspace) {
                Ok(Some(true)) => {
                    info!(
                        "{} restore keyspace task deleted, restore_id {}",
                        keyspace, restore_id
                    );
                    Ok(make_response(StatusCode::OK, "Restore task deleted"))
                }
                Ok(Some(false)) => {
                    info!(
                        "{} delete restore keyspace task ignored, restore_id {}",
                        keyspace, restore_id
                    );
                    Ok(make_response(
                        StatusCode::CONFLICT,
                        "Restore task is not in final state",
                    ))
                }
                Ok(None) => {
                    info!(
                        "{} delete restore keyspace task not found, restore_id {}",
                        keyspace, restore_id
                    );
                    Ok(make_response(
                        StatusCode::NOT_FOUND,
                        format!("Restore task not found: {}", restore_id),
                    ))
                }
                Err(err) => {
                    error!(
                        "{} delete restore keyspace task error, restore_id {}, error {:?}",
                        keyspace, restore_id, err
                    );
                    handle_error(err)
                }
            }
        }
        _ => Ok(make_response(StatusCode::BAD_REQUEST, "Invalid method")),
    }
}

fn handle_get_all_restore_task(manager: &Arc<NativeBrManger>) -> hyper::Result<Response<Body>> {
    let resp_vec: Vec<RestoreProgressResponse> = manager
        .get_all_restore_task()
        .into_iter()
        .map(|(id, task)| RestoreProgressResponse::from_restore_status(id, task))
        .collect();
    Ok(make_json_response(StatusCode::OK, &resp_vec))
}

fn handle_restore_status(
    manager: &Arc<NativeBrManger>,
    restore_id: u64,
    keyspace: &str,
) -> hyper::Result<Response<Body>> {
    match manager.restore_status(restore_id, keyspace) {
        Ok(Some(task)) => {
            let resp = RestoreProgressResponse::from_restore_status(restore_id, task);
            Ok(make_json_response(StatusCode::ACCEPTED, &resp))
        }
        Ok(None) => Ok(make_response(
            StatusCode::NOT_FOUND,
            format!("Restore task not found: {}", restore_id),
        )),
        Err(err) => {
            error!(
                "{} query restore keyspace task status error, restore_id {}, error {:?}",
                keyspace, restore_id, err
            );
            handle_error(err)
        }
    }
}

fn handle_error(err: Error) -> hyper::Result<Response<Body>> {
    match err {
        Error::CheckError(msg) => Ok(make_response(StatusCode::BAD_REQUEST, msg)),
        err => Ok(make_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            err.to_string(),
        )),
    }
}

pub(crate) async fn handle_native_br_whitelist(
    manager: Arc<NativeBrManger>,
    req: hyper::Request<hyper::Body>,
) -> hyper::Result<hyper::Response<hyper::Body>> {
    let query = req.uri().query().unwrap_or("");
    let query_pairs: HashMap<_, _> = url::form_urlencoded::parse(query.as_bytes()).collect();

    match get_u64_param(&query_pairs, "cluster_id") {
        Some(cluster_id) if cluster_id == manager.get_cluster_id().unwrap() => {}
        _ => {
            return Ok(make_response(
                StatusCode::BAD_REQUEST,
                "Cluster ID mismatch",
            ));
        }
    }

    let keyspace = req
        .uri()
        .path()
        .strip_prefix(WHITELIST_API_PATH)
        .unwrap()
        .to_string();
    match *req.method() {
        Method::GET => {
            if keyspace.is_empty() {
                // Get all whitelist, for debug use only.
                Ok(make_json_response(
                    StatusCode::OK,
                    &manager.config.read().unwrap().whitelist,
                ))
            } else {
                let resp = WhitelistResponse {
                    is_allowed: manager.is_keyspace_allowed(&keyspace),
                    keyspace,
                };
                Ok(make_json_response(StatusCode::OK, &resp))
            }
        }
        _ => Ok(make_response(
            StatusCode::BAD_REQUEST,
            "Invalid whitelist method",
        )),
    }
}

#[derive(Default, Serialize, Deserialize, Debug)]
#[serde(default)]
struct BackupItem {
    id: u64,
    name: String,
    time: String,
}

impl From<IncrementalBackupFile> for BackupItem {
    fn from(backup: IncrementalBackupFile) -> Self {
        Self {
            id: backup.id(),
            name: backup.created_at().format(BACKUP_NAME_FORMAT).to_string(),
            time: backup.created_at().format(JSON_TIME_FORMAT).to_string(),
        }
    }
}

#[derive(Default, Serialize, Deserialize, Debug)]
#[serde(default)]
struct ListBackupResponse {
    items: Vec<BackupItem>,
    has_more: bool,
}

#[derive(Default, Serialize, Deserialize, Debug)]
#[serde(default)]
struct RestoreProgressResponse {
    status: RestoreState,
    error: String,
    duration: i64, // in seconds
    id: u64,
    keyspace: String,
}

impl RestoreProgressResponse {
    fn from_restore_status(restore_id: u64, status: RestoreTask) -> Self {
        let duration = status
            .end
            .unwrap_or_else(|| Utc::now())
            .signed_duration_since(status.start);
        Self {
            status: status.state,
            error: status.error,
            duration: duration.num_seconds().clamp(0, i64::MAX),
            id: restore_id,
            keyspace: status.keyspace_name,
        }
    }
}

#[derive(Default, Serialize, Deserialize, Debug)]
#[serde(default)]
struct RestoreConflictResponse {
    keyspace: String,
    id: u64,
    conflict_restore_id: u64,
}

#[derive(Default, Debug, Serialize, Deserialize, Clone, PartialEq, PartialOrd)]
enum RestoreState {
    #[default]
    Pending,
    Init,
    Running,
    Succeed,
    Error,
}

impl RestoreState {
    fn is_final(&self) -> bool {
        *self == Self::Succeed || *self == Self::Error
    }

    // Error -> Init
    fn is_retry(&self, new_state: &Self) -> bool {
        *self == Self::Error && *new_state == Self::Init
    }
}

#[derive(Default, Serialize, Deserialize, Debug)]
#[serde(default)]
struct WhitelistResponse {
    keyspace: String,
    is_allowed: bool,
}

#[derive(Clone)]
pub(crate) struct RestoreTask {
    state: RestoreState,
    keyspace_name: String,
    error: String,
    start: DateTime<Utc>,
    end: Option<DateTime<Utc>>,
}

type TasksMap = HashMap<u64 /* restore_id */, RestoreTask>;

type KeyspacesMap = HashMap<String /* keyspace */, u64 /* restore_id */>;

pub(crate) struct BrContext {
    pub pd: pd_client::Config,
    pub security: SecurityConfig,
    pub pd_client: Arc<dyn PdClient>,
    pub working_path: Option<String>,
    pub s3fs: Arc<S3Fs>,
    pub runtime: Arc<Runtime>,
    pub restore_tasks: RwLock<TasksMap>,
    pub keyspace_tasks: RwLock<KeyspacesMap>,
}

impl BrContext {
    // Return false if the state is falling back.
    fn change_restore_state(
        &self,
        restore_id: u64,
        keyspace_name: &str,
        new_state: RestoreState,
        err: Option<Error>,
    ) -> Result<bool> {
        let new_task = |tasks: &mut TasksMap| -> Result<()> {
            let mut keyspaces = self.keyspace_tasks.write().unwrap();
            if let Some(&restore_id) = keyspaces.get(keyspace_name) {
                return Err(Error::RestoreKeyspaceTaskConflict(restore_id));
            }
            keyspaces.insert(keyspace_name.to_owned(), restore_id);
            tasks.insert(
                restore_id,
                RestoreTask {
                    state: RestoreState::Init,
                    keyspace_name: keyspace_name.to_string(),
                    error: String::new(),
                    start: Utc::now(),
                    end: None,
                },
            );
            Ok(())
        };

        let mut tasks = self.restore_tasks.write().unwrap();
        match tasks.get_mut(&restore_id) {
            None => {
                debug_assert_eq!(new_state, RestoreState::Init);
                new_task(&mut tasks)?;
            }
            Some(task) if task.state.is_retry(&new_state) => {
                check_task(task, keyspace_name)?;
                new_task(&mut tasks)?;
            }
            Some(task) => {
                check_task(task, keyspace_name)?;
                if task.state >= new_state {
                    return Ok(false);
                }
                task.state = new_state;
                task.error = err.map(|err| format!("{:?}", err)).unwrap_or_default();

                if task.state.is_final() {
                    task.end = Some(Utc::now());
                    self.keyspace_tasks.write().unwrap().remove(keyspace_name);
                }
            }
        }
        Ok(true)
    }

    fn restore_keyspace(
        &self,
        restore_id: u64,
        keyspace_name: String,
        backup_name: String,
    ) -> Result<()> {
        self.change_restore_state(restore_id, &keyspace_name, RestoreState::Running, None)?;
        let config = RestoreConfig {
            pd: self.pd.clone(),
            security: self.security.clone(),
            ..Default::default()
        };
        let _ = match restore_keyspace_with_cfg(
            config,
            &keyspace_name,
            &backup_name,
            self.working_path.as_deref(),
            self.s3fs.clone(),
            self.pd_client.clone(),
            &self.runtime,
        ) {
            Ok(()) => {
                self.change_restore_state(restore_id, &keyspace_name, RestoreState::Succeed, None)
            }
            Err(err) => {
                error!(
                    "{} restore_keyspace error, restore_id {}, error {:?}",
                    keyspace_name, restore_id, err
                );
                self.change_restore_state(
                    restore_id,
                    &keyspace_name,
                    RestoreState::Error,
                    Some(err.into()),
                )
            }
        };
        Ok(())
    }
}

#[derive(Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
#[serde(default)]
#[serde(rename_all = "kebab-case")]
pub struct WhiteList {
    enable: bool,
    list: HashSet<String>,
}

impl WhiteList {
    pub(crate) fn is_allowed(&self, keyspace: &String) -> bool {
        !self.enable || self.list.contains(keyspace)
    }
}

#[derive(Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
#[serde(default)]
#[serde(rename_all = "kebab-case")]
pub struct NativeBrConfig {
    whitelist: WhiteList,
}

pub(crate) struct NativeBrManger {
    pub context: Arc<BrContext>,
    pub config: RwLock<NativeBrConfig>,
}

impl NativeBrManger {
    pub(crate) fn new(
        runtime: Arc<Runtime>,
        pd: pd_client::Config,
        security: SecurityConfig,
        pd_client: Arc<dyn PdClient>,
        s3fs: Arc<S3Fs>,
        working_path: Option<String>,
        config: NativeBrConfig,
    ) -> Self {
        Self {
            context: Arc::new(BrContext {
                pd,
                security,
                pd_client,
                s3fs,
                working_path,
                runtime,
                restore_tasks: Default::default(),
                keyspace_tasks: Default::default(),
            }),
            config: RwLock::new(config),
        }
    }

    pub(crate) fn update_config(&self, config: NativeBrConfig) {
        let mut ori_config = self.config.write().unwrap();
        if *ori_config != config {
            info!(
                "Update native br config from {:?} to {:?}",
                *ori_config, config
            );
            *ori_config = config;
        }
    }

    async fn list_backups(
        &self,
        start_backup_time: &DateTime<Utc>,
        max_count: usize,
    ) -> Result<(Vec<IncrementalBackupFile>, bool)> {
        let (backups, has_more) = backup::get_all_incremental_backups(
            &self.context.s3fs,
            &start_backup_time.date(),
            Some(&start_backup_time.time()),
            max_count,
        )
        .await?;
        let backups = backups
            .into_iter()
            .filter_map(|file_path| IncrementalBackupFile::try_from_full_path(&file_path))
            .collect::<Vec<_>>();
        Ok((backups, has_more))
    }

    fn get_cluster_id(&self) -> Result<u64> {
        Ok(self.context.pd_client.get_cluster_id()?)
    }

    /// Return:
    ///   Ok(true): task started.
    ///   Ok(false): request ignored due to duplicated.
    ///   Err(err): error occurred.
    fn restore_keyspace(
        &self,
        restore_id: u64,
        keyspace_name: String,
        backup_name: String,
    ) -> Result<bool> {
        if self.get_all_restore_task().len() >= MAX_RESTORE_CONCURRENCY {
            return Err(Error::ReachConcurrencyLimit(MAX_RESTORE_CONCURRENCY));
        }

        if self.context.change_restore_state(
            restore_id,
            &keyspace_name,
            RestoreState::Init,
            None,
        )? {
            let context = self.context.clone();
            thread::spawn(move || context.restore_keyspace(restore_id, keyspace_name, backup_name));
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Return:
    ///   Ok(Some): succeed.
    ///   Ok(None): not found.
    ///   Err(err): error occurred.
    fn restore_status(&self, restore_id: u64, keyspace_name: &str) -> Result<Option<RestoreTask>> {
        let tasks = self.context.restore_tasks.rl();
        if let Some(task) = tasks.get(&restore_id) {
            check_task(task, keyspace_name)?;
            Ok(Some(task.clone()))
        } else {
            Ok(None)
        }
    }

    /// Return all restore tasks in memory.
    fn get_all_restore_task(&self) -> HashMap<u64, RestoreTask> {
        self.context.restore_tasks.rl().clone()
    }

    /// Return:
    ///   Ok(Some(true)): deleted.
    ///   Ok(Some(false)): ignored due state is not final.
    ///   Ok(None): restore task not found.
    ///   Err(err): error occurred.
    fn delete_restore(&self, restore_id: u64, keyspace_name: &str) -> Result<Option<bool>> {
        let mut tasks = self.context.restore_tasks.wl();
        if let Some(task) = tasks.get(&restore_id) {
            check_task(task, keyspace_name)?;
            if task.state.is_final() {
                tasks.remove(&restore_id);
                Ok(Some(true))
            } else {
                Ok(Some(false))
            }
        } else {
            Ok(None)
        }
    }

    fn is_keyspace_allowed(&self, keyspace: &String) -> bool {
        self.config.read().unwrap().whitelist.is_allowed(keyspace)
    }
}

fn check_task(task: &RestoreTask, keyspace_name: &str) -> Result<()> {
    if task.keyspace_name != keyspace_name {
        Err(Error::CheckError(
            "restore id & keyspace not match".to_string(),
        ))
    } else {
        Ok(())
    }
}
