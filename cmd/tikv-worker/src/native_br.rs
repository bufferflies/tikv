// Copyright 2023 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    borrow::Cow,
    collections::{HashMap, HashSet},
    ops::Deref,
    sync::{Arc, Mutex, RwLock},
    thread,
    time::Duration,
};

use chrono::{DateTime, NaiveDateTime, Utc};
use http::{Method, StatusCode};
use hyper::{Body, Response};
use kvengine::dfs::S3Fs;
use native_br::{
    backup,
    backup::IncrementalBackupFile,
    backup_worker::BackupWorker,
    restore_keyspace::{
        restore_keyspace_with_cfg, ReportRestoreStepTrait, RestoreStep, RestoredKeyspace,
    },
};
use pd_client::PdClient;
use serde::Deserialize;
use tikv::storage::mvcc::TimeStamp;
use tikv_util::{config::ReadableDuration, debug, error, info, time::Instant, HandyRwLock};
use tokio::runtime::Runtime;

use crate::{
    common::{get_u64_param, make_json_response, make_response},
    error::Error,
    metrics::{NATIVE_BR_COUNTER_VEC, NATIVE_BR_HISTOGRAM_VEC},
    Config,
};

pub(crate) type Result<T> = std::result::Result<T, Error>;

const MIN_PITR_INTERVAL_GAP_SECONDS: i64 = 1; // 1s
pub(crate) const MAX_RESTORE_CONCURRENCY: usize = 128;
const MAX_BACKUP_COUNT_PER_PAGE: usize = 1000; // Same with dfs list.
const JSON_TIME_FORMAT: &str = "%Y-%m-%d %H:%M:%S%.3f"; // e.g. 2006-01-02 15:04:05.000
const BACKUP_NAME_FORMAT: &str = "%Y%m%d%H%M%S";
const INSTANT_BACKUP_INTERVAL: Duration = Duration::from_secs(60);
pub(crate) const BACKUPS_API_PATH: &str = "/api/v1/backups";
pub(crate) const RESTORE_KEYSPACE_API_PATH: &str = "/api/v1/restore_keyspace/";
pub(crate) const WHITELIST_API_PATH: &str = "/api/v1/native_br/whitelist/";

/// Backup and restore keyspace API:
///
/// 1. backup:
///   * GET    /api/v1/backups?cluster_id=%d&last_backup_time=<JSON_TIME_FORMAT>
///
/// 2. restore keyspace:
///    Specify backup_id&backup_name for normal restore or point_in_time for
///    pitr
///   * PUT    /api/v1/restore_keyspace/<restore_id>?cluster_id=%d&keyspace=%s&
///     backup_id=%d&backup_name=%s[&source_keyspace=%s]&
///     point_in_time=<JSON_TIME_FORMAT>
///   * GET    /api/v1/restore_keyspace/<restore_id>?cluster_id=%d&keyspace=%s
///   * DELETE /api/v1/restore_keyspace/<restore_id>?cluster_id=%d&keyspace=%s
///   * GET    /api/v1/restore_keyspace/?cluster_id=%d
/// 3. whitelist
///   * GET    /api/v1/native_br/whitelist/?cluster_id=%d
///   * GET    /api/v1/native_br/whitelist/<keyspace>?cluster_id=%d

pub(crate) async fn handle_backup(
    manager: Arc<NativeBrManager>,
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
            let ob_start_time = Instant::now();
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
                    NATIVE_BR_HISTOGRAM_VEC
                        .with_label_values(&["list_backup"])
                        .observe(ob_start_time.saturating_elapsed_secs());
                    Ok(make_json_response(StatusCode::OK, &resp))
                }
                Err(e) => {
                    NATIVE_BR_COUNTER_VEC
                        .with_label_values(&["list_backup_fail"])
                        .inc();
                    Ok(make_response(
                        StatusCode::NOT_FOUND,
                        format!("Backups not found: {:?}", e),
                    ))
                }
            }
        }
        _ => Ok(make_response(StatusCode::BAD_REQUEST, "Invalid method")),
    }
}

fn parse_restore_type(query_pairs: &HashMap<Cow<'_, str>, Cow<'_, str>>) -> Result<RestoreType> {
    let existed_backup =
        query_pairs.get("backup_id").is_some() || query_pairs.get("backup_name").is_some();
    let pitr = query_pairs.get("point_in_time").is_some();

    if existed_backup && pitr {
        return Err(Error::CheckError(
            "Request for normal restore and PiTR at the same time".to_string(),
        ));
    }
    if existed_backup {
        Ok(RestoreType::Normal)
    } else {
        Ok(RestoreType::Pitr)
    }
}

async fn get_backup_from_query(
    manager: &Arc<NativeBrManager>,
    query_pairs: &HashMap<Cow<'_, str>, Cow<'_, str>>,
    restore_type: RestoreType,
) -> Result<RestoreSource> {
    match restore_type {
        RestoreType::Normal => {
            let backup_id = match get_u64_param(query_pairs, "backup_id") {
                Some(id) => id,
                None => {
                    return Err(Error::CheckError("Backup ID is invalid".to_string()));
                }
            };
            let backup_name = query_pairs
                .get("backup_name")
                .map(|s| s.to_string())
                .unwrap_or_default();
            let backup = IncrementalBackupFile::from_id(backup_id);
            if backup_name != backup.created_at().format(BACKUP_NAME_FORMAT).to_string() {
                return Err(Error::CheckError("Backup ID & name mismatch".to_string()));
            }
            Ok(RestoreSource::ExistFile((backup, None)))
        }
        RestoreType::Pitr => {
            let ts = query_pairs
                .get("point_in_time")
                .map(|s| s.to_string())
                .unwrap_or_default();
            if ts.is_empty() {
                return Err(Error::CheckError("Recover time is empty".to_string()));
            }
            let utc_time = NaiveDateTime::parse_from_str(&ts, JSON_TIME_FORMAT)
                .map(|t| DateTime::<Utc>::from_utc(t, Utc))?;
            // To avoid the system time gap between tikv-api and pd nodes.
            if Utc::now().signed_duration_since(utc_time).num_seconds()
                < MIN_PITR_INTERVAL_GAP_SECONDS
            {
                return Err(Error::CheckError(format!(
                    "Future time {:?} is not supported",
                    ts
                )));
            }
            match manager.get_next_backup_after_ts(&utc_time).await? {
                Some(f) => Ok(RestoreSource::ExistFile((f, Some(utc_time)))),
                None => Ok(RestoreSource::InstantBackup(utc_time)),
            }
        }
    }
}

pub(crate) async fn handle_restore_keyspace(
    manager: Arc<NativeBrManager>,
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

    let target_keyspace = query_pairs.get("keyspace");
    if target_keyspace.is_none() {
        return Ok(make_response(
            StatusCode::BAD_REQUEST,
            "Keyspace name is none",
        ));
    }
    let mut source_keyspace = query_pairs.get("source_keyspace");
    let inplace_restore = source_keyspace.is_none();
    if inplace_restore {
        // treat source as target
        source_keyspace = target_keyspace;
    }

    let source_keyspace = source_keyspace.unwrap().to_string();
    let target_keyspace = target_keyspace.unwrap().to_string();
    if !manager.is_keyspace_allowed(&source_keyspace)
        || !manager.is_keyspace_allowed(&target_keyspace)
    {
        return Ok(make_response(
            StatusCode::FORBIDDEN,
            format!(
                "Keyspace source {} or target {} is not in whitelist",
                source_keyspace, target_keyspace
            ),
        ));
    }

    let keyspace_tag = format!("{}->{}", source_keyspace, target_keyspace);

    match *req.method() {
        Method::GET => {
            debug!(
                "{} request to GET restore_keyspace, restore_id {}",
                keyspace_tag, restore_id
            );
            handle_restore_status(&manager, restore_id, &target_keyspace)
        }
        Method::PUT => {
            let restore_type = match parse_restore_type(&query_pairs) {
                Ok(t) => t,
                Err(e) => {
                    return Ok(make_response(StatusCode::BAD_REQUEST, e.to_string()));
                }
            };
            let backup = get_backup_from_query(&manager, &query_pairs, restore_type).await;
            if backup.is_err() {
                return Ok(make_response(
                    StatusCode::BAD_REQUEST,
                    format!("Fail to get backup: {}", backup.unwrap_err()),
                ));
            }
            let backup = backup.unwrap();
            debug!(
                "{} request to PUT restore_keyspace, restore_id {}, backup {:?}",
                keyspace_tag, restore_id, backup
            );
            match manager.restore_keyspace(
                restore_id,
                source_keyspace,
                target_keyspace.clone(),
                backup.clone(),
                restore_type,
            ) {
                Ok(true) => {
                    info!(
                        "{} restore keyspace started, restore_id {}, backup {:?}",
                        keyspace_tag, restore_id, backup
                    );
                    let (progress, step_name) =
                        RestoreProgressReporter::step_to_progress(&RestoreStep::Pending);
                    let resp = RestoreProgressResponse {
                        status: RestoreState::Pending,
                        error: String::new(),
                        duration: 0,
                        id: restore_id,
                        keyspace: target_keyspace,
                        restore_type,
                        restore_bytes: 0,
                        progress: RestoreProgress {
                            step: step_name.to_string(),
                            progress,
                        },
                    };
                    Ok(make_json_response(StatusCode::CREATED, &resp))
                }
                Ok(false) => {
                    info!(
                        "{} restore keyspace ignored, restore_id {}, backup {:?}",
                        keyspace_tag, restore_id, backup
                    );
                    // Return current status when `restore_keyspace` request is ignored.
                    handle_restore_status(&manager, restore_id, &target_keyspace)
                }
                Err(Error::RestoreKeyspaceTaskConflict(conflict_restore_id)) => {
                    info!(
                        "{} restore keyspace conflict, restore_id {}, backup {:?}, conflict restore_id {}",
                        keyspace_tag, restore_id, backup, conflict_restore_id,
                    );
                    let resp = RestoreConflictResponse {
                        keyspace: target_keyspace,
                        id: restore_id,
                        conflict_restore_id,
                    };
                    Ok(make_json_response(StatusCode::CONFLICT, &resp))
                }
                Err(e) => {
                    error!(
                        "{} restore keyspace error, restore_id {}, backup {:?}, error {:?}",
                        keyspace_tag, restore_id, backup, e
                    );
                    handle_error(e)
                }
            }
        }
        Method::DELETE => {
            debug!(
                "{} request to DELETE restore_keyspace, restore_id {}",
                keyspace_tag, restore_id
            );
            match manager.delete_restore(restore_id, &target_keyspace) {
                Ok(Some(true)) => {
                    info!(
                        "{} restore keyspace task deleted, restore_id {}",
                        keyspace_tag, restore_id
                    );
                    Ok(make_response(StatusCode::OK, "Restore task deleted"))
                }
                Ok(Some(false)) => {
                    info!(
                        "{} delete restore keyspace task ignored, restore_id {}",
                        keyspace_tag, restore_id
                    );
                    Ok(make_response(
                        StatusCode::CONFLICT,
                        "Restore task is not in final state",
                    ))
                }
                Ok(None) => {
                    info!(
                        "{} delete restore keyspace task not found, restore_id {}",
                        keyspace_tag, restore_id
                    );
                    Ok(make_response(
                        StatusCode::NOT_FOUND,
                        format!("Restore task not found: {}", restore_id),
                    ))
                }
                Err(err) => {
                    error!(
                        "{} delete restore keyspace task error, restore_id {}, error {:?}",
                        keyspace_tag, restore_id, err
                    );
                    handle_error(err)
                }
            }
        }
        _ => Ok(make_response(StatusCode::BAD_REQUEST, "Invalid method")),
    }
}

fn handle_get_all_restore_task(manager: &Arc<NativeBrManager>) -> hyper::Result<Response<Body>> {
    let resp_vec: Vec<RestoreProgressResponse> = manager
        .get_all_restore_task()
        .into_iter()
        .map(|(id, task)| RestoreProgressResponse::from_restore_status(id, task))
        .collect();
    Ok(make_json_response(StatusCode::OK, &resp_vec))
}

fn handle_restore_status(
    manager: &Arc<NativeBrManager>,
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
    manager: Arc<NativeBrManager>,
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
                    &manager.config.read().unwrap().native_br.whitelist,
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

#[derive(Clone, Debug)]
enum RestoreSource {
    ExistFile(
        (IncrementalBackupFile, Option<DateTime<Utc>>), // truncate_ts
    ),
    InstantBackup(DateTime<Utc> /* truncate_ts */),
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
struct RestoreProgress {
    step: String,
    progress: i32,
}

#[derive(Default, Serialize, Deserialize, Debug)]
#[serde(default)]
struct RestoreProgressResponse {
    status: RestoreState,
    error: String,
    duration: i64, // in seconds
    id: u64,
    keyspace: String,
    restore_type: RestoreType,
    restore_bytes: u64,
    progress: RestoreProgress,
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
            restore_type: status.restore_type,
            restore_bytes: status.restore_bytes,
            progress: status.progress_reporter.get_progress(),
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

#[derive(Clone, Copy, Default, Serialize, Deserialize, Debug, PartialEq)]
enum RestoreType {
    #[default]
    Normal,
    Pitr,
}

#[derive(Clone)]
pub(crate) struct RestoreTask {
    state: RestoreState,
    keyspace_name: String,
    error: String,
    start: DateTime<Utc>,
    end: Option<DateTime<Utc>>,
    restore_type: RestoreType,
    restore_bytes: u64,
    progress_reporter: Arc<RestoreProgressReporter>,
}

impl RestoreTask {
    fn ttl_expired(&self, ttl: Duration) -> bool {
        self.state.is_final()
            && Utc::now()
                .signed_duration_since(self.end.unwrap())
                .num_seconds()
                >= ttl.as_secs() as i64
    }
}

struct RestoreProgressReporter {
    restore_id: u64,
    step: Mutex<RestoreStep>,
}

impl RestoreProgressReporter {
    fn new(restore_id: u64, step: RestoreStep) -> Self {
        Self {
            restore_id,
            step: Mutex::new(step),
        }
    }

    fn step_to_progress(step: &RestoreStep) -> (i32, &'static str) {
        match step {
            // The difference value between this and next step is the estimated time cost of current
            // step.
            // Note: the message will be seen by customers.
            RestoreStep::Pending => (0, "pending"),
            RestoreStep::Init => (5, "init"),
            RestoreStep::InstantBackup => (10, "prepare backup"),
            RestoreStep::RemoveTiFlashReplicas => (30, "remove tiflash replicas"),
            RestoreStep::LoadBackupMeta => (35, "load backup meta"),
            RestoreStep::ExtractBackupShards => (40, "extract backup shards"),
            RestoreStep::FlushShards => (50, "flush shards"),
            RestoreStep::TruncateTs => (52, "truncate ts"),
            RestoreStep::SplitRegions => (55, "split regions"),
            RestoreStep::AlignRegions => (60, "align regions"),
            RestoreStep::RestoreSnapshotsToServers => (65, "restore snapshots to servers"),
            RestoreStep::RetainSstFiles => (85, "retain files"),
            RestoreStep::Finalize => (90, "finalize"), // Wait for ClusterCR become normal.
        }
    }

    fn get_progress(&self) -> RestoreProgress {
        let step = self.step.lock().unwrap();
        let (progress, step_name) = Self::step_to_progress(&step);
        RestoreProgress {
            step: step_name.to_string(),
            progress,
        }
    }
}

impl ReportRestoreStepTrait for RestoreProgressReporter {
    fn report_step(&self, step: RestoreStep) {
        info!("restore {} report_step: {:?}", self.restore_id, step);
        let mut current_step = self.step.lock().unwrap();
        if *current_step < step {
            *current_step = step;
        }
    }
}

type TasksMap = HashMap<u64 /* restore_id */, RestoreTask>;

type KeyspacesMap = HashMap<String /* keyspace */, u64 /* restore_id */>;

pub(crate) struct BrContext {
    pub pd_client: Arc<dyn PdClient>,
    pub working_path: Option<String>,
    pub s3fs: Arc<S3Fs>,
    pub runtime: Arc<Runtime>,
    pub restore_tasks: RwLock<TasksMap>,
    pub keyspace_tasks: RwLock<KeyspacesMap>,
    pub backup_worker: BackupWorker,
}

impl BrContext {
    // Return false if the state is falling back.
    fn change_restore_state(
        &self,
        restore_id: u64,
        keyspace_name: &str,
        restore_type: RestoreType,
        new_state: RestoreState,
        err: Option<Error>,
        restore_bytes: u64,
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
                    restore_type,
                    restore_bytes,
                    progress_reporter: Arc::new(RestoreProgressReporter::new(
                        restore_id,
                        RestoreStep::Init,
                    )),
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
                task.restore_bytes = restore_bytes;
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

    fn get_progress_reporter(&self, restore_id: u64) -> Option<Arc<RestoreProgressReporter>> {
        let tasks = self.restore_tasks.read().unwrap();
        tasks
            .get(&restore_id)
            .map(|task| task.progress_reporter.clone())
    }

    fn restore_keyspace_core(
        &self,
        config: Config,
        keyspace_name: &str,
        target_keyspace_name: &str,
        restore_source: RestoreSource,
        restore_type: RestoreType,
        progress_reporter: Arc<RestoreProgressReporter>,
    ) -> Result<RestoredKeyspace> {
        // Instant backup must be performed before every restore.
        // Otherwise the data from previous backup to now will be lost, and can not be
        // restored by PiTR.
        progress_reporter.report_step(RestoreStep::InstantBackup);
        let lightweight = config.native_br.enable_lightweight_backup;
        let instant_backup = self
            .runtime
            .block_on(self.backup_worker.instant_backup(lightweight))?;

        let get_truncate_ts =
            |utc_time: Option<DateTime<Utc>>, restore_type: RestoreType| -> Option<u64> {
                match utc_time {
                    Some(t) => {
                        debug_assert!(restore_type == RestoreType::Pitr);
                        let tso = TimeStamp::compose(t.timestamp_millis() as u64, 0);
                        Some(tso.into_inner())
                    }
                    None => {
                        debug_assert!(restore_type == RestoreType::Normal);
                        None
                    }
                }
            };

        let (backup_file, truncate_ts) = match restore_source {
            RestoreSource::ExistFile((f, utc_time)) => (f, get_truncate_ts(utc_time, restore_type)),
            RestoreSource::InstantBackup(utc_time) => (
                instant_backup.deref().clone(),
                get_truncate_ts(Some(utc_time), restore_type),
            ),
        };

        Ok(restore_keyspace_with_cfg(
            config.to_restore_config(),
            keyspace_name,
            target_keyspace_name,
            backup_file.name(),
            self.working_path.as_deref(),
            self.s3fs.clone(),
            self.pd_client.clone(),
            &self.runtime,
            truncate_ts,
            progress_reporter,
        )?)
    }

    fn restore_keyspace(
        &self,
        config: Config,
        restore_id: u64,
        keyspace_name: String,
        target_keyspace_name: String,
        restore_source: RestoreSource,
        restore_type: RestoreType,
    ) -> Result<()> {
        let ob_start_time = Instant::now();

        self.change_restore_state(
            restore_id,
            &target_keyspace_name,
            restore_type,
            RestoreState::Running,
            None,
            0,
        )?;
        let progress_reporter = self.get_progress_reporter(restore_id).unwrap();

        let res = match self.restore_keyspace_core(
            config,
            &keyspace_name,
            &target_keyspace_name,
            restore_source,
            restore_type,
            progress_reporter.clone(),
        ) {
            Ok(ret) => {
                NATIVE_BR_COUNTER_VEC
                    .with_label_values(&["restore_keyspace_succeed"])
                    .inc();
                NATIVE_BR_HISTOGRAM_VEC
                    .with_label_values(&["restore_keyspace"])
                    .observe(ob_start_time.saturating_elapsed_secs());
                progress_reporter.report_step(RestoreStep::Finalize);
                self.change_restore_state(
                    restore_id,
                    &target_keyspace_name,
                    restore_type,
                    RestoreState::Succeed,
                    None,
                    ret.restore_bytes,
                )
            }
            Err(err) => {
                error!(
                    "{}->{} restore_keyspace error, restore_id {}, error {:?}",
                    keyspace_name, target_keyspace_name, restore_id, err
                );
                NATIVE_BR_COUNTER_VEC
                    .with_label_values(&["restore_keyspace_fail"])
                    .inc();
                self.change_restore_state(
                    restore_id,
                    &target_keyspace_name,
                    restore_type,
                    RestoreState::Error,
                    Some(err),
                    0,
                )
            }
        };
        if let Err(e) = res {
            NATIVE_BR_COUNTER_VEC
                .with_label_values(&["change_restore_state_fail"])
                .inc();
            return Err(e);
        }
        Ok(())
    }
}

impl Drop for BrContext {
    fn drop(&mut self) {
        self.backup_worker.stop();
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

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(default)]
#[serde(rename_all = "kebab-case")]
pub struct NativeBrConfig {
    whitelist: WhiteList,
    // The time-to-live when restore task has been in final state.
    restore_task_ttl: ReadableDuration,
    // Enable lightweight instant backup during restore.
    enable_lightweight_backup: bool,
}

impl Default for NativeBrConfig {
    fn default() -> Self {
        Self {
            whitelist: WhiteList::default(),
            restore_task_ttl: ReadableDuration::minutes(60),
            enable_lightweight_backup: false,
        }
    }
}

pub(crate) struct NativeBrManager {
    pub context: Arc<BrContext>,
    pub config: RwLock<Config>,
}

impl NativeBrManager {
    pub(crate) fn new(
        runtime: Arc<Runtime>,
        pd_client: Arc<dyn PdClient>,
        s3fs: Arc<S3Fs>,
        working_path: Option<String>,
        config: Config,
    ) -> Self {
        let backup_config = config.to_backup_config();
        let backup_worker = BackupWorker::new(
            backup_config,
            pd_client.clone(),
            INSTANT_BACKUP_INTERVAL,
            MAX_RESTORE_CONCURRENCY,
        );
        Self {
            context: Arc::new(BrContext {
                pd_client,
                s3fs,
                working_path,
                runtime,
                restore_tasks: Default::default(),
                keyspace_tasks: Default::default(),
                backup_worker,
            }),
            config: RwLock::new(config),
        }
    }

    pub(crate) fn update_native_br_config(&self, config: NativeBrConfig) {
        let mut ori_config = self.config.write().unwrap();
        if ori_config.native_br != config {
            info!(
                "Update native br config from {:?} to {:?}",
                ori_config.native_br, config
            );
            ori_config.native_br = config;
        }
    }

    async fn list_backups(
        &self,
        start_backup_time: &DateTime<Utc>,
        max_count: usize,
    ) -> Result<(Vec<IncrementalBackupFile>, bool)> {
        let (backups, has_more) = backup::get_all_incremental_backups(
            &self.context.s3fs,
            &start_backup_time.date_naive(),
            Some(&start_backup_time.time()),
            max_count,
        )
        .await?;
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
        target_keyspace_name: String,
        restore_source: RestoreSource,
        restore_type: RestoreType,
    ) -> Result<bool> {
        if self.get_not_final_task_count() >= MAX_RESTORE_CONCURRENCY {
            return Err(Error::ReachConcurrencyLimit(MAX_RESTORE_CONCURRENCY));
        }

        if self.context.change_restore_state(
            restore_id,
            &target_keyspace_name,
            restore_type,
            RestoreState::Init,
            None,
            0,
        )? {
            let context = self.context.clone();
            let config = self.config.read().unwrap().clone();
            thread::spawn(move || {
                context.restore_keyspace(
                    config,
                    restore_id,
                    keyspace_name,
                    target_keyspace_name,
                    restore_source,
                    restore_type,
                )
            });
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

    /// Return number of tasks which are in not final state.
    fn get_not_final_task_count(&self) -> usize {
        self.context
            .restore_tasks
            .rl()
            .values()
            .filter(|task| !task.state.is_final())
            .count()
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

    pub fn cleanup_expired_restores(&self) {
        let ttl = self.config.rl().native_br.restore_task_ttl.0;
        let restores = {
            let tasks = self.context.restore_tasks.rl();
            tasks
                .iter()
                .filter_map(|(&restore_id, task)| {
                    task.ttl_expired(ttl)
                        .then_some((restore_id, task.keyspace_name.clone()))
                })
                .collect::<Vec<_>>()
        };
        for (restore_id, keyspace_name) in restores {
            if let Ok(Some(true)) = self.delete_restore(restore_id, &keyspace_name) {
                info!("{}({}) expired and removed", keyspace_name, restore_id);
            }
        }
    }

    fn is_keyspace_allowed(&self, keyspace: &String) -> bool {
        self.config
            .read()
            .unwrap()
            .native_br
            .whitelist
            .is_allowed(keyspace)
    }

    async fn get_next_backup_after_ts(
        &self,
        ts: &DateTime<Utc>,
    ) -> Result<Option<IncrementalBackupFile>> {
        let (backup_files, _) = self.list_backups(ts, 1).await?;
        Ok(backup_files.first().cloned())
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

#[cfg(test)]
mod tests {
    use std::str::FromStr;

    use kvengine::dfs::test_util::new_test_s3fs;
    use pd_client::PdClient;
    use security::GetSecurityManager;
    use tikv_util::config::ReadableDuration;

    use super::*;
    use crate::native_br::RestoreState::{Init, Running, Succeed};

    #[test]
    fn test_restore_task_state() {
        let br_manager = new_test_br_manager("2s");

        // TODO: test for illegal state transition.
        let states_cases = vec![
            vec![Init],
            vec![Init, Running],
            vec![Init, Running, Succeed],
            vec![Init, Running, RestoreState::Error],
            vec![Init, Running, RestoreState::Error, Init],
        ];

        for (restore_id, states) in states_cases.into_iter().enumerate() {
            for state in states {
                assert!(
                    br_manager
                        .context
                        .change_restore_state(
                            restore_id as u64,
                            &format!("ks{}", restore_id),
                            RestoreType::Normal,
                            state,
                            None,
                            0
                        )
                        .unwrap()
                );
            }
        }

        assert_eq!(br_manager.get_not_final_task_count(), 3);
        assert_eq!(br_manager.get_all_restore_task().len(), 5);

        thread::sleep(Duration::from_secs(2));
        br_manager.cleanup_expired_restores();
        assert_eq!(br_manager.get_not_final_task_count(), 3);
        assert_eq!(br_manager.get_all_restore_task().len(), 3);

        assert!(
            br_manager
                .context
                .change_restore_state(1, "ks1", RestoreType::Normal, Succeed, None, 0)
                .unwrap()
        );
        br_manager.cleanup_expired_restores();
        assert_eq!(br_manager.get_not_final_task_count(), 2);
        assert_eq!(br_manager.get_all_restore_task().len(), 3);
    }

    fn new_test_br_manager(restore_task_ttl: &str) -> NativeBrManager {
        let thread_pool = Arc::new(
            tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .worker_threads(1)
                .thread_name("br_manager_test")
                .build()
                .unwrap(),
        );

        let file_data = "abcdefgh".to_string().into_bytes();
        let s3fs = Arc::new(new_test_s3fs(&file_data));

        NativeBrManager::new(
            thread_pool,
            Arc::new(MockPdClient {}),
            s3fs,
            None,
            Config {
                native_br: NativeBrConfig {
                    restore_task_ttl: ReadableDuration::from_str(restore_task_ttl).unwrap(),
                    ..Default::default()
                },
                ..Default::default()
            },
        )
    }

    struct MockPdClient {}

    impl PdClient for MockPdClient {}

    impl GetSecurityManager for MockPdClient {}
}
