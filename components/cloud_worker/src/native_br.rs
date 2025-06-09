// Copyright 2023 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    borrow::Cow,
    collections::HashMap,
    fmt, fs,
    io::Write,
    ops::Deref,
    path::PathBuf,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Mutex, RwLock,
    },
    thread,
    time::Duration,
};

use chrono::{DateTime, NaiveDateTime, TimeZone, Utc};
use http::{Method, StatusCode};
use hyper::{Body, Response};
use kvengine::dfs::S3Fs;
use native_br::{
    backup::IncrementalBackupFile,
    backup_worker,
    backup_worker::BackupWorker,
    common::get_all_incremental_backups,
    restore,
    restore_keyspace::{
        restore_keyspace_with_cfg, ReportRestoreStepTrait, RestoreStep, RestoredKeyspace,
    },
};
use pd_client::{pd_control::PdControl, PdClient};
use serde::Deserialize;
use tikv::storage::mvcc::TimeStamp;
use tikv_util::{
    config::ReadableDuration, debug, error, errors::Context as _, info, time::Instant, warn,
    HandyRwLock,
};
use tokio::runtime::Runtime;

use crate::{
    common::{get_param, make_json_response, make_response},
    error::{Error, Result},
    metrics::{NATIVE_BR_COUNTER_VEC, NATIVE_BR_HISTOGRAM_VEC},
    Config,
};

const MIN_PITR_INTERVAL_GAP_SECONDS: i64 = 1; // 1s
pub(crate) const MAX_RESTORE_CONCURRENCY: usize = 128;
const MAX_BACKUP_COUNT_PER_PAGE: usize = 1000; // Same with dfs list.
const JSON_TIME_FORMAT: &str = "%Y-%m-%d %H:%M:%S%.3f"; // e.g. 2006-01-02 15:04:05.000
const BACKUP_NAME_FORMAT: &str = "%Y%m%d%H%M%S";
pub(crate) const BACKUPS_API_PATH: &str = "/api/v1/backups";
pub(crate) const RESTORE_KEYSPACE_API_PATH: &str = "/api/v1/restore_keyspace/";

const RESTORE_TASK_WORKING_PATH_PREFIX: &str = "r";

/// Backup and restore keyspace API:
///
/// 1. backup:
///   * GET    /api/v1/backups?cluster_id=%d&last_backup_time=<JSON_TIME_FORMAT>
///
/// 2. restore keyspace: Specify backup_id&backup_name for normal restore or
///    point_in_time for pitr
///   * PUT    /api/v1/restore_keyspace/<restore_id>?cluster_id=%d&keyspace=%s&
///     backup_id=%d&backup_name=%s[&source_keyspace=%s]&
///     point_in_time=<JSON_TIME_FORMAT>
///   * GET    /api/v1/restore_keyspace/<restore_id>?cluster_id=%d&keyspace=%s
///   * DELETE /api/v1/restore_keyspace/<restore_id>?cluster_id=%d&keyspace=%s
///   * GET    /api/v1/restore_keyspace/?cluster_id=%d

pub(crate) async fn handle_backup(
    manager: Arc<NativeBrManager>,
    req: hyper::Request<hyper::Body>,
) -> hyper::Result<hyper::Response<hyper::Body>> {
    let query = req.uri().query().unwrap_or("");
    let query_pairs: HashMap<_, _> = url::form_urlencoded::parse(query.as_bytes()).collect();
    match get_param::<u64>(&query_pairs, "cluster_id") {
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
            let backup_id = match get_param::<u64>(query_pairs, "backup_id") {
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

    match get_param::<u64>(&query_pairs, "cluster_id") {
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

#[derive(Clone, Debug)]
enum RestoreSource {
    ExistFile(
        (IncrementalBackupFile, Option<DateTime<Utc>>), // truncate_ts
    ),
    InstantBackup(DateTime<Utc> /* truncate_ts */),
}

#[derive(Default, Serialize, Deserialize, Debug)]
#[serde(default)]
pub struct BackupItem {
    pub id: u64,
    pub name: String,
    pub time: String,
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
pub struct ListBackupResponse {
    pub items: Vec<BackupItem>,
    pub has_more: bool,
}

#[derive(Default, Serialize, Deserialize, Debug)]
#[serde(default)]
pub struct RestoreProgress {
    pub step: String,
    pub progress: i32,
}

#[derive(Default, Serialize, Deserialize, Debug)]
#[serde(default)]
pub struct RestoreProgressResponse {
    pub status: RestoreState,
    pub error: String,
    pub duration: i64, // in seconds
    pub id: u64,
    pub keyspace: String,
    pub restore_type: RestoreType,
    pub restore_bytes: u64,
    pub progress: RestoreProgress,
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
pub enum RestoreState {
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

#[derive(Clone, Copy, Default, Serialize, Deserialize, Debug, PartialEq)]
pub enum RestoreType {
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

    fn get_meta(&self) -> RestoreTaskMeta {
        RestoreTaskMeta {
            keyspace_name: self.keyspace_name.clone(),
            restore_type: self.restore_type,
            start: self.start.timestamp(),
        }
    }

    fn from_meta(meta: RestoreTaskMeta) -> Self {
        let start = meta.start();
        Self {
            state: RestoreState::Error, // Consider as being interrupted by node restart.
            keyspace_name: meta.keyspace_name,
            error: "interrupted".to_string(),
            start,
            end: Some(Utc::now()),
            restore_type: meta.restore_type,
            restore_bytes: 0,
            progress_reporter: Arc::new(RestoreProgressReporter::new(0, RestoreStep::Init)),
        }
    }
}

#[derive(Clone, Default, Serialize, Deserialize)]
pub(crate) struct RestoreTaskMeta {
    keyspace_name: String,
    restore_type: RestoreType,
    start: i64, // DateTime<Utc>::timestamp().
}

impl fmt::Debug for RestoreTaskMeta {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RestoreTaskMeta")
            .field("keyspace_name", &self.keyspace_name)
            .field("restore_type", &self.restore_type)
            .field("start", &self.start())
            .finish()
    }
}

impl RestoreTaskMeta {
    fn start(&self) -> DateTime<Utc> {
        Utc.timestamp_opt(self.start, 0).unwrap()
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
            RestoreStep::ResolveLocks => (48, "resolve locks"),
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
    pub data_dir: PathBuf,
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
        let new_task = |tasks: &mut TasksMap| -> Result<RestoreTaskMeta> {
            let mut keyspaces = self.keyspace_tasks.write().unwrap();
            if let Some(&restore_id) = keyspaces.get(keyspace_name) {
                return Err(Error::RestoreKeyspaceTaskConflict(restore_id));
            }
            keyspaces.insert(keyspace_name.to_owned(), restore_id);
            let task = RestoreTask {
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
            };
            let task_meta = task.get_meta();
            tasks.insert(restore_id, task);
            Ok(task_meta)
        };

        let mut tasks = self.restore_tasks.write().unwrap();
        match tasks.get_mut(&restore_id) {
            None => {
                debug_assert_eq!(new_state, RestoreState::Init);
                let meta = new_task(&mut tasks)?;
                self.persist_task_meta(restore_id, &meta)?;
            }
            Some(task) if task.state.is_retry(&new_state) => {
                check_task(task, keyspace_name)?;
                let meta = new_task(&mut tasks)?;
                self.persist_task_meta(restore_id, &meta)?;
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
        working_path: PathBuf,
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
        let instant_backup = self.runtime.block_on(
            self.backup_worker
                .instant_backup_with_retry(config.native_br.instant_backup_timeout.0),
        )?;

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
            Some(working_path),
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

        let working_path = self.working_path(restore_id);
        let res = match self.restore_keyspace_core(
            config,
            working_path,
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
                let res = self.change_restore_state(
                    restore_id,
                    &target_keyspace_name,
                    restore_type,
                    RestoreState::Succeed,
                    None,
                    ret.restore_bytes,
                );
                self.remove_working_path(restore_id);
                res
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

    fn working_path(&self, restore_id: u64) -> PathBuf {
        self.data_dir
            .join(format!("{RESTORE_TASK_WORKING_PATH_PREFIX}{restore_id}"))
    }

    fn remove_working_path(&self, restore_id: u64) {
        let working_path = self.working_path(restore_id);
        if let Err(e) = fs::remove_dir_all(&working_path) {
            warn!("fail to remove working path: {:?}", e; "path" => working_path.display());
        }
    }

    fn persist_task_meta(&self, restore_id: u64, meta: &RestoreTaskMeta) -> Result<()> {
        static TMP_ID: AtomicU64 = AtomicU64::new(0);

        let working_path = self.working_path(restore_id);
        fs::create_dir_all(&working_path).ctx("create_working_path")?;
        let tmp_path = working_path.join(format!(
            "meta.json.{}",
            TMP_ID.fetch_add(1, Ordering::Relaxed)
        ));
        let mut tmp = fs::File::create(&tmp_path).ctx("create_tmp")?;
        tmp.write_all(&serde_json::to_vec(&meta)?)
            .ctx("write_meta")?;
        tmp.sync_data().ctx("sync")?;
        drop(tmp);

        let meta_file = working_path.join("meta.json");
        fs::rename(tmp_path, meta_file).ctx("rename")?;
        Ok(())
    }

    fn read_task_meta(&self, restore_id: u64) -> Result<RestoreTaskMeta> {
        let working_path = self.working_path(restore_id);
        let meta_file = working_path.join("meta.json");
        if !meta_file.exists() {
            return Err(Error::CheckError(format!(
                "Meta file not found: {}",
                meta_file.display()
            )));
        }
        let meta = fs::read(&meta_file).ctx("read_meta")?;
        let meta = serde_json::from_slice::<RestoreTaskMeta>(&meta)?;
        Ok(meta)
    }

    fn init(&mut self) -> Result<()> {
        let entries = fs::read_dir(&self.data_dir).ctx("read_dir")?;
        for entry in entries {
            let entry = entry.ctx("entry")?;
            let path = entry.path();
            if path.is_dir() && path.file_name().is_some() {
                let name = path.file_name().unwrap().to_string_lossy();
                let name = name.as_ref();
                if let Some(restore_id) = name
                    .strip_prefix(RESTORE_TASK_WORKING_PATH_PREFIX)
                    .and_then(|s| s.parse::<u64>().ok())
                {
                    match self.read_task_meta(restore_id) {
                        Ok(meta) => {
                            info!("restore task: {:?}", meta);
                            let task = RestoreTask::from_meta(meta);
                            self.restore_tasks.wl().insert(restore_id, task);
                        }
                        Err(err) => {
                            warn!("fail to read task meta: {:?}", err; "restore_id" => restore_id);
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

impl Drop for BrContext {
    fn drop(&mut self) {
        self.backup_worker.stop();
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(default)]
#[serde(rename_all = "kebab-case")]
pub struct NativeBrConfig {
    /// The time-to-live when restore task has been in final state.
    pub restore_task_ttl: ReadableDuration,

    /// The timeout for waiting the flush of mem-tables.
    pub restore_timeout_wait_flush: ReadableDuration,
    /// The timeout for the requests of restoring snapshots to TiKV servers.
    pub restore_timeout_restore_snapshot: ReadableDuration,
    /// The timeout for fetching the latest wal chunk from store.
    pub restore_timeout_fetch_wal: ReadableDuration,
    pub restore_timeout_pd_control: ReadableDuration,
    /// The maximum number of retries for the process from split regions to
    /// restore snapshots.
    pub restore_max_retry: usize,
    pub restore_coarse_split_regions_factor: usize,

    /// The timeout for instant backup.
    pub instant_backup_timeout: ReadableDuration,
    /// Interval for periodic backup. `0s` to disable periodic backup.
    pub backup_interval: ReadableDuration,
    #[cfg(feature = "testexport")]
    pub backup_skip_keyspace_meta: bool,

    /// Whether to tolerate unavailability of no more than one store when
    /// backup.
    pub backup_tolerate_err: bool,
    /// Whether to tolerate unavailability of no more than one store when
    /// restore.
    pub restore_tolerate_err: bool,
}

impl Default for NativeBrConfig {
    fn default() -> Self {
        Self {
            restore_task_ttl: ReadableDuration::minutes(60),
            restore_timeout_wait_flush: restore::DEFAULT_TIMEOUT_WAIT_FLUSH,
            restore_timeout_restore_snapshot: restore::DEFAULT_TIMEOUT_RESTORE_SNAPSHOT,
            restore_timeout_fetch_wal: restore::DEFAULT_TIMEOUT_FETCH_WAL,
            restore_timeout_pd_control: restore::DEFAULT_TIMEOUT_PD_CONTROL,
            restore_max_retry: restore::DEFAULT_RESTORE_MAX_RETRY,
            restore_coarse_split_regions_factor: 64,
            instant_backup_timeout: backup_worker::DEFAULT_TIMEOUT_INSTANT_BACKUP,
            backup_interval: ReadableDuration::ZERO,
            #[cfg(feature = "testexport")]
            backup_skip_keyspace_meta: false,
            backup_tolerate_err: false,
            restore_tolerate_err: false,
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
        data_dir: PathBuf,
        config: Config,
    ) -> Self {
        let backup_config = config.to_backup_config();
        let backup_worker = BackupWorker::new(
            backup_config,
            pd_client.clone(),
            config.native_br.backup_interval.0,
        );
        let mut context = BrContext {
            pd_client,
            s3fs,
            data_dir,
            runtime,
            restore_tasks: Default::default(),
            keyspace_tasks: Default::default(),
            backup_worker,
        };
        if let Err(err) = context.init() {
            warn!("BR context init failed: {:?}", err);
        }
        Self {
            context: Arc::new(context),
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
        let (backups, has_more) = get_all_incremental_backups(
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
                self.context.remove_working_path(restore_id);
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

    async fn get_next_backup_after_ts(
        &self,
        ts: &DateTime<Utc>,
    ) -> Result<Option<IncrementalBackupFile>> {
        let (backup_files, _) = self.list_backups(ts, 1).await?;
        Ok(backup_files.first().cloned())
    }

    #[allow(dead_code)]
    pub fn get_pd_ctl(&self) -> Result<PdControl> {
        let cfg = self.config.rl().pd.clone();
        let sec_mgr = self.context.pd_client.get_security_mgr();
        let pd_ctl = PdControl::new(cfg, sec_mgr)?;
        Ok(pd_ctl)
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

pub mod v1x {
    use std::{collections::HashMap, fmt::Display, marker::PhantomData, str::FromStr, sync::Arc};

    use futures::{stream::FuturesUnordered, TryStreamExt};
    use http::StatusCode;
    use hyper::{Body, Request, Response};
    use kvengine::dfs::S3Fs;
    use native_br::backup::IncrementalBackupFile;
    use pd_client::pd_control::KeyspaceMeta;
    use protobuf::Message;
    use rfenginepb::ClusterBackupMeta;
    use serde::{
        de::{value::MapDeserializer, IntoDeserializer},
        Deserialize, Deserializer,
    };
    use tikv_util::{box_try, warn};

    use crate::{
        common::{make_json_response, make_response},
        error::Error,
        metrics::{NATIVE_BR_COUNTER_VEC, NATIVE_BR_HISTOGRAM_VEC},
        native_br::{BackupItem, NativeBrManager},
    };

    pub(crate) const V1X_BACKUPS_API_PATH: &str = "/api/v1/x/backups";
    pub type Result<T> = std::result::Result<T, HttpError>;

    #[derive(Debug)]
    pub struct HttpError(Response<Body>);

    impl<T: Into<Error>> From<T> for HttpError {
        fn from(err: T) -> Self {
            let crate_err: Error = err.into();
            match crate_err {
                Error::CheckError(msg) => Self::error_response(StatusCode::BAD_REQUEST, msg),
                otherwise => {
                    Self::error_response(StatusCode::INTERNAL_SERVER_ERROR, otherwise.to_string())
                }
            }
        }
    }

    impl serde::de::Error for Error {
        fn custom<T>(msg: T) -> Self
        where
            T: Display,
        {
            Error::CheckError(msg.to_string())
        }
    }

    impl std::error::Error for HttpError {}

    impl Display for HttpError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(
                f,
                "<ERROR HTTP/{}> (body = {:?})",
                self.0.status(),
                self.0.body()
            )
        }
    }

    impl HttpError {
        fn error_response(status_code: StatusCode, body: impl Into<Body>) -> Self {
            assert!(!status_code.is_success(), "{}", status_code);
            Self(make_response(status_code, body))
        }
    }

    /// Deserialize a type implementing `FromStr`.
    /// Thanks https://users.rust-lang.org/t/serde-fromstr-on-a-field/99457/7
    ///
    /// NOTE: This isn't effective enough(requires an extra copy), maybe
    /// replace it with `serde_with`.
    fn de_from_str<'de, D, T: FromStr<Err = E>, E: Display>(
        deserializer: D,
    ) -> std::result::Result<T, D::Error>
    where
        D: Deserializer<'de>,
    {
        let buf = String::deserialize(deserializer)?;
        T::from_str(&buf)
            .map_err(|err| serde::de::Error::custom(format!("failed to parse a field: {}", err)))
    }

    struct FieldHinttedStrDe<'a, E> {
        content: &'a str,
        hint: &'a str,
        phantom: PhantomData<E>,
    }

    impl<'a, E> FieldHinttedStrDe<'a, E> {
        fn new(content: &'a str, hint: &'a str) -> Self {
            Self {
                content,
                hint,
                phantom: PhantomData,
            }
        }
    }

    impl<'de, E: serde::de::Error> IntoDeserializer<'de, E> for FieldHinttedStrDe<'_, E> {
        type Deserializer = Self;

        fn into_deserializer(self) -> Self::Deserializer {
            self
        }
    }

    impl<'de, E: serde::de::Error> Deserializer<'de> for FieldHinttedStrDe<'_, E> {
        type Error = E;

        fn deserialize_any<V>(self, visitor: V) -> std::result::Result<V::Value, Self::Error>
        where
            V: serde::de::Visitor<'de>,
        {
            visitor.visit_str(self.content).map_err(|err: Self::Error| {
                <E as serde::de::Error>::custom(format!(
                    "during handing key {}: {}",
                    self.hint, err
                ))
            })
        }

        serde::forward_to_deserialize_any! {
            bool i8 i16 i32 i64 u8 u16 u32 u64 f32 f64 char str string bytes byte_buf
            option unit unit_struct newtype_struct seq tuple tuple_struct map struct
            enum identifier ignored_any
        }
    }

    fn v1_compat_id(key: &str) -> Option<u64> {
        IncrementalBackupFile::try_from_full_path(key).map(|file| file.id())
    }

    #[derive(Deserialize, Serialize)]
    pub struct ListBackupRequest {
        #[serde(default)]
        from_prefix: String,
        #[serde(deserialize_with = "de_from_str", default)]
        load_details: bool,
        #[serde(deserialize_with = "de_from_str", default = "default_max_count")]
        max_count: u64,
    }

    fn default_max_count() -> u64 {
        1
    }

    #[derive(Serialize, Deserialize, Debug)]
    pub struct ListBackupResponse {
        pub data: Vec<Backup>,
        pub cluster_id: u64,
        pub has_more: bool,
    }

    #[derive(Debug, Serialize, Deserialize)]
    pub struct Backup {
        pub name: String,
        pub last_modify_time: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub details: Option<BackupDetails>,

        pub v1_compat_id: Option<u64>,
    }

    impl Backup {
        pub fn try_to_backup_item(&self) -> Option<BackupItem> {
            IncrementalBackupFile::try_from_full_path(&format!("/backup/{}", self.name))
                .map(From::from)
        }
    }

    #[derive(Debug, Default, Serialize, Deserialize)]
    pub struct KeyspaceBackupInfo {
        pub keyspace_id: u32,
        // Skip this field when failed to fetch it.
        // So the caller knows there are something wrong...
        #[serde(skip_serializing_if = "String::is_empty")]
        pub keyspace_name: String,
        pub keyspace_state: String,
        pub size: u64,
    }

    #[derive(Debug, Serialize, Deserialize)]
    pub struct BackupDetails {
        pub backup_ts: u64,
        pub safe_ts: u64,
        pub keyspaces: Vec<KeyspaceBackupInfo>,
    }
    pub(crate) async fn serve_backup(
        br_ctx: Arc<NativeBrManager>,
        req: Request<Body>,
    ) -> hyper::Result<Response<Body>> {
        let ctx = RequestContext::new(req, br_ctx);

        Ok(match handle_backup(ctx).await {
            Ok(resp) => resp,
            Err(err) => {
                warn!("handle backup error: {:?}", err);
                err.0
            }
        })
    }

    async fn handle_backup(ctx: RequestContext) -> Result<Response<Body>> {
        let res = match ctx.raw_req.method() {
            &hyper::Method::GET => handle_list_backup(ctx)
                .await
                .map(|v| make_json_response(StatusCode::OK, &v)),
            _ => Err(HttpError::error_response(
                StatusCode::METHOD_NOT_ALLOWED,
                "",
            )),
        };
        if res.is_err() {
            NATIVE_BR_COUNTER_VEC
                .with_label_values(&["list_backup_x_fail"])
                .inc();
        }
        res
    }

    struct RequestContext {
        br: Arc<NativeBrManager>,
        raw_req: Request<Body>,
        request_params: HashMap<String, String>,
    }

    impl RequestContext {
        fn new(req: Request<Body>, br: Arc<NativeBrManager>) -> Self {
            Self {
                br,
                raw_req: req,
                request_params: Default::default(),
            }
        }

        fn query_params<'de, T: Deserialize<'de>>(&self) -> Result<T> {
            let query = self.raw_req.uri().query().unwrap_or("");
            let query_pairs: HashMap<_, _> =
                url::form_urlencoded::parse(query.as_bytes()).collect();

            let it = query_pairs
                .iter()
                .map(|(k, v)| (k.as_ref(), FieldHinttedStrDe::new(v.as_ref(), k)))
                .chain(
                    self.request_params
                        .iter()
                        .map(|(k, v)| (k.as_str(), FieldHinttedStrDe::new(v.as_str(), k))),
                );
            let de = MapDeserializer::<_, Error>::new(it);

            Ok(T::deserialize(de)?)
        }
    }

    async fn handle_list_backup(ctx: RequestContext) -> Result<ListBackupResponse> {
        let ob_start_time = tikv_util::time::Instant::now();
        let cluster_id = ctx.br.get_cluster_id()?;

        let params: ListBackupRequest = ctx.query_params()?;
        let backup_prefix = "backup/";
        let (files, more, _) = ctx
            .br
            .context
            .s3fs
            .list(
                &params.from_prefix,
                Some(backup_prefix),
                Some(params.max_count as _),
            )
            .await?;
        let full_prefix = format!("{}/{}", ctx.br.context.s3fs.get_prefix(), backup_prefix);

        let mut backups = Vec::with_capacity(files.len());
        let max_conc = 128;

        let br_ctx = ctx.br.context.as_ref();
        let full_prefix = &full_prefix;
        let load_details = params.load_details;
        let keyspaces = if load_details {
            load_keyspaces(&ctx.br).await?
        } else {
            Default::default()
        };
        let keyspaces = &keyspaces;
        {
            let mut st = FuturesUnordered::new();
            for v in files {
                st.push(async move {
                    Result::Ok(Backup {
                        name: v.key.strip_prefix(full_prefix).unwrap().to_string(),
                        last_modify_time: v.last_modified,
                        details: if load_details {
                            Some(
                                fetch_backup_details(br_ctx.s3fs.as_ref(), &v.key, keyspaces)
                                    .await?,
                            )
                        } else {
                            None
                        },
                        v1_compat_id: v1_compat_id(&v.key),
                    })
                });

                if st.len() > max_conc {
                    backups.push(
                        st.try_next()
                            .await?
                            .expect("should be something in the stream"),
                    )
                }
            }
            while let Some(v) = st.try_next().await? {
                backups.push(v);
            }
        }

        backups.sort_by(|bk1, bk2| bk1.name.cmp(&bk2.name));

        NATIVE_BR_HISTOGRAM_VEC
            .with_label_values(&["list_backup_x"])
            .observe(ob_start_time.saturating_elapsed_secs());

        Ok(ListBackupResponse {
            data: backups,
            cluster_id,
            has_more: more,
        })
    }

    async fn fetch_backup_details(
        s3fs: &S3Fs,
        key: &str,
        keyspaces: &HashMap<u32, KeyspaceMeta>,
    ) -> Result<BackupDetails> {
        let backup_meta_bytes = s3fs
            .get_object(key.to_owned(), key.to_owned(), Default::default())
            .await?;
        let mut backup_meta = ClusterBackupMeta::default();
        box_try!(backup_meta.merge_from_bytes(&backup_meta_bytes));

        let mut keyspace_sizes = HashMap::<u32, KeyspaceBackupInfo>::new();
        for store in backup_meta.get_stores().iter() {
            for (&keyspace_id, backup_size) in store.get_keyspace_size().iter() {
                let keyspace_meta = keyspaces.get(&keyspace_id);
                let sz = keyspace_sizes
                    .entry(keyspace_id)
                    .or_insert_with(|| KeyspaceBackupInfo {
                        keyspace_id,
                        keyspace_name: keyspace_meta
                            .map(|meta| meta.name.to_owned())
                            .unwrap_or_default(),
                        keyspace_state: keyspace_meta
                            .map(|meta| meta.state.to_owned())
                            .unwrap_or_else(|| "NOT_FOUND".to_owned()),
                        size: 0,
                    });
                sz.size += backup_size.size;
            }
        }

        Ok(BackupDetails {
            backup_ts: backup_meta.backup_ts,
            safe_ts: backup_meta.safe_ts,
            keyspaces: keyspace_sizes.into_values().collect(),
        })
    }

    #[cfg(not(feature = "testexport"))]
    async fn load_keyspaces(br_ctx: &NativeBrManager) -> Result<HashMap<u32, KeyspaceMeta>> {
        let ctl = br_ctx.get_pd_ctl()?;
        let mut keyspaces = HashMap::default();
        let mut next_page_token = None;
        loop {
            let kss = ctl.list_keyspaces(None, next_page_token).await?;
            keyspaces.extend(kss.keyspaces.into_iter().map(|meta| (meta.id, meta)));
            if kss.next_page_token.is_empty() {
                return Ok(keyspaces);
            }
            next_page_token = Some(kss.next_page_token);
        }
    }

    // for now the test cluster doesn't support PD's http API...
    // use a mocked keyspace meta to make the test case happy...
    #[cfg(feature = "testexport")]
    async fn load_keyspaces(_br_ctx: &NativeBrManager) -> Result<HashMap<u32, KeyspaceMeta>> {
        let meta = |id, name: &str| {
            let mut meta = KeyspaceMeta::default();
            meta.id = id;
            meta.name = name.to_owned();
            meta
        };

        Ok(vec![meta(0, "DEFAULT")]
            .into_iter()
            .chain((1..100).map(|id| meta(id, &format!("ks{id}"))))
            .map(|meta| (meta.id, meta))
            .collect::<HashMap<_, _>>())
    }
}

#[cfg(any(test, feature = "testexport"))]
pub mod test_utils {
    use std::sync::Arc;

    use chrono::{DateTime, Utc};
    use security::{HttpResult, RestfulClient, SecurityManager};
    use tikv_util::box_try;

    use crate::native_br::{
        BackupItem, ListBackupResponse, RestoreProgressResponse, JSON_TIME_FORMAT,
    };

    #[derive(Default, Serialize, Deserialize, Debug)]
    #[serde(default)]
    struct DummyRequest {}

    pub struct NativeBrSvcClient {
        cluster_id: u64,
        inner: RestfulClient,
    }

    impl NativeBrSvcClient {
        pub fn new(
            cluster_id: u64,
            endpoints: Vec<String>,
            security_mgr: Arc<SecurityManager>,
        ) -> HttpResult<Self> {
            Ok(Self {
                cluster_id,
                inner: box_try!(RestfulClient::new("native_br_cli", endpoints, security_mgr)),
            })
        }

        pub async fn list_backups(
            &self,
            last_backup_time: &DateTime<Utc>,
        ) -> HttpResult<ListBackupResponse> {
            let query = url::form_urlencoded::Serializer::new(String::new())
                .extend_pairs([
                    ("cluster_id", self.cluster_id.to_string()),
                    (
                        "last_backup_time",
                        last_backup_time.format(JSON_TIME_FORMAT).to_string(),
                    ),
                ])
                .finish();
            Ok(box_try!(
                self.inner.get(format!("api/v1/backups?{query}")).await
            ))
        }

        pub async fn list_backups_x(
            &self,
            from_prefix: String,
            load_details: bool,
            max_count: u64,
        ) -> HttpResult<super::v1x::ListBackupResponse> {
            let query = url::form_urlencoded::Serializer::new(String::new())
                .extend_pairs([
                    ("from_prefix", from_prefix),
                    ("load_details", load_details.to_string()),
                    ("max_count", max_count.to_string()),
                ])
                .finish();
            Ok(box_try!(
                self.inner.get(format!("api/v1/x/backups?{query}")).await
            ))
        }

        pub async fn restore_keyspace(
            &self,
            restore_id: u64,
            keyspace: String,
            backup: &BackupItem,
        ) -> HttpResult<RestoreProgressResponse> {
            let query = url::form_urlencoded::Serializer::new(String::new())
                .extend_pairs([
                    ("cluster_id", self.cluster_id.to_string()),
                    ("keyspace", keyspace),
                    ("backup_id", backup.id.to_string()),
                    ("backup_name", backup.name.clone()),
                ])
                .finish();
            Ok(box_try!(
                self.inner
                    .put(
                        format!("api/v1/restore_keyspace/{restore_id}?{query}"),
                        &DummyRequest {}
                    )
                    .await
            ))
        }

        pub async fn get_restore_progress(
            &self,
            restore_id: u64,
            keyspace: String,
        ) -> HttpResult<RestoreProgressResponse> {
            let query = url::form_urlencoded::Serializer::new(String::new())
                .extend_pairs([
                    ("cluster_id", self.cluster_id.to_string()),
                    ("keyspace", keyspace),
                ])
                .finish();
            Ok(box_try!(
                self.inner
                    .get(format!("api/v1/restore_keyspace/{restore_id}?{query}"))
                    .await
            ))
        }
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
        let (br_manager, _temp_dir) = new_test_br_manager("2s");

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

    fn new_test_br_manager(restore_task_ttl: &str) -> (NativeBrManager, tempfile::TempDir) {
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
        let temp_dir = tempfile::tempdir().unwrap();

        (
            NativeBrManager::new(
                thread_pool,
                Arc::new(MockPdClient {}),
                s3fs,
                temp_dir.path().to_path_buf(),
                Config {
                    native_br: NativeBrConfig {
                        restore_task_ttl: ReadableDuration::from_str(restore_task_ttl).unwrap(),
                        ..Default::default()
                    },
                    ..Default::default()
                },
            ),
            temp_dir,
        )
    }

    struct MockPdClient {}

    impl PdClient for MockPdClient {}

    impl GetSecurityManager for MockPdClient {}
}
