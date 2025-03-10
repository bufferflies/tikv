// Copyright 2023 TiKV Project Authors. Licensed under Apache-2.0.

use std::{cmp::max, default::Default, fs, ops::Deref, str::FromStr, sync::Arc, time::Duration};

use dashmap::{mapref::entry::Entry, DashMap};
use futures::StreamExt;
use http::Uri;
use k8s_openapi::{
    api::{
        apps::v1::StatefulSet,
        core::v1::{
            EnvVar, PersistentVolumeClaim, Pod, Service, ServicePort, ServiceSpec, VolumeMount,
        },
    },
    apimachinery::pkg::{apis::meta::v1::ObjectMeta, util::intstr::IntOrString},
    serde_json,
};
use kube::{
    api::{Api, AttachParams, AttachedProcess, ListParams, PostParams, ResourceExt},
    Client as KubeClient,
};
use load_data::task::LoadTaskStates;
use security::SecurityManager;
use serde_json::json;
use tikv_util::{box_err, config::ReadableSize, error, info, time::Instant, warn};
use tokio::sync::RwLock;

use crate::metrics::WORKER_SCALER_QUERY_FAILURES_COUNTER_VEC;

const CLEAN_UP_WORKER_TICK_INTERVAL: u64 = 60;
const WORKER_MIN_STORAGE_GB: usize = 16;
const DEFAULT_LOAD_DATA_WORKER_NAME: &str = "load-data-worker";
const DEFAULT_LOAD_DATA_WORKER_PORT: u16 = 19500;
const DEFAULT_MAX_SIZE: ReadableSize = ReadableSize::gb(1024);
const DEFAULT_SPAWN_DATA_SIZE: ReadableSize = ReadableSize::gb(2);
const DEFAULT_SPAWN_RUNNING_TASKS: usize = 8;
const DEFAULT_WORKER_MAX_CORES: f64 = 4.0;
const DEFAULT_EXPIRE_SECONDS: i64 = 60 * 10;
const DEFAULT_NAMESPACE: &str = "tidb-serverless";
const DEFAULT_TEMPLATE_STS_NAME: &str = "tikv-api";
const DEFAULT_WORKER_COUNT_LIMIT: usize = 1024;
const DEFAULT_WAIT_POD_READY_TIMEOUT: u64 = 60 * 5;

const APP_LABEL: &str = "app";
const K8S_LABEL_NAME: &str = "app.kubernetes.io/name";
const K8S_LABEL_SERVICE: &str = "app.kubernetes.io/service";
const K8S_LABEL_COMPONENT: &str = "app.kubernetes.io/component";
const K8S_NAMESPACE_PATH: &str = "/var/run/secrets/kubernetes.io/serviceaccount/namespace";

const LOAD_DATA_WORKER_NODE_GROUP_NAME: &str = "load-data-worker";
// In order to let the load data worker pod exclusively occupy the node of the
// node group, we use one-half of the node's CPU number as the request.
const LOAD_DATA_WORKER_NODE_GROUP_CORES_NUM: f64 = 48.0 / 2.0 + 1.0;

const NETWORK_PROTOCOL: &str = "TCP";

const WORKER_SCALER_QUERY_FAILURES_LIMIT: usize = 10;

pub const LOAD_DATA_WORKER_ENV: &str = "TIKV_LOAD_DATA_WORKER";
pub const LOAD_DATA_WORKER_WORKER_NUM_ENV: &str = "TIKV_LOAD_DATA_WORKER_WORKER_NUM_ENV";

#[derive(Clone, Serialize, Deserialize, PartialEq, Debug)]
#[serde(default)]
#[serde(rename_all = "kebab-case")]
pub struct WorkerScalerConfig {
    pub run: bool,
    pub namespace: String,
    pub template_sts_name: String,
    pub name: String,
    pub worker_port: u16,
    pub max_size: ReadableSize,
    pub spawn_data_size: ReadableSize,
    pub spawn_running_tasks: usize,
    pub worker_max_cores: f64,
    pub expire_seconds: i64,
    pub worker_count_limit: usize,
}

impl Default for WorkerScalerConfig {
    fn default() -> Self {
        Self {
            run: false,
            namespace: DEFAULT_NAMESPACE.to_string(),
            template_sts_name: DEFAULT_TEMPLATE_STS_NAME.to_string(),
            name: DEFAULT_LOAD_DATA_WORKER_NAME.to_string(),
            worker_port: DEFAULT_LOAD_DATA_WORKER_PORT,
            max_size: DEFAULT_MAX_SIZE,
            spawn_data_size: DEFAULT_SPAWN_DATA_SIZE,
            spawn_running_tasks: DEFAULT_SPAWN_RUNNING_TASKS,
            worker_max_cores: DEFAULT_WORKER_MAX_CORES,
            expire_seconds: DEFAULT_EXPIRE_SECONDS,
            worker_count_limit: DEFAULT_WORKER_COUNT_LIMIT,
        }
    }
}

impl WorkerScalerConfig {
    fn label_selector(&self) -> String {
        format!("{}={}", K8S_LABEL_NAME, self.name)
    }
}

#[derive(Clone)]
pub(crate) struct WorkerScaler {
    core: Arc<WorkerScalerCore>,
}

impl Deref for WorkerScaler {
    type Target = WorkerScalerCore;

    fn deref(&self) -> &Self::Target {
        &self.core
    }
}

pub(crate) struct WorkerScalerCore {
    sts_template: StatefulSet,
    sts_api: Api<StatefulSet>,
    svc_api: Api<Service>,
    pvc_api: Api<PersistentVolumeClaim>,
    pod_api: Api<Pod>,
    config: WorkerScalerConfig,
    cluster_id: u64,
    in_k8s: bool,
    http_client: security::HttpClient,
    pods_map: DashMap<String, Arc<RwLock<WorkerPod>>>,
    security_mgr: Arc<SecurityManager>,
}

struct StsConfig {
    // The following fields are used to generate sts.
    core_num: f64,
    storage_size_gb: usize,
    enable_node_group: bool,

    // The following fields are used to configure the load-data-worker.
    worker_num: usize,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct WorkerPod {
    name: String,
    started_at: i64,
    updated_at: i64,
    svc_name: String,
    canceled: bool,
    finished_at: i64,
    flushed_files: usize,
    created_files: usize,
    ingested_regions: usize,
    query_failures: usize,
}

impl WorkerPod {
    fn new(task_id: &str, pod: &Pod) -> Self {
        let mut worker_pod = Self::default();
        worker_pod.name = pod.name_any();
        if let Some(status) = &pod.status {
            if let Some(start) = &status.start_time {
                worker_pod.started_at = start.0.timestamp();
                worker_pod.updated_at = start.0.timestamp();
                worker_pod.svc_name = new_worker_svc_name(task_id);
            }
        }
        worker_pod
    }

    fn init(&mut self, task_id: &str, pod: &Pod) {
        self.name = pod.name_any();
        if let Some(status) = &pod.status {
            if let Some(start) = &status.start_time {
                self.started_at = start.0.timestamp();
                self.updated_at = start.0.timestamp();
                self.svc_name = new_worker_svc_name(task_id);
            }
        }
    }

    fn update_task_states(
        &mut self,
        tasks: Option<Vec<LoadTaskStates>>,
        now_timestamp: i64,
        expire_seconds: i64,
    ) {
        if self.started_at == 0 {
            return;
        }
        if self.started_at > 0 {
            let start_duration_secs = now_timestamp - self.started_at;
            if start_duration_secs < expire_seconds {
                return;
            }
        }
        if let Some(tasks) = tasks {
            if tasks.is_empty() {
                // the task is cleaned up by client and GCed by the load data worker.
                info!("task {} is GCed by the load data worker", self.name);
                self.canceled = true;
            } else {
                let task = &tasks[0];
                if task.canceled {
                    // the task is cleaned up by client.
                    info!("task {} is cleaned up by the client", self.name);
                    self.canceled = true;
                }
                if task.finished && self.finished_at == 0 {
                    info!("task {} is finished", self.name);
                    self.finished_at = now_timestamp;
                }
                if self.flushed_files != task.flushed_files
                    || self.created_files != task.created_files
                    || self.ingested_regions != task.ingested_regions
                {
                    self.flushed_files = task.flushed_files;
                    self.created_files = task.created_files;
                    self.ingested_regions = task.ingested_regions;
                    self.updated_at = now_timestamp;
                    info!("task {} progress updated", self.name;
                        "flushed_files" => self.flushed_files,
                        "created_files" => self.created_files,
                        "ingested_regions" => self.ingested_regions,
                        "updated_at" => self.updated_at,
                    );
                }
            }
            // reset query_failures
            self.query_failures = 0;
            WORKER_SCALER_QUERY_FAILURES_COUNTER_VEC
                .with_label_values(&[&self.name])
                .reset();
        } else {
            warn!("failed get pod {} task states", self.name);
            WORKER_SCALER_QUERY_FAILURES_COUNTER_VEC
                .with_label_values(&[&self.name])
                .inc();
            self.query_failures += 1;
            if self.query_failures > WORKER_SCALER_QUERY_FAILURES_LIMIT {
                self.canceled = true;
            }
        }
        if self.finished_at > 0 {
            let finished_duration = now_timestamp - self.finished_at;
            if finished_duration > expire_seconds {
                // The client finished the task but failed to explicit clean up.
                warn!(
                    "task {} is finished but the client failed to explicit clean up",
                    self.name
                );
                self.canceled = true;
            }
        } else {
            let updated_duration = now_timestamp - self.updated_at;
            if updated_duration > expire_seconds {
                // The client didn't finish the task and no progress for a long time.
                warn!(
                    "pod {} no progress for {} seconds",
                    &self.name, updated_duration
                );
                if updated_duration > expire_seconds * 30 {
                    warn!(
                        "pod {} canceled for no progress for {} seconds",
                        &self.name, updated_duration
                    );
                    self.canceled = true;
                }
            }
        }
    }
}

impl WorkerScaler {
    pub(crate) async fn new(
        cfg: &WorkerScalerConfig,
        cluster_id: u64,
        security_mgr: Arc<SecurityManager>,
    ) -> kube::Result<Self> {
        let kube_client = KubeClient::try_default().await?;
        let pvc_api: Api<PersistentVolumeClaim> =
            Api::namespaced(kube_client.clone(), &cfg.namespace);
        let sts_api: Api<StatefulSet> = Api::namespaced(kube_client.clone(), &cfg.namespace);
        let svc_api: Api<Service> = Api::namespaced(kube_client.clone(), &cfg.namespace);
        let pod_api: Api<Pod> = Api::namespaced(kube_client.clone(), &cfg.namespace);
        let sts_template = sts_api.get(&cfg.template_sts_name).await?;
        let in_k8s = fs::read(K8S_NAMESPACE_PATH).is_ok();
        let worker_scaler = Self {
            core: Arc::new(WorkerScalerCore {
                sts_template,
                svc_api,
                sts_api,
                pvc_api,
                pod_api,
                pods_map: dashmap::DashMap::default(),
                config: cfg.clone(),
                cluster_id,
                in_k8s,
                http_client: security_mgr.http_client(hyper::Client::builder()).unwrap(),
                security_mgr,
            }),
        };
        worker_scaler.init().await?;
        Ok(worker_scaler)
    }

    pub(crate) async fn run(&self) {
        info!("worker scaler started");
        let interval = Duration::from_secs(CLEAN_UP_WORKER_TICK_INTERVAL);
        let mut timer = tokio::time::interval(interval);
        let pvc_ticks = self.config.expire_seconds as u64 / CLEAN_UP_WORKER_TICK_INTERVAL;
        let mut tick_cnt = 0;
        loop {
            timer.tick().await;
            self.clean_up_finished_workers().await;
            tick_cnt += 1;
            if tick_cnt % pvc_ticks == 0 {
                self.clean_up_orphan_pvcs().await;
            }
        }
    }

    async fn init(&self) -> kube::Result<()> {
        let list_params = ListParams::default()
            .labels(&self.config.label_selector())
            .timeout(15);
        // TODO: the number of pods may not be consistent with the number of sts.
        let mut pods = self.pod_api.list(&list_params).await?;
        for pod in pods.items.drain(..) {
            let name = pod.name_any();
            let task_id = parse_task_id_by_pod_name(name.as_str());
            assert!(!task_id.is_empty());
            info!("load pod {} task {}", name, task_id);
            let worker_pod = WorkerPod::new(&task_id, &pod);
            self.pods_map
                .insert(task_id, Arc::new(RwLock::new(worker_pod)));
        }
        Ok(())
    }

    async fn clean_up_finished_workers(&self) {
        let task_ids: Vec<String> = self.pods_map.iter().map(|x| x.key().clone()).collect();
        for task_id in task_ids {
            self.maybe_clean_up(&task_id).await;
        }
    }

    async fn clean_up_orphan_pvcs(&self) {
        let list_params = ListParams::default()
            .labels(&self.config.label_selector())
            .timeout(15);
        let res = self.pvc_api.list(&list_params).await;
        if let Err(err) = res {
            error!("list pvc err {:?}", err);
            return;
        }
        let mut pvcs = res.unwrap();
        let mut orphan_pvcs = vec![];
        for pvc in pvcs.items.iter_mut() {
            let name = pvc.name_any();
            let task_id = parse_task_id_by_pvc_name(&name);
            assert!(!task_id.is_empty());
            if !self.pods_map.contains_key(&task_id) {
                orphan_pvcs.push(name);
            }
        }
        self.delete_pvc_by_names(orphan_pvcs).await;
    }

    pub(crate) async fn create_worker(
        &self,
        task_id: &str,
        data_size_gb: usize,
    ) -> kube::Result<()> {
        if self.pods_map.len() >= self.config.worker_count_limit {
            return Err(kube::Error::Service(box_err!(
                "worker count limit {} reached",
                self.config.worker_count_limit
            )));
        }

        let mut create_worker = false;
        match self.pods_map.entry(task_id.to_string()) {
            Entry::Occupied(_) => {
                // don't return directly because the pod may be not ready.
            }
            Entry::Vacant(e) => {
                let mut worker_pod = WorkerPod::default();
                worker_pod.name = new_worker_pod_name(task_id);
                e.insert(Arc::new(RwLock::new(worker_pod)));
                create_worker = true;
            }
        }

        if create_worker {
            let sts_config = gen_sts_config(data_size_gb, self.config.worker_max_cores);
            let sts_name = new_worker_sts_name(task_id);
            let svc_name = new_worker_svc_name(task_id);
            self.create_sts(task_id, sts_name.clone(), sts_config)
                .await?;
            self.create_svc(
                svc_name.clone(),
                sts_name.clone(),
                self.config.worker_port as i32,
            )
            .await?;
            info!("created sts {:?} with svc {:?}", sts_name, svc_name);
        }

        let pod = self
            .wait_worker_pod_ready(task_id, Duration::from_secs(DEFAULT_WAIT_POD_READY_TIMEOUT))
            .await?;
        let locked_worker_pod = self.pods_map.get_mut(task_id).unwrap().clone();
        let mut worker_pod = locked_worker_pod.write().await;
        if worker_pod.started_at == 0 {
            worker_pod.init(task_id, &pod);
        }
        Ok(())
    }

    async fn create_sts(
        &self,
        task_id: &str,
        sts_name: String,
        sts_config: StsConfig,
    ) -> kube::Result<()> {
        let mut sts = self.sts_template.clone();
        sts.metadata = serde_json::from_value(json!({
            "name": sts_name.clone(),
            "namespace": self.config.namespace.clone(),
            "labels": {
                K8S_LABEL_NAME: self.config.name.clone()
            },
        }))
        .unwrap();
        let spec = sts.spec.as_mut().unwrap();
        spec.selector = serde_json::from_value(json!({
            "matchLabels": {
                K8S_LABEL_NAME: self.config.name.clone()
            }
        }))
        .unwrap();
        spec.replicas = Some(1);
        let pod_template = &mut spec.template;
        let pod_metadata = pod_template.metadata.as_mut().unwrap();
        let mut labels = pod_metadata.labels.take().unwrap_or_default();
        labels.insert(APP_LABEL.to_string(), self.config.name.clone());
        labels.insert(K8S_LABEL_NAME.to_string(), self.config.name.clone());
        labels.insert(K8S_LABEL_COMPONENT.to_string(), self.config.name.clone());
        labels.insert(K8S_LABEL_SERVICE.to_string(), sts_name.clone());
        pod_metadata.labels = Some(labels);
        let pod_template_spec = pod_template.spec.as_mut().unwrap();
        if sts_config.enable_node_group {
            pod_template_spec.node_selector = Some(
                serde_json::from_value(json!({
                    "serverless.tidbcloud.com/node": LOAD_DATA_WORKER_NODE_GROUP_NAME,
                }))
                .unwrap(),
            );
            let tolerations = pod_template_spec.tolerations.as_mut().unwrap();
            tolerations.first_mut().unwrap().value =
                Some(LOAD_DATA_WORKER_NODE_GROUP_NAME.to_string());
        }
        let pod_container = pod_template_spec.containers.first_mut().unwrap();
        let core_num = sts_config.core_num;
        let request_cpu = format!("{}", core_num);
        let request_memory = format!("{}Gi", core_num * 4.0);
        let limit_cpu = format!("{}", core_num);
        let limit_memory = format!("{}Gi", core_num * 4.0);
        let resources = if sts_config.enable_node_group {
            // If the pod use `LOAD_DATA_WORKER_NODE_GROUP_NAME`, the pod will
            // exclusively occupy the node, so we don't need to set limits.
            serde_json::from_value(json!({
                "requests": {
                    "cpu": request_cpu,
                    "memory": request_memory
                },
            }))
            .unwrap()
        } else {
            serde_json::from_value(json!({
                "requests": {
                    "cpu": request_cpu,
                    "memory": request_memory
                },
                "limits": {
                    "cpu": limit_cpu,
                    "memory": limit_memory
                },
            }))
            .unwrap()
        };

        pod_container.resources = Some(resources);
        // disable liveness probe because we don't want the pod to be restarted by k8s.
        pod_container.liveness_probe = None;
        // add environment variable to prevent the load-data-worker to run
        // worker-scaler.
        let env = pod_container.env.as_mut().unwrap();
        env.push(EnvVar {
            name: LOAD_DATA_WORKER_ENV.to_string(),
            value: Some("true".to_string()),
            value_from: None,
        });
        env.push(EnvVar {
            name: LOAD_DATA_WORKER_WORKER_NUM_ENV.to_string(),
            value: Some(sts_config.worker_num.to_string()),
            value_from: None,
        });

        let pvcs = spec.volume_claim_templates.as_mut().unwrap();
        let first_pvc = pvcs.first_mut().unwrap();
        first_pvc.metadata.labels = Some(
            serde_json::from_value(json!({
                K8S_LABEL_NAME: self.config.name.clone()
            }))
            .unwrap(),
        );

        let mut pvc_template = first_pvc.clone();
        let volume_mounts = pod_container.volume_mounts.as_mut().unwrap();
        let root_path = volume_mounts.first().unwrap().mount_path.clone();
        for worker_id in 0..sts_config.worker_num {
            let name = format!("worker{}", worker_id);
            pvc_template.metadata.name = Some(name.clone());
            let pvc_template_spec = pvc_template.spec.as_mut().unwrap();
            pvc_template_spec.resources = Some(
                serde_json::from_value(json!({
                    "requests": {
                        "storage": format!("{}Gi", sts_config.storage_size_gb)
                    }
                }))
                .unwrap(),
            );
            pvcs.push(pvc_template.clone());
            volume_mounts.push(VolumeMount {
                mount_path: format!("{}/{}/worker-{}", root_path, task_id, worker_id),
                mount_propagation: None,
                name,
                read_only: None,
                sub_path: None,
                sub_path_expr: None,
            })
        }

        info!("create sts {:?}", sts_name);
        self.sts_api.create(&PostParams::default(), &sts).await?;
        Ok(())
    }

    async fn wait_worker_pod_ready(&self, task_id: &str, timeout: Duration) -> kube::Result<Pod> {
        let start = Instant::now();
        let pod_name = new_worker_pod_name(task_id);
        loop {
            if start.saturating_elapsed() > timeout {
                return Err(kube::Error::Service(box_err!(format!(
                    "wait worker {} ready timeout",
                    pod_name
                ))));
            }
            tokio::time::sleep(Duration::from_secs(3)).await;

            let pod = self.pod_api.get(&pod_name).await?;
            if pod.status.is_none() {
                continue;
            }
            let status = pod.status.as_ref().unwrap();
            if status.phase == Some("Running".into()) {
                return Ok(pod);
            }
        }
    }

    async fn delete_sts(&self, task_id: &str) {
        let sts_name = new_worker_sts_name(task_id);
        if let Err(err) = self.sts_api.delete(&sts_name, &Default::default()).await {
            warn!("delete sts {} err {:?}", sts_name, err);
        }
    }

    async fn delete_svc(&self, task_id: &str) {
        let svc_name = new_worker_svc_name(task_id);
        if let Err(err) = self.svc_api.delete(&svc_name, &Default::default()).await {
            warn!("delete sts {} err {:?}", svc_name, err);
        }
    }

    async fn list_pvcs_by_task_id(&self, task_id: String) -> kube::Result<Vec<String>> {
        let list_params = ListParams::default()
            .labels(&self.config.label_selector())
            .timeout(15);
        let pvcs = self.pvc_api.list(&list_params).await?;
        let mut pvc_names = vec![];
        for pvc in pvcs.items.iter() {
            let name = pvc.name_any();
            let cur_task_id = parse_task_id_by_pvc_name(&name);
            assert!(!task_id.is_empty());
            if cur_task_id == task_id {
                pvc_names.push(name);
            }
        }
        Ok(pvc_names)
    }

    async fn delete_pvc(&self, task_id: &str) {
        match self.list_pvcs_by_task_id(task_id.to_string()).await {
            Ok(pvc_names) => {
                self.delete_pvc_by_names(pvc_names).await;
            }
            Err(err) => {
                warn!("list {} pvc err {:?}", task_id, err);
            }
        }
    }

    async fn delete_pvc_by_names(&self, pvc_names: Vec<String>) {
        for pvc_name in &pvc_names {
            if let Err(err) = self.pvc_api.delete(pvc_name, &Default::default()).await {
                warn!("delete pvc {} err {:?}", pvc_name, err);
            }
        }
    }

    async fn get_pod_task_states(
        &self,
        worker_pod_name: &str,
        worker_addr: &str,
    ) -> Option<Vec<LoadTaskStates>> {
        let load_data_url = self.get_worker_tasks_url(worker_addr);
        let resp_body = if self.in_k8s {
            let uri = Uri::from_str(load_data_url.as_str()).unwrap();
            let resp_fut = self.http_client.get(uri);
            let timeout_dur = Duration::from_secs(5);
            let result = tokio::time::timeout(timeout_dur, resp_fut).await.ok()?;
            let resp = result.ok()?;
            if !resp.status().is_success() {
                error!("pod task status error: {}", resp.status());
                return None;
            }
            let body = hyper::body::to_bytes(resp.into_body()).await.ok()?;
            String::from_utf8(body.to_vec()).ok()?
        } else {
            let ap = AttachParams::default();
            let proc = self
                .pod_api
                .exec(worker_pod_name, vec!["curl", load_data_url.as_str()], &ap)
                .await
                .ok()?;
            get_proc_output(proc).await
        };
        let tasks: Vec<LoadTaskStates> = serde_json::from_str(&resp_body).ok()?;
        Some(tasks)
    }

    async fn get_pod_task_states_with_retry(
        &self,
        worker_pod_name: &str,
        worker_addr: &str,
    ) -> Option<Vec<LoadTaskStates>> {
        for _ in 0..18 {
            let tasks = self.get_pod_task_states(worker_pod_name, worker_addr).await;
            if tasks.is_some() {
                return tasks;
            }
            info!(
                "{} can't get tasks with addr {}",
                worker_pod_name, worker_addr
            );
            tokio::time::sleep(Duration::from_secs(10)).await;
        }
        None
    }

    async fn maybe_clean_up(&self, task_id: &str) {
        info!("try to clean up {}", task_id);
        let pod_name = new_worker_pod_name(task_id);
        let pod = self.pod_api.get(&pod_name).await;
        if let Err(err) = pod {
            warn!("get pod {} err {:?}", pod_name, err);
            return;
        }
        let pod = pod.unwrap();
        if pod.status.is_none() || pod.status.clone().unwrap().phase.unwrap() != "Running" {
            return;
        }

        let locked_worker_pod = self.pods_map.get_mut(task_id).unwrap().clone();
        let mut worker_pod = locked_worker_pod.write().await;
        if worker_pod.started_at == 0 {
            worker_pod.init(task_id, &pod);
        }

        // `canceled` flag will be set. We did not clean up the task
        // immediately, this way we can ensure that the pod can survive for
        // `CLEAN_UP_WORKER_TICK_INTERVAL` after completion.
        if worker_pod.canceled {
            info!("clean up task {}", task_id);
            self.delete_sts(task_id).await;
            self.delete_svc(task_id).await;
            self.delete_pvc(task_id).await;
            self.pods_map.remove(task_id);
            return;
        }
        let worker_addr = self.get_worker_addr(&worker_pod);
        if worker_addr.is_none() {
            error!("{} cann't get worker addr", task_id);
            return;
        }
        let worker_pod_name = worker_pod.name.clone();
        // release lock
        drop(worker_pod);

        let tasks = self
            .get_pod_task_states_with_retry(&worker_pod_name, &worker_addr.unwrap())
            .await;
        let now_timestamp = chrono::Utc::now().timestamp();

        let mut worker_pod = locked_worker_pod.write().await;
        worker_pod.update_task_states(tasks, now_timestamp, self.config.expire_seconds);
    }

    pub(crate) fn get_worker_addr(&self, worker_pod: &WorkerPod) -> Option<String> {
        let svc_name = worker_pod.svc_name.as_str();
        self.security_mgr
            .build_uri(format!(
                "{}.{}.svc.cluster.local:{}/load_data",
                svc_name, &self.config.namespace, self.config.worker_port,
            ))
            .ok()
            .map(|uri| uri.to_string())
    }

    pub(crate) async fn get_worker_addr_by_task_id(&self, task_id: &str) -> Option<String> {
        let start = Instant::now();
        let timeout = Duration::from_secs(DEFAULT_WAIT_POD_READY_TIMEOUT);
        loop {
            {
                let locked_worker_pod = self.pods_map.get(task_id)?;
                let worker_pod = locked_worker_pod.read().await;
                if !worker_pod.svc_name.is_empty() {
                    return self.get_worker_addr(&worker_pod);
                }
            }
            if start.saturating_elapsed() > timeout {
                warn!("{} wait for load data worker timeout", task_id);
                return None;
            }
            tokio::time::sleep(Duration::from_secs(3)).await;
        }
    }

    pub(crate) fn get_worker_tasks_url(&self, worker_addr: &str) -> String {
        format!("{}?cluster_id={}", worker_addr, self.cluster_id)
    }

    pub(crate) async fn create_svc(
        &self,
        name: String,
        service: String,
        port: i32,
    ) -> kube::Result<()> {
        let svc = Service {
            metadata: ObjectMeta {
                name: Some(name.clone()),
                ..Default::default()
            },
            spec: Some(ServiceSpec {
                // create a headless svc
                cluster_ip: None,
                selector: Some(
                    [
                        (K8S_LABEL_NAME.to_string(), self.config.name.to_string()),
                        (K8S_LABEL_SERVICE.to_string(), service.to_string()),
                    ]
                    .iter()
                    .cloned()
                    .collect(),
                ),
                ports: Some(vec![ServicePort {
                    name: Some(format!("{}-port", name)),
                    port,
                    protocol: Some(NETWORK_PROTOCOL.to_string()),
                    target_port: Some(IntOrString::Int(port)),
                    ..Default::default()
                }]),
                ..Default::default()
            }),
            ..Default::default()
        };
        info!("create svc {:?}", name);
        self.svc_api.create(&PostParams::default(), &svc).await?;
        Ok(())
    }
}

fn parse_task_id_by_pod_name(pod_name: &str) -> String {
    // pod name format: load-data-worker-{task-id}-0
    // task-id:         {keyspace-id}-{lightning-task-id}-{table-id}-{engine-id}
    //
    let fields: Vec<&str> = pod_name.split('-').collect();
    if fields.len() == 8 {
        // load-data-worker-1-1701753296955293956-139-0-0
        return format!("{}-{}-{}-{}", fields[3], fields[4], fields[5], fields[6]);
    } else if fields.len() == 9 {
        //  load-data-worker-1-11701753296955293956-139--1-0
        return format!("{}-{}-{}--{}", fields[3], fields[4], fields[5], fields[7]);
    }
    "".to_string()
}

fn parse_task_id_by_pvc_name(pvc_name: &str) -> String {
    // pvc name format: {pvc-template-name}-load-data-worker-{task-id}-0
    // task-id:         {keyspace-id}-{lightning-task-id}-{table-id}-{engine-id}
    //
    let fields: Vec<&str> = pvc_name.split('-').collect();
    if fields.len() == 9 {
        return format!("{}-{}-{}-{}", fields[4], fields[5], fields[6], fields[7]);
    } else if fields.len() == 10 {
        return format!("{}-{}-{}--{}", fields[4], fields[5], fields[6], fields[8]);
    }
    "".to_string()
}

fn new_worker_sts_name(task_id: &str) -> String {
    format!("load-data-worker-{}", task_id)
}

fn new_worker_svc_name(task_id: &str) -> String {
    format!("load-data-worker-{}-service", task_id)
}

fn new_worker_pod_name(task_id: &str) -> String {
    format!("load-data-worker-{}-0", task_id)
}

async fn get_proc_output(mut attached: AttachedProcess) -> String {
    let stdout = tokio_util::io::ReaderStream::new(attached.stdout().unwrap());
    let out = stdout
        .filter_map(|r| async { r.ok().and_then(|v| String::from_utf8(v.to_vec()).ok()) })
        .collect::<Vec<_>>()
        .await
        .join("");
    attached.join().await.unwrap();
    out
}

fn gen_sts_config(data_size_gb: usize, max_cores: f64) -> StsConfig {
    let core_num = calculate_num_cores(data_size_gb, max_cores);
    let worker_num = calculate_num_workers(core_num);
    let storage_size_gb = max(data_size_gb / worker_num * 3, WORKER_MIN_STORAGE_GB);

    // align with `calculate_num_cores`
    let enable_node_group = core_num == LOAD_DATA_WORKER_NODE_GROUP_CORES_NUM;
    StsConfig {
        core_num,
        storage_size_gb,
        enable_node_group,
        worker_num,
    }
}

fn calculate_num_workers(core_num: f64) -> usize {
    // align with `calculate_num_cores`
    if core_num == LOAD_DATA_WORKER_NODE_GROUP_CORES_NUM {
        16
    } else if core_num == 14.0 {
        8
    } else {
        1
    }
}

fn calculate_num_cores(data_size_gb: usize, max_cores: f64) -> f64 {
    let cores: f64 = if data_size_gb > 500 {
        LOAD_DATA_WORKER_NODE_GROUP_CORES_NUM
    } else if data_size_gb > 100 {
        14.0
    } else if data_size_gb > 50 {
        8.0
    } else if data_size_gb > 10 {
        4.0
    } else if data_size_gb > 5 {
        2.0
    } else {
        1.0
    };
    cores.min(max_cores)
}

#[cfg(test)]
mod tests {
    use k8s_openapi::{api::core::v1::Pod, apimachinery::pkg::apis::meta::v1::Time};
    use load_data::task::LoadTaskStates;

    use crate::worker_scaler::{
        new_worker_pod_name, new_worker_svc_name, parse_task_id_by_pod_name,
        parse_task_id_by_pvc_name, WorkerPod, WORKER_SCALER_QUERY_FAILURES_LIMIT,
    };

    fn new_worker_pvc_name(pvc_template_name: &str, task_id: &str) -> String {
        format!("{}-load-data-worker-{}-0", pvc_template_name, task_id)
    }

    #[test]
    fn test_worker_pod_update() {
        let mut pod = Pod::default();
        let task_id = "taskid-1-0";
        pod.metadata.name = Some(new_worker_pod_name(task_id));
        pod.status = Some(Default::default());
        let status = pod.status.as_mut().unwrap();
        status.pod_ip = Some("127.0.0.1".to_string());
        let now = chrono::Utc::now();
        status.start_time = Some(Time(now));
        let now_ts = now.timestamp();

        let mut worker_pod = WorkerPod::new(task_id, &pod);
        assert_eq!(worker_pod.started_at, now_ts);
        assert_eq!(worker_pod.updated_at, now_ts);
        assert_eq!(worker_pod.svc_name, new_worker_svc_name(task_id));

        // worker pod is not canceled if no task states and not expired.
        worker_pod.update_task_states(None, now_ts + 40, 60);
        assert!(!worker_pod.canceled);

        // worker pod is not canceled if no task states and expired,
        // and retry retry count is less than the limit
        for _i in 0..WORKER_SCALER_QUERY_FAILURES_LIMIT {
            worker_pod.update_task_states(None, now_ts + 100, 60);
            assert!(!worker_pod.canceled);
        }

        // worker pod is canceled if no task states and expired.
        worker_pod.update_task_states(None, now_ts + 100, 60);
        assert!(worker_pod.canceled);

        // worker pod is canceled if task is empty and expired.
        worker_pod = WorkerPod::new(task_id, &pod);
        worker_pod.update_task_states(Some(vec![]), now_ts + 100, 60);
        assert!(worker_pod.canceled);

        // worker pod is canceled if task is canceled.
        let mut task_states = LoadTaskStates::default();
        task_states.canceled = true;
        worker_pod = WorkerPod::new(task_id, &pod);
        worker_pod.update_task_states(Some(vec![task_states]), now_ts + 100, 60);
        assert!(worker_pod.canceled);

        // worker pod is canceled if task is finished and not canceled, expired after
        // finish time.
        let mut task_states = LoadTaskStates::default();
        task_states.finished = true;
        worker_pod = WorkerPod::new(task_id, &pod);
        worker_pod.update_task_states(Some(vec![task_states.clone()]), now_ts + 100, 60);
        // worker pod is not canceled if task finished duration not exceed expire time.
        assert!(!worker_pod.canceled);
        worker_pod.update_task_states(Some(vec![task_states.clone()]), now_ts + 180, 60);
        // worker pod is canceled if task finished duration exceed expire time.
        assert!(worker_pod.canceled);

        // worker pod is canceled if not finished and no progress for a long time.
        worker_pod = WorkerPod::new(task_id, &pod);
        task_states = LoadTaskStates::default();
        task_states.created_files = 1;
        worker_pod.update_task_states(Some(vec![task_states.clone()]), now_ts + 100, 60);
        assert_eq!(worker_pod.created_files, 1);
        assert_eq!(worker_pod.updated_at, now_ts + 100);
        assert!(!worker_pod.canceled);
        task_states.created_files = 2;
        worker_pod.update_task_states(Some(vec![task_states.clone()]), now_ts + 180, 60);
        assert_eq!(worker_pod.created_files, 2);
        assert_eq!(worker_pod.updated_at, now_ts + 180);
        assert!(!worker_pod.canceled);
        worker_pod.update_task_states(Some(vec![task_states]), now_ts + 2000, 60);
        assert_eq!(worker_pod.created_files, 2);
        assert_eq!(worker_pod.updated_at, now_ts + 180);
        assert!(worker_pod.canceled);
    }

    #[test]
    fn test_parse_task_id() {
        let task_ids = vec!["1-taskid-1-0", "1-taskid-1--0"];
        for task_id in task_ids {
            let pvc_template_name = "data0";
            let pod_name = new_worker_pod_name(task_id);
            let pvc_name = new_worker_pvc_name(pvc_template_name, task_id);
            assert_eq!(task_id, parse_task_id_by_pod_name(pod_name.as_str()));
            assert_eq!(task_id, parse_task_id_by_pvc_name(pvc_name.as_str()));
        }
    }
}
