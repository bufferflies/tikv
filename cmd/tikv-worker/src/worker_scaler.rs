// Copyright 2023 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    collections::{HashMap, VecDeque},
    default::Default,
    fs,
    str::FromStr,
    time::Duration,
};

use async_trait::async_trait;
use futures::StreamExt;
use http::Uri;
use hyper::client::HttpConnector;
use k8s_openapi::{
    api::{apps::v1::Deployment, core::v1::Pod},
    serde_json,
};
use kube::{
    api::{Api, AttachParams, AttachedProcess, ListParams, Patch, PatchParams, ResourceExt},
    Client as KubeClient, Resource,
};
use tikv_util::{error, info};

const WORKER_SCALE_OUT_SAMPLES: usize = 3;
const WORKER_SCALE_IN_SAMPLES: usize = 30;

#[derive(Clone, Serialize, Deserialize, PartialEq, Debug)]
#[serde(default)]
#[serde(rename_all = "kebab-case")]
pub struct WorkerScalerConfig {
    pub run: bool,
    pub namespace: String,
    pub label: String,
    pub worker_port: u16,
    pub cpu_max_ratio: f64,
    pub cpu_min_ratio: f64,
    pub sample_interval: f64,
    pub max_replicas: usize,
    pub min_replicas: usize,
}

impl Default for WorkerScalerConfig {
    fn default() -> Self {
        Self {
            run: false,
            namespace: "tidb-serverless".to_string(),
            label: "app.kubernetes.io/name=tikv-worker".to_string(),
            worker_port: 19000,
            cpu_max_ratio: 0.6,
            cpu_min_ratio: 0.4,
            sample_interval: 10.0,
            max_replicas: 64,
            min_replicas: 1,
        }
    }
}

pub(crate) async fn new_worker_scaler(cfg: &WorkerScalerConfig) -> kube::Result<WorkerScaler> {
    let kube_client = KubeClient::try_default().await?;
    let pods: Api<Pod> = Api::namespaced(kube_client.clone(), &cfg.namespace);
    let worker_pods = WorkerPods::new(pods, cfg);
    let deploy: Api<Deployment> = Api::namespaced(kube_client.clone(), &cfg.namespace);
    let worker_deploy = WorkerDeploy::new(deploy);
    let worker_scaler =
        WorkerScaler::new(cfg, Box::new(worker_pods), Box::new(worker_deploy)).await?;
    Ok(worker_scaler)
}

pub struct WorkerScaler {
    pods: Box<dyn WorkerPodsApi>,
    deploy: Box<dyn WorkerDeployApi>,
    config: WorkerScalerConfig,
    cpu_usages: HashMap<String, f64>, // (pod_name, cpu_seconds_accumulated)
    cpu_delta_samples: VecDeque<(f64, usize)>, // (cpu_seconds_delta, replicas)
    current_replicas: usize,
}

impl WorkerScaler {
    pub(crate) async fn new(
        cfg: &WorkerScalerConfig,
        pods: Box<dyn WorkerPodsApi>,
        deploy: Box<dyn WorkerDeployApi>,
    ) -> kube::Result<Self> {
        let current_replicas = deploy.get_replica().await?;
        Ok(Self {
            pods,
            deploy,
            config: cfg.clone(),
            cpu_usages: HashMap::default(),
            cpu_delta_samples: VecDeque::default(),
            current_replicas,
        })
    }

    pub(crate) async fn run(&mut self) {
        info!("worker scaler started");
        let interval = Duration::from_secs_f64(self.config.sample_interval);
        let mut timer = tokio::time::interval(interval);
        loop {
            timer.tick().await;
            if let Err(err) = self.update_cpu_usage().await {
                error!("update cpu usage error: {}", err);
                continue;
            }
            if let Err(err) = self.update_replica().await {
                error!("update replica error: {}", err);
            }
        }
    }

    async fn update_cpu_usage(&mut self) -> kube::Result<()> {
        let mut total_cpu_delta = 0.0;
        let mut replicas = 0;
        let mut max_pod_ratio = 0.0;
        let pods = self.pods.get_running_pods().await?;
        for pod in &pods {
            let pod_cpu_usage = self.pods.get_pod_cpu_usage(pod).await;
            if pod_cpu_usage.is_none() {
                continue;
            }
            let cpu_usage = pod_cpu_usage.unwrap();
            let pod_name = pod.name_any();
            if let Some(&last) = self.cpu_usages.get(&pod_name) {
                if last < cpu_usage {
                    let pod_cpu_delta = cpu_usage - last;
                    let pod_cpu_ratio = pod_cpu_delta / self.config.sample_interval;
                    total_cpu_delta += pod_cpu_delta;
                    if max_pod_ratio < pod_cpu_ratio {
                        max_pod_ratio = pod_cpu_ratio;
                    }
                }
            }
            self.cpu_usages.insert(pod_name, cpu_usage);
            replicas += 1;
        }
        let cpu_util_ratio = total_cpu_delta / self.config.sample_interval;
        info!(
            "total cpu util ratio: {}, max cpu util ratio: {}, replicas: {}",
            cpu_util_ratio, max_pod_ratio, replicas
        );
        self.cpu_delta_samples
            .push_back((total_cpu_delta, replicas));
        if self.cpu_delta_samples.len() > WORKER_SCALE_IN_SAMPLES {
            self.cpu_delta_samples.pop_front();
        }
        self.current_replicas = self.deploy.get_replica().await?;
        Ok(())
    }

    async fn update_replica(&mut self) -> kube::Result<()> {
        if !self.last_sample_replica_match_current() {
            return Ok(());
        }
        let new_replicas = if let Some(new_replicas) = self.need_scale_out() {
            new_replicas
        } else if self.need_scale_in() {
            // scale in one replica at a time to avoid high failure rate.
            self.current_replicas - 1
        } else {
            self.current_replicas
        };
        if new_replicas != self.current_replicas {
            info!(
                "scale tikv-worker to {} replicas, samples: {:?}",
                new_replicas, self.cpu_delta_samples
            );
            self.deploy.update_replica(new_replicas).await?;
        }
        Ok(())
    }

    fn last_sample_replica_match_current(&self) -> bool {
        if let Some(&(_, last_replicas)) = self.cpu_delta_samples.back() {
            last_replicas == self.current_replicas
        } else {
            false
        }
    }

    fn need_scale_out(&self) -> Option<usize> {
        let num_samples = self.cpu_delta_samples.len();
        if num_samples < WORKER_SCALE_OUT_SAMPLES {
            // didn't get enough data.
            return None;
        }
        if self.current_replicas >= self.config.max_replicas {
            // already reached max replicas.
            return None;
        }
        let mut samples_total_cpu_delta = 0.0;
        for &(cpu_delta, _) in self
            .cpu_delta_samples
            .iter()
            .skip(num_samples - WORKER_SCALE_OUT_SAMPLES)
        {
            let ratio = cpu_delta / self.config.sample_interval / self.current_replicas as f64;
            if ratio < self.config.cpu_max_ratio {
                // any sample ratio didn't reach max cpu usage, we will not scale out.
                return None;
            }
            samples_total_cpu_delta += cpu_delta;
        }
        let sample_avg_cpu_delta = samples_total_cpu_delta / WORKER_SCALE_OUT_SAMPLES as f64;
        let target_cpu_ratio = self.config.cpu_max_ratio;
        let target_replicas = sample_avg_cpu_delta / self.config.sample_interval / target_cpu_ratio;
        if target_replicas <= self.current_replicas as f64 {
            // no need to scale out.
            return None;
        }
        Some(std::cmp::min(
            target_replicas.ceil() as usize,
            self.config.max_replicas,
        ))
    }

    fn need_scale_in(&self) -> bool {
        if self.current_replicas <= self.config.min_replicas {
            // no need to scale in.
            return false;
        }
        let mut samples_total_cpu_delta = 0.0;
        for &(cpu_delta, _) in self.cpu_delta_samples.iter() {
            let ratio = cpu_delta / self.config.sample_interval / self.current_replicas as f64;
            if ratio > self.config.cpu_min_ratio {
                // any sample ratio reached min cpu usage, we will not scale in.
                return false;
            }
            samples_total_cpu_delta += cpu_delta;
        }
        let sample_avg_cpu_usage = samples_total_cpu_delta / WORKER_SCALE_IN_SAMPLES as f64;
        let target_cpu_ratio = self.config.cpu_min_ratio;
        let target_replicas = sample_avg_cpu_usage / self.config.sample_interval / target_cpu_ratio;
        if target_replicas >= self.current_replicas as f64 {
            // no need to scale in.
            return false;
        }
        true
    }
}

#[async_trait]
pub(crate) trait WorkerPodsApi: Send + Sync {
    async fn get_running_pods(&self) -> kube::Result<Vec<Pod>>;
    async fn get_pod_cpu_usage(&self, pod: &Pod) -> Option<f64>;
}

struct WorkerPods {
    pods: Api<Pod>,
    in_cluster: bool,
    http_client: hyper::Client<HttpConnector>,
    port: u16,
    label: String,
}

impl WorkerPods {
    fn new(pods: Api<Pod>, cfg: &WorkerScalerConfig) -> Self {
        let in_cluster =
            fs::read("/var/run/secrets/kubernetes.io/serviceaccount/namespace").is_ok();
        Self {
            pods,
            in_cluster,
            http_client: hyper::Client::new(),
            port: cfg.worker_port,
            label: cfg.label.clone(),
        }
    }
}

#[async_trait]
impl WorkerPodsApi for WorkerPods {
    async fn get_running_pods(&self) -> kube::Result<Vec<Pod>> {
        let list_params = ListParams::default()
            .labels(self.label.as_str())
            .timeout(15);
        let mut pods = self.pods.list(&list_params).await?;
        pods.items.retain(|p| {
            p.status.as_ref().map_or(false, |s| {
                s.phase.as_ref().map_or(false, |p| p == "Running")
            })
        });
        Ok(pods.items)
    }

    async fn get_pod_cpu_usage(&self, pod: &Pod) -> Option<f64> {
        let metrics_url = metrics_url(pod, self.port)?;
        let metrics_string = if self.in_cluster {
            let uri = Uri::from_str(metrics_url.as_str()).unwrap();
            let resp_fut = self.http_client.get(uri);
            let timeout_dur = Duration::from_secs(1);
            let result = tokio::time::timeout(timeout_dur, resp_fut).await.ok()?;
            let resp = result.ok()?;
            let body = hyper::body::to_bytes(resp.into_body()).await.ok()?;
            String::from_utf8(body.to_vec()).ok()?
        } else {
            let ap = AttachParams::default();
            let proc = self
                .pods
                .exec(&pod.name_any(), vec!["curl", metrics_url.as_str()], &ap)
                .await
                .ok()?;
            get_proc_output(proc).await
        };
        let proc_metrics = metrics_string.split('\n').collect::<Vec<_>>();
        let mut cpu_cores = get_cpu_limit(pod).unwrap_or(2.0);
        let mut cpu_seconds = 0.0;
        for proc_metric in proc_metrics {
            if proc_metric.starts_with("process_cpu_seconds_total") {
                cpu_seconds = parse_second_field_f64(proc_metric);
            } else if proc_metric.starts_with("tikv_worker_cpu_cores_quota") {
                cpu_cores = parse_second_field_f64(proc_metric);
            }
        }
        Some(cpu_seconds / cpu_cores)
    }
}

#[async_trait]
pub(crate) trait WorkerDeployApi: Send + Sync {
    async fn get_replica(&self) -> kube::Result<usize>;
    async fn update_replica(&self, replicas: usize) -> kube::Result<()>;
}

struct WorkerDeploy {
    deploy: Api<Deployment>,
}

impl WorkerDeploy {
    fn new(deploy: Api<Deployment>) -> Self {
        Self { deploy }
    }
}

#[async_trait]
impl WorkerDeployApi for WorkerDeploy {
    async fn get_replica(&self) -> kube::Result<usize> {
        let worker = self.deploy.get("tikv-worker").await?;
        Ok(worker.spec.as_ref().unwrap().replicas.unwrap() as usize)
    }

    async fn update_replica(&self, replicas: usize) -> kube::Result<()> {
        let patch = Patch::Merge(serde_json::json!({
            "spec": {
                "replicas": replicas
            }
        }));
        self.deploy
            .patch("tikv-worker", &PatchParams::default(), &patch)
            .await?;
        Ok(())
    }
}

fn parse_second_field_f64(line: &str) -> f64 {
    let fields: Vec<&str> = line.split_whitespace().collect();
    if fields.len() == 2 {
        return fields[1].parse::<f64>().unwrap_or(0.0);
    }
    0.0
}

fn metrics_url(pod: &Pod, port: u16) -> Option<String> {
    let status = pod.status.as_ref()?;
    let ip = status.pod_ip.as_ref()?;
    let namespace = pod.meta().namespace.as_ref()?;
    Some(format!(
        "http://{}.{}.pod.cluster.local:{}/metrics",
        ip.replace('.', "-"),
        namespace,
        port
    ))
}

fn get_cpu_limit(pod: &Pod) -> Option<f64> {
    let spec = pod.spec.as_ref()?;
    let container = spec.containers.first()?;
    let resources = container.resources.as_ref()?;
    let limits = resources.limits.as_ref()?;
    let cpu_limit = limits.get("cpu")?;
    cpu_limit.0.parse::<f64>().ok()
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

#[cfg(test)]
mod tests {
    use std::sync::{
        atomic::{AtomicUsize, Ordering::SeqCst},
        Arc, Mutex,
    };

    use test_util::init_log_for_test;
    use tokio::time::Interval;

    use super::*;

    #[derive(Clone)]
    struct MockWorkerPods {
        core: Arc<Mutex<MockWorkerPodsCore>>,
    }

    struct MockWorkerPodsCore {
        pods: Vec<Pod>,
        usages: HashMap<String, f64>,
        replicas: Arc<AtomicUsize>,
    }

    impl MockWorkerPods {
        fn new() -> Self {
            let core = MockWorkerPodsCore {
                pods: vec![],
                usages: HashMap::new(),
                replicas: Arc::new(AtomicUsize::new(1)),
            };
            let pods = Self {
                core: Arc::new(Mutex::new(core)),
            };
            pods.scale();
            pods
        }

        fn update_workloads(&self, workload: f64, interval: f64) {
            let mut core = self.core.lock().unwrap();
            let names = core
                .pods
                .iter()
                .map(|pod| pod.name_any())
                .collect::<Vec<_>>();
            let avg_workload = workload / names.len() as f64;
            for name in names {
                let old = core.usages.get(&name).cloned().unwrap_or_default();
                core.usages.insert(name, old + avg_workload * interval);
            }
        }

        async fn tick_update_workloads(
            &self,
            ticks: usize,
            interval: &mut Interval,
            workload: f64,
        ) {
            for _ in 0..ticks {
                interval.tick().await;
                self.update_workloads(workload, interval.period().as_secs_f64());
                self.scale();
            }
        }

        fn scale(&self) {
            let mut core = self.core.lock().unwrap();
            let new_replicas = core.replicas.load(SeqCst);
            if new_replicas == core.pods.len() {
                return;
            }
            if new_replicas < core.pods.len() {
                for _ in 0..core.pods.len() - new_replicas {
                    let pod = core.pods.pop().unwrap();
                    let name = pod.name_any();
                    core.usages.remove(&name);
                }
            } else {
                for i in core.pods.len()..new_replicas {
                    let mut pod = Pod::default();
                    pod.metadata.name = Some(format!("worker-{}", i));
                    core.usages.insert(pod.name_any(), 0.0);
                    core.pods.push(pod);
                }
            }
        }

        fn get_replicas(&self) -> Arc<AtomicUsize> {
            self.core.lock().unwrap().replicas.clone()
        }
    }

    #[async_trait]
    impl WorkerPodsApi for MockWorkerPods {
        async fn get_running_pods(&self) -> kube::Result<Vec<Pod>> {
            let core = self.core.lock().unwrap();
            Ok(core.pods.clone())
        }

        async fn get_pod_cpu_usage(&self, pod: &Pod) -> Option<f64> {
            let name = pod.name_any();
            let core = self.core.lock().unwrap();
            core.usages.get(&name).cloned()
        }
    }

    #[derive(Clone)]
    struct MockWorkerDeploy {
        replicas: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl WorkerDeployApi for MockWorkerDeploy {
        async fn get_replica(&self) -> kube::Result<usize> {
            Ok(self.replicas.load(SeqCst))
        }

        async fn update_replica(&self, replicas: usize) -> kube::Result<()> {
            self.replicas.store(replicas, SeqCst);
            Ok(())
        }
    }

    #[test]
    fn test_worker_scaler() {
        init_log_for_test();
        let pods = Box::new(MockWorkerPods::new());
        let replicas = pods.get_replicas();
        let deploy = Box::new(MockWorkerDeploy { replicas });
        let mut cfg = WorkerScalerConfig::default();
        cfg.sample_interval = 0.1;
        cfg.max_replicas = 32;
        let rt = tokio::runtime::Runtime::new().unwrap();
        let mut scaler = rt
            .block_on(WorkerScaler::new(&cfg, pods.clone(), deploy.clone()))
            .unwrap();
        rt.spawn(async move {
            scaler.run().await;
        });
        rt.block_on(async move {
            let mut interval = tokio::time::interval(Duration::from_secs_f64(0.05));
            // test scale out.
            let mut replicas = deploy.replicas.load(SeqCst);
            assert_eq!(replicas, 1);
            pods.tick_update_workloads(WORKER_SCALE_OUT_SAMPLES * 3, &mut interval, 0.9)
                .await;
            replicas = deploy.replicas.load(SeqCst);
            assert_eq!(replicas, 2);
            pods.tick_update_workloads(WORKER_SCALE_OUT_SAMPLES * 3, &mut interval, 1.9)
                .await;
            replicas = deploy.replicas.load(SeqCst);
            assert!(replicas > 2 && replicas <= 4, "{}", replicas);
            pods.tick_update_workloads(WORKER_SCALE_OUT_SAMPLES * 3, &mut interval, 3.1)
                .await;
            replicas = deploy.replicas.load(SeqCst);
            assert!(replicas > 4 && replicas <= 6, "{}", replicas);
            pods.tick_update_workloads(WORKER_SCALE_OUT_SAMPLES * 3, &mut interval, 5.1)
                .await;
            replicas = deploy.replicas.load(SeqCst);
            assert!(replicas > 6 && replicas <= 9, "{}", replicas);
            pods.tick_update_workloads(WORKER_SCALE_OUT_SAMPLES * 3, &mut interval, 8.9)
                .await;
            replicas = deploy.replicas.load(SeqCst);
            assert!(replicas > 9 && replicas <= 15, "{}", replicas);
            pods.tick_update_workloads(WORKER_SCALE_OUT_SAMPLES * 3, &mut interval, 14.9)
                .await;
            replicas = deploy.replicas.load(SeqCst);
            assert!(replicas > 15 && replicas <= 25, "{}", replicas);
            pods.tick_update_workloads(WORKER_SCALE_OUT_SAMPLES * 3, &mut interval, 24.9)
                .await;
            replicas = deploy.replicas.load(SeqCst);
            assert_eq!(replicas, 32);

            // test scale in.
            pods.tick_update_workloads(WORKER_SCALE_IN_SAMPLES * 2 + 5, &mut interval, 1.9)
                .await;
            replicas = deploy.replicas.load(SeqCst);
            assert!(replicas > 20 && replicas < 32, "{}", replicas);

            pods.tick_update_workloads(WORKER_SCALE_IN_SAMPLES, &mut interval, 0.1)
                .await;
            replicas = deploy.replicas.load(SeqCst);
            assert!(replicas > 10 && replicas < 20, "{}", replicas);

            pods.tick_update_workloads(WORKER_SCALE_IN_SAMPLES * 2, &mut interval, 0.0)
                .await;
            replicas = deploy.replicas.load(SeqCst);
            assert_eq!(replicas, 1);
        });
    }
}
