// Copyright 2024 TiKV Project Authors. Licensed under Apache-2.0.

//! Helpers to start/stop tikv-server & tikv-worker binary for tests.

use std::{
    collections::HashMap,
    fs,
    path::PathBuf,
    process,
    process::{Command, ExitStatus},
    sync::Arc,
    time::Duration,
};

use dashmap::DashMap;
use hyper::Method;
use nix::sys::signal::Signal;
use rand::prelude::*;
use security::{RestfulClient, SecurityConfig, SecurityManager};
use tikv::config::TikvConfig;
use tikv_util::{box_err, info, time::Instant, warn};

use crate::{
    new_test_config,
    tidb::{check_binary, send_signal_to_child},
    try_wait_result_async,
};

pub struct TikvServers {
    tag: String,
    bin_path: PathBuf,
    data_path: PathBuf,
    working_path: PathBuf,
    nodes_count: usize,
    memory_capacity_ratio: f64,
    security_mgr: Arc<SecurityManager>,
    confs: DashMap<u16 /* node_id */, TikvConfig>,
    children: DashMap<u16 /* node_id */, process::Child>,
}

impl TikvServers {
    pub fn new(
        tag: String,
        bin_path: PathBuf,
        data_path: PathBuf,
        working_path: PathBuf,
        nodes_count: usize,
        memory_capacity_ratio: f64,
        security_conf: &SecurityConfig,
    ) -> Self {
        check_binary("tikv_server", &bin_path);
        let security_mgr = Arc::new(SecurityManager::new(security_conf).unwrap());
        Self {
            tag,
            bin_path,
            data_path,
            working_path,
            nodes_count,
            memory_capacity_ratio,
            security_mgr,
            confs: Default::default(),
            children: Default::default(),
        }
    }

    pub fn tag(&self) -> &str {
        &self.tag
    }

    pub fn configs(&self) -> HashMap<u16 /* node_id */, TikvConfig> {
        self.confs
            .iter()
            .map(|r| (*r.key(), r.value().clone()))
            .collect()
    }

    pub fn start_node<F>(&self, node_id: u16, update_conf: F)
    where
        F: Fn(u16, &mut TikvConfig),
    {
        let mut config = if let Some((_, config)) = self.confs.remove(&node_id) {
            config
        } else {
            new_test_config(
                &self.data_path,
                node_id,
                self.nodes_count,
                self.memory_capacity_ratio,
            )
        };
        update_conf(node_id, &mut config);
        self.confs.insert(node_id, config.clone());

        fs::create_dir_all(&config.storage.data_dir).unwrap();

        let log_file = self.working_path.join(format!("tikv-server-{node_id}.log"));
        let config_file = self
            .working_path
            .join(format!("tikv-server-{node_id}.toml"));
        let toml = toml::to_string(&config).unwrap();
        fs::write(&config_file, toml).unwrap();

        let mut cmd = Command::new(&self.bin_path);
        cmd.arg(format!("--config={}", config_file.display()))
            .arg(format!("--log-file={}", log_file.display()));
        info!("tikv-server: start"; "cmd" => ?cmd);
        let child = cmd.spawn().unwrap();
        let old = self.children.insert(node_id, child);
        assert!(old.is_none(), "tikv-server {} already started", node_id);
    }

    pub fn stop_node(&self, node_id: u16, force: bool) -> ExitStatus {
        let (_, mut child) = self.children.remove(&node_id).unwrap();
        let signal = if force {
            Signal::SIGKILL
        } else {
            Signal::SIGTERM
        };
        info!("tikv-server: stopping"; "node_id" => node_id, "force" => force);
        let start_time = Instant::now_coarse();
        send_signal_to_child(&child, signal).unwrap();
        let exit_status = child.wait().unwrap();
        info!("tikv-server: stopped"; "node_id" => node_id, "exit_status" => ?exit_status, "takes" => ?start_time.saturating_elapsed());
        exit_status
    }

    pub fn random_restart_node(&self, rng: &mut ThreadRng) -> std::result::Result<u16, ExitStatus> {
        let node_id = *self.get_all_nodes().choose(rng).unwrap();
        let stop_sec = rng.gen_range(0..3);
        let force = rng.gen();

        let exit_status = self.stop_node(node_id, force);
        // The exit status of force kill must be 9.
        if !force && !exit_status.success() {
            return Err(exit_status);
        }

        std::thread::sleep(Duration::from_secs(stop_sec));

        self.start_node(node_id, |_, _| {});
        Ok(node_id)
    }

    pub fn get_all_nodes(&self) -> Vec<u16> {
        self.children.iter().map(|r| *r.key()).collect()
    }

    pub fn get_status_client(&self, node_id: u16) -> StatusClient {
        let endpoint = self.confs.get(&node_id).unwrap().server.status_addr.clone();
        StatusClient::new(endpoint, self.security_mgr.clone())
    }

    pub async fn must_healthy(&self, node_id: u16, timeout: Duration) {
        let cli = self.get_status_client(node_id);
        try_wait_result_async(
            || {
                let cli = cli.clone();
                Box::pin(async move {
                    match cli.status().await {
                        Ok(_) => Ok(()),
                        Err(err) => {
                            let err = Err(format!("tikv-server: check status failed: {err:?}"));
                            info!("{err:?}");
                            err
                        }
                    }
                })
            },
            timeout.as_secs() as usize,
        )
        .await
        .unwrap_or_else(|err| {
            panic!(
                "tikv-server: wait healthy timeout: node_id {}: {:?}",
                node_id, err
            )
        });
        info!("tikv-server is ready"; "node_id" => node_id);
    }

    pub async fn must_all_healthy(&self, timeout: Duration) {
        let all_nodes = self.get_all_nodes();
        for node_id in all_nodes {
            self.must_healthy(node_id, timeout).await;
        }
    }
}

const STATUS_PATH: &str = "status";

/// TidbControl provides access to HTTP APIs of TiDB, which are not included in
/// gRPC interface. It's also expected to act like the tool `tidb-ctl`.
#[derive(Clone)]
pub struct StatusClient {
    client: RestfulClient,
}

impl StatusClient {
    pub fn new(endpoint: String, security_mgr: Arc<SecurityManager>) -> Self {
        let client = RestfulClient::new("tikv-status-cli", vec![endpoint], security_mgr).unwrap();
        Self { client }
    }

    pub async fn status(&self) -> crate::tidb::Result<()> {
        let resp = self.client.request(STATUS_PATH, Method::GET, None).await;
        match resp {
            Ok(status) => {
                info!("tikv-status-cli: status: {:?}", status);
                Ok(())
            }
            Err(e) => Err(box_err!("tikv-status-cli: check status failed: {:?}", e)),
        }
    }
}

pub struct TikvWorkers {
    tag: String,
    bin_path: PathBuf,
    working_path: PathBuf,
    security_mgr: Arc<SecurityManager>,
    endpoints: DashMap<u16, String>,
    children: DashMap<u16 /* idx */, process::Child>,
}

impl TikvWorkers {
    pub fn new(
        tag: String,
        bin_path: PathBuf,
        working_path: PathBuf,
        security_conf: &SecurityConfig,
    ) -> Self {
        check_binary("tikv_worker", &bin_path);
        let security_mgr = Arc::new(SecurityManager::new(security_conf).unwrap());
        Self {
            tag,
            bin_path,
            working_path,
            security_mgr,
            endpoints: Default::default(),
            children: Default::default(),
        }
    }

    pub fn tag(&self) -> &str {
        &self.tag
    }

    pub fn start_all(&self, confs: &HashMap<u16 /* idx */, cloud_worker::Config>) {
        for (idx, conf) in confs {
            let log_file = self.working_path.join(format!("tikv-worker-{idx}.log"));
            let config_file = self.working_path.join(format!("tikv-worker-{idx}.toml"));
            let toml = toml::to_string(conf).unwrap();
            fs::write(&config_file, toml).unwrap();

            let mut cmd = Command::new(&self.bin_path);
            cmd.arg(format!("--config={}", config_file.display()))
                .arg(format!("--log-file={}", log_file.display()));
            info!("tikv-worker: start"; "cmd" => ?cmd);
            let child = cmd.spawn().unwrap();
            let old = self.children.insert(*idx, child);
            assert!(old.is_none(), "tikv-worker: {} already started", idx);

            let endpoint = conf.addr.clone();
            self.endpoints.insert(*idx, endpoint);
        }
    }

    pub fn stop_all(&self) {
        let all = self.get_all_indexes();
        for idx in all {
            let (_, mut child) = self.children.remove(&idx).unwrap();
            send_signal_to_child(&child, Signal::SIGTERM).unwrap();
            let exit_status = child.wait().unwrap();
            if exit_status.success() {
                info!("tikv-worker: stopped"; "idx" => idx);
            } else {
                warn!("tikv-worker: exit with error"; "idx" => idx, "exit_status" => ?exit_status);
            }
        }
    }

    pub async fn must_all_healthy(&self, timeout: Duration) {
        let all = self.get_all_indexes();
        for idx in all {
            self.must_healthy(idx, timeout).await;
        }
    }

    async fn must_healthy(&self, idx: u16, timeout: Duration) {
        let ep = self.endpoints.get(&idx).map(|r| r.value().clone()).unwrap();
        wait_tikv_worker_healthy(ep, self.security_mgr.clone(), timeout)
            .await
            .unwrap_or_else(|err| {
                panic!("tikv-worker: wait healthy timeout, idx {}: {:?}", idx, err)
            });
        info!("tikv-worker is ready"; "idx" => idx);
    }

    pub fn endpoints(&self) -> Vec<String> {
        self.endpoints.iter().map(|r| r.value().clone()).collect()
    }

    fn get_all_indexes(&self) -> Vec<u16> {
        self.children.iter().map(|r| *r.key()).collect()
    }
}

pub async fn wait_tikv_worker_healthy(
    endpoint: String,
    security_mgr: Arc<SecurityManager>,
    timeout: Duration,
) -> security::HttpResult<()> {
    let cli = RestfulClient::new(
        format!("tikv-worker-{endpoint}"),
        vec![endpoint],
        security_mgr,
    )?;
    try_wait_result_async(
        || {
            let cli = cli.clone();
            Box::pin(async move { cli.request("healthz", Method::GET, None).await.map(|_| ()) })
        },
        timeout.as_secs() as usize,
    )
    .await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_toml() {
        let tikv_configs = TikvConfig::default();
        toml::to_string(&tikv_configs).unwrap();

        let worker_configs = cloud_worker::Config::default();
        toml::to_string(&worker_configs).unwrap();
    }
}
