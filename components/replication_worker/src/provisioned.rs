// Copyright 2025 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    fs,
    path::PathBuf,
    process::{Child, Command},
    sync::Arc,
};

use async_trait::async_trait;
use nix::{
    sys::signal::{kill, Signal},
    unistd::Pid,
};
use pd_client::PdClient;
use resolved_ts::Resolver;
use security::SecurityConfig;
use tikv_util::info;

use crate::{
    bootstrap, worker::new_keyspace_pd_client, KeyspaceService, KeyspaceStates,
    ReplicationWorkerConfig, Result,
};

pub(crate) struct KeyspaceProvisionedService {
    keyspace_id: u32,
    conf: ReplicationWorkerConfig,
    sec_conf: SecurityConfig,
    pub(crate) states: KeyspaceStates,
    pd_client: Option<Arc<dyn PdClient>>,
    resolver: Resolver,
}

impl KeyspaceProvisionedService {
    pub(crate) fn new(
        keyspace_id: u32,
        conf: &ReplicationWorkerConfig,
        sec_conf: &SecurityConfig,
        states: KeyspaceStates,
    ) -> Self {
        Self {
            keyspace_id,
            conf: conf.clone(),
            sec_conf: sec_conf.clone(),
            states,
            pd_client: None,
            resolver: Resolver::new(0),
        }
    }
}

#[async_trait]
impl KeyspaceService for KeyspaceProvisionedService {
    fn keyspace_id(&self) -> u32 {
        self.keyspace_id
    }

    async fn start(&mut self) -> Result<()> {
        let pd_url = self.states.pd_url.clone();
        let pd_client = new_keyspace_pd_client(pd_url, &self.sec_conf).await;
        bootstrap(
            pd_client.clone(),
            self.conf.merged_engine.merged_store_id,
            self.conf.advertise_addr.clone(),
        )
        .await?;
        self.pd_client = Some(pd_client);
        Ok(())
    }

    async fn destroy(&mut self) -> Result<()> {
        Ok(())
    }

    fn get_states(&self) -> &KeyspaceStates {
        &self.states
    }

    fn get_states_mut(&mut self) -> &mut KeyspaceStates {
        &mut self.states
    }

    fn get_pd_client(&self) -> Arc<dyn PdClient> {
        self.pd_client.as_ref().unwrap().clone()
    }

    fn get_resolver(&mut self) -> &mut Resolver {
        &mut self.resolver
    }
}

pub struct LocalProvider {
    pub keyspace_id: u32,
    pub data_dir: PathBuf,
    pd_bin_path: String,
    cdc_bin_path: String,
    tidb_bin_path: String,
    base_port: u16,
    pub pd_child: Option<Child>,
    pub cdc_child: Option<Child>,
    pub tidb_child: Option<Child>,
}

impl LocalProvider {
    pub fn new(keyspace_id: u32, data_dir: PathBuf, base_port: u16) -> Self {
        Self {
            keyspace_id,
            data_dir,
            pd_bin_path: std::env::var("PD_BIN").unwrap(),
            cdc_bin_path: std::env::var("CDC_BIN").unwrap(),
            tidb_bin_path: std::env::var("TIDB_BIN").unwrap_or_default(),
            base_port,
            pd_child: None,
            cdc_child: None,
            tidb_child: None,
        }
    }

    pub fn start(&mut self) {
        self.start_local_pd();
        self.start_local_cdc(&self.cdc_server_addr(), &self.pd_client_url());
        self.tidb_child = Some(self.start_local_tidb());
    }

    pub fn start_local_pd(&mut self) {
        let name = format!("pd-{}", self.keyspace_id);
        let data_dir = self.data_dir.join(&name);
        fs::create_dir_all(&data_dir).unwrap();
        let log_file = self.data_dir.join(format!("{}.log", name));
        let initial_cluster = format!("pd={}", self.local_pd_peer_url());
        let config_file = self.data_dir.join(format!("{}.toml", name));
        let pd_config = "[replication]\nmax-replicas = 1\n";
        fs::write(&config_file, pd_config).unwrap();
        let mut cmd = Command::new(&self.pd_bin_path);
        cmd.arg("--name=pd")
            .arg(format!("--data-dir={}", data_dir.display()))
            .arg(format!("--log-file={}", log_file.display()))
            .arg(format!("--client-urls={}", self.pd_client_url()))
            .arg(format!("--peer-urls={}", self.local_pd_peer_url()))
            .arg(format!("--initial-cluster={}", initial_cluster))
            .arg(format!("--config={}", config_file.display()));
        info!("start pd-server"; "cmd" => ?cmd);
        self.pd_child = Some(cmd.spawn().unwrap())
    }

    pub fn start_local_cdc(&mut self, cdc_addr: &str, pd_url: &str) {
        let keyspace_id = self.keyspace_id;
        let data_dir = self.data_dir.join(format!("cdc-{keyspace_id}"));
        let log_file = self.data_dir.join(format!("cdc-{keyspace_id}.log"));
        let mut cmd = Command::new(&self.cdc_bin_path);
        cmd.arg("server")
            .arg(format!("--addr={}", cdc_addr))
            .arg(format!("--log-file={}", log_file.display()))
            .arg(format!("--data-dir={}", data_dir.display()))
            .arg(format!("--pd={}", pd_url));
        info!("start cdc-server"; "cmd" => ?cmd);
        self.cdc_child = Some(cmd.spawn().unwrap());
    }

    fn start_local_tidb(&self) -> Child {
        let mut cmd = Command::new(&self.tidb_bin_path);
        let keyspace_id = self.keyspace_id;
        let data_dir = self.data_dir.join(format!("tidb-{keyspace_id}"));
        let log_file = self.data_dir.join(format!("tidb-{keyspace_id}.log"));
        let config_file = self.data_dir.join("tidb-SYSTEM.toml");
        let tidb_config = "keyspace-name = \"SYSTEM\"\n";
        fs::write(&config_file, tidb_config).unwrap();
        cmd.arg(format!("--P={}", self.local_tidb_port()))
            .arg(format!("--log-file={}", log_file.display()))
            .arg("--store=unistore")
            .arg(format!("--path={}", data_dir.display()))
            .arg(format!("--status={}", self.local_tidb_status_port()))
            .arg(format!("--config={}", config_file.display()));
        info!("start local tidb-server"; "cmd" => ?cmd);
        cmd.spawn().unwrap()
    }

    pub fn pd_client_url(&self) -> String {
        Self::local_proc_url(self.local_pd_client_port())
    }

    pub fn cdc_server_addr(&self) -> String {
        Self::local_proc_addr(self.local_cdc_server_port())
    }

    fn local_pd_peer_url(&self) -> String {
        Self::local_proc_url(self.local_pd_peer_port())
    }

    fn local_cdc_server_port(&self) -> u16 {
        self.base_port + 2000 + self.keyspace_id as u16
    }

    fn local_pd_peer_port(&self) -> u16 {
        self.base_port + 1000 + self.keyspace_id as u16
    }

    fn local_pd_client_port(&self) -> u16 {
        self.base_port + self.keyspace_id as u16
    }

    fn local_proc_addr(port: u16) -> String {
        format!("127.0.0.1:{}", port)
    }

    fn local_proc_url(port: u16) -> String {
        format!("http://{}", Self::local_proc_addr(port))
    }

    fn local_tidb_port(&self) -> u16 {
        self.base_port + 3000 + self.keyspace_id as u16
    }

    fn local_tidb_status_port(&self) -> u16 {
        self.base_port + 4000 + self.keyspace_id as u16
    }

    fn kill_process(&self, pid: i32) -> Result<()> {
        if pid > 0 {
            let pid = Pid::from_raw(pid);
            kill(pid, Signal::SIGTERM)?;
        }
        Ok(())
    }

    pub fn destroy(&mut self) -> Result<()> {
        info!("destroy local provider");
        if let Some(cdc) = self.cdc_child.take() {
            self.kill_process(cdc.id() as i32)?;
        }
        if let Some(pd) = self.pd_child.take() {
            self.kill_process(pd.id() as i32)?;
        }
        if let Some(tidb) = self.tidb_child.take() {
            self.kill_process(tidb.id() as i32)?;
        }
        Ok(())
    }

    pub fn restart_local_pd(&mut self) -> Result<()> {
        if let Some(pd) = self.pd_child.take() {
            self.kill_process(pd.id() as i32)?;
            std::thread::sleep(std::time::Duration::from_secs(1));
        }
        self.start_local_pd();
        std::thread::sleep(std::time::Duration::from_secs(1));
        Ok(())
    }
}
