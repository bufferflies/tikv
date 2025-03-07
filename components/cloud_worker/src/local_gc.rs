// Copyright 2025 TiKV Project Authors. Licensed under Apache-2.0.

use std::path::PathBuf;

use kvengine::ia::{
    gc::{IaGcConfig, IaGcRunner},
    manager::IaManager,
};
use tikv_util::{config::ReadableDuration, info};

use crate::common::Running;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(default)]
#[serde(rename_all = "kebab-case")]
pub struct LocalGcConfig {
    pub interval: ReadableDuration,
    pub ia: IaGcConfig,
}

impl Default for LocalGcConfig {
    fn default() -> Self {
        Self {
            interval: ReadableDuration::hours(1),
            ia: IaGcConfig::default(),
        }
    }
}

pub struct LocalGcRunner {
    config: LocalGcConfig,
    ia_runner: IaGcRunner,
}

impl LocalGcRunner {
    pub fn new(config: LocalGcConfig, ia_mgr: IaManager, meta_path: PathBuf) -> Self {
        let ia_runner = IaGcRunner::new(config.ia.clone(), ia_mgr.clone(), meta_path);
        Self { config, ia_runner }
    }

    pub fn run(&mut self, running: Running) {
        info!("worker gc runner start"; "config" => ?self.config);

        while running.get() {
            self.ia_runner.run(|_| false);

            std::thread::sleep(self.config.interval.0);
        }

        info!("worker gc runner stopped");
    }
}
