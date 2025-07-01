// Copyright 2023 TiKV Project Authors. Licensed under Apache-2.0.

use std::{collections::HashMap, path::PathBuf, process, time::Duration};

use lazy_static::lazy_static;
use regex::Regex;
use tikv_util::{box_err, info, warn};
use tokio::process::Command;

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

/// Wrapper of go-tpc https://github.com/pingcap/go-tpc
pub struct Tpc {
    tag: String,
    path: PathBuf,
    arguments: HashMap<String, String>,
    workload: String,
}

impl Tpc {
    pub fn new(tag: String, path: PathBuf) -> Self {
        Self {
            tag,
            path,
            arguments: HashMap::new(),
            workload: "tpcc".to_owned(),
        }
    }

    fn set_arg(&mut self, key: impl Into<String>, val: impl Into<String>) -> &mut Self {
        self.arguments.insert(key.into(), val.into());
        self
    }

    pub fn host(&mut self, host: impl Into<String>) -> &mut Self {
        self.set_arg("host", host)
    }

    pub fn port(&mut self, port: u16) -> &mut Self {
        self.set_arg("port", port.to_string())
    }

    pub fn user(&mut self, user: impl Into<String>) -> &mut Self {
        self.set_arg("user", user)
    }

    pub fn password(&mut self, password: impl Into<String>) -> &mut Self {
        self.set_arg("password", password)
    }

    pub fn db(&mut self, db: impl Into<String>) -> &mut Self {
        self.set_arg("db", db)
    }

    pub fn warehouses(&mut self, warehouse: usize) -> &mut Self {
        self.set_arg("warehouses", warehouse.to_string())
    }

    pub fn max_procs(&mut self, max_procs: usize) -> &mut Self {
        self.set_arg("max-procs", max_procs.to_string())
    }

    pub fn tpcc(&mut self) -> &mut Self {
        self.workload = "tpcc".to_owned();
        self
    }

    pub fn tpch(&mut self) -> &mut Self {
        self.workload = "tpch".to_owned();
        self
    }

    pub async fn prepare(&self, threads: usize) -> Result<()> {
        let mut cmd = Command::new(&self.path);
        cmd.arg(&self.workload)
            .arg("prepare")
            .arg(format!("--threads={threads}"));
        for (k, v) in &self.arguments {
            cmd.arg(format!("--{}={}", k, v));
        }

        self.execute("prepare", cmd).await?;
        Ok(())
    }

    /// Return number of transactions executed.
    pub async fn run(
        &self,
        threads: usize,
        wait: bool,
        ignore_error: bool,
        duration: Duration,
    ) -> Result<usize> {
        let mut cmd = Command::new(&self.path);
        cmd.arg(&self.workload)
            .arg("run")
            .arg(format!("--threads={threads}"))
            .arg(format!("--time={}s", duration.as_secs()));
        if wait {
            cmd.arg("--wait");
        }
        if ignore_error {
            cmd.arg("--ignore-error");
        }
        for (k, v) in &self.arguments {
            cmd.arg(format!("--{}={}", k, v));
        }

        let output = self.execute("run", cmd).await?;
        let metrics = extract_metrics(&self.tag, &output.stdout)?;
        info!("{} tpcc metrics {:?}", self.tag, metrics);
        let tpmc = metrics
            .get("tpmC")
            .cloned()
            .ok_or_else(|| format!("{} tpmC not found in metrics", self.tag))?;
        Ok(get_trx_number_from_tpmc(tpmc, duration))
    }

    pub async fn check(&self) -> Result<()> {
        let mut cmd = Command::new(&self.path);
        cmd.arg(&self.workload).arg("check");
        for (k, v) in &self.arguments {
            cmd.arg(format!("--{}={}", k, v));
        }

        self.execute("check", cmd).await?;
        Ok(())
    }

    pub async fn cleanup(&self) -> Result<()> {
        let mut cmd = Command::new(&self.path);
        cmd.arg(&self.workload).arg("cleanup");
        for (k, v) in &self.arguments {
            cmd.arg(format!("--{}={}", k, v));
        }

        self.execute("cleanup", cmd).await?;
        Ok(())
    }

    async fn execute(&self, cmd_name: &str, mut cmd: Command) -> Result<process::Output> {
        // Log for both start & finished to help looking for relevant txn requests.
        info!("{} tpc {} start", self.tag, cmd_name; "cmd" => ?cmd);
        let output = cmd.output().await.unwrap();
        info!("{} tpc {} finished", self.tag, cmd_name; "output" => ?output);

        // Filter the warning "maxprocs: Leaving GOMAXPROCS=xx: CPU quota undefined".
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stderr = stderr
            .split('\n')
            .filter(|x| !x.is_empty() && !x.contains("maxprocs"))
            .collect::<Vec<_>>();

        let stdout = String::from_utf8_lossy(&output.stdout);
        // "[DATA ERROR]" exists in https://github.com/pingyu/go-tpc/tree/small-size only.
        let data_errs = stdout
            .split('\n')
            .filter(|x| x.contains("[DATA ERROR]"))
            .collect::<Vec<_>>();

        let stdout_failed = stdout
            .split('\n')
            .filter(|x| x.contains("fail"))
            .collect::<Vec<_>>();
        if !stdout_failed.is_empty() {
            warn!(
                "{} tpc {} meet error: {:?}",
                self.tag, cmd_name, stdout_failed
            );
        }

        if output.status.success() && data_errs.is_empty() && stderr.is_empty() {
            Ok(output)
        } else if !data_errs.is_empty() {
            #[cfg(feature = "debug-trace-txn-tasks")]
            {
                tikv::storage::txn::debug::dump_txn_tasks();
            }
            panic!("{} tpc {} data_errs: {:?}", self.tag, cmd_name, data_errs);
        } else {
            Err(box_err!(
                "{} tpc {} stderr: {:?}",
                self.tag,
                cmd_name,
                stderr
            ))
        }
    }
}

// E.g.: tpmC: 2289.9, tpmTotal: 4792.4, efficiency: 17806.3%
fn extract_metrics(tag: &str, stdout: &[u8]) -> Result<HashMap<String, f64>> {
    lazy_static! {
        // Ignore the "%" of efficiency.
        static ref TPMC_RE: Regex =
            Regex::new(r"([a-zA-Z]+): ([0-9.]+)").unwrap();
    }
    let stdout = String::from_utf8_lossy(stdout);
    let final_line = stdout
        .split('\n')
        .rev()
        .find(|x| x.starts_with("tpmC:"))
        .ok_or_else(|| format!("{} tpmC line not found in stdout {:?}", tag, stdout))?;
    let mut metrics: HashMap<String, f64> = HashMap::new();
    for cap in TPMC_RE.captures_iter(final_line) {
        let name = &cap[1].to_string();
        let value = &cap[2].parse::<f64>().map_err(|err| {
            format!(
                "{} parse float failed, error {}, stdout {:?}",
                tag, err, stdout
            )
        })?;
        metrics.insert(name.clone(), *value);
    }
    Ok(metrics)
}

fn get_trx_number_from_tpmc(tpmc: f64, duration: Duration) -> usize {
    (tpmc * duration.as_secs() as f64 / 60.0).round() as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tpc_extract_metrics() {
        let stdout = b"
[Current] DELIVERY - Takes(s): 9.9, Count: 36, TPM: 218.3, Sum(ms): 4589.1, Avg(ms): 127.3, 50th(ms): 109.1, 90th(ms): 218.1, 95th(ms): 243.3, 99th(ms): 260.0, 99.9th(ms): 260.0, Max(ms): 260.0
[Current] NEW_ORDER - Takes(s): 9.9, Count: 380, TPM: 2293.3, Sum(ms): 20222.6, Avg(ms): 53.2, 50th(ms): 37.7, 90th(ms): 75.5, 95th(ms): 92.3, 99th(ms): 285.2, 99.9th(ms): 1140.9, Max(ms): 1140.9
[Current] ORDER_STATUS - Takes(s): 9.5, Count: 32, TPM: 203.0, Sum(ms): 727.4, Avg(ms): 22.7, 50th(ms): 12.6, 90th(ms): 46.1, 95th(ms): 52.4, 99th(ms): 184.5, 99.9th(ms): 184.5, Max(ms): 184.5
[Current] PAYMENT - Takes(s): 9.9, Count: 314, TPM: 1897.1, Sum(ms): 13467.9, Avg(ms): 42.9, 50th(ms): 35.7, 90th(ms): 65.0, 95th(ms): 79.7, 99th(ms): 201.3, 99.9th(ms): 838.9, Max(ms): 838.9
[Current] STOCK_LEVEL - Takes(s): 9.9, Count: 29, TPM: 176.3, Sum(ms): 748.7, Avg(ms): 25.9, 50th(ms): 17.8, 90th(ms): 30.4, 95th(ms): 32.5, 99th(ms): 201.3, 99.9th(ms): 201.3, Max(ms): 201.3
Finished
[Summary] DELIVERY - Takes(s): 9.9, Count: 36, TPM: 217.4, Sum(ms): 4589.1, Avg(ms): 127.3, 50th(ms): 109.1, 90th(ms): 218.1, 95th(ms): 243.3, 99th(ms): 260.0, 99.9th(ms): 260.0, Max(ms): 260.0
[Summary] NEW_ORDER - Takes(s): 10.0, Count: 381, TPM: 2289.9, Sum(ms): 20255.6, Avg(ms): 53.1, 50th(ms): 37.7, 90th(ms): 75.5, 95th(ms): 92.3, 99th(ms): 285.2, 99.9th(ms): 1140.9, Max(ms): 1140.9
[Summary] ORDER_STATUS - Takes(s): 9.5, Count: 32, TPM: 202.2, Sum(ms): 727.4, Avg(ms): 22.7, 50th(ms): 12.6, 90th(ms): 46.1, 95th(ms): 52.4, 99th(ms): 184.5, 99.9th(ms): 184.5, Max(ms): 184.5
[Summary] PAYMENT - Takes(s): 10.0, Count: 317, TPM: 1907.4, Sum(ms): 13636.9, Avg(ms): 43.0, 50th(ms): 35.7, 90th(ms): 62.9, 95th(ms): 79.7, 99th(ms): 201.3, 99.9th(ms): 838.9, Max(ms): 838.9
[Summary] STOCK_LEVEL - Takes(s): 9.9, Count: 29, TPM: 175.6, Sum(ms): 748.7, Avg(ms): 25.9, 50th(ms): 17.8, 90th(ms): 30.4, 95th(ms): 32.5, 99th(ms): 201.3, 99.9th(ms): 201.3, Max(ms): 201.3
tpmC: 2289.9, tpmTotal: 4792.4, efficiency: 17806.3%
";
        let metrics = extract_metrics("test", stdout).unwrap();
        assert_eq!(metrics.get("tpmC"), Some(&2289.9));
        assert_eq!(metrics.get("tpmTotal"), Some(&4792.4));
        assert_eq!(metrics.get("efficiency"), Some(&17806.3));
    }

    #[test]
    fn test_tpc_get_trx_number_from_tpmc() {
        let tpmc = 2289.9;
        let duration = Duration::from_secs(10);
        let trx_number = get_trx_number_from_tpmc(tpmc, duration);
        assert_eq!(trx_number, 382);
    }
}
