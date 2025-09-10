// Copyright 2025 TiKV Project Authors. Licensed under Apache-2.0.

use std::{fs, path::PathBuf, process::Command, thread, time::Duration};

use anyhow::{Context, Result};
use lazy_static::lazy_static;
use serde_derive::Serialize;
use tikv_util::{error, future::paired_future_callback, info};

use crate::tidb::ConnParams;

const OUTPUT_DIR: &str = "output";
const DIFF_CONFIG_FILE: &str = "diff_config.toml";
const SYNC_DIFF_INSPECTOR_BIN: &str = "/usr/bin/sync_diff_inspector";

/// Wrapper of sync_diff_inspector https://github.com/pingcap/tidb-tools/tree/master/sync_diff_inspector
pub struct SyncDiffInspector {
    work_dir: PathBuf,
}

impl SyncDiffInspector {
    /// `tables`: Can use wildcard, e.g. ["test.*"]
    pub fn new(
        work_dir: PathBuf,
        upstream: ConnParams,
        downstream: ConnParams,
        tables: Vec<String>,
        use_snapshot: bool,
        skip_non_existing_table: bool,
    ) -> Self {
        fs::create_dir_all(&work_dir).unwrap();
        let config = Config {
            check_thread_count: 4,
            export_fix_sql: true,
            check_struct_only: false,
            skip_non_existing_table,
            task: Task {
                output_dir: work_dir.join(OUTPUT_DIR).to_string_lossy().to_string(),
                target_check_tables: tables,
                ..Default::default()
            },
            data_sources: DataSources {
                upstream: DbConfig::new(upstream, use_snapshot),
                downstream: DbConfig::new(downstream, use_snapshot),
            },
        };
        let toml = toml::to_string(&config).unwrap();
        let config_file = work_dir.join(DIFF_CONFIG_FILE);
        fs::write(config_file, toml).unwrap();

        Self { work_dir }
    }

    pub fn compare(&self) -> Result<CompareSummary> {
        let config_file = self.work_dir.join(DIFF_CONFIG_FILE);
        let mut cmd = Command::new(SYNC_DIFF_INSPECTOR_BIN);
        cmd.arg(format!("--config={}", config_file.display()));
        let output = cmd.output().expect("failed to execute sync_diff_inspector");

        let success = output.status.success();
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);

        if success {
            info!("sync_diff_inspector success"; "stdout" => stdout.as_ref(), "stderr" => &stderr.as_ref());
        } else {
            error!("sync_diff_inspector failed"; "stdout" => stdout.as_ref(), "stderr" => &stderr.as_ref());
        }

        let summary_file = self.work_dir.join(OUTPUT_DIR).join("summary.txt");
        let mut summary = CompareSummary {
            success,
            ..Default::default()
        };
        let content = fs::read_to_string(summary_file).unwrap();
        parse_compare_summary(&content, &mut summary).unwrap();

        Ok(summary)
    }
}

// Diff config:
#[rustfmt::skip]
/*
check-thread-count = 4
export-fix-sql = true
check-struct-only = false

[task]
    output-dir = "output"
    source-instances = ["upstream"]
    target-instance = "downstream"
    target-check-tables = ["db.*"]

[data-sources]
[data-sources.upstream]
    host = "10.2.8.125"
    port = 4444
    user = "test8.root"
    password = ""
    snapshot = "auto"

[data-sources.downstream]
    host = "10.2.8.125"
    port = 4455
    user = "root"
    password = ""
    snapshot = "auto"
 */
#[derive(Serialize)]
#[serde(rename_all = "kebab-case")]
struct Config {
    check_thread_count: u32,
    export_fix_sql: bool,
    check_struct_only: bool,
    skip_non_existing_table: bool,
    task: Task,
    data_sources: DataSources,
}

#[derive(Serialize)]
#[serde(rename_all = "kebab-case")]
struct Task {
    output_dir: String,
    source_instances: Vec<String>,
    target_instance: String,
    target_check_tables: Vec<String>,
}

impl Default for Task {
    fn default() -> Self {
        Self {
            output_dir: OUTPUT_DIR.into(),
            source_instances: vec!["upstream".to_string()],
            target_instance: "downstream".to_string(),
            target_check_tables: vec!["test.*".to_string()],
        }
    }
}

#[derive(Serialize)]
#[serde(rename_all = "kebab-case")]
struct DataSources {
    upstream: DbConfig,
    downstream: DbConfig,
}

#[derive(Serialize)]
#[serde(rename_all = "kebab-case")]
struct DbConfig {
    host: String,
    port: u16,
    user: String,
    password: String,
    snapshot: String,
}

impl DbConfig {
    fn new(params: ConnParams, use_snapshot: bool) -> Self {
        Self {
            host: params.host,
            port: params.port,
            user: params.user,
            password: params.password,
            snapshot: if use_snapshot {
                "auto".to_string()
            } else {
                "".to_string()
            },
        }
    }
}

// Result summary:
#[rustfmt::skip]
/*
Summary



Source Database



host = "10.2.8.125"
port = 4444
user = "test8.root"
snapshot = "460427039207849984"

Target Databases



host = "10.2.8.125"
port = 4455
user = "root"
snapshot = "460427040118800384"

Comparison Result



The table structure and data in following tables are equivalent

+--------------------+---------+-----------+
|       TABLE        | UPCOUNT | DOWNCOUNT |
+--------------------+---------+-----------+
| `sbtest`.`sbtest1` |    1000 |      1000 |
| `sbtest`.`sbtest2` |    1000 |      1000 |
+--------------------+---------+-----------+



Time Cost: 35.593733ms
Average Speed: 10.556569MB/s
 */
#[derive(Default, Debug, PartialEq)]
pub struct CompareSummary {
    pub success: bool,
    pub upstream_snapshot: Option<u64>,
    pub downstream_snapshot: Option<u64>,
}

const SOURCE_DATABASE_SECTION: &str = "Source Database";
const TARGET_DATABASES_SECTION: &str = "Target Databases";

fn parse_compare_summary(content: &str, summary: &mut CompareSummary) -> Result<()> {
    lazy_static! {
        static ref SNAPSHOT_RE: regex::Regex =
            regex::Regex::new(r#"snapshot\s*=\s*"(\d+)""#).unwrap();
    }
    let mut section = "";
    for line in content.lines() {
        let line = line.trim();
        match line {
            SOURCE_DATABASE_SECTION => section = SOURCE_DATABASE_SECTION,
            TARGET_DATABASES_SECTION => section = TARGET_DATABASES_SECTION,
            line if line.starts_with("snapshot = ") => {
                let caps = SNAPSHOT_RE.captures(line).context("captures")?;
                let m = caps.get(1).unwrap();
                let snapshot = m.as_str().parse::<u64>().context("parse::u64")?;
                match section {
                    SOURCE_DATABASE_SECTION => summary.upstream_snapshot = Some(snapshot),
                    TARGET_DATABASES_SECTION => summary.downstream_snapshot = Some(snapshot),
                    _ => {}
                }
            }
            _ => {}
        }
    }

    Ok(())
}

pub enum SyncDiffTask {
    Compare(Box<dyn FnOnce(Option<CompareSummary>) + Send>),
    Stop(Box<dyn FnOnce(()) + Send>),

    /// To skip the assertion of compare result UNTIL the
    /// `CompareSummary.upstream_snapshot` >= `SkipUntil.snapshot`.
    ///
    /// When changefeed is paused & resumed, TiCDC will sync from the latest
    /// checkpoint. In this case, changes will be replayed from the checkpoint
    /// to last actual sync point, and the upstream & downstream will not be
    /// identical during this process.
    SkipUntil {
        snapshot: u64,
        cb: Box<dyn FnOnce(()) + Send>,
    },
}

pub struct SyncDiffer {
    task_tx: tikv_util::mpsc::Sender<SyncDiffTask>,
}

impl SyncDiffer {
    pub fn new(
        work_dir: PathBuf,
        upstream: ConnParams,
        downstream: ConnParams,
        tables: Vec<String>,
        compare_interval: Duration,
    ) -> Self {
        fs::create_dir_all(&work_dir).unwrap();
        let (task_tx, task_rx) = tikv_util::mpsc::unbounded();
        thread::spawn(move || {
            let use_snapshot = true;
            // Tolerate the error when the "create table" has not been synced.
            let skip_non_existing_table = true;
            let sync_diff_inspector = SyncDiffInspector::new(
                work_dir,
                upstream,
                downstream,
                tables,
                use_snapshot,
                skip_non_existing_table,
            );

            let mut runner = SyncDiffRunner {
                sync_diff_inspector,
                task_rx,
                compare_interval,
                skip_until_snapshot: None,
            };
            runner.run();
        });

        Self { task_tx }
    }

    pub async fn compare(&self) -> Option<CompareSummary> {
        let (cb, fut) = paired_future_callback();
        self.task_tx
            .send(SyncDiffTask::Compare(Box::new(cb)))
            .unwrap();
        fut.await.unwrap()
    }

    pub async fn stop(&self) {
        let (cb, fut) = paired_future_callback();
        self.task_tx.send(SyncDiffTask::Stop(Box::new(cb))).unwrap();
        fut.await.unwrap();
    }

    pub async fn skip_until(&self, snapshot: u64) {
        let (cb, fut) = paired_future_callback();
        self.task_tx
            .send(SyncDiffTask::SkipUntil {
                snapshot,
                cb: Box::new(cb),
            })
            .unwrap();
        fut.await.unwrap();
    }
}

struct SyncDiffRunner {
    sync_diff_inspector: SyncDiffInspector,
    task_rx: tikv_util::mpsc::Receiver<SyncDiffTask>,
    compare_interval: Duration,
    skip_until_snapshot: Option<u64>,
}

impl SyncDiffRunner {
    fn run(&mut self) {
        loop {
            if let Ok(task) = self.task_rx.recv_timeout(self.compare_interval) {
                match task {
                    SyncDiffTask::Compare(cb) => {
                        let summary = self.compare();
                        cb(summary);
                    }
                    SyncDiffTask::Stop(cb) => {
                        cb(());
                        return;
                    }
                    SyncDiffTask::SkipUntil { snapshot, cb } => {
                        info!("sync_diff_inspector: pause until {}", snapshot);
                        self.skip_until_snapshot = Some(snapshot);
                        cb(());
                    }
                }
            } else if let Some(summary) = self.compare() {
                info!("sync_diff_inspector compare"; "summary" => ?summary);
                assert!(summary.success);
            }
        }
    }

    /// Return `None` when the snapshot of compare result is skipped.
    fn compare(&mut self) -> Option<CompareSummary> {
        let summary = self.sync_diff_inspector.compare().unwrap();
        if let (Some(skip_until), Some(upstream_snapshot)) =
            (self.skip_until_snapshot, summary.upstream_snapshot)
        {
            if upstream_snapshot <= skip_until {
                info!("sync_diff_inspector compare skipped"; "summary" => ?summary, "skip_until" => skip_until);
                return None;
            } else {
                self.skip_until_snapshot = None;
            }
        }
        Some(summary)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_compare_summary() {
        #[rustfmt::skip]
        let content = r#"
Summary



Source Database



host = "10.2.8.125"
port = 4444
user = "test8.root"
snapshot = "460427039207849984"

Target Databases



host = "10.2.8.125"
port = 4455
user = "root"
snapshot = "460427040118800384"

Comparison Result



The table structure and data in following tables are equivalent

+--------------------+---------+-----------+
|       TABLE        | UPCOUNT | DOWNCOUNT |
+--------------------+---------+-----------+
| `sbtest`.`sbtest1` |    1000 |      1000 |
| `sbtest`.`sbtest2` |    1000 |      1000 |
+--------------------+---------+-----------+



Time Cost: 35.593733ms
Average Speed: 10.556569MB/s
"#;

        let mut summary = CompareSummary {
            success: true,
            ..Default::default()
        };
        parse_compare_summary(content, &mut summary).unwrap();
        let expect = CompareSummary {
            success: true,
            upstream_snapshot: Some(460427039207849984),
            downstream_snapshot: Some(460427040118800384),
        };
        assert_eq!(summary, expect);
    }
}
