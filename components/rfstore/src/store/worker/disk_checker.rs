// Copyright 2025 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    fmt::{self, Display, Formatter},
    io::Write,
    path::PathBuf,
    sync::{
        mpsc::{self, Receiver as StdReceiver, RecvTimeoutError, Sender as StdSender},
        Arc,
    },
    thread::JoinHandle,
    time::Duration,
};

use crossbeam::channel::{bounded, Receiver, Sender};
use health_controller::types::LatencyInspector;
use tikv_util::{
    debug,
    time::Instant,
    warn,
    worker::{Builder, Runnable, Worker},
};

#[derive(Debug)]
pub enum Task {
    InspectLatency { inspector: LatencyInspector },
}

impl Display for Task {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match *self {
            Task::InspectLatency { .. } => write!(f, "InspectLatency"),
        }
    }
}

#[derive(Clone)]
/// A simple inspector to measure the latency of disk IO.
///
/// This is used to measure the latency of disk IO, which is used to determine
/// the health status of the TiKV server.
/// The inspector writes a file to the disk and measures the time it takes to
/// complete the write operation.
pub struct Runner {
    /// One or more target file paths used to probe individual disk paths.
    /// Each entry is the full path to a temporary file (inspect filename
    /// joined into the directory to inspect).
    _targets: Arc<Cleanup>,
    notifier: Sender<Task>,
    receiver: Receiver<Task>,
    bg_worker: Option<Worker>,
    /// Internal thread pool used to execute per-target inspections. Shared
    /// through an Arc so `Runner` remains cheap to clone.
    pool: Arc<InnerPool>,
}

/// Shared cleanup guard to ensure temp files are removed exactly once when the
/// last `Runner` reference is dropped.
struct Cleanup {
    targets: Vec<PathBuf>,
}

impl Drop for Cleanup {
    fn drop(&mut self) {
        for target in &self.targets {
            if let Err(e) = std::fs::remove_file(target) {
                warn!("remove disk latency inspector file failed"; "err" => ?e, "file" => ?target);
            }
        }
    }
}

struct InnerPool {
    cmd_senders: Vec<StdSender<StdSender<Option<Duration>>>>,
    handles: Vec<JoinHandle<()>>,
}

impl Drop for InnerPool {
    fn drop(&mut self) {
        // Dropping the senders will cause worker threads to exit their recv
        // loop. Join the threads to ensure clean shutdown.
        self.cmd_senders.clear();
        for h in self.handles.drain(..) {
            let _ = h.join();
        }
    }
}

impl Runner {
    /// The filename to write to the disk to measure the latency.
    const DISK_IO_LATENCY_INSPECT_FILENAME: &'static str = ".disk_latency_inspector.tmp";
    /// The content to write to the file to measure the latency.
    const DISK_IO_LATENCY_INSPECT_FLUSH_STR: &'static [u8] = b"inspect disk io latency";
    /// Timeout for one inspection.
    const DISK_IO_LATENCY_INSPECT_TIMEOUT: Duration = Duration::from_secs(5);

    #[inline]
    fn build(target_paths: Vec<PathBuf>) -> Self {
        // The disk check mechanism only cares about the latency of the most
        // recent request; older requests become stale and irrelevant. To avoid
        // unnecessary accumulation of multiple requests, we set a small
        // `capacity` for the disk check worker.
        let (notifier, receiver) = bounded(3);
        // Create an inner pool with one worker thread per target. Each
        // worker listens for a oneshot Sender<Option<Duration>> which it
        // uses to return the measurement result.
        let mut cmd_senders = Vec::with_capacity(target_paths.len());
        let mut handles = Vec::with_capacity(target_paths.len());
        for target in target_paths.clone() {
            let (cmd_tx, cmd_rx): (
                StdSender<StdSender<Option<Duration>>>,
                StdReceiver<StdSender<Option<Duration>>>,
            ) = mpsc::channel();
            let handle = std::thread::spawn(move || {
                // Worker loop: wait for a oneshot sender to perform the
                // inspection and return the result. If the channel is closed
                // we exit the loop and terminate the thread.
                while let Ok(resp_tx) = cmd_rx.recv() {
                    let res = (|| -> Option<Duration> {
                        let mut file = std::fs::OpenOptions::new()
                            .create(true)
                            .write(true)
                            .truncate(true)
                            .open(target.clone())
                            .ok()?;
                        let start = Instant::now();
                        file.write_all(Self::DISK_IO_LATENCY_INSPECT_FLUSH_STR)
                            .ok()?;
                        file.sync_all().ok()?;
                        Some(start.saturating_elapsed())
                    })();
                    // Ignore send errors (receiver might be dropped).
                    let _ = resp_tx.send(res);
                }
            });
            cmd_senders.push(cmd_tx);
            handles.push(handle);
        }

        let pool = InnerPool {
            cmd_senders,
            handles,
        };

        Self {
            _targets: Arc::new(Cleanup {
                targets: target_paths,
            }),
            notifier,
            receiver,
            // Use individual background thread for inspecting latencies to prevent mutual
            // interference between tasks.
            bg_worker: Some(Builder::new("disk_checker").thread_count(1).create()),
            pool: Arc::new(pool),
        }
    }

    #[inline]
    /// Create a new runner that inspects the given directory. For backwards
    /// compatibility this constructor accepts a single `inspect_dir` and will
    /// probe the standard temporary filename under that directory.
    pub fn new(inspect_dir: PathBuf) -> Self {
        Self::build(vec![
            inspect_dir.join(Self::DISK_IO_LATENCY_INSPECT_FILENAME),
        ])
    }

    /// Create a new runner that will inspect multiple directories concurrently.
    /// Each directory is probed by writing the standard temporary filename
    /// under the directory.
    pub fn new_multi(inspect_dirs: Vec<PathBuf>) -> Self {
        let targets = inspect_dirs
            .into_iter()
            .map(|d| d.join(Self::DISK_IO_LATENCY_INSPECT_FILENAME))
            .collect();
        Self::build(targets)
    }

    #[cfg(test)]
    #[inline]
    /// Only for test.
    /// Generate a dummy Runner.
    /// Only for test. Generate a dummy Runner that inspects the current
    /// directory.
    pub fn dummy() -> Self {
        Self::new_multi(vec![PathBuf::from("./")])
    }

    #[inline]
    pub fn bind_background_worker(&mut self, bg_worker: Worker) {
        if let Some(bg_worker) = self.bg_worker.take() {
            bg_worker.stop();
        }
        self.bg_worker = Some(bg_worker);
    }

    /// Inspect all configured targets (each target corresponds to a specific
    /// disk path). When a background `Worker` is bound, schedule per-target
    /// inspections on the worker's task queue; otherwise fall back to a
    /// sequential inspection. The function returns the slowest successful
    /// duration among the targets. If all targets fail, `None` is returned.
    fn inspect(&self) -> Option<Duration> {
        // Use the internal per-target worker threads to perform concurrent
        // inspections. For each worker we send a oneshot response sender and
        // receive its measurement.
        let n = self.pool.cmd_senders.len();
        let mut recvers = Vec::with_capacity(n);
        for sender in &self.pool.cmd_senders {
            let (resp_tx, resp_rx) = mpsc::channel();
            // If send fails the worker thread may have exited; treat as a
            // failed probe for that target.
            if sender.send(resp_tx).is_ok() {
                recvers.push(resp_rx);
            }
        }

        let mut max: Option<Duration> = None;
        for rx in recvers {
            match rx.recv_timeout(Self::DISK_IO_LATENCY_INSPECT_TIMEOUT) {
                Ok(Some(dur)) => {
                    max = Some(match max {
                        Some(prev) if prev > dur => prev,
                        _ => dur,
                    });
                }
                Err(RecvTimeoutError::Timeout) => {
                    max = Some(Self::DISK_IO_LATENCY_INSPECT_TIMEOUT);
                }
                _ => {}
            }
        }
        max
    }

    fn execute(&self) {
        if let Ok(task) = self.receiver.try_recv() {
            match task {
                Task::InspectLatency { mut inspector } => {
                    if let Some(latency) = self.inspect() {
                        inspector.record_process_duration(latency);
                        inspector.finish();
                    } else {
                        warn!("failed to inspect disk io latency");
                    }
                }
            }
        }
    }
}

impl Runnable for Runner {
    type Task = Task;

    fn run(&mut self, task: Task) {
        // Send the task to the limited capacity channel.
        if let Err(e) = self.notifier.try_send(task) {
            debug!("failed to send task to disk check bg_worker: {:?}", e);
        } else {
            let runner = self.clone();
            if let Some(bg_worker) = self.bg_worker.as_ref() {
                bg_worker.spawn_async_task(async move {
                    runner.execute();
                });
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use tikv_util::worker::Builder;

    use super::*;

    #[test]
    fn test_disk_check_runner() {
        let background_worker = Builder::new("disk-check-worker")
            .pending_capacity(256)
            .create();
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        let mut runner = Runner::dummy();
        runner.bind_background_worker(background_worker);
        // Validate the disk check runner.
        {
            let tx_1 = tx.clone();
            let inspector = LatencyInspector::new(
                1,
                Box::new(move |_, duration| {
                    let dur = duration.sum(true);
                    tx_1.send(dur).unwrap();
                }),
            );
            runner.run(Task::InspectLatency { inspector });
            let latency = rx.recv().unwrap();
            assert!(latency > Duration::from_secs(0));
        }
        // Invalid bg_worker and out of capacity
        {
            runner.bg_worker = None;
            for i in 2..=10 {
                let tx_2 = tx.clone();
                let inspector = LatencyInspector::new(
                    i as u64,
                    Box::new(move |_, duration| {
                        let dur = duration.sum(true);
                        tx_2.send(dur).unwrap();
                    }),
                );
                runner.run(Task::InspectLatency { inspector });
                rx.recv_timeout(Duration::from_secs(1)).unwrap_err();
            }
        }
        drop(runner);
    }
}
