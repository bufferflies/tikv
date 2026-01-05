use std::{
    cmp,
    sync::atomic::{AtomicU64, Ordering},
};

use tikv_util::{error, sys::disk, time::Duration};

/// Collecting storage stats.
/// Uses the same interval of reporing store heartbeat for it to reduce
/// frequently accessing statistics on engines.
static DEFAULT_STORAGE_STATS_INTERVAL: AtomicU64 =
    AtomicU64::new(Duration::from_secs(10).as_millis() as u64);

// Only for tests.
pub fn reset_update_storage_stats_interval(duration: Duration) {
    DEFAULT_STORAGE_STATS_INTERVAL.store(duration.as_millis() as u64, Ordering::Relaxed);
}

pub fn get_update_storage_stats_interval() -> Duration {
    Duration::from_millis(DEFAULT_STORAGE_STATS_INTERVAL.load(Ordering::Relaxed))
}

fn calculate_disk_usage(a: disk::DiskUsage, b: disk::DiskUsage) -> disk::DiskUsage {
    match (a, b) {
        (disk::DiskUsage::AlreadyFull, _) => disk::DiskUsage::AlreadyFull,
        (_, disk::DiskUsage::AlreadyFull) => disk::DiskUsage::AlreadyFull,
        (disk::DiskUsage::AlmostFull, _) => disk::DiskUsage::AlmostFull,
        (_, disk::DiskUsage::AlmostFull) => disk::DiskUsage::AlmostFull,
        (disk::DiskUsage::Normal, disk::DiskUsage::Normal) => disk::DiskUsage::Normal,
    }
}

/// A checker to inspect the disk usage of kv engine and raft engine.
/// The caller should call `inspect` periodically to get the disk usage status
/// manually.
#[derive(Clone)]
pub struct DiskUsageChecker {
    /// The path of kv engine.
    kvdb_path: String,
    /// The path of raft engine.
    raft_path: String,
    /// The path of auxiliary directory of raft engine if specified.
    raft_auxiliary_path: Option<String>,
    /// Whether the main directory of raft engine is separated from kv engine.
    separated_raft_mount_path: bool,
    /// Whether the auxiliary directory of raft engine is separated from kv
    /// engine.
    separated_raft_auxiliary_mount_path: bool,
    /// Whether the auxiliary directory of raft engine is both separated from
    /// the main directory of raft engine and kv engine.
    separated_raft_auxiliary_and_kvdb_mount_path: bool,
    /// The threshold of disk usage of kv engine to trigger the almost full
    /// status.
    kvdb_almost_full_thd: u64,
    /// The threshold of disk usage of raft engine to trigger the almost full
    /// status.
    raft_almost_full_thd: u64,
    /// The specified disk capacity for the whole disk.
    config_disk_capacity: u64,
}

impl DiskUsageChecker {
    pub fn new(
        kvdb_path: String,
        raft_path: String,
        raft_auxiliary_path: Option<String>,
        separated_raft_mount_path: bool,
        separated_raft_auxiliary_mount_path: bool,
        separated_raft_auxiliary_and_kvdb_mount_path: bool,
        kvdb_almost_full_thd: u64,
        raft_almost_full_thd: u64,
        config_disk_capacity: u64,
    ) -> Self {
        DiskUsageChecker {
            kvdb_path,
            raft_path,
            raft_auxiliary_path,
            separated_raft_mount_path,
            separated_raft_auxiliary_mount_path,
            separated_raft_auxiliary_and_kvdb_mount_path,
            kvdb_almost_full_thd,
            raft_almost_full_thd,
            config_disk_capacity,
        }
    }

    /// Inspect the disk usage of kv engine and raft engine.
    /// The `kvdb_used_size` is the used size of kv engine, and the
    /// `raft_used_size` is the used size of raft engine.
    ///
    /// Returns the disk usage status of the whole disk, kv engine and raft
    /// engine, the whole disk capacity and available size.
    pub fn inspect(
        &self,
        kvdb_used_size: u64,
        raft_used_size: u64,
    ) -> (
        disk::DiskUsage, // whole disk status
        disk::DiskUsage, // kvdb disk status
        disk::DiskUsage, // raft disk status
        u64,             // whole capacity
        u64,             // whole available
    ) {
        // By default, the almost full threshold of kv engine is half of the
        // configured value.
        let kvdb_already_full_thd = self.kvdb_almost_full_thd / 2;
        let raft_already_full_thd = self.raft_almost_full_thd / 2;
        // Check the disk space of raft engine.
        let raft_disk_status = {
            if !self.separated_raft_mount_path || self.raft_almost_full_thd == 0 {
                disk::DiskUsage::Normal
            } else {
                let (raft_disk_cap, raft_disk_avail) = match disk::get_disk_space_stats(
                    &self.raft_path,
                ) {
                    Err(e) => {
                        error!(
                            "get disk stat for raft engine failed";
                            "raft_engine_path" => &self.raft_path,
                            "err" => ?e
                        );
                        return (
                            disk::DiskUsage::Normal,
                            disk::DiskUsage::Normal,
                            disk::DiskUsage::Normal,
                            0,
                            0,
                        );
                    }
                    Ok((cap, avail)) => {
                        if !self.separated_raft_auxiliary_mount_path {
                            // If the auxiliary directory of raft engine is not separated from
                            // kv engine, returns u64::MAX to indicate that the disk space of
                            // the raft engine should not be checked.
                            (u64::MAX, u64::MAX)
                        } else if self.separated_raft_auxiliary_and_kvdb_mount_path {
                            // If the auxiliary directory of raft engine is separated from kv
                            // engine and the main directory of
                            // raft engine, the disk space of
                            // the auxiliary directory should be
                            // checked.
                            assert!(self.raft_auxiliary_path.is_some());
                            let (auxiliary_disk_cap, auxiliary_disk_avail) =
                                match disk::get_disk_space_stats(
                                    self.raft_auxiliary_path.as_ref().unwrap(),
                                ) {
                                    Err(e) => {
                                        error!(
                                            "get auxiliary disk stat for raft engine failed";
                                            "raft_engine_path" => self.raft_auxiliary_path.as_ref().unwrap(),
                                            "err" => ?e
                                        );
                                        (0_u64, 0_u64)
                                    }
                                    Ok((total, avail)) => (total, avail),
                                };
                            (cap + auxiliary_disk_cap, avail + auxiliary_disk_avail)
                        } else {
                            (cap, avail)
                        }
                    }
                };
                let raft_disk_available = cmp::min(
                    raft_disk_cap
                        .checked_sub(raft_used_size)
                        .unwrap_or_default(),
                    raft_disk_avail,
                );
                if raft_disk_available <= raft_already_full_thd {
                    disk::DiskUsage::AlreadyFull
                } else if raft_disk_available <= self.raft_almost_full_thd {
                    disk::DiskUsage::AlmostFull
                } else {
                    disk::DiskUsage::Normal
                }
            }
        };
        // Check the disk space of kv engine.
        let (disk_cap, disk_avail) = match disk::get_disk_space_stats(&self.kvdb_path) {
            Err(e) => {
                error!(
                    "get disk stat for kv store failed";
                    "kv_path" => &self.kvdb_path,
                    "err" => ?e
                );
                return (
                    disk::DiskUsage::Normal,
                    disk::DiskUsage::Normal,
                    disk::DiskUsage::Normal,
                    0,
                    0,
                );
            }
            Ok((total, avail)) => (total, avail),
        };
        let capacity = if self.config_disk_capacity == 0 || disk_cap < self.config_disk_capacity {
            disk_cap
        } else {
            self.config_disk_capacity
        };
        let available = cmp::min(
            capacity.checked_sub(kvdb_used_size).unwrap_or_default(),
            disk_avail,
        );
        let cur_kv_disk_status = if available <= kvdb_already_full_thd {
            disk::DiskUsage::AlreadyFull
        } else if available <= self.kvdb_almost_full_thd {
            disk::DiskUsage::AlmostFull
        } else {
            disk::DiskUsage::Normal
        };
        let cur_disk_status = calculate_disk_usage(raft_disk_status, cur_kv_disk_status);
        (
            cur_disk_status,
            cur_kv_disk_status,
            raft_disk_status,
            capacity,
            available,
        )
    }
}
