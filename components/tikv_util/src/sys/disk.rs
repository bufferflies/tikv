// Copyright 2021 TiKV Project Authors. Licensed under Apache-2.0.
use std::{
    io,
    os::unix::fs::MetadataExt,
    path::{Path, PathBuf},
    sync::atomic::{AtomicI32, AtomicU64, Ordering},
};

use fail::fail_point;
pub use kvproto::disk_usage::DiskUsage;

// DISK_RESERVED_SPACE means if left space is less than this, tikv will
// turn to maintenance mode. There are another 2 value derived from this,
// 50% for a migration only mode and 20% for disk space holder size.
// Percent is not configurable, But if you want to change, please make sure
// the percent in both the init fs and store monitor are keep the same.
static DISK_RESERVED_SPACE: AtomicU64 = AtomicU64::new(0);
static RAFT_DISK_RESERVED_SPACE: AtomicU64 = AtomicU64::new(0);
static DISK_STATUS: AtomicI32 = AtomicI32::new(0);

pub fn set_disk_reserved_space(v: u64) {
    DISK_RESERVED_SPACE.store(v, Ordering::Release)
}

pub fn get_disk_reserved_space() -> u64 {
    DISK_RESERVED_SPACE.load(Ordering::Acquire)
}

pub fn set_raft_disk_reserved_space(v: u64) {
    RAFT_DISK_RESERVED_SPACE.store(v, Ordering::Release)
}

pub fn get_raft_disk_reserved_space() -> u64 {
    RAFT_DISK_RESERVED_SPACE.load(Ordering::Acquire)
}

pub fn set_disk_status(status: DiskUsage) {
    let v = match status {
        DiskUsage::Normal => 0,
        DiskUsage::AlmostFull => 1,
        DiskUsage::AlreadyFull => 2,
    };
    DISK_STATUS.store(v, Ordering::Release);
}

pub fn get_disk_status(_store_id: u64) -> DiskUsage {
    fail_point!("disk_almost_full_peer_1", _store_id == 1, |_| {
        DiskUsage::AlmostFull
    });
    fail_point!("disk_almost_full_peer_2", _store_id == 2, |_| {
        DiskUsage::AlmostFull
    });
    fail_point!("disk_almost_full_peer_3", _store_id == 3, |_| {
        DiskUsage::AlmostFull
    });
    fail_point!("disk_almost_full_peer_4", _store_id == 4, |_| {
        DiskUsage::AlmostFull
    });
    fail_point!("disk_almost_full_peer_5", _store_id == 5, |_| {
        DiskUsage::AlmostFull
    });
    fail_point!("disk_already_full_peer_1", _store_id == 1, |_| {
        DiskUsage::AlreadyFull
    });
    fail_point!("disk_already_full_peer_2", _store_id == 2, |_| {
        DiskUsage::AlreadyFull
    });
    fail_point!("disk_already_full_peer_3", _store_id == 3, |_| {
        DiskUsage::AlreadyFull
    });
    fail_point!("disk_already_full_peer_4", _store_id == 4, |_| {
        DiskUsage::AlreadyFull
    });
    fail_point!("disk_already_full_peer_5", _store_id == 5, |_| {
        DiskUsage::AlreadyFull
    });

    let s = DISK_STATUS.load(Ordering::Acquire);
    match s {
        0 => DiskUsage::Normal,
        1 => DiskUsage::AlmostFull,
        2 => DiskUsage::AlreadyFull,
        _ => panic!("Disk Status Value not meet expectations"),
    }
}

pub fn get_disk_capacity(dir: impl AsRef<Path>) -> io::Result<u64> {
    fs2::total_space(dir)
}

/// get_disk_stats returns (total, available) disk space in bytes.
pub fn get_disks_stats(dirs: &[PathBuf]) -> io::Result<(u64, u64)> {
    let mut available_spaces = vec![];

    // It is possible that multiple dirs are created on the same device, in this
    // case we should avoid counting the same device multiple times.
    let mut device_count_map = std::collections::HashMap::new();
    for dir in dirs {
        let m = dir.metadata()?;
        let device_id = m.dev();
        let count = device_count_map.entry(device_id).or_default();
        *count += 1;
    }
    let mut total_space = 0;
    for dir in dirs {
        let m = dir.metadata()?;
        // We only calculate the device once.
        if let Some(count) = device_count_map.remove(&m.dev()) {
            let stat = fs2::statvfs(dir)?;
            // The same device shared by multiple dirs, we should divide the space by the
            // count.
            total_space += stat.total_space();
            let available_space_per_dir = stat.available_space() / count as u64;
            for _ in 0..count {
                available_spaces.push(available_space_per_dir);
            }
        }
    }
    let num_dirs = dirs.len() as u64;
    let available_space = available_spaces.into_iter().min().unwrap_or_default() * num_dirs;
    Ok((total_space, available_space))
}
