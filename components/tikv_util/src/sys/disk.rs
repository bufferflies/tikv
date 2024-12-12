// Copyright 2021 TiKV Project Authors. Licensed under Apache-2.0.
use std::{
    io,
    path::Path,
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

pub fn get_disk_capacity(dir: &Path) -> io::Result<u64> {
    use nix::NixPath;
    use sysinfo::{DiskExt, RefreshKind, System, SystemExt};

    lazy_static::lazy_static! {
        static ref SYS_INFO: System = System::new_with_specifics(RefreshKind::new().with_disks_list());
    }

    // find the mounted disk of the data dir.
    let mut data_disk = None;
    let mut mount_point_len = 0;
    for disk in SYS_INFO.disks() {
        let mp = disk.mount_point();
        if dir.starts_with(mp) && mp.len() > mount_point_len {
            data_disk = Some(disk);
            mount_point_len = mp.len();
        }
    }
    data_disk.map(|disk| disk.total_space()).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::NotFound,
            format!("Unable to find disk for dir: {dir:?}"),
        )
    })
}
