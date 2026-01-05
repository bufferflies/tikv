use health_controller::space_usage::DiskUsageChecker;
use tikv_util::sys::disk;

#[test]
fn test_disk_usage_checker() {
    let kvdb_path = "/tmp/tikv-kvdb".to_owned();
    let raft_path = "/tmp/tikv-raft".to_owned();
    let raft_spill_path = "/tmp/tikv-raft/spill".to_owned();

    // Case 1: mock the kvdb and raft engine are not separated.
    fail::cfg("mock_disk_space_stats", "return(10000,5000)").unwrap();
    let disk_usage_checker = DiskUsageChecker::new(
        kvdb_path.clone(),
        raft_path.clone(),
        Some(raft_spill_path.clone()),
        false,
        true,
        false,
        100,
        100,
        1000,
    );
    let (disk_status, kvdb_status, raft_status, ..) = disk_usage_checker.inspect(4000, 1000);
    assert_eq!(disk_status, disk::DiskUsage::AlreadyFull);
    assert_eq!(kvdb_status, disk::DiskUsage::AlreadyFull);
    assert_eq!(raft_status, disk::DiskUsage::Normal);

    let disk_usage_checker = DiskUsageChecker::new(
        kvdb_path.clone(),
        raft_path.clone(),
        Some(raft_spill_path.clone()),
        false,
        true,
        false,
        100,
        100,
        4100,
    );
    let (disk_status, kvdb_status, raft_status, ..) = disk_usage_checker.inspect(4000, 1000);
    assert_eq!(raft_status, disk::DiskUsage::Normal);
    assert_eq!(kvdb_status, disk::DiskUsage::AlmostFull);
    assert_eq!(disk_status, disk::DiskUsage::AlmostFull);
    let (disk_status, kvdb_status, raft_status, ..) = disk_usage_checker.inspect(3999, 1000);
    assert_eq!(raft_status, disk::DiskUsage::Normal);
    assert_eq!(kvdb_status, disk::DiskUsage::Normal);
    assert_eq!(disk_status, disk::DiskUsage::Normal);
    fail::remove("mock_disk_space_stats");

    // Case 2: mock the kvdb and raft engine are separated.
    fail::cfg(
            "mock_disk_space_stats",
            "1*return(500,200)->1*return(5000,2000)->1*return(500,200)->1*return(5000,2000)->1*return(500,200)->1*return(5000,2000)",
        )
        .unwrap();
    let disk_usage_checker = DiskUsageChecker::new(
        kvdb_path.clone(),
        raft_path.clone(),
        Some(raft_spill_path.clone()),
        true,
        true,
        false,
        100,
        100,
        6000,
    );
    let (disk_status, kvdb_status, raft_status, ..) = disk_usage_checker.inspect(4000, 450);
    assert_eq!(raft_status, disk::DiskUsage::AlreadyFull);
    assert_eq!(kvdb_status, disk::DiskUsage::Normal);
    assert_eq!(disk_status, disk::DiskUsage::AlreadyFull);
    let (disk_status, kvdb_status, raft_status, ..) = disk_usage_checker.inspect(4000, 400);
    assert_eq!(raft_status, disk::DiskUsage::AlmostFull);
    assert_eq!(kvdb_status, disk::DiskUsage::Normal);
    assert_eq!(disk_status, disk::DiskUsage::AlmostFull);
    let (disk_status, kvdb_status, raft_status, ..) = disk_usage_checker.inspect(4000, 399);
    assert_eq!(raft_status, disk::DiskUsage::Normal);
    assert_eq!(kvdb_status, disk::DiskUsage::Normal);
    assert_eq!(disk_status, disk::DiskUsage::Normal);
    fail::remove("mock_disk_space_stats");

    fail::cfg(
            "mock_disk_space_stats",
            "1*return(500,200)->1*return(5000,2000)->1*return(500,200)->1*return(5000,2000)->1*return(500,200)->1*return(5000,2000)",
        )
        .unwrap();
    let disk_usage_checker = DiskUsageChecker::new(
        kvdb_path.clone(),
        raft_path.clone(),
        Some(raft_spill_path.clone()),
        true,
        false,
        false,
        100,
        100,
        6000,
    );
    let (disk_status, kvdb_status, raft_status, ..) = disk_usage_checker.inspect(4000, 450);
    assert_eq!(raft_status, disk::DiskUsage::Normal);
    assert_eq!(kvdb_status, disk::DiskUsage::Normal);
    assert_eq!(disk_status, disk::DiskUsage::Normal);
    let (disk_status, kvdb_status, raft_status, ..) = disk_usage_checker.inspect(4000, 500);
    assert_eq!(raft_status, disk::DiskUsage::Normal);
    assert_eq!(kvdb_status, disk::DiskUsage::Normal);
    assert_eq!(disk_status, disk::DiskUsage::Normal);
    let (disk_status, kvdb_status, raft_status, ..) = disk_usage_checker.inspect(4900, 500);
    assert_eq!(raft_status, disk::DiskUsage::Normal);
    assert_eq!(kvdb_status, disk::DiskUsage::AlmostFull);
    assert_eq!(disk_status, disk::DiskUsage::AlmostFull);
    fail::remove("mock_disk_space_stats");

    // Case 3: mock the kvdb and raft engine are separated and the auxiliary
    // directory of raft engine is separated from the main directory of
    // raft.
    fail::cfg(
        "mock_disk_space_stats",
        "1*return(500,200)->1*return(100,20)->1*return(5000,2000)",
    )
    .unwrap();
    let disk_usage_checker = DiskUsageChecker::new(
        kvdb_path.clone(),
        raft_path.clone(),
        Some(raft_spill_path.clone()),
        true,
        true,
        true,
        100,
        100,
        6000,
    );
    let (disk_status, kvdb_status, raft_status, ..) = disk_usage_checker.inspect(4000, 450);
    assert_eq!(raft_status, disk::DiskUsage::Normal);
    assert_eq!(kvdb_status, disk::DiskUsage::Normal);
    assert_eq!(disk_status, disk::DiskUsage::Normal);
    fail::remove("mock_disk_space_stats");
}
