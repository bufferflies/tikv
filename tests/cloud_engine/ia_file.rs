// Copyright 2024 TiKV Project Authors. Licensed under Apache-2.0.

use std::{assert_matches::assert_matches, path::PathBuf, time::Duration};

use bytes::{Buf, Bytes};
use kvengine::{
    dfs::{FileType, S3Fs},
    ia::{
        manager::{IaManager, FOOTERS_SUB_DIR, SEGMENTS_SUB_DIR},
        types::{FileSegmentData, FileSegmentIdent, FooterInfo},
        util::{
            test_util::verify_local_segments, IaCapacity, IaManagerOptionsBuilder, LocalFileStore,
            LocalStore,
        },
    },
    table::file::File,
};
use proptest::prelude::*;
use rand::prelude::*;
use rstest::rstest;
use test_cloud_server::oss::prepare_dfs;
use test_util::init_log_for_test;
use tikv_util::{debug, info};

const SEGMENT_SIZE: i64 = 10;
const FREQ_UPDATE_INTERVAL: Duration = Duration::from_secs(1);
const PREPARE_CONCURRENCY: usize = 2;

prop_compose! {
    fn arb_range_args(min: u64, max: u64)
        (start in min..max)
        (
            start in Just(start),
            end in start+1..=max,
        )
        -> (u64, u64)
    {
        (start, end)
    }
}

#[rstest]
#[case(IaCapacity::MemoryAndDiskCap(80, PathBuf::from("ia"), 720))]
#[case::memory(IaCapacity::MemoryCap(800))]
#[case::big_cap(IaCapacity::MemoryAndDiskCap(200, PathBuf::from("ia"), 1800))]
#[case::small_cap(IaCapacity::MemoryAndDiskCap(20, PathBuf::from("ia"), 180))]
fn test_read(#[case] mut ia_cap: IaCapacity) {
    init_log_for_test();

    const FILE_SIZE: u64 = 1024;

    let (temp_dir, mut oss, dfs_conf) = prepare_dfs("test");
    let temp_dir = temp_dir.path();

    let s3fs = S3Fs::new_from_config(dfs_conf);
    let _s3fs = s3fs.clone();

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .enable_all()
        .build()
        .unwrap();

    ia_cap.set_parent_dir(temp_dir.to_path_buf());
    let options = IaManagerOptionsBuilder::default()
        .capacity(ia_cap)
        .segment_size(SEGMENT_SIZE)
        .freq_update_interval(Duration::ZERO)
        .build()
        .unwrap();
    let (small_cap, main_cap) = (options.small_queue.cap, options.main_queue.cap);

    let rt = runtime.handle().clone();
    let (mgr, file_data, ia_file) = runtime.block_on(async move {
        let file_id = 42;
        let file_type = FileType::Sst;
        let file_data = {
            let mut rng = thread_rng();
            Bytes::from(
                (0..FILE_SIZE / 8)
                    .flat_map(|_| rng.gen::<u64>().to_le_bytes())
                    .collect::<Vec<_>>(),
            )
        };
        assert_eq!(file_data.len(), FILE_SIZE as usize);

        s3fs.put_object(
            s3fs.file_key(file_id, file_type),
            file_data.clone(),
            format!("{}.{}", file_id, file_type.suffix()),
        )
        .await
        .unwrap();

        let mgr = IaManager::new(options, s3fs.clone(), rt).await.unwrap();
        mgr.prepare_footers(&[(file_id, file_type)], PREPARE_CONCURRENCY)
            .await
            .unwrap();
        let ia_file = mgr.open_file(file_id, file_type).await.unwrap();

        let seg = ia_file.read_async(0, file_data.len()).await.unwrap();
        assert_eq!(seg, file_data);

        let mut buf = vec![0; file_data.len()];
        ia_file.read_at_async(&mut buf, 0).await.unwrap();
        assert_eq!(buf, file_data.chunk());

        (mgr, file_data, ia_file)
    });

    proptest!(|(
        (start_off, end_off) in arb_range_args(0, file_data.len() as u64)
    )| {
        debug!("test_read: start_off: {}, end_off: {}", start_off, end_off);
        let expected = file_data.slice(start_off as usize..end_off as usize);

        let seg = runtime.block_on(ia_file.read_async(start_off, (end_off-start_off) as usize)).unwrap();
        prop_assert_eq!(&seg, &expected);

        let mut buf = vec![0; (end_off-start_off) as usize];
        runtime.block_on(ia_file.read_at_async(&mut buf, start_off)).unwrap();
        prop_assert_eq!(buf, expected.chunk());
    });

    runtime.block_on(async {
        mgr.flush_tasks(Duration::from_secs(5)).await.unwrap();

        let segments = mgr.get_local_segments().await;
        verify_local_segments(&segments, small_cap, main_cap, Some(FILE_SIZE));
    });

    info!("cache hit rate: {}", mgr.cache_hit_rate());
    oss.shutdown();
}

#[test]
fn test_init() {
    init_log_for_test();

    let (temp_dir, mut oss, dfs_conf) = prepare_dfs("test");
    let temp_dir = temp_dir.path();

    let s3fs = S3Fs::new_from_config(dfs_conf);
    let _s3fs = s3fs.clone();

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .enable_all()
        .build()
        .unwrap();
    let rt = runtime.handle().clone();
    runtime.block_on(async move {
        let local_path = temp_dir.join("ia");
        let ia_cap = IaCapacity::MemoryAndDiskCap(0, local_path, 1000);
        let options = IaManagerOptionsBuilder::default()
            .capacity(ia_cap)
            .segment_size(SEGMENT_SIZE)
            .freq_update_interval(FREQ_UPDATE_INTERVAL)
            .build()
            .unwrap();
        {
            let mgr = IaManager::new(options.clone(), s3fs.clone(), rt.clone())
                .await
                .unwrap();

            for i in 1..10 {
                let file_id = i as u64;
                let file_type = FileType::Sst;
                let file_data = Bytes::from((0u8..100).collect::<Vec<_>>());
                s3fs.put_object(
                    s3fs.file_key(file_id, file_type),
                    file_data.clone(),
                    format!("{}.{}", file_id, file_type.suffix()),
                )
                .await
                .unwrap();
            }

            let files = (1..10).map(|i| (i, FileType::Sst)).collect::<Vec<_>>();
            mgr.prepare_footers(&files, PREPARE_CONCURRENCY)
                .await
                .unwrap();

            let ia1 = mgr.open_file(1, FileType::Sst).await.unwrap();
            let _ = ia1.read_async(5, 10).await.unwrap();
            let _ = ia1.read_async(5, 10).await.unwrap();

            let ia2 = mgr.open_file(2, FileType::Sst).await.unwrap();
            let _ = ia2.read_async(20, 20).await.unwrap();
            let _ = ia2.read_async(20, 20).await.unwrap();

            // To make sure that segments are written to local store.
            mgr.flush_tasks(Duration::from_secs(5)).await.unwrap();
        }

        {
            let mgr = IaManager::new(options, s3fs.clone(), rt).await.unwrap();

            let mut segments_ident = mgr.get_local_segments().await;
            segments_ident.sort_by(|(m_ident, ..), (n_ident, ..)| m_ident.cmp(n_ident));
            let expected_segments = [
                // file_id, start_off, end_off
                (1u64, 0u64, 10u64),
                (1, 10, 20),
                (2, 20, 30),
                (2, 30, 40),
            ]
            .iter()
            .map(|&(file_id, start_off, end_off)| FileSegmentIdent {
                file_id,
                start_off,
                end_off,
            })
            .collect::<Vec<_>>();
            assert_eq!(segments_ident.len(), expected_segments.len());
            for ((ident, segment, _), expected) in segments_ident
                .into_iter()
                .zip(expected_segments.into_iter())
            {
                assert_eq!(ident, expected);
                assert_matches!(segment, FileSegmentData::InStore);
            }

            // Open file to verify footers exist.
            for file_id in 1..10 {
                let _ = mgr.open_file(file_id, FileType::Sst).await.unwrap();
            }
        }
    });

    oss.shutdown();
}

#[test]
fn test_abnormal_local_file() {
    init_log_for_test();

    let (temp_dir, mut oss, dfs_conf) = prepare_dfs("test");
    let temp_dir = temp_dir.path();
    let local_path = temp_dir.join("ia");

    let s3fs = S3Fs::new_from_config(dfs_conf);
    let _s3fs = s3fs.clone();

    let runtime = tokio::runtime::Runtime::new().unwrap();
    let rt = runtime.handle().clone();
    runtime.block_on(async move {
        let file_id = 42;
        let file_type = FileType::Sst;
        let file_data = {
            let mut rng = thread_rng();
            let mut buf = vec![0u8; 64];
            rng.fill_bytes(buf.as_mut_slice());
            Bytes::from(buf)
        };
        s3fs.put_object(
            s3fs.file_key(file_id, file_type),
            file_data.clone(),
            format!("{}.{}", file_id, file_type.suffix()),
        )
        .await
        .unwrap();

        let ia_cap = IaCapacity::MemoryAndDiskCap(0, local_path.clone(), 100000);
        let options = IaManagerOptionsBuilder::default()
            .capacity(ia_cap)
            .segment_size(SEGMENT_SIZE)
            .freq_update_interval(FREQ_UPDATE_INTERVAL)
            .build()
            .unwrap();
        let mgr = IaManager::new(options, s3fs.clone(), rt).await.unwrap();

        {
            mgr.prepare_footers(&[(file_id, file_type)], PREPARE_CONCURRENCY)
                .await
                .unwrap();
            let ia_file = mgr.open_file(file_id, file_type).await.unwrap();
            let seg = ia_file.read_async(0, file_data.len()).await.unwrap();
            assert_eq!(seg, file_data);
        }

        // Remove the local footer & segment.
        {
            let footer_store = LocalFileStore::new(local_path.join(FOOTERS_SUB_DIR));
            footer_store
                .remove(file_id, &FooterInfo::local_filename(file_id))
                .await
                .unwrap();

            let main_store = LocalFileStore::new(local_path.join(SEGMENTS_SUB_DIR));
            main_store
                .remove(
                    file_id,
                    &FileSegmentIdent {
                        file_id,
                        start_off: 10,
                        end_off: 20,
                    }
                    .local_filename(),
                )
                .await
                .unwrap();
        }

        {
            // First open file failed due to local footer not found.
            mgr.open_file(file_id, file_type).await.unwrap_err();
            // Check another prepare can restore the lost footer.
            mgr.prepare_footers(&[(file_id, file_type)], PREPARE_CONCURRENCY)
                .await
                .unwrap();
            let ia_file = mgr.open_file(file_id, file_type).await.unwrap();

            // Read can handle local segment not found by retry to get from remote.
            let seg = ia_file.read_async(0, file_data.len()).await.unwrap();
            assert_eq!(seg, file_data);
        }
    });

    oss.shutdown();
}
