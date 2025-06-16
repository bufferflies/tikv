// Copyright 2025 TiKV Project Authors. Licensed under Apache-2.0.

use std::{rc::Rc, sync::Arc, time::Duration};

use futures::executor::block_on;
use schema::schema::{StorageClass, StorageClassSpec, StorageClassTransitRule};
use test_util::init_log_for_test;
use txn_types::TimeStamp;

use crate::{
    ia::{
        ia_auto_file::{IaAutoFile, TransitResult},
        manager::IaManager,
        util::IaManagerOptionsBuilder,
    },
    table::sstable::{BlockCache, SsTable},
    tests::{
        new_table, new_test_engine_opt,
        test_ia_file::{open_ia_file_from_sst, verify_table_by_scan, verify_table_by_scan_async},
    },
};

#[test]
fn test_ia_auto_file() {
    init_log_for_test();

    let file_id = 1;
    let (engine, _) = new_test_engine_opt(true, 1024, "");
    let mut saved_vals: Vec<Rc<Vec<u8>>> = Vec::new();
    let t = new_table(&engine, file_id, 0, 100, 1000, false, &mut saved_vals);
    // `disable_sync_read()` to check that `IaAutoFile::local_file` is available.
    let mgr = engine.new_ia_manager(
        IaManagerOptionsBuilder::default()
            .segment_size(4096)
            .disable_sync_read()
            .build()
            .unwrap(),
    );

    // Open IaAutoFile & SsTable.
    let sc_spec = StorageClassSpec {
        default_tier: StorageClass::Unspecified,
        transit_rules: vec![
            StorageClassTransitRule {
                tier: StorageClass::Ia,
                transit_after: Duration::from_secs(3),
            },
            StorageClassTransitRule {
                // To check that `IA` -> `Unspecified` is ignored.
                tier: StorageClass::Unspecified,
                transit_after: Duration::from_secs(6),
            },
        ],
    };
    let ts10 = TimeStamp::compose(10 * 1000, 0);
    let sst_auto = convert_local_sst_to_auto(&t, sc_spec, Some(ts10.into_inner()), mgr.clone());
    verify_table_by_scan(&engine.key_builder, &sst_auto, 0, 100, 1000);

    // Downcast.
    let ia_auto_f = sst_auto.try_get_auto_ia_file().unwrap();

    // Transit.
    let ts12 = TimeStamp::compose(12 * 1000, 0);
    let res = ia_auto_f.try_transit(ts12.into_inner());
    assert_eq!(res, TransitResult::NoChange);

    let ts13 = TimeStamp::compose(13 * 1000, 0);
    let res = ia_auto_f.try_transit(ts13.into_inner());
    assert_eq!(res, TransitResult::TransitedToIa);
    // verify_table_by_scan() will panic for "IaMgr(sync read disabled)".
    block_on(verify_table_by_scan_async(
        &engine.key_builder,
        &sst_auto,
        0,
        100,
        1000,
    ));

    // Transit from IA to unspecified will be ignored.
    let ts16 = TimeStamp::compose(16 * 1000, 0);
    let res = ia_auto_f.try_transit(ts16.into_inner());
    assert_eq!(res, TransitResult::NoChange);
    block_on(verify_table_by_scan_async(
        &engine.key_builder,
        &sst_auto,
        0,
        100,
        1000,
    ));
}

pub(crate) fn convert_local_sst_to_auto(
    t: &SsTable,
    sc_spec: StorageClassSpec,
    max_ts: Option<u64>,
    mgr: IaManager,
) -> SsTable {
    let ia_file = open_ia_file_from_sst(t, mgr);
    let ia_auto_file = IaAutoFile::new(
        ia_file,
        Some(t.file().clone()),
        max_ts.unwrap_or(t.max_ts),
        sc_spec,
    );
    SsTable::new(Arc::new(ia_auto_file), BlockCache::None, None).unwrap()
}
