// Copyright 2021 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    collections::HashMap,
    env,
    iter::Iterator,
    ops::Deref,
    path::Path,
    rc::Rc,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Mutex,
    },
    thread,
    time::Duration,
    u64, vec,
};

use api_version::{api_v2::KEYSPACE_PREFIX_LEN, ApiV2};
use bytes::{Buf, Bytes};
use cloud_encryption::{EncryptionKey, MasterKey};
use file_system::IoRateLimiter;
use kvenginepb as pb;
use kvenginepb::{TxnFileRef, TxnFileRefs};
use protobuf::Message;
use rand::prelude::*;
use rstest::rstest;
use security::SecurityManager;
use table::columnar::{
    tests::{i_to_common_handle, merge_refs},
    Block, ColumnarFile, ColumnarReader, ColumnarRowTableReader,
};
use tempfile::TempDir;
use tidb_query_datatype::{codec::table::encode_row_key, expr::EvalContext};
use tikv_util::{mpsc, time::Instant};
use util::test_util::KeyBuilder;

use crate::{
    dfs::{FileType, InMemFs},
    limiter::{RegionLimiter, StoreLimiter},
    table::{
        columnar::{
            build_schema_file,
            tests::{build_table, new_schema, verify_with_ref_rows},
            ColumnarFilterReader, Schema, SchemaFile,
        },
        file::{File, InMemFile},
        memtable::CfTable,
        sstable::{L0Builder, L0Table, SsTable},
        ChecksumType, InnerKey, NoPrefixKey, TxnChunkBuilder, TxnCtx, TxnFile, TxnFileId,
        BIT_DELETE, OP_PUT,
    },
    *,
};

macro_rules! unwrap_or_return {
    ($e:expr, $m:expr) => {
        match $e {
            Ok(x) => x,
            Err(y) => {
                error!("{:?} {:?}", y, $m);
                return;
            }
        }
    };
}

const KEYSPACE_ID: u32 = 1;

const DEF_BLOCK_SIZE: usize = 4 << 10;
const DEF_MIN_BLOB_SIZE: u32 = 64;

// The shard get from TestEngine has been ingest once, so the update counter
// starts from NEW_DATA_UPDATE_COUNTER + 1.
const TEST_ENGINE_NEW_DATA_UPDATE_COUNTER: u64 = NEW_DATA_UPDATE_COUNTER + 1;

const TABLE_KEY_PREFIX: &str = "t_";

/// Wrap `Engine` to make sure that it will be closed after the test, and not
/// interfere with other tests.
struct TestEngine {
    engine: Engine,
    key_builder: KeyBuilder,
}

impl std::ops::Deref for TestEngine {
    type Target = Engine;

    fn deref(&self) -> &Self::Target {
        &self.engine
    }
}

impl Drop for TestEngine {
    fn drop(&mut self) {
        self.engine.close();
    }
}

impl TestEngine {
    fn inner_key_off(&self) -> usize {
        KEYSPACE_PREFIX_LEN * self.key_builder.get_enable_inner_key_off() as usize
    }

    fn key_builder(&self) -> &KeyBuilder {
        &self.key_builder
    }
}

fn new_test_engine() -> (TestEngine, mpsc::Sender<ApplyTask>) {
    new_test_engine_opt(false, DEF_BLOCK_SIZE, "")
}

fn new_test_engine_opt(
    enable_inner_key_off: bool,
    block_size: usize,
    key_prefix: &str,
) -> (TestEngine, mpsc::Sender<ApplyTask>) {
    let (listener_tx, listener_rx) = mpsc::unbounded();
    let tester = EngineTester::new(enable_inner_key_off, block_size);
    let meta_change_listener = Box::new(TestMetaChangeListener {
        sender: listener_tx,
    });
    let rate_limiter = Arc::new(IoRateLimiter::new_for_test());
    let store_limiter = Arc::new(StoreLimiter::dummy());
    let mut meta_iter = tester.clone();
    let engine = Engine::open(
        tester.fs.clone(),
        tester.opts.clone(),
        tester.config.clone(),
        &mut meta_iter,
        tester.clone(),
        tester.core.clone(),
        meta_change_listener,
        rate_limiter,
        store_limiter,
        None,
        MasterKey::new(&[1u8; 32]),
        Arc::new(SecurityManager::default()),
    )
    .unwrap();
    {
        let shard = engine.get_shard(1).unwrap();
        store_bool(&shard.active, true);
    }
    let (applier_tx, applier_rx) = mpsc::unbounded();
    let meta_listener = MetaListener::new(listener_rx, applier_tx.clone());
    thread::spawn(move || {
        meta_listener.run();
    });
    let applier = Applier::new(engine.clone(), applier_rx);
    thread::spawn(move || {
        applier.run();
    });
    let key_builder = KeyBuilder::new(KEYSPACE_ID, enable_inner_key_off, key_prefix);
    (
        TestEngine {
            engine,
            key_builder,
        },
        applier_tx,
    )
}

#[test]
fn test_engine() {
    ::test_util::init_log_for_test();
    let (engine, applier_tx) = new_test_engine();
    // FIXME(youjiali1995): split has bugs.
    //
    // let mut keys = vec![];
    // for i in &[1000, 3000, 6000, 9000] {
    // keys.push(i_to_key(*i,
    // engine.opts.blob_table_build_options.min_blob_size)); }
    // let mut splitter = Splitter::new(keys.clone(), applier_tx.clone());
    // let handle = thread::spawn(move || {
    // splitter.run();
    // });
    let (begin, end) = (0, 10000);
    load_data(
        begin,
        end,
        1,
        applier_tx,
        engine.opts.blob_table_build_options.min_blob_size,
    );
    // handle.join().unwrap();
    check_get(
        begin,
        end,
        2,
        &[0, 1, 2],
        &engine,
        true,
        None,
        engine.opts.blob_table_build_options.min_blob_size,
    );
    check_iterater(begin, end, &engine);
}

#[test]
fn test_destroy_range() {
    ::test_util::init_log_for_test();
    let (engine, applier_tx) = new_test_engine();
    let mem_table_count = engine.get_shard_stat(1).mem_table_count;
    load_data(
        10,
        50,
        1,
        applier_tx.clone(),
        engine.opts.blob_table_build_options.min_blob_size,
    );
    // Unsafe destroy keys [10, 30).
    for prefix in [10, 20] {
        let mut wb = WriteBatch::new(1, 0);
        let key = i_to_key(prefix, engine.opts.blob_table_build_options.min_blob_size);
        wb.set_property(DEL_PREFIXES_KEY, key[..key.len() - 1].as_bytes());
        write_data(wb, &applier_tx);
    }
    assert!(!engine.get_shard(1).unwrap().get_del_prefixes().is_empty());
    // Memtable is switched because it contains data covered by the delete-prefixes.
    let stats = engine.get_shard_stat(1);
    assert_eq!(
        stats.mem_table_count + stats.l0_table_count + stats.blob_table_count,
        mem_table_count + 1
    );
    let wait_for_destroying_range = || {
        for _ in 0..30 {
            let shard = engine.get_shard(1).unwrap();
            let data = shard.get_data();
            let del_prefixes = shard.get_del_prefixes();
            if del_prefixes.is_empty() {
                break;
            }
            if Shard::ready_to_destroy_range(&del_prefixes, &data) {
                engine.trigger_compact(shard.id_ver());
            }
            thread::sleep(Duration::from_millis(100));
        }
        let shard = engine.get_shard(1).unwrap();
        let length = shard
            .get_property(DEL_PREFIXES_KEY)
            .map(|v| v.len())
            .unwrap_or_default();
        // Delete-prefixes is cleaned up.
        assert_eq!(length, 0);
    };
    wait_for_destroying_range();
    // After destroying range, key [10, 30) should be removed.
    check_get(
        10,
        30,
        2,
        &[0, 1, 2],
        &engine,
        false,
        None,
        engine.opts.blob_table_build_options.min_blob_size,
    );
    check_get(
        30,
        50,
        2,
        &[0, 1, 2],
        &engine,
        true,
        None,
        engine.opts.blob_table_build_options.min_blob_size,
    );
    check_iterater(30, 50, &engine);

    // Trigger L0 compaction.
    for i in 1..=10 {
        load_data(
            50 + (i - 1) * 10,
            50 + i * 10,
            1,
            applier_tx.clone(),
            engine.opts.blob_table_build_options.min_blob_size,
        );
        let mut wb = WriteBatch::new(1, 0);
        wb.set_switch_mem_table();
        write_data(wb, &applier_tx);
    }
    // Waiting for L0 compaction.
    for _ in 0..30 {
        if engine.get_shard_stat(1).l0_table_count == 0 {
            break;
        }
        thread::sleep(Duration::from_millis(100));
    }
    assert!(engine.get_shard_stat(1).l0_table_count < 10);
    // Unsafe destroy keys [100, 150).
    let mut wb = WriteBatch::new(1, 0);
    let key = i_to_key(100, engine.opts.blob_table_build_options.min_blob_size);
    wb.set_property(DEL_PREFIXES_KEY, key[..key.len() - 2].as_bytes());
    write_data(wb, &applier_tx);
    wait_for_destroying_range();
    check_get(
        100,
        150,
        2,
        &[0, 1, 2],
        &engine,
        false,
        None,
        engine.opts.blob_table_build_options.min_blob_size,
    );
    check_get(
        50,
        100,
        2,
        &[0, 1, 2],
        &engine,
        true,
        None,
        engine.opts.blob_table_build_options.min_blob_size,
    );

    // Clean all data.
    let mut wb = WriteBatch::new(1, 0);
    wb.set_property(DEL_PREFIXES_KEY, b"key");
    write_data(wb, &applier_tx);
    wait_for_destroying_range();
    check_get(
        10,
        150,
        2,
        &[0, 1, 2],
        &engine,
        false,
        None,
        engine.opts.blob_table_build_options.min_blob_size,
    );

    // No data exists and delete-prefixes can be cleaned too.
    let mut wb = WriteBatch::new(1, 0);
    wb.set_property(DEL_PREFIXES_KEY, b"key");
    write_data(wb, &applier_tx);
    wait_for_destroying_range();
}

#[test]
fn test_truncate_ts_request() {
    ::test_util::init_log_for_test();
    // TODO: disable compaction, otherwise this case would be unstable.
    let (engine, applier_tx) = new_test_engine();
    let version = 1000;
    load_data(
        10,
        50,
        version,
        applier_tx.clone(),
        engine.opts.blob_table_build_options.min_blob_size,
    );
    // In case auto truncate_ts compaction finishes too fast.
    engine.get_shard(1).unwrap().set_active(false);
    // truncate ts.
    let mut wb = WriteBatch::new(1, 0);
    let truncate_ts = TruncateTs::from(version + 10);
    wb.set_property(TRUNCATE_TS_KEY, truncate_ts.marshal().as_slice());
    write_data(wb, &applier_tx);
    assert_eq!(
        Some(truncate_ts),
        engine.get_shard(1).unwrap().get_truncate_ts()
    );
    assert_eq!(
        truncate_ts.marshal().as_slice(),
        engine
            .get_shard(1)
            .unwrap()
            .get_property(TRUNCATE_TS_KEY)
            .unwrap()
    );
    let mut cs = pb::ChangeSet::new();
    cs.set_shard_id(1);
    cs.set_shard_ver(1);
    cs.set_sequence(2);
    let tc = pb::TableChange::new();
    cs.set_truncate_ts(tc);
    cs.set_property_key(TRUNCATE_TS_KEY.to_owned());

    // truncated_ts is larger than current truncate ts, don't change it.
    let truncated_ts = TruncateTs::from(version + 20);
    cs.set_property_value(truncated_ts.marshal().to_vec());
    let ret = engine.apply_change_set(apply::ChangeSet::new(cs.clone()));
    ret.unwrap();
    assert_eq!(
        Some(truncate_ts),
        engine.get_shard(1).unwrap().get_truncate_ts()
    );

    // truncated_ts is equal than current truncate ts, remove truncate ts in shard.
    let truncated_ts = TruncateTs::from(version + 10);
    cs.set_sequence(3);
    cs.set_property_value(truncated_ts.marshal().to_vec());
    let ret = engine.apply_change_set(apply::ChangeSet::new(cs));
    ret.unwrap();
    assert_eq!(None, engine.get_shard(1).unwrap().get_truncate_ts());
}

#[test]
fn test_truncate_ts() {
    ::test_util::init_log_for_test();
    let (engine, applier_tx) = new_test_engine();

    // `tolerate_none`: set `true` when there is no data to be truncated.
    // In this scene, truncate ts may have been done before we check
    // `ShardData.truncate_ts`.
    let set_truncate_ts = |ts: u64, tolerate_none: bool| {
        let mut wb = WriteBatch::new(1, 0);
        let truncate_ts = TruncateTs::from(ts);
        wb.set_property(TRUNCATE_TS_KEY, truncate_ts.marshal().as_slice());
        write_data(wb, &applier_tx);

        let res = engine.get_shard(1).unwrap().get_truncate_ts();
        if res.is_none() && tolerate_none {
            return;
        }
        assert_eq!(Some(truncate_ts), res);

        let res_bin = engine
            .get_shard(1)
            .unwrap()
            .get_property(TRUNCATE_TS_KEY)
            .unwrap();
        if res_bin.is_empty() && tolerate_none {
            return;
        }
        assert_eq!(truncate_ts.marshal().as_slice(), res_bin);
    };

    let wait_for_truncate_ts = || {
        let ok = try_wait(
            || engine.get_shard(1).unwrap().get_truncate_ts().is_none(),
            10,
        );
        assert!(
            ok,
            "wait_for_truncate_ts timeout, shard:{:?}",
            engine.get_shard_stat(1)
        );
        let shard = engine.get_shard(1).unwrap();
        let length = shard
            .get_property(TRUNCATE_TS_KEY)
            .map(|v| v.len())
            .unwrap_or_default();
        assert_eq!(length, 0);
    };

    load_data(
        0,
        300,
        1000,
        applier_tx.clone(),
        engine.opts.blob_table_build_options.min_blob_size,
    );
    load_data(
        100,
        400,
        2000,
        applier_tx.clone(),
        engine.opts.blob_table_build_options.min_blob_size,
    );
    load_data(
        200,
        500,
        3000,
        applier_tx.clone(),
        engine.opts.blob_table_build_options.min_blob_size,
    );

    const ALL_CFS: &[usize] = &[0, 1, 2];
    const WRT_EXT_CFS: &[usize] = &[0, 2];

    {
        // No truncate.
        set_truncate_ts(3000, true);
        thread::sleep(Duration::from_secs(1));
        wait_for_truncate_ts();
        check_get(
            0,
            100,
            u64::MAX,
            ALL_CFS,
            &engine,
            true,
            Some(1000),
            engine.opts.blob_table_build_options.min_blob_size,
        );
        check_get(
            100,
            300,
            1000,
            ALL_CFS,
            &engine,
            true,
            Some(1000),
            engine.opts.blob_table_build_options.min_blob_size,
        );
        check_get(
            100,
            200,
            u64::MAX,
            ALL_CFS,
            &engine,
            true,
            Some(2000),
            engine.opts.blob_table_build_options.min_blob_size,
        );
        check_get(
            200,
            400,
            2000,
            ALL_CFS,
            &engine,
            true,
            Some(2000),
            engine.opts.blob_table_build_options.min_blob_size,
        );
        check_get(
            200,
            500,
            u64::MAX,
            ALL_CFS,
            &engine,
            true,
            Some(3000),
            engine.opts.blob_table_build_options.min_blob_size,
        );
    }

    for (i, truncate_ts) in [2999, 2000].iter().enumerate() {
        set_truncate_ts(*truncate_ts as u64, i > 0);
        wait_for_truncate_ts();
        check_get(
            0,
            100,
            u64::MAX,
            ALL_CFS,
            &engine,
            true,
            Some(1000),
            engine.opts.blob_table_build_options.min_blob_size,
        );
        check_get(
            100,
            300,
            1000,
            ALL_CFS,
            &engine,
            true,
            Some(1000),
            engine.opts.blob_table_build_options.min_blob_size,
        );
        check_get(
            100,
            400,
            u64::MAX,
            ALL_CFS,
            &engine,
            true,
            Some(2000),
            engine.opts.blob_table_build_options.min_blob_size,
        );
        check_get(
            400,
            500,
            u64::MAX,
            WRT_EXT_CFS,
            &engine,
            false,
            None,
            engine.opts.blob_table_build_options.min_blob_size,
        );
    }

    for (i, truncate_ts) in [1999, 1000].iter().enumerate() {
        set_truncate_ts(*truncate_ts as u64, i > 0);
        wait_for_truncate_ts();
        check_get(
            0,
            300,
            u64::MAX,
            ALL_CFS,
            &engine,
            true,
            Some(1000),
            engine.opts.blob_table_build_options.min_blob_size,
        );
        check_get(
            300,
            500,
            u64::MAX,
            WRT_EXT_CFS,
            &engine,
            false,
            None,
            engine.opts.blob_table_build_options.min_blob_size,
        );
    }

    {
        // Truncate all.
        // Note: Truncate all would be very fast as it just remove SSTs from meta, so
        // tolerate none here. If set truncate ts does not succeed, the `check_get`
        // must fail.
        set_truncate_ts(999, true);
        wait_for_truncate_ts();
        check_get(
            0,
            500,
            u64::MAX,
            WRT_EXT_CFS,
            &engine,
            false,
            None,
            engine.opts.blob_table_build_options.min_blob_size,
        );
    }
}

// This test case construct a LSM tree and trigger a particular compaction to
// verify deleted entry will not reappear after the compaction.
// In the LSM true, L3 contains data, L2 contains tombstone, L1 contains data.
// The range for overlap check should use both L1 and L2 instead of only L1.
// The tombstone entries will be discarded when the compaction is
// non-overlapping. If only use L1's range for overlap check, some of the L2's
// tombstone will be lost, cause deleted entries reappear.
#[test]
fn test_lost_tombstone_issue() {
    ::test_util::init_log_for_test();
    let (engine, _) = new_test_engine();
    let shard = engine.get_shard(1).unwrap();

    let mut cf_builder = ShardCfBuilder::new(0);
    let mut saved_vals: Vec<Rc<Vec<u8>>> = Vec::new();

    cf_builder.add_table(
        new_table(&engine, 11, 0, 100, 101, false, &mut saved_vals),
        3,
    );
    cf_builder.add_table(
        new_table(&engine, 12, 50, 150, 102, true, &mut saved_vals),
        2,
    );
    cf_builder.add_table(
        new_table(&engine, 13, 120, 200, 103, false, &mut saved_vals),
        1,
    );
    let data = ShardData::new(
        shard.range.clone(),
        vec![CfTable::new()],
        vec![],
        Arc::new(HashMap::default()),
        [cf_builder.build(), ShardCf::new(1), ShardCf::new(2)],
        HashMap::new(),
        vec![],
        RegionLimiter::dummy(),
        TEST_ENGINE_NEW_DATA_UPDATE_COUNTER,
        None,
        ColumnarLevels::new(),
    );
    shard.set_data(data);
    let pri = CompactionPriority::L1Plus {
        cf: 0,
        score: 2.0,
        level: 1,
    };
    let mut guard = shard.compaction_priority.write().unwrap();
    *guard = Some(pri);
    drop(guard);
    engine.update_managed_safe_ts(104);
    engine.trigger_compact(IdVer::new(1, 1));
    thread::sleep(Duration::from_secs(1));
    check_get(50, 100, 104, &[0], &engine, false, None, 0);
}

// This test case consturct a three level LSM tree and with different version
// and add tombstone with latest version. Iterator should return all versions of
// the key, excluding the tombstoned key.
#[test]
fn test_read_iterator_all_versions() {
    ::test_util::init_log_for_test();
    let (engine, _) = new_test_engine();
    let shard = engine.get_shard(1).unwrap();
    let cf = 0;

    let mut cf_builder = ShardCfBuilder::new(cf);
    let mut saved_vals: Vec<Rc<Vec<u8>>> = Vec::new();

    // 0..50 has only version 101
    // 50..70 has version 101 and 102
    // 70..90 has version 103 marked deleted, and older version 101 and 102
    // 90..100 has version 101 and 102
    // 100..150 has version 102
    cf_builder.add_table(
        new_table(&engine, 11, 0, 100, 101, false, &mut saved_vals),
        3,
    );
    cf_builder.add_table(
        new_table(&engine, 12, 50, 150, 102, false, &mut saved_vals),
        2,
    );
    cf_builder.add_table(
        new_table(&engine, 13, 70, 90, 103, true, &mut saved_vals),
        1,
    );

    let data = ShardData::new(
        shard.range.clone(),
        vec![CfTable::new()],
        vec![],
        Arc::new(HashMap::default()),
        [cf_builder.build(), ShardCf::new(1), ShardCf::new(2)],
        HashMap::new(),
        vec![],
        RegionLimiter::dummy(),
        TEST_ENGINE_NEW_DATA_UPDATE_COUNTER,
        None,
        ColumnarLevels::new(),
    );
    shard.set_data(data);

    let snap = SnapAccess::new(&shard);
    let mut iter = snap.new_iterator(cf, false, true, None, false);
    iter.seek(shard.outer_start.chunk());

    let mut expected_keys = Vec::with_capacity(160);
    for i in 0..50 {
        expected_keys.push(i);
    }
    for i in 50..70 {
        expected_keys.push(i);
        expected_keys.push(i);
    }

    // 70..90 should not returned

    for i in 90..100 {
        expected_keys.push(i);
        expected_keys.push(i);
    }
    for i in 100..150 {
        expected_keys.push(i);
    }
    let mut expected_keys_iter = expected_keys.iter();

    while iter.valid() {
        let key = engine
            .key_builder
            .i_to_outer_key(expected_keys_iter.next().unwrap().to_owned());
        assert_eq!(iter.key(), key.as_slice());
        assert_eq!(iter.val(), key.repeat(2).as_slice());
        iter.next();
    }
}

#[test]
fn test_lock_cf_repeatable_read() {
    let enable_inner_key_off = true;

    ::test_util::init_log_for_test();
    let (engine, applier_tx) =
        new_test_engine_opt(enable_inner_key_off, DEF_BLOCK_SIZE, TABLE_KEY_PREFIX);
    let kb = engine.key_builder();
    let shard = engine.get_shard(1).unwrap();

    let verify_locks = |snap: &SnapAccess, start: usize, end: usize, version: u64, del: bool| {
        let tag = format!("{start}->{end}, {version}, {del}");
        let mut it = snap.new_iterator(LOCK_CF, false, false, None, false);
        it.seek(&kb.i_to_outer_key(start));

        if del {
            assert!(
                !it.valid() || it.key() >= kb.i_to_outer_key(end).as_slice(),
                "{}",
                tag
            );
            return;
        }

        for i in start..end {
            assert!(it.valid(), "{}", tag);
            assert_eq!(it.key(), kb.i_to_outer_key(i).as_slice(), "{}", tag);
            assert_eq!(it.version(), version, "{}", tag);
            assert_eq!(table::is_deleted(it.meta()), del, "{}", tag);
            it.next();
        }
    };

    // Prepare level data.
    {
        let mut cf_builder = ShardCfBuilder::new(LOCK_CF);
        let mut saved_vals: Vec<Rc<Vec<u8>>> = Vec::new();
        cf_builder.add_table(
            new_table(&engine, 1000, 100, 300, 1000, false, &mut saved_vals),
            2,
        );
        cf_builder.add_table(
            new_table(&engine, 2000, 110, 300, 2000, true, &mut saved_vals),
            1,
        );

        let l0_file3 = new_l0table_file(
            &engine,
            3000,
            [0, 120, 0],
            [0, 300, 0],
            3000,
            [false, false, false],
        );
        let l0_tbl3 = L0Table::new(l0_file3, None, false, None).unwrap().unwrap();
        let l0_file4 = new_l0table_file(
            &engine,
            4000,
            [0, 130, 0],
            [0, 300, 0],
            4000,
            [false, true, false],
        );
        let l0_tbl4 = L0Table::new(l0_file4, None, false, None).unwrap().unwrap();

        let data = ShardDataBuilder::from_data(shard.get_data())
            .lv_tables(LOCK_CF, cf_builder.build())
            .l0_tables(vec![l0_tbl4, l0_tbl3])
            .build();
        shard.set_data(data);
        shard.set_base_version(10000);
    }

    let base_ver = shard.get_base_version();

    // Write mem table.
    let mem_tbl_ver5 = load_data_ext(
        &engine,
        [0, 140, 0],
        [0, 300, 0],
        5000, // useless
        [false, false, false],
        &applier_tx,
    ) + base_ver;

    // Write txn file lock.
    let txn_file_ver6 = {
        let chunk_id = 6000;
        build_txn_chunk(&engine, 150, 300, chunk_id, None, enable_inner_key_off);
        let primary = kb.i_to_outer_key(150);
        let mut wb = WriteBatch::new(1, engine.inner_key_off());
        let txn_file_refs = make_txn_file_refs(
            6000, // useless
            vec![chunk_id],
            make_lock_prefix(primary.clone(), 6000),
            vec![],
        );
        wb.set_property(TXN_FILE_REF, &txn_file_refs);
        engine.txn_chunk_mgr.prepare(chunk_id, None).unwrap();
        write_data(wb, &applier_tx)
    } + base_ver;

    let snap = shard.new_snap_access();
    print_locks(&snap, false, None);
    verify_locks(&snap, 100, 110, 1000, false);
    verify_locks(&snap, 110, 120, 2000, true);
    verify_locks(&snap, 120, 130, 3000, false);
    verify_locks(&snap, 130, 140, 4000, true);
    verify_locks(&snap, 140, 150, mem_tbl_ver5, false);
    verify_locks(&snap, 150, 300, snap.get_mem_table_version(), false);

    // Write more mem table data.
    {
        switch_mem_table(&engine, &applier_tx);
        load_data_ext(
            &engine,
            [0, 160, 0],
            [0, 300, 0],
            7000, // useless
            [false, false, false],
            &applier_tx,
        );
    }

    // Verify repeatable read.
    verify_locks(&snap, 140, 150, mem_tbl_ver5, false);
    verify_locks(&snap, 150, 300, snap.get_mem_table_version(), false);

    // Check latest snapshot.
    let snap = shard.new_snap_access();
    verify_locks(&snap, 140, 150, mem_tbl_ver5, false);
    verify_locks(&snap, 150, 160, txn_file_ver6, false);
    verify_locks(&snap, 160, 300, snap.get_mem_table_version(), false);
}

#[rstest]
#[case::enable_key_off(true)]
#[case::disable_key_off(false)]
fn test_level_overlapping_tables(#[case] enable_inner_key_off: bool) {
    ::test_util::init_log_for_test();
    let (engine, _) = new_test_engine_opt(enable_inner_key_off, DEF_BLOCK_SIZE, "");
    let shard = engine.get_shard(1).unwrap();

    let mut cf_builder = ShardCfBuilder::new(0);
    let mut saved_vals: Vec<Rc<Vec<u8>>> = Vec::new();

    let table_ranges: Vec<(usize, usize)> = vec![(10, 20), (50, 100), (120, 200)];
    for (i, (start, end)) in table_ranges.into_iter().enumerate() {
        cf_builder.add_table(
            new_table(
                &engine,
                i as u64,
                start,
                end,
                i as u64 + 100,
                false,
                &mut saved_vals,
            ),
            1, // level
        );
    }
    let data = ShardData::new(
        shard.range.clone(),
        vec![CfTable::new()],
        vec![],
        Arc::new(HashMap::default()),
        [cf_builder.build(), ShardCf::new(1), ShardCf::new(2)],
        HashMap::new(),
        vec![],
        RegionLimiter::dummy(),
        TEST_ENGINE_NEW_DATA_UPDATE_COUNTER,
        None,
        ColumnarLevels::new(),
    );

    let cf0 = data.get_cf(0);
    let level1 = cf0.get_level(1);
    let get_overlapping_tables = |start: usize, end: usize| -> (usize, usize) {
        level1.overlapping_tables_exclusive_end(
            engine.key_builder.i_to_inner_key(start).as_ref(),
            engine.key_builder.i_to_inner_key(end).as_ref(),
        )
    };

    assert_eq!(get_overlapping_tables(0, 10), (0, 0));
    assert_eq!(get_overlapping_tables(0, 20), (0, 1));
    assert_eq!(get_overlapping_tables(0, 50), (0, 1));
    assert_eq!(get_overlapping_tables(0, 100), (0, 2));
    assert_eq!(get_overlapping_tables(0, 120), (0, 2));
    assert_eq!(get_overlapping_tables(0, 200), (0, 3));
    assert_eq!(get_overlapping_tables(0, 300), (0, 3));

    assert_eq!(get_overlapping_tables(10, 20), (0, 1));
    assert_eq!(get_overlapping_tables(10, 50), (0, 1));
    assert_eq!(get_overlapping_tables(10, 100), (0, 2));
    assert_eq!(get_overlapping_tables(10, 120), (0, 2));
    assert_eq!(get_overlapping_tables(10, 200), (0, 3));
    assert_eq!(get_overlapping_tables(10, 300), (0, 3));

    assert_eq!(get_overlapping_tables(20, 50), (1, 1));
    assert_eq!(get_overlapping_tables(20, 100), (1, 2));
    assert_eq!(get_overlapping_tables(20, 120), (1, 2));
    assert_eq!(get_overlapping_tables(20, 200), (1, 3));
    assert_eq!(get_overlapping_tables(20, 300), (1, 3));

    assert_eq!(get_overlapping_tables(50, 100), (1, 2));
    assert_eq!(get_overlapping_tables(50, 120), (1, 2));
    assert_eq!(get_overlapping_tables(50, 200), (1, 3));
    assert_eq!(get_overlapping_tables(50, 300), (1, 3));

    assert_eq!(get_overlapping_tables(100, 120), (2, 2));
    assert_eq!(get_overlapping_tables(100, 200), (2, 3));
    assert_eq!(get_overlapping_tables(100, 300), (2, 3));

    assert_eq!(get_overlapping_tables(120, 200), (2, 3));
    assert_eq!(get_overlapping_tables(120, 300), (2, 3));

    assert_eq!(get_overlapping_tables(200, 300), (3, 3));
}

#[rstest]
#[case::enable_key_off(true)]
#[case::disable_key_off(false)]
fn test_get_suggest_split_key(#[case] enable_inner_key_off: bool) {
    ::test_util::init_log_for_test();
    let (engine, _) = new_test_engine_opt(enable_inner_key_off, 4096, "");
    let shard = engine.get_shard(1).unwrap();

    let range = ShardRange::new(
        &engine.key_builder.i_to_outer_key(30),
        &engine.key_builder.i_to_outer_key(600),
        shard.inner_key_off,
    );
    let shard = Shard::new(
        shard.engine_id,
        &shard.properties.to_pb(shard.id),
        shard.ver,
        range.clone(),
        engine.opts.clone(),
        &engine.master_key,
    );

    let cases: Vec<(
        Vec<(usize, usize)>,
        Vec<(usize, usize)>, // table ranges
        // expected suggest split key, inner key off (enable,disable)
        (Option<usize>, Option<usize>),
    )> = vec![
        (vec![], vec![(20, 50)], (None, None)),
        (
            vec![],
            vec![(20, 50), (100, 150), (150, 180)],
            (Some(150), Some(150)),
        ),
        (
            vec![],
            vec![
                (20, 50),
                (100, 150), // in range
                (150, 180), // in range
                (180, 300), // in range
                (300, 400),
            ],
            (Some(180), Some(180)),
        ),
        (
            vec![],
            vec![
                (20, 50),
                (100, 150), // in range, key in block
            ],
            (Some(100), Some(100)),
        ),
        (
            vec![(50, 400), (60, 460), (70, 670)], // L0 only
            vec![],
            (Some(320), Some(270)),
        ),
        (
            vec![(70, 370)], // L0 + L1+
            vec![(50, 80), (80, 100), (100, 120)],
            (Some(100), Some(100)),
        ),
        (
            vec![(80, 380), (190, 570)], // more L0 + L1+
            vec![(50, 80), (80, 100), (100, 120)],
            (Some(333), Some(280)),
        ),
    ];
    let mut id_alloc = 1000;
    for (idx, (l0_ranges, table_ranges, expect_split_key)) in cases.into_iter().enumerate() {
        let mut l0_tables = vec![];
        for (start, end) in l0_ranges {
            let id = id_alloc;
            id_alloc += 1;
            let (begins, ends) = if idx % 2 == 0 {
                ([start, 0, 0], [end, 0, 0])
            } else {
                ([0, start, 0], [0, end, 0])
            };
            let l0_file = new_l0table_file(&engine, id, begins, ends, id, [false, false, false]);
            let l0_tbl = L0Table::new(l0_file, None, false, None).unwrap().unwrap();
            l0_tables.push(l0_tbl);
        }
        let mut cf_builder = ShardCfBuilder::new(idx % 2);
        let mut saved_vals: Vec<Rc<Vec<u8>>> = Vec::new();

        for (i, (start, end)) in table_ranges.into_iter().enumerate() {
            cf_builder.add_table(
                new_table(
                    &engine,
                    i as u64 + 1,
                    start,
                    end,
                    100 + i as u64,
                    false,
                    &mut saved_vals,
                ),
                1,
            );
        }

        let data = ShardData::new(
            range.clone(),
            vec![CfTable::new()],
            l0_tables,
            Arc::new(HashMap::default()),
            [cf_builder.build(), ShardCf::new(1), ShardCf::new(2)],
            HashMap::new(),
            vec![],
            RegionLimiter::dummy(),
            TEST_ENGINE_NEW_DATA_UPDATE_COUNTER,
            None,
            ColumnarLevels::new(),
        );
        shard.set_data_opt(data, false);

        let key = shard.get_suggest_split_key();
        let expect_split_key = if enable_inner_key_off {
            expect_split_key.0
        } else {
            expect_split_key.1
        };
        assert_eq!(
            key.map(|k| k.to_vec()),
            expect_split_key.map(|i| engine.key_builder.i_to_outer_key(i)),
            "case {}",
            idx
        );
    }
}

#[rstest]
#[case::enable_key_off(true)]
#[case::disable_key_off(false)]
fn test_get_evenly_split_keys(#[case] enable_inner_key_off: bool) {
    ::test_util::init_log_for_test();
    let (engine, _) = new_test_engine_opt(enable_inner_key_off, 4096, "");
    let shard = engine.get_shard(1).unwrap();

    let range = ShardRange::new(
        &engine.key_builder.i_to_outer_key(30),
        &engine.key_builder.i_to_outer_key(600),
        shard.inner_key_off,
    );
    let shard = Shard::new(
        shard.engine_id,
        &shard.properties.to_pb(shard.id),
        shard.ver,
        range.clone(),
        engine.opts.clone(),
        &engine.master_key,
    );

    let cases: Vec<(
        Vec<(usize, usize)>, // l0 ranges
        Vec<(usize, usize)>, // table ranges
        usize,               // split count
        // expected evenly split keys, inner key off (enable,disable)
        (Option<Vec<usize>>, Option<Vec<usize>>),
    )> = vec![
        (vec![], vec![(20, 50), (50, 100)], 1, (None, None)),
        (vec![], vec![(20, 50), (50, 100)], 2, (None, None)),
        (
            vec![],
            vec![(0, 20), (20, 50), (100, 150), (150, 180)],
            2,
            (Some(vec![150]), Some(vec![150])),
        ),
        (
            vec![],
            vec![
                (20, 50),
                (100, 150), // in range
                (150, 180), // in range
                (180, 300), // in range
                (300, 400),
            ],
            1,
            (None, None),
        ),
        (
            vec![],
            vec![
                (20, 50),
                (100, 150), // in range
                (150, 180), // in range
                (180, 300), // in range
            ],
            2,
            (Some(vec![180]), Some(vec![180])),
        ),
        (
            vec![],
            vec![
                (100, 120), // in range
                (120, 150), // in range
                (150, 180), // in range
                (180, 300), // in range
            ],
            3,
            (Some(vec![150, 180]), Some(vec![150, 180])),
        ),
        (
            vec![],
            vec![
                (20, 50),
                (100, 150), // in range
                (150, 180), // in range
                (180, 300), // in range
                (300, 400),
            ],
            4,
            (Some(vec![150, 180, 300]), Some(vec![150, 180, 300])),
        ),
        (
            vec![],
            vec![
                (20, 50),
                (100, 150), // in range
                (150, 180), // in range
                (180, 300), // in range
                (300, 400),
            ],
            5,
            (Some(vec![150, 180, 300]), Some(vec![150, 180, 300])),
        ),
        (
            vec![],
            vec![
                (20, 50),
                (100, 150), // in range
                (150, 180), // in range
                (180, 300), // in range
                (300, 400),
            ],
            6,
            (Some(vec![150, 180, 300]), Some(vec![150, 180, 300])),
        ),
        (
            vec![(100, 400), (200, 500), (300, 600), (400, 700)], // L0 only
            vec![],
            4,
            (Some(vec![456, 556]), Some(vec![406, 506])),
        ),
        (
            vec![(100, 400), (150, 550), (300, 700), (200, 800)], // L0 only
            vec![],
            4,
            (Some(vec![400, 456, 556]), Some(vec![350, 406, 506])),
        ),
        (
            vec![(100, 800), (200, 900)], // L0 + L1+
            vec![(30, 80), (80, 100), (100, 130)],
            4,
            (Some(vec![100, 356, 456]), Some(vec![306, 406, 512])),
        ),
    ];
    let mut id_alloc = 1000;
    for (idx, (l0_ranges, table_ranges, split_count, expect_split_keys)) in
        cases.into_iter().enumerate()
    {
        let mut l0_tables = vec![];
        for (start, end) in l0_ranges {
            let id = id_alloc;
            id_alloc += 1;
            let (begins, ends) = if idx % 2 == 0 {
                ([start, 0, 0], [end, 0, 0])
            } else {
                ([0, start, 0], [0, end, 0])
            };
            let l0_file = new_l0table_file(&engine, id, begins, ends, id, [false, false, false]);
            let l0_tbl = L0Table::new(l0_file, None, false, None).unwrap().unwrap();
            l0_tables.push(l0_tbl);
        }
        let mut cf_builder = ShardCfBuilder::new(idx % 2);
        let mut saved_vals: Vec<Rc<Vec<u8>>> = Vec::new();

        for (i, (start, end)) in table_ranges.into_iter().enumerate() {
            cf_builder.add_table(
                new_table(
                    &engine,
                    i as u64 + 1,
                    start,
                    end,
                    100 + i as u64,
                    false,
                    &mut saved_vals,
                ),
                1, // level
            );
        }

        let data = ShardData::new(
            range.clone(),
            vec![CfTable::new()],
            l0_tables,
            Arc::new(HashMap::default()),
            [cf_builder.build(), ShardCf::new(1), ShardCf::new(2)],
            HashMap::new(),
            vec![],
            RegionLimiter::dummy(),
            TEST_ENGINE_NEW_DATA_UPDATE_COUNTER,
            None,
            ColumnarLevels::new(),
        );
        shard.set_data_opt(data, false);

        let split_keys = shard.get_evenly_split_keys(split_count);
        let expect_split_keys = if enable_inner_key_off {
            expect_split_keys.0
        } else {
            expect_split_keys.1
        };
        assert_eq!(
            split_keys.map(|keys| keys.into_iter().map(|k| k.to_vec()).collect::<Vec<_>>()),
            expect_split_keys.map(|keys| keys
                .into_iter()
                .map(|i| engine.key_builder.i_to_outer_key(i))
                .collect::<Vec<_>>()),
            "case {}",
            idx
        );
    }
}

#[test]
fn test_refresh_stats() {
    ::test_util::init_log_for_test();
    let (engine, _) = new_test_engine_opt(true, DEF_BLOCK_SIZE, "");
    let shard = engine.get_shard(1).unwrap();

    let mut saved_vals: Vec<Rc<Vec<u8>>> = Vec::new();

    let mut write_cf_builder = ShardCfBuilder::new(WRITE_CF);
    write_cf_builder.add_table(
        new_table(&engine, 12, 50, 150, 102, false, &mut saved_vals),
        2,
    );
    write_cf_builder.add_table(
        new_table(&engine, 13, 70, 90, 103, true, &mut saved_vals),
        1,
    );

    let mut lock_cf_builder = ShardCfBuilder::new(LOCK_CF);
    lock_cf_builder.add_table(
        new_table(&engine, 21, 50, 150, 1001, false, &mut saved_vals),
        2,
    );
    lock_cf_builder.add_table(
        new_table(&engine, 22, 70, 90, 1002, true, &mut saved_vals),
        1,
    );

    let mut extra_cf_builder = ShardCfBuilder::new(EXTRA_CF);
    extra_cf_builder.add_table(
        new_table(&engine, 31, 50, 150, 150, false, &mut saved_vals),
        1,
    );

    let data = ShardData::new(
        shard.range.clone(),
        vec![CfTable::new()],
        vec![],
        Arc::new(HashMap::default()),
        [
            write_cf_builder.build(),
            lock_cf_builder.build(),
            extra_cf_builder.build(),
        ],
        HashMap::new(),
        vec![],
        RegionLimiter::dummy(),
        TEST_ENGINE_NEW_DATA_UPDATE_COUNTER,
        None,
        ColumnarLevels::new(),
    );
    shard.set_data(data);
    shard.refresh_states();

    // size per entry: key 9, value 18
    assert_eq!(shard.get_estimated_entries(), 340); // 100 + 20 + 100 + 20 + 100
    assert_eq!(load_u64(&shard.sst_max_ts), 150); // max(WRITE_CF, EXTRA_CF). TODO: test with mem tables.
    assert_eq!(shard.get_max_ts(), 150); // max(WRITE_CF, EXTRA_CF)
    assert_eq!(shard.get_estimated_kv_size(), 2880); // 100*27 + 20*9
    assert_eq!(load_u64(&shard.tombs), 20); // WRITE_CF only
    assert_eq!(load_u64(&shard.entries_write_cf), 120); // WRITE_CF only
}

#[test]
fn test_l0table_ignore_lock() {
    ::test_util::init_log_for_test();
    let (engine, _) = new_test_engine();
    let file = new_l0table_file(
        &engine,
        1,
        [0, 0, 0],
        [0, 100, 0],
        100,
        [false, false, false],
    );

    let l0table = L0Table::new(file.clone(), None, false, None).unwrap();
    assert!(l0table.is_some());

    // l0table would be none when `ignore_lock` is true and only LOCK_CF has data.
    // See https://github.com/tidbcloud/cloud-storage-engine/issues/1026.
    let l0table_ignore_lock = L0Table::new(file, None, true, None).unwrap();
    assert!(l0table_ignore_lock.is_none());
}

#[rstest]
#[case::inner_key_off_enable(true)]
#[case::inner_key_off_disable(false)]
fn test_columnar_l0_compaction(#[case] enable_inner_key_off: bool) {
    ::test_util::init_log_for_test();
    let keyspace_id = 1;
    let table_id = 30;
    let mut file_id = 100;
    let mut allocate_id = || {
        file_id += 1;
        file_id
    };
    let (engine, apply_tx) = new_test_engine_opt(enable_inner_key_off, DEF_BLOCK_SIZE, "");
    let shard_id = prepare_table_region(&engine, &apply_tx, keyspace_id, table_id);
    let shard = engine.get_shard(shard_id).unwrap();
    let schema = new_schema(table_id, true);
    let schemas = vec![schema.clone()];
    let schema_version = 10;
    let schema_file_data = build_schema_file(keyspace_id, schema_version, schemas);
    let fs = engine.fs.clone();
    let schema_raw_file = Arc::new(InMemFile::new(allocate_id(), Bytes::from(schema_file_data)));
    fs.get_runtime()
        .block_on(
            fs.create(
                schema_raw_file.id(),
                schema_raw_file
                    .read(0, schema_raw_file.size() as usize)
                    .unwrap(),
                dfs::Options::default().with_type(FileType::Schema),
            ),
        )
        .unwrap();
    let schema_file = SchemaFile::open(schema_raw_file).unwrap();
    let opts = dfs::Options::default().with_type(FileType::Columnar);
    let (l0_tbl_0, l0_tbl_0_ref) =
        build_table(allocate_id(), enable_inner_key_off, &schema, 0, 600, 300);
    let (l0_tbl_1, l0_tbl_1_ref) =
        build_table(allocate_id(), enable_inner_key_off, &schema, 300, 900, 400);
    let (l0_tbl_2, l0_tbl_2_ref) = build_table(
        allocate_id(),
        enable_inner_key_off,
        &schema,
        1000,
        2000,
        400,
    );
    let (l1_tbl_0, l1_tbl_0_ref) =
        build_table(allocate_id(), enable_inner_key_off, &schema, 0, 100, 100);
    for file in [&l0_tbl_0, &l0_tbl_1, &l0_tbl_2, &l1_tbl_0] {
        info!("build file_id: {}, file size: {}", file.id(), file.size());
        fs.get_runtime()
            .block_on(fs.create(file.id(), file.read(0, file.size() as usize).unwrap(), opts))
            .unwrap()
    }

    let mut col_levels = ColumnarLevels::new();
    col_levels.add_file(0, ColumnarFile::open(l0_tbl_0).unwrap());
    col_levels.add_file(0, ColumnarFile::open(l0_tbl_1).unwrap());
    col_levels.add_file(0, ColumnarFile::open(l0_tbl_2).unwrap());
    col_levels.add_file(1, ColumnarFile::open(l1_tbl_0).unwrap());

    let data = ShardData::new(
        shard.range.clone(),
        vec![CfTable::new()],
        vec![],
        Arc::new(HashMap::default()),
        [
            ShardCf::new(WRITE_CF),
            ShardCf::new(LOCK_CF),
            ShardCf::new(EXTRA_CF),
        ],
        HashMap::new(),
        vec![],
        RegionLimiter::dummy(),
        shard.get_data().update_counter + 1,
        Some(schema_file),
        col_levels,
    );
    shard.set_data(data);
    shard.initial_flushed.store(true, Ordering::SeqCst);
    let id_ver = shard.id_ver();
    *shard.compaction_priority.write().unwrap() =
        Some(CompactionPriority::ColumnarL0 { score: 2.0 });
    engine.trigger_compact(id_ver);
    info!("trigger columnar l0 compaction {}", shard.tag());
    let ok = try_wait(
        || {
            info!(
                "wait columnar l0 compaction {} l0 files: {}, l1 files: {}",
                shard.tag(),
                shard.get_data().col_levels.levels[0].files.len(),
                shard.get_data().col_levels.levels[1].files.len()
            );
            shard.get_data().col_levels.levels[0].files.is_empty()
        },
        5,
    );
    assert!(ok, "columnar l0 compaction failed");
    let snap = shard.new_snap_access();
    let mut mvcc_reader = snap
        .new_columnar_mvcc_reader(table_id, &schema.columns, 500)
        .unwrap();
    mvcc_reader
        .set_handle_range(&i_to_common_handle(0), &i_to_common_handle(2100))
        .unwrap();
    let mut without_txn_id_schema_buf = schema.to_schema_buf();
    without_txn_id_schema_buf.txn_id_column = None;
    let without_txn_id_schema = without_txn_id_schema_buf.into();
    let mut block = Block::new(&without_txn_id_schema);
    mvcc_reader.read_block(&mut block, usize::MAX).unwrap();
    let tbl_refs = merge_refs(
        vec![l0_tbl_0_ref, l0_tbl_1_ref, l0_tbl_2_ref, l1_tbl_0_ref],
        1,
        Some(500),
        None,
        Some((i_to_common_handle(0), i_to_common_handle(2100))),
    );
    assert_eq!(block.length(), tbl_refs.len());
    verify_with_ref_rows(&block, &tbl_refs);
}

#[rstest]
#[case::inner_key_off_enable(true)]
#[case::inner_key_off_disable(false)]
fn test_columnar_l1_compaction(#[case] enable_inner_key_off: bool) {
    ::test_util::init_log_for_test();
    let keyspace_id = 1;
    let table_id = 30;
    let mut file_id = 100;
    let mut allocate_id = || {
        file_id += 1;
        file_id
    };
    let (engine, apply_tx) = new_test_engine_opt(enable_inner_key_off, DEF_BLOCK_SIZE, "");
    let shard_id = prepare_table_region(&engine, &apply_tx, keyspace_id, table_id);
    let shard = engine.get_shard(shard_id).unwrap();
    let schema = new_schema(table_id, true);
    let schemas = vec![schema.clone()];
    let schema_version = 10;
    let schema_file_data = build_schema_file(keyspace_id, schema_version, schemas);
    let fs = engine.fs.clone();
    let schema_raw_file = Arc::new(InMemFile::new(allocate_id(), Bytes::from(schema_file_data)));
    fs.get_runtime()
        .block_on(
            fs.create(
                schema_raw_file.id(),
                schema_raw_file
                    .read(0, schema_raw_file.size() as usize)
                    .unwrap(),
                dfs::Options::default().with_type(FileType::Schema),
            ),
        )
        .unwrap();
    let schema_file = SchemaFile::open(schema_raw_file).unwrap();
    let opts = dfs::Options::default().with_type(FileType::Columnar);
    let (l1_tbl_0, l1_tbl_0_ref) =
        build_table(allocate_id(), enable_inner_key_off, &schema, 0, 600, 300);
    let (l1_tbl_1, l1_tbl_1_ref) =
        build_table(allocate_id(), enable_inner_key_off, &schema, 300, 900, 400);
    let (l1_tbl_2, l1_tbl_2_ref) = build_table(
        allocate_id(),
        enable_inner_key_off,
        &schema,
        1000,
        1500,
        400,
    );
    let (l2_tbl_0, l2_tbl_0_ref) =
        build_table(allocate_id(), enable_inner_key_off, &schema, 0, 1000, 100);
    let (l2_tbl_1, l2_tbl_1_ref) = build_table(
        allocate_id(),
        enable_inner_key_off,
        &schema,
        1000,
        2000,
        100,
    );
    for file in [&l1_tbl_0, &l1_tbl_1, &l1_tbl_2, &l2_tbl_0, &l2_tbl_1] {
        info!("build file_id: {}, file size: {}", file.id(), file.size());
        fs.get_runtime()
            .block_on(fs.create(file.id(), file.read(0, file.size() as usize).unwrap(), opts))
            .unwrap()
    }

    let mut col_levels = ColumnarLevels::new();
    col_levels.add_file(1, ColumnarFile::open(l1_tbl_0).unwrap());
    col_levels.add_file(1, ColumnarFile::open(l1_tbl_1).unwrap());
    col_levels.add_file(1, ColumnarFile::open(l1_tbl_2).unwrap());
    col_levels.add_file(2, ColumnarFile::open(l2_tbl_0).unwrap());
    col_levels.add_file(2, ColumnarFile::open(l2_tbl_1).unwrap());

    let data = ShardData::new(
        shard.range.clone(),
        vec![CfTable::new()],
        vec![],
        Arc::new(HashMap::default()),
        [
            ShardCf::new(WRITE_CF),
            ShardCf::new(LOCK_CF),
            ShardCf::new(EXTRA_CF),
        ],
        HashMap::new(),
        vec![],
        RegionLimiter::dummy(),
        shard.get_data().update_counter + 1,
        Some(schema_file),
        col_levels,
    );
    shard.set_data(data);
    shard.initial_flushed.store(true, Ordering::SeqCst);
    let id_ver = shard.id_ver();
    *shard.compaction_priority.write().unwrap() =
        Some(CompactionPriority::ColumnarL1 { score: 2.0 });
    engine.trigger_compact(id_ver);
    info!("trigger columnar l1 compaction {}", shard.tag());
    let ok = try_wait(
        || {
            info!(
                "wait columnar l1 compaction {} l1 files: {}, l2 files: {}",
                shard.tag(),
                shard.get_data().col_levels.levels[1].files.len(),
                shard.get_data().col_levels.levels[2].files.len()
            );
            shard.get_data().col_levels.levels[1].files.is_empty()
        },
        5,
    );
    assert!(ok, "columnar l1 compaction failed");
    let snap = shard.new_snap_access();
    let mut mvcc_reader = snap
        .new_columnar_mvcc_reader(table_id, &schema.columns, 500)
        .unwrap();
    mvcc_reader
        .set_handle_range(&i_to_common_handle(0), &i_to_common_handle(2100))
        .unwrap();
    let mut without_txn_id_schema = schema.to_schema_buf();
    without_txn_id_schema.txn_id_column = None;
    let mut block = Block::new(&without_txn_id_schema.into());
    mvcc_reader.read_block(&mut block, usize::MAX).unwrap();
    let tbl_refs = merge_refs(
        vec![
            l1_tbl_0_ref,
            l1_tbl_1_ref,
            l1_tbl_2_ref,
            l2_tbl_0_ref,
            l2_tbl_1_ref,
        ],
        2,
        Some(500),
        None,
        Some((i_to_common_handle(0), i_to_common_handle(2100))),
    );
    assert_eq!(block.length(), tbl_refs.len());
    verify_with_ref_rows(&block, &tbl_refs);
}

#[rstest]
#[case::inner_key_off_enable(true)]
#[case::inner_key_off_disable(false)]
fn test_columnar_major_compaction(#[case] enable_inner_key_off: bool) {
    use table::columnar::{ColumnarMergeReader, ColumnarMvccReader};

    ::test_util::init_log_for_test();
    let keyspace_id = KEYSPACE_ID;
    let table_id = 30;
    let mut file_id = 100;
    let mut allocate_id = || {
        file_id += 1;
        file_id
    };
    let (engine, apply_tx) = new_test_engine_opt(enable_inner_key_off, DEF_BLOCK_SIZE, "");
    let shard_id = prepare_table_region(&engine, &apply_tx, keyspace_id, table_id);
    let shard = engine.get_shard(shard_id).unwrap();
    let schema = new_schema(table_id, false);
    let schemas = vec![schema.clone()];
    let schema_version = 10;
    let schema_file_data = build_schema_file(keyspace_id, schema_version, schemas);
    let fs = engine.fs.clone();
    let schema_raw_file = Arc::new(InMemFile::new(allocate_id(), Bytes::from(schema_file_data)));
    fs.get_runtime()
        .block_on(
            fs.create(
                schema_raw_file.id(),
                schema_raw_file
                    .read(0, schema_raw_file.size() as usize)
                    .unwrap(),
                dfs::Options::default().with_type(FileType::Schema),
            ),
        )
        .unwrap();
    let schema_file = SchemaFile::open(schema_raw_file).unwrap();
    let mut saved_vals = vec![];
    let l1_tbl_0 = new_sst_table_for_columnar(
        &engine,
        allocate_id(),
        table_id,
        300,
        900,
        350,
        &mut saved_vals,
    );
    let l1_tbl_1 = new_sst_table_for_columnar(
        &engine,
        allocate_id(),
        table_id,
        1000,
        1500,
        350,
        &mut saved_vals,
    );
    let l2_tbl_0 = new_sst_table_for_columnar(
        &engine,
        allocate_id(),
        table_id,
        0,
        1000,
        100,
        &mut saved_vals,
    );
    let l2_tbl_1 = new_sst_table_for_columnar(
        &engine,
        allocate_id(),
        table_id,
        1000,
        2000,
        100,
        &mut saved_vals,
    );

    let level_handler_1 = LevelHandler::new(1, vec![l1_tbl_0.clone(), l1_tbl_1.clone()]);
    let level_handler_2 = LevelHandler::new(2, vec![l2_tbl_0.clone(), l2_tbl_1.clone()]);
    let mut write_cf = ShardCf::new(WRITE_CF);
    write_cf.set_level(level_handler_1);
    write_cf.set_level(level_handler_2);

    let data = ShardData::new(
        shard.range.clone(),
        vec![CfTable::new()],
        vec![],
        Arc::new(HashMap::default()),
        [write_cf, ShardCf::new(LOCK_CF), ShardCf::new(EXTRA_CF)],
        HashMap::new(),
        vec![],
        RegionLimiter::dummy(),
        shard.get_data().update_counter + 1,
        Some(schema_file),
        ColumnarLevels::new(),
    );
    shard.set_data(data);
    shard.initial_flushed.store(true, Ordering::SeqCst);
    let id_ver = shard.id_ver();
    *shard.compaction_priority.write().unwrap() =
        Some(CompactionPriority::ColumnarMajor { score: 2.0 });
    engine.trigger_compact(id_ver);
    info!("trigger columnar major compaction {}", shard.tag());
    let ok = try_wait(
        || {
            info!(
                "wait columnar major compaction {} l0 files: {}, l1 files: {}, l2 files: {}",
                shard.tag(),
                shard.get_data().col_levels.levels[0].files.len(),
                shard.get_data().col_levels.levels[1].files.len(),
                shard.get_data().col_levels.levels[2].files.len()
            );
            !shard.get_data().col_levels.levels[2].files.is_empty()
        },
        5,
    );
    assert!(ok, "columnar major compaction failed");
    let snap = shard.new_snap_access();
    let mut mvcc_reader = snap
        .new_columnar_mvcc_reader(table_id, &schema.columns, 500)
        .unwrap();
    // Use a random end int handle
    let end_handle = thread_rng().gen_range(500..2100);
    mvcc_reader
        .set_int_handle_range(0, Some(end_handle))
        .unwrap();
    let mut no_txn_id_schema_buf = schema.to_schema_buf();
    no_txn_id_schema_buf.txn_id_column = None;
    let no_txn_id_schema = Schema::new(no_txn_id_schema_buf);
    let mut block = Block::new(&no_txn_id_schema);
    let mvcc_reader_counts = mvcc_reader.read_block(&mut block, usize::MAX).unwrap();
    info!("mvcc_reader read block with {} rows", mvcc_reader_counts);
    let inner_key_off = if enable_inner_key_off { 4 } else { 0 };
    let mut columnar_readers: Vec<Box<dyn ColumnarReader>> = vec![];
    for tbl in [l1_tbl_0, l1_tbl_1, l2_tbl_0, l2_tbl_1] {
        let iter = tbl.new_iterator(false, false);
        let reader = ColumnarRowTableReader::new(
            keyspace_id,
            inner_key_off,
            no_txn_id_schema.clone(),
            iter,
            None,
            false,
        );
        columnar_readers.push(Box::new(reader));
    }
    let merged_reader = ColumnarMergeReader::new(no_txn_id_schema.clone(), columnar_readers);
    let mut mvcc_row_reader =
        ColumnarMvccReader::new(Box::new(merged_reader), &no_txn_id_schema, 500);
    mvcc_row_reader
        .set_int_handle_range(0, Some(end_handle))
        .unwrap();
    let mut row_block = Block::new(&no_txn_id_schema);
    let merge_reader_counts = mvcc_row_reader
        .read_block(&mut row_block, usize::MAX)
        .unwrap();
    info!("merge_reader read block with {} rows", merge_reader_counts);
    verify_columnar_with_blocks(&row_block, &block);

    // Remove columnar compaction.
    let data = shard.get_data();
    let new_data = ShardData::new(
        data.range.clone(),
        data.mem_tbls.clone(),
        data.l0_tbls.clone(),
        data.blob_tbl_map.clone(),
        data.cfs.clone(),
        data.unloaded_tbls.clone(),
        data.lock_txn_files.clone(),
        data.limiter.clone(),
        data.update_counter + 1,
        data.schema_file.clone(), // schema file will be cleared after compaction
        data.col_levels.clone(),
    );
    shard.set_data(new_data);
    *shard.compaction_priority.write().unwrap() = Some(CompactionPriority::ColumnarClear);
    engine.trigger_compact(id_ver);
    info!("trigger remove columnar compaction {}", shard.tag());
    let ok = try_wait(
        || {
            info!(
                "wait remove columnar compaction {} l0 files: {}, l1 files: {}, l2 files: {}",
                shard.tag(),
                shard.get_data().col_levels.levels[0].files.len(),
                shard.get_data().col_levels.levels[1].files.len(),
                shard.get_data().col_levels.levels[2].files.len()
            );
            shard.get_data().col_levels.levels[2].files.is_empty()
        },
        5,
    );
    assert!(ok);
    assert_eq!(shard.get_columnar_snap_version(), 0);
    assert!(shard.get_data().schema_file.is_none());
}

#[rstest]
#[case::inner_key_off_enable(true)]
#[case::inner_key_off_disable(false)]
fn test_columnar_destroy_range(#[case] enable_inner_key_off: bool) {
    ::test_util::init_log_for_test();
    let keyspace_id = 1;
    let table_id = 30;
    let mut file_id = 100;
    let mut allocate_id = || {
        file_id += 1;
        file_id
    };
    let (engine, apply_tx) = new_test_engine_opt(enable_inner_key_off, DEF_BLOCK_SIZE, "");
    let shard_id = prepare_table_region(&engine, &apply_tx, keyspace_id, table_id);
    let shard = engine.get_shard(shard_id).unwrap();
    let schema = new_schema(table_id, true);
    let schemas = vec![schema.clone()];
    let schema_version = 10;
    let schema_file_data = build_schema_file(keyspace_id, schema_version, schemas);
    let fs = engine.fs.clone();
    let schema_raw_file = Arc::new(InMemFile::new(allocate_id(), Bytes::from(schema_file_data)));
    fs.get_runtime()
        .block_on(
            fs.create(
                schema_raw_file.id(),
                schema_raw_file
                    .read(0, schema_raw_file.size() as usize)
                    .unwrap(),
                dfs::Options::default().with_type(FileType::Schema),
            ),
        )
        .unwrap();
    let schema_file = SchemaFile::open(schema_raw_file).unwrap();
    let opts = dfs::Options::default().with_type(FileType::Columnar);
    let (l0_tbl_0, _) = build_table(allocate_id(), enable_inner_key_off, &schema, 0, 600, 500);
    let (l0_tbl_1, _) = build_table(allocate_id(), enable_inner_key_off, &schema, 300, 900, 500);
    let (l0_tbl_2, _) = build_table(
        allocate_id(),
        enable_inner_key_off,
        &schema,
        1000,
        2000,
        500,
    );
    let (l1_tbl_0, _) = build_table(allocate_id(), enable_inner_key_off, &schema, 0, 600, 300);
    let (l1_tbl_1, _) = build_table(allocate_id(), enable_inner_key_off, &schema, 300, 900, 400);
    let (l1_tbl_2, _) = build_table(
        allocate_id(),
        enable_inner_key_off,
        &schema,
        1000,
        1500,
        400,
    );
    let (l2_tbl_0, _) = build_table(allocate_id(), enable_inner_key_off, &schema, 0, 1000, 100);
    let (l2_tbl_1, _) = build_table(
        allocate_id(),
        enable_inner_key_off,
        &schema,
        1000,
        2000,
        100,
    );
    for file in [
        &l0_tbl_0, &l0_tbl_1, &l0_tbl_2, &l1_tbl_0, &l1_tbl_1, &l1_tbl_2, &l2_tbl_0, &l2_tbl_1,
    ] {
        info!("build file_id: {}, file size: {}", file.id(), file.size());
        fs.get_runtime()
            .block_on(fs.create(file.id(), file.read(0, file.size() as usize).unwrap(), opts))
            .unwrap()
    }

    let mut col_levels = ColumnarLevels::new();
    col_levels.add_file(0, ColumnarFile::open(l0_tbl_0).unwrap());
    col_levels.add_file(0, ColumnarFile::open(l0_tbl_1).unwrap());
    col_levels.add_file(0, ColumnarFile::open(l0_tbl_2).unwrap());
    col_levels.add_file(1, ColumnarFile::open(l1_tbl_0).unwrap());
    col_levels.add_file(1, ColumnarFile::open(l1_tbl_1).unwrap());
    col_levels.add_file(1, ColumnarFile::open(l1_tbl_2).unwrap());
    col_levels.add_file(2, ColumnarFile::open(l2_tbl_0).unwrap());
    col_levels.add_file(2, ColumnarFile::open(l2_tbl_1).unwrap());

    let data = ShardData::new(
        shard.range.clone(),
        vec![CfTable::new()],
        vec![],
        Arc::new(HashMap::default()),
        [
            ShardCf::new(WRITE_CF),
            ShardCf::new(LOCK_CF),
            ShardCf::new(EXTRA_CF),
        ],
        HashMap::new(),
        vec![],
        RegionLimiter::dummy(),
        shard.get_data().update_counter + 1,
        Some(schema_file),
        col_levels,
    );
    shard.set_data(data);
    let mut del_prefixes = DeletePrefixes::new_with_inner_key_off(shard.inner_key_off);
    let mut table_prefix = api_version::ApiV2::get_txn_keyspace_prefix(keyspace_id);
    table_prefix.extend_from_slice(b"t");
    del_prefixes.merge_prefix_in_place(&table_prefix);
    shard.set_property(DEL_PREFIXES_KEY, &del_prefixes.marshal());
    shard.initial_flushed.store(true, Ordering::SeqCst);
    let id_ver = shard.id_ver();
    *shard.compaction_priority.write().unwrap() = Some(CompactionPriority::DestroyRange);
    shard.col_snap_version.store(1, Ordering::SeqCst);
    engine.trigger_compact(id_ver);
    info!("trigger columnar destroy range compaction {}", shard.tag());
    let ok = try_wait(
        || {
            shard
                .get_data()
                .col_levels
                .levels
                .iter()
                .any(|l| l.files.is_empty())
        },
        5,
    );
    assert!(ok, "columnar destroy range compaction failed");
}

#[rstest]
#[case::inner_key_off_enable(true)]
#[case::inner_key_off_disable(false)]
fn test_columnar_truncate_ts(#[case] enable_inner_key_off: bool) {
    ::test_util::init_log_for_test();
    let keyspace_id = KEYSPACE_ID;
    let table_id = 30;
    let mut file_id = 100;
    let mut allocate_id = || {
        file_id += 1;
        file_id
    };
    let (engine, apply_tx) = new_test_engine_opt(enable_inner_key_off, DEF_BLOCK_SIZE, "");
    let shard_id = prepare_table_region(&engine, &apply_tx, keyspace_id, table_id);
    let shard = engine.get_shard(shard_id).unwrap();
    let schema = new_schema(table_id, false);
    let schemas = vec![schema.clone()];
    let schema_version = 10;
    let schema_file_data = build_schema_file(keyspace_id, schema_version, schemas);
    let fs = engine.fs.clone();
    let schema_raw_file = Arc::new(InMemFile::new(allocate_id(), Bytes::from(schema_file_data)));
    fs.get_runtime()
        .block_on(
            fs.create(
                schema_raw_file.id(),
                schema_raw_file
                    .read(0, schema_raw_file.size() as usize)
                    .unwrap(),
                dfs::Options::default().with_type(FileType::Schema),
            ),
        )
        .unwrap();
    let schema_file = SchemaFile::open(schema_raw_file).unwrap();
    let opts = dfs::Options::default().with_type(FileType::Columnar);
    let (l0_tbl_0, _) = build_table(allocate_id(), enable_inner_key_off, &schema, 0, 600, 500);
    let (l0_tbl_1, _) = build_table(allocate_id(), enable_inner_key_off, &schema, 300, 900, 500);
    let (l0_tbl_2, _) = build_table(
        allocate_id(),
        enable_inner_key_off,
        &schema,
        1000,
        2000,
        500,
    );
    let (l1_tbl_0, _) = build_table(allocate_id(), enable_inner_key_off, &schema, 0, 600, 300);
    let (l1_tbl_1, _) = build_table(allocate_id(), enable_inner_key_off, &schema, 300, 900, 400);
    let (l1_tbl_2, _) = build_table(
        allocate_id(),
        enable_inner_key_off,
        &schema,
        1000,
        1500,
        400,
    );
    let (l2_tbl_0, _) = build_table(allocate_id(), enable_inner_key_off, &schema, 0, 1000, 100);
    let (l2_tbl_1, _) = build_table(
        allocate_id(),
        enable_inner_key_off,
        &schema,
        1000,
        2000,
        100,
    );
    for file in [
        &l0_tbl_0, &l0_tbl_1, &l0_tbl_2, &l1_tbl_0, &l1_tbl_1, &l1_tbl_2, &l2_tbl_0, &l2_tbl_1,
    ] {
        info!("build file_id: {}, file size: {}", file.id(), file.size());
        fs.get_runtime()
            .block_on(fs.create(file.id(), file.read(0, file.size() as usize).unwrap(), opts))
            .unwrap()
    }

    let mut col_levels = ColumnarLevels::new();
    col_levels.add_file(0, ColumnarFile::open(l0_tbl_0).unwrap());
    col_levels.add_file(0, ColumnarFile::open(l0_tbl_1).unwrap());
    col_levels.add_file(0, ColumnarFile::open(l0_tbl_2).unwrap());
    col_levels.add_file(1, ColumnarFile::open(l1_tbl_0).unwrap());
    col_levels.add_file(1, ColumnarFile::open(l1_tbl_1).unwrap());
    col_levels.add_file(1, ColumnarFile::open(l1_tbl_2).unwrap());
    col_levels.add_file(2, ColumnarFile::open(l2_tbl_0).unwrap());
    col_levels.add_file(2, ColumnarFile::open(l2_tbl_1).unwrap());

    let data = ShardData::new(
        shard.range.clone(),
        vec![CfTable::new()],
        vec![],
        Arc::new(HashMap::default()),
        [
            ShardCf::new(WRITE_CF),
            ShardCf::new(LOCK_CF),
            ShardCf::new(EXTRA_CF),
        ],
        HashMap::new(),
        vec![],
        RegionLimiter::dummy(),
        shard.get_data().update_counter + 1,
        Some(schema_file),
        col_levels,
    );
    shard.set_data(data);
    shard.initial_flushed.store(true, Ordering::SeqCst);
    let id_ver = shard.id_ver();
    shard.pending_ops.write().unwrap().truncate_ts = Some(TruncateTs::from(200));
    *shard.compaction_priority.write().unwrap() = Some(CompactionPriority::TruncateTs);
    shard.col_snap_version.store(1, Ordering::SeqCst);
    engine.trigger_compact(id_ver);
    info!("trigger columnar truncate ts compaction {}", shard.tag());
    let ok = try_wait(
        || {
            shard.get_data().col_levels.levels[0].files.is_empty()
                && shard.get_data().col_levels.levels[1].files.is_empty()
                && shard.get_data().col_levels.levels[2].files.len() == 2
        },
        5,
    );
    assert!(ok, "columnar truncate_ts compaction failed");
    let snap = shard.new_snap_access();
    let mut mvcc_reader = snap
        .new_columnar_mvcc_reader(table_id, &schema.columns, u64::MAX)
        .unwrap();
    mvcc_reader.set_int_handle_range(0, Some(3000)).unwrap();
    let mut no_txn_id_schema_buf = schema.to_schema_buf();
    no_txn_id_schema_buf.txn_id_column = None;
    let mut block = Block::new(&no_txn_id_schema_buf.into());
    let counts = mvcc_reader.read_block(&mut block, usize::MAX).unwrap();
    for i in 0..counts {
        assert_eq!(100, block.versions.get_version(i));
    }
}

#[rstest]
#[case::inner_key_off_enable(true)]
#[case::inner_key_off_disable(false)]
fn test_columnar_trim_over_bound(#[case] enable_inner_key_off: bool) {
    ::test_util::init_log_for_test();
    let keyspace_id = KEYSPACE_ID;
    let table_id = 30;
    let mut file_id = 100;
    let mut allocate_id = || {
        file_id += 1;
        file_id
    };
    let (engine, apply_tx) = new_test_engine_opt(enable_inner_key_off, DEF_BLOCK_SIZE, "");
    let shard_id = prepare_table_region(&engine, &apply_tx, keyspace_id, table_id);
    let shard = engine.get_shard(shard_id).unwrap();
    let schema = new_schema(table_id, false);
    let schemas = vec![schema.clone()];
    let schema_version = 10;
    let schema_file_data = build_schema_file(keyspace_id, schema_version, schemas);
    let fs = engine.fs.clone();
    let schema_raw_file = Arc::new(InMemFile::new(allocate_id(), Bytes::from(schema_file_data)));
    fs.get_runtime()
        .block_on(
            fs.create(
                schema_raw_file.id(),
                schema_raw_file
                    .read(0, schema_raw_file.size() as usize)
                    .unwrap(),
                dfs::Options::default().with_type(FileType::Schema),
            ),
        )
        .unwrap();
    let schema_file = SchemaFile::open(schema_raw_file).unwrap();
    let opts = dfs::Options::default().with_type(FileType::Columnar);
    let (l0_tbl_0, _) = build_table(allocate_id(), enable_inner_key_off, &schema, 0, 600, 500);
    let (l0_tbl_1, _) = build_table(allocate_id(), enable_inner_key_off, &schema, 300, 900, 600);
    let (l0_tbl_2, _) = build_table(
        allocate_id(),
        enable_inner_key_off,
        &schema,
        1000,
        2000,
        510,
    );
    let (l1_tbl_0, _) = build_table(allocate_id(), enable_inner_key_off, &schema, 0, 600, 300);
    let (l1_tbl_1, _) = build_table(allocate_id(), enable_inner_key_off, &schema, 300, 900, 410);
    let (l1_tbl_2, _) = build_table(
        allocate_id(),
        enable_inner_key_off,
        &schema,
        1000,
        1500,
        420,
    );
    let (l2_tbl_0, _) = build_table(allocate_id(), enable_inner_key_off, &schema, 0, 1000, 100);
    let (l2_tbl_1, _) = build_table(
        allocate_id(),
        enable_inner_key_off,
        &schema,
        1000,
        2000,
        110,
    );
    for file in [
        &l0_tbl_0, &l0_tbl_1, &l0_tbl_2, &l1_tbl_0, &l1_tbl_1, &l1_tbl_2, &l2_tbl_0, &l2_tbl_1,
    ] {
        info!("build file_id: {}, file size: {}", file.id(), file.size());
        fs.get_runtime()
            .block_on(fs.create(file.id(), file.read(0, file.size() as usize).unwrap(), opts))
            .unwrap()
    }

    let mut col_levels = ColumnarLevels::new();
    col_levels.add_file(0, ColumnarFile::open(l0_tbl_0).unwrap());
    col_levels.add_file(0, ColumnarFile::open(l0_tbl_1).unwrap());
    col_levels.add_file(0, ColumnarFile::open(l0_tbl_2).unwrap());
    col_levels.add_file(1, ColumnarFile::open(l1_tbl_0).unwrap());
    col_levels.add_file(1, ColumnarFile::open(l1_tbl_1).unwrap());
    col_levels.add_file(1, ColumnarFile::open(l1_tbl_2).unwrap());
    col_levels.add_file(2, ColumnarFile::open(l2_tbl_0).unwrap());
    col_levels.add_file(2, ColumnarFile::open(l2_tbl_1).unwrap());

    let data = ShardData::new(
        shard.range.clone(),
        vec![CfTable::new()],
        vec![],
        Arc::new(HashMap::default()),
        [
            ShardCf::new(WRITE_CF),
            ShardCf::new(LOCK_CF),
            ShardCf::new(EXTRA_CF),
        ],
        HashMap::new(),
        vec![],
        RegionLimiter::dummy(),
        shard.get_data().update_counter + 1,
        Some(schema_file),
        col_levels,
    );
    shard.set_data(data);
    // Split shard by handle key 800
    let split_key = [keyspace_prefix(keyspace_id), encode_row_key(table_id, 800)].concat();
    let mut splitter = Splitter::new(vec![split_key], IdVer::new(4, 4), 5, apply_tx);
    let handle = thread::spawn(move || {
        splitter.run();
    });
    let ok = try_wait(|| engine.shards.len() == 6, 2);
    assert!(ok, "split failed");
    handle.join().unwrap();

    let shard = engine.get_shard(shard_id).unwrap();
    // Read before trim over bound data.
    shard.initial_flushed.store(true, Ordering::SeqCst);
    shard.col_snap_version.store(1, Ordering::SeqCst);
    let snap = shard.new_snap_access();
    let mut mvcc_reader = snap
        .new_columnar_mvcc_reader(table_id, &schema.columns, u64::MAX)
        .unwrap();
    mvcc_reader.set_int_handle_range(0, Some(3000)).unwrap();
    let mut no_txn_id_schema_buf = schema.to_schema_buf();
    no_txn_id_schema_buf.txn_id_column = None;
    let no_txn_id_schema = Schema::new(no_txn_id_schema_buf);
    let mut block = Block::new(&no_txn_id_schema);
    let read_before_trim = mvcc_reader.read_block(&mut block, usize::MAX).unwrap();

    let id_ver = shard.id_ver();
    shard.pending_ops.write().unwrap().trim_over_bound = true;
    *shard.compaction_priority.write().unwrap() = Some(CompactionPriority::TrimOverBound);
    engine.trigger_compact(id_ver);
    info!(
        "trigger columnar trim over bound compaction {}",
        shard.tag()
    );
    let ok = try_wait(|| shard.get_data().get_col_table_counts(0) == 2, 5);
    assert!(ok, "columnar trim over bound compaction failed");
    std::thread::sleep(Duration::from_secs(2));
    shard.col_snap_version.store(1, Ordering::SeqCst);
    let snap = shard.new_snap_access();
    let mut mvcc_reader = snap
        .new_columnar_mvcc_reader(table_id, &schema.columns, u64::MAX)
        .unwrap();
    mvcc_reader.set_int_handle_range(0, Some(3000)).unwrap();
    let mut block = Block::new(&no_txn_id_schema);
    let read_after_trim = mvcc_reader.read_block(&mut block, usize::MAX).unwrap();
    info!(
        "read_before_trim: {}, read_after_trim: {}",
        read_before_trim, read_after_trim
    );
    assert!(read_before_trim > read_after_trim);
    for i in 0..read_after_trim {
        let handle = i64::from_le_bytes(
            std::convert::TryInto::try_into(block.handles.get_not_null_value(i)).unwrap(),
        );
        assert!(handle >= 800);
        let version = block.versions.get_version(i);
        if handle == 800 {
            assert!(version == 600 || version == 599 || version == 598);
        }
    }
    // Check old version exists
    let mut mvcc_reader = snap
        .new_columnar_mvcc_reader(table_id, &schema.columns, 300)
        .unwrap();
    mvcc_reader.set_int_handle_range(0, Some(3000)).unwrap();
    let mut block = Block::new(&no_txn_id_schema);
    let read_old_version = mvcc_reader.read_block(&mut block, usize::MAX).unwrap();
    for i in 0..read_old_version {
        let handle = i64::from_le_bytes(
            std::convert::TryInto::try_into(block.handles.get_not_null_value(i)).unwrap(),
        );
        if handle == 800 {
            assert!(block.versions.get_version(i) <= 100);
        }
    }
}

fn verify_columnar_with_blocks(b1: &Block, b2: &Block) {
    assert_eq!(b1.length(), b2.length());
    let len = b1.length();
    for i in 0..len {
        assert_eq!(
            b1.handles.get_not_null_value(i),
            b2.handles.get_not_null_value(i),
            "{}",
            i
        );
        assert_eq!(b1.versions.get_version(i), b2.versions.get_version(i));
        assert_eq!(b1.versions.is_null(i), b2.versions.is_null(i));
        assert_eq!(b1.columns[0].is_null(i), b2.columns[0].is_null(i));
        assert_eq!(b1.columns[1].is_null(i), b2.columns[1].is_null(i));
        assert_eq!(
            b1.columns[0].get_value(i).unwrap().get_i64_le(),
            b2.columns[0].get_value(i).unwrap().get_i64_le()
        );
        assert_eq!(
            b1.columns[1].get_value(i).unwrap(),
            b2.columns[1].get_value(i).unwrap()
        );
    }
}

fn new_sst_table_for_columnar(
    engine: &TestEngine,
    id: u64,
    table_id: i64,
    begin: usize,
    end: usize,
    version: u64,
    saved_vals: &mut Vec<Rc<Vec<u8>>>,
) -> SsTable {
    let block_size = engine.opts.table_builder_options.block_size;
    let comp_tp = engine.opts.table_builder_options.compression_tps[0];
    let comp_lvl = engine.opts.table_builder_options.compression_lvl;
    let fs = engine.fs.clone();

    let mut builder = table::sstable::builder::Builder::new(
        id,
        block_size,
        comp_tp,
        comp_lvl,
        ChecksumType::Crc32c,
        None,
    );
    let ctx = Mutex::new(EvalContext::default());
    for i in begin..end {
        let key = engine.key_builder.gen_row_inner_key(table_id, i);
        let val = Rc::new(engine.key_builder.gen_row_val(&ctx, i));
        // Save the val with Rc, make sure the val_rc stay valid during the iterator
        // lifecycle.
        saved_vals.push(val.clone());
        let val = table::Value::new_with_meta_version(0, version, 0, &val);
        builder.add(key.as_ref(), &val, None);
    }
    let mut data_buf = Vec::new();
    builder.finish(0, &mut data_buf);
    let data = Bytes::from(data_buf);
    let opts = dfs::Options::default();
    let runtime = fs.get_runtime();
    runtime.block_on(fs.create(id, data.clone(), opts)).unwrap();
    let file = InMemFile::new(id, data);
    SsTable::new(Arc::new(file), None, true, None).unwrap()
}

#[rstest]
#[case::base(None, true)]
#[case::enc(Some(generate_encryption_key()), true)]
#[case::disable_key_off(None, false)]
#[case::enc_disable_key_off(Some(generate_encryption_key()), false)]
fn test_txn_file(#[case] enc_key: Option<EncryptionKey>, #[case] enable_inner_key_off: bool) {
    ::test_util::init_log_for_test();
    let enc_key = enc_key.as_ref();
    let (engine, tx) = new_test_engine_opt(enable_inner_key_off, DEF_BLOCK_SIZE, TABLE_KEY_PREFIX);
    let kb = engine.key_builder();
    let chunk_id = 200;
    build_txn_chunk(&engine, 200, 300, chunk_id, enc_key, enable_inner_key_off);
    let primary = kb.i_to_outer_key(0);
    let mut wb = WriteBatch::new(1, 0);
    let txn_file_refs = make_txn_file_refs(
        1000,
        vec![chunk_id],
        make_lock_prefix(primary.clone(), 200),
        vec![],
    );
    wb.set_property(TXN_FILE_REF, &txn_file_refs);
    engine
        .txn_chunk_mgr
        .prepare(chunk_id, enc_key.cloned())
        .unwrap();
    write_data(wb, &tx);
    verify_lock(&engine, 200, 300, kb);

    // rollback the txn file.
    let txn_file_refs = make_txn_file_refs(1000, vec![chunk_id], vec![], make_user_meta(1000, 0));
    let mut wb = WriteBatch::new(1, 0);
    wb.set_property(TXN_FILE_REF, &txn_file_refs);
    write_data(wb, &tx);
    let shard = engine.get_shard(1).unwrap();
    assert!(shard.get_property(TXN_FILE_REF).unwrap().is_empty());
    assert!(shard.get_txn_chunks().is_empty());

    // concurrent txn files.
    let txn1_chunk_id = 201;
    build_txn_chunk(
        &engine,
        200,
        300,
        txn1_chunk_id,
        enc_key,
        enable_inner_key_off,
    );
    let txn1_start_ts = 1003;
    let txn1_lock = make_txn_file_refs(
        txn1_start_ts,
        vec![txn1_chunk_id],
        make_lock_prefix(primary.clone(), txn1_start_ts),
        vec![],
    );
    let mut wb = WriteBatch::new(1, 0);
    wb.set_property(TXN_FILE_REF, &txn1_lock);
    engine
        .txn_chunk_mgr
        .prepare_txn_chunks(vec![txn1_chunk_id], enc_key.cloned())
        .unwrap();
    write_data(wb, &tx);
    let txn2_chunk_id = 202;
    build_txn_chunk(
        &engine,
        300,
        400,
        txn2_chunk_id,
        enc_key,
        enable_inner_key_off,
    );
    let txn2_start_ts = 1004;
    let txn2_lock = make_txn_file_refs(
        txn2_start_ts,
        vec![txn2_chunk_id],
        make_lock_prefix(primary.clone(), txn2_start_ts),
        vec![],
    );
    let mut wb = WriteBatch::new(1, 0);
    wb.set_property(TXN_FILE_REF, &txn2_lock);
    engine
        .txn_chunk_mgr
        .prepare_txn_chunks(vec![txn2_chunk_id], enc_key.cloned())
        .unwrap();
    write_data(wb, &tx);
    verify_lock(&engine, 200, 400, kb);

    let txn1_commit = make_txn_file_refs(
        txn1_start_ts,
        vec![txn1_chunk_id],
        vec![],
        make_user_meta(txn1_start_ts, txn1_start_ts + 2),
    );
    let mut wb = WriteBatch::new(1, 0);
    wb.set_property(TXN_FILE_REF, &txn1_commit);
    write_data(wb, &tx);
    verify_write(&engine, 200, 300, kb);
    verify_lock(&engine, 300, 400, kb);

    let conflict_chunk_id = 203;
    let conflict_start_ts = 999;
    build_txn_chunk(
        &engine,
        250,
        350,
        conflict_chunk_id,
        enc_key,
        enable_inner_key_off,
    );
    engine
        .txn_chunk_mgr
        .prepare(conflict_chunk_id, enc_key.cloned())
        .unwrap();
    let conflict_chunk = engine.txn_chunk_mgr.get(conflict_chunk_id).unwrap();
    let lower_bound = InnerKey::from_inner_buf(b"");
    let upper_bound = InnerKey::from_inner_buf(GLOBAL_SHARD_END_KEY);
    let conflict_ctx = TxnCtx::new(
        vec![].into(),
        make_lock_prefix(primary, conflict_start_ts).into(),
        conflict_start_ts,
        lower_bound,
        upper_bound,
    );
    let conflict_txn_file = TxnFile::new(
        TxnFileId::new(1, 1, conflict_start_ts),
        vec![conflict_chunk],
        conflict_ctx.clone(),
    )
    .unwrap();
    let snap = engine.get_snap_access(1).unwrap();
    let (key, um) = snap
        .get_txn_file_conflict_write(&conflict_txn_file)
        .unwrap();
    assert_eq!(key, kb.i_to_outer_key(250));
    assert_eq!(um.start_ts, txn1_start_ts);
    let (key, lock) = snap.get_txn_file_conflict_lock(&conflict_txn_file).unwrap();
    assert_eq!(key, kb.i_to_outer_key(300));
    assert_eq!(lock.ts.into_inner(), txn2_start_ts);

    let empty_txn_file = TxnFile::new(
        TxnFileId::new(1, 1, conflict_start_ts),
        vec![],
        conflict_ctx,
    )
    .unwrap();
    assert!(snap.get_txn_file_conflict_write(&empty_txn_file).is_none());
    assert!(snap.get_txn_file_conflict_lock(&empty_txn_file).is_none());

    verify_write(&engine, 200, 300, kb);
    verify_lock(&engine, 300, 400, kb);

    let txn2_commit = make_txn_file_refs(
        txn2_start_ts,
        vec![txn2_chunk_id],
        vec![],
        make_user_meta(txn2_start_ts, txn2_start_ts + 2),
    );
    let mut wb = WriteBatch::new(1, 0);
    wb.set_property(TXN_FILE_REF, &txn2_commit);
    write_data(wb, &tx);
    verify_write(&engine, 200, 400, kb);
}

#[rstest]
#[case::base(None, true)]
#[case::enc(Some(generate_encryption_key()), true)]
#[case::disable_key_off(None, false)]
#[case::enc_disable_key_off(Some(generate_encryption_key()), false)]
fn test_txn_file_multiple(
    #[case] enc_key: Option<EncryptionKey>,
    #[case] enable_inner_key_off: bool,
) {
    ::test_util::init_log_for_test();
    let enc_key = enc_key.as_ref();
    let (engine, tx) = new_test_engine_opt(enable_inner_key_off, DEF_BLOCK_SIZE, TABLE_KEY_PREFIX);
    let kb = engine.key_builder();

    let chunks_id: Vec<u64> = (100..500).step_by(10).collect();
    let primary = kb.i_to_outer_key(0);
    let start_ts = 2000;
    for &chunk_id in &chunks_id {
        let start = chunk_id as usize;
        build_txn_chunk(
            &engine,
            start,
            start + 10,
            chunk_id,
            enc_key,
            enable_inner_key_off,
        );
    }
    let mut wb = WriteBatch::new(1, 0);
    let txn_file_refs = make_txn_file_refs(
        start_ts,
        chunks_id.clone(),
        make_lock_prefix(primary, start_ts),
        vec![],
    );
    wb.set_property(TXN_FILE_REF, &txn_file_refs);
    engine
        .txn_chunk_mgr
        .prepare_txn_chunks(chunks_id.clone(), enc_key.cloned())
        .unwrap();
    write_data(wb, &tx);
    verify_lock(&engine, 100, 500, kb);

    let txn4_commit = make_txn_file_refs(
        start_ts,
        chunks_id,
        vec![],
        make_user_meta(start_ts, start_ts + 2),
    );
    let mut wb = WriteBatch::new(1, 0);
    wb.set_property(TXN_FILE_REF, &txn4_commit);
    write_data(wb, &tx);
    verify_write(&engine, 100, 500, kb);
}

fn build_txn_chunk(
    engine: &TestEngine,
    start: usize,
    end: usize,
    id: u64,
    enc_key: Option<&EncryptionKey>,
    enable_inner_key_off: bool,
) {
    let kb = engine.key_builder();
    let mut chunk_builder =
        TxnChunkBuilder::new(id, 10, enc_key.cloned(), KEYSPACE_ID, enable_inner_key_off);
    for i in start..end {
        let key = kb.i_to_key(i);
        chunk_builder.add_entry(NoPrefixKey(&key), OP_PUT, &key);
    }
    let mut buf = vec![];
    chunk_builder.finish(&mut buf);
    let runtime = engine.fs.get_runtime();
    let fs = engine.fs.clone();
    let opts = dfs::Options::default().with_type(FileType::TxnChunk);
    runtime.block_on(fs.create(id, buf.into(), opts)).unwrap();
}

fn make_txn_file_refs(
    start_ts: u64,
    chunks_id: Vec<u64>,
    lock_prefix: Vec<u8>,
    user_meta: Vec<u8>,
) -> Vec<u8> {
    let mut txn_file_ref = TxnFileRef::new();
    txn_file_ref.set_chunk_ids(chunks_id);
    txn_file_ref.set_shard_ver(1);
    txn_file_ref.set_start_ts(start_ts);
    if !lock_prefix.is_empty() {
        txn_file_ref.set_lock_val_prefix(lock_prefix);
    }
    if !user_meta.is_empty() {
        txn_file_ref.set_user_meta(user_meta);
    }
    txn_file_ref.set_inner_lower_bound(vec![]);
    txn_file_ref.set_inner_upper_bound(GLOBAL_SHARD_END_KEY.to_vec());
    let mut txn_file_refs = TxnFileRefs::new();
    txn_file_refs.mut_txn_file_refs().push(txn_file_ref);
    txn_file_refs.write_to_bytes().unwrap()
}

fn make_lock_prefix(primary: Vec<u8>, start_ts: u64) -> Vec<u8> {
    let min_commit_ts = start_ts + 1;
    let mut lock = txn_types::Lock::new(
        txn_types::LockType::Put,
        primary,
        start_ts.into(),
        3000,
        None,
        0.into(),
        100,
        min_commit_ts.into(),
    );
    lock.is_txn_file = true;
    lock.to_bytes()
}

fn make_user_meta(start_ts: u64, commit_ts: u64) -> Vec<u8> {
    UserMeta::new(start_ts, commit_ts).to_array().to_vec()
}

fn verify_lock(engine: &TestEngine, start: usize, end: usize, kb: &KeyBuilder) {
    let snap = engine.get_snap_access(1).unwrap();
    for i in start..end {
        let key = kb.i_to_outer_key(i);
        let val = kb.i_to_key(i);
        let item = snap.get(LOCK_CF, &key, 0);
        assert!(item.is_valid());
        let lock = txn_types::Lock::parse(item.get_value()).unwrap();
        assert_eq!(lock.primary, kb.i_to_outer_key(0));
        assert!(lock.is_txn_file);
        assert_eq!(lock.short_value.unwrap(), val.as_slice());
    }
    let mut it = snap.new_iterator(LOCK_CF, false, false, None, false);
    it.rewind();
    let mut i = start;
    while it.valid() {
        let lock = txn_types::Lock::parse(it.val()).unwrap();
        let val = kb.i_to_key(i);
        assert_eq!(lock.short_value.unwrap(), val.as_slice());
        it.next();
        i += 1;
    }
    assert_eq!(i, end);
}

fn verify_write(engine: &TestEngine, start: usize, end: usize, kb: &KeyBuilder) {
    let snap = engine.get_snap_access(1).unwrap();
    for i in start..end {
        let key = kb.i_to_outer_key(i);
        let val = kb.i_to_key(i);
        let item = snap.get(WRITE_CF, &key, u64::MAX);
        assert_eq!(item.get_value(), val.as_slice());
    }
    let mut it = snap.new_iterator(WRITE_CF, false, false, None, false);
    it.rewind();
    let mut i = start;
    while it.valid() {
        let val = kb.i_to_key(i);
        assert_eq!(it.val(), val.as_slice());
        it.next();
        i += 1;
    }
    assert_eq!(i, end);
}

fn print_locks(snap: &SnapAccess, all_versions: bool, read_ts: Option<u64>) {
    let mut it = snap.new_iterator(LOCK_CF, false, all_versions, read_ts, false);
    it.rewind();
    while it.valid() {
        debug!(
            "key: {:?}, version: {:?}, meta: {:?}",
            tikv_util::escape(it.key()),
            it.version(),
            it.meta()
        );
        it.next();
    }
}

fn keyspace_prefix(keyspace_id: u32) -> Vec<u8> {
    let mut prefix = keyspace_id.to_be_bytes().to_vec();
    prefix[0] = b'x';
    prefix
}

#[inline]
fn encode_i64_to_comparable_u64(v: i64) -> u64 {
    (v as u64) ^ 1 << 63_u64
}

fn prepare_table_region(
    engine: &TestEngine,
    apply_tx: &mpsc::Sender<ApplyTask>,
    keyspace_id: u32,
    table_id: i64,
) -> u64 {
    // Split keyspace
    let keyspace_1_prefix = keyspace_prefix(keyspace_id);
    let mut table_1_prefix = keyspace_1_prefix.clone();
    table_1_prefix.extend_from_slice(&[b't']);
    table_1_prefix.extend_from_slice(
        encode_i64_to_comparable_u64(table_id)
            .to_be_bytes()
            .as_slice(),
    );
    let mut table_2_prefix = keyspace_1_prefix;
    table_2_prefix.extend_from_slice(&[b't']);
    table_2_prefix.extend_from_slice(
        encode_i64_to_comparable_u64(table_id + 1)
            .to_be_bytes()
            .as_slice(),
    );
    let split_keys = vec![
        keyspace_prefix(keyspace_id),
        table_1_prefix,
        table_2_prefix,
        keyspace_prefix(keyspace_id + 1),
    ];
    let mut splitter = Splitter::new(split_keys, IdVer::new(1, 1), 1, apply_tx.clone());
    let handle = thread::spawn(move || {
        splitter.run();
    });
    let ok = try_wait(|| engine.shards.len() == 5, 2);
    assert!(ok, "split failed");
    handle.join().unwrap();
    4
}

#[derive(Clone)]
struct TestMetaChangeListener {
    sender: mpsc::Sender<pb::ChangeSet>,
}

impl MetaChangeListener for TestMetaChangeListener {
    fn on_change_set(&self, cs: pb::ChangeSet) {
        info!("on meta change listener");
        self.sender.send(cs).unwrap();
    }
}

#[derive(Clone)]
struct EngineTester {
    core: Arc<EngineTesterCore>,
}

impl Deref for EngineTester {
    type Target = EngineTesterCore;
    fn deref(&self) -> &Self::Target {
        &self.core
    }
}

impl EngineTester {
    fn new(enable_inner_key_off: bool, block_size: usize) -> Self {
        let initial_cs = new_initial_cs(enable_inner_key_off);
        let initial_meta = ShardMeta::new(1, &initial_cs);
        let metas = dashmap::DashMap::new();
        metas.insert(1, Arc::new(initial_meta));
        let tmp_dir = TempDir::new().unwrap();
        let opts = new_test_options(tmp_dir.path(), enable_inner_key_off, block_size);
        let config = KvEngineConfig::default();

        Self {
            core: Arc::new(EngineTesterCore {
                _tmp_dir: tmp_dir,
                metas,
                fs: Arc::new(InMemFs::new()),
                opts: Arc::new(opts),
                config,
                id: AtomicU64::new(0),
            }),
        }
    }
}

struct EngineTesterCore {
    _tmp_dir: TempDir,
    metas: dashmap::DashMap<u64, Arc<ShardMeta>>,
    fs: Arc<dfs::InMemFs>,
    opts: Arc<Options>,
    config: KvEngineConfig,
    id: AtomicU64,
}

impl MetaIterator for EngineTester {
    fn iterate<F>(&mut self, mut f: F) -> Result<()>
    where
        F: FnMut(kvenginepb::ChangeSet),
    {
        for meta in &self.metas {
            f(meta.value().to_change_set())
        }
        Ok(())
    }

    fn engine_id(&self) -> u64 {
        1
    }
}

impl RecoverHandler for EngineTester {
    fn recover(&self, _engine: &Engine, _shard: &Arc<Shard>, _info: &ShardMeta) -> Result<()> {
        Ok(())
    }
}

impl IdAllocator for EngineTesterCore {
    fn alloc_id(&self, count: usize) -> Result<Vec<u64>> {
        let start_id = self
            .id
            .fetch_add(count as u64, std::sync::atomic::Ordering::Relaxed)
            + 1;
        let end_id = start_id + count as u64;
        let mut ids = Vec::with_capacity(count);
        for id in start_id..end_id {
            ids.push(id);
        }
        Ok(ids)
    }
}

struct MetaListener {
    applier_tx: mpsc::Sender<ApplyTask>,
    meta_rx: mpsc::Receiver<pb::ChangeSet>,
}

impl MetaListener {
    fn new(meta_rx: mpsc::Receiver<pb::ChangeSet>, applier_tx: mpsc::Sender<ApplyTask>) -> Self {
        Self {
            meta_rx,
            applier_tx,
        }
    }

    fn run(&self) {
        loop {
            let cs = unwrap_or_return!(self.meta_rx.recv(), "meta_listener_a");
            let (tx, rx) = mpsc::bounded(1);
            let task = ApplyTask::new_cs(cs, tx);
            self.applier_tx.send(task).unwrap();
            let res = unwrap_or_return!(rx.recv(), "meta_listener_b");
            unwrap_or_return!(res, "meta_listener_c");
        }
    }
}

struct Applier {
    engine: Engine,
    task_rx: mpsc::Receiver<ApplyTask>,
}

impl Applier {
    fn new(engine: Engine, task_rx: mpsc::Receiver<ApplyTask>) -> Self {
        Self { engine, task_rx }
    }

    fn run(&self) {
        let mut seq = 2;
        loop {
            let mut task = unwrap_or_return!(self.task_rx.recv(), "apply recv task");
            seq += 1;
            if let Some(wb) = task.wb.as_mut() {
                wb.set_sequence(seq);
                self.engine.write(wb);
            }
            if let Some(mut cs) = task.cs.take() {
                cs.set_sequence(seq);
                if cs.has_split() {
                    let mut ids = vec![];
                    for new_shard in cs.get_split().get_new_shards() {
                        ids.push(new_shard.shard_id);
                    }
                    unwrap_or_return!(self.engine.split(cs, 1), "apply split");
                    for id in ids {
                        let shard = self.engine.get_shard(id).unwrap();
                        shard.set_active(true);
                    }
                    info!("applier executed split");
                } else {
                    self.engine.meta_committed(&cs, false);
                    unwrap_or_return!(
                        self.engine.apply_change_set(
                            self.engine
                                .prepare_change_set(cs, false, None, None)
                                .unwrap()
                        ),
                        "applier apply changeset"
                    );
                }
            }
            task.result_tx.send(Ok(seq)).unwrap();
        }
    }
}

struct ApplyTask {
    wb: Option<WriteBatch>,
    cs: Option<pb::ChangeSet>,
    result_tx: mpsc::Sender<Result<u64 /* write_sequence */>>,
}

impl ApplyTask {
    fn new_cs(cs: pb::ChangeSet, result_tx: mpsc::Sender<Result<u64>>) -> Self {
        Self {
            wb: None,
            cs: Some(cs),
            result_tx,
        }
    }

    fn new_wb(wb: WriteBatch, result_tx: mpsc::Sender<Result<u64>>) -> Self {
        Self {
            wb: Some(wb),
            cs: None,
            result_tx,
        }
    }
}

struct Splitter {
    apply_sender: mpsc::Sender<ApplyTask>,
    keys: Vec<Vec<u8>>,
    current_shard_id: u64,
    shard_ver: u64,
    new_id: u64,
}

#[allow(dead_code)]
impl Splitter {
    fn new(
        keys: Vec<Vec<u8>>,
        current_shard_id_ver: IdVer,
        new_id_base: u64,
        apply_sender: mpsc::Sender<ApplyTask>,
    ) -> Self {
        Self {
            keys,
            apply_sender,
            current_shard_id: current_shard_id_ver.id,
            shard_ver: current_shard_id_ver.ver,
            new_id: new_id_base,
        }
    }

    fn run(&mut self) {
        let keys = self.keys.clone();
        for key in keys {
            thread::sleep(Duration::from_millis(200));
            self.new_id += 1;
            self.split(key.clone(), vec![self.new_id, self.current_shard_id]);
        }
    }

    fn send_task(&mut self, cs: pb::ChangeSet) {
        let (tx, rx) = mpsc::bounded(1);
        let task = ApplyTask {
            cs: Some(cs),
            wb: None,
            result_tx: tx,
        };
        self.apply_sender.send(task).unwrap();
        let res = unwrap_or_return!(rx.recv(), "splitter recv");
        res.unwrap();
    }

    fn split(&mut self, key: Vec<u8>, new_ids: Vec<u64>) {
        let mut cs = pb::ChangeSet::new();
        cs.set_shard_id(self.current_shard_id);
        cs.set_shard_ver(self.shard_ver);
        let mut finish_split = pb::Split::new();
        finish_split.set_keys(protobuf::RepeatedField::from_vec(vec![key]));
        let mut new_shards = Vec::new();
        for new_id in &new_ids {
            let mut new_shard = pb::Properties::new();
            new_shard.set_shard_id(*new_id);
            new_shards.push(new_shard);
        }
        finish_split.set_new_shards(protobuf::RepeatedField::from_vec(new_shards));
        cs.set_split(finish_split);
        self.send_task(cs);
        self.shard_ver += 1;
    }
}

fn new_initial_cs(enable_inner_key_off: bool) -> pb::ChangeSet {
    let mut cs = pb::ChangeSet::new();
    cs.set_shard_id(1);
    cs.set_shard_ver(1);
    cs.set_sequence(1);
    let mut snap = pb::Snapshot::new();
    snap.set_base_version(1);
    if enable_inner_key_off {
        let (start, end) = ApiV2::get_txn_keyspace_range(KEYSPACE_ID);
        snap.set_outer_start(start);
        snap.set_outer_end(end);
        snap.set_inner_key_off(KEYSPACE_PREFIX_LEN as u32);
    } else {
        snap.set_outer_end(GLOBAL_SHARD_END_KEY.to_vec());
    }
    let props = snap.mut_properties();
    props.shard_id = 1;
    cs.set_snapshot(snap);
    cs
}

fn new_test_options(
    path: impl AsRef<Path>,
    enable_inner_key_off: bool,
    block_size: usize,
) -> Options {
    let min_blob_size: u32 = match env::var("MIN_BLOB_SIZE") {
        Ok(val) => match val.trim().parse() {
            Ok(n) => n,
            Err(e) => {
                warn!("MIN_BLOB_SIZE=<number>, got {}", e);
                DEF_MIN_BLOB_SIZE
            }
        },
        Err(_) => DEF_MIN_BLOB_SIZE,
    };
    info!("MIN_BLOB_SIZE={}", min_blob_size);
    let mut opts = Options::default();
    opts.local_dir = path.as_ref().to_path_buf();
    opts.base_size = 64 << 10;
    opts.table_builder_options.block_size = block_size;
    opts.table_builder_options.max_table_size = 8 << 10;
    opts.table_builder_options.flush_split_l0 = true;
    opts.columnar_build_options.max_columnar_table_size = 1024;
    opts.columnar_build_options.pack_max_row_count = 9;
    opts.max_mem_table_size = 32 << 10; // mem-table size should be much larger than max_table_size.
    opts.num_compactors = 2;
    opts.blob_table_build_options.min_blob_size = min_blob_size;
    opts.max_del_range_delay = Duration::from_secs(1);
    opts.enable_inner_key_offset = enable_inner_key_off;
    opts
}

fn i_to_key(i: i32, min_blob_size: u32) -> String {
    if min_blob_size > 0 {
        // 3 -> strlen("key")
        format!("key{:0>1$}", i, min_blob_size as usize - 3)
    } else {
        format!("key{:0>1$}", i, 6)
    }
}

fn new_table(
    engine: &TestEngine,
    id: u64,
    begin: usize,
    end: usize,
    version: u64,
    del: bool,
    saved_vals: &mut Vec<Rc<Vec<u8>>>,
) -> SsTable {
    let block_size = engine.opts.table_builder_options.block_size;
    let comp_tp = engine.opts.table_builder_options.compression_tps[0];
    let comp_lvl = engine.opts.table_builder_options.compression_lvl;
    let fs = engine.fs.clone();

    let mut builder = table::sstable::builder::Builder::new(
        id,
        block_size,
        comp_tp,
        comp_lvl,
        ChecksumType::Crc32c,
        None,
    );
    for i in begin..end {
        let key = engine.key_builder.i_to_inner_key(i);
        let val = if del {
            table::Value::new_with_meta_version(BIT_DELETE, version, 0, &[])
        } else {
            let val = Rc::new(key.as_ref().repeat(2));
            // Save the val with Rc, make sure the val_rc stay valid during the iterator
            // lifecycle.
            saved_vals.push(val.clone());
            table::Value::new_with_meta_version(0, version, 0, &val)
        };
        builder.add(key.as_ref(), &val, None);
    }
    let mut data_buf = Vec::new();
    builder.finish(0, &mut data_buf);
    let data = Bytes::from(data_buf);
    let opts = dfs::Options::default();
    let runtime = fs.get_runtime();
    runtime.block_on(fs.create(id, data.clone(), opts)).unwrap();
    let file = InMemFile::new(id, data);
    SsTable::new(Arc::new(file), None, true, None).unwrap()
}

fn new_l0table_file(
    engine: &TestEngine,
    id: u64,
    begin: [usize; NUM_CFS],
    end: [usize; NUM_CFS],
    version: u64,
    del: [bool; NUM_CFS],
) -> Arc<dyn File> {
    let block_size = engine.opts.table_builder_options.block_size;
    let fs = engine.fs.clone();

    let mut builder = L0Builder::new(id, block_size, version, ChecksumType::Crc32c, None);
    for cf in 0..NUM_CFS {
        for i in begin[cf]..end[cf] {
            let key = engine.key_builder.i_to_inner_key(i);
            if del[cf] {
                let val = table::Value::new_with_meta_version(BIT_DELETE, version, 0, &[]);
                builder.add(cf, key.as_ref(), &val, None);
            } else {
                let val_buf = key.as_ref().repeat(2);
                let val = table::Value::new_with_meta_version(0, version, 0, &val_buf);
                builder.add(cf, key.as_ref(), &val, None);
            }
        }
    }
    let (_, data) = builder.finish();
    let opts = dfs::Options::default();
    let runtime = fs.get_runtime();
    runtime.block_on(fs.create(id, data.clone(), opts)).unwrap();
    Arc::new(InMemFile::new(id, data))
}

fn load_data(
    begin: usize,
    end: usize,
    version: u64,
    tx: mpsc::Sender<ApplyTask>,
    min_blob_size: u32,
) {
    let mut wb = WriteBatch::new(1, 0);
    for i in begin..end {
        let key = i_to_key(i as i32, min_blob_size);
        for cf in 0..3 {
            let val = key.repeat(cf + 2);
            let version = if cf == 1 { 0 } else { version };
            wb.put(cf, key.as_bytes(), val.as_bytes(), 0, &[], version);
        }
        if i % 100 == 99 {
            info!("load data {}:{}", i - 99, i);
            write_data(wb, &tx);
            wb = WriteBatch::new(1, 0);
            thread::sleep(Duration::from_millis(10));
        }
    }
    if wb.num_entries() > 0 {
        write_data(wb, &tx);
    }
}

fn load_data_ext(
    engine: &TestEngine,
    begin: [usize; NUM_CFS],
    end: [usize; NUM_CFS],
    version: u64,
    del: [bool; NUM_CFS],
    tx: &mpsc::Sender<ApplyTask>,
) -> u64 {
    let mut wb = WriteBatch::new(1, engine.inner_key_off());
    for cf in 0..NUM_CFS {
        for i in begin[cf]..end[cf] {
            let key = engine.key_builder.i_to_outer_key(i);
            let version = if cf == 1 { 0 } else { version };
            if del[cf] {
                wb.put(cf, &key, &[], BIT_DELETE, &[], version);
            } else {
                let val = key.repeat(2);
                wb.put(cf, &key, &val, 0, &[], version);
            }
        }
    }
    if wb.num_entries() > 0 {
        write_data(wb, tx)
    } else {
        0
    }
}

fn switch_mem_table(engine: &TestEngine, tx: &mpsc::Sender<ApplyTask>) {
    let mut wb = WriteBatch::new(1, engine.inner_key_off());
    wb.set_switch_mem_table();
    write_data(wb, tx);
}

fn write_data(wb: WriteBatch, applier_tx: &mpsc::Sender<ApplyTask>) -> u64 /* write_sequence */ {
    let (result_tx, result_rx) = mpsc::bounded(1);
    let task = ApplyTask::new_wb(wb, result_tx);
    if let Err(err) = applier_tx.send(task) {
        panic!("{:?}", err);
    }
    result_rx.recv().unwrap().unwrap()
}

fn check_get(
    begin: usize,
    end: usize,
    version: u64,
    cfs: &[usize],
    en: &Engine,
    exist: bool,
    check_version: Option<u64>,
    min_blob_size: u32,
) {
    for i in begin..end {
        let key = i_to_key(i as i32, min_blob_size);
        let shard = get_shard_for_key(key.as_bytes(), en);
        let snap = SnapAccess::new(&shard);
        for &cf in cfs {
            let version = if cf == 1 { 0 } else { version };
            let item = snap.get(cf, key.as_bytes(), version);
            if item.is_valid() {
                if !exist {
                    if item.is_deleted() {
                        continue;
                    }
                    let shard_stats = shard.get_stats();
                    panic!(
                        "got key {}, shard {}:{}, cf {}, stats {:?}",
                        key, shard.id, shard.ver, cf, shard_stats,
                    );
                }
                assert_eq!(item.get_value(), key.repeat(cf + 2).as_bytes());
                if cf != 1 && check_version.is_some() {
                    assert_eq!(item.version, check_version.unwrap());
                }
            } else if exist {
                let shard_stats = shard.get_stats();
                panic!(
                    "failed to get key {}, shard {}, stats {:?}",
                    key,
                    shard.tag(),
                    shard_stats,
                );
            }
        }
    }
}

fn check_iterater(begin: usize, end: usize, en: &Engine) {
    thread::sleep(Duration::from_secs(1));
    for cf in 0..3 {
        let mut i = begin;
        // let ids = vec![2, 3, 4, 5, 1];
        let ids = vec![1];
        for id in ids {
            let shard = en.get_shard(id).unwrap();
            let snap = SnapAccess::new(&shard);
            let mut iter = snap.new_iterator(cf, false, false, None, true);
            iter.seek(shard.outer_start.chunk());
            while iter.valid() {
                if iter.key.chunk() >= shard.outer_end.chunk() {
                    break;
                }
                let key = i_to_key(i as i32, en.opts.blob_table_build_options.min_blob_size);
                assert_eq!(iter.key(), key.as_bytes());
                assert_eq!(iter.val(), key.repeat(cf + 2).as_bytes());
                i += 1;
                iter.next();
            }
        }
        assert_eq!(i, end);
    }
}

fn get_shard_for_key(key: &[u8], en: &Engine) -> Arc<Shard> {
    for id in 1_u64..=5 {
        if let Some(shard) = en.get_shard(id) {
            if shard.overlap_key(InnerKey::from_inner_buf(key)) {
                return shard;
            }
        }
    }
    en.get_shard(1).unwrap()
}

#[must_use]
fn try_wait<F>(f: F, seconds: usize) -> bool
where
    F: Fn() -> bool,
{
    let begin = Instant::now_coarse();
    let timeout = Duration::from_secs(seconds as u64);
    while begin.saturating_elapsed() < timeout {
        if f() {
            return true;
        }
        thread::sleep(Duration::from_millis(100))
    }
    false
}

pub(crate) fn generate_encryption_key() -> EncryptionKey {
    let master_key_plain_text = thread_rng().gen::<[u8; 32]>().to_vec();
    let master_key = MasterKey::new(&master_key_plain_text);
    master_key.generate_encryption_key()
}
