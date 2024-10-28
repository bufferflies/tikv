// Copyright 2024 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    rc::Rc,
    sync::{atomic::Ordering, Arc, Mutex},
    thread,
    time::Duration,
};

use bytes::{Buf, Bytes};
use rand::prelude::*;
use rstest::rstest;
use tidb_query_datatype::{codec::table::encode_row_key, expr::EvalContext};

use crate::{
    compaction::CompactionPriority,
    dfs,
    dfs::FileType,
    shard::{ColumnarLevels, ShardCf, ShardDataBuilder},
    table,
    table::{
        columnar::{
            build_schema_file,
            tests::{
                build_table, i_to_common_handle, merge_refs, new_schema, verify_with_ref_rows,
            },
            Block, ColumnarFile, ColumnarFilterReader, ColumnarReader, ColumnarRowTableReader,
            Schema, SchemaFile,
        },
        file::{File, InMemFile},
        sstable::{BlockCache, SsTable},
        ChecksumType,
    },
    tests::{
        keyspace_prefix, new_test_engine_opt, prepare_table_region, try_wait, Splitter, TestEngine,
        DEF_BLOCK_SIZE, KEYSPACE_ID,
    },
    DeletePrefixes, IdVer, LevelHandler, TruncateTs, DEL_PREFIXES_KEY, EXTRA_CF, LOCK_CF, WRITE_CF,
};

#[rstest]
#[case::inner_key_off_enable(true)]
#[case::inner_key_off_disable(false)]
fn test_columnar_l0_compaction(#[case] enable_inner_key_off: bool) {
    ::test_util::init_log_for_test();
    let keyspace_id = 1;
    let table_id = 30;
    let table_id2 = 31;
    let mut file_id = 100;
    let mut allocate_id = || {
        file_id += 1;
        file_id
    };
    let (engine, apply_tx) = new_test_engine_opt(enable_inner_key_off, DEF_BLOCK_SIZE, "");
    let shard_id = prepare_table_region(&engine, &apply_tx, keyspace_id, table_id);
    let shard = engine.get_shard(shard_id).unwrap();
    let schema = new_schema(table_id, true);
    let schema2 = new_schema(table_id2, true);
    let schemas = vec![schema.clone(), schema2.clone()];
    let schema_version = 10;
    let schema_file_data = build_schema_file(keyspace_id, schema_version, schemas, 0);
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
    let (l0_tbl_3, l0_tbl_3_ref) =
        build_table(allocate_id(), enable_inner_key_off, &schema2, 0, 600, 300);
    let (l1_tbl_0, l1_tbl_0_ref) =
        build_table(allocate_id(), enable_inner_key_off, &schema, 0, 100, 100);
    let (l1_tbl_1, l1_tbl_1_ref) =
        build_table(allocate_id(), enable_inner_key_off, &schema2, 0, 100, 100);
    for file in [
        &l0_tbl_0, &l0_tbl_1, &l0_tbl_2, &l0_tbl_3, &l1_tbl_0, &l1_tbl_1,
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
    col_levels.add_file(0, ColumnarFile::open(l0_tbl_3).unwrap());
    col_levels.add_file(1, ColumnarFile::open(l1_tbl_0).unwrap());
    col_levels.add_file(1, ColumnarFile::open(l1_tbl_1).unwrap());

    let mut builder = ShardDataBuilder::new(shard.get_data());
    builder.set_schema_file(Some(schema_file));
    builder.set_columnar_levels(col_levels);
    shard.set_data(builder.build());
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

    let mut mvcc_reader2 = snap
        .new_columnar_mvcc_reader(table_id2, &schema2.columns, 500)
        .unwrap();
    mvcc_reader2
        .set_handle_range(&i_to_common_handle(0), &i_to_common_handle(2100))
        .unwrap();
    let mut without_txn_id_schema_buf2 = schema2.to_schema_buf();
    without_txn_id_schema_buf2.txn_id_column = None;
    let without_txn_id_schema2 = without_txn_id_schema_buf2.into();
    let mut block2 = Block::new(&without_txn_id_schema2);
    mvcc_reader2.read_block(&mut block2, usize::MAX).unwrap();
    let tbl_refs2 = merge_refs(
        vec![l0_tbl_3_ref, l1_tbl_1_ref],
        1,
        Some(500),
        None,
        Some((i_to_common_handle(0), i_to_common_handle(2100))),
    );
    assert_eq!(block2.length(), tbl_refs2.len());
    verify_with_ref_rows(&block2, &tbl_refs2);
}

#[rstest]
#[case::inner_key_off_enable(true)]
#[case::inner_key_off_disable(false)]
fn test_columnar_l1_compaction(#[case] enable_inner_key_off: bool) {
    ::test_util::init_log_for_test();
    let keyspace_id = 1;
    let table_id = 30;
    let table_id2 = 31;
    let mut file_id = 100;
    let mut allocate_id = || {
        file_id += 1;
        file_id
    };
    let (engine, apply_tx) = new_test_engine_opt(enable_inner_key_off, DEF_BLOCK_SIZE, "");
    let shard_id = prepare_table_region(&engine, &apply_tx, keyspace_id, table_id);
    let shard = engine.get_shard(shard_id).unwrap();
    let schema = new_schema(table_id, true);
    let schema2 = new_schema(table_id2, true);
    let schemas = vec![schema.clone(), schema2.clone()];
    let schema_version = 10;
    let schema_file_data = build_schema_file(keyspace_id, schema_version, schemas, 0);
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
    let (l1_tbl_3, l1_tbl_3_ref) = build_table(
        allocate_id(),
        enable_inner_key_off,
        &schema2,
        300,
        1200,
        500,
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
    let (l2_tbl_2, l2_tbl_2_ref) =
        build_table(allocate_id(), enable_inner_key_off, &schema2, 0, 1000, 200);
    for file in [
        &l1_tbl_0, &l1_tbl_1, &l1_tbl_2, &l1_tbl_3, &l2_tbl_0, &l2_tbl_1, &l2_tbl_2,
    ] {
        info!("build file_id: {}, file size: {}", file.id(), file.size());
        fs.get_runtime()
            .block_on(fs.create(file.id(), file.read(0, file.size() as usize).unwrap(), opts))
            .unwrap()
    }

    let mut col_levels = ColumnarLevels::new();
    col_levels.add_file(1, ColumnarFile::open(l1_tbl_0).unwrap());
    col_levels.add_file(1, ColumnarFile::open(l1_tbl_1).unwrap());
    col_levels.add_file(1, ColumnarFile::open(l1_tbl_2).unwrap());
    col_levels.add_file(1, ColumnarFile::open(l1_tbl_3).unwrap());
    col_levels.add_file(2, ColumnarFile::open(l2_tbl_0).unwrap());
    col_levels.add_file(2, ColumnarFile::open(l2_tbl_1).unwrap());
    col_levels.add_file(2, ColumnarFile::open(l2_tbl_2).unwrap());

    let mut builder = ShardDataBuilder::new(shard.get_data());
    builder.set_schema_file(Some(schema_file));
    builder.set_columnar_levels(col_levels);
    shard.set_data(builder.build());
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

    let mut mvcc_reader2 = snap
        .new_columnar_mvcc_reader(table_id2, &schema2.columns, 600)
        .unwrap();
    mvcc_reader2
        .set_handle_range(&i_to_common_handle(0), &i_to_common_handle(2100))
        .unwrap();
    let mut without_txn_id_schema2 = schema2.to_schema_buf();
    without_txn_id_schema2.txn_id_column = None;
    let mut block2 = Block::new(&without_txn_id_schema2.into());
    mvcc_reader2.read_block(&mut block2, usize::MAX).unwrap();
    let tbl_refs2 = merge_refs(
        vec![l1_tbl_3_ref, l2_tbl_2_ref],
        2,
        Some(600),
        None,
        Some((i_to_common_handle(0), i_to_common_handle(2100))),
    );
    assert_eq!(block2.length(), tbl_refs2.len());
    verify_with_ref_rows(&block2, &tbl_refs2);
}

#[rstest]
#[case::inner_key_off_enable(true)]
#[case::inner_key_off_disable(false)]
fn test_columnar_major_compaction(#[case] enable_inner_key_off: bool) {
    use crate::table::columnar::{ColumnarMergeReader, ColumnarMvccReader};

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
    let schema_file_data = build_schema_file(keyspace_id, schema_version, schemas, 0);
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

    let mut builder = ShardDataBuilder::new(shard.get_data());
    builder.set_cfs([write_cf, ShardCf::new(LOCK_CF), ShardCf::new(EXTRA_CF)]);
    builder.set_schema_file(Some(schema_file));
    shard.set_data(builder.build());
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
            None,
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
    shard.set_data(ShardDataBuilder::new(data).build());
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
    let schema_file_data = build_schema_file(keyspace_id, schema_version, schemas, 0);
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
    let mut builder = ShardDataBuilder::new(shard.get_data());
    builder.set_schema_file(Some(schema_file));
    builder.set_columnar_levels(col_levels);
    shard.set_data(builder.build());
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
    let schema_file_data = build_schema_file(keyspace_id, schema_version, schemas, 0);
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
    let mut builder = ShardDataBuilder::new(shard.get_data());
    builder.set_schema_file(Some(schema_file));
    builder.set_columnar_levels(col_levels);
    shard.set_data(builder.build());
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
    let schema_file_data = build_schema_file(keyspace_id, schema_version, schemas, 0);
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

    let mut builder = ShardDataBuilder::new(shard.get_data());
    builder.set_schema_file(Some(schema_file));
    builder.set_columnar_levels(col_levels);
    shard.set_data(builder.build());
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
        ChecksumType::Crc32,
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
    SsTable::new(Arc::new(file), BlockCache::None, None).unwrap()
}
