// Copyright 2024 TiKV Project Authors. Licensed under Apache-2.0.

use kvengine::WRITE_CF;
use rstest::rstest;
use test_cloud_server::util::Mutation;
use test_util::init_log_for_test;
use tikv_util::debug;

use crate::engine_basic::test_remote_coprocessor::{DagTest, ProductTable, Txn};

// TODO: move storage relevant test cases from `mod.rs` to here.

#[rstest]
#[case(false)]
#[case::ia(true)]
fn test_point_get(#[case] enable_ia: bool) {
    init_log_for_test();

    let make_row = |i| -> (i64, Option<String>, i64) {
        let id = i;
        let name = format!("v:{i}");
        let count = 1;
        (id, Some(name), count)
    };

    let write_rows = |dag_test: &mut DagTest<'_>, i, j| -> Vec<Mutation> {
        let rows: Vec<_> = (i..j).map(make_row).collect();
        let rows_ref: Vec<_> = rows
            .iter()
            .map(|(id, name, count)| (*id, name.as_deref(), *count))
            .collect();

        let mut row_cache = dag_test.get_row_cache();
        row_cache.add_rows(&rows_ref);

        let mut txn = Txn::new();
        txn.begin(dag_test.get_ts());

        txn.prewrite(dag_test.get_inserter(), row_cache);
        txn.commit();

        txn.into_prewrites()
            .into_iter()
            .flat_map(|(_, muts)| muts)
            .collect()
    };

    let product = ProductTable::new();
    let mut dag_test = DagTest::new(&product);
    dag_test.setup_ia(enable_ia);
    let _enter = dag_test.enter_runtime();

    let muts = write_rows(&mut dag_test, 1000, 4000);

    if enable_ia {
        dag_test.perform_major_compaction();
    }

    let select_key_range = dag_test.get_key_range_all();
    let snap_access =
        dag_test.get_snap_access(dag_test.get_ts().into_inner(), vec![select_key_range]);

    let read_ts = dag_test.get_ts().into_inner();
    let task = async move {
        for m in muts {
            let item = snap_access.get_async(WRITE_CF, m.get_key(), read_ts).await;
            debug!(
                "get_async key {:?}, item {:?}, expect {:?}",
                m.get_key(),
                item.get_value(),
                m.get_value()
            );
            if m.get_value() == [0] {
                assert!(
                    item.is_value_empty() || item.get_value() == [0],
                    "item {:?}, expect {:?}",
                    item.get_value(),
                    m.get_value()
                );
            } else {
                assert_eq!(
                    item.get_value(),
                    m.get_value(),
                    "item {:?}, expect {:?}",
                    item.get_value(),
                    m.get_value()
                );
            }
        }
    };
    dag_test.ctx.rt.block_on(task);
    dag_test.check_ia();
}
