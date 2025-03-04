// Copyright 2021 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    cmp,
    iter::Iterator as StdIterator,
    ops::Deref,
    sync::{
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
        Arc, Mutex,
    },
};

use super::{Arena, SkipList};
use crate::{
    table::{memtable::skl_ext::SkipListExt, DataBound, TxnFile},
    EXTRA_CF, NUM_CFS, WRITE_CF,
};

#[derive(Clone)]
pub struct CfTable {
    pub core: Arc<CfTableCore>,
}

impl Deref for CfTable {
    type Target = CfTableCore;

    fn deref(&self) -> &Self::Target {
        &self.core
    }
}

impl Default for CfTable {
    fn default() -> Self {
        Self::new()
    }
}

impl CfTable {
    pub fn new() -> Self {
        Self {
            core: Arc::new(CfTableCore::new()),
        }
    }

    pub fn new_split(&self) -> Self {
        let tbls = self.core.tbls.clone();
        let arena = self.core.arena.clone();
        let ver = AtomicU64::new(self.ver.load(Ordering::Acquire));
        let force_switch = AtomicBool::new(self.force_switch.load(Ordering::Acquire));
        let props = Mutex::new(self.core.props.lock().unwrap().clone());
        let unpersisted_props_size = AtomicUsize::new(self.core.unpersisted_props_size());
        Self {
            core: Arc::new(CfTableCore {
                tbls,
                arena,
                ver,
                force_switch,
                props,
                unpersisted_props_size,
            }),
        }
    }

    #[must_use]
    pub fn add_write_cf_txn_files(&self, txn_files: &[TxnFile]) -> Self {
        let mut tbls = self.core.tbls.clone();
        tbls[WRITE_CF] = tbls[WRITE_CF].add_txn_files(txn_files);
        let arena = self.core.arena.clone();
        let ver = AtomicU64::new(self.ver.load(Ordering::Acquire));
        let force_switch = AtomicBool::new(self.force_switch.load(Ordering::Acquire));
        let props = Mutex::new(self.core.props.lock().unwrap().clone());
        let unpersisted_props_size = AtomicUsize::new(self.core.unpersisted_props_size());
        Self {
            core: Arc::new(CfTableCore {
                tbls,
                arena,
                ver,
                force_switch,
                props,
                unpersisted_props_size,
            }),
        }
    }
}

pub struct CfTableCore {
    tbls: [SkipListExt; NUM_CFS],
    arena: Arc<Arena>,
    ver: AtomicU64,
    force_switch: AtomicBool,
    props: Mutex<Option<kvenginepb::Properties>>,
    unpersisted_props_size: AtomicUsize,
}

impl Default for CfTableCore {
    fn default() -> Self {
        Self::new()
    }
}

impl CfTableCore {
    pub fn new() -> Self {
        let arena = Arc::new(Arena::new());
        Self {
            tbls: [
                SkipListExt::new(SkipList::new(Some(arena.clone()))),
                SkipListExt::new(SkipList::new(Some(arena.clone()))),
                SkipListExt::new(SkipList::new(Some(arena.clone()))),
            ],
            arena,
            ver: AtomicU64::new(0),
            force_switch: AtomicBool::new(false),
            props: Mutex::new(None),
            unpersisted_props_size: AtomicUsize::default(),
        }
    }

    pub fn get_cf(&self, cf: usize) -> &SkipListExt {
        &self.tbls[cf]
    }

    pub fn is_empty(&self) -> bool {
        for tbl in &self.tbls {
            if !tbl.is_empty() {
                return false;
            }
        }
        true
    }

    pub fn size(&self) -> u64 {
        self.tbls.iter().map(|t| t.size() as u64).sum()
    }

    pub fn skip_list_entries(&self) -> usize {
        self.tbls.iter().map(|t| t.skip_list_entries()).sum()
    }

    pub fn set_version(&self, ver: u64) {
        self.ver.store(ver, Ordering::Release)
    }

    pub fn set_properties(&self, props: kvenginepb::Properties) {
        self.props.lock().unwrap().replace(props);
    }

    pub fn get_properties(&self) -> Option<kvenginepb::Properties> {
        self.props.lock().unwrap().clone()
    }

    pub fn add_unpersisted_props_size(&self, size: usize) {
        self.unpersisted_props_size
            .fetch_add(size, Ordering::Release);
    }

    pub fn unpersisted_props_size(&self) -> usize {
        self.unpersisted_props_size.load(Ordering::Acquire)
    }

    pub fn get_version(&self) -> u64 {
        self.ver.load(Ordering::Acquire)
    }

    pub fn set_force_switch(&self) {
        self.force_switch.store(true, Ordering::Release);
    }

    pub fn is_force_switch(&self) -> bool {
        self.force_switch.load(Ordering::Acquire)
    }

    pub fn has_data_in_bound(&self, bound: DataBound<'_>) -> bool {
        if self.is_empty() {
            return false;
        }
        for cf in 0..NUM_CFS {
            let tbl = &self.tbls[cf];
            let mut iter = tbl.new_iterator(false);
            iter.seek(bound.lower_bound);
            if iter.valid() && !bound.less_than_key(iter.key()) {
                return true;
            }
        }
        false
    }

    pub(crate) fn data_max_ts(&self) -> u64 {
        // Ignore LOCK_CF, as `ts` in LOCK_CF is not a TSO.
        // TODO: in async_commit, LOCK_CF may contains data, need handle it later.
        cmp::max(
            self.tbls[WRITE_CF].data_max_ts(),
            self.tbls[EXTRA_CF].data_max_ts(),
        )
    }
}

#[cfg(feature = "debug-trace-mem-table")]
impl CfTableCore {
    pub fn skl_and_txn_files_size(&self) -> (u64 /* skl_size */, u64 /* txn_files_size */) {
        self.tbls
            .iter()
            .fold((0, 0), |(skl_size, txn_files_size), t| {
                (
                    skl_size + t.skl_size() as u64,
                    txn_files_size + t.txn_files_size() as u64,
                )
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::table::{memtable::WriteBatch, InnerKey};

    /// Test basic operations on an empty CfTable
    #[test]
    fn test_empty_cftable() {
        let table = CfTable::new();

        // Test empty state
        assert!(table.is_empty());
        assert_eq!(table.size(), 0);
        assert_eq!(table.skip_list_entries(), 0);
        assert_eq!(table.get_version(), 0);
        assert!(!table.is_force_switch());

        // Test empty properties
        assert!(table.get_properties().is_none());
    }

    /// Test version management
    #[test]
    fn test_version_management() {
        let table = CfTable::new();

        // Set and verify version
        table.set_version(42);
        assert_eq!(table.get_version(), 42);

        // Create split table and verify version inheritance
        let split_table = table.new_split();
        assert_eq!(split_table.get_version(), 42);

        // Verify version independence after split
        split_table.set_version(100);
        assert_eq!(split_table.get_version(), 100);
        assert_eq!(table.get_version(), 42);
    }

    /// Test force switch functionality
    #[test]
    fn test_force_switch() {
        let table = CfTable::new();
        assert!(!table.is_force_switch());

        table.set_force_switch();
        assert!(table.is_force_switch());

        // Test force switch in split table
        let split_table = table.new_split();
        assert!(split_table.is_force_switch());
    }

    /// Test data operations across different CFs
    #[test]
    fn test_cf_operations() {
        let table = CfTable::new();

        // Write data to different CFs
        for cf in 0..NUM_CFS {
            let tbl = table.get_cf(cf);
            let mut wb = WriteBatch::new();
            wb.put(
                InnerKey::from_inner_buf(format!("key_{}", cf).as_bytes()),
                0,
                &[0],
                1,
                format!("value_{}", cf).as_bytes(),
            );
            tbl.put_batch(&mut wb, None, cf);
        }

        // Verify data presence
        assert!(!table.is_empty());
        assert!(table.size() > 0);
        assert_eq!(table.skip_list_entries(), NUM_CFS);

        // Check data in each CF
        for cf in 0..NUM_CFS {
            let tbl = table.get_cf(cf);
            let mut it = tbl.new_iterator(false);
            it.rewind();
            assert!(it.valid());
            assert_eq!(it.key().deref(), format!("key_{}", cf).as_bytes());
        }
    }

    /// Test data bound checking
    #[test]
    fn test_data_bound_checking() {
        let table = CfTable::new();

        // Insert test data
        let cf = WRITE_CF;
        let tbl = table.get_cf(cf);
        let mut wb = WriteBatch::new();
        wb.put(InnerKey::from_inner_buf(b"key_50"), 0, &[0], 1, b"value_50");
        tbl.put_batch(&mut wb, None, cf);

        // Test bounds
        let bound = DataBound {
            lower_bound: InnerKey::from_inner_buf(b"key_00"),
            upper_bound: InnerKey::from_inner_buf(b"key_99"),
            upper_inclusive: true,
        };
        assert!(table.has_data_in_bound(bound));

        let bound = DataBound {
            lower_bound: InnerKey::from_inner_buf(b"key_60"),
            upper_bound: InnerKey::from_inner_buf(b"key_99"),
            upper_inclusive: true,
        };
        assert!(!table.has_data_in_bound(bound));
    }

    /// Test properties management
    #[test]
    fn test_properties() {
        let table = CfTable::new();

        // Initially no properties
        assert!(table.get_properties().is_none());

        // Set and get properties
        let props = kvenginepb::Properties::default();
        table.set_properties(props.clone());

        // Test properties in split table
        let split_table = table.new_split();
        assert!(split_table.get_properties().is_some());
    }

    /// Test data max timestamp
    #[test]
    fn test_data_max_ts() {
        let table = CfTable::new();

        // Write data with different timestamps to different CFs
        let write_cf = table.get_cf(WRITE_CF);
        let mut wb = WriteBatch::new();
        wb.put(
            InnerKey::from_inner_buf(b"key1"),
            0,
            &[0],
            100, // timestamp
            b"value1",
        );
        write_cf.put_batch(&mut wb, None, WRITE_CF);

        let extra_cf = table.get_cf(EXTRA_CF);
        wb.reset();
        wb.put(
            InnerKey::from_inner_buf(b"key2"),
            0,
            &[0],
            200, // timestamp
            b"value2",
        );
        extra_cf.put_batch(&mut wb, None, EXTRA_CF);

        // Max timestamp should be 200
        assert_eq!(table.data_max_ts(), 200);
    }
}
