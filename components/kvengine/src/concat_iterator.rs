// Copyright 2021 TiKV Project Authors. Licensed under Apache-2.0.

use std::fmt;

use crate::{
    table::{
        search,
        sstable::{SsTable, TableIterator},
        InnerKey, Iterator, Value,
    },
    LevelHandler,
};

// ConcatIterator concatenates the sequences defined by several iterators.  (It
// only works with TableIterators, probably just because it's faster to not be
// so generic.)
pub(crate) struct ConcatIterator {
    // Should always use `set_idx` to set `idx` & `iter`.
    idx: i32,
    iter: Option<Box<TableIterator>>,

    level: LevelHandler,
    reversed: bool,
    fill_cache: bool,
}

impl fmt::Debug for ConcatIterator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ConcatIterator")
            .field("idx", &self.idx)
            .field("iter", &self.iter)
            .field("reversed", &self.reversed)
            .finish()
    }
}

#[allow(dead_code)]
impl ConcatIterator {
    pub(crate) fn new(level: LevelHandler, reversed: bool, fill_cache: bool) -> Self {
        ConcatIterator {
            idx: -1,
            iter: None,
            level,
            reversed,
            fill_cache,
        }
    }

    pub(crate) fn new_with_tables(tables: Vec<SsTable>, reversed: bool, fill_cache: bool) -> Self {
        let level = LevelHandler::new(1, tables);
        ConcatIterator {
            idx: -1,
            iter: None,
            level,
            reversed,
            fill_cache,
        }
    }

    fn get_table(&self, idx: usize) -> &SsTable {
        &self.level.tables[idx]
    }

    fn num_tables(&self) -> usize {
        self.level.tables.len()
    }

    fn set_idx(&mut self, idx: i32) {
        if self.idx == idx {
            return;
        }
        self.idx = idx;
        if idx < 0 || idx as usize >= self.num_tables() {
            self.iter = None;
        } else {
            let mut iter = self
                .get_table(idx as usize)
                .new_iterator(self.reversed, self.fill_cache);
            iter.rewind();
            self.iter = Some(iter);
        }
    }
}

impl Iterator for ConcatIterator {
    fn next(&mut self) {
        if self.iter.is_none() {
            return;
        }
        if let Some(iter) = &mut self.iter {
            iter.next();
            if iter.valid() {
                return;
            }
        }
        loop {
            if !self.reversed {
                self.set_idx(self.idx + 1);
            } else {
                self.set_idx(self.idx - 1);
            }
            match &mut self.iter {
                None => return,
                Some(iter) => {
                    iter.rewind();
                    if iter.valid() {
                        return;
                    }
                }
            }
        }
    }

    fn next_version(&mut self) -> bool {
        self.iter.as_mut().unwrap().next_version()
    }

    fn rewind(&mut self) {
        if self.num_tables() == 0 {
            return;
        }
        if !self.reversed {
            self.set_idx(0);
        } else {
            self.set_idx(self.num_tables() as i32 - 1);
        }
        self.iter.as_mut().unwrap().rewind();
    }

    fn seek(&mut self, key: InnerKey<'_>) {
        use std::cmp::Ordering::*;
        let idx = if !self.reversed {
            search(self.num_tables(), |idx| {
                self.get_table(idx).biggest().cmp(&key) != Less
            }) as i32
        } else {
            let n = self.num_tables();
            let ridx = search(n, |idx| {
                self.get_table(n - 1 - idx).smallest().cmp(&key) != Greater
            }) as i32;
            n as i32 - 1 - ridx
        };
        self.set_idx(idx);
        if let Some(iter) = self.iter.as_mut() {
            iter.seek(key);
        }
    }

    fn key(&self) -> InnerKey<'_> {
        self.iter.as_ref().unwrap().key()
    }

    fn value(&self) -> Value {
        self.iter.as_ref().unwrap().value()
    }

    fn valid(&self) -> bool {
        match &self.iter {
            Some(x) => x.valid(),
            None => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{iter::Iterator as StdIterator, ops::Deref};

    use proptest::{prelude::*, test_runner::TestCaseResult};
    use rstest::rstest;

    use crate::{
        concat_iterator::ConcatIterator,
        table::{
            sstable::{test_util::*, SsTable},
            InnerKey, Iterator,
        },
    };

    #[test]
    fn test_concat_iterator_one_table() {
        let tables = vec![build_test_table_with_kvs(&vec![
            ("k1".to_string(), "a1".to_string()),
            ("k2".to_string(), "a2".to_string()),
        ])];
        let mut it = ConcatIterator::new_with_tables(tables, false, true);
        it.rewind();
        assert_eq!(it.valid(), true);
        assert_eq!(it.key().deref(), "k1".as_bytes());
        let v = it.value();
        assert_eq!(v.get_value(), "a1".as_bytes());
    }

    #[test]
    fn test_concat_iterator() {
        let (t1, _) = build_test_table_with_prefix("keya", 10000);
        let (t2, _) = build_test_table_with_prefix("keyb", 10000);
        let (t3, _) = build_test_table_with_prefix("keyc", 10000);
        let tables = vec![t1, t2, t3];
        {
            let mut it = ConcatIterator::new_with_tables(tables.clone(), false, true);
            it.rewind();
            assert_eq!(it.valid(), true);
            let mut cnt = 0;
            while it.valid() {
                let v = it.value();
                assert_eq!(v.get_value(), get_test_value(cnt % 10000).as_bytes());
                cnt += 1;
                it.next();
            }
            assert_eq!(cnt, 30000);
            it.seek(InnerKey::from_inner_buf("a".as_bytes()));
            assert_eq!(it.key().deref(), "keya0000".as_bytes());
            assert_eq!(it.value().get_value(), get_test_value(0).as_bytes());

            it.seek(InnerKey::from_inner_buf("keyb".as_bytes()));
            assert_eq!(it.key().deref(), "keyb0000".as_bytes());
            assert_eq!(it.value().get_value(), get_test_value(0).as_bytes());

            it.seek(InnerKey::from_inner_buf("keyb9999b".as_bytes()));
            assert_eq!(it.key().deref(), "keyc0000".as_bytes());
            assert_eq!(it.value().get_value(), get_test_value(0).as_bytes());

            it.seek(InnerKey::from_inner_buf("keyd".as_bytes()));
            assert_eq!(it.valid(), false);

            it.seek(InnerKey::from_inner_buf("keyb9999b".as_bytes()));
            assert_eq!(it.key().deref(), "keyc0000".as_bytes());
            assert_eq!(it.value().get_value(), get_test_value(0).as_bytes());
        }
        {
            let mut it = ConcatIterator::new_with_tables(tables, true, true);
            it.rewind();
            assert_eq!(it.valid(), true);
            let mut cnt = 0;
            while it.valid() {
                let v = it.value();
                assert_eq!(
                    v.get_value(),
                    get_test_value(10000 - (cnt % 10000) - 1).as_bytes()
                );
                cnt += 1;
                it.next();
            }
            assert_eq!(cnt, 30000);

            it.seek(InnerKey::from_inner_buf("a".as_bytes()));
            assert_eq!(it.valid(), false);

            it.seek(InnerKey::from_inner_buf("keyb".as_bytes()));
            assert_eq!(it.key().deref(), "keya9999".as_bytes());
            assert_eq!(it.value().get_value(), get_test_value(9999).as_bytes());

            it.seek(InnerKey::from_inner_buf("keyb9999b".as_bytes()));
            assert_eq!(it.key().deref(), "keyb9999".as_bytes());
            assert_eq!(it.value().get_value(), get_test_value(9999).as_bytes());

            it.seek(InnerKey::from_inner_buf("keyd".as_bytes()));
            assert_eq!(it.key().deref(), "keyc9999".as_bytes());
            assert_eq!(it.value().get_value(), get_test_value(9999).as_bytes());
        }
    }

    /// Generate `max_tables` number of SsTables, each with `max_rows` number of
    /// rows.
    ///
    /// Given there are 10 rows totally, the generated keys will be like:
    /// `xkey0001`, `xkey0003`, ... `xkey0019`
    fn arb_sstables(
        max_tables: usize,
        max_rows: usize,
    ) -> impl Strategy<Value = (Vec<SsTable>, Vec<(String, String)>)> {
        prop::collection::vec(1..=max_rows, 1..max_tables).prop_map(move |rows| {
            let mut total_kvs = Vec::with_capacity(rows.iter().sum());
            let mut tables = Vec::with_capacity(rows.len());
            let mut i = 1;
            for row in rows {
                let mut kvs = Vec::with_capacity(row);
                for _ in 0..row {
                    let key = get_test_key("xkey", i);
                    let val = get_test_value(i);
                    kvs.push((key, val));
                    i += 2;
                }
                tables.push(build_test_table_with_kvs(&kvs));
                total_kvs.extend(kvs);
            }
            (tables, total_kvs)
        })
    }

    #[rstest]
    #[case(false)]
    #[case::reverse(true)]
    fn test_concat_iterator_arb(#[case] reverse: bool) {
        proptest!(|((tables, total_kvs) in arb_sstables(5, 10), seek_keys in prop::collection::vec(0usize..=100, 20))| {
            // NOTE: the `ref_pos` is exclusive for reverse iterator, to avoid to handle negative `ref_pos`.
            let verify_next = |it: &mut ConcatIterator, ref_pos: usize| -> TestCaseResult {
                let rng: Box<dyn StdIterator<Item=usize>> = if !reverse {
                    Box::new(ref_pos..total_kvs.len()) as _
                } else {
                    Box::new((0..ref_pos).rev()) as _
                };
                for pos in rng {
                    let (ref_key, ref_val) = &total_kvs[pos];
                    let k = it.key();
                    let v = it.value();

                    prop_assert!(it.valid());
                    prop_assert_eq!(k.deref(), ref_key.as_bytes());
                    prop_assert_eq!(v.get_value(), ref_val.as_bytes());

                    it.next();
                }
                prop_assert!(!it.valid(), "{:?}", it);
                Ok(())
            };
            // The `ref_pos` is exclusive for reverse iterator.
            let verify_valid = |it: &ConcatIterator, ref_pos: usize| -> TestCaseResult {
                let expect_valid = if !reverse {
                    ref_pos < total_kvs.len()
                } else {
                    ref_pos > 0
                };
                prop_assert_eq!(it.valid(), expect_valid);
                Ok(())
            };

            let mut it = ConcatIterator::new_with_tables(tables.clone(), reverse, true);

            let first_pos = if !reverse {
                0
            } else {
                total_kvs.len()
            };

            it.rewind();
            verify_next(&mut it, first_pos)?;

            for seek_only in [false, true] {
                for &i in &seek_keys {
                    let key = get_test_key("xkey", i);
                    it.seek(InnerKey::from_inner_buf(key.as_bytes()));

                    let ref_pos = match total_kvs.binary_search_by(|(k, _)| k.cmp(&key)) {
                        Ok(pos) => if !reverse {
                            pos
                        } else {
                            pos + 1
                        },
                        Err(pos) => pos
                    };

                    if seek_only {
                        // For https://github.com/tidbcloud/cloud-storage-engine/pull/1956. Which only happens without `next`.
                        verify_valid(&it, ref_pos)?;
                    } else {
                        verify_next(&mut it, ref_pos)?;
                    }
                }
            }
        })
    }
}
