// Copyright 2024 TiKV Project Authors. Licensed under Apache-2.0.

use crate::checkpoint::FileMeta;

#[derive(Debug)]
struct RangesGroup {
    ranges: Vec<FileMeta>,
    start_key: Vec<u8>,
    end_key: Vec<u8>,
}

#[allow(dead_code)]
struct RangesSplitter {
    group_num: usize,
    file_metas: Vec<FileMeta>,
}

#[allow(dead_code)]
impl RangesSplitter {
    fn new(mut file_metas: Vec<FileMeta>, group_num: usize) -> RangesSplitter {
        file_metas.sort_by(|a, b| a.first_key.cmp(&b.first_key));
        Self {
            group_num,
            file_metas,
        }
    }

    fn split_ranges_groups(&self) -> Vec<RangesGroup> {
        let mut group_len = self.file_metas.len() / self.group_num;
        if self.file_metas.len() % self.group_num != 0 {
            group_len += 1;
        }

        let mut ranges_groups: Vec<RangesGroup> = Vec::with_capacity(self.group_num);
        for file_meta in self.file_metas.iter().step_by(group_len) {
            if let Some(last) = ranges_groups.last_mut() {
                if last.start_key == file_meta.first_key {
                    continue;
                }
                last.end_key = file_meta.first_key.clone();
            }
            ranges_groups.push(RangesGroup {
                ranges: vec![],
                start_key: file_meta.first_key.clone(),
                end_key: vec![],
            });
        }

        for file_meta in &self.file_metas {
            for group in &mut ranges_groups {
                if file_meta.last_key < group.start_key
                    || (!group.end_key.is_empty() && file_meta.first_key >= group.end_key)
                {
                    continue;
                }
                group.ranges.push(file_meta.clone());
            }
        }

        ranges_groups
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::HashSet, mem, path::PathBuf};

    use proptest::prelude::*;
    use rand::prelude::*;
    use tidb_query_datatype::codec::table;

    use super::*;
    use crate::checkpoint::FileMeta;

    prop_compose! {
        fn arb_file_metas(min_file_meta_num: usize, max_file_meta_num: usize)
            (file_meta_num in min_file_meta_num..=max_file_meta_num)
            -> Vec<FileMeta> {
                let mut file_metas = Vec::with_capacity(file_meta_num);
                for i in 0..file_meta_num {
                    let mut first_key = get_random_key();
                    let mut last_key = get_random_key();
                    if first_key > last_key {
                        mem::swap(&mut first_key, &mut last_key);
                    }
                    file_metas.push(
                        FileMeta {
                            file_path: PathBuf::from(format!("path{}", i)),
                            kv_count: 10,
                            kv_size: 200,
                            first_key,
                            last_key,
                        }
                    );
                }
                file_metas
        }
    }

    #[test]
    fn test_range_splitter() {
        proptest!(|(
            file_metas in arb_file_metas(1, 100)
        )| {
            let mut rng = thread_rng();
            let group_num: usize = rng.gen_range(4..16);
            let splitter = RangesSplitter::new(file_metas.clone(), group_num);
            let groups = splitter.split_ranges_groups();

            prop_assert!(groups.len() <= group_num);
            let mut prev_end_key = groups.first().unwrap().start_key.clone();
            let mut all_file_metas = HashSet::new();
            for group in &groups {
                if !group.end_key.is_empty() {
                    prop_assert!(group.start_key < group.end_key);
                }
                prop_assert_eq!(group.start_key.as_slice(), prev_end_key.as_slice());
                prev_end_key = group.end_key.clone();

                let mut group_file_metas = HashSet::new();
                // The file meta in the group should overlap with `[group.start_key,
                // group.end_key)`.
                for file_meta in &group.ranges {
                    prop_assert!(file_meta.last_key >= group.start_key);
                    if !group.end_key.is_empty() {
                        prop_assert!(file_meta.first_key < group.end_key);
                    }

                    all_file_metas.insert(file_meta.file_path.to_str().unwrap().to_string());
                    group_file_metas.insert(file_meta.file_path.to_str().unwrap().to_string());
                }
                prop_assert_eq!(group_file_metas.len(), group.ranges.len());

                // The file meta that overlaps with `[group.start_key, group.end_key)` should be
                // in group.
                for file_meta in &file_metas {
                    if (!group.end_key.is_empty() && file_meta.first_key >= group.end_key) ||
                        (file_meta.last_key < group.start_key) {
                            continue;
                    }
                    prop_assert!(group_file_metas.contains(file_meta.file_path.to_str().unwrap()));
                }
            }
            prop_assert_eq!(all_file_metas.len(), file_metas.len());
            prop_assert!(prev_end_key.is_empty());
        });
    }

    fn get_random_key() -> Vec<u8> {
        let mut rng = thread_rng();
        let i: i64 = rng.gen_range(0..50);
        table::encode_row_key(1, i)
    }
}
