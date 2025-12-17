// Copyright 2019 TiKV Project Authors. Licensed under Apache-2.0.

use bytes::BufMut;
use engine_traits::CfName;
use kvengine::{EXTRA_CF, WRITE_CF};
use txn_types::Key;

use crate::storage::{
    kv::{CfStatistics, Snapshot},
    mvcc::Result,
};

#[maybe_async::both]
pub async fn has_data_in_range<S: Snapshot>(
    snapshot: S,
    _cf: CfName,
    left: &Key,
    right: &Key,
    _statistic: &mut CfStatistics,
) -> Result<bool> {
    if let Some(snap) = snapshot.get_kvengine_snap() {
        let raw_left = left.to_raw().unwrap();
        let mut raw_right = right.to_raw().unwrap();
        let mut iter = snap
            .new_iterator(WRITE_CF, false, false, Some(u64::MAX), true)
            .await;
        iter.seek(&raw_left).await;
        if iter.valid() && iter.key() < raw_right.as_slice() {
            return Ok(true);
        }
        let mut extra_iter = snap.new_iterator(EXTRA_CF, false, false, Some(u64::MAX), true);
        extra_iter.seek(&raw_left);
        raw_right.put_u64(u64::MAX);
        return Ok(extra_iter.valid() && extra_iter.key() < raw_right.as_slice());
    }
    unimplemented!()
}
