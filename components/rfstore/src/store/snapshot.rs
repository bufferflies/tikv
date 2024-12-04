// Copyright 2021 TiKV Project Authors. Licensed under Apache-2.0.

use std::{num::NonZeroU64, sync::Arc};

use kvengine::SnapAccess;
use kvproto::{kvrpcpb::ExtraOp as TxnExtraOp, metapb::Region};
use pd_client::BucketMeta;
use raftstore::store::TxnExt;
use tikv_util::{metrics::CRITICAL_ERROR, panic_when_unexpected_key_or_data, set_panic_mark};

/// Snapshot of a region.
///
/// Only data within a region can be accessed.
#[derive(Debug)]
pub struct RegionSnapshot {
    pub snap: SnapAccess,
    // `None` means the snapshot does not provide peer related transaction extensions.
    pub txn_ext: Option<Arc<TxnExt>>,
    pub term: Option<NonZeroU64>,
    pub txn_extra_op: TxnExtraOp,
    pub bucket_meta: Option<Arc<BucketMeta>>,
}

impl RegionSnapshot {
    pub fn from_raw(db: &kvengine::Engine, region: &Region) -> RegionSnapshot {
        let snap = db.get_snap_access(region.get_id()).unwrap();
        RegionSnapshot::from_snapshot(snap)
    }

    pub fn from_snapshot(snap: SnapAccess) -> RegionSnapshot {
        RegionSnapshot {
            snap,
            txn_ext: None,
            term: None,
            txn_extra_op: TxnExtraOp::Noop,
            bucket_meta: None,
        }
    }

    #[inline]
    pub fn get_start_key(&self) -> &[u8] {
        self.snap.get_start_key()
    }

    #[inline]
    pub fn get_end_key(&self) -> &[u8] {
        self.snap.get_end_key()
    }

    #[inline]
    pub fn is_sync(&self) -> bool {
        self.snap.is_sync()
    }
}

impl Clone for RegionSnapshot {
    fn clone(&self) -> Self {
        RegionSnapshot {
            snap: self.snap.clone(),
            txn_ext: self.txn_ext.clone(),
            term: self.term,
            txn_extra_op: self.txn_extra_op,
            bucket_meta: self.bucket_meta.clone(),
        }
    }
}

/// `SnapshotIterator` is dummy implementation for the tikv_kv::Iterator.
pub struct RegionSnapshotIterator {}

#[allow(unused)]
#[inline(never)]
fn handle_check_key_in_region_error(e: crate::Error) -> crate::Result<()> {
    // Split out the error case to reduce hot-path code size.
    CRITICAL_ERROR
        .with_label_values(&["key not in region"])
        .inc();
    if panic_when_unexpected_key_or_data() {
        set_panic_mark();
        panic!("key exceed bound: {:?}", e);
    } else {
        Err(e)
    }
}
