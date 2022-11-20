// Copyright 2021 TiKV Project Authors. Licensed under Apache-2.0.

use std::{num::NonZeroU64, sync::Arc};

use engine_traits::{CfName, IterOptions, Peekable, ReadOptions, Snapshot};
use keys::DataKey;
use kvproto::{errorpb, kvrpcpb::ExtraOp as TxnExtraOp};
use pd_client::BucketMeta;
use raftstore::{
    store::{RegionIterator, RegionSnapshot, TxnExt},
    Error as RaftServerError,
};
use txn_types::{Key, Value};

use crate::{
    self as kv, Error, Error as KvError, ErrorInner, Iterator as EngineIterator,
    Snapshot as EngineSnapshot, SnapshotExt,
};

impl From<RaftServerError> for Error {
    fn from(e: RaftServerError) -> Error {
        Error(Box::new(ErrorInner::Request(errorpb::Error::from(e))))
    }
}

pub struct RegionSnapshotExt<'a, S: Snapshot, K: DataKey> {
    snapshot: &'a RegionSnapshot<S, K>,
}

impl<'a, S: Snapshot, K: DataKey> SnapshotExt for RegionSnapshotExt<'a, S, K> {
    #[inline]
    fn get_data_version(&self) -> Option<u64> {
        self.snapshot.get_apply_index().ok()
    }

    fn is_max_ts_synced(&self) -> bool {
        self.snapshot
            .txn_ext
            .as_ref()
            .map(|txn_ext| txn_ext.is_max_ts_synced())
            .unwrap_or(false)
    }

    fn get_term(&self) -> Option<NonZeroU64> {
        self.snapshot.term
    }

    fn get_txn_extra_op(&self) -> TxnExtraOp {
        self.snapshot.txn_extra_op
    }

    fn get_txn_ext(&self) -> Option<&Arc<TxnExt>> {
        self.snapshot.txn_ext.as_ref()
    }

    fn get_buckets(&self) -> Option<Arc<BucketMeta>> {
        self.snapshot.bucket_meta.clone()
    }
}

impl<S: Snapshot, K: DataKey + 'static> EngineSnapshot for RegionSnapshot<S, K> {
    type Iter = RegionIterator<S, K>;
    type Ext<'a> = RegionSnapshotExt<'a, S, K>;

    fn get(&self, key: &Key) -> kv::Result<Option<Value>> {
        fail_point!("raftkv_snapshot_get", |_| Err(box_err!(
            "injected error for get"
        )));
        let v = box_try!(self.get_value(key.as_encoded()));
        Ok(v.map(|v| v.to_vec()))
    }

    fn get_cf(&self, cf: CfName, key: &Key) -> kv::Result<Option<Value>> {
        fail_point!("raftkv_snapshot_get_cf", |_| Err(box_err!(
            "injected error for get_cf"
        )));
        let v = box_try!(self.get_value_cf(cf, key.as_encoded()));
        Ok(v.map(|v| v.to_vec()))
    }

    fn get_cf_opt(&self, opts: ReadOptions, cf: CfName, key: &Key) -> kv::Result<Option<Value>> {
        fail_point!("raftkv_snapshot_get_cf", |_| Err(box_err!(
            "injected error for get_cf"
        )));
        let v = box_try!(self.get_value_cf_opt(&opts, cf, key.as_encoded()));
        Ok(v.map(|v| v.to_vec()))
    }

    fn iter(&self, cf: CfName, iter_opt: IterOptions) -> kv::Result<Self::Iter> {
        fail_point!("raftkv_snapshot_iter", |_| Err(box_err!(
            "injected error for iter_cf"
        )));
        RegionSnapshot::iter(self, cf, iter_opt).map_err(kv::Error::from)
    }

    #[inline]
    fn lower_bound(&self) -> Option<&[u8]> {
        Some(self.get_start_key())
    }

    #[inline]
    fn upper_bound(&self) -> Option<&[u8]> {
        Some(self.get_end_key())
    }

    fn ext(&self) -> RegionSnapshotExt<'_, S, K> {
        RegionSnapshotExt { snapshot: self }
    }
}

impl<S: Snapshot, K: DataKey> EngineIterator for RegionIterator<S, K> {
    fn next(&mut self) -> kv::Result<bool> {
        RegionIterator::next(self).map_err(KvError::from)
    }

    fn prev(&mut self) -> kv::Result<bool> {
        RegionIterator::prev(self).map_err(KvError::from)
    }

    fn seek(&mut self, key: &Key) -> kv::Result<bool> {
        fail_point!("raftkv_iter_seek", |_| Err(box_err!(
            "injected error for iter_seek"
        )));
        RegionIterator::seek(self, key.as_encoded()).map_err(From::from)
    }

    fn seek_for_prev(&mut self, key: &Key) -> kv::Result<bool> {
        fail_point!("raftkv_iter_seek_for_prev", |_| Err(box_err!(
            "injected error for iter_seek_for_prev"
        )));
        RegionIterator::seek_for_prev(self, key.as_encoded()).map_err(From::from)
    }

    fn seek_to_first(&mut self) -> kv::Result<bool> {
        RegionIterator::seek_to_first(self).map_err(KvError::from)
    }

    fn seek_to_last(&mut self) -> kv::Result<bool> {
        RegionIterator::seek_to_last(self).map_err(KvError::from)
    }

    fn valid(&self) -> kv::Result<bool> {
        RegionIterator::valid(self).map_err(KvError::from)
    }

    fn validate_key(&self, key: &Key) -> kv::Result<()> {
        self.should_seekable(key.as_encoded()).map_err(From::from)
    }

    fn key(&self) -> &[u8] {
        RegionIterator::key(self)
    }

    fn value(&self) -> &[u8] {
        RegionIterator::value(self)
    }
}
