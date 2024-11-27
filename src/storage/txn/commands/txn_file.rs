// Copyright 2023 TiKV Project Authors. Licensed under Apache-2.0.

use std::{cmp::Ordering, iter::Iterator as StdIterator, mem, ops::Deref};

use api_version::ApiV2;
use kvengine::{
    table::{
        txn_file::{TxnCtx, TxnFile, TxnFileId, TxnFileIterator, OP_CHECK_NOT_EXIST, OP_INSERT},
        BoundedDataSet, InnerKey,
    },
    txn_chunk_manager::TxnChunkManager,
    Iterator, SnapAccess, UserMeta, LOCK_CF, WRITE_CF,
};
use kvenginepb::TxnFileRef;
use kvproto::kvrpcpb::{CommandPri, WriteConflictReason};
use log_wrappers::Value as LogValue;
use protobuf::Message;
use tikv_kv::{Snapshot, WriteData};
use txn_types::{Key, LockType, TimeStamp, WriteType};

use crate::storage::{
    lock_manager::LockManager,
    mvcc,
    mvcc::{CloudReader, ErrorInner, TxnCommitRecord},
    txn::{
        commands::{
            CheckTxnStatus, CommandExt, Commit, Prewrite, ReleasedLocks, ResolveLock,
            ResolveLockLite, ResolveLockReadPhase, ResponsePolicy, Rollback, TxnHeartBeat,
            WriteCommand, WriteContext, WriteResult,
        },
        Command, Error, Lock, Result,
    },
    types, ProcessResult, TxnStatus,
};

pub struct TxnFileCommand {
    pub txn_file_ref: TxnFileRef,
    modified: bool,
    pub lock_prefix: Option<txn_types::Lock>,
    // `txn_file` represents the request data from command. Get existed locks from
    // `SnapAccess::get_lock_txn_file()`.
    pub txn_file: TxnFile,
    pub inner_cmd: Option<Box<Command>>,
}

impl TxnFileCommand {
    ///   Return a  `TypedCommand`  that encapsulates the result of executing
    /// this command.
    pub fn new(inner_cmd: Command, txn_file_ref: TxnFileRef, txn_file: TxnFile) -> Command {
        let lock_prefix = if txn_file_ref.lock_val_prefix.is_empty() {
            None
        } else {
            Some(txn_types::Lock::parse(&txn_file_ref.lock_val_prefix).unwrap())
        };
        debug!(
            "new txn file command {:?}, txn_file_ref {:?}, lock {:?}",
            inner_cmd, txn_file_ref, lock_prefix
        );
        Command::TxnFile(TxnFileCommand {
            inner_cmd: Some(Box::new(inner_cmd)),
            modified: false,
            txn_file_ref,
            lock_prefix,
            txn_file,
        })
    }

    pub fn priority(&self) -> CommandPri {
        // Force to low priority for prewrite as it's of high throughput but latency
        // insensitive.
        match self.inner_cmd.as_ref().unwrap() {
            box Command::Prewrite(_) => CommandPri::Low,
            cmd => cmd.priority(),
        }
    }

    fn build_prewrite_txn_file_ref(req: &Prewrite, snap: &SnapAccess) -> TxnFileRef {
        let mut lock = txn_types::Lock::new(
            LockType::Put,
            req.primary.clone(),
            req.start_ts,
            req.lock_ttl,
            None,
            TimeStamp::zero(),
            req.txn_size,
            req.min_commit_ts,
        );
        lock.is_txn_file = true;
        let lock_prefix = lock.to_bytes();
        let mut txn_file_ref = TxnFileRef::new();
        txn_file_ref.set_lock_val_prefix(lock_prefix);
        txn_file_ref.set_shard_ver(req.get_ctx().get_region_epoch().get_version());
        txn_file_ref.set_start_ts(req.start_ts.into_inner());
        txn_file_ref.set_chunk_ids(req.txn_file_chunks.clone());
        txn_file_ref.set_inner_lower_bound(snap.get_inner_start().to_vec());
        txn_file_ref.set_inner_upper_bound(snap.get_inner_end().to_vec());
        txn_file_ref
    }

    // Note: modify native_br::lock::LockResolver accordingly if here is changed.
    fn build_commit_txn_file_ref(req: &Commit, snap: &SnapAccess) -> TxnFileRef {
        let mut txn_file_ref = TxnFileRef::new();
        let user_meta = UserMeta::new(req.lock_ts.into_inner(), req.commit_ts.into_inner());
        txn_file_ref.set_user_meta(user_meta.to_array().to_vec());
        txn_file_ref.set_shard_ver(req.get_ctx().get_region_epoch().get_version());
        txn_file_ref.set_start_ts(user_meta.start_ts);
        Self::build_txn_file_ref_from_region(snap, txn_file_ref.start_ts, &mut txn_file_ref);
        txn_file_ref
    }

    // Note: modify native_br::lock::LockResolver accordingly if here is changed.
    fn build_rollback_txn_file_ref(req: &Rollback, snap: &SnapAccess) -> TxnFileRef {
        let mut txn_file_ref = TxnFileRef::new();
        let user_meta = UserMeta::new(req.start_ts.into_inner(), 0);
        txn_file_ref.set_user_meta(user_meta.to_array().to_vec());
        txn_file_ref.set_shard_ver(req.get_ctx().get_region_epoch().get_version());
        txn_file_ref.set_start_ts(user_meta.start_ts);
        Self::build_txn_file_ref_from_region(snap, txn_file_ref.start_ts, &mut txn_file_ref);
        txn_file_ref
    }

    fn build_txn_heartbeat_txn_file_ref(req: &TxnHeartBeat, snap: &SnapAccess) -> TxnFileRef {
        let mut txn_file_ref = TxnFileRef::new();
        txn_file_ref.set_shard_ver(req.get_ctx().get_region_epoch().get_version());
        txn_file_ref.set_start_ts(req.start_ts.into_inner());
        Self::build_txn_file_ref_from_region(snap, txn_file_ref.start_ts, &mut txn_file_ref);
        txn_file_ref
    }

    fn build_check_txn_status_txn_file_ref(req: &CheckTxnStatus, snap: &SnapAccess) -> TxnFileRef {
        let mut txn_file_ref = TxnFileRef::new();
        txn_file_ref.set_shard_ver(req.get_ctx().get_region_epoch().get_version());
        txn_file_ref.set_start_ts(req.ts().into_inner());
        Self::build_txn_file_ref_from_region(snap, txn_file_ref.start_ts, &mut txn_file_ref);
        txn_file_ref
    }

    fn build_resolve_lock_lite_txn_file_ref(
        req: &ResolveLockLite,
        snap: &SnapAccess,
    ) -> TxnFileRef {
        let mut txn_file_ref = TxnFileRef::new();
        txn_file_ref.set_shard_ver(req.get_ctx().get_region_epoch().get_version());
        txn_file_ref.set_start_ts(req.ts().into_inner());
        let user_meta = UserMeta::new(req.start_ts.into_inner(), req.commit_ts.into_inner());
        txn_file_ref.set_user_meta(user_meta.to_array().to_vec());
        Self::build_txn_file_ref_from_region(snap, txn_file_ref.start_ts, &mut txn_file_ref);
        txn_file_ref
    }

    fn build_resolve_lock_txn_file_ref(req: &ResolveLock, snap: &SnapAccess) -> TxnFileRef {
        // If there are multiple txn file transactions to resolve, we resolve them one
        // by one until txn_file_status map is empty.
        let (txn_id, commit_ts) = req.txn_file_status.iter().next().unwrap();
        let mut txn_file_ref = TxnFileRef::new();
        txn_file_ref.set_shard_ver(req.get_ctx().get_region_epoch().get_version());
        txn_file_ref.set_start_ts(txn_id.into_inner());
        let user_meta = UserMeta::new(txn_id.into_inner(), commit_ts.into_inner());
        txn_file_ref.set_user_meta(user_meta.to_array().to_vec());
        Self::build_txn_file_ref_from_region(snap, txn_file_ref.start_ts, &mut txn_file_ref);
        txn_file_ref
    }

    fn try_build_txn_file_ref(cmd: &Command, snap: &SnapAccess) -> Result<TxnFileRef> {
        debug_assert!(cmd.can_build_txn_file());
        Ok(match cmd {
            Command::Prewrite(req) => {
                if !req.mutations.is_empty() {
                    return Err(box_err!(
                        "invalid txn file prewrite with mutations {:?}",
                        cmd
                    ));
                }
                Self::build_prewrite_txn_file_ref(req, snap)
            }
            Command::Commit(req) => Self::build_commit_txn_file_ref(req, snap),
            Command::Rollback(req) => Self::build_rollback_txn_file_ref(req, snap),
            Command::TxnHeartBeat(req) => Self::build_txn_heartbeat_txn_file_ref(req, snap),
            Command::CheckTxnStatus(req) => Self::build_check_txn_status_txn_file_ref(req, snap),
            Command::ResolveLockLite(req) => Self::build_resolve_lock_lite_txn_file_ref(req, snap),
            Command::ResolveLock(req) => Self::build_resolve_lock_txn_file_ref(req, snap),
            _ => unreachable!("unexpected command {:?}", cmd),
        })
    }

    pub(crate) fn try_from<S: Snapshot>(
        cmd: Command,
        snapshot: S,
        txn_chunk_manager: TxnChunkManager,
    ) -> Result<Command> {
        let snap = snapshot.get_kvengine_snap().unwrap();
        let txn_file_ref = Self::try_build_txn_file_ref(&cmd, snap)?;
        let mut txn_chunks = Vec::with_capacity(txn_file_ref.chunk_ids.len());
        let start_ts = txn_file_ref.start_ts;
        let txn_ctx = TxnCtx::from_txn_file_ref(&txn_file_ref);
        for &chunk_id in &txn_file_ref.chunk_ids {
            let txn_chunk = txn_chunk_manager
                .get(chunk_id)
                .ok_or_else(|| -> Error { box_err!("txn chunk not prepared: {}", chunk_id) })?;
            txn_chunks.push(txn_chunk);
        }
        let txn_file_id = TxnFileId::new(cmd.ctx().region_id, txn_file_ref.shard_ver, start_ts);
        let txn_file = TxnFile::new(txn_file_id, txn_chunks, txn_ctx).unwrap();
        Ok(Self::new(cmd, txn_file_ref, txn_file))
    }

    fn build_txn_file_ref_from_region(
        snap: &SnapAccess,
        start_ts: u64,
        txn_file_ref: &mut TxnFileRef,
    ) {
        if let Some(txn_file) = snap.get_lock_txn_file(start_ts) {
            txn_file_ref.set_chunk_ids(txn_file.chunk_ids());
            txn_file_ref.set_inner_lower_bound(txn_file.lower_bound().to_vec());
            txn_file_ref.set_inner_upper_bound(txn_file.upper_bound().to_vec());
        } else {
            txn_file_ref.set_inner_lower_bound(snap.get_inner_start().to_vec());
            txn_file_ref.set_inner_upper_bound(snap.get_inner_end().to_vec());
        }
    }

    fn get_conflict_lock(&self, snap_access: &SnapAccess) -> Option<(Vec<u8>, txn_types::Lock)> {
        if let Some((key, conflict_lock)) = snap_access.get_txn_file_conflict_lock(&self.txn_file) {
            return Some((key, conflict_lock));
        }
        None
    }

    fn get_conflict_write(&self, snap_access: &SnapAccess) -> Option<(Vec<u8>, UserMeta)> {
        if let Some((key, um)) = snap_access.get_txn_file_conflict_write(&self.txn_file) {
            return Some((key, um));
        }
        None
    }

    // TODO: choose best match algorithm based on stats.
    fn check_constraint(&self, snap_access: &SnapAccess) -> crate::storage::mvcc::Result<()> {
        if let Some(already_exist_key) =
            snap_access.check_txn_file_constraint(&self.txn_file, self.ts().into_inner())
        {
            return Err(ErrorInner::AlreadyExist {
                key: already_exist_key,
            }
            .into());
        }
        Ok(())
    }

    #[allow(dead_code)]
    // TODO: support async (for write process).
    fn check_constraint_by_linear_match(
        &self,
        snap_access: &SnapAccess,
    ) -> crate::storage::mvcc::Result<()> {
        let mut txn_file_iter = TxnFileIterator::new(self.txn_file.clone(), false);
        let prefix = if snap_access.get_keyspace_id() > 0 {
            ApiV2::get_keyspace_prefix(snap_access.get_start_key()).unwrap()
        } else {
            &[]
        };
        let prefix_len = prefix.len();
        let mut start_key = prefix.to_vec();
        start_key.extend_from_slice(self.txn_file.smallest().deref());
        let mut end_key = prefix.to_vec();
        end_key.extend_from_slice(self.txn_file.biggest().deref());
        end_key.push(0);
        let mut snap_iter =
            snap_access.new_iterator(WRITE_CF, false, false, Some(self.ts().into_inner()), true);
        snap_iter.set_range(start_key.into(), end_key.into());
        txn_file_iter.rewind();
        while txn_file_iter.valid() && snap_iter.valid() {
            let txn_key = txn_file_iter.key();
            let snap_key = snap_iter.key();
            let txn_op = txn_file_iter.get_op();
            match &snap_key[prefix_len..].cmp(txn_key.deref()) {
                Ordering::Less => {
                    snap_iter.next();
                }
                Ordering::Equal => {
                    if !snap_iter.val().is_empty()
                        && (txn_op == OP_INSERT || txn_op == OP_CHECK_NOT_EXIST)
                    {
                        return Err(ErrorInner::AlreadyExist {
                            key: snap_iter.key().to_vec(),
                        }
                        .into());
                    }
                    snap_iter.next();
                    txn_file_iter.next();
                }
                Ordering::Greater => {
                    txn_file_iter.next();
                }
            }
        }
        Ok(())
    }

    fn try_merge_txn_file_locks(
        &mut self,
        snap_access: &SnapAccess,
    ) -> crate::storage::mvcc::Result<bool /* is_existed_and_equal */> {
        if let Some(existed) = snap_access.get_lock_txn_file(self.txn_file.start_ts()) {
            // The only scene to merge txn file locks is region merge after parts of txn
            // chunks have been prewrite.
            if existed.lower_bound() < self.txn_file.lower_bound()
                || self.txn_file.upper_bound() < existed.upper_bound()
                || self.txn_file.id().shard_id != existed.id().shard_id
                || self.txn_file.shard_ver() < existed.shard_ver()
            {
                error!("unexpected txn file locks of same transaction"; "start_ts" => self.ts(), "txn_file" => ?self.txn_file, "existed" => ?existed);
                return Err(box_err!(
                    "unexpected txn file locks of same transaction, start_ts: {}",
                    self.ts()
                ));
            }

            if self.txn_file.chunk_ids() == existed.chunk_ids()
                && self.txn_file.lower_bound() == existed.lower_bound()
                && self.txn_file.upper_bound() == existed.upper_bound()
            {
                info!("txn file locks are equal"; "start_ts" => self.ts(), "txn_file" => ?self.txn_file, "existed" => ?existed);
                return Ok(true);
            }

            let merged_chunks = TxnFile::merge_chunks(&self.txn_file, &existed);
            let mut txn_ctx = self.txn_file.txn_ctx().clone();
            // Use existed `lock_val_prefix` as it would be updated, e.g, by
            // check_txn_status.
            txn_ctx.set_lock_val_prefix(existed.lock_val_prefix());
            let new_txn_file = TxnFile::new(self.txn_file.id(), merged_chunks, txn_ctx)?;
            new_txn_file.validate()?;

            debug!("merge txn file locks"; "start_ts" => self.ts(), "req" => ?self.txn_file, "existed" => ?existed, "merged" => ?new_txn_file);
            self.txn_file = new_txn_file;
            self.txn_file_ref.set_chunk_ids(self.txn_file.chunk_ids());
            self.txn_file_ref
                .set_lock_val_prefix(self.txn_file.get_lock_val_prefix().to_vec());
        }
        Ok(false)
    }

    fn process_prewrite(
        &mut self,
        snap_access: &SnapAccess,
    ) -> crate::storage::mvcc::Result<ProcessResult> {
        self.txn_file.validate()?;

        if !self.txn_file.has_data_in_bound(snap_access.data_bound()) {
            error!("process_prewrite: txn file has no data in region"; "start_ts" => self.ts());
            return Err(box_err!("txn file has no data in region"));
        }

        let lock_is_existed_and_equal = self.try_merge_txn_file_locks(snap_access)?;
        if lock_is_existed_and_equal {
            info!("process_prewrite: ignore duplicated prewrite"; "start_ts" => self.ts());
        } else if let Some((key, conflict_lock)) = self.get_conflict_lock(snap_access) {
            if conflict_lock.ts != self.ts() {
                return Err(ErrorInner::KeyIsLocked(conflict_lock.into_lock_info(key)).into());
            }
            error!("unexpected lock of same transaction"; "start_ts" => self.ts(), "conflict_lock" => ?conflict_lock, "key" => &LogValue::key(&key));
            return Err(box_err!(
                "unexpected lock of same transaction, start_ts: {}",
                self.ts()
            ));
        } else {
            if let Some(commit_ts) = self.check_txn_commit_record(snap_access)? {
                // already committed
                let lock = self.lock_prefix.as_ref().unwrap();
                let start_ts = lock.ts;
                let key = lock.primary.clone();
                return Err(ErrorInner::Committed {
                    start_ts,
                    commit_ts,
                    key,
                }
                .into());
            }
            if let Some((key, conflict_um)) = self.get_conflict_write(snap_access) {
                let lock_prefix = self.lock_prefix.as_ref().unwrap();
                return Err(ErrorInner::WriteConflict {
                    start_ts: self.ts(),
                    conflict_start_ts: conflict_um.start_ts.into(),
                    conflict_commit_ts: conflict_um.commit_ts.into(),
                    key,
                    primary: lock_prefix.primary.clone(),
                    reason: Default::default(),
                }
                .into());
            }
            if self.txn_file.get_inserts() + self.txn_file.get_check_not_exists() > 0 {
                self.check_constraint(snap_access)?;
            }
            self.modified = true;
        }
        Ok(ProcessResult::PrewriteResult {
            result: types::PrewriteResult {
                locks: vec![],
                min_commit_ts: Default::default(),
                one_pc_commit_ts: Default::default(),
            },
        })
    }

    fn check_txn_commit_record(
        &mut self,
        snap_access: &SnapAccess,
    ) -> crate::storage::mvcc::Result<Option<TimeStamp>> {
        let lock_prefix = self.lock_prefix.as_ref().unwrap();
        let primary = lock_prefix.primary.as_slice();
        if snap_access.get_start_key() <= primary && primary < snap_access.get_end_key() {
            let mut cloud_reader = CloudReader::new(snap_access.clone(), true);
            let primary = Key::from_raw(&lock_prefix.primary);
            if let TxnCommitRecord::SingleRecord { commit_ts, write } =
                cloud_reader.get_txn_commit_record(&primary, lock_prefix.ts)?
            {
                if write.write_type == WriteType::Rollback {
                    return Err(ErrorInner::WriteConflict {
                        start_ts: write.start_ts,
                        conflict_start_ts: write.start_ts,
                        conflict_commit_ts: write.start_ts,
                        key: lock_prefix.primary.clone(),
                        primary: lock_prefix.primary.clone(),
                        reason: WriteConflictReason::SelfRolledBack,
                    }
                    .into());
                } else {
                    // already committed
                    return Ok(Some(commit_ts));
                }
            }
        }
        Ok(None)
    }

    // Return Ok(None) when all keys are out of region, and the status of txn is
    // unknown.
    fn check_txn_commit_record_by_keys(
        &self,
        keys: &[Key],
        start_ts: TimeStamp,
        commit_ts: TimeStamp,
        snap_access: &SnapAccess,
    ) -> crate::storage::mvcc::Result<Option<TimeStamp>> {
        let mut reader = CloudReader::new(snap_access.clone(), true);
        for key in keys {
            let raw_key = key.to_raw()?;
            if snap_access.get_start_key() <= raw_key.as_slice()
                && raw_key.as_slice() < snap_access.get_end_key()
            {
                return match reader.get_txn_commit_record(key, start_ts)?.info() {
                    Some((_, WriteType::Rollback)) | None => {
                        // None: related Rollback has been collapsed.
                        // Rollback: rollback by concurrent transaction.
                        info!(
                            "txn conflict (lock not found)";
                            "key" => %key,
                            "start_ts" => start_ts,
                            "commit_ts" => commit_ts,
                        );
                        Err(ErrorInner::TxnLockNotFound {
                            start_ts,
                            commit_ts,
                            key: raw_key,
                        }
                        .into())
                    }
                    // Committed by concurrent transaction.
                    Some((commit_ts, WriteType::Put))
                    | Some((commit_ts, WriteType::Delete))
                    | Some((commit_ts, WriteType::Lock)) => Ok(Some(commit_ts)),
                };
            }
        }
        Ok(None)
    }

    /// Check the transaction status when a "not primary txn file lock" commit
    /// to primary region (i.e. the region contains primary key).
    ///
    /// This condition will happen when the primary txn file lock is committed
    /// or rollback, and then the primary region is merged.
    ///
    /// See https://github.com/tidbcloud/cloud-storage-engine/issues/1800.
    fn check_commit_primary_region(
        &self,
        lock: &txn_types::Lock,
        lock_txn_file: &TxnFile,
        start_ts: TimeStamp,
        commit_ts: TimeStamp,
        snap_access: &SnapAccess,
    ) -> crate::storage::mvcc::Result<()> {
        let is_primary_txn_file = || {
            let primary_inner =
                InnerKey::from_outer_key(&lock.primary, snap_access.get_inner_key_offset());
            lock_txn_file.lower_bound() <= primary_inner
                && primary_inner < lock_txn_file.upper_bound()
        };

        if snap_access.key_is_in_range(&lock.primary) && !is_primary_txn_file() {
            let txn_lock_not_found = || -> mvcc::Error {
                ErrorInner::TxnLockNotFound {
                    start_ts,
                    commit_ts,
                    key: vec![],
                }
                .into()
            };

            let mut cloud_reader = CloudReader::new(snap_access.clone(), true);
            let record = cloud_reader
                .get_txn_commit_record(&Key::from_raw(&lock.primary), start_ts)?
                .info();
            return match record {
                Some((_, WriteType::Rollback)) | None => Err(txn_lock_not_found()),
                Some((committed_ts, WriteType::Put | WriteType::Delete | WriteType::Lock)) => {
                    debug_assert_eq!(
                        committed_ts, commit_ts,
                        "commit_ts mismatch: start_ts {}, commit_ts {}, committed_ts {}",
                        start_ts, commit_ts, committed_ts
                    );
                    Ok(())
                }
            };
        }
        Ok(())
    }

    fn process_commit(
        &mut self,
        snap_access: &SnapAccess,
        commit_ts: TimeStamp,
        keys: Option<Vec<Key>>,
    ) -> crate::storage::mvcc::Result<ProcessResult> {
        let start_ts = self.ts();
        if commit_ts <= start_ts {
            return Err(box_err!(
                "Invalid transaction tso with start_ts:{start_ts}, commit_ts:{commit_ts}"
            ));
        }
        if let Some(lock_txn_file) = snap_access.get_lock_txn_file(self.txn_file_ref.start_ts) {
            let lock = txn_types::Lock::parse(lock_txn_file.get_lock_val_prefix())?;
            self.check_commit_primary_region(
                &lock,
                &lock_txn_file,
                start_ts,
                commit_ts,
                snap_access,
            )?;

            debug_assert_eq!(lock.ts, self.ts());
            if commit_ts < lock.min_commit_ts {
                info!(
                    "trying to commit with smaller commit_ts than min_commit_ts";
                    "lock" => ?lock,
                    "start_ts" => start_ts,
                    "commit_ts" => commit_ts,
                    "min_commit_ts" => lock.min_commit_ts,
                );
                return Err(ErrorInner::CommitTsExpired {
                    start_ts,
                    commit_ts,
                    key: lock.primary.clone(),
                    min_commit_ts: lock.min_commit_ts,
                }
                .into());
            }
            self.modified = true;
            return Ok(ProcessResult::TxnStatus {
                txn_status: TxnStatus::committed(commit_ts),
            });
        } else if let Some(keys) = keys.as_ref() {
            if keys.is_empty() {
                // There is no way to check committed when there is no key in commit request.
                // So we just return committed.
                info!("process_commit: no keys in txn file commit request"; "start_ts" => self.ts(), "commit_ts" => commit_ts);
                return Ok(ProcessResult::TxnStatus {
                    txn_status: TxnStatus::committed(commit_ts),
                });
            }

            if let Some(_new_commit_ts) = self.check_txn_commit_record_by_keys(
                keys.as_slice(),
                start_ts,
                commit_ts,
                snap_access,
            )? {
                // Normal txn return the commit_ts in request other than the committed record.
                // We keep the same here.
                return Ok(ProcessResult::TxnStatus {
                    txn_status: TxnStatus::committed(commit_ts),
                });
            }
        }
        Err(ErrorInner::TxnLockNotFound {
            start_ts: self.ts(),
            commit_ts,
            key: vec![],
        }
        .into())
    }

    fn process_rollback(
        &mut self,
        snap_access: &SnapAccess,
        keys: Option<Vec<Key>>,
    ) -> crate::storage::mvcc::Result<ProcessResult> {
        let start_ts = self.ts();
        if let Some(lock_txn_file) = snap_access.get_lock_txn_file(self.txn_file_ref.start_ts) {
            // TODO: remove this check when it's stable enough.
            let lock = txn_types::Lock::parse(lock_txn_file.get_lock_val_prefix())?;
            debug_assert_eq!(lock.ts, start_ts);

            self.modified = true;
            Ok(ProcessResult::Res)
        } else if let Some(keys) = keys {
            let mut cloud_reader = CloudReader::new(snap_access.clone(), true);
            for key in keys {
                match self.process_check_txn_status_missing_lock(&mut cloud_reader, key.clone())? {
                    TxnStatus::RolledBack | TxnStatus::LockNotExist => {}
                    TxnStatus::Committed { commit_ts } => {
                        return Err(ErrorInner::Committed {
                            start_ts: self.ts(),
                            commit_ts,
                            key: key.into_raw()?,
                        }
                        .into());
                    }
                    _ => unreachable!(),
                }
            }
            // Note: normal txn will proceed to write the rollback record when lock not
            // exist. But we can not do so here because when lock file not
            // exist, we don't know all the keys.
            Ok(ProcessResult::Res)
        } else {
            Err(ErrorInner::TxnLockNotFound {
                start_ts,
                commit_ts: Default::default(),
                key: vec![],
            }
            .into())
        }
    }

    fn process_check_txn_status(
        &mut self,
        snap_access: &SnapAccess,
        primary_key: Key,
        current_ts: TimeStamp,
        caller_start_ts: TimeStamp,
    ) -> crate::storage::mvcc::Result<ProcessResult> {
        let mut reader = CloudReader::new(snap_access.clone(), true);
        if let Some(lock) = reader.load_lock(&primary_key).unwrap() {
            if lock.ts == self.txn_file_ref.start_ts.into() {
                return self.process_check_txn_status_lock_exists(
                    lock,
                    current_ts,
                    caller_start_ts,
                );
            }
        }
        let txn_status = self.process_check_txn_status_missing_lock(&mut reader, primary_key)?;
        Ok(ProcessResult::TxnStatus { txn_status })
    }

    fn process_check_txn_status_lock_exists(
        &mut self,
        mut lock: txn_types::Lock,
        current_ts: TimeStamp,
        caller_start_ts: TimeStamp,
    ) -> crate::storage::mvcc::Result<ProcessResult> {
        let txn_status = if lock.ts.physical() + lock.ttl < current_ts.physical() {
            // rollback.
            let user_meta = UserMeta::new(self.txn_file_ref.start_ts, 0);
            self.txn_file_ref
                .set_user_meta(user_meta.to_array().to_vec());
            self.txn_file_ref.set_lock_val_prefix(vec![]);
            self.modified = true;
            TxnStatus::TtlExpire
        } else {
            // If lock.min_commit_ts is 0, it's not a large transaction and we can't push
            // forward its min_commit_ts otherwise the transaction can't be committed by
            // old version TiDB during rolling update.
            if !lock.min_commit_ts.is_zero()
                && !caller_start_ts.is_max()
                // Push forward the min_commit_ts so that reading won't be blocked by locks.
                && caller_start_ts >= lock.min_commit_ts
            {
                lock.min_commit_ts = caller_start_ts.next();

                if lock.min_commit_ts < current_ts {
                    lock.min_commit_ts = current_ts;
                }

                lock.short_value = None;
                self.txn_file_ref.set_lock_val_prefix(lock.to_bytes());
                self.modified = true;
            }

            // As long as the primary lock's min_commit_ts > caller_start_ts, locks belong
            // to the same transaction can't block reading. Return MinCommitTsPushed
            // result to the client to let it bypass locks.
            let min_commit_ts_pushed = (!caller_start_ts.is_zero() && lock.min_commit_ts > caller_start_ts)
                // If the caller_start_ts is max, it's a point get in the autocommit transaction.
                // We don't push forward lock's min_commit_ts and the point get can ignore the lock
                // next time because it's not committed yet.
                || caller_start_ts.is_max();

            TxnStatus::uncommitted(lock, min_commit_ts_pushed)
        };
        Ok(ProcessResult::TxnStatus { txn_status })
    }

    fn process_check_txn_status_missing_lock(
        &mut self,
        reader: &mut CloudReader,
        primary_key: Key,
    ) -> crate::storage::mvcc::Result<TxnStatus> {
        let txn_status =
            match reader.get_txn_commit_record(&primary_key, self.txn_file_ref.start_ts.into())? {
                TxnCommitRecord::SingleRecord { commit_ts, write } => {
                    if write.write_type == WriteType::Rollback {
                        TxnStatus::RolledBack
                    } else {
                        TxnStatus::committed(commit_ts)
                    }
                }
                TxnCommitRecord::None { .. } => TxnStatus::LockNotExist,
                _ => unreachable!(),
            };
        Ok(txn_status)
    }

    fn process_txn_heartbeat(
        &mut self,
        snap_access: &SnapAccess,
        primary_key: Vec<u8>,
        advise_ttl: u64,
    ) -> crate::storage::mvcc::Result<ProcessResult> {
        let item = snap_access.get(LOCK_CF, &primary_key, 0);
        let start_ts = TimeStamp::new(self.txn_file_ref.start_ts);
        if item.is_value_empty() {
            return Err(ErrorInner::TxnNotFound {
                start_ts,
                key: primary_key,
            }
            .into());
        }
        let mut lock = txn_types::Lock::parse(item.get_value()).unwrap();
        if lock.ts != start_ts {
            return Err(ErrorInner::TxnNotFound {
                start_ts,
                key: primary_key,
            }
            .into());
        }
        if lock.ttl < advise_ttl {
            lock.ttl = advise_ttl;
            lock.short_value = None;
            self.txn_file_ref.set_lock_val_prefix(lock.to_bytes());
            self.modified = true;
        }
        Ok(ProcessResult::TxnStatus {
            txn_status: TxnStatus::uncommitted(lock, false),
        })
    }

    fn process_resolve_txn_lock(
        &mut self,
        snap_access: &SnapAccess,
        commit_ts: TimeStamp,
        resolve_keys: Option<Vec<Key>>,
    ) -> crate::storage::mvcc::Result<ProcessResult> {
        let res = if commit_ts.is_zero() {
            self.process_rollback(snap_access, resolve_keys)
        } else {
            self.process_commit(snap_access, commit_ts, resolve_keys)
        };
        match res {
            Ok(_) => Ok(ProcessResult::Res),
            Err(err @ mvcc::Error(box ErrorInner::TxnLockNotFound { .. })) => {
                // If the lock is not found, it means the transaction is already committed or
                // rollbacked by others. We can safely ignore it.
                info!("process_resolve_txn_lock: txn lock not found"; "err" => ?err, "start_ts" => self.ts());
                Ok(ProcessResult::Res)
            }
            Err(err) => Err(err),
        }
    }

    fn process_resolve_lock(
        &mut self,
        snap_access: &SnapAccess,
        mut resolve_lock: ResolveLock,
    ) -> crate::storage::mvcc::Result<ProcessResult> {
        let um = UserMeta::from_slice(self.txn_file_ref.get_user_meta());
        self.process_resolve_txn_lock(snap_access, um.commit_ts.into(), None)?;
        let resolved_txn = resolve_lock.txn_file_status.remove(&self.ts());
        debug_assert!(resolved_txn.is_some());
        if resolve_lock.txn_file_status.is_empty() && resolve_lock.txn_status.is_empty() {
            // All locks are resolved.
            return Ok(ProcessResult::Res);
        }
        if resolve_lock.txn_file_status.is_empty() {
            // Only non-txn-file locks.
            let next_cmd = ResolveLockReadPhase {
                ctx: resolve_lock.ctx.clone(),
                deadline: resolve_lock.deadline,
                txn_status: resolve_lock.txn_status,
                scan_key: None,
            };
            return Ok(ProcessResult::NextCommand {
                cmd: Command::ResolveLockReadPhase(next_cmd),
            });
        }
        // More locks to resolve,
        Ok(ProcessResult::NextCommand {
            cmd: Command::ResolveLock(resolve_lock),
        })
    }
}

impl CommandExt for TxnFileCommand {
    fn get_ctx(&self) -> &crate::storage::Context {
        self.inner_cmd.as_ref().unwrap().ctx()
    }
    fn get_ctx_mut(&mut self) -> &mut crate::storage::Context {
        self.inner_cmd.as_mut().unwrap().ctx_mut()
    }
    fn deadline(&self) -> ::tikv_util::deadline::Deadline {
        self.inner_cmd.as_ref().unwrap().deadline()
    }
    tag!(txn_file);

    fn ts(&self) -> TimeStamp {
        TimeStamp::new(self.txn_file_ref.start_ts)
    }

    fn write_bytes(&self) -> usize {
        let mut bytes = self.txn_file_ref.compute_size() as usize;
        if let Command::Prewrite(_) = self.inner_cmd.as_ref().unwrap().as_ref() {
            bytes += self.txn_file.size();
        }
        bytes
    }

    fn gen_lock(&self) -> Lock {
        Lock {
            required_hashes: vec![],
            keyspace_id: self.get_ctx().get_keyspace_id(),
            owned_count: 0,
            region_id: self.inner_cmd.as_ref().unwrap().ctx().region_id,
            start_ts: self.txn_file_ref.start_ts,
            checked_txn_cid: 0,
            txn_file: Some(self.txn_file.clone()),
            count_added: false,
        }
    }
}

impl<S: Snapshot, L: LockManager> WriteCommand<S, L> for TxnFileCommand {
    fn process_write(
        mut self,
        snapshot: S,
        _context: WriteContext<'_, L>,
    ) -> crate::storage::txn::Result<WriteResult> {
        let snap = snapshot.get_kvengine_snap().unwrap();
        let cmd = mem::take(&mut self.inner_cmd).unwrap();
        let ctx = cmd.ctx().clone();
        debug!("txn file process write"; "cmd" => ?cmd, "txn_file_ref" => ?self.txn_file_ref, "ctx" => ?ctx, "snap" => ?snap);

        let pr = match *cmd {
            Command::Prewrite(_) => self.process_prewrite(snap)?,
            Command::Commit(commit) => {
                self.process_commit(snap, commit.commit_ts, Some(commit.keys))?
            }
            Command::Rollback(rollback) => self.process_rollback(snap, Some(rollback.keys))?,
            Command::TxnHeartBeat(txn_heartbeat) => {
                let primary_key = txn_heartbeat.primary_key.to_raw().unwrap();
                self.process_txn_heartbeat(snap, primary_key, txn_heartbeat.advise_ttl)?
            }
            Command::CheckTxnStatus(check_txn_status) => self.process_check_txn_status(
                snap,
                check_txn_status.primary_key,
                check_txn_status.current_ts,
                check_txn_status.caller_start_ts,
            )?,
            Command::ResolveLock(resolve) => self.process_resolve_lock(snap, resolve)?,
            Command::ResolveLockLite(resolve) => {
                self.process_resolve_txn_lock(snap, resolve.commit_ts, Some(resolve.resolve_keys))?
            }
            _ => {
                return Err(box_err!("unsupported txn file command"));
            }
        };
        let mut write_data = WriteData::default();
        if self.modified {
            write_data.txn_file = Some(self.txn_file_ref.clone());
        }
        let result = WriteResult {
            ctx,
            to_be_write: write_data,
            rows: 1,
            pr,
            lock_info: vec![],
            released_locks: ReleasedLocks::new(),
            lock_guards: vec![],
            response_policy: ResponsePolicy::OnApplied,
        };
        Ok(result)
    }
}

impl std::fmt::Display for TxnFileCommand {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "kv::command::txn_file {:?} {:?}",
            self.txn_file_ref,
            self.inner_cmd
                .as_ref()
                .map(|cmd| cmd.to_string())
                .unwrap_or_default(),
        )
    }
}

impl std::fmt::Debug for TxnFileCommand {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self)
    }
}

pub(crate) fn command_chunks_to_load(cmd: &Command) -> &[u64] {
    if let Command::Prewrite(req) = cmd {
        &req.txn_file_chunks
    } else {
        &[]
    }
}
