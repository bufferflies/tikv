// Copyright 2023 TiKV Project Authors. Licensed under Apache-2.0.

use std::{default::Default, sync::Arc};

use cloud_server::modifies_to_requests;
use dashmap::DashMap;
use kvengine::Shard;
use kvproto::kvrpcpb;
use tikv::storage::{
    kv::WriteData,
    mvcc::{CloudReader, Key, MvccTxn, TxnCommitRecord, WriteType},
};
use tikv_util::{box_err, info};
use tokio::sync::RwLock;
use txn_types::{Lock, ReqType, TimeStamp};

use crate::{
    common::{RawRegion, RegionMetaGetter},
    error::Error,
    Result,
};

// Limit the batch size to avoid exceed the size of mem tables.
// At the same time, the batch size should not be too small to reduce the
// overhead of creating applier.
const DEFAULT_TXN_BATCH_SIZE: usize = 32 * 1024 * 1024; // 32MB

#[derive(Clone)]
pub struct LockResolver {
    tag: Arc<String>,
    en: kvengine::Engine,
    shards: Arc<Vec<RawRegion>>,
    shard_meta_getter: RegionMetaGetter,
    truncate_ts: u64,
    batch_size: usize,
    txn_status: TxnStatus,
    // `cm` is actually not used but meet the requirement of `MvccTxn`.
    cm: concurrency_manager::ConcurrencyManager,
}

#[derive(Debug)]
pub struct ResolvedLocks {
    pub total_locks_cnt: usize,
    pub resolved_shards: Vec<u64>, // The shards that have locks which are resolved.
}

impl LockResolver {
    pub fn new(
        tag: &str,
        en: kvengine::Engine,
        truncate_ts: u64,
        batch_size: usize,
        shard_meta_getter: RegionMetaGetter,
    ) -> Self {
        let shards = Arc::new(Self::collect_shards(&en));
        info!("{} collect_shards", tag; "shards" => ?shards);

        let txn_status = TxnStatus {
            en: en.clone(),
            shards: shards.clone(),
            txns: Default::default(),
        };
        Self {
            tag: Arc::new(tag.to_string()),
            en,
            shards,
            shard_meta_getter,
            truncate_ts,
            batch_size,
            txn_status,
            cm: concurrency_manager::ConcurrencyManager::new(1.into()),
        }
    }

    pub async fn resolve_locks(&self) -> Result<ResolvedLocks> {
        let mut resolved_locks = ResolvedLocks {
            total_locks_cnt: 0,
            resolved_shards: vec![],
        };
        let mut handles = Vec::with_capacity(self.shards.len());
        let shards = self.shards.clone();
        for shard in shards.iter() {
            let shard_id = shard.id;
            let clone = self.clone();
            handles.push((
                shard_id,
                tokio::spawn(async move { clone.resolve_shard(shard_id).await }),
            ));
        }
        for (shard_id, handle) in handles {
            let locks_cnt = handle.await.unwrap()?;
            if locks_cnt > 0 {
                resolved_locks.total_locks_cnt += locks_cnt;
                resolved_locks.resolved_shards.push(shard_id);
            }
        }
        Ok(resolved_locks)
    }

    fn collect_shards(en: &kvengine::Engine) -> Vec<RawRegion> {
        let mut shards: Vec<RawRegion> = en
            .collect_shard_id_vers()
            .into_iter()
            .map(|idver| {
                let shard = en.get_shard(idver.id).unwrap();
                RawRegion {
                    id: shard.id,
                    raw_start: shard.outer_start.to_vec(),
                    raw_end: shard.outer_end.to_vec(),
                    ..Default::default()
                }
            })
            .collect();
        shards.sort_unstable_by(|x, y| x.raw_start.cmp(&y.raw_start));
        shards
    }

    async fn resolve_shard(&self, shard_id: u64) -> Result<usize /* locks_cnt */> {
        let mut locks_cnt = 0;
        let mut mvcc_txn = MvccTxn::new(TimeStamp::zero(), self.cm.clone());

        let shard = self.en.get_shard(shard_id).unwrap();
        let snap = shard.new_snap_access();
        let mut reader = CloudReader::new(snap, false); // TODO: check memory usage & enable fill_cache
        let mut start = Key::from_raw(&shard.outer_start);
        let end = Key::from_raw(&shard.outer_end);
        loop {
            // We don't resolve locks with timestamp > max_ts, as they will be
            // truncated even if they are committed.
            let (kv_pairs, is_remain) = reader.scan_locks(
                Some(&start),
                Some(&end),
                |lock| lock.ts.into_inner() <= self.truncate_ts,
                self.batch_size,
            )?;
            info!("{} scan_locks", self.tag; "shard_id" => shard_id, "locks" => ?kv_pairs);
            if is_remain {
                let mut raw_last = kv_pairs
                    .last()
                    .unwrap()
                    .0
                    .to_raw()
                    .map_err(|err| -> Error {
                        box_err!(
                            "to_raw error: key {:?}, err {:?}",
                            kv_pairs.last().unwrap().0,
                            err
                        )
                    })?;
                raw_last.push(0);
                start = Key::from_raw(&raw_last);
            }

            for (key, lock) in kv_pairs {
                let commit_ts = self.txn_status.check_txn_status(&self.tag, &lock).await?;
                if commit_ts > 0 {
                    Self::commit_lock(&self.tag, &mut mvcc_txn, key, &lock, commit_ts.into())?;
                    locks_cnt += 1;

                    if mvcc_txn.write_size() >= DEFAULT_TXN_BATCH_SIZE {
                        let mt = std::mem::replace(
                            &mut mvcc_txn,
                            MvccTxn::new(TimeStamp::zero(), self.cm.clone()),
                        );
                        self.apply(&shard, mt)?;
                    }
                }
            }

            if !is_remain {
                break;
            }
        }

        if !mvcc_txn.is_empty() {
            self.apply(&shard, mvcc_txn)?;
        }
        Ok(locks_cnt)
    }

    // Ref: tikv::storage::txn::actions::commit
    fn commit_lock(
        tag: &str,
        txn: &mut MvccTxn,
        key: Key,
        lock: &Lock,
        commit_ts: TimeStamp,
    ) -> Result<()> {
        info!("{} commit_lock", tag; "key" => ?key, "lock" => ?lock, "commit_ts" => ?commit_ts);
        if commit_ts < lock.min_commit_ts {
            return Err(box_err!(
                "{} trying to commit with smaller commit_ts than min_commit_ts, key {:?}, lock {:?}, commit_ts {:?}, min_commit_ts {:?}",
                tag,
                key,
                lock,
                commit_ts,
                lock.min_commit_ts
            ));
        }

        let write = txn_types::Write::new(
            WriteType::from_lock_type(lock.lock_type).unwrap(),
            lock.ts,
            None,
        )
        .set_last_change(lock.last_change_ts, lock.versions_to_last_change)
        .set_txn_source(lock.txn_source);

        txn.put_write(key, commit_ts, write.as_ref().to_bytes());
        // It's not necessary to unlock key. The lock will be removed on applying
        // commit.
        Ok(())
    }

    fn apply(&self, shard: &Arc<Shard>, txn: MvccTxn) -> Result<()> {
        let store_id = self.en.get_engine_id();
        let region_meta = self
            .shard_meta_getter
            .load_region_meta(shard.id, shard.ver, store_id)
            .ok_or::<Error>(
                box_err!(
                    "{} apply: load_region_meta failed, shard_id {}, shard_ver {}, shard_meta_getter {:?}",
                    self.tag,
                    shard.id,
                    shard.ver,
                    self.shard_meta_getter
                )
            )?;

        let mut write_data = WriteData::from_modifies(txn.into_modifies());
        write_data.set_req_type(ReqType::Commit); // We don't rollback locks, so `Commit` is used.
        let custom_req = modifies_to_requests(&kvrpcpb::Context::default(), &mut write_data);
        rfstore::store::apply_custom_log_in_recover(
            &self.en,
            store_id,
            shard,
            region_meta,
            custom_req,
        )
        .map_err(|err| -> Error {
            box_err!(
                "{} apply_custom_log_in_recover error: shard_id {:?}, err {:?}",
                self.tag,
                shard.id,
                err
            )
        })?;
        Ok(())
    }
}

#[derive(Clone)]
struct TxnStatus {
    en: kvengine::Engine,
    shards: Arc<Vec<RawRegion>>,
    txns: Arc<DashMap<u64 /* txn_id */, Arc<RwLock<Option<u64 /* commit_id */>>>>>,
}

impl TxnStatus {
    pub async fn check_txn_status(&self, tag: &str, lock: &Lock) -> Result<u64 /* commit_ts */> {
        let txn_status = self.txns.entry(lock.ts.into_inner()).or_default().clone();
        {
            let status = txn_status.read().await;
            if let Some(commit_ts) = *status {
                return Ok(commit_ts);
            }
        }

        let mut status = txn_status.write().await;
        if let Some(commit_ts) = *status {
            return Ok(commit_ts);
        }

        let commit_ts = self.check_txn_status_from_engine(lock)?;
        *status = Some(commit_ts);
        info!("{} check_txn_status", tag; "lock" => ?lock, "commit_ts" => ?commit_ts);
        Ok(commit_ts)
    }

    fn check_txn_status_from_engine(&self, lock: &Lock) -> Result<u64 /* commit_id */> {
        let shard_id = self.get_shard_by_key(&lock.primary);
        if shard_id.is_none() {
            return Err(box_err!(
                "shard not found for primary {:?}, lock {:?}",
                tikv_util::escape(&lock.primary),
                lock
            ));
        }

        let shard_id = shard_id.unwrap();
        let mut snap = self.en.get_snap_access(shard_id).unwrap();
        if snap.has_unloaded_tables() {
            self.en
                .load_unloaded_tables(snap.get_id(), snap.get_version(), false)?;
            snap = self.en.get_snap_access(shard_id).unwrap();
        }

        let mut cloud_reader = CloudReader::new(snap, false); // TODO: check memory usage & enable fill_cache
        // Ref: check_txn_status_missing_lock
        match cloud_reader.get_txn_commit_record(&Key::from_raw(&lock.primary), lock.ts)? {
            TxnCommitRecord::SingleRecord { commit_ts, write } => {
                if write.write_type == WriteType::Rollback {
                    Ok(0)
                } else {
                    // TODO: Check secondaries for async commit.
                    Ok(commit_ts.into_inner())
                }
            }
            _ => Ok(0),
        }
    }

    fn get_shard_by_key(&self, key: &[u8]) -> Option<u64 /* shard_id */> {
        self.shards
            .binary_search_by(|x| x.compare_with_key(key))
            .ok()
            .map(|idx| self.shards[idx].id)
    }
}
