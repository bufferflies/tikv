// Copyright 2022 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    collections::{BTreeMap, HashMap, HashSet},
    ops::{
        Bound::{Excluded, Included, Unbounded},
        Deref, DerefMut, Range,
    },
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Mutex,
    },
    thread::{self, sleep},
    time::Duration,
};

use api_version::{
    api_v2::{self, TXN_KEY_PREFIX},
    ApiV2, KvFormat,
};
use futures::executor::block_on;
use grpcio::Channel;
use kvengine::ShardTag;
use kvproto::{
    coprocessor as coppb, errorpb, kvrpcpb,
    kvrpcpb::{
        CommitRequest, Context, GetRequest, IsolationLevel, Mutation, Op, PrewriteRequest,
        SplitRegionRequest,
    },
    metapb,
    metapb::{Peer, Region, RegionEpoch},
    tikvpb::TikvClient,
};
use pd_client::PdClient;
use protobuf::ProtobufEnum;
use rfstore::store::RegionIdVer;
use test_pd_client;
use tikv::storage::mvcc::TimeStamp;
use tikv_client::{
    proto::kvrpcpb::Mutation as KvMutation, CheckLevel, IntoOwnedRange, TransactionOptions,
};
use tikv_util::{
    box_err,
    codec::bytes::{decode_bytes, encode_bytes},
    debug, error, info,
    time::Instant,
    warn,
};

use crate::{
    must_wait, try_wait,
    txnlock::lock_resolver::{LockResolver, ResolveLocksOptions},
};

const MAX_WAIT_LOCK_DURATION: Duration = Duration::from_millis(500);

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Other error {0}")]
    Other(#[from] Box<dyn std::error::Error + Send + Sync>),
    #[error("Pd error {0}")]
    Pd(#[from] pd_client::Error),
    #[error("Write conflict {0:?}")]
    WriteConflict(kvrpcpb::WriteConflict),
    #[error("Transaction not found {0:?}")]
    TxnNotFound(kvrpcpb::TxnNotFound),
    #[error(transparent)]
    Grpc(#[from] grpcio::Error),
    #[error("TiKV Client error {0}")]
    TikvClient(#[from] tikv_client::Error),
}

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Default, Clone, Debug)]
pub struct RefStore(HashMap<Vec<u8>, Option<Vec<u8>>>); // `None` means the key has been deleted.

impl RefStore {
    pub fn put_kv(&mut self, key: Vec<u8>, value: Vec<u8>) {
        self.0.insert(key, Some(value));
    }

    pub fn del_kv(&mut self, key: Vec<u8>) {
        self.0.insert(key, None);
    }

    pub fn ingest(&mut self, other: RefStore) {
        for (k, v) in other.0 {
            self.0.insert(k, v);
        }
    }

    pub fn destroy_range(&mut self, start: &[u8], end: &[u8]) {
        for (k, v) in self.0.iter_mut() {
            if k.as_slice() >= start && k.as_slice() < end {
                *v = None;
            }
        }
    }

    pub fn rewrite_keyspace_prefix(&mut self, target_keyspace_id: u32) {
        let old_ref_store = std::mem::take(&mut self.0);
        let target_prefix = ApiV2::get_txn_keyspace_prefix(target_keyspace_id);
        let inner = old_ref_store
            .into_iter()
            .map(|(mut k, v)| {
                assert_eq!(ApiV2::parse_key_mode(&k), api_version::KeyMode::Txn);
                k[0..api_v2::KEYSPACE_PREFIX_LEN].copy_from_slice(&target_prefix);
                (k, v)
            })
            .collect::<HashMap<_, _>>();
        self.0 = inner;
    }
}

impl Deref for RefStore {
    type Target = HashMap<Vec<u8>, Option<Vec<u8>>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for RefStore {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

pub struct ClusterClient {
    pub pd_client: Arc<dyn test_pd_client::PdClientExt>,
    pub channels: HashMap<u64, Channel>,
    /// region_raw_end_key -> region_id
    pub(crate) region_ranges: BTreeMap<Vec<u8>, RegionIdVer>,
    /// region_id -> region
    pub(crate) regions: HashMap<u64, RawRegion>,
    pub(crate) ref_store: Arc<Mutex<RefStore>>,
    pub(crate) max_ts: AtomicU64,
    pub(crate) async_commit: bool,

    // `lock_resolver` embed a `ClusterClient` to reuse methods to communicate with tikv-server.
    // So we need the `Option<Box>` to resolve circular dependency.
    // TODO: separate methods of RPCs (kv_xxx) from ClusterClient.
    pub(crate) lock_resolver: Option<Box<LockResolver>>,

    pub(crate) api_version: kvrpcpb::ApiVersion,
}

// Named as `RequestPeerRole` to avoid conflict with `metapb::PeerRole`.
pub enum RequestPeerRole {
    Leader,
    Learner,
}

pub struct RequestOptions {
    pub peer_role: RequestPeerRole,
}

impl Default for RequestOptions {
    fn default() -> Self {
        Self {
            peer_role: RequestPeerRole::Leader,
        }
    }
}

impl RequestOptions {
    pub fn replica_read(&self) -> bool {
        !matches!(self.peer_role, RequestPeerRole::Leader)
    }
}

#[derive(Clone, Debug)]
pub struct RawRegion {
    id: u64,
    raw_start: Vec<u8>,
    raw_end: Vec<u8>,
    epoch: RegionEpoch,
    peers: Vec<Peer>,
    leader_idx: usize,
}

impl From<Region> for RawRegion {
    fn from(mut region: Region) -> Self {
        let raw_start = if region.start_key.is_empty() {
            vec![]
        } else {
            let mut slice = region.start_key.as_slice();
            decode_bytes(&mut slice, false).unwrap()
        };
        let raw_end = if region.end_key.is_empty() {
            vec![255; 8]
        } else {
            let mut slice = region.end_key.as_slice();
            decode_bytes(&mut slice, false).unwrap()
        };
        RawRegion {
            id: region.id,
            raw_start,
            raw_end,
            epoch: region.take_region_epoch(),
            peers: region.take_peers().into_vec(),
            leader_idx: 0,
        }
    }
}

impl RawRegion {
    fn get_leader(&self) -> &Peer {
        &self.peers[self.leader_idx]
    }

    fn id_ver(&self) -> RegionIdVer {
        RegionIdVer::new(self.id, self.epoch.version)
    }

    fn update_leader(&mut self, leader: &Peer) -> bool {
        if let Some(idx) = self.peers.iter().position(|p| p.id == leader.id) {
            self.leader_idx = idx;
            true
        } else {
            false
        }
    }

    pub fn id(&self) -> u64 {
        self.id
    }

    pub fn raw_start(&self) -> &[u8] {
        &self.raw_start
    }

    pub fn raw_end(&self) -> &[u8] {
        &self.raw_end
    }

    pub fn peers(&self) -> &[Peer] {
        &self.peers
    }
}

pub enum CommitAction {
    /// Sync commit all keys.
    SyncCommit,
    /// Sync commit the primary key, then async commit the secondary keys.
    /// When `delay == Duration::MAX`, the secondary keys will not be committed.
    AsyncCommitSecondaryKeys(Duration /* delay */),
    /// No key is committed. The transaction should be rolled back.
    NoCommit,
    /// Async commit all keys.
    AsyncCommit(Duration /* delay */),
}

impl Clone for ClusterClient {
    fn clone(&self) -> Self {
        // Do not copy region cache to reduce memory usage.
        let make_clone = || -> Self {
            Self {
                pd_client: self.pd_client.clone(),
                channels: self.channels.clone(),
                region_ranges: Default::default(),
                regions: Default::default(),
                ref_store: self.ref_store.clone(),
                max_ts: Default::default(),
                async_commit: self.async_commit,
                lock_resolver: None,
                api_version: self.api_version,
            }
        };
        let mut cloned = make_clone();
        let lock_resolver = if self.lock_resolver.is_some() {
            Some(LockResolver::new(make_clone()))
        } else {
            None
        };
        cloned.lock_resolver = lock_resolver.map(|x| Box::new(x));
        cloned
    }
}

impl ClusterClient {
    pub fn pd_client(&self) -> Arc<dyn PdClient> {
        self.pd_client.clone() as Arc<dyn PdClient>
    }

    pub fn get_ts(&self) -> TimeStamp {
        block_on(self.pd_client.get_tso()).unwrap()
    }

    pub fn del_table_rows_commit(&mut self, start_ts: TimeStamp, mutations: &[Mutation]) {
        let commit_ts = self.get_ts();
        let keys = mutations.iter().map(|m| m.get_key().to_vec()).collect();
        self.kv_commit(keys, start_ts, commit_ts);
        self.del_kv_in_ref_store(mutations.to_vec());
        self.set_max_ts(commit_ts.into_inner());
    }

    pub fn del_kv<F>(&mut self, rng: Range<usize>, gen_key: F)
    where
        F: Fn(usize) -> Vec<u8>,
    {
        self.try_del_kv(rng, gen_key, CommitAction::SyncCommit)
            .unwrap();
    }

    pub fn try_del_kv<F>(
        &mut self,
        rng: Range<usize>,
        gen_key: F,
        commit_action: CommitAction,
    ) -> Result<()>
    where
        F: Fn(usize) -> Vec<u8>,
    {
        let start_key = gen_key(rng.start);
        let start_ts = self.get_ts();

        let mut mutations = vec![];
        for i in rng {
            let mut m = Mutation::default();
            m.set_op(Op::Del);
            m.set_key(gen_key(i));
            mutations.push(m)
        }
        let keys: Vec<Vec<u8>> = mutations.iter().map(|m| m.get_key().to_vec()).collect();
        let secondaries = if self.async_commit {
            Some(keys[1..].to_vec())
        } else {
            None
        };
        self.kv_prewrite(mutations.clone(), start_key, start_ts, secondaries);
        let commit_ts = self.get_ts();

        match commit_action {
            CommitAction::NoCommit => {
                return Ok(());
            }
            commit_action => self.kv_commit_ext(keys, start_ts, commit_ts, commit_action),
        }

        self.del_kv_in_ref_store(mutations);
        self.set_max_ts(commit_ts.into_inner());
        Ok(())
    }

    fn del_kv_in_ref_store(&mut self, mutations: Vec<Mutation>) {
        let mut ref_store = self.ref_store.lock().unwrap();
        for mut m in mutations {
            ref_store.del_kv(m.take_key());
        }
    }

    pub fn put_commit(&mut self, start_ts: TimeStamp, mutations: &[Mutation]) -> Result<TimeStamp> {
        let put_time = Instant::now();
        let commit_ts = self.get_ts();
        let first = mutations.first().unwrap().clone();
        let keys = mutations.iter().map(|m| m.get_key().to_vec()).collect();
        self.kv_commit(keys, start_ts, commit_ts);
        self.verify_key_value(
            first.get_key(),
            Some(first.get_value()),
            put_time,
            &RequestOptions::default(),
        )?;
        self.put_kv_in_ref_store(mutations.to_vec());
        self.set_max_ts(commit_ts.into_inner());
        Ok(commit_ts)
    }

    pub fn put_kv<F, G>(&mut self, rng: Range<usize>, gen_key: F, gen_val: G)
    where
        F: Fn(usize) -> Vec<u8>,
        G: Fn(usize) -> Vec<u8>,
    {
        self.try_put_kv(rng, gen_key, gen_val, CommitAction::SyncCommit)
            .unwrap();
    }

    pub fn try_put_kv<F, G>(
        &mut self,
        rng: Range<usize>,
        gen_key: F,
        gen_val: G,
        commit_action: CommitAction,
    ) -> Result<()>
    where
        F: Fn(usize) -> Vec<u8>,
        G: Fn(usize) -> Vec<u8>,
    {
        let start_key = gen_key(rng.start);
        let start_ts = self.get_ts();

        let mut mutations = vec![];
        for i in rng {
            let mut m = Mutation::default();
            m.set_op(Op::Put);
            m.set_key(gen_key(i));
            m.set_value(gen_val(i));
            mutations.push(m)
        }
        let keys: Vec<Vec<u8>> = mutations.iter().map(|m| m.get_key().to_vec()).collect();
        let put_time = Instant::now();
        let secondaries = if self.async_commit {
            Some(keys[1..].to_vec())
        } else {
            None
        };
        self.kv_prewrite(mutations.clone(), start_key, start_ts, secondaries);
        let commit_ts = self.get_ts();

        match commit_action {
            CommitAction::NoCommit => {
                return Ok(());
            }
            commit_action => self.kv_commit_ext(keys, start_ts, commit_ts, commit_action),
        }

        let first = mutations.first().unwrap();
        self.verify_key_value(
            first.get_key(),
            Some(first.get_value()),
            put_time,
            &RequestOptions::default(),
        )?;
        self.put_kv_in_ref_store(mutations);
        self.set_max_ts(commit_ts.into_inner());
        Ok(())
    }

    fn put_kv_in_ref_store(&mut self, mutations: Vec<Mutation>) {
        let mut ref_store = self.ref_store.lock().unwrap();
        for mut m in mutations {
            ref_store.put_kv(m.take_key(), m.take_value());
        }
    }

    fn set_max_ts(&mut self, max_ts: u64) {
        self.max_ts.fetch_max(max_ts, Ordering::Relaxed);
    }

    pub fn max_ts(&mut self) -> u64 {
        self.max_ts.load(Ordering::Relaxed)
    }

    pub fn kv_mutate(&mut self, muts: Vec<Mutation>) -> Result<()> {
        assert!(!muts.is_empty());
        let keys = muts
            .iter()
            .map(|m| m.get_key().to_vec())
            .collect::<Vec<_>>();
        let start_ts = self.get_ts();
        self.kv_prewrite(muts, keys[0].clone(), start_ts, None);

        let commit_ts = self.get_ts();
        self.kv_commit(keys, start_ts, commit_ts);
        self.set_max_ts(commit_ts.into_inner());
        Ok(())
    }

    pub fn kv_prewrite(
        &mut self,
        muts: Vec<Mutation>,
        pk: Vec<u8>,
        ts: TimeStamp,
        secondaries: Option<Vec<Vec<u8>>>,
    ) {
        let groups = self.group_mutations_by_region(muts);
        for (id_ver, group_muts) in groups {
            self.kv_prewrite_single_region(id_ver, group_muts, pk.clone(), ts, &secondaries);
        }
    }

    pub fn kv_prewrite_single_region(
        &mut self,
        id_ver: RegionIdVer,
        muts: Vec<Mutation>,
        pk: Vec<u8>,
        ts: TimeStamp,
        secondary_keys: &Option<Vec<Vec<u8>>>,
    ) {
        let region_id = id_ver.id();
        let mut store_id_errors = vec![];
        let start_time = Instant::now();
        let timeout = Duration::from_secs(15);
        while start_time.saturating_elapsed() < timeout {
            let ctx = self
                .new_rpc_ctx(region_id)
                .filter(|x| x.get_region_epoch().get_version() == id_ver.ver());
            if ctx.is_none() {
                self.kv_prewrite(muts, pk, ts, secondary_keys.as_ref().cloned());
                return;
            }
            let ctx = ctx.unwrap();
            let tag = Self::tag_from_ctx(&ctx);
            let store_id = ctx.get_peer().get_store_id();
            let kv_client = self.get_kv_client(store_id);
            let mut prewrite_req = PrewriteRequest::default();
            prewrite_req.set_context(ctx);
            prewrite_req.set_mutations(muts.clone().into());
            prewrite_req.primary_lock = pk.clone();
            prewrite_req.start_version = ts.into_inner();
            prewrite_req.lock_ttl = 3000;
            prewrite_req.min_commit_ts = prewrite_req.start_version + 1;
            prewrite_req.use_async_commit = self.async_commit;
            if let Some(secondary_keys) = secondary_keys {
                if muts[0].get_key() == pk.as_slice() {
                    prewrite_req.set_secondaries(secondary_keys.clone().into());
                }
            }
            debug!("{} prewrite {:?}", tag, prewrite_req);
            let result = kv_client.kv_prewrite(&prewrite_req);
            if result.is_err() {
                store_id_errors.push((store_id, format!("{:?}", result.unwrap_err())));
                sleep(Duration::from_millis(100));
                self.update_cache_by_id(region_id, None);
                continue;
            }
            let mut resp = result.unwrap();
            if resp.has_region_error() {
                let region_err = resp.get_region_error();
                if self.handle_retryable_error(region_id, region_err) {
                    store_id_errors.push((store_id, format!("{:?}", region_err)));
                    continue;
                }
                if self.handle_region_epoch_not_match_or_not_found(region_err) {
                    self.kv_prewrite(muts, pk, ts, secondary_keys.as_ref().cloned());
                    return;
                }
                panic!("unexpected error {:?}", region_err);
            }
            let key_errors = resp.take_errors();
            if !key_errors.is_empty() {
                info!("{} prewrite: encounters key_errors: {:?}", tag, key_errors);
                self.handle_key_errors(
                    &tag,
                    prewrite_req.start_version,
                    false,
                    key_errors.into_vec(),
                )
                .expect("handle_key_errors");
                continue;
            }
            return;
        }
        panic!("{} prewrite failed {:?}", region_id, store_id_errors,);
    }

    pub fn kv_commit(&mut self, keys: Vec<Vec<u8>>, start_ts: TimeStamp, commit_ts: TimeStamp) {
        // fail_point!("kv_commmit");
        // println!("{:?}", fail::list());

        let groups = self.group_keys_by_region(keys);
        for (id_ver, group_keys) in groups {
            self.kv_commit_single_region(id_ver, group_keys, start_ts, commit_ts);
        }
    }

    /// Primary key is the first item of `keys`.
    pub fn kv_commit_ext(
        &mut self,
        keys: Vec<Vec<u8>>,
        start_ts: TimeStamp,
        commit_ts: TimeStamp,
        commit_action: CommitAction,
    ) {
        match commit_action {
            CommitAction::SyncCommit => {
                self.kv_commit(keys, start_ts, commit_ts);
            }
            CommitAction::AsyncCommitSecondaryKeys(delay) => {
                let pk = keys[0].clone();
                self.kv_commit(vec![pk], start_ts, commit_ts);

                if delay < Duration::MAX {
                    let secondary_keys = keys[1..].to_vec();
                    let mut client = self.clone();
                    thread::spawn(move || {
                        thread::sleep(delay);
                        client.kv_commit(secondary_keys, start_ts, commit_ts);
                    });
                }
            }
            CommitAction::NoCommit => {}
            CommitAction::AsyncCommit(delay) => {
                let mut client = self.clone();
                thread::spawn(move || {
                    thread::sleep(delay);
                    client.kv_commit(keys, start_ts, commit_ts);
                });
            }
        }
    }

    pub fn kv_commit_single_region(
        &mut self,
        id_ver: RegionIdVer,
        keys: Vec<Vec<u8>>,
        start_ts: TimeStamp,
        commit_ts: TimeStamp,
    ) {
        let region_id = id_ver.id();
        let mut store_id_errors = vec![];
        let start_time = Instant::now();
        let timeout = Duration::from_secs(15);
        while start_time.saturating_elapsed() < timeout {
            let ctx = self
                .new_rpc_ctx(region_id)
                .filter(|x| x.get_region_epoch().get_version() == id_ver.ver());
            if ctx.is_none() {
                self.kv_commit(keys, start_ts, commit_ts);
                return;
            }
            let ctx = ctx.unwrap();
            let store_id = ctx.get_peer().get_store_id();
            let kv_client = self.get_kv_client(store_id);
            let mut commit_req = CommitRequest::default();
            commit_req.set_context(ctx);
            commit_req.start_version = start_ts.into_inner();
            commit_req.set_keys(keys.clone().into());
            commit_req.commit_version = commit_ts.into_inner();
            let result = kv_client.kv_commit(&commit_req);
            if result.is_err() {
                store_id_errors.push((store_id, format!("{:?}", result.unwrap_err())));
                sleep(Duration::from_millis(100));
                self.update_cache_by_id(region_id, None);
                continue;
            }
            let commit_resp = result.unwrap();
            if commit_resp.has_region_error() {
                let region_err = commit_resp.get_region_error();
                if self.handle_retryable_error(region_id, region_err) {
                    store_id_errors.push((store_id, format!("{:?}", region_err)));
                    continue;
                }
                if self.handle_region_epoch_not_match_or_not_found(region_err) {
                    self.kv_commit(keys, start_ts, commit_ts);
                    return;
                }
                panic!("unexpected error {:?}", region_err);
            }
            if commit_resp.has_error() {
                let key_err = commit_resp.get_error();
                panic!("{} commit failed with key error {:?}", region_id, key_err);
            }
            return;
        }
        panic!("{} commit failed {:?}", region_id, store_id_errors,);
    }

    // TODO: eliminate duplicated codes for handling network & region errors.
    pub fn kv_check_txn_status(
        &mut self,
        primary_key: &[u8],
        lock_ts: u64,
        caller_start_ts: u64,
        current_ts: u64,
        rollback_if_not_exist: bool,
        force_sync_commit: bool,
        resolving_pessimistic_lock: bool,
    ) -> Result<kvrpcpb::CheckTxnStatusResponse> {
        let stat_time = Instant::now();
        let timeout = Duration::from_secs(5);
        let mut last_err: Option<Error> = None;
        let mut tag = ShardTag::default();
        while stat_time.saturating_elapsed() < timeout {
            let region_id = self.get_region_id(primary_key);
            let ctx = self.new_rpc_ctx(region_id).unwrap();
            tag = Self::tag_from_ctx(&ctx);
            let client = self.get_kv_client(ctx.get_peer().get_store_id());
            let mut req = kvrpcpb::CheckTxnStatusRequest::default();
            req.set_context(ctx);
            req.set_primary_key(primary_key.to_vec());
            req.set_lock_ts(lock_ts);
            req.set_caller_start_ts(caller_start_ts);
            req.set_current_ts(current_ts);
            req.set_rollback_if_not_exist(rollback_if_not_exist);
            req.set_force_sync_commit(force_sync_commit);
            req.set_resolving_pessimistic_lock(resolving_pessimistic_lock);
            // TODO: req.set_verify_is_primary
            let result = client.kv_check_txn_status(&req);
            if result.is_err() {
                last_err = Some(box_err!(
                    "{} kv_check_txn_status error: {:?}",
                    tag,
                    result.unwrap_err()
                ));
                warn!("{:?}", last_err);
                sleep(Duration::from_millis(100));
                self.update_cache_by_id(region_id, None);
                continue;
            }

            let resp = result.unwrap();
            if resp.has_region_error() {
                let region_err = resp.get_region_error();
                last_err = Some(box_err!(
                    "{} kv_check_txn_status: region_err {:?}",
                    tag,
                    region_err
                ));
                warn!("{:?}", last_err);
                if self.handle_retryable_error(region_id, region_err) {
                    continue;
                }
                if self.handle_region_epoch_not_match_or_not_found(region_err) {
                    continue;
                }
                panic!("{:?}", last_err);
            }

            return Ok(resp);
        }
        panic!("{} kv_check_txn_status failed {:?}", tag, last_err.unwrap());
    }

    // TODO: eliminate duplicated codes for handling network & region errors.
    // TODO: support lite: resolve single lock when number of keys is small.
    pub fn kv_resolve_lock(
        &mut self,
        start_version: u64,
        commit_version: Option<u64>,
        key: Vec<u8>,
        clean_regions: &mut HashSet<RegionIdVer>,
    ) -> Result<()> {
        let stat_time = Instant::now();
        let timeout = Duration::from_secs(15);
        let mut last_err: Option<Error> = None;
        let mut tag = ShardTag::default();
        while stat_time.saturating_elapsed() < timeout {
            let region = self.get_region_by_key(&key);
            if clean_regions.contains(&region.id_ver()) {
                return Ok(());
            }

            let ctx = self.new_rpc_ctx(region.id()).unwrap();
            tag = Self::tag_from_ctx(&ctx);
            let client = self.get_kv_client(ctx.get_peer().get_store_id());
            let mut req = kvrpcpb::ResolveLockRequest::default();
            req.set_context(ctx);
            req.set_start_version(start_version);
            if let Some(commit_version) = commit_version {
                req.set_commit_version(commit_version);
            }
            let result = client.kv_resolve_lock(&req);
            if result.is_err() {
                last_err = Some(box_err!(
                    "{} kv_resolve_lock error: {:?}",
                    tag,
                    result.unwrap_err()
                ));
                warn!("{:?}", last_err);
                sleep(Duration::from_millis(100));
                self.update_cache_by_id(region.id(), None);
                continue;
            }

            let resp = result.unwrap();
            if resp.has_region_error() {
                let region_err = resp.get_region_error();
                last_err = Some(box_err!(
                    "{} kv_resolve_lock: region_err {:?}",
                    tag,
                    region_err
                ));
                warn!("{:?}", last_err);
                if self.handle_retryable_error(region.id(), region_err) {
                    continue;
                }
                if self.handle_region_epoch_not_match_or_not_found(region_err) {
                    continue;
                }
                panic!("{:?}", last_err);
            }

            if resp.has_error() {
                panic!(
                    "{} kv_resolve_lock failed, start_version {}, error {:?}",
                    tag,
                    start_version,
                    resp.get_error()
                );
            }

            clean_regions.insert(region.id_ver());
            return Ok(());
        }
        panic!("{} kv_resolve_lock failed {:?}", tag, last_err.unwrap());
    }

    fn group_mutations_by_region(
        &mut self,
        mut mutations: Vec<Mutation>,
    ) -> HashMap<RegionIdVer, Vec<Mutation>> {
        let mut groups: HashMap<RegionIdVer, Vec<Mutation>> = HashMap::new();
        for m in mutations.drain(..) {
            let region = self.get_region_by_key(m.get_key());
            groups.entry(region.id_ver()).or_default().push(m);
        }
        groups
    }

    fn group_keys_by_region(
        &mut self,
        mut keys: Vec<Vec<u8>>,
    ) -> HashMap<RegionIdVer, Vec<Vec<u8>>> {
        let mut groups: HashMap<RegionIdVer, Vec<Vec<u8>>> = HashMap::new();
        for key in keys.drain(..) {
            let region = self.get_region_by_key(&key);
            groups.entry(region.id_ver()).or_default().push(key);
        }
        groups
    }

    pub fn get_region_id(&mut self, key: &[u8]) -> u64 {
        let region = self.get_region_by_key(key);
        region.id
    }

    pub fn get_peer_id(&mut self, key: &[u8], store_id: u64) -> u64 {
        let region = self.get_region_by_key(key);
        region
            .peers
            .iter()
            .find(|x| x.store_id == store_id)
            .unwrap()
            .id
    }

    pub fn get_region_by_key(&mut self, key: &[u8]) -> RawRegion {
        if let Some(region) = self.get_region_from_cache(key) {
            return region;
        }
        let region: RawRegion = self
            .pd_client
            .get_region(&encode_bytes(key))
            .unwrap()
            .into();
        for (raw_end, id) in self.get_regions_in_range(&region.raw_start, &region.raw_end) {
            self.region_ranges.remove(&raw_end);
            self.regions.remove(&id);
        }
        self.region_ranges
            .insert(region.raw_end.clone(), region.id_ver());
        self.regions.insert(region.id, region);
        self.get_region_from_cache(key).unwrap()
    }

    pub fn clear_region_cache(&mut self) {
        self.region_ranges.clear();
        self.regions.clear();
    }

    fn update_cache_by_id(&mut self, region_id: u64, opt_region: Option<RawRegion>) {
        let region: RawRegion = if let Some(region) = opt_region {
            region
        } else if let Some((region, leader)) =
            block_on(self.pd_client.get_region_leader_by_id(region_id)).unwrap()
        {
            let mut region = RawRegion::from(region);
            region.update_leader(&leader);
            region
        } else {
            return;
        };
        for (raw_end, id) in self.get_regions_in_range(&region.raw_start, &region.raw_end) {
            self.region_ranges.remove(&raw_end);
            self.regions.remove(&id);
        }
        self.region_ranges
            .insert(region.raw_end.clone(), region.id_ver());
        self.regions.insert(region.id, region);
    }

    fn get_region_from_cache(&self, key: &[u8]) -> Option<RawRegion> {
        if let Some((_, &id_ver)) = self
            .region_ranges
            .range((Excluded(key.to_vec()), Unbounded))
            .next()
        {
            if let Some(region) = self.regions.get(&id_ver.id()) {
                if region.id_ver() == id_ver && region.raw_start.as_slice() <= key {
                    return Some(region.clone());
                }
            }
        }
        None
    }

    fn get_regions_in_range(&self, start: &[u8], end: &[u8]) -> Vec<(Vec<u8>, u64)> {
        self.region_ranges
            .range((Excluded(start.to_vec()), Included(end.to_vec())))
            .map(|(raw_end, &id_ver)| (raw_end.clone(), id_ver.id()))
            .collect()
    }

    fn handle_retryable_error(&mut self, region_id: u64, region_err: &errorpb::Error) -> bool {
        if region_err.has_not_leader() {
            let region = self.regions.get_mut(&region_id).unwrap();
            if region_err.get_not_leader().has_leader() {
                let leader = region_err.get_not_leader().get_leader();
                if region.update_leader(leader) {
                    sleep(Duration::from_millis(100));
                    return true;
                }
            }
            sleep(Duration::from_millis(100));
            self.update_cache_by_id(region_id, None);
            return true;
        }
        if region_err.has_proposal_in_merging_mode() {
            sleep(Duration::from_millis(100));
            return true;
        }
        if region_err.has_stale_command() {
            sleep(Duration::from_millis(100));
            self.update_cache_by_id(region_id, None);
            return true;
        }
        if region_err.has_server_is_busy() {
            sleep(Duration::from_millis(100));
            return true;
        }
        if region_err.has_read_index_not_ready() {
            sleep(Duration::from_millis(100));
            return true;
        }
        if region_err.get_message().contains("mismatch peer") {
            sleep(Duration::from_millis(100));
            self.update_cache_by_id(region_id, None);
            return true;
        }
        if region_err.get_message().contains("proposal dropped") {
            sleep(Duration::from_millis(100));
            return true;
        }
        if region_err
            .get_message()
            .contains("peer has not applied to current term")
        {
            // Occurs when split region.
            // See `rfstore::peer::Peer::propose_normal`.
            sleep(Duration::from_millis(100));
            return true;
        }
        false
    }

    fn handle_region_epoch_not_match_or_not_found(&mut self, region_err: &errorpb::Error) -> bool {
        if region_err.has_epoch_not_match() {
            let not_match = region_err.get_epoch_not_match();
            for region in not_match.get_current_regions() {
                self.update_cache_by_id(region.id, Some(region.clone().into()));
            }
            return true;
        }
        if region_err.has_region_not_found() {
            let region_id = region_err.get_region_not_found().get_region_id();
            if let Some(region) = self.regions.remove(&region_id) {
                self.region_ranges.remove(&region.raw_end);
            }
            return true;
        }
        false
    }

    fn lock_resolver(&mut self) -> &mut LockResolver {
        self.lock_resolver.as_mut().unwrap()
    }

    // Note: only support optimistic transactions by now.
    // TODO: support pessimistic transactions.
    fn handle_key_errors(
        &mut self,
        tag: &ShardTag,
        start_ts: u64,
        is_pessimistic: bool,
        key_errors: Vec<kvrpcpb::KeyError>,
    ) -> Result<()> {
        debug_assert!(!is_pessimistic);

        let mut locks: Vec<kvrpcpb::LockInfo> = Vec::with_capacity(key_errors.len());
        for key_err in key_errors {
            let lock = self.handle_single_key_error(tag, start_ts, is_pessimistic, key_err)?;
            locks.push(lock);
        }
        let opts = ResolveLocksOptions {
            caller_start_ts: start_ts,
            locks,
        };
        let res = self.lock_resolver().resolve_locks(opts)?;
        if res.until_expire_ms() > 0 {
            info!(
                "{} txn {} resolve_locks: sleep {}ms for locks",
                tag,
                start_ts,
                res.until_expire_ms()
            );
            // Sleep no more than MAX_WAIT_LOCK_DURATION to speed up tests.
            sleep(Duration::from_millis(res.until_expire_ms()).min(MAX_WAIT_LOCK_DURATION));
        } else {
            info!(
                "{} txn {} resolve_locks: all locks are resolved",
                tag, start_ts
            );
        }
        Ok(())
    }

    fn handle_single_key_error(
        &self,
        tag: &ShardTag,
        start_ts: u64,
        is_pessimistic: bool,
        mut key_err: kvrpcpb::KeyError,
    ) -> Result<kvrpcpb::LockInfo> {
        if key_err.has_locked() {
            let lock = key_err.take_locked();
            debug!(
                "{} encounters lock, start_ts {}, lock {:?}",
                tag, start_ts, lock
            );

            // If an optimistic transaction encounters a lock with larger ts, this
            // transaction will certainly fail due to a WriteConflict error.
            // So we can construct and return an error here early.
            // Pessimistic transactions don't need such an optimization. If this key needs a
            // pessimistic lock, TiKV will return a PessimisticLockNotFound
            // error directly if it encounters a different lock. Otherwise,
            // TiKV returns lock.lock_ttl = 0, and we still need to resolve the lock.
            if lock.lock_version > start_ts && !is_pessimistic {
                let write_conflict = kvrpcpb::WriteConflict {
                    start_ts,
                    conflict_ts: lock.lock_version,
                    key: lock.key,
                    conflict_commit_ts: 0,
                    reason: kvrpcpb::WriteConflictReason::Optimistic,
                    ..Default::default()
                };
                return Err(Error::WriteConflict(write_conflict));
            }

            return Ok(lock);
        }

        // TODO: handle already_exist error
        Err(box_err!("{} unexpected key error {:?}", tag, key_err))
    }

    pub fn get_kv_client(&self, store_id: u64) -> TikvClient {
        TikvClient::new(self.get_client_channel(store_id))
    }

    pub fn get_client_channel(&self, store_id: u64) -> Channel {
        self.channels.get(&store_id).unwrap().clone()
    }

    pub fn get_stores(&self) -> Vec<u64> {
        self.channels.keys().copied().collect()
    }

    pub fn new_rpc_ctx(&mut self, region_id: u64) -> Option<Context> {
        self.new_rpc_ctx_opt(region_id, &RequestOptions::default())
    }

    fn tag_from_ctx(ctx: &Context) -> kvengine::ShardTag {
        kvengine::ShardTag {
            engine_id: ctx.get_peer().get_store_id(),
            id_ver: kvengine::IdVer::new(ctx.get_region_id(), ctx.get_region_epoch().get_version()),
        }
    }

    fn get_peer_for_request(
        &self,
        region_id: u64,
        options: &RequestOptions,
    ) -> Option<(&RawRegion, &Peer)> {
        self.regions
            .get(&region_id)
            .and_then(|region| match options.peer_role {
                RequestPeerRole::Leader => Some((region, region.get_leader())),
                RequestPeerRole::Learner => {
                    let learner = region
                        .peers
                        .iter()
                        .find(|peer| peer.role == metapb::PeerRole::Learner);
                    if learner.is_none() {
                        warn!("{} has no learner, region: {:?}", region_id, region);
                    }
                    learner.map(|learner| (region, learner))
                }
            })
    }

    pub fn new_rpc_ctx_opt(&mut self, region_id: u64, options: &RequestOptions) -> Option<Context> {
        if self.get_peer_for_request(region_id, options).is_none() {
            let ok = try_wait(
                || {
                    self.update_cache_by_id(region_id, None);
                    self.get_peer_for_request(region_id, options).is_some()
                },
                3,
            );
            if !ok {
                return None;
            }
        }
        let (region, peer) = self.get_peer_for_request(region_id, options).unwrap();
        let mut ctx = Context::new();
        ctx.set_api_version(self.api_version);
        ctx.set_region_id(region_id);
        ctx.set_region_epoch(region.epoch.clone());
        ctx.set_peer(peer.clone());
        ctx.set_replica_read(options.replica_read());
        Some(ctx)
    }

    pub fn split(&mut self, key: &[u8]) {
        self.try_split(key).expect("ClusterClient::split");
    }

    pub fn try_split(&mut self, key: &[u8]) -> Result<()> {
        for _ in 0..10 {
            let region_id = self.get_region_id(key);
            let ctx = self.new_rpc_ctx(region_id).unwrap();
            let client = self.get_kv_client(ctx.get_peer().get_store_id());
            let mut split_req = SplitRegionRequest::default();
            split_req.set_context(ctx);
            split_req.set_split_key(key.to_vec());
            let mut resp = client.split_region(&split_req)?;
            if resp.has_region_error() {
                let region_err = resp.get_region_error();
                if self.handle_retryable_error(region_id, region_err) {
                    sleep(Duration::from_millis(100));
                    continue;
                }
                if self.handle_region_epoch_not_match_or_not_found(region_err) {
                    sleep(Duration::from_millis(100));
                    continue;
                }
                return Err(box_err!(
                    "failed to split key {:?} error {:?}",
                    key,
                    region_err
                ));
            }
            for region in resp.take_regions().into_iter() {
                self.update_cache_by_id(region.get_id(), Some(region.into()));
            }
            return Ok(());
        }
        Err(box_err!("failed to split key {:?}", key))
    }

    pub fn split_keyspace(&mut self, keyspace_id: u32) {
        let (start_key, end_key) = api_version::ApiV2::get_txn_keyspace_range(keyspace_id);
        let encoded_start = encode_bytes(&start_key);
        let encoded_end = encode_bytes(&end_key);
        let region = self.pd_client.get_region(&encoded_start).unwrap();
        let keys = vec![encoded_start, encoded_end];
        self.pd_client
            .must_split_region(region, kvproto::pdpb::CheckPolicy::Usekey, keys);
    }

    pub fn split_keyspaces(&mut self, keyspace_ids: Range<u32>) {
        let mut keys = vec![];
        for keyspace_id in keyspace_ids {
            let prefix = api_version::ApiV2::get_txn_keyspace_prefix(keyspace_id);
            keys.push(encode_bytes(&prefix));
        }
        let region = self.pd_client.get_region(keys.first().unwrap()).unwrap();
        self.pd_client
            .split_region(region, kvproto::pdpb::CheckPolicy::Usekey, keys.clone());
        must_wait(
            || {
                let region = self.pd_client.get_region(keys.first().unwrap()).unwrap();
                region.get_end_key() == keys.get(1).unwrap()
            },
            20,
            "split_keyspace",
        );
    }

    pub fn merge(&mut self, source_key: &[u8], target_key: &[u8]) {
        let source_region = self
            .pd_client
            .get_region(&encode_bytes(source_key))
            .unwrap();
        let target_region = self
            .pd_client
            .get_region(&encode_bytes(target_key))
            .unwrap();
        assert_ne!(source_region.id, target_region.id);
        self.pd_client
            .merge_region(source_region.id, target_region.id);
    }

    /// Try to merge.
    /// Return true: merge request sent.
    /// Return false: `source_key` & `target_key` had been merged.
    pub fn try_merge(&mut self, source_key: &[u8], target_key: &[u8]) -> bool /* sent */ {
        let source_region = self
            .pd_client
            .get_region(&encode_bytes(source_key))
            .unwrap();
        let target_region = self
            .pd_client
            .get_region(&encode_bytes(target_key))
            .unwrap();
        if source_region.id == target_region.id {
            return false;
        }
        self.pd_client
            .try_merge_region(source_region.id, target_region.id);
        true
    }

    pub fn try_merge_adjacent_region(
        &mut self,
        source_key: &[u8],
        boundary_prefix: Option<&[u8]>,
        timeout: Duration,
    ) -> Result<()> {
        let source_region = self.pd_client.get_region(source_key)?;
        let raw_end_key = rfstore::store::raw_end_key(&source_region);
        if let Some(prefix) = boundary_prefix {
            if !raw_end_key.starts_with(prefix) {
                return Err(box_err!(
                    "adjacent region out of boundary prefix, raw_end_key: {:?}",
                    raw_end_key
                ));
            }
        }

        let target_region = self.pd_client.get_region(source_region.get_end_key())?;
        assert_ne!(source_region.id, target_region.id);
        self.pd_client
            .merge_region(source_region.id, target_region.id);

        let start = Instant::now();
        loop {
            if block_on(self.pd_client.get_region_by_id(source_region.id))
                .unwrap()
                .is_none()
            {
                break;
            }
            if start.saturating_elapsed() >= timeout {
                return Err(box_err!("region {:?} is still not merged.", source_region));
            }
            std::thread::sleep(Duration::from_millis(100));
        }

        Ok(())
    }

    pub fn must_get_key(&mut self, key: &[u8], put_time: Instant) -> (Vec<u8>, Context) {
        self.must_get_key_version(key, u64::MAX, put_time)
    }

    pub fn must_get_key_version(
        &mut self,
        key: &[u8],
        version: u64,
        put_time: Instant,
    ) -> (Vec<u8>, Context) {
        let (value, ctx) = self
            .get_key_version_opt(key, version, put_time, &RequestOptions::default())
            .unwrap();
        let value = value.unwrap_or_else(|| {
            panic!(
                "{} key {} not found",
                Self::tag_from_ctx(&ctx),
                log_wrappers::hex_encode_upper(key),
            )
        });
        (value, ctx)
    }

    pub fn get_memtable_snapshot(
        &mut self,
        req_ctx: Option<kvproto::kvrpcpb::Context>,
        start_ts: u64,
        ranges: Vec<coppb::KeyRange>,
    ) -> Result<(Vec<u8>, Vec<u8>, Context)> {
        assert!(ranges.len() == 1);
        let start_time = Instant::now();
        let timeout = Duration::from_secs(15);
        let mut region_id = 0;
        let mut store_id_errors = vec![];

        let mut req = coppb::DelegateRequest::default();

        // All ranges should be in the same region.
        for range in ranges {
            let id = self.get_region_id(range.get_start());
            if region_id != 0 && id != region_id {
                return Err(box_err!("Key ranges not in the same region".to_owned()));
            }
            region_id = id;
            req.mut_ranges().push(range);
        }

        if let Some(req_ctx) = req_ctx {
            req.set_context(req_ctx);
        }
        req.set_start_ts(start_ts);
        req.mut_context().set_isolation_level(IsolationLevel::Si);

        while start_time.saturating_elapsed() < timeout {
            let ctx = self.new_rpc_ctx(region_id);

            if ctx.is_none() {
                continue;
            }

            let ctx = ctx.unwrap();
            let store_id = ctx.get_peer().get_store_id();
            let client = self.get_kv_client(store_id);

            req.set_context(ctx.clone());

            let result = client.delegate_coprocessor(&req);

            if result.is_err() {
                store_id_errors.push((store_id, format!("{:?}", result.unwrap_err())));
                sleep(Duration::from_millis(100));
                continue;
            }

            let mut resp = result.unwrap();

            if resp.has_locked() {
                sleep(Duration::from_millis(100));
                continue;
            }

            let other_err = resp.get_other_error();

            if !other_err.is_empty() {
                panic!("unexpected error {:?}", other_err);
            } else if resp.has_region_error() {
                let region_err = resp.get_region_error();

                store_id_errors.push((store_id, format!("{:?}", region_err)));

                if self.handle_retryable_error(region_id, region_err) {
                    continue;
                }

                if self.handle_region_epoch_not_match_or_not_found(region_err) {
                    continue;
                }
                panic!("unexpected error {:?}", region_err);
            }

            return Ok((resp.take_mem_table_data(), resp.take_snapshot(), ctx));
        }

        panic!(
            "region {} failed to get memtable snapshot: {:?}",
            region_id, store_id_errors
        );
    }

    pub fn get_key_version_opt(
        &mut self,
        key: &[u8],
        version: u64,
        put_time: Instant,
        options: &RequestOptions,
    ) -> Result<(Option<Vec<u8>>, Context)> {
        let start_time = Instant::now();
        let timeout = Duration::from_secs(15);
        let mut tag = ShardTag::default();
        let mut store_id_errors = vec![];
        while start_time.saturating_elapsed() < timeout {
            let region_id = self.get_region_id(key);
            let ctx = self.new_rpc_ctx_opt(region_id, options);
            if ctx.is_none() {
                continue;
            }
            let ctx = ctx.unwrap();
            tag = Self::tag_from_ctx(&ctx);
            let store_id = ctx.get_peer().get_store_id();
            let client = self.get_kv_client(store_id);
            let mut get_req = GetRequest::default();
            get_req.set_context(ctx.clone());
            get_req.set_key(key.to_vec());
            get_req.set_version(version);
            let result = client.kv_get(&get_req);
            if result.is_err() {
                store_id_errors.push((store_id, format!("{:?}", result.unwrap_err())));
                sleep(Duration::from_millis(100));
                self.update_cache_by_id(region_id, None);
                continue;
            }
            let mut resp = result.unwrap();
            if resp.has_region_error() {
                let region_err = resp.get_region_error();
                store_id_errors.push((store_id, format!("{:?}", region_err)));
                if self.handle_retryable_error(region_id, region_err) {
                    continue;
                }
                if self.handle_region_epoch_not_match_or_not_found(region_err) {
                    continue;
                }

                if region_err
                    .get_message()
                    .contains("peer is applying snapshot")
                {
                    continue;
                }
                return Err(box_err!("{} unexpected error {:?}", tag, region_err));
            }
            if resp.has_error() {
                let key_err = resp.take_error();
                info!(
                    "{} get_key_version_opt: encounters key_error: {:?}",
                    tag, key_err
                );
                self.handle_key_errors(&tag, version, false, vec![key_err])
                    .expect("handle_key_errors");
                continue;
            }
            if resp.get_not_found() {
                return Ok((None, ctx));
            }
            return Ok((Some(resp.take_value()), ctx));
        }
        Err(box_err!(
            "{} failed to get key {}, errors {:?}, put elapsed {:?}",
            tag,
            log_wrappers::hex_encode_upper(key),
            store_id_errors,
            put_time.saturating_elapsed()
        ))
    }

    pub fn verify_data_with_ref_store(&mut self) {
        let ref_store = self.ref_store.lock().unwrap().clone();
        self.verify_data_with_given_ref_store(&ref_store, None, &RequestOptions::default())
            .expect("verify_data_with_ref_store");
    }

    pub fn verify_data_with_given_ref_store(
        &mut self,
        ref_store: &RefStore,
        range: Option<(&[u8], &[u8])>,
        options: &RequestOptions,
    ) -> Result<usize> {
        let mut cnt = 0;
        let start_time = Instant::now();
        for (k, v) in ref_store.iter() {
            if let Some(range) = range {
                if k.as_slice() < range.0 || k.as_slice() >= range.1 {
                    continue;
                }
            }
            self.verify_key_value(k, v.as_ref(), start_time, options)?;
            cnt += 1;
        }
        info!(
            "verify_data_with_given_ref_store: verified keys {}, takes {:?}",
            cnt,
            start_time.saturating_elapsed()
        );
        Ok(cnt)
    }

    pub fn verify_key_value<T: AsRef<[u8]> + ?Sized>(
        &mut self,
        key: &[u8],
        expect_val: Option<&T>,
        put_time: Instant,
        options: &RequestOptions,
    ) -> Result<()> {
        let (val, ctx) = self.get_key_version_opt(key, u64::MAX, put_time, options)?;
        let val = val.as_deref();
        let expect_val = expect_val.map(|v| v.as_ref());
        if val != expect_val {
            return Err(box_err!(
                "{} val not equal for key {}, db: {:?}, ref store {:?}",
                Self::tag_from_ctx(&ctx),
                log_wrappers::hex_encode_upper(key),
                val.map(|v| (v.len(), tikv_util::escape(v))),
                expect_val.map(|v| (v.len(), tikv_util::escape(v)))
            ));
        }
        Ok(())
    }

    pub fn ref_store_contains_key(&self, key: &[u8]) -> bool {
        self.ref_store.lock().unwrap().contains_key(key)
    }

    pub fn dump_ref_store(&mut self) -> RefStore {
        self.ref_store.lock().unwrap().clone()
    }

    pub fn ingest_ref_store(&mut self, mut ref_store: RefStore) {
        *self.ref_store.lock().unwrap() = std::mem::take(&mut ref_store);
    }

    pub fn set_async_commit(&mut self) {
        self.async_commit = true;
    }
}

const MIN_TXN_KEY: &[u8] = &[TXN_KEY_PREFIX];
const MAX_TXN_KEY: &[u8] = &[TXN_KEY_PREFIX, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff];

#[derive(Clone, Default)]
pub struct ApiV2NoPrefixCodec {}

impl tikv_client::codec::Codec for ApiV2NoPrefixCodec {
    fn encode_request<R: tikv_client::request::KvRequest>(&self, req: &mut R) {
        req.set_api_version(tikv_client::proto::kvrpcpb::ApiVersion::V2);
    }
}

type TxnClient = tikv_client::TransactionClient<ApiV2NoPrefixCodec>;

/// ClusterTxnClient provides transaction operations.
pub struct ClusterTxnClient {
    pub inner: TxnClient,

    // Used to get extra info for debug.
    pd_client: Arc<dyn PdClient>,
    cluster_client: ClusterClient,
}

impl Deref for ClusterTxnClient {
    type Target = TxnClient;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl ClusterTxnClient {
    pub fn new(
        inner: TxnClient,
        pd_client: Arc<dyn PdClient>,
        cluster_client: ClusterClient,
    ) -> Self {
        Self {
            inner,
            pd_client,
            cluster_client,
        }
    }

    /// Verify data by scanning the given range.
    pub async fn verify_data_by_scan(
        &mut self,
        ref_store: &RefStore,
        range: Option<(&[u8], &[u8])>,
    ) -> Result<usize> {
        let start_time = Instant::now();
        let mut ref_cnt = 0;
        for (k, v) in ref_store.iter() {
            if let Some(range) = range {
                if k.as_slice() < range.0 || k.as_slice() >= range.1 {
                    continue;
                }
            }
            if v.is_some() {
                ref_cnt += 1;
            }
        }
        let limit = ref_cnt + 1; // +1 to check if there are more keys
        let kv_pairs = self
            .kv_scan(range, limit, Duration::from_secs(60))
            .await
            .unwrap()
            .collect::<Vec<_>>();

        let mut db_cnt = 0;
        for kv in &kv_pairs {
            db_cnt += 1;
            let key: &[u8] = kv.key().into();
            let ref_val = ref_store.get(key);

            // Not found in ref store.
            if ref_val.is_none() || ref_val.as_ref().unwrap().is_none() {
                let err: Error = box_err!(
                    "verify_data_by_scan: not found in ref store, db: {} -> {}, ref_val {:?}",
                    log_wrappers::hex_encode_upper(key),
                    tikv_util::escape(kv.value()),
                    ref_val
                );
                self.log_verify_error(key, None, &err);
                return Err(err);
            }

            // Value not match.
            let ref_val = ref_val.as_ref().unwrap().as_ref().unwrap();
            if ref_val != kv.value() {
                let err: Error = box_err!(
                    "verify_data_by_scan: value not match for key {}, db: {}(len:{}), ref store: {}(len:{})",
                    log_wrappers::hex_encode_upper(key),
                    tikv_util::escape(kv.value()),
                    kv.value().len(),
                    tikv_util::escape(ref_val),
                    ref_val.len()
                );
                self.log_verify_error(key, Some(ref_val), &err);
                return Err(err);
            }
        }

        // Not found in db.
        if ref_cnt != db_cnt {
            assert!(ref_cnt > db_cnt, "ref_cnt: {}, db_cnt: {}", ref_cnt, db_cnt);
            let mut diff_cnt = 0;
            for (key, ref_val) in ref_store.iter() {
                if ref_val.is_none() {
                    continue;
                }
                if let Some(range) = range {
                    if key.as_slice() < range.0 || key.as_slice() >= range.1 {
                        continue;
                    }
                }

                let res = kv_pairs.binary_search_by(|kv| {
                    let k: &[u8] = kv.key().into();
                    k.cmp(key)
                });
                if res.is_err() {
                    let err: Error = box_err!(
                        "verify_data_by_scan: not found in db, ref store: {} -> {:?}",
                        log_wrappers::hex_encode_upper(key),
                        tikv_util::escape(ref_val.as_ref().unwrap())
                    );
                    self.log_verify_error(key, ref_val.as_ref().map(|x| x.as_slice()), &err);

                    diff_cnt += 1;
                    if diff_cnt >= ref_cnt - db_cnt {
                        break;
                    }
                }
            }
            return Err(box_err!(
                "verify_data_by_scan: entries count not match, db: {}, ref store: {}",
                db_cnt,
                ref_cnt
            ));
        }

        info!(
            "verify_data_by_scan: verified entries {}, takes {:?}",
            ref_cnt,
            start_time.saturating_elapsed()
        );
        Ok(ref_cnt)
    }

    pub async fn kv_scan(
        &self,
        range: Option<(&[u8], &[u8])>,
        limit: usize,
        timeout: Duration,
    ) -> tikv_client::Result<impl Iterator<Item = tikv_client::KvPair>> {
        let start_ts = self.inner.current_timestamp().await?;
        let scan_range = if let Some(range) = range {
            (range.0, range.1)
        } else {
            (MIN_TXN_KEY, MAX_TXN_KEY)
        };
        let tag = self.tag_from_key("kv_scan", scan_range.0);

        let mut snapshot = self
            .inner
            .snapshot(start_ts, TransactionOptions::new_pessimistic());

        let start_time = Instant::now();
        let mut last_error: Option<tikv_client::Error> = None;
        while start_time.saturating_elapsed() < timeout {
            match tokio::time::timeout(
                timeout,
                snapshot.scan(scan_range.into_owned(), limit as u32),
            )
            .await
            {
                Ok(Ok(kvs)) => return Ok(kvs),
                Ok(Err(err)) if Self::is_kv_error_retryable(&tag, &err) => {
                    self.log_kv_error(&tag, &err);
                    last_error = Some(err);
                    tokio::time::sleep(Duration::from_millis(200)).await;
                    continue;
                }
                Ok(Err(err)) => {
                    self.log_kv_error(&tag, &err);
                    return Err(err);
                }
                Err(elapsed) => {
                    let err_msg = format!("{} kv_scan timeout: {:?}", tag, elapsed);
                    error!("{}", err_msg);
                    return Err(tikv_client::Error::StringError(err_msg));
                }
            }
        }
        Err(last_error.unwrap())
    }

    async fn kv_mutate_inner(
        &self,
        txn: &mut tikv_client::Transaction<ApiV2NoPrefixCodec>,
        muts: Vec<KvMutation>,
        started_commit: &mut bool,
    ) -> std::result::Result<(), tikv_client::Error> {
        if !*started_commit {
            txn.batch_mutate(muts).await?;
        }

        *started_commit = true;
        let _ = txn.commit().await?;
        Ok(())
    }

    pub async fn kv_mutate(&self, muts: Vec<Mutation>, timeout: Duration) -> Result<()> {
        assert!(!muts.is_empty());
        let tag = self.tag_from_key("kv_mutate", muts[0].get_key());
        let muts: Vec<KvMutation> = muts
            .into_iter()
            .map(|m| KvMutation {
                op: m.op.value(),
                key: m.key,
                value: m.value,
                ..Default::default()
            })
            .collect();

        // Must use the same transaction during retries.
        // Otherwise the later transaction in retries would be blocked by the locks of a
        // previous one.
        let option = TransactionOptions::new_pessimistic().drop_check(CheckLevel::Warn);
        let mut txn = self.begin_with_options(option).await?;
        let mut started_commit = false;

        let start_time = Instant::now();
        let mut last_error: Option<tikv_client::Error> = None;
        while start_time.saturating_elapsed() < timeout {
            match self
                .kv_mutate_inner(&mut txn, muts.clone(), &mut started_commit)
                .await
            {
                Ok(_) => return Ok(()),
                Err(err) if Self::is_kv_error_retryable(&tag, &err) => {
                    self.log_kv_error(&tag, &err);
                    last_error = Some(err);
                    tokio::time::sleep(Duration::from_millis(200)).await;
                    continue;
                }
                Err(err) => {
                    self.log_kv_error(&tag, &err);
                    return Err(err.into());
                }
            }
        }
        Err(last_error.unwrap().into())
    }

    pub async fn kv_unsafe_destroy_range(
        &self,
        start: &[u8],
        end: &[u8],
        timeout: Duration,
    ) -> Result<()> {
        let tag = self.tag_from_key("kv_unsafe_destroy_range", start);
        let is_error_retryable = |err: &tikv_client::Error| {
            // Will get `MultipleKeyErrors` when region version not match.
            matches!(err, tikv_client::Error::MultipleKeyErrors(_))
                || Self::is_kv_error_retryable(&tag, err)
        };

        let start_time = Instant::now();
        let mut last_error: Option<tikv_client::Error> = None;
        while start_time.saturating_elapsed() < timeout {
            match self
                .inner
                .unsafe_destroy_range((start, end).into_owned())
                .await
            {
                Ok(_) => return Ok(()),
                Err(err) if is_error_retryable(&err) => {
                    self.log_kv_error(&tag, &err);
                    last_error = Some(err);
                    tokio::time::sleep(Duration::from_millis(200)).await;
                    continue;
                }
                Err(err) => {
                    self.log_kv_error(&tag, &err);
                    return Err(err.into());
                }
            }
        }
        Err(last_error.unwrap().into())
    }

    pub(crate) fn is_kv_error_retryable(tag: &str, err: &tikv_client::Error) -> bool {
        match err {
            tikv_client::Error::Grpc(_)
            | tikv_client::Error::GrpcAPI(_)
            | tikv_client::Error::Channel(_) => true,
            tikv_client::Error::RegionError(_) => true,
            tikv_client::Error::PessimisticLockError { inner, .. } => {
                Self::is_kv_error_retryable(tag, inner)
            }
            tikv_client::Error::MultipleKeyErrors(key_errors) => key_errors
                .iter()
                .all(|err| Self::is_kv_error_retryable(tag, err)),
            tikv_client::Error::KeyError(key_err) => {
                Self::is_key_error_retryable(tag, key_err.as_ref())
            }
            tikv_client::Error::ResolveLockError(_) => true,
            _ => false,
        }
    }

    fn is_key_error_retryable(tag: &str, key_err: &tikv_client::proto::kvrpcpb::KeyError) -> bool {
        let ok = !key_err.retryable.is_empty();
        info!("{} ClusterTxnClient::is_key_error_retryable", tag; "key_err" => ?key_err, "ok" => ok);
        ok
    }

    fn log_kv_error(&self, tag: &str, err: &tikv_client::Error) {
        match err {
            tikv_client::Error::PessimisticLockError {
                inner,
                success_keys,
            } => {
                error!(
                    "{} tikv_client::PessimisticLockError: inner: {:?}, success_keys: {:?}",
                    tag, inner, success_keys
                );
                self.log_kv_error(tag, inner);
            }
            tikv_client::Error::ResolveLockError(lock_info) => {
                if let Some(first) = lock_info.first() {
                    let region = self.pd_client.get_region(&first.key).unwrap();
                    error!(
                        "{} tikv_client::ResolveLockError (first), lock region: {}:{}, lock_info: {:?}",
                        tag,
                        region.get_id(),
                        region.get_region_epoch().get_version(),
                        lock_info
                    );
                }
            }
            _ => {
                error!("{} tikv_client::Error: {:?}", tag, err);
            }
        }
    }

    fn log_verify_error(&mut self, key: &[u8], expect_val: Option<&[u8]>, err: &Error) {
        let res = self.cluster_client.verify_key_value(
            key,
            expect_val,
            Instant::now(),
            &RequestOptions::default(),
        );
        assert!(
            res.is_err(),
            "verify_key_value should fail, verify error: {:?}",
            err
        );
        error!("verify_key_value error: {:?}", res.unwrap_err());
    }

    fn tag_from_key(&self, interface: &str, key: &[u8]) -> String {
        let region = self.pd_client.get_region(&encode_bytes(key)).unwrap();
        format!(
            "[{}] {}:{}",
            interface,
            region.get_id(),
            region.get_region_epoch().get_version()
        )
    }
}
