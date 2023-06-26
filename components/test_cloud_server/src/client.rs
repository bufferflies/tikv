// Copyright 2022 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    collections::{BTreeMap, HashMap},
    ops::{
        Bound::{Excluded, Included, Unbounded},
        Deref, DerefMut, Range,
    },
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Mutex,
    },
    thread::sleep,
    time::Duration,
};

use futures::executor::block_on;
use grpcio::Channel;
use kvproto::{
    coprocessor as coppb,
    errorpb::Error,
    kvrpcpb::{
        ApiVersion, CommitRequest, Context, GetRequest, IsolationLevel, Mutation, Op,
        PrewriteRequest, SplitRegionRequest,
    },
    metapb,
    metapb::{Peer, Region, RegionEpoch},
    tikvpb::TikvClient,
};
use pd_client::PdClient;
use rfstore::store::RegionIdVer;
use test_pd_client::TestPdClient;
use tikv::storage::mvcc::TimeStamp;
use tikv_util::{
    box_err,
    codec::bytes::{decode_bytes, encode_bytes},
    time::Instant,
    warn,
};

// use fail::fail_point;
use crate::{must_wait, try_wait};

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Sync + Send>>;

#[derive(Default, Clone)]
pub struct RefStore(HashMap<Vec<u8>, Option<Vec<u8>>>); // `None` means the key has been deleted.

impl RefStore {
    pub fn put_kv(&mut self, key: Vec<u8>, value: Vec<u8>) {
        self.0.insert(key, Some(value));
    }

    pub fn del_kv(&mut self, key: Vec<u8>) {
        self.0.insert(key, None);
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
    pub pd_client: Arc<TestPdClient>,
    pub channels: HashMap<u64, Channel>,
    /// region_raw_end_key -> region_id
    pub(crate) region_ranges: BTreeMap<Vec<u8>, RegionIdVer>,
    /// region_id -> region
    pub(crate) regions: HashMap<u64, RawRegion>,
    pub(crate) ref_store: Arc<Mutex<RefStore>>,
    pub(crate) max_ts: AtomicU64,
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

    pub fn raw_end(&self) -> &[u8] {
        &self.raw_end
    }

    pub fn peers(&self) -> &[Peer] {
        &self.peers
    }
}

impl ClusterClient {
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
        let start_key = gen_key(rng.start);
        let start_ts = self.get_ts();

        let mut mutations = vec![];
        for i in rng {
            let mut m = Mutation::default();
            m.set_op(Op::Del);
            m.set_key(gen_key(i));
            mutations.push(m)
        }
        let keys = mutations.iter().map(|m| m.get_key().to_vec()).collect();
        self.kv_prewrite(mutations.clone(), start_key, start_ts);
        let commit_ts = self.get_ts();
        self.kv_commit(keys, start_ts, commit_ts);
        self.del_kv_in_ref_store(mutations);
        self.set_max_ts(commit_ts.into_inner());
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
        self.try_put_kv(rng, gen_key, gen_val).unwrap();
    }

    pub fn try_put_kv<F, G>(&mut self, rng: Range<usize>, gen_key: F, gen_val: G) -> Result<()>
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
        let keys = mutations.iter().map(|m| m.get_key().to_vec()).collect();
        let put_time = Instant::now();
        self.kv_prewrite(mutations.clone(), start_key, start_ts);
        let commit_ts = self.get_ts();
        self.kv_commit(keys, start_ts, commit_ts);
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
        self.kv_prewrite(muts, keys[0].clone(), start_ts);

        let commit_ts = self.get_ts();
        self.kv_commit(keys, start_ts, commit_ts);
        self.set_max_ts(commit_ts.into_inner());
        Ok(())
    }

    pub fn kv_prewrite(&mut self, muts: Vec<Mutation>, pk: Vec<u8>, ts: TimeStamp) {
        let groups = self.group_mutations_by_region(muts);
        for (id_ver, group_muts) in groups {
            self.kv_prewrite_single_region(id_ver, group_muts, pk.clone(), ts);
        }
    }

    pub fn kv_prewrite_single_region(
        &mut self,
        id_ver: RegionIdVer,
        muts: Vec<Mutation>,
        pk: Vec<u8>,
        ts: TimeStamp,
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
                self.kv_prewrite(muts, pk, ts);
                return;
            }
            let ctx = ctx.unwrap();
            let store_id = ctx.get_peer().get_store_id();
            let kv_client = self.get_kv_client(store_id);
            let mut prewrite_req = PrewriteRequest::default();
            prewrite_req.set_context(ctx);
            prewrite_req.set_mutations(muts.clone().into());
            prewrite_req.primary_lock = pk.clone();
            prewrite_req.start_version = ts.into_inner();
            prewrite_req.lock_ttl = 3000;
            prewrite_req.min_commit_ts = prewrite_req.start_version + 1;
            let result = kv_client.kv_prewrite(&prewrite_req);
            if result.is_err() {
                store_id_errors.push((store_id, format!("{:?}", result.unwrap_err())));
                sleep(Duration::from_millis(100));
                self.update_cache_by_id(region_id, None);
                continue;
            }
            let resp = result.unwrap();
            if resp.has_region_error() {
                let region_err = resp.get_region_error();
                if self.handle_retryable_error(region_id, region_err) {
                    store_id_errors.push((store_id, format!("{:?}", region_err)));
                    continue;
                }
                if self.handle_region_epoch_not_match_or_not_found(region_err) {
                    self.kv_prewrite(muts, pk, ts);
                    return;
                }
                panic!("unexpected error {:?}", region_err);
            }
            let key_errors = resp.get_errors();
            if !key_errors.is_empty() {
                // TODO: resolve locks
                panic!(
                    "{} prewrite failed with key errors {:?}",
                    region_id, key_errors
                );
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

    fn handle_retryable_error(&mut self, region_id: u64, region_err: &Error) -> bool {
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
        false
    }

    fn handle_region_epoch_not_match_or_not_found(&mut self, region_err: &Error) -> bool {
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
        ctx.set_api_version(ApiVersion::V2);
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
            let resp = client.split_region(&split_req).unwrap();
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
                return Err(format!("failed to split key {:?} error {:?}", key, region_err).into());
            }
            return Ok(());
        }
        Err(format!("failed to split key {:?}", key).into())
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
                return Err(format!(
                    "adjacent region out of boundary prefix, raw_end_key: {:?}",
                    raw_end_key,
                )
                .into());
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
                return Err(format!("region {:?} is still not merged.", source_region).into());
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
        let mut tag = kvengine::ShardTag::default();
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
                return Err(box_err!("{} unexpected error {:?}", tag, region_err));
            }
            if resp.has_error() {
                return Err(box_err!(
                    "{} key {} get key_error: {:?}",
                    tag,
                    log_wrappers::hex_encode_upper(key),
                    resp.get_error()
                ));
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
        let put_time = Instant::now();
        for (k, v) in ref_store.iter() {
            if let Some(range) = range {
                if k.as_slice() < range.0 || k.as_slice() >= range.1 {
                    continue;
                }
            }
            self.verify_key_value(k, v.as_ref(), put_time, options)?;
            cnt += 1;
        }
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
            return Err(format!(
                "{} val not equal for key {}, db: {:?}, ref store {:?}",
                Self::tag_from_ctx(&ctx),
                log_wrappers::hex_encode_upper(key),
                val.map(|v| (v.len(), log_wrappers::hex_encode_upper(v))),
                expect_val.map(|v| (v.len(), log_wrappers::hex_encode_upper(v))),
            )
            .into());
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
}
