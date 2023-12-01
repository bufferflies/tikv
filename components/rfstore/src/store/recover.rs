// Copyright 2021 TiKV Project Authors. Licensed under Apache-2.0.

use std::{collections::HashMap, iter::FromIterator, sync::Arc};

use api_version::{ApiV2, KeyMode, KvFormat};
use bytes::Buf;
use collections::HashSet;
use kvengine::{Engine, Shard, ShardMeta};
use kvenginepb::ChangeSet;
use kvproto::{
    metapb,
    raft_cmdpb::{CustomRequest, RaftCmdRequest},
    raft_serverpb::{self, RegionLocalState},
};
use protobuf::Message;
use raft_proto::eraftpb;
use raftstore::store::metrics::BLACKLIST_REGION_GAUGE;
use rfengine::{
    load_store_ident, raft_state_key, region_state_key, RfEngine, WriteBatch, KV_ENGINE_META_KEY,
    TRUNCATE_ALL_INDEX,
};
use slog_global::info;
use tikv_util::warn;

use crate::store::{
    is_property_change_set, load_raft_truncated_state, load_region_state, rlog, Applier,
    ApplyContext, CustomRaftLog, PeerTag, RaftApplyState, RaftState, RegionIdVer, TERM_KEY,
};

#[derive(Clone)]
pub struct RecoverHandler {
    rf_engine: rfengine::RfEngine,
    store_id: u64,
    region_peer_map: Arc<HashMap<u64, u64>>,
    black_list: Option<BlackList>,
    contained_region_ids: Option<HashSet<u64>>,
}

pub const BLACK_LIST_FILE: &str = "black_list_file";

#[derive(Clone)]
pub struct BlackList {
    keyspace_ids: HashSet<u32>,
    region_ids: HashSet<u64>,
}

impl BlackList {
    pub fn new(mut keyspace_ids: Vec<u32>, mut region_ids: Vec<u64>) -> Self {
        BLACKLIST_REGION_GAUGE.add(region_ids.len() as i64);
        Self {
            keyspace_ids: HashSet::from_iter(keyspace_ids.drain(..)),
            region_ids: HashSet::from_iter(region_ids.drain(..)),
        }
    }

    pub(crate) fn check_blocked(&mut self, region_id: u64, start: &[u8], end: &[u8]) -> bool {
        if let Some(keyspace_id) = get_keyspace_id(start, end) {
            if self.keyspace_ids.contains(&keyspace_id) {
                BLACKLIST_REGION_GAUGE.inc();
                self.region_ids.insert(region_id);
            }
        }
        self.region_ids.contains(&region_id)
    }

    pub(crate) fn is_region_blocked(&self, region_id: u64) -> bool {
        self.region_ids.contains(&region_id)
    }

    pub(crate) fn is_keyspace_blocked(&self, keyspace_id: u32) -> bool {
        self.keyspace_ids.contains(&keyspace_id)
    }

    pub fn add_regions(&mut self, region_ids: Vec<u64>) {
        BLACKLIST_REGION_GAUGE.add(region_ids.len() as i64);
        self.region_ids.extend(region_ids.into_iter());
    }
}

pub(crate) fn get_keyspace_id(start: &[u8], end: &[u8]) -> Option<u32> {
    let start_mode = ApiV2::parse_key_mode(start);
    let end_mode = ApiV2::parse_key_mode(end);
    if start_mode != KeyMode::Txn || end_mode != KeyMode::Txn {
        return None;
    }
    Some(ApiV2::get_u32_keyspace_id(ApiV2::get_keyspace_id(start)))
}

impl RecoverHandler {
    pub fn new(rf_engine: rfengine::RfEngine) -> Self {
        let store_id = match load_store_ident(&rf_engine) {
            Some(ident) => ident.store_id,
            None => 0,
        };
        let region_peer_map = Arc::new(rf_engine.get_region_peer_map());
        Self {
            rf_engine,
            store_id,
            region_peer_map,
            black_list: None,
            contained_region_ids: None,
        }
    }

    pub fn set_contained_region_ids(&mut self, mut region_ids: Vec<u64>) {
        self.contained_region_ids = Some(HashSet::from_iter(region_ids.drain(..)))
    }

    pub fn set_black_list(&mut self, black_list: BlackList) {
        self.black_list = Some(black_list)
    }

    pub fn take_black_list(&mut self) -> Option<BlackList> {
        self.black_list.take()
    }

    fn load_region_meta(&self, shard_id: u64, shard_ver: u64) -> (metapb::Region, u64) {
        let &peer_id = self.region_peer_map.get(&shard_id).unwrap();
        let tag = PeerTag::new(self.store_id, RegionIdVer::new(shard_id, shard_ver));

        let mut region_state = self
            .rf_engine
            .load_region_state(peer_id, shard_ver)
            .unwrap_or_else(|| {
                panic!(
                    "{} failed to get region state, state key {:?}, state keys {:?}",
                    tag,
                    region_state_key(shard_ver),
                    self.get_state_keys(peer_id)
                );
            });
        let region = region_state.take_region();

        let raft_state_key = raft_state_key(shard_ver);
        let raft_state_val = self
            .rf_engine
            .get_state(peer_id, &raft_state_key)
            .unwrap_or_else(|| {
                panic!(
                    "{} failed to get raft state, state keys {:?}",
                    tag,
                    self.get_state_keys(peer_id)
                );
            });
        let mut raft_state = RaftState::default();
        raft_state.unmarshal(raft_state_val.as_ref());
        (region, raft_state.last_preprocessed_index)
    }

    fn get_state_keys(&self, peer_id: u64) -> Vec<Vec<u8>> {
        let mut state_keys = vec![];
        self.rf_engine.iterate_peer_states(peer_id, false, |k, _| {
            state_keys.push(k.to_vec());
        });
        state_keys
    }

    fn execute_admin_request(
        applier: &mut Applier,
        ctx: &mut ApplyContext,
        req: RaftCmdRequest,
    ) -> kvengine::Result<()> {
        let admin_req = req.get_admin_request();
        if admin_req.has_change_peer() {
            applier
                .exec_change_peer(ctx, admin_req)
                .map_err(|x| kvengine::Error::ErrOpen(format!("{}", x)))?;
        }
        Ok(())
    }
}

impl kvengine::RecoverHandler for RecoverHandler {
    fn recover(
        &self,
        engine: &Engine,
        shard: &Arc<Shard>,
        meta: &ShardMeta,
    ) -> kvengine::Result<()> {
        let applied_index = shard.get_write_sequence();
        let mut ctx = ApplyContext::new(engine.clone(), None);
        let applied_index_term = shard.get_property(TERM_KEY).unwrap().get_u64_le();
        let apply_state = RaftApplyState::new(applied_index, applied_index_term);
        let (region_meta, preprocessed_index) = self.load_region_meta(shard.id, shard.ver);
        let low_idx = applied_index + 1;
        let high_idx = preprocessed_index + 1;
        info!(
            "{} recover from applied {} to index {}",
            shard.tag(),
            applied_index,
            preprocessed_index,
        );
        let mut entries = Vec::with_capacity((high_idx.saturating_sub(low_idx)) as usize);
        let &peer_id = self.region_peer_map.get(&shard.id).unwrap();
        self.rf_engine
            .fetch_raft_entries_to(peer_id, low_idx, high_idx, None, &mut entries)
            .map_err(|e| {
                let stats = self.rf_engine.get_peer_stats(peer_id);
                let truncated_state = load_raft_truncated_state(&self.rf_engine, peer_id);
                let err_msg = format!(
                    "{} entries unavailable err: {:?}, stats {:?}, truncated_state: {:?}, low: {}, high {}",
                    shard.tag(),
                    e,
                    stats,
                    truncated_state,
                    low_idx,
                    high_idx
                );
                kvengine::Error::ErrOpen(err_msg)
            })?;

        let snap = shard.new_snap_access();
        let mut applier = Applier::new_for_recover(self.store_id, region_meta, snap, apply_state);

        for e in &entries {
            if e.data.is_empty() || e.entry_type != eraftpb::EntryType::EntryNormal {
                continue;
            }
            ctx.exec_log_index = e.get_index();
            ctx.exec_log_term = e.get_term();
            let req = applier.parse_cmd(e);
            if req.get_header().get_region_epoch().version != shard.ver {
                continue;
            }
            if req.has_admin_request() {
                let admin = req.get_admin_request();
                if admin.has_splits() || admin.has_prepare_merge() || admin.has_commit_merge() {
                    // We are recovering an parent shard, we need to switch the mem-table for
                    // children to copy.
                    engine.switch_mem_table(shard, meta.base_version + ctx.exec_log_index);
                    // It is the last command for a parent shard, we should return here.
                    return Ok(());
                }
                Self::execute_admin_request(&mut applier, &mut ctx, req)?;
            } else if let Some(custom) = rlog::get_custom_log(&req) {
                if let Some(mut cs) = get_async_change_set(&custom) {
                    cs.sequence = e.get_index();
                    if meta.ver == cs.get_shard_ver() && !meta.is_duplicated_change_set(&mut cs) {
                        // We don't have a background region worker now, should do it synchronously.
                        let cs = engine.prepare_change_set(cs, false, None)?;
                        engine.apply_change_set(cs)?;
                    }
                } else if let Err(e) = applier.exec_custom_log(&mut ctx, &custom) {
                    // Only duplicated pre-split may fail, we can ignore this error.
                    warn!("failed to execute custom log {:?}", e);
                }
            }
            applier.apply_state.applied_index = ctx.exec_log_index;
            applier.apply_state.applied_index_term = ctx.exec_log_term;
        }
        Ok(())
    }
}

// change set that only set property are applied synchronously by
// exec_custom_log, other change set are applied asynchronously. During recover,
// we don't have background worker, so we need to apply the async change sets
// directly. And we must exclude property change set, because it has side effect
// of switch mem-table, if we skip it, later apply flush mem-table would panic.
fn get_async_change_set(custom: &CustomRaftLog<'_>) -> Option<ChangeSet> {
    if rlog::is_engine_meta_log(custom.data.chunk()) {
        let cs = custom.get_change_set().unwrap();
        if !is_property_change_set(&cs) {
            return Some(cs);
        }
    }
    None
}

impl kvengine::MetaIterator for RecoverHandler {
    fn iterate<F>(&mut self, mut f: F) -> kvengine::Result<()>
    where
        F: FnMut(ChangeSet),
    {
        let mut wb = WriteBatch::new();
        let region_to_peers = self.rf_engine.get_region_peer_map();
        let destroy_peer = |rf: &RfEngine,
                            mut region_local_state: RegionLocalState,
                            wb: &mut WriteBatch,
                            region_id,
                            peer_id,
                            shard_ver| {
            rf.iterate_peer_states(peer_id, false, |k, _| {
                wb.set_state(peer_id, region_id, k, &[]);
            });
            region_local_state.state = raft_serverpb::PeerState::Tombstone;
            let region_state_val = region_local_state.write_to_bytes().unwrap();
            let region_state_key = region_state_key(shard_ver);
            wb.set_state(peer_id, region_id, &region_state_key, &region_state_val);
            wb.truncate_raft_log(peer_id, region_id, TRUNCATE_ALL_INDEX);
        };
        for (region_id, peer_id) in region_to_peers {
            if let Some(val) = self.rf_engine.get_state(peer_id, KV_ENGINE_META_KEY) {
                let mut cs = kvenginepb::ChangeSet::new();
                if let Err(e) = cs.merge_from_bytes(&val) {
                    return Err(kvengine::Error::ErrOpen(e.to_string()));
                }
                assert_eq!(region_id, cs.shard_id);

                let store_id = self.engine_id();
                let region_local_state = load_region_state(&self.rf_engine, peer_id, cs.shard_ver)
                    .unwrap_or_else(|| {
                        tikv_util::set_current_region(region_id);
                        panic!(
                            "{}:{}:{} failed to get region state, state key {:?}",
                            store_id,
                            region_id,
                            cs.shard_ver,
                            region_state_key(cs.shard_ver),
                        );
                    });

                // Check if the store exists in the region peers. The peer may be already
                // removed in region local state, but not destroyed yet. It's safe to destroy
                // it.
                if !region_local_state
                    .get_region()
                    .get_peers()
                    .iter()
                    .any(|p| p.get_store_id() == store_id)
                {
                    warn!(
                        "store {} not found in region peers {:?}, set it to tombstone",
                        store_id,
                        region_local_state.get_region()
                    );
                    destroy_peer(
                        &self.rf_engine,
                        region_local_state,
                        &mut wb,
                        region_id,
                        peer_id,
                        cs.shard_ver,
                    );
                    info!("destroy stale region {}", region_id);
                    continue;
                }
                if let Some(contained_region_ids) = self.contained_region_ids.as_mut() {
                    if !contained_region_ids.contains(&region_id) {
                        warn!("region {} removed by pd", region_id);
                        self.rf_engine.iterate_peer_states(peer_id, false, |k, _| {
                            wb.set_state(peer_id, region_id, k, &[]);
                        });

                        destroy_peer(
                            &self.rf_engine,
                            region_local_state,
                            &mut wb,
                            region_id,
                            peer_id,
                            cs.shard_ver,
                        );
                        info!("destroy removed region {}", region_id);
                        continue;
                    }
                }
                if let Some(black_list) = self.black_list.as_mut() {
                    let snap = cs.get_snapshot();
                    if black_list.check_blocked(
                        cs.shard_id,
                        snap.get_outer_start(),
                        snap.get_outer_end(),
                    ) {
                        warn!("region {} blocked by black list", cs.shard_id);
                        continue;
                    }
                }
                f(cs);
            }
        }
        if !wb.is_empty() {
            self.rf_engine.write(wb).unwrap();
        }
        Ok(())
    }

    fn engine_id(&self) -> u64 {
        self.store_id
    }
}

pub fn apply_custom_log_in_recover(
    engine: &Engine,
    store_id: u64,
    shard: &Arc<Shard>,
    region_meta: metapb::Region,
    custom_req: CustomRequest,
) -> crate::errors::Result<()> {
    let custom = CustomRaftLog::new_from_data(custom_req.get_data());
    let applied_index = shard.get_write_sequence();
    let mut ctx = ApplyContext::new(engine.clone(), None);
    let applied_index_term = shard.get_property(TERM_KEY).unwrap().get_u64_le();
    let apply_state = RaftApplyState::new(applied_index, applied_index_term);

    let snap = shard.new_snap_access();
    let mut applier = Applier::new_for_recover(store_id, region_meta, snap, apply_state);
    ctx.exec_log_index = applied_index + 1;
    ctx.exec_log_term = applied_index_term;
    let _ = applier.exec_custom_log(&mut ctx, &custom)?;
    Ok(())
}
