// Copyright 2021 TiKV Project Authors. Licensed under Apache-2.0.

use std::{collections::HashMap, sync::Arc};

use bytes::Buf;
use kvengine::{Engine, Shard, ShardMeta};
use kvenginepb::ChangeSet;
use kvproto::{metapb, raft_cmdpb::RaftCmdRequest, raft_serverpb};
use protobuf::Message;
use raft_proto::eraftpb;
use slog_global::info;
use tikv_util::warn;

use crate::store::{
    is_property_change_set, load_raft_truncated_state, raft_state_key, region_state_key, rlog,
    Applier, ApplyContext, CustomRaftLog, PeerTag, RaftApplyState, RaftState, RegionIDVer,
    KV_ENGINE_META_KEY, STORE_IDENT_KEY, TERM_KEY,
};

#[derive(Clone)]
pub struct RecoverHandler {
    rf_engine: rfengine::RfEngine,
    store_id: u64,
    region_peer_map: HashMap<u64, u64>,
}

impl RecoverHandler {
    pub fn new(rf_engine: rfengine::RfEngine) -> Self {
        let store_id = match load_store_ident(&rf_engine) {
            Some(ident) => ident.store_id,
            None => 0,
        };
        let region_peer_map = rf_engine.get_region_peer_map();
        Self {
            rf_engine,
            store_id,
            region_peer_map,
        }
    }

    fn load_region_meta(&self, shard_id: u64, shard_ver: u64) -> (metapb::Region, u64) {
        let &peer_id = self.region_peer_map.get(&shard_id).unwrap();
        let tag = PeerTag::new(self.store_id, RegionIDVer::new(shard_id, shard_ver));
        let region_state_key = region_state_key(shard_ver);
        let region_state_val = self
            .rf_engine
            .get_state(peer_id, &region_state_key)
            .unwrap_or_else(|| {
                panic!(
                    "{} failed to get region state, state keys {:?}",
                    tag,
                    self.get_state_keys(peer_id)
                );
            });
        let mut region_state = raft_serverpb::RegionLocalState::new();
        region_state.merge_from_bytes(&region_state_val).unwrap();
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

fn load_store_ident(rf_engine: &rfengine::RfEngine) -> Option<raft_serverpb::StoreIdent> {
    let val = rf_engine.get_state(0, STORE_IDENT_KEY);
    val.as_ref()?;
    let mut ident = raft_serverpb::StoreIdent::new();
    ident.merge_from_bytes(val.unwrap().chunk()).unwrap();
    Some(ident)
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
            let mut req = RaftCmdRequest::new();
            req.merge_from_bytes(e.data.chunk()).unwrap();
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
                        let cs = engine.prepare_change_set(cs, false)?;
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

// change set that only set property are applied synchronously by exec_custom_log, other change set
// are applied asynchronously. During recover, we don't have background worker, so we need to
// apply the async change sets directly.
// And we must exclude property change set, because it has side effect of switch mem-table, if we
// skip it, later apply flush mem-table would panic.
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
    fn iterate<F>(&self, mut f: F) -> kvengine::Result<()>
    where
        F: FnMut(ChangeSet),
    {
        let region_to_peers = self.rf_engine.get_region_peer_map();
        for (_, peer_id) in region_to_peers {
            if let Some(val) = self.rf_engine.get_state(peer_id, KV_ENGINE_META_KEY) {
                let mut cs = kvenginepb::ChangeSet::new();
                if let Err(e) = cs.merge_from_bytes(&val) {
                    return Err(kvengine::Error::ErrOpen(e.to_string()));
                }
                f(cs);
            }
        }
        Ok(())
    }

    fn engine_id(&self) -> u64 {
        self.store_id
    }
}
