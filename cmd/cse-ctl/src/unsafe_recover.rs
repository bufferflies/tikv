// Copyright 2022 TiKV Project Authors. Licensed under Apache-2.0.

use std::{collections::HashSet, path::PathBuf, str::FromStr};

use bytes::{BufMut, Bytes, BytesMut};
use clap::Args;
use kvproto::raft_serverpb::{PeerState, RegionLocalState};
use protobuf::Message;
use rfengine::{RfEngine, WriteBatch};
use tikv_util::{codec::number::NumberEncoder, info};

#[derive(Args)]
pub struct UnsafeRecoverArgs {
    /// The path of the raft engine.
    #[clap(long)]
    pub path: PathBuf,

    /// Filter the region id to operate.
    #[clap(long)]
    pub region: Option<u64>,

    /// Filter the keyspace id regions to operate.
    #[clap(long)]
    pub keyspace: Option<u32>,

    /// Filter the table id regions to operate.
    #[clap(long)]
    pub table: Option<u64>,

    /// Destroy the target regions on the store.
    #[clap(long)]
    pub destroy: bool,

    /// Update the regions by removing the peers on the stores.
    /// Multiple stores are separated by ",".
    #[clap(long)]
    pub remove_stores: Option<String>,

    /// Commit the change.
    #[clap(long)]
    pub commit: bool,

    /// Operate on all regions.
    #[clap(long)]
    pub all: bool,
}

const REGION_META_KEY_BYTE: u8 = 2;
const KV_ENGINE_META_KEY: &[u8] = &[5];

pub(crate) fn execute_unsafe_recover(args: UnsafeRecoverArgs) {
    let rf = rfengine::RfEngine::open(&args.path, 512 * 1024 * 1024).unwrap();
    let target_regions = if let Some(region_id) = args.region {
        let region_to_peers = rf.get_region_peer_map();
        let &peer_id = region_to_peers.get(&region_id).unwrap();
        let v = rf
            .get_state(peer_id, KV_ENGINE_META_KEY)
            .expect("region not found");
        let mut cs = kvenginepb::ChangeSet::new();
        cs.merge_from_bytes(&v).unwrap();
        vec![(peer_id, region_id, cs.shard_ver)]
    } else if let Some(keyspace_id) = args.keyspace {
        let mut keyspace = keyspace_id.to_be_bytes();
        keyspace[0] = 'x' as u8;
        let mut prefix = keyspace.to_vec();
        if let Some(table_id) = args.table {
            prefix.put_u8('t' as u8);
            prefix.encode_i64(table_id as i64).unwrap();
        }
        collect_prefix_regions(&rf, &prefix)
    } else if let Some(table_id) = args.table {
        let mut prefix = vec!['t' as u8];
        prefix.encode_i64(table_id as i64).unwrap();
        collect_prefix_regions(&rf, &prefix)
    } else if args.all {
        collect_prefix_regions(&rf, &[])
    } else {
        panic!("no filter specified");
    };
    let mut wb = WriteBatch::new();
    if let Some(failed_stores_str) = args.remove_stores {
        let failed_stores: HashSet<u64> = parse_stores(&failed_stores_str);
        for (peer_id, region_id, region_version) in target_regions {
            let region_state_key = region_state_key(region_version);
            let mut region_local_state = load_region_state(&rf, peer_id, &region_state_key);
            let region = region_local_state.mut_region();
            let old_region = region.clone();
            region.mut_region_epoch().conf_ver += 1;
            let mut new_peers = region.get_peers().to_vec();
            new_peers.retain(|peer| !failed_stores.contains(&peer.store_id));
            region.set_peers(new_peers.into());
            let region_state_val = region_local_state.write_to_bytes().unwrap();
            wb.set_state(peer_id, region_id, &region_state_key, &region_state_val);
            info!(
                "update region from {:?} to {:?}",
                old_region,
                region_local_state.get_region(),
            );
        }
    } else if args.destroy {
        for (peer_id, region_id, region_version) in target_regions {
            rf.iterate_peer_states(peer_id, false, |k, _| {
                wb.set_state(peer_id, region_id, k, &[]);
            });
            let region_state_key = region_state_key(region_version);
            let mut region_local_state = load_region_state(&rf, peer_id, &region_state_key);
            region_local_state.state = PeerState::Tombstone;
            let region_state_val = region_local_state.write_to_bytes().unwrap();
            wb.set_state(peer_id, region_id, &region_state_key, &region_state_val);
            wb.truncate_raft_log(peer_id, region_id, u64::MAX);
            info!("destroy region {:?}", region_local_state);
        }
    } else {
        for (peer_id, _, region_version) in target_regions {
            let region_state_key = region_state_key(region_version);
            let region_local_state = load_region_state(&rf, peer_id, &region_state_key);
            info!("region: {:?}", region_local_state.get_region());
        }
    }
    if args.commit {
        info!("commit changes");
        rf.write(wb).unwrap();
        info!("done")
    }
}

fn collect_prefix_regions(rf: &RfEngine, prefix: &[u8]) -> Vec<(u64, u64, u64)> {
    let region_peers = rf.get_region_peer_map();
    let mut prefix_peers = vec![];
    for (region_id, peer_id) in region_peers {
        if region_id == 0 {
            continue;
        }
        let engine_meta = load_engine_meta(rf, peer_id);
        let snap = engine_meta.get_snapshot();
        let start = snap.get_start();
        if start.starts_with(prefix) {
            prefix_peers.push((peer_id, region_id, engine_meta.shard_ver));
        }
    }
    prefix_peers
}

fn parse_stores(stores_str: &str) -> HashSet<u64> {
    stores_str
        .split(",")
        .map(|x| u64::from_str(x).unwrap())
        .collect()
}

fn load_region_state(rf: &RfEngine, peer_id: u64, key: &[u8]) -> RegionLocalState {
    let region_state_val = rf.get_state(peer_id, key).expect("region state not found");
    let mut region_local_state = RegionLocalState::new();
    region_local_state
        .merge_from_bytes(&region_state_val)
        .unwrap();
    region_local_state
}

fn load_engine_meta(rf: &RfEngine, peer_id: u64) -> kvenginepb::ChangeSet {
    let engine_meta_val = rf
        .get_state(peer_id, KV_ENGINE_META_KEY)
        .expect("engine meta not found");
    let mut cs = kvenginepb::ChangeSet::new();
    cs.merge_from_bytes(&engine_meta_val).unwrap();
    cs
}

fn region_state_key(version: u64) -> Bytes {
    let mut key = BytesMut::with_capacity(5);
    key.put_u8(REGION_META_KEY_BYTE);
    key.put_u32(version as u32);
    key.freeze()
}
