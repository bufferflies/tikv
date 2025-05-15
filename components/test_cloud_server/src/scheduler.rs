// Copyright 2022 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    collections::HashSet,
    sync::{Arc, Mutex},
    time::Duration,
};

use api_version::api_v2::KEYSPACE_PREFIX_LEN;
use dashmap::DashMap;
use futures::executor::block_on;
use kvproto::metapb::{Peer, PeerRole, Region};
use pd_client::PdClient;
use rand::Rng;
use test_pd_client::{PdClientExt, TestPdClient};
use tikv_util::{info, time::Instant, warn};

use crate::{must_wait_with_premise, try_wait};

pub struct Scheduler {
    pub(crate) pd: Arc<TestPdClient>,
    pub(crate) store_ids: Vec<u64>,
    pub(crate) lock: Arc<DashMap<u64, Arc<Mutex<()>>>>,
}

impl Scheduler {
    pub fn move_random_region(&self) -> bool {
        let regions = self.pd.get_all_regions();
        let region_idx = rand::thread_rng().gen_range(0..regions.len());
        let region = &regions[region_idx];
        if region.get_peers().len() != 3 {
            return false;
        }
        let &target_store_id = self
            .store_ids
            .iter()
            .find(|&&store_id| {
                let contains = region
                    .get_peers()
                    .iter()
                    .any(|peer| peer.store_id == store_id);
                !contains
            })
            .unwrap();
        let mutex = self.get_region_mutex(region.id);
        let _guard = mutex.lock().unwrap();
        if self.is_region_changed(region) {
            return false;
        }
        self.move_peer(region.id, target_store_id, Duration::from_secs(30))
            .is_some()
    }

    fn is_region_changed(&self, region: &Region) -> bool {
        let new_region = block_on(self.pd.get_region_by_id(region.id)).unwrap();
        new_region
            .map(|n| n.get_region_epoch().ne(region.get_region_epoch()))
            .unwrap_or(true)
    }

    async fn check_region_stats_exists(
        &self,
        store_id: u64,
        region_id: u64,
        region_ver: u64,
    ) -> bool {
        let store = self.pd.get_store(store_id).unwrap();
        let uri = format!("http://{}/kvengine/{}", store.status_address, region_id);
        let resp = reqwest::blocking::get(&uri);
        if resp.is_err() {
            warn!("check_region_stats_exists failed, err {:?}", resp.err());
            return false;
        }
        let resp = resp.unwrap();
        if resp.status() != reqwest::StatusCode::OK {
            warn!(
                "check_region_stats_exists failed, status {:?}",
                resp.status()
            );
            return false;
        }
        let stats: kvengine::ShardStats = resp.json().unwrap();
        stats.ver == region_ver
    }

    fn move_peer(&self, region_id: u64, store_id: u64, timeout: Duration) -> Option<()> {
        let timeout_secs = timeout.as_secs() as usize;
        let peer_id = self.pd.alloc_id().unwrap();

        // Add learner.
        let mut peer = Peer::new();
        peer.store_id = store_id;
        peer.id = peer_id;
        peer.role = PeerRole::Learner;
        let add_learner = || self.pd.add_peer(region_id, peer.clone());
        must_wait_with_premise(
            || block_on(self.pd.get_region_by_id(region_id)).unwrap(),
            |region, retry_idx| {
                if retry_idx % 10 == 0 {
                    add_learner();
                }
                // Wait until the learner complete restore snapshots, or the raft group will
                // lose majority if leader down at this point.
                let region_ver = region.get_region_epoch().get_version();
                block_on(self.check_region_stats_exists(store_id, region_id, region_ver))
            },
            timeout_secs * 2, /* Need more time to wait for store restart (required by
                               * `check_region_stats_exists`). */
            || {
                format!(
                    "failed to add learner, region id {}, store id {}, peer id {}, region {:?}",
                    region_id,
                    store_id,
                    peer_id,
                    block_on(self.pd.get_region_by_id(region_id)).unwrap()
                )
            },
        )?;

        // Add voter.
        let mut peer = Peer::new();
        peer.store_id = store_id;
        peer.id = peer_id;
        peer.role = PeerRole::Voter;
        let add_voter = || self.pd.add_peer(region_id, peer.clone());
        must_wait_with_premise(
            || block_on(self.pd.get_region_by_id(region_id)).unwrap(),
            |region, retry_idx| {
                if retry_idx % 10 == 0 {
                    add_voter();
                }
                region
                    .get_peers()
                    .iter()
                    .any(|peer| peer.id == peer_id && peer.role == PeerRole::Voter)
            },
            timeout_secs,
            || {
                format!(
                    "failed to promote learner, region id {}, store id {}, peer id {}, region {:?}",
                    region_id,
                    store_id,
                    peer_id,
                    block_on(self.pd.get_region_by_id(region_id)).unwrap()
                )
            },
        )?;

        // Select target.
        let mut old_leader = Peer::default();
        let mut to_remove = Peer::default();
        must_wait_with_premise(
            || block_on(self.pd.get_region_leader_by_id(region_id)).unwrap(),
            |(region, leader), _| {
                old_leader = leader.clone();
                let target = region
                    .peers
                    .iter()
                    .find(|peer| peer.id != leader.id)
                    .unwrap();
                to_remove = target.clone();
                to_remove.id != old_leader.id
            },
            timeout_secs,
            || format!("failed to get target peer, region id {}", region_id),
        )?;

        // Remove peer.
        let transfer_leader = || {
            self.pd.try_transfer_leader(region_id, old_leader.clone());
            info!(
                "move_peer: transfer leader, region id {} old leader id {} to_remove id {}",
                region_id, old_leader.id, to_remove.id
            );
        };
        let remove_peer = || self.pd.try_remove_peer(region_id, to_remove.clone());
        must_wait_with_premise(
            || block_on(self.pd.get_region_leader_by_id(region_id)).unwrap(),
            |(region, leader), retry_idx| {
                let target_is_leader = leader.id == to_remove.id;
                if target_is_leader {
                    if retry_idx % 10 == 0 {
                        transfer_leader();
                    }
                    return false;
                }

                if retry_idx % 30 == 0 {
                    remove_peer();
                }
                region.get_peers().len() == 3
            },
            timeout_secs,
            || {
                format!(
                    "failed to remove peer id {} region id {} leader id {}",
                    to_remove.id, region_id, old_leader.id
                )
            },
        )
    }

    pub fn transfer_random_leader(&self) -> bool {
        let regions = self.pd.get_all_regions();
        if regions.is_empty() {
            return false;
        }
        let region_idx = rand::thread_rng().gen_range(0..regions.len());
        let region = &regions[region_idx];
        if region.get_peers().len() != 3 {
            return false;
        }
        let region_id = region.get_id();
        let mutex = self.get_region_mutex(region_id);
        let _guard = mutex.lock().unwrap();
        block_on(self.pd.get_region_leader_by_id(region_id))
            .unwrap()
            .map(|(_, leader)| {
                let old_leader_id = leader.id;
                region
                    .peers
                    .iter()
                    .find(|peer| peer.id != old_leader_id && peer.role == PeerRole::Voter)
                    .map(|target| {
                        self.pd.try_transfer_leader(region_id, target.clone());
                        let new_leader_id = target.id;
                        try_wait(
                            || {
                                block_on(self.pd.get_region_leader_by_id(region_id))
                                    .unwrap()
                                    .map(|(_, leader)| leader.id == new_leader_id)
                                    .unwrap_or(false)
                            },
                            3,
                        )
                    })
                    .unwrap_or(false)
            })
            .unwrap_or(false)
    }

    pub fn get_region_mutex(&self, region_id: u64) -> Arc<Mutex<()>> {
        self.lock
            .entry(region_id)
            .or_insert(Arc::new(Mutex::default()))
            .clone()
    }

    pub fn merge_random_region(&self, disallow_cross_keyspace: bool) -> bool {
        let mut rng = rand::thread_rng();
        let mut regions = self.pd.get_all_regions();
        regions.sort_by(|a, b| a.start_key.cmp(&b.start_key));

        let in_same_keyspace = |left: &Region, right: &Region| -> bool {
            left.start_key.len() >= KEYSPACE_PREFIX_LEN
                && right.start_key.len() >= KEYSPACE_PREFIX_LEN
                && left.start_key[..KEYSPACE_PREFIX_LEN] == right.start_key[..KEYSPACE_PREFIX_LEN]
        };

        let mut get_adjacent_regions = || -> Option<(&Region, &Region)> {
            for _ in 0..3 {
                let merge_idx = rng.gen_range(1..regions.len());
                let left = &regions[merge_idx - 1];
                let right = &regions[merge_idx];

                if !left.end_key.eq(&right.start_key) {
                    continue;
                }
                if disallow_cross_keyspace && !in_same_keyspace(left, right) {
                    continue;
                }

                return Some((left, right));
            }
            None
        };

        let (left, right) = match get_adjacent_regions() {
            Some((left, right)) => (left, right),
            None => return false,
        };

        // To avoid deadlock, we need lock regions in same order. Lock the left region
        // first.
        let left_mutex = self.get_region_mutex(left.id);
        let _left_guard = left_mutex.lock().unwrap();
        if self.is_region_changed(left) {
            return false;
        }

        // Also need to lock the right region to avoid schedule conflicts.
        let right_mutex = self.get_region_mutex(right.id);
        let _right_guard = right_mutex.lock().unwrap();
        if self.is_region_changed(right) {
            return false;
        }

        let (source, target) = if rng.gen_ratio(1, 2) {
            (left, right)
        } else {
            (right, left)
        };

        let source_stores: HashSet<u64> = source.get_peers().iter().map(|p| p.store_id).collect();
        let target_stores: Vec<u64> = target.get_peers().iter().map(|p| p.store_id).collect();
        for target_store_id in &target_stores {
            if !source_stores.contains(target_store_id) {
                self.move_peer(source.id, *target_store_id, Duration::from_secs(30));
                break;
            }
        }
        self.pd.try_merge_region(source.id, target.id);

        let start_time = Instant::now_coarse();
        while start_time.saturating_elapsed() < Duration::from_secs(5) {
            let region = match block_on(self.pd.get_region_by_id(target.id)).unwrap() {
                Some(region) => region,
                None => {
                    warn!("Scheduler::merge_random_region: target region not exists"; "target" => ?target, "source" => ?source);
                    return false;
                }
            };
            if region.start_key.eq(&left.start_key) && region.end_key.eq(&right.end_key) {
                return true;
            }
            std::thread::sleep(Duration::from_millis(100));
        }
        false
    }
}
