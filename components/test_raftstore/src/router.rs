// Copyright 2018 TiKV Project Authors. Licensed under Apache-2.0.

use std::sync::{Arc, Mutex};

use collections::HashMap;
use engine_rocks::{RocksEngine, RocksSnapshot};
use kvproto::raft_serverpb::RaftMessage;
use raftstore::{
    errors::{Error as RaftStoreError, Result as RaftStoreResult},
    router::RaftStoreRouter,
    store::{
        msg::{PeerMsg, SignificantMsg},
        SignificantRouter,
    },
};
use tikv_util::mpsc::{loose_bounded, LooseBoundedSender, Receiver};

#[derive(Clone)]
#[allow(clippy::type_complexity)]
pub struct MockRaftStoreRouter {
    senders: Arc<Mutex<HashMap<u64, LooseBoundedSender<PeerMsg<RocksEngine>>>>>,
}

impl MockRaftStoreRouter {
    pub fn new() -> MockRaftStoreRouter {
        MockRaftStoreRouter {
            senders: Arc::default(),
        }
    }
    pub fn add_region(&self, region_id: u64, cap: usize) -> Receiver<PeerMsg<RocksEngine>> {
        let (tx, rx) = loose_bounded(cap);
        self.senders.lock().unwrap().insert(region_id, tx);
        rx
    }
}

impl Default for MockRaftStoreRouter {
    fn default() -> Self {
        Self::new()
    }
}

impl SignificantRouter<RocksEngine> for MockRaftStoreRouter {
    fn significant_send(
        &self,
        region_id: u64,
        msg: SignificantMsg<RocksSnapshot>,
    ) -> RaftStoreResult<()> {
        let mut senders = self.senders.lock().unwrap();
        if let Some(tx) = senders.get_mut(&region_id) {
            tx.force_send(PeerMsg::SignificantMsg(msg)).unwrap();
            Ok(())
        } else {
            error!("failed to send significant msg"; "msg" => ?msg);
            Err(RaftStoreError::RegionNotFound(region_id))
        }
    }
}

impl RaftStoreRouter<RocksEngine> for MockRaftStoreRouter {
    fn send_raft_msg(&self, _: RaftMessage) -> RaftStoreResult<()> {
        unimplemented!()
    }

    fn broadcast_normal(&self, _: impl FnMut() -> PeerMsg<RocksEngine>) {}
}
