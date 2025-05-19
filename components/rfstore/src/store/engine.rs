// Copyright 2021 TiKV Project Authors. Licensed under Apache-2.0.

use std::sync::{Arc, Mutex};

use kvenginepb::ChangeSet;
use tikv_util::{
    info, mpsc,
    mpsc::{Receiver, Sender},
};

use crate::store::{BlackList, Callback, StoreMsg};

#[derive(Clone)]
pub struct Engines {
    pub kv: kvengine::Engine,
    pub raft: rfengine::RfEngine,
    #[allow(clippy::type_complexity)]
    pub meta_change_channel: Arc<Mutex<Option<(Sender<StoreMsg>, Receiver<StoreMsg>)>>>,
    pub black_list: Option<BlackList>,
}

impl Engines {
    pub fn new(
        kv: kvengine::Engine,
        raft: rfengine::RfEngine,
        meta_change_channel: (Sender<StoreMsg>, Receiver<StoreMsg>),
        black_list: Option<BlackList>,
    ) -> Self {
        Self {
            kv,
            raft,
            meta_change_channel: Arc::new(Mutex::new(Some(meta_change_channel))),
            black_list,
        }
    }
}

impl From<Engines> for engine_traits::Engines<kvengine::Engine, rfengine::RfEngine> {
    fn from(engines: Engines) -> Self {
        Self {
            kv: engines.kv.clone(),
            raft: engines.raft,
        }
    }
}

#[derive(Clone)]
pub struct MetaChangeListener {
    pub sender: mpsc::Sender<StoreMsg>,
}

impl kvengine::MetaChangeListener for MetaChangeListener {
    fn on_change_set(&self, cs: ChangeSet) {
        let msg = StoreMsg::GenerateEngineChangeSet(cs, Callback::None);
        if let Err(e) = self.sender.send(msg) {
            info!(
                "failed to to send meta change message, are we shutting down?";
                "err" => ?e,
            )
        }
    }
}
