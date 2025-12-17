// Copyright 2022 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    marker::PhantomData,
    ops::{Deref, DerefMut},
};

use kvproto::raft_serverpb::RaftMessage;
use raftstore::router::RaftStoreRouter;

#[derive(Clone)]
pub struct RaftRouterWrap<S, E> {
    router: S,
    _phantom: PhantomData<E>,
}

impl<S, E> RaftRouterWrap<S, E> {
    pub fn new(router: S) -> Self {
        Self {
            router,
            _phantom: PhantomData,
        }
    }
}

impl<S, E> Deref for RaftRouterWrap<S, E> {
    type Target = S;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.router
    }
}

impl<S, E> DerefMut for RaftRouterWrap<S, E> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.router
    }
}

impl<S, E> tikv_kv::RaftExtension for RaftRouterWrap<S, E>
where
    S: RaftStoreRouter<E> + 'static,
    E: engine_traits::KvEngine,
{
    #[inline]
    fn feed(&self, msg: RaftMessage, key_message: bool) {
        let region_id = msg.get_region_id();
        let msg_ty = msg.get_message().get_msg_type();
        // Channel full and region not found are ignored unless it's a key message.
        if let Err(e) = self.router.send_raft_msg(msg)
            && key_message
        {
            error!("failed to send raft message"; "region_id" => region_id, "msg_ty" => ?msg_ty, "err" => ?e);
        }
    }

    #[inline]
    fn report_peer_unreachable(&self, region_id: u64, to_peer_id: u64) {
        let _ = self.router.report_unreachable(region_id, to_peer_id);
    }

    #[inline]
    fn report_resolved(&self, store_id: u64, group_id: u64) {
        self.router.report_resolved(store_id, group_id);
    }
}
