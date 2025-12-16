// Copyright 2018 TiKV Project Authors. Licensed under Apache-2.0.

// #[PerformanceCriticalPath]
use std::{borrow::Cow, u64};

use batch_system::{BasicMailbox, Fsm};
use engine_traits::{KvEngine, RaftEngine};
use kvproto::metapb::{self};

use crate::store::{Peer, PeerMsg};

pub const MAX_PROPOSAL_SIZE_RATIO: f64 = 0.4;

pub struct DestroyPeerJob {
    pub initialized: bool,
    pub region_id: u64,
    pub peer: metapb::Peer,
}

pub struct PeerFsm<EK, ER>
where
    EK: KvEngine,
    ER: RaftEngine,
{
    pub peer: Peer<EK, ER>,
    stopped: bool,
    mailbox: Option<BasicMailbox<PeerFsm<EK, ER>>>,
}

impl<EK, ER> Fsm for PeerFsm<EK, ER>
where
    EK: KvEngine,
    ER: RaftEngine,
{
    type Message = PeerMsg<EK>;

    #[inline]
    fn is_stopped(&self) -> bool {
        self.stopped
    }

    /// Set a mailbox to FSM, which should be used to send message to itself.
    #[inline]
    fn set_mailbox(&mut self, mailbox: Cow<'_, BasicMailbox<Self>>)
    where
        Self: Sized,
    {
        self.mailbox = Some(mailbox.into_owned());
    }

    /// Take the mailbox from FSM. Implementation should ensure there will be
    /// no reference to mailbox after calling this method.
    #[inline]
    fn take_mailbox(&mut self) -> Option<BasicMailbox<Self>>
    where
        Self: Sized,
    {
        self.mailbox.take()
    }
}
