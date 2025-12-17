// Copyright 2016 TiKV Project Authors. Licensed under Apache-2.0.

// #[PerformanceCriticalPath]
use std::sync::mpsc;

use crossbeam::channel::TrySendError;
use engine_traits::{KvEngine, Snapshot};

use crate::{
    store::{CasualMessage, RaftCommand, SignificantMsg, StoreMsg},
    DiscardReason, Error, Result,
};

/// Routes message to target region.
///
/// Messages are not guaranteed to be delivered by this trait.
pub trait CasualRouter<EK>: Send
where
    EK: KvEngine,
{
    fn send(&self, region_id: u64, msg: CasualMessage<EK>) -> Result<()>;
}

/// Routes message to target region.
///
/// Messages aret guaranteed to be delivered by this trait.
pub trait SignificantRouter<EK>: Send
where
    EK: KvEngine,
{
    fn significant_send(&self, region_id: u64, msg: SignificantMsg<EK::Snapshot>) -> Result<()>;
}

/// Routes proposal to target region.
pub trait ProposalRouter<S>
where
    S: Snapshot,
{
    fn send(&self, cmd: RaftCommand<S>) -> std::result::Result<(), TrySendError<RaftCommand<S>>>;
}

/// Routes message to store FSM.
///
/// Messages are not guaranteed to be delivered by this trait.
pub trait StoreRouter<EK>: Send
where
    EK: KvEngine,
{
    fn send(&self, msg: StoreMsg<EK>) -> Result<()>;
}

impl<EK> CasualRouter<EK> for mpsc::SyncSender<(u64, CasualMessage<EK>)>
where
    EK: KvEngine,
{
    fn send(&self, region_id: u64, msg: CasualMessage<EK>) -> Result<()> {
        match self.try_send((region_id, msg)) {
            Ok(()) => Ok(()),
            Err(mpsc::TrySendError::Disconnected(_)) => {
                Err(Error::Transport(DiscardReason::Disconnected))
            }
            Err(mpsc::TrySendError::Full(_)) => Err(Error::Transport(DiscardReason::Full)),
        }
    }
}

impl<S: Snapshot> ProposalRouter<S> for mpsc::SyncSender<RaftCommand<S>> {
    fn send(&self, cmd: RaftCommand<S>) -> std::result::Result<(), TrySendError<RaftCommand<S>>> {
        match self.try_send(cmd) {
            Ok(()) => Ok(()),
            Err(mpsc::TrySendError::Disconnected(cmd)) => Err(TrySendError::Disconnected(cmd)),
            Err(mpsc::TrySendError::Full(cmd)) => Err(TrySendError::Full(cmd)),
        }
    }
}

impl<EK> StoreRouter<EK> for mpsc::Sender<StoreMsg<EK>>
where
    EK: KvEngine,
{
    fn send(&self, msg: StoreMsg<EK>) -> Result<()> {
        match self.send(msg) {
            Ok(()) => Ok(()),
            Err(mpsc::SendError(_)) => Err(Error::Transport(DiscardReason::Disconnected)),
        }
    }
}
