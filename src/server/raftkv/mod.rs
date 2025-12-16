// Copyright 2016 TiKV Project Authors. Licensed under Apache-2.0.

mod raft_extension;

// #[PerformanceCriticalPath]
use std::{
    borrow::Cow,
    fmt::{self, Debug, Display, Formatter},
    io::Error as IoError,
    result,
    sync::Arc,
    time::Duration,
};

use collections::HashMap;
use concurrency_manager::ConcurrencyManager;
use engine_traits::{KvEngine, Snapshot};
use futures::{future::BoxFuture, Future, Stream};
use futures_util::stream::empty;
use kvproto::{
    errorpb,
    kvrpcpb::{Context, IsolationLevel},
    raft_cmdpb::{RaftCmdRequest, RaftCmdResponse, Response},
};
use raft::{eraftpb, eraftpb::MessageType, StateRole};
pub use raft_extension::RaftRouterWrap;
use raftstore::{
    coprocessor::{
        dispatcher::BoxReadIndexObserver, Coprocessor, CoprocessorHost, ReadIndexObserver,
    },
    errors::Error as RaftServerError,
    router::{LocalReadRouter, RaftStoreRouter},
    store::{RaftCmdExtraOpts, ReadIndexContext, RegionSnapshot},
    RegionInfoAccessor,
};
use thiserror::Error;
use tikv_kv::{Modify, OnAppliedCb, WriteEvent};
use tikv_util::{future::paired_future_callback, time::Instant};
use txn_types::{TxnExtra, TxnExtraScheduler};

use crate::{
    server::metrics::REPLICA_READ_LOCK_CHECK_HISTOGRAM_VEC_STATIC,
    storage::{
        kv,
        kv::{Engine, Error as KvError, ErrorInner as KvErrorInner, SnapContext, WriteData},
    },
};

#[derive(Debug, Error)]
pub enum Error {
    #[error("{}", .0.get_message())]
    RequestFailed(errorpb::Error),

    #[error("{0}")]
    Io(#[from] IoError),

    #[error("{0}")]
    Server(#[from] RaftServerError),

    #[error("{0}")]
    InvalidResponse(String),

    #[error("{0}")]
    InvalidRequest(String),

    #[error("timeout after {0:?}")]
    Timeout(Duration),
}

pub type Result<T> = result::Result<T, Error>;

impl From<Error> for kv::Error {
    fn from(e: Error) -> kv::Error {
        match e {
            Error::RequestFailed(e) => KvError::from(KvErrorInner::Request(e)),
            Error::Server(e) => e.into(),
            e => box_err!(e),
        }
    }
}

pub enum CmdRes<S>
where
    S: Snapshot,
{
    Resp(Vec<Response>),
    Snap(RegionSnapshot<S>),
}

fn check_raft_cmd_response(resp: &mut RaftCmdResponse) -> Result<()> {
    if resp.get_header().has_error() {
        return Err(Error::RequestFailed(resp.take_header().take_error()));
    }

    Ok(())
}

fn exec_admin<E: KvEngine, S: RaftStoreRouter<E>>(
    router: &S,
    req: RaftCmdRequest,
) -> BoxFuture<'static, kv::Result<()>> {
    let (cb, f) = paired_future_callback();
    let res = router.send_command(
        req,
        raftstore::store::Callback::write(cb),
        RaftCmdExtraOpts::default(),
    );
    Box::pin(async move {
        res?;
        let mut resp = box_try!(f.await);
        check_raft_cmd_response(&mut resp.response)?;
        Ok(())
    })
}

/// `RaftKv` is a storage engine base on `RaftStore`.
#[derive(Clone)]
pub struct RaftKv<E, S>
where
    E: KvEngine,
    S: RaftStoreRouter<E> + LocalReadRouter<E> + 'static,
{
    router: RaftRouterWrap<S, E>,
    engine: E,
    txn_extra_scheduler: Option<Arc<dyn TxnExtraScheduler>>,
}

impl<E, S> RaftKv<E, S>
where
    E: KvEngine,
    S: RaftStoreRouter<E> + LocalReadRouter<E> + 'static,
{
    /// Create a RaftKv using specified configuration.
    pub fn new(router: S, engine: E, _: RegionInfoAccessor) -> RaftKv<E, S> {
        RaftKv {
            router: RaftRouterWrap::new(router),
            engine,
            txn_extra_scheduler: None,
        }
    }

    pub fn set_txn_extra_scheduler(&mut self, txn_extra_scheduler: Arc<dyn TxnExtraScheduler>) {
        self.txn_extra_scheduler = Some(txn_extra_scheduler);
    }
}

impl<E, S> Display for RaftKv<E, S>
where
    E: KvEngine,
    S: RaftStoreRouter<E> + LocalReadRouter<E> + 'static,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "RaftKv")
    }
}

impl<E, S> Debug for RaftKv<E, S>
where
    E: KvEngine,
    S: RaftStoreRouter<E> + LocalReadRouter<E> + 'static,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "RaftKv")
    }
}

impl<E, S> Engine for RaftKv<E, S>
where
    E: KvEngine,
    S: RaftStoreRouter<E> + LocalReadRouter<E> + 'static,
{
    type Snap = RegionSnapshot<E::Snapshot>;
    type Local = E;

    fn kv_engine(&self) -> Option<E> {
        Some(self.engine.clone())
    }

    type RaftExtension = RaftRouterWrap<S, E>;
    #[inline]
    fn raft_extension(&self) -> &Self::RaftExtension {
        &self.router
    }

    fn modify_on_kv_engine(&self, _region_modifies: HashMap<u64, Vec<Modify>>) -> kv::Result<()> {
        unimplemented!()
    }

    fn precheck_write_with_ctx(&self, _: &Context) -> kv::Result<()> {
        unimplemented!()
    }

    type WriteRes = impl Stream<Item = WriteEvent> + Send + Unpin;
    fn async_write(
        &self,
        _ctx: &Context,
        _batch: WriteData,
        _subscribed: u8,
        _on_applied: Option<OnAppliedCb>,
    ) -> Self::WriteRes {
        empty()
    }

    type SnapshotRes = impl Future<Output = kv::Result<Self::Snap>> + Send;
    fn async_snapshot(&mut self, _ctx: SnapContext<'_>) -> Self::SnapshotRes {
        async move { unimplemented!() }
    }

    fn schedule_txn_extra(&self, txn_extra: TxnExtra) {
        if let Some(tx) = self.txn_extra_scheduler.as_ref() {
            if !txn_extra.is_empty() {
                tx.schedule(txn_extra);
            }
        }
    }
}

#[derive(Clone)]
pub struct ReplicaReadLockChecker {
    concurrency_manager: ConcurrencyManager,
}

impl ReplicaReadLockChecker {
    pub fn new(concurrency_manager: ConcurrencyManager) -> Self {
        ReplicaReadLockChecker {
            concurrency_manager,
        }
    }

    pub fn register<E: KvEngine + 'static>(self, host: &mut CoprocessorHost<E>) {
        host.registry
            .register_read_index_observer(1, BoxReadIndexObserver::new(self));
    }
}

impl Coprocessor for ReplicaReadLockChecker {}

impl ReadIndexObserver for ReplicaReadLockChecker {
    fn on_step(&self, msg: &mut eraftpb::Message, role: StateRole) {
        // Only check and return result if the current peer is a leader.
        // If it's not a leader, the read index request will be redirected to the leader
        // later.
        if msg.get_msg_type() != MessageType::MsgReadIndex || role != StateRole::Leader {
            return;
        }
        assert_eq!(msg.get_entries().len(), 1);
        let mut rctx = ReadIndexContext::parse(msg.get_entries()[0].get_data()).unwrap();
        if let Some(mut request) = rctx.request.take() {
            let begin_instant = Instant::now();

            let start_ts = request.get_start_ts().into();
            self.concurrency_manager.update_max_ts(start_ts);
            for range in request.mut_key_ranges().iter_mut() {
                let key_bound = |key: Vec<u8>| {
                    if key.is_empty() {
                        None
                    } else {
                        Some(txn_types::Key::from_encoded(key))
                    }
                };
                let start_key = key_bound(range.take_start_key());
                let end_key = key_bound(range.take_end_key());
                // The replica read is not compatible with `RcCheckTs` isolation level yet.
                // It's ensured in the tidb side when `RcCheckTs` is enabled for read requests,
                // the replica read would not be enabled at the same time.
                let res = self.concurrency_manager.read_range_check(
                    start_key.as_ref(),
                    end_key.as_ref(),
                    |key, lock| {
                        txn_types::Lock::check_ts_conflict(
                            Cow::Borrowed(lock),
                            key,
                            start_ts,
                            &Default::default(),
                            IsolationLevel::Si,
                        )
                    },
                );
                if let Err(txn_types::Error(box txn_types::ErrorInner::KeyIsLocked(lock))) = res {
                    rctx.locked = Some(lock);
                    REPLICA_READ_LOCK_CHECK_HISTOGRAM_VEC_STATIC
                        .locked
                        .observe(begin_instant.saturating_elapsed().as_secs_f64());
                } else {
                    REPLICA_READ_LOCK_CHECK_HISTOGRAM_VEC_STATIC
                        .unlocked
                        .observe(begin_instant.saturating_elapsed().as_secs_f64());
                }
            }
            msg.mut_entries()[0].set_data(rctx.to_bytes().into());
        }
    }
}
