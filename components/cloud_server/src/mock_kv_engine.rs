// Copyright 2025 TiKV Project Authors. Licensed under Apache-2.0.

/// A mock Engine implementation for kvengine.
///
/// NOTE: we put this implementation here instead of under "tikv_kv" module
/// just because it depends on the `cloud_server::modifies_to_requests`,
/// but tikv_kv can't depend on "cloud_server" due to cyclic dependency.
use std::{
    fmt,
    pin::Pin,
    sync::{Arc, Mutex},
    task::Poll,
    time::Duration,
};

use collections::HashMap;
use engine_traits::{CfName, CF_WRITE};
use futures::{
    channel::{mpsc, oneshot},
    stream, Future, Stream,
};
use kvengine::test_engine::{new_test_engine_api_v2, ApplyTask};
use kvproto::kvrpcpb::Context;
use rfstore::store::{Applier, ApplyContext, CustomRaftLog, RegionSnapshot};
use tikv_kv::{
    Callback, Engine, Error, ErrorInner, Modify, OnAppliedCb, Result as EngineResult, SnapContext,
    TrackerToken, WriteData, WriteEvent,
};
use tikv_util::{
    box_err, box_try,
    worker::{Runnable, Scheduler, Worker},
};
use trace_event::types::TraceContext;
use txn_types::Key;

use crate::modifies_to_requests;

enum Task {
    Write(WriteData, Callback<()>),
    Snapshot(oneshot::Sender<RegionSnapshot>),
    #[allow(dead_code)]
    Pause(Duration),
}

impl fmt::Display for Task {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Task::Write(..) => write!(f, "write task"),
            Task::Snapshot(_) => write!(f, "snapshot task"),
            Task::Pause(_) => write!(f, "pause"),
        }
    }
}

struct Runner {
    engine: kvengine::Engine,
    applier: Arc<Mutex<TestApplier>>,
}

struct TestApplier {
    applier: Applier,
    apply_ctx: ApplyContext,
}

impl TestApplier {
    fn new(engine: kvengine::Engine) -> Self {
        let applier = Applier::new_for_test();
        let apply_ctx = ApplyContext::new(engine, None, None);
        Self { applier, apply_ctx }
    }

    fn write_data(&mut self, mut data: WriteData) {
        let req = modifies_to_requests(&Context::default(), &mut data);
        self.apply_ctx.exec_log_index += 1;
        let custom_log = CustomRaftLog::new_from_data(req.get_data());
        self.applier
            .exec_custom_log(
                &mut self.apply_ctx,
                TraceContext::default(),
                &custom_log,
                None,
            )
            .unwrap();
    }
}

impl Runnable for Runner {
    type Task = Task;

    fn run(&mut self, t: Task) {
        match t {
            Task::Write(modifies, cb) => {
                let mut applier = self.applier.lock().unwrap();
                applier.write_data(modifies);
                cb(Ok(()));
            }
            Task::Snapshot(sender) => {
                let _ = sender.send(RegionSnapshot::from_snapshot(
                    self.engine.get_shard(1).unwrap().new_snap_access(),
                    None,
                ));
            }
            Task::Pause(dur) => std::thread::sleep(dur),
        }
    }
}

#[derive(Clone)]
pub struct TestKvEngine {
    pub engine: kvengine::Engine,
    #[allow(dead_code)]
    applier: Arc<Mutex<TestApplier>>,
    sched: Scheduler<Task>,
    extras: Arc<Mutex<EngineExtras>>,
}

struct EngineExtras {
    _sender: tikv_util::mpsc::Sender<ApplyTask>,
    worker: Worker,
}

impl TestKvEngine {
    pub fn new() -> EngineResult<Self> {
        let (engine, sender) = new_test_engine_api_v2();
        let worker = Worker::new("engine-test-kvengine");
        let applier = Arc::new(Mutex::new(TestApplier::new(engine.clone())));
        let runner = Runner {
            engine: engine.clone(),
            applier: applier.clone(),
        };
        let sched = worker.start("engine-test-kvengine", runner);
        let extras = Arc::new(Mutex::new(EngineExtras {
            _sender: sender,
            worker,
        }));
        Ok(Self {
            engine: engine.engine.clone(),
            applier,
            sched,
            extras,
        })
    }

    pub fn stop(&self) {
        self.extras.lock().unwrap().worker.stop();
        self.engine.close();
    }
}

impl Engine for TestKvEngine {
    type Snap = rfstore::store::RegionSnapshot;
    type Local = kvengine::Engine;

    fn kv_engine(&self) -> Option<kvengine::Engine> {
        Some(self.engine.clone())
    }

    fn modify_on_kv_engine(&self, _region_modifies: HashMap<u64, Vec<Modify>>) -> EngineResult<()> {
        unimplemented!()
    }

    fn precheck_write_with_ctx(&self, _ctx: &Context) -> EngineResult<()> {
        Ok(())
    }

    type WriteRes = impl Stream<Item = WriteEvent> + Send + 'static;
    fn async_write(
        &self,
        _ctx: &Context,
        batch: WriteData,
        subscribed: u8,
        on_applied: Option<OnAppliedCb>,
        _tracker: Option<TrackerToken>,
    ) -> Self::WriteRes {
        if !batch.extra.one_pc && batch.extra.req_type == txn_types::ReqType::Noop {
            unreachable!("modifies: {:?}", batch.modifies);
        }
        let (mut tx, mut rx) = mpsc::channel::<WriteEvent>(WriteEvent::event_capacity(subscribed));
        let res = (move || {
            if batch.modifies.is_empty() {
                return Err(Error::from(ErrorInner::EmptyRequest));
            }

            if WriteEvent::subscribed_proposed(subscribed) {
                let _ = tx.try_send(WriteEvent::Proposed);
            }
            if WriteEvent::subscribed_committed(subscribed) {
                let _ = tx.try_send(WriteEvent::Committed);
            }
            let cb = Box::new(move |mut res| {
                if let Some(cb) = on_applied {
                    cb(&mut res);
                }
                let _ = tx.try_send(WriteEvent::Finished(res));
            });
            box_try!(self.sched.schedule(Task::Write(batch, cb)));
            Ok(())
        })();

        let mut res = Some(res);
        stream::poll_fn(move |cx| {
            if res.as_ref().map_or(false, |r| r.is_err()) {
                return Poll::Ready(res.take().map(WriteEvent::Finished));
            }
            // If it's none, it means an error is returned, it should not be polled again.
            assert!(res.is_some());
            Pin::new(&mut rx).poll_next(cx)
        })
    }

    type SnapshotRes = impl Future<Output = EngineResult<Self::Snap>> + Send;
    fn async_snapshot(&mut self, _: SnapContext<'_>) -> Self::SnapshotRes {
        let res: EngineResult<_> = (|| {
            let (tx, rx) = oneshot::channel();
            if self.sched.schedule(Task::Snapshot(tx)).is_err() {
                return Err(box_err!("failed to schedule snapshot"));
            }
            Ok(rx)
        })();

        async move { Ok(res?.await.unwrap()) }
    }

    fn delete_cf(&self, _ctx: &Context, cf: CfName, _key: Key) -> EngineResult<()> {
        assert_eq!(cf, CF_WRITE);
        // NOTE: this approach can delete the key, but it may directly remove the key
        // in the memory table, which violate the mvcc rule.

        // let ts = key.decode_ts().unwrap().next();
        // let mut wb = kvengine::write::WriteBatch::default();
        // let seq = {
        //     let mut applier = self.applier.lock().unwrap();
        //     applier.apply_ctx.exec_log_index += 1;
        //     applier.apply_ctx.exec_log_index
        // };
        // wb.set_sequence(seq);
        // wb.reset(1);
        // let raw_key = key.into_raw().unwrap();
        // wb.delete(WRITE_CF, &raw_key, ts.into_inner());
        // self.engine.write(&mut wb, &[]);
        // Ok(())
        unimplemented!("nexte gen does not support directly delete key.")
    }
}
