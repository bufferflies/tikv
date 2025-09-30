// Copyright 2025 TiKV Project Authors. Licensed under Apache-2.0.

use std::{collections::VecDeque, sync::Arc, time::Duration};

use collections::{HashMap, HashMapExt};
use kvproto::metapb;
use merged_engine::StoreProgress;
use native_br::wal::AssembledWalData;
use parking_lot::Mutex;
use pd_client::{util::get_all_stores_except_tiflash_async, PdClient};
use rfengine::service_worker::WalProgress;
use security::HttpClient;
use tikv_util::{box_err, debug, error, time::Instant, warn};
use tokio::task::JoinSet;
use txn_types::TimeStamp;

use crate::{util::send_request_to_store, Error, Result};

// None: When the store is not ready.
pub(crate) type StoreWalProgresses = HashMap<u64 /* store_id */, Option<StoreProgress>>;

#[derive(Clone, Default)]
pub(crate) struct WalProgressTargets {
    queue: Arc<Mutex<VecDeque<(TimeStamp, StoreWalProgresses)>>>,
}

impl WalProgressTargets {
    pub(crate) fn push_back(&self, target: (TimeStamp, StoreWalProgresses)) {
        // TODO: limit the queue length.
        self.queue.lock().push_back(target);
    }

    pub(crate) fn pop_front(&self) -> Option<(TimeStamp, StoreWalProgresses)> {
        self.queue.lock().pop_front()
    }

    pub(crate) fn front(&self) -> Option<(TimeStamp, StoreWalProgresses)> {
        self.queue.lock().front().cloned()
    }
}

pub(crate) struct WalProgressFetcher {
    pd: Arc<dyn PdClient>,
    timeout: Duration,
    thread_pool: tokio::runtime::Handle,
    http_client: HttpClient,
}

impl WalProgressFetcher {
    pub(crate) fn run(
        pd: Arc<dyn PdClient>,
        timeout: Duration,
        thread_pool: tokio::runtime::Handle,
        targets: WalProgressTargets,
        interval: Duration,
    ) {
        let fetcher = Arc::new(Self::new(pd, timeout, thread_pool.clone()));
        thread_pool.spawn(async move {
            loop {
                let start_time = Instant::now_coarse();

                match fetcher.fetch_target_ts_and_progress().await {
                    Ok((ts, progress)) => {
                        debug!("WalProgressFetcher: new target"; "ts" => ts, "progress" => ?progress);
                        targets.push_back((ts, progress));
                    }
                    Err(err) => {
                        error!("fetch_target_ts_and_progress: failed: {:?}", err);
                    }
                }

                let elapsed = start_time.saturating_elapsed();
                tokio::time::sleep(interval.saturating_sub(elapsed)).await;
            }
        });
    }

    fn new(pd: Arc<dyn PdClient>, timeout: Duration, thread_pool: tokio::runtime::Handle) -> Self {
        let http_client = pd
            .get_security_mgr()
            .http_client(hyper::Client::builder())
            .unwrap();
        Self {
            pd,
            timeout,
            http_client,
            thread_pool,
        }
    }

    async fn fetch_target_ts_and_progress(
        self: &Arc<Self>,
    ) -> Result<(TimeStamp, StoreWalProgresses)> {
        let stores = get_all_stores_except_tiflash_async(self.pd.as_ref()).await?;
        let ts = self.pd.get_min_tso().await?;

        let mut errors = vec![];
        let mut progresses = HashMap::with_capacity(stores.len());
        let mut join_set = JoinSet::new();
        for store in stores {
            let fetcher = self.clone();
            join_set.spawn_on(
                async move {
                    let res = fetcher.track_store_wal_progress(&store, ts).await;
                    (store.id, res)
                },
                &self.thread_pool,
            );
        }

        while let Some(task) = join_set.join_next().await {
            let (store_id, res) = task.expect("task panic");
            match res {
                Ok(wal_progress) => {
                    progresses.insert(
                        store_id,
                        Some(StoreProgress {
                            store_id,
                            epoch: wal_progress.epoch,
                            offset: wal_progress.offset,
                        }),
                    );
                }
                Err(err) => {
                    warn!("track_store_wal_progress: failed: {:?}", err; "store" => store_id);
                    progresses.insert(store_id, None);
                    errors.push(err);
                }
            }
        }

        if errors.len() > 1 {
            return Err(errors.pop().unwrap());
        }

        Ok((ts, progresses))
    }

    async fn track_store_wal_progress(
        &self,
        store: &metapb::Store,
        ts: TimeStamp,
    ) -> Result<StoreProgress> {
        let security_mgr = self.pd.get_security_mgr();
        let uri = security_mgr
            .build_uri(format!(
                "{}/rfengine/track_wal_progress",
                &store.status_address
            ))
            .unwrap();
        let track_req = cloud_server::TrackWalProgressRequest {
            ts: ts.into_inner(),
        };
        let req = http::Request::post(&uri)
            .body(serde_json::to_vec(&track_req).unwrap().into())
            .unwrap();
        let (status, data) = send_request_to_store(req, &self.http_client, self.timeout).await?;
        if !status.is_success() {
            let err_str = String::from_utf8_lossy(&data);
            return Err(box_err!("track_store_wal_progress: {}", err_str));
        }
        let progress: WalProgress = serde_json::from_slice(&data).map_err(|e| -> Error {
            box_err!(
                "track_store_wal_progress: decode error: {}, store {}",
                e,
                store.id
            )
        })?;
        debug_assert_ne!(progress.offset, 0); // 0 will be conflict with "read to end" during dump WAL.
        Ok(StoreProgress {
            store_id: store.id,
            epoch: progress.epoch,
            offset: progress.offset,
        })
    }
}

#[derive(Default)]
pub(crate) struct WalCache {
    inner: HashMap<u64 /* store_id */, Option<(u32 /* epoch_id */, AssembledWalData)>>,
}

impl WalCache {
    pub(crate) fn get_mut(&mut self, store_id: u64, epoch: u32) -> Option<&mut AssembledWalData> {
        let entry = self.inner.get_mut(&store_id)?.as_mut()?;
        if entry.0 != epoch {
            debug_assert!(false);
            None
        } else {
            Some(&mut entry.1)
        }
    }

    pub(crate) fn insert(&mut self, store_id: u64, epoch: u32, wal_data: AssembledWalData) {
        self.inner.insert(store_id, Some((epoch, wal_data)));
    }

    pub(crate) fn remove_cache(&mut self, store_id: u64) {
        if let Some(entry) = self.inner.get_mut(&store_id) {
            entry.take();
        }
    }

    pub(crate) fn contains_store(&self, store_id: u64) -> bool {
        self.inner.contains_key(&store_id)
    }
}
