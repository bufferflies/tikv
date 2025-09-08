// Copyright 2025 TiKV Project Authors. Licensed under Apache-2.0.

use std::collections::HashMap;

use api_version::ApiV2;
use kvengine::{
    table::memtable::WriteBatchEntry, IdVer, ShardTag, SnapAccess, UserMeta, WriteBatch, LOCK_CF,
    WRITE_CF,
};
use kvproto::{cdcpb, cdcpb::Event, raft_cmdpb::AdminRequest};
use log_wrappers::Value as LogValue;
use rfstore::store::ApplyObserver;
use tidb_query_datatype::codec::table::{INDEX_PREFIX_SEP, PREFIX_LEN, TABLE_PREFIX};
use tikv_util::{debug, error, info};
use txn_types::{Lock, LockType};

use crate::CdcMsg;

pub struct CdcApplyObserver {
    store_id: u64,
    kv: kvengine::Engine,
    sender: tikv_util::mpsc::Sender<CdcMsg>,
    region_events: HashMap<u64, RegionEvents>,
    runtime: tokio::runtime::Handle,
    join_handles: Vec<tokio::task::JoinHandle<EventBuilder>>,
}

#[derive(Default)]
pub struct RegionEvents {
    pub events: Vec<cdcpb::Event>,
    pub tracked_locks: Vec<(Vec<u8>, u64)>,
}

impl CdcApplyObserver {
    pub fn new(
        kv: kvengine::Engine,
        sender: tikv_util::mpsc::Sender<CdcMsg>,
        runtime: tokio::runtime::Handle,
    ) -> Self {
        let store_id = kv.get_engine_id();
        Self {
            store_id,
            kv,
            sender,
            region_events: HashMap::default(),
            runtime,
            join_handles: Vec::new(),
        }
    }

    fn send_msg(
        sender: &tikv_util::mpsc::Sender<CdcMsg>,
        region_id: u64,
        region_events: RegionEvents,
    ) {
        let msg = CdcMsg::Applied {
            region_id,
            region_events,
        };
        if let Err(e) = sender.send(msg) {
            error!("send cdc event failed"; "error" => ?e);
        }
    }
}

impl ApplyObserver for CdcApplyObserver {
    fn on_apply(&mut self, region_id: u64, log_index: u64, wb: &WriteBatch) {
        let write_cf = wb.get_cf(WRITE_CF);
        let snap_access = self.kv.get_snap_access(region_id).unwrap();
        let mut event_builder = EventBuilder::new(snap_access, log_index);
        write_cf.iterate(|entry, buf| {
            event_builder.add_row(entry, buf);
        });
        let lock_cf = wb.get_cf(LOCK_CF);
        lock_cf.iterate(|entry, buf| {
            event_builder.add_lock(entry, buf);
        });
        let handle = self.runtime.spawn(async move {
            event_builder.fetch_old_values().await;
            event_builder
        });
        self.join_handles.push(handle);
    }

    fn on_apply_admin(
        &mut self,
        region_id: u64,
        region_version: u64,
        _log_index: u64,
        admin: &AdminRequest,
    ) {
        if admin.has_splits()
            || admin.has_prepare_merge()
            || admin.has_commit_merge()
            || admin.has_rollback_merge()
        {
            let tag = ShardTag::new(self.store_id, IdVer::new(region_id, region_version));
            if let Some(region_events) = self.region_events.remove(&region_id) {
                Self::send_msg(&self.sender, region_id, region_events);
            }
            info!("{} on apply admin {:?}", tag, admin);
            let msg = CdcMsg::AppliedAdmin {
                region_id,
                region_version,
                admin: admin.clone(),
            };
            if let Err(e) = self.sender.send(msg) {
                error!("{} send cdc event failed", tag; "error" => ?e);
            }
        }
    }

    fn flush(&mut self) {
        for handle in self.join_handles.drain(..) {
            let Ok(mut event_builder) = self.runtime.block_on(handle) else {
                error!("Failed to join event builder task");
                continue;
            };
            let event = event_builder.build();
            let events = self
                .region_events
                .entry(event.region_id)
                .or_insert_with(|| RegionEvents::default());
            events.events.push(event);
            events
                .tracked_locks
                .extend_from_slice(&event_builder.tracked_locks);
        }
        for (region_id, region_events) in self.region_events.drain() {
            Self::send_msg(&self.sender, region_id, region_events);
        }
    }
}

struct EventBuilder {
    snap_access: SnapAccess,
    index: u64,
    rows: cdcpb::EventEntries,
    tracked_locks: Vec<(Vec<u8>, u64)>,
}

impl EventBuilder {
    fn new(snap_access: SnapAccess, index: u64) -> Self {
        Self {
            snap_access,
            index,
            rows: cdcpb::EventEntries::default(),
            tracked_locks: Vec::new(),
        }
    }

    fn add_row(&mut self, entry: &WriteBatchEntry, buf: &[u8]) {
        let entry_key = entry.key(buf);
        if is_index_key(entry_key) {
            // Skip index keys as CDC does not need them.
            return;
        }
        let mut event_row = cdcpb::EventRow::default();
        event_row.set_key(entry_key.to_vec());
        event_row.set_value(entry.value(buf).to_vec());
        event_row.set_commit_ts(entry.version);
        let user_meta = UserMeta::from_slice(entry.user_meta(buf));
        event_row.set_start_ts(user_meta.start_ts);
        if event_row.get_value().is_empty() {
            event_row.set_op_type(cdcpb::EventRowOpType::Delete);
        } else {
            event_row.set_op_type(cdcpb::EventRowOpType::Put);
        }
        event_row.set_type(cdcpb::EventLogType::Committed);
        self.rows.mut_entries().push(event_row);

        debug!("{} add_row", self.snap_access.get_tag();
            "key" => LogValue::key(entry_key),
            "version" => entry.version,
            "start_ts" => user_meta.start_ts,
            "commit_ts" => user_meta.commit_ts,
            "log_index" => self.index,
        );
    }

    fn add_lock(&mut self, entry: &WriteBatchEntry, buf: &[u8]) {
        let entry_key = entry.key(buf);
        if is_index_key(entry_key) {
            // Skip index keys as CDC does not need them.
            return;
        }
        if entry.value(buf).is_empty() {
            self.tracked_locks.push((entry_key.to_vec(), 0));

            debug!("{} add_lock (untrack)", self.snap_access.get_tag();
                "key" => LogValue::key(entry_key),
                "version" => entry.version,
                "log_index" => self.index,
            );
            return;
        }
        let lock = Lock::parse(entry.value(buf)).unwrap();
        self.tracked_locks
            .push((entry.key(buf).to_vec(), lock.ts.into_inner()));
        debug!("{} add_lock", self.snap_access.get_tag();
            "key" => LogValue::key(entry_key),
            "version" => entry.version,
            "lock.ts" => lock.ts,
            "lock.ty" => ?lock.lock_type,
            "log_index" => self.index,
        );

        let mut event_row = cdcpb::EventRow::default();
        event_row.set_key(entry_key.to_vec());
        event_row.set_start_ts(lock.ts.into_inner());
        let short_value = lock.short_value.unwrap_or_default();
        event_row.set_value(short_value);
        match lock.lock_type {
            LockType::Put => {
                event_row.set_op_type(cdcpb::EventRowOpType::Put);
            }
            LockType::Delete => {
                event_row.set_op_type(cdcpb::EventRowOpType::Delete);
            }
            LockType::Lock | LockType::Pessimistic => {
                // ignore op_lock & pessimistic lock
                return;
            }
        }
        event_row.set_type(cdcpb::EventLogType::Prewrite);
        self.rows.mut_entries().push(event_row);
    }

    async fn fetch_old_values(&mut self) {
        for row in self.rows.entries.iter_mut() {
            if row.get_type() != cdcpb::EventLogType::Committed {
                continue;
            }
            let mut keyspace_key =
                ApiV2::get_keyspace_prefix_by_id(self.snap_access.get_keyspace_id());
            keyspace_key.extend_from_slice(row.get_key());
            let old_item = self
                .snap_access
                .get_async(WRITE_CF, &keyspace_key, row.commit_ts - 1)
                .await;
            if !old_item.get_value().is_empty() {
                row.set_old_value(old_item.get_value().to_vec());
            }
        }
    }

    fn build(&mut self) -> Event {
        let mut event = cdcpb::Event::default();
        event.index = self.index;
        event.region_id = self.snap_access.get_id();
        event.mut_entries().set_entries(self.rows.take_entries());
        event
    }
}

pub(crate) fn is_index_key(key: &[u8]) -> bool {
    if key.len() < PREFIX_LEN {
        return false;
    }
    let trimmed_key = &key[..PREFIX_LEN];
    trimmed_key.starts_with(TABLE_PREFIX) && trimmed_key.ends_with(INDEX_PREFIX_SEP)
}
