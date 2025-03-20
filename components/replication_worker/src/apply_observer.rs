// Copyright 2025 TiKV Project Authors. Licensed under Apache-2.0.

use std::collections::HashMap;

use api_version::ApiV2;
use kvengine::{UserMeta, WriteBatch, LOCK_CF, WRITE_CF};
use kvproto::{cdcpb, raft_cmdpb::AdminRequest};
use rfstore::store::ApplyObserver;
use tikv_util::{error, info};
use txn_types::{Lock, LockType};

use crate::CdcMsg;

pub struct CdcApplyObserver {
    #[allow(dead_code)]
    kv: kvengine::Engine, // TODO: get old value.
    sender: tikv_util::mpsc::Sender<CdcMsg>,
    region_events: HashMap<u64, RegionEvents>,
}

#[derive(Default)]
pub struct RegionEvents {
    pub events: Vec<cdcpb::Event>,
    pub tracked_locks: Vec<(Vec<u8>, u64)>,
}

impl CdcApplyObserver {
    pub fn new(kv: kvengine::Engine, sender: tikv_util::mpsc::Sender<CdcMsg>) -> Self {
        Self {
            kv,
            sender,
            region_events: HashMap::default(),
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
        let events = self
            .region_events
            .entry(region_id)
            .or_insert_with(|| RegionEvents::default());
        let mut event = cdcpb::Event::new();
        event.set_index(log_index);
        event.set_region_id(region_id);
        let mut entries = cdcpb::EventEntries::new();
        let write_cf = wb.get_cf(WRITE_CF);
        let snap_access = self.kv.get_snap_access(region_id).unwrap();
        write_cf.iterate(|entry, buf| {
            let mut row = cdcpb::EventRow::default();
            row.set_key(entry.key(buf).to_vec());
            row.set_value(entry.value(buf).to_vec());
            row.set_commit_ts(entry.version);
            let user_meta = UserMeta::from_slice(entry.user_meta(buf));
            row.set_start_ts(user_meta.start_ts);
            if row.get_value().is_empty() {
                row.set_op_type(cdcpb::EventRowOpType::Delete);
            } else {
                row.set_op_type(cdcpb::EventRowOpType::Put);
            }
            let mut keyspace_key = ApiV2::get_keyspace_prefix_by_id(snap_access.get_keyspace_id());
            keyspace_key.extend_from_slice(row.get_key());
            let old_item = snap_access.get(WRITE_CF, &keyspace_key, row.commit_ts - 1);
            if !old_item.get_value().is_empty() {
                row.set_old_value(old_item.get_value().to_vec());
            }
            row.set_type(cdcpb::EventLogType::Committed);
            entries.mut_entries().push(row);
        });
        let lock_cf = wb.get_cf(LOCK_CF);
        lock_cf.iterate(|entry, buf| {
            if entry.value(buf).is_empty() {
                events.tracked_locks.push((entry.key(buf).to_vec(), 0));
                return;
            }
            let mut event = cdcpb::EventRow::default();
            event.set_key(entry.key(buf).to_vec());
            let lock = Lock::parse(entry.value(buf)).unwrap();
            event.set_start_ts(lock.ts.into_inner());
            events
                .tracked_locks
                .push((entry.key(buf).to_vec(), event.get_start_ts()));
            let short_value = lock.short_value.unwrap_or_default();
            event.set_value(short_value);
            match lock.lock_type {
                LockType::Put => {
                    event.set_op_type(cdcpb::EventRowOpType::Put);
                }
                LockType::Delete => {
                    event.set_op_type(cdcpb::EventRowOpType::Delete);
                }
                LockType::Lock | LockType::Pessimistic => {
                    // ignore op_lock & pessimistic lock
                    return;
                }
            }
            event.set_type(cdcpb::EventLogType::Prewrite);
            entries.mut_entries().push(event);
        });
        event.set_entries(entries);
        events.events.push(event);
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
            if let Some(region_events) = self.region_events.remove(&region_id) {
                Self::send_msg(&self.sender, region_id, region_events);
            }
            info!(
                "{}:{} on apply admin {:?}",
                region_id, region_version, admin
            );
            let msg = CdcMsg::AppliedAdmin {
                region_id,
                region_version,
                admin: admin.clone(),
            };
            if let Err(e) = self.sender.send(msg) {
                error!("send cdc event failed"; "error" => ?e);
            }
        }
    }

    fn flush(&mut self) {
        for (region_id, region_events) in self.region_events.drain() {
            Self::send_msg(&self.sender, region_id, region_events);
        }
    }
}
