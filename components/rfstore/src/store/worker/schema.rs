// Copyright 2025 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    fmt,
    fmt::{Display, Formatter},
};

use kvengine::{
    table::{BoundedDataSet, OwnedInnerKey},
    Shard, STORAGE_CLASS_KEY,
};
use kvproto::{metapb, metapb::Region};
use schema::schema::StorageClass;
use tikv_util::{codec::bytes::encode_bytes, info, time::Instant, warn, worker::Runnable, Either};

use crate::{
    store::{Callback, CasualMessage, PeerMsg, PeerTag, RegionIdVer, StoreMsg},
    RaftRouter,
};

pub enum SchemaTask {
    StorageClass {
        region: metapb::Region,
        schema_version: i64,
    },
}

impl Display for SchemaTask {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            SchemaTask::StorageClass {
                ref region,
                ref schema_version,
            } => {
                write!(
                    f,
                    "check schema of region {} schema_version {}",
                    region.id, schema_version
                )
            }
        }
    }
}

pub struct SchemaRunner {
    store_id: u64,
    kv: kvengine::Engine,
    router: RaftRouter,
}

impl Runnable for SchemaRunner {
    type Task = SchemaTask;
    fn run(&mut self, task: SchemaTask) {
        match task {
            SchemaTask::StorageClass {
                region,
                schema_version,
            } => self.handle_storage_class(region, schema_version),
        }
    }
}

impl SchemaRunner {
    pub fn new(store_id: u64, kv: kvengine::Engine, router: RaftRouter) -> Self {
        Self {
            store_id,
            kv,
            router,
        }
    }

    fn peer_tag(&self, region: &Region) -> PeerTag {
        let id_ver = RegionIdVer::from_region(region);
        PeerTag::new(self.store_id, id_ver)
    }

    fn handle_storage_class(&self, region: metapb::Region, schema_version: i64) {
        let tag = self.peer_tag(&region);
        if let Ok(shard) = self
            .kv
            .get_shard_with_ver(region.get_id(), region.get_region_epoch().version)
        {
            if let Some(schema_file) = shard.get_schema_file() {
                if schema_version != schema_file.get_version() {
                    return;
                }
                if shard.get_checked_schema_ver() >= schema_version {
                    return;
                }

                info!("{} handle storage class", tag; "schema_version" => schema_version);
                match schema_file.overlap_storage_class_tables(shard.range.data_bound()) {
                    Either::Left(table_id) => {
                        // The table fully covers the region.
                        let schema = schema_file.get_table(table_id).unwrap();
                        let sc = schema.get_storage_class();
                        debug_assert!(sc.is_specified());
                        if self.update_storage_class(&tag, &shard, sc, schema_version) {
                            shard.set_checked_schema_ver(schema_version);
                            info!("{} update storage class", tag; "sc" => ?sc, "schema_version" => schema_version);
                        }
                    }
                    Either::Right(table_inner_keys) => {
                        if !table_inner_keys.is_empty() {
                            // The overlapped keys of tables require exclusive region.
                            self.split_regions_for_tables(&tag, &shard, &region, &table_inner_keys);
                        } else {
                            // No table requires exclusive region, nothing to do.
                            shard.set_checked_schema_ver(schema_version);
                        }
                    }
                }
            }
        } else {
            info!("{} handle storage class: skip, shard not found/match", tag;
                "region" => ?region, "schema_version" => schema_version);
        }
    }

    fn split_regions_for_tables(
        &self,
        tag: &PeerTag,
        shard: &Shard,
        region: &metapb::Region,
        table_inner_keys: &[OwnedInnerKey],
    ) {
        info!(
            "{} handle storage class: schedule ask split", tag;
            "table_keys" => ?table_inner_keys,
        );
        let split_keys = table_inner_keys
            .iter()
            .map(|inner_key| encode_bytes(&shard.to_outer_key(inner_key.as_ref())))
            .collect::<Vec<_>>();
        let msg = CasualMessage::SplitRegion {
            region_epoch: region.get_region_epoch().clone(),
            split_keys,
            callback: Callback::None,
            source: "schema".into(),
        };
        self.router.send(shard.id, PeerMsg::CasualMessage(msg));
    }

    fn update_storage_class(
        &self,
        tag: &PeerTag,
        shard: &Shard,
        storage_class: StorageClass,
        schema_version: i64,
    ) -> bool {
        let mut cs = kvengine::new_change_set(shard.id, shard.ver);
        cs.set_property_key(STORAGE_CLASS_KEY.to_string());
        cs.set_property_value(storage_class.marshal());
        info!("{} propose update storage_class property", tag; "sc" => ?storage_class);
        let msg = StoreMsg::GenerateEngineChangeSet(cs);
        if let Err(e) = self.router.store_sender.send(msg) {
            warn!("{} failed to to send meta change message", tag; "err" => ?e, "sc" => ?storage_class);
            false
        } else {
            // Wait to finish updating the storage class property.
            let begin = Instant::now_coarse();
            let timeout = std::time::Duration::from_secs(10);
            loop {
                if let Some(schema_file) = shard.get_schema_file() {
                    if schema_version != schema_file.get_version() {
                        // If the schema version changes, recheck.
                        return false;
                    }
                }
                let current = shard.get_storage_class();
                if current == storage_class {
                    return true;
                }
                if begin.saturating_elapsed() < timeout {
                    std::thread::sleep(std::time::Duration::from_millis(100));
                    continue;
                }

                warn!("{} wait for updating storage class property timeout", tag;
                    "current" => ?current, "expect" => ?storage_class);
                break false;
            }
        }
    }
}
