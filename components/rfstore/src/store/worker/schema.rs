// Copyright 2025 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    fmt,
    fmt::{Display, Formatter},
};

use api_version::api_v2::KEYSPACE_PREFIX_LEN;
use bytes::Buf;
use kvengine::{
    get_split_keys_for_exclusive_tables,
    table::columnar::{IA_STORAGE_CLASS, STANDARD_STORAGE_CLASS},
    STORAGE_CLASS_KEY,
};
use kvproto::{metapb, metapb::Region};
use tidb_query_datatype::codec::table::decode_table_id;
use tikv_util::{info, time::Instant, warn, worker::Runnable};

use crate::{
    store::{Callback, CasualMessage, PeerMsg, PeerTag, RegionIdVer, StoreMsg},
    RaftRouter,
};

pub enum SchemaTask {
    StorageClass {
        region: metapb::Region,
        peer: metapb::Peer,
        schema_version: i64,
    },
}

impl Display for SchemaTask {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            SchemaTask::StorageClass {
                ref region,
                ref peer,
                ref schema_version,
            } => {
                write!(
                    f,
                    "check schema of region {} peer {} schema_version {}",
                    region.id, peer.id, schema_version
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
                peer,
                schema_version,
            } => self.handle_storage_class(region, peer, schema_version),
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

    fn handle_storage_class(
        &self,
        region: metapb::Region,
        peer: metapb::Peer,
        schema_version: i64,
    ) {
        let tag = self.peer_tag(&region);
        let region_id = region.get_id();
        let peer_id = peer.get_id();
        if let Some(shard) = self.kv.get_shard(region.get_id()) {
            if let Some(schema_file) = shard.get_schema_file() {
                if schema_version != schema_file.get_version() {
                    return;
                }
                if shard.get_checked_schema_ver() >= schema_version {
                    return;
                }
                let ia_storage_class_tables = schema_file.overlap_storage_class_tables(
                    &shard.range.outer_start,
                    &shard.range.outer_end,
                    IA_STORAGE_CLASS,
                );
                let mut storage_class_property: Option<u8> = None;
                if !ia_storage_class_tables.is_empty() {
                    let ia_tables_len = ia_storage_class_tables.len();
                    let split_keys = get_split_keys_for_exclusive_tables(
                        &shard.range.outer_start,
                        &shard.range.outer_end,
                        shard.range.keyspace_id,
                        ia_storage_class_tables,
                    );
                    if !split_keys.is_empty() {
                        info!(
                            "schedule ask split";
                            "tag" => tag,
                            "peer_id" => peer_id,
                            "ia_tables" => ia_tables_len,
                        );
                        let msg = CasualMessage::SplitRegion {
                            region_epoch: region.get_region_epoch().clone(),
                            split_keys,
                            callback: Callback::None,
                            source: "schema".into(),
                        };
                        self.router.send(region_id, PeerMsg::CasualMessage(msg));
                        return;
                    } else if ia_tables_len == 1 && shard.get_storage_class() != IA_STORAGE_CLASS {
                        storage_class_property = Some(IA_STORAGE_CLASS)
                    }
                } else {
                    let standard_storage_class_tables = schema_file.overlap_storage_class_tables(
                        &shard.range.outer_start,
                        &shard.range.outer_end,
                        STANDARD_STORAGE_CLASS,
                    );
                    if standard_storage_class_tables.len() == 1 {
                        let mut start_key: &[u8] = &shard.range.outer_start;
                        let mut end_key: &[u8] = &shard.range.outer_end;
                        start_key.advance(KEYSPACE_PREFIX_LEN);
                        end_key.advance(KEYSPACE_PREFIX_LEN);
                        let start_table_id = decode_table_id(start_key).unwrap_or(0);
                        let end_table_id = decode_table_id(end_key).unwrap_or(i64::MAX);
                        if end_table_id - start_table_id < 2
                            && shard.get_storage_class() != STANDARD_STORAGE_CLASS
                        {
                            storage_class_property = Some(STANDARD_STORAGE_CLASS)
                        }
                    }
                }
                if let Some(storage_class) = storage_class_property {
                    let mut cs = kvengine::new_change_set(shard.id, shard.ver);
                    cs.set_property_key(STORAGE_CLASS_KEY.to_string());
                    cs.set_property_value([storage_class].to_vec());
                    info!(
                        "{} propose update storage_class property {}",
                        tag, storage_class
                    );
                    let msg = StoreMsg::GenerateEngineChangeSet(cs);
                    if let Err(e) = self.router.store_sender.send(msg) {
                        info!(
                            "failed to to send meta change message";
                            "err" => ?e,
                        )
                    } else {
                        // Wait to finish updating the storage class property.
                        let begin = Instant::now_coarse();
                        let timeout = std::time::Duration::from_secs(10);
                        loop {
                            if let Some(schema_file) = shard.get_schema_file() {
                                if schema_version != schema_file.get_version() {
                                    // If the schema version changes, recheck.
                                    return;
                                }
                            }
                            let current = shard.get_storage_class();
                            if current == storage_class {
                                shard.set_checked_schema_ver(schema_version);
                                return;
                            }
                            if begin.saturating_elapsed() < timeout {
                                std::thread::sleep(std::time::Duration::from_millis(100));
                                continue;
                            }

                            warn!("{} wait for updating storage class property timeout", tag; "current" => current, "expect" => storage_class);
                            break;
                        }
                    }
                } else {
                    shard.set_checked_schema_ver(schema_version);
                }
            }
        }
    }
}
