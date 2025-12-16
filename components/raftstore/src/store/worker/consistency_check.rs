// Copyright 2016 TiKV Project Authors. Licensed under Apache-2.0.

use std::fmt::{self, Display, Formatter};

use byteorder::{BigEndian, WriteBytesExt};
use engine_traits::{KvEngine, Snapshot};
use kvproto::metapb::Region;
use tikv_util::{error, info, warn, worker::Runnable};

use super::metrics::*;
use crate::{
    coprocessor::CoprocessorHost,
    store::{metrics::*, CasualMessage, CasualRouter},
};

/// Consistency checking task.
pub enum Task<S> {
    ComputeHash {
        index: u64,
        context: Vec<u8>,
        region: Region,
        snap: S,
    },
}

impl<S: Snapshot> Task<S> {
    pub fn compute_hash(region: Region, index: u64, context: Vec<u8>, snap: S) -> Task<S> {
        Task::ComputeHash {
            index,
            context,
            region,
            snap,
        }
    }
}

impl<S: Snapshot> Display for Task<S> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match *self {
            Task::ComputeHash {
                ref region, index, ..
            } => write!(f, "Compute Hash Task for {:?} at {}", region, index),
        }
    }
}

pub struct Runner<EK: KvEngine, C: CasualRouter<EK>> {
    router: C,
    coprocessor_host: CoprocessorHost<EK>,
}

impl<EK: KvEngine, C: CasualRouter<EK>> Runner<EK, C> {
    /// Computes the hash of the Region.
    fn compute_hash(&mut self, region: Region, index: u64, context: Vec<u8>, snap: EK::Snapshot) {
        if context.is_empty() {
            // For backward compatibility.
            warn!("skip compute hash without context"; "region_id" => region.get_id());
            return;
        }

        info!("computing hash"; "region_id" => region.get_id(), "index" => index);
        REGION_HASH_COUNTER.compute.all.inc();

        let timer = REGION_HASH_HISTOGRAM.start_coarse_timer();

        let hashes = match self
            .coprocessor_host
            .on_compute_hash(&region, &context, snap)
        {
            Ok(hashes) => hashes,
            Err(e) => {
                error!("calculate hash"; "region_id" => region.get_id(), "err" => ?e);
                REGION_HASH_COUNTER.compute.failed.inc();
                return;
            }
        };

        for (ctx, sum) in hashes {
            let mut checksum = Vec::with_capacity(4);
            checksum.write_u32::<BigEndian>(sum).unwrap();
            let msg = CasualMessage::ComputeHashResult {
                index,
                context: ctx,
                hash: checksum,
            };
            if let Err(e) = self.router.send(region.get_id(), msg) {
                warn!(
                    "failed to send hash compute result";
                    "region_id" => region.get_id(),
                    "err" => %e,
                );
            }
        }

        timer.observe_duration();
    }
}

impl<EK, C> Runnable for Runner<EK, C>
where
    EK: KvEngine,
    C: CasualRouter<EK>,
{
    type Task = Task<EK::Snapshot>;

    fn run(&mut self, task: Task<EK::Snapshot>) {
        match task {
            Task::ComputeHash {
                index,
                context,
                region,
                snap,
            } => self.compute_hash(region, index, context, snap),
        }
    }
}
