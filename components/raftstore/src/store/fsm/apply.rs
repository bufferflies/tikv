// Copyright 2017 TiKV Project Authors. Licensed under Apache-2.0.

// #[PerformanceCriticalPath]
use std::fmt::Debug;

use crate::coprocessor::ObserveHandle;

#[derive(Debug)]
enum ObserverType {
    Cdc(ObserveHandle),
    Rts(ObserveHandle),
    Pitr(ObserveHandle),
}

impl ObserverType {
    #[allow(dead_code)]
    fn handle(&self) -> &ObserveHandle {
        match self {
            ObserverType::Cdc(h) => h,
            ObserverType::Rts(h) => h,
            ObserverType::Pitr(h) => h,
        }
    }
}

#[derive(Debug)]
pub struct ChangeObserver {
    #[allow(dead_code)]
    ty: ObserverType,
    #[allow(dead_code)]
    region_id: u64,
}

impl ChangeObserver {
    pub fn from_cdc(region_id: u64, id: ObserveHandle) -> Self {
        Self {
            ty: ObserverType::Cdc(id),
            region_id,
        }
    }

    pub fn from_rts(region_id: u64, id: ObserveHandle) -> Self {
        Self {
            ty: ObserverType::Rts(id),
            region_id,
        }
    }

    pub fn from_pitr(region_id: u64, id: ObserveHandle) -> Self {
        Self {
            ty: ObserverType::Pitr(id),
            region_id,
        }
    }
}
