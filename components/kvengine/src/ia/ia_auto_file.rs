// Copyright 2025 TiKV Project Authors. Licensed under Apache-2.0.

use std::sync::Arc;

use async_trait::async_trait;
use bytes::Bytes;
use parking_lot::Mutex;
use schema::schema::{StorageClass, StorageClassSpec};
use txn_types::TimeStamp;

use crate::{
    ia::ia_file::IaFile,
    metrics::ENGINE_STORAGE_CLASS_TRANSITION_COUNTER,
    table::{
        file::{File, MmapData},
        Result,
    },
};

pub struct IaAutoFile {
    pub(crate) ia_file: IaFile,
    pub(crate) local_file: Mutex<Option<Arc<dyn File>>>,
    max_ts: u64,
    spec: StorageClassSpec,
}

impl IaAutoFile {
    pub fn new(
        ia_file: IaFile,
        local_file: Option<Arc<dyn File>>,
        max_ts: u64,
        spec: StorageClassSpec,
    ) -> Self {
        Self {
            ia_file,
            local_file: Mutex::new(local_file),
            max_ts,
            spec,
        }
    }

    fn local_file(&self) -> Option<Arc<dyn File>> {
        self.local_file.lock().clone()
    }

    pub(crate) fn has_local_file(&self) -> bool {
        self.local_file.lock().is_some()
    }

    pub(crate) fn transit_to_ia(&self) -> TransitResult {
        if self.local_file.lock().take().is_some() {
            info!("{} transit to IA", self.id());
            TransitResult::TransitedToIa
        } else {
            TransitResult::NoChange
        }
    }

    fn target_storage_class(&self, current_ts: u64) -> StorageClass {
        let elapsed = TimeStamp::new(current_ts).duration_since(self.max_ts.into());
        self.spec.target_storage_class(elapsed)
    }

    fn should_transit(&self, current_ts: u64) -> Option<StorageClass> {
        let target = self.target_storage_class(current_ts);
        let curr = self.storage_class();
        (target != curr).then_some(target)
    }

    pub fn try_transit(&self, current_ts: u64) -> TransitResult {
        let Some(target) = self.should_transit(current_ts) else {
            return TransitResult::NoChange;
        };

        if target.is_ia() {
            self.transit_to_ia()
        } else {
            if !self.has_local_file() {
                warn!("{}: can not transit to local", self.id());
            }
            TransitResult::NoChange
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransitResult {
    TransitedToIa,
    NoChange,
}

#[async_trait]
impl File for IaAutoFile {
    fn id(&self) -> u64 {
        self.ia_file.id()
    }

    fn size(&self) -> u64 {
        self.ia_file.size()
    }

    fn is_sync(&self) -> bool {
        false
    }

    fn read(&self, off: u64, length: usize) -> Result<Bytes> {
        if let Some(local_file) = self.local_file() {
            local_file.read(off, length)
        } else {
            self.ia_file.read(off, length)
        }
    }

    fn read_at(&self, buf: &mut [u8], offset: u64) -> Result<()> {
        if let Some(local_file) = self.local_file() {
            local_file.read_at(buf, offset)
        } else {
            self.ia_file.read_at(buf, offset)
        }
    }

    fn read_table_meta(&self, off: u64, length: usize) -> Result<Bytes> {
        self.ia_file.read_table_meta(off, length)
    }

    async fn read_async(&self, off: u64, length: usize) -> Result<Bytes> {
        if let Some(local_file) = self.local_file() {
            local_file.read(off, length)
        } else {
            self.ia_file.read_async(off, length).await
        }
    }

    async fn read_at_async(&self, buf: &mut [u8], offset: u64) -> Result<()> {
        if let Some(local_file) = self.local_file() {
            local_file.read_at(buf, offset)
        } else {
            self.ia_file.read_at_async(buf, offset).await
        }
    }

    fn mmap(&self) -> Result<MmapData> {
        unimplemented!()
    }

    async fn mmap_async(&self) -> Result<MmapData> {
        self.ia_file.mmap_async().await
    }

    fn storage_class(&self) -> StorageClass {
        if self.local_file().is_some() {
            StorageClass::Unspecified
        } else {
            StorageClass::Ia
        }
    }

    fn as_any(self: Arc<Self>) -> Arc<dyn std::any::Any + Send + Sync> {
        self
    }
}

pub fn report_transitions(transited_to_ia: usize) {
    if transited_to_ia > 0 {
        ENGINE_STORAGE_CLASS_TRANSITION_COUNTER
            .with_label_values(&["to_ia"])
            .inc_by(transited_to_ia as u64);
    }
}
