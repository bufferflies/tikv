// Copyright 2023 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    fmt,
    sync::atomic::{AtomicBool, AtomicI64, AtomicU8, Ordering},
};

use schema::schema::StorageClass;

static NEXT_TABLE_ID: AtomicI64 = AtomicI64::new(1);

#[derive(Default)]
pub struct TableMeta {
    id: i64,
    /// `is_available` indicates whether the table is available for read/write.
    /// It is set to `false` when the table is being "load_data" after
    /// "destroy_table".
    is_available: AtomicBool,
    /// `is_schema_enabled` indicates whether the table is schema awareness.
    /// It is set to `true` when writes table data, mainly used for columnar
    /// test.
    is_schema_enabled: AtomicBool,
    /// `storage_class` indicates the expected storage class of table. Used to
    /// verify consistency with `Shard`.
    storage_class: AtomicU8,
}

impl Clone for TableMeta {
    fn clone(&self) -> Self {
        Self {
            id: self.id,
            is_available: AtomicBool::new(self.is_available()),
            is_schema_enabled: AtomicBool::new(self.is_schema_enabled()),
            storage_class: AtomicU8::new(self.storage_class() as u8),
        }
    }
}

impl fmt::Debug for TableMeta {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TableMeta")
            .field("id", &self.id)
            .field("is_available", &self.is_available())
            .field("is_schema_enabled", &self.is_schema_enabled())
            .field("storage_class", &self.storage_class())
            .finish()
    }
}

impl TableMeta {
    pub fn new(is_available: bool, is_schema_enabled: bool, storage_class: StorageClass) -> Self {
        let id = NEXT_TABLE_ID.fetch_add(1, Ordering::SeqCst);
        Self {
            id,
            is_available: AtomicBool::new(is_available),
            is_schema_enabled: AtomicBool::new(is_schema_enabled),
            storage_class: AtomicU8::new(storage_class as u8),
        }
    }

    pub fn id(&self) -> i64 {
        self.id
    }

    pub fn is_available(&self) -> bool {
        self.is_available.load(Ordering::SeqCst)
    }

    pub fn set_available(&self, is_available: bool) -> bool {
        self.is_available.swap(is_available, Ordering::SeqCst)
    }

    pub fn is_schema_enabled(&self) -> bool {
        self.is_schema_enabled.load(Ordering::SeqCst)
    }

    pub fn storage_class(&self) -> StorageClass {
        self.storage_class
            .load(Ordering::Acquire)
            .try_into()
            .unwrap()
    }

    pub fn set_storage_class(&self, storage_class: StorageClass) {
        self.storage_class
            .store(storage_class as u8, Ordering::Release);
    }
}
