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
    keyspace_id: u32,
    db_name: String,
    table_name: String,
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
            keyspace_id: self.keyspace_id,
            db_name: self.db_name.clone(),
            table_name: self.table_name.clone(),
            is_available: AtomicBool::new(self.is_available()),
            is_schema_enabled: AtomicBool::new(self.is_schema_enabled()),
            storage_class: AtomicU8::new(self.storage_class() as u8),
        }
    }
}

impl fmt::Debug for TableMeta {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut de = f.debug_struct("TableMeta");
        de.field("id", &self.id)
            .field("keyspace", &self.keyspace_id);
        if !self.db_name.is_empty() {
            de.field("db_name", &self.db_name);
        }
        if !self.table_name.is_empty() {
            de.field("table_name", &self.table_name);
        }
        de.field("is_available", &self.is_available())
            .field("is_schema_enabled", &self.is_schema_enabled())
            .field("storage_class", &self.storage_class())
            .finish()
    }
}

impl TableMeta {
    pub fn new(
        keyspace_id: u32,
        is_available: bool,
        is_schema_enabled: bool,
        storage_class: StorageClass,
    ) -> Self {
        let id = NEXT_TABLE_ID.fetch_add(1, Ordering::SeqCst);
        Self {
            id,
            keyspace_id,
            db_name: String::new(),
            table_name: String::new(),
            is_available: AtomicBool::new(is_available),
            is_schema_enabled: AtomicBool::new(is_schema_enabled),
            storage_class: AtomicU8::new(storage_class as u8),
        }
    }

    pub fn new_tidb_table(
        table_id: i64,
        keyspace_id: u32,
        db_name: String,
        table_name: String,
        storage_class: StorageClass,
    ) -> Self {
        Self {
            id: table_id,
            keyspace_id,
            db_name,
            table_name,
            is_available: AtomicBool::new(true),
            is_schema_enabled: AtomicBool::new(true),
            storage_class: AtomicU8::new(storage_class as u8),
        }
    }

    pub fn id(&self) -> i64 {
        self.id
    }

    pub fn keyspace_id(&self) -> u32 {
        self.keyspace_id
    }

    pub fn db_name(&self) -> &str {
        &self.db_name
    }

    pub fn table_name(&self) -> &str {
        &self.table_name
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
