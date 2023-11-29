// Copyright 2023 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    fmt,
    sync::atomic::{AtomicBool, AtomicI64, Ordering},
};

static NEXT_TABLE_ID: AtomicI64 = AtomicI64::new(1);

#[derive(Default)]
pub struct TableMeta {
    id: i64,
    /// `is_available` indicates whether the table is available for read/write.
    /// It is set to `false` when the table is being "load_data" after
    /// "destroy_table".
    is_available: AtomicBool,
}

impl Clone for TableMeta {
    fn clone(&self) -> Self {
        Self {
            id: self.id,
            is_available: AtomicBool::new(self.is_available()),
        }
    }
}

impl fmt::Debug for TableMeta {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TableMeta")
            .field("id", &self.id)
            .field("is_available", &self.is_available())
            .finish()
    }
}

impl TableMeta {
    pub fn new(is_available: bool) -> Self {
        let id = NEXT_TABLE_ID.fetch_add(1, Ordering::SeqCst);
        Self {
            id,
            is_available: AtomicBool::new(is_available),
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
}
