// Copyright 2023 TiKV Project Authors. Licensed under Apache-2.0.

#[macro_use]
extern crate serde_derive;

mod load;
pub mod schema;
mod sync;

pub use load::{load_schema, KvScanner};
pub use sync::{generate_storage_class_schema_data_for_test, sync_schema, KvGetter};
