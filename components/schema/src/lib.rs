// Copyright 2023 TiKV Project Authors. Licensed under Apache-2.0.

#[macro_use]
extern crate serde_derive;

mod load;
pub mod schema;
mod sync;

pub use load::{load_schema, KvScanner};
#[cfg(feature = "testexport")]
pub use sync::test_utils::generate_storage_class_schema_data_for_test;
pub use sync::{get_schema_version, sync_schema, KvGetter};
