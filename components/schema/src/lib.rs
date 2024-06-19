// Copyright 2023 TiKV Project Authors. Licensed under Apache-2.0.

#[macro_use]
extern crate serde_derive;

mod load;
pub mod schema;
mod sync;

pub use load::{load_schema, KvScanner};
pub use sync::{sync_schema, KvGetter};
