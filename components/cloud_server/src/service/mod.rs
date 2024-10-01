// Copyright 2021 TiKV Project Authors. Licensed under Apache-2.0.

mod batch;
pub mod diagnostics;
mod kv;

pub use kv::Service as KvService;

pub use self::diagnostics::Service as DiagnosticsService;
