// Copyright 2018 TiKV Project Authors. Licensed under Apache-2.0.

//! Generally peers are state machines that represent a replica of a region,
//! and store is also a special state machine that handles all requests across
//! stores. They are mixed for now, will be separated in the future.

pub mod apply;
pub mod metrics;

pub use self::apply::ChangeObserver;
