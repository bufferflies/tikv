// Copyright 2021 TiKV Project Authors. Licensed under Apache-2.0.

#![feature(box_patterns)]
#![feature(let_chains)]
#![recursion_limit = "400"]
#![feature(type_alias_impl_trait)]
#![feature(impl_trait_in_assoc_type)]

#[macro_use(fail_point)]
extern crate fail;
#[macro_use]
extern crate serde_derive;

#[macro_use]
extern crate tikv_util;

#[allow(unused_extern_crates)]
extern crate tikv_alloc;

#[macro_use]
pub mod setup;
pub mod node;
mod raftkv;
pub mod server;
pub mod service;
pub mod signal_handler;
pub use raftkv::*;
mod memory;
mod raft_client;
mod resolve;
pub mod status_server;
mod tikv_server;
mod transport;

pub use status_server::{RestoreShardResponse, StatusServer, TruncateTsConfig};
pub use tikv_server::*;
