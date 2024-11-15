// Copyright 2021 TiKV Project Authors. Licensed under Apache-2.0.

#![cfg_attr(test, feature(test))]
// Bytes as map key
#![allow(clippy::mutable_key_type)]

#[cfg(test)]
extern crate test;
#[allow(unused_extern_crates)]
extern crate tikv_alloc;
#[macro_use]
extern crate serde_derive;

mod config;
pub use config::Config as RfEngineConfig;

pub mod dfs_worker;
pub mod engine;
pub mod iterator;
pub mod load;
mod log_batch;
pub mod manifest;
mod metrics;
pub mod traits;
pub mod utils;
pub mod worker;
mod write_batch;
pub mod writer;

use std::num::ParseIntError;

pub use dfs_worker::*;
pub use engine::*;
use iterator::*;
use metrics::*;
use thiserror::Error as ThisError;
pub use traits::*;
pub use utils::*;
pub use worker::*;
pub use write_batch::WriteBatch;
pub use writer::*;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, ThisError)]
pub enum Error {
    #[error("IO error: {0:?}")]
    Io(std::io::Error),
    #[error("EOF")]
    Eof,
    #[error("parse error")]
    ParseError,
    #[error("Open error: {0}")]
    Open(String),
    #[error("Corruption: {msg}, epoch_id {epoch_id}, offset {offset}")]
    Corruption {
        msg: String,
        epoch_id: u32,
        offset: u64,
        data: Vec<u8>,
    },
    #[error("WAL Epoch {epoch_id} is overwritten")]
    WalEpochOverwritten { epoch_id: u32 },
    #[error("Other error: {0}")]
    Other(String),
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        if e.kind() == std::io::ErrorKind::UnexpectedEof {
            return Error::Eof;
        }
        Error::Io(e)
    }
}

impl From<ParseIntError> for Error {
    fn from(_: ParseIntError) -> Self {
        Error::ParseError
    }
}

impl From<String> for Error {
    fn from(msg: String) -> Self {
        Error::Other(msg)
    }
}
