// Copyright 2024 TiKV Project Authors. Licensed under Apache-2.0.

//! Infrequent Access File & Relevant Components.
//!
//! Used for the scene where the data is infrequently accessed. The files are
//! not always kept in local disk/memory, but used some limited capacity through
//! the S3-FIFO algorithm to improve the efficiency of the capacity.
//!
//! If the file is not exist in the local disk/memory, it will be downloaded
//! from the remote storage.
//!
//! It's a tradeoff between the latency and the cost of local disk/memory.
//!
//! See https://github.com/tidbcloud/cloud-storage-engine/issues/1710 for more details.

pub mod ia_file;
pub mod manager;
mod queue;
pub mod types;
pub mod util;
pub use util::{IA_FREQ_UPDATE_INTERVAL_DEF, IA_SEGMENT_SIZE_DEF};
