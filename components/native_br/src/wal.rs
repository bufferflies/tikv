// Copyright 2025 TiKV Project Authors. Licensed under Apache-2.0.

use bytes::Bytes;

#[derive(Clone)]
pub struct WalOnlineChunk {
    pub data: Bytes,
    pub start_off: u64, // Handle the case when WAL chunks is empty.
}

impl WalOnlineChunk {
    pub fn end_off(&self) -> u64 {
        self.start_off + self.data.len() as u64
    }
}
