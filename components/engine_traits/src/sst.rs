// Copyright 2019 TiKV Project Authors. Licensed under Apache-2.0.

use kvproto::import_sstpb::SstMeta;

#[derive(Clone, Debug)]
pub struct SstMetaInfo {
    pub total_bytes: u64,
    pub total_kvs: u64,
    pub meta: SstMeta,
}

// compression type used for write sst file
#[derive(Copy, Clone)]
pub enum SstCompressionType {
    Lz4,
    Snappy,
    Zstd,
}
