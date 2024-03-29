// Copyright 2023 TiKV Project Authors. Licensed under Apache-2.0.

//! Handle transactional locks.
//! Port from https://github.com/tikv/client-go/tree/master/txnkv/txnlock.

pub(crate) mod lock_resolver;
pub mod txn_file;
