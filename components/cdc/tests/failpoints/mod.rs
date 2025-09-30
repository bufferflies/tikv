// Copyright 2020 TiKV Project Authors. Licensed under Apache-2.0.

#![feature(custom_test_frameworks)]
#![test_runner(test_util::run_failpoint_tests)]

mod test_endpoint;
#[cfg(NEXT_GEN_COMPATIBLE_TODO)]
mod test_memory_quota;
#[cfg(NEXT_GEN_COMPATIBLE_TODO)]
mod test_observe;
#[cfg(NEXT_GEN_COMPATIBLE_TODO)]
mod test_register;
#[cfg(NEXT_GEN_COMPATIBLE_TODO)]
mod test_resolve;

#[path = "../mod.rs"]
mod testsuite;
pub use testsuite::*;
