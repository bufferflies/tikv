// Copyright 2025 TiKV Project Authors. Licensed under Apache-2.0.
//
// Branch prediction helpers. Adapted from the hashbrown crate
// (https://github.com/rust-lang/hashbrown), which is dual-licensed under
// Apache-2.0 and MIT. Once `core::hint::{likely, unlikely}` is stable with the
// necessary fixes, we can delete this module and use the std helpers directly.
//
// The toolchain being used is outdated, so we cannot use std intrinsics with
// its bugfixes or crates. So we adapted from hashbrown.
// TODO: after updating the toolchain, use intrinsics from std.

#[inline(always)]
#[cold]
fn cold_path() {}

#[inline(always)]
pub fn likely(b: bool) -> bool {
    if b {
        true
    } else {
        cold_path();
        false
    }
}

#[inline(always)]
pub fn unlikely(b: bool) -> bool {
    if b {
        cold_path();
        true
    } else {
        false
    }
}
