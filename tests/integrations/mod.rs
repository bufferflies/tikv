// Copyright 2016 TiKV Project Authors. Licensed under Apache-2.0.

#![feature(test)]
#![feature(box_patterns)]
#![feature(custom_test_frameworks)]
#![feature(assert_matches)]
#![test_runner(test_util::run_tests)]

#[macro_use]
extern crate tikv_util;

mod config;
mod coprocessor;
mod import;
mod pd;
mod resource_metering;
mod server_encryption;
