// Copyright 2023 TiKV Project Authors. Licensed under Apache-2.0.

use native_br::{
    packing::ReportPackBackupStepTrait,
    restore_keyspace::{ReportRestoreStepTrait, RestoreStep},
};
use rand::Rng;

mod backup;
mod limiter;
mod restore_keyspace;

fn random_value<const N: usize>(_: usize) -> Vec<u8>
where
    [u8; N]: rand::Fill,
{
    let mut bytes = [0u8; N];
    rand::thread_rng().fill(&mut bytes);
    bytes.to_vec()
}

#[derive(Default)]
struct DummyStepReporter {}

impl ReportRestoreStepTrait for DummyStepReporter {
    fn report_step(&self, _step: RestoreStep) {}
}

impl ReportPackBackupStepTrait for DummyStepReporter {
    fn report_step(&self, _step: native_br::packing::PackBackupStep) {}
}
