// Copyright 2022 TiKV Project Authors. Licensed under Apache-2.0.

#[macro_use]
extern crate serde_derive;

mod backup;
mod common;
mod dfsgc;
mod restore;
mod truncate_ts;
mod unsafe_recover;

use std::io;

use clap::{Parser, Subcommand};

use crate::{
    backup::{execute_backup, BackupArgs},
    dfsgc::{execute_dfsgc, DFSGCArgs},
    restore::{execute_restore, RestoreArgs},
    truncate_ts::{execute_truncate_ts, TruncateTsArgs},
    unsafe_recover::{execute_unsafe_recover, UnsafeRecoverArgs},
    Commands::{Backup, Restore, TruncateTs, UnsafeRecover, DFSGC},
};

fn main() {
    init_logger(io::stdout());
    let x: Cli = Cli::parse();
    match x.command {
        DFSGC(dfsgc_arg) => {
            execute_dfsgc(dfsgc_arg);
        }
        UnsafeRecover(unsafe_recover) => {
            execute_unsafe_recover(unsafe_recover);
        }
        Backup(backup_args) => {
            execute_backup(backup_args);
        }
        Restore(restore_args) => {
            execute_restore(restore_args);
        }
        TruncateTs(args) => {
            execute_truncate_ts(args);
        }
    }
}

fn init_logger<W: 'static + io::Write + Send>(writer: W) {
    use slog::Drain;
    let decorator = slog_term::PlainDecorator::new(writer);
    let drain = slog_term::CompactFormat::new(decorator).build();
    let drain = std::sync::Mutex::new(drain).fuse();
    let logger = slog::Logger::root(drain, slog::o!());
    slog_global::set_global(logger);
}

#[derive(Parser)]
#[clap(author, version, about, long_about = None)]
pub struct Cli {
    #[clap(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Scan and mark the unused DFS files as deleted.
    DFSGC(DFSGCArgs),
    /// Unsafely recover the cluster by directly modifying the data on the raft engine.
    UnsafeRecover(UnsafeRecoverArgs),
    /// Backup backups the cluster.
    Backup(BackupArgs),
    /// Restore a backup.
    Restore(RestoreArgs),
    /// Truncate newer data than given ts
    TruncateTs(TruncateTsArgs),
}
