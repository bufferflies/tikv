// Copyright 2022 TiKV Project Authors. Licensed under Apache-2.0.

#[macro_use]
extern crate serde_derive;

mod dfsgc;
mod unsafe_recover;

use std::{env, fs::OpenOptions, io};

use clap::{Parser, Subcommand};
use cse_ctl::{
    backup::{execute_backup, BackupArgs},
    restore::{execute_restore_command, RestoreCommand},
    truncate_ts::{execute_truncate_ts, TruncateTsArgs},
};
use slog::Drain;

use crate::{
    dfsgc::{execute_dfsgc, DfsGcArgs},
    unsafe_recover::{execute_unsafe_recover, UnsafeRecoverArgs},
    Commands::{Backup, DfsGc, Restore, TruncateTs, UnsafeRecover},
};

fn main() {
    init_logger();
    let x: Cli = Cli::parse();
    match x.command {
        DfsGc(dfsgc_arg) => {
            execute_dfsgc(dfsgc_arg);
        }
        UnsafeRecover(unsafe_recover) => {
            execute_unsafe_recover(unsafe_recover);
        }
        Backup(backup_args) => {
            execute_backup(backup_args);
        }
        Restore(restore_cmd) => {
            execute_restore_command(restore_cmd);
        }
        TruncateTs(args) => {
            execute_truncate_ts(args);
        }
    }
}

fn init_logger() {
    let output = env::var("LOG_FILE").ok();
    let level = tikv_util::logger::get_level_by_string(
        &env::var("LOG_LEVEL").unwrap_or_else(|_| "info".to_owned()),
    )
    .unwrap();
    let append_instead_truncate = env::var("LOG_APPEND").is_ok();

    match output {
        Some(log_file) => {
            let f = OpenOptions::new()
                .create(true)
                .write(!append_instead_truncate)
                .truncate(!append_instead_truncate)
                .append(append_instead_truncate)
                .open(log_file)
                .unwrap();
            init_logger_impl(f, level);
        }
        None => init_logger_impl(io::stdout(), level),
    };
}

fn init_logger_impl<W: 'static + io::Write + Send>(writer: W, level: slog::Level) {
    let decorator = slog_term::PlainDecorator::new(writer);
    let drain = slog_term::CompactFormat::new(decorator).build();
    let drain = std::sync::Mutex::new(drain).filter_level(level).fuse();
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
    DfsGc(DfsGcArgs),
    /// Unsafely recover the cluster by directly modifying the data on the raft
    /// engine.
    UnsafeRecover(UnsafeRecoverArgs),
    /// Backup backups the cluster.
    Backup(BackupArgs),
    /// Restore a backup.
    Restore(RestoreCommand),
    /// Truncate newer data than given ts
    TruncateTs(TruncateTsArgs),
}
