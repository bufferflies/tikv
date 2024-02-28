// Copyright 2022 TiKV Project Authors. Licensed under Apache-2.0.

#[macro_use]
extern crate serde_derive;

mod archive;
mod backup;
mod check_table;
mod dfsgc;
mod restore;
mod sst;
mod stats;
mod truncate_ts;
mod unsafe_recover;

use std::{env, fs::OpenOptions, io};

use backup::{execute_show_backup_list, ShowBackupListArgs};
use clap::{Args, Parser, Subcommand};
use slog::Drain;

use crate::{
    archive::{execute_archive, ArchiveArgs},
    backup::{execute_backup, execute_show_backup, BackupArgs, ShowBackupArgs},
    check_table::{execute_check_table, CheckTableArgs},
    dfsgc::{execute_dfsgc, DfsGcArgs},
    restore::{execute_restore_command, RestoreCommand},
    sst::{execute_show_sst, ShowSstArgs},
    stats::{execute_stats, StatsArgs},
    truncate_ts::{execute_truncate_ts, TruncateTsArgs},
    unsafe_recover::{execute_unsafe_recover, UnsafeRecoverArgs},
    Commands::*,
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
        Archive(archive_args) => {
            execute_archive(archive_args);
        }
        Stats(stats_arg) => {
            execute_stats(stats_arg);
        }
        TruncateTs(args) => {
            execute_truncate_ts(args);
        }
        CheckTable(args) => {
            execute_check_table(args);
        }
        Show(args) => {
            execute_show(args);
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
    /// Archive old backups.
    Archive(ArchiveArgs),
    /// Stats s3 objects.
    Stats(StatsArgs),
    /// Truncate newer data than given ts
    TruncateTs(TruncateTsArgs),
    /// CheckTable check data consistency on each table.
    CheckTable(CheckTableArgs),
    /// Show some information.
    Show(ShowArgs),
}

#[derive(Args)]
pub struct ShowArgs {
    #[clap(subcommand)]
    command: ShowCommands,
}

#[derive(Subcommand)]
enum ShowCommands {
    /// Show the backup meta data.
    Backup(ShowBackupArgs),
    BackupList(ShowBackupListArgs),
    Sst(ShowSstArgs),
}

fn execute_show(args: ShowArgs) {
    match args.command {
        ShowCommands::Backup(args) => {
            execute_show_backup(args);
        }
        ShowCommands::BackupList(args) => {
            execute_show_backup_list(args);
        }
        ShowCommands::Sst(args) => {
            execute_show_sst(args);
        }
    }
}
