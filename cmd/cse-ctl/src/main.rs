// Copyright 2022 TiKV Project Authors. Licensed under Apache-2.0.

#[macro_use]
extern crate serde_derive;

mod dfsgc;
mod unsafe_recover;

use std::io;

use clap::{arg, Command};

use crate::{dfsgc::execute_dfsgc, unsafe_recover::execute_unsafe_recover};

fn cli() -> Command<'static> {
    Command::new("cse-ctl")
        .about("command-line tool for cloud storage engine")
        .subcommand_required(true)
        .arg_required_else_help(true)
        .allow_external_subcommands(true)
        .subcommand(
            Command::new("unsafe-recover")
                .about("unsafe recover")
                .args(&[
                    arg!(--path <DIR> "the path of the raft engine").required(true),
                    arg!(--region <ID> "the region id to operate"),
                    arg!(--keyspace <ID> "the APIv2 keyspace id to operate"),
                    arg!(--table <ID> "the table id to operate"),
                    arg!(--all "operate on all regions"),
                    arg!(--destroy "destroy the filtered regions"),
                    arg!(--remove-stores <STORE_IDS> "remove the peers on the stores"),
                    arg!(--commit "commit the change"),
                ])
                .arg_required_else_help(true),
        )
        .subcommand(Command::new("dfsgc").about("gc DFS files").args(&[
            arg!(--config <FILE> "the config file"),
            arg!(--start <STRING> "The start file suffix to GC"),
        ]))
}

fn main() {
    init_logger(io::stdout());
    let matches = cli().get_matches();
    match matches.subcommand() {
        Some(("unsafe-recover", recover_matches)) => {
            execute_unsafe_recover(recover_matches);
        }
        Some(("dfsgc", dfsgc_matches)) => {
            execute_dfsgc(dfsgc_matches);
        }
        _ => unreachable!(),
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
