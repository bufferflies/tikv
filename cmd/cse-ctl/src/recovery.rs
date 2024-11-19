// Copyright 2024 TiKV Project Authors. Licensed under Apache-2.0.

use std::path::PathBuf;

use clap::{Args, Subcommand};
use kvproto::metapb::Store;

use crate::{
    common::{CommonConfig, CtlContext},
    http::{build_uri, send_request},
};

#[derive(Args)]
pub struct RecoveryArgs {
    #[clap(subcommand)]
    pub command: RecoveryCommands,
    /// The path of the config file.
    #[clap(long, default_value = "")]
    pub config: PathBuf,
    /// PD endpoints, use `,` to separate multiple PDs
    #[clap(long, default_value_t = String::new())]
    pub pd: String,
    /// Exclude stores to execute the command.
    #[clap(long, default_value_t = String::new())]
    pub exclude_stores: String,
    /// Keyspace IDs to execute the command.
    #[clap(long, default_value_t = String::new())]
    pub keyspace_ids: String,
}

#[derive(Subcommand)]
pub enum RecoveryCommands {
    Status,
    Enable,
    Disable,
    AddWhiteList,
    AddBlackList,
    RemoveBlackList,
}

pub fn execute_recovery(args: RecoveryArgs) {
    let config = CommonConfig::from_args(args.config.as_path(), &args.pd);
    let ctx = config.create_context();
    let stores = ctx.get_tikv_stores(&args.exclude_stores);
    let keyspace_ids = &args.keyspace_ids;
    match args.command {
        RecoveryCommands::Status => {
            execute_status(&ctx, &stores);
        }
        RecoveryCommands::Enable => {
            execute_mode(&ctx, &stores, true);
        }
        RecoveryCommands::Disable => {
            execute_mode(&ctx, &stores, false);
        }
        RecoveryCommands::AddWhiteList => {
            execute_add_white_list(&ctx, &stores, keyspace_ids);
        }
        RecoveryCommands::AddBlackList => {
            execute_black_list(&ctx, &stores, keyspace_ids, false);
        }
        RecoveryCommands::RemoveBlackList => {
            execute_black_list(&ctx, &stores, keyspace_ids, true);
        }
    }
}

fn execute_status(ctx: &CtlContext, stores: &[Store]) {
    for store in stores {
        println!("store: {}", store.id);
        let uri = build_uri(ctx, store, "recovery/mode");
        let (status, body) = send_request(ctx, false, uri, "");
        if status.is_success() {
            println!("in recovery mode:{}", String::from_utf8_lossy(&body));
        } else {
            println!("failed: {}, {}", status, String::from_utf8_lossy(&body));
        }
        let uri = build_uri(ctx, store, "recovery/rejected");
        let (status, body) = send_request(ctx, false, uri, "");
        if status.is_success() {
            println!("rejected:{}", String::from_utf8_lossy(&body));
        } else {
            println!("failed: {}, {}", status, String::from_utf8_lossy(&body));
        }
        let uri = build_uri(ctx, store, "recovery/white_list");
        let (status, body) = send_request(ctx, false, uri, "");
        if status.is_success() {
            println!("white_list:{}", String::from_utf8_lossy(&body));
        } else {
            println!("failed: {}, {}", status, String::from_utf8_lossy(&body));
        }
        let uri = build_uri(ctx, store, "recovery/black_list");
        let (status, body) = send_request(ctx, false, uri, "");
        if status.is_success() {
            println!("black_list:{}", String::from_utf8_lossy(&body));
        } else {
            println!("failed: {}, {}", status, String::from_utf8_lossy(&body));
        }
    }
}

fn execute_mode(ctx: &CtlContext, stores: &[Store], enable: bool) {
    for store in stores {
        println!("store: {}", store.id);
        let uri = build_uri(ctx, store, &format!("recovery/mode?enable={}", enable));
        let (status, body) = send_request(ctx, true, uri, "");
        let str_body = String::from_utf8_lossy(&body);
        if status.is_success() {
            println!("mode enabled: {}", str_body);
        } else {
            println!("failed: {}, {}", status, str_body);
        }
    }
}

fn execute_add_white_list(ctx: &CtlContext, stores: &[Store], keyspace_ids: &str) {
    for store in stores {
        println!("store: {}", store.id);
        let uri = build_uri(
            ctx,
            store,
            &format!("recovery/white_list?keyspace_ids={}", keyspace_ids),
        );
        let (status, body) = send_request(ctx, true, uri, "");
        let str_body = String::from_utf8_lossy(&body);
        if status.is_success() {
            println!("new white_list: {}", str_body);
        } else {
            println!("failed: {}, {}", status, str_body);
        }
    }
}

fn execute_black_list(ctx: &CtlContext, stores: &[Store], keyspace_ids: &str, remove: bool) {
    for store in stores {
        println!("store: {}", store.id);
        let uri = build_uri(
            ctx,
            store,
            &format!(
                "recovery/black_list?keyspace_ids={}&remove={}",
                keyspace_ids, remove
            ),
        );
        let (status, body) = send_request(ctx, true, uri, "");
        let str_body = String::from_utf8_lossy(&body);
        if status.is_success() {
            println!("new black_list: {}", str_body);
        } else {
            println!("failed: {}, {}", status, str_body);
        }
    }
}
