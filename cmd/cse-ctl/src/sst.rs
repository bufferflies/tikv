// Copyright 2023 TiKV Project Authors. Licensed under Apache-2.0.

use std::{ops::Deref, path::PathBuf, sync::Arc};

use clap::Args;
use kvengine::{
    dfs::{DFSConfig, Dfs, Options, S3Fs},
    table::{
        blobtable::blobtable::BlobTable,
        sstable::{InMemFile, L0Table, SsTable},
    },
};

const BLOB_LEVEL_FLAG: u32 = 255;

#[derive(Args)]
pub struct ShowSstArgs {
    /// The path of the config file.
    #[clap(long, default_value = "")]
    pub config: PathBuf,
    /// The id of the SST file.
    #[clap(long)]
    pub id: u64,
    /// The level of the SST file. 255 for BLOB.
    #[clap(long)]
    pub level: u32,
    /// The path of local SST file.
    ///
    /// If specified, the SST file will be get from the path instead of DFS.
    #[clap(long)]
    pub local: Option<PathBuf>,
}

#[derive(Clone, Serialize, Deserialize, PartialEq, Debug, Default)]
#[serde(default)]
#[serde(rename_all = "kebab-case")]
pub struct ShowSstConfig {
    pub dfs: DFSConfig,
}

fn get_file_data_from_local(local: PathBuf) -> bytes::Bytes {
    let data = std::fs::read(&local).unwrap_or_else(|err| {
        panic!("failed to read local file from {:?}: {:?}", local, err);
    });
    bytes::Bytes::from(data)
}

fn get_file_data_from_dfs(id: u64, config: ShowSstConfig) -> bytes::Bytes {
    let s3fs = S3Fs::new(
        config.dfs.prefix,
        config.dfs.s3_endpoint,
        config.dfs.s3_key_id,
        config.dfs.s3_secret_key,
        config.dfs.s3_region,
        config.dfs.s3_bucket,
    );

    let runtime = s3fs.get_runtime();
    runtime
        .block_on(s3fs.read_file(id, Options::new(0, 0)))
        .expect("failed to read file from dfs")
}

pub fn execute_show_sst(args: ShowSstArgs) {
    let mut config = ShowSstConfig::default();
    if args.config.exists() {
        let data = std::fs::read(args.config.clone()).expect("failed to read config file");
        config = toml::from_slice(&data).unwrap();
    }
    config.dfs.override_from_env();

    let data = match args.local {
        Some(local) => get_file_data_from_local(local),
        None => get_file_data_from_dfs(args.id, config),
    };
    let file = Arc::new(InMemFile::new(args.id, data));
    if args.level == 0 {
        let l0 = L0Table::new(file, None, false, None).unwrap().unwrap();
        println!("[SST {}, level {}]", l0.id(), 0);
        println!("  size: {}", l0.size());
        println!("  max_ts: {}", l0.max_ts());
        println!("  entries: {}", l0.entries());
        println!("  tombs: {}", l0.tombs());
        println!("  entries_write_cf: {}", l0.entries_write_cf());
        println!("  kv_size: {}", l0.kv_size());
        println!("  version: {}", l0.version());
        println!(
            "  smallest: {}",
            log_wrappers::hex_encode_upper(l0.smallest().deref())
        );
        println!(
            "  biggest: {}",
            log_wrappers::hex_encode_upper(l0.biggest().deref())
        );
        println!("  total_blob_size: {}", l0.total_blob_size());

        for cf in 0..kvengine::NUM_CFS {
            println!("  [CF {}]", cf);
            if let Some(tbl) = l0.get_cf(cf) {
                print_sstable(tbl, 4);
            } else {
                println!("    None");
            }
        }
    } else if args.level == BLOB_LEVEL_FLAG {
        let blob = BlobTable::new(file).unwrap();
        println!("[BLOB {}]", blob.id());
        println!("  version: {}", blob.version());
        println!("  size: {}", blob.size());
        println!(
            "  smallest: {}",
            log_wrappers::hex_encode_upper(blob.smallest_key().deref())
        );
        println!(
            "  biggest: {}",
            log_wrappers::hex_encode_upper(blob.biggest_key().deref())
        );
        println!("  total_blob_size: {}", blob.total_blob_size());
        println!("  compression_tp: {}", blob.compression_tp());
        println!("  compression_lvl: {}", blob.compression_lvl());
        println!("  min_blob_size: {}", blob.min_blob_size());
    } else {
        let ln = SsTable::new(file, None, false, None).unwrap();
        println!("[SST {}, level {}]", ln.id(), args.level);
        print_sstable(&ln, 2);
    }
}

fn print_sstable(tbl: &SsTable, indent: usize) {
    let indent = " ".repeat(indent);
    println!("{}size: {}", indent, tbl.size());
    println!("{}index_size: {}", indent, tbl.index_size());
    println!("{}filter_size: {}", indent, tbl.filter_size());
    println!("{}max_ts: {}", indent, tbl.max_ts);
    println!("{}entries: {}", indent, tbl.entries);
    println!("{}old_entries: {}", indent, tbl.old_entries);
    println!("{}tombs: {}", indent, tbl.tombs);
    println!("{}kv_size: {}", indent, tbl.kv_size);
    println!(
        "{}smallest: {}",
        indent,
        log_wrappers::hex_encode_upper(tbl.smallest().deref())
    );
    println!(
        "{}biggest: {}",
        indent,
        log_wrappers::hex_encode_upper(tbl.biggest().deref())
    );
    println!("{}compression_type: {}", indent, tbl.compression_type());
    println!("{}total_blob_size: {}", indent, tbl.total_blob_size());
    println!("{}encryption_ver: {}", indent, tbl.encryption_ver());

    let idx = tbl.load_index();
    println!("{}num_blocks: {}", indent, idx.num_blocks());
}
