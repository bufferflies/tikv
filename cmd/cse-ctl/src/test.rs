// Copyright 2024 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    collections::HashSet,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};

use clap::{Args, Subcommand};
use engine_traits::ObjectStorage;
use kvengine::dfs::{DFSConfig, Dfs, S3Fs};
use tikv_util::{error, info, warn};

#[derive(Args)]
pub struct TestArgs {
    #[clap(subcommand)]
    command: TestCommands,
}

#[derive(Subcommand)]
enum TestCommands {
    BurnCpus(BurnCpusArgs),
    S3Permission(S3PermissionArgs),
}

#[derive(Args)]
pub struct BurnCpusArgs {
    /// The number of CPUs to burn.
    #[clap(long, default_value_t = 1)]
    pub cpus: usize,
    /// The burn duration in seconds.
    #[clap(long, default_value_t = 1)]
    pub seconds: usize,
}

#[derive(Args)]
pub struct S3PermissionArgs {
    /// The duration in seconds.
    #[clap(long, default_value_t = 0)]
    pub seconds: usize,

    /// The path of the config file.
    #[clap(long, default_value = "")]
    pub prefix: String,
    /// The path of the config file.
    #[clap(long)]
    pub read_only: bool,
}

#[derive(Clone, Serialize, Deserialize, PartialEq, Debug, Default)]
#[serde(default)]
#[serde(rename_all = "kebab-case")]
pub struct S3Config {
    pub dfs: DFSConfig,
}

pub fn execute_test(args: TestArgs) {
    match args.command {
        TestCommands::BurnCpus(args) => {
            execute_burn_cpus(args);
        }
        TestCommands::S3Permission(args) => {
            execute_s3_read_write(args);
        }
    }
}

fn execute_s3_read_write(args: S3PermissionArgs) {
    let mut config = S3Config::default();
    config.dfs.override_from_env();
    let s3fs = Arc::new(S3Fs::new(
        config.dfs.prefix.clone(),
        config.dfs.s3_endpoint,
        config.dfs.s3_key_id,
        config.dfs.s3_secret_key,
        config.dfs.s3_region,
        config.dfs.s3_bucket,
    ));

    let prefix = if config.dfs.prefix.is_empty() {
        args.prefix.clone()
    } else {
        config.dfs.prefix
    };

    if !args.prefix.is_empty() && prefix != args.prefix {
        error!("args prefix and env prefix are not match");
        return;
    }

    let start_time = Instant::now();
    let mut counter = 0;
    let runtime = s3fs.get_runtime();
    let mut start_after = String::new();
    let mut to_be_cleanup = HashSet::new();

    loop {
        // Stop condition
        if args.seconds != 0 && start_time.elapsed().as_secs() >= args.seconds as u64 {
            info!("Test duration reached: {} seconds", args.seconds);
            break;
        }

        if args.read_only {
            info!("Start to list objects from {}", start_after);
            let (files, _has_more) = match s3fs.list_objects(start_after.as_str(), None, None) {
                Ok(result) => result,
                Err(err) => {
                    error!("Failed to list objects: {:?}", err);
                    break;
                }
            };

            info!("Listed {} files", files.len());
            if files.is_empty() {
                info!("Finished listing files, start it over");
                start_after = String::new();
                std::thread::sleep(Duration::from_secs(3)); // Sleep to avoid tight loop or API throttling
                continue;
            }

            for file in &files {
                let key = &file.key;
                if let Err(err) = runtime.block_on(s3fs.get_object(
                    key.clone(),
                    key.clone(),
                    engine_traits::GetObjectOptions::default(),
                )) {
                    error!("Failed to get object {}: {:?}", key, err);
                }
            }
            start_after = files.last().unwrap().key.clone();
        } else {
            // READ-WRITE MODE
            if counter > 10 {
                counter = 0
            }
            let key = format!("{}/test_{}.txt", prefix, counter);
            let content = bytes::Bytes::from(format!("test content #{}", counter));

            match runtime.block_on(s3fs.put_object(key.clone(), content.clone(), key.clone())) {
                Ok(_) => {
                    info!("Uploaded object: {}", key);
                    to_be_cleanup.insert(key.clone());
                }

                Err(err) => {
                    error!("Failed to upload object: {:?}", err);
                    break;
                }
            }

            // Optional read-back verification
            match runtime.block_on(s3fs.get_object(
                key.clone(),
                key.clone(),
                engine_traits::GetObjectOptions::default(),
            )) {
                Ok(data) => {
                    if data != content {
                        warn!("Read content mismatch for key {}", key);
                    } else {
                        info!("Verified uploaded object: {}", key);
                    }
                }
                Err(err) => warn!("Failed to verify object: {:?}", err),
            }
        }

        counter += 1;
        std::thread::sleep(Duration::from_secs(3)); // Sleep to avoid tight loop or API throttling
    }

    if !to_be_cleanup.is_empty() {
        for key in to_be_cleanup {
            match runtime.block_on(s3fs.delete_object(key.clone(), key.clone())) {
                Ok(()) => info!("cleanup uploaded object: {}", key),
                Err(err) => warn!("Failed to verify object: {:?}", err),
            }
        }
    }

    info!(
        "S3 permission test completed after {} seconds",
        start_time.elapsed().as_secs()
    );
}

// TODO: support number of cpus in decimal.
fn execute_burn_cpus(args: BurnCpusArgs) {
    let run = Arc::new(AtomicBool::new(true));
    let mut handles = Vec::with_capacity(args.cpus);
    for _ in 0..args.cpus {
        let run = run.clone();
        let h = std::thread::spawn(move || {
            let mut count = 0u64;
            while run.load(Ordering::Relaxed) {
                for _ in 0..u16::MAX {
                    count = count.wrapping_add(1u64);
                }
            }
            count
        });
        handles.push(h);
    }

    std::thread::sleep(Duration::from_secs(args.seconds as u64));
    run.store(false, Ordering::Relaxed);

    let total: u64 = handles.into_iter().map(|h| h.join().unwrap()).sum();
    let avg = total as f64 / args.seconds as f64 / args.cpus as f64;
    println!("counter: {total}, avg (per cpu x second): {avg}");
}
