// Copyright 2024 TiKV Project Authors. Licensed under Apache-2.0.

use std::{sync::atomic::Ordering, time::Duration};

use rand::Rng;
use test_cloud_server::{
    client::{ClusterClient, CommitAction, MutateOptions, RequestOptions, TxnWriteMethod},
    keyspace,
    keyspace::KeyspaceManager,
};
use tikv_util::{debug, info, time::Instant};

use crate::{generate_random_string, i_to_key, TXN_FILE_WRITE_COUNTER};

pub(crate) const TXN_CHUNK_MAX_SIZE: usize = 1024;

pub(crate) fn spawn_txn_file_write(
    begin_idx: usize,
    thread_idx: usize,
    mut client: ClusterClient,
    keyspace_manager: KeyspaceManager,
    timeout: Duration,
) -> std::thread::JoinHandle<()> {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(1)
        .max_blocking_threads(1)
        .enable_all()
        .build()
        .unwrap();
    std::thread::spawn(move || {
        let _enter = rt.enter(); // For tikv worker client.

        // Key generation (take begin_idx=4 as example):
        // First, randomly pick i from [8000..8190), take i=8100 as example.
        // Then:
        // * Keys (idx=0): <keyspace>t<table>_rxkey[0000810000, 0000810100, ...,
        //   0000810900]
        // * Keys (idx=1): <keyspace>t<table>_rxkey[0000810001, 0000810101, ...,
        //   0000810901]
        let begin = begin_idx * 2000;
        let end = begin + 50 - 10;

        let mut rng = rand::thread_rng();
        let start_time = Instant::now();
        while start_time.saturating_elapsed() < timeout {
            let keyspace_id = keyspace_manager.get_zipf_random_keyspace(&mut rng);
            {
                let lock = keyspace_manager.get_keyspace_lock(keyspace_id);
                let _guard = rt.block_on(lock.shared_lock());

                let table_id =
                    match keyspace_manager.get_random_available_table(keyspace_id, &mut rng) {
                        Some(table_id) => table_id,
                        None => continue,
                    };
                let i = rng.gen_range(begin..end);
                let put_kv = rng.gen_ratio(2, 3);

                let ver = client.get_ts().into_inner();
                info!(
                    "[{}] thread txn file write on keyspace {}, ver {}",
                    thread_idx, keyspace_id, ver
                );

                let gen_key = |i| {
                    let mut user_key = i_to_key(i);
                    user_key.extend_from_slice(format!("{:02}", thread_idx).as_bytes());
                    keyspace::make_key(keyspace_id, table_id, &user_key)
                };
                let ref_store = keyspace_manager
                    .ref_stores()
                    .get_keyspace_ref_store(keyspace_id);
                client.replace_ref_store(ref_store.clone());
                let put_time = Instant::now();
                if put_kv {
                    client
                        .try_put_kv(
                            i..(i + 10),
                            gen_key,
                            generate_random_string(format!("txnf-{}-", ver)),
                            MutateOptions {
                                commit_action: CommitAction::AsyncCommitSecondaryKeys(
                                    Duration::ZERO,
                                ),
                                write_method: TxnWriteMethod::FileBased,
                            },
                        )
                        .unwrap();
                } else {
                    client
                        .try_del_kv(
                            i..(i + 10),
                            gen_key,
                            MutateOptions {
                                commit_action: CommitAction::AsyncCommitSecondaryKeys(
                                    Duration::ZERO,
                                ),
                                write_method: TxnWriteMethod::FileBased,
                            },
                        )
                        .unwrap();
                };

                debug!(
                    "[{}] txn file write keys {:?}",
                    thread_idx,
                    (i..(i + 10))
                        .map(|j| tikv_util::escape(&gen_key(j)))
                        .collect::<Vec<_>>()
                );

                let req_opts = RequestOptions::default();
                let ref_store = ref_store.lock().unwrap().clone();
                for j in i..(i + 10) {
                    let key = gen_key(j);
                    let empty: Option<Vec<u8>> = None;
                    let expect_val = ref_store.get(&key).unwrap_or(&empty);
                    client
                        .verify_key_value(&key, expect_val.as_ref(), put_time, &req_opts)
                        .unwrap_or_else(|err| {
                            panic!(
                                "[{}] verify_key_value failed (after txn-file-write): {:?}",
                                thread_idx, err
                            );
                        });
                }
            }
            TXN_FILE_WRITE_COUNTER.fetch_add(10, Ordering::SeqCst);
        }
        info!("txn file write thread {} exit", thread_idx);
    })
}
