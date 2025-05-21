// Copyright 2025 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    thread,
};

use bytes::Bytes;
use fail::cfg_callback;
use kvengine::dfs::S3Fs;
use kvproto::kvrpcpb::Op;
use native_br::{
    backup,
    restore::RestoreConfig,
    restore_keyspace,
    restore_keyspace::{ReportRestoreStepTrait, RestoreStep},
};
use test_cloud_server::{
    alloc_node_id_vec,
    client::{PessimisticLockExt, PrewriteExt, RequestOptions, TxnMutations},
    oss::prepare_dfs,
    util::Mutation,
    ServerCluster,
};
use tikv::config::TikvConfig;
use tikv_util::info;
use tokio::{
    runtime::Runtime,
    sync::oneshot::{Receiver, Sender},
};
use txn_types::Key;

use crate::cases::{i_to_keyspace_key, random_value};

#[test]
fn test_backup_pessimistic_lock() {
    test_util::init_log_for_test();
    const KEYSPACE_ID: u32 = 1;
    const DATA_LEN: usize = 100;
    const VALUE_SIZE: usize = 64;

    let (_temp_dir, mut oss, dfs_config) = prepare_dfs("t_");
    let s3fs = Arc::new(S3Fs::new(
        dfs_config.prefix.clone(),
        dfs_config.s3_endpoint.clone(),
        dfs_config.s3_key_id.clone(),
        dfs_config.s3_secret_key.clone(),
        dfs_config.s3_region.clone(),
        dfs_config.s3_bucket.clone(),
    ));
    let reporter = Arc::new(DummyStepReporter::default());
    let runtime = Runtime::new().unwrap();

    let nodes = alloc_node_id_vec(2);
    let mut cluster = ServerCluster::new(nodes.clone(), |_, conf: &mut TikvConfig| {
        conf.dfs = dfs_config.clone();
        conf.rfengine.lightweight_backup = true;
        conf.enable_inner_key_offset = true;
    });
    cluster.wait_region_replicated(&[], 1);

    let pd_client = cluster.get_pd_client();
    let mut client = cluster.new_client();

    pd_client.disable_default_operator();
    cluster.remove_node_peers(nodes[1]);
    info!("ConfChange removed node peers {}", nodes[1]);

    client.split_keyspace(KEYSPACE_ID);
    client.set_async_commit();

    let i_to_key = i_to_keyspace_key(KEYSPACE_ID);

    client.put_kv(1..DATA_LEN, &i_to_key, random_value::<VALUE_SIZE>);
    client.verify_data_with_ref_store();

    // split at DATA_LEN / 2
    let split_key = i_to_key(DATA_LEN / 2);
    client.split(&split_key);
    let enc = |k| Key::from_raw(k).into_encoded();

    let pk = i_to_key(DATA_LEN);
    let sk = i_to_key(0);

    let region = client.pd_client.get_region(&enc(&pk)).unwrap();
    assert_eq!(region.peers.len(), 1, "{:?}", region);
    cluster.evict_peer(region.peers[0].id);
    let region = client.pd_client.get_region(&enc(&sk)).unwrap();
    let region2 = client.pd_client.get_region(&enc(&pk)).unwrap();
    assert_eq!(region.peers.len(), 1, "{:?}", region);
    assert_eq!(region2.peers.len(), 1, "{:?}", region);
    assert_ne!(
        region.peers[0].store_id, region2.peers[0].store_id,
        "{:?}\n{:?}",
        region, region2
    );

    let start_ts = client.get_ts();
    client
        .kv_pessimistic_lock_ext(
            Bytes::copy_from_slice(&pk),
            TxnMutations::from_normal(vec![
                pessimistic_lock(pk.clone()),
                pessimistic_lock(sk.clone()),
            ]),
            PessimisticLockExt {
                start_ts,
                for_update_ts: start_ts,
            },
        )
        .unwrap();
    let origin_ref_store = client.dump_ref_store();

    let backup_config = backup::BackupConfig {
        dfs: dfs_config.clone(),
        tolerate_err: 1,
        skip_keyspace_meta: true,
        ..Default::default()
    };

    // Perform backup.
    let backup_name = generate_backup_name();
    thread::scope(|s| {
        let backup_config = backup_config.clone();
        let backup_name = backup_name.clone();
        let pd_client = pd_client.clone();
        let backup_ts = client.get_ts().into_inner();

        let rx = cfg_notify_once("native_br::backup_store").unwrap();
        let rx_finish = cfg_notify_once("native_br::backup_store::ret").unwrap();

        let h = s.spawn(move || {
            backup::backup_cluster_with_ts(
                backup_config,
                backup::BackupType::Lightweight,
                backup_name,
                pd_client.as_ref(),
                backup_ts,
                None,
            )
        });

        let tx = rx.blocking_recv().unwrap();
        let muts = vec![
            put(pk.clone(), random_value::<VALUE_SIZE>(0)),
            put(sk.clone(), random_value::<VALUE_SIZE>(0)),
        ];
        let txn = TxnMutations::from_normal(muts.clone());
        let _ = rx_finish.blocking_recv();
        client
            .kv_prewrite_ext_with_retry(
                Bytes::copy_from_slice(&pk),
                Some(&vec![Bytes::copy_from_slice(&sk)]),
                txn.clone(),
                start_ts,
                PrewriteExt {
                    pessimistic_action:
                        kvproto::kvrpcpb::PrewriteRequestPessimisticAction::DoPessimisticCheck,
                    for_update_ts: start_ts,
                },
            )
            .unwrap();
        tx.send(()).unwrap();

        let (_, backup_meta) = h.join().unwrap().expect("backup");
        info!("backup cluster result: {}", backup_meta);
    });

    // Perform restore to verify backup.
    let restore_config = RestoreConfig {
        tolerate_err: 0,
        strict_tolerate: true,
        ..Default::default()
    };
    let res = restore_keyspace::restore_keyspace(
        KEYSPACE_ID,
        KEYSPACE_ID,
        &backup_name,
        None,
        s3fs.clone(),
        restore_config.clone(),
        cluster.get_pd_client(),
        &runtime,
        None,
        reporter.clone(),
    )
    .expect("restore");
    info!(
        "restore keyspace result for backup {}: {:?}",
        backup_name, res
    );
    let (existed, deleted) = client
        .verify_data_with_given_ref_store(&origin_ref_store, None, &RequestOptions::default())
        .unwrap();
    assert_eq!(existed, DATA_LEN - 1);
    assert_eq!(deleted, 0);
    assert_eq!(res.resolved_ts, start_ts);

    cluster.stop();
    oss.shutdown();
}

#[derive(Default)]
struct DummyStepReporter {}

impl ReportRestoreStepTrait for DummyStepReporter {
    fn report_step(&self, _step: RestoreStep) {}
}

fn generate_backup_name() -> String {
    static BACKUP_ID: AtomicUsize = AtomicUsize::new(0);
    format!("{:04}", BACKUP_ID.fetch_add(1, Ordering::Relaxed))
}

fn put(k: Vec<u8>, v: Vec<u8>) -> Mutation {
    let mut m = Mutation::default();
    m.set_op(Op::Put);
    m.set_key(k);
    m.set_value(v);
    m
}

fn pessimistic_lock(k: Vec<u8>) -> Mutation {
    let mut m = Mutation::default();
    m.set_op(Op::PessimisticLock);
    m.set_key(k);
    m
}

fn cfg_notify_once(name: &str) -> std::result::Result<Receiver<Sender<()>>, String> {
    let (tx, rx) = tokio::sync::oneshot::channel();
    let pack = std::sync::Mutex::new(Some(tx));
    cfg_callback(name, move || {
        info!("injecting failpoint");
        let mut v = pack.lock().unwrap();
        if v.is_none() {
            return;
        }
        let tx = v.take().unwrap();
        drop(v);
        let (tx2, rx) = tokio::sync::oneshot::channel();

        let _ = tx.send(tx2);
        tokio::task::block_in_place(|| {
            let _ = rx.blocking_recv();
        })
    })?;

    Ok(rx)
}
