// Copyright 2023 TiKV Project Authors. Licensed under Apache-2.0.

use std::{mem, time::Duration};

use bytes::{BufMut, BytesMut};
use futures::executor::block_on;
use load_data::task::{
    LoadDataConfig, LoadDataContext, LoadTaskMsg, LoadTaskScheduler, LoadTaskWorker, TaskContext,
};
use tikv_util::{error, time::Instant};

use crate::{client::RefStore, try_wait};

pub fn init_task(
    config: LoadDataConfig,
    ctx: LoadDataContext,
    start_ts: u64,
    commit_ts: u64,
) -> LoadTaskScheduler {
    let task_id = format!("load_data_{}", start_ts);
    let task_ctx = TaskContext {
        task_id,
        start_ts,
        commit_ts,
        inner_key_off: None,
        key_prefix: vec![],
        encryption_key: None,
    };

    let mut worker = LoadTaskWorker::new(config, ctx, task_ctx);
    let scheduler = worker.get_scheduler();
    std::thread::spawn(move || {
        worker.run();
    });

    assert!(
        !scheduler.is_canceled(),
        "task canceled: {}",
        scheduler.error_msg()
    );
    scheduler
}

pub fn put_chunks<FnKey, FnVal, FnDup>(
    scheduler: &LoadTaskScheduler,
    data_count: usize,
    batch_size: usize,
    i_to_key: FnKey,
    i_to_val: FnVal,
    timeout: Duration,
    dup_count: FnDup,
) -> (Vec<u64>, RefStore)
where
    FnKey: Fn(usize) -> Vec<u8>,
    FnVal: Fn(usize) -> Vec<u8>,
    FnDup: Fn(usize) -> usize,
{
    let mut chunk_ids = Vec::with_capacity((data_count as f64 / batch_size as f64).ceil() as usize);
    let mut ref_store = RefStore::default();

    let capacity = (mem::size_of::<u16>() /* key length */ + i_to_key(0).len() + mem::size_of::<u32>() /* val length */ + i_to_val(0).len())
        * batch_size;
    for i in (0..data_count).step_by(batch_size) {
        let mut buf = BytesMut::with_capacity(capacity);
        for j in 0..batch_size {
            let idx = i + j;
            if idx >= data_count {
                break;
            }
            let key = i_to_key(idx);
            let val = i_to_val(idx);

            buf.put_u16_le(key.len() as u16);
            buf.put_slice(&key);
            buf.put_u32_le(val.len() as u32);
            buf.put_slice(&val);

            ref_store.put_kv(key, val);
            for k in 1..=dup_count(idx) {
                let dup_key = i_to_key(idx);
                let dup_val = i_to_val(idx + k);
                buf.put_u16_le(dup_key.len() as u16);
                buf.put_slice(&dup_key);
                buf.put_u32_le(dup_val.len() as u32);
                buf.put_slice(&dup_val);
                // do not put dup_key into ref_store
            }
        }

        let chunk_id = i as u64;
        scheduler
            .sender
            .send(LoadTaskMsg::AddChunk {
                chunk_id,
                chunk_data: buf.freeze(),
            })
            .unwrap();

        chunk_ids.push(chunk_id)
    }

    let mut unhandled_chunk_ids = chunk_ids.clone();
    let ok = try_wait(
        || {
            assert!(
                !scheduler.is_canceled(),
                "task canceled: {}",
                scheduler.error_msg()
            );
            let mut res = block_on(scheduler.query_unhandled_chunks(unhandled_chunk_ids.clone()));
            unhandled_chunk_ids = mem::take(&mut res);
            unhandled_chunk_ids.is_empty()
        },
        timeout.as_secs() as usize,
    );
    assert!(ok, "put_chunks timeout, states: {:?}", scheduler.states());

    (chunk_ids, ref_store)
}

pub fn build(
    scheduler: &LoadTaskScheduler,
    chunk_ids: Vec<u64>,
    compression_type: u8,
    timeout: Duration,
) -> std::result::Result<(), String> {
    scheduler
        .sender
        .send(LoadTaskMsg::Build {
            chunk_ids,
            compression_type,
        })
        .unwrap();

    let start_time = Instant::now_coarse();
    while start_time.saturating_elapsed() < timeout {
        if scheduler.is_finished() {
            return Ok(());
        } else if scheduler.is_canceled() {
            error!("build task canceled"; "error_msg" => scheduler.error_msg(), "states" => ?scheduler.states());
            return Err(format!("task canceled: {}", scheduler.error_msg()));
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    panic!("build timeout, states: {:?}", scheduler.states());
}

pub fn cleanup(scheduler: &LoadTaskScheduler) {
    scheduler.cancel("deleted".to_string());
    scheduler.sender.send(LoadTaskMsg::Cleanup).unwrap();
}
