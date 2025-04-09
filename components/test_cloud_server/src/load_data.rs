// Copyright 2023 TiKV Project Authors. Licensed under Apache-2.0.

use std::{collections::HashMap, mem, time::Duration};

use bytes::{BufMut, BytesMut};
use futures::executor::block_on;
use load_data::{
    checkpoint::LoadDataCheckpointCtx,
    dispatcher::Dispatcher,
    task::{
        FlushStates, LoadDataConfig, LoadDataContext, LoadTaskMsg, LoadTaskScheduler, TaskContext,
    },
};
use rand::prelude::SliceRandom;
use thiserror::Error;
use tikv_util::{error, time::Instant};

use crate::{client::RefStore, try_wait};

#[derive(Debug, Error)]
pub enum Error {
    #[error("task canceled: {0}")]
    Canceled(String),
    #[error("wait task finished timeout: {0:?}")]
    Timeout(load_data::task::LoadTaskStates),
}

pub type Result<T> = std::result::Result<T, Error>;

pub fn init_task(
    config: LoadDataConfig,
    ctx: LoadDataContext,
    start_ts: u64,
    commit_ts: u64,
) -> (LoadTaskScheduler, std::thread::JoinHandle<()>) {
    let task_id = format!("load_data_{}", start_ts);

    let task_ctx = TaskContext {
        task_id,
        start_ts,
        commit_ts,
        inner_key_off: None,
        outer_key_prefix: vec![],
        encryption_key: None,
        keyspace_id: None,
    };
    let checkpoint_ctx = LoadDataCheckpointCtx::new(task_ctx.clone());
    let mut dispatcher = Dispatcher::new(config.clone(), ctx.clone(), task_ctx, checkpoint_ctx);
    let scheduler = dispatcher.get_scheduler();
    let worker_handle = std::thread::spawn(move || {
        dispatcher.run();
    });

    assert!(
        !scheduler.is_canceled(),
        "task canceled: {}",
        scheduler.error_msg()
    );
    (scheduler, worker_handle)
}

pub fn put_chunks<FnKey, FnVal, FnRowId, FnDup>(
    scheduler: &LoadTaskScheduler,
    writer_count: usize,
    data_count: usize,
    batch_size: usize,
    i_to_key: FnKey,
    i_to_val: FnVal,
    i_to_row_id: FnRowId,
    timeout: Duration,
    dup_count: FnDup,
) -> RefStore
where
    FnKey: Fn(usize) -> Vec<u8>,
    FnVal: Fn(usize) -> Vec<u8>,
    FnRowId: Fn(usize) -> Vec<u8>,
    FnDup: Fn(usize) -> usize,
{
    let mut chunk_ids: HashMap<u64, u64> = HashMap::with_capacity(5);
    let mut ref_store = RefStore::default();
    let mut rng = rand::thread_rng();
    let mut integers: Vec<usize> = (0..data_count).collect();
    integers.shuffle(&mut rng);

    let capacity = (mem::size_of::<u16>() /* key length */ + i_to_key(0).len() + mem::size_of::<u32>() /* val length */ + i_to_val(0).len())
        * batch_size;
    for i in (0..data_count).step_by(batch_size) {
        let mut buf = BytesMut::with_capacity(capacity);
        for _ in 0..batch_size {
            if integers.is_empty() {
                break;
            }
            let n = integers.pop().unwrap();
            let key = i_to_key(n);
            let val = i_to_val(n);
            let row_id = i_to_row_id(n);

            buf.put_u16_le(key.len() as u16);
            buf.put_slice(&key);
            buf.put_u32_le(val.len() as u32);
            buf.put_slice(&val);
            buf.put_u16_le(row_id.len() as u16);
            buf.put_slice(&row_id);

            ref_store.put_kv(key, val);
            for k in 1..=dup_count(n) {
                // Repeated key is the same as the original key with some row_id.
                // Repeated key is caused by resending some data after the client restarts.
                let repeated_key = i_to_key(n);
                let repeated_val = i_to_val(n);
                let repeated_row_id = i_to_row_id(n);
                buf.put_u16_le(repeated_key.len() as u16);
                buf.put_slice(&repeated_key);
                buf.put_u32_le(repeated_val.len() as u32);
                buf.put_slice(&repeated_val);
                buf.put_u16_le(repeated_row_id.len() as u16);
                buf.put_slice(&repeated_row_id);

                // Duplicate key is the same as the original key with different row_id.
                let dup_key = i_to_key(n);
                let dup_val = i_to_val(n + k);
                let dup_row_id = i_to_val(n + k);
                buf.put_u16_le(dup_key.len() as u16);
                buf.put_slice(&dup_key);
                buf.put_u32_le(dup_val.len() as u32);
                buf.put_slice(&dup_val);
                buf.put_u16_le(dup_row_id.len() as u16);
                buf.put_slice(&dup_row_id);
                // do not put repeated_key/dup_key into ref_store
            }
        }

        let writer_id = ((i / batch_size) % writer_count) as u64;
        let chunk_id = chunk_ids.entry(writer_id).or_default();
        let (cb, fut) = tikv_util::future::paired_future_callback();
        *chunk_id += 1;
        scheduler
            .sender
            .send(LoadTaskMsg::AddChunk {
                writer_id,
                chunk_id: *chunk_id,
                chunk_data: buf.freeze(),
                cb,
            })
            .unwrap();
        let put_chunk_res = block_on(fut).unwrap();
        assert!(!put_chunk_res.canceled);
        assert!(!put_chunk_res.finished);
        assert!(
            put_chunk_res.error.is_empty(),
            "put_chunks error: {}",
            put_chunk_res.error
        );
    }

    let mut file_counts: HashMap<u64, Option<usize>> = HashMap::default();
    let ok = try_wait(
        || {
            assert!(
                !scheduler.is_canceled(),
                "task canceled: {}",
                scheduler.error_msg()
            );
            let mut flushed = true;
            for writer_id in 0..writer_count as u64 {
                let file_count = file_counts.entry(writer_id).or_default();
                let (cb, fut) = tikv_util::future::paired_future_callback();
                scheduler
                    .sender
                    .send(LoadTaskMsg::Flush {
                        writer_id,
                        flush_file_count: *file_count,
                        cb,
                    })
                    .unwrap();
                let flush_states = block_on(fut).unwrap();
                match flush_states {
                    FlushStates::FlushFileCount { flush_file_count } => {
                        *file_count = Some(flush_file_count);
                        flushed = false;
                    }
                    FlushStates::FlushResult { flush_result } => {
                        if flush_result.canceled
                            || flush_result.finished
                            || *flush_result.flushed_chunk_ids.get(&writer_id).unwrap()
                                != *chunk_ids.get(&writer_id).unwrap()
                        {
                            flushed = false;
                        }
                    }
                }
            }
            flushed
        },
        timeout.as_secs() as usize,
    );
    assert!(ok, "put_chunks timeout, states: {:?}", scheduler.states());

    ref_store
}

pub fn build(scheduler: &LoadTaskScheduler, compression_type: u8, timeout: Duration) -> Result<()> {
    let (cb, fut) = tikv_util::future::paired_future_callback();
    scheduler
        .sender
        .send(LoadTaskMsg::Build {
            compression_type,
            cb,
        })
        .unwrap();
    block_on(fut).unwrap();
    try_wait_finished(scheduler, timeout)
}

pub fn try_wait_finished(scheduler: &LoadTaskScheduler, timeout: Duration) -> Result<()> {
    let start_time = Instant::now_coarse();
    while start_time.saturating_elapsed() < timeout {
        if scheduler.is_finished() {
            return Ok(());
        } else if scheduler.is_canceled() {
            error!("build task canceled"; "error_msg" => scheduler.error_msg(), "states" => ?scheduler.states());
            return Err(Error::Canceled(scheduler.error_msg()));
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    Err(Error::Timeout(scheduler.states()))
}

pub fn cleanup(scheduler: &LoadTaskScheduler, worker_handle: std::thread::JoinHandle<()>) {
    scheduler.cancel("deleted".to_string());
    scheduler.sender.send(LoadTaskMsg::Cleanup).unwrap();
    worker_handle.join().unwrap();
}
