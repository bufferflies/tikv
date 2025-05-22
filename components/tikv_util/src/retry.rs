// Copyright 2024 TiKV Project Authors. Licensed under Apache-2.0.

use std::time::Duration;

use futures_util::{compat::Future01CompatExt, future::BoxFuture};

use crate::{time::Instant, timer::GLOBAL_TIMER_HANDLE};

/// Async sleep when tokio runtime may be not available.
pub async fn sleep_async(dur: Duration) {
    GLOBAL_TIMER_HANDLE
        .delay(std::time::Instant::now() + dur)
        .compat()
        .await
        .unwrap();
}

pub fn try_wait_result<T, E, F, Bo>(f: F, timeout: Duration, mut backoff: Bo) -> Result<T, E>
where
    F: Fn() -> Result<T, E>,
    Bo: FnMut() -> Duration,
{
    let start_time = Instant::now_coarse();
    let mut last_err = None;
    while start_time.saturating_elapsed() < timeout {
        match f() {
            Ok(r) => return Ok(r),
            Err(e) => {
                last_err = Some(e);
                std::thread::sleep(backoff());
            }
        }
    }
    Err(last_err.unwrap())
}

pub async fn try_wait_result_async<'a, T, E, F, Bo>(
    f: F,
    timeout: Duration,
    mut backoff: Bo,
) -> Result<T, E>
where
    F: Fn() -> BoxFuture<'a, Result<T, E>>,
    Bo: FnMut() -> Duration,
{
    let start_time = Instant::now_coarse();
    let mut last_err = None;
    while start_time.saturating_elapsed() < timeout {
        match f().await {
            Ok(r) => return Ok(r),
            Err(e) => {
                last_err = Some(e);
                sleep_async(backoff()).await;
            }
        }
    }
    Err(last_err.unwrap())
}
