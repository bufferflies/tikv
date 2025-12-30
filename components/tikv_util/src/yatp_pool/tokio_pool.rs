// Copyright 2025 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    future::Future,
    sync::{Arc, Condvar, Mutex},
};

use tokio::{
    runtime::{Handle, Runtime},
    task::JoinHandle,
};

#[derive(Default)]
// A (bool, Condvar) pair used to pause/resume a worker thread.
// the bool means pause state, the default value is false because
// the thread is running by default when it's created.
struct ThreadPauser(Arc<(Mutex<bool>, Condvar)>);

impl ThreadPauser {
    fn pause(&self) -> impl Future<Output = ()> + Send + Sync + 'static {
        {
            let mut paused = self.0.0.lock().unwrap();
            assert_eq!(*paused, false);
            *paused = true;
        }

        let pair = self.0.clone();
        async move {
            let (lock, cvar) = &*pair;
            let mut is_paused = lock.lock().unwrap();
            while *is_paused {
                is_paused = cvar.wait(is_paused).unwrap();
            }
        }
    }

    fn resume(&self) {
        let mut is_paused = self.0.0.lock().unwrap();
        assert_eq!(*is_paused, true);
        *is_paused = false;
        self.0.1.notify_one();
    }
}

pub struct ScalableTokioRuntime {
    // use Option to support shutdown runtime.
    runtime: Option<Runtime>,
    // we use Mutex to ensure the thread pool scaling is always running sequentially.
    active_thread_count: Arc<Mutex<usize>>,
    thread_pausers: Arc<[ThreadPauser]>,
}

impl ScalableTokioRuntime {
    pub fn new(runtime: Runtime, max_thread_count: usize, active_thread_count: usize) -> Self {
        assert!(active_thread_count > 0 && active_thread_count <= max_thread_count);
        let mut thread_pausers = Vec::with_capacity(max_thread_count);
        for _i in 0..max_thread_count {
            thread_pausers.push(ThreadPauser::default());
        }

        // spawn some blocking tasks to pause the extra threads.
        for i in active_thread_count..max_thread_count {
            let fut = thread_pausers[i].pause();
            runtime.spawn(fut);
        }

        Self {
            runtime: Some(runtime),
            active_thread_count: Arc::new(Mutex::new(active_thread_count)),
            thread_pausers: Arc::from(thread_pausers),
        }
    }

    pub fn handle(&self) -> Option<ScalableTokioHandle> {
        self.runtime.as_ref().map(|r| ScalableTokioHandle {
            handle: r.handle().clone(),
            active_thread_count: self.active_thread_count.clone(),
            thread_pausers: self.thread_pausers.clone(),
        })
    }

    pub fn shutdown(&mut self) {
        if let Some(handle) = self.handle() {
            // try to release all blocking tasks.
            handle.scale_pool_size(self.thread_pausers.len());
        }
        if let Some(runtime) = self.runtime.take() {
            runtime.shutdown_background();
        }
    }
}

impl Drop for ScalableTokioRuntime {
    fn drop(&mut self) {
        self.shutdown();
    }
}

#[derive(Clone)]
pub struct ScalableTokioHandle {
    handle: Handle,
    active_thread_count: Arc<Mutex<usize>>,
    thread_pausers: Arc<[ThreadPauser]>,
}

impl ScalableTokioHandle {
    pub fn scale_pool_size(&self, new_count: usize) {
        if new_count == 0 || new_count > self.thread_pausers.len() {
            crate::warn!(
                "scaling tokio thread pool out of bound, skipped. valid range: [1, {}], got: {}",
                self.thread_pausers.len(),
                new_count
            );
            return;
        }
        let mut current_count = self.active_thread_count.lock().unwrap();
        if new_count > *current_count {
            for i in *current_count..new_count {
                self.thread_pausers[i].resume();
            }
        } else {
            for i in new_count..*current_count {
                let wait_fut = self.thread_pausers[i].pause();
                self.handle.spawn(wait_fut);
            }
        }
        *current_count = new_count;
    }

    pub fn get_pool_size(&self) -> usize {
        *self.active_thread_count.lock().unwrap()
    }

    #[track_caller]
    pub fn spawn<F>(&self, future: F) -> JoinHandle<F::Output>
    where
        F: Future + Send + 'static,
        F::Output: Send + 'static,
    {
        self.handle.spawn(future)
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    #[test]
    fn test_scale_tokio_pool_size() {
        const TEST_THREAD_COUNT: usize = 4;
        let raw_runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(TEST_THREAD_COUNT)
            .enable_all()
            .build()
            .unwrap();
        let runtime = ScalableTokioRuntime::new(raw_runtime, TEST_THREAD_COUNT, 2);
        let handle = runtime.handle().unwrap();

        fn check_pool_size(handle: &ScalableTokioHandle, pool_size: usize) {
            // first spawn a simple task to ensure the thread pool can handle tasks.
            let (tx, rx) = crate::mpsc::bounded(1);
            handle.spawn(async move {
                tx.send(()).unwrap();
            });
            rx.recv_timeout(Duration::from_millis(500)).unwrap();

            // spawn N-1 blocking tasks to hold all running threads
            let mut senders = vec![];
            for _i in 0..pool_size {
                let (tx, rx) = crate::mpsc::bounded(1);
                handle.spawn(async move {
                    rx.recv().unwrap();
                });
                senders.push(tx);
            }

            // spawn another task, it should be block due to no available running threads.
            let (tx, rx) = crate::mpsc::bounded(1);
            handle.spawn(async move {
                tx.send(()).unwrap();
            });
            rx.recv_timeout(Duration::from_millis(500)).unwrap_err();
            // release a blocking tasks.
            senders[0].send(()).unwrap();
            // the test task should execute now.
            rx.recv_timeout(Duration::from_millis(500)).unwrap();

            // clean up.
            for i in 1..pool_size {
                senders[i].send(()).unwrap();
            }
        }

        check_pool_size(&handle, 2);

        for size in [4, 1, 2, 3, 4, 2, 1] {
            handle.scale_pool_size(size);
            check_pool_size(&handle, size);
        }
    }
}
