// Copyright 2025 TiKV Project Authors. Licensed under Apache-2.0.

use std::future::Future;

use tokio::task::JoinHandle;

/// Note: `WorkerPool::Pool` is shutdown in background when dropped.
#[derive(Debug)]
pub enum WorkerPool {
    Pool(Option<tokio::runtime::Runtime>),
    Handle(tokio::runtime::Handle),
}

impl Drop for WorkerPool {
    fn drop(&mut self) {
        if let Self::Pool(pool) = self {
            pool.take().unwrap().shutdown_background();
        }
    }
}

impl WorkerPool {
    pub fn handle(&self) -> WorkerPoolHandle {
        match self {
            Self::Pool(pool) => WorkerPoolHandle::new(pool.as_ref().unwrap().handle().clone()),
            Self::Handle(handle) => WorkerPoolHandle::new(handle.clone()),
        }
    }

    pub fn spawn<F>(&self, future: F) -> JoinHandle<F::Output>
    where
        F: Future + Send + 'static,
        F::Output: Send + 'static,
    {
        let future = crate::init_task_local(future);
        match self {
            Self::Pool(pool) => pool.as_ref().unwrap().spawn(future),
            Self::Handle(handle) => handle.spawn(future),
        }
    }

    pub fn enter(&self) -> tokio::runtime::EnterGuard<'_> {
        match self {
            Self::Pool(pool) => pool.as_ref().unwrap().enter(),
            Self::Handle(handle) => handle.enter(),
        }
    }
}

impl From<tokio::runtime::Handle> for WorkerPool {
    fn from(handle: tokio::runtime::Handle) -> Self {
        Self::Handle(handle)
    }
}

impl From<tokio::runtime::Runtime> for WorkerPool {
    fn from(runtime: tokio::runtime::Runtime) -> Self {
        Self::Pool(Some(runtime))
    }
}

#[derive(Clone)]
pub struct WorkerPoolHandle {
    handle: tokio::runtime::Handle,
}

impl WorkerPoolHandle {
    fn new(handle: tokio::runtime::Handle) -> Self {
        Self { handle }
    }

    pub fn spawn<F>(&self, future: F) -> JoinHandle<F::Output>
    where
        F: Future + Send + 'static,
        F::Output: Send + 'static,
    {
        self.handle.spawn(crate::init_task_local(future))
    }

    pub fn spawn_blocking<F, R>(&self, func: F) -> JoinHandle<R>
    where
        F: FnOnce() -> R + Send + 'static,
        R: Send + 'static,
    {
        self.handle
            .spawn_blocking(move || crate::init_task_local_sync(func))
    }

    pub fn block_on<F, R>(&self, future: F) -> R
    where
        F: Future<Output = R> + Send + 'static,
        R: Send + 'static,
    {
        self.handle.block_on(crate::init_task_local(future))
    }
}
