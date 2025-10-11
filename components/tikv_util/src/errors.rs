// Copyright 2025 TiKV Project Authors. Licensed under Apache-2.0.

use std::{error, fmt, io, ops};

pub type BoxError = Box<dyn std::error::Error + Sync + Send>;

pub struct ErrorWithCtx<E> {
    pub inner: E,
    ctx: String,
}

impl<E: error::Error> fmt::Debug for ErrorWithCtx<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?} ({})", self.inner, self.ctx)
    }
}

impl<E: error::Error> fmt::Display for ErrorWithCtx<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} ({})", self.inner, self.ctx)
    }
}

impl<E: error::Error> error::Error for ErrorWithCtx<E> {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        self.inner.source()
    }
}

impl<E> ops::Deref for ErrorWithCtx<E> {
    type Target = E;

    fn deref(&self) -> &E {
        &self.inner
    }
}

impl<E> ErrorWithCtx<E> {
    pub fn new(e: E, ctx: String) -> Self {
        Self { inner: e, ctx }
    }

    pub fn ctx(&self) -> &str {
        &self.ctx
    }
}

// Ref: [`anyhow::Context`](https://github.com/dtolnay/anyhow/blob/1.0.26/src/lib.rs#L543)
pub trait Context<T, E> {
    fn ctx<C>(self, ctx: C) -> Result<T, ErrorWithCtx<E>>
    where
        C: fmt::Display + Send + Sync + 'static;

    fn with_ctx<C, F>(self, f: F) -> Result<T, ErrorWithCtx<E>>
    where
        C: fmt::Display + Send + Sync + 'static,
        F: FnOnce() -> C;
}

pub type IoError = ErrorWithCtx<io::Error>;

impl<T> Context<T, io::Error> for io::Result<T> {
    fn ctx<C>(self, ctx: C) -> Result<T, IoError>
    where
        C: fmt::Display + Send + Sync + 'static,
    {
        self.map_err(|err| ErrorWithCtx {
            inner: err,
            ctx: format!("{ctx}"),
        })
    }

    fn with_ctx<C, F>(self, ctx_fn: F) -> Result<T, IoError>
    where
        C: fmt::Display + Send + Sync + 'static,
        F: FnOnce() -> C,
    {
        self.map_err(|err| ErrorWithCtx {
            inner: err,
            ctx: format!("{}", ctx_fn()),
        })
    }
}
