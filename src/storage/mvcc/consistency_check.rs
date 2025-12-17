// Copyright 2020 TiKV Project Authors. Licensed under Apache-2.0.

// #[PerformanceCriticalPath] called by raftstore
use std::{cmp::Ordering, marker::PhantomData};

use engine_traits::{
    IterOptions, Iterator as EngineIterator, KvEngine, CF_DEFAULT, CF_LOCK, CF_WRITE,
};
use kvproto::kvrpcpb::{MvccInfo, MvccLock, MvccValue, MvccWrite, Op};
use raftstore::{coprocessor::Coprocessor, Result};
use tikv_util::keybuilder::KeyBuilder;
use txn_types::Key;

use crate::storage::mvcc::{Lock, LockType, WriteRef, WriteType};

#[derive(Clone)]
pub struct Mvcc<E: KvEngine> {
    _engine: PhantomData<E>,
}

impl<E: KvEngine> Coprocessor for Mvcc<E> {}

pub trait MvccInfoObserver {
    type Target;

    // Meet a new mvcc record prefixed `key`.
    fn on_new_item(&mut self, key: &[u8]);
    // Emit a complete mvcc record.
    fn emit(&mut self) -> Self::Target;

    fn on_write(&mut self, key: &[u8], value: &[u8]) -> Result<bool>;
    fn on_lock(&mut self, key: &[u8], value: &[u8]) -> Result<bool>;
    fn on_default(&mut self, key: &[u8], value: &[u8]) -> Result<bool>;
}

pub struct MvccInfoScanner<Iter: EngineIterator, Ob: MvccInfoObserver> {
    lock_iter: Iter,
    default_iter: Iter,
    write_iter: Iter,
    observer: Ob,
}

impl<Iter: EngineIterator, Ob: MvccInfoObserver> MvccInfoScanner<Iter, Ob> {
    pub fn new<F>(f: F, from: Option<&[u8]>, to: Option<&[u8]>, ob: Ob) -> Result<Self>
    where
        F: Fn(&str, IterOptions) -> Result<Iter>,
    {
        let from = from.unwrap_or(keys::DATA_MIN_KEY);
        let to = to.unwrap_or(keys::DATA_MAX_KEY);
        let key_builder = |key: &[u8]| -> Result<Option<KeyBuilder>> {
            if !keys::validate_data_key(key) && key != keys::DATA_MAX_KEY {
                return Err(box_err!("non-mvcc area {}", log_wrappers::Value::key(key)));
            }
            Ok(Some(KeyBuilder::from_vec(key.to_vec(), 0, 0)))
        };

        let iter_opts = IterOptions::new(key_builder(from)?, key_builder(to)?, false);
        let gen_iter = |cf: &str| -> Result<Iter> {
            let mut iter = f(cf, iter_opts.clone())?;
            box_try!(iter.seek(from));
            Ok(iter)
        };

        Ok(MvccInfoScanner {
            lock_iter: gen_iter(CF_LOCK)?,
            default_iter: gen_iter(CF_DEFAULT)?,
            write_iter: gen_iter(CF_WRITE)?,
            observer: ob,
        })
    }

    fn next_item(&mut self) -> Result<Option<Ob::Target>> {
        let mut lock_ok = box_try!(self.lock_iter.valid());
        let mut writes_ok = box_try!(self.write_iter.valid());

        let prefix = match (lock_ok, writes_ok) {
            (false, false) => return Ok(None),
            (true, false) => self.lock_iter.key(),
            (false, true) => box_try!(Key::truncate_ts_for(self.write_iter.key())),
            (true, true) => {
                let prefix1 = self.lock_iter.key();
                let prefix2 = box_try!(Key::truncate_ts_for(self.write_iter.key()));
                match prefix1.cmp(prefix2) {
                    Ordering::Less => {
                        writes_ok = false;
                        prefix1
                    }
                    Ordering::Greater => {
                        lock_ok = false;
                        prefix2
                    }
                    Ordering::Equal => prefix1,
                }
            }
        };
        self.observer.on_new_item(prefix);

        while writes_ok {
            let (key, value) = (self.write_iter.key(), self.write_iter.value());
            writes_ok = self.observer.on_write(key, value)? && box_try!(self.write_iter.next());
        }
        while lock_ok {
            let (key, value) = (self.lock_iter.key(), self.lock_iter.value());
            lock_ok = self.observer.on_lock(key, value)? && box_try!(self.lock_iter.next());
        }

        let mut ok = box_try!(self.default_iter.valid());
        while ok {
            let (key, value) = (self.default_iter.key(), self.default_iter.value());
            ok = self.observer.on_default(key, value)? && box_try!(self.default_iter.next());
        }

        Ok(Some(self.observer.emit()))
    }
}

#[derive(Clone, Default)]
struct MvccInfoCollector {
    current_item: Vec<u8>,
    mvcc_info: MvccInfo,
}

impl MvccInfoObserver for MvccInfoCollector {
    type Target = (Vec<u8>, MvccInfo);

    fn on_new_item(&mut self, key: &[u8]) {
        self.current_item = key.to_vec();
    }

    fn emit(&mut self) -> Self::Target {
        let item = std::mem::take(&mut self.current_item);
        let info = std::mem::take(&mut self.mvcc_info);
        (item, info)
    }

    fn on_write(&mut self, key: &[u8], value: &[u8]) -> Result<bool> {
        let (prefix, commit_ts) = box_try!(Key::split_on_ts_for(key));
        if prefix != AsRef::<[u8]>::as_ref(&self.current_item) {
            return Ok(false);
        }

        let write = box_try!(WriteRef::parse(value));
        let mut write_info = MvccWrite::default();
        match write.write_type {
            WriteType::Put => write_info.set_type(Op::Put),
            WriteType::Delete => write_info.set_type(Op::Del),
            WriteType::Lock => write_info.set_type(Op::Lock),
            WriteType::Rollback => write_info.set_type(Op::Rollback),
        }
        write_info.set_start_ts(write.start_ts.into_inner());
        write_info.set_commit_ts(commit_ts.into_inner());
        if let Some(value) = write.short_value {
            write_info.set_short_value(value.to_vec());
        }

        self.mvcc_info.mut_writes().push(write_info);
        Ok(true)
    }

    fn on_lock(&mut self, key: &[u8], value: &[u8]) -> Result<bool> {
        if key != AsRef::<[u8]>::as_ref(&self.current_item) {
            return Ok(false);
        }

        let lock = box_try!(Lock::parse(value));
        let mut lock_info = MvccLock::default();
        match lock.lock_type {
            LockType::Put => lock_info.set_type(Op::Put),
            LockType::Delete => lock_info.set_type(Op::Del),
            LockType::Lock => lock_info.set_type(Op::Lock),
            LockType::Pessimistic => lock_info.set_type(Op::PessimisticLock),
        }
        lock_info.set_start_ts(lock.ts.into_inner());
        lock_info.set_primary(lock.primary);
        lock_info.set_short_value(lock.short_value.unwrap_or_default());

        self.mvcc_info.set_lock(lock_info);
        Ok(true)
    }

    fn on_default(&mut self, key: &[u8], value: &[u8]) -> Result<bool> {
        let (prefix, start_ts) = box_try!(Key::split_on_ts_for(key));
        if prefix != AsRef::<[u8]>::as_ref(&self.current_item) {
            return Ok(false);
        }

        let mut value_info = MvccValue::default();
        value_info.set_start_ts(start_ts.into_inner());
        value_info.set_value(value.to_vec());

        self.mvcc_info.mut_values().push(value_info);
        Ok(true)
    }
}

pub struct MvccInfoIterator<Iter: EngineIterator> {
    scanner: MvccInfoScanner<Iter, MvccInfoCollector>,
    limit: usize,
    count: usize,
}

impl<Iter: EngineIterator> MvccInfoIterator<Iter> {
    pub fn new<F>(f: F, from: Option<&[u8]>, to: Option<&[u8]>, limit: usize) -> Result<Self>
    where
        F: Fn(&str, IterOptions) -> Result<Iter>,
    {
        let scanner = MvccInfoScanner::new(f, from, to, MvccInfoCollector::default())?;
        Ok(Self {
            scanner,
            limit,
            count: 0,
        })
    }
}

impl<Iter: EngineIterator> Iterator for MvccInfoIterator<Iter> {
    type Item = Result<(Vec<u8>, MvccInfo)>;

    fn next(&mut self) -> Option<Result<(Vec<u8>, MvccInfo)>> {
        if self.limit != 0 && self.count >= self.limit {
            return None;
        }

        match self.scanner.next_item() {
            Ok(Some(item)) => {
                self.count += 1;
                Some(Ok(item))
            }
            Ok(None) => None,
            Err(e) => Some(Err(e)),
        }
    }
}
