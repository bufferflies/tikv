// Copyright 2019 TiKV Project Authors. Licensed under Apache-2.0.

use crate::engine::KvEngine;
use crate::errors::Result;
use crate::options::WriteOptions;
use crate::raft_engine::RaftEngine;
use crate::write_batch::WriteBatch;
use std::fmt::{self, Debug};
use std::path::{Path, PathBuf};

pub trait TabletFactory<EK> {
    fn loop_tablet_cache(&self, _f: Box<dyn FnMut(u64, u64, &EK) + '_>) {}
    fn destroy_tablet(&self, _id: u64, _suffix: u64) -> crate::Result<()> {
        Ok(())
    }
    fn create_tablet(&self, id: u64, suffix: u64) -> EK;
    fn open_tablet(&self, id: u64, suffix: u64) -> EK {
        self.open_tablet_raw(&self.tablet_path(id, suffix), false)
    }
    fn open_tablet_cache(&self, id: u64, suffix: u64) -> Option<EK> {
        Some(self.open_tablet_raw(&self.tablet_path(id, suffix), false))
    }
    fn open_tablet_cache_any(&self, id: u64) -> Option<EK> {
        Some(self.open_tablet_raw(&self.tablet_path(id, 0), false))
    }
    fn open_tablet_raw(&self, path: &Path, readonly: bool) -> EK;
    fn create_root_db(&self) -> EK;
    #[inline]
    fn exists(&self, id: u64, suffix: u64) -> bool {
        self.exists_raw(&self.tablet_path(id, suffix))
    }
    fn exists_raw(&self, path: &Path) -> bool;
    fn tablet_path(&self, id: u64, suffix: u64) -> PathBuf;
    fn tablets_path(&self) -> PathBuf;
    fn clone(&self) -> Box<dyn TabletFactory<EK> + Send>;
    fn load_tablet(&self, _path: &Path, id: u64, suffix: u64) -> EK {
        self.open_tablet(id, suffix)
    }
    fn mark_tombstone(&self, _region_id: u64, _suffix: u64) {}
    fn is_tombstoned(&self, _region_id: u64, _suffix: u64) -> bool {
        false
    }
}

pub struct DummyFactory;

impl<EK> TabletFactory<EK> for DummyFactory {
    fn create_tablet(&self, _id: u64, _suffix: u64) -> EK {
        unimplemented!()
    }
    fn open_tablet_raw(&self, _path: &Path, _readonly: bool) -> EK {
        unimplemented!()
    }
    fn create_root_db(&self) -> EK {
        unimplemented!()
    }
    fn exists_raw(&self, _path: &Path) -> bool {
        unimplemented!()
    }
    fn tablet_path(&self, _id: u64, _suffix: u64) -> PathBuf {
        unimplemented!()
    }
    fn tablets_path(&self) -> PathBuf {
        unimplemented!()
    }

    fn clone(&self) -> Box<dyn TabletFactory<EK> + Send> {
        Box::new(DummyFactory)
    }
}

pub struct Engines<K, R> {
    pub kv: K,
    pub raft: R,
    pub tablets: Box<dyn TabletFactory<K> + Send>,
}

impl<K: KvEngine, R: RaftEngine> Engines<K, R> {
    pub fn new(kv_engine: K, raft_engine: R) -> Self {
        Engines {
            kv: kv_engine,
            raft: raft_engine,
            tablets: Box::new(DummyFactory),
        }
    }

    pub fn write_kv(&self, wb: &K::WriteBatch) -> Result<()> {
        wb.write()
    }

    pub fn write_kv_opt(&self, wb: &K::WriteBatch, opts: &WriteOptions) -> Result<()> {
        wb.write_opt(opts)
    }

    pub fn sync_kv(&self) -> Result<()> {
        self.kv.sync()
    }
}

impl<K: Clone, R: Clone> Clone for Engines<K, R> {
    #[inline]
    fn clone(&self) -> Engines<K, R> {
        Engines {
            kv: self.kv.clone(),
            raft: self.raft.clone(),
            tablets: self.tablets.clone(),
        }
    }
}

impl<K: Debug, R: Debug> Debug for Engines<K, R> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt.debug_struct("Engines")
            .field("kv", &self.kv)
            .field("raft", &self.raft)
            .field("tablets", &self.tablets.tablets_path().display())
            .finish()
    }
}
