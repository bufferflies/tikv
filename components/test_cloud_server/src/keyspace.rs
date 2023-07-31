// Copyright 2023 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    collections::HashMap,
    mem,
    ops::{Deref, DerefMut, Range},
    sync::{Arc, Mutex, MutexGuard, RwLock, RwLockReadGuard, RwLockWriteGuard},
};

use api_version::ApiV2;
use bytes::{BufMut, BytesMut};
use dashmap::{
    mapref::{entry::Entry, one::Ref},
    DashMap,
};
use kvproto::kvrpcpb::{Mutation, Op};
use rand::{
    distributions::Distribution,
    prelude::{IteratorRandom, SliceRandom, ThreadRng},
};
use tikv_util::{info, HandyRwLock};

use crate::{
    client::{ClusterClient, RefStore, RequestOptions, Result},
    table::TableMeta,
};

#[derive(Default)]
pub struct KeyspaceInnerLock {
    /// `write` is used to synchronize write and restore with verification
    /// operations.
    /// RwLock is used as writes are multi-threaded.
    pub(crate) write: RwLock<()>,
    /// `restore` is used to synchronize restore operations to the same keyspace
    /// (especially for restores with and without verification).
    pub(crate) restore: Mutex<()>,
}

#[derive(Clone, Default)]
pub struct KeyspaceManager {
    keyspaces: Arc<DashMap<u32 /* keyspace_id */, KeyspaceMeta>>,

    /// `backup_lock` is used to block all writes & restores for backup.
    ///
    /// Should always acquire this lock before acquiring any keyspace lock, to
    /// avoid deadlock.
    ///
    /// TODO: use MVCC ref store to eliminate this lock.
    backup_lock: Arc<RwLock<()>>,

    ref_stores: Arc<KeyspaceRefStores>,

    random: Arc<Mutex<RandomHelper>>,
}

impl KeyspaceManager {
    pub fn create_keyspaces(
        &self,
        keyspace_ids: &[u32],
        inner_key_off: usize,
        table_count: usize,
        need_shuffle: Option<&mut ThreadRng>,
    ) {
        for &keyspace_id in keyspace_ids {
            match self.keyspaces.entry(keyspace_id) {
                Entry::Occupied(_) => panic!("duplicated keyspace {}", keyspace_id),
                Entry::Vacant(entry) => {
                    entry.insert(KeyspaceMeta::new(inner_key_off, table_count));
                }
            }
        }

        self.random
            .lock()
            .unwrap()
            .add_keyspaces(keyspace_ids, need_shuffle);
    }

    pub fn get_uniform_random_keyspace(&self, rng: &mut ThreadRng) -> u32 {
        self.random.lock().unwrap().uniformly_choose(rng)
    }

    /// Get a random keyspace conforms zipf distribution.
    ///
    /// Note that zipf distribution generate sample from 1 to n conforms the
    /// probability mass function (see https://en.wikipedia.org/wiki/Zipf's_law).
    ///
    /// We need to shuffle keyspaces in advance before pick a random keyspace.
    /// Otherwise the first keyspace will always have the maximum chance.
    pub fn get_zipf_random_keyspace(&self, rng: &mut ThreadRng) -> u32 {
        self.random.lock().unwrap().zipf_choose(rng)
    }

    pub fn max_keyspace_id(&self) -> Option<u32> {
        self.keyspaces.iter().map(|item| *item.key()).max()
    }

    pub fn ref_stores(&self) -> &KeyspaceRefStores {
        &self.ref_stores
    }

    pub fn get_keyspace_meta(&self, keyspace_id: u32) -> Option<Ref<'_, u32, KeyspaceMeta>> {
        self.keyspaces.get(&keyspace_id)
    }

    pub fn get_random_available_table(
        &self,
        keyspace_id: u32,
        rng: &mut ThreadRng,
    ) -> Option<i64 /* table_id */> {
        self.get_keyspace_meta(keyspace_id)
            .unwrap()
            .get_random_available_table(rng)
    }
}

#[derive(Default)]
pub struct KeyspaceMeta {
    _inner_key_off: usize,
    inner_lock: Arc<KeyspaceInnerLock>,
    tables: DashMap<i64 /* table_id */, TableMeta>,
}

impl KeyspaceMeta {
    pub fn new(inner_key_off: usize, table_count: usize) -> Self {
        let tables = DashMap::default();
        for _ in 0..table_count {
            let table = TableMeta::new(true);
            tables.insert(table.id(), table);
        }
        Self {
            _inner_key_off: inner_key_off,
            inner_lock: Default::default(),
            tables,
        }
    }

    pub fn inner_lock(&self) -> Arc<KeyspaceInnerLock> {
        self.inner_lock.clone()
    }

    pub fn get_random_available_table(&self, rng: &mut ThreadRng) -> Option<i64> {
        self.tables
            .iter()
            .filter_map(|x| x.is_available().then_some(*x.key()))
            .choose(rng)
    }

    pub fn new_table(&self, is_available: bool) -> i64 {
        let table = TableMeta::new(is_available);
        let table_id = table.id();
        self.tables.insert(table.id(), table);
        table_id
    }

    pub fn get_table(&self, table_id: i64) -> Option<Ref<'_, i64, TableMeta>> {
        self.tables.get(&table_id)
    }
}

/// Helper to get locks for different workload.
/// Note: must keep order of locks acquiring to avoid deadlock.
pub struct KeyspaceLockHelper {
    backup_lock: Arc<RwLock<()>>,
    keyspace_inner: Arc<KeyspaceInnerLock>,
}

impl KeyspaceLockHelper {
    /// Get locks for write workload.
    /// Returns `None` if failed to acquire keyspace write lock. The caller can
    /// retry another keyspace.
    pub fn try_lock_for_write(&self) -> Option<(RwLockReadGuard<'_, ()>, RwLockReadGuard<'_, ()>)> {
        let backup_guard = self.backup_lock.rl();
        let write_guard = self.keyspace_inner.write.try_read();
        if let Ok(write_guard) = write_guard {
            return Some((backup_guard, write_guard));
        }
        None
    }

    /// Get locks for restore with verification workload.
    pub fn lock_for_restore_with_verify(
        &self,
    ) -> (
        RwLockReadGuard<'_, ()>,
        RwLockWriteGuard<'_, ()>,
        MutexGuard<'_, ()>,
    ) {
        let backup_guard = self.backup_lock.rl();
        let write_guard = self.keyspace_inner.write.wl();
        let restore_guard = self.keyspace_inner.restore.lock().unwrap();
        (backup_guard, write_guard, restore_guard)
    }

    /// Get locks for restore without verification workload.
    /// Do not acquire write lock so that write workloads are not blocked.
    pub fn lock_for_restore_without_verify(&self) -> (RwLockReadGuard<'_, ()>, MutexGuard<'_, ()>) {
        let backup_guard = self.backup_lock.rl();
        let restore_guard = self.keyspace_inner.restore.lock().unwrap();
        (backup_guard, restore_guard)
    }

    /// Get locks for `load_data` operation.
    /// Block backup, restore_with_verify, and restore_without_verify.
    pub fn lock_for_load_data(
        &self,
    ) -> (
        RwLockReadGuard<'_, ()>,
        RwLockReadGuard<'_, ()>,
        MutexGuard<'_, ()>,
    ) {
        let backup_guard = self.backup_lock.rl();
        let write_guard = self.keyspace_inner.write.rl();
        let restore_guard = self.keyspace_inner.restore.lock().unwrap();
        (backup_guard, write_guard, restore_guard)
    }

    /// Get locks for `destroy_table` operation.
    /// Block backup, writes, restore_with_verify, and restore_without_verify.
    /// Downgrade `write_guard` on keyspace to read lock after set table to
    /// unavailable.
    pub fn _lock_for_destroy_table(
        &self,
    ) -> (
        RwLockReadGuard<'_, ()>,
        RwLockWriteGuard<'_, ()>,
        MutexGuard<'_, ()>,
    ) {
        let backup_guard = self.backup_lock.rl();
        let write_guard = self.keyspace_inner.write.wl();
        let restore_guard = self.keyspace_inner.restore.lock().unwrap();
        (backup_guard, write_guard, restore_guard)
    }
}

impl KeyspaceManager {
    pub fn lock_for_backup(&self) -> RwLockWriteGuard<'_, ()> {
        self.backup_lock.wl()
    }

    pub fn get_keyspace_lock(&self, keyspace_id: u32) -> KeyspaceLockHelper {
        KeyspaceLockHelper {
            backup_lock: self.backup_lock.clone(),
            keyspace_inner: self.keyspaces.get(&keyspace_id).unwrap().inner_lock(),
        }
    }
}

/// `RandomHelper` helps to get a random keyspace.
///
/// Note that zipf distribution generate sample from 1 to n conforms the
/// probability mass function (see https://en.wikipedia.org/wiki/Zipf's_law).
///
/// We need to shuffle keyspaces in advance before choose randomly.
/// Otherwise the first keyspace will always have the maximum chance.
#[derive(Default)]
struct RandomHelper {
    shuffle_keyspace_ids: Vec<u32>,
    zipf: Option<zipf::ZipfDistribution>,
}

impl RandomHelper {
    pub fn add_keyspaces(&mut self, keyspace_ids: &[u32], need_shuffle: Option<&mut ThreadRng>) {
        for &keyspace_id in keyspace_ids {
            self.shuffle_keyspace_ids.push(keyspace_id);
        }
        if let Some(rng) = need_shuffle {
            self.shuffle_keyspace_ids.shuffle(rng);
        }

        // ZipfDistribution do not support to update `num_elements` after created.
        self.zipf =
            Some(zipf::ZipfDistribution::new(self.shuffle_keyspace_ids.len(), 1.03).unwrap());
    }

    pub fn zipf_choose(&self, rng: &mut ThreadRng) -> u32 {
        let zipf = self
            .zipf
            .unwrap_or_else(|| panic!("add keyspace before choose"));
        self.shuffle_keyspace_ids[zipf.sample(rng) - 1]
    }

    pub fn uniformly_choose(&self, rng: &mut ThreadRng) -> u32 {
        self.shuffle_keyspace_ids
            .iter()
            .choose(rng)
            .copied()
            .unwrap()
    }
}

pub struct ClusterKeyspaceClient {
    pub inner: ClusterClient,
    keyspace_manager: KeyspaceManager,
}

impl Deref for ClusterKeyspaceClient {
    type Target = ClusterClient;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl DerefMut for ClusterKeyspaceClient {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl ClusterKeyspaceClient {
    pub fn new(inner: ClusterClient, keyspace_manager: KeyspaceManager) -> Self {
        Self {
            inner,
            keyspace_manager,
        }
    }

    pub fn keyspace_manager(&self) -> &KeyspaceManager {
        &self.keyspace_manager
    }

    pub fn keyspace_put_kv<F, G>(
        &mut self,
        keyspace_id: u32,
        table_id: i64,
        rng: Range<usize>,
        gen_user_key: F,
        gen_val: G,
    ) -> Result<()>
    where
        F: Fn(usize) -> Vec<u8>,
        G: Fn(usize) -> Vec<u8>,
    {
        let mut mutations = vec![];
        for i in rng {
            let mut m = Mutation::default();
            m.set_op(Op::Put);
            m.set_key(make_key(keyspace_id, table_id, &gen_user_key(i)));
            m.set_value(gen_val(i));
            mutations.push(m)
        }
        self.kv_mutate(mutations.clone())?;

        self.keyspace_manager
            .ref_stores()
            .keyspace_put_kv(keyspace_id, mutations);
        Ok(())
    }

    pub fn keyspace_del_kv<F>(
        &mut self,
        keyspace_id: u32,
        table_id: i64,
        rng: Range<usize>,
        gen_user_key: F,
    ) -> Result<()>
    where
        F: Fn(usize) -> Vec<u8>,
    {
        let mut mutations = vec![];
        for i in rng {
            let mut m = Mutation::default();
            m.set_op(Op::Del);
            m.set_key(make_key(keyspace_id, table_id, &gen_user_key(i)));
            mutations.push(m)
        }
        self.kv_mutate(mutations.clone())?;

        self.keyspace_manager
            .ref_stores()
            .keyspace_del_kv(keyspace_id, mutations);
        Ok(())
    }

    pub fn verify_keyspace_with_ref_store(&mut self, keyspace_id: u32) -> Result<usize> {
        let ref_store = self
            .keyspace_manager
            .ref_stores()
            .get_keyspace_ref_store(keyspace_id);
        let ref_store = ref_store.lock().unwrap().clone();
        self.verify_data_with_given_ref_store(&ref_store, None, &RequestOptions::default())
    }

    pub fn verify_all_keyspaces(&mut self) -> Result<usize> {
        let mut cnt = 0;
        for keyspace_id in self.keyspace_manager.ref_stores().all_keyspace_ids() {
            cnt += self.verify_keyspace_with_ref_store(keyspace_id)?;
        }
        Ok(cnt)
    }
}

const TABLE_PREFIX: &[u8] = b"t";
const RECORD_PREFIX_SEP: &[u8] = b"_r";

// Ref: tidb_query_datatype::codec::table::append_table_record_prefix
pub fn make_key(keyspace_id: u32, table_id: i64, user_key: &[u8]) -> Vec<u8> {
    let mut buf = BytesMut::with_capacity(
        4 + TABLE_PREFIX.len()
            + mem::size_of_val(&table_id)
            + RECORD_PREFIX_SEP.len()
            + user_key.len(),
    );
    buf.extend_from_slice(&ApiV2::get_txn_keyspace_prefix(keyspace_id));
    buf.put_slice(TABLE_PREFIX);
    buf.put_i64(table_id);
    buf.put_slice(RECORD_PREFIX_SEP);
    buf.put_slice(user_key);
    buf.freeze().to_vec()
}

pub struct KeyspaceBackup {
    pub backup_ts: u64,
    pub ref_stores: HashMap<u32 /* keyspace_id */, RefStore>,
}

#[derive(Default)]
pub struct KeyspaceRefStores {
    ref_stores: DashMap<u32 /* keyspace_id */, Arc<Mutex<RefStore>>>,
    backups: DashMap<String /* backup_name */, KeyspaceBackup>,
}

impl KeyspaceRefStores {
    pub fn get_keyspace_ref_store(&self, keyspace_id: u32) -> Arc<Mutex<RefStore>> {
        match self.ref_stores.entry(keyspace_id) {
            Entry::Occupied(e) => e.get().clone(),
            Entry::Vacant(e) => {
                let ref_store = Arc::new(Mutex::new(RefStore::default()));
                e.insert(ref_store.clone());
                ref_store
            }
        }
    }

    pub fn dump(&self) -> HashMap<u32, RefStore> {
        HashMap::from_iter(self.ref_stores.iter().map(|r| {
            let (&keyspace_id, ref_store) = r.pair();
            let ref_store = ref_store.lock().unwrap();
            (keyspace_id, ref_store.clone())
        }))
    }

    pub fn add_backup(
        &self,
        backup_name: String,
        backup_ts: u64,
        ref_stores: HashMap<u32, RefStore>,
    ) {
        self.backups.insert(
            backup_name,
            KeyspaceBackup {
                backup_ts,
                ref_stores,
            },
        );
    }

    pub fn backup(&self, backup_name: String, backup_ts: u64) {
        let ref_stores = self.dump();
        self.add_backup(backup_name, backup_ts, ref_stores);
    }

    pub fn get_random_backup(
        &self,
        rng: &mut ThreadRng,
    ) -> Option<(String /* backup_name */, u64 /* backup_ts */)> {
        self.backups
            .iter()
            .choose(rng)
            .map(|r| (r.key().clone(), r.backup_ts))
    }

    pub fn restore_keyspace(
        &self,
        source_keyspace_id: u32,
        backup_name: &str,
        target_keyspace_id: u32,
    ) {
        let backup = self.backups.get(backup_name).unwrap();
        let keyspace_backup = backup.ref_stores.get(&source_keyspace_id);
        let target_ref_store = self.get_keyspace_ref_store(target_keyspace_id);
        let mut target_ref_store = target_ref_store.lock().unwrap();
        if let Some(keyspace_backup) = keyspace_backup {
            *target_ref_store = keyspace_backup.clone();
        } else {
            // Keyspace will not be found in backup if it has never been written after
            // created.
            info!(
                "keyspace {} not found in backup {}, set target_ref_store to all none",
                source_keyspace_id, backup_name
            );
            for v in target_ref_store.values_mut() {
                *v = None;
            }
        }
    }

    pub fn keyspace_put_kv(&self, keyspace_id: u32, mutations: Vec<Mutation>) {
        let ref_store = self.get_keyspace_ref_store(keyspace_id);
        let mut ref_store = ref_store.lock().unwrap();
        for mut m in mutations {
            ref_store.put_kv(m.take_key(), m.take_value());
        }
    }

    pub fn keyspace_del_kv(&self, keyspace_id: u32, mutations: Vec<Mutation>) {
        let ref_store = self.get_keyspace_ref_store(keyspace_id);
        let mut ref_store = ref_store.lock().unwrap();
        for mut m in mutations {
            ref_store.del_kv(m.take_key());
        }
    }

    pub fn all_keyspace_ids(&self) -> Vec<u32> {
        self.ref_stores.iter().map(|r| *r.key()).collect()
    }

    pub fn ingest(&self, keyspace_id: u32, ref_store: RefStore) {
        let target_ref_store = self.get_keyspace_ref_store(keyspace_id);
        let mut target_ref_store = target_ref_store.lock().unwrap();
        target_ref_store.ingest(ref_store);
    }
}
