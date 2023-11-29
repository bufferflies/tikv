// Copyright 2023 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    fmt, mem, ops,
    ops::{Deref, DerefMut, Range},
    sync::{Arc, Mutex, MutexGuard},
    time::Duration,
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
use tikv_util::info;
use tokio::sync::{RwLock, RwLockReadGuard, RwLockWriteGuard};

use crate::{
    client::{ClusterTxnClient, RefStore, Result},
    table::TableMeta,
};

#[derive(Clone, Default)]
pub struct KeyspaceManager {
    core: Arc<KeyspaceManagerCore>,
}

impl Deref for KeyspaceManager {
    type Target = KeyspaceManagerCore;

    fn deref(&self) -> &Self::Target {
        &self.core
    }
}

#[derive(Default)]
pub struct KeyspaceManagerCore {
    keyspaces: DashMap<u32 /* keyspace_id */, KeyspaceMeta>,
    ref_stores: KeyspaceRefStores,
    backups: Mutex<Vec<KeyspaceBackup>>,
    random: Mutex<RandomHelper>,
}

impl KeyspaceManagerCore {
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

    // Note: use with caution.
    // The returned value will keep a read lock on `self.keyspaces`.
    pub fn get_keyspace_meta(&self, keyspace_id: u32) -> Option<Ref<'_, u32, KeyspaceMeta>> {
        self.keyspaces.get(&keyspace_id)
    }

    fn set_keyspace_meta(&self, keyspace_id: u32, meta: KeyspaceMetaCore) {
        let mut keyspace = self.keyspaces.get_mut(&keyspace_id).unwrap();
        assert_eq!(keyspace.inner_key_off, meta.inner_key_off);

        info!(
            "KeyspaceManager.set_keyspace_meta, keyspace_id {}, old {:?}, new {:?}",
            keyspace_id, keyspace.core, meta
        );

        keyspace.core = meta;
    }

    pub fn get_random_available_table(
        &self,
        keyspace_id: u32,
        rng: &mut ThreadRng,
    ) -> Option<i64 /* table_id */> {
        self.keyspaces
            .get(&keyspace_id)
            .unwrap()
            .get_random_available_table(rng)
    }

    pub fn add_backup(&self, backup: KeyspaceBackup) {
        self.backups.lock().unwrap().push(backup);
    }

    pub fn get_random_backup(&self, rng: &mut ThreadRng) -> Option<KeyspaceBackup> {
        let backups = self.backups.lock().unwrap();
        backups.iter().choose(rng).cloned()
    }

    pub fn backup_keyspace(&self, keyspace_id: u32, backup_ts: u64) -> KeyspaceBackup {
        let ref_store = self.ref_stores.dump(keyspace_id);
        let meta = self.keyspaces.get(&keyspace_id).unwrap().core.clone();
        KeyspaceBackup {
            backup_name: None,
            backup_ts,
            keyspace_id,
            ref_store,
            meta,
        }
    }

    pub fn restore_keyspace(&self, tag: &str, backup: KeyspaceBackup, target_keyspace: u32) {
        let KeyspaceBackup {
            ref_store, meta, ..
        } = backup;
        self.set_keyspace_meta(target_keyspace, meta);
        self.ref_stores
            .restore_keyspace(tag, ref_store, target_keyspace);
    }
}

pub struct KeyspaceMeta {
    core: KeyspaceMetaCore,
    inner_lock: Arc<RwLock<()>>,
    extra_lock: Arc<Mutex<()>>,
}

impl ops::Deref for KeyspaceMeta {
    type Target = KeyspaceMetaCore;

    fn deref(&self) -> &Self::Target {
        &self.core
    }
}

impl ops::DerefMut for KeyspaceMeta {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.core
    }
}

/// Used for keyspace meta backup & restore without locks.
#[derive(Clone)]
pub struct KeyspaceMetaCore {
    inner_key_off: usize,
    tables: DashMap<i64 /* table_id */, TableMeta>,

    /// Used to verify data considering the delete prefixes state of shard.
    ///
    /// As the operation of destroy range is async, in some scene the data
    /// within the range to be destroyed is not determined.
    del_prefixes: kvengine::DeletePrefixes,
}

impl fmt::Debug for KeyspaceMetaCore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KeyspaceMetaCore")
            .field("inner_key_off", &self.inner_key_off)
            .field(
                "tables",
                &self
                    .tables
                    .iter()
                    .map(|x| (*x.key(), x.value().is_available()))
                    .collect::<Vec<_>>(),
            )
            .field("del_prefixes", &self.del_prefixes)
            .finish()
    }
}

impl KeyspaceMeta {
    pub fn new(inner_key_off: usize, table_count: usize) -> Self {
        let tables = DashMap::default();
        for _ in 0..table_count {
            let table = TableMeta::new(true);
            tables.insert(table.id(), table);
        }
        Self {
            core: KeyspaceMetaCore {
                inner_key_off,
                tables,
                del_prefixes: kvengine::DeletePrefixes::new_with_inner_key_off(inner_key_off),
            },
            inner_lock: Default::default(),
            extra_lock: Default::default(),
        }
    }

    pub fn get_locks(&self) -> (Arc<RwLock<()>>, Arc<Mutex<()>>) {
        (self.inner_lock.clone(), self.extra_lock.clone())
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

/// Helper to get locks for different workloads.
///
/// Shared (read) lock for: reads/writes, restore (without verification),
/// load_data.
///
/// Mutual-exclusive (write) lock for: backup, restore (with verification),
/// destroy_table.
///
/// For load_data:
/// * Data is ingested to new table, and the new table is not available until
///   the ingest finished. So load_data doesn't need to be mutual-exclusive with
///   reads/writes.
/// * Besides, acquire extra_lock to be mutual-exclusive with major_compaction.
///
/// For destroy_table: downgrade to shared lock after set table to unavailable.
///
/// For major_compaction: acquire extra_lock to be mutual-exclusive with
/// load_data.
pub struct KeyspaceLockHelper {
    inner: Arc<RwLock<()>>,
    extra: Arc<Mutex<()>>,
}

impl KeyspaceLockHelper {
    pub fn try_shared_lock(&self) -> Option<RwLockReadGuard<'_, ()>> {
        self.inner.try_read().ok()
    }

    pub async fn shared_lock(&self) -> RwLockReadGuard<'_, ()> {
        self.inner.read().await
    }

    pub async fn mutex_lock(&self) -> RwLockWriteGuard<'_, ()> {
        self.inner.write().await
    }

    pub fn extra_lock(&self) -> MutexGuard<'_, ()> {
        self.extra.lock().unwrap()
    }
}

impl KeyspaceManager {
    pub fn get_keyspace_lock(&self, keyspace_id: u32) -> KeyspaceLockHelper {
        let (inner, extra) = self.keyspaces.get(&keyspace_id).unwrap().get_locks();
        KeyspaceLockHelper { inner, extra }
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
    pub inner: ClusterTxnClient,
    keyspace_manager: KeyspaceManager,
}

impl Deref for ClusterKeyspaceClient {
    type Target = ClusterTxnClient;

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
    pub fn new(inner: ClusterTxnClient, keyspace_manager: KeyspaceManager) -> Self {
        Self {
            inner,
            keyspace_manager,
        }
    }

    pub fn keyspace_manager(&self) -> &KeyspaceManager {
        &self.keyspace_manager
    }

    pub async fn keyspace_put_kv<F, G>(
        &self,
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
        self.kv_mutate(mutations.clone(), Duration::from_secs(30))
            .await?;

        self.keyspace_manager
            .ref_stores()
            .keyspace_put_kv(keyspace_id, mutations);
        Ok(())
    }

    pub async fn keyspace_del_kv<F>(
        &self,
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
        self.kv_mutate(mutations.clone(), Duration::from_secs(30))
            .await?;

        self.keyspace_manager
            .ref_stores()
            .keyspace_del_kv(keyspace_id, mutations);
        Ok(())
    }

    pub async fn verify_keyspace(&mut self, keyspace_id: u32) -> Result<usize> {
        let ref_store = self
            .keyspace_manager
            .ref_stores()
            .get_keyspace_ref_store(keyspace_id);
        let ref_store = ref_store.lock().unwrap().clone();
        let (start_key, end_key) = ApiV2::get_txn_keyspace_range(keyspace_id);
        self.verify_data_by_scan(&ref_store, Some((&start_key, &end_key)))
            .await
    }

    pub async fn verify_all_keyspaces(&mut self) -> Result<usize> {
        let mut cnt = 0;
        for keyspace_id in self.keyspace_manager.ref_stores().all_keyspace_ids() {
            cnt += self.verify_keyspace(keyspace_id).await?;
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

#[derive(Clone)]
pub struct KeyspaceBackup {
    pub backup_name: Option<String>,
    pub backup_ts: u64,
    pub keyspace_id: u32,
    pub ref_store: RefStore,
    pub meta: KeyspaceMetaCore,
}

impl KeyspaceBackup {
    pub fn backup_name(&self) -> &str {
        self.backup_name.as_ref().unwrap()
    }
}

#[derive(Default)]
pub struct KeyspaceRefStores {
    ref_stores: DashMap<u32 /* keyspace_id */, Arc<Mutex<RefStore>>>,
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

    pub fn dump(&self, keyspace_id: u32) -> RefStore {
        self.get_keyspace_ref_store(keyspace_id)
            .lock()
            .unwrap()
            .clone()
    }

    pub fn restore_keyspace(&self, tag: &str, ref_store: RefStore, target_keyspace_id: u32) {
        if ref_store.is_empty() {
            info!("{} ref store is empty in backup", tag);
        }
        let target_ref_store = self.get_keyspace_ref_store(target_keyspace_id);
        let mut target_ref_store = target_ref_store.lock().unwrap();
        *target_ref_store = ref_store;
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
