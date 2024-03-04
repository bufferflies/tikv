// Copyright 2022 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    fmt::{Display, Formatter, Write},
    fs,
    fs::Metadata,
    path::{Path, PathBuf},
    sync::Arc,
    time::Duration,
};

use collections::HashSet;
use kvengine::table::sstable;
use kvproto::import_sstpb::SwitchMode;
use sst_importer::SstImporter;
use tikv_util::{error, info, warn, worker::Runnable};

pub struct GcTask {}

impl Display for GcTask {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "GcTask")
    }
}

/// The GC worker periodically removes unused sst files to release storage
/// resource.
pub struct GcRunner {
    kv: kvengine::Engine,
    importer: Arc<SstImporter>,
    timeout: Duration,
}

impl Runnable for GcRunner {
    type Task = GcTask;

    fn run(&mut self, _: GcTask) {
        if let Err(err) = self.gc_kv_files() {
            error!("local file gc kv files failed {:?}", err);
        }
        if let Err(err) = self.gc_importer_files() {
            error!("local file gc importer files failed {:?}", err);
        }
    }
}

impl GcRunner {
    pub fn new(kv: kvengine::Engine, importer: Arc<SstImporter>, timeout: Duration) -> Self {
        Self {
            kv,
            importer,
            timeout,
        }
    }

    fn gc_kv_files(&self) -> kvengine::Result<()> {
        let kv_file_ids = self.collect_kv_file_ids();
        let blacklist_file_ids = self.kv.get_files_in_blacklist();
        let kv_txn_chunk_ids = self.collect_kv_txn_chunk_ids();
        self.remove_kv_garbage_files(&kv_file_ids, blacklist_file_ids.as_ref(), &kv_txn_chunk_ids)?;
        Ok(())
    }

    fn collect_kv_file_ids(&self) -> HashSet<u64> {
        loop {
            if let Some(all_file_ids) = self.try_collect_kv_file_ids() {
                return all_file_ids;
            }
        }
    }

    fn try_collect_kv_file_ids(&self) -> Option<HashSet<u64>> {
        let shard_id_vers = self.kv.get_all_shard_id_vers();
        let mut all_file_ids = HashSet::default();
        for &id_ver in &shard_id_vers {
            let shard = self.kv.get_shard_with_ver(id_ver.id, id_ver.ver).ok()?;
            all_file_ids.extend(shard.get_all_files());
        }
        Some(all_file_ids)
    }

    fn collect_kv_txn_chunk_ids(&self) -> HashSet<u64> {
        loop {
            if let Some(all_file_ids) = self.try_collect_kv_txn_chunk_ids() {
                return all_file_ids;
            }
        }
    }

    fn try_collect_kv_txn_chunk_ids(&self) -> Option<HashSet<u64>> {
        let shard_id_vers = self.kv.get_all_shard_id_vers();
        let mut all_file_ids = HashSet::default();
        for &id_ver in &shard_id_vers {
            let shard = self.kv.get_shard_with_ver(id_ver.id, id_ver.ver).ok()?;
            all_file_ids.extend(shard.get_txn_chunks());
        }
        Some(all_file_ids)
    }

    fn remove_kv_garbage_files(
        &self,
        kv_file_ids: &HashSet<u64>,
        blacklist_file_ids: &HashSet<u64>,
        txn_chunk_ids: &HashSet<u64>,
    ) -> kvengine::Result<()> {
        let store_id = self.kv.get_engine_id();
        let entries = fs::read_dir(&self.kv.opts.local_dir)?;
        for e in entries {
            let entry = e?;
            let path = entry.path();
            if path.is_dir() && path.ends_with(".txn") {
                self.remove_kv_garbage_txn_files(path, txn_chunk_ids)?;
                continue;
            }
            let path_str = path.to_str().unwrap();
            if path_str.ends_with(".tmp") {
                let meta = entry.metadata()?;
                if !self.is_old_file(meta) {
                    continue;
                }
                Self::remove_file(store_id, &path)?;
            } else if path_str.ends_with(".sst") {
                let id = sstable::parse_file_id(&path)?;
                if !kv_file_ids.contains(&id) {
                    let _guard = self.kv.lock_file(id);
                    if blacklist_file_ids.contains(&id) {
                        continue;
                    }
                    let meta = fs::metadata(&path)?;
                    if self.is_old_file(meta) {
                        Self::remove_file(store_id, &path)?;
                    }
                }
            } else if !path_str.ends_with("LOCK") {
                warn!("unexpected file {:?}", path);
            }
        }
        Ok(())
    }

    fn remove_file(store_id: u64, file: &Path) -> std::io::Result<()> {
        info!("{} local file GC remove file {:?}", store_id, file);
        fs::remove_file(file)
    }

    fn gc_importer_files(&self) -> sst_importer::Result<()> {
        if self.importer.get_mode() == SwitchMode::Import {
            return Ok(());
        }
        let store_id = self.kv.get_engine_id();
        let ssts = self.importer.list_ssts()?;
        for sst_meta in &ssts {
            let path = self.importer.get_path(sst_meta);
            let meta = fs::metadata(&path)?;
            if self.is_old_file(meta) {
                self.importer.delete(sst_meta)?;
                let mut uuid = String::new();
                for &b in sst_meta.get_uuid() {
                    write!(uuid, "{:X}", b).expect("Unable to write");
                }
                info!(
                    "{} gc runner delete sst uuid {} file {:?} timeout {:?}",
                    store_id, uuid, path, self.timeout
                );
            }
        }
        Ok(())
    }

    fn remove_kv_garbage_txn_files(
        &self,
        txn_path: PathBuf,
        kv_txn_chunk_ids: &HashSet<u64>,
    ) -> kvengine::Result<()> {
        let store_id = self.kv.get_engine_id();
        let entries = fs::read_dir(txn_path.as_path())?;
        let txn_chunk_manager = self.kv.get_txn_chunk_manager();
        for e in entries {
            let entry = e?;
            let path = entry.path();
            let meta = entry.metadata()?;
            let filename = path
                .file_name()
                .unwrap_or_default()
                .to_str()
                .unwrap_or_default();
            if filename.ends_with(".tmp") {
                if !self.is_old_file(meta) {
                    continue;
                }
                Self::remove_file(store_id, &path)?;
            } else if filename.ends_with(".txn") {
                if let Some(id) = kvengine::txn_chunk_manager::parse_txn_chunk_id(filename) {
                    if !kv_txn_chunk_ids.contains(&id) {
                        let _guard = self.kv.lock_file(id);
                        if self.is_old_file(meta) {
                            txn_chunk_manager.remove(id);
                        }
                    }
                } else {
                    warn!("failed to parse txn file id {:?}", path);
                }
            } else {
                warn!("unexpected file {:?}", path);
            }
        }
        Ok(())
    }

    fn is_old_file(&self, meta: Metadata) -> bool {
        let modified = meta.modified().unwrap();
        let dur = modified.elapsed().unwrap_or_default();
        dur > self.timeout
    }
}
