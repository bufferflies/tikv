// Copyright 2021 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    collections::HashMap,
    fs,
    io::{Read, Seek, SeekFrom, Write},
    iter::Iterator,
    ops::Deref,
    os::unix::fs::{FileExt, MetadataExt},
    path::PathBuf,
    sync::{
        atomic::{AtomicU64, Ordering::Relaxed},
        Arc,
    },
};

use bytes::{Buf, Bytes};
use cloud_encryption::EncryptionKey;
use dashmap::mapref::entry::Entry;
use file_system::{IoOp, IoType};
use kvenginepb::{TxnFileRef, TxnFileRefs};
use protobuf::Message;
use schema::schema::StorageClass;
use tikv_util::{mpsc::Receiver, time::Instant};
use tokio::sync::OwnedSemaphorePermit;

use crate::{
    apply::ChangeSet,
    context::IaCtx,
    dfs::FileType,
    error::IoContext,
    ia::{
        ia_file::{table_meta_file_local_path, IaFile},
        manager::IaManager,
    },
    metrics::ENGINE_LEVEL_WRITE_VEC,
    table::{
        columnar::SchemaFile,
        file::{FdCache, File, InMemFile, LocalFile},
        sstable::SsTable,
        BoundedDataSet,
    },
    EngineCore, *,
};

pub const LOAD_FILE_CONCURRENCY: usize = 8;

impl EngineCore {
    pub fn prepare_change_set(
        &self,
        cs: kvenginepb::ChangeSet,
        use_direct_io: bool,
        shard_use_ia: bool, // The shard_use_ia means shard's storage class is IA.
        reload_snap: Option<kvenginepb::Snapshot>, /* The snap contains the current files that
                             * need to be reloaded. */
        table_filter: Option<LoadTableFilterFn>,
        encryption_key: Option<EncryptionKey>, /* encryption_key will be ignored if cs is
                                                * snapshot or restore_shard */
    ) -> Result<ChangeSet> {
        let mut ids: HashMap<u64, FileMeta> = HashMap::new();
        let mut lock_txn_file_refs: Vec<TxnFileRef> = vec![];
        let mut cs = ChangeSet::new(cs);
        let mut snap = None;
        let mut schema_meta = None;
        if cs.has_flush() {
            let flush = cs.get_flush();
            if flush.has_l0_create() {
                ids.insert(
                    flush.get_l0_create().id,
                    FileMeta::from_l0_table(flush.get_l0_create()),
                );
            }
            for l0 in flush.get_l0_creates() {
                ids.insert(l0.get_id(), FileMeta::from_l0_table(l0));
            }
        }
        if cs.has_compaction() {
            let comp = cs.get_compaction();
            if !is_move_down(comp) {
                for tbl in &comp.table_creates {
                    ids.insert(tbl.id, FileMeta::from_table(tbl));
                }
                for bt in comp.get_blob_tables() {
                    ids.insert(bt.get_id(), FileMeta::from_blob_table(bt));
                }
            } else if comp.level == 0 && shard_use_ia {
                for tbl in &comp.table_creates {
                    ids.insert(tbl.id, FileMeta::from_table(tbl));
                }
            }
        }
        if cs.has_destroy_range() {
            for t in cs.get_destroy_range().get_table_creates() {
                ids.insert(t.id, FileMeta::from_table(t));
            }
            for col in cs.get_destroy_range().get_columnar_creates() {
                ids.insert(col.id, FileMeta::from_columnar_table(col));
            }
        }
        if cs.has_truncate_ts() {
            for t in cs.get_truncate_ts().get_table_creates() {
                ids.insert(t.id, FileMeta::from_table(t));
            }
            for col in cs.get_truncate_ts().get_columnar_creates() {
                ids.insert(col.id, FileMeta::from_columnar_table(col));
            }
        }
        if cs.has_trim_over_bound() {
            for t in cs.get_trim_over_bound().get_table_creates() {
                ids.insert(t.id, FileMeta::from_table(t));
            }
            for col in cs.get_trim_over_bound().get_columnar_creates() {
                ids.insert(col.id, FileMeta::from_columnar_table(col));
            }
        }
        if cs.has_snapshot() {
            snap = Some(cs.get_snapshot());
        }
        if cs.has_initial_flush() {
            let snap = cs.get_initial_flush();
            self.collect_snap_ids(snap, &mut ids);
            // Load schema file when prepare initial_flush
            if snap.has_schema_meta() {
                schema_meta = Some(snap.get_schema_meta());
            }
        }
        if cs.has_restore_shard() {
            snap = Some(cs.get_restore_shard());
        }
        if cs.has_ingest_files() {
            let ingest_files = cs.get_ingest_files();
            for l0 in ingest_files.get_l0_creates() {
                ids.insert(l0.id, FileMeta::from_l0_table(l0));
            }
            for tbl in ingest_files.get_table_creates() {
                ids.insert(tbl.id, FileMeta::from_table(tbl));
            }
            for blob in ingest_files.get_blob_creates() {
                ids.insert(blob.id, FileMeta::from_blob_table(blob));
            }
        }
        if cs.has_major_compaction() {
            let major_comp = cs.get_major_compaction();
            for tbl in major_comp.get_sstable_change().get_table_creates() {
                ids.insert(tbl.get_id(), FileMeta::from_table(tbl));
            }
            for blob in major_comp.get_new_blob_tables() {
                ids.insert(blob.get_id(), FileMeta::from_blob_table(blob));
            }
        }
        if cs.has_update_schema_meta() {
            schema_meta = Some(cs.get_update_schema_meta());
        }
        if let Some(reload_snap) = &reload_snap {
            self.collect_snap_ids(reload_snap, &mut ids);
        }
        if cs.has_columnar_compaction() {
            let columnar_comp = cs.get_columnar_compaction();
            for tbl in columnar_comp.get_columnar_change().get_columnar_creates() {
                ids.insert(tbl.get_id(), FileMeta::from_columnar_table(tbl));
            }
        }
        if cs.has_update_vector_index() {
            let update_vec_idx = cs.get_update_vector_index();
            for vec_idx_file in update_vec_idx.get_added() {
                ids.insert(
                    vec_idx_file.get_id(),
                    FileMeta::from_vector_index_file(vec_idx_file),
                );
            }
        }
        let mut encryption_key = encryption_key;
        let mut shard_use_ia = shard_use_ia;
        if let Some(snap) = snap {
            self.collect_snap_ids(snap, &mut ids);
            lock_txn_file_refs.extend(collect_snap_lock_txn_file_refs(snap));
            encryption_key = get_shard_property(ENCRYPTION_KEY, snap.get_properties())
                .map(|v| self.master_key.decrypt_encryption_key(&v).unwrap());
            let sc = StorageClass::unmarshal(
                get_shard_property(STORAGE_CLASS_KEY, snap.get_properties()).as_deref(),
            );
            shard_use_ia = sc == StorageClass::Ia;
            if snap.has_schema_meta() {
                schema_meta = Some(snap.get_schema_meta());
            }
        }
        let schema_file = if let Some(schema_meta) = schema_meta {
            let schema_file_id = schema_meta.get_file_id();
            if schema_file_id == 0 {
                // Set schema_file to None if schema file id is 0, no need to mark tombstone.
                None
            } else {
                Some(
                    self.fs
                        .get_runtime()
                        .block_on(self.load_schema_file(schema_file_id))?,
                )
            }
        } else {
            None
        };

        let tag = ShardTag::new(self.get_engine_id(), IdVer::from_change_set(&cs));
        if let Some(table_filter) = table_filter {
            info!(
                "{} is preparing change set (before table filter)", tag;
                "ids" => ?ids.keys(),
            );
            cs.unloaded_tables = ids
                .extract_if(|_, tb| !table_filter(cs.shard_id, tb)) // !table_filter: table will not load
                .collect();
        }

        if self.opts.ignore_columnar_table_load {
            ids.retain(|_, meta| {
                meta.file_type != FileType::Columnar && meta.file_type != FileType::VectorIndex
            });
        }

        info!(
            "{} is preparing change set, loading file by ids", tag;
            "ids" => ?ids.keys(),
            "encryption_key" => ?encryption_key,
            "shard_use_ia" => shard_use_ia,
        );
        self.load_tables_by_ids(
            tag,
            cs.shard_id,
            cs.shard_ver,
            &ids,
            &mut cs,
            use_direct_io,
            shard_use_ia,
            encryption_key.clone(),
        )?;

        if !lock_txn_file_refs.is_empty() {
            cs.lock_txn_files = self.txn_chunk_mgr.load_txn_files_from_refs(
                cs.shard_id,
                cs.shard_ver,
                &lock_txn_file_refs,
                encryption_key.clone(),
            )?;
            info!(
                "{} is preparing change set, load lock txn files", tag;
                "lock_txn_files" => ?cs.lock_txn_files,
                "encryption_key" => ?encryption_key,
            );
        }
        cs.schema_file = schema_file;
        Ok(cs)
    }

    fn collect_snap_ids(&self, snap: &kvenginepb::Snapshot, ids: &mut HashMap<u64, FileMeta>) {
        for blob in snap.get_blob_creates() {
            ids.insert(blob.id, FileMeta::from_blob_table(blob));
        }
        for l0 in snap.get_l0_creates() {
            ids.insert(l0.id, FileMeta::from_l0_table(l0));
        }
        for ln in snap.get_table_creates() {
            ids.insert(ln.id, FileMeta::from_table(ln));
        }
        for col in snap.get_columnar_creates() {
            ids.insert(col.id, FileMeta::from_columnar_table(col));
        }
        for vec_idx in snap.get_vector_indexes() {
            for vec_idx_file in vec_idx.get_files() {
                ids.insert(
                    vec_idx_file.id,
                    FileMeta::from_vector_index_file(vec_idx_file),
                );
            }
        }
    }

    fn load_tables_by_ids(
        &self,
        tag: ShardTag,
        shard_id: u64,
        shard_ver: u64,
        ids: &HashMap<u64, FileMeta>,
        cs: &mut ChangeSet,
        use_direct_io: bool,
        shard_use_ia: bool,
        encryption_key: Option<EncryptionKey>,
    ) -> Result<()> {
        let (result_tx, result_rx) = tikv_util::mpsc::bounded(ids.len());
        let runtime = self.fs.get_runtime();
        let opts = dfs::Options::default().with_shard(shard_id, shard_ver);
        let mut msg_count = 0;
        for (&id, fm) in ids {
            let fs = self.fs.clone();
            let use_ia_file = self.ia_ctx.is_enabled() && fm.use_ia(shard_use_ia);
            let ia_ctx = if use_ia_file {
                match self.ia_ctx.clone() {
                    IaCtx::Disabled => {
                        unreachable!() // Checked before. It won't happen.
                    }
                    IaCtx::Enabled(ia_mgr, data_dir) => {
                        let meta_file_path =
                            table_meta_file_local_path(id, fm.file_type, data_dir.deref());
                        let mut table_meta_file = self.open_local_file_with_file_path(
                            id,
                            ia_mgr.get_meta_fd_cache(),
                            meta_file_path.clone(),
                        );
                        if table_meta_file.is_err()
                            && let Ok(local_file) = self.open_local_file(id, fm.file_type)
                        {
                            let table_meta_off = fm.table_meta_off as u64;
                            // Read table meta from local sst file.
                            if let Ok(data) = local_file
                                .read(
                                    table_meta_off,
                                    local_file.size() as usize - table_meta_off as usize,
                                )
                                .map_err(Into::into)
                                .and_then(|data| validate_table_meta_off(tag, id, fm, data))
                            {
                                self.write_local_file_with_file_path(
                                    id,
                                    data,
                                    use_direct_io,
                                    meta_file_path.clone(),
                                )?;
                                table_meta_file = self.open_local_file_with_file_path(
                                    id,
                                    ia_mgr.get_meta_fd_cache(),
                                    meta_file_path,
                                );
                            };
                        }
                        if let Ok(table_meta_file) = table_meta_file {
                            if let Ok(ia_file) =
                                IaFile::open(id, fm.file_type, Arc::new(table_meta_file), ia_mgr)
                            {
                                cs.add_file(
                                    id,
                                    Arc::new(ia_file),
                                    fm,
                                    self.cache.clone(),
                                    encryption_key.clone(),
                                )?;
                                continue;
                            }
                        };
                    }
                };
                self.ia_ctx.clone()
            } else {
                if let Ok(local_file) = self.open_local_file(id, fm.file_type) {
                    cs.add_file(
                        id,
                        Arc::new(local_file),
                        fm,
                        self.cache.clone(),
                        encryption_key.clone(),
                    )?;
                    continue;
                }
                IaCtx::Disabled
            };
            let tx = result_tx.clone();
            let fm = fm.clone();
            let dfs_load_limiter = self.dfs_load_limiter.clone();
            runtime.spawn(async move {
                let permit = dfs_load_limiter.acquire_permit().await;
                let res = match ia_ctx {
                    IaCtx::Disabled => fs
                        .read_file(id, opts.with_type(fm.file_type))
                        .await
                        .map_err(Into::into)
                        .map(|data| (data, None)),
                    IaCtx::Enabled(ia_mgr, data_dir) => {
                        let opts = opts
                            .with_type(fm.file_type)
                            .with_start_off(fm.table_meta_off as u64);
                        fs.read_file(id, opts)
                            .await
                            .map_err(Into::into)
                            .and_then(|data| validate_table_meta_off(tag, id, &fm, data))
                            .map(|data| (data, Some((ia_mgr, data_dir))))
                    }
                };
                let _ = tx.send(res.map(|(data, ia_mgr)| (id, fm, data, ia_mgr, permit)));
            });
            if msg_count < LOAD_FILE_CONCURRENCY {
                msg_count += 1;
            } else {
                self.recv_file_data(cs, use_direct_io, &result_rx, encryption_key.clone())?;
            }
        }
        for _ in 0..msg_count {
            self.recv_file_data(cs, use_direct_io, &result_rx, encryption_key.clone())?;
        }
        Ok(())
    }

    fn recv_file_data(
        &self,
        cs: &mut ChangeSet,
        use_direct_io: bool,
        result_tx: &Receiver<
            Result<(
                u64,
                FileMeta,
                Bytes,
                Option<(IaManager, Arc<PathBuf>)>,
                OwnedSemaphorePermit,
            )>,
        >,
        encryption_key: Option<EncryptionKey>,
    ) -> Result<()> {
        let (id, meta, data, ia_ctx, _permit) = result_tx.recv().unwrap()?;
        let data_len = data.len();
        let file = if let Some((ia_mgr, data_dir)) = ia_ctx {
            let meta_file_path = table_meta_file_local_path(id, meta.file_type, data_dir.deref());
            self.write_local_file_with_file_path(id, data, use_direct_io, meta_file_path.clone())?;
            let meta_file = self.open_local_file_with_file_path(
                id,
                ia_mgr.get_meta_fd_cache(),
                meta_file_path,
            )?;
            let table_meta_file = Arc::new(meta_file);
            Arc::new(IaFile::open(id, meta.file_type, table_meta_file, ia_mgr)?) as _
        } else {
            self.write_local_file(id, data, use_direct_io, meta.file_type)?;
            let file = self.open_local_file(id, meta.file_type)?;
            Arc::new(file) as _
        };
        cs.add_file(id, file, &meta, self.cache.clone(), encryption_key)?;
        ENGINE_LEVEL_WRITE_VEC
            .with_label_values(&[&meta.get_level().to_string()])
            .inc_by(data_len as u64);
        Ok(())
    }

    pub fn load_unloaded_tables(
        &self,
        shard_id: u64,
        shard_ver: u64,
        use_direct_io: bool,
    ) -> Result<()> {
        let shard = self.shards.get(&shard_id).unwrap();
        let tag = shard.tag();
        let mut cs = ChangeSet::new(kvenginepb::ChangeSet::default());
        let data = shard.get_data();
        let mut builder = ShardDataBuilder::new(data.clone());

        let load_tables = data
            .unloaded_tbls
            .iter()
            .filter(|&(_, tbl)| shard.overlap_bound(tbl.data_bound()))
            .map(|(id, tbl)| (*id, tbl.clone()))
            .collect();
        info!("{} load_unloaded_tables: {:?}", tag, load_tables);

        self.load_tables_by_ids(
            tag,
            shard_id,
            shard_ver,
            &load_tables,
            &mut cs,
            use_direct_io,
            false,
            shard.encryption_key.clone(),
        )?;

        // level 0
        let mut new_l0s = data.l0_tbls.clone();
        // blob
        let mut new_blob_tbl_map = data.blob_tbl_map.as_ref().clone();
        // columnar
        let mut new_columnar_levels = data.col_levels.clone();
        // level n
        let mut scf_builders = vec![];
        for cf in 0..NUM_CFS {
            let scf = ShardCfBuilder::new(cf);
            scf_builders.push(scf);
        }
        for (cf, shard_cf) in data.cfs.iter().enumerate() {
            for lh in &shard_cf.levels {
                for sst in &(*lh.tables) {
                    let scf = &mut scf_builders.as_mut_slice()[cf];
                    scf.add_table(sst.clone(), lh.level);
                }
            }
        }
        // vector index
        let mut new_vec_indexes = data.vector_indexes.clone();
        for (id, tbl) in load_tables.into_iter() {
            match tbl.file_type {
                FileType::Sst => {
                    if tbl.level == 0 {
                        let l0 = cs.l0_tables.get(&id).unwrap().clone();
                        new_l0s.push(l0);
                    } else {
                        let sst = cs.ln_tables.get(&id).unwrap().clone();
                        let scf = &mut scf_builders.as_mut_slice()[tbl.get_cf() as usize];
                        scf.add_table(sst, tbl.get_level() as usize);
                    }
                }
                FileType::Columnar => {
                    let col_file = cs.col_files.get(&id).unwrap().clone();
                    new_columnar_levels.add_file(tbl.level as usize, col_file);
                }
                FileType::Blob => {
                    let blob_file = cs.blob_tables.get(&id).unwrap().clone();
                    new_blob_tbl_map.insert(id, blob_file);
                }
                FileType::TxnChunk | FileType::Schema => unreachable!("not supported"),
                FileType::VectorIndex => {
                    let vec_idx_file = cs.vec_index_files.get(&id).unwrap().clone();
                    new_vec_indexes.add_index_file(vec_idx_file);
                }
            }
        }
        new_l0s.sort_by(|a, b| b.version().cmp(&a.version()));
        let mut scfs = [ShardCf::new(0), ShardCf::new(1), ShardCf::new(2)];
        for cf in 0..NUM_CFS {
            let scf = &mut scf_builders.as_mut_slice()[cf];
            scfs[cf] = scf.build();
        }
        new_columnar_levels.sort();
        new_vec_indexes.sort();

        builder.set_l0_tbls(new_l0s);
        builder.set_blob_tbls(new_blob_tbl_map);
        builder.set_cfs(scfs);
        builder.set_unloaded_tbls(HashMap::new());
        builder.set_columnar_levels(new_columnar_levels);
        builder.set_vector_indexes(new_vec_indexes);
        shard.set_data(builder.build());
        Ok(())
    }

    fn write_local_file(
        &self,
        id: u64,
        data: Bytes,
        use_direct_io: bool,
        file_type: FileType,
    ) -> Result<()> {
        let local_file_name = self.local_file_path(id, file_type);
        self.write_local_file_with_file_path(id, data, use_direct_io, local_file_name)
    }

    fn write_local_file_with_file_path(
        &self,
        id: u64,
        data: Bytes,
        use_direct_io: bool,
        file_path: PathBuf,
    ) -> Result<()> {
        let start = Instant::now();
        let tmp_file_name = self.tmp_file_path(id);
        if use_direct_io {
            let mut writer =
                file_system::DirectWriter::new(self.rate_limiter.clone(), IoType::Compaction);
            writer
                .write_to_file(data.chunk(), &tmp_file_name)
                .table_ctx(id, "write_local_file.direct_write_tmp")?;
        } else {
            let mut file = std::fs::File::create(&tmp_file_name)
                .table_ctx(id, "write_local_file.create_tmp")?;
            let mut start_off = 0;
            let write_batch_size = 256 * 1024;
            while start_off < data.len() {
                self.rate_limiter
                    .request(IoType::Compaction, IoOp::Write, write_batch_size);
                let end_off = std::cmp::min(start_off + write_batch_size, data.len());
                file.write_all(&data[start_off..end_off])
                    .table_ctx(id, "write_local_file.write_tmp")?;
                start_off = end_off;
            }
            file.sync_data()
                .table_ctx(id, "write_local_file.sync_tmp")?;
        }
        std::fs::rename(tmp_file_name, file_path).table_ctx(id, "write_local_file.rename")?;
        info!(
            "write local file {} size: {} takes {:?}",
            id,
            data.len(),
            start.saturating_elapsed()
        );
        Ok(())
    }

    pub fn write_local_file_if_not_exists(
        &self,
        id: u64,
        data: Bytes,
        file_type: FileType,
    ) -> Result<()> {
        let file_path = self.local_file_path(id, file_type);
        let _guard = self.lock_file(id);
        if !file_path.exists() {
            self.write_local_file(id, data, false, file_type)?;
        } else {
            let meta = file_path.metadata().table_ctx(id, "metadata")?;
            if meta.size() != data.len() as u64 {
                return Err(table::Error::InvalidFileSize.into());
            }
        }
        Ok(())
    }

    pub fn read_local_file(
        &self,
        id: u64,
        file_type: FileType,
        start_off: u64,
        end_off: Option<u64>,
    ) -> Result<Bytes> {
        let _guard = self.lock_file(id);
        let path = self.local_file_path(id, file_type);
        let mut f = fs::File::open(path).table_ctx(id, "read_local_file.open")?;
        if let Some(end_off) = end_off {
            let mut buf = vec![0; (end_off - start_off) as usize];
            f.read_at(&mut buf, start_off).dfs_ctx(id, "read")?;
            Ok(buf.into())
        } else {
            if start_off > 0 {
                f.seek(SeekFrom::Start(start_off))
                    .table_ctx(id, "read_local_file.seek")?;
            }
            let mut data = vec![];
            f.read_to_end(&mut data).table_ctx(id, "read_local_file")?;
            Ok(data.into())
        }
    }

    fn open_local_file(&self, id: u64, file_type: FileType) -> Result<LocalFile> {
        let path = self.local_file_path(id, file_type);
        self.open_local_file_with_file_path(id, Some(self.fd_cache.clone()), path)
    }

    fn open_local_file_with_file_path(
        &self,
        id: u64,
        fd_cache: Option<FdCache>,
        file_path: PathBuf,
    ) -> Result<LocalFile> {
        let _guard = self.lock_file(id);
        Ok(LocalFile::open(
            id,
            file_path,
            fd_cache,
            self.loaded.load(Relaxed),
        )?)
    }

    pub async fn load_schema_file(&self, id: u64) -> Result<SchemaFile> {
        if let Some(schema_file) = self.schema_files.get(&id) {
            return Ok(schema_file.clone());
        }
        let local_schema_file_path = self.local_schema_file_path(id);
        let res = fs::read(local_schema_file_path.as_path());
        if res.is_ok() {
            let in_mem_file = InMemFile::new(id, res.unwrap().into());
            let schema_file = SchemaFile::open(Arc::new(in_mem_file))?;
            return match self.schema_files.entry(id) {
                Entry::Occupied(e) => Ok(e.get().clone()),
                Entry::Vacant(e) => {
                    e.insert(schema_file.clone());
                    Ok(schema_file)
                }
            };
        }
        let opts = dfs::Options::default().with_type(FileType::Schema);
        let data = self.fs.read_file(id, opts).await?;
        let in_mem_file = InMemFile::new(id, data.clone());
        let schema_file = SchemaFile::open(Arc::new(in_mem_file))?;
        match self.schema_files.entry(id) {
            Entry::Occupied(e) => return Ok(e.get().clone()),
            Entry::Vacant(e) => {
                e.insert(schema_file.clone());
            }
        }
        // Only the first one will write the file to disk.
        let tmp_file_name = self.tmp_file_path(id);
        fs::write(&tmp_file_name, data.chunk()).table_ctx(id, "load_schema_file.write_tmp")?;
        fs::rename(&tmp_file_name, local_schema_file_path.as_path())
            .table_ctx(id, "load_schema_file.rename")?;
        Ok(schema_file)
    }

    pub(crate) fn local_file_path(&self, file_id: u64, file_type: FileType) -> PathBuf {
        match file_type {
            FileType::Sst => self.local_sst_file_path(file_id),
            FileType::Blob => self.local_blob_file_path(file_id),
            FileType::Schema => self.local_schema_file_path(file_id),
            FileType::Columnar => self.local_columnar_file_path(file_id),
            FileType::TxnChunk => panic!("TxnChunk files are managed by TxnChunkManager"),
            FileType::VectorIndex => self.local_vector_index_file_path(file_id),
        }
    }

    pub(crate) fn local_sst_file_path(&self, file_id: u64) -> PathBuf {
        self.opts.local_dir.join(new_sst_filename(file_id))
    }

    pub(crate) fn local_blob_file_path(&self, file_id: u64) -> PathBuf {
        self.opts.local_dir.join(new_blob_filename(file_id))
    }

    pub(crate) fn local_schema_file_path(&self, file_id: u64) -> PathBuf {
        self.opts.local_dir.join(new_schema_filename(file_id))
    }

    pub(crate) fn local_columnar_file_path(&self, file_id: u64) -> PathBuf {
        self.opts.local_dir.join(new_columnar_filename(file_id))
    }

    pub(crate) fn local_vector_index_file_path(&self, file_id: u64) -> PathBuf {
        self.opts.local_dir.join(new_vector_index_filename(file_id))
    }

    fn tmp_file_path(&self, file_id: u64) -> PathBuf {
        lazy_static::lazy_static! {
            static ref TMP_FILE_ID: AtomicU64 = AtomicU64::default();
        }
        let tmp_id = TMP_FILE_ID.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        self.opts.local_dir.join(new_tmp_filename(file_id, tmp_id))
    }
}

pub fn collect_snap_lock_txn_file_refs(snap: &kvenginepb::Snapshot) -> Vec<TxnFileRef> {
    let props = snap.get_properties();
    debug_assert_eq!(props.keys.len(), props.values.len());
    for (key, val) in props.get_keys().iter().zip(props.get_values().iter()) {
        if key == TXN_FILE_REF {
            let mut txn_file_refs = TxnFileRefs::new();
            txn_file_refs.merge_from_bytes(val).unwrap();
            return txn_file_refs
                .take_txn_file_refs()
                .into_iter()
                .filter(|r| !r.lock_val_prefix.is_empty())
                .collect::<Vec<_>>();
        }
    }
    vec![]
}

// For safety. Table meta offset should have been fixed. See
// `ShardMeta::fix_table_meta_offset`.
fn validate_table_meta_off(tag: ShardTag, id: u64, fm: &FileMeta, data: Bytes) -> Result<Bytes> {
    debug_assert!(fm.can_use_ia(), "{}: {}: {:?}", tag, id, fm);
    if fm.table_meta_off > 0 {
        return Ok(data);
    }

    warn!("{} validate_table_meta_off: table meta offset is 0", tag; "id" => id, "fm" => ?fm);
    match fm.file_type {
        FileType::Sst => SsTable::get_meta_data(&data).map_err(|err| {
            error!("{} validate_table_meta_off: get meta data failed", tag;
                "err" => ?err, "id" => id, "fm" => ?fm);
            Error::TableError(err)
        }),
        FileType::Columnar => {
            // Columnar must have meta offset.
            debug_assert!(false);
            Ok(data)
        }
        _ => Ok(data),
    }
}
