// Copyright 2023 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    collections::{BTreeMap, HashMap},
    fmt::Display,
    path::{Path, PathBuf},
    sync::Arc,
    time::Duration,
};

use chrono::Utc;
use futures::{stream::FuturesUnordered, TryStreamExt};
use kvengine::{
    dfs::{DFSConfig, Dfs, FileType, OverlaidFs, S3Fs},
    util::TxnFileRefPropertyHelper,
    IdAllocator, TXN_FILE_REF,
};
use kvenginepb::{ChangeSet, FileRef, PackedBackup};
use pd_client::{
    pd_control::{KeyspaceMeta, PdControl},
    PdClient,
};
use protobuf::Message;
use rfenginepb::ClusterBackupMeta;
use rfstore::store::PdIdAllocator;
use security::SecurityConfig;
use tempdir::TempDir;
use tikv_util::{box_err, box_try, config::ReadableSize, info, warn};
use tokio::runtime::RuntimeFlavor;
use uuid::Uuid;

use self::offline_pd::OfflinePd;
use crate::{
    archive::ArchiveReader,
    backup::packed_backup_prefixed,
    common::{create_pd_client, PACKED_META_NAME_FORMAT},
    error::Error,
    limiter::{ByteLimiter, ThroughputLimiter, TABLE_SIZE_MULTIPLIER},
    lock::LockResolver,
    restore::RestoreConfig,
    restore_keyspace::{
        self, load_norm_backup_meta, BackupCluster, BackupClusterOptions, ReportRestoreStepTrait,
        RestoredKeyspace,
    },
    Result,
};

const WORKING_PATH_PREFIX: &str = "pack";

const DEFAULT_UNKNOWN_FILE_SIZE_BYTES: u64 = 1024 * 1024; // 1MiB

fn build_file_size_map(packed: &PackedBackup) -> HashMap<u64, u64> {
    let mut sizes = HashMap::new();
    for cs in packed.get_shards() {
        if !cs.has_snapshot() {
            continue;
        }
        let snap = cs.get_snapshot();

        for l0 in snap.get_l0_creates() {
            sizes.insert(l0.id, l0.size as u64);
        }
        for table in snap.get_table_creates() {
            let size = (table.meta_offset as f64 * TABLE_SIZE_MULTIPLIER) as u64;
            if size > 0 {
                sizes.insert(table.id, size);
            }
        }
        for blob in snap.get_blob_creates() {
            let size = blob.meta_offset as u64;
            if size > 0 {
                sizes.insert(blob.id, size);
            }
        }
        for col in snap.get_columnar_creates() {
            let size = (col.meta_offset as f64 * TABLE_SIZE_MULTIPLIER) as u64;
            if size > 0 {
                sizes.insert(col.id, size);
            }
        }

        // TODO: Schema and TxnChunk files are not rate-limited because their
        // actual size is unknown here (txn_chunk_max_size is a
        // TiDB-side config, and querying S3 for the size requires a
        // size-by-key API that DFS doesn't support yet). To fix, either
        // add a DFS::size_by_key() method, or store chunk size in
        // the backup metadata.
    }
    sizes
}

pub struct PackContext {
    dfs: Arc<OverlaidFs>,

    cluster_backup: ClusterBackupMeta,
    cluster: BackupCluster,
    keyspace_meta: KeyspaceMeta,
    reporter: Arc<dyn ReportPackBackupStepTrait>,
}

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct PackConfig {
    pub pd_config: pd_client::Config,
    pub dfs: DFSConfig,
    pub security: SecurityConfig,

    pub data_dir: Option<PathBuf>,
    pub backup_name: String,

    pub keyspace_name: String,
    pub offline: bool,
}

pub struct PackedBackupMeta {
    pub copyable_path: String,
    pub packed_content: PackedBackup,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub enum PackBackupStep {
    CreateAdHocKvEngine,
    ResolveLock,
    FlushShard,
    CreateMeta,
}

pub struct NoopReporter;

pub trait ReportPackBackupStepTrait: Send + Sync + 'static {
    fn report_step(&self, _step: PackBackupStep) {}
}

impl ReportPackBackupStepTrait for NoopReporter {}

pub fn pack_backup_with_cfg(
    reporter: Arc<dyn ReportPackBackupStepTrait>,
    cfg: PackConfig,
) -> Result<PackedBackupMeta> {
    info!("Welcome to pack backup."; "cfg" => ?cfg);
    // `ctx.execute` contains `load_tables_by_id` which is blocking.
    // If execute it in the dfs runtime, it may block the whole dfs runtime
    // which finally sticks the whole procedure.
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    let mut ctx = PackContext::create_from_config(&cfg)?;
    ctx.set_reporter(reporter);

    let (path, content) = runtime.block_on(ctx.execute(&cfg.backup_name))?;
    Ok(PackedBackupMeta {
        copyable_path: path,
        packed_content: content,
    })
}

pub struct PackEnv {
    pub pd_client: Arc<dyn PdClient>,
    pub dfs: Arc<S3Fs>,
    pub tmpfs: Arc<S3Fs>,
    pub offline: bool,

    pub work_path: PathBuf,
    pub _temp_dir: Option<TempDir>,

    pub restore_conf: RestoreConfig,
    pub reporter: Arc<dyn ReportPackBackupStepTrait>,
}

pub fn maybe_block_in_place<T>(f: impl FnOnce() -> T) -> T {
    if let Ok(hnd) = tokio::runtime::Handle::try_current() {
        if hnd.runtime_flavor() == RuntimeFlavor::MultiThread {
            return tokio::task::block_in_place(f);
        }
    }
    f()
}

impl PackContext {
    pub fn overlay_fs(&self) -> &OverlaidFs {
        &self.dfs
    }

    fn packed_backup_path(&self, backup_name: &str) -> String {
        let now = Utc::now();
        packed_backup_prefixed(
            &self.dfs.base_fs().get_prefix(),
            &format!(
                "native/{}/{}/{}.packed.meta",
                backup_name,
                self.keyspace_meta.name,
                now.format(PACKED_META_NAME_FORMAT)
            ),
        )
    }

    pub fn create_from_config(config: &PackConfig) -> Result<Self> {
        let dfs = Arc::new(S3Fs::new_from_config(config.dfs.clone()));
        let hnd = dfs.get_runtime().handle();

        let (backup_meta, archive_reader) =
            hnd.block_on(load_norm_backup_meta(&dfs, &config.backup_name))?;
        let (pd_client, keyspace_meta) = if config.offline {
            let pd_client = hnd.block_on(OfflinePd::new(
                backup_meta.cluster_id,
                backup_meta.keyspace_meta.clone(),
            ))?;
            let keysapce_id = pd_client.keyspace_id(&config.keyspace_name)?;
            let mut meta = KeyspaceMeta::default();
            meta.name = config.keyspace_name.clone();
            meta.id = keysapce_id;
            (Arc::new(pd_client) as Arc<dyn PdClient>, meta)
        } else {
            let pd_client = Arc::new(create_pd_client(&config.security, &config.pd_config))
                as Arc<dyn PdClient>;
            let pd_ctl = PdControl::new(config.pd_config.clone(), pd_client.get_security_mgr())?;
            let keyspace_meta = hnd.block_on(pd_ctl.get_keyspace_by_name(&config.keyspace_name))?;
            (pd_client, keyspace_meta)
        };

        if !config.offline {
            let enc = pd_client.get_keyspace_encryption(keyspace_meta.id)?;
            if enc.enabled {
                return Err(Error::Incompatible(format!(
                    "Keyspace {} was encrypted hence cannot be packed for now.",
                    keyspace_meta.name
                )));
            }
        }

        let (work_path, _temp_dir) = match &config.data_dir {
            Some(path) => (path.clone(), None),
            None => {
                let dir = box_try!(TempDir::new(WORKING_PATH_PREFIX));
                (dir.path().to_owned(), Some(dir))
            }
        };
        let restore_conf = RestoreConfig {
            security: config.security.clone(),
            ..Default::default()
        };

        let mut tmpfs_config = config.dfs.clone();
        tmpfs_config.prefix = packed_backup_prefixed(
            &tmpfs_config.prefix,
            &format!("native/{}/new_tables", config.backup_name),
        );
        let tmpfs = Arc::new(S3Fs::new_from_config(tmpfs_config));

        let env = PackEnv {
            pd_client,
            dfs,
            tmpfs,
            work_path,
            offline: config.offline,
            _temp_dir,
            restore_conf,
            reporter: Arc::new(NoopReporter),
        };
        Self::create_from_env(env, backup_meta, archive_reader, keyspace_meta)
    }

    pub fn create_from_env(
        env: PackEnv,
        backup_meta: ClusterBackupMeta,
        archive_reader: Option<ArchiveReader>,
        keyspace_meta: KeyspaceMeta,
    ) -> Result<Self> {
        env.reporter
            .report_step(PackBackupStep::CreateAdHocKvEngine);
        let cluster_id = env.pd_client.get_cluster_id()?;
        if !env.offline && backup_meta.cluster_id != cluster_id {
            return Err(Error::Incompatible(format!(
                "cluster ID not match, backup meta: {}, current pd: {}",
                backup_meta.cluster_id, cluster_id
            )));
        }

        let overlay_fs = Arc::new(OverlaidFs::new(
            Arc::clone(&env.dfs),
            Arc::clone(&env.tmpfs),
        ));

        let opt = BackupClusterOptions {
            cluster_meta: backup_meta.clone(),
            path: env.work_path,
            pd_client: env.pd_client.clone(),
            // pack backup don't need pd-ctl
            pd_control: None,
            dfs: Arc::clone(&env.dfs),
            restore_conf: env.restore_conf,
            keyspace_id: keyspace_meta.id,
            target_keyspace_id: keyspace_meta.id,
            truncate_ts: backup_meta.backup_ts,
            archiving: false,
            archive_reader,
            load_all_tables: false,
            override_dfs: Some(Arc::clone(&overlay_fs) as Arc<dyn Dfs>),
            offline_packing: env.offline,
        };
        let cluster = maybe_block_in_place(|| BackupCluster::new_opt(opt))?;

        Ok(Self {
            dfs: overlay_fs,
            cluster_backup: backup_meta,
            cluster,
            keyspace_meta,
            reporter: env.reporter,
        })
    }

    pub fn set_reporter(&mut self, reporter: Arc<dyn ReportPackBackupStepTrait>) {
        self.reporter = reporter;
    }

    async fn create_pack(&mut self) -> Result<PackedBackup> {
        self.reporter.report_step(PackBackupStep::ResolveLock);
        let lock_resolver = LockResolver::new(
            self.cluster.tag(),
            self.cluster.get_kvengine(),
            1024,
            self.cluster.get_shard_meta_getter(),
        );
        let resolved_locks = lock_resolver.resolve_locks().await?;
        self.cluster
            .add_shards_need_flush(&resolved_locks.resolved_shards);

        self.reporter.report_step(PackBackupStep::FlushShard);
        self.cluster.flush_shards(Duration::MAX)?;

        self.reporter.report_step(PackBackupStep::CreateMeta);
        let vers = self.cluster.get_kvengine().get_all_shard_id_vers();
        let mut packed = PackedBackup::new();
        packed.set_backup_ts(self.cluster_backup.backup_ts);
        packed.set_cluster_id(self.cluster_backup.cluster_id);
        packed.set_safe_ts(self.cluster_backup.safe_ts);
        packed.set_keyspace_id(self.keyspace_meta.id);
        packed.set_keyspace_name(self.keyspace_meta.name.clone());
        packed.set_keyspace_size({
            let mut size = 0;
            for shard in self.cluster_backup.stores.iter() {
                size += shard
                    .keyspace_size
                    .get(&self.keyspace_meta.id)
                    .unwrap_or_default()
                    .size;
            }
            size
        });
        packed.set_content_bucket(self.dfs.base_fs().get_bucket());
        // Note: we need to get the actual size from the backup meta after they were
        // added. This size is probably not accurate.
        packed.set_engine_size(self.cluster.get_kvengine().size());
        packed.set_resolved_ts(
            resolved_locks
                .resolved_ts
                .map(|v| v.0)
                .unwrap_or(self.cluster_backup.backup_ts),
        );
        for ver in vers {
            let shard = self.cluster.get_shard(ver.id).unwrap();
            packed.mut_shards().push(shard.meta.to_change_set());
        }
        packed.set_content_file_refs(self.engine_files().into());
        Ok(packed)
    }

    pub fn engine_files(&self) -> Vec<FileRef> {
        let schema_files = self
            .cluster
            .shards
            .values()
            .filter(|v| v.meta.schema.is_valid())
            .map(|v| {
                let mut file_ref = FileRef::new();
                file_ref.file_abs_path = self
                    .dfs
                    .base_fs()
                    .file_key(v.meta.schema.schema_file_id, FileType::Schema);
                file_ref.file_id = v.meta.schema.schema_file_id;
                file_ref.file_type = FileType::Schema as _;
                file_ref
            });
        let table_files = self
            .cluster
            .get_all_shard_files(None, None)
            .into_iter()
            .map(|f| {
                let mut file_ref = FileRef::new();
                file_ref.file_abs_path = if self.dfs.file_is_modified(f.ftype, f.id) {
                    self.dfs.overlay_fs().file_key(f.id, f.ftype)
                } else {
                    self.dfs.base_fs().file_key(f.id, f.ftype)
                };
                file_ref.file_id = f.id;
                file_ref.file_type = f.ftype as u32;
                file_ref
            });
        table_files.chain(schema_files).collect()
    }

    /// execute the pack procedure and save the packed backup with the specified
    /// name.
    ///
    /// # Returns
    ///
    /// A tuple contains the full path (`<bucket>/<key>`) to the saved backup
    /// meta and the content of backup meta.
    pub async fn execute(&mut self, save_to: &str) -> Result<(String, PackedBackup)> {
        let manifest = self.create_pack().await?;
        let output_path = self.packed_backup_path(save_to);
        let json_content = protobuf::Message::write_to_bytes(&manifest).unwrap();
        self.dfs
            .base_fs()
            .put_object(
                output_path.clone(),
                json_content.into(),
                output_path.clone(),
            )
            .await?;
        let abs_output_path = format!("{}/{}", self.dfs.base_fs().get_bucket(), output_path);
        Ok((abs_output_path, manifest))
    }
}

pub struct MigratePackEnv {
    dfs: Arc<S3Fs>,
    packed: PackedBackup,
    exotic_path: String,
    limiter: Option<Arc<ByteLimiter>>,
    file_sizes: HashMap<u64, u64>,
}

impl MigratePackEnv {
    pub fn get_packed_backup(&self) -> &PackedBackup {
        &self.packed
    }

    pub fn get_file_copy_source(&self, file: &FileRef) -> String {
        format!(
            "{}/{}",
            self.packed.content_bucket,
            file.get_file_abs_path()
        )
    }

    pub async fn save_patched_meta_to(
        &self,
        key: &str,
        patch: impl FnOnce(&mut PackedBackup),
    ) -> Result<()> {
        let mut packed = self.packed.clone();
        patch(&mut packed);
        let data = protobuf::Message::write_to_bytes(&packed).unwrap();
        self.dfs
            .put_object(key.to_owned(), data.into(), key.to_owned())
            .await?;
        Ok(())
    }

    pub async fn save_patched_meta_to_rel(
        &self,
        rel_key: &str,
        patch: impl FnOnce(&mut PackedBackup),
    ) -> Result<()> {
        let key = packed_backup_prefixed(&self.dfs.get_prefix(), rel_key);
        self.save_patched_meta_to(&key, patch).await
    }

    pub async fn load_exotic(s3fs: Arc<S3Fs>, exotic_path: &str) -> Result<Self> {
        let uuid = Uuid::new_v4();
        let now = Utc::now();
        let tmp = packed_backup_prefixed(
            &s3fs.get_prefix(),
            &format!("unpack_tmp/{}/{}.meta", now.format("%Y%m%d"), uuid),
        );

        // Don't reconfigurate the client but directly call `copy-object` to verify the
        // origin meta is accessible from this context.
        s3fs.raw_copy_object(exotic_path, &tmp, None, None).await?;

        let result = s3fs
            .get_object(tmp.clone(), format!("temp({uuid})"), Default::default())
            .await
            .map_err(Error::from)
            .and_then(|packed_bytes| {
                let mut packed = PackedBackup::new();
                packed
                    .merge_from_bytes(&packed_bytes)
                    .map(|_| packed)
                    .map_err(|e| box_err!(e))
            });

        if let Err(err) = s3fs.delete_object(tmp, format!("temp({uuid})")).await {
            warn!("failed to delete tempfile during loading exotic packed backup."; "err" => ?err);
        }

        let packed = result?;
        let file_sizes = build_file_size_map(&packed);
        Ok(Self {
            dfs: s3fs,
            packed,
            exotic_path: exotic_path.to_owned(),
            limiter: None,
            file_sizes,
        })
    }

    pub fn with_rate_limit(
        mut self,
        max_throughput: ReadableSize,
        source: &'static str,
        request_id: u64,
    ) -> Self {
        self.limiter = (max_throughput.0 > 0)
            .then(|| Arc::new(ByteLimiter::new(max_throughput.0, source, request_id)));
        self
    }

    /// Sets an externally-created ByteLimiter (e.g. one sharing a global token
    /// bucket).
    pub fn with_byte_limiter(mut self, limiter: Arc<ByteLimiter>) -> Self {
        self.limiter = Some(limiter);
        self
    }

    pub fn limiter(&self) -> Option<Arc<ByteLimiter>> {
        self.limiter.clone()
    }

    pub fn estimate_file_size(&self, file: &FileRef) -> u64 {
        self.file_sizes
            .get(&file.file_id)
            .copied()
            .unwrap_or(DEFAULT_UNKNOWN_FILE_SIZE_BYTES)
    }

    pub fn estimated_total_file_size(&self) -> u64 {
        self.file_sizes.values().sum()
    }
}

pub struct RestorePackEnv<'a> {
    pub dfs: Arc<S3Fs>,
    pub pd_client: Arc<dyn PdClient>,
    pub pd_control: Option<PdControl>,
    pub target_keyspace: u32,
    pub reporter: Arc<dyn ReportRestoreStepTrait>,
    pub restore_config: RestoreConfig,
    pub data_dir: &'a Path,
    pub truncate_ts: Option<u64>,
    pub limiter: Option<Arc<ThroughputLimiter>>,
}

impl<'a> RestorePackEnv<'a> {
    pub fn execute(&mut self, packed: PackedBackup) -> Result<RestoredKeyspace> {
        if !packed.unpacked {
            return Err(Error::Incompatible(
                "the specified packed backup isn't ready, please `unpack` it first before restore"
                    .to_owned(),
            ));
        }
        info!("Restore Started.");
        let mut cluster = BackupCluster::of_packed(
            &packed,
            self.data_dir.to_owned(),
            self.pd_client.clone(),
            self.pd_control.clone(),
            self.dfs.clone(),
            self.restore_config.clone(),
            self.target_keyspace,
            self.truncate_ts.unwrap_or(packed.backup_ts),
        )?;

        restore_keyspace::prepare_and_restore_cluster(
            &mut cluster,
            self.restore_config.clone(),
            self.dfs.get_runtime(),
            &self.reporter,
            &self.limiter,
        )
    }
}

#[derive(Serialize, Debug)]
#[serde(untagged)]
pub enum CopyStep {
    FetchMeta,
    CopySst {
        total: usize,
        running: usize,
        done: usize,
    },
    SaveMeta,
}

impl CopyStep {
    pub fn stage_name(&self) -> &'static str {
        match self {
            CopyStep::FetchMeta => "FetchMeta",
            CopyStep::CopySst { .. } => "CopySst",
            CopyStep::SaveMeta => "SaveMeta",
        }
    }
}

pub trait ReportCopyStepTrait: Send + Sync + 'static {
    fn report_step(&self, _step: CopyStep) {}
}

impl ReportCopyStepTrait for NoopReporter {}

#[derive(Debug, Serialize, Deserialize)]
pub struct CopiedPackedBackup {
    pub new_exotic_path: String,
    pub files_copied: usize,
}

pub struct CopyPackedRun {
    menv: MigratePackEnv,
    target_prefix: String,
    new_name: String,
    reporter: Arc<dyn ReportCopyStepTrait>,
}

impl CopyPackedRun {
    pub fn new(
        menv: MigratePackEnv,
        target_prefix: impl ToString,
        reporter: Arc<dyn ReportCopyStepTrait>,
    ) -> Self {
        Self {
            menv,
            target_prefix: target_prefix.to_string(),
            new_name: Utc::now().format("_meta/%Y%d%m/%H%M%S.meta").to_string(),
            reporter,
        }
    }

    fn prefixed(&self, pathlike: impl Display) -> String {
        let cluster_id = self.menv.get_packed_backup().cluster_id;
        format!(
            "{}/{}/{}/{}",
            self.menv.dfs.get_prefix().trim_matches('/'),
            self.target_prefix.trim_matches('/'),
            cluster_id,
            pathlike
        )
    }

    pub async fn execute(&mut self) -> Result<CopiedPackedBackup> {
        let new_files = self.copy_ssts().await?;

        self.reporter.report_step(CopyStep::SaveMeta);
        let files_copied = new_files.len();
        let new_meta_path = self.prefixed(&self.new_name);
        let new_content_bucket = self.menv.dfs.get_bucket();
        self.menv
            .save_patched_meta_to(&new_meta_path, move |meta| {
                meta.content_file_refs = new_files.into();
                meta.content_bucket = new_content_bucket;
            })
            .await?;
        let copied = CopiedPackedBackup {
            new_exotic_path: format!("{}/{}", self.menv.dfs.get_bucket(), new_meta_path),
            files_copied,
        };
        Ok(copied)
    }

    async fn copy_ssts(&self) -> Result<Vec<FileRef>> {
        let max_conc = 128;
        let mut futures = FuturesUnordered::new();
        let mut new_files = vec![];
        let total_ssts = self.menv.packed.content_file_refs.len();
        for file in self.menv.packed.get_content_file_refs() {
            self.reporter.report_step(CopyStep::CopySst {
                total: total_ssts,
                running: futures.len(),
                done: new_files.len(),
            });

            if futures.len() > max_conc {
                new_files.push(futures.try_next().await?.unwrap());
            }

            let source_key = self.menv.get_file_copy_source(file);
            let file_key = self.menv.dfs.rel_file_key(
                file.file_id,
                FileType::from_u8(file.file_type as u8).unwrap_or(FileType::Sst),
            );
            let target_key = self.prefixed(file_key);
            let mut new_file = file.clone();
            let file_size = self.menv.estimate_file_size(file);
            let limiter = self.menv.limiter();
            futures.push(async move {
                if let Some(limiter) = limiter.as_ref() {
                    limiter.consume(file_size).await;
                }
                self.menv
                    .dfs
                    .raw_copy_object(&source_key, &target_key, None, None)
                    .await?;
                new_file.set_file_abs_path(target_key);
                Result::Ok(new_file)
            });
        }

        while let Some(file) = futures.try_next().await? {
            new_files.push(file);
            self.reporter.report_step(CopyStep::CopySst {
                total: total_ssts,
                running: futures.len(),
                done: new_files.len(),
            });
        }

        Ok(new_files)
    }
}

#[derive(Serialize, Debug)]
#[serde(untagged)]
pub enum UnpackStep {
    VerifyBackup { total: usize, finished: usize },
    AllocateNewIds,
    MoveTableFiles { total: usize, finished: usize },
    WriteNewMeta,
}

impl UnpackStep {
    pub fn stage_name(&self) -> &'static str {
        match self {
            UnpackStep::VerifyBackup { .. } => "VerifyBackup",
            UnpackStep::AllocateNewIds => "AllocateNewIds",
            UnpackStep::MoveTableFiles { .. } => "MoveTableFiles",
            UnpackStep::WriteNewMeta => "WriteNewMeta",
        }
    }
}

pub trait ReportUnpackStepTrait: Send + Sync + 'static {
    fn report_step(&self, _step: UnpackStep) {}
}

impl ReportUnpackStepTrait for NoopReporter {}

pub struct UnpackRun {
    menv: MigratePackEnv,
    pd_client: Arc<dyn PdClient>,
    id_remap: BTreeMap<u64, u64>,
    reporter: Arc<dyn ReportUnpackStepTrait>,
}

impl UnpackRun {
    pub fn new(
        env: MigratePackEnv,
        pd_client: Arc<dyn PdClient>,
        reporter: Arc<dyn ReportUnpackStepTrait>,
    ) -> Self {
        Self {
            menv: env,
            pd_client,
            id_remap: BTreeMap::new(),
            reporter,
        }
    }

    pub async fn execute(&mut self) -> Result<String> {
        if self.menv.packed.unpacked {
            return Err(Error::Incompatible(format!(
                "the backup is already unpacked, you may unpack {:?} instead",
                self.menv.packed.unpacked_from
            )));
        }
        self.unpack_ssts().await?;

        let packed_backup_name = Utc::now()
            .format("exotic/%Y%m%d/%H%M%S.packed.meta")
            .to_string();
        let cluster_id = self.pd_client.get_cluster_id()?;
        self.menv
            .save_patched_meta_to_rel(&packed_backup_name, |packed| {
                // Set to the current cluster id.
                // So we know the backup meta was deeply copied.
                packed.set_cluster_id(cluster_id);
                packed.set_unpacked(true);
                packed.set_unpacked_from(self.menv.exotic_path.clone());
                packed.clear_content_file_refs();
            })
            .await?;
        Ok(packed_backup_name)
    }

    pub async fn unpack_ssts(&mut self) -> Result<()> {
        let total = self.menv.packed.shards.len();
        let mut finished = 0usize;
        let max_conc = 128;
        let mut futures = FuturesUnordered::new();
        for cs in self.menv.packed.get_shards().iter() {
            self.reporter
                .report_step(UnpackStep::VerifyBackup { total, finished });
            futures.push(self.verify_changeset_recoverable(cs));
            if futures.len() >= max_conc {
                futures.try_next().await?;
                finished += 1;
            }
        }
        futures
            .try_for_each(|()| {
                finished += 1;
                self.reporter
                    .report_step(UnpackStep::VerifyBackup { total, finished });
                futures::future::ok(())
            })
            .await?;
        self.move_files().await?;
        let mut shards = self.menv.packed.take_shards();
        for shard in shards.iter_mut() {
            self.rewrite_changeset(shard)?;
        }
        self.menv.packed.set_shards(shards);
        Ok(())
    }

    async fn paged_allocate_new_ids(&self) -> Result<Vec<u64>> {
        let id_alloc = PdIdAllocator::new(self.pd_client.clone());
        let to_allocate = self.menv.packed.get_content_file_refs().len();
        let mut new_ids = Vec::with_capacity(to_allocate);
        while new_ids.len() < to_allocate {
            // PD's default configuration allows allocate 1 << 18 tso per `tso` call.
            // Leave a buffer here to avoid cannot allocate tso when tso server is busy.
            const MAX_ID_BATCH: usize = 1 << 15;
            let need_to_allocate = to_allocate - new_ids.len();
            let allocate_batch = Ord::min(MAX_ID_BATCH, need_to_allocate);
            new_ids.extend(id_alloc.alloc_id_async(allocate_batch).await?);
        }
        assert_eq!(
            new_ids.len(),
            self.menv.packed.get_content_file_refs().len()
        );
        Ok(new_ids)
    }

    async fn move_files(&mut self) -> Result<()> {
        let max_conc = 128;
        let mut futures = FuturesUnordered::new();
        self.reporter.report_step(UnpackStep::AllocateNewIds);
        let new_ids = self.paged_allocate_new_ids().await?;
        let total = self.menv.packed.content_file_refs.len();
        let mut finished = 0usize;

        for (file, new_id) in self
            .menv
            .packed
            .get_content_file_refs()
            .iter()
            .zip(new_ids.iter())
        {
            self.reporter
                .report_step(UnpackStep::MoveTableFiles { total, finished });
            if futures.len() > max_conc {
                futures.try_next().await?;
                finished += 1;
            }
            let source_key = format!(
                "{}/{}",
                self.menv.packed.content_bucket,
                file.get_file_abs_path()
            );
            let file_size = self.menv.estimate_file_size(file);
            let new_id = if file.file_type == FileType::Blob as u32 {
                file.file_id
            } else {
                self.id_remap.insert(file.file_id, *new_id);
                *new_id
            };
            let target_key = self.menv.dfs.file_key(
                new_id,
                FileType::from_u8(file.file_type as u8).ok_or_else(|| {
                    Error::Other(box_err!("unknown file type {}", file.file_type))
                })?,
            );
            let dfs = self.menv.dfs.as_ref();
            let limiter = self.menv.limiter();
            futures.push(async move {
                if let Some(limiter) = limiter.as_ref() {
                    limiter.consume(file_size).await;
                }
                dfs.raw_copy_object(&source_key, &target_key, None, None)
                    .await?;
                Result::Ok(())
            });
        }

        futures
            .try_for_each(|()| {
                finished += 1;
                self.reporter
                    .report_step(UnpackStep::MoveTableFiles { total, finished });

                futures::future::ok(())
            })
            .await?;
        Ok(())
    }

    async fn verify_changeset_recoverable(&self, changeset: &ChangeSet) -> Result<()> {
        macro_rules! zz_compatible_check {
            ($e:expr) => {
                if !$e {
                    return Err(Error::Incompatible(format!(
                        "requirement {} not satisfied",
                        stringify!($e)
                    )));
                }
            };
        }
        zz_compatible_check!(!changeset.has_compaction());
        zz_compatible_check!(!changeset.has_flush());
        zz_compatible_check!(!changeset.has_initial_flush());
        zz_compatible_check!(!changeset.has_split());
        zz_compatible_check!(!changeset.shard_delete);
        zz_compatible_check!(!changeset.has_ingest_files());
        zz_compatible_check!(!changeset.has_destroy_range());
        zz_compatible_check!(!changeset.has_truncate_ts());
        zz_compatible_check!(!changeset.has_trim_over_bound());
        zz_compatible_check!(!changeset.has_restore_shard());
        zz_compatible_check!(!changeset.has_major_compaction());
        zz_compatible_check!(!changeset.has_update_schema_meta());
        zz_compatible_check!(!changeset.has_columnar_compaction());
        zz_compatible_check!(!changeset.has_update_vector_index());
        zz_compatible_check!(!changeset.clear_columnar);

        // NOTE: maybe parallelize this? this might be slow once there are many blob
        // files.
        for blob_file in changeset.get_snapshot().get_blob_creates() {
            let key = self.menv.dfs.file_key(blob_file.id, FileType::Blob);
            let conflicting_blob = self.menv.dfs.exist(key.clone(), key).await?;
            if conflicting_blob {
                // Note: after a failed restoration this may make the backup cannot be
                // restored forever, perhaps we need to compare the sha256 between the files.
                return Err(Error::Incompatible(format!(
                    "the blob create {:?} conflicts",
                    blob_file
                )));
            }
        }
        Ok(())
    }

    fn rewrite_changeset(&self, changeset: &mut ChangeSet) -> Result<()> {
        // This changeset was generated by `to_pb`, it should contains a snapshot only.

        let shard_id = changeset.shard_id;
        let snap = changeset.snapshot.get_mut_ref();
        let remap = |id: &mut u64, ty: &str| -> Result<()> {
            let old_id = *id;
            let new_id = self.id_remap.get(&old_id).ok_or_else(|| {
                Error::Other(box_err!(
                    "remap of type {} with id {} not found",
                    ty,
                    old_id
                ))
            })?;
            info!("remapping table id."; "old_id" => old_id, "new_id" => *new_id, "type" => %ty);
            *id = *new_id;
            Ok(())
        };

        // Blob files cannot be rewritten as they are referenced by KV pairs directly.
        for table_create in snap.mut_table_creates().iter_mut() {
            remap(&mut table_create.id, "NormalTable")?;
        }
        for l0_create in snap.mut_l0_creates().iter_mut() {
            remap(&mut l0_create.id, "L0TableFile")?;
        }
        for unconverted_l0 in snap.mut_unconverted_l0s().iter_mut() {
            remap(unconverted_l0, "UnconvertedL0")?;
        }
        for columnar_create in snap.mut_columnar_creates().iter_mut() {
            remap(&mut columnar_create.id, "ColumnarTable")?;
        }
        for vector_index in snap.mut_vector_indexes().iter_mut() {
            for vfile in vector_index.mut_files().iter_mut() {
                remap(&mut vfile.id, "VectorIndexFile")?;
            }
        }

        let properties = kvengine::shard::Properties::new().apply_pb(snap.get_properties());
        let mut helper =
            TxnFileRefPropertyHelper::from_property(properties.get(TXN_FILE_REF)).unwrap();
        for tref in helper.mut_txn_file_refs() {
            for chunk_id in tref.mut_chunk_ids() {
                remap(chunk_id, "TxnFiles")?;
            }
        }
        properties.set(TXN_FILE_REF, &helper.marshall());
        snap.set_properties(properties.to_pb(shard_id));

        if changeset.has_parent() {
            self.rewrite_changeset(changeset.mut_parent())?;
        }

        Ok(())
    }
}

pub mod offline_pd {
    use std::{
        collections::{BTreeMap, HashMap},
        sync::{Arc, Mutex},
    };

    use async_trait::async_trait;
    use bstr::ByteSlice;
    use dashmap::DashMap;
    use futures::{future, FutureExt};
    use pd_client::{pd_control::KeyspaceMeta, PdClient, PdFuture};
    use security::GetSecurityManager;
    use tikv_util::box_err;
    use txn_types::TimeStamp;

    use crate::error::{Error, Result};

    #[derive(Default)]
    pub struct OfflinePd {
        cluster_id: u64,
        etcd: BTreeMap<Vec<u8>, Vec<u8>>,
        last_tso: Mutex<TimeStamp>,
    }

    impl GetSecurityManager for OfflinePd {}

    #[async_trait]
    impl PdClient for OfflinePd {
        fn batch_get_tso(&self, count: u32) -> PdFuture<TimeStamp> {
            let now = TimeStamp::compose(TimeStamp::physical_now(), 0);
            let mut last_tso = self.last_tso.lock().unwrap();
            if now > *last_tso {
                *last_tso = (last_tso.into_inner() + count as u64).into();
            } else {
                *last_tso = (now.into_inner() + count as u64).into();
            }
            future::ok(*last_tso).boxed()
        }

        fn get_keyspace_gc_safepoint_v2_cache(&self) -> Arc<DashMap<u32, u64>> {
            Arc::default()
        }

        fn get_cluster_id(&self) -> pd_client::Result<u64> {
            Ok(self.cluster_id)
        }
    }

    impl OfflinePd {
        pub async fn new(
            cluster_id: u64,
            keyspace_meta: HashMap<Vec<u8>, Vec<u8>>,
        ) -> Result<Self> {
            let tso = TimeStamp::compose(TimeStamp::physical_now(), 0).into_inner();
            Ok(Self {
                cluster_id,
                etcd: keyspace_meta.into_iter().collect(),
                last_tso: Mutex::new(tso.into()),
            })
        }

        pub fn keyspace_id(&self, keyspace: &str) -> Result<u32> {
            fail::fail_point!("offline_pd::mock_get_keyspace_by_name", |_| {
                let id: u32 = keyspace.strip_prefix("ks").unwrap().parse().unwrap();
                Ok(id)
            });
            self.etcd
                .get(&keyspace_id_key(keyspace, self.cluster_id))
                .ok_or_else(|| {
                    Error::Other(box_err!(
                        "keyspace {} not found in the backup PD meta; cluster id = {}",
                        keyspace,
                        self.cluster_id
                    ))
                })
                .and_then(|val| {
                    val.as_slice()
                        .to_str_lossy()
                        .parse()
                        .map_err(|err| Error::Other(box_err!("parse keyspace ID failed: {}", err)))
                })
        }

        pub fn get_all_keyspaces(&self) -> Result<HashMap<u32, KeyspaceMeta>> {
            let mut res = HashMap::new();

            fail::fail_point!("offline_pd::mock_get_keyspace_by_name", |_| {
                for i in 0..100 {
                    let mut meta = KeyspaceMeta::default();
                    meta.id = i;
                    meta.name = format!("ks{i}");
                    res.insert(i, meta);
                }
                Ok(res.clone())
            });

            let prefix = keyspace_prefix_key(self.cluster_id);
            for (k, v) in self.etcd.range(prefix.clone()..) {
                if !k.starts_with(&prefix) {
                    break;
                }
                let name = k[prefix.len()..].to_str_lossy();
                let id: u32 = match v.as_slice().to_str_lossy().parse() {
                    Ok(id) => id,
                    Err(err) => {
                        return Err(Error::Other(box_err!("parse keyspace ID failed: {}", err)));
                    }
                };
                let mut meta = KeyspaceMeta::default();
                meta.id = id;
                meta.name = name.to_string();
                meta.state = "UNKNOWN_BECAUSE_LOADED_FROM_BACKUP".to_owned();
                res.insert(id, meta);
            }

            Ok(res)
        }
    }

    fn keyspace_id_key(keyspace: &str, cluster_id: u64) -> Vec<u8> {
        format!("/pd/{}/keyspaces/id/{}", cluster_id, keyspace).into_bytes()
    }

    fn keyspace_prefix_key(cluster_id: u64) -> Vec<u8> {
        format!("/pd/{}/keyspaces/id/", cluster_id).into_bytes()
    }
}
