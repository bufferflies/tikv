// Copyright 2025 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    ffi::OsStr,
    fs,
    fs::DirEntry,
    io,
    path::{Path, PathBuf},
};

use kvengine::{
    ia::{
        ia_file::parse_table_meta_filename,
        manager::IaManager,
        types::{FileSegmentIdent, SEGMENT_LOCAL_FILE_SUFFIX, TABLE_META_LOCAL_FILE_SUFFIX},
        util::TEMPORARY_FILE_SUFFIX,
    },
    IoContext,
};
use tikv_util::{box_err, config::ReadableDuration, error, info, time::Instant, warn};

use crate::common::Running;

type Error = Box<dyn std::error::Error + Sync + Send>;
type Result<T> = std::result::Result<T, Error>;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(default)]
#[serde(rename_all = "kebab-case")]
pub struct LocalGcConfig {
    pub interval: ReadableDuration,
    pub meta_lifetime: ReadableDuration,
    pub segment_interval: ReadableDuration,
    pub segment_tmp_lifetime: ReadableDuration,
}

impl Default for LocalGcConfig {
    fn default() -> Self {
        Self {
            interval: ReadableDuration::hours(1),
            meta_lifetime: ReadableDuration::days(1),
            segment_interval: ReadableDuration::days(1),
            segment_tmp_lifetime: ReadableDuration::minutes(1),
        }
    }
}

pub struct LocalGcRunner {
    config: LocalGcConfig,
    ia_mgr: IaManager,

    meta_path: PathBuf,

    segment_path: Option<PathBuf>,
    segment_last_gc_time: Instant,
}

impl LocalGcRunner {
    pub fn new(config: LocalGcConfig, ia_mgr: IaManager, meta_path: PathBuf) -> Self {
        let segment_path = ia_mgr.main_store_path().map(|x| x.to_path_buf());
        Self {
            config,
            ia_mgr,
            meta_path,
            segment_path,
            segment_last_gc_time: Instant::now_coarse(),
        }
    }

    pub fn run(&mut self, running: Running) {
        info!("worker gc runner start"; "config" => ?self.config);

        while running.get() {
            if let Err(err) = self.meta_file_gc() {
                error!("worker gc: meta file gc failed"; "err" => ?err);
                debug_assert!(false, "meta file gc failed: {:?}", err);
            }
            if let Err(err) = self.segment_gc() {
                error!("worker gc: segment gc failed"; "err" => ?err);
                debug_assert!(false, "segment gc failed: {:?}", err);
            }
            std::thread::sleep(self.config.interval.0);
        }

        info!("worker gc runner stopped");
    }

    pub fn meta_file_gc(&mut self) -> Result<usize> {
        Self::walk_dir(
            &self.meta_path,
            &[TABLE_META_LOCAL_FILE_SUFFIX],
            |_extension, path, entry| self.handle_meta_file(path, &entry),
        )
    }

    fn handle_meta_file(&self, path: &Path, entry: &DirEntry) -> Result<bool /* is_removed */> {
        let filename = path
            .file_name()
            .ok_or_else(|| -> Error { box_err!("no filename: {path:?}") })?;
        let filename = filename.to_string_lossy();
        let Some((file_id, _)) = parse_table_meta_filename(filename.as_ref()) else {
            debug_assert!(false, "invalid filename of table meta: {:?}", path);
            warn!("worker gc: invalid filename of table meta"; "path" => ?path);
            return Ok(false);
        };
        let metadata = entry.metadata().ctx("gc.metadata")?;
        let modified_dur = metadata
            .modified()
            .ctx("gc.modified")?
            .elapsed()
            .unwrap_or_default();
        if modified_dur > self.config.meta_lifetime.0 {
            self.ia_mgr.remove_table_meta(file_id);
            let is_removed = self
                .remove_file(path)
                .with_ctx(|| format!("remove_meta.{}", path.display()))?;
            debug_assert!(is_removed, "meta file not removed: {}:{:?}", file_id, path);
            return Ok(is_removed);
        }
        Ok(false)
    }

    pub fn segment_gc(&mut self) -> Result<usize> {
        let Some(segment_path) = self.segment_path.as_ref() else {
            return Ok(0);
        };
        if self.segment_last_gc_time.saturating_elapsed() < self.config.segment_interval.0 {
            return Ok(0);
        }

        self.segment_last_gc_time = Instant::now_coarse();
        Self::walk_dir(
            segment_path,
            &[SEGMENT_LOCAL_FILE_SUFFIX, TEMPORARY_FILE_SUFFIX],
            |extension, path, entry| {
                if extension == SEGMENT_LOCAL_FILE_SUFFIX {
                    self.handle_segment(path)
                } else if extension == TEMPORARY_FILE_SUFFIX {
                    self.handle_segment_temp(path, &entry)
                } else {
                    unreachable!()
                }
            },
        )
    }

    fn handle_segment(&self, path: &Path) -> Result<bool /* is_removed */> {
        let file_name = path
            .file_name()
            .ok_or_else(|| format!("gc.file_name.{path:?}"))?;
        let file_name = file_name.to_string_lossy();
        let Some(segment_ident) = FileSegmentIdent::parse_local_filename(file_name.as_ref()) else {
            debug_assert!(false, "illegal segment file name: {:?}", file_name.as_ref());
            return Ok(false);
        };
        if !self.ia_mgr.contains_segment(&segment_ident) {
            warn!("worker gc: segment file is leaked"; "path" => ?path, "segment" => %segment_ident);
            let is_removed = self
                .remove_file(path)
                .with_ctx(|| format!("remove_segment.{}", path.display()))?;
            return Ok(is_removed);
        }
        Ok(false)
    }

    fn handle_segment_temp(&self, path: &Path, entry: &DirEntry) -> Result<bool /* is_removed */> {
        let metadata = entry.metadata().ctx("gc.metadata")?;
        let modified_dur = metadata
            .modified()
            .ctx("gc.modified")?
            .elapsed()
            .unwrap_or_default();
        if modified_dur >= self.config.segment_tmp_lifetime.0 {
            let is_removed = self
                .remove_file(path)
                .with_ctx(|| format!("remove_seg_tmp.{}", path.display()))?;
            return Ok(is_removed);
        }
        Ok(false)
    }

    fn remove_file(&self, path: &Path) -> io::Result<bool /* is_removed */> {
        info!("worker gc: remove file {:?}", path);
        match fs::remove_file(path) {
            Ok(()) => Ok(true),
            Err(err) if err.kind() == io::ErrorKind::NotFound => {
                // Removed by segment eviction of IA manager.
                info!("worker gc: remove file not found: {:?}", path);
                Ok(false)
            }
            Err(err) => Err(err),
        }
    }

    fn walk_dir<F>(dir: &Path, extensions: &[&str], mut f: F) -> Result<usize>
    where
        F: FnMut(&OsStr /* extension */, &Path, DirEntry) -> Result<bool>,
    {
        let mut removed_count = 0;
        let entries = fs::read_dir(dir).with_ctx(|| format!("gc.read_dir.{dir:?}"))?;
        for e in entries {
            let entry = e.ctx("gc.entry")?;
            let path = entry.path();
            let Some(ext) = path.extension() else {
                continue;
            };
            if extensions.iter().any(|&x| x == ext) {
                match f(ext, &path, entry) {
                    Ok(is_removed) => removed_count += is_removed as usize,
                    Err(err) => {
                        error!("worker gc: handle file failed"; "err" => ?err, "path" => ?path);
                        debug_assert!(false, "handle file failed: {:?}: {:?}", path, err);
                    }
                }
            }
        }
        Ok(removed_count)
    }

    // For test purpose.
    #[cfg(any(test, feature = "testexport"))]
    pub fn set_segment_path(&mut self, path: PathBuf) {
        self.segment_path = Some(path);
    }
}
