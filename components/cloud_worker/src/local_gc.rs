// Copyright 2025 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    fs,
    fs::DirEntry,
    path::{Path, PathBuf},
};

use kvengine::{
    ia::{
        ia_file::parse_table_meta_filename,
        manager::IaManager,
        types::{FileSegmentIdent, SEGMENT_LOCAL_FILE_SUFFIX, TABLE_META_LOCAL_FILE_SUFFIX},
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
}

impl Default for LocalGcConfig {
    fn default() -> Self {
        Self {
            interval: ReadableDuration::hours(1),
            meta_lifetime: ReadableDuration::days(1),
            segment_interval: ReadableDuration::days(1),
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
            |path, entry| self.handle_meta_file(path, &entry),
            TABLE_META_LOCAL_FILE_SUFFIX,
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
            self.remove_file(path)?;
            return Ok(true);
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
            |path, _| self.handle_segment(path),
            SEGMENT_LOCAL_FILE_SUFFIX,
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
            self.remove_file(path)?;
            return Ok(true);
        }
        Ok(false)
    }

    fn remove_file(&self, path: &Path) -> Result<()> {
        info!("worker gc: remove file {:?}", path);
        fs::remove_file(path).with_ctx(|| format!("remove_file.{path:?}"))?;
        Ok(())
    }

    fn walk_dir<F>(dir: &Path, mut f: F, extension: &str) -> Result<usize>
    where
        F: FnMut(&Path, DirEntry) -> Result<bool>,
    {
        let mut removed_count = 0;
        let entries = fs::read_dir(dir).with_ctx(|| format!("gc.read_dir.{dir:?}"))?;
        for e in entries {
            let entry = e.ctx("gc.entry")?;
            let path = entry.path();
            if path.extension().is_some_and(|x| x == extension) {
                match f(&path, entry) {
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
