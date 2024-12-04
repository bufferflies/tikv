// Copyright 2024 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    fmt,
    path::{Path, PathBuf},
    sync::Arc,
};

use async_trait::async_trait;
use bytes::{Bytes, BytesMut};
use log_wrappers::Value as LogValue;

use crate::{
    dfs,
    dfs::FileType,
    ia::{
        manager::{IaManager, ReadAt},
        types::FileSegmentIdent,
    },
    new_columnar_filename, new_sst_filename,
    table::{
        columnar::{ColumnarFileFooter, TableMeta, TableOffsets},
        file::{File, MmapData},
        search, sstable,
        sstable::SsTable,
        Error, Result,
    },
};

const TABLE_META_LOCAL_FILE_SUFFIX: &str = ".meta";

#[derive(Clone)]
pub struct IaFile {
    pub(crate) id: u64,
    pub(crate) size: u64,
    pub(crate) ftype: FileType,
    pub(crate) table_meta_off: u64,
    pub(crate) segment_offsets: Vec<u64>,
    pub(crate) table_meta_file: Arc<dyn File>,
    pub(crate) mgr: IaManager,
}

impl fmt::Debug for IaFile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("IaFile")
            .field("id", &self.id)
            .field("size", &self.size)
            .field("ftype", &self.ftype)
            .field("table_meta_off", &self.table_meta_off)
            .field("segment_offsets", &self.segment_offsets)
            .finish()
    }
}

impl IaFile {
    pub fn open(
        id: u64,
        ftype: FileType,
        table_meta_file: Arc<dyn File>,
        mgr: IaManager,
    ) -> Result<Self> {
        match ftype {
            FileType::Sst => Self::open_for_sst(id, ftype, table_meta_file, mgr),
            FileType::Columnar => Self::open_for_columnar(id, table_meta_file, mgr),
            _ => Err(Error::IaMgr(format!(
                "{id} open: file type not supported: {ftype:?}"
            ))),
        }
    }

    fn open_for_sst(
        id: u64,
        ftype: FileType,
        table_meta_file: Arc<dyn File>,
        mgr: IaManager,
    ) -> Result<Self> {
        let footer_data = table_meta_file.read_footer(SsTable::footer_size())?;
        let mut footer = sstable::Footer::default();
        footer.unmarshal(&footer_data);
        if !footer.is_match() {
            error!("{} open for sst: footer not match", id; "footer" => LogValue::value(&footer_data));
            if let Some(path) = table_meta_file.path() {
                if let Err(err) = std::fs::remove_file(path) {
                    warn!("{} open for sst: remove interrupted meta file: failed", id; "err" => ?err);
                }
            }
            return Err(Error::IaMgr(
                format!("{id} open for sst: footer not match",),
            ));
        }

        let meta_size = table_meta_file.size();
        let segment_size = mgr.segment_size();
        let table_meta_off = footer.index_offset as u64;
        let mut f = Self {
            id,
            size: table_meta_off + meta_size,
            ftype,
            table_meta_off,
            segment_offsets: vec![],
            table_meta_file: table_meta_file.clone(),
            mgr,
        };

        // Generate segment offsets.
        {
            let mut builder = SegmentOffsetsBuilder::new(segment_size as u64);

            let idx_data = f.read_table_meta(footer.index_offset as u64, footer.index_len())?;
            sstable::validate_checksum_with_fix(
                &idx_data,
                &footer,
                table_meta_file.as_ref(),
                None,
            )?;
            let idx = sstable::Index::new(idx_data)?;
            builder.push_from_sstable_idx(&idx);
            builder.push_boundary(footer.data_len() as u64);

            if footer.old_data_len() > 0 {
                debug_assert!(footer.old_index_len() > 0);
                let old_idx_data =
                    f.read_table_meta(footer.old_index_offset as u64, footer.old_index_len())?;
                sstable::validate_checksum_with_fix(
                    &old_idx_data,
                    &footer,
                    table_meta_file.as_ref(),
                    None,
                )?;
                let old_idx = sstable::Index::new(old_idx_data)?;
                builder.push_from_sstable_idx(&old_idx);
                builder.push_boundary((footer.data_len() + footer.old_data_len()) as u64);
            }

            f.segment_offsets = builder.finish();
        }

        debug!("{} open for sst: {:?}", id, f);
        Ok(f)
    }

    fn open_for_columnar(id: u64, table_meta_file: Arc<dyn File>, mgr: IaManager) -> Result<Self> {
        let footer_data = table_meta_file.read_footer(ColumnarFileFooter::compute_size())?;
        let footer = ColumnarFileFooter::parse(&footer_data);
        let meta_size = table_meta_file.size();
        let table_offsets_size = TableOffsets::compute_size(footer.number_tables as usize) as u64;
        let table_offsets_offset =
            meta_size - ColumnarFileFooter::compute_size() as u64 - table_offsets_size;
        let table_offsets_data =
            table_meta_file.read_table_meta(table_offsets_offset, table_offsets_size as usize)?;
        let table_offsets = TableOffsets::parse(&table_offsets_data, footer.number_tables);
        let table_meta_off = table_offsets.index_offset() as u64;
        let segment_size = mgr.segment_size();
        let mut f = Self {
            id,
            size: table_meta_off + meta_size,
            ftype: FileType::Columnar,
            table_meta_off,
            segment_offsets: vec![],
            table_meta_file: table_meta_file.clone(),
            mgr,
        };

        // Generate segment offsets.
        {
            let mut builder = SegmentOffsetsBuilder::new(segment_size as u64);
            for i in 0..footer.number_tables {
                let (idx_start, idx_end) = table_offsets.get_index_range(i as usize);
                let table_index_data = f.read_table_meta(
                    table_meta_off + idx_start as u64,
                    (idx_end - idx_start) as usize,
                )?;
                let table_meta =
                    TableMeta::parse(table_offsets.table_ids[i as usize], &table_index_data);
                builder.push_from_columnar_table_meta(&table_meta);
                builder.push_boundary(table_meta_off);
            }
            f.segment_offsets = builder.finish();
        }

        debug!("{} ia open for columnar: {:?}", id, f);
        Ok(f)
    }

    async fn read_range(&self, read_at: ReadAt<'_>) -> Result<()> {
        let (start_off, end_off) = (read_at.start_off(), read_at.end_off());

        // Read across meta should not happen.
        // But following process can still handle the read correctly.
        debug_assert!(
            end_off <= self.table_meta_off,
            "{} read across meta, end {}, file {:?}",
            self.id,
            end_off,
            self
        );

        let ident = self.align_to_segment(start_off, end_off)?;
        debug!("{} read range", self.id; "start" => start_off, "end" => end_off, "ident" => %ident);
        self.mgr.read_segment(ident, self.ftype, read_at).await
    }

    fn align_to_segment(&self, start_off: u64, end_off: u64) -> Result<FileSegmentIdent> {
        align_to_segment(self.id, &self.segment_offsets, start_off, end_off)
    }

    /// Used for remote-cop scene.
    pub async fn prepare_table_meta(
        file_id: u64,
        ftype: FileType,
        table_meta_off: u64,
        data_dir: &Path,
        dfs: &dyn dfs::Dfs,
    ) -> Result<Bytes> {
        let local_path = table_meta_file_local_path(file_id, ftype, data_dir);
        let table_meta_data = match tokio::fs::read(&local_path).await {
            Ok(bytes) => {
                if let Err(err) = filetime::set_file_mtime(&local_path, filetime::FileTime::now()) {
                    warn!("{} prepare meta: set file mtime failed", file_id; "err" => ?err);
                }
                Bytes::from(bytes)
            }
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
                let opts = dfs::Options::default()
                    .with_type(ftype)
                    .with_start_off(table_meta_off);
                let bytes = dfs.read_file(file_id, opts).await.map_err(|err| {
                    Error::IaMgr(format!("{} prepare meta: failed: {:?}", file_id, err))
                })?;
                if let Err(err) = tokio::fs::write(local_path, &bytes).await {
                    warn!("{} prepare meta: write failed", file_id; "err" => ?err);
                }
                bytes
            }
            Err(err) => {
                return Err(Error::IaMgr(format!(
                    "{} prepare meta: read failed: {:?}",
                    file_id, err
                )));
            }
        };
        Ok(table_meta_data)
    }
}

#[async_trait]
impl File for IaFile {
    fn id(&self) -> u64 {
        self.id
    }

    fn size(&self) -> u64 {
        self.size
    }

    fn is_sync(&self) -> bool {
        false
    }

    fn read(&self, _off: u64, _length: usize) -> Result<Bytes> {
        unimplemented!()
    }

    fn read_at(&self, _buf: &mut [u8], _offset: u64) -> Result<()> {
        unimplemented!()
    }

    fn read_table_meta(&self, off: u64, length: usize) -> Result<Bytes> {
        if off < self.table_meta_off || off + length as u64 > self.size {
            error!("{} read table meta: out of range", self.id;
                "off" => off, "length" => length, "file" => ?self);
            return Err(Error::IaMgr(format!(
                "{} read table meta: out of range",
                self.id
            )));
        }
        self.table_meta_file.read(off - self.table_meta_off, length)
    }

    async fn read_async(&self, off: u64, length: usize) -> Result<Bytes> {
        if off >= self.table_meta_off {
            return self.read_table_meta(off, length);
        }

        let mut buf = BytesMut::new();
        buf.resize(length, 0);
        let read_at = ReadAt::new(buf.as_mut(), off);
        self.read_range(read_at).await?;
        Ok(buf.freeze())
    }

    async fn read_at_async(&self, buf: &mut [u8], offset: u64) -> Result<()> {
        let buf_len = buf.len();
        if offset >= self.table_meta_off {
            buf.copy_from_slice(&self.read_table_meta(offset, buf_len)?);
            return Ok(());
        }

        let read_at = ReadAt::new(buf, offset);
        self.read_range(read_at).await
    }

    fn mmap(&self) -> Result<MmapData> {
        unimplemented!()
    }
}

struct SegmentOffsetsBuilder {
    segment_size: u64,
    offsets: Vec<u64>,
    last_off: u64,
}

impl SegmentOffsetsBuilder {
    fn new(segment_size: u64) -> Self {
        debug_assert!(segment_size > 0);
        Self {
            segment_size,
            offsets: vec![0],
            last_off: 0,
        }
    }

    fn push_block_off(&mut self, off: u64) {
        if off >= self.last_off + self.segment_size {
            self.push_boundary(off);
        }
    }

    fn push_boundary(&mut self, off: u64) {
        debug_assert!(off > self.last_off);
        self.offsets.push(off);
        self.last_off = off;
    }

    fn push_from_sstable_idx(&mut self, idx: &sstable::Index) {
        for pos in 0..idx.num_blocks() {
            let addr = idx.get_block_addr(pos);
            self.push_block_off(addr.curr_off as u64);
        }
    }

    fn push_from_columnar_table_meta(&mut self, meta: &TableMeta) {
        // Since the workload likely to read the whole column, we split the segment by
        // the column boundary.
        // handle column
        let (pack_start, _) = meta.handle_column.pack_offsets.get(0);
        self.push_block_off(pack_start as u64);
        // version column
        let (pack_start, _) = meta.version_column.pack_offsets.get(0);
        self.push_block_off(pack_start as u64);
        // columns
        let mut unordered_offsets = vec![];
        for col in meta.columns.values() {
            let (pack_start, _) = col.pack_offsets.get(0);
            unordered_offsets.push(pack_start as u64);
        }
        unordered_offsets.sort();
        for off in unordered_offsets {
            self.push_block_off(off);
        }
    }

    fn finish(self) -> Vec<u64> {
        self.offsets
    }
}

fn align_to_segment(
    file_id: u64,
    segment_offsets: &[u64],
    start_off: u64,
    end_off: u64,
) -> Result<FileSegmentIdent> {
    debug_assert!(!segment_offsets.is_empty());

    let segment_len = segment_offsets.len();
    if end_off <= start_off
        || start_off < segment_offsets[0]
        || segment_offsets[segment_len - 1] < end_off
    {
        return Err(Error::IaMgr(format!(
            "{} read out of range: start {}, end {}",
            file_id, start_off, end_off
        )));
    }

    let pos = search(segment_len, |i| start_off < segment_offsets[i]) - 1;
    if segment_offsets[pos + 1] < end_off {
        return Err(Error::IaMgr(format!(
            "{} read more than one segment: start {}, end {}",
            file_id, start_off, end_off
        )));
    }

    Ok(FileSegmentIdent {
        file_id,
        start_off: segment_offsets[pos],
        end_off: segment_offsets[pos + 1],
    })
}

pub fn table_meta_file_local_path(file_id: u64, file_type: FileType, data_dir: &Path) -> PathBuf {
    match file_type {
        FileType::Sst => data_dir.join(format!(
            "{}{}",
            new_sst_filename(file_id).display(),
            TABLE_META_LOCAL_FILE_SUFFIX
        )),
        FileType::Columnar => data_dir.join(format!(
            "{}{}",
            new_columnar_filename(file_id).display(),
            TABLE_META_LOCAL_FILE_SUFFIX
        )),
        _ => unimplemented!("file type not supported: {:?}", file_type),
    }
}

#[cfg(any(test, feature = "testexport"))]
impl IaFile {
    pub fn open_in_path(id: u64, ftype: FileType, data_dir: &Path, mgr: IaManager) -> Result<Self> {
        use crate::table::file::LocalFile;

        // mtime is set during prepare.
        let table_meta_file =
            LocalFile::open(id, &table_meta_file_local_path(id, ftype, data_dir), false).map_err(
                |err| Error::IaMgr(format!("{} open: open meta file failed: {:?}", id, err)),
            )?;
        Self::open(id, ftype, Arc::new(table_meta_file), mgr)
    }

    pub async fn multi_read_async(&self, off: u64, length: usize) -> Result<Bytes> {
        let mut segments = vec![];
        let end_off = off + length as u64;
        let mut seg_off = off;
        while seg_off < end_off {
            let ident = self.align_to_segment(seg_off, seg_off + 1)?;
            seg_off = ident.end_off;
            segments.push(ident);
        }

        Ok(if segments.len() == 1 {
            self.read_async(off, length).await?
        } else {
            let mut bytes = BytesMut::with_capacity(length);
            let segments_len = segments.len();
            let mut handles = Vec::with_capacity(segments_len);
            for (idx, ident) in segments.into_iter().enumerate() {
                let seg_start_off = if idx == 0 { off } else { ident.start_off };
                let seg_end_off = if idx == segments_len - 1 {
                    end_off
                } else {
                    ident.end_off
                };
                let seg_len = seg_end_off - seg_start_off;
                let ia_file = self.clone();
                let task = async move { ia_file.read_async(seg_start_off, seg_len as usize).await };
                handles.push(tokio::spawn(task));
            }

            let res = futures::future::try_join_all(handles).await.unwrap();
            for r in res {
                let r = r?;
                bytes.extend_from_slice(&r);
            }
            bytes.freeze()
        })
    }

    pub async fn multi_read_at_async(&self, mut buf: &mut [u8], off: u64) -> Result<()> {
        use bytes::BufMut as _;

        let mut segments = vec![];
        let end_off = off + buf.len() as u64;
        let mut seg_off = off;
        while seg_off < end_off {
            let ident = self.align_to_segment(seg_off, seg_off + 1)?;
            seg_off = ident.end_off;
            segments.push(ident);
        }

        if segments.len() == 1 {
            self.read_at_async(buf, off).await?;
        } else {
            let segments_len = segments.len();
            let mut handles = Vec::with_capacity(segments_len);
            for (idx, ident) in segments.into_iter().enumerate() {
                let seg_start_off = if idx == 0 { off } else { ident.start_off };
                let seg_end_off = if idx == segments_len - 1 {
                    end_off
                } else {
                    ident.end_off
                };
                let seg_len = seg_end_off - seg_start_off;
                let mut seg_buf = vec![0; seg_len as usize];
                let ia_file = self.clone();
                let task = async move {
                    ia_file
                        .read_at_async(seg_buf.as_mut_slice(), seg_start_off)
                        .await
                        .map(|()| seg_buf)
                };
                handles.push(tokio::spawn(task));
            }

            let res = futures::future::try_join_all(handles).await.unwrap();
            for r in res {
                let r = r?;
                buf.put_slice(&r);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_align_to_segment() {
        let segment_offsets = vec![0, 100, 200, 300];
        let cases = [
            // start_off, end_off, expected
            (0, 1, Some((0, 100))),
            (0, 100, Some((0, 100))),
            (0, 101, None),
            (99, 100, Some((0, 100))),
            (100, 101, Some((100, 200))),
            (101, 102, Some((100, 200))),
            (200, 300, Some((200, 300))),
            (299, 300, Some((200, 300))),
            (0, 300, None),
            (0, 1000, None),
            (1, 0, None),
        ];
        for (start, end, expected) in cases {
            let expected = expected.map(|(start_off, end_off)| FileSegmentIdent {
                file_id: 1,
                start_off,
                end_off,
            });
            assert_eq!(
                align_to_segment(1, &segment_offsets, start, end).ok(),
                expected,
            );
        }
    }
}
