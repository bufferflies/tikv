// Copyright 2021 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    alloc::{self, Layout},
    cmp,
    fs::{File, OpenOptions},
    io::Read,
    os::unix::fs::FileExt,
    path::{Path, PathBuf},
    ptr::NonNull,
    sync::{
        atomic::{AtomicU32, Ordering},
        Arc,
    },
    time::Duration,
};

use bytes::{Buf, BufMut};
use file_system::open_direct_file;
use tikv_util::{time::Instant, warn};

use crate::{write_batch::PeerBatch, *};

// WAL file will be rotated and overwritten on every `EPOCH_ROTATE_LEN` epoches.
pub(crate) const EPOCH_ROTATE_LEN: u32 = 4;
pub(crate) const EPOCH_SNAPSHOT_LEN: u32 = 8;

pub const BATCH_HEADER_SIZE: usize = 4 /* epoch_id */ + 4 /* checksum */ + 4 /* batch_len */;
pub(crate) const INITIAL_BUF_SIZE: usize = 8 * 1024 * 1024;

#[derive(PartialEq)]
pub enum WriterType {
    Sync,
    Async,
    CliMode, // Used for cli tools like full restoration.
}

/// `DmaBuffer` is a buffer used for direct I/O that follows the alignment
/// restrictions on the length and address of user-space buffers.
///
/// The typical usage is:
///
/// ```ignore
/// let mut buf = DmaBuffer::new(16*1024);
/// let data = b"data";
/// buf.ensure_space(data.len());
/// let chunk = unsafe { buf.chunk_mut() };
/// chunk.copy_from_slice(data);
/// unsafe { buf.advance_mut(data.len()) };
/// buf.pad_to_align();
/// write(buf.as_ref());
/// ```
pub(crate) struct DmaBuffer {
    data: NonNull<u8>,
    layout: Layout,
    len: usize,
}

unsafe impl Send for DmaBuffer {}

impl DmaBuffer {
    const DMA_ALIGN: usize = 4096;

    pub(crate) fn new(cap: usize) -> Self {
        debug_assert!(0 < cap && cap <= isize::MAX as usize);
        let layout = Layout::from_size_align(cap, Self::DMA_ALIGN)
            .unwrap()
            .pad_to_align();
        let data = unsafe { alloc::alloc(layout) };
        let data = NonNull::new(data).expect("memory allocation success");
        Self {
            data,
            layout,
            len: 0,
        }
    }

    fn len(&self) -> usize {
        self.len
    }

    fn capacity(&self) -> usize {
        self.layout.size()
    }

    /// Shortens the buffer, keeping the first `len` elements and dropping
    /// the rest.
    ///
    /// If `len` is greater than the buffer's current length, this has no
    /// effect.
    fn truncate(&mut self, len: usize) {
        self.len = cmp::min(self.len, len);
    }

    /// Ensures enough space for `size`.
    fn ensure_space(&mut self, size: usize) {
        if self.capacity() - self.len >= size {
            return;
        }
        let require_cap = self
            .len
            .checked_add(size)
            .expect("capacity shouldn't overflow");
        let new_cap = cmp::max(self.layout.size() * 2, require_cap);
        let new_layout = Layout::from_size_align(new_cap, Self::DMA_ALIGN)
            .unwrap()
            .pad_to_align();
        let data = unsafe { alloc::realloc(self.data.as_ptr(), self.layout, new_layout.size()) };
        self.data = NonNull::new(data).expect("memory allocation success");
        self.layout = new_layout;
    }

    /// Pads the length of buf to the alignment. It doesn't pad zeros.
    pub(crate) fn pad_to_align(&mut self) {
        self.len = Self::aligned_len(self.len);
        assert!(self.len <= self.capacity());
    }

    /// Returns a mutable slice starting at the current position.
    ///
    /// This function is unsafe because the returned byte slice may represent
    /// uninitialized memory.
    unsafe fn chunk_mut(&mut self) -> &mut [u8] {
        &mut std::slice::from_raw_parts_mut(self.data.as_ptr(), self.capacity())[self.len..]
    }

    /// Advances the internal cursor of the Buffer.
    ///
    /// The next call to `chunk_mut` will return a slice starting `cnt` bytes
    /// further into the underlying buf.
    ///
    /// This function is unsafe because there is no guarantee that the bytes
    /// being advanced past have been initialized.
    unsafe fn advance_mut(&mut self, cnt: usize) {
        self.len += cnt;
        assert!(self.len <= self.capacity());
    }

    pub(crate) fn aligned_len(len: usize) -> usize {
        len.wrapping_add(Self::DMA_ALIGN - 1) & !(Self::DMA_ALIGN - 1)
    }
}

impl Drop for DmaBuffer {
    fn drop(&mut self) {
        unsafe {
            alloc::dealloc(self.data.as_ptr(), self.layout);
        }
    }
}

impl AsRef<[u8]> for DmaBuffer {
    fn as_ref(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.data.as_ptr(), self.len) }
    }
}

impl AsMut<[u8]> for DmaBuffer {
    fn as_mut(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.data.as_ptr(), self.len) }
    }
}

/// Magic Number of the WAL file. It's picked by running
///    echo rfengine.wal | sha1sum
/// and taking the leading 64 bits.
const WAL_MAGIC_NUMBER: u64 = 0xf126b8135c90588e;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u64)]
pub(crate) enum Version {
    V2 = 2,
}

impl Version {
    fn from(version: u64) -> Result<Version> {
        match version {
            2 => Ok(Version::V2),
            _ => Err(Error::Corruption {
                msg: format!("WAL version mismatch: version {:x}", version),
                epoch_id: 0,
                offset: 0,
                data: vec![],
            }),
        }
    }
}

#[derive(PartialEq, Eq, Debug)]
pub(crate) struct WalHeader {
    pub(crate) version: Version,
    pub(crate) epoch_id: u32,
}

impl WalHeader {
    pub(crate) fn new(version: Version, epoch_id: u32) -> Self {
        Self { version, epoch_id }
    }
}

impl WalHeader {
    pub(crate) const fn len() -> usize {
        DmaBuffer::DMA_ALIGN
    }

    fn encode_to(&self, mut buf: &mut [u8]) {
        assert!(buf.len() >= Self::len());
        buf.put_u64_le(WAL_MAGIC_NUMBER);
        buf.put_u64_le(self.version as u64);
        buf.put_u32_le(self.epoch_id);
    }

    pub(crate) fn decode(mut buf: &[u8]) -> Result<Self> {
        if buf.len() < Self::len() {
            return Err(Error::Corruption {
                msg: format!("WAL header mismatch: len {}", buf.len()),
                epoch_id: 0,
                offset: 0,
                data: buf.to_vec(),
            });
        }
        let magic_number = buf.get_u64_le();
        if magic_number != WAL_MAGIC_NUMBER {
            return Err(Error::Corruption {
                msg: format!("WAL magic number mismatch: magic_number {:x}", magic_number),
                epoch_id: 0,
                offset: 0,
                data: buf.to_vec(),
            });
        }
        let version = Version::from(buf.get_u64_le())?;
        let epoch_id = buf.get_u32_le();
        Ok(Self { version, epoch_id })
    }
}

pub(crate) fn check_wal_header(dir: &Path, epoch_id: u32) -> Result<WalHeader> {
    let filename = wal_file_name(dir, epoch_id);
    if let Ok(mut file) = File::open(filename) {
        let mut buf = vec![0u8; WalHeader::len()];
        if file.read_exact(&mut buf).is_ok() {
            return match WalHeader::decode(&buf) {
                Ok(header) => {
                    if header.epoch_id != epoch_id {
                        return Err(Error::Corruption {
                            msg: format!(
                                "WAL epoch id mismatch: header.epoch_id {} != epoch_id {}",
                                header.epoch_id, epoch_id
                            ),
                            epoch_id,
                            offset: 0,
                            data: buf.to_vec(),
                        });
                    }
                    Ok(header)
                }
                Err(err) => {
                    // Haven't written the header.
                    if buf.iter().all(|v| *v == 0) {
                        return Err(Error::Eof);
                    }
                    // Header is corrupt, but the first batch header is empty which means there
                    // is no data in this WAL. Treat it like EOF and WAL writer will rewrite the
                    // header.
                    file.read_exact(&mut buf[..BATCH_HEADER_SIZE])?;
                    if buf.iter().take(BATCH_HEADER_SIZE).all(|v| *v == 0) {
                        return Err(Error::Eof);
                    }
                    // Header corruption.
                    Err(err)
                }
            };
        }
    }
    Err(Error::Eof)
}

pub(crate) struct WalWriter {
    pub(crate) dir: PathBuf,
    pub(crate) version: Version,
    pub(crate) epoch_id: u32,
    pub(crate) wal_size: usize,
    fd: Option<File>,
    buf: DmaBuffer,
    // batch_buf is unformatted data.
    batch_buf: DmaBuffer,
    compression_threshold: usize,
    // file_off is always aligned.
    pub(crate) file_off: u64,
    pub(crate) compacted_epoch: Arc<AtomicU32>,
    pub(crate) writer_type: WriterType,
    pub(crate) write_throttle_duration: Duration,
}

impl WalWriter {
    pub(crate) fn new(
        dir: &Path,
        wal_size: usize,
        compression_threshold: usize,
        compacted_epoch: Arc<AtomicU32>,
        writer_type: WriterType,
        write_throttle_duration: Duration,
    ) -> Self {
        let version = Version::V2;
        let mut buf = DmaBuffer::new(INITIAL_BUF_SIZE);
        buf.ensure_space(BATCH_HEADER_SIZE);
        // Safety: ensured enough space and `flush` will init the header.
        unsafe {
            buf.advance_mut(BATCH_HEADER_SIZE);
        }
        Self {
            dir: dir.to_path_buf(),
            version,
            epoch_id: 0,
            wal_size: DmaBuffer::aligned_len(wal_size),
            fd: None,
            buf,
            batch_buf: DmaBuffer::new(INITIAL_BUF_SIZE),
            compression_threshold,
            file_off: 0,
            compacted_epoch,
            writer_type,
            write_throttle_duration,
        }
    }

    pub(crate) fn open_file(&mut self, epoch_id: u32, file_off: u64) -> Result<()> {
        self.epoch_id = epoch_id;
        self.file_off = file_off;

        let filename = wal_file_name(&self.dir, epoch_id);
        let file = match self.writer_type {
            WriterType::Sync => open_direct_file(&filename, true)?,
            WriterType::Async => {
                if let Some(fd) = &self.fd {
                    fd.sync_all()?;
                }
                // Must not use Direct I/O for async writer.
                // Otherwise readers (`Worker` & `ObjectStorageWorker`) using buffer I/O would
                // get incomplete data.
                OpenOptions::new()
                    .read(true)
                    .write(true)
                    .create(true)
                    .open(filename)?
            }
            WriterType::CliMode => {
                // For cli tools like full restoration, avoid using `O_DSYNC` to improve I/O
                // performance. Data in the buffer will be flushed automatically
                // when file is dropped.
                OpenOptions::new()
                    .read(true)
                    .write(true)
                    .create(true)
                    .open(filename)?
            }
        };
        self.fd = Some(file);

        if file_off == 0 {
            self.write_header()?;
        } else {
            match check_wal_header(self.dir.as_path(), epoch_id) {
                Ok(_) => {}
                Err(Error::Eof) => {
                    self.file_off = 0;
                    self.write_header()?;
                }
                Err(e) => return Err(e),
            };
        }
        Ok(())
    }

    pub(crate) fn file(&self) -> &File {
        self.fd.as_ref().unwrap()
    }

    pub(crate) fn append_region_data(&mut self, peer_batch: &PeerBatch) {
        let data_len = peer_batch.encoded_len();
        ENGINE_REGION_WRITE_BATCH_SIZE_HISTOGRAM.observe(data_len as f64);
        self.batch_buf.ensure_space(data_len);
        // Safety: `data_len` is the length of data encoded by `encode_to` and
        // `ensure_space` ensures enough space.
        unsafe {
            peer_batch.encode_to(&mut self.batch_buf.chunk_mut());
            self.batch_buf.advance_mut(data_len);
        }
    }

    pub(crate) fn flush(&mut self) -> Result<(u32, u64, bool)> {
        self.compress_batch();
        let batch = self.buf.as_mut();
        let (mut batch_header, batch_payload) = batch.split_at_mut(BATCH_HEADER_SIZE);
        let checksum = crc32c::crc32c(batch_payload);
        batch_header.put_u32_le(self.epoch_id);
        batch_header.put_u32_le(checksum);
        batch_header.put_u32_le(batch_payload.len() as u32);
        self.buf.pad_to_align();
        let aligned_len = self.buf.len();
        // An empty batch header is added after each new batch to differentiate the old
        // record.
        write_eof(&mut self.buf);

        let mut rotated = false;
        // Check should_rotate or should_chunk after put this write batch to buf avoid
        // file size overflow.
        let mut throttle_sleep = Duration::ZERO;
        if self.should_rotate() {
            while !self.safe_to_rotate() {
                std::thread::sleep(std::time::Duration::from_secs(1));
                throttle_sleep += std::time::Duration::from_secs(1);
                warn!("epoch {} is not safe to rotate", self.epoch_id);
            }
            self.rotate()?;
            // Writer epoch increased, also need update epoch_id in buf
            self.buf.as_mut().put_u32_le(self.epoch_id);
            rotated = true;
        }
        if let Some(duration) = self.need_throttle() {
            std::thread::sleep(duration);
            throttle_sleep += duration;
        }
        if throttle_sleep > Duration::ZERO {
            ENGINE_WAL_WRITE_THROTTLE_DURATION_HISTOGRAM.observe(throttle_sleep.as_secs_f64());
        }

        let timer = Instant::now();
        self.file().write_all_at(self.buf.as_ref(), self.file_off)?;
        let write_duration = timer.saturating_elapsed();
        Self::maybe_log_slow_write(write_duration, aligned_len);
        ENGINE_WAL_WRITE_DURATION_HISTOGRAM.observe(write_duration.as_secs_f64());
        self.file_off += aligned_len as u64;
        self.buf.truncate(BATCH_HEADER_SIZE);

        Ok((self.epoch_id, self.file_off, rotated))
    }

    fn maybe_log_slow_write(write_duration: Duration, aligned_len: usize) {
        if write_duration > Duration::from_millis(40) && aligned_len < 64 * 1024 {
            warn!(
                "wal write takes too long {:?}, size: {}",
                write_duration, aligned_len
            );
        }
    }

    pub(crate) fn write_batch(&mut self, wb: &WriteBatch) -> Result<(u32, u64, bool)> {
        for peer_batch in wb.peers.values() {
            self.append_region_data(peer_batch);
        }
        self.flush()
    }

    fn compress_batch(&mut self) {
        unsafe {
            let compression = self.batch_buf.len() >= self.compression_threshold;
            let compression_type = u32::from(compression);
            self.buf.ensure_space(4);
            self.buf.chunk_mut().put_u32_le(compression_type);
            self.buf.advance_mut(4);
            if compression {
                self.buf.ensure_space(4);
                self.buf.chunk_mut().put_u32_le(self.batch_buf.len() as u32);
                self.buf.advance_mut(4);
                let compress_bound = lz4::liblz4::LZ4_compressBound(self.batch_buf.len() as i32);
                self.buf.ensure_space(compress_bound as usize);
                let src = self.batch_buf.as_mut();
                let dst = self.buf.chunk_mut();
                let size = lz4::liblz4::LZ4_compress_default(
                    src.as_ptr() as *const libc::c_char,
                    dst.as_mut_ptr() as *mut libc::c_char,
                    src.len() as i32,
                    compress_bound,
                ) as usize;
                self.buf.advance_mut(size);
            } else {
                self.buf.ensure_space(self.batch_buf.len());
                self.buf.chunk_mut().put_slice(self.batch_buf.as_ref());
                self.buf.advance_mut(self.batch_buf.len());
            }
            self.batch_buf.truncate(0);
        }
    }

    fn should_rotate(&self) -> bool {
        let current_size = self.buf.len() + self.file_off as usize;
        current_size > self.wal_size && self.writer_type != WriterType::Async
    }

    // If the current epoch id is 5, the rotated epoch id is 6, it would overwrite
    // epoch 2 wal, so we need to make sure epoch 2 is compacted.
    fn safe_to_rotate(&self) -> bool {
        let compacted_epoch = self.compacted_epoch.load(Ordering::SeqCst);
        compacted_epoch + 4 > self.epoch_id
    }

    // When WAL compact is slow, we should slow down to make the compaction catch
    // up.
    // If the current epoch id is 5, the compacted_epoch can be 1, 2, 3, 4,
    // we sleep for write_throttle_duration when compacted_epoch is 2,
    // sleep for write_throttle_duration * 4 when compacted_epoch is 1.
    fn need_throttle(&self) -> Option<Duration> {
        let compacted_epoch = self.compacted_epoch.load(Ordering::SeqCst);
        if self.epoch_id < 4 {
            return None;
        }
        match (compacted_epoch + 3).cmp(&self.epoch_id) {
            cmp::Ordering::Less => Some(self.write_throttle_duration * 4),
            cmp::Ordering::Equal => Some(self.write_throttle_duration),
            cmp::Ordering::Greater => None,
        }
    }

    pub(crate) fn rotate(&mut self) -> Result<()> {
        let timer = Instant::now_coarse();
        self.open_file(self.epoch_id + 1, 0)?;
        ENGINE_ROTATE_DURATION_HISTOGRAM.observe(timer.saturating_elapsed_secs());
        Ok(())
    }

    fn write_header(&mut self) -> Result<()> {
        let mut buf = DmaBuffer::new(WalHeader::len());
        unsafe {
            let header = WalHeader::new(self.version, self.epoch_id);
            header.encode_to(buf.chunk_mut());
            buf.advance_mut(WalHeader::len());
            buf.pad_to_align();
        }
        self.file_off = buf.len() as u64;
        write_eof(&mut buf);
        self.file().write_all_at(buf.as_ref(), 0)?;
        let wal_size = self.wal_size as u64;
        self.file().set_len(wal_size)?;
        Ok(())
    }
}

pub(crate) fn write_eof(buf: &mut DmaBuffer) {
    buf.ensure_space(BATCH_HEADER_SIZE);
    unsafe {
        let chunk = buf.chunk_mut();
        chunk[..BATCH_HEADER_SIZE].fill(0);
        buf.advance_mut(BATCH_HEADER_SIZE);
    }
    buf.pad_to_align();
}

pub(crate) fn epoch_to_idx(epoch_id: u32) -> usize {
    (epoch_id % EPOCH_ROTATE_LEN) as usize
}

#[cfg(test)]
mod tests {
    use super::DmaBuffer;
    use crate::{writer::Version::V2, WalHeader};

    #[test]
    fn test_dma_buffer() {
        let mut buf = DmaBuffer::new(4095);
        assert_eq!(buf.layout.size(), DmaBuffer::aligned_len(4095));
        let addr = buf.data.as_ptr() as usize;
        assert_eq!(addr, DmaBuffer::aligned_len(addr));

        let data = b"data";
        for i in 1..=1025 {
            buf.ensure_space(data.len());
            let chunk = unsafe { buf.chunk_mut() };
            chunk[..data.len()].copy_from_slice(data);
            unsafe { buf.advance_mut(data.len()) };
            assert_eq!(buf.len(), data.len() * i);
            assert_eq!(buf.as_ref(), data.repeat(i));
            assert_eq!(buf.as_mut(), data.repeat(i));
        }
        assert_eq!(buf.layout.size(), 8192);
        let addr = buf.data.as_ptr() as usize;
        assert_eq!(addr, DmaBuffer::aligned_len(addr));
    }

    #[test]
    fn test_wal_header() {
        let wal_header = WalHeader::new(V2, 1);
        let mut buf = [0_u8; WalHeader::len()];
        wal_header.encode_to(buf.as_mut_slice());
        assert_eq!(WalHeader::decode(buf.as_slice()).unwrap(), wal_header);
    }
}
