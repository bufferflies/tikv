// Copyright 2021 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    alloc::{self, Layout},
    cmp,
    fs::File,
    os::unix::prelude::FileExt,
    path::{Path, PathBuf},
    ptr::NonNull,
    sync::{
        atomic::{AtomicU32, Ordering},
        Arc,
    },
};

use bytes::{Buf, BufMut};
use file_system::open_direct_file;
use tikv_util::time::Instant;

use crate::{write_batch::PeerBatch, *};

pub const BATCH_HEADER_SIZE: usize = 4 /* epoch_id */ + 4 /* checksum */ + 4 /* batch_len */;
pub(crate) const INITIAL_BUF_SIZE: usize = 8 * 1024 * 1024;

/// `DmaBuffer` is a buffer used for direct I/O that follows the alignment restrictions
/// on the length and address of user-space buffers.
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

    fn new(cap: usize) -> Self {
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
    fn pad_to_align(&mut self) {
        self.len = Self::aligned_len(self.len);
        assert!(self.len <= self.capacity());
    }

    /// Returns a mutable slice starting at the current position.
    ///
    /// This function is unsafe because the returned byte slice may represent uninitialized memory.
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
enum Version {
    V1 = 1,
}

#[derive(PartialEq, Eq, Debug)]
pub(crate) struct WalHeader {
    version: Version,
    pub(crate) epoch_id: u32,
}

impl WalHeader {
    pub(crate) fn new(epoch_id: u32) -> Self {
        Self {
            version: Version::V1,
            epoch_id,
        }
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
            return Err(Error::Corruption("WAL header mismatch".to_owned()));
        }
        let magic_number = buf.get_u64_le();
        if magic_number != WAL_MAGIC_NUMBER {
            return Err(Error::Corruption("WAL magic number mismatch".to_owned()));
        }
        let version = buf.get_u64_le();
        if version != Version::V1 as u64 {
            return Err(Error::Corruption("WAL version mismatch".to_owned()));
        }
        let epoch_id = buf.get_u32_le();
        Ok(Self {
            version: Version::V1,
            epoch_id,
        })
    }
}

pub(crate) struct WalWriter {
    dir: PathBuf,
    pub(crate) epoch_id: u32,
    pub(crate) wal_size: usize,
    fd: Option<File>,
    buf: DmaBuffer,
    // file_off is always aligned.
    file_off: u64,
    pub(crate) compacted_epoch: Arc<AtomicU32>,
}

impl WalWriter {
    pub(crate) fn new(dir: &Path, wal_size: usize, compacted_epoch: Arc<AtomicU32>) -> Self {
        let mut buf = DmaBuffer::new(INITIAL_BUF_SIZE);
        buf.ensure_space(BATCH_HEADER_SIZE);
        // Safety: ensured enough space and `flush` will init the header.
        unsafe {
            buf.advance_mut(BATCH_HEADER_SIZE);
        }
        let writer = Self {
            dir: dir.to_path_buf(),
            epoch_id: 0,
            wal_size: DmaBuffer::aligned_len(wal_size),
            fd: None,
            buf,
            file_off: 0,
            compacted_epoch,
        };
        writer
    }

    pub(crate) fn open_file(&mut self, epoch_id: u32, file_off: u64) -> Result<()> {
        self.epoch_id = epoch_id;
        self.file_off = file_off;
        let file = open_direct_file(&wal_file_name(&self.dir, epoch_id), true)?;
        self.fd = Some(file);
        if file_off == 0 {
            self.write_header()?
        }
        Ok(())
    }

    fn file(&self) -> &File {
        self.fd.as_ref().unwrap()
    }

    pub(crate) fn append_region_data(&mut self, peer_batch: &PeerBatch) {
        let data_len = peer_batch.encoded_len();
        self.buf.ensure_space(data_len);
        // Safety: `data_len` is the length of data encoded by `encode_to` and
        // `ensure_space` ensures enough space.
        unsafe {
            peer_batch.encode_to(&mut self.buf.chunk_mut());
            self.buf.advance_mut(data_len);
        }
    }

    pub(crate) fn flush(&mut self) -> Result<(usize, bool)> {
        let mut rotated = false;
        if self.should_rotate() {
            self.rotate()?;
            rotated = true;
        }

        let data_len = self.buf.len();
        let batch = self.buf.as_mut();
        let (mut batch_header, batch_payload) = batch.split_at_mut(BATCH_HEADER_SIZE);
        let checksum = crc32c::crc32c(batch_payload);
        batch_header.put_u32_le(self.epoch_id);
        batch_header.put_u32_le(checksum);
        batch_header.put_u32_le(batch_payload.len() as u32);
        self.buf.pad_to_align();
        let aligned_len = self.buf.len();
        // An empty batch header is added after each new batch to differentiate the old record.
        write_eof(&mut self.buf);

        let timer = Instant::now_coarse();
        self.file().write_all_at(self.buf.as_ref(), self.file_off)?;
        ENGINE_WAL_WRITE_DURATION_HISTOGRAM.observe(timer.saturating_elapsed_secs());
        self.file_off += aligned_len as u64;
        self.buf.truncate(BATCH_HEADER_SIZE);

        Ok((data_len, rotated))
    }

    fn should_rotate(&self) -> bool {
        let eof_len = DmaBuffer::aligned_len(BATCH_HEADER_SIZE);
        let current_size =
            DmaBuffer::aligned_len(self.buf.len()) + eof_len + self.file_off as usize;
        let compacted_epoch = self.compacted_epoch.load(Ordering::SeqCst);
        // If the current epoch id is 5, the rotated epoch id is 6, it would overwrite epoch 2 wal,
        // so we need to make sure epoch 2 is compacted.
        current_size > self.wal_size && compacted_epoch + 4 > self.epoch_id
    }

    fn rotate(&mut self) -> Result<()> {
        let timer = Instant::now_coarse();
        self.open_file(self.epoch_id + 1, 0)?;
        ENGINE_ROTATE_DURATION_HISTOGRAM.observe(timer.saturating_elapsed_secs());
        Ok(())
    }

    fn write_header(&mut self) -> Result<()> {
        let mut buf = DmaBuffer::new(WalHeader::len());
        unsafe {
            let header = WalHeader::new(self.epoch_id);
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
    (epoch_id % 4) as usize
}

#[cfg(test)]
mod tests {
    use super::DmaBuffer;
    use crate::WalHeader;

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
        let wal_header = WalHeader::new(1);
        let mut buf = [0_u8; WalHeader::len()];
        wal_header.encode_to(buf.as_mut_slice());
        assert_eq!(WalHeader::decode(buf.as_slice()).unwrap(), wal_header);
    }
}
