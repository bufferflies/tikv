// Copyright 2021 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    fs,
    io::{BufReader, Cursor, Read},
    path::PathBuf,
};

use bytes::{Buf, Bytes, BytesMut};
use tikv_util::error;

use crate::{
    compact_worker::wal_file_name,
    decompress_lz4,
    write_batch::PeerBatch,
    writer::{DmaBuffer, WalHeader, BATCH_HEADER_SIZE},
    ChunkHeader, CompressionType, Error, Result, WriteBatch,
};

pub struct WalIterator {
    dir: PathBuf,
    chunks_file: Vec<PathBuf>,
    epoch_id: u32,
    buf: BytesMut,
    pub(crate) offset: u64,
    in_mem_reader: Option<Box<dyn Read>>,
}

const MAX_BATCH_SIZE: usize = 256 * 1024 * 1024;

struct WalConcatReader<F>
where
    F: Fn(&PathBuf) -> std::io::Result<Box<dyn Read>>,
{
    paths: Vec<PathBuf>,
    current: Option<Box<dyn Read>>,
    opener: F,
    extra: Option<Box<dyn Read>>,
}

impl<F> WalConcatReader<F>
where
    F: Fn(&PathBuf) -> std::io::Result<Box<dyn Read>>,
{
    #[allow(dead_code)]
    pub fn new(paths: Vec<PathBuf>, opener: F) -> Self {
        Self {
            paths,
            current: None,
            opener,
            extra: None,
        }
    }

    pub fn with_extra(paths: Vec<PathBuf>, opener: F, extra: Box<dyn Read>) -> Self {
        Self {
            paths,
            current: None,
            opener,
            extra: Some(extra),
        }
    }

    fn open_next(&mut self) -> std::io::Result<bool> {
        if let Some(path) = self.paths.first().cloned() {
            self.paths.remove(0);
            let r = (self.opener)(&path)?;
            self.current = Some(r);
            Ok(true)
        } else {
            Ok(false)
        }
    }
}

impl<F> Read for WalConcatReader<F>
where
    F: Fn(&PathBuf) -> std::io::Result<Box<dyn Read>>,
{
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        loop {
            if self.current.is_none() && !self.open_next()? {
                if let Some(extra) = &mut self.extra {
                    let n = extra.read(buf)?;
                    if n == 0 {
                        self.extra = None;
                        return Ok(0);
                    }
                    return Ok(n);
                }
                return Ok(0);
            }

            if let Some(r) = &mut self.current {
                match r.read(buf) {
                    Ok(0) => {
                        self.current = None;
                        continue;
                    }
                    other => return other,
                }
            }
        }
    }
}

fn open_file_with_decompression(path: &PathBuf) -> Result<Box<dyn Read>> {
    let file = fs::File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut header_buf = [0u8; ChunkHeader::len()];

    reader.read_exact(&mut header_buf)?;

    let header = ChunkHeader::decode(&header_buf)?;

    match header.compression_type {
        CompressionType::NoCompression => Ok(Box::new(reader)),
        CompressionType::Lz4Compression => {
            let meta_len = reader.get_ref().metadata()?.len() as usize;
            let compressed_len = meta_len.checked_sub(ChunkHeader::len()).ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "file shorter than header",
                )
            })?;

            // at most load one chunk size into memory
            let mut compressed = Vec::with_capacity(compressed_len);
            reader
                .take(compressed_len as u64)
                .read_to_end(&mut compressed)?;

            let decompressed = decompress_lz4(&compressed)?;

            Ok(Box::new(Cursor::new(decompressed)))
        }
    }
}

impl WalIterator {
    pub(crate) fn new(dir: PathBuf, epoch_id: u32) -> Self {
        Self {
            dir,
            chunks_file: vec![],
            epoch_id,
            buf: BytesMut::new(),
            offset: 0,
            in_mem_reader: None,
        }
    }

    pub fn new_from_chunks_file_and_extra_chunk(
        files: Vec<PathBuf>,
        extra_chunk: Option<Box<dyn Read>>,
        epoch_id: u32,
    ) -> Self {
        Self {
            dir: PathBuf::new(),
            chunks_file: files,
            epoch_id,
            buf: BytesMut::new(),
            offset: 0,
            in_mem_reader: extra_chunk,
        }
    }

    pub fn new_from_chunks(
        chunks_reader: Option<Box<dyn Read>>,
        epoch_id: u32,
        offset: u64,
    ) -> Self {
        Self {
            dir: PathBuf::new(),
            chunks_file: vec![],
            epoch_id,
            buf: BytesMut::new(),
            offset,
            in_mem_reader: chunks_reader,
        }
    }

    fn in_mem_iterator(&self) -> bool {
        self.in_mem_reader.is_some()
    }

    pub(crate) fn iterate_peer_batch(data: Bytes, mut f: impl FnMut(PeerBatch)) {
        let mut batch = data.chunk();
        while !batch.is_empty() {
            let peer_data = PeerBatch::decode(batch);
            batch = &batch[peer_data.encoded_len()..];
            f(peer_data);
        }
    }

    pub(crate) fn iterate_batch<F>(&mut self, mut f: F) -> Result<()>
    where
        F: FnMut(Bytes, u64),
    {
        let mut buf_reader: Box<dyn Read> = if !self.chunks_file.is_empty() {
            // chain all chunk files and combine the last chunk(if have) into one reader.
            let extra = if self.in_mem_iterator() {
                self.in_mem_reader.take().unwrap()
            } else {
                Box::new(std::io::empty())
            };

            Box::new(WalConcatReader::with_extra(
                self.chunks_file.clone(),
                |p| {
                    let f = open_file_with_decompression(p).map_err(|e| {
                        std::io::Error::new(
                            std::io::ErrorKind::NotFound,
                            format!("open_file_with_decompression failed: {}", e),
                        )
                    })?;
                    Ok(Box::new(f) as Box<dyn Read>)
                },
                extra,
            ))
        } else if self.in_mem_iterator() {
            // the chunk files has been combine outside, just take it
            // Note: This method may cause OOM, if load all chunks into Memory when doing
            // restore with to many stores.
            self.in_mem_reader.take().unwrap()
        } else {
            let filename = wal_file_name(self.dir.as_path(), self.epoch_id);
            let fd = fs::File::open(filename)?;
            Box::new(BufReader::new(fd))
        };
        if self.offset == 0 {
            match self.check_wal_header(&mut buf_reader) {
                Ok(()) => {}
                Err(Error::Eof) => {
                    return Ok(());
                }
                Err(e) => return Err(e),
            };
        }
        loop {
            match self.read_batch(&mut buf_reader) {
                Err(err) => {
                    if let Error::Eof = err {
                        return Ok(());
                    }
                    return Err(err);
                }
                Ok(data) => {
                    if data.is_empty() {
                        return Ok(());
                    }
                    f(data, self.offset);
                }
            }
        }
    }

    pub fn iterate_write_batch<F>(&mut self, mut f: F) -> Result<()>
    where
        F: FnMut(WriteBatch),
    {
        self.iterate_batch(|data, _| {
            let mut wb = WriteBatch::new();
            WalIterator::iterate_peer_batch(data, |peer_batch| {
                wb.peers.insert(peer_batch.peer_id, peer_batch);
            });
            f(wb);
        })
    }

    pub(crate) fn check_wal_header(&mut self, reader: &mut Box<dyn std::io::Read>) -> Result<()> {
        let mut buf = [0u8; WalHeader::len()];
        reader.read_exact(&mut buf)?;
        self.offset += WalHeader::len() as u64;
        match WalHeader::decode(&buf) {
            Ok(header) => {
                if header.epoch_id != self.epoch_id {
                    return Err(Error::Corruption {
                        msg: format!(
                            "check wal header: epoch mismatch: header.epoch_id {} != self.epoch_id {}",
                            header.epoch_id, self.epoch_id
                        ),
                        epoch_id: header.epoch_id,
                        offset: self.offset,
                        data: buf.to_vec(),
                    });
                }
                Ok(())
            }
            Err(err) => {
                // Haven't written the header.
                if buf.iter().all(|v| *v == 0) {
                    return Err(Error::Eof);
                }
                // Header is corrupt, but the first batch header is empty which means there
                // is no data in this WAL. Treat it like EOF and WAL writer will rewrite the
                // header.
                reader.read_exact(&mut buf[..BATCH_HEADER_SIZE])?;
                if buf.iter().take(BATCH_HEADER_SIZE).all(|v| *v == 0) {
                    return Err(Error::Eof);
                }
                // Header corruption.
                Err(err)
            }
        }
    }

    pub(crate) fn read_batch(&mut self, reader: &mut Box<dyn std::io::Read>) -> Result<Bytes> {
        let mut header_array = [0u8; BATCH_HEADER_SIZE];
        reader.read_exact(header_array.as_mut_slice())?;
        let mut header_buf = header_array.as_slice();
        let epoch_id = header_buf.get_u32_le();
        let checksum = header_buf.get_u32_le();
        let length = header_buf.get_u32_le() as usize;
        if epoch_id == 0 && checksum == 0 && length == 0 {
            return Err(Error::Eof);
        }
        if epoch_id != self.epoch_id {
            return Err(Error::Corruption {
                msg: format!(
                    "read batch: epoch mismatch: header.epoch_id {} != self.epoch_id {}",
                    epoch_id, self.epoch_id
                ),
                epoch_id,
                offset: self.offset,
                data: header_array.to_vec(),
            });
        }
        if length > MAX_BATCH_SIZE {
            return Err(Error::Corruption {
                msg: format!("length mismatch: length {}", length),
                epoch_id,
                offset: self.offset,
                data: header_array.to_vec(),
            });
        }
        let aligned_length = DmaBuffer::aligned_len(BATCH_HEADER_SIZE + length);
        let remained_length = aligned_length - BATCH_HEADER_SIZE;
        self.buf.resize(remained_length, 0);
        reader.read_exact(&mut self.buf[..])?;
        let batch = &self.buf[..length];
        let actual_checksum = crc32c::crc32c(batch);
        if checksum != actual_checksum {
            error!("read_batch:checksum mismatch";
                "epoch_id" => epoch_id,
                "checksum" => checksum,
                "actual_checksum" => actual_checksum,
                "length" => length,
                "aligned_length" => aligned_length,
                "remained_length" => remained_length,
                "self.offset" => self.offset,
                "header" => log_wrappers::hex_encode_upper(header_array),
                "batch" => log_wrappers::hex_encode_upper(batch),
            );
            return Err(Error::Corruption {
                msg: format!(
                    "checksum mismatch: header.checksum {:x}, batch.checksum {:x}",
                    checksum, actual_checksum
                ),
                epoch_id,
                offset: self.offset,
                data: batch.to_vec(),
            });
        }
        self.offset += aligned_length as u64;
        let (mut compression_type, batch_data) = batch.split_at(4);
        let compression = compression_type.get_u32_le() > 0;
        if compression {
            let dst = lz4::block::decompress(batch_data, None)?;
            Ok(Bytes::from(dst))
        } else {
            Ok(Bytes::from(batch_data.to_vec()))
        }
    }
}

#[cfg(test)]
mod tests {
    use std::io::{Read, Write};

    use tempfile::NamedTempFile;

    use super::*;

    fn make_header_bytes(kind: CompressionType) -> Vec<u8> {
        let header = match kind {
            CompressionType::NoCompression => ChunkHeader::new(CompressionType::NoCompression),
            CompressionType::Lz4Compression => ChunkHeader::new(CompressionType::Lz4Compression),
        };
        let mut buf = Vec::with_capacity(ChunkHeader::len());
        header.encode_to(&mut buf);
        buf
    }

    fn make_lz4_block_file(payload: &[u8]) -> std::io::Result<NamedTempFile> {
        let mut f = NamedTempFile::new()?;
        let header = make_header_bytes(CompressionType::Lz4Compression);
        let compressed = lz4::block::compress(payload, None, /* include_size */ true)
            .expect("lz4 block compress");

        f.write_all(&header)?;
        f.write_all(&compressed)?;
        f.flush()?;
        Ok(f)
    }

    #[test]
    fn read_no_compression_file_returns_payload() {
        let payload = b"hello wal chunk".to_vec();
        let header = make_header_bytes(CompressionType::NoCompression);

        let mut file_data = header.clone();
        file_data.extend_from_slice(&payload);

        let mut file = NamedTempFile::new().unwrap();
        file.write_all(&file_data).unwrap();
        file.flush().unwrap();
        let path = file.path().to_path_buf();

        let mut reader =
            super::open_file_with_decompression(&path).expect("open & select decompressor");

        let mut out = Vec::new();
        reader.read_to_end(&mut out).expect("read_to_end");

        assert_eq!(out, payload);
    }

    #[test]
    fn read_lz4_block_file_returns_decompressed_payload() {
        use std::io::Read;

        let payload = b"hello wal chunk".to_vec();
        let f = make_lz4_block_file(&payload).unwrap();
        let path = f.path().to_path_buf();

        let mut reader =
            super::open_file_with_decompression(&path).expect("open & select decompressor");

        let mut out = Vec::new();
        reader.read_to_end(&mut out).expect("read_to_end");

        assert_eq!(out, payload);
    }

    #[test]
    fn test_wal_concat_reader_with_extra() -> std::io::Result<()> {
        let mut f1 = NamedTempFile::new()?;
        let mut f2 = NamedTempFile::new()?;
        write!(f1, "Hello")?;
        write!(f2, "World")?;

        let path1 = f1.path().to_path_buf();
        let path2 = f2.path().to_path_buf();
        let files = vec![path1, path2];

        let opener = |p: &PathBuf| -> std::io::Result<Box<dyn Read>> {
            let f = std::fs::File::open(p)?;
            Ok(Box::new(f))
        };

        let extra = Cursor::new(b"!Extra".to_vec());

        let mut reader = WalConcatReader::with_extra(files, opener, Box::new(extra));

        let mut buf = String::new();
        reader.read_to_string(&mut buf)?;

        assert_eq!(buf, "HelloWorld!Extra");
        Ok(())
    }

    #[test]
    fn test_lazy_concat_reader_lz4_with_extra() -> std::io::Result<()> {
        let f1 = make_lz4_block_file(b"hello ")?;
        let f2 = make_lz4_block_file(b"wal chunk")?;
        let files: Vec<PathBuf> = vec![f1.path().into(), f2.path().into()];

        let opener = |p: &PathBuf| -> std::io::Result<Box<dyn Read>> {
            let mut bytes = Vec::new();
            std::fs::File::open(p)?.read_to_end(&mut bytes)?;

            let header_len = make_header_bytes(CompressionType::Lz4Compression).len();
            if bytes.len() < header_len {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "header too short",
                ));
            }
            let compressed = &bytes[header_len..];

            let decompressed = lz4::block::decompress(compressed, None).map_err(|e| {
                std::io::Error::new(std::io::ErrorKind::InvalidData, format!("lz4: {e}"))
            })?;

            Ok(Box::new(Cursor::new(decompressed)) as Box<dyn Read>)
        };

        let extra = Cursor::new(b"!EXTRA".to_vec());

        let mut reader = WalConcatReader::new(files.clone(), opener);
        let mut out = String::new();
        reader.read_to_string(&mut out)?;
        assert_eq!(out, "hello wal chunk");

        let mut extra_reader = WalConcatReader::with_extra(files, opener, Box::new(extra));
        let mut extra_out = String::new();
        extra_reader.read_to_string(&mut extra_out)?;

        assert_eq!(extra_out, "hello wal chunk!EXTRA");

        Ok(())
    }
}
