// Copyright 2021 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    io::{BufReader, Read},
    path::PathBuf,
    sync::Arc,
};

use bytes::{Buf, Bytes, BytesMut};
use file_system::IoRateLimiter;

use crate::{
    worker::wal_file_name,
    write_batch::PeerBatch,
    writer::{DmaBuffer, WalHeader, BATCH_HEADER_SIZE},
    Error, Result, Version,
};

pub(crate) struct WalIterator {
    dir: PathBuf,
    epoch_id: u32,
    buf: BytesMut,
    pub(crate) offset: u64,
    rate_limiter: Option<Arc<IoRateLimiter>>,
}

const MAX_BATCH_SIZE: usize = 256 * 1024 * 1024;

impl WalIterator {
    pub(crate) fn new(
        dir: PathBuf,
        epoch_id: u32,
        rate_limiter: Option<Arc<IoRateLimiter>>,
    ) -> Self {
        Self {
            dir,
            epoch_id,
            buf: BytesMut::new(),
            offset: 0,
            rate_limiter,
        }
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
        F: FnMut(Bytes),
    {
        let filename = wal_file_name(self.dir.as_path(), self.epoch_id);
        let fd = file_system::File::open_with_limiter(filename, self.rate_limiter.clone())?;
        let mut buf_reader = BufReader::new(fd);
        let header = match self.check_wal_header(&mut buf_reader) {
            Ok(header) => header,
            Err(Error::Eof) => {
                return Ok(());
            }
            Err(e) => return Err(e),
        };
        loop {
            match self.read_batch(&mut buf_reader, &header) {
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
                    f(data);
                }
            }
        }
    }

    pub(crate) fn check_wal_header(
        &mut self,
        reader: &mut BufReader<file_system::File>,
    ) -> Result<WalHeader> {
        let mut buf = [0u8; WalHeader::len()];
        reader.read_exact(&mut buf)?;
        self.offset += WalHeader::len() as u64;
        match WalHeader::decode(&buf) {
            Ok(header) => Ok(header),
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

    pub(crate) fn read_batch(
        &mut self,
        reader: &mut BufReader<file_system::File>,
        header: &WalHeader,
    ) -> Result<Bytes> {
        let mut header_buf = [0u8; BATCH_HEADER_SIZE];
        reader.read_exact(header_buf.as_mut_slice())?;
        let mut header_buf = header_buf.as_slice();
        let epoch_id = header_buf.get_u32_le();
        let checksum = header_buf.get_u32_le();
        let length = header_buf.get_u32_le() as usize;
        if epoch_id == 0 && checksum == 0 && length == 0 {
            return Err(Error::Eof);
        }
        if epoch_id != self.epoch_id {
            return Err(Error::Corruption("epoch mismatch".to_owned()));
        }
        if length > MAX_BATCH_SIZE {
            return Err(Error::Corruption("length mismatch".to_owned()));
        }
        let aligned_length = DmaBuffer::aligned_len(BATCH_HEADER_SIZE + length);
        let remained_length = aligned_length - BATCH_HEADER_SIZE;
        self.buf.resize(remained_length, 0);
        reader.read_exact(&mut self.buf[..])?;
        let batch = &self.buf[..length];
        if checksum != crc32c::crc32c(batch) {
            return Err(Error::Corruption("checksum mismatch".to_owned()));
        }
        self.offset += aligned_length as u64;
        match header.version {
            Version::V1 => Ok(Bytes::from(batch.to_vec())),
            Version::V2 => {
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
    }
}
