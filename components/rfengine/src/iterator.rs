// Copyright 2021 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    fs,
    io::{BufReader, Read},
    path::PathBuf,
};

use bytes::{Buf, Bytes, BytesMut};
use tikv_util::error;

use crate::{
    worker::wal_file_name,
    write_batch::PeerBatch,
    writer::{DmaBuffer, WalHeader, BATCH_HEADER_SIZE},
    Error, Result,
};

pub(crate) struct WalIterator {
    dir: PathBuf,
    epoch_id: u32,
    buf: BytesMut,
    pub(crate) offset: u64,
    in_mem_reader: Option<Box<dyn Read>>,
}

const MAX_BATCH_SIZE: usize = 256 * 1024 * 1024;

impl WalIterator {
    pub(crate) fn new(dir: PathBuf, epoch_id: u32) -> Self {
        Self {
            dir,
            epoch_id,
            buf: BytesMut::new(),
            offset: 0,
            in_mem_reader: None,
        }
    }

    pub(crate) fn new_from_chunks(file_data: Bytes, epoch_id: u32) -> Self {
        Self {
            dir: PathBuf::new(),
            epoch_id,
            buf: BytesMut::new(),
            offset: 0,
            in_mem_reader: Some(Box::new(file_data.reader())),
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
        let mut buf_reader: Box<dyn std::io::Read> = if self.in_mem_iterator() {
            self.in_mem_reader.take().unwrap()
        } else {
            let filename = wal_file_name(self.dir.as_path(), self.epoch_id);
            let fd = fs::File::open(filename)?;
            Box::new(BufReader::new(fd))
        };
        match self.check_wal_header(&mut buf_reader) {
            Ok(()) => {}
            Err(Error::Eof) => {
                return Ok(());
            }
            Err(e) => return Err(e),
        };
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
