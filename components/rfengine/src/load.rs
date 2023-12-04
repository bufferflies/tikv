// Copyright 2021 TiKV Project Authors. Licensed under Apache-2.0.

use std::{fs, path::Path};

use byteorder::{ByteOrder, LittleEndian};
use bytes::{Buf, Bytes};
use tikv_util::info;

use crate::{log_batch::RaftLogOp, manifest::Manifest, *};

impl RfEngineCore {
    pub(crate) fn load(&mut self, manifest: &Manifest) -> Result<u64> {
        for (&peer_id, peer_meta) in &manifest.peers {
            let peer_ref = self.get_or_init_peer_data(peer_id, peer_meta.region_id);
            let mut peer_data = peer_ref.write().unwrap();
            peer_data.meta.merge(peer_meta, false);
            drop(peer_data);
            drop(peer_ref);
            for file in &peer_meta.files {
                self.load_raft_log_file(
                    peer_id,
                    peer_meta.region_id,
                    file.first_index,
                    file.last_index,
                )?;
            }
        }
        let mut epoch_id = manifest.epoch_id + 1;
        let mut wal_offset = 0;
        let mut async_offset = 0;
        if wal_exists(self.wal_dir(), epoch_id) {
            (wal_offset, async_offset) = self.load_wal_file(epoch_id, true)?;
        }
        while wal_exists(self.wal_dir(), epoch_id + 1) {
            self.task_sender.send(Task::Rotate { epoch_id }).unwrap();
            epoch_id += 1;
            let (offset, _) = self.load_wal_file(epoch_id, false)?;
            wal_offset = offset;
        }
        let mut writer = self.writer.lock().unwrap();
        writer.open_file(epoch_id, wal_offset)?;
        Ok(async_offset)
    }

    pub(crate) fn load_wal_file(&mut self, epoch_id: u32, load_async: bool) -> Result<(u64, u64)> {
        info!("load wal {}", epoch_id);
        let mut async_batch_cnt = 0;
        let mut async_offset = 0;
        if self.is_async_wal_enabled() && load_async {
            let mut async_it = WalIterator::new(self.dir.to_path_buf(), epoch_id);
            async_it.iterate_batch(|_, _| {
                async_batch_cnt += 1;
            })?;
            async_offset = async_it.offset;
        }
        let mut sync_batch_idx = 0;
        let mut it = WalIterator::new(self.wal_dir().to_path_buf(), epoch_id);
        it.iterate_batch(|data, _| {
            sync_batch_idx += 1;
            let mut wb = if self.is_async_wal_enabled() && sync_batch_idx > async_batch_cnt {
                Some(WriteBatch::new())
            } else {
                None
            };
            WalIterator::iterate_peer_batch(data, |peer_batch| {
                let peer_ref =
                    self.get_or_init_peer_data(peer_batch.peer_id, peer_batch.meta.region_id);
                let mut peer_data = peer_ref.write().unwrap();
                let _ = peer_data.apply(&peer_batch);
                if let Some(wb) = &mut wb {
                    wb.peers.insert(peer_batch.peer_id, peer_batch);
                }
            });
            if let Some(wb) = wb {
                self.task_sender.send(Task::Write { wb }).unwrap();
            }
        })?;
        info!("load wal done, it.offset {}", it.offset);
        Ok((it.offset, async_offset))
    }

    // `replay_wal_file` replay a wal chunk data to rfengine. It's used for
    // lightweight restoration to recover a `BackupCluster` from an previous
    // snapshot.
    pub fn replay_wal_file(
        &self,
        file_data: Bytes,
        epoch_id: u32,
        end_offset: u64, // u64::MAX means replay all the chunk
        full_restore: bool,
    ) -> Result<()> {
        let mut it = WalIterator::new_from_chunks(file_data, epoch_id);
        it.iterate_batch(|data, offset| {
            // `offset` is the data read position after `data` be read.
            if offset > end_offset {
                return;
            }
            let mut wb = WriteBatch::new();
            WalIterator::iterate_peer_batch(data, |peer_batch| {
                wb.peers.insert(peer_batch.peer_id, peer_batch);
            });
            if full_restore {
                self.write(wb).unwrap();
            } else {
                // In scene of restore keyspace, we don't persist WAL to avoid unnecessray I/O,
                // as the rfengine is only used temporarily during the restoration process.
                self.apply(&mut wb);
            }
        })?;
        Ok(())
    }

    pub(crate) fn load_raft_log_file(
        &mut self,
        peer_id: u64,
        region_id: u64,
        first: u64,
        last: u64,
    ) -> Result<()> {
        let rlog_filename = raft_log_file_name(&self.dir, peer_id, first, last);
        let bin = fs::read(rlog_filename)?;
        // See format in Worker::write_raft_log_file
        let header = RlogHeader::decode(bin.as_slice())?;
        let mut data = &bin[RlogHeader::len()..];
        let mut end_offs = vec![];
        for _ in 0..header.count {
            end_offs.push(data.get_u32_le());
        }
        let peer_data_ref = self.get_or_init_peer_data(peer_id, region_id);
        let mut peer_data = peer_data_ref.write().unwrap();
        for i in 0..header.count as usize {
            if first + i as u64 <= peer_data.truncated_idx {
                continue;
            }
            let start_off = if i == 0 {
                0usize
            } else {
                end_offs[i - 1] as usize
            };
            let end_off = end_offs[i] as usize;
            let log_data = &data[start_off..end_off - 4];
            let checksum = LittleEndian::read_u32(&data[end_off - 4..]);
            let actual_checksum = crc32c::crc32c(log_data);
            if checksum != actual_checksum {
                return Err(Error::Corruption {
                    msg: format!(
                        "checksum mismatch: header.checksum {:x}, log_data.checksum {:x}",
                        checksum, actual_checksum
                    ),
                    epoch_id: 0,
                    offset: start_off as u64,
                    data: log_data.to_vec(),
                });
            }
            let raft_log = RaftLogOp::decode(log_data);
            peer_data.raft_logs.append(raft_log);
        }
        Ok(())
    }
}

pub(crate) fn wal_exists(dir: &Path, epoch_id: u32) -> bool {
    check_wal_header(dir, epoch_id).is_ok()
}
