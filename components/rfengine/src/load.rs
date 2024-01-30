// Copyright 2021 TiKV Project Authors. Licensed under Apache-2.0.

use std::{fs, os::unix::fs::FileExt, path::Path};

use byteorder::{ByteOrder, LittleEndian};
use bytes::{Buf, Bytes};
use tikv_util::{info, warn};

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
            match async_it.iterate_batch(|_, _| {
                async_batch_cnt += 1;
            }) {
                Ok(_) => {
                    async_offset = async_it.offset;
                }
                Err(Error::Corruption { msg, offset, .. }) => {
                    warn!(
                        "async wal corrupted at epoch: {}, offset: {}, msg: {}, try to truncate it to offset {}",
                        epoch_id, offset, msg, offset
                    );
                    // If the async wal file is corrupted, just ignored the data after the offset,
                    // and the remaining data will be recovered from sync wal later.
                    async_offset = offset;
                }
                Err(e) => return Err(e),
            }
        }
        let mut sync_batch_idx = 0;
        let mut it = WalIterator::new(self.wal_dir().to_path_buf(), epoch_id);
        match it.iterate_batch(|data, _| {
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
        }) {
            Ok(_) => {}
            Err(Error::Corruption {
                msg, offset, data, ..
            }) => {
                if !is_last_wal(self.wal_dir(), epoch_id) {
                    return Err(Error::Corruption {
                        msg,
                        offset,
                        epoch_id,
                        data,
                    });
                }

                warn!(
                    "sync wal corrupted at epoch: {}, offset: {}, msg: {}, try to truncate it to offset {}",
                    epoch_id, offset, msg, offset,
                );
                // If the write batch data corrupted and it's in the last wal file,
                // we reset the aligned wal header to EOF.
                let file = fs::File::options()
                    .write(true)
                    .open(wal_file_name(self.wal_dir(), epoch_id))?;
                file.write_all_at(&[0u8; BATCH_HEADER_SIZE], offset)?;
                file.sync_all()?;
            }
            Err(e) => return Err(e),
        }

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
                // In scene of restore keyspace, we don't persist WAL to avoid unnecessary I/O,
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

pub(crate) fn is_last_wal(dir: &Path, epoch_id: u32) -> bool {
    wal_exists(dir, epoch_id) && !wal_exists(dir, epoch_id + 1)
}

#[cfg(test)]
mod tests {

    use std::{fs::OpenOptions, sync::atomic::Ordering};

    use bytes::{BufMut, BytesMut};
    use raft_proto::eraftpb;
    use slog::o;

    use super::{config::Config, *};

    fn init_logger() {
        use slog::Drain;
        let decorator = slog_term::PlainDecorator::new(std::io::stdout());
        let drain = slog_term::CompactFormat::new(decorator).build();
        let drain = std::sync::Mutex::new(drain).fuse();
        let logger = slog::Logger::root(drain, o!());
        slog_global::set_global(logger);
    }

    fn make_log_data(index: u64, size: usize) -> eraftpb::Entry {
        let mut entry = eraftpb::Entry::new();
        entry.set_entry_type(eraftpb::EntryType::EntryConfChange);
        entry.set_index(index);
        entry.set_term(1);

        let mut data = BytesMut::with_capacity(size);
        data.resize(size, 0);
        entry.set_data(data.freeze());
        entry
    }

    fn make_state_kv(key_byte: u8, idx: u64) -> (BytesMut, BytesMut) {
        let mut key = BytesMut::new();
        key.put_u8(key_byte);
        let mut val = BytesMut::new();
        val.put_u64_le(idx);
        (key, val)
    }

    fn prepare_rfengine(engine: &RfEngine) {
        let mut wb = WriteBatch::new();
        for peer_id in 1..=10_u64 {
            let (key, val) = make_state_kv(2, 1);
            let region_id = peer_id + 1;
            wb.set_state(peer_id, region_id, key.chunk(), val.chunk());
        }
        engine.write(wb).unwrap();
        for idx in 1..=1050_u64 {
            let mut wb = WriteBatch::new();
            for peer_id in 1..=10_u64 {
                let region_id = peer_id + 1;
                wb.append_raft_log(peer_id, region_id, &make_log_data(idx, 128));
                let (key, val) = make_state_kv(1, idx);
                wb.set_state(peer_id, region_id, key.chunk(), val.chunk());
            }
            engine.write(wb).unwrap();
        }
        assert_eq!(engine.peers.len(), 10);
    }

    #[test]
    fn test_load_async_corruption() {
        init_logger();
        let tmp_dir = tempfile::tempdir().unwrap();
        let wal_size = 128 * 1024_usize;
        let dir_path = tmp_dir.path();
        let sync_wal_path = dir_path.join("wal-sync");
        let mut cfg = Config::new(wal_size);
        cfg.wal_sync_dir = sync_wal_path.to_str().unwrap().to_owned();

        let engine = RfEngine::open(dir_path, &cfg, None, None).unwrap();
        prepare_rfengine(&engine);
        engine.stop_worker(true);

        let writer = engine.writer.lock().unwrap();
        let current_epoch = writer.epoch_id;

        let mut it = WalIterator::new(dir_path.to_owned(), current_epoch);
        it.iterate_batch(|_, _| {
            // Do nothing.
        })
        .unwrap();

        // Write some corrupted data to the last page of the async wal file.
        let filename = wal_file_name(dir_path, current_epoch);
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(filename)
            .unwrap();
        let write_offset = it.offset - 4096 + BATCH_HEADER_SIZE as u64;

        info!("write corrupted data to async wal offset: {}", write_offset);
        file.write_all_at(&[0u8; 10], write_offset).unwrap();
        file.sync_all().unwrap();
        drop(file);

        // RfEngine should be able to recover from the corrupted async wal file.
        let engine = RfEngine::open(dir_path, &cfg, None, None).unwrap();
        assert_eq!(engine.peers.len(), 10);
    }

    #[test]
    fn test_load_sync_corruption() {
        init_logger();
        let tmp_dir = tempfile::tempdir().unwrap();
        let wal_size = 128 * 1024_usize;
        let dir_path = tmp_dir.path();
        let sync_wal_path = dir_path.join("wal-sync");
        let mut cfg = Config::new(wal_size);
        cfg.wal_sync_dir = sync_wal_path.to_str().unwrap().to_owned();

        let engine = RfEngine::open(dir_path, &cfg, None, None).unwrap();
        prepare_rfengine(&engine);
        engine.stop_worker(true);

        let writer = engine.writer.lock().unwrap();
        let current_epoch = writer.epoch_id;

        let mut async_it = WalIterator::new(dir_path.to_owned(), current_epoch);
        async_it
            .iterate_batch(|_, _| {
                // Do nothing.
            })
            .unwrap();

        let offset = async_it.offset;
        // Truncate the last write batch of async wal file.
        let filename = wal_file_name(dir_path, current_epoch);
        let file = OpenOptions::new().write(true).open(filename).unwrap();
        info!("truncate async wal to offset: {}", offset - 4096);
        file.write_all_at(&[0u8; BATCH_HEADER_SIZE], offset - 4096)
            .unwrap();
        file.sync_all().unwrap();
        drop(file);

        let filename = wal_file_name(&sync_wal_path, current_epoch);
        let mut sync_it = WalIterator::new(sync_wal_path.to_owned(), current_epoch);
        sync_it
            .iterate_batch(|_, _| {
                // Do nothing.
            })
            .unwrap();

        // Write corrupted data to the last write batch of sync wal file.
        let file = OpenOptions::new().write(true).open(filename).unwrap();
        let write_offset = sync_it.offset - 4096 + BATCH_HEADER_SIZE as u64;
        info!(
            "write corrupted data to sync wal offset: {} at epoch: {}",
            write_offset, current_epoch
        );
        file.write_all_at(&[0u8; 10], write_offset).unwrap();
        file.sync_all().unwrap();
        drop(file);

        // Backup MANIFEST to rollback later.
        let manifest_filename = dir_path.join("MANIFEST");
        let manifest_filename_bak = dir_path.join("MANIFEST.bak");
        fs::copy(&manifest_filename, &manifest_filename_bak).unwrap();

        // RfEngine should be able to recover from the last corrupted sync wal file.
        let engine = RfEngine::open(dir_path, &cfg, None, None).unwrap();
        assert_eq!(engine.peers.len(), 10);

        // Write some more data to rotate the wal file.
        for idx in 1050..=1080 {
            let mut wb = WriteBatch::new();
            for peer_id in 1..=10_u64 {
                let region_id = peer_id + 1;
                wb.append_raft_log(peer_id, region_id, &make_log_data(idx, 128));
                let (key, val) = make_state_kv(1, idx);
                wb.set_state(peer_id, region_id, key.chunk(), val.chunk());
            }
            engine.write(wb).unwrap();
        }
        engine.stop_worker(false);
        drop(engine);

        // Rollback MANIFEST to the previous backup one.
        fs::copy(&manifest_filename_bak, &manifest_filename).unwrap();
        let engine = RfEngine::open(dir_path, &cfg, None, None).unwrap();
        let (compacted_epoch, current_epoch) = {
            let writer = engine.writer.lock().unwrap();
            (
                writer.compacted_epoch.load(Ordering::SeqCst),
                writer.epoch_id,
            )
        };
        engine.stop_worker(false);
        drop(engine);

        info!(
            "compacted_epoch: {}, current_epoch: {}",
            compacted_epoch, current_epoch
        );
        assert!(compacted_epoch + 1 < current_epoch);

        // Write corrupted data to `current_epoch - 1` wal file.
        let filename = wal_file_name(&sync_wal_path, current_epoch - 1);
        let mut sync_it = WalIterator::new(sync_wal_path, current_epoch - 1);
        sync_it.iterate_batch(|_, _| {}).unwrap();
        let file = OpenOptions::new().write(true).open(filename).unwrap();
        let write_offset = sync_it.offset - 4096 + BATCH_HEADER_SIZE as u64;
        info!(
            "write corrupted data to sync wal offset: {} at epoch: {}",
            write_offset,
            current_epoch - 1
        );
        file.write_all_at(&[0u8; 10], write_offset).unwrap();
        file.sync_all().unwrap();
        drop(file);

        // Rollback MANIFEST to the previous backup one.
        fs::copy(&manifest_filename_bak, &manifest_filename).unwrap();
        // RfEngine can not recover from the corrupted wal file in previous epoch.
        assert!(RfEngine::open(dir_path, &cfg, None, None).is_err());
    }
}
