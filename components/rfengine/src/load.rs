// Copyright 2021 TiKV Project Authors. Licensed under Apache-2.0.

use std::{fs, io::Read, path::Path};

use byteorder::{ByteOrder, LittleEndian};
use bytes::Buf;
use file_system::File;
use tikv_util::info;

use crate::{log_batch::RaftLogOp, manifest::Manifest, *};

impl RfEngineCore {
    pub(crate) fn load(&mut self, manifest: &Manifest) -> Result<()> {
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
        if wal_exists(self.dir.as_path(), epoch_id) {
            wal_offset = self.load_wal_file(epoch_id)?;
        }
        while wal_exists(self.dir.as_path(), epoch_id + 1) {
            self.task_sender.send(Task::Rotate { epoch_id }).unwrap();
            epoch_id += 1;
            wal_offset = self.load_wal_file(epoch_id)?;
        }
        let mut writer = self.writer.lock().unwrap();
        writer.open_file(epoch_id, wal_offset)
    }

    pub(crate) fn load_wal_file(&mut self, epoch_id: u32) -> Result<u64> {
        info!("load wal {}", epoch_id);
        let mut it = WALIterator::new(self.dir.clone(), epoch_id);
        it.iterate(|new_data| {
            let peer_ref = self.get_or_init_peer_data(new_data.peer_id, new_data.meta.region_id);
            let mut peer_data = peer_ref.write().unwrap();
            let _ = peer_data.apply(&new_data);
        })?;
        Ok(it.offset)
    }

    pub(crate) fn load_raft_log_file(
        &mut self,
        peer_id: u64,
        region_id: u64,
        first: u64,
        last: u64,
    ) -> Result<()> {
        let rlog_filename = raft_log_file_name(&self.dir, peer_id, first, last);
        let bin = fs::read(&rlog_filename)?;
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
            if checksum != crc32c::crc32c(log_data) {
                return Err(Error::Corruption("checksum mismatch".to_owned()));
            }
            let raft_log = RaftLogOp::decode(log_data);
            peer_data.raft_logs.append(raft_log);
        }
        Ok(())
    }
}

pub(crate) fn wal_exists(dir: &Path, epoch_id: u32) -> bool {
    let filename = wal_file_name(dir, epoch_id);
    if let Ok(mut file) = File::open(filename) {
        let mut buf = vec![0u8; WalHeader::len()];
        if file.read_exact(&mut buf).is_ok() {
            if let Ok(header) = WalHeader::decode(&buf) {
                if header.epoch_id == epoch_id {
                    return true;
                }
            }
        }
    }
    false
}
