// Copyright 2022 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    cmp::Ordering as CmpOrdering,
    collections::{HashMap, VecDeque},
    fs,
    fs::{File, OpenOptions},
    io,
    ops::{Deref, DerefMut},
    os::unix::fs::FileExt,
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
};

use bytes::{Buf, BufMut};
use protobuf::Message;
use tikv_util::{error, info, warn};

use crate::{
    metrics::RFENGINE_RLOG_GC_SIZE, raft_log_file_name, writer::EPOCH_ROTATE_LEN, PeerMeta,
    TRUNCATE_ALL_INDEX,
};

const REWRITE_DIFF: u32 = 10;

#[derive(Debug)]
pub(crate) struct Manifest {
    pub(crate) epoch_id: u32,
    pub(crate) file_path: PathBuf,
    pub(crate) file: File,
    pub(crate) peers: HashMap<u64, PeerMetaFiles>,
    pub(crate) offset: u64,
    pub(crate) engine_id: Arc<AtomicU64>,
    first_epoch: u32,
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct EntryRange {
    pub(crate) start: u64, // inclusive
    pub(crate) end: u64,   // inclusive
}

impl EntryRange {
    pub(crate) fn new(start: u64, end: u64) -> Self {
        Self { start, end }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct PeerMetaFiles {
    pub(crate) meta: PeerMeta,
    pub(crate) files: VecDeque<PeerFile>,
}

impl Deref for PeerMetaFiles {
    type Target = PeerMeta;

    fn deref(&self) -> &Self::Target {
        &self.meta
    }
}

impl DerefMut for PeerMetaFiles {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.meta
    }
}

impl PeerMetaFiles {
    fn new(region_id: u64) -> Self {
        Self {
            meta: PeerMeta::new(region_id),
            files: Default::default(),
        }
    }

    pub(crate) fn need_truncate(&self) -> bool {
        self.files
            .front()
            .map(|f| f.last_index <= self.truncated_idx)
            .unwrap_or_default()
    }
}

#[derive(Debug, Default, Copy, Clone, PartialEq, Eq)]
pub struct PeerFile {
    pub first_index: u64,
    pub last_index: u64,
    pub last_term: u32,
    pub epoch_id: u32,
}

impl PeerFile {
    pub fn new(epoch_id: u32, first_index: u64, last_index: u64, last_term: u32) -> Self {
        Self {
            epoch_id,
            first_index,
            last_index,
            last_term,
        }
    }
}

impl From<&rfenginepb::RaftLogFile> for PeerFile {
    fn from(file: &rfenginepb::RaftLogFile) -> Self {
        Self {
            first_index: file.first_index,
            last_index: file.last_index,
            last_term: file.last_term,
            epoch_id: file.epoch_id,
        }
    }
}

impl Manifest {
    pub(crate) fn open(dir: &Path, engine_id: Arc<AtomicU64>) -> crate::Result<Self> {
        let file_path = manifest_path(dir);
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .read(true)
            .open(&file_path)?;
        let mut manifest = Self {
            epoch_id: 0,
            engine_id,
            file_path,
            file,
            peers: HashMap::new(),
            offset: 0,
            first_epoch: 0,
        };
        manifest.init()?;
        Ok(manifest)
    }

    fn init(&mut self) -> crate::Result<()> {
        let file_data_vec = fs::read(&self.file_path)?;
        let mut file_data = file_data_vec.as_slice();
        while file_data.len() > 8 {
            let checksum = file_data.get_u32_le();
            let length = file_data.get_u32_le() as usize;
            if file_data.len() < length {
                warn!("manifest file unexpected EOF");
                break;
            }
            let change_set_data = &file_data[..length];
            if crc32c::crc32c(change_set_data) != checksum {
                warn!("manifest file checksum mismatch");
                break;
            }
            self.offset += length as u64 + 8;
            let mut change_set = rfenginepb::ChangeSet::new();
            change_set.merge_from_bytes(change_set_data).unwrap();
            self.apply_change_set(&change_set)?;
            file_data = &file_data[length..];
        }
        self.truncate_files();
        info!("init manifest epoch {}", self.epoch_id);
        Ok(())
    }

    pub(crate) fn handle_compaction(&mut self, cs: rfenginepb::ChangeSet) -> crate::Result<()> {
        self.apply_change_set(&cs)?;
        let engine_id = self.get_engine_id();
        if self.epoch_id - self.first_epoch > REWRITE_DIFF {
            info!("{}: rewrite manifest epoch: {}", engine_id, self.epoch_id);
            self.rewrite()?;
            self.first_epoch = self.epoch_id;
        } else {
            info!("{}: append manifest epoch: {}", engine_id, self.epoch_id);
            self.persist_change_set(cs)?;
        }
        self.truncate_files();
        Ok(())
    }

    fn persist_change_set(&mut self, cs: rfenginepb::ChangeSet) -> io::Result<()> {
        self.offset = persist_change_set(&self.file, self.offset, &cs)?;
        Ok(())
    }

    fn apply_change_set(&mut self, cs: &rfenginepb::ChangeSet) -> crate::Result<()> {
        assert!(cs.epoch_id > self.epoch_id);
        self.epoch_id = cs.epoch_id;
        if self.first_epoch == 0 {
            self.first_epoch = cs.epoch_id;
        }
        for peer_meta_pb in cs.get_peers() {
            let peer_meta = self
                .peers
                .entry(peer_meta_pb.peer_id)
                .or_insert_with(|| PeerMetaFiles::new(peer_meta_pb.region_id));
            if peer_meta.truncated_idx < peer_meta_pb.truncated_index {
                peer_meta.truncated_idx = peer_meta_pb.truncated_index;
            }
            for state_pb in peer_meta_pb.get_states() {
                if state_pb.value.is_empty() {
                    peer_meta.remove_state(state_pb.get_key());
                } else {
                    peer_meta.set_state(state_pb.get_key(), state_pb.get_value());
                }
            }
            for file in peer_meta_pb.get_files() {
                peer_meta.files.push_back(file.into());
            }
        }
        Ok(())
    }

    fn rewrite(&mut self) -> crate::Result<()> {
        let dir = self.file_path.parent().unwrap();
        let tmp_path = self.file_path.with_extension("tmp");
        let tmp_file = File::create(&tmp_path)?;
        let change_set = self.to_change_set(false);
        self.offset = persist_change_set(&tmp_file, 0, &change_set)?;
        fs::rename(&tmp_path, &self.file_path)?;
        file_system::sync_dir(dir)?;
        self.file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&self.file_path)?;
        Ok(())
    }

    pub(crate) fn to_change_set(&self, exclude_tombstone: bool) -> rfenginepb::ChangeSet {
        let mut cs = rfenginepb::ChangeSet::default();
        cs.epoch_id = self.epoch_id;
        for (&peer_id, peer_meta) in &self.peers {
            if exclude_tombstone && peer_meta.truncated_idx == TRUNCATE_ALL_INDEX {
                continue;
            }
            let mut meta_pb = rfenginepb::PeerMeta::new();
            meta_pb.set_peer_id(peer_id);
            meta_pb.set_region_id(peer_meta.region_id);
            meta_pb.set_truncated_index(peer_meta.truncated_idx);
            for (key, val) in &peer_meta.states {
                let mut state = rfenginepb::PeerState::new();
                state.set_key(key.to_vec());
                state.set_value(val.to_vec());
                meta_pb.mut_states().push(state);
            }
            for file in &peer_meta.files {
                let mut raft_log_file = rfenginepb::RaftLogFile::default();
                raft_log_file.set_first_index(file.first_index);
                raft_log_file.set_last_index(file.last_index);
                raft_log_file.set_last_term(file.last_term);
                raft_log_file.set_epoch_id(file.epoch_id);
                meta_pb.mut_files().push(raft_log_file);
            }
            cs.mut_peers().push(meta_pb);
        }
        cs
    }

    fn truncate_files(&mut self) {
        let dir = self.file_path.parent().unwrap();
        let engine_id = self.engine_id.load(Ordering::SeqCst);
        for (&peer_id, peer_meta) in &mut self.peers {
            let mut removed_count = 0;
            let mut last_index = 0;
            while peer_meta.need_truncate() {
                let file = peer_meta.files.pop_front().unwrap();
                let filename = raft_log_file_name(dir, peer_id, file.first_index, file.last_index);
                if filename.exists() {
                    let region_id = peer_meta.region_id;
                    let file_size = fs::metadata(filename.as_path()).unwrap().len();
                    if let Err(err) = fs::remove_file(filename.as_path()) {
                        error!(
                            "{}:{} failed to remove rlog file {:?}, {:?}",
                            engine_id, region_id, filename, err
                        );
                        RFENGINE_RLOG_GC_SIZE
                            .with_label_values(&["error"])
                            .observe(file_size as f64);
                    } else {
                        removed_count += 1;
                        last_index = file.last_index;
                        RFENGINE_RLOG_GC_SIZE
                            .with_label_values(&["success"])
                            .observe(file_size as f64);
                    }
                }
            }
            if removed_count > 0 {
                info!(
                    "{}:{} removed {} rlog files before index {}",
                    engine_id, peer_meta.region_id, removed_count, last_index
                );
            }
        }
    }

    pub(crate) fn peer_rlog_files(&self) -> HashMap<u64, VecDeque<PeerFile>> {
        self.peers
            .iter()
            .map(|(&peer_id, meta_files)| (peer_id, meta_files.files.clone()))
            .collect()
    }

    pub(crate) fn get_engine_id(&self) -> u64 {
        self.engine_id.load(Ordering::SeqCst)
    }

    /// Check if we should do snapshot in `handle_rotate`.
    ///
    /// We only do snapshot every EPOCH_ROTATE_LEN(4) epochs.
    pub(crate) fn should_snapshot(&self) -> bool {
        self.epoch_id % EPOCH_ROTATE_LEN == 0
    }

    /// The epoch of next snapshot.
    pub(crate) fn next_snapshot_epoch(epoch_id: u32) -> u32 {
        (epoch_id + EPOCH_ROTATE_LEN) / EPOCH_ROTATE_LEN * EPOCH_ROTATE_LEN
    }
}

pub(crate) fn manifest_path(dir: &Path) -> PathBuf {
    dir.join("MANIFEST")
}

pub(crate) fn persist_change_set(
    file: &File,
    mut offset: u64,
    cs: &rfenginepb::ChangeSet,
) -> io::Result<u64> {
    let buf = cs.write_to_bytes().unwrap();
    if buf.is_empty() {
        return Ok(0);
    }
    let mut header_buf = Vec::with_capacity(8);
    header_buf.put_u32_le(crc32c::crc32c(&buf));
    header_buf.put_u32_le(buf.len() as u32);
    file.write_at(&header_buf, offset)?;
    offset += 8;
    file.write_at(&buf, offset)?;
    offset += buf.len() as u64;
    file.sync_data()?;
    Ok(offset)
}

/// Generates a read plan for fetching Raft log entries within the range `[low,
/// high)`.
///
/// Given a list of rlog files and a desired entry range, this function returns
/// a sequence of `(EntryRange, PeerFile)` pairs that describe which portion of
/// the range should be read from which file.
pub(crate) fn generate_rlog_read_plan(
    rlog_files: &VecDeque<PeerFile>,
    low: u64,  // inclusive
    high: u64, // exclusive
) -> engine_traits::Result<Vec<(EntryRange, PeerFile)>> {
    let ranges = range_to_rlog_mapping(rlog_files);
    let mut plan = vec![];
    let mut current = low;
    for (range, file) in ranges {
        if range.end < current {
            continue; // the range is too small, move on
        }

        if range.start > current {
            return Err(engine_traits::Error::EntriesUnavailable); // unexpected gap
        }

        // At this point, we know: range.start <= current <= range.end
        let end = range.end.min(high - 1);
        plan.push((EntryRange::new(current, end), file));
        current = end + 1;
        if current >= high {
            return Ok(plan);
        }
    }
    Err(engine_traits::Error::EntriesUnavailable)
}

/// Builds a mapping of non-overlapping, continuous entry ranges to their
/// corresponding `PeerFile`.
///
/// The semantics of the function are identical with `RaftLogs::append`
/// - If a log gap is detected, all previous entries are dropped.
/// - If an overlap occurs, newer entries overwrite the older ones.
pub(crate) fn range_to_rlog_mapping(files: &VecDeque<PeerFile>) -> Vec<(EntryRange, PeerFile)> {
    let mut stack: Vec<(EntryRange, PeerFile)> = Vec::new();
    for file in files.iter() {
        if let Some((r, _)) = stack.last() {
            let next_index = r.end + 1;
            let incoming_index = file.first_index;
            match incoming_index.cmp(&next_index) {
                CmpOrdering::Greater => {
                    // There is gap between existing logs and next log, clear all.
                    stack.clear();
                }
                CmpOrdering::Less => {
                    while let Some((EntryRange { start, .. }, file)) = stack.pop() {
                        if start < incoming_index {
                            stack.push((EntryRange::new(start, incoming_index - 1), file));
                            break;
                        }
                    }
                }
                CmpOrdering::Equal => {}
            }
        }
        stack.push((EntryRange::new(file.first_index, file.last_index), *file));
    }
    stack
}

/// Returns the `PeerFile` that contains the Raft entry at index `idx`.
///
/// Traverses from newest to oldest, respecting semantics that newer rlog files
/// overwrite older ones.
pub(crate) fn rlog_by_entry_index(files: &VecDeque<PeerFile>, idx: u64) -> Option<PeerFile> {
    let mut next_start_index = None;
    for file in files.iter().rev() {
        if next_start_index.map_or(false, |n| file.last_index + 1 < n) {
            // If a gap exists between this file and the previously seen (newer)
            // file, entries from this and all older files are considered
            // invalid. This matches the semantics of `range_to_rlog_mapping`.
            return None;
        }

        if file.last_index < idx {
            return None;
        }
        if file.first_index <= idx {
            return Some(*file);
        }

        next_start_index = Some(file.first_index);
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_next_snapshot_epoch() {
        assert_eq!(Manifest::next_snapshot_epoch(0), 4);
        assert_eq!(Manifest::next_snapshot_epoch(1), 4);
        assert_eq!(Manifest::next_snapshot_epoch(3), 4);
        assert_eq!(Manifest::next_snapshot_epoch(4), 8);
    }

    #[test]
    fn test_get_rlog_by_entry_index() {
        let files = VecDeque::from(vec![
            PeerFile::new(1, 1, 5, 0),    // discarded due to gap
            PeerFile::new(2, 8, 9, 0),    // actual mapped range: 8–9
            PeerFile::new(3, 10, 100, 0), // actual mapped range: 10–29
            PeerFile::new(4, 30, 90, 0),  // actual mapped range: 30–49
            PeerFile::new(5, 50, 70, 0),  // actual mapped range: 50–70
        ]);

        let mapping = range_to_rlog_mapping(&files);
        assert_eq!(mapping.len(), 4);
        assert_eq!(mapping[0].0, EntryRange::new(8, 9));
        assert_eq!(mapping[0].1.epoch_id, 2);
        assert_eq!(mapping[1].0, EntryRange::new(10, 29));
        assert_eq!(mapping[1].1.epoch_id, 3);
        assert_eq!(mapping[2].0, EntryRange::new(30, 49));
        assert_eq!(mapping[2].1.epoch_id, 4);
        assert_eq!(mapping[3].0, EntryRange::new(50, 70));
        assert_eq!(mapping[3].1.epoch_id, 5);

        for (range, file) in mapping.iter() {
            for i in range.start..=range.end {
                let entry = rlog_by_entry_index(&files, i);
                assert!(entry.is_some(), "index {}", i);
                assert_eq!(entry.unwrap(), *file, "index {}", i);
            }
        }
        assert!(rlog_by_entry_index(&files, 1).is_none());
        assert!(rlog_by_entry_index(&files, 7).is_none());
        assert!(rlog_by_entry_index(&files, 71).is_none());
        assert!(rlog_by_entry_index(&files, 100).is_none());
    }

    #[test]
    fn test_generate_rlog_read_plan() {
        let files = VecDeque::from(vec![
            PeerFile::new(1, 1, 5, 0),    // discarded due to gap
            PeerFile::new(2, 10, 100, 0), // 10–29
            PeerFile::new(3, 30, 90, 0),  // 30–49
            PeerFile::new(4, 50, 70, 0),  // 50–70
        ]);

        for (desc, low, high, expected) in [
            ("first range", 12, 15, Some(vec![(12, 14, 2)])),
            ("last range", 69, 71, Some(vec![(69, 70, 4)])),
            ("two files", 28, 32, Some(vec![(28, 29, 2), (30, 31, 3)])),
            ("gap before start", 2, 5, None),
            ("gap at beginning", 1, 11, None),
            ("end out of range", 69, 75, None),
            (
                "full span",
                10,
                71,
                Some(vec![(10, 29, 2), (30, 49, 3), (50, 70, 4)]),
            ),
        ] {
            let result = generate_rlog_read_plan(&files, low, high);
            match expected {
                Some(ranges) => {
                    let actual: Vec<_> = result
                        .unwrap()
                        .iter()
                        .map(|(r, pf)| (r.start, r.end, pf.epoch_id))
                        .collect();
                    assert_eq!(actual, ranges, "{}", desc);
                }
                None => {
                    assert!(result.is_err(), "{}", desc);
                }
            }
        }
    }
}
