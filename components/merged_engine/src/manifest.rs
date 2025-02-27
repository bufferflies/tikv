use std::{
    collections::{HashMap, HashSet},
    fs::{self, create_dir_all, File},
    os::unix::fs::FileExt,
    path::{Path, PathBuf},
};

use bytes::{Buf, BufMut};

use crate::{Result, StoreProgress};

const CHANGESET_VERSION: u32 = 1;
const CHANGESET_META_SIZE: usize = 16; // 4(version) + 4 (num_keyspace_ids) + 4 (num_stores) + 4 (checksum) 

fn manifest_path(dir: &Path) -> PathBuf {
    dir.join("MANIFEST")
}

#[derive(Debug)]
pub(crate) struct Manifest {
    file_path: PathBuf,
    pub(crate) store_progresses: HashMap<u64, StoreProgress>,
    pub(crate) keyspace_ids: HashSet<u32>,
}

impl Manifest {
    pub(crate) fn open(dir: &Path) -> Result<Self> {
        if !dir.exists() {
            create_dir_all(dir)?;
        }
        let file_path = manifest_path(dir);
        let mut keyspace_ids = HashSet::new();
        let mut store_progresses = HashMap::new();
        let file_data_vec = fs::read(&file_path).unwrap_or_default();
        if file_data_vec.len() < CHANGESET_META_SIZE {
            return Ok(Self {
                file_path,
                store_progresses,
                keyspace_ids,
            });
        }
        let content_length = file_data_vec.len() - 4;
        let mut content = &file_data_vec[..content_length];
        let checksum = (&file_data_vec[content_length..]).get_u32_le();
        if crc32fast::hash(content) != checksum {
            return Err(
                std::io::Error::new(std::io::ErrorKind::InvalidData, "checksum mismatch").into(),
            );
        }
        let version = content.get_u32_le();
        if version > CHANGESET_VERSION {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "unsupported changeset version",
            )
            .into());
        }
        let num_keyspaces = content.get_u32_le() as usize;
        for _ in 0..num_keyspaces {
            let keyspace_id = content.get_u32_le();
            keyspace_ids.insert(keyspace_id);
        }
        let num_stores = content.get_u32_le() as usize;
        for _ in 0..num_stores {
            let store_progress = StoreProgress::decode(&mut content)?;
            store_progresses.insert(store_progress.store_id, store_progress);
        }
        Ok(Self {
            file_path,
            store_progresses,
            keyspace_ids,
        })
    }

    pub(crate) fn persist(&self) -> Result<()> {
        let dir = self.file_path.parent().unwrap();
        let tmp_path = self.file_path.with_extension("tmp");
        let tmp_file = File::create(&tmp_path)?;

        let mut buf = vec![];
        buf.put_u32_le(CHANGESET_VERSION);
        buf.put_u32_le(self.keyspace_ids.len() as u32);
        for &keyspace_id in &self.keyspace_ids {
            buf.put_u32_le(keyspace_id);
        }
        buf.put_u32_le(self.store_progresses.len() as u32);
        for progress in self.store_progresses.values() {
            progress.encode(&mut buf);
        }
        let checksum = crc32fast::hash(&buf);
        buf.put_u32_le(checksum);

        tmp_file.write_all_at(&buf, 0)?;
        tmp_file.sync_all()?;
        fs::rename(&tmp_path, &self.file_path)?;
        file_system::sync_dir(dir)?;
        Ok(())
    }

    pub(crate) fn update_store_progress(&mut self, store_id: u64, epoch: u32, offset: u64) {
        let store_progress =
            self.store_progresses
                .entry(store_id)
                .or_insert_with(|| StoreProgress {
                    store_id,
                    epoch,
                    offset,
                });
        store_progress.epoch = epoch;
        store_progress.offset = offset;
    }

    pub(crate) fn add_keyspace_id(&mut self, keyspace_id: u32) -> bool {
        self.keyspace_ids.insert(keyspace_id)
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use crate::{manifest::Manifest, Result};

    #[test]
    fn test_manifest() -> Result<()> {
        let dir = PathBuf::from("/tmp/merged_engine/manifest_test");
        std::fs::create_dir_all(&dir)?;
        let mut manifest = Manifest::open(&dir)?;
        manifest.update_store_progress(1, 2, 3);
        manifest.add_keyspace_id(4);
        manifest.persist()?;
        drop(manifest);
        let manifest = Manifest::open(&dir)?;
        assert_eq!(manifest.store_progresses.len(), 1);
        let persisted = manifest.store_progresses.get(&1).unwrap();
        assert_eq!(persisted.store_id, 1);
        assert_eq!(persisted.epoch, 2);
        assert_eq!(persisted.offset, 3);
        assert_eq!(manifest.keyspace_ids.len(), 1);
        Ok(())
    }
}
