use std::{
    collections::HashMap,
    fs::{self, File, OpenOptions},
    os::unix::fs::FileExt,
    path::{Path, PathBuf},
};

use bytes::{Buf, BufMut, BytesMut};

use crate::{Result, StoreProgress};

const ROTATION_THRESHOLD: u64 = 16 * 1024 * 1024; // 16MB
const CHANGESET_VERSION: u32 = 1;
const CHANGESET_META_SIZE: usize = 12; // 4 bytes for version + 4 bytes for length + 4 bytes for checksum 

fn manifest_path(dir: &Path) -> PathBuf {
    dir.join("MANIFEST")
}

#[derive(Debug)]
pub(crate) struct Manifest {
    file: File,
    file_path: PathBuf,
    file_offset: u64,
    pub(crate) store_progresses: HashMap<u64, StoreProgress>,
}

impl Manifest {
    pub(crate) fn open(dir: &Path) -> Result<Self> {
        let file_path = manifest_path(dir);
        let mut store_progresses = HashMap::new();
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .read(true)
            .open(&file_path)?;
        let file_data_vec = fs::read(&file_path)?;
        let mut file_data = file_data_vec.as_slice();
        while !file_data.is_empty() {
            if file_data.len() < CHANGESET_META_SIZE {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "invalid manifest",
                )
                .into());
            }
            let version = file_data.get_u32_le();
            if version > CHANGESET_VERSION {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "unsupported changeset version",
                )
                .into());
            }
            let length = file_data.get_u32_le() as usize;
            if file_data.len() < length + 4 {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "malformed change set",
                )
                .into());
            }

            let mut change_set_data = &file_data[..length];
            file_data.advance(length);
            let checksum = file_data.get_u32_le();
            if crc32c::crc32c(change_set_data) != checksum {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "checksum mismatch",
                )
                .into());
            }
            while !change_set_data.is_empty() {
                let store_progress = StoreProgress::decode(&mut change_set_data)?;
                store_progresses.insert(store_progress.store_id, store_progress);
            }
        }
        Ok(Self {
            file,
            file_path,
            file_offset: file_data_vec.len() as u64,
            store_progresses,
        })
    }

    fn rotate(&mut self) -> Result<()> {
        let dir = self.file_path.parent().unwrap();
        let tmp_path = self.file_path.with_extension("tmp");
        let tmp_file = File::create(&tmp_path)?;

        let offset = persist_store_progresses(
            &tmp_file,
            0,
            self.store_progresses.values(),
            self.store_progresses.len(),
        )?;

        fs::rename(&tmp_path, &self.file_path)?;
        file_system::sync_dir(dir)?;
        self.file_offset = offset;
        Ok(())
    }

    pub(crate) fn update_store_progresses<'a, I>(
        &mut self,
        store_progresses: I,
        len: usize,
    ) -> Result<()>
    where
        I: Iterator<Item = &'a StoreProgress>,
    {
        if self.file_offset > ROTATION_THRESHOLD {
            self.rotate()?;
        }
        let progresses: Vec<StoreProgress> = store_progresses.cloned().collect();
        let written =
            persist_store_progresses(&self.file, self.file_offset, progresses.iter(), len)?;
        self.file_offset += written;
        progresses.iter().for_each(|store_progress| {
            self.store_progresses
                .insert(store_progress.store_id, store_progress.clone());
        });
        Ok(())
    }
}

fn persist_store_progresses<'a, I>(
    file: &File,
    offset: u64,
    progresses: I,
    len: usize,
) -> Result<u64>
where
    I: Iterator<Item = &'a StoreProgress>,
{
    let encoding_body_len = len * StoreProgress::SIZE;
    let mut buf = BytesMut::with_capacity(encoding_body_len + 12);
    buf.put_u32_le(CHANGESET_VERSION);
    buf.put_u32_le(encoding_body_len as u32);
    progresses.for_each(|progress| {
        progress.encode(&mut buf);
    });
    let checksum = crc32c::crc32c(&buf[8..]);
    buf.put_u32_le(checksum);

    file.write_all_at(buf.chunk(), offset)?;
    file.sync_all()?;

    Ok(buf.len() as u64)
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use crate::{manifest::Manifest, Result, StoreProgress};

    #[test]
    fn test_manifest() -> Result<()> {
        let dir = PathBuf::from("/tmp/merged_engine/manifest_test");
        std::fs::create_dir_all(&dir)?;
        let mut manifest = Manifest::open(&dir)?;
        let store_progress = StoreProgress {
            store_id: 1,
            epoch: 2,
            offset: 3,
        };
        manifest.update_store_progresses(vec![&store_progress].into_iter(), 1)?;
        drop(manifest);
        let manifest = Manifest::open(&dir)?;
        assert_eq!(manifest.store_progresses.len(), 1);
        let persisted = manifest.store_progresses.get(&1).unwrap();
        assert_eq!(persisted.store_id, 1);
        assert_eq!(persisted.epoch, 2);
        assert_eq!(persisted.offset, 3);
        Ok(())
    }
}
