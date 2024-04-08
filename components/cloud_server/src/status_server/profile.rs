// Copyright 2021 TiKV Project Authors. Licensed under Apache-2.0.
use std::{fs::File, io::Read};

use tempfile::NamedTempFile;
#[cfg(not(test))]
use tikv_alloc::dump_prof;

#[cfg(test)]
use self::test_utils::dump_prof;
#[cfg(test)]
pub use self::test_utils::TEST_PROFILE_MUTEX;

/// Trigger a heap profile and return the content.
pub fn dump_one_heap_profile() -> Result<NamedTempFile, String> {
    let f = NamedTempFile::new().map_err(|e| format!("create tmp file fail: {}", e))?;
    let path = f.path();
    dump_prof(path.to_string_lossy().as_ref()).map_err(|e| format!("dump_prof: {}", e))?;
    Ok(f)
}

pub fn read_file(path: &str) -> Result<Vec<u8>, String> {
    let mut f = File::open(path).map_err(|e| format!("open {} fail: {}", path, e))?;
    let mut buf = Vec::new();
    f.read_to_end(&mut buf)
        .map_err(|e| format!("read {} fail: {}", path, e))?;
    Ok(buf)
}

// Re-define some heap profiling functions because heap-profiling is not enabled
// for tests.
#[cfg(test)]
mod test_utils {
    use std::sync::Mutex;

    use lazy_static::lazy_static;
    use tikv_alloc::error::ProfResult;

    lazy_static! {
        pub static ref TEST_PROFILE_MUTEX: Mutex<()> = Mutex::new(());
    }

    pub fn dump_prof(_: &str) -> ProfResult<()> {
        Ok(())
    }
}
