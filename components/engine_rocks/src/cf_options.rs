// Copyright 2019 TiKV Project Authors. Licensed under Apache-2.0.

use std::ops::{Deref, DerefMut};

use rocksdb::ColumnFamilyOptions as RawCfOptions;
#[derive(Default, Clone)]
pub struct RocksCfOptions(RawCfOptions);

impl RocksCfOptions {
    pub fn from_raw(raw: RawCfOptions) -> RocksCfOptions {
        RocksCfOptions(raw)
    }

    pub fn into_raw(self) -> RawCfOptions {
        self.0
    }
}

impl Deref for RocksCfOptions {
    type Target = RawCfOptions;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for RocksCfOptions {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
