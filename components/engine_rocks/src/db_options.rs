// Copyright 2019 TiKV Project Authors. Licensed under Apache-2.0.

use std::ops::{Deref, DerefMut};

use rocksdb::{DBOptions as RawDBOptions, TitanDBOptions as RawTitanDBOptions};

#[derive(Default)]
pub struct RocksDbOptions(RawDBOptions);

impl RocksDbOptions {
    pub fn from_raw(raw: RawDBOptions) -> RocksDbOptions {
        RocksDbOptions(raw)
    }

    pub fn into_raw(self) -> RawDBOptions {
        self.0
    }

    pub fn get_max_background_flushes(&self) -> i32 {
        self.0.get_max_background_flushes()
    }
}

impl Deref for RocksDbOptions {
    type Target = RawDBOptions;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for RocksDbOptions {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

pub struct RocksTitanDbOptions(RawTitanDBOptions);

impl RocksTitanDbOptions {
    pub fn from_raw(raw: RawTitanDBOptions) -> RocksTitanDbOptions {
        RocksTitanDbOptions(raw)
    }

    pub fn as_raw(&self) -> &RawTitanDBOptions {
        &self.0
    }
}

impl Deref for RocksTitanDbOptions {
    type Target = RawTitanDBOptions;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for RocksTitanDbOptions {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl Default for RocksTitanDbOptions {
    fn default() -> Self {
        RocksTitanDbOptions::from_raw(RawTitanDBOptions::new())
    }
}
