// Copyright 2022 TiKV Project Authors. Licensed under Apache-2.0.

use std::{mem::ManuallyDrop, pin::Pin, sync::Arc};

use engine_rocks::{RocksSstIterator, RocksSstReader};
use engine_traits::{IterOptions, Iterator as TraitIterator, RefIterable, CF_DEFAULT, CF_WRITE};
use kvengine::{
    table::{table, InnerKey, Value},
    ShardMeta, UserMeta,
};
use kvproto::{import_sstpb::SstMeta, raft_cmdpb::RaftCmdRequest};
use sst_importer::SstImporter;
use tikv_util::{box_err, codec, error, info};
use txn_types::{Key, WriteRef, WriteType};

struct RocksEntriesIterator {
    default: RocksMultiFileIterator,
    write: RocksMultiFileIterator,

    cur_entry: Option<Entry>,
    result: crate::Result<()>,
}

impl table::Iterator for RocksEntriesIterator {
    fn next(&mut self) {
        let res = self.move_to_next_entry().and_then(|has_next| {
            if has_next {
                self.fill_cur_entry()
            } else {
                self.cur_entry = None;
                Ok(())
            }
        });
        self.handle_result(res)
    }

    fn next_version(&mut self) -> bool {
        unreachable!()
    }

    fn rewind(&mut self) {
        self.result = Ok(());

        let res = self.rewind_impl().and_then(|_| self.fill_cur_entry());
        self.handle_result(res);
    }

    fn seek(&mut self, _key: InnerKey<'_>) {
        unreachable!()
    }

    fn key(&self) -> InnerKey<'_> {
        InnerKey::from_outer_key(&self.cur_entry.as_ref().expect("key() on invalid iter").key)
    }

    fn value(&self) -> Value {
        Value::decode(
            &self
                .cur_entry
                .as_ref()
                .expect("value() on invalid iter")
                .val,
        )
    }

    fn valid(&self) -> bool {
        self.cur_entry.is_some()
    }
}

impl RocksEntriesIterator {
    fn new(importer: Arc<SstImporter>, req: &RaftCmdRequest) -> Self {
        let mut write_metas = vec![];
        let mut default_metas = vec![];
        for import_req in req.get_requests() {
            let sst_meta = import_req.get_ingest_sst().get_sst();
            match sst_meta.get_cf_name() {
                CF_WRITE => {
                    write_metas.push(sst_meta.clone());
                }
                CF_DEFAULT | "" => default_metas.push(sst_meta.clone()),
                otherwise => panic!("cf should't be {:?}", otherwise),
            }
        }
        Self {
            default: RocksMultiFileIterator::new(importer.clone(), default_metas),
            write: RocksMultiFileIterator::new(importer, write_metas),
            cur_entry: None,
            result: Ok(()),
        }
    }

    fn handle_result(&mut self, res: crate::Result<()>) {
        if let Err(err) = res {
            error!(
                ?err;
                "RocksEntriesIterator: encountered an error, the backup data may be invalid."
            );
            self.result = Err(err);
            self.write.make_invalid();
            self.default.make_invalid();
        }
    }

    fn ingest_id(&self) -> &[u8] {
        self.write.sst_metas.first().unwrap().get_uuid()
    }

    fn rewind_impl(&mut self) -> crate::Result<()> {
        self.write.open_reader_at(0)?;
        if !self.default.sst_metas.is_empty() {
            self.default.open_reader_at(0)?;
        }
        Ok(())
    }

    fn move_to_next_entry(&mut self) -> crate::Result<bool> {
        self.write.next_impl()?;
        if !self.write.valid_impl()? {
            self.cur_entry = None;
            return Ok(false);
        }

        Ok(true)
    }

    fn fill_cur_entry(&mut self) -> crate::Result<()> {
        let wv = WriteRef::parse(self.write.value())?;
        if wv.short_value.is_some() {
            self.cur_entry = Some(entry(self.must_write_iter(), None)?);
            return Ok(());
        }

        while self.default.key_without_ts() != self.write.key_without_ts() {
            if self.default.key_without_ts() > self.write.key_without_ts() {
                return Err(crate::Error::Other(box_err!(
                    "default not found for key {}",
                    log_wrappers::hex_encode(self.write.key_without_ts())
                )));
            }
            self.default.next_impl()?;
            if !self.default.valid_impl()? {
                return Err(crate::Error::Other(box_err!(
                    "default not found for key {}",
                    log_wrappers::hex_encode(self.write.key_without_ts())
                )));
            }
        }

        self.cur_entry = Some(entry(self.must_write_iter(), Some(self.default.value()))?);
        Ok(())
    }

    fn must_write_iter(&self) -> &RocksSstIterator<'_> {
        &self
            .write
            .state
            .as_ref()
            .expect("mst_write_inner_iter when write is invalid")
            .iter
    }
}

fn entry(write_iter: &RocksSstIterator<'_>, default_value: Option<&[u8]>) -> crate::Result<Entry> {
    let (key, commit_ts) = parse_rocksdb_key(write_iter.key())?;
    let write_ref = WriteRef::parse(write_iter.value())?;
    let start_ts = write_ref.start_ts.into_inner();
    let user_meta = UserMeta::new(start_ts, commit_ts);
    let val = match write_ref.write_type {
        WriteType::Put => match write_ref.short_value {
            Some(short_val) => encode_table_value(user_meta, short_val),
            None => encode_table_value(user_meta, default_value.unwrap()),
        },
        WriteType::Delete => encode_table_value(user_meta, &[]),
        _ => panic!("unexpected write type"),
    };
    Ok(Entry { key, val })
}

struct RocksMultiFileIterator {
    importer: Arc<SstImporter>,
    sst_metas: Vec<SstMeta>,
    state: Option<RocksMultiFileIterState>,
}

// An iterator that yields kv pairs from multiple rocksdb files.
struct RocksMultiFileIterState {
    idx: usize,
    _reader: Pin<Box<RocksSstReader>>,
    // references `reader`.
    // shouldn't outlives `self`.
    iter: ManuallyDrop<RocksSstIterator<'static>>,
}

impl Drop for RocksMultiFileIterState {
    fn drop(&mut self) {
        // SAFETY: `iter` won't be used anymore.
        // `iter` must be dropped before `_reader` because the former references the
        // latter.
        unsafe { ManuallyDrop::drop(&mut self.iter) }
    }
}

impl RocksMultiFileIterator {
    fn new(importer: Arc<SstImporter>, mut sst_metas: Vec<SstMeta>) -> Self {
        // NOTE: maybe make sure there isn't range overlapping?
        sst_metas.sort_by(|a, b| a.get_range().get_start().cmp(b.get_range().get_start()));
        Self {
            importer,
            sst_metas,
            state: None,
        }
    }

    fn make_invalid(&mut self) {
        self.state = None;
    }

    fn key_without_ts(&self) -> &[u8] {
        let key = self
            .state
            .as_ref()
            .expect("key called with invalid iterator")
            .iter
            .key();
        Key::truncate_ts_for(key).expect("key without ts")
    }

    fn value(&self) -> &[u8] {
        self.state
            .as_ref()
            .expect("value() called with invalid iterator")
            .iter
            .value()
    }

    fn valid_impl(&self) -> crate::Result<bool> {
        match self.state.as_ref() {
            Some(x) => Ok(x.iter.valid()?),
            None => Ok(false),
        }
    }

    fn next_impl(&mut self) -> crate::Result<()> {
        if self.state.is_none() {
            return Ok(());
        }

        if !self.state.as_ref().unwrap().iter.valid()? && !self.move_to_next_file()? {
            return Ok(());
        }

        self.state.as_mut().unwrap().iter.next()?;
        Ok(())
    }

    /// Move cursor to next SST Meta.
    ///
    /// # Returns
    ///
    /// `true` if there is one more SST Meta.
    /// `false` if no more SST Meta.
    fn move_to_next_file(&mut self) -> crate::Result<bool> {
        if self.sst_metas.is_empty() {
            return Ok(false);
        }

        match self.state.take() {
            None => Ok(false),
            Some(state) => {
                let next_idx = state.idx + 1;
                if next_idx < self.sst_metas.len() {
                    self.open_reader_at(next_idx)?;
                    Ok(true)
                } else {
                    Ok(false)
                }
            }
        }
    }

    fn open_reader_at(&mut self, idx: usize) -> crate::Result<()> {
        let reader = Box::pin(self.importer.get_reader(&self.sst_metas[idx])?);
        // SAFETY: `reader` lives as long as `iter` lives.
        // and there isn't public API that allows the caller to extend `iter`'s
        // lifetime.
        let mut iter =
            unsafe { (*(&*reader as *const RocksSstReader)).iter(IterOptions::default()) }?;
        iter.seek_to_first()?;
        self.state = Some(RocksMultiFileIterState {
            idx,
            _reader: reader,
            iter: ManuallyDrop::new(iter),
        });
        Ok(())
    }
}

pub(crate) fn convert_sst(
    kv: kvengine::Engine,
    importer: Arc<SstImporter>,
    req: &RaftCmdRequest,
    shard_meta: ShardMeta,
) -> crate::Result<kvenginepb::ChangeSet> {
    info!("{} convert sst {:?}", shard_meta.tag(), req);
    let region_id = req.get_header().get_region_id();
    let region_ver = req.get_header().get_region_epoch().get_version();
    let mut it = RocksEntriesIterator::new(importer, req);
    let ingest_id = it.ingest_id().to_vec();
    let cs = kv.build_ingest_files(region_id, region_ver, &mut it, ingest_id, shard_meta)?;
    it.result?;
    Ok(cs)
}

struct Entry {
    key: Vec<u8>,
    val: Vec<u8>,
}

fn parse_rocksdb_key(data_key: &[u8]) -> codec::Result<(Vec<u8>, u64)> {
    let origin_key = keys::origin_key(data_key);
    let key = txn_types::Key::from_encoded(origin_key.to_vec());
    let ts = key.decode_ts()?.into_inner();
    let raw_key = key.to_raw()?;
    Ok((raw_key, ts))
}

fn encode_table_value(user_meta: UserMeta, val: &[u8]) -> Vec<u8> {
    Value::encode_buf(0, &user_meta.to_array(), user_meta.commit_ts, val)
}
