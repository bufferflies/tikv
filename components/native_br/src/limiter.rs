// Copyright 2025 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::Duration,
};

use kvengine::{get_shard_property, STORAGE_CLASS_KEY};
use kvenginepb as pb;
use kvproto::{metapb, metapb::PeerRole};
use pd_client::PdClient;
use schema::schema::StorageClass;
use tikv_util::{
    box_try,
    config::{ReadableDuration, ReadableSize},
    info,
    time::Limiter,
};

use crate::{
    error::Result,
    metrics::{NATIVE_BR_RATE_LIMITER_BYTES_TOTAL, NATIVE_BR_RATE_LIMITER_WAIT_DURATION_SECS},
    tikv::{FileWithId, StoresFiles},
};

/// The typical table file of 16MB has average meta size about 360KB.
const AVG_TABLE_META_SIZE: u64 = 360 * 1024; // 360KB

/// Used to estimate table file size based on meta offset.
///
/// 16MB / (16MB - 360KB) = 1.022
pub(crate) const TABLE_SIZE_MULTIPLIER: f64 = 1.022;

const BYTES_PER_MIB: f64 = (1024 * 1024) as f64;

/// A generic bytes throughput limiter based on a token bucket.
pub struct ByteLimiter {
    limiter: Limiter,
    source: &'static str,
    request_id: u64,
    total_consumed: AtomicU64,
    period_bytes: AtomicU64,
    period_start_ms: AtomicU64,
}

impl ByteLimiter {
    /// Creates a ByteLimiter with its own independent token bucket.
    pub fn new(max_throughput_bytes_per_sec: u64, source: &'static str, request_id: u64) -> Self {
        let limiter = <Limiter>::builder(max_throughput_bytes_per_sec as f64)
            .refill(Duration::from_millis(10))
            .build();
        Self::from_limiter(limiter, source, request_id)
    }

    /// Creates a ByteLimiter wrapping an existing Limiter (e.g. cloned from a
    /// global shared instance) with its own per-operation metrics/logging.
    pub fn from_limiter(limiter: Limiter, source: &'static str, request_id: u64) -> Self {
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
        Self {
            limiter,
            source,
            request_id,
            total_consumed: AtomicU64::new(0),
            period_bytes: AtomicU64::new(0),
            period_start_ms: AtomicU64::new(now_ms),
        }
    }

    pub async fn consume(&self, bytes: u64) {
        let begin = std::time::Instant::now();
        self.limiter.consume(bytes as usize).await;
        let wait_secs = begin.elapsed().as_secs_f64();
        NATIVE_BR_RATE_LIMITER_WAIT_DURATION_SECS
            .with_label_values(&[self.source])
            .observe(wait_secs);
        NATIVE_BR_RATE_LIMITER_BYTES_TOTAL
            .with_label_values(&[self.source])
            .inc_by(bytes);

        let total = self.total_consumed.fetch_add(bytes, Ordering::Relaxed) + bytes;
        self.period_bytes.fetch_add(bytes, Ordering::Relaxed);
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
        let start_ms = self.period_start_ms.load(Ordering::Relaxed);
        let elapsed_ms = now_ms.saturating_sub(start_ms);
        if elapsed_ms >= 10_000
            && self
                .period_start_ms
                .compare_exchange(start_ms, now_ms, Ordering::AcqRel, Ordering::Relaxed)
                .is_ok()
        {
            let period_bytes = self.period_bytes.swap(0, Ordering::Relaxed);
            let throughput = period_bytes as f64 / elapsed_ms as f64 * 1000.0;
            info!(
                "rate limiter [{}][id={}]: {:.2} MB/s (total: {:.1} MB, limit: {:.1} MB/s)",
                self.source,
                self.request_id,
                throughput / BYTES_PER_MIB,
                total as f64 / BYTES_PER_MIB,
                self.limiter.speed_limit() / BYTES_PER_MIB,
            );
        }
    }

    pub fn unconsume(&self, bytes: u64) {
        self.limiter.unconsume(bytes as usize);
    }
}

#[derive(Clone, Serialize, Deserialize, PartialEq, Debug)]
#[serde(default)]
#[serde(rename_all = "kebab-case")]
pub struct RateLimitConfig {
    pub enable: bool,
    pub max_throughput: ReadableSize,
    /// Calibrate estimated restore size by requesting existed files on TiKV
    /// stores when exceeds this threshold.
    ///
    /// Current defaults to MAX to disable calibration. Set to 10GiB after TiKV
    /// supports it.
    pub calibrate_restore_size_threshold: ReadableSize,
    pub store_req_timeout: ReadableDuration,
    pub store_cache_ttl: ReadableDuration,
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            enable: false,
            max_throughput: ReadableSize::mb(500), // 500 MB/s
            calibrate_restore_size_threshold: ReadableSize(u64::MAX),
            store_req_timeout: ReadableDuration::secs(15),
            store_cache_ttl: ReadableDuration::minutes(5),
        }
    }
}

pub struct ThroughputLimiter {
    bytes: ByteLimiter,
    calibrate_threshold: u64,
    stores_files: StoresFiles,
}

#[derive(Clone)]
pub struct SnapshotSize {
    /// Estimated size of the snapshot (SINGLE replica).
    pub estimated_size: u64,
    /// `files` is empty if not need to calibrate from TiKV.
    pub files: Vec<FileWithSize>,
}

impl ThroughputLimiter {
    pub fn new(
        config: &RateLimitConfig,
        pd_client: Arc<dyn PdClient>,
        runtime: tokio::runtime::Handle,
        request_id: u64,
    ) -> Result<Self> {
        info!(
            "create throughput limiter: {} MiB/sec, request_id: {}",
            config.max_throughput.as_mb_f64(),
            request_id,
        );
        let bytes = ByteLimiter::new(config.max_throughput.0, "restore", request_id);
        Self::with_byte_limiter(bytes, config, pd_client, runtime)
    }

    /// Creates a ThroughputLimiter using a pre-built ByteLimiter (which may
    /// share a global token bucket).
    pub fn with_byte_limiter(
        bytes: ByteLimiter,
        config: &RateLimitConfig,
        pd_client: Arc<dyn PdClient>,
        runtime: tokio::runtime::Handle,
    ) -> Result<Self> {
        let stores_files = box_try!(StoresFiles::new(
            config.store_cache_ttl.0,
            config.store_req_timeout.0,
            pd_client,
            runtime,
        ));
        Ok(Self {
            bytes,
            calibrate_threshold: config.calibrate_restore_size_threshold.0,
            stores_files,
        })
    }

    pub fn calibrate_threshold(&self) -> u64 {
        self.calibrate_threshold
    }

    /// The returned value is the estimated data size of all replicas.
    pub fn estimate_snapshot_size_locally(cs: &pb::ChangeSet) -> SnapshotSize {
        if !cs.has_restore_shard() {
            debug_assert!(false);
            return SnapshotSize {
                estimated_size: 0,
                files: vec![],
            };
        }

        let snap = cs.get_restore_shard();

        let storage_class = StorageClass::unmarshal(
            get_shard_property(STORAGE_CLASS_KEY, snap.get_properties()).as_deref(),
        );
        if storage_class == StorageClass::Ia {
            let estimated_size = Self::estimate_snapshot_data_size_for_ia(snap);
            return SnapshotSize {
                estimated_size,
                files: vec![],
            };
        }

        let mut files: Vec<FileWithSize> = vec![];
        files.extend(snap.get_l0_creates().iter().map(FileWithSize::from));
        files.extend(snap.get_table_creates().iter().map(FileWithSize::from));
        files.extend(snap.get_blob_creates().iter().map(FileWithSize::from));

        let estimated_size = files.iter().map(|f| f.size).sum::<u64>();
        SnapshotSize {
            estimated_size,
            files,
        }
    }

    pub async fn calibrate_restore_size_from_tikv(
        &self,
        snapshot_size: &SnapshotSize,
        region: &metapb::Region,
    ) -> Result<u64> {
        if snapshot_size.files.is_empty() {
            return Ok(snapshot_size.estimated_size);
        }

        let stores = region
            .get_peers()
            .iter()
            .filter_map(|p| (p.role == PeerRole::Voter).then_some(p.store_id))
            .collect::<Vec<u64>>();

        let nonexisted_files_on_stores = box_try!(
            self.stores_files
                .get_sst_files_nonexisted_on_stores(snapshot_size.files.clone(), stores)
                .await
        );
        let total_size = nonexisted_files_on_stores
            .into_iter()
            .flat_map(|(_, fs)| fs)
            .map(|f| f.size)
            .sum::<u64>();
        Ok(total_size)
    }

    fn estimate_snapshot_data_size_for_ia(snap: &pb::Snapshot) -> u64 {
        let mut estimated_size = 0;

        estimated_size += snap
            .get_l0_creates()
            .iter()
            .map(|l0| l0.size as u64)
            .sum::<u64>();
        // Calculate only meta size of IA tables.
        estimated_size += snap.get_table_creates().len() as u64 * AVG_TABLE_META_SIZE;
        estimated_size += snap
            .get_blob_creates()
            .iter()
            .map(|blob| blob.meta_offset as u64)
            .sum::<u64>();

        estimated_size
    }

    pub async fn consume_restore_size(&self, restore_size: u64) {
        self.bytes.consume(restore_size).await;
    }

    pub fn unconsume(&self, restore_size: u64) {
        self.bytes.unconsume(restore_size);
    }
}

#[derive(Clone, Copy, Debug)]
pub struct FileWithSize {
    id: u64,
    size: u64,
}

impl FileWithId for FileWithSize {
    fn file_id(&self) -> u64 {
        self.id
    }
}

impl From<&pb::L0Create> for FileWithSize {
    fn from(l0: &pb::L0Create) -> Self {
        Self {
            id: l0.id,
            size: l0.size as u64,
        }
    }
}

impl From<&pb::TableCreate> for FileWithSize {
    fn from(table: &pb::TableCreate) -> Self {
        Self {
            id: table.id,
            size: (table.meta_offset as f64 * TABLE_SIZE_MULTIPLIER) as u64,
        }
    }
}

// TODO: estimate blob file size more accurately if needed.
impl From<&pb::BlobCreate> for FileWithSize {
    fn from(blob: &pb::BlobCreate) -> Self {
        Self {
            id: blob.id,
            size: blob.meta_offset as u64,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;

    #[tokio::test]
    async fn test_byte_limiter_concurrent_consume() {
        let limiter = Arc::new(ByteLimiter::new(100 * 1024 * 1024, "test", 0)); // 100MB/s
        let num_tasks = 10;
        let bytes_per_task = 1024u64;

        let mut handles = vec![];
        for _ in 0..num_tasks {
            let limiter = limiter.clone();
            handles.push(tokio::spawn(async move {
                limiter.consume(bytes_per_task).await;
            }));
        }
        for h in handles {
            h.await.unwrap();
        }

        assert_eq!(
            limiter.total_consumed.load(Ordering::Relaxed),
            num_tasks * bytes_per_task,
        );
    }

    #[tokio::test]
    async fn test_byte_limiter_unconsume() {
        let limiter = ByteLimiter::new(100 * 1024 * 1024, "test", 0);
        limiter.consume(1000).await;
        limiter.unconsume(400);
        // total_consumed still reflects consumed amount (unconsume only returns
        // tokens).
        assert_eq!(limiter.total_consumed.load(Ordering::Relaxed), 1000);
    }

    #[test]
    fn test_file_with_size_from_l0() {
        let mut l0 = pb::L0Create::default();
        l0.id = 1;
        l0.size = 4096;
        let f = FileWithSize::from(&l0);
        assert_eq!(f.id, 1);
        assert_eq!(f.size, 4096);
    }

    #[test]
    fn test_file_with_size_from_table() {
        let mut table = pb::TableCreate::default();
        table.id = 2;
        table.meta_offset = 16_000_000; // ~15.26MB data region
        let f = FileWithSize::from(&table);
        assert_eq!(f.id, 2);
        assert_eq!(f.size, (16_000_000.0 * TABLE_SIZE_MULTIPLIER) as u64);
    }

    #[test]
    fn test_file_with_size_from_blob() {
        let mut blob = pb::BlobCreate::default();
        blob.id = 3;
        blob.meta_offset = 8_000_000;
        let f = FileWithSize::from(&blob);
        assert_eq!(f.id, 3);
        assert_eq!(f.size, 8_000_000);
    }
}
