// Copyright 2021 TiKV Project Authors. Licensed under Apache-2.0.

use std::time::Duration;

use batch_system::Config as BatchSystemConfig;
use engine_traits::{perf_level_serde, PerfLevel};
use fail::fail_point;
use online_config::OnlineConfig;
use raftstore::{
    coprocessor::config::{SPLIT_KEYS_PER_MB, SPLIT_SIZE_MB},
    store::config::SPLIT_REGION_MAX_KEYS_DEF,
    Result,
};
use serde::{Deserialize, Serialize};
use serde_with::with_prefix;
use tikv_util::{
    box_err,
    config::{ReadableDuration, ReadableSize},
    sys::SysQuota,
    warn,
};
use time::Duration as TimeDuration;

with_prefix!(prefix_apply "apply-");
with_prefix!(prefix_store "store-");
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, OnlineConfig)]
#[serde(default)]
pub struct Config {
    // =====================================================================
    // Deprecated configs (for compatibility with Raftstore)
    // ---------------------------------------------------------------------
    // These fields are retained for legacy compatibility and migration.
    // They may be ignored or have no effect in next-gen implementations.
    // =====================================================================
    #[online_config(skip)]
    pub raftdb_path: String,
    // Interval to compact unnecessary raft log.
    pub raft_log_compact_sync_interval: ReadableDuration,
    // A threshold to gc stale raft log, must >= 1.
    pub raft_log_gc_threshold: u64,
    // When entry count exceed this value, gc will be forced trigger.
    pub raft_log_gc_count_limit: Option<u64>,
    // When the approximate size of raft log entries exceed this value,
    // gc will be forced trigger.
    pub raft_log_gc_size_limit: Option<ReadableSize>,
    // Old Raft logs could be reserved if `raft_log_gc_threshold` is not reached.
    // GC them after ticks `raft_log_reserve_max_ticks` times.
    #[doc(hidden)]
    #[online_config(hidden)]
    pub raft_log_reserve_max_ticks: usize,
    // Old logs in Raft engine needs to be purged peridically.
    pub raft_engine_purge_interval: ReadableDuration,
    // When a peer is not responding for this time, leader will not keep entry cache for it.
    pub raft_entry_cache_life_time: ReadableDuration,
    // Deprecated! The configuration has no effect.
    // They are preserved for compatibility check.
    // When a peer is newly added, reject transferring leader to the peer for a while.
    #[doc(hidden)]
    #[serde(skip_serializing)]
    #[online_config(skip)]
    pub raft_reject_transfer_leader_duration: ReadableDuration,
    /// When size change of region exceed the diff since last check, it
    /// will be checked again whether it should be split.
    pub region_split_check_diff: Option<ReadableSize>,
    /// Interval (ms) to check whether start compaction for a region.
    pub region_compact_check_interval: ReadableDuration,
    /// Number of regions for each time checking.
    pub region_compact_check_step: u64,
    /// Minimum number of tombstones to trigger manual compaction.
    pub region_compact_min_tombstones: u64,
    /// Minimum percentage of tombstones to trigger manual compaction.
    /// Should between 1 and 100.
    pub region_compact_tombstones_percent: u64,
    // used to periodically check whether schedule pending applies in region runner
    #[doc(hidden)]
    #[online_config(skip)]
    pub region_worker_tick_interval: ReadableDuration,
    #[online_config(hidden)]
    pub report_region_flow_interval: ReadableDuration,
    // Right region derive origin region id when split.
    #[online_config(hidden)]
    pub right_derive_when_split: bool,

    #[online_config(skip)]
    pub snap_apply_batch_size: ReadableSize,
    pub snap_gc_timeout: ReadableDuration,
    #[online_config(skip)]
    pub snap_generator_pool_size: usize,
    pub snap_mgr_gc_tick_interval: ReadableDuration,

    // used to periodically check whether we should delete a stale peer's range in
    // region runner
    #[doc(hidden)]
    #[online_config(skip)]
    pub clean_stale_ranges_tick: usize,

    // Interval (ms) to check region whether the data is consistent.
    pub consistency_check_interval: ReadableDuration,

    pub lock_cf_compact_interval: ReadableDuration,
    pub lock_cf_compact_bytes_threshold: ReadableSize,

    pub peer_stale_state_check_interval: ReadableDuration,
    // Interval of scheduling a tick to check the leader lease.
    // It will be set to raft_store_max_leader_lease/4 by default.
    pub check_leader_lease_interval: ReadableDuration,

    #[online_config(skip)]
    pub notify_capacity: usize,
    pub messages_per_tick: usize,

    /// Max log gap allowed to propose merge.
    #[online_config(hidden)]
    pub merge_max_log_gap: u64,
    /// Interval to re-propose merge.
    pub merge_check_tick_interval: ReadableDuration,
    #[online_config(hidden)]
    pub use_delete_range: bool,
    pub cleanup_import_sst_interval: ReadableDuration,

    /// If it is 0, it means io tasks are handled in store threads.
    #[online_config(skip)]
    pub store_io_pool_size: usize,
    #[online_config(skip)]
    pub store_io_notify_capacity: usize,

    #[online_config(skip)]
    pub future_poll_size: usize,
    #[online_config(hidden)]
    pub apply_yield_duration: ReadableDuration,
    /// yield the fsm when apply flushed data size exceeds this threshold.
    /// the yield is check after commit, so the actual handled messages can be
    /// bigger than the configed value.
    // NOTE: the default value is much smaller than the default max raft batch msg size(0.2
    // * raft_entry_max_size), this is intentional because in the common case, a raft entry
    // is unlikely to exceed this threshold, but in case when raftstore is the bottleneck,
    // we still allow big raft batch for better throughput.
    pub apply_yield_write_size: ReadableSize,
    /// Whether to enable `hiberate` feature.
    ///
    ///  Not implemented in Next-Gen. Use `true` by default for compatibility.
    #[online_config(skip)]
    pub hibernate_regions: bool,
    #[doc(hidden)]
    #[online_config(hidden)]
    pub dev_assert: bool,
    #[serde(with = "perf_level_serde")]
    #[online_config(skip)]
    pub perf_level: PerfLevel,

    #[doc(hidden)]
    #[online_config(skip)]
    /// Disable this feature by set to 0, logic will be removed in other pr.
    /// When TiKV memory usage reaches `memory_usage_high_water` it will try to
    /// limit memory increasing. For raftstore layer entries will be evicted
    /// from entry cache, if they utilize memory more than
    /// `evict_cache_on_memory_ratio` * total.
    ///
    /// Set it to 0 can disable cache evict.
    // By default it's 0.2. So for different system memory capacity, cache evict happens:
    // * system=8G,  memory_usage_limit=6G,  evict=1.2G
    // * system=16G, memory_usage_limit=12G, evict=2.4G
    // * system=32G, memory_usage_limit=24G, evict=4.8G
    pub evict_cache_on_memory_ratio: f64,

    /// When the count of concurrent ready exceeds this value, command will not
    /// be proposed until the previous ready has been persisted.
    /// If `cmd_batch` is 0, this config will have no effect.
    /// If it is 0, it means no limit.
    pub cmd_batch_concurrent_ready_max_count: usize,

    /// When the size of raft db writebatch exceeds this value, write will be
    /// triggered.
    pub raft_write_size_limit: ReadableSize,
    pub waterfall_metrics: bool,

    pub io_reschedule_concurrent_max_count: usize,
    pub io_reschedule_hotpot_duration: ReadableDuration,

    // Deprecated! Batch is done in raft client.
    #[doc(hidden)]
    #[serde(skip_serializing)]
    #[online_config(skip)]
    pub raft_msg_flush_interval: ReadableDuration,

    // Deprecated! The time to clean stale peer safely can be decided based on RocksDB snapshot
    // sequence number.
    #[doc(hidden)]
    #[serde(skip_serializing)]
    #[online_config(skip)]
    pub clean_stale_peer_delay: ReadableDuration,

    // Interval to inspect the latency of raftstore for slow store detection.
    pub inspect_interval: ReadableDuration,

    // Interval to report min resolved ts, if it is zero, it means disabled.
    pub report_min_resolved_ts_interval: ReadableDuration,
    /// Interval to check whether to reactivate in-memory pessimistic lock after
    /// being disabled before transferring leader.
    pub reactive_memory_lock_tick_interval: ReadableDuration,
    /// Max tick count before reactivating in-memory pessimistic lock.
    pub reactive_memory_lock_timeout_tick: usize,
    // Interval of scheduling a tick to report region buckets.
    pub report_region_buckets_tick_interval: ReadableDuration,
    /// Interval to check long uncommitted proposals.
    #[doc(hidden)]
    pub check_long_uncommitted_interval: ReadableDuration,
    /// Base threshold of long uncommitted proposal.
    #[doc(hidden)]
    pub long_uncommitted_base_threshold: ReadableDuration,

    /// Max duration for the entry cache to be warmed up.
    /// Set it to 0 to disable warmup.
    pub max_entry_cache_warmup_duration: ReadableDuration,

    #[doc(hidden)]
    pub max_snapshot_file_raw_size: ReadableSize,

    pub unreachable_backoff: ReadableDuration,

    #[doc(hidden)]
    #[serde(skip_serializing)]
    #[online_config(hidden)]
    // Interval to check peers availability info.
    pub check_peers_availability_interval: ReadableDuration,

    // =====================================================================
    // Valid configurations
    // ---------------------------------------------------------------------
    // These fields are actively used and maintained in the current engine.
    // They represent the main operational parameters for Raftstore.
    // =====================================================================

    // minimizes disruption when a partitioned node rejoins the cluster by using a two phase
    // election.
    #[online_config(skip)]
    pub prevote: bool,

    // store capacity. 0 means no limit.
    #[online_config(skip)]
    pub capacity: ReadableSize,

    // raft_base_tick_interval is a base tick interval (ms).
    #[online_config(hidden)]
    pub raft_base_tick_interval: ReadableDuration,
    #[online_config(hidden)]
    pub raft_heartbeat_ticks: usize,
    #[online_config(hidden)]
    pub raft_election_timeout_ticks: usize,
    #[online_config(hidden)]
    pub raft_min_election_timeout_ticks: usize,
    #[online_config(hidden)]
    pub raft_max_election_timeout_ticks: usize,
    #[online_config(hidden)]
    pub raft_max_size_per_msg: ReadableSize,
    #[online_config(hidden)]
    pub raft_max_inflight_msgs: usize,
    // When the entry exceed the max size, reject to propose it.
    pub raft_entry_max_size: ReadableSize,

    // Interval to gc unnecessary raft log.
    pub raft_log_gc_tick_interval: ReadableDuration,

    // Interval (ms) to check region whether need to be split or not.
    pub split_region_check_tick_interval: ReadableDuration,

    pub pd_heartbeat_tick_interval: ReadableDuration,
    pub pd_store_heartbeat_tick_interval: ReadableDuration,

    /// When a peer is not active for max_peer_down_duration,
    /// the peer is considered to be down and is reported to PD.
    pub max_peer_down_duration: ReadableDuration,

    /// If the leader of a peer is missing for longer than
    /// max_leader_missing_duration, the peer would ask pd to confirm
    /// whether it is valid in any region. If the peer is stale and is not
    /// valid in any region, it will destroy itself.
    pub max_leader_missing_duration: ReadableDuration,
    /// Similar to the max_leader_missing_duration, instead it will log warnings
    /// and try to alert monitoring systems, if there is any.
    pub abnormal_leader_missing_duration: ReadableDuration,

    /// Interval to check peer states with ver low frequency.
    pub peer_long_check_interval: ReadableDuration,

    #[online_config(hidden)]
    pub leader_transfer_max_log_lag: u64,

    // The lease provided by a successfully proposed and applied entry.
    pub raft_store_max_leader_lease: ReadableDuration,
    // Check if leader lease will expire at `current_time + renew_leader_lease_advance_duration`.
    // It will be set to raft_store_max_leader_lease/4 by default.
    pub renew_leader_lease_advance_duration: ReadableDuration,

    /// This setting can only ensure conf remove will not be proposed by the
    /// peer being removed. But it can't guarantee the remove is applied
    /// when the target is not leader. That means we always need to check if
    /// it's working as expected when a leader applies a self-remove conf
    /// change. Keep the configuration only for convenient test.
    pub allow_remove_leader: bool,

    /// Maximum size of every local read task batch.
    pub local_read_batch_size: u64,

    #[online_config(submodule)]
    #[serde(flatten, with = "prefix_apply")]
    pub apply_batch_system: BatchSystemConfig,

    #[online_config(submodule)]
    #[serde(flatten, with = "prefix_store")]
    pub store_batch_system: BatchSystemConfig,

    pub cmd_batch: bool,

    // Deprecated! These configuration has been moved to Coprocessor.
    // They are preserved for compatibility check.
    #[doc(hidden)]
    #[serde(skip_serializing)]
    #[online_config(skip)]
    pub region_max_size: ReadableSize,
    #[doc(hidden)]
    #[serde(skip_serializing)]
    #[online_config(skip)]
    pub region_split_size: ReadableSize,

    /// The minimal count of region pending on applying raft logs.
    /// Only when the count of regions which not pending on applying logs is
    /// less than the threshold, can the raftstore supply service.
    #[online_config(hidden)]
    pub min_pending_apply_region_count: u64,
    // =====================================================================
    // Extra configs for Next-gen
    // ---------------------------------------------------------------------
    // These fields are exclusive to the next-generation engine features.
    // They enable advanced optimizations and new capabilities.
    // =====================================================================

    // cloud engine configs
    pub local_file_gc_tick_interval: ReadableDuration,
    pub local_file_gc_timeout: ReadableDuration,
    /// Number of auxiliary worker threads for background tasks.
    /// Used to offload non-critical work and improve throughput.
    pub aux_worker_count: usize,
    /// Number of schema worker threads for schema-related background tasks.
    /// Used for DDL, schema changes, and metadata management in next-gen
    /// engine.
    pub schema_worker_count: usize,
    /// Maximum CPU utilization threshold for auxiliary workers.
    /// Lower than main worker to ensure synchronization and avoid contention.
    pub aux_worker_max_util: usize,
    /// Maximum CPU utilization threshold for main raft worker before offloading
    /// to aux workers. When exceeded, auxiliary workers are activated to
    /// help process tasks.
    pub main_worker_max_util: usize,

    /// The maximum batch size for raft worker in bytes.
    /// Controls how much data is processed in each raft worker batch for
    /// efficiency.
    pub raft_worker_max_batch_size: ReadableSize,

    /// The minimum duration for IO worker to write data (ms).
    /// Used to avoid too frequent writes and reduce write amplification in
    /// next-gen engine.
    pub io_worker_min_write_duration: ReadableDuration,

    /// Capacity of the internal channel for raftstore batch processing.
    /// Larger values allow more concurrent requests but increase memory usage.
    pub channel_capacity: usize,

    /// Enable inner key offset optimization for next-gen engine.
    /// Used for advanced key encoding and fast lookups.
    #[doc(hidden)]
    #[online_config(skip)]
    pub enable_inner_key_offset: bool,

    /// Enable region bucket feature for fine-grained region management.
    /// When true, regions are split into buckets for parallelism and isolation.
    pub enable_region_bucket: bool,
    /// Size of each region bucket in bytes.
    /// Used only when enable_region_bucket is true.
    pub region_bucket_size: ReadableSize,
    /// Number of split keys to use when splitting regions in next-gen engine.
    /// Controls granularity of region splits for balancing and compaction.
    pub region_split_keys: u64,
    // Maximum number of split_keys in a split region request.
    pub split_region_max_keys: usize,

    /// Interval (ms) to check region whether to switch mem-table for write
    /// separation. Used in next-gen engine to optimize write amplification
    /// and memory usage.
    pub switch_mem_table_check_tick_interval: ReadableDuration,

    /// Interval (ms) to update GC safe point (aka update_safe_ts_interval).
    /// Controls how frequently the system advances the global GC timestamp for
    /// MVCC.
    #[serde(alias = "update_safe_ts_interval")]
    pub update_gc_safe_point_interval: ReadableDuration,

    /// When mem-table is empty and applied to last index,
    /// if no kv raft log entries exceeds this value, gc will be triggered.
    pub raft_log_gc_no_kv_count: u64,
}

impl Default for Config {
    fn default() -> Config {
        let num_cpus = tikv_util::sys::SysQuota::cpu_cores_quota() as usize;
        Config {
            // Deprecated configs [for compatibility to Raftstore]
            raftdb_path: String::new(),
            raft_log_compact_sync_interval: ReadableDuration::secs(2),
            raft_log_gc_threshold: 50,
            raft_log_gc_count_limit: None,
            raft_log_gc_size_limit: Some(ReadableSize::mb(32)),
            raft_log_reserve_max_ticks: 6,
            raft_engine_purge_interval: ReadableDuration::secs(10),
            raft_entry_cache_life_time: ReadableDuration::secs(30),
            raft_reject_transfer_leader_duration: ReadableDuration::secs(3),
            region_split_check_diff: None,
            region_compact_check_interval: ReadableDuration::minutes(5),
            region_compact_check_step: 100,
            region_compact_min_tombstones: 10000,
            region_compact_tombstones_percent: 30,
            region_worker_tick_interval: if cfg!(feature = "test") {
                ReadableDuration::millis(200)
            } else {
                ReadableDuration::millis(1000)
            },
            report_region_flow_interval: ReadableDuration::minutes(1),
            right_derive_when_split: true,
            snap_apply_batch_size: ReadableSize::mb(10),
            snap_gc_timeout: ReadableDuration::hours(4),
            snap_generator_pool_size: 2,
            snap_mgr_gc_tick_interval: ReadableDuration::minutes(1),
            clean_stale_ranges_tick: if cfg!(feature = "test") { 1 } else { 10 },
            consistency_check_interval: ReadableDuration::secs(0),
            lock_cf_compact_interval: ReadableDuration::minutes(10),
            lock_cf_compact_bytes_threshold: ReadableSize::mb(256),
            peer_stale_state_check_interval: ReadableDuration::minutes(5),
            check_leader_lease_interval: ReadableDuration::secs(0),
            notify_capacity: 40960,
            messages_per_tick: 4096,
            merge_max_log_gap: 10,
            merge_check_tick_interval: ReadableDuration::secs(2),
            use_delete_range: false,
            cleanup_import_sst_interval: ReadableDuration::minutes(10),
            store_io_pool_size: 0,
            store_io_notify_capacity: 40960,
            future_poll_size: 1,
            apply_yield_duration: ReadableDuration::millis(500),
            apply_yield_write_size: ReadableSize::kb(32),
            hibernate_regions: true,
            dev_assert: false,
            perf_level: PerfLevel::Uninitialized,
            evict_cache_on_memory_ratio: 0.0,
            cmd_batch_concurrent_ready_max_count: 1,
            raft_write_size_limit: ReadableSize::mb(1),
            waterfall_metrics: true,
            io_reschedule_concurrent_max_count: 4,
            io_reschedule_hotpot_duration: ReadableDuration::secs(5),
            raft_msg_flush_interval: ReadableDuration::micros(250),
            clean_stale_peer_delay: ReadableDuration::minutes(0),
            inspect_interval: ReadableDuration::millis(500),
            report_min_resolved_ts_interval: ReadableDuration::secs(1),
            report_region_buckets_tick_interval: ReadableDuration::secs(10),
            max_snapshot_file_raw_size: ReadableSize::mb(100),
            unreachable_backoff: ReadableDuration::secs(10),
            check_peers_availability_interval: ReadableDuration::secs(30),
            reactive_memory_lock_tick_interval: ReadableDuration::secs(2),
            reactive_memory_lock_timeout_tick: 5,
            check_long_uncommitted_interval: ReadableDuration::secs(10),
            // In some cases, such as rolling upgrade, some regions' commit log
            // duration can be 12 seconds. Before #13078 is merged,
            // the commit log duration can be 2.8 minutes. So maybe
            // 20s is a relatively reasonable base threshold. Generally,
            // the log commit duration is less than 1s. Feel free to adjust
            // this config :)
            long_uncommitted_base_threshold: ReadableDuration::secs(20),
            max_entry_cache_warmup_duration: ReadableDuration::secs(1),
            // Valid configs
            prevote: true,
            capacity: ReadableSize(0),
            raft_base_tick_interval: ReadableDuration::secs(1),
            raft_heartbeat_ticks: 2,
            raft_election_timeout_ticks: 10,
            raft_min_election_timeout_ticks: 0,
            raft_max_election_timeout_ticks: 0,
            raft_max_size_per_msg: ReadableSize::mb(1),
            raft_max_inflight_msgs: 256,
            raft_entry_max_size: ReadableSize::mb(8),
            raft_log_gc_tick_interval: ReadableDuration::secs(3),
            split_region_check_tick_interval: ReadableDuration::secs(3),
            pd_heartbeat_tick_interval: ReadableDuration::minutes(1),
            pd_store_heartbeat_tick_interval: ReadableDuration::secs(10),
            max_peer_down_duration: ReadableDuration::minutes(10),
            max_leader_missing_duration: ReadableDuration::hours(2),
            abnormal_leader_missing_duration: ReadableDuration::minutes(10),
            peer_long_check_interval: ReadableDuration::minutes(5),
            leader_transfer_max_log_lag: 512,
            raft_store_max_leader_lease: ReadableDuration::secs(9),
            renew_leader_lease_advance_duration: ReadableDuration::secs(0),
            allow_remove_leader: false,
            local_read_batch_size: 1024,
            apply_batch_system: BatchSystemConfig {
                pool_size: (num_cpus / 4).max(1),
                low_priority_pool_size: (num_cpus / 4).max(1),
                ..Default::default()
            },
            store_batch_system: BatchSystemConfig::default(),
            cmd_batch: true,
            region_max_size: ReadableSize(0),
            region_split_size: ReadableSize::mb(SPLIT_SIZE_MB),
            // Extra configs for Next-gen
            local_file_gc_tick_interval: ReadableDuration::minutes(10),
            local_file_gc_timeout: ReadableDuration::minutes(30),
            aux_worker_count: num_cpus / 8,
            schema_worker_count: (num_cpus / 16).max(1),
            aux_worker_max_util: 60,
            main_worker_max_util: 80,
            raft_worker_max_batch_size: ReadableSize::mb(1),
            io_worker_min_write_duration: ReadableDuration::millis(1),
            channel_capacity: 40960,
            enable_inner_key_offset: false,
            enable_region_bucket: false,
            region_bucket_size: ReadableSize::mb(96),
            region_split_keys: SPLIT_SIZE_MB * SPLIT_KEYS_PER_MB,
            split_region_max_keys: SPLIT_REGION_MAX_KEYS_DEF,
            switch_mem_table_check_tick_interval: ReadableDuration::minutes(1),
            update_gc_safe_point_interval: ReadableDuration::secs(60),
            raft_log_gc_no_kv_count: 4,
            min_pending_apply_region_count: 10,
        }
    }
}

impl Config {
    pub fn new() -> Config {
        Config::default()
    }

    pub fn raft_store_max_leader_lease(&self) -> TimeDuration {
        TimeDuration::from_std(self.raft_store_max_leader_lease.0).unwrap()
    }

    pub fn renew_leader_lease_advance_duration(&self) -> TimeDuration {
        TimeDuration::from_std(self.renew_leader_lease_advance_duration.0).unwrap()
    }

    pub fn raft_heartbeat_interval(&self) -> Duration {
        self.raft_base_tick_interval.0 * self.raft_heartbeat_ticks as u32
    }

    pub fn raft_log_gc_count_limit(&self) -> u64 {
        self.raft_log_gc_count_limit.unwrap()
    }

    pub fn raft_log_gc_size_limit(&self) -> ReadableSize {
        self.raft_log_gc_size_limit.unwrap()
    }

    pub fn region_split_check_diff(&self) -> ReadableSize {
        self.region_split_check_diff.unwrap()
    }

    #[cfg(any(test, feature = "testexport"))]
    pub fn allow_remove_leader(&self) -> bool {
        self.allow_remove_leader
    }

    #[cfg(not(any(test, feature = "testexport")))]
    pub fn allow_remove_leader(&self) -> bool {
        false
    }

    pub fn validate(
        &mut self,
        region_split_size: ReadableSize,
        region_split_keys: Option<u64>,
        enable_region_bucket: bool,
        region_bucket_size: ReadableSize,
    ) -> Result<()> {
        if self.raft_heartbeat_ticks == 0 {
            return Err(box_err!("heartbeat tick must greater than 0"));
        }

        if self.raft_election_timeout_ticks != 10 {
            warn!(
                "Election timeout ticks needs to be same across all the cluster, \
                 otherwise it may lead to inconsistency."
            );
        }

        if self.raft_election_timeout_ticks <= self.raft_heartbeat_ticks {
            return Err(box_err!(
                "election tick must be greater than heartbeat tick"
            ));
        }

        if self.raft_min_election_timeout_ticks == 0 {
            self.raft_min_election_timeout_ticks = self.raft_election_timeout_ticks;
        }

        if self.raft_max_election_timeout_ticks == 0 {
            self.raft_max_election_timeout_ticks = self.raft_election_timeout_ticks * 2;
        }

        if self.raft_min_election_timeout_ticks < self.raft_election_timeout_ticks
            || self.raft_min_election_timeout_ticks >= self.raft_max_election_timeout_ticks
        {
            return Err(box_err!(
                "invalid timeout range [{}, {}) for timeout {}",
                self.raft_min_election_timeout_ticks,
                self.raft_max_election_timeout_ticks,
                self.raft_election_timeout_ticks
            ));
        }

        // The adjustment of this value is related to the number of regions, usually
        // 16384 is already a large enough value
        if self.raft_max_inflight_msgs == 0 || self.raft_max_inflight_msgs > 16384 {
            return Err(box_err!(
                "raft max inflight msgs should be greater than 0 and less than or equal to 16384"
            ));
        }

        if self.raft_max_size_per_msg.0 == 0 || self.raft_max_size_per_msg.0 > ReadableSize::gb(3).0
        {
            return Err(box_err!(
                "raft max size per message should be greater than 0 and less than or equal to 3GiB"
            ));
        }

        if self.raft_entry_max_size.0 == 0 || self.raft_entry_max_size.0 > ReadableSize::gb(3).0 {
            return Err(box_err!(
                "raft entry max size should be greater than 0 and less than or equal to 3GiB"
            ));
        }

        let election_timeout =
            self.raft_base_tick_interval.as_millis() * self.raft_election_timeout_ticks as u64;
        let lease = self.raft_store_max_leader_lease.as_millis();
        if election_timeout < lease {
            return Err(box_err!(
                "election timeout {} ms is less than lease {} ms",
                election_timeout,
                lease
            ));
        }

        let tick = self.raft_base_tick_interval.as_millis();
        if lease > election_timeout - tick {
            return Err(box_err!(
                "lease {} ms should not be greater than election timeout {} ms - 1 tick({} ms)",
                lease,
                election_timeout,
                tick
            ));
        }

        let stale_state_check = self.peer_stale_state_check_interval.as_millis();
        if stale_state_check < election_timeout * 2 {
            return Err(box_err!(
                "peer stale state check interval {} ms is less than election timeout x 2 {} ms",
                stale_state_check,
                election_timeout * 2
            ));
        }

        if self.leader_transfer_max_log_lag < 10 {
            return Err(box_err!(
                "raftstore.leader-transfer-max-log-lag should be >= 10."
            ));
        }

        let abnormal_leader_missing = self.abnormal_leader_missing_duration.as_millis();
        if abnormal_leader_missing < stale_state_check {
            return Err(box_err!(
                "abnormal leader missing {} ms is less than peer stale state check interval {} ms",
                abnormal_leader_missing,
                stale_state_check
            ));
        }

        let max_leader_missing = self.max_leader_missing_duration.as_millis();
        if max_leader_missing < abnormal_leader_missing {
            return Err(box_err!(
                "max leader missing {} ms is less than abnormal leader missing {} ms",
                max_leader_missing,
                abnormal_leader_missing
            ));
        }

        if self.io_worker_min_write_duration.as_millis() > 1000 {
            return Err(box_err!(
                "io-worker-min-write-duration should be less than 1s, current value is {}s",
                self.io_worker_min_write_duration.as_millis()
            ));
        }

        // Since the following configuration supports online update, in order to
        // prevent mistakenly inputting too large values, the max limit is made
        // according to the cpu quota * 10. Notice 10 is only an estimate, not an
        // empirical value.
        let limit = (SysQuota::cpu_cores_quota() * 10.0) as usize;
        if self.apply_batch_system.pool_size == 0 || self.apply_batch_system.pool_size > limit {
            return Err(box_err!(
                "apply-pool-size should be greater than 0 and less than or equal to: {}",
                limit
            ));
        }
        if let Some(size) = self.apply_batch_system.max_batch_size {
            if size == 0 || size > 10240 {
                return Err(box_err!(
                    "apply-max-batch-size should be greater than 0 and less than or equal to 10240"
                ));
            }
        } else {
            self.apply_batch_system.max_batch_size = Some(256);
        }
        if self.store_batch_system.pool_size == 0 || self.store_batch_system.pool_size > limit {
            return Err(box_err!(
                "store-pool-size should be greater than 0 and less than or equal to: {}",
                limit
            ));
        }
        if self.store_batch_system.low_priority_pool_size > 0 {
            // The store thread pool doesn't need a low-priority thread currently.
            self.store_batch_system.low_priority_pool_size = 0;
        }
        if let Some(size) = self.store_batch_system.max_batch_size {
            if size == 0 || size > 10240 {
                return Err(box_err!(
                    "store-max-batch-size should be greater than 0 and less than or equal to 10240"
                ));
            }
        } else if self.hibernate_regions {
            self.store_batch_system.max_batch_size = Some(256);
        } else {
            self.store_batch_system.max_batch_size = Some(1024);
        }

        // Avoid hibernated peer being reported as down peer.
        if self.hibernate_regions {
            self.max_peer_down_duration = std::cmp::max(
                self.max_peer_down_duration,
                self.peer_stale_state_check_interval * 2,
            );
        }

        if self.renew_leader_lease_advance_duration.as_millis() == 0 && self.hibernate_regions {
            self.renew_leader_lease_advance_duration = self.raft_store_max_leader_lease / 4;
        }

        #[cfg(not(any(test, feature = "testexport")))]
        if self.max_snapshot_file_raw_size.0 != 0 && self.max_snapshot_file_raw_size.as_mb() < 100 {
            return Err(box_err!(
                "max_snapshot_file_raw_size should be no less than 100MB."
            ));
        }

        if self.local_file_gc_timeout.as_millis() < self.local_file_gc_tick_interval.as_millis() {
            return Err(box_err!(
                "local file gc timeout should be no less than local_file_gc_tick_interval"
            ));
        }

        if self.local_file_gc_tick_interval.as_millis() < self.raft_base_tick_interval.as_millis() {
            return Err(box_err!(
                "local_file_gc_tick_interval should be at lest than raft_base_tick_interval"
            ));
        }

        if self.raft_log_gc_threshold < 1 {
            return Err(box_err!(
                "raft log gc threshold must >= 1, not {}",
                self.raft_log_gc_threshold
            ));
        }

        self.region_split_size = region_split_size;
        if let Some(split_keys) = region_split_keys {
            self.region_split_keys = split_keys;
        }
        self.enable_region_bucket = enable_region_bucket;
        self.region_bucket_size = region_bucket_size;
        match self.raft_log_gc_size_limit {
            Some(size_limit) => {
                if size_limit.0 == 0 {
                    return Err(box_err!("raft log gc size limit should large than 0."));
                }
            }
            None => self.raft_log_gc_size_limit = Some(region_split_size * 3 / 4),
        }
        match self.raft_log_gc_count_limit {
            Some(count_limit) => {
                if self.merge_max_log_gap >= count_limit {
                    return Err(box_err!(
                        "merge log gap {} should be less than log gc limit {}.",
                        self.merge_max_log_gap,
                        count_limit
                    ));
                }
            }
            None => {
                // Assume the average size of entries is 1k.
                self.raft_log_gc_count_limit =
                    Some(region_split_size * 3 / 4 / ReadableSize::kb(1));
            }
        }
        match self.region_split_check_diff {
            Some(split_check_diff) => {
                if split_check_diff.0 == 0 {
                    return Err(box_err!("region split check diff should large than 0."));
                }
            }
            None => {
                self.region_split_check_diff = if !enable_region_bucket {
                    Some(region_split_size / 16)
                } else {
                    Some(ReadableSize(std::cmp::min(
                        region_split_size.0 / 16,
                        region_bucket_size.0,
                    )))
                }
            }
        }

        if self.min_pending_apply_region_count == 0 {
            return Err(box_err!(
                "min_pending_apply_region_count must be greater than 0"
            ));
        }

        // For tests only.
        if cfg!(debug_assertions) && self.raft_base_tick_interval.as_millis() < 100 {
            // It is a test config, adjust the fields not included in the old.
            self.update_gc_safe_point_interval.0 = self.raft_base_tick_interval.0 * 60;
            self.switch_mem_table_check_tick_interval.0 = self.raft_base_tick_interval.0 * 60;
            if self.local_file_gc_timeout.0 > self.raft_base_tick_interval.0 * 20 * 30 {
                self.local_file_gc_timeout.0 = self.raft_base_tick_interval.0 * 20 * 30; // 30s, see `new_test_config`.
                self.local_file_gc_tick_interval.0 = self.raft_base_tick_interval.0 * 20 * 10; // 10s, see `new_test_config`.
            }
            // cover aux worker in test
            if self.aux_worker_count == 0 {
                self.aux_worker_count = 1;
            }
            self.main_worker_max_util = 16;
            self.aux_worker_max_util = 12;

            self.schema_worker_count = self.schema_worker_count.max(2);
        }

        (|| {
            fail_point!(
                "rfstore_config_from_old_force_short_update_gc_safe_point_interval",
                |_| {
                    self.update_gc_safe_point_interval = self.raft_base_tick_interval;
                }
            )
        })();

        Ok(())
    }

    // TODO
    pub fn write_into_metrics(&self) {}
}

// TODO: online configurations for RfStoreConfig
