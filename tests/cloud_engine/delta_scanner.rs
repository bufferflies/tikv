// Copyright 2025 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    collections::{HashMap, HashSet},
    ops::Deref,
    time::Duration,
};

use kvengine::table::InnerKey;
use kvproto::kvrpcpb::Assertion;
use rand::{rngs::StdRng, Rng, SeedableRng};
use test_cloud_server::{
    client::{ClusterClient, TxnMutations},
    util::Mutation,
    ServerCluster,
};
use tikv::storage::txn::{CloudDeltaScanner, TxnEntry, TxnEntryScanner};
use tikv_util::config::ReadableSize;
use txn_types::{Key, Lock, OldValue};

use crate::{alloc_node_id_vec, i_to_key, i_to_val};

fn wait_region_ready(cluster: &ServerCluster, _node_id: u16, region_id: u64) {
    // Force flush the region's memtable and wait until it's empty so snapshot reads
    // are stable.
    let _ = cluster.flush_memtable(region_id);
    let ok = cluster.wait_for_memtable_flushed(region_id, Duration::from_secs(5));
    assert!(ok, "memtable not flushed for region {}", region_id);
}

const THAT_MAGIC_KEYSPACE_ID: u32 = u32::from_be_bytes(*b"\x00key");

#[derive(Default, Clone, PartialEq, Eq)]
struct EntryAgg {
    commit_vals: Vec<Vec<u8>>,
    prewrite_vals: Vec<Vec<u8>>,
}

impl std::fmt::Debug for EntryAgg {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EntryAgg")
            .field("commit_vals", &self.commit_vals.len())
            .field("prewrite_vals", &self.prewrite_vals.len())
            .finish()
    }
}

fn normalize_key(raw: &[u8]) -> Vec<u8> {
    // Compare on InnerKey to be robust with/without inner key offset.
    InnerKey::from_outer_key(raw).deref().to_vec()
}

fn raw_key_for_entry(entry: &TxnEntry) -> Vec<u8> {
    match entry {
        TxnEntry::Commit { .. } => entry.to_key().unwrap().into_raw().unwrap(),
        TxnEntry::Prewrite {
            lock: (lock_key, _),
            ..
        } => Key::from_encoded_slice(lock_key.as_slice())
            .into_raw()
            .unwrap(),
    }
}

// Additional aggregation used by randomized test to validate old_value per
// version
#[derive(Default)]
struct SeqAgg {
    // Commit values in the order produced by the delta scanner (newest -> oldest)
    commit_vals_scan_order: Vec<Vec<u8>>,
    // For each commit above, the corresponding old_value converted to Option<Vec<u8>>:
    //   Some(v) when OldValue::Value(v), None when it's a timestamp or None variant.
    commit_old_opt: Vec<Option<Vec<u8>>>,
    // Prewrite values (order doesn't matter for validation; kept for completeness)
    prewrite_vals: Vec<Vec<u8>>,
    // For each prewrite above, the corresponding old_value converted to Option<Vec<u8>>
    // (Some(v) when OldValue::Value(v), None otherwise).
    prewrite_old_opt: Vec<Option<Vec<u8>>>,
}

fn to_hex(bytes: &[u8]) -> String {
    hex::encode(bytes)
}

impl std::fmt::Debug for SeqAgg {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let commit_vals_hex: Vec<String> = self
            .commit_vals_scan_order
            .iter()
            .map(|v| to_hex(v))
            .collect();
        let commit_old_opt_hex: Vec<Option<String>> = self
            .commit_old_opt
            .iter()
            .map(|ov| ov.as_ref().map(|v| to_hex(v)))
            .collect();
        let prewrite_vals_hex: Vec<String> = self.prewrite_vals.iter().map(|v| to_hex(v)).collect();
        let prewrite_old_opt_hex: Vec<Option<String>> = self
            .prewrite_old_opt
            .iter()
            .map(|ov| ov.as_ref().map(|v| to_hex(v)))
            .collect();

        f.debug_struct("SeqAgg")
            .field("commit_vals_scan_order", &commit_vals_hex)
            .field("commit_old_opt", &commit_old_opt_hex)
            .field("prewrite_vals", &prewrite_vals_hex)
            .field("prewrite_old_opt", &prewrite_old_opt_hex)
            .finish()
    }
}

fn old_value_to_opt(ov: &OldValue) -> Option<Vec<u8>> {
    match ov {
        OldValue::Value { value } => Some(value.clone()),
        _ => None,
    }
}

fn add_entry_seq(agg: &mut HashMap<Vec<u8>, SeqAgg>, entry: &TxnEntry) {
    let raw = raw_key_for_entry(entry);
    let key = normalize_key(&raw);
    let e = agg.entry(key).or_default();
    match entry {
        TxnEntry::Commit { .. } => {
            let (_k, v) = entry.clone().into_kvpair().unwrap();
            // Extract old_value from the entry without consuming it fully
            if let TxnEntry::Commit { old_value, .. } = entry {
                e.commit_vals_scan_order.push(v);
                e.commit_old_opt.push(old_value_to_opt(old_value));
            }
        }
        TxnEntry::Prewrite {
            lock: (_raw_k, lock_bytes),
            default: (_k, v),
            ..
        } => {
            let l = Lock::parse(lock_bytes).expect("parse lock bytes");
            if let Some(sv) = l.short_value {
                e.prewrite_vals.push(sv);
            } else {
                e.prewrite_vals.push(v.clone());
            }
            if let TxnEntry::Prewrite { old_value, .. } = entry {
                e.prewrite_old_opt.push(old_value_to_opt(old_value));
            }
        }
    }
}

fn sort_agg_map(m: &mut HashMap<Vec<u8>, EntryAgg>) {
    for (_k, v) in m.iter_mut() {
        v.commit_vals.sort_unstable();
        v.prewrite_vals.sort_unstable();
    }
}

// Use rand crate with a deterministic seed for test randomization.

// Convert a SeqAgg map into an EntryAgg-like map and sort values for
// set-equality checks.
fn to_sorted_sets_from_seq(actual_seqs: &HashMap<Vec<u8>, SeqAgg>) -> HashMap<Vec<u8>, EntryAgg> {
    let mut derived: HashMap<Vec<u8>, EntryAgg> = HashMap::new();
    for (k, seq) in actual_seqs.iter() {
        let e = derived.entry(k.clone()).or_default();
        e.commit_vals
            .extend(seq.commit_vals_scan_order.iter().cloned());
        e.prewrite_vals.extend(seq.prewrite_vals.iter().cloned());
    }
    sort_agg_map(&mut derived);
    derived
}

// Shared verification: compare sets (commit/prewrite) and, optionally, verify
// per-key commit sequence (newest->oldest) and old_value semantics.
#[track_caller]
fn verify_scan(
    actual_seqs: &HashMap<Vec<u8>, SeqAgg>,
    expected_sets: &HashMap<Vec<u8>, EntryAgg>,
    verify_seq_and_old: bool,
) {
    // Set-wise equality (existing behavior)
    let actual_sets_sorted = to_sorted_sets_from_seq(actual_seqs);
    let mut expected_sorted = expected_sets.clone();
    sort_agg_map(&mut expected_sorted);
    assert_eq!(expected_sorted, actual_sets_sorted);

    if !verify_seq_and_old {
        return;
    }

    // Sequence and old_value checks derived from expected commit insertion
    // order (oldest->newest) recorded by RecordingClient.
    for (k, exp) in expected_sets.iter() {
        let expected_commit_scan: Vec<Vec<u8>> = exp.commit_vals.iter().cloned().rev().collect();
        // For old_value: when the immediate older version is a delete (empty vec),
        // the correct old_value should be None; otherwise it's the older value.
        let expected_old_opt: Vec<Option<Vec<u8>>> = (0..expected_commit_scan.len())
            .map(|i| {
                if i + 1 < expected_commit_scan.len() {
                    let older = &expected_commit_scan[i + 1];
                    if older.is_empty() {
                        None
                    } else {
                        Some(older.clone())
                    }
                } else {
                    None
                }
            })
            .collect();

        if let Some(actual) = actual_seqs.get(k) {
            assert_eq!(
                expected_commit_scan, actual.commit_vals_scan_order,
                "commit sequence (newest->oldest) mismatch for key {:?}",
                k
            );
            assert_eq!(
                expected_old_opt, actual.commit_old_opt,
                "old_value sequence mismatch for key {:?}",
                k
            );

            // Additionally verify prewrite old_value equals the latest committed value
            // for the key (or None when the latest is delete or no commit exists).
            if !actual.prewrite_old_opt.is_empty() {
                let latest_commit_opt = exp.commit_vals.last();
                let expected_prewrite_old = latest_commit_opt
                    .and_then(|v| if v.is_empty() { None } else { Some(v.clone()) });
                let expected_vec = vec![expected_prewrite_old; actual.prewrite_old_opt.len()];
                assert_eq!(
                    expected_vec, actual.prewrite_old_opt,
                    "prewrite old_value mismatch for key {:?}",
                    k
                );
            }
        } else {
            assert!(
                expected_commit_scan.is_empty(),
                "missing scan entries for key {:?}",
                k
            );
        }
    }
}

/// A thin wrapper over the cluster client that records the expected EntryAggs
/// while inserting key-values (commits) and issuing prewrites (uncommitted).
struct RecordingClient {
    inner: ClusterClient,
    expected: HashMap<Vec<u8>, EntryAgg>,
}

impl RecordingClient {
    fn new(inner: ClusterClient) -> Self {
        Self {
            inner,
            expected: HashMap::new(),
        }
    }

    /// Insert one committed key-value (equivalent to inner.put_kv with a single
    /// item) and record it into expected.commit_vals.
    fn put_commit(&mut self, key: Vec<u8>, val: Vec<u8>) {
        // Execute
        self.inner.put_kv(0..1, |_| key.clone(), |_| val.clone());
        // Record
        let k = normalize_key(&key);
        let e = self.expected.entry(k).or_default();
        e.commit_vals.push(val);
    }

    /// Prewrite a single Put mutation and record the value into
    /// expected.prewrite_vals.
    fn prewrite_put(&mut self, key: Vec<u8>, val: Vec<u8>) {
        let start_ts = self.inner.get_ts();
        let muts = vec![Mutation {
            key: key.clone().into(),
            value: val.clone().into(),
            op: kvproto::kvrpcpb::Op::Put,
            assertion: Assertion::None,
        }];
        self.inner
            .kv_prewrite_with_retry(
                key.clone().into(),
                None,
                TxnMutations::from_normal(muts),
                start_ts,
            )
            .expect("prewrite");
        // Record
        let k = normalize_key(&key);
        let e = self.expected.entry(k).or_default();
        e.prewrite_vals.push(val);
    }

    /// Commit a Delete on a single key and record an empty value in
    /// expected.commit_vals (deletes are represented by empty values).
    fn delete_commit(&mut self, key: Vec<u8>) {
        // Execute a single-key delete transaction
        self.inner.del_kv(0..1, |_| key.clone());
        // Record as a delete (empty value)
        let k = normalize_key(&key);
        let e = self.expected.entry(k).or_default();
        e.commit_vals.push(Vec::new());
    }

    /// Acquire a pessimistic lock on a single key.
    /// Note: These locks should be filtered out by CloudDeltaScanner, so we
    /// don't record them in expected.
    fn acquire_pessimistic_lock(&mut self, key: Vec<u8>) {
        let start_ts = self.inner.get_ts();
        let for_update_ts = self.inner.get_ts();
        let lock_ttl = 3000; // 3 seconds
        self.inner
            .kv_pessimistic_lock(
                key.clone().into(),
                vec![key.into()],
                start_ts,
                lock_ttl,
                for_update_ts,
                None,
            )
            .expect("pessimistic lock");
        // DO NOT record in expected - these locks should be filtered out
    }

    /// Prewrite a Lock-type mutation (optimistic lock without value).
    /// Note: These locks should be filtered out by CloudDeltaScanner, so we
    /// don't record them in expected.
    fn prewrite_lock(&mut self, key: Vec<u8>) {
        let start_ts = self.inner.get_ts();
        let muts = vec![Mutation {
            key: key.clone().into(),
            value: vec![].into(),
            op: kvproto::kvrpcpb::Op::Lock,
            assertion: Assertion::None,
        }];
        self.inner
            .kv_prewrite_with_retry(key.into(), None, TxnMutations::from_normal(muts), start_ts)
            .expect("prewrite lock");
        // DO NOT record in expected - these locks should be filtered out
    }
}

impl std::ops::Deref for RecordingClient {
    type Target = ClusterClient;
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl std::ops::DerefMut for RecordingClient {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

#[test]
fn delta_scanner_multi_versions_same_key() {
    test_util::init_log_for_test();
    let cluster = ServerCluster::new(alloc_node_id_vec(1), |_, conf| {
        conf.raft_store.enable_inner_key_offset = false;
        conf.coprocessor.region_split_size = ReadableSize::gb(1);
    });
    let client = cluster.new_client();
    let mut client = RecordingClient::new(client);
    let node_id = cluster.get_nodes()[0];

    // Pick one key and write multiple committed versions
    let k = i_to_key(100);
    for i in 0..3usize {
        let val = i_to_val(i + 1);
        client.put_commit(k.clone(), val);
    }

    let region_id = client.get_region_id(b"xkey");

    wait_region_ready(&cluster, node_id, region_id);

    let snap = cluster
        .get_kvengine(node_id)
        .get_snap_access(region_id)
        .unwrap();
    // Scan only the target key range
    let mut upper_raw = k.clone();
    tidb_query_common::util::convert_to_prefix_next(&mut upper_raw);

    for filter_ts in [true, false] {
        let lower = Key::from_raw(&k);
        let upper = Key::from_raw(&upper_raw);
        // Scan delta from 0 to include all versions
        let mut scanner =
            CloudDeltaScanner::new(snap.clone(), 0, Some(lower.clone()), Some(upper), filter_ts);
        scanner.init().unwrap();

        let mut actual_seqs: HashMap<Vec<u8>, SeqAgg> = HashMap::new();
        while let Some(entry) = scanner.next_entry().unwrap() {
            add_entry_seq(&mut actual_seqs, &entry);
        }

        let _expected_raw = lower.into_raw().unwrap();
        verify_scan(&actual_seqs, &client.expected, true);
    }
}

#[test]
fn delta_scanner_versions_and_lock_on_same_key() {
    test_util::init_log_for_test();
    let cluster = ServerCluster::new(alloc_node_id_vec(1), |_, conf| {
        conf.raft_store.enable_inner_key_offset = false;
        conf.coprocessor.region_split_size = ReadableSize::gb(1);
    });
    let client = cluster.new_client();
    let mut client = RecordingClient::new(client);
    let node_id = cluster.get_nodes()[0];

    let k = i_to_key(200);

    // Write two committed versions
    for i in 0..2usize {
        let val = i_to_val(i + 1);
        client.put_commit(k.clone(), val);
    }

    // Create an uncommitted prewrite (lock) on the same key
    client.prewrite_put(k.clone(), i_to_val(99));

    let region_id = client.get_region_id(b"xkey");
    wait_region_ready(&cluster, node_id, region_id);

    let snap = cluster
        .get_kvengine(node_id)
        .get_snap_access(region_id)
        .unwrap();
    // Scan only the target key range
    let mut upper_raw = k.clone();
    tidb_query_common::util::convert_to_prefix_next(&mut upper_raw);

    for filter_ts in [true, false] {
        let lower = Key::from_raw(&k);
        let upper = Key::from_raw(&upper_raw);
        let mut scanner =
            CloudDeltaScanner::new(snap.clone(), 0, Some(lower.clone()), Some(upper), filter_ts);
        scanner.init().unwrap();

        let mut actual_seqs: HashMap<Vec<u8>, SeqAgg> = HashMap::new();
        while let Some(entry) = scanner.next_entry().unwrap() {
            add_entry_seq(&mut actual_seqs, &entry);
        }

        let _expected_raw = lower.into_raw().unwrap();
        verify_scan(&actual_seqs, &client.expected, true);
    }
}

#[test]
fn delta_scanner_mixed_keys_committed_and_uncommitted() {
    test_util::init_log_for_test();
    let cluster = ServerCluster::new(alloc_node_id_vec(1), |_, conf| {
        conf.raft_store.enable_inner_key_offset = false;
        conf.coprocessor.region_split_size = ReadableSize::gb(1);
    });
    let client = cluster.new_client();
    let mut client = RecordingClient::new(client);
    let node_id = cluster.get_nodes()[0];

    client.split_keyspace(THAT_MAGIC_KEYSPACE_ID);

    let k1 = i_to_key(300);
    let k2 = i_to_key(301);
    let k3 = i_to_key(302);

    // k1: committed
    client.put_commit(k1.clone(), i_to_val(1));

    // k2: prewrite but not commit
    client.prewrite_put(k2.clone(), i_to_val(2));

    // k3: two committed versions
    client.put_commit(k3.clone(), i_to_val(30));
    client.put_commit(k3.clone(), i_to_val(31));

    let region_id = client.get_region_id(b"xkey");
    wait_region_ready(&cluster, node_id, region_id);

    let snap = cluster
        .get_kvengine(node_id)
        .get_snap_access(region_id)
        .unwrap();
    // Scan only the k1..k3 range
    let mut upper_raw = k3.clone();
    tidb_query_common::util::convert_to_prefix_next(&mut upper_raw);

    for filter_ts in [true, false] {
        let lower = Key::from_raw(&k1);
        let upper = Key::from_raw(&upper_raw);
        let mut scanner =
            CloudDeltaScanner::new(snap.clone(), 0, Some(lower), Some(upper), filter_ts);
        scanner.init().unwrap();

        let mut actual_seqs: HashMap<Vec<u8>, SeqAgg> = HashMap::new();
        while let Some(entry) = scanner.next_entry().unwrap() {
            add_entry_seq(&mut actual_seqs, &entry);
        }

        verify_scan(&actual_seqs, &client.expected, true);
    }
}

#[test]
fn delta_scanner_filtered_by_start_ts() {
    let cluster = ServerCluster::new(alloc_node_id_vec(1), |_, conf| {
        conf.raft_store.enable_inner_key_offset = false;
        conf.coprocessor.region_split_size = ReadableSize::gb(1);
    });
    let client = cluster.new_client();
    let mut client = RecordingClient::new(client);
    let node_id = cluster.get_nodes()[0];

    let k = i_to_key(400);

    // Pre-barrier commits
    client.put_commit(k.clone(), i_to_val(1));
    client.put_commit(k.clone(), i_to_val(2));

    // Establish a barrier timestamp; entries before this should be filtered out.
    let barrier_ts = client.get_ts();

    // Post-barrier commit and a lock (prewrite)
    client.put_commit(k.clone(), i_to_val(3));
    client.prewrite_put(k.clone(), i_to_val(99));

    let region_id = client.get_region_id(b"xkey");
    wait_region_ready(&cluster, node_id, region_id);

    let snap = cluster
        .get_kvengine(node_id)
        .get_snap_access(region_id)
        .unwrap();

    // Scan only the target key
    let mut upper_raw = k.clone();
    tidb_query_common::util::convert_to_prefix_next(&mut upper_raw);

    for filter_ts in [true, false] {
        let lower = Key::from_raw(&k);
        let upper = Key::from_raw(&upper_raw);

        // Use from_ts == barrier_ts to filter out the first two commits.
        let mut scanner = CloudDeltaScanner::new(
            snap.clone(),
            barrier_ts.into_inner(),
            Some(lower.clone()),
            Some(upper),
            filter_ts,
        );
        scanner.init().unwrap();

        let mut actual_seqs: HashMap<Vec<u8>, SeqAgg> = HashMap::new();
        while let Some(entry) = scanner.next_entry().unwrap() {
            add_entry_seq(&mut actual_seqs, &entry);
        }

        // Expected contains only the post-barrier writes: commit=3 and prewrite=99.
        let mut expected: HashMap<Vec<u8>, EntryAgg> = HashMap::new();
        let key_norm = normalize_key(&lower.into_raw().unwrap());
        let e = expected.entry(key_norm).or_default();
        e.commit_vals.push(i_to_val(3));
        e.prewrite_vals.push(i_to_val(99));
        // For filtered-by-ts, sequence/old_value may refer to filtered-out versions;
        // only set check.
        verify_scan(&actual_seqs, &expected, false);
    }
}

fn randomized_writes_and_verify(key_space: usize) {
    // Separate helper so we can run multiple sizes in a single test.
    let cluster = ServerCluster::new(alloc_node_id_vec(1), |_, conf| {
        conf.raft_store.enable_inner_key_offset = false;
        conf.coprocessor.region_split_size = ReadableSize::gb(1);
    });
    let client = cluster.new_client();
    let mut client = RecordingClient::new(client);
    let node_id = cluster.get_nodes()[0];

    // Plan: perform ~1000 writes: 80% commits first, then 20% prewrites to avoid
    // conflicts from writing after placing locks.
    let total_writes = 1000usize;
    let commits = total_writes * 4 / 5; // 800
    let prewrites = total_writes - commits; // 200

    let seed: u64 = rand::random();
    println!("randomized_writes_and_verify({}, seed={})", key_space, seed);
    let mut rng = StdRng::seed_from_u64(seed);

    // Track keys used to compute scan bounds and avoid duplicate prewrite locks.
    let mut touched_keys: Vec<Vec<u8>> = Vec::with_capacity(total_writes);
    let mut prewrite_keys: HashSet<Vec<u8>> = HashSet::new();

    // First, commits (mix puts and deletes).
    for i in 0..commits {
        let idx = rng.gen_range(0..key_space);
        let k = i_to_key(10_000 + idx); // offset to avoid overlap with other tests
        touched_keys.push(k.clone());
        // 30% chance to issue a delete commit, 70% put commit
        if rng.gen_bool(0.3) {
            client.delete_commit(k);
        } else {
            let v = i_to_val(10_000 + i);
            client.put_commit(k, v);
        }
    }

    // Then, prewrites on distinct keys to avoid lock conflicts.
    // Don't write too prewrite keys.
    let mut created = 0usize;
    while created < prewrites && prewrite_keys.len() < (key_space / 3 * 2) {
        let idx = rng.gen_range(0..key_space);
        let k = i_to_key(10_000 + idx);
        if prewrite_keys.insert(k.clone()) {
            let v = i_to_val(90_000 + created);
            touched_keys.push(k.clone());
            client.prewrite_put(k, v);
            created += 1;
        }
    }

    // Compute scan bounds over the touched keys.
    let mut min_key = None::<Vec<u8>>;
    let mut max_key = None::<Vec<u8>>;
    for k in &touched_keys {
        if min_key.as_ref().map_or(true, |mk| k < mk) {
            min_key = Some(k.clone());
        }
        if max_key.as_ref().map_or(true, |mk| k > mk) {
            max_key = Some(k.clone());
        }
    }
    let min_key = min_key.expect("at least one key touched");
    let mut upper_raw = max_key.expect("at least one key touched");
    tidb_query_common::util::convert_to_prefix_next(&mut upper_raw);

    let region_id = client.get_region_id(b"xkey");
    wait_region_ready(&cluster, node_id, region_id);

    let snap = cluster
        .get_kvengine(node_id)
        .get_snap_access(region_id)
        .unwrap();

    for filter_ts in [true, false] {
        let lower = Key::from_raw(&min_key);
        let upper = Key::from_raw(&upper_raw);
        let mut scanner =
            CloudDeltaScanner::new(snap.clone(), 0, Some(lower), Some(upper), filter_ts);
        scanner.init().unwrap();

        // Collect results using SeqAgg and verify both set-equality and
        // sequence/old_value.
        let mut actual_seqs: HashMap<Vec<u8>, SeqAgg> = HashMap::new();
        while let Some(entry) = scanner.next_entry().unwrap() {
            add_entry_seq(&mut actual_seqs, &entry);
        }

        verify_scan(&actual_seqs, &client.expected, true);
    }
}

#[test]
fn delta_scanner_randomized_writes_10() {
    randomized_writes_and_verify(10);
}

#[test]
fn delta_scanner_randomized_writes_1000() {
    randomized_writes_and_verify(1000);
}

#[test]
fn delta_scanner_randomized_writes_100000() {
    randomized_writes_and_verify(100000);
}

#[test]
fn delta_scanner_filter_pessimistic_locks() {
    test_util::init_log_for_test();
    let cluster = ServerCluster::new(alloc_node_id_vec(1), |_, conf| {
        conf.raft_store.enable_inner_key_offset = false;
        conf.coprocessor.region_split_size = ReadableSize::gb(1);
    });
    let client = cluster.new_client();
    let mut client = RecordingClient::new(client);
    let node_id = cluster.get_nodes()[0];

    let k1 = i_to_key(500);
    let k2 = i_to_key(501);

    // k1: committed value + pessimistic lock (lock should be filtered)
    client.put_commit(k1.clone(), i_to_val(1));
    client.acquire_pessimistic_lock(k1.clone());

    // k2: only pessimistic lock (should be completely filtered out)
    client.acquire_pessimistic_lock(k2.clone());

    let region_id = client.get_region_id(b"xkey");
    wait_region_ready(&cluster, node_id, region_id);

    let snap = cluster
        .get_kvengine(node_id)
        .get_snap_access(region_id)
        .unwrap();
    // Scan the range covering all keys
    let mut upper_raw = k2.clone();
    tidb_query_common::util::convert_to_prefix_next(&mut upper_raw);

    for filter_ts in [true, false] {
        let lower = Key::from_raw(&k1);
        let upper = Key::from_raw(&upper_raw);
        let mut scanner =
            CloudDeltaScanner::new(snap.clone(), 0, Some(lower), Some(upper), filter_ts);
        scanner.init().unwrap();

        let mut actual_seqs: HashMap<Vec<u8>, SeqAgg> = HashMap::new();
        while let Some(entry) = scanner.next_entry().unwrap() {
            add_entry_seq(&mut actual_seqs, &entry);
        }

        // Verify that pessimistic locks are filtered out
        // k1 should have only the committed value, no lock
        // k2 should not appear at all (only had pessimistic lock)
        verify_scan(&actual_seqs, &client.expected, false);
    }
}

#[test]
fn delta_scanner_filter_lock_type_locks() {
    test_util::init_log_for_test();
    let cluster = ServerCluster::new(alloc_node_id_vec(1), |_, conf| {
        conf.raft_store.enable_inner_key_offset = false;
        conf.coprocessor.region_split_size = ReadableSize::gb(1);
    });
    let client = cluster.new_client();
    let mut client = RecordingClient::new(client);
    let node_id = cluster.get_nodes()[0];

    let k1 = i_to_key(600);
    let k2 = i_to_key(601);
    let k3 = i_to_key(602);

    // k1: committed value + Lock-type prewrite (lock should be filtered)
    client.put_commit(k1.clone(), i_to_val(1));
    client.prewrite_lock(k1.clone());

    // k2: only Lock-type prewrite (should be completely filtered out)
    client.prewrite_lock(k2.clone());

    // k3: Lock-type prewrite + committed value (commit should appear, lock
    // filtered)
    client.prewrite_lock(k3.clone());
    client.put_commit(k3.clone(), i_to_val(30));

    let region_id = client.get_region_id(b"xkey");
    wait_region_ready(&cluster, node_id, region_id);

    let snap = cluster
        .get_kvengine(node_id)
        .get_snap_access(region_id)
        .unwrap();
    // Scan the range covering all keys
    let mut upper_raw = k3.clone();
    tidb_query_common::util::convert_to_prefix_next(&mut upper_raw);

    for filter_ts in [true, false] {
        let lower = Key::from_raw(&k1);
        let upper = Key::from_raw(&upper_raw);
        let mut scanner =
            CloudDeltaScanner::new(snap.clone(), 0, Some(lower), Some(upper), filter_ts);
        scanner.init().unwrap();

        let mut actual_seqs: HashMap<Vec<u8>, SeqAgg> = HashMap::new();
        while let Some(entry) = scanner.next_entry().unwrap() {
            add_entry_seq(&mut actual_seqs, &entry);
        }

        // Verify that Lock-type locks are filtered out
        // k1: only committed value
        // k2: should not appear (only Lock-type lock)
        // k3: only committed value
        verify_scan(&actual_seqs, &client.expected, false);
    }
}
