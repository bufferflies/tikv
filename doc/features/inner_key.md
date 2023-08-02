# Shard Inner Key in CSE

## Background

In CSE, the data may be shared by multi-tenant. In this circumstance, the data must be copy multiple times to every tenant. This will cause the storage cost increase and inefficient data copy, the time cost linearly increases with the size of the dataset. To solve this problem, we introduce the shard `Inner Key` to CSE. Using `Inner Key`, the data shared between tenants are easy and efficient. The data branching can be done in one minute with forking TB of data.

## Implementation

```proto
message Snapshot {
  bytes outer_start = 1;
  bytes outer_end = 2;
  Properties properties = 3;
  repeated L0Create l0Creates = 5;
  repeated TableCreate tableCreates = 6;
  uint64 baseVersion = 7;
  uint64 data_sequence = 8;
  repeated BlobCreate BlobCreates = 9;
  uint64 max_ts = 10;
  uint32 inner_key_off = 11;  // new added
}
```

We add a new field `inner_key_off` to Snapshot proto to record the offset of `InnerKey` in the snapshot. If `inner_key_off` is 0, it means the `InnerKey` is not enabled for this shard. If `inner_key_off` is 4, it means the `InnerKey` is enabled for this shard and the offset of `InnerKey` is 4 bytes. The key format details please refer to doc [APIv2](https://github.com/tikv/rfcs/blob/master/text/0069-api-v2.md).

`ShardRange` is a struct to record the shard range of a shard. It contains the `start_key`, `end_key` and `inner_key_off` of the shard.

```rust
#[derive(Clone, Default)]
pub struct ShardRange {
    pub outer_start: Bytes,
    pub outer_end: Bytes,
    pub inner_key_off: usize,
}
```

`ShardRange` is constructed in two ways. One is using `ShardRange::from_snap()` to construct from a snapshot which is for replica shard. See [ShardMeta::new()](https://github.com/tidbcloud/cloud-storage-engine/blob/66e20566f8bdf378c69df39afec4dbf6019b188a/components/kvengine/src/meta.rs#L49)，[Shard::new_for_ingest()](https://github.com/tidbcloud/cloud-storage-engine/blob/9a9541294d39c548861df444dff6e1444104de85/components/kvengine/src/shard.rs#L178)， [EngineCore::apply_restore_shard()](https://github.com/tidbcloud/cloud-storage-engine/blob/66e20566f8bdf378c69df39afec4dbf6019b188a/components/kvengine/src/apply.rs#L722).
The other way is created by `ShardRange::new()` duiring the region split. See  [ShardMeta::new_split()](https://github.com/tidbcloud/cloud-storage-engine/blob/66e20566f8bdf378c69df39afec4dbf6019b188a/components/kvengine/src/meta.rs#L89)，[Engine::split()](https://github.com/tidbcloud/cloud-storage-engine/blob/9a9541294d39c548861df444dff6e1444104de85/components/kvengine/src/split.rs#L42).

The `InnerKey` feature depends on whether the parent shard `InnerKey` is enabled and the configure `enable_inner_key_offset=true` in the server side. After enabling the `enable_inner_key_offset` configure, the new keyspace will be created with `InnerKey` feature enabled. The new shard in keyspace depends on the parent shard. If the parent shard `InnerKey` is enabled, the new shard will be created with `InnerKey` enabled. If the parent shard `InnerKey` is disabled, the new shard will be created with `InnerKey` disabled. 

```rust
let range = if self.core.opts.enable_inner_key_offset
    && is_whole_keyspace_range(start_key, end_key)
{
    ShardRange::new(start_key, end_key, KEYSPACE_PREFIX_LEN)
} else {
    ShardRange::new(start_key, end_key, old_shard.inner_key_off)
};
```

In this way, we can make sure the `InnerKey` feature can be enabled smoothly in the whole cluster and not affect the old existing keyspace data.

The shards data are persisted to SST files during compaction. In compaction the `InnerKey` will be used during save to SST files.  

```rust
impl CompactionRequest {
    pub fn inner_start(&self) -> InnerKey<'_> {
        InnerKey::from_outer_key(&self.outer_start, self.inner_key_off)
    }

    pub fn inner_end(&self) -> InnerKey<'_> {
        InnerKey::from_outer_end_key(&self.outer_end, self.inner_key_off)
    }
}
```

`InnerKey` is constructed from the outer key and `inner_key_off`. It simply slice the outer key with `inner_key_off` to get the inner key. 

```rust
pub fn from_outer_key<'b: 'a>(outer_key: &'b [u8], inner_offset: usize) -> Self {
    let key = &outer_key[inner_offset..];
    Self { key }
}
```

In SST files there is no keyspace prefix related information. When reading the SST files, we need to prepend the keyspace prefix to the `InnerKey` to get the outer key and return to the upper layer. So the `InnerKey` is transparent to the upper layer. 


## Data Branching

The data branching feature is mainly depends on Native BR, more details please refer doc [Native BR](https://github.com/tidbcloud/cloud-storage-engine/blob/24fecee17ce09626feca0d0e4fda4838fb6000ba/doc/features/native_br.md#L1).

The major difference is the target keyspace. In data branching restore, the target keyspace is the newly created one. And the `InnerKey` of source keyspace must be enabled or an error will be returned. The main procedure is as follows:

1. User triggers keyspace branching.
2. Infra creates a new TiDB instance, bind with a new empty keyspace taken from the keyspace pool.
3. Call tikv-worker backup api to generate the whole cluster backup.
4. Call tikv-worker restore api to restore the meta to the new keyspace.
	1. Filter out the source keyspace related shards from the whole cluster backup.
	2. Create a local cluster to manipulate the cluster meta.
	3. Flush shard WAL to kvengine.
	4. Check source keyspace if inner key enabled, return error if not.
	5. Split target regions same as the source regions and replace the shard meta keyspace prefix to target keyspace.
	6. Restore shards meta to TiKV.
5. Return success to user.

## More works

1. In data branching, the shards placement of target keyspace is not considered. So the same SST file may be copied to different nodes if there are more than 3 TiKV servers. The placement rule can be added to make corresponding shard of source and target keyspace be placed in the same node to avoid redundant SST file downloads.
2. How to enable the `InnerKey` feature to the old keyspaces smoothly. 

