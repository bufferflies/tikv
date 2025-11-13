# cse-ctl accessibility

## CSE-CTL

`cse-ctl` is the command-line management tool for Cloud Storage Engine, used for troubleshooting and emergency recovery operations.

### Available Troubleshooting Functions

#### 1. Unsafe Recover (`unsafe-recover`)

Emergency operations for Raft replica loss. Directly modifies Raft Engine data.

- Destroy specific regions
- Create empty regions
- Remove peers from specific stores
- Batch operations on keyspace/table regions
- **Warning:** Always backup before use

#### 2. SST File Management (`show sst`, `show scan-bad-table-file`)

- View SST file information
- Scan for corrupted SST files
- Detect damaged files in storage engine

#### 3. MVCC Inspection (`mvcc`)

- Inspect key version history
- Debug MVCC version issues
- View key's MVCC information with timestamps

#### 4. Data Consistency Check

- **Check Table** (`check-table`): Verify data consistency across replicas
- **Resolve Locks** (`resolve-lock`): Clean up stale locks from failed transactions

#### 5. Recovery Mode (`recovery`)

- Enter/exit emergency recovery mode
- Check recovery status
- Used when cluster is in inconsistent state

#### 6. Region Management (`region`)

- **Merge**: Reduce region count after data deletion
- **Split**: Split large regions for better load distribution

#### 7. Transaction Log Inspection (`show txn-chunk`, `show txn-log`)

- View transaction chunks
- Inspect transaction logs with timestamp range
- Debug transaction-related issues

#### 8. HTTP Debug (`http`)

- Send HTTP requests to TiKV stores
- Collect metrics and debug information

**Available HTTP APIs:**

**Monitoring & Status:**

- `GET /metrics` - Prometheus metrics
- `GET /status` - Server status
- `GET /ready` - Readiness check
- `GET /config` - Get current configuration
- `POST /config` - Update configuration

**Profiling & Debug:**

- `GET /debug/pprof/profile` - CPU profiling
- `GET /debug/pprof/heap` - Heap profiling
- `GET /debug/pprof/heap_list` - List heap profiles (deprecated)
- `GET /debug/pprof/heap_activate` - Activate heap profiling (deprecated)
- `GET /debug/pprof/heap_deactivate` - Deactivate heap profiling (deprecated)
- `GET /debug/pprof/cmdline` - Command line arguments
- `GET /debug/pprof/symbol` - Symbol lookup
- `POST /debug/pprof/symbol` - Symbol resolution
- `GET /debug/fail_point` - Fail point status
- `PUT /fail/<name>` - Configure fail point
- `DELETE /fail/<name>` - Remove fail point
- `GET /fail` - List all fail points

**Region & Metadata:**

- `GET /region/<region_id>` - Get region metadata
- `GET /sync_region` - Sync region information
- `GET /sync_region_by_id` - Sync region by ID
- `PUT /log-level` - Change log level

**KV Engine:**

- `GET /kvengine/snapshot/<id>` - Get snapshot info
- `GET /kvengine/columnar_status` - Columnar storage status
- `GET /kvengine/meta/<shard_id>/<shard_ver>` - Get shard metadata
- `GET /kvengine/keyspace` - Keyspace information
- `GET /kvengine/all` - All shards information
- `GET /kvengine/active_lite` - Active lite shards
- `GET /kvengine/files` - File information
- `GET /kvengine/compactor` - Compactor status
- `POST /kvengine/compactor` - Trigger compaction

**Raft Engine:**

- `GET /rfengine/wal_chunk` - WAL chunk information
- `POST /rfengine/backup` - Trigger backup

**Operations:**

- `POST /restore-shard` - Restore shard
- `POST /ingest_files` - Ingest files
- `POST /ingest_s3` - Ingest from S3
- `POST /unsafe_recover` - Unsafe recovery operations
- `POST /major-compact` - Major compaction
- `POST /schema_file` - Schema file operations
- `POST /clear_columnar` - Clear columnar data
- `POST /build_columnar` - Build columnar data
- `GET /dfs/` - DFS operations (read)
- `POST /dfs/` - DFS operations (write)
- `GET/POST /recovery/` - Recovery mode operations

## TiKV Malfunction

# KvEngine File Damage

SST files may become corrupted due to hardware failures, software bugs, or unexpected shutdowns. Corrupted SST files can lead to data loss or unavailability.

## SST Corruption

- Use `cse-ctl show scan-bad-table-file` to scan for corrupted SST files on local disk.
But looks like this command can only detect the file meta checksum mismatch, not the data checksum mismatch.


## Blob/Txn File Corruption

As we don't enable blob file and txn file by default, we do not consider these errors now.


- Use `GET /kvengine/files` HTTP API to check file status
- Use `cse-ctl show scan-bad-table-file` to scan for corrupted SST files
- Monitor logs for "checksum mismatch" errors

**Recovery Actions:**

1. If corruption is detected in DFS files:
   - Files will be re-downloaded from S3 automatically
   - Check if the file exists in S3 with correct checksum

2. If corruption persists:
   - Use `POST /kvengine/compactor` to trigger compaction
   - Compaction will rebuild SST files from valid data sources

3. For severe corruption:
   - Check hardware health (disk, memory)
   - Review system logs for I/O errors
   - Consider restoring from backup if data loss is unacceptable

# RaftEngine File Damage


## Loss of the majority of replicas

Unsafe recovery is required when a region loses the majority of its replicas, making it impossible to reach consensus and serve requests.



## Replication Inconsistency

## MVCC Inconsistency

## Disk Full Recovery

### Manual Compaction

## Reference

<https://pingcap.feishu.cn/wiki/CuUBwVQKIitbbtkvGMjclPsFn2b>


### GC not in time

Garbage collection (GC) not occurring in a timely manner can lead to increased storage usage and potential performance degradation in TiKV. 

