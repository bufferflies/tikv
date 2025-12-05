# Lightweight backup

## Background

In production environment, cluster backup operations are triggered in two ways: periodic backups and real-time backups during keyspace restores. Although incremental backups are supported to minimize backup costs, the large volume of writes in the cluster often results in most backups falling back to full backups. Considering the costs and the frequency of real-time backups, there is a requirement to optimize the cluster native backup and restore process.

## Lightweight Backups

To make backups lightweight and avoid backing up WAL (Write-Ahead Log) and raft logs every time, which can lead to large backup sizes, historical epoch WAL files are persisted to S3. This means that each backup operation only needs to take a snapshot of the WAL files and save the corresponding WAL epoch and file_offset as metadata.

To speed up the replay process, periodic snapshots of `rfengine` are taken, and during recovery, replay starts directly from the latest snapshot to the backup point. To simplify the snapshot handling logic, snapshots are taken immediately after `rfengine` manifest be rewritten. At this point, the WAL of the epoch is compacted to the raft log. So we can only save the raft logs up to the current epoch as a snapshot of this epoch.

### WAL Persistence to S3

The persistence of WAL to S3 is handled by a separate thread called `dfs worker` to avoid worker thread interference caused by S3 failures or access delays. When an async WAL writes a new `write batch`, the worker thread sends an `ObjectStorageTask::Sync { epoch_id, file_off }` task to `dfs worker`. Upon receiving the Sync task, the `dfs worker` thread reads the corresponding epoch's WAL file up to the specified file offset, maintaining a WAL chunk in memory. When the WAL chunk reaches a specified size, `dfs worker` flushes it to S3 and clears WAL chunk in memory. The naming format of WAL chunks in S3 is `e{epoch_id}_{start_offset}_{end_offset}`, e.g., `e00000001_0000000000000000_000000000001f000.wal`. If S3 fails to upload, `dfs worker` will enter the `unhealthy` mode, and the lightweight backup will be disabled until manual handling the issue.

### WAL Chunk Compression

The WAL files in `rfengine` are stored in a 4k-aligned format with padding, and historical WALs are saved for at least 90 days. To save storage costs, WAL chunks are compressed with LZ4 before being uploaded to S3. Compression ratios have been verified up to `5~10`.

### Backup

Backup operations are performed in the `rfengine` main thread, where the WAL writer's epoch_id and file offset are obtained as backup metadata. Since this operation is lightweight, it is performed synchronously.

### Keyspace Restore

Keyspace restore involves creating a `BackupCluster` from the backup to manipulate the data. The main difference between lightweight and full/incremental backups is the `setup_raft_engine` process during `BackupCluster` initialization. For each store, the system first attempts to find the latest raft logs snapshot earlier than the backup metadata epoch. If a snapshot exists, it retrieves the raft logs file and manifest file from the store backup metadata on S3. Then, it replays the WAL chunks sequentially for epochs after the snapshot or starts from `1` if no snapshot available. Since the WAL chunks on S3 are compressed, each WAL chunk for the replayed epoch must be decompressed, assembled into a complete WAL file, and then replayed into the `rfengine`. It's important to note that WAL chunks are uploaded asynchronously, so during restoration, the last WAL chunk might not have been uploaded to S3 yet. In such cases, the missing data is directly retrieved from the store through `HTTP API` and replayed into the `rfengine`.

Once `rfengine` is fully restored, subsequent processing logic is same as incremental backups or full backups.

### Full Cluster Restore

Lightweight backups also support full cluster restores. In both full and incremental restore scenarios, cluster restoration is performed by downloading `rfengine` data files from S3 to the local machine. In the case of lightweight backups, WAL chunks are replayed from the snapshot epoch to generate complete `rfengine` storage files locally. At last, the real store are bootstrapped with the data files.

Considering that WAL chunks in S3 are persistent and cannot be deleted, conflicts arise when a cluster is restored using data from a previous epoch. The WAL chunks from the prior data conflict with the new data after restoration, leading to corruption of WAL chunks from that epoch or later epochs. This corruption renders lightweight backups invalid and unrecoverable.

To prevent this, we establish a new store ID during full cluster restoration. This new store ID is generated using the alloc_id from the latest cluster backup metadata in S3:

```
new_store_id = alloc_id in latest backup meta + store_id ordered index in backup meta + 1 + delta
```

The `delta` is typically set to 0 by default. However, if conflicts arise with existing store IDs in S3, we should increase delta to a positive number to avoid these conflicts.

After replaying WAL chunks, we iterate through all peers’ region local state in `rfengine` and update the store ID of peers to this new store ID. We also need to update the ident of the store itself in rfengine.

Special attention should be paid to the delta value when restoring PD data. The delta value used in PD restoration is crucial for restoring the alloc_id. Therefore, we need to specify a delta value in PD restore that is either the same or larger than the one used in restore TiKV.

The `wal-target-size` argument is set by default to `512MB`, which matches the `target-file-size` in the `rfengine` configuration. It’s important to adjust the `wal-target-size` whenever the config `target-file-size` in `rfengine` is altered, to ensure they remain consistent.

### Rebuilding WAL Persistence State

WAL chunks are maintained in memory and are uploaded to S3 once they reach a certain size. Or  `dfs worker` will upload the WAL chunk in memory to S3 during `rfengine` gracefully shutting down. However, in cases of abnormal `rfengine` shutdown or crashes, the data in the memory-maintained WAL chunks will be lost. When `rfengine` is restarted, it searches for the last synchronized WAL epoch and offset from S3, then initializes `dfs worker`. When `dfs worker` processes `ObjectStorageTask::Sync` tasks, it verifies if the epoch matches. If not, it attempts to read earlier epoch WALs from S3 to rebuild the WAL chunk state.

## Enable Lightweight Backup

### Server Side

To enable lightweight backup feature in server side, we should set the following configuration items:

```toml
[rfengine]
# wal-sync-dir must be set.
wal-sync-dir = "/data/data/tikv-20160/raft-wal"

# lightweight-backup must be set to true.
lightweight-backup = true

# target-file-size is the size of WAL file. In production, it is recommended to set it to 512MB.
target-file-size = "512MB"

# wal-chunk-target-file-size is the size of WAL chunk before compression. After lz4 compression, the actual size is expected to 10%~20% of this value.
wal-chunk-target-file-size = "128MB"
```

### TiKV API Server Side

To enable lightweight backup when doing instant backup during restore, we should set the following configuration items:

```toml
[native-br]
enable_lightweight_backup = true
```

### Client Side

To enable lightweight backup in periodic backup we should add `--lightweight` to the backup command, such as:

```bash
cse-ctl backup --pd 127.0.0.1:2379 --config tikv-worker.toml --lightweight
```