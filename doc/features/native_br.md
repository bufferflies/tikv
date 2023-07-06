# Native Backup and Restore Functionality

This document outlines the critical code path associated with the **Native Backup and Restore** feature, described in the sequence of processing procedure.

## Backup

Compared to traditional backup methods, **Native Backup** is a cloud-native solution that backups the WAL (Write Ahead Logs), Raft logs, and metadata only. All other data has been stored in S3.

### Frontend

The backup process can be initiated through the following ways:

- Manual backup through the command line interface of [cse-ctl](https://github.com/tidbcloud/cloud-storage-engine/tree/cloud-engine/cmd/cse-ctl) (see `cse-ctl::execute_full_backup`).

- Automatic backup through the same command line interface of [cse-ctl](https://github.com/tidbcloud/cloud-storage-engine/tree/cloud-engine/cmd/cse-ctl) with `incremental` and `interval` argument (see `cse-ctl::execute_incremental_backup`), which runs periodically via a `cse-ctl` binary deployed in the [`cse-backup`](https://github.com/tidbcloud/aws-shared-cd/blob/main/pool/eks-freetier/manifests/serverless/backup.yaml) node.

- On-demand backup via the HTTP interface of [tikv-worker](https://github.com/tidbcloud/cloud-storage-engine/tree/cloud-engine/cmd/tikv-worker) for **PiTR (Point-in-Time Restore)** and **Data Branching** feature (see `tikv-worker::native_br`), when the required restore time is later than the latest automatic backup.

### Client Side

The client-side code required for backup of the entire cluster resides in [`native_br::backup`](https://github.com/tidbcloud/cloud-storage-engine/blob/cloud-engine/components/native_br/src/backup.rs).

In case of automatic backup, **Incremental Backup** can be utilized (see `native_br::execute_incremental_backup`) to reduce cost (see `rfengine::Worker::incremental_backup`).

On the successful completion of each backup, a **GC Service Safepoint** named **native_br** is set for automatic backups to ensure that every timestamp of the next backup is available for **PiTR** and **Data Branching**.

Client-side requests all TiKV stores to perform backup at the store level (an optional argument `tolerate_err` can be set to `1` to tolerate no more than one unavailable TiKV store).

Once all TiKV stores return success, the client-side saves the backup meta (see `rfenginepb::ClusterBackupMeta`) to S3.

### Server Side

1. CSE provides a status server interface `/rfengine/backup` that accepts backup requests (refer to `StatusServer::backup_rfengine`).

2. Upon receiving a request from the `/rfengine/backup` interface, the status service invokes `RfEngineCore::backup` method of `rfengine`.

3. `rfengine` sends a `rfengine::Task::Backup` message to worker.

4. For a full backup request, the worker packs the WAL, manifest (peer metadata), and Raft logs to S3 (see `rfengine::Worker::full_backup`).

5. For an incremental backup request, if incremental backup is available (when current WAL `epoch` is the same with last backup), the worker packs the new WAL since last backup only (see `rfengine::Worker::incremental_backup`). Otherwise, fall back to full backup.

> **Note:** When `wal_sync_dir` (see `rfengine::Config::wal_sync_dir`) is enabled, WAL is read from files written by asynchronous writer (see `rfengine::RfEngineCore::persist` and `rfengine::Task::Write`).

6. Finally, returns the `rfenginepb::StoreBackupMeta` to the client-side. 

## Restore Keyspace

### Frontend

The restore process can be initiated either through the [cse-ctl](https://github.com/tidbcloud/cloud-storage-engine/tree/cloud-engine/cmd/cse-ctl) command line interface (`cse-ctl::execute_restore_command`), or the [tikv-worker](https://github.com/tidbcloud/cloud-storage-engine/tree/cloud-engine/cmd/tikv-worker) HTTP interface (`tikv-worker::native_br`).

#### Instant Backup

Before performing any restore operations, the frontend initiates an **instant backup** for the following reasons:

1. For Point-in-Time Recovery (PiTR) beyond the latest backup. Since there is no backup that contains data for such a specific point in time, we need to restore from the instant backup.

2. Consider the following scenario:
   - At time `T0`, backup `B0` is created.
   - At time `T1`, data `D` is written.
   - At time `T2`, we perform snapshot restore to `B0`.
   - At time `T3`, backup `B1` is created.
   - At time `T4`, we restore to a point in time `Tr`, where `T1 < Tr < T2`. We expect data `D` to be restored, but it is not. The reason is that after restoring to `B0` at `T2`, data `D` was no longer available and was not backed up by `B1`. In fact, data `D` is not included in any backup and can never be recovered.

   To resolve this issue, we must perform an instant backup, say `Bi`, at `T2` before restoring to `B0`. This way, when we restore to the point in time `Tr` (`T1 < Tr < T2`), we can retrieve data `D` from `Bi`.

In addition, it is important to note that instant backups can be expensive if there are frequent restore operations. Therefore, we have a frequency limitation that allows for a maximum of one instant backup per minute. Restore request will need to wait less than a minute for next instant backup is created.

Additionally, it is important to be aware that frequent restore operations can make instant backups costly. To address this, we have imposed a frequency limitation, allowing for a maximum of one instant backup per minute. Restore requests need to wait for less than a minute for the next instant backup to be created.

See `native_br::backup_worker`.

#### Progress Report

To provide users with restore progress updates, the frontend will generate a progress reporter and pass it to the client-side. During each step of the process, the client-side will invoke the progress reporter's callback function, allowing the frontend to obtain and display the latest progress to users.

Moreover, the frontend will calculate a completion percentage, based on an empirical estimation for the duration of each step.

See `tikv_worker::native_br::RestoreProgressReporter`.

### Client Side

The client-side code required for restoring a keyspace resides in [`native_br::restore_keyspace`](https://github.com/tidbcloud/cloud-storage-engine/blob/cloud-engine/components/native_br/src/restore_keyspace.rs).

Process steps for client-side:

1. **Remove TiFlash Replicas**: Since TiFlash does not support the operation of restoring shards, we first remove TiFlash replicas before the restore process. After the restore is complete, we add them back and synchronize the data to TiFlash using Raft snapshot. See `tiflash::remove_tiflash_replica_of_keyspace`.

2. **Load Backup Meta**: We load the backup meta from S3. This meta contains information about shard metas, WAL (Write-Ahead Log), and Raft logs. See `restore::get_cluster_backup_meta`.

3. **Extract Backup Shards**: In this step, we replay the WAL to update the shard metas and append Raft logs. We load the shards from the shard meta and insert them into the `kvengine`. See `restore_keyspace::BackupCluster::new`.

   - Note:
     - Sometimes, the `commit_index` of a peer in the `rfengine` meta may not be the latest. To infer the real `commit_index`, we use the Raft log `last_index` of a quorum of peers. See `restore_keyspace::get_quorum_last_index`.
     - Raft logs need to be preprocessed when `last_preprocessed_index < commit_index`. See `restore_keyspace::BackupCluster::preprocess_shard`.

4. **Flush Shards**: Since the server-side retrieves data for restore from S3, we need to flush the `mem-table` to SSTable files. See `restore_keyspace::BackupCluster::flush_shards`.

5. **Truncate Ts**: To ensure a consistent snapshot, we truncate data with a timestamp larger than `backup_ts` (or `restore_ts` of PiTR). See `restore_keyspace::BackupCluster::truncate_ts`.

6. **Split Regions**: Split regions when restoring to a new keyspace (a feature of `data branching`). As the new keyspace is empty, it is more efficient to pre-split the online regions before the restore, compared to splitting the regions with data after the restore on the server-side. See `restore_keyspace::split_target_keyspace`.

7. **Align Regions and Gather SSTables**: In this step, we find the overlapping backup shards for each online region and gather the corresponding SSTables. See `restore_keyspace::BackupCluster::align_target_regions` and `restore_keyspace::BackupCluster::gather_sstables`.

8. **Generate Snapshots and Request to Server Side**: Lastly, we generate shard snapshot metas from the SSTables obtained in the previous step and send it to the `/restore-shard` interface on the server-side for further processing. See `restore_keyspace::BackupCluster::generate_snapshots` and `restore_keyspace::restore_snapshots`.

### Server Side

> **Note:** Restoring a keyspace on the server-side is performed at the region level and is termed **restore shard**.

#### Request Reception

1. CSE provides a status service interface `/restore-shard` that accepts restore requests (refer to `StatusServer::restore_shard`).

2. Upon receiving a request from the `/restore-shard` interface, the status service sends `send_causal_msg` with `CasualMessage::RestoreShard` to `rfstore`.

3. `PeerFsm` handles `CasualMessage::RestoreShard` in `PeerMsgHandler::on_restore_shard`, validates the request, and then proposes a Raft command with a changeset containing the field `ChangeSet::restore_shard`.

#### On Raft Committed

1. After the Raft command is committed, the changeset is preprocessed through `Peer::preprocess_restore_shard`. This includes operations such as updating the `data_sequence` according to the Raft index.

#### On Apply

1. The changeset is applied to `ShardMeta` through `ShardMeta::apply_restore_shard`. Next, `Peer` sends an `ApplyMsg::PrepareChangeSet` message to `Applier`.

2. `Applier` clears relevant caches including the lock cache, and pauses the apply of `custom_log` in `Applier::handle_prepare_restore_shard` when it receives the `ApplyMsg::PrepareChangeSet`. Then, a thread is spawned to prepare the changeset.

3. Tables are loaded to prepare the changeset for apply in `EngineCore::prepare_change_set`.

4. The changeset is then applied to `kvengine` through `EngineCore::apply_restore_shard`.

#### Post-Apply

1. After the changeset is applied, the apply of `custom_log` is resumed, the shard version of `Applier` is updated in `Applier::handle_apply_change_set_result`, and the apply result is sent through `ExecResult::RestoreShard` using `PeerMsg::ApplyResult`.

2. After `custom_log` is applied (without any changes to `mem-table`), the request callback is triggered to notify the client-side that the shard has been restored (in `Applier::handle_apply_result`).

3. `ExecResult::RestoreShard` is processed in `StoreMsgHandle::on_restore_shard_result`, which modifies the local region version and notifies PD about the new region version.

#### Response Return

On receiving callback of restore shard, the status service calculate the `kv_size` of restored shard, and return to the client-side for billing purposes. See `cloud_server::status_server::RestoreShardResponse`.
