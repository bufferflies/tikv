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

### Client Side

The client-side code required for restoring a keyspace resides in [`native_br::restore_keyspace`](https://github.com/tidbcloud/cloud-storage-engine/blob/cloud-engine/components/native_br/src/restore_keyspace.rs).

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
