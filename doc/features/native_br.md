# Native Backup and Restore Functionality

This document outlines the critical code path associated with the **Native Backup and Restore** feature, described in the sequence of processing procedure.

## Backup

*todo: add details about the backup process*

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

1. The changeset is applied to `ShardMeta` through `ShardMeta::apply_restore_shard`. The apply of `custom_log` is then paused, and a thread is spawned to prepare the changeset.

2. Tables are loaded to prepare the changeset for apply in `EngineCore::prepare_change_set`.

3. The changeset is then applied to `kvengine` through `EngineCore::apply_restore_shard`.

#### Post-Apply

1. After the changeset is applied, the apply of `custom_log` is resumed, the shard version of `Applier` is updated in `Applier::handle_apply_change_set_result`, and the apply result is sent through `ExecResult::RestoreShard` using `PeerMsg::ApplyResult`.

2. After `custom_log` is applied (without any changes to `mem-table`), the request callback is triggered to notify the client-side that the shard has been restored (in `Applier::handle_apply_result`).

3. `ExecResult::RestoreShard` is processed in `StoreMsgHandle::on_restore_shard_result`, which modifies the local region version and notifies PD about the new region version.
