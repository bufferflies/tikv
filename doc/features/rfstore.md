# Rfstore Main Procedure

This doc mainly explains the workflow and principles of `rfstore`, which helps to better understand the implementation logic of `rfstore`.

## 1. Initialization

`rfstore` initialized in [TikvServer::setup](https://github.com/tidbcloud/cloud-storage-engine/blob/2bd99c8c36efb6dfb9178b122bc5debadaf483e2/components/cloud_server/src/tikv_server.rs#L238).

The main task of initialization is to initialize two channels to transmit `StoreMsg` and `PeerMsg`. See [RaftBatchSystem::new](https://github.com/tidbcloud/cloud-storage-engine/blob/78cff98b979e041edb71b2b5057f5f3029ba6d8c/components/rfstore/src/store/store_fsm.rs#L81).

`StoreMsg` channel is created during `meta_change_listener` creation. The `meta_change_listener` is saved in kvengine and implement `on_change_set()` to generate and send `StoreMsg::GenerateEngineChangeSet` to rfstore and transmit it to other peers. See [TikvServer::init_kv_engine](https://github.com/tidbcloud/cloud-storage-engine/blob/2bd99c8c36efb6dfb9178b122bc5debadaf483e2/components/cloud_server/src/tikv_server.rs#L944) and [MetaChangeListener::on_change_set](https://github.com/tidbcloud/cloud-storage-engine/blob/83524f9c95e5cac9898aca345136c519abdcb4aa/components/rfstore/src/store/engine.rs#L79).

After `rfstore` initialization, `TikvServer` will save the `RaftRouter` and pass it to other components(StatusServer, RaftKv). See [TikvServer::setup](https://github.com/tidbcloud/cloud-storage-engine/blob/2bd99c8c36efb6dfb9178b122bc5debadaf483e2/components/cloud_server/src/tikv_server.rs#L272).

## 2. Start

`rfstore` started in [Node::start_store](https://github.com/tidbcloud/cloud-storage-engine/blob/d48ea573c740ee314365cf8112dbd3291985113b/components/cloud_server/src/node.rs#L187), entry [RaftBatchSystem::spawn](https://github.com/tidbcloud/cloud-storage-engine/blob/78cff98b979e041edb71b2b5057f5f3029ba6d8c/components/rfstore/src/store/store_fsm.rs#L100).

The main task of start is to run three types of workers:

- RaftWorker: runs the raftstore thread, which is responsible for handling primary tasks of rfstore and sending leader peer messages to follower peers. See [raftstore](https://github.com/tidbcloud/cloud-storage-engine/blob/78cff98b979e041edb71b2b5057f5f3029ba6d8c/components/rfstore/src/store/store_fsm.rs#L203)
- ApplyWorker: runs the apply thread, which is responsible for applying committed raft log to kvengine(data). See [apply](https://github.com/tidbcloud/cloud-storage-engine/blob/78cff98b979e041edb71b2b5057f5f3029ba6d8c/components/rfstore/src/store/store_fsm.rs#L216)
- IoWorker: runs the raft_io thread, which is responsible for persisting raft log and send follower peer messages to leader peer. See [raft_io](https://github.com/tidbcloud/cloud-storage-engine/blob/78cff98b979e041edb71b2b5057f5f3029ba6d8c/components/rfstore/src/store/store_fsm.rs#L186).

# 3. RaftWorker main loop

Main loop see [RaftWorker::run](https://github.com/tidbcloud/cloud-storage-engine/blob/78cff98b979e041edb71b2b5057f5f3029ba6d8c/components/rfstore/src/store/peer_worker.rs#L110).

## 3.1 handle_store_msg

- Receive `StoreMsg` messages from the channel.
- Call [StoreMsgHandler::handle_msg](https://github.com/tidbcloud/cloud-storage-engine/blob/78cff98b979e041edb71b2b5057f5f3029ba6d8c/components/rfstore/src/store/peer_worker.rs#L149) to handle `StoreMsg` messages.
  - `GenerateEngineChangeSet` is sent by `meta_change_listener` when the kvengine meta changed (shard compaction and flush) and be replicated to other peers through raft. See [MetaChangeListener::on_change_set](https://github.com/tidbcloud/cloud-storage-engine/blob/83524f9c95e5cac9898aca345136c519abdcb4aa/components/rfstore/src/store/engine.rs#L79).
  - `RaftMessage` call `StoreMsgHandler::on_raft_message` to process it. If keyspace or region of this message is in blacklist or `store not match`, the message will be dropped directly. After a series of pre-checks, the message will be sent to `PeerMsgHandler`.
  - `ApplyResult` is the post-process after apply that will be handled by store, including split/merge related operations. See [StoreMsgHandler::on_split_region](https://github.com/tidbcloud/cloud-storage-engine/blob/78cff98b979e041edb71b2b5057f5f3029ba6d8c/components/rfstore/src/store/store_fsm.rs#L1681). Then [Peer::post_apply](https://github.com/tidbcloud/cloud-storage-engine/blob/d6c209b873cd250af2f0000e2c66fd134c0207f6/components/rfstore/src/store/peer.rs#L2192) will be called to update raft committed index and update stale read progress. [Peer::handle_raft_ready](https://github.com/tidbcloud/cloud-storage-engine/blob/d6c209b873cd250af2f0000e2c66fd134c0207f6/components/rfstore/src/store/peer.rs#L1492) to trigger peer to handle raft ready messages of next round. The apply will be sent to `ApplyWorker` if region has messages to apply. 
  - More type messages see [StoreMsgHandler::handle_msg](https://github.com/tidbcloud/cloud-storage-engine/blob/78cff98b979e041edb71b2b5057f5f3029ba6d8c/components/rfstore/src/store/store_fsm.rs#L722).
- Send `Tick` if time duration longer enough after last tick to trigger pd_heartbeat. See [StoreMsgHandler::on_tick](https://github.com/tidbcloud/cloud-storage-engine/blob/78cff98b979e041edb71b2b5057f5f3029ba6d8c/components/rfstore/src/store/store_fsm.rs#L774)

## 3.2 receive_msgs

- Gather messages from the peer message channel and save then to the inboxes. See [RaftWorker::append_msg](https://github.com/tidbcloud/cloud-storage-engine/blob/78cff98b979e041edb71b2b5057f5f3029ba6d8c/components/rfstore/src/store/peer_worker.rs#L184). If peer not exists, `PeerMsg::RaftMessage` will be sent to store to handle it. For `PeerMsg::RaftCommand` will return `RegionNotFound` error.
- Append `PeerMsg::Tick` to the inbox message if needed.

## 3.3 process_inbox

Process the `PeerMsg` messages in the inboxes. `PeerMsgHandler` will be created and will call `handle_msgs`. See [PeerMsgHandler::handle_msgs](https://github.com/tidbcloud/cloud-storage-engine/blob/78cff98b979e041edb71b2b5057f5f3029ba6d8c/components/rfstore/src/store/peer_worker.rs#L277).

- `RaftMessage` to step raft state.
- `RaftCommand` to propose raft command.
- `ApplyResult` is a post-process after apply, see [Applier::handle_raft_committed_entries](https://github.com/tidbcloud/cloud-storage-engine/blob/83524f9c95e5cac9898aca345136c519abdcb4aa/components/rfstore/src/store/apply.rs#L1038). If shard meta changes (split/merge/conf change) or if the peer already has pending messages, the apply result will be pushed to the `pending_apply_result` and send the result to store FSM for handling. Otherwise, [Peer::post_apply](https://github.com/tidbcloud/cloud-storage-engine/blob/d6c209b873cd250af2f0000e2c66fd134c0207f6/components/rfstore/src/store/peer.rs#L2192) will be called to update the raft committed index and update stale read progress.
- `CasualMessage` to implement custom functions through raft, such as restore shard, split region or trim overbound.
- ChangeSet related messages.
- `Persisted` is callbacks messages from IoWorker after message sent to peer.

## 3.4 handle_raft_ready

Handle raft procedure.
- Obtain raft ready messages from the raft group.
- Send raft message to other peers if it's leader. See [Peer::send_raft_message](https://github.com/tidbcloud/cloud-storage-engine/blob/d6c209b873cd250af2f0000e2c66fd134c0207f6/components/rfstore/src/store/peer.rs#L1538).
- Handle raft committed entries.
  - Preprocess committed entry, See [Peer::handle_raft_committed_entry](https://github.com/tidbcloud/cloud-storage-engine/blob/d6c209b873cd250af2f0000e2c66fd134c0207f6/components/rfstore/src/store/peer.rs#L1638).
  - Build `ApplyMsgs` and save to `ctx.apply_msgs.msgs`.
- Call `PeerStorage::handle_raft_ready` to persist raft logs.
- Build raft messages from ready persist messages and send IoTask to IoWorker. See [RaftWorker::persist_state](https://github.com/tidbcloud/cloud-storage-engine/blob/78cff98b979e041edb71b2b5057f5f3029ba6d8c/components/rfstore/src/store/peer_worker.rs#L128).
- Send apply message to apply worker. See [RaftWorker::maybe_send_apply](https://github.com/tidbcloud/cloud-storage-engine/blob/78cff98b979e041edb71b2b5057f5f3029ba6d8c/components/rfstore/src/store/peer_worker.rs#L279).

## 3.5 aux_worker

RaftAuxWorker is used to handle part of the raft task to eliminate single thread bottleneck.
It handles raft worker tasks except `handle_store_msg` and `receive_msgs`.
On each loop, the main raft worker must sync the aux raft worker to avoid race.
The aux worker count can be configured by 

```
[raftstore]
store-pool-size = 2
````

The aux worker count is `store-pool-size - 1`.

# 4. ApplyWorker main loop

Main loop see [ApplyWorker::run](https://github.com/tidbcloud/cloud-storage-engine/blob/78cff98b979e041edb71b2b5057f5f3029ba6d8c/components/rfstore/src/store/peer_worker.rs#L339).

- Receive ApplyBatch from channel.
- Handle every `ApplyMsg` from batch and call [Applier::handle_msg](https://github.com/tidbcloud/cloud-storage-engine/blob/78cff98b979e041edb71b2b5057f5f3029ba6d8c/components/rfstore/src/store/peer_worker.rs#L349).
  - Append proposal.
  - Apply committed raft entries. Call `handle_raft_entry_normal` if the entry is `EntryNormal` or `handle_raft_entry_conf_change` if the entry is `EntryConfChange`. If entry is normal, call `exec_admin_cmd` or `exec_custom_log` to apply the command and call `handle_apply_result` to handle the apply result. e.g. split/merge/conf change results in peer meta updates. See [Applier::apply_raft_log](https://github.com/tidbcloud/cloud-storage-engine/blob/83524f9c95e5cac9898aca345136c519abdcb4aa/components/rfstore/src/store/apply.rs#L731).
  - Send `PeerMsg:ApplyResult` to raftworker to handle the apply result. See [Applier::finish_for](https://github.com/tidbcloud/cloud-storage-engine/blob/83524f9c95e5cac9898aca345136c519abdcb4aa/components/rfstore/src/store/apply.rs#L1080).
- Handle ChangeSet related message. `ApplyMsg::PrepareChangeSet` and `ApplyMsg::ApplyChangeSet`.
- Other `ApplyMsg` see [Applier::handle_apply_msg](https://github.com/tidbcloud/cloud-storage-engine/blob/83524f9c95e5cac9898aca345136c519abdcb4aa/components/rfstore/src/store/apply.rs#L1442).


# 5. IoWorker main loop

raft_io thread loop receives ready messages from channel and create `IoTask` to handle them. See [IoWorker::run](https://github.com/tidbcloud/cloud-storage-engine/blob/78cff98b979e041edb71b2b5057f5f3029ba6d8c/components/rfstore/src/store/peer_worker.rs#L390).

- Persist task.raft_wb to WAL. See [RfEngineCore::persist](https://github.com/tidbcloud/cloud-storage-engine/blob/78cff98b979e041edb71b2b5057f5f3029ba6d8c/components/rfstore/src/store/peer_worker.rs#L416)
- Transmit follower peer messages to leader peer and send `PeerMsg::Persisted` to raftworker to update raft state. See [PeerMsgHandler::on_persisted](https://github.com/tidbcloud/cloud-storage-engine/blob/6106a6b05c0965d7adbdbd5b033fae52cd694a0b/components/rfstore/src/store/peer_fsm.rs#L1454).