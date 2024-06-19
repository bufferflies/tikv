# File Based Transaction

**File Based Transaction** is a feature which pre-build all data of a transaction as one or more files (i.e. **txn file**). Then the **"txn files"** are written to **Cloud Engine** as a whole (through **DFS**). This approach will greatly reduce overhead compared to writing row by row.

This document outlines the critical code path associated with the **File Based Transaction** feature.

## Write Flow

### Clients

On transaction committing, clients (e.g. TiDB) submit all data of the transaction to **tikv-worker**. Data is separated to files (called **txn chunk**) according to size. For each **txn chunk**, **tikv-worker** will return an unique **txn chunk id**. See [`chunkWriterClient.buildChunk`](https://github.com/tikv/client-go/blob/074830f9cb5b9735fd425ce45e8a90f69f917186/txnkv/transaction/txn_file.go#L887).

After all **txn chunks** are built, clients commit the transaction by 2PC. The **txn chunk ids** will be submit to **Cloud Engine** in prewrite request. See [tidbcloud/pull/26](https://github.com/tidbcloud/kvproto/pull/26).

### TiKV Worker

**tikv-worker** receive transaction data, and build **txn chunks**. See [`cloud_worker::txn_chunk::create_txn_chunk`](https://github.com/tidbcloud/cloud-storage-engine/blob/32a1e0037a7da11604ec4dfeba5f434f58d902bb/components/cloud_worker/src/txn_chunk.rs#L117).

### Cloud Engine

#### Scheduler

* Prewrite
  
  On received prewrite request of transaction with **txn file**, **scheduler** will first **prepare** the txn chunks (pull data files from **DFS**, see [`Scheduler::build_and_scheduler_txn_file_cmd`](https://github.com/tidbcloud/cloud-storage-engine/blob/32a1e0037a7da11604ec4dfeba5f434f58d902bb/src/storage/txn/scheduler.rs#L1758)). Then check conflict with existing latches and locks of other transactions (see [`GlobalLatches::acquire`](https://github.com/tidbcloud/cloud-storage-engine/blob/32a1e0037a7da11604ec4dfeba5f434f58d902bb/src/storage/txn/region_latch.rs#L31) and [`TxnFileCommand::process_prewrite`](https://github.com/tidbcloud/cloud-storage-engine/blob/32a1e0037a7da11604ec4dfeba5f434f58d902bb/src/storage/txn/commands/txn_file.rs#L313)). If no conflict is found, **schduler** constructs and propose the `TxnFileRef` (see **Data Structures** chapter below) carried by `CustomRequest` (see [`modifies_to_requests`](https://github.com/tidbcloud/cloud-storage-engine/blob/32a1e0037a7da11604ec4dfeba5f434f58d902bb/components/cloud_server/src/raftkv.rs#L654)).

* Other Commands

  Process of other commands are similar to normal transactions. They all execute the procedures of transaction, then modify and propose the `TxnFileRef` if necessary. See `process_xxx` of [`TxnFileCommand`](https://github.com/tidbcloud/cloud-storage-engine/blob/32a1e0037a7da11604ec4dfeba5f434f58d902bb/src/storage/txn/commands/txn_file.rs#L33).

#### Preprocess

During preprocess, `ShardMeta` will trace alive transactions (neither committed nor rollback), which is used to block the region to be splited or merged as source. The reason of blocking is that the latch of **txn file** in **scheduler** is of region granularity (for scalability), so the region id must not be changed before transaction is committed or rollback.

Alive transactions are kept in [`ShardMeta::txn_file_locks`](https://github.com/tidbcloud/cloud-storage-engine/blob/32a1e0037a7da11604ec4dfeba5f434f58d902bb/components/kvengine/src/meta.rs#L42), and persisted by property [`TXN_FILE_LOCKS`](https://github.com/tidbcloud/cloud-storage-engine/blob/32a1e0037a7da11604ec4dfeba5f434f58d902bb/components/kvengine/src/shard.rs#L119).

After preprocessed, the `TxnFileRef` will be prepared (see [`Applier::handle_prepare_txn_file`](https://github.com/tidbcloud/cloud-storage-engine/blob/32a1e0037a7da11604ec4dfeba5f434f58d902bb/components/rfstore/src/store/apply.rs#L1623)) and ready for applying.

See [`preprocess_txn_file_ref`](https://github.com/tidbcloud/cloud-storage-engine/blob/32a1e0037a7da11604ec4dfeba5f434f58d902bb/components/rfstore/src/store/peer.rs#L2355) and [`ShardMeta::merge_txn_file_ref`](https://github.com/tidbcloud/cloud-storage-engine/blob/32a1e0037a7da11604ec4dfeba5f434f58d902bb/components/kvengine/src/meta.rs#L1010).

#### Apply

During apply, for primary region, first write corresponding record to `EXTRA_CF` (see [`Applier::exec_txn_file_ref`](https://github.com/tidbcloud/cloud-storage-engine/blob/32a1e0037a7da11604ec4dfeba5f434f58d902bb/components/rfstore/src/store/apply.rs#L503)). Then pass the `TxnFileRef` to **kvengine** through property of `TXN_FILE_REF` in write batch.

Process of applying in **kvengine**:

1. Merge `TxnFileRef` to currenty property of `TXN_FILE_REF`.
  
   Value of the property is a collection of `TxnFileRef` of transactions. The "Merge" action will overwrite `TxnFileRef` value of the same `start_ts`, or add to the collection if not existed. See [`Engine::merge_txn_file_ref`](https://github.com/tidbcloud/cloud-storage-engine/blob/32a1e0037a7da11604ec4dfeba5f434f58d902bb/components/kvengine/src/write.rs#L313).
  
   The property of `TXN_FILE_REF` is mainly for persistence, which will be written to `ShardMeta` along with mem table flush. On recovery, **kvengine** will recover [`ShardData::lock_txn_files`](https://github.com/tidbcloud/cloud-storage-engine/blob/32a1e0037a7da11604ec4dfeba5f434f58d902bb/components/kvengine/src/shard.rs#L965) from value of this property.

2. Merge `TxnFileRef` to `ShardData::lock_txn_files`.

   Value of `ShardData::lock_txn_files` is a collection of `TxnFile` of alive transactions. It's used to check conflict or return locks of the transactions.

   The `TxnFile` (see **Data Structures** below) is loaded by [`TxnChunkManager`](https://github.com/tidbcloud/cloud-storage-engine/blob/32a1e0037a7da11604ec4dfeba5f434f58d902bb/components/kvengine/src/txn_chunk_manager.rs#L33) according to `TxnFileRef`.

   The "Merge" is similar with merge of property `TXN_FILE_REF`, overwrite value for the same `start_ts`, or add if not existed.

   See [`Engine::merge_lock_txn_files`](https://github.com/tidbcloud/cloud-storage-engine/blob/32a1e0037a7da11604ec4dfeba5f434f58d902bb/components/kvengine/src/write.rs#L326).

3. Write to Mem Table

   If the transaction is committed, the `TxnFile` will be added to current writable mem table. As each mem table can have only one `TxnFile`, the writable mem table will be switched immediately.

   See [`Engine::write_txn_file_ref`](https://github.com/tidbcloud/cloud-storage-engine/blob/32a1e0037a7da11604ec4dfeba5f434f58d902bb/components/kvengine/src/write.rs#L272).

4. Flush

   After switched, data of **txn file** will be flush to `L0` during flush process. Then the lifetime of the **txn file** is ended.

## Read Flow

Keys of alive transactions or all key-values of committed transactions can be retrieved by table iterator with `cf = LOCK_CF` or `cf = WRITE_CF`, respectively. See [`SnapAccess::new_table_iterator`](https://github.com/tidbcloud/cloud-storage-engine/blob/32a1e0037a7da11604ec4dfeba5f434f58d902bb/components/kvengine/src/read.rs#L558).

Note that for alive transactions, table iterator only return keys of locks. Other fields of locks can be acquired by [`TxnFile::get_lock_val_prefix()`](https://github.com/tidbcloud/cloud-storage-engine/blob/32a1e0037a7da11604ec4dfeba5f434f58d902bb/components/kvengine/src/table/txn_file.rs#L185). Refer to [`TxnFileLocks::get_key_errors`](https://github.com/tidbcloud/cloud-storage-engine/blob/32a1e0037a7da11604ec4dfeba5f434f58d902bb/components/kvengine/src/util.rs#L270).

## Data Structures

### TxnFileRef

[`TxnFileRef`](https://github.com/tidbcloud/cloud-storage-engine/blob/32a1e0037a7da11604ec4dfeba5f434f58d902bb/components/kvenginepb/src/changeset.proto#L155) is the **reference** to **txn file**, which contains reference of data (i.e. id list of **txn chunks**) and meta data of the transaction.

### TxnFile

[`TxnFile`](https://github.com/tidbcloud/cloud-storage-engine/blob/32a1e0037a7da11604ec4dfeba5f434f58d902bb/components/kvengine/src/table/txn_file.rs#L130) is the memory object of a **txn file**, which contains meta data and all data of the transaction.
