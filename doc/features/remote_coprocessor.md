# Remote Coprocessor Functionality

This document elaborates the key code path associated with the **Remote Coprocessor** feature, presented in the sequence of processing procedure.

## tidb-server side

The code is housed in [tidb-server](https://github.com/tidbcloud/tidb-cse/pull/395):

- The tidb-server configures the remote-coprocessor address in the config file.
- It utilizes a session variable to regulate the usage of the remote coprocessor.
- When enabled, coprocessor requests are directed to the remote-coprocessor address, bypassing the tikv-server.

## tikv-worker side

The tikv-worker, running a `RemoteCopServer`, handles the remote-coprocessor requests.

Upon receipt of the `Coprocessor` request, the tikv-worker:

- Constructs a `DelegateRequest` and dispatches it to the tikv-server.

## tikv-server side

Upon receipt of the `DelegateRequest`, the tikv-server:

- Executes `check_memory_locks`.
- Acquires the `SnapAccess` and forms a `CloudStore`.
- Checks locks and assembles the memory table data via `new_memtable_iterator`.
- Responds to the tikv-worker with the collected memory table data and snapshot change set.

## tikv-worker side

Upon receipt of the `DelegateResponse`, the tikv-worker:

- Constructs the SnapAccess using the received memory table data and snapshot change set.
- Reads the SST file in the Moka cache. If a cache miss occurs, it reads from S3.
- Handle the request via `parse_request_and_handle_remote_cop`.

## offload heavy requests

tikv-server can be configured to offload resource-intensive coprocessor requests to remote workers. 
This can significantly improve service stability in the case of unexpected large queries exhaust the cluster resource.

example config:
```
[kvengine]
remote-coprocessor-addr = "http://127.0.0.1:19000/coprocessor"
remote-coprocessor-min-blocks = 256
remote-coprocessor-white-list = [1]
```

- The `remote-coprocessor-addr` parameter specifies the address of the remote coprocessor worker to which heavy coprocessor requests will be offloaded.
- The `remote-coprocessor-min-blocks` parameter defines the threshold for offloading a coprocessor request to the remote worker.
- The `remote-coprocessor-white-list` parameter is used to specify a list of keyspaces that are eligible for this offloading feature. An empty white list ([]) enables the feature for all keyspaces.
