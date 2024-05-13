# Remote Coprocessor Functionality

This document elaborates the key code path associated with the **Remote Coprocessor** feature, presented in the sequence of processing procedure.

There are two different implementations of  the remote coprocessor. The main call path of the first way is `tidb` -> `tikv-worker` -> `tikv` -> `tikv-worker` -> `tidb`. The second way's call path is `tidb` -> `tikv` -> `tikv-worker` -> `tikv` -> `tidb`. The second way is easier to operate in the production environment, so we use the second way to enable the remote coprocessor.

## First way

### tidb-server side

The code is housed in [tidb-server](https://github.com/tidbcloud/tidb-cse/pull/395):

- The tidb-server configures the remote-coprocessor address in the config file.
- It utilizes a session variable to regulate the usage of the remote coprocessor.
- When enabled, coprocessor requests are directed to the remote-coprocessor address, bypassing the tikv-server.

### tikv-worker side

The tikv-worker, running a `RemoteCopServer`, handles the remote-coprocessor requests.

Upon receipt of the `Coprocessor` request, the tikv-worker:

- Constructs a `DelegateRequest` and dispatches it to the tikv-server.

### tikv-server side

Upon receipt of the `DelegateRequest`, the tikv-server:

- Executes `check_memory_locks`.
- Acquires the `SnapAccess` and forms a `CloudStore`.
- Checks locks and assembles the memory table data via `new_memtable_iterator`.
- Responds to the tikv-worker with the collected memory table data and snapshot change set.

### tikv-worker side

Upon receipt of the `DelegateResponse`, the tikv-worker:

- Constructs the SnapAccess using the received memory table data and snapshot change set.
- Reads the SST files in the Moka cache. If a cache miss occurs, it reads from S3.
- Handle the request via `parse_request_and_handle_remote_cop`.

## Second way

### tikv-server side

The tikv-server can be configured to offload resource-intensive coprocessor requests to remote workers. Before the offloading, the tikv-server check the range blocks of the coprocessor request. If the size exceeds the threshold, tikv-server will offload the request to the remote worker.
This can significantly improve service stability in the case of unexpected large queries exhaust the cluster resource.

example config:
```
[kvengine]
remote-coprocessor-addr = "http://127.0.0.1:19000/coprocessor"
remote-coprocessor-min-blocks-size = 32000000
```

- The `remote-coprocessor-addr` parameter specifies the address of the remote coprocessor worker to which heavy coprocessor requests will be offloaded.
- The `remote-coprocessor-min-blocks-size` parameter defines the blocks size threshold for offloading a coprocessor request to the remote worker.

The tikv-server encodes the offload request with related memtable and snapshot data, and then dispatches the encoded request to worker using http/https.

### tikv-worker side

The tikv-worker using `/coprocessor` endpoint to handle the offload request.

- Decode the request to get the memtable data and snapshot data.
- Construct the SnapAccess using the received memory table data and snapshot change set.
- Reads the SST files in the cachefs.
- Handle the request via `parse_request_and_handle_remote_cop`.

### cop-limiter

The cop-limit is configured on the tikv-worker to throttle the coprocessor requests when the memory
usage exceeds the threshold to avoid OOM.

example config:
```
[cop-limiter]
high-mem-ratio = 0.8
wait-min-duration = "100ms"
wait-max-duration = "3s"
sample-ttl = "60s"
```

When the memory usage above `high-mem-ratio`, the cop-limiter will throttle the coprocessor requests.
Each throttled request will wait until the memory usage drops below the `high-mem-ratio`.
To avoid thundering herd, a single wait sleeps for a random duration between `wait-min-duration` and `wait-max-duration`.
Each request's response size is sampled and collected in a sliding window to calculate each keyspace's throughput.
The `sample-ttl` defined the sliding window duration.
