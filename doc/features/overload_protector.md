# Overload Protector Functionality

The Overload Protector is a crucial feature designed to safeguard TiDB from being overwhelmed by coprocessor requests. 
It monitors the number of processed keys in a transaction and rejects coprocessor requests if the threshold is exceeded.

## Configuration

The default configuration values are as follows:

```toml
[overload]
enable = false
max-keys = 200000000
max-data-size = "16GB"
discard-keys = 10000
process-interval = "30s"
task-expire = "10m"
```

Online configuration changes can be made using the following command:

```shell
curl -X POST 'http://localhost:20280/config' -d '{"overload.enable": false, "overload.max-keys": 1000000, "overload.max-data-size": "1GB"}'
```

Please note that only enable, max-keys, and max-data-size can be changed online, 
while other configurations require restarting the TiKV process.

## Implementation

The `OverloadProtectorWorker` operates as a background thread on `TikvServer` start. 
It receives `CopTaskStats` from `Coprocessor` and records the number of processed keys and data size for each transaction.

The collected stats are processed periodically, and if any transaction exceeds the threshold, the overload map is updated. 
As a result, the Coprocessor will reject the request and return an error message indicating Overload protection.

To save memory, transactions that don't reach the `discard-keys` threshold within the `process-interval` are removed from the stats map.

Furthermore, transactions that don't surpass the overload threshold (`keys > max-keys || data-size > max-data-size`) are 
removed from the stats map after the `task-expire` time to optimize memory usage.