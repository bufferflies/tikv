# Schema Manager

## Background

Schemas are used for the data transformation from row-base format to columnar. These schemas, stored in a schema store, are fetched using a Rust client in JSON format. After fetching, the JSON-encoded schemas are decoded into their respective column information.

To decouple the schema logic from the store server, we employ a dedicated worker known as the `Schema Manager`. This Schema Manager is responsible for handling all schema-related operations, ensuring that the store server remains clean.

## Implementation

### Fetch Keyspaces and Stats

The schema manager periodically updates the schema of keyspaces, typically every minute, which can be configured. The update process involves the following steps:

- Fetching TiKV Stores Status Addresses:

The manager retrieves all TiKV store status addresses from the Placement Driver (PD).

- Fetching Shard Statistics:

The manager then fetches shard statistics from the TiKV status server using the `/kvengine/active_lite` endpoint.
To minimize unnecessary shard statistics data, the manager uses `ShardStatsLite` for serialization and deserialization, focusing only on the leader's active shard statistics.
If some shards do not have a leader during the data fetch, it has no adverse effects; the process will retry in the next update cycle.

- Filtering and Estimating Keyspace Data Size:

The manager filters transaction shards using the APIv2. It groups the shard stats by keyspace to a map, used for next steps.

### Sync Schemas

The process works as follows in one loop:

- Read Local Schema Version

The schema files are persisted on the local disk by keyspace. To manage these files conveniently and efficiently, we use a meta file to save all the file IDs and schema versions for all keyspaces. We record the schema version as `checked_version` to avoid building new schema files unnecessarily if the schemas are already included in a previous schema file.

- Find Smallest Schema Version in Shard Stats
 
Next, we identify the smallest schema version in the `shard_stats` for this keyspace. If a shard overlaps with the local schema file and the schema version in `shard_stats` is smaller than that in the persisted schema file, we broadcast a request to update the shard meta to all stores without rebuilding the schema file.

- Sync Schema Changes

In the `sync_schema` process, we pass the fetched schema version. `sync_schema` collects all schema changes between the current and the latest schema version. If `None` is passed, `sync_schema` loads all schema metas of the keyspace directly.

- Retrieve and Process Schema Changes

After retrieving all schema changes for the keyspace, we iterate through all table information. For tables with a TiFlash replica, we convert the column information to TiPB format and merge it with the schemas from the previous schema file. For tables without a TiFlash replica, we record their IDs to remove them from the previous schema file.

- Build and Upload New Schema File

After constructing the new schema file data, we create a new DFS file with a new file ID allocated by the TSO server. Once uploaded to DFS, we broadcast the schema change to all stores, including the keyspace ID and the new file ID.

- Save to Local

Finally, we write the new schema file to the local disk and update the meta file accordingly, persisting these changes to the local disk.

## Configs

The schema manager compile to `tikv-worker`. To enable it, we should add the following configuration to config file.

```
[schema-manager]
enabled = true
dir = "/data/deploy/tikv-api/schemas"
keyspace-refresh-interval = '60s'
schema-refresh-threshold = 268435456
http-timeout = '5s'
```

- enabled: Enable the schema manager, default is false.
- dir: The schemas file base directory.
- keyspace-refresh-interval: The main loop check interval, default is 60s.
- schema-refresh-threshold: Only sync schemas if keyspace size bigger than threshold, default is 256MB.
- http-timeout: The timeout of http request, default is 5s.