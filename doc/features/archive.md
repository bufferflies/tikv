# Archive S3 objects

## Background

As the cluster runs for a long time, unused S3 objects accumulate, resulting in a gradual rise in storage costs. To
mitigate this, unused S3 objects that have exceeded a certain retention period can be archived to reduce costs.

## Usage

Use the command `archive` in cse-ctl.

```sh
./cse-ctl archive --pd 127.0.0.1:2379 --start-archive-duration "91d" --expiration-date "20230401" --dry-run
```

### Command line flags

**--pd**

- PD endpoints, use `,` to separate multiple PDs.
- Default: ""

**--cacert**

- Specifies the path to the certificate file of the trusted CA in PEM format.
- Default: ""

**--cert**

- Specifies the path to the certificate of SSL in PEM format.
- Default: ""

**--key**

- Specifies the path of file that contains X509 key in PEM format.
- Default: ""

**--data-dir**

- Specifies the data dir for loading data.
- Default: ""

**--max-archive-file-size**

- Specifies the max size of archive package file.
- Default: 1073741824

**--start-archive-duration**

- Specifies the duration of the start of the archive from now.
- Default: "91d"

**--expiration-date**

- Specifies the expiration date that does not need to archive. The expiration date from now should be greater than
  the `start-archive-duration` above. The date format is `%Y%m%d`. e.g. `20060102`
- Default: ""

**--dry-run**

- The dry run mode. To run the command but do not really archive the S3 objects.
- Default: false

## Implementation

### 1. Archive cluster backup meta daily

- The cluster backup meta files are stored in S3 at the path `/<prefix>/backup/YYYYmmdd/HHMMSS.meta`.
- For meta files older than 3 months located at `/<prefix>/backup/YYYYmmdd/`, only the first meta file is archived each
  day.
- Configure S3 lifecycle policies to delete all cluster backup meta files older than 3 months.

### 2. Identify the SST files deleted each day

In backups older than three months, compare the files contained in backups of adjacent dates. If a file that existed in
the previous day's backup meta is not found in the next day's backup meta, it is considered a deleted file.

- List backups older than three months.
- Use the cluster backup meta to restore full shards, load all shards, and aggregate the files that the cluster is using
  that day.
- Collect the files used on adjacent dates.
- Check whether the SST files of the previous day exist on the next day. If a file does not exist on the next day, it is
  considered a file that was deleted between adjacent dates. Collect these deleted SST files

### 3. Collect store snapshot meta, rlog and wal chunks each day

- Only support lightweight backups.
- The store snapshot meta files are stored in S3 at the
  path `/<prefix>/store_backup/<store_id>/snapshots/m<epoch_id>.meta`.
- The store snapshot rlog files are stored in S3 at the
  path `/<prefix>/store_backup/<store_id>/snapshots/r<epoch_id>.rlog`.
- The store wak chunks files are stored in S3 at the
  path `/<prefix>/store_backup/<store_id>/wal_chunks/e<epoch_id>_<start_off>_<end_off>.wal`
  and `/<prefix>/store_backup/<store_id>/wal_chunks/e<epoch_id>_<start_off>_<end_off>.wal.last`.
- Collect the required snapshot meta, rlog and wal chunks files for each store rfengine.
- Configure S3 lifecycle policies to delete all snapshot meta, rlog and wal chunks files older than 3 months.

### 4. Archive store snapshot meta, rlog and wal chunks and deleted SST files daily

- Implementation in the archive writer. The archive file format is as follows:

```text

Object address format:

+---------------------------------+
|         package id: u32         |
+---------------------------------+
|           offset: u64           |
+---------------------------------+
|           length: u64           |
+---------------------------------+

Archive index format v1:

+---------------------------------+
|        magic number: u32        |
+---------------------------------+
|          version: u32           |
+---------------------------------+
|   backup meta object address    |
+---------------------------------+
|      num of file ids: u32       |
+---------------------------------+
|            file id 1            |
+---------------------------------+
|            file id 2            |
+---------------------------------+
|             ...                 |
+---------------------------------+
|            file id n            |
+---------------------------------+
|       sst object address 1      |
+---------------------------------+
|       sst object address 2      |
+---------------------------------+
|             ...                 |
+---------------------------------+
|       sst object address n      |
+---------------------------------+

Wal meta format:

+---------------------------------+
|          epoch id: u32          |
+---------------------------------+
|      num of wal chunks: u32     |
+---------------------------------+
|    wal chunk object address 1   |
+---------------------------------+
|    wal chunk object address 2   |
+---------------------------------+
|               ...               |
+---------------------------------+
|    wal chunk object address n   |
+---------------------------------+

Store backup meta format:

+---------------------------------+
|          store id: u64          |
+---------------------------------+
|     snapshot epoch id: u32      |
+---------------------------------+
|  snapshot meta object address   |
+---------------------------------+
|  snapshot rlog object address   |
+---------------------------------+
|      num of wal metas: u32      |
+---------------------------------+
|      wal meta length 1: u32     |
+---------------------------------+
|      wal meta length 2: u32     |
+---------------------------------+
|               ...               |
+---------------------------------+
|      wal meta length n: u32     |
+---------------------------------+
|            wal meta 1           |
+---------------------------------+
|            wal meta 2           |
+---------------------------------+
|               ...               |
+---------------------------------+
|            wal meta n           |
+---------------------------------+

Archive index format v2:

+---------------------------------+
|        magic number: u32        |
+---------------------------------+
|          version: u32           |
+---------------------------------+
|   cluster meta object address   |
+---------------------------------+
|        num of stores: u32       |
+---------------------------------+
|    store meta length 1: u32     |
+---------------------------------+
|    store meta length 2: u32     |
+---------------------------------+
|               ...               |
+---------------------------------+
|    store meta length n: u32     |
+---------------------------------+
|          store meta 1           |
+---------------------------------+
|          store meta 2           |
+---------------------------------+
|               ...               |
+---------------------------------+
|          store meta n           |
+---------------------------------+
|      num of file ids: u32       |
+---------------------------------+
|          file id 1: u64         |
+---------------------------------+
|          file id 2: u64         |
+---------------------------------+
|             ...                 |
+---------------------------------+
|          file id n: u64         |
+---------------------------------+
|       sst object address 1      |
+---------------------------------+
|       sst object address 2      |
+---------------------------------+
|             ...                 |
+---------------------------------+
|       sst object address n      |
+---------------------------------+

Archive package format:

+---------------------------------+
|          object file 1          |
+---------------------------------+
|          object file 2          |
+---------------------------------+
|             ...                 |
+---------------------------------+
|          object file n          |
+---------------------------------+
```

- Archive Index Version 2 contains store metas compared to version 1, and each store meta has the object addresses of
  snapshot meta, rlog and wal chunks files.
- Collect the required snapshot meta, rlog and wal chunks files by store and append them to the archive writer.
- Sort the IDs of SST files that are deleted each day and append them to the archive writer.
- Package above files sequentially into a batch of large files, with each file not exceeding `max-archive-file-size`.
  The storage paths are as follows:

```sh
/<prefix>/archive/package/YYYYmmdd/00000000.pack
/<prefix>/archive/package/YYYYmmdd/00000001.pack
...
```

- Set the storage class of the packaged large files to `GLACIER_IR`.
- Store the archived object addresses in an index file `/<prefix>/archive/index/YYYYmmdd.idx` every day.
- The index file records the SST file IDs contained in each packaged file, along with their offset and length.
- The object address of that day's cluster backup meta file is also stored in the index file.

### 5. Configure AWS S3 lifecycle

Configure AWS S3 lifecycle for deletion of backup meta files and unused SST files older than three months.

- The `dfsgc` command marks currently unused SST files with a `deleted=true` tag and resets their file's last modified
  time.
- Configure two AWS S3 lifecycle rules for sst files, both with an expiration time of 100 days. One rule permanently
  deletes the SST file with `deleted=true` tag and prefix `s3_prefix` after 100 days, and the other rule deletes cluster
  backup meta file with prefix `s3_prefix/backup/` after 100 days.
- The store rfengine rotates wal file at least once a day and generates a snapshot every 4 epochs.
- Configure two AWS S3 lifecycle rules for wal and rlog files, both with an expiration time of 110 days. One rule
  permanently deletes the full backup wal and rlog files with prefix `s3_prefix/00000000` after 110 days, and the
  other rule deletes the lightweight backup wal and rlog files with prefix `s3_prefix/store_backup/` after 110 days.

```xml

<LifecycleConfiguration>
    <Rule>
        <ID>Backup Meta Expiration Rule</ID>
        <Filter>
            <Prefix>s3_prefix/backup/</Prefix>
        </Filter>
        <Status>Enabled</Status>
        <Expiration>
            <Days>100</Days>
        </Expiration>
    </Rule>
    <Rule>
        <ID>Unused SST Expiration Rule</ID>
        <Filter>
            <And>
                <Prefix>s3_prefix/</Prefix>
                <Tag>
                    <Key>deleted</Key>
                    <Value>true</Value>
                </Tag>
            </And>
        </Filter>
        <Status>Enabled</Status>
        <Expiration>
            <Days>100</Days>
        </Expiration>
    </Rule>
    <Rule>
        <ID>WAL And Rlog Expiration Rule</ID>
        <Filter>
            <Prefix>s3_prefix/00000000</Prefix>
        </Filter>
        <Status>Enabled</Status>
        <Expiration>
            <Days>110</Days>
        </Expiration>
    </Rule>
    <Rule>
        <ID>Lightweight WAL And Rlog Expiration Rule</ID>
        <Filter>
            <Prefix>s3_prefix/store_backup/</Prefix>
        </Filter>
        <Status>Enabled</Status>
        <Expiration>
            <Days>110</Days>
        </Expiration>
    </Rule>
</LifecycleConfiguration>
```

### 6. Restore keyspace from archived backup:

- This functionality is implemented within the archive reader.
- Users can restore keyspace to a specific date from the archived backups.
- During the `restore keyspace` process, if the `backup_name` does not exist the corresponding cluster backup meta, to
  initialize an archive reader, retrieve the cluster backup meta for that day from the archive reader.
- When initializing an archive reader with this specific date, load all index files after that day.
- The archive reader restores snapshot meta, rlog and wal chunks files from archive package by the store id.
- The archive reader decodes archived sst object addresses and aggregate them together with the file id as the key for
  easy retrieval of archived sst files.
- BackupCluster restores the store rfengine using snapshot meta, rlog and wal chunks files as parameters.
- During the initialization of the BackupCluster, after loading keyspace shards, get all SST files used by the keyspace
  shards.
- Before starting the KV engine, perform a check to determine the existence of these SST files. If any are missing,
  proceed to restore them from the archive reader.
- Since only the first backup of that day is archived, it is not necessary to truncate ts, so reset the value
  of `truncate_ts` to `backup_ts`.
