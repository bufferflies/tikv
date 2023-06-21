# Blob Store

## Background

When compacting an LSM tree, data entries need to be read from persistent storage, reorganized, and then written back to persistent storage. As a result, the same entry may be written multiple times, leading to a problem known as write amplification. To address this issue, in certain scenarios, it is beneficial to store larger values in a separate location that is infrequently modified. During compaction, only keys and references to the values are reorganized, significantly reducing write amplification. Additionally, using a blob store can reduce the need for read I/O in certain scenarios where only the key needs to be read, such as the data load phase during compaction. Based on these facts, blob store has been implemented.

## Overview

Please refer to [`blobtable.rs`](../../components/kvengine/src/table/blobtable/blobtable.rs) for more details about data serialization.

Some facts:

* Blob tables are not associated with any layers; a blob table can be referenced by multiple SSTs from different layers simultaneously.
* A blob table can be referenced by two shards simultaneously after a split.
* Blob table do not organize data in blocks. Checksum and compression are performed at the granularity of a single blob.

## Writing to the Blob Store

In Memtable and L0, we do not differentiate between values of different sizes. The blob store is only enabled when entries start entering L1.
During L0 compaction (compacting L0 files to L1), the following conditions are checked:

1. Whether the blob store feature is enabled.
2. Whether the estimated size of the blob table exceeds a threshold (to avoid generating too many small blob tables during a single compaction).

If both conditions are met, blob store is enabled. Once blob store is enabled, if the value's `val.value_len() > BlobTableBuilderOptions::min_blob_size`, the value is added to the blob table, and a `BlobRef` is returned. This `BlobRef` is then written to the SST for on-demand reading. When a blob table exceeds `BlobTableBuilderOptions::max_table_size`, the current blob table is finalized and rotated to the next empty blob table.

## Reading Blob

During a point-get operation, if the value read from the SST is a `BlobRef`, the corresponding blob table is accessed for reading.

During iteration, you can choose to dereference the blob or not. If dereferencing is selected, the target data segment is checked firstly in the blob_prefetcher. If it is not present, the data starting from the target data segment is loaded from the file into the blob_prefetcher, and then the target segment data is decompressed.

## GC Blob

Each SST has a property called `in_use_total_blob_size` that records the length of the blobs referenced by that SST. Additionally, Each shard keeps track of all referenced blob tables. Whenever the shard's state changes and `refresh_compaction_priority()` is called, the following condition is checked:

$$\frac{\sum_{}^{all\space ssts}{InUseTotalBlobSize}} {\sum_{}^{all\space blob\space tables}{FileSize}} \lt {BlobTableGcRatio}$$

If the condition is met, a major compaction is triggered, combining all persisted SST files based on column families into the lowest level. At the same time, all blobs are reorganized by reading them from the original blob tables and writing them into the new blob tables.

## Others

We also explored the option that enabling the blob store when flushing the memtable to L0. However, due to the smaller size of the data set at this stage, it easily leads to the generation of very small blob tables, which is not conducive to efficient resource utilization and meta data management. Therefore, we later adopted the approach of enabling the blob store during L0 compaction to L1.
