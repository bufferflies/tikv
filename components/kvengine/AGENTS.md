# AGENTS.md

This file provides guidance to work with code in this repository.

## Project Overview

`kvengine` is a distributed storage engine that serves as the "state machine" component in the TiKV architecture. It implements a sharded, multi-level LSM-Tree (Log-Structured Merge-Tree) storage system with pluggable Distributed File System (DFS) support, making it ideal for cloud deployments using object storage like S3.

`kvengine` is a sub-component of TiKV Rust project workspace. The all TiKV source code is available at `../..`.

If you want to learn more about `kvengine`, please refer to `../../doc/architecture/kvengine.md`.

## Architecture

### High-Level Components

- **Engine/EngineCore** (`src/engine.rs`): Main storage API and shard management
- **Shard** (`src/shard.rs`): Independent LSM-Tree for a key range, unit of concurrency
- **DFS Layer** (`src/dfs/`): Abstraction for storage backends (primarily S3)
- **LSM-Tree Components**:
  - **MemTable** (`src/table/memtable/`): In-memory buffer (SkipList implementation)
  - **SSTable** (`src/table/sstable/`): Sorted string tables for persistent storage
  - **BlobTable** (`src/table/blobtable/`): Separate storage for large values
- **Multi-Format Storage**:
  - **Columnar** (`src/table/columnar/`): Column-oriented storage for OLAP workloads
  - **Vector Index** (`src/table/vector_index/`): Vector similarity search using usearch
  - **Full-Text Search** (`src/table/fts/`): Text search capabilities

### Core Processes

- **Write Path** (`src/write.rs`): Writes go through rfstore → kvengine → Shard → MemTable
- **Read Path** (`src/read.rs`): Searches MemTables → L0Tables → Ln-SSTables in order
- **Flush Process** (`src/flush.rs`): MemTable → L0 SSTable via background workers
- **Compaction** (`src/compaction.rs`): LSM-Tree level merging, including remote compaction
- **MVCC** (`src/mvcc.rs`): Multi-version concurrency control with timestamps
- **Shard Operations** (`src/split.rs`, `src/prepare.rs`): Split/merge operations

### Key Design Patterns

- **Sharded LSM-Tree**: Each shard handles a contiguous key range independently
- **Multi-Level Storage**: Row (LSM), columnar (OLAP), and vector (similarity search) formats
- **Remote Compaction**: Offload intensive compaction to separate services
- **DFS Abstraction**: Pluggable storage backends via `dfs` module
- **Iterator Architecture**: `ConcatIterator` and `MergeIterator` for efficient scanning
- **MVCC with Raft**: Transaction consistency through Raft consensus coordination

## Test Guidelines

### Unit Tests

IMPORTANT: When you want to run tests, you should always run tests like this to effectively only test for the changes you made, as there are too many tests if you do not specify a filter:

```bash
cargo test --package kvengine --lib -- table::vector_index --nocapture
```

### Integration Tests

IMPORTANT: Most integration tests of this module are placed in `../../tests/cloud_engine`. You should only run relevant integration tests that are affected by your changes as well.

For example, to run all integration tests whose names contain "vector_index":

```bash
cargo test --package tests --test cloud_engine --features testexport vector_index -- --nocapture
```

Notice the `--features testexport` flag is required to enable exporting some test-only APIs from kvengine.

You should always run integration tests along with `tail` (for at least 100 lines), `grep` or redirecting test stdout to a file to see the detailed logs, because integration tests usually print a lot, which prevent you from seeing test failures.

To debug integration test failures step by step, you could add debug logs with tags like `testdebug`, and then use `grep` to filter out relevant logs. Never read all logs, there are too many.

## Important Notes

- **Raft Integration**: kvengine is called by rfstore for actual data storage - understand this interaction
- **Multi-Format Support**: Changes to core storage may affect columnar and vector index generation
- **Remote Operations**: Compaction can be performed remotely - consider this in distributed scenarios
- **MVCC Timestamps**: All operations use start_ts/commit_ts for transaction consistency
- **DFS Abstraction**: Storage operations go through DFS layer - test with different backends
- **Memory Management**: MemTable size limits trigger flushes - understand flush triggers

## Testing Best Practices & Common Pitfalls

### Testing Philosophy

### Test Development Strategy

- **Run tests incrementally**: Always run each test immediately after writing it to catch issues early
- **Use targeted test commands**: `cargo test --package kvengine --lib -- specific::test::path --nocapture`
- **Start simple, add complexity**: Begin with basic functionality, then add edge cases and complex scenarios
- **Test both positive and negative cases**: Verify expected behavior AND error conditions
- **Ergonomic test utilities**: Test helper functions should return clean types and handle errors internally (use `.unwrap()` for fail-fast behavior)

### MVCC and Version Handling

- **Test data limitations**: `TiDbTableBuilder` uses high default versions (likely u64::MAX), making MVCC tests challenging
- **Version semantics**: `has_newer_version_int(pk, version, max_version)` finds versions > version AND <= max_version
- **Multiple SST files**: Use non-overlapping handles across SST files to avoid "Versions must be in descending order" errors
- **Version debugging**: Use debug prints to understand actual stored versions in test vs. production data

### FTS-Specific Testing

- **Logical partition keys**: Format is `lp_key(table_id, index_id)` - ensure consistency
- **IndexReader integration**: Test actual search functionality, not just index creation
- **Query object creation**: Use proper `FtsQueryInfo` setup for different query types
- **Search validation**: Verify both document presence and absence in search results
- **Multi-index isolation**: Ensure searches in one index don't return results from other indexes

### Debugging Strategies

- **Use debug prints**: Add `println!` statements to understand actual data values during test failures
- **Check intermediate results**: Validate logical partition existence before testing search functionality
- **Incremental validation**: Test each component (schema, SST creation, FTS conversion, search) separately
- **Error message analysis**: Pay attention to specific assertion failures and trace back to root cause

### Performance Considerations

- **Large test suites**: Always use specific test filters to avoid running hundreds of unrelated tests

### Protobuf Integration (tipb)

Working with the tipb crate and protobuf-generated code requires understanding several key patterns:

- **Generated code location**: tipb protobuf definitions are generated at build time in `target/debug/build/tipb-*/out/protos/`
- **Method patterns**: Use `mut_field().push()` instead of creating `RepeatedField` manually (e.g., `info.mut_columns().push(column_info)`)
- **Workspace dependencies**: tipb is a workspace dependency, not directly listed in kvengine Cargo.toml
- **Field evolution**: API methods may change (e.g., `set_column_id()` → `set_columns()` with ColumnInfo), check generated code
- **Exploration technique**: Use `find /path -name "*.rs" | xargs grep -l "TypeName"` to locate generated struct definitions and available methods

## File Organization

- `src/engine.rs` - Main engine implementation and shard coordination
- `src/shard.rs` - Individual shard management and LSM-Tree operations
- `src/table/` - All table format implementations (SSTable, columnar, vector, etc.)
- `src/dfs/` - Distributed file system abstraction and S3 implementation
- `src/compaction.rs` - Compaction logic including remote compaction
- `src/flush.rs` - MemTable flushing to persistent storage
- `src/read.rs` & `src/write.rs` - Core read/write operation implementations
- `src/meta.rs` - Metadata management for shards and tables
