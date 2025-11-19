# AGENTS.md

This file provides guidance to work with code in this repository.

## Project Overview

`kvenginepb` is a protobuf definitions library that provides serialization formats for the `kvengine` distributed storage engine. It defines the protocol buffer schemas for changeset operations, FTS (Full-Text Search) structures, and storage metadata used throughout the TiKV Cloud Storage Engine.

`kvenginepb` is a sub-component of the TiKV Rust project workspace. All TiKV source code is available at `../..`.

## Architecture

### Core Components

**Protobuf Definitions:**

- `changeset.proto` - Defines `ChangeSet` and related messages for tracking storage engine state changes
- `fts.proto` - Defines Full-Text Search structures including `FullTextIndexDef`, packed file formats, and dedicated file structures

**Generated Code:**

IMPORTANT: These code are generated and very long. Prefer reading from protobuf definition files first.

- `changeset.rs` - Generated Rust bindings for changeset protocol buffers
- `fts.rs` - Generated Rust bindings for FTS protocol buffers
- `lib.rs` - Main library interface with utility functions

### Key Design Patterns

- **Storage Operation Tracking**: `ChangeSet` messages capture all types of storage operations (flush, compaction, split, etc.)
- **Multi-Format Storage Support**: Supports row storage, columnar storage, vector indexes, and full-text search
- **Incremental Processing**: Includes snapshot versions and sequence numbers for incremental operations

### Core Data Structures

**ChangeSet Operations:**

- `Flush` - MemTable to L0 SSTable operations
- `Compaction` - LSM-Tree level merging operations
- `Snapshot` - Complete shard state capture
- `Split` - Shard split operations
- `FtsUpdate` - Unified FTS metadata updates (tracked indexes, L0/L1 files, snap watermark)
- `ColumnarCompaction` - Columnar storage operations
- `UpdateVectorIndex` - Vector index updates

**FTS Structures:**

- `FullTextIndexDef` - Full-text index definitions with parser configuration
- `PackedFileIndexBlock` - Efficient packed file storage format
- `DedicatedFilePropertyBlock` - Dedicated file metadata and structure

## File Organization

- `src/changeset.proto` - Storage operation definitions
- `src/fts.proto` - Full-text search structure definitions
- `src/lib.rs` - Library interface with utility functions
- `build.rs` - Protobuf code generation build script
- Generated files (`changeset.rs`, `fts.rs`) are created in `src/` during build

## Integration Points

- **kvengine**: Primary consumer of changeset messages for state management
- **clara_fts**: Uses FTS protobuf definitions for index file formats
- **Storage Layer**: ChangeSet messages coordinate distributed storage operations
- **Compaction Services**: Remote compaction uses changeset messages for coordination
