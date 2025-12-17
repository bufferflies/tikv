# `native_br` crate guidelines

Scope: `components/native_br/**`

## Purpose

`native_br` provides TiKV “native” backup/restore logic used by the cloud storage engine, including backup workers, restore (incl. keyspace restore), and supporting utilities like WAL, rate limiting, and metrics.

## Key Modules

- `components/native_br/src/backup.rs`, `components/native_br/src/backup_worker.rs`: backup orchestration and worker implementation.
- `components/native_br/src/restore.rs`, `components/native_br/src/restore_keyspace.rs`: restore flows (general + keyspace-specific).
- `components/native_br/src/wal.rs`, `components/native_br/src/archive.rs`: write-ahead logging and archive handling.
- `components/native_br/src/limiter.rs`: throttling / rate limiting (often exercised by integration tests).
- `components/native_br/src/metrics.rs`: Prometheus metrics for this crate.
- `components/native_br/src/error.rs`: crate error types (prefer `thiserror` + structured context).

## Features & Flags

- `failpoints`: guard failpoint-only code with `#[cfg(feature = "failpoints")]`.
- `testexport`: exposes test-only helpers/types; used by integration tests in `tests/`.
- `pprof-fp`: forwards profiling feature to the top-level `tikv` crate.

## Build / Test / Lint

- Unit tests (crate): `cargo test -p native_br`
- With failpoints: `cargo test -p native_br --features failpoints`
- Lint (crate): `cargo clippy -p native_br --all-targets`
- Workspace format: `make format`

## Coding Conventions

- Keep changes localized within this crate; avoid drive-by refactors across `components/`.
- Prefer explicit module APIs over large “common” catch-alls; if adding shared helpers, keep them narrowly scoped.
- Metrics: add/extend counters/histograms in `components/native_br/src/metrics.rs` and follow existing naming patterns.
- Errors: propagate with context; avoid stringly-typed errors when a structured enum works.
