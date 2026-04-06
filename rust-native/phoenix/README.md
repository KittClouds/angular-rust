# Phoenix Native Workspace

This is the active native Rust workspace.

- Use `rust-native/phoenix` for native runtime, ingest, kernel, LMDB, perf, and post-processing work.
- The old mixed workspace remains in `rust/phoenix` as legacy/wasm history and compatibility reference.
- Shared crates that native still depends on were copied here on purpose so native work can stay isolated from the old tree.

Current native entrypoints:

- `crates/phoenix-runtime`
- `crates/phoenix-runtime-native`
- `crates/phoenix-store-lmdb`
- `crates/phoenix-perf`

Rule of thumb:

- Native changes go here first.
- Only touch `rust/phoenix` when explicitly working on wasm or old compatibility surfaces.
