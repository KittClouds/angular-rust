# Native / Wasm Divorce Plan

## Goal

Finish the architectural divorce between the native Phoenix V2 app and the wasm app.

This is not a portability exercise. The native app and the wasm app may share concepts, but they must stop sharing behavior-bearing crates and mixed orchestration. Native should be free to optimize for:

- LMDB-native durability
- graph-kernel-first semantics
- ingest throughput
- entity retrieval and resolution
- long-context memory quality

Wasm should remain free to keep its Cozo-centered design without constraining native.

## Current Slowdown Root

The immediate `perfect_run` regression is in native collective ER:

- crate: `phoenix-invarant-v2`
- file: `rust/phoenix/crates/phoenix-invarant-v2/src/lib.rs`
- stage: `resolve_mentions()`
- issue: document-local same-surface support still rescans all prepared mentions for each mention

This is the direct hot-loop failure.

But the larger source is architectural: native still lives inside mixed crates and shared abstractions that were shaped around the wasm app.

## Current Mixed Stack

### Native execution today

`phoenix-runtime`
-> `phoenix-invarant-v2`
-> `phoenix-graph-kernel`
-> `phoenix-semantic-v2`
-> `phoenix-store-lmdb`
-> `phoenix-triverse-v2`

### Wasm execution today

`phoenix-runtime`
-> `phoenix-scanner`
-> `phoenix-structure`
-> `phoenix-store-cozo`
-> `phoenix-graptor`
-> `phoenix-wasm`

### Core problem

The native and wasm worlds still meet inside:

- `phoenix-runtime`
- `phoenix-invarant-v2`
- `phoenix-triverse-v2`
- `phoenix-store-native`
- `phoenix-store-lmdb`

That means native still carries wasm-era nouns, contracts, and compatibility burdens even when the code path is native-only.

## Severance Classes

### 1. Must become native-only

These crates contain behavior and should not be shared with the wasm app.

#### `phoenix-runtime`

- Problem:
  - mixed orchestration
  - `native_graph_enabled()` branches everywhere
  - both `PhoenixCozoStore` and `PhoenixLmdbStore` live in the same runtime surface
- Native pain:
  - native design stays constrained by wasm request routing and store-command compatibility
- Action:
  - split into `phoenix-runtime-native` and `phoenix-runtime-wasm`

#### `phoenix-invarant-v2`

- Problem:
  - native analyze logic still imports `phoenix-store-cozo::StoreError`
  - native ingest still carries bundle-oriented compatibility nouns like `BundleKind`
- Native pain:
  - native ER and archive logic are still expressed through wasm-born storage language
- Action:
  - replace with `phoenix-analyze-native`

#### `phoenix-triverse-v2`

- Problem:
  - native query projection still depends on `phoenix-store-cozo::StoreError`
  - carries mixed assumptions around lexical/query composition
- Native pain:
  - query stack remains partially shaped by legacy store/query semantics
- Action:
  - replace with `phoenix-query-native`

#### `phoenix-store-native`

- Problem:
  - depends on `phoenix-store-cozo`
  - native contracts still inherit compatibility shape
- Native pain:
  - store interfaces are not purely native-first
- Action:
  - replace with `phoenix-store-native-core`

#### `phoenix-store-lmdb`

- Problem:
  - depends on `phoenix-store-cozo`
  - imports Cozo-shaped error/constants/snapshot compatibility nouns
- Native pain:
  - LMDB persistence still speaks a partially wasm-era language
- Action:
  - keep crate, but remove all `phoenix-store-cozo` dependencies and rename the remaining native contract types if needed

### 2. Must become wasm-only

These crates should stop influencing native design.

#### `phoenix-scanner`

- Purpose:
  - wasm scan/discovery pipeline
- Action:
  - wasm-only

#### `phoenix-structure`

- Purpose:
  - wasm structure builder
- Action:
  - wasm-only

#### `phoenix-store-cozo`

- Purpose:
  - wasm / legacy persistence center
- Action:
  - wasm-only

#### `phoenix-graptor`

- Purpose:
  - wasm/legacy graph-semantic world
- Action:
  - wasm-only

#### `phoenix-wasm`

- Purpose:
  - wasm entrypoint
- Action:
  - should depend on wasm runtime only

### 3. Can remain native-owned or neutral

These are acceptable to keep on the native side because they are already native-first or sufficiently low-level.

#### `phoenix-graph-kernel`

- Role:
  - authoritative native graph kernel
- Action:
  - native-owned

#### `phoenix-semantic-v2`

- Role:
  - native archive/segment schema
- Action:
  - native-owned

#### `phoenix-hyperbolic`

- Role:
  - ANN/HNSW engine
- Action:
  - native-owned

#### `phoenix-qgram`, `phoenix-lex`, `phoenix-chunker`

- Role:
  - low-level algorithmic utilities
- Action:
  - may remain shared only if treated as utility libraries, not app-design carriers
- Note:
  - if "full divorce" is interpreted literally, these also split later

#### `phoenix-types`

- Role:
  - shared data model and request/response types
- Action:
  - can stay shared as ABI/wire types for now
- Note:
  - this is the only shared crate that should be tolerated short-term

## Current Native Leaks

### Leak: runtime is still dual-world

`phoenix-runtime/src/lib.rs`

- native scan routes to `invarant-v2`
- wasm scan routes to `phoenix-scanner`
- native structure routes to `invarant-v2`
- wasm structure routes to `phoenix-structure`
- many store commands still branch on `native_graph_enabled()`

This is the biggest architectural leak. The runtime itself is still a mixed shell around both apps.

### Leak: native analyze still imports Cozo-shaped storage language

`phoenix-invarant-v2/src/lib.rs`

- imports `phoenix_store_cozo::StoreError`
- imports bundle traits and bundle kinds from `phoenix-store-native`
- still uses `BundleKind::DocumentArchive`, `BundleKind::SessionArchive`, `BundleKind::ScopeLexSidecar`

Even though native no longer persists monolithic bundles, the crate still thinks in those compatibility nouns.

### Leak: native query still depends on Cozo-shaped errors/contracts

`phoenix-triverse-v2/src/lib.rs`

- imports `phoenix_store_cozo::StoreError`

This is a smaller leak, but it proves the native query crate is not fully sovereign yet.

### Leak: native LMDB store still depends on wasm-era store crate

`phoenix-store-lmdb/src/lib.rs`

- imports `phoenix_store_cozo::StoreError`
- imports semantic constants from `phoenix-store-cozo`
- still uses Cozo-shaped snapshot envelope compatibility

The durability layer is native, but its language is still partially borrowed.

### Leak: native store contract still depends on wasm-era store crate

`phoenix-store-native/Cargo.toml`

- directly depends on `phoenix-store-cozo`

That means the native contract layer has not been fully cut free.

## Pain Points That The Divorce Must Solve

### 1. Ingest/ER must be indexed native-first

The current ER regression happened because the native analyzer evolved inside a shared compatibility shell instead of as a dedicated native indexing pipeline.

Native analyze should explicitly own:

- mention indexing
- same-surface support indexing
- alias memory lookup
- candidate aggregation
- resolved mention emission

without carrying bundle-era or Cozo-era abstractions.

### 2. Runtime orchestration must stop branching per feature

The native app should not keep asking:

- am I native?
- am I legacy?
- should I call Cozo?
- should I call LMDB?

The runtime itself should be split so those questions disappear.

### 3. Query must target kernel + native lexical state only

Native query should be built around:

- kernel view
- native lexical sidecars / qgram / scope-local indexes
- native ANN
- native session state

not compatibility projections shaped by store-cozo errors or legacy command surfaces.

### 4. Snapshot and durability must become fully native-language

Native snapshot/export/import should stop borrowing Cozo-shaped envelope concepts.

The native store should own:

- native error type
- native snapshot envelope
- native semantic constants
- native archive/query contracts

## Target Native Stack

The target stack should look like this:

`phoenix-runtime-native`
-> `phoenix-analyze-native`
-> `phoenix-entity-native` (optional internal split later)
-> `phoenix-graph-kernel`
-> `phoenix-semantic-v2`
-> `phoenix-store-native-core`
-> `phoenix-store-lmdb`
-> `phoenix-query-native`
-> `phoenix-hyperbolic`

And the wasm stack should look like this:

`phoenix-runtime-wasm`
-> `phoenix-scanner`
-> `phoenix-structure`
-> `phoenix-store-cozo`
-> `phoenix-graptor`
-> `phoenix-wasm`

## Execution Order

### Step 1. Split runtime

Create separate runtime crates or separate runtime roots:

- `phoenix-runtime-native`
- `phoenix-runtime-wasm`

This is the most important cut because it removes the giant `native_graph_enabled()` branch surface.

### Step 2. Split native store contract

Create a native-only contract crate and remove `phoenix-store-cozo` from:

- `phoenix-store-native`
- `phoenix-store-lmdb`
- `phoenix-invarant-v2`
- `phoenix-triverse-v2`

### Step 3. Replace `phoenix-invarant-v2`

Turn it into a native analyzer crate with native nouns only:

- no `StoreError` from Cozo
- no `BundleKind`
- no compatibility archive naming language

### Step 4. Replace `phoenix-triverse-v2`

Turn it into a native query engine with:

- native error types
- kernel-first query surfaces
- native lexical + ANN composition only

### Step 5. Freeze wasm stack

After runtime split, wasm should continue on its own stack untouched.

From that point on, native performance work can proceed without risking wasm behavior.

## Immediate Next Technical Fix

After the runtime/store/analyze severance begins, the first native ER fix is:

- replace the per-mention full-scan same-surface support path in `resolve_mentions()`
- build a precomputed `normalized_surface -> mention indexes / entity support` index once

That is the direct hot-loop fix, but it should land in the native-only analyzer, not be layered deeper into a mixed crate.
