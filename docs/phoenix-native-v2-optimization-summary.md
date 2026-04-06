# Phoenix Native V2 Optimization Summary

## Purpose

This document captures the major native V2 optimization work completed during the ingest and large-corpus performance pass, plus the crates that materially shaped the final implementation.

The intent is to close the loop on optimization work before shifting focus back to feature delivery.

## Scope

This pass optimized the native runtime path only.

- Wasm behavior was kept isolated.
- Cozo-era assumptions were removed from the native durability and snapshot path.
- The main benchmark used during this pass was `docs/perfect_run.md`.

## High-Level Outcomes

The largest gains came from removing hidden architectural drag and algorithmic mistakes, not from micro-optimizing Rust syntax.

Representative `perfect_run` progress from this pass:

- `ingest_document`: from roughly `100074 ms` baseline in the native V2 path to `8076 ms`
- `commit_session`: from `2236 ms` to `1146 ms`
- `rebuild_lex`: from `3814 ms` baseline to `565 ms`
- `restore_query`: from multi-second cold restore paths to `44 ms`
- `snapshot_bytes`: down to about `41.3 MB`

These wins came from a sequence of focused changes, not one single rewrite.

## Major Optimizations

### 1. Native Runtime Was Fully Separated From Cozo

Native runtime no longer routes snapshots or hot-path storage through Cozo-era assumptions.

Key results:

- native snapshot import rejects legacy Cozo snapshots
- native runtime no longer depends on Cozo for ingest durability
- wasm and native can evolve independently

This removed design drag that kept distorting native storage decisions.

### 2. Native Durability Was Re-Centered On LMDB

Native storage now treats LMDB as the authoritative durability layer.

Key ideas:

- ordinal-keyed identities for scopes, documents, and sessions
- logical document revisions persisted as manifests plus typed segments
- dirty-scope tracking for lexical rebuild handoff
- sidecars treated as derived caches, not authoritative state

This let ingest stop pretending the native truth was “one monolithic bundle blob.”

### 3. Ingest Was Fixed at the Algorithm Level

The biggest ingest win came from correcting hot-path algorithm shape.

Important changes:

- removed a pathological sentence lookup pattern in `scan_parts`
- added subphase timing to expose real hotspots
- moved session graph counts to manifest-derived estimates instead of live recounts
- stripped non-authoritative archive payloads from native persistence
- stopped cloning large payloads into in-memory archive shapes just to serialize them immediately

This was the turning point that collapsed `ingest_document` from catastrophic timings to something reasonable.

### 4. Native Archive Payload Was Pruned Aggressively

Several archive-shaped segments were useful for historical compatibility but not needed on the native hot path.

Examples removed from native persistence or duplicate in-memory assembly:

- `StructureRelations`
- `StringArena`
- `SentenceTable`
- `MentionTable`
- `ResolverLinkTable`
- `EvidenceTable`
- `NarrativeHitTable`
- duplicate `GraphMutation` payload persistence

This reduced ingest serialization cost and snapshot size.

### 5. Lexical Rebuild Was Moved Off the Ingest Hot Path

Native ingest no longer rebuilds lexical state before returning.

Instead:

- ingest persists canonical docs and graph durability
- touched scopes are marked dirty
- explicit rebuild persists sidecars
- query can use a scope-local fallback

That change removed a major source of ingest-time waste.

### 6. Qgram Was Tightened Before Reusing It on Native Fallback

The qgram engine received a structural cleanup before being reused more heavily.

Key changes:

- prepared-query shape per search
- dense ordinal-local candidate counting instead of hash-map churn
- top-k heap retention instead of sorting all hits
- one-pass bigram plus trigram extraction during reindex

This made qgram a stronger engine for native lexical fallback and future feature work.

### 7. Native Lexical Fallback Now Uses Scope-Local Qgram Caching

The native runtime now uses a bounded scope-local qgram cache instead of rebuilding a global lex index or scanning every span linearly on restore.

Important guardrails:

- tiny scopes still use the cheap linear scan path
- cache invalidation is centralized across ingest, rebuild, and snapshot import
- restore can prewarm scoped qgram caches after import

This is what collapsed `restore_query` back down after the global lex rebuild was removed.

### 8. Commit Path Journal Lookup Was Reduced to O(1)-ish Behavior

`commit_session` had a hidden graph tax: it was scanning the full graph kernel journal to discover the current generation, and the runtime was effectively paying for that lookup twice.

The fix:

- use the newest journal entry via reverse iteration instead of scanning the entire journal
- stop recomputing the same generation twice in `commit()`

This cut `commit_session` nearly in half.

## Crates Used and Why They Matter

These are the crates that materially shaped the optimized native path.

### `heed3`

Primary typed LMDB wrapper for the native durability layer.

Used for:

- native LMDB environment and DB access
- ordered key/value storage
- reverse iteration and prefix iteration
- native row, manifest, segment, and graph durability access

### `lmdb-master3-sys`

Transitive native LMDB sys layer behind `heed3`.

Used for:

- LMDB master3-backed storage behavior
- native mmap-based persistence foundation

### `lz4_flex`

Used in native segment persistence.

Used for:

- fast compression of persisted native segment payloads
- reducing snapshot and on-disk size without making decode unbearably expensive

### `rmp-serde`

Current binary serialization format used for many stored native values and segments.

Used for:

- manifest and segment encoding
- compact binary persistence without JSON in the hot path

### `rayon`

Used in the native ingest pipeline for document-parallel analysis work.

Used for:

- parallel analyzer work before single-writer persistence

### `smallvec`

Used in several hot data shapes.

Used for:

- keeping small collections inline
- reducing heap churn in tight loops
- qgram prepared query and match-related structures

### `rustc-hash`

Used where fast non-cryptographic hashing is useful.

Used for:

- hot-path maps and sets in analysis and query layers

### `roaring`

Used in qgram postings.

Used for:

- bitmap-backed large posting sets
- keeping selective and broad postings efficient under one abstraction

### `daachorse`

Used by lexical and verification paths.

Used for:

- exact Aho-Corasick verification in qgram
- strong exact second-stage matching after gram filtering

### `fst`

Used in lexical and scanner-related indexing layers.

Used for:

- compact lookup-oriented structures in the lexical stack

### `zerocopy`

Used in semantic and native storage layers.

Used for:

- fixed-layout binary-friendly shapes
- helping keep segment and metadata structures close to packed binary form

## What We Learned

The main lesson from this pass is that large wins came from:

- removing legacy architectural coupling
- fixing hidden full scans
- deleting non-authoritative payloads
- separating hot-path work from rebuild work
- adding just enough instrumentation to reveal the real bottlenecks

The biggest traps were:

- assuming LMDB itself was the main bottleneck before proving it
- letting native fallback behavior drift into “works, but scans everything”
- keeping compatibility-shaped payloads alive in a greenfield native path

## Current Direction Boundary

This optimization pass intentionally pushed hard on large-corpus edge-case behavior using `perfect_run.md`.

That was the right thing to do for system integrity, but it should not dominate roadmap thinking forever.

Next direction:

- keep the current optimized native foundation
- stop reopening broad architecture rewrites unless a feature requires it
- return focus to feature work, using this optimized path as the baseline

## Recommended Follow-Up

If another performance pass is needed later, the clean next candidates are:

- `lexical_query_batch` warm-query behavior after rebuild
- any remaining graph query hot spots
- selective serialization upgrades for the heaviest native segment types

But those should now be deliberate follow-ups, not blockers to resuming feature delivery.
