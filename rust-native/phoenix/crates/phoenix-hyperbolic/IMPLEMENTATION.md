# Hyperbolic HNSW Implementation Summary

## Overview
Successfully implemented and tested a Hyperbolic HNSW (Hierarchical Navigable Small World) index with DiskANN-style memory-mapped file persistence.

## Implementation Details

### Location
- **Crate**: `rust/phoenix/crates/phoenix-hyperbolic`
- **Main file**: `src/lib.rs`

### Key Components

1. **Metric System**
   - `MetricF32` trait for hyperbolic distance calculations
   - `PoincareMetric` implementation using Poincaré ball model
   - Proper handling of curvature and boundary conditions

2. **In-Memory Builder** (`HyperbolicHnswBuilder`)
   - Layered graph construction with HNSW algorithm
   - Heuristic neighbor selection to prevent edge occlusion
   - Configurable parameters (m, m0, ef_construction, level_mult)
   - Random level assignment for new nodes

3. **Disk Persistence**
   - Single-file binary format with memory-mapped access
   - Efficient CSR (Compressed Sparse Row) layout for adjacency lists
   - Metadata header with offsets for all data sections
   - 1MB padding at file start for metadata

4. **Memory-Mapped Reader** (`HyperbolicDiskHnsw`)
   - Zero-copy vector access via mmap
   - Fast neighbor traversal using byte-level parsing
   - Two-phase search: top-down descent + layer-0 beam search
   - Vector caching to avoid alignment issues

### File Format
```
[Metadata Length (8 bytes)]
[Metadata (variable)]
[Padding to 1MB]
[Vectors (f32 array)]
[Levels (u32 array)]
[Offsets (u64 array)]
[Adjacency (u32 array)]
```

### Search Algorithm
1. **Phase 1**: Greedy descent from max level to level 1
2. **Phase 2**: Beam search at level 0 with ef_search candidates
3. **Result**: Top-k nearest neighbors sorted by distance

## Tests

All 6 tests pass successfully:

1. **test_poincare_metric**: Validates distance calculation and projection
2. **test_metric_symmetry**: Ensures d(a,b) = d(b,a)
3. **test_metric_triangle_inequality**: Verifies triangle inequality holds
4. **test_empty_index**: Handles edge case of empty index
5. **test_single_vector**: Tests index with single vector
6. **test_build_and_search**: Full integration test with 100 vectors

### Test Results
```
running 6 tests
test tests::test_metric_symmetry ... ok
test tests::test_metric_triangle_inequality ... ok
test tests::test_poincare_metric ... ok
test tests::test_empty_index ... ok
test tests::test_single_vector ... ok
test tests::test_build_and_search ... ok

test result: ok. 6 passed; 0 failed
```

## Key Design Decisions

1. **Alignment Handling**: Used manual byte-to-f32 conversion instead of bytemuck to avoid alignment issues with mmap
2. **Vector Caching**: Pre-loaded vectors into Vec<Vec<f32>> to avoid repeated conversions
3. **File Size**: Ensured minimum file size for proper mmap operation even with empty indices
4. **Memory Efficiency**: CSR layout for compact graph storage with fast sequential access

## Dependencies
- memmap2: Memory-mapped file I/O
- rand: Random level generation
- serde/bincode: Metadata serialization
- bytemuck: Safe byte casting (where alignment permits)
- thiserror: Error handling

## Usage Example
```rust
// Build index
let mut builder = HyperbolicHnswBuilder::new(
    dim,
    PoincareMetric { curvature: 1.0 },
    HnswBuildParams::default()
);

for vector in vectors {
    builder.insert(vector);
}

builder.save_to_disk("index.bin")?;

// Search index
let index = HyperbolicDiskHnsw::open(
    "index.bin",
    PoincareMetric { curvature: 1.0 }
)?;

let results = index.search(&query, k=10, ef_search=50);
```

## Performance Characteristics
- **Build Time**: O(n log n) with heuristic pruning
- **Search Time**: O(log n) average case with HNSW layers
- **Memory Usage**: O(n) for mmap + O(n) for vector cache
- **Disk Usage**: ~1MB + n * (dim * 4 + avg_edges * 4) bytes

## Future Improvements
1. Add support for incremental updates
2. Implement batch insertion optimization
3. Add compression for vectors (e.g., scalar quantization)
4. Support for different distance metrics
5. Async I/O for large indices
