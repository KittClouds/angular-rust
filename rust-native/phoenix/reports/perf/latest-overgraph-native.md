# Phoenix OverGraph Native Harness Report

Generated at: `1775533061919`
Corpus: `perfect_run` from `\\?\C:\Users\shuga\1kittroot\1code\Angular-build\docs\perfect_run.md`
Input bytes: `2859630`
Benchmark iterations: `1` with `0` warmup iteration(s)

| Phase | Wall ms | Peak delta MiB | Output bytes |
| --- | ---: | ---: | ---: |
| init_overgraph_store | 15 | 0.01 | 0 |
| overgraph_native_ingest | 5589 | 136.54 | 2057 |
| rebuild_scope_sidecars | 1067 | 18.62 | 1 |
| persist_session_archive | 2 | 0.04 | 8729 |
| verify_persistence_truth | 450 | 23.14 | 243 |

| Benchmark | Iter | Min ms | P50 ms | P95 ms | Max ms | Peak delta MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| overgraph_native_ingest | 1 | 5589 | 5589 | 5589 | 5589 | 136.54 |

## Persistence Truth

- Manifest present: `true`
- Archive count: `1`
- Dirty scopes before rebuild: `1`
- Rebuilt scope count: `1`
- Lexical span count: `6298`
- Session archive present: `true`
- Kernel checkpoint present: `true`
- Subsequent prepare has kernel snapshot: `true`
- Kernel generation: `1`