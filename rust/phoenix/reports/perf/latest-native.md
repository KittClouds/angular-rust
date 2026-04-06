# Phoenix Native Performance Report

Generated at: `1775503194705`
Benchmark iterations: `1` with `0` warmup iteration(s)
Budget failures: `0`

## Perfect Run

- Path: `\\?\C:\Users\shuga\1kittroot\1code\Angular-build\docs\perfect_run.md`
- Input bytes: `2859630`
- Snapshot bytes: `28660326`
- Budget status: `passing`

| Phase | Wall ms | Peak delta MiB | Output bytes |
| --- | ---: | ---: | ---: |
| init_runtime | 14 | 0.00 | 0 |
| create_session | 23 | 0.00 | 0 |
| analyze_text | 5738 | 365.15 | 10536371 |
| excerpt_scan | 140 | 1.42 | 1153104 |
| excerpt_structure | 3 | 1.30 | 762225 |
| ingest_document | 7841 | 369.81 | 3273 |
| commit_session | 733 | 61.00 | 149 |
| rebuild_lex | 533 | 13.73 | 175 |
| session_state | 0 | 0.01 | 8407 |
| session_stats | 0 | 0.01 | 266 |
| graph_delta | 3801 | 144.51 | 1738509 |
| snapshot_export | 4233 | 139.66 | 28660326 |
| snapshot_import | 9466 | 343.91 | 28660326 |
| restore_query | 27 | 0.05 | 693 |
| lexical_query_batch | 4152 | 20.84 | 140 |
| graph_query_batch | 2329 | 22.34 | 99 |

| Benchmark | Iter | Min ms | P50 ms | P95 ms | Max ms | Peak delta MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| scan_excerpt_steady | 1 | 178 | 178 | 178 | 178 | 2.35 |
| structure_excerpt_steady | 1 | 49 | 49 | 49 | 49 | 1.66 |
| session_stats_steady | 1 | 0 | 0 | 0 | 0 | 0.01 |
| graph_delta_steady | 1 | 1711 | 1711 | 1711 | 1711 | 144.51 |
| lexical_query_steady | 1 | 357 | 357 | 357 | 357 | 0.08 |
| graph_query_steady | 1 | 2308 | 2308 | 2308 | 2308 | 22.34 |
| restore_query_steady | 1 | 32 | 32 | 32 | 32 | 0.05 |
