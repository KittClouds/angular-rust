# Phoenix Native Performance Report

Generated at: `1775520916692`
Benchmark iterations: `1` with `0` warmup iteration(s)
Budget failures: `0`

## Perfect Run

- Path: `\\?\C:\Users\shuga\1kittroot\1code\Angular-build\docs\perfect_run.md`
- Input bytes: `2859630`
- Snapshot bytes: `25171001`
- Budget status: `passing`

| Phase | Wall ms | Peak delta MiB | Output bytes |
| --- | ---: | ---: | ---: |
| init_runtime | 1 | 0.00 | 0 |
| create_session | 1 | 0.00 | 0 |
| analyze_text | 5265 | 365.15 | 10536371 |
| excerpt_scan | 142 | 1.42 | 1153104 |
| excerpt_structure | 3 | 1.30 | 762225 |
| ingest_document | 6414 | 319.60 | 3272 |
| commit_session | 625 | 55.67 | 149 |
| rebuild_lex | 509 | 13.54 | 175 |
| session_state | 0 | 0.01 | 8407 |
| session_stats | 0 | 0.01 | 265 |
| graph_delta | 3086 | 120.81 | 1542092 |
| snapshot_export | 3797 | 128.23 | 25171001 |
| snapshot_import | 8643 | 297.68 | 25171001 |
| restore_query | 35 | 0.05 | 693 |
| lexical_query_batch | 4204 | 20.78 | 140 |
| graph_query_batch | 2230 | 21.21 | 99 |

| Benchmark | Iter | Min ms | P50 ms | P95 ms | Max ms | Peak delta MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| scan_excerpt_steady | 1 | 193 | 193 | 193 | 193 | 2.35 |
| structure_excerpt_steady | 1 | 48 | 48 | 48 | 48 | 1.66 |
| session_stats_steady | 1 | 0 | 0 | 0 | 0 | 0.01 |
| graph_delta_steady | 1 | 1415 | 1415 | 1415 | 1415 | 120.81 |
| lexical_query_steady | 1 | 361 | 361 | 361 | 361 | 0.08 |
| graph_query_steady | 1 | 2215 | 2215 | 2215 | 2215 | 21.21 |
| restore_query_steady | 1 | 34 | 34 | 34 | 34 | 0.05 |
