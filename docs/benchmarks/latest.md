# Latest recommendation benchmark

Generated: `2026-07-13T14:01:08.053860+00:00`. Results contain IDs and aggregates only; query and candidate text are intentionally omitted.

## Environment and models

- Embedding: `intfloat/multilingual-e5-small` (configured revision: `absent`, resolved commit: `614241f622f53c4eeff9890bdc4f31cfecc418b3`)
- Reranker: `BAAI/bge-reranker-v2-m3` (configured revision: `absent`, resolved commit: `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`)
- Embedding dimensions: 384; fixture SHA-256: `e73eba5ee6d2f08c38d9e85d0029eba2d781367055ed3946c32941c76ac1c05b`
- OS: `Windows-11-10.0.26200-SP0`; CPU: `Intel64 Family 6 Model 183 Stepping 1, GenuineIntel`; Python: `3.12.13 (main, Mar  3 2026, 15:01:35) [MSC v.1944 64 bit (AMD64)]`

## Vector-only quality

| Language | Queries | Relevant candidates | Recall@10 | MRR |
| --- | ---: | ---: | ---: | ---: |
| overall | 15 | 54 | 0.9444 | 0.9667 |
| az | 5 | 18 | 0.9667 | 1.0000 |
| tr | 5 | 18 | 0.8667 | 0.9000 |
| en | 5 | 18 | 1.0000 | 1.0000 |

## Reranker comparison

| Language | Vector Recall@10 | Reranked Recall@10 | Δ Recall@10 | Vector MRR | Reranked MRR | Δ MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| overall | 0.9444 | 1.0000 | +0.0556 | 0.9667 | 1.0000 | +0.0333 |
| az | 0.9667 | 1.0000 | +0.0333 | 1.0000 | 1.0000 | +0.0000 |
| tr | 0.8667 | 1.0000 | +0.1333 | 0.9000 | 1.0000 | +0.1000 |
| en | 1.0000 | 1.0000 | +0.0000 | 1.0000 | 1.0000 | +0.0000 |

Query outcomes: {'improved': 1, 'unchanged': 14, 'worse': 0}.

## Latency and process RSS

Load durations exclude downloads from steady-state inference measurements. RSS values are process RSS in MiB; a background sampler runs during model loading and inference, so peak is the true observed process RSS during those operations.

- Embedding model load: 7.122 s; reranker model load: 1.6458444999998392 s.

| Operation | Runs | Median ms | P95 ms | Min ms | Max ms | Items/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| embedding single_query | 5 | 11.569 | 11.911 | 10.853 | 11.911 | 87.259 |
| embedding batch_8 | 5 | 27.655 | 29.598 | 27.159 | 29.598 | 285.480 |
| embedding batch_32 | 5 | 83.845 | 84.425 | 83.179 | 84.425 | 381.475 |
| reranker 5 candidates | 5 | 264.803 | 270.540 | 258.795 | 270.540 | 18.856 |
| reranker 10 candidates | 5 | 464.289 | 483.831 | 456.046 | 483.831 | 21.430 |
| reranker 20 candidates | 5 | 2519.850 | 2741.026 | 878.454 | 2741.026 | 9.189 |
| reranker 50 candidates | 5 | 6385.702 | 6443.671 | 6341.770 | 6443.671 | 7.821 |

- Baseline RSS: 30.88 MiB; after embedding load: 594.33 MiB; observed peak during embedding load: 641.04 MiB; observed peak during embedding inference: 2159.35 MiB.
- Reranker after-load RSS: 852.33; observed peak during reranker load: 852.33; observed peak during reranker inference: 2160.07; true observed process peak: 2160.07 MiB; peak delta: 2129.19 MiB.

## Recommendation

**Keep reranker disabled** — Recall@10 change was +0.0556, MRR change was +0.0333, p95 at 50 candidates was 6443.671 ms, and observed RSS delta was 2129.19 MiB. At least one keep-disabled threshold was met.

Thresholds: Enable requires Recall@10 and MRR gains of at least 0.0200, p95 at 50 candidates at most 750 ms, and observed RSS delta at most 768 MiB. Keep disabled applies for no quality gain, p95 above 2000 ms, or RSS delta above 1536 MiB; other results require more evidence.

## Limitations

This is a small deterministic regression fixture (five queries per language), not a production relevance study. A 10 ms sampler greatly improves peak RSS observation but cannot prove an instantaneous native allocation did not occur between samples.
