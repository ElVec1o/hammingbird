# hammingbird

> **The fastest exact Hamming-distance pair-search library on PyPI.**
> Same algorithm as FAISS `IndexBinaryMultiHash` (Norouzi 2012), with a Rust
> core, rayon-parallel candidate generation, prefetched popcount verify,
> and a streaming `Index` with sub-microsecond query latency.

```sh
pip install hammingbird
```

```python
import numpy as np
from hammingbird import find_pairs_self, Index

# 1M random 256-bit codes
A = np.random.default_rng(0).integers(0, 256, size=(1_000_000, 32), dtype=np.uint8)

# All-pairs near-duplicate search — exact, 100% recall
pairs = find_pairs_self(A, k=2)     # list of (i, j, hamming_dist)

# Streaming index — sub-microsecond per query
idx = Index(d_bytes=32, k=2)
idx.add_batch(A)
hits = idx.query(A[0])              # list of (id, hamming_dist)
```

## Headline numbers (Apple M2 Pro, full reproducibility on GitHub)

| Workload | hammingbird | FAISS IndexBinaryFlat | Speedup |
|---|---|---|---|
| **n=2,000,000, k=0** (exact dedup) | 0.196 s | 798.4 s | **4073×** |
| **n=2,000,000, k=2** (uniform random) | 0.312 s | 919.3 s | **2945×** |
| **n=100k Index query latency** | 0.21 µs median | 194 µs median | **924× per query** |
| **n=100k clustered, k=4** (real-shape) | 0.040 s | 1.755 s | **44×** |

vs FAISS `IndexBinaryMultiHash` (same algorithm) at full recall: **3–45× faster**.
vs usearch / Annoy / FAISS-LSH at clustered n=1M: **strict dominance** —
hammingbird is both faster AND more accurate (those libraries' recall
drops below 12% at scale on this workload).

Full head-to-head benchmark report (16 demo presets, every number
reproducible): **https://github.com/ElVec1o/hammingbird/blob/main/BENCHMARK_RESULTS.md**

## What it's for

- **Exact dedup** at scale (k=0 — the common production workload)
- **Near-duplicate detection** on perceptual hashes, SimHash, learned binary
  embeddings (k ≤ 8 typically)
- **Real-time content moderation** — the streaming `Index` API delivers
  ~1.6M exact queries/sec per thread, releases the GIL on `add_batch`
- **Cross-corpus matching** — `find_pairs_cross(A, B, k)` for "dedup
  incoming batch against blacklist" workloads

## What it's not

- Not a general-purpose ANN library. For approximate-recall workloads at
  k ≥ 16 or single-query latency budgets < 1 µs with k ≥ 8, FAISS's
  HNSW family or usearch may be a better fit.
- Not algorithmically novel. The pigeonhole prefilter is published
  (Norouzi-Punjani-Fleet 2012). The win is implementation quality.

## License

MIT © 2026 Vico Bonfioli.

## Source / issues / docs

https://github.com/ElVec1o/hammingbird
