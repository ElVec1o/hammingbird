# Changelog

All notable changes to **hammingbird** are documented here.

## 0.5.0 — Initial public release

First release on PyPI. Renamed from internal codename.

### Public API

- `find_pairs_self(A, k)` — all-pairs near-duplicate search, byte-aligned chunks.
- `find_pairs_self_bit(A, k)` — bit-aligned chunks; supports `k < 8*d_bytes`.
- `find_pairs_self_adaptive(A, k)` — entropy-aware chunk planning; opt-in
  for non-uniform / structured signatures.
- `find_pairs_cross(A, B, k)` — cross-corpus pair search (byte-aligned).
- `find_pairs_cross_bit(A, B, k)` — cross-corpus pair search (bit-aligned).
- `Index(d_bytes, k)` — streaming near-duplicate index with `add`,
  `add_batch`, `query`, `query_batch`. Sub-microsecond exact query
  latency (~0.21 µs median at n=100k). Thread-safe for cross-thread
  `Send`; concurrent reads are safe.

### Algorithm

- Pigeonhole prefilter for exact Hamming-distance pair search
  (Norouzi-Punjani-Fleet 2012, *Fast Search in Hamming Space with
  Multi-Index Hashing*, CVPR).
- Default chunk-position-parallel candidate generation; no global
  candidate dedup (the verify step is cheap enough that double-verifying
  a few duplicates beats sorting/deduplicating the full candidate set).
- Bit-level chunks use `u128`-packed keys for the hot path; multiple
  disjoint chunks at wide `d` keep the pigeonhole guarantee intact at
  any `(d, k)` where `k < 8*d_bytes`.
- Parallel verify step uses `hamming_le_k` early-exit popcount.

### Constraints

- Byte-aligned paths (`find_pairs_self`, `find_pairs_cross`) require
  `k < d_bytes` for exact correctness. Violations raise `ValueError` at
  the Python boundary. Bit-aligned paths lift this to `k < 8*d_bytes`.
- Index releases the GIL on `add_batch`; `query` does not (~µs call —
  release overhead would dominate). Concurrent `query()` from multiple
  Python threads is safe; concurrent `add` + `query` on the same Index
  raises `RuntimeError: Already mutably borrowed`.

### Distribution

- Pure-Python reference implementation at `code/python/hammingbird_ref/`
  (~120 LOC, for educational reading / cross-validation).
- Rust core at `code/rust/hammingbird-core/`.
- PyO3 bindings at `code/rust/hammingbird-py/`, distributed as an abi3
  wheel covering Python 3.9–3.13.
- CI builds wheels for Linux x86_64, macOS arm64, and Windows x86_64
  via the workflow at `.github/workflows/wheels.yml`.

### License

MIT. See `LICENSE`.
