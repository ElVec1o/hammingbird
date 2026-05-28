# Rust port — planned

This directory will hold the production Rust implementation.

## Why Rust

The pure-Python prototype already beats FAISS's optimized C++ flat index by
~90× at n=200k for d=256, k=2 (see `../../logs/`). A Rust implementation
should extend that lead substantially via:

1. **Real SIMD popcount.** AVX-512 has `_mm512_popcnt_epi64`; AVX2 has the
   Mula-Kurz trick. Should give ~10× on the verification step alone.
2. **Cache-friendly chunk hashing.** Replace Python dict with a flat
   hashtable (e.g. `hashbrown::HashMap` with `nohash_hasher` since chunk
   bytes are already random-looking).
3. **Parallel candidate generation.** Each chunk's bucketing is embarrassingly
   parallel — split across threads with `rayon`.
4. **Zero-copy NumPy bridge.** Use `pyo3` + `numpy` crates so Python users
   get the Rust speed via `pip install pigeon`.

## API contract (must match Python)

```rust
pub fn find_pairs_within(
    a: ArrayView2<u8>,
    k: u32,
    b: Option<ArrayView2<u8>>,
) -> Vec<(usize, usize, u32)>
```

## Performance targets (vs current Python at n=200k, d=256, k=2)

| Component | Python (s) | Rust target (s) | Speedup goal |
|-----------|-----------:|----------------:|-------------:|
| Bucket build | ~0.4 | ~0.05 | 8× |
| Candidate gen | ~0.2 | ~0.02 | 10× |
| Verification (popcount) | ~0.2 | ~0.01 | 20× |
| **Total** | **~0.8** | **~0.08** | **~10×** |

Combined with the algorithmic 90× lead over FAISS Flat, this would put
us at **~900× over FAISS Flat** at n=200k. That's a real product story.

## Crate layout (planned)

```
rust/
├── Cargo.toml
├── hammingbird-core/         ← algorithm, no Python bindings
│   ├── Cargo.toml
│   └── src/lib.rs
├── hammingbird-py/           ← PyO3 bindings + wheel build
│   ├── Cargo.toml
│   ├── pyproject.toml
│   └── src/lib.rs
└── pigeon-cli/          ← optional standalone binary
    ├── Cargo.toml
    └── src/main.rs
```

## First task for the new agent

Stand up `hammingbird-core` with a single function matching `find_pairs_within`,
ported line-by-line from `../python/pigeon/core.py`. Reuse the Python tests
as a correctness oracle: same inputs, same outputs.

Then benchmark: re-run `../benchmarks/bench_scale.py` with the Rust-backed
package and confirm the 10× implementation speedup is real.

## Open questions

- **Hash key type.** For chunks > 8 bytes, can't fit in u64. Options:
  `[u8; N]` (compile-time N), `&[u8]` (lifetime hassle), or hash-then-store.
  Probably `SmallVec<[u8; 16]>` for flexibility.
- **k > 1 chunk sizes.** With n_chunks = k+1, chunks of size ⌈d_bytes / (k+1)⌉.
  For d=256, k=2: 11 bytes (88 bits) per chunk. Doesn't fit in u64.
- **Threshold for the prefilter.** At very small n the bucket-building
  overhead might exceed brute force. Profile and pick a threshold below
  which we just do all-pairs.
