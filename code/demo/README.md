# pigeon — interactive benchmark demo

A Streamlit app for running pigeon side-by-side with industry-standard
near-duplicate libraries on Hamming pair search. **You** drive `n`, `d`,
and `k`. The app shows a memory estimate before you commit, and writes
no hard ceilings — if the run blows up, hit Ctrl-C in the terminal.

## Install (one-time)

```sh
pip install streamlit pandas usearch annoy
# faiss-cpu and hammingbird should already be installed; verify:
python3 -c "import hammingbird, faiss, usearch, annoy; print('ok')"
```

If any of `usearch` or `annoy` fail to install, the app will still run
— that method just disappears from the available list.

## Run

From the repo root:

```sh
streamlit run code/demo/app.py
```

A browser tab opens at `http://localhost:8501`. Pick a dataset, methods,
and `k`, then click **Run benchmark**.

## What's compared

### Exact methods (always 100% recall)
- **pigeon `find_pairs_self`** (byte-aligned chunks, the v0.4.3 default)
- **pigeon `find_pairs_self_bit`** (bit-aligned chunks, works at `k ≥ d_bytes`)
- **pigeon `find_pairs_self_adaptive`** (entropy-aware chunks, wins on structured data)
- **FAISS `IndexBinaryFlat`** — brute-force O(n²), the universal exact baseline
- **FAISS `IndexBinaryMultiHash`** with `nflip=0` — FAISS's own pigeonhole-style index

### Approximate methods (recall reported vs the exact ground truth)
- **FAISS `IndexBinaryHash`** — LSH variant
- **usearch HNSW** with Hamming metric — modern competitor, supports binary natively
- **Annoy** with Hamming metric — Spotify's classical ANN library

## Datasets

- **Uniform random** — the hardest case for prefilters; every pair is far. Tests pure overhead.
- **Clustered (real-world shape)** — 10 centroids, 10% near-duplicates. Closer to how perceptual hashes look.
- **Low-entropy (adversarial)** — half the bits constant. Shows where naive chunkers degenerate and adaptive shines.
- **Real CIFAR-10 pHashes** — 50,000 64-bit perceptual hashes from CIFAR-10 train (requires `code/experiments/_real_data/phashes_cifar10_50000.bin`; regenerate with `python3 code/experiments/gen_real_phashes.py`).

## Reading the results

| Column            | Meaning                                       |
|-------------------|-----------------------------------------------|
| `method`          | Method name                                   |
| `category`        | `exact` / `approximate-default` / `approximate-tunable` |
| `time_s`          | Wall-clock in seconds (lower is better)       |
| `pairs`           | Number of pairs found                         |
| `recall_vs_truth` | Fraction of true pairs recovered (1.0 = exact). `None` if no ground truth available. |
| `extra`           | Method-specific parameters (e.g. `{"nflip": 0}`) |

Ground truth is whichever exact method ran first in priority order:
FAISS Flat → pigeon byte → pigeon bit. Approximate methods are scored
against it.

## Memory & safety

The sidebar shows a rough peak-memory estimate (🟢/🟡/🟠/🔴) before you
hit Run. It includes the dataset, prefilter hashmaps, candidate sets,
FAISS index, and ~200 MB of Streamlit overhead.

- 🟢 under 1 GB — safe on any machine
- 🟡 1–4 GB — fine on a normal laptop
- 🟠 4–12 GB — close other apps first
- 🔴 over 12 GB — only if you know what you're doing

The app does NOT enforce these. You may run at any `n`. The reason: we
specifically want you to push the benchmark beyond what was safe in
agent-driven sessions earlier in the project. **Watch the terminal.**

## Scaling curves

If you run the same `(method, d_bits, k)` at multiple `n` values, a
line chart appears automatically. This is how you build the "does
pigeon's lead grow with n?" plot.

## Reset

The 🗑 **Clear run history** button in the sidebar wipes the session.
Closing the browser tab without hitting Clear preserves the session so
you can come back later.
