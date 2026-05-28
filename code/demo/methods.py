"""Uniform method wrappers for the pigeon demo.

Each method exposes:
  - name (str): display name
  - category (str): "exact" / "approximate-tunable" / "approximate-default"
  - available (bool): whether the underlying library imports cleanly
  - run_pairs(A, k) -> dict with 'time', 'pairs' (set of (i,j) with i<j), 'extra'

Some methods (LSH, HNSW) are approximate by design — they report a recall
relative to the exact ground truth (computed by FAISS Flat).
"""
from __future__ import annotations
import time
import numpy as np
from typing import Optional

# ---- soft imports ----------------------------------------------------------
try:
    import hammingbird  # type: ignore
    _HAS_PIGEON = True
except Exception:
    _HAS_PIGEON = False

try:
    import faiss  # type: ignore
    _HAS_FAISS = True
except Exception:
    _HAS_FAISS = False

try:
    from usearch.index import Index as USearchIndex, MetricKind as USearchMetric  # type: ignore
    _HAS_USEARCH = True
except Exception:
    _HAS_USEARCH = False

try:
    from annoy import AnnoyIndex  # type: ignore
    _HAS_ANNOY = True
except Exception:
    _HAS_ANNOY = False


def _faiss_flat_pairs(A: np.ndarray, k: int):
    d_bits = A.shape[1] * 8
    idx = faiss.IndexBinaryFlat(d_bits)
    idx.add(A)
    lims, _, ids = idx.range_search(A, k + 1)  # FAISS uses radius +1 semantics
    pairs = set()
    n = A.shape[0]
    for i in range(n):
        for jx in range(lims[i], lims[i + 1]):
            j = int(ids[jx])
            if j > i:
                pairs.add((i, j))
    return pairs


def _faiss_multihash_pairs(A: np.ndarray, k: int, nflip: int = 0):
    """Build FAISS IndexBinaryMultiHash with a valid (nhash, b) tuple.

    FAISS asserts `nhash * b <= d`. The pigeonhole lemma says nhash = k+1
    suffices for 100% recall at nflip=0. We pick nhash = max(k+1, 2) and
    let b = d_bits // nhash (floor division → invariant holds by construction).
    Also clamp b to 32, since FAISS's hash code uses 32-bit signatures
    internally for IndexBinaryHash-family indices.
    """
    d_bits = A.shape[1] * 8
    n_hashes = max(k + 1, 2)
    b = max(1, min(32, d_bits // n_hashes))
    # Defensive: if floor-div somehow leaves slack, do not push n_hashes up.
    assert n_hashes * b <= d_bits, f"invalid faiss MH config nhash={n_hashes} b={b} d={d_bits}"
    idx = faiss.IndexBinaryMultiHash(d_bits, n_hashes, b)
    idx.nflip = nflip
    idx.add(A)
    lims, _, ids = idx.range_search(A, k + 1)
    pairs = set()
    n = A.shape[0]
    for i in range(n):
        for jx in range(lims[i], lims[i + 1]):
            j = int(ids[jx])
            if j > i:
                pairs.add((i, j))
    return pairs


def _faiss_hash_pairs(A: np.ndarray, k: int, nflip: int = 0):
    """FAISS IndexBinaryHash (LSH). At nflip=0 it's heavily approximate;
    higher nflip = more candidates per query = higher recall, slower."""
    d_bits = A.shape[1] * 8
    b = 16  # bits per hash table
    idx = faiss.IndexBinaryHash(d_bits, b)
    idx.nflip = nflip
    idx.add(A)
    lims, _, ids = idx.range_search(A, k + 1)
    pairs = set()
    n = A.shape[0]
    for i in range(n):
        for jx in range(lims[i], lims[i + 1]):
            j = int(ids[jx])
            if j > i:
                pairs.add((i, j))
    return pairs


def _faiss_multihash_pairs_tuned(A: np.ndarray, k: int):
    """FAISS MultiHash with aggressive nflip for higher recall on clustered data."""
    return _faiss_multihash_pairs(A, k, nflip=2)


def _faiss_hash_pairs_tuned(A: np.ndarray, k: int):
    """FAISS LSH with nflip=4 — much higher recall, slower."""
    return _faiss_hash_pairs(A, k, nflip=4)


def _usearch_pairs_tuned(A: np.ndarray, k: int):
    """usearch with aggressive ef_search + larger neighbor count for high recall."""
    n, d_bytes = A.shape
    d_bits = d_bytes * 8
    idx = USearchIndex(
        ndim=d_bits, metric=USearchMetric.Hamming, dtype="b1",
        connectivity=32, expansion_add=128, expansion_search=256,
    )
    idx.add(np.arange(n, dtype=np.uint64), A)
    pairs = set()
    matches = idx.search(A, count=200, exact=False)
    keys_2d = matches.keys
    dists_2d = matches.distances
    for i in range(n):
        for c in range(keys_2d.shape[1]):
            j = int(keys_2d[i, c])
            if j == i or dists_2d[i, c] > k:
                continue
            a, b = (i, j) if i < j else (j, i)
            pairs.add((a, b))
    return pairs


def _annoy_pairs_tuned(A: np.ndarray, k: int):
    """Annoy with 50 trees + search_k=10000 — much higher recall."""
    n, d_bytes = A.shape
    d_bits = d_bytes * 8
    bits = np.unpackbits(A, axis=1).astype(np.float32)
    idx = AnnoyIndex(d_bits, "hamming")
    for i in range(n):
        idx.add_item(i, bits[i])
    idx.build(50)
    pairs = set()
    for i in range(n):
        nn, dists = idx.get_nns_by_item(i, 200, search_k=10000, include_distances=True)
        for j, d in zip(nn, dists):
            if j == i or d > k:
                continue
            a, b = (i, j) if i < j else (j, i)
            pairs.add((a, b))
    return pairs


def _usearch_pairs(A: np.ndarray, k: int):
    """Build a usearch binary index, search each row, collect pairs."""
    n, d_bytes = A.shape
    d_bits = d_bytes * 8
    idx = USearchIndex(ndim=d_bits, metric=USearchMetric.Hamming, dtype="b1")
    idx.add(np.arange(n, dtype=np.uint64), A)
    # For each row search the top-50 neighbors, filter by hamming <= k.
    pairs = set()
    matches = idx.search(A, count=50, exact=False)
    keys_2d = matches.keys
    dists_2d = matches.distances
    for i in range(n):
        for c in range(keys_2d.shape[1]):
            j = int(keys_2d[i, c])
            if j == i or dists_2d[i, c] > k:
                continue
            a, b = (i, j) if i < j else (j, i)
            pairs.add((a, b))
    return pairs


def _annoy_pairs(A: np.ndarray, k: int):
    """Annoy's Hamming metric works on float vectors of 0/1. Slow build but
    well-known industry comparison point."""
    n, d_bytes = A.shape
    d_bits = d_bytes * 8
    bits = np.unpackbits(A, axis=1).astype(np.float32)
    idx = AnnoyIndex(d_bits, "hamming")
    for i in range(n):
        idx.add_item(i, bits[i])
    idx.build(10)  # 10 trees — reasonable default
    pairs = set()
    for i in range(n):
        nn, dists = idx.get_nns_by_item(i, 50, include_distances=True)
        for j, d in zip(nn, dists):
            if j == i or d > k:
                continue
            a, b = (i, j) if i < j else (j, i)
            pairs.add((a, b))
    return pairs


def _time(fn, *args, **kwargs):
    t = time.perf_counter()
    result = fn(*args, **kwargs)
    return time.perf_counter() - t, result


# ---- registered methods ----------------------------------------------------
METHODS = []


def _register(name, category, available, runner):
    METHODS.append({"name": name, "category": category, "available": available, "run": runner})


# pigeon variants (exact)
def _run_pigeon_self(A, k):
    elapsed, raw = _time(hammingbird.find_pairs_self, A, k)
    return {"time": elapsed, "pairs": {(i, j) for (i, j, _) in raw}, "extra": {}}

def _run_pigeon_bit(A, k):
    elapsed, raw = _time(hammingbird.find_pairs_self_bit, A, k)
    return {"time": elapsed, "pairs": {(i, j) for (i, j, _) in raw}, "extra": {}}

def _run_pigeon_adaptive(A, k):
    elapsed, raw = _time(hammingbird.find_pairs_self_adaptive, A, k)
    return {"time": elapsed, "pairs": {(i, j) for (i, j, _) in raw}, "extra": {}}

# FAISS variants
def _run_faiss_flat(A, k):
    elapsed, pairs = _time(_faiss_flat_pairs, A, k)
    return {"time": elapsed, "pairs": pairs, "extra": {}}

def _run_faiss_multihash(A, k):
    elapsed, pairs = _time(_faiss_multihash_pairs, A, k, 0)
    return {"time": elapsed, "pairs": pairs, "extra": {"nflip": 0}}

def _run_faiss_hash(A, k):
    elapsed, pairs = _time(_faiss_hash_pairs, A, k)
    return {"time": elapsed, "pairs": pairs, "extra": {}}

def _run_usearch(A, k):
    elapsed, pairs = _time(_usearch_pairs, A, k)
    return {"time": elapsed, "pairs": pairs, "extra": {}}

def _run_annoy(A, k):
    elapsed, pairs = _time(_annoy_pairs, A, k)
    return {"time": elapsed, "pairs": pairs, "extra": {"trees": 10}}

# Tuned variants — same libraries, parameters cranked for higher recall.
def _run_faiss_multihash_tuned(A, k):
    elapsed, pairs = _time(_faiss_multihash_pairs_tuned, A, k)
    return {"time": elapsed, "pairs": pairs, "extra": {"nflip": 2}}

def _run_faiss_hash_tuned(A, k):
    elapsed, pairs = _time(_faiss_hash_pairs_tuned, A, k)
    return {"time": elapsed, "pairs": pairs, "extra": {"nflip": 4}}

def _run_usearch_tuned(A, k):
    elapsed, pairs = _time(_usearch_pairs_tuned, A, k)
    return {"time": elapsed, "pairs": pairs, "extra": {"ef_search": 256, "count": 200}}

def _run_annoy_tuned(A, k):
    elapsed, pairs = _time(_annoy_pairs_tuned, A, k)
    return {"time": elapsed, "pairs": pairs, "extra": {"trees": 50, "search_k": 10000}}

# Streaming Index methods — these measure per-query latency, not all-pairs.
def _run_pigeon_index_query(A, k):
    """Build pigeon Index, hold out 1000 vectors, time each query
    with perf_counter_ns. Reports median/mean/p99 latency."""
    n = A.shape[0]
    n_q = min(1000, max(10, n // 50))
    rng = np.random.default_rng(42)
    q_idx = rng.choice(n, size=n_q, replace=False)
    queries = np.ascontiguousarray(A[q_idx])
    mask = np.ones(n, dtype=bool); mask[q_idx] = False
    train = np.ascontiguousarray(A[mask])
    t_build_0 = time.perf_counter()
    idx = hammingbird.Index(d_bytes=A.shape[1], k=k)
    idx.add_batch(train)
    build_s = time.perf_counter() - t_build_0
    # Warm-up.
    for q in queries[:10]:
        idx.query(q)
    latencies_ns = []
    t_total_0 = time.perf_counter()
    for q in queries:
        t0 = time.perf_counter_ns()
        idx.query(q)
        latencies_ns.append(time.perf_counter_ns() - t0)
    total_s = time.perf_counter() - t_total_0
    latencies_ns.sort()
    return {
        "time": total_s,
        "pairs": set(),  # not an all-pairs method
        "extra": {
            "mode": "index_query",
            "build_s": round(build_s, 3),
            "n_queries": n_q,
            "median_us": round(latencies_ns[n_q // 2] / 1000, 2),
            "mean_us": round(sum(latencies_ns) / n_q / 1000, 2),
            "p99_us": round(latencies_ns[int(n_q * 0.99)] / 1000, 2),
            "min_us": round(latencies_ns[0] / 1000, 2),
        },
    }


def _run_faiss_flat_index_query(A, k):
    """FAISS IndexBinaryFlat used as a streaming index: range_search per query."""
    n = A.shape[0]
    n_q = min(1000, max(10, n // 50))
    rng = np.random.default_rng(42)
    q_idx = rng.choice(n, size=n_q, replace=False)
    queries = A[q_idx]
    mask = np.ones(n, dtype=bool); mask[q_idx] = False
    train = A[mask]
    d_bits = A.shape[1] * 8
    t_build_0 = time.perf_counter()
    idx = faiss.IndexBinaryFlat(d_bits)
    idx.add(train)
    build_s = time.perf_counter() - t_build_0
    # Warm-up.
    for q in queries[:10]:
        idx.range_search(q.reshape(1, -1), k + 1)
    latencies_ns = []
    t_total_0 = time.perf_counter()
    for q in queries:
        q2 = q.reshape(1, -1)
        t0 = time.perf_counter_ns()
        idx.range_search(q2, k + 1)
        latencies_ns.append(time.perf_counter_ns() - t0)
    total_s = time.perf_counter() - t_total_0
    latencies_ns.sort()
    return {
        "time": total_s,
        "pairs": set(),
        "extra": {
            "mode": "index_query",
            "build_s": round(build_s, 3),
            "n_queries": n_q,
            "median_us": round(latencies_ns[n_q // 2] / 1000, 2),
            "mean_us": round(sum(latencies_ns) / n_q / 1000, 2),
            "p99_us": round(latencies_ns[int(n_q * 0.99)] / 1000, 2),
            "min_us": round(latencies_ns[0] / 1000, 2),
        },
    }


def _run_usearch_index_query(A, k):
    """usearch HNSW used as a streaming index: search() per query."""

    n = A.shape[0]
    n_q = min(1000, max(10, n // 50))
    rng = np.random.default_rng(42)
    q_idx = rng.choice(n, size=n_q, replace=False)
    queries = A[q_idx]
    mask = np.ones(n, dtype=bool); mask[q_idx] = False
    train = A[mask]
    d_bits = A.shape[1] * 8
    t_build_0 = time.perf_counter()
    idx = USearchIndex(ndim=d_bits, metric=USearchMetric.Hamming, dtype="b1")
    idx.add(np.arange(train.shape[0], dtype=np.uint64), train)
    build_s = time.perf_counter() - t_build_0
    for q in queries[:10]:
        idx.search(q, count=50, exact=False)
    latencies_ns = []
    t_total_0 = time.perf_counter()
    for q in queries:
        t0 = time.perf_counter_ns()
        idx.search(q, count=50, exact=False)
        latencies_ns.append(time.perf_counter_ns() - t0)
    total_s = time.perf_counter() - t_total_0
    latencies_ns.sort()
    return {
        "time": total_s,
        "pairs": set(),
        "extra": {
            "mode": "index_query",
            "build_s": round(build_s, 3),
            "n_queries": n_q,
            "median_us": round(latencies_ns[n_q // 2] / 1000, 2),
            "mean_us": round(sum(latencies_ns) / n_q / 1000, 2),
            "p99_us": round(latencies_ns[int(n_q * 0.99)] / 1000, 2),
            "min_us": round(latencies_ns[0] / 1000, 2),
        },
    }


_register("pigeon — find_pairs_self (byte-aligned)",        "exact", _HAS_PIGEON, _run_pigeon_self)
_register("pigeon — find_pairs_self_bit (bit-aligned)",     "exact", _HAS_PIGEON, _run_pigeon_bit)
_register("pigeon — find_pairs_self_adaptive (entropy)",    "exact", _HAS_PIGEON, _run_pigeon_adaptive)
_register("FAISS IndexBinaryFlat (brute force, exact)",     "exact", _HAS_FAISS,  _run_faiss_flat)
_register("FAISS IndexBinaryMultiHash (exact, nflip=0)",    "exact", _HAS_FAISS,  _run_faiss_multihash)
_register("FAISS IndexBinaryHash (LSH, approximate)",       "approximate-default", _HAS_FAISS, _run_faiss_hash)
_register("usearch HNSW (binary, approximate)",             "approximate-default", _HAS_USEARCH, _run_usearch)
_register("Annoy (random projection, approximate)",         "approximate-default", _HAS_ANNOY, _run_annoy)

# Tuned variants — same libraries with parameters cranked for higher recall.
_register("FAISS MultiHash TUNED (nflip=2)",                "approximate-tunable", _HAS_FAISS, _run_faiss_multihash_tuned)
_register("FAISS LSH TUNED (nflip=4)",                      "approximate-tunable", _HAS_FAISS, _run_faiss_hash_tuned)
_register("usearch HNSW TUNED (ef=256, count=200)",         "approximate-tunable", _HAS_USEARCH, _run_usearch_tuned)
_register("Annoy TUNED (50 trees, search_k=10k)",           "approximate-tunable", _HAS_ANNOY, _run_annoy_tuned)

# Streaming Index methods — per-query latency, not all-pairs.
_register("pigeon Index (streaming query latency)",         "streaming-index", _HAS_PIGEON, _run_pigeon_index_query)
_register("FAISS Flat (streaming query latency)",           "streaming-index", _HAS_FAISS, _run_faiss_flat_index_query)
_register("usearch HNSW (streaming query latency)",         "streaming-index", _HAS_USEARCH, _run_usearch_index_query)

# ============================================================================
# CROSS-JOIN METHODS — "dedup set B against blacklist A"
#
# All cross-join methods auto-split the input A into two halves (corpus + queries)
# so they fit the unified `run(A, k) -> {...}` API. Pairs returned are
# (i_in_first_half, j_in_second_half) — comparing recall across methods
# works because they all use the same indexing convention.
# ============================================================================

def _split_in_half(A):
    n_a = A.shape[0] // 2
    return np.ascontiguousarray(A[:n_a]), np.ascontiguousarray(A[n_a:])


def _pigeon_cross_byte(A_part, B_part, k):
    raw = hammingbird.find_pairs_cross(A_part, B_part, k)
    return {(i, j) for (i, j, _) in raw}


def _pigeon_cross_bit(A_part, B_part, k):
    raw = hammingbird.find_pairs_cross_bit(A_part, B_part, k)
    return {(i, j) for (i, j, _) in raw}


def _faiss_flat_cross(A_part, B_part, k):
    d_bits = A_part.shape[1] * 8
    idx = faiss.IndexBinaryFlat(d_bits)
    idx.add(A_part)
    lims, _, ids = idx.range_search(B_part, k + 1)
    pairs = set()
    n_b = B_part.shape[0]
    for j in range(n_b):
        for ix in range(lims[j], lims[j + 1]):
            i = int(ids[ix])
            pairs.add((i, j))
    return pairs


def _faiss_multihash_cross(A_part, B_part, k):
    d_bits = A_part.shape[1] * 8
    n_hashes = max(k + 1, 2)
    b = max(1, min(32, d_bits // n_hashes))
    idx = faiss.IndexBinaryMultiHash(d_bits, n_hashes, b)
    idx.nflip = 0
    idx.add(A_part)
    lims, _, ids = idx.range_search(B_part, k + 1)
    pairs = set()
    n_b = B_part.shape[0]
    for j in range(n_b):
        for ix in range(lims[j], lims[j + 1]):
            i = int(ids[ix])
            pairs.add((i, j))
    return pairs


def _usearch_cross(A_part, B_part, k):
    d_bits = A_part.shape[1] * 8
    n_a = A_part.shape[0]
    idx = USearchIndex(ndim=d_bits, metric=USearchMetric.Hamming, dtype="b1")
    idx.add(np.arange(n_a, dtype=np.uint64), A_part)
    pairs = set()
    matches = idx.search(B_part, count=50, exact=False)
    keys_2d = matches.keys
    dists_2d = matches.distances
    n_b = B_part.shape[0]
    for j in range(n_b):
        for c in range(keys_2d.shape[1]):
            if dists_2d[j, c] > k:
                continue
            i = int(keys_2d[j, c])
            pairs.add((i, j))
    return pairs


def _run_pigeon_cross_byte(A, k):
    Ap, Bp = _split_in_half(A)
    elapsed, pairs = _time(_pigeon_cross_byte, Ap, Bp, k)
    return {"time": elapsed, "pairs": pairs,
            "extra": {"n_a": Ap.shape[0], "n_b": Bp.shape[0], "mode": "cross-join"}}


def _run_pigeon_cross_bit(A, k):
    Ap, Bp = _split_in_half(A)
    elapsed, pairs = _time(_pigeon_cross_bit, Ap, Bp, k)
    return {"time": elapsed, "pairs": pairs,
            "extra": {"n_a": Ap.shape[0], "n_b": Bp.shape[0], "mode": "cross-join"}}


def _run_faiss_flat_cross(A, k):
    Ap, Bp = _split_in_half(A)
    elapsed, pairs = _time(_faiss_flat_cross, Ap, Bp, k)
    return {"time": elapsed, "pairs": pairs,
            "extra": {"n_a": Ap.shape[0], "n_b": Bp.shape[0], "mode": "cross-join"}}


def _run_faiss_multihash_cross(A, k):
    Ap, Bp = _split_in_half(A)
    elapsed, pairs = _time(_faiss_multihash_cross, Ap, Bp, k)
    return {"time": elapsed, "pairs": pairs,
            "extra": {"n_a": Ap.shape[0], "n_b": Bp.shape[0], "mode": "cross-join", "nflip": 0}}


def _run_usearch_cross(A, k):
    Ap, Bp = _split_in_half(A)
    elapsed, pairs = _time(_usearch_cross, Ap, Bp, k)
    return {"time": elapsed, "pairs": pairs,
            "extra": {"n_a": Ap.shape[0], "n_b": Bp.shape[0], "mode": "cross-join"}}


_register("pigeon — find_pairs_cross (byte)",               "cross-join", _HAS_PIGEON, _run_pigeon_cross_byte)
_register("pigeon — find_pairs_cross_bit (bit)",            "cross-join", _HAS_PIGEON, _run_pigeon_cross_bit)
_register("FAISS IndexBinaryFlat (cross)",                  "cross-join", _HAS_FAISS,  _run_faiss_flat_cross)
_register("FAISS IndexBinaryMultiHash (cross, nflip=0)",    "cross-join", _HAS_FAISS,  _run_faiss_multihash_cross)
_register("usearch HNSW (cross, approximate)",              "cross-join", _HAS_USEARCH, _run_usearch_cross)


def list_methods(only_available: bool = True):
    return [m for m in METHODS if (m["available"] or not only_available)]


def compute_recall(method_pairs: set, ground_truth: set) -> Optional[float]:
    """Recall = |found ∩ truth| / |truth|. Returns None if truth is empty."""
    if not ground_truth:
        return None
    return len(method_pairs & ground_truth) / len(ground_truth)
