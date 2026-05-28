"""Core pigeonhole prefilter implementation."""
from __future__ import annotations
import numpy as np
from collections import defaultdict
from typing import Optional

__version__ = "0.1.0"

# Precomputed popcount table for uint8.
_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def hamming_distance(a: np.ndarray, b: np.ndarray) -> int:
    """Hamming distance between two 1D uint8 arrays of equal length."""
    xor = np.bitwise_xor(a, b)
    return int(_POPCOUNT[xor].sum())


def _chunk_boundaries(n_bytes: int, n_chunks: int) -> list[tuple[int, int]]:
    """Split n_bytes evenly into n_chunks contiguous byte ranges."""
    edges = np.linspace(0, n_bytes, n_chunks + 1).astype(int)
    return [(int(edges[i]), int(edges[i + 1])) for i in range(n_chunks)
            if int(edges[i + 1]) > int(edges[i])]


def find_pairs_within(
    A: np.ndarray,
    k: int,
    B: Optional[np.ndarray] = None,
    return_distances: bool = True,
) -> list[tuple]:
    """Find all pairs of binary vectors within Hamming distance k.

    Parameters
    ----------
    A : (n, d_bytes) uint8 ndarray
        Each row is one binary code, packed into bytes (MSB-first within byte).
    k : int
        Maximum Hamming distance (inclusive).
    B : (m, d_bytes) uint8 ndarray, optional
        If given, return cross-join pairs (i, j) with A[i] vs B[j].
        If None, return self-join pairs (i, j) with i < j.
    return_distances : bool
        If True, return (i, j, dist) tuples; if False, return (i, j).

    Returns
    -------
    list of tuples
        Pairs satisfying the distance constraint.

    Notes
    -----
    Performance:
    - Speedup over O(n^2) grows with n.
    - Sweet spot: k <= 5 on 64-512 bit codes.
    - At k >= ~16 the candidate set is so large that this is slower than naive.
    """
    if A.dtype != np.uint8:
        raise TypeError(f"A must be uint8, got {A.dtype}")
    if A.ndim != 2:
        raise ValueError(f"A must be 2D, got shape {A.shape}")
    if k < 0:
        raise ValueError(f"k must be >= 0, got {k}")

    n_bytes = A.shape[1]
    self_join = B is None
    if not self_join:
        if B.dtype != np.uint8:
            raise TypeError(f"B must be uint8, got {B.dtype}")
        if B.ndim != 2 or B.shape[1] != n_bytes:
            raise ValueError(f"B shape {B.shape} incompatible with A shape {A.shape}")

    # Pigeonhole: with at most k differing bits across the full vector,
    # if we partition the vector into k+1 contiguous chunks, at least
    # one chunk must be bit-identical between the two vectors.
    chunks = _chunk_boundaries(n_bytes, k + 1)

    candidate_set: set[tuple[int, int]] = set()

    if self_join:
        # Self-join: bucket A by each chunk, emit all within-bucket pairs.
        for (lo, hi) in chunks:
            buckets: dict[bytes, list[int]] = defaultdict(list)
            for i in range(A.shape[0]):
                buckets[A[i, lo:hi].tobytes()].append(i)
            for idxs in buckets.values():
                if len(idxs) < 2:
                    continue
                for a in range(len(idxs)):
                    for b in range(a + 1, len(idxs)):
                        i, j = idxs[a], idxs[b]
                        candidate_set.add((i, j) if i < j else (j, i))
    else:
        # Cross-join: bucket A and B separately by each chunk, emit cross pairs.
        for (lo, hi) in chunks:
            buckets_a: dict[bytes, list[int]] = defaultdict(list)
            buckets_b: dict[bytes, list[int]] = defaultdict(list)
            for i in range(A.shape[0]):
                buckets_a[A[i, lo:hi].tobytes()].append(i)
            for j in range(B.shape[0]):
                buckets_b[B[j, lo:hi].tobytes()].append(j)
            for key, ai_list in buckets_a.items():
                bj_list = buckets_b.get(key)
                if not bj_list:
                    continue
                for ai in ai_list:
                    for bj in bj_list:
                        candidate_set.add((ai, bj))

    if not candidate_set:
        return []

    # Verify candidates with full popcount.
    cand = np.array(list(candidate_set), dtype=np.int64)
    if self_join:
        xor = np.bitwise_xor(A[cand[:, 0]], A[cand[:, 1]])
    else:
        xor = np.bitwise_xor(A[cand[:, 0]], B[cand[:, 1]])
    dists = _POPCOUNT[xor].sum(axis=1)
    keep = dists <= k

    if return_distances:
        return [(int(cand[idx, 0]), int(cand[idx, 1]), int(dists[idx]))
                for idx in np.where(keep)[0]]
    return [(int(cand[idx, 0]), int(cand[idx, 1]))
            for idx in np.where(keep)[0]]
