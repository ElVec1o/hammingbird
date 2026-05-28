"""pigeon — fast exact pair search for binary vectors under Hamming distance.

Public API:
    find_pairs_within(A, k, B=None) -> list[(i, j, dist)]
        Find all pairs (i, j) where Hamming(A[i], A[j or B[j]]) <= k.
        If B is None, searches within A (self-join, returns i < j).
        Otherwise cross-join between A and B.

Uses the pigeonhole principle for prefiltering: any two codes within
Hamming distance k must share at least one of (k+1) chunks exactly.
Hash-bucket on each chunk, then verify candidates with popcount.

Best at small k (1-5) on binary codes of 64-512 bits. Beyond k≈8 the
candidate set explodes and naive all-pairs becomes competitive.
"""
from .core import find_pairs_within, hamming_distance, __version__

__all__ = ["find_pairs_within", "hamming_distance", "__version__"]
