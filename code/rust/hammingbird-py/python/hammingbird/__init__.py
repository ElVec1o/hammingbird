"""Rust-backed pigeon — same algorithm, ~5-6× faster than pure Python.

Usage:
    import numpy as np
    from hammingbird import find_pairs_self, find_pairs_cross
    A = np.random.default_rng(0).integers(0, 256, size=(10000, 32), dtype=np.uint8)
    pairs = find_pairs_self(A, k=2)              # list of (i, j, dist)
    B = np.random.default_rng(1).integers(0, 256, size=(5000, 32), dtype=np.uint8)
    cross = find_pairs_cross(A, B, k=2)          # list of (a_id, b_id, dist)
"""
from ._hammingbird import (  # type: ignore
    find_pairs_self,
    find_pairs_self_bit,
    find_pairs_self_adaptive,
    find_pairs_cross,
    find_pairs_cross_bit,
    Index,
    __version__,
)

__all__ = [
    "find_pairs_self",
    "find_pairs_self_bit",
    "find_pairs_self_adaptive",
    "find_pairs_cross",
    "find_pairs_cross_bit",
    "Index",
    "__version__",
]
