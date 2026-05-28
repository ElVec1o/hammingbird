"""Tests for pigeon — verify correctness against naive all-pairs ground truth.

Run with: python3 -m pigeon.test
"""
import numpy as np
from itertools import combinations
from .core import find_pairs_within, hamming_distance, _POPCOUNT


def naive_self_pairs(A, k):
    """Ground-truth O(n^2) self-join."""
    n = A.shape[0]
    out = []
    for i, j in combinations(range(n), 2):
        d = hamming_distance(A[i], A[j])
        if d <= k:
            out.append((i, j, d))
    return sorted(out)


def naive_cross_pairs(A, B, k):
    """Ground-truth O(n*m) cross-join."""
    out = []
    for i in range(A.shape[0]):
        for j in range(B.shape[0]):
            d = hamming_distance(A[i], B[j])
            if d <= k:
                out.append((i, j, d))
    return sorted(out)


def _gen(n, d_bytes, seed):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(n, d_bytes), dtype=np.uint8)


def test_self_join_random():
    for seed in range(5):
        for k in [0, 1, 2, 3, 5]:
            A = _gen(50, 8, seed)  # 64-bit codes
            got = sorted(find_pairs_within(A, k))
            want = naive_self_pairs(A, k)
            assert got == want, f"seed={seed} k={k}: got {len(got)} want {len(want)}"
    print("  ✓ self-join random matches naive")


def test_self_join_with_dupes():
    """Inject near-duplicates and verify they're all found."""
    rng = np.random.default_rng(42)
    A = rng.integers(0, 256, size=(100, 16), dtype=np.uint8)  # 128-bit
    # Make rows 0 and 1 identical; rows 2 and 3 differ by 1 bit
    A[1] = A[0]
    A[3] = A[2].copy()
    A[3, 0] ^= 1
    # rows 4 and 5 differ by 2 bits
    A[5] = A[4].copy()
    A[5, 0] ^= 1
    A[5, 8] ^= 1

    pairs = find_pairs_within(A, k=2)
    pair_set = {(i, j) for i, j, _ in pairs}
    assert (0, 1) in pair_set, "missed identical pair"
    assert (2, 3) in pair_set, "missed 1-bit pair"
    assert (4, 5) in pair_set, "missed 2-bit pair"
    print("  ✓ injected near-duplicates all found")


def test_cross_join():
    for seed in range(3):
        for k in [0, 2, 5]:
            A = _gen(30, 8, seed)
            B = _gen(40, 8, seed + 100)
            got = sorted(find_pairs_within(A, k, B=B))
            want = naive_cross_pairs(A, B, k)
            assert got == want, f"cross seed={seed} k={k}: got {len(got)} want {len(want)}"
    print("  ✓ cross-join random matches naive")


def test_empty_results():
    A = _gen(20, 32, 0)  # 256-bit random — no pairs expected at k=2
    pairs = find_pairs_within(A, k=2)
    assert pairs == [], f"expected empty, got {pairs}"
    print("  ✓ empty result handled")


def test_input_validation():
    A_bad = np.zeros((5, 8), dtype=np.int32)
    try:
        find_pairs_within(A_bad, k=2)
        assert False, "should have raised TypeError"
    except TypeError:
        pass
    A = _gen(5, 8, 0)
    try:
        find_pairs_within(A, k=-1)
        assert False, "should have raised ValueError"
    except ValueError:
        pass
    print("  ✓ input validation works")


def test_k_zero_finds_exact_dupes_only():
    rng = np.random.default_rng(0)
    A = rng.integers(0, 256, size=(20, 8), dtype=np.uint8)
    A[5] = A[3]  # exact duplicate
    A[10] = A[7].copy()
    A[10, 0] ^= 1  # 1-bit off — should NOT be found at k=0
    pairs = find_pairs_within(A, k=0)
    pair_set = {(i, j) for i, j, _ in pairs}
    assert (3, 5) in pair_set
    assert (7, 10) not in pair_set
    print("  ✓ k=0 finds exact duplicates only")


if __name__ == "__main__":
    print("Running pigeon test suite...")
    test_self_join_random()
    test_self_join_with_dupes()
    test_cross_join()
    test_empty_results()
    test_input_validation()
    test_k_zero_finds_exact_dupes_only()
    print("\nAll tests passed.")
