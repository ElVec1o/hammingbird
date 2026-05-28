"""Dataset generators for the pigeon demo."""
from __future__ import annotations
import os
import numpy as np
from typing import Optional

REAL_PHASH_PATH = os.path.join(
    os.path.dirname(__file__), "_real_data",
    "phashes_cifar10_50000.bin"
)


def gen_uniform(n: int, d_bytes: int, seed: int = 0) -> np.ndarray:
    """Uniform random codes — the hardest case for the chunk prefilter."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(n, d_bytes), dtype=np.uint8)


def gen_clustered(
    n: int, d_bytes: int, *,
    n_centroids: int = 10,
    dup_fraction: float = 0.10,
    noise_bits_mean: int = 5,
    seed: int = 1,
) -> np.ndarray:
    """Simulates real-world signature distributions: a sparse population of
    centroids with a fraction of vectors being noisy near-duplicates.

    This pattern shows up in: perceptual hashes (image dups), SimHash on text
    (genuine vs near-dup documents), watermarked fingerprints.
    """
    rng = np.random.default_rng(seed)
    A = rng.integers(0, 256, size=(n, d_bytes), dtype=np.uint8)
    # Reserve the first n_centroids rows as the centroids.
    centroids = A[:n_centroids].copy()
    # Replace dup_fraction of remaining rows with near-duplicates of centroids.
    n_dups = int((n - n_centroids) * dup_fraction)
    if n_dups <= 0:
        return A
    idxs = rng.choice(n - n_centroids, size=n_dups, replace=False) + n_centroids
    for i in idxs:
        c = rng.integers(0, n_centroids)
        A[i] = centroids[c].copy()
        flips = max(1, int(rng.poisson(noise_bits_mean)))
        for _ in range(flips):
            bit = int(rng.integers(0, d_bytes * 8))
            A[i, bit // 8] ^= np.uint8(1 << (bit % 8))
    return A


def gen_low_entropy(n: int, d_bytes: int, *, fraction_constant: float = 0.5,
                    seed: int = 2) -> np.ndarray:
    """Half the bits random, half constant — the adversarial case where
    naive chunkers blow up and adaptive shines."""
    rng = np.random.default_rng(seed)
    A = np.zeros((n, d_bytes), dtype=np.uint8)
    bytes_random = int(d_bytes * (1.0 - fraction_constant))
    if bytes_random > 0:
        A[:, :bytes_random] = rng.integers(0, 256, size=(n, bytes_random), dtype=np.uint8)
    return A


def gen_with_exact_dupes(n: int, d_bytes: int, *,
                         dup_fraction: float = 0.01, seed: int = 3) -> np.ndarray:
    """Uniform random plus a fraction of exact duplicates injected.

    Targets the k=0 exact-dedup use case (the most common in production:
    "find every pair of identical hashes"). Without injection, uniform
    random has essentially no exact duplicates so k=0 is unmeasurable.
    """
    rng = np.random.default_rng(seed)
    A = rng.integers(0, 256, size=(n, d_bytes), dtype=np.uint8)
    n_dupes = int(n * dup_fraction)
    for _ in range(n_dupes):
        src = int(rng.integers(0, n))
        dst = int(rng.integers(0, n))
        if src != dst:
            A[dst] = A[src].copy()
    return A


def load_cifar10_phashes(n: Optional[int] = None) -> Optional[np.ndarray]:
    """Load the pre-computed CIFAR-10 64-bit perceptual hashes.

    Returns None if the file doesn't exist (the user can regenerate it via
    `code/experiments/gen_real_phashes.py`).
    """
    if not os.path.exists(REAL_PHASH_PATH):
        return None
    raw = np.fromfile(REAL_PHASH_PATH, dtype=np.uint8)
    full_n = len(raw) // 8
    A = raw.reshape(full_n, 8)
    if n is not None and n < full_n:
        A = A[:n]
    return np.ascontiguousarray(A)


# Registry consumed by the Streamlit app.
DATASETS = {
    "Synthetic — uniform random": gen_uniform,
    "Synthetic — clustered (real-world shape)": gen_clustered,
    "Synthetic — low-entropy (adversarial)": gen_low_entropy,
    "Synthetic — uniform + 1% exact dupes (for k=0)": gen_with_exact_dupes,
    "Real — CIFAR-10 pHashes (64-bit, fixed)": load_cifar10_phashes,
}


def estimate_memory_mb(n: int, d_bytes: int, k: int, methods: list[str]) -> float:
    """Rough peak-memory estimate for picking ceilings. Conservative."""
    data_mb = (n * d_bytes) / 1024 / 1024
    # Hashmaps: ~50 bytes per entry × (k+1) chunks × n entries.
    pigeon_mb = (k + 1) * n * 50 / 1024 / 1024 if any("pigeon" in m.lower() for m in methods) else 0
    # FAISS Flat index ≈ raw data.
    faiss_flat_mb = data_mb if any("Flat" in m for m in methods) else 0
    # MultiHash ≈ k+1 × hashmap.
    faiss_mh_mb = (k + 1) * n * 40 / 1024 / 1024 if any("MultiHash" in m for m in methods) else 0
    # Candidate sets at large k: estimate up to n²/4096 pairs.
    cand_mb = min(n * n / 4096 * 8, n * 32) / 1024 / 1024
    # Streamlit overhead.
    overhead_mb = 200
    return data_mb + pigeon_mb + faiss_flat_mb + faiss_mh_mb + cand_mb + overhead_mb
