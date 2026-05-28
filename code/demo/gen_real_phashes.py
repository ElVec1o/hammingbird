"""Generate 64-bit pHashes from a real image dataset.

Memory-safe: streams the dataset, does not load all images into RAM, caps n.
Writes a flat binary file (n * 8 bytes, uint8 packed MSB-first per row).

Default: CIFAR-10 train split — 50,000 32x32 images, ~170 MB cached.
"""
from __future__ import annotations

import os
import resource
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
import imagehash

HERE = Path(__file__).parent
OUT_DIR = HERE / "_real_data"
OUT_DIR.mkdir(exist_ok=True)


def rss_mb() -> float:
    # Darwin returns bytes; Linux returns KB. We're on Darwin per env.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024


def phash64_to_bytes(ph: imagehash.ImageHash) -> bytes:
    # imagehash.phash returns 8x8 bool array. Pack MSB-first to 8 bytes.
    bits = ph.hash.flatten()  # length 64, bool
    arr = np.packbits(bits.astype(np.uint8))  # 8 bytes, MSB-first by default
    return bytes(arr)


def main(n_cap: int = 50_000, dataset_name: str = "cifar10") -> None:
    from datasets import load_dataset  # imported lazily

    t0 = time.perf_counter()
    print(f"[gen] loading dataset={dataset_name} cap={n_cap}", flush=True)
    if dataset_name == "cifar10":
        ds = load_dataset("uoft-cs/cifar10", split="train", streaming=True)
        img_key = "img"
    elif dataset_name == "imagenette":
        ds = load_dataset("frgfm/imagenette", "320px", split="train", streaming=True)
        img_key = "image"
    else:
        raise ValueError(f"unknown dataset {dataset_name}")

    n = 0
    out = np.zeros((n_cap, 8), dtype=np.uint8)
    t_phash = 0.0
    for sample in ds:
        if n >= n_cap:
            break
        img = sample[img_key]
        if not isinstance(img, Image.Image):
            img = Image.fromarray(np.asarray(img))
        # Force RGB; pHash needs 8-bit luminance.
        if img.mode != "RGB":
            img = img.convert("RGB")
        t1 = time.perf_counter()
        ph = imagehash.phash(img, hash_size=8)  # 64-bit
        t_phash += time.perf_counter() - t1
        b = phash64_to_bytes(ph)
        out[n] = np.frombuffer(b, dtype=np.uint8)
        n += 1
        if n % 5000 == 0:
            elapsed = time.perf_counter() - t0
            print(
                f"[gen]   n={n} elapsed={elapsed:.1f}s phash_total={t_phash:.1f}s "
                f"rss={rss_mb():.0f} MB",
                flush=True,
            )

    out = out[:n]
    fname = OUT_DIR / f"phashes_{dataset_name}_{n}.bin"
    out.tofile(fname)
    elapsed = time.perf_counter() - t0
    print(
        f"[gen] done: n={n} saved={fname} size={fname.stat().st_size/1e6:.2f} MB "
        f"wall={elapsed:.1f}s phash={t_phash:.1f}s rss={rss_mb():.0f} MB",
        flush=True,
    )


if __name__ == "__main__":
    n_cap = int(sys.argv[1]) if len(sys.argv) > 1 else 50_000
    ds = sys.argv[2] if len(sys.argv) > 2 else "cifar10"
    main(n_cap=n_cap, dataset_name=ds)
